"""Minimal Hallo loader and single-timestep paired forward for response PGD.

Loads the frozen model and constructs matched audio/silent conditions.
"""
import torch
from torch import nn
from omegaconf import OmegaConf
from diffusers import AutoencoderKL, DDIMScheduler
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel
from hallo.models.image_proj import ImageProjModel
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.mutual_self_attention import ReferenceAttentionControl


def process_audio_emb(audio):
    offsets = torch.arange(-2, 3, device=audio.device)
    indices = (torch.arange(len(audio), device=audio.device)[:, None] + offsets).clamp(0, len(audio) - 1)
    return audio[indices]


class ResponseNet(nn.Module):
    def __init__(self, reference_unet, denoising_unet, face_locator, imageproj, audioproj):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj
        self.writer = ReferenceAttentionControl(reference_unet, mode='write', fusion_blocks='full',
                                                do_classifier_free_guidance=False)
        self.reader = ReferenceAttentionControl(denoising_unet, mode='read', fusion_blocks='full',
                                                do_classifier_free_guidance=False)

    def clear(self):
        self.writer.clear()
        self.reader.clear()

    def forward(self, noisy, timestep, reference, face_emb, audio, region, masks, silent=False):
        face_emb = self.imageproj(face_emb)
        mask_features = self.face_locator(region)
        audio = self.audioproj(audio)
        # Match original uncond_audio_fwd: zero AFTER audio projection.
        if silent:
            audio = torch.zeros_like(audio)
        ref_t = torch.zeros_like(timestep).repeat(reference.shape[0] // timestep.shape[0])
        self.reference_unet(reference, ref_t, encoder_hidden_states=face_emb, return_dict=False)
        self.reader.update(self.writer)
        return self.denoising_unet(
            noisy, timestep, mask_cond_fea=mask_features, encoder_hidden_states=face_emb,
            audio_embedding=audio, full_mask=masks['full_mask'],
            face_mask=masks['face_mask'], lip_mask=masks['lip_mask']).sample


class HalloResponseRuntime:
    def __init__(self, config, dtype=None):
        self.cfg = cfg = OmegaConf.load(config)
        self.device = torch.device('cuda')
        self.dtype = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}[dtype or cfg.weight_dtype]
        self.vae = AutoencoderKL.from_pretrained(cfg.vae_model_path).to(self.device, self.dtype).eval().requires_grad_(False)
        reference = UNet2DConditionModel.from_pretrained(cfg.base_model_path, subfolder='unet')
        denoiser = UNet3DConditionModel.from_pretrained_2d(
            cfg.base_model_path, cfg.motion_module_path, subfolder='unet',
            unet_additional_kwargs=OmegaConf.to_container(cfg.unet_additional_kwargs), use_landmark=False)
        self.net = ResponseNet(
            reference, denoiser, FaceLocator(conditioning_embedding_channels=320),
            ImageProjModel(cross_attention_dim=denoiser.config.cross_attention_dim,
                           clip_embeddings_dim=512, clip_extra_context_tokens=4),
            AudioProjModel(seq_len=5, blocks=12, channels=768, intermediate_dim=512,
                           output_dim=768, context_tokens=32)).to(self.device, self.dtype)
        from pathlib import Path
        state = torch.load(Path(cfg.audio_ckpt_dir) / 'net.pth', map_location='cpu')
        self.net.load_state_dict(state, strict=True)
        del state
        self.net.eval().requires_grad_(False)
        if cfg.solver.enable_xformers_memory_efficient_attention:
            self.net.reference_unet.enable_xformers_memory_efficient_attention()
            self.net.denoising_unet.enable_xformers_memory_efficient_attention()
        # eval mode, as in the original attack; no checkpoint replay in hooks.
        kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
        if cfg.enable_zero_snr:
            kwargs.update(rescale_betas_zero_snr=True, timestep_spacing='trailing', prediction_type='v_prediction')
        kwargs['beta_schedule'] = 'scaled_linear'
        self.scheduler = DDIMScheduler(**kwargs)

    def latent_shape(self, image, frames):
        with torch.no_grad():
            z = self.vae.encode(image[None].to(self.device, self.dtype)).latent_dist.mean
        return (1, z.shape[1], frames, z.shape[2], z.shape[3])

    def pair(self, image, source, audio, noise, timestep, controller):
        """One differentiable audio forward and one no-grad silent forward.

        image: [3,H,W] in [-1,1]. Noise/timestep shared between branches.
        The caller owns/reset features; reference banks clear on every branch.
        """
        controller.reset()
        frames = audio.shape[1]
        x = image[None].to(self.device, self.dtype)
        z = self.vae.encode(x.repeat(frames, 1, 1, 1)).latent_dist.mean
        z = z.permute(1, 0, 2, 3)[None] * 0.18215
        noisy = self.scheduler.add_noise(z, noise.to(z), timestep)
        motion = x.new_zeros((self.cfg.data.n_motion_frames, *x.shape[1:]))
        reference = self.vae.encode(torch.cat((x, motion), dim=0)).latent_dist.mean * 0.18215
        face_emb = torch.as_tensor(source['face_emb']).reshape(1, -1).to(self.device, self.dtype)
        region = source['face_region']
        if region.ndim == 3:
            region = region[None]
        region = region.to(self.device, self.dtype).unsqueeze(2).repeat(1, 1, frames, 1, 1)
        masks = {k: [m.repeat(frames, 1).to(self.device, self.dtype) for m in source[k]]
                 for k in ('full_mask', 'face_mask', 'lip_mask')}
        audio = audio.to(self.device, self.dtype)
        try:
            self.net.clear()
            controller.begin('audio')
            self.net(noisy, timestep, reference, face_emb, audio, region, masks)
            self.net.clear()
            controller.begin('silent')
            with torch.no_grad():
                self.net(noisy.detach(), timestep, reference.detach(), face_emb,
                         torch.zeros_like(audio), region, masks, silent=True)
            return controller.pair()
        finally:
            self.net.clear()
