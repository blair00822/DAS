#!/usr/bin/env python3
"""Protect portraits using frozen clean-response RMS weights (experiment C).

Run from the project root: python protect/protect_hallo_response.py --image portrait.png
Epsilon and alpha are expressed in 0..255 pixel units.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image', nargs='+', required=True, help='Files or directories (non-recursive)')
    p.add_argument('--output', default='protect/response_pgd')
    p.add_argument('--experiment', choices=('C',), default='C')
    p.add_argument('--layers', nargs='+', default=['up:0', 'up:1'])
    p.add_argument('--config', default='protect/configs/response.yaml')
    p.add_argument('--audio', required=True, help='Precomputed audio embedding (.pt)')
    p.add_argument('--audio-start', type=int)
    p.add_argument('--clip-length', type=int, default=3)
    p.add_argument('--min-timestep', type=int, default=200)
    p.add_argument('--max-timestep', type=int)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--epsilon', type=float, default=16.)
    p.add_argument('--alpha', type=float, default=1.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--calibration-samples', type=int, default=4)
    p.add_argument('--baseline-floor', type=float, default=1e-8)
    p.add_argument('--lpips-weight', type=float, default=0.)
    p.add_argument('--dtype', choices=('fp16', 'fp32', 'bf16'))
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    args.max_timestep = args.max_timestep if args.max_timestep is not None else args.min_timestep + 100
    if min(args.steps, args.clip_length, args.calibration_samples) < 1:
        p.error('steps, clip-length and calibration-samples must be positive')
    if not 0 <= args.min_timestep < args.max_timestep:
        p.error('Expected 0 <= min-timestep < max-timestep')
    if not all(math.isfinite(v) for v in (args.epsilon, args.alpha, args.lpips_weight, args.baseline_floor)):
        p.error('Optimization parameters must be finite')
    if not 0 <= args.epsilon <= 255 or args.alpha <= 0 or args.lpips_weight < 0 or args.baseline_floor <= 0:
        p.error('Invalid optimization parameters')
    if args.audio_start is not None and args.audio_start < 0:
        p.error('audio-start must be nonnegative')
    # Paths are relative to the project root; retain this form in saved metadata.
    for value in (*args.image, args.output, args.config, args.audio):
        if Path(value).is_absolute() or '..' in Path(value).parts:
            p.error('Use project-relative paths without parent traversal')
    images = []
    for value in args.image:
        path = Path(value)
        if path.is_dir():
            images.extend(sorted(x for x in path.iterdir() if x.is_file() and x.suffix.lower() in ('.png', '.jpg', '.jpeg')))
        elif path.is_file():
            images.append(path)
        else:
            p.error(f'Image not found: {value}')
    args.image = [x.as_posix() for x in dict.fromkeys(images)]
    if not args.image or len({Path(x).stem for x in args.image}) != len(args.image):
        p.error('Need images with unique filename stems')
    for value in (args.config, args.audio):
        if not Path(value).is_file():
            p.error(f'File not found: {value}')
    return args


def sample_condition(runtime, shape, args, generator):
    import torch
    noise = torch.randn(shape, generator=generator).to(runtime.device, runtime.dtype)
    timestep = torch.randint(args.min_timestep, args.max_timestep, (1,), generator=generator).to(runtime.device)
    return noise, timestep


def calibrate(runtime, controller, source, audio, shape, args):
    import torch
    from protect.response_attack_loss import response_energy_maps, make_spatial_weights, clean_baselines
    generator = torch.Generator().manual_seed(args.seed + 10000)
    mean_energy, mean_rms, conditions = {}, {}, []
    with torch.no_grad():
        for _ in range(args.calibration_samples):
            noise, timestep = sample_condition(runtime, shape, args, generator)
            energies = response_energy_maps(*runtime.pair(source['image'], source, audio, noise, timestep, controller))
            for layer, energy in energies.items():
                mean_energy[layer] = mean_energy.get(layer, 0) + energy / args.calibration_samples
                mean_rms[layer] = mean_rms.get(layer, 0) + energy.sqrt() / args.calibration_samples
            conditions.append(int(timestep.item()))
            controller.reset()
    weights = make_spatial_weights(mean_energy, mean_rms)
    baselines = clean_baselines(mean_energy, weights, args.baseline_floor)
    return weights, baselines, conditions


def attack(runtime, controller, source, audio, args, lpips_fn=None):
    import torch
    from tqdm import tqdm
    from protect.response_attack_loss import response_energy_maps, compute_response_loss
    clean = source['image'].detach().to(runtime.device, torch.float32)
    shape = runtime.latent_shape(clean, args.clip_length)
    weights, baselines, calibration_ts = calibrate(runtime, controller, source, audio, shape, args)
    # Independent random streams for calibration and PGD.
    generator = torch.Generator().manual_seed(args.seed + 20000)
    eps, alpha = args.epsilon * 2 / 255, args.alpha * 2 / 255
    adv = clean.clone()
    history = []
    pbar = tqdm(range(args.steps), desc='PGD-response-C')
    for step in pbar:
        adv = adv.detach().requires_grad_(True)
        noise, timestep = sample_condition(runtime, shape, args, generator)
        energies = response_energy_maps(*runtime.pair(adv, source, audio, noise, timestep, controller))
        response_loss, per_layer = compute_response_loss(energies, weights, baselines)
        quality_loss = lpips_fn(adv[None], clean[None]).mean() if lpips_fn is not None else adv.new_zeros(())
        total_loss = response_loss + args.lpips_weight * quality_loss
        grad = torch.autograd.grad(total_loss, adv, retain_graph=False, create_graph=False)[0]
        if not torch.isfinite(grad).all() or not torch.isfinite(total_loss):
            raise FloatingPointError('Non-finite PGD loss/gradient; try --dtype fp32')
        grad_mean = float(grad.abs().mean())
        if grad_mean == 0:
            raise RuntimeError('Zero image gradient: check selected layers and hook gradient flow')
        entry = dict(step=step, timestep=int(timestep.item()), objective='energy', response=float(response_loss.detach()),
                     lpips=float(quality_loss.detach()), total=float(total_loss.detach()),
                     gradient_mean_abs=grad_mean,
                     layers={l: float(v.detach()) for l, v in per_layer.items()})
        # Same minimization sign update, Linf projection, and [-1,1] clamp as v2.
        with torch.no_grad():
            adv = adv - alpha * grad.sign()
            adv = torch.maximum(torch.minimum(adv, clean + eps), clean - eps).clamp(-1, 1)
        controller.reset()
        history.append(entry)
        pbar.set_postfix(response=f'{entry["response"]:.4f}', lpips=f'{entry["lpips"]:.4f}', t=entry['timestep'])
        del energies, per_layer, response_loss, quality_loss, total_loss, grad

    # Reuse calibration conditions to compare clean and final responses fairly.
    # This is an in-sample proxy check, NOT a generated-video attack evaluation.
    eval_generator = torch.Generator().manual_seed(args.seed + 10000)
    final_response = 0.
    with torch.no_grad():
        for _ in range(args.calibration_samples):
            noise, timestep = sample_condition(runtime, shape, args, eval_generator)
            energies = response_energy_maps(*runtime.pair(adv, source, audio, noise, timestep, controller))
            loss, _ = compute_response_loss(energies, weights, baselines)
            final_response += float(loss) / args.calibration_samples
            controller.reset()
        final_lpips = float(lpips_fn(adv[None], clean[None]).mean()) if lpips_fn is not None else None
    delta = (adv - clean) / 2
    linf = float(delta.abs().max())
    if linf > args.epsilon / 255 + 1e-6:
        raise AssertionError('PGD projection exceeded epsilon')
    metrics = dict(calibration_timesteps=calibration_ts,
                   layer_baselines={l: float(d) for l, d in baselines.items()},
                   clean_calibration_response=1., final_calibration_response=final_response,
                   final_lpips=final_lpips, linf_01=linf,
                   mae_01=float(delta.abs().mean()), rmse_01=float(delta.square().mean().sqrt()),
                   history=history)
    metrics['response_objective'] = 'energy'
    return adv.detach().cpu(), metrics



def main():
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)
    args = parse_args()
    os.environ.setdefault('TORCH_HOME', str(ROOT))
    import numpy as np
    import torch
    from PIL import Image
    from hallo.datasets.image_processor import ImageProcessor
    from protect.hallo_response_runtime import HalloResponseRuntime, process_audio_emb
    from protect.response_attack_controller import ResponseAttackController
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; select GPU with CUDA_VISIBLE_DEVICES')
    pending = []
    for path in args.image:
        out = Path(args.output) / args.experiment / Path(path).stem
        if out.is_dir() and not args.overwrite:
            print(f'Skipping {path}: output directory already exists: {out}', flush=True)
            continue
        pending.append((path, out))
    if not pending:
        print('No images to process: all output directories already exist.', flush=True)
        return
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    runtime = HalloResponseRuntime(args.config, args.dtype)
    if args.max_timestep > runtime.scheduler.config.num_train_timesteps:
        raise ValueError('Timestep range exceeds scheduler')
    selectors = args.layers
    controller = ResponseAttackController(runtime.net.denoising_unet, selectors)
    print('Selected layers:', list(controller.layers), flush=True)
    audio_all = process_audio_emb(torch.load(args.audio, map_location='cpu'))
    last = len(audio_all) - args.clip_length
    start = args.audio_start
    if start is None:
        if last < runtime.cfg.data.n_motion_frames:
            raise ValueError('Audio too short for default random clip selection')
        start = random.Random(args.seed).randint(runtime.cfg.data.n_motion_frames, last)
    if start > last:
        raise ValueError('Requested audio clip exceeds embeddings')
    audio = audio_all[start:start + args.clip_length][None].to(runtime.device, runtime.dtype)
    lpips_fn = None
    if args.lpips_weight:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg').to(runtime.device).eval().requires_grad_(False)
    try:
        with ImageProcessor((512, 512), runtime.cfg.face_analysis_model_path) as processor:
            for path, out in pending:
                print(f'Processing {path}; audio start={start}', flush=True)
                with tempfile.TemporaryDirectory(prefix='hallo_pgd_') as cache:
                    values = processor.preprocess(path, cache, runtime.cfg.face_expand_ratio)
                    source = dict(zip(('image', 'face_region', 'face_emb', 'full_mask', 'face_mask', 'lip_mask'), values))
                    source['face_emb'] = torch.as_tensor(source['face_emb']).detach()
                    adv, metrics = attack(runtime, controller, source, audio, args, lpips_fn)
                    # Round and clamp against the clean image in integer pixel space
                    # so saved quantization cannot silently expand the pixel budget.
                    clean_u8 = ((source['image'] + 1) * 127.5).round().clamp(0, 255)
                    adv_u8 = ((adv + 1) * 127.5).round().clamp(0, 255)
                    integer_eps = int(args.epsilon)
                    adv_u8 = torch.maximum(torch.minimum(adv_u8, clean_u8 + integer_eps), clean_u8 - integer_eps)
                    metrics['saved_linf_255'] = float((adv_u8 - clean_u8).abs().max())
                    metrics.update(arguments=vars(args), image=path, audio_start=start,
                                   layer_paths={l: p for l, (p, _) in controller.layers.items()},
                                   note='Final proxy and LPIPS are pre-quantization; proxy uses calibration conditions, not generated video.')
                    out.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(adv_u8.permute(1, 2, 0).byte().numpy()).save(out / f'{out.name}.png')
                    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding='utf-8')
                    print(f'Saved {out}; response={metrics["final_calibration_response"]:.4f}; Linf={metrics["saved_linf_255"]}/255', flush=True)
    finally:
        controller.close()
        runtime.net.clear()


if __name__ == '__main__':
    main()
