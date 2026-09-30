# pylint: disable=E1120
"""
This module contains the implementation of mutual self-attention, 
which is a type of attention mechanism used in deep learning models. 
The module includes several classes and functions related to attention mechanisms, 
such as BasicTransformerBlock and TemporalBasicTransformerBlock. 
The main purpose of this module is to provide a comprehensive attention mechanism for various tasks in deep learning, 
such as image and video processing, natural language processing, and so on.
"""

from typing import Any, Dict, Optional

import torch
from einops import rearrange

from .attention import BasicTransformerBlock, TemporalBasicTransformerBlock


def torch_dfs(model: torch.nn.Module):
    """
    Perform a depth-first search (DFS) traversal on a PyTorch model's neural network architecture.

    This function recursively traverses all the children modules of a given PyTorch model and returns a list
    containing all the modules in the model's architecture. The DFS approach starts with the input model and
    explores its children modules depth-wise before backtracking and exploring other branches.

    Args:
        model (torch.nn.Module): The root module of the neural network to traverse.

    Returns:
        list: A list of all the modules in the model's architecture.
    """
    result = [model]
    for child in model.children():
        result += torch_dfs(child)
    return result

## 在不改 UNet 结构的前提下，通过‘hook’的方式替换 UNet 里 TransformerBlock 的 forward，
# 只改动了 self-attn（attn1）的 K/V 来源（引入 bank）实现一种 reference attention：
## 它在初始化时就把 UNet 里一批 TransformerBlock 的 forward 整个替换掉了；
# update(writer) 只是把 writer 在“write”模式里缓存的 bank 拷贝到 reader 在“read”模式里对应 block 的 bank，
# 从而让 denoising_unet 的自注意力在 K/V 里额外看到 reference 特征。
class ReferenceAttentionControl:
    """
    This class is used to control the reference attention mechanism in a neural network model.
    It is responsible for managing the guidance and fusion blocks, and modifying the self-attention
    and group normalization mechanisms. The class also provides methods for registering reference hooks
    and updating/clearing the internal state of the attention control object.

    Attributes:
        unet: The UNet model associated with this attention control object.
        mode: The operating mode of the attention control object, either 'write' or 'read'.
        do_classifier_free_guidance: Whether to use classifier-free guidance in the attention mechanism.
        attention_auto_machine_weight: The weight assigned to the attention auto-machine.
        gn_auto_machine_weight: The weight assigned to the group normalization auto-machine.
        style_fidelity: The style fidelity parameter for the attention mechanism.
        reference_attn: Whether to use reference attention in the model.
        reference_adain: Whether to use reference AdaIN in the model.
        fusion_blocks: The type of fusion blocks to use in the model ('midup', 'late', or 'nofusion').
        batch_size: The batch size used for processing video frames.

    Methods:
        register_reference_hooks: Registers the reference hooks for the attention control object.
        hacked_basic_transformer_inner_forward: The modified inner forward method for the basic transformer block.  ##
        update: Updates the internal state of the attention control object using the provided writer and dtype.
        clear: Clears the internal state of the attention control object.
    """
    def __init__(
        self,
        unet,
        mode="write",
        do_classifier_free_guidance=False,
        attention_auto_machine_weight=float("inf"),
        gn_auto_machine_weight=1.0,
        style_fidelity=1.0,
        reference_attn=True,
        reference_adain=False,
        fusion_blocks="midup",
        batch_size=1,
    ) -> None:
        """
       Initializes the ReferenceAttentionControl class.

       Args:
           unet (torch.nn.Module): The UNet model.
           mode (str, optional): The mode of operation. Defaults to "write".
           do_classifier_free_guidance (bool, optional): Whether to do classifier-free guidance. Defaults to False.
           attention_auto_machine_weight (float, optional): The weight for attention auto-machine. Defaults to infinity.
           gn_auto_machine_weight (float, optional): The weight for group-norm auto-machine. Defaults to 1.0.
           style_fidelity (float, optional): The style fidelity. Defaults to 1.0.
           reference_attn (bool, optional): Whether to use reference attention. Defaults to True.
           reference_adain (bool, optional): Whether to use reference AdaIN. Defaults to False.
           fusion_blocks (str, optional): The fusion blocks to use. Defaults to "midup".
           batch_size (int, optional): The batch size. Defaults to 1.

       Raises:
           ValueError: If the mode is not recognized.
           ValueError: If the fusion blocks are not recognized.
       """
        # 10. Modify self attention and group norm
        self.unet = unet
        assert mode in ["read", "write"]
        assert fusion_blocks in ["midup", "full"]
        self.reference_attn = reference_attn
        self.reference_adain = reference_adain
        self.fusion_blocks = fusion_blocks
        self.register_reference_hooks(
            mode,
            do_classifier_free_guidance,
            attention_auto_machine_weight,
            gn_auto_machine_weight,
            style_fidelity,
            reference_attn,
            reference_adain,
            _fusion_blocks=fusion_blocks,
            batch_size=batch_size,
        )

    def register_reference_hooks(
        self,
        mode,
        do_classifier_free_guidance,
        _attention_auto_machine_weight,
        _gn_auto_machine_weight,
        _style_fidelity,
        _reference_attn,
        _reference_adain,
        _dtype=torch.float16,
        batch_size=1,
        num_images_per_prompt=1,
        device=torch.device("cpu"),
        _fusion_blocks="midup",
    ):
        """
        Registers reference hooks for the model.

        This function is responsible for registering reference hooks in the model, 
        which are used to modify the attention mechanism and group normalization layers.
        It takes various parameters as input, such as mode, 
        do_classifier_free_guidance, _attention_auto_machine_weight, _gn_auto_machine_weight, _style_fidelity,
        _reference_attn, _reference_adain, _dtype, batch_size, num_images_per_prompt, device, and _fusion_blocks.

        Args:
            self: Reference to the instance of the class.
            mode: The mode of operation for the reference hooks.
            do_classifier_free_guidance: A boolean flag indicating whether to use classifier-free guidance.
            _attention_auto_machine_weight: The weight for the attention auto-machine.
            _gn_auto_machine_weight: The weight for the group normalization auto-machine.
            _style_fidelity: The style fidelity for the reference hooks.
            _reference_attn: A boolean flag indicating whether to use reference attention.
            _reference_adain: A boolean flag indicating whether to use reference AdaIN.
            _dtype: The data type for the reference hooks.
            batch_size: The batch size for the reference hooks.
            num_images_per_prompt: The number of images per prompt for the reference hooks.
            device: The device for the reference hooks.
            _fusion_blocks: The fusion blocks for the reference hooks.

        Returns:
            None
        """
        MODE = mode
        if do_classifier_free_guidance:
            uc_mask = (
                torch.Tensor(
                    [1] * batch_size * num_images_per_prompt * 16
                    + [0] * batch_size * num_images_per_prompt * 16
                )
                .to(device)
                .bool()
            )   ## 指明哪些样本是 unconditional
        else:
            uc_mask = (
                torch.Tensor([0] * batch_size * num_images_per_prompt * 2)
                .to(device)
                .bool()
            )
            
        ## 像 原本的 transformer block forward，但在 self-attention 部分注入了 reference bank。
        def hacked_basic_transformer_inner_forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            video_length=None,
        ):
            gate_msa = None
            shift_mlp = None
            scale_mlp = None
            gate_mlp = None
            ## 标准 diffusers transformer block 的写法
            if self.use_ada_layer_norm:  # False
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                (
                    norm_hidden_states,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                ) = self.norm1(
                    hidden_states,
                    timestep,
                    class_labels,
                    hidden_dtype=hidden_states.dtype,
                )
            else:
                norm_hidden_states = self.norm1(hidden_states) ## [B*T, L, C]

            # 1. Self-Attention
            # self.only_cross_attention = False
            cross_attention_kwargs = (
                cross_attention_kwargs if cross_attention_kwargs is not None else {}
            )
            if self.only_cross_attention:   ## 直接走 cross_attention 和reference无关
                attn_output = self.attn1(
                    norm_hidden_states,
                    encoder_hidden_states=(
                        encoder_hidden_states if self.only_cross_attention else None
                    ),
                    attention_mask=attention_mask,
                    **cross_attention_kwargs,
                )
            else:   ## 
                if MODE == "write":
                    self.bank.append(norm_hidden_states.clone())    ## 会把当前 step 的 norm_hidden_states 存到 self.bank
                    attn_output = self.attn1(
                        norm_hidden_states,
                        encoder_hidden_states=(
                            encoder_hidden_states if self.only_cross_attention else None
                        ),
                        attention_mask=attention_mask,
                        **cross_attention_kwargs,
                    )
                if MODE == "read":
                    bank_fea = [
                        rearrange(
                            rearrange(
                                d,
                                "(b s) l c -> b s l c",
                                b=norm_hidden_states.shape[0] // video_length,  ## [B, T, L, C] 以及 b=2？-> s=3
                            )[:, 0, :, :]   ## 只取 [:, 0, :, :]（只拿第 0 帧当 reference）
                            # .unsqueeze(1)
                            .repeat(1, video_length, 1, 1), ## 再复制成 T 帧：让每一帧都“看到同一个 reference”
                            "b t l c -> (b t) l c",
                        )
                        for d in self.bank
                    ]
                    motion_frames_fea = [rearrange(
                        d,
                        "(b s) l c -> b s l c",
                        b=norm_hidden_states.shape[0] // video_length,
                    )[:, 1:, :, :] for d in self.bank] ## 取 bank 里除第 0 帧之外的 1:，当作 motion 特征缓存（返回给外部）
                    modify_norm_hidden_states = torch.cat(
                        [norm_hidden_states] + bank_fea, dim=1
                    )   
                    # read 模式下 K/V 会变长，attention 会显式看到 reference 特征 → 生成会被 reference 强约束
                    # print('modify_norm_hidden_states', modify_norm_hidden_states.shape) # [32, 8192, 320]
                    # print('norm_hidden_states', norm_hidden_states.shape) # [32, 8192, 320]
                    # print(self.attn1)
                    hidden_states_uc = (
                        self.attn1(
                            norm_hidden_states, ## Q
                            encoder_hidden_states=modify_norm_hidden_states,    ## K,V
                            attention_mask=attention_mask,
                        )  
                        + hidden_states
                    ) ## 用 attn1/self-attn 做“query 对 bank+当前”的 attention 
                    ## 把原来的 K/V = norm_hidden_states（自己）--> K/V = cat(当前, bank_fea)（当前+reference）
                    ## 实现“reference 注入发生在 self-attn”

                    if do_classifier_free_guidance: # False
                        hidden_states_c = hidden_states_uc.clone()
                        _uc_mask = uc_mask.clone()
                        if hidden_states.shape[0] != _uc_mask.shape[0]:
                            _uc_mask = (
                                torch.Tensor(
                                    [1] * (hidden_states.shape[0] // 2)
                                    + [0] * (hidden_states.shape[0] // 2)
                                )
                                .to(device)
                                .bool()
                            )
                        hidden_states_c[_uc_mask] = (
                            self.attn1(
                                norm_hidden_states[_uc_mask],
                                encoder_hidden_states=norm_hidden_states[_uc_mask],
                                attention_mask=attention_mask,
                            )
                            + hidden_states[_uc_mask]
                        )
                        hidden_states = hidden_states_c.clone()
                    else:
                        hidden_states = hidden_states_uc

                    # self.bank.clear()
                    if self.attn2 is not None:  ## 在 MODE=="read" 分支里，直接做完 attn2、ff、temporal-attn 并 return
                        # Cross-Attention
                        norm_hidden_states = (
                            self.norm2(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm2(hidden_states)
                        )
                        hidden_states = (
                            self.attn2(
                                norm_hidden_states,
                                encoder_hidden_states=encoder_hidden_states,
                                attention_mask=attention_mask,
                            )
                            + hidden_states
                        )
                    # print(encoder_hidden_states.shape) # [1 4 1768]

                    # Feed-forward
                    hidden_states = self.ff(self.norm3(
                        hidden_states)) + hidden_states

                    # Temporal-Attention
                    if self.unet_use_temporal_attention:
                        d = hidden_states.shape[1]
                        hidden_states = rearrange(
                            hidden_states, "(b f) d c -> (b d) f c", f=video_length
                        )
                        norm_hidden_states = (
                            self.norm_temp(hidden_states, timestep)
                            if self.use_ada_layer_norm
                            else self.norm_temp(hidden_states)
                        )
                        hidden_states = (
                            self.attn_temp(norm_hidden_states) + hidden_states
                        )
                        hidden_states = rearrange(
                            hidden_states, "(b d) f c -> (b f) d c", d=d
                        )
                    # print(hidden_states.shape) # 1 4096 320
                    return hidden_states, motion_frames_fea ##

            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep)
                    if self.use_ada_layer_norm
                    else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                tmp = norm_hidden_states.shape[0] // encoder_hidden_states.shape[0]
                attn_output = self.attn2(
                    norm_hidden_states,
                    # TODO: repeat这个地方需要斟酌一下
                    encoder_hidden_states=encoder_hidden_states.repeat(
                        tmp, 1, 1),
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = (
                    norm_hidden_states *
                    (1 + scale_mlp[:, None]) + shift_mlp[:, None]
                )

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

        if self.reference_attn:
            if self.fusion_blocks == "midup":
                attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) +
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            elif self.fusion_blocks == "full":
                attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]   ## 
            attn_modules = sorted(
                attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )   ## 过滤出 transformer blocks, 按通道数从大到小排序：通常用于确保 writer/reader 对齐一致

            for i, module in enumerate(attn_modules):
                module._original_inner_forward = module.forward
                if isinstance(module, BasicTransformerBlock):
                    module.forward = hacked_basic_transformer_inner_forward.__get__(    ## __get__：把普通函数绑定为该 module 的方法（让 hacked forward 的 self 指向 module）
                        module,
                        BasicTransformerBlock)  ## 这个 block 的 forward() 确实被“完全替换”了——原 forward 不再被调用。
                if isinstance(module, TemporalBasicTransformerBlock):   ## 和AudioTemporalBasicTransformerBlock无关
                    module.forward = hacked_basic_transformer_inner_forward.__get__(
                        module,
                        TemporalBasicTransformerBlock)
                module.bank = []    ## bank：每个 transformer block 一个 bank（list）
                module.attn_weight = float(i) / float(len(attn_modules))

    def update(self, writer, dtype=torch.float16):  ## 把 writer.bank → reader.bank
        """
        Update the model's parameters.

        Args:
            writer (torch.nn.Module): The model's writer object.
            dtype (torch.dtype, optional): The data type to be used for the update. Defaults to torch.float16.

        Returns:
            None.
        """
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) +
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in (
                        torch_dfs(writer.unet.mid_block)
                        + torch_dfs(writer.unet.up_blocks)
                    )
                    if isinstance(module, BasicTransformerBlock)
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, TemporalBasicTransformerBlock)
                ]
                writer_attn_modules = [
                    module
                    for module in torch_dfs(writer.unet)
                    if isinstance(module, BasicTransformerBlock)
                ]
            assert len(reader_attn_modules) == len(writer_attn_modules)
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            writer_attn_modules = sorted(
                writer_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r, w in zip(reader_attn_modules, writer_attn_modules):
                r.bank = [v.clone().to(dtype) for v in w.bank]


    def clear(self):    ## 清空所有被 patch 的模块的 bank
        """
        Clears the attention bank of all reader attention modules.

        This method is used when the `reference_attn` attribute is set to `True`.
        It clears the attention bank of all reader attention modules inside the UNet
        model based on the selected `fusion_blocks` mode.

        If `fusion_blocks` is set to "midup", it searches for reader attention modules
        in both the mid block and up blocks of the UNet model. If `fusion_blocks` is set
        to "full", it searches for reader attention modules in the entire UNet model.

        It sorts the reader attention modules by the number of neurons in their
        `norm1.normalized_shape[0]` attribute in descending order. This sorting ensures
        that the modules with more neurons are cleared first.

        Finally, it iterates through the sorted list of reader attention modules and
        calls the `clear()` method on each module's `bank` attribute to clear the
        attention bank.
        """
        if self.reference_attn:
            if self.fusion_blocks == "midup":
                reader_attn_modules = [
                    module
                    for module in (
                        torch_dfs(self.unet.mid_block) +
                        torch_dfs(self.unet.up_blocks)
                    )
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            elif self.fusion_blocks == "full":
                reader_attn_modules = [
                    module
                    for module in torch_dfs(self.unet)
                    if isinstance(module, (BasicTransformerBlock, TemporalBasicTransformerBlock))
                ]
            reader_attn_modules = sorted(
                reader_attn_modules, key=lambda x: -x.norm1.normalized_shape[0]
            )
            for r in reader_attn_modules:
                r.bank.clear()
