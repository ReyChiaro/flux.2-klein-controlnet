import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Any
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FluxTransformer2DLoadersMixin, FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import apply_lora_scale, logging
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous
from diffusers.utils import BaseOutput
from diffusers.models.controlnets.controlnet import zero_module
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel

from diffusers.models.transformers.transformer_flux2 import (
    Flux2PosEmbed,
    Flux2Modulation,
    Flux2SingleTransformerBlock,
    Flux2TimestepGuidanceEmbeddings,
    Flux2TransformerBlock,
    FluxTransformer2DLoadersMixin,
    Flux2Transformer2DModel,
)


@dataclass
class Flux2KleinControlNetOutput(BaseOutput):
    controlnet_block_samples: tuple[torch.Tensor]
    controlnet_single_block_samples: tuple[torch.Tensor]


class FLUX2KleinControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: int | None = None,
        num_layers: int = 8,
        num_single_layers: int = 48,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        joint_attention_dim: int = 15360,
        timestep_guidance_channels: int = 256,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
        eps: float = 1e-6,
        guidance_embeds: bool = True,
        extra_condition_channles: int = 3,
    ):
        r"""
        :param extra_condition_channels (int): We input mask with shape (1, H, W), the mask will
            be concatenated on the image and then be packed together so there are (16+1) * 4 channels.
        """
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            in_channels=timestep_guidance_channels,
            embedding_dim=self.inner_dim,
            bias=False,
            guidance_embeds=guidance_embeds,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.x_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                Flux2TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList(
            [
                Flux2SingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                    bias=False,
                )
                for _ in range(num_single_layers)
            ]
        )

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=eps, bias=False
        )
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

        # 8. ControlNet
        self.controlnet_x_embedder = zero_module(nn.Linear(in_channels + extra_condition_channles, self.inner_dim))
        self.controlnet_blocks = nn.ModuleList()
        self.controlnet_single_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.controlnet_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))
        for _ in range(num_single_layers):
            self.controlnet_single_blocks.append(zero_module(nn.Linear(self.inner_dim, self.inner_dim)))

        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(
        cls: "FLUX2KleinControlNetModel",
        transformer: Flux2Transformer2DModel,
        num_layers: int = 4,
        num_single_layers: int = 10,
        attention_head_dim: int = 128,
        num_attention_heads: int = 32,
        load_weights_from_transformer: bool = True,
    ):
        config = dict(transformer.config)
        config["num_layers"] = num_layers
        config["num_single_layers"] = num_single_layers
        config["attention_head_dim"] = attention_head_dim
        config["num_attention_heads"] = num_attention_heads

        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_guidance_embed.load_state_dict(transformer.time_guidance_embed.state_dict())
            controlnet.double_stream_modulation_img.load_state_dict(
                transformer.double_stream_modulation_img.state_dict()
            )
            controlnet.double_stream_modulation_txt.load_state_dict(
                transformer.double_stream_modulation_txt.state_dict()
            )
            controlnet.single_stream_modulation.load_state_dict(transformer.single_stream_modulation.state_dict())
            controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(
                transformer.transformer_blocks.state_dict(),
                strict=False,
            )
            controlnet.single_transformer_blocks.load_state_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )
            controlnet.norm_out.load_state_dict(transformer.norm_out.state_dict())
            controlnet.proj_out.load_state_dict(transformer.proj_out.state_dict())

        return controlnet

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        timestep: torch.LongTensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ) -> torch.Tensor | Flux2KleinControlNetOutput:
        if encoder_hidden_states is not None:
            num_txt_tokens = encoder_hidden_states.shape[1]
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = self.time_guidance_embed(timestep, guidance)

        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)

        # Add control signal to noisy latents
        hidden_states = self.x_embedder(hidden_states)
        cond_states = self.controlnet_x_embedder(controlnet_cond)
        hidden_states[:, : cond_states.shape[1], ...] = hidden_states[:, : cond_states.shape[1], ...] + cond_states

        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]

        # cos, sin
        img_rotary_emb = self.pos_embed(img_ids)
        txt_rotary_emb = self.pos_embed(txt_ids)
        rotary_emb = (
            torch.cat([txt_rotary_emb[0], img_rotary_emb[0]], dim=0),
            torch.cat([txt_rotary_emb[1], img_rotary_emb[1]], dim=0),
        )

        cached_block_samples = ()
        for block_index, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    double_stream_mod_img,
                    double_stream_mod_txt,
                    rotary_emb,
                    joint_attention_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=double_stream_mod_img,
                    temb_mod_txt=double_stream_mod_txt,
                    image_rotary_emb=rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            cached_block_samples = cached_block_samples + (hidden_states,)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        cached_single_block_samples = ()
        for block_index, block in enumerate(self.single_transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    None,
                    single_stream_mod,
                    rotary_emb,
                    joint_attention_kwargs,
                    False,
                    None,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=None,
                    temb_mod=single_stream_mod,
                    image_rotary_emb=rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                    split_hidden_states=False,
                    text_seq_len=None,
                )
            cached_single_block_samples = cached_single_block_samples + (hidden_states[:, num_txt_tokens:, ...],)

        cached_controlnet_samples = ()
        for block_sample, controlnet_block in zip(cached_block_samples, self.controlnet_blocks):
            block_sample = controlnet_block(block_sample)
            cached_controlnet_samples = cached_controlnet_samples + (block_sample,)

        cached_controlnet_single_samples = ()
        for block_sample, controlnet_block in zip(cached_single_block_samples, self.controlnet_single_blocks):
            block_sample = controlnet_block(block_sample)
            cached_controlnet_single_samples = cached_controlnet_single_samples + (block_sample,)

        cached_controlnet_samples = [sample * conditioning_scale for sample in cached_controlnet_samples]
        cached_controlnet_single_samples = [sample * conditioning_scale for sample in cached_controlnet_single_samples]

        if not return_dict:
            return (cached_controlnet_samples, cached_controlnet_single_samples)

        return Flux2KleinControlNetOutput(
            controlnet_block_samples=cached_controlnet_samples,
            controlnet_single_block_samples=cached_controlnet_single_samples,
        )
