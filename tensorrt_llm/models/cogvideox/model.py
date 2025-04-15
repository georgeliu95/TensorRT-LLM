# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

from diffusers import CogVideoXImageToVideoPipeline

from ..._utils import str_dtype_to_torch
from ...functional import Tensor, allgather, chunk, concat, pad, shape, split
from ...layers import LayerNorm, Linear
from ...layers.attention import DiffusersAttention
from ...layers.embedding import (CogVideoXPatchEmbed, TimestepEmbedding,
                                 Timesteps)
from ...layers.mlp import (LinearActivation, LinearApproximateGELU, LinearGEGLU,
                           LinearGELU, LinearSwiGLU)
from ...layers.normalization import AdaLayerNorm, CogVideoXLayerNormZero
from ...logger import logger
from ...mapping import Mapping
from ...module import Module, ModuleList
from ..model_weights_loader import ModelWeightsLoader
from ..modeling_utils import PretrainedModel
from .config import CogVideoXTransformer3DModelConfig


def pad_chunk(tensor, chunks, dim):
    assert not tensor.is_dynamic(dim)
    ndim = tensor.ndim()
    if dim < 0:
        dim += ndim
    dim_value = tensor.size()[dim]
    if dim_value % chunks == 0:
        return chunk(tensor, chunks, dim)
    else:
        # print(f">>> chunks: {chunks}")
        # print(f">>> dim_value: {dim_value}")
        # print(f">>> tensor.shape before pad: {tensor.shape}")
        pad_size = [0] * ndim * 2
        pad_size[2 * (ndim - dim - 1) + 1] = chunks - (dim_value % chunks)
        # print(f">>> pad_size: {pad_size}")
        tensor = pad(tensor, pad_size)
        # print(f">>> tensor.shape after pad: {tensor.shape}")
        return chunk(tensor, chunks, dim)


class FeedForward(Module):

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            mult: int = 4,
            activation_fn: str = "geglu",
            inner_dim=None,
            bias: bool = True,
            mapping=Mapping(),
            dtype=None,
    ):
        super().__init__()

        self.mapping = mapping
        self.dtype = dtype

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            raise NotImplementedError('GELU only support tanh now.')
        if activation_fn == "gelu-approximate":
            act_fn = LinearGELU(dim,
                                inner_dim,
                                approximate="tanh",
                                bias=bias,
                                mapping=mapping,
                                dtype=self.dtype)
        elif activation_fn == "geglu":
            act_fn = LinearGEGLU(dim,
                                 inner_dim,
                                 approximate="tanh",
                                 bias=bias,
                                 mapping=mapping,
                                 dtype=self.dtype)
        elif activation_fn == "geglu-approximate":
            act_fn = LinearApproximateGELU(dim,
                                           inner_dim,
                                           bias=bias,
                                           mapping=mapping,
                                           dtype=self.dtype)
        elif activation_fn == "swiglu":
            act_fn = LinearSwiGLU(dim,
                                  inner_dim,
                                  bias=bias,
                                  mapping=mapping,
                                  dtype=self.dtype)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim,
                                      inner_dim,
                                      bias=bias,
                                      activation="silu",
                                      mapping=mapping,
                                      dtype=self.dtype)

        self.net = ModuleList([
            act_fn,
            Linear(inner_dim,
                   dim_out,
                   bias=bias,
                   tp_group=self.mapping.tp_group,
                   tp_size=self.mapping.tp_size,
                   dtype=self.dtype)
        ])

    def forward(self, hidden_states: Tensor):
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class CogVideoXBlock(Module):

    def __init__(self,
                 dim: int,
                 num_attention_heads: int,
                 attention_head_dim: int,
                 time_embed_dim: int,
                 activation_fn: str = "gelu-approximate",
                 attention_bias: bool = False,
                 qk_norm: bool = True,
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 ff_inner_dim: Optional[int] = None,
                 ff_bias: bool = True,
                 attention_out_bias: bool = True,
                 mapping=Mapping(),
                 dtype=None):
        super().__init__()
        self.mapping = mapping
        self.dtype = dtype

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim,
                                            dim,
                                            norm_elementwise_affine,
                                            norm_eps,
                                            bias=True,
                                            dtype=self.dtype)

        self.attn1 = DiffusersAttention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor_fn="cogvideox_attn_forward",
            mapping=self.mapping,
            dtype=self.dtype,
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim,
                                            dim,
                                            norm_elementwise_affine,
                                            norm_eps,
                                            bias=True,
                                            dtype=self.dtype)

        self.ff = FeedForward(dim=dim,
                              activation_fn=activation_fn,
                              inner_dim=ff_inner_dim,
                              bias=ff_bias,
                              mapping=self.mapping,
                              dtype=self.dtype)

    def forward(self,
                hidden_states: Tensor,
                encoder_hidden_states: Tensor,
                temb: Tensor,
                image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
                *args,
                **kwargs):
        text_seq_length = shape(encoder_hidden_states, 1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb)

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb)

        # feed-forward
        norm_hidden_states = concat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        ff_output_part0, ff_output_part1 = split(
            ff_output, [text_seq_length,
                        shape(ff_output, 1) - text_seq_length],
            dim=1)
        hidden_states = hidden_states + gate_ff * ff_output_part1
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output_part0

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(PretrainedModel):
    config_class = CogVideoXTransformer3DModelConfig

    def __init__(self, config: CogVideoXTransformer3DModelConfig):
        super().__init__(config)
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        self.dtype = config.dtype

        self.in_channels = config.in_channels
        default_out_channels = config.in_channels
        self.out_channels = config.out_channels if config.out_channels is not None else default_out_channels
        self.inner_dim = config.num_attention_heads * config.attention_head_dim

        if not config.use_rotary_positional_embeddings and config.use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues.")

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=config.patch_size,
            patch_size_t=config.patch_size_t,
            in_channels=config.in_channels,
            embed_dim=config.inner_dim,
            text_embed_dim=config.text_embed_dim,
            bias=config.patch_bias,
            sample_width=config.sample_width,
            sample_height=config.sample_height,
            sample_frames=config.sample_frames,
            temporal_compression_ratio=config.temporal_compression_ratio,
            max_text_seq_length=config.max_text_seq_length,
            spatial_interpolation_scale=config.spatial_interpolation_scale,
            temporal_interpolation_scale=config.temporal_interpolation_scale,
            use_positional_embeddings=not config.
            use_rotary_positional_embeddings,
            use_learned_positional_embeddings=config.
            use_learned_positional_embeddings,
            dtype=self.dtype)

        # 2. Time embeddings and ofs embedding(Only CogVideoX1.5-5B I2V have)
        self.time_proj = Timesteps(config.inner_dim, config.flip_sin_to_cos,
                                   config.freq_shift)
        self.time_embedding = TimestepEmbedding(config.inner_dim,
                                                config.time_embed_dim,
                                                config.timestep_activation_fn,
                                                dtype=self.dtype)

        self.ofs_proj = None
        self.ofs_embedding = None
        if config.ofs_embed_dim:
            self.ofs_proj = Timesteps(config.ofs_embed_dim,
                                      config.flip_sin_to_cos, config.freq_shift)
            # same as time embeddings, for ofs
            self.ofs_embedding = TimestepEmbedding(
                config.ofs_embed_dim,
                config.ofs_embed_dim,
                config.timestep_activation_fn,
                dtype=self.dtype)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = ModuleList([
            CogVideoXBlock(
                dim=config.inner_dim,
                num_attention_heads=config.num_attention_heads,
                attention_head_dim=config.attention_head_dim,
                time_embed_dim=config.time_embed_dim,
                activation_fn=config.activation_fn,
                attention_bias=config.attention_bias,
                norm_elementwise_affine=config.norm_elementwise_affine,
                norm_eps=config.norm_eps,
                mapping=self.mapping,
                dtype=self.dtype) for _ in range(config.num_layers)
        ])
        self.norm_final = LayerNorm(config.inner_dim,
                                    config.norm_eps,
                                    config.norm_elementwise_affine,
                                    dtype=self.dtype)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=config.time_embed_dim,
            output_dim=2 * config.inner_dim,
            norm_elementwise_affine=config.norm_elementwise_affine,
            norm_eps=config.norm_eps,
            chunk_dim=1,
            dtype=self.dtype)

        if config.patch_size_t is None:
            # For CogVideox 1.0
            output_dim = config.patch_size * config.patch_size * config.out_channels
        else:
            # For CogVideoX 1.5
            output_dim = config.patch_size * config.patch_size * config.patch_size_t * config.out_channels

        self.proj_out = Linear(config.inner_dim,
                               output_dim,
                               tp_group=self.mapping.tp_group,
                               tp_size=self.mapping.tp_size,
                               dtype=self.dtype)

        self.use_pretrained_pos_emb = config.use_pretrained_pos_emb
        self.config = config

    def forward(self,
                hidden_states: Tensor,
                encoder_hidden_states: Optional[Tensor] = None,
                timestep: Optional[Tensor] = None,
                timestep_cond: Optional[Tensor] = None,
                ofs: Optional[Union[int, float, Tensor]] = None,
                image_rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
                update_pos_embedding: Optional[Tensor] = None,
                attention_kwargs: Optional[Dict[str, Any]] = None):
        # [TODO] Add support for LoRA

        batch_size = shape(hidden_states, 0)
        num_frames = shape(hidden_states, 1)
        height = shape(hidden_states, 3)
        width = shape(hidden_states, 4)

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.cast(hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.cast(hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(
            encoder_hidden_states,
            hidden_states,
            update_pos_embedding=update_pos_embedding)

        text_seq_length = shape(encoder_hidden_states, 1)
        encoder_hidden_states, hidden_states = split(
            hidden_states,
            [text_seq_length,
             shape(hidden_states, 1) - text_seq_length],
            dim=1)

        # print("hidden_states.shape", hidden_states.shape)
        # print("encoder_hidden_states.shape", encoder_hidden_states.shape)
        if self.mapping.cp_size > 1:
            hidden_states_ogn_dim = hidden_states.size()[1]
            if hidden_states_ogn_dim % self.mapping.cp_size == 0:
                hidden_states_ogn_dim = None

            hidden_states = pad_chunk(hidden_states,
                                      chunks=self.mapping.cp_size,
                                      dim=1)[self.mapping.cp_rank]
            for idx in range(len(image_rotary_emb)):
                # print(f">>> image_rotary_emb[{idx}].shape: {image_rotary_emb[idx].shape}")
                image_rotary_emb[idx] = pad_chunk(image_rotary_emb[idx],
                                                  chunks=self.mapping.cp_size,
                                                  dim=0)[self.mapping.cp_rank]
            # print(f">>> encoder_hidden_states.shape before pad: {encoder_hidden_states.shape}")
            encoder_hidden_states_ogn_dim = encoder_hidden_states.size()[1]
            if encoder_hidden_states_ogn_dim % self.mapping.cp_size == 0:
                encoder_hidden_states_ogn_dim = None
            encoder_hidden_states = pad_chunk(encoder_hidden_states,
                                              chunks=self.mapping.cp_size,
                                              dim=1)[self.mapping.cp_rank]
            # print(f">>> encoder_hidden_states.shape after pad: {encoder_hidden_states.shape}")

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        # All gather after CP
        if self.mapping.cp_size > 1:
            hidden_states = allgather(hidden_states,
                                      self.mapping.cp_group,
                                      gather_dim=1)
            encoder_hidden_states = allgather(encoder_hidden_states,
                                              self.mapping.cp_group,
                                              gather_dim=1)
            # print(f">>> hidden_states.shape after allgather: {hidden_states.shape}")
            # print(f">>> encoder_hidden_states.shape after allgather: {encoder_hidden_states.shape}")

            if hidden_states_ogn_dim is not None:
                hidden_states, _ = split(hidden_states, [
                    hidden_states_ogn_dim,
                    shape(hidden_states, 1) - hidden_states_ogn_dim
                ],
                                         dim=1)
            if encoder_hidden_states_ogn_dim is not None:
                encoder_hidden_states, _ = split(encoder_hidden_states, [
                    encoder_hidden_states_ogn_dim,
                    shape(encoder_hidden_states, 1) -
                    encoder_hidden_states_ogn_dim
                ],
                                                 dim=1)
            # print(f">>> hidden_states.shape after reconstruction: {hidden_states.shape}")
            # print(f">>> encoder_hidden_states.shape after reconstruction: {encoder_hidden_states.shape}")
            # exit()

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            hidden_states = concat([encoder_hidden_states, hidden_states],
                                   dim=1)
            hidden_states = self.norm_final(hidden_states)
            _, hidden_states = split(
                hidden_states,
                [text_seq_length,
                 shape(hidden_states, 1) - text_seq_length],
                dim=1)

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.view(
                concat(
                    [batch_size, num_frames, height // p, width // p, -1, p,
                     p]))
            output = output.permute([0, 1, 4, 2, 5, 3,
                                     6]).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.view(
                concat([
                    batch_size, (num_frames + p_t - 1) // p_t, height // p,
                    width // p, -1, p_t, p, p
                ]))
            output = output.permute([0, 1, 5, 4, 2, 6, 3,
                                     7]).flatten(6, 7).flatten(4,
                                                               5).flatten(1, 2)

        output.mark_output("output")
        return output

    def prepare_inputs(self, max_batch_size, **kwargs):

        def cogvideox_default_range(max_batch_size):
            # return [1, max(1, (max_batch_size + 1) // 2), max_batch_size]
            return [max_batch_size] * 3

        default_range = cogvideox_default_range

        latent_channels = self.config.in_channels
        post_time_compression_frames = (
            self.config.sample_frames -
            1) // self.config.temporal_compression_ratio + 1
        hidden_states = Tensor(
            name='hidden_states',
            dtype=self.dtype,
            shape=[
                # -1,
                # -1,
                max_batch_size,
                post_time_compression_frames,
                latent_channels,
                self.config.sample_height,
                self.config.sample_width,
            ],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                # ('latent_frames', [default_range(post_time_compression_frames)
                #                    ]),
                ('latent_frames', [[post_time_compression_frames] * 3]),
                ('latent_channels', [[latent_channels] * 3]),
                ('height', [[self.config.sample_height] * 3]),
                ('width', [[self.config.sample_width] * 3]),
            ]))
        encoder_hidden_states = Tensor(
            name='encoder_hidden_states',
            dtype=self.dtype,
            shape=[
                # -1,
                max_batch_size,
                self.config.max_text_seq_length,
                self.config.text_embed_dim
            ],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
                ('max_text_seq_length', [[self.config.max_text_seq_length] * 3
                                         ]),
                ('text_embed_dim', [[self.config.text_embed_dim] * 3]),
            ]))
        timestep = Tensor(
            name='timestep',
            dtype='int64',
            #   shape=[-1],
            shape=[max_batch_size],
            dim_range=OrderedDict([
                ('batch_size', [default_range(max_batch_size)]),
            ]))
        post_patch_height = self.config.sample_height // self.config.patch_size
        post_patch_width = self.config.sample_width // self.config.patch_size
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames
        image_rotary_emb_cos = Tensor(
            name='image_rotary_emb_cos',
            dtype='float32',
            shape=[num_patches, self.config.attention_head_dim])
        image_rotary_emb_sin = Tensor(
            name='image_rotary_emb_sin',
            dtype='float32',
            shape=[num_patches, self.config.attention_head_dim])
        pos_embedding_dim = self.config.num_attention_heads * self.config.attention_head_dim
        update_pos_embedding = Tensor(
            name='update_pos_embedding',
            dtype=self.dtype,
            shape=[
                1, num_patches + self.config.max_text_seq_length,
                pos_embedding_dim
            ])
        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "image_rotary_emb": [image_rotary_emb_cos, image_rotary_emb_sin],
            "update_pos_embedding": update_pos_embedding,
        }

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path: str,
                        dtype='float16',
                        mapping=Mapping(),
                        **kwargs):
        quant_ckpt_path = kwargs.pop('quant_ckpt_path', None)

        transformer = CogVideoXImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=str_dtype_to_torch(dtype)).transformer

        config = CogVideoXTransformer3DModelConfig.from_hugging_face_config(
            transformer.config, dtype=dtype, mapping=mapping, **kwargs)

        hf_model_dir = transformer.config._name_or_path
        custom_dict = {}
        if quant_ckpt_path is not None:
            hf_model_dir = quant_ckpt_path

        loader = CogVideoXModelWeightsLoader(hf_model_dir, custom_dict)
        model = cls(config)
        loader.generate_tllm_weights(model)
        return model

    def load(self, weights, from_pruned=False):
        required_names = set()
        for name, param in self.named_parameters():
            if self.use_pretrained_pos_emb and 'pos_embed' in name:
                required_names.add(name)
                continue
            if param.is_inited():
                continue
            if name not in weights:
                # Exemption for embedding sharing
                if name.endswith('lm_head.weight') and any(
                        k.endswith('vocab_embedding.weight')
                        for k in weights.keys()):
                    continue
                if name.endswith('lm_head.per_channel_scale') and any(
                        k.endswith('vocab_embedding.per_channel_scale')
                        for k in weights.keys()):
                    continue
            required_names.add(name)

        provided_names = set(weights.keys())
        if not required_names.issubset(provided_names):
            raise RuntimeError(
                f"Required but not provided tensors:{required_names.difference(provided_names)}"
            )
        if not provided_names.issubset(required_names):
            logger.warning(
                f"Provided but not required tensors: {provided_names.difference(required_names)}"
            )

        for name, param in self.named_parameters():
            if name in provided_names:
                if not from_pruned:
                    try:
                        param.value = weights[name]
                    except Exception as e:
                        raise RuntimeError(
                            f"Encounter error '{e}' for parameter '{name}'")
                else:
                    param.set_value_or_dummy(weights[name])

    def enable_forward_chunking(self,
                                chunk_size: Optional[int] = None,
                                dim: int = 0):
        raise NotImplementedError()

    def disable_forward_chunking(self):
        raise NotImplementedError()

    @property
    def attn_processors(self):
        return None

    def set_attn_processor(self, processor):
        raise NotImplementedError()

    def fuse_qkv_projections(self):
        raise NotImplementedError()

    def unfuse_qkv_projections(self):
        raise NotImplementedError()

    def _set_gradient_checkpointing(self, module, value=False):
        raise NotImplementedError()


class CogVideoXModelWeightsLoader(ModelWeightsLoader):

    def translate_to_external_key(self, tllm_key: str,
                                  tllm_to_externel_key_dict: dict):
        """Convert and load external checkpoint into a TensorRT-LLM model.
        """
        trtllm_to_hf_name = {
            r"transformer_blocks.(\d+).ff(\w*).net.1.weight":
            "transformer_blocks.*.ff*.net.2.weight",
            r"transformer_blocks.(\d+).ff(\w*).net.1.bias":
            "transformer_blocks.*.ff*.net.2.bias",
        }
        import re
        for k, v in trtllm_to_hf_name.items():
            m = re.match(k, tllm_key)
            if m is not None:
                matched_pos = m.groups()
                placeholders = v.count('*')
                assert len(matched_pos) == placeholders
                for i in range(len(matched_pos)):
                    v = v.replace('*', matched_pos[i], 1)
                return v
        return tllm_key
