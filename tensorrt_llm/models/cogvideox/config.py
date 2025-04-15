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
from typing import Any, Dict, Optional

from ...mapping import Mapping
from ..convert_utils import infer_dtype
from ..modeling_utils import PretrainedConfig, QuantConfig


class CogVideoXTransformer3DModelConfig(PretrainedConfig):

    def __init__(self,
                 *,
                 num_attention_heads: int = 30,
                 attention_head_dim: int = 64,
                 in_channels: int = 16,
                 out_channels: Optional[int] = 16,
                 flip_sin_to_cos: bool = True,
                 freq_shift: int = 0,
                 time_embed_dim: int = 512,
                 ofs_embed_dim: Optional[int] = None,
                 text_embed_dim: int = 4096,
                 num_layers: int = 30,
                 attention_bias: bool = True,
                 sample_width: int = 90,
                 sample_height: int = 60,
                 sample_frames: int = 49,
                 patch_size: int = 2,
                 patch_size_t: Optional[int] = None,
                 temporal_compression_ratio: int = 4,
                 max_text_seq_length: int = 226,
                 activation_fn: str = "gelu-approximate",
                 timestep_activation_fn: str = "silu",
                 norm_elementwise_affine: bool = True,
                 norm_eps: float = 1e-5,
                 spatial_interpolation_scale: float = 1.875,
                 temporal_interpolation_scale: float = 1.0,
                 use_rotary_positional_embeddings: bool = False,
                 use_learned_positional_embeddings: bool = False,
                 patch_bias: bool = True,
                 use_pretrained_pos_emb: bool = False,
                 **kwargs):

        kwargs.update({
            'hidden_size': attention_head_dim * num_attention_heads,
            'num_hidden_layers': num_layers,
            'num_attention_heads': num_attention_heads
        })
        super().__init__(**kwargs)
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.freq_shift = freq_shift
        self.time_embed_dim = time_embed_dim
        self.ofs_embed_dim = ofs_embed_dim
        self.text_embed_dim = text_embed_dim
        self.num_layers = num_layers
        self.attention_bias = attention_bias
        self.sample_width = sample_width
        self.sample_height = sample_height
        self.sample_frames = sample_frames
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.activation_fn = activation_fn
        self.timestep_activation_fn = timestep_activation_fn
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings
        self.patch_bias = patch_bias
        self.use_pretrained_pos_emb = use_pretrained_pos_emb

    def to_dict(self):
        output = super().to_dict()
        # Serialize the fields added in CogVideoXTransformer3DModelConfig
        output['inner_dim'] = self.inner_dim
        output['num_attention_heads'] = self.num_attention_heads
        output['attention_head_dim'] = self.attention_head_dim
        output['in_channels'] = self.in_channels
        output['out_channels'] = self.out_channels
        output['flip_sin_to_cos'] = self.flip_sin_to_cos
        output['freq_shift'] = self.freq_shift
        output['time_embed_dim'] = self.time_embed_dim
        output['ofs_embed_dim'] = self.ofs_embed_dim
        output['text_embed_dim'] = self.text_embed_dim
        output['num_layers'] = self.num_layers
        output['attention_bias'] = self.attention_bias
        output['sample_width'] = self.sample_width
        output['sample_height'] = self.sample_height
        output['sample_frames'] = self.sample_frames
        output['patch_size'] = self.patch_size
        output['patch_size_t'] = self.patch_size_t
        output['temporal_compression_ratio'] = self.temporal_compression_ratio
        output['max_text_seq_length'] = self.max_text_seq_length
        output['activation_fn'] = self.activation_fn
        output['timestep_activation_fn'] = self.timestep_activation_fn
        output['norm_elementwise_affine'] = self.norm_elementwise_affine
        output['norm_eps'] = self.norm_eps
        output['spatial_interpolation_scale'] = self.spatial_interpolation_scale
        output[
            'temporal_interpolation_scale'] = self.temporal_interpolation_scale
        output[
            'use_rotary_positional_embeddings'] = self.use_rotary_positional_embeddings
        output[
            'use_learned_positional_embeddings'] = self.use_learned_positional_embeddings
        output['patch_bias'] = self.patch_bias
        output['use_pretrained_pos_emb'] = self.use_pretrained_pos_emb
        output['hidden_size'] = self.hidden_size
        output['num_hidden_layers'] = self.num_hidden_layers

        return output

    @classmethod
    def from_hugging_face_config(cls,
                                 hf_config: Dict[str, Any],
                                 dtype: str = 'auto',
                                 mapping: Optional[Mapping] = None,
                                 quant_config: Optional[QuantConfig] = None,
                                 **kwargs):
        num_attention_heads = hf_config['num_attention_heads']
        attention_head_dim = hf_config['attention_head_dim']
        in_channels = hf_config['in_channels']
        out_channels = hf_config['out_channels']
        flip_sin_to_cos = hf_config['flip_sin_to_cos']
        freq_shift = hf_config['freq_shift']
        time_embed_dim = hf_config['time_embed_dim']
        text_embed_dim = hf_config['text_embed_dim']
        num_layers = hf_config['num_layers']
        attention_bias = hf_config['attention_bias']
        sample_width = hf_config['sample_width']
        sample_height = hf_config['sample_height']
        sample_frames = hf_config['sample_frames']
        patch_size = hf_config['patch_size']
        temporal_compression_ratio = hf_config['temporal_compression_ratio']
        max_text_seq_length = hf_config['max_text_seq_length']
        activation_fn = hf_config['activation_fn']
        timestep_activation_fn = hf_config['timestep_activation_fn']
        norm_elementwise_affine = hf_config['norm_elementwise_affine']
        norm_eps = hf_config['norm_eps']
        spatial_interpolation_scale = hf_config['spatial_interpolation_scale']
        temporal_interpolation_scale = hf_config['temporal_interpolation_scale']
        use_rotary_positional_embeddings = hf_config[
            'use_rotary_positional_embeddings']
        patch_bias = hf_config['patch_bias']
        use_pretrained_pos_emb = kwargs.get('use_pretrained_pos_emb', False)
        dtype = infer_dtype(dtype, hf_config.get('torch_dtype'))

        return cls(
            architecture='CogVideoXTransformer3DModel',
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            flip_sin_to_cos=flip_sin_to_cos,
            freq_shift=freq_shift,
            time_embed_dim=time_embed_dim,
            text_embed_dim=text_embed_dim,
            num_layers=num_layers,
            attention_bias=attention_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            patch_size=patch_size,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            activation_fn=activation_fn,
            timestep_activation_fn=timestep_activation_fn,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            patch_bias=patch_bias,
            use_pretrained_pos_emb=use_pretrained_pos_emb,
            dtype=dtype,
            mapping=mapping,
            quantization=quant_config,
            **kwargs)
