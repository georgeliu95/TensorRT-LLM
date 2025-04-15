import argparse
import json
import os
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from cuda import cudart
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.models.embeddings import get_3d_sincos_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import export_to_video, load_image

import tensorrt_llm
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.logger import logger
from tensorrt_llm.models.cogvideox.config import \
    CogVideoXTransformer3DModelConfig
from tensorrt_llm.runtime.session import Session, TensorInfo


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class TllmCogVideoX(object):

    def __init__(
        self,
        config,
        engine_dir,
        debug_mode: bool = True,
        stream: torch.cuda.Stream = None,
    ):
        self.dtype = config['pretrained_config']['dtype']
        self.config = CogVideoXTransformer3DModelConfig.from_dict(
            config['pretrained_config'])

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        gpus_per_node = config['pretrained_config']['mapping']['gpus_per_node']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=gpus_per_node)

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(engine_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None

        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = [
            'hidden_states', 'encoder_hidden_states', 'timestep',
            'image_rotary_emb_cos', 'image_rotary_emb_sin',
            'update_pos_embedding', 'output'
        ]

        found_tensor_names = [
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected.")
        if self.debug_mode:
            self.debug_tensors = list(
                set(found_tensor_names) - set(expected_tensor_names))

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, batch_size):
        post_time_compression_frames = (
            self.config.sample_frames -
            1) // self.config.temporal_compression_ratio + 1
        post_patch_height = self.config.sample_height // self.config.patch_size
        post_patch_width = self.config.sample_width // self.config.patch_size
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames
        pos_embedding_dim = self.config.num_attention_heads * self.config.attention_head_dim
        input_infos = [
            TensorInfo(name='hidden_states',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(batch_size, post_time_compression_frames,
                              self.config.in_channels,
                              self.config.sample_height,
                              self.config.sample_width)),
            TensorInfo(name='encoder_hidden_states',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(batch_size, self.config.max_text_seq_length,
                              self.config.text_embed_dim)),
            TensorInfo(name='timestep',
                       dtype=str_dtype_to_trt('int64'),
                       shape=(batch_size, )),
            TensorInfo(name='image_rotary_emb_cos',
                       dtype=str_dtype_to_trt('float32'),
                       shape=(num_patches, self.config.attention_head_dim)),
            TensorInfo(name='image_rotary_emb_sin',
                       dtype=str_dtype_to_trt('float32'),
                       shape=(num_patches, self.config.attention_head_dim)),
            TensorInfo(name='update_pos_embedding',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(1, num_patches + self.config.max_text_seq_length,
                              pos_embedding_dim)),
        ]
        output_info = self.session.infer_shapes(input_infos)
        for t_info in output_info:
            self.outputs[t_info.name] = torch.empty(tuple(t_info.shape),
                                                    dtype=trt_dtype_to_torch(
                                                        t_info.dtype),
                                                    device=self.device)
        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @cuda_stream_guard
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        ofs: Optional[Union[int, float, torch.LongTensor]] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        self._setup(batch_size=hidden_states.shape[0])
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        _, num_frames, _, height, width = hidden_states.shape
        pre_time_compression_frames = (
            num_frames - 1) * self.config.temporal_compression_ratio + 1
        update_pos_embedding = self._get_positional_embeddings(
            height,
            width,
            pre_time_compression_frames,
            device=hidden_states.device)

        inputs = {
            'hidden_states':
            hidden_states.to(str_dtype_to_torch(self.dtype)),
            'encoder_hidden_states':
            encoder_hidden_states.to(str_dtype_to_torch(self.dtype)),
            'timestep':
            timestep.to(str_dtype_to_torch('int64')),
            'image_rotary_emb_cos':
            image_rotary_emb[0].to(str_dtype_to_torch('float32')),
            'image_rotary_emb_sin':
            image_rotary_emb[1].to(str_dtype_to_torch('float32')),
            'update_pos_embedding':
            update_pos_embedding.to(str_dtype_to_torch(self.dtype)),
        }

        for k, v in inputs.items():
            inputs[k] = v.cuda().contiguous()
        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)
        ok = self.session.run(self.inputs, self.outputs,
                              self.stream.cuda_stream)

        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        output = self.outputs['output'].to(hidden_states.device)

        if self.debug_mode:
            torch.cuda.synchronize()
            output_np = {
                k: v.cpu().float().numpy()
                for k, v in self.outputs.items()
            }
            np.savez("tllm_output.npz", **output_np)
            self.outputs.pop('output')
            if not return_dict:
                return (output, self.outputs)
            else:
                return Transformer2DModelOutput(sample=output), self.outputs

        if not return_dict:
            return (output, )
        else:
            return Transformer2DModelOutput(sample=output)

    def _get_positional_embeddings(self,
                                   sample_height: int,
                                   sample_width: int,
                                   sample_frames: int,
                                   device: Optional[torch.device] = None):
        post_patch_height = sample_height // self.config.patch_size
        post_patch_width = sample_width // self.config.patch_size
        post_time_compression_frames = (
            sample_frames - 1) // self.config.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        embed_dim = self.config.num_attention_heads * self.config.attention_head_dim
        pos_embedding = get_3d_sincos_pos_embed(
            embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.config.spatial_interpolation_scale,
            self.config.temporal_interpolation_scale,
            device=device,
            output_type="pt",
        )

        pos_embedding = pos_embedding.flatten(0, 1)
        joint_pos_embedding = pos_embedding.new_zeros(
            1,
            self.config.max_text_seq_length + num_patches,
            embed_dim,
            requires_grad=False)
        joint_pos_embedding.data[:, self.config.max_text_seq_length:].copy_(
            pos_embedding)

        return joint_pos_embedding


def main(args):
    tensorrt_llm.logger.set_level(args.log_level)
    assert torch.cuda.is_available()

    config_file = os.path.join(args.tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        config['pretrained_config']['model_path'],
        torch_dtype=str_dtype_to_torch(config['pretrained_config']['dtype']))
    pipe.to(f"cuda:{tensorrt_llm.mpi_rank()}")

    del pipe.transformer
    torch.cuda.empty_cache()

    # Load model
    model = TllmCogVideoX(
        config,
        engine_dir=args.tllm_model_dir,
        debug_mode=args.debug_mode,
    )
    pipe.transformer = model

    if args.optimize_cuda_memory:
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()

    # Load image from path
    if args.image_path is None:
        args.image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
    image = load_image(args.image_path)

    with torch.no_grad():
        frames = pipe(image,
                      args.prompt,
                      height=480,
                      width=720,
                      num_inference_steps=args.num_inference_steps,
                      generator=torch.Generator("cpu").manual_seed(0)).frames[0]
    if tensorrt_llm.mpi_rank() == 0:
        # # 打印GPU内存使用情况
        # print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        # print(f"Reserved memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        # print(f"Max allocated memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        # print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
        export_to_video(frames, "cogvideox_output.mp4", fps=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'prompt',
        nargs='*',
        default=
        "An astronaut hatching from an egg and dancing on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot.",
        help="Text prompt(s) to guide image generation")
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--optimize_cuda_memory", action='store_true')
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument("--debug_mode", action='store_true')
    args = parser.parse_args()
    main(args)
