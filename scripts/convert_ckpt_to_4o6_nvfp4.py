#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Convert HF-style checkpoints to 4o6 NVFP4 weight dumps.

This is a streaming checkpoint rewriter for TRT-LLM PyTorch backend inputs.
It targets MoE expert Linear weights by default and supports three source
formats:

* BF16/FP16/FP32 dense HF safetensors: ``*.weight`` -> NVFP4 4o6.
* Existing NVFP4 safetensors: dequantize ``weight/weight_scale/weight_scale_2``
  and re-encode with 4o6.
* Compressed-tensors INT4 pack-quantized safetensors, as used by
  ``Kimi-K2-Thinking``: ``weight_packed/weight_scale/weight_shape`` -> NVFP4 4o6.

Run inside the TRT-LLM container or another environment where torch, safetensors,
and the TRT-LLM fp4 quantization ops are available.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_MOE_WEIGHT_RE = (
    r".*(?:^|\.)(?:experts|block_sparse_moe\.experts)\.\d+\."
    r"(?:gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$"
)
DEFAULT_SKIP_COPY_RE = r".*(?:\.safetensors|\.index\.json)$"
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"}
ADAPTIVE_FP4_DEFAULT_SO = "/tmp/libfp4QuantizeAdaptive.so"
ADAPTIVE_FP4_QUANT_RANGE = 1536.0
STANDARD_NVFP4_QUANT_RANGE = 448.0 * 6.0
torch = None
safe_open = None
save_file = None
_adaptive_fp4_ops_loaded = False
_e2m1_lut_cache = {}


def import_runtime_deps() -> None:
    global torch, safe_open, save_file
    if torch is not None:
        return
    import torch as _torch
    from safetensors import safe_open as _safe_open
    from safetensors.torch import save_file as _save_file

    torch = _torch
    safe_open = _safe_open
    save_file = _save_file


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    shape: Tuple[int, ...]
    filename: str


@dataclass
class SourceInfo:
    fmt: str
    config: Dict
    hf_quant_config: Optional[Dict]


def config_quantization_configs(config: Dict) -> List[Dict]:
    configs: List[Dict] = []
    top_level = config.get("quantization_config")
    if top_level:
        configs.append(top_level)
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        nested = text_config.get("quantization_config")
        if nested:
            configs.append(nested)
    return configs


class TensorStore:
    def __init__(self,
                 root: Path,
                 weight_map: Dict[str, str],
                 cache_size: int = 128):
        self.root = root
        self.weight_map = weight_map
        self.cache_size = cache_size
        self._stack = ExitStack()
        self._handles: Dict[str, object] = {}

    def close(self) -> None:
        self._stack.close()
        self._handles.clear()

    def _get_handle(self, filename: str):
        if self.cache_size <= 0:
            return None
        handle = self._handles.get(filename)
        if handle is not None:
            return handle
        if len(self._handles) >= self.cache_size:
            return None
        handle = self._stack.enter_context(
            safe_open(self.root / filename, framework="pt", device="cpu"))
        self._handles[filename] = handle
        return handle

    def get(self, key: str) -> torch.Tensor:
        import_runtime_deps()
        filename = self.weight_map[key]
        handle = self._get_handle(filename)
        if handle is not None:
            return handle.get_tensor(key)
        with safe_open(self.root / filename, framework="pt", device="cpu") as f:
            return f.get_tensor(key)


class ShardWriter:
    def __init__(self, out_dir: Path, max_shard_bytes: int,
                 shard_prefix: str = "model"):
        self.out_dir = out_dir
        self.max_shard_bytes = max_shard_bytes
        self.shard_prefix = shard_prefix
        self.buffer: Dict[str, torch.Tensor] = {}
        self.buffer_bytes = 0
        self.shard_idx = 0
        self.weight_map: Dict[str, str] = {}
        self.total_size = 0

    @staticmethod
    def _nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.numel() * tensor.element_size())

    def add(self, key: str, tensor: torch.Tensor) -> None:
        tensor = tensor.detach().cpu().contiguous()
        if tensor.numel() <= 1:
            tensor = tensor.clone()
        nbytes = self._nbytes(tensor)
        if self.buffer and self.buffer_bytes + nbytes > self.max_shard_bytes:
            self.flush()
        self.buffer[key] = tensor
        self.buffer_bytes += nbytes
        self.total_size += nbytes

    def flush(self) -> None:
        if not self.buffer:
            return
        import_runtime_deps()
        self.shard_idx += 1
        filename = f"{self.shard_prefix}-{self.shard_idx:05d}.safetensors"
        save_file(self.buffer, self.out_dir / filename)
        for key in self.buffer:
            self.weight_map[key] = filename
        self.buffer.clear()
        self.buffer_bytes = 0

    def finalize(self) -> None:
        self.flush()
        index = {
            "metadata": {"total_size": self.total_size},
            "weight_map": dict(sorted(self.weight_map.items())),
        }
        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")


def parse_size(value: str) -> int:
    text = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "tb": 1024**4,
    }
    for unit, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if text.endswith(unit):
            return int(float(text[:-len(unit)]) * mult)
    return int(text)


def read_safetensors_header(path: Path) -> Dict:
    size = path.stat().st_size
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path} is too small to be a safetensors shard")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > size - 8:
            f.seek(0)
            head = f.read(80)
            if head.startswith(b"version https://git-lfs.github.com/spec"):
                raise ValueError(
                    f"{path} is a Git LFS pointer, not a materialized "
                    "safetensors shard")
            raise ValueError(
                f"{path} has invalid safetensors header length {header_len} "
                f"for file size {size}")
        return json.loads(f.read(header_len))


def scan_safetensors_weight_map(root: Path) -> Dict[str, str]:
    weight_map: Dict[str, str] = {}
    for shard in sorted(root.glob("*.safetensors")):
        header = read_safetensors_header(shard)
        for key in header:
            if key != "__metadata__":
                weight_map[key] = shard.name
    return weight_map


def discover_weight_map(root: Path) -> Tuple[Dict[str, str], Dict[str, TensorMeta]]:
    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        try:
            with open(index_path) as f:
                weight_map = json.load(f)["weight_map"]
            if not weight_map:
                raise ValueError("empty weight_map")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(
                f"WARNING: ignoring invalid safetensors index {index_path}: {exc}; "
                "scanning shard headers instead.",
                file=sys.stderr,
            )
            weight_map = scan_safetensors_weight_map(root)
    else:
        weight_map = scan_safetensors_weight_map(root)

    metas: Dict[str, TensorMeta] = {}
    for filename in sorted(set(weight_map.values())):
        header = read_safetensors_header(root / filename)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            if key in weight_map:
                metas[key] = TensorMeta(
                    dtype=meta["dtype"],
                    shape=tuple(meta["shape"]),
                    filename=filename,
                )
    return weight_map, metas


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with open(path) as f:
        text = f.read().strip()
    if not text:
        return None
    return json.loads(text)


def detect_source(root: Path, explicit: str) -> SourceInfo:
    config = load_json(root / "config.json") or {}
    hf_quant_config = load_json(root / "hf_quant_config.json")
    if explicit != "auto":
        return SourceInfo(explicit, config, hf_quant_config)

    if hf_quant_config:
        q = hf_quant_config.get("quantization", hf_quant_config)
        if q.get("quant_algo") == "NVFP4" or q.get("quant_method") == "nvfp4":
            return SourceInfo("nvfp4", config, hf_quant_config)

    qcfgs = config_quantization_configs(config)
    if not qcfgs:
        return SourceInfo("bf16", config, hf_quant_config)
    for qcfg in qcfgs:
        if qcfg.get("quant_method") == "nvfp4" or qcfg.get("quant_algo") == "NVFP4":
            return SourceInfo("nvfp4", config, hf_quant_config)
        if qcfg.get("quant_method") == "compressed-tensors":
            group = qcfg.get("config_groups", {}).get("group_0", {})
            weights = group.get("weights", {})
            if qcfg.get("format") == "pack-quantized" and weights.get("num_bits") == 4:
                return SourceInfo("int4-compressed-tensors", config, hf_quant_config)
    raise NotImplementedError(
        f"Unsupported quantization_config for auto-detect: {qcfgs}")


def copy_sidecar_files(src: Path, dst: Path) -> None:
    skip = re.compile(DEFAULT_SKIP_COPY_RE)
    for path in sorted(src.iterdir()):
        if path.name == ".git":
            continue
        if skip.match(path.name):
            continue
        target = dst / path.name
        if path.is_dir():
            shutil.copytree(path, target, ignore=shutil.ignore_patterns(".git"))
        else:
            shutil.copy2(path, target)


def write_quant_configs(out_dir: Path, source: SourceInfo,
                        exclude_modules: Sequence[str]) -> None:
    config = dict(source.config)
    nvfp4_quant_config = {
        "quant_method": "nvfp4",
        "group_size": 16,
    }
    config["quantization_config"] = dict(nvfp4_quant_config)
    if isinstance(config.get("text_config"), dict):
        config["text_config"] = dict(config["text_config"])
        config["text_config"]["quantization_config"] = dict(nvfp4_quant_config)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")

    quant = {
        "quant_algo": "NVFP4",
        "group_size": 16,
        "exclude_modules": sorted(set(exclude_modules)),
    }
    old_quant = None
    if source.hf_quant_config:
        old_quant = source.hf_quant_config.get("quantization",
                                               source.hf_quant_config)
    else:
        old_quant_configs = config_quantization_configs(source.config)
        if old_quant_configs:
            old_quant = old_quant_configs[0]
    if old_quant and old_quant.get("kv_cache_quant_algo"):
        quant["kv_cache_quant_algo"] = old_quant["kv_cache_quant_algo"]

    payload = {
        "producer": {
            "name": "llm_4o6.convert_ckpt_to_4o6_nvfp4",
            "version": "0.1",
        },
        "quantization": quant,
    }
    with open(out_dir / "hf_quant_config.json", "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _load_adaptive_fp4_ops() -> None:
    global _adaptive_fp4_ops_loaded
    if _adaptive_fp4_ops_loaded:
        return
    if not hasattr(torch.ops.trtllm, "fp4_quantize_ex"):
        torch.ops.load_library(
            os.environ.get("TRTLLM_ADAPTIVE_FP4_SO",
                           ADAPTIVE_FP4_DEFAULT_SO))
    _adaptive_fp4_ops_loaded = True


def _get_e2m1_lut(device: torch.device) -> torch.Tensor:
    key = str(device)
    lut = _e2m1_lut_cache.get(key)
    if lut is None:
        lut = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0,
             -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32,
            device=device)
        _e2m1_lut_cache[key] = lut
    return lut


def _local_nvfp4_dequantize_linear_sf(weight_fp4: torch.Tensor,
                                      weight_scale: torch.Tensor,
                                      weight_scale_2: torch.Tensor,
                                      device: torch.device) -> torch.Tensor:
    weight_u8 = weight_fp4.contiguous().view(torch.uint8).to(device)
    rows = weight_u8.shape[0]
    cols = weight_u8.shape[1] * 2
    sf_cols = cols // 16

    lut = _get_e2m1_lut(device)
    low = (weight_u8 & 0x0F).long()
    high = ((weight_u8 >> 4) & 0x0F).long()
    weight_f32 = torch.empty(rows, cols, dtype=torch.float32, device=device)
    weight_f32[:, 0::2] = lut[low]
    weight_f32[:, 1::2] = lut[high]

    scale_fp8 = weight_scale.contiguous().view(torch.float8_e4m3fn).to(
        device).reshape(rows, -1)
    if scale_fp8.shape[1] < sf_cols:
        raise ValueError(
            f"NVFP4 weight scale has {scale_fp8.shape[1]} columns, "
            f"but packed weight requires {sf_cols}.")
    scale_f32 = scale_fp8[:, :sf_cols].to(torch.float32)
    scale_f32 = scale_f32.unsqueeze(-1).expand(rows, sf_cols, 16).reshape(
        rows, cols)

    global_scale = weight_scale_2.to(device=device,
                                     dtype=torch.float32).reshape(
                                         []).reciprocal()
    return (weight_f32 * scale_f32 / global_scale).to(torch.bfloat16)


def _local_adaptive_4o6_quantize_dense_group(
        weights: List[torch.Tensor],
        scale_rule: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
                                 torch.Tensor]:
    if not torch.cuda.is_available():
        raise RuntimeError("adaptive 4o6 weight conversion requires CUDA.")
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    _load_adaptive_fp4_ops()

    dense_weights = [
        weight.to(device=device, dtype=torch.bfloat16).contiguous()
        for weight in weights
    ]
    rows_per_entry = [tensor.shape[0] for tensor in dense_weights]
    combined = torch.cat(dense_weights, dim=0).contiguous()

    quant_range = (STANDARD_NVFP4_QUANT_RANGE
                   if scale_rule == 0 else ADAPTIVE_FP4_QUANT_RANGE)
    amax = torch.clamp(torch.amax(torch.abs(combined)).float(), min=1e-12)
    dynamic_global_scale = torch.tensor(quant_range,
                                        device=device,
                                        dtype=torch.float32) / amax
    dynamic_global_scale = dynamic_global_scale.reshape(())
    weight_scale_2 = (amax.detach().cpu() / quant_range).to(
        dtype=torch.float32).reshape(())

    if scale_rule == 0:
        quantized, scale_flat = torch.ops.trtllm.fp4_quantize(
            combined, dynamic_global_scale, 16, False, False)
    else:
        quantized, scale_flat = torch.ops.trtllm.fp4_quantize_ex(
            combined, dynamic_global_scale, 16, False, False, 1, scale_rule)

    sf_cols = combined.shape[1] // 16
    split_weights = quantized.split(rows_per_entry, dim=0)
    split_scales = scale_flat[:combined.shape[0] * sf_cols].reshape(
        combined.shape[0], sf_cols).split(rows_per_entry, dim=0)
    result = []
    for original_weight, new_weight, new_scale in zip(weights, split_weights,
                                                      split_scales):
        result.append((new_weight.contiguous().to(original_weight.device),
                       new_scale.contiguous().to(original_weight.device)))
    return result, weight_scale_2


def _float_to_e4m3_bytes_and_values(values: torch.Tensor
                                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    values_fp8 = values.to(torch.float8_e4m3fn)
    return values_fp8.view(torch.uint8), values_fp8.to(torch.float32)


def _nearest_e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    thresholds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                              dtype=torch.float32,
                              device=values.device)
    mag = torch.bucketize(values.abs(), thresholds).to(torch.uint8)
    sign = (values < 0).to(torch.uint8) << 3
    return mag | sign


def _pack_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    low = codes[:, 0::2]
    high = codes[:, 1::2] << 4
    return (low | high).contiguous()


def _block_error(original: torch.Tensor, codes: torch.Tensor,
                 scales: torch.Tensor, global_scale: float,
                 scale_rule: int) -> torch.Tensor:
    lut = _get_e2m1_lut(original.device)
    reconstructed = (
        lut[codes.long()] * scales.unsqueeze(-1) / global_scale)
    error = reconstructed - original
    if scale_rule == 1:
        return error.square().mean(dim=-1)
    if scale_rule == 2:
        return error.abs().mean(dim=-1)
    if scale_rule == 3:
        return error.abs().amax(dim=-1)
    raise ValueError(f"unsupported scale_rule={scale_rule}")


def _torch_adaptive_4o6_quantize_dense_group(
        weights: List[torch.Tensor],
        scale_rule: int) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]],
                                 torch.Tensor]:
    dense_weights = [
        weight.to(device="cpu", dtype=torch.float32).contiguous()
        for weight in weights
    ]
    rows_per_entry = [tensor.shape[0] for tensor in dense_weights]
    combined = torch.cat(dense_weights, dim=0).contiguous()
    rows, cols = combined.shape
    if cols % 16 != 0:
        raise ValueError(f"last dimension must be divisible by 16, got {cols}")
    sf_cols = cols // 16

    quant_range = (STANDARD_NVFP4_QUANT_RANGE
                   if scale_rule == 0 else ADAPTIVE_FP4_QUANT_RANGE)
    amax = max(float(combined.abs().amax().item()), 1e-12)
    global_scale = quant_range / amax
    weight_scale_2 = torch.tensor(amax / quant_range, dtype=torch.float32)

    blocks = combined.reshape(rows, sf_cols, 16)
    block_amax = blocks.abs().amax(dim=-1)
    scale6_bytes, scale6 = _float_to_e4m3_bytes_and_values(
        block_amax * global_scale / 6.0)

    safe_scale6 = torch.where(scale6 == 0, torch.ones_like(scale6), scale6)
    codes6 = _nearest_e2m1_codes(blocks * global_scale
                                 / safe_scale6.unsqueeze(-1))
    if scale_rule == 0:
        codes = codes6.reshape(rows, cols)
        scale_bytes = scale6_bytes
    else:
        scale4_bytes, scale4 = _float_to_e4m3_bytes_and_values(
            block_amax * global_scale / 4.0)
        safe_scale4 = torch.where(scale4 == 0, torch.ones_like(scale4),
                                  scale4)
        codes4 = _nearest_e2m1_codes(blocks * global_scale
                                     / safe_scale4.unsqueeze(-1))
        err6 = _block_error(blocks, codes6, scale6, global_scale, scale_rule)
        err4 = _block_error(blocks, codes4, scale4, global_scale, scale_rule)
        use4 = err4 < err6
        codes = torch.where(use4.unsqueeze(-1), codes4, codes6).reshape(
            rows, cols)
        scale_bytes = torch.where(use4, scale4_bytes, scale6_bytes)

    quantized = _pack_e2m1_codes(codes)
    split_weights = quantized.split(rows_per_entry, dim=0)
    split_scales = scale_bytes.reshape(rows, sf_cols).split(rows_per_entry,
                                                            dim=0)
    result = []
    for original_weight, new_weight, new_scale in zip(weights, split_weights,
                                                      split_scales):
        result.append((new_weight.contiguous().to(original_weight.device),
                       new_scale.contiguous().to(original_weight.device)))
    return result, weight_scale_2.reshape(())


def import_quant_helpers(trtllm_path: Optional[Path],
                         adaptive_so: Optional[str],
                         quant_backend: str):
    import_runtime_deps()
    if trtllm_path is not None:
        sys.path.insert(0, str(trtllm_path))
    if adaptive_so:
        os.environ["TRTLLM_ADAPTIVE_FP4_SO"] = adaptive_so
    if quant_backend == "torch":
        return (_torch_adaptive_4o6_quantize_dense_group,
                _local_nvfp4_dequantize_linear_sf)
    if quant_backend == "local":
        return (_local_adaptive_4o6_quantize_dense_group,
                _local_nvfp4_dequantize_linear_sf)
    try:
        from tensorrt_llm._torch.modules.fused_moe.quantization import (
            _adaptive_4o6_quantize_dense_group,
            _nvfp4_dequantize_linear_sf,
        )
        return _adaptive_4o6_quantize_dense_group, _nvfp4_dequantize_linear_sf
    except Exception as exc:
        if quant_backend == "trtllm":
            raise
        print(
            "WARNING: falling back to local 4o6 quant helpers because "
            f"TRT-LLM helper import failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return (_local_adaptive_4o6_quantize_dense_group,
                _local_nvfp4_dequantize_linear_sf)


def scale_rule_id(name: str) -> int:
    rules = {"mse": 1, "mae": 2, "abs_max": 3, "absmax": 3}
    if name in rules:
        return rules[name]
    value = int(name)
    if value not in (1, 2, 3):
        raise ValueError("--scale-rule must be mse, mae, abs_max, 1, 2, or 3")
    return value


def unpack_compressed_tensors_int4_nk(
        packed_i32: torch.Tensor,
        weight_shape: torch.Tensor,
        scale: torch.Tensor,
        device: Optional[torch.device] = None) -> torch.Tensor:
    import_runtime_deps()
    out_dim, in_dim = [int(x) for x in weight_shape.reshape(-1).tolist()]
    if device is not None and device.type != "cpu":
        packed_i32 = packed_i32.to(device=device, non_blocking=True)
        scale = scale.to(device=device, non_blocking=True)
    packed_u8 = packed_i32.contiguous().view(torch.uint8).reshape(out_dim, -1)
    # compressed-tensors pack-quantized stores signed INT4 as offset-binary
    # nibbles: pack_to_int32 writes value + 8, not two's-complement bits.
    low = (packed_u8 & 0x0F).to(torch.int16) - 8
    high = ((packed_u8 >> 4) & 0x0F).to(torch.int16) - 8
    q = torch.stack((low, high), dim=-1).reshape(out_dim, -1)[:, :in_dim]
    group_size = in_dim // int(scale.shape[-1])
    dense = q.to(torch.float32) * scale.to(torch.float32).repeat_interleave(
        group_size, dim=1)[:, :in_dim]
    return dense.to(torch.bfloat16)


def stage_for_base(base: str) -> str:
    name = base.rsplit(".", 1)[-1]
    if name in ("down_proj", "w2"):
        return "fc2"
    if name in ("gate_proj", "up_proj", "w1", "w3"):
        return "fc31"
    return "linear"


def input_scale(stage: str, activation_mode: str) -> torch.Tensor:
    import_runtime_deps()
    if activation_mode == "standard":
        value = 1.0 / (448.0 * 6.0)
    elif stage == "fc2":
        value = 1.0 / 16.0
    else:
        value = 1.0 / 1536.0
    return torch.tensor(value, dtype=torch.float32)


EXPERT_BASE_RE = re.compile(
    r"^(?P<prefix>.*\.(?:experts|block_sparse_moe\.experts)\.\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj|w1|w2|w3)$")


# Dense (non-MoE) fused projections. TRT-LLM fuses attention q/k/v into one
# qkv_proj and MLP gate/up into one gate_up_proj, and the fused loader requires
# all members of a fused group to SHARE weight_scale_2. Group them here so they
# are quantized with a single shared global scale (the fused GEMM uses one alpha).
DENSE_FUSE_RE = re.compile(
    r"^(?P<prefix>.*)\.(?P<proj>q_proj|k_proj|v_proj|gate_proj|up_proj)$")


def make_groups(bases: Iterable[str]) -> List[List[str]]:
    buckets: Dict[Tuple[str, str], List[str]] = {}
    singles: List[str] = []
    for base in bases:
        match = EXPERT_BASE_RE.match(base)
        if match:
            proj = match.group("proj")
            if proj in ("gate_proj", "up_proj"):
                buckets.setdefault((match.group("prefix"), "gate_up"), []).append(base)
            elif proj in ("w1", "w3"):
                buckets.setdefault((match.group("prefix"), "w1_w3"), []).append(base)
            else:
                singles.append(base)
            continue
        dmatch = DENSE_FUSE_RE.match(base)
        if dmatch:
            proj = dmatch.group("proj")
            # prefix includes the parent module (...self_attn / ...mlp), so qkv
            # and gate_up groups never collide.
            label = "qkv" if proj in ("q_proj", "k_proj", "v_proj") else "gate_up"
            buckets.setdefault((dmatch.group("prefix"), label), []).append(base)
            continue
        singles.append(base)

    order = {"gate_proj": 0, "up_proj": 1, "w1": 0, "w3": 1,
             "q_proj": 0, "k_proj": 1, "v_proj": 2}
    groups = []
    for values in buckets.values():
        groups.append(sorted(values, key=lambda x: order[x.rsplit(".", 1)[-1]]))
    groups.extend([base] for base in singles)
    return sorted(groups, key=lambda group: group[0])


def build_target_bases(source: SourceInfo, metas: Dict[str, TensorMeta],
                       include_re: re.Pattern,
                       exclude_re: Optional[re.Pattern]) -> List[str]:
    bases = []
    if source.fmt == "int4-compressed-tensors":
        candidates = (k[:-len(".weight_packed")] for k in metas
                      if k.endswith(".weight_packed"))
    else:
        candidates = (k[:-len(".weight")] for k, meta in metas.items()
                      if k.endswith(".weight") and len(meta.shape) == 2)

    for base in candidates:
        weight_key = f"{base}.weight"
        if not include_re.match(weight_key):
            continue
        if exclude_re and exclude_re.match(weight_key):
            continue
        bases.append(base)
    return sorted(set(bases))


def source_aux_keys(source: SourceInfo, base: str) -> List[str]:
    if source.fmt == "int4-compressed-tensors":
        return [
            f"{base}.weight_packed",
            f"{base}.weight_scale",
            f"{base}.weight_shape",
        ]
    if source.fmt == "nvfp4":
        return [
            f"{base}.weight",
            f"{base}.weight_scale",
            f"{base}.weight_scale_2",
            f"{base}.input_scale",
        ]
    return [f"{base}.weight"]


def load_dense_source(source: SourceInfo, store: TensorStore, base: str,
                      dequant_nvfp4, int4_unpack_device: str) -> torch.Tensor:
    import_runtime_deps()
    if source.fmt == "bf16":
        return store.get(f"{base}.weight")
    if source.fmt == "nvfp4":
        return dequant_nvfp4(
            store.get(f"{base}.weight"),
            store.get(f"{base}.weight_scale"),
            store.get(f"{base}.weight_scale_2"),
            torch.device(f"cuda:{torch.cuda.current_device()}"),
        ).cpu()
    if source.fmt == "int4-compressed-tensors":
        device = None
        if int4_unpack_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("--int4-unpack-device=cuda requires CUDA.")
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        return unpack_compressed_tensors_int4_nk(
            store.get(f"{base}.weight_packed"),
            store.get(f"{base}.weight_shape"),
            store.get(f"{base}.weight_scale"),
            device=device,
        )
    raise AssertionError(source.fmt)


def existing_source_excludes(source: SourceInfo) -> List[str]:
    excludes: List[str] = []
    if source.hf_quant_config:
        q = source.hf_quant_config.get("quantization", source.hf_quant_config)
        excludes.extend(q.get("exclude_modules") or [])
    for qcfg in config_quantization_configs(source.config):
        excludes.extend(qcfg.get("ignore") or [])
        excludes.extend(qcfg.get("modules_to_not_convert") or [])
    return excludes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-format",
                        choices=["auto", "bf16", "nvfp4",
                                 "int4-compressed-tensors"],
                        default="auto")
    parser.add_argument("--trtllm-path", type=Path,
                        help="Optional local TensorRT-LLM source path to import.")
    parser.add_argument("--adaptive-fp4-so",
                        help="Optional path for TRTLLM_ADAPTIVE_FP4_SO.")
    parser.add_argument("--quant-backend",
                        choices=["auto", "trtllm", "local", "torch"],
                        default="auto",
                        help="Use TRT-LLM CUDA helpers, local CUDA helpers, "
                             "or a portable PyTorch packer.")
    parser.add_argument("--include-regex", default=DEFAULT_MOE_WEIGHT_RE)
    parser.add_argument("--exclude-regex")
    parser.add_argument("--all-linear", action="store_true",
                        help="Convert every 2D *.weight tensor instead of only common MoE experts.")
    parser.add_argument("--activation-mode", choices=["4o6", "standard"],
                        default="4o6")
    parser.add_argument("--scale-rule", default="mse")
    parser.add_argument("--max-shard-size", default="2GB")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-untargeted-copy", action="store_true",
                        help="Skip copying non-target tensors + skip "
                             "unconverted_packed check + skip writing "
                             "config.json/hf_quant_config.json. For parallel "
                             "worker mode; the finalize tool handles those.")
    parser.add_argument("--shard-prefix", default="model",
                        help="Prefix for output shard filenames "
                             "(e.g. 'w0' -> w0-00001.safetensors). Useful "
                             "for parallel workers writing into a shared dir.")
    parser.add_argument("--tensor-cache-size", type=int, default=128,
                        help="Number of source safetensors shards to keep open "
                             "per process. Kimi-K2.5 has 64 shards, so the "
                             "default avoids reopening the same NFS files tens "
                             "of thousands of times. Use 0 to disable.")
    parser.add_argument("--int4-unpack-device",
                        choices=["cpu", "cuda"],
                        default="cpu",
                        help="Device used to unpack INT4 compressed-tensors "
                             "weights before NVFP4 quantization. Default keeps "
                             "the original CPU behavior; cuda can reduce CPU "
                             "bottlenecks on B200/H100-class nodes.")
    args = parser.parse_args()

    src = args.input.resolve()
    dst = args.output.resolve()
    source = detect_source(src, args.source_format)
    include = re.compile(r".*\.weight$" if args.all_linear else args.include_regex)
    exclude = re.compile(args.exclude_regex) if args.exclude_regex else None

    try:
        weight_map, metas = discover_weight_map(src)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from None
    if args.source_format == "auto" and source.fmt == "bf16":
        packed_count = sum(k.endswith(".weight_packed") for k in metas)
        if packed_count:
            print(
                "WARNING: config has no quantization_config, but checkpoint "
                f"contains {packed_count} .weight_packed tensors; treating "
                "source as int4-compressed-tensors.",
                file=sys.stderr,
            )
            source = SourceInfo("int4-compressed-tensors", source.config,
                                source.hf_quant_config)
    target_bases = build_target_bases(source, metas, include, exclude)
    if not target_bases:
        raise RuntimeError(
            "No target weights matched. Use --all-linear or --include-regex "
            "if this is not a MoE expert checkpoint.")

    consumed = set()
    for base in target_bases:
        consumed.update(k for k in source_aux_keys(source, base) if k in metas)

    if source.fmt == "int4-compressed-tensors" and not args.skip_untargeted_copy:
        unconverted_packed = [
            k for k in metas if k.endswith(".weight_packed") and k not in consumed
        ]
        if unconverted_packed:
            raise RuntimeError(
                "Some INT4 packed weights would remain in the output. Adjust "
                f"--include-regex/--exclude-regex. Example: {unconverted_packed[0]}")

    groups = make_groups(target_bases)
    print(f"source_format={source.fmt}")
    print(f"input={src}")
    print(f"output={dst}")
    print(f"target_bases={len(target_bases)} groups={len(groups)}")
    if args.dry_run:
        for group in groups[:20]:
            print("GROUP", group)
        if len(groups) > 20:
            print(f"... {len(groups) - 20} more groups")
        return

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    copy_sidecar_files(src, dst)

    import_runtime_deps()
    quantize_dense_group, dequant_nvfp4 = import_quant_helpers(
        args.trtllm_path, args.adaptive_fp4_so, args.quant_backend)
    rule = scale_rule_id(args.scale_rule)
    store = TensorStore(src, weight_map, cache_size=args.tensor_cache_size)
    writer = ShardWriter(dst, parse_size(args.max_shard_size),
                         shard_prefix=args.shard_prefix)
    start_time = time.monotonic()
    try:
        output_keys = set()
        for group_idx, group in enumerate(groups, start=1):
            dense = [load_dense_source(source, store, base, dequant_nvfp4,
                                       args.int4_unpack_device)
                     for base in group]
            quantized, scale2 = quantize_dense_group(dense, rule)
            stage = stage_for_base(group[0])
            in_scale = input_scale(stage, args.activation_mode)
            for base, (qweight, block_scale) in zip(group, quantized):
                outputs = {
                    f"{base}.weight": qweight,
                    f"{base}.weight_scale": block_scale,
                    f"{base}.weight_scale_2": scale2,
                    f"{base}.input_scale": in_scale,
                }
                for key, tensor in outputs.items():
                    writer.add(key, tensor)
                    output_keys.add(key)
            del dense, quantized, scale2
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if group_idx % 100 == 0:
                elapsed = time.monotonic() - start_time
                rate = group_idx / elapsed if elapsed > 0 else 0.0
                print(
                    f"converted_groups={group_idx}/{len(groups)} "
                    f"elapsed_s={elapsed:.1f} groups_per_s={rate:.2f}",
                    flush=True)

        if not args.skip_untargeted_copy:
            exclude_modules = existing_source_excludes(source)
            for key in sorted(weight_map):
                if key in consumed or key in output_keys:
                    continue
                tensor = store.get(key)
                writer.add(key, tensor)
                if (key.endswith(".weight") and len(metas[key].shape) == 2
                        and metas[key].dtype in FLOAT_DTYPES):
                    exclude_modules.append(key[:-len(".weight")])

            writer.finalize()
            write_quant_configs(dst, source, exclude_modules)
        else:
            writer.finalize()
    finally:
        store.close()
    elapsed = time.monotonic() - start_time
    print(f"wrote {len(writer.weight_map)} tensors to {writer.shard_idx} shards")
    print(f"elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()
