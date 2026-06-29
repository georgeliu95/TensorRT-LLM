# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure configuration and tensor math for NVFP4 MoE SVDQuant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

import torch

ENV_ENABLED: Final = "TRTLLM_SVDQUANT_NVFP4"
ENV_RANK: Final = "TRTLLM_SVDQUANT_RANK"
ENV_US_DTYPE: Final = "TRTLLM_SVDQUANT_US_DTYPE"
ENV_DEVICE: Final = "TRTLLM_SVDQUANT_DEVICE"
ENV_FC13: Final = "TRTLLM_SVDQUANT_FC13"
ENV_FC2: Final = "TRTLLM_SVDQUANT_FC2"
ENV_NAMES: Final = (
    ENV_ENABLED,
    ENV_RANK,
    ENV_US_DTYPE,
    ENV_DEVICE,
    ENV_FC13,
    ENV_FC2,
)

ProjectionMode = Literal["column", "row"]
SvdDevice = Literal["cpu", "cuda"]

_TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final = frozenset({"0", "false", "no", "off", ""})
_DTYPES: Final = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class SvdquantConfigError(ValueError):
    """An SVDQuant environment option is malformed."""


class SvdquantLoadError(RuntimeError):
    """The requested checkpoint or MoE lifecycle cannot host SVDQuant."""


@dataclass(frozen=True, slots=True)
class SvdquantConfig:
    """Resolved immutable configuration for one NVFP4 MoE instance."""

    enabled: bool
    rank: int
    us_dtype: torch.dtype
    device: SvdDevice
    fc13: bool
    fc2: bool

    @property
    def any_stage(self) -> bool:
        """Return whether at least one low-rank correction stage is active."""
        return self.enabled and (self.fc13 or self.fc2)


def _parse_flag(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    value = raw_value.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise SvdquantConfigError(
        f"{name} must be a boolean (0/1, false/true, no/yes, off/on); "
        f"got {raw_value!r}.")


def load_config() -> SvdquantConfig:
    """Parse all public environment variables at the configuration boundary."""
    enabled = _parse_flag(ENV_ENABLED, False)
    rank_raw = os.environ.get(ENV_RANK, "64").strip()
    try:
        rank = int(rank_raw)
    except ValueError as error:
        raise SvdquantConfigError(
            f"{ENV_RANK} must be a positive integer; got {rank_raw!r}.") from error
    if rank <= 0:
        raise SvdquantConfigError(f"{ENV_RANK} must be > 0; got {rank}.")

    dtype_name = os.environ.get(ENV_US_DTYPE, "bf16").strip().lower()
    if dtype_name not in _DTYPES:
        raise SvdquantConfigError(
            f"{ENV_US_DTYPE} must be one of {sorted(_DTYPES)}; "
            f"got {dtype_name!r}.")

    device = os.environ.get(ENV_DEVICE, "cuda").strip().lower()
    if device not in ("cpu", "cuda"):
        raise SvdquantConfigError(
            f"{ENV_DEVICE} must be cpu or cuda; got {device!r}.")

    return SvdquantConfig(
        enabled=enabled,
        rank=rank,
        us_dtype=_DTYPES[dtype_name],
        device=device,
        fc13=enabled and _parse_flag(ENV_FC13, enabled),
        fc2=enabled and _parse_flag(ENV_FC2, enabled),
    )


def _resolve_svd_device(weight: torch.Tensor,
                        preference: SvdDevice) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return weight.device


def _validate_decomposition(weight: torch.Tensor, rank: int,
                            expected_dims: int) -> None:
    if weight.dim() != expected_dims:
        raise SvdquantLoadError(
            f"SVDQuant expects a {expected_dims}-D weight; "
            f"got shape {tuple(weight.shape)}.")
    max_rank = min(weight.shape[-2:])
    if rank <= 0 or rank > max_rank:
        raise SvdquantLoadError(
            f"SVDQuant rank must be in (0, {max_rank}] for shape "
            f"{tuple(weight.shape[-2:])}; got {rank}.")


def decompose_per_tensor(
    weight: torch.Tensor,
    rank: int,
    us_dtype: torch.dtype = torch.bfloat16,
    device: SvdDevice = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(US, Vh, residual)`` for one ``(out, in)`` tensor."""
    _validate_decomposition(weight, rank, 2)
    source_device = weight.device
    source_dtype = weight.dtype
    weight_fp32 = weight.detach().to(
        device=_resolve_svd_device(weight, device), dtype=torch.float32)
    u, singular_values, vh = torch.linalg.svd(weight_fp32,
                                               full_matrices=False)
    us_fp32 = (u[:, :rank] * singular_values[:rank]).contiguous()
    vh_fp32 = vh[:rank].contiguous()
    residual = (weight_fp32 - us_fp32 @ vh_fp32).to(
        device=source_device, dtype=source_dtype)
    us = us_fp32.to(device=source_device, dtype=us_dtype).contiguous()
    vh_result = vh_fp32.to(device=source_device,
                           dtype=us_dtype).contiguous()
    return us, vh_result, residual


def decompose_batched(
    weights: torch.Tensor,
    rank: int,
    us_dtype: torch.dtype = torch.bfloat16,
    device: SvdDevice = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batched per-expert ``(US, Vh, residual)`` tensors."""
    _validate_decomposition(weights, rank, 3)
    source_device = weights.device
    source_dtype = weights.dtype
    weights_fp32 = weights.detach().to(
        device=_resolve_svd_device(weights, device), dtype=torch.float32)
    u, singular_values, vh = torch.linalg.svd(weights_fp32,
                                               full_matrices=False)
    us_fp32 = (u[:, :, :rank]
               * singular_values[:, None, :rank]).contiguous()
    vh_fp32 = vh[:, :rank, :].contiguous()
    residual = (weights_fp32 - us_fp32 @ vh_fp32).to(
        device=source_device, dtype=source_dtype)
    us = us_fp32.to(device=source_device, dtype=us_dtype).contiguous()
    vh_result = vh_fp32.to(device=source_device,
                           dtype=us_dtype).contiguous()
    return us, vh_result, residual


def reconstruct_lowrank(us: torch.Tensor, vh: torch.Tensor) -> torch.Tensor:
    """Materialize ``US @ Vh`` in FP32 for diagnostics and tests."""
    return us.float() @ vh.float()


def lowrank_gemm(x: torch.Tensor, us: torch.Tensor,
                 vh: torch.Tensor) -> torch.Tensor:
    """Evaluate the runtime formula ``(x @ vh.T) @ us.T`` in ``x.dtype``."""
    local_x = x[..., :vh.shape[-1]]
    return ((local_x @ vh.to(dtype=x.dtype).T)
            @ us.to(dtype=x.dtype).T)


def _shard_bounds(width: int, tp_size: int, tp_rank: int) -> tuple[int, int]:
    if tp_size <= 0 or tp_rank < 0 or tp_rank >= tp_size:
        raise SvdquantLoadError(
            f"Invalid tensor-parallel coordinate rank={tp_rank}, size={tp_size}.")
    shard_width = (width + tp_size - 1) // tp_size
    start = tp_rank * shard_width
    return start, min(start + shard_width, width)


def shard_lowrank_factors(
    us: torch.Tensor,
    vh: torch.Tensor,
    *,
    tp_size: int,
    tp_rank: int,
    projection: ProjectionMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign global factors to the same TP shard consumed by an MoE weight."""
    if us.dim() != 2 or vh.dim() != 2 or us.shape[1] != vh.shape[0]:
        raise SvdquantLoadError(
            f"Incompatible low-rank factor shapes: US={tuple(us.shape)}, "
            f"Vh={tuple(vh.shape)}.")
    if projection == "column":
        start, end = _shard_bounds(us.shape[0], tp_size, tp_rank)
        return us[start:end].contiguous(), vh.contiguous()
    if projection == "row":
        start, end = _shard_bounds(vh.shape[1], tp_size, tp_rank)
        return us.contiguous(), vh[:, start:end].contiguous()
    raise SvdquantLoadError(f"Unknown TP projection mode {projection!r}.")


def lowrank_storage_shape(
    num_experts: int,
    out_features: int,
    in_features: int,
    rank: int,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the local ``(E,out,r)`` and ``(E,r,in)`` storage shapes."""
    return ((num_experts, out_features, rank),
            (num_experts, rank, in_features))
