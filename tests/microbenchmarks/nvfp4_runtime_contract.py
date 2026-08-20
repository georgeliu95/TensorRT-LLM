# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measured-kernel contract shared by the NVFP4 runner and summarizer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, assert_never


class Nvfp4Strategy(str, Enum):
    """Mutually exclusive NVFP4 runtime strategies measured by the benchmark."""

    NATIVE = "native"
    FOUR_O_SIX = "4o6"
    FOUR_O_SIX_SVDQ_R64 = "4o6_svdq_r64"


class Nvfp4StrategyError(ValueError):
    """An NVFP4 benchmark strategy is incompatible with its runtime config."""


@dataclass(frozen=True, slots=True)
class Nvfp4RuntimeContractError(RuntimeError):
    """A measured kernel set does not implement its declared strategy."""

    strategy: Nvfp4Strategy
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.strategy.value} runtime activation quantization contract failed: "
            f"{self.detail}"
        )


def validate_runtime_quantization_contract(
    strategy: Nvfp4Strategy,
    kernel_names: Iterable[str],
) -> None:
    """Reject measured kernels that do not implement the declared strategy."""
    normalized = tuple(name.lower() for name in kernel_names)
    if any("dequant_nvfp4" in name for name in normalized):
        raise Nvfp4RuntimeContractError(
            strategy=strategy,
            detail="unexpected activation dequantization in the measured forward",
        )
    if any("quantize_with_block_size" in name for name in normalized):
        raise Nvfp4RuntimeContractError(
            strategy=strategy,
            detail="legacy static activation quantization in the measured forward",
        )
    match strategy:
        case Nvfp4Strategy.NATIVE:
            has_fc13_runtime_quant = any(
                "fused_prologue_quantize_v1" in name
                or (
                    "fused_prologue_quantize_v2" in name
                    and "fused_moe_prologue" not in name
                    and "adaptivescalerule)0" in name
                )
                for name in normalized
            )
            has_fc2_runtime_quant = any(
                (
                    "fused_moe_prologue_quantize_v2" in name
                    or "phase2_only_quantize" in name
                )
                and "adaptivescalerule)0" in name
                for name in normalized
            )
            if not (has_fc13_runtime_quant and has_fc2_runtime_quant):
                raise Nvfp4RuntimeContractError(
                    strategy=strategy,
                    detail="FC13 and FC2 must both use runtime activation quantization",
                )
        case Nvfp4Strategy.FOUR_O_SIX:
            has_fc13_runtime_quant = any(
                "fused_prologue_quantize_v2" in name
                and "fused_moe_prologue" not in name
                and "adaptivescalerule)1" in name
                for name in normalized
            )
            has_fc2_runtime_quant = any(
                (
                    "fused_moe_prologue_quantize_v2" in name
                    or "phase2_only_quantize" in name
                )
                and "adaptivescalerule)1" in name
                for name in normalized
            )
            if not (has_fc13_runtime_quant and has_fc2_runtime_quant):
                raise Nvfp4RuntimeContractError(
                    strategy=strategy,
                    detail="FC13 and FC2 must both use adaptive rule 1",
                )
        case Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64:
            has_fc13_runtime_quant = any(
                "fused_prologue_quantize_v2" in name
                and "fused_moe_prologue" not in name
                and "adaptivescalerule)1" in name
                for name in normalized
            )
            has_fc2_runtime_quant = any(
                (
                    "fused_moe_prologue_quantize_v2" in name
                    or "fused_moe_swiglu_prologue_quantize_v2" in name
                )
                and "adaptivescalerule)1" in name
                for name in normalized
            )
            if not (has_fc13_runtime_quant and has_fc2_runtime_quant):
                raise Nvfp4RuntimeContractError(
                    strategy=strategy,
                    detail="FC13 and FC2 must both use adaptive rule 1",
                )
        case unreachable:
            assert_never(unreachable)
