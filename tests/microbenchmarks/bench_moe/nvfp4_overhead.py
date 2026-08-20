# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 strategy controls for the Kimi 4o6/SVDQuant microbenchmark."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping, TypeAlias, assert_never

import torch

from tensorrt_llm._torch.modules.fused_moe import svdquant_helpers
from nvfp4_runtime_contract import (
    Nvfp4Strategy as Nvfp4Strategy,
    Nvfp4StrategyError as Nvfp4StrategyError,
    validate_runtime_quantization_contract as validate_runtime_quantization_contract,
)


@dataclass(frozen=True, slots=True)
class TokenCase:
    """One local-token M value used by the single-rank EP4-shard benchmark."""

    name: str
    num_tokens: int


@dataclass(frozen=True, slots=True)
class GemmCase:
    """One grouped-MoE projection shape in the performance contract.

    ``total_tokens`` is the model-level batch/sequence M.  ``expert_m_*`` is
    the actual per-expert M after top-k expansion and balanced routing; this is
    the M consumed by each member of the grouped GEMM.
    """

    name: str
    phase: str
    projection: str
    total_tokens: int
    active_experts: int
    expert_m_min: int
    expert_m_max: int
    n: int
    k: int


KIMI_TOKEN_CASES: tuple[TokenCase, ...] = (
    TokenCase(name="decode_m1", num_tokens=1),
    TokenCase(name="decode_m4", num_tokens=4),
    TokenCase(name="prefill_m1536", num_tokens=1536),
    TokenCase(name="prefill_m2048", num_tokens=2048),
)

# Kimi-K2.5 EP=4 local shard: 96 experts, top-k 8, H=7168, I=2048.
# Forced balanced routing makes the per-expert M deterministic.
KIMI_GEMM_CASES: tuple[GemmCase, ...] = tuple(
    GemmCase(
        name=f"{case_name}_{projection.lower()}",
        phase=phase,
        projection=projection,
        total_tokens=total_tokens,
        active_experts=active_experts,
        expert_m_min=expert_m_min,
        expert_m_max=expert_m_max,
        n=n,
        k=k,
    )
    for (
        case_name,
        phase,
        total_tokens,
        active_experts,
        expert_m_min,
        expert_m_max,
    ) in (
        ("decode_m1", "decode", 1, 8, 1, 1),
        ("decode_m4", "decode", 4, 32, 1, 1),
        ("prefill_m1536", "prefill", 1536, 96, 128, 128),
        ("prefill_m2048", "prefill", 2048, 96, 170, 171),
    )
    for projection, n, k in (
        ("FC13", 4096, 7168),
        ("FC2", 7168, 2048),
    )
)

_MANAGED_ENV_NAMES = (
    "TRTLLM_MOE_FORCE_CUTEDSL",
    "TRTLLM_NVFP4_RUNTIME_ACTIVATION",
    "TRTLLM_ADAPTIVE_FP4",
    "TRTLLM_ADAPTIVE_FP4_FC2",
    "TRTLLM_ADAPTIVE_FP4_WEIGHT",
    "TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31",
    "TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13",
    "TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2",
    "TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE",
    "TRTLLM_ADAPTIVE_FP4_WEIGHT_FALLBACK_SCALE_RULE",
    *svdquant_helpers.ENV_NAMES,
)
EnvironmentSnapshot: TypeAlias = tuple[tuple[str, str | None], ...]


def optional_strategy(value: str | None) -> Nvfp4Strategy | None:
    """Parse an explicitly requested strategy without changing generic runs."""
    return Nvfp4Strategy(value) if value is not None else None


def strategy_environment(strategy: Nvfp4Strategy) -> Mapping[str, str]:
    """Return a complete environment for one isolated benchmark strategy."""
    match strategy:
        case Nvfp4Strategy.NATIVE:
            activation_mode = "standard"
            adaptive_weight = False
            svdquant = False
        case Nvfp4Strategy.FOUR_O_SIX:
            activation_mode = "4o6"
            adaptive_weight = True
            svdquant = False
        case Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64:
            activation_mode = "4o6"
            adaptive_weight = True
            svdquant = True
        case unreachable:
            assert_never(unreachable)
    adaptive_weight_flag = "1" if adaptive_weight else "0"
    svdquant_flag = "1" if svdquant else "0"
    return {
        "TRTLLM_MOE_FORCE_CUTEDSL": "1",
        "TRTLLM_NVFP4_RUNTIME_ACTIVATION": activation_mode,
        "TRTLLM_ADAPTIVE_FP4": "0",
        "TRTLLM_ADAPTIVE_FP4_FC2": "0",
        "TRTLLM_ADAPTIVE_FP4_WEIGHT": adaptive_weight_flag,
        "TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31": adaptive_weight_flag,
        "TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13": adaptive_weight_flag,
        "TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2": adaptive_weight_flag,
        "TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE": "mse",
        "TRTLLM_ADAPTIVE_FP4_WEIGHT_FALLBACK_SCALE_RULE": "standard",
        svdquant_helpers.ENV_ENABLED: svdquant_flag,
        svdquant_helpers.ENV_RANK: "64",
        svdquant_helpers.ENV_US_DTYPE: "bf16",
        svdquant_helpers.ENV_DEVICE: "cuda",
        svdquant_helpers.ENV_FC13: svdquant_flag,
        svdquant_helpers.ENV_FC2: svdquant_flag,
    }


def validate_strategy(
    strategy: Nvfp4Strategy,
    *,
    quant_algo: str | None,
    backend: str,
    cuda_graph: bool,
) -> None:
    """Reject configurations that cannot isolate the requested overhead."""
    if quant_algo != "NVFP4":
        raise Nvfp4StrategyError(
            f"{strategy.value} requires NVFP4, got {quant_algo!r}"
        )
    if backend.upper() != "CUTEDSL":
        raise Nvfp4StrategyError(
            f"{strategy.value} requires the CUTEDSL backend"
        )
    match strategy:
        case Nvfp4Strategy.NATIVE | Nvfp4Strategy.FOUR_O_SIX:
            return
        case Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64:
            if cuda_graph:
                raise Nvfp4StrategyError(
                    "SVDQuant does not support CUDA Graph capture")
        case unreachable:
            assert_never(unreachable)


def validate_strategy_if_requested(
    strategy: Nvfp4Strategy | None,
    *,
    quant_algo: str | None,
    backend: str,
    cuda_graph: bool,
) -> None:
    """Validate only an explicit overhead experiment, never a generic case."""
    if strategy is None:
        return
    validate_strategy(
        strategy,
        quant_algo=quant_algo,
        backend=backend,
        cuda_graph=cuda_graph,
    )


@contextmanager
def strategy_environment_scope(strategy: Nvfp4Strategy) -> Iterator[None]:
    """Apply and restore all feature flags around one benchmark candidate."""
    previous = apply_strategy_environment(strategy)
    try:
        yield
    finally:
        restore_environment(previous)


def apply_strategy_environment(
    strategy: Nvfp4Strategy,
) -> EnvironmentSnapshot:
    """Apply one strategy and return the exact environment state to restore."""
    previous = tuple((name, os.environ.get(name)) for name in _MANAGED_ENV_NAMES)
    os.environ.update(strategy_environment(strategy))
    return previous


def apply_strategy_environment_if_requested(
    strategy: Nvfp4Strategy | None,
) -> EnvironmentSnapshot:
    """Apply feature flags only for an explicit NVFP4 overhead experiment."""
    if strategy is None:
        return ()
    return apply_strategy_environment(strategy)


def restore_environment(snapshot: EnvironmentSnapshot) -> None:
    """Restore an environment snapshot produced by this module."""
    for name, value in snapshot:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


@contextmanager
def synthetic_svdquant_load_scope(
    strategy: Nvfp4Strategy | None,
) -> Iterator[None]:
    """Disable correction loading while synthetic residual weights are installed."""
    if strategy is not Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64:
        yield
        return
    previous = tuple((name, os.environ.get(name)) for name in svdquant_helpers.ENV_NAMES)
    os.environ[svdquant_helpers.ENV_ENABLED] = "0"
    os.environ[svdquant_helpers.ENV_FC13] = "0"
    os.environ[svdquant_helpers.ENV_FC2] = "0"
    try:
        yield
    finally:
        for name, value in previous:
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def initialize_synthetic_svdquant_factors(
    module: torch.nn.Module,
    strategy: Nvfp4Strategy | None,
) -> bool:
    """Zero existing rank-64 factors so runtime overhead is measured without SVD."""
    if strategy is not Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64:
        return False
    factors = tuple(
        getattr(module, f"{projection}_{factor}")
        for projection in ("w1", "w3", "w2")
        for factor in ("us", "vh")
    )
    with torch.no_grad():
        for factor in factors:
            factor.zero_()
    # The real checkpoint path packs the two FC13 Vh factors during
    # ``post_load_weights`` finalization.  This synthetic fixture deliberately
    # bypasses that load, so mirror only the same one-time finalization step;
    # otherwise the benchmark silently measures the old two-dispatch FC13
    # path instead of the production dual-low-rank path.
    quant_method = getattr(module, "quant_method", None)
    pack_fc13_vh = getattr(quant_method, "_pack_svdquant_fc13_vh", None)
    if pack_fc13_vh is not None:
        pack_fc13_vh(module)
    module._svdquant_loaded = True
    return True
