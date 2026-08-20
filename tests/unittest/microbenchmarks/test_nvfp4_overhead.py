# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit coverage for the Kimi NVFP4 overhead microbenchmark contract."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch

_MICROBENCHMARKS = Path(__file__).parents[2] / "microbenchmarks"
_REPO_ROOT = Path(__file__).parents[3]
_BENCH_MOE_DIR = _MICROBENCHMARKS / "bench_moe"
_BENCH_MOE_PACKAGE = ModuleType("bench_moe")
_BENCH_MOE_PACKAGE.__path__ = [str(_BENCH_MOE_DIR)]


def _load_module(module_name: str, path: Path) -> ModuleType:
    """Load one source file without executing its package entrypoint."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load benchmark module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SVDQUANT_HELPERS = _load_module(
    "_test_svdquant_helpers",
    _REPO_ROOT
    / "tensorrt_llm"
    / "_torch"
    / "modules"
    / "fused_moe"
    / "svdquant_helpers.py",
)
_FUSED_MOE = ModuleType("tensorrt_llm._torch.modules.fused_moe")
_FUSED_MOE.svdquant_helpers = _SVDQUANT_HELPERS
with patch.dict(
    sys.modules,
    {
        "bench_moe": _BENCH_MOE_PACKAGE,
        "tensorrt_llm": ModuleType("tensorrt_llm"),
        "tensorrt_llm._torch": ModuleType("tensorrt_llm._torch"),
        "tensorrt_llm._torch.modules": ModuleType("tensorrt_llm._torch.modules"),
        "tensorrt_llm._torch.modules.fused_moe": _FUSED_MOE,
    },
):
    _load_module(
        "nvfp4_runtime_contract",
        _MICROBENCHMARKS / "nvfp4_runtime_contract.py",
    )
    _NVFP4_OVERHEAD = _load_module(
        "bench_moe.nvfp4_overhead",
        _BENCH_MOE_DIR / "nvfp4_overhead.py",
    )
with patch.dict(sys.modules, {"bench_moe": _BENCH_MOE_PACKAGE}):
    _load_module("bench_moe.backend", _BENCH_MOE_DIR / "backend.py")
    _SPECS = _load_module("bench_moe.specs", _BENCH_MOE_DIR / "specs.py")

KIMI_TOKEN_CASES = _NVFP4_OVERHEAD.KIMI_TOKEN_CASES
KIMI_GEMM_CASES = _NVFP4_OVERHEAD.KIMI_GEMM_CASES
Nvfp4Strategy = _NVFP4_OVERHEAD.Nvfp4Strategy
initialize_synthetic_svdquant_factors = (
    _NVFP4_OVERHEAD.initialize_synthetic_svdquant_factors
)
apply_strategy_environment_if_requested = (
    _NVFP4_OVERHEAD.apply_strategy_environment_if_requested
)
optional_strategy = _NVFP4_OVERHEAD.optional_strategy
strategy_environment = _NVFP4_OVERHEAD.strategy_environment
strategy_environment_scope = _NVFP4_OVERHEAD.strategy_environment_scope
synthetic_svdquant_load_scope = _NVFP4_OVERHEAD.synthetic_svdquant_load_scope
validate_strategy = _NVFP4_OVERHEAD.validate_strategy
validate_strategy_if_requested = (
    _NVFP4_OVERHEAD.validate_strategy_if_requested
)
KIMI_K2_EP4_SHARD = _SPECS.KIMI_K2_EP4_SHARD
ConfigSpec = _SPECS.ConfigSpec


def test_nvfp4_strategy_is_opt_in_for_generic_bench_moe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given an ordinary bench_moe configuration with no overhead experiment.
    config = ConfigSpec(backend="TRTLLM", parallel_mode="DEP")
    monkeypatch.setenv("TRTLLM_MOE_FORCE_CUTEDSL", "caller-value")

    # Then it requests no NVFP4 strategy, accepts FP8, and changes no feature env.
    assert config.nvfp4_strategy is None
    strategy = optional_strategy(config.nvfp4_strategy)
    assert strategy is None
    validate_strategy_if_requested(
        strategy,
        quant_algo="FP8",
        backend=config.backend,
        cuda_graph=config.cuda_graph,
    )
    assert apply_strategy_environment_if_requested(None) == ()
    assert os.environ["TRTLLM_MOE_FORCE_CUTEDSL"] == "caller-value"


def test_kimi_ep4_shard_matches_profiled_projection_shapes() -> None:
    # Given the Kimi-K2.5 EP=4 local-shard benchmark profile.
    model = KIMI_K2_EP4_SHARD

    # When its routed-expert dimensions are expanded into FC13 and FC2 GEMMs.
    fc13 = (model.intermediate_size * 2, model.hidden_size)
    fc2 = (model.hidden_size, model.intermediate_size)

    # Then they match the M-dependent shapes observed in the OCI-HSG trace.
    assert model.num_experts == 96
    assert model.top_k == 8
    assert fc13 == (4096, 7168)
    assert fc2 == (7168, 2048)


def test_kimi_token_cases_cover_decode_and_prefill_boundaries() -> None:
    # Given the agreed unit-test scope, when the named cases are inspected.
    observed = {case.name: case.num_tokens for case in KIMI_TOKEN_CASES}

    # Then both decode batch sizes and both prefill lengths are present.
    assert observed == {
        "decode_m1": 1,
        "decode_m4": 4,
        "prefill_m1536": 1536,
        "prefill_m2048": 2048,
    }


def test_kimi_gemm_cases_are_the_real_grouped_gemm_shapes() -> None:
    # Given forced balanced top-k=8 routing over the 96-expert local shard.
    observed = {
        case.name: (
            case.phase,
            case.total_tokens,
            case.active_experts,
            case.expert_m_min,
            case.expert_m_max,
            case.n,
            case.k,
        )
        for case in KIMI_GEMM_CASES
    }

    # Then the contract records per-expert grouped-GEMM M, not just prompt M.
    assert observed == {
        "decode_m1_fc13": ("decode", 1, 8, 1, 1, 4096, 7168),
        "decode_m1_fc2": ("decode", 1, 8, 1, 1, 7168, 2048),
        "decode_m4_fc13": ("decode", 4, 32, 1, 1, 4096, 7168),
        "decode_m4_fc2": ("decode", 4, 32, 1, 1, 7168, 2048),
        "prefill_m1536_fc13": ("prefill", 1536, 96, 128, 128, 4096, 7168),
        "prefill_m1536_fc2": ("prefill", 1536, 96, 128, 128, 7168, 2048),
        "prefill_m2048_fc13": ("prefill", 2048, 96, 170, 171, 4096, 7168),
        "prefill_m2048_fc2": ("prefill", 2048, 96, 170, 171, 7168, 2048),
    }


@pytest.mark.parametrize(
    ("strategy", "activation_mode", "adaptive_weight", "svdquant"),
    [
        (Nvfp4Strategy.NATIVE, "standard", "0", "0"),
        (Nvfp4Strategy.FOUR_O_SIX, "4o6", "1", "0"),
        (Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64, "4o6", "1", "1"),
    ],
)
def test_strategy_environment_is_complete_and_isolated(
    strategy: Nvfp4Strategy,
    activation_mode: str,
    adaptive_weight: str,
    svdquant: str,
) -> None:
    # Given one benchmark strategy, when its fresh-process environment is built.
    environment = strategy_environment(strategy)

    # Then activation, weight, and correction switches describe only that strategy.
    assert environment["TRTLLM_NVFP4_RUNTIME_ACTIVATION"] == activation_mode
    assert environment["TRTLLM_ADAPTIVE_FP4_WEIGHT"] == adaptive_weight
    assert environment["TRTLLM_MOE_FORCE_CUTEDSL"] == "1"
    assert environment["TRTLLM_SVDQUANT_NVFP4"] == svdquant
    assert environment["TRTLLM_SVDQUANT_FC13"] == svdquant
    assert environment["TRTLLM_SVDQUANT_FC2"] == svdquant
    assert environment["TRTLLM_SVDQUANT_RANK"] == "64"


@pytest.mark.parametrize(
    ("strategy", "quant_algo", "backend", "cuda_graph", "message"),
    [
        (Nvfp4Strategy.FOUR_O_SIX, "FP8", "CUTEDSL", False, "NVFP4"),
        (Nvfp4Strategy.FOUR_O_SIX, "NVFP4", "TRTLLM", False, "CUTEDSL"),
        (
            Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64,
            "NVFP4",
            "CUTEDSL",
            True,
            "CUDA Graph",
        ),
    ],
)
def test_strategy_validation_rejects_incomparable_cases(
    strategy: Nvfp4Strategy,
    quant_algo: str,
    backend: str,
    cuda_graph: bool,
    message: str,
) -> None:
    # Given an incompatible benchmark configuration, when it crosses the boundary.
    with pytest.raises(ValueError, match=message):
        validate_strategy(
            strategy,
            quant_algo=quant_algo,
            backend=backend,
            cuda_graph=cuda_graph,
        )


def test_native_strategy_requires_same_cutedsl_backend() -> None:
    # Given the runtime-standard NVFP4 control on a different backend.
    with pytest.raises(ValueError, match="CUTEDSL"):
        validate_strategy(
            Nvfp4Strategy.NATIVE,
            quant_algo="NVFP4",
            backend="TRTLLM",
            cuda_graph=True,
        )

    # Then backend tactic differences cannot contaminate the activation comparison.


def test_strategy_scope_restores_preexisting_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a caller-owned feature value before a benchmark candidate starts.
    monkeypatch.setenv("TRTLLM_NVFP4_RUNTIME_ACTIVATION", "caller-value")

    # When the 4o6 strategy is active, then its value is visible only in scope.
    with strategy_environment_scope(Nvfp4Strategy.FOUR_O_SIX):
        assert os.environ["TRTLLM_NVFP4_RUNTIME_ACTIVATION"] == "4o6"
    assert os.environ["TRTLLM_NVFP4_RUNTIME_ACTIVATION"] == "caller-value"


def test_synthetic_svdquant_load_scope_only_suppresses_factor_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given SVDQ runtime flags that were enabled before synthetic weight setup.
    for name in (
        "TRTLLM_SVDQUANT_NVFP4",
        "TRTLLM_SVDQUANT_FC13",
        "TRTLLM_SVDQUANT_FC2",
    ):
        monkeypatch.setenv(name, "1")

    # When regular benchmark weights load, correction loading is suppressed.
    with synthetic_svdquant_load_scope(Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64):
        assert os.environ["TRTLLM_SVDQUANT_NVFP4"] == "0"
        assert os.environ["TRTLLM_SVDQUANT_FC13"] == "0"
        assert os.environ["TRTLLM_SVDQUANT_FC2"] == "0"

    # Then all runtime flags are restored for the timed forward.
    assert os.environ["TRTLLM_SVDQUANT_NVFP4"] == "1"
    assert os.environ["TRTLLM_SVDQUANT_FC13"] == "1"
    assert os.environ["TRTLLM_SVDQUANT_FC2"] == "1"


def test_synthetic_svdquant_factors_are_zeroed_and_marked_loaded() -> None:
    # Given all six rank-factor tensors registered by the SVDQ MoE backend.
    module = torch.nn.Module()
    factor_names = tuple(
        f"{projection}_{factor}"
        for projection in ("w1", "w3", "w2")
        for factor in ("us", "vh")
    )
    for name in factor_names:
        module.register_parameter(name, torch.nn.Parameter(torch.ones(2, 2)))

    # When the synthetic overhead-only fixture is initialized.
    initialized = initialize_synthetic_svdquant_factors(
        module, Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64
    )

    # Then the runtime sees loaded factors without a residual-value contribution.
    assert initialized is True
    assert module._svdquant_loaded is True
    for name in factor_names:
        factor = getattr(module, name)
        assert torch.equal(factor, torch.zeros_like(factor))


def test_synthetic_svdquant_factors_run_the_production_fc13_packer() -> None:
    # Given a benchmark module exposing the real quant-method finalization hook.
    module = torch.nn.Module()
    for projection in ("w1", "w3", "w2"):
        module.register_parameter(
            f"{projection}_us", torch.nn.Parameter(torch.ones(2, 2)))
        module.register_parameter(
            f"{projection}_vh", torch.nn.Parameter(torch.ones(2, 2)))

    calls: list[torch.nn.Module] = []

    class _QuantMethod:

        def _pack_svdquant_fc13_vh(self, target: torch.nn.Module) -> None:
            calls.append(target)

    module.quant_method = _QuantMethod()

    # When the synthetic overhead fixture is initialized.
    initialize_synthetic_svdquant_factors(
        module, Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64)

    # Then it mirrors the production one-time FC13 packing step exactly once.
    assert calls == [module]
