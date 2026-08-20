# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the measured NVFP4 runtime kernel contract."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_MICROBENCHMARKS_DIR = _REPO_ROOT / "tests" / "microbenchmarks"
_BENCH_MOE_DIR = _REPO_ROOT / "tests" / "microbenchmarks" / "bench_moe"


def _load_nvfp4_overhead() -> ModuleType:
    svdquant_helpers = ModuleType("svdquant_helpers")
    svdquant_helpers.ENV_ENABLED = "TRTLLM_SVDQUANT_NVFP4"
    svdquant_helpers.ENV_RANK = "TRTLLM_SVDQUANT_RANK"
    svdquant_helpers.ENV_US_DTYPE = "TRTLLM_SVDQUANT_US_DTYPE"
    svdquant_helpers.ENV_DEVICE = "TRTLLM_SVDQUANT_DEVICE"
    svdquant_helpers.ENV_FC13 = "TRTLLM_SVDQUANT_FC13"
    svdquant_helpers.ENV_FC2 = "TRTLLM_SVDQUANT_FC2"
    svdquant_helpers.ENV_NAMES = (
        svdquant_helpers.ENV_ENABLED,
        svdquant_helpers.ENV_RANK,
        svdquant_helpers.ENV_US_DTYPE,
        svdquant_helpers.ENV_DEVICE,
        svdquant_helpers.ENV_FC13,
        svdquant_helpers.ENV_FC2,
    )
    fused_moe = ModuleType("tensorrt_llm._torch.modules.fused_moe")
    fused_moe.svdquant_helpers = svdquant_helpers
    bench_moe = ModuleType("bench_moe")
    bench_moe.__path__ = [str(_BENCH_MOE_DIR)]
    contract_path = _MICROBENCHMARKS_DIR / "nvfp4_runtime_contract.py"
    overhead_path = _BENCH_MOE_DIR / "nvfp4_overhead.py"
    with patch.dict(
        sys.modules,
        {
            "bench_moe": bench_moe,
            "tensorrt_llm": ModuleType("tensorrt_llm"),
            "tensorrt_llm._torch": ModuleType("tensorrt_llm._torch"),
            "tensorrt_llm._torch.modules": ModuleType("tensorrt_llm._torch.modules"),
            "tensorrt_llm._torch.modules.fused_moe": fused_moe,
        },
    ):
        contract_spec = importlib.util.spec_from_file_location(
            "nvfp4_runtime_contract",
            contract_path,
        )
        if contract_spec is None or contract_spec.loader is None:
            raise ImportError(f"cannot load {contract_path}")
        contract = importlib.util.module_from_spec(contract_spec)
        sys.modules[contract_spec.name] = contract
        contract_spec.loader.exec_module(contract)
        overhead_spec = importlib.util.spec_from_file_location(
            "bench_moe.nvfp4_overhead",
            overhead_path,
        )
        if overhead_spec is None or overhead_spec.loader is None:
            raise ImportError(f"cannot load {overhead_path}")
        module = importlib.util.module_from_spec(overhead_spec)
        sys.modules[overhead_spec.name] = module
        overhead_spec.loader.exec_module(module)
    return module


_NVFP4_OVERHEAD = _load_nvfp4_overhead()


@pytest.mark.parametrize(
    ("strategy", "kernel_names"),
    [
        (
            "NATIVE",
            (
                "fused_prologue_quantize_v1<BlockScaleQuantizationType::NVFP4, bf16, 16>",
                "phase2_only_quantize<AdaptiveScaleRule)0>",
            ),
        ),
        (
            "FOUR_O_SIX",
            (
                "fused_prologue_quantize_v2<AdaptiveScaleRule)1>",
                "phase2_only_quantize<AdaptiveScaleRule)1>",
            ),
        ),
        (
            "FOUR_O_SIX_SVDQ_R64",
            (
                "fused_prologue_quantize_v2<AdaptiveScaleRule)1>",
                "fused_moe_swiglu_prologue_quantize_v2<AdaptiveScaleRule)1>",
            ),
        ),
    ],
)
def test_report_contract_accepts_runtime_quantization_signatures(
    strategy: str,
    kernel_names: tuple[str, ...],
) -> None:
    # Given the expected FC13 and FC2 runtime signatures for one strategy.
    selected_strategy = _NVFP4_OVERHEAD.Nvfp4Strategy[strategy]

    # When the measured kernel set crosses the contract.
    _NVFP4_OVERHEAD.validate_runtime_quantization_contract(
        selected_strategy,
        kernel_names,
    )

    # Then no contract error is raised.


def test_report_contract_rejects_native_without_runtime_quantization() -> None:
    # Given the invalid native trace that used the legacy static activation path.
    kernel_names = (
        "quantize_with_block_size<BlockScaleQuantizationType::NVFP4, bf16, 16>",
        "BlockScaledContiguousGatherGroupedGemmKernel",
        "Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel",
    )

    # When it crosses the strategy-specific measured-kernel contract.
    with pytest.raises(RuntimeError, match="native.*runtime activation quantization"):
        _NVFP4_OVERHEAD.validate_runtime_quantization_contract(
            _NVFP4_OVERHEAD.Nvfp4Strategy.NATIVE,
            kernel_names,
        )

    # Then the report cannot publish this run as a native NVFP4 baseline.


def test_report_contract_rejects_4o6_with_standard_quantization() -> None:
    # Given a trace whose two activation quantizers both use native rule 0.
    kernel_names = (
        "fused_prologue_quantize_v1<BlockScaleQuantizationType::NVFP4, bf16, 16>",
        "fused_moe_prologue_quantize_v2<AdaptiveScaleRule)0>",
    )

    # When it is labeled as adaptive 4o6.
    with pytest.raises(RuntimeError, match="4o6.*adaptive rule 1"):
        _NVFP4_OVERHEAD.validate_runtime_quantization_contract(
            _NVFP4_OVERHEAD.Nvfp4Strategy.FOUR_O_SIX,
            kernel_names,
        )

    # Then the report cannot publish a mislabeled native quantization path.


def test_report_contract_rejects_svdquant_with_standard_quantization() -> None:
    # Given an SVDQuant-labeled trace whose activation quantizers use rule 0.
    kernel_names = (
        "fused_prologue_quantize_v1<BlockScaleQuantizationType::NVFP4, bf16, 16>",
        "fused_moe_swiglu_prologue_quantize_v2<AdaptiveScaleRule)0>",
    )

    # When it crosses the SVDQuant runtime contract.
    with pytest.raises(RuntimeError, match="4o6_svdq_r64.*adaptive rule 1"):
        _NVFP4_OVERHEAD.validate_runtime_quantization_contract(
            _NVFP4_OVERHEAD.Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64,
            kernel_names,
        )

    # Then the report cannot publish a mislabeled rule-0 correction path.


def test_report_contract_rejects_activation_dequantization() -> None:
    # Given otherwise-valid SVDQuant rule-1 kernels plus an activation dequantizer.
    kernel_names = (
        "fused_prologue_quantize_v2<AdaptiveScaleRule)1>",
        "fused_moe_swiglu_prologue_quantize_v2<AdaptiveScaleRule)1>",
        "dequant_nvfp4_swizzled_sf_kernel<bf16>",
    )

    # When the measured kernel set crosses the contract.
    with pytest.raises(RuntimeError, match="unexpected activation dequantization"):
        _NVFP4_OVERHEAD.validate_runtime_quantization_contract(
            _NVFP4_OVERHEAD.Nvfp4Strategy.FOUR_O_SIX_SVDQ_R64,
            kernel_names,
        )

    # Then reconstruction of BF16 from the FP4 main path cannot be published.


def test_report_contract_rejects_legacy_static_activation_quantization() -> None:
    # Given valid native runtime kernels plus a legacy checkpoint-scale quantizer.
    kernel_names = (
        "fused_prologue_quantize_v1<BlockScaleQuantizationType::NVFP4, bf16, 16>",
        "fused_moe_prologue_quantize_v2<AdaptiveScaleRule)0>",
        "quantize_with_block_size<BlockScaleQuantizationType::NVFP4>",
    )

    # When the mixed measured kernel set crosses the contract.
    with pytest.raises(RuntimeError, match="legacy static activation quantization"):
        _NVFP4_OVERHEAD.validate_runtime_quantization_contract(
            _NVFP4_OVERHEAD.Nvfp4Strategy.NATIVE,
            kernel_names,
        )

    # Then a partially corrected native path cannot be published.
