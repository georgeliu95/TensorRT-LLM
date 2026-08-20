# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Report-level regression tests for the NVFP4 performance contract."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_MICROBENCHMARKS = _REPO_ROOT / "tests" / "microbenchmarks"


def _load_summary() -> ModuleType:
    contract_path = _MICROBENCHMARKS / "nvfp4_runtime_contract.py"
    provenance_path = _MICROBENCHMARKS / "nvfp4_provenance.py"
    summary_path = _MICROBENCHMARKS / "summarize_kimi_nvfp4_overhead.py"
    contract_spec = importlib.util.spec_from_file_location(
        "nvfp4_runtime_contract",
        contract_path,
    )
    if contract_spec is None or contract_spec.loader is None:
        raise ImportError(f"cannot load {contract_path}")
    contract = importlib.util.module_from_spec(contract_spec)
    provenance_spec = importlib.util.spec_from_file_location(
        "nvfp4_provenance",
        provenance_path,
    )
    if provenance_spec is None or provenance_spec.loader is None:
        raise ImportError(f"cannot load {provenance_path}")
    provenance = importlib.util.module_from_spec(provenance_spec)
    summary_spec = importlib.util.spec_from_file_location(
        "_test_summarize_kimi_nvfp4_overhead",
        summary_path,
    )
    if summary_spec is None or summary_spec.loader is None:
        raise ImportError(f"cannot load {summary_path}")
    summary = importlib.util.module_from_spec(summary_spec)
    with patch.dict(
        sys.modules,
        {
            contract_spec.name: contract,
            provenance_spec.name: provenance,
            summary_spec.name: summary,
        },
    ):
        contract_spec.loader.exec_module(contract)
        provenance_spec.loader.exec_module(provenance)
        summary_spec.loader.exec_module(summary)
    return summary


_SUMMARY = _load_summary()


def _write_reports_with_invalid_native(output_dir: Path) -> None:
    kernel_names_by_strategy = {
        "native": (
            "quantize_with_block_size<BlockScaleQuantizationType::NVFP4>",
            "BlockScaledContiguousGatherGroupedGemmKernel",
        ),
        "4o6": (
            "fused_prologue_quantize_v2<AdaptiveScaleRule)1>",
            "fused_moe_prologue_quantize_v2<AdaptiveScaleRule)1>",
        ),
        "4o6_svdq_r64": (
            "fused_prologue_quantize_v2<AdaptiveScaleRule)1>",
            "fused_moe_swiglu_prologue_quantize_v2<AdaptiveScaleRule)1>",
        ),
    }
    for strategy, kernel_names in kernel_names_by_strategy.items():
        results = []
        for num_tokens in (1, 4, 1536, 2048):
            results.append(
                {
                    "workload": {"num_tokens": num_tokens},
                    "status": "success",
                    "skip_reason": None,
                    "raw_data": {
                        "kernel_times_ms": {
                            "moe_forward_kernels": [
                                {
                                    "name": name,
                                    "per_rank": {"rank0": [0.01]},
                                }
                                for name in kernel_names
                            ]
                        }
                    },
                }
            )
        report = {"environment": {}, "results": results}
        (output_dir / f"{strategy}.json").write_text(
            json.dumps(report),
            encoding="utf-8",
        )
    _write_matching_provenance(output_dir)


def _write_matching_provenance(output_dir: Path) -> None:
    rules = {
        "native": {"fc13": 0, "fc2": 0},
        "4o6": {"fc13": 1, "fc2": 1},
        "4o6_svdq_r64": {"fc13": 1, "fc2": 1},
    }
    for strategy in rules:
        payload = {
            "strategy": strategy,
            "activation_input_dtype": "bfloat16",
            "runtime_quantization_rule": rules[strategy],
            "source_manifest_sha256": "source-a",
            "workspace": {
                "git_commit": "commit-a",
                "dirty_diff_sha256": "diff-a",
            },
            "custom_op_library": {"sha256": "library-a"},
        }
        (output_dir / f"{strategy}.provenance.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


def test_report_loader_rejects_native_without_runtime_quantization(
    tmp_path: Path,
) -> None:
    # Given a complete three-strategy result set with an invalid native trace.
    _write_reports_with_invalid_native(tmp_path)

    # When the publishable report contract loads the result set.
    with pytest.raises(RuntimeError, match="native.*runtime activation quantization"):
        _SUMMARY._load_reports(tmp_path)

    # Then invalid native data is rejected before any comparison is calculated.


def test_provenance_rejects_runtime_source_mismatch(tmp_path: Path) -> None:
    # Given workspace and runtime copies of one module with different bytes.
    workspace_path = tmp_path / "workspace_fused_moe_cute_dsl.py"
    runtime_path = tmp_path / "runtime_fused_moe_cute_dsl.py"
    workspace_path.write_text("RUNTIME_MODE = '4o6'\n", encoding="utf-8")
    runtime_path.write_text("RUNTIME_MODE = 'legacy'\n", encoding="utf-8")

    # When the exact-process provenance gate compares them.
    with pytest.raises(RuntimeError, match="fused_moe_cute_dsl.*does not match"):
        _SUMMARY.verify_source_identity(
            "fused_moe_cute_dsl",
            workspace_path,
            runtime_path,
        )

    # Then a stale installed Python module cannot enter benchmark timing.


def test_provenance_rejects_stale_operator_schema() -> None:
    # Given a wheel whose adaptive quantizer predates the routing-mask ABI.
    schemas = {
        "trtllm::fp4_quantize_fused": (
            "fp4_quantize_fused(Tensor input, int sfVecSize, "
            "int scaleRule=0) -> (Tensor, Tensor, Tensor)"
        ),
        "trtllm::fp4_swiglu_quantize_fused": (
            "fp4_swiglu_quantize_fused(Tensor preactivation, int sfVecSize, "
            "int scaleRule, Tensor tileIdxToMnLimit, "
            "Tensor numNonExitingTiles, int tileSize) "
            "-> (Tensor, Tensor, Tensor, Tensor)"
        ),
    }

    # When the exact-process provenance gate checks the loaded custom ops.
    with pytest.raises(RuntimeError, match="fp4_quantize_fused.*tileIdxToMnLimit"):
        _SUMMARY.verify_operator_schemas(schemas)

    # Then a stale custom-op library cannot enter benchmark timing.


def test_provenance_rejects_grouped_gemm_without_runtime_alpha_abi() -> None:
    # Given current quantizer schemas but a pre-fusion grouped-GEMM schema.
    schemas = {
        "trtllm::fp4_quantize_fused": (
            "fp4_quantize_fused(Tensor input, int scaleRule, "
            "Tensor tileIdxToMnLimit, Tensor numNonExitingTiles, "
            "int tileSize) -> (Tensor, Tensor, Tensor)"
        ),
        "trtllm::fp4_swiglu_quantize_fused": (
            "fp4_swiglu_quantize_fused(Tensor input, int scaleRule, "
            "Tensor tileIdxToMnLimit, Tensor numNonExitingTiles, "
            "int tileSize) -> (Tensor, Tensor, Tensor, Tensor)"
        ),
        "trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell": (
            "cute_dsl_nvfp4_grouped_gemm_blackwell(Tensor input, "
            "Tensor alpha) -> Tensor"
        ),
    }

    with pytest.raises(
        RuntimeError,
        match="cute_dsl_nvfp4_grouped_gemm_blackwell.*alpha_numerator",
    ):
        _SUMMARY.verify_operator_schemas(schemas)


def test_provenance_rejects_loaded_library_mismatch() -> None:
    # Given a manifest created for the current sources but a different .so.
    build_manifest = {
        "source_manifest_sha256": "source-a",
        "custom_op_library": {"sha256": "library-a"},
    }

    # When the benchmark process verifies its loaded binary identity.
    with pytest.raises(RuntimeError, match="custom-op library.*library-b.*library-a"):
        _SUMMARY.verify_build_identity(
            build_manifest,
            source_manifest_sha256="source-a",
            loaded_library_sha256="library-b",
        )

    # Then results from a stale or different binary cannot enter timing.


def test_source_manifest_digest_changes_with_build_input(tmp_path: Path) -> None:
    # Given one source file included in the build identity.
    source = tmp_path / "kernel.cu"
    source.write_text("constexpr int kRule = 0;\n", encoding="utf-8")
    first = _SUMMARY.build_source_manifest(tmp_path, ("kernel.cu",))

    # When that build input changes.
    source.write_text("constexpr int kRule = 1;\n", encoding="utf-8")
    second = _SUMMARY.build_source_manifest(tmp_path, ("kernel.cu",))

    # Then the source and aggregate build identities both change.
    assert first["files"]["kernel.cu"] != second["files"]["kernel.cu"]
    assert first["sha256"] != second["sha256"]


def test_provenance_loader_rejects_mixed_custom_op_build(tmp_path: Path) -> None:
    # Given three strategy sidecars where SVDQ loaded a different custom-op .so.
    _write_matching_provenance(tmp_path)
    svdq_path = tmp_path / "4o6_svdq_r64.provenance.json"
    svdq = json.loads(svdq_path.read_text(encoding="utf-8"))
    svdq["custom_op_library"]["sha256"] = "library-b"
    svdq_path.write_text(json.dumps(svdq), encoding="utf-8")

    # When the summary validates the comparison set.
    with pytest.raises(RuntimeError, match="same build.*custom_op_library"):
        _SUMMARY._load_provenance(tmp_path)

    # Then cross-build timing data cannot be combined.


def test_svdq_projection_and_lowrank_kernels_are_attributed_separately() -> None:
    # Given one shared main blockscaled kernel and a distinct BF16 low-rank kernel.
    result = {
        "latency_ms": {"score": 1.0},
        "raw_data": {
            "kernel_times_ms": {
                "moe_forward_kernels": [
                    {
                        "name": (
                            "blockscaled_contiguous_grouped_gemm main kernel"
                        ),
                        "per_rank": {
                            "rank0": [0.30, 0.20, 0.31, 0.21, 0.29, 0.19]
                        },
                    },
                    {
                        "name": "bf16_contiguous_grouped_gemm lowrank kernel",
                        "per_rank": {"rank0": [0.05] * 15},
                    },
                ]
            }
        },
    }

    # When the summary separates main projections from low-rank corrections.
    projection = _SUMMARY._projection_gemm_ms("4o6_svdq_r64", result)
    timeline = _SUMMARY._timeline_ms(result)

    # Then FC13/FC2 use alternating main samples and low-rank is not main GEMM.
    assert projection == {"fc13": 0.30, "fc2": 0.20}
    assert timeline["main_gemm"] == pytest.approx(0.50)
    assert timeline["lowrank_gemm"] == pytest.approx(0.25)
    assert "host_orchestration_gap" not in timeline


def test_timeline_does_not_infer_cpu_overhead_from_separate_passes() -> None:
    # Given an E2E score and kernel samples collected by different passes.
    result = {
        "latency_ms": {"score": 1.0},
        "raw_data": {
            "kernel_times_ms": {
                "moe_forward_kernels": [
                    {
                        "name": "phase2_only_quantize<AdaptiveScaleRule)1>",
                        "per_rank": {"rank0": [0.10, 0.10, 0.10]},
                    }
                ]
            }
        },
    }

    timeline = _SUMMARY._timeline_ms(result)

    # Then only directly measured GPU data is published; CPU timing requires Nsys.
    assert timeline == {
        "main_gemm": 0.0,
        "activation_quantization": pytest.approx(0.10),
        "lowrank_gemm": 0.0,
        "copies": 0.0,
        "other_gpu": 0.0,
        "gpu_kernel_total": pytest.approx(0.10),
    }
