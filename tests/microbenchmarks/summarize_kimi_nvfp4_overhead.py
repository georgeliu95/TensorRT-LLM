# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and summarize the Kimi NVFP4 GPU performance contract."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from nvfp4_runtime_contract import (
    Nvfp4RuntimeContractError,
    Nvfp4Strategy,
    validate_runtime_quantization_contract,
)
from nvfp4_provenance import (
    build_source_manifest as build_source_manifest,
    verify_build_identity as verify_build_identity,
    verify_operator_schemas as verify_operator_schemas,
    verify_source_identity as verify_source_identity,
)

_STRATEGIES = ("native", "4o6", "4o6_svdq_r64")
_TOKEN_CASES = (1, 4, 1536, 2048)
_PROFILER_ITERS = 3


class PerformanceContractError(RuntimeError):
    """A benchmark report is missing or cannot be compared safely."""


def _load_provenance(output_dir: Path) -> dict[str, dict[str, Any]]:
    expected_rules = {
        "native": {"fc13": 0, "fc2": 0},
        "4o6": {"fc13": 1, "fc2": 1},
        "4o6_svdq_r64": {"fc13": 1, "fc2": 1},
    }
    provenance: dict[str, dict[str, Any]] = {}
    baseline_identity: dict[str, str] | None = None
    for strategy in _STRATEGIES:
        path = output_dir / f"{strategy}.provenance.json"
        if not path.is_file():
            raise PerformanceContractError(f"missing provenance: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("strategy") != strategy:
            raise PerformanceContractError(
                f"{path}: strategy is {payload.get('strategy')!r}, expected {strategy!r}"
            )
        if payload.get("activation_input_dtype") != "bfloat16":
            raise PerformanceContractError(
                f"{path}: activation input must be bfloat16"
            )
        if payload.get("runtime_quantization_rule") != expected_rules[strategy]:
            raise PerformanceContractError(
                f"{path}: unexpected runtime quantization rule"
            )
        workspace = payload.get("workspace") or {}
        library = payload.get("custom_op_library") or {}
        identity = {
            "source_manifest_sha256": str(
                payload.get("source_manifest_sha256", "")
            ),
            "git_commit": str(workspace.get("git_commit", "")),
            "dirty_diff_sha256": str(workspace.get("dirty_diff_sha256", "")),
            "custom_op_library.sha256": str(library.get("sha256", "")),
        }
        missing = [key for key, value in identity.items() if not value]
        if missing:
            raise PerformanceContractError(f"{path}: missing identity fields {missing}")
        if baseline_identity is None:
            baseline_identity = identity
        elif identity != baseline_identity:
            differing = [
                key
                for key, value in identity.items()
                if value != baseline_identity[key]
            ]
            raise PerformanceContractError(
                "all strategies must use the same build; differing provenance: "
                f"{differing}"
            )
        provenance[strategy] = payload
    return provenance


def _load_reports(output_dir: Path) -> dict[str, dict[str, Any]]:
    provenance = _load_provenance(output_dir)
    reports: dict[str, dict[str, Any]] = {}
    for strategy in _STRATEGIES:
        path = output_dir / f"{strategy}.json"
        if not path.is_file():
            raise PerformanceContractError(f"missing report: {path}")
        with path.open(encoding="utf-8") as stream:
            report = json.load(stream)
        results = report.get("results")
        if not isinstance(results, list):
            raise PerformanceContractError(f"{path}: results must be a list")
        by_tokens = {item["workload"]["num_tokens"]: item for item in results}
        if tuple(sorted(by_tokens)) != _TOKEN_CASES:
            raise PerformanceContractError(
                f"{path}: expected token cases {_TOKEN_CASES}, got {tuple(sorted(by_tokens))}"
            )
        failures = [
            (tokens, item.get("status"), item.get("skip_reason"))
            for tokens, item in by_tokens.items()
            if item.get("status") != "success"
        ]
        if failures:
            raise PerformanceContractError(f"{path}: failed cases: {failures}")
        try:
            for result in by_tokens.values():
                kernel_names = (
                    item["name"]
                    for item in result["raw_data"]["kernel_times_ms"][
                        "moe_forward_kernels"
                    ]
                )
                validate_runtime_quantization_contract(
                    Nvfp4Strategy(strategy),
                    kernel_names,
                )
        except Nvfp4RuntimeContractError as exc:
            raise PerformanceContractError(f"{path}: {exc}") from exc
        reports[strategy] = {
            **report,
            "_by_tokens": by_tokens,
            "_provenance": provenance[strategy],
        }
    return reports


def _kernel_samples(result: dict[str, Any]) -> list[dict[str, Any]]:
    return result["raw_data"]["kernel_times_ms"]["moe_forward_kernels"]


def _is_lowrank_gemm(name: str) -> bool:
    lowered = name.lower()
    return (
        "bf16_contiguous_grouped_gemm" in lowered
        or "nvjet_" in lowered
        or "cublaslt::splitkreduce" in lowered
    )


def _is_main_projection_gemm(name: str) -> bool:
    return "grouped_gemm" in name.lower() and not _is_lowrank_gemm(name)


def _projection_gemm_ms(
    strategy: str, result: dict[str, Any]
) -> dict[str, float]:
    kernels = [
        item
        for item in _kernel_samples(result)
        if _is_main_projection_gemm(item["name"])
    ]
    if strategy != "4o6_svdq_r64":
        if len(kernels) != 2:
            raise PerformanceContractError(
                f"{strategy}: expected separate FC13/FC2 kernels, got {len(kernels)}"
            )
        return {
            "fc13": statistics.median(kernels[0]["per_rank"]["rank0"]),
            "fc2": statistics.median(kernels[1]["per_rank"]["rank0"]),
        }

    if len(kernels) != 1:
        raise PerformanceContractError(
            f"{strategy}: expected one shared grouped-GEMM kernel, got {len(kernels)}"
        )
    samples = kernels[0]["per_rank"]["rank0"]
    if len(samples) != 2 * _PROFILER_ITERS:
        raise PerformanceContractError(
            f"{strategy}: expected alternating FC13/FC2 samples, got {len(samples)}"
        )
    return {
        "fc13": statistics.median(samples[0::2]),
        "fc2": statistics.median(samples[1::2]),
    }


def _timeline_ms(result: dict[str, Any]) -> dict[str, float]:
    categories = {
        "main_gemm": 0.0,
        "activation_quantization": 0.0,
        "lowrank_gemm": 0.0,
        "copies": 0.0,
        "other_gpu": 0.0,
    }
    for kernel in _kernel_samples(result):
        name = kernel["name"].lower()
        duration = sum(kernel["per_rank"]["rank0"]) / _PROFILER_ITERS
        if _is_lowrank_gemm(name):
            category = "lowrank_gemm"
        elif _is_main_projection_gemm(name):
            category = "main_gemm"
        elif "quantize" in name or "dequant" in name:
            category = "activation_quantization"
        elif "memcpy" in name:
            category = "copies"
        else:
            category = "other_gpu"
        categories[category] += duration
    gpu_total = sum(categories.values())
    return {
        **categories,
        "gpu_kernel_total": gpu_total,
    }


def _shape(tokens: int) -> dict[str, Any]:
    expanded = tokens * 8
    active_experts = min(expanded, 96)
    return {
        "phase": "decode" if tokens <= 4 else "prefill",
        "total_tokens": tokens,
        "active_experts": active_experts,
        "expert_m_min": expanded // active_experts,
        "expert_m_max": (expanded + active_experts - 1) // active_experts,
        "fc13_nk": [4096, 7168],
        "fc2_nk": [7168, 2048],
    }


def summarize(output_dir: Path) -> dict[str, Any]:
    """Return a normalized summary, failing if any GPU case did not run."""
    reports = _load_reports(output_dir)
    cases: list[dict[str, Any]] = []
    for tokens in _TOKEN_CASES:
        native = reports["native"]["_by_tokens"][tokens]
        four_o_six = reports["4o6"]["_by_tokens"][tokens]
        svdq = reports["4o6_svdq_r64"]["_by_tokens"][tokens]
        e2e = {
            "native": float(native["latency_ms"]["score"]),
            "4o6": float(four_o_six["latency_ms"]["score"]),
            "4o6_svdq_r64": float(svdq["latency_ms"]["score"]),
        }
        cases.append(
            {
                **_shape(tokens),
                "e2e_ms": e2e,
                "overhead_percent": {
                    "4o6_vs_native": (e2e["4o6"] / e2e["native"] - 1.0) * 100.0,
                    "svdq_vs_native": (
                        e2e["4o6_svdq_r64"] / e2e["native"] - 1.0
                    )
                    * 100.0,
                    "svdq_vs_4o6": (
                        e2e["4o6_svdq_r64"] / e2e["4o6"] - 1.0
                    )
                    * 100.0,
                },
                "projection_gemm_ms": {
                    strategy: _projection_gemm_ms(
                        strategy, reports[strategy]["_by_tokens"][tokens]
                    )
                    for strategy in _STRATEGIES
                },
                "timeline_ms": {
                    strategy: _timeline_ms(
                        reports[strategy]["_by_tokens"][tokens]
                    )
                    for strategy in _STRATEGIES
                },
            }
        )
    return {
        "benchmark": "kimi_k2_ep4_shard_nvfp4_overhead",
        "baseline": "native",
        "device": reports["native"]["environment"].get("device_name"),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    summary = summarize(args.output_dir)
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Performance contract passed; summary written to {path}")


if __name__ == "__main__":
    main()
