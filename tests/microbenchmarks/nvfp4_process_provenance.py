# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect NVFP4 benchmark provenance inside the measured Python process."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nvfp4_provenance import (
    build_source_manifest,
    verify_build_identity,
    verify_operator_schemas,
    verify_source_identity,
)

_BUILD_INPUTS = (
    "cpp/tensorrt_llm/kernels/cuteDslKernels/moeUtils.cu",
    "cpp/tensorrt_llm/kernels/cuteDslKernels/moeUtils.h",
    "cpp/tensorrt_llm/kernels/fp4QuantizeAdaptive.cu",
    "cpp/tensorrt_llm/kernels/fp4QuantizeAdaptive.cuh",
    "cpp/tensorrt_llm/kernels/fp4QuantizeAdaptive.h",
    "cpp/tensorrt_llm/thop/fp4QuantizeAdaptiveOp.cpp",
    "cpp/tensorrt_llm/thop/cuteDslMoeUtilsOp.cpp",
    "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py",
    "tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py",
    "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/bf16_contiguous_grouped_gemm.py",
    "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/bf16_deinterleave.py",
    "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_gather_grouped_gemm_act_fusion.py",
    "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm.py",
    "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/blockscaled_contiguous_grouped_gemm_finalize_fusion.py",
    "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/utils.py",
    "tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py",
    "tensorrt_llm/_torch/modules/fused_moe/quantization.py",
    "tests/microbenchmarks/bench_moe/__main__.py",
    "tests/microbenchmarks/bench_moe/build.py",
    "tests/microbenchmarks/bench_moe/case_runner.py",
    "tests/microbenchmarks/bench_moe/cli.py",
    "tests/microbenchmarks/bench_moe/nvfp4_overhead.py",
    "tests/microbenchmarks/bench_moe/search.py",
    "tests/microbenchmarks/bench_moe/specs.py",
    "tests/microbenchmarks/bench_moe/timing/eager.py",
    "tests/microbenchmarks/nvfp4_process_provenance.py",
    "tests/microbenchmarks/nvfp4_provenance.py",
    "tests/microbenchmarks/nvfp4_runtime_contract.py",
    "tests/microbenchmarks/run_kimi_nvfp4_overhead.sh",
    "tests/microbenchmarks/summarize_kimi_nvfp4_overhead.py",
)

_RUNTIME_MODULES = {
    "tensorrt_llm": "tensorrt_llm/__init__.py",
    "fused_moe_cute_dsl": (
        "tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py"
    ),
    "quantization": "tensorrt_llm/_torch/modules/fused_moe/quantization.py",
    "cpp_custom_ops": "tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py",
    "cute_dsl_custom_ops": "tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py",
}

_MODULE_IMPORTS = {
    "tensorrt_llm": "tensorrt_llm",
    "fused_moe_cute_dsl": "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl",
    "quantization": "tensorrt_llm._torch.modules.fused_moe.quantization",
    "cpp_custom_ops": "tensorrt_llm._torch.custom_ops.cpp_custom_ops",
    "cute_dsl_custom_ops": "tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops",
}

_OPERATORS = (
    "trtllm::fp4_quantize_fused",
    "trtllm::fp4_swiglu_quantize_fused",
    "trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell",
    "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell",
    "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell",
    "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_bf16_blackwell",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(workspace: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(workspace), *args),
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _loaded_custom_op_library() -> dict[str, str]:
    import torch

    candidates = sorted(
        Path(path).resolve()
        for path in torch.classes.loaded_libraries
        if Path(path).name == "libth_common.so"
    )
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one loaded libth_common.so, got "
            f"{[str(path) for path in candidates]}"
        )
    path = candidates[0]
    return {"path": str(path), "sha256": _sha256(path)}


def _operator_schemas() -> dict[str, str]:
    import torch

    schemas = {
        name: str(torch._C._dispatch_find_schema_or_throw(name, "").schema())
        for name in _OPERATORS
    }
    verify_operator_schemas(schemas)
    return schemas


def _runtime_sources(workspace: Path) -> list[dict[str, str]]:
    identities = []
    for logical_name, import_name in _MODULE_IMPORTS.items():
        module = importlib.import_module(import_name)
        runtime_file = getattr(module, "__file__", None)
        if not runtime_file:
            raise RuntimeError(f"{import_name} has no runtime source path")
        identity = verify_source_identity(
            logical_name,
            workspace / _RUNTIME_MODULES[logical_name],
            Path(runtime_file),
        )
        identities.append(
            {
                "logical_name": identity.logical_name,
                "workspace_path": identity.workspace_path,
                "runtime_path": identity.runtime_path,
                "sha256": identity.sha256,
            }
        )
    return identities


def _platform() -> dict[str, Any]:
    import torch
    import tensorrt_llm

    driver = subprocess.run(
        ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()[0]
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "tensorrt_llm_file": str(Path(tensorrt_llm.__file__).resolve()),
        "cuda_version": torch.version.cuda,
        "driver_version": driver,
        "gpu_name": torch.cuda.get_device_name(0),
    }


def _workspace_identity(workspace: Path) -> dict[str, str]:
    declared_commit = os.environ.get("NVFP4_BENCHMARK_GIT_COMMIT")
    git_commit = declared_commit or _git_output(
        workspace,
        "rev-parse",
        "HEAD",
    ).decode().strip()
    diff = _git_output(workspace, "diff", "--binary", git_commit)
    return {
        "root": str(workspace.resolve()),
        "git_commit": git_commit,
        "dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def create_build_manifest(workspace: Path, output: Path) -> None:
    """Bind the just-built custom-op library to the current source snapshot."""
    _runtime_sources(workspace)
    source_manifest = build_source_manifest(workspace, _BUILD_INPUTS)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workspace": _workspace_identity(workspace),
        "source_manifest_sha256": source_manifest["sha256"],
        "source_files": source_manifest["files"],
        "custom_op_library": _loaded_custom_op_library(),
        "operator_schemas": _operator_schemas(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def collect_run_provenance(
    workspace: Path,
    build_manifest_path: Path,
    strategy: str,
) -> dict[str, Any]:
    """Validate and return provenance before benchmark timing starts."""
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    source_manifest = build_source_manifest(workspace, _BUILD_INPUTS)
    library = _loaded_custom_op_library()
    verify_build_identity(
        build_manifest,
        source_manifest_sha256=source_manifest["sha256"],
        loaded_library_sha256=library["sha256"],
    )
    rules = {
        "native": {"fc13": 0, "fc2": 0},
        "4o6": {"fc13": 1, "fc2": 1},
        "4o6_svdq_r64": {"fc13": 1, "fc2": 1},
    }
    if strategy not in rules:
        raise ValueError(f"unsupported NVFP4 benchmark strategy: {strategy}")
    return {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "strategy": strategy,
        "activation_input_dtype": "bfloat16",
        "runtime_quantization_rule": rules[strategy],
        "workspace": _workspace_identity(workspace),
        "source_manifest_sha256": source_manifest["sha256"],
        "runtime_sources": _runtime_sources(workspace),
        "custom_op_library": library,
        "operator_schemas": _operator_schemas(),
        "platform": _platform(),
    }


def run_preflight_from_environment() -> None:
    """Run only for the dedicated NVFP4 benchmark wrapper."""
    output_value = os.environ.get("NVFP4_BENCHMARK_PROVENANCE_OUTPUT")
    if output_value is None:
        return
    required = {
        name: os.environ.get(name)
        for name in (
            "NVFP4_BENCHMARK_WORKSPACE",
            "NVFP4_BENCHMARK_BUILD_MANIFEST",
            "NVFP4_BENCHMARK_STRATEGY",
        )
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(f"missing provenance environment: {missing}")
    payload = collect_run_provenance(
        Path(str(required["NVFP4_BENCHMARK_WORKSPACE"])),
        Path(str(required["NVFP4_BENCHMARK_BUILD_MANIFEST"])),
        str(required["NVFP4_BENCHMARK_STRATEGY"]),
    )
    output = Path(output_value)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("create-build-manifest",))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    create_build_manifest(args.workspace, args.output)


if __name__ == "__main__":
    main()
