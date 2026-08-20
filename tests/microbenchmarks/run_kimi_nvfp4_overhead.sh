#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output_dir="${1:-${repo_root}/out/kimi_nvfp4_overhead}"
build_manifest="${2:-${NVFP4_BENCHMARK_BUILD_MANIFEST:-}}"
python_bin="${PYTHON_BIN:-python3}"

if [[ -z "${build_manifest}" || ! -f "${build_manifest}" ]]; then
    echo "A build manifest created after compiling the current source is required." >&2
    echo "Pass it as argument 2 or NVFP4_BENCHMARK_BUILD_MANIFEST." >&2
    exit 2
fi

mkdir -p "${output_dir}"

for strategy in native 4o6 4o6_svdq_r64; do
    PYTHONPATH="${repo_root}/tests/microbenchmarks:${PYTHONPATH:-}" \
        NVFP4_BENCHMARK_WORKSPACE="${repo_root}" \
        NVFP4_BENCHMARK_BUILD_MANIFEST="${build_manifest}" \
        NVFP4_BENCHMARK_STRATEGY="${strategy}" \
        NVFP4_BENCHMARK_PROVENANCE_OUTPUT="${output_dir}/${strategy}.provenance.json" \
        "${python_bin}" -m bench_moe \
        --world_size 1 \
        --model kimi_k2_ep4_shard_nvfp4 \
        --backend CUTEDSL \
        --parallel_mode DEP \
        --nvfp4_strategy "${strategy}" \
        --balanced_total_num_tokens 1 4 1536 2048 \
        --routing_mode forced \
        --comm_pattern local_only \
        --expert_pattern balanced \
        --no_cuda_graph \
        --warmup 3 \
        --iters 20 \
        --analysis kernels \
        --output_file "${output_dir}/${strategy}.json"
done

PYTHONPATH="${repo_root}/tests/microbenchmarks:${PYTHONPATH:-}" \
    "${python_bin}" "${repo_root}/tests/microbenchmarks/summarize_kimi_nvfp4_overhead.py" \
    "${output_dir}"
