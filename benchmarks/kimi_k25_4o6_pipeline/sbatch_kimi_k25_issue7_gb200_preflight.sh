#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llm4o6-kimi-issue7-gb200-preflight
#SBATCH --partition=gb200nvl4
#SBATCH --account=blackwell
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=gb200-nvl4-ts2-93
#SBATCH --time=00:20:00
#SBATCH --output=/home/scratch.georgel_gpu/.ssh-gw/job-logs/sbatch_kimi_k25_issue7_gb200_preflight_%j.out

set -euo pipefail

ROOT="${ROOT:-/home/scratch.georgel_gpu/projects/llm_4o6}"
REPO="${REPO:-${ROOT}/TensorRT-LLM-weight-act-4o6-rc14}"
MODEL="${MODEL:-${ROOT}/checkpoints/4o6_nvfp4/kimi_k2_5_int4_to_4o6_nvfp4_qwen_workflow_reuse_20260615T084458Z}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14}"

echo "=== host ==="
date
hostname
uname -m
nvidia-smi -L
command -v docker || true

echo "=== paths ==="
for path in \
    "${ROOT}" \
    "${REPO}" \
    "${MODEL}" \
    /home/scratch.trt_llm_data \
    /home/scratch.trt_llm_data/llm-models \
    /llm-models
do
    echo "PATH ${path}"
    ls -ld "${path}" || true
done

echo "=== container ==="
if command -v docker >/dev/null 2>&1; then
    docker run --rm --runtime=nvidia \
        --net=host --ipc=host --pid=host --shm-size=32g \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -v /home/scratch.georgel_gpu:/home/scratch.georgel_gpu \
        -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
        -v /home/scratch.trt_llm_data/llm-models:/llm-models:ro \
        -w "${REPO}" \
        "${IMAGE}" \
        bash -lc "
set -euo pipefail
uname -m
nvidia-smi -L
test -d '${REPO}'
test -f '${MODEL}/model.safetensors.index.json'
test -f '${MODEL}/hf_quant_config.json'
test -d /llm-models/datasets/openai/gsm8k
test -d /llm-models/datasets/mmlu
python3 - <<'PY'
import platform
print(platform.machine())
PY
"
else
    echo "docker unavailable"
    exit 2
fi
