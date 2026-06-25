#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=llm4o6-kimi-issue7-probe-autotuner-off
#SBATCH --partition=b200@500-1000W/umbriel-b200@ts4/8gpu-224cpu-2048gb
#SBATCH --account=all-users
#SBATCH --qos=batch-short
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclude=umb-b200-259
#SBATCH --time=02:00:00
#SBATCH --output=/home/scratch.georgel_gpu/projects/llm_4o6/benchmarks/sbatch_kimi_k25_issue7_probe_b200_autotuner_off_%j.out

set -euo pipefail

ROOT="${ROOT:-/home/scratch.georgel_gpu/projects/llm_4o6}"
REPO="${REPO:-${ROOT}/TensorRT-LLM-weight-act-4o6-rc14}"
MODEL="${MODEL:-${ROOT}/checkpoints/4o6_nvfp4/kimi_k2_5_int4_to_4o6_nvfp4_qwen_workflow_reuse_20260615T084458Z}"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmarks/kimi_k25_issue7_probe_b200_autotuner_off_${SLURM_JOB_ID:-manual}_${RUN_STAMP}}"

cd "${REPO}"

MODEL="${MODEL}" \
OUT_ROOT="${OUT_ROOT}" \
RUN_GSM8K=1 \
RUN_MMLU=0 \
GSM8K_NUM_SAMPLES=1 \
MMLU_NUM_SAMPLES=1 \
MAX_BATCH_SIZE=4 \
MAX_NUM_TOKENS=4096 \
MAX_SEQ_LEN=8192 \
TRTLLM_EVAL_MAX_INFLIGHT=4 \
TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1 \
TRTLLM_ENABLE_PDL=0 \
TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=1 \
TRTLLM_4O6_LOAD_TIMING=1 \
TRTLLM_4O6_LOAD_TIMING_RANKS=all \
ENABLE_AUTOTUNER=false \
bash benchmarks/kimi_k25_4o6_pipeline/run_kimi_k25_8xb200_eval_cli.sh
