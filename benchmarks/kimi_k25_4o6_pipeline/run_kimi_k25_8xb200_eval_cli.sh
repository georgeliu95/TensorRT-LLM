#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT="${ROOT:-/home/scratch.georgel_gpu/projects/llm_4o6}"
REPO="${REPO:-${ROOT}/TensorRT-LLM-weight-act-4o6-rc14}"
IMAGE="${IMAGE:-nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14}"
MODEL="${MODEL:?set MODEL to the converted Kimi-K2.5 4o6 NVFP4 checkpoint path}"
ADAPTIVE_FP4_SO="${ADAPTIVE_FP4_SO:-${ROOT}/tmp/libfp4QuantizeAdaptive_sm100a.so}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/benchmarks/kimi_k25_8xb200_eval_cli_$(date -u +%Y%m%dT%H%M%SZ)}"
EP_SIZE="${EP_SIZE:-8}"
TP_SIZE="${TP_SIZE:-8}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${TP_SIZE}}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-}"
KV_FREE_FRACTION="${KV_FREE_FRACTION:-0.72}"
GSM8K_NUM_SAMPLES="${GSM8K_NUM_SAMPLES:-1319}"
MMLU_NUM_SAMPLES="${MMLU_NUM_SAMPLES:-full}"
MMLU_NUM_FEWSHOT="${MMLU_NUM_FEWSHOT:-5}"
RANDOM_SEED="${RANDOM_SEED:-0}"
RUN_GSM8K="${RUN_GSM8K:-1}"
RUN_MMLU="${RUN_MMLU:-1}"
TRTLLM_EVAL_MAX_INFLIGHT="${TRTLLM_EVAL_MAX_INFLIGHT:-32}"
TLLM_DISABLE_ALLREDUCE_AUTOTUNE="${TLLM_DISABLE_ALLREDUCE_AUTOTUNE:-0}"
TRTLLM_ENABLE_PDL="${TRTLLM_ENABLE_PDL:-1}"
TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-}"
TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-}"
TRTLLM_ADAPTIVE_FP4="${TRTLLM_ADAPTIVE_FP4:-1}"
TRTLLM_ADAPTIVE_FP4_FC2="${TRTLLM_ADAPTIVE_FP4_FC2:-1}"
TRTLLM_ADAPTIVE_FP4_WEIGHT="${TRTLLM_ADAPTIVE_FP4_WEIGHT:-1}"
TRTLLM_MOE_FORCE_CUTEDSL="${TRTLLM_MOE_FORCE_CUTEDSL:-}"
TRTLLM_ADAPTIVE_FP4_DEBUG="${TRTLLM_ADAPTIVE_FP4_DEBUG:-}"
TRTLLM_4O6_LOAD_TIMING="${TRTLLM_4O6_LOAD_TIMING:-}"
TRTLLM_4O6_LOAD_TIMING_RANKS="${TRTLLM_4O6_LOAD_TIMING_RANKS:-}"
GSM8K_DATASET_PATH="${GSM8K_DATASET_PATH:-/llm-models/datasets/openai/gsm8k}"
MMLU_DATASET_PATH="${MMLU_DATASET_PATH:-/llm-models/datasets/mmlu}"
RUN_AS_ROOT="${RUN_AS_ROOT:-0}"

mkdir -p "${OUT_ROOT}"
echo "${OUT_ROOT}"

CUDA_VISIBLE_DEVICES_LIST="$(seq -s, 0 "$((GPUS_PER_NODE - 1))")"

docker_user_args=()
if [[ "${RUN_AS_ROOT}" != "1" ]]; then
    docker_user_args+=(--user "$(id -u):$(id -g)")
fi

docker run --rm --runtime=nvidia \
    "${docker_user_args[@]}" \
    --net=host --ipc=host --pid=host --shm-size=128g --ulimit memlock=-1:-1 \
    --security-opt seccomp=unconfined \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_LIST}" \
    -e HOME=/tmp/llm4o6-home \
    -e USER=georgel \
    -e LOGNAME=georgel \
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor-georgel \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e PYTHONUNBUFFERED=1 \
    -e HF_HOME=/tmp/llm4o6-hf-cache \
    -e TRANSFORMERS_CACHE=/tmp/llm4o6-hf-cache \
    -e TRTLLM_ADAPTIVE_FP4="${TRTLLM_ADAPTIVE_FP4}" \
    -e TRTLLM_ADAPTIVE_FP4_FC2="${TRTLLM_ADAPTIVE_FP4_FC2}" \
    -e TRTLLM_ADAPTIVE_FP4_WEIGHT="${TRTLLM_ADAPTIVE_FP4_WEIGHT}" \
    -e TRTLLM_ADAPTIVE_FP4_SO="${ADAPTIVE_FP4_SO}" \
    -e TRTLLM_MOE_FORCE_CUTEDSL="${TRTLLM_MOE_FORCE_CUTEDSL}" \
    -e TRTLLM_ADAPTIVE_FP4_DEBUG="${TRTLLM_ADAPTIVE_FP4_DEBUG}" \
    -e TRTLLM_4O6_LOAD_TIMING="${TRTLLM_4O6_LOAD_TIMING}" \
    -e TRTLLM_4O6_LOAD_TIMING_RANKS="${TRTLLM_4O6_LOAD_TIMING_RANKS}" \
    -e TRTLLM_4O6_FORCE_LAZY=1 \
    -e TRTLLM_4O6_LAZY_PREFETCH=1 \
    -e TRTLLM_4O6_LAZY_PREFETCH_WORKERS="${TRTLLM_4O6_LAZY_PREFETCH_WORKERS:-4}" \
    -e TRTLLM_4O6_LAZY_PREFETCH_ORDER="${TRTLLM_4O6_LAZY_PREFETCH_ORDER:-demand}" \
    -e TRTLLM_4O6_LAZY_PREFETCH_WAIT_FOR_TENSOR="${TRTLLM_4O6_LAZY_PREFETCH_WAIT_FOR_TENSOR:-0}" \
    -e TRTLLM_EVAL_MAX_INFLIGHT="${TRTLLM_EVAL_MAX_INFLIGHT}" \
    -e TLLM_DISABLE_ALLREDUCE_AUTOTUNE="${TLLM_DISABLE_ALLREDUCE_AUTOTUNE}" \
    -e TRTLLM_ENABLE_PDL="${TRTLLM_ENABLE_PDL}" \
    -e TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE}" \
    -e TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE}" \
    -v /home/scratch.georgel_gpu:/home/scratch.georgel_gpu \
    -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
    -v /home/scratch.trt_llm_data/llm-models:/llm-models:ro \
    -w "${REPO}" \
    "${IMAGE}" \
    bash -lc "
set -euo pipefail
mkdir -p /tmp/llm4o6-home /tmp/torchinductor-georgel /tmp/llm4o6-hf-cache '${OUT_ROOT}'
PYDEPS=/tmp/llm4o6-pydeps-cli-\${SLURM_JOB_ID:-manual}
rm -rf \"\${PYDEPS}\"
python3 -m pip install --target \"\${PYDEPS}\" --no-deps \
  lm_eval==0.4.8 \
  jsonlines \
  pytablewriter \
  rouge-score \
  sacrebleu \
  sqlitedict \
  tqdm-multiprocess \
  zstandard \
  word2number \
  more_itertools \
  tenacity \
  nltk \
  portalocker \
  colorama \
  pathvalidate \
  tcolorpy \
  typepy==1.3.5 \
  DataProperty \
  tabledata \
  mbstrdecoder \
  chardet==6.0.0.post1 \
  pytz
export PYTHONPATH=\"\${PYDEPS}:${REPO}:\${PYTHONPATH:-}\"

python3 - <<'PY'
import importlib.util
missing = [m for m in ('lm_eval', 'datasets', 'pandas') if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit('missing python packages: ' + ', '.join(missing))
PY

cat > '${OUT_ROOT}/trtllm_eval_config.yaml' <<'YAML'
cuda_graph_config: null
speculative_config: null
YAML

echo '=== node ==='
hostname
nvidia-smi -L
echo GPUS_PER_NODE='${GPUS_PER_NODE}'
echo CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_LIST}'
echo '=== model ==='
test -d '${MODEL}'
test -f '${MODEL}/model.safetensors.index.json'
test -f '${MODEL}/hf_quant_config.json'
du -sh '${MODEL}' || true
echo '=== datasets ==='
test -d '${GSM8K_DATASET_PATH}'
test -d '${MMLU_DATASET_PATH}'
echo GSM8K_DATASET_PATH='${GSM8K_DATASET_PATH}'
echo MMLU_DATASET_PATH='${MMLU_DATASET_PATH}'
echo TRTLLM_EVAL_MAX_INFLIGHT='${TRTLLM_EVAL_MAX_INFLIGHT}'
echo TLLM_DISABLE_ALLREDUCE_AUTOTUNE='${TLLM_DISABLE_ALLREDUCE_AUTOTUNE}'
echo TRTLLM_ENABLE_PDL='${TRTLLM_ENABLE_PDL}'
echo MAX_SEQ_LEN='${MAX_SEQ_LEN}'
echo TORCHDYNAMO_DISABLE='${TORCHDYNAMO_DISABLE}'
echo TORCH_COMPILE_DISABLE='${TORCH_COMPILE_DISABLE}'
echo TRTLLM_ADAPTIVE_FP4='${TRTLLM_ADAPTIVE_FP4}'
echo TRTLLM_ADAPTIVE_FP4_FC2='${TRTLLM_ADAPTIVE_FP4_FC2}'
echo TRTLLM_ADAPTIVE_FP4_WEIGHT='${TRTLLM_ADAPTIVE_FP4_WEIGHT}'
echo TRTLLM_MOE_FORCE_CUTEDSL='${TRTLLM_MOE_FORCE_CUTEDSL}'
echo TRTLLM_ADAPTIVE_FP4_DEBUG='${TRTLLM_ADAPTIVE_FP4_DEBUG}'
echo TRTLLM_4O6_LOAD_TIMING='${TRTLLM_4O6_LOAD_TIMING}'
echo TRTLLM_4O6_LOAD_TIMING_RANKS='${TRTLLM_4O6_LOAD_TIMING_RANKS}'

base_args=(
  trtllm-eval
  --model '${MODEL}'
  --backend pytorch
  --tp_size '${TP_SIZE}'
  --pp_size 1
  --ep_size '${EP_SIZE}'
  --gpus_per_node '${GPUS_PER_NODE}'
  --max_batch_size '${MAX_BATCH_SIZE}'
  --max_num_tokens '${MAX_NUM_TOKENS}'
  --kv_cache_free_gpu_memory_fraction '${KV_FREE_FRACTION}'
  --trust_remote_code
  --config '${OUT_ROOT}/trtllm_eval_config.yaml'
  --no-telemetry
)

if [[ -n '${MAX_SEQ_LEN}' ]]; then
  base_args+=(--max_seq_len '${MAX_SEQ_LEN}')
fi

if [[ '${RUN_GSM8K}' == '1' ]]; then
  mkdir -p '${OUT_ROOT}/gsm8k'
  SECONDS=0
  \"\${base_args[@]}\" \
    gsm8k \
    --dataset_path '${GSM8K_DATASET_PATH}' \
    --num_samples '${GSM8K_NUM_SAMPLES}' \
    --random_seed '${RANDOM_SEED}' \
    --max_output_length 256 \
    --output_path '${OUT_ROOT}/gsm8k_results' \
    --output_dir '${OUT_ROOT}/gsm8k' \
    2>&1 | tee '${OUT_ROOT}/gsm8k.log'
  echo GSM8K_WALL_SEC=\${SECONDS} | tee '${OUT_ROOT}/gsm8k_time.txt'
fi

if [[ '${RUN_MMLU}' == '1' ]]; then
  mkdir -p '${OUT_ROOT}/mmlu'
  mmlu_args=()
  case \"\$(echo '${MMLU_NUM_SAMPLES}' | tr A-Z a-z)\" in
    ''|full|all|none|-1)
      ;;
    *)
      mmlu_args+=(--num_samples '${MMLU_NUM_SAMPLES}')
      ;;
  esac
  SECONDS=0
  \"\${base_args[@]}\" \
    mmlu \
    --dataset_path '${MMLU_DATASET_PATH}' \
    \"\${mmlu_args[@]}\" \
    --num_fewshot '${MMLU_NUM_FEWSHOT}' \
    --random_seed '${RANDOM_SEED}' \
    --max_input_length 4094 \
    --max_output_length 2 \
    --output_dir '${OUT_ROOT}/mmlu' \
    2>&1 | tee '${OUT_ROOT}/mmlu.log'
  echo MMLU_WALL_SEC=\${SECONDS} | tee '${OUT_ROOT}/mmlu_time.txt'
fi
"

python3 - "${OUT_ROOT}" "${MODEL}" "${GSM8K_NUM_SAMPLES}" "${MMLU_NUM_SAMPLES}" <<'PY'
import json
import re
import sys
from pathlib import Path

out = Path(sys.argv[1])
summary = {
    "model": sys.argv[2],
    "gsm8k_num_samples": sys.argv[3],
    "mmlu_num_samples_arg": sys.argv[4],
    "tasks": {},
}

gsm8k_result = out / "gsm8k_results" / "samples_gsm8k.json"
if gsm8k_result.exists():
    payload = json.loads(gsm8k_result.read_text(encoding="utf-8"))
    scores = payload.get("results", {}).get("gsm8k", {})
    numeric = [v for k, v in scores.items() if isinstance(v, (int, float)) and "_stderr" not in k]
    summary["tasks"]["gsm8k"] = {
        "accuracy": sum(numeric) / len(numeric) if numeric else None,
        "scores": scores,
        "result_file": str(gsm8k_result),
        "status": "ok",
    }

mmlu_log = out / "mmlu.log"
if mmlu_log.exists():
    text = mmlu_log.read_text(errors="replace")
    matches = re.findall(r"MMLU weighted average accuracy:\s*([0-9.]+)\s*\((\d+)\)", text)
    if matches:
        acc, count = matches[-1]
        summary["tasks"]["mmlu"] = {
            "accuracy": float(acc),
            "num_requests": int(count),
            "status": "ok",
        }

summary["status"] = "ok" if summary["tasks"] else "missing_results"
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("SUMMARY_JSON", out / "summary.json", flush=True)
print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
if summary["status"] != "ok":
    raise SystemExit(1)
PY
