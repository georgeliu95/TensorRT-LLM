<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Kimi-K2.5 INT4 to NVFP4 4o6 - Accuracy Evaluation Example

End-to-end example: convert a Kimi-K2.5 INT4 compressed-tensors checkpoint to
an exported NVFP4 4-over-6 (4o6) checkpoint, then use that converted checkpoint
with the TensorRT-LLM PyTorch backend.

This follows the same workflow shape as `examples/qwen3_nvfp4_4o6`:

1. Set paths.
2. Use the branch converter from `convert_ckpt_to_4o6_nvfp4.py`.
3. Validate the exported 4o6 checkpoint metadata.

Kimi differs from Qwen3-8B in the conversion command. Qwen3-8B is a dense BF16
model and uses one `convert_ckpt_to_4o6_nvfp4.py --source-format bf16
--all-linear` command. Kimi-K2.5 starts from INT4 packed MoE tensors, so this
example runs eight explicit `convert_ckpt_to_4o6_nvfp4.py` workers over layer
slices and then consolidates them with `finalize_parallel_4o6_nvfp4.py`.

---

## Paths

Use the checked-in conversion scripts under this TensorRT-LLM checkout.

```bash
export REPO=/home/user/workspace/customer/TensorRT-LLM

export KIMI_INT4_MODEL=/llm-models/Kimi-K2.5
export OUT_4O6=/home/user/workspace/Kimi-K2.5-4o6-nvfp4
export TRTLLM_ADAPTIVE_FP4_SO=/home/user/workspace/tmp/libfp4QuantizeAdaptive_sm100a.so
export PY=/usr/bin/python3

test -f "$REPO/scripts/convert_ckpt_to_4o6_nvfp4.py"
test -f "$REPO/scripts/finalize_parallel_4o6_nvfp4.py"
test -f "$TRTLLM_ADAPTIVE_FP4_SO"
```

---

## 0. Prerequisites

- One 8-GPU Blackwell node for the parallel Kimi conversion.
- A Kimi-K2.5 INT4 compressed-tensors checkpoint with `.weight_packed`,
  `.weight_scale`, and `.weight_shape` tensors.
- A TensorRT-LLM wheel from this branch installed in the container.
- `convert_ckpt_to_4o6_nvfp4.py` and `finalize_parallel_4o6_nvfp4.py` available
  under `$REPO/scripts`.
- The adaptive FP4 helper shared object available at `$TRTLLM_ADAPTIVE_FP4_SO`.

The default Kimi conversion below uses `NON_TARGET_MODE=symlink-source`. This is
the fastest mode, but the converted checkpoint contains symlinks to the source
checkpoint shards. Keep `$KIMI_INT4_MODEL` mounted at the same absolute path for
load/eval. Set `NON_TARGET_MODE=rewrite` for a self-contained checkpoint.

---

## 1. Build and Install the Wheel

Use the wheel build/install flow from `examples/qwen3_nvfp4_4o6`. The important
requirements are:

- Build against the container's exact torch/CUDA stack.
- Install the wheel with `--no-deps`.
- Run verification from a non-repo directory.

Save this as `/tmp/kimi_verify_runtime.py`:

```python
import inspect

import tensorrt_llm
import torch

print("tensorrt_llm_version", tensorrt_llm.__version__)
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("tensorrt_llm_file", inspect.getfile(tensorrt_llm))
has_fp4_quantize_ex = hasattr(torch.ops.trtllm, "fp4_quantize_ex")
print("has_fp4_quantize_ex", has_fp4_quantize_ex)
if not has_fp4_quantize_ex:
    raise SystemExit("missing torch.ops.trtllm.fp4_quantize_ex")
```

Run it from a non-repo directory:

```bash
cd /tmp
"$PY" /tmp/kimi_verify_runtime.py
```

---

## 2. Install the Evaluation Dependency

Install `lm_eval` the same way as the Qwen3 example, preserving the container
torch package:

Save this as `/tmp/kimi_eval_constraints.py`:

```python
import torch

print(f"torch=={torch.__version__}")
print("transformers==4.57.3")
print("datasets==3.1.0")
```

```bash
"$PY" /tmp/kimi_eval_constraints.py > /tmp/c.txt
sudo "$PY" -m pip install --break-system-packages -c /tmp/c.txt 'lm_eval[api]==0.4.10'
```

---

## 3. Convert Kimi-K2.5 INT4 to NVFP4 4o6

Do not use the Qwen dense command for Kimi. In particular:

- Do not pass `--source-format bf16`.
- Do not pass `--all-linear`.

The documented conversion entrypoint is `convert_ckpt_to_4o6_nvfp4.py`. The
commands below expand the parallel Kimi conversion into eight direct converter
invocations, followed by `finalize_parallel_4o6_nvfp4.py`.

### 3a. Preflight the Source and Runtime

Save this as `/tmp/kimi_k2_5_preflight.py`:

```python
import json
import os
import struct
from pathlib import Path

import tensorrt_llm
import torch

root = Path(os.environ["KIMI_INT4_MODEL"])
shards = sorted(root.glob("*.safetensors"))
print("source_path", root)
print("num_safetensors", len(shards))
cfg = json.loads((root / "config.json").read_text())
print("model_type", cfg.get("model_type"))
print("text_model_type", cfg.get("text_config", {}).get("model_type"))

packed = 0
for shard in shards:
    with shard.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    packed += sum(1 for key in header if key.endswith(".weight_packed"))
print("weight_packed_tensors", packed)
if packed == 0:
    raise SystemExit("expected INT4 .weight_packed tensors, got 0")

print("cuda_device_count", torch.cuda.device_count())
print("cuda_devices", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("tensorrt_llm_version", tensorrt_llm.__version__)
has_fp4_quantize_ex = hasattr(torch.ops.trtllm, "fp4_quantize_ex")
print("has_fp4_quantize_ex", has_fp4_quantize_ex)
if torch.cuda.device_count() != 8:
    raise SystemExit("expected exactly 8 visible GPUs for Kimi-K2.5 conversion")
if not has_fp4_quantize_ex:
    raise SystemExit("missing torch.ops.trtllm.fp4_quantize_ex")
```

Run it inside the container:

```bash
set -euo pipefail

test -d "$REPO"
test -d "$REPO/scripts"
test -d "$KIMI_INT4_MODEL"
test -f "$REPO/scripts/convert_ckpt_to_4o6_nvfp4.py"
test -f "$REPO/scripts/finalize_parallel_4o6_nvfp4.py"

cd /tmp
nvidia-smi -L
"$PY" /tmp/kimi_k2_5_preflight.py
```

### 3b. Run Eight Direct Converter Workers

This is the Kimi equivalent of the Qwen `convert_ckpt_to_4o6_nvfp4.py` step.
Each worker calls the same converter directly on one layer slice. The converter
uses its default `--source-format auto` path; the validated run printed
`source_format=int4-compressed-tensors` in every worker log. Worker outputs use
`--skip-untargeted-copy`; the finalizer adds all non-target tensors and writes
the final `config.json`, `model.safetensors.index.json`, and
`hf_quant_config.json`.

```bash
set -euo pipefail

export QUANT_BACKEND="${QUANT_BACKEND:-trtllm}"
export INT4_UNPACK_DEVICE="${INT4_UNPACK_DEVICE:-cuda}"
export NON_TARGET_MODE="${NON_TARGET_MODE:-symlink-source}"
export TENSOR_CACHE_SIZE="${TENSOR_CACHE_SIZE:-128}"
export MAX_SHARD_SIZE="${MAX_SHARD_SIZE:-2GB}"
export JOB_TAG="${JOB_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
export LOG_ROOT="${LOG_ROOT:-$(dirname "$OUT_4O6")/kimi_k2_5_4o6_logs_$JOB_TAG}"
export PARTS_ROOT="${PARTS_ROOT:-${OUT_4O6}_parts_$JOB_TAG}"

test -f "$TRTLLM_ADAPTIVE_FP4_SO"

rm -rf "$PARTS_ROOT" "$OUT_4O6"
mkdir -p "$PARTS_ROOT" "$LOG_ROOT" "$(dirname "$OUT_4O6")"

LAYER_REGEX_PREFIX='^language_model\.model\.layers\.'
LAYERS=(
  "1|2|3|4|5|6|7"
  "8|9|10|11|12|13|14|15"
  "16|17|18|19|20|21|22"
  "23|24|25|26|27|28|29|30"
  "31|32|33|34|35|36|37"
  "38|39|40|41|42|43|44|45"
  "46|47|48|49|50|51|52"
  "53|54|55|56|57|58|59|60"
)

SECONDS=0
PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    part_dir="$PARTS_ROOT/w$i"
    log="$LOG_ROOT/w${i}.log"
    regex="${LAYER_REGEX_PREFIX}(${LAYERS[$i]})\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$"
    rm -rf "$part_dir"

    CUDA_VISIBLE_DEVICES="$i" nohup "$PY" "$REPO/scripts/convert_ckpt_to_4o6_nvfp4.py" \
        --input "$KIMI_INT4_MODEL" \
        --output "$part_dir" \
        --include-regex "$regex" \
        --skip-untargeted-copy \
        --shard-prefix "w$i" \
        --max-shard-size "$MAX_SHARD_SIZE" \
        --tensor-cache-size "$TENSOR_CACHE_SIZE" \
        --int4-unpack-device "$INT4_UNPACK_DEVICE" \
        --trtllm-path "$REPO" \
        --quant-backend "$QUANT_BACKEND" \
        --adaptive-fp4-so "$TRTLLM_ADAPTIVE_FP4_SO" \
        --activation-mode 4o6 \
        --scale-rule mse \
        --overwrite \
        > "$log" 2>&1 &
    PIDS+=($!)
    echo "worker w$i pid=${PIDS[$i]} gpu=$i layers={${LAYERS[$i]}} log=$log"
done

fail=0
for i in 0 1 2 3 4 5 6 7; do
    if wait "${PIDS[$i]}"; then
        echo "[ok] w$i"
    else
        rc=$?
        echo "[fail] w$i exit=$rc; see $LOG_ROOT/w${i}.log"
        fail=1
    fi
done
if [[ "$fail" -ne 0 ]]; then
    exit 1
fi

worker_dirs=()
for i in 0 1 2 3 4 5 6 7; do
    worker_dirs+=("$PARTS_ROOT/w$i")
done

"$PY" "$REPO/scripts/finalize_parallel_4o6_nvfp4.py" \
    --input "$KIMI_INT4_MODEL" \
    --worker-dirs "${worker_dirs[@]}" \
    --output "$OUT_4O6" \
    --max-shard-size "$MAX_SHARD_SIZE" \
    --non-target-mode "$NON_TARGET_MODE" \
    --overwrite \
    2>&1 | tee "$LOG_ROOT/finalize.log"

echo "CONVERSION_WALL_SEC=$SECONDS" | tee "$LOG_ROOT/convert_time.txt"
```

Expected evidence:

```text
source_format=int4-compressed-tensors
worker w0 ...
...
worker w7 ...
[ok] w0
...
[ok] w7
[finalize] moving worker shards from 8 workers
[finalize] unified index: ... tensors / ... shards / total_size=...
[finalize] done: ...
```

The validated Kimi conversion used these values:

```text
QUANT_BACKEND=trtllm
INT4_UNPACK_DEVICE=cuda
NON_TARGET_MODE=symlink-source
TENSOR_CACHE_SIZE=128
MAX_SHARD_SIZE=2GB
TRTLLM_ADAPTIVE_FP4_SO=/code/tmp/libfp4QuantizeAdaptive_sm100a.so
```

### 3c. Validate the Converted Checkpoint

Save this as `/tmp/kimi_k2_5_validate_4o6.py`:

```python
import json
import os
from pathlib import Path

from tensorrt_llm._torch.models.checkpoints.hf.weight_loader import HfWeightLoader

out = Path(os.environ["OUT_4O6"])
index_path = out / "model.safetensors.index.json"
quant_path = out / "hf_quant_config.json"
if not index_path.exists() or not quant_path.exists():
    raise SystemExit("converted checkpoint missing model.safetensors.index.json or hf_quant_config.json")

index = json.loads(index_path.read_text())
qcfg = json.loads(quant_path.read_text())
weight_map = index["weight_map"]
payload = {
    "converted_model": str(out),
    "producer": qcfg.get("producer", {}).get("name"),
    "quantization": qcfg.get("quantization"),
    "is_4o6_exported": HfWeightLoader._is_4o6_exported_checkpoint(str(out)),
    "num_tensors": len(weight_map),
    "num_shards": len(set(weight_map.values())),
    "total_size": index.get("metadata", {}).get("total_size"),
    "num_safetensors_entries": len(list(out.glob("*.safetensors"))),
    "num_symlink_safetensors": sum(1 for p in out.glob("*.safetensors") if p.is_symlink()),
    "num_weight_packed_keys": sum(1 for key in weight_map if key.endswith(".weight_packed")),
    "num_input_scale_tensors": sum(1 for key in weight_map if key.endswith(".input_scale")),
    "num_weight_scale_2_tensors": sum(1 for key in weight_map if key.endswith(".weight_scale_2")),
}
print(json.dumps(payload, indent=2, sort_keys=True))

if payload["producer"] != "llm_4o6.finalize_parallel_4o6_nvfp4":
    raise SystemExit("expected Kimi finalizer producer metadata")
if not payload["is_4o6_exported"]:
    raise SystemExit("converted checkpoint is not recognized as exported 4o6")
if payload["num_weight_packed_keys"] != 0:
    raise SystemExit("converted checkpoint still references INT4 .weight_packed tensors")
if payload["num_input_scale_tensors"] == 0 or payload["num_weight_scale_2_tensors"] == 0:
    raise SystemExit("converted checkpoint is missing 4o6 scale tensors")
```

Run validation from a non-repo directory:

```bash
cd /tmp
"$PY" /tmp/kimi_k2_5_validate_4o6.py
```

For the validated Kimi-K2.5 conversion, the final checkpoint had:

```text
producer=llm_4o6.finalize_parallel_4o6_nvfp4
is_4o6_exported=True
num_weight_packed_keys=0
num_input_scale_tensors=69120
num_weight_scale_2_tensors=69120
num_tensors=277670
num_shards=336
```
