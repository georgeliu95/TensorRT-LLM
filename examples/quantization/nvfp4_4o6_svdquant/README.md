<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routed-MoE NVFP4 Adaptive 4o6 and SVDQuant

End-to-end runtime example: load a routed Mixture-of-Experts (MoE) checkpoint
with the TensorRT-LLM PyTorch backend, apply NVFP4 adaptive 4-over-6 (4o6) to
routed expert weights and activations, and optionally retain a low-rank
SVDQuant correction. The implementation supports dense BF16/FP16 load-time
SVDQuant and offline export of NVFP4 4o6 checkpoints. For Kimi-K2.5, the
offline converter can also derive and persist SVDQuant factors from the native
compressed-tensors INT4 checkpoint.

This example targets `nvfp4-4o6-svdq-v1.3.0rc19`. The in-loader path transforms
dense weights without rewriting the source checkpoint; the offline paths write
an exported 4o6 checkpoint, optionally with persisted SVDQuant factors, for
later inference. Start a fresh process for every runtime configuration.

---

## Paths

All commands below use these variables. Replace the example values with your
own ready TensorRT-LLM environment and model paths.

| Variable | Meaning | Example value |
|---|---|---|
| `REPO` | TensorRT-LLM checkout, needed only by the optional parallel finalizer | `/workspace/TensorRT-LLM` |
| `MODEL` | Source Hugging Face checkpoint | `/models/JoyAI-LLM-Flash` |
| `EXPORTED_4O6` | Offline-converted NVFP4 4o6 checkpoint | `/models/JoyAI-LLM-Flash-4o6-nvfp4` |
| `EXPORTED_SVDQ` | INT4-derived NVFP4 4o6 + persisted SVDQuant checkpoint | `/models/Kimi-K2.5-4o6-nvfp4-svdq-r64` |
| `SERVED_MODEL_NAME` | OpenAI-compatible API model name | `joyai-llm-flash` |
| `PY` | Python from the ready TensorRT-LLM environment | `python3` |

```bash
export REPO=/workspace/TensorRT-LLM
export MODEL=/models/JoyAI-LLM-Flash
export EXPORTED_4O6=/models/JoyAI-LLM-Flash-4o6-nvfp4
export EXPORTED_SVDQ=/models/Kimi-K2.5-4o6-nvfp4-svdq-r64
export SERVED_MODEL_NAME=joyai-llm-flash
export PY=python3
```

---

## 0. Prerequisites

- A ready TensorRT-LLM build from this branch, with the adaptive FP4 operators
  compiled into the build.
- Blackwell SM100 or SM103. JoyAI-LLM-Flash was validated with TP4 on four
  GB200 GPUs.
- A complete source checkpoint. Load-time SVDQuant requires dense BF16/FP16
  weights. The offline converter accepts dense BF16/FP16, existing NVFP4, and
  compressed-tensors INT4 checkpoints; offline SVDQuant export currently
  requires compressed-tensors INT4.
- Vanilla MoE loading with complete local routed-expert ownership.
- Enough host and device memory for FP32 SVD during model loading or offline
  conversion.

Verify the installed runtime from outside the repository so that the source
tree does not shadow the installed package:

```bash
command -v trtllm-convert-4o6-ckpt
cd /tmp
python3 -c 'import inspect, tensorrt_llm, torch; \
print(inspect.getfile(tensorrt_llm)); \
required=("fp4_quantize_ex", "calculate_global_amax", \
"fp4_quantize_fused", "dequant_nvfp4_swizzled_sf"); \
missing=[name for name in required if not hasattr(torch.ops.trtllm, name)]; \
assert not missing, missing'
```

The Python Cutlass package must match the package tree used to build the
wheel. Validate the imported MLIR binding as well as the package version:

```bash
cd /tmp
python3 - <<'PY'
import inspect
from importlib import metadata
from cutlass._mlir.dialects import nvvm

signature = str(inspect.signature(nvvm.fmin))
print("nvidia-cutlass-dsl", metadata.version("nvidia-cutlass-dsl"))
print("nvvm.fmin", signature)
assert signature.startswith("(a, b,"), signature
PY
```

If a dependency directory is exposed through a `.pth` file, add its explicit
`nvidia_cutlass_dsl/python_packages` directory to the environment. Python does
not recursively process a nested `.pth` file. Package version alone is not a
sufficient check because incompatible package trees can report the same
version.

For example, when a clean wheel environment reuses a known-compatible runtime
dependency tree:

```bash
export RUNTIME_DEPS=/path/to/compatible/site-packages
SITE="$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n%s\n' \
    "$RUNTIME_DEPS/nvidia_cutlass_dsl/python_packages" \
    "$RUNTIME_DEPS" \
    > "$SITE/rc19-runtime-dependencies.pth"
```

The TensorRT-LLM wheel does not own benchmark dependencies. An environment
that runs the packaged MMLU or GSM8K evaluator must separately provide
compatible `datasets`, `lm_eval`, and `mpi4py` packages. This is also required
when the wheel is intentionally installed with `pip install --no-deps`:

```bash
cd /tmp
python3 -c 'import datasets, lm_eval, mpi4py, tensorrt_llm; \
print("TensorRT-LLM and evaluation dependencies are importable")'
```

---

## 1. Feature modes

| Mode | Routed expert weights | Routed expert activations | Low-rank correction |
|---|---|---|---|
| BF16 control | BF16/FP16 | BF16/FP16 | No |
| Adaptive 4o6 | Adaptive NVFP4 4o6 | Adaptive NVFP4 4o6 for FC13 and FC2 | No |
| Adaptive 4o6 + SVDQuant | Adaptive NVFP4 4o6 residual | Adaptive NVFP4 4o6 for FC13 and FC2 | Rank-`r` factors |

`FC13` is the fused gate/up projection (`w1` and `w3`), and `FC2` is the down
projection (`w2`). FC13 SVDQuant currently supports SwiGLU only. Attention,
dense MLPs, and shared experts are not transformed by this path.

### Model applicability

| Model | 4o6 status | SVDQuant status on this branch |
|---|---|---|
| JoyAI-LLM-Flash | Validated from a dense checkpoint with TP4 on four GB200 GPUs; the BF16 offline-conversion path uses the same routed-expert keys | Validated for FC13 + FC2 at rank 64 |
| Qwen3-30B-A3B | Uses the supported gate/up/down routed-expert layout; the adaptive and offline-export paths apply, but the rc19 port has not had a full-model accuracy run | Routed experts have the supported SwiGLU layout, but rc19 SVDQuant has not been model-validated |
| Kimi-K2.5 | INT4 compressed-tensors conversion and exported NVFP4 4o6 loading were validated on rc19 | Validated with an offline INT4-derived, persisted rank-64 FC13 + FC2 checkpoint using TP4/EP4 on four GB200 GPUs; see the supplementary benchmark results below. Direct load-time SVDQuant still requires dense BF16/FP16 weights |

The offline conversion workflow was ported from the historical rc14 examples
under `examples/qwen3_nvfp4_4o6/` and
`examples/kimi_k2_5_nvfp4_4o6/` on
`experiment/weight-act-adaptive-4o6-rc14`. This branch carries the converter as
the installed `trtllm-convert-4o6-ckpt` command and the optional large-model
worker consolidator as `scripts/finalize_parallel_4o6_nvfp4.py`.

---

## 2. Checkpoint flows

Adaptive 4o6 starts from the dense checkpoint and transforms selected routed
expert weights to adaptive NVFP4 in memory during model loading.

SVDQuant uses this load order for every selected projection:

1. Load the dense BF16/FP16 expert weight.
2. Compute FP32 SVD and retain rank-`r` factors.
3. Quantize the residual with adaptive NVFP4 4o6.
4. Run the normal rc19 NVFP4 weight finalization.
5. Copy TP-local low-rank factors after finalization succeeds.

At runtime, CuteDSL evaluates the NVFP4 residual path and adds the low-rank
correction at the FC13 and FC2 boundaries. No converted checkpoint is written;
every fresh process repeats the transform.

Plain offline export is a separate path:

1. Read dense BF16/FP16, existing NVFP4, or compressed-tensors INT4 weights.
2. Write selected projections as an NVFP4 4o6 Hugging Face checkpoint.
3. Load the exported checkpoint directly in later inference processes.

Offline INT4-derived SVDQuant export is a third path:

1. Dequantize each selected projection from its native INT4 representation.
2. Compute FP32 SVD and retain rank-`r` BF16 factors.
3. Quantize the residual as NVFP4 4o6.
4. Persist the residual, factors, and immutable SVDQuant metadata together.
5. Load the artifact with matching runtime rank, dtype, and stage flags; no SVD
   decomposition or adaptive weight re-quantization occurs during model load.

Do not enable load-time SVDQuant on a plain exported 4o6 checkpoint. Use a
dense checkpoint for load-time SVDQuant, or create a recognized persisted
SVDQuant artifact with the offline converter.

---

## 3. BF16 control

Create a configuration without `quant_config`:

```bash
cat > /tmp/moe_bf16.yaml <<'YAML'
backend: pytorch
dtype: bfloat16
tensor_parallel_size: 4
pipeline_parallel_size: 1
gpus_per_node: 4
moe_config:
  backend: TRTLLM
kv_cache_config:
  dtype: auto
max_batch_size: 4
max_num_tokens: 4096
max_seq_len: 8192
YAML
```

Start the BF16 control with every adaptive and SVDQuant switch disabled:

```bash
TRTLLM_MOE_FORCE_CUTEDSL=0 \
TRTLLM_ADAPTIVE_FP4=0 \
TRTLLM_ADAPTIVE_FP4_FC2=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2=0 \
TRTLLM_SVDQUANT_NVFP4=0 \
TRTLLM_SVDQUANT_FC13=0 \
TRTLLM_SVDQUANT_FC2=0 \
python3 -m tensorrt_llm.commands.serve serve "$MODEL" \
    --config /tmp/moe_bf16.yaml \
    --tokenizer "$MODEL" \
    --backend pytorch \
    --tp_size 4 \
    --gpus_per_node 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --served_model_name "${SERVED_MODEL_NAME}-bf16" \
    --trust_remote_code \
    --no-telemetry
```

---

## 4. Shared NVFP4 configuration

Adaptive 4o6 and SVDQuant use the same conservative serving configuration.
`cuda_graph_config: null` is required by the current SVDQuant correction path.

```bash
cat > /tmp/moe_nvfp4_4o6.yaml <<'YAML'
backend: pytorch
dtype: bfloat16
tensor_parallel_size: 4
pipeline_parallel_size: 1
gpus_per_node: 4
moe_config:
  backend: CUTEDSL
quant_config:
  quant_algo: NVFP4
  group_size: 16
cuda_graph_config: null
kv_cache_config:
  dtype: auto
max_batch_size: 4
max_num_tokens: 4096
max_seq_len: 8192
YAML
```

---

## 5. Adaptive NVFP4 4o6

This recipe enables adaptive 4o6 for FC13 and FC2 weights, the routed-MoE
input activation, and the FC13-to-FC2 intermediate activation.

```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE=mse \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FALLBACK_SCALE_RULE=standard \
TRTLLM_SVDQUANT_NVFP4=0 \
TRTLLM_SVDQUANT_FC13=0 \
TRTLLM_SVDQUANT_FC2=0 \
python3 -m tensorrt_llm.commands.serve serve "$MODEL" \
    --config /tmp/moe_nvfp4_4o6.yaml \
    --tokenizer "$MODEL" \
    --backend pytorch \
    --tp_size 4 \
    --gpus_per_node 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --served_model_name "${SERVED_MODEL_NAME}-4o6" \
    --trust_remote_code \
    --no-telemetry
```

`TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13` is retained as a compatibility alias. New
recipes should also set `TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31` for the fused gate/up
weight stage. FC2 activation currently uses a dequantize-to-BF16 then
adaptive-requantize correctness path rather than a fused epilogue.

---

## 6. Export and load an NVFP4 4o6 checkpoint

This avoids re-quantizing the routed expert weights on every model start. The
converter is installed with the TensorRT-LLM wheel and does not require a
source checkout. For JoyAI-LLM-Flash and Qwen3-30B-A3B, the default selector
converts routed expert `gate_proj`, `up_proj`, and `down_proj` weights:

```bash
cd /tmp
trtllm-convert-4o6-ckpt \
    --input "$MODEL" \
    --output "$EXPORTED_4O6" \
    --source-format bf16 \
    --quant-backend trtllm \
    --activation-mode 4o6 \
    --scale-rule mse \
    --overwrite
```

For a dense model such as Qwen3-8B, add `--all-linear` and
`--exclude-regex '.*(embed_tokens|lm_head).*'`. For a Kimi-K2.5 compressed-
tensors checkpoint, use `--source-format int4-compressed-tensors` and
optionally `--int4-unpack-device cuda`; do not add `--all-linear`. Large
checkpoints can be split with `--include-regex`, `--skip-untargeted-copy`, and
`--shard-prefix`, then consolidated with
`"$PY" "$REPO/scripts/finalize_parallel_4o6_nvfp4.py" --worker-dirs ...` as in
the rc14 Kimi workflow. Only this optional parallel consolidation step still
requires a checkout.

Confirm that the output is a recognized exported 4o6 checkpoint:

```bash
test -f "$EXPORTED_4O6/model.safetensors.index.json"
"$PY" - "$EXPORTED_4O6/hf_quant_config.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["quantization"]["quant_algo"] == "NVFP4"
assert payload["producer"]["name"] in {
    "llm_4o6.convert_ckpt_to_4o6_nvfp4",
    "llm_4o6.finalize_parallel_4o6_nvfp4",
}
print(payload["producer"]["name"])
PY
```

Serve the converted checkpoint with adaptive activations enabled and the
in-loader weight transform disabled:

```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2=0 \
TRTLLM_SVDQUANT_NVFP4=0 \
TRTLLM_SVDQUANT_FC13=0 \
TRTLLM_SVDQUANT_FC2=0 \
python3 -m tensorrt_llm.commands.serve serve "$EXPORTED_4O6" \
    --config /tmp/moe_nvfp4_4o6.yaml \
    --tokenizer "$MODEL" \
    --backend pytorch \
    --tp_size 4 \
    --gpus_per_node 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --served_model_name "${SERVED_MODEL_NAME}-exported-4o6" \
    --trust_remote_code \
    --no-telemetry
```

The loader reads `hf_quant_config.json`, preserves the exported-4o6 metadata,
and does not re-encode the already converted expert weights.

### Export INT4-derived SVDQuant factors for Kimi-K2.5

Set `MODEL` to the native compressed-tensors INT4 checkpoint, then export the
rank-64 FC13 + FC2 residual and factors in one conversion. Also select a Kimi
service name for the later launch:

```bash
export SERVED_MODEL_NAME=kimi-k25
cd /tmp
trtllm-convert-4o6-ckpt \
    --input "$MODEL" \
    --output "$EXPORTED_SVDQ" \
    --source-format int4-compressed-tensors \
    --int4-unpack-device cuda \
    --quant-backend trtllm \
    --activation-mode 4o6 \
    --scale-rule mse \
    --svdquant-rank 64 \
    --svdquant-factor-dtype bfloat16 \
    --svdquant-device cuda \
    --overwrite
```

For a split Kimi conversion, pass the same `--svdquant-rank 64`,
`--svdquant-factor-dtype bfloat16`, and `--svdquant-device cuda` arguments to
every converter worker. The finalizer must also receive the artifact metadata:

```bash
"$PY" "$REPO/scripts/finalize_parallel_4o6_nvfp4.py" \
    --input "$MODEL" \
    --worker-dirs /models/worker-0 /models/worker-1 ... \
    --output "$EXPORTED_SVDQ" \
    --max-shard-size 2GB \
    --non-target-shard-prefix base \
    --non-target-mode rewrite \
    --svdquant-rank 64 \
    --svdquant-factor-dtype bfloat16
```

Confirm that the output metadata describes the immutable artifact contract:

```bash
"$PY" - "$EXPORTED_SVDQ/hf_quant_config.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["quantization"]["quant_algo"] == "NVFP4"
assert payload["svdquant"] == {
    "factor_dtype": "bfloat16",
    "format": "int4-derived-offline-v1",
    "rank": 64,
    "reference": "dequantized-native-int4",
    "source_format": "int4-compressed-tensors",
    "stages": ["fc13", "fc2"],
}
PY
```

---

## 7. Adaptive NVFP4 4o6 + SVDQuant

### Dense load-time SVDQuant

Start a fresh process and enable the rank-64 FC13 + FC2 correction. Keep the
adaptive weight switches enabled because they quantize the SVD residual.

```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE=mse \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FALLBACK_SCALE_RULE=standard \
TRTLLM_SVDQUANT_NVFP4=1 \
TRTLLM_SVDQUANT_RANK=64 \
TRTLLM_SVDQUANT_US_DTYPE=bf16 \
TRTLLM_SVDQUANT_DEVICE=cuda \
TRTLLM_SVDQUANT_FC13=1 \
TRTLLM_SVDQUANT_FC2=1 \
python3 -m tensorrt_llm.commands.serve serve "$MODEL" \
    --config /tmp/moe_nvfp4_4o6.yaml \
    --tokenizer "$MODEL" \
    --backend pytorch \
    --tp_size 4 \
    --gpus_per_node 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --served_model_name "${SERVED_MODEL_NAME}-4o6-svdq" \
    --trust_remote_code \
    --no-telemetry
```

SVD decomposition increases model-load time and memory. The low-rank
correction is a correctness and accuracy-validation path, not a fused
production kernel.

### Persisted INT4-derived SVDQuant

Serve the offline artifact with adaptive activation 4o6 and SVDQuant enabled.
Keep every adaptive weight switch disabled because the checkpoint already
contains the NVFP4 4o6 residual. The runtime flags must exactly match the
artifact rank, factor dtype, and FC13/FC2 stages.

```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_4O6_FORCE_LAZY=1 \
TRTLLM_4O6_LAZY_PREFETCH=1 \
TRTLLM_4O6_LAZY_PREFETCH_WORKERS=4 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FORCE_REQUANT_EXPORTED=0 \
TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE=mse \
TRTLLM_ADAPTIVE_FP4_WEIGHT_FALLBACK_SCALE_RULE=standard \
TRTLLM_SVDQUANT_NVFP4=1 \
TRTLLM_SVDQUANT_RANK=64 \
TRTLLM_SVDQUANT_US_DTYPE=bf16 \
TRTLLM_SVDQUANT_DEVICE=cuda \
TRTLLM_SVDQUANT_FC13=1 \
TRTLLM_SVDQUANT_FC2=1 \
TLLM_DISABLE_ALLREDUCE_AUTOTUNE=1 \
TRTLLM_ENABLE_PDL=0 \
TORCHDYNAMO_DISABLE=1 \
TORCH_COMPILE_DISABLE=1 \
TOKENIZERS_PARALLELISM=false \
OMP_NUM_THREADS=8 \
python3 -m tensorrt_llm.commands.serve serve "$EXPORTED_SVDQ" \
    --config /tmp/moe_nvfp4_4o6.yaml \
    --tokenizer "$MODEL" \
    --backend pytorch \
    --tp_size 4 \
    --ep_size 4 \
    --gpus_per_node 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --served_model_name "${SERVED_MODEL_NAME}-4o6-svdq-r64" \
    --trust_remote_code \
    --no-telemetry
```

The lazy loader and prefetch settings are the validated Kimi-K2.5 load path.
The all-reduce autotuner, PDL, TorchDynamo, and `torch.compile` settings above
keep the current Python low-rank correction outside unsupported capture and
compilation paths.

---

## 8. Environment reference

| Variable | Default | Meaning |
|---|---:|---|
| `TRTLLM_MOE_FORCE_CUTEDSL` | `0` | Route the selected NVFP4 MoE execution through CuteDSL. |
| `TRTLLM_ADAPTIVE_FP4` | `0` | Enable adaptive 4o6 for the FC13 input activation. |
| `TRTLLM_ADAPTIVE_FP4_FC2` | `0` | Enable adaptive 4o6 for the FC2 input activation. |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT` | `0` | Master switch for adaptive routed-MoE weight quantization. |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31` | master | Gate/up weight-stage override. |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2` | master | Down-projection weight-stage override. |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE` | `mse` | Weight rule: `mse`, `mae`, `abs_max`, or numeric `1`, `2`, `3`. |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_FALLBACK_SCALE_RULE` | `standard` | Fallback rule: `standard`, `mse`, `mae`, or `abs_max`. |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_FORCE_REQUANT_EXPORTED` | `0` | Re-quantize an already exported 4o6 checkpoint. Keep disabled for persisted SVDQuant. |
| `TRTLLM_4O6_FORCE_LAZY` | `0` | Force the checkpoint loader's lazy-tensor path. Used by the validated Kimi configuration. |
| `TRTLLM_4O6_LAZY_PREFETCH` | `1` | Prefetch lazy checkpoint tensors. |
| `TRTLLM_4O6_LAZY_PREFETCH_WORKERS` | `4` | Number of lazy-prefetch workers. |
| `TRTLLM_SVDQUANT_NVFP4` | `0` | Enable the SVDQuant routed-MoE path. |
| `TRTLLM_SVDQUANT_RANK` | `64` | Positive low-rank dimension. |
| `TRTLLM_SVDQUANT_US_DTYPE` | `bf16` | Factor storage: `bf16`, `fp16`, or `fp32`. |
| `TRTLLM_SVDQUANT_DEVICE` | `cuda` | SVD compute device: `cuda` or `cpu`. |
| `TRTLLM_SVDQUANT_FC13` | master | Enable gate/up low-rank correction. |
| `TRTLLM_SVDQUANT_FC2` | master | Enable down-projection low-rank correction. |

---

## 9. Verify the service

Adaptive weight loading reports:

```text
An in-loader NVFP4 weight transform is enabled
```

SVDQuant additionally reports:

```text
Using SVDQuant NVFP4 MoE fallback (rank=64, fc13=True, fc2=True, ...)
```

The persisted Kimi path instead reports:

```text
Loading persisted INT4-derived SVDQuant factors (rank=64, fc13=True, fc2=True, experts=96).
```

After the selected service is ready, set its served name and send one ordinary
API request. This example selects the exported-checkpoint service from section
6; use the matching name for the in-loader or SVDQuant service instead.

```bash
export API_MODEL="${SERVED_MODEL_NAME}-exported-4o6"

curl -sS http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${API_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: smoke\"}],\"temperature\":0,\"max_tokens\":16}"
```

The response must name the requested served model and contain a non-empty
assistant message before running a full evaluation.

---

## 10. Kimi-K2.5 supplementary benchmark

The following results validate the persisted INT4-derived SVDQuant path. They
were produced from the same rank-64 FC13 + FC2 artifact with BF16 factors,
NVFP4 4o6 residual weights, adaptive 4o6 activations, CuteDSL, TP4/EP4 on four
GB200 GPUs, `cuda_graph_config: null`, and a clean packaged wheel built from
the experiment source tree based on commit
`22f328c057d0b949b8959f8c65cadd4134c97df3` (wheel SHA256
`e85449b9d250fa654beeeb82a763068ffd568c1a693e6c557a01127d79e82792`).

| Benchmark | Evaluation protocol | Samples | Result (%) |
|---|---|---:|---:|
| MMLU | 57 subjects, 5-shot, weighted average accuracy, seed 0, max output length 2 | 14,042 | **89.37** |
| GSM8K | 5-shot, strict-match and flexible-extract exact match, seed 0, max output length 256 | 1,319 | **93.48** |
| MBPP | OpenCompass code evaluator, max output length 2,048 | 500 | **64.60** (323 pass, 161 failed, 16 wrong answer, 0 timeout) |

MMLU and GSM8K used maximum batch size 16. The MBPP request driver used batch
size 16 with four workers, while the service used maximum batch size 4. For
the same MBPP harness and serving configuration, the exported 4o6 checkpoint
without SVDQuant scored 69.40 (347/500 pass), so the SVDQuant result was 4.80
percentage points lower. These are supplementary engineering results, not a
general model-quality guarantee; prompt templates, evaluator versions, and
decoding settings affect the scores.

MMLU and GSM8K can run from the TensorRT-LLM evaluation environment. MBPP code
scoring is a third-party OpenCompass dependency and should be isolated from the
serving process when it uses CPU-only PyTorch. Run inference in the
TensorRT-LLM wheel environment, and run the CPU evaluator in a separate process
or environment with CUDA disabled and a writable temporary directory. Verify
that evaluator before a full run:

```bash
CUDA_VISIBLE_DEVICES= python3 - <<'PY'
import torch
assert not torch.cuda.is_available()
from opencompass.cli.main import main
print("OpenCompass CPU evaluator ready")
PY
```

Do not prepend a CPU PyTorch installation to the TensorRT-LLM serving process,
and do not allow a CUDA `torchvision` package to shadow the CPU evaluator. This
separation is an evaluation-environment requirement, not a TensorRT-LLM source
or wheel-build change.

---

## 11. Unsupported combinations

Direct dense load-time SVDQuant rejects these combinations before partially
mutating a module:

- a pre-quantized NVFP4 checkpoint with `weight_scale` or `input_scale` keys;
- an INT4 compressed-tensors checkpoint passed directly to model loading;
- fused or partial MoE loading, RLHF reload, or a second load;
- online EPLB, shared-expert remapping, or ConfigurableMoE VA-DWDP;
- missing or duplicated local experts, incomplete projection pairs, or an
  unsupported tensor shape or dtype;
- FC13 with an activation other than SwiGLU;
- an SVD rank larger than the expert projection dimensions.

The current low-rank correction is not CUDA-graph compatible. Keep
`cuda_graph_config: null` for SVDQuant. `TRTLLM_SVDQUANT_DEVICE=cpu` is a
slower fallback that can reduce decomposition pressure on the GPU.

Offline INT4-derived SVDQuant supports only compressed-tensors INT4 input,
rank-matched BF16 factors, and FC13 + FC2 together. A persisted artifact is
rejected if its factor keys are incomplete or its format, rank, dtype, stages,
source format, or reference metadata conflicts with the runtime flags.

---

## Common pitfalls

> **WARNING**: Run Python and TensorRT-LLM commands from outside the repository
> checkout so that the source tree does not shadow the intended runtime.

> **WARNING**: Disabling only `TRTLLM_SVDQUANT_NVFP4` leaves adaptive 4o6
> enabled. That is the no-SVDQuant control, not the BF16 baseline.

> **WARNING**: Use a fresh process for BF16, adaptive 4o6, and SVDQuant. The
> in-memory weight transform is performed during every model load.

> **WARNING**: A plain exported 4o6 checkpoint is not an SVDQuant input. Use a
> dense BF16/FP16 checkpoint for load-time SVDQuant or an offline artifact that
> contains recognized persisted SVDQuant factors and metadata.
