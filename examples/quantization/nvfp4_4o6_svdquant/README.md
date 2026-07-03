<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routed-MoE NVFP4 Adaptive 4o6 and SVDQuant

End-to-end runtime example: load a dense BF16/FP16 routed Mixture-of-Experts
(MoE) checkpoint with the TensorRT-LLM PyTorch backend, apply NVFP4 adaptive
4-over-6 (4o6) to routed expert weights and activations, and optionally retain
a low-rank SVDQuant correction. The same implementation can also export the
4o6 expert weights once and load the converted checkpoint in later processes.

This example targets `nvfp4-4o6-svdq-v1.3.0rc19`. The in-loader path transforms
weights without rewriting the source checkpoint; the offline path writes an
exported 4o6 checkpoint for later inference. Start a fresh process for every
runtime configuration.

---

## Paths

All commands below use these variables. Replace the example values with your
own ready TensorRT-LLM environment and model paths.

| Variable | Meaning | Example value |
|---|---|---|
| `REPO` | TensorRT-LLM checkout, needed only by the optional parallel finalizer | `/workspace/TensorRT-LLM` |
| `MODEL` | Source Hugging Face checkpoint | `/models/JoyAI-LLM-Flash` |
| `EXPORTED_4O6` | Offline-converted NVFP4 4o6 checkpoint | `/models/JoyAI-LLM-Flash-4o6-nvfp4` |
| `SERVED_MODEL_NAME` | OpenAI-compatible API model name | `joyai-llm-flash` |
| `PY` | Python from the ready TensorRT-LLM environment | `python3` |

```bash
export REPO=/workspace/TensorRT-LLM
export MODEL=/models/JoyAI-LLM-Flash
export EXPORTED_4O6=/models/JoyAI-LLM-Flash-4o6-nvfp4
export SERVED_MODEL_NAME=joyai-llm-flash
export PY=python3
```

---

## 0. Prerequisites

- A ready TensorRT-LLM build from this branch, with the adaptive FP4 operators
  compiled into the build.
- Blackwell SM100 or SM103. JoyAI-LLM-Flash was validated with TP4 on four
  GB200 GPUs.
- A complete source checkpoint. The in-loader and SVDQuant paths require dense
  BF16/FP16 weights; the offline converter additionally accepts existing NVFP4
  and compressed-tensors INT4 checkpoints.
- Vanilla MoE loading with complete local routed-expert ownership.
- Enough host and device memory for FP32 SVD during model loading.

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
| Kimi-K2.5 | INT4 compressed-tensors conversion and exported NVFP4 4o6 loading were validated on rc19; the packaged runtime completed a TP4 MMLU run without an editable install or source-tree import | Requires a dense BF16/FP16 source; INT4 and exported/pre-quantized checkpoints cannot enter SVDQuant, and rc19 SVDQuant has not been model-validated |

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

Offline export is a separate path:

1. Read dense BF16/FP16, existing NVFP4, or compressed-tensors INT4 weights.
2. Write selected projections as an NVFP4 4o6 Hugging Face checkpoint.
3. Load the exported checkpoint directly in later inference processes.

Use either the in-loader weight transform or the exported checkpoint. Do not
feed the exported checkpoint to SVDQuant; SVDQuant must decompose dense weights
before the residual is quantized.

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
build_config:
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
build_config:
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

---

## 7. Adaptive NVFP4 4o6 + SVDQuant

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

## 10. Unsupported combinations

SVDQuant rejects these combinations before partially mutating a module:

- a pre-quantized NVFP4 checkpoint with `weight_scale` or `input_scale` keys;
- an INT4 compressed-tensors checkpoint;
- fused or partial MoE loading, RLHF reload, or a second load;
- online EPLB, shared-expert remapping, or ConfigurableMoE VA-DWDP;
- missing or duplicated local experts, incomplete projection pairs, or an
  unsupported tensor shape or dtype;
- FC13 with an activation other than SwiGLU;
- an SVD rank larger than the expert projection dimensions.

The current low-rank correction is not CUDA-graph compatible. Keep
`cuda_graph_config: null` for SVDQuant. `TRTLLM_SVDQUANT_DEVICE=cpu` is a
slower fallback that can reduce decomposition pressure on the GPU.

---

## Common pitfalls

> **WARNING**: Run Python and TensorRT-LLM commands from outside the repository
> checkout so that the source tree does not shadow the intended runtime.

> **WARNING**: Disabling only `TRTLLM_SVDQUANT_NVFP4` leaves adaptive 4o6
> enabled. That is the no-SVDQuant control, not the BF16 baseline.

> **WARNING**: Use a fresh process for BF16, adaptive 4o6, and SVDQuant. The
> in-memory weight transform is performed during every model load.

> **WARNING**: An exported 4o6 checkpoint is an alternative to the in-loader
> weight transform, not an SVDQuant input. SVDQuant requires dense BF16/FP16
> routed-expert weights.
