<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Routed-MoE NVFP4 Adaptive 4o6 and SVDQuant

End-to-end runtime example: load a dense BF16/FP16 routed Mixture-of-Experts
(MoE) checkpoint with the TensorRT-LLM PyTorch backend, apply NVFP4 adaptive
4-over-6 (4o6) to routed expert weights and activations, and optionally retain
a low-rank SVDQuant correction.

This example targets `nvfp4-4o6-svdq-v1.3.0rc19`. The transform runs while
the model loads and does not rewrite the source checkpoint. Start a fresh
process for every configuration.

---

## Paths

All commands below use these variables. Replace the example values with your
own ready TensorRT-LLM environment and dense model path.

| Variable | Meaning | Example value |
|---|---|---|
| `MODEL` | Dense BF16/FP16 Hugging Face checkpoint | `/models/JoyAI-LLM-Flash` |
| `SERVED_MODEL_NAME` | OpenAI-compatible API model name | `joyai-llm-flash` |

```bash
export MODEL=/models/JoyAI-LLM-Flash
export SERVED_MODEL_NAME=joyai-llm-flash
```

---

## 0. Prerequisites

- A ready TensorRT-LLM build from this branch, with the adaptive FP4 operators
  compiled into the build.
- Blackwell SM100 or SM103. JoyAI-LLM-Flash was validated with TP4 on four
  GB200 GPUs.
- A complete dense BF16/FP16 checkpoint. SVDQuant does not accept an INT4
  compressed-tensors or pre-quantized NVFP4 source checkpoint.
- Vanilla MoE loading with complete local routed-expert ownership.
- Enough host and device memory for FP32 SVD during model loading.

Verify the installed runtime from outside the repository so that the source
tree does not shadow the installed package:

```bash
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

| Model | Adaptive 4o6 evidence | SVDQuant status on this branch |
|---|---|---|
| JoyAI-LLM-Flash | Validated from a dense checkpoint with TP4 on four GB200 GPUs | Validated for FC13 + FC2 at rank 64 |
| Qwen3-30B-A3B | Adaptive weight/activation 4o6 was validated on the rc14 experiment branch | Routed experts have the supported SwiGLU layout, but rc19 SVDQuant has not been model-validated |
| Kimi-K2.5 | The rc14 branch validated an INT4-to-exported-NVFP4 4o6 workflow | Requires dense BF16/FP16 routed-expert weights; the official INT4 or an exported/pre-quantized checkpoint cannot enter this SVDQuant loader, and rc19 SVDQuant has not been model-validated |

The historical rc14 workflows are documented under
`examples/qwen3_nvfp4_4o6/` and `examples/kimi_k2_5_nvfp4_4o6/` on
`experiment/weight-act-adaptive-4o6-rc14`. Their offline converters and
external adaptive-FP4 shared-library setup are not part of this rc19 example.

---

## 2. Checkpoint flow

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

## 6. Adaptive NVFP4 4o6 + SVDQuant

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

## 7. Environment reference

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

## 8. Verify the service

Adaptive weight loading reports:

```text
An in-loader NVFP4 weight transform is enabled
```

SVDQuant additionally reports:

```text
Using SVDQuant NVFP4 MoE fallback (rank=64, fc13=True, fc2=True, ...)
```

After the SVDQuant service is ready, send one ordinary API request:

```bash
curl -sS http://127.0.0.1:8000/v1/models

curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${SERVED_MODEL_NAME}-4o6-svdq\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: smoke\"}],\"temperature\":0,\"max_tokens\":16}"
```

The response must name the requested served model and contain a non-empty
assistant message before running a full evaluation.

---

## 9. Unsupported combinations

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

> **WARNING**: Do not reuse the rc14 Kimi INT4 conversion command for this
> SVDQuant path. SVDQuant requires dense BF16/FP16 routed-expert weights.
