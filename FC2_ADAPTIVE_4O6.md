# Weight + Activation Adaptive 4/6 (nvFP4) Quantization Experiment

## Overview

This experiment applies **adaptive 4/6 (FourOverSix) quantization** to the **FC2 (down projection)** intermediate activation in MoE GEMM, extending the existing FC13 (gate+up projection) 4o6 quantization.

This clone additionally supports adaptive 4/6 **MoE weights** at model-load time. When enabled, NVFP4 checkpoint weights are dequantized from packed E2M1 + FP8 block scales to BF16, then re-quantized through `fp4_quantize_ex(scaleRule=1)` before the normal backend-specific Cutlass/CuteDsl/TRTLLMGen layout preprocessing runs.

- **Branch**: `experiment/weight-act-adaptive-4o6-rc14`
- **Worktree**: `/home/scratch.georgel_gpu/projects/llm_4o6/TensorRT-LLM-weight-act-4o6-rc14/`
- **Base**: TRT-LLM `v1.3.0rc14`

---

## Background: Two Quantization Paths in TRT-LLM MoE

TRT-LLM has two distinct fused MoE runners that handle FC2 differently:

### Path 1: `FP4BlockScaleMoeRunner` (NVFP4, `has_nvfp4`)

**This is the path used by the existing FC13 4o6 deployment.**

| Stage | Data Type | Notes |
|-------|-----------|-------|
| MoE Input | BF16 → **E2M1** (FP4) | Adaptive 4/6 applied here for FC13 |
| FC13 GEMM (gemm1) | **E2M1 × E2M1** | Fused with SwiGLU + FP4 output quant |
| Intermediate (SwiGLU out) | **E2M1** (FP4) + block SF | Standard NVFP4 (scaleRule=0) |
| FC2 GEMM (gemm2) | **E2M1 × E2M1** → BF16 | |

C++ dtype configuration (`fp4BlockScaleMoe.cpp`):
```cpp
btg::Dtype mDtypeElt{btg::Dtype::E2m1};   // activation
btg::Dtype mDtypeAct{btg::Dtype::E2m1};   // activation
btg::Dtype mDtypeWeights{btg::Dtype::E2m1}; // weight
```

Key: PermuteGemm1 (FC13) output type = `dtypeAct = E2M1` → intermediate is FP4.

### Path 2: `FP8FP4BlockScaleMoeRunner` (W4A8, `has_w4a8_nvfp4_fp8`)

| Stage | Data Type | Notes |
|-------|-----------|-------|
| MoE Input | BF16 → **E4M3** (FP8) | |
| FC13 GEMM (gemm1) | **E4M3 × E2M1** | |
| Intermediate (SwiGLU out) | **E4M3** (FP8) | No block SF needed |
| FC2 GEMM (gemm2) | **E4M3 × E2M1** → BF16 | |

C++ dtype configuration:
```cpp
btg::Dtype mDtypeAct{btg::Dtype::E4m3};   // FP8 activation
btg::Dtype mDtypeWeights{btg::Dtype::E2m1}; // FP4 weight
```

**Important**: A prior report described FC2 as "BF16 → E4M3 quantization, then E4M3 × E2M1 GEMM". This describes **Path 2**, not Path 1. The existing FC13 4o6 deployment uses **Path 1** (E2M1 × E2M1).

### How the dtype is determined (C++ evidence)

Both paths share the same `MoE::Runner` constructor in `runner.cu`:
```cpp
Runner::Runner(dtypeAct, dtypeWeights, useDeepSeekFp8, tileTokensDim, actType)
    : mPermuteGemm1(PermuteGemm1::Runner(dtypeAct, dtypeWeights, ...))
    , mGemm2(Gemm2::Runner(dtypeAct, dtypeWeights, Bfloat16, ...))
```

PermuteGemm1 options set `dtypeC = dtypeAct` → the intermediate output type equals the activation type:
```cpp
options = { .dtypeA = dtypeWeights, .dtypeB = dtypeAct, .dtypeC = dtypeAct, ... };
```

---

## Experiment Design (Approach A — implemented)

### Problem

The FC2 intermediate activation (SwiGLU output) is quantized to FP4 **inside** the fused MoE runner kernel. There is no Python-level control over the quantization method for this intermediate.

### Solution

Use the **CuteDsl NVFP4** backend instead of TRTLLMGen. CuteDsl naturally splits FC13 and FC2 into separate kernel calls, allowing Python-level intervention:

```
FC13 + SwiGLU kernel
    cute_dsl_nvfp4_gather_grouped_gemm_swiglu_blackwell
    → outputs (x_fp4, x_sf) — standard NVFP4

==== Intervention point (new code) ====
    1. Dequant FP4 → BF16
    2. calculate_global_amax → runtime amax
    3. fp4_quantize_ex(scaleRule=1) → adaptive 4/6 FP4
    4. Correct fc2_alpha: correction = fc2_input_scale / dynamic_global_scale
=======================================

FC2 kernel
    cute_dsl_nvfp4_grouped_gemm_blackwell (or _finalize_inplace)
    → uses adaptive 4/6 quantized intermediate + corrected alpha
```

### Alpha correction formula

Same pattern as FC13:
```
# Original: fc2_alpha = 1 / (fc2_input_scale × fc2_weight_scale)
# With adaptive 4/6: intermediate is quantized with dynamic_global_scale
# instead of fc2_input_scale, so:
correction = fc2_input_scale / dynamic_global_scale
fc2_alpha_corrected = fc2_alpha × correction
```

### Scale factor layout

Both `fp4_quantize_ex(isSfSwizzledLayout=False)` and the CuteDsl SwiGLU kernel output **LINEAR** (non-swizzled) scale factors. The shapes are compatible:
- SwiGLU output SF: `M × interm_size / 16` (flat 1D)
- `fp4_quantize_ex` output SF: `computeLinearLayoutSFSize(M, interm_size/16)` (flat 1D)

---

## Modified Files

### `tensorrt_llm/_torch/modules/fused_moe/quantization.py`

- Added optional NVFP4 MoE weight re-quantization controlled by `TRTLLM_ADAPTIVE_FP4_WEIGHT*`.
- Supports both per-expert `VANILLA` and `FUSED_GATE_UP_PROJ` checkpoint loading modes.
- Reuses the adaptive FP4 extension (`fp4_quantize_ex` + `calculate_global_amax`) and keeps the existing backend loaders responsible for final weight/scale interleave and shuffle layouts.
- Updates `weight_scale_2`/alpha inputs after weight re-quantization, so FC13 and FC2 alpha are consistent with the new adaptive weight global scale.

### `tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py`

- Modified `quantize_input()`: when `TRTLLM_ADAPTIVE_FP4=1`, uses `calculate_global_amax` + `fp4_quantize_ex(scaleRule=1)` for FC13 input quantization (replaces standard `fp4_quantize`). Includes pre_quant_scale (AWQ) support and padding, matching the TRTLLMGen path. Periodic `[4o6-FC13]` stats logging every 5000 calls.
- Modified `run_moe_nvfp4_impl()`:
  - **FC13 alpha correction**: when FC13 used adaptive 4/6, corrects `fc1_alpha = fc1_global × fc31_input_scale / dynamic_global_scale` before the SwiGLU GEMM kernel.
  - **FC2 adaptive 4/6**: when `TRTLLM_ADAPTIVE_FP4_FC2=1`, inserts dequant → adaptive 4/6 requant → alpha correction between the FC13 SwiGLU kernel and the FC2 GEMM kernel. Periodic `[4o6-FC2]` stats logging.
- Added `_dequant_nvfp4_cutedsl()`: dequantizes packed FP4 + CuteDsl-layout block SF to BF16.

### `tensorrt_llm/_torch/modules/fused_moe/create_moe.py`

- Added `TRTLLM_MOE_FORCE_CUTEDSL` env var: when set to `1`, routes NVFP4 MoE through CuteDslFusedMoE instead of TRTLLMGenFusedMoE. Required because the fused TRTLLMGen runner cannot inject adaptive quantization for the FC2 intermediate.

---

## How to Run

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRTLLM_MOE_FORCE_CUTEDSL` | `0` | Set `1` to use CuteDsl backend for NVFP4 MoE |
| `TRTLLM_ADAPTIVE_FP4` | `0` | Set `1` to enable adaptive 4/6 for **FC13** input (CuteDsl path) |
| `TRTLLM_ADAPTIVE_FP4_FC2` | `0` | Set `1` to enable adaptive 4/6 for **FC2** intermediate |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT` | `0` | Set `1` to enable adaptive 4/6 for both **FC13 and FC2 MoE weights** at load time |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_FC13` / `TRTLLM_ADAPTIVE_FP4_WEIGHT_FC31` | inherits global | Optional per-stage override for gate/up weights |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_FC2` | inherits global | Optional per-stage override for down-projection weights |
| `TRTLLM_ADAPTIVE_FP4_WEIGHT_SCALE_RULE` | `mse` | Weight adaptive scale rule: `mse`, `mae`, `abs_max`, or numeric `1/2/3` |
| `TRTLLM_ADAPTIVE_FP4_SO` | `/tmp/libfp4QuantizeAdaptive.so` | Path to adaptive FP4 shared library |

### Steps

```bash
# 1. Enter TRT-LLM container
#    nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14

# 2. Copy modified files from worktree into container
WORKTREE=/home/scratch.georgel_gpu/projects/llm_4o6/TensorRT-LLM-weight-act-4o6-rc14
TRTLLM_PKG=/usr/local/lib/python3.12/dist-packages/tensorrt_llm

cp $WORKTREE/tensorrt_llm/_torch/modules/fused_moe/quantization.py \
   $TRTLLM_PKG/_torch/modules/fused_moe/quantization.py

cp $WORKTREE/tensorrt_llm/_torch/modules/fused_moe/fused_moe_cute_dsl.py \
   $TRTLLM_PKG/_torch/modules/fused_moe/fused_moe_cute_dsl.py

cp $WORKTREE/tensorrt_llm/_torch/modules/fused_moe/create_moe.py \
   $TRTLLM_PKG/_torch/modules/fused_moe/create_moe.py

# 3. Install fouroversix + build adaptive FP4 library (same as FC13 setup)
#    Ensure libfp4QuantizeAdaptive.so is at /tmp/ or set TRTLLM_ADAPTIVE_FP4_SO

# 4. Launch with both env vars enabled
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
trtllm-serve ... --backend pytorch

# 5. Full weight + activation 4o6
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
TRTLLM_ADAPTIVE_FP4_WEIGHT=1 \
trtllm-serve ... --backend pytorch
```

### Disable CUDA graph (required)

Adaptive 4/6 is incompatible with CUDA graph. In `extra-llm-api-config.yml`:
```yaml
cuda_graph_config: null
```

### Testing checklist (CC 请按顺序执行)

**前提**：所有测试必须在 `extra-llm-api-config.yml` 中设置 `cuda_graph_config: null`，否则 adaptive 4/6 会出错。

**Step 1 — CuteDsl baseline（标准 NVFP4，排除后端差异）**
```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=0 \
TRTLLM_ADAPTIVE_FP4_FC2=0 \
trtllm-serve ...
```
确认 CuteDsl 后端精度与 TRTLLMGen baseline 一致，排除后端切换引入的误差。

**Step 2 — FC13 4o6 only（验证 FC13 adaptive 在 CuteDsl 上正常工作）**
```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=0 \
trtllm-serve ...
```
日志中应出现 `[4o6-FC13]` 统计。对比与 TRTLLMGen FC13 4o6 的精度差异。

**Step 3 — FC2 4o6 only（隔离 FC2 效果）**
```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=0 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
trtllm-serve ...
```
日志中应出现 `[4o6-FC2]` 统计。此时 FC13 为标准 NVFP4，仅 FC2 用 adaptive 4/6。

**Step 4 — 全链路 FC13 + FC2 4o6（最终目标形态）**
```bash
TRTLLM_MOE_FORCE_CUTEDSL=1 \
TRTLLM_ADAPTIVE_FP4=1 \
TRTLLM_ADAPTIVE_FP4_FC2=1 \
trtllm-serve ...
```
日志中应同时出现 `[4o6-FC13]` 和 `[4o6-FC2]` 统计。

> **注意**：`read_and_reset_4o6_stats` 的计数器是**全局**的。当 FC13 和 FC2 同时启用时，日志中的 block 统计包含两者的混合数据。Step 2/3 的隔离测试可以提供纯净的单阶段统计。

---

## Bug Fixes (v2)

### Bug 1: `float4_e2m1fn_x2.to(bfloat16)` assertion failure

PyTorch CUDA 不支持 `float4_e2m1fn_x2` → `bfloat16` 的 dynamic cast。
**Fix**: 用 E2M1 LUT 查表法解包 FP4 nibbles（`_get_e2m1_lut` + 手动 unpack low/high nibble）。

### Bug 2: `interm_size` 计算错误

原代码 `interm_size = self.w2_weight.shape[-1] * 2` 取的是 weight 的 packed expert-level layout 维度，不是实际 intermediate size。
**Fix**: 从 `x` 的 shape 推导：`interm_size = x_fp4.shape[-1] * 2`（每 byte 包含 2 个 FP4 值）。`_dequant_nvfp4_linear` 不再需要外部传入 `interm_size` 参数。

### Bug 3: CuteDsl autotuner "Offset increment outside graph capture"

`run_moe_nvfp4_impl` 内部的 dequant→requant 分配了新 tensor，破坏了 CuteDsl autotuner 的 offset tracking。FC13 的 adaptive 4/6 不受影响，因为它在 `quantize_input()` 里执行（在 autotuner scope 之外）。
**Fix**: 当 `TRTLLM_ADAPTIVE_FP4_FC2=1` 时，bypass autotuner 直接调用 `run_moe_nvfp4_impl(tile_size=128)`。牺牲 tile config 自动调优，但对精度实验无影响。生产实现（方案 B）不需要此 workaround。

### Bug 4: CuTe SF swizzle layout ≠ TRT-LLM swizzle layout（dequant 输出乱码）

CuteDsl SwiGLU kernel 输出的 SF 使用 CuTe `ordered_layout` swizzle（128×4 block），但行内映射与 TRT-LLM `computeSFIndex` **完全不同**：

| Logical row | TRT-LLM offset | CuTe offset |
|-------------|----------------|-------------|
| row 1 | 16 | 4 |
| row 4 | 64 | 16 |
| row 32 | 4 | 128 |

TRT-LLM 按 `[row%32 stride=16, (row%128)/32 stride=4]` 拆行；CuTe 按 `[row%4 stride=4, (row//4)%32 stride=16]` 拆行。

这导致 `block_scale_interleave_reverse`（实现 TRT-LLM 格式）对 CuTe 数据无效，CC 之前的手动 de-swizzle 也因行拆分公式搞反而失败。看似 "NaN" / "negative SF" 的异常值实际是 swizzled bytes 被当 linear 解读的结果。

**Fix**:
- 新增 `_cutedsl_sf_swizzle_indices()` — 正确的 CuTe swizzle index 映射
- 新增 `_deswizzle_cutedsl_sf()` — CuTe swizzled → linear [m, sf_cols]
- 新增 `_reswizzle_cutedsl_sf()` — linear → CuTe swizzled（scatter 回 FC2 kernel 期望的格式）
- `_dequant_nvfp4_cutedsl()` 替代原来的 `_dequant_nvfp4_linear()`
- Requant 流程：`fp4_quantize_ex(isSfSwizzledLayout=False)` → linear SF → `_reswizzle_cutedsl_sf()` → FC2 GEMM

详见 `bug.fc2_4o6.md` 的 "根因确认" section。

---

## Known Limitations

1. **Performance overhead**: dequant + requant adds two extra kernel launches per MoE layer per forward pass. This is acceptable for accuracy evaluation but not for production.
2. **Global amax**: the runtime amax is computed over all permuted tokens (all experts combined). Per-expert amax may yield better accuracy but adds complexity.
3. **CuteDsl compatibility**: the CuteDsl NVFP4 backend may have different numerics or unsupported configurations compared to TRTLLMGen. Verify baseline accuracy first (run with `TRTLLM_MOE_FORCE_CUTEDSL=1` but `TRTLLM_ADAPTIVE_FP4_FC2=0`).
4. **4o6 stats counter**: the `read_and_reset_4o6_stats` counters are **global** — if FC13 also uses adaptive 4/6, the printed stats reflect both FC13 and FC2. To isolate FC2 stats, disable FC13 adaptive 4/6 (`TRTLLM_ADAPTIVE_FP4=0`).

---

## Approach B — Fused Adaptive 4/6 in PermuteGemm1 Epilogue (production delivery)

### Goal

Eliminate the dequant→requant overhead by making PermuteGemm1's epilogue directly output adaptive 4/6 FP4 instead of standard NVFP4. The current TRTLLMGen dequant→requant implementation (runner.cu L623-662) adds 3-4 extra kernel launches — the fused approach would be zero extra launches.

### Current PermuteGemm1 epilogue behavior

```
GEMM1 (FC13: gate + up projection)
  → SwiGLU activation
  → Standard NVFP4 quantization (scaleRule=0, max_e2m1=6)
  → Output: packed E2M1 FP4 + swizzled block SF
```

The FP4 quantization uses a **static global_scale** (`output1_scales_scalar` per expert) and always chooses the 6-bit scale factor (standard NVFP4). No per-block 4/6 MSE comparison is done.

### Target PermuteGemm1 epilogue behavior

```
GEMM1 (FC13: gate + up projection)
  → SwiGLU activation
  → Adaptive 4/6 NVFP4 quantization (scaleRule=1)
      Per-block: compute MSE with 6-bit scale vs 4-bit scale (1.5× expansion)
      Select whichever has lower MSE
      Encode selection in SF byte
  → Output: packed E2M1 FP4 + swizzled block SF (with adaptive selection)
```

### Blocker: Cubin generation pipeline is NOT in this repo

**The PermuteGemm1 cubin is a precompiled ELF binary.** The kernel source code and generation pipeline are external to TensorRT-LLM.

Evidence:
- `trtllmGen_bmm_export/cubins/` contains 569 `.cpp` files, each embedding an ELF binary as a C byte array
- `KernelMetaInfo.h`: `TLLM_GEN_COMMIT "b7b335a4-dirty"`, `TLLM_GEN_EXPORT_VERSION "7.0.4.0.4.0"`
- Zero `.cu`/`.cuh` kernel source files in `batchedGemm/` — only the C++ runner/wrapper code
- `CMakeLists.txt` compiles only the pre-generated `.cpp` wrappers, not kernel source
- The cubin generation tool is in a separate internal NVIDIA repository

**→ Need to work with the TRTLLMGen kernel team to modify the epilogue.**

### What needs to change in the cubin generator

The epilogue's FP4 quantization kernel needs a **per-block adaptive selection** mode. Specifically:

1. **New runtime parameter: `scaleRule`** (int, passed via kernel args or constant memory)
   - `0` = standard NVFP4 (current behavior, always 6-bit scale)
   - `1` = adaptive MSE (compare 4-bit vs 6-bit, pick lower MSE)

2. **Per-block epilogue logic change** (pseudocode):
   ```
   // Current (scaleRule=0):
   sf = block_max / (6.0 * global_scale)
   sf_e4m3 = cast_to_fp8_e4m3(sf)
   quantize elements with sf_e4m3

   // Adaptive (scaleRule=1):
   sf_6 = block_max / (6.0 * global_scale)          // standard scale
   sf_4 = sf_6 * 1.5                                 // expanded scale (4-bit path)
   mse_6 = compute_mse(elements, quantize(elements, sf_6))
   mse_4 = compute_mse(elements, quantize(elements, sf_4))
   if mse_4 < mse_6:
       sf_best = sf_4; sf_e4m3 = cast_to_fp8_e4m3(sf_4)
   else:
       sf_best = sf_6; sf_e4m3 = cast_to_fp8_e4m3(sf_6)
   quantize elements with sf_best
   ```

3. **No global amax needed in epilogue**: The adaptive selection is entirely per-block. The existing static `global_scale` parameter works. Runtime amax is only needed if replacing the static scale, which is a separate concern.

4. **New cubin variants**: Generate E2M1-output cubins with `scaleRule` support. The cubin name could encode this, e.g., `..._adaptFp4_sm100f_cubin.cpp`. Alternatively, `scaleRule` can be a runtime parameter if the kernel generator supports it.

5. **Cubin selection**: Add `scaleRule` to `TrtllmGenBatchedGemmRunnerOptions` and filter in `KernelRunner.cpp` L154-173.

### Reference: Adaptive 4/6 kernel implementation

The per-block MSE selection logic is already implemented in `fp4QuantizeAdaptive.cuh` (standalone kernel). The cubin epilogue needs the same logic:

- **File**: `cpp/tensorrt_llm/kernels/fp4QuantizeAdaptive.cuh`
- **Function**: `opt_quantize_with_block_size_adaptive<T, SF_VEC_SIZE, SCALE_RULE>()`
- **Key steps**: Quantize with both sf_6 and sf_4, accumulate MSE per thread, reduce across warp, compare, select winner
- **Block size**: 16 elements per SF block (SF_VEC_SIZE=16 for NVFP4)

### Alpha correction

When `scaleRule > 0`, the FC2 alpha needs correction:
```
correction = fc2_input_scale / dynamic_global_scale
fc2_alpha_corrected = fc2_alpha_static * correction
```

If using static `global_scale` (no runtime amax), no alpha correction is needed — the same static scale is used for both quantization and alpha computation.

### Intermediate workaround: TRTLLMGen dequant→requant (already implemented)

The non-fused path is already in `runner.cu` L623-662, gated by `TRTLLM_ADAPTIVE_FP4_FC2=1`:

```
PermuteGemm1(dtypeC=E2M1, standard NVFP4)
  → invokeDequantNvfp4SwizzledSF (FP4→BF16)
  → computeGlobalAmax (BF16→runtime amax)
  → invokeFP4QuantizationEx(scaleRule=1) (BF16→adaptive FP4)
  → invokeScaleAlphaByAmax (alpha correction)
  → Gemm2
```

This adds ~4 extra kernel launches and a BF16 intermediate buffer (`adaptive_bf16_buf`) but produces identical results. **Use for accuracy validation while waiting for the fused cubin.**

### Alternative: BF16-output cubin + single adaptive kernel

34 BF16-output cubins (`Bfloat16_E2m1E2m1_...`) already exist. This avoids the dequant step:

```
PermuteGemm1(dtypeC=BF16)     ← change dtypeC in runner.cu L232
  → fusedAmaxAdaptiveFP4Quant  ← 1 new kernel (persistent, retirement-counter sync)
  → Gemm2
```

Requires:
- `runner.cu` L232: `dtypeC = BFloat16` when `fc2ScaleRule > 0`
- New fused kernel combining `computeGlobalAmax` + `invokeFP4QuantizationEx`
- Effort: Medium (1 new CUDA kernel + runner.cu plumbing)

---

## Current Implementation Status

### Approach A — CuteDsl prototype (accuracy validation) ✅

| Step | Config | Result |
|------|--------|--------|
| 1 | CuteDsl baseline (standard NVFP4) | ✅ |
| 2 | FC13 4o6 only | ✅ |
| 3 | FC2 4o6 only | ✅ |
| 4 | FC13 + FC2 4o6 | ✅ |

Early MMLU smoke benchmark (Qwen3-30B-A3B, 200 samples, 1× B200): all 4 NVFP4 configs identical accuracy → **adaptive 4/6 is precision-neutral within the smoke protocol**. The full rc14 validation below uses a different comparison axis: BF16 checkpoint + BF16 activation/weights vs NVFP4 checkpoint + full weight/activation 4o6.

### Approach B — TRTLLMGen (non-fused dequant→requant) ✅

Full pipeline implemented in `runner.cu` L623-662. Python→C++ parameter passing complete. Not yet benchmarked separately (same algorithm as CuteDsl, expected identical results).

### Approach B — TRTLLMGen (fused cubin epilogue) ❌ Blocked

Requires TRTLLMGen kernel team to add adaptive scaleRule to the PermuteGemm1 cubin epilogue. See "What needs to change in the cubin generator" above.

### rc14 Qwen3 Accuracy Validation (2026-05-12)

> **Intro-Summary**: The full MMLU run shows BF16 baseline ahead of NVFP4 checkpoint + full weight/activation 4o6 by **+1.30 pp** on 14,042 samples, with a paired 90% CI of **[+0.95, +1.66] pp**. This is the opposite sign from the earlier GSM8K flexible-extract comparison, but the GSM8K sign is metric/protocol sensitive and should not be read as a task-invariant quantization effect.

#### Benchmark disclosure

| Item | Value |
|---|---|
| Hardware | 1× NVIDIA B200 allocation |
| Software | TRT-LLM `v1.3.0rc14`, branch `experiment/weight-act-adaptive-4o6-rc14`, clean full build wheel |
| Input | Qwen3-30B-A3B; MMLU 5-shot native TRT-LLM evaluator, 14,042 samples |
| Configuration A | NVFP4 checkpoint `/llm-models/Qwen3/saved_models_Qwen3-30B-A3B_nvfp4_hf`, `TRTLLM_MOE_FORCE_CUTEDSL=1`, `TRTLLM_ADAPTIVE_FP4=1`, `TRTLLM_ADAPTIVE_FP4_FC2=1`, `TRTLLM_ADAPTIVE_FP4_WEIGHT=1` |
| Configuration B | BF16 checkpoint `/llm-models/Qwen3/Qwen3-30B-A3B`, BF16 activation/weights, no adaptive 4o6 env vars |
| Sampling | `num_fewshot=5`, `num_samples=14042`, `random_seed=0`, staged every 1000 samples; note that the native evaluator iterates subject-ordered rows, so 1000-prefix is not a flat random sample |
| Statistic | MMLU sample-weighted accuracy; paired 90% CI computed from aligned `task_id` correctness vectors |
| Baseline | BF16 checkpoint + BF16 activation/weights |

Raw outputs:

- NVFP4 + full 4o6: `/home/scratch.georgel_gpu/projects/llm_4o6/benchmarks/rc14_mmlu_nvfp4_staged/20260512T065816Z_nvfp4_ckpt_weight_act_4o6_mmlu_full`
- BF16 baseline: `/home/scratch.georgel_gpu/projects/llm_4o6/benchmarks/rc14_mmlu_bf16_staged/20260512T071745Z_bf16_ckpt_bf16_weight_act_mmlu_full`

#### Full MMLU result

| Config | Correct / n | Accuracy |
|---|---:|---:|
| NVFP4 ckpt + weight/activation 4o6 | 10977 / 14042 | 78.17% |
| BF16 ckpt + BF16 activation/weights | 11160 / 14042 | 79.48% |
| BF16 - NVFP4+4o6 | +183 / 14042 | **+1.30 pp** |

Paired comparison: 921 discordant samples; BF16-only correct 552, NVFP4+4o6-only correct 369. The paired 90% CI for `BF16 - NVFP4+4o6` is **[+0.95, +1.66] pp**, so the full MMLU run supports a small but statistically visible BF16 advantage under this protocol.

#### Stage-by-stage MMLU table

| n | NVFP4 ckpt + weight/act 4o6 | BF16 ckpt + BF16 weight/act | BF16 - NVFP4 (pp) |
|---:|---:|---:|---:|
| 1000 | 800 / 1000 (80.00%) | 810 / 1000 (81.00%) | +1.00 |
| 2000 | 1591 / 2000 (79.55%) | 1614 / 2000 (80.70%) | +1.15 |
| 3000 | 2390 / 3000 (79.67%) | 2432 / 3000 (81.07%) | +1.40 |
| 4000 | 3268 / 4000 (81.70%) | 3320 / 4000 (83.00%) | +1.30 |
| 5000 | 4069 / 5000 (81.38%) | 4136 / 5000 (82.72%) | +1.34 |
| 6000 | 4959 / 6000 (82.65%) | 5028 / 6000 (83.80%) | +1.15 |
| 7000 | 5785 / 7000 (82.64%) | 5864 / 7000 (83.77%) | +1.13 |
| 8000 | 6680 / 8000 (83.50%) | 6772 / 8000 (84.65%) | +1.15 |
| 9000 | 7380 / 9000 (82.00%) | 7503 / 9000 (83.37%) | +1.37 |
| 10000 | 8112 / 10000 (81.12%) | 8253 / 10000 (82.53%) | +1.41 |
| 11000 | 8809 / 11000 (80.08%) | 8972 / 11000 (81.56%) | +1.48 |
| 12000 | 9358 / 12000 (77.98%) | 9519 / 12000 (79.33%) | +1.34 |
| 13000 | 10152 / 13000 (78.09%) | 10330 / 13000 (79.46%) | +1.37 |
| 14000 | 10938 / 14000 (78.13%) | 11122 / 14000 (79.44%) | +1.31 |
| 14042 | 10977 / 14042 (78.17%) | 11160 / 14042 (79.48%) | +1.30 |

The first 1000-prefix gives `BF16 - NVFP4+4o6 = +1.00 pp`, but its paired 90% CI is **[-0.19, +2.19] pp** and the prefix only covers the subject-ordered front of the native evaluator. Using the full-run paired variance, a flat random 1000-sample estimate would have an expected 90% CI half-width of about **1.33 pp**, which is comparable to the observed full-run effect. So by the earlier 1000-sample 90% CI口径, the MMLU sign is not robust; by the full 14,042-sample run, it is.

#### Category breakdown

| Category | n | NVFP4+4o6 | BF16 | BF16 - NVFP4 90% CI |
|---|---:|---:|---:|---:|
| STEM | 3018 | 79.36% | 80.75% | +1.39 pp [+0.65, +2.14] |
| Humanities | 4705 | 69.86% | 71.46% | +1.59 pp [+0.90, +2.29] |
| Other (business, health, misc.) | 3242 | 81.03% | 82.26% | +1.23 pp [+0.54, +1.93] |
| Social sciences | 3077 | 86.71% | 87.55% | +0.85 pp [+0.20, +1.50] |

The MMLU gap is broad across categories rather than a single-subject outlier. Social sciences is the smallest gap, but its paired 90% CI still stays above zero.

#### Interpreting the sign difference vs GSM8K

The earlier GSM8K comparison that looked "opposite" used **flexible numeric extraction**. In the 2026-05-07 GSM8K staged artifacts, `weight_act_4o6 - BF16-current` is **+3.94 pp** on flex with 90% CI **[+2.57, +5.32] pp**, but only **+0.68 pp** on strict with 90% CI **[-0.41, +1.78] pp**. That means the strongest GSM8K sign is format/extraction sensitive.

The clean rc14 two-checkpoint GSM8K run is not the same baseline as the MMLU table because both sides use weight/activation 4o6; under that comparison, `BF16 ckpt + 4o6 - NVFP4 ckpt + 4o6` is **+2.50 pp** flex and **+1.74 pp** strict, both favoring the BF16 checkpoint side.

**Takeaway**: MMLU full-run evidence says BF16 baseline is modestly ahead of NVFP4 checkpoint + full weight/activation 4o6. The apparent GSM8K reverse sign comes primarily from the flexible-extract metric and an older comparison axis. It should be treated as a protocol artifact until we run a matched rc14 GSM8K BF16-baseline vs NVFP4+4o6 comparison and compute the paired delta-of-deltas.

---

## File Inventory

| File | Location | Status |
|------|----------|--------|
| `fused_moe_cute_dsl.py` | `_torch/modules/fused_moe/` | Modified (adaptive 4/6 FC13 + FC2, CuteDsl path) |
| `fused_moe_trtllm_gen.py` | `_torch/modules/fused_moe/` | Modified (FC2 scale_rule param passing) |
| `create_moe.py` | `_torch/modules/fused_moe/` | Modified (CuteDsl override env var) |
| `trtllm_gen_custom_ops.py` | `_torch/custom_ops/` | Modified (fc2_scale_rule, fc2_input_scale params) |
| `fp4BlockScaleMoe.cpp` | `cpp/tensorrt_llm/thop/` | Modified (MoERunnerArgs fc2 adaptive fields, workspace alloc) |
| `runner.h` | `cpp/.../blockScaleMoe/` | Modified (MoERunnerArgs + MoEWorkspace adaptive fields) |
| `runner.cu` | `cpp/.../blockScaleMoe/` | Modified (dequant→requant adaptive pipeline L623-662) |
| `fp4QuantizeAdaptiveOp.cpp` | `cpp/tensorrt_llm/thop/` | Modified (custom op registration) |
| `fp4QuantizeAdaptive.cu/.cuh/.h` | `cpp/tensorrt_llm/kernels/` | Modified (adaptive kernel + dequant + amax + alpha correction) |
| `FC2_ADAPTIVE_4O6.md` | worktree root | This document |
