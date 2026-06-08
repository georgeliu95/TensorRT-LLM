# Qwen3-8B NVFP4 4o6 — Accuracy Evaluation Example

End-to-end example: quantize a **dense** Qwen3-8B into **NVFP4 4-over-6 (4o6)** — weight + activation
— and score it on **GSM8K** and **MMLU** with `trtllm-eval`, on a single Blackwell GPU.

This example targets the `experiment/weight-act-adaptive-4o6-rc14` branch, which adds the dense 4o6
runtime path and the checkpoint converter (see [What the branch adds](#what-the-branch-adds)).

---

## Paths

All commands below use these variables. Replace the example values with your own.

| Variable | Meaning | Example value |
|----------|---------|---------------|
| `REPO` | TensorRT-LLM checkout (this branch) | `/home/user/workspace/customer/TensorRT-LLM` |
| `TOOLS` | Directory holding `convert_ckpt_to_4o6_nvfp4.py` | `/home/user/workspace/tools` |
| `BF16_MODEL` | Source HF BF16 model | `/llm-models/Qwen3/Qwen3-8B` |
| `OUT_4O6` | Output dir for the converted 4o6 ckpt | `/home/user/workspace/Qwen3-8B-4o6-nvfp4` |
| `PY` | Container system Python | `/usr/bin/python3` |

```bash
export REPO=/home/user/workspace/customer/TensorRT-LLM
export TOOLS=/home/user/workspace/tools
export BF16_MODEL=/llm-models/Qwen3/Qwen3-8B
export OUT_4O6=/home/user/workspace/Qwen3-8B-4o6-nvfp4
export PY=/usr/bin/python3
```

---

## 0. Prerequisites

- A single **Blackwell** GPU (validated on **B200**, sm_100, 183 GB).<br>
- Inside the TensorRT-LLM dev container `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc14`,
  which ships **torch 2.11 (nv build) / CUDA 13.1 / Python 3.12**.<br>
- Passwordless `sudo` available inside the container (install steps write to the system Python).

> **NOTE**: The wheel's C++ extensions must be compiled against the container's exact torch build.
> If you change containers (and thus the torch version), rebuild the wheel — otherwise importing
> `tensorrt_llm` fails with `undefined symbol`.

> **NOTE**: A `sudo: unable to resolve host <node>` warning is harmless (sudo failing a reverse-DNS
> lookup for logging). It does not affect any command.

---

## 1. Build & install the wheel (from source)

Building from this branch produces a wheel that already contains the dense 4o6 code — no manual file
copying afterward.

### 1a. Build dependencies

```bash
sudo "$PY" -m pip install --break-system-packages 'setuptools<80' 'conan==2.14.0'
sudo "$PY" -m pip install --break-system-packages --no-deps 'nvidia-nccl-cu13==2.29.2'

# The container ships only libnccl.so.2; the build needs the unversioned libnccl.so.
NCCL_ROOT=$("$PY" -c "import os, site; \
  print([c for c in [os.path.join(s, 'nvidia', 'nccl') for s in site.getsitepackages()] \
  if os.path.exists(os.path.join(c, 'include', 'nccl.h'))][0])")
[ -e "$NCCL_ROOT/lib/libnccl.so" ] || sudo ln -s libnccl.so.2 "$NCCL_ROOT/lib/libnccl.so"
echo "NCCL_ROOT=$NCCL_ROOT"
```

> **WARNING**: `setuptools<80` is required — setuptools 80+ breaks the CUTLASS
> `setup_library.py develop` step during the build.

### 1b. Build (sm_100 only, ~55 min on 28 cores)

```bash
cd "$REPO"
export CONAN_HOME="$HOME/.conan2"
export PIP_CACHE_DIR="$HOME/.cache/pip"
export PIP_BREAK_SYSTEM_PACKAGES=1
export TRTLLM_SKIP_REQUIREMENTS_INSTALL=1
export PYTHONPATH="$REPO/cpp/build/_deps/cutlass-src/python:$REPO/cpp/build/_deps/flashmla-src/csrc/cutlass/python:$REPO/cpp/build/_deps/deepgemm-src/third-party/cutlass/python:${PYTHONPATH:-}"
mkdir -p "$CONAN_HOME" "$PIP_CACHE_DIR"

"$PY" ./scripts/build_wheel.py \
    --cuda_architectures "100-real" \
    --nccl_root "$NCCL_ROOT" \
    --no-venv --clean
```

Output: `build/tensorrt_llm-1.3.0rc14-cp312-cp312-linux_x86_64.whl`.

> **WARNING**: Do **not** export `PYTHONNOUSERSITE` — it breaks the CUTLASS `develop --user` step.
> Build with `--no-venv` against the system Python; do not create a virtualenv.

### 1c. Install

```bash
# --no-deps protects the container's nv torch; --force-reinstall overwrites the stock package.
sudo "$PY" -m pip install --break-system-packages --no-deps --force-reinstall \
    "$REPO"/build/tensorrt_llm-1.3.0rc14-cp312-cp312-linux_x86_64.whl
```

> **WARNING**: `--no-deps` is mandatory. Letting pip resolve dependencies makes it judge the local
> nv torch "incompatible" and try to replace it, which breaks CUDA.

### 1d. Verify

Run from a **non-repo** directory so Python imports the installed wheel, not the source tree:

```bash
cd /tmp
"$PY" -c "import tensorrt_llm, torch; \
  print(tensorrt_llm.__version__, torch.__version__, torch.cuda.is_available())"

# Dense 4o6 path is baked into the wheel (expects 2):
grep -c fp4_quantize_ex /usr/local/lib/python3.12/dist-packages/tensorrt_llm/_torch/modules/linear.py
```

> **TIP**: If you already have a wheel built against the *same* torch version (e.g. on shared
> storage from a prior run), you can skip 1a–1b and install it directly with the 1c command.

---

## 2. Install the evaluation dependency

GSM8K scoring uses [`lm_eval`](https://github.com/EleutherAI/lm-evaluation-harness). Pin torch /
transformers / datasets so installing it cannot disturb the container's torch:

```bash
printf 'torch==%s\ntransformers==4.57.3\ndatasets==3.1.0\n' \
  "$("$PY" -c 'import torch; print(torch.__version__)')" > /tmp/c.txt
sudo "$PY" -m pip install --break-system-packages -c /tmp/c.txt 'lm_eval[api]==0.4.10'
```

---

## 3. Convert Qwen3-8B → NVFP4 4o6 (~90 s)

```bash
"$PY" "$TOOLS/convert_ckpt_to_4o6_nvfp4.py" \
    --input "$BF16_MODEL" \
    --output "$OUT_4O6" \
    --source-format bf16 \
    --all-linear \
    --exclude-regex '.*(embed_tokens|lm_head).*' \
    --quant-backend trtllm \
    --activation-mode 4o6 \
    --scale-rule mse \
    --overwrite
```

- `--all-linear`: required for dense models (the default include-regex matches only MoE experts).<br>
- `--exclude-regex`: skip `embed_tokens` / `lm_head` (1D norms are auto-skipped).<br>
- The output contains `hf_quant_config.json` (`quant_algo=NVFP4`); the loader recognizes it as an
  exported 4o6 checkpoint.

Sanity check:

```bash
ls "$OUT_4O6" && cat "$OUT_4O6/hf_quant_config.json"
```

---

## 4. Run `trtllm-eval`

Both tasks need the 4o6 runtime switches. Create the shared config once (NVFP4 4o6 is incompatible
with CUDA graphs, so disable them):

```bash
printf 'cuda_graph_config: null\n' > /tmp/eval_4o6.yml
```

The two tasks differ in how the prompt is fed: GSM8K is generative (chat template, long output);
MMLU is completion-style (no chat template, 2-token output). See each subsection.

### 4a. GSM8K — generative, **with** chat template

```bash
cd /tmp
TRTLLM_ADAPTIVE_FP4_DENSE=1 \
trtllm-eval \
    --model "$OUT_4O6" --tokenizer "$BF16_MODEL" --backend pytorch \
    --config /tmp/eval_4o6.yml --kv_cache_free_gpu_memory_fraction 0.8 \
    gsm8k \
    --apply_chat_template \
    --chat_template_kwargs '{"enable_thinking": false}' \
    --max_output_length 512
```

- `TRTLLM_ADAPTIVE_FP4_DENSE=1` — enables dense activation 4o6. The exported checkpoint's static
  `input_scale` is a placeholder, so the runtime **must** do dynamic activation quantization.<br>
- `--config /tmp/eval_4o6.yml` — `cuda_graph_config: null` (required for 4o6).<br>
- `--chat_template_kwargs '{"enable_thinking": false}'` — Qwen3 is a thinking model; leaving thinking
  on makes GSM8K reasoning overrun `max_output_length` and get truncated, deflating the score.<br>
- Omit `--num_samples` to run the full **1319** items; add `--num_samples 20` for a quick smoke test.

### 4b. MMLU — completion-style, **without** chat template

```bash
cd /tmp
TRTLLM_ADAPTIVE_FP4_DENSE=1 \
trtllm-eval \
    --model "$OUT_4O6" --tokenizer "$BF16_MODEL" --backend pytorch \
    --config /tmp/eval_4o6.yml --kv_cache_free_gpu_memory_fraction 0.8 \
    mmlu
```

The 4o6 switches (`TRTLLM_ADAPTIVE_FP4_DENSE=1`, `cuda_graph_config: null`) are unchanged. MMLU
defaults (`--num_fewshot 5`, `--max_output_length 2`) are correct as-is; add `--num_samples 200`
for a quick smoke test.

> **WARNING**: Do **not** pass `--apply_chat_template` to MMLU. This evaluator is completion-style —
> the prompt ends in `Answer:`, generation is capped at `--max_output_length 2`, and scoring checks
> whether the output *starts with* the gold letter (A/B/C/D). A chat template makes the model emit
> template / thinking preamble first, which the 2-token budget truncates before the answer ever
> appears, collapsing accuracy to ~0.

### BF16 baseline (optional)

In either command above, swap `--model "$OUT_4O6"` → `--model "$BF16_MODEL"` and drop the
`TRTLLM_ADAPTIVE_FP4_DENSE=1` prefix.

---

## 5. Expected results

Qwen3-8B, single B200.

**GSM8K** (1319 items):

| Configuration | average | flexible-extract | strict-match |
|---------------|--------:|-----------------:|-------------:|
| BF16 baseline | ~87.2% | ~88.1% | ~86.4% |
| NVFP4 4o6 (weight + act) | ~77–78% | ~80% | ~75% |

**MMLU** (14042 items, 5-shot):

| Configuration | weighted average |
|---------------|-----------------:|
| NVFP4 4o6 (weight + act) | ~72.6% |

`trtllm-eval` prints a summary line per task:

```text
[TRT-LLM] [I] [evaluate] lm-eval gsm8k average accuracy: 78.01
[TRT-LLM] [I] [evaluate] MMLU weighted average accuracy: 72.56 (14042)
```

---

## What the branch adds

The dense 4o6 support lives in `experiment/weight-act-adaptive-4o6-rc14`. When starting from clean
upstream, these are the pieces to port:

- **`tensorrt_llm/_torch/modules/linear.py`** — `NVFP4LinearMethod.load_weight_scales` accepts a
  `uint8` `weight_scale` (exported checkpoints store E4M3 bytes as `uint8`); `_input_prepare` runs
  the dense activation 4o6 path via `torch.ops.trtllm.fp4_quantize_ex` with a **swizzled** activation
  scale-factor layout, gated by `TRTLLM_ADAPTIVE_FP4_DENSE`.<br>
- **C++ ops** `fp4_quantize_ex` / `calculate_global_amax` — custom ops added on this branch; they are
  compiled into the wheel (step 1), which is why a from-source build is required.<br>
- **`convert_ckpt_to_4o6_nvfp4.py` → `make_groups`** — groups dense `q/k/v` and `gate/up` so fused
  projections share a single `weight_scale_2` (one alpha per fused GEMM).

---

## Common pitfalls

> **WARNING**: Always run `python` / `trtllm-eval` from a **non-repo** directory (e.g. `/tmp`). The
> `tensorrt_llm/` source tree under the repo root shadows the installed wheel on import.

> **WARNING**: Install the wheel and `lm_eval` with the dependency guards shown above
> (`--no-deps` / constraints file). Otherwise pip may replace the container's nv torch and break CUDA.
