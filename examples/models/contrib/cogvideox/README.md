# CogVideoX for image-to-video task
This document shows how to build and run a [CogVideoX](https://github.com/huggingface/diffusers/blob/v0.32.2/src/diffusers/pipelines/cogvideo/pipeline_cogvideox_image2video.py) for image-to-video task with TensorRT-LLM.

## Overview

The TensorRT-LLM implementation of CogVideoX can be found in [tensorrt_llm/models/cogvideox/model.py](../../tensorrt_llm/models/cogvideox/model.py). The TensorRT-LLM image-to-video example code is located in [`examples/cogvideox`](./). There are main files to build and run MMDiT with TensorRT-LLM:

* [`convert_checkpoint.py`](./convert_checkpoint.py) to convert the `CogVideoXTransformer3D` model into tensorrt-llm checkpoint format.
* [`sample.py`](./sample.py) to run the [diffusers](https://huggingface.co/docs/diffusers/index) pipeline with TensorRT engine(s) to generate videos.

## Support Matrix

- [ ] TP
- [ ] CP
- [ ] FP8

## Usage

The TensorRT-LLM image-to-video example code locates at [examples/cogvideox](./). It takes HuggingFace checkpiont as input, and builds the corresponding TensorRT engines. The number of TensorRT engines depends on the number of GPUs used to run inference.

### Build CogVideoXTransformer3D TensorRT engine(s)

This checkpoint will be converted to the TensorRT-LLM checkpoint format by [`convert_checkpoint.py`](./convert_checkpoint.py). After that, we can build TensorRT engine(s) with the TensorRT-LLM checkpoint.

```
# Convert to TRT-LLM
python convert_checkpoint.py --model_path='THUDM/CogVideoX-5b-I2V'
# Build engine in BF16 (recommended)
trtllm-build --checkpoint_dir=./tllm_checkpoint/ \
             --max_batch_size=2 \
             --remove_input_padding=disable \
             --bert_attention_plugin=auto
# or build engine in FP16
trtllm-build --checkpoint_dir=./tllm_checkpoint/ \
             --max_batch_size=2 \
             --remove_input_padding=disable \
             --bert_attention_plugin=auto \
             --bert_context_fmha_fp32_acc=enable
```

Set `--max_batch_size` to tell how many images at most you would like to generate. We disable `--remove_input_padding` since we don't need to padding `CogVideoXTransformer3D`'s patches.

After build, we can find a `./engine_output` directory, it is ready for running `CogVideoTransformer3D` model with TensorRT-LLM now.

### Generate video from image and text

A [`sample.py`](./sample.py) is provided to generated images with the optimized TensorRT engines.

If using `float16` for inference, `FusedRMSNorm` from `Apex` used by T5-encoder should be disabled in the [huggingface/transformers](https://github.com/huggingface/transformers/blob/v4.48.3/src/transformers/models/t5/modeling_t5.py#L259) or just uninstall the `apex`:
```python
try:
    from apex.normalization import FusedRMSNorm

    # [NOTE] Avoid using `FusedRMSNorm` for T5 encoder.
    # T5LayerNorm = FusedRMSNorm  # noqa

    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm")
except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")
    pass

ALL_LAYERNORM_LAYERS.append(T5LayerNorm)
```

Just run `python sample.py` and we can see a video named `cogvideox_output.mp4` will be generated.

### Tensor Parallel

Not supported yet.

### Context Parallel

Not supported yet.
