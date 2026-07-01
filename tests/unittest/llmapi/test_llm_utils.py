import asyncio
import tempfile
import threading
import time
from pathlib import Path

import torch

from tensorrt_llm.llmapi.llm_utils import *
from tensorrt_llm.llmapi.llm_utils import (
    _direct_nvfp4_transform_enabled, _svdquant_direct_load_enabled)
from tensorrt_llm.llmapi.utils import AsyncQueue
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# isort: off
from .test_llm import llama_model_path
# isort: on

from tensorrt_llm.llmapi.llm_args import *


def test_svdquant_master_without_stages_does_not_bypass_modelopt(
    monkeypatch,
):
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC13", "0")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC2", "0")

    assert not _svdquant_direct_load_enabled(QuantAlgo.NVFP4)


def test_direct_nvfp4_transform_requires_adaptive_weight_opt_in(monkeypatch):
    # Given: an explicit NVFP4 configuration for a dense checkpoint.
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4, group_size=16)
    monkeypatch.setenv("TRTLLM_ADAPTIVE_FP4_WEIGHT", "1")
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "0")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC13", "0")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC2", "0")

    # When: the serving path decides whether to bridge the config to PyTorch.
    enabled = _direct_nvfp4_transform_enabled(quant_config)

    # Then: an explicit adaptive stage enables the direct NVFP4 path.
    assert enabled


def test_ModelLoader():
    kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)

    # Test with HF model
    temp_dir = tempfile.TemporaryDirectory()

    def build_engine():
        args = TrtLlmArgs(model=llama_model_path,
                          kv_cache_config=kv_cache_config)
        model_loader = ModelLoader(args)
        engine_dir = model_loader(engine_dir=Path(temp_dir.name))
        assert engine_dir
        return engine_dir

    # Test with engine
    args = TrtLlmArgs(model=build_engine(), kv_cache_config=kv_cache_config)
    assert args.model_format is _ModelFormatKind.TLLM_ENGINE
    print(f'engine_dir: {args.model}')
    model_loader = ModelLoader(args)
    engine_dir = model_loader()
    assert engine_dir == args.model


def test_CachedModelLoader():
    # CachedModelLoader enables engine caching and multi-gpu building
    args = TrtLlmArgs(
        model=llama_model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        enable_build_cache=True)
    stats = LlmBuildStats()
    model_loader = CachedModelLoader(args, llm_build_stats=stats)
    engine_dir, _ = model_loader()
    assert engine_dir
    assert engine_dir.exists() and engine_dir.is_dir()
    model_format = get_model_format(engine_dir, trust_remote_code=True)
    assert model_format is _ModelFormatKind.TLLM_ENGINE


def test_LlmArgs_default_gpus_per_node():
    # default
    llm_args = TrtLlmArgs(model=llama_model_path)
    assert llm_args.gpus_per_node == torch.cuda.device_count()

    # set explicitly
    llm_args = TrtLlmArgs(model=llama_model_path, gpus_per_node=6)
    assert llm_args.gpus_per_node == 6


def test_AsyncQueue():
    queue = AsyncQueue()

    # put data to queue sync in a thread
    # async get data from queue in the current event loop
    # NOTE: the event loop in the two threads are different

    def put_data_to_queue():
        for i in range(10):
            time.sleep(0.1)
            queue.put(i)

    async def get_data_from_queue():
        for i in range(10):
            print(f"get: {queue.get()}")

    thread = threading.Thread(target=put_data_to_queue)
    thread.start()
    asyncio.run(get_data_from_queue())
    thread.join()
