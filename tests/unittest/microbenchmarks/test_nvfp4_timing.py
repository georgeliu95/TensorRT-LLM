# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timing-boundary regressions for the NVFP4 MoE benchmark."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


_REPO_ROOT = Path(__file__).parents[3]
_EAGER_PATH = _REPO_ROOT / "tests" / "microbenchmarks" / "bench_moe" / "timing" / "eager.py"
_AUTOTUNE_PATH = _EAGER_PATH.with_name("autotune.py")


def _load_eager_module() -> ModuleType:
    package_names = (
        "bench_moe",
        "bench_moe.timing",
    )
    modules = {name: ModuleType(name) for name in package_names}
    modules["bench_moe"].__path__ = [
        str(_REPO_ROOT / "tests" / "microbenchmarks" / "bench_moe")
    ]
    modules["bench_moe.timing"].__path__ = [str(_EAGER_PATH.parent)]

    utils = ModuleType("bench_moe.utils")
    utils._maybe_print_rank0 = lambda _message: None
    utils._sync = lambda: None
    tllm_utils = ModuleType("tensorrt_llm._utils")
    tllm_utils.nvtx_range_debug = contextlib.nullcontext

    spec = importlib.util.spec_from_file_location("bench_moe.timing.eager", _EAGER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_EAGER_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            **modules,
            "bench_moe.utils": utils,
            "tensorrt_llm": ModuleType("tensorrt_llm"),
            "tensorrt_llm._utils": tllm_utils,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def _load_autotune_module() -> ModuleType:
    autotuner = ModuleType("tensorrt_llm._torch.autotuner")
    autotuner.AutoTuner = object
    autotuner.autotune = contextlib.nullcontext
    spec = importlib.util.spec_from_file_location(
        "_test_bench_moe_timing_autotune",
        _AUTOTUNE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_AUTOTUNE_PATH}")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "tensorrt_llm": ModuleType("tensorrt_llm"),
            "tensorrt_llm._torch": ModuleType("tensorrt_llm._torch"),
            "tensorrt_llm._torch.autotuner": autotuner,
            spec.name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


def test_scored_forward_range_excludes_l2_flush_and_setup() -> None:
    eager = _load_eager_module()
    events: list[str] = []

    class FlushBuffer:
        def zero_(self) -> None:
            events.append("flush")

    class Event:
        def __init__(self, name: str) -> None:
            self.name = name

        def record(self) -> None:
            events.append(self.name)

    @contextlib.contextmanager
    def nvtx_range(name: str):
        events.append(f"enter:{name}")
        yield
        events.append(f"exit:{name}")

    eager.nvtx_range_debug = nvtx_range
    eager._run_scored_forward_iteration(
        lambda: events.append("forward"),
        FlushBuffer(),
        Event("start"),
        Event("end"),
    )

    assert events == [
        "flush",
        "enter:bench_moe.scored_forward",
        "start",
        "forward",
        "end",
        "exit:bench_moe.scored_forward",
    ]


def test_nvfp4_comparison_does_not_load_a_cross_process_tactic_cache(
    monkeypatch,
) -> None:
    autotune = _load_autotune_module()
    monkeypatch.setenv("NVFP4_BENCHMARK_STRATEGY", "4o6_svdq_r64")

    assert autotune._autotune_cache_path() is None


def test_generic_benchmark_keeps_its_existing_disk_tactic_cache(monkeypatch) -> None:
    autotune = _load_autotune_module()
    monkeypatch.delenv("NVFP4_BENCHMARK_STRATEGY", raising=False)

    assert autotune._autotune_cache_path().endswith(
        "bench_moe_autotuner_cache.json"
    )
