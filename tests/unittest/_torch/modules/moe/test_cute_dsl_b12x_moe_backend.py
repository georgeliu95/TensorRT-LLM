# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Negative-path + dispatch tests for CuteDslB12xFusedMoE.

These checks run without a GPU: they verify the can_implement() gating
matrix, the SM120/SM121 + NVFP4 selection in create_moe.get_moe_cls (the
backend is selected on the `moe_backend=CUTEDSL` path when flashinfer
is importable, never from `moe_backend=CUTLASS`), and the hybrid
CUTLASS-prefill / b12x-decode dispatch predicate. Functional
correctness of the b12x kernel is covered by end-to-end model tests on
SM120/SM121 hardware.
"""

from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.modules.fused_moe.create_moe import get_moe_cls
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import CuteDslFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl_b12x import CuteDslB12xFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cutlass import CutlassFusedMoE
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

_FUSED_MOE_MODULE = "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl_b12x"


@pytest.mark.parametrize("sm_version", [80, 89, 90, 100, 103])
def test_can_implement_rejects_unsupported_sm(sm_version):
    """can_implement returns False on every SM outside the supported set."""
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=sm_version):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4)
    assert not ok
    assert reason is not None and f"SM{sm_version}" in reason


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_can_implement_accepts_supported_sm_with_nvfp4(sm_version):
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=sm_version):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4)
    assert ok
    assert reason is None


@pytest.mark.parametrize(
    "quant_algo",
    [
        None,
        QuantAlgo.FP8,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
    ],
)
def test_can_implement_rejects_non_nvfp4(quant_algo):
    """Only NVFP4 is supported; everything else must be turned away."""
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=120):
        ok, reason = CuteDslB12xFusedMoE.can_implement(quant_algo)
    assert not ok
    assert reason is not None and "NVFP4" in reason


def test_can_implement_rejects_swiglu_gptoss_style():
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=120):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4, swiglu_gptoss_style=True)
    assert not ok
    assert reason is not None and "swiglu_gptoss_style" in reason


@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_can_implement_rejects_unsupported_activation_dtype(dtype):
    with patch(f"{_FUSED_MOE_MODULE}.get_sm_version", return_value=120):
        ok, reason = CuteDslB12xFusedMoE.can_implement(QuantAlgo.NVFP4, dtype_activation=dtype)
    assert not ok
    assert reason is not None


def test_get_moe_cls_cutlass_path_never_auto_promotes():
    """Explicit ``moe_backend=CUTLASS`` always returns ``CutlassFusedMoE`` —
    no silent override to the b12x backend even on eligible hardware. b12x
    is opted into via ``moe_backend=CUTEDSL``."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTLASS"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


def test_get_moe_cls_cutedsl_falls_back_to_cutlass_on_unsupported_quant():
    """CUTEDSL + non-(fp8_block_scales|nvfp4) → warn + fall back to CutlassFusedMoE."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


def test_get_moe_cls_cutedsl_falls_back_to_cutlass_on_missing_quant():
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = None
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CutlassFusedMoE


def test_get_moe_cls_cutedsl_returns_plain_cutedsl_on_unsupported_sm():
    """CUTEDSL + NVFP4 + non-SM120/121 → plain CuteDslFusedMoE (the SM100/103
    cuteDSL backend); the b12x branch is bypassed."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=100):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslFusedMoE


@pytest.mark.parametrize("sm_version", sorted(CuteDslB12xFusedMoE._SUPPORTED_SM_VERSIONS))
def test_get_moe_cls_cutedsl_selects_b12x_on_supported_sm(sm_version):
    """CUTEDSL + NVFP4 + SM120/121 + flashinfer importable → CuteDslB12xFusedMoE."""
    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=sm_version):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslB12xFusedMoE


def test_get_moe_cls_cutedsl_falls_back_to_plain_cutedsl_when_flashinfer_missing(monkeypatch):
    """CUTEDSL + NVFP4 + SM120/121 + flashinfer NOT importable → CuteDslFusedMoE."""
    import builtins

    cfg = ModelConfig()
    cfg.moe_backend = "CUTEDSL"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    real_import = builtins.__import__

    def _raise_on_flashinfer(name, *args, **kwargs):
        if name == "flashinfer":
            raise ImportError("flashinfer not installed (simulated)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_on_flashinfer)
    with patch("tensorrt_llm._utils.get_sm_version", return_value=120):
        cls = get_moe_cls(cfg)
    assert cls is CuteDslFusedMoE


def test_get_moe_cls_trtllm_keeps_nvfp4_backend_by_default(monkeypatch):
    cfg = ModelConfig()
    cfg.moe_backend = "TRTLLM"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    monkeypatch.delenv("TRTLLM_MOE_FORCE_CUTEDSL", raising=False)
    monkeypatch.delenv("TRTLLM_SVDQUANT_NVFP4", raising=False)

    assert get_moe_cls(cfg) is TRTLLMGenFusedMoE


def test_get_moe_cls_trtllm_allows_explicit_cutedsl_override(monkeypatch):
    cfg = ModelConfig()
    cfg.moe_backend = "TRTLLM"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    monkeypatch.setenv("TRTLLM_MOE_FORCE_CUTEDSL", "1")

    assert get_moe_cls(cfg) is CuteDslFusedMoE


def test_get_moe_cls_trtllm_forces_cutedsl_for_nvfp4_svdquant(monkeypatch):
    """The SVD correction boundaries are available only in plain CuteDSL."""
    cfg = ModelConfig()
    cfg.moe_backend = "TRTLLM"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    monkeypatch.delenv("TRTLLM_MOE_FORCE_CUTEDSL", raising=False)
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")

    assert get_moe_cls(cfg) is CuteDslFusedMoE


def test_get_moe_cls_trtllm_ignores_svdquant_without_active_stage(
    monkeypatch,
):
    """The master switch alone cannot reroute TRTLLM when both stages are off."""
    cfg = ModelConfig()
    cfg.moe_backend = "TRTLLM"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    monkeypatch.delenv("TRTLLM_MOE_FORCE_CUTEDSL", raising=False)
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC13", "0")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC2", "0")

    assert get_moe_cls(cfg) is TRTLLMGenFusedMoE


def test_get_moe_cls_does_not_force_cutedsl_for_non_nvfp4_svdquant(
    monkeypatch,
):
    """The opt-in is scoped to NVFP4 and cannot rewrite another quant mode."""
    cfg = ModelConfig()
    cfg.moe_backend = "TRTLLM"
    cfg.quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")

    assert get_moe_cls(cfg) is TRTLLMGenFusedMoE


class _WeightRegisteringBackend(torch.nn.Module):

    def create_weights(self):
        self.register_parameter("svd_factor", torch.nn.Parameter(torch.ones(1)))


def _configurable_moe_stub(*, eplb: bool) -> ConfigurableMoE:
    wrapper = ConfigurableMoE.__new__(ConfigurableMoE)
    torch.nn.Module.__init__(wrapper)
    wrapper.dtype = torch.bfloat16
    wrapper.num_experts = 2
    wrapper.hidden_size = 32
    wrapper.intermediate_size = 24
    wrapper.reduce_results = False
    wrapper.aux_stream_dict = None
    wrapper.weight_loading_mode = None
    wrapper.apply_router_weight_on_input = False
    wrapper.activation_type = 0
    wrapper.layer_idx = 3
    wrapper.layer_idx_str = "3"
    wrapper.num_slots = 2
    wrapper.layer_load_balancer = object() if eplb else None
    wrapper.repeat_count = 1
    wrapper.repeat_idx = 0
    wrapper.initial_local_expert_ids = [0, 1]
    wrapper.initial_global_assignments = [0, 1]
    wrapper.slot_start = 0
    wrapper.slot_end = 2
    wrapper.expert_size_per_partition = 2
    return wrapper


@pytest.mark.parametrize(
    ("mode", "message"),
    [("dwdp", "VA-DWDP"), ("eplb", "EPLB")],
)
def test_svdquant_rejects_remapping_before_backend_parameter_registration(
    monkeypatch,
    mode,
    message,
):
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC13", "1")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC2", "1")
    wrapper = _configurable_moe_stub(eplb=mode == "eplb")
    backend = _WeightRegisteringBackend()
    model_config = ModelConfig()
    wrapper.model_config = model_config
    if mode == "dwdp":
        wrapper._should_enable_dwdp = lambda: True

    with (
        patch(
            "tensorrt_llm._torch.modules.fused_moe.create_moe.resolve_moe_cls",
            return_value=object,
        ),
        patch(
            "tensorrt_llm._torch.modules.fused_moe.create_moe.create_moe_backend",
            return_value=backend,
        ),
        patch(
            "tensorrt_llm._torch.modules.fused_moe.configurable_moe.get_global_dwdp_manager",
            return_value=object() if mode == "dwdp" else None,
        ),
        pytest.raises(RuntimeError, match=message),
    ):
        wrapper._create_and_sync_backend(
            model_config=model_config,
            routing_method=object(),
            override_quant_config=None,
        )

    assert dict(backend.named_parameters()) == {}


def test_inactive_svdquant_does_not_block_backend_parameter_registration(
    monkeypatch,
):
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC13", "0")
    monkeypatch.setenv("TRTLLM_SVDQUANT_FC2", "0")
    wrapper = _configurable_moe_stub(eplb=True)
    backend = _WeightRegisteringBackend()
    model_config = ModelConfig()
    wrapper.model_config = model_config

    with (
        patch(
            "tensorrt_llm._torch.modules.fused_moe.create_moe.resolve_moe_cls",
            return_value=object,
        ),
        patch(
            "tensorrt_llm._torch.modules.fused_moe.create_moe.create_moe_backend",
            return_value=backend,
        ),
    ):
        wrapper._create_and_sync_backend(
            model_config=model_config,
            routing_method=object(),
            override_quant_config=None,
        )

    assert set(dict(backend.named_parameters())) == {"svd_factor"}


class _SvdquantConfigurableMoeStub:

    def __init__(self, *, enable_dwdp=False, eplb=False):
        self.enable_dwdp = enable_dwdp
        self._eplb = eplb

    def _using_load_balancer(self):
        return self._eplb


@pytest.mark.parametrize(
    ("enable_dwdp", "eplb", "message"),
    [(True, False, "VA-DWDP"), (False, True, "EPLB")],
)
def test_svdquant_rejects_dynamic_expert_remapping(
    monkeypatch,
    enable_dwdp,
    eplb,
    message,
):
    """ConfigurableMoE state is validated after wrapper synchronization."""
    monkeypatch.setenv("TRTLLM_SVDQUANT_NVFP4", "1")
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    wrapper = _SvdquantConfigurableMoeStub(enable_dwdp=enable_dwdp,
                                           eplb=eplb)

    with pytest.raises(RuntimeError, match=message):
        backend.validate_configurable_moe(wrapper)


# --------------------------------------------------------------------------
# Hybrid CUTLASS-prefill / b12x-decode dispatch predicate tests
#
# ``_route_to_cutlass`` is a pure shape predicate on its input ``x``; we test
# it via a stub that holds the class constant, sidestepping the full
# CutlassFusedMoE constructor (which needs a routing method, real model
# config, etc.).
# --------------------------------------------------------------------------


class _RoutePredicateStub:
    """Minimal carrier for ``_PREFILL_VIA_CUTLASS_THRESHOLD`` so we can call
    the unbound ``_route_to_cutlass`` without instantiating the whole MoE
    backend."""

    _PREFILL_VIA_CUTLASS_THRESHOLD = CuteDslB12xFusedMoE._PREFILL_VIA_CUTLASS_THRESHOLD

    _route_to_cutlass = CuteDslB12xFusedMoE._route_to_cutlass


def test_dispatch_routes_prefill_shape_via_cutlass():
    stub = _RoutePredicateStub()
    x = torch.empty(_RoutePredicateStub._PREFILL_VIA_CUTLASS_THRESHOLD, 1024)
    assert stub._route_to_cutlass(x) is True


def test_dispatch_just_below_threshold_takes_b12x():
    stub = _RoutePredicateStub()
    x = torch.empty(_RoutePredicateStub._PREFILL_VIA_CUTLASS_THRESHOLD - 1, 1024)
    assert stub._route_to_cutlass(x) is False


def test_dispatch_decode_shape_takes_b12x():
    stub = _RoutePredicateStub()
    x = torch.empty(1, 1024)
    assert stub._route_to_cutlass(x) is False


def test_dispatch_rejects_non_tensor():
    """Non-tensor inputs (e.g. Fp4QuantizedTensor) stay on the b12x path
    so the existing ValueError surfaces in quantize_input."""
    stub = _RoutePredicateStub()
    assert stub._route_to_cutlass(object()) is False
