# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-side contract for the fused runtime-alpha grouped-GEMM ABI.

The NVFP4 CuteDSL MoE path used to correct the per-expert alpha with a
standalone Torch ``Div``/``Mul`` pair whenever the activation was quantized at
runtime.  The grouped GEMMs now take that correction as two optional
single-element float32 operands and fold it in their epilogue, so what these
cases pin is the wiring: the *static* checkpoint alpha reaches the kernel
untouched, the two scalars carry the runtime correction, and the feature-off
path passes neither.  Device numerics live in
``tests/unittest/_torch/thop/parallel/test_cute_dsl_moe.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import (
    CuteDslFusedMoE,
    _runtime_alpha_scalar,
)
from tensorrt_llm._torch.utils import ActivationType, AuxStreamType, EventType


# ``run_moe_nvfp4_impl`` reinterprets the packed activation before any stub can
# intercept it, so the wiring cases need the real dtype.
_HAS_FP4_DTYPE = hasattr(torch, "float4_e2m1fn_x2")

_FC13_CHECKPOINT_SCALE = 3.25
_FC13_RUNTIME_SCALE = 0.5
_FC2_CHECKPOINT_SCALE = 1.75
_FC2_RUNTIME_SCALE = 0.25


def test_runtime_alpha_scalar_is_a_metadata_only_view() -> None:
    """The ABI wants a one-element float32 tensor and no extra kernel."""
    amax_buf = torch.tensor([7.0, 11.0], dtype=torch.float32)

    shaped = _runtime_alpha_scalar(amax_buf[1:2])

    assert shaped.shape == (1, )
    assert shaped.dtype == torch.float32
    # A view, not a copy: no device work and nothing to synchronize on.
    assert shaped.data_ptr() == amax_buf[1:2].data_ptr()
    assert shaped.item() == 11.0


def test_runtime_alpha_scalar_reshapes_a_zero_dim_checkpoint_scale() -> None:
    """Checkpoint scales are 0-dim parameters; the ABI needs shape ``(1,)``."""
    scale = torch.tensor(2.5, dtype=torch.float32)

    shaped = _runtime_alpha_scalar(scale)

    assert shaped.shape == (1, )
    assert shaped.data_ptr() == scale.data_ptr()


def _run_moe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime_fc13: str | None,
    runtime_fc2: str | None,
    use_fused_finalize: bool,
) -> dict:
    """Drive ``run_moe_nvfp4_impl`` once with every device op stubbed.

    ``runtime_fc13``/``runtime_fc2`` select the runtime activation
    quantization rule (``"standard"`` for native rule 0, ``"4o6"`` for the
    adaptive rule 1) or leave the legacy feature-off path in place.
    """
    tile_size, num_tiles, num_experts = 2, 3, 2
    rows = num_tiles * tile_size
    hidden, inter = 64, 128
    calls: dict[str, dict] = {}

    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    backend.num_slots = num_experts
    backend.activation_type = int(ActivationType.Swiglu)
    backend.use_fused_finalize = use_fused_finalize
    backend.fc31_input_scale = torch.tensor(_FC13_CHECKPOINT_SCALE)
    backend.fc2_input_scale = torch.tensor(_FC2_CHECKPOINT_SCALE)
    backend.mapping = SimpleNamespace(moe_ep_size=1)
    backend.aux_stream_dict = {AuxStreamType.MoeOutputMemset: None}
    backend.event_dict = {
        EventType.Main: SimpleNamespace(record=lambda: None,
                                        wait=lambda: None),
        EventType.MoeOutputMemset: SimpleNamespace(record=lambda: None,
                                                   wait=lambda: None),
    }
    if runtime_fc13 is not None:
        # Exactly what ``quantize_input`` stores: the second slot of the
        # quantizer's ``[amax, quantRange / amax]`` buffer, as a 1-element view.
        backend._runtime_fc13_global_scale = torch.tensor(
            [1.0, _FC13_RUNTIME_SCALE], dtype=torch.float32)[1:2]

    tile_map = torch.tensor([0, 1, 1], dtype=torch.int32)
    mn_limit = torch.tensor([2, 4, 6], dtype=torch.int32)
    gate = torch.tensor([num_tiles], dtype=torch.int32)

    def moe_sort(**_kwargs):
        return (tile_map, mn_limit, torch.zeros(rows, dtype=torch.int32),
                torch.zeros(rows, dtype=torch.int32),
                torch.tensor(rows, dtype=torch.int32), gate)

    def act_fusion_fp4(**kwargs):
        calls["fc13"] = kwargs
        return (torch.zeros(rows, inter // 2, dtype=torch.uint8),
                torch.zeros(rows, inter // 16, dtype=torch.uint8))

    fc2_input_bf16 = torch.zeros(rows, inter, dtype=torch.bfloat16)
    runtime_fc2_amax = torch.tensor(
        [1.0, _FC2_RUNTIME_SCALE], dtype=torch.float32)

    def act_fusion_bf16_amax(**kwargs):
        calls["fc13"] = kwargs
        calls["fc13_producer"] = {
            "output": fc2_input_bf16,
            "amax_scale": runtime_fc2_amax,
        }
        return fc2_input_bf16, runtime_fc2_amax

    def reject_full_quantize(*_args, **_kwargs):
        pytest.fail("FC13 producer amax must bypass full runtime quantization")

    def quantize_phase2(input_: torch.Tensor, amax_scale: torch.Tensor,
                        vec_size: int, swizzled: bool,
                        scale_rule: int):
        calls["fc13_phase2"] = {
            "input": input_,
            "amax_scale": amax_scale,
            "vec_size": vec_size,
            "swizzled": swizzled,
            "scale_rule": scale_rule,
        }
        return (torch.zeros(rows, inter // 2, dtype=torch.uint8),
                torch.zeros(rows, inter // 16, dtype=torch.uint8))

    def grouped_gemm(**kwargs):
        calls["fc2"] = kwargs
        return torch.zeros(rows, hidden, dtype=torch.bfloat16)

    def finalize_inplace(**kwargs):
        calls["fc2"] = kwargs

    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", moe_sort, raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell",
        act_fusion_fp4,
        raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_bf16_amax_blackwell",
        act_fusion_bf16_amax,
        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "fp4_quantize_fused",
                        reject_full_quantize,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "fp4_quantize_phase2",
                        quantize_phase2,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "cute_dsl_nvfp4_grouped_gemm_blackwell",
                        grouped_gemm,
                        raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell",
        finalize_inplace,
        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "moe_output_memset_inplace",
                        lambda **_kwargs: None,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "moe_unpermute_inplace",
                        lambda **_kwargs: None,
                        raising=False)
    # ``record_stream`` is CUDA-only; the fused-finalize branch calls it before
    # any stub can intervene.
    monkeypatch.setattr(torch.Tensor,
                        "record_stream",
                        lambda self, stream: None,
                        raising=False)

    for name, value in (("TRTLLM_NVFP4_RUNTIME_ACTIVATION", runtime_fc2),
                        ("TRTLLM_ADAPTIVE_FP4_FC2", "1" if runtime_fc2 else
                         None)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    weight_view = SimpleNamespace(
        expert_size_per_partition=num_experts,
        slot_start=0,
        fc1_global_scale=torch.full((num_experts, ), 2.0),
        fc2_global_scale=torch.full((num_experts, ), 4.0),
        w3_w1_weight=torch.zeros(num_experts, 2 * inter, hidden // 2,
                                 dtype=torch.uint8),
        fc1_weight_scale=torch.zeros(4, dtype=torch.uint8),
        w2_weight=torch.zeros(num_experts, hidden, inter // 2,
                              dtype=torch.uint8),
        fc2_weight_scale=torch.zeros(4, dtype=torch.uint8),
    )

    backend.run_moe_nvfp4_impl(
        x=torch.zeros(rows, hidden // 2, dtype=torch.uint8),
        token_selected_experts=torch.zeros(rows, 1, dtype=torch.int32),
        token_final_scales=torch.ones(rows, 1),
        x_sf=torch.zeros(rows, hidden // 16, dtype=torch.uint8),
        moe_output=torch.zeros(rows, hidden, dtype=torch.bfloat16),
        weight_view=weight_view,
        tile_size=tile_size,
    )
    return {"calls": calls, "weight_view": weight_view}


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
@pytest.mark.parametrize("runtime_fc13", ["standard", "4o6"])
@pytest.mark.parametrize("runtime_fc2", [None, "standard", "4o6"])
def test_fc13_gemm_gets_the_static_alpha_plus_runtime_scalars(
        monkeypatch: pytest.MonkeyPatch, runtime_fc13: str,
        runtime_fc2: str | None) -> None:
    """Native rule 0 and adaptive rule 1 share one contract.

    The alpha handed to FC13 must be the checkpoint tensor itself -- a
    corrected copy would mean a standalone Div/Mul produced it -- and the
    correction must arrive as the two runtime scalars instead.
    """
    result = _run_moe(monkeypatch,
                      runtime_fc13=runtime_fc13,
                      runtime_fc2=runtime_fc2,
                      use_fused_finalize=False)
    fc13 = result["calls"]["fc13"]

    assert fc13["alpha"] is result["weight_view"].fc1_global_scale
    assert fc13["alpha_numerator"].item() == _FC13_CHECKPOINT_SCALE
    assert fc13["alpha_denominator"].item() == _FC13_RUNTIME_SCALE


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
@pytest.mark.parametrize("runtime_fc2", ["standard", "4o6"])
@pytest.mark.parametrize("use_fused_finalize", [False, True])
def test_fc2_gemm_gets_the_static_alpha_plus_runtime_scalars(
        monkeypatch: pytest.MonkeyPatch, runtime_fc2: str,
        use_fused_finalize: bool) -> None:
    """Both FC2 epilogues -- plain grouped GEMM and fused finalize."""
    result = _run_moe(monkeypatch,
                      runtime_fc13=None,
                      runtime_fc2=runtime_fc2,
                      use_fused_finalize=use_fused_finalize)
    fc2 = result["calls"]["fc2"]

    assert fc2["alpha"] is result["weight_view"].fc2_global_scale
    assert fc2["alpha_numerator"].item() == _FC2_CHECKPOINT_SCALE
    assert fc2["alpha_denominator"].item() == _FC2_RUNTIME_SCALE


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
@pytest.mark.parametrize(
    ("runtime_fc2", "scale_rule", "quant_range"),
    [("standard", 0, 448.0 * 6.0), ("4o6", 1, 1536.0)],
)
def test_runtime_fc2_uses_fc13_producer_amax_and_phase2_only(
        monkeypatch: pytest.MonkeyPatch, runtime_fc2: str, scale_rule: int,
        quant_range: float) -> None:
    """The normal FC13 path must not rescan its materialized BF16 output."""
    result = _run_moe(monkeypatch,
                      runtime_fc13=None,
                      runtime_fc2=runtime_fc2,
                      use_fused_finalize=False)
    calls = result["calls"]

    assert calls["fc13"]["quant_range"] == quant_range
    assert calls["fc13_phase2"] == {
        "input": calls["fc13_producer"]["output"],
        "amax_scale": calls["fc13_producer"]["amax_scale"],
        "vec_size": 16,
        "swizzled": True,
        "scale_rule": scale_rule,
    }


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
@pytest.mark.parametrize("runtime_fc2", ["standard", "4o6"])
def test_runtime_fc2_never_dequantizes_the_fc13_activation(
        monkeypatch: pytest.MonkeyPatch, runtime_fc2: str) -> None:
    """Runtime activation quantization starts from the fused BF16 epilogue.

    Both the native rule and adaptive 4o6 rule must feed FC2 by quantizing the
    current BF16 SwiGLU output directly.  A standard-NVFP4 -> BF16 round trip
    here would add a kernel and would no longer represent the production data
    flow.
    """

    def reject_dequantize(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("runtime FC2 activation must not be dequantized")

    monkeypatch.setattr(
        "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl."
        "_dequant_nvfp4_cutedsl", reject_dequantize)

    result = _run_moe(monkeypatch,
                      runtime_fc13=runtime_fc2,
                      runtime_fc2=runtime_fc2,
                      use_fused_finalize=False)

    assert result["calls"]["fc13_phase2"]["input"] is (
        result["calls"]["fc13_producer"]["output"])


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
@pytest.mark.parametrize("use_fused_finalize", [False, True])
def test_feature_off_passes_no_runtime_scalars(
        monkeypatch: pytest.MonkeyPatch, use_fused_finalize: bool) -> None:
    """Legacy behaviour: no runtime quantization, no correction operands."""
    result = _run_moe(monkeypatch,
                      runtime_fc13=None,
                      runtime_fc2=None,
                      use_fused_finalize=use_fused_finalize)

    for stage in ("fc13", "fc2"):
        call = result["calls"][stage]
        assert call["alpha_numerator"] is None
        assert call["alpha_denominator"] is None
    assert result["calls"]["fc13"]["alpha"] is (
        result["weight_view"].fc1_global_scale)
    assert result["calls"]["fc2"]["alpha"] is (
        result["weight_view"].fc2_global_scale)


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
def test_runtime_scalars_reproduce_the_legacy_corrected_alpha(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``alpha * numerator / denominator`` is the value the epilogue must use."""
    result = _run_moe(monkeypatch,
                      runtime_fc13="4o6",
                      runtime_fc2="4o6",
                      use_fused_finalize=False)
    weight_view = result["weight_view"]

    for stage, static_alpha, checkpoint_scale, runtime_scale in (
        ("fc13", weight_view.fc1_global_scale, _FC13_CHECKPOINT_SCALE,
         _FC13_RUNTIME_SCALE),
        ("fc2", weight_view.fc2_global_scale, _FC2_CHECKPOINT_SCALE,
         _FC2_RUNTIME_SCALE),
    ):
        call = result["calls"][stage]
        effective = (call["alpha"] * call["alpha_numerator"] /
                     call["alpha_denominator"])
        legacy = static_alpha * (torch.tensor(checkpoint_scale) /
                                 torch.tensor(runtime_scale))
        torch.testing.assert_close(effective, legacy)
