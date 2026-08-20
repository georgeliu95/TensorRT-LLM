# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused host-side tests for FC13 adaptive 4o6 wiring."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe import fused_moe_cute_dsl as moe
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    interleave_linear_and_gate,
)
from tensorrt_llm._torch.utils import ActivationType, swizzle_sf, unswizzle_sf
from tensorrt_llm._utils import get_sm_version


_SM100_AVAILABLE = torch.cuda.is_available() and get_sm_version() in (100, 103)


@pytest.mark.parametrize(
    ("mode", "scale_rule", "quant_range"),
    [
        ("standard", 0, 448.0 * 6.0),
        ("4o6", 1, 1536.0),
    ],
)
def test_runtime_activation_mode_selects_quantization_rule(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    scale_rule: int,
    quant_range: float,
) -> None:
    # Given a benchmark-selected runtime activation mode.
    monkeypatch.setenv("TRTLLM_NVFP4_RUNTIME_ACTIVATION", mode)

    # When the FC13 and FC2 runtime quantizers resolve their configuration.
    fc13 = moe._runtime_activation_quantization("TRTLLM_ADAPTIVE_FP4")
    fc2 = moe._runtime_activation_quantization("TRTLLM_ADAPTIVE_FP4_FC2")

    # Then both stages quantize the current tensor with the requested NVFP4 rule.
    assert fc13 is not None
    assert fc2 is not None
    assert (fc13.scale_rule, fc13.quant_range) == (scale_rule, quant_range)
    assert fc2 == fc13


@pytest.mark.parametrize(
    ("mode", "expected_rule", "expected_range"),
    [
        ("standard", 0, 448.0 * 6.0),
        ("4o6", 1, 1536.0),
    ],
)
def test_runtime_swiglu_quantization_fuses_activation_into_phase1(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_rule: int,
    expected_range: float,
) -> None:
    """FC13 activation is one fused dispatch that still returns its BF16."""
    # Given an FC13 preactivation and routing metadata.
    monkeypatch.setenv("TRTLLM_NVFP4_RUNTIME_ACTIVATION", mode)
    rows, interm, tile_size = 4, 32, 2
    preact = torch.randn(rows, 2 * interm, dtype=torch.bfloat16)
    tile_limits = torch.tensor([2, 4], dtype=torch.int32)
    live_tiles = torch.tensor([2], dtype=torch.int32)
    activated = torch.randn(rows, interm, dtype=torch.bfloat16)
    fp4 = torch.zeros(rows, interm // 2, dtype=torch.uint8)
    sf = torch.zeros(rows, interm // 16, dtype=torch.uint8)
    amax = torch.tensor([7.0, 1536.0 / 7.0], dtype=torch.float32)
    observed: dict[str, torch.Tensor | int | bool | float] = {}

    def fail_if_called(*_args: object, **_kwargs: object) -> torch.Tensor:
        pytest.fail("the standalone BF16 SwiGLU kernel must not be launched")

    def swiglu_quantize(
        input_: torch.Tensor, vec_size: int, swizzled: bool, scale_rule: int,
        quant_range: float, eps: float, max_blocks: int,
        limits: torch.Tensor, live: torch.Tensor, routing_tile_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        observed["preact"] = input_
        observed["vec_size"] = vec_size
        observed["swizzled"] = swizzled
        observed["scale_rule"] = scale_rule
        observed["quant_range"] = quant_range
        observed["eps"] = eps
        observed["max_blocks"] = max_blocks
        observed["limits"] = limits
        observed["live"] = live
        observed["tile"] = routing_tile_size
        return fp4, sf, amax, activated

    monkeypatch.setattr(torch.ops.trtllm, "moe_swiglu", fail_if_called,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "fp4_quantize_fused", fail_if_called,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "fp4_swiglu_quantize_fused",
                        swiglu_quantize, raising=False)

    # When FC13 activation is converted directly to adaptive NVFP4.
    quantization = moe._runtime_activation_quantization(
        "TRTLLM_ADAPTIVE_FP4_FC2")
    assert quantization is not None
    output, output_sf, output_amax, output_bf16 = (
        moe._runtime_swiglu_nvfp4_quantize(
            preact, tile_limits, live_tiles, tile_size, quantization))

    # Then one dispatch consumed the preactivation and returned the BF16 the
    # FC2 low-rank correction needs, under the mode's own scale rule.
    assert output is fp4
    assert output_sf is sf
    assert output_amax is amax
    assert output_bf16 is activated
    assert observed == {
        "preact": preact,
        "vec_size": 16,
        "swizzled": True,
        "scale_rule": expected_rule,
        "quant_range": expected_range,
        "eps": 1e-12,
        "max_blocks": 0,
        "limits": tile_limits,
        "live": live_tiles,
        "tile": tile_size,
    }


def test_runtime_swiglu_quantization_requires_the_fused_operator(
    monkeypatch: pytest.MonkeyPatch, ) -> None:
    """A build without the fused op must say so instead of silently degrading."""
    # Given a build whose fused SwiGLU quantization operator is missing.
    monkeypatch.setenv("TRTLLM_NVFP4_RUNTIME_ACTIVATION", "4o6")
    monkeypatch.setattr(
        torch.ops, "trtllm",
        SimpleNamespace(fp4_quantize_fused=lambda *a, **k: None))
    quantization = moe._runtime_activation_quantization(
        "TRTLLM_ADAPTIVE_FP4_FC2")
    assert quantization is not None

    # When the FC13 epilogue asks for a runtime-quantized activation.
    with pytest.raises(RuntimeError, match="fp4_swiglu_quantize_fused"):
        moe._runtime_swiglu_nvfp4_quantize(
            torch.randn(4, 64, dtype=torch.bfloat16),
            torch.tensor([2, 4], dtype=torch.int32),
            torch.tensor([2], dtype=torch.int32),
            2,
            quantization,
        )


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="MoE runtime quantization requires SM100/SM103.")
@pytest.mark.parametrize(
    ("scale_rule", "quant_range"),
    [
        (0, 448.0 * 6.0),
        (1, 1536.0),
    ],
)
def test_moe_runtime_amax_excludes_uneven_routing_padding(
    scale_rule: int,
    quant_range: float,
) -> None:
    """Routing padding must not influence FC2's dynamic global scale."""
    # Given two uneven routing tiles whose padding contains a larger sentinel.
    tile_size, rows, cols = 128, 256, 64
    activated = torch.full((rows, cols),
                           1024.0,
                           dtype=torch.bfloat16,
                           device="cuda")
    activated[:3].fill_(2.0)
    activated[tile_size:tile_size + 2].fill_(-4.0)
    limits = torch.tensor([3, tile_size + 2],
                          dtype=torch.int32,
                          device="cuda")
    live_tiles = torch.tensor([2], dtype=torch.int32, device="cuda")

    # When persistent runtime quantization uses the MoE routing mask.
    _, _, amax = torch.ops.trtllm.fp4_quantize_fused(
        activated, 16, False, True, scale_rule, quant_range, 1e-12, 0, 0, limits,
        live_tiles, tile_size)

    # Then only the five valid rows contribute to the reduction.
    assert amax[0].item() == pytest.approx(4.0)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="MoE runtime quantization requires SM100/SM103.")
def test_moe_adaptive_amax_replays_in_cuda_graph() -> None:
    """The routing-aware persistent kernel must consume replay-time values."""
    # Given a warmed uneven-routing call with fixed graph-safe metadata.
    tile_size, rows, cols = 128, 256, 64
    activated = torch.full((rows, cols),
                           1024.0,
                           dtype=torch.bfloat16,
                           device="cuda")
    activated[:5].fill_(3.0)
    limits = torch.tensor([5, tile_size],
                          dtype=torch.int32,
                          device="cuda")
    live_tiles = torch.tensor([1], dtype=torch.int32, device="cuda")

    def quantize() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp4_quantize_fused(
            activated, 16, False, True, 1, 1536.0, 1e-12, 0, 0, limits,
            live_tiles, tile_size)

    quantize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, _, captured_amax = quantize()

    # When replay sees new valid values but the same large padding sentinel.
    activated[:5].fill_(7.0)
    graph.replay()
    torch.cuda.synchronize()

    # Then the persistent retirement counter reset and replay recomputed amax.
    assert captured_amax[0].item() == pytest.approx(7.0)


def test_phase2_only_quantize_fake_shape_contract() -> None:
    """The opt-in phase2 op exposes packed data and swizzled SF metadata."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        activated = torch.empty((7, 64),
                                dtype=torch.bfloat16,
                                device="cuda")
        amax_scale = torch.empty((2, ), dtype=torch.float32, device="cuda")
        packed, sf = torch.ops.trtllm.fp4_quantize_phase2(
            activated, amax_scale, 16, True, 1)

    assert packed.shape == (7, 32)
    assert packed.dtype == torch.uint8
    assert sf.ndim == 1
    assert sf.numel() > 0
    assert sf.dtype == torch.uint8


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="Phase2-only NVFP4 quantization requires SM100/SM103.")
@pytest.mark.parametrize(
    ("scale_rule", "quant_range"),
    [
        (0, 448.0 * 6.0),
        (1, 1536.0),
    ],
)
def test_phase2_only_quantize_matches_full_runtime_quantize(
    scale_rule: int,
    quant_range: float,
) -> None:
    """Skipping phase 1 must not change packed FP4 or its scale factors."""
    torch.manual_seed(11)
    activated = torch.randn((256, 512),
                            dtype=torch.bfloat16,
                            device="cuda")

    expected_fp4, expected_sf, amax_scale = (
        torch.ops.trtllm.fp4_quantize_fused(
            activated,
            16,
            False,
            True,
            scale_rule,
            quant_range,
            1e-12,
            0,
            1,
        ))
    actual_fp4, actual_sf = torch.ops.trtllm.fp4_quantize_phase2(
        activated, amax_scale, 16, True, scale_rule)

    assert torch.equal(actual_fp4.view(torch.uint8),
                       expected_fp4.view(torch.uint8))
    assert torch.equal(actual_sf.view(torch.uint8),
                       expected_sf.view(torch.uint8))


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="Phase2-only NVFP4 quantization requires SM100/SM103.")
def test_phase2_only_quantize_replays_input_and_scale_in_cuda_graph() -> None:
    """Graph replay consumes current input and the producer-published scale."""
    rows, cols = 256, 512
    activated = torch.randn((rows, cols),
                            dtype=torch.bfloat16,
                            device="cuda")
    amax_scale = torch.tensor([4.0, 1536.0 / 4.0],
                              dtype=torch.float32,
                              device="cuda")

    torch.ops.trtllm.fp4_quantize_phase2(activated, amax_scale, 16, True, 1)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_fp4, captured_sf = torch.ops.trtllm.fp4_quantize_phase2(
            activated, amax_scale, 16, True, 1)

    activated.copy_(torch.randn_like(activated))
    amax_scale.copy_(torch.tensor([8.0, 1536.0 / 8.0],
                                  dtype=torch.float32,
                                  device="cuda"))
    graph.replay()
    torch.cuda.synchronize()
    replay_fp4 = captured_fp4.clone()
    replay_sf = captured_sf.clone()

    expected_fp4, expected_sf = torch.ops.trtllm.fp4_quantize_phase2(
        activated, amax_scale, 16, True, 1)
    assert torch.equal(replay_fp4.view(torch.uint8),
                       expected_fp4.view(torch.uint8))
    assert torch.equal(replay_sf.view(torch.uint8),
                       expected_sf.view(torch.uint8))


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="Phase2-only NVFP4 quantization requires SM100/SM103.")
def test_phase2_only_quantize_rejects_unimplemented_configs() -> None:
    activated = torch.randn((4, 64),
                            dtype=torch.bfloat16,
                            device="cuda")
    amax_scale = torch.tensor([2.0, 768.0],
                              dtype=torch.float32,
                              device="cuda")

    with pytest.raises(RuntimeError, match="swizzled"):
        torch.ops.trtllm.fp4_quantize_phase2(activated, amax_scale, 16, False,
                                             1)
    with pytest.raises(RuntimeError, match="standard rule"):
        torch.ops.trtllm.fp4_quantize_phase2(activated, amax_scale, 16, True,
                                             2)


def _reference_swiglu(preactivation: torch.Tensor) -> torch.Tensor:
    """SwiGLU as the FC13 epilogue defines it: linear half first, gate second."""
    linear, gate = preactivation.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * linear).to(torch.bfloat16)


def _uneven_routing_preactivation(
    rows: int,
    interm: int,
    tile_size: int,
    live_rows: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a preactivation whose routing padding carries a loud sentinel."""
    preact = torch.randn(rows, 2 * interm, dtype=torch.bfloat16, device="cuda")
    num_tiles = rows // tile_size
    assert rows % tile_size == 0
    assert len(live_rows) <= num_tiles
    limits = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    limits[:len(live_rows)] = torch.tensor(
        [tile * tile_size + count for tile, count in enumerate(live_rows)],
        dtype=torch.int32,
        device="cuda")
    live_tiles = torch.tensor([len(live_rows)], dtype=torch.int32,
                              device="cuda")
    live_mask = torch.zeros(rows, dtype=torch.bool, device="cuda")
    for tile, count in enumerate(live_rows):
        live_mask[tile * tile_size:tile * tile_size + count] = True
    # Padding rows must not be able to hide inside the live value range.
    preact[~live_mask] = 512.0
    return preact, limits, live_tiles, live_mask


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="MoE runtime quantization requires SM100/SM103.")
@pytest.mark.parametrize(
    ("scale_rule", "quant_range"),
    [
        (0, 448.0 * 6.0),
        (1, 1536.0),
    ],
)
def test_fused_swiglu_quantize_matches_reference_under_uneven_routing(
    scale_rule: int,
    quant_range: float,
) -> None:
    """The fused activation must equal the standalone SwiGLU it replaces."""
    # Given two uneven routing tiles plus one fully exited tile.
    rows, interm, tile_size = 384, 512, 128
    preact, limits, live_tiles, live_mask = _uneven_routing_preactivation(
        rows, interm, tile_size, (3, tile_size))

    # When FC13 quantizes its activation in a single fused dispatch.
    fp4, sf, amax, activated = torch.ops.trtllm.fp4_swiglu_quantize_fused(
        preact, 16, True, scale_rule, quant_range, 1e-12, 0, limits, live_tiles,
        tile_size)

    # Then the materialized BF16 matches SwiGLU on every live row, including
    # the tile that routing left only partially filled.
    reference = _reference_swiglu(preact)
    assert activated.shape == (rows, interm)
    assert activated.dtype == torch.bfloat16
    torch.testing.assert_close(activated[live_mask],
                               reference[live_mask],
                               rtol=8e-3,
                               atol=0.0)

    # And the dynamic scale sees the live rows only -- the padding sentinel
    # would dominate the reduction if it leaked in.
    live_amax = reference[live_mask].abs().max().float()
    assert amax[0].item() == pytest.approx(live_amax.item(), rel=1e-3)
    assert amax[1].item() == pytest.approx(quant_range / amax[0].item(),
                                           rel=1e-5)

    # And phase 2 encoded that same activation rather than stale memory.
    decoded = torch.ops.trtllm.dequant_nvfp4_swizzled_sf(
        fp4.view(torch.uint8), sf.view(torch.uint8), amax[1:2], 16)
    similarity = torch.nn.functional.cosine_similarity(
        decoded[live_mask].float(), activated[live_mask].float(), dim=-1)
    assert similarity.min().item() > 0.98


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="MoE runtime quantization requires SM100/SM103.")
def test_fused_swiglu_quantize_replays_in_cuda_graph() -> None:
    """Replay must recompute both the activation and its dynamic scale."""
    # Given a warmed uneven-routing call with fixed graph-safe metadata.
    rows, interm, tile_size = 256, 512, 128
    preact, limits, live_tiles, live_mask = _uneven_routing_preactivation(
        rows, interm, tile_size, (5, ))

    def quantize():
        return torch.ops.trtllm.fp4_swiglu_quantize_fused(
            preact, 16, True, 1, 1536.0, 1e-12, 0, limits, live_tiles,
            tile_size)

    quantize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, _, captured_amax, captured_bf16 = quantize()

    # When replay sees a new preactivation behind the same routing metadata.
    preact[live_mask] = torch.randn_like(preact[live_mask])
    graph.replay()
    torch.cuda.synchronize()

    # Then the captured buffers hold the replay-time SwiGLU and its amax.
    reference = _reference_swiglu(preact)
    torch.testing.assert_close(captured_bf16[live_mask],
                               reference[live_mask],
                               rtol=8e-3,
                               atol=0.0)
    assert captured_amax[0].item() == pytest.approx(
        reference[live_mask].abs().max().float().item(), rel=1e-3)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="Production NVFP4 MoE requires SM100/SM103.")
def test_runtime_4o6_production_moe_replays_in_cuda_graph(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Capture the ordinary production chain, not isolated component ops.

    This exercises routing, fused FC13 BF16+amax, phase-2 4o6 quantization,
    FC2 grouped GEMM, and unpermute through ``run_moe_nvfp4_impl``.
    """
    torch.manual_seed(29)
    monkeypatch.setenv("TRTLLM_NVFP4_RUNTIME_ACTIVATION", "4o6")
    monkeypatch.setenv("TRTLLM_ADAPTIVE_FP4_FC2", "1")
    tokens, hidden, interm = 64, 1024, 1024
    num_experts, top_k, tile_size = 8, 2, 128
    sf_vec_size = 16

    backend = moe.CuteDslFusedMoE.__new__(moe.CuteDslFusedMoE)
    backend.num_slots = num_experts
    backend.activation_type = int(ActivationType.Swiglu)
    backend.use_fused_finalize = False
    backend.fc31_input_scale = torch.ones(1,
                                          dtype=torch.float32,
                                          device="cuda")
    backend.fc2_input_scale = torch.ones(1,
                                         dtype=torch.float32,
                                         device="cuda")

    def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor]:
        global_scale = weight.abs().amax(dim=(1, 2)).float() / (448.0 * 6.0)
        packed, scales = torch.ops.trtllm.fp4_quantize(
            weight, 1.0 / global_scale, sf_vec_size, False)
        return packed.view(torch.float4_e2m1fn_x2), scales, global_scale

    fc13_weight_bf16 = torch.randn((num_experts, 2 * interm, hidden),
                                   dtype=torch.bfloat16,
                                   device="cuda")
    fc13_weight, fc13_sf, fc13_global = quantize_weight(fc13_weight_bf16)
    fc13_weight = interleave_linear_and_gate(
        fc13_weight.view(torch.uint8), group_size=64,
        dim=1).view(torch.float4_e2m1fn_x2)
    fc13_sf = fc13_sf.view(num_experts, 2 * interm,
                          hidden // sf_vec_size)
    fc13_sf = unswizzle_sf(fc13_sf, 2 * interm, hidden).view(
        num_experts, 2 * interm, hidden // sf_vec_size)
    fc13_sf = interleave_linear_and_gate(fc13_sf, group_size=64, dim=1)
    fc13_sf = swizzle_sf(fc13_sf, 2 * interm, hidden).view(
        num_experts, 2 * interm, hidden // sf_vec_size)

    fc2_weight_bf16 = torch.randn((num_experts, hidden, interm),
                                  dtype=torch.bfloat16,
                                  device="cuda")
    fc2_weight, fc2_sf, fc2_global = quantize_weight(fc2_weight_bf16)
    fc2_sf = fc2_sf.view(num_experts, hidden, interm // sf_vec_size)
    weight_view = moe.NvFp4WeightView(
        w3_w1_weight=fc13_weight,
        fc1_weight_scale=fc13_sf,
        fc1_global_scale=fc13_global,
        w2_weight=fc2_weight,
        fc2_weight_scale=fc2_sf,
        fc2_global_scale=fc2_global,
        expert_size_per_partition=num_experts,
        slot_start=0,
    )

    routing_logits = torch.randn((tokens, num_experts), device="cuda")
    token_scales, token_experts = routing_logits.topk(top_k, dim=-1)
    token_scales = token_scales.softmax(dim=-1).float()
    token_experts = token_experts.to(torch.int32)
    token_experts[0] = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    output = torch.empty((tokens, hidden),
                         dtype=torch.bfloat16,
                         device="cuda")

    input_bf16 = torch.randn((tokens, hidden),
                             dtype=torch.bfloat16,
                             device="cuda")
    input_fp4, input_sf, input_amax = moe._runtime_nvfp4_quantize(
        input_bf16, moe._RUNTIME_4O6, swizzled=False)
    backend._runtime_fc13_global_scale = input_amax[1:2]

    def run() -> torch.Tensor:
        return backend.run_moe_nvfp4_impl(
            input_fp4,
            token_experts,
            token_scales,
            input_sf.view(tokens, -1),
            output,
            weight_view,
            tile_size=tile_size,
        )

    # Warm every CuTeDSL tactic/cache used by the captured production path.
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = run()
    initial_output = captured_output.clone()

    next_input = torch.randn_like(input_bf16)
    next_fp4, next_sf, next_amax = moe._runtime_nvfp4_quantize(
        next_input, moe._RUNTIME_4O6, swizzled=False)
    input_fp4.copy_(next_fp4)
    input_sf.copy_(next_sf)
    backend._runtime_fc13_global_scale.copy_(next_amax[1:2])
    graph.replay()
    torch.cuda.synchronize()
    replay_output = captured_output.clone()

    expected_output = run().clone()
    torch.cuda.synchronize()
    assert not torch.equal(replay_output, initial_output)
    torch.testing.assert_close(replay_output, expected_output, rtol=0, atol=0)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="MoE runtime quantization requires SM100/SM103.")
def test_fused_swiglu_quantize_rejects_unsupported_scale_rules() -> None:
    """Only the standard and MSE 4/6 rules are wired for the MoE epilogue."""
    # Given a valid uneven-routing pre-activation.
    rows, interm, tile_size = 128, 512, 128
    preact, limits, live_tiles, _ = _uneven_routing_preactivation(
        rows, interm, tile_size, (7, ))

    # When an unsupported adaptive rule is requested.
    with pytest.raises(RuntimeError, match="standard"):
        torch.ops.trtllm.fp4_swiglu_quantize_fused(preact, 16, True, 2, 1536.0,
                                                   1e-12, 0, limits, live_tiles,
                                                   tile_size)
