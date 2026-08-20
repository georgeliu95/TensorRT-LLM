# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ABI and launch-geometry coverage for the grouped SVDQuant low-rank path.

These cases stub the fused CuteDSL low-rank op so the host-side contract --
argument forwarding, the optional gate, and one dispatch per factor pair --
can be checked without a Blackwell device.  Device numerics live in
``test_bf16_grouped_gemm_sm100.py``.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe import svdquant_helpers as svdh
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import (
    CuteDslFusedMoE,
    _deinterleave_linear_and_gate_cutedsl,
    _supports_grouped_lowrank,
    _svdquant_grouped_dual_lowrank_accumulate_cutedsl,
    _svdquant_grouped_lowrank_accumulate_cutedsl,
    _svdquant_grouped_lowrank_cutedsl,
)
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    SVDQUANT_FC13_PACKED_ORDER,
    SVDQUANT_FC13_PACKED_VH,
    SVDQUANT_FC13_SEPARATED_WEIGHT_LAYOUT,
    NVFP4CuteDslFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
)
from tensorrt_llm._torch.utils import ActivationType

_MODULE = "tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl"

# ``run_moe_nvfp4_impl`` reinterprets the packed FC13 activation before any
# stub can intercept it, so the production-wiring case needs the real dtype.
_HAS_FP4_DTYPE = hasattr(torch, "float4_e2m1fn_x2")


def test_forward_chunk_rejects_svdquant_fc13_with_data_parallelism_before_work(
        ) -> None:
    """DP must not discard the exact BF16 activation SVDQuant FC13 needs."""

    def unexpected_work(*_args, **_kwargs):
        pytest.fail("the unsupported DP path must fail before routing or quantization")

    backend = SimpleNamespace(
        use_dp=True,
        parallel_size=2,
        has_svdquant_fc13=lambda: True,
        routing_method=SimpleNamespace(apply=unexpected_work),
    )

    with pytest.raises(svdh.SvdquantLoadError,
                       match="data parallel.*BF16 activation"):
        CuteDslFusedMoE.forward_chunk(
            backend,
            torch.empty((1, 8), dtype=torch.bfloat16),
            torch.empty((1, 4), dtype=torch.float32),
        )


def test_fc13_deinterleave_uses_dedicated_cutedsl_op(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The SVDQuant hot path must not fall back to TensorIterator copy."""
    source = torch.randn(3, 256, dtype=torch.bfloat16)
    expected = torch.randn_like(source)
    calls: list[tuple[torch.Tensor, int]] = []

    def deinterleave(input_: torch.Tensor, group_size: int) -> torch.Tensor:
        calls.append((input_, group_size))
        return expected

    monkeypatch.setattr(torch.ops.trtllm,
                        "cute_dsl_bf16_deinterleave_blackwell",
                        deinterleave,
                        raising=False)

    actual = _deinterleave_linear_and_gate_cutedsl(source, group_size=64)

    assert actual is expected
    assert calls == [(source, 64)]


def _install_reference_lowrank(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the fused low-rank op with a reference and record every call.

    The stub does the whole two-stage contraction itself, so a helper that
    silently fell back to two single-GEMM dispatches would record two entries
    here and fail the dispatch-count assertions.
    """
    calls: list[dict] = []

    def grouped_lowrank(
        input: torch.Tensor,
        us: torch.Tensor,
        vh: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_size: int,
        num_non_exiting_tiles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        calls.append({
            "input": input,
            "us": us,
            "vh": vh,
            "tile_map": tile_idx_to_group_idx,
            "tile_size": tile_size,
            "gate": num_non_exiting_tiles,
            "output_shape": (input.shape[0], us.shape[1]),
        })
        num_groups = us.shape[0]
        contracted = input[:, :vh.shape[2]]
        out = torch.empty((input.shape[0], us.shape[1]), dtype=input.dtype)
        for tile_idx, expert_idx in enumerate(tile_idx_to_group_idx.tolist()):
            start = tile_idx * tile_size
            end = start + tile_size
            slot = min(max(expert_idx, 0), num_groups - 1)
            # Two stages, one dispatch: down-project to the rank, then back up.
            down = contracted[start:end] @ vh[slot].T
            out[start:end] = down @ us[slot].T
        return out

    def rejected_single_gemm(*_args, **_kwargs):
        pytest.fail(
            "the helper must issue one fused dispatch, not per-stage "
            "single-GEMM calls")

    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_grouped_lowrank_blackwell",
        grouped_lowrank,
        raising=False,
    )
    # Any per-stage call is a regression to the two-dispatch shape.
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_grouped_gemm_blackwell",
        rejected_single_gemm,
        raising=False,
    )
    return calls


def _factors(num_experts: int, out_features: int, rank: int,
             in_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    us = torch.randn(num_experts, out_features, rank, dtype=torch.bfloat16)
    vh = torch.randn(num_experts, rank, in_features, dtype=torch.bfloat16)
    return us, vh


def _reference(x: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
               tile_map: torch.Tensor, tile_size: int) -> torch.Tensor:
    out = torch.empty((x.shape[0], us.shape[1]), dtype=x.dtype)
    for tile_idx, expert_idx in enumerate(tile_map.tolist()):
        start = tile_idx * tile_size
        end = start + tile_size
        out[start:end] = ((x[start:end, :vh.shape[2]] @ vh[expert_idx].T)
                          @ us[expert_idx].T)
    return out


@pytest.mark.parametrize(
    "tile_map",
    [
        # Every expert used exactly once.
        [0, 1, 2, 3],
        # One expert repeated across consecutive tiles, others skipped.
        [2, 2, 2, 2],
        # Mixed repeats with a skipped middle expert.
        [0, 0, 3, 3],
        # Descending order: the map, not the tile index, picks the operand.
        [3, 2, 1, 0],
    ],
)
def test_grouped_lowrank_matches_per_tile_reference(
        monkeypatch: pytest.MonkeyPatch, tile_map: list[int]) -> None:
    # Given a permuted layout whose tiles repeat and skip experts.
    torch.manual_seed(7)
    calls = _install_reference_lowrank(monkeypatch)
    tile_size = 3
    tiles = torch.tensor(tile_map, dtype=torch.int32)
    x = torch.randn(len(tile_map) * tile_size, 16, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=4, out_features=5, rank=2, in_features=16)

    # When the grouped path evaluates the correction.
    actual = _svdquant_grouped_lowrank_cutedsl(x, us, vh, tiles, tile_size)

    # Then one fused dispatch produced the per-tile two-stage answer.
    assert len(calls) == 1
    torch.testing.assert_close(actual, _reference(x, us, vh, tiles, tile_size))


def test_grouped_lowrank_costs_one_dispatch_per_factor_pair(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given two routings that differ in how many experts are touched.
    torch.manual_seed(11)
    calls = _install_reference_lowrank(monkeypatch)
    tile_size = 2
    x = torch.randn(8, 16, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=4, out_features=5, rank=2, in_features=16)

    # When both routings run.
    for tile_map in ([0, 1, 2, 3], [1, 1, 1, 1]):
        _svdquant_grouped_lowrank_cutedsl(
            x, us, vh, torch.tensor(tile_map, dtype=torch.int32), tile_size)

    # Then each factor pair cost exactly one host dispatch -- two would mean the
    # per-stage single-GEMM path came back -- with routing-independent geometry.
    assert len(calls) == 2
    assert [call["output_shape"] for call in calls] == [(8, 5), (8, 5)]


def test_grouped_lowrank_forwards_every_argument_unchanged(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given the device tensors produced by moe_sort.
    torch.manual_seed(13)
    calls = _install_reference_lowrank(monkeypatch)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)

    # When the grouped path runs.
    _svdquant_grouped_lowrank_cutedsl(x, us, vh, tiles, tile_size=2)

    # Then every operand reaches the op as the identical object, so no host
    # copy or synchronization can have taken place, and no gate was invented.
    assert len(calls) == 1
    call = calls[0]
    assert call["input"] is x
    assert call["us"] is us
    assert call["vh"] is vh
    assert call["tile_map"] is tiles
    assert call["tile_size"] == 2
    assert call["gate"] is None


def test_grouped_lowrank_forwards_the_optional_valid_tile_gate(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a device-side count of non-exiting tiles.
    torch.manual_seed(17)
    calls = _install_reference_lowrank(monkeypatch)
    valid = torch.tensor([1], dtype=torch.int32)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)

    # When the caller supplies the gate.
    _svdquant_grouped_lowrank_cutedsl(x,
                                      us,
                                      vh,
                                      tiles,
                                      tile_size=2,
                                      num_non_exiting_tiles=valid)

    # Then the single fused dispatch carries it, so both stages can skip the
    # padded tail.
    assert len(calls) == 1
    assert calls[0]["gate"] is valid


def test_grouped_lowrank_leaves_the_leading_column_slice_to_the_op(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given an activation padded beyond the Vh contraction width.
    torch.manual_seed(19)
    calls = _install_reference_lowrank(monkeypatch)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.randn(4, 12, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)

    # When the grouped path runs.
    actual = _svdquant_grouped_lowrank_cutedsl(x, us, vh, tiles, tile_size=2)

    # Then the full activation is handed over unsliced -- the op takes the
    # leading columns itself -- and only those columns affect the result.
    assert len(calls) == 1
    assert calls[0]["input"] is x
    torch.testing.assert_close(actual, _reference(x, us, vh, tiles, 2))


def test_grouped_lowrank_returns_an_empty_result_for_no_rows(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a permuted layout with no tiles at all.
    calls = _install_reference_lowrank(monkeypatch)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)

    # When the correction is evaluated.
    actual = _svdquant_grouped_lowrank_cutedsl(
        torch.zeros(0, 8, dtype=torch.bfloat16),
        us,
        vh,
        torch.zeros(0, dtype=torch.int32),
        tile_size=2,
    )

    # Then the op is still the one deciding the shape, and it is [0, N].
    assert len(calls) == 1
    assert actual.shape == (0, 3)
    assert actual.dtype == torch.bfloat16


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda kwargs: kwargs.update(us=torch.randn(
            2, 3, 5, dtype=torch.bfloat16)), "rank"),
        (lambda kwargs: kwargs.update(us=torch.randn(
            3, 3, 2, dtype=torch.bfloat16)), "expert count"),
        (lambda kwargs: kwargs.update(tile_size=3), "multiple of tile size"),
        (lambda kwargs: kwargs.update(tile_idx_to_expert_idx=torch.tensor(
            [0], dtype=torch.int32)), "tile map holds"),
        (lambda kwargs: kwargs.update(tile_idx_to_expert_idx=torch.tensor(
            [0, 1], dtype=torch.int64)), "must be int32"),
        (lambda kwargs: kwargs.update(vh=torch.randn(
            2, 2, 32, dtype=torch.bfloat16)), "narrower than"),
        (lambda kwargs: kwargs.update(us=torch.randn(2, 3, 2)),
         "BF16 operands"),
        (lambda kwargs: kwargs.update(vh=torch.randn(
            2, 8, dtype=torch.bfloat16)), "3-D per-expert factors"),
    ],
)
def test_grouped_lowrank_rejects_inexpressible_operands(
        monkeypatch: pytest.MonkeyPatch, mutate, message: str) -> None:
    # Given operands the grouped ABI cannot represent.
    torch.manual_seed(23)
    calls = _install_reference_lowrank(monkeypatch)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)
    kwargs = {
        "x_bf16": torch.randn(4, 8, dtype=torch.bfloat16),
        "us": us,
        "vh": vh,
        "tile_idx_to_expert_idx": torch.tensor([0, 1], dtype=torch.int32),
        "tile_size": 2,
    }
    mutate(kwargs)

    # When the grouped path is asked to run.
    with pytest.raises(svdh.SvdquantLoadError, match=message):
        _svdquant_grouped_lowrank_cutedsl(**kwargs)

    # Then it fails before launching anything.
    assert calls == []


def test_grouped_lowrank_is_gated_on_cuda_and_sm100(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a host tensor on a machine that reports SM 100.
    monkeypatch.setattr(f"{_MODULE}.is_sm_100f", lambda: True)
    host = torch.zeros(2, 4, dtype=torch.bfloat16)

    # Then the CuteDSL kernel is still refused, because it needs device memory.
    assert not _supports_grouped_lowrank(host)

    # And it is refused on non-Blackwell devices too.
    monkeypatch.setattr(f"{_MODULE}.is_sm_100f", lambda: False)
    assert not _supports_grouped_lowrank(host)


def test_lr_permuted_uses_the_grouped_path_when_supported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a backend on a platform that supports the grouped kernel.
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank", lambda _x: True)
    forwarded: dict = {}

    def fake_grouped(x_bf16, us, vh, tile_idx_to_expert_idx, tile_size,
                     num_non_exiting_tiles=None):
        forwarded.update(tile_size=tile_size,
                         tile_map=tile_idx_to_expert_idx,
                         gate=num_non_exiting_tiles)
        return torch.zeros((x_bf16.shape[0], us.shape[1]),
                           dtype=torch.bfloat16)

    monkeypatch.setattr(f"{_MODULE}._svdquant_grouped_lowrank_cutedsl",
                        fake_grouped)
    monkeypatch.setattr(
        f"{_MODULE}.CuteDslFusedMoE._compute_svdquant_lr_permuted_ref",
        staticmethod(lambda *a, **k: pytest.fail(
            "the reference loop must not run on a supported platform")),
    )
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    gate = torch.tensor([1], dtype=torch.int32)

    # When the low-rank correction is evaluated.
    backend._compute_svdquant_lr_permuted(
        torch.zeros(4, 8, dtype=torch.bfloat16),
        torch.zeros(2, 3, 2, dtype=torch.bfloat16),
        torch.zeros(2, 2, 8, dtype=torch.bfloat16),
        tiles,
        torch.tensor([2, 4], dtype=torch.int32),
        tile_size=2,
        slot_start=96,
        num_local_experts=2,
        num_non_exiting_tiles=gate,
    )

    # Then the routing tensors reach the kernel untouched, including the gate
    # that lets it skip the padded tail.
    assert forwarded["tile_size"] == 2
    assert forwarded["tile_map"] is tiles
    assert forwarded["gate"] is gate


def test_lr_permuted_keeps_the_reference_loop_when_unsupported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a platform without the grouped kernel.
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank", lambda _x: False)
    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_lowrank_cutedsl",
        lambda *a, **k: pytest.fail(
            "the grouped kernel must not run on an unsupported platform"),
    )
    x = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.bfloat16)
    us = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.bfloat16)
    vh = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.bfloat16)

    # When the correction is evaluated with a gate the reference cannot use.
    result = backend._compute_svdquant_lr_permuted(
        x,
        us,
        vh,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
        tile_size=1,
        slot_start=96,
        num_local_experts=2,
        num_non_exiting_tiles=torch.tensor([2], dtype=torch.int32),
    )

    # Then the per-tile reference still produces the rank-local answer.
    torch.testing.assert_close(
        result, torch.tensor([[2.0, 4.0], [21.0, 28.0]],
                             dtype=torch.bfloat16))


# ---------------------------------------------------------------------- #
#  Accumulating variant: the correction is added by the op, never by the  #
#  host, and the destination may be a strided column slice                #
# ---------------------------------------------------------------------- #


def _install_reference_lowrank_accumulate(
        monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stub the accumulating op with an in-place reference and record calls.

    Both the functional low-rank op and the per-stage single GEMM are wired to
    fail, so a helper that regressed to "materialize then add" or to two
    dispatches is caught here rather than in a numerics assertion.
    """
    calls: list[dict] = []

    def grouped_lowrank_accumulate(
        input: torch.Tensor,
        us: torch.Tensor,
        vh: torch.Tensor,
        out: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_size: int,
        num_non_exiting_tiles: torch.Tensor | None = None,
    ) -> None:
        calls.append({
            "input": input,
            "us": us,
            "vh": vh,
            "out": out,
            "tile_map": tile_idx_to_group_idx,
            "tile_size": tile_size,
            "gate": num_non_exiting_tiles,
            "out_shape": tuple(out.shape),
            "out_stride": tuple(out.stride()),
            "out_data_ptr": out.data_ptr(),
        })
        num_groups = us.shape[0]
        contracted = input[:, :vh.shape[2]]
        for tile_idx, expert_idx in enumerate(tile_idx_to_group_idx.tolist()):
            start = tile_idx * tile_size
            end = start + tile_size
            slot = min(max(expert_idx, 0), num_groups - 1)
            # Two stages, one dispatch, added onto the destination in place.
            down = contracted[start:end] @ vh[slot].T
            out[start:end] += down @ us[slot].T

    def rejected(*_args, **_kwargs):
        pytest.fail(
            "the accumulating helper must issue one fused in-place dispatch")

    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_grouped_lowrank_accumulate_blackwell",
        grouped_lowrank_accumulate,
        raising=False,
    )
    monkeypatch.setattr(torch.ops.trtllm,
                        "cute_dsl_bf16_grouped_lowrank_blackwell",
                        rejected,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "cute_dsl_bf16_grouped_gemm_blackwell",
                        rejected,
                        raising=False)
    return calls


def test_accumulating_lowrank_adds_onto_the_destination_in_one_dispatch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a destination that already holds a base value.
    torch.manual_seed(47)
    calls = _install_reference_lowrank_accumulate(monkeypatch)
    tile_size = 2
    tiles = torch.tensor([1, 0, 1], dtype=torch.int32)
    x = torch.randn(6, 8, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=2, out_features=5, rank=2, in_features=8)
    base = torch.randn(6, 5, dtype=torch.bfloat16)
    out = base.clone()

    # When the accumulating helper runs.
    _svdquant_grouped_lowrank_accumulate_cutedsl(x, us, vh, out, tiles,
                                                 tile_size)

    # Then one dispatch left base + correction behind, with no temporary of the
    # correction's own shape ever returned to the host.
    assert len(calls) == 1
    torch.testing.assert_close(out,
                               base + _reference(x, us, vh, tiles, tile_size))


def test_accumulating_lowrank_forwards_every_argument_unchanged(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given the device tensors produced by moe_sort plus a destination.
    torch.manual_seed(53)
    calls = _install_reference_lowrank_accumulate(monkeypatch)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    gate = torch.tensor([1], dtype=torch.int32)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)
    out = torch.zeros(4, 3, dtype=torch.bfloat16)

    # When the helper runs with the optional gate.
    _svdquant_grouped_lowrank_accumulate_cutedsl(x,
                                                 us,
                                                 vh,
                                                 out,
                                                 tiles,
                                                 tile_size=2,
                                                 num_non_exiting_tiles=gate)

    # Then every operand reaches the op as the identical object -- no host copy,
    # no synchronization, and the destination is the caller's own tensor.
    assert len(calls) == 1
    call = calls[0]
    assert call["input"] is x
    assert call["us"] is us
    assert call["vh"] is vh
    assert call["out"] is out
    assert call["tile_map"] is tiles
    assert call["tile_size"] == 2
    assert call["gate"] is gate


def test_accumulating_lowrank_omits_the_gate_when_the_caller_has_none(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a caller with no device-side tile count.
    torch.manual_seed(59)
    calls = _install_reference_lowrank_accumulate(monkeypatch)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)

    _svdquant_grouped_lowrank_accumulate_cutedsl(
        torch.randn(4, 8, dtype=torch.bfloat16),
        us,
        vh,
        torch.zeros(4, 3, dtype=torch.bfloat16),
        torch.tensor([0, 1], dtype=torch.int32),
        tile_size=2,
    )

    # Then no gate is invented on its behalf.
    assert len(calls) == 1
    assert calls[0]["gate"] is None


def test_accumulating_lowrank_writes_through_a_strided_half_view(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given the FC13 destination: two halves of one wider pre-activation.
    torch.manual_seed(61)
    calls = _install_reference_lowrank_accumulate(monkeypatch)
    tile_size, half = 2, 3
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    us, vh = _factors(num_experts=2, out_features=half, rank=2, in_features=8)
    base = torch.randn(4, 2 * half, dtype=torch.bfloat16)
    preact = base.clone()

    # When only the gate half is accumulated onto.
    _svdquant_grouped_lowrank_accumulate_cutedsl(x, us, vh, preact[:, half:],
                                                 tiles, tile_size)

    # Then the op received a non-contiguous destination whose row stride is the
    # full width, and only those columns moved.
    assert len(calls) == 1
    assert calls[0]["out_shape"] == (4, half)
    assert calls[0]["out_stride"] == (2 * half, 1)
    assert torch.equal(preact[:, :half], base[:, :half])
    torch.testing.assert_close(
        preact[:, half:],
        base[:, half:] + _reference(x, us, vh, tiles, tile_size))


def test_accumulating_lowrank_accepts_an_empty_destination(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a permuted layout with no tiles at all.
    calls = _install_reference_lowrank_accumulate(monkeypatch)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)
    out = torch.zeros(0, 3, dtype=torch.bfloat16)

    _svdquant_grouped_lowrank_accumulate_cutedsl(
        torch.zeros(0, 8, dtype=torch.bfloat16),
        us,
        vh,
        out,
        torch.zeros(0, dtype=torch.int32),
        tile_size=2,
    )

    # Then the op still owns the decision and nothing about the destination
    # changed.
    assert len(calls) == 1
    assert out.shape == (0, 3)


@pytest.mark.parametrize(
    "destination, message",
    [
        (torch.zeros(4, 4, dtype=torch.bfloat16), "does not match"),
        (torch.zeros(2, 3, dtype=torch.bfloat16), "does not match"),
        (torch.zeros(4, 3, dtype=torch.float32), "must be BF16"),
        (torch.zeros(4, dtype=torch.bfloat16), "must be 2-D"),
        (torch.zeros(3, 4, dtype=torch.bfloat16).T, "contiguous along"),
    ],
)
def test_accumulating_lowrank_rejects_an_inexpressible_destination(
        monkeypatch: pytest.MonkeyPatch, destination: torch.Tensor,
        message: str) -> None:
    # Given a destination the accumulating ABI cannot write.
    torch.manual_seed(67)
    calls = _install_reference_lowrank_accumulate(monkeypatch)
    us, vh = _factors(num_experts=2, out_features=3, rank=2, in_features=8)

    # When the helper is asked to run.
    with pytest.raises(svdh.SvdquantLoadError, match=message):
        _svdquant_grouped_lowrank_accumulate_cutedsl(
            torch.randn(4, 8, dtype=torch.bfloat16),
            us,
            vh,
            destination,
            torch.tensor([0, 1], dtype=torch.int32),
            tile_size=2,
        )

    # Then it failed before launching anything.
    assert calls == []


def test_lr_accumulate_permuted_uses_the_grouped_path_when_supported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a backend on a platform that supports the grouped kernel.
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank", lambda _x: True)
    forwarded: dict = {}

    def fake_accumulate(x_bf16, us, vh, out, tile_idx_to_expert_idx, tile_size,
                        num_non_exiting_tiles=None):
        forwarded.update(tile_size=tile_size,
                         tile_map=tile_idx_to_expert_idx,
                         gate=num_non_exiting_tiles,
                         out=out)

    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_lowrank_accumulate_cutedsl",
        fake_accumulate)
    # Neither the host reference loop nor the functional temporary may run on a
    # platform that has the kernel.
    monkeypatch.setattr(
        f"{_MODULE}.CuteDslFusedMoE._compute_svdquant_lr_permuted_ref",
        staticmethod(lambda *a, **k: pytest.fail(
            "the reference loop must not run on a supported platform")),
    )
    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_lowrank_cutedsl",
        lambda *a, **k: pytest.fail(
            "production must not materialize a low-rank temporary"),
    )
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    gate = torch.tensor([1], dtype=torch.int32)
    out = torch.zeros(4, 3, dtype=torch.bfloat16)

    # When the correction is accumulated.
    result = backend._accumulate_svdquant_lr_permuted(
        torch.zeros(4, 8, dtype=torch.bfloat16),
        torch.zeros(2, 3, 2, dtype=torch.bfloat16),
        torch.zeros(2, 2, 8, dtype=torch.bfloat16),
        out,
        tiles,
        torch.tensor([2, 4], dtype=torch.int32),
        tile_size=2,
        slot_start=96,
        num_local_experts=2,
        num_non_exiting_tiles=gate,
    )

    # Then the routing tensors and the destination reach the kernel untouched,
    # and the method itself returns nothing because it mutates in place.
    assert result is None
    assert forwarded["tile_size"] == 2
    assert forwarded["tile_map"] is tiles
    assert forwarded["gate"] is gate
    assert forwarded["out"] is out


def test_lr_accumulate_permuted_adds_the_reference_loop_when_unsupported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a platform without the grouped kernel.
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank",
                        lambda _x: False)
    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_lowrank_accumulate_cutedsl",
        lambda *a, **k: pytest.fail(
            "the grouped kernel must not run on an unsupported platform"),
    )
    x = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.bfloat16)
    us = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=torch.bfloat16)
    vh = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=torch.bfloat16)
    out = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.bfloat16)

    # When the correction is accumulated with a gate the reference cannot use.
    backend._accumulate_svdquant_lr_permuted(
        x,
        us,
        vh,
        out,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1, 2], dtype=torch.int32),
        tile_size=1,
        slot_start=96,
        num_local_experts=2,
        num_non_exiting_tiles=torch.tensor([2], dtype=torch.int32),
    )

    # Then the per-tile reference answer was added onto the destination in
    # place, preserving CPU behaviour without a device kernel.
    torch.testing.assert_close(
        out, torch.tensor([[3.0, 5.0], [22.0, 29.0]], dtype=torch.bfloat16))


# ---------------------------------------------------------------------- #
#  Packed dual variant: both FC13 corrections share one dispatch, and the #
#  two Vh factors are read out of one contiguous buffer                    #
# ---------------------------------------------------------------------- #


def _install_reference_dual_lowrank_accumulate(
        monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Stub the packed dual op with an in-place reference and record calls.

    Every narrower op -- the single-pair accumulate, the functional low-rank,
    and the bare grouped GEMM -- is wired to fail, so a helper that regressed
    to two dispatches or to "materialize then add" is caught here rather than
    in a numerics assertion.
    """
    calls: list[dict] = []

    def grouped_dual_lowrank_accumulate(
        input: torch.Tensor,
        us_lo: torch.Tensor,
        us_hi: torch.Tensor,
        vh_packed: torch.Tensor,
        out: torch.Tensor,
        tile_idx_to_group_idx: torch.Tensor,
        tile_size: int,
        num_non_exiting_tiles: torch.Tensor | None = None,
    ) -> None:
        calls.append({
            "input": input,
            "us_lo": us_lo,
            "us_hi": us_hi,
            "vh_packed": vh_packed,
            "out": out,
            "tile_map": tile_idx_to_group_idx,
            "tile_size": tile_size,
            "gate": num_non_exiting_tiles,
            "out_shape": tuple(out.shape),
            "out_stride": tuple(out.stride()),
            "out_data_ptr": out.data_ptr(),
        })
        num_groups = us_lo.shape[0]
        rank = us_lo.shape[2]
        out_features = us_lo.shape[1]
        contracted = input[:, :vh_packed.shape[2]]
        for tile_idx, expert_idx in enumerate(tile_idx_to_group_idx.tolist()):
            start = tile_idx * tile_size
            end = start + tile_size
            slot = min(max(expert_idx, 0), num_groups - 1)
            rows = contracted[start:end]
            # The pack's low half drives the low half of the destination.
            low = rows @ vh_packed[slot][:rank].T
            high = rows @ vh_packed[slot][rank:].T
            out[start:end, :out_features] += low @ us_lo[slot].T
            out[start:end, out_features:] += high @ us_hi[slot].T

    def rejected(*_args, **_kwargs):
        pytest.fail("the packed FC13 pair must issue one fused dual dispatch")

    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_grouped_dual_lowrank_accumulate_blackwell",
        grouped_dual_lowrank_accumulate,
        raising=False,
    )
    for name in ("cute_dsl_bf16_grouped_lowrank_accumulate_blackwell",
                 "cute_dsl_bf16_grouped_lowrank_blackwell",
                 "cute_dsl_bf16_grouped_gemm_blackwell"):
        monkeypatch.setattr(torch.ops.trtllm, name, rejected, raising=False)
    return calls


def _dual_factors(num_experts: int, out_features: int, in_features: int,
                  rank: int) -> tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor]:
    """Build a low/high US pair and the packed Vh their halves come from."""
    us_lo = torch.randn(num_experts, out_features, rank, dtype=torch.bfloat16)
    us_hi = torch.randn(num_experts, out_features, rank, dtype=torch.bfloat16)
    vh_packed = torch.randn(num_experts, 2 * rank, in_features,
                            dtype=torch.bfloat16)
    return us_lo, us_hi, vh_packed


def test_dual_lowrank_adds_both_halves_in_one_dispatch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a destination that already holds a base value in both halves.
    torch.manual_seed(53)
    calls = _install_reference_dual_lowrank_accumulate(monkeypatch)
    tile_size = 2
    tiles = torch.tensor([1, 0, 1], dtype=torch.int32)
    x = torch.randn(6, 8, dtype=torch.bfloat16)
    us_lo, us_hi, vh_packed = _dual_factors(2, 4, 8, 2)
    base = torch.randn(6, 8, dtype=torch.bfloat16)
    out = base.clone()

    # When both corrections are accumulated through the packed entry point.
    _svdquant_grouped_dual_lowrank_accumulate_cutedsl(
        x, us_lo, us_hi, vh_packed, out, tiles, tile_size)

    # Then one dispatch covered both pairs, and each half of the destination
    # took the correction built from its own half of the pack.
    assert len(calls) == 1
    expected = base.clone()
    expected[:, :4] += _reference(x, us_lo, vh_packed[:, :2], tiles, tile_size)
    expected[:, 4:] += _reference(x, us_hi, vh_packed[:, 2:], tiles, tile_size)
    torch.testing.assert_close(out, expected)


def test_dual_lowrank_forwards_every_argument_unchanged(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given operands the helper must hand through without copying or slicing.
    torch.manual_seed(59)
    calls = _install_reference_dual_lowrank_accumulate(monkeypatch)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    gate = torch.tensor([2], dtype=torch.int32)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    us_lo, us_hi, vh_packed = _dual_factors(2, 3, 8, 2)
    out = torch.zeros(4, 6, dtype=torch.bfloat16)

    # When the packed accumulate runs with a gate.
    _svdquant_grouped_dual_lowrank_accumulate_cutedsl(
        x, us_lo, us_hi, vh_packed, out, tiles, 2, gate)

    # Then the op saw the caller's own tensors -- in particular the whole pack
    # rather than two halves -- and the caller's tile size and gate.
    call = calls[0]
    assert call["input"] is x
    assert call["us_lo"] is us_lo
    assert call["us_hi"] is us_hi
    assert call["vh_packed"] is vh_packed
    assert call["out"] is out
    assert call["tile_map"] is tiles
    assert call["gate"] is gate
    assert call["tile_size"] == 2


def test_dual_lowrank_omits_the_gate_when_the_caller_has_none(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given no device-side tile count.
    torch.manual_seed(61)
    calls = _install_reference_dual_lowrank_accumulate(monkeypatch)
    us_lo, us_hi, vh_packed = _dual_factors(2, 3, 8, 2)

    # When the packed accumulate runs.
    _svdquant_grouped_dual_lowrank_accumulate_cutedsl(
        torch.randn(4, 8, dtype=torch.bfloat16), us_lo, us_hi, vh_packed,
        torch.zeros(4, 6, dtype=torch.bfloat16),
        torch.tensor([0, 1], dtype=torch.int32), 2)

    # Then the optional argument is left off rather than passed as None, so the
    # op keeps its own default and the kernel skips the gate entirely.
    assert calls[0]["gate"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda kwargs: kwargs.update(us_hi=torch.zeros(
            2, 5, 2, dtype=torch.bfloat16)), "share a shape"),
        (lambda kwargs: kwargs.update(vh_packed=torch.zeros(
            2, 3, 8, dtype=torch.bfloat16)), "two ranks"),
        (lambda kwargs: kwargs.update(vh_packed=torch.zeros(
            2, 4, 16, dtype=torch.bfloat16)[:, :, ::2]), "contiguous"),
        (lambda kwargs: kwargs.update(out=torch.zeros(
            4, 5, dtype=torch.bfloat16)), "paired correction shape"),
        (lambda kwargs: kwargs.update(out=torch.zeros(
            4, 6, dtype=torch.float32)), "BF16"),
    ],
)
def test_dual_lowrank_rejects_inexpressible_operands(
    monkeypatch: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    # Given operands that break one of the packing's own invariants.
    def unreachable(*_args, **_kwargs):
        pytest.fail("validation must reject the operands before dispatch")

    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_bf16_grouped_dual_lowrank_accumulate_blackwell",
        unreachable,
        raising=False)
    kwargs = {
        "x_bf16": torch.zeros(4, 8, dtype=torch.bfloat16),
        "us_lo": torch.zeros(2, 3, 2, dtype=torch.bfloat16),
        "us_hi": torch.zeros(2, 3, 2, dtype=torch.bfloat16),
        "vh_packed": torch.zeros(2, 4, 8, dtype=torch.bfloat16),
        "out": torch.zeros(4, 6, dtype=torch.bfloat16),
        "tile_idx_to_expert_idx": torch.tensor([0, 1], dtype=torch.int32),
        "tile_size": 2,
    }
    mutate(kwargs)

    # When the packed accumulate is asked to run, then it fails on the host.
    with pytest.raises(svdh.SvdquantLoadError, match=message):
        _svdquant_grouped_dual_lowrank_accumulate_cutedsl(**kwargs)


def _fc13_backend(num_experts: int, half: int, hidden: int, rank: int, *,
                  packed: bool) -> CuteDslFusedMoE:
    """Build a bare backend carrying FC13 factors in one storage form or the
    other, mirroring what finalization does or has not yet done."""
    torch.manual_seed(67)
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    for projection in SVDQUANT_FC13_PACKED_ORDER:
        setattr(backend, f"{projection}_us",
                torch.randn(num_experts, half, rank, dtype=torch.bfloat16))
    halves = {projection: torch.randn(num_experts, rank, hidden,
                                      dtype=torch.bfloat16)
              for projection in SVDQUANT_FC13_PACKED_ORDER}
    if packed:
        setattr(backend, SVDQUANT_FC13_PACKED_VH,
                torch.cat([halves[projection]
                           for projection in SVDQUANT_FC13_PACKED_ORDER],
                          dim=1).contiguous())
    else:
        for projection, factor in halves.items():
            setattr(backend, f"{projection}_vh", factor)
    return backend


def test_fc13_accumulate_uses_one_dual_dispatch_when_packed_and_supported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a finalized backend on a platform that has the kernel.
    calls = _install_reference_dual_lowrank_accumulate(monkeypatch)
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank", lambda _x: True)
    monkeypatch.setattr(
        CuteDslFusedMoE, "_accumulate_svdquant_lr_permuted",
        lambda *a, **k: pytest.fail(
            "a packed FC13 pair must not fall back to two single-pair calls"))
    backend = _fc13_backend(2, 4, 8, 2, packed=True)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    gate = torch.tensor([2], dtype=torch.int32)
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    out = torch.zeros(4, 8, dtype=torch.bfloat16)

    # When the FC13 pair is accumulated onto the whole pre-activation.
    result = backend._accumulate_svdquant_fc13_lr_permuted(
        x, (backend.w3_us, None), (backend.w1_us, None), out, tiles,
        torch.tensor([2, 4], dtype=torch.int32), 2, 96, 2, gate)

    # Then exactly one dispatch went out, carrying the linear (w3) factors as
    # the low half, the gate (w1) factors as the high half, the whole pack, and
    # the caller's own destination and routing tensors.
    assert result is None
    assert len(calls) == 1
    call = calls[0]
    assert call["input"] is x
    assert call["us_lo"] is backend.w3_us
    assert call["us_hi"] is backend.w1_us
    assert call["vh_packed"] is getattr(backend, SVDQUANT_FC13_PACKED_VH)
    assert call["out"] is out
    assert call["tile_map"] is tiles
    assert call["gate"] is gate
    assert call["tile_size"] == 2


def test_fc13_accumulate_falls_back_to_two_reference_pairs_when_unsupported(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a packed backend on a platform without the kernel.
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank",
                        lambda _x: False)
    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_dual_lowrank_accumulate_cutedsl",
        lambda *a, **k: pytest.fail(
            "the packed kernel must not run on an unsupported platform"))
    backend = _fc13_backend(2, 2, 4, 2, packed=True)
    packed = getattr(backend, SVDQUANT_FC13_PACKED_VH)
    tiles = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.randn(4, 4, dtype=torch.bfloat16)
    base = torch.randn(4, 4, dtype=torch.bfloat16)
    out = base.clone()

    # When the FC13 pair is accumulated.
    backend._accumulate_svdquant_fc13_lr_permuted(
        x, (backend.w3_us, packed[:, :2]), (backend.w1_us, packed[:, 2:]),
        out, tiles, torch.tensor([2, 4], dtype=torch.int32), 2, 96, 2,
        torch.tensor([2], dtype=torch.int32))

    # Then the two host-side reference accumulations landed on the two halves,
    # which is what keeps this path meaningful in CPU unit tests.
    expected = base.clone()
    expected[:, :2] += _reference(x, backend.w3_us, packed[:, :2], tiles, 2)
    expected[:, 2:] += _reference(x, backend.w1_us, packed[:, 2:], tiles, 2)
    torch.testing.assert_close(out, expected)


def test_fc13_accumulate_stays_on_single_pairs_before_finalization(
        monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a backend whose factors were never packed, on a supported platform.
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank", lambda _x: True)
    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_dual_lowrank_accumulate_cutedsl",
        lambda *a, **k: pytest.fail(
            "the packed kernel needs a packed factor to read"))
    backend = _fc13_backend(2, 3, 8, 2, packed=False)
    single: list[dict] = []

    def record_single(_self, _x, us, vh, out, *_args, **_kwargs):
        single.append({
            "us": us,
            "vh": vh,
            "shape": tuple(out.shape),
            "stride": tuple(out.stride()),
            "data_ptr": out.data_ptr(),
        })

    monkeypatch.setattr(CuteDslFusedMoE, "_accumulate_svdquant_lr_permuted",
                        record_single)
    out = torch.zeros(4, 6, dtype=torch.bfloat16)

    # When the FC13 pair is accumulated.
    backend._accumulate_svdquant_fc13_lr_permuted(
        torch.zeros(4, 8, dtype=torch.bfloat16),
        (backend.w3_us, backend.w3_vh), (backend.w1_us, backend.w1_vh), out,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([2, 4], dtype=torch.int32), 2, 96, 2, None)

    # Then it degrades to the pre-6F wiring: two single-pair accumulations onto
    # the two strided halves, linear first.
    assert len(single) == 2
    linear, gated = single
    assert linear["us"] is backend.w3_us and linear["vh"] is backend.w3_vh
    assert gated["us"] is backend.w1_us and gated["vh"] is backend.w1_vh
    assert linear["shape"] == (4, 3) and gated["shape"] == (4, 3)
    assert linear["stride"] == (6, 1) and gated["stride"] == (6, 1)
    assert (gated["data_ptr"] - linear["data_ptr"] == 3 * out.element_size())


def _run_production_svdquant_moe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    packed: bool,
    adaptive_fc2: bool = False,
    separated_weight_layout: bool = False,
) -> dict:
    """Drive ``run_moe_nvfp4_impl`` once with every device op stubbed.

    Every device op is stubbed, so the cases built on this check integration --
    op order, destination identity, and what each correction is handed --
    rather than any numerics, which live in
    ``test_bf16_grouped_gemm_sm100.py``.

    ``packed`` selects the FC13 Vh storage form: the single packed Parameter
    finalization leaves behind, or the two independent ones that precede it.
    The kernel is reported as available either way, so the choice between one
    dual dispatch and two single-pair ones is made on the storage alone.
    """
    torch.manual_seed(71)
    tile_size, num_tiles, num_experts = 2, 3, 2
    rows = num_tiles * tile_size
    hidden, inter, rank = 64, 128, 8
    events: list[str] = []
    nvtx_ranges: list[str] = []
    accumulated: list[dict] = []
    dual: list[dict] = []

    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    backend.num_slots = num_experts
    backend.activation_type = int(ActivationType.Swiglu)
    # Left on so the assertion that FC2 SVDQuant forces it off is exercised.
    backend.use_fused_finalize = True
    backend.fc31_input_scale = torch.tensor(1.0)
    backend.fc2_input_scale = torch.tensor(1.0)
    setattr(backend, SVDQUANT_FC13_SEPARATED_WEIGHT_LAYOUT,
            separated_weight_layout)
    fc13_vh: dict[str, torch.Tensor] = {}
    for projection in ("w1", "w3"):
        setattr(backend, f"{projection}_us",
                torch.zeros(num_experts, inter, rank, dtype=torch.bfloat16))
        fc13_vh[projection] = torch.zeros(num_experts, rank, hidden,
                                          dtype=torch.bfloat16)
    if packed:
        setattr(
            backend, SVDQUANT_FC13_PACKED_VH,
            torch.cat([fc13_vh[projection]
                       for projection in SVDQUANT_FC13_PACKED_ORDER],
                      dim=1).contiguous())
    else:
        for projection, factor in fc13_vh.items():
            setattr(backend, f"{projection}_vh", factor)
    backend.w2_us = torch.zeros(num_experts, hidden, rank,
                                dtype=torch.bfloat16)
    backend.w2_vh = torch.zeros(num_experts, rank, inter, dtype=torch.bfloat16)

    tile_map = torch.tensor([0, 1, 1], dtype=torch.int32)
    mn_limit = torch.tensor([2, 4, 6], dtype=torch.int32)
    gate = torch.tensor([num_tiles], dtype=torch.int32)
    fc2_output = torch.randn(rows, hidden, dtype=torch.bfloat16)
    fc2_base = fc2_output.clone()
    seen: dict = {}

    def moe_sort(**_kwargs):
        events.append("moe_sort")
        return (tile_map, mn_limit, torch.zeros(rows, dtype=torch.int32),
                torch.zeros(rows, dtype=torch.int32),
                torch.tensor(rows, dtype=torch.int32), gate)

    permuted_fc13_bf16 = torch.randn(rows, hidden, dtype=torch.bfloat16)

    def moe_permute(source, source_sf, *_args, **_kwargs):
        if source_sf is not None:
            pytest.fail("SVDQuant FC13 must gather inside its main GEMM")
        events.append("fc13_bf16_permute")
        seen["bf16_permute_source"] = source
        return permuted_fc13_bf16, None

    def nvfp4_gather_grouped_gemm_identity(**kwargs):
        events.append("fc13_gather_gemm")
        assert kwargs["activation_type"] == int(ActivationType.Identity)
        # Zeros make the accumulated corrections visible by themselves.
        return torch.zeros(rows, 2 * inter, dtype=torch.bfloat16)

    def nvfp4_grouped_gemm(**_kwargs):
        events.append("fc2_gemm")
        return fc2_output

    def swiglu_quantize(preact, *_args, **_kwargs):
        events.append("swiglu_quantize")
        seen["preact"] = preact.clone()
        seen["preact_ptr"] = preact.data_ptr()
        return (torch.zeros(rows, inter // 2, dtype=torch.uint8),
                torch.zeros(rows, inter // 16, dtype=torch.uint8))

    activated_bf16 = torch.randn(rows, inter, dtype=torch.bfloat16)

    def deinterleave(input_: torch.Tensor, group_size: int = 64):
        events.append("deinterleave")
        grouped = input_.view(rows, input_.shape[1] // (2 * group_size), 2,
                              group_size)
        return grouped.transpose(1, 2).contiguous().view_as(input_)

    def swiglu_bf16(preact, *_args, **_kwargs):
        # The fused adaptive epilogue replaced this kernel; a call means the
        # standalone BF16 SwiGLU dispatch came back.
        pytest.fail("FC13 must not launch a standalone BF16 SwiGLU kernel")

    def adaptive_swiglu_quantize(preact, *_args, **_kwargs):
        events.append("adaptive_swiglu_quantize")
        seen["preact"] = preact.clone()
        seen["preact_ptr"] = preact.data_ptr()
        return (torch.zeros(rows, inter // 2, dtype=torch.uint8),
                torch.zeros(rows, inter // 16, dtype=torch.uint8),
                torch.tensor([1.0, 1.0], dtype=torch.float32), activated_bf16)

    def adaptive_quantize(input_, *_args, **_kwargs):
        events.append("adaptive_quantize")
        seen["adaptive_input"] = input_
        return (torch.zeros(rows, inter // 2, dtype=torch.uint8),
                torch.zeros(rows, inter // 16, dtype=torch.uint8),
                torch.tensor([1.0, 1.0], dtype=torch.float32))

    def unpermute(**_kwargs):
        events.append("unpermute")

    def fake_accumulate(_self, x_bf16, us, vh, out, tile_idx_to_expert_idx,
                        tile_idx_to_mn_limit, tile_size_, slot_start,
                        num_local_experts, num_non_exiting_tiles=None):
        events.append("accumulate")
        accumulated.append({
            "input": x_bf16,
            "us": us,
            "vh": vh,
            "out": out,
            "shape": tuple(out.shape),
            "stride": tuple(out.stride()),
            "data_ptr": out.data_ptr(),
            "gate": num_non_exiting_tiles,
            "tile_map": tile_idx_to_expert_idx,
        })
        # Prove the write reaches the caller's buffer through the view.
        out += 1.0

    def fake_dual(x_bf16, us_lo, us_hi, vh_packed, out,
                  tile_idx_to_expert_idx, tile_size_,
                  num_non_exiting_tiles=None):
        events.append("fc13_dual")
        dual.append({
            "input": x_bf16,
            "us_lo": us_lo,
            "us_hi": us_hi,
            "vh_packed": vh_packed,
            "out": out,
            "shape": tuple(out.shape),
            "stride": tuple(out.stride()),
            "data_ptr": out.data_ptr(),
            "gate": num_non_exiting_tiles,
            "tile_map": tile_idx_to_expert_idx,
            "tile_size": tile_size_,
        })
        # Prove the write reaches the caller's own pre-activation.
        out += 1.0

    @contextmanager
    def record_nvtx_range(message: str, **_kwargs):
        nvtx_ranges.append(message)
        yield

    if adaptive_fc2:
        monkeypatch.setenv("TRTLLM_ADAPTIVE_FP4_FC2", "1")
    else:
        monkeypatch.delenv("TRTLLM_ADAPTIVE_FP4_FC2", raising=False)
    monkeypatch.setattr(torch.ops.trtllm, "moe_sort", moe_sort, raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "moe_permute",
                        moe_permute,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "cute_dsl_nvfp4_grouped_gemm_blackwell",
                        nvfp4_grouped_gemm,
                        raising=False)
    monkeypatch.setattr(
        torch.ops.trtllm,
        "cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_bf16_blackwell",
        nvfp4_gather_grouped_gemm_identity,
        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "moe_swiglu_nvfp4_quantize",
                        swiglu_quantize,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "cute_dsl_bf16_deinterleave_blackwell",
                        deinterleave,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "moe_swiglu",
                        swiglu_bf16,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "fp4_swiglu_quantize_fused",
                        adaptive_swiglu_quantize,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "fp4_quantize_fused",
                        adaptive_quantize,
                        raising=False)
    monkeypatch.setattr(torch.ops.trtllm,
                        "moe_unpermute_inplace",
                        unpermute,
                        raising=False)
    def dequantize(*_args, **_kwargs):
        if adaptive_fc2:
            pytest.fail("adaptive FC13 output must not be dequantized for FC2")
        return torch.randn(rows, inter, dtype=torch.bfloat16)

    monkeypatch.setattr(f"{_MODULE}._dequant_nvfp4_cutedsl", dequantize)
    # Reported as available for both storage forms, so only the storage
    # decides whether FC13 fuses into one dispatch.
    monkeypatch.setattr(f"{_MODULE}._supports_grouped_lowrank", lambda _x: True)
    monkeypatch.setattr(
        f"{_MODULE}._svdquant_grouped_dual_lowrank_accumulate_cutedsl",
        fake_dual)
    monkeypatch.setattr(f"{_MODULE}.nvtx_range_debug",
                        record_nvtx_range,
                        raising=False)
    monkeypatch.setattr(CuteDslFusedMoE, "_accumulate_svdquant_lr_permuted",
                        fake_accumulate)
    monkeypatch.setattr(
        CuteDslFusedMoE, "_compute_svdquant_lr_permuted",
        lambda *a, **k: pytest.fail(
            "production must not materialize a low-rank temporary"))

    weight_view = SimpleNamespace(
        expert_size_per_partition=num_experts,
        slot_start=0,
        fc1_global_scale=torch.tensor(1.0),
        fc2_global_scale=torch.tensor(1.0),
        w3_w1_weight=torch.zeros(num_experts, 2 * inter, hidden // 2,
                                 dtype=torch.uint8),
        fc1_weight_scale=torch.zeros(4, dtype=torch.uint8),
        w2_weight=torch.zeros(num_experts, hidden, inter // 2,
                              dtype=torch.uint8),
        fc2_weight_scale=torch.zeros(4, dtype=torch.uint8),
    )
    moe_output = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    source_fc13_bf16 = torch.randn(rows, hidden, dtype=torch.bfloat16)

    backend.run_moe_nvfp4_impl(
        x=torch.zeros(rows, hidden // 2, dtype=torch.uint8),
        token_selected_experts=torch.zeros(rows, 1, dtype=torch.int32),
        token_final_scales=torch.ones(rows, 1),
        x_sf=torch.zeros(rows, hidden // 16, dtype=torch.uint8),
        moe_output=moe_output,
        weight_view=weight_view,
        fc13_input_bf16=source_fc13_bf16,
        tile_size=tile_size,
    )
    return {
        "backend": backend,
        "events": events,
        "nvtx_ranges": nvtx_ranges,
        "accumulated": accumulated,
        "dual": dual,
        "seen": seen,
        "fc2_output": fc2_output,
        "fc2_base": fc2_base,
        "tile_map": tile_map,
        "gate": gate,
        "rows": rows,
        "hidden": hidden,
        "half": inter,
        "activated_bf16": activated_bf16,
        "permuted_fc13_bf16": permuted_fc13_bf16,
        "source_fc13_bf16": source_fc13_bf16,
    }


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
def test_run_moe_nvfp4_impl_marks_svdquant_kernel_stages_with_nvtx(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose every SVDQuant kernel-owning stage in an Nsys timeline."""
    # Given the finalized SVDQuant production path and an NVTX range recorder.
    observed = _run_production_svdquant_moe(monkeypatch, packed=True)

    # Then the ranges follow the same enqueue order as the device operations.
    assert observed["nvtx_ranges"] == [
        "[CUTEDSL][NVFP4] moe_sort",
        "[CUTEDSL][NVFP4] fc13.gather_gemm",
        "[CUTEDSL][NVFP4] fc13.deinterleave",
        "[CUTEDSL][NVFP4] fc13.bf16_permute",
        "[CUTEDSL][NVFP4] fc13.svdq_lowrank",
        "[CUTEDSL][NVFP4] fc13.activation_quantize",
        "[CUTEDSL][NVFP4] fc2.dequantize",
        "[CUTEDSL][NVFP4] fc2.gemm",
        "[CUTEDSL][NVFP4] fc2.svdq_lowrank",
        "[CUTEDSL][NVFP4] unpermute",
    ]


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
def test_run_moe_nvfp4_impl_skips_deinterleave_for_separated_fc13_weights(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Static [w3 | w1] weights make the FC13 GEMM output ready to correct."""
    observed = _run_production_svdquant_moe(
        monkeypatch, packed=True, separated_weight_layout=True)

    assert "deinterleave" not in observed["events"]
    assert "[CUTEDSL][NVFP4] fc13.deinterleave" not in observed["nvtx_ranges"]
    assert observed["events"] == [
        "moe_sort", "fc13_gather_gemm", "fc13_bf16_permute",
        "fc13_dual", "swiglu_quantize", "fc2_gemm", "accumulate",
        "unpermute"
    ]


@pytest.mark.parametrize("fc13_enabled", [False, True])
def test_cutedsl_weight_finalization_keeps_fc13_separated_for_svdquant(
        monkeypatch: pytest.MonkeyPatch, fc13_enabled: bool) -> None:
    """Only SVDQuant FC13 skips the offline CuteDSL gate/up interleave."""
    monkeypatch.setattr(NVFP4CutlassFusedMoEMethod,
                        "process_weights_after_loading", lambda *_args: None)
    weight_calls: list[torch.Tensor] = []
    scale_calls: list[torch.Tensor] = []
    monkeypatch.setattr(
        NVFP4CuteDslFusedMoEMethod,
        "_interleave_w3_w1_weight",
        staticmethod(lambda tensor: weight_calls.append(tensor)),
    )
    monkeypatch.setattr(
        NVFP4CuteDslFusedMoEMethod,
        "_interleave_w3_w1_weight_scale_cute_dsl",
        lambda _self, _module, tensor: scale_calls.append(tensor),
    )
    module = SimpleNamespace(
        _svdquant_config=SimpleNamespace(fc13=fc13_enabled),
        is_gated_activation=True,
        w3_w1_weight=torch.zeros(2, 1),
        w3_w1_weight_scale=torch.zeros(2, 1),
    )

    NVFP4CuteDslFusedMoEMethod().process_weights_after_loading(module)

    assert getattr(module,
                   SVDQUANT_FC13_SEPARATED_WEIGHT_LAYOUT) is fc13_enabled
    expected_calls = 0 if fc13_enabled else 2
    assert len(weight_calls) == expected_calls
    assert len(scale_calls) == expected_calls


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
def test_run_moe_nvfp4_impl_accumulates_in_place_in_production_order(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the pre-finalization wiring: two single-pair corrections in place.

    A module whose FC13 Vh factors were never packed keeps the wiring this
    replaced, so the two halves are still reachable one pair at a time.
    """
    # Given the SVDQuant NVFP4 path over unpacked FC13 factors.
    observed = _run_production_svdquant_moe(monkeypatch, packed=False)
    backend = observed["backend"]
    events, accumulated = observed["events"], observed["accumulated"]
    rows, hidden, half = observed["rows"], observed["hidden"], observed["half"]
    seen, fc2_output = observed["seen"], observed["fc2_output"]

    # Then both FC13 corrections landed before the SwiGLU quantize, and the FC2
    # correction landed after its GEMM and before the unpermute.
    assert not observed["dual"]
    assert events == [
        "moe_sort", "fc13_gather_gemm", "deinterleave",
        "fc13_bf16_permute", "accumulate", "accumulate",
        "swiglu_quantize", "fc2_gemm", "accumulate", "unpermute"
    ]

    # And both FC13 corrections read BF16 rows gathered directly from the
    # activation the NVFP4 operand was quantized from.
    assert seen["bf16_permute_source"] is observed["source_fc13_bf16"]
    assert all(call["input"] is observed["permuted_fc13_bf16"]
               for call in accumulated[:2])

    # And the two FC13 destinations are the two strided halves of one buffer,
    # in linear-then-gate order, taking the w3 and w1 factors respectively.
    linear, gated, fc2 = accumulated
    assert linear["shape"] == (rows, half) and gated["shape"] == (rows, half)
    assert linear["stride"] == (2 * half, 1)
    assert gated["stride"] == (2 * half, 1)
    assert linear["data_ptr"] == seen["preact_ptr"]
    assert (gated["data_ptr"] - linear["data_ptr"] == half *
            linear["out"].element_size())
    assert linear["us"] is backend.w3_us and linear["vh"] is backend.w3_vh
    assert gated["us"] is backend.w1_us and gated["vh"] is backend.w1_vh

    # And the FC2 correction went straight onto the GEMM's own output object.
    assert fc2["out"] is fc2_output
    assert fc2["shape"] == (rows, hidden)
    assert fc2["stride"] == (hidden, 1)
    assert fc2["us"] is backend.w2_us and fc2["vh"] is backend.w2_vh

    # Every correction saw the same routing tensors, gate included.
    assert all(call["tile_map"] is observed["tile_map"]
               for call in accumulated)
    assert all(call["gate"] is observed["gate"] for call in accumulated)

    # And the pre-activation the SwiGLU quantize consumed carries both in-place
    # corrections: the FC13 GEMM returned zeros and each half took one +1.0, so
    # nothing was written to a copy that got discarded.
    assert torch.equal(seen["preact"], torch.ones_like(seen["preact"]))

    # The FC2 correction likewise landed in the GEMM's own buffer, which is the
    # one the unpermute goes on to consume -- no rebinding to a fresh sum.
    assert torch.equal(fc2_output, observed["fc2_base"] + 1.0)


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
def test_run_moe_nvfp4_impl_issues_one_dual_fc13_dispatch_when_packed(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the finalized wiring: one FC13 dispatch, FC2 still a single pair."""
    # Given the SVDQuant NVFP4 path over a packed FC13 Vh.
    observed = _run_production_svdquant_moe(monkeypatch, packed=True)
    backend = observed["backend"]
    events, accumulated = observed["events"], observed["accumulated"]
    rows, hidden, half = observed["rows"], observed["hidden"], observed["half"]
    seen, fc2_output = observed["seen"], observed["fc2_output"]

    # Then FC13 cost one dispatch instead of two, still sitting after the main
    # NVFP4 GEMM and its deinterleave and before the activation quantize.
    assert events == [
        "moe_sort", "fc13_gather_gemm", "deinterleave",
        "fc13_bf16_permute", "fc13_dual", "swiglu_quantize", "fc2_gemm",
        "accumulate", "unpermute"
    ]
    assert len(observed["dual"]) == 1
    fc13 = observed["dual"][0]

    # And it wrote the whole pre-activation in one go -- no [M, half] view, no
    # forward-time cat or copy of the two Vh factors.
    assert fc13["shape"] == (rows, 2 * half)
    assert fc13["stride"] == (2 * half, 1)
    assert fc13["data_ptr"] == seen["preact_ptr"]

    # And it took the linear (w3) factors as the low half, the gate (w1)
    # factors as the high half, and the packed Vh whole.
    assert fc13["us_lo"] is backend.w3_us
    assert fc13["us_hi"] is backend.w1_us
    assert fc13["vh_packed"] is getattr(backend, SVDQUANT_FC13_PACKED_VH)
    assert fc13["tile_map"] is observed["tile_map"]
    assert fc13["gate"] is observed["gate"]

    # And it read the directly gathered BF16 rows, never a dequantized NVFP4
    # reconstruction.
    assert fc13["input"] is observed["permuted_fc13_bf16"]

    # And FC2 is untouched by the packing: one single-pair correction, onto the
    # GEMM's own output object, after the GEMM and before the unpermute.
    assert len(accumulated) == 1
    fc2 = accumulated[0]
    assert fc2["out"] is fc2_output
    assert fc2["shape"] == (rows, hidden)
    assert fc2["stride"] == (hidden, 1)
    assert fc2["us"] is backend.w2_us and fc2["vh"] is backend.w2_vh
    assert fc2["tile_map"] is observed["tile_map"]
    assert fc2["gate"] is observed["gate"]

    # And the in-place writes reached the buffers the downstream stages read:
    # the FC13 GEMM returned zeros and the dual correction added one.
    assert torch.equal(seen["preact"], torch.ones_like(seen["preact"]))
    assert torch.equal(fc2_output, observed["fc2_base"] + 1.0)


@pytest.mark.skipif(not _HAS_FP4_DTYPE,
                    reason="run_moe_nvfp4_impl needs torch.float4_e2m1fn_x2.")
def test_run_moe_nvfp4_impl_reuses_adaptive_fc13_bf16_for_svdquant_fc2(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Adaptive FC13 must not encode, decode, then re-encode FC2 input."""
    # Given packed SVDQuant factors with adaptive FC2-input quantization.
    observed = _run_production_svdquant_moe(monkeypatch,
                                            packed=True,
                                            adaptive_fc2=True)

    # Then one fused dispatch activates and quantizes: no standalone SwiGLU
    # kernel (its stub fails the test) and no separate adaptive quantize.
    assert observed["events"] == [
        "moe_sort", "fc13_gather_gemm", "deinterleave",
        "fc13_bf16_permute", "fc13_dual", "adaptive_swiglu_quantize",
        "fc2_gemm", "accumulate", "unpermute"
    ]

    # And the FC2 low-rank correction reuses that exact BF16 activation; the
    # dequantization stub fails the test if the old round trip is reintroduced.
    assert observed["accumulated"][0]["input"] is observed["activated_bf16"]
