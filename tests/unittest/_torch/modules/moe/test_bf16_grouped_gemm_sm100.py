# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device numerics for the SM100 BF16 contiguous grouped GEMM.

The kernel backs the SVDQuant low-rank correction over the ``moe_sort``
padded permuted layout: the tile-to-expert map supplies the batch coordinate
of the weight operand, and the grid follows from the padded row count alone.
"""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import (
    _svdquant_grouped_lowrank_accumulate_cutedsl,
    _svdquant_grouped_lowrank_cutedsl,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="CuteDSL BF16 grouped GEMM requires SM 100 / SM 103.",
)

# The routing tile size the SVDQuant path pins in run_moe_nvfp4.
TILE_SIZE = 128
# Padding written by moe_sort into the trailing tile-to-expert entries is
# uninitialized; GroupedGemmInputsHelper uses this sentinel to expose misuse.
PAD_SENTINEL = int(2e9)


@pytest.mark.parametrize("rows, width, group_size", [(7, 256, 64),
                                                       (128, 4096, 64)])
def test_bf16_deinterleave_matches_reference(rows: int, width: int,
                                             group_size: int) -> None:
    """The vectorized copy preserves the FC13 up/gate block permutation."""
    source = torch.randn(rows,
                         width,
                         dtype=torch.bfloat16,
                         device="cuda")
    grouped = source.view(rows, width // (2 * group_size), 2, group_size)
    expected = grouped.transpose(1, 2).contiguous().view_as(source)

    actual = torch.ops.trtllm.cute_dsl_bf16_deinterleave_blackwell(
        source, group_size)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.is_contiguous()


def _reference(a: torch.Tensor, b: torch.Tensor, tile_map: torch.Tensor,
               num_valid: int) -> torch.Tensor:
    out = torch.zeros((a.shape[0], b.shape[1]),
                      dtype=a.dtype,
                      device=a.device)
    for tile_idx in range(num_valid):
        expert_idx = int(tile_map[tile_idx])
        start = tile_idx * TILE_SIZE
        end = start + TILE_SIZE
        out[start:end] = a[start:end] @ b[expert_idx].T
    return out


def _run(a: torch.Tensor, b: torch.Tensor, tile_map: torch.Tensor,
         num_valid: torch.Tensor | None) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]),
                      dtype=a.dtype,
                      device=a.device)
    if num_valid is None:
        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_blackwell(
            a, b, out, tile_map, TILE_SIZE)
    else:
        torch.ops.trtllm.cute_dsl_bf16_grouped_gemm_blackwell(
            a, b, out, tile_map, TILE_SIZE, num_non_exiting_tiles=num_valid)
    return out


def _operands(num_tiles: int, k: int, n: int,
              num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn(num_tiles * TILE_SIZE,
                    k,
                    dtype=torch.bfloat16,
                    device="cuda")
    b = torch.randn(num_experts, n, k, dtype=torch.bfloat16, device="cuda")
    return a, b


@pytest.mark.parametrize("n, k", [(64, 7168), (2048, 64), (64, 2048),
                                  (7168, 64)])
def test_grouped_gemm_matches_reference_for_rank64_stage_shapes(
        n: int, k: int) -> None:
    # Given both rank-64 stage shapes the SVDQuant correction needs.
    torch.manual_seed(3)
    num_tiles, num_experts = 6, 4
    a, b = _operands(num_tiles, k, n, num_experts)
    tile_map = torch.tensor([0, 1, 1, 2, 3, 3],
                            dtype=torch.int32,
                            device="cuda")

    # When the grouped GEMM runs over every tile.
    actual = _run(a, b, tile_map, None)

    # Then it matches the per-tile reference.
    torch.testing.assert_close(actual,
                               _reference(a, b, tile_map, num_tiles),
                               atol=2e-2,
                               rtol=2e-2)


@pytest.mark.parametrize("tile_map, num_experts", [
    ([0, 1, 2, 3, 4, 5, 6, 7], 8),
    ([3, 3, 3, 3, 3, 3, 3, 3], 8),
    ([0, 0, 7, 7, 7, 1, 1, 1], 8),
    ([0], 1),
])
def test_grouped_gemm_handles_repeated_and_skipped_experts(
        tile_map: list[int], num_experts: int) -> None:
    # Given routings that repeat some experts and skip others entirely.
    torch.manual_seed(5)
    a, b = _operands(len(tile_map), 512, 64, num_experts)
    tiles = torch.tensor(tile_map, dtype=torch.int32, device="cuda")

    # When the grouped GEMM runs.
    actual = _run(a, b, tiles, None)

    # Then every tile uses the operand its map entry names.
    torch.testing.assert_close(actual,
                               _reference(a, b, tiles, len(tile_map)),
                               atol=2e-2,
                               rtol=2e-2)


@pytest.mark.parametrize("num_valid", [0, 1, 3])
def test_grouped_gemm_skips_padded_tiles_with_uninitialized_map_entries(
        num_valid: int) -> None:
    # Given a moe_sort map whose trailing entries were never written.
    torch.manual_seed(7)
    num_tiles, num_experts = 6, 4
    a, b = _operands(num_tiles, 512, 64, num_experts)
    tile_map = torch.full((num_tiles, ),
                          PAD_SENTINEL,
                          dtype=torch.int32,
                          device="cuda")
    tile_map[:num_valid] = torch.arange(num_valid,
                                        dtype=torch.int32,
                                        device="cuda") % num_experts
    valid = torch.tensor([num_valid], dtype=torch.int32, device="cuda")

    # When the kernel is told how many tiles are live.
    actual = _run(a, b, tile_map, valid)

    # Then the live rows are correct and the out-of-range entries never index
    # the weight operand (assert under compute-sanitizer for the OOB claim).
    expected = _reference(a, b, tile_map, num_valid)
    torch.testing.assert_close(actual[:num_valid * TILE_SIZE],
                               expected[:num_valid * TILE_SIZE],
                               atol=2e-2,
                               rtol=2e-2)


def test_grouped_gemm_clamps_out_of_range_map_entries_without_the_gate() -> None:
    # Given a map with an out-of-range entry and no valid-tile gate, which is
    # how a caller that cannot supply num_non_exiting_tiles behaves.
    torch.manual_seed(11)
    num_tiles, num_experts = 3, 2
    a, b = _operands(num_tiles, 512, 64, num_experts)
    tile_map = torch.tensor([0, 1, PAD_SENTINEL],
                            dtype=torch.int32,
                            device="cuda")

    # When the grouped GEMM runs.
    actual = _run(a, b, tile_map, None)

    # Then the in-range tiles are correct; the clamped tile must not fault.
    reference = _reference(a, b, tile_map, 2)
    torch.testing.assert_close(actual[:2 * TILE_SIZE],
                               reference[:2 * TILE_SIZE],
                               atol=2e-2,
                               rtol=2e-2)
    assert torch.isfinite(actual[2 * TILE_SIZE:]).all()


def test_grouped_lowrank_is_capturable_and_routing_agnostic() -> None:
    # Given the two-stage low-rank path with a fixed padded row count.
    torch.manual_seed(13)
    num_tiles, num_experts, rank = 4, 4, 64
    hidden, inter = 512, 256
    x = torch.randn(num_tiles * TILE_SIZE,
                    hidden,
                    dtype=torch.bfloat16,
                    device="cuda")
    us = torch.randn(num_experts, inter, rank, dtype=torch.bfloat16,
                     device="cuda") / rank
    vh = torch.randn(num_experts, rank, hidden, dtype=torch.bfloat16,
                     device="cuda") / hidden
    tiles = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    num_valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")

    def eager(tile_map: list[int], valid_tiles: int) -> torch.Tensor:
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        num_valid.fill_(valid_tiles)
        return _svdquant_grouped_lowrank_cutedsl(
            x, us, vh, tiles, TILE_SIZE, num_non_exiting_tiles=num_valid)

    # Warm up so CuteDSL JIT compilation happens outside the capture.
    expected_first = eager([0, 1, 2, 3], num_tiles)
    torch.cuda.synchronize()

    # When the path is captured once and replayed under different routings.
    graph = torch.cuda.CUDAGraph()
    tiles.copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"))
    with torch.cuda.graph(graph):
        captured = _svdquant_grouped_lowrank_cutedsl(x, us, vh, tiles,
                                                     TILE_SIZE, num_valid)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected_first, atol=2e-2, rtol=2e-2)

    for tile_map, valid_tiles in (([3, 2, 1, 0], 4),
                                  ([1, 1, PAD_SENTINEL, PAD_SENTINEL], 2)):
        expected = eager(tile_map, valid_tiles)
        torch.cuda.synchronize()
        tiles.copy_(
            torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        num_valid.fill_(valid_tiles)
        graph.replay()
        torch.cuda.synchronize()
        # Then the replay follows the new routing without a recapture.
        live_rows = valid_tiles * TILE_SIZE
        torch.testing.assert_close(captured[:live_rows],
                                   expected[:live_rows],
                                   atol=2e-2,
                                   rtol=2e-2)

    # A zero valid-tile count is a real decode/DP boundary.  It must replay
    # without touching the sentinel map or deadlocking any pipeline role.
    tiles.fill_(PAD_SENTINEL)
    num_valid.zero_()
    graph.replay()
    torch.cuda.synchronize()


def test_grouped_gemm_accepts_zero_rows() -> None:
    a = torch.empty((0, 512), dtype=torch.bfloat16, device="cuda")
    b = torch.randn(2, 64, 512, dtype=torch.bfloat16, device="cuda")
    tile_map = torch.empty((0, ), dtype=torch.int32, device="cuda")
    valid = torch.zeros((1, ), dtype=torch.int32, device="cuda")

    actual = _run(a, b, tile_map, valid)

    assert actual.shape == (0, 64)


# ---------------------------------------------------------------------- #
#  Fused two-stage low-rank op: one dispatch, still two kernels           #
# ---------------------------------------------------------------------- #


def _lowrank_operands(
    num_tiles: int,
    hidden: int,
    inter: int,
    rank: int,
    num_experts: int,
    activation_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one SVDQuant factor pair plus its permuted activation."""
    width = hidden if activation_width is None else activation_width
    x = torch.randn(num_tiles * TILE_SIZE,
                    width,
                    dtype=torch.bfloat16,
                    device="cuda")
    # Scale the factors down so the rank-64 contraction stays in bf16 range.
    us = torch.randn(
        num_experts, inter, rank, dtype=torch.bfloat16, device="cuda") / rank
    vh = torch.randn(
        num_experts, rank, hidden, dtype=torch.bfloat16, device="cuda") / hidden
    return x, us, vh


def _lowrank_reference(x: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
                       tile_map: torch.Tensor, num_valid: int) -> torch.Tensor:
    out = torch.zeros((x.shape[0], us.shape[1]),
                      dtype=x.dtype,
                      device=x.device)
    for tile_idx in range(num_valid):
        expert_idx = int(tile_map[tile_idx])
        start = tile_idx * TILE_SIZE
        end = start + TILE_SIZE
        down = x[start:end, :vh.shape[2]] @ vh[expert_idx].T
        out[start:end] = down @ us[expert_idx].T
    return out


@pytest.mark.parametrize("gated", [False, True])
def test_fused_lowrank_matches_the_two_stage_reference(gated: bool) -> None:
    # Given a factor pair over a permuted layout with repeats and skips.
    torch.manual_seed(29)
    num_tiles, num_experts = 6, 4
    x, us, vh = _lowrank_operands(num_tiles, 512, 256, 64, num_experts)
    tile_map = torch.tensor([0, 1, 1, 3, 3, 3],
                            dtype=torch.int32,
                            device="cuda")
    valid = (torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
             if gated else None)

    # When the helper evaluates it through the fused op.
    actual = _svdquant_grouped_lowrank_cutedsl(x,
                                               us,
                                               vh,
                                               tile_map,
                                               TILE_SIZE,
                                               num_non_exiting_tiles=valid)

    # Then it agrees with the per-tile two-stage formula, gated or not.
    assert actual.shape == (num_tiles * TILE_SIZE, us.shape[1])
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual,
                               _lowrank_reference(x, us, vh, tile_map,
                                                  num_tiles),
                               atol=2e-2,
                               rtol=2e-2)


def test_fused_lowrank_equals_two_isolated_single_gemms() -> None:
    # Given the same operands driven both ways.
    torch.manual_seed(31)
    num_tiles, num_experts, rank = 4, 4, 64
    x, us, vh = _lowrank_operands(num_tiles, 512, 256, rank, num_experts)
    tile_map = torch.tensor([2, 0, 0, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")

    # When the fused op runs, and when the two single-stage ops run in sequence.
    fused = _svdquant_grouped_lowrank_cutedsl(x,
                                              us,
                                              vh,
                                              tile_map,
                                              TILE_SIZE,
                                              num_non_exiting_tiles=valid)
    staged = _run(_run(x, vh, tile_map, valid), us, tile_map, valid)

    # Then fusing the dispatch changed nothing about the arithmetic.
    torch.testing.assert_close(fused, staged, atol=2e-2, rtol=2e-2)


def test_fused_lowrank_contracts_only_the_leading_columns() -> None:
    # Given an activation wider than the Vh contraction, as the FC13 path can
    # produce after padding.
    torch.manual_seed(37)
    num_tiles, num_experts, hidden = 3, 2, 512
    x, us, vh = _lowrank_operands(num_tiles,
                                  hidden,
                                  128,
                                  64,
                                  num_experts,
                                  activation_width=hidden + 128)
    tile_map = torch.tensor([0, 1, 1], dtype=torch.int32, device="cuda")

    # When the fused op runs.
    actual = _svdquant_grouped_lowrank_cutedsl(x, us, vh, tile_map, TILE_SIZE)

    # Then the trailing columns are ignored: overwriting them changes nothing.
    reference = _lowrank_reference(x, us, vh, tile_map, num_tiles)
    torch.testing.assert_close(actual, reference, atol=2e-2, rtol=2e-2)
    x[:, hidden:].fill_(float("nan"))
    again = _svdquant_grouped_lowrank_cutedsl(x, us, vh, tile_map, TILE_SIZE)
    torch.testing.assert_close(again, reference, atol=2e-2, rtol=2e-2)


def test_fused_lowrank_accepts_zero_rows() -> None:
    # Given an empty permuted layout, which decode/DP boundaries can produce.
    torch.manual_seed(41)
    num_experts, rank, inter, hidden = 2, 64, 256, 512
    x = torch.empty((0, hidden), dtype=torch.bfloat16, device="cuda")
    us = torch.randn(
        num_experts, inter, rank, dtype=torch.bfloat16, device="cuda") / rank
    vh = torch.randn(
        num_experts, rank, hidden, dtype=torch.bfloat16, device="cuda") / hidden
    tile_map = torch.empty((0, ), dtype=torch.int32, device="cuda")
    valid = torch.zeros((1, ), dtype=torch.int32, device="cuda")

    # When the fused op runs, gated and ungated.
    for gate in (None, valid):
        actual = _svdquant_grouped_lowrank_cutedsl(x,
                                                   us,
                                                   vh,
                                                   tile_map,
                                                   TILE_SIZE,
                                                   num_non_exiting_tiles=gate)

        # Then it returns the correctly shaped empty result rather than failing.
        assert actual.shape == (0, inter)
        assert actual.dtype == torch.bfloat16
        assert actual.device == x.device


def test_fused_lowrank_is_capturable_with_the_gate() -> None:
    # Given the fused op inside a CUDA graph.
    torch.manual_seed(43)
    num_tiles, num_experts, rank = 4, 4, 64
    inter, hidden = 256, 512
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, rank, num_experts)
    tiles = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")

    def eager(tile_map: list[int], valid_tiles: int) -> torch.Tensor:
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        valid.fill_(valid_tiles)
        return torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_blackwell(
            x, us, vh, tiles, TILE_SIZE, num_non_exiting_tiles=valid)

    # Warm up so the CuteDSL JIT for both stages runs outside the capture.
    expected = eager([0, 1, 2, 3], num_tiles)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    tiles.copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"))
    valid.fill_(num_tiles)
    with torch.cuda.graph(graph):
        captured = torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_blackwell(
            x, us, vh, tiles, TILE_SIZE, num_non_exiting_tiles=valid)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=2e-2, rtol=2e-2)

    # When the routing changes under a replay, including a shrinking gate.
    for tile_map, valid_tiles in (([3, 2, 1, 0], 4),
                                  ([2, 2, PAD_SENTINEL, PAD_SENTINEL], 2)):
        want = eager(tile_map, valid_tiles)
        torch.cuda.synchronize()
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        valid.fill_(valid_tiles)
        graph.replay()
        torch.cuda.synchronize()
        # Then the replay follows the new routing with no recapture.
        live_rows = valid_tiles * TILE_SIZE
        torch.testing.assert_close(captured[:live_rows],
                                   want[:live_rows],
                                   atol=2e-2,
                                   rtol=2e-2)

    # And a zero valid-tile count must replay without deadlocking either stage.
    tiles.fill_(PAD_SENTINEL)
    valid.zero_()
    graph.replay()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------- #
#  Accumulating low-rank op: the correction lands in the destination's    #
#  own epilogue, so no [M, N] temporary and no elementwise add exist      #
# ---------------------------------------------------------------------- #


def _accumulate(x: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
                out: torch.Tensor, tile_map: torch.Tensor,
                num_valid: torch.Tensor | None) -> None:
    if num_valid is None:
        torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_accumulate_blackwell(
            x, us, vh, out, tile_map, TILE_SIZE)
    else:
        torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_accumulate_blackwell(
            x, us, vh, out, tile_map, TILE_SIZE,
            num_non_exiting_tiles=num_valid)


@pytest.mark.parametrize("gated", [False, True])
def test_accumulating_lowrank_adds_onto_a_contiguous_destination(
        gated: bool) -> None:
    # Given the FC2 shape: a plain contiguous [M, hidden] GEMM output.
    torch.manual_seed(101)
    num_tiles, num_experts, hidden = 4, 4, 512
    x, us, vh = _lowrank_operands(num_tiles, hidden, hidden, 64, num_experts)
    tile_map = torch.tensor([0, 2, 2, 1], dtype=torch.int32, device="cuda")
    valid = (torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
             if gated else None)
    base = torch.randn(num_tiles * TILE_SIZE,
                       hidden,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = base.clone()

    # When the correction accumulates onto it.
    _accumulate(x, us, vh, out, tile_map, valid)

    # Then the destination holds base + correction, gated or not.
    expected = base + _lowrank_reference(x, us, vh, tile_map, num_tiles)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


def test_accumulating_lowrank_adds_onto_the_two_strided_half_views() -> None:
    # Given the FC13 shape: one [M, 2 * inter] pre-activation whose two halves
    # take different factor pairs, exactly as run_moe_nvfp4_impl drives it.
    torch.manual_seed(103)
    num_tiles, num_experts, hidden, half = 3, 2, 512, 256
    x, w3_us, w3_vh = _lowrank_operands(num_tiles, hidden, half, 64,
                                        num_experts)
    _, w1_us, w1_vh = _lowrank_operands(num_tiles, hidden, half, 64,
                                        num_experts)
    tile_map = torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       2 * half,
                       dtype=torch.bfloat16,
                       device="cuda")
    preact = base.clone()

    # When only the linear half is accumulated onto.
    _accumulate(x, w3_us, w3_vh, preact[:, :half], tile_map, valid)

    # Then the gate half is bit-for-bit untouched: a strided destination writes
    # only its own columns.
    assert torch.equal(preact[:, half:], base[:, half:])

    # When the gate half takes its own factor pair.
    _accumulate(x, w1_us, w1_vh, preact[:, half:], tile_map, valid)

    # Then each half carries its own correction.
    torch.testing.assert_close(
        preact[:, :half],
        base[:, :half] + _lowrank_reference(x, w3_us, w3_vh, tile_map,
                                            num_tiles),
        atol=2e-2,
        rtol=2e-2)
    torch.testing.assert_close(
        preact[:, half:],
        base[:, half:] + _lowrank_reference(x, w1_us, w1_vh, tile_map,
                                            num_tiles),
        atol=2e-2,
        rtol=2e-2)


@pytest.mark.parametrize("tile_map_values", [
    [0, 1, 2, 3],
    [2, 2, 2, 2],
    [0, 0, 3, 3],
    [3, 2, 1, 0],
])
def test_accumulating_lowrank_handles_repeated_and_skipped_experts(
        tile_map_values: list[int]) -> None:
    # Given a routing whose tiles repeat experts and skip others.
    torch.manual_seed(107)
    num_tiles, num_experts, hidden, inter = 4, 4, 512, 256
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, 64, num_experts)
    tile_map = torch.tensor(tile_map_values, dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       inter,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = base.clone()

    # When the correction accumulates.
    _accumulate(x, us, vh, out, tile_map, valid)

    # Then the map, not the tile index, picked every weight slice.
    torch.testing.assert_close(out,
                               base +
                               _lowrank_reference(x, us, vh, tile_map,
                                                  num_tiles),
                               atol=2e-2,
                               rtol=2e-2)


@pytest.mark.parametrize("num_valid", [0, 1, 3])
def test_accumulating_lowrank_leaves_gated_off_tiles_bit_identical(
        num_valid: int) -> None:
    # Given a gate that retires the trailing tiles, whose map entries moe_sort
    # leaves uninitialized.
    torch.manual_seed(109)
    num_tiles, num_experts, hidden, inter = 4, 2, 512, 256
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, 64, num_experts)
    tile_map = torch.full((num_tiles, ),
                          PAD_SENTINEL,
                          dtype=torch.int32,
                          device="cuda")
    tile_map[:num_valid] = torch.arange(num_valid,
                                        dtype=torch.int32,
                                        device="cuda") % num_experts
    valid = torch.tensor([num_valid], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       inter,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = base.clone()

    # When the correction accumulates.
    _accumulate(x, us, vh, out, tile_map, valid)

    # Then the retired rows are untouched -- not "base plus zero", but the same
    # bits -- so a sentinel map entry can never fold garbage into them.
    live = num_valid * TILE_SIZE
    assert torch.equal(out[live:], base[live:])
    if live:
        torch.testing.assert_close(
            out[:live],
            (base + _lowrank_reference(x, us, vh, tile_map, num_valid))[:live],
            atol=2e-2,
            rtol=2e-2)


def test_accumulating_lowrank_accepts_zero_rows() -> None:
    # Given an empty permuted layout, which decode/DP boundaries produce.
    torch.manual_seed(113)
    num_experts, rank, inter, hidden = 2, 64, 256, 512
    x = torch.empty((0, hidden), dtype=torch.bfloat16, device="cuda")
    us = torch.randn(
        num_experts, inter, rank, dtype=torch.bfloat16, device="cuda") / rank
    vh = torch.randn(
        num_experts, rank, hidden, dtype=torch.bfloat16, device="cuda") / hidden
    tile_map = torch.empty((0, ), dtype=torch.int32, device="cuda")
    valid = torch.zeros((1, ), dtype=torch.int32, device="cuda")

    # When it runs, gated and ungated, against an empty destination.
    for gate in (None, valid):
        out = torch.empty((0, inter), dtype=torch.bfloat16, device="cuda")
        _accumulate(x, us, vh, out, tile_map, gate)

        # Then nothing is launched and the destination keeps its shape.
        assert out.shape == (0, inter)


def test_accumulating_lowrank_equals_the_functional_op_plus_an_add() -> None:
    # Given the same operands driven both ways.
    torch.manual_seed(127)
    num_tiles, num_experts, hidden, inter = 4, 4, 512, 256
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, 64, num_experts)
    tile_map = torch.tensor([2, 0, 0, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       inter,
                       dtype=torch.bfloat16,
                       device="cuda")

    # When the correction is folded into the epilogue, and when it is computed
    # out of place and added afterwards.
    fused = base.clone()
    _accumulate(x, us, vh, fused, tile_map, valid)
    staged = base + _svdquant_grouped_lowrank_cutedsl(
        x, us, vh, tile_map, TILE_SIZE, num_non_exiting_tiles=valid)

    # Then folding the add into the store changed nothing about the arithmetic:
    # both round the fp32 accumulator once and then add in bf16.
    torch.testing.assert_close(fused, staged, atol=2e-2, rtol=2e-2)


def test_accumulating_lowrank_is_deterministic_across_launches() -> None:
    # Given one CTA owning each output element, the reduce-add store is a fused
    # read-modify-write rather than a racy accumulation.
    torch.manual_seed(131)
    num_tiles, num_experts, hidden, inter = 4, 4, 512, 256
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, 64, num_experts)
    tile_map = torch.tensor([1, 1, 3, 0], dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       inter,
                       dtype=torch.bfloat16,
                       device="cuda")

    first = base.clone()
    _accumulate(x, us, vh, first, tile_map, valid)

    # When the identical launch is repeated.
    for _ in range(8):
        again = base.clone()
        _accumulate(x, us, vh, again, tile_map, valid)

        # Then the result is bit-identical, not merely close.
        assert torch.equal(again, first)


def test_accumulating_lowrank_is_capturable_with_the_gate() -> None:
    # Given the base value written inside the graph, as the FC13/FC2 GEMMs do
    # in production, followed by the accumulating correction.
    torch.manual_seed(137)
    num_tiles, num_experts, rank = 4, 4, 64
    inter, hidden = 256, 512
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, rank, num_experts)
    tiles = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       inter,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = torch.empty_like(base)

    def eager(tile_map: list[int], valid_tiles: int) -> torch.Tensor:
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        valid.fill_(valid_tiles)
        want = base.clone()
        _accumulate(x, us, vh, want, tiles, valid)
        return want

    # Warm up so the CuteDSL JIT for both the plain and the accumulating
    # variant runs outside the capture -- they are separate cache entries.
    expected = eager([0, 1, 2, 3], num_tiles)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    tiles.copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"))
    valid.fill_(num_tiles)
    with torch.cuda.graph(graph):
        out.copy_(base)
        _accumulate(x, us, vh, out, tiles, valid)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

    # When the routing changes under a replay, including a shrinking gate.
    for tile_map, valid_tiles in (([3, 2, 1, 0], 4),
                                  ([2, 2, PAD_SENTINEL, PAD_SENTINEL], 2)):
        want = eager(tile_map, valid_tiles)
        torch.cuda.synchronize()
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        valid.fill_(valid_tiles)
        graph.replay()
        torch.cuda.synchronize()
        # Then the replay follows the new routing with no recapture, and the
        # retired rows are the freshly copied base rather than an accumulation
        # of every previous replay.
        live_rows = valid_tiles * TILE_SIZE
        torch.testing.assert_close(out[:live_rows],
                                   want[:live_rows],
                                   atol=2e-2,
                                   rtol=2e-2)
        assert torch.equal(out[live_rows:], base[live_rows:])

    # And a zero valid-tile count must replay without deadlocking either stage,
    # leaving the destination exactly as the in-graph copy left it.
    tiles.fill_(PAD_SENTINEL)
    valid.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, base)


def test_accumulating_custom_op_rejects_a_mismatched_destination() -> None:
    # Given a destination that does not match the correction it must receive.
    torch.manual_seed(139)
    num_tiles, num_experts, hidden, inter = 2, 2, 512, 256
    x, us, vh = _lowrank_operands(num_tiles, hidden, inter, 64, num_experts)
    tile_map = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    rows = num_tiles * TILE_SIZE

    # Then the CUDA custom-op boundary refuses before anything is launched.
    # The production wrapper intentionally avoids duplicating these shape
    # checks on every forward; host-side wrapper validation is covered in
    # ``test_svdquant_grouped_lowrank.py``.
    for bad in (
            torch.empty((rows, inter + 8),
                        dtype=torch.bfloat16,
                        device="cuda"),
            torch.empty((rows, inter), dtype=torch.float32, device="cuda"),
            torch.empty((inter, rows), dtype=torch.bfloat16,
                        device="cuda").T,
    ):
        with pytest.raises(AssertionError):
            _svdquant_grouped_lowrank_accumulate_cutedsl(
                x, us, vh, bad, tile_map, TILE_SIZE)


# ---------------------------------------------------------------------- #
#  Packed dual low-rank op: two factor pairs over one activation, whose   #
#  down projections share a launch and whose up projections land on the   #
#  two halves of one destination                                          #
# ---------------------------------------------------------------------- #


def _dual_operands(
    num_tiles: int,
    hidden: int,
    out_features: int,
    rank: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """Build two SVDQuant factor pairs sharing one permuted activation.

    Returns the unpacked ``Vh`` factors alongside the packed one so a test can
    drive the same arithmetic through the single-pair op.
    """
    x, us_lo, vh_lo = _lowrank_operands(num_tiles, hidden, out_features, rank,
                                        num_experts)
    _, us_hi, vh_hi = _lowrank_operands(num_tiles, hidden, out_features, rank,
                                        num_experts)
    vh_packed = torch.cat((vh_lo, vh_hi), dim=1).contiguous()
    return x, us_lo, us_hi, vh_lo, vh_hi, vh_packed


def _dual_accumulate(x: torch.Tensor, us_lo: torch.Tensor,
                     us_hi: torch.Tensor, vh_packed: torch.Tensor,
                     out: torch.Tensor, tile_map: torch.Tensor,
                     num_valid: torch.Tensor | None) -> None:
    ops = torch.ops.trtllm
    op = ops.cute_dsl_bf16_grouped_dual_lowrank_accumulate_blackwell
    if num_valid is None:
        op(x, us_lo, us_hi, vh_packed, out, tile_map, TILE_SIZE)
    else:
        op(x,
           us_lo,
           us_hi,
           vh_packed,
           out,
           tile_map,
           TILE_SIZE,
           num_non_exiting_tiles=num_valid)


@pytest.mark.parametrize("gated", [False, True])
def test_dual_lowrank_equals_two_single_pair_accumulates(gated: bool) -> None:
    # Given two factor pairs whose corrections belong on the two halves of one
    # FC13 pre-activation.
    torch.manual_seed(149)
    num_tiles, num_experts, hidden, half = 4, 4, 512, 256
    x, us_lo, us_hi, vh_lo, vh_hi, vh_packed = _dual_operands(
        num_tiles, hidden, half, 64, num_experts)
    tile_map = torch.tensor([0, 2, 2, 1], dtype=torch.int32, device="cuda")
    valid = (torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
             if gated else None)
    base = torch.randn(num_tiles * TILE_SIZE,
                       2 * half,
                       dtype=torch.bfloat16,
                       device="cuda")

    # When the packed op runs, and when the single-pair op runs once per half.
    fused = base.clone()
    _dual_accumulate(x, us_lo, us_hi, vh_packed, fused, tile_map, valid)
    staged = base.clone()
    _accumulate(x, us_lo, vh_lo, staged[:, :half], tile_map, valid)
    _accumulate(x, us_hi, vh_hi, staged[:, half:], tile_map, valid)

    # Then packing the down projections changed nothing about the arithmetic,
    # and each half still carries its own pair's correction.
    torch.testing.assert_close(fused, staged, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        fused[:, :half],
        base[:, :half] + _lowrank_reference(x, us_lo, vh_lo, tile_map,
                                            num_tiles),
        atol=2e-2,
        rtol=2e-2)
    torch.testing.assert_close(
        fused[:, half:],
        base[:, half:] + _lowrank_reference(x, us_hi, vh_hi, tile_map,
                                            num_tiles),
        atol=2e-2,
        rtol=2e-2)


def test_dual_lowrank_accumulates_onto_a_strided_full_destination() -> None:
    # Given a destination that is itself a column slice of a wider buffer, so
    # the op's own halves are strided twice over.
    torch.manual_seed(151)
    num_tiles, num_experts, hidden, half = 3, 2, 512, 256
    x, us_lo, us_hi, vh_lo, vh_hi, vh_packed = _dual_operands(
        num_tiles, hidden, half, 64, num_experts)
    tile_map = torch.tensor([1, 0, 1], dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    wide = torch.randn(num_tiles * TILE_SIZE,
                       2 * half + 128,
                       dtype=torch.bfloat16,
                       device="cuda")
    base = wide.clone()

    # When only the leading [M, 2 * half] window is accumulated onto.
    _dual_accumulate(x, us_lo, us_hi, vh_packed, wide[:, :2 * half], tile_map,
                     valid)

    # Then the trailing columns are bit-for-bit untouched, and both halves of
    # the window carry their own correction despite the row stride.
    assert torch.equal(wide[:, 2 * half:], base[:, 2 * half:])
    torch.testing.assert_close(
        wide[:, :half],
        base[:, :half] + _lowrank_reference(x, us_lo, vh_lo, tile_map,
                                            num_tiles),
        atol=2e-2,
        rtol=2e-2)
    torch.testing.assert_close(
        wide[:, half:2 * half],
        base[:, half:2 * half] + _lowrank_reference(x, us_hi, vh_hi, tile_map,
                                                    num_tiles),
        atol=2e-2,
        rtol=2e-2)


@pytest.mark.parametrize("tile_map_values", [
    [2, 2, 2, 2],
    [0, 0, 3, 3],
    [3, 2, 1, 0],
])
def test_dual_lowrank_handles_repeated_and_skipped_experts(
        tile_map_values: list[int]) -> None:
    # Given a routing whose tiles repeat experts and skip others.
    torch.manual_seed(157)
    num_tiles, num_experts, hidden, half = 4, 4, 512, 256
    x, us_lo, us_hi, vh_lo, vh_hi, vh_packed = _dual_operands(
        num_tiles, hidden, half, 64, num_experts)
    tile_map = torch.tensor(tile_map_values, dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       2 * half,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = base.clone()

    # When the correction accumulates.
    _dual_accumulate(x, us_lo, us_hi, vh_packed, out, tile_map, valid)

    # Then the map, not the tile index, picked the weight slice for both halves
    # -- the packed down projection did not smear one tile's expert onto
    # another's.
    expected = torch.cat(
        (base[:, :half] + _lowrank_reference(x, us_lo, vh_lo, tile_map,
                                             num_tiles),
         base[:, half:] + _lowrank_reference(x, us_hi, vh_hi, tile_map,
                                             num_tiles)),
        dim=1)
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_valid", [0, 1, 3])
def test_dual_lowrank_leaves_gated_off_tiles_bit_identical(
        num_valid: int) -> None:
    # Given a gate that retires the trailing tiles, whose map entries moe_sort
    # leaves uninitialized.
    torch.manual_seed(163)
    num_tiles, num_experts, hidden, half = 4, 2, 512, 256
    x, us_lo, us_hi, vh_lo, vh_hi, vh_packed = _dual_operands(
        num_tiles, hidden, half, 64, num_experts)
    tile_map = torch.full((num_tiles, ),
                          PAD_SENTINEL,
                          dtype=torch.int32,
                          device="cuda")
    tile_map[:num_valid] = torch.arange(num_valid,
                                        dtype=torch.int32,
                                        device="cuda") % num_experts
    valid = torch.tensor([num_valid], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       2 * half,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = base.clone()

    # When the correction accumulates.
    _dual_accumulate(x, us_lo, us_hi, vh_packed, out, tile_map, valid)

    # Then the retired rows keep the same bits in both halves -- a sentinel map
    # entry can never fold garbage into either one.
    live = num_valid * TILE_SIZE
    assert torch.equal(out[live:], base[live:])
    if live:
        torch.testing.assert_close(
            out[:live, :half],
            (base[:, :half] +
             _lowrank_reference(x, us_lo, vh_lo, tile_map, num_valid))[:live],
            atol=2e-2,
            rtol=2e-2)
        torch.testing.assert_close(
            out[:live, half:],
            (base[:, half:] +
             _lowrank_reference(x, us_hi, vh_hi, tile_map, num_valid))[:live],
            atol=2e-2,
            rtol=2e-2)


def test_dual_lowrank_accepts_zero_rows() -> None:
    # Given an empty permuted layout, which decode/DP boundaries produce.
    torch.manual_seed(167)
    num_experts, rank, half, hidden = 2, 64, 256, 512
    x = torch.empty((0, hidden), dtype=torch.bfloat16, device="cuda")
    us_lo = torch.randn(
        num_experts, half, rank, dtype=torch.bfloat16, device="cuda") / rank
    us_hi = torch.randn(
        num_experts, half, rank, dtype=torch.bfloat16, device="cuda") / rank
    vh_packed = torch.randn(
        num_experts, 2 * rank, hidden, dtype=torch.bfloat16,
        device="cuda") / hidden
    tile_map = torch.empty((0, ), dtype=torch.int32, device="cuda")
    valid = torch.zeros((1, ), dtype=torch.int32, device="cuda")

    # When it runs, gated and ungated, against an empty destination.
    for gate in (None, valid):
        out = torch.empty((0, 2 * half), dtype=torch.bfloat16, device="cuda")
        _dual_accumulate(x, us_lo, us_hi, vh_packed, out, tile_map, gate)

        # Then nothing is launched and the destination keeps its shape.
        assert out.shape == (0, 2 * half)


def test_dual_lowrank_is_deterministic_across_launches() -> None:
    # Given one CTA owning each output element, both accumulating epilogues are
    # fused read-modify-writes rather than racy accumulations.
    torch.manual_seed(173)
    num_tiles, num_experts, hidden, half = 4, 4, 512, 256
    x, us_lo, us_hi, _, _, vh_packed = _dual_operands(num_tiles, hidden, half,
                                                      64, num_experts)
    tile_map = torch.tensor([1, 1, 3, 0], dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       2 * half,
                       dtype=torch.bfloat16,
                       device="cuda")

    first = base.clone()
    _dual_accumulate(x, us_lo, us_hi, vh_packed, first, tile_map, valid)

    # When the identical launch is repeated.
    for _ in range(8):
        again = base.clone()
        _dual_accumulate(x, us_lo, us_hi, vh_packed, again, tile_map, valid)

        # Then the result is bit-identical, not merely close -- the two halves
        # sharing one intermediate buffer introduced no ordering dependence.
        assert torch.equal(again, first)


def test_dual_lowrank_is_capturable_with_the_gate() -> None:
    # Given the base value written inside the graph, as the FC13 GEMM does in
    # production, followed by the accumulating dual correction.
    torch.manual_seed(179)
    num_tiles, num_experts, hidden, half = 4, 4, 512, 256
    x, us_lo, us_hi, _, _, vh_packed = _dual_operands(num_tiles, hidden, half,
                                                      64, num_experts)
    tiles = torch.zeros(num_tiles, dtype=torch.int32, device="cuda")
    valid = torch.tensor([num_tiles], dtype=torch.int32, device="cuda")
    base = torch.randn(num_tiles * TILE_SIZE,
                       2 * half,
                       dtype=torch.bfloat16,
                       device="cuda")
    out = torch.empty_like(base)

    def eager(tile_map: list[int], valid_tiles: int) -> torch.Tensor:
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        valid.fill_(valid_tiles)
        want = base.clone()
        _dual_accumulate(x, us_lo, us_hi, vh_packed, want, tiles, valid)
        return want

    # Warm up so the CuteDSL JIT for the packed down projection and for the two
    # accumulating up projections all runs outside the capture -- the plain and
    # the accumulating variants are separate cache entries.
    expected = eager([0, 1, 2, 3], num_tiles)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    tiles.copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"))
    valid.fill_(num_tiles)
    with torch.cuda.graph(graph):
        out.copy_(base)
        _dual_accumulate(x, us_lo, us_hi, vh_packed, out, tiles, valid)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)

    # When the routing changes under a replay, including a shrinking gate.
    for tile_map, valid_tiles in (([3, 2, 1, 0], 4),
                                  ([2, 2, PAD_SENTINEL, PAD_SENTINEL], 2)):
        want = eager(tile_map, valid_tiles)
        torch.cuda.synchronize()
        tiles.copy_(torch.tensor(tile_map, dtype=torch.int32, device="cuda"))
        valid.fill_(valid_tiles)
        graph.replay()
        torch.cuda.synchronize()
        # Then the replay follows the new routing with no recapture, and the
        # retired rows are the freshly copied base in both halves.
        live_rows = valid_tiles * TILE_SIZE
        torch.testing.assert_close(out[:live_rows],
                                   want[:live_rows],
                                   atol=2e-2,
                                   rtol=2e-2)
        assert torch.equal(out[live_rows:], base[live_rows:])

    # And a zero valid-tile count must replay without deadlocking any of the
    # three launches, leaving the destination as the in-graph copy left it.
    tiles.fill_(PAD_SENTINEL)
    valid.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, base)


def _valid_dual_case() -> dict:
    """One accepted operand set, for a rejection test to perturb."""
    num_tiles, num_experts, hidden, half, rank = 2, 2, 512, 256, 64
    x, us_lo, us_hi, _, _, vh_packed = _dual_operands(num_tiles, hidden, half,
                                                      rank, num_experts)
    return {
        "x": x,
        "us_lo": us_lo,
        "us_hi": us_hi,
        "vh_packed": vh_packed,
        "out": torch.zeros(num_tiles * TILE_SIZE,
                           2 * half,
                           dtype=torch.bfloat16,
                           device="cuda"),
        "tile_map": torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        "rows": num_tiles * TILE_SIZE,
        "num_experts": num_experts,
        "hidden": hidden,
        "half": half,
        "rank": rank,
    }


@pytest.mark.parametrize("field", [
    "out_width",
    "out_dtype",
    "out_transposed",
    "unpacked_vh",
    "us_shape_mismatch",
    "us_expert_mismatch",
    "vh_not_contiguous",
    "rank_not_aligned",
])
def test_dual_lowrank_rejects_malformed_operands(field: str) -> None:
    # Given one operand that breaks a contract the op depends on.
    torch.manual_seed(181)
    case = _valid_dual_case()
    kwargs = {
        key: case[key]
        for key in ("x", "us_lo", "us_hi", "vh_packed", "out", "tile_map")
    }
    experts, half, rank = case["num_experts"], case["half"], case["rank"]
    hidden, rows = case["hidden"], case["rows"]

    if field == "out_width":
        # A destination that is not exactly the two halves side by side.
        kwargs["out"] = torch.zeros(rows,
                                    2 * half + 8,
                                    dtype=torch.bfloat16,
                                    device="cuda")
    elif field == "out_dtype":
        kwargs["out"] = torch.zeros(rows,
                                    2 * half,
                                    dtype=torch.float32,
                                    device="cuda")
    elif field == "out_transposed":
        # Contiguous along M instead of N, which the epilogue cannot store to.
        kwargs["out"] = torch.zeros(2 * half,
                                    rows,
                                    dtype=torch.bfloat16,
                                    device="cuda").T
    elif field == "unpacked_vh":
        # A single pair's Vh, so the high half has no factor behind it.
        kwargs["vh_packed"] = kwargs["vh_packed"][:, :rank].contiguous()
    elif field == "us_shape_mismatch":
        kwargs["us_hi"] = torch.zeros(experts,
                                      half + 8,
                                      rank,
                                      dtype=torch.bfloat16,
                                      device="cuda")
    elif field == "us_expert_mismatch":
        kwargs["us_hi"] = kwargs["us_hi"][:experts - 1].contiguous()
        kwargs["us_lo"] = kwargs["us_lo"][:experts - 1].contiguous()
    elif field == "vh_not_contiguous":
        # Splitting the rank dimension of a strided pack would interleave the
        # two factors rather than separate them.
        wider = torch.zeros(experts,
                            2 * rank + 8,
                            hidden,
                            dtype=torch.bfloat16,
                            device="cuda")
        kwargs["vh_packed"] = wider[:, :2 * rank]
    elif field == "rank_not_aligned":
        # A rank whose half-views would start off a 16B boundary.
        bad_rank = 4
        kwargs["us_lo"] = torch.zeros(experts,
                                      half,
                                      bad_rank,
                                      dtype=torch.bfloat16,
                                      device="cuda")
        kwargs["us_hi"] = torch.zeros_like(kwargs["us_lo"])
        kwargs["vh_packed"] = torch.zeros(experts,
                                          2 * bad_rank,
                                          hidden,
                                          dtype=torch.bfloat16,
                                          device="cuda")

    # Then the op refuses before anything is launched.
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        _dual_accumulate(kwargs["x"], kwargs["us_lo"], kwargs["us_hi"],
                         kwargs["vh_packed"], kwargs["out"],
                         kwargs["tile_map"], None)


def test_dual_lowrank_rejects_a_destination_aliasing_an_input() -> None:
    # Given a destination carved out of the activation's own storage, which
    # would make the epilogue race the stages that read it.
    torch.manual_seed(191)
    num_tiles, num_experts, hidden, half = 2, 2, 512, 256
    x, us_lo, us_hi, _, _, vh_packed = _dual_operands(num_tiles, hidden, half,
                                                      64, num_experts)
    tile_map = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    # Then the aliasing destination is refused before anything is launched.
    with pytest.raises((AssertionError, RuntimeError, ValueError)):
        _dual_accumulate(x, us_lo, us_hi, vh_packed, x[:, :2 * half], tile_map,
                         None)
