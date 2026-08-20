# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pair the SVDQuant FC13 permute+GEMM baseline with gather-GEMM.

The default shape is the Kimi-K2.5 T1536 forced-balanced case used by the
NVFP4 overhead report: 96 experts, top-k 8, and 128 routed rows per expert.
Only CUDA device time between events is reported; setup and CuTeDSL tuning are
warmed before sampling.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from typing import Any

import torch

import tensorrt_llm  # noqa: F401  # Load the custom operators.
from tensorrt_llm._torch.utils import ActivationType


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.fmean(ordered),
        "median_ms": statistics.median(ordered),
        "p10_ms": ordered[len(ordered) // 10],
        "p90_ms": ordered[9 * len(ordered) // 10],
    }


def _measure(operation: Callable[[], Any], iterations: int) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    output: Any = None
    for index in range(iterations):
        starts[index].record()
        output = operation()
        ends[index].record()
    del output
    torch.cuda.synchronize()
    return [starts[index].elapsed_time(ends[index]) for index in range(iterations)]


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.tokens * args.top_k % args.experts != 0:
        raise ValueError("tokens * top_k must be divisible by experts")
    if args.tokens * args.top_k // args.experts != args.rows_per_expert:
        raise ValueError("rows_per_expert does not match tokens * top_k / experts")

    device = torch.device("cuda")
    rows = args.tokens * args.top_k
    selected_experts = (torch.arange(rows, dtype=torch.int32, device=device) % args.experts).view(
        args.tokens, args.top_k
    )
    selected_scales = torch.full(
        (args.tokens, args.top_k),
        1.0 / args.top_k,
        dtype=torch.float32,
        device=device,
    )
    (
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        _expanded_idx_to_permuted_idx,
        permuted_idx_to_expanded_idx,
        total_num_padded_tokens,
        num_non_exiting_tiles,
    ) = torch.ops.trtllm.moe_sort(
        token_selected_experts=selected_experts,
        token_final_scales=selected_scales,
        num_experts=args.experts,
        top_k=args.top_k,
        local_expert_offset=0,
        local_num_experts=args.experts,
        tile_tokens_dim=args.tile_size,
    )

    # Use finite, nonzero encoded operands so the one-time correctness check
    # below cannot pass merely because both paths happened to read zeros.
    activation = torch.full(
        (args.tokens, args.hidden // 2), 0x11, dtype=torch.uint8, device=device
    ).view(torch.float4_e2m1fn_x2)
    activation_scale = torch.full(
        (args.tokens, args.hidden // 16), 120, dtype=torch.uint8, device=device
    )
    weight = torch.full(
        (args.experts, args.fc13_width, args.hidden // 2),
        0x11,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float4_e2m1fn_x2)
    weight_scale = torch.full(
        (args.experts, args.fc13_width, args.hidden // 16),
        120,
        dtype=torch.uint8,
        device=device,
    )
    alpha = torch.ones(args.experts, dtype=torch.float32, device=device)
    global_scale = torch.ones(1, dtype=torch.float32, device=device)

    def permute_then_gemm() -> torch.Tensor:
        permuted, permuted_scale = torch.ops.trtllm.moe_permute(
            activation,
            activation_scale,
            tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx,
            num_non_exiting_tiles,
            args.tile_size,
            args.top_k,
        )
        if permuted_scale is None:
            raise RuntimeError("NVFP4 permutation did not return scale factors")
        return torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
            input=permuted,
            weight=weight,
            input_scale=permuted_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            num_experts=args.experts,
            top_k=args.top_k,
            num_local_experts=args.experts,
            local_expert_offset=0,
            tile_size=args.tile_size,
            output_dtype=torch.bfloat16,
        )

    def gather_gemm() -> torch.Tensor:
        return torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_bf16_blackwell(
            input=activation,
            weight=weight,
            input_scale=activation_scale,
            weight_scale=weight_scale,
            alpha=alpha,
            tile_idx_to_group_idx=tile_idx_to_expert_idx,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            global_sf=global_scale,
            num_experts=args.experts,
            top_k=args.top_k,
            num_local_experts=args.experts,
            local_expert_offset=0,
            tile_size=args.tile_size,
            activation_type=int(ActivationType.Identity),
        )

    baseline_output = permute_then_gemm()
    gather_output = gather_gemm()
    valid_rows = int(total_num_padded_tokens.item())
    row_indices = torch.arange(valid_rows, dtype=torch.int64, device=device)
    live_mask = (
        row_indices
        < tile_idx_to_mn_limit[
            : (valid_rows + args.tile_size - 1) // args.tile_size
        ].repeat_interleave(args.tile_size)[:valid_rows]
    )
    torch.testing.assert_close(
        gather_output[:valid_rows][live_mask],
        baseline_output[:valid_rows][live_mask],
        rtol=1.6e-2,
        atol=1e-5,
    )
    del baseline_output, gather_output, live_mask, row_indices

    for _ in range(args.warmup):
        permute_then_gemm()
        gather_gemm()
    torch.cuda.synchronize()

    baseline_samples: list[float] = []
    gather_samples: list[float] = []
    for round_index in range(args.rounds):
        if round_index % 2 == 0:
            baseline_samples.extend(_measure(permute_then_gemm, args.iterations))
            gather_samples.extend(_measure(gather_gemm, args.iterations))
        else:
            gather_samples.extend(_measure(gather_gemm, args.iterations))
            baseline_samples.extend(_measure(permute_then_gemm, args.iterations))

    baseline = _summary(baseline_samples)
    gather = _summary(gather_samples)
    return {
        "case": {
            "tokens": args.tokens,
            "experts": args.experts,
            "top_k": args.top_k,
            "rows_per_expert": args.rows_per_expert,
            "hidden": args.hidden,
            "fc13_width": args.fc13_width,
            "tile_size": args.tile_size,
        },
        "permute_then_gemm": baseline,
        "gather_gemm": gather,
        "speedup": baseline["median_ms"] / gather["median_ms"],
        "saved_ms": baseline["median_ms"] - gather["median_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1536)
    parser.add_argument("--experts", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rows-per-expert", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--fc13-width", type=int, default=4096)
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(benchmark(args), sort_keys=True))


if __name__ == "__main__":
    main()
