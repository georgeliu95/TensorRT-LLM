# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vectorized BF16 block deinterleave used by the split FC13 MoE path."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


class Sm100Bf16DeinterleaveKernel:
    """Reorder ``[up64, gate64, ...]`` into contiguous up/gate halves.

    PyTorch expresses this permutation as transpose + contiguous and lowers it
    to a generic TensorIterator copy.  Each thread here moves one aligned 128b
    vector, keeping both the source and destination transactions coalesced at
    the 64-element block boundary used by the FC13 weight layout.
    """

    threads_per_block = 256
    elements_per_thread = 8

    @cute.kernel
    def kernel(
        self,
        source: cute.Tensor,
        output: cute.Tensor,
        rows: cutlass.Constexpr,
        width: cutlass.Constexpr,
        group_size: cutlass.Constexpr,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        vector_idx = (block_idx * self.threads_per_block + thread_idx)
        vectors_per_row = width // self.elements_per_thread
        total_vectors = rows * vectors_per_row

        if vector_idx < total_vectors:
            row = vector_idx // vectors_per_row
            output_col = ((vector_idx - row * vectors_per_row)
                          * self.elements_per_thread)
            half_width = width // 2
            high_half = output_col >= half_width
            half_col = output_col
            if high_half:
                half_col = output_col - half_width
            block_in_half = half_col // group_size
            col_in_block = half_col - block_in_half * group_size
            source_col = block_in_half * (2 * group_size) + col_in_block
            if high_half:
                source_col += group_size

            copy_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            source_regs = cute.make_rmem_tensor(
                (self.elements_per_thread, ), cutlass.BFloat16)
            source_tile = cute.local_tile(
                source[row, None],
                (self.elements_per_thread, ),
                (source_col // self.elements_per_thread, ),
            )
            output_tile = cute.local_tile(
                output[row, None],
                (self.elements_per_thread, ),
                (output_col // self.elements_per_thread, ),
            )
            cute.copy(copy_atom, cute.coalesce(source_tile),
                      cute.coalesce(source_regs))
            cute.copy(copy_atom, cute.coalesce(source_regs),
                      cute.coalesce(output_tile))

    @cute.jit
    def __call__(
        self,
        source_ptr: cute.Pointer,
        output_ptr: cute.Pointer,
        rows: cutlass.Constexpr,
        width: cutlass.Constexpr,
        group_size: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        source = cute.make_tensor(
            source_ptr,
            layout=cute.make_ordered_layout((rows, width), order=(1, 0)),
        )
        output = cute.make_tensor(
            output_ptr,
            layout=cute.make_ordered_layout((rows, width), order=(1, 0)),
        )
        total_vectors = rows * width // self.elements_per_thread
        grid = ((total_vectors + self.threads_per_block - 1)
                // self.threads_per_block, 1, 1)
        self.kernel(source, output, rows, width, group_size).launch(
            grid=grid,
            block=(self.threads_per_block, 1, 1),
            stream=stream,
        )
