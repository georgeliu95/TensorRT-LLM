# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""BF16 contiguous grouped GEMM over the ``moe_sort`` padded permuted layout.

``C[M, N] = A[M, K] @ B[group(m), :, :].T`` where ``group(m)`` comes from
``tile_idx_to_group_idx[m // tile_size]``.  The tile-to-group map supplies the
batch ("fake L") coordinate of the weight operand, so one launch covers every
expert without a per-expert gather and without any host readback.

The kernel body is the persistent dense GEMM in
:mod:`.dense_gemm_persistent`; this module only rewires where the B batch
coordinate comes from and skips the padded tail tiles.  The tile-to-group
contract -- how ``A``/``C`` stay at ``L == 1`` while ``B`` is indexed per tile,
and how ``num_non_exiting_tiles`` gates the scheduler -- is the same one
implemented by
:class:`~.blockscaled_contiguous_grouped_gemm.Sm100BlockScaledContiguousGroupedGemmKernel`;
keep the two in sync.
"""

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from .dense_gemm_persistent import PersistentDenseGemmKernel
from .utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)


class Sm100Bf16ContiguousGroupedGemmKernel(PersistentDenseGemmKernel):
    """Persistent BF16 grouped GEMM whose B batch index follows the M tile.

    Restrictions relative to the dense parent, all enforced by
    :meth:`can_implement`:

    - ``use_2cta_instrs`` is always ``False``.  With a 2-CTA MMA a single MMA
      tile spans two ``moe_sort`` tiles, which may belong to different experts.
    - ``mma_tiler_mn[0]`` must equal ``tile_size`` so one MMA tile is exactly
      one routing tile, which is what ``tile_idx_to_group_idx`` indexes.
    - Rasterization is along M so the tiles of one expert are visited
      consecutively and its weight slice stays resident between them.

    With ``accumulate`` the epilogue swaps its TMA store atom for the reduce-add
    variant, so the kernel computes ``C += A @ B[group(m)].T`` instead of
    overwriting C.  This is a fused read-modify-write, not a racy accumulation:
    the persistent scheduler hands each ``(M tile, N tile)`` pair to exactly one
    CTA, so every output element has a single writer and the result stays
    bit-deterministic across launches.  The addition happens in ``C``'s own dtype
    after the single accumulator rounding, which is what an out-of-place store
    followed by a separate elementwise add would also do.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        tile_size: int,
        swizzle_size: int = 1,
        accumulate: bool = False,
    ):
        if mma_tiler_mn[0] != tile_size:
            raise ValueError(
                f"Grouped GEMM needs one MMA tile per routing tile: "
                f"mma_tiler_m={mma_tiler_mn[0]} != tile_size={tile_size}.")
        super().__init__(
            acc_dtype,
            use_2cta_instrs=False,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=True,
            swizzle_size=swizzle_size,
            raster_along="m",
        )
        self.tile_size = tile_size
        # Selects the epilogue's TMA store atom at trace time, so the two
        # behaviours compile into distinct kernels instead of branching.
        self.accumulate = accumulate

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        tile_size: int,
        m: int,
        n: int,
        k: int,
        num_groups: int,
    ) -> bool:
        """Check whether this tactic can run the grouped problem."""
        if mma_tiler_mn[0] != tile_size:
            return False
        if m % tile_size != 0 or num_groups <= 0:
            return False
        # A and C carry a single batch; only B is indexed per tile.
        return PersistentDenseGemmKernel.can_implement(
            ab_dtype,
            acc_dtype,
            c_dtype,
            False,  # use_2cta_instrs
            mma_tiler_mn,
            cluster_shape_mn,
            m,
            n,
            k,
            num_groups,
            "k",  # a_major
            "k",  # b_major
            "n",  # c_major
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the grouped GEMM.

        :param a: Activation ``(M, K, 1)`` with K innermost.
        :param b: Weights ``(N, K, num_groups)`` with K innermost.
        :param c: Output ``(M, N, 1)`` with N innermost.  The M stride may
            exceed N, so C can be a column slice of a wider tensor.  Overwritten
            when ``self.accumulate`` is false and added onto when it is true.
        :param tile_idx_to_group_idx: ``(M // tile_size,)`` int32 map from
            routing tile to the batch coordinate of ``b``.  Entries at or past
            ``num_non_exiting_tiles`` are uninitialized by ``moe_sort`` and are
            clamped before use.
        :param num_non_exiting_tiles: Optional ``(1,)`` int32 live tile count.
            When present the padded tail tiles are skipped instead of computed;
            when absent every tile runs and only the clamp protects the load.
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        # Resolved at trace time so the kernel compiles two distinct variants
        # instead of branching on a runtime value.
        self.has_valid_tile_gate = num_non_exiting_tiles is not None

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for A
        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn,
                                                       tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for B
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn,
                                                       tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged,
                                    (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # Setup TMA store for C, or TMA reduce-add when accumulating onto it.
        # Only the atom differs; the epilogue's copy call is the same either way.
        if cutlass.const_expr(self.accumulate):
            c_tma_op = cpasync.CopyReduceBulkTensorTileS2GOp()
        else:
            c_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        epi_smem_layout = cute.select(self.c_smem_layout_staged, mode=[0, 1])
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            c_tma_op, c, epi_smem_layout, self.epi_tile)

        # The grid follows from C's padded shape alone, so the launch geometry
        # is independent of routing and safe to capture in a CUDA graph.
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        num_groups = cutlass.Int32(cute.size(b, mode=[2]))

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            num_groups,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tile_idx_to_group_idx: cute.Tensor,
        num_non_exiting_tiles: Optional[cute.Tensor],
        num_groups: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """Persistent grouped GEMM device kernel."""
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        # Prefetch tma desc
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        # Setup cta/thread coordinates
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                   self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64,
                                                    self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Initialize mainloop ab_pipeline (barrier) and states
        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        # Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn,
                             is_relaxed=True)

        # Setup smem tensor A/B
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        # Compute multicast mask for A/B buffer full
        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)

        # Local_tile partition global tensors
        gA_mkl = cute.local_tile(mA_mkl,
                                 cute.slice_(self.mma_tiler, (None, 0, None)),
                                 (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl,
                                 cute.slice_(self.mma_tiler, (0, None, None)),
                                 (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl,
                                 cute.slice_(self.mma_tiler, (None, None, 0)),
                                 (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        # Partition global tensor for TiledMMA_A/B/C
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        # Partition global/shared tensor for TMA load A/B
        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage))

        # Cluster wait before tensor memory alloc
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # Construct the scheduler
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
        )
        work_tile = tile_sched.initial_work_tile_info()

        # PDL: Wait for previous grid to finish.  This must come before the
        # first read of tile_idx_to_group_idx / num_non_exiting_tiles, both of
        # which the routing kernel writes.
        griddepcontrol_wait()

        # One gmem read shared by every warp role.  The skip predicate below is
        # a pure function of the tile coordinate and this value, so all three
        # roles skip exactly the same tiles and the pipelines cannot desync.
        num_valid_tiles = cutlass.Int32(0)
        if cutlass.const_expr(self.has_valid_tile_gate):
            num_valid_tiles = num_non_exiting_tiles[0]

        max_group_idx = num_groups - cutlass.Int32(1)

        # Specialized TMA load warp
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape)

                is_live_tile = cutlass.Boolean(1)
                if cutlass.const_expr(self.has_valid_tile_gate):
                    is_live_tile = mma_tile_coord_m < num_valid_tiles

                if is_live_tile:
                    # Fake L: the batch coordinate of B comes from the M tile.
                    # moe_sort leaves the entries past the live tile count
                    # uninitialized, so clamp before using it as a coordinate.
                    group_idx = tile_idx_to_group_idx[mma_tile_coord_m]
                    group_idx = cutlass.min(
                        cutlass.max(group_idx, cutlass.Int32(0)), max_group_idx)

                    # A carries a single batch; only B is indexed per tile.
                    tAgA_slice = tAgA[(None, mma_tile_coord_m, None, 0)]
                    tBgB_slice = tBgB[(None, cur_tile_coord[1], None, group_idx)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(
                            peek_ab_empty_status)

                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=b_full_mcast_mask,
                        )

                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        # Specialized MMA warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape)

                is_live_tile = cutlass.Boolean(1)
                if cutlass.const_expr(self.has_valid_tile_gate):
                    is_live_tile = mma_tile_coord_m < num_valid_tiles

                if is_live_tile:
                    tCtAcc = tCtAcc_base[(None, None, None,
                                          acc_producer_state.index)]

                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if is_leader_cta:
                        peek_ab_full_status = ab_consumer.try_wait()

                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    # Reset ACCUMULATE for each new output tile
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in range(k_tile_cnt):
                        if is_leader_cta:
                            handle = ab_consumer.wait_and_advance(
                                peek_ab_full_status)

                            num_kblocks = cute.size(tCrA, mode=[2])
                            for kblock_idx in cutlass.range(num_kblocks,
                                                            unroll_full=True):
                                kblock_crd = (None, None, kblock_idx,
                                              handle.index)
                                cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_crd],
                                          tCrB[kblock_crd], tCtAcc)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                            handle.release()

                            peek_ab_full_status = cutlass.Boolean(1)
                            if handle.count + 1 < k_tile_cnt:
                                peek_ab_full_status = ab_consumer.try_wait()

                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        sC = smem.allocate_tensor(
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        # Specialized epilogue warps
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            # TMEM -> RMEM copy setup
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                False,  # use_2cta_instrs
            )
            tAcc_epi = cute.flat_divide(
                tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r,
                                                    tAcc_epi[(None, None, 0, 0,
                                                              0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

            gC_mnl_epi = cute.flat_divide(
                tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc = cute.make_fragment(
                tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)

            # RMEM -> SMEM copy setup
            copy_atom_r2s = utils.sm100.get_smem_store_op(
                self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r)
            tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s,
                                                    tiled_copy_t2r)
            thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            tRS_sC = thr_copy_r2s.partition_D(sC)
            tTR_rC = cute.make_fragment(tTR_rAcc.shape, self.c_dtype)
            tRS_rC = tiled_copy_r2s.retile(tTR_rC)

            # SMEM -> GMEM TMA store setup
            sC_for_tma = cute.group_modes(sC, 0, 2)
            gC_for_tma = cute.group_modes(gC_mnl_epi, 0, 2)
            bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
                tma_atom_c, 0, cute.make_layout(1), sC_for_tma, gC_for_tma)

            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilogue_warp_id),
                32 * len(self.epilogue_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage, producer_group=c_producer_group)

            # Count only the subtiles this CTA actually stores.  The scheduler's
            # num_tiles_executed keeps counting skipped tiles, which would make
            # the C buffer index jump and break the store pipeline's
            # acquire/commit correspondence.
            num_prev_subtiles = cutlass.Int32(0)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                    tiled_mma.thr_id.shape)

                is_live_tile = cutlass.Boolean(1)
                if cutlass.const_expr(self.has_valid_tile_gate):
                    is_live_tile = mma_tile_coord_m < num_valid_tiles

                if is_live_tile:
                    # C carries a single batch, like A.
                    bSG_gC = bSG_gC_partitioned[(None, None, None,
                                                 mma_tile_coord_m,
                                                 cur_tile_coord[1], 0)]
                    acc_stage_index = acc_consumer_state.index
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None,
                                              acc_stage_index)]

                    # Wait for accumulator buffer full
                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3,
                                                cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                    for subtile_idx in cutlass.range(subtile_cnt):
                        # Load accumulator from TMEM to RMEM
                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

                        # Convert to output type and apply epilogue op
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                        tRS_rC.store(acc_vec)

                        # Store to SMEM
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage
                        cute.copy(tiled_copy_r2s, tRS_rC,
                                  tRS_sC[(None, None, None, c_buffer)])

                        # Fence and barrier
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        epilog_threads = 32 * len(self.epilogue_warp_id)
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                        # TMA store from SMEM to GMEM
                        if warp_idx == self.epilogue_warp_id[0]:
                            cute.copy(tma_atom_c, bSG_sC[(None, c_buffer)],
                                      bSG_gC[(None, subtile_idx)])
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=epilog_threads,
                        )

                    # Release accumulator buffer
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Wait for C store complete and deallocate TMEM
            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # PDL: Launch dependent kernels
        griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        num_groups: cutlass.Int32,
        a_stride_m: cutlass.Int32,
        c_stride_m: cutlass.Int32,
        tile_size: cutlass.Constexpr,
        has_valid_tile_gate: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Build the grouped GEMM tensors from raw pointers and run it.

        ``a_stride_m`` lets the activation be a column slice of a wider tensor
        (the SVDQuant path contracts only the leading ``K`` columns) without a
        ``.contiguous()`` copy.  The K stride is always 1.

        ``c_stride_m`` is the same escape hatch on the output side, which is what
        lets the accumulating variant add straight onto one half of a wider
        destination.  It equals ``n`` for a plain contiguous output.  The N
        stride is always 1, and TMA needs the row stride 16B aligned; the caller
        is responsible for both.

        ``has_valid_tile_gate`` selects the skip variant at compile time.  When
        it is false ``num_non_exiting_tiles_ptr`` is never dereferenced, so the
        caller may pass any valid pointer; keeping the parameter non-optional
        avoids handing ``None`` to ``cute.compile``.
        """
        num_tiles = m // tile_size
        # A and C keep a single batch so only B varies per tile.
        a = cute.make_tensor(
            a_ptr,
            layout=cute.make_layout((m, k, 1),
                                    stride=(a_stride_m, 1, m * a_stride_m)),
        )
        b = cute.make_tensor(
            b_ptr,
            layout=cute.make_ordered_layout((n, k, num_groups),
                                            order=(1, 0, 2)),
        )
        c = cute.make_tensor(
            c_ptr,
            layout=cute.make_layout((m, n, 1),
                                    stride=(c_stride_m, 1, m * c_stride_m)),
        )
        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles, )))
        num_non_exiting_tiles = None
        if cutlass.const_expr(has_valid_tile_gate):
            num_non_exiting_tiles = cute.make_tensor(
                num_non_exiting_tiles_ptr, layout=cute.make_layout((1, )))

        return self(
            a,
            b,
            c,
            tile_idx_to_group_idx,
            num_non_exiting_tiles,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )
