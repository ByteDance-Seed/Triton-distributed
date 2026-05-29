#!/usr/bin/env python3
################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

from typing import Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler

from .cutedsl_utils import decode_token_src_rank_topk_and_indices


class MegaMoEGroupGEMMCombine:
    """Fused M-contig grouped GEMM + EP push-combine (Blackwell SM100).

    Warp layout (6 warps, 192 threads):
      * Warps 0-3: epilogue (TMEM -> RMEM -> SMEM -> peer GMEM SIMT push).
      * Warp 4:    MMA (tcgen05 UMMA mainloop).
      * Warp 5:    TMA (A/B GMEM -> SMEM).

    Epilogue: TMEM -> RMEM -> single-slot SMEM -> 16-lane cooperative
    SIMT push.
    """

    kMaxWorldSize: int = 128
    reserved_smem_bytes = 1024

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
    ):
        self.acc_dtype = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = (tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE)

        self.num_mcast_ctas_a = 1
        self.num_mcast_ctas_b = 1
        self.is_a_mcast = False
        self.is_b_mcast = False
        self.occupancy = 1

        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        self.num_threads_per_warp: int = 32

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tma_load_bytes = 0

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_tile_shape_mnk = tuple(x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1)))
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape, ),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        cta_tile_n = self.cta_tile_shape_mnk[1]
        cta_tile_m = self.cta_tile_shape_mnk[0]
        epi_tile_n_target = 128
        if cta_tile_n % epi_tile_n_target == 0:
            self.epi_tile = (
                cute.make_layout(cta_tile_m),
                cute.make_layout(epi_tile_n_target),
            )
        else:
            self.epi_tile = utils.compute_epilogue_tile_shape(
                self.cta_tile_shape_mnk,
                self.use_2cta_instrs,
                self.c_layout,
                self.c_dtype,
            )

        (self.num_acc_stage, self.num_ab_stage, self.num_epi_stage) = (self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        ))

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )

        self.epi_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
        )

        self.num_tmem_alloc_cols = self._compute_num_tmem_alloc_cols(
            tiled_mma,
            self.mma_tiler,
            self.num_acc_stage,
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        output_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        dispatched_weights: cute.Tensor,
        weight_output_ptrs: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        """Launch fused grouped GEMM + EP push-combine.

        :param mA: ``(total_tokens_padded, K)`` BF16.
        :param mB: ``(N, K, num_experts)`` BF16.
        :param mC: ``(total_tokens_padded, N)`` BF16 -- used only for the C
            TMA descriptor / tile coords; the kernel never writes to it.
        :param output_ptrs: Int64 ``(world_size,)`` symmetric base pointers
            for the peer combine staging buffers.
        :param token_src_rank_topk_and_indices: Int64
            ``(total_tokens_padded,)`` encoding
            ``(src_rank << 48) | (src_topk_idx << 32) | src_token_idx`` (or
            all-ones for dropped rows).
        :param problem_sizes_m: ``(G,)`` Int32 real (un-padded) per-group
            token counts; host pre-pads A on M to ``cluster_tile_m``.
        """
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(world_size > self.kMaxWorldSize):
            raise ValueError(f"world_size={world_size} exceeds kMaxWorldSize="
                             f"{self.kMaxWorldSize}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            mA,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            mB,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.epi_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mC,
            epi_smem_layout,
            self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            total_num_clusters,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            output_ptrs,
            token_src_rank_topk_and_indices,
            dispatched_weights,
            weight_output_ptrs,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            group_count,
            problem_shape_n,
            problem_shape_k,
            problem_sizes_m,
            topk,
            rank,
            world_size,
            has_weight,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
        output_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        dispatched_weights: cute.Tensor,
        weight_output_ptrs: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            # tma_atom_c is never used to store; kept so the JIT signature
            # matches the standalone GEMM.
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_tma_producer,
        )
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_acc_consumer_threads,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # NOTE on sC swizzle: the default ``make_smem_layout_epi``
        # picks ``K_SW128`` (``Swizzle<3,4,3>``) for this BF16 +
        # (M,N)=(128,128) epi tile. With ``num_dp=32`` forcing
        # ``CopyUniversalOp`, that yields bank conflicts for BF16.
        # A custom swizzle can reduce bank conflicts, but it is
        # not on the critical path for BF16. We therefore keep
        # the default heuristic.
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer,
            swizzle=epi_smem_layout_staged.inner,
        )
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        ab_empty_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mcast_mode=2,
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mcast_mode=1,
            )
            ab_empty_mcast_mask = a_full_mcast_mask | b_full_mcast_mask
        if cutlass.const_expr(use_2cta_instrs):
            block_in_cluster_coord_vmnk_peer = (
                block_in_cluster_coord_vmnk[0] ^ 1,
                *block_in_cluster_coord_vmnk[1:],
            )
            a_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=2,
            )
            b_full_mcast_mask_peer = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk_peer,
                mcast_mode=1,
            )
            ab_empty_mcast_mask = (a_full_mcast_mask_peer
                                   | b_full_mcast_mask_peer
                                   | cutlass.Int16(0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask))

        # A/C: 2D; B: 3D with L=num_experts as a bystander dim indexed by
        # ``group_idx``.
        gA_mk = cute.local_tile(
            mA_mk,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gC_mn = cute.local_tile(
            mC_mn,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None),
        )

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mk)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mn)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        # tBgB rank-4: (grouped_mma_tile, num_n_tiles, num_k_tiles,
        # num_experts).
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        grid_dim = cute.arch.grid_dim()

        tile_sched = MContiguousGroupTileScheduler.create(
            tile_sched_params,
            bid,
            grid_dim,
            self.cluster_tile_shape_mnk,
            create_initial_search_state(),
            group_count,
            problem_sizes_m,
            problem_shape_n,
            problem_shape_k,
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        # Scheduler returns group-local indices in cta-tile units; the
        # mma_tiler partitions need global mma-tile units. Conversion:
        #   global_m_mma_tile = (tile_count_prev_group / ncluster_tile_n)
        #                       * cluster_to_mma_m
        #                       + cta_tile_idx_m / cta_group_size
        # cluster_to_mma_m collapses to 1 for (2cta, Cm=2) but stays
        # explicit so other cluster shapes remain correct.
        cta_group_size = cute.size(tiled_mma.thr_id.shape)
        cluster_to_mma_m = self.cluster_tile_shape_mnk[0] // self.mma_tiler[0]
        ncluster_tile_n = cutlass.const_expr(
            (problem_shape_n + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1])
        cta_k_tile_cnt_constexpr = cutlass.const_expr(
            (problem_shape_k + self.cluster_tile_shape_mnk[2] - 1) // self.cluster_tile_shape_mnk[2])

        # ===== TMA WARP =====
        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            work_tile = initial_work_tile_info
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_ab_stage,
            )

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                # K is constexpr in this variant, so cur_k_tile_cnt folds
                # at JIT.
                cur_k_tile_cnt = cta_k_tile_cnt_constexpr
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                         grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                tAgA_slice = tAgA[(None, global_m_mma_tile_idx, None)]
                tBgB_slice = tBgB[(None, n_tile_idx, None, cur_group_idx)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cur_k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state, )

                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state,
                        peek_ab_empty_status,
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state, ),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state, ),
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state, )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_producer_state)

        # ===== MMA WARP =====
        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            work_tile = initial_work_tile_info
            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_ab_stage,
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_acc_stage,
            )

            while work_tile.is_valid_tile:
                cur_k_tile_cnt = cta_k_tile_cnt_constexpr

                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state, )
                    acc_pipeline.producer_acquire(acc_producer_state)

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(
                            ab_consumer_state,
                            peek_ab_full_status,
                        )
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        ab_pipeline.consumer_release(ab_consumer_state)
                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state, )

                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ===== EPILOGUE WARPS =====
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx

            (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc, per_subtile_acc_shape) = (self._epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCgC,
                epi_tile,
                use_2cta_instrs,
            ))

            # subtile_cnt = cta_tile_n / epi_tile_n (constexpr). With the
            # single-slot SMEM optimisation, Phase 1 (R2S) and Phase 2
            # (cooperative push) are interleaved per-subtile so the one
            # epi-stage is freed for the next subtile immediately after
            # the push reads complete (per-warp program order guards the
            # WAR). This frees ~ ``(subtile_cnt - 1) * epi_bytes_per_stage``
            # SMEM which funds an extra ab-stage on M=128/N=256/K=2560
            # bf16, the dominant MFU-limited workload.
            subtile_cnt_const: cutlass.Constexpr[int] = cutlass.const_expr(self.cta_tile_shape_mnk[1] //
                                                                           cute.size(epi_tile[1]))

            # 4 epi warps x 32 lanes cover epi_tile_m=cta_tile_m=128 rows:
            # lane (warp, lane_idx) owns row ``warp*32 + lane_idx``.
            epi_tile_m = cutlass.const_expr(cute.size(epi_tile[0]))
            epi_tile_n = cutlass.const_expr(cute.size(epi_tile[1]))
            cta_tile_m = cutlass.const_expr(self.cta_tile_shape_mnk[0])
            cta_tile_n = cutlass.const_expr(self.cta_tile_shape_mnk[1])

            atom_v: cutlass.Constexpr[int] = 8  # 16 bytes / bf16 (2 bytes)
            assert epi_tile_n % atom_v == 0, (f"epi_tile_n={epi_tile_n} must be a multiple of atom_v=8 bf16")
            atoms_per_row: cutlass.Constexpr[int] = epi_tile_n // atom_v

            num_epi_warps = cutlass.const_expr(len(self.epilog_warp_id))
            cpy_m_per_warp = cutlass.const_expr(epi_tile_m // num_epi_warps)
            assert epi_tile_m == cpy_m_per_warp * num_epi_warps, ("epi_tile_m must be a multiple of num_epi_warps=4")
            assert epi_tile_m == cta_tile_m, ("v1 fused kernel assumes epi_tile_m == cta_tile_m")
            assert cpy_m_per_warp == self.num_threads_per_warp, ("lane-per-row push requires cpy_m_per_warp == 32")

            # 128b peer-GMEM SIMT atom (GPU scope, L1::no_allocate).
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.c_dtype,
                num_bits_per_copy=128,
                memory_scope=cute.nvgpu.MemoryScope.GPU,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

            dtype_bytes = cutlass.const_expr(self.c_dtype.width // 8)
            nbytes_per_row = cutlass.const_expr(problem_shape_n * dtype_bytes)

            weight_bytes: cutlass.Constexpr[int] = (dispatched_weights.element_type.width // 8)

            epi_lane = epi_tidx % self.num_threads_per_warp
            epi_warp_idx = epi_tidx // self.num_threads_per_warp

            # Cooperative-push layout invariants used to build the
            # 16-lane cooperative ``row_tiled_copy`` over the SMEM slot.
            lanes_per_row = cutlass.const_expr(atoms_per_row)
            assert (self.num_threads_per_warp %
                    lanes_per_row == 0), "num_threads_per_warp must be a multiple of lanes_per_row"
            row_groups_per_warp = cutlass.const_expr(self.num_threads_per_warp // lanes_per_row)
            cycles_per_warp = cutlass.const_expr(cpy_m_per_warp // row_groups_per_warp)
            assert (cpy_m_per_warp == cycles_per_warp * row_groups_per_warp), ("cpy_m_per_warp must equal "
                                                                               "cycles_per_warp * row_groups_per_warp")
            group_idx = epi_lane // lanes_per_row
            lane_in_group = epi_lane % lanes_per_row

            # Rendezvous all 4 epi warps after TMEM alloc.
            self.epilog_sync_barrier.arrive_and_wait()

            work_tile = initial_work_tile_info
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_acc_stage,
            )

            # Single-slot SMEM epilogue, two-phase per work-tile:
            #
            #   For each work-tile:
            #     For s in subtile_cnt:
            #       TMEM -> RMEM (cast to bf16)
            #       on the LAST subtile only: release acc-stage early
            #         so MMA can refill while we still drain the slot
            #       R2S bf16 -> SMEM slot 0
            #       cooperative SIMT push slot 0 -> peer GMEM
            tTR_rC = cute.make_rmem_tensor(
                per_subtile_acc_shape,
                self.c_dtype,
            )
            tiled_copy_r2s, tRS_rC, tRS_sC = (self._epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rC,
                epi_tidx,
                sC,
            ))
            assert self.num_epi_stage == 1, ("single-slot SMEM optimisation requires "
                                             f"num_epi_stage==1; got {self.num_epi_stage}")

            row_tiled_copy = cute.make_tiled_copy_tv(
                store_atom,
                cute.make_layout(atoms_per_row),
                cute.make_layout(atom_v),
            )
            row_thr_copy = row_tiled_copy.get_slice(lane_in_group)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result

                m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                         grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                mma_tile_m = cutlass.const_expr(self.mma_tiler[0])
                global_m_base = (global_m_mma_tile_idx * mma_tile_m + mma_tile_coord_v * cta_tile_m)
                global_n_base = n_tile_idx * cta_tile_n

                my_global_row = (global_m_base + epi_warp_idx * cpy_m_per_warp + epi_lane)
                encoded = token_src_rank_topk_and_indices[my_global_row]
                src_token_idx, my_dst_rank, src_topk_idx = (decode_token_src_rank_topk_and_indices(encoded))
                my_dst_slot = (src_token_idx * cutlass.Int32(topk) + src_topk_idx)
                # ``my_dst_rank == -1`` marks a dropped slot. Use rank 0
                # as a harmless pointer-table index for invalid lanes,
                # then fold the validity bit into the final address:
                # invalid lanes end up with ``my_row_base == 0`` and the
                # consumer side gates on ``addr != 0``. This avoids an
                # out-of-bounds ``output_ptrs[-1]`` load for dropped rows.
                my_pred = (my_dst_rank + cutlass.Int32(1)) != cutlass.Int32(0)
                safe_dst_rank = cutlass.Int32(0)
                safe_dst_slot = cutlass.Int32(0)
                if my_pred:
                    safe_dst_rank = my_dst_rank
                    safe_dst_slot = my_dst_slot

                my_remote_base = output_ptrs[safe_dst_rank]
                my_row_base = ((my_remote_base + cutlass.Int64(safe_dst_slot) * cutlass.Int64(nbytes_per_row)) *
                               cutlass.Int64(my_pred))

                # Weight push: one scalar per row, gated on
                # ``n_tile_idx == 0`` so multi-N-tile epilogues emit
                # exactly one store per (lane, row). Placed before
                # ``consumer_wait`` so the scalar load co-schedules
                # with the acc-stage barrier stall.
                if cutlass.const_expr(has_weight):
                    if n_tile_idx == cutlass.Int32(0) and my_pred:
                        w_val = dispatched_weights[my_global_row]
                        w_dst_addr = (weight_output_ptrs[safe_dst_rank] +
                                      cutlass.Int64(safe_dst_slot) * cutlass.Int64(weight_bytes))
                        w_dst = cute.make_tensor(
                            cute.make_ptr(
                                dispatched_weights.element_type,
                                w_dst_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=weight_bytes,
                            ),
                            cute.make_layout(1),
                        )
                        w_dst[0] = w_val

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))

                for subtile_idx in cutlass.range_constexpr(subtile_cnt_const):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc_mn,
                        tTR_rAcc,
                    )
                    cute.arch.fence_view_async_tmem_load()
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc, ).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))

                    if cutlass.const_expr(subtile_idx == subtile_cnt_const - 1):
                        cute.arch.fence_view_async_tmem_load()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_consumer_state, )
                        acc_consumer_state.advance()

                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, 0)],
                    )

                    cute.arch.sync_warp()
                    # cute.arch.fence_proxy(
                    #     "async.shared",
                    #     space="cta",
                    # )

                    n_sub_offset = subtile_idx * epi_tile_n
                    cur_global_n_base = global_n_base + n_sub_offset
                    n_byte_offset = (cutlass.Int64(cur_global_n_base) * cutlass.Int64(dtype_bytes))
                    sC_subtile = sC[(None, None, 0)]
                    for cycle in cutlass.range_constexpr(cycles_per_warp):
                        leader_lane = (cycle * row_groups_per_warp + group_idx)
                        bk_row_base = cute.arch.shuffle_sync(my_row_base, leader_lane)
                        cta_row_in_subtile = (epi_warp_idx * cpy_m_per_warp + cycle * row_groups_per_warp + group_idx)
                        sC_row = sC_subtile[(cta_row_in_subtile, None)]
                        gOut_row = cute.make_tensor(
                            cute.make_ptr(
                                self.c_dtype,
                                bk_row_base + n_byte_offset,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(epi_tile_n),
                        )
                        tCsT_row = row_thr_copy.partition_S(sC_row)
                        tCgT_row = row_thr_copy.partition_D(gOut_row)
                        # ``bk_row_base == 0`` only happens for lanes
                        # whose source row was dropped (see comment on
                        # ``my_row_base`` -- the null page is unmapped
                        # so no real allocation can land at 0).
                        if bk_row_base != cutlass.Int64(0):
                            cute.copy(store_atom, tCsT_row, tCgT_row)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

    def _epilog_tmem_copy_and_partition(self, tidx, tAcc, tCgC, epi_tile, use_2cta_instrs):
        """``tCgC`` is the partitioned output tensor. Since C is 2D (no L),
        its rank is one less than the original MKL-style example (MMA atom
        modes + num_m_tiles + num_n_tiles, without a trailing num_l_tiles).
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r,
            tAcc_epi[(None, None, 0, 0, 0)],
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # 2D C: tCgC has shape ((VEC_M, VEC_N), atom_m, atom_n, num_m, num_n).
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None)],
            epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_epi)
        per_subtile_acc_shape = tTR_gC[(None, None, None, 0, 0, 0, 0)].shape
        tTR_rAcc = cute.make_rmem_tensor(
            per_subtile_acc_shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc, per_subtile_acc_shape

    def _epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            tiled_copy_t2r,
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def _epilog_gmem_copy_and_partition(self, tma_atom_c, tCgC, epi_tile, sC):
        """2D C: tCgC rank is ((VEC_M, VEC_N), atom_m, atom_n, num_m, num_n)."""
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None)],
            epi_tile,
        )
        sC_for_tma = cute.group_modes(sC, 0, 2)
        gC_for_tma = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma,
            gC_for_tma,
        )
        return tma_atom_c, bSG_sC, bSG_gC

    @staticmethod
    def _compute_stages(tiled_mma, mma_tiler_mnk, a_dtype, b_dtype, epi_tile, c_dtype, c_layout, smem_capacity,
                        occupancy):
        num_acc_stage = 2
        # Hard-pinned to a single epi slot.
        num_epi_stage = 1

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        epi_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = cute.size_in_bytes(a_dtype, a_smem_layout_stage_one) + cute.size_in_bytes(
            b_dtype, b_smem_layout_staged_one)
        epi_bytes_per_stage = cute.size_in_bytes(
            c_dtype,
            epi_smem_layout_staged_one,
        )
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        # ab-stages take all remaining SMEM
        num_ab_stage = (smem_capacity // occupancy - MegaMoEGroupGEMMCombine.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage

        return num_acc_stage, num_ab_stage, num_epi_stage

    @staticmethod
    def _compute_grid(total_num_clusters, cluster_shape_mn, max_active_clusters):
        problem_shape_ntile_mnl = (
            cluster_shape_mn[0],
            cluster_shape_mn[1],
            cutlass.Int32(total_num_clusters),
        )
        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl,
            (*cluster_shape_mn, 1),
        )
        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(
            tile_sched_params,
            max_active_clusters,
        )
        return tile_sched_params, grid

    @staticmethod
    def _compute_num_tmem_alloc_cols(tiled_mma, mma_tiler, num_acc_stage):
        acc_shape = tiled_mma.partition_shape_C(mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, num_acc_stage))
        return utils.get_num_tmem_alloc_cols(tCtAcc_fake)
