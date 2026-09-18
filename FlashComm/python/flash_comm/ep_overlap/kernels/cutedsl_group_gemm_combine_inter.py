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
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import nccl.core.device.cute as nccl_cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .cutedsl_group_gemm_combine import MegaMoEGroupGEMMCombine
from .cutedsl_group_gemm_combine_sm90_inter import (
    MegaMoEGroupGEMMCombineInterSM90,
    _rail_slot_stride_bytes,
)
from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


class MegaMoEGroupGEMMCombineInter(
        MegaMoEGroupGEMMCombine,
        MegaMoEGroupGEMMCombineInterSM90,
):
    """Single-launch Blackwell FC2 UMMA GEMM + inter-node combine.

    The GEMM warps are phase-switched after all FC2 TMA stores become globally
    visible.  The post-GEMM phase deliberately reuses the proven bulk
    inter-node protocol from the Hopper kernel: local MNNVL pull/reduce, NCCL
    GIN exchange, then the final per-node reduction.

    Warp layout during GEMM:
      * Warps 0-3: TMEM epilogue and TMA stores to peer-visible FC2 staging.
      * Warp 4:    2-CTA UMMA mainloop.
      * Warp 5:    A/B TMA producer.
      * Warps 6-11: Reserved for the post-GEMM combine phase.

    After GEMM every thread helps build or reduce partials, except block 0
    warp 0, which owns the GIN control path.
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
    ):
        MegaMoEGroupGEMMCombine.__init__(
            self,
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
        )
        # Extra warps do not participate in UMMA or its pipelines.  They only
        # increase the SIMT parallelism of the communication/reduction phase;
        # the persistent grid already limits this kernel to one CTA per SM.
        self.threads_per_cta = 12 * cute.arch.WARP_SIZE
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_cta,
        )

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        c_layout,
        smem_capacity,
        occupancy,
    ):
        """Size a multi-stage TMA epilogue for FC2 materialization."""
        num_acc_stage = 2
        num_epi_stage = 2
        a_stage = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_stage = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        epi_stage = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
        ab_bytes = (cute.size_in_bytes(a_dtype, a_stage) + cute.size_in_bytes(b_dtype, b_stage))
        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_stage)
        epi_bytes = epi_bytes_per_stage * num_epi_stage
        num_ab_stage = (smem_capacity // occupancy - MegaMoEGroupGEMMCombineInter.reserved_smem_bytes -
                        epi_bytes) // ab_bytes
        remaining_smem = (smem_capacity - occupancy * ab_bytes * num_ab_stage - occupancy *
                          (MegaMoEGroupGEMMCombineInter.reserved_smem_bytes + epi_bytes))
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_epi_stage

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        output: cute.Tensor,
        output_weight: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        barrier_workspace: cute.Tensor,
        barrier_workspace_ptrs: cute.Tensor,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_topk_send_mask: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
        num_tokens: cutlass.Int32,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        run_reduce: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype != cutlass.Float16 and self.a_dtype != cutlass.BFloat16):
            raise TypeError("SM100 inter group_gemm_combine supports fp16/bf16 inputs")
        if cutlass.const_expr(problem_shape_n % self.post_atom_v != 0):
            raise ValueError("SM100 inter group_gemm_combine requires hidden output divisible by post_atom_v")

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
            output,
            output_weight,
            combine_x_ptrs,
            barrier_workspace,
            barrier_workspace_ptrs,
            dev_comm_ptr,
            rdma_rail_send_win_handle,
            num_tokens_per_rank,
            local_topk_indices,
            local_topk_send_mask,
            local_token_dst_scatter_indices,
            num_tokens,
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
            local_world_size,
            max_slot_num_token,
            has_weight,
            run_reduce,
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
        output: cute.Tensor,
        output_weight: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        barrier_workspace: cute.Tensor,
        barrier_workspace_ptrs: cute.Tensor,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_topk_send_mask: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
        num_tokens: cutlass.Int32,
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
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        run_reduce: cutlass.Constexpr[int],
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        gdx, gdy, gdz = cute.arch.grid_dim()
        block_linear = (bidz * gdy + bidy) * gdx + bidx
        grid_ctas = gdx * gdy * gdz

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

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
            ab_empty_mcast_mask = (a_full_mcast_mask_peer | b_full_mcast_mask_peer
                                   | cutlass.Int16(0 if ab_empty_mcast_mask is None else ab_empty_mcast_mask))

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
                grouped_info = work_tile.group_search_result
                cur_group_idx = grouped_info.group_idx

                m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                         grouped_info.cta_tile_idx_m // cta_group_size)
                n_tile_idx = grouped_info.cta_tile_idx_n

                tAgA_slice = tAgA[(None, global_m_mma_tile_idx, None)]
                tBgB_slice = tBgB[(None, n_tile_idx, None, cur_group_idx)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < cta_k_tile_cnt_constexpr:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for _ in cutlass.range(0, cta_k_tile_cnt_constexpr, 1, unroll=1):
                    ab_pipeline.producer_acquire(
                        ab_producer_state,
                        peek_ab_empty_status,
                    )
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, ab_producer_state.count)],
                        tAsA[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, ab_producer_state.count)],
                        tBsB[(None, ab_producer_state.index)],
                        tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state),
                        mcast_mask=b_full_mcast_mask,
                    )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cta_k_tile_cnt_constexpr:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

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
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cta_k_tile_cnt_constexpr:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)
                    acc_pipeline.producer_acquire(acc_producer_state)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for _ in cutlass.range(0, cta_k_tile_cnt_constexpr, 1, unroll=1):
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
                        if ab_consumer_state.count < cta_k_tile_cnt_constexpr:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    acc_pipeline.producer_commit(acc_producer_state)
                    acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ===== EPILOGUE WARPS =====
        if (warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile):
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc, _) = self._epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                tCgC,
                epi_tile,
                use_2cta_instrs,
            )

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = (self._epilog_smem_copy_and_partition(
                tiled_copy_t2r,
                tTR_rC,
                epi_tidx,
                sC,
            ))
            (tma_atom_c, bSG_sC, bSG_gC_partitioned) = self._epilog_gmem_copy_and_partition(
                tma_atom_c,
                tCgC,
                epi_tile,
                sC,
            )

            work_tile = initial_work_tile_info
            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_acc_stage,
            )
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_epi_stage,
                producer_group=c_producer_group,
            )

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                         grouped_info.cta_tile_idx_m // cta_group_size)
                n_tile_idx = grouped_info.cta_tile_idx_n

                bSG_gC = bSG_gC_partitioned[(None, None, None, global_m_mma_tile_idx, n_tile_idx)]
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                acc_pipeline.consumer_wait(acc_consumer_state)
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                for subtile_idx in range(subtile_cnt):
                    epi_buffer = ((num_prev_subtiles + subtile_idx) % self.num_epi_stage)
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                    tRS_rC.store(acc_vec.to(self.c_dtype))
                    cute.copy(
                        tiled_copy_r2s,
                        tRS_rC,
                        tRS_sC[(None, None, None, epi_buffer)],
                    )
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(
                            tma_atom_c,
                            bSG_sC[(None, epi_buffer)],
                            bSG_gC[(None, subtile_idx)],
                        )
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            c_pipeline.producer_tail()

        # The TMA store tail above completes every FC2 row before any CTA
        # pull-reads local peers.  All twelve warps now switch to combine work.
        self._grid_barrier(
            barrier_workspace,
            self.kGridBarrierGemmDone,
            grid_ctas,
            tidx,
        )
        self._publish_and_wait_local_gemm_ready(
            barrier_workspace,
            barrier_workspace_ptrs,
            block_linear,
            tidx,
            local_world_size,
        )
        self._pipeline_inter_partials_and_exchange(
            barrier_workspace,
            mC_mn,
            combine_x_ptrs,
            dev_comm_ptr,
            rdma_rail_send_win_handle,
            num_tokens_per_rank,
            local_topk_indices,
            local_topk_send_mask,
            local_token_dst_scatter_indices,
            block_linear,
            grid_ctas,
            tidx,
            warp_idx,
            problem_shape_n,
            topk,
            rank,
            world_size,
            local_world_size,
            max_slot_num_token,
            group_count,
        )
        self._wait_local_flag(
            barrier_workspace,
            self.kExchangeDoneFlag,
            tidx,
        )
        if cutlass.const_expr(run_reduce):
            self._reduce_inter_partials(
                output,
                output_weight,
                rdma_rail_send_win_handle,
                block_linear,
                grid_ctas,
                tidx,
                problem_shape_n,
                topk,
                rank,
                world_size,
                local_world_size,
                max_slot_num_token,
                group_count,
                has_weight,
                num_tokens,
            )

    @cute.jit
    def _build_inter_partial_owner(
        self,
        fc2_output: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
        owner_node: cutlass.Int32,
        block_linear: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        tidx: cutlass.Int32,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
    ):
        """Build two adjacent vectors per thread and reuse routing metadata."""
        _ = fc2_output
        dtype = self.c_dtype
        dtype_bytes: cutlass.Constexpr[int] = dtype.width // 8
        atom_v: cutlass.Constexpr[int] = self.post_atom_v
        hidden_vecs: cutlass.Constexpr[int] = hidden_size // atom_v
        task_vecs: cutlass.Constexpr[int] = (hidden_vecs + 1) // 2
        row_byte_stride: cutlass.Constexpr[int] = hidden_size * dtype_bytes
        slot_stride_bytes: cutlass.Constexpr[int] = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        atom_layout = cute.make_layout(atom_v)
        tCr0 = cute.make_fragment((atom_v, ), dtype)
        tCr1 = cute.make_fragment((atom_v, ), dtype)
        tCrAcc0 = cute.make_fragment((atom_v, ), cutlass.Float32)
        tCrAcc1 = cute.make_fragment((atom_v, ), cutlass.Float32)
        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)

        nnodes = cutlass.Int32(world_size // local_world_size)
        my_node = cutlass.Int32(rank // local_world_size)
        local_rank = cutlass.Int32(rank % local_world_size)
        my_node_rank_begin = my_node * cutlass.Int32(local_world_size)
        my_node_rank_end = my_node_rank_begin + cutlass.Int32(local_world_size)
        src_rank = owner_node * cutlass.Int32(local_world_size) + local_rank
        src_num_token = num_tokens_per_rank[src_rank]
        partial_slot = nnodes + owner_node
        if owner_node == my_node:
            partial_slot = nnodes * cutlass.Int32(2) + my_node

        global_thread = block_linear * cutlass.Int32(self.threads_per_cta) + tidx
        total_threads = grid_ctas * cutlass.Int32(self.threads_per_cta)
        worker_id = global_thread
        worker_count = total_threads
        if block_linear == cutlass.Int32(0) and tidx < cutlass.Int32(32):
            worker_id = cutlass.Int32(-1)
        else:
            worker_id = global_thread - cutlass.Int32(32)
            worker_count = total_threads - cutlass.Int32(32)

        total_work = cutlass.Int32(max_slot_num_token * task_vecs)
        linear = worker_id
        while linear >= cutlass.Int32(0) and linear < total_work:
            task_vec = linear % cutlass.Int32(task_vecs)
            token_offset = linear // cutlass.Int32(task_vecs)
            if token_offset < src_num_token:
                vec0 = task_vec * cutlass.Int32(2)
                vec1 = vec0 + cutlass.Int32(1)
                atom_byte_offset0 = (cutlass.Int64(vec0 * atom_v) * cutlass.Int64(dtype_bytes))
                atom_byte_offset1 = (cutlass.Int64(vec1 * atom_v) * cutlass.Int64(dtype_bytes))
                tCrAcc0.fill(0.0)
                tCrAcc1.fill(0.0)
                for k in cutlass.range_constexpr(topk):
                    expert_idx = local_topk_indices[owner_node, token_offset, k]
                    expert_rank = expert_idx // cutlass.Int32(num_experts_per_rank)
                    scatter = local_token_dst_scatter_indices[owner_node, token_offset, k]
                    is_valid = (expert_rank >= my_node_rank_begin and expert_rank < my_node_rank_end and scatter >= 0)
                    if is_valid:
                        expert_local_rank = expert_rank - my_node_rank_begin
                        src_base = combine_x_ptrs[expert_local_rank]
                        src_row = (src_base + cutlass.Int64(scatter) * cutlass.Int64(row_byte_stride))
                        gSrc0 = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                src_row + atom_byte_offset0,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            atom_layout,
                        )
                        cute.copy(copy_atom, gSrc0, tCr0)
                        tCrAcc0.store(tCrAcc0.load() + tCr0.load().to(cutlass.Float32))
                        if vec1 < cutlass.Int32(hidden_vecs):
                            gSrc1 = cute.make_tensor(
                                cute.make_ptr(
                                    dtype,
                                    src_row + atom_byte_offset1,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                atom_layout,
                            )
                            cute.copy(copy_atom, gSrc1, tCr1)
                            tCrAcc1.store(tCrAcc1.load() + tCr1.load().to(cutlass.Float32))

                dst_row_off = (cutlass.Int64(partial_slot) * cutlass.Int64(slot_stride_bytes) +
                               cutlass.Int64(token_offset) * cutlass.Int64(row_byte_stride))
                tCr0.store(tCrAcc0.load().to(dtype))
                gDst0 = cute.make_tensor(
                    cute.make_ptr(
                        dtype,
                        rdma_win.local_pointer(dst_row_off + atom_byte_offset0),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    atom_layout,
                )
                cute.copy(copy_atom, tCr0, gDst0)
                if vec1 < cutlass.Int32(hidden_vecs):
                    tCr1.store(tCrAcc1.load().to(dtype))
                    gDst1 = cute.make_tensor(
                        cute.make_ptr(
                            dtype,
                            rdma_win.local_pointer(dst_row_off + atom_byte_offset1),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        atom_layout,
                    )
                    cute.copy(copy_atom, tCr1, gDst1)
            linear += worker_count
