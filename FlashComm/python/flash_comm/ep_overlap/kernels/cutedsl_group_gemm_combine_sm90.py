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

import math
from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .cutedsl_utils import decode_token_src_rank_topk_and_indices
from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


class MegaMoEGroupGEMMCombineSM90:
    """Fused M-contiguous grouped GEMM + push-combine for Hopper.

    The GEMM mainloop mirrors the tested SM90 TMA+WGMMA standalone kernel.
    The epilogue casts register accumulators to SMEM and pushes valid rows
    directly to peer combine staging buffers using the encoded token metadata.
    """

    kMaxWorldSize: int = 128
    reserved_smem_bytes = 1024

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        wait_signals: bool = False,
    ):
        if use_2cta_instrs:
            raise ValueError("SM90 WGMMA path does not support SM100 2-CTA instructions")

        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.wait_signals = wait_signals

        self.atom_layout_mnk = ((2, 1, 1) if self.mma_tiler[0] > 64 and self.mma_tiler[1] > 128 else (1, 1, 1))
        self.num_mcast_ctas_a = cluster_shape_mn[1]
        self.num_mcast_ctas_b = cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.occupancy = 1

        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = 32 * self.num_warps_per_warp_group
        self.threads_per_cta = (self.num_dma_warp_groups + self.num_mma_warp_groups) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group

        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_threads,
        )

    def _setup_attributes(self):
        if self.mma_tiler[0] not in [64, 128]:
            raise ValueError("SM90 CTA tile shape M must be 64 or 128")
        if self.mma_tiler[1] not in [64, 128, 256]:
            raise ValueError("SM90 CTA tile shape N must be 64, 128, or 256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.mma_tiler[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * 4,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.cluster_tile_shape_mnk = (
            self.mma_tiler[0] * self.cluster_shape_mn[0],
            self.mma_tiler[1] * self.cluster_shape_mn[1],
            self.mma_tiler[2],
        )

        # Keep the full N tile so each combine row is pushed by a full warp.
        # The generic SM90 helper may split cooperative tiles into 128-column
        # epilogue subtiles, which is very slow for peer-GMEM combine stores.
        self.epi_tile = (self.mma_tiler[0], self.mma_tiler[1])

        self.num_ab_stage, self.num_epi_stage = self._compute_stages(
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_epi_stage,
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
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(mA)
        self.b_layout = utils.LayoutEnum.from_tensor(mB)
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype != cutlass.Float16 and self.a_dtype != cutlass.BFloat16):
            raise TypeError("SM90 group GEMM combine supports fp16/bf16 inputs")
        if cutlass.const_expr(world_size > self.kMaxWorldSize):
            raise ValueError(f"world_size={world_size} exceeds kMaxWorldSize={self.kMaxWorldSize}")

        self._setup_attributes()

        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            mA,
            self.a_smem_layout_staged,
            (self.mma_tiler[0], self.mma_tiler[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = self._make_tma_atoms_and_tensors(
            mB,
            self.b_smem_layout_staged,
            (self.mma_tiler[1], self.mma_tiler[2]),
            self.cluster_shape_mn[0],
        )
        tile_sched_params, grid = self._compute_grid(
            total_num_clusters,
            self.cluster_shape_mn,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype,
                    cute.cosize(self.a_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype,
                    cute.cosize(self.b_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            mC,
            output_ptrs,
            token_src_rank_topk_and_indices,
            dispatched_weights,
            weight_output_ptrs,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
            tile_sched_params,
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
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mC_mn: cute.Tensor,
        output_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        dispatched_weights: cute.Tensor,
        weight_output_ptrs: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
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
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        bid = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=1)
        b_mcast_mask = cute.make_layout_image_mask(cta_layout_mnk, cluster_coord_mnk, mode=0)
        a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
        b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout) + cute.size_in_bytes(
            self.b_dtype, b_smem_layout)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                (self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1) * self.num_mma_warp_groups *
                self.num_warps_per_warp_group,
            ),
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner,
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer,
            swizzle=epi_smem_layout_staged.inner,
        )

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

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_mnk[1],
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mk, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_mnk[0],
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups,
            stride=self.num_threads_per_warp_group,
        )
        thr_mma = tiled_mma.get_slice(mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups))
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        tCgC = thr_mma.partition_C(gC_mn)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

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
        work_tile = tile_sched.initial_work_tile_info()

        ncluster_tile_n = cutlass.const_expr(
            (problem_shape_n + self.cluster_tile_shape_mnk[1] - 1) // self.cluster_tile_shape_mnk[1])
        cta_k_tile_cnt_constexpr = cutlass.const_expr(
            (problem_shape_k + self.cluster_tile_shape_mnk[2] - 1) // self.cluster_tile_shape_mnk[2])
        cluster_to_mma_m = self.cluster_tile_shape_mnk[0] // self.mma_tiler[0]

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        if warp_idx == self.load_warp_id:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_ab_stage,
            )

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                cur_group_idx = grouped_info.group_idx
                cur_k_tile_cnt = cta_k_tile_cnt_constexpr

                m_cluster_tile_base = tile_sched.search_state.tile_count_prev_group // ncluster_tile_n
                global_m_tile_idx = m_cluster_tile_base * cluster_to_mma_m + grouped_info.cta_tile_idx_m
                n_tile_idx = grouped_info.cta_tile_idx_n

                tAgA_slice = tAgA[(None, global_m_tile_idx, None)]
                tBgB_slice = tBgB[(None, n_tile_idx, None, cur_group_idx)]

                producer_state.reset_count()
                for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, producer_state.count)],
                        tAsA[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                        mcast_mask=a_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, producer_state.count)],
                        tBsB[(None, producer_state.index)],
                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(producer_state),
                        mcast_mask=b_mcast_mask,
                    )
                    mainloop_pipeline.producer_commit(producer_state)
                    producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(producer_state)

        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)

            consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_ab_stage,
            )
            consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_ab_stage,
            )
            num_k_blocks = cute.size(tCrA, mode=[2])

            (
                tiled_copy_r2s,
                tRS_sC,
                tRS_rAcc,
                tRS_rC,
                tRS_rC_out,
                size_tRS_rC,
            ) = self._epilog_smem_copy_and_partition(tidx, tiled_mma, accumulators, sC)

            k_pipe_mmas = 1

            while work_tile.is_valid_tile:
                grouped_info = work_tile.group_search_result
                cur_k_tile_cnt = cta_k_tile_cnt_constexpr

                m_cluster_tile_base = tile_sched.search_state.tile_count_prev_group // ncluster_tile_n
                global_m_tile_idx = m_cluster_tile_base * cluster_to_mma_m + grouped_info.cta_tile_idx_m
                n_tile_idx = grouped_info.cta_tile_idx_n

                consumer_read_state.reset_count()
                consumer_release_state.reset_count()
                accumulators.fill(0.0)
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                cute.nvgpu.warpgroup.fence()

                prologue_mma_cnt = cutlass.min(k_pipe_mmas, cur_k_tile_cnt)
                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    mainloop_pipeline.consumer_wait(consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    consumer_read_state.advance()

                for k_tile in cutlass.range(prologue_mma_cnt, cur_k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.consumer_wait(consumer_read_state)
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            consumer_read_state.index,
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA[k_block_coord],
                            tCrB[k_block_coord],
                            accumulators,
                        )
                    cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    mainloop_pipeline.consumer_release(consumer_release_state)
                    consumer_release_state.advance()
                    consumer_read_state.advance()

                cute.nvgpu.warpgroup.wait_group(0)
                for k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                    mainloop_pipeline.consumer_release(consumer_release_state)
                    consumer_release_state.advance()

                global_m_base = global_m_tile_idx * self.mma_tiler[0]
                global_n_base = n_tile_idx * self.mma_tiler[1]
                self._epilog(
                    sC,
                    tiled_copy_r2s,
                    tRS_sC,
                    tRS_rAcc,
                    tRS_rC,
                    tRS_rC_out,
                    size_tRS_rC,
                    tile_sched.num_tiles_executed,
                    warp_idx,
                    global_m_base,
                    global_n_base,
                    problem_shape_n,
                    topk,
                    output_ptrs,
                    token_src_rank_topk_and_indices,
                    dispatched_weights,
                    weight_output_ptrs,
                    has_weight,
                )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

    def _epilog_smem_copy_and_partition(self, tidx, tiled_mma, accumulators, sC):
        copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
            self.c_layout,
            elem_ty_d=self.c_dtype,
            elem_ty_acc=self.acc_dtype,
        )
        copy_atom_c = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(
                self.c_layout.is_m_major_c(),
                4,
            ),
            self.c_dtype,
        )
        tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_c_atom)

        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rAcc = tiled_copy_r2s.retile(accumulators)
        rC_shape = cute.shape(thr_copy_r2s.partition_S(sC))
        tRS_rC_layout = cute.make_layout(rC_shape[:3])
        tRS_rC = cute.make_rmem_tensor(tRS_rC_layout.shape, self.acc_dtype)
        tRS_rC_out = cute.make_rmem_tensor(tRS_rC_layout.shape, self.c_dtype)
        size_tRS_rC = cute.size(tRS_rC)
        return tiled_copy_r2s, tRS_sC, tRS_rAcc, tRS_rC, tRS_rC_out, size_tRS_rC

    @cute.jit
    def _epilog(
        self,
        sC,
        tiled_copy_r2s,
        tRS_sC,
        tRS_rAcc,
        tRS_rC,
        tRS_rC_out,
        size_tRS_rC,
        num_tiles_executed,
        warp_idx,
        global_m_base,
        global_n_base,
        problem_shape_n: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        output_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        dispatched_weights: cute.Tensor,
        weight_output_ptrs: cute.Tensor,
        has_weight: cutlass.Constexpr[int],
    ):
        epi_tile_m = cutlass.const_expr(self.epi_tile[0])
        epi_tile_n = cutlass.const_expr(self.epi_tile[1])
        atom_v: cutlass.Constexpr[int] = 8
        atoms_per_row: cutlass.Constexpr[int] = epi_tile_n // atom_v
        row_groups_per_warp: cutlass.Constexpr[int] = 32 // atoms_per_row
        cycles_per_warp: cutlass.Constexpr[int] = epi_tile_m // row_groups_per_warp
        dtype_bytes = cutlass.const_expr(self.c_dtype.width // 8)
        nbytes_per_row = cutlass.const_expr(problem_shape_n * dtype_bytes)
        weight_bytes: cutlass.Constexpr[int] = dispatched_weights.element_type.width // 8

        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.c_dtype,
            num_bits_per_copy=128,
            memory_scope=cute.nvgpu.MemoryScope.GPU,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
        )
        row_tiled_copy = cute.make_tiled_copy_tv(
            store_atom,
            cute.make_layout(atoms_per_row),
            cute.make_layout(atom_v),
        )
        lane_idx = cute.arch.lane_idx()
        group_idx = lane_idx // atoms_per_row
        lane_in_group = lane_idx % atoms_per_row
        row_thr_copy = row_tiled_copy.get_slice(lane_in_group)

        epi_tile_num: cutlass.Constexpr[int] = cutlass.const_expr(self.mma_tiler[1] // epi_tile_n)
        num_prev_epi_tiles = num_tiles_executed * epi_tile_num

        for epi_idx in range(epi_tile_num):
            for epi_v in range(size_tRS_rC):
                tRS_rC[epi_v] = tRS_rAcc[epi_idx * size_tRS_rC + epi_v]

            acc_vec = tRS_rC.load()
            tRS_rC_out.store(acc_vec.to(self.c_dtype))

            epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(tRS_sC, mode=[3])
            cute.copy(
                tiled_copy_r2s,
                tRS_rC_out,
                tRS_sC[(None, None, None, epi_buffer)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            self.epilog_sync_barrier.arrive_and_wait()

            if warp_idx == self.epi_store_warp_id:
                n_sub_offset = epi_idx * epi_tile_n
                cur_global_n_base = global_n_base + n_sub_offset
                n_byte_offset = cutlass.Int64(cur_global_n_base) * cutlass.Int64(dtype_bytes)
                sC_subtile = sC[(None, None, epi_buffer)]

                for cycle in cutlass.range_constexpr(cycles_per_warp):
                    cta_row = cycle * row_groups_per_warp + group_idx
                    my_global_row = global_m_base + cta_row
                    encoded = token_src_rank_topk_and_indices[my_global_row]
                    src_token_idx, my_dst_rank, src_topk_idx = decode_token_src_rank_topk_and_indices(encoded)
                    my_dst_slot = src_token_idx * cutlass.Int32(topk) + src_topk_idx
                    my_pred = (my_dst_rank + cutlass.Int32(1)) != cutlass.Int32(0)

                    safe_dst_rank = cutlass.Int32(0)
                    safe_dst_slot = cutlass.Int32(0)
                    if my_pred:
                        safe_dst_rank = my_dst_rank
                        safe_dst_slot = my_dst_slot

                    my_remote_base = output_ptrs[safe_dst_rank]
                    my_row_base = (my_remote_base + cutlass.Int64(safe_dst_slot) *
                                   cutlass.Int64(nbytes_per_row)) * cutlass.Int64(my_pred)

                    if cutlass.const_expr(has_weight):
                        if epi_idx == 0 and global_n_base == cutlass.Int32(0) and my_pred:
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

                    sC_row = sC_subtile[(cta_row, None)]
                    gOut_row = cute.make_tensor(
                        cute.make_ptr(
                            self.c_dtype,
                            my_row_base + n_byte_offset,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout(epi_tile_n),
                    )
                    tCsT_row = row_thr_copy.partition_S(sC_row)
                    tCgT_row = row_thr_copy.partition_D(gOut_row)
                    if my_row_base != cutlass.Int64(0):
                        cute.copy(store_atom, tCsT_row, tCgT_row)

            self.epilog_sync_barrier.arrive_and_wait()

    @staticmethod
    def _compute_stages(
        mma_tiler_mnk,
        a_dtype,
        b_dtype,
        epi_tile,
        c_dtype,
        smem_capacity,
        occupancy,
    ):
        a_shape = cute.slice_(mma_tiler_mnk, (None, 0, None))
        b_shape = cute.slice_(mma_tiler_mnk, (0, None, None))
        ab_bytes_per_stage = (cute.size(a_shape) * a_dtype.width // 8 + cute.size(b_shape) * b_dtype.width // 8)
        epi_stage = 1
        epi_bytes = cute.size(epi_tile) * c_dtype.width // 8 * epi_stage
        num_ab_stage = (smem_capacity // occupancy - MegaMoEGroupGEMMCombineSM90.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage
        return num_ab_stage, epi_stage

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
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ):
        op = (cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
              if mcast_dim == 1 else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp())
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            op,
            tensor,
            smem_layout,
            smem_tile,
            num_multicast=mcast_dim,
        )


MegaMoEGroupGEMMCombine = MegaMoEGroupGEMMCombineSM90
