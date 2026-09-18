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
import nccl.core.device.cute as nccl_cute
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


def _rail_slot_stride_bytes(
    max_slot_num_token: cutlass.Constexpr[int],
    hidden_size: cutlass.Constexpr[int],
    topk: cutlass.Constexpr[int],
    dtype_bytes: cutlass.Constexpr[int],
) -> cutlass.Constexpr[int]:
    token_region_bytes = max_slot_num_token * hidden_size * dtype_bytes
    meta_region_bytes = max_slot_num_token * topk * 4
    weight_region_bytes = max_slot_num_token * topk * 4
    slot_payload_bytes = token_region_bytes + 3 * meta_region_bytes + weight_region_bytes
    return ((slot_payload_bytes + 4095) // 4096) * 4096


def _rail_weight_region_offset(
    max_slot_num_token: cutlass.Constexpr[int],
    hidden_size: cutlass.Constexpr[int],
    topk: cutlass.Constexpr[int],
    dtype_bytes: cutlass.Constexpr[int],
) -> cutlass.Constexpr[int]:
    token_region_bytes = max_slot_num_token * hidden_size * dtype_bytes
    meta_region_bytes = max_slot_num_token * topk * 4
    return token_region_bytes + 3 * meta_region_bytes


class MegaMoEGroupGEMMCombineInterSM90:
    """Single-launch Hopper FC2 WGMMA GEMM + inter-node combine.

    Correctness-first structure:
      1. persistent SM90 TMA+WGMMA grouped GEMM writes FC2 rows into a
         local-world peer-visible staging tensor;
      2. an in-kernel local-world ready wait makes all local FC2 staging
         visible before any rank pull-reads peer rows;
      3. all CTAs build per-owner-node partial rows into the RDMA rail slots;
      4. one GIN warp exchanges remote owner partials;
      5. all CTAs reduce incoming partial slots into the final dense output.

    This intentionally mirrors Kernel 3's public combine semantics, including
    source-side output weight materialization from the owner node's original
    RDMA topk/weight region.
    """

    kGridBarrierGemmDone: int = 0
    kGridBarrierPartialDone: int = 1
    kLocalGemmReadyFlag: int = 2
    kExchangeDoneFlag: int = 3
    kPipelineBarrierCount: int = 4
    kPipelineBarrierPhase: int = 5
    kBarrierWorkspaceMinElems: int = 6

    reserved_smem_bytes = 1024
    post_atom_v: int = 8

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
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_cta,
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

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        self.epi_tile = sm90_utils.compute_tile_shape_or_override(
            self.mma_tiler,
            self.c_dtype,
            is_cooperative=is_cooperative,
        )

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
        self.a_layout = utils.LayoutEnum.from_tensor(mA)
        self.b_layout = utils.LayoutEnum.from_tensor(mB)
        self.c_layout = utils.LayoutEnum.from_tensor(mC)

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype != cutlass.Float16 and self.a_dtype != cutlass.BFloat16):
            raise TypeError("SM90 inter group_gemm_combine supports fp16/bf16 inputs")
        if cutlass.const_expr(problem_shape_n % self.post_atom_v != 0):
            raise ValueError("SM90 inter group_gemm_combine requires hidden output divisible by post_atom_v")

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
        tma_atom_c, tma_tensor_c = self._make_tma_store_atoms_and_tensors(
            mC,
            self.epi_smem_layout_staged,
            self.epi_tile,
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
            local_world_size,
            max_slot_num_token,
            has_weight,
            run_reduce,
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
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        run_reduce: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        gdx, gdy, gdz = cute.arch.grid_dim()
        block_linear = (bidz * gdy + bidy) * gdx + bidx
        grid_ctas = gdx * gdy * gdz
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

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
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
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

            store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_epi_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread,
                    self.num_mma_threads,
                ),
            )
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

                gC_slice = gC_mn[(None, None, global_m_tile_idx, n_tile_idx)]
                self._epilog(
                    tma_atom_c,
                    gC_slice,
                    sC,
                    tiled_copy_r2s,
                    tRS_sC,
                    tRS_rAcc,
                    tRS_rC,
                    tRS_rC_out,
                    size_tRS_rC,
                    store_pipeline,
                    tile_sched.num_tiles_executed,
                    warp_idx,
                )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            store_pipeline.producer_tail()

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
    def _grid_barrier(
        self,
        barrier_workspace: cute.Tensor,
        barrier_idx: cutlass.Constexpr[int],
        grid_ctas: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        self.cta_sync_barrier.arrive_and_wait()
        if tidx == 0:
            ptr = (barrier_workspace.iterator + barrier_idx).llvm_ptr
            cute.arch.atomic_add(
                ptr,
                cutlass.Int32(1),
                sem="release",
                scope="sys",
            )
            v = cute.arch.load(
                ptr,
                cutlass.Int32,
                sem="acquire",
                scope="sys",
            )
            while v < grid_ctas:
                v = cute.arch.load(
                    ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="sys",
                )
        self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _publish_and_wait_local_gemm_ready(
        self,
        barrier_workspace: cute.Tensor,
        barrier_workspace_ptrs: cute.Tensor,
        block_linear: cutlass.Int32,
        tidx: cutlass.Int32,
        local_world_size: cutlass.Constexpr[int],
    ):
        if block_linear == 0 and tidx == 0:
            cute.arch.store(
                (barrier_workspace.iterator + self.kLocalGemmReadyFlag).llvm_ptr,
                cutlass.Int32(1),
                sem="release",
                scope="sys",
            )
        self.cta_sync_barrier.arrive_and_wait()
        if tidx == 0:
            peer = cutlass.Int32(0)
            while peer < local_world_size:
                peer_base = barrier_workspace_ptrs[peer]
                peer_ptr = cute.make_ptr(
                    cutlass.Int32,
                    peer_base + cutlass.Int64(self.kLocalGemmReadyFlag * 4),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                v = cute.arch.load(
                    peer_ptr.llvm_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="sys",
                )
                while v == 0:
                    v = cute.arch.load(
                        peer_ptr.llvm_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="sys",
                    )
                peer += cutlass.Int32(1)
        self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _wait_local_flag(
        self,
        barrier_workspace: cute.Tensor,
        flag_idx: cutlass.Constexpr[int],
        tidx: cutlass.Int32,
    ):
        if tidx == 0:
            ptr = (barrier_workspace.iterator + flag_idx).llvm_ptr
            v = cute.arch.load(
                ptr,
                cutlass.Int32,
                sem="acquire",
                scope="sys",
            )
            while v == 0:
                v = cute.arch.load(
                    ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="sys",
                )
        self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _grid_barrier_phase(
        self,
        barrier_workspace: cute.Tensor,
        phase: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        tidx: cutlass.Int32,
    ):
        self.cta_sync_barrier.arrive_and_wait()
        if tidx == 0:
            count_ptr = (barrier_workspace.iterator + self.kPipelineBarrierCount).llvm_ptr
            phase_ptr = (barrier_workspace.iterator + self.kPipelineBarrierPhase).llvm_ptr
            old = cute.arch.atomic_add(
                count_ptr,
                cutlass.Int32(1),
                sem="release",
                scope="sys",
            )
            if old + cutlass.Int32(1) == grid_ctas:
                cute.arch.store(
                    count_ptr,
                    cutlass.Int32(0),
                    sem="release",
                    scope="sys",
                )
                cute.arch.store(
                    phase_ptr,
                    phase,
                    sem="release",
                    scope="sys",
                )
            else:
                v = cute.arch.load(
                    phase_ptr,
                    cutlass.Int32,
                    sem="acquire",
                    scope="sys",
                )
                while v < phase:
                    v = cute.arch.load(
                        phase_ptr,
                        cutlass.Int32,
                        sem="acquire",
                        scope="sys",
                    )
        self.cta_sync_barrier.arrive_and_wait()

    @cute.jit
    def _pipeline_inter_partials_and_exchange(
        self,
        barrier_workspace: cute.Tensor,
        fc2_output: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_topk_send_mask: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
        block_linear: cutlass.Int32,
        grid_ctas: cutlass.Int32,
        tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
    ):
        _ = local_topk_send_mask
        my_node = cutlass.Int32(rank // local_world_size)
        nnodes = cutlass.Int32(world_size // local_world_size)
        prev_owner_node = cutlass.Int32(-1)
        node_iter = cutlass.Int32(0)
        while node_iter < nnodes:
            owner_node = (my_node + cutlass.Int32(1) + node_iter) % nnodes
            if prev_owner_node >= cutlass.Int32(0):
                self._exchange_owner_partial(
                    dev_comm_ptr,
                    rdma_rail_send_win_handle,
                    num_tokens_per_rank,
                    prev_owner_node,
                    block_linear,
                    warp_idx,
                    hidden_size,
                    topk,
                    rank,
                    world_size,
                    local_world_size,
                    max_slot_num_token,
                )
            self._build_inter_partial_owner(
                fc2_output,
                combine_x_ptrs,
                rdma_rail_send_win_handle,
                num_tokens_per_rank,
                local_topk_indices,
                local_token_dst_scatter_indices,
                owner_node,
                block_linear,
                grid_ctas,
                tidx,
                hidden_size,
                topk,
                rank,
                world_size,
                local_world_size,
                max_slot_num_token,
                num_experts_per_rank,
            )
            self._grid_barrier_phase(
                barrier_workspace,
                node_iter + cutlass.Int32(1),
                grid_ctas,
                tidx,
            )
            prev_owner_node = owner_node
            node_iter += cutlass.Int32(1)

        if prev_owner_node >= cutlass.Int32(0):
            self._exchange_owner_partial(
                dev_comm_ptr,
                rdma_rail_send_win_handle,
                num_tokens_per_rank,
                prev_owner_node,
                block_linear,
                warp_idx,
                hidden_size,
                topk,
                rank,
                world_size,
                local_world_size,
                max_slot_num_token,
            )
        self._wait_exchange_partials(
            barrier_workspace,
            dev_comm_ptr,
            rank,
            world_size,
            local_world_size,
            block_linear,
            warp_idx,
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
        _ = fc2_output
        dtype = self.c_dtype
        dtype_bytes: cutlass.Constexpr[int] = dtype.width // 8
        atom_v: cutlass.Constexpr[int] = self.post_atom_v
        hidden_vecs: cutlass.Constexpr[int] = hidden_size // atom_v
        row_byte_stride: cutlass.Constexpr[int] = hidden_size * dtype_bytes
        slot_stride_bytes: cutlass.Constexpr[int] = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        atom_layout = cute.make_layout(atom_v)
        tCrSrc = cute.make_fragment((atom_v, ), dtype)
        tCrOut = cute.make_fragment((atom_v, ), dtype)
        tCrAcc = cute.make_fragment((atom_v, ), cutlass.Float32)
        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)

        nnodes = cutlass.Int32(world_size // local_world_size)
        my_node = cutlass.Int32(rank // local_world_size)
        local_rank = cutlass.Int32(rank % local_world_size)
        my_node_rank_begin = my_node * cutlass.Int32(local_world_size)
        my_node_rank_end = my_node_rank_begin + cutlass.Int32(local_world_size)
        src_rank = owner_node * cutlass.Int32(local_world_size) + local_rank
        src_num_token = num_tokens_per_rank[src_rank]

        global_thread = block_linear * cutlass.Int32(self.threads_per_cta) + tidx
        total_threads = grid_ctas * cutlass.Int32(self.threads_per_cta)
        worker_id = global_thread
        worker_count = total_threads
        if block_linear == cutlass.Int32(0) and tidx < cutlass.Int32(32):
            worker_id = cutlass.Int32(-1)
        else:
            worker_id = global_thread - cutlass.Int32(32)
            worker_count = total_threads - cutlass.Int32(32)

        total_work = cutlass.Int32(max_slot_num_token * hidden_vecs)
        linear = worker_id
        while linear >= cutlass.Int32(0) and linear < total_work:
            vec_idx = linear % cutlass.Int32(hidden_vecs)
            token_offset = linear // cutlass.Int32(hidden_vecs)
            if token_offset < src_num_token:
                tCrAcc.fill(0.0)
                atom_byte_offset = cutlass.Int64(vec_idx * atom_v) * cutlass.Int64(dtype_bytes)
                for k in cutlass.range_constexpr(topk):
                    expert_idx = local_topk_indices[owner_node, token_offset, k]
                    expert_rank = expert_idx // cutlass.Int32(num_experts_per_rank)
                    scatter = local_token_dst_scatter_indices[owner_node, token_offset, k]
                    is_valid = (expert_rank >= my_node_rank_begin and expert_rank < my_node_rank_end and scatter >= 0)
                    if is_valid:
                        expert_local_rank = expert_rank - my_node_rank_begin
                        src_base = combine_x_ptrs[expert_local_rank]
                        src_addr = (src_base + cutlass.Int64(scatter) * cutlass.Int64(row_byte_stride) +
                                    atom_byte_offset)
                        gSrc = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                src_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            atom_layout,
                        )
                        cute.copy(load_atom, gSrc, tCrSrc)
                        tCrAcc.store(tCrAcc.load() + tCrSrc.load().to(cutlass.Float32))

                tCrOut.store(tCrAcc.load().to(dtype))
                partial_slot = nnodes + owner_node
                if owner_node == my_node:
                    partial_slot = nnodes * cutlass.Int32(2) + my_node
                dst_off = (cutlass.Int64(partial_slot) * cutlass.Int64(slot_stride_bytes) +
                           cutlass.Int64(token_offset) * cutlass.Int64(row_byte_stride) + atom_byte_offset)
                gDst = cute.make_tensor(
                    cute.make_ptr(
                        dtype,
                        rdma_win.local_pointer(dst_off),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    atom_layout,
                )
                cute.copy(store_atom, tCrOut, gDst)
            linear += worker_count

    @cute.jit
    def _exchange_owner_partial(
        self,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        owner_node: cutlass.Int32,
        block_linear: cutlass.Int32,
        warp_idx: cutlass.Int32,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
    ):
        dtype_bytes: cutlass.Constexpr[int] = self.c_dtype.width // 8
        slot_stride_bytes: cutlass.Constexpr[int] = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        my_node = cutlass.Int32(rank // local_world_size)
        if block_linear == 0 and warp_idx == 0 and owner_node != my_node:
            dev_comm = nccl_cute.DevComm(dev_comm_ptr)
            rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
            team = dev_comm.team_world
            gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
            coop = nccl_cute.warp()
            local_rank = cutlass.Int32(rank % local_world_size)
            nnodes = cutlass.Int32(world_size // local_world_size)
            signal_op_inc = 0

            src_rank = owner_node * cutlass.Int32(local_world_size) + local_rank
            src_num_token = num_tokens_per_rank[src_rank]
            peer = owner_node * cutlass.Int32(local_world_size) + local_rank
            src_slot = nnodes + owner_node
            dst_slot = nnodes * cutlass.Int32(2) + my_node
            src_off = cutlass.Int64(src_slot) * cutlass.Int64(slot_stride_bytes)
            dst_off = cutlass.Int64(dst_slot) * cutlass.Int64(slot_stride_bytes)
            token_bytes = src_num_token * cutlass.Int32(hidden_size * dtype_bytes)
            signal_id = nnodes * cutlass.Int32(32) + my_node
            if token_bytes > 0:
                src_tensor = rdma_win.tensor(cutlass.Int8, cute.make_layout(token_bytes), src_off)
                dst_tensor = rdma_win.tensor(cutlass.Int8, cute.make_layout(token_bytes), dst_off)
                gin.put(
                    team,
                    peer,
                    rdma_win,
                    dst_tensor,
                    rdma_win,
                    src_tensor,
                    coop,
                    is_signal=True,
                    signal_id=signal_id,
                    signal_op=signal_op_inc,
                    signal_op_arg=1,
                )
            else:
                gin.signal(
                    team,
                    peer,
                    True,
                    signal_id,
                    signal_op_inc,
                    1,
                    coop,
                )

    @cute.jit
    def _wait_exchange_partials(
        self,
        barrier_workspace: cute.Tensor,
        dev_comm_ptr: cutlass.Int64,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        block_linear: cutlass.Int32,
        warp_idx: cutlass.Int32,
    ):
        if block_linear == 0 and warp_idx == 0:
            dev_comm = nccl_cute.DevComm(dev_comm_ptr)
            gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
            coop = nccl_cute.warp()
            my_node = cutlass.Int32(rank // local_world_size)
            nnodes = cutlass.Int32(world_size // local_world_size)
            node = cutlass.Int32(0)
            while node < nnodes:
                if node != my_node:
                    gin.wait_signal(coop, signal=nnodes * cutlass.Int32(32) + node, least=1)
                    gin.flush(coop)
                node += cutlass.Int32(1)

            with cute.arch.elect_one():
                cute.arch.store(
                    (barrier_workspace.iterator + self.kExchangeDoneFlag).llvm_ptr,
                    cutlass.Int32(1),
                    sem="release",
                    scope="sys",
                )

    @cute.jit
    def _build_inter_partials(
        self,
        fc2_output: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_topk_send_mask: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
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
        _ = fc2_output
        _ = local_topk_send_mask
        dtype = self.c_dtype
        dtype_bytes: cutlass.Constexpr[int] = dtype.width // 8
        atom_v: cutlass.Constexpr[int] = self.post_atom_v
        hidden_vecs: cutlass.Constexpr[int] = hidden_size // atom_v
        row_byte_stride: cutlass.Constexpr[int] = hidden_size * dtype_bytes
        slot_stride_bytes: cutlass.Constexpr[int] = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        atom_layout = cute.make_layout(atom_v)
        tCrSrc = cute.make_fragment((atom_v, ), dtype)
        tCrOut = cute.make_fragment((atom_v, ), dtype)
        tCrAcc = cute.make_fragment((atom_v, ), cutlass.Float32)
        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)

        nnodes = cutlass.Int32(world_size // local_world_size)
        my_node = cutlass.Int32(rank // local_world_size)
        local_rank = cutlass.Int32(rank % local_world_size)
        my_node_rank_begin = my_node * cutlass.Int32(local_world_size)
        my_node_rank_end = my_node_rank_begin + cutlass.Int32(local_world_size)
        linear = block_linear * cutlass.Int32(self.threads_per_cta) + tidx
        stride = grid_ctas * cutlass.Int32(self.threads_per_cta)
        total_work = nnodes * cutlass.Int32(max_slot_num_token * hidden_vecs)
        while linear < total_work:
            vec_idx = linear % cutlass.Int32(hidden_vecs)
            token_linear = linear // cutlass.Int32(hidden_vecs)
            token_offset = token_linear % cutlass.Int32(max_slot_num_token)
            owner_node = token_linear // cutlass.Int32(max_slot_num_token)
            src_rank = owner_node * cutlass.Int32(local_world_size) + local_rank
            src_num_token = num_tokens_per_rank[src_rank]
            if token_offset < src_num_token:
                tCrAcc.fill(0.0)
                atom_byte_offset = cutlass.Int64(vec_idx * atom_v) * cutlass.Int64(dtype_bytes)
                for k in cutlass.range_constexpr(topk):
                    expert_idx = local_topk_indices[owner_node, token_offset, k]
                    expert_rank = expert_idx // cutlass.Int32(num_experts_per_rank)
                    scatter = local_token_dst_scatter_indices[owner_node, token_offset, k]
                    is_valid = (expert_rank >= my_node_rank_begin and expert_rank < my_node_rank_end and scatter >= 0)
                    if is_valid:
                        expert_local_rank = expert_rank - my_node_rank_begin
                        src_base = combine_x_ptrs[expert_local_rank]
                        src_addr = (src_base + cutlass.Int64(scatter) * cutlass.Int64(row_byte_stride) +
                                    atom_byte_offset)
                        gSrc = cute.make_tensor(
                            cute.make_ptr(
                                dtype,
                                src_addr,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            atom_layout,
                        )
                        cute.copy(load_atom, gSrc, tCrSrc)
                        tCrAcc.store(tCrAcc.load() + tCrSrc.load().to(cutlass.Float32))

                tCrOut.store(tCrAcc.load().to(dtype))
                partial_slot = nnodes + owner_node
                if owner_node == my_node:
                    partial_slot = nnodes * cutlass.Int32(2) + my_node
                dst_off = (cutlass.Int64(partial_slot) * cutlass.Int64(slot_stride_bytes) +
                           cutlass.Int64(token_offset) * cutlass.Int64(row_byte_stride) + atom_byte_offset)
                gDst = cute.make_tensor(
                    cute.make_ptr(
                        dtype,
                        rdma_win.local_pointer(dst_off),
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    atom_layout,
                )
                cute.copy(store_atom, tCrOut, gDst)
            linear += stride

    @cute.jit
    def _exchange_partials(
        self,
        barrier_workspace: cute.Tensor,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        block_linear: cutlass.Int32,
        warp_idx: cutlass.Int32,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
    ):
        dtype_bytes: cutlass.Constexpr[int] = self.c_dtype.width // 8
        slot_stride_bytes: cutlass.Constexpr[int] = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        if block_linear == 0 and warp_idx == 0:
            dev_comm = nccl_cute.DevComm(dev_comm_ptr)
            rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
            team = dev_comm.team_world
            gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
            coop = nccl_cute.warp()
            local_rank = cutlass.Int32(rank % local_world_size)
            my_node = cutlass.Int32(rank // local_world_size)
            nnodes = cutlass.Int32(world_size // local_world_size)
            signal_op_inc = 0

            owner_node = cutlass.Int32(0)
            while owner_node < nnodes:
                if owner_node != my_node:
                    src_rank = owner_node * cutlass.Int32(local_world_size) + local_rank
                    src_num_token = num_tokens_per_rank[src_rank]
                    peer = owner_node * cutlass.Int32(local_world_size) + local_rank
                    src_slot = nnodes + owner_node
                    dst_slot = nnodes * cutlass.Int32(2) + my_node
                    src_off = cutlass.Int64(src_slot) * cutlass.Int64(slot_stride_bytes)
                    dst_off = cutlass.Int64(dst_slot) * cutlass.Int64(slot_stride_bytes)
                    token_bytes = src_num_token * cutlass.Int32(hidden_size * dtype_bytes)
                    signal_id = nnodes * cutlass.Int32(32) + my_node
                    if token_bytes > 0:
                        src_tensor = rdma_win.tensor(cutlass.Int8, cute.make_layout(token_bytes), src_off)
                        dst_tensor = rdma_win.tensor(cutlass.Int8, cute.make_layout(token_bytes), dst_off)
                        gin.put(
                            team,
                            peer,
                            rdma_win,
                            dst_tensor,
                            rdma_win,
                            src_tensor,
                            coop,
                            is_signal=True,
                            signal_id=signal_id,
                            signal_op=signal_op_inc,
                            signal_op_arg=1,
                        )
                    else:
                        gin.signal(
                            team,
                            peer,
                            True,
                            signal_id,
                            signal_op_inc,
                            1,
                            coop,
                        )
                owner_node += cutlass.Int32(1)

            node = cutlass.Int32(0)
            while node < nnodes:
                if node != my_node:
                    gin.wait_signal(coop, signal=nnodes * cutlass.Int32(32) + node, least=1)
                    gin.flush(coop)
                node += cutlass.Int32(1)

            with cute.arch.elect_one():
                cute.arch.store(
                    (barrier_workspace.iterator + self.kExchangeDoneFlag).llvm_ptr,
                    cutlass.Int32(1),
                    sem="release",
                    scope="sys",
                )

    @cute.jit
    def _reduce_inter_partials(
        self,
        output: cute.Tensor,
        output_weight: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
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
        has_weight: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
    ):
        dtype = self.c_dtype
        dtype_bytes: cutlass.Constexpr[int] = dtype.width // 8
        atom_v: cutlass.Constexpr[int] = self.post_atom_v
        hidden_vecs: cutlass.Constexpr[int] = hidden_size // atom_v
        row_byte_stride: cutlass.Constexpr[int] = hidden_size * dtype_bytes
        token_region_bytes: cutlass.Constexpr[int] = max_slot_num_token * hidden_size * dtype_bytes
        slot_stride_bytes: cutlass.Constexpr[int] = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        weight_region_off: cutlass.Constexpr[int] = _rail_weight_region_offset(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        load_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            dtype,
            num_bits_per_copy=128,
        )
        atom_layout = cute.make_layout(atom_v)
        tCrSrc = cute.make_fragment((atom_v, ), dtype)
        tCrOut = cute.make_fragment((atom_v, ), dtype)
        tCrAcc = cute.make_fragment((atom_v, ), cutlass.Float32)
        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
        nnodes = cutlass.Int32(world_size // local_world_size)
        my_node = cutlass.Int32(rank // local_world_size)
        token_vec_count = num_tokens * cutlass.Int32(hidden_vecs)
        weight_count = cutlass.Int32(0)
        if cutlass.const_expr(has_weight):
            weight_count = num_tokens * cutlass.Int32(topk)
        total_work = token_vec_count
        if weight_count > total_work:
            total_work = weight_count

        gOut = cute.zipped_divide(output, (1, atom_v))
        linear = block_linear * cutlass.Int32(self.threads_per_cta) + tidx
        stride = grid_ctas * cutlass.Int32(self.threads_per_cta)
        while linear < total_work:
            if linear < token_vec_count:
                token_idx = linear // cutlass.Int32(hidden_vecs)
                vec_idx = linear - token_idx * cutlass.Int32(hidden_vecs)
                atom_byte_offset = cutlass.Int64(vec_idx * atom_v) * cutlass.Int64(dtype_bytes)
                tCrAcc.fill(0.0)
                node = cutlass.Int32(0)
                while node < nnodes:
                    slot = nnodes * cutlass.Int32(2) + node
                    src_off = (cutlass.Int64(slot) * cutlass.Int64(slot_stride_bytes) +
                               cutlass.Int64(token_idx) * cutlass.Int64(row_byte_stride) + atom_byte_offset)
                    gSrc = cute.make_tensor(
                        cute.make_ptr(
                            dtype,
                            rdma_win.local_pointer(src_off),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        atom_layout,
                    )
                    cute.copy(load_atom, gSrc, tCrSrc)
                    tCrAcc.store(tCrAcc.load() + tCrSrc.load().to(cutlass.Float32))
                    node += cutlass.Int32(1)
                tCrOut.store(tCrAcc.load().to(dtype))
                cute.copy(
                    store_atom,
                    tCrOut,
                    gOut[(0, None), (token_idx, vec_idx)],
                )

            if cutlass.const_expr(has_weight):
                if linear < weight_count:
                    token_idx_w = linear // cutlass.Int32(topk)
                    topk_idx = linear - token_idx_w * cutlass.Int32(topk)
                    weight_offset = cutlass.Int64(token_idx_w * cutlass.Int32(topk) + topk_idx) * cutlass.Int64(4)
                    slot_off = cutlass.Int64(my_node) * cutlass.Int64(slot_stride_bytes)
                    gTopk = cute.make_ptr(
                        cutlass.Int32,
                        rdma_win.local_pointer(slot_off + cutlass.Int64(token_region_bytes) + weight_offset),
                        cute.AddressSpace.gmem,
                        assumed_align=4,
                    )
                    expert_idx = gTopk[0]
                    weight_value = cutlass.Float32(0.0)
                    if expert_idx < cutlass.Int32(world_size * num_experts_per_rank):
                        gSrcWeight = cute.make_ptr(
                            cutlass.Float32,
                            rdma_win.local_pointer(slot_off + cutlass.Int64(weight_region_off) + weight_offset),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        weight_value = gSrcWeight[0]
                    output_weight[token_idx_w, topk_idx] = weight_value
            linear += stride

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
        tma_atom_c,
        gC_slice,
        sC,
        tiled_copy_r2s,
        tRS_sC,
        tRS_rAcc,
        tRS_rC,
        tRS_rC_out,
        size_tRS_rC,
        store_pipeline,
        num_tiles_executed,
        warp_idx,
    ):
        tCgC_for_tma = cute.zipped_divide(gC_slice, self.epi_tile)
        bSG_sC, bSG_gC = cute.nvgpu.cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            tCgC_for_tma,
        )

        epi_tile_num = cute.size(tCgC_for_tma, mode=[1])
        epi_tile_shape = tCgC_for_tma.shape[1]
        epi_tile_layout = cute.make_layout(epi_tile_shape, stride=(epi_tile_shape[1], 1))
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
                gmem_coord = epi_tile_layout.get_hier_coord(epi_idx)
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, epi_buffer)],
                    bSG_gC[(None, gmem_coord)],
                )
                store_pipeline.producer_commit()
                store_pipeline.producer_acquire()

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
        epi_stage = 4
        epi_bytes = cute.size(epi_tile) * c_dtype.width // 8 * epi_stage
        num_ab_stage = (smem_capacity // occupancy - MegaMoEGroupGEMMCombineInterSM90.reserved_smem_bytes -
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

    @staticmethod
    def _make_tma_store_atoms_and_tensors(
        tensor_c: cute.Tensor,
        epi_smem_layout_staged: cute.ComposedLayout,
        epi_tile: tuple[int, int],
    ):
        epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
        return cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            tensor_c,
            epi_smem_layout,
            epi_tile,
        )


MegaMoEGroupGEMMCombineInter = MegaMoEGroupGEMMCombineInterSM90
