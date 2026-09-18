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
from cutlass.cutlass_dsl import Int32
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .cutedsl_utils import decode_token_src_rank_topk_and_indices
from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


class CuTeDSLDispatchGroupGemmKernelSM90:
    """Warp-specialized persistent M-contiguous grouped GEMM for Hopper.

    This keeps the same public signature and M-contiguous scheduling contract as
    the SM100 tcgen05 kernel, but uses Hopper TMA + WGMMA.  A single DMA warpgroup
    produces A/B tiles and one or two MMA warpgroups consume them, accumulate in
    registers, and store through SMEM with TMA.
    """

    reserved_smem_bytes = 1024

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        wait_signals: bool = True,
        dispatch_num_stages: int = 1,
    ):
        if dispatch_num_stages < 1 or dispatch_num_stages > 5:
            raise ValueError("dispatch_num_stages must be in [1, 5]")
        if use_2cta_instrs:
            raise ValueError("SM90 WGMMA path does not support SM100 2-CTA instructions")

        self.acc_dtype = acc_dtype
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.wait_signals = wait_signals
        self.dispatch_num_stages = dispatch_num_stages

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
        self.dispatch_g2s_warp_id = 1
        self.dispatch_s2g_warp_id = 2
        self.epi_store_warp_id = self.num_dma_warp_groups * self.num_warps_per_warp_group
        self.num_mma_threads = self.num_mma_warp_groups * self.num_threads_per_warp_group

        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")
        self.buffer_align_bytes = 1024
        self.dispatch_buffer_align_bytes = 128
        self.dispatch_max_world_size = 128
        self.dispatch_buffer_rows = self.dispatch_num_stages
        self.dispatch_bar_count = 2 * self.dispatch_num_stages

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.num_mma_threads,
        )

    def _dispatch_smem_bytes(self, problem_shape_k: int, has_weight: bool = False, world_size: int = 0) -> int:
        dtype_bytes = self.a_dtype.width // 8
        weight_ptrs_bytes = world_size * 8 if has_weight else 0
        return (self.dispatch_buffer_rows * problem_shape_k * dtype_bytes + self.dispatch_max_world_size * 8 +
                weight_ptrs_bytes + self.dispatch_bar_count * 8 + self.dispatch_buffer_align_bytes)

    def _setup_attributes(self, has_weight: bool = False, world_size: int = 0):
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
            self.smem_capacity - self._dispatch_smem_bytes(
                self.problem_shape_k,
                has_weight=bool(has_weight),
                world_size=int(world_size),
            ),
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
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        group_count: cutlass.Constexpr[int],
        problem_shape_n: cutlass.Constexpr[int],
        problem_shape_k: cutlass.Constexpr[int],
        problem_sizes_m: cute.Tensor,
        total_num_clusters: cutlass.Constexpr[int],
        max_active_clusters: cutlass.Constexpr[int],
        stream: cuda.CUstream,
        expert_signals: cute.Tensor,
        expert_signal_counters: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        input_weight_ptrs: cute.Tensor,
        output_weight: cute.Tensor,
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(mA)
        self.b_layout = utils.LayoutEnum.from_tensor(mB)
        self.c_layout = utils.LayoutEnum.from_tensor(mC)
        self.problem_shape_k = problem_shape_k

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype != cutlass.Float16 and self.a_dtype != cutlass.BFloat16):
            raise TypeError("SM90 M-contiguous grouped GEMM supports fp16/bf16 inputs")

        self._setup_attributes(
            has_weight=bool(has_weight),
            world_size=int(world_size),
        )

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

        if cutlass.const_expr(has_weight):

            @cute.struct
            class SharedStorage:
                dispatch_pipeline_bars: cute.struct.MemRange[cutlass.Int64, self.dispatch_bar_count]
                dispatch_input_ptrs: cute.struct.MemRange[cutlass.Int64, self.dispatch_max_world_size]
                dispatch_input_weight_ptrs: cute.struct.MemRange[cutlass.Int64, world_size]
                dispatch_tma_buffer: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, problem_shape_k * self.dispatch_buffer_rows],
                    self.dispatch_buffer_align_bytes,
                ]
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
        else:

            @cute.struct
            class SharedStorage:
                dispatch_pipeline_bars: cute.struct.MemRange[cutlass.Int64, self.dispatch_bar_count]
                dispatch_input_ptrs: cute.struct.MemRange[cutlass.Int64, self.dispatch_max_world_size]
                dispatch_tma_buffer: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, problem_shape_k * self.dispatch_buffer_rows],
                    self.dispatch_buffer_align_bytes,
                ]
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
            input_ptrs,
            token_src_rank_topk_and_indices,
            recv_token_count,
            mA,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
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
            expert_signals,
            expert_signal_counters,
            rank,
            world_size,
            expert_alignment,
            input_weight_ptrs,
            output_weight,
            topk,
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
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        mA_raw: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
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
        expert_signals: cute.Tensor,
        expert_signal_counters: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        input_weight_ptrs: cute.Tensor,
        output_weight: cute.Tensor,
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)

        if warp_idx == self.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

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

        lane_idx = cute.arch.lane_idx()
        dtype_bytes = cutlass.const_expr(self.a_dtype.width // 8)
        nbytes_per_token = cutlass.const_expr(problem_shape_k * dtype_bytes)
        bulk_g2s_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyBulkG2SOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        bulk_s2g_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyBulkS2GOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        dispatch_row_layout = cute.make_layout(problem_shape_k)
        dispatch_sbuf = storage.dispatch_tma_buffer.get_tensor(
            cute.make_layout(
                (self.dispatch_buffer_rows, problem_shape_k),
                stride=(problem_shape_k, 1),
            ))
        dispatch_bar_ptr = storage.dispatch_pipeline_bars.data_ptr()
        dispatch_input_ptrs = storage.dispatch_input_ptrs.get_tensor(cute.make_layout(self.dispatch_max_world_size))
        if cutlass.const_expr(has_weight):
            dispatch_input_weight_ptrs = storage.dispatch_input_weight_ptrs.get_tensor(cute.make_layout(world_size))

        dispatch_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.dispatch_num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            tx_count=nbytes_per_token,
            barrier_storage=dispatch_bar_ptr,
            tidx=lane_idx,
        )
        if tidx < cutlass.Int32(world_size):
            dispatch_input_ptrs[tidx] = input_ptrs[tidx]
            if cutlass.const_expr(has_weight):
                dispatch_input_weight_ptrs[tidx] = input_weight_ptrs[tidx]
        cute.arch.barrier()
        dispatch_producer = dispatch_pipeline.make_producer()
        dispatch_consumer_read = dispatch_pipeline.make_consumer()
        dispatch_consumer_release = dispatch_consumer_read.clone()

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

        if warp_idx == self.dispatch_g2s_warp_id or warp_idx == self.dispatch_s2g_warp_id:
            dispatch_grid_dim = cute.arch.grid_dim()
            num_blocks = cute.size(dispatch_grid_dim)
            linear_dispatch_block = (bid[2] * dispatch_grid_dim[0] * dispatch_grid_dim[1] +
                                     bid[1] * dispatch_grid_dim[0] + bid[0])
            warp_batch = cutlass.Int32(32)
            pipe_cnt_rt = cutlass.Int32(0)
            num_recv_tokens = recv_token_count[rank]
            _ = num_recv_tokens + cutlass.Int32(world_size)

            if warp_idx == self.dispatch_g2s_warp_id:
                weight_bytes: cutlass.Constexpr[int] = output_weight.element_type.width // 8
                expert_start = cutlass.Int32(0)
                expert_idx = cutlass.Int32(0)
                while expert_idx < group_count:
                    expert_tokens = problem_sizes_m[expert_idx]
                    expert_end = expert_start + expert_tokens
                    batch_base = expert_start + linear_dispatch_block

                    while batch_base < expert_end:
                        lane_token_idx = batch_base + lane_idx * num_blocks
                        encoded = cutlass.Int64(-1)
                        if lane_token_idx < expert_end:
                            encoded = token_src_rank_topk_and_indices[lane_token_idx]
                        src_token_idx, src_rank, topk_idx = decode_token_src_rank_topk_and_indices(encoded)

                        if cutlass.const_expr(has_weight):
                            if lane_token_idx < expert_end and (src_rank + cutlass.Int32(1)) != cutlass.Int32(0):
                                w_src_addr = (dispatch_input_weight_ptrs[src_rank] +
                                              cutlass.Int64(src_token_idx * cutlass.Int32(topk) + topk_idx) *
                                              cutlass.Int64(weight_bytes))
                                w_src_tensor = cute.make_tensor(
                                    cute.make_ptr(
                                        output_weight.element_type,
                                        w_src_addr,
                                        cute.AddressSpace.gmem,
                                        assumed_align=weight_bytes,
                                    ),
                                    cute.make_layout(1),
                                )
                                output_weight[lane_token_idx] = w_src_tensor[0]

                        for lane in cutlass.range(32, unroll_full=True):
                            b_token_idx = batch_base + cutlass.Int32(lane) * num_blocks
                            b_src_token_idx = cute.arch.shuffle_sync(src_token_idx, lane)
                            b_src_rank = cute.arch.shuffle_sync(src_rank, lane)

                            if b_token_idx < expert_end:
                                handle = dispatch_producer.acquire_and_advance()
                                safe_b_src_rank = cutlass.Int32(0)
                                safe_b_src_token_idx = cutlass.Int32(0)
                                if (b_src_rank + cutlass.Int32(1)) != cutlass.Int32(0):
                                    safe_b_src_rank = b_src_rank
                                    safe_b_src_token_idx = b_src_token_idx
                                remote_base_i64 = dispatch_input_ptrs[safe_b_src_rank]
                                remote_row_i64 = remote_base_i64 + (cutlass.Int64(safe_b_src_token_idx) *
                                                                    cutlass.Int64(nbytes_per_token))
                                with cute.arch.elect_one():
                                    sDst = cute.make_tensor(
                                        dispatch_sbuf.iterator + handle.index * problem_shape_k,
                                        dispatch_row_layout,
                                    )
                                    gSrc = cute.make_tensor(
                                        cute.make_ptr(
                                            self.a_dtype,
                                            remote_row_i64,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        dispatch_row_layout,
                                    )
                                    cute.copy(
                                        bulk_g2s_atom,
                                        gSrc,
                                        sDst,
                                        mbar_ptr=handle.barrier,
                                    )
                                handle.commit()
                        batch_base += num_blocks * warp_batch

                    aligned_tokens = expert_tokens
                    if cutlass.const_expr(expert_alignment > 1):
                        aligned_tokens = ((expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                    expert_start += aligned_tokens
                    expert_idx += cutlass.Int32(1)

            if warp_idx == self.dispatch_s2g_warp_id:
                expert_start = cutlass.Int32(0)
                expert_idx = cutlass.Int32(0)
                while expert_idx < group_count:
                    cnt = cutlass.Int32(0)
                    expert_tokens = problem_sizes_m[expert_idx]
                    expert_end = expert_start + expert_tokens
                    batch_base = expert_start + linear_dispatch_block

                    while batch_base < expert_end:
                        for lane in cutlass.range(32, unroll_full=True):
                            b_token_idx = batch_base + cutlass.Int32(lane) * num_blocks
                            if b_token_idx < expert_end:
                                handle = dispatch_consumer_read.wait_and_advance()
                                with cute.arch.elect_one():
                                    sSrc = cute.make_tensor(
                                        dispatch_sbuf.iterator + handle.index * problem_shape_k,
                                        dispatch_row_layout,
                                    )
                                    gDst = cute.make_tensor(
                                        cute.domain_offset((b_token_idx, cutlass.Int32(0)), mA_raw).iterator,
                                        dispatch_row_layout,
                                    )
                                    cute.copy(bulk_s2g_atom, sSrc, gDst)
                                    cute.arch.cp_async_bulk_commit_group()
                                    cute.arch.cp_async_bulk_wait_group(pipe_cnt_rt, read=True)
                                if cnt >= pipe_cnt_rt:
                                    dispatch_consumer_release.release()
                                    dispatch_consumer_release.advance()
                                cnt += cutlass.Int32(1)
                        batch_base += num_blocks * warp_batch

                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=False)

                    with cute.arch.elect_one():
                        old = cute.arch.atomic_add(
                            (expert_signal_counters.iterator + expert_idx).llvm_ptr,
                            cutlass.Int32(1),
                            sem="release",
                            scope="gpu",
                        )
                        if old + cutlass.Int32(1) == num_blocks:
                            cute.arch.store(
                                (expert_signals.iterator + expert_idx).llvm_ptr,
                                cutlass.Int32(1),
                                sem="release",
                                scope="gpu",
                            )

                    aligned_tokens = expert_tokens
                    if cutlass.const_expr(expert_alignment > 1):
                        aligned_tokens = ((expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                    expert_start += aligned_tokens
                    expert_idx += cutlass.Int32(1)

        if warp_idx == self.load_warp_id:
            if cutlass.const_expr(self.wait_signals):
                last_waited_group = Int32(-1)

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

                if cutlass.const_expr(self.wait_signals):
                    if cur_group_idx != last_waited_group:
                        sig_ptr = (expert_signals.iterator + cur_group_idx).llvm_ptr
                        with cute.arch.elect_one():
                            v = cute.arch.load(
                                sig_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="gpu",
                            )
                            while v == 0:
                                v = cute.arch.load(
                                    sig_ptr,
                                    cutlass.Int32,
                                    sem="acquire",
                                    scope="gpu",
                                )
                        cute.arch.sync_warp()
                        last_waited_group = cur_group_idx

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
        num_ab_stage = (smem_capacity // occupancy - CuTeDSLDispatchGroupGemmKernelSM90.reserved_smem_bytes -
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


CuTeDSLDispatchGroupGemmKernel = CuTeDSLDispatchGroupGemmKernelSM90
