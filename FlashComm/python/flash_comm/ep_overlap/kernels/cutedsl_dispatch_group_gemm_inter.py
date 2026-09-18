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

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cuda.bindings.driver as cuda
import nccl.core.device.cute as nccl_cute
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cutlass_dsl import (
    Int32, )
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


class _CuTeDSLDispatchGroupGemmInterKernelImpl:
    """Persistent inter-node dispatch + group GEMM (Blackwell SM100).

    10 warps / 320 threads:
      * WG0 (warps 0-3): epilogue.
      * Warp 4: NCCL GIN ring producer and dispatch TMA G2S.
      * Warps 5-7: MNNVL dispatch TMA S2G and per-expert release-store.
      * Warp 8: 2-CTA MMA (tcgen05).
      * Warp 9: GEMM TMA loads; acquires on the per-expert signal
        before issuing the first A/B load.

    Per-expert (N, K) are identical so the scheduler runs off a compact 1D
    ``problem_sizes_m`` and uses a single TMA descriptor over monolithic
    A/B/C.
    """

    reserved_smem_bytes = 1024
    dispatch_buffer_align_bytes = 128
    kMaxLocalWorldSize = 16
    inter_store_pipe_cnt = 2

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        dispatch_num_stages: int = 1,
    ):
        if dispatch_num_stages < 1 or dispatch_num_stages > 5:
            raise ValueError("dispatch_num_stages must be in [1, 5]")
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
        self.inter_producer_warp_id = 4
        self.inter_consumer_warp_id_base = 5
        self.inter_num_consumer_warps = 3
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.cta_num_warps = 10
        self.gemm_warp_id_end = self.cta_num_warps
        self.dispatch_num_stages = dispatch_num_stages
        self.dispatch_buffer_rows = max(
            self.inter_num_consumer_warps,
            ((dispatch_num_stages + self.inter_num_consumer_warps - 1) // self.inter_num_consumer_warps) *
            self.inter_num_consumer_warps,
        )
        self.dispatch_bar_count = 2 * self.dispatch_buffer_rows
        self.threads_per_cta = 32 * self.cta_num_warps

        # Named barriers must NOT include dispatch warps: epilog_sync_barrier
        # is epilogue-local; tmem_alloc_barrier joins MMA + epilogue warps.
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

    def _dispatch_smem_bytes(self, problem_shape_k: int, has_weight: bool = False, world_size: int = 0) -> int:
        _ = world_size
        dtype_bytes = self.a_dtype.width // 8
        ptr_tables = self.kMaxLocalWorldSize * 3 * 8
        if has_weight:
            ptr_tables += self.kMaxLocalWorldSize * 8
        return (self.dispatch_buffer_rows * problem_shape_k * dtype_bytes + ptr_tables + self.dispatch_bar_count * 8 +
                self.dispatch_buffer_align_bytes)

    def _setup_attributes(self, has_weight: bool = False, world_size: int = 0):
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

        self.epi_tile = utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )

        # Reserve dispatch SMEM before sizing the GEMM pipeline stages so the

        self.dispatch_smem_bytes = self._dispatch_smem_bytes(
            self.problem_shape_k,
            has_weight=bool(has_weight),
            world_size=int(world_size),
        )
        gemm_smem_capacity = self.smem_capacity - self.dispatch_smem_bytes
        (self.num_acc_stage, self.num_ab_stage, self.num_epi_stage) = (self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            gemm_smem_capacity,
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
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        recv_x_ptrs: cute.Tensor,
        recv_weight_ptrs: cute.Tensor,
        recv_topk_scatter_indices_ptrs: cute.Tensor,
        node_topk_indices: cute.Tensor,
        node_topk_send_mask: cute.Tensor,
        node_token_dst_scatter_indices: cute.Tensor,
        full_splits: cute.Tensor,
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
        expert_signal_state_ptrs: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        max_recv_tokens: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        output_weight: cute.Tensor,
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
    ):
        """Launch NCCL-GIN ring dispatch, MNNVL scatter, and FC1 UMMA."""
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)
        self.problem_shape_k = problem_shape_k

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype != cutlass.Float16 and self.a_dtype != cutlass.BFloat16):
            raise TypeError("SM100 inter-node fused dispatch+GEMM supports fp16/bf16 inputs")

        self._setup_attributes(
            has_weight=bool(has_weight),
            world_size=int(world_size),
        )

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

        dtype_bytes = self.a_dtype.width // 8
        weight_dtype = output_weight.element_type
        weight_bytes = weight_dtype.width // 8
        nbytes_per_token = problem_shape_k * dtype_bytes
        token_region_bytes = max_slot_num_token * problem_shape_k * dtype_bytes
        meta_region_bytes = max_slot_num_token * topk * 4
        weight_region_off = token_region_bytes + 3 * meta_region_bytes
        slot_payload_bytes = weight_region_off + max_slot_num_token * topk * weight_bytes
        slot_stride_bytes = ((slot_payload_bytes + 4095) // 4096) * 4096

        # The local peer tables are staged in SMEM. The weight table is
        # omitted from the weightless artifact to preserve GEMM stages.
        if cutlass.const_expr(has_weight):

            @cute.struct
            class SharedStorage:
                dispatch_pipeline_bars: cute.struct.MemRange[cutlass.Int64, self.dispatch_bar_count]
                dispatch_recv_x_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_recv_weight_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_recv_scatter_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_signal_state_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_tma_buffer: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, problem_shape_k * self.dispatch_buffer_rows],
                    self.dispatch_buffer_align_bytes,
                ]
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
        else:

            @cute.struct
            class SharedStorage:
                dispatch_pipeline_bars: cute.struct.MemRange[cutlass.Int64, self.dispatch_bar_count]
                dispatch_recv_x_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_recv_scatter_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_signal_state_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_tma_buffer: cute.struct.Align[
                    cute.struct.MemRange[self.a_dtype, problem_shape_k * self.dispatch_buffer_rows],
                    self.dispatch_buffer_align_bytes,
                ]
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
            dev_comm_ptr,
            rdma_rail_send_win_handle,
            num_tokens_per_rank,
            recv_x_ptrs,
            recv_weight_ptrs,
            recv_topk_scatter_indices_ptrs,
            node_topk_indices,
            node_topk_send_mask,
            node_token_dst_scatter_indices,
            full_splits,
            output_weight,
            mA,
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
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
            expert_signals,
            expert_signal_state_ptrs,
            rank,
            world_size,
            local_world_size,
            max_slot_num_token,
            max_recv_tokens,
            expert_alignment,
            topk,
            has_weight,
            nbytes_per_token,
            weight_bytes,
            token_region_bytes,
            weight_region_off,
            slot_stride_bytes,
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
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        recv_x_ptrs: cute.Tensor,
        recv_weight_ptrs: cute.Tensor,
        recv_topk_scatter_indices_ptrs: cute.Tensor,
        node_topk_indices: cute.Tensor,
        node_topk_send_mask: cute.Tensor,
        node_token_dst_scatter_indices: cute.Tensor,
        full_splits: cute.Tensor,
        output_weight: cute.Tensor,
        mA_raw: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mn: cute.Tensor,
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
        expert_signals: cute.Tensor,
        expert_signal_state_ptrs: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        max_recv_tokens: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        nbytes_per_token: cutlass.Constexpr[int],
        weight_bytes: cutlass.Constexpr[int],
        token_region_bytes: cutlass.Constexpr[int],
        weight_region_off: cutlass.Constexpr[int],
        slot_stride_bytes: cutlass.Constexpr[int],
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

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
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        dispatch_g2s_atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        dispatch_s2g_atom = cute.make_copy_atom(
            cpasync.CopyBulkS2GOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        dispatch_row_layout = cute.make_layout(problem_shape_k)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        dispatch_sbuf = storage.dispatch_tma_buffer.get_tensor(
            cute.make_layout(
                (
                    self.dispatch_buffer_rows,
                    problem_shape_k,
                ),
                stride=(problem_shape_k, 1),
            ))
        smem_recv_x_ptrs = storage.dispatch_recv_x_ptrs.get_tensor(cute.make_layout(self.kMaxLocalWorldSize))
        smem_recv_scatter_ptrs = storage.dispatch_recv_scatter_ptrs.get_tensor(cute.make_layout(
            self.kMaxLocalWorldSize))
        smem_signal_state_ptrs = storage.dispatch_signal_state_ptrs.get_tensor(cute.make_layout(
            self.kMaxLocalWorldSize))
        if cutlass.const_expr(has_weight):
            smem_recv_weight_ptrs = storage.dispatch_recv_weight_ptrs.get_tensor(
                cute.make_layout(self.kMaxLocalWorldSize))
        ab_full_mbar_ptr = storage.ab_full_mbar_ptr.data_ptr()
        acc_full_mbar_ptr = storage.acc_full_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf
        tmem_dealloc_mbar = storage.tmem_dealloc_mbar
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

        dispatch_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.dispatch_buffer_rows,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=1,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=1,
            ),
            tx_count=nbytes_per_token,
            barrier_storage=storage.dispatch_pipeline_bars.data_ptr(),
            tidx=lane_idx,
        )
        if tidx < cutlass.Int32(local_world_size):
            smem_recv_x_ptrs[tidx] = recv_x_ptrs[tidx]
            smem_recv_scatter_ptrs[tidx] = recv_topk_scatter_indices_ptrs[tidx]
            smem_signal_state_ptrs[tidx] = expert_signal_state_ptrs[tidx]
            if cutlass.const_expr(has_weight):
                smem_recv_weight_ptrs[tidx] = recv_weight_ptrs[tidx]
        cute.arch.barrier()

        if warp_idx < self.gemm_warp_id_end:
            ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
            num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
            ab_pipeline_consumer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                num_tma_producer,
            )

            ab_pipeline = pipeline.PipelineTmaUmma.create(
                barrier_storage=ab_full_mbar_ptr,
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
                barrier_storage=acc_full_mbar_ptr,
                num_stages=self.num_acc_stage,
                producer_group=acc_pipeline_producer_group,
                consumer_group=acc_pipeline_consumer_group,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

            tmem = utils.TmemAllocator(
                tmem_holding_buf,
                barrier_for_retrieve=self.tmem_alloc_barrier,
                allocator_warp_id=self.epilog_warp_id[0],
                is_two_cta=use_2cta_instrs,
                two_cta_tmem_dealloc_mbar_ptr=tmem_dealloc_mbar,
            )

            if warp_idx < self.gemm_warp_id_end:
                # mbarrier_init_fence + cluster arrive; must run before any
                # dispatch/GEMM main-loop barrier wait.
                pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

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

            # A/C: monolithic 2D; B: 3D with L=num_experts as a bystander dim
            # indexed by ``group_idx``.
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

            if warp_idx < self.gemm_warp_id_end:
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

            # Every CTA owns a strided token shard. Only CTA 0 injects the
            # rank's payload into the reverse GIN ring; all CTAs consume each
            # arrived node slot and scatter rows to local peers over MNNVL.
            linear_dispatch_block = (bid[2] * grid_dim[0] * grid_dim[1] + bid[1] * grid_dim[0] + bid[0])
            num_dispatch_blocks = cute.size(grid_dim)
            dev_comm = nccl_cute.DevComm(dev_comm_ptr)
            rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
            team = dev_comm.team_world
            local_rank = cutlass.Int32(rank % local_world_size)
            my_node = cutlass.Int32(rank // local_world_size)
            nnodes = cutlass.Int32(world_size // local_world_size)
            dtype_bytes = cutlass.Int32(nbytes_per_token // problem_shape_k)
            meta_elem_bytes = cutlass.Int32(4)
            gin_context_id = 0
            signal_op_inc = 0

            if warp_idx == self.inter_producer_warp_id:
                gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, gin_context_id)
                coop = nccl_cute.warp()
                dispatch_producer = dispatch_pipeline.make_producer()
                node_offset = cutlass.Int32(0)
                while node_offset < nnodes:
                    src_node = (my_node + node_offset) % nnodes
                    src_global_rank = src_node * local_world_size + local_rank
                    src_num_token = num_tokens_per_rank[src_global_rank]
                    local_num_token = num_tokens_per_rank[rank]
                    prefetch_dst_node = (my_node + nnodes - ((node_offset + cutlass.Int32(1)) % nnodes)) % nnodes

                    if linear_dispatch_block == 0 and prefetch_dst_node != my_node:
                        dst_rank = prefetch_dst_node * local_world_size + local_rank
                        src_off = cutlass.Int64(my_node) * cutlass.Int64(slot_stride_bytes)
                        token_bytes = cutlass.Int32(local_num_token * problem_shape_k * dtype_bytes)
                        meta_bytes = cutlass.Int32(local_num_token * topk * meta_elem_bytes)
                        meta_bundle_bytes = meta_bytes * cutlass.Int32(3)
                        weight_region_bytes = cutlass.Int32(local_num_token * topk * weight_bytes)
                        remaining = cutlass.Int32(0)
                        if token_bytes > 0:
                            remaining += cutlass.Int32(1)
                        if cutlass.const_expr(has_weight):
                            if weight_region_bytes > 0:
                                remaining += cutlass.Int32(1)
                        if meta_bundle_bytes > 0:
                            remaining += cutlass.Int32(1)

                        if remaining == 0:
                            gin.signal(team, dst_rank, True, my_node * cutlass.Int32(32), signal_op_inc, 1, coop)
                        else:
                            if token_bytes > 0:
                                remaining -= cutlass.Int32(1)
                                src_tensor = rdma_win.tensor(cutlass.Int8, cute.make_layout(token_bytes), src_off)
                                dst_tensor = rdma_win.tensor(cutlass.Int8, cute.make_layout(token_bytes), src_off)
                                gin.put(
                                    team,
                                    dst_rank,
                                    rdma_win,
                                    dst_tensor,
                                    rdma_win,
                                    src_tensor,
                                    coop,
                                    is_signal=(remaining == 0),
                                    signal_id=my_node * cutlass.Int32(32),
                                    signal_op=signal_op_inc,
                                    signal_op_arg=1,
                                )
                            if cutlass.const_expr(has_weight):
                                if weight_region_bytes > 0:
                                    remaining -= cutlass.Int32(1)
                                    weight_off = src_off + cutlass.Int64(weight_region_off)
                                    src_weight = rdma_win.tensor(cutlass.Int8, cute.make_layout(weight_region_bytes),
                                                                 weight_off)
                                    dst_weight = rdma_win.tensor(cutlass.Int8, cute.make_layout(weight_region_bytes),
                                                                 weight_off)
                                    gin.put(
                                        team,
                                        dst_rank,
                                        rdma_win,
                                        dst_weight,
                                        rdma_win,
                                        src_weight,
                                        coop,
                                        is_signal=(remaining == 0),
                                        signal_id=my_node * cutlass.Int32(32),
                                        signal_op=signal_op_inc,
                                        signal_op_arg=1,
                                    )
                            if meta_bundle_bytes > 0:
                                remaining -= cutlass.Int32(1)
                                meta_off = src_off + cutlass.Int64(token_region_bytes)
                                src_meta = rdma_win.tensor(cutlass.Int8, cute.make_layout(meta_bundle_bytes), meta_off)
                                dst_meta = rdma_win.tensor(cutlass.Int8, cute.make_layout(meta_bundle_bytes), meta_off)
                                gin.put(
                                    team,
                                    dst_rank,
                                    rdma_win,
                                    dst_meta,
                                    rdma_win,
                                    src_meta,
                                    coop,
                                    is_signal=(remaining == 0),
                                    signal_id=my_node * cutlass.Int32(32),
                                    signal_op=signal_op_inc,
                                    signal_op_arg=1,
                                )

                    if src_node != my_node:
                        gin.wait_signal(coop, signal=src_node * cutlass.Int32(32), least=1)
                        gin.flush(coop)
                        cute.arch.sync_warp()

                    slot_off = cutlass.Int64(src_node) * cutlass.Int64(slot_stride_bytes)
                    token_idx = linear_dispatch_block
                    while token_idx < src_num_token:
                        handle = dispatch_producer.acquire_and_advance()
                        src_row_off = (slot_off + cutlass.Int64(token_idx) * cutlass.Int64(nbytes_per_token))
                        with cute.arch.elect_one():
                            sDst = cute.make_tensor(
                                dispatch_sbuf.iterator + handle.index * problem_shape_k,
                                dispatch_row_layout,
                            )
                            gSrc = cute.make_tensor(
                                cute.make_ptr(
                                    self.a_dtype,
                                    rdma_win.local_pointer(src_row_off),
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                dispatch_row_layout,
                            )
                            cute.copy(dispatch_g2s_atom, gSrc, sDst, mbar_ptr=handle.barrier)
                        handle.commit()
                        token_idx += num_dispatch_blocks
                    node_offset += cutlass.Int32(1)
                dispatch_producer.tail()

            elif (warp_idx >= self.inter_consumer_warp_id_base
                  and warp_idx < self.inter_consumer_warp_id_base + self.inter_num_consumer_warps):
                consumer_group_id = warp_idx - cutlass.Int32(self.inter_consumer_warp_id_base)
                dispatch_consumer_read = dispatch_pipeline.make_consumer()
                dispatch_consumer_release = dispatch_consumer_read.clone()
                for group in cutlass.range(self.inter_num_consumer_warps, unroll_full=True):
                    if group < consumer_group_id:
                        dispatch_consumer_read.advance()
                        dispatch_consumer_release.advance()
                node_base_count = cutlass.Int32(0)
                node_offset = cutlass.Int32(0)
                while node_offset < nnodes:
                    src_node = (my_node + node_offset) % nnodes
                    src_global_rank = src_node * local_world_size + local_rank
                    src_num_token = num_tokens_per_rank[src_global_rank]
                    node_block_tokens = cutlass.Int32(0)
                    if linear_dispatch_block < src_num_token:
                        node_block_tokens = (
                            (src_num_token - cutlass.Int32(1) - linear_dispatch_block) // num_dispatch_blocks +
                            cutlass.Int32(1))
                    slot_off = cutlass.Int64(src_node) * cutlass.Int64(slot_stride_bytes)
                    meta_bytes = cutlass.Int64(src_num_token) * cutlass.Int64(topk) * cutlass.Int64(4)
                    topk_off = slot_off + cutlass.Int64(token_region_bytes)
                    mask_off = topk_off + meta_bytes
                    scatter_off = mask_off + meta_bytes
                    weights_off = slot_off + cutlass.Int64(weight_region_off)

                    first_node_ordinal = (consumer_group_id + cutlass.Int32(self.inter_num_consumer_warps) -
                                          (node_base_count % cutlass.Int32(self.inter_num_consumer_warps))
                                          ) % cutlass.Int32(self.inter_num_consumer_warps)
                    node_ordinal = first_node_ordinal
                    while node_ordinal < node_block_tokens:
                        token_idx = linear_dispatch_block + node_ordinal * num_dispatch_blocks
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(self.inter_store_pipe_cnt - 1, read=True)
                        cute.arch.sync_warp()

                        handle = dispatch_consumer_read.wait()

                        my_expert_idx = cutlass.Int32(-1)
                        my_target_rank = cutlass.Int32(-1)
                        my_store_idx = cutlass.Int32(-1)
                        my_weight_bits = cutlass.Int32(0)
                        if lane_idx < topk:
                            meta_idx = token_idx * topk + lane_idx
                            g_topk = cute.make_ptr(
                                node_topk_indices.element_type,
                                rdma_win.local_pointer(topk_off),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            g_mask = cute.make_ptr(
                                cutlass.Int32,
                                rdma_win.local_pointer(mask_off),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            g_scatter = cute.make_ptr(
                                node_token_dst_scatter_indices.element_type,
                                rdma_win.local_pointer(scatter_off),
                                cute.AddressSpace.gmem,
                                assumed_align=4,
                            )
                            my_expert_idx = g_topk[meta_idx]
                            my_target_rank = my_expert_idx // group_count
                            my_store_idx = g_scatter[meta_idx]
                            my_is_need_send = g_mask[meta_idx]
                            node_meta_idx = ((src_node * max_slot_num_token + token_idx) * topk + lane_idx)
                            node_topk_indices[node_meta_idx] = my_expert_idx
                            node_topk_send_mask[node_meta_idx] = my_is_need_send
                            node_token_dst_scatter_indices[node_meta_idx] = my_store_idx
                            if cutlass.const_expr(has_weight):
                                g_weight_bits = cute.make_ptr(
                                    cutlass.Int32,
                                    rdma_win.local_pointer(weights_off),
                                    cute.AddressSpace.gmem,
                                    assumed_align=4,
                                )
                                my_weight_bits = g_weight_bits[meta_idx]

                        my_target_local_rank = cutlass.Int32(-1)
                        dst_node = cutlass.Int32(-1)
                        if my_target_rank >= 0:
                            my_target_local_rank = my_target_rank % local_world_size
                            dst_node = my_target_rank // local_world_size
                        should_store = cutlass.Int32(0)
                        if (my_target_rank >= 0 and my_target_rank < world_size and my_store_idx >= 0
                                and my_store_idx < max_recv_tokens and dst_node == my_node and my_target_local_rank >= 0
                                and my_target_local_rank < local_world_size):
                            should_store = cutlass.Int32(1)

                        sSrc = cute.make_tensor(
                            dispatch_sbuf.iterator + handle.index * problem_shape_k,
                            dispatch_row_layout,
                        )
                        for send_lane in cutlass.range(topk, unroll_full=True):
                            b_should_store = cute.arch.shuffle_sync(should_store, send_lane)
                            b_target_local_rank = cute.arch.shuffle_sync(my_target_local_rank, send_lane)
                            b_target_global_rank = cute.arch.shuffle_sync(my_target_rank, send_lane)
                            b_store_idx = cute.arch.shuffle_sync(my_store_idx, send_lane)
                            b_weight_bits = cutlass.Int32(0)
                            if cutlass.const_expr(has_weight):
                                b_weight_bits = cute.arch.shuffle_sync(my_weight_bits, send_lane)
                            if b_should_store != 0:
                                dst_base_i64 = smem_recv_x_ptrs[b_target_local_rank]
                                dst_row_i64 = (dst_base_i64 +
                                               cutlass.Int64(b_store_idx) * cutlass.Int64(nbytes_per_token))
                                with cute.arch.elect_one():
                                    gDst = cute.make_tensor(
                                        cute.make_ptr(
                                            self.a_dtype,
                                            dst_row_i64,
                                            cute.AddressSpace.gmem,
                                            assumed_align=16,
                                        ),
                                        dispatch_row_layout,
                                    )
                                    cute.copy(dispatch_s2g_atom, sSrc, gDst)

                                if lane_idx < topk:
                                    scatter_base_i64 = smem_recv_scatter_ptrs[b_target_local_rank]
                                    scatter_idx = b_store_idx * topk + lane_idx
                                    out_val = cutlass.Int32(-1)
                                    if my_target_rank == b_target_global_rank:
                                        out_val = my_store_idx
                                    g_out_scatter = cute.make_ptr(
                                        node_token_dst_scatter_indices.element_type,
                                        scatter_base_i64,
                                        cute.AddressSpace.gmem,
                                        assumed_align=4,
                                    )
                                    g_out_scatter[scatter_idx] = out_val

                                if cutlass.const_expr(has_weight):
                                    with cute.arch.elect_one():
                                        weight_base_i64 = smem_recv_weight_ptrs[b_target_local_rank]
                                        weight_addr = (weight_base_i64 +
                                                       cutlass.Int64(b_store_idx) * cutlass.Int64(weight_bytes))
                                        g_weight_out = cute.make_ptr(
                                            cutlass.Int32,
                                            weight_addr,
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        g_weight_out[0] = b_weight_bits
                            cute.arch.sync_warp()

                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                        cute.arch.sync_warp()

                        for send_lane in cutlass.range(topk, unroll_full=True):
                            b_should_store = cute.arch.shuffle_sync(should_store, send_lane)
                            b_target_local_rank = cute.arch.shuffle_sync(my_target_local_rank, send_lane)
                            b_expert_idx = cute.arch.shuffle_sync(my_expert_idx, send_lane)
                            if b_should_store != 0:
                                b_local_expert_idx = b_expert_idx % group_count
                                target_base = smem_signal_state_ptrs[b_target_local_rank]
                                counter_addr = (target_base +
                                                cutlass.Int64(group_count + b_local_expert_idx) * cutlass.Int64(4))
                                counter_ptr = cute.make_ptr(
                                    cutlass.Int32,
                                    counter_addr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=4,
                                )
                                with cute.arch.elect_one():
                                    expert_total = cutlass.Int32(0)
                                    src_rank_for_count = cutlass.Int32(0)
                                    while src_rank_for_count < world_size:
                                        expert_total += full_splits[src_rank_for_count, b_expert_idx]
                                        src_rank_for_count += cutlass.Int32(1)

                                    old = cute.arch.atomic_add(
                                        counter_ptr.llvm_ptr,
                                        cutlass.Int32(1),
                                        sem="release",
                                        scope="sys",
                                    )
                                    if old + cutlass.Int32(1) == expert_total:
                                        sig_addr = (target_base + cutlass.Int64(b_local_expert_idx) * cutlass.Int64(4))
                                        sig_ptr = cute.make_ptr(
                                            cutlass.Int32,
                                            sig_addr,
                                            cute.AddressSpace.gmem,
                                            assumed_align=4,
                                        )
                                        cute.arch.store(
                                            sig_ptr.llvm_ptr,
                                            cutlass.Int32(1),
                                            sem="release",
                                            scope="sys",
                                        )
                        cute.arch.sync_warp()

                        dispatch_consumer_release.release()
                        for _ in cutlass.range(self.inter_num_consumer_warps, unroll_full=True):
                            dispatch_consumer_release.advance()
                        for _ in cutlass.range(self.inter_num_consumer_warps, unroll_full=True):
                            dispatch_consumer_read.advance()
                        node_ordinal += cutlass.Int32(self.inter_num_consumer_warps)
                    node_base_count += node_block_tokens
                    node_offset += cutlass.Int32(1)

            # ===== TMA WARP =====
            if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
                work_tile = initial_work_tile_info
                ab_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_ab_stage,
                )
                last_waited_group = Int32(-1)

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

                    # Acquire BEFORE producer_acquire so we never hold an SMEM
                    # stage idle while dispatch is still finishing this expert.
                    if cur_group_idx != last_waited_group:
                        sig_ptr = (expert_signals.iterator + cur_group_idx).llvm_ptr
                        # The last peer dispatcher publishes with a system
                        # release store after its S2G row stores complete.
                        with cute.arch.elect_one():
                            v = cute.arch.load(
                                sig_ptr,
                                cutlass.Int32,
                                sem="acquire",
                                scope="sys",
                            )
                            while v == 0:
                                v = cute.arch.load(
                                    sig_ptr,
                                    cutlass.Int32,
                                    sem="acquire",
                                    scope="sys",
                                )
                        cute.arch.sync_warp()
                        last_waited_group = cur_group_idx

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
            if (warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile):
                # Rendezvous with the epilogue allocator before TMEM retrieve.
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
            if (warp_idx >= self.epilog_warp_id[0] and warp_idx < self.epilog_warp_id[0] + len(self.epilog_warp_id)
                    and initial_work_tile_info.is_valid_tile):
                tmem.allocate(self.num_tmem_alloc_cols)
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
                tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

                epi_tidx = tidx - cutlass.Int32(self.epilog_warp_id[0] * 32)

                (tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc) = (self._epilog_tmem_copy_and_partition(
                    epi_tidx,
                    tCtAcc_base,
                    tCgC,
                    epi_tile,
                    use_2cta_instrs,
                ))

                tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
                tiled_copy_r2s, tRS_rC, tRS_sC = self._epilog_smem_copy_and_partition(
                    tiled_copy_t2r,
                    tTR_rC,
                    epi_tidx,
                    sC,
                )
                (tma_atom_c, bSG_sC,
                 bSG_gC_partitioned) = (self._epilog_gmem_copy_and_partition(tma_atom_c, tCgC, epi_tile, sC))

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
                    grouped_gemm_cta_tile_info = work_tile.group_search_result
                    cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                    m_cluster_tile_base = (tile_sched.search_state.tile_count_prev_group // ncluster_tile_n)
                    global_m_mma_tile_idx = (m_cluster_tile_base * cluster_to_mma_m +
                                             grouped_gemm_cta_tile_info.cta_tile_idx_m // cta_group_size)
                    n_tile_idx = grouped_gemm_cta_tile_info.cta_tile_idx_n

                    bSG_gC = bSG_gC_partitioned[(None, None, None, global_m_mma_tile_idx, n_tile_idx)]
                    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

                    acc_pipeline.consumer_wait(acc_consumer_state)

                    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                    num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
                    for subtile_idx in range(subtile_cnt):
                        epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_epi_stage

                        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
                        tRS_rC.store(acc_vec.to(self.c_dtype))

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, epi_buffer)],
                        )
                        # Order epilogue SMEM writes before TMA-store reads;
                        # the named barrier is epilogue-warp scoped.
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
                # In 2CTA mode, free() does a peer-CTA arrive/wait; keep it
                # after the epilogue-local barrier so no warp reads freed TMEM.
                tmem.free(tmem_ptr)
                c_pipeline.producer_tail()

    def _epilog_tmem_copy_and_partition(self, tidx, tAcc, tCgC, epi_tile, use_2cta_instrs):
        """Partition the per-tile C accumulator for TMEM->RF copy.

        ``tCgC`` is 2D (no L), one rank below the MKL example.
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

        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None)],
            epi_tile,
        )
        tTR_gC = thr_copy_t2r.partition_D(gC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0)].shape,
            self.acc_dtype,
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

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
        num_epi_stage = 2

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
        epi_bytes_per_stage = cute.size_in_bytes(c_dtype, epi_smem_layout_staged_one)
        epi_bytes = epi_bytes_per_stage * num_epi_stage

        num_ab_stage = (smem_capacity // occupancy - _CuTeDSLDispatchGroupGemmInterKernelImpl.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage

        remaining_smem = (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy *
                          (_CuTeDSLDispatchGroupGemmInterKernelImpl.reserved_smem_bytes + epi_bytes))
        num_epi_stage += remaining_smem // (occupancy * epi_bytes_per_stage)
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


class CuTeDSLDispatchGroupGemmInterKernelSM100(_CuTeDSLDispatchGroupGemmInterKernelImpl):
    """Production fused inter-node dispatch + group-GEMM kernel for SM100."""

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        *,
        dispatch_num_stages: int = 1,
    ):
        super().__init__(
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            dispatch_num_stages=dispatch_num_stages,
        )
