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
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from .cutedsl_utils import decode_token_src_rank_topk_and_indices
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cutlass_dsl import (
    Int32, )
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


class _CuTeDSLDispatchGroupGemmKernelImpl:
    """Persistent M-contiguous fused dispatch + group GEMM (Blackwell SM100).

    8 warps / 256 threads / 2 WGs, no idle warps:
      * WG0 (warps 0-3): epilogue.
      * WG1 (warps 4-5): pull dispatch (TMA G2S+S2G) and per-expert
        release-store.
      * WG1 warp 6:      MMA (tcgen05).
      * WG1 warp 7:      GEMM TMA loads; acquires on the per-expert signal
        before issuing the first A/B load.

    Per-expert (N, K) are identical so the scheduler runs off a compact 1D
    ``problem_sizes_m`` and uses a single TMA descriptor over monolithic
    A/B/C.
    """

    reserved_smem_bytes = 1024
    dispatch_buffer_align_bytes = 128

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
        self.dispatch_warp_id_base = 4
        self.num_dispatch_warps = 2
        self.mma_warp_id = 6
        self.tma_warp_id = 7
        self.cta_num_warps = 8
        self.gemm_warp_id_end = self.cta_num_warps
        self.dispatch_num_stages = dispatch_num_stages
        self.dispatch_max_world_size = 128
        self.dispatch_buffer_rows = self.dispatch_num_stages
        self.dispatch_bar_count = 2 * self.dispatch_num_stages
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
        # ``dispatch_input_weight_ptrs`` is only laid out for the
        # has_weight=True cache entry and is sized to ``world_size`` (not
        # the legacy ``dispatch_max_world_size`` upper bound) -- intranode
        # workloads have <= 8 ranks so the added SMEM is at most 64 B,
        # which stays well below the boundary that would force the GEMM
        # scheduler to drop an A/B stage. The weightless build keeps a
        # byte-identical SMEM layout and the same pipeline depth.
        dtype_bytes = self.a_dtype.width // 8
        weight_ptrs_bytes = world_size * 8 if has_weight else 0
        return (self.dispatch_buffer_rows * problem_shape_k * dtype_bytes + self.dispatch_max_world_size * 8 +
                weight_ptrs_bytes + self.dispatch_bar_count * 8 + self.dispatch_buffer_align_bytes)

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
        """Launch fused pull dispatch + grouped GEMM.

        ``problem_sizes_m`` carries real per-expert token counts; the host
        pads A/C rows to ``cluster_tile_m``. ``expert_signals[i]`` is
        release-stored by dispatch when all rows for expert ``i`` are visible
        in A; the GEMM TMA warp acquires on it before its first load.

        When ``has_weight`` is set, the dispatch producer warp
        (``dispatch_warp_id_base``) issues a single 4 B
        ``ld.global / st.global`` per dispatched row in lockstep with the
        token-row G2S issue. Each of the 32 lanes serves its own row in
        parallel (no ``elect_one``) so the extra traffic overlaps the
        elected lane's TMA issue and stays off the critical path.
        ``output_weight`` is a local 1D tensor (one scalar per dispatched
        row, dtype = caller's weight dtype, 4 B). ``input_weight_ptrs`` is
        a 1D ``(world_size,)`` Int64 view of peer ``(num_tokens, topk)``
        symmetric weight matrices.
        """
        self.a_dtype = mA.element_type
        self.b_dtype = mB.element_type
        self.c_dtype = mC.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(mC)
        self.problem_shape_k = problem_shape_k

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self.dispatch_pipe_cnt = 0
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

        # Two SharedStorage layouts: the weight ptr table is only
        # included for the has_weight=True artifact AND is sized to
        # ``world_size`` (not the dispatch_max_world_size upper bound)
        # so it stays well below 1 KB even at world_size=128 and -- in
        # the practical intranode case (world_size <= 8) -- adds only
        # 64 B of SMEM, well below the threshold that would force the
        # GEMM scheduler to drop an A/B stage.
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
                dispatch_input_ptrs: cute.struct.MemRange[cutlass.Int64, self.dispatch_max_world_size]
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
            input_ptrs,
            token_src_rank_topk_and_indices,
            recv_token_count,
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
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
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
        expert_signal_counters: cute.Tensor,
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        input_weight_ptrs: cute.Tensor,
        output_weight: cute.Tensor,
        topk: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
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
        dtype_bytes = cutlass.const_expr(self.a_dtype.width // 8)
        nbytes_per_token = cutlass.const_expr(problem_shape_k * dtype_bytes)
        nbytes_per_dispatch_g2s = nbytes_per_token

        bulk_g2s_atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            self.a_dtype,
            num_bits_per_copy=nbytes_per_dispatch_g2s * 8,
        )
        bulk_s2g_atom = cute.make_copy_atom(
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
        dispatch_bar_ptr = storage.dispatch_pipeline_bars.data_ptr()
        dispatch_input_ptrs = storage.dispatch_input_ptrs.get_tensor(cute.make_layout(self.dispatch_max_world_size))
        # ``dispatch_input_weight_ptrs`` is only laid out in the
        # has_weight SharedStorage variant; the no-weight build doesn't
        # carry it so the SMEM layout matches the legacy kernel exactly.
        # Sized to ``world_size`` (the smallest constexpr that fits all
        # peers) so the SMEM cost stays in the noise.
        if cutlass.const_expr(has_weight):
            dispatch_input_weight_ptrs = (storage.dispatch_input_weight_ptrs.get_tensor(cute.make_layout(world_size)))
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
            num_stages=self.dispatch_num_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=1,
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                size=1,
            ),
            tx_count=nbytes_per_dispatch_g2s,
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

            if (warp_idx >= self.dispatch_warp_id_base
                    and warp_idx < (self.dispatch_warp_id_base + self.num_dispatch_warps)):
                dispatch_grid_dim = cute.arch.grid_dim()
                dispatch_warp_rank = warp_idx - self.dispatch_warp_id_base
                # Persistent-GEMM grid is 3D (cluster_m, cluster_n, clusters);
                # flatten so each CTA owns a unique dispatch shard.
                num_blocks = cute.size(dispatch_grid_dim)
                linear_dispatch_block = (bid[2] * dispatch_grid_dim[0] * dispatch_grid_dim[1] +
                                         bid[1] * dispatch_grid_dim[0] + bid[0])
                pipe_cnt_rt = cutlass.Int32(self.dispatch_pipe_cnt)
                warp_batch = cutlass.Int32(32)
                num_recv_tokens = recv_token_count[rank]
                _ = num_recv_tokens + cutlass.Int32(world_size)

                if dispatch_warp_rank == cutlass.Int32(0):

                    weight_bytes: cutlass.Constexpr[int] = (output_weight.element_type.width // 8)
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
                            src_token_idx, src_rank, topk_idx = (decode_token_src_rank_topk_and_indices(encoded))

                            # Side-load: each of the 32 lanes pulls its
                            # own (token, topk_idx) weight in parallel
                            # from the source rank's symmetric weight
                            # buffer into the local ``output_weight``
                            # row.
                            if cutlass.const_expr(has_weight):
                                if (lane_token_idx < expert_end and (src_rank + cutlass.Int32(1)) != cutlass.Int32(0)):
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
                                    output_weight[lane_token_idx] = (w_src_tensor[0])

                            for lane in cutlass.range(32, unroll_full=True):
                                b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)
                                b_src_token_idx = cute.arch.shuffle_sync(src_token_idx, lane)
                                b_src_rank = cute.arch.shuffle_sync(src_rank, lane)

                                if b_token_idx < expert_end:
                                    # acquire_and_advance blocks on the empty
                                    # mbarrier; the G2S TMA arrives the matching
                                    # full transaction barrier on completion.
                                    handle = (dispatch_producer.acquire_and_advance())
                                    # ``b_src_rank == -1`` should be excluded
                                    # by ``problem_sizes_m`` for valid rows,
                                    # but keep the ptr-table load in-bounds if
                                    # bad metadata reaches this path.
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
                            aligned_tokens = (
                                (expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                        expert_start += aligned_tokens
                        expert_idx += cutlass.Int32(1)

                elif dispatch_warp_rank == cutlass.Int32(1):
                    expert_start = cutlass.Int32(0)
                    expert_idx = cutlass.Int32(0)
                    while expert_idx < group_count:
                        cnt = cutlass.Int32(0)
                        expert_tokens = problem_sizes_m[expert_idx]
                        expert_end = expert_start + expert_tokens
                        batch_base = expert_start + linear_dispatch_block

                        while batch_base < expert_end:
                            for lane in cutlass.range(32, unroll_full=True):
                                b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)

                                if b_token_idx < expert_end:
                                    # wait_and_advance blocks on the full mbarrier
                                    # set by the G2S TMA; the cloned consumer
                                    # handle releases the stage once S2G is done.
                                    handle = (dispatch_consumer_read.wait_and_advance())
                                    with cute.arch.elect_one():
                                        sSrc = cute.make_tensor(
                                            dispatch_sbuf.iterator + handle.index * problem_shape_k,
                                            dispatch_row_layout,
                                        )
                                        gDst = cute.make_tensor(
                                            cute.domain_offset(
                                                (b_token_idx, cutlass.Int32(0)),
                                                mA_raw,
                                            ).iterator,
                                            dispatch_row_layout,
                                        )
                                        cute.copy(bulk_s2g_atom, sSrc, gDst)
                                        cute.arch.cp_async_bulk_commit_group()
                                        cute.arch.cp_async_bulk_wait_group(
                                            self.dispatch_pipe_cnt,
                                            read=True,
                                        )
                                    if cnt >= pipe_cnt_rt:
                                        dispatch_consumer_release.release()
                                        dispatch_consumer_release.advance()

                                    cnt += cutlass.Int32(1)
                            batch_base += num_blocks * warp_batch

                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(0, read=False)

                        for _i in cutlass.range(self.dispatch_pipe_cnt, unroll_full=True):
                            if cnt > cutlass.Int32(0):
                                dispatch_consumer_release.release()
                                dispatch_consumer_release.advance()
                                cnt -= cutlass.Int32(1)

                        with cute.arch.elect_one():
                            # Per-expert readiness: each CTA bumps the counter;
                            # the last one release-stores the signal. Orders
                            # memory for GEMM acquire, no CTA-wide sync.
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
                            aligned_tokens = (
                                (expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                        expert_start += aligned_tokens
                        expert_idx += cutlass.Int32(1)

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
                        # One lane spins on the GPU-scope acquire load;
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

        num_ab_stage = (smem_capacity // occupancy - _CuTeDSLDispatchGroupGemmKernelImpl.reserved_smem_bytes -
                        epi_bytes) // ab_bytes_per_stage

        remaining_smem = (smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy *
                          (_CuTeDSLDispatchGroupGemmKernelImpl.reserved_smem_bytes + epi_bytes))
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


class CuTeDSLDispatchGroupGemmKernel(_CuTeDSLDispatchGroupGemmKernelImpl):
    """Production fused dispatch + group-GEMM kernel.

    Dispatch warps emit per-expert release signals; the GEMM TMA warp
    acquires on them before issuing the first A/B load, yielding the full
    producer/consumer overlap path.
    """

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
