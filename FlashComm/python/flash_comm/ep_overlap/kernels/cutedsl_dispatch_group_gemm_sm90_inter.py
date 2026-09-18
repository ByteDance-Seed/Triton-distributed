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

from typing import Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import nccl.core.device.cute as nccl_cute
from cutlass.cutlass_dsl import Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    create_initial_search_state, )

from .cutedsl_dispatch_group_gemm_sm90 import (
    CuTeDSLDispatchGroupGemmKernelSM90 as _IntraNodeDispatchGroupGemmKernelSM90, )
from .m_contig_group_tile_scheduler import MContiguousGroupTileScheduler


class CuTeDSLDispatchGroupGemmInterKernelSM90(_IntraNodeDispatchGroupGemmKernelSM90):
    """One-launch inter-node dispatch + FC1 WGMMA GEMM.

    The CTA shape intentionally keeps the existing Hopper fused FC1 kernel's
    12-warps footprint. Warp 0 belongs to the DMA warpgroup and runs the
    NCCL GIN ring producer. Warps 1-2 run local dispatch consumers and
    publish per-expert readiness as soon as each target expert's last row is
    stored. Warp 3 runs the GEMM TMA load path so matrix tile loading is
    decoupled from the GIN producer without adding register pressure.
    """

    kMaxLocalWorldSize: int = 16
    inter_producer_warp_id: int = 0
    inter_consumer_warp_id_base: int = 1
    inter_num_consumer_warps: int = 2
    gemm_load_warp_id: int = 3
    inter_store_pipe_cnt: int = 2

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: tuple,
        cluster_shape_mn: tuple,
        dispatch_num_stages: int = 1,
    ):
        super().__init__(
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            wait_signals=True,
            dispatch_num_stages=dispatch_num_stages,
        )
        if self.num_dma_warp_groups != 1:
            raise ValueError("inter-node fused SM90 kernel expects one DMA warpgroup")
        if self.num_warps_per_warp_group < 4:
            raise ValueError("inter-node fused SM90 kernel needs four DMA warps")
        consumer_warps = self.inter_num_consumer_warps
        self.dispatch_buffer_rows = max(
            consumer_warps,
            ((self.dispatch_buffer_rows + consumer_warps - 1) // consumer_warps) * consumer_warps,
        )
        self.dispatch_bar_count = 2 * self.dispatch_buffer_rows

    def _dispatch_smem_bytes(self, problem_shape_k: int, has_weight: bool = False, world_size: int = 0) -> int:
        _ = world_size
        dtype_bytes = self.a_dtype.width // 8
        ptr_tables = self.kMaxLocalWorldSize * 3 * 8
        if has_weight:
            ptr_tables += self.kMaxLocalWorldSize * 8
        return (self.dispatch_buffer_rows * problem_shape_k * dtype_bytes + ptr_tables + self.dispatch_bar_count * 8 +
                self.dispatch_buffer_align_bytes)

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
            raise TypeError("SM90 inter-node fused dispatch+GEMM supports fp16/bf16 inputs")

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

        dtype_bytes = self.a_dtype.width // 8
        weight_dtype = output_weight.element_type
        if cutlass.const_expr(has_weight and weight_dtype != cutlass.Float32):
            raise TypeError("SM90 inter-node dispatch+GEMM has_weight path requires float32 output_weight "
                            "because weights are moved as 32-bit bit patterns")
        weight_bytes = weight_dtype.width // 8
        nbytes_per_token = problem_shape_k * dtype_bytes
        token_region_bytes = max_slot_num_token * problem_shape_k * dtype_bytes
        meta_region_bytes = max_slot_num_token * topk * 4
        weight_region_off = token_region_bytes + 3 * meta_region_bytes
        slot_payload_bytes = weight_region_off + max_slot_num_token * topk * weight_bytes
        slot_stride_bytes = ((slot_payload_bytes + 4095) // 4096) * 4096

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
                dispatch_recv_x_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_recv_scatter_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
                dispatch_signal_state_ptrs: cute.struct.MemRange[cutlass.Int64, self.kMaxLocalWorldSize]
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
            min_blocks_per_mp=1,
            stream=stream,
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
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
        lane_idx = cute.arch.lane_idx()

        if warp_idx == self.gemm_load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        bid = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()
        linear_dispatch_block = (bid[2] * grid_dim[0] * grid_dim[1] + bid[1] * grid_dim[0] + bid[0])
        num_dispatch_blocks = cute.size(grid_dim)
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
        dispatch_sbuf = storage.dispatch_tma_buffer.get_tensor(
            cute.make_layout(
                (self.dispatch_buffer_rows, problem_shape_k),
                stride=(problem_shape_k, 1),
            ))
        dispatch_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.dispatch_buffer_rows,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
            tx_count=nbytes_per_token,
            barrier_storage=storage.dispatch_pipeline_bars.data_ptr(),
            tidx=lane_idx,
        )
        smem_recv_x_ptrs = storage.dispatch_recv_x_ptrs.get_tensor(cute.make_layout(self.kMaxLocalWorldSize))
        smem_recv_scatter_ptrs = storage.dispatch_recv_scatter_ptrs.get_tensor(cute.make_layout(
            self.kMaxLocalWorldSize))
        smem_signal_state_ptrs = storage.dispatch_signal_state_ptrs.get_tensor(cute.make_layout(
            self.kMaxLocalWorldSize))
        if cutlass.const_expr(has_weight):
            smem_recv_weight_ptrs = storage.dispatch_recv_weight_ptrs.get_tensor(
                cute.make_layout(self.kMaxLocalWorldSize))

        if tidx < local_world_size:
            smem_recv_x_ptrs[tidx] = recv_x_ptrs[tidx]
            smem_recv_scatter_ptrs[tidx] = recv_topk_scatter_indices_ptrs[tidx]
            smem_signal_state_ptrs[tidx] = expert_signal_state_ptrs[tidx]
            if cutlass.const_expr(has_weight):
                smem_recv_weight_ptrs[tidx] = recv_weight_ptrs[tidx]
        cute.arch.barrier()

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
                        gin.signal(
                            team,
                            dst_rank,
                            True,
                            my_node * cutlass.Int32(32),
                            signal_op_inc,
                            1,
                            coop,
                        )
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
                    src_row_off = slot_off + cutlass.Int64(token_idx) * cutlass.Int64(nbytes_per_token)
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
                        cute.copy(
                            dispatch_g2s_atom,
                            gSrc,
                            sDst,
                            mbar_ptr=handle.barrier,
                        )
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
                                      (node_base_count % cutlass.Int32(self.inter_num_consumer_warps))) % cutlass.Int32(
                                          self.inter_num_consumer_warps)
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
                        node_meta_idx = (src_node * max_slot_num_token + token_idx) * topk + lane_idx
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
                            dst_row_i64 = dst_base_i64 + cutlass.Int64(b_store_idx) * cutlass.Int64(nbytes_per_token)
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
                                    weight_addr = weight_base_i64 + cutlass.Int64(b_store_idx) * cutlass.Int64(
                                        weight_bytes)
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
                            counter_addr = target_base + cutlass.Int64(group_count +
                                                                       b_local_expert_idx) * cutlass.Int64(4)
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
                                    sig_addr = target_base + cutlass.Int64(b_local_expert_idx) * cutlass.Int64(4)
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

        if warp_idx == self.gemm_load_warp_id:
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

                if cur_group_idx != last_waited_group:
                    sig_ptr = (expert_signals.iterator + cur_group_idx).llvm_ptr
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


CuTeDSLDispatchGroupGemmInterKernel = CuTeDSLDispatchGroupGemmInterKernelSM90
