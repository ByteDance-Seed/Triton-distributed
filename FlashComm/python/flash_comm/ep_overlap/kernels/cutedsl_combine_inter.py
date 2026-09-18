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

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as cute_utils
import nccl.core.device.cute as nccl_cute
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import Agent, CooperativeGroup, PipelineTmaAsync


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


class MoECombineInterPartial:
    """Build per-owner-node partial rows in CUDA-compatible combine slots.

    Dispatch materializes every valid ``(token, topk)`` expert row before FC2,
    including duplicate same-rank topk rows.  Therefore combine validity follows
    ``expert_rank``/``scatter`` metadata directly and intentionally does not
    gate on ``token_topk_send_mask``.
    """

    num_threads_per_warp: int = 32
    producer_warp_id: int = 0
    consumer_warps: int = 8
    num_threads: int = 288
    atom_v: int = 8
    num_stages: int = 2
    buffer_align_bytes: int = 128

    @cute.jit
    def __call__(
        self,
        input_buf: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        combine_weight_ptrs: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_topk_send_mask: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        dtype = input_buf.element_type
        dtype_bytes = dtype.width // 8
        tile_n = self.consumer_warps * self.num_threads_per_warp * self.atom_v
        assert hidden_size % tile_n == 0, (f"hidden_size={hidden_size} must be a multiple of "
                                           f"consumer_warps*warp_size*atom_v={tile_n}")
        num_hidden_tiles = hidden_size // tile_n
        tile_bytes = tile_n * dtype_bytes
        slot_stride_bytes = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        weight_region_off = _rail_weight_region_offset(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )

        @cute.struct
        class SharedStorage:
            pipeline_bars: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages]
            tma_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, self.num_stages * tile_n],
                self.buffer_align_bytes,
            ]

        self.kernel(
            input_buf,
            combine_x_ptrs,
            combine_weight_ptrs,
            rdma_rail_send_win_handle,
            num_tokens_per_rank,
            local_topk_indices,
            local_topk_send_mask,
            local_token_dst_scatter_indices,
            hidden_size,
            topk,
            rank,
            world_size,
            local_world_size,
            max_slot_num_token,
            num_experts_per_rank,
            has_weight,
            slot_stride_bytes,
            weight_region_off,
            tile_n,
            tile_bytes,
            SharedStorage,
        ).launch(
            grid=(world_size // local_world_size, max_slot_num_token, num_hidden_tiles),
            block=(self.num_threads, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_buf: cute.Tensor,
        combine_x_ptrs: cute.Tensor,
        combine_weight_ptrs: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        local_topk_indices: cute.Tensor,
        local_topk_send_mask: cute.Tensor,
        local_token_dst_scatter_indices: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        slot_stride_bytes: cutlass.Constexpr[int],
        weight_region_off: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        tile_bytes: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        _ = max_slot_num_token
        _ = local_topk_send_mask
        _ = combine_weight_ptrs
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        owner_node, token_offset, hidden_tile = cute.arch.block_idx()

        nnodes = cutlass.Int32(world_size // local_world_size)
        my_node = cutlass.Int32(rank // local_world_size)
        local_rank = cutlass.Int32(rank % local_world_size)
        my_node_rank_begin = my_node * cutlass.Int32(local_world_size)
        my_node_rank_end = my_node_rank_begin + cutlass.Int32(local_world_size)
        src_rank = owner_node * cutlass.Int32(local_world_size) + local_rank
        src_num_token = num_tokens_per_rank[src_rank]
        if token_offset < src_num_token:
            dtype = input_buf.element_type
            atom_v: cutlass.Constexpr[int] = self.atom_v
            dtype_bytes: cutlass.Constexpr[int] = dtype.width // 8
            tile_byte_offset = cutlass.Int64(hidden_tile * tile_n) * cutlass.Int64(dtype_bytes)
            row_byte_stride = cutlass.Int64(hidden_size) * cutlass.Int64(dtype_bytes)

            bulk_g2s_atom = cute.make_copy_atom(
                cpasync.CopyBulkG2SOp(),
                dtype,
                num_bits_per_copy=tile_bytes * 8,
            )
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                dtype,
                num_bits_per_copy=128,
            )
            tile_layout = cute.make_layout(tile_n)
            atom_layout = cute.make_layout(atom_v)

            smem = cute_utils.SmemAllocator()
            storage = smem.allocate(SharedStorage)
            sBuf = storage.tma_buffer.get_tensor(cute.make_layout(
                (self.num_stages, tile_n),
                stride=(tile_n, 1),
            ))
            tma_pipeline = PipelineTmaAsync.create(
                num_stages=self.num_stages,
                producer_group=CooperativeGroup(Agent.Thread, size=1),
                consumer_group=CooperativeGroup(Agent.Thread, size=self.consumer_warps),
                tx_count=tile_bytes,
                barrier_storage=storage.pipeline_bars.data_ptr(),
            )

            if warp_idx == self.producer_warp_id:
                producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_stages,
                )
                for k in cutlass.range_constexpr(topk):
                    expert_idx = local_topk_indices[owner_node, token_offset, k]
                    expert_rank = expert_idx // cutlass.Int32(num_experts_per_rank)
                    scatter = local_token_dst_scatter_indices[owner_node, token_offset, k]
                    is_valid = (expert_rank >= my_node_rank_begin and expert_rank < my_node_rank_end and scatter >= 0)
                    if is_valid:
                        expert_local_rank = expert_rank - my_node_rank_begin
                        src_base = combine_x_ptrs[expert_local_rank]
                        src_addr = (src_base + cutlass.Int64(scatter) * row_byte_stride + tile_byte_offset)
                        tma_pipeline.producer_acquire(producer_state)
                        with cute.arch.elect_one():
                            sDst = cute.make_tensor(
                                sBuf.iterator + producer_state.index * tile_n,
                                tile_layout,
                            )
                            gSrc = cute.make_tensor(
                                cute.make_ptr(
                                    dtype,
                                    src_addr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                tile_layout,
                            )
                            cute.copy(
                                bulk_g2s_atom,
                                gSrc,
                                sDst,
                                mbar_ptr=tma_pipeline.producer_get_barrier(producer_state),
                            )
                        tma_pipeline.producer_commit(producer_state)
                        producer_state.advance()
                tma_pipeline.producer_tail(producer_state)

                if cutlass.const_expr(has_weight):
                    if hidden_tile == 0 and tidx < topk:
                        weight_bits = cutlass.Int32(0)

                        partial_slot = nnodes + owner_node
                        if owner_node == my_node:
                            partial_slot = nnodes * cutlass.Int32(2) + my_node
                        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
                        dst_off = (cutlass.Int64(partial_slot) * cutlass.Int64(slot_stride_bytes) +
                                   cutlass.Int64(weight_region_off) +
                                   cutlass.Int64(token_offset * topk + tidx) * cutlass.Int64(4))
                        gDstWeight = cute.make_ptr(
                            cutlass.Int32,
                            rdma_win.local_pointer(dst_off),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        gDstWeight[0] = weight_bits

            elif warp_idx > self.producer_warp_id:
                consumer_tid = tidx - self.num_threads_per_warp
                consumer_lane = consumer_tid % self.num_threads_per_warp
                atom_elem_offset = hidden_tile * tile_n + consumer_tid * atom_v
                atom_byte_offset = cutlass.Int64(atom_elem_offset) * cutlass.Int64(dtype_bytes)
                tCrOut = cute.make_fragment((atom_v, ), dtype)
                tCrAcc = cute.make_fragment((atom_v, ), cutlass.Float32)
                tCrAcc.fill(0.0)
                consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer,
                    self.num_stages,
                )

                for k in cutlass.range_constexpr(topk):
                    expert_idx = local_topk_indices[owner_node, token_offset, k]
                    expert_rank = expert_idx // cutlass.Int32(num_experts_per_rank)
                    scatter = local_token_dst_scatter_indices[owner_node, token_offset, k]
                    is_valid = (expert_rank >= my_node_rank_begin and expert_rank < my_node_rank_end and scatter >= 0)
                    if is_valid:
                        tma_pipeline.consumer_wait(consumer_state)
                        sSrc = cute.make_tensor(
                            sBuf.iterator + consumer_state.index * tile_n + consumer_tid * atom_v,
                            atom_layout,
                        )
                        for v in cutlass.range_constexpr(atom_v):
                            tCrAcc[v] = tCrAcc[v] + sSrc[v].to(cutlass.Float32)
                        cute.arch.sync_warp()
                        if consumer_lane == 0:
                            tma_pipeline.sync_object_empty.arrive(
                                consumer_state.index,
                                tma_pipeline.consumer_mask,
                            )
                        consumer_state.advance()

                tCrOut.store(tCrAcc.load().to(dtype))
                partial_slot = nnodes + owner_node
                if owner_node == my_node:
                    partial_slot = nnodes * cutlass.Int32(2) + my_node
                rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
                dst_off = (cutlass.Int64(partial_slot) * cutlass.Int64(slot_stride_bytes) +
                           cutlass.Int64(token_offset) * row_byte_stride + atom_byte_offset)
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


class MoECombineInterExchange:
    """Move remote-owner partial slots with NCCL GIN and wait for arrivals."""

    num_threads: int = 32

    @cute.jit
    def __call__(
        self,
        dev_comm_ptr: cutlass.Int64,
        input_buf: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        dtype_bytes = input_buf.element_type.width // 8
        slot_stride_bytes = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        weight_region_off = _rail_weight_region_offset(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        self.kernel(
            dev_comm_ptr,
            rdma_rail_send_win_handle,
            num_tokens_per_rank,
            hidden_size,
            topk,
            rank,
            world_size,
            local_world_size,
            slot_stride_bytes,
            weight_region_off,
            has_weight,
            dtype_bytes,
        ).launch(
            grid=(1, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        slot_stride_bytes: cutlass.Constexpr[int],
        weight_region_off: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        dtype_bytes: cutlass.Constexpr[int],
    ):
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
                    if cutlass.const_expr(has_weight):
                        gin.put(
                            team,
                            peer,
                            rdma_win,
                            dst_tensor,
                            rdma_win,
                            src_tensor,
                            coop,
                            is_signal=False,
                            signal_id=signal_id,
                            signal_op=signal_op_inc,
                            signal_op_arg=1,
                        )
                        weight_bytes = src_num_token * cutlass.Int32(topk * 4)
                        weight_src_off = src_off + cutlass.Int64(weight_region_off)
                        weight_dst_off = dst_off + cutlass.Int64(weight_region_off)
                        src_weight = rdma_win.tensor(
                            cutlass.Int8,
                            cute.make_layout(weight_bytes),
                            weight_src_off,
                        )
                        dst_weight = rdma_win.tensor(
                            cutlass.Int8,
                            cute.make_layout(weight_bytes),
                            weight_dst_off,
                        )
                        gin.put(
                            team,
                            peer,
                            rdma_win,
                            dst_weight,
                            rdma_win,
                            src_weight,
                            coop,
                            is_signal=True,
                            signal_id=signal_id,
                            signal_op=signal_op_inc,
                            signal_op_arg=1,
                        )
                    else:
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


class MoECombineInterReduce:
    """Reduce incoming per-node partial slots into the dense local output."""

    num_threads: int = 256
    atom_v: int = 8

    @cute.jit
    def __call__(
        self,
        output: cute.Tensor,
        output_weight: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        dtype = output.element_type
        dtype_bytes = dtype.width // 8
        tile_n = self.num_threads * self.atom_v
        assert hidden_size % tile_n == 0, (f"hidden_size={hidden_size} must be a multiple of "
                                           f"num_threads*atom_v={tile_n}")
        num_hidden_tiles = hidden_size // tile_n
        token_region_bytes = max_slot_num_token * hidden_size * dtype_bytes
        slot_stride_bytes = _rail_slot_stride_bytes(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        weight_region_off = _rail_weight_region_offset(
            max_slot_num_token,
            hidden_size,
            topk,
            dtype_bytes,
        )
        self.kernel(
            output,
            output_weight,
            rdma_rail_send_win_handle,
            hidden_size,
            topk,
            rank,
            world_size,
            local_world_size,
            slot_stride_bytes,
            token_region_bytes,
            weight_region_off,
            num_experts_per_rank,
            has_weight,
            num_tokens,
        ).launch(
            grid=(num_tokens, num_hidden_tiles, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        output: cute.Tensor,
        output_weight: cute.Tensor,
        rdma_rail_send_win_handle: cutlass.Int64,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        slot_stride_bytes: cutlass.Constexpr[int],
        token_region_bytes: cutlass.Constexpr[int],
        weight_region_off: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
        has_weight: cutlass.Constexpr[int],
        num_tokens: cutlass.Int32,
    ):
        _ = num_tokens
        tidx, _, _ = cute.arch.thread_idx()
        token_idx, hidden_tile, _ = cute.arch.block_idx()

        dtype = output.element_type
        dtype_bytes: cutlass.Constexpr[int] = dtype.width // 8
        atom_v: cutlass.Constexpr[int] = self.atom_v
        tile_n: cutlass.Constexpr[int] = self.num_threads * atom_v
        atom_elem_offset = hidden_tile * tile_n + tidx * atom_v
        atom_byte_offset = cutlass.Int64(atom_elem_offset) * cutlass.Int64(dtype_bytes)
        row_byte_stride = cutlass.Int64(hidden_size) * cutlass.Int64(dtype_bytes)

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
        tCrAcc.fill(0.0)

        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
        nnodes = cutlass.Int32(world_size // local_world_size)
        node = cutlass.Int32(0)
        while node < nnodes:
            slot = nnodes * cutlass.Int32(2) + node
            src_off = (cutlass.Int64(slot) * cutlass.Int64(slot_stride_bytes) +
                       cutlass.Int64(token_idx) * row_byte_stride + atom_byte_offset)
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
        gOut = cute.zipped_divide(output, (1, atom_v))
        cute.copy(
            store_atom,
            tCrOut,
            gOut[(0, None), (token_idx, hidden_tile * self.num_threads + tidx)],
        )

        if cutlass.const_expr(has_weight):
            if hidden_tile == 0 and tidx < topk:
                my_node = cutlass.Int32(rank // local_world_size)
                weight_offset = cutlass.Int64(token_idx * topk + tidx) * cutlass.Int64(4)
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
                output_weight[token_idx, tidx] = weight_value
