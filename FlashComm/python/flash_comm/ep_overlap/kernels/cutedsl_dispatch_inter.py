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
import cutlass.utils as cute_utils
import nccl.core.device.cute as nccl_cute
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import Agent, CooperativeGroup, PipelineTmaAsync


class MoEDispatchInter:
    """Inter-node EP dispatch using NCCL GIN plus local TMA.

    The RDMA rail buffer layout, signal ids, and node ring order match
    ``kernel_dispatch_internode`` in ``csrc/ep/kernels/internode_cuda.cu``.
    This standalone dispatch materializes the final per-expert ``A_padded``:
    duplicate topk rows are stored even when layout ``need_send`` is false.
    The send mask is still copied into node-view metadata for later combine.
    """

    kMaxLocalWorldSize: int = 16
    num_consumer_warps: int = 3
    num_warps: int = 1 + num_consumer_warps
    num_threads_per_warp: int = 32
    producer_warp_id: int = 0
    consumer_warp_id_base: int = 1
    cluster_shape_mn: tuple = (1, 1)
    buffer_align_bytes: int = 128
    max_smem_bytes: int = 228 * 1024

    @staticmethod
    def compute_num_stages(hidden_size: int, dtype_bytes: int, max_smem_bytes: int = 228 * 1024) -> int:
        fixed_bytes = MoEDispatchInter.kMaxLocalWorldSize * 2 * 8
        alignment_overhead = MoEDispatchInter.buffer_align_bytes
        per_stage = hidden_size * dtype_bytes + 2 * 8
        available = max_smem_bytes - fixed_bytes - alignment_overhead
        capped = min(12, available // per_stage)
        num_consumer_warps = MoEDispatchInter.num_consumer_warps
        return max(num_consumer_warps, (capped // num_consumer_warps) * num_consumer_warps)

    @property
    def num_threads(self) -> int:
        return self.num_warps * self.num_threads_per_warp

    @cute.jit
    def __call__(
        self,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        recv_x_ptrs: cute.Tensor,
        recv_topk_scatter_indices_ptrs: cute.Tensor,
        node_topk_indices: cute.Tensor,
        node_topk_send_mask: cute.Tensor,
        node_token_dst_scatter_indices: cute.Tensor,
        output_buf: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        max_recv_tokens: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
        num_sms: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        dtype = output_buf.element_type
        dtype_bytes = dtype.width // 8
        nbytes_per_token = hidden_size * dtype_bytes
        num_stages = MoEDispatchInter.compute_num_stages(
            hidden_size,
            dtype_bytes,
            self.max_smem_bytes,
        )
        self.pipe_cnt = 2

        token_region_bytes = max_slot_num_token * hidden_size * dtype_bytes
        meta_region_bytes = max_slot_num_token * topk * 4
        slot_payload_bytes = (token_region_bytes + 3 * meta_region_bytes + max_slot_num_token * topk * 4)
        slot_stride_bytes = ((slot_payload_bytes + 4095) // 4096) * 4096

        @cute.struct
        class SharedStorage:
            pipeline_bars: cute.struct.MemRange[cutlass.Int64, 2 * num_stages]
            recv_x_ptrs_smem: cute.struct.MemRange[cutlass.Int64, MoEDispatchInter.kMaxLocalWorldSize]
            recv_scatter_ptrs_smem: cute.struct.MemRange[cutlass.Int64, MoEDispatchInter.kMaxLocalWorldSize]
            tma_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, hidden_size * num_stages],
                MoEDispatchInter.buffer_align_bytes,
            ]

        self.kernel(
            dev_comm_ptr,
            rdma_rail_send_win_handle,
            num_tokens_per_rank,
            recv_x_ptrs,
            recv_topk_scatter_indices_ptrs,
            node_topk_indices,
            node_topk_send_mask,
            node_token_dst_scatter_indices,
            dtype,
            hidden_size,
            topk,
            nbytes_per_token,
            token_region_bytes,
            slot_stride_bytes,
            num_stages,
            rank,
            world_size,
            local_world_size,
            max_slot_num_token,
            max_recv_tokens,
            num_experts_per_rank,
            SharedStorage,
        ).launch(
            grid=(num_sms, 1, 1),
            block=(self.num_threads, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        dev_comm_ptr: cutlass.Int64,
        rdma_rail_send_win_handle: cutlass.Int64,
        num_tokens_per_rank: cute.Tensor,
        recv_x_ptrs: cute.Tensor,
        recv_topk_scatter_indices_ptrs: cute.Tensor,
        node_topk_indices: cute.Tensor,
        node_topk_send_mask: cute.Tensor,
        node_token_dst_scatter_indices: cute.Tensor,
        token_dtype: cutlass.Constexpr,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        nbytes_per_token: cutlass.Constexpr[int],
        token_region_bytes: cutlass.Constexpr[int],
        slot_stride_bytes: cutlass.Constexpr[int],
        num_stages: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        local_world_size: cutlass.Constexpr[int],
        max_slot_num_token: cutlass.Constexpr[int],
        max_recv_tokens: cutlass.Constexpr[int],
        num_experts_per_rank: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        block_id, _, _ = cute.arch.block_idx()
        num_blocks, _, _ = cute.arch.grid_dim()

        meta_dtype = node_topk_indices.element_type

        bulk_g2s_atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            token_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        bulk_s2g_atom = cute.make_copy_atom(
            cpasync.CopyBulkS2GOp(),
            token_dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        row_layout = cute.make_layout(hidden_size)

        smem = cute_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sBuf = storage.tma_buffer.get_tensor(cute.make_layout(
            (num_stages, hidden_size),
            stride=(hidden_size, 1),
        ))
        smem_recv_x_ptrs = storage.recv_x_ptrs_smem.get_tensor(cute.make_layout(MoEDispatchInter.kMaxLocalWorldSize))
        smem_recv_scatter_ptrs = storage.recv_scatter_ptrs_smem.get_tensor(
            cute.make_layout(MoEDispatchInter.kMaxLocalWorldSize))

        pipeline = PipelineTmaAsync.create(
            num_stages=num_stages,
            producer_group=CooperativeGroup(Agent.Thread, size=1),
            consumer_group=CooperativeGroup(Agent.Thread, size=1),
            tx_count=nbytes_per_token,
            barrier_storage=storage.pipeline_bars.data_ptr(),
        )

        if tidx < local_world_size:
            smem_recv_x_ptrs[tidx] = recv_x_ptrs[tidx]
            smem_recv_scatter_ptrs[tidx] = recv_topk_scatter_indices_ptrs[tidx]
        cute.arch.barrier()

        dev_comm = nccl_cute.DevComm(dev_comm_ptr)
        rdma_win = nccl_cute.Window(rdma_rail_send_win_handle)
        team = dev_comm.team_world
        local_rank = cutlass.Int32(rank % local_world_size)
        my_node = cutlass.Int32(rank // local_world_size)
        nnodes = cutlass.Int32(world_size // local_world_size)
        dtype_bytes = cutlass.Int32(nbytes_per_token // hidden_size)
        meta_elem_bytes = cutlass.Int32(4)
        gin_context_id = 0
        signal_op_inc = 0

        if warp_idx == self.producer_warp_id:
            gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, gin_context_id)
            coop = nccl_cute.warp()
            producer = pipeline.make_producer()
            node_offset = cutlass.Int32(0)
            while node_offset < nnodes:
                src_node = (my_node + node_offset) % nnodes
                src_global_rank = src_node * local_world_size + local_rank
                src_num_token = num_tokens_per_rank[src_global_rank]
                local_num_token = num_tokens_per_rank[rank]
                prefetch_dst_node = (my_node + nnodes - ((node_offset + cutlass.Int32(1)) % nnodes)) % nnodes

                if block_id == 0 and prefetch_dst_node != my_node:
                    dst_rank = prefetch_dst_node * local_world_size + local_rank
                    src_off = cutlass.Int64(my_node) * cutlass.Int64(slot_stride_bytes)
                    token_bytes = cutlass.Int32(local_num_token * hidden_size * dtype_bytes)
                    meta_bytes = cutlass.Int32(local_num_token * topk * meta_elem_bytes)
                    meta_bundle_bytes = meta_bytes * cutlass.Int32(3)
                    remaining = cutlass.Int32(0)
                    if token_bytes > 0:
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
                            src_tensor = rdma_win.tensor(
                                cutlass.Int8,
                                cute.make_layout(token_bytes),
                                src_off,
                            )
                            dst_tensor = rdma_win.tensor(
                                cutlass.Int8,
                                cute.make_layout(token_bytes),
                                src_off,
                            )
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
                        if meta_bundle_bytes > 0:
                            remaining -= cutlass.Int32(1)
                            meta_off = src_off + cutlass.Int64(token_region_bytes)
                            src_meta = rdma_win.tensor(
                                cutlass.Int8,
                                cute.make_layout(meta_bundle_bytes),
                                meta_off,
                            )
                            dst_meta = rdma_win.tensor(
                                cutlass.Int8,
                                cute.make_layout(meta_bundle_bytes),
                                meta_off,
                            )
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
                token_idx = block_id
                while token_idx < src_num_token:
                    handle = producer.acquire_and_advance()
                    src_row_off = slot_off + cutlass.Int64(token_idx) * cutlass.Int64(nbytes_per_token)
                    with cute.arch.elect_one():
                        sDst = cute.make_tensor(
                            sBuf.iterator + handle.index * hidden_size,
                            row_layout,
                        )
                        gSrc = cute.make_tensor(
                            cute.make_ptr(
                                token_dtype,
                                rdma_win.local_pointer(src_row_off),
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            row_layout,
                        )
                        cute.copy(
                            bulk_g2s_atom,
                            gSrc,
                            sDst,
                            mbar_ptr=handle.barrier,
                        )
                    handle.commit()
                    token_idx += num_blocks
                node_offset += cutlass.Int32(1)
            producer.tail()

        elif (warp_idx >= self.consumer_warp_id_base
              and warp_idx < self.consumer_warp_id_base + self.num_consumer_warps):
            consumer_group_id = warp_idx - cutlass.Int32(self.consumer_warp_id_base)
            consumer_read = pipeline.make_consumer()
            consumer_release = consumer_read.clone()
            for group in cutlass.range(self.num_consumer_warps, unroll_full=True):
                if group < consumer_group_id:
                    consumer_read.advance()
                    consumer_release.advance()
            pipe_cnt_rt = cutlass.Int32(self.pipe_cnt)
            tokens_processed = cutlass.Int32(0)
            node_base_count = cutlass.Int32(0)
            node_offset = cutlass.Int32(0)
            while node_offset < nnodes:
                src_node = (my_node + node_offset) % nnodes
                src_global_rank = src_node * local_world_size + local_rank
                src_num_token = num_tokens_per_rank[src_global_rank]
                node_block_tokens = cutlass.Int32(0)
                if block_id < src_num_token:
                    node_block_tokens = (src_num_token - cutlass.Int32(1) - block_id) // num_blocks + cutlass.Int32(1)
                slot_off = cutlass.Int64(src_node) * cutlass.Int64(slot_stride_bytes)
                meta_bytes = cutlass.Int64(src_num_token) * cutlass.Int64(topk) * cutlass.Int64(4)
                topk_off = slot_off + cutlass.Int64(token_region_bytes)
                mask_off = topk_off + meta_bytes
                scatter_off = mask_off + meta_bytes

                first_node_ordinal = (consumer_group_id + cutlass.Int32(self.num_consumer_warps) -
                                      (node_base_count % cutlass.Int32(self.num_consumer_warps))) % cutlass.Int32(
                                          self.num_consumer_warps)
                node_ordinal = first_node_ordinal
                while node_ordinal < node_block_tokens:
                    token_idx = block_id + node_ordinal * num_blocks
                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(
                            self.pipe_cnt - 1,
                            read=True,
                        )
                    cute.arch.sync_warp()

                    if tokens_processed >= pipe_cnt_rt:
                        consumer_release.release()
                        for _ in cutlass.range(self.num_consumer_warps, unroll_full=True):
                            consumer_release.advance()

                    handle = consumer_read.wait()

                    my_expert_idx = cutlass.Int32(-1)
                    my_target_rank = cutlass.Int32(-1)
                    my_is_need_send = cutlass.Int32(0)
                    my_store_idx = cutlass.Int32(-1)
                    if lane_idx < topk:
                        meta_idx = token_idx * topk + lane_idx
                        g_topk = cute.make_ptr(
                            meta_dtype,
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
                            meta_dtype,
                            rdma_win.local_pointer(scatter_off),
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        my_expert_idx = g_topk[meta_idx]
                        my_target_rank = my_expert_idx // num_experts_per_rank
                        my_is_need_send = g_mask[meta_idx]
                        my_store_idx = g_scatter[meta_idx]
                        node_meta_idx = (src_node * max_slot_num_token + token_idx) * topk + lane_idx
                        node_topk_indices[node_meta_idx] = my_expert_idx
                        node_topk_send_mask[node_meta_idx] = my_is_need_send
                        node_token_dst_scatter_indices[node_meta_idx] = my_store_idx

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
                        sBuf.iterator + handle.index * hidden_size,
                        row_layout,
                    )
                    for send_lane in cutlass.range(topk, unroll_full=True):
                        b_should_store = cute.arch.shuffle_sync(should_store, send_lane)
                        b_target_local_rank = cute.arch.shuffle_sync(my_target_local_rank, send_lane)
                        b_target_global_rank = cute.arch.shuffle_sync(my_target_rank, send_lane)
                        b_store_idx = cute.arch.shuffle_sync(my_store_idx, send_lane)
                        if b_should_store != 0:
                            dst_base_i64 = smem_recv_x_ptrs[b_target_local_rank]
                            dst_row_i64 = (dst_base_i64 + cutlass.Int64(b_store_idx) * cutlass.Int64(nbytes_per_token))
                            with cute.arch.elect_one():
                                gDst = cute.make_tensor(
                                    cute.make_ptr(
                                        token_dtype,
                                        dst_row_i64,
                                        cute.AddressSpace.gmem,
                                        assumed_align=16,
                                    ),
                                    row_layout,
                                )
                                cute.copy(bulk_s2g_atom, sSrc, gDst)

                            if lane_idx < topk:
                                scatter_base_i64 = smem_recv_scatter_ptrs[b_target_local_rank]
                                scatter_idx = b_store_idx * topk + lane_idx
                                out_val = cutlass.Int32(-1)
                                if my_target_rank == b_target_global_rank:
                                    out_val = my_store_idx
                                g_out_scatter = cute.make_ptr(
                                    meta_dtype,
                                    scatter_base_i64,
                                    cute.AddressSpace.gmem,
                                    assumed_align=4,
                                )
                                g_out_scatter[scatter_idx] = out_val
                        cute.arch.sync_warp()

                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_commit_group()
                    cute.arch.sync_warp()

                    for _ in cutlass.range(self.num_consumer_warps, unroll_full=True):
                        consumer_read.advance()
                    tokens_processed += cutlass.Int32(1)
                    node_ordinal += cutlass.Int32(self.num_consumer_warps)
                node_base_count += node_block_tokens
                node_offset += cutlass.Int32(1)

            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            for _ in cutlass.range(self.pipe_cnt, unroll_full=True):
                if tokens_processed > cutlass.Int32(0):
                    consumer_release.release()
                    for _ in cutlass.range(self.num_consumer_warps, unroll_full=True):
                        consumer_release.advance()
                    tokens_processed -= cutlass.Int32(1)
