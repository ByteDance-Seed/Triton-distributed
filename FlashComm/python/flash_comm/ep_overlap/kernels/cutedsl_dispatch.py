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
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import Agent, CooperativeGroup, PipelineTmaAsync

from .cutedsl_utils import decode_token_src_rank_topk_and_indices

# ---------------------------------------------------------------------------
# Pull-mode dispatch kernel
# ---------------------------------------------------------------------------


class MoEDispatch:
    """Warp-specialized pull-mode intranode MoE dispatch (CuTeDSL).

    Two warps + ``PipelineTmaAsync``:
      * Producer (warp 0): acquire -> G2S pull -> commit.
      * Consumer (warp 1): wait -> S2G store -> delayed release.

    ``num_stages`` is auto-computed at JIT time from the SMEM budget.
    Padding slots (sentinel ``-1``) are skipped entirely. ``enable_signals``
    processes experts in order and publishes one release signal per local
    expert when all of its rows are written.
    """

    kMaxWorldSize: int = 128
    num_warps: int = 2
    num_threads_per_warp: int = 32
    producer_warp_id: int = 0
    consumer_warp_id: int = 1
    cluster_shape_mn: tuple = (2, 1)
    buffer_align_bytes: int = 128
    max_smem_bytes: int = 228 * 1024

    @staticmethod
    def compute_num_stages(hidden_size: int, dtype_bytes: int, max_smem_bytes: int = 228 * 1024) -> int:
        """Max pipeline stages that fit in SMEM after fixed allocations."""
        fixed_bytes = MoEDispatch.kMaxWorldSize * 8
        alignment_overhead = MoEDispatch.buffer_align_bytes
        per_stage = hidden_size * dtype_bytes + 2 * 8
        available = max_smem_bytes - fixed_bytes - alignment_overhead
        return max(2, available // per_stage)

    @property
    def num_threads(self) -> int:
        return self.num_warps * self.num_threads_per_warp

    @cute.jit
    def __call__(
        self,
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        output: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        num_sms: cutlass.Int32,
        stream: cuda.CUstream,
        recv_expert_counts: cute.Tensor,
        expert_signals: cute.Tensor,
        expert_signal_counters: cute.Tensor,
        experts_per_rank: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        enable_signals: cutlass.Constexpr[int],
    ):
        dtype = output.element_type
        dtype_bytes = dtype.width // 8
        nbytes_per_token = hidden_size * dtype_bytes
        num_stages = MoEDispatch.compute_num_stages(hidden_size, dtype_bytes, self.max_smem_bytes)
        self.pipe_cnt = 2

        @cute.struct
        class SharedStorage:
            pipeline_bars: cute.struct.MemRange[cutlass.Int64, 2 * num_stages]
            src_x_ptrs: cute.struct.MemRange[cutlass.Int64, MoEDispatch.kMaxWorldSize]
            tma_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, hidden_size * num_stages],
                MoEDispatch.buffer_align_bytes,
            ]

        self.kernel(
            input_ptrs,
            token_src_rank_topk_and_indices,
            recv_token_count,
            output,
            hidden_size,
            nbytes_per_token,
            num_stages,
            rank,
            world_size,
            SharedStorage,
            recv_expert_counts,
            expert_signals,
            expert_signal_counters,
            experts_per_rank,
            expert_alignment,
            enable_signals,
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
        input_ptrs: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        gO: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        nbytes_per_token: cutlass.Constexpr[int],
        num_stages: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
        recv_expert_counts: cute.Tensor,
        expert_signals: cute.Tensor,
        expert_signal_counters: cute.Tensor,
        experts_per_rank: cutlass.Constexpr[int],
        expert_alignment: cutlass.Constexpr[int],
        enable_signals: cutlass.Constexpr[int],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()

        linear_block_idx = bidx
        num_blocks = gdimx
        lane_idx = cute.arch.lane_idx()
        warp_batch = cutlass.Int32(self.num_threads_per_warp)

        dtype = gO.element_type
        # 1D bulk copy atoms (per-token cp.async.bulk). ``num_bits_per_copy``
        # is the full token width so a single elected thread issues one
        # bulk transfer per token.
        bulk_g2s_atom = cute.make_copy_atom(
            cpasync.CopyBulkG2SOp(),
            dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        bulk_s2g_atom = cute.make_copy_atom(
            cpasync.CopyBulkS2GOp(),
            dtype,
            num_bits_per_copy=nbytes_per_token * 8,
        )
        row_layout = cute.make_layout(hidden_size)

        smem = cute_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        sBuf = storage.tma_buffer.get_tensor(cute.make_layout(
            (num_stages, hidden_size),
            stride=(hidden_size, 1),
        ))
        smem_input_ptrs = storage.src_x_ptrs.get_tensor(cute.make_layout(self.kMaxWorldSize))

        pipeline = PipelineTmaAsync.create(
            num_stages=num_stages,
            producer_group=CooperativeGroup(Agent.Thread, size=1),
            consumer_group=CooperativeGroup(Agent.Thread, size=1),
            tx_count=nbytes_per_token,
            barrier_storage=storage.pipeline_bars.data_ptr(),
        )

        if tidx < world_size:
            smem_input_ptrs[tidx] = input_ptrs[tidx]
        cute.arch.barrier()

        num_recv_tokens = recv_token_count[rank]

        mainloop_producer = pipeline.make_producer()
        mainloop_consumer_read = pipeline.make_consumer()
        mainloop_consumer_release = mainloop_consumer_read.clone()

        if warp_idx == self.producer_warp_id:
            token_idx = linear_block_idx
            if cutlass.const_expr(enable_signals):
                expert_start = cutlass.Int32(0)
                expert_idx = cutlass.Int32(0)
                while expert_idx < experts_per_rank:
                    expert_tokens = recv_expert_counts[expert_idx]
                    expert_end = expert_start + expert_tokens
                    batch_base = expert_start + linear_block_idx
                    while batch_base < expert_end:
                        lane_token_idx = batch_base + lane_idx * num_blocks
                        encoded = cutlass.Int64(-1)
                        if lane_token_idx < expert_end:
                            encoded = token_src_rank_topk_and_indices[lane_token_idx]
                        src_token_idx, src_rank, _ = (decode_token_src_rank_topk_and_indices(encoded))

                        for lane in cutlass.range(self.num_threads_per_warp, unroll_full=True):
                            b_src_token_idx = cute.arch.shuffle_sync(src_token_idx, lane)
                            b_src_rank = cute.arch.shuffle_sync(src_rank, lane)

                            if ((b_src_rank + cutlass.Int32(1)) != cutlass.Int32(0)):
                                handle = mainloop_producer.acquire_and_advance()

                                remote_base_i64 = smem_input_ptrs[b_src_rank]
                                offset_i64 = (cutlass.Int64(b_src_token_idx) * cutlass.Int64(nbytes_per_token))
                                remote_row_i64 = remote_base_i64 + offset_i64

                                with cute.arch.elect_one():
                                    sDst = cute.make_tensor(
                                        sBuf.iterator + handle.index * hidden_size,
                                        row_layout,
                                    )
                                    gSrc = cute.make_tensor(
                                        cute.make_ptr(
                                            dtype,
                                            remote_row_i64,
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
                        batch_base += num_blocks * warp_batch

                    aligned_tokens = expert_tokens
                    if cutlass.const_expr(expert_alignment > 1):
                        aligned_tokens = ((expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                    expert_start += aligned_tokens
                    expert_idx += cutlass.Int32(1)
            else:
                batch_base = token_idx
                while batch_base < num_recv_tokens:
                    lane_token_idx = batch_base + lane_idx * num_blocks
                    encoded = cutlass.Int64(-1)
                    if lane_token_idx < num_recv_tokens:
                        encoded = token_src_rank_topk_and_indices[lane_token_idx]
                    src_token_idx, src_rank, _ = (decode_token_src_rank_topk_and_indices(encoded))

                    for lane in cutlass.range(self.num_threads_per_warp, unroll_full=True):
                        b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)
                        b_src_token_idx = cute.arch.shuffle_sync(src_token_idx, lane)
                        b_src_rank = cute.arch.shuffle_sync(src_rank, lane)

                        if (b_src_rank + cutlass.Int32(1)) != cutlass.Int32(0):
                            handle = mainloop_producer.acquire_and_advance()

                            remote_base_i64 = smem_input_ptrs[b_src_rank]
                            offset_i64 = (cutlass.Int64(b_src_token_idx) * cutlass.Int64(nbytes_per_token))
                            remote_row_i64 = remote_base_i64 + offset_i64

                            with cute.arch.elect_one():
                                sDst = cute.make_tensor(
                                    sBuf.iterator + handle.index * hidden_size,
                                    row_layout,
                                )
                                gSrc = cute.make_tensor(
                                    cute.make_ptr(
                                        dtype,
                                        remote_row_i64,
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
                    batch_base += num_blocks * warp_batch

            mainloop_producer.tail()

        elif warp_idx == self.consumer_warp_id:
            pipe_cnt_rt = cutlass.Int32(self.pipe_cnt)

            if cutlass.const_expr(enable_signals):
                expert_start = cutlass.Int32(0)
                expert_idx = cutlass.Int32(0)
                while expert_idx < experts_per_rank:
                    cnt = cutlass.Int32(0)
                    expert_tokens = recv_expert_counts[expert_idx]
                    expert_end = expert_start + expert_tokens
                    batch_base = expert_start + linear_block_idx
                    while batch_base < expert_end:
                        lane_token_idx = batch_base + lane_idx * num_blocks
                        encoded_c = cutlass.Int64(-1)
                        if lane_token_idx < expert_end:
                            encoded_c = token_src_rank_topk_and_indices[lane_token_idx]
                        _, src_rank_c, _ = (decode_token_src_rank_topk_and_indices(encoded_c))

                        for lane in cutlass.range(self.num_threads_per_warp, unroll_full=True):
                            b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)
                            b_src_rank_c = cute.arch.shuffle_sync(src_rank_c, lane)

                            if ((b_src_rank_c + cutlass.Int32(1)) != cutlass.Int32(0)):
                                handle = mainloop_consumer_read.wait_and_advance()

                                with cute.arch.elect_one():
                                    sSrc = cute.make_tensor(
                                        sBuf.iterator + handle.index * hidden_size,
                                        row_layout,
                                    )
                                    gDst = cute.make_tensor(
                                        gO.iterator + b_token_idx * hidden_size,
                                        row_layout,
                                    )
                                    cute.copy(bulk_s2g_atom, sSrc, gDst)
                                    cute.arch.cp_async_bulk_commit_group()
                                    cute.arch.cp_async_bulk_wait_group(self.pipe_cnt, read=True)

                                if cnt >= pipe_cnt_rt:
                                    mainloop_consumer_release.release()
                                    mainloop_consumer_release.advance()

                                cnt += cutlass.Int32(1)
                        batch_base += num_blocks * warp_batch

                    with cute.arch.elect_one():
                        cute.arch.cp_async_bulk_wait_group(0, read=False)

                    for _i in cutlass.range(self.pipe_cnt, unroll_full=True):
                        if cnt > cutlass.Int32(0):
                            mainloop_consumer_release.release()
                            mainloop_consumer_release.advance()
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
                        aligned_tokens = ((expert_tokens + expert_alignment - 1) // expert_alignment) * expert_alignment
                    expert_start += aligned_tokens
                    expert_idx += cutlass.Int32(1)
            else:
                cnt = cutlass.Int32(0)
                batch_base = linear_block_idx
                while batch_base < num_recv_tokens:
                    lane_token_idx = batch_base + lane_idx * num_blocks
                    encoded_c = cutlass.Int64(-1)
                    if lane_token_idx < num_recv_tokens:
                        encoded_c = token_src_rank_topk_and_indices[lane_token_idx]
                    _, src_rank_c, _ = (decode_token_src_rank_topk_and_indices(encoded_c))

                    for lane in cutlass.range(self.num_threads_per_warp, unroll_full=True):
                        b_token_idx = (batch_base + cutlass.Int32(lane) * num_blocks)
                        b_src_rank_c = cute.arch.shuffle_sync(src_rank_c, lane)

                        if (b_src_rank_c + cutlass.Int32(1)) != cutlass.Int32(0):
                            handle = mainloop_consumer_read.wait_and_advance()

                            with cute.arch.elect_one():
                                sSrc = cute.make_tensor(
                                    sBuf.iterator + handle.index * hidden_size,
                                    row_layout,
                                )
                                gDst = cute.make_tensor(
                                    gO.iterator + b_token_idx * hidden_size,
                                    row_layout,
                                )
                                cute.copy(bulk_s2g_atom, sSrc, gDst)
                                cute.arch.cp_async_bulk_commit_group()
                                cute.arch.cp_async_bulk_wait_group(self.pipe_cnt, read=True)

                            if cnt >= pipe_cnt_rt:
                                mainloop_consumer_release.release()
                                mainloop_consumer_release.advance()

                            cnt += cutlass.Int32(1)
                    batch_base += num_blocks * warp_batch

                with cute.arch.elect_one():
                    cute.arch.cp_async_bulk_wait_group(0, read=False)

                for _i in cutlass.range(self.pipe_cnt, unroll_full=True):
                    if cnt > cutlass.Int32(0):
                        mainloop_consumer_release.release()
                        mainloop_consumer_release.advance()
                        cnt -= cutlass.Int32(1)
