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
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import Agent, CooperativeGroup, PipelineTmaAsync

from .cutedsl_utils import decode_token_src_rank_topk_and_indices


class MoECombinePush:
    """Warp-specialized push-mode intranode MoE combine."""

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
        fixed_bytes = MoECombinePush.kMaxWorldSize * 8
        alignment_overhead = MoECombinePush.buffer_align_bytes
        per_stage = hidden_size * dtype_bytes + 2 * 8
        available = max_smem_bytes - fixed_bytes - alignment_overhead
        return max(2, min(5, available // per_stage))

    @property
    def num_threads(self) -> int:
        return self.num_warps * self.num_threads_per_warp

    @cute.jit
    def __call__(
        self,
        input: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        output_ptrs: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        num_sms: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        dtype = input.element_type
        dtype_bytes = dtype.width // 8
        nbytes_per_token = hidden_size * dtype_bytes
        num_stages = MoECombinePush.compute_num_stages(hidden_size, dtype_bytes, self.max_smem_bytes)
        self.pipe_cnt = 2

        @cute.struct
        class SharedStorage:
            pipeline_bars: cute.struct.MemRange[cutlass.Int64, 2 * num_stages]
            output_ptrs: cute.struct.MemRange[cutlass.Int64, MoECombinePush.kMaxWorldSize]
            tma_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, hidden_size * num_stages],
                MoECombinePush.buffer_align_bytes,
            ]

        self.kernel(
            input,
            token_src_rank_topk_and_indices,
            recv_token_count,
            output_ptrs,
            hidden_size,
            nbytes_per_token,
            num_stages,
            topk,
            rank,
            world_size,
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
        gI: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        output_ptrs: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        nbytes_per_token: cutlass.Constexpr[int],
        num_stages: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = tidx % self.num_threads_per_warp
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        lane_idx = cute.arch.lane_idx()
        warp_batch = cutlass.Int32(self.num_threads_per_warp)

        dtype = gI.element_type
        # 1D bulk copy atoms (per-row cp.async.bulk). ``num_bits_per_copy``
        # is the full row width so a single elected thread issues one
        # bulk transfer per row.
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
        smem_output_ptrs = storage.output_ptrs.get_tensor(cute.make_layout(self.kMaxWorldSize))

        pipeline = PipelineTmaAsync.create(
            num_stages=num_stages,
            producer_group=CooperativeGroup(Agent.Thread, size=1),
            consumer_group=CooperativeGroup(Agent.Thread, size=1),
            tx_count=nbytes_per_token,
            barrier_storage=storage.pipeline_bars.data_ptr(),
        )

        if tidx < world_size:
            smem_output_ptrs[tidx] = output_ptrs[tidx]
        cute.arch.barrier()

        num_recv_tokens = recv_token_count[rank]
        mainloop_producer = pipeline.make_producer()
        mainloop_consumer_read = pipeline.make_consumer()
        mainloop_consumer_release = mainloop_consumer_read.clone()

        batch_base = cutlass.Int32(0)
        if warp_idx == self.producer_warp_id:
            batch_base = bidx
            while batch_base < num_recv_tokens:
                lane_token_idx = batch_base + lane_idx * gdimx
                encoded = cutlass.Int64(-1)
                if lane_token_idx < num_recv_tokens:
                    encoded = token_src_rank_topk_and_indices[lane_token_idx]
                _, dst_rank, _ = decode_token_src_rank_topk_and_indices(encoded)

                for lane in cutlass.range(self.num_threads_per_warp, unroll_full=True):
                    b_token_idx = batch_base + cutlass.Int32(lane) * gdimx
                    b_dst_rank = cute.arch.shuffle_sync(dst_rank, lane)
                    if (b_dst_rank + cutlass.Int32(1)) != cutlass.Int32(0):
                        handle = mainloop_producer.acquire_and_advance()
                        with cute.arch.elect_one():
                            sDst = cute.make_tensor(
                                sBuf.iterator + handle.index * hidden_size,
                                row_layout,
                            )
                            gSrc = cute.make_tensor(
                                gI.iterator + b_token_idx * hidden_size,
                                row_layout,
                            )
                            cute.copy(
                                bulk_g2s_atom,
                                gSrc,
                                sDst,
                                mbar_ptr=handle.barrier,
                            )
                        handle.commit()
                batch_base += gdimx * warp_batch

            mainloop_producer.tail()

        elif warp_idx == self.consumer_warp_id:
            pipe_cnt_rt = cutlass.Int32(self.pipe_cnt)
            cnt = cutlass.Int32(0)
            batch_base = bidx
            while batch_base < num_recv_tokens:
                lane_token_idx = batch_base + lane_idx * gdimx
                encoded = cutlass.Int64(-1)
                if lane_token_idx < num_recv_tokens:
                    encoded = token_src_rank_topk_and_indices[lane_token_idx]
                src_token_idx, dst_rank, src_topk_idx = (decode_token_src_rank_topk_and_indices(encoded))

                for lane in cutlass.range(self.num_threads_per_warp, unroll_full=True):
                    b_dst_rank = cute.arch.shuffle_sync(dst_rank, lane)
                    b_src_token_idx = cute.arch.shuffle_sync(src_token_idx, lane)
                    b_src_topk_idx = cute.arch.shuffle_sync(src_topk_idx, lane)
                    if (b_dst_rank + cutlass.Int32(1)) != cutlass.Int32(0):
                        handle = mainloop_consumer_read.wait_and_advance()

                        with cute.arch.elect_one():
                            sSrc = cute.make_tensor(
                                sBuf.iterator + handle.index * hidden_size,
                                row_layout,
                            )
                            remote_base_i64 = smem_output_ptrs[b_dst_rank]
                            dst_slot = b_src_token_idx * topk + b_src_topk_idx
                            remote_row_i64 = (remote_base_i64 +
                                              cutlass.Int64(dst_slot) * cutlass.Int64(nbytes_per_token))
                            gDst = cute.make_tensor(
                                cute.make_ptr(
                                    dtype,
                                    remote_row_i64,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                ),
                                row_layout,
                            )
                            cute.copy(bulk_s2g_atom, sSrc, gDst)
                            cute.arch.cp_async_bulk_commit_group()
                            cute.arch.cp_async_bulk_wait_group(self.pipe_cnt, read=True)

                        if cnt >= pipe_cnt_rt:
                            mainloop_consumer_release.release()
                            mainloop_consumer_release.advance()
                        cnt += cutlass.Int32(1)
                batch_base += gdimx * warp_batch

            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_wait_group(0, read=False)

            for _i in cutlass.range(self.pipe_cnt, unroll_full=True):
                if cnt > cutlass.Int32(0):
                    mainloop_consumer_release.release()
                    mainloop_consumer_release.advance()
                    cnt -= cutlass.Int32(1)


class MoECombineTilePush:
    """Tile TMA-load + 4-warp SIMT-store push-mode combine.

    Full tiles use one bulk TMA load from local input into SMEM. A consumer
    warp-group then scalar-stores elements to remote symmetric staging slots.
    The final partial tile falls back to direct SIMT loads to avoid OOB reads.
    """

    kMaxWorldSize: int = 128
    num_threads_per_warp: int = 32
    producer_warp_id: int = 0
    consumer_warps: int = 4
    num_warps: int = 1 + consumer_warps
    buffer_align_bytes: int = 128
    # Single stage: depth comes from CTA-level parallelism, not intra-CTA
    # double buffering. Keeps SMEM at ~32 KB / CTA so registers and warps
    # bound occupancy.
    num_stages: int = 1

    @property
    def num_threads(self) -> int:
        return self.num_warps * self.num_threads_per_warp

    @cute.jit
    def __call__(
        self,
        input: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        output_ptrs: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        num_sms: cutlass.Int32,
        tile_m: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        stream: cuda.CUstream,
    ):
        dtype = input.element_type
        dtype_bytes = dtype.width // 8
        nbytes_per_token = hidden_size * dtype_bytes
        tile_bytes = tile_m * tile_n * dtype_bytes
        assert tile_n % 8 == 0, "2D combine tile_n must be a multiple of 8 bf16 elements"
        assert hidden_size % tile_n == 0, ("2D combine requires hidden_size to be a multiple of tile_n")
        tma_smem_layout = cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            input,
            tma_smem_layout,
            (tile_m, tile_n),
        )

        num_stages = MoECombineTilePush.num_stages

        @cute.struct
        class SharedStorage:
            pipeline_bars: cute.struct.MemRange[cutlass.Int64, 2 * num_stages]
            output_ptrs: cute.struct.MemRange[cutlass.Int64, MoECombineTilePush.kMaxWorldSize]
            tma_buffer: cute.struct.Align[
                cute.struct.MemRange[dtype, num_stages * tile_m * tile_n],
                MoECombineTilePush.buffer_align_bytes,
            ]

        smem_bytes = SharedStorage.size_in_bytes()
        grid_dim = num_sms
        self.kernel(
            tma_atom,
            tma_tensor,
            input,
            token_src_rank_topk_and_indices,
            recv_token_count,
            output_ptrs,
            hidden_size,
            nbytes_per_token,
            tile_bytes,
            dtype_bytes,
            topk,
            rank,
            world_size,
            tile_m,
            tile_n,
            SharedStorage,
        ).launch(
            grid=(grid_dim, 1, 1),
            block=(self.num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom: cute.CopyAtom,
        tma_tensor: cute.Tensor,
        gI: cute.Tensor,
        token_src_rank_topk_and_indices: cute.Tensor,
        recv_token_count: cute.Tensor,
        output_ptrs: cute.Tensor,
        hidden_size: cutlass.Constexpr[int],
        nbytes_per_token: cutlass.Constexpr[int],
        tile_bytes: cutlass.Constexpr[int],
        dtype_bytes: cutlass.Constexpr[int],
        topk: cutlass.Constexpr[int],
        rank: cutlass.Constexpr[int],
        world_size: cutlass.Constexpr[int],
        tile_m: cutlass.Constexpr[int],
        tile_n: cutlass.Constexpr[int],
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, _, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()

        num_stages = self.num_stages
        smem = cute_utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sStaged = storage.tma_buffer.get_tensor(
            cute.make_layout(
                (num_stages, tile_m, tile_n),
                stride=(tile_m * tile_n, tile_n, 1),
            ))
        smem_output_ptrs = storage.output_ptrs.get_tensor(cute.make_layout(self.kMaxWorldSize))

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        # One arrival per consumer warp (lane 0). Intra-warp lockstep makes
        # this safe and keeps the SMEM atomic count to 4 (vs 128).
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.consumer_warps)
        tma_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=num_stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tile_bytes,
            barrier_storage=storage.pipeline_bars.data_ptr(),
        )

        if tidx < world_size:
            smem_output_ptrs[tidx] = output_ptrs[tidx]
        cute.arch.barrier()

        # Warp-coalesced SIMT stores: each warp owns ``cpy_m`` rows; 32 lanes
        # issue ``atoms_per_row`` 128 b stores per row, forming a contiguous
        # NVLink train and amortising the mbarrier traffic.
        atom_v: cutlass.Constexpr[int] = 8  # 128b / bf16 width
        assert tile_n % (self.num_threads_per_warp * atom_v) == 0, (
            "tile_n must be a positive multiple of num_threads_per_warp "
            "* atom_v (e.g. 256 = 32 * 8 bf16 elements) for warp-coalesced "
            "stores")
        atoms_per_row: cutlass.Constexpr[int] = (tile_n // (self.num_threads_per_warp * atom_v))
        # tile_m may be < consumer threads: shrinks per-stage SMEM and lets
        # 2 CTAs reside on an SM, doubling the warp pool for the LSU.
        # Lane mapping: ``lane k of warp w -> row (k * consumer_warps + w)``.
        cpy_m: cutlass.Constexpr[int] = tile_m // self.consumer_warps
        assert tile_m == cpy_m * self.consumer_warps, ("tile_m must be a positive multiple of consumer_warps")
        assert cpy_m <= self.num_threads_per_warp, ("tile_m / consumer_warps must be <= 32; the broadcast lanes "
                                                    "per warp own up to one row each")

        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            gI.element_type,
            num_bits_per_copy=128,
            memory_scope=cute.nvgpu.MemoryScope.GPU,
            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
        )
        # Per-warp 1D tiled-copy: 32 lanes cooperatively store one row of
        # ``tile_n`` elements, each lane handling ``atom_v`` contiguous values.
        row_tiled_copy = cute.make_tiled_copy_tv(
            store_atom,
            cute.make_layout(self.num_threads_per_warp),
            cute.make_layout(atom_v),
        )

        num_recv_tokens = recv_token_count[rank]
        num_token_tiles = (num_recv_tokens + tile_m - 1) // tile_m
        num_hidden_tiles: cutlass.Constexpr[int] = hidden_size // tile_n
        ctile_idx = bidx
        nbytes_per_token_i64 = cutlass.Int64(nbytes_per_token)
        tile_n_bytes_i64 = cutlass.Int64(tile_n * dtype_bytes)

        # (num_stages, tile_m, tile_n) SMEM viewed as a canonical
        # (tile_m, tile_n, num_stages)-major tensor for ``tma_partition``.
        sStaged_tnm = cute.make_tensor(
            sStaged.iterator,
            cute.make_layout(
                (tile_m, tile_n, num_stages),
                stride=(tile_n, 1, tile_m * tile_n),
            ),
        )

        if warp_idx == self.producer_warp_id:
            producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, num_stages)
            sStaged_for_tma = cute.group_modes(sStaged_tnm, 0, 2)
            gI_tiles = cute.zipped_divide(tma_tensor, (tile_m, tile_n))
            gI_for_tma = cute.group_modes(gI_tiles, 0, 1)
            tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
                tma_atom,
                0,
                cute.make_layout(1),
                sStaged_for_tma,
                gI_for_tma,
            )
            # Constexpr inner sweep over hidden tiles unrolls into
            # ``num_hidden_tiles`` back-to-back TMA issues per token tile.
            while ctile_idx < num_token_tiles:
                for h in cutlass.range_constexpr(num_hidden_tiles):
                    tma_pipeline.producer_acquire(producer_state)
                    cute.copy(
                        tma_atom,
                        tXgX[(None, (ctile_idx, h))],
                        tXsX[(None, producer_state.index)],
                        tma_bar_ptr=tma_pipeline.producer_get_barrier(producer_state),
                    )
                    tma_pipeline.producer_commit(producer_state)
                    producer_state.advance()
                ctile_idx += gdimx

            tma_pipeline.producer_tail(producer_state)

        else:
            consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, num_stages)
            consumer_tid = tidx - self.num_threads_per_warp
            consumer_lane = consumer_tid % self.num_threads_per_warp
            consumer_warp_idx = consumer_tid // self.num_threads_per_warp
            row_thr_copy = row_tiled_copy.get_slice(consumer_lane)

            while ctile_idx < num_token_tiles:
                tile_start = ctile_idx * tile_m
                # Sentinel ``-1`` for OOB / padding lanes; the per-row
                # predicate below skips their cute.copy.
                my_dst_rank = cutlass.Int32(-1)
                my_dst_slot = cutlass.Int32(0)
                if consumer_lane < cpy_m:
                    my_token_idx = tile_start + (consumer_lane * self.consumer_warps + consumer_warp_idx)
                    if my_token_idx < num_recv_tokens:
                        encoded = token_src_rank_topk_and_indices[my_token_idx]
                        src_token_idx, dr, src_topk_idx = (decode_token_src_rank_topk_and_indices(encoded))
                        my_dst_rank = dr
                        my_dst_slot = src_token_idx * topk + src_topk_idx

                # Pre-broadcast per-row dst addr + validity. Column axis
                # never needs masking (asserted above), so the predicate
                # uses a (cpy_m, atoms_per_row):(1, 0) broadcast fragment.
                bk_row_base = [cutlass.Int64(0)] * cpy_m
                pred_tile = cute.make_fragment(
                    cute.make_layout((cpy_m, atoms_per_row), stride=(1, 0)),
                    cutlass.Boolean,
                )
                for k in cutlass.range_constexpr(cpy_m):
                    bk_dst_rank = cute.arch.shuffle_sync(my_dst_rank, k)
                    bk_dst_slot = cute.arch.shuffle_sync(my_dst_slot, k)
                    # ``bk_dst_rank == -1`` marks an OOB lane or dropped
                    # padding slot. Index rank 0 for invalid lanes so the
                    # pointer-table load is always in-bounds; ``pred_tile``
                    # still gates the store.
                    bk_pred = (bk_dst_rank + cutlass.Int32(1)) != cutlass.Int32(0)
                    safe_bk_dst_rank = cutlass.Int32(0)
                    safe_bk_dst_slot = cutlass.Int32(0)
                    if bk_pred:
                        safe_bk_dst_rank = bk_dst_rank
                        safe_bk_dst_slot = bk_dst_slot
                    bk_remote_base = smem_output_ptrs[safe_bk_dst_rank]
                    bk_row_base[k] = (bk_remote_base + cutlass.Int64(safe_bk_dst_slot) * nbytes_per_token_i64)
                    pred_tile[(k, 0)] = bk_pred

                for h in cutlass.range_constexpr(num_hidden_tiles):
                    tma_pipeline.consumer_wait(consumer_state)
                    hidden_byte_offset_i64 = (tile_n_bytes_i64 * cutlass.Int64(h))

                    # Rows go to different peers (no uniform M-stride), so we
                    # issue cpy_m per-row cute.copy calls with a broadcast
                    # predicate slice instead of a single 2D copy.
                    for k in cutlass.range_constexpr(cpy_m):
                        sStaged_row = sStaged_tnm[(
                            k * self.consumer_warps + consumer_warp_idx,
                            None,
                            consumer_state.index,
                        )]
                        gOut_row = cute.make_tensor(
                            cute.make_ptr(
                                gI.element_type,
                                bk_row_base[k] + hidden_byte_offset_i64,
                                cute.AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            cute.make_layout(tile_n),
                        )
                        tCsT_row = row_thr_copy.partition_S(sStaged_row)
                        tCgT_row = row_thr_copy.partition_D(gOut_row)
                        cute.copy(
                            store_atom,
                            tCsT_row,
                            tCgT_row,
                            pred=pred_tile[(k, None)],
                        )

                    cute.arch.sync_warp()
                    if consumer_lane == 0:
                        tma_pipeline.sync_object_empty.arrive(
                            consumer_state.index,
                            tma_pipeline.consumer_mask,
                        )
                    consumer_state.advance()

                ctile_idx += gdimx
