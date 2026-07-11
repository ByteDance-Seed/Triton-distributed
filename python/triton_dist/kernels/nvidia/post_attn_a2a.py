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
"""Warp-specialized TMA kernel for post-attention AllToAll.

Aligned with Flux ``PostAttnA2AKernel`` (``use_tma_store`` path):

  - Warp 0 (producer): TMA G2S loads from local input -> shared-memory ring.
  - Warp 1 (consumer): TMA S2G stores from shared-memory -> remote NVSHMEM output.
  - Multi-stage pipeline with mbarrier synchronisation.
  - Producer and consumer run independent loops (true warp specialization).
  - Consumer single-loop: prefill + steady-state merged via ``cnt >= PIPE_CNT``.

Data movement::

    input  (seq_len, local_nh, hd)  [local]
       -> NVSHMEM push ->
    output (local_seq, nh, hd)      [remote, per-rank column slice]

Launch with ``num_warps=2``.
"""
import triton.language as tl
from triton.language import core as tlc

import triton_dist
import triton_dist.language as dl
from triton_dist.language.smem_ops import allocate_smem
from triton_dist.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.language.extra.cuda.tma_language import (
    cp_async_bulk_wait_group,
    elect_one_sync,
)
from triton_dist.language.tma import (
    ELEM_BF16,
    SWIZZLE_NONE,
    PipelineState,
    TmaPipeline,
    create_tmap_2d,
)


@triton_dist.jit(do_not_specialize=["rank", "sp_rank"])
def kernel_post_attn_a2a(
    src_ptr,
    dst_ptr,
    tmap_scratch_ptr,
    local_seq_len,
    local_head: tl.constexpr,
    head_dim: tl.constexpr,
    sp_size: tl.constexpr,
    rank,
    sp_rank,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_SM: tl.constexpr,
    PIPE_CNT: tl.constexpr,
):
    pid = tl.program_id(0)
    _t = tid(0)
    warp_idx = _t // 32

    SRC_N: tl.constexpr = local_head * head_dim
    DST_N: tl.constexpr = SRC_N * sp_size
    TILE_BYTES: tl.constexpr = BM * BN * 2

    tl.static_assert(NUM_SM % sp_size == 0)
    NUM_SM_PER_RANK: tl.constexpr = NUM_SM // sp_size
    dst_sp_rank = pid // NUM_SM_PER_RANK
    pid_in_rank = pid % NUM_SM_PER_RANK

    rank_offset = rank - sp_rank
    remote_rank = dst_sp_rank + rank_offset
    if sp_size == 1:
        remote_dst_ptr = dst_ptr
    else:
        remote_dst_ptr = dl.symm_at(dst_ptr, remote_rank)

    # -- shared memory & pipeline --
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bufs = allocate_smem(tl.bfloat16, [NUM_STAGES, BM, BN])
    smem_ready = allocate_smem(tl.int64, [NUM_STAGES])
    smem_empty = allocate_smem(tl.int64, [NUM_STAGES])
    pipeline = TmaPipeline(smem_ready, smem_empty, smem_bufs, NUM_STAGES)

    # -- device-side tensormaps --
    in_stride = tl.cast(SRC_N * 2, tl.int64)
    tmap_global_in = tl.cast(tmap_scratch_ptr + pid * 256, tl.uint64, bitcast=True)
    in_base = src_ptr + dst_sp_rank * local_seq_len * SRC_N
    in_tmap = create_tmap_2d(
        smem_tmap,
        tmap_global_in,
        in_base,
        SRC_N,
        local_seq_len,
        in_stride,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_NONE,
    )

    out_stride = tl.cast(DST_N * 2, tl.int64)
    tmap_global_out = tl.cast(tmap_scratch_ptr + pid * 256 + 128, tl.uint64, bitcast=True)
    out_base = remote_dst_ptr + sp_rank * SRC_N
    out_tmap = create_tmap_2d(
        smem_tmap,
        tmap_global_out,
        out_base,
        SRC_N,
        local_seq_len,
        out_stride,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_NONE,
    )

    one_c = tl.cast(1, tl.int32)
    if _t == 0:
        pipeline.init_barriers(one_c)
    __syncthreads()

    num_tile_m = tl.cdiv(local_seq_len, BM)
    tl.static_assert(SRC_N % BN == 0)
    num_tile_n: tl.constexpr = SRC_N // BN

    # ===================================================================
    # Warp 0 — PRODUCER (TMA G2S from local input)
    # ===================================================================
    if warp_idx == 0:
        p_state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        for tile_m in range(pid_in_rank, num_tile_m, NUM_SM_PER_RANK):
            for tile_n in tl.static_range(num_tile_n):
                pipeline.producer_acquire(p_state)
                _e = elect_one_sync()
                if _e != 0:
                    pipeline.producer_tma_load(p_state, in_tmap, tile_n * BN, tile_m * BM, TILE_BYTES)
                p_state = p_state.advance()

    # ===================================================================
    # Warp 1 — CONSUMER (TMA S2G to remote output)
    # Single loop: first PIPE_CNT iterations are prefill (no release),
    # subsequent iterations release the oldest in-flight store.
    # ===================================================================
    else:  # warp_idx == 1
        cr_state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        ce_state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        cnt = 0

        for tile_m in range(pid_in_rank, num_tile_m, NUM_SM_PER_RANK):
            for tile_n in tl.static_range(num_tile_n):
                pipeline.consumer_wait(cr_state)
                _e = elect_one_sync()
                if _e != 0:
                    pipeline.consumer_tma_store(cr_state, out_tmap, tile_n * BN, tile_m * BM)
                if cnt >= PIPE_CNT:
                    if _e != 0:
                        cp_async_bulk_wait_group(PIPE_CNT, read=tlc.constexpr(False))
                        pipeline.consumer_release(ce_state)
                    ce_state = ce_state.advance()
                cr_state = cr_state.advance()
                cnt = cnt + 1

        # -- tail: drain remaining in-flight stores --
        _e = elect_one_sync()
        if _e != 0:
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(False))
        for _s in tl.static_range(PIPE_CNT):
            if _e != 0:
                pipeline.consumer_release(ce_state)
            ce_state = ce_state.advance()

    # -- cleanup --
    __syncthreads()
    if _t == 0:
        pipeline.invalidate_barriers()
