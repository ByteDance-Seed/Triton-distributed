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
"""Warp-specialized TMA kernel for pre-attention AllToAll (QKV unpack).

Aligned with Flux ``PreAttnQKVPackA2AKernel`` (TMA-store path):

  - Warp 0 (producer): TMA G2S loads from remote NVSHMEM input -> smem ring.
  - Warp 1 (consumer): TMA S2G stores from smem -> local Q / K / V outputs.
  - Multi-stage pipeline with mbarrier synchronisation.
  - Producer and consumer run independent loops (true warp specialization).
  - Consumer single-loop: prefill + steady-state merged via ``cnt >= PIPE_CNT``.

Data movement::

    input  (local_seq, total_nh, hd)   [remote, via NVSHMEM pull]
       -> unpack Q/K/V ->
    Q      (seq_len, local_q_nh, hd)   [local]
    K      (seq_len, local_k_nh, hd)   [local]
    V      (seq_len, local_v_nh, hd)   [local]

Tile-column boundaries are aligned so each tile maps to exactly one of Q/K/V.

For single-tensor mode (no K/V split), set k_nheads = v_nheads = 0.

Launch with ``num_warps=2``.
"""
import triton
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


@triton.jit
def _dispatch_qkv_store(pipeline: TmaPipeline, state: PipelineState, q_tmap, k_tmap, v_tmap, tile_n, tile_m,
                        BN: tl.constexpr, BM: tl.constexpr, NUM_TILE_N_Q: tl.constexpr, NUM_TILE_N_K: tl.constexpr):
    """Route a consumer TMA S2G store to the correct Q, K, or V tensormap."""
    if tile_n < NUM_TILE_N_Q:
        pipeline.consumer_tma_store(state, q_tmap, tile_n * BN, tile_m * BM)
    elif tile_n < NUM_TILE_N_Q + NUM_TILE_N_K:
        pipeline.consumer_tma_store(state, k_tmap, (tile_n - NUM_TILE_N_Q) * BN, tile_m * BM)
    else:
        pipeline.consumer_tma_store(state, v_tmap, (tile_n - NUM_TILE_N_Q - NUM_TILE_N_K) * BN, tile_m * BM)


@triton_dist.jit(do_not_specialize=["rank", "sp_rank"])
def kernel_pre_attn_a2a(
    input_buf_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    tmap_scratch_ptr,
    local_seq_len,
    q_nheads: tl.constexpr,
    k_nheads: tl.constexpr,
    v_nheads: tl.constexpr,
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

    LOCAL_Q_NH: tl.constexpr = q_nheads // sp_size
    LOCAL_K_NH: tl.constexpr = k_nheads // sp_size if k_nheads > 0 else 0
    LOCAL_V_NH: tl.constexpr = v_nheads // sp_size if v_nheads > 0 else 0
    TOTAL_NH: tl.constexpr = q_nheads + k_nheads + v_nheads

    INPUT_N: tl.constexpr = TOTAL_NH * head_dim
    Q_OUT_N: tl.constexpr = LOCAL_Q_NH * head_dim
    K_OUT_N: tl.constexpr = LOCAL_K_NH * head_dim if LOCAL_K_NH > 0 else 0
    V_OUT_N: tl.constexpr = LOCAL_V_NH * head_dim if LOCAL_V_NH > 0 else 0
    TILE_BYTES: tl.constexpr = BM * BN * 2

    tl.static_assert(NUM_SM % sp_size == 0)
    NUM_SM_PER_RANK: tl.constexpr = NUM_SM // sp_size
    src_sp_rank = pid // NUM_SM_PER_RANK
    pid_in_rank = pid % NUM_SM_PER_RANK

    rank_offset = rank - sp_rank
    remote_rank = src_sp_rank + rank_offset
    if sp_size == 1:
        remote_input_ptr = input_buf_ptr
    else:
        remote_input_ptr = dl.symm_at(input_buf_ptr, remote_rank)

    q_input_col = sp_rank * LOCAL_Q_NH * head_dim
    k_input_col = q_nheads * head_dim + sp_rank * LOCAL_K_NH * head_dim
    v_input_col = (q_nheads + k_nheads) * head_dim + sp_rank * LOCAL_V_NH * head_dim

    NUM_TILE_N_Q: tl.constexpr = Q_OUT_N // BN
    NUM_TILE_N_K: tl.constexpr = K_OUT_N // BN if K_OUT_N > 0 else 0
    NUM_TILE_N_V: tl.constexpr = V_OUT_N // BN if V_OUT_N > 0 else 0
    NUM_TILE_N: tl.constexpr = NUM_TILE_N_Q + NUM_TILE_N_K + NUM_TILE_N_V

    num_tile_m = tl.cdiv(local_seq_len, BM)
    total_tiles = num_tile_m * NUM_TILE_N

    # -- shared memory & pipeline --
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bufs = allocate_smem(tl.bfloat16, [NUM_STAGES, BM, BN])
    smem_ready = allocate_smem(tl.int64, [NUM_STAGES])
    smem_empty = allocate_smem(tl.int64, [NUM_STAGES])
    pipeline = TmaPipeline(smem_ready, smem_empty, smem_bufs, NUM_STAGES)

    # -- device-side tensormaps --
    in_stride = tl.cast(INPUT_N * 2, tl.int64)
    tmap_global_in = tl.cast(tmap_scratch_ptr + pid * 512, tl.uint64, bitcast=True)
    in_tmap = create_tmap_2d(
        smem_tmap,
        tmap_global_in,
        remote_input_ptr,
        INPUT_N,
        local_seq_len,
        in_stride,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_NONE,
    )

    q_stride = tl.cast(Q_OUT_N * 2, tl.int64)
    tmap_global_q = tl.cast(tmap_scratch_ptr + pid * 512 + 128, tl.uint64, bitcast=True)
    q_base = q_out_ptr + src_sp_rank * local_seq_len * Q_OUT_N
    q_tmap = create_tmap_2d(
        smem_tmap,
        tmap_global_q,
        q_base,
        Q_OUT_N,
        local_seq_len,
        q_stride,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_NONE,
    )

    K_DIM: tl.constexpr = K_OUT_N if K_OUT_N > 0 else Q_OUT_N
    k_stride = tl.cast(K_DIM * 2, tl.int64)
    tmap_global_k = tl.cast(tmap_scratch_ptr + pid * 512 + 256, tl.uint64, bitcast=True)
    if K_OUT_N > 0:
        k_base = k_out_ptr + src_sp_rank * local_seq_len * K_OUT_N
    else:
        k_base = q_base
    k_tmap = create_tmap_2d(
        smem_tmap,
        tmap_global_k,
        k_base,
        K_DIM,
        local_seq_len,
        k_stride,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_NONE,
    )

    V_DIM: tl.constexpr = V_OUT_N if V_OUT_N > 0 else Q_OUT_N
    v_stride = tl.cast(V_DIM * 2, tl.int64)
    tmap_global_v = tl.cast(tmap_scratch_ptr + pid * 512 + 384, tl.uint64, bitcast=True)
    if V_OUT_N > 0:
        v_base = v_out_ptr + src_sp_rank * local_seq_len * V_OUT_N
    else:
        v_base = q_base
    v_tmap = create_tmap_2d(
        smem_tmap,
        tmap_global_v,
        v_base,
        V_DIM,
        local_seq_len,
        v_stride,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_NONE,
    )

    one_c = tl.cast(1, tl.int32)
    if _t == 0:
        pipeline.init_barriers(one_c)
    __syncthreads()

    # ===================================================================
    # Warp 0 — PRODUCER: TMA G2S from remote packed QKV
    # ===================================================================
    if warp_idx == 0:
        p_state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        for tile_id in range(pid_in_rank, total_tiles, NUM_SM_PER_RANK):
            tile_m = tile_id // NUM_TILE_N
            tile_n = tile_id % NUM_TILE_N

            if tile_n < NUM_TILE_N_Q:
                input_col = q_input_col + tile_n * BN
            elif tile_n < NUM_TILE_N_Q + NUM_TILE_N_K:
                input_col = k_input_col + (tile_n - NUM_TILE_N_Q) * BN
            else:
                input_col = v_input_col + (tile_n - NUM_TILE_N_Q - NUM_TILE_N_K) * BN

            pipeline.producer_acquire(p_state)
            _e = elect_one_sync()
            if _e != 0:
                pipeline.producer_tma_load(p_state, in_tmap, input_col, tile_m * BM, TILE_BYTES)
            p_state = p_state.advance()

    # ===================================================================
    # Warp 1 — CONSUMER: TMA S2G to local Q/K/V
    # Single loop: first PIPE_CNT iterations are prefill (no release),
    # subsequent iterations release the oldest in-flight store.
    # ===================================================================
    else:  # warp_idx == 1
        cr_state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        ce_state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        cnt = 0

        for tile_id in range(pid_in_rank, total_tiles, NUM_SM_PER_RANK):
            tile_m = tile_id // NUM_TILE_N
            tile_n = tile_id % NUM_TILE_N

            pipeline.consumer_wait(cr_state)
            _e = elect_one_sync()
            if _e != 0:
                _dispatch_qkv_store(pipeline, cr_state, q_tmap, k_tmap, v_tmap, tile_n, tile_m, BN, BM, NUM_TILE_N_Q,
                                    NUM_TILE_N_K)
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
