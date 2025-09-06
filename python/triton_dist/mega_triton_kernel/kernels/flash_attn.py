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

import triton
import triton.language as tl
from .task_context import TaskBaseInfo, Scoreboard
from .utils import tanh


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    dtype: tl.constexpr, start_m, qk_scale, SOFT_CAP: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX, NUM_STAGES: tl.constexpr, warp_specialize: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    # loop over k, v and update accumulator

    for start_n in tl.range(lo, hi, BLOCK_N, num_stages=NUM_STAGES, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)
        if SOFT_CAP > 0.0:
            SOFT_CAP = SOFT_CAP.to(tl.float32)
            qk = SOFT_CAP * tanh(qk / SOFT_CAP)

        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # prepare p and v for the dot
        v = desc_v.load([start_n, 0])
        p = p.to(dtype)
        # note that this non transposed v for FP8 is only supported on Blackwell
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        # place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij
    return acc, l_i, m_i


@triton.jit
def _make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _qkv_pack_attn_fwd(
    tile_id,
    qkv_ptr,
    out_ptr,  #
    N_CTX,  #
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    IS_CAUSAL: tl.constexpr,
):
    STAGE: tl.constexpr = 3 if IS_CAUSAL else 1
    dtype = out_ptr.dtype.element_ty
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_N <= BLOCK_M)
    # seq_dim
    num_tile_m = tl.cdiv(N_CTX, BLOCK_M)
    tile_id_m = tile_id % num_tile_m

    start_m = tile_id_m
    off_hz = tile_id // num_tile_m
    off_z = off_hz // H_Q
    off_hq = off_hz % H_Q

    group_size = H_Q // H_KV
    off_hkv = off_hq // group_size

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_hd = tl.arange(0, HEAD_DIM)

    total_H = (H_Q + 2 * H_KV)
    # qkv: [batch, seq, HQ + 2 * H_KV, HEAD_DIM]
    # out: [batch, seq, HQ, HEAD_DIM]
    offset_q = off_z.to(tl.int64) * N_CTX * total_H * HEAD_DIM + off_hq.to(tl.int64) * HEAD_DIM
    offset_k = off_z.to(tl.int64) * N_CTX * total_H * HEAD_DIM + (off_hkv + H_Q) * HEAD_DIM
    offset_v = off_z.to(tl.int64) * N_CTX * total_H * HEAD_DIM + (off_hkv + H_Q + H_KV) * HEAD_DIM
    offset_o = off_z.to(tl.int64) * N_CTX * H_Q * HEAD_DIM + off_hq * HEAD_DIM
    stride_q = HEAD_DIM * total_H
    stride_kv = HEAD_DIM * total_H
    stride_o = HEAD_DIM * H_Q

    desc_q = _make_tensor_desc(qkv_ptr + offset_q, shape=[N_CTX, HEAD_DIM], strides=[stride_q, 1],
                               block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = _make_tensor_desc(qkv_ptr + offset_k, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = _make_tensor_desc(qkv_ptr + offset_v, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])
    # desc_o = _make_tensor_desc(out_ptr + offset_o, shape=[N_CTX, HEAD_DIM], strides=[stride_o, 1],
    #                                  block_shape=[BLOCK_M, HEAD_DIM])

    qo_offset_y = start_m * BLOCK_M

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = SM_SCALE
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    warp_specialize: tl.constexpr = False
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        dtype, start_m, qk_scale, SOFT_CAP,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        NUM_STAGES, warp_specialize)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        dtype, start_m, qk_scale, SOFT_CAP,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        NUM_STAGES, warp_specialize)
    # epilogue

    acc = acc / l_i[:, None]
    o_ptrs = out_ptr + offset_o + offs_m[:, None] * stride_o + offs_hd[None, :]
    o_mask = offs_m[:, None] < N_CTX

    tl.store(o_ptrs, acc.to(dtype), mask=o_mask)
    # desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_fwd(
    tile_id,
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,  #
    N_CTX,  #
    H_Q: tl.constexpr,
    H_KV: tl.constexpr,
    SM_SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,  #
    IS_CAUSAL: tl.constexpr,
):
    STAGE: tl.constexpr = 3 if IS_CAUSAL else 1
    dtype = out_ptr.dtype.element_ty
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tl.static_assert(BLOCK_N <= BLOCK_M)
    # seq_dim
    num_tile_m = tl.cdiv(N_CTX, BLOCK_M)
    tile_id_m = tile_id % num_tile_m

    start_m = tile_id_m
    off_hz = tile_id // num_tile_m
    off_z = off_hz // H_Q
    off_hq = off_hz % H_Q

    group_size = H_Q // H_KV
    off_hkv = off_hq // group_size

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_hd = tl.arange(0, HEAD_DIM)

    # q: [batch, seq, HQ, HEAD_DIM]
    # k/v: [batch, seq, H_KV, HEAD_DIM]
    # out: [batch, seq, HQ, HEAD_DIM]
    offset_q = off_z.to(tl.int64) * N_CTX * H_Q * HEAD_DIM + off_hq.to(tl.int64) * HEAD_DIM
    offset_k = off_z.to(tl.int64) * N_CTX * H_KV * HEAD_DIM + off_hkv * HEAD_DIM
    offset_v = off_z.to(tl.int64) * N_CTX * H_KV * HEAD_DIM + off_hkv * HEAD_DIM
    offset_o = off_z.to(tl.int64) * N_CTX * H_Q * HEAD_DIM + off_hq * HEAD_DIM
    stride_q = HEAD_DIM * H_Q
    stride_kv = HEAD_DIM * H_KV
    stride_o = HEAD_DIM * H_Q

    desc_q = _make_tensor_desc(q_ptr + offset_q, shape=[N_CTX, HEAD_DIM], strides=[stride_q, 1],
                               block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = _make_tensor_desc(k_ptr + offset_k, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = _make_tensor_desc(v_ptr + offset_v, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])

    qo_offset_y = start_m * BLOCK_M

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = SM_SCALE
    qk_scale *= 1.44269504  # 1/log(2)

    # load q: it will stay in SRAM throughout
    q = desc_q.load([qo_offset_y, 0])

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    warp_specialize: tl.constexpr = False
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        dtype, start_m, qk_scale, SOFT_CAP,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        NUM_STAGES, warp_specialize)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        dtype, start_m, qk_scale, SOFT_CAP,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        NUM_STAGES, warp_specialize)
    # epilogue

    acc = acc / l_i[:, None]
    o_ptrs = out_ptr + offset_o + offs_m[:, None] * stride_o + offs_hd[None, :]
    o_mask = offs_m[:, None] < N_CTX

    tl.store(o_ptrs, acc.to(dtype), mask=o_mask)


@triton.jit
def qkv_pack_flash_attn_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    SM_SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):

    tile_id = task_base_info.tile_id_or_start
    qkv_tensor = task_base_info.get_tensor(0)
    qkv_ptr = qkv_tensor.data_ptr(INPUT_DTYPE)
    out_ptr = task_base_info.get_tensor(1).data_ptr(OUTPUT_DTYPE)
    N_CTX = qkv_tensor.size(1)
    _qkv_pack_attn_fwd(
        tile_id,
        qkv_ptr,
        out_ptr,  #
        N_CTX,
        H_Q=NUM_Q_HEADS,
        H_KV=NUM_KV_HEADS,  #
        SM_SCALE=SM_SCALE,
        SOFT_CAP=SOFT_CAP,
        HEAD_DIM=HEAD_DIM,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        NUM_STAGES=NUM_STAGES,
        IS_CAUSAL=IS_CAUSAL,
    )
    scoreboard.release_tile(task_base_info, tile_id)


@triton.jit
def flash_attn_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    SM_SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    OUTPUT_DTYPE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):

    tile_id = task_base_info.tile_id_or_start
    q_tensor = task_base_info.get_tensor(0)
    k_tensor = task_base_info.get_tensor(1)
    v_tensor = task_base_info.get_tensor(2)

    q_ptr = q_tensor.data_ptr(INPUT_DTYPE)
    k_ptr = k_tensor.data_ptr(INPUT_DTYPE)
    v_ptr = v_tensor.data_ptr(INPUT_DTYPE)

    out_ptr = task_base_info.get_tensor(3).data_ptr(OUTPUT_DTYPE)
    N_CTX = q_tensor.size(1)
    _attn_fwd(
        tile_id,
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,  #
        N_CTX,
        H_Q=NUM_Q_HEADS,
        H_KV=NUM_KV_HEADS,  #
        SM_SCALE=SM_SCALE,
        SOFT_CAP=SOFT_CAP,
        HEAD_DIM=HEAD_DIM,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        NUM_STAGES=NUM_STAGES,
        IS_CAUSAL=IS_CAUSAL,
    )
    scoreboard.release_tile(task_base_info, tile_id)
