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
import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    dtype: tl.constexpr, start_m, qk_scale,  #
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
        k = desc_k.load([start_n, 0])
        k = k.T
        qk = tl.dot(q, k)
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
def _attn_fwd_with_qkv_desc(
    desc_q,
    desc_k,
    desc_v,
    start_m,
    offset_o,
    out_ptr,
    sm_scale,  #
    Z,
    H_Q,
    H_KV,
    N_CTX,  #
    dtype: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    STAGE: tl.constexpr = 3 if IS_CAUSAL else 1
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_hd = tl.arange(0, HEAD_DIM)
    qo_offset_y = start_m * BLOCK_M

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    # q_ptrs = qkv_ptr + offset_q + offs_m[:, None] * stride_q + offs_n[None, :]
    q = desc_q.load([qo_offset_y, 0])
    # q = tl.load(q_ptrs)
    # tl.device_print("q =  ", q)
    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    warp_specialize: tl.constexpr = False
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        NUM_STAGES, warp_specialize)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        dtype, start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        NUM_STAGES, warp_specialize)
    # epilogue

    acc = acc / l_i[:, None]
    stride_o = HEAD_DIM * H_Q
    o_ptrs = out_ptr + offset_o + offs_m[:, None] * stride_o + offs_hd[None, :]
    o_mask = offs_m[:, None] < N_CTX

    tl.store(o_ptrs, acc.to(dtype), mask=o_mask)


@triton.jit
def _make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# configs = [
#     triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
#     for BM in [64, 128]\
#     for BN in [64, 128]\
#     for s in [2, 3, 4] \
#     for w in [8]\
# ]


# @triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit(do_not_specialize=["N_CTX", "Z"])
def _qkv_pack_attn_fwd(
    qkv_ptr,
    out_ptr,
    sm_scale,  #
    Z,
    H_Q,
    H_KV,
    N_CTX,  #
    dtype: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # dtype = out_ptr.dtype.element_ty
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tile_id = tl.program_id(0)
    # seq_dim
    num_tile_m = tl.cdiv(N_CTX, BLOCK_M)
    tile_id_m = tile_id % num_tile_m

    start_m = tile_id_m
    off_hz = tile_id // num_tile_m
    off_z = off_hz // H_Q
    off_hq = off_hz % H_Q

    group_size = H_Q // H_KV
    off_hkv = off_hq // group_size

    total_H = (H_Q + 2 * H_KV)
    # qkv: [batch, seq, HQ + 2 * H_KV, HEAD_DIM]
    # out: [batch, seq, HQ, HEAD_DIM]
    offset_q = off_z.to(tl.int64) * N_CTX * total_H * HEAD_DIM + off_hq.to(tl.int64) * HEAD_DIM
    offset_k = off_z.to(tl.int64) * N_CTX * total_H * HEAD_DIM + (off_hkv + H_Q) * HEAD_DIM
    offset_v = off_z.to(tl.int64) * N_CTX * total_H * HEAD_DIM + (off_hkv + H_Q + H_KV) * HEAD_DIM
    offset_o = off_z.to(tl.int64) * N_CTX * H_Q * HEAD_DIM + off_hq * HEAD_DIM
    stride_q = HEAD_DIM * total_H
    stride_kv = HEAD_DIM * total_H
    # stride_o = HEAD_DIM * H_Q

    desc_q = _make_tensor_desc(qkv_ptr + offset_q, shape=[N_CTX, HEAD_DIM], strides=[stride_q, 1],
                               block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = _make_tensor_desc(qkv_ptr + offset_k, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = _make_tensor_desc(qkv_ptr + offset_v, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])
    # desc_o = _make_tensor_desc(out_ptr + offset_o, shape=[N_CTX, HEAD_DIM], strides=[stride_o, 1],
    #                            block_shape=[BLOCK_M, HEAD_DIM])

    _attn_fwd_with_qkv_desc(
        desc_q,
        desc_k,
        desc_v,
        start_m,
        offset_o,
        out_ptr,
        sm_scale,  #
        Z,
        H_Q,
        H_KV,
        N_CTX,  #
        dtype=dtype,
        HEAD_DIM=HEAD_DIM,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        NUM_STAGES=NUM_STAGES,
        IS_CAUSAL=IS_CAUSAL,
    )


# @triton.autotune(configs=configs, key=["N_CTX", "HEAD_DIM"])
@triton.jit(do_not_specialize=["N_CTX", "Z"])
def _attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    sm_scale,  #
    Z,
    H_Q,
    H_KV,
    N_CTX,  #
    dtype: tl.constexpr,
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    NUM_STAGES: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # dtype = out_ptr.dtype.element_ty
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    tile_id = tl.program_id(0)
    # seq_dim
    num_tile_m = tl.cdiv(N_CTX, BLOCK_M)
    tile_id_m = tile_id % num_tile_m

    start_m = tile_id_m
    off_hz = tile_id // num_tile_m
    off_z = off_hz // H_Q
    off_hq = off_hz % H_Q
    # off_hq = tile_id % H_Q
    # off_mz = tile_id // H_Q
    # off_z = off_mz // num_tile_m
    # tile_id_m = off_mz % num_tile_m
    # start_m = tile_id_m

    group_size = H_Q // H_KV
    off_hkv = off_hq // group_size

    # q: [batch, seq, HQ, HEAD_DIM]
    # k/v: [batch, seq, H_KV, HEAD_DIM]
    # out: [batch, seq, HQ, HEAD_DIM]
    offset_q = off_z.to(tl.int64) * N_CTX * H_Q * HEAD_DIM + off_hq.to(tl.int64) * HEAD_DIM
    offset_k = off_z.to(tl.int64) * N_CTX * H_KV * HEAD_DIM + off_hkv * HEAD_DIM
    offset_v = off_z.to(tl.int64) * N_CTX * H_KV * HEAD_DIM + off_hkv * HEAD_DIM
    offset_o = off_z.to(tl.int64) * N_CTX * H_Q * HEAD_DIM + off_hq * HEAD_DIM
    stride_q = HEAD_DIM * H_Q
    stride_kv = HEAD_DIM * H_KV
    # stride_o = HEAD_DIM * H_Q

    desc_q = _make_tensor_desc(q_ptr + offset_q, shape=[N_CTX, HEAD_DIM], strides=[stride_q, 1],
                               block_shape=[BLOCK_M, HEAD_DIM])
    desc_k = _make_tensor_desc(k_ptr + offset_k, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])
    desc_v = _make_tensor_desc(v_ptr + offset_v, shape=[N_CTX, HEAD_DIM], strides=[stride_kv, 1],
                               block_shape=[BLOCK_N, HEAD_DIM])

    _attn_fwd_with_qkv_desc(
        desc_q,
        desc_k,
        desc_v,
        start_m,
        offset_o,
        out_ptr,
        sm_scale,  #
        Z,
        H_Q,
        H_KV,
        N_CTX,  #
        dtype=dtype,
        HEAD_DIM=HEAD_DIM,  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        NUM_STAGES=NUM_STAGES,
        IS_CAUSAL=IS_CAUSAL,
    )


def triton_attn(input_tensors, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM, IS_CAUSAL, qkv_pack):
    sm_scale = HEAD_DIM**-0.5
    grid = lambda meta: (triton.cdiv(N_CTX, meta["BLOCK_M"]) * Q_HEAD * BATCH, )
    BLOCK_M = 128
    BLOCK_N = 128
    NUM_STAGES = 3
    out = torch.empty((BATCH, N_CTX, Q_HEAD, HEAD_DIM), dtype=input_tensors[0].dtype, device=input_tensors[0].device,
                      requires_grad=True)

    if not qkv_pack:
        q, k, v = input_tensors
        _attn_fwd[grid](
            q,
            k,
            v,
            out,
            sm_scale,  #
            BATCH,
            Q_HEAD,
            KV_HEAD,
            N_CTX,  #
            dtype=tl.bfloat16,
            HEAD_DIM=HEAD_DIM,  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_STAGES=NUM_STAGES,
            IS_CAUSAL=IS_CAUSAL,  #
            num_warps=8,
        )
    else:
        qkv = input_tensors[0]
        _qkv_pack_attn_fwd[grid](
            qkv,
            out,
            sm_scale,  #
            BATCH,
            Q_HEAD,
            KV_HEAD,
            N_CTX,  #
            dtype=tl.bfloat16,
            HEAD_DIM=HEAD_DIM,  #
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            NUM_STAGES=NUM_STAGES,
            IS_CAUSAL=IS_CAUSAL,  #
            num_warps=8,
        )
    return out


@triton.jit(do_not_specialize=["seq_len"])
def kernel_qkv_pack_qk_norm_rope_split_v(qkv_ptr, kv_lens_ptr, q_rms_weight_ptr, k_rms_weight_ptr,  #
                                         q_rms_eps, k_rms_eps, sin_ptr, cos_ptr,  #
                                         q_norm_rope_ptr, k_norm_rope_ptr, v_ptr,  #
                                         bs, seq_len, num_q_heads, num_kv_heads,  #
                                         head_dim, BLOCK_SEQ: tl.constexpr, BLOCK_HD: tl.constexpr):
    """
        qkv: (bs, seq_len, num_total_heads, head_dim)
        BLOCK_HD equal to next_power_of_2(head_dim)
    """
    pid = tl.program_id(axis=0)
    num_total_heads = num_q_heads + 2 * num_kv_heads
    group_size = num_q_heads // num_kv_heads
    head_type = pid % (group_size + 2)
    num_tiles_seq = tl.cdiv(seq_len, BLOCK_SEQ)
    tile_id_bs_seq = pid // num_total_heads
    tile_id_bs = tile_id_bs_seq // num_tiles_seq
    tild_id_seq = tile_id_bs_seq % num_tiles_seq

    head_group_id = (pid % num_total_heads) // (group_size + 2)
    offs_seq = tl.arange(0, BLOCK_SEQ) + tild_id_seq * BLOCK_SEQ
    offs_hd = tl.arange(0, BLOCK_HD)
    stride_qkv_seq = num_total_heads * head_dim
    stride_nh = head_dim
    bs_idx = tile_id_bs.to(tl.int64)
    history_kv = tl.load(kv_lens_ptr + bs_idx)

    mask = (offs_seq[:, None] < seq_len) & (offs_hd[None, :] < head_dim)

    # split v
    if head_type == group_size + 1:
        head_id_no_pack = head_group_id
        head_id_pack = head_id_no_pack + num_q_heads + num_kv_heads
        v_input_ptrs = qkv_ptr + bs_idx * seq_len * stride_qkv_seq + offs_seq[:,
                                                                              None] * stride_qkv_seq + head_id_pack * stride_nh + offs_hd[
                                                                                  None:]
        v_val = tl.load(v_input_ptrs, mask=mask)
        v_out_ptrs = v_ptr + bs_idx * seq_len * num_kv_heads * head_dim + offs_seq[:,
                                                                                   None] * num_kv_heads * head_dim + head_id_no_pack * stride_nh + offs_hd[
                                                                                       None:]
        tl.store(v_out_ptrs, v_val, mask=mask)
    else:
        # q/k norm & rope
        if head_type < group_size:
            head_id_no_pack = head_group_id * group_size + head_type
            head_id_pack = head_id_no_pack
            RMS_EPS = q_rms_eps
            rms_weight_ptr = q_rms_weight_ptr
            stride_seq = num_q_heads * head_dim
            out_ptr = q_norm_rope_ptr + bs_idx * seq_len * stride_seq + head_id_no_pack * head_dim
        else:
            head_id_no_pack = head_group_id
            head_id_pack = head_id_no_pack + num_q_heads
            RMS_EPS = k_rms_eps
            rms_weight_ptr = k_rms_weight_ptr
            stride_seq = num_kv_heads * head_dim
            out_ptr = k_norm_rope_ptr + bs_idx * seq_len * stride_seq + head_id_no_pack * head_dim

        # apply rmsnorm in qk head_dim
        _rms = tl.zeros([BLOCK_SEQ, BLOCK_HD], dtype=tl.float32)
        input_ptrs = qkv_ptr + bs_idx * seq_len * stride_qkv_seq + offs_seq[:,
                                                                            None] * stride_qkv_seq + head_id_pack * stride_nh + offs_hd[
                                                                                None:]
        a = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
        _rms += a * a  # [BLOCK_SEQ, BLOCK_HD]
        rms = tl.rsqrt(tl.sum(_rms, axis=1, keep_dims=True) / head_dim + RMS_EPS)  # [BLOCK_SEQ, 1]

        mask_hd = (offs_hd < head_dim)
        w = tl.load(rms_weight_ptr + offs_hd, mask=mask_hd).to(tl.float32)  # [BLOCK_HD]
        x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)  # [BLOCK_SEQ, BLOCK_HD]
        y = w[None, :] * (x * rms)  # [BLOCK_SEQ, BLOCK_HD]

        cos_offsets = tl.arange(0, BLOCK_HD // 2)
        cos_ptrs = cos_ptr + (offs_seq[:, None] + history_kv) * head_dim + cos_offsets[None, :]
        sin_ptrs = sin_ptr + (offs_seq[:, None] + history_kv) * head_dim + cos_offsets[None, :]

        cos_mask = (offs_seq[:, None] < seq_len) & (cos_offsets[None, :] < head_dim // 2)
        cos_row = tl.load(cos_ptrs, mask=cos_mask, other=0)  # [BLOCK_SEQ, BLOCK_HD // 2]
        sin_row = tl.load(sin_ptrs, mask=cos_mask, other=0)  # [BLOCK_SEQ, BLOCK_HD // 2]

        y = tl.reshape(y, (BLOCK_SEQ, 2, BLOCK_HD // 2))
        y = tl.permute(y, (0, 2, 1))
        y0, y1 = tl.split(y)  # [BLOCK_SEQ, BLOCK_HD // 2]
        first_half_qk_offsets = offs_seq[:, None] * stride_seq + cos_offsets
        first_qk_mask = cos_mask

        # # right half of the head
        second_half_qk_offsets = first_half_qk_offsets + (head_dim // 2)
        second_qk_mask = first_qk_mask
        # qkv_tile_2 = tl.load(X + second_half_qk_offsets, mask=second_qk_mask, other=0).to(sin_row.dtype)

        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        # cast to bf16 to keep same behavior
        y0 = y0.to(tl.bfloat16)
        y1 = y1.to(tl.bfloat16)
        new_qkv_tile_0 = y0 * cos_row - y1 * sin_row
        new_qkv_tile_1 = y1 * cos_row + y0 * sin_row

        tl.store(out_ptr + first_half_qk_offsets, new_qkv_tile_0, mask=first_qk_mask)
        tl.store(out_ptr + second_half_qk_offsets, new_qkv_tile_1, mask=second_qk_mask)


def triton_qkv_pack_qk_norm_rope_split_v(qkv, kv_lens, q_rms_weight, k_rms_weight, q_rms_eps, k_rms_eps, sin_cache,
                                         cos_cache, num_q_heads, num_kv_heads):
    assert len(qkv.shape) == 4
    bs, seq_len, num_total_heads, head_dim = qkv.shape
    assert num_total_heads == num_q_heads + num_kv_heads * 2

    BLOCK_HD = triton.next_power_of_2(head_dim)
    BLOCK_SEQ = 32
    grid = lambda meta: (bs * triton.cdiv(seq_len, BLOCK_SEQ) * num_total_heads, )

    q_norm_rope = torch.empty((bs, seq_len, num_q_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    k_norm_rope = torch.empty((bs, seq_len, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    v = torch.empty((bs, seq_len, num_kv_heads, head_dim), dtype=qkv.dtype, device=qkv.device)
    kernel_qkv_pack_qk_norm_rope_split_v[grid](
        qkv, kv_lens, q_rms_weight, k_rms_weight,  #
        q_rms_eps, k_rms_eps, sin_cache, cos_cache,  #
        q_norm_rope, k_norm_rope, v,  #
        bs, seq_len, num_q_heads, num_kv_heads,  #
        head_dim, BLOCK_SEQ=BLOCK_SEQ, BLOCK_HD=BLOCK_HD)
    return q_norm_rope, k_norm_rope, v
