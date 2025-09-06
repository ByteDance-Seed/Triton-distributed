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
from triton.language.extra import libdevice
from .task_context import TaskBaseInfo, Scoreboard
from .utils import tanh


@triton.jit
def attn_gqa_fwd_batch_decode_split_kv_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    SM_SCALE: tl.constexpr,
    SOFT_CAP: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    Q_HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_NUM_BLOCKS_PER_SEQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    tile_id = task_base_info.tile_id_or_start
    q_ptr = task_base_info.get_tensor(0).data_ptr(tl.bfloat16)
    k_cache_ptr = task_base_info.get_tensor(1).data_ptr(tl.bfloat16)
    v_cache_ptr = task_base_info.get_tensor(2).data_ptr(tl.bfloat16)
    block_table_ptr = task_base_info.get_tensor(3).data_ptr(tl.int32)
    kv_length_ptr = task_base_info.get_tensor(4).data_ptr(tl.int32)
    partial_out_ptr = task_base_info.get_tensor(5).data_ptr(tl.float32)
    lse_ptr = task_base_info.get_tensor(6).data_ptr(tl.float32)

    tl.static_assert(NUM_Q_HEADS % NUM_KV_HEADS == 0)
    NUM_Q_HEADS_PER_GROUP: tl.constexpr = NUM_Q_HEADS // NUM_KV_HEADS
    K_HEAD_DIM: tl.constexpr = Q_HEAD_DIM
    stride_q_bs: tl.constexpr = NUM_Q_HEADS * Q_HEAD_DIM
    stride_q_h: tl.constexpr = Q_HEAD_DIM

    stride_table_bs: tl.constexpr = MAX_NUM_BLOCKS_PER_SEQ
    # stride_table_bs: tl.constexpr = PAGE_SIZE * NUM_KV_HEADS * K_HEAD_DIM
    stride_k_cache_bs: tl.constexpr = NUM_KV_HEADS * K_HEAD_DIM
    stride_k_cache_h: tl.constexpr = K_HEAD_DIM
    stride_v_cache_bs: tl.constexpr = NUM_KV_HEADS * V_HEAD_DIM
    stride_v_cache_h: tl.constexpr = V_HEAD_DIM
    # partial out stride
    stride_o_bs: tl.constexpr = NUM_Q_HEADS * NUM_KV_SPLITS * V_HEAD_DIM
    stride_o_h: tl.constexpr = NUM_KV_SPLITS * V_HEAD_DIM
    stride_o_split: tl.constexpr = V_HEAD_DIM

    stride_lse_bs: tl.constexpr = NUM_Q_HEADS * NUM_KV_SPLITS
    stride_lse_h: tl.constexpr = NUM_KV_SPLITS

    head_blocks = tl.cdiv(NUM_Q_HEADS, min(NUM_Q_HEADS_PER_GROUP, BLOCK_H))
    bid = tile_id // (head_blocks * NUM_KV_SPLITS)
    hid = tile_id % (head_blocks * NUM_KV_SPLITS) // NUM_KV_SPLITS
    kv_hid = hid // tl.cdiv(NUM_Q_HEADS_PER_GROUP, BLOCK_H)
    split_kv_id = tile_id % NUM_KV_SPLITS

    if NUM_Q_HEADS_PER_GROUP > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = NUM_Q_HEADS_PER_GROUP

    cur_head = hid * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (hid + 1) * VALID_BLOCK_H) & (cur_head < NUM_Q_HEADS)

    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < K_HEAD_DIM
    mask_dv = offs_dv < V_HEAD_DIM
    cur_kv_seq_len = tl.load(kv_length_ptr + bid)

    offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :] * 1  # stride_q_d
    q = tl.load(q_ptr + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_HEAD_DIM + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < K_HEAD_DIM
        offs_qpe = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_dpe[:, None] * 1  # stride_q_d
        qpe = tl.load(q_ptr + offs_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(block_table_ptr + bid * stride_table_bs + offs_n // PAGE_SIZE * 1,  # stride_table_d,
                                 mask=offs_n < split_kv_end, other=0)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_cache_k = kv_loc[
            None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_d[:, None] * 1  # stride_k_cache_d
        k = tl.load(k_cache_ptr + offs_cache_k, mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_cache_kpe = kv_loc[
                None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_dpe[:, None] * 1  # stride_k_cache_d
            kpe = tl.load(k_cache_ptr + offs_cache_kpe, mask=(offs_n[None, :] < split_kv_end)
                          & mask_dpe[:, None], other=0.0)
            qk += tl.dot(qpe, kpe.to(qpe.dtype))

        qk *= SM_SCALE

        if SOFT_CAP > 0.0:
            SOFT_CAP = SOFT_CAP.to(tl.float32)
            qk = SOFT_CAP * tanh(qk / SOFT_CAP)

        qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

        offs_cache_v = kv_loc[:, None] * stride_v_cache_bs + kv_hid * stride_v_cache_h + offs_dv[
            None, :] * 1  # stride_v_cache_d
        v = tl.load(v_cache_ptr + offs_cache_v, mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = libdevice.fast_expf(e_max - n_e_max)
        p = libdevice.fast_expf(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_out = bid * stride_o_bs + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_dv[
        None, :] * 1  # stride_o_d
    tl.store(partial_out_ptr + offs_out, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])

    offs_log = bid * stride_lse_bs + cur_head * stride_lse_h + split_kv_id
    tl.store(lse_ptr + offs_log, e_max + tl.log(e_sum), mask=mask_h)

    scoreboard.release_tile(task_base_info, tile_id)


@triton.jit
def attn_gqa_fwd_batch_decode_combine_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    NUM_Q_HEADS: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
):
    tile_id = task_base_info.tile_id_or_start

    kv_length_ptr = task_base_info.get_tensor(0).data_ptr(tl.int32)
    partial_out_ptr = task_base_info.get_tensor(1).data_ptr(tl.float32)
    lse_ptr = task_base_info.get_tensor(2).data_ptr(tl.float32)
    out_ptr = task_base_info.get_tensor(3).data_ptr(tl.bfloat16)

    # partial out stride
    stride_o_bs: tl.constexpr = NUM_Q_HEADS * NUM_KV_SPLITS * V_HEAD_DIM
    stride_o_h: tl.constexpr = NUM_KV_SPLITS * V_HEAD_DIM
    stride_o_split: tl.constexpr = V_HEAD_DIM

    # attn out stride
    stride_final_o_bs: tl.constexpr = NUM_Q_HEADS * V_HEAD_DIM
    stride_final_o_h: tl.constexpr = V_HEAD_DIM

    stride_lse_bs: tl.constexpr = NUM_Q_HEADS * NUM_KV_SPLITS
    stride_lse_h: tl.constexpr = NUM_KV_SPLITS

    cur_batch = tile_id // NUM_Q_HEADS
    cur_head = tile_id % NUM_Q_HEADS

    cur_batch_seq_len = tl.load(kv_length_ptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < V_HEAD_DIM

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_o_bs + cur_head * stride_o_h + offs_d
    offs_logic = cur_batch * stride_lse_bs + cur_head * stride_lse_h

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(partial_out_ptr + offs_v + split_kv_id * stride_o_split, mask=mask_d, other=0.0)
            tlogic = tl.load(lse_ptr + offs_logic + split_kv_id)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        out_ptr + cur_batch * stride_final_o_bs + cur_head * stride_final_o_h + offs_d,
        (acc / e_sum).to(out_ptr.dtype.element_ty),
        mask=mask_d,
    )

    scoreboard.release_tile(task_base_info, tile_id)
