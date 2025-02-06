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
# Part of the code adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_decode_attention.py
# https://github.com/vllm-project/vllm/blob/main/tests/kernels/test_flashinfer.py
# which was originally adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py
#
# The original copyright is:
#
# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2025 vLLM Team
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
################################################################################

import torch
import triton
import math
import os
import triton.language as tl
from triton.language.extra import libdevice

from triton.distributed.tools import aot_compile_spaces

if "USE_TRITON_DISTRIBUTED_AOT" in os.environ and os.environ["USE_TRITON_DISTRIBUTED_AOT"] in [
        "1", "true", "on", "ON", "On", True
]:
    use_aot = True
else:
    use_aot = False

if use_aot:
    from triton._C.libtriton_distributed import distributed

from triton.distributed.kernels.nvidia.common_ops import thread_id, ld_acquire, red_release
from triton.language.extra.cuda.language_extra import (
    __syncthreads, )


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


split_kv_signature = ((
    "*{input_dtype}:16, *{cache_dtype}:16, *{cache_dtype}:16, *{output_dtype}:16, "  # q/k_cache/v_cache/output
    "fp32, "  # sm_scale
    "*i32:16, *i32, "  # block_table/kv_length
    "i32, ") +  # batch
                      (
                          ", ".join([
                              "i32:16", "i32:16",  # q
                              "i32:16", "i32:16",  # k
                              "i32:16", "i32:16",  # v
                              "i32:16", "i32:16", "i32",  # o
                              "i32",  # table
                          ]) + ", ") +  # strides
                      ("%kv_group_num, "
                       "%q_head_num, "
                       "%BLOCK_HEAD_DIM, "
                       "%BLOCK_DPE, "
                       "%BLOCK_DV, "
                       "%BLOCK_N, "
                       "%BLOCK_H, "
                       "%NUM_KV_SPLITS, "
                       "%PAGE_SIZE, "
                       "%soft_cap, "
                       "%K_DIM, "
                       "%V_DIM"))

_split_kv_grid = [
    "batch",
    "(%q_head_num + (%BLOCK_H < %kv_group_num ? %BLOCK_H : %kv_group_num) - 1) / (%BLOCK_H < %kv_group_num ? %BLOCK_H : %kv_group_num)",
    "%NUM_KV_SPLITS"
]


def get_triton_split_kv_algo_info(q_heads, kv_heads, q_head_dim, v_head_dim, page_size, split_kv=32, soft_cap=0.0):
    return {
        "kv_group_num": q_heads // kv_heads, "q_head_num": q_heads, "BLOCK_HEAD_DIM": 2**int(math.log2(q_head_dim)),
        "BLOCK_DPE": q_head_dim - 2**int(math.log2(q_head_dim)), "BLOCK_DV": triton.next_power_of_2(v_head_dim),
        "BLOCK_N": 64, "BLOCK_H": 16, "NUM_KV_SPLITS": split_kv, "PAGE_SIZE": page_size, "soft_cap": soft_cap, "K_DIM":
        q_head_dim, "V_DIM": v_head_dim, "num_warps": 4, "num_stages": 2
    }


@aot_compile_spaces({
    "gqa_fwd_batch_decode_split_kv_fp16_fp16_fp32": {
        "signature":
        split_kv_signature.format(input_dtype="fp16", cache_dtype="fp16", output_dtype="fp32"), "grid":
        _split_kv_grid, "triton_algo_infos": [
            get_triton_split_kv_algo_info(96, 12, 128, 128, 1, split_kv=32, soft_cap=0),
            get_triton_split_kv_algo_info(96 // 4, 12 // 4, 128, 128, 1, split_kv=32, soft_cap=0),
        ]
    }, "gqa_fwd_batch_decode_split_kv_fp16_fp16_fp16": {
        "signature":
        split_kv_signature.format(input_dtype="fp16", cache_dtype="fp16", output_dtype="fp16"), "grid":
        _split_kv_grid, "triton_algo_infos": [
            get_triton_split_kv_algo_info(96, 12, 128, 128, 1, split_kv=32, soft_cap=0),
            get_triton_split_kv_algo_info(96 // 4, 12 // 4, 128, 128, 1, split_kv=32, soft_cap=0),
        ]
    }
})
@triton.jit
def kernel_gqa_fwd_batch_decode_split_kv(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    sm_scale,
    block_table_ptr,
    kv_length_ptr,
    # shape
    batch,
    # strides
    stride_q_bs,
    stride_q_h,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_o_bs,
    stride_o_h,
    stride_o_split,
    stride_table_bs,
    # constants
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    soft_cap: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    kv_hid = hid // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num

    cur_head = hid * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (hid + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_HEAD_DIM)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < K_DIM
    mask_dv = offs_dv < V_DIM
    cur_kv_seq_len = tl.load(kv_length_ptr + bid)

    offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :]
    q = tl.load(q_ptr + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_HEAD_DIM + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < K_DIM
        offs_qpe = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_dpe[:, None]
        qpe = tl.load(q_ptr + offs_qpe, mask=mask_h[:, None] & mask_dpe[None, :], other=0.0)

    kv_len_per_split = tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_page_number = tl.load(block_table_ptr + bid * stride_table_bs + offs_n // PAGE_SIZE, mask=offs_n
                                 < split_kv_end, other=0)
        kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
        offs_cache_k = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_d[:, None]
        k = tl.load(k_cache_ptr + offs_cache_k, mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None], other=0.0)
        qk = tl.dot(q, k.to(q.dtype))

        if BLOCK_DPE > 0:
            offs_cache_kpe = kv_loc[None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_dpe[:, None]
            kpe = tl.load(k_cache_ptr + offs_cache_kpe, mask=(offs_n[None, :] < split_kv_end) & mask_dpe[:, None],
                          other=0.0)
            qk += tl.dot(qpe, kpe.to(qpe.dtype))

        qk *= sm_scale

        if soft_cap > 0:
            qk = soft_cap * tanh(qk / soft_cap)

        qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf"))

        offs_cache_v = kv_loc[:, None] * stride_v_cache_bs + kv_hid * stride_v_cache_h + offs_dv[None, :]
        v = tl.load(v_cache_ptr + offs_cache_v, mask=(offs_n[:, None] < split_kv_end) & mask_dv[None, :], other=0.0)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = libdevice.fast_expf(e_max - n_e_max)
        p = libdevice.fast_expf(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_out = bid * stride_o_bs + cur_head[:, None] * stride_o_h + split_kv_id * stride_o_split + offs_dv[None, :]
    tl.store(output_ptr + offs_out, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])

    offs_log = bid * stride_o_bs + cur_head * stride_o_h + split_kv_id * stride_o_split + V_DIM
    tl.store(output_ptr + offs_log, e_max + tl.log(e_sum), mask=mask_h)


combine_kv_signature = ((
    "*{input_dtype}:16, *{output_dtype}:16, "  # mid_o/o
    "*i32, "  # kv_length
) + (
    ", ".join([
        "i32", "i32", "i32:16", "i32:16", "i32",  # mid_o
        "i32:16", "i32:16",  # o
    ]) + ", ") +  # strides
                        ("%NUM_KV_SPLITS, "
                         "%BLOCK_DV, "
                         "%Lv"))

combine_kv_signature_intra_rank = ((
    "*{input_dtype}:16, *{output_dtype}:16, "  # mid_o/o
    "*i32, "  # kv_length
) + (
    ", ".join([
        "i32", "i32", "i32:16", "i32:16", "i32",  # mid_o
        "i32:16", "i32",  # o
    ]) + ", ") +  # strides
                                   ("%NUM_KV_SPLITS, "
                                    "%BLOCK_DV, "
                                    "%Lv"))

combine_kv_signature_inter_rank = ((
    "*{input_dtype}:16, *{output_dtype}:16, "  # mid_o/o
    "*i32, "  # kv_length
) + (
    ", ".join([
        "i32", "i32", "i32:16", "i32", "i32:16",  # mid_o
        "i32:16", "i32:16",  # o
    ]) + ", ") +  # strides
                                   ("%NUM_KV_SPLITS, "
                                    "%BLOCK_DV, "
                                    "%Lv"))

_combine_kv_grid = ["batch", "q_heads", "1"]


def get_triton_combine_kv_algo_info(split_kv, v_head_dim, block_dv=None):
    return {
        "NUM_KV_SPLITS": split_kv, "BLOCK_DV": triton.next_power_of_2(v_head_dim) if block_dv is None else block_dv,
        "Lv": v_head_dim, "num_warps": 4, "num_stages": 2
    }


@aot_compile_spaces({
    "gqa_fwd_batch_decode_combine_kv_fp32_fp16": {
        "signature":
        combine_kv_signature.format(input_dtype="fp32", output_dtype="fp16"), "grid":
        _combine_kv_grid, "triton_algo_infos": [
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128),
        ]
    }, "gqa_fwd_batch_decode_combine_kv_fp16_fp16": {
        "signature":
        combine_kv_signature.format(input_dtype="fp16", output_dtype="fp16"), "grid":
        _combine_kv_grid, "triton_algo_infos": [
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128),
        ]
    }
})
@triton.jit
def kernel_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    B_Seqlen,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


@aot_compile_spaces({
    "intra_rank_gqa_fwd_batch_decode_combine_kv_fp32_fp16": {
        "signature":
        combine_kv_signature_intra_rank.format(input_dtype="fp32", output_dtype="fp16"), "grid":
        _combine_kv_grid, "triton_algo_infos": [
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128, block_dv=1024),
        ]
    }, "intra_rank_gqa_fwd_batch_decode_combine_kv_fp16_fp16": {
        "signature":
        combine_kv_signature_intra_rank.format(input_dtype="fp16", output_dtype="fp16"), "grid":
        _combine_kv_grid, "triton_algo_infos": [
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128, block_dv=1024),
        ]
    }
})
@triton.jit
def kernel_intra_rank_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    B_Seqlen,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + Lv,
        e_max + tl.log(e_sum),
    )


@aot_compile_spaces({
    "inter_rank_gqa_fwd_batch_decode_combine_kv_fp32_fp16": {
        "signature":
        combine_kv_signature_inter_rank.format(input_dtype="fp32", output_dtype="fp16"), "grid":
        _combine_kv_grid, "triton_algo_infos": [
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128, block_dv=1024),
        ]
    }, "inter_rank_gqa_fwd_batch_decode_combine_kv_fp16_fp16": {
        "signature":
        combine_kv_signature_inter_rank.format(input_dtype="fp16", output_dtype="fp16"), "grid":
        _combine_kv_grid, "triton_algo_infos": [
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128),
            get_triton_combine_kv_algo_info(split_kv=8, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=16, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=32, v_head_dim=128, block_dv=1024),
            get_triton_combine_kv_algo_info(split_kv=64, v_head_dim=128, block_dv=1024),
        ]
    }
})
@triton.jit
def kernel_inter_rank_gqa_fwd_batch_decode_combine_kv(
    Mid_O,
    o,
    B_Seqlens,
    batch,
    q_heads,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len_ptr = B_Seqlens + cur_batch

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        effective_kv_len = tl.load(cur_batch_seq_len_ptr + split_kv_id * batch)

        if effective_kv_len > 0:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = libdevice.fast_expf(e_max - n_e_max)
            acc *= old_scale
            exp_logic = libdevice.fast_expf(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


persistent_signature = (
    (
        "*{input_dtype}:16, *{cache_dtype}:16, *{cache_dtype}:16, *{output_dtype}:16, *{output_dtype}:16, "  # q/k_cache/v_cache/output/final_output
        "fp32, "  # sm_scale
        "*i32:16, *i32, *i32:16, "  # block_table/kv_length/workspace
        "i32, ") +  # batch
    (
        ", ".join([
            "i32:16", "i32:16",  # "i32:1", 
            "i32:16", "i32:16",  # "i32:1",
            "i32:16", "i32:16",  # "i32:1", 
            "i32:16", "i32:16", "i32",  # "i32:1",
            "i32:16", "i32:16", "i32",  # "i32:1"
        ]) + ", ")
    +  # strides: q_bs/q_h/q_d/k_cache_bs/k_cache_h/k_cache_d/v_cache_bs/v_cache_h/v_cache_d/o_bs/o_h/o_split/o_d/final_o_bs/final_o_h/table_bs/table_d
    ("%kv_group_num, "
     "%max_kv_seq_len, "
     "%q_head_num, "
     "%BLOCK_HEAD_DIM, "
     "%BLOCK_DPE, "
     "%BLOCK_DV, "
     "%BLOCK_N, "
     "%BLOCK_H, "
     "%NUM_KV_SPLITS, "
     "%PAGE_SIZE, "
     "%soft_cap, "
     "%K_DIM, "
     "%V_DIM"))

_persistent_grid = ["132", "1", "1"]


def get_triton_persistent_algo_info(q_heads, kv_heads, q_head_dim, v_head_dim, page_size, split_kv=32, soft_cap=0.0,
                                    max_kv_len=8192):
    return {
        "kv_group_num": q_heads // kv_heads, "max_kv_seq_len": max_kv_len, "q_head_num": q_heads, "BLOCK_HEAD_DIM":
        2**int(math.log2(q_head_dim)), "BLOCK_DPE": q_head_dim - 2**int(math.log2(q_head_dim)), "BLOCK_DV":
        triton.next_power_of_2(v_head_dim), "BLOCK_N": 64, "BLOCK_H": 16, "NUM_KV_SPLITS": split_kv, "PAGE_SIZE":
        page_size, "soft_cap": soft_cap, "K_DIM": q_head_dim, "V_DIM": v_head_dim, "num_warps": 8, "num_stages": 2
    }


@aot_compile_spaces({
    "gqa_fwd_batch_decode_split_kv_persistent_fp16_fp16_fp16": {
        "signature":
        persistent_signature.format(input_dtype="fp16", cache_dtype="fp16", output_dtype="fp16"), "grid":
        _persistent_grid, "triton_algo_infos": [
            get_triton_persistent_algo_info(96, 12, 128, 128, 1, split_kv=32, soft_cap=0, max_kv_len=8192),
        ]
    }
})
@triton.jit
def kernel_gqa_fwd_batch_decode_split_kv_persistent(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    output_ptr,
    final_output_ptr,
    sm_scale,
    block_table_ptr,
    kv_length_ptr,
    workspace_ptr,
    # shape
    batch,
    # strides
    stride_q_bs,
    stride_q_h,
    # stride_q_d,
    stride_k_cache_bs,
    stride_k_cache_h,
    # stride_k_cache_d,
    stride_v_cache_bs,
    stride_v_cache_h,
    # stride_v_cache_d,
    stride_o_bs,
    stride_o_h,
    stride_o_split,
    # stride_o_d,
    stride_final_o_bs,
    stride_final_o_h,
    stride_table_bs,
    # stride_table_d,
    # constants
    kv_group_num: tl.constexpr,
    max_kv_seq_len: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    soft_cap: tl.constexpr,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    head_blocks = tl.cdiv(q_head_num, min(kv_group_num, BLOCK_H))
    num_tiles = batch * head_blocks * NUM_KV_SPLITS
    for tile_id in range(sm_id, num_tiles, num_sms):
        bid = tile_id // (head_blocks * NUM_KV_SPLITS)
        hid = tile_id % (head_blocks * NUM_KV_SPLITS) // NUM_KV_SPLITS
        kv_hid = hid // tl.cdiv(kv_group_num, BLOCK_H)
        split_kv_id = tile_id % NUM_KV_SPLITS

        if kv_group_num > BLOCK_H:
            VALID_BLOCK_H: tl.constexpr = BLOCK_H
        else:
            VALID_BLOCK_H: tl.constexpr = kv_group_num

        cur_head = hid * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = (cur_head < (hid + 1) * VALID_BLOCK_H) & (cur_head < q_head_num)

        offs_d = tl.arange(0, BLOCK_HEAD_DIM)
        offs_dv = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < K_DIM
        mask_dv = offs_dv < V_DIM
        cur_kv_seq_len = tl.load(kv_length_ptr + bid)

        offs_q = bid * stride_q_bs + cur_head[:, None] * stride_q_h + offs_d[None, :] * 1  # stride_q_d
        q = tl.load(q_ptr + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)

        if BLOCK_DPE > 0:
            offs_dpe = BLOCK_HEAD_DIM + tl.arange(0, BLOCK_DPE)
            mask_dpe = offs_dpe < K_DIM
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
            kv_page_number = tl.load(block_table_ptr + bid * stride_table_bs +
                                     offs_n // PAGE_SIZE * 1,  # stride_table_d,
                                     mask=offs_n < split_kv_end, other=0)
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_cache_k = kv_loc[
                None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_d[:, None] * 1  # stride_k_cache_d
            k = tl.load(k_cache_ptr + offs_cache_k, mask=(offs_n[None, :] < split_kv_end) & mask_d[:, None], other=0.0)
            qk = tl.dot(q, k.to(q.dtype))

            if BLOCK_DPE > 0:
                offs_cache_kpe = kv_loc[
                    None, :] * stride_k_cache_bs + kv_hid * stride_k_cache_h + offs_dpe[:, None] * 1  # stride_k_cache_d
                kpe = tl.load(k_cache_ptr + offs_cache_kpe, mask=(offs_n[None, :] < split_kv_end) & mask_dpe[:, None],
                              other=0.0)
                qk += tl.dot(qpe, kpe.to(qpe.dtype))

            qk *= sm_scale

            if soft_cap > 0:
                soft_cap = soft_cap.to(tl.float32)
                qk = soft_cap * tanh(qk / soft_cap)

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
        tl.store(output_ptr + offs_out, acc / e_sum[:, None], mask=mask_h[:, None] & mask_dv[None, :])

        offs_log = bid * stride_o_bs + cur_head * stride_o_h + split_kv_id * stride_o_split + V_DIM
        tl.store(output_ptr + offs_log, e_max + tl.log(e_sum), mask=mask_h)

    tx = thread_id("x")
    if tx == 0:
        red_release(workspace_ptr + sm_id, 1, "gpu")
    __syncthreads()

    if tx < num_sms:
        while ld_acquire(workspace_ptr + tx, "gpu") != 1:
            pass
    __syncthreads()

    if sm_id < batch * q_head_num:

        cur_batch = sm_id // q_head_num
        cur_head = sm_id % q_head_num

        cur_batch_seq_len = tl.load(kv_length_ptr + cur_batch)

        offs_d = tl.arange(0, BLOCK_DV)
        mask_d = offs_d < V_DIM

        e_sum = 0.0
        e_max = -float("inf")
        acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

        offs_v = cur_batch * stride_o_bs + cur_head * stride_o_h + offs_d
        offs_logic = cur_batch * stride_o_bs + cur_head * stride_o_h + V_DIM

        for split_kv_id in range(0, NUM_KV_SPLITS):
            kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
            split_kv_start = kv_len_per_split * split_kv_id
            split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

            if split_kv_end > split_kv_start:
                tv = tl.load(output_ptr + offs_v + split_kv_id * stride_o_split, mask=mask_d, other=0.0)
                tlogic = tl.load(output_ptr + offs_logic + split_kv_id * stride_o_split)
                n_e_max = tl.maximum(tlogic, e_max)

                old_scale = libdevice.fast_expf(e_max - n_e_max)
                acc *= old_scale
                exp_logic = libdevice.fast_expf(tlogic - n_e_max)
                acc += exp_logic * tv

                e_sum = e_sum * old_scale + exp_logic
                e_max = n_e_max

        tl.store(
            final_output_ptr + cur_batch * stride_final_o_bs + cur_head * stride_final_o_h + offs_d,
            acc / e_sum,
            mask=mask_d,
        )
    __syncthreads()
    if tx == 0:
        red_release(workspace_ptr + sm_id, -1, "gpu")


def gqa_fwd_batch_decode(q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale, soft_cap=0.0,
                         output_split=None, output_combine=None, kv_split=-1):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
    v_head_dim = v_cache.shape[-1]

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)

    kv_group_num = q_heads // kv_heads
    assert q_heads % kv_heads == 0

    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)

    output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                1], dtype=torch.float16, device=q.device) if output_split is None else output_split
    output_combine = torch.empty([batch, q_heads, v_head_dim], dtype=torch.float16,
                                 device=q.device) if output_combine is None else output_combine

    kernel_gqa_fwd_batch_decode_split_kv[grid_split_kv](
        q,
        k_cache,
        v_cache,
        output_split,
        scale,
        block_table,
        kv_lens,
        # shape
        batch,
        # strides
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),
        k_cache.stride(-2),
        v_cache.stride(-3),
        v_cache.stride(-2),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        block_table.stride(0),
        # constants
        kv_group_num,
        q_heads,
        BLOCK_HEAD_DIM,
        BLOCK_DPE,
        BLOCK_DV,
        BLOCK_N,
        BLOCK_H,
        NUM_KV_SPLITS,
        page_size,
        soft_cap,
        k_head_dim,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    kernel_gqa_fwd_batch_decode_combine_kv[(batch, q_heads)](
        output_split,
        output_combine,
        kv_lens,
        batch,
        q_heads,
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        output_combine.stride(0),
        output_combine.stride(1),
        NUM_KV_SPLITS,
        BLOCK_DV,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    return output_combine


def gqa_fwd_batch_decode_intra_rank(q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale, soft_cap=0.0,
                                    output_split=None, output_combine=None, kv_split=-1):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
    v_head_dim = v_cache.shape[-1]

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)

    kv_group_num = q_heads // kv_heads
    assert q_heads % kv_heads == 0

    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    grid_split_kv = (batch, triton.cdiv(q_heads, min(BLOCK_H, kv_group_num)), NUM_KV_SPLITS)

    output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                1], dtype=q.dtype, device=q.device) if output_split is None else output_split
    output_combine = torch.empty([batch, q_heads, v_head_dim +
                                  1], dtype=q.dtype, device=q.device) if output_combine is None else output_combine

    kernel_gqa_fwd_batch_decode_split_kv[grid_split_kv](
        q,
        k_cache,
        v_cache,
        output_split,
        scale,
        block_table,
        kv_lens,
        # shape
        batch,
        # strides
        q.stride(0),
        q.stride(1),
        k_cache.stride(-3),
        k_cache.stride(-2),
        v_cache.stride(-3),
        v_cache.stride(-2),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        block_table.stride(0),
        # constants
        kv_group_num,
        q_heads,
        BLOCK_HEAD_DIM,
        BLOCK_DPE,
        BLOCK_DV,
        BLOCK_N,
        BLOCK_H,
        NUM_KV_SPLITS,
        page_size,
        soft_cap,
        k_head_dim,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    kernel_intra_rank_gqa_fwd_batch_decode_combine_kv[(batch, q_heads)](
        output_split,
        output_combine,
        kv_lens,
        batch,
        q_heads,
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        output_combine.stride(0),
        output_combine.stride(1),
        NUM_KV_SPLITS,
        BLOCK_DV,
        v_head_dim,
        num_warps=4,
        num_stages=2,
    )

    return output_combine


def gqa_fwd_batch_decode_persistent(q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale, soft_cap=0,
                                    output_split=None, output_combine=None, kv_split=-1):
    batch, q_heads, q_head_dim = q.shape
    _, page_size, kv_heads, k_head_dim = k_cache.shape
    assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
    v_head_dim = v_cache.shape[-1]

    BLOCK_N = 64
    BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
    BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
    BLOCK_DV = triton.next_power_of_2(v_head_dim)

    kv_group_num = q_heads // kv_heads
    assert q_heads % kv_heads == 0

    BLOCK_H = 16
    NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

    grid_split_kv = (132, )

    output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                1], dtype=torch.float32, device=q.device) if output_split is None else output_split
    output_combine = torch.empty([batch, q_heads, v_head_dim], dtype=torch.float16,
                                 device=q.device) if output_combine is None else output_combine

    kernel_gqa_fwd_batch_decode_split_kv_persistent[grid_split_kv](
        q,
        k_cache,
        v_cache,
        output_split,
        output_combine,
        scale,
        block_table,
        kv_lens,
        workspace,
        # shape,
        batch,
        # strides
        q.stride(0),
        q.stride(1),
        # q.stride(2),
        k_cache.stride(-3),
        k_cache.stride(-2),
        # k_cache.stride(-1),
        v_cache.stride(-3),
        v_cache.stride(-2),
        # v_cache.stride(-1),
        output_split.stride(0),
        output_split.stride(1),
        output_split.stride(2),
        # output_split.stride(3),
        output_combine.stride(0),
        output_combine.stride(1),
        block_table.stride(0),
        # block_table.stride(1),
        # constants
        kv_group_num,
        8192,  # max_kv_seq_len
        q_heads,
        BLOCK_HEAD_DIM,
        BLOCK_DPE,
        BLOCK_DV,
        BLOCK_N,
        BLOCK_H,
        NUM_KV_SPLITS,
        page_size,
        soft_cap,
        k_head_dim,
        v_head_dim,
        num_warps=8,
        num_stages=2,
    )

    return output_combine


def gqa_fwd_batch_decode_aot(stream, q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale, soft_cap=0,
                             output_split=None, output_combine=None, kv_split=-1):
    if use_aot:
        batch, q_heads, q_head_dim = q.shape
        _, page_size, kv_heads, k_head_dim = k_cache.shape
        assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
        v_head_dim = v_cache.shape[-1]

        assert q_heads % kv_heads == 0

        NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

        output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                    1], dtype=torch.float16, device=q.device) if output_split is None else output_split
        output_combine = torch.empty([batch, q_heads, v_head_dim], dtype=torch.float16,
                                     device=q.device) if output_combine is None else output_combine

        if output_split.dtype == torch.float32:
            kernel_split = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp32
            kernel_combine = distributed.gqa_fwd_batch_decode_combine_kv_fp32_fp16
            split_algo_info = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp32__triton_algo_info_t()
            combine_algo_info = distributed.gqa_fwd_batch_decode_combine_kv_fp32_fp16__triton_algo_info_t()
        elif output_split.dtype == torch.float16:
            kernel_split = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp16
            kernel_combine = distributed.gqa_fwd_batch_decode_combine_kv_fp16_fp16
            split_algo_info = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp16__triton_algo_info_t()
            combine_algo_info = distributed.gqa_fwd_batch_decode_combine_kv_fp16_fp16__triton_algo_info_t()
        else:
            raise RuntimeError("Unsupported data type of intermediate output:", output_split.dtype)

        py_split_algo_info = get_triton_split_kv_algo_info(q_heads, kv_heads, q_head_dim, v_head_dim, page_size,
                                                           split_kv=NUM_KV_SPLITS, soft_cap=soft_cap)
        py_combine_algo_info = get_triton_combine_kv_algo_info(split_kv=NUM_KV_SPLITS, v_head_dim=v_head_dim)
        for k, v in py_split_algo_info.items():
            setattr(split_algo_info, k, v)
        for k, v in py_combine_algo_info.items():
            setattr(combine_algo_info, k, v)

        kernel_split(stream.cuda_stream, q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), output_split.data_ptr(),
                     scale, block_table.data_ptr(), kv_lens.data_ptr(),
                     # shape,
                     batch,
                     # strides
                     q.stride(0), q.stride(1),  # q.strides
                     k_cache.stride(-3), k_cache.stride(-2),  # k_cache
                     v_cache.stride(-3), v_cache.stride(-2),  # v_cache
                     output_split.stride(0), output_split.stride(1), output_split.stride(2),  # output_split
                     block_table.stride(0),  # block_table
                     # algo_info
                     split_algo_info)
        kernel_combine(stream.cuda_stream, output_split.data_ptr(), output_combine.data_ptr(), kv_lens.data_ptr(),
                       batch, q_heads, output_split.stride(0), output_split.stride(1), output_split.stride(2),
                       output_combine.stride(0), output_combine.stride(1), combine_algo_info)
        return output_combine
    else:
        raise RuntimeError("Should enable USE_TRITON_DISTRIBUTED_AOT")


def gqa_fwd_batch_decode_intra_rank_aot(stream, q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale,
                                        soft_cap=0, output_split=None, output_combine=None, kv_split=-1):
    if use_aot:
        batch, q_heads, q_head_dim = q.shape
        _, page_size, kv_heads, k_head_dim = k_cache.shape
        assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
        v_head_dim = v_cache.shape[-1]

        assert q_heads % kv_heads == 0

        NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

        output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                    1], dtype=torch.float16, device=q.device) if output_split is None else output_split
        output_combine = torch.empty([batch, q_heads, v_head_dim + 1], dtype=torch.float16,
                                     device=q.device) if output_combine is None else output_combine

        if output_split.dtype == torch.float32:
            kernel_split = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp32
            kernel_combine = distributed.intra_rank_gqa_fwd_batch_decode_combine_kv_fp32_fp16
            split_algo_info = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp32__triton_algo_info_t()
            combine_algo_info = distributed.intra_rank_gqa_fwd_batch_decode_combine_kv_fp32_fp16__triton_algo_info_t()
        elif output_split.dtype == torch.float16:
            kernel_split = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp16
            kernel_combine = distributed.intra_rank_gqa_fwd_batch_decode_combine_kv_fp16_fp16
            split_algo_info = distributed.gqa_fwd_batch_decode_split_kv_fp16_fp16_fp16__triton_algo_info_t()
            combine_algo_info = distributed.intra_rank_gqa_fwd_batch_decode_combine_kv_fp16_fp16__triton_algo_info_t()
        else:
            raise RuntimeError("Unsupported data type of intermediate output:", output_split.dtype)

        py_split_algo_info = get_triton_split_kv_algo_info(q_heads, kv_heads, q_head_dim, v_head_dim, page_size,
                                                           split_kv=NUM_KV_SPLITS, soft_cap=soft_cap)
        py_combine_algo_info = get_triton_combine_kv_algo_info(split_kv=NUM_KV_SPLITS, v_head_dim=v_head_dim)
        for k, v in py_split_algo_info.items():
            setattr(split_algo_info, k, v)
        for k, v in py_combine_algo_info.items():
            setattr(combine_algo_info, k, v)

        kernel_split(stream.cuda_stream, q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), output_split.data_ptr(),
                     scale, block_table.data_ptr(), kv_lens.data_ptr(),
                     # shape,
                     batch,
                     # strides
                     q.stride(0), q.stride(1),  # q.strides
                     k_cache.stride(-3), k_cache.stride(-2),  # k_cache
                     v_cache.stride(-3), v_cache.stride(-2),  # v_cache
                     output_split.stride(0), output_split.stride(1), output_split.stride(2),  # output_split
                     block_table.stride(0),  # block_table
                     # algo_info
                     split_algo_info)
        kernel_combine(stream.cuda_stream, output_split.data_ptr(), output_combine.data_ptr(), kv_lens.data_ptr(),
                       batch, q_heads, output_split.stride(0), output_split.stride(1), output_split.stride(2),
                       output_combine.stride(0), output_combine.stride(1), combine_algo_info)
        return output_combine
    else:
        raise RuntimeError("Should enable USE_TRITON_DISTRIBUTED_AOT")


def gqa_fwd_batch_decode_persistent_aot(stream, q, k_cache, v_cache, workspace, q_lens, kv_lens, block_table, scale,
                                        soft_cap=0, output_split=None, output_combine=None, kv_split=-1):
    if use_aot:
        batch, q_heads, q_head_dim = q.shape
        _, page_size, kv_heads, k_head_dim = k_cache.shape
        assert page_size == v_cache.shape[1] and kv_heads == v_cache.shape[2] and k_head_dim == q_head_dim
        v_head_dim = v_cache.shape[-1]

        assert q_heads % kv_heads == 0

        NUM_KV_SPLITS = 32 if kv_split == -1 else kv_split

        output_split = torch.empty([batch, q_heads, NUM_KV_SPLITS, v_head_dim +
                                    1], dtype=torch.float32, device=q.device) if output_split is None else output_split
        output_combine = torch.empty([batch, q_heads, v_head_dim], dtype=torch.float16,
                                     device=q.device) if output_combine is None else output_combine

        kernel = distributed.gqa_fwd_batch_decode_split_kv_persistent_fp16_fp16_fp16
        algo_info = distributed.gqa_fwd_batch_decode_split_kv_persistent_fp16_fp16_fp16__triton_algo_info_t()
        py_algo_info = get_triton_persistent_algo_info(q_heads, kv_heads, q_head_dim, v_head_dim, page_size,
                                                       split_kv=NUM_KV_SPLITS, soft_cap=soft_cap, max_kv_len=8192)
        for k, v in py_algo_info.items():
            setattr(algo_info, k, v)
        kernel(stream.cuda_stream, q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), output_split.data_ptr(),
               output_combine.data_ptr(), scale, block_table.data_ptr(), kv_lens.data_ptr(), workspace.data_ptr(),
               # shape,
               batch,
               # strides
               q.stride(0), q.stride(1),  # q.stride(2), 
               k_cache.stride(-3), k_cache.stride(-2),  # k_cache.stride(-1),
               v_cache.stride(-3), v_cache.stride(-2),  # v_cache.stride(-1),
               output_split.stride(0), output_split.stride(1), output_split.stride(2),  # output_split.stride(3), 
               output_combine.stride(0), output_combine.stride(1), block_table.stride(0),  # block_table.stride(1),
               # algo_info
               algo_info)
        return output_combine
    else:
        raise RuntimeError("Should enable USE_TRITON_DISTRIBUTED_AOT")
