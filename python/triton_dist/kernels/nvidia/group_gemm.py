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
import torch
from .memory_ops import fill_tensor

GROUP_GEMM_BLOCK_SIZE_M = 128


@triton.jit
def element_at(x: tl.tensor, idx: int) -> tl.tensor:
    return tl.sum(tl.where(tl.arange(0, x.numel) == idx, x, 0))


@triton.jit
def build_block_row_idx_info_kernel(rows_splits_ptr, rows_splits_cum_per_expert_ptr, block_row_idx_to_expert_idx_ptr,
                                    block_row_idx_to_row_offset_ptr, block_row_idx_to_tile_split_ptr,
                                    block_row_idx_to_tile_cumsum_ptr, expert_idx_to_tile_offset_ptr,
                                    num_tiles_total_ptr, E: tl.constexpr, E_PAD: tl.constexpr,
                                    BLOCK_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr):
    sm_id = tl.program_id(0)
    idx = tl.arange(0, E_PAD)
    mask = idx < E
    row_splits = tl.load(rows_splits_ptr + idx, mask=mask, other=0)
    row_cumsums = tl.cumsum(row_splits, axis=0)
    row_offs = row_cumsums - row_splits
    tl.store(rows_splits_cum_per_expert_ptr + idx, row_offs, mask=mask)
    tiles_splits = tl.cdiv(row_splits, BLOCK_SIZE_M)
    tiles_cumsum = tl.cumsum(tiles_splits, axis=0)
    num_tiles_total = tl.sum(tiles_splits, axis=0)

    tl.store(expert_idx_to_tile_offset_ptr + idx, tiles_cumsum - tiles_splits, mask=mask)

    if sm_id == 0:
        tl.store(num_tiles_total_ptr, num_tiles_total)

    for pid in range(sm_id, num_tiles_total, NUM_SMS):

        if pid < num_tiles_total:
            expert_idx = tl.argmax((pid < tiles_cumsum).to(tl.int1), axis=0, tie_break_left=True)
            if expert_idx == 0:
                row_offset = 0
                tile_split = element_at(tiles_splits, 0)
                tile_cumsum = tile_split
            else:
                row_offset = element_at(row_offs, expert_idx)
                tile_split = element_at(tiles_splits, expert_idx)
                tile_cumsum = element_at(tiles_cumsum, expert_idx)
            tl.store(block_row_idx_to_expert_idx_ptr + pid, expert_idx)
            tl.store(block_row_idx_to_row_offset_ptr + pid, row_offset)
            tl.store(block_row_idx_to_tile_split_ptr + pid, tile_split)
            tl.store(block_row_idx_to_tile_cumsum_ptr + pid, tile_cumsum)


@triton.jit(do_not_specialize=["M"])
def fused_build_block_row_idx_info_copy_tensor_kernel(
    rows_splits_ptr,
    rows_splits_cum_per_expert_ptr,
    block_row_idx_to_expert_idx_ptr,
    block_row_idx_to_row_offset_ptr,
    block_row_idx_to_tile_split_ptr,
    block_row_idx_to_tile_cumsum_ptr,
    num_tiles_total_ptr,
    E: tl.constexpr,
    E_PAD: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    NUM_SMS_1: tl.constexpr,
    # for copy
    dst_ptr,
    src_ptr,  #
    M,  #
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    stride_dst_m: tl.constexpr,
    stride_dst_n: tl.constexpr,
    BLOCK_SIZE_CPY_M: tl.constexpr,  #
    BLOCK_SIZE_CPY_N: tl.constexpr,
    NUM_SMS_2: tl.constexpr,
):
    sm_id = tl.program_id(0)

    if sm_id < NUM_SMS_1:
        idx = tl.arange(0, E_PAD)
        mask = idx < E
        row_splits = tl.load(rows_splits_ptr + idx, mask=mask, other=0)
        row_cumsums = tl.cumsum(row_splits, axis=0)
        row_offs = row_cumsums - row_splits
        tl.store(rows_splits_cum_per_expert_ptr + idx, row_offs, mask=mask)
        tiles_splits = tl.cdiv(row_splits, BLOCK_SIZE_M)
        tiles_cumsum = tl.cumsum(tiles_splits, axis=0)
        num_tiles_total = tl.sum(tiles_splits, axis=0)

        if sm_id == 0:
            tl.store(num_tiles_total_ptr, num_tiles_total)

        for pid in range(sm_id, num_tiles_total, NUM_SMS_1):

            if pid < num_tiles_total:
                expert_idx = tl.argmax((pid < tiles_cumsum).to(tl.int1), axis=0, tie_break_left=True)
                if expert_idx == 0:
                    row_offset = 0
                    tile_split = element_at(tiles_splits, 0)
                    tile_cumsum = tile_split
                else:
                    row_offset = element_at(row_offs, expert_idx)
                    tile_split = element_at(tiles_splits, expert_idx)
                    tile_cumsum = element_at(tiles_cumsum, expert_idx)
                tl.store(block_row_idx_to_expert_idx_ptr + pid, expert_idx)
                tl.store(block_row_idx_to_row_offset_ptr + pid, row_offset)
                tl.store(block_row_idx_to_tile_split_ptr + pid, tile_split)
                tl.store(block_row_idx_to_tile_cumsum_ptr + pid, tile_cumsum)
    else:
        sm_id -= NUM_SMS_1
        pid = sm_id
        NUM_COPY_SMS: tl.constexpr = NUM_SMS_2
        num_tiles_m = tl.cdiv(M, BLOCK_SIZE_CPY_M)
        num_tiles_n = tl.cdiv(N, BLOCK_SIZE_CPY_N)
        num_tiles = num_tiles_m * num_tiles_n

        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            pid_m = tile_id // num_tiles_n
            pid_n = tile_id % num_tiles_n
            offs_m = pid_m * BLOCK_SIZE_CPY_M + tl.arange(0, BLOCK_SIZE_CPY_M)
            offs_n = pid_n * BLOCK_SIZE_CPY_N + tl.arange(0, BLOCK_SIZE_CPY_N)
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            data = tl.load(src_ptr + (offs_m[:, None].to(tl.int64) * stride_m) + offs_n[None, :] * stride_n, mask=mask)
            tl.store(dst_ptr + (offs_m[:, None].to(tl.int64) * stride_dst_m) + offs_n[None, :] * stride_dst_n, data,
                     mask=mask)


@triton.jit(do_not_specialize=["M"])
def dot_k_const(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    M,
    N,
    K: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    need_mask: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if need_mask:
            a = tl.load(
                a_ptrs, mask=((tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
                              (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[None, :]))
        else:
            a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[None, :])
        b = tl.load(b_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(a_ptrs.dtype.element_ty)
    if need_mask:
        c_mask = (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        c_mask = (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit(do_not_specialize=["M"])
def dot_2parts_k_const(
    a_ptrs,
    b1_ptrs,
    b2_ptrs,
    c_ptrs,
    M,
    N,
    K1: tl.constexpr,
    K2: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk1: tl.constexpr,
    stride_bk2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    need_mask: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K1, BLOCK_SIZE_K)):
        if need_mask:
            a = tl.load(
                a_ptrs, mask=(tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
                (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K1)[None, :])
        else:
            a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K1)[None, :])
        b = tl.load(b1_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K1)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b1_ptrs += BLOCK_SIZE_K * stride_bk1

    for k in range(0, tl.cdiv(K2, BLOCK_SIZE_K)):
        if need_mask:
            a = tl.load(
                a_ptrs, mask=(tl.arange(0, BLOCK_SIZE_M) < M)[:, None] &
                (k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K2)[None, :])
        else:
            a = tl.load(a_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K2)[None, :])
        b = tl.load(b2_ptrs, mask=(k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K) < K2)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b2_ptrs += BLOCK_SIZE_K * stride_bk2

    accumulator = accumulator.to(a_ptrs.dtype.element_ty)
    if need_mask:
        c_mask = (tl.arange(0, BLOCK_SIZE_M) < M)[:, None] & (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        c_mask = (tl.arange(0, BLOCK_SIZE_N) < N)[None, :]
        tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit(do_not_specialize=["M"])
def moe_grouped_gemm_kernel_nk_const(
    a_ptr,
    b_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    PERSISTENT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tl.load(num_total_tiles_ptr)

    for tile_id in range(pid, total_tiles * num_block_n, num_pid):
        pid_m = tile_id // num_block_n
        pid_n = tile_id % num_block_n

        expert_id = tl.load(expert_ids_ptr + pid_m)
        split_size = tl.load(split_size_ptr + expert_id)
        split_size_cum = tl.load(split_size_cum_ptr + pid_m)
        row_begin = split_size_cum
        tile_num = tl.load(tile_num_ptr + pid_m)
        tile_num_cum = tl.load(tile_num_cum_ptr + pid_m)
        tile_begin = tile_num_cum - tile_num
        local_pid_m = pid_m - tile_begin

        local_pid_m, pid_n = tl.swizzle2d(local_pid_m, pid_n, tile_num, num_block_n, GROUP_SIZE_M)
        row_remain = split_size - local_pid_m * BLOCK_SIZE_M

        offs_bn = (pid_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        b_ptrs = (b_ptr + expert_id.to(tl.int64) * stride_be + offs_bn[None, :] * stride_bn +
                  offs_k[:, None] * stride_bk)

        offs_token = row_begin.to(tl.int64) * stride_ak + local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        a_ptrs = (a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

        if row_remain >= BLOCK_SIZE_M:
            dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                        stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False)
        else:
            dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K, stride_ak,
                        stride_bk, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, True)


@triton.jit(do_not_specialize=["M"])
def moe_grouped_two_weights_split_N_gemm_kernel_nk_const(
    a_ptr,
    b1_ptr,
    b2_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    M,
    N1: tl.constexpr,
    N2: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be1: tl.constexpr,
    stride_bn1: tl.constexpr,
    stride_bk1: tl.constexpr,
    stride_be2: tl.constexpr,
    stride_bn2: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_n1 = tl.cdiv(N1, BLOCK_SIZE_N)
    num_block_n2 = tl.cdiv(N2, BLOCK_SIZE_N)
    num_block_n = num_block_n1 + num_block_n2
    total_tiles = tl.load(num_total_tiles_ptr)

    # tl.static_assert(N1 % BLOCK_SIZE_N == 0)
    # tl.static_assert(N2 % BLOCK_SIZE_N == 0)
    tl.static_assert(stride_bk1 == stride_bk2)

    pid_m = pid // num_block_n
    pid_n = pid % num_block_n

    if pid_m >= total_tiles:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m)
    split_size = tl.load(split_size_ptr + expert_id)
    split_size_cum = tl.load(split_size_cum_ptr + pid_m)
    row_begin = split_size_cum
    tile_num = tl.load(tile_num_ptr + pid_m)
    tile_num_cum = tl.load(tile_num_cum_ptr + pid_m)
    tile_begin = tile_num_cum - tile_num
    local_pid_m = pid_m - tile_begin

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if pid_n < num_block_n1:
        local_pid_m, pid_n = tl.swizzle2d(local_pid_m, pid_n, tile_num, num_block_n1, GROUP_SIZE_M)
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N1
        b_ptrs = (b1_ptr + expert_id * stride_be1 + offs_bn[None, :] * stride_bn1 + offs_k[:, None] * stride_bk1)
        offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        remain_n = min(BLOCK_SIZE_N, N1 - pid_n * BLOCK_SIZE_N)
    else:
        local_pid_m, pid_n = tl.swizzle2d(local_pid_m, pid_n - num_block_n1, tile_num, num_block_n2, GROUP_SIZE_M)
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N2
        b_ptrs = (b2_ptr + expert_id * stride_be2 + offs_bn[None, :] * stride_bn2 + offs_k[:, None] * stride_bk2)
        offs_cn = (N1 + pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        remain_n = min(BLOCK_SIZE_N, N2 - pid_n * BLOCK_SIZE_N)
    row_remain = split_size - local_pid_m * BLOCK_SIZE_M

    offs_token = row_begin.to(tl.int64) * stride_ak + local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    if row_remain >= BLOCK_SIZE_M:
        return dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, remain_n, K, stride_ak, stride_bk1, BLOCK_SIZE_M,
                           BLOCK_SIZE_N, BLOCK_SIZE_K, False)
    else:
        return dot_k_const(a_ptrs, b_ptrs, c_ptrs, row_remain, remain_n, K, stride_ak, stride_bk1, BLOCK_SIZE_M,
                           BLOCK_SIZE_N, BLOCK_SIZE_K, True)


@triton.jit(do_not_specialize=["M"])
def moe_grouped_two_weights_split_K_gemm_kernel_nk_const(
    a_ptr,
    b1_ptr,
    b2_ptr,
    c_ptr,
    expert_ids_ptr,
    split_size_ptr,
    split_size_cum_ptr,
    tile_num_ptr,
    tile_num_cum_ptr,
    num_total_tiles_ptr,
    M,
    N: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be1: tl.constexpr,
    stride_bn1: tl.constexpr,
    stride_bk1: tl.constexpr,
    stride_be2: tl.constexpr,
    stride_bn2: tl.constexpr,
    stride_bk2: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = tl.load(num_total_tiles_ptr)

    pid_m = pid // num_block_n
    pid_n = pid % num_block_n

    if pid_m >= total_tiles:
        return

    expert_id = tl.load(expert_ids_ptr + pid_m)
    split_size = tl.load(split_size_ptr + expert_id)
    split_size_cum = tl.load(split_size_cum_ptr + pid_m)
    row_begin = split_size_cum
    tile_num = tl.load(tile_num_ptr + pid_m)
    tile_num_cum = tl.load(tile_num_cum_ptr + pid_m)
    tile_begin = tile_num_cum - tile_num
    local_pid_m = pid_m - tile_begin

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    local_pid_m, pid_n = tl.swizzle2d(local_pid_m, pid_n, tile_num, num_block_n, GROUP_SIZE_M)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    b1_ptrs = (b1_ptr + expert_id * stride_be1 + offs_bn[None, :] * stride_bn1 + offs_k[:, None] * stride_bk1)
    b2_ptrs = (b2_ptr + expert_id * stride_be2 + offs_bn[None, :] * stride_bn2 + offs_k[:, None] * stride_bk2)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_remain = split_size - local_pid_m * BLOCK_SIZE_M

    offs_token = row_begin.to(tl.int64) * stride_ak + local_pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    a_ptrs = (a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak)

    c_ptrs = (c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn)

    if row_remain >= BLOCK_SIZE_M:
        return dot_2parts_k_const(a_ptrs, b1_ptrs, b2_ptrs, c_ptrs, row_remain,
                                  min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K1, K2, stride_ak, stride_bk1,
                                  stride_bk2, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False)
    else:
        return dot_2parts_k_const(a_ptrs, b1_ptrs, b2_ptrs, c_ptrs, row_remain,
                                  min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N), K1, K2, stride_ak, stride_bk1,
                                  stride_bk2, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, True)


@triton.jit
def transposed_dot(
    a_ptrs,
    b_ptrs,
    c_ptrs,
    split_size,
    N,
    K,
    stride_am: tl.constexpr,
    stride_bm: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    for m in range(0, split_size, BLOCK_SIZE_M):
        a = tl.load(a_ptrs, mask=(m + tl.arange(0, BLOCK_SIZE_M) < split_size)[None, :])
        b = tl.load(b_ptrs, mask=(m + tl.arange(0, BLOCK_SIZE_M) < split_size)[:, None])

        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_M * stride_am
        b_ptrs += BLOCK_SIZE_M * stride_bm

    accumulator = accumulator.to(c_ptrs.dtype.element_ty)
    mask_c = (tl.arange(0, BLOCK_SIZE_N) < N)[:, None] & (tl.arange(0, BLOCK_SIZE_K) < K)[None, :]
    tl.store(c_ptrs, accumulator, mask=mask_c)


@triton.jit
def transposed_moe_grouped_gemm_kernel_nk_const(
    grad_output_ptr,
    orig_input_ptr,
    grad_weight_ptr,
    split_size_ptr,
    split_size_cum_per_expert_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ck: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # tl.static_assert(K % BLOCK_SIZE_K == 0)
    # tl.static_assert(N % BLOCK_SIZE_N == 0)
    pid_g = tl.program_id(axis=1)

    split_size = tl.load(split_size_ptr + pid_g)
    # if split_size <= 0:
    #     return
    split_begin = tl.load(split_size_cum_per_expert_ptr + pid_g)

    num_block_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid // num_block_k
    pid_k = pid % num_block_k
    pid_n, pid_k = tl.swizzle2d(pid_n, pid_k, num_block_n, num_block_k, GROUP_SIZE_M)

    offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
    offs_m = tl.arange(0, BLOCK_SIZE_M)
    b_ptrs = (orig_input_ptr + split_begin.to(tl.int64) * stride_bm + offs_m[:, None] * stride_bm +
              offs_bk[None, :] * stride_bk)

    offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    a_ptrs = (grad_output_ptr + split_begin.to(tl.int64) * stride_am + offs_m[None, :] * stride_am +
              offs_an[:, None] * stride_an)

    offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_ck = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
    c_ptrs = (grad_weight_ptr + pid_g * stride_ce + offs_cn[:, None] * stride_cn + offs_ck[None, :] * stride_ck)

    remain_n = min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N)
    remain_k = min(BLOCK_SIZE_K, K - pid_k * BLOCK_SIZE_K)
    if split_size != 0:
        transposed_dot(
            a_ptrs,
            b_ptrs,
            c_ptrs,
            split_size,
            remain_n,
            remain_k,
            stride_am,
            stride_bm,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
        )
    else:
        mask_c = (tl.arange(0, BLOCK_SIZE_N) < remain_n)[:, None] & (tl.arange(0, BLOCK_SIZE_K) < remain_k)[None, :]
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=c_ptrs.dtype.element_ty), mask=mask_c)


@triton.jit
def transposed_moe_grouped_gemm_kernel_nk_const_persistent(
    grad_output_ptr,
    orig_input_ptr,
    grad_weight_ptr,
    split_size_ptr,
    split_size_cum_per_expert_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ck: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    NUM_SMS = tl.num_programs(axis=0)
    num_block_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_block_k * num_block_n * G

    for tile_id in range(pid, num_tiles, NUM_SMS):
        pid_g = tile_id // (num_block_k * num_block_n)
        pid_k = tile_id % num_block_k
        pid_n = tile_id // num_block_k % num_block_n

        split_size = tl.load(split_size_ptr + pid_g)
        split_begin = tl.load(split_size_cum_per_expert_ptr + pid_g)

        num_block_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_n, pid_k = tl.swizzle2d(pid_n, pid_k, num_block_n, num_block_k, GROUP_SIZE_M)

        offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        b_ptrs = (orig_input_ptr + split_begin.to(tl.int64) * stride_bm + offs_m[:, None] * stride_bm +
                  offs_bk[None, :] * stride_bk)

        offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = (grad_output_ptr + split_begin.to(tl.int64) * stride_am + offs_m[None, :] * stride_am +
                  offs_an[:, None] * stride_an)

        offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        offs_ck = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c_ptrs = (grad_weight_ptr + pid_g * stride_ce + offs_cn[:, None] * stride_cn + offs_ck[None, :] * stride_ck)

        remain_n = min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N)
        remain_k = min(BLOCK_SIZE_K, K - pid_k * BLOCK_SIZE_K)
        if split_size != 0:
            transposed_dot(
                a_ptrs,
                b_ptrs,
                c_ptrs,
                split_size,
                remain_n,
                remain_k,
                stride_am,
                stride_bm,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )
        else:
            mask_c = (tl.arange(0, BLOCK_SIZE_N) < remain_n)[:, None] & (tl.arange(0, BLOCK_SIZE_K) < remain_k)[None, :]
            tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=c_ptrs.dtype.element_ty), mask=mask_c)


@triton.jit
def transposed_moe_grouped_gemm_kernel_nk_const_persistent_dynamic(
    dynamic_schedule_ptr,
    grad_output_ptr,
    orig_input_ptr,
    grad_weight_ptr,
    split_size_ptr,
    split_size_cum_per_expert_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    stride_am: tl.constexpr,
    stride_an: tl.constexpr,
    stride_bm: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ck: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    num_block_k = tl.cdiv(K, BLOCK_SIZE_K)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_block_k * num_block_n * G

    tile_id = tl.atomic_add(dynamic_schedule_ptr, 1)

    while tile_id < num_tiles:
        pid_g = tile_id // (num_block_k * num_block_n)
        pid_k = tile_id % num_block_k
        pid_n = tile_id // num_block_k % num_block_n

        split_size = tl.load(split_size_ptr + pid_g)
        split_begin = tl.load(split_size_cum_per_expert_ptr + pid_g)

        num_block_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
        pid_n, pid_k = tl.swizzle2d(pid_n, pid_k, num_block_n, num_block_k, GROUP_SIZE_M)

        offs_bk = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) % K
        offs_m = tl.arange(0, BLOCK_SIZE_M)
        b_ptrs = (orig_input_ptr + split_begin.to(tl.int64) * stride_bm + offs_m[:, None] * stride_bm +
                  offs_bk[None, :] * stride_bk)

        offs_an = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = (grad_output_ptr + split_begin.to(tl.int64) * stride_am + offs_m[None, :] * stride_am +
                  offs_an[:, None] * stride_an)

        offs_cn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        offs_ck = (pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K))
        c_ptrs = (grad_weight_ptr + pid_g * stride_ce + offs_cn[:, None] * stride_cn + offs_ck[None, :] * stride_ck)

        remain_n = min(BLOCK_SIZE_N, N - pid_n * BLOCK_SIZE_N)
        remain_k = min(BLOCK_SIZE_K, K - pid_k * BLOCK_SIZE_K)
        if split_size != 0:
            transposed_dot(
                a_ptrs,
                b_ptrs,
                c_ptrs,
                split_size,
                remain_n,
                remain_k,
                stride_am,
                stride_bm,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                BLOCK_SIZE_K,
            )
        else:
            mask_c = (tl.arange(0, BLOCK_SIZE_N) < remain_n)[:, None] & (tl.arange(0, BLOCK_SIZE_K) < remain_k)[None, :]
            tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=c_ptrs.dtype.element_ty), mask=mask_c)

        tile_id = tl.atomic_add(dynamic_schedule_ptr, 1)


def moe_grouped_gemm(
    input_data,
    weight,
    expert_ids,
    split_size,
    split_size_cum,
    tile_num,
    tile_num_cum,
    num_tiles_total,
    input_reduce_last_dim=True,
    weight_reduce_last_dim=True,
    output_data=None,
    BLOCK_SIZE_N: int = 256,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 3,
    num_warps=8,
    num_stages=3,
    num_sms=-1,
):
    if not input_reduce_last_dim:
        assert not weight_reduce_last_dim
        K, M = input_data.shape
        G, K_, N = weight.shape
        input_stride1 = input_data.stride(0)
        input_stride0 = input_data.stride(1)
        weight_stride1 = weight.stride(2)
        weight_stride2 = weight.stride(1)
    elif not weight_reduce_last_dim:
        M, K = input_data.shape
        G, K_, N = weight.shape
        input_stride0 = input_data.stride(0)
        input_stride1 = input_data.stride(1)
        weight_stride1 = weight.stride(2)
        weight_stride2 = weight.stride(1)
    else:
        M, K = input_data.shape
        G, N, K_ = weight.shape
        input_stride0 = input_data.stride(0)
        input_stride1 = input_data.stride(1)
        weight_stride1 = weight.stride(1)
        weight_stride2 = weight.stride(2)

    assert K == K_

    if output_data is None:
        output_data = torch.empty([M, N], dtype=input_data.dtype, device=input_data.device)
    else:
        M_, N_ = output_data.shape
        assert M == M_
        assert N == N_

    M_grid = triton.cdiv(M, GROUP_GEMM_BLOCK_SIZE_M) + G
    assert expert_ids.shape[0] >= M_grid, f"expert_ids.shape[0] ({expert_ids.shape[0]}) must be >= M_grid ({M_grid})"
    assert split_size.shape[0] == G, f"split_size.shape[0] ({split_size.shape[0]}) must be == G ({G})"
    assert split_size_cum.shape[
        0] >= M_grid, f"split_size_cum.shape[0] ({split_size_cum.shape[0]}) must be >= M_grid ({M_grid})"
    assert tile_num.shape[0] >= M_grid, f"tile_num.shape[0] ({tile_num.shape[0]}) must be >= M_grid ({M_grid})"
    assert tile_num_cum.shape[
        0] >= M_grid, f"tile_num_cum.shape[0] ({tile_num_cum.shape[0]}) must be >= M_grid ({M_grid})"

    BLOCK_SIZE_M = GROUP_GEMM_BLOCK_SIZE_M

    if num_sms == -1:
        grid = lambda meta: (M_grid * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    else:
        grid = lambda meta: (num_sms, )
    moe_grouped_gemm_kernel_nk_const[grid](
        input_data,
        weight,
        output_data,
        expert_ids,  # aligned
        split_size,
        split_size_cum,
        tile_num,
        tile_num_cum,
        num_tiles_total,
        M,
        N,
        K,
        input_stride0,
        input_stride1,
        weight.stride(0),
        weight_stride1,
        weight_stride2,
        output_data.stride(0),
        output_data.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=num_warps,
        num_stages=num_stages,
        PERSISTENT=num_sms == -1,
    )

    return output_data


def moe_grouped_gemm_2weights(
    input_data,
    weight1,
    weight2,
    expert_ids,
    split_size,
    split_size_cum,
    tile_num,
    tile_num_cum,
    num_tiles_total,
    input_reduce_last_dim=True,
    weight_reduce_last_dim=True,
    output_data=None,
    BLOCK_SIZE_N: int = 256,
    BLOCK_SIZE_K: int = 64,
    GROUP_SIZE_M: int = 3,
    num_warps=8,
    num_stages=3,
):
    BLOCK_SIZE_M = GROUP_GEMM_BLOCK_SIZE_M

    def call_split_N(M_grid, M, N1, N2, K, input_stride0, input_stride1, weight1_stride1, weight1_stride2,
                     weight2_stride1, weight2_stride2):
        grid = lambda meta: (M_grid * (triton.cdiv(N1, meta['BLOCK_SIZE_N']) + triton.cdiv(N2, meta['BLOCK_SIZE_N'])), )
        moe_grouped_two_weights_split_N_gemm_kernel_nk_const[grid](
            input_data,
            weight1,
            weight2,
            output_data,
            expert_ids,  # aligned
            split_size,
            split_size_cum,
            tile_num,
            tile_num_cum,
            num_tiles_total,
            M,
            N1,
            N2,
            K,
            input_stride0,
            input_stride1,
            weight1.stride(0),
            weight1_stride1,
            weight1_stride2,
            weight2.stride(0),
            weight2_stride1,
            weight2_stride2,
            output_data.stride(0),
            output_data.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    def call_split_K(M_grid, M, N, K1, K2, input_stride0, input_stride1, weight1_stride1, weight1_stride2,
                     weight2_stride1, weight2_stride2):
        grid = lambda meta: (M_grid * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        moe_grouped_two_weights_split_K_gemm_kernel_nk_const[grid](
            input_data,
            weight1,
            weight2,
            output_data,
            expert_ids,  # aligned
            split_size,
            split_size_cum,
            tile_num,
            tile_num_cum,
            num_tiles_total,
            M,
            N,
            K1,
            K2,
            input_stride0,
            input_stride1,
            weight1.stride(0),
            weight1_stride1,
            weight1_stride2,
            weight2.stride(0),
            weight2_stride1,
            weight2_stride2,
            output_data.stride(0),
            output_data.stride(1),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if not input_reduce_last_dim:
        assert not weight_reduce_last_dim
        K, M = input_data.shape
        G1, K1, N1 = weight1.shape
        G2, K2, N2 = weight2.shape
        input_stride0 = input_data.stride(1)
        input_stride1 = input_data.stride(0)
        weight1_stride1 = weight1.stride(2)
        weight1_stride2 = weight1.stride(1)
        weight2_stride1 = weight2.stride(2)
        weight2_stride2 = weight2.stride(1)
        kernel_to_call = call_split_K
        to_call_shapes = (M, N1, K1, K2)
        output_shape = (M, N1)
        assert N1 == N2
        assert K == (K1 + K2)
    elif not weight_reduce_last_dim:
        M, K = input_data.shape
        G1, K1, N1 = weight1.shape
        G2, K2, N2 = weight2.shape
        input_stride0 = input_data.stride(0)
        input_stride1 = input_data.stride(1)
        weight1_stride1 = weight1.stride(2)
        weight1_stride2 = weight1.stride(1)
        weight2_stride1 = weight2.stride(2)
        weight2_stride2 = weight2.stride(1)
        kernel_to_call = call_split_K
        to_call_shapes = (M, N1, K1, K2)
        output_shape = (M, N1)
        assert N1 == N2
        assert K == (K1 + K2)
    else:
        M, K = input_data.shape
        G1, N1, K1 = weight1.shape
        G2, N2, K2 = weight2.shape
        input_stride0 = input_data.stride(0)
        input_stride1 = input_data.stride(1)
        weight1_stride1 = weight1.stride(1)
        weight1_stride2 = weight1.stride(2)
        weight2_stride1 = weight2.stride(1)
        weight2_stride2 = weight2.stride(2)
        kernel_to_call = call_split_N
        to_call_shapes = (M, N1, N2, K)
        output_shape = (M, N1 + N2)
        assert K1 == K2 == K

    assert G1 == G2
    G = G1

    if output_data is None:
        output_data = torch.empty(output_shape, dtype=input_data.dtype, device=input_data.device)
    else:
        assert output_data.shape == output_shape

    M_grid = triton.cdiv(M, GROUP_GEMM_BLOCK_SIZE_M) + G
    assert expert_ids.shape[0] >= M_grid, f"expert_ids.shape[0] ({expert_ids.shape[0]}) must be >= M_grid ({M_grid})"
    assert split_size.shape[0] == G, f"split_size.shape[0] ({split_size.shape[0]}) must be == G ({G})"
    assert split_size_cum.shape[
        0] >= M_grid, f"split_size_cum.shape[0] ({split_size_cum.shape[0]}) must be >= M_grid ({M_grid})"
    assert tile_num.shape[0] >= M_grid, f"tile_num.shape[0] ({tile_num.shape[0]}) must be >= M_grid ({M_grid})"
    assert tile_num_cum.shape[
        0] >= M_grid, f"tile_num_cum.shape[0] ({tile_num_cum.shape[0]}) must be >= M_grid ({M_grid})"

    kernel_to_call(M_grid, *to_call_shapes, input_stride0, input_stride1, weight1_stride1, weight1_stride2,
                   weight2_stride1, weight2_stride2)

    return output_data


def transposed_moe_grouped_gemm(
    grad_output,
    original_input,
    split_size,
    split_size_cum_per_expert,
    grad_weight=None,
    BLOCK_SIZE_M: int = 64,
    BLOCK_SIZE_N: int = 128,
    BLOCK_SIZE_K: int = 256,
    GROUP_SIZE_M: int = 4,
    num_warps=8,
    num_stages=3,
    persistent="none",
    sm_margin=0,
):
    M, N = grad_output.shape
    M_, K = original_input.shape
    G = split_size.shape[0]
    assert M == M_
    assert persistent in ["none", "static", "dynamic"]
    if grad_weight is None:
        grad_weight = torch.empty([G, N, K], dtype=grad_output.dtype, device=grad_output.device)
    else:
        G_, N_, K_ = grad_weight.shape
        assert G == G_
        assert N == N_
        assert K == K_

    if persistent == "none":
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']) * triton.cdiv(K, meta['BLOCK_SIZE_K']), G)

        transposed_moe_grouped_gemm_kernel_nk_const[grid](
            grad_output,
            original_input,
            grad_weight,
            split_size,
            split_size_cum_per_expert,
            M,
            N,
            K,
            G,
            grad_output.stride(0),
            grad_output.stride(1),
            original_input.stride(0),
            original_input.stride(1),
            grad_weight.stride(0),
            grad_weight.stride(1),
            grad_weight.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif persistent == "static":
        max_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        grid = lambda meta: (max(1, max_sms - sm_margin), )
        transposed_moe_grouped_gemm_kernel_nk_const_persistent[grid](
            grad_output,
            original_input,
            grad_weight,
            split_size,
            split_size_cum_per_expert,
            M,
            N,
            K,
            G,
            grad_output.stride(0),
            grad_output.stride(1),
            original_input.stride(0),
            original_input.stride(1),
            grad_weight.stride(0),
            grad_weight.stride(1),
            grad_weight.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif persistent == "dynamic":
        max_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_sms = max(1, max_sms - sm_margin)
        grid = lambda meta: (num_sms, )
        dynamic_schedule_ptr = torch.empty([1], dtype=torch.int32, device=grad_output.device)
        fill_tensor(dynamic_schedule_ptr, 0, num_sms=1)
        transposed_moe_grouped_gemm_kernel_nk_const_persistent_dynamic[grid](
            dynamic_schedule_ptr,
            grad_output,
            original_input,
            grad_weight,
            split_size,
            split_size_cum_per_expert,
            M,
            N,
            K,
            G,
            grad_output.stride(0),
            grad_output.stride(1),
            original_input.stride(0),
            original_input.stride(1),
            grad_weight.stride(0),
            grad_weight.stride(1),
            grad_weight.stride(2),
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            GROUP_SIZE_M=GROUP_SIZE_M,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return grad_weight
