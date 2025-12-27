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
import functools

import pyrocshmem
import torch

import triton
import triton.language as tl
import triton_dist.language as dl
from triton.language.extra.hip.librocshmem_device import set_rocshmem_ctx
from triton_dist.kernels.amd.common_ops import (barrier_all_kernel, barrier_all_with_ctx_on_stream,
                                                barrier_on_this_grid)
from triton_dist.kernels.amd.memcpy import memcpy_async_kernel
from triton_dist.utils import launch_cooperative_grid_options


@triton.jit(do_not_specialize=["rank"])
def allgather_no_barrier_kernel(
    ctx,
    symm_ptr,
    local_ptr,
    N_per_rank,
    rank,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    npid = tl.num_programs(0)
    pid = tl.program_id(0)
    npid_per_rank = npid // num_ranks
    pid_this_rank = pid % npid_per_rank
    target_rank = pid // npid_per_rank
    dst_ptr = dl.symm_at(symm_ptr, target_rank) + rank * N_per_rank
    total_tiles = tl.cdiv(N_per_rank, BLOCK_SIZE)
    # copy to rank
    for bid in range(pid_this_rank, total_tiles, npid_per_rank):
        offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_per_rank
        val = tl.load(local_ptr + offs, mask)
        tl.store(dst_ptr + offs, val, mask)


@triton.jit(do_not_specialize=["rank"])
def allgather_kernel(
    ctx,
    symm_ptr,
    local_ptr,
    N_per_rank,
    rank,
    num_ranks: tl.constexpr,
    group_barrier_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    npid = tl.num_programs(0)
    pid = tl.program_id(0)
    npid_per_rank = npid // num_ranks
    pid_this_rank = pid % npid_per_rank
    target_rank = pid // npid_per_rank
    dst_ptr = dl.symm_at(symm_ptr, target_rank) + rank * N_per_rank
    dst_ptr = tl.multiple_of(dst_ptr, 16)
    total_tiles = tl.cdiv(N_per_rank, BLOCK_SIZE)

    # copy to rank
    for bid in range(pid_this_rank, total_tiles, npid_per_rank):
        offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_per_rank
        val = tl.load(local_ptr + offs, mask)
        tl.store(dst_ptr + offs, val, mask)

    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)


@triton.jit(do_not_specialize=["rank"])
def allgather_strided_chunked_kernel(
    ctx,
    symm_ptr,  # (M, N), M = M_per_rank * num_ranks
    local_ptr,  # (M_per_rank, N)
    M_per_rank,
    N,
    stride_m,
    stride_n,  # almost always 1
    rank,
    num_ranks: tl.constexpr,
    group_barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # should be as large as possible
    SPLIT_N: tl.constexpr,
):
    """
    with the local_ptr
    """
    set_rocshmem_ctx(ctx)
    npid = tl.num_programs(0)
    pid = tl.program_id(0)

    N_per_split = tl.cdiv(N // SPLIT_N, BLOCK_SIZE_N) * BLOCK_SIZE_N
    tiled_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    tiled_n_per_split = tl.cdiv(N_per_split, BLOCK_SIZE_N)
    total_tiles_per_split = tiled_m * tiled_n_per_split

    symm_ptr += rank * M_per_rank * stride_m

    if num_ranks == 8:
        dst_ptr_rank0 = dl.symm_at(symm_ptr, 0)
        dst_ptr_rank1 = dl.symm_at(symm_ptr, 1)
        dst_ptr_rank2 = dl.symm_at(symm_ptr, 2)
        dst_ptr_rank3 = dl.symm_at(symm_ptr, 3)
        dst_ptr_rank4 = dl.symm_at(symm_ptr, 4)
        dst_ptr_rank5 = dl.symm_at(symm_ptr, 5)
        dst_ptr_rank6 = dl.symm_at(symm_ptr, 6)
        dst_ptr_rank7 = dl.symm_at(symm_ptr, 7)
    elif num_ranks == 4:
        dst_ptr_rank0 = dl.symm_at(symm_ptr, 0)
        dst_ptr_rank1 = dl.symm_at(symm_ptr, 1)
        dst_ptr_rank2 = dl.symm_at(symm_ptr, 2)
        dst_ptr_rank3 = dl.symm_at(symm_ptr, 3)
    elif num_ranks == 2:
        dst_ptr_rank0 = dl.symm_at(symm_ptr, 0)
        dst_ptr_rank1 = dl.symm_at(symm_ptr, 1)
    else:
        tl.static_assert(False, "only support num_ranks=2/4/8")

    for cid in range(SPLIT_N):
        # copy to rank
        for bid in range(pid, total_tiles_per_split, npid):
            bid_m = bid // tiled_n_per_split
            bid_n = bid % tiled_n_per_split
            offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = cid * N_per_split + bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            val = tl.load(local_ptr + offs, mask)
            if num_ranks == 8:
                tl.store(dst_ptr_rank0 + offs, val, mask)
                tl.store(dst_ptr_rank1 + offs, val, mask)
                tl.store(dst_ptr_rank2 + offs, val, mask)
                tl.store(dst_ptr_rank3 + offs, val, mask)
                tl.store(dst_ptr_rank4 + offs, val, mask)
                tl.store(dst_ptr_rank5 + offs, val, mask)
                tl.store(dst_ptr_rank6 + offs, val, mask)
                tl.store(dst_ptr_rank7 + offs, val, mask)
            elif num_ranks == 4:
                tl.store(dst_ptr_rank0 + offs, val, mask)
                tl.store(dst_ptr_rank1 + offs, val, mask)
                tl.store(dst_ptr_rank2 + offs, val, mask)
                tl.store(dst_ptr_rank3 + offs, val, mask)
            elif num_ranks == 2:
                tl.store(dst_ptr_rank0 + offs, val, mask)
                tl.store(dst_ptr_rank1 + offs, val, mask)
            else:
                tl.static_assert(False, "only support num_ranks=2/4/8")

    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)


@triton.jit(do_not_specialize=["pid", "npid", "rank"])
def allgather_strided_chunked_pull_kernel(
    pid,
    npid,
    symm_ptr,  # (M, N), M = M_per_rank * num_ranks
    M_per_rank,
    N,
    stride_m,
    stride_n,  # almost always 1
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # should be as large as possible
    SPLIT_N: tl.constexpr,
):
    N_per_split = tl.cdiv(N // SPLIT_N, BLOCK_SIZE_N) * BLOCK_SIZE_N
    tiled_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    tiled_n_per_split = tl.cdiv(N_per_split, BLOCK_SIZE_N)
    total_tiles_per_split = tiled_m * tiled_n_per_split

    if num_ranks == 8:
        src_ptr_rank0 = dl.symm_at(symm_ptr, 0) + 0 * M_per_rank * stride_m
        src_ptr_rank1 = dl.symm_at(symm_ptr, 1) + 1 * M_per_rank * stride_m
        src_ptr_rank2 = dl.symm_at(symm_ptr, 2) + 2 * M_per_rank * stride_m
        src_ptr_rank3 = dl.symm_at(symm_ptr, 3) + 3 * M_per_rank * stride_m
        src_ptr_rank4 = dl.symm_at(symm_ptr, 4) + 4 * M_per_rank * stride_m
        src_ptr_rank5 = dl.symm_at(symm_ptr, 5) + 5 * M_per_rank * stride_m
        src_ptr_rank6 = dl.symm_at(symm_ptr, 6) + 6 * M_per_rank * stride_m
        src_ptr_rank7 = dl.symm_at(symm_ptr, 7) + 7 * M_per_rank * stride_m
    elif num_ranks == 4:
        src_ptr_rank0 = dl.symm_at(symm_ptr, 0) + 0 * M_per_rank * stride_m
        src_ptr_rank1 = dl.symm_at(symm_ptr, 1) + 1 * M_per_rank * stride_m
        src_ptr_rank2 = dl.symm_at(symm_ptr, 2) + 2 * M_per_rank * stride_m
        src_ptr_rank3 = dl.symm_at(symm_ptr, 3) + 3 * M_per_rank * stride_m
    elif num_ranks == 2:
        src_ptr_rank0 = dl.symm_at(symm_ptr, 0) + 0 * M_per_rank * stride_m
        src_ptr_rank1 = dl.symm_at(symm_ptr, 1) + 1 * M_per_rank * stride_m
    else:
        tl.static_assert(False, "only support num_ranks=2/4/8")

    for cid in range(SPLIT_N):
        # copy to rank
        for bid in range(pid, total_tiles_per_split, npid):
            bid_m = bid // tiled_n_per_split
            bid_n = bid % tiled_n_per_split
            offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = cid * N_per_split + bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs = offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            if num_ranks == 8:
                # unroll too much hurt performance: register spill
                val0 = tl.load(src_ptr_rank0 + offs, mask)
                val1 = tl.load(src_ptr_rank1 + offs, mask)
                if rank != 0:
                    tl.store(symm_ptr + 0 * M_per_rank * stride_m + offs, val0, mask)
                if rank != 1:
                    tl.store(symm_ptr + 1 * M_per_rank * stride_m + offs, val1, mask)
                val2 = tl.load(src_ptr_rank2 + offs, mask)
                val3 = tl.load(src_ptr_rank3 + offs, mask)
                if rank != 2:
                    tl.store(symm_ptr + 2 * M_per_rank * stride_m + offs, val2, mask)
                if rank != 3:
                    tl.store(symm_ptr + 3 * M_per_rank * stride_m + offs, val3, mask)
                val4 = tl.load(src_ptr_rank4 + offs, mask)
                val5 = tl.load(src_ptr_rank5 + offs, mask)
                if rank != 4:
                    tl.store(symm_ptr + 4 * M_per_rank * stride_m + offs, val4, mask)
                if rank != 5:
                    tl.store(symm_ptr + 5 * M_per_rank * stride_m + offs, val5, mask)
                val6 = tl.load(src_ptr_rank6 + offs, mask)
                val7 = tl.load(src_ptr_rank7 + offs, mask)
                if rank != 6:
                    tl.store(symm_ptr + 6 * M_per_rank * stride_m + offs, val6, mask)
                if rank != 7:
                    tl.store(symm_ptr + 7 * M_per_rank * stride_m + offs, val7, mask)
            elif num_ranks == 4:
                tl.store(symm_ptr + 0 * M_per_rank * stride_m + offs, tl.load(src_ptr_rank0 + offs, mask), mask)
                tl.store(symm_ptr + 1 * M_per_rank * stride_m + offs, tl.load(src_ptr_rank1 + offs, mask), mask)
                tl.store(symm_ptr + 2 * M_per_rank * stride_m + offs, tl.load(src_ptr_rank2 + offs, mask), mask)
                tl.store(symm_ptr + 3 * M_per_rank * stride_m + offs, tl.load(src_ptr_rank3 + offs, mask), mask)
            elif num_ranks == 2:
                tl.store(symm_ptr + 0 * M_per_rank * stride_m + offs, tl.load(src_ptr_rank0 + offs, mask), mask)
                tl.store(symm_ptr + 1 * M_per_rank * stride_m + offs, tl.load(src_ptr_rank1 + offs, mask), mask)
            else:
                tl.static_assert(False, "only support num_ranks=2/4/8")


@triton.jit
def make_8x_ptrs(val0, val1, val2, val3, val4, val5, val6, val7):
    vals = tl.cast(
        tl.join(
            tl.join(
                tl.join(tl.full((1, ), tl.cast(val0, tl.uint64, bitcast=True), dtype=tl.uint64),
                        tl.full((1, ), tl.cast(val1, tl.uint64, bitcast=True), dtype=tl.uint64)),
                tl.join(tl.full((1, ), tl.cast(val2, tl.uint64, bitcast=True), dtype=tl.uint64),
                        tl.full((1, ), tl.cast(val3, tl.uint64, bitcast=True), dtype=tl.uint64))),
            tl.join(
                tl.join(tl.full((1, ), tl.cast(val4, tl.uint64, bitcast=True), dtype=tl.uint64),
                        tl.full((1, ), tl.cast(val5, tl.uint64, bitcast=True), dtype=tl.uint64)),
                tl.join(tl.full((1, ), tl.cast(val6, tl.uint64, bitcast=True), dtype=tl.uint64),
                        tl.full((1, ), tl.cast(val7, tl.uint64, bitcast=True), dtype=tl.uint64)))).reshape((8, )),
        val0.dtype, bitcast=True)
    vals = tl.multiple_of(vals, 16)
    return vals


@triton.jit
def make_4x_ptrs(val0, val1, val2, val3):
    vals = tl.cast(
        tl.join(
            tl.join(tl.full((1, ), tl.cast(val0, tl.uint64, bitcast=True), dtype=tl.uint64),
                    tl.full((1, ), tl.cast(val1, tl.uint64, bitcast=True), dtype=tl.uint64)),
            tl.join(tl.full((1, ), tl.cast(val2, tl.uint64, bitcast=True), dtype=tl.uint64),
                    tl.full((1, ), tl.cast(val3, tl.uint64, bitcast=True), dtype=tl.uint64))).reshape((4, )),
        val0.dtype, bitcast=True)
    vals = tl.multiple_of(vals, 16)
    return vals


@triton.jit
def make_2x_ptrs(val0, val1):
    vals = tl.cast(
        tl.join(tl.full((1, ), tl.cast(val0, tl.uint64, bitcast=True), dtype=tl.uint64),
                tl.full((1, ), tl.cast(val1, tl.uint64, bitcast=True), dtype=tl.uint64)).reshape((2, )), val0.dtype,
        bitcast=True)
    vals = tl.multiple_of(vals, 16)
    return vals


@triton.jit(do_not_specialize=["rank"])
def allgather_strided_chunked_pull_packed_kernel(
    symm_ptr,  # (M, N), M = M_per_rank * num_ranks
    M_per_rank,
    N,
    stride_m,
    stride_n,  # almost always 1
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  # should be as large as possible
    BLOCK_SIZE_N: tl.constexpr,  # should be as large as possible
    SPLIT_N: tl.constexpr,
):
    """
    with the local_ptr
    """
    npid = tl.num_programs(0)
    pid = tl.program_id(0)

    tiled_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    N_per_split = N // SPLIT_N
    tiled_n_per_split = tl.cdiv(N_per_split, BLOCK_SIZE_N)
    total_tiles_per_split = tiled_n_per_split * tiled_m

    if num_ranks == 8:
        src_ptrs = make_8x_ptrs(
            dl.symm_at(symm_ptr, 0) + 0 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 1) + 1 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 2) + 2 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 3) + 3 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 4) + 4 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 5) + 5 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 6) + 6 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 7) + 7 * M_per_rank * stride_m,
        )
        dst_ptrs = make_8x_ptrs(
            symm_ptr + 0 * M_per_rank * stride_m,
            symm_ptr + 1 * M_per_rank * stride_m,
            symm_ptr + 2 * M_per_rank * stride_m,
            symm_ptr + 3 * M_per_rank * stride_m,
            symm_ptr + 4 * M_per_rank * stride_m,
            symm_ptr + 5 * M_per_rank * stride_m,
            symm_ptr + 6 * M_per_rank * stride_m,
            symm_ptr + 7 * M_per_rank * stride_m,
        )
    elif num_ranks == 4:
        src_ptrs = make_8x_ptrs(
            dl.symm_at(symm_ptr, 0) + 0 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 1) + 1 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 2) + 2 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 3) + 3 * M_per_rank * stride_m,
        )
        dst_ptrs = make_8x_ptrs(
            symm_ptr + 0 * M_per_rank * stride_m,
            symm_ptr + 1 * M_per_rank * stride_m,
            symm_ptr + 2 * M_per_rank * stride_m,
            symm_ptr + 3 * M_per_rank * stride_m,
        )
    elif num_ranks == 2:
        src_ptrs = make_8x_ptrs(
            dl.symm_at(symm_ptr, 0) + 0 * M_per_rank * stride_m,
            dl.symm_at(symm_ptr, 1) + 1 * M_per_rank * stride_m,
        )
        dst_ptrs = make_8x_ptrs(
            symm_ptr + 0 * M_per_rank * stride_m,
            symm_ptr + 1 * M_per_rank * stride_m,
        )
    else:
        tl.static_assert(False, "only support num_ranks=2/4/8")

    for cid in range(SPLIT_N):
        # copy to rank
        for bid in range(pid, total_tiles_per_split, npid):
            bid_n = bid % tiled_n_per_split
            bid_m = bid // tiled_n_per_split
            offs_m = bid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = cid * N_per_split + bid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            offs = offs_n[None, :] * stride_n + offs_m[:, None] * stride_m
            mask_m = offs_m < M_per_rank
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            # TODO(houqi.1993) with val of 8x2xx1024 of i8 and nthreads=1024, triton won't generate global_load_dwordx4 AMDGCN, but global_load_dwordx2. don't know why
            val = tl.load(src_ptrs[None, :, None] + offs[:, None, :], mask=mask[:, None, :])
            tl.store(dst_ptrs[None, :, None] + offs[:, None, :], val, mask=mask[:, None, :])


@triton.jit(do_not_specialize=["rank"])
def allgather_strided_chunked_pull_ctx_wrapper_kernel(
    ctx,
    symm_ptr,  # (M, N), M = M_per_rank * num_ranks
    M_per_rank,
    N,
    stride_m,
    stride_n,  # almost always 1
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    group_barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # should be as large as possible
    SPLIT_N: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    allgather_strided_chunked_pull_kernel(
        pid,
        npid,
        symm_ptr,  # (M, N), M = M_per_rank * num_ranks
        M_per_rank,
        N,
        stride_m,
        stride_n,  # almost always 1
        rank,
        num_ranks,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,  # should be as large as possible
        SPLIT_N,
    )
    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)


@triton.jit(do_not_specialize=["rank"])
def allgather_strided_chunked_pull_fused_kernel(
    ctx,
    symm_ptr,  # (M, N), M = M_per_rank * num_ranks
    local_ptr,  # (M_per_rank, N)
    M_per_rank,
    N,
    stride_m,
    stride_n,  # almost always 1
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    grid_barrier_ptr,
    group_barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,  # should be as large as possible
    SPLIT_N: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    pid = tl.program_id(0)
    npid = tl.num_programs(0)

    memcpy_async_kernel(local_ptr, symm_ptr + M_per_rank * stride_m * rank, M_per_rank * N, BLOCK_SIZE_M * BLOCK_SIZE_N)
    barrier_on_this_grid(grid_barrier_ptr, False)
    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)
    barrier_on_this_grid(grid_barrier_ptr, False)

    allgather_strided_chunked_pull_kernel(
        pid,
        npid,
        symm_ptr,  # (M, N), M = M_per_rank * num_ranks
        M_per_rank,
        N,
        stride_m,
        stride_n,  # almost always 1
        rank,
        num_ranks,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,  # should be as large as possible
        SPLIT_N,
    )
    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)


@triton.jit(do_not_specialize=["rank"])
def allgather_chunked_pull_fused_packed_kernel(
    ctx,
    symm_ptr,  # (M, N), M = M_per_rank * num_ranks
    local_ptr,  # (M_per_rank, N)
    M_per_rank,
    N,
    stride_m,
    stride_n,  # almost always 1
    rank,
    num_ranks: tl.constexpr,
    grid_barrier_ptr,
    group_barrier_ptr,
    BLOCK_SIZE_M: tl.constexpr,  # should be as large as possible
    BLOCK_SIZE_N: tl.constexpr,  # should be as large as possible
    SPLIT_N: tl.constexpr,
):
    set_rocshmem_ctx(ctx)
    pid = tl.program_id(0)

    memcpy_async_kernel(local_ptr, symm_ptr + M_per_rank * stride_m * rank, M_per_rank * N, 8 * BLOCK_SIZE_N)
    barrier_on_this_grid(grid_barrier_ptr, False)
    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)
    barrier_on_this_grid(grid_barrier_ptr, False)

    allgather_strided_chunked_pull_packed_kernel(
        symm_ptr,  # (M, N), M = M_per_rank * num_ranks
        M_per_rank,
        N,
        stride_m,
        stride_n,  # almost always 1
        rank,
        num_ranks,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,  # should be as large as possible
        SPLIT_N,
    )
    if pid == 0:
        barrier_all_kernel(rank, num_ranks, group_barrier_ptr)


@triton.jit(do_not_specialize=["rank"])
def allgather_opt_kernel(
    ctx,
    symm_ptr,
    local_ptr,
    N_per_rank,
    rank,
    num_ranks: tl.constexpr,
    barrier_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """ compared to allagther_kernel, this use less CTAs to achieve to the same level """
    set_rocshmem_ctx(ctx)
    npid = tl.num_programs(0)
    pid = tl.program_id(0)
    total_tiles = tl.cdiv(N_per_rank, BLOCK_SIZE)
    symm_ptr += rank * N_per_rank

    if num_ranks == 8:
        dst_ptr_rank0 = dl.symm_at(symm_ptr, 0)
        dst_ptr_rank1 = dl.symm_at(symm_ptr, 1)
        dst_ptr_rank2 = dl.symm_at(symm_ptr, 2)
        dst_ptr_rank3 = dl.symm_at(symm_ptr, 3)
        dst_ptr_rank4 = dl.symm_at(symm_ptr, 4)
        dst_ptr_rank5 = dl.symm_at(symm_ptr, 5)
        dst_ptr_rank6 = dl.symm_at(symm_ptr, 6)
        dst_ptr_rank7 = dl.symm_at(symm_ptr, 7)

        # copy to rank
        for bid in range(pid, total_tiles, npid):
            offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N_per_rank
            val = tl.load(local_ptr + offs, mask)
            tl.store(dst_ptr_rank0 + offs, val, mask)
            tl.store(dst_ptr_rank1 + offs, val, mask)
            tl.store(dst_ptr_rank2 + offs, val, mask)
            tl.store(dst_ptr_rank3 + offs, val, mask)
            tl.store(dst_ptr_rank4 + offs, val, mask)
            tl.store(dst_ptr_rank5 + offs, val, mask)
            tl.store(dst_ptr_rank6 + offs, val, mask)
            tl.store(dst_ptr_rank7 + offs, val, mask)
    elif num_ranks == 4:
        dst_ptr_rank0 = dl.symm_at(symm_ptr, 0)
        dst_ptr_rank1 = dl.symm_at(symm_ptr, 1)
        dst_ptr_rank2 = dl.symm_at(symm_ptr, 2)
        dst_ptr_rank3 = dl.symm_at(symm_ptr, 3)
        # copy to rank
        for bid in range(pid, total_tiles, npid):
            offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N_per_rank
            val = tl.load(local_ptr + offs, mask)
            tl.store(dst_ptr_rank0 + offs, val, mask)
            tl.store(dst_ptr_rank1 + offs, val, mask)
            tl.store(dst_ptr_rank2 + offs, val, mask)
            tl.store(dst_ptr_rank3 + offs, val, mask)
    elif num_ranks == 2:
        dst_ptr_rank0 = dl.symm_at(symm_ptr, 0)
        dst_ptr_rank1 = dl.symm_at(symm_ptr, 1)
        # copy to rank
        for bid in range(pid, total_tiles, npid):
            offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < N_per_rank
            val = tl.load(local_ptr + offs, mask)
            tl.store(dst_ptr_rank0 + offs, val, mask)
            tl.store(dst_ptr_rank1 + offs, val, mask)
    else:
        tl.static_assert(False, "only support num_ranks=2/4/8")

    if pid == 0:
        barrier_all_kernel(rank, num_ranks, barrier_ptr)


@triton.jit(do_not_specialize=["rank"])
def allgather_ipc_kernel(
    symm_ptr_ptr,
    local_ptr,
    N_per_rank,
    rank,
    num_ranks: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    npid = tl.num_programs(0)
    pid = tl.program_id(0)
    npid_per_rank = npid // num_ranks
    pid_this_rank = pid % npid_per_rank
    target_rank = pid // npid_per_rank
    dst_ptr = tl.load(symm_ptr_ptr + target_rank).to(local_ptr.dtype)
    total_tiles = tl.cdiv(N_per_rank, BLOCK_SIZE)
    # copy to rank
    for bid in range(pid_this_rank, total_tiles, npid_per_rank):
        offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N_per_rank
        val = tl.load(local_ptr + offs, mask)
        tl.store(dst_ptr + offs, val, mask)


@functools.lru_cache()
def get_pyrocshmem_device_ctx():
    return pyrocshmem.rocshmem_get_device_ctx()


def allgather_no_fuse(full_symm: torch.Tensor, shard: torch.Tensor, barrier: torch.Tensor):
    rank = pyrocshmem.rocshmem_my_pe()
    nranks = pyrocshmem.rocshmem_n_pes()
    assert shard.numel() * nranks == full_symm.numel()
    assert shard.dtype == full_symm.dtype
    ctx = get_pyrocshmem_device_ctx()
    current_stream = torch.cuda.current_stream()
    barrier_all_with_ctx_on_stream(ctx, rank, nranks, barrier, current_stream)
    allgather_no_barrier_kernel[(nranks * 4, )](ctx, full_symm, shard, shard.numel(), rank, nranks,
                                                BLOCK_SIZE=32 * 1024, **launch_cooperative_grid_options())
    barrier_all_with_ctx_on_stream(ctx, rank, nranks, barrier, current_stream)


def allgather(
    full_symm: torch.Tensor,
    shard: torch.Tensor,
    barrier: torch.Tensor,
    workgroups_per_rank: int = 2,
):
    rank = pyrocshmem.rocshmem_my_pe()
    nranks = pyrocshmem.rocshmem_n_pes()
    assert shard.numel() * nranks == full_symm.numel()
    assert shard.dtype == full_symm.dtype
    ctx = get_pyrocshmem_device_ctx()

    workgroups_per_rank = 1
    # use only 8 ranks can achieve similay bandwidth
    allgather_opt_kernel[(nranks * workgroups_per_rank, )](ctx, full_symm, shard, shard.numel(), rank, nranks, barrier,
                                                           BLOCK_SIZE=32 * 1024, num_warps=16,
                                                           **launch_cooperative_grid_options())


def allgather_chunked_pull(
    full_symm: torch.Tensor,
    shard: torch.Tensor,
    group_barrier: torch.Tensor,
    grid_barrier: torch.Tensor,
    SPLIT_N: int,
    workgroups_per_rank: int = 2,
):
    ctx = get_pyrocshmem_device_ctx()
    assert full_symm.dim() == 2 and shard.dim() == 2
    M_per_rank, N = shard.shape
    rank = pyrocshmem.rocshmem_my_pe()
    nranks = pyrocshmem.rocshmem_n_pes()
    assert full_symm.shape == (M_per_rank * nranks, N)
    assert shard.dtype == full_symm.dtype
    assert shard.is_cuda and full_symm.is_cuda
    assert shard.stride() == full_symm.stride()  # just for simple implementation

    # first copy to symm memory, then sync and pull
    M_start = M_per_rank * rank
    M_end = M_start + M_per_rank
    full_symm[M_start:M_end, :].copy_(shard)

    current_stream = torch.cuda.current_stream()
    barrier_all_with_ctx_on_stream(ctx, rank, nranks, group_barrier, current_stream)

    workgroups_per_rank = 2
    BLOCK_SIZE_N = triton.next_power_of_2(N // SPLIT_N)
    BLOCK_SIZE_M = max(1, 16 * 1024 // BLOCK_SIZE_N)
    # use only 8 ranks can achieve similay bandwidth
    allgather_strided_chunked_pull_ctx_wrapper_kernel[(nranks * workgroups_per_rank, )](
        ctx, full_symm, M_per_rank, N, shard.stride(0), shard.stride(1), rank, nranks, group_barrier,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, SPLIT_N=SPLIT_N, num_warps=16)
    barrier_all_with_ctx_on_stream(ctx, rank, nranks, group_barrier, current_stream)


def allgather_chunked_pull_fused(
    full_symm: torch.Tensor,
    shard: torch.Tensor,
    group_barrier: torch.Tensor,
    grid_barrier: torch.Tensor,
    SPLIT_N: int,
    workgroups_per_rank: int = 2,
):
    ctx = get_pyrocshmem_device_ctx()
    assert full_symm.dim() == 2 and shard.dim() == 2
    M_per_rank, N = shard.shape
    rank = pyrocshmem.rocshmem_my_pe()
    nranks = pyrocshmem.rocshmem_n_pes()
    assert full_symm.shape == (M_per_rank * nranks, N)
    assert shard.dtype == full_symm.dtype
    assert shard.is_cuda and full_symm.is_cuda
    assert shard.stride() == full_symm.stride()  # just for simple implementation

    workgroups_per_rank = 2
    BLOCK_SIZE_N = triton.next_power_of_2(N // SPLIT_N)
    BLOCK_SIZE_M = max(1, 16 * 1024 // BLOCK_SIZE_N)
    # use only 8 ranks can achieve similay bandwidth
    allgather_strided_chunked_pull_fused_kernel[(nranks * workgroups_per_rank, )](
        ctx, full_symm, shard, M_per_rank, N, shard.stride(0), shard.stride(1), rank, nranks, grid_barrier,
        group_barrier, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, SPLIT_N=SPLIT_N, num_warps=16,
        **launch_cooperative_grid_options())


def allgather_chunked_pull_packed_fused(
    full_symm: torch.Tensor,
    shard: torch.Tensor,
    group_barrier: torch.Tensor,
    grid_barrier: torch.Tensor,
    SPLIT_N: int,
    workgroups_per_rank: int = 2,
):
    ctx = get_pyrocshmem_device_ctx()
    assert full_symm.dim() == 2 and shard.dim() == 2
    M_per_rank, N = shard.shape
    rank = pyrocshmem.rocshmem_my_pe()
    nranks = pyrocshmem.rocshmem_n_pes()
    assert full_symm.shape == (M_per_rank * nranks, N)
    assert shard.dtype == full_symm.dtype
    assert shard.is_cuda and full_symm.is_cuda
    assert shard.stride() == full_symm.stride()  # just for simple implementation

    workgroups_per_rank = 4
    BLOCK_SIZE_N = triton.next_power_of_2(N // SPLIT_N)
    BLOCK_SIZE_M = max(1, 2 * 1024 // BLOCK_SIZE_N)
    # use only 8 ranks can achieve similar bandwidth
    allgather_chunked_pull_fused_packed_kernel[(nranks * workgroups_per_rank, )](ctx, full_symm, shard, M_per_rank, N,
                                                                                 shard.stride(0), shard.stride(1), rank,
                                                                                 nranks, grid_barrier, group_barrier,
                                                                                 BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                                                 BLOCK_SIZE_N=BLOCK_SIZE_N,
                                                                                 SPLIT_N=SPLIT_N, num_warps=16,
                                                                                 **launch_cooperative_grid_options())


def allgather_chunked(
    full_symm: torch.Tensor,
    shard: torch.Tensor,
    group_barrier: torch.Tensor,
    grid_barrier: torch.Tensor,
    SPLIT_N: int,
    workgroups_per_rank: int = 2,
):
    assert full_symm.dim() == 2 and shard.dim() == 2
    M_per_rank, N = shard.shape
    rank = pyrocshmem.rocshmem_my_pe()
    nranks = pyrocshmem.rocshmem_n_pes()
    assert full_symm.shape == (M_per_rank * nranks, N)
    assert shard.dtype == full_symm.dtype
    assert shard.is_cuda and full_symm.is_cuda
    assert shard.stride() == full_symm.stride()  # just for simple implementation

    ctx = get_pyrocshmem_device_ctx()

    workgroups_per_rank = 1
    BLOCK_SIZE_N = triton.next_power_of_2(N // SPLIT_N)
    BLOCK_SIZE_M = max(1, 32 * 1024 // BLOCK_SIZE_N)
    # use only 8 ranks can achieve similay bandwidth
    allgather_strided_chunked_kernel[(nranks * workgroups_per_rank, )](ctx, full_symm, shard, M_per_rank, N,
                                                                       shard.stride(0), shard.stride(1), rank, nranks,
                                                                       group_barrier, BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                                       BLOCK_SIZE_N=BLOCK_SIZE_N, SPLIT_N=SPLIT_N,
                                                                       num_warps=16,
                                                                       **launch_cooperative_grid_options())
