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
import argparse
from typing import Optional

import torch

import triton
import triton.language as tl
from triton_dist.kernels.nvidia.reduce_scatter import (kernel_ring_reduce_non_tma)
from triton_dist.utils import perf_func, sleep_async


@triton.jit
def kernel_ring_reduce_atomic_tma(
    c_ptr,  # [M, N]
    out_ptr,  # [M_per_split, N]
    # shape of matrix
    M_per_rank,
    N,
    begin_idx,
    num_splits: tl.constexpr,
    # reduce tile shape
    BLOCK_SIZE_M: tl.constexpr = 256,
    BLOCK_SIZE_N: tl.constexpr = 64,
):
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M_per_rank * num_splits, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)
    num_tiles_m = tl.cdiv(M_per_rank, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_tiles_m * num_tiles_n
    for tile_id in range(pid, total_tiles, num_pid):
        tile_id_m = tile_id // num_tiles_n
        tile_id_n = tile_id % num_tiles_n
        # accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_ptr.dtype.element_ty)
        cur_rank = (begin_idx + 1) % num_splits
        # accum = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
        for i in range(1, num_splits):
            cur_rank = (i + begin_idx + 1) % num_splits
            data = c_desc.load([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N])
            c_desc.atomic_add([tile_id_m * BLOCK_SIZE_M + cur_rank * M_per_rank, tile_id_n * BLOCK_SIZE_N], data)


def ring_reduce_tma(
    input: torch.Tensor,
    output: torch.Tensor,
    num_split: int,
    num_warps: int,
    num_sms: int,
    block_size_m: int,
    block_size_n: int,
):
    M, N = input.shape
    M_per_rank = M // num_split
    assert M_per_rank == output.shape[0]
    assert N == output.shape[1]
    kernel_ring_reduce_atomic_tma[(num_sms, )](
        input,
        output,
        M_per_rank,
        N,
        0,
        num_split,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        num_warps=num_warps,
    )


def ring_reduce_non_tma(
    input: torch.Tensor,
    output: torch.Tensor,
    num_split: int,
    num_warps: int,
    num_sms: int,
    block_size: int,
):
    M, N = output.shape
    M_per_rank = M // num_split
    assert M_per_rank == output.shape[0]
    assert N == output.shape[1]
    kernel_ring_reduce_non_tma[(num_sms, )](
        input,
        output,
        M_per_rank * N,
        0,
        num_split,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", default=10, type=int)
    parser.add_argument("--warmup_iters", default=10, type=int)
    parser.add_argument("--num_splits", default=8, type=int)
    parser.add_argument("--M", "-M", default=None, type=int)
    parser.add_argument("--N", "-N", default=None, type=int)
    parser.add_argument("--num_warps", default=None, type=int)
    parser.add_argument("--num_sms", default=None, type=int)
    parser.add_argument("--block_size_m", default=None, type=int)
    parser.add_argument("--block_size_n", default=None, type=int)
    parser.add_argument("--dtype", default=None, choices=["fp32", "fp16", "int8"])
    return parser.parse_args()


def perf_tma_with(
    dtype,
    M,
    N,
    num_splits,
    num_warps,
    num_sms,
    block_size_m,
    block_size_n,
    iters,
    warmup_iters,
):
    assert M % num_splits == 0
    in_tensor = torch.empty((M, N), dtype=dtype, device="cuda")
    out_tensor = torch.empty((M // num_splits, N), dtype=dtype, device="cuda")
    func = lambda: ring_reduce_tma(
        in_tensor,
        out_tensor,
        num_splits,
        num_warps=num_warps,
        num_sms=num_sms,
        block_size_m=block_size_m,
        block_size_n=block_size_n,
    )
    sleep_async(100)
    _, duration_ms = perf_func(
        func,
        iters=iters,
        warmup_iters=warmup_iters,
    )
    gbps_read = in_tensor.nbytes / 1e9 / (duration_ms * 1e-3)
    gbps_write = out_tensor.nbytes / 1e9 / (duration_ms * 1e-3)
    print(
        f"M: {M} N: {N} of {dtype} num_sms: {num_sms} num_warps: {num_warps} block_size_m: {block_size_m}, {block_size_n} duration: {duration_ms:0.3f} ms read {gbps_read:.2f} GB/s, write {gbps_write:.2f} GB/s"
    )


# TMA descriptors require a global memory allocation
def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)


def _optional_or(arg, default):
    if arg is None:
        return default
    return [arg]


if __name__ == "__main__":
    cap_major, cap_minor = torch.cuda.get_device_capability()
    has_tma = cap_major >= 9
    args = parse_args()
    num_splits = args.num_splits
    triton.set_allocator(alloc_fn)
    for dtype in _optional_or(
        {"fp32": torch.float32, "fp16": torch.float16, "int8": torch.int8}.get(args.dtype, None),
        [torch.int8, torch.float16, torch.float32],
    ):
        for M in _optional_or(args.M, [1024]):  # [8, 16, 32, 64, 128, 256, 1024]:
            for N in _optional_or(args.N, [1024, 2048, 4096]):
                for block_size_m in _optional_or(args.block_size_m, [8, 16, 32, 128]):
                    if block_size_m > M:
                        continue
                    for block_size_n in _optional_or(args.block_size_n, [32, 64, 128]):
                        for num_sms in _optional_or(args.num_sms, [1, 2, 4, 8, 16, 32, 64, 128]):
                            if num_sms > M * N // (block_size_m * block_size_n):
                                continue
                            for num_warps in _optional_or(args.num_warps, [1, 2, 4, 8, 16, 32]):
                                perf_tma_with(
                                    dtype,
                                    M,
                                    N,
                                    num_splits,
                                    num_warps,
                                    num_sms,
                                    block_size_m,
                                    block_size_n,
                                    iters=args.iters,
                                    warmup_iters=args.warmup_iters,
                                )
