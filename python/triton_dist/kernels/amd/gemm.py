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
import torch
import argparse
import triton.language as tl
from triton_dist.test.utils import assert_allclose
from triton_dist.profiler_utils import perf_func
import triton_dist.tune
from triton_dist.kernels.amd.perf_model import get_max_shared_memory_size


@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit
def kernel_gemm(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr, NUM_XCDS: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rk = tl.arange(0, BLOCK_SIZE_K)
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

    A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
    B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    tl.assume(pid_m > 0)
    tl.assume(pid_n > 0)

    loop_k = tl.cdiv(K, BLOCK_SIZE_K)
    if not EVEN_K:
        loop_k -= 1

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, loop_k):
        a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
        b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
        acc += tl.dot(a, b)
        A_BASE += BLOCK_SIZE_K * stride_ak
        B_BASE += BLOCK_SIZE_K * stride_bk

    if not EVEN_K:
        k = loop_k
        rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        A_BASE = tl.multiple_of(A_BASE, (1, 16))
        B_BASE = tl.multiple_of(B_BASE, (16, 1))
        a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
        b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
        acc += tl.dot(a, b)

    c = acc.to(C.type.element_ty)
    rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
    rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_, c, c_mask)


@triton.heuristics({'EVEN_K': lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0})
@triton.jit
def kernel_gemm_persistent(A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                           BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                           GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr, NUM_XCDS: tl.constexpr,
                           EVEN_K: tl.constexpr):
    pid = tl.program_id(0)
    if NUM_XCDS != 1:
        pid = (pid % NUM_XCDS) * (NUM_SMS // NUM_XCDS) + (pid // NUM_XCDS)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    acc_dtype = tl.float32 if C.type.element_ty != tl.int8 else tl.int32
    for tile_id in range(pid, total_tiles, NUM_SMS):

        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rk = tl.arange(0, BLOCK_SIZE_K)
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)

        A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
        B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
        tl.assume(pid_m > 0)
        tl.assume(pid_n > 0)

        loop_k = tl.cdiv(K, BLOCK_SIZE_K)
        if not EVEN_K:
            loop_k -= 1

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
        for k in range(0, loop_k):
            a = tl.load(tl.multiple_of(A_BASE, (1, 16)))
            b = tl.load(tl.multiple_of(B_BASE, (16, 1)))
            acc += tl.dot(a, b)
            A_BASE += BLOCK_SIZE_K * stride_ak
            B_BASE += BLOCK_SIZE_K * stride_bk

        if not EVEN_K:
            k = loop_k
            rk = k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            A_BASE = A + rm[:, None] * stride_am + rk[None, :] * stride_ak
            B_BASE = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn
            A_BASE = tl.multiple_of(A_BASE, (1, 16))
            B_BASE = tl.multiple_of(B_BASE, (16, 1))
            a = tl.load(A_BASE, mask=rk[None, :] < K, other=0.0)
            b = tl.load(B_BASE, mask=rk[:, None] < K, other=0.0)
            acc += tl.dot(a, b)

        c = acc.to(C.type.element_ty)
        rm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        rn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        rm = tl.max_contiguous(tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M)
        rn = tl.max_contiguous(tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N)
        c_mask = (rm[:, None] < M) & (rn[None, :] < N)
        C_ = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        tl.store(C_, c, c_mask)


NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
DEFAULT_CONFIG = triton.Config(
    {
        "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "NUM_SMS": NUM_SMS, "NUM_XCDS":
        4, "waves_per_eu": 2
    }, num_warps=8, num_stages=2)

DEFAULT_CONFIG_PERSISTENT = triton.Config(
    {
        "BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "NUM_SMS": NUM_SMS, "NUM_XCDS":
        4, "waves_per_eu": 0
    }, num_warps=8, num_stages=2)


def get_config_space():
    NUM_XCDS = 4
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "GROUP_SIZE_M": GROUP_SIZE_M,
                "NUM_XCDS": NUM_XCDS,
                "waves_per_eu": waves_per_eu,
                "matrix_instr_nonkdim": 16,
            }, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_M in [128, 256]
        for BLOCK_SIZE_N in [128, 256]
        for BLOCK_SIZE_K in [32, 64, 128]
        for GROUP_SIZE_M in [1, 4]
        for waves_per_eu in [0, 2, 3]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ]


def get_config_space_persistent():
    # persistent has more limit
    NUM_XCDS = 4
    return [DEFAULT_CONFIG_PERSISTENT] + [
        triton.Config(
            {
                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                "BLOCK_SIZE_K": BLOCK_SIZE_K,
                "GROUP_SIZE_M": GROUP_SIZE_M,
                "NUM_SMS": NUM_SMS,
                "NUM_XCDS": NUM_XCDS,
                "waves_per_eu": waves_per_eu,
                "matrix_instr_nonkdim": 16,
            }, num_stages=num_stages, num_warps=num_warps)
        for BLOCK_SIZE_M in [128, 256]
        for BLOCK_SIZE_N in [128, 256]
        for BLOCK_SIZE_K in [32, 64, 128]
        for GROUP_SIZE_M in [1, 4]
        for waves_per_eu in [0, 2]
        for num_warps in [4, 8]
        for num_stages in [2, 3]
    ]


def key_fn(A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B)


def prune_fn(config, A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    gemm_config: triton.Config = config["config"]
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config.kwargs["BLOCK_SIZE_K"]
    num_stages = gemm_config.num_stages
    itemsize = A.dtype.itemsize
    shared_memory_size = (BLOCK_SIZE_M * BLOCK_SIZE_K + BLOCK_SIZE_K * BLOCK_SIZE_N) * itemsize * (num_stages - 1)
    max_occupancy = int(get_max_shared_memory_size(0)) // shared_memory_size
    if max_occupancy < 1:
        return False
    return True


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_config_space_persistent()], key_fn=key_fn)
def matmul_persistent_triton(A: torch.Tensor, B: torch.Tensor, config: triton.Config = DEFAULT_CONFIG_PERSISTENT):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)
    # print(config)

    kernel_gemm_persistent[(NUM_SMS, )](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1),
                                        C.stride(0), C.stride(1), **config.all_kwargs())
    return C


@triton_dist.tune.autotune(config_space=[{"config": c} for c in get_config_space()], key_fn=key_fn)
def matmul_triton(A: torch.Tensor, B: torch.Tensor, config: triton.Config = DEFAULT_CONFIG):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), dtype=A.dtype, device=A.device)

    kernel_gemm[(NUM_SMS, )](A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0),
                             C.stride(1), **config.all_kwargs())
    return C


def _pretty_duration(duration_ms):
    if duration_ms < 100 * 1e-3:
        return f"{duration_ms * 1e3:0.2f} us"
    if duration_ms < 10:
        return f"{duration_ms:0.3f} ms"
    return f"{duration_ms:0.2f} ms"


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("-M", default=1024, type=int)
        parser.add_argument("-N", default=1024, type=int)
        parser.add_argument("-K", default=1024, type=int)
        parser.add_argument("--trans_a", "--trans_A", default=False, action="store_true")
        parser.add_argument("--trans_b", "--trans_B", default=False, action="store_true")
        parser.add_argument("--iters", type=int, default=10)
        parser.add_argument("--warmup_iters", type=int, default=5)
        parser.add_argument("--autotune", default=False, action="store_true")
        parser.add_argument("--verbose", "-v", default=False, action="store_true")
        parser.add_argument("--persistent", "-p", default=False, action="store_true")
        return parser.parse_args()

    args = parse_args()
    M, N, K = args.M, args.N, args.K
    trans_a, trans_b = args.trans_a, args.trans_b
    dtype = torch.bfloat16
    if trans_a:
        A = torch.randn((K, M), dtype=dtype, device="cuda").T
    else:
        A = torch.randn((M, K), dtype=dtype, device="cuda")

    if trans_b:
        B = torch.randn((N, K), dtype=dtype, device="cuda").T
    else:
        B = torch.randn((K, N), dtype=dtype, device="cuda")

    if args.persistent:
        fn_triton = lambda: matmul_persistent_triton(A, B, autotune=args.autotune, autotune_verbose=args.verbose)
    else:
        fn_triton = lambda: matmul_triton(A, B, autotune=args.autotune, autotune_verbose=args.verbose)
    fn_torch = lambda: torch.matmul(A, B)

    C_torch = fn_torch()
    if torch.any(C_torch.isnan()):
        print("C has nan")
    if torch.any(C_torch.isinf()):
        print("C has inf")

    C_triton = fn_triton()
    assert_allclose(C_triton, C_torch, atol=1e-2, rtol=1e-2)

    gflops = 2 * M * N * K / 1e9
    mem_read_in_mb = (dtype.itemsize * M * K + dtype.itemsize * N * K) / 2**20
    mem_write_in_mb = dtype.itemsize * M * N / 2**20
    for n in range(3):  # in case AMD GPU frequency throttle
        _, duration_ms_triton = perf_func(fn_triton, iters=args.iters, warmup_iters=args.warmup_iters)
        _, duration_ms_torch = perf_func(fn_torch, iters=args.iters, warmup_iters=args.warmup_iters)

        tflops_torch = gflops / duration_ms_torch
        mem_read_gbps_torch = mem_read_in_mb / duration_ms_torch
        mem_write_gbps_torch = mem_write_in_mb / duration_ms_torch

        tflops_triton = gflops / duration_ms_triton
        mem_read_gbps_triton = mem_read_in_mb / duration_ms_triton
        mem_write_gbps_triton = mem_write_in_mb / duration_ms_triton
        print(f"iter {n:02}: torch {_pretty_duration(duration_ms_torch)}/iter {tflops_torch:0.1f} TFLOPS  mem read {mem_read_gbps_torch:0.1f} GB/s write {mem_write_gbps_torch:0.1f} GB/s" \
          f"  triton {_pretty_duration(duration_ms_triton)}/iter {tflops_triton:0.1f} TFLOPS mem read {mem_read_gbps_triton:0.1f} GB/s write {mem_write_gbps_triton:0.1f} GB/s")
