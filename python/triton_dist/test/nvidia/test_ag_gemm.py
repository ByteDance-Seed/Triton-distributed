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
import os

import torch

from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, rand_tensor)
from triton_dist.kernels.nvidia.gemm_perf_model import get_tensorcore_tflops
from triton_dist.nv_utils import get_intranode_max_speed_gbps

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser("Usage: python test_ag_gemm.py --case check/perf")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="LLaMA-3.1-70B", choices=LAYER_CONFIGS.keys())
    parser.add_argument("--M", "-M", default=8192, type=int)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--autotune", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    return args


def ag_gemm_torch(A: torch.Tensor, B: torch.Tensor, tp_group: torch.distributed.ProcessGroup):
    M_per_rank, K = A.shape
    K, N = B.shape
    M = M_per_rank * tp_group.size()
    A_full = torch.empty([M, K], dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(A_full, A, group=args.default_group)
    return torch.matmul(A_full, B)


def make_data(M, N, K, dtype: torch.dtype, trans_b, tp_group: torch.distributed.ProcessGroup):
    rank = tp_group.rank()
    num_ranks = tp_group.size()
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks
    scale = (rank + 1) * 0.01

    current_device = torch.cuda.current_device()
    A = rand_tensor([M_per_rank, K], dtype=dtype, device=current_device) * scale
    if trans_b:
        B = rand_tensor([N_per_rank, K], dtype=dtype, device=current_device).T
    else:
        B = rand_tensor([K, N_per_rank], dtype=dtype, device=current_device)

    return A, B


@register_test("check")
def test_ag_gemm(args):
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    M = 4091 * num_ranks
    N = 5120
    K = 1024

    assert M % num_ranks == 0
    assert N % num_ranks == 0

    A, B = make_data(M, N, K, dtype, args.trans_b, args.default_group)

    ctx = create_ag_gemm_context(M, N, K, dtype, rank, num_ranks, num_local_ranks=LOCAL_WORLD_SIZE)
    if rank == 0:
        print(f"all gather with: {ctx.all_gather_method}")

    for i in range(5):
        # every time, use a new input data to check correctness
        A, B = make_data(M, N, K, dtype, args.trans_b, args.default_group)
        ctx.symm_workspace[:M].random_()
        C_triton = ag_gemm(A, B, ctx=ctx, autotune=args.autotune)
        C_torch = ag_gemm_torch(A, B, args.default_group)

        for i in range(num_ranks):
            torch.distributed.barrier(args.default_group)
            if rank == i:
                assert_allclose(C_torch, C_triton, atol=1e-3, rtol=1e-3, verbose=False)
    print("✅ check passed")
    ctx.finalize()


@register_test("perf")
def perf_ag_gemm(args):
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    shape_config = LAYER_CONFIGS[args.shape_id]
    M = args.M
    N = shape_config["N"]
    K = shape_config["K"]

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    N_per_rank = N // num_ranks

    A, B = make_data(M, N, K, dtype, args.trans_b, args.default_group)

    ctx = create_ag_gemm_context(M, N, K, dtype, rank, num_ranks, LOCAL_WORLD_SIZE)

    def func():
        return ag_gemm(A, B, ctx=ctx, autotune=args.autotune)

    C, duration_ms = perf_func(func, iters=10, warmup_iters=5)
    dist_print(f"rank{RANK}: {duration_ms:0.2f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    flops = 2 * M * N_per_rank * K
    tflops = flops / duration_ms / 1e9
    memory_read = dtype.itemsize * M * K + dtype.itemsize * N_per_rank * K
    memory_read_gbps = memory_read / 2**30 / duration_ms * 1e3
    memory_write = dtype.itemsize * M * N_per_rank
    memory_write_gbps = memory_write / 2**30 / duration_ms * 1e3
    memcpy_bus_bw_gbps = M * K * dtype.itemsize / 2**30 / duration_ms * 1e3 * (WORLD_SIZE - 1) / WORLD_SIZE
    print(
        f"rank{RANK}: GEMM {tflops:.2f} TFLOPS, {memory_read_gbps:.2f} GB/s read, {memory_write_gbps:.2f} GB/s write. AllGather {memcpy_bus_bw_gbps:.2f} GB/s"
    )
    print(
        f"GEMM ideal TFLOPS: {get_tensorcore_tflops(dtype)} TFLOPS.  AllGather ideal bus BW: {get_intranode_max_speed_gbps():0.1f} GB/s"
    )

    with group_profile(f"ag_gemm_perf_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=args.default_group):
        for i in range(20):
            func()

    C_torch = ag_gemm_torch(A, B, args.default_group)
    assert_allclose(C, C_torch, atol=1e-3, rtol=1e-3)

    ctx.finalize()
    return duration_ms


if __name__ == "__main__":
    args = get_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
    torch.cuda.set_device(LOCAL_RANK)
    args.default_group = initialize_distributed()

    args.rank = RANK
    args.num_ranks = WORLD_SIZE

    func = ALL_TESTS[args.case]
    func(args)

    finalize_distributed()
