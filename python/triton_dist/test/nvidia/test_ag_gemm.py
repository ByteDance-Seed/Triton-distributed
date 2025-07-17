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
import sys

import nvshmem.core
import torch

from triton_dist.autotuner import contextual_autotune
from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
from triton_dist.utils import (assert_allclose, dist_print, group_profile, initialize_distributed, perf_func, TP_GROUP)

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", default=False)
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="LLaMA-3.1-70B", choices=configs.keys())
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))
    parser.add_argument("--profile", default=False, action="store_true")

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


@register_test("correctness")
def test_ag_gemm(args, autotune=False):
    device = "cuda"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    M = 999 * num_ranks
    N = 1024
    K = 1024

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    debug = args.debug
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    ctx = create_ag_gemm_context(A, B, rank, num_ranks, num_local_ranks=LOCAL_WORLD_SIZE, max_M=M,
                                 for_correctness=debug)
    if rank == 0:
        print(f"all gather with: {ctx.all_gather_method}")

    def func():
        return ag_gemm(A, B, ctx=ctx, persistent=args.persistent, autotune=autotune)

    if autotune:
        _func = func
        func = contextual_autotune(is_dist=True)(lambda: _func())

    if rank == 0 and debug:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        func()
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["MLIR_ENABLE_DUMP"] = "0"

    with group_profile("ag_gemm_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=args.default_group):
        for i in range(5):
            # every time, use a new input data to check correctness
            A.random_()
            B.random_()
            ctx.symm_workspace[:M].random_()
            C = func()

    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    for i in range(num_ranks):
        torch.distributed.barrier(args.default_group)
        if rank == i:
            print(f"Rank {rank}")
            assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)


register_test("correctness_autotune")(lambda args: test_ag_gemm(args, autotune=True))

configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}


@register_test("perf")
def test_perf_ag_gemm_tma(args, autotune=False):
    device = "cuda"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    shape_config = configs[args.shape_id]
    M = shape_config["M"]
    N = shape_config["N"]
    K = shape_config["K"]
    BLOCK_M = shape_config["BM"]
    BLOCK_N = shape_config["BN"]
    BLOCK_K = shape_config["BK"]
    stages = shape_config["Stage"]

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    ag_intranode_stream = torch.cuda.Stream(priority=-1)

    ctx = create_ag_gemm_context(A, B, rank, num_ranks, max_M=M, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                 stages=stages, for_correctness=False, ag_intranode_stream=ag_intranode_stream)

    def func():
        return ag_gemm(A, B, ctx=ctx, persistent=args.persistent, autotune=autotune)

    if autotune:
        _func = func
        func = contextual_autotune(is_dist=True)(lambda: _func())

    C, duration_ms = perf_func(func, iters=10, warmup_iters=5)
    dist_print(f"rank{RANK}: {duration_ms:0.2f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    with group_profile("ag_gemm_perf_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=args.default_group):
        for i in range(20):
            func()
    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    return duration_ms


register_test("perf_tma_autotune")(lambda args: test_perf_ag_gemm_tma(args, autotune=True))

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    initialize_distributed()

    args = get_args()
    if torch.cuda.get_device_capability() < (9, 0):
        if args.persistent:
            print("Persistent is not supported on device with capability < (9, 0). exit...")
            sys.exit()

    args.default_group = TP_GROUP()
    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
