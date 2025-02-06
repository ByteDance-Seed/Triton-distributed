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
from triton.distributed.autotuner import contextual_autotune
from triton.distributed.kernels.nvidia import ag_gemm_intra_node, create_ag_gemm_intra_node_context

import argparse
import os
import sys
import datetime
import numpy as np

import pynvshmem

from triton.distributed.utils import (
    perf_func,
    dist_print,
)

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="")

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


@register_test("correctness_tma")
def test_ag_gemm_tma_intra_node(args, autotune=False):
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

    debug = False

    ag_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()

    ctx = create_ag_gemm_intra_node_context(A, B, rank, num_ranks, BLOCK_M=128, BLOCK_N=256, BLOCK_K=64, stages=3,
                                            for_correctness=debug, ag_stream=ag_stream, gemm_stream=gemm_stream,
                                            serial=False, autotune=False)

    def func():
        return ag_gemm_intra_node(A, B, ctx=ctx)

    if autotune:
        _func = func
        ctx.autotune = True
        func = contextual_autotune(is_dist=True)(lambda: _func())

    if rank == 0 and debug:
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["MLIR_ENABLE_DUMP"] = "1"
        func()
        os.environ["TRITON_ALWAYS_COMPILE"] = "0"
        os.environ["MLIR_ENABLE_DUMP"] = "0"

    for i in range(5):
        # every time, use a new input data to check correctness
        A.copy_(torch.randn([M_per_rank, K], dtype=dtype, device=device))
        B.copy_(torch.randn([N_per_rank, K], dtype=dtype, device=device))
        ctx.workspace_tensors[rank][:M].copy_(torch.randn([M, K], dtype=dtype, device=device))
        pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
        torch.cuda.synchronize()
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
            if not torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3):
                print("Golden")
                print(C_golden)
                print("Output")
                print(C)
                print("Max diff", torch.max(torch.abs(C_golden - C)))
                print("Avg diff", torch.mean(torch.abs(C_golden - C)))
                print("Wrong Answer!")
            else:
                print("Pass!")


register_test("correctness_tma_autotune")(lambda args: test_ag_gemm_tma_intra_node(args, autotune=True))

configs = {
    "LLaMA-7B": {"M": 8192, "N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-8B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "LLaMA-3.1-70B": {"M": 8192, "N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "LLaMA-3.1-405B": {"M": 8192, "N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
    "Mistral-7B": {"M": 8192, "N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "Stage": 5},
    "Qwen2-72B": {"M": 8192, "N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "Stage": 3},
}


@register_test("perf_tma")
def test_perf_ag_gemm_tma_intra_node(args, autotune=False):
    device = "cuda"
    dtype = torch.float16
    rank = args.rank
    num_ranks = args.num_ranks
    if args.shape_id:
        shape_config = configs[args.shape_id]
        M = shape_config["M"]
        N = shape_config["N"]
        K = shape_config["K"]
        BLOCK_M = shape_config["BM"]
        BLOCK_N = shape_config["BN"]
        BLOCK_K = shape_config["BK"]
        stages = shape_config["Stage"]
    else:
        M = 1024 * num_ranks
        N = 28672
        K = 8192
        BLOCK_M = 128
        BLOCK_N = 256
        BLOCK_K = 64
        stages = 3

    assert M % num_ranks == 0
    assert N % num_ranks == 0
    M_per_rank = M // num_ranks
    N_per_rank = N // num_ranks

    A = torch.randn([M_per_rank, K], dtype=dtype, device=device)
    B = torch.randn([N_per_rank, K], dtype=dtype, device=device)

    ag_stream = torch.cuda.Stream()
    gemm_stream = torch.cuda.Stream()

    ctx = create_ag_gemm_intra_node_context(A, B, rank, num_ranks, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                            stages=stages, for_correctness=False, ag_stream=ag_stream,
                                            gemm_stream=gemm_stream, serial=False, autotune=False)

    def func():
        return ag_gemm_intra_node(A, B, ctx=ctx)

    if autotune:
        _func = func
        ctx.autotune = True
        func = contextual_autotune(is_dist=True)(lambda: _func())

    C, perf = perf_func(func, iters=100, warmup_iters=20)
    dist_print(f"rank{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            profile_memory=True,
    ) as profiler:
        for i in range(20):
            func()

    prof_dir = "prof/trace_ag_gemm_intra_node"
    os.makedirs(prof_dir, exist_ok=True)
    profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")
    ag_A = torch.empty([M, K], dtype=dtype, device=device)
    torch.distributed.all_gather_into_tensor(
        ag_A,
        A,
        group=args.default_group,
    )
    C_golden = torch.matmul(ag_A, B.T)
    assert torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3)
    return perf


register_test("perf_tma_autotune")(lambda args: test_perf_ag_gemm_tma_intra_node(args, autotune=True))

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    args = get_args()
    args.default_group = TP_GROUP
    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    torch.distributed.destroy_process_group()
