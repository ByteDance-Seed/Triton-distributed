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
from functools import partial
import os
import random
import numpy as np
import datetime

import torch
import pyrocshmem
from triton_dist.kernels.amd.gemm_allreduce import create_gemm_ar_context, gemm_allreduce_op
from triton_dist.utils import dist_print, perf_func, assert_allclose, generate_data, group_profile


def create_rand_tensor(shape, dtype=torch.float16, device="cuda", scale=1.0):
    return (torch.rand(shape, dtype=dtype, device=device) * 2 - 1) * scale


def gemm_allreduce_torch(a: torch.Tensor, b: torch.Tensor, pg: torch.distributed.ProcessGroup):
    """Reference torch implementation for gemm+allreduce"""
    # Perform local GEMM
    c = torch.matmul(a, b.T)
    # Allreduce across ranks
    torch.distributed.all_reduce(c, group=pg)
    return c


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def _make_data(M, N, K, TP_GROUP):
    torch.cuda.synchronize()
    torch.distributed.barrier()
    scale = TP_GROUP.rank() + 1
    data_config = [((M, K), dtype, (0.01 * scale, 0)),  # A
                   ((N, K), dtype, (0.01 * scale, 0)),  # B
                   ]
    generator = generate_data(data_config)
    input, weight = next(generator)
    return input, weight


def run_stress_test(args, TP_GROUP, dtype, atol, rtol):
    """Run stress test with random shapes"""
    RANK = torch.distributed.get_rank()
    WORLD_SIZE = torch.distributed.get_world_size()

    max_M, max_N, max_K = args.M, args.N, args.K

    dist_print(f"Running stress test: {args.stress_rounds} rounds")
    dist_print(f"Max M={max_M}, Max N={max_N}, Max K={max_K}, dtype={dtype}")

    for round_idx in range(args.stress_rounds):
        M = random.randint(256, max_M) // 256 * 256
        N = random.randint(256, max_N) // 256 * 256
        K_per_rank = random.randint(256, max_N) // 256 * 256
        ar_stream = torch.cuda.Stream(priority=-1)
        dist_print(f"\nRound {round_idx + 1}/{args.stress_rounds}: M={M}, N={N}, K_per_rank={K_per_rank}")

        try:
            ctx = create_gemm_ar_context(ar_stream=ar_stream, rank=RANK, world_size=WORLD_SIZE, max_M=M, N=N,
                                         dtype=dtype)

            for _ in range(10):
                a, b = _make_data(M, N, K_per_rank, TP_GROUP)
                output_triton = gemm_allreduce_op(ctx, a, b)
                output_torch = gemm_allreduce_torch(a, b, TP_GROUP)
                assert_allclose(output_triton, output_torch, atol=atol, rtol=rtol, verbose=False)
            dist_print(f"✅ Round {round_idx + 1} passed")

        except Exception as e:
            dist_print(f"❌ Round {round_idx + 1} failed: {str(e)}")
            torch.cuda.synchronize()
            torch.distributed.barrier()
            raise RuntimeError(f"Stress test failed at round {round_idx + 1}: {str(e)}")

    torch.cuda.synchronize()
    torch.distributed.barrier()

    dist_print("\n" + "=" * 60)
    dist_print(f"✅ Stress test completed: All {args.stress_rounds} rounds passed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="perf iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--stress", default=False, action="store_true", help="run stress test with random shapes")
    parser.add_argument("--stress_rounds", type=int, default=10, help="number of stress test rounds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"
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
    torch.set_printoptions(precision=5)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)
    random.seed(args.seed)

    num_ranks = torch.distributed.get_world_size()
    rank_id = torch.distributed.get_rank()

    if rank_id == 0:
        uid = pyrocshmem.rocshmem_get_uniqueid()
        bcast_obj = [uid]
    else:
        bcast_obj = [None]

    torch.distributed.broadcast_object_list(bcast_obj, src=0)
    torch.distributed.barrier()

    pyrocshmem.rocshmem_init_attr(rank_id, num_ranks, bcast_obj[0])

    torch.cuda.synchronize()
    torch.distributed.barrier()
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)

    dtype = DTYPE_MAP[args.dtype]
    atol = THRESHOLD_MAP[dtype]
    rtol = THRESHOLD_MAP[dtype]

    M = args.M
    N = args.N
    K = args.K // WORLD_SIZE

    iters = args.iters
    warmup_iters = args.warmup

    if args.stress:
        run_stress_test(args, TP_GROUP, dtype, atol, rtol)

    # Create context for GEMM+AllReduce
    ar_stream = torch.cuda.Stream(priority=-1)
    ctx = create_gemm_ar_context(ar_stream=ar_stream, rank=RANK, world_size=WORLD_SIZE, max_M=M, N=N, dtype=dtype)
    a, b = _make_data(M, N, K, TP_GROUP)
    torch_output = gemm_allreduce_torch(a, b, TP_GROUP)
    triton_output = gemm_allreduce_op(ctx, a, b)
    assert_allclose(torch_output, triton_output, atol=THRESHOLD_MAP[dtype], rtol=THRESHOLD_MAP[dtype])

    with group_profile("gemm_ar", args.profile, group=TP_GROUP):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch_output, duration_ms_torch = perf_func(partial(gemm_allreduce_torch, a, b, TP_GROUP), iters=iters,
                                                    warmup_iters=warmup_iters)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        triton_output, duration_ms_triton = perf_func(partial(
            gemm_allreduce_op,
            ctx,
            a,
            b,
        ), iters=iters, warmup_iters=warmup_iters)

    dist_print(f"torch #{RANK} {duration_ms_torch:0.2f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"triton #{RANK} {duration_ms_triton:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    speedup = duration_ms_torch / duration_ms_triton
    dist_print(f"Speedup: {speedup:0.2f}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()
