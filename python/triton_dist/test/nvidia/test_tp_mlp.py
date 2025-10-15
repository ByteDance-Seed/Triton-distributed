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

import os
import argparse
import torch
import torch.distributed
from functools import partial
from transformers import AutoConfig

import triton
import nvshmem.core
from triton_dist.kernels.allreduce import to_allreduce_method
from triton_dist.layers.nvidia.tp_mlp import TP_MLP
from triton_dist.models.utils import init_model_cpu
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed, dist_print, nvshmem_barrier_all_on_stream

from triton_dist.kernels.allreduce import get_allreduce_methods

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def rand_tensor(shape: list[int], dtype: torch.dtype):
    if dtype in [torch.int32, torch.int8]:
        return torch.randint(-127, 128, shape, dtype=dtype).cuda()
    else:
        return torch.rand(shape, dtype=dtype).cuda() / 10


def make_cuda_graph(mempool, func):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(30):
            func()
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", default=4096, type=int, help="M dimension of the input tensor")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--autotune", default=True, action=argparse.BooleanOptionalAction,
                        help="enable autotune, default=True")

    # Refactored mode selection
    parser.add_argument(
        "--mode", type=str, default="ag_rs", choices=["ag_rs", "allreduce", "gemm_ar"], help=
        "Execution mode for TP MLP. 'ag_rs' for AllGather+ReduceScatter, 'allreduce' for AllReduce, 'gemm_ar' for Fused GEMM+AllReduce."
    )
    # Arguments specific to 'allreduce' mode
    parser.add_argument("--allreduce_method", type=str, default="two_shot_multimem", choices=get_allreduce_methods(),
                        help="All-reduce method (allreduce mode only)")

    return parser.parse_args()


def run_benchmark(test_name: str, torch_func, triton_func, args: argparse.Namespace, group, rank: int, world_size: int):
    """
    A helper function to encapsulate the benchmarking logic.
    """
    mempool = torch.cuda.graph_pool_handle()
    torch_graph = make_cuda_graph(mempool, torch_func)
    triton_dist_graph = make_cuda_graph(mempool, triton_func)

    with group_profile(f"tp_mlp_{test_name}", args.profile, group=group):
        torch.cuda.synchronize()
        _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

    dist_print(f"TP MLP {test_name} #{rank} torch {torch_perf:0.3f} ms/iter" \
               f" dist-triton {dist_triton_perf:0.3f} ms/iter", f"speedup {torch_perf/dist_triton_perf:0.3f}x",
               need_sync=True, allowed_ranks=list(range(world_size)))

    del torch_graph, triton_dist_graph, mempool
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP[DTYPE]
    RTOL = THRESHOLD_MAP[DTYPE]
    torch.manual_seed(args.seed)

    # Common setup
    config = AutoConfig.from_pretrained(args.model)
    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_mlp = hf_model.model.layers[0].mlp.eval().cuda()

    mlp = TP_MLP(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    mlp._init_parameters(hf_mlp, verbose=True)
    M = args.M
    K = hf_mlp.gate_proj.weight.shape[1]
    x = rand_tensor([M, K], dtype=DTYPE)

    # Golden reference from HuggingFace
    with torch.inference_mode():
        golden = hf_mlp(x)

    # Torch baseline correctness check
    torch_out = mlp.torch_fwd(x)
    assert_allclose(torch_out, golden, atol=ATOL, rtol=RTOL)
    # Mode-specific execution
    if args.mode == 'ag_rs':
        assert M % WORLD_SIZE == 0
        M_per_rank = M // WORLD_SIZE
        x_triton_dist = x.split(M_per_rank, dim=0)[RANK].contiguous()

        # Custom allocator for triton
        def alloc_fn(size: int, alignment: int, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        # Init context for ag_rs
        mlp._init_ctx(max_M=M, ag_intranode_stream=torch.cuda.Stream(priority=-1),
                      ag_internode_stream=torch.cuda.Stream())

        # Correctness check
        out_triton = mlp.dist_triton_fwd(x_triton_dist)
        out_golden_slice = golden.split(M_per_rank, dim=0)[RANK].contiguous()
        assert_allclose(out_triton, out_golden_slice, atol=ATOL, rtol=RTOL)

        # E2E Benchmark
        triton_fwd_func = partial(mlp.dist_triton_fwd, x_triton_dist, autotune=args.autotune)
        run_benchmark("e2e", partial(mlp.torch_fwd, x), triton_fwd_func, args, TP_GROUP, RANK, WORLD_SIZE)

        # Sub-module benchmarks for ag_rs mode
        N_ag, K_ag = mlp.gate_up_proj.size()
        triton_ag_gemm_func = partial(mlp.dist_triton_ag_gemm, x_triton_dist, autotune=args.autotune)
        assert_allclose(mlp.torch_ag_gemm(x_triton_dist), triton_ag_gemm_func(), atol=ATOL, rtol=RTOL)
        run_benchmark(f"ag_gemm_{M}x{N_ag}x{K_ag}", partial(mlp.torch_ag_gemm, x_triton_dist), triton_ag_gemm_func,
                      args, TP_GROUP, RANK, WORLD_SIZE)

        N_rs, K_rs = mlp.down_proj.size()
        x_rs_input = rand_tensor([M, K_rs], dtype=DTYPE)
        triton_gemm_rs_func = partial(mlp.dist_triton_gemm_rs, x_rs_input, autotune=args.autotune)
        assert_allclose(mlp.torch_gemm_rs(x_rs_input), triton_gemm_rs_func(), atol=ATOL, rtol=RTOL)
        run_benchmark(f"gemm_rs_{M}x{N_rs}x{K_rs}", partial(mlp.torch_gemm_rs, x_rs_input), triton_gemm_rs_func, args,
                      TP_GROUP, RANK, WORLD_SIZE)

    elif args.mode == 'allreduce':
        ar_method = to_allreduce_method(args.allreduce_method)
        mlp._init_AR_ctx(max_M=M, method=ar_method, dtype=DTYPE)

        # Correctness check
        out_triton_AR = mlp.dist_triton_AR_fwd(x)
        assert_allclose(out_triton_AR, golden, atol=ATOL, rtol=RTOL)

        # E2E Benchmark
        test_name = f"e2e_AR_{args.allreduce_method}"
        run_benchmark(test_name, partial(mlp.torch_fwd, x), partial(mlp.dist_triton_AR_fwd, x), args, TP_GROUP, RANK,
                      WORLD_SIZE)

    elif args.mode == 'gemm_ar':
        mlp._init_gemm_ar_ctx(max_M=M, dtype=DTYPE)

        # Correctness check
        out_triton_gemm_ar = mlp.dist_triton_gemm_ar_fwd(x)
        assert_allclose(out_triton_gemm_ar, golden, atol=ATOL, rtol=RTOL)

        # E2E Benchmark
        run_benchmark("e2e_gemm_ar", partial(mlp.torch_fwd, x), partial(mlp.dist_triton_gemm_ar_fwd, x), args, TP_GROUP,
                      RANK, WORLD_SIZE)

    # Final cleanup
    mlp.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group(TP_GROUP)
