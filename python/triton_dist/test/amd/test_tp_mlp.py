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

import pyrocshmem
import triton
from triton_dist.layers.amd.tp_mlp import TP_MLP
from triton_dist.models.utils import init_model_cpu
from triton_dist.utils import perf_func, dist_print, group_profile

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}


def check_allclose(out: torch.Tensor, golden: torch.Tensor, atol=1e-3, rtol=1e-3):
    """
    Check if two tensors are close within a tolerance.
    """
    assert out.shape == golden.shape, f"Output shape mismatch: {out.shape} vs {golden.shape}"
    if torch.allclose(out, golden, atol=atol, rtol=rtol):
        dist_print(f"✅ [RANK {RANK}] All close.", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    else:
        max_diff = torch.max(torch.abs(out - golden))
        dist_print(f"❗ [RANK {RANK}] Max difference: {max_diff.item()} (atol={atol}, rtol={rtol})")
        dist_print(f"Output: {out}\nGolden: {golden}", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        assert False, f"❌ [RANK {RANK}] Output mismatch."


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
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--ag_gemm_persistent", default=False, action="store_true")
    parser.add_argument("--gemm_rs_persistent", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_op", action="store_true", help="test ag_gemm and gemm_rs separately")

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)
    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP[DTYPE]
    RTOL = THRESHOLD_MAP[DTYPE]

    config = AutoConfig.from_pretrained(args.model)
    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_mlp = hf_model.model.layers[0].mlp.eval()
    mlp = TP_MLP(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    mlp._init_parameters(hf_mlp, verbose=True)

    torch.manual_seed(args.seed)
    M = args.M
    K = hf_mlp.gate_proj.weight.shape[1]
    x = rand_tensor([M, K], dtype=DTYPE)
    hf_mlp = hf_mlp.cuda()
    AG_GEMM_PERSISTENT = args.ag_gemm_persistent
    GEMM_RS_PERSISTENT = args.gemm_rs_persistent

    # Preicision Test

    # golden from HF
    with torch.inference_mode():
        golden = hf_mlp(x)

    # torch fwd
    torch_out = mlp.torch_fwd(x)
    check_allclose(torch_out, golden, atol=ATOL, rtol=RTOL)

    # triton_dist fwd
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    stages = 3

    assert M % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    x_triton_dist = x.split(M_per_rank, dim=0)[RANK].contiguous()
    ag_intranode_stream = [torch.cuda.Stream(priority=-1) for i in range(WORLD_SIZE)]

    mlp._init_ctx(max_M=M, ag_intranode_stream=ag_intranode_stream, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                  stages=stages, serial=False)
    out_triton = mlp.dist_triton_fwd(x_triton_dist)

    out = golden.split(M_per_rank, dim=0)[RANK].contiguous()
    check_allclose(out_triton, out, atol=ATOL, rtol=RTOL)

    profile = args.profile
    # Efficiency Test
    if not args.per_op:
        if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True']:
            mempool = torch.cuda.graph_pool_handle()
            torch_graph = make_cuda_graph(mempool, partial(mlp.torch_fwd, x))
            triton_dist_graph = make_cuda_graph(mempool, partial(mlp.dist_triton_fwd, x_triton_dist))
            torch_run = torch_graph.replay
            triton_dist_run = triton_dist_graph.replay
        else:
            torch_run = partial(mlp.torch_fwd, x)
            triton_dist_run = partial(mlp.dist_triton_fwd, x_triton_dist)

        with group_profile("tp_mlp_graph" if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True'] else "tp_mlp", profile,
                           group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_run, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_run, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

        dist_print(f"torch tp mlp e2e #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"dist-triton tp mlp e2e #{RANK}", dist_triton_perf, f"{torch_perf/dist_triton_perf}x",
                   need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

        # we need to del cuda graphs to avoid dist hang
        if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True']:
            torch_graph.reset()
            triton_dist_graph.reset()
            del torch_graph, triton_dist_graph, mempool

    else:
        check_allclose(mlp.torch_ag_gemm(x_triton_dist), mlp.dist_triton_ag_gemm(x_triton_dist), atol=ATOL, rtol=RTOL)
        # benchmark ag gemm
        if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True']:
            mempool = torch.cuda.graph_pool_handle()
            torch_graph = make_cuda_graph(mempool, partial(mlp.torch_ag_gemm, x_triton_dist))
            triton_dist_graph = make_cuda_graph(mempool, partial(mlp.dist_triton_ag_gemm, x_triton_dist))
            torch_run = torch_graph.replay
            triton_dist_run = triton_dist_graph.replay
        else:
            torch_run = partial(mlp.torch_ag_gemm, x_triton_dist)
            triton_dist_run = partial(mlp.dist_triton_ag_gemm, x_triton_dist)

        N, K = mlp.gate_up_proj.size()
        with group_profile(
                f"tp_mlp_ag_gemm_{M}x{N}x{K}_graph" if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True'] else
                f"tp_mlp_ag_gemm_{M}x{N}x{K}", profile, group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_run, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_run, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

        dist_print(f"torch tp mlp ag_gemm_{M}x{N}x{K} #{RANK}", torch_perf, need_sync=True,
                   allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"dist-triton tp mlp ag_gemm_{M}x{N}x{K} #{RANK}", dist_triton_perf,
                   f"{torch_perf/dist_triton_perf}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

        if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True']:
            torch_graph.reset()
            triton_dist_graph.reset()
            del torch_graph, triton_dist_graph, mempool

        torch.cuda.synchronize()
        torch.distributed.barrier(TP_GROUP)

        # benchmark gemm rs
        N, K = mlp.down_proj.size()
        x = rand_tensor([M, K], dtype=DTYPE)
        check_allclose(mlp.torch_gemm_rs(x), mlp.dist_triton_gemm_rs(x), atol=ATOL, rtol=RTOL)
        if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True']:
            mempool = torch.cuda.graph_pool_handle()
            torch_graph = make_cuda_graph(mempool, partial(mlp.torch_gemm_rs, x))
            triton_dist_graph = make_cuda_graph(mempool, partial(mlp.dist_triton_gemm_rs, x))
            torch_run = torch_graph.replay
            triton_dist_run = triton_dist_graph.replay
        else:
            torch_run = partial(mlp.torch_gemm_rs, x)
            triton_dist_run = partial(mlp.dist_triton_gemm_rs, x)

        with group_profile(f"tp_mlp_gemm_rs_{M}x{N}x{K}", profile, group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_run, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_run, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

        dist_print(f"torch tp mlp gemm_rs_{M}x{N}x{K} #{RANK}", torch_perf, need_sync=True,
                   allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"dist-triton tp mlp gemm_rs_{M}x{N}x{K} #{RANK}", dist_triton_perf,
                   f"{torch_perf/dist_triton_perf}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        if os.getenv('CUDA_GRAPH') in ['1', 'true', 'True']:
            torch_graph.reset()
            triton_dist_graph.reset()
            del torch_graph, triton_dist_graph, mempool

        torch.cuda.synchronize()
        torch.distributed.barrier(TP_GROUP)

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()
