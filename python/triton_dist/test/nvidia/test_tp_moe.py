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
from triton_dist.layers.nvidia.tp_moe import TP_MoE
from triton_dist.models.utils import init_model_cpu
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed, dist_print, nvshmem_barrier_all_on_stream

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
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
    parser.add_argument("--bsz", default=32, type=int, help="batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="sequence length")
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autotune", default=False, action="store_true")

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
    TP_GROUP = initialize_distributed()

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP[DTYPE]
    RTOL = THRESHOLD_MAP[DTYPE]

    config = AutoConfig.from_pretrained(args.model)
    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_mlp = hf_model.model.layers[0].mlp.eval()
    mlp = TP_MoE(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP, autotune=args.autotune)
    mlp._init_parameters(hf_mlp, verbose=True)

    torch.manual_seed(args.seed)
    BSZ = args.bsz
    SEQ_LEN = args.seq_len
    K = mlp.hidden_size
    x = rand_tensor([BSZ, SEQ_LEN, K], dtype=DTYPE)
    hf_mlp = hf_mlp.cuda()

    # Preicision Test

    # golden from HF
    with torch.inference_mode():
        golden, _ = hf_mlp(x)

    # torch fwd
    torch_out = mlp.torch_fwd(x)
    assert_allclose(torch_out, golden, atol=ATOL, rtol=RTOL)

    # torch fwd w/o loop
    torch_out_no_loop = mlp.torch_fwd_no_loop(x)
    assert_allclose(torch_out_no_loop, golden, atol=ATOL, rtol=RTOL)

    # triton_dist fwd
    assert BSZ % WORLD_SIZE == 0
    bsz_per_rank = BSZ // WORLD_SIZE

    def alloc_fn(size: int, alignment: int, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    x_triton_dist = x.split(bsz_per_rank, dim=0)[RANK].contiguous()
    mlp._init_ctx(M=BSZ * SEQ_LEN)
    out_triton = mlp.dist_triton_fwd(x_triton_dist)
    out = golden.split(bsz_per_rank, dim=0)[RANK].contiguous()
    assert_allclose(out_triton, out, atol=ATOL, rtol=RTOL)

    # Efficiency Test
    mempool = torch.cuda.graph_pool_handle()
    triton_dist_graph = make_cuda_graph(mempool, partial(mlp.dist_triton_fwd, x_triton_dist))

    profile = args.profile
    with group_profile("tp_moe", profile, group=TP_GROUP):
        torch.cuda.synchronize()
        _, torch_perf = perf_func(partial(mlp.torch_fwd, x), iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()
        _, torch_no_loop_perf = perf_func(partial(mlp.torch_fwd_no_loop, x), iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()
        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

    dist_print(f"torch tp moe e2e #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch tp moe e2e no loop #{RANK}", torch_no_loop_perf, need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(
        f"dist-triton tp moe e2e #{RANK}", dist_triton_perf,
        f"{torch_perf/dist_triton_perf}x over torch and {torch_no_loop_perf/dist_triton_perf}x over torch no loop",
        need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    del triton_dist_graph

    mlp.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group(TP_GROUP)
