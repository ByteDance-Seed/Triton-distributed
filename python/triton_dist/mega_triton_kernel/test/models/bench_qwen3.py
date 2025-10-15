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
import triton
from functools import partial
from triton_dist.kernels.allreduce import to_allreduce_method
from triton_dist.kernels.allreduce import get_allreduce_methods
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.mega_triton_kernel.models import DenseModel
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.models import ModelConfig
from triton_dist.models import AutoLLM
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.utils import (
    initialize_distributed,
    finalize_distributed,
    nvshmem_barrier_all_on_stream,
    dist_print,
)


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
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--seq_len", default=128, type=int, help="sequence length")
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="perf iterations")
    parser.add_argument("--allreduce_method", type=str, default="one_shot_multimem", choices=get_allreduce_methods())

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

if __name__ == "__main__":
    args = parse_args()
    TP_GROUP = initialize_distributed(seed=0)
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    assert args.dtype == "bfloat16"
    batch_size = 1
    seq_len = args.seq_len

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    dtype = DTYPE_MAP[args.dtype]
    input_ids = torch.randint(0, 1000, (batch_size, 1), dtype=torch.long, device="cuda")
    position_ids = torch.arange(seq_len, seq_len + 1, dtype=torch.int64,
                                device="cuda").unsqueeze(0).expand(batch_size, -1)
    model_config = ModelConfig(model_name=args.model, max_length=args.seq_len + 4, dtype=dtype, rank=RANK,
                               world_size=WORLD_SIZE, local_only=False)
    # mega kernel
    builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE)
    mege_kernel_model = DenseModel(batch_size, model_config, builder, build_lm_head=True)
    mege_kernel_model.kv_cache.inc_offset(seq_len)

    # torch/dist-triton
    model = AutoLLM.from_pretrained(model_config)
    model.init_triton_dist_AR_ctx(max_M=batch_size, ar_method=to_allreduce_method(args.allreduce_method))

    kv_cache = KV_Cache(num_layers=model.num_layers, kv_heads=model.num_key_value_heads, head_dim=model.head_dim,
                        batch_size=batch_size, dtype=dtype, max_length=model.max_length, world_size=WORLD_SIZE)
    kv_cache.kv_offset.fill_(seq_len)

    model.set_fwd(mode='torch')
    torch_eager = partial(model.inference, input_ids, position_ids, kv_cache, False)

    mempool = torch.cuda.graph_pool_handle()
    model.set_fwd(mode='torch')
    torch_graph = make_cuda_graph(mempool, partial(model.inference, input_ids, position_ids, kv_cache, False))

    model.set_fwd(mode='triton_dist_AR')
    triton_dist_graph = make_cuda_graph(mempool, partial(model.inference, input_ids, position_ids, kv_cache, False))

    with group_profile("tp_e2e_decode", args.profile, group=TP_GROUP):
        torch.cuda.synchronize()
        _, torch_eager_perf = perf_func(torch_eager, iters=args.iters, warmup_iters=args.warmup)

        _, torch_wi_graph_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        _, mega_kernel_perf = perf_func(partial(mege_kernel_model.mega_forwrad, input_ids), iters=args.iters,
                                        warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    dist_print(f"torch eager decode #{RANK}:", torch_eager_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    dist_print(f"torch + cudagraph decode #{RANK}:", torch_wi_graph_perf, need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    dist_print(f"dist-triton-AR_{args.allreduce_method} decode #{RANK}:", dist_triton_perf, need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"mega-kernel decode #{RANK}:", mega_kernel_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(
        "mega-kernel speedup:",
        f"torch = {torch_eager_perf/mega_kernel_perf:.2f}x, torch + cudagraph = {torch_wi_graph_perf/mega_kernel_perf:.2f}, dist-triton = {dist_triton_perf/mega_kernel_perf:.2f}x",
        need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    del torch_graph
    del triton_dist_graph, mempool

    torch.cuda.synchronize()
    model.finalize()
    builder.finalize()
    torch.distributed.barrier()
    finalize_distributed()
