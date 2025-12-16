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
import os
import gc
import argparse
from functools import partial

from triton_dist.kernels.allreduce import to_allreduce_method, get_allreduce_methods
from triton_dist.models.config import ModelConfig
from triton_dist.models import AutoLLM
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import seed_everything, init_model_cpu
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.utils import (finalize_distributed, initialize_distributed, dist_print, nvshmem_barrier_all_on_stream)

RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", default=128, type=int, help="Batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="Sequence length for prefill")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=10, type=int, help="Warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="Performance iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="Enable torch.profiler")
    parser.add_argument("--check", default=False, action="store_true", help="Run correctness check and exit")

    # Aligned arguments for clarity and consistency
    parser.add_argument("--run_type", default="prefill", type=str, choices=["prefill", "decode"],
                        help="Type of performance test to run: 'prefill' or 'decode'.")
    parser.add_argument("--mode", default="ag_rs", type=str, choices=["ag_rs", "allreduce", "gemm_ar"],
                        help="Communication mode: 'ag_rs' (AllGather+RS), 'allreduce', or 'gemm_ar' (Fused GEMM+AR).")
    parser.add_argument("--allreduce_method", type=str, default="two_shot_multimem", choices=get_allreduce_methods(),
                        help="Method for 'allreduce' mode.")
    return parser.parse_args()


def check_allclose(out: torch.Tensor, golden: torch.Tensor, atol=1e-3, rtol=1e-3, mode_name=""):
    """Checks if two tensors are close, with detailed logging on failure."""
    assert out.shape == golden.shape, f"Shape mismatch for {mode_name}: {out.shape} vs {golden.shape}"
    if torch.allclose(out, golden, atol=atol, rtol=rtol):
        dist_print(f"✅ [RANK {RANK}] Correctness check passed for {mode_name}.", need_sync=True, allowed_ranks=[0])
    else:
        max_diff = torch.max(torch.abs(out - golden))
        dist_print(f"❗ [RANK {RANK}] Max difference for {mode_name}: {max_diff.item()} (atol={atol}, rtol={rtol})")
        raise AssertionError(f"❌ [RANK {RANK}] Output mismatch for {mode_name}.")


def make_cuda_graph(mempool, func):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(30):
            func()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


def run_hf_baseline(model_name, input_ids, position_ids, dtype):
    """Runs the HuggingFace baseline model to get the golden output, then releases memory."""
    dist_print("Running HuggingFace baseline to get golden result...")
    hf_model = init_model_cpu(model_name=model_name, dtype=dtype).cuda()
    with torch.inference_mode():
        golden = hf_model.forward(input_ids=input_ids, position_ids=position_ids).logits.float()
    golden = golden[:, -1:, :].contiguous()
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    dist_print("Finished HuggingFace baseline and freed memory.")
    return golden


def run_correctness_check(model, golden, input_ids, position_ids, kv_cache, args, atol, rtol):
    """Encapsulates all correctness checking logic."""
    dist_print("\n--- Running Correctness Checks ---")

    # 1. Check torch implementation
    model.set_fwd(mode='torch')
    logits_torch = model.inference(input_ids=input_ids, position_ids=position_ids, kv_cache=kv_cache)
    check_allclose(logits_torch.softmax(dim=-1, dtype=torch.float32), golden.softmax(dim=-1, dtype=torch.float32),
                   atol=atol, rtol=rtol, mode_name="torch")

    # 2. Check selected Triton mode
    max_M = args.bsz * args.seq_len
    mode_name = f"triton_dist_{args.mode}"
    dist_input = input_ids
    golden_check = golden

    if args.mode == 'ag_rs':
        model.init_triton_dist_ctx(max_M=max_M)
        model.set_fwd(mode='triton_dist')
        dist_input = input_ids.split(args.bsz // WORLD_SIZE, dim=0)[RANK]
        golden_check = golden.split(args.bsz // WORLD_SIZE, dim=0)[RANK]
    elif args.mode == 'allreduce':
        model.init_triton_dist_AR_ctx(max_M=max_M, ar_method=to_allreduce_method(args.allreduce_method))
        model.set_fwd(mode='triton_dist_AR')
        mode_name += f"_{args.allreduce_method}"
    elif args.mode == 'gemm_ar':
        model.init_triton_dist_gemm_ar_ctx(max_M=max_M)
        model.set_fwd(mode='triton_dist_gemm_ar')

    logits_triton = model.inference(input_ids=dist_input, position_ids=position_ids, kv_cache=kv_cache)
    check_allclose(logits_triton.softmax(dim=-1, dtype=torch.float32),
                   golden_check.softmax(dim=-1, dtype=torch.float32), atol=atol, rtol=rtol, mode_name=mode_name)


def run_performance_test(model, run_type, kv_cache, args, tp_group):
    """Encapsulates all performance testing logic."""
    dist_print(f"\n--- Running Performance Test: {run_type.capitalize()} (Mode: {args.mode}) ---")

    # Setup inputs based on run_type
    if run_type == 'prefill':
        seq_len, max_M = args.seq_len, args.bsz * args.seq_len
        input_ids = torch.randint(0, 1000, (args.bsz, seq_len), dtype=torch.long, device="cuda")
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device="cuda").unsqueeze(0).expand(args.bsz, -1)
        kv_cache.kv_offset.fill_(0)
    else:  # decode
        seq_len, max_M = 1, args.bsz
        input_ids = torch.randint(0, 1000, (args.bsz, seq_len), dtype=torch.long, device="cuda")
        position_ids = torch.arange(args.seq_len, args.seq_len + 1, dtype=torch.int64,
                                    device="cuda").unsqueeze(0).expand(args.bsz, -1)
        kv_cache.kv_offset.fill_(args.seq_len)

    # Initialize context for the chosen mode
    if args.mode == 'ag_rs':
        model.init_triton_dist_ctx(max_M=max_M)
    elif args.mode == 'allreduce':
        model.init_triton_dist_AR_ctx(max_M=max_M, ar_method=to_allreduce_method(args.allreduce_method))
    elif args.mode == 'gemm_ar':
        model.init_triton_dist_gemm_ar_ctx(max_M=max_M)

    # Prepare functions for benchmarking
    torch_func = partial(model.inference, input_ids, position_ids, kv_cache, True)

    triton_func_input = input_ids.split(args.bsz //
                                        WORLD_SIZE, dim=0)[RANK].contiguous() if args.mode == 'ag_rs' else input_ids
    triton_func = partial(model.inference, triton_func_input, position_ids, kv_cache, True)

    # Create CUDA graphs BEFORE profiling to exclude compilation overhead.
    mempool = torch.cuda.graph_pool_handle()
    torch_graph = None
    if model.model_type == 'dense':
        model.set_fwd(mode='torch')
        torch_graph = make_cuda_graph(mempool, torch_func)

    if args.mode == 'ag_rs':
        model.set_fwd(mode='triton_dist')
    elif args.mode == 'allreduce':
        model.set_fwd(mode='triton_dist_AR')
    elif args.mode == 'gemm_ar':
        model.set_fwd(mode='triton_dist_gemm_ar')
    triton_dist_graph = make_cuda_graph(mempool, triton_func)

    # Run benchmark using the pre-compiled graphs.
    with group_profile(f"e2e_{run_type}", args.profile, group=tp_group):
        # Benchmark Torch
        if torch_graph:
            _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        else:  # MoE model cannot use CUDA graph, must set fwd_mode right before running.
            model.set_fwd(mode='torch')
            _, torch_perf = perf_func(torch_func, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

        # Benchmark Triton
        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    # Print results
    dist_print(f"torch {run_type} #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    mode_name = f"dist-triton-{args.mode}"
    if args.mode == 'allreduce' and args.allreduce_method:
        mode_name += f"_{args.allreduce_method}"
    dist_print(f"{mode_name} {run_type} #{RANK}", dist_triton_perf, f"{torch_perf/dist_triton_perf:.2f}x",
               need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    # Cleanup
    if torch_graph:
        del torch_graph
    del triton_dist_graph, mempool


if __name__ == "__main__":
    args = parse_args()
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(LOCAL_RANK)
    TP_GROUP = initialize_distributed()

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP.get(DTYPE, 1e-2)
    RTOL = THRESHOLD_MAP.get(DTYPE, 1e-2)
    seed_everything(args.seed)

    model = None
    if args.check:
        # Run baseline on its own to avoid OOM, then free the memory.
        input_ids = torch.randint(10, 1000, (args.bsz, args.seq_len), dtype=torch.long, device="cuda")
        position_ids = torch.arange(0, args.seq_len, dtype=torch.long, device="cuda").unsqueeze(0).repeat(args.bsz, 1)
        golden = run_hf_baseline(args.model, input_ids, position_ids, DTYPE)

        # Now that baseline memory is freed, load our model.
        model_config = ModelConfig(model_name=args.model, max_length=args.seq_len + 4, dtype=DTYPE, rank=RANK,
                                   world_size=WORLD_SIZE, local_only=True)
        model = AutoLLM.from_pretrained(model_config, TP_GROUP)
        kv_cache = KV_Cache(num_layers=model.num_layers, kv_heads=model.num_key_value_heads, head_dim=model.head_dim,
                            batch_size=args.bsz, dtype=DTYPE, max_length=model.max_length, world_size=WORLD_SIZE)

        # Run the correctness check.
        run_correctness_check(model, golden, input_ids, position_ids, kv_cache, args, ATOL, RTOL)
    else:
        # For performance test, just load our model and run.
        model_config = ModelConfig(model_name=args.model, max_length=args.seq_len + 4, dtype=DTYPE, rank=RANK,
                                   world_size=WORLD_SIZE, local_only=True)
        model = AutoLLM.from_pretrained(model_config, TP_GROUP)
        kv_cache = KV_Cache(num_layers=model.num_layers, kv_heads=model.num_key_value_heads, head_dim=model.head_dim,
                            batch_size=args.bsz, dtype=DTYPE, max_length=model.max_length, world_size=WORLD_SIZE)

        run_performance_test(model, args.run_type, kv_cache, args, TP_GROUP)

    if model:
        model.finalize()
    finalize_distributed()
