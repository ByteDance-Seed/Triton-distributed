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

from triton_dist.kernels.allreduce import to_allreduce_method, get_allreduce_methods
from triton_dist.layers.nvidia.tp_attn import TP_Attn, _set_cos_sin_cache
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.utils import init_model_cpu
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import initialize_distributed, dist_print, nvshmem_barrier_all_on_stream, finalize_distributed, rand_tensor

THRESHOLD_MAP = {torch.float16: (1e-2, 1e-2), torch.bfloat16: (5e-2, 5e-2)}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", default=128, type=int, help="Batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="Sequence length for prefill")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="Warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="Performance iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=False, action="store_true", help="Enable torch.profiler")

    # Argument to select which test to run: prefill or decode
    parser.add_argument(
        "--run_type", default="prefill", type=str, choices=["prefill", "decode"],
        help="Type of test to run: 'prefill' for long sequences or 'decode' for single-token generation.")

    # Argument to select the communication strategy
    parser.add_argument(
        "--mode", type=str, default="ag_rs", choices=["ag_rs", "allreduce", "gemm_ar"], help=
        "Communication strategy mode: 'ag_rs' (AllGather+ReduceScatter), 'allreduce', or 'gemm_ar' (Fused GEMM+AllReduce)."
    )
    # Strategy-specific arguments
    parser.add_argument("--allreduce_method", type=str, default="two_shot_multimem", choices=get_allreduce_methods(),
                        help="All-reduce method (allreduce mode only)")

    return parser.parse_args()


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


def run_benchmark(test_name: str, torch_func, triton_func, args: argparse.Namespace, group, rank: int, world_size: int):
    """Encapsulates the benchmarking logic for a given torch and triton function."""
    mempool = torch.cuda.graph_pool_handle()
    torch_graph = make_cuda_graph(mempool, torch_func)
    triton_dist_graph = make_cuda_graph(mempool, triton_func)

    with group_profile(f"tp_attn_{test_name}", args.profile, group=group):
        torch.cuda.synchronize()
        _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

        torch.cuda.synchronize()
        _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()

    dist_print(f"torch {test_name} #{rank}", torch_perf, need_sync=True, allowed_ranks=list(range(world_size)))
    dist_print(f"dist-triton {test_name} #{rank}", dist_triton_perf, f"{torch_perf/dist_triton_perf:.2f}x",
               need_sync=True, allowed_ranks=list(range(world_size)))

    del torch_graph, triton_dist_graph, mempool
    torch.cuda.empty_cache()


def run_attention_test(run_type: str, attn: TP_Attn, cos_sin_cache, kv_cache: KV_Cache, args: argparse.Namespace,
                       rank: int, world_size: int, tp_group, dtype: torch.dtype, atol: float, rtol: float):
    """
    Runs a full test (correctness and performance) for a given attention run_type (prefill/decode).
    The communication strategy is determined by args.mode.
    """
    # 1. Setup inputs based on 'prefill' or 'decode' run_type
    if run_type == "prefill":
        seq_len = args.seq_len
        position_ids = torch.arange(0, seq_len, dtype=torch.int64, device="cuda").unsqueeze(0).expand(args.bsz, -1)
        kv_cache.kv_offset.fill_(0)
    else:  # decode
        seq_len = 1
        position_ids = torch.arange(args.seq_len, args.seq_len + 1, dtype=torch.int64,
                                    device="cuda").unsqueeze(0).expand(args.bsz, -1)
        kv_cache.kv_offset.fill_(args.seq_len)
        kv_cache.rand_fill_kv_cache(args.seq_len)

    x = rand_tensor([args.bsz, seq_len, attn.wqkv.shape[1]], dtype=dtype)
    M = args.bsz * seq_len
    bsz_per_rank = args.bsz // world_size

    # 2. Get PyTorch baseline result
    torch_func = partial(attn.torch_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
    golden_output = torch_func()

    # 3. Configure Triton function and inputs based on communication strategy (args.mode)
    triton_func = None
    golden_for_assert = None
    test_name_suffix = ""

    ag_intranode_stream = torch.cuda.Stream(priority=-1)
    ag_internode_stream = torch.cuda.Stream()

    if args.mode == 'ag_rs':
        assert args.bsz % world_size == 0, f"Batch size {args.bsz} must be divisible by world size {world_size} for 'ag_rs' mode."
        dist_x = x.split(bsz_per_rank, dim=0)[rank].contiguous()
        attn._init_ctx(max_M=M, ag_intranode_stream=ag_intranode_stream, ag_internode_stream=ag_internode_stream)
        triton_func = partial(attn.dist_triton_fwd, dist_x, position_ids, cos_sin_cache, kv_cache, layer_idx=0,
                              autotune=True)
        golden_for_assert = golden_output.split(bsz_per_rank, dim=0)[rank].contiguous()
        test_name_suffix = "ag_rs"
    elif args.mode == 'allreduce':
        attn._init_AR_ctx(max_M=M, method=to_allreduce_method(args.allreduce_method), dtype=dtype)
        triton_func = partial(attn.dist_triton_AR_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
        golden_for_assert = golden_output
        test_name_suffix = f"AR_{args.allreduce_method}"
    elif args.mode == 'gemm_ar':
        attn._init_gemm_ar_ctx(M, dtype)
        triton_func = partial(attn.dist_triton_gemm_ar_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
        golden_for_assert = golden_output
        test_name_suffix = "gemm_ar"

    # 4. Correctness Check
    triton_output = triton_func()
    assert_allclose(triton_output, golden_for_assert, atol=atol, rtol=rtol)

    # 5. Performance Benchmark
    test_name = f"attn_{run_type}_{test_name_suffix}"
    run_benchmark(test_name, torch_func, triton_func, args, tp_group, rank, world_size)


if __name__ == "__main__":
    args = parse_args()
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    torch.manual_seed(args.seed)

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL, RTOL = THRESHOLD_MAP.get(DTYPE, (1e-2, 1e-2))

    # Common setup
    config = AutoConfig.from_pretrained(args.model)
    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_attn = hf_model.model.layers[0].self_attn.eval().cuda()

    attn = TP_Attn(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    attn._init_parameters(hf_attn, verbose=True)

    cos_sin_cache = _set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.cuda(), max_length=args.seq_len + 128)
    kv_cache = KV_Cache(
        num_layers=1,
        batch_size=args.bsz,
        max_length=args.seq_len + 128,
        kv_heads=hf_attn.config.num_key_value_heads,
        head_dim=hf_attn.head_dim,
        dtype=DTYPE,
        world_size=WORLD_SIZE,
    )

    # --- Run the selected test (prefill or decode) based on the --run_type argument ---
    dist_print(f"\n===== Running {args.run_type.capitalize()} Test (Communication Mode: {args.mode}) =====")
    run_attention_test(run_type=args.run_type, attn=attn, cos_sin_cache=cos_sin_cache, kv_cache=kv_cache, args=args,
                       rank=RANK, world_size=WORLD_SIZE, tp_group=TP_GROUP, dtype=DTYPE, atol=ATOL, rtol=RTOL)

    # Final cleanup
    attn.finalize()
    finalize_distributed()
