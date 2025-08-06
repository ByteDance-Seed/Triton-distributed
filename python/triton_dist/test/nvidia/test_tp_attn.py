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
from transformers import AutoModelForCausalLM
import nvshmem.core

from triton_dist.kernels.allreduce import to_allreduce_method
from triton_dist.kernels.allreduce import get_allreduce_methods
from triton_dist.layers.nvidia.tp_attn import TP_Attn, _set_cos_sin_cache
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.utils import assert_allclose, initialize_distributed, perf_func, dist_print, group_profile, nvshmem_barrier_all_on_stream

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
    parser.add_argument("--bsz", default=128, type=int, help="batch size")
    parser.add_argument("--seq_len", default=128, type=int, help="sequence length")
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--mode", default="prefill", type=str, choices=["prefill", "decode"],
                        help="mode of operation: prefill or decode")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--ag_gemm_persistent", default=False, action="store_true")
    parser.add_argument("--gemm_rs_persistent", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # for triton_dist_AR
    parser.add_argument("--use_allreduce", default=False, action="store_true")
    parser.add_argument("--allreduce_method", type=str, default="", choices=get_allreduce_methods())

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
    MODE = args.mode

    hf_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=DTYPE,
                                                    attn_implementation="flash_attention_2")
    hf_attn = hf_model.model.layers[0].self_attn.eval()
    attn = TP_Attn(rank=RANK, world_size=WORLD_SIZE, group=TP_GROUP)
    cos_sin_cache = _set_cos_sin_cache(hf_model.model.rotary_emb.inv_freq.cuda(), max_length=args.seq_len + 128)
    attn._init_parameters(hf_attn, verbose=True)

    torch.manual_seed(args.seed)
    BSZ = args.bsz
    SEQ_LEN = args.seq_len
    K = attn.wqkv.shape[1]
    AG_GEMM_PERSISTENT = args.ag_gemm_persistent
    GEMM_RS_PERSISTENT = args.gemm_rs_persistent

    kv_cache = KV_Cache(
        num_layers=1,
        batch_size=BSZ,
        max_length=SEQ_LEN + 8,
        kv_heads=hf_attn.config.num_key_value_heads,
        head_dim=hf_attn.head_dim,
        dtype=DTYPE,
        world_size=WORLD_SIZE,
    )
    bsz_per_rank = BSZ // WORLD_SIZE
    if not args.use_allreduce:
        assert BSZ % WORLD_SIZE == 0, f"BSZ {BSZ} must be divisible by WORLD_SIZE {WORLD_SIZE}"
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 128
    stages = 3
    ag_intranode_stream = torch.cuda.Stream(priority=-1)
    ag_internode_stream = torch.cuda.Stream()
    profile = args.profile

    # prefill
    if MODE == "prefill":
        # Preicision Test
        position_ids = torch.arange(0, SEQ_LEN, dtype=torch.int64, device="cuda").unsqueeze(0).expand(BSZ, -1)
        x = rand_tensor([BSZ, SEQ_LEN, K], dtype=DTYPE)
        # golden prefill from HF
        with torch.inference_mode():
            t = torch.arange(4096, device="cuda", dtype=hf_model.model.rotary_emb.inv_freq.cuda().dtype)
            freqs = torch.outer(t, hf_model.model.rotary_emb.inv_freq.cuda())
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos().to(torch.bfloat16)
            sin = emb.sin().to(torch.bfloat16)
            position_embeddings = hf_model.model.rotary_emb(x, position_ids)
            golden = hf_attn.cuda().forward(x, position_ids=position_ids, position_embeddings=position_embeddings,
                                            attention_mask=None)[0]

        # torch prefill
        out_torch = attn.torch_fwd(x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
        assert_allclose(out_torch, golden, atol=ATOL, rtol=RTOL)

        M = BSZ * SEQ_LEN
        if not args.use_allreduce:
            # dist triton prefill
            attn._init_ctx(max_M=M, ag_intranode_stream=ag_intranode_stream, ag_internode_stream=ag_internode_stream,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, stages=stages)
            dist_x = x.split(bsz_per_rank, dim=0)[RANK].contiguous()

            out_triton = attn.dist_triton_fwd(dist_x, position_ids, cos_sin_cache, kv_cache, layer_idx=0,
                                              ag_gemm_persistent=AG_GEMM_PERSISTENT,
                                              gemm_rs_persistent=GEMM_RS_PERSISTENT, autotune=True)
            out = golden.split(bsz_per_rank, dim=0)[RANK].contiguous()
            assert_allclose(out_triton, out, atol=ATOL, rtol=RTOL)
        else:
            # dist triton prefill with AR
            attn._init_AR_ctx(max_M=M, method=to_allreduce_method(args.allreduce_method), dtype=DTYPE)
            out_triton = attn.dist_triton_AR_fwd(x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
            assert_allclose(out_triton, out_torch, atol=ATOL, rtol=RTOL)

        # Efficiency Test
        x = rand_tensor([BSZ, SEQ_LEN, K], dtype=DTYPE)
        position_ids = torch.arange(0, SEQ_LEN, dtype=torch.int64, device="cuda").unsqueeze(0).expand(BSZ, -1)
        kv_cache.kv_offset.fill_(0)
        mempool = torch.cuda.graph_pool_handle()
        torch_graph = make_cuda_graph(mempool,
                                      partial(attn.torch_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0))
        if not args.use_allreduce:
            dist_x = x.split(bsz_per_rank, dim=0)[RANK].contiguous()
            triton_dist_graph = make_cuda_graph(
                mempool,
                partial(attn.dist_triton_fwd, dist_x, position_ids, cos_sin_cache, kv_cache, layer_idx=0,
                        ag_gemm_persistent=AG_GEMM_PERSISTENT, gemm_rs_persistent=GEMM_RS_PERSISTENT, autotune=True))
        else:
            triton_dist_graph = make_cuda_graph(
                mempool, partial(attn.dist_triton_AR_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0))

        with group_profile("tp_attn_prefill", profile, group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()

        dist_print(f"torch attn prefill #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        if not args.use_allreduce:
            dist_print(f"dist-triton attn prefill #{RANK}", dist_triton_perf, f"{torch_perf/dist_triton_perf}x",
                       need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        else:
            dist_print(f"dist-triton_AR_{args.allreduce_method} attn prefill #{RANK}", dist_triton_perf,
                       f"{torch_perf/dist_triton_perf}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        del torch_graph, triton_dist_graph, mempool

    else:
        # decode
        # torch decode
        x = rand_tensor([BSZ, 1, K], dtype=DTYPE)
        position_ids = torch.arange(SEQ_LEN, SEQ_LEN + 1, dtype=torch.int64, device="cuda").unsqueeze(0).expand(BSZ, -1)
        kv_cache.kv_offset.fill_(SEQ_LEN)
        kv_cache.rand_fill_kv_cache(SEQ_LEN)

        out_torch = attn.torch_fwd(x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)

        # triton decode
        M = BSZ
        if not args.use_allreduce:
            out_torch = out_torch.split(bsz_per_rank, dim=0)[RANK].contiguous()
            attn._init_ctx(max_M=M, ag_intranode_stream=ag_intranode_stream, ag_internode_stream=ag_internode_stream,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, stages=stages)
            dist_x = x.split(bsz_per_rank, dim=0)[RANK].contiguous()
            out_triton = attn.dist_triton_fwd(dist_x, position_ids, cos_sin_cache, kv_cache, layer_idx=0,
                                              ag_gemm_persistent=AG_GEMM_PERSISTENT,
                                              gemm_rs_persistent=GEMM_RS_PERSISTENT, autotune=True)
            assert_allclose(out_triton, out_torch, atol=ATOL, rtol=RTOL)
        else:
            attn._init_AR_ctx(max_M=M, method=to_allreduce_method(args.allreduce_method), dtype=DTYPE)
            out_triton = attn.dist_triton_AR_fwd(x, position_ids, cos_sin_cache, kv_cache, layer_idx=0)
            assert_allclose(out_triton, out_torch, atol=ATOL, rtol=RTOL)

        # Efficiency Test

        # decode
        position_ids = torch.arange(SEQ_LEN, SEQ_LEN + 1, dtype=torch.int64, device="cuda").unsqueeze(0).expand(BSZ, -1)
        kv_cache.kv_offset.fill_(SEQ_LEN)
        mempool = torch.cuda.graph_pool_handle()
        torch_graph = make_cuda_graph(mempool,
                                      partial(attn.torch_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0))

        if not args.use_allreduce:
            dist_x = x.split(bsz_per_rank, dim=0)[RANK].contiguous()
            triton_dist_graph = make_cuda_graph(
                mempool,
                partial(attn.dist_triton_fwd, dist_x, position_ids, cos_sin_cache, kv_cache, layer_idx=0,
                        ag_gemm_persistent=AG_GEMM_PERSISTENT, gemm_rs_persistent=GEMM_RS_PERSISTENT, autotune=True))
        else:
            triton_dist_graph = make_cuda_graph(
                mempool, partial(attn.dist_triton_AR_fwd, x, position_ids, cos_sin_cache, kv_cache, layer_idx=0))

        with group_profile("tp_attn_decode", profile, group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
            torch.cuda.synchronize()

        dist_print(f"torch attn decode #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        if not args.use_allreduce:
            dist_print(f"dist-triton attn decode #{RANK}", dist_triton_perf, f"{torch_perf/dist_triton_perf}x",
                       need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        else:
            dist_print(f"dist-triton_AR_{args.allreduce_method} attn decode #{RANK}", dist_triton_perf,
                       f"{torch_perf/dist_triton_perf}x", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        del torch_graph, triton_dist_graph, mempool

    attn.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
