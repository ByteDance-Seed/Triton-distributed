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
from transformers import AutoConfig

import pyrocshmem
from triton_dist.models.kv_cache import KV_Cache
from triton_dist.models.config import ModelConfig
from triton_dist.models import AutoLLM
from triton_dist.models.utils import seed_everything, init_model_cpu
from triton_dist.utils import perf_func, dist_print, group_profile

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 3e-2,
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
        dist_print(f"❗ [RANK {RANK}] Max difference: {max_diff.item()} (atol={atol}, rtol={rtol})",
                   allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"Output: {out}\nGolden: {golden}", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        assert False, f"❌ [RANK {RANK}] Output mismatch."


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
                        help="mode of operation, prefill or decode")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="check correctness")
    parser.add_argument("--seed", type=int, default=42)

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
    MODE = args.mode
    BSZ = args.bsz
    SEQ_LEN = args.seq_len

    model_config = ModelConfig(model_name=args.model, max_length=args.seq_len + 128, dtype=DTYPE, rank=RANK,
                               world_size=WORLD_SIZE, local_only=True)
    model = AutoLLM.from_pretrained(model_config, group=TP_GROUP)
    model.init_triton_dist_ctx(max_M=BSZ * SEQ_LEN if MODE == 'prefill' else BSZ)
    print(f"[RANK {RANK}] Model {args.model} initialized with dtype {DTYPE}.")
    seed_everything(args.seed)

    BSZ = args.bsz
    SEQ_LEN = args.seq_len
    input_ids = torch.randint(10, 1000, (BSZ, SEQ_LEN), dtype=torch.long, device="cuda")
    position_ids = torch.arange(0, SEQ_LEN, dtype=torch.long, device="cuda").unsqueeze(0).repeat(BSZ, 1)
    kv_cache = KV_Cache(num_layers=model.num_layers, kv_heads=model.num_key_value_heads, head_dim=model.head_dim,
                        batch_size=BSZ, dtype=DTYPE, max_length=model.max_length, world_size=WORLD_SIZE)

    if args.check:
        # Precision Test
        # golden
        config = AutoConfig.from_pretrained(args.model)
        hf_model = init_model_cpu(args.model, DTYPE).cuda()
        with torch.inference_mode():
            golden = hf_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
            ).logits.float()
        golden = golden[:, -1:, :].contiguous()

        del hf_model
        gc.collect()
        torch.cuda.empty_cache()

        # torch prefill
        logits = model.inference(input_ids=input_ids, position_ids=position_ids, kv_cache=kv_cache)
        check_allclose(logits.softmax(dim=-1), golden.softmax(dim=-1), atol=ATOL, rtol=RTOL)

        # triton dist prefill
        model.init_triton_dist_ctx(max_M=BSZ * SEQ_LEN)
        model.set_fwd(mode='triton_dist')
        input_ids = input_ids.split(BSZ // WORLD_SIZE, dim=0)[RANK]
        logits = model.inference(input_ids=input_ids, position_ids=position_ids, kv_cache=kv_cache)
        golden = golden.split(BSZ // WORLD_SIZE, dim=0)[RANK]
        check_allclose(logits.softmax(dim=-1), golden.softmax(dim=-1), atol=ATOL, rtol=RTOL)

        torch.distributed.destroy_process_group()
        exit(0)

    profile = args.profile
    # Efficiency Test
    if MODE == 'prefill':
        # prefill
        input_ids = torch.randint(0, 1000, (BSZ, SEQ_LEN), dtype=torch.long, device="cuda")
        position_ids = torch.arange(0, SEQ_LEN, dtype=torch.int64, device="cuda").unsqueeze(0).expand(BSZ, -1)
        kv_cache.kv_offset.fill_(0)
        mempool = torch.cuda.graph_pool_handle()
        model.set_fwd(mode='torch')
        torch_graph = make_cuda_graph(mempool, partial(model.inference, input_ids, position_ids, kv_cache, True))

        dist_x = input_ids.split(BSZ // WORLD_SIZE, dim=0)[RANK].contiguous()
        model.set_fwd(mode='triton_dist')
        triton_dist_graph = make_cuda_graph(mempool, partial(model.inference, dist_x, position_ids, kv_cache, True))

        with group_profile("tp_e2e_prefill", profile, group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

        dist_print(f"torch prefill #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"dist-triton prefill #{RANK}", dist_triton_perf, need_sync=True,
                   allowed_ranks=list(range(WORLD_SIZE)))
        del torch_graph, triton_dist_graph, mempool
    else:
        # decode
        input_ids = torch.randint(0, 1000, (BSZ, 1), dtype=torch.long, device="cuda")
        position_ids = torch.arange(SEQ_LEN, SEQ_LEN + 1, dtype=torch.int64, device="cuda").unsqueeze(0).expand(BSZ, -1)
        kv_cache.kv_offset.fill_(SEQ_LEN)
        mempool = torch.cuda.graph_pool_handle()
        model.set_fwd(mode='torch')
        torch_graph = make_cuda_graph(mempool, partial(model.inference, input_ids, position_ids, kv_cache, True))

        dist_x = input_ids.split(BSZ // WORLD_SIZE, dim=0)[RANK].contiguous()
        model.set_fwd(mode='triton_dist')
        triton_dist_graph = make_cuda_graph(mempool, partial(model.inference, dist_x, position_ids, kv_cache, True))

        with group_profile("tp_e2e_decode", profile, group=TP_GROUP):
            torch.cuda.synchronize()
            _, torch_perf = perf_func(torch_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

            torch.cuda.synchronize()
            _, dist_triton_perf = perf_func(triton_dist_graph.replay, iters=args.iters, warmup_iters=args.warmup)
            torch.cuda.synchronize()

        dist_print(f"torch decode #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
        dist_print(f"dist-triton decode #{RANK}", dist_triton_perf, need_sync=True,
                   allowed_ranks=list(range(WORLD_SIZE)))
        del torch_graph, triton_dist_graph, mempool

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()
