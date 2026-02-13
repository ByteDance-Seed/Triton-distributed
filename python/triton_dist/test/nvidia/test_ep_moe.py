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
from functools import partial
from transformers import AutoConfig

from triton_dist.utils import sleep_async
from triton_dist.layers.nvidia.ep_moe import EP_MoE
from triton_dist.models.utils import init_model_cpu
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import finalize_distributed, initialize_distributed, dist_print, nvshmem_barrier_all_on_stream, rand_tensor

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 2e-2,
    torch.float8_e4m3fn: 2e-2,
    torch.float8_e5m2: 2e-2,
    torch.int8: 0,
    torch.int32: 0,
}


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
    EP_GROUP = initialize_distributed(args.seed)

    DTYPE = DTYPE_MAP[args.dtype]
    ATOL = THRESHOLD_MAP[DTYPE]
    RTOL = THRESHOLD_MAP[DTYPE]

    config = AutoConfig.from_pretrained(args.model)
    hf_model = init_model_cpu(model_name=args.model, dtype=DTYPE)
    hf_mlp = hf_model.model.layers[0].mlp.eval()
    mlp = EP_MoE(rank=RANK, world_size=WORLD_SIZE, group=EP_GROUP)
    mlp._init_parameters(hf_mlp, verbose=True)

    torch.manual_seed(args.seed + RANK)
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

    # dist triton fwd
    mlp._init_ctx(EP_GROUP=EP_GROUP, max_tokens_per_rank=BSZ * SEQ_LEN)
    out_triton = mlp.dist_triton_fwd(x)

    assert_allclose(golden, torch_out, atol=ATOL, rtol=RTOL)
    assert_allclose(out_triton, torch_out, atol=ATOL, rtol=RTOL)

    # Efficiency Test
    profile = args.profile
    with group_profile("ep_moe", profile, group=EP_GROUP):
        torch.cuda.synchronize()
        sleep_async(100)
        _, torch_perf = perf_func(partial(mlp.torch_fwd, x), iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()
        sleep_async(100)
        _, dist_triton_perf = perf_func(partial(mlp.dist_triton_fwd, x), iters=args.iters, warmup_iters=args.warmup)
        nvshmem_barrier_all_on_stream()
        torch.cuda.synchronize()

    dist_print(f"torch ep moe e2e #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"dist-triton ep moe e2e #{RANK}", dist_triton_perf, f"{torch_perf/dist_triton_perf}x over torch",
               need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    mlp.finalize()
    finalize_distributed()
