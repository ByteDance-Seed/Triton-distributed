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
import torch.distributed
from triton_dist.utils import finalize_distributed, initialize_distributed, perf_func, get_torch_prof_ctx
from functools import partial

import argparse
import random
import os

from triton_dist.layers.nvidia import EPLowLatencyAllToAllLayer
from triton_dist.test.nvidia.ep_a2a_utils import (
    torch_ll_dispatch,
    torch_ll_combine,
    dequant_fp8_bf16,
)

EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def generate_random_exp_indices(token_num, total_num_experts, topk, drop_ratio=0.0):
    exp_indices = []
    exp_list = list(range(total_num_experts))

    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        for i, _ in enumerate(top_selected):
            if random.uniform(0, 1) < drop_ratio:
                # current topk choice will be dropped
                top_selected[i] = total_num_experts
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-N", type=int, default=7168)
    parser.add_argument("-G", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bench_iters", default=1000, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--weight_dtype", default="float32", help="weight type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--quant_group_size", type=int, default=128, help="quantization group size")
    parser.add_argument("--no_online_quant_fp8", action="store_false", dest="online_quant_fp8", default=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--intra_kernel_profile", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--iters", default=3, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=5, type=int)

    return parser.parse_args()


def straggler(rank):
    clock_rate = torch.cuda.clock_rate() * 1e6
    r = max(int(clock_rate * 0.00001), 1000)
    cycles = random.randint(0, r) * (rank + 1)
    torch.cuda._sleep(cycles)


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    assert (args.G % WORLD_SIZE == 0), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"

    experts_per_rank = args.G // WORLD_SIZE
    input_dtype = DTYPE_MAP[args.dtype]
    weight_dtype = DTYPE_MAP[args.weight_dtype]

    assert input_dtype == torch.bfloat16

    ep_ll_a2a_layer = EPLowLatencyAllToAllLayer(
        args.M,
        args.N,
        args.topk,
        online_quant_fp8=args.online_quant_fp8,
        rank=RANK,
        num_experts=args.G,
        local_world_size=LOCAL_WORLD_SIZE,
        world_size=WORLD_SIZE,
        dtype=input_dtype,
        enable_profiling=args.intra_kernel_profile,
    )

    def _make_data(token_num):
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = (torch.rand(token_num, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
        weight = torch.randn(token_num, args.topk, dtype=torch.float32).to("cuda")
        weight = torch.nn.functional.softmax(weight, dim=1).to(weight_dtype)
        return input, weight, exp_indices

    if args.check:
        for n in range(args.iters):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            input_list = [_make_data(random.randint(1, args.M)) for _ in range(args.verify_iters)]
            triton_combine_out_list, dispatch_out_list, torch_input_list, torch_dispatch_out_list, torch_combine_out_list = [], [], [], [], []

            # torch impl
            for input, weight, exp_indices in input_list:
                ref_dispatch_out, ref_dispatch_scale = torch_ll_dispatch(EP_GROUP, input, exp_indices, args.G,
                                                                         args.quant_group_size, args.online_quant_fp8)
                if not args.online_quant_fp8:
                    assert ref_dispatch_scale is None

                ref_combine_input = dequant_fp8_bf16(ref_dispatch_out, ref_dispatch_scale)

                ref_combine_out = torch_ll_combine(EP_GROUP, ref_combine_input, exp_indices, weight, args.G)
                torch_combine_out_list.append(ref_combine_out)

            # dist triton impl
            for input, weight, exp_indices in input_list:
                straggler(RANK)
                scales = None
                triton_dispatch_out, triton_dispatch_scale, expert_recv_count, dispatch_meta = ep_ll_a2a_layer.dispatch(
                    input, scales, exp_indices)

                triton_combine_input = dequant_fp8_bf16(triton_dispatch_out, triton_dispatch_scale)

                triton_combine_out = ep_ll_a2a_layer.combine(triton_combine_input, exp_indices, weight, dispatch_meta)

                triton_combine_out_list.append(triton_combine_out)

            for idx, (torch_combine_out,
                      triton_combine_out) in enumerate(zip(torch_combine_out_list, triton_combine_out_list)):
                if RANK == 0:
                    print(
                        f"combine: shape = {torch_combine_out.shape} {torch_combine_out.dtype}, {triton_combine_out.shape} {triton_combine_out.dtype}"
                    )
                try:
                    torch.testing.assert_close(torch_combine_out, triton_combine_out, atol=0, rtol=0)
                except Exception as e:
                    raise e

        print(f"RANK[{RANK}]: pass.")
        ep_ll_a2a_layer.finalize()
        finalize_distributed()
        exit(0)

    for rid in range(args.rounds):
        # random simulate token received from dataloader
        L = args.M // 2 if not args.profile else args.M

        token_num = random.randint(L, args.M)

        print(f"Rank-{RANK}: Received {token_num} tokens")

        input, weight, exp_indices = _make_data(token_num)
        scales = None
        ctx = get_torch_prof_ctx(args.profile)
        with ctx:
            (ref_dispatch_out, ref_dispatch_scale), _ = perf_func(
                partial(torch_ll_dispatch, EP_GROUP, input, exp_indices, args.G, args.quant_group_size,
                        args.online_quant_fp8), iters=100, warmup_iters=20)
            if not args.online_quant_fp8:
                assert ref_dispatch_scale is None

            ref_combine_input = dequant_fp8_bf16(ref_dispatch_out, ref_dispatch_scale)

            ref_combine_out, _ = perf_func(
                partial(torch_ll_combine, EP_GROUP, ref_combine_input, exp_indices, weight, args.G), iters=100,
                warmup_iters=20)

            # warm up to avoid cudaMalloc caused by torch.empty
            _ = ep_ll_a2a_layer.dispatch(input, scales, exp_indices)

            # avoid bound in host
            torch.cuda._sleep(int(1e9))

            (triton_dispatch_out, triton_dispatch_scale, expert_recv_count,
             dispatch_meta), triton_perf = perf_func(partial(ep_ll_a2a_layer.dispatch, input, scales, exp_indices),
                                                     iters=100, warmup_iters=20)

            triton_combine_input = dequant_fp8_bf16(triton_dispatch_out, triton_dispatch_scale)

            triton_combine_out, triton_combine_perf = perf_func(
                partial(ep_ll_a2a_layer.combine, triton_combine_input, exp_indices, weight, dispatch_meta), iters=100,
                warmup_iters=20)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        torch.distributed.barrier()  # wait all rank dispatch

        if args.profile:
            run_id = os.environ["TORCHELASTIC_RUN_ID"]
            prof_dir = f"prof/{run_id}"
            os.makedirs(prof_dir, exist_ok=True)
            ctx.export_chrome_trace(f"{prof_dir}/trace_rank{EP_GROUP.rank()}.json.gz")

        torch.testing.assert_close(ref_combine_out, triton_combine_out, rtol=0, atol=0)

        print(f"RANK {RANK}: triton dispatch perf = {triton_perf}ms, triton combine perf = {triton_combine_perf}ms")

    ep_ll_a2a_layer.dump_dispatch_trace()
    ep_ll_a2a_layer.dump_combine_trace()

    ep_ll_a2a_layer.finalize()
    finalize_distributed()
