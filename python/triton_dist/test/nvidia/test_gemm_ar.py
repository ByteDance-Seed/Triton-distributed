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
import argparse
import random
import os
from functools import partial
from typing import Optional
import warnings
import sys

from triton_dist.utils import (is_nvshmem_multimem_supported, assert_allclose, dist_print, generate_data, group_profile,
                               initialize_distributed, nvshmem_barrier_all_on_stream, perf_func, finalize_distributed,
                               sleep_async)
from triton_dist.layers.nvidia import GemmARLayer


def torch_gemm_ar(
    input: torch.Tensor,  # [M, local_k]
    weight: torch.Tensor,  # [N, local_K]
    bias: Optional[torch.Tensor],
    tp_group,
):
    output = torch.matmul(input, weight.T)
    if bias:
        output = output + bias
    torch.distributed.all_reduce(output, group=tp_group)
    return output


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 6e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def straggler(rank):
    clock_rate = torch.cuda.clock_rate() * 1e6
    cycles = random.randint(0, clock_rate * 0.01) * (rank + 1)
    torch.cuda._sleep(cycles)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=30, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))

    parser.add_argument("--low-latency", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--copy-to-local", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_comm_sms", default=16, type=int, help="num sm for allreduce")

    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=True,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # init
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)

    args = parse_args()
    tp_group = initialize_distributed(args.seed)
    if torch.cuda.get_device_capability()[0] < 9:
        assert not args.persistent, "persistent is not supported on cuda < 9.0"

    if not is_nvshmem_multimem_supported():
        warnings.warn("Skip because nvshmem multimem is not supported")
        sys.exit(0)

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.K % WORLD_SIZE == 0
    local_K = args.K // WORLD_SIZE

    scale = RANK + 1

    def _make_data(M):
        data_config = [
            ((M, local_K), input_dtype, (0.01 * scale, 0)),  # A
            ((args.N, local_K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((M, args.N), input_dtype, (1, 0))),
        ]
        generator = generate_data(data_config)
        input, weight, bias = next(generator)
        return input, weight, bias

    gemm_ar_op = GemmARLayer(tp_group, args.M, args.N, args.K, input_dtype, output_dtype, LOCAL_WORLD_SIZE,
                             persistent=args.persistent, use_ll_kernel=args.low_latency,
                             copy_to_local=args.copy_to_local, NUM_COMM_SMS=args.num_comm_sms)

    if args.check:
        assert args.copy_to_local
        for n in range(args.iters):
            torch.cuda.empty_cache()
            input_list = [_make_data(args.M) for _ in range(args.verify_iters)]
            dist_out_list, torch_out_list = [], []

            # torch impl
            for input, weight, bias in input_list:
                torch_out = torch_gemm_ar(input, weight, bias, tp_group)
                torch_out_list.append(torch_out)

            # dist triton impl
            for input, weight, bias in input_list:
                straggler(RANK)
                dist_out = gemm_ar_op.forward(input, weight, bias)
                dist_out_list.append(dist_out)
            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
                assert_allclose(torch_out, dist_out, atol=atol, rtol=rtol, verbose=False)
        print(f"RANK[{RANK}]: pass.")
        gemm_ar_op.finalize()
        finalize_distributed()
        exit(0)

    # warm up
    input, weight, bias = _make_data(args.M)
    _ = gemm_ar_op.forward(input, weight, bias)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    with group_profile(f"gemm_ar_{args.M}x{args.N}x{args.K}_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile,
                       group=tp_group):
        torch_output, torch_perf = perf_func(partial(torch_gemm_ar, input, weight, bias, tp_group), iters=args.iters,
                                             warmup_iters=args.warmup)

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        sleep_async(1000)

        dist_triton_output, dist_triton_perf = perf_func(partial(gemm_ar_op.forward, input, weight, bias),
                                                         iters=args.iters, warmup_iters=args.warmup)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
    assert_allclose(torch_output, dist_triton_output, atol=atol, rtol=rtol)
    torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", dist_triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    gemm_ar_op.finalize()
    finalize_distributed()
