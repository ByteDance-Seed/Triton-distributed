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
import itertools
import os
import random
from typing import Optional
import torch
import triton
import sys
import torch.distributed as dist
from triton_dist.kernels.allreduce import AllReduceMethod, get_allreduce_methods, to_allreduce_method
from triton_dist.kernels.nvidia.allreduce import (create_allreduce_ctx, all_reduce)
from triton_dist.utils import (assert_allclose, group_profile, initialize_distributed, finalize_distributed, perf_func,
                               sleep_async)

DATA_SIZES = [
    128,  # 128B
    1024,  # 1KB
    16 * 1024,  # 16K
    32 * 1024,  # 32K
    64 * 1024,  # 64K
    128 * 1024,  # 128K
    256 * 1024,  # 256K
    512 * 1024,  # 512K
    1024 * 1024,  # 1M
    2 * 1024 * 1024,  # 2M
    4 * 1024 * 1024,  # 4M
    8 * 1024 * 1024,  # 8M
    16 * 1024 * 1024,  # 16M
    32 * 1024 * 1024,  # 32M
    64 * 1024 * 1024,  # 64M
    128 * 1024 * 1024  # 128M
]


def _pretty_format(nbytes):
    if nbytes < 1024:
        return f"{nbytes}B"
    if nbytes < 1024 * 1024:
        return f"{nbytes / 1024}KB"
    if nbytes < 1024 * 1024 * 1024:
        return f"{nbytes / 1024 / 1024}MB"
    return f"{nbytes / 1024 / 1024 / 1024}GB"


def _create_data(numel, dtype=torch.float32):
    input = torch.rand((numel, ), dtype=dtype, device="cuda")
    if args.debug:
        input = input.fill_((RANK + 1) / 10)
    return input


def torch_all_reduce(local_input: torch.Tensor, pg: torch.distributed.ProcessGroup):
    output = torch.clone(local_input)
    dist.all_reduce(output, group=pg)
    return output


def _randint_with_align(max_M, alignment: int):
    return random.randint(1, max_M // alignment) * alignment


def _random_straggler_option():
    rank = random.randint(0, WORLD_SIZE - 1)
    # this may be not accurate. but never mind
    clock_rate = torch.cuda.clock_rate() * 1e6
    # 0ms ~ 10ms. 10ms is enough for all-reduce kernel: on H800 with >1GB data sent/received
    cycles = random.randint(0, int(clock_rate * 0.01))
    return (rank, cycles)


def stress_test(dtype: torch.dtype, args, method: AllReduceMethod):
    random.seed(args.seed)  # set all ranks to the same seed

    atol, rtol = {
        torch.bfloat16: (3e-2, 3e-2),
        torch.float16: (1e-2, 1e-2),
        torch.float32: (1e-3, 1e-3),
    }[dtype]

    ctx = create_allreduce_ctx(args.max_nbytes, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE)

    for n in range(args.iters):
        # generate data for verify
        tensor_inputs = [
            _create_data(_randint_with_align(args.max_nbytes // dtype.itemsize, WORLD_SIZE * args.alignment),
                         dtype=dtype) for _ in range(args.verify_shapes)
        ]
        triton_out_list = [all_reduce(x, None, method=method, ctx=ctx) for x in tensor_inputs]
        torch_out_list = [torch_all_reduce(x, pg=TP_GROUP) for x in tensor_inputs]

        # verify
        try:
            for i, (triton_res, torch_res) in enumerate(zip(triton_out_list, torch_out_list)):
                assert_allclose(triton_res, torch_res, atol=atol, rtol=rtol, verbose=False)
        except Exception as e:
            print(f"RANK = {RANK}, {i}-th iteration failed with {e}", file=sys.stderr)
            raise e

        sleep_async(1000)
        for x in itertools.islice(itertools.cycle(tensor_inputs), args.verify_hang):
            straggler_opt = _random_straggler_option() if args.simulate_straggler else None
            all_reduce(x, None, method=method, ctx=ctx, straggler_option=straggler_opt)

        print(f"runs {n + 1} iterations done")
        if (n + 1) % 10 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    if TP_GROUP.rank() == 0:
        print(f"✅ AllReduce {method.name} Pass!")

    ctx.finalize()


def _is_one_shot(method):
    return "one_shot" in method


def run_perf(dtype: torch.dtype, method: AllReduceMethod, warmup=5, iters=10):
    bytes_per_elem = torch.finfo(dtype).bits // 8
    if method in ["double_tree", "one_shot", "one_shot_tma"]:
        available_ds = DATA_SIZES[:13]
    else:
        available_ds = DATA_SIZES

    ctx = create_allreduce_ctx(available_ds[-1], RANK, WORLD_SIZE, LOCAL_WORLD_SIZE)

    for nbytes in available_ds:
        num_elem = nbytes // bytes_per_elem

        local_input = _create_data(num_elem, dtype=dtype)
        output = torch.empty_like(local_input)

        def allreduce_op():
            all_reduce(local_input, output, method=method, ctx=ctx)

        sleep_async(100)  # in case CPU bound
        _, duration_ms = perf_func(allreduce_op, warmup_iters=warmup, iters=iters)

        algo_bw = nbytes * 1e-9 / (duration_ms * 1e-3) * 2
        hw_bw = algo_bw * (WORLD_SIZE - 1) / WORLD_SIZE
        if _is_one_shot(method):
            hw_bw = algo_bw * WORLD_SIZE // 2

        if RANK == 0:
            print(
                f"RANK = {RANK}, " + _pretty_format(nbytes) +
                f" Latency = {duration_ms * 1000:0.2f} us, HW Bandwith = {hw_bw:0.2f} GB/s, Algo Bandwith = {algo_bw:0.2f} GB/s   "
            )

    ctx.finalize()


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_nbytes", type=int, default=1024 * 4096)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--verify_shapes", type=int, default=25)
    parser.add_argument("--verify_hang", type=int, default=100)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--alignment", type=int, default=16)
    parser.add_argument("--method", type=str, default="double_tree", choices=get_allreduce_methods())
    parser.add_argument("--simulate_straggler", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32", "fp16"])
    parser.add_argument("--stress", default=False, action="store_true")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true")
    args = parser.parse_args()

    DTYPE = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]

    # for TMA reduce
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=DTYPE)

    triton.set_allocator(alloc_fn)

    if "multicast" in args.method or "ld_reduce" in args.method:
        assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0"  # for multicast

    method = to_allreduce_method(args.method)

    if args.stress:
        stress_test(DTYPE, args, method=method)
    else:
        with group_profile(f"all_reduce_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=TP_GROUP):
            run_perf(DTYPE, method, warmup=args.warmup_iters, iters=args.iters)

    finalize_distributed()
