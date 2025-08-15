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
from functools import partial

import torch

from triton_dist.kernels.nvidia import create_all_to_all_single_2d_context, all_to_all_single_2d
from triton_dist.utils import (assert_allclose, dist_print, group_profile, initialize_distributed, perf_func,
                               finalize_distributed)
import numpy as np


class AlltoAll2D(torch.nn.Module):

    def __init__(self, max_m, hidden_dim, rank, world_size, dtype):
        super().__init__()
        self.max_m = max_m
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.ctx = create_all_to_all_single_2d_context(max_m, hidden_dim, rank, world_size, dtype)

    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        input_splits: torch.Tensor,
        output_splits: torch.Tensor,
        num_sm: int,
    ):
        all_to_all_single_2d(self.ctx, input, output, input_splits, output_splits, num_sm)
        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)

    parser.add_argument("--num_comm_sm", type=int, default=8, help="num comm sm")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    input_splits_cpu: torch.Tensor,
    output_splits_cpu: torch.Tensor,
):
    output = torch.empty((sum(output_splits_cpu), input.shape[1]), dtype=input.dtype, device=input.device)

    torch.distributed.all_to_all_single(output, input, output_splits_cpu, input_splits_cpu, group=EP_GROUP)

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

if __name__ == "__main__":
    args = parse_args()
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)
    EP_GROUP = initialize_distributed(args.seed)

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    max_split = args.M // WORLD_SIZE
    input_splits = torch.tensor(
        list(np.random.randint(max(1, max_split - 32), max_split, size=(WORLD_SIZE, ))),
        dtype=torch.int32,
    ).cuda()
    output_splits = torch.zeros_like(input_splits)
    torch.distributed.all_to_all_single(output_splits, input_splits, group=EP_GROUP)
    input_splits_cpu = input_splits.cpu().tolist()
    output_splits_cpu = output_splits.cpu().tolist()
    input_shape = (input_splits.sum().cpu().item(), args.N)
    input = (-2 * torch.rand(input_shape, dtype=input_dtype).cuda() + 1) * (RANK + 1)
    torch.distributed.barrier()
    a2a_single_2d_op = AlltoAll2D(args.M, args.N, RANK, WORLD_SIZE, input_dtype)
    output_buf = torch.empty((max_split * EP_GROUP.size(), input.size(1)), dtype=input.dtype, device=input.device)
    with group_profile(f"all2all_singel2d__{os.environ['TORCHELASTIC_RUN_ID']}", args.profile, group=EP_GROUP):
        torch_output, torch_perf = perf_func(partial(perf_torch, input, input_splits_cpu, output_splits_cpu),
                                             iters=args.iters, warmup_iters=args.warmup)

        dist_triton_output, dist_triton_perf = perf_func(
            partial(a2a_single_2d_op.forward, input, output_buf, input_splits, output_splits, args.num_comm_sm),
            iters=args.iters, warmup_iters=args.warmup)
        dist_triton_output = dist_triton_output[:torch_output.shape[0]]
        assert_allclose(torch_output, dist_triton_output, atol=atol, rtol=rtol)
    torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", dist_triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    a2a_single_2d_op.ctx.finalize()
    finalize_distributed()
