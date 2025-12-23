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
import random
from functools import partial
from typing import Optional

import torch

from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import (dist_print, initialize_distributed, finalize_distributed,
                               wait_until_max_gpu_clock_or_warning, rand_tensor)
from triton_dist.kernels.nvidia.gemm import get_config_space


def torch_gemm_rs(
    A: torch.Tensor,  # [M, K_per_rank]
    B: torch.Tensor,  # [K_per_rank, N]
    bias: Optional[torch.Tensor],
    tp_group: torch.distributed.ProcessGroup,
):
    M, _ = A.shape
    _, N = B.shape
    output = torch.matmul(A, B)
    if bias:
        output = output + bias
    rs_output = torch.empty((M // WORLD_SIZE, N), dtype=output.dtype, device=A.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=tp_group)
    return rs_output


def torch_gemm_only(
    A: torch.Tensor,  # [M, K_per_rank]
    B: torch.Tensor,  # [K_per_rank, N]
    bias: Optional[torch.Tensor],
):
    output = torch.matmul(A, B)
    if bias:
        output = output + bias
    return output


def torch_reduce_scatter_only(
    A: torch.Tensor,  # [M, K_per_rank]
    B: torch.Tensor,  # [K_per_rank, N]
    tp_group: torch.distributed.ProcessGroup,
):
    M, _ = A.shape
    _, N = B.shape
    output = torch.empty((M, N), dtype=A.dtype, device=A.device)
    rs_output = torch.empty((M // WORLD_SIZE, N), dtype=A.dtype, device=A.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=tp_group)
    return rs_output


class GemmRS(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_M: int,
        N: int,
        K: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        local_world_size: int = -1,
        persistent: bool = True,
        fuse_scatter: bool = False,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.local_world_size = local_world_size if local_world_size != -1 else self.world_size
        self.local_rank = self.rank % self.local_world_size

        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.rs_stream: torch.cuda.Stream = torch.cuda.Stream(priority=-1)

        self.ctx = create_gemm_rs_context(max_M, N, self.rank, self.world_size, self.local_world_size, output_dtype,
                                          self.rs_stream)
        self.fuse_scatter = fuse_scatter
        self.persistent = persistent

    def forward(
        self,
        input: torch.Tensor,  # [M, K_per_rank]
        weight: torch.Tensor,  # [K_per_rank, N]
        bias: Optional[torch.Tensor] = None,
    ):
        if args.autotune:
            return gemm_rs(input, weight, self.ctx, autotune=args.autotune)
        else:
            return gemm_rs.fn(input, weight, self.ctx, gemm_config=get_config_space(self.persistent)[0],
                              persistent=self.persistent, fuse_scatter=self.fuse_scatter)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", "--M", type=int, required=False)
    parser.add_argument("--M_range", "--M-range", type=str, help="M range in [start, end, step], e.g. 256-8192-8")
    parser.add_argument("-N", "--N", type=int)
    parser.add_argument("-K", "--K", type=int)
    parser.add_argument("--warmup", default=20, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--trans_b", default=True, type=bool, action=argparse.BooleanOptionalAction)
    parser.add_argument("--autotune", default=False, type=bool, action=argparse.BooleanOptionalAction)

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--persistent", action=argparse.BooleanOptionalAction,
                        default=torch.cuda.get_device_capability() >= (9, 0))

    parser.add_argument("--fuse_scatter", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--has_bias", default=False, action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # init
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    args = parse_args()
    tp_group = initialize_distributed(args.seed)
    if torch.cuda.get_device_capability()[0] < 9:
        assert not args.persistent, "persistent is not supported on cuda < 9.0"

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    if args.M is None:
        assert args.M_range is not None, "M_range is required when M is None"
        M_range = args.M_range.split("-")
        assert len(M_range) == 3, "M_range must be in format of [start, end, step]"
        M_start, M_end, M_step = [int(x) for x in M_range]
        assert M_start % WORLD_SIZE == 0, "M_start must be divisible by WORLD_SIZE"
        assert M_end % WORLD_SIZE == 0, "M_end must be divisible by WORLD_SIZE"
        assert M_start <= M_end, "M_start must be less than M_end"
        assert M_step % WORLD_SIZE == 0, "M_step must be divisible by WORLD_SIZE"
        M_max = M_end
        M_list = list(range(M_start, M_end + 1, M_step))
    else:
        M_max = args.M
        M_list = [args.M]

    assert args.K % WORLD_SIZE == 0
    K_per_rank = args.K // WORLD_SIZE

    def _make_data(M):
        scale = RANK + 1
        A = rand_tensor((M, K_per_rank), dtype=input_dtype, device="cuda")
        A *= 0.01 * scale
        if args.trans_b:
            B = rand_tensor((args.N, K_per_rank), dtype=input_dtype, device="cuda").T
        else:
            B = rand_tensor((K_per_rank, args.N), dtype=input_dtype, device="cuda")
        if args.has_bias:
            bias = rand_tensor((M, args.N), dtype=input_dtype, device="cuda")
        else:
            bias = None
        return A, B, bias

    gemm_rs_op = GemmRS(tp_group, M_max, args.N, args.K, input_dtype, output_dtype, LOCAL_WORLD_SIZE, args.persistent,
                        args.fuse_scatter)

    if args.check:
        for n in range(args.iters):
            torch.cuda.empty_cache()
            input_list = [
                _make_data(random.randint(1, args.M // WORLD_SIZE) * WORLD_SIZE) for _ in range(args.verify_iters)
            ]
            dist_out_list, torch_out_list = [], []

            # torch impl
            for A, weight, bias in input_list:
                torch_out = torch_gemm_rs(A, weight, bias, tp_group)
                torch_out_list.append(torch_out)

            # dist triton impl
            for A, weight, bias in input_list:
                dist_out = gemm_rs_op.forward(A, weight, bias)
                dist_out_list.append(dist_out)
            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
                assert_allclose(torch_out, dist_out, atol=atol, rtol=rtol, verbose=False)
        print(f"✅ RANK[{RANK}]: pass.")

        gemm_rs_op.ctx.finalize()
        finalize_distributed()
        exit(0)

    for M in M_list:
        A, weight, bias = _make_data(M)
        exp_name = f"{M}x{args.N}x{args.K}_{os.environ['TORCHELASTIC_RUN_ID']}"
        with group_profile(f"gemm_rs_{exp_name}", args.profile, group=tp_group):
            wait_until_max_gpu_clock_or_warning()
            torch_output, torch_duration_ms = perf_func(partial(torch_gemm_rs, A, weight, bias, tp_group),
                                                        iters=args.iters, warmup_iters=args.warmup)

            wait_until_max_gpu_clock_or_warning()
            triton_output, triton_duration_ms = perf_func(partial(gemm_rs_op.forward, A, weight, bias),
                                                          iters=args.iters, warmup_iters=args.warmup)
            wait_until_max_gpu_clock_or_warning()
            _, torch_gemm_ms = perf_func(lambda: torch_gemm_only(A, weight, bias), iters=args.iters,
                                         warmup_iters=args.warmup)
            wait_until_max_gpu_clock_or_warning()
            _, torch_rs_ms = perf_func(lambda: torch_reduce_scatter_only(A, weight, tp_group), iters=args.iters,
                                       warmup_iters=args.warmup)

        assert_allclose(torch_output, triton_output, atol=atol, rtol=rtol)

        dist_print(
            f"M_{M}_N_{args.N}_K_{args.K}_TP_{WORLD_SIZE} #{RANK} triton total: {triton_duration_ms:0.3f} ms/iter torch total: {torch_duration_ms:0.3f} ms/iter, GEMM {torch_gemm_ms:0.3f} ms/iter, ReduceScatter {torch_rs_ms:0.3f} ms/iter",
            need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

        # performance metrics
        tflops = 2 * M * args.N * K_per_rank / 1e12
        gemm_mem_read_gb = (M * K_per_rank * input_dtype.itemsize + K_per_rank * args.N * input_dtype.itemsize) / 2**30
        gemm_mem_write_gb = (M * args.N * output_dtype.itemsize) / 2**30
        triton_tflops = tflops / triton_duration_ms * 1e3
        triton_gemm_mem_read_gbps = gemm_mem_read_gb / triton_duration_ms * 1e3
        triton_gemm_mem_write_gbps = gemm_mem_write_gb / triton_duration_ms * 1e3

        torch_tflops = tflops / torch_gemm_ms * 1e3
        torch_gemm_mem_read_gbps = gemm_mem_read_gb / torch_gemm_ms * 1e3
        torch_gemm_mem_write_gbps = gemm_mem_write_gb / torch_gemm_ms * 1e3

        reduce_scatter_gb = M * args.N * output_dtype.itemsize / 2**30 * (WORLD_SIZE - 1) / WORLD_SIZE
        torch_reduce_scatter_gbps = reduce_scatter_gb / torch_rs_ms * 1e3
        print(
            f"triton #{RANK} GEMM full {triton_tflops:0.2f} TFLOPS, read {triton_gemm_mem_read_gbps:0.2f} GB/s, write {triton_gemm_mem_write_gbps:0.2f} GB/s"
        )
        print(
            f"torch  #{RANK} GEMM only {torch_tflops:0.2f} TFLOPS, read {torch_gemm_mem_read_gbps:0.2f} GB/s, write {torch_gemm_mem_write_gbps:0.2f} GB/s, ReduceScatter only {torch_reduce_scatter_gbps:0.2f} GB/s"
        )

    gemm_rs_op.ctx.finalize()
    finalize_distributed()
