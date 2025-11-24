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
import os
from typing import Optional

from functools import partial

from triton_dist.profiler_utils import get_torch_prof_ctx, perf_func
from triton_dist.utils import (dist_print, initialize_distributed, finalize_distributed, rand_tensor)

from triton_dist.kernels.amd import gemm_rs_intra_node, create_gemm_rs_intra_node_context


def torch_gemm_rs(
    A: torch.Tensor,  # [M, local_k]
    B: torch.Tensor,  # [N, local_K]
    bias: Optional[torch.Tensor],
    pg: torch.distributed.ProcessGroup,
):
    """ return C = reduce_scatter(A @ B.T + bias) """
    M, local_K = A.shape
    N, K = B.shape
    output = torch.matmul(A, B.T)
    if bias is not None:
        output = output + bias
    rs_output = torch.empty((M // pg.size(), N), dtype=output.dtype, device=A.device)
    torch.distributed.reduce_scatter_tensor(rs_output, output, group=pg)
    return rs_output


class GemmRSIntraNode(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_M: int,
        N: int,
        K: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        fuse_scatter: bool = True,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.fuse_scatter = fuse_scatter
        self.ctx = create_gemm_rs_intra_node_context(self.max_M, self.N, self.output_dtype, self.rank, self.world_size,
                                                     self.tp_group, fuse_scatter)

    def forward(self, A: torch.Tensor,  # [M, local_K]
                weight: torch.Tensor,  # [N, local_K]
                ):
        return gemm_rs_intra_node(A, weight, ctx=self.ctx)


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
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=10, type=int)

    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight",
        default=False,
    )
    parser.add_argument("--fuse_scatter", default=True, action="store_true", help="fuse scatter into gemm")
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()
    if args.has_bias:
        raise NotImplementedError("bias is not supported yet")

    if not args.fuse_scatter:
        raise NotImplementedError()

    os.environ["TRITON_HIP_USE_BLOCK_PINGPONG"] = "1"
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed(args.seed)

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.M % TP_GROUP.size() == 0
    assert args.K % TP_GROUP.size() == 0
    local_K = args.K // TP_GROUP.size()

    scale = (TP_GROUP.rank() + 1) * 0.01

    def _make_data(M):
        current_device = torch.cuda.current_device()
        A = rand_tensor((M, local_K), dtype=input_dtype, device=current_device) * scale
        if args.transpose_weight:
            B = rand_tensor((local_K, args.N), dtype=input_dtype, device=current_device).T * scale
        else:
            B = rand_tensor((args.N, local_K), dtype=input_dtype, device=current_device) * scale

        bias = None
        if args.has_bias:
            bias = rand_tensor((M, args.N), dtype=input_dtype, device=current_device)
        return A, B, bias

    dist_gemm_rs_op = GemmRSIntraNode(TP_GROUP, args.M, args.N, args.K, input_dtype, output_dtype, args.fuse_scatter)
    if args.check:
        for n in range(args.iters):
            torch.cuda.empty_cache()
            input_list = [_make_data(args.M) for _ in range(args.verify_iters)]
            dist_out_list, torch_out_list = [], []

            # torch impl
            for input, weight, bias in input_list:
                torch_out = torch_gemm_rs(input, weight, bias, TP_GROUP)
                torch_out_list.append(torch_out)

            # dist triton impl
            for input, weight, bias in input_list:
                dist_out = dist_gemm_rs_op.forward(input, weight)
                dist_out_list.append(dist_out)
            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dist_out_list)):
                try:
                    torch.testing.assert_close(torch_out, dist_out, atol=atol, rtol=rtol)
                except Exception as e:
                    raise e
        print(f"RANK[{RANK}]: pass.")
        exit(0)

    ctx = get_torch_prof_ctx(args.profile)
    input, weight, bias = _make_data(args.M)
    with ctx:
        torch_output, torch_perf = perf_func(partial(torch_gemm_rs, input, weight, bias, TP_GROUP), iters=args.iters,
                                             warmup_iters=args.warmup)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        dist_triton_output, dist_triton_perf = perf_func(partial(dist_gemm_rs_op.forward, input, weight),
                                                         iters=args.iters, warmup_iters=args.warmup)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
    torch.testing.assert_close(torch_output, dist_triton_output, atol=atol, rtol=rtol)
    torch.cuda.synchronize()

    dist_print(f"#{RANK} dist-triton {dist_triton_perf:0.3f} ms/iter" \
            f"torch {torch_perf:0.3f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    # Explicitly delete rocSHMEM-backed tensors before finalization
    # without explicit cleanup, rocshmem barrier_all collective operation
    # is called during python shutdown when some ranks may already have exited,
    # which may cause segfaults.
    del dist_gemm_rs_op
    finalize_distributed()
