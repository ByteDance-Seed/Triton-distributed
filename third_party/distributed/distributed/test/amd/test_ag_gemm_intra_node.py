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
import random

import argparse
import os
from typing import Optional
import datetime
import numpy as np

from functools import partial

from triton.distributed.utils import (
    generate_data,
    get_torch_prof_ctx,
    perf_func,
    dist_print,
)
from triton.distributed.kernels.amd import ag_gemm_intra_node, create_ag_gemm_intra_node_context


def torch_ag_gemm(
    input: torch.Tensor,  # [local_M, k]
    weight: torch.Tensor,  # [local_N, K]
    transed_weight: bool,
    bias: Optional[torch.Tensor],
    TP_GROUP,
):
    local_M, K = input.shape
    world_size = TP_GROUP.size()
    if transed_weight:
        assert K == weight.shape[0]
    else:
        assert K == weight.shape[1]
        weight = weight.T
    assert input.device == weight.device
    # AG
    full_input = torch.empty((local_M * world_size, K), dtype=input.dtype, device=input.device)
    torch.distributed.all_gather_into_tensor(full_input, input, group=TP_GROUP)
    # Gemm
    output = torch.matmul(full_input, weight)

    if bias:
        output = output + bias

    return output


class AGGemmIntraNode(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_M: int,
        N: int,
        K: int,
        M_PER_CHUNK: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
    ):
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.M_PER_CHUNK = M_PER_CHUNK
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        # NOTE: use the default size of `M_PER_CHUNK`.
        self.ctx = create_ag_gemm_intra_node_context(
            self.max_M,
            self.N,
            self.K,
            self.input_dtype,
            self.output_dtype,
            self.rank,
            self.world_size,
            self.tp_group,
            M_PER_CHUNK=M_PER_CHUNK,
        )

    def forward(self, input: torch.Tensor,  # [local_M, K]
                weight: torch.Tensor,  # [local_N, K]
                transed_weight: bool,  # indicates whether weight already transposed
                ):

        _, K = input.shape

        assert K == self.K
        assert self.max_M % self.world_size == 0
        if transed_weight:
            assert weight.shape[0] == K
        else:
            assert weight.shape[1] == K
        output = ag_gemm_intra_node(input, weight, transed_weight, ctx=self.ctx)

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
    torch.bfloat16: 1e-2,
    torch.float8_e4m3fn: 1e-2,
    torch.float8_e5m2: 1e-2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--chunk_m", default=256, type=int, help="chunk size at dim m")
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
        help="transpose weight, default shape is [N, K]",
        default=False,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    # init
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=5)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)
    random.seed(args.seed)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    assert args.M % WORLD_SIZE == 0
    assert args.N % WORLD_SIZE == 0
    assert args.K % WORLD_SIZE == 0
    local_M = args.M // WORLD_SIZE
    local_N = args.N // WORLD_SIZE

    scale = TP_GROUP.rank() + 1

    def _make_data():
        data_config = [
            ((local_M, args.K), input_dtype, (0.01 * scale, 0)),  # A
            ((local_N, args.K), input_dtype, (0.01 * scale, 0)),  # B
            (  # bias
                None if not args.has_bias else ((args.M, local_N), input_dtype, (1, 0))),
        ]
        generator = generate_data(data_config)
        input, weight, bias = next(generator)
        if args.transpose_weight:
            weight = weight.T.contiguous()  # from N,K to K,N
        return input, weight, bias

    dist_ag_gemm_op = AGGemmIntraNode(TP_GROUP, args.M, args.N, args.K, args.chunk_m, input_dtype, output_dtype)

    ctx = get_torch_prof_ctx(args.profile)
    input, weight, bias = _make_data()

    with ctx:
        torch_output, torch_perf = perf_func(
            partial(torch_ag_gemm, input, weight, args.transpose_weight, bias, TP_GROUP), iters=args.iters,
            warmup_iters=args.warmup)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        dist_triton_output, dist_triton_perf = perf_func(
            partial(dist_ag_gemm_op.forward, input, weight, args.transpose_weight), iters=args.iters,
            warmup_iters=args.warmup)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]

    if torch.allclose(dist_triton_output, torch_output, atol=atol, rtol=rtol):
        dist_print("✅ Triton and Torch match")
    else:
        dist_print(
            f"The maximum difference between torch and triton is {torch.max(torch.abs(dist_triton_output - torch_output))}"
        )
        dist_print("❌ Triton and Torch differ")

    torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", dist_triton_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()
