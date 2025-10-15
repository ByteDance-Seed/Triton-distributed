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
import random

from triton_dist.profiler_utils import perf_func, group_profile
from triton_dist.test.utils import assert_allclose
from triton_dist.utils import generate_data, dist_print, initialize_distributed, finalize_distributed
from triton_dist.kernels.amd.all_gather_gemm import ag_gemm_intra_node, create_ag_gemm_intra_node_context, gemm_only, allgather


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


def torch_ag_gemm(
    input: torch.Tensor,  # [local_M, K]
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
        use_copy_kernel: bool = False,
        comm_sms_per_rank: int = 8,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.M_PER_CHUNK = M_PER_CHUNK
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        self.ctx = create_ag_gemm_intra_node_context(self.max_M, self.N, self.K, self.input_dtype, self.output_dtype,
                                                     self.rank, self.world_size, tp_group, M_PER_CHUNK=M_PER_CHUNK,
                                                     use_copy_kernel=use_copy_kernel,
                                                     comm_sms=comm_sms_per_rank * (self.world_size - 1))

    def forward(
        self,
        A: torch.Tensor,  # [local_M, K]
        weight: torch.Tensor,  # [local_N, K]
        transed_weight: bool,  # indicates whether weight already transposed
        use_fused_kernel: bool = False,  # whether to use fused kernel
        autotune: bool = False,
    ):

        _, K = A.shape

        assert K == self.K
        assert self.max_M % self.world_size == 0
        if transed_weight:
            assert weight.shape[0] == K
        else:
            assert weight.shape[1] == K

        output = ag_gemm_intra_node(A, weight, transed_weight, ctx=self.ctx, use_fused_kernel=use_fused_kernel,
                                    autotune=autotune)

        return output

    def gemm_only(self, A: torch.Tensor, weight: torch.Tensor, transed_weight: bool, NUM_SMS: int):
        return gemm_only(A, weight, transed_weight, ctx=self.ctx, NUM_SMS=NUM_SMS)

    def gemm_only_perf(self, A: torch.Tensor, weight: torch.Tensor, transed_weight: bool, iters: int = 100,
                       warmup_iters: int = 10):

        barrier_ptr = self.ctx.barrier_tensors[self.ctx.rank]
        barrier_ptr.fill_(1)
        torch.cuda.synchronize()
        NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
        with group_profile("gemm_only", True, group=self.tp_group):
            _, gemm_perf = perf_func(partial(self.gemm_only, A, weight, transed_weight, NUM_SMS), iters=iters,
                                     warmup_iters=warmup_iters)

        dist_print("gemm only perf: ", gemm_perf, need_sync=True, allowed_ranks=list(range(self.world_size)))

        return gemm_perf

    def comm_only(self, input: torch.Tensor):
        return allgather(input, ctx=self.ctx)

    def comm_only_perf(self, input: torch.Tensor, iters: int = 100, warmup_iters: int = 10):
        for comm_sms_per_rank in range(
                1, min(torch.cuda.get_device_properties(0).multi_processor_count // (self.world_size - 1), 17)):
            self.ctx.comm_sms = comm_sms_per_rank * (self.world_size - 1)
            _, comm_perf = perf_func(partial(self.comm_only, input), iters=iters, warmup_iters=warmup_iters)
            dist_print(f"comm only perf with {self.ctx.comm_sms} sms: ", comm_perf, need_sync=True,
                       allowed_ranks=list(range(self.world_size)))
        return comm_perf


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
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=20, type=int, help="perf iterations")
    parser.add_argument("--dtype", default="float16", type=str, help="data type")
    parser.add_argument("--use_copy_kernel", default=False, action="store_true", help="use copy kernel")
    parser.add_argument("--comm_sms_per_rank", default=8, type=int, help="communication sms per rank")
    parser.add_argument("--autotune", default=False, action="store_true", help="autotune")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--verify-iters", default=10, type=int)
    parser.add_argument("--stress", default=False, action="store_true", help="run stress test with random shapes")
    parser.add_argument("--stress_rounds", type=int, default=100, help="number of stress test rounds")

    parser.add_argument(
        "--transpose_weight",
        dest="transpose_weight",
        action=argparse.BooleanOptionalAction,
        help="transpose weight, default shape is [N, K]",
        default=False,
    )
    parser.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gemm_only_perf", default=False, action="store_true", help="gemm only perf")
    parser.add_argument("--comm_only_perf", default=False, action="store_true", help="comm only perf")
    parser.add_argument("--use_fused_kernel", default=False, action="store_true",
                        help="use fused all-gather-gemm kernel")

    return parser.parse_args()


def run_stress_test(args, TP_GROUP):
    """Run stress test with random shapes"""

    RANK = torch.distributed.get_rank()
    WORLD_SIZE = torch.distributed.get_world_size()

    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = input_dtype
    atol = THRESHOLD_MAP[output_dtype]
    rtol = THRESHOLD_MAP[output_dtype]

    max_M, max_N, max_K = args.M, args.N, args.K

    dist_print(f"Running stress test: {args.stress_rounds} rounds")
    dist_print(f"Max M={max_M}, Max N={max_N}, Max K={max_K}, dtype={input_dtype}")

    for round_idx in range(args.stress_rounds):
        # Generate random dimensions for each round
        M_per_rank = random.randint(128, max_K) // 128 * 128
        M = M_per_rank * WORLD_SIZE
        N = random.randint(128, max_N)
        K = random.randint(128, max_K) // 128 * 128
        chunk_m = random.choice([64, 128, 256, 512])

        local_M = M // WORLD_SIZE
        local_N = N // WORLD_SIZE

        dist_print(f"\nRound {round_idx + 1}/{args.stress_rounds}: M={M}, N={N}, K={K}, chunk_m={chunk_m}")

        try:
            # Create AG-GEMM operator
            dist_ag_gemm_op = AGGemmIntraNode(TP_GROUP, M, N, K, chunk_m, input_dtype, output_dtype,
                                              args.use_copy_kernel, args.comm_sms_per_rank)

            # Generate test data with different scales for each rank
            scale = RANK + 1

            input = (torch.rand((local_M, K), dtype=input_dtype, device="cuda") * 2 - 1) * 0.01 * scale
            weight = (torch.rand((local_N, K), dtype=input_dtype, device="cuda") * 2 - 1) * 0.01 * scale

            if args.transpose_weight:
                weight = weight.T.contiguous()

            bias = None
            if args.has_bias:
                bias = (torch.rand((M, local_N), dtype=input_dtype, device="cuda") * 2 - 1)
            use_fused = args.use_fused_kernel

            for _ in range(10):
                input.uniform_(-0.1 * scale, 0.1 * scale)
                weight.uniform_(-0.1 * scale, 0.1 * scale)
                # Run distributed AG-GEMM
                dist_output = dist_ag_gemm_op.forward(input, weight, args.transpose_weight, use_fused, args.autotune)

                # Run reference PyTorch implementation
                torch_output = torch_ag_gemm(input, weight, args.transpose_weight, bias, TP_GROUP)

                # Check correctness
                assert_allclose(torch_output, dist_output, atol=atol, rtol=rtol, verbose=False)

            dist_print(f"✅ Round {round_idx + 1} passed (fused={use_fused})")

        except Exception as e:
            dist_print(f"❌ Round {round_idx + 1} failed: {str(e)}")
            torch.cuda.synchronize()
            torch.distributed.barrier()
            raise RuntimeError(f"Stress test failed at round {round_idx + 1}: {str(e)}")

    torch.cuda.synchronize()
    torch.distributed.barrier()

    dist_print("\n" + "=" * 60)
    dist_print(f"✅ Stress test completed: All {args.stress_rounds} rounds passed")


if __name__ == "__main__":
    args = parse_args()
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed(args.seed, initialize_shmem=True)

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

    dist_ag_gemm_op = AGGemmIntraNode(TP_GROUP, args.M, args.N, args.K, args.chunk_m, input_dtype, output_dtype,
                                      args.use_copy_kernel, args.comm_sms_per_rank)

    A, weight, bias = _make_data()

    if args.gemm_only_perf:
        dist_ag_gemm_op.gemm_only_perf(A, weight, args.transpose_weight, iters=args.iters, warmup_iters=args.warmup)

    elif args.comm_only_perf:
        dist_ag_gemm_op.comm_only_perf(A, iters=args.iters, warmup_iters=args.warmup)

    else:

        if args.stress:
            # Run stress test with random shapes
            run_stress_test(args, TP_GROUP)

        torch_output = torch_ag_gemm(A, weight, args.transpose_weight, bias, TP_GROUP)
        dist_triton_output = dist_ag_gemm_op.forward(A, weight, args.transpose_weight, args.use_fused_kernel)

        with group_profile("ag_gemm", args.profile, group=TP_GROUP):

            _, dist_triton_perf = perf_func(
                partial(dist_ag_gemm_op.forward, A, weight, args.transpose_weight, args.use_fused_kernel),
                iters=args.iters, warmup_iters=args.warmup)

            _, torch_perf = perf_func(partial(torch_ag_gemm, A, weight, args.transpose_weight, bias, TP_GROUP),
                                      iters=args.iters, warmup_iters=args.warmup)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
        assert_allclose(torch_output, dist_triton_output, atol=atol, rtol=rtol)

        torch.cuda.synchronize()

        kernel_type = "fused" if args.use_fused_kernel else "producer-consumer"
        dist_print(f"#{RANK} dist-triton {kernel_type} {dist_triton_perf:0.3f} ms/iter torch {torch_perf:0.3f} ms/iter",
                   need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    finalize_distributed()
