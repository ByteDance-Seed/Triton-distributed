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
from triton_dist.utils import dist_print, initialize_distributed, finalize_distributed, rand_tensor
from triton_dist.kernels.amd.allgather_gemm import ag_gemm_intra_node, create_ag_gemm_intra_node_context, gemm_only, allgather


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
    A: torch.Tensor,  # [local_M, K]
    B: torch.Tensor,  # [local_N, K]
    bias: Optional[torch.Tensor],
    tp_group: torch.distributed.ProcessGroup,
):
    """ return C = all_gather(A) @ B.T """
    local_M, K = A.shape
    world_size = tp_group.size()
    assert K == B.shape[1]
    assert A.device == B.device
    # AG
    full_A = torch.empty((local_M * world_size, K), dtype=A.dtype, device=A.device)
    torch.distributed.all_gather_into_tensor(full_A, A, group=tp_group)
    # Gemm
    output = torch.matmul(full_A, B.T)

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
        B: torch.Tensor,  # [local_N, K]
        use_fused_kernel: bool = False,  # whether to use fused kernel
        autotune: bool = False,
    ):
        """ return C = all_gather(A) @ B.T """
        _, K = A.shape
        assert K == self.K
        assert self.max_M % self.world_size == 0
        assert B.shape == (self.N // self.world_size, K)

        output = ag_gemm_intra_node(A, B, ctx=self.ctx, use_fused_kernel=use_fused_kernel, autotune=autotune)

        return output

    def gemm_only(self, A: torch.Tensor, weight: torch.Tensor, NUM_SMS: int):
        return gemm_only(A, weight, ctx=self.ctx, NUM_SMS=NUM_SMS)

    def gemm_only_perf(self, A: torch.Tensor, weight: torch.Tensor, iters: int = 100, warmup_iters: int = 10,
                       profile=False):

        barrier_ptr = self.ctx.barrier_tensors[self.ctx.rank]
        barrier_ptr.fill_(1)
        torch.cuda.synchronize()
        NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
        with group_profile("gemm_only", profile, group=self.tp_group):
            _, gemm_perf = perf_func(partial(self.gemm_only, A, weight, NUM_SMS), iters=iters,
                                     warmup_iters=warmup_iters)
        barrier_ptr.fill_(0)
        dist_print("gemm only perf: ", gemm_perf, need_sync=True, allowed_ranks=list(range(self.world_size)))

        return gemm_perf

    def comm_only(self, input: torch.Tensor):
        return allgather(input, ctx=self.ctx)

    def comm_only_perf(self, input: torch.Tensor, iters: int = 100, warmup_iters: int = 10, profile=False):
        for comm_sms_per_rank in range(
                1, min(torch.cuda.get_device_properties(0).multi_processor_count // (self.world_size - 1), 8)):
            self.ctx.comm_sms = comm_sms_per_rank * (self.world_size - 1)
            with group_profile(f"comm_only_sm{self.ctx.comm_sms}", profile, group=self.tp_group):
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


def run_stress_test(args, TP_GROUP: torch.distributed.ProcessGroup):
    """Run stress test with random shapes"""
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
        M = M_per_rank * TP_GROUP.size()
        N = random.randint(128, max_N)
        K = random.randint(128, max_K) // 128 * 128
        chunk_m = random.choice([64, 128, 256, 512])

        dist_print(f"\nRound {round_idx + 1}/{args.stress_rounds}: M={M}, N={N}, K={K}, chunk_m={chunk_m}")

        try:
            # Create AG-GEMM operator
            dist_ag_gemm_op = AGGemmIntraNode(TP_GROUP, M, N, K, chunk_m, input_dtype, output_dtype,
                                              args.use_copy_kernel, args.comm_sms_per_rank)

            use_fused = args.use_fused_kernel

            for _ in range(10):
                A, B, bias = _make_data(M, N, K, args.has_bias, TP_GROUP)
                # Run distributed AG-GEMM
                dist_output = dist_ag_gemm_op.forward(A, B, use_fused, args.autotune)

                # Run reference PyTorch implementation
                torch_output = torch_ag_gemm(A, B, bias, TP_GROUP)

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


def _make_data(M, N, K, has_bias, tp_group: torch.distributed.ProcessGroup):
    current_device = torch.cuda.current_device()
    rank, num_ranks = tp_group.rank(), tp_group.size()
    local_M = M // num_ranks
    local_N = N // num_ranks
    scale = (rank + 1) * 0.01

    A = rand_tensor((local_M, K), dtype=input_dtype, device=current_device) * scale
    if args.transpose_weight:
        B = rand_tensor((K, local_N), dtype=input_dtype, device=current_device).T
    else:
        B = rand_tensor((local_N, K), dtype=input_dtype, device=current_device)
    bias = None
    if has_bias:
        bias = rand_tensor((M, local_N), dtype=input_dtype, device=current_device)
    return A, B, bias


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
    local_M = args.M // WORLD_SIZE
    local_N = args.N // WORLD_SIZE

    dist_ag_gemm_op = AGGemmIntraNode(TP_GROUP, args.M, args.N, args.K, args.chunk_m, input_dtype, output_dtype,
                                      args.use_copy_kernel, args.comm_sms_per_rank)

    A, B, bias = _make_data(args.M, args.N, args.K, args.has_bias, TP_GROUP)

    if args.gemm_only_perf:
        dist_ag_gemm_op.gemm_only_perf(A, B, iters=args.iters, warmup_iters=args.warmup, profile=args.profile)

    elif args.comm_only_perf:
        dist_ag_gemm_op.comm_only_perf(A, iters=args.iters, warmup_iters=args.warmup, profile=args.profile)

    else:

        if args.stress:
            # Run stress test with random shapes
            run_stress_test(args, TP_GROUP)

        torch_output = torch_ag_gemm(A, B, bias, TP_GROUP)
        dist_triton_output = dist_ag_gemm_op.forward(A, B, args.use_fused_kernel, args.autotune)

        with group_profile("ag_gemm", args.profile, group=TP_GROUP):

            _, dist_triton_perf = perf_func(
                partial(dist_ag_gemm_op.forward, A, B, args.use_fused_kernel, args.autotune), iters=args.iters,
                warmup_iters=args.warmup)

            _, torch_perf = perf_func(partial(torch_ag_gemm, A, B, bias, TP_GROUP), iters=args.iters,
                                      warmup_iters=args.warmup)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        atol, rtol = THRESHOLD_MAP[input_dtype], THRESHOLD_MAP[input_dtype]
        assert_allclose(torch_output, dist_triton_output, atol=atol, rtol=rtol)

        torch.cuda.synchronize()

        kernel_type = "fused" if args.use_fused_kernel else "producer-consumer"
        dist_print(f"#{RANK} dist-triton {kernel_type} {dist_triton_perf:0.3f} ms/iter torch {torch_perf:0.3f} ms/iter",
                   need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    # Explicitly delete rocSHMEM-backed tensors before finalization
    # without explicit cleanup, rocshmem barrier_all collective operation
    # is called during python shutdown when some ranks may already have exited,
    # which may cause segfaults.
    del dist_ag_gemm_op
    finalize_distributed()
