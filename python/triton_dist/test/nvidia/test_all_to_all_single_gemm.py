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
import torch
import torch.distributed
from typing import Optional
from tabulate import tabulate

from triton_dist.kernels.nvidia import (
    create_all_to_all_single_gemm_context,
    all_to_all_single_gemm,
)
from triton_dist.kernels.nvidia.all_to_all_single_gemm import gemm_only
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    is_fp8_dtype,
    dist_print,
    assert_allclose,
    group_profile,
    sleep_async,
)


def make_cuda_graph(mempool, func):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            func()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=mempool):
        func()
    return graph


class PerfResult:

    def __init__(
        self,
        name: str,
        output: torch.Tensor,
        a2a_output: torch.Tensor,
        total_ms: float,
        time1: str,
        gemm_time_ms: float,
        time2: str,
        comm_time_ms: float,
    ) -> None:
        self.name = name
        self.output = output
        self.a2a_output = a2a_output
        self.total_ms = total_ms
        self.time1 = time1
        self.time2 = time2
        self.gemm_time_ms = gemm_time_ms
        self.comm_time_ms = comm_time_ms

    def __repr__(self) -> str:
        return (f"{self.name}: total {self.total_ms:.3f} ms, {self.time1} {self.gemm_time_ms:.3f} ms"
                f", {self.time2} {self.comm_time_ms:.3f} ms")


DTYPE_MAP = {
    "int8": torch.int8,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=1024, help="Number of rows")
    parser.add_argument("--N", type=int, default=512, help="Number of output columns")
    parser.add_argument("--K", type=int, default=256, help="Number of input columns")
    parser.add_argument("--dtype", type=str, default="bf16", choices=DTYPE_MAP.keys())
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--check", default=False, action="store_true", help="Run correctness check with random shapes")
    parser.add_argument("--check_rounds", type=int, default=10, help="Number of stress test rounds")
    parser.add_argument("--profile", default=False, action="store_true", help="Run with profiler")
    return parser.parse_args()


def matmul_int8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    INT8 matrix multiplication using torch._int_mm
    torch._int_mm requires A.size(0) needs to be greater than 16
    b is expected to be (N, K) and will be transposed to (K, N)
    """
    M, _ = a.shape
    b_t = b.t()  # Transpose from (N, K) to (K, N)
    if M <= 16:
        return torch._int_mm(torch.nn.functional.pad(a, (0, 0, 0, 32 - M)), b_t)[:M, :]
    return torch._int_mm(a, b_t)


def torch_a2a(input, sp_group, output=None):
    """Reference PyTorch all-to-all implementation"""
    if input is None:
        return None

    if output is None:
        output = torch.empty_like(input)

    assert input.dtype == output.dtype and input.shape == output.shape
    is_fp8 = is_fp8_dtype(input.dtype)
    dtype = input.dtype

    if is_fp8:
        input = input.view(torch.int8)
        output = output.view(torch.int8)

    torch.distributed.all_to_all_single(output, input, group=sp_group, async_op=False)

    if is_fp8:
        input = input.view(dtype)
        output = output.view(dtype)

    return output


@torch.no_grad()
def perf_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    warmup: int,
    iters: int,
    sp_group,
):
    """Benchmark PyTorch implementation with separated timing"""
    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    a2a_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    is_s8 = input.dtype == torch.int8
    is_fp8 = is_fp8_dtype(input.dtype)

    torch.distributed.barrier()
    sleep_async(200)
    for i in range(total_iters):
        start_events[i].record()

        # All-to-all communication
        input_after_a2a = torch_a2a(input, sp_group=sp_group)
        if input_scale is not None:
            input_scale_after_a2a = torch_a2a(input_scale, sp_group=sp_group)
        else:
            input_scale_after_a2a = None

        a2a_end_events[i].record()

        # GEMM computation
        if is_s8:
            accum = matmul_int8(input_after_a2a, weight).to(torch.float32)
            output = input_scale_after_a2a * weight_scale * accum
        elif is_fp8:
            dequant_scale = input_scale_after_a2a * weight_scale if input_scale_after_a2a is not None else 1.0
            input_after_a2a = input_after_a2a.to(torch.bfloat16)
            weight_bf16 = weight.to(torch.bfloat16)
            output = dequant_scale * torch.matmul(input_after_a2a, weight_bf16.t())
        else:
            output = torch.matmul(input_after_a2a, weight.t())

        if (is_s8 or is_fp8) and output.dtype != torch.bfloat16:
            output = output.to(torch.bfloat16)

        end_events[i].record()

    comm_times = []
    gemm_times = []

    for i in range(total_iters):
        a2a_end_events[i].synchronize()
        end_events[i].synchronize()
        comm_times.append(start_events[i].elapsed_time(a2a_end_events[i]))
        gemm_times.append(a2a_end_events[i].elapsed_time(end_events[i]))

    comm_time_ms = sum(comm_times[warmup_iters:]) / iters
    gemm_time_ms = sum(gemm_times[warmup_iters:]) / iters
    total_ms = comm_time_ms + gemm_time_ms

    return PerfResult(
        name="PyTorch",
        output=output,
        a2a_output=input_after_a2a,
        total_ms=total_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
    )


@torch.no_grad()
def perf_triton(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor],
    weight_scale: Optional[torch.Tensor],
    context,
    warmup: int,
    iters: int,
    sp_group,
):
    """Benchmark Triton implementation with separated timing"""
    torch.distributed.barrier()

    warmup_iters = warmup
    total_iters = warmup_iters + iters

    # get the input after A2A for gemm_only measurement
    input_after_a2a = torch_a2a(input, sp_group=sp_group)
    input_scale_after_a2a = torch_a2a(input_scale, sp_group=sp_group) if input_scale is not None else None

    # Measure GEMM-only time
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    sleep_async(200)
    for i in range(total_iters):
        start_events[i].record()

        _ = gemm_only(
            input=input_after_a2a,
            weight=weight,
            input_scale=input_scale_after_a2a,
            weight_scale=weight_scale,
        )

        end_events[i].record()

    torch.cuda.current_stream().synchronize()

    gemm_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            gemm_times.append(start_events[i].elapsed_time(end_events[i]))

    gemm_time_ms = sum(gemm_times) / iters

    # Wait a bit between measurements
    torch.cuda.synchronize()
    torch.distributed.barrier()
    output = all_to_all_single_gemm(
        input=input,
        weight=weight,
        context=context,
        input_scale=input_scale,
        weight_scale=weight_scale,
    )

    # Measure fused A2A+GEMM time
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    torch.distributed.barrier()
    sleep_async(200)
    for i in range(total_iters):
        start_events[i].record()
        output = all_to_all_single_gemm(
            input=input,
            weight=weight,
            context=context,
            input_scale=input_scale,
            weight_scale=weight_scale,
        )

        end_events[i].record()

    torch.distributed.barrier()
    torch.cuda.current_stream().synchronize()

    # Calculate total time
    total_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            total_times.append(start_events[i].elapsed_time(end_events[i]))

    total_ms = sum(total_times) / iters

    # Calculate communication time
    comm_time_ms = total_ms - gemm_time_ms

    return PerfResult(
        name="Triton",
        output=output,
        a2a_output=input_after_a2a,
        total_ms=total_ms,
        time1="gemm",
        gemm_time_ms=gemm_time_ms,
        time2="comm",
        comm_time_ms=comm_time_ms,
    )


def run_perf_test(args):
    """Run performance test and correctness verification"""
    # Get rank info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    dtype = DTYPE_MAP[args.dtype]
    M, N, K = args.M, args.N, args.K

    dist_print(f"Running performance test: M={M}, N={N}, K={K}, dtype={dtype}")

    # Ensure M is divisible by world size
    M = (M // world_size) * world_size

    # Create test tensors using flux-style data generation
    is_int8 = dtype == torch.int8
    is_fp8 = is_fp8_dtype(dtype)

    scale = rank + 1  # Different scale for each rank

    if is_int8:
        input = torch.randint(-127, 127, (M, K), dtype=dtype, device="cuda")
        weight = torch.randint(-127, 127, (N, K), dtype=dtype, device="cuda")
        input_scale = torch.rand((M, 1), dtype=torch.float32, device="cuda") * 2 - 1
        weight_scale = torch.rand((1, N), dtype=torch.float32, device="cuda") * 2 - 1
    elif is_fp8:
        # Generate FP8 tensors with controlled range
        input_f16 = (torch.rand((M, K), dtype=torch.float16, device="cuda") * 2 - 1) * 0.01 * scale
        weight_f16 = (torch.rand((N, K), dtype=torch.float16, device="cuda") * 2 - 1) * 0.01 * scale
        input = input_f16.to(dtype)
        weight = weight_f16.to(dtype)
        input_scale = torch.rand((M, 1), dtype=torch.float32, device="cuda") * 2 - 1
        weight_scale = torch.rand((1, N), dtype=torch.float32, device="cuda") * 2 - 1
    else:
        input = (torch.rand((M, K), dtype=dtype, device="cuda") * 2 - 1) * 0.01 * scale
        weight = (torch.rand((N, K), dtype=dtype, device="cuda") * 2 - 1) * 0.01 * scale
        input_scale = None
        weight_scale = None

    # Create context
    context = create_all_to_all_single_gemm_context(
        max_m=M,
        n=N,
        k=K,
        rank=rank,
        local_world_size=local_world_size,
        dtype=dtype,
    )

    # Run implementations
    sp_group = torch.distributed.group.WORLD
    torch_result = perf_torch(input, weight, input_scale, weight_scale, warmup=1, iters=1, sp_group=sp_group)
    triton_result = perf_triton(input, weight, input_scale, weight_scale, context, warmup=1, iters=1, sp_group=sp_group)

    # Compare results
    rtol = 1e-2
    atol = 1e-2

    assert_allclose(triton_result.output, torch_result.output, rtol=rtol, atol=atol)


def benchmark(args):
    """Benchmark the implementation"""
    # Get rank info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    dtype = DTYPE_MAP[args.dtype]
    M, N, K = args.M, args.N, args.K

    # Ensure M is divisible by world size
    M = (M // world_size) * world_size

    dist_print(f"Benchmarking: M={M}, N={N}, K={K}, dtype={dtype}")
    dist_print(f"Warmup: {args.warmup}, Iterations: {args.iters}")

    is_int8 = dtype == torch.int8
    is_fp8 = is_fp8_dtype(dtype)

    scale = rank + 1  # Different scale for each rank

    if is_int8:
        input = torch.randint(-127, 127, (M, K), dtype=dtype, device="cuda")
        weight = torch.randint(-127, 127, (N, K), dtype=dtype, device="cuda")
        input_scale = torch.rand((M, 1), dtype=torch.float32, device="cuda") * 2 - 1
        weight_scale = torch.rand((1, N), dtype=torch.float32, device="cuda") * 2 - 1
    elif is_fp8:
        # Generate FP8 tensors with controlled range
        input_f16 = (torch.rand((M, K), dtype=torch.float16, device="cuda") * 2 - 1) * 0.01 * scale
        weight_f16 = (torch.rand((N, K), dtype=torch.float16, device="cuda") * 2 - 1) * 0.01 * scale
        input = input_f16.to(dtype)
        weight = weight_f16.to(dtype)
        input_scale = torch.rand((M, 1), dtype=torch.float32, device="cuda") * 2 - 1
        weight_scale = torch.rand((1, N), dtype=torch.float32, device="cuda") * 2 - 1
    else:
        input = (torch.rand((M, K), dtype=dtype, device="cuda") * 2 - 1) * 0.01 * scale
        weight = (torch.rand((N, K), dtype=dtype, device="cuda") * 2 - 1) * 0.01 * scale
        input_scale = None
        weight_scale = None

    # Create context
    context = create_all_to_all_single_gemm_context(
        max_m=M,
        n=N,
        k=K,
        rank=rank,
        local_world_size=local_world_size,
        dtype=dtype,
    )

    sp_group = torch.distributed.group.WORLD

    with group_profile("a2a single gemm", args.profile):
        # Run benchmarks
        dist_print("\nRunning PyTorch benchmark...")
        torch_result = perf_torch(input, weight, input_scale, weight_scale, args.warmup, args.iters, sp_group)

        dist_print("Running Triton benchmark...")
        triton_result = perf_triton(input, weight, input_scale, weight_scale, context, args.warmup, args.iters,
                                    sp_group)

    # check allclose
    rtol = 1e-2
    atol = 1e-2
    assert_allclose(triton_result.output, torch_result.output, rtol=rtol, atol=atol)

    # Calculate FLOPS
    flops = 2 * M * N * K
    torch_tflops = flops / (torch_result.total_ms * 1e9)
    triton_tflops = flops / (triton_result.total_ms * 1e9)

    # Print results
    speedup = torch_result.total_ms / triton_result.total_ms
    dist_print("\n" + "=" * 80)
    dist_print(f"Performance Results: Speedup: {speedup:.2f}x")
    dist_print("=" * 80)

    results = [
        {
            "Implementation": "PyTorch",
            "Total (ms)": f"{torch_result.total_ms:.3f}",
            "GEMM (ms)": f"{torch_result.gemm_time_ms:.3f}",
            "Comm (ms)": f"{torch_result.comm_time_ms:.3f}",
            "TFLOPS": f"{torch_tflops:.2f}",
        },
        {
            "Implementation": "Triton",
            "Total (ms)": f"{triton_result.total_ms:.3f}",
            "GEMM (ms)": f"{triton_result.gemm_time_ms:.3f}",
            "Comm (ms)": f"{triton_result.comm_time_ms:.3f}",
            "TFLOPS": f"{triton_tflops:.2f}",
        },
    ]

    dist_print(tabulate(results, headers="keys", tablefmt="grid"))


def check_correctness(args):
    """Check correctness with random shapes (similar to flux)"""
    import random

    # Get rank info from environment
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    dtype = DTYPE_MAP[args.dtype]
    max_M, N, K = args.M, args.N, args.K
    is_int8 = dtype == torch.int8
    is_fp8 = is_fp8_dtype(dtype)

    dist_print(f"Running stress test: {args.check_rounds} rounds")
    dist_print(f"Max M={max_M}, N={N}, K={K}, dtype={dtype}")

    passed_rounds = 0
    failed_rounds = 0

    for round_idx in range(args.check_rounds):
        # Generate random M for each iteration
        torch.cuda.empty_cache()

        # Random M values
        # TODO: hanshi.s, currently only supports perfect blocks
        BLOCK_M = 128
        M_values = [
            ((random.randint(1, max_M) + (world_size * BLOCK_M) - 1) // (world_size * BLOCK_M) * (world_size * BLOCK_M))
            for _ in range(args.iters)
        ]

        dist_print(f"\nRound {round_idx + 1}/{args.check_rounds}")

        all_passed = True
        for _, M in enumerate(M_values):
            # Generate test data
            scale = rank + 1

            if is_int8:
                input = torch.randint(-127, 127, (M, K), dtype=dtype, device="cuda")
                weight = torch.randint(-127, 127, (N, K), dtype=dtype, device="cuda")
                input_scale = torch.rand((M, 1), dtype=torch.float32, device="cuda") * 2 - 1
                weight_scale = torch.rand((1, N), dtype=torch.float32, device="cuda") * 2 - 1
            elif is_fp8:
                input_f16 = (torch.rand((M, K), dtype=torch.float16, device="cuda") * 2 - 1) * 0.01 * scale
                weight_f16 = (torch.rand((N, K), dtype=torch.float16, device="cuda") * 2 - 1) * 0.01 * scale
                input = input_f16.to(dtype)
                weight = weight_f16.to(dtype)
                input_scale = torch.rand((M, 1), dtype=torch.float32, device="cuda") * 2 - 1
                weight_scale = torch.rand((1, N), dtype=torch.float32, device="cuda") * 2 - 1
            else:
                input = (torch.rand((M, K), dtype=dtype, device="cuda") * 2 - 1) * 0.01 * scale
                weight = (torch.rand((N, K), dtype=dtype, device="cuda") * 2 - 1) * 0.01 * scale
                input_scale = None
                weight_scale = None

            context = create_all_to_all_single_gemm_context(
                max_m=M,
                n=N,
                k=K,
                rank=rank,
                local_world_size=local_world_size,
                dtype=dtype,
            )

            sp_group = torch.distributed.group.WORLD
            torch_result = perf_torch(input, weight, input_scale, weight_scale, warmup=1, iters=1, sp_group=sp_group)
            triton_result = perf_triton(input, weight, input_scale, weight_scale, context, warmup=1, iters=1,
                                        sp_group=sp_group)

            rtol = 1e-2
            atol = 1e-2

            assert_allclose(triton_result.output, torch_result.output, rtol=rtol, atol=atol, verbose=False)

        if all_passed:
            dist_print(f"✅ Round {round_idx + 1} passed")
            passed_rounds += 1
        else:
            failed_rounds += 1

    dist_print("\n" + "=" * 60)
    dist_print(f"✅ Stress test completed: {passed_rounds}/{args.check_rounds} rounds passed")
    if failed_rounds > 0:
        dist_print(f"Failed rounds: {failed_rounds}")
    assert passed_rounds == args.check_rounds


if __name__ == "__main__":
    args = parse_args()
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE if LOCAL_WORLD_SIZE > 0 else 1
    SP_GROUP = initialize_distributed(seed=args.seed)
    torch.manual_seed(args.seed)

    dist_print("=" * 80)
    dist_print("All-to-All Single GEMM Test")
    dist_print("=" * 80)
    dist_print(f"World Size: {WORLD_SIZE}, Nodes: {NNODES}, Local World Size: {LOCAL_WORLD_SIZE}")
    dist_print(f"M={args.M}, N={args.N}, K={args.K}")
    dist_print(f"Dtype: {args.dtype}")
    dist_print("=" * 80)

    if args.check:
        check_correctness(args)

    benchmark(args)

    torch.distributed.barrier()
    finalize_distributed()
