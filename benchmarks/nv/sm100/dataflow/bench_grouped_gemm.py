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
"""Build, validate, and benchmark the SM100 cluster-level grouped GEMM."""

from __future__ import annotations

import argparse
import os
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.nv.sm100.dataflow.bench_report import ReportRow, SpeedupRow, print_benchmark_report, print_speedups
from examples.nv.sm100.grouped_gemm import GroupGemmV2

GROUPED_MM = getattr(F, "grouped_mm", getattr(torch, "_grouped_mm", None))

CHECK_RTOL = 6e-2
CHECK_ATOL = 6e-2
REFRESH_SECONDS = 0.4


def _expert_counts(*, m: int, num_experts: int, skew: float, empty_experts: int, seed: int) -> tuple[int, ...]:
    if not 0 <= empty_experts < num_experts:
        raise ValueError("empty_experts must be in [0, num_experts)")
    active = num_experts - empty_experts
    rng = random.Random(seed)
    weights = [pow(2.0, rng.uniform(-skew, skew)) for _ in range(active)]
    weight_sum = sum(weights)
    exact = [m * weight / weight_sum for weight in weights]
    counts = [int(value) for value in exact]
    remaining = m - sum(counts)
    order = sorted(range(active), key=lambda idx: exact[idx] - counts[idx], reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1
    counts.extend([0] * empty_experts)
    return tuple(counts)


@dataclass
class GroupGemmBench:
    compiled_kernel: object
    target: GroupGemmV2
    inputs: torch.Tensor
    weights: torch.Tensor
    offsets: torch.Tensor
    expert_counts: tuple[int, ...]
    m: int
    n: int
    k: int

    @property
    def device(self) -> torch.device:
        return self.inputs.device

    @property
    def padded_m(self) -> int:
        return int(self.target.a_padded.shape[0])

    def torch_grouped_mm(self) -> torch.Tensor:
        return GROUPED_MM(self.inputs, self.weights.transpose(-2, -1), offs=self.offsets)

    def launch_role(self, stream: torch.cuda.Stream | None = None) -> None:
        self.target.launch(stream=torch.cuda.current_stream() if stream is None else stream)


def prepare(args, compiled_kernel) -> GroupGemmBench:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", torch.cuda.current_device())))
    counts = _expert_counts(
        m=args.m,
        num_experts=args.num_experts,
        skew=args.skew,
        empty_experts=args.empty_experts,
        seed=args.seed,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    inputs = torch.randn((args.m, args.k), dtype=torch.bfloat16, device=device, generator=generator).mul_(0.1)
    weights = torch.randn(
        (args.num_experts, args.n, args.k),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.1)
    offsets = torch.tensor(counts, dtype=torch.int32, device=device).cumsum(0, dtype=torch.int32)
    target = GroupGemmV2.create(
        compiled_kernel,
        inputs=inputs,
        weights=weights,
        expert_counts=counts,
        num_clusters=args.num_clusters,
    )
    return GroupGemmBench(
        compiled_kernel=compiled_kernel,
        target=target,
        inputs=inputs,
        weights=weights,
        offsets=offsets,
        expert_counts=counts,
        m=args.m,
        n=args.n,
        k=args.k,
    )


def check_correctness(ctx: GroupGemmBench, *, iterations: int, rtol: float, atol: float) -> None:
    for iteration in range(iterations):
        if iteration:
            ctx.inputs.add_(0.01)
            ctx.target.copy_inputs(ctx.inputs)
        reference = ctx.torch_grouped_mm()
        ctx.target.output.fill_(float("nan"))
        ctx.launch_role()
        torch.cuda.synchronize(ctx.device)
        actual = ctx.target.valid_output()
        if not torch.isfinite(actual).all():
            raise AssertionError("LittleKernel grouped GEMM produced NaN/Inf")
        torch.testing.assert_close(actual.float(), reference.float(), rtol=rtol, atol=atol)
        max_abs = float((actual.float() - reference.float()).abs().max().item())
        print(f"correctness iteration {iteration}: pass; M={ctx.m}, padded_M={ctx.padded_m}, "
              f"N={ctx.n}, K={ctx.k}, max_abs={max_abs:.6g}")


def _time_once(run: Callable[[], object]) -> float:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    result = run()
    end.record(stream)
    end.synchronize()
    del result
    return start.elapsed_time(end)


def _time_role_once(launch: Callable[..., None]) -> float:
    stream = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    launch(stream=stream)
    end.record(stream)
    end.synchronize()
    return start.elapsed_time(end)


def _bench(once: Callable[[], float], *, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    time.sleep(REFRESH_SECONDS)
    once()  # Discard the first launch after idle clock wake-up.
    for _ in range(warmup):
        once()
    samples = [once() for _ in range(iters)]
    return statistics.median(samples)


def bench_compare(ctx: GroupGemmBench, *, warmup: int, iters: int) -> dict[str, float]:
    runners = {
        "torch_grouped_mm": partial(_time_once, ctx.torch_grouped_mm),
        "littlekernel": partial(_time_role_once, ctx.launch_role),
    }
    times = {name: _bench(run, warmup=warmup, iters=iters) for name, run in runners.items()}

    effective_flops = 2.0 * ctx.m * ctx.n * ctx.k
    physical_flops = 2.0 * ctx.padded_m * ctx.n * ctx.k
    print_benchmark_report(
        title=(f"GroupGEMM E={len(ctx.expert_counts)} M={ctx.m} padded_M={ctx.padded_m} N={ctx.n} K={ctx.k} "
               f"clusters={ctx.target.num_clusters}"),
        times=times,
        rows=[
            ReportRow("torch.grouped_mm:", "torch_grouped_mm", "tflops", "compute"),
            ReportRow("LittleKernel 2CTA:", "littlekernel", "tflops", "compute", "(effective FLOPs)"),
        ],
        flops=effective_flops,
        comm_bytes=0.0,
        device=ctx.device,
    )
    physical_tflops = physical_flops / (times["littlekernel"] * 1e9)
    padding_overhead = ctx.padded_m / ctx.m
    print(f"LittleKernel physical: {physical_tflops:.1f} TFLOP/s; padding_work={padding_overhead:.3f}x")
    print_speedups(times, [SpeedupRow("speedup:", "torch_grouped_mm", "littlekernel", "(torch / LittleKernel)")])
    return times


def export_trace(ctx: GroupGemmBench, trace_dir: str | Path, *, trace_iters: int, with_stack: bool) -> Path:
    from torch.profiler import ProfilerActivity, profile, record_function

    runners: dict[str, Callable[[], object]] = {
        "torch_grouped_mm": ctx.torch_grouped_mm,
        "littlekernel_groupgemm": lambda: ctx.launch_role(),
    }
    for run in runners.values():
        run()
    torch.cuda.synchronize(ctx.device)

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
    )
    prof.start()
    for label, run in runners.items():
        for _ in range(trace_iters):
            with record_function(label):
                run()
    prof.stop()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace = trace_dir / f"groupgemm_e{len(ctx.expert_counts)}_m{ctx.m}_n{ctx.n}_k{ctx.k}.json.gz"
    prof.export_chrome_trace(str(trace))
    print(f"trace: {trace}")
    return trace


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SM100 cluster-level grouped GEMM")
    parser.add_argument("--m", type=int, default=29440)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--num-experts", type=int, default=40)
    parser.add_argument("--skew", type=float, default=0.0)
    parser.add_argument("--empty-experts", type=int, default=0)
    parser.add_argument("--num-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check-iters", type=int, default=2)
    parser.add_argument("--check-rtol", type=float, default=CHECK_RTOL)
    parser.add_argument("--check-atol", type=float, default=CHECK_ATOL)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_groupgemm")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-dir", default=f"{os.path.dirname(__file__)}/lk_trace_role_v2_groupgemm")
    parser.add_argument("--trace-iters", type=int, default=5)
    parser.add_argument("--trace-with-stack", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        parser.error("--m, --n, and --k must be positive")
    if args.num_experts <= 0:
        parser.error("--num-experts must be positive")
    if args.skew < 0:
        parser.error("--skew must be non-negative")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("--warmup must be non-negative and --iters positive")
    if args.check_iters <= 0 or args.trace_iters <= 0:
        parser.error("--check-iters and --trace-iters must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("v2_groupgemm requires SM100 or newer")
    if GROUPED_MM is None:
        raise RuntimeError("PyTorch grouped_mm is required as the baseline")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    compiled_kernel = GroupGemmV2.build(cache_dir=args.cache_dir, verbose=args.verbose)
    _ = compiled_kernel.launcher
    ctx = prepare(args, compiled_kernel)
    torch.cuda.synchronize(ctx.device)

    print(f"args: {args}")
    print(f"work_items={ctx.target.num_work}, clusters={ctx.target.num_clusters}, ctas={ctx.target.num_ctas}, "
          f"counts=[{min(ctx.expert_counts)}, {max(ctx.expert_counts)}]")
    if not args.no_check:
        check_correctness(ctx, iterations=args.check_iters, rtol=args.check_rtol, atol=args.check_atol)
    bench_compare(ctx, warmup=args.warmup, iters=args.iters)
    if args.trace:
        export_trace(ctx, args.trace_dir, trace_iters=args.trace_iters, with_stack=args.trace_with_stack)
    ctx.target.release()


if __name__ == "__main__":
    main()
