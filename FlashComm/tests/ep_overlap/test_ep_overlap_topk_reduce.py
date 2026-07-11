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
import time
from typing import Callable

import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


# Drop sentinel is ``num_experts``; the matching staging row is zeroed
# so naive blanket-add kernels still produce the right sum (isolates
# bandwidth-skip regressions from correctness regressions).
def make_inputs(num_tokens, topk, hidden_size, num_experts, drop_ratio, dtype, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    staging = torch.randn(
        num_tokens * topk,
        hidden_size,
        generator=g,
        dtype=dtype,
        device=device,
    )
    topk_indices = torch.randint(
        0,
        num_experts,
        (num_tokens, topk),
        generator=g,
        dtype=torch.int32,
        device=device,
    )
    drop_mask = ((torch.rand(num_tokens, topk, generator=g, device=device) < drop_ratio)
                 if drop_ratio > 0.0 else torch.zeros(num_tokens, topk, dtype=torch.bool, device=device))
    if drop_ratio > 0.0:
        topk_indices[drop_mask] = num_experts
        flat = (torch.arange(num_tokens, device=device).unsqueeze(1) * topk +
                torch.arange(topk, device=device).unsqueeze(0))[drop_mask]
        staging[flat] = 0

    # Mirror the kernel's sequential fp32 accumulation order (cast to
    # bf16 once at the end) so the reference is byte-equal, not just
    # numerically close.
    staging_3d = staging.view(num_tokens, topk, hidden_size).float()
    acc = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=device)
    for k in range(topk):
        acc = acc + staging_3d[:, k]
    expected = acc.to(dtype)
    return staging, topk_indices, expected, drop_mask


def bytes_naive(num_tokens, topk, hidden_size, dtype_bytes):
    return (topk + 1) * num_tokens * hidden_size * dtype_bytes


def bytes_effective(num_tokens, topk, hidden_size, dtype_bytes, drop_ratio):
    return ((1.0 - drop_ratio) * topk + 1.0) * num_tokens * hidden_size * dtype_bytes


# Kernel inputs are static-layout (so the verifier picks a 128 b SIMT
# atom); only the outer row count is captured via cute.copy indexing, so
# one compiled artifact runs for any ``num_tokens``.


def _wrap(staging, topk_indices, output):
    return (
        from_dlpack(staging, assumed_align=16),
        from_dlpack(topk_indices, assumed_align=4),
        from_dlpack(output, assumed_align=16),
    )


def compile_kernel(staging, topk_indices, output, hidden_size, topk, num_experts):
    from flash_comm.ep_overlap.kernels.cutedsl_topk_reduce import (
        MoETopkReduceBlockPerToken, )
    stream = cutlass_torch.current_stream()
    s_t, k_t, o_t = _wrap(staging, topk_indices, output)
    return cute.compile(
        MoETopkReduceBlockPerToken(),
        s_t,
        k_t,
        o_t,
        int(hidden_size),
        int(topk),
        int(num_experts),
        cutlass.Int32(0),
        stream,
    )


def make_runner(compiled, staging, topk_indices, output, num_tokens):
    s_t, k_t, o_t = _wrap(staging, topk_indices, output)
    stream = cutlass_torch.current_stream()
    nt32 = cutlass.Int32(int(num_tokens))

    def _run():
        compiled(s_t, k_t, o_t, nt32, stream)

    return _run


def assert_match(output, expected, *, label: str):
    """Byte-equality check; ``make_inputs`` mirrors the kernel reduce order."""
    if output.shape != expected.shape:
        raise AssertionError(f"[{label}] shape mismatch: {tuple(output.shape)} vs "
                             f"{tuple(expected.shape)}")
    if output.dtype != expected.dtype:
        raise AssertionError(f"[{label}] dtype mismatch: {output.dtype} vs {expected.dtype}")
    if torch.equal(output.view(torch.uint8), expected.view(torch.uint8)):
        return
    diff = (output.float() - expected.float()).abs()
    rel = diff / (expected.float().abs() + 1e-9)
    raise AssertionError(f"[{label}] FAIL non-byte-equal: max_abs={diff.max().item():.4e} "
                         f"max_rel={rel.max().item():.4e} "
                         f"(elements differing: {(diff > 0).sum().item()}/{output.numel()})")


def stress_check(runner, staging, topk_indices, output, expected, rounds, base_seed, *, num_tokens, topk, hidden_size,
                 num_experts, drop_ratio, dtype, device):
    """Re-randomise inputs ``rounds`` times and verify each round."""
    for round_idx in range(rounds):
        seed = base_seed + round_idx + 1
        staging_r, topk_r, expected_r, _ = make_inputs(
            num_tokens,
            topk,
            hidden_size,
            num_experts,
            drop_ratio,
            dtype,
            device,
            seed,
        )
        staging.copy_(staging_r)
        topk_indices.copy_(topk_r)
        output.zero_()
        runner()
        torch.cuda.synchronize()
        assert_match(
            output,
            expected_r,
            label=f"stress round {round_idx+1} (seed={seed})",
        )
    print(f"stress: {rounds} round(s) PASS")


def token_count_sweep_check(hidden_size, topk, num_experts, drop_ratio, *, dtype, device, base_seed):
    """Re-compile and verify across a sweep of ``num_tokens``."""
    sweep = [1, 16, 128, 1024, 2048, 4096, 8192]
    for n in sweep:
        staging, topk_indices, expected, _ = make_inputs(
            n,
            topk,
            hidden_size,
            num_experts,
            drop_ratio,
            dtype,
            device,
            base_seed + n,
        )
        output = torch.empty_like(expected)
        compiled = compile_kernel(
            staging,
            topk_indices,
            output,
            hidden_size,
            topk,
            num_experts,
        )
        runner = make_runner(compiled, staging, topk_indices, output, n)
        output.zero_()
        runner()
        torch.cuda.synchronize()
        assert_match(output, expected, label=f"num_tokens={n}")
    print(f"num_tokens sweep PASS: {sweep}")


def perf_event_ms(fn: Callable, iters: int, warmup: int, device_sleep_cycles: int = 0) -> float:
    """One CUDA event pair per iter, averaged; optional inter-iter sleep
    to dodge clock-throttling on tight loops."""
    if iters <= 0:
        raise ValueError(f"iters must be positive, got {iters}")
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    for i in range(iters):
        starts[i].record()
        fn()
        stops[i].record()
        if device_sleep_cycles > 0:
            torch.cuda._sleep(int(device_sleep_cycles))
    torch.cuda.synchronize()
    total = sum(s.elapsed_time(e) for s, e in zip(starts, stops))
    return total / iters


def perf_report(runner, args, dtype_bytes):
    ms = perf_event_ms(
        runner,
        args.iters,
        args.warmup,
        device_sleep_cycles=args.device_sleep_cycles,
    )
    bw_naive = bytes_naive(args.num_tokens, args.topk, args.hidden_size, dtype_bytes) / (ms * 1e-3) / 1e12
    bw_eff = bytes_effective(args.num_tokens, args.topk, args.hidden_size, dtype_bytes,
                             args.drop_ratio) / (ms * 1e-3) / 1e12
    print(f"{ms:.4f} ms  "
          f"BW_naive={bw_naive:.3f} TB/s  "
          f"BW_effective={bw_eff:.3f} TB/s")
    return ms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tokens", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=8192)
    parser.add_argument("--num_experts", type=int, default=160)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--drop_ratio", type=float, default=0.1)
    parser.add_argument("--num_sm", type=int, default=148, help="Number of SMs the topk-reduce kernel uses.")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check", action="store_true", help="Verify correctness before benchmarking.")
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--stress_rounds", type=int, default=0,
                        help="Stress-mode correctness rounds with rotated seeds.")
    parser.add_argument("--token_count_sweep", action="store_true",
                        help="Recompile + verify at several num_tokens values.")
    parser.add_argument("--cooldown_s", type=float, default=1.0, help="Sleep before each perf measurement to avoid "
                        "GPU clock-throttling bleed-through.")
    parser.add_argument(
        "--device_sleep_cycles", type=int, default=2_000_000, help="torch.cuda._sleep cycles between perf iters "
        "to prevent SM throttling (default 2e6 ~= "
        "1.4 ms at 1.5 GHz; set 0 to disable).")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16
    dtype_bytes = 2

    print(f"args: num_tokens={args.num_tokens} hidden_size={args.hidden_size} "
          f"topk={args.topk} num_experts={args.num_experts} "
          f"drop_ratio={args.drop_ratio} "
          f"iters={args.iters} warmup={args.warmup} num_sm={args.num_sm}")

    staging, topk_indices, expected, _ = make_inputs(
        args.num_tokens,
        args.topk,
        args.hidden_size,
        args.num_experts,
        args.drop_ratio,
        dtype,
        device,
        args.seed,
    )
    output = torch.empty_like(expected)
    bn = bytes_naive(args.num_tokens, args.topk, args.hidden_size, dtype_bytes)
    be = bytes_effective(args.num_tokens, args.topk, args.hidden_size, dtype_bytes, args.drop_ratio)
    print(f"naive bytes={bn / 1e9:.3f} GB  effective bytes={be / 1e9:.3f} GB  "
          f"(ratio={be / bn:.3f}x)")

    t0 = time.time()
    compiled = compile_kernel(
        staging,
        topk_indices,
        output,
        args.hidden_size,
        args.topk,
        args.num_experts,
    )
    compile_dt = time.time() - t0
    runner = make_runner(
        compiled,
        staging,
        topk_indices,
        output,
        args.num_tokens,
    )

    output.zero_()
    runner()
    torch.cuda.synchronize()
    if args.check or args.check_only:
        assert_match(output, expected, label="initial")
        print(f"correctness PASS (compile={compile_dt:.2f}s)")

    if args.stress_rounds > 0:
        stress_check(
            runner,
            staging,
            topk_indices,
            output,
            expected,
            args.stress_rounds,
            args.seed,
            num_tokens=args.num_tokens,
            topk=args.topk,
            hidden_size=args.hidden_size,
            num_experts=args.num_experts,
            drop_ratio=args.drop_ratio,
            dtype=dtype,
            device=device,
        )
        staging_d, topk_d, _, _ = make_inputs(
            args.num_tokens,
            args.topk,
            args.hidden_size,
            args.num_experts,
            args.drop_ratio,
            dtype,
            device,
            args.seed,
        )
        staging.copy_(staging_d)
        topk_indices.copy_(topk_d)

    if args.token_count_sweep:
        token_count_sweep_check(
            args.hidden_size,
            args.topk,
            args.num_experts,
            args.drop_ratio,
            dtype=dtype,
            device=device,
            base_seed=args.seed,
        )

    if args.check_only:
        return

    if args.cooldown_s > 0:
        time.sleep(args.cooldown_s)
    perf_report(runner, args, dtype_bytes)


if __name__ == "__main__":
    main()
