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
"""
Test command:
python3 tests/ep/test_token_within_expert_offset_intranode.py
python3 tests/ep/test_token_within_expert_offset_intranode.py --profile --num_sm 140
"""

import argparse
import os

import torch
from torch.profiler import profile, ProfilerActivity

import flash_comm._C.ep_intranode as _ep  # noqa: E402


def _time_cuda_ms_per_iter(fn, iters: int = 50, warmup: int = 10) -> float:
    """
    Time a CUDA workload using CUDA events.
    Returns avg_ms_per_iter.
    """
    assert torch.cuda.is_available()
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / float(iters)


def _reference_global_rank(flat_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    flat_idx: [M] int32/int64 on CUDA, values in [0, num_experts] (num_experts is drop token)
    Return: [M] int32 where out[p] = #{q < p | flat_idx[q] == flat_idx[p]}
    """
    # Stable-sort by key so ties keep original order, then do run-length rank within each key.
    M = flat_idx.numel()
    perm = torch.argsort(flat_idx, stable=True)
    k = flat_idx[perm]

    i = torch.arange(M, device=flat_idx.device, dtype=torch.int64)
    head = torch.empty(M, device=flat_idx.device, dtype=torch.bool)
    head[0] = True
    head[1:] = k[1:] != k[:-1]

    boundary = torch.where(head, i, torch.full_like(i, -1))
    run_start = torch.cummax(boundary, dim=0).values
    rank_sorted = (i - run_start).to(torch.int32)

    out = torch.empty(M, device=flat_idx.device, dtype=torch.int32)
    out[perm] = rank_sorted
    return out


def _reference_block_cumsum_hist(flat_idx: torch.Tensor, num_experts: int, tile_size: int = 1024) -> torch.Tensor:
    """
    Returns [num_tiles, num_experts + 1] int32 exclusive prefix across tiles:
    out[t, e] = sum_{t' < t} hist[t', e]
    """
    M = flat_idx.numel()
    num_tiles = (M + tile_size - 1) // tile_size
    hist = torch.zeros((num_tiles, num_experts + 1), device=flat_idx.device, dtype=torch.int32)
    for t in range(num_tiles):
        s = t * tile_size
        e = min((t + 1) * tile_size, M)
        tile = flat_idx[s:e]
        hist[t] = torch.bincount(tile, minlength=num_experts + 1).to(torch.int32)
    # exclusive cumsum over tiles (torch.cumsum promotes to int64 for int tensors)
    return (torch.cumsum(hist.to(torch.int64), dim=0) - hist.to(torch.int64)).to(torch.int32)


def _run_with_profiler(fn, prof_dir: str, trace_name: str, iters: int = 10, warmup: int = 3):
    """
    Run fn under torch profiler and export trace to prof_dir.
    """
    os.makedirs(prof_dir, exist_ok=True)

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Profile and export chrome trace
    chrome_trace_path = os.path.join(prof_dir, f"{trace_name}.json")
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
    ) as prof:
        for _ in range(iters):
            fn()
            torch.cuda.synchronize()

    prof.export_chrome_trace(chrome_trace_path)
    print(f"[profiler] Chrome trace saved to: {chrome_trace_path}")


def test_token_within_expert_offset_intranode(*, num_experts: int, topk: int, num_token: int, seed: int,
                                              do_profile: bool, num_sm: int):
    if not torch.cuda.is_available():
        print("CUDA not available, skipping.")
        return

    torch.cuda.set_device(0)

    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")
    if num_token <= 0:
        raise ValueError(f"num_token must be > 0, got {num_token}")

    # Profile behavior: local_num_token fixed to num_token.
    # Non-profile behavior: randomly pick token count in [1, num_token] each run.
    if do_profile:
        local_num_token = num_token
    else:
        # CPU-side RNG to avoid coupling to CUDA RNG state used elsewhere.
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        local_num_token = int(torch.randint(1, num_token + 1, (1, ), generator=g).item())

    torch.manual_seed(seed)
    # Include drop token (num_experts) with some probability.
    topk_indices = torch.randint(low=0, high=num_experts + 1, size=(local_num_token, topk), device="cuda",
                                 dtype=torch.int32)

    # --- timing: kernel path ---
    def _run_offsets_only():
        return _ep.compute_stable_local_token_within_expert_offset(topk_indices, num_experts, num_sm)

    def _run_offsets_and_counts():
        return _ep.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, num_experts, num_sm)

    # --- torch profiler (only when --profile is set) ---
    if do_profile:
        prof_dir = os.path.join("./", "prof")
        trace_name = f"token_within_expert_offset_e{num_experts}_k{topk}_t{num_token}_sm{num_sm}"
        _run_with_profiler(_run_offsets_and_counts, prof_dir, trace_name, iters=10, warmup=3)

    kernel_ms = _time_cuda_ms_per_iter(_run_offsets_only, iters=50, warmup=10)
    kernel_ms2 = _time_cuda_ms_per_iter(_run_offsets_and_counts, iters=50, warmup=10)

    token_within_expert_offset, block_cumsum_hist = _run_offsets_only()
    token_within_expert_offset2, block_cumsum_hist2, expert_counts = _run_offsets_and_counts()

    # Also test the "write into provided expert_count" path.
    expert_count_out = torch.empty((num_experts + 1, ), device="cuda", dtype=torch.int32)
    _, _, expert_count2 = _ep.compute_stable_local_token_within_expert_offset_and_expert_counts(
        topk_indices, num_experts, num_sm, expert_count_out)
    torch.testing.assert_close(expert_count2, expert_count_out, atol=0, rtol=0)

    # --- timing: reference path ---
    flat = topk_indices.flatten()

    ref_rank_ms = _time_cuda_ms_per_iter(lambda: _reference_global_rank(flat, num_experts), iters=20, warmup=5)
    ref_hist_ms = _time_cuda_ms_per_iter(lambda: _reference_block_cumsum_hist(flat, num_experts, tile_size=1024),
                                         iters=10, warmup=2)

    print(f"[timing] offsets {kernel_ms:.3f} ms | "
          f"offsets+counts {kernel_ms2:.3f} ms | "
          f"ref_rank {ref_rank_ms:.3f} ms | "
          f"ref_hist {ref_hist_ms:.3f} ms | "
          f"num_experts={num_experts} topk={topk} num_token={num_token} local_num_token={local_num_token} "
          f"profile={int(do_profile)} num_sm={num_sm}")

    # --- correctness ---
    # 1) token_within_expert_offset should match stable "previous same value" count (within-expert ordinal).
    ref_rank = _reference_global_rank(flat, num_experts).view_as(topk_indices)
    torch.testing.assert_close(token_within_expert_offset, ref_rank, atol=0, rtol=0)
    torch.testing.assert_close(token_within_expert_offset2, token_within_expert_offset, atol=0, rtol=0)
    torch.testing.assert_close(block_cumsum_hist2, block_cumsum_hist, atol=0, rtol=0)

    # 2) block_cumsum_hist: validate first num_tiles rows (kernel tile size = 1024).
    ref_hist = _reference_block_cumsum_hist(flat, num_experts, tile_size=1024)
    assert block_cumsum_hist.shape[1] == num_experts + 1
    assert block_cumsum_hist.shape[0] >= ref_hist.shape[0]
    torch.testing.assert_close(block_cumsum_hist[:ref_hist.shape[0]], ref_hist, atol=0, rtol=0)

    # 3) expert_counts should equal global bincount of indices (including drop token)
    ref_count = torch.bincount(flat, minlength=num_experts + 1).to(torch.int32)
    torch.testing.assert_close(expert_counts, ref_count, atol=0, rtol=0)
    torch.testing.assert_close(expert_count_out, ref_count, atol=0, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num_token", type=int, default=4096, help="Max tokens (or exact tokens if --profile is set).")
    parser.add_argument("--seed", type=int, default=12345, help="Seed for sampling local_num_token and torch RNG.")
    parser.add_argument("--num_sm", type=int, default=-1,
                        help="CUDA grid size override (SM count). Default -1 lets the kernel decide.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If set, local_num_token == num_token. Otherwise sample local_num_token in [1, num_token].",
    )
    args = parser.parse_args()

    test_token_within_expert_offset_intranode(
        num_experts=args.num_experts,
        topk=args.topk,
        num_token=args.num_token,
        seed=args.seed,
        do_profile=args.profile,
        num_sm=args.num_sm,
    )
    print("✅ ep intranode token_within_expert_offset test passed.")
