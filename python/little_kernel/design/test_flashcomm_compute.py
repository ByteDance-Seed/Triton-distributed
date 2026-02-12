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
Correctness test for FlashComm compute kernels via LittleKernel.

Tests:
  1. kernel_compute_offset (single GPU)

Usage:
  CUDA_VISIBLE_DEVICES=3 python design/test_flashcomm_compute.py
"""

import os
import sys
import time

# LittleKernel imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
# Ensure design/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

# Import the LK kernel definitions
from flashcomm_compute import (
    kernel_compute_offset,
    NUM_WARPS,
    MAX_EXPERTS_PLUS_1,
    BLOCK_SIZE,
)


# ============================================================================
# Reference implementation (from FlashComm test)
# ============================================================================
def reference_global_rank(flat_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Stable within-expert offset:
    out[p] = #{q < p | flat_idx[q] == flat_idx[p]}
    """
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


def reference_expert_counts(flat_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Global bincount of expert indices (including drop token at index num_experts)."""
    return torch.bincount(flat_idx, minlength=num_experts + 1).to(torch.int32)


# ============================================================================
# Build kernel using LittleKernel's elegant Python binding
# ============================================================================
def build_compute_offset_kernel(num_sm: int = 4, num_experts: int = 256, verbose: bool = False):
    """Build the compute offset kernel using LK's build() API.
    
    shared_mem_bytes = NUM_WARPS * (num_experts + 1) * sizeof(int32)
    The kernel uses smem[NUM_WARPS * (num_experts + 1)] indexed as warp_hist = smem + warp_id * (num_experts + 1)
    """
    # smem size depends on actual num_experts, not just MAX_EXPERTS_PLUS_1
    actual_experts_plus_1 = max(num_experts + 1, MAX_EXPERTS_PLUS_1)
    smem_bytes = NUM_WARPS * actual_experts_plus_1 * 4
    compiled = kernel_compute_offset.build(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        grid=(num_sm, 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem_bytes=smem_bytes,
        verbose=verbose,
    )
    return compiled


# ============================================================================
# Test
# ============================================================================
def test_compute_offset(
    num_experts: int = 256,
    topk: int = 8,
    num_token: int = 4096,
    num_sm: int = 4,
    seed: int = 42,
):
    print(f"[test_compute_offset] num_experts={num_experts}, topk={topk}, "
          f"num_token={num_token}, num_sm={num_sm}, seed={seed}")

    torch.manual_seed(seed)
    device = "cuda"

    # Create test data: topk expert indices per token (includes drop token = num_experts)
    topk_indices = torch.randint(
        low=0,
        high=num_experts + 1,
        size=(num_token, topk),
        device=device,
        dtype=torch.int32,
    )

    # Allocate output tensors
    total_elements = num_token * topk
    num_tiles = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    # block_cumsum_hist: [num_tiles, num_experts + 1]
    block_cumsum_hist = torch.zeros(
        (num_tiles, num_experts + 1),
        device=device,
        dtype=torch.int32,
    )
    token_within_expert_offset = torch.zeros(
        (num_token, topk),
        device=device,
        dtype=torch.int32,
    )
    expert_counts = torch.zeros(
        (num_experts + 1, ),
        device=device,
        dtype=torch.int32,
    )

    # Build kernel
    print("  Building kernel...")
    t0 = time.time()
    compiled_kernel = build_compute_offset_kernel(num_sm=num_sm, num_experts=num_experts, verbose=True)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")

    # Launch kernel
    print("  Launching kernel...")
    compiled_kernel(
        topk_indices,
        num_token,
        topk,
        num_experts,
        block_cumsum_hist,
        token_within_expert_offset,
        expert_counts,
    )
    compiled_kernel.synchronize()

    # Reference
    flat = topk_indices.flatten()
    ref_rank = reference_global_rank(flat, num_experts).view(num_token, topk)
    ref_counts = reference_expert_counts(flat, num_experts)

    # Check token_within_expert_offset
    match_offset = torch.equal(token_within_expert_offset, ref_rank)
    if not match_offset:
        diff_mask = token_within_expert_offset != ref_rank
        num_diff = diff_mask.sum().item()
        print(f"  FAIL: token_within_expert_offset mismatch at {num_diff}/{num_token * topk} positions")
        # Show first few mismatches
        indices = diff_mask.nonzero()[:10]
        for idx in indices:
            t, k = idx[0].item(), idx[1].item()
            print(f"    [{t},{k}]: got={token_within_expert_offset[t,k].item()}, "
                  f"ref={ref_rank[t,k].item()}, expert={topk_indices[t,k].item()}")
        return False
    print("  PASS: token_within_expert_offset matches reference")

    # Check expert_counts
    match_counts = torch.equal(expert_counts, ref_counts)
    if not match_counts:
        diff_mask = expert_counts != ref_counts
        num_diff = diff_mask.sum().item()
        print(f"  FAIL: expert_counts mismatch at {num_diff} positions")
        diff_indices = diff_mask.nonzero().flatten()[:10]
        for idx in diff_indices:
            print(f"    expert[{idx.item()}]: got={expert_counts[idx].item()}, ref={ref_counts[idx].item()}")
        return False
    print("  PASS: expert_counts matches reference")

    # Performance
    def bench():
        compiled_kernel(
            topk_indices,
            num_token,
            topk,
            num_experts,
            block_cumsum_hist,
            token_within_expert_offset,
            expert_counts,
        )

    # Warmup
    for _ in range(10):
        bench()
    compiled_kernel.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    iters = 50
    start.record()
    for _ in range(iters):
        bench()
    end.record()
    torch.cuda.synchronize()
    kernel_ms = start.elapsed_time(end) / iters

    print(f"  PERF: {kernel_ms:.3f} ms/iter "
          f"({total_elements / kernel_ms / 1e6:.1f} M elements/s)")

    return True


# ============================================================================
# Test: kernel_compute_dispatch_layout (single-GPU, simulated num_ranks=1)
# ============================================================================
from flashcomm_compute import kernel_compute_dispatch_layout  # noqa: E402


def build_dispatch_layout_kernel(num_sm: int = 4, verbose: bool = False):
    """Build the dispatch layout kernel."""
    compiled = kernel_compute_dispatch_layout.build(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        grid=(num_sm, 1, 1),
        block=(BLOCK_SIZE, 1, 1),
        shared_mem_bytes=NUM_WARPS * 4,  # scan_warp_prefix_sum: 32 int32s
        verbose=verbose,
    )
    return compiled


def ref_send_mask(topk_indices: torch.Tensor, num_experts: int, num_ranks: int) -> torch.Tensor:
    """
    Return [num_token, topk] int32 mask where mask[t, j] == 1 iff this is the
    first occurrence (smallest j) whose expert maps to a given target rank.
    """
    import torch.nn.functional as F
    experts_per_rank = num_experts // num_ranks
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    target_rank = torch.div(topk_indices, experts_per_rank, rounding_mode="floor")
    target_rank = target_rank.clamp(min=0, max=num_ranks - 1)
    one_hot = F.one_hot(target_rank.to(torch.int64), num_classes=num_ranks).to(torch.int32)
    one_hot = one_hot * valid.to(torch.int32).unsqueeze(-1)
    prefix = torch.cumsum(one_hot, dim=1)
    first = (one_hot == 1) & (prefix == 1)
    return (first.any(dim=2)).to(torch.int32)


def make_ptr_tensor(tensors: list) -> torch.Tensor:
    """Create an int64 device tensor holding data_ptr() values (simulating int32_t**)."""
    ptrs = [t.data_ptr() for t in tensors]
    return torch.tensor(ptrs, dtype=torch.int64, device=tensors[0].device)


def test_dispatch_layout(
    num_experts: int = 256,
    topk: int = 8,
    num_token: int = 4096,
    num_sm: int = 4,
    seed: int = 42,
):
    # Simulate single rank for single-GPU test
    num_ranks = 1
    rank = 0
    experts_per_rank = num_experts // num_ranks

    print(f"[test_dispatch_layout] num_experts={num_experts}, topk={topk}, "
          f"num_token={num_token}, num_sm={num_sm}, num_ranks={num_ranks}, seed={seed}")

    torch.manual_seed(seed)
    device = "cuda"

    # Generate test data
    topk_indices = torch.randint(
        low=0,
        high=num_experts + 1,
        size=(num_token, topk),
        device=device,
        dtype=torch.int32,
    )

    # First compute token_within_expert_offset using our verified kernel
    flat = topk_indices.flatten()
    token_within_expert_offset = reference_global_rank(flat, num_experts).view(num_token, topk)
    local_splits = reference_expert_counts(flat, num_experts)

    # Allocate symmetric-like buffers for ptr-to-ptr arguments
    # full_splits: [num_ranks, num_ranks * (num_experts + 1)]
    # With num_ranks=1: full_splits_bufs[0] = tensor of [1 * (num_experts + 1)]
    full_splits_bufs = [
        torch.zeros(num_ranks * (num_experts + 1), device=device, dtype=torch.int32) for _ in range(num_ranks)
    ]
    full_splits_ptrs = make_ptr_tensor(full_splits_bufs)

    # barrier: [num_ranks] per rank
    barrier_bufs = [torch.zeros(num_ranks, device=device, dtype=torch.int32) for _ in range(num_ranks)]
    barrier_ptrs = make_ptr_tensor(barrier_bufs)

    # Output tensors
    recv_base_offset = torch.zeros(
        num_ranks * experts_per_rank * num_ranks,
        device=device,
        dtype=torch.int32,
    )
    token_dst_scatter_indices = torch.full(
        (num_token, topk),
        -1,
        device=device,
        dtype=torch.int32,
    )
    token_topk_send_mask = torch.zeros(
        (num_token, topk),
        device=device,
        dtype=torch.int32,
    )
    recv_token_count = torch.zeros(num_ranks, device=device, dtype=torch.int32)

    # Build kernel
    print("  Building kernel...")
    t0 = time.time()
    compiled_kernel = build_dispatch_layout_kernel(num_sm=num_sm, verbose=True)
    build_time = time.time() - t0
    print(f"  Build time: {build_time:.2f}s")

    # Launch kernel
    # Args: topk_indices, token_within_expert_offset, local_splits,
    #        full_splits_ptrs, barrier_ptrs,
    #        recv_base_offset, token_dst_scatter_indices, token_topk_send_mask,
    #        recv_token_count_cpu, recv_token_count,
    #        num_token, topk, num_experts, rank, num_ranks
    print("  Launching kernel...")
    compiled_kernel(
        topk_indices,
        token_within_expert_offset,
        local_splits,
        full_splits_ptrs,  # int64 tensor -> int32_t**
        barrier_ptrs,  # int64 tensor -> int32_t**
        recv_base_offset,
        token_dst_scatter_indices,
        token_topk_send_mask,
        0,  # recv_token_count_cpu (NULL = pass 0)
        recv_token_count,
        num_token,
        topk,
        num_experts,
        rank,
        num_ranks,
    )
    compiled_kernel.synchronize()

    # Verify send mask
    ref_mask = ref_send_mask(topk_indices, num_experts, num_ranks)
    match_mask = torch.equal(token_topk_send_mask, ref_mask)
    if not match_mask:
        diff = (token_topk_send_mask != ref_mask).sum().item()
        print(f"  FAIL: send_mask mismatch at {diff} positions")
        return False
    print("  PASS: token_topk_send_mask matches reference")

    # Verify recv_token_count
    # With num_ranks=1, recv_token_count[0] should be total non-drop tokens
    valid_count = (topk_indices < num_experts).sum().item()
    got_count = recv_token_count[0].item()
    if got_count != valid_count:
        print(f"  FAIL: recv_token_count[0]={got_count}, expected={valid_count}")
        return False
    print(f"  PASS: recv_token_count={got_count} (expected {valid_count})")

    # Verify scatter indices
    # With num_ranks=1, scatter_idx = recv_base_offset[0 * num_experts + local_expert * 1 + 0]
    #                                + token_within_expert_offset
    # recv_base_offset for single rank is just exclusive cumsum of local_splits
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    expert_cumsum = torch.cumsum(local_splits[:num_experts], dim=0) - local_splits[:num_experts]
    ref_scatter = torch.where(
        valid,
        expert_cumsum[topk_indices.clamp(0, num_experts - 1).long()] + token_within_expert_offset,
        torch.tensor(-1, device=device, dtype=torch.int32),
    )
    match_scatter = torch.equal(token_dst_scatter_indices, ref_scatter)
    if not match_scatter:
        diff = (token_dst_scatter_indices != ref_scatter).sum().item()
        print(f"  FAIL: scatter_indices mismatch at {diff} positions")
        indices = (token_dst_scatter_indices != ref_scatter).nonzero()[:10]
        for idx in indices:
            t, k = idx[0].item(), idx[1].item()
            print(f"    [{t},{k}]: got={token_dst_scatter_indices[t,k].item()}, "
                  f"ref={ref_scatter[t,k].item()}, expert={topk_indices[t,k].item()}")
        return False
    print("  PASS: token_dst_scatter_indices matches reference")

    return True


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("FlashComm Kernel Correctness Tests (via LittleKernel)")
    print("=" * 70)

    all_results = []

    # Test 1: kernel_compute_offset
    print("\n--- Test: kernel_compute_offset ---")
    offset_configs = [
        {"num_experts": 256, "topk": 8, "num_token": 1024, "num_sm": 4, "seed": 42},
        {"num_experts": 384, "topk": 8, "num_token": 4096, "num_sm": 4, "seed": 123},
        {"num_experts": 64, "topk": 4, "num_token": 512, "num_sm": 4, "seed": 999},
    ]
    for cfg in offset_configs:
        print()
        passed = test_compute_offset(**cfg)
        all_results.append(("compute_offset", cfg, passed))

    # Test 2: kernel_compute_dispatch_layout (single GPU, num_ranks=1)
    print("\n--- Test: kernel_compute_dispatch_layout ---")
    layout_configs = [
        {"num_experts": 256, "topk": 8, "num_token": 1024, "num_sm": 4, "seed": 42},
        {"num_experts": 384, "topk": 8, "num_token": 4096, "num_sm": 4, "seed": 123},
        {"num_experts": 64, "topk": 4, "num_token": 512, "num_sm": 4, "seed": 999},
    ]
    for cfg in layout_configs:
        print()
        passed = test_dispatch_layout(**cfg)
        all_results.append(("dispatch_layout", cfg, passed))

    print()
    print("=" * 70)
    print("Summary:")
    for name, cfg, passed in all_results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name} E={cfg['num_experts']}, K={cfg['topk']}, T={cfg['num_token']}")
    total_pass = sum(1 for _, _, p in all_results if p)
    print(f"\n  {total_pass}/{len(all_results)} tests passed")
    print("=" * 70)

    if total_pass < len(all_results):
        sys.exit(1)
