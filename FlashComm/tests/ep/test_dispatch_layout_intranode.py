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
torchrun --nproc_per_node=2 tests/ep/test_dispatch_layout_intranode.py
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

import flash_comm._C.ep_intranode as _ep  # noqa: E402
from flash_comm.buffer import SymmetricTensor  # noqa: E402


def _time_cuda_ms_per_iter(fn, warmup: int = 5, iters: int = 20) -> float:
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


def ref_recv_base_offset(full_splits_src_dst_local: torch.Tensor) -> torch.Tensor:
    """
    full_splits_src_dst_local: [src_rank, dst_rank, local_expert] int32
    Return recv_base_offset[dst_rank, local_expert, src_rank] int32 where
      NOTE: this matches the CUDA kernel's flatten/scan order:
        linear order is (local_expert-major, then src-minor), i.e.
          (le=0,src=0..R-1), (le=1,src=0..R-1), ...

      offset(dst, le, src) =
        sum_{le'<le} sum_{src'} splits[src', dst, le']   +   sum_{src'<src} splits[src', dst, le].
    """
    # x: [dst, local_expert, src]
    x = full_splits_src_dst_local.permute(1, 2, 0).contiguous()
    # One cumsum along flattened (local_expert-major, src-minor) order.
    flat = x.reshape(x.size(0), -1)  # [dst, num_experts]
    prefix_excl = torch.cumsum(flat, dim=1) - flat
    return prefix_excl.reshape_as(x).to(torch.int32).contiguous()


def ref_send_mask(topk_indices: torch.Tensor, num_experts: int, num_ranks: int) -> torch.Tensor:
    """
    Return [num_token, topk] int32 mask where mask[t, j] == 1 iff this (token, j) is the
    first occurrence (smallest j) whose expert maps to a given target rank, ignoring drop/invalid.
    """
    experts_per_rank = num_experts // num_ranks
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    # target_rank could be == num_ranks for drop token (== num_experts); clamp for one_hot safety.
    target_rank = torch.div(topk_indices, experts_per_rank, rounding_mode="floor")
    target_rank = target_rank.clamp(min=0, max=num_ranks - 1)
    one_hot = F.one_hot(target_rank.to(torch.int64), num_classes=num_ranks).to(torch.int32)  # [T, K, R]
    one_hot = one_hot * valid.to(torch.int32).unsqueeze(-1)
    prefix = torch.cumsum(one_hot, dim=1)  # [T, K, R]
    first = (one_hot == 1) & (prefix == 1)
    return (first.any(dim=2)).to(torch.int32)


def test_dispatch_layout(*, num_experts: int, topk: int, num_token: int, profile: bool, num_sm: int):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    ep_group = dist.new_group(ranks=list(range(world)), backend="nccl")
    # Use LOCAL_RANK for correct device mapping on multi-GPU single-node runs.
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    ndev = torch.cuda.device_count()
    if ndev < world:
        # Avoid accidental oversubscription which can hang NCCL.
        if rank == 0:
            print(f"Skipping: need >= {world} GPUs but only have {ndev}.")
        dist.destroy_process_group()
        return
    torch.cuda.set_device(local_rank)

    if num_experts <= 0:
        raise ValueError(f"num_experts must be > 0, got {num_experts}")
    assert num_experts % world == 0
    experts_per_rank = num_experts // world
    if topk <= 0:
        raise ValueError(f"topk must be > 0, got {topk}")
    if num_token <= 0:
        raise ValueError(f"num_token must be > 0, got {num_token}")

    # Unless profiling, each rank randomly picks its local token count in [1, num_token].
    if profile:
        local_num_token = num_token
    else:
        # CPU-side RNG to avoid coupling to CUDA RNG state used elsewhere.
        g = torch.Generator(device="cpu")
        g.manual_seed(12345 + rank)
        local_num_token = int(torch.randint(1, num_token + 1, (1, ), generator=g).item())

    torch.manual_seed(0 + rank)
    # include drop token with some prob
    topk_indices = torch.randint(low=0, high=num_experts + 1, size=(local_num_token, topk), device="cuda",
                                 dtype=torch.int32)

    # local stable within-expert offsets + local_splits
    def _run_preprocess():
        return _ep.compute_stable_local_token_within_expert_offset_and_expert_counts(topk_indices, num_experts, num_sm)

    t_pre_ms = _time_cuda_ms_per_iter(_run_preprocess)
    token_within_expert_offset, _, local_splits = _run_preprocess()

    # reference full_splits via torch.distributed (GPU)
    full_splits = torch.empty((world, num_experts + 1), device="cuda", dtype=torch.int32)
    dist.all_gather_into_tensor(full_splits, local_splits, group=ep_group)
    full_splits_src_dst_local = full_splits[:, :num_experts].reshape(world, world, experts_per_rank)
    ref_recv = ref_recv_base_offset(full_splits_src_dst_local)

    ref_mask = ref_send_mask(topk_indices, num_experts, world)
    # Allocate symmetric buffers for full_splits_ptrs and barrier_ptrs
    # full_splits symm tensor: [world, num_experts+1] int32
    full_splits_symm = SymmetricTensor((world, num_experts + 1), torch.int32, group=ep_group, backend="auto")
    full_splits_ptrs = full_splits_symm.ptrs  # int64 CUDA tensor

    # barrier symm buffer: [world] int32
    barrier_buf = SymmetricTensor((world, ), torch.int32, group=ep_group, backend="auto")
    barrier_buf.get_local_tensor().zero_()
    barrier_ptrs = barrier_buf.ptrs
    torch.distributed.barrier(group=ep_group)

    def _run_layout():
        return _ep.compute_dispatch_layout(
            topk_indices,
            token_within_expert_offset,
            local_splits,
            full_splits_ptrs,
            barrier_ptrs,
            num_experts,
            rank,
            world,
            num_sm,
        )

    t_layout_ms = _time_cuda_ms_per_iter(_run_layout)
    recv_base_offset, token_dst_scatter_indices, token_topk_send_mask, recv_token_count_cpu, _ = _run_layout()

    # Aggregate timings (rank0 prints)
    t = torch.tensor([t_pre_ms, t_layout_ms], device="cuda", dtype=torch.float32)
    t_sum = t.clone()
    t_max = t.clone()
    dist.all_reduce(t_sum, op=dist.ReduceOp.SUM, group=ep_group)
    dist.all_reduce(t_max, op=dist.ReduceOp.MAX, group=ep_group)
    if rank == 0:
        t_avg = (t_sum / float(world)).tolist()
        t_mx = t_max.tolist()
        print(f"[timing] preprocess ms/iter avg={t_avg[0]:.3f} max={t_mx[0]:.3f} | "
              f"layout ms/iter avg={t_avg[1]:.3f} max={t_mx[1]:.3f} | "
              f"num_experts={num_experts} topk={topk} num_token_max={num_token} profile={int(profile)} num_sm={num_sm}")

    torch.testing.assert_close(token_topk_send_mask, ref_mask, atol=0, rtol=0)
    torch.testing.assert_close(recv_base_offset, ref_recv, atol=0, rtol=0)
    # Check the kernel-populated full_splits matches all_gather_into_tensor result
    torch.testing.assert_close(full_splits_symm.get_local_tensor(), full_splits, atol=0, rtol=0)
    # recv_token_count_cpu is pinned CPU written by GPU; synchronize before reading/validating.
    torch.cuda.synchronize()
    assert recv_token_count_cpu.device.type == "cpu"
    assert recv_token_count_cpu.is_pinned()
    ref_recv_token_count = full_splits_src_dst_local.sum(dim=(0, 2)).to(torch.int32).cpu()
    torch.testing.assert_close(recv_token_count_cpu, ref_recv_token_count, atol=0, rtol=0)

    # Also test passing an external pinned tensor (optional output)
    recv_token_count_out = torch.empty((world, ), dtype=torch.int32, pin_memory=True)
    _, _, _, recv_token_count_cpu, recv_token_count = _ep.compute_dispatch_layout(
        topk_indices,
        token_within_expert_offset,
        local_splits,
        full_splits_ptrs,
        barrier_ptrs,
        num_experts,
        rank,
        world,
        4,
        recv_token_count_cpu=recv_token_count_out,
    )
    torch.cuda.synchronize()
    assert recv_token_count_cpu.data_ptr() == recv_token_count_out.data_ptr()
    torch.testing.assert_close(recv_token_count_cpu, ref_recv_token_count, atol=0, rtol=0)
    assert recv_token_count.device.type == "cuda"
    torch.testing.assert_close(recv_token_count, ref_recv_token_count.to(recv_token_count.device), atol=0, rtol=0)

    # token_dst_scatter_indices reference
    valid = (topk_indices >= 0) & (topk_indices < num_experts)
    dst = torch.div(topk_indices, experts_per_rank, rounding_mode="floor")
    le = topk_indices - dst * experts_per_rank
    # safe indexing for invalid entries (will be overwritten to -1 by valid mask)
    dst_i = dst.clamp(min=0, max=world - 1).to(torch.int64)
    le_i = le.clamp(min=0, max=experts_per_rank - 1).to(torch.int64)
    base = ref_recv[dst_i, le_i, rank]  # [local_num_token, topk]
    ref_scatter = (base + token_within_expert_offset).to(torch.int32)
    ref_scatter = torch.where(valid, ref_scatter, torch.full_like(ref_scatter, -1))

    torch.testing.assert_close(token_dst_scatter_indices, ref_scatter, atol=0, rtol=0)

    dist.destroy_process_group(ep_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num_token", type=int, default=4096,
                        help="Max tokens per rank (or exact tokens if --profile).")
    parser.add_argument("--num_sm", type=int, default=4,
                        help="CUDA grid size override (SM count). Default -1 lets the kernel decide.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="If set, every rank uses exactly num_token tokens. Otherwise each rank samples tokens in [1, num_token].",
    )
    args = parser.parse_args()

    test_dispatch_layout(
        num_experts=args.num_experts,
        topk=args.topk,
        num_token=args.num_token,
        profile=args.profile,
        num_sm=args.num_sm,
    )
    print("✅ ep intranode dispatch_layout test passed.")
