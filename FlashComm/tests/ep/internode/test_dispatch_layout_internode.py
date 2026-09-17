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
import sys
from pathlib import Path

import torch
import torch.distributed as dist

import flash_comm._C.ep_intranode as _ep_intra
import flash_comm._C.ep_internode as _ep_inter
from flash_comm.buffer.symmetric import SymmetricTensor

INTERNODE_DIR = Path(__file__).resolve().parent
EP_DIR = INTERNODE_DIR.parent
sys.path.insert(0, str(INTERNODE_DIR))
sys.path.insert(0, str(EP_DIR))
from test_dispatch_layout_intranode import (  # noqa: E402
    _time_cuda_ms_per_iter, ref_recv_base_offset, ref_send_mask,
)
from common import (  # noqa: E402
    destroy_ep_nccl, init_ep_nccl, resolve_local_world_size,
)


def test_dispatch_layout_internode(
    *,
    num_experts: int,
    topk: int,
    num_token: int,
    profile: bool,
    num_sm: int,
    expert_alignments: list,
    layout_rounds: int,
    max_layout_us: float,
):
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    torch.cuda.set_device(local_rank)

    local_world_size = resolve_local_world_size(world)
    assert world % local_world_size == 0, f"world={world} local_world_size={local_world_size}"
    nnodes = world // local_world_size
    assert num_experts % world == 0

    init_ep_nccl(dist.group.WORLD, local_world_size)

    experts_per_rank = num_experts // world
    if profile:
        local_num_token = num_token
    else:
        g = torch.Generator(device="cpu")
        g.manual_seed(12345 + rank)
        local_num_token = int(torch.randint(1, num_token + 1, (1, ), generator=g).item())

    torch.manual_seed(0 + rank)
    topk_indices = torch.randint(0, num_experts + 1, (local_num_token, topk), device="cuda", dtype=torch.int32)

    def _run_preprocess():
        return _ep_intra.compute_stable_local_token_within_expert_offset_and_expert_counts(
            topk_indices, num_experts, num_sm)

    t_pre_ms = _time_cuda_ms_per_iter(_run_preprocess)
    token_within_expert_offset, _, local_splits = _run_preprocess()

    full_splits_ref = torch.empty((world, num_experts + 1), device="cuda", dtype=torch.int32)
    gathered = [torch.empty_like(local_splits, device="cpu") for _ in range(world)]
    dist.all_gather(gathered, local_splits.cpu())
    full_splits_ref.copy_(torch.stack(gathered, dim=0).cuda())

    full_splits_row_ints = num_experts + 2
    full_splits_symm = SymmetricTensor(shape=(world, full_splits_row_ints), dtype=torch.int32, group=dist.group.WORLD,
                                       backend="nccl", local_world_size=local_world_size)
    full_splits_buf = full_splits_symm.get_local_tensor()
    full_splits_win = full_splits_symm.get_window_handle()
    num_tokens_per_rank = torch.empty((world, ), dtype=torch.int32, device="cuda")
    dist.barrier()

    for ea in expert_alignments:
        if rank == 0:
            print(f"\n--- internode expert_alignment={ea} (nnodes={nnodes}) ---")

        for layout_round in range(layout_rounds):
            if not profile:
                g = torch.Generator(device="cpu")
                g.manual_seed(12345 + rank + layout_round * 10007)
                round_num_token = int(torch.randint(1, num_token + 1, (1, ), generator=g).item())
            else:
                round_num_token = num_token

            torch.manual_seed(layout_round + rank)
            round_topk = torch.randint(0, num_experts + 1, (round_num_token, topk), device="cuda", dtype=torch.int32)
            round_off, _, round_splits = _ep_intra.compute_stable_local_token_within_expert_offset_and_expert_counts(
                round_topk, num_experts, num_sm)
            round_ref_mask = ref_send_mask(round_topk, num_experts, world)
            round_splits_gathered = [torch.empty_like(round_splits, device="cpu") for _ in range(world)]
            dist.all_gather(round_splits_gathered, round_splits.cpu())
            round_full_splits = torch.stack(round_splits_gathered, dim=0).cuda()
            round_full_splits_src_dst = round_full_splits[:, :num_experts].reshape(world, world, experts_per_rank)
            round_ref_recv = ref_recv_base_offset(round_full_splits_src_dst, ea)
            round_ref_recv_token_count = round_full_splits_src_dst.sum(dim=(0, 2)).to(torch.int32).cpu()
            round_ref_expert_counts = round_full_splits_src_dst[:, rank, :].sum(dim=0).to(torch.int32)

            def _run_layout():
                return _ep_inter.compute_dispatch_layout(
                    round_topk,
                    round_off,
                    round_splits,
                    full_splits_win,
                    num_tokens_per_rank,
                    num_experts,
                    num_sm,
                    expert_alignment=ea,
                )

            if layout_round == 0:
                t_layout_ms = _time_cuda_ms_per_iter(_run_layout)
            (
                recv_base_offset,
                token_dst_scatter_indices,
                token_topk_send_mask,
                recv_token_count_cpu,
                recv_token_count,
                recv_aligned_token_count_cpu,
                recv_aligned_token_count,
                recv_expert_counts,
            ) = _run_layout()

            if layout_round == 0:
                t = torch.tensor([t_pre_ms, t_layout_ms], device="cuda", dtype=torch.float32)
                t_sum = t.clone()
                t_max = t.clone()
                dist.all_reduce(t_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(t_max, op=dist.ReduceOp.MAX)
                if rank == 0:
                    t_avg = (t_sum / float(world)).tolist()
                    t_mx = t_max.tolist()
                    print(f"[timing] preprocess ms/iter avg={t_avg[0]:.3f} max={t_mx[0]:.3f} | "
                          f"layout ms/iter avg={t_avg[1]:.3f} max={t_mx[1]:.3f} | "
                          f"num_experts={num_experts} topk={topk} num_token_max={num_token} "
                          f"expert_alignment={ea} profile={int(profile)} num_sm={num_sm}")
                    if profile and max_layout_us > 0:
                        budget_us = max_layout_us * 1000.0
                        measured_us = t_mx[1] * 1000.0
                        assert t_mx[1] <= max_layout_us, (f"layout kernel steady min-max {measured_us:.1f}us exceeds "
                                                          f"{budget_us:.0f}us budget ({max_layout_us}ms)")

            torch.testing.assert_close(recv_base_offset, round_ref_recv, atol=0, rtol=0)
            torch.testing.assert_close(token_topk_send_mask, round_ref_mask, atol=0, rtol=0)
            torch.cuda.synchronize()
            torch.testing.assert_close(recv_token_count_cpu, round_ref_recv_token_count, atol=0, rtol=0)
            torch.testing.assert_close(
                recv_token_count,
                round_ref_recv_token_count.to(recv_token_count.device),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(recv_expert_counts, round_ref_expert_counts, atol=0, rtol=0)

            if ea > 1:
                x_dst_le_src = round_full_splits_src_dst.permute(1, 2, 0).contiguous()
                expert_totals = x_dst_le_src.sum(dim=2)
                aligned_totals = ((expert_totals + ea - 1) // ea) * ea
                ref_aligned_count = aligned_totals.sum(dim=1).to(torch.int32).cpu()
                torch.testing.assert_close(recv_aligned_token_count_cpu, ref_aligned_count, atol=0, rtol=0)
                torch.testing.assert_close(
                    recv_aligned_token_count,
                    ref_aligned_count.to(recv_aligned_token_count.device),
                    atol=0,
                    rtol=0,
                )

            round_dst = torch.div(round_topk, experts_per_rank, rounding_mode="floor")
            round_le = round_topk - round_dst * experts_per_rank
            round_dst_i = round_dst.clamp(0, world - 1).to(torch.int64)
            round_le_i = round_le.clamp(0, experts_per_rank - 1).to(torch.int64)
            round_valid = (round_topk >= 0) & (round_topk < num_experts)
            base = round_ref_recv[round_dst_i, round_le_i, rank]
            ref_scatter = (base + round_off).to(torch.int32)
            ref_scatter = torch.where(round_valid, ref_scatter, torch.full_like(ref_scatter, -1))
            torch.testing.assert_close(token_dst_scatter_indices, ref_scatter, atol=0, rtol=0)
            torch.testing.assert_close(full_splits_buf[:, :num_experts], round_full_splits[:, :num_experts], atol=0,
                                       rtol=0)
            ref_nt = torch.tensor([round_num_token], dtype=torch.int32, device="cpu")
            gathered_nt = [torch.empty(1, dtype=torch.int32) for _ in range(world)]
            dist.all_gather(gathered_nt, ref_nt)
            ref_nt_dev = torch.cat(gathered_nt).cuda()
            torch.testing.assert_close(num_tokens_per_rank, ref_nt_dev, atol=0, rtol=0)

        if rank == 0:
            print(f"  ✅ expert_alignment={ea} passed ({layout_rounds} layout rounds)")
        dist.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    full_splits_buf = None
    full_splits_symm.free()
    torch.cuda.synchronize()
    dist.barrier()
    destroy_ep_nccl()
    dist.destroy_process_group()
    if rank == 0:
        print("✅ internode dispatch_layout test passed (all alignment values).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num_token", type=int, default=4096,
                        help="Max tokens per rank (or exact tokens if --profile).")
    parser.add_argument("--num_sm", type=int, default=4)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--expert-alignment", type=str, default="1,32,128,256")
    parser.add_argument("--layout-rounds", type=int, default=10)
    parser.add_argument("--max-layout-us", type=float, default=0.05,
                        help="Max layout ms/iter (rank0 max) when --profile; 0.05 = 50us.")
    args = parser.parse_args()
    expert_alignments = [int(x) for x in args.expert_alignment.split(",")]
    test_dispatch_layout_internode(
        num_experts=args.num_experts,
        topk=args.topk,
        num_token=args.num_token,
        profile=args.profile,
        num_sm=args.num_sm,
        expert_alignments=expert_alignments,
        layout_rounds=args.layout_rounds,
        max_layout_us=args.max_layout_us,
    )
