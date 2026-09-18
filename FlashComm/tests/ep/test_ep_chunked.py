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

################################################################################
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
################################################################################
"""Deferred bitwise stress for serial fixed-buffer EP on one or many nodes.

Examples:
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunked.py
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunked.py --local-world-size=4
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunked.py --local-world-size=2
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist

from chunked_ep_reference import (
    assert_bytes_equal,
    compact_layout_witnesses,
    execute_range_identity,
    fixed_capacity,
    make_routing,
    one_shot_capacity,
    range_identity,
    standard_identity,
)
from flash_comm.ep import EPChunkPlanner, EPKernels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-token-per-rank", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--capacity-chunks", type=int, default=1)
    parser.add_argument("--stress-rounds", type=int, default=12)
    parser.add_argument("--hot-expert-rounds", type=int, default=0)
    parser.add_argument("--hot-rank-rounds", type=int, default=0)
    parser.add_argument("--num-sm", type=int, default=16)
    parser.add_argument("--local-world-size", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    return parser.parse_args()


def enqueue_intranode_dispatch_reuse_probe(ep: EPKernels, planner: EPChunkPlanner, *, capacity: int,
                                           max_num_tokens: int, hidden: int, topk: int, num_experts: int,
                                           expert_alignment: int, rank: int, world: int):
    """Keep a peer-visible dispatch read live across the next range dispatch.

    Each expert receives exactly ``expert_alignment`` routes, so the compared
    receive prefix contains no padding or stale slots.  The first A snapshot
    must remain identical to the final A snapshot even though a distinct B
    dispatch is pushed while the first D2D read is still queued.
    """
    numerator = expert_alignment * num_experts
    denominator = world * topk
    if numerator % denominator:
        raise AssertionError("dispatch reuse probe requires an integral token count")
    num_tokens = numerator // denominator
    if num_tokens <= 0 or num_tokens > max_num_tokens:
        raise AssertionError("dispatch reuse probe token count is outside planner bounds")

    routing = make_routing("round_robin", rank, world, num_tokens, topk, num_experts,
                           91031).pin_memory().cuda(non_blocking=True)
    offsets, _counts = (ep.compute_stable_local_token_within_expert_offset_and_expert_counts(routing))
    plan = planner.build(routing, recv_capacity_tokens=capacity, expert_alignment=expert_alignment)
    layout = ep.prepare_chunk_layouts(routing, offsets, plan)[0]

    generator = torch.Generator(device="cuda").manual_seed(92011 + rank)
    input_a = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda", generator=generator)
    input_b = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device="cuda", generator=generator)
    recv_tokens = (num_experts // world) * expert_alignment

    dispatch_a1, _weights, _layout = ep.dispatch(input_a, routing, None, layout)
    snapshot_a1 = dispatch_a1[:recv_tokens].clone()
    dispatch_b, _weights, _layout = ep.dispatch(input_b, routing, None, layout)
    # Keep a second peer-visible consumer in flight so the following A push
    # exercises the same retirement edge in consecutive calls.
    snapshot_b = dispatch_b[:recv_tokens].clone()
    dispatch_a2, _weights, _layout = ep.dispatch(input_a, routing, None, layout)
    snapshot_a2 = dispatch_a2[:recv_tokens].clone()
    return snapshot_a1, snapshot_a2, snapshot_b


def main():
    args = parse_args()
    os.environ.setdefault("NCCL_GIN_ENABLE", "1")
    os.environ.setdefault("NCCL_GIN_TYPE", "3")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        timeout=datetime.timedelta(seconds=args.timeout_seconds),
    )
    rank = dist.get_rank()
    world = dist.get_world_size()
    group = dist.new_group(list(range(world)), backend="nccl")
    local_world_size = args.local_world_size or int(
        os.environ.get("EP_LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", world)))
    if world % local_world_size:
        raise ValueError("local-world-size must divide world size")
    num_experts = args.num_experts
    if num_experts % world:
        raise ValueError("num-experts must be divisible by world size")
    experts_per_rank = num_experts // world
    capacity = fixed_capacity(world, args.chunk_size, args.topk, experts_per_rank, args.expert_alignment,
                              args.capacity_chunks)
    oracle_capacity = one_shot_capacity(world, args.max_token_per_rank, args.topk, experts_per_rank,
                                        args.expert_alignment)

    planner = EPChunkPlanner(
        max_num_tokens=args.max_token_per_rank,
        chunk_size=args.chunk_size,
        num_experts=num_experts,
        group=group,
        local_world_size=local_world_size,
    )
    range_ep = EPKernels(
        args.max_token_per_rank,
        args.hidden,
        args.topk,
        num_experts,
        local_world_size,
        group,
        num_sm=args.num_sm,
        expert_alignment=args.expert_alignment,
        num_worst_tokens=capacity,
        check_num_worst_tokens=True,
    )
    oracle_ep = EPKernels(
        args.max_token_per_rank,
        args.hidden,
        args.topk,
        num_experts,
        local_world_size,
        group,
        num_sm=args.num_sm,
        expert_alignment=args.expert_alignment,
        num_worst_tokens=oracle_capacity,
    )
    patterns = (
        "arbitrary",
        "uniform",
        "hot_rank",
        "one_expert",
        "duplicate_expert",
        "all_to_self",
        "all_to_next",
        "round_robin",
        "drop_heavy",
        "all_drop",
    )
    edge_sizes = (
        0,
        1,
        max(1, args.chunk_size - 1),
        args.chunk_size,
        min(args.max_token_per_rank, args.chunk_size + 1),
        args.max_token_per_rank,
    )
    workloads = []
    for index, size in enumerate(edge_sizes):
        local_size = max(0, size - (rank * 19 + index * 11) % max(1, min(size, 89)))
        workloads.append((f"edge_{index}", patterns[index % len(patterns)], local_size, 29000 + index))
    rank_boundaries = (
        0,
        1,
        max(0, args.chunk_size - 1),
        min(args.max_token_per_rank, args.chunk_size),
        min(args.max_token_per_rank, args.chunk_size + 1),
        max(0, args.max_token_per_rank - 1),
        args.max_token_per_rank,
    )
    workloads.extend((
        ("rank_boundary_matrix", "arbitrary", rank_boundaries[rank % len(rank_boundaries)], 29990),
        ("alternating_empty_full", "arbitrary", args.max_token_per_rank if rank % 2 else 0, 29991),
        ("single_nonempty_rank", "arbitrary", args.max_token_per_rank if rank == world - 1 else 0, 29992),
        ("rank_staircase", "arbitrary", args.max_token_per_rank * rank // max(1, world - 1), 29993),
    ))
    workloads.append(("mixed_zero_rank", "hot_rank", 0 if rank % 3 == 0 else args.chunk_size, 29997))
    workloads.append(("all_empty", "uniform", 0, 29998))
    workloads.append(("all_drop_nonempty", "all_drop", rank_boundaries[(rank + 3) % len(rank_boundaries)], 29999))
    workloads.append(("forced_multichunk", "one_expert_with_drop", args.max_token_per_rank, 30000))
    replay_index = len(workloads) - 1
    size_generator = torch.Generator(device="cpu").manual_seed(2731 + rank)
    for index in range(args.stress_rounds):
        size = int(torch.randint(0, args.max_token_per_rank + 1, (), generator=size_generator))
        workloads.append((f"round_{index}", patterns[index % len(patterns)], size, 30000 + index))
    for index in range(args.hot_expert_rounds):
        workloads.append((
            f"hot_expert_{index}",
            "rotating_hot_expert",
            args.max_token_per_rank,
            index,
        ))
    for index in range(args.hot_rank_rounds):
        workloads.append((
            f"hot_rank_{index}",
            "global_hot_rank",
            args.max_token_per_rank,
            index,
        ))
    workloads.append(workloads[replay_index])

    pending = []
    replay = None
    replay_layouts = None
    replay_plan = None
    all_empty_layouts = None
    x_generator = torch.Generator(device="cuda").manual_seed(4049 + rank)
    prepared = []
    # Finish every input allocation before queueing the synchronization-free batch.
    for index, (label, pattern, size, seed) in enumerate(workloads):
        if index == len(workloads) - 1:
            x, routing, weights = replay
        else:
            routing_cpu = make_routing(pattern, rank, world, size, args.topk, num_experts, seed)
            routing = routing_cpu.pin_memory().cuda(non_blocking=True)
            x = torch.randn((size, args.hidden), dtype=torch.bfloat16, device="cuda", generator=x_generator)
            weights = None
            if index % 2 or label in ("all_drop_nonempty", "forced_multichunk"):
                weights = torch.randn((size, args.topk), dtype=torch.float32, device="cuda", generator=x_generator)
            if index == replay_index:
                replay = (x, routing, weights)
        prepared.append((label, x, routing, weights))
    torch.cuda.synchronize()

    for index, (label, x, routing, weights) in enumerate(prepared):
        # Deliberately vary rank readiness without synchronizing the round.
        torch.cuda._sleep((rank + 1) * (index % 7 + 1) * 4096)
        if index == len(workloads) - 1:
            actual, actual_weight = execute_range_identity(range_ep, x, routing, replay_layouts, weights)
            plan = replay_plan
        else:
            actual, actual_weight, plan, layouts = range_identity(range_ep, planner, x, routing, capacity,
                                                                  args.expert_alignment, weights)
            if label == "all_empty":
                all_empty_layouts = layouts
            if index == replay_index:
                replay_layouts = layouts
                replay_plan = plan
        pending.append((
            label,
            actual.clone(),
            actual_weight.clone() if actual_weight is not None else None,
            x,
            routing,
            weights,
            plan.logical_token_ranges.clone(),
        ))

    # Reference phase begins only after every optimized workload was enqueued.
    # Standard EP may poll its own CPU count buffers here; it cannot hide races
    # between optimized rounds because all of those rounds are already queued.
    expected = [
        tuple(value.clone() if value is not None else None
              for value in standard_identity(oracle_ep, x, routing, weights))
        for _label, _actual, _actual_weight, x, routing, weights, _schedule in pending
    ]
    local_sizes = [
        int(routing.shape[0]) for (_label, _actual, _actual_weight, _x, routing, _weights, _schedule) in pending
    ]
    size_matrix = [None for _ in range(world)]
    dist.all_gather_object(size_matrix, local_sizes)
    layout_witnesses = compact_layout_witnesses(oracle_ep, replay[0], replay[1], replay_layouts)
    dispatch_reuse_probe = None
    if not range_ep.is_internode:
        dispatch_reuse_probe = enqueue_intranode_dispatch_reuse_probe(
            range_ep, planner, capacity=capacity, max_num_tokens=args.max_token_per_rank, hidden=args.hidden,
            topk=args.topk, num_experts=num_experts, expert_alignment=args.expert_alignment, rank=rank, world=world)

    # Deferred verification: no per-round host synchronization or recovery in
    # the optimized phase.
    torch.cuda.synchronize()
    failure = ""
    try:
        if all_empty_layouts is None:
            raise AssertionError("stress suite did not retain all-empty layouts")
        for step, layout in enumerate(all_empty_layouts):
            assert_bytes_equal(
                layout.recv_expert_counts.cpu(),
                torch.zeros(tuple(layout.recv_expert_counts.shape), dtype=layout.recv_expert_counts.dtype),
                f"all_empty/step{step}/expert_count/rank{rank}")
        if (world > 1 and not any(
                len({size_matrix[peer][workload]
                     for peer in range(world)}) > 1
                for workload in range(len(workloads)))):
            raise AssertionError("stress suite did not exercise different token counts per rank")
        for (label, actual, actual_weight, _x, _routing, _weights,
             _schedule), (reference, reference_weight) in zip(pending, expected):
            assert_bytes_equal(actual.cpu(), reference.cpu(), f"{label}/rank{rank}/output")
            if (actual_weight is None) != (reference_weight is None):
                raise AssertionError(f"{label}/rank{rank}/output_weight presence mismatch")
            if actual_weight is not None:
                assert_bytes_equal(actual_weight.cpu(), reference_weight.cpu(), f"{label}/rank{rank}/output_weight")
        assert_bytes_equal(pending[replay_index][1].cpu(), pending[-1][1].cpu(), f"replay/rank{rank}/output")
        assert_bytes_equal(pending[replay_index][6].cpu(), pending[-1][6].cpu(), f"replay/rank{rank}/plan")
        for label, actual_layout, sliced_layout in layout_witnesses:
            assert_bytes_equal(actual_layout.cpu(), sliced_layout.cpu(), f"layout/{label}/rank{rank}")
        if dispatch_reuse_probe is not None:
            snapshot_a1, snapshot_a2, _snapshot_b = dispatch_reuse_probe
            assert_bytes_equal(snapshot_a1.cpu(), snapshot_a2.cpu(), f"dispatch_reuse/rank{rank}")
    except Exception as exc:  # noqa: BLE001 - reduce all rank failures below
        failure = f"rank {rank}: {type(exc).__name__}: {exc}"
    failures = [None for _ in range(world)]
    dist.all_gather_object(failures, failure)
    if any(failures):
        raise AssertionError("chunked EP bitwise stress failed: " + " | ".join(item for item in failures if item))
    if rank == 0:
        print(
            f"serial chunked EP passed: world={world}, "
            f"local_world_size={local_world_size}, workloads={len(workloads)}",
            flush=True,
        )

    oracle_ep.finalize()
    range_ep.finalize()
    planner.finalize()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
