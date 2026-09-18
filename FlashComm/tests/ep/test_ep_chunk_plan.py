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
"""Deferred bitwise stress and trace-based performance for EPChunkPlanner.

Examples:
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunk_plan.py
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunk_plan.py --local-world-size=4
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunk_plan.py --local-world-size=2 --profile
"""

from __future__ import annotations

import argparse
import datetime
import os

import torch
import torch.distributed as dist

from flash_comm.ep import EPChunkPlanner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-token-per-rank", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--capacity-chunks", type=int, default=1)
    parser.add_argument("--stress-rounds", type=int, default=24)
    parser.add_argument("--hot-expert-rounds", type=int, default=0)
    parser.add_argument("--hot-rank-rounds", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--local-world-size", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--launch-num-sm", type=int, default=0,
                        help="SM budget for chunk-plan cooperative launch; 0 keeps the current full-device behavior")
    parser.add_argument("--kernel-limit-us", type=float, default=120.0)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def make_routing(pattern: str, rank: int, world: int, num_tokens: int, topk: int, num_experts: int,
                 seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + rank * 104729)
    experts_per_rank = num_experts // world
    if pattern == "uniform":
        value = torch.randint(num_experts, (num_tokens, topk), dtype=torch.int32, generator=generator)
    elif pattern == "arbitrary":
        value = torch.randint(num_experts + 1, (num_tokens, topk), dtype=torch.int32, generator=generator)
    elif pattern == "hot_rank":
        target = (rank * 7 + seed) % world
        value = target * experts_per_rank + torch.randint(experts_per_rank,
                                                          (num_tokens, topk), dtype=torch.int32, generator=generator)
    elif pattern == "global_hot_rank":
        target = seed % world
        columns = torch.arange(topk, dtype=torch.int64).remainder(experts_per_rank)
        value = (target * experts_per_rank + columns).view(1, topk).expand(num_tokens, topk).to(torch.int32).clone()
    elif pattern == "one_expert":
        value = torch.full((num_tokens, topk), seed % num_experts, dtype=torch.int32)
    elif pattern == "one_expert_with_drop":
        value = torch.full((num_tokens, topk), seed % num_experts, dtype=torch.int32)
        routes = torch.arange(num_tokens * topk, dtype=torch.int64).reshape(num_tokens, topk)
        value[(routes + rank) % 4 == 0] = num_experts
    elif pattern == "rotating_hot_expert":
        hot_expert = seed % num_experts
        value = torch.empty((num_tokens, topk), dtype=torch.int32)
        if topk:
            value[:, 0] = hot_expert
        if topk > 1:
            other = (torch.arange(num_tokens, dtype=torch.int64).unsqueeze(1) * (topk - 1) +
                     torch.arange(topk - 1, dtype=torch.int64).unsqueeze(0)).remainder(num_experts - 1)
            other += other >= hot_expert
            value[:, 1:] = other.to(torch.int32)
    elif pattern == "duplicate_expert":
        expert = torch.randint(num_experts, (num_tokens, 1), dtype=torch.int32, generator=generator)
        value = expert.expand(num_tokens, topk).clone()
    elif pattern == "round_robin":
        route = torch.arange(num_tokens * topk, dtype=torch.int64)
        value = ((route + rank * 17 + seed) % num_experts).reshape(num_tokens, topk).to(torch.int32)
    elif pattern == "drop_heavy":
        value = torch.randint(num_experts, (num_tokens, topk), dtype=torch.int32, generator=generator)
        value[torch.rand((num_tokens, topk), generator=generator) < 0.7] = num_experts
    elif pattern == "all_drop":
        value = torch.full((num_tokens, topk), num_experts, dtype=torch.int32)
    else:
        raise ValueError(pattern)
    return value


def local_prefix(routing: torch.Tensor, max_tokens: int, chunk_size: int, num_experts: int) -> torch.Tensor:
    q = (max_tokens + chunk_size - 1) // chunk_size
    counts = torch.zeros((q, num_experts + 1), dtype=torch.int64)
    for chunk in range(q):
        begin = chunk * chunk_size
        end = min(begin + chunk_size, routing.shape[0])
        if begin < end:
            counts[chunk] = torch.bincount(routing[begin:end].reshape(-1).to(torch.int64), minlength=num_experts + 1)
    return torch.cat((torch.zeros((1, num_experts + 1), dtype=torch.int64), counts.cumsum(dim=0)), dim=0)


def reference_schedule(global_prefix: torch.Tensor, capacity: int, world: int, alignment: int):
    if int(global_prefix[-1].sum()) == 0:
        return []
    q = global_prefix.shape[0] - 1
    num_experts = global_prefix.shape[1] - 1
    experts_per_rank = num_experts // world
    begin = 0
    schedule = []
    while begin < q:
        accepted = begin
        for end in range(begin + 1, q + 1):
            counts = global_prefix[end, :num_experts] - global_prefix[begin, :num_experts]
            fits = True
            for target in range(world):
                local = counts[target * experts_per_rank:(target + 1) * experts_per_rank]
                aligned = ((local + alignment - 1) // alignment) * alignment
                fits &= int(aligned.sum()) <= capacity
            if not fits:
                break
            accepted = end
        if accepted == begin:
            raise AssertionError("one logical chunk exceeds the configured capacity")
        schedule.append((begin, accepted - begin))
        begin = accepted
    return schedule


def expected_plan(global_prefix: torch.Tensor, *, capacity: int, world: int, alignment: int, chunk_size: int,
                  max_tokens: int):
    q = global_prefix.shape[0] - 1
    schedule = reference_schedule(global_prefix, capacity, world, alignment)
    logical = torch.zeros((q, 2), dtype=torch.int32)
    for step, (begin, count) in enumerate(schedule):
        end = begin + count
        logical[step, 0] = min(begin * chunk_size, max_tokens)
        logical[step, 1] = min(end * chunk_size, max_tokens)
    return logical, schedule


def assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor, label: str):
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    if torch.equal(actual_bytes, expected_bytes):
        return
    mismatch = torch.nonzero(actual_bytes != expected_bytes, as_tuple=False).flatten()
    first = int(mismatch[0]) if mismatch.numel() else -1
    raise AssertionError(f"{label}: first byte mismatch at {first}")


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
    capacity = (world * args.chunk_size * args.topk * args.capacity_chunks + experts_per_rank *
                (args.expert_alignment - 1))
    planner = EPChunkPlanner(
        max_num_tokens=args.max_token_per_rank,
        chunk_size=args.chunk_size,
        num_experts=num_experts,
        group=group,
        local_world_size=local_world_size,
    )

    patterns = (
        "arbitrary",
        "uniform",
        "hot_rank",
        "one_expert",
        "duplicate_expert",
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
        max(0, args.max_token_per_rank - 1),
        args.max_token_per_rank,
    )
    workloads = []
    for index, size in enumerate(edge_sizes):
        local_size = max(0, size - (rank * 17 + index * 13) % max(1, min(size, 97)))
        workloads.append((f"edge_{index}", patterns[index % len(patterns)], local_size, 7000 + index))
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
        ("rank_boundary_matrix", "arbitrary", rank_boundaries[rank % len(rank_boundaries)], 7988),
        ("alternating_empty_full", "arbitrary", args.max_token_per_rank if rank % 2 else 0, 7989),
        ("single_nonempty_rank", "arbitrary", args.max_token_per_rank if rank == world - 1 else 0, 7990),
        ("rank_staircase", "arbitrary", args.max_token_per_rank * rank // max(1, world - 1), 7991),
    ))
    workloads.append(("all_drop_nonempty", "all_drop", rank_boundaries[(rank + 3) % len(rank_boundaries)], 7998))
    workloads.append(("forced_multichunk", "one_expert_with_drop", args.max_token_per_rank, 7999))
    replay_index = len(workloads) - 1
    size_generator = torch.Generator(device="cpu").manual_seed(881 + rank)
    for index in range(args.stress_rounds):
        size = int(torch.randint(0, args.max_token_per_rank + 1, (), generator=size_generator))
        workloads.append((f"random_{index}", patterns[index % len(patterns)], size, 8000 + index))
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
    local_reference_prefixes = []
    replay = None
    prepared = []
    # Finish every input transfer before queueing the synchronization-free batch.
    for index, (label, pattern, size, seed) in enumerate(workloads):
        if index == len(workloads) - 1:
            routing_cpu, routing = replay
        else:
            routing_cpu = make_routing(pattern, rank, world, size, args.topk, num_experts, seed)
            routing = routing_cpu.pin_memory().cuda(non_blocking=True)
            if index == replay_index:
                replay = (routing_cpu, routing)
        reference_prefix = local_prefix(routing_cpu, args.max_token_per_rank, args.chunk_size, num_experts)
        prepared.append((label, routing, reference_prefix))
    torch.cuda.synchronize()

    for index, (label, routing, reference_prefix) in enumerate(prepared):
        # Alternate the slowest rank.  Root-slowest rounds directly stress the
        # ready/publication versus workspace-retirement boundary: non-root
        # ranks can finish call N and enter call N+1 while rank 0 is still
        # completing its caller-owned output snapshot.
        if index % 2 == 0:
            delay_factor = world if rank == 0 else 1
        else:
            delay_factor = rank + 1
        torch.cuda._sleep(delay_factor * (index % 7 + 1) * 4096)
        plan = planner.build(
            routing,
            recv_capacity_tokens=capacity,
            expert_alignment=args.expert_alignment,
            launch_num_sms=args.launch_num_sm or None,
        )
        local_reference_prefixes.append(reference_prefix)
        pending.append((
            label,
            plan.logical_token_ranges.clone(),
            plan.rank_chunk_prefix.clone(),
        ))

    # One deferred fence for the whole stress batch.  No round receives a
    # rescue synchronization, and every returned plan was snapshotted first.
    torch.cuda.synchronize()
    plans_cpu = [(logical.cpu(), rank_prefix.cpu()) for _label, logical, rank_prefix in pending]
    plan_batches = [None for _ in range(world)]
    dist.all_gather_object(plan_batches, plans_cpu)
    reference_batches = [None for _ in range(world)]
    dist.all_gather_object(reference_batches, local_reference_prefixes)
    local_sizes = [int(prefix[-1].sum().item() // args.topk) for prefix in local_reference_prefixes]
    size_matrix = [None for _ in range(world)]
    dist.all_gather_object(size_matrix, local_sizes)
    failure = ""
    saw_multichunk = False
    require_multichunk = planner.num_chunks > args.capacity_chunks
    try:
        for peer in range(1, world):
            for workload_index, ((root_logical, root_prefix),
                                 (peer_logical, peer_prefix)) in enumerate(zip(plan_batches[0], plan_batches[peer])):
                label = workloads[workload_index][0]
                assert_bytes_equal(peer_logical, root_logical, f"{label}/cross_rank_logical/rank{peer}")
                assert_bytes_equal(peer_prefix, root_prefix, f"{label}/cross_rank_prefix/rank{peer}")
        if not any(
                len({size_matrix[peer][workload]
                     for peer in range(world)}) > 1
                for workload in range(len(workloads))):
            raise AssertionError("stress suite did not exercise different token counts per rank")
        for workload_index, item in enumerate(pending):
            prefixes = [reference_batches[peer][workload_index] for peer in range(world)]
            label, logical, rank_prefix = item
            global_prefix = sum(prefixes)
            expected_logical, schedule = expected_plan(
                global_prefix,
                capacity=capacity,
                world=world,
                alignment=args.expert_alignment,
                chunk_size=args.chunk_size,
                max_tokens=args.max_token_per_rank,
            )
            assert_bytes_equal(logical.cpu(), expected_logical, f"{label}/logical/rank{rank}")
            assert_bytes_equal(rank_prefix.cpu(),
                               torch.stack(prefixes).to(torch.int32), f"{label}/rank_prefix/rank{rank}")
            if label == "forced_multichunk":
                saw_multichunk = len(schedule) > 1
        assert_bytes_equal(pending[replay_index][1].cpu(), pending[-1][1].cpu(), f"replay/logical/rank{rank}")
        assert_bytes_equal(pending[replay_index][2].cpu(), pending[-1][2].cpu(), f"replay/rank_prefix/rank{rank}")
        if require_multichunk and not saw_multichunk:
            raise AssertionError("forced_multichunk did not exercise multiple plan steps")
    except Exception as exc:  # noqa: BLE001 - reduce all rank failures below
        failure = f"rank {rank}: {type(exc).__name__}: {exc}"
    failures = [None for _ in range(world)]
    dist.all_gather_object(failures, failure)
    if any(failures):
        raise AssertionError("chunk-plan bitwise stress failed: " + " | ".join(item for item in failures if item))

    perf_routing = make_routing("hot_rank", rank, world, args.max_token_per_rank, args.topk, num_experts, 99173).cuda()
    for _ in range(args.warmup):
        planner.build(
            perf_routing,
            recv_capacity_tokens=capacity,
            expert_alignment=args.expert_alignment,
            launch_num_sms=args.launch_num_sm or None,
        )
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if args.profile:
        torch.cuda.nvtx.range_push("ep_chunk_plan_profile")
    start.record()
    for _ in range(args.iters):
        planner.build(
            perf_routing,
            recv_capacity_tokens=capacity,
            expert_alignment=args.expert_alignment,
            launch_num_sms=args.launch_num_sm or None,
        )
    end.record()
    if args.profile:
        torch.cuda.nvtx.range_pop()
    end.synchronize()
    device_average_us = start.elapsed_time(end) * 1000.0 / args.iters
    device_max = torch.tensor([device_average_us], dtype=torch.float64, device="cuda")
    dist.all_reduce(device_max, op=dist.ReduceOp.MAX, group=group)
    device_average_us = float(device_max.cpu()[0])
    if args.kernel_limit_us > 0 and device_average_us > args.kernel_limit_us:
        raise AssertionError(f"planner device time {device_average_us:.2f} us exceeds "
                             f"{args.kernel_limit_us:.2f} us")

    if rank == 0:
        print(
            f"EPChunkPlanner passed: world={world}, local_world_size={local_world_size}, "
            f"workloads={len(workloads)}, Q={planner.num_chunks}, "
            f"multi_chunk={'yes' if saw_multichunk else 'not-applicable'}",
            flush=True,
        )
        print(
            f"EPChunkPlanner device time: {device_average_us:.2f} us/build; "
            "use Nsight Systems on NVTX range ep_chunk_plan_profile for "
            "authoritative kernel time",
            flush=True,
        )

    planner.finalize()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
