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
"""Kernel-trace benchmark for serial fixed-buffer EP.

This benchmark compares one prepared chunk-plan step with standard EP on the
equivalent sliced input.  Plan construction, batched layout construction,
active communication, and a zero-tail step have separate NVTX ranges so Nsys
kernel time is attributable without charging the model-owned GEMM output copy.

Example:
  torchrun --nproc-per-node=8 tests/ep/test_ep_chunked_perf.py \
      --local-world-size=4 --capacity-chunks=8 --interleave-active --profile
"""

from __future__ import annotations

import argparse
import datetime
import os
import statistics

import torch
import torch.distributed as dist

from chunked_ep_reference import fixed_capacity, make_routing, stable_routing
from flash_comm.ep import EPChunkPlanner, EPCommLayoutDesc, EPKernels
from flash_comm.ep.ep_kernels import _ep_inter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-token-per-rank", type=int, default=8192)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=384)
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--capacity-chunks", type=int, default=8)
    parser.add_argument("--num-sm", type=int, default=16)
    parser.add_argument("--local-world-size", type=int, default=0)
    parser.add_argument("--pattern", choices=("uniform", "hot_rank"), default="hot_rank")
    parser.add_argument("--with-weights", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--cooldown-cycles", type=int, default=2_000_000)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--interleave-active", action="store_true")
    parser.add_argument("--min-stage-roof-ratio", type=float, default=0.8)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    return parser.parse_args()


def nvtx_call(label: str, function):
    torch.cuda.nvtx.range_push(label)
    try:
        return function()
    finally:
        torch.cuda.nvtx.range_pop()


def measure(label: str, function, *, warmup: int, iters: int, cooldown_cycles: int, prepare=None):
    keepalive = []
    for _ in range(warmup):
        if prepare is not None:
            prepare()
        keepalive.append(function())
    torch.cuda.synchronize()
    keepalive.clear()
    dist.barrier()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for iteration in range(iters):
        if cooldown_cycles > 0:
            torch.cuda._sleep(cooldown_cycles)
        if prepare is not None:
            prepare()
        starts[iteration].record()
        keepalive.append(nvtx_call(f"ep_chunk/{label}/profile_iteration_{iteration}", function))
        ends[iteration].record()
    ends[-1].synchronize()
    elapsed_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    keepalive.clear()
    return elapsed_us


def report_timings(label: str, local_times: list[float], rank: int, world: int):
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_times)
    if rank != 0:
        return
    critical = [max(gathered[peer][iteration] for peer in range(world)) for iteration in range(len(local_times))]
    rank_medians = [statistics.median(values) for values in gathered]
    print(
        f"EVENT {label}: critical_us min={min(critical):.2f} "
        f"median={statistics.median(critical):.2f} max={max(critical):.2f}; "
        f"rank_median_us min={min(rank_medians):.2f} "
        f"max={max(rank_medians):.2f}",
        flush=True,
    )


def measure_interleaved(pair, *, warmup: int, iters: int, cooldown_cycles: int):
    keepalive = []
    for _ in range(warmup):
        for _label, function, prepare in pair:
            if prepare is not None:
                prepare()
            keepalive.append(function())
    torch.cuda.synchronize()
    keepalive.clear()
    dist.barrier()

    starts = {
        label: [torch.cuda.Event(enable_timing=True)
                for _ in range(iters)]
        for label, _function, _prepare in pair
    }
    ends = {label: [torch.cuda.Event(enable_timing=True) for _ in range(iters)] for label, _function, _prepare in pair}
    for iteration in range(iters):
        ordered = pair if iteration % 2 == 0 else tuple(reversed(pair))
        for label, function, prepare in ordered:
            if cooldown_cycles > 0:
                torch.cuda._sleep(cooldown_cycles)
            if prepare is not None:
                prepare()
            starts[label][iteration].record()
            keepalive.append(nvtx_call(f"ep_chunk/{label}/profile_iteration_{iteration}", function))
            ends[label][iteration].record()
    for label, _function, _prepare in pair:
        ends[label][-1].synchronize()
    keepalive.clear()
    return {
        label: [start.elapsed_time(end) * 1000.0
                for start, end in zip(starts[label], ends[label])]
        for label, _function, _prepare in pair
    }


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
    if args.num_experts % world:
        raise ValueError("num-experts must be divisible by world size")
    if args.capacity_chunks <= 0:
        raise ValueError("capacity-chunks must be positive")

    experts_per_rank = args.num_experts // world
    capacity = fixed_capacity(world, args.chunk_size, args.topk, experts_per_rank, args.expert_alignment,
                              args.capacity_chunks)
    planner = EPChunkPlanner(
        max_num_tokens=args.max_token_per_rank,
        chunk_size=args.chunk_size,
        num_experts=args.num_experts,
        group=group,
        local_world_size=local_world_size,
    )
    range_ep = EPKernels(
        args.max_token_per_rank,
        args.hidden,
        args.topk,
        args.num_experts,
        local_world_size,
        group,
        num_sm=args.num_sm,
        expert_alignment=args.expert_alignment,
        num_worst_tokens=capacity,
    )
    standard_ep = EPKernels(
        args.max_token_per_rank,
        args.hidden,
        args.topk,
        args.num_experts,
        local_world_size,
        group,
        num_sm=args.num_sm,
        expert_alignment=args.expert_alignment,
        num_worst_tokens=capacity,
    )

    routing_cpu = make_routing(args.pattern, rank, world, args.max_token_per_rank, args.topk, args.num_experts, 71137)
    routing = routing_cpu.pin_memory().cuda(non_blocking=True)
    generator = torch.Generator(device="cuda").manual_seed(9127 + rank)
    x = torch.randn((args.max_token_per_rank, args.hidden), dtype=torch.bfloat16, device="cuda", generator=generator)
    weights = None
    if args.with_weights:
        weights = torch.randn((args.max_token_per_rank, args.topk), dtype=torch.float32, device="cuda",
                              generator=generator)

    offsets, _ = stable_routing(range_ep, routing)
    plan = planner.build(routing, recv_capacity_tokens=capacity, expert_alignment=args.expert_alignment)
    layouts = range_ep.prepare_chunk_layouts(routing, offsets, plan)
    torch.cuda.synchronize()
    logical_ranges = plan.logical_token_ranges.cpu().tolist()
    active_steps = [index for index, (begin, end) in enumerate(logical_ranges) if end > begin]
    zero_steps = [index for index, (begin, end) in enumerate(logical_ranges) if end <= begin]
    if not active_steps:
        raise AssertionError("profile routing produced no active chunk step")
    active_step = active_steps[0]
    begin, end = logical_ranges[active_step]
    end = min(end, args.max_token_per_rank)
    x_slice = x[begin:end]
    routing_slice = routing[begin:end]
    weights_slice = weights[begin:end] if weights is not None else None

    standard_offsets, standard_counts = stable_routing(standard_ep, routing_slice)
    standard_layout = EPCommLayoutDesc(
        token_within_expert_offset=standard_offsets,
        expert_counts=standard_counts,
    )

    def standard_forward():
        dispatch, dispatch_weight, layout = standard_ep.dispatch(x_slice, routing_slice, weights_slice, standard_layout)
        return standard_ep.dispatch_postprocess(dispatch, dispatch_weight, layout)

    def range_forward():
        dispatch, dispatch_weight, layout = range_ep.dispatch(x, routing, weights, layouts[active_step])
        return range_ep.dispatch_postprocess(dispatch, dispatch_weight, layout)

    stage_variants = ()
    if range_ep.is_internode:
        active_layout = layouts[active_step]
        l2_flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        stage_views = range_ep.ep_context.rdma_rail_send_slot_views(end - begin, args.hidden, args.topk,
                                                                    max_slot_num_token=args.max_token_per_rank)

        def stage_copy():
            return _ep_inter.stage_dispatch_internode_range(
                x,
                routing,
                active_layout.token_topk_send_mask,
                active_layout.token_dst_scatter_indices,
                weights,
                range_ep.ep_context.rdma_rail_send_buf,
                range_ep.ep_context.rdma_rail_send_win_handle,
                args.max_token_per_rank,
                active_layout.logical_token_range,
            )

        def stage_d2d_roof():
            return stage_views["x"].copy_(x_slice)

        def flush_l2():
            l2_flush.zero_()

        stage_variants = (
            ("stage_copy", stage_copy, flush_l2),
            ("stage_d2d_roof", stage_d2d_roof, flush_l2),
        )

    standard_saved = standard_forward()
    range_saved = range_forward()
    torch.cuda.synchronize()
    standard_dispatch = standard_saved[0].clone()
    standard_dispatch_weight = (standard_saved[1].clone() if standard_saved[1] is not None else None)
    range_dispatch = range_saved[0].clone()
    range_dispatch_weight = (range_saved[1].clone() if range_saved[1] is not None else None)
    torch.cuda.synchronize()

    standard_combine_input = standard_ep.get_combine_buffer(standard_dispatch.shape[0])
    range_combine_input = range_ep.get_combine_buffer(range_dispatch.shape[0])
    standard_output_weight = (torch.empty_like(weights_slice) if weights_slice is not None else None)
    range_output = torch.empty_like(x)
    range_output_weight = torch.empty_like(weights) if weights is not None else None

    def prepare_standard_backward():
        standard_combine_input.copy_(standard_dispatch)

    def standard_backward():
        combine_input, combine_weight = standard_ep.combine_preprocess(standard_combine_input, standard_layout,
                                                                       standard_dispatch_weight, zero_copy=True)
        result = standard_ep.combine(combine_input, standard_layout, combine_weight)
        if standard_output_weight is not None and result[1] is None:
            raise AssertionError("standard weight output unexpectedly absent")
        return result

    def prepare_range_backward():
        range_combine_input.copy_(range_dispatch)

    def range_backward():
        layout = layouts[active_step]
        combine_input, combine_weight = range_ep.combine_preprocess(range_combine_input, layout, range_dispatch_weight,
                                                                    zero_copy=True)
        return range_ep.combine(combine_input, layout, combine_weight, output=range_output,
                                output_weight=range_output_weight)

    zero_step = zero_steps[0] if zero_steps else None

    def zero_forward():
        layout = layouts[zero_step]
        dispatch, dispatch_weight, layout = range_ep.dispatch(x, routing, weights, layout)
        return range_ep.dispatch_postprocess(dispatch, dispatch_weight, layout)

    def zero_backward():
        layout = layouts[zero_step]
        combine_input, combine_weight = range_ep.combine_preprocess(range_combine_input, layout, range_dispatch_weight,
                                                                    zero_copy=True)
        return range_ep.combine(combine_input, layout, combine_weight, output=range_output,
                                output_weight=range_output_weight)

    if rank == 0:
        print(
            f"PROFILE_CONFIG world={world} local_world_size={local_world_size} "
            f"M={args.max_token_per_rank} H={args.hidden} K={args.topk} "
            f"E={args.num_experts} chunk={args.chunk_size} "
            f"capacity_chunks={args.capacity_chunks} capacity={capacity} "
            f"pattern={args.pattern} active_steps={active_steps} "
            f"zero_steps={zero_steps} selected_range=[{begin},{end}) "
            f"stage_x_bytes={(end - begin) * args.hidden * 2}",
            flush=True,
        )

    if args.profile:
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStart()
        dist.barrier()

    setup_variants = (
        ("plan_build",
         lambda: planner.build(routing, recv_capacity_tokens=capacity, expert_alignment=args.expert_alignment), None),
        ("layout_build", lambda: range_ep.prepare_chunk_layouts(routing, offsets, plan), None),
    ) + stage_variants
    active_variants = (
        ("standard_forward", standard_forward, None),
        ("range_forward", range_forward, None),
        ("standard_backward", standard_backward, prepare_standard_backward),
        ("range_backward", range_backward, prepare_range_backward),
    )
    setup_timings = {}
    for label, function, prepare in setup_variants:
        timings = measure(label, function, warmup=args.warmup, iters=args.iters, cooldown_cycles=args.cooldown_cycles,
                          prepare=prepare)
        setup_timings[label] = timings
        report_timings(label, timings, rank, world)
    if stage_variants:
        local_ratio = (statistics.median(setup_timings["stage_d2d_roof"]) /
                       statistics.median(setup_timings["stage_copy"]))
        ratio_min = torch.tensor(local_ratio, dtype=torch.float64, device="cuda")
        dist.all_reduce(ratio_min, op=dist.ReduceOp.MIN, group=group)
        if rank == 0:
            print(
                f"STAGE_HBM_ROOF_RATIO min={ratio_min.item():.4f} "
                f"required={args.min_stage_roof_ratio:.4f}",
                flush=True,
            )
        if ratio_min.item() < args.min_stage_roof_ratio:
            raise AssertionError(f"internode stage copy reached only {ratio_min.item():.4f} "
                                 f"of the cold D2D HBM roof; require "
                                 f"{args.min_stage_roof_ratio:.4f}")
    if args.interleave_active:
        forward_pair = active_variants[:2]
        backward_pair = active_variants[2:]
        for pair in (forward_pair, backward_pair):
            pair_timings = measure_interleaved(pair, warmup=args.warmup, iters=args.iters,
                                               cooldown_cycles=args.cooldown_cycles)
            for label, _function, _prepare in pair:
                report_timings(label, pair_timings[label], rank, world)
    else:
        for label, function, prepare in active_variants:
            timings = measure(label, function, warmup=args.warmup, iters=args.iters,
                              cooldown_cycles=args.cooldown_cycles, prepare=prepare)
            report_timings(label, timings, rank, world)
    if zero_step is not None:
        timings = measure("zero_forward", zero_forward, warmup=args.warmup, iters=args.iters,
                          cooldown_cycles=args.cooldown_cycles)
        report_timings("zero_forward", timings, rank, world)
        timings = measure("zero_backward", zero_backward, warmup=args.warmup, iters=args.iters,
                          cooldown_cycles=args.cooldown_cycles)
        report_timings("zero_backward", timings, rank, world)

    if args.profile:
        dist.barrier()
        if rank == 0:
            torch.cuda.cudart().cudaProfilerStop()
        dist.barrier()

    standard_ep.finalize()
    range_ep.finalize()
    planner.finalize()
    dist.destroy_process_group(group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
