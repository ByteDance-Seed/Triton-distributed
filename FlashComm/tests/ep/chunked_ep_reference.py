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
"""Test-only references for serial fixed-buffer range EP."""

from __future__ import annotations

import torch

from flash_comm.ep import (
    EPChunkPlanner,
    EPCommLayoutDesc,
    EPKernels,
)


def make_routing(pattern: str, rank: int, world: int, num_tokens: int, topk: int, num_experts: int,
                 seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed + rank * 104729)
    experts_per_rank = num_experts // world
    if pattern == "uniform":
        value = torch.randint(num_experts, (num_tokens, topk), dtype=torch.int32, generator=generator)
    elif pattern == "arbitrary":
        # Every route is independent and spans the complete public routing
        # contract: real experts [0, E) plus the drop sentinel E.
        value = torch.randint(num_experts + 1, (num_tokens, topk), dtype=torch.int32, generator=generator)
    elif pattern == "hot_rank":
        target_rank = seed % world
        value = target_rank * experts_per_rank + torch.randint(experts_per_rank, (num_tokens, topk), dtype=torch.int32,
                                                               generator=generator)
    elif pattern == "global_hot_rank":
        target_rank = seed % world
        columns = torch.arange(topk, dtype=torch.int64).remainder(experts_per_rank)
        value = (target_rank * experts_per_rank + columns).view(1, topk).expand(num_tokens,
                                                                                topk).to(torch.int32).clone()
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
    elif pattern == "all_to_self":
        value = rank * experts_per_rank + torch.randint(experts_per_rank,
                                                        (num_tokens, topk), dtype=torch.int32, generator=generator)
    elif pattern == "all_to_next":
        target_rank = (rank + 1) % world
        value = target_rank * experts_per_rank + torch.randint(experts_per_rank, (num_tokens, topk), dtype=torch.int32,
                                                               generator=generator)
    elif pattern == "round_robin":
        routes = torch.arange(num_tokens * topk, dtype=torch.int64)
        value = ((routes + rank * 17) % num_experts).reshape(num_tokens, topk).to(torch.int32)
    elif pattern == "drop_heavy":
        value = torch.randint(num_experts, (num_tokens, topk), dtype=torch.int32, generator=generator)
        value[torch.rand((num_tokens, topk), generator=generator) < 0.7] = num_experts
    elif pattern == "all_drop":
        value = torch.full((num_tokens, topk), num_experts, dtype=torch.int32)
    else:
        raise ValueError(pattern)
    return value


def fixed_capacity(world: int, chunk_size: int, topk: int, experts_per_rank: int, alignment: int,
                   capacity_chunks: int = 1) -> int:
    return (world * chunk_size * topk * capacity_chunks + experts_per_rank * (alignment - 1))


def one_shot_capacity(world: int, max_tokens: int, topk: int, experts_per_rank: int, alignment: int) -> int:
    return (world * max_tokens * topk + experts_per_rank * (alignment - 1))


def stable_routing(ep: EPKernels, routing: torch.Tensor):
    if routing.shape[0] == 0:
        return (
            torch.empty_like(routing),
            torch.zeros((ep.ep_context.config.num_experts + 1, ), dtype=torch.int32, device=routing.device),
        )
    return ep.compute_stable_local_token_within_expert_offset_and_expert_counts(routing)


def standard_identity(ep: EPKernels, x: torch.Tensor, routing: torch.Tensor, weights: torch.Tensor | None = None):
    offsets, counts = stable_routing(ep, routing)
    layout = EPCommLayoutDesc(
        token_within_expert_offset=offsets,
        expert_counts=counts,
    )
    dispatch, dispatch_weights, layout = ep.dispatch(x, routing, weights, layout)
    dispatch, dispatch_weights, layout = ep.dispatch_postprocess(dispatch, dispatch_weights, layout)
    combine_input, combine_weights = ep.combine_preprocess(dispatch, layout, dispatch_weights)
    return ep.combine(combine_input, layout, combine_weights)


def execute_range_identity(ep: EPKernels, x: torch.Tensor, routing: torch.Tensor, layouts,
                           weights: torch.Tensor | None = None):
    output = torch.empty_like(x)
    output_weight = torch.empty_like(weights) if weights is not None else None
    for layout in layouts:
        dispatch, dispatch_weights, layout = ep.dispatch(x, routing, weights, layout)
        dispatch, dispatch_weights, layout = ep.dispatch_postprocess(dispatch, dispatch_weights, layout)
        combine_input, combine_weights = ep.combine_preprocess(dispatch, layout, dispatch_weights)
        ep.combine(combine_input, layout, combine_weights, output=output, output_weight=output_weight)
    return output, output_weight


def range_identity(ep: EPKernels, planner: EPChunkPlanner, x: torch.Tensor, routing: torch.Tensor, capacity: int,
                   expert_alignment: int, weights: torch.Tensor | None = None):
    offsets, _counts = stable_routing(ep, routing)
    plan = planner.build(
        routing,
        recv_capacity_tokens=capacity,
        expert_alignment=expert_alignment,
    )
    layouts = ep.prepare_chunk_layouts(routing, offsets, plan)
    output, output_weight = execute_range_identity(ep, x, routing, layouts, weights)
    return output, output_weight, plan, layouts


def compact_layout_witnesses(ep: EPKernels, x: torch.Tensor, routing: torch.Tensor, layouts):
    """Compare every prepared layout with its sliced standard counterpart."""
    witnesses = []
    for step, layout in enumerate(layouts):
        begin, end = (int(value) for value in layout.logical_token_range.cpu())
        begin = min(begin, routing.shape[0])
        end = min(end, routing.shape[0])
        if end <= begin:
            witnesses.append((
                f"step{step}/zero_expert_count",
                layout.recv_expert_counts.clone(),
                torch.zeros_like(layout.recv_expert_counts),
            ))
            continue
        sliced_routing = routing[begin:end]
        offsets, counts = stable_routing(ep, sliced_routing)
        standard = EPCommLayoutDesc(
            token_within_expert_offset=offsets,
            expert_counts=counts,
        )
        ep.dispatch(x[begin:end], sliced_routing, None, standard)
        witnesses.extend((
            (f"step{step}/token_dst", layout.token_dst_scatter_indices[:end - begin].clone(),
             standard.token_dst_scatter_indices.clone()),
            (f"step{step}/send_mask", layout.token_topk_send_mask[:end - begin].clone(),
             standard.token_topk_send_mask.clone()),
            (f"step{step}/recv_base", layout.recv_base_offset.clone(), standard.recv_base_offset.clone()),
            (f"step{step}/recv_count", layout.recv_token_count.clone(), standard.recv_token_count.clone()),
            (f"step{step}/aligned_count", layout.recv_aligned_token_count.clone(),
             (standard.recv_aligned_token_count
              if standard.recv_aligned_token_count is not None else standard.recv_token_count).clone()),
            (f"step{step}/expert_count", layout.recv_expert_counts.clone(), standard.recv_expert_counts.clone()),
        ))
        if layout.num_tokens_per_rank is not None:
            witnesses.append((
                f"step{step}/num_tokens_per_rank",
                layout.num_tokens_per_rank.clone(),
                standard.num_tokens_per_rank.clone(),
            ))
    return witnesses


def assert_bytes_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    actual_bytes = actual.contiguous().view(torch.uint8)
    expected_bytes = expected.contiguous().view(torch.uint8)
    if torch.equal(actual_bytes, expected_bytes):
        return
    mismatch = torch.nonzero(actual_bytes != expected_bytes, as_tuple=False).flatten()
    first = int(mismatch[0]) if mismatch.numel() else -1
    raise AssertionError(f"{label}: first byte mismatch at {first}")
