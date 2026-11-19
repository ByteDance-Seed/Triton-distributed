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
"""Shared graph/pipeline helpers used by scheduling and transforms."""

from __future__ import annotations

import ast
from typing import Callable, Optional, Set

from little_kernel.core.memory_effects import Access
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import BindRegion, MemEvent, RegionEvent
from little_kernel.backends.nv.sm100.passes.dataflow.channels import consumer_roles as _consumer_roles, producer_roles as _producer_roles


def _ident(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_") or "anon"


def _pipe_name(producers: tuple[str, ...], consumers: tuple[str, ...], via: list[str]) -> str:
    src = "_and_".join(_ident(role) for role in producers)
    dst = "_and_".join(_ident(role) for role in consumers)
    return f"PIPE_{src}_to_{dst}_{'_'.join(_ident(v) for v in via)}"


def _pipeline_var_name(pipeline: dict) -> str:
    return _pipe_name(_producer_roles(pipeline), _consumer_roles(pipeline), pipeline["via"])


def _role_mem_events(graph: dict, role: str, via: set[str], pred) -> list[MemEvent]:
    return [e for e in graph["mem_events"] if e.role == role and e.mem in via and pred(e.access)]


def _pipeline_side_via(pipeline: dict, side: str) -> set[str]:
    return set(pipeline.get(f"{side}_via", pipeline["via"]))


def _pipeline_side_keys(pipeline: dict, side: str) -> set[int]:
    return set(pipeline.get(f"{side}_keys", (pipeline["key"], )))


def _region_bounds(
    region: BindRegion,
    via: set[str],
    pred: Callable[[Access], bool],
) -> tuple[int, int, BindRegion] | None:
    events = [item for item in region.events if item.event.mem in via and pred(item.event.access)]
    carriers = [item.carrier_id for item in events]
    if not carriers:
        return None
    return carriers[0], carriers[-1], region


def _pipeline_region_bounds(
    graph: dict,
    pipeline: dict,
    side: str,
    role: str,
    pred: Callable[[Access], bool],
) -> list[tuple[int, int, BindRegion]]:
    via = _pipeline_side_via(pipeline, side)
    keys = _pipeline_side_keys(pipeline, side)
    grouped: dict[tuple[tuple, str], tuple[int, int, list[RegionEvent], BindRegion]] = {}
    for region in graph["bind_regions"]:
        if region.key not in keys or region.role != role:
            continue
        region_bounds = _region_bounds(region, via, pred)
        if region_bounds is None:
            continue
        first, last, _ = region_bounds
        group_key = (region.scope_id, ast.dump(region.value, include_attributes=False))
        if group_key not in grouped:
            grouped[group_key] = (first, last, list(region.events), region)
            continue
        old_first, _old_last, events, old_region = grouped[group_key]
        events.extend(region.events)
        grouped[group_key] = (old_first, last, events, old_region)

    bounds: list[tuple[int, int, BindRegion]] = []
    for first, last, events, region in grouped.values():
        bounds.append((
            first,
            last,
            BindRegion(region.key, region.role, region.value, region.scope_id, tuple(events), region.loops),
        ))
    return bounds


def _pipeline_stages(graph: dict, pipeline: dict):
    rmw = graph.get("rmw")
    for group in getattr(rmw, "groups", ()):
        if pipeline["key"] in {group.forward_key, group.feedback_key}:
            return group.sync_depth

    stages = [graph["memories"][name].stages for name in pipeline["via"]]
    if any(stage != stages[0] for stage in stages):
        raise ValueError(f"{pipeline['name']}: mixed stage counts are not supported yet: {stages}")
    return stages[0]
