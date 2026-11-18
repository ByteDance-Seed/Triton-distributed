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
"""Analyze loop-carried read/modify/write channels.

RMW is not buffer overlay.  Overlay is "two logical buffers share physical
storage"; RMW is "one logical state is updated by multiple roles across loop
iterations".

The canonical FA4 case is:

    PV writes O  -> correction reads/writes O -> next PV reads/writes O

The first PV MMA must be write-only (`init`) so the cycle has an origin token.
Later PV MMAs are read/write (`accum`) and must wait for the correction feedback.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import combinations
from typing import Any

DATAFLOW_CTX_KEY = "__LK_DATAFLOW_GRAPHS__"


@dataclass(frozen=True)
class RmwGroup:
    buffer: str
    producer_role: str
    consumer_role: str
    forward_key: int
    feedback_key: int
    origin_role: str
    copies: int
    sync_depth: int


@dataclass(frozen=True)
class RmwAnalysis:
    groups: tuple[RmwGroup, ...]


@dataclass(frozen=True)
class _CycleChannel:
    key: int
    producer: str
    consumer: str
    mem: str


def _channel_cycles(graph: dict, ) -> list[tuple[_CycleChannel, _CycleChannel]]:
    channels: list[_CycleChannel] = []
    for channel in graph.get("channels", ()):
        producers = channel.get("producers", (channel["src"], ) if "src" in channel else ())
        consumers = channel.get("consumers", (channel["dst"], ) if "dst" in channel else ())
        if len(producers) != 1 or len(consumers) != 1:
            continue
        for mem in channel.get("via", ()):
            channels.append(_CycleChannel(channel["key"], producers[0], consumers[0], mem))

    out: list[tuple[_CycleChannel, _CycleChannel]] = []
    seen: set[tuple[int, int, str]] = set()
    for a, b in combinations(channels, 2):
        if a.mem != b.mem:
            continue
        if not (a.producer == b.consumer and a.consumer == b.producer):
            continue
        key = (min(a.key, b.key), max(a.key, b.key), a.mem)
        if key in seen:
            continue
        seen.add(key)
        out.append((a, b))
    return out


def _mem_events(graph: dict, role: str, mem: str) -> list[Any]:
    return [ev for ev in graph.get("mem_events", ()) if ev.role == role and ev.mem == mem]


def _has_events(graph: dict, roles: tuple[str, str], mem: str) -> bool:
    role_set = set(roles)
    return any(ev.role in role_set and ev.mem == mem for ev in graph.get("mem_events", ()))


def _first_access_is_write_only(graph: dict, role: str, mem: str) -> bool:
    events = sorted(_mem_events(graph, role, mem), key=lambda ev: ev.order)
    for ev in events:
        access = getattr(ev.access, "value", ev.access)
        if access in {"r", "rw"}:
            return False
        if access == "w":
            return True
    return False


def _origin_roles(graph: dict, roles: tuple[str, str], mem: str) -> tuple[str, ...]:
    return tuple(role for role in roles if _first_access_is_write_only(graph, role, mem))


def _copies(graph: dict, mem: str) -> int:
    obj = graph.get("memories", {}).get(mem)
    if obj is None:
        return 1
    return int(getattr(obj, "copies", None) or getattr(obj, "stages", 1) or 1)


def _sync_depth(graph: dict, mem: str, copies: int) -> int:
    obj = graph.get("memories", {}).get(mem)
    stages = int(getattr(obj, "stages", 1) or 1) if obj is not None else 1
    return max(1, min(stages, copies))


def _group_from_cycle(graph: dict, a: _CycleChannel, b: _CycleChannel) -> RmwGroup | None:
    roles = (a.producer, a.consumer)

    if not _has_events(graph, roles, a.mem):
        return None

    origins = _origin_roles(graph, roles, a.mem)
    if not origins:
        raise ValueError(
            f"RMW cycle {a.producer}->{a.consumer}->{a.producer} via "
            f"{a.mem!r} has no write-only init origin", )

    origin = origins[0]
    if a.producer == origin:
        forward, feedback = a, b
    else:
        forward, feedback = b, a
    copies = _copies(graph, a.mem)
    return RmwGroup(
        buffer=a.mem,
        producer_role=forward.producer,
        consumer_role=forward.consumer,
        forward_key=forward.key,
        feedback_key=feedback.key,
        origin_role=origin,
        copies=copies,
        sync_depth=_sync_depth(graph, a.mem, copies),
    )


def analyze_rmw_groups(graph: dict) -> RmwAnalysis:
    groups: list[RmwGroup] = []
    for a, b in _channel_cycles(graph):
        group = _group_from_cycle(graph, a, b)
        if group is not None:
            groups.append(group)

    analysis = RmwAnalysis(tuple(groups))
    graph["rmw"] = analysis
    return analysis


def rmw_analysis_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    """Run after dataflow analysis; records RMW groups in ``ctx``/graphs."""

    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY)
    if graphs is None:
        raise ValueError("RMW analysis requires dataflow analysis first")
    for graph in graphs.values():
        analyze_rmw_groups(graph)
    return tree
