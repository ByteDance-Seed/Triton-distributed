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
"""Analyze physical overlay between logical role-proposal buffers.

The dataflow graph keeps logical buffers separate.  This pass adds the missing
physical view: two logical buffers may occupy overlapping TMEM columns or SMEM
bytes.  When that happens, a later write must be ordered after the previous
logical value has been consumed, otherwise the new write clobbers live data.

This pass is analysis-only.  It records constraints in the graph for later
pipeline/scheduling passes; it does not rewrite the AST.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations
from typing import Any, Iterable

DATAFLOW_CTX_KEY = "__LK_DATAFLOW_GRAPHS__"
BUFFER_OVERLAY_CTX_KEY = "__LK_BUFFER_OVERLAYS__"
BUFFER_MAPPING_CTX_KEY = "__LK_BUFFER_OVERLAY_MAPPING__"
BUFFER_OVERLAY_STRICT_CTX_KEY = "__LK_BUFFER_OVERLAY_STRICT__"


@dataclass(frozen=True)
class MemoryMappingEntry:
    """Logical buffer placement in one physical allocation.

    Native units match fabricdsl: TMEM uses 32-bit columns, SMEM uses bytes.
    ``extent`` and ``stage_stride`` are per-stage native units.
    """

    logical: str
    physical: str
    space: str | None = None
    offset: int = 0
    extent: int | None = None
    stage_stride: int | None = None


@dataclass(frozen=True)
class Placement:
    logical: str
    physical: str
    space: str
    offset: int
    size: int
    stages: int
    stage_stride: int


@dataclass(frozen=True)
class OverlayPair:
    a: str
    b: str
    stage_overlaps: tuple["StageOverlap", ...] = ()


@dataclass(frozen=True)
class StageOverlap:
    a_stage: int
    b_stage: int
    begin: int
    end: int


class OverlayConstraintKind(StrEnum):
    PRODUCER_WAIT_RELEASE = "producer_wait_release"
    CONSUMER_RELEASE_BEFORE_WAIT = "consumer_release_before_wait"
    TRANSFORM_READ_BEFORE_WRITE = "transform_read_before_write"


@dataclass(frozen=True)
class OverlayConstraint:
    """Ordering needed because ``old_mem`` and ``new_mem`` overlap."""

    kind: OverlayConstraintKind
    old_mem: str
    new_mem: str
    release_roles: tuple[str, ...] = ()
    acquire_roles: tuple[str, ...] = ()
    role: str | None = None
    stage_overlaps: tuple[StageOverlap, ...] = ()
    note: str = ""


@dataclass(frozen=True)
class OverlayDiagnostic:
    severity: str
    message: str
    pair: tuple[str, str] | None = None
    role: str | None = None


@dataclass(frozen=True)
class BufferOverlayAnalysis:
    placements: dict[str, Placement]
    pairs: tuple[OverlayPair, ...]
    constraints: tuple[OverlayConstraint, ...]
    diagnostics: tuple[OverlayDiagnostic, ...]


def _space(mem: Any) -> str:
    cls = mem.__class__.__name__
    if cls == "TMem":
        return "tmem"
    if cls == "SMem":
        return "smem"
    return cls.lower()


def _dtype_size(dtype: Any) -> int:
    width = getattr(dtype, "width", None)
    if width is not None:
        return int(width)
    text = str(dtype).lower().split(".")[-1]
    return {
        "bfloat16": 2,
        "bf16": 2,
        "float16": 2,
        "f16": 2,
        "uint16": 2,
        "int16": 2,
        "float32": 4,
        "f32": 4,
        "uint32": 4,
        "int32": 4,
        "float64": 8,
        "f64": 8,
        "uint64": 8,
        "int64": 8,
        "uint8": 1,
        "int8": 1,
    }[text]


def _int_dim(dim: Any) -> int:
    try:
        return int(dim)
    except Exception as exc:
        raise ValueError(f"buffer overlay requires concrete memory shapes, got {dim!r}") from exc


def _shape(mem: Any) -> list[Any]:
    return list(mem.shape) if isinstance(mem.shape, (tuple, list)) else [mem.shape]


def _numel(shape: Iterable[Any]) -> int:
    out = 1
    for dim in shape:
        out *= _int_dim(dim)
    return out


def _natural_stage_size(mem: Any) -> int:
    shape = _shape(mem)
    if _space(mem) == "tmem":
        rows = _int_dim(shape[0]) if len(shape) >= 2 else 128
        cols = _int_dim(shape[-1]) if len(shape) >= 2 else 128
        row_groups = (rows + 127) // 128
        return row_groups * ((cols * _dtype_size(mem.dtype) + 3) // 4)
    return _numel(shape) * _dtype_size(mem.dtype)


def _mapping_entries(raw: Any) -> dict[str, MemoryMappingEntry]:
    if raw is None:
        return {}
    if hasattr(raw, "entries"):
        raw = raw.entries
    if isinstance(raw, dict):
        items = []
        for logical, value in raw.items():
            if isinstance(value, str):
                items.append(MemoryMappingEntry(logical=logical, physical=value))
            elif isinstance(value, dict):
                items.append(MemoryMappingEntry(logical=logical, **value))
            else:
                items.append(value)
        raw = items

    out: dict[str, MemoryMappingEntry] = {}
    for entry in raw:
        if isinstance(entry, MemoryMappingEntry):
            out[entry.logical] = entry
        elif isinstance(entry, dict):
            item = MemoryMappingEntry(**entry)
            out[item.logical] = item
        else:
            item = MemoryMappingEntry(
                logical=entry.logical,
                physical=entry.physical,
                space=getattr(entry, "space", None),
                offset=getattr(entry, "offset", 0),
                extent=getattr(entry, "extent", None),
                stage_stride=getattr(entry, "stage_stride", None),
            )
            out[item.logical] = item
    return out


def build_placements(graph: dict, mapping_plan: Any = None) -> dict[str, Placement]:
    entries = _mapping_entries(mapping_plan)
    placements: dict[str, Placement] = {}
    for name, mem in graph.get("memories", {}).items():
        entry = entries.get(name)
        space = str(entry.space).lower() if entry and entry.space is not None else _space(mem)
        if "." in space:
            space = space.rsplit(".", 1)[-1]
        natural = _natural_stage_size(mem)
        stage_stride = int(entry.stage_stride) if entry and entry.stage_stride is not None else natural
        size = min(int(entry.extent), stage_stride) if entry and entry.extent is not None else natural
        placements[name] = Placement(
            logical=name,
            physical=entry.physical if entry else name,
            space=space,
            offset=int(entry.offset) if entry else 0,
            size=size,
            stages=int(getattr(mem, "stages", 1) or 1),
            stage_stride=stage_stride,
        )
    return placements


def _intervals(p: Placement) -> tuple[tuple[int, int], ...]:
    if p.size <= 0:
        return ()
    return tuple(
        (p.offset + stage * p.stage_stride, p.offset + stage * p.stage_stride + p.size) for stage in range(p.stages))


def stage_overlaps(a: Placement, b: Placement) -> tuple[StageOverlap, ...]:
    if a.physical != b.physical or a.space != b.space:
        return ()
    out: list[StageOverlap] = []
    for a_stage, (alo, ahi) in enumerate(_intervals(a)):
        for b_stage, (blo, bhi) in enumerate(_intervals(b)):
            if alo < bhi and blo < ahi:
                out.append(StageOverlap(a_stage, b_stage, max(alo, blo), min(ahi, bhi)))
    return tuple(out)


def overlaps(a: Placement, b: Placement) -> bool:
    return bool(stage_overlaps(a, b))


def build_overlay_pairs(graph: dict, placements: dict[str, Placement]) -> tuple[OverlayPair, ...]:
    pairs: list[OverlayPair] = []
    for a, b in combinations(sorted(placements), 2):
        intervals = stage_overlaps(placements[a], placements[b])
        if not intervals:
            continue
        pairs.append(OverlayPair(a, b, intervals))
    return tuple(pairs)


def _is_read(access: Any) -> bool:
    return getattr(access, "value", access) in {"r", "rw"}


def _is_write(access: Any) -> bool:
    return getattr(access, "value", access) in {"w", "rw"}


def _role_orders(graph: dict) -> dict[str, dict[str, list[int | None]]]:
    """Map role/memory to first-read, last-read, first-write, last-write."""
    orders: dict[str, dict[str, list[int | None]]] = defaultdict(dict)
    for ev in graph.get("mem_events", ()):
        rec = orders[ev.role].setdefault(ev.mem, [None, None, None, None])
        if _is_read(ev.access):
            if rec[0] is None:
                rec[0] = ev.order
            rec[1] = ev.order
        if _is_write(ev.access):
            if rec[2] is None:
                rec[2] = ev.order
            rec[3] = ev.order
    return orders


def _channel_roles_by_mem(graph: dict) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    producers: dict[str, set[str]] = defaultdict(set)
    consumers: dict[str, set[str]] = defaultdict(set)
    for channel in graph.get("channels", ()):
        src = channel.get("producers", (channel.get("src"), ))
        dst = channel.get("consumers", (channel.get("dst"), ))
        for mem in channel.get("via", ()):
            producers[mem].update(role for role in src if role is not None)
            consumers[mem].update(role for role in dst if role is not None)
    return producers, consumers


def _role_graph(graph: dict) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for channel in graph.get("channels", ()):
        producers = channel.get("producers", (channel.get("src"), ))
        consumers = channel.get("consumers", (channel.get("dst"), ))
        for producer in producers:
            out[producer].update(consumers)
    return out


def _reachable(graph: dict[str, set[str]], src: str, dst: str) -> bool:
    pending = list(graph.get(src, ()))
    seen = {src}
    while pending:
        role = pending.pop()
        if role == dst:
            return True
        if role in seen:
            continue
        seen.add(role)
        pending.extend(graph.get(role, ()))
    return False


def _ordered(a: int | None, b: int | None) -> bool:
    return a is not None and b is not None and a < b


def _lifetime_precedes(
    old_mem: str,
    new_mem: str,
    *,
    orders: dict[str, dict[str, list[int | None]]],
    producers: dict[str, set[str]],
    consumers: dict[str, set[str]],
    role_graph: dict[str, set[str]],
) -> bool:
    release_roles = consumers.get(old_mem, ())
    acquire_roles = producers.get(new_mem, ())
    if not release_roles or not acquire_roles:
        return False
    for release in release_roles:
        for acquire in acquire_roles:
            if release == acquire:
                old_read = orders[release].get(old_mem, [None] * 4)[1]
                new_write = orders[acquire].get(new_mem, [None] * 4)[2]
                if not _ordered(old_read, new_write):
                    return False
            elif not _reachable(role_graph, release, acquire):
                return False
    return True


def _has_transform_chain(
    orders: dict[str, dict[str, list[int | None]]],
    producers: dict[str, set[str]],
    consumers: dict[str, set[str]],
    role: str,
    old_mem: str,
    new_mem: str,
) -> bool:
    """True for producer -> transform -> consumer overlay reuse.

    FA4 score/prob overlay has this shape:
    ``mma writes score``, ``softmax reads score then writes prob``,
    ``mma reads prob``.  The producer/consumer role sees a write-before-read on
    overlapping views, but the middle transform orders the physical reuse.
    """

    if role not in producers.get(old_mem, ()) or role not in consumers.get(new_mem, ()):
        return False
    for transform_role, mems in orders.items():
        old_read = mems.get(old_mem, [None] * 4)[1]
        new_write = mems.get(new_mem, [None] * 4)[2]
        if transform_role != role and _ordered(old_read, new_write):
            return True
    return False


def _paths_compatible(a: Any, b: Any) -> bool:
    a_arms = {node_id: arm for kind, node_id, arm in getattr(a, "control_path", ()) if kind == "if"}
    return all(
        a_arms.get(node_id, arm) == arm
        for node_id, arm in ((node_id, arm) for kind, node_id, arm in getattr(b, "control_path", ()) if kind == "if"))


def _event_interval(event: Any, placement: Placement) -> tuple[int, int]:
    offset = getattr(event, "logical_offset", None)
    extent = getattr(event, "logical_extent", None)
    if offset is None or extent is None:
        return placement.offset, placement.offset + placement.size
    return placement.offset + int(offset), placement.offset + int(offset) + int(extent)


def _transform_order(
    graph: dict,
    role: str,
    old_mem: str,
    new_mem: str,
    placements: dict[str, Placement],
) -> tuple[bool, bool]:
    reads = [
        event for event in graph.get("mem_events", ())
        if event.role == role and event.mem == old_mem and _is_read(event.access)
    ]
    writes = [
        event for event in graph.get("mem_events", ())
        if event.role == role and event.mem == new_mem and _is_write(event.access)
    ]
    initializers = [
        event for event in graph.get("mem_events", ())
        if event.role == role and event.mem == old_mem and _is_write(event.access) and not _is_read(event.access)
    ]
    overlaps_accessed_region = False
    for read in reads:
        rlo, rhi = _event_interval(read, placements[old_mem])
        for write in writes:
            if not _paths_compatible(read, write):
                continue
            wlo, whi = _event_interval(write, placements[new_mem])
            if rlo >= whi or wlo >= rhi:
                continue
            overlaps_accessed_region = True
            if read.order >= write.order:
                overlap_lo, overlap_hi = max(rlo, wlo), min(rhi, whi)
                if any(write.order < init.order < read.order and _paths_compatible(read, init) and _paths_compatible(
                        write, init) and _event_interval(init, placements[old_mem])[0] <= overlap_lo
                       and _event_interval(init, placements[old_mem])[1] >= overlap_hi for init in initializers):
                    continue
                return True, False
    return overlaps_accessed_region, True


def _add_constraint(out: list[OverlayConstraint], item: OverlayConstraint) -> None:
    if item not in out:
        out.append(item)


def _analyze_constraints(
    graph: dict,
    pairs: tuple[OverlayPair, ...],
    placements: dict[str, Placement],
) -> tuple[tuple[OverlayConstraint, ...], tuple[OverlayDiagnostic, ...]]:
    orders = _role_orders(graph)
    producers, consumers = _channel_roles_by_mem(graph)
    role_graph = _role_graph(graph)
    constraints: list[OverlayConstraint] = []
    diagnostics: list[OverlayDiagnostic] = []

    for pair in pairs:
        a, b = pair.a, pair.b
        pa, pb = placements[a], placements[b]
        exact_read_view = next(
            (view for view, storage in ((a, b), (b, a)) if placements[view].logical != placements[view].physical
             and placements[storage].logical == placements[storage].physical and (
                 placements[view].physical, placements[view].space, placements[view].offset, placements[view].size,
                 placements[view].stages, placements[view].stage_stride) == (
                     placements[storage].physical, placements[storage].space, placements[storage].offset,
                     placements[storage].size, placements[storage].stages, placements[storage].stage_stride)),
            None,
        )
        if exact_read_view is not None and not any(event.mem == exact_read_view
                                                   for event in graph.get("mem_events", ())):
            continue
        shifted = [overlap for overlap in pair.stage_overlaps if overlap.a_stage != overlap.b_stage]
        if pa.stages != pb.stages or pa.stage_stride != pb.stage_stride or shifted:
            diagnostics.append(
                OverlayDiagnostic(
                    "error",
                    f"{a!r} and {b!r} require unsupported shifted-stage overlay "
                    f"({pa.stages}/{pb.stages}, stride {pa.stage_stride}/{pb.stage_stride}, "
                    f"pairs {[(item.a_stage, item.b_stage) for item in pair.stage_overlaps]})",
                    (a, b),
                ))

        for role, mems in orders.items():
            ar, ar_last, aw, _aw_last = mems.get(a, [None] * 4)
            br, br_last, bw, _bw_last = mems.get(b, [None] * 4)

            # Same producer writes two overlapping logical buffers.  The later
            # write must wait until consumers of the earlier buffer release it.
            if _ordered(aw, bw):
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.PRODUCER_WAIT_RELEASE,
                        old_mem=a,
                        new_mem=b,
                        release_roles=tuple(sorted(consumers.get(a, ()))),
                        acquire_roles=(role, ),
                        stage_overlaps=pair.stage_overlaps,
                        note=f"{role} writes {a} before overlapping {b}",
                    ))
            elif _ordered(bw, aw):
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.PRODUCER_WAIT_RELEASE,
                        old_mem=b,
                        new_mem=a,
                        release_roles=tuple(sorted(consumers.get(b, ()))),
                        acquire_roles=(role, ),
                        stage_overlaps=pair.stage_overlaps,
                        note=f"{role} writes {b} before overlapping {a}",
                    ))

            # Same consumer reads two overlapping logical buffers.  It must
            # release the earlier one before waiting on the later one.
            if _ordered(ar_last, br):
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.CONSUMER_RELEASE_BEFORE_WAIT,
                        old_mem=a,
                        new_mem=b,
                        role=role,
                        stage_overlaps=pair.stage_overlaps,
                        note=f"{role} reads {a} before overlapping {b}",
                    ))
            elif _ordered(br_last, ar):
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.CONSUMER_RELEASE_BEFORE_WAIT,
                        old_mem=b,
                        new_mem=a,
                        role=role,
                        stage_overlaps=pair.stage_overlaps,
                        note=f"{role} reads {b} before overlapping {a}",
                    ))

            # In-place transform: read one view, then overwrite an overlapping
            # view.  Writing before reading is real corruption.
            a_to_b_overlap, a_to_b_safe = _transform_order(graph, role, a, b, placements)
            if a_to_b_overlap and a_to_b_safe:
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.TRANSFORM_READ_BEFORE_WRITE,
                        old_mem=a,
                        new_mem=b,
                        role=role,
                        stage_overlaps=pair.stage_overlaps,
                        note=f"{role} transforms {a} into overlapping {b}",
                    ))
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.PRODUCER_WAIT_RELEASE,
                        old_mem=b,
                        new_mem=a,
                        release_roles=tuple(sorted(consumers.get(b, ()))),
                        acquire_roles=tuple(sorted(producers.get(a, ()))),
                        stage_overlaps=pair.stage_overlaps,
                        note=f"reuse {a}/{b} physical slot after transform",
                    ))
            elif a_to_b_overlap and not _has_transform_chain(orders, producers, consumers, role, b, a):
                diagnostics.append(
                    OverlayDiagnostic(
                        "error",
                        f"{role!r} writes {b!r} before finishing overlapping reads from {a!r}",
                        (a, b),
                        role,
                    ))

            b_to_a_overlap, b_to_a_safe = _transform_order(graph, role, b, a, placements)
            if b_to_a_overlap and b_to_a_safe:
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.TRANSFORM_READ_BEFORE_WRITE,
                        old_mem=b,
                        new_mem=a,
                        role=role,
                        stage_overlaps=pair.stage_overlaps,
                        note=f"{role} transforms {b} into overlapping {a}",
                    ))
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.PRODUCER_WAIT_RELEASE,
                        old_mem=a,
                        new_mem=b,
                        release_roles=tuple(sorted(consumers.get(a, ()))),
                        acquire_roles=tuple(sorted(producers.get(b, ()))),
                        stage_overlaps=pair.stage_overlaps,
                        note=f"reuse {a}/{b} physical slot after transform",
                    ))
            elif b_to_a_overlap and not _has_transform_chain(orders, producers, consumers, role, a, b):
                diagnostics.append(
                    OverlayDiagnostic(
                        "error",
                        f"{role!r} writes {a!r} before finishing overlapping reads from {b!r}",
                        (a, b),
                        role,
                    ))

        # If one role writes A before B but another reads B before A, the safe
        # overlay protocol creates a wait cycle: writer waits for A release
        # before B, while reader waits for B before it can release A.
        for prod_role, prod_mems in orders.items():
            aw, bw = prod_mems.get(a, [None] * 4)[2], prod_mems.get(b, [None] * 4)[2]
            if aw is None or bw is None:
                continue
            producer_first = a if aw < bw else b
            for cons_role, cons_mems in orders.items():
                ar, br = cons_mems.get(a, [None] * 4)[0], cons_mems.get(b, [None] * 4)[0]
                if ar is None or br is None:
                    continue
                consumer_first = a if ar < br else b
                if producer_first != consumer_first:
                    diagnostics.append(
                        OverlayDiagnostic(
                            "error",
                            f"overlay deadlock: {prod_role!r} writes {producer_first!r} before the other "
                            f"buffer, but {cons_role!r} consumes {consumer_first!r} first",
                            (a, b),
                        ))

        has_reuse_order = any(
            item.kind == OverlayConstraintKind.PRODUCER_WAIT_RELEASE and {item.old_mem, item.new_mem} == {a, b}
            for item in constraints)
        if not has_reuse_order:
            a_before_b = _lifetime_precedes(
                a,
                b,
                orders=orders,
                producers=producers,
                consumers=consumers,
                role_graph=role_graph,
            )
            b_before_a = _lifetime_precedes(
                b,
                a,
                orders=orders,
                producers=producers,
                consumers=consumers,
                role_graph=role_graph,
            )
            if a_before_b != b_before_a:
                first, last = (a, b) if a_before_b else (b, a)
                _add_constraint(
                    constraints,
                    OverlayConstraint(
                        OverlayConstraintKind.PRODUCER_WAIT_RELEASE,
                        old_mem=last,
                        new_mem=first,
                        release_roles=tuple(sorted(consumers.get(last, ()))),
                        acquire_roles=tuple(sorted(producers.get(first, ()))),
                        stage_overlaps=pair.stage_overlaps,
                        note=f"reuse {a}/{b} physical slot across dataflow iterations",
                    ))
            else:
                reason = "cyclic" if a_before_b else "unordered"
                diagnostics.append(
                    OverlayDiagnostic("error", f"{a!r} and {b!r} overlap but have {reason} logical lifetimes", (a, b)))

    return tuple(constraints), tuple(diagnostics)


def analyze_buffer_overlays(graph: dict, mapping_plan: Any = None, *, strict: bool = True) -> BufferOverlayAnalysis:
    placements = build_placements(graph, mapping_plan)
    pairs = build_overlay_pairs(graph, placements)
    constraints, diagnostics = _analyze_constraints(graph, pairs, placements)
    analysis = BufferOverlayAnalysis(placements, pairs, constraints, diagnostics)
    if strict:
        errors = [d.message for d in diagnostics if d.severity == "error"]
        if errors:
            raise ValueError("buffer overlay analysis failed:\n" + "\n".join(f"- {e}" for e in errors))
    graph["buffer_overlay"] = analysis
    return analysis


def _mapping_for(ctx: dict, func_name: str, graph: dict) -> Any:
    if "memory_mapping_plan" in graph:
        return graph["memory_mapping_plan"]
    raw = ctx.get(BUFFER_MAPPING_CTX_KEY)
    if isinstance(raw, dict) and func_name in raw:
        return raw[func_name]
    return raw


def buffer_overlay_analysis_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    """Run after dataflow analysis; records overlay analysis in ``ctx``/graphs."""

    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY)
    if graphs is None:
        raise ValueError("buffer overlay analysis requires dataflow analysis first")
    strict = bool(ctx.get(BUFFER_OVERLAY_STRICT_CTX_KEY, True))
    out = ctx.setdefault(BUFFER_OVERLAY_CTX_KEY, {})
    for func_name, graph in graphs.items():
        out[func_name] = analyze_buffer_overlays(graph, _mapping_for(ctx, func_name, graph), strict=strict)
    return tree


__all__ = [
    "BUFFER_MAPPING_CTX_KEY",
    "BUFFER_OVERLAY_CTX_KEY",
    "BUFFER_OVERLAY_STRICT_CTX_KEY",
    "BufferOverlayAnalysis",
    "MemoryMappingEntry",
    "OverlayConstraint",
    "OverlayConstraintKind",
    "OverlayDiagnostic",
    "OverlayPair",
    "Placement",
    "StageOverlap",
    "analyze_buffer_overlays",
    "buffer_overlay_analysis_pass",
    "build_overlay_pairs",
    "build_placements",
    "overlaps",
    "stage_overlaps",
]
