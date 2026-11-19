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
"""Role/dataflow memory-event and stage-region analysis (RF05).

This module collects intrinsic memory events, binding regions and static
control-structure facts for scheduling and lowering. It emits no backend text.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import math
import operator
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Iterator

from little_kernel.backends.nv.sm100.passes.dataflow.common import call_name as _call_name

from little_kernel.core.memory_effects import Access, Completion, Effect, MemoryEffectKind, R_SMEM_TCGEN05_COMMIT, R_SMEM_TMA_BULK, W_SMEM_TMA_TX, MP, MS
from little_kernel.core.passes.utils.extract_empty_params import extract_empty_params, get_shape_size
from little_kernel.language.builtin_base import CTA_GROUP_ATTR, MEMORY_EFFECTS_ATTR, ORIGINAL_PY_FUNC_ATTR, get_intrin_ctx
from little_kernel.backends.nv.sm100.language.dataflow.declarations import Flow, TMATransferPlan
from little_kernel.backends.nv.sm100.passes.dataflow.channels import dtype_size as _dtype_size, reads as _reads, writes as _writes
from little_kernel.backends.nv.sm100.passes.dataflow.constants import CLUSTER_SHAPE_CTX_KEY, CTA_GROUP_SHAPE_CTX_KEY, DEFAULT_TMA_TRANSFER_PLAN
from little_kernel.backends.nv.sm100.passes.dataflow.intrinsics_registry import get_intrin_cta_group, get_intrin_mem_effects, intrin_has_completion as _intrin_has_completion
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import build_placements
from little_kernel.backends.nv.sm100.passes.dataflow.umma_modes import NORMALIZED_EFFECTS_ATTR
from little_kernel.backends.nv.sm100.passes.dataflow.logging import get_logger

logger = get_logger("dataflow.analysis")


@dataclass(frozen=True)
class MemEvent:
    role: str
    mem: str
    stmt_id: int
    op_id: object
    order: int
    intrin: str
    effect_kind: MemoryEffectKind
    cta_group: int
    cluster_multicast: bool
    control_path: tuple[tuple[str, int, str], ...] = ()
    logical_offset: int | None = None
    logical_extent: int | None = None
    logical_coord: tuple[int, ...] | None = None
    logical_shape: tuple[int, ...] | None = None
    transfer_plan: TMATransferPlan | None = None

    @property
    def access(self) -> Access:
        return self.effect_kind.access


@dataclass(frozen=True)
class RegionEvent:
    carrier_id: int
    event: MemEvent


@dataclass(frozen=True)
class BindRegion:
    key: int
    role: str
    value: ast.expr
    scope_id: tuple
    events: tuple[RegionEvent, ...]
    loops: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageRegionAnalysis:
    mem_events: list[MemEvent]
    bind_regions: list[BindRegion]
    bind_stmt_ids: set[int]


@dataclass
class _BindRegionBuilder:
    key: int
    role: str
    value: ast.expr
    scope_id: tuple
    body: list[ast.stmt]
    carrier_indices: list[int]
    events: list[RegionEvent]
    loops: tuple[str, ...]

    def finish(self) -> BindRegion | None:
        if not self.carrier_indices:
            return None
        first = min(self.carrier_indices)
        last = max(self.carrier_indices)
        interval = self.body[first:last + 1]
        for stmt in interval:
            if _has_early_exit(stmt):
                raise ValueError(f"{self.role}: stage key {self.key} interval cannot contain break/continue/return")
        return BindRegion(
            key=self.key,
            role=self.role,
            value=self.value,
            scope_id=self.scope_id,
            events=tuple(self.events),
            loops=self.loops,
        )


def _name(node: ast.AST | None) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _call_args(intrin: str, call: ast.Call) -> dict:
    from little_kernel.backends.nv.sm100.passes.dataflow.intrinsics_registry import get_dataflow_intrinsic
    fn = get_dataflow_intrinsic(intrin)
    if fn is None:
        return {}
    kwargs = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    try:
        return inspect.signature(getattr(fn, ORIGINAL_PY_FUNC_ATTR, fn)).bind_partial(*call.args, **kwargs).arguments
    except TypeError:
        return {}


def _ref_vars(node: ast.AST | None) -> set[str]:
    """Identifier names appearing in an expression AST (not definitions)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} if node is not None else set()


def _effect_operand_expr(
    effect: Effect,
    args: dict,
    assignments: dict[str, ast.expr],
) -> ast.AST | None:
    """Resolve an effect operand, including producer paths such as ``desc_a.mem``."""
    path = effect.operand.split(".")
    value = args.get(path[0])
    for producer_operand in path[1:]:
        seen: set[str] = set()
        while isinstance(value, ast.Name) and value.id in assignments:
            if value.id in seen:
                raise ValueError(f"cyclic definition while resolving effect operand {effect.operand!r}")
            seen.add(value.id)
            value = assignments[value.id]
        if not isinstance(value, ast.Call):
            rendered = ast.unparse(value) if isinstance(value, ast.AST) else value
            raise ValueError(f"effect operand {effect.operand!r} requires a descriptor-producing call, got {rendered}")
        producer = _call_name(value)
        producer_args = _call_args(producer, value)
        if producer_operand not in producer_args:
            raise ValueError(f"{producer}: no operand {producer_operand!r} for effect {effect.operand!r}")
        value = producer_args[producer_operand]
    return value


_CONST_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitAnd: operator.and_,
}


def _const_value(node: ast.AST | None, env: dict) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        value = env.get(node.id)
        if isinstance(value, TMATransferPlan):
            return value
        if isinstance(value, (int, bool, tuple, list)):
            return value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_const_value(item, env) for item in node.elts)
    if isinstance(node, ast.UnaryOp):
        value = _const_value(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -int(value)
        if isinstance(node.op, ast.UAdd):
            return int(value)
        if isinstance(node.op, ast.Not):
            return not bool(value)
    if isinstance(node, ast.BinOp) and type(node.op) in _CONST_BINOPS:
        return _CONST_BINOPS[type(node.op)](int(_const_value(node.left, env)), int(_const_value(node.right, env)))
    if isinstance(node, ast.Call) and len(node.args) == 1:
        name = _call_name(node)
        if name in {"int", "int32", "uint32", "int64", "uint64", "const", "constexpr"}:
            return _const_value(node.args[0], env)
    raise ValueError(f"expected a compile-time expression, got {ast.unparse(node) if node else None}")


_TMEM_ARRAY_REPEATS = {
    "smem_ld_128": {4},
    "tmem_ld_16dp_256bx2": {8},
    "tmem_ld_16dp_256bx4": {16},
    "tmem_ld_16dp_32bxn": {1, 2, 4, 8, 16, 32},
    "tmem_ld_32dp_32bxn": {1, 2, 4, 8, 16, 32},
    "tmem_st_16dp_128bx2": {4},
    "tmem_st_16dp_32bxn": {1, 2, 4, 8, 16, 32},
    "tmem_st_32dp_32bxn": {1, 2, 4, 8, 16, 32},
}


class _TmemRegisterArrayExpander(ast.NodeTransformer):

    def __init__(self, tree: ast.AST, ctx: dict) -> None:
        self.ctx = ctx
        self.arrays: dict[str, int | None] = {}
        for node in ast.walk(tree):
            value = None
            target = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target, value = node.targets[0], node.value
            elif isinstance(node, ast.AnnAssign):
                target, value = node.target, node.value
            if not (isinstance(target, ast.Name) and isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute) and isinstance(value.func.value, ast.Name)
                    and value.func.value.id == "ll" and value.func.attr in {"empty", "zeros"}):
                continue
            shape, _dtype, scope = extract_empty_params(value)
            if _const_value(scope, ctx) != "local":
                continue
            self.arrays[target.id] = (get_shape_size(shape)
                                      if isinstance(shape, (ast.List, ast.Tuple)) and len(shape.elts) == 1 else None)

    def visit_Call(self, node: ast.Call):
        node = self.generic_visit(node)
        if not (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ri" and node.func.attr in _TMEM_ARRAY_REPEATS and len(node.args) == 4):
            return node
        array, length_expr = node.args[2:]
        if not isinstance(array, ast.Name) or array.id not in self.arrays:
            raise ValueError(
                f"ri.{node.func.attr} array overload expects a named local ll.empty/ll.zeros register array")
        capacity = self.arrays[array.id]
        if capacity is None:
            raise ValueError(f"ri.{node.func.attr} array overload requires a one-dimensional register array")
        length = _const_value(length_expr, self.ctx)
        if type(length) is not int or length not in _TMEM_ARRAY_REPEATS[node.func.attr]:
            raise ValueError(f"unsupported {node.func.attr} repeat {length}")
        if capacity != length:
            raise ValueError(f"{array.id} declares {capacity} registers but {node.func.attr} length is {length}")
        node.args = node.args[:2] + [
            ast.copy_location(
                ast.Subscript(value=ast.Name(id=array.id, ctx=ast.Load()), slice=ast.Constant(value=i), ctx=ast.Load()),
                array) for i in range(length)
        ]
        return node


def tmem_register_array_expansion_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    _TmemRegisterArrayExpander(tree, ctx or {}).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


class _LocalExprResolver(ast.NodeTransformer):

    def __init__(self, assignments: dict[str, ast.expr], active: set[str] | None = None) -> None:
        self.assignments = assignments
        self.active = active or set()

    def visit_Name(self, node: ast.Name):
        value = self.assignments.get(node.id)
        if value is None or node.id in self.active:
            return node
        resolved = ast.parse(ast.unparse(value), mode="eval").body
        return _LocalExprResolver(self.assignments, self.active | {node.id}).visit(resolved)


def _resolve_local_expr(expr: ast.expr, assignments: dict[str, ast.expr]) -> ast.expr:
    cloned = ast.parse(ast.unparse(expr), mode="eval").body
    return _LocalExprResolver(assignments).visit(cloned)


def _topology_dependencies(
    expr: ast.expr,
    assignments: dict[str, ast.expr],
) -> set[tuple[str, str | None]]:
    """Find topology coordinates used by an expression after local substitution."""
    resolved = _resolve_local_expr(expr, assignments)
    dependencies: set[tuple[str, str | None]] = set()
    for node in ast.walk(resolved):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name in {"cluster_coord", "group_coord"}:
            if len(node.args) != 1:
                raise ValueError(f"{name}(axis) requires one literal axis")
            axis = _const_value(node.args[0], {})
            if axis not in {"x", "y", "z"}:
                raise ValueError(f"{name} axis must be literal 'x', 'y', or 'z'")
            dependencies.add((name, axis))
        elif name == "rank_in_group":
            dependencies.add((name, None))
        elif name == "cluster_rank":
            dependencies.add((name, None))
    return dependencies


def _logical_region(
    effect: Effect,
    args: dict,
    env: dict,
    assignments: dict[str, ast.expr],
    mem: str,
) -> tuple[int | None, int | None]:
    region = effect.region
    if region is None:
        return None, None

    raw_offset = (args.get(region.offset_operand, region.offset_default)
                  if region.offset_operand is not None else region.offset_default)
    if raw_offset is None:
        return None, None
    if isinstance(raw_offset, int):
        offset = raw_offset
    else:
        resolved_offset = _resolve_local_expr(raw_offset, assignments)
        try:
            offset = int(_const_value(resolved_offset, {**env, mem: 0}))
        except ValueError:
            return None, None

    raw_extent = region.extent
    if isinstance(raw_extent, str):
        extent_value = args.get(raw_extent)
        if isinstance(extent_value, tuple):
            extent = len(extent_value)
        elif extent_value is None:
            extent = None
        elif isinstance(extent_value, int):
            extent = extent_value
        else:
            try:
                extent = int(_const_value(_resolve_local_expr(extent_value, assignments), env))
            except ValueError:
                extent = None
    else:
        extent = raw_extent
    if offset < 0 or (extent is not None and extent <= 0):
        raise ValueError(f"invalid logical memory region offset={offset}, extent={extent}")
    return offset, extent


def _static_tuple(
    value: ast.AST | tuple | list | None,
    env: dict,
    assignments: dict[str, ast.expr],
    what: str,
) -> tuple[int, ...]:
    if isinstance(value, ast.expr):
        value = _const_value(_resolve_local_expr(value, assignments), env)
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{what} must be a compile-time tuple")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result):
        raise ValueError(f"{what} must be non-negative")
    return result


class ChannelMemEventCollector(ast.NodeVisitor):

    def __init__(self, channel_mems: set[str], role: str | None = None, const_env: dict | None = None,
                 assignments: dict[str, ast.expr] | None = None) -> None:
        self.channel_mems = channel_mems
        self.role = role
        self.const_env = const_env or {}
        self.stmt: ast.stmt | None = None
        self.events: list[MemEvent] = []
        self.seen: set[tuple[int, object, str, Access]] = set()
        self.control_path: tuple[tuple[str, int, str], ...] = ()
        self.assignments = dict(assignments or {})
        self.static_instance: tuple[tuple[int, int], ...] = ()

    def visit(self, node):
        if isinstance(node, ast.stmt):
            old = self.stmt
            self.stmt = node
            out = super().visit(node)
            self.stmt = old
            return out
        return super().visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self.role is not None and self.stmt is not None:
            intrin = _call_name(node) or ""
            args = _call_args(intrin, node)
            for effect in getattr(node, NORMALIZED_EFFECTS_ATTR, get_intrin_mem_effects(intrin)):
                if effect.space == MS.RMEM:
                    continue
                for mem in _ref_vars(_effect_operand_expr(effect, args, self.assignments)) & self.channel_mems:
                    self.add(mem, effect, intrin, node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.assignments[node.targets[0].id] = node.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assignments[node.target.id] = node.value
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_control_body("if", node, "body", node.body)
        self._visit_control_body("if", node, "else", node.orelse)

    def visit_For(self, node: ast.For) -> None:
        values = self._unroll_values(node)
        if values is not None:
            assert isinstance(node.target, ast.Name)
            old_value = self.const_env.get(node.target.id, None)
            had_value = node.target.id in self.const_env
            base_assignments = self.assignments.copy()
            for value in values:
                self.const_env[node.target.id] = value
                self.assignments = base_assignments.copy()
                old_instance = self.static_instance
                self.static_instance = old_instance + ((id(node), value), )
                self._visit_control_body("unroll", node, "body", node.body)
                self.static_instance = old_instance
            self.assignments = base_assignments
            if had_value:
                self.const_env[node.target.id] = old_value
            else:
                self.const_env.pop(node.target.id, None)
            return
        self.visit(node.iter)
        kind = "unroll" if _call_name(node.iter) == "unroll" else "for"
        self._visit_control_body(kind, node, "body", node.body)
        self._visit_control_body(kind, node, "else", node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit_For(node)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_control_body("while", node, "body", node.body)
        self._visit_control_body("while", node, "else", node.orelse)

    def _unroll_values(self, node: ast.For) -> list[int] | None:
        if (not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call)
                or _call_name(node.iter) != "unroll" or len(node.iter.args) != 1):
            return None
        range_call = node.iter.args[0]
        if not isinstance(range_call, ast.Call) or _call_name(range_call) != "range":
            return None
        try:
            args = [int(_const_value(arg, self.const_env)) for arg in range_call.args]
        except ValueError:
            return None
        return list(range(*args))

    def _visit_control_body(self, kind: str, node: ast.AST, arm: str, body: list[ast.stmt]) -> None:
        old = self.control_path
        control_id = getattr(node, "_lk_control_origin", id(node))
        self.control_path = old + ((kind, control_id, arm), )
        for stmt in body:
            self.visit(stmt)
        self.control_path = old

    def add(self, mem: str, effect: Effect, intrin: str, call: ast.Call) -> None:
        assert self.role is not None and self.stmt is not None
        op_id = (id(call), self.static_instance)
        key = (id(self.stmt), op_id, mem, effect.access)
        if key in self.seen:
            return
        self.seen.add(key)
        call_args = _call_args(intrin, call)
        offset, extent = _logical_region(effect, call_args, self.const_env, self.assignments, mem)
        logical_coord = logical_shape = None
        if intrin in {"tma_ld_2d", "tma_ld_3d", "tma_st_2d", "tma_st_3d"}:
            logical_shape = _static_tuple(call_args.get("shape"), self.const_env, self.assignments, f"{intrin} shape")
            if not logical_shape or any(item == 0 for item in logical_shape):
                raise ValueError(f"{intrin} shape must be positive")
            logical_coord = _static_tuple(call_args.get("smem_coord", (0, ) * len(logical_shape)), self.const_env,
                                          self.assignments, f"{intrin} smem_coord")
            if len(logical_coord) != len(logical_shape):
                raise ValueError(f"{intrin} smem_coord and shape ranks differ")
        transfer_plan = None
        cta_group = get_intrin_cta_group(intrin)
        cluster_multicast = False
        if intrin in {"tma_ld_2d", "tma_ld_3d"}:
            plan_expr = call_args.get("plan")
            plan_value = None if plan_expr is None else _const_value(plan_expr, self.const_env)
            transfer_plan = DEFAULT_TMA_TRANSFER_PLAN if plan_value is None else plan_value
            if not isinstance(transfer_plan, TMATransferPlan):
                raise ValueError(f"{intrin}: plan must be a compile-time TMATransferPlan")
            base_coord = call_args.get("gmem_coord")
            if not isinstance(base_coord, ast.expr):
                raise ValueError(f"{intrin}: gmem_coord must be an expression")
            dependencies = _topology_dependencies(base_coord, self.assignments)
            forbidden = {
                dependency
                for dependency in dependencies
                if dependency[0] in {"cluster_coord", "cluster_rank", "rank_in_group"} or (
                    dependency[0] == "group_coord" and dependency[1] == transfer_plan.mcast_group_dim)
            }
            if forbidden:
                details = ", ".join(name if axis is None else f"{name}({axis!r})" for name, axis in sorted(forbidden))
                raise ValueError(f"{intrin}: gmem_coord must be group-tile based, "
                                 "independent of rank_in_group and mcast_group_dim; "
                                 f"found {details}")
            # plan=None is always a plain single-CTA descriptor load (former
            # tma_ld_*_1sm), even when the kernel uses cg2 for UMMA. A named
            # g.tma_transfer(...) uses the kernel cta_group_shape even if its
            # fields match the default plan.
            if plan_value is None:
                cta_group = 1
                cluster_multicast = False
            else:
                group_shape = self.const_env.get(CTA_GROUP_SHAPE_CTX_KEY)
                cluster_shape = self.const_env.get(CLUSTER_SHAPE_CTX_KEY)
                if group_shape is None:
                    raise ValueError(f"{intrin}: non-default TMATransferPlan requires explicit cta_group_shape")
                cta_group = math.prod(group_shape)
                if cluster_shape is None:
                    if cta_group != 1 or transfer_plan.mcast_group_dim is not None:
                        raise ValueError(f"{intrin}: clustered TMATransferPlan requires cluster_shape")
                    cluster_shape = (1, 1, 1)
                axis = {"x": 0, "y": 1, "z": 2}.get(transfer_plan.mcast_group_dim)
                cluster_multicast = (axis is not None and cluster_shape[axis] // group_shape[axis] > 1)
        self.events.append(
            MemEvent(
                role=self.role,
                mem=mem,
                stmt_id=id(self.stmt),
                op_id=op_id,
                order=len(self.events),
                intrin=intrin,
                effect_kind=effect.kind,
                cta_group=cta_group,
                cluster_multicast=cluster_multicast,
                control_path=self.control_path,
                logical_offset=offset,
                logical_extent=extent,
                logical_coord=logical_coord,
                logical_shape=logical_shape,
                transfer_plan=transfer_plan,
            ))


def _stmt_mem_events(stmt: ast.stmt, role: str, channel_mems: set[str], const_env: dict,
                     assignments: dict[str, ast.expr] | None = None) -> list[MemEvent]:
    collector = ChannelMemEventCollector(channel_mems, role, const_env, assignments)
    collector.visit(stmt)
    return collector.events


def _read_aliases_for_channel(dataflow: Flow, via: set[str]) -> dict[str, str]:
    """Map exact, read-only typed views onto a canonical channel allocation."""
    placements = build_placements({"memories": dataflow.memories}, getattr(dataflow, "memory_mapping_plan", None))
    aliases: dict[str, str] = {}
    for channel_mem in via:
        channel = placements[channel_mem]
        if channel.logical != channel.physical:
            continue
        for logical, candidate in placements.items():
            if logical in via or candidate.logical == candidate.physical:
                continue
            if (candidate.physical, candidate.space, candidate.offset, candidate.size, candidate.stages,
                    candidate.stage_stride) == (channel.physical, channel.space, channel.offset, channel.size,
                                                channel.stages, channel.stage_stride):
                aliases[logical] = channel_mem
    return aliases


def _canonicalize_read_aliases(events: list[MemEvent], aliases: dict[str, str]) -> list[MemEvent]:
    return [
        replace(event, mem=aliases[event.mem])
        if event.mem in aliases and _reads(event.access) and not _writes(event.access) else event for event in events
    ]


def _has_early_exit(stmt: ast.stmt) -> bool:
    return any(isinstance(node, (ast.Break, ast.Continue, ast.Return)) for node in ast.walk(stmt))


def _bind_key(stmt: ast.stmt, stage_key_vars: dict[str, int]) -> int | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    func = stmt.value.func
    if not isinstance(func, ast.Attribute) or func.attr != "bind":
        return None
    if not isinstance(func.value, ast.Name):
        return None
    return stage_key_vars.get(func.value.id)


def _bind_value(stmt: ast.stmt) -> ast.expr:
    assert isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
    if len(stmt.value.args) != 1 or stmt.value.keywords:
        raise ValueError("stage_key.bind(...) requires exactly one positional stage value")
    return stmt.value.args[0]


def _contains_bind_key(stmt: ast.stmt, key: int, stage_key_vars: dict[str, int]) -> bool:
    return any(isinstance(node, ast.stmt) and _bind_key(node, stage_key_vars) == key for node in ast.walk(stmt))


def _assigned_value(stmt: ast.stmt, name: str) -> ast.expr | None:
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.value if stmt.target.id == name else None
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        return stmt.value if stmt.targets[0].id == name else None
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == name:
        return ast.BinOp(left=ast.Name(id=name, ctx=ast.Load()), op=stmt.op, right=stmt.value)
    return None


class _LoopVarNormalizer(ast.NodeTransformer):

    def __init__(self, name: str) -> None:
        self.name = name

    def visit_Name(self, node: ast.Name):
        if node.id == self.name:
            return ast.copy_location(ast.Name(id="$iv", ctx=node.ctx), node)
        return node


def _loop_expr(expr: ast.expr, induction_var: str) -> str:
    return ast.dump(
        _LoopVarNormalizer(induction_var).visit(
            ast.fix_missing_locations(ast.parse(ast.unparse(expr), mode="eval").body)),
        include_attributes=False,
    )


def _is_multiple_of(expr: ast.expr, factor: int) -> bool:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, int):
        return expr.value % factor == 0
    if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, ast.Mult):
        return False
    return _is_multiple_of(expr.left, factor) or _is_multiple_of(expr.right, factor)


def _positive_int_factor(expr: ast.expr) -> tuple[int, ast.expr]:
    if not isinstance(expr, ast.BinOp) or not isinstance(expr.op, ast.Mult):
        return 1, expr
    if isinstance(expr.left, ast.Constant) and isinstance(expr.left.value, int) and expr.left.value > 0:
        return expr.left.value, expr.right
    if isinstance(expr.right, ast.Constant) and isinstance(expr.right.value, int) and expr.right.value > 0:
        return expr.right.value, expr.left
    return 1, expr


def _while_signature(node: ast.While, parent: list[ast.stmt], index: int) -> str:
    if (not isinstance(node.test, ast.Compare) or len(node.test.ops) != 1 or len(node.test.comparators) != 1
            or not isinstance(node.test.left, ast.Name)):
        return f"unproven:while:{getattr(node, 'lineno', 0)}"
    induction_var = node.test.left.id
    init = next(
        (value for stmt in reversed(parent[:index]) if (value := _assigned_value(stmt, induction_var)) is not None),
        None,
    )
    updates = [value for stmt in node.body if (value := _assigned_value(stmt, induction_var)) is not None]
    if init is None or len(updates) != 1 or node.orelse:
        return f"unproven:while:{getattr(node, 'lineno', 0)}"
    op = type(node.test.ops[0]).__name__
    update = updates[0]
    if (isinstance(init, ast.Constant) and init.value == 0 and op == "Lt" and isinstance(update, ast.BinOp)
            and isinstance(update.op, ast.Add) and isinstance(update.left, ast.Name) and update.left.id == induction_var
            and isinstance(update.right, ast.Constant) and isinstance(update.right.value, int)
            and update.right.value > 0):
        step = update.right.value
        upper = node.test.comparators[0]
        if step == 1 or _is_multiple_of(upper, step):
            scale, base = _positive_int_factor(upper)
            scale_text = f"|scale{scale}" if scale > 1 else ""
            return f"while|0|Lt|{_loop_expr(base, induction_var)}{scale_text}|step{step}"
    return "|".join((
        "while",
        _loop_expr(init, induction_var),
        op,
        _loop_expr(node.test.comparators[0], induction_var),
        _loop_expr(update, induction_var),
    ))


def _for_signature(node: ast.For | ast.AsyncFor) -> str:
    if not isinstance(node.target, ast.Name) or node.orelse:
        return f"unproven:for:{getattr(node, 'lineno', 0)}"
    call = node.iter
    if isinstance(call, ast.Call) and _call_name(call) == "unroll" and len(call.args) == 1:
        call = call.args[0]
    if not isinstance(call, ast.Call) or _call_name(call) != "range":
        return f"unproven:for:{getattr(node, 'lineno', 0)}"
    if (len(call.args) == 1 and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, int)
            and call.args[0].value >= 0):
        return f"repeat|{call.args[0].value}"
    args = ",".join(_loop_expr(arg, node.target.id) for arg in call.args)
    return f"for|{args}"


class BindRegionCollector:

    def __init__(self, dataflow: Flow, const_env: dict) -> None:
        self.roles = set(dataflow.roles)
        self.channels_by_key = {channel["key"]: channel for channel in dataflow.channels}
        declared_channel_mems = {mem for channel in dataflow.channels for mem in channel["via"]}
        self.read_aliases_by_key = {
            channel["key"]: {
                alias: canonical
                for alias, canonical in _read_aliases_for_channel(dataflow, set(channel["via"])).items()
                if alias not in declared_channel_mems
            }
            for channel in dataflow.channels
        }
        self.stage_key_vars = dataflow.stage_key_vars
        self.const_env = const_env
        self.bind_stmt_ids: set[int] = set()
        self.regions: list[BindRegion] = []

    def collect(self, stmts: list[ast.stmt]) -> list[BindRegion]:
        self._scan_block(stmts, None)
        return self.regions

    def _scan_block(
        self,
        body: list[ast.stmt],
        role: str | None,
        scope_id: tuple | None = None,
        loops: tuple[str, ...] = (),
        assignments: dict[str, ast.expr] | None = None,
        *,
        conditional: bool = False,
    ) -> None:
        scope_id = (id(body), ) if scope_id is None else scope_id
        assignments = dict(assignments or {})
        active: dict[int, _BindRegionBuilder] = {}
        for idx, stmt in enumerate(body):
            key = _bind_key(stmt, self.stage_key_vars)
            if role is not None and key is not None:
                if conditional:
                    raise ValueError(f"{role}: stage_key.bind(...) must be outside runtime conditionals")
                self.bind_stmt_ids.add(id(stmt))
                self._finish(active.pop(key, None))
                active[key] = _BindRegionBuilder(key, role, _bind_value(stmt), scope_id, body, [], [], loops)
                continue

            for active_key, builder in list(active.items()):
                if _contains_bind_key(stmt, active_key, self.stage_key_vars):
                    self._finish(active.pop(active_key))
                    continue
                channel = self.channels_by_key[active_key]
                aliases = self.read_aliases_by_key[active_key]
                events = _stmt_mem_events(stmt, role, set(channel["via"]) | set(aliases), self.const_env, assignments)
                events = _canonicalize_read_aliases(events, aliases)
                if events:
                    if not builder.carrier_indices or builder.carrier_indices[-1] != idx:
                        builder.carrier_indices.append(idx)
                    builder.events.extend(RegionEvent(id(stmt), event) for event in events)

            self._scan_nested(stmt, role, scope_id, loops, conditional, body, idx, assignments)

            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                assignments[stmt.target.id] = stmt.value
            elif (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
                assignments[stmt.targets[0].id] = stmt.value

        for builder in active.values():
            self._finish(builder)

    def _finish(self, builder: _BindRegionBuilder | None) -> None:
        if builder is None:
            return
        region = builder.finish()
        if region is not None:
            self.regions.append(region)

    def _scan_nested(
        self,
        stmt: ast.stmt,
        role: str | None,
        scope_id: tuple,
        loops: tuple[str, ...],
        conditional: bool,
        parent: list[ast.stmt],
        index: int,
        assignments: dict[str, ast.expr],
    ) -> None:
        name = _role_context(stmt, self.roles)
        if name is not None:
            assert isinstance(stmt, ast.With)
            self._scan_block(stmt.body, name, scope_id, loops, assignments, conditional=conditional)
            return

        if isinstance(stmt, ast.If):
            self._scan_block(stmt.body, role, scope_id + (("if", id(stmt), "body"), ), loops, assignments,
                             conditional=True)
            self._scan_block(stmt.orelse, role, scope_id + (("if", id(stmt), "else"), ), loops, assignments,
                             conditional=True)
            return

        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            loop = (_while_signature(stmt, parent, index) if isinstance(stmt, ast.While) else _for_signature(stmt))
            self._scan_block(
                stmt.body,
                role,
                scope_id + ((type(stmt).__name__.lower(), id(stmt), "body"), ),
                loops + (loop, ),
                assignments,
                conditional=conditional,
            )
            self._scan_block(
                stmt.orelse,
                role,
                scope_id + ((type(stmt).__name__.lower(), id(stmt), "else"), ),
                loops + (loop, ),
                assignments,
                conditional=conditional,
            )
            return

        for body in _nested_blocks(stmt):
            self._scan_block(body, role, loops=loops, assignments=assignments, conditional=True)


def _nested_blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        return [stmt.body, stmt.orelse]
    if isinstance(stmt, ast.If):
        return [stmt.body, stmt.orelse]
    if isinstance(stmt, (ast.With, ast.AsyncWith)):
        return [stmt.body]
    if isinstance(stmt, ast.Try):
        return [stmt.body, stmt.orelse, stmt.finalbody] + [h.body for h in stmt.handlers]
    return []


def _role_context(stmt: ast.stmt, roles: set[str]) -> str | None:
    if not isinstance(stmt, ast.With) or len(stmt.items) != 1:
        return None
    ctx = stmt.items[0].context_expr
    if not isinstance(ctx, ast.Name):
        return None
    return ctx.id if ctx.id in roles else None


def collect_channel_mem_events(
    func: ast.FunctionDef,
    init_idx: int,
    dataflow: Flow,
    const_env: dict | None = None,
) -> list[MemEvent]:
    """Collect RW events on channel ``via`` buffers inside ``with role:`` bodies."""
    roles = set(dataflow.roles)
    channel_mems = {m for channel in dataflow.channels for m in channel["via"]}
    declared_channel_mems = set(channel_mems)
    placements = build_placements({"memories": dataflow.memories}, getattr(dataflow, "memory_mapping_plan", None))
    channel_mems.update(placements[mem].physical for mem in tuple(channel_mems))
    read_aliases: dict[str, str] = {}
    for channel in dataflow.channels:
        read_aliases.update({
            alias: canonical
            for alias, canonical in _read_aliases_for_channel(dataflow, set(channel["via"])).items()
            if alias not in declared_channel_mems
        })
    channel_mems.update(read_aliases)
    collector = ChannelMemEventCollector(channel_mems, const_env=const_env)

    def scan_block(body: list[ast.stmt], role: str | None = None) -> None:
        for stmt in body:
            child_role = _role_context(stmt, roles)
            if child_role is not None:
                assert isinstance(stmt, ast.With)
                scan_block(stmt.body, child_role)
            elif role is not None:
                collector.role = role
                collector.visit(stmt)
            else:
                for child in _nested_blocks(stmt):
                    scan_block(child)

    scan_block(func.body[init_idx:])
    return _canonicalize_read_aliases(collector.events, read_aliases)


def collect_bind_regions(
    func: ast.FunctionDef,
    init_idx: int,
    dataflow: Flow,
    const_env: dict | None = None,
) -> tuple[list[BindRegion], set[int]]:
    collector = BindRegionCollector(dataflow, const_env or {})
    collector.collect(func.body[init_idx:])
    return collector.regions, collector.bind_stmt_ids


def analyze_stage_regions(
    func: ast.FunctionDef,
    init_idx: int,
    dataflow: Flow,
    const_env: dict | None = None,
) -> StageRegionAnalysis:
    """Collect memory effects and the bind regions that own them."""
    mem_events = collect_channel_mem_events(func, init_idx, dataflow, const_env)
    bind_regions, bind_stmt_ids = collect_bind_regions(func, init_idx, dataflow, const_env)
    logger.debug("analyze_stage_regions: mem_events=%d bind_regions=%d", len(mem_events), len(bind_regions))
    if logger.isEnabledFor(logging.DEBUG):
        json_default = (lambda o: ast.unparse(o) if isinstance(o, ast.AST) else o.value
                        if isinstance(o, Access) else list(o) if isinstance(o, tuple) else str(o))
        for i, event in enumerate(mem_events):
            logger.debug("mem_events[%d]\n%s", i, json.dumps(asdict(event), indent=2, default=json_default))
        for i, region in enumerate(bind_regions):
            logger.debug("bind_regions[%d]\n%s", i, json.dumps(asdict(region), indent=2, default=json_default))
    return StageRegionAnalysis(mem_events, bind_regions, bind_stmt_ids)
