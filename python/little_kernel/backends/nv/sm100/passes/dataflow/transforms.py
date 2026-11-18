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
"""Role/dataflow AST transform passes (RF05).

This module owns structural effect partitioning, graph analysis and stage
insertion planning. CUDA primitive emission and resource materialization live
under :mod:`little_kernel.backends.nv.sm100.codegen.dataflow`.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from little_kernel.backends.nv.sm100.passes.dataflow.common import call_name as _call_name
from little_kernel.core.memory_effects import R_SMEM_TMA_BULK, W_SMEM_TMA_TX
from little_kernel.backends.nv.sm100.passes.dataflow.constants import DATAFLOW_CTX_KEY
from little_kernel.backends.nv.sm100.passes.dataflow.graph_construction import build_dataflow, dataflow_decl
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import BindRegion, ChannelMemEventCollector, MemEvent, RegionEvent, _const_value, _name, _ref_vars
from little_kernel.backends.nv.sm100.passes.dataflow.graph_ops import _ident, _pipe_name, _pipeline_region_bounds, _pipeline_side_keys, _pipeline_side_via, _pipeline_stages, _pipeline_var_name, _region_bounds, _role_mem_events
from little_kernel.backends.nv.sm100.passes.dataflow.channels import barrier_participation, consumer_roles as _consumer_roles, producer_roles as _producer_roles, reads as _reads, writes as _writes
from little_kernel.backends.nv.sm100.passes.dataflow.scheduling import IssuerAnalysis, _resolve_issuer_site, _tma_store_sites, infer_tma_store_streams, specialize_same_role_tma_relays
from little_kernel.backends.nv.sm100.core.pipeline import PIPELINE_GENERIC_GENERIC, PIPELINE_GENERIC_UMMA, PIPELINE_TMEM_STORE_TMEM_LOAD, PIPELINE_TMEM_STORE_UMMA, PIPELINE_TMA_GENERIC, PIPELINE_TMA_STORE_CROSS_ROLE, PIPELINE_TMA_STORE_SAME_ROLE, PIPELINE_TMA_TMA_STORE, PIPELINE_TMA_TMA_STORE_SAME_ROLE, PIPELINE_TMA_UMMA, PIPELINE_UMMA_TMEM_LOAD, PIPELINE_UMMA_UMMA, PipelineSpec

PipelineSite = tuple[str, str]


@dataclass
class StageRegionInsertionPlan:
    before: dict[int, list[PipelineSite]] = field(default_factory=dict)
    after: dict[int, list[PipelineSite]] = field(default_factory=dict)
    role_tail: dict[str, list[str]] = field(default_factory=dict)
    live_pipelines: list[dict] = field(default_factory=list)


def _add_plan_site(bucket: dict[int, list[PipelineSite]], stmt_id: int, pipe: str, method: str) -> None:
    site = (pipe, method)
    bucket.setdefault(stmt_id, [])
    if site not in bucket[stmt_id]:
        bucket[stmt_id].append(site)


def _if_child_locations(func: ast.FunctionDef, ) -> tuple[dict[int, ast.If], dict[tuple[int, int], tuple[str, int]]]:
    if_nodes: dict[int, ast.If] = {}
    locations: dict[tuple[int, int], tuple[str, int]] = {}
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        if_nodes[id(node)] = node
        for arm, body in (("body", node.body), ("else", node.orelse)):
            for index, child in enumerate(body):
                for descendant in ast.walk(child):
                    if isinstance(descendant, ast.stmt):
                        locations[(id(node), id(descendant))] = (arm, index)
    return if_nodes, locations


def _stored_names(stmts: list[ast.stmt]) -> set[str]:
    return {
        node.id
        for stmt in stmts
        for node in ast.walk(stmt)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }


def _repeatable_if_test(node: ast.If) -> bool:
    impure = (ast.Call, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom, ast.Subscript)
    if any(isinstance(item, impure) for item in ast.walk(node.test)):
        return False
    return not (_ref_vars(node.test) & _stored_names(node.body + node.orelse))


def _uniform_names(func: ast.FunctionDef) -> set[str]:
    forbidden = (ast.Call, ast.Subscript, ast.Attribute, ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)
    assignments: dict[str, list[tuple[ast.expr | None, tuple[ast.expr, ...]]]] = {}

    def record(name: str, value: ast.expr | None, parents: tuple[ast.AST, ...]) -> None:
        controls = tuple(parent.test for parent in parents if isinstance(parent, ast.If))
        assignments.setdefault(name, []).append((value, controls))

    def scan(node: ast.AST, parents: tuple[ast.AST, ...] = ()) -> None:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                record(target.id, node.value, parents)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            record(node.target.id, node.value, parents)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            record(
                node.target.id,
                ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value),
                parents,
            )
        for child in ast.iter_child_nodes(node):
            scan(child, parents + (node, ))

    scan(func)
    assigned = set(assignments)
    uniform: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, entries in assignments.items():
            values = [expression for value, controls in entries for expression in ((value, ) + controls)]
            if name in uniform or any(value is None for value in values):
                continue
            assert all(value is not None for value in values)
            if any(any(isinstance(item, forbidden) for item in ast.walk(value)) for value in values):
                continue
            deps = {dep for value in values for dep in _ref_vars(value) if dep != name}
            if all(dep not in assigned or dep in uniform for dep in deps):
                uniform.add(name)
                changed = True
    return uniform


def _uniform_if_ids(func: ast.FunctionDef) -> set[int]:
    assigned = {node.id for node in ast.walk(func) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    uniform = _uniform_names(func)
    return {
        id(node)
        for node in ast.walk(func)
        if isinstance(node, ast.If) and node.orelse and _repeatable_if_test(node) and all(
            name not in assigned or name in uniform for name in _ref_vars(node.test))
    }


def _collapse_effect_free_cuts(cuts: set[int], effect_indices: set[int]) -> list[int]:
    collapsed: list[int] = []
    internal = [
        cut for cut in sorted(cuts)
        if any(index < cut for index in effect_indices) and any(index >= cut for index in effect_indices)
    ]
    for cut in internal:
        if (collapsed and not any(collapsed[-1] <= index < cut for index in effect_indices)):
            continue
        collapsed.append(cut)
    return collapsed


def _effect_carrier_cuts(func: ast.FunctionDef, graph: dict) -> dict[int, tuple[list[int], list[int]]]:
    """Find legal ``if`` split points at channel effect boundaries."""
    if_nodes, locations = _if_child_locations(func)
    uniform_if_ids = _uniform_if_ids(func)
    cuts: dict[int, dict[str, set[int]]] = {}
    effect_indices: dict[int, dict[str, set[int]]] = {}

    for event in graph["mem_events"]:
        for if_id, node in if_nodes.items():
            location = locations.get((if_id, event.stmt_id))
            if location is None:
                continue
            arm, index = location
            effect_indices.setdefault(if_id, {"body": set(), "else": set()})[arm].add(index)

    for pipeline in graph["pipelines"]:
        for side, roles, pred in (
            ("producer", _producer_roles(pipeline), _writes),
            ("consumer", _consumer_roles(pipeline), _reads),
        ):
            via = _pipeline_side_via(pipeline, side)
            for role in roles:
                role_events = _role_mem_events(graph, role, via, pred)
                if any(event.effect_kind == W_SMEM_TMA_TX for event in role_events):
                    # TMA load completion belongs to its transaction mbarrier. TMA
                    # stores are different: splitting a repeatable issuer guard at
                    # source-lifetime boundaries exposes independent bulk groups.
                    continue
                for _first, _last, region in _pipeline_region_bounds(graph, pipeline, side, role, pred):
                    by_carrier: dict[int, list[RegionEvent]] = {}
                    for item in region.events:
                        if item.event.mem in via and pred(item.event.access):
                            by_carrier.setdefault(item.carrier_id, []).append(item)

                    for carrier_id, events in by_carrier.items():
                        node = if_nodes.get(carrier_id)
                        if node is None:
                            continue
                        positions: dict[str, list[int]] = {}
                        for item in events:
                            location = locations.get((carrier_id, item.event.stmt_id))
                            if location is None:
                                continue
                            arm, index = location
                            positions.setdefault(arm, []).append(index)
                        for arm, indices in positions.items():
                            body = node.body if arm == "body" else node.orelse
                            before = min(indices)
                            after = max(indices) + 1
                            bucket = cuts.setdefault(carrier_id, {"body": set(), "else": set()})[arm]
                            if before:
                                bucket.add(before)
                            if after < len(body):
                                bucket.add(after)

    result: dict[int, tuple[list[int], list[int]]] = {}
    for if_id, arm_cuts in cuts.items():
        node = if_nodes[if_id]
        if if_id in uniform_if_ids:
            continue
        if not _repeatable_if_test(node):
            continue
        body_cuts = _collapse_effect_free_cuts(arm_cuts["body"], effect_indices.get(if_id, {}).get("body", set()))
        else_cuts = _collapse_effect_free_cuts(arm_cuts["else"], effect_indices.get(if_id, {}).get("else", set()))
        if node.orelse:
            if (not body_cuts or not else_cuts or len(body_cuts) != len(else_cuts)
                    or (len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If))):
                continue
        if body_cuts or else_cuts:
            result[if_id] = (body_cuts, else_cuts)
    return result


def _split_body(body: list[ast.stmt], cuts: list[int]) -> list[list[ast.stmt]]:
    bounds = [0, *cuts, len(body)]
    return [body[begin:end] for begin, end in zip(bounds, bounds[1:])]


class _EffectCarrierSplitter(ast.NodeTransformer):

    def __init__(self, cuts: dict[int, tuple[list[int], list[int]]]) -> None:
        self.cuts = cuts

    def _visit_body(self, body: list[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []
        for stmt in body:
            transformed = self.visit(stmt)
            if isinstance(transformed, list):
                result.extend(transformed)
            elif transformed is not None:
                result.append(transformed)
        return result

    def visit_If(self, node: ast.If):
        split = self.cuts.get(id(node))
        if split is None:
            return self.generic_visit(node)

        body_parts = _split_body(node.body, split[0])
        else_parts = (_split_body(node.orelse, split[1]) if node.orelse else [[] for _ in body_parts])
        if len(body_parts) != len(else_parts):
            return self.generic_visit(node)

        origin = getattr(node, "_lk_control_origin", id(node))
        node._lk_control_origin = origin
        result: list[ast.If] = []
        for index, (body, orelse) in enumerate(zip(body_parts, else_parts, strict=True)):
            if not body:
                return self.generic_visit(node)
            if index == 0:
                node.body = self._visit_body(body)
                node.orelse = self._visit_body(orelse)
                result.append(node)
                continue
            test = ast.parse(ast.unparse(node.test), mode="eval").body
            part = ast.If(test=test, body=self._visit_body(body), orelse=self._visit_body(orelse))
            part._lk_control_origin = origin
            result.append(ast.copy_location(part, node))
        return result


def _pipeline_region_site_key(pipeline: dict, role: str, region: BindRegion, method: str) -> tuple:
    return (
        _pipeline_var_name(pipeline),
        role,
        region.key,
        region.loops,
        ast.dump(region.value, include_attributes=False),
        method,
    )


def _uniform_branch_sites(func: ast.FunctionDef, graph: dict) -> dict[tuple, tuple[int, ...]]:
    if_nodes, locations = _if_child_locations(func)
    uniform_if_ids = _uniform_if_ids(func)
    sites: dict[tuple, list[int]] = {}

    for pipeline in graph["pipelines"]:
        for side, roles, pred in (
            ("producer", _producer_roles(pipeline), _writes),
            ("consumer", _consumer_roles(pipeline), _reads),
        ):
            via = _pipeline_side_via(pipeline, side)
            for role in roles:
                role_events = _role_mem_events(graph, role, via, pred)
                if any(event.effect_kind in {W_SMEM_TMA_TX, R_SMEM_TMA_BULK} for event in role_events):
                    continue
                for _first, _last, region in _pipeline_region_bounds(graph, pipeline, side, role, pred):
                    by_carrier: dict[int, list[RegionEvent]] = {}
                    for item in region.events:
                        if item.event.mem in via and pred(item.event.access):
                            by_carrier.setdefault(item.carrier_id, []).append(item)

                    for carrier_id, events in by_carrier.items():
                        node = if_nodes.get(carrier_id)
                        if node is None or carrier_id not in uniform_if_ids:
                            continue
                        positions: dict[str, list[int]] = {"body": [], "else": []}
                        for item in events:
                            location = locations.get((carrier_id, item.event.stmt_id))
                            if location is not None:
                                positions[location[0]].append(location[1])
                        if not positions["body"] or not positions["else"]:
                            continue

                        if side == "producer":
                            indices = [
                                max(positions["body"]),
                                max(positions["else"]),
                            ]
                            if (indices[0] + 1 >= len(node.body) or indices[1] + 1 >= len(node.orelse)):
                                continue
                            method = "producer_commit"
                        else:
                            first_indices = [
                                min(positions["body"]),
                                min(positions["else"]),
                            ]
                            if first_indices[0] and first_indices[1]:
                                key = _pipeline_region_site_key(pipeline, role, region, "consumer_wait")
                                sites.setdefault(key, []).extend((
                                    id(node.body[first_indices[0]]),
                                    id(node.orelse[first_indices[1]]),
                                ))
                            indices = [
                                max(positions["body"]),
                                max(positions["else"]),
                            ]
                            if (indices[0] + 1 >= len(node.body) or indices[1] + 1 >= len(node.orelse)):
                                continue
                            method = "consumer_release"

                        key = _pipeline_region_site_key(pipeline, role, region, method)
                        sites.setdefault(key, []).extend((id(node.body[indices[0]]), id(node.orelse[indices[1]])))
    return {key: tuple(dict.fromkeys(value)) for key, value in sites.items()}


def _producer_acquire_sites(graph: dict) -> dict[tuple, int]:
    sites: dict[tuple, int] = {}
    for pipeline in graph["pipelines"]:
        for role in _producer_roles(pipeline):
            for first, _last, region in _pipeline_region_bounds(graph, pipeline, "producer", role, _writes):
                key = (
                    _pipeline_var_name(pipeline),
                    role,
                    region.key,
                    region.loops,
                    ast.dump(region.value, include_attributes=False),
                )
                sites[key] = first
    return sites


def pipeline_effect_partition_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    """Split repeatable control flow so pipeline calls use precise effect bounds."""
    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY)
    if graphs is None:
        raise ValueError("dataflow analysis result is not found in compilation context")

    changed: list[tuple[ast.FunctionDef, dict[tuple, int], dict[tuple, tuple[int, ...]]]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in graphs:
            continue
        graph = graphs[node.name]
        branch_sites = _uniform_branch_sites(node, graph)
        cuts = _effect_carrier_cuts(node, graph)
        if not cuts:
            if branch_sites:
                graph["pipeline_branch_sites"] = branch_sites
            continue
        _EffectCarrierSplitter(cuts).visit(node)
        changed.append((node, _producer_acquire_sites(graph), branch_sites))

    if changed:
        ast.fix_missing_locations(tree)
        for node, acquire_sites, branch_sites in changed:
            graph = build_dataflow(node, ctx)
            graph["producer_acquire_sites"] = acquire_sites
            graph["pipeline_branch_sites"] = branch_sites
            graphs[node.name] = graph
    return tree


def build_stage_region_insertion_plan(graph: dict) -> StageRegionInsertionPlan:
    """Map each runtime pipeline side to its transaction insertion sites."""
    plan = StageRegionInsertionPlan()
    for pipeline in graph["pipelines"]:
        pipe = _pipeline_var_name(pipeline)
        used = False

        for side, roles, pred in (
            ("producer", _producer_roles(pipeline), _writes),
            ("consumer", _consumer_roles(pipeline), _reads),
        ):
            for role in roles:
                via = _pipeline_side_via(pipeline, side)
                events = _role_mem_events(graph, role, via, pred)
                regions = _pipeline_region_bounds(graph, pipeline, side, role, pred)
                if events and not regions:
                    action = "writes to" if side == "producer" else "reads from"
                    raise ValueError(f"{pipeline['name']}: missing {role} stage_key.bind(...) "
                                     f"around {action} {sorted(via)}")

                for first, last, region in regions:
                    if side == "producer":
                        acquire_key = (
                            pipe,
                            role,
                            region.key,
                            region.loops,
                            ast.dump(region.value, include_attributes=False),
                        )
                        acquire_site = graph.get("producer_acquire_sites", {}).get(acquire_key, first)
                        _add_plan_site(plan.before, acquire_site, pipe, "producer_acquire")
                        commit_sites = graph.get("pipeline_branch_sites", {}).get(
                            _pipeline_region_site_key(pipeline, role, region, "producer_commit"),
                            (last, ),
                        )
                        for site in commit_sites:
                            _add_plan_site(plan.after, site, pipe, "producer_commit")
                    else:
                        wait_sites = graph.get("pipeline_branch_sites", {}).get(
                            _pipeline_region_site_key(pipeline, role, region, "consumer_wait"),
                            (first, ),
                        )
                        release_sites = graph.get("pipeline_branch_sites", {}).get(
                            _pipeline_region_site_key(pipeline, role, region, "consumer_release"),
                            (last, ),
                        )
                        for site in wait_sites:
                            _add_plan_site(plan.before, site, pipe, "consumer_wait")
                        for site in release_sites:
                            _add_plan_site(plan.after, site, pipe, "consumer_release")
                    used = True

        if pipeline["template"] == PIPELINE_TMA_TMA_STORE_SAME_ROLE:
            for role in _consumer_roles(pipeline):
                if not _role_mem_events(graph, role, _pipeline_side_via(pipeline, "consumer"), _reads):
                    continue
                plan.role_tail.setdefault(role, [])
                if pipe not in plan.role_tail[role]:
                    plan.role_tail[role].append(pipe)
        if used:
            plan.live_pipelines.append(pipeline)

    for reuse in graph.get("overlay_reuse_acquires", ()):
        pipe = str(reuse["pipe"])
        via = set(reuse["via"])
        for role in reuse["roles"]:
            events = _role_mem_events(graph, role, via, _writes)
            pipeline = {
                "key": -1,
                "producer_keys": reuse["keys"],
                "producer_via": tuple(via),
                "via": tuple(via),
            }
            regions = _pipeline_region_bounds(graph, pipeline, "producer", role, _writes)
            if events and not regions:
                raise ValueError(f"overlay {reuse['old_mem']}/{reuse['new_mem']}: "
                                 f"missing {role} stage_key.bind(...)")
            for first, _last, _region in regions:
                _add_plan_site(plan.before, first, pipe, "producer_acquire")
    return plan


def dataflow_analysis_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    """Analyze ``with *.dataflow() as g:`` blocks and stash graphs in ``ctx``."""
    ctx = ctx if ctx is not None else {}

    graphs = ctx.setdefault(DATAFLOW_CTX_KEY, {})
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or dataflow_decl(node, allow_missing=True) is None:
            continue
        graphs[node.name] = build_dataflow(node, ctx)
    return tree
