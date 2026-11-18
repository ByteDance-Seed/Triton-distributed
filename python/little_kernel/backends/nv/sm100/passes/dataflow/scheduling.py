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
"""Role/dataflow scheduling analyses (RF05).

Responsibility split of the former monolith: this module owns scheduling
decisions only -- buffer-overlay inference, same-role TMA relay specialization,
TMA store-stream inference and pipeline-protocol validation. It produces plans
consumed by the transform/lowering passes and emits no backend text.
"""

from __future__ import annotations

from little_kernel.backends.nv.sm100.passes.dataflow.common import call_name as _call_name
from little_kernel.backends.nv.sm100.passes.dataflow.constants import DATAFLOW_CTX_KEY
from little_kernel.backends.nv.sm100.core.pipeline import PIPELINE_TMA_STORE_CROSS_ROLE, PIPELINE_TMA_STORE_SAME_ROLE, PIPELINE_TMA_TMA_STORE, PIPELINE_TMA_TMA_STORE_SAME_ROLE
from little_kernel.backends.nv.sm100.passes.dataflow.channels import channel_name as _channel_name, event_text as _event_text, _select_pipeline_template
from little_kernel.core.memory_effects import Access, MP, R_SMEM_TMA_BULK, W_SMEM_TMA_TX

import ast
from dataclasses import dataclass
from typing import Any, Callable

from little_kernel.core.memory_effects import Completion
from little_kernel.backends.nv.sm100.language.dataflow.declarations import Flow
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import BindRegion, MemEvent, _bind_key, _bind_value, _call_args, _call_name, _const_value, _contains_bind_key, _effect_operand_expr, _has_early_exit, _name, _nested_blocks, _ref_vars, _role_context, collect_bind_regions, collect_channel_mem_events, get_intrin_mem_effects  # noqa: F811
from little_kernel.backends.nv.sm100.passes.dataflow.channels import analyze_channels, barrier_participation, consumer_roles as _consumer_roles, producer_roles as _producer_roles, reads as _reads, writes as _writes, side_kind as _side_kind, validate_pipeline_protocols
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import OverlayConstraintKind

from little_kernel.backends.nv.sm100.passes.dataflow.graph_ops import _ident, _pipe_name, _pipeline_region_bounds, _pipeline_side_keys, _pipeline_side_via, _pipeline_stages, _pipeline_var_name, _region_bounds, _role_mem_events


def _bind_keys(graph: dict, role: str, memories: set[str], pred: Callable[[Access], bool]) -> tuple[int, ...]:
    return tuple(
        sorted({
            region.key
            for region in graph["bind_regions"]
            if region.role == role and any(item.event.mem in memories and pred(item.event.access)
                                           for item in region.events)
        }))


def _overlay_order(
    graph: dict,
    role: str,
    mem: str,
    pred: Callable[[Access], bool],
) -> tuple[MemEvent, MemEvent] | None:
    events = sorted(
        (event for event in graph["mem_events"] if event.role == role and event.mem == mem and pred(event.access)),
        key=lambda event: event.order,
    )
    return (events[0], events[-1]) if events else None


def _prove_local_overlay_constraint(graph: dict, constraint) -> None:
    assert constraint.role is not None
    if constraint.kind == OverlayConstraintKind.CONSUMER_RELEASE_BEFORE_WAIT:
        old = _overlay_order(graph, constraint.role, constraint.old_mem, _reads)
        new = _overlay_order(graph, constraint.role, constraint.new_mem, _reads)
        action = "release-before-wait"
    elif constraint.kind == OverlayConstraintKind.TRANSFORM_READ_BEFORE_WRITE:
        old = _overlay_order(graph, constraint.role, constraint.old_mem, _reads)
        new = _overlay_order(graph, constraint.role, constraint.new_mem, _writes)
        action = "read-before-write"
    else:
        raise ValueError(f"unexpected local overlay constraint {constraint.kind}")

    if old is None or new is None:
        raise ValueError(f"overlay {constraint.old_mem}/{constraint.new_mem}: missing events "
                         f"for {constraint.role} {action} proof")
    if constraint.kind == OverlayConstraintKind.TRANSFORM_READ_BEFORE_WRITE:
        if any(event.effect_kind.proxy == MP.ASYNC
               for event in graph["mem_events"]
               if event.role == constraint.role and event.mem == constraint.old_mem and _reads(event.access)):
            raise ValueError(f"overlay {constraint.old_mem}/{constraint.new_mem}: async transform "
                             "reads require an explicit modeled completion fence")
        return

    old_last, new_first = old[1], new[0]
    if old_last.order >= new_first.order or old_last.stmt_id == new_first.stmt_id:
        raise ValueError(f"overlay {constraint.old_mem}/{constraint.new_mem}: cannot prove "
                         f"{constraint.role} {action} at distinct ordered statements")


def _overlay_events_compatible(a: MemEvent, b: MemEvent) -> bool:
    a_loops = tuple(item for item in a.control_path if item[0] != "if")
    b_loops = tuple(item for item in b.control_path if item[0] != "if")
    if a_loops != b_loops:
        return False
    a_arms = {node_id: arm for kind, node_id, arm in a.control_path if kind == "if"}
    return all(a_arms.get(node_id, arm) == arm for kind, node_id, arm in b.control_path if kind == "if")


def _has_transitive_overlay_order(graph: dict, constraint) -> bool:
    for release_role in constraint.release_roles:
        for acquire_role in constraint.acquire_roles:
            ordered = False
            for pipeline in graph["pipelines"]:
                if (release_role not in _producer_roles(pipeline) or acquire_role not in _consumer_roles(pipeline)):
                    continue
                for bridge in pipeline.get("producer_via", pipeline["via"]):
                    old_reads = _role_mem_events(graph, release_role, {constraint.old_mem}, _reads)
                    bridge_writes = _role_mem_events(graph, release_role, {bridge}, _writes)
                    bridge_reads = _role_mem_events(graph, acquire_role, {bridge}, _reads)
                    new_writes = _role_mem_events(graph, acquire_role, {constraint.new_mem}, _writes)
                    old_before_bridge = all(
                        any(
                            _overlay_events_compatible(old, write) and (
                                old.order < write.order or old.stmt_id == write.stmt_id)
                            for write in bridge_writes)
                        for old in old_reads)
                    bridge_before_new = all(
                        any(
                            _overlay_events_compatible(read, new) and (
                                read.order < new.order or read.stmt_id == new.stmt_id)
                            for read in bridge_reads)
                        for new in new_writes)
                    if (old_reads and bridge_writes and bridge_reads and new_writes and old_before_bridge
                            and bridge_before_new):
                        ordered = True
                        break
                if ordered:
                    break
            if not ordered:
                return False
    return True


def infer_overlay_pipelines(graph: dict) -> list[dict]:
    analysis = graph.get("buffer_overlay")
    if analysis is None:
        return []

    states: list[dict[str, object]] = []
    reuse_acquires: list[dict[str, object]] = []
    groups: dict[
        tuple[tuple[str, ...], tuple[str, ...], str, str, str],
        dict[str, set[str]],
    ] = {}
    for constraint in analysis.constraints:
        if constraint.kind in {
                OverlayConstraintKind.CONSUMER_RELEASE_BEFORE_WAIT,
                OverlayConstraintKind.TRANSFORM_READ_BEFORE_WRITE,
        }:
            _prove_local_overlay_constraint(graph, constraint)
            states.append({"constraint": constraint, "state": "PROVEN_LOCAL"})
            continue
        if constraint.kind != OverlayConstraintKind.PRODUCER_WAIT_RELEASE:
            raise ValueError(f"overlay {constraint.old_mem}/{constraint.new_mem}: "
                             f"unknown constraint kind {constraint.kind}")
        if not constraint.acquire_roles or not constraint.release_roles:
            raise ValueError(f"overlay {constraint.old_mem}/{constraint.new_mem} has no "
                             "acquire or release endpoint")
        if _has_transitive_overlay_order(graph, constraint):
            states.append({"constraint": constraint, "state": "PROVEN_TRANSITIVE"})
            continue
        source_pipelines = [
            pipeline for pipeline in graph["pipelines"]
            if not pipeline.get("overlay") and constraint.old_mem in set(pipeline.get("producer_via", pipeline["via"]))
            and tuple(sorted(_producer_roles(pipeline))) == constraint.acquire_roles and set(constraint.release_roles).
            issubset(_consumer_roles(pipeline)) and pipeline["template"].consumer is not None
        ]
        if len(source_pipelines) == 1:
            if _pipeline_stages(graph, source_pipelines[0]) != 1:
                raise ValueError(f"overlay {constraint.old_mem}/{constraint.new_mem}: "
                                 "same-producer reuse currently requires single-stage buffers")
            reuse_acquires.append({
                "pipe":
                _pipeline_var_name(source_pipelines[0]),
                "roles":
                constraint.acquire_roles,
                "via": (constraint.new_mem, ),
                "keys":
                tuple(
                    sorted({
                        key
                        for role in constraint.acquire_roles
                        for key in _bind_keys(graph, role, {constraint.new_mem}, _writes)
                    })),
                "old_mem":
                constraint.old_mem,
                "new_mem":
                constraint.new_mem,
            })
            states.append({"constraint": constraint, "state": "REUSED_SOURCE_RELEASE"})
            continue
        placement = analysis.placements[constraint.new_mem]
        key = (
            constraint.acquire_roles,
            constraint.release_roles,
            placement.space,
            placement.physical,
            constraint.new_mem,
        )
        group = groups.setdefault(key, {"producer": set(), "consumer": set()})
        group["producer"].add(constraint.new_mem)
        group["consumer"].add(constraint.old_mem)
        states.append({"constraint": constraint, "state": "LOWERED"})

    reuse_pipelines: list[dict] = []
    for index, ((producers, consumers, _space, _physical, _new_mem), sides) in enumerate(groups.items()):
        producer_via = sorted(sides["producer"])
        consumer_via = sorted(sides["consumer"])
        via = producer_via + consumer_via
        name = _channel_name(producers, consumers)
        producer_writes = [
            event for role in producers for event in _role_mem_events(graph, role, set(producer_via), _writes)
        ]
        consumer_reads = [
            event for role in consumers for event in _role_mem_events(graph, role, set(consumer_via), _reads)
        ]
        if not producer_writes or not consumer_reads:
            raise ValueError(f"{name}: cannot place physical reuse barrier for {via}")

        producer_keys = tuple(
            sorted({key
                    for role in producers
                    for key in _bind_keys(graph, role, set(producer_via), _writes)}))
        consumer_keys = tuple(
            sorted({key
                    for role in consumers
                    for key in _bind_keys(graph, role, set(consumer_via), _reads)}))
        if not producer_keys or not consumer_keys:
            raise ValueError(f"{name}: physical reuse for {via} has no data stage binding")
        producer_kind = _side_kind(name, "producer", producer_writes, Access.W)
        consumer_kind = _side_kind(name, "consumer", consumer_reads, Access.R)

        reuse_pipelines.append({
            "key":
            -index - 1,
            "name":
            name,
            "producers":
            producers,
            "consumers":
            consumers,
            "via":
            via,
            "template":
            _select_pipeline_template(name, producers, consumers, producer_kind, consumer_kind),
            "producer_kind":
            producer_kind,
            "consumer_kind":
            consumer_kind,
            "producer_writes":
            _event_text(producer_writes),
            "consumer_reads":
            _event_text(consumer_reads),
            "producer_keys":
            producer_keys,
            "consumer_keys":
            consumer_keys,
            "producer_via":
            producer_via,
            "consumer_via":
            consumer_via,
            "overlay":
            True,
        })

    graph["pipelines"] = [pipeline for pipeline in graph["pipelines"] if not pipeline.get("overlay")] + reuse_pipelines
    graph["overlay_reuse_acquires"] = tuple(reuse_acquires)
    graph["overlay_constraint_states"] = tuple(states)
    return reuse_pipelines


def overlay_pipeline_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY)
    if graphs is None:
        raise ValueError("overlay pipeline inference requires dataflow analysis first")
    for graph in graphs.values():
        infer_overlay_pipelines(graph)
    return tree


def _event_key(event: MemEvent) -> tuple:
    return event.role, event.mem, event.op_id, event.access


@dataclass(frozen=True)
class TmaStoreSite:
    domain_id: int
    site_id: int


@dataclass(frozen=True)
class IssuerAnalysis:
    parents: dict[int, ast.AST]
    blocks: dict[int, list[ast.stmt]]
    definitions: dict[str, list[tuple[ast.stmt, ast.expr]]]
    warp_domain_names: set[str]


def _issuer_analysis(func: ast.FunctionDef) -> IssuerAnalysis:
    parents = {id(child): node for node in ast.walk(func) for child in ast.iter_child_nodes(node)}
    blocks = {
        id(stmt): body
        for node in ast.walk(func)
        for _field, body in ast.iter_fields(node)
        if isinstance(body, list) and all(isinstance(stmt, ast.stmt) for stmt in body) for stmt in body
    }
    definitions: dict[str, list[tuple[ast.stmt, ast.expr]]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.stmt) and (item := _assignment(node)) is not None:
            definitions.setdefault(item[0], []).append((node, item[1]))
    thread_index_names: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, entries in definitions.items():
            if len(entries) != 1 or name in thread_index_names:
                continue
            value = entries[0][1]
            direct = _call_name(value) == "threadIdx_x"
            alias = isinstance(value, ast.Name) and value.id in thread_index_names
            affine = (isinstance(value, ast.BinOp) and isinstance(value.op, (ast.Add, ast.Sub))
                      and isinstance(value.left, ast.Name) and value.left.id in thread_index_names
                      and isinstance(value.right, ast.Constant))
            if direct or alias or affine:
                thread_index_names.add(name)
                changed = True
    warp_domain_names = {
        name
        for name, entries in definitions.items()
        if len(entries) == 1 and isinstance(entries[0][1], ast.BinOp) and isinstance(entries[0][1].op, ast.FloorDiv)
        and isinstance(entries[0][1].right, ast.Constant) and entries[0][1].right.value == 32 and (
            _call_name(entries[0][1].left) == "threadIdx_x" or
            (isinstance(entries[0][1].left, ast.Name) and entries[0][1].left.id in thread_index_names))
    }
    return IssuerAnalysis(parents, blocks, definitions, warp_domain_names)


def _is_issuer_domain_expr(
        value: ast.expr,
        analysis: IssuerAnalysis,
        seen: frozenset[str] = frozenset(),
) -> bool:
    if _call_name(value) in {"elect_one", "get_lane_idx", "threadIdx_x"}:
        return True
    names = _ref_vars(value) - {"ll"}
    for name in names:
        definitions = analysis.definitions.get(name, ())
        if (name not in seen and len(definitions) == 1
                and _is_issuer_domain_expr(definitions[0][1], analysis, seen | {name})):
            return True
    return False


def _resolve_issuer_site(
    event: MemEvent,
    call: ast.Call,
    analysis: IssuerAnalysis,
    *,
    require_elected: bool,
) -> tuple[str, ast.stmt, list[ast.stmt], ast.stmt]:
    candidates: list[tuple[bool, int, str, ast.stmt]] = []
    child: ast.AST = call
    distance = 0
    while (parent := analysis.parents.get(id(child))) is not None:
        distance += 1
        if isinstance(parent, ast.If) and child in parent.body:
            guard = _issuer_guard(parent.test)
            if guard is not None:
                defs = analysis.definitions.get(guard, ())
                if len(defs) == 1 and _is_issuer_domain_expr(defs[0][1], analysis):
                    definition, value = defs[0]
                    candidates.append((_call_name(value) == "elect_one", distance, guard, definition))
        child = parent
    if not candidates:
        raise ValueError(f"{event.role}:{event.intrin}:{event.mem}: {event.effect_kind.value} "
                         "must be dominated by a named issuer guard")
    elected = [item for item in candidates if item[0]]
    if require_elected and not elected:
        raise ValueError(f"{event.role}:{event.intrin}:{event.mem}: TMA load issuer must be "
                         "defined by ll.elect_one()")
    _is_elected, _distance, name, definition = min(elected or candidates, key=lambda item: item[1])
    ancestor = analysis.parents.get(id(definition))
    while ancestor is not None and not isinstance(ancestor, ast.With):
        if isinstance(ancestor, ast.If):
            guard_names = _ref_vars(ancestor.test)
            if (not guard_names or not guard_names <= analysis.warp_domain_names):
                raise ValueError(f"{event.role}:{name}: issuer definition must execute "
                                 "before any payload predicate")
        ancestor = analysis.parents.get(id(ancestor))
    domain = analysis.blocks.get(id(definition))
    if domain is None:
        raise ValueError(f"{event.role}:{name}: issuer definition has no block")

    definition_ancestors: set[int] = set()
    ancestor = analysis.parents.get(id(definition))
    while ancestor is not None:
        definition_ancestors.add(id(ancestor))
        ancestor = analysis.parents.get(id(ancestor))
    child: ast.AST = call
    while (parent := analysis.parents.get(id(child))) is not None:
        if isinstance(parent, (ast.For, ast.While)) and id(parent) not in definition_ancestors:
            # A payload loop inside an issuer guard executes on only one lane.
            # Keep all of its stores in one convergent group outside that guard;
            # otherwise commit/retire advances only the issuer's pipeline state.
            ancestor = analysis.parents.get(id(parent))
            guarded = False
            while ancestor is not None and id(ancestor) not in definition_ancestors:
                if isinstance(ancestor, ast.If):
                    names = _ref_vars(ancestor.test)
                    if not names or not names <= analysis.warp_domain_names:
                        guarded = True
                        break
                ancestor = analysis.parents.get(id(ancestor))
            if not guarded:
                domain = parent.body
                break
        child = parent
    sites = [stmt for stmt in domain if any(id(node) == id(call) for node in ast.walk(stmt))]
    if len(sites) != 1:
        raise ValueError(f"{event.role}:{name}: cannot locate a unique TMA submission site")
    return name, definition, domain, sites[0]


def _issuer_guard(test: ast.expr) -> str | None:
    if isinstance(test, ast.Name):
        return test.id
    if (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
        left, right = test.left, test.comparators[0]
        if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
            left, right = right, left
        if not isinstance(left, ast.Name) or not isinstance(right, ast.Constant):
            return None
        if isinstance(test.ops[0], (ast.Eq, ast.NotEq)) and right.value in {0, 1}:
            return left.id
    return None


def _assignment(stmt: ast.stmt) -> tuple[str, ast.expr] | None:
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
        return stmt.targets[0].id, stmt.value
    if (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None):
        return stmt.target.id, stmt.value
    return None


def _tma_store_sites(func: ast.FunctionDef, graph: dict) -> dict[tuple, TmaStoreSite]:
    """Resolve each TMA store to a convergent group-commit site."""
    analysis = _issuer_analysis(func)
    calls = {id(node): node for node in ast.walk(func) if isinstance(node, ast.Call)}
    sites_by_event: dict[tuple, TmaStoreSite] = {}
    for event in graph["mem_events"]:
        if event.effect_kind != R_SMEM_TMA_BULK or not _reads(event.access):
            continue
        call = calls.get(event.op_id[0])
        if call is None:
            raise ValueError(f"{event.role}:{event.mem}: missing TMA store AST node")
        _, _definition, domain, site = _resolve_issuer_site(event, call, analysis, require_elected=False)
        sites_by_event[_event_key(event)] = TmaStoreSite(id(domain), id(site))
    return sites_by_event


def specialize_same_role_tma_relays(
    func: ast.FunctionDef,
    graph: dict,
    store_sites: dict[tuple, TmaStoreSite],
) -> None:
    """Use the one-cursor relay when one warp owns one load/store stream."""
    analysis = _issuer_analysis(func)
    calls = {id(node): node for node in ast.walk(func) if isinstance(node, ast.Call)}
    for pipeline in graph["pipelines"]:
        if pipeline["template"] != PIPELINE_TMA_TMA_STORE or pipeline.get("overlay"):
            continue
        producers, consumers = _producer_roles(pipeline), _consumer_roles(pipeline)
        if len(producers) != 1 or producers != consumers or _pipeline_stages(graph, pipeline) > 8:
            continue
        role = producers[0]
        via = set(pipeline["via"])
        loads = [
            event for event in graph["mem_events"]
            if event.role == role and event.mem in via and _writes(event.access) and event.effect_kind == W_SMEM_TMA_TX
        ]
        stores = [
            event for event in graph["mem_events"]
            if event.role == role and event.mem in via and _reads(event.access) and event.effect_kind == R_SMEM_TMA_BULK
        ]
        load_calls = {event.op_id[0] for event in loads}
        store_calls = {event.op_id[0] for event in stores}
        all_role_store_calls = {
            event.op_id[0]
            for event in graph["mem_events"]
            if event.role == role and _reads(event.access) and event.effect_kind == R_SMEM_TMA_BULK
        }
        if len(load_calls) != 1 or len(store_calls) != 1 or store_calls != all_role_store_calls:
            continue
        if any(event.cta_group != 1 or event.cluster_multicast for event in (*loads, *stores)):
            continue
        producer_regions = _pipeline_region_bounds(graph, pipeline, "producer", role, _writes)
        consumer_regions = _pipeline_region_bounds(graph, pipeline, "consumer", role, _reads)
        if len(producer_regions) != 1 or len(consumer_regions) != 1:
            continue
        if max(event.order for event in loads) >= min(event.order for event in stores):
            continue
        if {event.control_path for event in loads} != {event.control_path for event in stores}:
            continue
        producer_region, consumer_region = producer_regions[0][2], consumer_regions[0][2]
        producer_key = (producer_region.key, producer_region.scope_id,
                        ast.dump(producer_region.value, include_attributes=False), producer_region.loops)
        consumer_key = (consumer_region.key, consumer_region.scope_id,
                        ast.dump(consumer_region.value, include_attributes=False), consumer_region.loops)
        if producer_key != consumer_key:
            continue
        load_event = loads[0]
        load_call = calls.get(next(iter(load_calls)))
        if load_call is None:
            continue
        _, _definition, load_domain, _site = _resolve_issuer_site(load_event, load_call, analysis, require_elected=True)
        store_domains = {store_sites[_event_key(event)].domain_id for event in stores}
        if store_domains != {id(load_domain)}:
            continue
        pipeline["template"] = PIPELINE_TMA_TMA_STORE_SAME_ROLE


def _validate_tma_load_issuers(func: ast.FunctionDef, graph: dict) -> None:
    """Require one statically elected TMA-load issuer domain per producer role."""
    analysis = _issuer_analysis(func)
    calls = {id(node): node for node in ast.walk(func) if isinstance(node, ast.Call)}
    domains: dict[str, int] = {}
    visited_calls: set[int] = set()
    for event in graph["mem_events"]:
        if event.effect_kind != W_SMEM_TMA_TX or not _writes(event.access):
            continue
        call_id = event.op_id[0]
        if call_id in visited_calls:
            continue
        visited_calls.add(call_id)
        role_warps = len(graph["roles"][event.role].warps)
        if role_warps != 1:
            raise ValueError(f"{event.role}:{event.intrin}:{event.mem}: TMA-load producer "
                             f"roles must own exactly one warp, got {role_warps}")
        call = calls.get(call_id)
        if call is None:
            raise ValueError(f"{event.role}:{event.mem}: missing TMA load AST node")
        _, _definition, domain, _site = _resolve_issuer_site(event, call, analysis, require_elected=True)
        previous = domains.setdefault(event.role, id(domain))
        if previous != id(domain):
            raise ValueError(f"{event.role}: TMA loads must share one ll.elect_one() "
                             "issuer domain")


def _issuer_control_id(
    event: MemEvent,
    call: ast.Call,
    analysis: IssuerAnalysis,
) -> int:
    """Return the source control id for the selected TMA issuer guard."""
    name, _definition, _domain, _site = _resolve_issuer_site(event, call, analysis,
                                                             require_elected=event.effect_kind == W_SMEM_TMA_TX)
    child: ast.AST = call
    while (parent := analysis.parents.get(id(child))) is not None:
        if (isinstance(parent, ast.If) and child in parent.body and _issuer_guard(parent.test) == name):
            return getattr(parent, "_lk_control_origin", id(parent))
        child = parent
    raise ValueError(f"{event.role}:{name}: cannot locate issuer guard")


def _transaction_control_path(
    event: MemEvent,
    controls: dict[int, ast.AST],
    calls: dict[int, ast.Call],
    analysis: IssuerAnalysis,
    exhaustive: set[int],
) -> tuple[tuple[str, str, str], ...]:
    """Remove topology-only guards, leaving dynamic transaction predicates."""
    ignored: set[int] = set()
    if event.effect_kind in {W_SMEM_TMA_TX, R_SMEM_TMA_BULK}:
        call = calls.get(event.op_id[0])
        if call is None:
            raise ValueError(f"{event.role}:{event.mem}: missing TMA AST node")
        ignored.add(_issuer_control_id(event, call, analysis))

    result: list[tuple[str, str, str]] = []
    for kind, control_id, arm in event.control_path:
        if kind in {"for", "unroll"}:
            continue
        node = controls.get(control_id)
        if isinstance(node, ast.If):
            if control_id in ignored or control_id in exhaustive:
                continue
            guard_names = _ref_vars(node.test)
            if guard_names and guard_names <= analysis.warp_domain_names:
                continue
            guard = _issuer_guard(node.test)
            definitions = () if guard is None else analysis.definitions.get(guard, ())
            if len(definitions) == 1 and _call_name(definitions[0][1]) == "elect_one":
                continue
            value = node.test
        elif isinstance(node, ast.While):
            value = node.test
        elif isinstance(node, ast.For):
            value = node.iter
        else:
            result.append((kind, str(control_id), arm))
            continue
        result.append((kind, ast.dump(value, include_attributes=False), arm))
    return tuple(result)


def _exhaustive_transaction_controls(items: list[tuple[int, MemEvent]]) -> set[int]:
    arms: dict[int, set[str]] = {}
    for _index, event in items:
        for kind, control_id, arm in event.control_path:
            if kind == "if":
                arms.setdefault(control_id, set()).add(arm)
    return {control_id for control_id, covered in arms.items() if {"body", "else"} <= covered}


def _transaction_region_key(region: BindRegion) -> tuple:
    return (
        region.key,
        region.scope_id,
        ast.dump(region.value, include_attributes=False),
        region.loops,
    )


def _validate_same_role_tma_store_transactions(func: ast.FunctionDef, graph: dict) -> None:
    """Reject same-role transactions whose producer cannot match the TMA store."""
    analysis = _issuer_analysis(func)
    calls = {id(node): node for node in ast.walk(func) if isinstance(node, ast.Call)}
    controls: dict[int, ast.AST] = {}
    for node in ast.walk(func):
        if isinstance(node, (ast.If, ast.For, ast.While)):
            controls.setdefault(getattr(node, "_lk_control_origin", id(node)), node)

    for pipeline in graph["pipelines"]:
        producers, consumers = _producer_roles(pipeline), _consumer_roles(pipeline)
        if (pipeline.get("overlay") or pipeline["consumer_kind"] != R_SMEM_TMA_BULK or len(producers) != 1
                or producers != consumers):
            continue
        role = producers[0]
        producer_regions = {
            _transaction_region_key(region): region
            for _first, _last, region in _pipeline_region_bounds(graph, pipeline, "producer", role, _writes)
        }
        consumer_regions = {
            _transaction_region_key(region): region
            for _first, _last, region in _pipeline_region_bounds(graph, pipeline, "consumer", role, _reads)
        }
        if producer_regions.keys() != consumer_regions.keys():
            raise ValueError(f"{pipeline['name']}: illegal same-role transaction bind/cardinality mismatch")

        producer_via = _pipeline_side_via(pipeline, "producer")
        consumer_via = _pipeline_side_via(pipeline, "consumer")
        for key, producer_region in producer_regions.items():
            consumer_region = consumer_regions[key]
            producer_items = [(index, item.event)
                              for index, item in enumerate(producer_region.events)
                              if item.event.mem in producer_via and _writes(item.event.access)]
            consumer_items = [(index, item.event)
                              for index, item in enumerate(consumer_region.events)
                              if (item.event.mem in consumer_via and _reads(item.event.access)
                                  and item.event.effect_kind == R_SMEM_TMA_BULK)]
            if max(index for index, _event in producer_items) >= min(index for index, _event in consumer_items):
                raise ValueError(f"{pipeline['name']}: illegal same-role transaction requires all producer "
                                 "effects before TMA store")

            producer_exhaustive = _exhaustive_transaction_controls(producer_items)
            consumer_exhaustive = _exhaustive_transaction_controls(consumer_items)
            producer_paths = {
                _transaction_control_path(event, controls, calls, analysis, producer_exhaustive)
                for _index, event in producer_items
            }
            consumer_paths = {
                _transaction_control_path(event, controls, calls, analysis, consumer_exhaustive)
                for _index, event in consumer_items
            }
            if producer_paths != consumer_paths:
                raise ValueError(f"{pipeline['name']}: illegal same-role transaction "
                                 "control-path/cardinality mismatch")


def infer_tma_store_streams(graph: dict, store_sites: dict[tuple, TmaStoreSite]) -> list[dict]:
    """Build one statically scheduled bulk-group stream per issuer domain."""
    store_pipelines = [
        pipeline for pipeline in graph["pipelines"] if pipeline["template"] in {
            PIPELINE_TMA_STORE_SAME_ROLE,
            PIPELINE_TMA_STORE_CROSS_ROLE,
            PIPELINE_TMA_TMA_STORE,
        }
    ]
    by_role: dict[str, list[dict]] = {}
    for pipeline in store_pipelines:
        for role in _consumer_roles(pipeline):
            by_role.setdefault(role, []).append(pipeline)

    global_events = {_event_key(event): event for event in graph["mem_events"]}
    site_by_call = {key[2][0]: site for key, site in store_sites.items()}

    def site_for(event: MemEvent) -> TmaStoreSite:
        site = store_sites.get(_event_key(event))
        if site is None:
            site = site_by_call.get(event.op_id[0])
        if site is None:
            raise ValueError(f"{event.role}:{event.intrin}:{event.mem}: missing TMA issuer")
        return site

    streams: list[dict] = []
    completion_domains: dict[str, set[tuple[str, int]]] = {}
    for role, pipelines in by_role.items():
        records: list[dict] = []
        submitted_calls: set[int] = set()
        for pipeline in pipelines:
            via = _pipeline_side_via(pipeline, "consumer")
            for first, last, region in _pipeline_region_bounds(graph, pipeline, "consumer", role, _reads):
                events: list[MemEvent] = []
                for item in region.events:
                    event = item.event
                    if (event.mem not in via or not _reads(event.access) or event.effect_kind != R_SMEM_TMA_BULK):
                        continue
                    key = _event_key(event)
                    instances = ([global_events[key]] if key in global_events else [
                        candidate for candidate in graph["mem_events"]
                        if candidate.role == event.role and candidate.mem == event.mem
                        and candidate.op_id[0] == event.op_id[0] and candidate.access == event.access
                    ])
                    events.extend(instances or [event])
                    submitted_calls.add(event.op_id[0])
                events = list({_event_key(event): event for event in events}.values())
                if not events:
                    continue
                by_domain: dict[int, list[MemEvent]] = {}
                for event in events:
                    domain = site_for(event).domain_id
                    by_domain.setdefault(domain, []).append(event)
                for domain, domain_events in by_domain.items():
                    records.append({
                        "pipeline": _pipeline_var_name(pipeline),
                        "domain": domain,
                        "first": site_for(domain_events[0]).site_id,
                        "last": site_for(domain_events[-1]).site_id,
                        "wait": first,
                        "release": last,
                        "same_role": (set(_producer_roles(pipeline)) == set(_consumer_roles(pipeline))),
                        "first_order": min(event.order for event in domain_events),
                        "last_order": max(event.order for event in domain_events),
                    })

        if not records:
            continue

        completions_by_site: dict[tuple[int, int], set[str]] = {}
        boundary_order: dict[tuple[int, int], int] = {}
        for record in records:
            boundary = (record["domain"], record["last"])
            completions_by_site.setdefault(boundary, set()).add(record["pipeline"])
            boundary_order[boundary] = max(boundary_order.get(boundary, -1), record["last_order"])

        role_events = sorted(
            (event for event in graph["mem_events"]
             if event.role == role and event.effect_kind == R_SMEM_TMA_BULK and _reads(event.access)),
            key=lambda event: event.order,
        )
        missing = [event for event in role_events if event.op_id[0] not in submitted_calls]
        if missing:
            detail = ", ".join(f"{event.intrin}:{event.mem}" for event in missing)
            raise ValueError(f"{role}: TMA store effects have no group submission: {detail}")

        groups_by_domain: dict[int, list[dict]] = {}
        previous_order: dict[int, int] = {}
        for boundary in sorted(completions_by_site, key=boundary_order.__getitem__):
            domain, site = boundary
            last_order = boundary_order[boundary]
            begin_order = previous_order.get(domain, -1)
            group_events = [
                event for event in role_events
                if site_for(event).domain_id == domain and begin_order < event.order <= last_order
            ]
            if not group_events:
                raise ValueError(f"{role}: TMA store completion frontier has no operations")
            completions = tuple(sorted(completions_by_site[boundary]))
            groups_by_domain.setdefault(domain, []).append({
                "first": site_for(group_events[0]).site_id,
                "last": site,
                "last_order": last_order,
                "completions": completions,
                "domain": domain,
            })
            previous_order[domain] = last_order

        if any(event.order > previous_order.get(site_for(event).domain_id, -1) for event in role_events):
            raise ValueError(f"{role}: TMA store operations remain after the last completion frontier")

        pipeline_by_var = {_pipeline_var_name(pipeline): pipeline for pipeline in pipelines}
        for pipe, pipeline in pipeline_by_var.items():
            stages = _pipeline_stages(graph, pipeline)
            if stages > 8:
                raise ValueError(f"{pipe}: TMA store pipeline depth {stages} exceeds 8")

        producer_intervals: dict[
            str,
            list[tuple[str, int, int, int, int]],
        ] = {}
        for pipe, pipeline in pipeline_by_var.items():
            via = _pipeline_side_via(pipeline, "producer")
            for producer in _producer_roles(pipeline):
                for first, last, region in _pipeline_region_bounds(graph, pipeline, "producer", producer, _writes):
                    events = [
                        global_events.get(_event_key(item.event), item.event)
                        for item in region.events
                        if item.event.mem in via and _writes(item.event.access)
                    ]
                    if events:
                        producer_intervals.setdefault(pipe, []).append((
                            producer,
                            first,
                            last,
                            min(event.order for event in events),
                            max(event.order for event in events),
                        ))

        parent = {pipe: pipe for pipe in pipeline_by_var}

        def root(pipe: str) -> str:
            while parent[pipe] != pipe:
                parent[pipe] = parent[parent[pipe]]
                pipe = parent[pipe]
            return pipe

        def union(left: str, right: str) -> None:
            left_root, right_root = root(left), root(right)
            if left_root != right_root:
                parent[right_root] = left_root

        pipe_names = list(pipeline_by_var)
        for index, left in enumerate(pipe_names):
            for right in pipe_names[index + 1:]:
                if any(left_role == right_role and (left_first == right_first or left_last == right_last
                                                    or not (left_end < right_begin or right_end < left_begin)) for (
                                                        left_role,
                                                        left_first,
                                                        left_last,
                                                        left_begin,
                                                        left_end,
                                                    ) in producer_intervals.get(left, ()) for (
                                                        right_role,
                                                        right_first,
                                                        right_last,
                                                        right_begin,
                                                        right_end,
                                                    ) in producer_intervals.get(right, ())):
                    union(left, right)

        coupled = {pipe: {other for other in pipe_names if root(other) == root(pipe)} for pipe in pipe_names}

        components = {tuple(sorted(members)) for members in coupled.values() if len(members) > 1}

        ordered_domains = sorted(groups_by_domain, key=lambda domain: groups_by_domain[domain][0]["last_order"])
        for stream_index, domain in enumerate(ordered_domains):
            groups = groups_by_domain[domain]
            group_index = {group["last"]: index for index, group in enumerate(groups)}
            sites: dict[tuple[int, bool], dict[str, object]] = {}

            def prepare(
                    site: int,
                    position: int,
                    *,
                    pipelines: set[str] | tuple[str, ...] = (),
                    capacity: bool = False,
                    late: bool = False,
            ) -> None:
                item = sites.setdefault((site, late), {"position": position, "pipelines": set(), "capacity": False})
                if item["position"] != position:
                    raise ValueError(f"{role}: ambiguous TMA store stream position")
                item["pipelines"].update(pipelines)
                item["capacity"] = bool(item["capacity"] or capacity)

            for index, group in enumerate(groups):
                prepare(group["first"], index, capacity=True)

            for record in records:
                if record["domain"] != domain:
                    continue
                prepare(
                    record["release"] if record["same_role"] else record["wait"],
                    group_index[record["last"]] + int(record["same_role"]),
                    pipelines={record["pipeline"]},
                    late=record["same_role"],
                )

            domain_pipelines = {pipe for group in groups for pipe in group["completions"]}
            for component in components:
                indices = [index for index, group in enumerate(groups) if set(group["completions"]) & set(component)]
                if indices:
                    index = min(indices)
                    prepare(groups[index]["first"], index, pipelines=set(component) & domain_pipelines)

            window = 8
            period = len(groups)
            for pipe in domain_pipelines:
                stages = _pipeline_stages(graph, pipeline_by_var[pipe])
                occurrences = [index for index, group in enumerate(groups) if pipe in group["completions"]]
                sequence = [index + cycle * period for cycle in range(stages + 1) for index in occurrences]
                window = min(
                    window,
                    min(sequence[index] - sequence[index - stages] for index in range(stages, len(sequence))),
                )

            suffix = "" if len(ordered_domains) == 1 else f"_{stream_index}"
            name = f"__lk_tma_store_stream_{_ident(role)}{suffix}"
            for group in groups:
                group["stream"] = name
                for pipe in group["completions"]:
                    completion_domains.setdefault(pipe, set()).add((role, domain))
            streams.append({
                "role":
                role,
                "name":
                name,
                "groups":
                tuple(groups),
                "window":
                window,
                "prepare":
                tuple({
                    "site": site,
                    "late": late,
                    "position": item["position"],
                    "capacity": item["capacity"],
                    "pipelines": tuple(sorted(set(item["pipelines"]) & domain_pipelines)),
                }
                      for (site, late), item in sites.items()
                      if item["capacity"] or set(item["pipelines"]) & domain_pipelines),
            })

    graph["tma_store_streams"] = streams
    graph["tma_store_completion_domains"] = {pipe: len(domains) for pipe, domains in completion_domains.items()}
    all_pipelines = {_pipeline_var_name(pipeline): pipeline for pipeline in graph["pipelines"]}
    for pipe, count in graph["tma_store_completion_domains"].items():
        if pipe not in all_pipelines:
            raise ValueError(f"{pipe}: completion domain has no runtime pipeline")
        all_pipelines[pipe]["completion_domains"] = count
    return streams


def tma_store_stream_analysis_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY)
    if graphs is None:
        raise ValueError("TMA store stream analysis requires dataflow analysis first")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in graphs:
            graph = graphs[node.name]
            sites = _tma_store_sites(node, graph)
            graph["tma_store_sites"] = sites
            _validate_same_role_tma_store_transactions(node, graph)
            specialize_same_role_tma_relays(node, graph, sites)
            infer_tma_store_streams(graph, sites)
    return tree


############################################################################
# Pipeline primitive insertion


def pipeline_protocol_validation_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    """Validate source issuer domains and resolved runtime participation."""
    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY)
    if graphs is None:
        raise ValueError("pipeline protocol validation requires dataflow analysis")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in graphs:
            graph = graphs[node.name]
            _validate_tma_load_issuers(node, graph)
            validate_pipeline_protocols(graph)
    return tree
