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
"""Role/dataflow frontend: build the typed graph from authored source (RF05).

Responsibility split of the former 4,413-line monolith. This module executes
only the ``ll.constexpr``/``ll.const`` prefix and the ``with dataflow()``
declaration block (never arbitrary example source at import time), then runs
the analyses/scheduling needed to produce the immutable compilation graph.
"""

from __future__ import annotations

import ast
import logging
import math
from typing import TYPE_CHECKING, Any

from little_kernel.backends.nv.sm100.language.dataflow.declarations import Flow, SMemLayoutPlan, StageKey, TMATransferPlan
from little_kernel.backends.nv.sm100.passes.dataflow.constants import DATAFLOW_CTX_KEY, CLUSTER_SHAPE_CTX_KEY, CTA_GROUP_SHAPE_CTX_KEY, DEFAULT_TMA_TRANSFER_PLAN
from little_kernel.backends.nv.sm100.passes.dataflow.channels import analyze_channels, dtype_size as _dtype_size, infer_tma_payload_bytes, lower_mpmc_pipelines, validate_channels, validate_fixed_payload_regions
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import analyze_stage_regions
from little_kernel.backends.nv.sm100.passes.dataflow.logging import configure_logging_from_env, get_logger
from little_kernel.backends.nv.sm100.passes.dataflow.common import call_name as _call_name
from little_kernel.core.memory_effects import R_SMEM_TCGEN05_COMMIT

if TYPE_CHECKING:
    from little_kernel.core.compile import LLKernel

logger = get_logger("dataflow.frontend")
configure_logging_from_env()


def _validate_cluster_requirements(
    cluster_shape: tuple[int, int, int] | None,
    cta_group_shape: tuple[int, int, int] | None,
    *,
    cta_group: int,
    uses_cluster_multicast: bool,
) -> None:
    if cluster_shape is not None and math.prod(cluster_shape) > 16:
        raise ValueError(f"cluster_shape {cluster_shape} exceeds the supported 16-CTA cluster limit")
    if cta_group == 1 and not uses_cluster_multicast:
        return
    if cta_group_shape is None:
        raise ValueError(f"cta_group::{cta_group} requires explicit cta_group_shape on @ll_kernel")
    if cluster_shape is None:
        feature = "cluster multicast" if uses_cluster_multicast else f"cta_group::{cta_group}"
        raise ValueError(f"{feature} requires an explicit cluster_shape on @ll_kernel")
    if (n := math.prod(cta_group_shape)) != cta_group:
        raise ValueError(f"{cta_group_shape} has {n} CTAs, but the kernel uses cta_group::{cta_group}")
    if any(c % g for c, g in zip(cluster_shape, cta_group_shape)):
        raise ValueError(f"cta_group_shape {cta_group_shape} must divide cluster_shape {cluster_shape}")
    if uses_cluster_multicast and math.prod(cluster_shape) == 1:
        raise ValueError("cluster multicast requires cluster_shape with more than one CTA")


def _analyze_transfer_plan(
    plan: TMATransferPlan,
    cluster_shape: tuple[int, int, int],
    cta_group_shape: tuple[int, int, int],
) -> dict:
    """Enumerate the small static cluster topology for one TMA plan."""
    if cta_group_shape not in {(1, 1, 1), (2, 1, 1)}:
        raise ValueError("planned TMA currently supports only cta_group_shape=(1,1,1) "
                         f"or (2,1,1), got {cta_group_shape}")
    group_grid = tuple(c // g for c, g in zip(cluster_shape, cta_group_shape))
    axis = (None if plan.mcast_group_dim is None else {"x": 0, "y": 1, "z": 2}[plan.mcast_group_dim])
    multicast = axis is not None and group_grid[axis] > 1
    effective_axis = axis if multicast else None

    cluster_ctas = math.prod(cluster_shape)
    group_size = math.prod(cta_group_shape)
    masks: list[int] = []
    issuers: list[bool] = []
    group_indices: list[int] = []
    group_peers: list[tuple[int, ...]] = []
    for rank in range(cluster_ctas):
        x = rank % cluster_shape[0]
        y = (rank // cluster_shape[0]) % cluster_shape[1]
        z = rank // (cluster_shape[0] * cluster_shape[1])
        local = (
            x % cta_group_shape[0],
            y % cta_group_shape[1],
            z % cta_group_shape[2],
        )
        group_coord = (
            x // cta_group_shape[0],
            y // cta_group_shape[1],
            z // cta_group_shape[2],
        )
        group_index = group_coord[0] + group_grid[0] * (group_coord[1] + group_grid[1] * group_coord[2])
        group_indices.append(group_index)
        issuers.append(effective_axis is None or group_coord[effective_axis] == 0)

        peers: list[int] = []
        mask = 0
        values = (range(group_grid[effective_axis]) if effective_axis is not None else (0, ))
        for value in values:
            destination_group = list(group_coord)
            if effective_axis is not None:
                destination_group[effective_axis] = value
            peer_group = destination_group[0] + group_grid[0] * (destination_group[1] +
                                                                 group_grid[1] * destination_group[2])
            peers.append(peer_group)
            dst_x = destination_group[0] * cta_group_shape[0] + local[0]
            dst_y = destination_group[1] * cta_group_shape[1] + local[1]
            dst_z = destination_group[2] * cta_group_shape[2] + local[2]
            dst_rank = dst_x + cluster_shape[0] * (dst_y + cluster_shape[1] * dst_z)
            mask |= 1 << dst_rank
        masks.append(mask)
        group_peers.append(tuple(peers))

    return {
        "group_grid": group_grid,
        "group_size": group_size,
        "multicast": multicast,
        "group_indices": tuple(group_indices),
        "group_peers_by_rank": tuple(group_peers),
        "issue_pred_by_rank": tuple(issuers),
        "multicast_mask_by_rank": tuple(masks),
    }


def build_kernel_ctx(kernel: LLKernel, param_values: dict | None = None) -> dict:
    """Compile namespace: ``kernel.ctx`` globals + concrete parameter / constexpr values."""
    ctx = kernel.ctx.copy()
    if param_values:
        ctx.update(param_values)
    return ctx


def dataflow_decl(func: ast.FunctionDef, *, allow_missing: bool = False) -> ast.With | None:
    matches: list[ast.With] = []
    for stmt in func.body:
        if not isinstance(stmt, ast.With) or len(stmt.items) != 1:
            continue

        if _call_name(stmt.items[0].context_expr) == "dataflow":
            matches.append(stmt)

    assert len(matches) <= 1, f"{func.name}: expected at most one `dataflow()` block"

    if len(matches) == 0:
        if allow_missing and not matches:
            return None
        raise ValueError(f"{func.name}: expected exactly one `with lg.dataflow() as g:` block")
    return matches[0]


def _annotation_name(ann: ast.expr | None) -> str | None:
    if isinstance(ann, ast.Subscript):
        ann = ann.value
    match ann:
        case ast.Name(id=name) | ast.Attribute(attr=name) if name in {"constexpr", "const"}:
            return name
        case _:
            return None


def _is_compile_time_annassign(stmt: ast.stmt) -> bool:
    if not isinstance(stmt, ast.AnnAssign) or stmt.value is None:
        return False
    return _annotation_name(stmt.annotation) in {"constexpr", "const"}


def _constexpr_prefix_stmts(func: ast.FunctionDef, dataflow_scope: ast.With) -> list[ast.stmt]:
    """``AnnAssign`` with ``ll.constexpr`` / ``ll.const`` before ``dataflow``."""
    prefix = func.body[:func.body.index(dataflow_scope)]
    return [stmt for stmt in prefix if _is_compile_time_annassign(stmt)]


def build_dataflow(func: ast.FunctionDef, ctx: dict) -> dict:
    dataflow_scope = dataflow_decl(func)
    dataflow_idx = func.body.index(dataflow_scope)
    init_insert_idx = dataflow_idx + 1

    if not isinstance(dataflow_scope.items[0].optional_vars, ast.Name):
        raise ValueError(f"{func.name}: dataflow block must use `with ... as <name>`")
    dataflow_scope_name = dataflow_scope.items[0].optional_vars.id

    decl_body = _constexpr_prefix_stmts(func, dataflow_scope) + [dataflow_scope]
    module = ast.Module(body=decl_body, type_ignores=[])
    ast.fix_missing_locations(module)

    envs = dict(ctx)
    exec(compile(module, f"<{func.name}-dataflow>", "exec"), envs)
    dataflow: Flow = envs[dataflow_scope_name]
    dataflow.stage_key_vars = {name: value.id for name, value in envs.items() if isinstance(value, StageKey)}

    dataflow.print(logger, level=logging.DEBUG)

    stage_regions = analyze_stage_regions(func, init_insert_idx, dataflow, envs)
    smem_layouts = _plan_smem_layouts(dataflow)
    _validate_smem_majors(dataflow, stage_regions.mem_events)
    validate_fixed_payload_regions(stage_regions.bind_regions)
    tma_bytes = infer_tma_payload_bytes(dataflow, stage_regions.bind_regions)

    channels = analyze_channels(dataflow, stage_regions.mem_events)
    validate_channels(dataflow, channels, stage_regions.bind_regions)
    pipelines = lower_mpmc_pipelines(channels, stage_regions.mem_events, tma_bytes)
    cta_group = 2 if any(event.cta_group == 2 for event in stage_regions.mem_events) else 1
    cluster_shape = ctx.get(CLUSTER_SHAPE_CTX_KEY)
    cta_group_shape = ctx.get(CTA_GROUP_SHAPE_CTX_KEY)
    _validate_cluster_requirements(
        cluster_shape,
        cta_group_shape,
        cta_group=cta_group,
        uses_cluster_multicast=any(event.cluster_multicast for event in stage_regions.mem_events),
    )

    resolved_group_shape = cta_group_shape or (1, 1, 1)
    transfer_plans = {name: value for name, value in envs.items() if isinstance(value, TMATransferPlan)}
    used_transfer_plans = {event.transfer_plan for event in stage_regions.mem_events if event.transfer_plan is not None}
    transfer_topologies = {
        plan: _analyze_transfer_plan(
            plan,
            cluster_shape or (1, 1, 1),
            resolved_group_shape,
        )
        for plan in used_transfer_plans
    }

    graph = {
        "roles": dataflow.roles,
        "memories": dataflow.memories,
        "init_insert_idx": init_insert_idx,
        "dataflow_node_id": id(dataflow_scope),
        "mem_events": stage_regions.mem_events,
        "bind_regions": stage_regions.bind_regions,
        "bind_stmt_ids": stage_regions.bind_stmt_ids,
        "channels": channels,
        "pipelines": pipelines,
        "cluster_shape": cluster_shape,
        "cta_group_shape": resolved_group_shape,
        "cta_group_shape_explicit": cta_group_shape is not None,
        "transfer_plans": transfer_plans,
        "transfer_topologies": transfer_topologies,
        "smem_layouts": smem_layouts,
        "const_values": {name: value
                         for name, value in envs.items()
                         if isinstance(value, (int, bool, tuple, list))},
    }
    if hasattr(dataflow, "memory_mapping_plan"):
        graph["memory_mapping_plan"] = dataflow.memory_mapping_plan
    return graph


def _validate_smem_majors(dataflow: Flow, mem_events: list) -> None:
    for name in sorted({event.mem for event in mem_events if event.effect_kind == R_SMEM_TCGEN05_COMMIT}):
        if dataflow.memories[name].major is None:
            raise ValueError(f"{name}: UMMA SMEM operand requires explicit g.smem(..., major=\"K\" or \"MN\")")


def _plan_smem_layout(mem) -> SMemLayoutPlan:
    shape = tuple(int(dim) for dim in mem.shape)
    if len(shape) != 2 or str(mem.dtype) not in {"bfloat16", "float16", "uint8", "int8"}:
        raise ValueError(f"{mem.name}: canonical UMMA layout requires rank-2 BF16/F16/U8 SMEM")
    elem_bytes = _dtype_size(mem.dtype)
    continuous = shape[-1] if mem.major == "K" else shape[-2]
    swizzle = next((size for size in (128, 64, 32) if continuous * elem_bytes % size == 0), None)
    if swizzle is None:
        swizzle = 0  # INTERLEAVE (no swizzle)
    return SMemLayoutPlan(shape, mem.dtype, mem.major, swizzle)


def _plan_smem_layouts(dataflow: Flow) -> dict[str, SMemLayoutPlan]:
    return {
        name: _plan_smem_layout(mem)
        for name, mem in dataflow.memories.items()
        if mem.__class__.__name__ == "SMem" and mem.major is not None
    }
