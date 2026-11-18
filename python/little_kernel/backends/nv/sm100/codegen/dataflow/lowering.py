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
"""Role/dataflow resource lowering to typed IR nodes (RF05/RF06).

Responsibility split of the former monolith. This module lowers the analyzed
role graph to explicit, typed lowered IR operations
(:mod:`little_kernel.core.ir`): backend statements/includes become
``TargetStatement``/``TargetInclude`` and backend expressions become
``TargetExpression`` with an explicitly inferred pointer/scalar result type.
There are no opaque ``_lk_cpp_*`` attributes and no ``int32`` catch-all.
"""

from __future__ import annotations

from little_kernel.backends.nv.sm100.passes.dataflow.common import call_name as _call_name

import ast
import re
from typing import Any, Optional

from little_kernel.core.memory_effects import Completion
from little_kernel.core import ir as lk_ir
from little_kernel.core.ir import IROpEffect
from little_kernel.core.type_system import LLType, Pointer, bfloat16, bool_, float16, int8, uint8, uint32, uint64
from little_kernel.backends.nv.sm100.passes.dataflow.graph_construction import DATAFLOW_CTX_KEY, DEFAULT_TMA_TRANSFER_PLAN
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import _call_args, _call_name, _const_value, _effect_operand_expr, _name, _resolve_local_expr, _static_tuple  # noqa: F811
from little_kernel.backends.nv.sm100.codegen.dataflow.layout import _layout_fields
from little_kernel.backends.nv.sm100.passes.dataflow.graph_construction import _plan_smem_layout, _plan_smem_layouts
from little_kernel.backends.nv.sm100.passes.dataflow.channels import barrier_participation, consumer_roles as _consumer_roles, dtype_size as _dtype_size, numel as _numel, producer_roles as _producer_roles
from little_kernel.backends.nv.sm100.core.pipeline import PIPELINE_GENERIC_GENERIC, PIPELINE_GENERIC_UMMA, PIPELINE_TMEM_STORE_TMEM_LOAD, PIPELINE_TMEM_STORE_UMMA, PIPELINE_TMA_GENERIC, PIPELINE_TMA_STORE_CROSS_ROLE, PIPELINE_TMA_STORE_SAME_ROLE, PIPELINE_TMA_TMA_STORE, PIPELINE_TMA_TMA_STORE_SAME_ROLE, PIPELINE_TMA_UMMA, PIPELINE_UMMA_TMEM_LOAD, PIPELINE_UMMA_UMMA, PipelineSpec
from little_kernel.backends.nv.sm100.passes.dataflow.intrinsics_registry import intrin_has_completion as _intrin_has_completion
from little_kernel.backends.nv.sm100.passes.dataflow.compile_time_unroll import _const_int
from little_kernel.backends.nv.sm100.passes.dataflow.umma_modes import umma_effect_normalization_pass

from little_kernel.backends.nv.sm100.codegen.dataflow.target_ops import PIPELINE_HEADER, cpp_builtin as _cpp_builtin, cpp_include as _cpp_include, cpp_raw_expr as _cpp_raw_expr, cpp_stmt as _cpp_stmt, infer_target_type as _infer_target_type

from little_kernel.backends.nv.sm100.passes.dataflow.graph_ops import _ident, _pipe_name, _pipeline_region_bounds, _pipeline_side_keys, _pipeline_side_via, _pipeline_stages, _pipeline_var_name, _region_bounds, _role_mem_events


def _attr(path: str) -> ast.expr:
    parts = path.split(".")
    node: ast.expr = ast.Name(id=parts[0], ctx=ast.Load())
    for part in parts[1:]:
        node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
    return node


def _ann_assign(name: str, annotation: str, value: ast.expr) -> ast.AnnAssign:
    return ast.AnnAssign(target=ast.Name(id=name, ctx=ast.Store()), annotation=_attr(annotation), value=value, simple=1)


def _call(path: str, args: list[ast.expr] | None = None) -> ast.Call:
    return ast.Call(func=_attr(path), args=args or [], keywords=[])


def _add(left: ast.expr, right: ast.expr) -> ast.expr:
    if isinstance(right, ast.Constant) and right.value == 0:
        return left
    return ast.BinOp(left=left, op=ast.Add(), right=right)


from little_kernel.core import ir as typed_ir
from little_kernel.core import type_system as types


def _element_type(dtype):
    if isinstance(dtype, types.LLType):
        return dtype
    result = getattr(types, str(dtype).rsplit(".", 1)[-1], None)
    if not isinstance(result, types.LLType):
        raise TypeError(f"Unsupported dataflow resource element type: {dtype!r}")
    return result


def _cpp_type(dtype) -> str:
    text = str(dtype)
    return {
        "bfloat16": "__nv_bfloat16",
        "float16": "__half",
        "float32": "float",
        "float64": "double",
        "uint64": "uint64_t",
        "uint32": "uint32_t",
        "uint16": "uint16_t",
        "uint8": "uint8_t",
        "int64": "int64_t",
        "int32": "int32_t",
        "int16": "int16_t",
        "int8": "int8_t",
    }.get(text, text)


def _tmem_stage_cols(mem, cta_group: int = 1) -> int:
    shape = mem.shape if isinstance(mem.shape, (tuple, list)) else [mem.shape]
    if len(shape) < 2:
        return 0
    # BF16/F16 TMEM values pack two per 32-bit column. uint8 TMEM declarations
    # describe tcgen05 scale-factor columns, not scalar byte arrays.
    elem_bytes = 2 if str(mem.dtype) in {"bfloat16", "float16"} else 4
    packed_cols = (int(shape[1]) * elem_bytes + 3) // 4
    cols = ((int(shape[0]) + 127) // 128) * packed_cols
    if cta_group == 2:
        if cols % cta_group:
            raise ValueError(f"2CTA TMEM columns must be even, got {cols}")
        cols //= cta_group
    return cols


def _tmem_cols(mem, cta_group: int = 1) -> int:
    return _tmem_stage_cols(mem, cta_group) * int(mem.stages)


def _pipelines_by_var(graph: dict) -> dict[str, dict]:
    return {_pipeline_var_name(pipeline): pipeline for pipeline in graph["pipelines"]}


def _stage_expr(pipe: str, side: str) -> str:
    return f"{pipe}.{side}_stage()"


def _smem_expr(mem: str, pipe: str, side: str, declaration) -> ast.expr:
    element = _element_type(declaration.dtype)
    base = typed_ir.ResourceReference(f"__lk_smem.{mem}[0]", element, space="smem")
    stage = _cpp_raw_expr(_stage_expr(pipe, side), types.uint32)
    return typed_ir.StageReference(base, stage, _numel(declaration.shape))


def _uses_2sm(graph: dict) -> bool:
    return any(event.cta_group == 2 for event in graph.get("mem_events", ()))


def _tma_descriptor_names(func: ast.FunctionDef) -> list[str]:
    names: list[str] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call) or len(node.args) < 2:
            continue
        name = _call_name(node)
        if not _intrin_has_completion(name, Completion.MBARRIER_TX, Completion.TMA_BULK):
            continue
        if name in {"tma_ld_1d", "tma_st_1d"}:
            continue
        desc = node.args[1]
        if isinstance(desc, ast.Name) and desc.id not in names:
            names.append(desc.id)
    return names


def _umma_descriptor_anchors(func: ast.FunctionDef) -> dict[str, tuple[str, ast.expr]]:
    bindings: dict[str, tuple[str, ast.expr]] = {}
    for node in ast.walk(func):
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        else:
            continue
        if (not isinstance(target, ast.Name) or not isinstance(value, ast.Call)
                or _call_name(value) != "make_umma_smem_desc" or len(value.args) != 2):
            continue
        mem = _name(value.args[0])
        if mem is None:
            continue
        previous = bindings.setdefault(target.id, (mem, value.args[1]))
        if previous[0] != mem:
            raise ValueError(f"descriptor {target.id!r} refers to both {previous[0]!r} and {mem!r}")
    return bindings


def _consumer_release_tid(graph: dict, pipeline: dict) -> int:
    warps = [int(warp) for role in _consumer_roles(pipeline) for warp in graph["roles"][role].warps]
    return 32 * min(warps)


def _pipeline_decl_code(graph: dict, pipeline: dict) -> str:
    pipe = _pipeline_var_name(pipeline)
    template: PipelineSpec = pipeline["template"]
    stages = _pipeline_stages(graph, pipeline)
    cpp_name = f"little_kernel::role_pipeline::{template.name}"
    participation = pipeline.get("participation")
    if participation is None:
        participation = barrier_participation(graph, pipeline)
    if template.has_barriers:
        producer = f"__lk_smem.{pipe}_producer"
        consumer = f"__lk_smem.{pipe}_consumer"
        producer_count = participation.producer_count
        consumer_count = participation.consumer_count
        cta_group = participation.cta_group_size
        if template in {PIPELINE_TMA_UMMA, PIPELINE_TMA_GENERIC}:
            tx = participation.expected_tx_bytes
            if tx is None:
                raise ValueError(f"{pipeline['name']}: missing expected TMA transaction bytes")
            args = f"{stages}, {tx}, {producer_count}, {consumer_count}, {cta_group}"
            masks = participation.consumer_masks
            release_suffix = ""
            if template == PIPELINE_TMA_GENERIC and (masks is not None or cta_group > 1):
                release_suffix = f", {_consumer_release_tid(graph, pipeline)}"
            if masks is not None:
                unique = set(masks)
                template_mask = next(iter(unique)) if len(unique) == 1 else 0
                args += f", true, {template_mask}{release_suffix}"
                if len(unique) > 1:
                    topology = ", ".join(str(value) for value in (*graph["cluster_shape"], *graph["cta_group_shape"]))
                    group_expr = (f"little_kernel::role_pipeline::group_index<{topology}>()")
                    mask_expr = str(masks[-1])
                    for group in range(len(masks) - 2, -1, -1):
                        mask_expr = (f"({group_expr} == {group} ? {masks[group]} : {mask_expr})")
                    return (f"{cpp_name}<{args}> {pipe}({producer}, {consumer}, uint16_t({mask_expr}));")
            elif release_suffix:
                args += f", false, 0{release_suffix}"
            return f"{cpp_name}<{args}> {pipe}({producer}, {consumer});"
        if template in {
                PIPELINE_GENERIC_UMMA,
                PIPELINE_TMEM_STORE_UMMA,
                PIPELINE_UMMA_TMEM_LOAD,
                PIPELINE_UMMA_UMMA,
        }:
            return (f"{cpp_name}<{stages}, {producer_count}, {consumer_count}, {cta_group}> "
                    f"{pipe}({producer}, {consumer});")
        if template in {PIPELINE_GENERIC_GENERIC, PIPELINE_TMEM_STORE_TMEM_LOAD}:
            return f"{cpp_name}<{stages}, {producer_count}, {consumer_count}> {pipe}({producer}, {consumer});"
        if template == PIPELINE_TMA_STORE_CROSS_ROLE:
            return f"{cpp_name}<{stages}, {producer_count}, {consumer_count}> {pipe}({producer}, {consumer});"
        if template == PIPELINE_TMA_TMA_STORE:
            tx = participation.expected_tx_bytes
            if tx is None:
                raise ValueError(f"{pipeline['name']}: missing expected TMA transaction bytes")
            return f"{cpp_name}<{stages}, {tx}, {producer_count}, {consumer_count}> {pipe}({producer}, {consumer});"
        if template == PIPELINE_TMA_TMA_STORE_SAME_ROLE:
            tx = participation.expected_tx_bytes
            if tx is None:
                raise ValueError(f"{pipeline['name']}: missing expected TMA transaction bytes")
            return f"{cpp_name}<{stages}, {tx}, {producer_count}> {pipe}({producer});"
        raise ValueError(f"{pipeline['name']}: no C++ declaration for {template.name}")
    if template == PIPELINE_TMA_STORE_SAME_ROLE:
        producer_threads = 32 * sum(len(graph["roles"][role].warps) for role in _producer_roles(pipeline))
        barrier_id = pipeline.get("named_barrier_id", 1)
        args = f"{stages}, {producer_threads}"
        if barrier_id != 1:
            args += f", {barrier_id}"
        return f"{cpp_name}<{args}> {pipe};"
    raise ValueError(f"{pipeline['name']}: no C++ declaration for {template.name}")


def _role_condition(graph: dict, role_name: str) -> ast.expr:
    role = graph["roles"][role_name]
    warp_tests = [
        ast.Compare(
            left=ast.Name(id="__lk_warp_idx", ctx=ast.Load()),
            ops=[ast.Eq()],
            comparators=[ast.Constant(value=int(warp))],
        ) for warp in role.warps
    ]
    if not warp_tests:
        warp_condition = ast.Constant(value=True)
    elif len(warp_tests) == 1:
        warp_condition = warp_tests[0]
    else:
        warp_condition = ast.BoolOp(op=ast.Or(), values=warp_tests)

    umma_cta_groups = {
        event.cta_group
        for event in graph["mem_events"]
        if event.role == role_name and event.effect_kind.completion == Completion.TCGEN05_COMMIT
    }
    if len(umma_cta_groups) > 1:
        raise ValueError(f"{role_name}: mixed UMMA CTA groups in one role are unsupported")
    if umma_cta_groups != {2}:
        return warp_condition
    cta_group = next(iter(umma_cta_groups))
    group_leader = _cpp_raw_expr(f"little_kernel::role_pipeline::cta_group_is_leader<{cta_group}>()", types.bool_)
    return ast.BoolOp(op=ast.And(), values=[warp_condition, group_leader])


def _role_pred_name(role_name: str) -> str:
    return f"ROLE_{role_name}"


def _first_tensorcore_warp(graph: dict) -> int:
    for role in graph["roles"].values():
        if getattr(role.hw, "value", role.hw) == "TensorCore" and role.warps:
            return int(role.warps[0])
    for role in graph["roles"].values():
        if role.warps:
            return int(role.warps[0])
    return 0


class RoleResourceMaterializer(ast.NodeTransformer):

    def __init__(self, graph: dict) -> None:
        self.graph = graph
        self.roles = set(graph["roles"])
        self.pipelines_by_var = _pipelines_by_var(graph)
        self.current_role: str | None = None
        self.cta_group = 2 if _uses_2sm(graph) else 1
        self.mapped_memory_names = self._mapped_memory_names()
        self.tmem_offsets, self.tmem_total_cols = self._tmem_layout()
        self.tma_descriptors: list[str] = []
        self.umma_descriptor_anchors: dict[str, tuple[str, ast.expr]] = {}
        self.unroll_values: dict[str, tuple[int, ...]] = {}

    def _role_comment_lines(self, role_name: str) -> tuple[str, ...]:
        role = self.graph["roles"][role_name]
        display_name = (role.name if role.name == role.var else f"{role.name} (var: {role.var})")
        warps = ", ".join(str(warp) for warp in role.warps)
        resources = ", ".join(
            sorted({
                event.mem
                for event in self.graph["mem_events"]
                if event.role == role_name and event.mem in self.graph["memories"]
            }))
        content = (
            f"// role: {display_name}",
            f"// hw: {role.hw.value} | warps: [{warps}]",
            f"// resources: [{resources}]",
        )
        border = "/" * max(90, *(len(line) for line in content))
        return (border, *content, border)

    def _default_tmem_offsets(self) -> dict[str, int]:
        out: dict[str, int] = {}
        cursor = 0
        overlay = self.graph.get("buffer_overlay")
        placements = getattr(overlay, "placements", {}) if overlay is not None else {}
        aliases = {
            name
            for name, placement in placements.items()
            if placement.space == "tmem" and placement.physical != name
        }
        for name, mem in self.graph["memories"].items():
            if mem.__class__.__name__ != "TMem" or name in aliases:
                continue
            out[name] = cursor
            cursor += _tmem_cols(mem, self.cta_group)
        return out

    def _mapped_memory_names(self) -> set[str]:
        raw = self.graph.get("memory_mapping_plan")
        if raw is None:
            return set()
        if hasattr(raw, "entries"):
            raw = raw.entries
        if isinstance(raw, dict):
            return set(raw)
        return {item["logical"] if isinstance(item, dict) else item.logical for item in raw}

    def _tmem_layout(self) -> tuple[dict[str, int], int]:
        offsets = self._default_tmem_offsets()
        overlay = self.graph.get("buffer_overlay")
        placements = getattr(overlay, "placements", {}) if overlay is not None else {}
        for name in self.mapped_memory_names:
            placement = placements.get(name)
            if placement is None:
                continue
            mem = self.graph["memories"].get(name)
            if mem is None or mem.__class__.__name__ != "TMem":
                continue
            if placement.physical == name:
                offsets[name] = placement.offset
            else:
                offsets[name] = offsets.get(placement.physical, 0) + placement.offset
        total = max(
            (offset + _tmem_cols(self.graph["memories"][name], self.cta_group) for name, offset in offsets.items()),
            default=0,
        )
        # tcgen05.alloc accepts 32..512 power-of-two columns.
        if total:
            total = max(32, 1 << (total - 1).bit_length())
            if total > 512:
                raise ValueError(f"TMEM allocation exceeds 512 columns: {total}")
        return offsets, total

    def _mapped_placement(self, mem: str):
        if mem not in self.mapped_memory_names:
            return None
        overlay = self.graph.get("buffer_overlay")
        placements = getattr(overlay, "placements", {}) if overlay is not None else {}
        return placements.get(mem)

    def _channel_storage(self, mem: str) -> str:
        placement = self._mapped_placement(mem)
        if placement is None or placement.logical == placement.physical:
            return mem
        overlay = self.graph.get("buffer_overlay")
        placements = getattr(overlay, "placements", {}) if overlay is not None else {}
        storage = placements.get(placement.physical)
        if storage is not None and (
                placement.physical,
                placement.space,
                placement.offset,
                placement.size,
                placement.stages,
                placement.stage_stride,
        ) == (
                storage.physical,
                storage.space,
                storage.offset,
                storage.size,
                storage.stages,
                storage.stage_stride,
        ):
            return placement.physical
        return mem

    def _shared_storage_code(self) -> str:
        lines = ["struct alignas(1024) SharedStorage {"]
        for name, mem in self.graph["memories"].items():
            if mem.__class__.__name__ == "TMem":
                continue
            placement = self._mapped_placement(name)
            if placement is not None and placement.space == "smem" and placement.physical != name:
                continue
            plan = self.graph["smem_layouts"].get(name)
            alignment = 8 * plan.swizzle_bytes if plan is not None and plan.swizzle_bytes > 0 else 16
            lines.append(
                f"  alignas({alignment}) {_cpp_type(mem.dtype)} {name}[{int(mem.stages)}][{_numel(mem.shape)}];")
        for pipeline in self.graph["pipelines"]:
            if not pipeline["template"].has_barriers:
                continue
            pipe = _pipeline_var_name(pipeline)
            stages = _pipeline_stages(self.graph, pipeline)
            lines.append(f"  alignas(8) uint64_t {pipe}_producer[{stages}];")
            if pipeline["template"].consumer is not None:
                lines.append(f"  alignas(8) uint64_t {pipe}_consumer[{stages}];")
        if self.tmem_total_cols:
            lines.append("  alignas(16) uint32_t __lk_tmem_alloc_addr[1];")
        lines.append("};")
        return "\n".join(lines)

    def _materialize_resources(self) -> list[ast.stmt]:
        cta_group = 2 if _uses_2sm(self.graph) else 1
        llpp = "little_kernel::role_pipeline"
        stmts: list[ast.stmt] = [
            _cpp_include(PIPELINE_HEADER),
            _cpp_builtin(self._shared_storage_code()),
            _ann_assign("__lk_warp_idx", "ll.int32",
                        ast.BinOp(left=_call("ll.threadIdx_x"), op=ast.FloorDiv(), right=ast.Constant(value=32))),
            _cpp_stmt("extern __shared__ __align__(1024) uint8_t __lk_smem_raw[];"),
            _cpp_stmt("SharedStorage& __lk_smem = *reinterpret_cast<SharedStorage*>(__lk_smem_raw);"),
        ]
        for role_name in self.graph["roles"]:
            stmts.append(_ann_assign(_role_pred_name(role_name), "ll.bool_", _role_condition(self.graph, role_name)))
        for name, offset in self.tmem_offsets.items():
            stmts.append(_ann_assign(f"__lk_tmem_{name}", "ll.uint32", _call("ll.uint32",
                                                                             [ast.Constant(value=offset)])))
        for pipeline in self.graph["pipelines"]:
            stmts.append(_cpp_stmt(_pipeline_decl_code(self.graph, pipeline)))
        for stream in self.graph.get("tma_store_streams", ()):
            stmts.append(_cpp_stmt(f"{llpp}::TmaStoreStream {stream['name']};"))
        if self.graph["pipelines"]:
            names = ", ".join(_pipeline_var_name(pipeline) for pipeline in self.graph["pipelines"])
            stmts.append(
                _cpp_stmt(f"{llpp}::init_pipelines(__lk_warp_idx, {_first_tensorcore_warp(self.graph)}, {names});"))
        if self.tma_descriptors:
            body = " ".join(f"{llpp}::prefetch_tma_descriptor(&{name});" for name in self.tma_descriptors)
            stmts.append(
                _cpp_stmt(f"if (__lk_warp_idx == {_first_tensorcore_warp(self.graph)} and "
                          f"{llpp}::pipeline_elect_one()) {{ {body} }}"))
        if self.tmem_total_cols:
            stmts.append(
                _cpp_stmt(
                    f"if (__lk_warp_idx == {_first_tensorcore_warp(self.graph)}) {{ "
                    f"{llpp}::tmem_alloc<{cta_group}>(__lk_smem.__lk_tmem_alloc_addr, {self.tmem_total_cols}); }}"))
        if self.graph["pipelines"] or self.tmem_total_cols:
            stmts.append(_cpp_stmt(f"{llpp}::cluster_sync<{cta_group}>();"))
        return stmts

    def _finalize_resources(self) -> list[ast.stmt]:
        if not self.tmem_total_cols:
            return []
        cta_group = 2 if _uses_2sm(self.graph) else 1
        runtime = "little_kernel::role_pipeline"
        sync = [_cpp_stmt("__syncthreads();")]
        if cta_group == 2:
            sync.append(_cpp_stmt(f"{runtime}::cluster_sync<2>();"))
        return sync + [
            _cpp_stmt(f"if (__lk_warp_idx == {_first_tensorcore_warp(self.graph)}) {{ "
                      f"{runtime}::tmem_relinquish_alloc_permit<{cta_group}>(); "
                      f"{runtime}::tmem_dealloc<{cta_group}>(0, {self.tmem_total_cols}); }}"),
        ]

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.tma_descriptors = _tma_descriptor_names(node)
        self.umma_descriptor_anchors = _umma_descriptor_anchors(node)
        out: list[ast.stmt] = []
        for stmt in node.body:
            if id(stmt) == self.graph["dataflow_node_id"]:
                out.extend(self._materialize_resources())
                continue
            visited = self.visit(stmt)
            if visited is None:
                continue
            out.extend(visited if isinstance(visited, list) else [visited])
        out.extend(self._finalize_resources())
        node.body = out
        return node

    def visit_With(self, node: ast.With):
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Name):
            role_name = node.items[0].context_expr.id
            if role_name in self.roles:
                old = self.current_role
                self.current_role = role_name
                body = [self.visit(stmt) for stmt in node.body]
                self.current_role = old
                flat_body = [
                    item for stmt in body if stmt is not None for item in (stmt if isinstance(stmt, list) else [stmt])
                ]
                comments = [ast.copy_location(_cpp_stmt(line), node) for line in self._role_comment_lines(role_name)]
                branch = ast.copy_location(
                    ast.If(test=ast.Name(id=_role_pred_name(role_name), ctx=ast.Load()), body=flat_body, orelse=[]),
                    node,
                )
                return [*comments, branch]
        return self.generic_visit(node)

    def visit_For(self, node: ast.For):
        values = None
        if (isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and _call_name(node.iter) == "unroll"
                and len(node.iter.args) == 1):
            range_call = node.iter.args[0]
            if isinstance(range_call, ast.Call) and _call_name(range_call) == "range":
                try:
                    args = [int(_const_value(arg, self.graph.get("const_values", {}))) for arg in range_call.args]
                    values = tuple(range(*args))
                except ValueError:
                    values = None
        if values is None:
            return self.generic_visit(node)
        name = node.target.id
        previous = self.unroll_values.get(name)
        self.unroll_values[name] = values
        node = self.generic_visit(node)
        if previous is None:
            self.unroll_values.pop(name, None)
        else:
            self.unroll_values[name] = previous
        return node

    def visit_Expr(self, node: ast.Expr):
        call = node.value
        if _call_name(call) == "align_memory":
            return ast.copy_location(_cpp_stmt("/* role SharedStorage owns dynamic_shared layout */"), node)
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "ri" and call.func.attr in {
                    "umma_f16_ss_1sm_kblock",
                    "umma_f16_ts_1sm_kblock",
                    "umma_f16_ts_2sm_kblock",
                    "umma_f16_ss_2sm_kblock",
                }):
            return [
                ast.copy_location(ast.Expr(value=item), node) for item in self._lower_umma_kblock(call.func.attr, call)
            ]
        if isinstance(call, ast.Call) and _call_name(call) in {"tma_ld_2d", "tma_ld_3d"}:
            rank = 2 if _call_name(call) == "tma_ld_2d" else 3
            lowered = self._lower_planned_tma(call, rank)
            if lowered is not None:
                pipe = self._tma_load_pipe(call)
                return [
                    ast.copy_location(_cpp_stmt(f"{pipe}.producer_prepare_transaction();"), node),
                    *(ast.copy_location(ast.Expr(value=item), node) for item in lowered),
                ]
        if isinstance(call, ast.Call) and _call_name(call) in {"tma_st_2d", "tma_st_3d"}:
            rank = 2 if _call_name(call) == "tma_st_2d" else 3
            lowered = self._lower_tma_store(call, rank)
            if lowered is not None:
                return [ast.copy_location(ast.Expr(value=item), node) for item in lowered]
        pipe = self._tma_load_pipe(call)
        if pipe is not None:
            return [
                ast.copy_location(_cpp_stmt(f"{pipe}.producer_prepare_transaction();"), node),
                ast.copy_location(ast.Expr(value=self.visit(call)), node),
            ]
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if isinstance(call.func.value, ast.Name) and call.func.value.id in self.pipelines_by_var:
                pipe = call.func.value.id
                return ast.copy_location(_cpp_stmt(f"{pipe}.{call.func.attr}();"), node)
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in self.pipelines_by_var):
            return None
        return self.generic_visit(node)

    def _transfer_plan(self, expr: ast.AST | None) -> TMATransferPlan:  # noqa: F821
        if expr is None or (isinstance(expr, ast.Constant) and expr.value is None):
            return DEFAULT_TMA_TRANSFER_PLAN
        plan_name = _name(expr)
        plan = self.graph.get("transfer_plans", {}).get(plan_name)
        if plan is None:
            raise ValueError("tma_ld_2d/tma_ld_3d plan must be a named flow.tma_transfer(...) compile-time value")
        return plan

    def _tma_regions(self, mem: str, call_args: dict) -> list[tuple[int, int]]:
        env = self.graph.get("const_values", {})
        shape = _static_tuple(call_args.get("shape"), env, {}, "TMA shape")
        coord = _static_tuple(call_args.get("smem_coord", (0, ) * len(shape)), env, {}, "TMA smem_coord")
        mem_shape = tuple(int(dim) for dim in self.graph["memories"][mem].shape)
        if len(shape) != 2 or len(coord) != 2 or len(mem_shape) != 2:
            raise ValueError(f"TMA logical SMEM region for {mem} must be rank 2")
        if any(size <= 0 or start + size > limit for start, size, limit in zip(coord, shape, mem_shape)):
            raise ValueError(f"TMA logical SMEM region {coord}+{shape} exceeds {mem} shape {mem_shape}")

        layout = self.graph["smem_layouts"].get(mem)
        if layout is None or layout.swizzle_bytes == 0:
            return [(coord[0] * mem_shape[1] + coord[1], 0)]
        atom = layout.swizzle_bytes // _dtype_size(layout.dtype)
        major_axis = 1 if layout.major == "K" else 0
        other_axis = 1 - major_axis
        if coord[major_axis] % atom or shape[major_axis] % atom:
            raise ValueError(f"{mem}: TMA {layout.major}-major region must be aligned to {atom} elements")
        return [(((coord[major_axis] + delta) // atom) * mem_shape[other_axis] * atom + coord[other_axis] * atom, delta)
                for delta in range(0, shape[major_axis], atom)]

    def _tma_load_pipe(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Call):
            return None
        name = _call_name(node)
        if name not in {"tma_ld_1d", "tma_ld_2d", "tma_ld_3d"}:
            return None
        return self._pipe_for_mem(_name(_call_args(name, node).get("mem")), "producer")

    def _topology_call(self, name: str, node: ast.Call) -> ast.expr:
        cluster_shape = self.graph.get("cluster_shape")
        group_shape = self.graph.get("cta_group_shape")
        if (cluster_shape is None or not self.graph.get("cta_group_shape_explicit", False)):
            raise ValueError(f"{name} requires explicit cluster_shape and cta_group_shape")
        topology = ", ".join(str(value) for value in (*cluster_shape, *group_shape))
        runtime = "little_kernel::role_pipeline"
        if name == "rank_in_group":
            if node.args or node.keywords:
                raise ValueError("rank_in_group() takes no arguments")
            return _cpp_raw_expr(f"{runtime}::rank_in_group<{topology}>()", types.uint32)
        if len(node.args) != 1 or node.keywords:
            raise ValueError(f"{name}(axis) requires one literal axis")
        axis = _const_value(node.args[0], {})
        if axis not in {"x", "y", "z"}:
            raise ValueError(f"{name} axis must be literal 'x', 'y', or 'z', got {axis!r}")
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        if name == "cluster_coord":
            cluster = ", ".join(str(value) for value in cluster_shape)
            return _cpp_raw_expr(f"{runtime}::cluster_coord<{cluster}, {axis_index}>()", types.uint32)
        return _cpp_raw_expr(f"{runtime}::group_coord<{topology}, {axis_index}>()", types.uint32)

    def _lower_planned_tma(self, node: ast.Call, rank: int):
        call_args = _call_args(f"tma_ld_{rank}d", node)
        mem_expr = call_args.get("mem")
        desc_expr = call_args.get("desc")
        coords_expr = call_args.get("gmem_coord")
        if not isinstance(mem_expr, ast.expr) or not isinstance(desc_expr, ast.expr):
            raise ValueError(f"tma_ld_{rank}d requires mem, desc, gmem_coord, and shape")
        mem = _name(mem_expr)
        pipe = self._pipe_for_mem(mem, "producer") if mem else None
        if pipe is None:
            return None

        if not isinstance(coords_expr, (ast.Tuple, ast.List)) or len(coords_expr.elts) != rank:
            raise ValueError(f"tma_ld_{rank}d gmem_coord must be a {rank}-element tuple")
        plan = self._transfer_plan(call_args.get("plan"))
        if (plan.inter_cta_coord_dim is not None and plan.inter_cta_coord_dim >= rank):
            raise ValueError(
                f"tma_ld_{rank}d inter_cta_coord_dim={plan.inter_cta_coord_dim} is outside its coordinate tuple")

        coords = [self.visit(coord) for coord in coords_expr.elts]
        if plan.inter_cta_coord_dim is not None:
            dim = plan.inter_cta_coord_dim
            topology = ", ".join(str(value) for value in (*self.graph["cluster_shape"], *self.graph["cta_group_shape"]))
            rank_in_group = _cpp_raw_expr(f"little_kernel::role_pipeline::rank_in_group<{topology}>()", types.uint32)
            coords[dim] = ast.BinOp(
                left=coords[dim],
                op=ast.Add(),
                right=ast.BinOp(
                    left=rank_in_group,
                    op=ast.Mult(),
                    right=ast.Constant(value=plan.inter_cta_stride),
                ),
            )

        topology = self.graph["transfer_topologies"][plan]
        # plan=None lowers via the DEFAULT singleton (plain cg1). Named plans
        # use the kernel cta_group even when their fields match the default.
        load_cta_group = 1 if plan is DEFAULT_TMA_TRANSFER_PLAN else self.cta_group
        lowered = []
        for physical_offset, k_delta in self._tma_regions(mem, call_args):
            box_coords = list(coords)
            if k_delta:
                box_coords[0] = _add(box_coords[0], ast.Constant(value=k_delta))
            smem_ptr = self._lower_mem(mem, "producer")
            if physical_offset:
                smem_ptr = _add(smem_ptr, ast.Constant(value=physical_offset))
            args = [
                self.visit(desc_expr),
                _cpp_raw_expr(f"{pipe}.producer_get_barrier()", types.Pointer(types.uint64)), smem_ptr, *box_coords
            ]
            cg_name = "1sm" if load_cta_group == 1 else "2sm"
            name = f"ll.tma_load_{rank}d_{cg_name}"
            if topology["multicast"]:
                axis = {"x": 0, "y": 1, "z": 2}[plan.mcast_group_dim]
                template_args = ", ".join(
                    str(value) for value in (*self.graph["cluster_shape"], *self.graph["cta_group_shape"], axis))
                runtime = "little_kernel::role_pipeline"
                mask = _cpp_raw_expr(f"{runtime}::tma_multicast_mask<{template_args}>()", types.uint32)
                issue_pred = _cpp_raw_expr(f"{runtime}::tma_multicast_is_issuer<{template_args}>()", types.bool_)
                name += "_mcast"
                args.extend((mask, issue_pred))
            lowered.append(ast.copy_location(_call(name, args), node))
        return lowered

    def _lower_tma_store(self, node: ast.Call, rank: int):
        name = f"tma_st_{rank}d"
        call_args = _call_args(name, node)
        mem_expr = call_args.get("mem")
        desc_expr = call_args.get("desc")
        coord_expr = call_args.get("gmem_coord")
        if not isinstance(mem_expr, ast.expr) or not isinstance(desc_expr, ast.expr):
            raise ValueError(f"{name} requires mem, desc, gmem_coord, and shape")
        mem = _name(mem_expr)
        if mem is None:
            return None
        if not isinstance(coord_expr, (ast.Tuple, ast.List)) or len(coord_expr.elts) != rank:
            raise ValueError(f"{name} gmem_coord must be a {rank}-element tuple")
        coords = [self.visit(item) for item in coord_expr.elts]
        lowered = []
        target = f"ll.tma_store_{rank}d"
        for physical_offset, k_delta in self._tma_regions(mem, call_args):
            smem_ptr = self._lower_mem(mem, "consumer")
            if physical_offset:
                smem_ptr = _add(smem_ptr, ast.Constant(value=physical_offset))
            box_coords = list(coords)
            if k_delta:
                box_coords[0] = _add(box_coords[0], ast.Constant(value=k_delta))
            lowered.append(ast.copy_location(_call(target, [self.visit(desc_expr), smem_ptr, *box_coords]), node))
        return lowered

    def _descriptor_mem(self, expr: ast.expr | None) -> str | None:
        anchor = self._descriptor_anchor(expr)
        return anchor[0] if anchor is not None else None

    def _descriptor_anchor(self, expr: ast.expr | None) -> tuple[str, ast.expr] | None:
        if isinstance(expr, ast.Name):
            return self.umma_descriptor_anchors.get(expr.id)
        if isinstance(expr, ast.Call) and _call_name(expr) == "make_umma_smem_desc":
            call_args = _call_args("make_umma_smem_desc", expr)
            mem = _name(call_args.get("mem"))
            coord = call_args.get("coord")
            if mem is not None and isinstance(coord, ast.expr):
                return mem, coord
        return None

    def _descriptor_at_k(self, expr: ast.expr | None, delta: int) -> ast.expr:
        anchor = self._descriptor_anchor(expr)
        if anchor is None:
            raise ValueError("UMMA K-block requires a typed SMEM descriptor anchor")
        mem, coord = anchor
        if not isinstance(coord, (ast.Tuple, ast.List)) or len(coord.elts) != 2:
            raise ValueError(f"{mem} UMMA descriptor coord must be a rank-2 tuple")
        advanced = ast.Tuple(
            elts=[coord.elts[0], _add(coord.elts[1], ast.Constant(value=delta))],
            ctx=ast.Load(),
        )
        lowered = self._lower_umma_descriptor(
            _call("ri.make_umma_smem_desc", [ast.Name(id=mem, ctx=ast.Load()), advanced]),
            k_extent=16,
        )
        if delta and isinstance(expr, ast.Name):
            base_offset = self._umma_descriptor_offset_16b(mem, coord)
            advanced_offset = self._umma_descriptor_offset_16b(mem, advanced)
            offset = ast.BinOp(left=advanced_offset, op=ast.Sub(), right=base_offset)
            desc = self.visit(expr)
            return ast.BinOp(
                left=ast.BinOp(
                    left=desc,
                    op=ast.BitAnd(),
                    right=ast.Constant(value=0xFFFFFFFFFFFFC000),
                ),
                op=ast.BitOr(),
                right=ast.BinOp(
                    left=ast.BinOp(left=desc, op=ast.Add(), right=offset),
                    op=ast.BitAnd(),
                    right=ast.Constant(value=0x3FFF),
                ),
            )
        return self.visit(expr) if delta == 0 and isinstance(expr, ast.expr) else lowered

    def _umma_descriptor_offset_16b(self, mem: str, coord_expr: ast.Tuple | ast.List) -> ast.expr:
        plan = self.graph["smem_layouts"][mem]
        shape = tuple(int(dim) for dim in plan.logical_shape)
        major_axis = 1 if plan.major == "K" else 0
        other_axis = 1 - major_axis
        atom = _layout_fields(plan)[0]
        major = self.visit(coord_expr.elts[major_axis])
        other = self.visit(coord_expr.elts[other_axis])
        physical_elem = ast.BinOp(
            left=ast.BinOp(
                left=ast.BinOp(left=major, op=ast.FloorDiv(), right=ast.Constant(value=atom)),
                op=ast.Mult(),
                right=ast.Constant(value=shape[other_axis] * atom),
            ),
            op=ast.Add(),
            right=ast.BinOp(
                left=ast.BinOp(left=other, op=ast.Mult(), right=ast.Constant(value=atom)),
                op=ast.Add(),
                right=ast.BinOp(left=major, op=ast.Mod(), right=ast.Constant(value=atom)),
            ),
        )
        return ast.BinOp(
            left=ast.BinOp(
                left=physical_elem,
                op=ast.Mult(),
                right=ast.Constant(value=_dtype_size(plan.dtype)),
            ),
            op=ast.FloorDiv(),
            right=ast.Constant(value=16),
        )

    def _lower_umma_kblock(self, attr: str, node: ast.Call) -> list[ast.Call]:
        call_args = _call_args(attr, node)
        zero_init = call_args.get("zero_init")
        if not isinstance(zero_init, ast.Constant) or zero_init.value not in {False, True}:
            raise ValueError(f"normalized {attr} requires a literal zero_init mode")
        try:
            k = int(_const_value(call_args.get("k"), self.graph.get("const_values", {})))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{attr} requires a compile-time logical K extent") from exc

        idesc = self._umma_idesc(attr, node)
        tmem_d = call_args.get("tmem_d") or call_args.get("d_mem")
        desc_b = call_args.get("desc_b")
        if not isinstance(tmem_d, ast.expr) or not isinstance(desc_b, ast.expr):
            raise ValueError(f"invalid normalized {attr} operands")
        calls: list[ast.Call] = []
        for delta in range(0, k, 16):
            mode = _call("ll.uint32", [ast.Constant(value=0 if delta == 0 and zero_init.value else 1)])
            if attr in {"umma_f16_ts_1sm_kblock", "umma_f16_ts_2sm_kblock"}:
                tmem_a = call_args.get("tmem_a")
                if not isinstance(tmem_a, ast.expr):
                    raise ValueError(f"invalid normalized {attr} operands")
                args = [
                    self.visit(tmem_d),
                    _add(self.visit(tmem_a), ast.Constant(value=delta // 2)),
                    self._descriptor_at_k(desc_b, delta), idesc, mode
                ]
                ll_name = ("ll.umma_f16_ts_2sm" if attr == "umma_f16_ts_2sm_kblock" else "ll.umma_f16_ts_1sm")
            else:
                desc_a = call_args.get("desc_a")
                if not isinstance(desc_a, ast.expr):
                    raise ValueError(f"invalid normalized {attr} operands")
                args = [
                    self.visit(tmem_d),
                    self._descriptor_at_k(desc_a, delta),
                    self._descriptor_at_k(desc_b, delta), idesc, mode
                ]
                ll_name = ("ll.umma_f16_ss_2sm" if attr == "umma_f16_ss_2sm_kblock" else "ll.umma_f16_ss_1sm")
            calls.append(_call(ll_name, args))
        return calls

    def _lower_umma_descriptor(self, node: ast.Call, *, k_extent: int = 1) -> ast.expr:
        call_args = _call_args("make_umma_smem_desc", node)
        mem = _name(call_args.get("mem"))
        plan = self.graph["smem_layouts"].get(mem or "")
        if mem is None or plan is None:
            raise ValueError(f"{mem}: UMMA descriptor requires a canonical SMEM layout plan")
        coord_expr = call_args.get("coord")
        if not isinstance(coord_expr, (ast.Tuple, ast.List)) or len(coord_expr.elts) != 2:
            raise ValueError(f"{mem} UMMA descriptor coord must be a rank-2 tuple")
        envs = [dict(self.graph.get("const_values", {}))]
        for name, values in self.unroll_values.items():
            envs = [{**env, name: value} for env in envs for value in values]
        try:
            coords = [tuple(int(_const_value(item, env)) for item in coord_expr.elts) for env in envs]
        except ValueError as exc:
            raise ValueError(
                f"{mem} UMMA descriptor coord must depend only on compile-time values or ll.unroll indices") from exc
        shape = tuple(int(dim) for dim in plan.logical_shape)
        if any(index < 0 or index >= limit
               for coord in coords
               for index, limit in zip(coord, shape)) or any(coord[1] + k_extent > shape[1] for coord in coords):
            raise ValueError(f"{mem}: UMMA descriptor coord exceeds logical shape {shape}")
        major_axis = 1 if plan.major == "K" else 0
        other_axis = 1 - major_axis
        atom, lbo, sbo, layout_type = _layout_fields(plan)
        elem_bytes = _dtype_size(plan.dtype)
        if any((((coord[major_axis] // atom) * shape[other_axis] * atom + coord[other_axis] * atom +
                 coord[major_axis] % atom) * elem_bytes) % 16 for coord in coords):
            raise ValueError(f"{mem}: UMMA descriptor coord is not 16-byte aligned")
        offset_16b = self._umma_descriptor_offset_16b(mem, coord_expr)
        start = ast.BinOp(
            left=ast.BinOp(left=_call("ll.cvta_generic_to_shared", [self._lower_mem(mem, "consumer")]), op=ast.RShift(),
                           right=ast.Constant(value=4)),
            op=ast.Add(),
            right=offset_16b,
        )
        return ast.copy_location(
            _call("ll.encode_umma_smem_desc",
                  [start, ast.Constant(value=lbo),
                   ast.Constant(value=sbo),
                   ast.Constant(value=layout_type)]), node)

    def _umma_idesc(self, attr: str, node: ast.Call) -> ast.Constant:
        call_args = _call_args(attr, node)
        shape = _static_tuple(call_args.get("out_shape"), self.graph.get("const_values", {}), {}, f"{attr} out_shape")
        if len(shape) != 2:
            raise ValueError(f"{attr} out_shape must be (M, N)")
        m, n = shape
        d_mem = _name(call_args.get("d_mem") or call_args.get("tmem_d"))
        d_shape = tuple(int(dim) for dim in self.graph["memories"][d_mem].shape) if d_mem else ()
        if m <= 0 or n <= 0 or m % 16 or n % 8 or not d_shape or m > d_shape[0] or n > d_shape[1]:
            raise ValueError(f"{attr} invalid output shape {shape} for {d_mem} {d_shape}")
        if str(self.graph["memories"][d_mem].dtype) != "float32":
            raise ValueError(f"{attr} requires a float32 TMEM output")
        a_major = "K"
        if "desc_a" in call_args:
            a_mem = self._descriptor_mem(call_args["desc_a"])
            if a_mem is None:
                raise ValueError(f"{attr} requires a typed A descriptor")
            a_major = self.graph["memories"][a_mem].major
        else:
            a_mem = _name(call_args.get("a_mem") or call_args.get("tmem_a"))
            if a_mem is None or a_mem not in self.graph["memories"]:
                raise ValueError(f"{attr} requires a typed TMEM A operand")
        b_mem = self._descriptor_mem(call_args.get("desc_b"))
        if b_mem is None:
            raise ValueError(f"{attr} requires a typed B descriptor")
        b_major = self.graph["memories"][b_mem].major
        formats = {"float16": 0, "bfloat16": 1}
        a_dtype = str(self.graph["memories"][a_mem].dtype)
        b_dtype = str(self.graph["memories"][b_mem].dtype)
        if a_dtype not in formats or b_dtype not in formats:
            raise ValueError(f"{attr} requires BF16/F16 A and B operands")
        encoded = ((1 << 4) | (formats[a_dtype] << 7) | (formats[b_dtype] << 10) | ((n // 8) << 17) | ((m // 16) << 24))
        if a_major == "MN":
            encoded |= 1 << 15
        if b_major == "MN":
            encoded |= 1 << 16
        for operand, bit in (("negate_a", 13), ("negate_b", 14)):
            value = call_args.get(operand)
            if value is None:
                continue
            value = _const_value(value, self.graph.get("const_values", {}))
            if not isinstance(value, bool):
                raise ValueError(f"{attr} {operand} must be a compile-time bool")
            if value:
                encoded |= 1 << bit
        return ast.Constant(value=encoded)

    def visit_Call(self, node: ast.Call):
        name = _call_name(node)
        if name in {"cluster_coord", "group_coord", "rank_in_group"}:
            return ast.copy_location(self._topology_call(name, node), node)
        if name == "val_cast" and len(node.args) == 2:
            return ast.copy_location(
                ast.Call(func=self.visit(node.args[1]), args=[self.visit(node.args[0])], keywords=[]),
                node,
            )
        if name == "tma_ld_2d":
            lowered = self._lower_planned_tma(node, 2)
            if lowered is not None:
                if len(lowered) != 1:
                    raise ValueError("tma_ld_2d may only appear as a statement")
                return lowered[0]
        if name == "tma_ld_3d":
            lowered = self._lower_planned_tma(node, 3)
            if lowered is not None:
                if len(lowered) != 1:
                    raise ValueError("tma_ld_3d may only appear as a statement")
                return lowered[0]
        if name == "tma_ld_1d" and len(node.args) == 3:
            mem = _name(node.args[0])
            pipe = self._pipe_for_mem(mem, "producer") if mem else None
            if pipe is not None:
                return ast.copy_location(
                    _call("ll.tma_copy_1d_g2s", [
                        self.visit(node.args[1]),
                        _cpp_raw_expr(f"{pipe}.producer_get_barrier()", types.Pointer(types.uint64)),
                        self._lower_mem(mem, "producer"),
                        self.visit(node.args[2]),
                    ]),
                    node,
                )
        if name in {"tma_st_2d", "tma_st_3d"}:
            rank = 2 if name == "tma_st_2d" else 3
            lowered = self._lower_tma_store(node, rank)
            if lowered is not None:
                if len(lowered) != 1:
                    raise ValueError(f"{name} may only appear as a statement")
                return lowered[0]
        if name == "tma_st_1d" and len(node.args) == 3:
            mem = _name(node.args[0])
            if mem:
                return ast.copy_location(
                    _call("ll.tma_copy_1d_s2g", [
                        self._lower_mem(mem, "consumer"),
                        self.visit(node.args[1]),
                        self.visit(node.args[2]),
                    ]),
                    node,
                )
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value,
                                                               ast.Name) and node.func.value.id == "ri":
            attr = node.func.attr
            if attr == "make_umma_smem_desc":
                return self._lower_umma_descriptor(node)
            if attr in {"tcgen05_cp_128x256b_1sm", "tcgen05_cp_128x256b_2sm"}:
                return ast.copy_location(_call(f"ll.{attr}", [self.visit(item) for item in node.args[1:]]), node)
            if attr in {"tcgen05_cp_32x128b_warpx4_1sm", "tcgen05_cp_32x128b_warpx4_2sm"}:
                return ast.copy_location(_call(f"ll.{attr}", [self.visit(item) for item in node.args[1:]]), node)
            if attr in {
                    "umma_mxfp_e2m1_e2m1_ss_1sm",
                    "umma_mxfp_e2m1_e2m1_ss_2sm",
                    "umma_mxfp_e4m3_e2m1_ss_1sm",
                    "umma_mxfp_e4m3_e2m1_ss_2sm",
                    "umma_mxfp_e4m3_e4m3_ss_1sm",
                    "umma_mxfp_e4m3_e4m3_ss_2sm",
                    "umma_nvfp4_ss_1sm",
                    "umma_nvfp4_ss_2sm",
            }:
                call_args = _call_args(attr, node)
                shape = _static_tuple(call_args.get("out_shape"), self.graph.get("const_values", {}), {},
                                      f"{attr} out_shape")
                if len(shape) != 2:
                    raise ValueError(f"{attr} out_shape must be (M, N)")
                c_mem = _name(call_args.get("c_mem"))
                c_shape = tuple(int(dim) for dim in self.graph["memories"][c_mem].shape)
                if (str(self.graph["memories"][c_mem].dtype) != "float32" or any(dim <= 0 for dim in shape)
                        or any(dim > limit for dim, limit in zip(shape, c_shape))):
                    raise ValueError(f"{attr} invalid output shape {shape} for {c_mem} {c_shape}")
                desc_a = call_args.get("desc_a")
                desc_b = call_args.get("desc_b")
                a_mem = self._descriptor_mem(desc_a)
                b_mem = self._descriptor_mem(desc_b)
                if a_mem is None or b_mem is None:
                    raise ValueError(f"{attr} requires typed A and B descriptors")
                major = {"K": 0, "MN": 1}
                physical = [call_args.get(key) for key in ("tmem_c", "desc_a", "desc_b", "accum", "tsfa", "tsfb")]
                if not all(isinstance(item, ast.expr) for item in physical):
                    raise ValueError(f"invalid {attr} operands")
                args = [self.visit(item) for item in physical[:3]]
                args += [ast.Constant(value=shape[0]), ast.Constant(value=shape[1])]
                args += [self.visit(item) for item in physical[3:]]
                args += [
                    ast.Constant(value=major[self.graph["memories"][a_mem].major]),
                    ast.Constant(value=major[self.graph["memories"][b_mem].major]),
                ]
                return ast.copy_location(_call(f"ll.{attr}", args), node)
            single_modes = {
                "umma_f16_ss_2sm": ("ll.umma_f16_ss_2sm", None),
                "umma_f16_ss_2sm_init": ("ll.umma_f16_ss_2sm", 0),
                "umma_f16_ss_2sm_accum": ("ll.umma_f16_ss_2sm", 1),
                "umma_f16_ss_1sm_init": ("ll.umma_f16_ss_1sm", 0),
                "umma_f16_ss_1sm_accum": ("ll.umma_f16_ss_1sm", 1),
                "umma_f16_ts_1sm_init": ("ll.umma_f16_ts_1sm", 0),
                "umma_f16_ts_1sm_accum": ("ll.umma_f16_ts_1sm", 1),
            }
            if attr in single_modes:
                call_args = _call_args(attr, node)
                ll_name, fixed_mode = single_modes[attr]
                if attr.startswith("umma_f16_ts"):
                    physical = [call_args.get(key) for key in ("tmem_d", "tmem_a", "desc_b")]
                else:
                    physical = [call_args.get(key) for key in ("tmem_d", "desc_a", "desc_b")]
                if not all(isinstance(item, ast.expr) for item in physical):
                    raise ValueError(f"invalid {attr} operands")
                mode = call_args.get("accum") if fixed_mode is None else ast.Constant(value=fixed_mode)
                args = [self.visit(item) for item in physical]
                args += [self._umma_idesc(attr, node), _call("ll.uint32", [self.visit(mode)])]
                return ast.copy_location(_call(ll_name, args), node)
            if attr == "tmem_ld_32dp_32bxn" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_ld_32dp_32bxn", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "tmem_ld_16dp_32bxn" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_ld_16dp_32bxn", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "tmem_ld_16dp_256bx2" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_ld_16dp_256bx2", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "tmem_ld_16dp_256bx4" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_ld_16dp_256bx4", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "smem_ld_128" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.smem_ld_128", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "smem_st_128" and len(node.args) >= 5:
                return ast.copy_location(_call("ll.smem_st_128", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "stmatrix_x4_trans" and len(node.args) == 6:
                return ast.copy_location(_call("ll.stmatrix_x4_trans", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "copy_bf16x16_linear_to_kmajor" and len(node.args) == 6:
                args = [self.visit(node.args[0]), *[self.visit(a) for a in node.args[2:]]]
                return ast.copy_location(_call("ll.copy_bf16x16_linear_to_kmajor", args), node)
            if attr == "tmem_ld_16dp_f32_to_smem_bf16_x16" and len(node.args) == 8:
                return ast.copy_location(
                    _call("ll.tmem_ld_16dp_f32_to_smem_bf16_x16", [self.visit(a) for a in node.args[2:]]),
                    node,
                )
            if attr == "tmem_ld_32dp_f32_to_smem_bf16_x8" and len(node.args) == 4:
                return ast.copy_location(
                    _call("ll.tmem_ld_32dp_f32_to_smem_bf16_x8", [self.visit(a) for a in node.args[2:]]),
                    node,
                )
            if attr == "tmem_ld_32dp_f32_to_smem_bf16_x32" and len(node.args) == 7:
                return ast.copy_location(
                    _call("ll.tmem_ld_32dp_f32_to_smem_bf16_x32", [self.visit(a) for a in node.args[2:]]),
                    node,
                )
            if attr in {"smem_ld_bf16", "smem_ld_f32", "smem_ld_f32x2", "smem_st_bf16", "smem_st_f32"}:
                node.args = [self.visit(a) for a in node.args]
                return node
            if attr == "tmem_ld_fence":
                return ast.copy_location(_call("ll.tmem_ld_fence", []), node)
            if attr == "tmem_st_32dp_32bxn" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_st_32dp_32bxn", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "tmem_st_16dp_128bx2" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_st_16dp_128bx2", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "tmem_st_16dp_32bxn" and len(node.args) >= 2:
                return ast.copy_location(_call("ll.tmem_st_16dp_32bxn", [self.visit(a) for a in node.args[1:]]), node)
            if attr == "tmem_st_fence":
                return ast.copy_location(_call("ll.tmem_st_fence", []), node)
            node.func.value.id = "ll"
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if not isinstance(node.ctx, ast.Load):
            return node
        if node.id in self.graph["memories"]:
            lowered = self._lower_mem(node.id, None)
            if lowered is not None:
                return ast.copy_location(lowered, node)
        return node

    def _pipe_for_mem(self, mem: str | None, side: str | None) -> str | None:
        if self.current_role is None or mem is None:
            return None
        channel_mem = self._channel_storage(mem)
        matches: list[str] = []
        for pipeline in self.graph["pipelines"]:
            if pipeline.get("overlay"):
                continue
            owned_via = set(pipeline.get("producer_via", pipeline["via"]))
            if channel_mem not in owned_via:
                continue
            producers = _producer_roles(pipeline)
            consumers = _consumer_roles(pipeline)
            if side == "producer" and self.current_role in producers:
                matches.append(_pipeline_var_name(pipeline))
            if side == "consumer" and self.current_role in consumers:
                matches.append(_pipeline_var_name(pipeline))
            if side is None:
                if self.current_role in producers or self.current_role in consumers:
                    matches.append(_pipeline_var_name(pipeline))
        matches = list(dict.fromkeys(matches))
        if len(matches) > 1:
            if side == "consumer":
                return matches[0]
            raise ValueError(f"{self.current_role}:{mem} belongs to multiple runtime pipelines: "
                             f"{matches}")
        return matches[0] if matches else None

    def _lower_mem(self, mem: str, side: str | None) -> ast.expr | None:
        if side is None and self.current_role is not None:
            channel_mem = self._channel_storage(mem)
            for pipeline in self.graph["pipelines"]:
                if pipeline.get("overlay"):
                    continue
                if channel_mem not in set(pipeline.get("producer_via", pipeline["via"])):
                    continue
                if self.current_role in _producer_roles(pipeline):
                    side = "producer"
                    break
                if self.current_role in _consumer_roles(pipeline):
                    side = "consumer"
                    break
        if mem in self.tmem_offsets:
            return self._lower_tmem(mem, side)
        pipe = self._pipe_for_mem(mem, side)
        if pipe is None:
            memory = self.graph["memories"][mem]
            if int(memory.stages) != 1:
                return None
            placement = self._mapped_placement(mem)
            if placement is not None and placement.space == "smem":
                base = placement.physical
                elem_size = _dtype_size(self.graph["memories"][base].dtype)
                return _cpp_raw_expr(f"(__lk_smem.{base}[0] + {placement.offset // elem_size})",
                                     types.Pointer(_element_type(self.graph["memories"][base].dtype)))
            return typed_ir.ResourceReference(f"__lk_smem.{mem}[0]", _element_type(memory.dtype), space="smem")
        return self._lower_smem(mem, pipe, side or "producer")

    def _lower_tmem(self, mem: str, side: str | None) -> ast.expr:
        base = ast.Name(id=f"__lk_tmem_{mem}", ctx=ast.Load())
        memory = self.graph["memories"][mem]
        if int(memory.stages) <= 1 or side is None:
            return base
        pipe = self._pipe_for_mem(mem, side)
        if pipe is None:
            return base
        stride = _tmem_stage_cols(memory, self.cta_group)
        return _cpp_raw_expr(f"(__lk_tmem_{mem} + {pipe}.{side}_stage() * {stride})", types.uint32)

    def _lower_smem(self, mem: str, pipe: str, side: str) -> ast.expr:
        placement = self._mapped_placement(mem)
        if placement is not None and placement.space == "smem":
            base = placement.physical
            base_mem = self.graph["memories"][base]
            elem_size = _dtype_size(base_mem.dtype)
            stage = _stage_expr(pipe, side)
            elem_offset = f"(({placement.offset}) + ({stage}) * {placement.stage_stride}) / {elem_size}"
            return _cpp_raw_expr(f"(__lk_smem.{base}[0] + {elem_offset})", types.Pointer(_element_type(base_mem.dtype)))
        return _smem_expr(mem, pipe, side, self.graph["memories"][mem])


def role_materialize_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY, {})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in graphs:
            RoleResourceMaterializer(graphs[node.name]).visit(node)
    ast.fix_missing_locations(tree)
    return tree
