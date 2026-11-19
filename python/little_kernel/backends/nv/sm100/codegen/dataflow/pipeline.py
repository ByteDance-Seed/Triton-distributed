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
"""Emit CUDA pipeline primitives from analyzed stage and store-stream plans."""
from __future__ import annotations
import ast
from little_kernel.backends.nv.sm100.passes.dataflow.constants import DATAFLOW_CTX_KEY
from little_kernel.backends.nv.sm100.passes.dataflow.graph_ops import _pipeline_var_name
from little_kernel.backends.nv.sm100.passes.dataflow.transforms import build_stage_region_insertion_plan
from little_kernel.backends.nv.sm100.codegen.dataflow.target_ops import cpp_stmt as _cpp_stmt


def _pipeline_call(pipe: str, method: str) -> ast.stmt:
    return ast.Expr(value=ast.Call(
        func=ast.Attribute(value=ast.Name(id=pipe, ctx=ast.Load()), attr=method, ctx=ast.Load()),
        args=[],
        keywords=[],
    ))


def _stream_release_code(group: dict) -> str:
    return " ".join(f"{pipe}.retire_completed();" for pipe in group["completions"])


def _stream_prepare_code(stream: dict, item: dict) -> str:
    groups = stream["groups"]
    window = stream["window"]
    position = item["position"]
    required = set(item["pipelines"])
    covered: set[str] = set()
    code = []

    for keep in range(window - 1, -1, -1):
        group = groups[(position - keep - 1) % len(groups)]
        condition = " || ".join(f"{pipe}.consumer_full()" for pipe in sorted(required))
        pending = f"{stream['name']}.pending()"
        if keep == window - 1 and item["capacity"]:
            full = f"{pending} == {window}"
            condition = (f"{full} || ({pending} > {keep} && ({condition}))" if condition else full)
        elif condition:
            condition = f"{pending} > {keep} && ({condition})"
        if not condition:
            break
        code.append(f"if ({condition}) {{ "
                    f"{stream['name']}.wait<{keep}>(); "
                    f"{_stream_release_code(group)} }}")
        covered.update(set(group["completions"]) & required)
        if covered == required:
            break

    if required - covered:
        raise ValueError(f"{stream['role']}: TMA store completion is outside the "
                         f"{window}-group window")
    return " ".join(code)


def _pipeline_decl(graph: dict, pipeline: dict) -> ast.Assign:
    return ast.Assign(
        targets=[ast.Name(id=_pipeline_var_name(pipeline), ctx=ast.Store())],
        value=ast.Call(
            func=ast.Attribute(value=ast.Name(id="ri", ctx=ast.Load()), attr="Pipeline", ctx=ast.Load()),
            args=[],
            keywords=[],
        ),
    )


class PipelinePrimitiveInserter(ast.NodeTransformer):
    BEFORE_ORDER = {"consumer_wait": 0, "producer_acquire": 1}
    AFTER_ORDER = {
        "producer_commit": 0,
        "consumer_release": 1,
    }

    def __init__(self, graph: dict) -> None:
        self.graph = graph
        plan = build_stage_region_insertion_plan(graph)
        self.before = plan.before
        self.after = plan.after
        self.bind_stmt_ids = set(graph.get("bind_stmt_ids", ()))
        self.role_tail = plan.role_tail
        self.live_pipelines = plan.live_pipelines
        self.raw_before: dict[int, list[str]] = {}
        self.raw_after: dict[int, list[str]] = {}
        self.raw_late_after: dict[int, list[str]] = {}
        self.raw_role_tail: dict[str, list[str]] = {}
        for stream in graph.get("tma_store_streams", ()):
            for item in stream["prepare"]:
                code = _stream_prepare_code(stream, item)
                if code:
                    target = self.raw_late_after if item["late"] else self.raw_before
                    target.setdefault(item["site"], []).append(code)
            for group in stream["groups"]:
                self.raw_after.setdefault(group["last"], []).append(f"{stream['name']}.commit();")
            self.raw_role_tail.setdefault(stream["role"], []).append(f"{stream['name']}.tail();")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.generic_visit(node)
        decls = [_pipeline_decl(self.graph, pipeline) for pipeline in self.live_pipelines]
        if not decls:
            return node
        for idx, stmt in enumerate(node.body):
            if id(stmt) == self.graph["dataflow_node_id"]:
                node.body[idx + 1:idx + 1] = decls
                break
        for role, codes in self.raw_role_tail.items():
            node.body.append(
                ast.With(
                    items=[ast.withitem(context_expr=ast.Name(id=role, ctx=ast.Load()), optional_vars=None)],
                    body=[_cpp_stmt(code) for code in codes],
                ))
        return node

    def visit_With(self, node: ast.With):
        node = self.generic_visit(node)
        if len(node.items) == 1 and isinstance(node.items[0].context_expr, ast.Name):
            role = node.items[0].context_expr.id
            for pipe in self.role_tail.get(role, []):
                node.body.append(_pipeline_call(pipe, "producer_tail"))
        return node

    def visit(self, node):
        if isinstance(node, ast.stmt) and id(node) in self.bind_stmt_ids:
            return []
        out = super().visit(node)
        if not isinstance(node, ast.stmt) or isinstance(out, list):
            return out
        before = sorted(self.before.get(id(node), []), key=lambda site: self.BEFORE_ORDER.get(site[1], 99))
        after = sorted(self.after.get(id(node), []), key=lambda site: self.AFTER_ORDER.get(site[1], 99))
        raw_before = self.raw_before.get(id(node), [])
        raw_after = self.raw_after.get(id(node), [])
        raw_late_after = self.raw_late_after.get(id(node), [])
        if (not before and not after and not raw_before and not raw_after and not raw_late_after):
            return out
        return ([_cpp_stmt(code) for code in raw_before] + [_pipeline_call(*site) for site in before] + [out] +
                [_cpp_stmt(code)
                 for code in raw_after] + [_pipeline_call(*site)
                                           for site in after] + [_cpp_stmt(code) for code in raw_late_after])


def insert_pipeline_primitives(func: ast.FunctionDef, graph: dict) -> ast.FunctionDef:
    return PipelinePrimitiveInserter(graph).visit(func)


def pipeline_primitive_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    ctx = ctx if ctx is not None else {}
    if DATAFLOW_CTX_KEY not in ctx:
        raise ValueError("dataflow analysis result is not found in compilation context")

    graphs = ctx.get(DATAFLOW_CTX_KEY, {})
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in graphs:
            insert_pipeline_primitives(node, graphs[node.name])
    ast.fix_missing_locations(tree)
    return tree
