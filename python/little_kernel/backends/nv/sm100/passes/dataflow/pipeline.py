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
"""Canonical role/dataflow pipeline and public frontend hook (RF05/RF07).

This is the single canonical owner of the role/dataflow pass ordering. The
source helpers and public LLKernel frontend integration both call
:func:`dataflow_frontend`; there is no second,
divergent pipeline.

Selection is explicit: :func:`tree_has_dataflow` returns True only for modules
that actually contain a ``with <...>.dataflow() as g:`` declaration block, so
ordinary scalar/SIMT kernels never run role passes.
"""

from __future__ import annotations

import ast
import hashlib
import linecache
from importlib.resources import files
from typing import Callable, Optional

from little_kernel.backends.nv.sm100.passes.dataflow.graph_construction import CLUSTER_SHAPE_CTX_KEY, CTA_GROUP_SHAPE_CTX_KEY, DATAFLOW_CTX_KEY, build_kernel_ctx, dataflow_decl
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import tmem_register_array_expansion_pass
from little_kernel.backends.nv.sm100.passes.dataflow.compile_time_unroll import compile_time_unroll_pass
from little_kernel.backends.nv.sm100.passes.dataflow.umma_modes import umma_codegen_coalescing_pass, umma_effect_normalization_pass
from little_kernel.backends.nv.sm100.passes.dataflow.transforms import dataflow_analysis_pass, pipeline_effect_partition_pass
from little_kernel.backends.nv.sm100.passes.dataflow.tmem_epilogue import tmem_epilogue_ordering_pass
from little_kernel.backends.nv.sm100.passes.dataflow.rmw import rmw_analysis_pass
from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import buffer_overlay_analysis_pass
from little_kernel.backends.nv.sm100.passes.dataflow.scheduling import overlay_pipeline_pass, pipeline_protocol_validation_pass, tma_store_stream_analysis_pass
from little_kernel.backends.nv.sm100.passes.dataflow.normalization import compile_time_branch_prune_pass, inline_functions_pass
from little_kernel.backends.nv.sm100.codegen.dataflow.lowering import role_materialize_pass
from little_kernel.backends.nv.sm100.codegen.dataflow.pipeline import pipeline_primitive_pass
from little_kernel.backends.nv.sm100.passes.dataflow.guard_coalescing import merge_repeated_guards_pass

#: Immutable canonical role/dataflow transform ordering. Each element is a
#: ``(pass_function, stable_id)`` pair so a compilation session can validate
#: phase ordering and derive an idempotent pipeline.
DATAFLOW_PIPELINE: tuple[tuple[Callable[..., ast.AST], str], ...] = (
    (inline_functions_pass, "role.inline_functions"),
    (compile_time_branch_prune_pass, "role.compile_time_branch_prune"),
    (compile_time_unroll_pass, "role.compile_time_unroll"),
    (tmem_register_array_expansion_pass, "role.tmem_register_array_expansion"),
    (umma_effect_normalization_pass, "role.umma_effect_normalization"),
    (dataflow_analysis_pass, "role.dataflow_analysis"),
    (tmem_epilogue_ordering_pass, "role.tmem_epilogue_ordering"),
    (pipeline_effect_partition_pass, "role.pipeline_effect_partition"),
    (rmw_analysis_pass, "role.rmw_analysis"),
    (buffer_overlay_analysis_pass, "role.buffer_overlay_analysis"),
    (overlay_pipeline_pass, "role.overlay_pipeline"),
    (tma_store_stream_analysis_pass, "role.tma_store_stream_analysis"),
    (pipeline_protocol_validation_pass, "role.pipeline_protocol_validation"),
    (pipeline_primitive_pass, "role.pipeline_primitive"),
    (role_materialize_pass, "role.resource_materialize"),
    (merge_repeated_guards_pass, "role.guard_coalescing"),
    (umma_codegen_coalescing_pass, "role.umma_codegen_coalescing"),
)

#: Role transforms complete in the frontend: they emit typed lowered nodes
#: (:mod:`little_kernel.core.ir`) which ordinary type inference/codegen already
#: resolve. No hidden post-lowering role pass is required; this explicit tuple
#: exists so a session hook can declare extra passes without reaching into
#: private state.
DATAFLOW_POST_LOWERING_PASSES: tuple[Callable[..., ast.AST], ...] = ()


def dataflow_pass_list() -> list[Callable[..., ast.AST]]:
    """Return a fresh copy of the canonical role pass functions.

    The returned list is a private copy; callers may customize it without
    mutating the shared immutable pipeline specification (RF07).
    """
    return [pass_fn for pass_fn, _stable_id in DATAFLOW_PIPELINE]


def tree_has_dataflow(tree: ast.AST) -> bool:
    """Return True only if ``tree`` declares a role ``dataflow()`` block."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and \
                dataflow_decl(node, allow_missing=True) is not None:
            return True
    return False


def run_dataflow_pipeline(tree: ast.Module, ctx: dict, *, through: str | None = None) -> ast.Module:
    """Apply the canonical transforms, optionally ending at an analysis stage."""
    if through is not None and through not in {name for _, name in DATAFLOW_PIPELINE}:
        raise ValueError(f"Unknown dataflow stage: {through}")
    for pass_fn, stable_id in DATAFLOW_PIPELINE:
        tree = pass_fn(tree, ctx)
        if stable_id == through:
            break
    return tree


def dataflow_frontend(tree: ast.Module, ctx: dict) -> ast.Module:
    """Public LLKernel frontend hook for authored role/dataflow kernels.

    Frontend hook contract::

        frontend(tree, ctx) -> ast.Module

    Called on a fresh compilation session *before* ordinary passes, type
    inference and codegen. It is a no-op for modules that do not declare a
    ``with dataflow()`` block, so default scalar compiles stay ordinary even if
    a session wires this hook unconditionally.
    """
    if not tree_has_dataflow(tree):
        return tree
    return run_dataflow_pipeline(tree, ctx)


def _exec_source(source: str) -> dict:
    filename = f"<dataflow-kernel-{hashlib.sha256(source.encode()).hexdigest()[:16]}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    module_ns: dict = {}
    exec(compile(source, filename, "exec"), module_ns)
    return module_ns


def _entry_kernel(module_ns: dict):
    from little_kernel.core.compile import LLKernel

    return next(v for v in module_ns.values() if isinstance(v, LLKernel) and v.is_entry)


def _context_for(kernel, param_values: Optional[dict]) -> dict:
    ctx = build_kernel_ctx(kernel, param_values)
    ctx[CLUSTER_SHAPE_CTX_KEY] = kernel.cluster_shape
    ctx[CTA_GROUP_SHAPE_CTX_KEY] = kernel.cta_group_shape
    return ctx


def analyze_dataflow_source(source: str, param_values: Optional[dict] = None) -> dict:
    """Analyze an authored role module and return its dataflow graph."""
    module_ns = _exec_source(source)
    kernel = _entry_kernel(module_ns)
    module = ast.parse(source)
    ctx = _context_for(kernel, param_values)
    run_dataflow_pipeline(module, ctx, through="role.tma_store_stream_analysis")
    return next(iter(ctx[DATAFLOW_CTX_KEY].values()))


def lower_dataflow_source(source: str, param_values: Optional[dict] = None) -> tuple[ast.Module, dict, object]:
    """Lower an authored role module through the canonical role pipeline."""
    module_ns = _exec_source(source)
    kernel = _entry_kernel(module_ns)
    module = ast.parse(source)
    ctx = _context_for(kernel, param_values)
    module = run_dataflow_pipeline(module, ctx)
    return module, ctx, kernel


def compile_dataflow_cuda(
    source: str,
    param_values: Optional[dict] = None,
    *,
    num_threads: int = 256,
    disable_var_hoisting: bool = False,
) -> str:
    """Canonical CPU public API: role transforms + ordinary CUDA codegen.

    This is the one canonical pipeline that legacy string/demo helpers
    delegate to. It reads only the source string supplied by the caller; it
    never reads neighboring example files at import time.
    """
    from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
    from little_kernel.core.passes import PASSES

    module, ctx, kernel = lower_dataflow_source(source, param_values)
    module.body = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == kernel.py_func.__name__
    ]
    for p in PASSES["cuda"]:
        module = p(module, ctx)

    cuda_source = codegen_cuda(module, ctx, num_threads=num_threads, disable_var_hoisting=disable_var_hoisting)
    header = files("little_kernel.backends.nv.sm100.runtime").joinpath("include/role_pipeline.cuh")
    pipeline_hash = hashlib.sha256(header.read_bytes()).hexdigest()[:16]
    return f"// role-pipeline-header-sha256: {pipeline_hash}\n{cuda_source}"
