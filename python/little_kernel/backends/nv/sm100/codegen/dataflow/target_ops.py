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
"""Typed target-operation factories for role/dataflow lowering (RF06).

Neutral module shared by the structural transforms and the resource lowerer:
every backend spelling becomes an explicit typed lowered IR node with a
declared effect and an inferred pointer/scalar result type. Nothing here
produces opaque ``_lk_cpp_*`` attributes or defaults expressions to int32.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Optional

from little_kernel.core import ir as lk_ir
from little_kernel.core.ir import IROpEffect
from little_kernel.core.type_system import LLType, Pointer, bool_, uint8, uint32, uint64

#: Canonical, packaged role-pipeline backend header (RF15 resource owner).
#: Resolved against the import-package root, added by runtime compilation for
#: both source checkouts and installed wheels.
PIPELINE_HEADER = '"little_kernel/backends/nv/sm100/runtime/include/role_pipeline.cuh"'

_BARRIER_RE = re.compile(r"(__syncthreads|cluster_sync|init_pipelines|_transaction\(\))")


def is_barrier_spelling(code: str) -> bool:
    return bool(_BARRIER_RE.search(code))


def cpp_stmt(code: str, *, source: Optional[ast.AST] = None) -> lk_ir.TargetStatement:
    """A typed backend statement with a declared ordering effect."""
    effect = (IROpEffect.BARRIER if is_barrier_spelling(code) else IROpEffect.ESCAPE)
    return lk_ir.TargetStatement(code, effect=effect, section="main", source=source)


def cpp_include(include: str, *, source: Optional[ast.AST] = None) -> lk_ir.TargetInclude:
    return lk_ir.TargetInclude(include, source=source)


def cpp_builtin(code: str, *, source: Optional[ast.AST] = None) -> lk_ir.TargetStatement:
    return lk_ir.TargetStatement(code, effect=IROpEffect.ESCAPE, section="builtin", source=source)


def infer_target_type(code: str) -> LLType:
    """Resolve only closed, known backend operations; unknown text needs metadata.

    TMEM column addresses are encoded uint32 values, not generic pointers.
    SMEM element types come from the resource declaration at the producer.
    """
    if re.fullmatch(r"[A-Za-z_]\w*\.(producer|consumer)_get_barrier\(\)", code):
        return Pointer(uint64)
    if re.fullmatch(r"__lk_tmem_[A-Za-z_]\w*", code):
        return uint32
    if re.fullmatch(r"[A-Za-z_]\w*\.(producer|consumer)_stage\(\)", code):
        return uint32
    prefix = "little_kernel::role_pipeline::"
    if code.startswith(prefix):
        operation = code[len(prefix):].split("<", 1)[0]
        if operation in {"cta_group_is_leader", "tma_multicast_is_issuer"}:
            return bool_
        if operation in {"rank_in_group", "cluster_coord", "group_coord", "tma_multicast_mask"}:
            return uint32
    raise TypeError(f"Unknown target expression {code!r}; supply its declared result_type")


def cpp_raw_expr(code: str, result_type: Optional[LLType] = None, *,
                 source: Optional[ast.AST] = None) -> lk_ir.TargetExpression:
    """A typed backend expression with an explicitly inferred result type."""
    if result_type is None:
        result_type = infer_target_type(code)
    return lk_ir.TargetExpression(code, result_type, effect=IROpEffect.ESCAPE, source=source)
