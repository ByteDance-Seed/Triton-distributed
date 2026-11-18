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
"""Transitional read compatibility for legacy ``_lk_cpp_*`` attributes.

Before RF06, role lowering hid raw backend text on unrelated standard nodes:

* ``ast.Pass`` with ``_lk_cpp_stmt`` / ``_lk_cpp_include`` /
  ``_lk_cpp_builtin``;
* ``ast.Name(id="__lk_cpp_expr")`` with ``_lk_cpp_expr``.

This module is the ONLY place that recognizes those attributes. New lowering
MUST construct the explicit typed nodes in :mod:`little_kernel.core.ir.nodes`.
These helpers remain only until the dataflow worker migrates every producer;
they never participate in the construction or verification of new nodes.
"""

from __future__ import annotations

import ast
from enum import Enum
from typing import Any, Optional

from little_kernel.core.type_system import LLType, int32

from little_kernel.core.ir.spans import span_from_node
from little_kernel.core.ir.effects import IROpEffect
from little_kernel.core.ir.nodes import TargetExpression, TargetStatement, TargetInclude, TargetHelper

#: The sentinel identifier the legacy lowering used for raw expressions.
LEGACY_CPP_EXPR_NAME = "__lk_cpp_expr"


class LegacyCppKind(str, Enum):
    STATEMENT = "stmt"
    INCLUDE = "include"
    BUILTIN = "builtin"
    EXPRESSION = "expression"


def is_legacy_cpp_node(node: Any) -> bool:
    """Return True if ``node`` carries a legacy backend-text attribute."""
    return legacy_cpp_kind(node) is not None


def legacy_cpp_kind(node: Any) -> Optional[LegacyCppKind]:
    if not isinstance(node, ast.AST):
        return None
    if isinstance(node, ast.Pass):
        if getattr(node, "_lk_cpp_include", None) is not None:
            return LegacyCppKind.INCLUDE
        if getattr(node, "_lk_cpp_builtin", None) is not None:
            return LegacyCppKind.BUILTIN
        if getattr(node, "_lk_cpp_stmt", None) is not None:
            return LegacyCppKind.STATEMENT
        return None
    if isinstance(node, ast.Name) and getattr(node, "_lk_cpp_expr", None) \
            is not None:
        return LegacyCppKind.EXPRESSION
    return None


def legacy_cpp_spelling(node: Any) -> str:
    """Return the raw backend text on a legacy node."""
    kind = legacy_cpp_kind(node)
    if kind is None:
        raise TypeError("node is not a legacy _lk_cpp_* node")
    if kind is LegacyCppKind.STATEMENT:
        return node._lk_cpp_stmt
    if kind is LegacyCppKind.INCLUDE:
        return node._lk_cpp_include
    if kind is LegacyCppKind.BUILTIN:
        return node._lk_cpp_builtin
    return node._lk_cpp_expr


def legacy_cpp_expr_type(node: Any, default: LLType = int32) -> LLType:
    """Declared result type for a legacy raw expression.

    A producer may attach an explicit LLType via the non-standard
    ``_lk_cpp_type`` (this lets migrating producers retain pointer types
    before they switch to :class:`TargetExpression`). When absent, the
    historical ``int32`` is returned solely to preserve pre-RF06 behavior for
    unmigrated producers; new code must use an explicit typed node and never
    rely on this fallback.
    """
    if legacy_cpp_kind(node) is not LegacyCppKind.EXPRESSION:
        raise TypeError("node is not a legacy _lk_cpp_expr node")
    declared = getattr(node, "_lk_cpp_type", None)
    if isinstance(declared, LLType):
        return declared
    return default


def upgrade_legacy_node(node: ast.AST, *, expr_type: Optional[LLType] = None):
    """Convert a legacy attribute-bearing node to an explicit typed IR node.

    The returned node copies the original source location. This is provided
    for the migrating dataflow worker; production codegen/inference also use
    it internally so there is a single typed emission path.
    """
    kind = legacy_cpp_kind(node)
    if kind is None:
        raise TypeError("node is not a legacy _lk_cpp_* node")
    span = span_from_node(node)
    if kind is LegacyCppKind.STATEMENT:
        return TargetStatement(node._lk_cpp_stmt, effect=IROpEffect.ESCAPE, section="main", span=span)
    if kind is LegacyCppKind.BUILTIN:
        return TargetStatement(node._lk_cpp_builtin, effect=IROpEffect.ESCAPE, section="builtin", span=span)
    if kind is LegacyCppKind.INCLUDE:
        return TargetInclude(node._lk_cpp_include, span=span)
    # Legacy raw expressions require an explicitly supplied/declared type to
    # upgrade: there is no sound int32 default for a pointer-valued escape.
    if expr_type is None:
        expr_type = legacy_cpp_expr_type(node)
    return TargetExpression(node._lk_cpp_expr, expr_type, effect=IROpEffect.ESCAPE, span=span)
