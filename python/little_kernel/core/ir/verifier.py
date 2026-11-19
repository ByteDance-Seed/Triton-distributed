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
"""Phase verifiers for typed lowered IR (RF06).

The verifier is intentionally structural and backend-independent: it checks
that every lowered node has a declared LLType result (for expressions), a
declared effect, a source span, valid construction invariants, and internally
consistent pointer/stage references. It never tries to parse C++.
"""

from __future__ import annotations

import ast
from typing import List, Optional

from little_kernel.core.type_system import LLType, Pointer

from little_kernel.core.ir.effects import IROpEffect
from little_kernel.core.ir.spans import SourceSpan
from little_kernel.core.ir.nodes import LoweredIRNode, TargetExpression, TargetStatement, TargetInclude, TargetHelper, ResourceReference, StageReference, is_lowered_node, iter_lowered_nodes, TARGET_SECTIONS, RESOURCE_SPACES, _INCLUDE_RE, is_valid_ident_path


class LoweredIRVerificationError(TypeError):
    """Raised when a lowered IR node violates its declared contract."""

    def __init__(self, message: str, node: Optional[ast.AST] = None):
        self.node = node
        where = ""
        if node is not None:
            span = getattr(node, "span", None)
            lineno = getattr(span, "lineno", None) if span is not None \
                else getattr(node, "lineno", None)
            if lineno is not None:
                where = f" (line {lineno})"
        super().__init__(f"{type(node).__name__ if node is not None else 'IR'}"
                         f"{where}: {message}")


def _check(condition: bool, message: str, node) -> None:
    if not condition:
        raise LoweredIRVerificationError(message, node)


def verify_lowered_node(node: ast.AST) -> ast.AST:
    """Verify a single lowered IR node; return it on success.

    Raises :class:`LoweredIRVerificationError` on any contract violation. A
    non-lowered ``ast`` node is returned unchanged (the verifier governs only
    lowered operations).
    """
    if not is_lowered_node(node):
        return node

    span = getattr(node, "span", None)
    _check(isinstance(span, SourceSpan), "missing declared SourceSpan", node)

    if isinstance(node, TargetExpression):
        _check(isinstance(node.spelling, str) and node.spelling.strip(), "spelling must be a non-empty string", node)
        _check(isinstance(node.result_type, LLType), "result_type must be an LLType", node)
        _check(isinstance(node.effect, IROpEffect), "effect must be an IROpEffect", node)

    elif isinstance(node, TargetStatement):
        _check(isinstance(node.spelling, str) and node.spelling.strip(), "spelling must be a non-empty string", node)
        _check(isinstance(node.effect, IROpEffect), "effect must be an IROpEffect", node)
        # A statement exists for its side effect. A "pure" statement is dead
        # code the verifier cannot soundly keep or move.
        _check(node.effect is not IROpEffect.PURE, "statement must declare a side effect (not pure)", node)
        _check(node.section in TARGET_SECTIONS, f"section must be one of {TARGET_SECTIONS}", node)

    elif isinstance(node, TargetInclude):
        _check(
            isinstance(node.include, str) and bool(_INCLUDE_RE.match(node.include)),
            "include must be an angle/quoted #include directive", node)

    elif isinstance(node, TargetHelper):
        _check(isinstance(node.spelling, str) and node.spelling.strip(), "spelling must be a non-empty string", node)

    elif isinstance(node, ResourceReference):
        # The resource path is required and must be a valid C++ identifier
        # path: a missing/None path previously slipped through and codegen
        # emitted ``T* p = None;``.
        _check(is_valid_ident_path(node.resource), "resource path is required and must be a valid identifier "
               "path (no operators/calls/commas)", node)
        _check(isinstance(node.element_type, LLType), "element_type must be an LLType", node)
        _check(node.space in RESOURCE_SPACES, f"space must be one of {RESOURCE_SPACES}", node)
        _check(isinstance(node.effect, IROpEffect), "effect must be an IROpEffect", node)
        _check(isinstance(node.result_type, Pointer), "resource reference result must be a Pointer type", node)
        _check(node.result_type.inner_type is node.element_type or node.result_type.inner_type == node.element_type,
               "pointer result element type must match element_type", node)

    elif isinstance(node, StageReference):
        _check(isinstance(node.base, ast.expr), "base must be an AST expression", node)
        _check(isinstance(node.stage, ast.expr), "stage must be an AST expression", node)
        _check(
            isinstance(node.stride, int) and not isinstance(node.stride, bool) and node.stride >= 0,
            "stride must be a non-negative integer", node)
        _check(isinstance(node.element_type, LLType), "element_type must be an LLType", node)
        _check(isinstance(node.effect, IROpEffect), "effect must be an IROpEffect", node)
        _check(isinstance(node.result_type, Pointer), "stage reference result must retain a Pointer type", node)
        _check(node.result_type.inner_type is node.element_type or node.result_type.inner_type == node.element_type,
               "stage pointer element type must match element_type", node)
        base_type = getattr(node.base, "result_type", None)
        if base_type is not None:
            _check(isinstance(base_type, Pointer), "stage base must itself have a pointer result type", node)
            _check(base_type.inner_type == node.result_type.inner_type,
                   "stage result element type must match base pointer", node)
        # Recurse into nested lowered operands.
        verify_lowered_node(node.base)
        verify_lowered_node(node.stage)

    return node


def verify_lowered_ir(tree: ast.AST, *, raise_on_error: bool = True) -> List[LoweredIRVerificationError]:
    """Verify every lowered node in ``tree``.

    When ``raise_on_error`` is True (default) the first violation raises.
    Otherwise all violations are collected and returned.
    """
    errors: List[LoweredIRVerificationError] = []
    for node in iter_lowered_nodes(tree):
        try:
            verify_lowered_node(node)
        except LoweredIRVerificationError as exc:
            if raise_on_error:
                raise
            errors.append(exc)
    return errors
