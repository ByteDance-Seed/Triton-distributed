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
"""Inline-callsite provenance for compiler diagnostics.

When an outlined function (e.g. an ``@lk.ll_kernel(is_entry=False)`` device
subroutine, or a user-level helper marked with the inline attribute) is
inlined at multiple callsites, downstream passes see N copies of the same
body.  Any diagnostic raised inside one of those copies must identify
*which* copy produced the failure.  Otherwise users have to resort to
divide-and-conquer (commenting out callsites one at a time) to find the
culprit.

This module is the canonical home for the provenance record and for the
helpers that:

* build a provenance record at inline time
  (:func:`make_inline_provenance`, :func:`attach_inline_provenance`),
* read a provenance record from an arbitrary AST node
  (:func:`get_inline_provenance`, :func:`find_inline_provenance`),
* render a human-readable suffix for error messages
  (:func:`format_inline_provenance`).

The attribute name and the dataclass are stable: other passes and code
generators that want provenance-aware diagnostics should consume them here
rather than re-implementing the search.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Optional

# Name of the attribute stamped on every AST node that came out of an
# inlined callsite.  Listed in ``builtin_base.AST_SPECIAL_ATTRS`` so
# ``preserve_attributes`` copies it across node rewrites.
INLINE_PROVENANCE_ATTR = "_lk_inline_provenance"


@dataclass(frozen=True)
class InlineProvenance:
    """Identifies one inlined copy of an outlined function body.

    Fields:
        function_name: Qualified-ish name of the inlined function (e.g.
            ``"_leaf"`` or ``"pkg.mod._leaf"``).  Never empty.
        copy_index: 1-based index of this inlined expansion within the
            current outermost function.  Increments for every inline
            expansion, including repeated callsites of the same callee.
        callsite_lineno: 1-based source line of the :py:class:`ast.Call`
            node that triggered this inline expansion, if available.
        callsite_col: 0-based source column of that same call, if
            available.
    """

    function_name: str
    copy_index: int
    callsite_lineno: Optional[int] = None
    callsite_col: Optional[int] = None

    def short(self) -> str:
        """Compact human-readable tag, e.g. ``copy 2 of _leaf``."""
        return f"copy {self.copy_index} of {self.function_name}"

    def describe(self) -> str:
        """Full sentence fragment used in diagnostics."""
        loc = ""
        if self.callsite_lineno is not None:
            loc = f" (callsite line {self.callsite_lineno}"
            if self.callsite_col is not None:
                loc += f", column {self.callsite_col}"
            loc += ")"
        return f"in inlined {self.short()}{loc}"


def make_inline_provenance(function_name: str, copy_index: int, call_node: Optional[ast.AST]) -> InlineProvenance:
    """Build an :class:`InlineProvenance` for one inline expansion."""
    lineno: Optional[int] = None
    col: Optional[int] = None
    if isinstance(call_node, ast.AST):
        lineno = getattr(call_node, "lineno", None)
        col = getattr(call_node, "col_offset", None)
    return InlineProvenance(
        function_name=function_name or "<anonymous>",
        copy_index=int(copy_index),
        callsite_lineno=lineno,
        callsite_col=col,
    )


def _is_stamping_safe(node: ast.AST) -> bool:
    """Return False for AST node kinds that must never carry provenance.

    CPython's parser interns singleton instances for the context-only node
    kinds (``Load``/``Store``/``Del`` and the operator/precedence leaf
    kinds): the *same* object is shared across independent ``ast.parse``
    calls.  Stamping provenance on one of those would silently leak the
    tag into every tree parsed afterwards, mis-attributing later errors.
    Error reporting only ever points at statements/expressions, so
    skipping leaf context/operator kinds loses nothing.
    """
    return not isinstance(node, (ast.expr_context, ast.operator, ast.unaryop, ast.cmpop, ast.boolop))


def attach_inline_provenance(node: ast.AST, provenance: InlineProvenance) -> ast.AST:
    """Recursively stamp ``provenance`` on ``node`` and all descendants.

    Every node created by splicing the inlined body into the caller gets
    the provenance so any downstream error pointing at any of them can be
    attributed back to the correct inlined copy.  Existing provenance is
    not overwritten: nested inline expansions keep their innermost tag,
    which is the most specific attribution available.
    """
    if not isinstance(node, ast.AST):
        return node
    for child in ast.walk(node):
        if not _is_stamping_safe(child):
            continue
        if not hasattr(child, INLINE_PROVENANCE_ATTR):
            setattr(child, INLINE_PROVENANCE_ATTR, provenance)
    return node


def get_inline_provenance(node: Optional[ast.AST]) -> Optional[InlineProvenance]:
    """Return the :class:`InlineProvenance` attached to ``node``, if any.

    Singleton leaf kinds (``Load``/``Store``/operators) are never queried:
    CPython interns them across parses, so any tag found on one is by
    definition leaked state from an unrelated tree, not real provenance.
    """
    if not isinstance(node, ast.AST) or not _is_stamping_safe(node):
        return None
    prov = getattr(node, INLINE_PROVENANCE_ATTR, None)
    return prov if isinstance(prov, InlineProvenance) else None


def find_inline_provenance(node: Optional[ast.AST]) -> Optional[InlineProvenance]:
    """Return the provenance attached to ``node``, or nearest ancestor.

    The first stop is the node itself (``get_inline_provenance``).  If the
    node is untagged -- e.g. a pass rewrote it without preserving special
    attributes -- we optionally climb a ``_lk_parent`` chain when the
    surrounding pipeline maintains one.  Returns ``None`` when no
    provenance can be found; callers must tolerate that (the tag is a
    diagnostic aid, not a correctness requirement).
    """
    seen: set[int] = set()
    current: Optional[ast.AST] = node
    while isinstance(current, ast.AST) and id(current) not in seen:
        seen.add(id(current))
        prov = get_inline_provenance(current)
        if prov is not None:
            return prov
        current = getattr(current, "_lk_parent", None)
        if current is None:
            break
    return None


def format_inline_provenance(prov: Optional[InlineProvenance]) -> str:
    """Render ``prov`` as a single-line suffix for error messages.

    Returns an empty string when ``prov`` is ``None`` so callers can do
    ``msg + format_inline_provenance(...)`` unconditionally.
    """
    if prov is None:
        return ""
    return f"\nInline provenance: {prov.describe()}"
