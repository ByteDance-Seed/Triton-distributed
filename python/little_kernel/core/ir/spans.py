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
"""Source spans attached to lowered IR operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class SourceSpan:
    """An immutable source position for a lowered operation.

    Line/column numbers mirror CPython's ``ast`` convention (1-based lines,
    0-based columns). ``lineno=None`` marks a synthesized node that has no
    origin in authored source; such nodes are still legal but report
    ``"<synthesized>"`` instead of a location.
    """

    lineno: Optional[int] = None
    col_offset: Optional[int] = None
    end_lineno: Optional[int] = None
    end_col_offset: Optional[int] = None
    source: Optional[str] = None

    @property
    def is_synthetic(self) -> bool:
        return self.lineno is None

    def format(self) -> str:
        if self.lineno is None:
            loc = "<synthesized>"
        elif self.col_offset is None:
            loc = f"line {self.lineno}"
        else:
            loc = f"line {self.lineno}, column {self.col_offset + 1}"
        if self.source:
            return f"{self.source}:{loc}"
        return loc

    def __str__(self) -> str:
        return self.format()


def span_from_node(node: Any) -> SourceSpan:
    """Extract a :class:`SourceSpan` from an ``ast.AST``-like node.

    Lowered nodes already expose a ``span`` attribute directly; ordinary
    ``ast`` nodes expose ``lineno``/``col_offset`` (and the ``end_*`` pair on
    Python 3.8+). Anything else yields a synthetic span.
    """
    if node is None:
        return SourceSpan()
    span = getattr(node, "span", None)
    if isinstance(span, SourceSpan):
        return span
    if hasattr(node, "lineno") and getattr(node, "lineno", None) is not None:
        return SourceSpan(
            lineno=getattr(node, "lineno", None),
            col_offset=getattr(node, "col_offset", None),
            end_lineno=getattr(node, "end_lineno", None),
            end_col_offset=getattr(node, "end_col_offset", None),
            source=getattr(node, "source", None),
        )
    return SourceSpan()
