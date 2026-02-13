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
import ast
from typing import Dict, Any
from .utils.preserve_attributes import get_special_attrs
from .utils.enhanced_unparse import enhanced_unparse
from .utils.enhanced_dump import enhanced_dump


class SpecialAttrsCollector(ast.NodeVisitor):
    """Collect all nodes with special attributes."""

    def __init__(self):
        self.nodes_with_attrs: Dict[ast.AST, Dict[str, Any]] = {}

    def visit(self, node: ast.AST):
        if node is None:
            return

        # Check if this node has special attributes
        special_attrs = get_special_attrs(node)
        if special_attrs:
            self.nodes_with_attrs[node] = special_attrs

        # Visit children
        super().visit(node)
        return node


def _format_attr_value(value: Any) -> str:
    """Format an attribute value for display."""
    if isinstance(value, ast.AST):
        return f"<AST: {type(value).__name__}>"
    elif isinstance(value, dict):
        return f"<dict with {len(value)} keys: {list(value.keys())[:3]}{'...' if len(value) > 3 else ''}>"
    elif isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} with {len(value)} items>"
    elif isinstance(value, str) and len(value) > 50:
        return f"'{value[:50]}...'"
    else:
        return repr(value)


def _dump_node_with_attrs(node: ast.AST, indent: int = 0) -> str:
    """Dump a single node with its special attributes."""
    lines = []
    indent_str = " " * indent

    # Get node type and location
    node_type = type(node).__name__
    lineno = getattr(node, 'lineno', None)
    col_offset = getattr(node, 'col_offset', None)
    location = f" (line {lineno}, col {col_offset})" if lineno is not None else ""

    # Get special attributes
    special_attrs = get_special_attrs(node)

    if special_attrs:
        lines.append(f"{indent_str}{node_type}{location}:")
        indent_str += "  "
        for attr_name, attr_value in special_attrs.items():
            formatted_value = _format_attr_value(attr_value)
            lines.append(f"{indent_str}{attr_name} = {formatted_value}")
    else:
        lines.append(f"{indent_str}{node_type}{location}: (no special attributes)")

    return "\n".join(lines)


def dump_ast(msg: str, simple_mode=True):
    """Create a pass that dumps the AST with special attributes."""

    def _dump_ast(tree, ctx, *args, **kwargs):
        print(msg)
        print("=" * 80)

        # Dump enhanced AST (with custom nodes and attributes inline)
        # Pass ctx to enable intrin detection
        if simple_mode:
            print(enhanced_unparse(tree, show_custom_attrs=True, show_ir_nodes=True, ctx=ctx))
        else:
            print(enhanced_dump(tree, indent=2, include_attributes=True, show_custom_attrs=True, ctx=ctx))

        print("=" * 80)
        print()

        return tree

    return _dump_ast
