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
"""
Enhanced AST unparser that supports custom AST nodes and attributes.

This module provides an enhanced version of ast.unparse that can:
1. Handle custom AST node types (e.g., EnumDef, SpecialStructDef)
2. Display custom attributes added to AST nodes
3. Maintain readability unlike ast.dump
"""

import ast
from typing import Any, Dict, Optional, Set
from .preserve_attributes import get_special_attrs
from .ir_nodes import EnumDef, SpecialStructDef, is_ir_node, get_ir_node_type, IRNodeType
from .resolve_attribute import recursive_resolve_attribute
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    CONST_FUNC_ATTR,
    INLINE_FUNC_ATTR,
    CODEGEN_FUNC_ATTR,
)


class EnhancedUnparser:
    """
    Enhanced AST unparser that handles custom nodes and attributes.
    
    Strategy: First unparse the entire tree, then find nodes with custom attributes
    and insert comments at the appropriate positions.
    """

    def __init__(self, show_custom_attrs: bool = True, show_ir_nodes: bool = True, ctx: Optional[Dict[str,
                                                                                                      Any]] = None):
        """
        Initialize the enhanced unparser.
        
        Args:
            show_custom_attrs: Whether to show custom attributes as comments
            show_ir_nodes: Whether to show IR nodes (EnumDef, SpecialStructDef)
            ctx: Optional context dictionary for resolving function objects (for intrin detection)
        """
        self.show_custom_attrs = show_custom_attrs
        self.show_ir_nodes = show_ir_nodes
        self.ctx = ctx
        self.nodes_with_attrs: Dict[ast.AST, Dict[str, Any]] = {}

    def _collect_nodes_with_attrs(self, node: ast.AST):
        """Collect all nodes with custom attributes and intrin info."""
        if node is None:
            return

        if self.show_custom_attrs:
            attrs = get_special_attrs(node)

            # For Call nodes, also try to detect intrin/builtin functions
            if isinstance(node, ast.Call):
                intrin_info = self._get_intrin_info(node)
                if intrin_info:
                    attrs.update(intrin_info)

            if attrs:
                self.nodes_with_attrs[node] = attrs

        # Recursively visit children
        for field, value in ast.iter_fields(node):
            if isinstance(value, ast.AST):
                self._collect_nodes_with_attrs(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._collect_nodes_with_attrs(item)

    def _get_intrin_info(self, node: ast.Call) -> Dict[str, Any]:
        """Try to detect if a Call node is calling an intrin/builtin function."""
        if not isinstance(node, ast.Call):
            return {}

        # Try to resolve the function object if context is available
        func_obj = None
        if self.ctx is not None:
            try:
                if isinstance(node.func, (ast.Name, ast.Attribute)):
                    func_obj = recursive_resolve_attribute(node.func, self.ctx)
            except Exception:
                pass

        # Check if it's a builtin/intrin function
        if func_obj is not None and callable(func_obj):
            intrin_attrs = {}

            # Check for builtin attributes (using constants from builtin_base)
            if hasattr(func_obj, BUILTIN_ATTR) and getattr(func_obj, BUILTIN_ATTR, False):
                intrin_attrs["is_builtin"] = True
                intrin_attrs["builtin_name"] = getattr(func_obj, "__name__", "?")

            if hasattr(func_obj, CODEGEN_FUNC_ATTR):
                intrin_attrs["has_codegen"] = True

            if hasattr(func_obj, EVAL_RETURN_TYPE_ATTR):
                intrin_attrs["has_eval_return_type"] = True

            if hasattr(func_obj, INLINE_FUNC_ATTR) and getattr(func_obj, INLINE_FUNC_ATTR, False):
                intrin_attrs["is_inline"] = True

            if hasattr(func_obj, CONST_FUNC_ATTR) and getattr(func_obj, CONST_FUNC_ATTR, False):
                intrin_attrs["is_const"] = True

            # Check if it's from intrin module
            module = getattr(func_obj, "__module__", "")
            if "intrin" in module:
                intrin_attrs["intrin_module"] = module.split(".")[-1] if "." in module else module

            return intrin_attrs

        # Fallback: check function name pattern for common intrin functions
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Common intrin function patterns (from little_kernel.language.intrin.*)
        if func_name and func_name in [
                "cdiv", "threadIdx", "blockIdx", "blockDim", "gridDim", "syncwarp", "get_lane_idx", "shfl_sync", "ldg",
                "sizeof"
        ]:
            return {"is_intrin_like": True, "intrin_name": func_name}

        return {}

    def _format_attr_value(self, value: Any, max_length: int = 60) -> str:
        """Format an attribute value for display."""
        if isinstance(value, ast.AST):
            node_type = type(value).__name__
            return f"<{node_type}>"
        elif isinstance(value, dict):
            if not value:
                return "{}"
            keys = list(value.keys())[:3]
            keys_str = ", ".join(repr(k) for k in keys)
            if len(value) > 3:
                keys_str += f", ... ({len(value)} total)"
            return f"{{...}}  # keys: {keys_str}"
        elif isinstance(value, (list, tuple)):
            if not value:
                return f"{type(value).__name__}()"
            if len(value) <= 2:
                items_str = ", ".join(self._format_attr_value(v, max_length=30) for v in value)
                return f"{type(value).__name__}({items_str})"
            else:
                return f"{type(value).__name__}(..., {len(value)} items)"
        elif isinstance(value, str):
            if len(value) > max_length:
                return f"'{value[:max_length-3]}...'"
            return repr(value)
        elif isinstance(value, type):
            return f"<class '{value.__name__}'>"
        elif hasattr(value, '__name__'):
            return f"<{type(value).__name__}: {value.__name__}>"
        else:
            val_str = repr(value)
            if len(val_str) > max_length:
                val_str = val_str[:max_length - 3] + "..."
            return val_str

    def _get_attr_comment(self, node: ast.AST) -> str:
        """Get the comment string for a node's custom attributes."""
        if not self.show_custom_attrs:
            return ""

        attrs = self.nodes_with_attrs.get(node)
        if not attrs:
            return ""

        attr_strs = []
        for attr_name, attr_value in attrs.items():
            formatted_value = self._format_attr_value(attr_value, max_length=50)
            attr_strs.append(f"{attr_name}={formatted_value}")

        return "  # " + ", ".join(attr_strs)

    def _insert_attr_comments(self, node: ast.AST, unparsed: str) -> str:
        """Insert attribute comments into unparsed code by matching node positions."""
        if not self.nodes_with_attrs:
            return unparsed

        # For now, use a simpler approach: find statement-level nodes and add comments
        # This works for top-level statements and some nested expressions
        lines = unparsed.split('\n')
        result_lines = []

        # Track which nodes we've already processed
        processed_nodes: Set[ast.AST] = set()

        # For each line, check if it corresponds to a node with attributes
        # We'll use a simple heuristic: match by line number if available
        for i, line in enumerate(lines):
            result_lines.append(line)

            # Try to find nodes that might correspond to this line
            # This is a simplified approach - in practice, we'd need more sophisticated matching
            for node_with_attrs in self.nodes_with_attrs:
                if node_with_attrs in processed_nodes:
                    continue

                # Check if this node might correspond to this line
                # We'll add comments for statement-level nodes
                if isinstance(node_with_attrs, (ast.Assign, ast.Expr, ast.Return, ast.Call)):
                    # For these node types, try to match by checking if the unparsed line
                    # contains the unparsed version of this node
                    try:
                        node_unparsed = ast.unparse(node_with_attrs)
                        # Check if this line contains the node's unparsed form
                        if node_unparsed in line or line.strip().startswith(node_unparsed.split('\n')[0]):
                            comment = self._get_attr_comment(node_with_attrs)
                            if comment:
                                # Add comment to the end of the line
                                result_lines[-1] = line.rstrip() + comment
                                processed_nodes.add(node_with_attrs)
                                break
                    except Exception:
                        pass

        return '\n'.join(result_lines)

    def visit_EnumDef(self, node: EnumDef):
        """Handle EnumDef IR node."""
        if not self.show_ir_nodes:
            return ""

        enum_name = node.enum_name
        enum_class = node.enum_class

        # Get enum members
        try:
            members = list(enum_class.__members__.keys())
            members_str = ", ".join(members[:5])
            if len(members) > 5:
                members_str += f", ... ({len(members)} total)"
        except Exception:
            members_str = "?"

        result = f"# IR: EnumDef(name='{enum_name}', class={enum_class.__name__}, members=[{members_str}])\n"

        # Add custom attributes if any
        if self.show_custom_attrs:
            attrs = get_special_attrs(node)
            if attrs:
                for attr_name, attr_value in attrs.items():
                    formatted_value = self._format_attr_value(attr_value)
                    result += f"# {attr_name} = {formatted_value}\n"

        return result

    def visit_SpecialStructDef(self, node: SpecialStructDef):
        """Handle SpecialStructDef IR node."""
        if not self.show_ir_nodes:
            return ""

        struct_name = node.struct_name

        # Get struct info summary
        struct_info = node.struct_info
        info_summary = []
        if 'class' in struct_info:
            cls = struct_info['class']
            info_summary.append(f"class={cls.__name__}")

        info_str = ", ".join(info_summary) if info_summary else "?"

        result = f"# IR: SpecialStructDef(name='{struct_name}', {info_str})\n"

        # Add custom attributes if any
        if self.show_custom_attrs:
            attrs = get_special_attrs(node)
            if attrs:
                for attr_name, attr_value in attrs.items():
                    formatted_value = self._format_attr_value(attr_value)
                    result += f"# {attr_name} = {formatted_value}\n"

        return result

    def unparse(self, node: ast.AST) -> str:
        """Unparse a node with custom attributes."""
        if node is None:
            return ""

        # Handle custom IR nodes directly
        if is_ir_node(node):
            ir_type = get_ir_node_type(node)
            if ir_type == IRNodeType.ENUM_DEF:
                return self.visit_EnumDef(node)
            elif ir_type == IRNodeType.SPECIAL_STRUCT_DEF:
                return self.visit_SpecialStructDef(node)

        # Special handling for Module: separate IR nodes from regular statements
        if isinstance(node, ast.Module):
            return self._unparse_module(node)

        # Collect all nodes with custom attributes
        self.nodes_with_attrs.clear()
        self._collect_nodes_with_attrs(node)

        # Unparse the entire tree
        try:
            unparsed = ast.unparse(node)
        except Exception as e:
            return f"# <Error unparsing: {e}>\n"

        # If we have nodes with attributes, try to insert comments
        if self.nodes_with_attrs:
            # Use a visitor-based approach to insert comments at the right places
            return self._unparse_with_comments(node, unparsed)

        return unparsed

    def _unparse_module(self, node: ast.Module) -> str:
        """Unparse a Module node, handling IR nodes separately."""
        result_lines = []

        # Separate IR nodes from regular statements
        ir_nodes = []
        regular_stmts = []

        for stmt in node.body:
            if is_ir_node(stmt):
                ir_nodes.append(stmt)
            else:
                regular_stmts.append(stmt)

        # Process IR nodes first
        for ir_node in ir_nodes:
            ir_type = get_ir_node_type(ir_node)
            if ir_type == IRNodeType.ENUM_DEF:
                result_lines.append(self.visit_EnumDef(ir_node).rstrip())
            elif ir_type == IRNodeType.SPECIAL_STRUCT_DEF:
                result_lines.append(self.visit_SpecialStructDef(ir_node).rstrip())

        # Process regular statements
        if regular_stmts:
            # Create a temporary module with only regular statements for unparsing
            temp_module = ast.Module(body=regular_stmts, type_ignores=node.type_ignores)

            # Collect nodes with custom attributes
            self.nodes_with_attrs.clear()
            self._collect_nodes_with_attrs(temp_module)

            # Unparse regular statements
            try:
                unparsed = ast.unparse(temp_module)
                unparsed_lines = unparsed.split('\n')

                # If we have nodes with attributes, insert comments
                if self.nodes_with_attrs:
                    unparsed = self._unparse_with_comments(temp_module, unparsed)
                    unparsed_lines = unparsed.split('\n')

                # Filter out empty lines at the start
                while unparsed_lines and not unparsed_lines[0].strip():
                    unparsed_lines.pop(0)

                result_lines.extend(unparsed_lines)
            except Exception as e:
                result_lines.append(f"# <Error unparsing module: {e}>")

        return '\n'.join(result_lines) + '\n'

    def _unparse_with_comments(self, node: ast.AST, base_unparsed: str) -> str:
        """Unparse with comments inserted at appropriate positions."""
        # Split by lines for easier processing
        lines = base_unparsed.split('\n')
        result_lines = []

        # Create a mapping from node to its unparsed representation
        node_to_unparsed: Dict[ast.AST, str] = {}

        def _build_node_map(n: ast.AST):
            """Build a map from nodes to their unparsed forms."""
            if n is None:
                return

            # Try to unparse this node
            try:
                node_unparsed = ast.unparse(n)
                # Store the first line for matching
                first_line = node_unparsed.split('\n')[0].strip()
                node_to_unparsed[n] = first_line
            except Exception:
                pass

            # Recursively process children
            for field, value in ast.iter_fields(n):
                if isinstance(value, ast.AST):
                    _build_node_map(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            _build_node_map(item)

        _build_node_map(node)

        # Now process each line and add comments
        for line in lines:
            result_lines.append(line)

            # Check if this line matches any node with attributes
            line_stripped = line.strip()
            for node_with_attrs, first_line in node_to_unparsed.items():
                if node_with_attrs not in self.nodes_with_attrs:
                    continue

                # Check if this line contains the node's first line
                if first_line and first_line in line_stripped:
                    comment = self._get_attr_comment(node_with_attrs)
                    if comment:
                        # Add comment to the end of the line
                        result_lines[-1] = line.rstrip() + comment
                        # Remove from nodes_with_attrs so we don't process it again
                        del self.nodes_with_attrs[node_with_attrs]
                        break

        return '\n'.join(result_lines)


def enhanced_unparse(tree: ast.AST, *, show_custom_attrs: bool = True, show_ir_nodes: bool = True, indent: int = 4,
                     ctx: Optional[Dict[str, Any]] = None) -> str:
    """
    Enhanced version of ast.unparse that supports custom nodes and attributes.
    
    Args:
        tree: The AST to unparse
        show_custom_attrs: Whether to show custom attributes as comments
        show_ir_nodes: Whether to show IR nodes (EnumDef, SpecialStructDef)
        indent: Number of spaces per indentation level (not used currently)
        ctx: Optional context dictionary for resolving function objects (for intrin detection)
    
    Returns:
        Unparsed code as a string with custom nodes and attributes shown
    """
    unparser = EnhancedUnparser(show_custom_attrs=show_custom_attrs, show_ir_nodes=show_ir_nodes, ctx=ctx)
    return unparser.unparse(tree)
