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
Enhanced AST dumper that supports custom AST nodes and attributes.

This module provides an enhanced version of ast.dump that can:
1. Handle custom AST node types (e.g., EnumDef, SpecialStructDef)
2. Display custom attributes added to AST nodes
3. Maintain similar format to standard ast.dump but with additional information
"""

import ast
from typing import Any, Dict, Optional
from .preserve_attributes import get_special_attrs
from .ir_nodes import is_ir_node, get_ir_node_type, IRNodeType
from .resolve_attribute import recursive_resolve_attribute
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    CONST_FUNC_ATTR,
    INLINE_FUNC_ATTR,
    CODEGEN_FUNC_ATTR,
)


def _get_intrin_info(node: ast.Call, ctx: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """
    Try to detect if a Call node is calling an intrin/builtin function and return its info.
    
    Args:
        node: The Call AST node
        ctx: Optional context dictionary for resolving function objects
    
    Returns:
        Dictionary with intrin attributes if detected, None otherwise
    """
    if not isinstance(node, ast.Call):
        return None

    # Try to resolve the function object if context is available
    func_obj = None
    if ctx is not None:
        try:
            if isinstance(node.func, (ast.Name, ast.Attribute)):
                func_obj = recursive_resolve_attribute(node.func, ctx)
        except Exception:
            pass

    # Check if it's a builtin/intrin function
    if func_obj is not None and callable(func_obj):
        intrin_attrs = {}

        # Check for builtin attributes (using constants from builtin_base)
        if hasattr(func_obj, BUILTIN_ATTR) and getattr(func_obj, BUILTIN_ATTR, False):
            intrin_attrs["_is_builtin"] = True
            intrin_attrs["_builtin_name"] = getattr(func_obj, "__name__", "?")

        if hasattr(func_obj, CODEGEN_FUNC_ATTR):
            intrin_attrs["_has_codegen"] = True

        if hasattr(func_obj, EVAL_RETURN_TYPE_ATTR):
            intrin_attrs["_has_eval_return_type"] = True

        if hasattr(func_obj, INLINE_FUNC_ATTR) and getattr(func_obj, INLINE_FUNC_ATTR, False):
            intrin_attrs["_is_inline"] = True

        if hasattr(func_obj, CONST_FUNC_ATTR) and getattr(func_obj, CONST_FUNC_ATTR, False):
            intrin_attrs["_is_const"] = True

        # Check if it's from intrin module
        module = getattr(func_obj, "__module__", "")
        if "intrin" in module:
            intrin_attrs["_intrin_module"] = module

        if intrin_attrs:
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
        return {"_is_intrin_like": True, "_intrin_name": func_name}

    return None


def _format_value(value: Any, max_depth: int = 3, max_length: int = 100) -> str:
    """Format a value for display in dump."""
    if max_depth <= 0:
        return "..."

    if isinstance(value, ast.AST):
        # For AST nodes, return a placeholder (will be handled by visitor)
        return f"<{type(value).__name__}>"
    elif isinstance(value, dict):
        if not value:
            return "{}"
        items = []
        for k, v in list(value.items())[:3]:  # Limit to 3 items
            key_str = repr(k) if isinstance(k, str) else str(k)
            val_str = _format_value(v, max_depth - 1, max_length)
            items.append(f"{key_str}: {val_str}")
        if len(value) > 3:
            items.append(f"... ({len(value)} more)")
        return "{" + ", ".join(items) + "}"
    elif isinstance(value, (list, tuple)):
        if not value:
            return f"{type(value).__name__}()"
        items = []
        for item in value[:3]:  # Limit to 3 items
            items.append(_format_value(item, max_depth - 1, max_length))
        if len(value) > 3:
            items.append(f"... ({len(value)} more)")
        bracket = "[" if isinstance(value, list) else "("
        close_bracket = "]" if isinstance(value, list) else ")"
        return bracket + ", ".join(items) + close_bracket
    elif isinstance(value, str):
        if len(value) > max_length:
            return repr(value[:max_length] + "...")
        return repr(value)
    elif isinstance(value, type):
        return f"<class '{value.__name__}'>"
    elif hasattr(value, '__name__'):
        return f"<{type(value).__name__}: {value.__name__}>"
    else:
        val_str = repr(value)
        if len(val_str) > max_length:
            val_str = val_str[:max_length] + "..."
        return val_str


def enhanced_dump(node: ast.AST, *, indent: int = 2, include_attributes: bool = True, show_custom_attrs: bool = True,
                  ctx: Optional[Dict[str, Any]] = None) -> str:
    """
    Enhanced version of ast.dump that supports custom nodes and attributes.
    
    This function uses the standard ast.dump as a base, then enhances it by:
    1. Adding custom attributes to the output
    2. Handling custom IR nodes (EnumDef, SpecialStructDef)
    
    Args:
        node: The AST node to dump
        indent: Number of spaces per indentation level
        include_attributes: Whether to include standard AST attributes (lineno, col_offset, etc.)
        show_custom_attrs: Whether to show custom attributes
        ctx: Optional context dictionary for resolving function objects (for intrin detection)
    
    Returns:
        Dumped AST as a string with custom nodes and attributes shown
    """
    # First, get the standard dump
    try:
        standard_dump = ast.dump(node, indent=indent, include_attributes=include_attributes)
    except Exception:
        # If standard dump fails (e.g., for custom nodes), build our own
        standard_dump = None

    # If we have custom nodes or attributes, we need to enhance the dump
    if standard_dump is None or show_custom_attrs or is_ir_node(node):
        return _enhanced_dump_recursive(node, indent, include_attributes, show_custom_attrs, 0, ctx)

    # Otherwise, just add custom attributes to the standard dump
    if show_custom_attrs:
        custom_attrs = get_special_attrs(node)
        if custom_attrs:
            # Try to append custom attributes to the dump
            # This is a simple approach - for complex cases, use recursive version
            attr_strs = []
            for attr_name, attr_value in custom_attrs.items():
                val_str = _format_value(attr_value)
                attr_strs.append(f"_{attr_name}={val_str}")
            if attr_strs:
                # Append to the last line
                lines = standard_dump.split('\n')
                if lines:
                    lines[-1] = lines[-1].rstrip() + ", " + ", ".join(attr_strs)
                standard_dump = '\n'.join(lines)

    return standard_dump


def _enhanced_dump_recursive(node: ast.AST, indent: int, include_attributes: bool, show_custom_attrs: bool,
                             current_indent: int, ctx: Optional[Dict[str, Any]] = None) -> str:
    """Recursively dump a node with custom attributes."""
    if node is None:
        return "None"

    indent_str = " " * (current_indent * indent)
    next_indent = current_indent + 1
    next_indent_str = " " * (next_indent * indent)

    # Handle custom IR nodes
    if is_ir_node(node):
        ir_type = get_ir_node_type(node)
        if ir_type == IRNodeType.ENUM_DEF:
            enum_node = node
            fields = [f"enum_name={repr(enum_node.enum_name)}", f"enum_class=<class '{enum_node.enum_class.__name__}'>"]

            # Get enum members
            try:
                members = list(enum_node.enum_class.__members__.keys())
                members_str = ", ".join(members[:5])
                if len(members) > 5:
                    members_str += f", ... ({len(members)} total)"
                fields.append(f"members=[{members_str}]")
            except Exception:
                pass

            if enum_node.ast_node is not None:
                fields.append(f"ast_node=<{type(enum_node.ast_node).__name__}>")

            # Add custom attributes
            if show_custom_attrs:
                custom_attrs = get_special_attrs(enum_node)
                for attr_name, attr_value in custom_attrs.items():
                    val_str = _format_value(attr_value)
                    fields.append(f"_{attr_name}={val_str}")

            return f"EnumDef({', '.join(fields)})"

        elif ir_type == IRNodeType.SPECIAL_STRUCT_DEF:
            struct_node = node
            fields = [f"struct_name={repr(struct_node.struct_name)}"]

            # Format struct_info
            struct_info = struct_node.struct_info
            if struct_info:
                info_items = []
                for k, v in list(struct_info.items())[:5]:
                    if k == 'class':
                        info_items.append(f"class=<class '{v.__name__}'>")
                    else:
                        val_str = _format_value(v)
                        info_items.append(f"{k}={val_str}")
                if len(struct_info) > 5:
                    info_items.append(f"... ({len(struct_info)} more)")
                fields.append(f"struct_info={{{', '.join(info_items)}}}")

            if struct_node.ast_node is not None:
                fields.append(f"ast_node=<{type(struct_node.ast_node).__name__}>")

            # Add custom attributes
            if show_custom_attrs:
                custom_attrs = get_special_attrs(struct_node)
                for attr_name, attr_value in custom_attrs.items():
                    val_str = _format_value(attr_value)
                    fields.append(f"_{attr_name}={val_str}")

            return f"SpecialStructDef({', '.join(fields)})"

    # For standard nodes, use ast.dump but enhance with custom attributes
    node_name = node.__class__.__name__

    # Get all fields
    field_values = []
    for field in node._fields:
        value = getattr(node, field, None)
        if value is not None:
            if isinstance(value, ast.AST):
                # Recursively dump child node
                child_dump = _enhanced_dump_recursive(value, indent, include_attributes, show_custom_attrs, next_indent,
                                                      ctx)
                field_values.append((field, child_dump, True))
            elif isinstance(value, list):
                # Handle list of nodes
                list_items = []
                for item in value:
                    if isinstance(item, ast.AST):
                        item_dump = _enhanced_dump_recursive(item, indent, include_attributes, show_custom_attrs,
                                                             next_indent, ctx)
                        list_items.append(item_dump)
                    else:
                        list_items.append(_format_value(item))
                field_values.append((field, list_items, True))
            else:
                # Simple value
                field_values.append((field, _format_value(value), False))

    # Add standard attributes if requested
    if include_attributes:
        if hasattr(node, 'lineno') and node.lineno is not None:
            field_values.append(('lineno', str(node.lineno), False))
        if hasattr(node, 'col_offset') and node.col_offset is not None:
            field_values.append(('col_offset', str(node.col_offset), False))
        if hasattr(node, 'end_lineno') and node.end_lineno is not None:
            field_values.append(('end_lineno', str(node.end_lineno), False))
        if hasattr(node, 'end_col_offset') and node.end_col_offset is not None:
            field_values.append(('end_col_offset', str(node.end_col_offset), False))

    # Add custom attributes
    if show_custom_attrs:
        custom_attrs = get_special_attrs(node)
        for attr_name, attr_value in custom_attrs.items():
            val_str = _format_value(attr_value)
            field_values.append((f"_{attr_name}", val_str, False))

        # For Call nodes, try to detect intrin/builtin functions
        if isinstance(node, ast.Call):
            intrin_info = _get_intrin_info(node, ctx)
            if intrin_info:
                for attr_name, attr_value in intrin_info.items():
                    val_str = _format_value(attr_value)
                    field_values.append((f"_{attr_name}", val_str, False))

    # Format output
    if not field_values:
        return f"{node_name}()"

    # Check if we should use single line or multi-line
    total_length = len(node_name) + sum(len(f"{name}={val}") for name, val, is_complex in field_values)
    use_multiline = total_length > 80 or any(is_complex for _, _, is_complex in field_values)

    if use_multiline:
        lines = [f"{node_name}("]
        for i, (name, val, is_complex) in enumerate(field_values):
            if isinstance(val, list):
                # List of items
                lines.append(f"{next_indent_str}{name}=[")
                list_indent_str = " " * ((next_indent + 1) * indent)
                for j, item in enumerate(val):
                    if j < len(val) - 1:
                        lines.append(f"{list_indent_str}{item},")
                    else:
                        lines.append(f"{list_indent_str}{item}")
                lines.append(f"{next_indent_str}],")
            elif is_complex:
                # Complex value (AST node)
                val_lines = val.split('\n')
                lines.append(f"{next_indent_str}{name}=")
                for val_line in val_lines:
                    lines.append(f"{next_indent_str}{val_line}")
                if i < len(field_values) - 1:
                    lines[-1] += ","
            else:
                # Simple value
                if i < len(field_values) - 1:
                    lines.append(f"{next_indent_str}{name}={val},")
                else:
                    lines.append(f"{next_indent_str}{name}={val}")
        lines.append(f"{indent_str})")
        return '\n'.join(lines)
    else:
        # Single line
        field_strs = [f"{name}={val}" for name, val, _ in field_values]
        return f"{node_name}({', '.join(field_strs)})"
