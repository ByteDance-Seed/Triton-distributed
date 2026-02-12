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
Utility functions for special struct codegen.

This module provides utility functions for resolving template parameters
and generating C++ struct definitions from Python classes.
"""

import ast
from typing import Optional
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from .struct_converter import PythonClassToCppStructConverter


def _resolve_template_param(node: ast.AST, ctx: dict) -> str:
    """
    Generic function to resolve a template parameter from AST node to C++ constant expression.
    
    This is a generic version that works for any special struct, not just Scheduler.
    """
    # Try to resolve as constant
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    elif isinstance(node, ast.Name):
        # Try to resolve from context
        if node.id in ctx:
            value = ctx[node.id]
            if isinstance(value, (int, float, bool)):
                if isinstance(value, bool):
                    return "true" if value else "false"
                return str(value)
            elif hasattr(value, '__name__'):
                # Handle enum values
                return value.__name__
        # Otherwise, return the variable name (will be resolved at compile time)
        return node.id
    elif isinstance(node, ast.Attribute):
        # Handle enum values like GemmType.Normal
        try:
            resolved = recursive_resolve_attribute(node.value, ctx)
            if hasattr(resolved, node.attr):
                enum_value = getattr(resolved, node.attr)
                from enum import Enum
                if isinstance(enum_value, Enum):
                    enum_class = type(enum_value)
                    enum_name = enum_class.__name__
                    # Track this enum if emitter is available in ctx
                    if 'emitter' in ctx and hasattr(ctx['emitter'], 'used_enums'):
                        ctx['emitter'].used_enums.add(enum_name)
                    return f"{enum_name}::{node.attr}"
        except Exception:
            pass
        # Fallback: generate attribute access
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}::{node.attr}"
        return f"{_resolve_template_param(node.value, ctx)}::{node.attr}"
    else:
        # For complex expressions, we'll need to visit them
        return str(node)


def generate_cpp_struct_from_python_class(class_name: str, module_path: str,
                                          template_params: Optional[list] = None) -> str:
    """
    Generate C++ struct definition from Python class.
    
    Args:
        class_name: Name of the Python class to convert
        module_path: Path to the Python module containing the class
        template_params: Optional list of template parameters for the C++ struct
    
    Returns:
        C++ struct definition as string
    """
    import importlib.util

    # Load the module
    spec = importlib.util.spec_from_file_location("module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Parse the source file
    with open(module_path, 'r') as f:
        source = f.read()
    module_ast = ast.parse(source)

    # Find the class
    converter = PythonClassToCppStructConverter(class_name, template_params)
    converter.visit(module_ast)

    return converter.code.getvalue()
