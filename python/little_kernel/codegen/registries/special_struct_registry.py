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
Registry for special struct code generators.

This module provides a generic mechanism for registering code generators
for special struct types (like Scheduler) that need custom C++ code generation.
"""

from typing import Dict, Callable, Any, Optional
import ast

# Global registry for special struct code generators
# Maps struct_name -> {'declaration': func, 'method': func, 'tuple_unpack': func, 'class': class_obj}
_SPECIAL_STRUCT_CODEGEN_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register_special_struct(struct_name=None, declaration_codegen=None, method_codegen=None, tuple_unpack_codegen=None,
                            struct_definition_codegen=None):
    """
    Decorator to register a special struct class for code generation.
    
    Usage:
        @register_special_struct()
        class Scheduler:
            ...
        
        # Or with custom name and codegen functions:
        @register_special_struct(
            struct_name="MyScheduler",
            declaration_codegen=custom_declaration_codegen,
            method_codegen=custom_method_codegen,
            tuple_unpack_codegen=custom_tuple_unpack_codegen
        )
        class Scheduler:
            ...
    
    Args:
        struct_name: Optional name for the struct. If not provided, uses the class name.
        declaration_codegen: Optional function that generates C++ code for struct declaration.
                            If not provided, uses generic_special_struct_declaration_codegen.
        method_codegen: Optional function that generates C++ code for method calls.
                       If not provided, uses generic_special_struct_method_codegen.
        tuple_unpack_codegen: Optional function that generates C++ code for tuple unpacking.
                             If not provided, uses generic_special_struct_tuple_unpack_codegen.
        struct_definition_codegen: Optional function that generates C++ struct definition.
                                   If not provided, uses generic_special_struct_definition_codegen.
    """
    # Support both @register_special_struct and @register_special_struct(...)
    if struct_name is not None and isinstance(struct_name, type):
        # Called as @register_special_struct without parentheses
        cls = struct_name
        struct_name = None
    else:
        cls = None

    def decorator(cls: type) -> type:
        from little_kernel.codegen.special_struct_codegen import (generic_special_struct_declaration_codegen,
                                                                  generic_special_struct_method_codegen,
                                                                  generic_special_struct_tuple_unpack_codegen,
                                                                  generic_special_struct_definition_codegen)

        name = struct_name if struct_name is not None else cls.__name__

        # Use provided codegen functions or default to generic ones
        decl_codegen = declaration_codegen or generic_special_struct_declaration_codegen
        meth_codegen = method_codegen or generic_special_struct_method_codegen
        tuple_codegen = tuple_unpack_codegen or generic_special_struct_tuple_unpack_codegen

        # For struct definition, wrap it to pass struct_name
        if struct_definition_codegen:
            def_codegen = struct_definition_codegen
        else:
            def_codegen = lambda emitter: generic_special_struct_definition_codegen(name, emitter)

        _SPECIAL_STRUCT_CODEGEN_REGISTRY[name] = {
            'declaration': decl_codegen, 'method': meth_codegen, 'tuple_unpack': tuple_codegen, 'struct_definition':
            def_codegen, 'class': cls
        }

        return cls

    if cls is not None:
        # Called as @register_special_struct without parentheses
        return decorator(cls)
    else:
        # Called as @register_special_struct() or @register_special_struct(...)
        return decorator


def register_special_struct_codegen(struct_name: str, declaration_codegen: Callable[[ast.Assign, Any], str],
                                    method_codegen: Callable[[ast.Call, Any], str],
                                    tuple_unpack_codegen: Callable[[ast.Tuple, ast.Call, Any], None],
                                    struct_definition_codegen: Optional[Callable[[Any], str]] = None,
                                    class_obj: Optional[type] = None):
    """
    Register code generators for a special struct type.
    
    Args:
        struct_name: Name of the special struct (e.g., 'Scheduler')
        declaration_codegen: Function that generates C++ code for struct declaration.
                            Signature: func(node: ast.Assign, emitter) -> str
        method_codegen: Function that generates C++ code for method calls.
                       Signature: func(node: ast.Call, emitter) -> str
        tuple_unpack_codegen: Function that generates C++ code for tuple unpacking.
                             Signature: func(target: ast.Tuple, value_node: ast.Call, emitter) -> None
        struct_definition_codegen: Optional function that generates C++ struct definition.
                                   Signature: func(emitter) -> str
                                   This will be called once to generate the struct definition
                                   and written to struct_buffer.
        class_obj: Optional class object for this special struct (e.g., Scheduler class).
                   Used for type inference to resolve method return types.
    """
    _SPECIAL_STRUCT_CODEGEN_REGISTRY[struct_name] = {
        'declaration': declaration_codegen, 'method': method_codegen, 'tuple_unpack': tuple_unpack_codegen,
        'struct_definition': struct_definition_codegen, 'class': class_obj
    }


def get_special_struct_codegen(struct_name: str) -> Optional[Dict[str, Any]]:
    """Get the code generators for a special struct type."""
    return _SPECIAL_STRUCT_CODEGEN_REGISTRY.get(struct_name)


def get_all_registered_special_structs() -> Dict[str, Dict[str, Any]]:
    """Get all registered special structs."""
    return _SPECIAL_STRUCT_CODEGEN_REGISTRY.copy()


def is_special_struct(node: ast.AST) -> bool:
    """Check if an AST node represents a special struct declaration."""
    # Check if node has the special struct marker
    if hasattr(node, '_is_special_struct') and node._is_special_struct:
        return True
    return False


def get_special_struct_name(node: ast.AST) -> Optional[str]:
    """Get the name of the special struct from an AST node."""
    if hasattr(node, '_special_struct_name'):
        return node._special_struct_name
    return None
