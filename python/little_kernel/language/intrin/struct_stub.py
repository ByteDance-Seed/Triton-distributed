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
Struct stub support for C++ structs with existing implementations.

This module provides a mechanism to create Python stubs for C++ structs
that already have implementations. The stub allows:
- Struct construction: MyStruct(arg1, arg2) -> C++ MyStruct(arg1, arg2)
- Method calls: my_struct.method(arg1, arg2) -> C++ my_struct.method(arg1, arg2)
- Automatic include generation
- Method-level configuration (name mapping, custom codegen, type annotations)
- Support for @staticmethod and @classmethod
"""

from typing import List, Optional, Union, Callable, Dict, Any
from dataclasses import dataclass, field
from little_kernel.core.type_system import LLType
import ast


@dataclass
class MethodStubInfo:
    """Information about a method in a struct stub."""
    python_method_name: str  # Python method name
    cpp_method_name: Optional[str] = None  # C++ method name (if different from Python)
    is_static: bool = False  # Is @staticmethod
    is_classmethod: bool = False  # Is @classmethod
    custom_codegen: Optional[Callable[[ast.Call, Any], str]] = None  # Custom codegen function
    return_type: Optional[LLType] = None  # Return type annotation
    is_tuple_return: bool = False  # Returns tuple
    tuple_return_types: Optional[List[LLType]] = None  # Types for tuple return
    param_types: Optional[List[Optional[LLType]]] = None  # Parameter type annotations


@dataclass
class StructStubInfo:
    """Information about a C++ struct stub."""
    cpp_struct_name: str  # C++ struct name (e.g., "MyStruct")
    includes: List[str]  # Include files (e.g., ['"my_struct.h"', '<cute/struct.hpp>'])
    template_params: Optional[List[str]] = None  # Template parameters if any
    namespace: Optional[str] = None  # Namespace (e.g., "cute")
    methods: Dict[str, MethodStubInfo] = field(default_factory=dict)  # Method configurations


# Registry: Python class name -> StructStubInfo
_struct_stub_registry: Dict[str, StructStubInfo] = {}


def struct_method(cpp_method_name: Optional[str] = None, is_static: bool = False, is_classmethod: bool = False,
                  custom_codegen: Optional[Callable[[ast.Call, Any],
                                                    str]] = None, return_type: Optional[Union[LLType, str]] = None,
                  is_tuple_return: bool = False, tuple_return_types: Optional[List[Union[LLType, str]]] = None,
                  param_types: Optional[List[Optional[Union[LLType, str]]]] = None):
    """
    Decorator to configure a method in a struct stub.
    
    Usage:
        @struct_method(
            cpp_method_name="cpp_method",  # If different from Python name
            is_static=True,  # For @staticmethod
            return_type=ll.bool_,
            param_types=[ll.int32, ll.int32]
        )
        def my_method(self, arg1, arg2):
            pass
    
    Args:
        cpp_method_name: C++ method name (if different from Python name)
        is_static: Whether this is a @staticmethod
        is_classmethod: Whether this is a @classmethod
        custom_codegen: Custom codegen function (signature: func(node: ast.Call, emitter) -> str)
        return_type: Return type annotation (LLType or string)
        is_tuple_return: Whether the method returns a tuple
        tuple_return_types: Types for tuple return elements
        param_types: Parameter type annotations (list of LLType or string, None for self/cls)
    """

    def decorator(func: Callable) -> Callable:
        # Store method configuration in function attributes
        func.__struct_method_stub__ = True
        func.__struct_method_info__ = MethodStubInfo(
            python_method_name=func.__name__, cpp_method_name=cpp_method_name, is_static=is_static,
            is_classmethod=is_classmethod, custom_codegen=custom_codegen,
            return_type=return_type if isinstance(return_type, LLType) else None, is_tuple_return=is_tuple_return,
            tuple_return_types=[t if isinstance(t, LLType) else None
                                for t in tuple_return_types] if tuple_return_types else None,
            param_types=[t if isinstance(t, LLType) else None for t in param_types] if param_types else None)
        return func

    return decorator


def struct_stub(cpp_struct_name: str, includes: Union[str, List[str]], template_params: Optional[List[str]] = None,
                namespace: Optional[str] = None, python_class_name: Optional[str] = None):
    """
    Decorator to register a Python class as a stub for an existing C++ struct.
    
    Usage:
        @struct_stub(
            cpp_struct_name="MyStruct",
            includes=['"my_struct.h"'],
            template_params=["T", "int N"]  # Optional
        )
        class MyStruct:
            def __init__(self, arg1, arg2):
                pass  # Stub, actual implementation is in C++
            
            @struct_method(
                cpp_method_name="cpp_method",
                return_type=ll.bool_,
                param_types=[ll.int32, ll.int32]
            )
            def method(self, arg1, arg2):
                pass  # Stub, actual implementation is in C++
    
    Args:
        cpp_struct_name: Name of the C++ struct (e.g., "MyStruct")
        includes: Include file(s) as string or list of strings
                  (e.g., '"my_struct.h"' or ['"my_struct.h"', '<cute/struct.hpp>'])
        template_params: Optional list of template parameter names/types
                         (e.g., ["T", "int N"] for "template <typename T, int N>")
        namespace: Optional namespace prefix (e.g., "cute" for "cute::MyStruct")
        python_class_name: Optional Python class name (defaults to class.__name__)
    """
    # Normalize includes to list
    if isinstance(includes, str):
        includes_list = [includes]
    else:
        includes_list = includes

    def decorator(cls: type) -> type:
        name = python_class_name if python_class_name is not None else cls.__name__

        # Collect method information from class
        methods = {}
        import inspect

        # First, collect methods with @struct_method decorator
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if callable(attr) and hasattr(attr, "__struct_method_stub__"):
                method_info = attr.__struct_method_info__
                methods[method_info.python_method_name] = method_info

        # Then, auto-detect @staticmethod and @classmethod
        for attr_name in dir(cls):
            if attr_name.startswith("__") or attr_name in methods:
                continue
            attr = getattr(cls, attr_name)
            if callable(attr):
                is_static = isinstance(attr, staticmethod)
                is_classmethod = isinstance(attr, classmethod)
                if is_static or is_classmethod:
                    # Get the underlying function
                    if is_static:
                        func = attr
                    else:
                        func = attr.__func__

                    # Try to get type annotations from function
                    # Store raw annotations, will be resolved during codegen
                    sig = inspect.signature(func)
                    param_types = []
                    return_type = None

                    # Parse parameter types (store as raw annotations)
                    for param_name, param in sig.parameters.items():
                        if param_name in ['self', 'cls']:
                            param_types.append(None)
                        elif param.annotation != inspect.Parameter.empty:
                            # Store raw annotation, will be resolved during codegen
                            param_types.append(param.annotation)
                        else:
                            param_types.append(None)

                    # Parse return type (store as raw annotation)
                    if sig.return_annotation != inspect.Signature.empty:
                        return_type = sig.return_annotation

                    method_info = MethodStubInfo(
                        python_method_name=attr_name, cpp_method_name=None, is_static=is_static,
                        is_classmethod=is_classmethod,
                        return_type=return_type if isinstance(return_type, LLType) else None,
                        param_types=[t if isinstance(t, LLType) else None for t in param_types])
                    methods[attr_name] = method_info

        # Register the struct stub
        stub_info = StructStubInfo(cpp_struct_name=cpp_struct_name, includes=includes_list,
                                   template_params=template_params, namespace=namespace, methods=methods)
        _struct_stub_registry[name] = stub_info

        # Mark the class as a struct stub
        cls.__struct_stub__ = True
        cls.__struct_stub_info__ = stub_info

        return cls

    return decorator


def get_struct_stub_info(python_class_name: str) -> Optional[StructStubInfo]:
    """Get struct stub information for a Python class name."""
    return _struct_stub_registry.get(python_class_name)


def get_all_struct_stubs() -> Dict[str, StructStubInfo]:
    """Get all registered struct stubs."""
    return _struct_stub_registry.copy()


def is_struct_stub(obj: Any) -> bool:
    """Check if an object is a struct stub."""
    return hasattr(obj, "__struct_stub__") and obj.__struct_stub__
