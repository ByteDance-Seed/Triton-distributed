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
import inspect
import textwrap
from typing import Dict, Any, Optional
from little_kernel.core.type_system import LLType, TupleType


class MethodResolver:
    """Centralized method return type resolver for type inference."""

    def __init__(self, ctx: Dict[str, Any], current_scope: Dict[str, LLType], resolve_annotation_func: callable):
        self.ctx = ctx
        self.current_scope = current_scope
        self._resolve_annotation = resolve_annotation_func

    def _get_method_return_type_from_class(self, class_obj: type, method_name: str) -> Optional[LLType]:
        """Get method return type from class definition by parsing its AST."""
        try:
            source = inspect.getsource(class_obj)
            source = textwrap.dedent(source)
            module_ast = ast.parse(source)

            for node in ast.walk(module_ast):
                if isinstance(node, ast.ClassDef) and node.name == class_obj.__name__:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            if item.returns is not None:
                                return self._resolve_annotation(item.returns)
                    break
        except Exception:
            pass
        return None

    def _try_resolve_from_struct_stub(self, struct_name: str, method_name: str) -> Optional[LLType]:
        """Try to resolve method return type from struct stub registry."""
        try:
            from little_kernel.language.intrin.struct_stub import get_struct_stub_info
            stub_info = get_struct_stub_info(struct_name)
            if stub_info and method_name in stub_info.methods:
                method_info = stub_info.methods[method_name]
                if method_info.is_tuple_return and method_info.tuple_return_types:
                    return TupleType(tuple(method_info.tuple_return_types))
                elif method_info.return_type is not None:
                    return method_info.return_type
        except (ImportError, AttributeError):
            pass
        return None

    def _try_resolve_from_special_struct(self, struct_name: str, method_name: str) -> Optional[LLType]:
        """Try to resolve method return type from special struct registry."""
        try:
            from little_kernel.codegen.special_struct_registry import get_special_struct_codegen
            struct_info = get_special_struct_codegen(struct_name)
            if struct_info and 'class' in struct_info:
                class_obj = struct_info['class']
                if class_obj is not None and inspect.isclass(class_obj):
                    return self._get_method_return_type_from_class(class_obj, method_name)
        except ImportError:
            pass
        return None

    def _try_resolve_from_ctx(self, method_name: str) -> Optional[LLType]:
        """Try to find class in ctx that has this method."""
        for key, value in self.ctx.items():
            if inspect.isclass(value) and hasattr(value, method_name):
                return_type = self._get_method_return_type_from_class(value, method_name)
                if return_type is not None:
                    return return_type
        return None

    def resolve_method_return_type(self, var_name: str, method_name: str,
                                   var_type: Optional[LLType] = None) -> Optional[LLType]:
        """
        Resolve method return type using multiple strategies.
        
        Args:
            var_name: Name of the variable (e.g., 'scheduler')
            method_name: Name of the method (e.g., 'get_next_block')
            var_type: Optional type of the variable (e.g., SpecialStructType)
        
        Returns:
            LLType if method return type can be inferred, None otherwise
        """
        # Strategy 1: If var_type is SpecialStructType, try struct stub and special struct registries
        if var_type is not None:
            from little_kernel.core.type_system import SpecialStructType
            if isinstance(var_type, SpecialStructType):
                struct_name = var_type.struct_name
                # Try struct stub first
                result = self._try_resolve_from_struct_stub(struct_name, method_name)
                if result is not None:
                    return result
                # Try special struct registry
                result = self._try_resolve_from_special_struct(struct_name, method_name)
                if result is not None:
                    return result

        # Strategy 2: Check if var_name is a struct stub class name
        result = self._try_resolve_from_struct_stub(var_name, method_name)
        if result is not None:
            return result

        # Strategy 3: Try to find class in ctx
        result = self._try_resolve_from_ctx(method_name)
        if result is not None:
            return result

        # Strategy 4: Try special struct registry by searching all registered structs
        try:
            from little_kernel.codegen.special_struct_registry import get_all_registered_special_structs
            registered_structs = get_all_registered_special_structs()
            for struct_name, struct_info in registered_structs.items():
                class_obj = struct_info.get('class')
                if class_obj is not None and inspect.isclass(class_obj):
                    if hasattr(class_obj, method_name):
                        return_type = self._get_method_return_type_from_class(class_obj, method_name)
                        if return_type is not None:
                            return return_type
        except ImportError:
            pass

        return None
