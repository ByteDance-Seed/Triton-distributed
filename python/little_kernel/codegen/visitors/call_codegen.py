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
Call codegen visitor for CppEmitter.

This module handles function call code generation, including:
- Builtin function calls
- Struct stub method calls
- Special struct method calls
- LLKernel recursive calls
"""

import ast
from typing import TYPE_CHECKING, Callable
from little_kernel.language.builtin_base import Builtin
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.compile import LLKernel
# PASSES imported lazily to avoid circular import (see _handle_llkernel_call)
from little_kernel.core.type_system import LLType, AnnotateTypeHelper, SpecialStructType
from little_kernel.language.intrin.struct_stub import (get_struct_stub_info, get_all_struct_stubs)
from little_kernel.language.intrin.dtype import to
from ..registries.special_struct_registry import get_special_struct_codegen
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CODEGEN_FUNC_ATTR,
)

if TYPE_CHECKING:
    pass


class CallCodegenMixin:
    """Mixin class for call codegen methods."""

    def visit_Call(self, node: ast.Call) -> str:
        """Process function calls."""
        # Check if this is a special struct method call FIRST (before struct stub checks)
        # This ensures special struct methods are handled correctly
        if hasattr(node, '_is_special_struct_method') and node._is_special_struct_method:
            return self._handle_special_struct_method_call(node)

        # First try to resolve the function
        func_resolved = None
        if isinstance(node.func, ast.Attribute) or isinstance(node.func, ast.Name):
            try:
                func_resolved = recursive_resolve_attribute(node.func, self.ctx)
            except Exception:
                pass

        func = self.visit(node.func)

        # Check if this is a "to" method call (val.to(ll.uint32))
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            # Convert val.to(ll.uint32) to to(val, ll.uint32)
            if len(node.args) == 1:
                val_cpp = self.visit(node.func.value)
                dtype_arg = node.args[0]
                # Try to resolve dtype to get LLType
                try:
                    dtype_resolved = recursive_resolve_attribute(dtype_arg, self.ctx)
                    if isinstance(dtype_resolved, LLType):
                        # Use to() builtin function
                        if hasattr(to, CODEGEN_FUNC_ATTR):
                            codegen_func = getattr(to, CODEGEN_FUNC_ATTR)
                            result = codegen_func(val_cpp, dtype_resolved)
                            if isinstance(result, Builtin):
                                return result.return_val
                except Exception:
                    pass
                # Fallback: generate static_cast
                dtype_cpp = self.visit(dtype_arg)
                try:
                    dtype_resolved = recursive_resolve_attribute(dtype_arg, self.ctx)
                    if isinstance(dtype_resolved, LLType):
                        # Use emitter's _lltype_to_cpp method if available
                        if hasattr(self, '_lltype_to_cpp'):
                            dtype_cpp = self._lltype_to_cpp(dtype_resolved)
                        else:
                            # Fallback: use CudaEmitter (import here to avoid circular import)
                            from little_kernel.codegen.codegen_cuda import CudaEmitter
                            emitter = CudaEmitter({}, {})
                            dtype_cpp = emitter._lltype_to_cpp(dtype_resolved)
                except Exception:
                    pass
                return f"static_cast<{dtype_cpp}>({val_cpp})"

        # Check if this is a struct stub method call
        if isinstance(node.func, ast.Attribute):
            result = self._handle_struct_stub_method_call(node)
            if result is not None:
                return result

        # Check if this is a struct stub constructor call
        if isinstance(node.func, ast.Name):
            result = self._handle_struct_stub_constructor_call(node)
            if result is not None:
                return result

        # Check if this is ll.const(condition) - should return the condition itself for if constexpr
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'll' and node.func.attr == 'const':
                # ll.const(condition) -> return the condition expression
                # This will be handled by visit_If to generate "if constexpr"
                if len(node.args) == 1:
                    return self.visit(node.args[0])
                else:
                    raise ValueError(f"ll.const() expects exactly 1 argument, got {len(node.args)}")

        # Check if this is a type constructor call (ll.uint32(value), ll.int32(value), etc.)
        # But skip helper types like ConstTypeHelper, AnnotateTypeHelper, etc.
        if func_resolved is not None and isinstance(func_resolved, LLType):
            # Skip helper types - they are not actual type constructors
            if isinstance(func_resolved, AnnotateTypeHelper):
                # Helper types like ll.const, ll.ptr, etc. should not be used as type constructors
                # They are only used in annotations like "x: const[int32] = 5"
                pass
            elif len(node.args) == 1:
                # Type constructor call: ll.uint32(value) -> static_cast<uint32_t>(value)
                val_cpp = self.visit(node.args[0])
                # Use emitter's _lltype_to_cpp method if available
                if hasattr(self, '_lltype_to_cpp'):
                    dtype_cpp = self._lltype_to_cpp(func_resolved)
                else:
                    # Fallback: import from codegen_cuda
                    from little_kernel.codegen.codegen_cuda import CudaEmitter
                    emitter = CudaEmitter({}, {})
                    dtype_cpp = emitter._lltype_to_cpp(func_resolved)
                return f"static_cast<{dtype_cpp}>({val_cpp})"

        # Check if we resolved a builtin function
        if func_resolved is not None and isinstance(func_resolved, Callable):
            result = self._handle_builtin_call(node, func_resolved)
            if result is not None:
                return result

        # Check if func is a Constant (e.g., print)
        if isinstance(func, ast.Constant):
            result = self._handle_constant_func_call(node, func)
            if result is not None:
                return result

        raise NotImplementedError(f"Unsupported function call: {func}")

    def _handle_struct_stub_method_call(self, node: ast.Call) -> str:
        """Handle struct stub method calls."""
        if not isinstance(node.func, ast.Attribute):
            return None

        method_name = node.func.attr

        # Check if this is a staticmethod or classmethod call on a struct stub class
        if isinstance(node.func.value, ast.Name):
            class_name = node.func.value.id
            stub_info = get_struct_stub_info(class_name)
            if stub_info and method_name in stub_info.methods:
                method_info = stub_info.methods[method_name]
                if method_info.is_static or method_info.is_classmethod:
                    return self._generate_static_method_call(node, stub_info, method_info, class_name)

        # Check if this is an instance method call
        if isinstance(node.func.value, ast.Name):
            var_name = node.func.value.id
            if var_name in self.scope_vars:
                return self._generate_instance_method_call(node, var_name, method_name)

        return None

    def _generate_static_method_call(self, node: ast.Call, stub_info, method_info, class_name: str) -> str:
        """Generate code for static/class method call."""
        # Use custom codegen if available
        if method_info.custom_codegen:
            return method_info.custom_codegen(node, self)

        cpp_method_name = method_info.cpp_method_name or method_info.python_method_name
        cpp_struct_name = stub_info.cpp_struct_name
        if stub_info.namespace:
            cpp_struct_name = f"{stub_info.namespace}::{cpp_struct_name}"

        args_cpp = [self.visit(arg) for arg in node.args]
        kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords if kv.arg}

        # Add includes
        for include in stub_info.includes:
            if include not in self.header_cache:
                self.writeln_header(f"#include {include}")
                self.header_cache.add(include)

        if kwargs_cpp:
            all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
            return f"{cpp_struct_name}::{cpp_method_name}({', '.join(all_args)})"
        else:
            return f"{cpp_struct_name}::{cpp_method_name}({', '.join(args_cpp)})"

    def _generate_instance_method_call(self, node: ast.Call, var_name: str, method_name: str) -> str:
        """Generate code for instance method call."""
        # Get variable type from scope_vars or all_types (for type verification)
        var_type = self.scope_vars.get(var_name)
        if var_type is None and hasattr(node.func, 'value') and node.func.value in getattr(self, 'all_types', {}):
            var_type = self.all_types.get(node.func.value)

        # Try to find the struct stub class this variable belongs to
        # Only use a stub when var_type matches - prevents wrong codegen when multiple stubs
        # define methods with the same name
        for stub_class_name, stub_info in get_all_struct_stubs().items():
            if method_name not in stub_info.methods:
                continue

            # Verify variable type matches this struct stub
            if var_type is not None:
                if not (isinstance(var_type, SpecialStructType) and var_type.struct_name == stub_class_name):
                    continue

            method_info = stub_info.methods[method_name]

            # Use custom codegen if available
            if method_info.custom_codegen:
                return method_info.custom_codegen(node, self)

            # Generate method call with proper C++ method name
            cpp_method_name = method_info.cpp_method_name or method_name
            args_cpp = [self.visit(arg) for arg in node.args]
            kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords if kv.arg}

            # Add includes
            for include in stub_info.includes:
                if include not in self.header_cache:
                    self.writeln_header(f"#include {include}")
                    self.header_cache.add(include)

            if kwargs_cpp:
                all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
                return f"{var_name}.{cpp_method_name}({', '.join(all_args)})"
            else:
                return f"{var_name}.{cpp_method_name}({', '.join(args_cpp)})"

        # Fallback: generate method call without method info
        args_cpp = [self.visit(arg) for arg in node.args]
        kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords if kv.arg}

        if kwargs_cpp:
            all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
            return f"{var_name}.{method_name}({', '.join(all_args)})"
        else:
            return f"{var_name}.{method_name}({', '.join(args_cpp)})"

    def _handle_struct_stub_constructor_call(self, node: ast.Call) -> str:
        """Handle struct stub constructor calls."""
        if not isinstance(node.func, ast.Name):
            return None

        func_name = node.func.id
        stub_info = get_struct_stub_info(func_name)
        if not stub_info:
            return None

        # This is a struct stub constructor used in an expression
        args_cpp = [self.visit(arg) for arg in node.args]
        kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords if kv.arg}

        cpp_struct_name = stub_info.cpp_struct_name
        if stub_info.namespace:
            cpp_struct_name = f"{stub_info.namespace}::{cpp_struct_name}"

        # Add includes
        for include in stub_info.includes:
            if include not in self.header_cache:
                self.writeln_header(f"#include {include}")
                self.header_cache.add(include)

        if kwargs_cpp:
            all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
            return f"{cpp_struct_name}({', '.join(all_args)})"
        else:
            return f"{cpp_struct_name}({', '.join(args_cpp)})"

    def _handle_builtin_call(self, node: ast.Call, func_resolved: Callable) -> str:
        """Handle builtin function calls."""
        if hasattr(func_resolved, BUILTIN_ATTR) and getattr(func_resolved, BUILTIN_ATTR, False):
            return self._handle_builtin_with_codegen(node, func_resolved)
        elif isinstance(func_resolved, LLKernel):
            return self._handle_llkernel_call(node, func_resolved)

        return None

    def _handle_builtin_with_codegen(self, node: ast.Call, func_resolved: Callable) -> str:
        """Handle builtin function with codegen function."""
        assert hasattr(func_resolved, CODEGEN_FUNC_ATTR)
        args_cpp = [self.visit(arg) for arg in node.args]
        kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords}
        codegen_func = getattr(func_resolved, CODEGEN_FUNC_ATTR)
        builtin = codegen_func(*args_cpp, **kwargs_cpp)
        assert isinstance(
            builtin,
            Builtin), f"{CODEGEN_FUNC_ATTR} should return a Builtin object, but got {type(builtin).__name__}: {builtin}"

        if func_resolved not in self.builtin_cache:
            if builtin.body:
                self.writeln_builtin(builtin.body)
            self.builtin_cache.add(func_resolved)
            for hd in builtin.includes:
                if hd not in self.header_cache:
                    self.writeln_header(f"#include {hd}")
                    self.header_cache.add(hd)

        return builtin.return_val

    def _handle_llkernel_call(self, node: ast.Call, func_resolved: LLKernel) -> str:
        """Handle LLKernel recursive calls."""
        # Deduplicate: only emit device function code once per function name
        if func_resolved not in self.builtin_cache:
            from little_kernel.core.passes import PASSES
            passes = PASSES[func_resolved.backend]
            func_tree = func_resolved.lower(passes)
            inferencer = TypeInferencer(func_resolved.ctx, {})
            inferencer.visit(func_tree)
            emitter = self.__class__(func_resolved.ctx, inferencer.all_types)
            emitter.visit(func_tree)
            func_cpp = emitter.get_code(need_header=False)

            for hd in emitter.header_cache:
                if hd not in self.header_cache:
                    self.writeln_header(f"#include {hd}")
                    self.header_cache.add(hd)

            self.writeln_builtin(func_cpp)
            self.builtin_cache.add(func_resolved)

        args_cpp = [self.visit(arg) for arg in node.args]
        assert len(node.keywords) == 0, "Keyword arguments are not supported for recursive LLKernel call"
        return f"{func_resolved.__name__}({', '.join(args_cpp)})"

    def _handle_constant_func_call(self, node: ast.Call, func: ast.Constant) -> str:
        """Handle function calls where func is a Constant."""
        func_name = func.value

        # Map Python print() to C++ printf
        if func_name == "print":
            args_cpp = [self.visit(arg) for arg in node.args]
            return f"printf({', '.join(args_cpp)})"

        if isinstance(func_name, Callable):
            if hasattr(func_name, BUILTIN_ATTR):
                return self._handle_builtin_with_codegen(node, func_name)
            elif isinstance(func_name, LLKernel):
                return self._handle_llkernel_call(node, func_name)

        return None

    def _handle_special_struct_method_call(self, node: ast.Call) -> str:
        """Handle special struct method calls."""
        struct_name = getattr(node, '_special_struct_name', None)
        if not struct_name:
            raise ValueError("Special struct method call missing struct_name")

        # Track that this special struct is used
        self.used_special_structs.add(struct_name)
        codegen_funcs = get_special_struct_codegen(struct_name)
        if codegen_funcs and 'method' in codegen_funcs:
            # Generate method call code
            return codegen_funcs['method'](node, self)

        raise ValueError(f"No method codegen found for special struct: {struct_name}")
