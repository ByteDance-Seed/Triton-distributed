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
from typing import Dict, Any, Callable, Optional
from little_kernel.core.type_system import LLType, SpecialStructType, AnnotateTypeHelper
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.compile import LLKernel
from little_kernel.codegen.registries.special_struct_registry import get_all_registered_special_structs
from little_kernel.language.intrin.struct_stub import is_struct_stub
from little_kernel.language.builtin_base import (
    EVAL_RETURN_TYPE_ATTR,
    EVAL_ARG_TYPE_ATTR,
    get_intrin_ctx,
)
from .type_inference_core import TypeInferenceError, UnsupportedNodeError, UndefinedVariableError, get_variable_type_from_scope
from ..scope_manager import ScopeManager
from .method_resolver import MethodResolver


class CallTypeInferencer:
    """Handles type inference for function/method calls."""

    def __init__(self, ctx: Dict[str, Any], scope_manager: ScopeManager, resolve_annotation_func: callable,
                 infer_node_type_func: callable, all_types: Dict[ast.AST, LLType]):
        self.ctx = ctx
        self.scope_manager = scope_manager
        self._resolve_annotation = resolve_annotation_func
        self._infer_node_type = infer_node_type_func
        self.all_types = all_types
        # For backwards compatibility, provide current_scope_types
        self.method_resolver = MethodResolver(ctx, scope_manager.current_scope_types, resolve_annotation_func)

    def infer_call_type(self, node: ast.Call) -> LLType:
        """Infer type of function/method calls."""
        # Handle method calls on objects
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return_type = self._infer_method_call(node)
            if return_type is not None:
                return return_type

        # Resolve the called function
        func = self._resolve_function(node)

        # Handle type constructor calls (e.g., ll.int32(5), ll.uint32(10))
        # These are LLType instances used as constructors
        if isinstance(func, LLType):
            # Skip helper types - they are not actual type constructors
            if isinstance(func, AnnotateTypeHelper):
                # Helper types like ll.const, ll.ptr, etc. should not be used as type constructors
                # They are only used in annotations like "x: const[int32] = 5"
                pass
            elif len(node.args) == 1:
                # Type constructor call: ll.int32(value) -> returns int32 type
                # The return type is the type itself
                self.all_types[node] = func
                return func

        # Handle different function types
        if isinstance(func, Callable) and hasattr(func, EVAL_RETURN_TYPE_ATTR):
            return self._infer_builtin_call(node, func)
        elif isinstance(func, LLKernel):
            return self._infer_kernel_call(node, func)
        elif isinstance(func, Callable):
            return self._infer_special_struct_constructor(node, func)
        elif isinstance(func, ast.AST):
            return self._infer_unresolved_call(node, func)
        else:
            raise UnsupportedNodeError(f"Unsupported function type for call: {type(func).__name__} (func: {func})",
                                       node=node)

    def _infer_method_call(self, node: ast.Call) -> Optional[LLType]:
        """Infer type for method calls (e.g., obj.method())."""
        var_name = node.func.value.id
        method_name = node.func.attr

        # Special handling for .to() method calls
        # x.to(dtype) should return the dtype type
        if method_name == 'to' and len(node.args) == 1:
            try:
                # Resolve the dtype argument
                dtype_arg = node.args[0]
                dtype_type = self._resolve_annotation(dtype_arg)
                if isinstance(dtype_type, LLType):
                    self.all_types[node] = dtype_type
                    return dtype_type
            except Exception:
                pass

        # Get variable type from scope if available
        var_type = None
        try:
            # Use scope_manager to search all active scopes
            var_type = get_variable_type_from_scope(self.scope_manager, var_name, node, self.ctx)
        except UndefinedVariableError:
            pass

        # Use method resolver to find return type
        return_type = self.method_resolver.resolve_method_return_type(var_name, method_name, var_type)

        if return_type is not None:
            self.all_types[node] = return_type
            return return_type

        return None

    def _resolve_function(self, node: ast.Call):
        """Resolve the function being called."""
        try:
            # Pass scope_manager to check for local shadowing
            func = recursive_resolve_attribute(node.func, self.ctx, self.scope_manager)
            # If recursive_resolve_attribute returned an AST node, try to resolve it from ctx
            if isinstance(func, ast.Name):
                if func.id in self.ctx:
                    func = self.ctx[func.id]
                else:
                    # Function not found in ctx - provide helpful error message
                    func_name = func.id
                    # Check if it's a builtin that should be in ctx
                    intrin_ctx = get_intrin_ctx()
                    if func_name not in intrin_ctx:
                        raise TypeInferenceError(
                            f"Function '{func_name}' is a builtin but not found in context. "
                            f"This may indicate that the function was defined in a different scope. "
                            f"Make sure the function is defined at module level before the kernel function.",
                            node=node.func)
                    func = intrin_ctx[func_name]
            return func
        except TypeInferenceError:
            # Re-raise TypeInferenceError as-is
            raise
        except Exception as e:
            # Try method resolver as fallback
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                var_name = node.func.value.id
                method_name = node.func.attr
                return_type = self.method_resolver.resolve_method_return_type(var_name, method_name)
                if return_type is not None:
                    self.all_types[node] = return_type
                    return return_type
            raise TypeInferenceError(f"Failed to resolve function for call: {str(e)}", node=node.func) from e

    def _infer_builtin_call(self, node: ast.Call, func: Callable) -> LLType:
        """Infer type for builtin functions with eval_return_type."""
        try:
            if hasattr(func, EVAL_ARG_TYPE_ATTR):
                eval_arg_type = getattr(func, EVAL_ARG_TYPE_ATTR)
                args = []
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        args.append(arg.id)
                    elif isinstance(arg, ast.Constant):
                        args.append(arg.value)
                    else:
                        args.append(arg)
                kwargs = {}
                for kv in node.keywords:
                    if isinstance(kv.value, ast.Name):
                        kwargs[kv.arg] = kv.value.id
                    elif isinstance(kv.value, ast.Constant):
                        kwargs[kv.arg] = kv.value.value
                    else:
                        kwargs[kv.arg] = kv.value
                eval_arg_type(self.scope_manager.current_scope_types, *args, **kwargs)

            # Infer argument types
            arg_types = []
            for i, arg in enumerate(node.args):
                try:
                    arg_type = self._infer_node_type(arg)
                    arg_types.append(arg_type)
                except Exception:
                    # If type inference fails, try to use function parameter annotations as fallback
                    try:
                        sig = inspect.signature(func)
                        params = list(sig.parameters.values())
                        if i < len(params) and params[i].annotation != inspect.Parameter.empty:
                            # Resolve annotation to LLType
                            ann = params[i].annotation
                            if isinstance(ann, str):
                                # String annotation - try to resolve from context
                                if ann in self.ctx:
                                    arg_type = self.ctx[ann]
                                    if isinstance(arg_type, LLType):
                                        arg_types.append(arg_type)
                                        continue
                            else:
                                # Try to resolve annotation
                                try:
                                    arg_type = self._resolve_annotation(ann)
                                    if isinstance(arg_type, LLType):
                                        arg_types.append(arg_type)
                                        continue
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    # If all else fails, re-raise the original exception
                    arg_type = self._infer_node_type(arg)
                    arg_types.append(arg_type)

            kwarg_types = {kv.arg: self._infer_node_type(kv.value) for kv in node.keywords}

            eval_return_type = getattr(func, EVAL_RETURN_TYPE_ATTR)
            call_type = eval_return_type(*arg_types, **kwarg_types)
            self.all_types[node] = call_type
            return call_type
        except Exception as e:
            raise TypeInferenceError(f"Failed to compute return type for function {func.__name__}: {str(e)}",
                                     node=node) from e

    def _infer_kernel_call(self, node: ast.Call, func: LLKernel) -> LLType:
        """Infer type for LLKernel calls."""
        # Import here to avoid circular dependency (kept inline to avoid actual circular import)
        from .infer_type import infer_type
        kernel_scope = infer_type(func.lower([]), func.ctx)
        kernel_return_type = kernel_scope.get(func.py_func.__name__)
        if kernel_return_type is None:
            raise TypeInferenceError(f"Could not infer return type for LLKernel {func.py_func.__name__}", node=node)
        self.all_types[node] = kernel_return_type
        return kernel_return_type

    def _infer_special_struct_constructor(self, node: ast.Call, func: Callable) -> LLType:
        """Infer type for special struct constructors."""
        registered_structs = get_all_registered_special_structs()
        if hasattr(func, '__name__'):
            func_name = func.__name__
            for struct_name, reg_info in registered_structs.items():
                class_obj = reg_info.get('class')
                if class_obj and (func == class_obj or func_name == struct_name):
                    struct_type = SpecialStructType(struct_name)
                    self.all_types[node] = struct_type
                    return struct_type

        if is_struct_stub(func):
            struct_name = func.__name__
            struct_type = SpecialStructType(struct_name)
            self.all_types[node] = struct_type
            return struct_type

        raise UnsupportedNodeError(f"Unsupported function type for call: {type(func).__name__} (func: {func})",
                                   node=node)

    def _infer_unresolved_call(self, node: ast.Call, func: ast.AST) -> LLType:
        """Handle unresolved function calls (method calls on variables)."""
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            var_name = func.value.id
            method_name = func.attr

            var_type = None
            try:
                # Use scope_manager to search all active scopes
                var_type = get_variable_type_from_scope(self.scope_manager, var_name, node, self.ctx)
            except UndefinedVariableError:
                pass

            return_type = self.method_resolver.resolve_method_return_type(var_name, method_name, var_type)

            if return_type is not None:
                self.all_types[node] = return_type
                return return_type

        raise TypeInferenceError(
            "Could not infer return type for method call. "
            "The class is not available in context. Please either:\n"
            "  1. Add the class to the context, or\n"
            "  2. Ensure the method has a return type annotation in its class definition.", node=node)
