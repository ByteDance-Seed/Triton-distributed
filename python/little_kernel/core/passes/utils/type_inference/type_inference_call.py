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
from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR, EVAL_ARG_TYPE_ATTR, INLINE_FUNC_ATTR, ORIGINAL_PY_FUNC_ATTR, get_intrin_ctx
from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError, UnsupportedNodeError, get_variable_type_from_scope
from little_kernel.core.passes.utils.scope_manager import ScopeManager
from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver


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
        # Handle method calls on objects (both direct and chained)
        if isinstance(node.func, ast.Attribute):
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
                # Type constructor call: ll.int32(value) -> returns int32 type.
                # The argument expression must be inferred in full: without it
                # the argument subtree holds no all_types entries, and codegen's
                # f32-literal contract (bare float constants are typed
                # float32; see type_inference_ast_visitors.visit_BinOp and
                # expression_codegen.visit_Constant) silently regresses to
                # bare C++ double literals inside constructor calls, e.g.
                # ll.float16(v * 2.0) emitted `__float2half_rn(v * 2.0)`
                # instead of `... * 2.0f)`. A cast never hides a program
                # error: undefined operands or type errors in the argument
                # propagate with their original node-level diagnosis.
                self._infer_node_type(node.args[0])
                self.all_types[node] = func
                return func

        # Handle different function types
        if isinstance(func, Callable) and hasattr(func, EVAL_RETURN_TYPE_ATTR):
            return self._infer_builtin_call(node, func)
        elif isinstance(func, LLKernel):
            return self._infer_kernel_call(node, func)
        elif isinstance(func, Callable) and hasattr(func, INLINE_FUNC_ATTR) and getattr(func, INLINE_FUNC_ATTR, False):
            return self._infer_inline_func_call(node, func)
        elif isinstance(func, Callable):
            return self._infer_special_struct_constructor(node, func)
        elif isinstance(func, ast.AST):
            return self._infer_unresolved_call(node, func)
        else:
            raise UnsupportedNodeError(f"Unsupported function type for call: {type(func).__name__} (func: {func})",
                                       node=node)

    def _infer_method_call(self, node: ast.Call) -> Optional[LLType]:
        """Infer type for method calls (e.g., obj.method() or expr[offset].method())."""
        method_name = node.func.attr

        # Special handling for .to() method calls: x.to(dtype) -> dtype type.
        # An argument that statically resolves to an LLType is a cast. A
        # bare Name/Attribute that is genuinely unresolvable is a supported
        # optional probe (it may be a real method of x's type) and falls
        # through to the method resolver; a concrete invalid annotation is
        # an input error and propagates with its diagnosis.
        if method_name == 'to' and len(node.args) == 1:
            dtype_arg = node.args[0]
            try:
                dtype_type = self._resolve_annotation(dtype_arg)
            except TypeInferenceError:
                unresolved_probe = isinstance(dtype_arg, (ast.Name, ast.Attribute))
                if not unresolved_probe:
                    raise
            else:
                if isinstance(dtype_type, LLType):
                    self.all_types[node] = dtype_type
                    return dtype_type

        # Direct method call on named variable: var.method()
        if isinstance(node.func.value, ast.Name):
            var_name = node.func.value.id
            var_type = get_variable_type_from_scope(self.scope_manager, var_name, node, self.ctx, required=False)

            return_type = self.method_resolver.resolve_method_return_type(var_name, method_name, var_type)
            if return_type is not None:
                self.all_types[node] = return_type
                return return_type
        else:
            # Chained method call: expr[offset].Method() or expr.attr.Method()
            return_type = self.method_resolver.resolve_method_return_type(None, method_name, None)
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

    _UNRESOLVED = object()

    def _resolve_attribute_value(self, node: ast.Attribute) -> Any:
        """Resolve an Attribute chain (e.g. ``ll.float32``) to its host value.

        Walks from the base Name, looked up in the kernel context, through
        getattr.  Returns ``_UNRESOLVED`` when the chain is not statically
        known (local variable base, missing attribute, non-Name base).
        """
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if not isinstance(cur, ast.Name) or cur.id not in self.ctx:
            return self._UNRESOLVED
        value = self.ctx[cur.id]
        try:
            for attr in reversed(parts):
                value = getattr(value, attr)
        except AttributeError:
            return self._UNRESOLVED
        return value

    def _resolve_static_builtin_arg(self, arg: ast.AST) -> Any:
        """Materialize an argument for ``eval_arg_type``.

        Constants pass through as values and bare Names as identifier strings
        (legacy semantics, e.g. buffer-name style parameters).  Names that
        denote an ``LLType`` in the kernel context (dtype aliases like
        ``F32 = ll.float32``) and Attribute chains that statically resolve to
        a host value (e.g. ``ll.float32``) are forwarded as the value itself,
        so ``eval_arg_type`` implementations receive the LLType they document
        instead of a raw AST node or identifier string.
        """
        if isinstance(arg, ast.Constant):
            return arg.value
        if isinstance(arg, ast.Name):
            value = self.ctx.get(arg.id)
            if isinstance(value, LLType):
                return value
            return arg.id
        if isinstance(arg, ast.Attribute):
            value = self._resolve_attribute_value(arg)
            if value is not self._UNRESOLVED:
                return value
        return arg

    def _try_annotation_arg_type(self, func: Callable, index: int) -> Optional[LLType]:
        """Optional fallback argument type from a parameter annotation.

        Returns the resolved LLType only when a concrete annotation is
        available and resolvable; returns None for the no-annotation case so
        the caller re-raises the operand's original error.
        """
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        if index >= len(params):
            return None
        ann = params[index].annotation
        if ann == inspect.Parameter.empty:
            return None
        if isinstance(ann, str):
            # Legacy spelling: a bare type-name string found in ctx.
            ctx_value = self.ctx.get(ann)
            return ctx_value if isinstance(ctx_value, LLType) else None
        if isinstance(ann, LLType):
            return ann
        try:
            resolved = self._resolve_annotation(ann)
        except TypeInferenceError:
            return None
        return resolved if isinstance(resolved, LLType) else None

    def _infer_builtin_call(self, node: ast.Call, func: Callable) -> LLType:
        """Infer type for builtin functions with eval_return_type."""
        try:
            if hasattr(func, EVAL_ARG_TYPE_ATTR):
                eval_arg_type = getattr(func, EVAL_ARG_TYPE_ATTR)
                # Shared static-arg resolution (merge of HEAD closure and
                # 70ad465e helper): Constants forward literal values; Names
                # bound to an LLType in ctx (dtype aliases) forward the type,
                # other names stay identifier strings for symbol-table-key
                # lookups; resolvable attribute chains forward values, else
                # the raw node (legacy fallback).
                args = [self._resolve_static_builtin_arg(arg) for arg in node.args]
                kwargs = {kv.arg: self._resolve_static_builtin_arg(kv.value) for kv in node.keywords}
                eval_arg_type(self.scope_manager.current_scope_types, *args, **kwargs)

            # Infer argument types. A program error in an operand is a
            # supported optional probe in exactly two narrow ways:
            #   1. Re-infer once: some visitors cache the operand's result
            #      while infer_node_type rejects a non-LLType (e.g. an
            #      unsupported constant such as None); the second lookup
            #      returns the cached value and the backend's own operand
            #      validation owns the TypeError/ValueError contract.
            #   2. Fall back to a concrete parameter annotation when one
            #      resolves to an LLType.
            # Undefined symbols and invalid expressions fail both probes
            # and propagate as the original typed diagnosis -- a call never
            # swallows them. Other exception classes (not TypeInferenceError)
            # are program errors and escape untouched.
            arg_types = []
            for i, arg in enumerate(node.args):
                try:
                    arg_types.append(self._infer_node_type(arg))
                    continue
                except TypeInferenceError as first_error:
                    try:
                        arg_types.append(self._infer_node_type(arg))
                        continue
                    except TypeInferenceError:
                        pass
                    fallback_type = self._try_annotation_arg_type(func, i)
                    if fallback_type is not None:
                        arg_types.append(fallback_type)
                        continue
                    raise first_error

            kwarg_types = {kv.arg: self._infer_node_type(kv.value) for kv in node.keywords}

            eval_return_type = getattr(func, EVAL_RETURN_TYPE_ATTR)
            call_type = eval_return_type(*arg_types, **kwarg_types)
            self.all_types[node] = call_type
            return call_type
        except TypeInferenceError:
            # Argument-validation errors (eval_arg_type validators) already
            # carry a precise diagnosis; do not bury them under a generic
            # "return type" wrapper.
            raise
        except Exception as e:
            raise TypeInferenceError(f"Failed to compute return type for function {func.__name__}: {str(e)}",
                                     node=node) from e

    def _infer_kernel_call(self, node: ast.Call, func: LLKernel) -> LLType:
        """Infer type for LLKernel calls."""
        # Import here to avoid circular dependency (kept inline to avoid actual circular import)
        from little_kernel.core.passes.utils.type_inference.infer_type import infer_type
        # RF07: lower the callee in an isolated session and infer against
        # the returned per-call context. Reading the shared ``func.ctx``
        # after ``lower`` would race with a concurrent compilation that
        # publishes a different last session there.
        func_tree, func_ctx = func.lower_with_context([])
        kernel_scope = infer_type(func_tree, func_ctx)
        kernel_return_type = kernel_scope.get(func.py_func.__name__)
        if kernel_return_type is None:
            raise TypeInferenceError(f"Could not infer return type for LLKernel {func.py_func.__name__}", node=node)
        self.all_types[node] = kernel_return_type
        return kernel_return_type

    def _infer_inline_func_call(self, node: ast.Call, func: Callable) -> LLType:
        """Infer return type for @inline_func decorated functions from annotation."""
        original = getattr(func, ORIGINAL_PY_FUNC_ATTR, func)
        ann = inspect.signature(original).return_annotation
        if ann == inspect.Parameter.empty:
            raise TypeInferenceError(
                f"Inline function {getattr(func, '__name__', '?')} must have return type annotation", node=node)
        # With ``from __future__ import annotations``, annotations are strings.
        # Parse them into AST nodes for the resolver.
        if isinstance(ann, str):
            ann = ast.parse(ann, mode="eval").body
        # If the annotation is already an LLType instance (e.g. ll.uint32),
        # use it directly without AST resolution.
        if isinstance(ann, LLType):
            self.all_types[node] = ann
            return ann
        try:
            ret_type = self._resolve_annotation(ann)
        except TypeInferenceError as e:
            raise TypeInferenceError(
                f"Could not resolve return annotation for inline function {getattr(func, '__name__', '?')}",
                node=node) from e
        if isinstance(ret_type, LLType):
            self.all_types[node] = ret_type
            return ret_type
        raise TypeInferenceError(
            f"Could not resolve return annotation for inline function {getattr(func, '__name__', '?')}", node=node)

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

            var_type = get_variable_type_from_scope(self.scope_manager, var_name, node, self.ctx, required=False)

            return_type = self.method_resolver.resolve_method_return_type(var_name, method_name, var_type)

            if return_type is not None:
                self.all_types[node] = return_type
                return return_type

        # For chained calls like expr[offset].Method(), try method resolver
        if isinstance(func, ast.Attribute):
            method_name = func.attr
            return_type = self.method_resolver.resolve_method_return_type(None, method_name, None)
            if return_type is not None:
                self.all_types[node] = return_type
                return return_type

        raise TypeInferenceError(
            "Could not infer return type for method call. "
            "The class is not available in context. Please either:\n"
            "  1. Add the class to the context, or\n"
            "  2. Ensure the method has a return type annotation in its class definition.", node=node)
