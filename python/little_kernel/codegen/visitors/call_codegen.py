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
from little_kernel.language.intrin.struct_stub import get_struct_stub_info, get_all_struct_stubs
from little_kernel.codegen.visitors.static_int_analysis import (try_eval_const_int, try_eval_const_scalar_int,
                                                                unwrap_scalar_extent_container)
from little_kernel.language.intrin.dtype import to, val_cast


def _emit_builtin_includes(emitter, includes):
    """Emit on-demand #include lines for a builtin helper body/expression."""
    for hd in includes:
        if hd not in emitter.header_cache:
            emitter.writeln_header(f"#include {hd}")
            emitter.header_cache.add(hd)


from little_kernel.language.intrin.scalar_literal import f32 as _lit_f32, f64 as _lit_f64, format_f32_constant, format_f64_constant
from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, CODEGEN_NEEDS_TYPES_ATTR

if TYPE_CHECKING:
    pass


class CallCodegenMixin:
    """Mixin class for call codegen methods."""

    # Array-allocation builtins whose element-count argument (positional
    # index 2) must be an integer constant expression on backends with
    # `_require_constant_array_extents` (CUDA: __shared__ and local arrays
    # both require constant extents; a runtime value is a variable-length
    # array, which nvcc rejects).
    _ARRAY_EXTENT_BUILTINS = frozenset({
        "alloc_shared_memory",
        "alloc_local_memory",
        "alloc_local_zeros",
        "alloc_local_full",
    })

    # Memory builtins whose codegen text is itself a complete C declaration
    # statement (``__shared__ [__align__(N)] <ctype> <name>[elems];`` or a
    # local-array equivalent).  Such text is never a C++ *expression*, so it
    # cannot be emitted where a value is expected:
    #   * as a bare expression statement it is spliced verbatim (visit_Expr
    #     appends the terminating ``;``);
    #   * as an assignment/annotation initializer the statement codegen
    #     splices it as its own declaration and binds the target (a pointer
    #     alias when the target name differs from the declared buffer),
    #     instead of inlining it as ``T* p = __shared__ T p[N];`` -- text a
    #     statement formatter may truncate after ``__shared__`` and nvcc then
    #     reports as a lone ``__attribute__((shared))`` line.
    _DECLARATION_EMITTING_BUILTINS = frozenset({
        "alloc_shared_memory",
        "alloc_local_memory",
        "alloc_local_zeros",
        "alloc_local_full",
        "alloc_dynamic_shared_memory",
    })

    def _declaration_emitting_builtin_name(self, node) -> "str | None":
        """If ``node`` is a call to a builtin whose codegen is a C
        declaration statement, return the builtin name; otherwise None.

        Resolution mirrors visit_Call: the stock intrinsic may be spelled
        ``ll.alloc_shared_memory(...)`` or reach codegen through the
        ctx-registered internal name inserted by InsertMemAlloc.
        """
        if not isinstance(node, ast.Call):
            return None
        if not isinstance(node.func, (ast.Attribute, ast.Name)):
            return None
        try:
            resolved = recursive_resolve_attribute(node.func, self.ctx)
        except Exception:
            return None
        name = getattr(resolved, "__name__", None)
        if name in self._DECLARATION_EMITTING_BUILTINS:
            return name
        return None

    def _validate_array_extent_call(self, node: ast.Call, func_resolved: Callable) -> None:
        """Early legality check: array extents must be constant expressions.

        A provably-static extent (integer literal, or arithmetic over
        static integer locals defined earlier at function top level)
        passes silently; the constexpr local emission elsewhere makes the
        referenced names legal in the generated CUDA.  A runtime-dependent
        or otherwise non-constant extent raises here, with the array name
        and source location, instead of letting invalid variable-length
        array text reach nvcc.  No value is frozen or guessed: the original
        expression text is always what reaches the generated source.
        """
        if not getattr(self, "_require_constant_array_extents", False):
            return
        if getattr(func_resolved, "__name__", None) not in self._ARRAY_EXTENT_BUILTINS:
            return
        if len(node.args) < 3:
            # Type inference/arity errors are reported elsewhere.
            return
        extent_node = node.args[2]
        statics = getattr(self, "_static_int_consts", {})
        # Only statics defined (at function top level) before this statement
        # may be referenced: their emitted declarations precede this call.
        stmt_index = self._top_level_statement_index(node)
        bindings = {
            name: rec.value
            for name, rec in statics.items()
            if stmt_index is None or rec.def_index < stmt_index
        }
        # A scalar extent may be spelled bare (``512``) or as a one-element
        # shape container (``[512]`` / ``(512,)`` — the conventional shape
        # form frontend code uses for a 1-D scratch buffer); both name the
        # same constant bound.  A multi-element list is a real shape, not a
        # scalar extent, and still fails here.
        if try_eval_const_scalar_int(extent_node, bindings) is not None:
            return
        # Also accept anything the type inferencer already folded to an
        # integer type composed only of constants — covered by the fold
        # above; anything else is genuinely not a constant expression.
        name_cpp = "?"
        try:
            name_cpp = self.visit(node.args[0])
        except Exception:
            pass
        loc = (self.get_source_location(node) if hasattr(self, "get_source_location") else "unknown location")
        raise ValueError(f"{func_resolved.__name__}: array '{name_cpp}' has a "
                         f"non-constant element count "
                         f"'{ast.unparse(extent_node) if hasattr(ast, 'unparse') else '?'}' "
                         f"({loc}). This backend requires an integer constant expression "
                         "for static array extents (literal, or arithmetic over typed "
                         "integer locals initialized from constants and never reassigned). "
                         "Use a compile-time constant bound, or allocate via "
                         "alloc_dynamic_shared_memory with a runtime byte size at launch.")

    def _top_level_statement_index(self, target: ast.AST):
        """Index of the innermost function's top-level statement containing
        ``target`` (identity search), or None when not found / nested.

        Array-alloc calls introduced by compiler passes sit at function top
        level; a call nested in a loop body can legally reference every
        top-level constant (their definitions dominate the loop), so None
        means "accept every recorded static".
        """
        stmts = getattr(self, "_current_function_top_stmts", None)
        if not stmts:
            return None
        for index, stmt in enumerate(stmts):
            for sub in ast.walk(stmt):
                if sub is target:
                    return index
        return None

    def _resolve_expr_lltype(self, node):
        """Best-effort LLType for an expression node at emission time.

        Order: the pre-computed all_types map (AST-node identity), then
        scope_vars for Name nodes, then a lazily cached TypeInferencer seeded
        with the current scope (covers BinOp/other rebuilt-by-pass nodes that
        lost their all_types identity).
        """
        t = getattr(self, 'all_types', {}).get(node)
        if isinstance(t, LLType):
            return t
        scope_vars = getattr(self, 'scope_vars', {})
        if isinstance(node, ast.Name):
            t = scope_vars.get(node.id)
            if isinstance(t, LLType):
                return t
        inferencer = getattr(self, '_cast_type_inferencer', None)
        if inferencer is None:
            inferencer = TypeInferencer(ctx=self.ctx, scope_vars=dict(scope_vars))
            self._cast_type_inferencer = inferencer
        try:
            t = inferencer.visit(node)
        except Exception:
            return None
        return t if isinstance(t, LLType) else None

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
                        # 16-bit float targets need intrinsic lowering: torch
                        # extension builds define the no-conversion macros that
                        # delete static_cast-ability of __half/__nv_bfloat16.
                        src_lltype = self._resolve_expr_lltype(node.func.value)
                        conv = self._native_scalar_conversion(val_cpp, src_lltype, dtype_resolved)
                        if conv is not None:
                            code, includes = conv
                            _emit_builtin_includes(self, includes)
                            return code
                        # Use to() builtin function
                        if hasattr(to, CODEGEN_FUNC_ATTR):
                            codegen_func = getattr(to, CODEGEN_FUNC_ATTR)
                            from little_kernel.codegen.policy import emission_scope
                            with emission_scope(self):
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
                            from little_kernel.codegen.policy import cpp_type
                            dtype_cpp = cpp_type(dtype_resolved)
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

        # Check if this is ll.const(condition)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'll' and node.func.attr == 'const':
                if len(node.args) == 1:
                    return self.visit(node.args[0])
                else:
                    raise ValueError(f"ll.const() expects exactly 1 argument, got {len(node.args)}")

        # Check if this is an explicit-width float literal constructor
        # (ll.f32(value) / ll.f64(value)) — pinned-literal emission, or a
        # narrowing static_cast when the argument is not a compile-time
        # constant.  Must run before generic builtin dispatch: these are
        # plain functions with no codegen attribute.
        if func_resolved is _lit_f32 or func_resolved is _lit_f64:
            return self._handle_float_literal_ctor(node, func_resolved)

        # Check if this is an attribute-based struct stub constructor (e.g., ll.Nd2NzParams())
        if isinstance(node.func, ast.Attribute) and func_resolved is not None:
            from little_kernel.language.intrin.struct_stub import is_struct_stub, get_struct_stub_info
            if is_struct_stub(func_resolved):
                stub_name = func_resolved.__name__
                stub_info = get_struct_stub_info(stub_name)
                if stub_info:
                    result = self._generate_struct_stub_constructor(node, stub_info)
                    if result is not None:
                        return result

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
                # 16-bit float targets need intrinsic lowering (see
                # half_float_cast_code); static_cast of __half/__nv_bfloat16
                # from arithmetic types is ill-formed in torch extension
                # builds.
                src_lltype = self._resolve_expr_lltype(node.args[0])
                conv = self._native_scalar_conversion(val_cpp, src_lltype, func_resolved)
                if conv is not None:
                    code, includes = conv
                    _emit_builtin_includes(self, includes)
                    return code
                # Use emitter's _lltype_to_cpp method if available
                if hasattr(self, '_lltype_to_cpp'):
                    dtype_cpp = self._lltype_to_cpp(func_resolved)
                else:
                    # Fallback: import from codegen_cuda
                    from little_kernel.codegen.policy import cpp_type as resolve_cpp_type
                    dtype_cpp = resolve_cpp_type(func_resolved)
                return f"static_cast<{dtype_cpp}>({val_cpp})"

        # Check if this is a val_cast(x, dtype) or to(x, dtype) call with a
        # 16-bit float target.  The builtins' codegen receives only code
        # strings (no source type), so the type-aware intrinsic lowering is
        # applied here; other dtypes fall through to the normal builtin path.
        if func_resolved in (val_cast, to) and len(node.args) == 2:
            try:
                dtype_resolved = recursive_resolve_attribute(node.args[1], self.ctx)
            except Exception:
                dtype_resolved = None
            # const_fold can rewrite a dtype attribute (ll.bfloat16) into a
            # Constant wrapping the LLType instance.
            if (isinstance(dtype_resolved, ast.Constant) and isinstance(dtype_resolved.value, LLType)):
                dtype_resolved = dtype_resolved.value
            elif (isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, LLType)):
                dtype_resolved = node.args[1].value
            if isinstance(dtype_resolved, LLType):
                val_cpp = self.visit(node.args[0])
                src_lltype = self._resolve_expr_lltype(node.args[0])
                conv = self._native_scalar_conversion(val_cpp, src_lltype, dtype_resolved)
                if conv is not None:
                    code, includes = conv
                    _emit_builtin_includes(self, includes)
                    return code

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

        # Check if this is an instance method call on a known scope variable
        if isinstance(node.func.value, ast.Name):
            var_name = node.func.value.id
            if var_name in self.scope_vars:
                return self._generate_instance_method_call(node, var_name, method_name)

        # Check for chained method calls (e.g., expr[offset].ReinterpretCast(ll.float16))
        # The receiver is a complex expression (Subscript, Attribute chain, etc.)
        if not isinstance(node.func.value, ast.Name):
            for _, stub_info in get_all_struct_stubs().items():
                if method_name in stub_info.methods:
                    receiver_cpp = self.visit(node.func.value)
                    return self._generate_instance_method_call(node, receiver_cpp, method_name)
        return None

    def _resolve_template_args(self, node: ast.Call, method_info, args_cpp: list) -> tuple:
        """Resolve template arguments from call args and return (template_str, remaining_args_cpp)."""
        if not method_info.template_args:
            return "", args_cpp

        num_template_args = len(method_info.template_args)
        if len(node.args) < num_template_args:
            return "", args_cpp

        template_parts = []
        for i in range(num_template_args):
            arg_node = node.args[i]
            try:
                resolved = recursive_resolve_attribute(arg_node, self.ctx)
                if isinstance(resolved, LLType):
                    template_parts.append(self._lltype_to_cpp(resolved))
                    continue
            except Exception:
                pass
            if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                template_parts.append(arg_node.value)
            else:
                template_parts.append(args_cpp[i])

        template_str = f"<{', '.join(template_parts)}>"
        remaining_args = args_cpp[num_template_args:]
        return template_str, remaining_args

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

        # Resolve template arguments
        template_str, args_cpp = self._resolve_template_args(node, method_info, args_cpp)

        # Add includes
        for include in stub_info.includes:
            if include not in self.header_cache:
                self.writeln_header(f"#include {include}")
                self.header_cache.add(include)

        if kwargs_cpp:
            all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
            return f"{cpp_struct_name}::{cpp_method_name}{template_str}({', '.join(all_args)})"
        else:
            return f"{cpp_struct_name}::{cpp_method_name}{template_str}({', '.join(args_cpp)})"

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

            # Resolve template arguments
            template_str, args_cpp = self._resolve_template_args(node, method_info, args_cpp)

            # Add includes
            for include in stub_info.includes:
                if include not in self.header_cache:
                    self.writeln_header(f"#include {include}")
                    self.header_cache.add(include)

            if kwargs_cpp:
                all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
                return f"{var_name}.{cpp_method_name}{template_str}({', '.join(all_args)})"
            else:
                return f"{var_name}.{cpp_method_name}{template_str}({', '.join(args_cpp)})"

        # Fallback: generate method call without method info
        args_cpp = [self.visit(arg) for arg in node.args]
        kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords if kv.arg}

        if kwargs_cpp:
            all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
            return f"{var_name}.{method_name}({', '.join(all_args)})"
        else:
            return f"{var_name}.{method_name}({', '.join(args_cpp)})"

    def _generate_struct_stub_constructor(self, node: ast.Call, stub_info) -> str:
        """Generate code for a struct stub constructor from stub_info."""
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

    def _handle_struct_stub_constructor_call(self, node: ast.Call) -> str:
        """Handle struct stub constructor calls (Name-based)."""
        if not isinstance(node.func, ast.Name):
            return None

        func_name = node.func.id
        stub_info = get_struct_stub_info(func_name)
        if not stub_info:
            return None

        return self._generate_struct_stub_constructor(node, stub_info)

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
        self._validate_array_extent_call(node, func_resolved)
        # For a static array-extent argument spelled as a one-element shape
        # container (``[N]`` / ``(N,)``), emit the wrapped scalar expression
        # instead of the generic braced-aggregate ``{N}`` rendering: the
        # braced form is not a valid C++ array bound inside ``T name[N];``.
        # The validator above has already proven the wrapped expression is a
        # compile-time constant whenever this backend requires one; backends
        # without that requirement accept either spelling and the unwrapped
        # scalar is equally valid there.
        visit_args = list(node.args)
        if (len(visit_args) >= 3 and getattr(func_resolved, "__name__", None) in self._ARRAY_EXTENT_BUILTINS):
            visit_args[2] = unwrap_scalar_extent_container(visit_args[2])
        args_cpp = [self.visit(arg) for arg in visit_args]
        kwargs_cpp = {kv.arg: self.visit(kv.value) for kv in node.keywords}
        codegen_func = getattr(func_resolved, CODEGEN_FUNC_ATTR)
        if getattr(func_resolved, CODEGEN_NEEDS_TYPES_ATTR, False):
            # Typed intrinsic: hand the LLType of each positional argument to
            # the codegen function so it can spell a dtype-specific target
            # intrinsic.  Types come from the same inferencer pass that the
            # emitter already uses (get_type / all_types).
            kwargs_cpp = dict(kwargs_cpp)
            kwargs_cpp["_arg_dtypes"] = [self.all_types.get(arg) for arg in node.args]
        from little_kernel.codegen.policy import emission_scope
        with emission_scope(self):
            builtin = codegen_func(*args_cpp, **kwargs_cpp)
        assert isinstance(
            builtin,
            Builtin), f"{CODEGEN_FUNC_ATTR} should return a Builtin object, but got {type(builtin).__name__}: {builtin}"

        # Helper-body emission cache key includes the generated body text:
        # a codegen builtin may produce *argument-dependent* helper bodies
        # (e.g. per-policy variants of gmem_ld_v4, per-immediate variants of
        # wgmma_wait_group/cp_async_wait_group), and every distinct body
        # must be emitted exactly once or later call sites reference
        # undefined helper functions. writeln_builtin additionally dedupes
        # by helper function name, so cross-builtin blocks that repeat an
        # already-emitted helper are skipped here as well.
        body_key = (func_resolved, builtin.body)
        if body_key not in self.builtin_cache:
            if builtin.body:
                self.writeln_builtin(builtin.body)
            self.builtin_cache.add(body_key)
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
            # Lower the callee in its own isolated session and consume THIS
            # call's returned context directly. We must not read a published
            # func_resolved.ctx: under concurrent entry compilation another
            # thread could replace it between lower() and use (the RF07
            # last-session race).
            func_tree, func_ctx = func_resolved.lower_with_context(passes)
            inferencer = TypeInferencer(func_ctx, {})
            inferencer.visit(func_tree)
            emitter = self.__class__(func_ctx, inferencer.all_types)
            emitter.builtin_cache = self.builtin_cache
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

    def _handle_float_literal_ctor(self, node: ast.Call, ctor: Callable) -> str:
        """Emit ll.f32(value)/ll.f64(value) as a width-pinned constant.

        Compile-time constants become C++ literals (``(1.5f)`` /
        ``(1.5)``); anything else lowers to a narrowing ``static_cast``.
        """
        name = "f32" if ctor is _lit_f32 else "f64"
        fmt = format_f32_constant if ctor is _lit_f32 else format_f64_constant
        cpp_t = "float" if ctor is _lit_f32 else "double"
        if len(node.args) != 1 or node.keywords:
            raise ValueError(f"ll.{name}() expects exactly one positional argument, got "
                             f"{len(node.args)} args and {len(node.keywords)} keywords")
        arg = node.args[0]
        not_a_constant = object()  # sentinel: literal_eval can return None
        try:
            value = ast.literal_eval(arg)  # folds unary +/- of constants
        except (ValueError, TypeError, SyntaxError, MemoryError):
            value = not_a_constant
        if value is not not_a_constant:
            # Compile-time constant: legality check before emission — never
            # emit unchecked text for a non-numeric constant (codegen
            # intercepts the call, so the host-side TypeError cannot fire).
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"ll.{name}() expects an int or float constant, got "
                                f"{type(value).__name__}")
            return fmt(float(value))
        # Non-constant expression: explicit narrowing conversion.
        return f"static_cast<{cpp_t}>({self.visit(arg)})"

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
