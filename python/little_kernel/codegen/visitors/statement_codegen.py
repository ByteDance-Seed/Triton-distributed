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
Statement codegen visitors for CppEmitter.

This module contains visitor methods for AST statement nodes.
"""

import ast
from typing import TYPE_CHECKING
import little_kernel.language as ll
from little_kernel.core.type_system import LLType, TupleType, ConstTypeHelper, ConstAnnotationType, const_annotation, Const
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.passes.utils.ir_nodes import EnumDef, SpecialStructDef, is_ir_node
from little_kernel.core import ir as lk_ir
from little_kernel.core.passes.utils.error_report import format_error_message
from little_kernel.language.intrin.struct_stub import get_struct_stub_info, get_all_struct_stubs
from little_kernel.codegen.registries.enum_registry import generate_enum_cpp_definition
from little_kernel.codegen.registries.special_struct_registry import is_special_struct, get_special_struct_name, get_special_struct_codegen, get_all_registered_special_structs

if TYPE_CHECKING:
    pass

# Allocation builtins whose codegen is itself a complete C array
# declaration (the *_zeros / *_full variants append an initialization
# statement).  The memory-analysis pass binds these to a name by inserting
# them as bare expression statements; the helpers below also accept the
# direct assignment spelling so a declaration is never wrapped in an
# ill-formed pointer initializer ("T* raw = T raw[N];").
_DECLARATION_ALLOC_BUILTINS = frozenset({
    "alloc_local_memory",
    "alloc_local_zeros",
    "alloc_local_full",
    "alloc_shared_memory",
})


class StatementCodegenMixin:
    """Mixin class for statement codegen methods."""

    def _declaration_alloc_binding(self, value_node):
        """Return ``(builtin_name, declared_array_name)`` for a declaration
        allocation call, or None when ``value_node`` is not one.

        The declared name is the allocator's first (string) argument, the
        identifier that actually appears in the emitted C declaration.
        """
        if not isinstance(value_node, ast.Call) or not value_node.args:
            return None
        try:
            callee = recursive_resolve_attribute(value_node.func, self.ctx)
        except Exception:
            return None
        builtin_name = getattr(callee, "__name__", None)
        if builtin_name not in _DECLARATION_ALLOC_BUILTINS:
            return None
        first = value_node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            return None
        return builtin_name, first.value.strip('"').strip("'")

    def _emit_declaration_alloc_assignment(self, var_name, value_node, lltype=None):
        """Emit a declaration allocation bound to ``var_name``.

        Returns True when ``value_node`` was such an allocation.  The
        builtin's own codegen is the declaration (so it is emitted exactly
        like the pass-inserted bare-statement form), and ``var_name`` is
        bound in scope to the allocated tensor/pointer type.

        Local register arrays must keep exactly one name: spelling
        ``x = ll.alloc_local_memory("y", ...)`` would require binding ``x``
        through a pointer to the C array ``y``, and taking the address of a
        local array forces it out of registers onto the stack — exactly the
        spill the register-array path exists to avoid — so it is a
        source-located compile error.  Shared memory already lives in a
        named memory segment, so a mismatched name emits the declaration and
        a typed pointer alias (``T* x = y;``), the hand-written form.
        """
        binding = self._declaration_alloc_binding(value_node)
        if binding is None:
            return False
        builtin_name, declared_name = binding
        if declared_name != var_name:
            if builtin_name != "alloc_shared_memory":
                raise ValueError(f"{builtin_name} declares array '{declared_name}', but it "
                                 f"is assigned to '{var_name}'; a register-scope local "
                                 f"array cannot be aliased under another name (its address "
                                 f"would escape and force a stack spill), so the names "
                                 f"must match (or call {builtin_name} as a bare "
                                 f"statement).")
            decl_cpp = self.visit(value_node)
            self.writeln_main(f"{decl_cpp};")
            if lltype is None:
                try:
                    lltype = self.get_type(value_node)
                except Exception:
                    lltype = None
            # auto* deduces the element pointer from the declared shared
            # array (mutable, matching later indexing/calls).
            self.writeln_main(f"auto* {var_name} = {declared_name};")
            self.scope_vars[var_name] = lltype
            return True
        decl_cpp = self.visit(value_node)
        self.writeln_main(f"{decl_cpp};")
        if lltype is None:
            lltype = self.get_type(value_node)
        self.scope_vars[var_name] = lltype
        return True

    def _maybe_static_constexpr_decl(self, var_name: str, cpp_type: str, value_cpp: str) -> str:
        """Declaration for a fresh local that may be a compile-time constant.

        When the per-function static-integer analysis proved ``var_name``
        constant (single constant definition, never mutated) and the backend
        opted in (``_emit_static_constexpr``), emit it as a C++ ``static
        constexpr`` local so it qualifies as an integer constant expression
        (legal ``__shared__``/local array extent, case label, etc.).  The
        analysis only binds the *original* initializer text, and nvcc
        re-verifies both constancy of the initializer and absence of later
        stores, so a wrong conclusion is a hard compile error, never silent
        miscompilation.  Everything else keeps the historical plain decl.
        """
        rec = getattr(self, "_static_int_consts", {}).get(var_name)
        if rec is not None and getattr(self, "_emit_static_constexpr", False):
            return f"static constexpr {cpp_type} {var_name} = {value_cpp};"
        return f"{cpp_type} {var_name} = {value_cpp};"

    def _declared_buffer_identifier(self, value_node: ast.Call):
        """The C identifier declared by a declaration-emitting alloc call.

        The first argument names the buffer: a quoted string constant for
        the stock ``ll.alloc_shared_memory("s", ...)`` spelling, or an
        already-resolved Name for the internal call InsertMemAlloc inserts.
        Returns None for non-identifier targets (e.g. ``A[0]`` slices).
        """
        args = getattr(value_node, "args", None)
        if not args:
            return None
        try:
            text = self.visit(args[0])
        except Exception:
            return None
        ident = text.strip().strip('"').strip("'")
        return ident if ident.isidentifier() else None

    def _emit_declaration_emitting_initializer(self, target: ast.Name, value_node: ast.AST) -> bool:
        """Emit an assignment/annotation whose value is a call to a
        declaration-emitting memory builtin.

        The builtin's codegen text is itself the C declaration
        (``__shared__ [__align__(N)] <ctype> <name>[elems]`` or a local
        equivalent, possibly followed by built-in init statements).  That
        text is not an expression and must be spliced as its own statement
        rather than inlined as ``<ctype>* <target> = <declaration>;``.

        The declared buffer name is registered in the type context by the
        builtin's ``eval_arg_type``; the assignment target binds the same
        tensor type here.  When the Python target name differs from the
        declared buffer name (e.g. ``tile = ll.alloc_shared_memory(
        "s_tile", ...)`` where ``tile`` is later passed by pointer), an
        explicit pointer alias is emitted after the declaration.

        Returns True when it handled the statement, False otherwise.
        """
        if self._declaration_emitting_builtin_name(value_node) is None:
            return False
        # The strict declaration-alloc contract handler owns the four
        # declaration allocators: a local register array must keep exactly
        # one name (a mismatched target is a compile error rather than a
        # pointer alias that would force the array onto the stack), while a
        # shared-memory declaration may bind a typed pointer alias. It must
        # run before the generic alias path below, which would otherwise
        # preempt it and emit an ill-formed pointer initializer.
        if self._declaration_alloc_binding(value_node) is not None:
            return self._emit_declaration_alloc_assignment(target.id, value_node)
        # Visiting the call also runs the constant-extent legality check
        # and returns the full declaration text (without a terminator).
        decl_text = self.visit(value_node)
        self.writeln_main(f"{decl_text.rstrip()};")

        elem_lltype = None
        try:
            elem_lltype = self.get_type(value_node)
        except Exception:
            pass
        declared = self._declared_buffer_identifier(value_node)
        target_name = target.id
        if declared and declared != target_name:
            # The target aliases the declared array; auto* deduces the
            # element pointer (mutable, matching later indexing/calls).
            self.writeln_main(f"auto* {target_name} = {declared};")
        if declared:
            self.scope_vars.setdefault(declared, elem_lltype)
        self.scope_vars[target_name] = elem_lltype
        return True

    def visit_Module(self, node: ast.Module) -> None:
        """Process top-level module, including IR nodes for enum and struct definitions."""

        # Separate IR nodes from regular statements
        ir_definitions = []
        regular_statements = []

        for stmt in node.body:
            if is_ir_node(stmt):
                ir_definitions.append(stmt)
            else:
                regular_statements.append(stmt)

        # Process IR definitions first (enum and struct definitions)
        if ir_definitions:
            self.writeln_struct("// Enum and special struct definitions from IR")
            for ir_node in ir_definitions:
                if isinstance(ir_node, EnumDef):
                    enum_def = generate_enum_cpp_definition(ir_node.enum_name, ir_node.enum_class)
                    self.writeln_struct(enum_def)
                    self.writeln_struct()  # Blank line between definitions
                elif isinstance(ir_node, SpecialStructDef):
                    struct_name = ir_node.struct_name
                    codegen_funcs = get_special_struct_codegen(struct_name)
                    if codegen_funcs and 'struct_definition' in codegen_funcs and codegen_funcs['struct_definition']:
                        struct_def = codegen_funcs['struct_definition'](self)
                        if struct_def:
                            self.writeln_struct(struct_def)
                            self.writeln_struct()  # Blank line between definitions

        # Process regular statements
        for stmt in regular_statements:
            self.visit(stmt)
            self.writeln_main()  # Blank line between top-level elements

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definition: initialize scope with parameters, then process body."""
        # Resolve return type
        return_lltype = self.get_type(node.returns)
        return_cpp = self._lltype_to_cpp(return_lltype)

        # Resolve parameters and populate initial scope
        cpp_params = []
        old_scope_vars = self.scope_vars
        self.scope_vars = {}  # Reset scope for new function
        for arg in node.args.args:
            arg_lltype = self.get_type(arg.annotation)
            arg_cpp = self._lltype_to_cpp(arg_lltype)
            cpp_params.append(f"{arg_cpp} {arg.arg}")
            # Add parameter to scope (prevents re-declaration in body)
            self.scope_vars[arg.arg] = arg_lltype

        # Write function signature
        self.writeln_main(f"{return_cpp} {node.name}({', '.join(cpp_params)}) {{")
        self.indent_level += 1

        # Provably-static integer locals (constant-expression extents).
        self._begin_function_static_consts(node)

        # Process function body (statements like Assign, Return)
        for stmt in node.body:
            self.visit(stmt)

        # Cleanup: close function and reset scope
        self.indent_level -= 1
        self.writeln_main("}")
        self.writeln_main("")
        self.scope_vars = old_scope_vars  # Restore scope after function
        self._end_function_static_consts()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Process annotated assignments (e.g., "x: const = 5" or "x: int32 = 5")."""
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"Annotated assignment target must be a single variable, got {type(node.target).__name__}")

        if node.value is not None:
            # A declaration-emitting memory builtin is spliced as its own
            # complete declaration (plus an optional target alias), never
            # inlined into this annotation's initializer.
            if self._emit_declaration_emitting_initializer(node.target, node.value):
                return

        var_name = node.target.id

        # Try to resolve the annotation to check if it's const and extract type
        is_const_annotation = False
        const_inner_type = None
        try:
            resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx)

            # Check if it's const[int32] (Const type with inner_type)
            if isinstance(resolved_ann, Const):
                is_const_annotation = True
                const_inner_type = resolved_ann.inner_type
            # Check if it's just "const" (ConstTypeHelper or ConstAnnotationType)
            elif (isinstance(resolved_ann, ConstTypeHelper) or isinstance(resolved_ann, ConstAnnotationType)
                  or resolved_ann == const_annotation):
                is_const_annotation = True
                const_inner_type = None  # Need to infer from value
        except Exception:
            # Fallback: check for "const" as a Name node
            if isinstance(node.annotation, ast.Name) and node.annotation.id == "const":
                is_const_annotation = True
                const_inner_type = None

        # Handle "const" annotation
        if is_const_annotation:
            if node.value is not None:
                value_cpp = self.visit(node.value)
                if var_name in self.scope_vars:
                    self.writeln_main(f"{var_name} = {value_cpp};")
                else:
                    # Use inner_type if specified (const[int32]), otherwise infer from value (const)
                    if const_inner_type is not None:
                        var_lltype = const_inner_type
                    else:
                        var_lltype = self.get_type(node.value)
                    var_cpp_type = self._lltype_to_cpp(var_lltype)
                    self.writeln_main(f"const {var_cpp_type} {var_name} = {value_cpp};")
                    self.scope_vars[var_name] = var_lltype
        else:
            # Regular type annotation - resolve the annotation AST node to LLType
            # First try to get from all_types (already resolved by type inference)
            if node.annotation in self.all_types:
                target_type = self.get_type(node.annotation)
            else:
                # Fallback: try to resolve using recursive_resolve_attribute
                try:
                    resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx)
                    if isinstance(resolved_ann, LLType):
                        target_type = resolved_ann
                    else:
                        # If it's a Subscript (like const[Type]), try to resolve using type inference logic
                        if isinstance(node.annotation, ast.Subscript):
                            temp_inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
                            target_type = temp_inferencer._resolve_annotation(node.annotation)
                        else:
                            raise ValueError(
                                f"Failed to resolve annotation to LLType, got {type(resolved_ann).__name__}")
                except Exception as e:
                    raise ValueError(
                        f"Failed to resolve type annotation for {var_name}: {e}. Annotation: {ast.dump(node.annotation) if hasattr(ast, 'dump') else str(node.annotation)}"
                    )

            target_cpp_type = self._lltype_to_cpp(target_type)

            if node.value is not None:
                if self._emit_declaration_alloc_assignment(var_name, node.value, lltype=target_type):
                    return
                value_cpp = self.visit(node.value)
                if var_name in self.scope_vars:
                    existing_type = self.scope_vars[var_name]
                    if existing_type != target_type:
                        self.writeln_main(f"{var_name} = static_cast<{target_cpp_type}>({value_cpp});")
                    else:
                        self.writeln_main(f"{var_name} = {value_cpp};")
                else:
                    decl = self._maybe_static_constexpr_decl(var_name, target_cpp_type, value_cpp)
                    self.writeln_main(decl)
                    self.scope_vars[var_name] = target_type
            else:
                # Just declaration without value (e.g., "x: int32")
                self.writeln_main(f"{target_cpp_type} {var_name};")
                self.scope_vars[var_name] = target_type

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignment (single-target or tuple unpacking)."""
        if len(node.targets) != 1:
            raise NotImplementedError("Only single-target assignments are supported")

        target = node.targets[0]
        value_node = node.value

        # A declaration-emitting memory builtin is spliced as its own
        # complete declaration (plus an optional target alias), never
        # inlined into this assignment's initializer.
        if isinstance(target, ast.Name):
            if self._emit_declaration_emitting_initializer(target, value_node):
                return

        # Check if this is a special struct declaration
        if hasattr(node, '_is_special_struct_decl') and node._is_special_struct_decl:
            struct_name = getattr(node, '_special_struct_name', None)
            if struct_name:
                self.used_special_structs.add(struct_name)
                codegen_func = get_special_struct_codegen(struct_name)
                if codegen_func and 'declaration' in codegen_func:
                    cpp_code = codegen_func['declaration'](node, self)
                    self.writeln_main(cpp_code)
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self.scope_vars[var_name] = ll.int32  # Placeholder
                    return

        # Check if this is a generic special struct marker
        if is_special_struct(node):
            struct_name = get_special_struct_name(node)
            if struct_name:
                codegen_funcs = get_special_struct_codegen(struct_name)
                if codegen_funcs and 'declaration' in codegen_funcs:
                    cpp_code = codegen_funcs['declaration'](node, self)
                    self.writeln_main(cpp_code)
                    return

        # Check if this is a struct stub constructor call
        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name):
            func_name = value_node.func.id
            stub_info = get_struct_stub_info(func_name)
            if stub_info:
                self._handle_struct_stub_constructor_assignment(node, target, value_node, stub_info)
                return

        # Check for attribute-based struct stub constructor (e.g., ll.Nd2NzParams())
        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Attribute):
            try:
                func_resolved = recursive_resolve_attribute(value_node.func, self.ctx)
                from little_kernel.language.intrin.struct_stub import is_struct_stub
                if is_struct_stub(func_resolved):
                    stub_info = get_struct_stub_info(func_resolved.__name__)
                    if stub_info:
                        self._handle_struct_stub_constructor_assignment(node, target, value_node, stub_info)
                        return
            except Exception:
                pass

        # Check if value_node is a special struct constructor call that wasn't transformed
        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Name):
            func_name = value_node.func.id
            registered_structs = get_all_registered_special_structs()
            for struct_name in registered_structs.keys():
                if func_name == struct_name:
                    raise NotImplementedError(
                        f"{struct_name} constructor call was not transformed by special_struct_materialize_pass. "
                        f"This is a compiler bug. Please check that special_struct_materialize_pass is running correctly. "
                        f"Node: {ast.dump(node, indent=2)}")

        # Handle single Name target
        if isinstance(target, ast.Name):
            self._handle_single_name_assignment(node, target, value_node)
        # Handle tuple unpacking
        elif isinstance(target, ast.Tuple):
            self._handle_tuple_assignment(node, target, value_node)
        # Handle attribute assignment (e.g. params.ndNum = 1)
        elif isinstance(target, ast.Attribute):
            target_cpp = self.visit(target)
            value_cpp = self.visit(value_node)
            value_cpp = self._map_enum_field(target.attr, value_cpp)
            self.writeln_main(f"{target_cpp} = {value_cpp};")
        # Handle Subscript target (array/tensor element assignment, e.g., array[idx] = value)
        elif isinstance(target, ast.Subscript):
            self._handle_subscript_assignment(node, target, value_node)
        else:
            message = f"Unsupported assignment target type: {type(target).__name__}"
            formatted = format_error_message(message, target, ctx=self.ctx)
            raise NotImplementedError(formatted)

    # Mapping from backend struct field names to enum type converters.
    _ENUM_FIELD_MAP = {}

    def _map_enum_field(self, field_name: str, value_cpp: str) -> str:
        """Map integer literals to C++ enum values for known struct fields."""
        mapping = self._ENUM_FIELD_MAP.get(field_name)
        if mapping and value_cpp in mapping:
            return mapping[value_cpp]
        return value_cpp

    # Mapping from DSL template tag strings to C++ enum values.
    _TEMPLATE_ENUM_MAP = {}

    def _resolve_constructor_template_args(self, value_node: ast.Call, stub_info):
        """Resolve template arguments for struct stub constructors."""
        if not stub_info.template_params:
            return "", list(value_node.args)
        num_template_params = len(stub_info.template_params)
        if len(value_node.args) < num_template_params:
            return "", list(value_node.args)
        template_parts = []
        for i in range(num_template_params):
            arg_node = value_node.args[i]
            try:
                resolved = recursive_resolve_attribute(arg_node, self.ctx)
                from little_kernel.core.type_system import LLType
                if isinstance(resolved, LLType):
                    template_parts.append(self._lltype_to_cpp(resolved))
                    continue
            except Exception:
                pass
            if isinstance(arg_node, ast.Constant) and isinstance(arg_node.value, str):
                raw = arg_node.value
                mapped = self._TEMPLATE_ENUM_MAP.get(raw, raw)
                template_parts.append(mapped)
            else:
                template_parts.append(self.visit(arg_node))
        template_str = f"<{', '.join(template_parts)}>"
        remaining_args = list(value_node.args[num_template_params:])
        return template_str, remaining_args

    def _handle_struct_stub_constructor_assignment(self, node: ast.Assign, target: ast.AST, value_node: ast.Call,
                                                   stub_info):
        """Handle struct stub constructor assignment."""
        target_name = target.id if isinstance(target, ast.Name) else None
        if target_name is None:
            raise NotImplementedError("Struct stub assignment must have a simple variable name as target")

        # Resolve template parameters (e.g., GlobalTensor(ll.float16) -> <half>)
        template_str, remaining_args = self._resolve_constructor_template_args(value_node, stub_info)

        # Generate remaining constructor arguments
        args_cpp = [self.visit(arg) for arg in remaining_args]
        kwargs_cpp = {}
        for kw in value_node.keywords:
            if kw.arg:
                kwargs_cpp[kw.arg] = self.visit(kw.value)

        # Build constructor call
        cpp_struct_name = stub_info.cpp_struct_name
        if stub_info.namespace:
            cpp_struct_name = f"{stub_info.namespace}::{cpp_struct_name}"
        cpp_type_name = f"{cpp_struct_name}{template_str}"

        # Separate args consumed by init_method from constructor args
        init_method_args = []
        if stub_info.init_method and args_cpp:
            init_method_args = args_cpp
            args_cpp = []

        # Generate construction + optional field assignments for kwargs
        if args_cpp:
            self.writeln_main(f"{cpp_type_name} {target_name}({', '.join(args_cpp)});")
        else:
            self.writeln_main(f"{cpp_type_name} {target_name};")

        if init_method_args:
            cpp_method = stub_info.init_method
            self.writeln_main(f"{target_name}.{cpp_method}({', '.join(init_method_args)});")

        for field_name, field_value in kwargs_cpp.items():
            field_value = self._map_enum_field(field_name, field_value)
            self.writeln_main(f"{target_name}.{field_name} = {field_value};")

        # Add includes
        for include in stub_info.includes:
            if include not in self.header_cache:
                self.writeln_header(f"#include {include}")
                self.header_cache.add(include)

        # Register variable in scope with correct struct type for type verification
        from little_kernel.core.type_system import SpecialStructType
        if isinstance(value_node.func, ast.Name):
            struct_name = value_node.func.id
        else:
            struct_name = recursive_resolve_attribute(value_node.func, self.ctx).__name__
        self.scope_vars[target_name] = SpecialStructType(struct_name)

    def _is_struct_stub_method_call(self, value_node: ast.AST) -> bool:
        """Check if a value node is a struct stub method call (instance or chained)."""
        if not isinstance(value_node, ast.Call):
            return False
        if not isinstance(value_node.func, ast.Attribute):
            return False
        method_name = value_node.func.attr
        if isinstance(value_node.func.value, ast.Name):
            var_name = value_node.func.value.id
            if var_name in self.scope_vars:
                for _, stub_info in get_all_struct_stubs().items():
                    if method_name in stub_info.methods:
                        return True
        else:
            for _, stub_info in get_all_struct_stubs().items():
                if method_name in stub_info.methods:
                    return True
        return False

    def _handle_single_name_assignment(self, node: ast.Assign, target: ast.Name, value_node: ast.AST):
        """Handle single name assignment (e.g., x = 5)."""
        var_name = target.id

        # A declaration allocation ("raw = ll.alloc_local_memory("raw", ...)")
        # emits the declaration itself instead of a pointer initialized from
        # a declaration expression.
        if self._emit_declaration_alloc_assignment(var_name, value_node):
            return

        # Check if this is a special struct method call
        if isinstance(value_node, ast.Call) and hasattr(value_node, '_is_special_struct_method'):
            struct_name = getattr(value_node, '_special_struct_name', None)
            if struct_name:
                codegen_funcs = get_special_struct_codegen(struct_name)
                if codegen_funcs and 'method' in codegen_funcs:
                    # Get return type info
                    return_type_info = getattr(value_node, '_return_type_info', None)

                    if return_type_info:
                        if isinstance(return_type_info, TupleType):
                            # Tuple return - handled by tuple unpacking codegen
                            pass
                        else:
                            # Single return value - assign from function return
                            # Declare variable if needed
                            if var_name not in self.scope_vars:
                                ret_cpp_type = self._lltype_to_cpp(return_type_info)
                                self.writeln_main(f"{ret_cpp_type} {var_name};")
                                self.scope_vars[var_name] = return_type_info

                            # Generate method call and assign return value
                            method_call = codegen_funcs['method'](value_node, self)
                            self.writeln_main(f"{var_name} = {method_call};")
                            return

        value_cpp = self.visit(value_node)
        value_to_infer = value_cpp if isinstance(value_cpp, ast.AST) else value_node

        # Case 1: Variable is already in scope
        if var_name in self.scope_vars:
            existing_type = self.scope_vars[var_name]
            new_value_type = self.get_type(value_to_infer)
            if existing_type != new_value_type:
                self.writeln_main(f"{var_name} = static_cast<{self._lltype_to_cpp(existing_type)}>({value_cpp});")
            else:
                self.writeln_main(f"{var_name} = {value_cpp};")
        # Case 2: Variable is new
        else:
            # For struct stub method calls (e.g., tbuf.Get<T>(), tensor[offset].ReinterpretCast<U>()),
            # use C++ auto type deduction since the return type is a complex template type.
            use_auto = False
            if self._is_struct_stub_method_call(value_node):
                try:
                    self.get_type(value_to_infer)
                except Exception:
                    use_auto = True
                else:
                    use_auto = True

            if use_auto:
                self.writeln_main(f"auto {var_name} = {value_cpp};")
                self.scope_vars[var_name] = ll.int32  # Placeholder
            else:
                var_lltype = self.get_type(value_to_infer)
                var_cpp_type = self._lltype_to_cpp(var_lltype)
                decl = self._maybe_static_constexpr_decl(var_name, var_cpp_type, value_cpp)
                self.writeln_main(decl)
                self.scope_vars[var_name] = var_lltype

    def _resolve_subscript_element_type(self, target: ast.Subscript):
        """Best-effort element LLType of a subscript store target.

        Mirrors the inference-side rule: tensor subscripts expose
        ``element_type`` and pointer subscripts expose ``inner_type``.
        Returns None when the base type cannot be resolved (emitters
        constructed without all_types, pass-rebuilt nodes, etc.), in which
        case the caller keeps the legacy unchecked emission.
        """
        base_lltype = self._resolve_expr_lltype(target.value)
        if not isinstance(base_lltype, LLType):
            return None
        if base_lltype.is_tensor():
            return base_lltype.element_type
        if base_lltype.is_pointer():
            return base_lltype.inner_type
        return None

    def _handle_subscript_assignment(self, node: ast.Assign, target: ast.Subscript, value_node: ast.AST):
        """Handle subscript assignment (e.g., latency[0] = value).

        The typed store validation (TypeInferencer visit_Assign) accepts
        C++-style promotion of the value onto the element type, so e.g. a
        float32 literal stored into a bfloat16 tensor reaches codegen still
        typed float32.  The emitter hook ``_convert_value_for_store``
        legalizes such mismatches (CUDA lowers 16-bit-float target stores to
        the RNE conversion intrinsics, which remain valid under the torch
        extension no-conversion macros); the default hook is a no-op and
        preserves the historical emission for other backends and for all
        already-matching stores (including explicit ll.float_to_bf16 values).
        """
        lhs_str = self.visit(target)
        rhs_str = self.visit(value_node)
        dst_lltype = self._resolve_subscript_element_type(target)
        src_lltype = self._resolve_expr_lltype(value_node)
        rhs_str = self._convert_value_for_store(rhs_str, src_lltype, dst_lltype)
        self.writeln_main(f"{lhs_str} = {rhs_str};")

    def _handle_tuple_assignment(self, node: ast.Assign, target: ast.Tuple, value_node: ast.AST):
        """Handle tuple assignment (e.g., (x, y) = (5, 3.14))."""
        # Check if this is a struct stub method call returning tuple
        if isinstance(value_node, ast.Call) and isinstance(value_node.func, ast.Attribute):
            method_name = value_node.func.attr
            if isinstance(value_node.func.value, ast.Name):
                var_name = value_node.func.value.id
                if var_name in self.scope_vars:
                    # Try to find struct stub method info
                    for stub_class_name, stub_info in get_all_struct_stubs().items():
                        if method_name in stub_info.methods:
                            method_info = stub_info.methods[method_name]
                            if method_info.is_tuple_return and method_info.tuple_return_types:
                                self._handle_struct_stub_tuple_unpack(target, value_node, var_name, method_info)
                                return

        # Check if this is a special struct method call returning tuple
        if isinstance(value_node, ast.Call) and hasattr(value_node, '_is_special_struct_method'):
            struct_name = getattr(value_node, '_special_struct_name', None)
            if struct_name:
                codegen_funcs = get_special_struct_codegen(struct_name)
                if codegen_funcs and 'tuple_unpack' in codegen_funcs:
                    codegen_funcs['tuple_unpack'](target, value_node, self)
                    return

        # Check if this is a builtin/device-function call returning a TupleType
        # (e.g. a @simple_builtin with -> ll.Tuple[...] annotation). Unpack via
        # a held temp + cuda::std::get for both fresh and pre-declared names.
        if isinstance(value_node, ast.Call):
            call_tuple_type = None
            try:
                candidate = self.get_type(value_node)
                if isinstance(candidate, TupleType):
                    call_tuple_type = candidate
            except Exception:
                call_tuple_type = None
            if call_tuple_type is not None:
                self._handle_tuple_call_unpack(target, value_node, call_tuple_type)
                return

        # Validate tuple length matches value length
        if not isinstance(value_node, ast.Tuple) or len(target.elts) != len(value_node.elts):
            raise ValueError(f"Tuple unpacking mismatch: target has {len(target.elts)} elements, "
                             f"value has {len(value_node.elts) if isinstance(value_node, ast.Tuple) else 1} elements")

        # Process each element in the tuple
        for target_elt, value_elt in zip(target.elts, value_node.elts):
            if not isinstance(target_elt, ast.Name):
                raise NotImplementedError(
                    f"Only Name elements are supported in tuple unpacking (got {type(target_elt).__name__})")
            elt_name = target_elt.id
            elt_cpp = self.visit(value_elt)
            elt_to_infer = elt_cpp if isinstance(elt_cpp, ast.AST) else value_elt

            # Case 1: Element is in scope
            if elt_name in self.scope_vars:
                existing_type = self.scope_vars[elt_name]
                new_elt_type = self.get_type(elt_to_infer)
                if existing_type != new_elt_type:
                    self.writeln_main(f"{elt_name} = static_cast<{self._lltype_to_cpp(existing_type)}>({elt_cpp});")
                else:
                    self.writeln_main(f"{elt_name} = {elt_cpp};")
            # Case 2: Element is new
            else:
                elt_lltype = self.get_type(elt_to_infer)
                elt_cpp_type = self._lltype_to_cpp(elt_lltype)
                self.writeln_main(f"{elt_cpp_type} {elt_name} = {elt_cpp};")
                self.scope_vars[elt_name] = elt_lltype

    def _handle_tuple_call_unpack(self, target: ast.Tuple, value_node: ast.Call, tuple_type: TupleType):
        """Unpack a TupleType-returning call (builtin/device function).

        Always routes through a held temporary so fresh and pre-declared
        target names share one lowering:

            a, b = f(x)  ->  auto _lk_tup_N = f(x);
                             <T_a> a = cuda::std::get<0>(_lk_tup_N);
                             b = cuda::std::get<1>(_lk_tup_N);   (existing b)

        C++17 structured bindings are not used: libcudacxx keeps its tuple in
        cuda::std and does not specialize std::tuple_size, so `auto [a, b]`
        does not compile for cuda::std::tuple.
        """
        element_types = list(tuple_type.element_types)
        if len(target.elts) != len(element_types):
            raise ValueError(f"Tuple unpacking mismatch: call returns {len(element_types)} values, "
                             f"got {len(target.elts)} targets")
        for target_elt in target.elts:
            if not isinstance(target_elt, ast.Name):
                raise NotImplementedError(
                    f"Only Name elements are supported in tuple unpacking (got {type(target_elt).__name__})")

        call_cpp = self.visit(value_node)

        self._tuple_temp_counter = getattr(self, "_tuple_temp_counter", 0) + 1
        temp_name = f"_lk_tup_{self._tuple_temp_counter}"
        self.writeln_main(f"auto {temp_name} = {call_cpp};")
        for i, (target_elt, elt_type) in enumerate(zip(target.elts, element_types)):
            elt_name = target_elt.id
            elt_cpp = f"{self.tuple_namespace}::get<{i}>({temp_name})"
            if elt_name in self.scope_vars:
                existing_type = self.scope_vars[elt_name]
                if existing_type != elt_type:
                    self.writeln_main(f"{elt_name} = static_cast<{self._lltype_to_cpp(existing_type)}>({elt_cpp});")
                else:
                    self.writeln_main(f"{elt_name} = {elt_cpp};")
            else:
                self.writeln_main(f"{self._lltype_to_cpp(elt_type)} {elt_name} = {elt_cpp};")
                self.scope_vars[elt_name] = elt_type

    def _handle_struct_stub_tuple_unpack(self, target: ast.Tuple, value_node: ast.Call, var_name: str, method_info):
        """Handle tuple unpacking for struct stub method calls."""
        target_names = [t.id if isinstance(t, ast.Name) else None for t in target.elts]
        if len(target_names) != len(method_info.tuple_return_types):
            raise ValueError(f"Tuple unpacking mismatch: expected {len(method_info.tuple_return_types)} values, "
                             f"got {len(target_names)} targets")

        # Generate variable declarations
        for i, (target_name, ret_type) in enumerate(zip(target_names, method_info.tuple_return_types)):
            if target_name:
                if ret_type:
                    cpp_type = self._lltype_to_cpp(ret_type)
                else:
                    cpp_type = "int32_t"  # Default to signed integer
                self.writeln_main(f"{cpp_type} {target_name};")
                self.scope_vars[target_name] = ret_type or ll.int32

        # Generate method call with reference parameters
        cpp_method_name = method_info.cpp_method_name or value_node.func.attr
        args_cpp = [self.visit(arg) for arg in value_node.args]
        all_args = args_cpp + target_names
        self.writeln_main(f"{var_name}.{cpp_method_name}({', '.join(all_args)});")

    def visit_Expr(self, node: ast.Expr) -> None:
        """Process expression statement (add C++ semicolon).

        Skips bare string literals (Python docstrings) since they have no
        C++ equivalent.
        """
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        expr_cpp = self.visit(node.value)
        self.writeln_main(f"{expr_cpp};")

    def visit_Return(self, node: ast.Return) -> None:
        """Process return statement."""
        if node.value is None:
            self.writeln_main("return;")
        else:
            value_cpp = self.visit(node.value)
            self.writeln_main(f"return {value_cpp};")

    def visit_Assert(self, node: ast.Assert) -> None:
        """Process Python assert statements → C++ assert."""
        test_cpp = self.visit(node.test)
        if node.msg is not None:
            msg_cpp = self.visit(node.msg)
            self.writeln_main(f"assert({test_cpp} && {msg_cpp});")
        else:
            self.writeln_main(f"assert({test_cpp});")

    def visit_CppArrayDecl(self, node) -> None:
        """Process C++ array declaration (from IR node)."""
        var_name = node.var_name
        element_type_cpp = node.element_type_cpp
        array_size = node.array_size
        self.writeln_main(f"{element_type_cpp} {var_name}[{array_size}];")
        self.scope_vars[var_name] = ll.int32  # Placeholder

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Process augmented assignment (e.g., x += 1, x -= 1, x %= n)."""
        target_cpp = self.visit(node.target)
        from little_kernel.codegen.registries.operator_codegen import get_operator_codegen_registry
        registry = get_operator_codegen_registry()
        op_cpp = registry.get_bin_op(node.op)
        value_cpp = self.visit(node.value)
        self.writeln_main(f"{target_cpp} {op_cpp}= {value_cpp};")

    def visit_Pass(self, node: ast.Pass) -> None:
        """Process Python 'pass' statement (no-op)."""
        # Transitional compatibility: role/dataflow lowering used to overload
        # ast.Pass with ``_lk_cpp_*`` attributes. Route it through the explicit
        # typed escape-hatch path; new lowering constructs lk_ir.Target*
        # nodes directly and never reaches this branch.
        if lk_ir.is_legacy_cpp_node(node):
            upgraded = lk_ir.upgrade_legacy_node(node)
            return self.visit(upgraded)
        # Pass statements are no-ops, so we don't emit anything
        pass

    def _emit_verified_lowered(self, node) -> None:
        """Emit a typed lowered statement after checking its declared contract."""
        lk_ir.verify_lowered_node(node)

    def visit_TargetStatement(self, node) -> None:
        """Emit an explicitly typed backend statement escape hatch."""
        self._emit_verified_lowered(node)
        if node.section == "builtin":
            self.writeln_builtin(node.spelling)
        else:
            self.writeln_main(node.spelling)

    def visit_TargetInclude(self, node) -> None:
        """Emit a backend #include directive, deduplicated by spelling."""
        self._emit_verified_lowered(node)
        if node.include not in self.header_cache:
            self.writeln_header(f"#include {node.include}")
            self.header_cache.add(node.include)

    def visit_TargetHelper(self, node) -> None:
        """Emit a backend helper/device-function definition."""
        self._emit_verified_lowered(node)
        if node.name is not None and node.name in self.builtin_body_cache:
            return
        # writeln_builtin extracts the function name and deduplicates by it,
        # so a named helper is emitted at most once per emitter.
        self.writeln_builtin(node.spelling)
