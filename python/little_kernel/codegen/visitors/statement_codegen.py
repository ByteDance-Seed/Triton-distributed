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
from little_kernel.core.type_system import (LLType, TupleType, ConstTypeHelper, ConstAnnotationType, const_annotation,
                                            Const)
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.passes.utils.ir_nodes import EnumDef, SpecialStructDef, is_ir_node
from little_kernel.core.passes.utils.error_report import format_error_message
from little_kernel.language.intrin.struct_stub import get_struct_stub_info, get_all_struct_stubs
from ..registries.enum_registry import generate_enum_cpp_definition
from ..registries.special_struct_registry import (is_special_struct, get_special_struct_name,
                                                  get_special_struct_codegen, get_all_registered_special_structs)

if TYPE_CHECKING:
    pass


class StatementCodegenMixin:
    """Mixin class for statement codegen methods."""

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

        # Process function body (statements like Assign, Return)
        for stmt in node.body:
            self.visit(stmt)

        # Cleanup: close function and reset scope
        self.indent_level -= 1
        self.writeln_main("}")
        self.writeln_main("")
        self.scope_vars = old_scope_vars  # Restore scope after function

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Process annotated assignments (e.g., "x: const = 5" or "x: int32 = 5")."""
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"Annotated assignment target must be a single variable, got {type(node.target).__name__}")

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
                value_cpp = self.visit(node.value)
                if var_name in self.scope_vars:
                    existing_type = self.scope_vars[var_name]
                    if existing_type != target_type:
                        self.writeln_main(f"{var_name} = static_cast<{target_cpp_type}>({value_cpp});")
                    else:
                        self.writeln_main(f"{var_name} = {value_cpp};")
                else:
                    self.writeln_main(f"{target_cpp_type} {var_name} = {value_cpp};")
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
        # Handle Subscript target (array/tensor element assignment, e.g., array[idx] = value)
        elif isinstance(target, ast.Subscript):
            self._handle_subscript_assignment(node, target, value_node)
        else:
            message = f"Unsupported assignment target type: {type(target).__name__}"
            formatted = format_error_message(message, target, ctx=self.ctx)
            raise NotImplementedError(formatted)

    def _handle_struct_stub_constructor_assignment(self, node: ast.Assign, target: ast.AST, value_node: ast.Call,
                                                   stub_info):
        """Handle struct stub constructor assignment."""
        target_name = target.id if isinstance(target, ast.Name) else None
        if target_name is None:
            raise NotImplementedError("Struct stub assignment must have a simple variable name as target")

        # Generate constructor arguments
        args_cpp = [self.visit(arg) for arg in value_node.args]
        kwargs_cpp = {}
        for kw in value_node.keywords:
            if kw.arg:
                kwargs_cpp[kw.arg] = self.visit(kw.value)

        # Build constructor call
        cpp_struct_name = stub_info.cpp_struct_name
        if stub_info.namespace:
            cpp_struct_name = f"{stub_info.namespace}::{cpp_struct_name}"

        # Generate constructor call
        if kwargs_cpp:
            all_args = args_cpp + [f"{k}={v}" for k, v in kwargs_cpp.items()]
            cpp_code = f"{cpp_struct_name} {target_name}({', '.join(all_args)});"
        else:
            cpp_code = f"{cpp_struct_name} {target_name}({', '.join(args_cpp)});"

        self.writeln_main(cpp_code)

        # Add includes
        for include in stub_info.includes:
            if include not in self.header_cache:
                self.writeln_header(f"#include {include}")
                self.header_cache.add(include)

        # Register variable in scope with correct struct type for type verification
        # in method call codegen (enables matching var to correct struct stub)
        from little_kernel.core.type_system import SpecialStructType
        func_name = value_node.func.id
        self.scope_vars[target_name] = SpecialStructType(func_name)

    def _handle_single_name_assignment(self, node: ast.Assign, target: ast.Name, value_node: ast.AST):
        """Handle single name assignment (e.g., x = 5)."""
        var_name = target.id

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
            var_lltype = self.get_type(value_to_infer)
            var_cpp_type = self._lltype_to_cpp(var_lltype)
            # Check if we're inside a loop - if so, declare variable outside the loop
            # by checking if we're at a higher indent level (indicating nested scope)
            # Actually, simpler: just declare it here, but mark it as needing hoisting
            # For now, declare it here - the issue is that in C++, variables declared
            # in a for loop body are scoped to that loop body, but in Python they're
            # visible outside. We need to handle this by checking if the variable
            # is used outside the loop and hoisting the declaration.
            # For simplicity, we'll declare it here and let the scope management handle it.
            # But the real issue is that we need to declare it before the loop starts.
            # Let's use a simpler approach: always declare variables at the function level
            # when they're first assigned, regardless of where the assignment happens.
            # This matches Python's scoping behavior.
            self.writeln_main(f"{var_cpp_type} {var_name} = {value_cpp};")
            self.scope_vars[var_name] = var_lltype

    def _handle_subscript_assignment(self, node: ast.Assign, target: ast.Subscript, value_node: ast.AST):
        """Handle subscript assignment (e.g., latency[0] = value)."""
        # Generate the left-hand side (subscript expression)
        # visit_Subscript handles both Load and Store contexts
        lhs_str = self.visit(target)

        # Generate the right-hand side (value expression)
        rhs_str = self.visit(value_node)

        # Generate C++ assignment statement
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
        """Process expression statement (add C++ semicolon)."""
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

    def visit_Pass(self, node: ast.Pass) -> None:
        """Process Python 'pass' statement (no-op)."""
        # Pass statements are no-ops, so we don't emit anything
        pass

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Process augmented assignment (e.g., x += 1, y -= 2)."""
        target_cpp = self.visit(node.target)
        value_cpp = self.visit(node.value)

        # Map Python AST operators to C++ augmented assignment operators
        op_map = {
            ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=", ast.Mod: "%=", ast.Pow:
            "**=",  # Note: C++ doesn't have **=, but we'll handle it
            ast.LShift: "<<=", ast.RShift: ">>=", ast.BitAnd: "&=", ast.BitOr: "|=", ast.BitXor: "^=", ast.FloorDiv:
            "/=",  # C++ uses / for integer division
        }

        op_type = type(node.op)
        if op_type not in op_map:
            raise NotImplementedError(f"Unsupported augmented assignment operator: {op_type.__name__}")

        op_str = op_map[op_type]

        # Special handling for **= (power assignment) - C++ doesn't have this
        if op_type == ast.Pow:
            # Convert x **= y to x = pow(x, y) or use std::pow
            self.writeln_main(f"{target_cpp} = std::pow({target_cpp}, {value_cpp});")
        else:
            self.writeln_main(f"{target_cpp} {op_str} {value_cpp};")
