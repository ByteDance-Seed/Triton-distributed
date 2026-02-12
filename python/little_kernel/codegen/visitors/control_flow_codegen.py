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
Control flow codegen visitors for CppEmitter.

This module contains visitor methods for control flow statements (if, while, for, etc.).
"""

import ast
from typing import TYPE_CHECKING
import little_kernel.language as ll
from little_kernel.core.type_system import (LLType, TupleType, ConstTypeHelper, ConstAnnotationType, const_annotation,
                                            Const)
from little_kernel.core.passes.utils.registries.loop_modifier_registry import get_loop_modifier_registry
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
from ..registries.loop_modifier_codegen import get_loop_modifier_codegen

if TYPE_CHECKING:
    pass


class ControlFlowCodegenMixin:
    """Mixin class for control flow codegen methods."""

    def _scan_for_assigned_vars(self, stmts, current_scope_vars=None):
        """Recursively scan statements for variables assigned anywhere (including nested For/If/While).
        Returns dict of var_name -> (var_cpp_type, var_lltype) for variables not yet in scope.
        Used for variable hoisting: Python variables are visible outside blocks, C++ requires declaration.
        """
        if current_scope_vars is None:
            current_scope_vars = self.scope_vars.copy()
        vars_found = {}
        for stmt in stmts:
            if isinstance(stmt, ast.Assign):
                # Handle single name assignment
                if len(stmt.targets) == 1:
                    target = stmt.targets[0]
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if var_name not in current_scope_vars:
                            try:
                                value_type = self.get_type(stmt.value)
                                var_cpp_type = self._lltype_to_cpp(value_type)
                                vars_found[var_name] = (var_cpp_type, value_type)
                            except Exception:
                                vars_found[var_name] = ("int32_t", ll.int32)
                    elif isinstance(target, ast.Tuple):
                        # Handle tuple unpacking: a, b = ...
                        return_type = None
                        if isinstance(stmt.value, ast.Call):
                            if hasattr(stmt.value, '_is_special_struct_method'):
                                struct_name = getattr(stmt.value, '_special_struct_name', None)
                                method_name = getattr(stmt.value, '_method_name', None)
                                if struct_name and method_name:
                                    codegen_info = get_special_struct_codegen(struct_name)
                                    if codegen_info and 'class' in codegen_info and codegen_info['class'] is not None:
                                        class_obj = codegen_info['class']
                                        if hasattr(class_obj, method_name):
                                            method = getattr(class_obj, method_name)
                                            if hasattr(method,
                                                       '__annotations__') and 'return' in method.__annotations__:
                                                inferencer = TypeInferencer(self.ctx if hasattr(self, 'ctx') else {},
                                                                            {})
                                                try:
                                                    return_type = inferencer._resolve_annotation(
                                                        ast.parse(f"x: {method.__annotations__['return']}",
                                                                  mode='eval').body)
                                                except Exception:
                                                    pass
                            elif hasattr(self, 'all_types') and stmt.value in self.all_types:
                                return_type = self.all_types[stmt.value]

                        target_names = [elt.id for elt in target.elts if isinstance(elt, ast.Name)]

                        skip_first = False
                        if isinstance(return_type, TupleType):
                            if (len(return_type.element_types) > len(target_names)
                                    and len(return_type.element_types) > 0):
                                first_elem_type = return_type.element_types[0]
                                if hasattr(first_elem_type, 'kind') and first_elem_type.kind == 'bool':
                                    skip_first = True
                        elif hasattr(self, 'all_types') and stmt.value in self.all_types:
                            call_return_type = self.all_types[stmt.value]
                            if isinstance(call_return_type, TupleType):
                                if (len(call_return_type.element_types) > len(target_names)
                                        and len(call_return_type.element_types) > 0):
                                    first_elem_type = call_return_type.element_types[0]
                                    if hasattr(first_elem_type, 'kind') and first_elem_type.kind == 'bool':
                                        skip_first = True

                        for idx, var_name in enumerate(target_names):
                            if var_name not in current_scope_vars:
                                var_cpp_type = "int32_t"
                                var_lltype = ll.int32
                                actual_idx = idx + (1 if skip_first else 0)

                                if isinstance(return_type, TupleType) and len(return_type.element_types) > actual_idx:
                                    element_type = return_type.element_types[actual_idx]
                                    var_cpp_type = self._lltype_to_cpp(element_type)
                                    var_lltype = element_type
                                elif hasattr(self, 'all_types') and stmt.value in self.all_types:
                                    call_return_type = self.all_types[stmt.value]
                                    if isinstance(call_return_type, TupleType) and len(
                                            call_return_type.element_types) > actual_idx:
                                        element_type = call_return_type.element_types[actual_idx]
                                        var_cpp_type = self._lltype_to_cpp(element_type)
                                        var_lltype = element_type

                                vars_found[var_name] = (var_cpp_type, var_lltype)
            elif isinstance(stmt, ast.AnnAssign):
                if isinstance(stmt.target, ast.Name):
                    var_name = stmt.target.id
                    if var_name not in current_scope_vars:
                        is_const_annotation = False
                        const_inner_type = None
                        try:
                            resolved_ann = recursive_resolve_attribute(stmt.annotation, self.ctx)
                            if isinstance(resolved_ann, Const):
                                is_const_annotation = True
                                const_inner_type = resolved_ann.inner_type
                            elif (isinstance(resolved_ann, ConstTypeHelper)
                                  or isinstance(resolved_ann, ConstAnnotationType) or resolved_ann == const_annotation):
                                is_const_annotation = True
                                const_inner_type = None
                        except Exception:
                            if isinstance(stmt.annotation, ast.Name) and stmt.annotation.id == "const":
                                is_const_annotation = True
                                const_inner_type = None

                        if is_const_annotation:
                            if const_inner_type is not None:
                                target_type = const_inner_type
                            else:
                                target_type = self.get_type(stmt.value) if stmt.value else ll.int32
                        else:
                            try:
                                resolved_ann = recursive_resolve_attribute(stmt.annotation, self.ctx)
                                if isinstance(resolved_ann, LLType):
                                    target_type = resolved_ann
                                else:
                                    if hasattr(self, 'all_types') and stmt.annotation in self.all_types:
                                        target_type = self.get_type(stmt.annotation)
                                    else:
                                        target_type = self.get_type(stmt.value) if stmt.value else ll.int32
                            except Exception:
                                target_type = self.get_type(stmt.value) if stmt.value else ll.int32

                        var_cpp_type = self._lltype_to_cpp(target_type)
                        vars_found[var_name] = (var_cpp_type, target_type)
            elif isinstance(stmt, ast.For):
                nested_vars = self._scan_for_assigned_vars(stmt.body, current_scope_vars)
                vars_found.update(nested_vars)
            elif isinstance(stmt, ast.If):
                nested_vars = self._scan_for_assigned_vars(stmt.body, current_scope_vars)
                vars_found.update(nested_vars)
                if stmt.orelse:
                    nested_vars_else = self._scan_for_assigned_vars(stmt.orelse, current_scope_vars)
                    vars_found.update(nested_vars_else)
            elif isinstance(stmt, ast.While):
                nested_vars = self._scan_for_assigned_vars(stmt.body, current_scope_vars)
                vars_found.update(nested_vars)
        return vars_found

    def _visit_if_body(self, node: ast.If, is_else_if: bool = False) -> None:
        """Helper method to process if statement body and else block.
        
        Args:
            node: ast.If node to process
            is_else_if: If True, this is part of an "else if" chain and should not write "if" prefix
        """
        if not is_else_if:
            # Generate condition expression
            condition_cpp = self.visit(node.test)
            self.writeln_main(f"if ({condition_cpp}) {{")
        else:
            # For else if, condition is already handled by caller
            condition_cpp = self.visit(node.test)
            self.write_main(f"if ({condition_cpp}) {{")
            self.write_main("\n")

        # Process statements in if block
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.writeln_main("}")  # Close if block

        # Process else block (if exists)
        if node.orelse:
            # Check if else block contains a single if statement (i.e., "else if" case)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # Generate "else if (...)" syntax
                self.write_main(" else ")
                # Recursively process nested if as else if
                self._visit_if_body(node.orelse[0], is_else_if=True)
            else:
                # Generate regular else block
                self.writeln_main("else {")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self.indent_level -= 1
                self.writeln_main("}")  # Close else block

    def visit_If(self, node: ast.If) -> None:
        """Process if-else statements."""

        # Before processing if/else blocks, scan for variables that are assigned
        # in if/else blocks (including loops and tuple unpacking). These need to be
        # declared before the if statement if they're used in both branches.
        vars_in_if = self._scan_for_assigned_vars(node.body)
        vars_in_else = {}
        if node.orelse:
            vars_in_else = self._scan_for_assigned_vars(node.orelse)

        # Variables used in both if and else blocks need to be declared before if
        vars_to_declare_before_if = {}
        all_vars = set(vars_in_if.keys()) | set(vars_in_else.keys())
        for var_name in all_vars:
            if var_name not in self.scope_vars:
                # Use type from if block if available, else from else block
                if var_name in vars_in_if:
                    vars_to_declare_before_if[var_name] = vars_in_if[var_name]
                elif var_name in vars_in_else:
                    vars_to_declare_before_if[var_name] = vars_in_else[var_name]

        # Declare variables before if statement
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare_before_if.items():
            self.writeln_main(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        # Use helper method to process if statement
        self._visit_if_body(node, is_else_if=False)

    def visit_While(self, node: ast.While) -> None:
        """Process Python while loops → C++ while loop."""
        # Generate condition expression
        condition_cpp = self.visit(node.test)
        self.writeln_main(f"while ({condition_cpp}) {{")

        # Process statements in while block
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.writeln_main("}")  # Close while block

        # Process else block (if exists) - Python allows else on while loops
        if node.orelse:
            self.writeln_main("else {")
            self.indent_level += 1
            for stmt in node.orelse:
                self.visit(stmt)
            self.indent_level -= 1
            self.writeln_main("}")  # Close else block

    def visit_For(self, node: ast.For) -> None:
        """Process Python for loops (primarily supports `for var in range(...)` → C++ for loop)."""
        # Detect loop modifiers (e.g., ll.unroll, ll.parallel)
        loop_modifier_registry = get_loop_modifier_registry()
        loop_modifier_codegen = get_loop_modifier_codegen()

        modifier_code = None
        if isinstance(node.iter, ast.Call):
            if loop_modifier_registry.is_modifier_call(node.iter, self.ctx):
                # Get codegen for the modifier (e.g., "#pragma unroll")
                modifier_code = loop_modifier_codegen.process_modifier(self, node)
                # Unwrap the modifier to get the inner iterator
                try:
                    node.iter = loop_modifier_registry.unwrap_modifier(node.iter, self.ctx)
                except ValueError:
                    pass  # If unwrap fails, continue with original iterator

        # Validate loop target
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"For loop target must be a single variable (ast.Name), got {type(node.target).__name__}. "
                "Tuple unpacking (e.g., for (a,b) in ...) is not supported yet.")
        loop_var_name = node.target.id

        # Validate & process iterator (only support `range()` call)
        if not (isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"):
            raise NotImplementedError(f"For loop iterator must be `range()`, got {ast.dump(node.iter, indent=2)}. "
                                      "Other iterables (e.g., lists, tuples) are not supported yet.")
        range_call = node.iter

        # Parse range arguments (start, stop, step) with defaults
        range_args = [self.visit(arg) for arg in range_call.args]
        # range() can be called as:
        # - range(stop) -> start=0, stop=stop, step=1
        # - range(start, stop) -> start=start, stop=stop, step=1
        # - range(start, stop, step) -> start=start, stop=stop, step=step
        if len(range_args) == 1:
            # range(stop)
            stop = range_args[0]
            start = "0"
            step_str = "1"
        elif len(range_args) == 2:
            # range(start, stop)
            start = range_args[0]
            stop = range_args[1]
            step_str = "1"
        else:
            # range(start, stop, step)
            start = range_args[0]
            stop = range_args[1]
            step_str = range_args[2]

        # Try to evaluate step as integer for comparison (for condition generation)
        # step_sign_known: False when step is a variable - need runtime-dependent condition
        step_sign_known = True
        step_int = 1
        if len(range_call.args) >= 3:
            step_arg = range_call.args[2]
            if isinstance(step_arg, ast.UnaryOp) and isinstance(step_arg.op, ast.USub):
                # Negative step: -1, -2, etc.
                operand = step_arg.operand
                if isinstance(operand, (ast.Constant, ast.Num)):
                    step_value = operand.value if isinstance(operand, ast.Constant) else operand.n
                    step_int = -int(step_value)
                    step_str = str(abs(step_value))  # Use absolute value for subtraction
            else:
                # Try to parse step_str as integer constant
                try:
                    step_int = int(step_str) if isinstance(step_str, str) and (
                        step_str.isdigit() or
                        (step_str.startswith('-') and step_str[1:].isdigit())) else int(float(step_str))
                    if step_int < 0:
                        step_str = str(abs(step_int))
                except (ValueError, TypeError):
                    # Step is not a compile-time constant (e.g., variable)
                    step_sign_known = False
                    step_int = 1  # Placeholder; condition will use runtime check

        # Validate step (only for compile-time constants)
        if step_sign_known and step_int == 0:
            raise ValueError("Range step cannot be 0 (infinite loop)")

        # Determine loop variable type & scope
        loop_var_in_scope = False
        if loop_var_name in self.scope_vars:
            loop_var_in_scope = True
            loop_var_type = self.scope_vars[loop_var_name]
            loop_var_cpp_type = self._lltype_to_cpp(loop_var_type)
            loop_init = f"{loop_var_name} = {start}"
        else:
            loop_var_type = ll.int32
            loop_var_cpp_type = self._lltype_to_cpp(loop_var_type)
            loop_init = f"{loop_var_cpp_type} {loop_var_name} = {start}"
            self.scope_vars[loop_var_name] = loop_var_type

        # Generate loop condition
        # When step is dynamic (variable), use runtime-dependent condition to avoid wrong
        # sign assumption (defaulting to step>0 would cause wrong condition if step<0 at runtime)
        if step_sign_known:
            if step_int > 0:
                loop_cond = f"{loop_var_name} < {stop}"
            else:
                loop_cond = f"{loop_var_name} > {stop}"
        else:
            # Runtime check: (step > 0 && i < stop) || (step < 0 && i > stop)
            loop_cond = f"(({step_str}) > 0 ? ({loop_var_name} < {stop}) : ({loop_var_name} > {stop}))"

        # Generate loop increment
        # For dynamic step, always use += (works for both: i += step when step is negative)
        if step_sign_known and step_int < 0:
            loop_incr = f"{loop_var_name} -= {step_str}"
        else:
            loop_incr = f"{loop_var_name} += {step_str}"

        # Before processing loop body, scan for variables assigned in loop body
        # (including nested if/for/while). In Python, these variables are visible outside
        # the loop, but in C++ they're scoped to the loop body. We need to declare
        # them before the loop starts.
        vars_to_declare = self._scan_for_assigned_vars(node.body)
        # Exclude loop variable - it is declared in the for loop header
        vars_to_declare = {k: v for k, v in vars_to_declare.items() if k != loop_var_name}

        # Declare variables before the loop
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare.items():
            self.writeln_main(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        # Write C++ for loop header (with modifier pragma if any)
        if modifier_code:
            self.writeln_main(modifier_code)
        self.writeln_main(f"for ({loop_init}; {loop_cond}; {loop_incr}) {{")

        # Process loop body
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1

        # Close loop block
        self.writeln_main("}")

        # Cleanup: Remove new loop variable from scope
        if not loop_var_in_scope:
            del self.scope_vars[loop_var_name]

        # Handle orelse clause
        if node.orelse:
            raise NotImplementedError("For loop 'orelse' clause is not supported in C++ (Python-specific feature). "
                                      "Remove the 'else' block from the for loop.")

    def visit_Continue(self, node: ast.Continue) -> None:
        """Process Python continue statement → C++ continue statement."""
        self.writeln_main("continue;")

    def visit_Break(self, node: ast.Break) -> None:
        """Process Python break statement → C++ break statement."""
        self.writeln_main("break;")
