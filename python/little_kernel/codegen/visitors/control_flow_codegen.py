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
from typing import TYPE_CHECKING, Optional
import little_kernel.language as ll
from little_kernel.core.type_system import LLType, TupleType, ConstTypeHelper, ConstAnnotationType, const_annotation, Const
from little_kernel.core.passes.utils.registries.loop_modifier_registry import get_loop_modifier_registry
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
from little_kernel.codegen.registries.loop_modifier_codegen import get_loop_modifier_codegen

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
        for var_name in sorted(all_vars):
            if var_name not in self.scope_vars:
                # Use type from if block if available, else from else block
                if var_name in vars_in_if:
                    vars_to_declare_before_if[var_name] = vars_in_if[var_name]
                elif var_name in vars_in_else:
                    vars_to_declare_before_if[var_name] = vars_in_else[var_name]

        # Declare variables before if statement (skip void/SpecialStruct types)
        from little_kernel.core.type_system import VoidType, SpecialStructType
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare_before_if.items():
            if isinstance(var_lltype, (VoidType, SpecialStructType)):
                continue
            if var_cpp_type == "void":
                continue
            self.writeln_main(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        # Use helper method to process if statement
        self._visit_if_body(node, is_else_if=False)

    def visit_While(self, node: ast.While) -> None:
        """Process Python while loops → C++ while loop."""
        # Hoist loop-body variables before the loop, mirroring visit_For:
        # Python variables assigned in the body are function-scoped, but a C++
        # declaration emitted on the body's first assignment would be scoped
        # to the loop's block. scope_vars otherwise still records the name,
        # so a later use in a sibling loop body or after the loop emits a
        # bare assignment to an undeclared C identifier
        # (emitted_cuda_undefined_identifier).
        vars_to_declare = self._scan_for_assigned_vars(node.body)

        # Declare variables before the loop (skip void/SpecialStruct types)
        from little_kernel.core.type_system import VoidType, SpecialStructType
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare.items():
            if isinstance(var_lltype, (VoidType, SpecialStructType)):
                continue
            if var_cpp_type == "void":
                continue
            self.writeln_main(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        # Generate condition expression
        condition_cpp = self.visit(node.test)
        self.writeln_main(f"while ({condition_cpp}) {{")

        # Process statements in while block
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.writeln_main("}")  # Close while block

        # Handle orelse clause
        if node.orelse:
            raise NotImplementedError("While loop 'else' clause is not supported in C++ (Python-specific feature). "
                                      "A C++ while loop has no else; the previous emission produced invalid C++. "
                                      "Track loop completion with an explicit flag variable instead.")

    # Upper bound on per-iteration copies emitted for a statically unrolled
    # loop; larger literal loops keep the ``#pragma unroll`` hint instead of
    # exploding the generated source.
    _STATIC_UNROLL_MAX_TRIP = 256

    @staticmethod
    def _const_range_int(arg) -> Optional[int]:
        """Resolve a range() argument AST node to a Python int literal, or None."""
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int) and not isinstance(arg.value, bool):
            return arg.value
        if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
            inner = ControlFlowCodegenMixin._const_range_int(arg.operand)
            return -inner if inner is not None else None
        return None

    @classmethod
    def _crosses_iteration(cls, stmt) -> bool:
        """True if stmt contains a break/continue binding to the loop being unrolled.

        A ``break``/``continue`` that is a direct child of ``stmt`` (top-level
        statement of the loop body) binds the loop being considered and blocks
        full unrolling: outside a real C++ loop these keywords are ill-formed.
        break/continue inside a nested for/while bind to that inner loop and
        are safe inside an unrolled copy, so nested loop subtrees are not
        descended.  ``return`` needs no check: it exits the kernel identically
        from a copy.
        """
        if isinstance(stmt, (ast.Break, ast.Continue)):
            return True
        if isinstance(stmt, (ast.For, ast.While)):
            # break/continue inside this nested loop bind to it, not to the
            # loop being considered for unrolling, so never descend.
            return False
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, (ast.For, ast.While)):
                continue
            if isinstance(child, ast.stmt) and cls._crosses_iteration(child):
                return True
        return False

    @staticmethod
    def _range_trip_count(start: int, stop: int, step: int):
        """Trip count of range(start, stop, step) without materializing it.

        Empty ranges return 0. Pure arithmetic over Python ints bounds the
        result before any enumeration, so a source range far larger than
        memory (or even larger than Py_ssize_t) cannot trigger a giant
        allocation or an OverflowError inside ``list(range(...))``.
        """
        if step > 0:
            if stop <= start:
                return 0
            span = stop - start
        else:
            if stop >= start:
                return 0
            span = start - stop
        return (span + abs(step) - 1) // abs(step)

    def _static_unroll_values(self, node: ast.For) -> Optional[list]:
        """Iteration values for a guaranteed full unroll of ``node``, or None.

        Eligible loops have the form ``for v in ll.unroll(range(...))`` where
        every range argument is an integer literal (constant trip count) and no
        break/continue crosses an iteration boundary (same legality class as
        LLVM LoopUnroll's full-unroll: compile-time trip count, no exiting
        edges through the header). Ineligible loops return None and keep the
        ``#pragma unroll`` hint path unchanged.
        """
        range_call = node.iter
        if not (isinstance(range_call, ast.Call) and isinstance(range_call.func, ast.Name)
                and range_call.func.id == "range"):
            return None
        const_args = [self._const_range_int(a) for a in range_call.args]
        if not const_args or any(v is None for v in const_args):
            return None
        if len(const_args) == 1:
            start, stop, step = 0, const_args[0], 1
        elif len(const_args) == 2:
            start, stop, step = const_args[0], const_args[1], 1
        else:
            start, stop, step = const_args
        if step == 0:
            return None
        # Bound the trip count with integer arithmetic before any enumeration:
        # materializing list(range(...)) first lets a large but otherwise
        # valid literal loop exhaust host memory (and ranges past Py_ssize_t
        # raise OverflowError from list(range(...)) outright).
        trip_count = self._range_trip_count(start, stop, step)
        if trip_count is None or trip_count > self._STATIC_UNROLL_MAX_TRIP:
            return None
        for stmt in node.body:
            if self._crosses_iteration(stmt):
                return None
        # Small legal loops still fully enumerate (and unroll) every copy.
        # Guard range() against Py_ssize_t overflow even though the cap above
        # already excludes such trips in practice.
        try:
            return list(range(start, stop, step))
        except OverflowError:
            return None

    def _resolve_int_bound_type(self, node, depth=0):
        """Best-effort integer type of a range() bound expression, or None.

        Layered resolution: the type-inference map (all_types) when the
        inferencer recorded the node, the codegen scope map (scope_vars) for
        plain Name loads, a structural walk of BinOp/UnaryOp nests with C++
        promotion across operands (so `hi - 1` inherits the uint32 of `hi`
        rather than collapsing to the literal's int32), and a Subscript load
        from a declared tensor that resolves to the tensor's element integer
        type (covering ragged/segment offsets like ``offsets[e]``). Anything
        else returns None and the caller falls back to int32.
        """
        from little_kernel.core.type_system import IntType
        from little_kernel.core.passes.utils.type_inference.type_inference_visitors import promote_type

        if depth > 8:
            return None
        candidate = None
        all_types = getattr(self, "all_types", None)
        if all_types is not None:
            candidate = all_types.get(node)
        if candidate is None and isinstance(node, ast.Name):
            candidate = self.scope_vars.get(node.id)
        if isinstance(candidate, IntType):
            if candidate.special is not None:
                return ll.int32
            # integer promotion: keep at least int32 width
            return candidate if candidate.bits >= 32 else ll.int32
        if candidate is not None:
            return None  # known but not a plain integer -> caller default
        if isinstance(node, ast.BinOp):
            left = self._resolve_int_bound_type(node.left, depth + 1)
            right = self._resolve_int_bound_type(node.right, depth + 1)
            left = left if left is not None else ll.int32
            right = right if right is not None else ll.int32
            merged, _, _ = promote_type(left, right)
            if merged.special is not None:
                return ll.int32
            return merged
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            return self._resolve_int_bound_type(node.operand, depth + 1)
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            # Subscript load from a declared tensor (e.g. ragged segment
            # offsets[e]). When type inference has not recorded a type for the
            # subscript itself, resolve through the tensor's declared element
            # type so the loop variable inherits the loaded integer width
            # (ll.Tensor[ll.int64] -> int64_t loop counter). This matters for
            # ragged-loop lowering where S > 2**31 would otherwise truncate.
            tensor_type = self.scope_vars.get(node.value.id)
            if tensor_type is None:
                all_types = getattr(self, "all_types", None)
                if all_types is not None:
                    tensor_type = all_types.get(node.value)
            from little_kernel.core.type_system import TensorType
            if isinstance(tensor_type, TensorType):
                element_type = tensor_type.element_type
                if isinstance(element_type, IntType):
                    if element_type.special is not None:
                        return ll.int32
                    return element_type if element_type.bits >= 32 else ll.int32
            return None
        return None

    def _promote_range_bound_types(self, range_args):
        """Promoted integer type for a fresh range() loop variable.

        Folds the resolved types of the range() bound expressions with the
        same C++ implicit-conversion promotion used by type inference
        (promote_type). Bounds that resolve to nothing usable contribute
        int32, so a plain int32/int-literal loop keeps the historical
        int32_t counter.
        """
        from little_kernel.core.passes.utils.type_inference.type_inference_visitors import promote_type

        result_type = ll.int32
        for arg in range_args:
            arg_type = self._resolve_int_bound_type(arg)
            if arg_type is None:
                arg_type = ll.int32
            result_type, _, _ = promote_type(result_type, arg_type)
            if result_type.special is not None:
                result_type = ll.int32
        return result_type

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

        # Parse range arguments: Python range(start, stop, step) or range(stop)
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
            # Pick the loop variable type from the range() bound expressions
            # instead of hardcoding int32. Ragged/irregular kernels iterate
            # with bounds loaded from typed tensors (e.g. uint32 segment
            # offsets): an int32 loop variable against a uint32 bound makes
            # the emitted condition a mixed-sign C++ comparison, which under
            # usual arithmetic conversions reinterprets the (signed) counter
            # as unsigned -- a descending scan then runs past the segment.
            # Promote across start/stop/step with the same C++ conversion
            # rules used by type inference, defaulting to int32 when the
            # bounds carry no wider/unsigned type (keeps prior behavior for
            # plain int32 loops).
            loop_var_type = self._promote_range_bound_types(range_call.args)
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

        # Declare variables before the loop (skip void/SpecialStruct types)
        from little_kernel.core.type_system import VoidType, SpecialStructType
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare.items():
            if isinstance(var_lltype, (VoidType, SpecialStructType)):
                continue
            if var_cpp_type == "void":
                continue
            self.writeln_main(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        # Guaranteed source-level full unroll: ll.unroll(range(...)) with all
        # integer-literal bounds and no break/continue crossing iterations is
        # emitted as per-iteration scoped copies. The `#pragma unroll` form is
        # only a hint that nvcc may ignore (e.g. constant-trip head-split loops
        # with pointer-select bodies), which measurably defeats hand schedules;
        # duplication guarantees the unrolled schedule for any nvcc heuristic.
        static_unroll_values = None
        if modifier_code == "#pragma unroll" and not loop_var_in_scope:
            static_unroll_values = self._static_unroll_values(node)

        if static_unroll_values is not None:
            # Each iteration gets its own scope so body-local declarations do
            # not collide; the loop variable is re-declared per copy with the
            # literal trip value, which nvcc constant-propagates into the body.
            # The same literal binding is published to expression codegen for
            # the duration of the copy so a compile-time-unrolled fragment
            # subscript lowers to the per-element register scalar instead of
            # an address-taken array slot (ir.wgmma.acc_fragment).
            unroll_envs = getattr(self, "_unroll_const_envs", None)
            for iter_value in static_unroll_values:
                self.writeln_main("{")
                self.indent_level += 1
                self.writeln_main(f"{loop_var_cpp_type} {loop_var_name} = {iter_value};")
                if unroll_envs is not None:
                    unroll_envs.append({loop_var_name: iter_value})
                try:
                    for stmt in node.body:
                        self.visit(stmt)
                finally:
                    if unroll_envs is not None:
                        unroll_envs.pop()
                self.indent_level -= 1
                self.writeln_main("}")
        else:
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
