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
import little_kernel.language as ll
from little_kernel.core.type_system import (IntType, FloatType, TupleType, Const, ConstTypeHelper, ConstAnnotationType,
                                            const_annotation, SpecialStructType, int32, uint32, float32, float64, bool_,
                                            void)
from .type_inference_core import (TypeInferenceError, TypeMismatchError, UnsupportedNodeError, UndefinedVariableError)
from .type_inference_visitors import promote_type
from ..registries.loop_modifier_registry import get_loop_modifier_registry
from ..resolve_attribute import recursive_resolve_attribute


class StatementTypeInferencer:
    """Handles type inference for statement nodes (FunctionDef, Assign, For, etc.)"""

    def __init__(self, inferencer):
        """Initialize with reference to parent TypeInferencer."""
        self.inferencer = inferencer

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions"""
        if node.returns is None:
            raise TypeInferenceError("Function requires return type annotation", node=node)
        return_type = self.inferencer._resolve_annotation(node.returns)
        # Store return type in all_types for codegen
        self.inferencer.all_types[node.returns] = return_type

        self.inferencer.push_scope()
        # Scan function body for local variable definitions
        # This ensures is_local() returns True for variables assigned in the function
        self.inferencer.scope_manager.define_from_ast(node.args)
        self.inferencer.scope_manager.scan_for_locals(node.body)
        try:
            for arg in node.args.args:
                if arg.annotation is None:
                    raise TypeInferenceError(f"Parameter '{arg.arg}' requires type annotation", node=arg)
                arg_type = self.inferencer._resolve_annotation(arg.annotation)
                # Store parameter type in all_types for codegen
                self.inferencer.all_types[arg.annotation] = arg_type
                self.inferencer.add_variable_to_scope(arg.arg, arg_type)

            for stmt in node.body:
                self.inferencer.visit(stmt)
                if isinstance(stmt, ast.Return):
                    if stmt.value is None:
                        if return_type != void:
                            raise TypeMismatchError(
                                f"Return statement has no value, but function return type is {return_type}", node=stmt)
                    else:
                        return_value_type = self.inferencer.infer_node_type(stmt.value)
                        if return_value_type != return_type:
                            raise TypeMismatchError(
                                f"Return value type {return_value_type} does not match function return type {return_type}",
                                node=stmt)

            # Store function name and return type in parent scope
            # We need to access the parent scope (before we pop)
            if len(self.inferencer.scope_stack) >= 2:
                self.inferencer.scope_stack[-2][node.name] = return_type
        finally:
            self.inferencer.pop_scope()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Process annotated assignments (e.g., "x: const = 5" or "x: int32 = 5")"""
        if not isinstance(node.target, ast.Name):
            raise UnsupportedNodeError(
                f"Annotated assignment target must be a single variable (got {type(node.target).__name__})",
                node=node.target)

        var_name = node.target.id
        annotation = node.annotation

        is_const_annotation = False
        try:
            resolved_ann = recursive_resolve_attribute(annotation, self.inferencer.ctx)
            if isinstance(resolved_ann, ast.Constant):
                resolved_ann = resolved_ann.value
            is_const_annotation = (isinstance(resolved_ann, ConstTypeHelper)
                                   or isinstance(resolved_ann, ConstAnnotationType) or resolved_ann == const_annotation)
        except Exception:
            if isinstance(annotation, ast.Name) and annotation.id == "const":
                is_const_annotation = True
            elif isinstance(annotation, ast.Attribute) and annotation.attr == "const":
                is_const_annotation = True

        if is_const_annotation:
            if node.value is None:
                raise TypeInferenceError("Const annotation requires a value (e.g., 'x: const = 5')", node=node)
            value_type = self.inferencer.infer_node_type(node.value)
            if var_name not in self.inferencer.current_scope:
                self.inferencer.add_variable_to_scope(var_name, value_type)
            else:
                existing_type = self.inferencer.get_variable_type(var_name, node)
                if existing_type != value_type:
                    raise TypeMismatchError(
                        f"Reassignment to '{var_name}' changes type: {existing_type} → {value_type}", node=node)
        else:
            target_type = self.inferencer._resolve_annotation(annotation)
            if node.value is not None:
                value_type = self.inferencer.infer_node_type(node.value)
                # Check if target_type is Const[inner_type] - allow assignment if inner_type matches value_type
                if isinstance(target_type, Const):
                    # For const[Type], check if inner_type matches value_type
                    if target_type.inner_type != value_type:
                        # Allow implicit conversion between compatible numeric types
                        # Check if both are integer types (int32, uint32, int64, uint64, etc.)
                        if isinstance(target_type.inner_type, IntType) and isinstance(value_type, IntType):
                            # Allow implicit conversion between integer types
                            pass
                        # Allow implicit conversion from integer to float
                        elif isinstance(target_type.inner_type, FloatType) and isinstance(value_type, IntType):
                            # Allow implicit conversion from integer to float
                            pass
                        # Allow implicit conversion between float types (float32 -> float64)
                        elif isinstance(target_type.inner_type, FloatType) and isinstance(value_type, FloatType):
                            # Allow implicit conversion between float types
                            pass
                        else:
                            raise TypeMismatchError(
                                f"Assignment annotation type {target_type} (inner type {target_type.inner_type}) does not match value type {value_type}",
                                node=node)
                elif target_type != value_type:
                    # Allow implicit conversion between compatible numeric types
                    # Check if both are integer types (int32, uint32, int64, uint64, etc.)
                    if isinstance(target_type, IntType) and isinstance(value_type, IntType):
                        # Allow implicit conversion between integer types
                        pass
                    # Allow implicit conversion from integer to float
                    elif isinstance(target_type, FloatType) and isinstance(value_type, IntType):
                        # Allow implicit conversion from integer to float
                        pass
                    # Allow implicit conversion between float types (float32 -> float64)
                    elif isinstance(target_type, FloatType) and isinstance(value_type, FloatType):
                        # Allow implicit conversion between float types
                        pass
                    else:
                        raise TypeMismatchError(
                            f"Assignment annotation type {target_type} does not match value type {value_type}",
                            node=node)
            if var_name not in self.inferencer.current_scope:
                self.inferencer.add_variable_to_scope(var_name, target_type)
            else:
                existing_type = self.inferencer.get_variable_type(var_name, node)
                if existing_type != target_type:
                    raise TypeMismatchError(
                        f"Reassignment to '{var_name}' changes type: {existing_type} → {target_type}", node=node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignment statements"""
        if len(node.targets) != 1:
            raise UnsupportedNodeError(
                "Only single-target assignments are supported (no multiple targets like 'a = b = 5')", node=node)

        target = node.targets[0]

        if hasattr(node, '_is_special_struct_decl') and node._is_special_struct_decl:
            struct_name = getattr(node, '_special_struct_name', None)
            if struct_name and isinstance(target, ast.Name):
                var_name = target.id
                special_struct_type = SpecialStructType(struct_name)
                if var_name not in self.inferencer.current_scope:
                    self.inferencer.add_variable_to_scope(var_name, special_struct_type)
                else:
                    existing_type = self.inferencer.get_variable_type(var_name, node)
                    if existing_type != special_struct_type:
                        raise TypeMismatchError(
                            f"Reassignment to '{var_name}' changes type: {existing_type} → {special_struct_type}",
                            node=node)
                self.inferencer.all_types[node.value] = special_struct_type
                return

        value_type = self.inferencer.infer_node_type(node.value)

        if isinstance(target, ast.Name):
            var_name = target.id
            if hasattr(target, "annotation") and target.annotation is not None:
                target_type = self.inferencer._resolve_annotation(target.annotation)
                if target_type != value_type:
                    raise TypeMismatchError(
                        f"Assignment annotation type {target_type} does not match value type {value_type}", node=node)
            else:
                target_type = value_type

            if var_name not in self.inferencer.current_scope:
                self.inferencer.add_variable_to_scope(var_name, target_type)
            else:
                existing_type = self.inferencer.get_variable_type(var_name, node)
                if existing_type != target_type:
                    raise TypeMismatchError(
                        f"Reassignment to '{var_name}' changes type: {existing_type} → {target_type}", node=node)

        elif isinstance(target, ast.Tuple):
            value_type = self.inferencer.infer_node_type(node.value)
            if isinstance(value_type, TupleType):
                if len(target.elts) != len(value_type.element_types):
                    raise TypeMismatchError(
                        f"Tuple unpacking length mismatch: target has {len(target.elts)} elements, "
                        f"value has {len(value_type.element_types)}", node=node)

                for target_elt, element_type in zip(target.elts, value_type.element_types):
                    if not isinstance(target_elt, ast.Name):
                        raise UnsupportedNodeError(
                            f"Tuple unpacking only supports variable targets (got {type(target_elt).__name__})",
                            node=target_elt)
                    elt_name = target_elt.id
                    if elt_name not in self.inferencer.current_scope:
                        self.inferencer.add_variable_to_scope(elt_name, element_type)
                    else:
                        existing_type = self.inferencer.get_variable_type(elt_name, node)
                        if existing_type != element_type:
                            raise TypeMismatchError(
                                f"Tuple unpacking changes type of '{elt_name}': {existing_type} → {element_type}",
                                node=node)
                return

            if isinstance(node.value, ast.Tuple):
                if len(target.elts) != len(node.value.elts):
                    raise TypeMismatchError(
                        f"Tuple unpacking length mismatch: target has {len(target.elts)} elements, "
                        f"value has {len(node.value.elts)}", node=node)
                for target_elt, value_elt in zip(target.elts, node.value.elts):
                    if not isinstance(target_elt, ast.Name):
                        raise UnsupportedNodeError(
                            f"Tuple unpacking only supports variable targets (got {type(target_elt).__name__})",
                            node=target_elt)
                    elt_value_type = self.inferencer.infer_node_type(value_elt)
                    elt_name = target_elt.id
                    if elt_name not in self.inferencer.current_scope:
                        self.inferencer.add_variable_to_scope(elt_name, elt_value_type)
                    else:
                        existing_type = self.inferencer.get_variable_type(elt_name, node)
                        if existing_type != elt_value_type:
                            raise TypeMismatchError(
                                f"Tuple unpacking changes type of '{elt_name}': {existing_type} → {elt_value_type}",
                                node=node)
            else:
                raise TypeMismatchError(
                    f"Tuple unpacking requires tuple value (got {type(value_type).__name__}). "
                    f"Value node: {ast.dump(node.value)}", node=node)

        elif isinstance(target, ast.Subscript):
            base_type = self.inferencer.infer_node_type(target.value)
            if not (base_type.is_tensor() or base_type.is_pointer()):
                raise TypeMismatchError(f"Subscript assignment requires tensor/pointer type (got {base_type})",
                                        node=target)
            if base_type.is_tensor():
                expected_type = base_type.element_type
            else:
                expected_type = base_type.inner_type

            # Apply type promotion (C++ style implicit conversion)
            # Try to promote value_type to match expected_type
            promoted_value_type, promoted_expected_type, was_promoted = promote_type(value_type, expected_type)

            # After promotion, both types should match
            if promoted_value_type != promoted_expected_type:
                raise TypeMismatchError(
                    f"Subscript assignment type {value_type} cannot be promoted to match base element type {expected_type}",
                    node=node)

            # If promotion occurred, update the value type in all_types for codegen
            # The promoted type should match expected_type
            if was_promoted:
                self.inferencer.all_types[node.value] = promoted_expected_type

        else:
            raise UnsupportedNodeError(f"Unsupported assignment target type: {type(target).__name__}", node=target)

    def visit_For(self, node: ast.For) -> None:
        """Process for loops (supports range() and loop modifiers like ll.unroll, ll.parallel)"""
        iter_node = node.iter
        loop_modifier_registry = get_loop_modifier_registry()

        # Check if iterator is wrapped in a loop modifier (e.g., ll.unroll, ll.parallel)
        if isinstance(iter_node, ast.Call):
            if loop_modifier_registry.is_modifier_call(iter_node, self.inferencer.ctx):
                try:
                    iter_node = loop_modifier_registry.unwrap_modifier(iter_node, self.inferencer.ctx)
                except ValueError as e:
                    raise TypeInferenceError(str(e), node=iter_node)

        # Validate iterator is range()
        if not (isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name)
                and iter_node.func.id == "range"):
            raise UnsupportedNodeError(
                f"For loop iterator must be range() or a loop modifier wrapping range() "
                f"(e.g., ll.unroll(range()), ll.parallel(range())) (got {type(iter_node).__name__})", node=iter_node)

        if not isinstance(node.target, ast.Name):
            raise UnsupportedNodeError(f"For loop target must be a single variable (got {type(node.target).__name__})",
                                       node=node.target)
        loop_var_name = node.target.id

        if hasattr(node.target, "annotation") and node.target.annotation is not None:
            loop_var_type = self.inferencer._resolve_annotation(node.target.annotation)
            if not (isinstance(loop_var_type, ll.IntType) and not loop_var_type.special):
                raise TypeMismatchError(f"Loop variable must be a regular integer type (got {loop_var_type})",
                                        node=node.target)
        else:
            loop_var_type = int32

        self.inferencer.push_scope()
        try:
            self.inferencer.add_variable_to_scope(loop_var_name, loop_var_type)
            for stmt in node.body:
                self.inferencer.visit(stmt)
            for stmt in node.orelse:
                self.inferencer.visit(stmt)
        finally:
            self.inferencer.pop_scope()

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Process augmented assignment (e.g., x += 1, y -= 2).
        
        For augmented assignment, we need to:
        1. Check that the target variable exists and get its type
        2. Infer the type of the value
        3. Check that the operation is valid for these types
        4. Ensure the result type matches the target type
        """
        if not isinstance(node.target, ast.Name):
            raise UnsupportedNodeError(
                f"Augmented assignment only supports variable targets (got {type(node.target).__name__})",
                node=node.target)

        var_name = node.target.id

        # Get the variable type (will search parent scopes if not in current scope)
        # This will raise UndefinedVariableError if the variable doesn't exist
        try:
            target_type = self.inferencer.get_variable_type(var_name, node.target)
        except UndefinedVariableError:
            raise TypeInferenceError(f"Variable '{var_name}' used in augmented assignment before assignment",
                                     node=node.target)
        value_type = self.inferencer.infer_node_type(node.value)

        # For augmented assignment, the result type should match the target type
        # We need to check if the operation is valid
        # For most operations (+, -, *, /, etc.), the result type should match the operand types
        # For now, we'll do a simple check: if both are numeric types, allow it
        # More sophisticated type checking can be added later

        # Check if types are compatible for the operation
        # For arithmetic operations, both operands should be numeric

        numeric_types = {int32, uint32, float32, float64}

        if target_type in numeric_types and value_type in numeric_types:
            # Numeric types are compatible for arithmetic operations
            # The result type should match the target type (or be promoted)
            # For now, we'll keep the target type
            result_type = target_type
        elif target_type == value_type:
            # Same types are always compatible
            result_type = target_type
        else:
            # Check for specific operator compatibility
            op_type = type(node.op)
            if op_type in (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow):
                # Arithmetic operations require numeric types
                raise TypeMismatchError(
                    f"Augmented assignment operator {op_type.__name__} requires compatible numeric types, "
                    f"got {target_type} and {value_type}", node=node)
            elif op_type in (ast.LShift, ast.RShift, ast.BitAnd, ast.BitOr, ast.BitXor):
                # Bitwise operations require integer types
                # Allow bool as it can be implicitly converted to int
                allowed_types = {int32, uint32, bool_}
                if target_type not in allowed_types or value_type not in allowed_types:
                    raise TypeMismatchError(
                        f"Bitwise augmented assignment operator {op_type.__name__} requires integer types (or bool), "
                        f"got {target_type} and {value_type}", node=node)
                # Result type should match target type (bool will be converted to int)
                result_type = target_type
            else:
                raise TypeMismatchError(
                    f"Augmented assignment operator {op_type.__name__} not supported for types "
                    f"{target_type} and {value_type}", node=node)

        # Store type information for codegen
        self.inferencer.all_types[node.value] = value_type
        self.inferencer.all_types[node.target] = result_type
