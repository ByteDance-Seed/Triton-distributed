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
from typing import Optional
import little_kernel.language as ll
from little_kernel.core.type_system import LLType, ScalarType, Pointer, TupleType, SpecialStructType, FloatType, Tensor, int32, float32, float64, bool_
from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError, TypeMismatchError, UnsupportedNodeError, UndefinedVariableError
from little_kernel.core.passes.utils.type_inference.type_inference_visitors import ConstantTypeInferencer, AnnotationResolver, OperatorTypeInferencer
from little_kernel.core.passes.utils.type_inference.type_inference_call import CallTypeInferencer
from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core import ir as lk_ir


class ASTNodeVisitors:
    """Handles type inference for various AST node types."""

    def __init__(self, inferencer):
        """Initialize with reference to parent TypeInferencer."""
        self.inferencer = inferencer
        self.constant_inferencer = ConstantTypeInferencer()
        self.operator_inferencer = OperatorTypeInferencer()
        self.annotation_resolver = AnnotationResolver(inferencer.ctx, inferencer._resolve_annotation,
                                                      inferencer.scope_manager)
        self.call_inferencer = CallTypeInferencer(inferencer.ctx, inferencer.scope_manager,
                                                  inferencer._resolve_annotation, inferencer.infer_node_type,
                                                  inferencer.all_types)

    def visit_Module(self, node: ast.Module) -> None:
        """Process top-level module (traverse all statements in global scope)"""
        for stmt in node.body:
            self.inferencer.visit(stmt)

    def visit_Constant(self, node: ast.Constant) -> LLType:
        """Infer type of ast.Constant nodes"""
        const_type = self.constant_inferencer.infer_constant_type(node)
        self.inferencer.all_types[node] = const_type
        return const_type

    def visit_Num(self, node: ast.Num) -> LLType:
        """Infer type of numeric constants (Python 3.7 and earlier)"""
        # Convert ast.Num to ast.Constant for compatibility
        const_node = ast.Constant(value=node.n)
        const_type = self.constant_inferencer.infer_constant_type(const_node)
        self.inferencer.all_types[node] = const_type
        return const_type

    def visit_Str(self, node: ast.Str) -> LLType:
        """Infer type of string constants (Python 3.7 and earlier)"""
        # Convert ast.Str to ast.Constant for compatibility
        const_node = ast.Constant(value=node.s)
        const_type = self.constant_inferencer.infer_constant_type(const_node)
        self.inferencer.all_types[node] = const_type
        return const_type

    def visit_Name(self, node: ast.Name) -> LLType:
        """Infer type of variable references (ast.Name)"""
        # Transitional compatibility for legacy role lowering that hid raw C++
        # on ast.Name. Prefer an explicitly attached declared type (so pointer
        # results retain pointer types); the historical int32 default applies
        # only to unmigrated producers. New lowering uses
        # lk_ir.TargetExpression, handled in visit_TargetExpression.
        if lk_ir.is_legacy_cpp_node(node):
            result_type = lk_ir.legacy_cpp_expr_type(node)
            self.inferencer.all_types[node] = result_type
            return result_type

        # Priority order:
        # 1. Current scope type map (for types set by eval_arg_type, e.g., alloc_shared_memory)
        # 2. Scope manager type map (for variables with inferred types)
        # 3. Global context (for global constants and types)
        # 4. ll module (for type aliases like int64 -> ll.int64)

        # First check current_scope - this catches types set by eval_arg_type
        # (e.g., alloc_shared_memory sets buffer type before we try to infer it)
        if node.id in self.inferencer.current_scope:
            var_type = self.inferencer.current_scope[node.id]
            self.inferencer.all_types[node] = var_type
            return var_type

        # Check scope manager's type map
        var_type = self.inferencer.scope_manager.get_variable_type(node.id)
        if var_type is not None:
            self.inferencer.all_types[node] = var_type
            return var_type

        # Check if this is a local variable without a type yet
        # If it's local but no type, don't check global ctx (it shadows global)
        if self.inferencer.scope_manager.is_local(node.id):
            # Local variable with no type - error
            raise UndefinedVariableError(f"Local variable '{node.id}' used before type is known", node=node)

        # Not local, check global context
        if node.id in self.inferencer.ctx:
            ctx_value = self.inferencer.ctx[node.id]
            if isinstance(ctx_value, LLType):
                self.inferencer.all_types[node] = ctx_value
                return ctx_value
            # Handle primitive types from global context (e.g., ArraySize = 128)
            elif isinstance(ctx_value, int):
                self.inferencer.all_types[node] = int32
                return int32
            elif isinstance(ctx_value, float):
                self.inferencer.all_types[node] = float32
                return float32
            elif isinstance(ctx_value, bool):
                self.inferencer.all_types[node] = bool_
                return bool_

        # If not found, try to resolve from little_kernel.language module (e.g., int64 -> ll.int64)
        # This handles cases where ll.int64 was converted to int64 by constfold
        if hasattr(ll, node.id):
            attr_value = getattr(ll, node.id)
            if isinstance(attr_value, LLType):
                self.inferencer.all_types[node] = attr_value
                return attr_value

        # Re-raise error if really undefined
        raise UndefinedVariableError(f"Undefined variable: {node.id}", node=node)

    def visit_BinOp(self, node: ast.BinOp) -> LLType:
        """Infer type of binary operations (e.g., x + y)"""
        left_type = self.inferencer.infer_node_type(node.left)
        right_type = self.inferencer.infer_node_type(node.right)

        bin_type = self.operator_inferencer.infer_bin_op(left_type, right_type, type(node.op), node)
        self.inferencer.all_types[node] = bin_type
        # Merge repair (conflict-11): operands keep their own inferred types
        # even when the BinOp promotes to float64. A float literal stays
        # typed float32 (ConstantTypeInferencer), so codegen emits the
        # f-suffixed literal and C++ usual arithmetic conversions perform
        # the promotion at the expression level. Retyping the literal to
        # float64 here (the alternative merged side) made codegen emit a
        # bare double literal, breaking the pinned candidate-base contract
        # in tests/nv/common/unit/test_scalar_literal.py.
        return bin_type

    def visit_UnaryOp(self, node: ast.UnaryOp) -> LLType:
        """Infer type of unary operations (e.g., -x, not x)"""
        operand_type = self.inferencer.infer_node_type(node.operand)

        un_type = self.operator_inferencer.infer_un_op(operand_type, type(node.op), node)
        self.inferencer.all_types[node] = un_type
        return un_type

    def visit_BoolOp(self, node: ast.BoolOp) -> LLType:
        """Infer type of boolean operations (and/or) always returns bool"""
        for value in node.values:
            value_type = self.inferencer.infer_node_type(value)
            if not isinstance(value_type, (ScalarType, Pointer)):
                raise TypeMismatchError(f"Boolean operation requires bool operand (got {value_type})", node=value)

        bool_type = self.operator_inferencer.infer_bool_op(type(node.op), node)
        self.inferencer.all_types[node] = bool_type
        return bool_type

    def visit_Compare(self, node: ast.Compare) -> LLType:
        """Infer type of comparison operations (e.g., x < y) always returns bool.
        
        Also allows pointer/tensor vs integer comparisons (null pointer checks)
        and integer width mismatches (e.g., int32 vs uint32).
        """
        left_type = self.inferencer.infer_node_type(node.left)

        for comparator in node.comparators:
            comp_type = self.inferencer.infer_node_type(comparator)
            if comp_type != left_type:
                # Allow pointer/tensor vs integer comparison (nullptr checks)
                left_is_ptr = left_type.is_pointer() or left_type.is_tensor()
                right_is_ptr = comp_type.is_pointer() or comp_type.is_tensor()
                left_is_int = isinstance(left_type, ll.IntType)
                right_is_int = isinstance(comp_type, ll.IntType)
                if (left_is_ptr and right_is_int) or (right_is_ptr and left_is_int):
                    pass  # Allow pointer vs int comparison
                elif left_is_int and right_is_int:
                    pass  # Allow int width/sign mismatches (e.g., int32 vs uint32)
                else:
                    raise TypeMismatchError(
                        f"Comparison requires matching types: {left_type} (left) vs {comp_type} (comparator)",
                        node=node)

        self.inferencer.all_types[node] = bool_
        return bool_

    def visit_Subscript(self, node: ast.Subscript) -> LLType:
        """Infer type of subscript expression (e.g., "A[i]" → Tensor element type)"""
        from little_kernel.core.type_system import SpecialStructType, VectorType, TupleType, VoidType
        base_type = self.inferencer.infer_node_type(node.value)

        # TupleType element access: t[<const int>] -> element type (lowers to
        # std::get<i> in codegen)
        if isinstance(base_type, TupleType):
            slc = node.slice
            if hasattr(ast, "Index") and isinstance(slc, ast.Index):
                slc = slc.value
            if not (isinstance(slc, ast.Constant) and isinstance(slc.value, int)):
                raise TypeMismatchError("Tuple subscript index must be a constant integer", node=node)
            if not 0 <= slc.value < len(base_type.element_types):
                raise TypeMismatchError(f"Tuple index {slc.value} out of range for {base_type}", node=node)
            ret = base_type.element_types[slc.value]
            self.inferencer.all_types[node] = ret
            return ret

        # SpecialStructType (GlobalTensor, LocalTensor, etc.) supports subscript
        if isinstance(base_type, SpecialStructType):
            self.inferencer.all_types[node] = base_type
            return base_type

        # one element of a register vector (ext_vector_type: v[i])
        if isinstance(base_type, VectorType):
            idx_type = self.inferencer.infer_node_type(node.slice)
            if not (isinstance(idx_type, ll.IntType) and not idx_type.special):
                raise TypeMismatchError(f"Vector element index must be a regular integer (got {idx_type})",
                                        node=node.slice)
            self.inferencer.all_types[node] = base_type.element_type
            return base_type.element_type

        # VoidType from opaque struct stub method returns also supports subscript
        if isinstance(base_type, VoidType):
            self.inferencer.all_types[node] = base_type
            return base_type

        # A const view (const[ptr[T]] / const[Tensor[T]]) does not change
        # indexability: peel plain const wrappers so the pointer/tensor below
        # governs the subscript. GridConstant wraps a by-value parameter and
        # is intentionally not peeled.
        while (base_type.is_const() and not base_type.is_grid_constant() and hasattr(base_type, "inner_type")):
            base_type = base_type.inner_type

        if not (base_type.is_tensor() or base_type.is_pointer() or base_type.is_generic()):
            raise TypeMismatchError(f"Subscript requires tensor or pointer or generic base type (got {base_type})",
                                    node=node)

        index_node = node.slice
        if isinstance(index_node, ast.Tuple):
            indices = index_node.elts
        else:
            indices = [index_node]

        eval_generic_type = base_type
        for idx in indices:
            idx_type = self.inferencer.infer_node_type(idx)
            if not base_type.is_generic():
                if not (isinstance(idx_type, ll.IntType) and not idx_type.special):
                    raise TypeMismatchError(f"Subscript index must be regular integer (got {idx_type})", node=idx)
            else:
                eval_generic_type = eval_generic_type[idx_type]

        if base_type.is_tensor():
            ret_type = base_type.element_type
        elif base_type.is_pointer():
            ret_type = base_type.inner_type
        elif base_type.is_generic():
            ret_type = eval_generic_type
        else:
            raise UnsupportedNodeError(f"Unsupported subscript node: {ast.dump(node)}")

        self.inferencer.all_types[node] = ret_type
        return ret_type

    def visit_List(self, node: ast.List) -> LLType:
        """Infer type of lists (assumes homogeneous elements → Tensor[element_type])"""
        if not node.elts:
            raise TypeInferenceError("Empty lists are not supported for type inference", node=node)

        elem_type = self.inferencer.infer_node_type(node.elts[0])
        for elem in node.elts[1:]:
            current_elem_type = self.inferencer.infer_node_type(elem)
            if current_elem_type != elem_type:
                raise TypeMismatchError(f"List has mixed types: {elem_type} (first) vs {current_elem_type} (later)",
                                        node=node)

        self.inferencer.all_types[node] = Tensor[elem_type]
        return Tensor[elem_type]

    def visit_Tuple(self, node: ast.Tuple) -> LLType:
        """Infer type of tuples → Tuple[type1, type2, ...]"""
        if not node.elts:
            raise TypeInferenceError("Empty tuples are not supported for type inference", node=node)

        element_types = []
        for elem in node.elts:
            elem_type = self.inferencer.infer_node_type(elem)
            element_types.append(elem_type)

        tuple_type = TupleType(tuple(element_types))
        self.inferencer.all_types[node] = tuple_type
        return tuple_type

    def visit_ListComp(self, node: ast.ListComp) -> LLType:
        """Infer type of list comprehensions (→ Tensor[element_type])"""
        elem_type = self.inferencer.infer_node_type(node.elt)

        for gen in node.generators:
            if isinstance(gen, ast.comprehension):
                if not (isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name)
                        and gen.iter.func.id == "range"):
                    raise UnsupportedNodeError(
                        f"List comprehension iterator must be range() (got {type(gen.iter).__name__})", node=gen.iter)
                if isinstance(gen.target, ast.Name):
                    self.inferencer.current_scope[gen.target.id] = int32
                else:
                    raise UnsupportedNodeError(
                        f"List comprehension target must be a single variable (got {type(gen.target).__name__})")

        self.inferencer.all_types[node] = Tensor[elem_type]
        return Tensor[elem_type]

    def visit_Expr(self, node: ast.Expr) -> None:
        """Process expression statements (side effects, no type to return)"""
        self.inferencer.infer_node_type(node.value)

    # ------------------------------ Typed lowered IR (RF06) ------------------------------
    def visit_TargetExpression(self, node) -> LLType:
        """Infer the declared result type of a backend expression escape hatch.

        Unlike the legacy ``int32`` catch-all, the result type is part of the
        node's contract and may be a pointer/address-space-bearing type.
        """
        lk_ir.verify_lowered_node(node)
        result_type = node.result_type
        self.inferencer.all_types[node] = result_type
        return result_type

    def visit_ResourceReference(self, node) -> LLType:
        """Infer the pointer result type of a lowered resource reference."""
        lk_ir.verify_lowered_node(node)
        self.inferencer.all_types[node] = node.result_type
        return node.result_type

    def visit_StageReference(self, node) -> LLType:
        """Infer a pipeline-stage-indexed resource view (pointer-typed)."""
        lk_ir.verify_lowered_node(node)
        # Infer the stage index; it must be integer when typed.
        stage_type = self.inferencer.infer_node_type(node.stage)
        if not isinstance(stage_type, ll.IntType) or getattr(stage_type, "special", False):
            raise TypeInferenceError(f"StageReference stage index must be an integer "
                                     f"(got {stage_type})", node=node.stage)
        # Infer the base so its declared pointer type is registered even
        # though the constructor already validated element consistency.
        base_type = self.inferencer.infer_node_type(node.base)
        if not base_type.is_pointer():
            raise TypeInferenceError(f"StageReference base must be a pointer (got {base_type})", node=node.base)
        if base_type.inner_type != node.result_type.inner_type:
            raise TypeInferenceError(
                f"StageReference result element type {node.result_type} "
                f"does not match base pointer {base_type}", node=node)
        self.inferencer.all_types[node] = node.result_type
        return node.result_type

    def visit_TargetStatement(self, node) -> None:
        """A backend statement has no expression result."""
        lk_ir.verify_lowered_node(node)

    def visit_TargetInclude(self, node) -> None:
        """A backend include has no expression result."""
        lk_ir.verify_lowered_node(node)

    def visit_TargetHelper(self, node) -> None:
        """A backend helper definition has no expression result."""
        lk_ir.verify_lowered_node(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Validate return statements against the enclosing function.

        Delegates to the return-context stack so direct and nested returns
        follow identical void/value/type rules at the return's own location.
        """
        self.inferencer.check_return_statement(node)

    def visit_IfExp(self, node: ast.IfExp) -> LLType:
        """Infer type of conditional expressions"""
        body_type = self.inferencer.visit(node.body)
        orelse_type = self.inferencer.visit(node.orelse)
        if body_type != orelse_type:
            raise TypeMismatchError(f"IfExp has mixed types: {body_type} (body) vs {orelse_type} (orelse)", node=node)
        self.inferencer.all_types[node] = body_type
        return body_type

    def visit_Attribute(self, node: ast.Attribute) -> LLType:
        """Infer type of attribute access"""
        if isinstance(node.value, ast.Name):
            var_name = node.value.id
            attr_name = node.attr

            # Check if var_name is local first (shadows global)
            if not self.inferencer.scope_manager.is_local(var_name):
                # Not local, try to resolve from context (could be ll module or any module with the attribute)
                if var_name in self.inferencer.ctx:
                    ctx_value = self.inferencer.ctx[var_name]
                    # Check if it's a module (like ll)
                    if hasattr(ctx_value, '__module__') or isinstance(ctx_value, type(ast)):
                        # Try to get attribute from module
                        if hasattr(ctx_value, attr_name):
                            attr_value = getattr(ctx_value, attr_name)
                            if isinstance(attr_value, LLType):
                                self.inferencer.all_types[node] = attr_value
                                return attr_value

            var_type = None
            if var_name in self.inferencer.current_scope:
                var_type = self.inferencer.current_scope[var_name]

            # Try to resolve as method first
            resolver = MethodResolver(self.inferencer.ctx, self.inferencer.current_scope,
                                      self.inferencer._resolve_annotation)
            return_type = resolver.resolve_method_return_type(var_name, attr_name, var_type)

            if return_type is not None:
                self.inferencer.all_types[node] = return_type
                return return_type

            # Try to resolve as attribute of special struct
            if var_type is not None:
                if isinstance(var_type, SpecialStructType):
                    attr_type = self._infer_special_struct_attribute(var_type.struct_name, attr_name)
                    if attr_type is not None:
                        self.inferencer.all_types[node] = attr_type
                        return attr_type

        tmp = recursive_resolve_attribute(node, self.inferencer.ctx, self.inferencer.scope_manager)
        if not isinstance(tmp, ast.AST):
            attr_type = self.inferencer.infer_node_type(ast.Constant(value=tmp))
            self.inferencer.all_types[node] = attr_type
            return attr_type
        else:
            # Attribute resolution failed - raise explicit error instead of silently
            # returning int32 (which masks type errors and causes hard-to-debug issues)
            if isinstance(node.value, ast.Name):
                var_name = node.value.id
                attr_name = node.attr
                if var_name in self.inferencer.current_scope:
                    var_type = self.inferencer.current_scope[var_name]
                    raise TypeInferenceError(
                        f"Cannot resolve attribute '{attr_name}' on variable '{var_name}' of type {var_type}. "
                        "The attribute may not exist or may not be supported for type inference.", node=node)
            raise UnsupportedNodeError(f"Cannot resolve attribute access: {ast.dump(node)}", node=node)

    def _infer_special_struct_attribute(self, struct_name: str, attr_name: str) -> Optional[LLType]:
        """Infer type of a special struct attribute by analyzing __init__ method."""
        try:

            struct_info = get_special_struct_codegen(struct_name)
            if struct_info and 'class' in struct_info:
                class_obj = struct_info['class']
                if class_obj is not None and inspect.isclass(class_obj):
                    # Try to get attribute type from __init__ method
                    try:
                        source = inspect.getsource(class_obj)
                        source = textwrap.dedent(source)
                        module_ast = ast.parse(source)

                        for node in ast.walk(module_ast):
                            if isinstance(node, ast.ClassDef) and node.name == class_obj.__name__:
                                for item in node.body:
                                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                                        # Look for self.attr_name = param_name assignments
                                        for stmt in item.body:
                                            if isinstance(stmt, ast.Assign):
                                                for target in stmt.targets:
                                                    if isinstance(target, ast.Attribute):
                                                        if isinstance(target.value,
                                                                      ast.Name) and target.value.id == 'self':
                                                            if target.attr == attr_name:
                                                                # Found self.attr_name = value
                                                                # Try to infer type from the value
                                                                if isinstance(stmt.value, ast.Name):
                                                                    # self.attr = param_name
                                                                    param_name = stmt.value.id
                                                                    # Find parameter type annotation
                                                                    for arg in item.args.args:
                                                                        if arg.arg == param_name:
                                                                            if arg.annotation:
                                                                                return self.inferencer._resolve_annotation(
                                                                                    arg.annotation)
                                                                            # If no annotation, try to infer from parameter name
                                                                            # Common patterns: shape_* -> int32, *_layout -> ptr[int32], etc.
                                                                            if 'shape' in param_name.lower():
                                                                                return ll.int32
                                                                            elif 'layout' in param_name.lower():
                                                                                return ll.ptr[ll.int32]
                                                                            # Default to int32 for numeric-looking parameters
                                                                            return ll.int32
                                                                else:
                                                                    # Try to infer type from the expression
                                                                    try:
                                                                        return self.inferencer.infer_node_type(
                                                                            stmt.value)
                                                                    except Exception:
                                                                        pass
                    except Exception:
                        pass
        except ImportError:
            pass
        return None
