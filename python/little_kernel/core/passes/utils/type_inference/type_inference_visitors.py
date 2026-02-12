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
from enum import Enum
from typing import Dict, Any, Optional
from typing import Tuple as PyTuple
from little_kernel.core.type_system import (LLType, IntType, FloatType, Pointer, Const, Template, TupleType,
                                            AnnotateTypeHelper, TupleTypeHelper, int32, int64, float32, bool_, str_)
from .type_inference_core import (TypeInferenceError, TypeMismatchError, UnsupportedNodeError)
from ..registries.operator_registry import get_operator_registry
from ..resolve_attribute import recursive_resolve_attribute


class ConstantTypeInferencer:
    """Handles type inference for constant values."""

    def infer_constant_type(self, node) -> LLType:
        """Infer LLType for ast.Constant nodes (handles primitives and LLType constants)"""
        # Handle both ast.Constant and ast.Num (Python 3.7)
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Num):
            value = node.n
        else:
            # Try to get value attribute
            value = getattr(node, 'value', None)
            if value is None:
                raise UnsupportedNodeError(f"Unsupported constant type: {type(node).__name__} (value: {value})",
                                           node=node)
        if isinstance(value, LLType):
            return value
        elif isinstance(value, bool):
            # bool values should be bool_ type, not int32
            return bool_
        elif isinstance(value, int):
            return int32
        elif isinstance(value, float):
            return float32
        elif isinstance(value, str):
            return str_
        elif hasattr(value, '__class__') and hasattr(value.__class__, '__module__'):
            if isinstance(value, Enum):
                return int32
        else:
            raise UnsupportedNodeError(f"Unsupported constant type: {type(value).__name__} (value: {value})", node=node)


class AnnotationResolver:
    """Handles resolution of type annotations."""

    def __init__(self, ctx: Dict[str, Any], resolve_annotation_func: callable, scope_manager=None):
        self.ctx = ctx
        self._resolve_annotation = resolve_annotation_func
        self.annotation_cache: Dict[ast.AST, LLType] = {}
        self.scope_manager = scope_manager

    def resolve_annotation(self, ann_node: ast.AST) -> LLType:
        """Evaluate an AST annotation node to LLType."""
        if ann_node in self.annotation_cache:
            return self.annotation_cache[ann_node]

        if isinstance(ann_node, ast.Subscript):
            result = self._resolve_subscript_annotation(ann_node)
            if result is not None:
                self.annotation_cache[ann_node] = result
                return result

        try:
            resolved = recursive_resolve_attribute(ann_node, self.ctx, self.scope_manager)
        except Exception as e:
            raise TypeInferenceError(f"Failed to resolve annotation (check aliases/imports): {str(e)}",
                                     node=ann_node) from e

        if isinstance(resolved, ast.Constant):
            resolved = resolved.value

        if isinstance(resolved, type) and issubclass(resolved, Enum):
            resolved = int32

        if not isinstance(resolved, LLType):
            raise TypeInferenceError(
                f"Annotation resolved to {type(resolved).__name__} (value: {resolved}), expected LLType", node=ann_node)

        self.annotation_cache[ann_node] = resolved
        return resolved

    def _resolve_subscript_annotation(self, ann_node: ast.Subscript) -> Optional[LLType]:
        """Resolve subscript annotations like template[...], Tuple[...], or const[...]"""
        # Check if it's const[...]
        is_const = self._check_is_const(ann_node)
        if is_const:
            try:
                # Handle Python 3.7: slice might be wrapped in ast.Index
                if isinstance(ann_node.slice, ast.Index):
                    slice_node = ann_node.slice.value
                else:
                    slice_node = ann_node.slice
                inner_type = self._resolve_annotation(slice_node)
            except (TypeInferenceError, ValueError) as e:
                # Only use int32 as fallback if we really can't resolve
                # Don't silently fail - this might indicate a real error
                raise TypeInferenceError(f"Failed to resolve inner type for const[...]: {e}", node=ann_node) from e
            return Const(inner_type)

        # Check if it's template[...]
        is_template = self._check_is_template(ann_node)
        if is_template:
            try:
                inner_type = self._resolve_annotation(ann_node.slice)
            except (TypeInferenceError, ValueError):
                inner_type = int32
            return Template(inner_type)

        # Check if it's Tuple[...]
        is_tuple = self._check_is_tuple(ann_node)
        if is_tuple:
            if isinstance(ann_node.slice, ast.Tuple):
                element_types = [self._resolve_annotation(elt) for elt in ann_node.slice.elts]
            else:
                element_type = self._resolve_annotation(ann_node.slice)
                element_types = [element_type]
            return TupleType(tuple(element_types))

        return None

    def _check_is_template(self, ann_node: ast.Subscript) -> bool:
        """Check if annotation is template[...]"""
        if isinstance(ann_node.value, ast.Name):
            return ann_node.value.id == "template"
        elif isinstance(ann_node.value, ast.Attribute):
            if ann_node.value.attr == "template":
                try:
                    base_resolved = recursive_resolve_attribute(ann_node.value, self.ctx, self.scope_manager)
                    if isinstance(base_resolved, ast.Constant):
                        base_resolved = base_resolved.value
                    if isinstance(base_resolved, AnnotateTypeHelper) and base_resolved.inner_type == Template:
                        return True
                    try:
                        if base_resolved == Template:
                            return True
                    except Exception:
                        pass
                except Exception:
                    pass
        return False

    def _check_is_const(self, ann_node: ast.Subscript) -> bool:
        """Check if annotation is const[...]"""
        if isinstance(ann_node.value, ast.Name):
            return ann_node.value.id == "const"
        elif isinstance(ann_node.value, ast.Attribute):
            if ann_node.value.attr == "const":
                try:
                    base_resolved = recursive_resolve_attribute(ann_node.value, self.ctx, self.scope_manager)
                    if isinstance(base_resolved, ast.Constant):
                        base_resolved = base_resolved.value
                    if isinstance(base_resolved, AnnotateTypeHelper) and base_resolved.inner_type == Const:
                        return True
                    try:
                        if base_resolved == Const:
                            return True
                    except Exception:
                        pass
                except Exception:
                    pass
        return False

    def _check_is_tuple(self, ann_node: ast.Subscript) -> bool:
        """Check if annotation is Tuple[...] or tuple[...]"""
        if isinstance(ann_node.value, ast.Name):
            return ann_node.value.id in ["Tuple", "tuple"]
        elif isinstance(ann_node.value, ast.Attribute):
            if ann_node.value.attr in ["Tuple", "tuple"]:
                try:
                    base_resolved = recursive_resolve_attribute(ann_node.value, self.ctx, self.scope_manager)
                    if isinstance(base_resolved, ast.Constant):
                        base_resolved = base_resolved.value
                    if isinstance(base_resolved, TupleTypeHelper):
                        return True
                except Exception:
                    pass
        elif isinstance(ann_node.value, ast.Constant):
            return ann_node.value.value is tuple
        return False


def promote_type(left_type: LLType, right_type: LLType) -> PyTuple[LLType, LLType, bool]:
    """
    Promote types following C++ implicit conversion rules.
    
    Returns:
        (promoted_left, promoted_right, was_promoted): The promoted types and whether promotion occurred.
    """
    # If types are already the same, no promotion needed
    if left_type == right_type:
        return left_type, right_type, False

    # Both are integer types
    if isinstance(left_type, IntType) and isinstance(right_type, IntType):
        # Handle special integer types (bool, binary) - promote to int32
        had_special = False
        if left_type.special is not None:
            left_type = int32
            had_special = True
        if right_type.special is not None:
            right_type = int32
            had_special = True
        if left_type == right_type:
            return left_type, right_type, had_special

        left_bits = left_type.bits
        right_bits = right_type.bits
        left_signed = left_type.signed
        right_signed = right_type.signed

        # C++ integer promotion rules:
        # 1. If sizes differ, promote to larger size
        # 2. If same size but one signed and one unsigned, prefer unsigned
        # 3. If both same size and same signedness, they should be same type

        if left_bits == right_bits:
            # Same size: prefer unsigned (C++ rule: unsigned wins in mixed operations)
            if not left_signed and right_signed:
                # left is unsigned, right is signed -> promote both to unsigned
                return left_type, left_type, True
            elif left_signed and not right_signed:
                # left is signed, right is unsigned -> promote both to unsigned
                return right_type, right_type, True
            else:
                # Both signed or both unsigned, should be same type
                return left_type, right_type, False
        else:
            # Different sizes: promote smaller to larger
            if left_bits > right_bits:
                # Promote right to left's size
                # Create a new type with left's size and signedness
                # If left is unsigned, promote right to unsigned of left's size
                # If left is signed, promote right to signed of left's size
                if not left_signed:
                    # Left is unsigned, promote right to unsigned of left's size
                    promoted_right = IntType(bits=left_bits, signed=False)
                    return left_type, promoted_right, True
                else:
                    # Left is signed, promote right to signed of left's size
                    promoted_right = IntType(bits=left_bits, signed=True)
                    return left_type, promoted_right, True
            else:
                # Promote left to right's size
                if not right_signed:
                    # Right is unsigned, promote left to unsigned of right's size
                    promoted_left = IntType(bits=right_bits, signed=False)
                    return promoted_left, right_type, True
                else:
                    # Right is signed, promote left to signed of right's size
                    promoted_left = IntType(bits=right_bits, signed=True)
                    return promoted_left, right_type, True

    # One is integer, one is float: promote integer to float
    if isinstance(left_type, IntType) and isinstance(right_type, FloatType):
        # Promote integer to float
        # Choose appropriate float size: use the float's size
        return right_type, right_type, True

    if isinstance(left_type, FloatType) and isinstance(right_type, IntType):
        # Promote integer to float
        return left_type, left_type, True

    # Both are float types: promote to larger float
    if isinstance(left_type, FloatType) and isinstance(right_type, FloatType):
        if left_type.bits > right_type.bits:
            return left_type, left_type, True
        elif right_type.bits > left_type.bits:
            return right_type, right_type, True
        else:
            return left_type, right_type, False

    # No promotion possible
    return left_type, right_type, False


class OperatorTypeInferencer:
    """Handles type inference for operators using registry."""

    def __init__(self):
        self.registry = get_operator_registry()

    def infer_bin_op(self, left_type: LLType, right_type: LLType, op_type: type, node: ast.BinOp) -> LLType:
        """Infer type of binary operations with implicit type promotion."""

        # Handle pointer arithmetic (C/C++ style)
        # pointer + integer -> pointer
        # pointer - integer -> pointer
        # pointer - pointer -> ptrdiff_t (int64)
        if left_type.is_pointer():
            if isinstance(right_type, IntType):
                # pointer +/- integer -> pointer
                if op_type in (ast.Add, ast.Sub):
                    return left_type
            elif right_type.is_pointer():
                # pointer - pointer -> ptrdiff_t
                if op_type == ast.Sub:
                    return int64

        # integer + pointer -> pointer (commutative for Add)
        if right_type.is_pointer() and isinstance(left_type, IntType):
            if op_type == ast.Add:
                return right_type

        # Handle tensor arithmetic (tensor can be treated as pointer to element type)
        # tensor + integer -> pointer to element type
        # tensor - integer -> pointer to element type
        if left_type.is_tensor():
            if isinstance(right_type, IntType):
                if op_type in (ast.Add, ast.Sub):
                    # Return pointer to the tensor's element type
                    element_type = left_type.element_type
                    return Pointer(element_type)

        # integer + tensor -> pointer to element type (commutative for Add)
        if right_type.is_tensor() and isinstance(left_type, IntType):
            if op_type == ast.Add:
                element_type = right_type.element_type
                return Pointer(element_type)

        # Try to promote types
        promoted_left, promoted_right, was_promoted = promote_type(left_type, right_type)

        if promoted_left != promoted_right:
            raise TypeMismatchError(
                f"Binary operation requires compatible types: {left_type} (left) vs {right_type} (right). "
                f"Promotion attempted but types remain incompatible.", node=node)

        # Issue warning if promotion occurred (optional, can be logged)
        if was_promoted:
            # Could add warning here if needed
            pass

        handler = self.registry.get_bin_op_handler(op_type)
        if handler is None:
            raise UnsupportedNodeError(f"Unsupported binary operator: {op_type.__name__}", node=node)

        return handler(promoted_left, promoted_right)

    def infer_un_op(self, operand_type: LLType, op_type: type, node: ast.UnaryOp) -> LLType:
        """Infer type of unary operations."""
        handler = self.registry.get_un_op_handler(op_type)
        if handler is None:
            raise UnsupportedNodeError(f"Unsupported unary operator: {op_type.__name__}", node=node)

        return handler(operand_type)

    def infer_bool_op(self, op_type: type, node: ast.BoolOp) -> LLType:
        """Infer type of boolean operations."""
        handler = self.registry.get_bool_op_handler(op_type)
        if handler is None:
            raise UnsupportedNodeError(f"Unsupported boolean operator: {op_type.__name__}", node=node)

        return handler()
