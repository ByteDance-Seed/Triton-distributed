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
Expression codegen visitors for CppEmitter.

This module contains visitor methods for AST expression nodes that return C++ code strings.
"""

import ast
from enum import Enum
from typing import TYPE_CHECKING
import little_kernel.language as ll
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.passes.utils.error_report import format_error_message
from ..registries.operator_codegen import get_operator_codegen_registry


# Import PASSES lazily to avoid circular import
def _get_passes():
    from little_kernel.core.passes import PASSES
    return PASSES


if TYPE_CHECKING:
    pass


class ExpressionCodegenMixin:
    """Mixin class for expression codegen methods."""

    def _is_local_variable(self, name: str) -> bool:
        """
        Check if a variable name is defined locally (should not be resolved from ctx).
        
        Supports both:
        - scope_vars (Dict[str, LLType]) used by codegen
        - scope_manager (ScopeManager) used by passes
        
        This unified check ensures consistent symbol resolution across all components.
        """
        # Check scope_vars (codegen style)
        if hasattr(self, 'scope_vars') and name in self.scope_vars:
            return True
        # Check scope_manager (pass style)
        if hasattr(self, 'scope_manager') and self.scope_manager.is_local(name):
            return True
        return False

    def visit_Name(self, node: ast.Name) -> str:
        """Process variable/identifier reference (return as-is for C++)."""
        # First check if variable is defined locally
        # Local variables should NOT be resolved from ctx
        if self._is_local_variable(node.id):
            return node.id

        # Not a local variable, check ctx for constant folding
        if node.id in self.ctx:
            ctx_value = self.ctx[node.id]
            # If it's a constant value, convert it to C++ string directly
            if isinstance(ctx_value, (int, float, bool)):
                return str(ctx_value)
            elif isinstance(ctx_value, str):
                return f'"{ctx_value}"'
            # Otherwise, return the variable name
        return node.id

    def visit_Constant(self, node: ast.Constant) -> str:
        """Process constants (map to C++ syntax)."""
        value = node.value

        # Handle None values
        if value is None:
            # Check if this None is used in a pointer context
            # Try to get the type from all_types
            node_type = None
            if node in self.all_types:
                node_type = self.all_types[node]

            # If we have type information, check if it's a pointer
            if node_type is not None:
                if node_type.is_pointer() or node_type.is_tensor():
                    # None for pointer/tensor types → nullptr
                    return "nullptr"
                else:
                    # None for non-pointer types is invalid
                    raise ValueError(f"None value is not allowed for non-pointer type {node_type}. "
                                     f"None can only be used for pointer types (will be converted to nullptr). "
                                     f"Node: {ast.dump(node, indent=2)}")
            else:
                # No type information available - this is ambiguous
                raise ValueError(f"None value encountered without type information. "
                                 f"None can only be used for pointer types (will be converted to nullptr). "
                                 f"Please ensure type information is available, or use explicit type annotations. "
                                 f"Node: {ast.dump(node, indent=2)}")

        if isinstance(value, str):
            if value.startswith("0x"):
                return value  # C++ hex literals (e.g., 0x1234)
            return f'"{value}"'  # C++ string literals (double quotes)
        elif isinstance(value, bool):
            return "true" if value else "false"  # C++ lowercase bools
        elif isinstance(value, int):
            # Try to preserve hex format if the original source was hex
            # Python 3.8+ AST nodes have a 'kind' attribute for hex literals
            if hasattr(node, 'kind') and node.kind == 'hex':
                return hex(value)
            # Note: AST doesn't preserve the original format (e.g., 0x1234 becomes 4660)
            # Without access to the original source code, we can't reliably determine
            # if a value was originally written in hex format.
            # The test expects 0x1234, but we only have the integer value 4660.
            # This is a limitation of using AST - we lose the original format information.
            # For now, we'll output the decimal representation.
            # To properly support this, we would need to:
            # 1. Store original source code alongside AST nodes, or
            # 2. Use a custom parser that preserves format information, or
            # 3. Use tokenize to extract format from source (requires source access)
            return str(value)
        elif isinstance(value, float):
            return str(value)
        elif isinstance(value, ll.LLType):
            return self._lltype_to_cpp(value)
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(value).__name__} (value: {value})")

    def visit_Num(self, node: ast.Num) -> str:
        """Process numeric constants (Python 3.7 and earlier)."""
        # Convert ast.Num to ast.Constant for compatibility
        const_node = ast.Constant(value=node.n)
        return self.visit_Constant(const_node)

    def visit_Str(self, node: ast.Str) -> str:
        """Process string constants (Python 3.7 and earlier)."""
        # Convert ast.Str to ast.Constant for compatibility
        const_node = ast.Constant(value=node.s)
        return self.visit_Constant(const_node)

    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Process attribute access (e.g., obj.attr)."""
        # Check if this is an enum value access (e.g., GemmType.Normal, IndexType.MN)
        try:
            # Try to resolve the attribute to check if it's an enum value
            attr = recursive_resolve_attribute(node, self.ctx)

            # Check if resolved value is an Enum instance
            if isinstance(attr, Enum):
                enum_class = type(attr)
                enum_name = enum_class.__name__
                enum_member = attr.name
                # Track that this enum is used
                if hasattr(self, 'used_enums'):
                    self.used_enums.add(enum_name)
                # Convert to C++ enum value format: EnumType::EnumValue
                return f"{enum_name}::{enum_member}"

            # Check if the base is an Enum class and attr is an enum member
            base = recursive_resolve_attribute(node.value, self.ctx)
            if isinstance(base, type) and issubclass(base, Enum):
                enum_name = base.__name__
                # Track that this enum is used
                if hasattr(self, 'used_enums'):
                    self.used_enums.add(enum_name)
                # Convert to C++ enum value format: EnumType::EnumValue
                return f"{enum_name}::{node.attr}"
        except Exception:
            # If resolution fails, fall through to normal handling
            pass

        attr = recursive_resolve_attribute(node, self.ctx)
        if isinstance(attr, ast.AST):
            # If it's still an AST node, generate C++ code for it
            base_str = self.visit(node.value)
            return f"{base_str}.{node.attr}"
        # If it's a resolved value (constant), convert to C++ string
        if isinstance(attr, (int, float, bool)):
            return str(attr)
        elif isinstance(attr, str):
            return f'"{attr}"'
        # For other types (like LLType), generate attribute access
        base_str = self.visit(node.value)
        return f"{base_str}.{node.attr}"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Process binary operations (map to C++ operators)."""
        left_cpp = self.visit(node.left)
        op_cpp = self._get_operator_str(node.op)
        right_cpp = self.visit(node.right)
        return f"({left_cpp} {op_cpp} {right_cpp})"

    def _get_operator_str(self, op: ast.operator) -> str:
        """Map Python AST operators to C++ operators."""
        registry = get_operator_codegen_registry()
        return registry.get_bin_op(op)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Process unary operations (e.g., not, -x, +x, ~x)."""
        operand_cpp = self.visit(node.operand)
        registry = get_operator_codegen_registry()
        op_str = registry.get_un_op(node.op)
        return f"({op_str}{operand_cpp})"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """Process boolean operations (Python 'and'/'or' → C++ '&&'/'||')."""
        registry = get_operator_codegen_registry()
        op_str = registry.get_bool_op(node.op)

        # Process each sub-expression in the boolean operation
        sub_exprs = [self.visit(expr) for expr in node.values]

        # Join sub-expressions with the C++ operator, wrap in parentheses
        return f"({f' {op_str} '.join(sub_exprs)})"

    def visit_Compare(self, node: ast.Compare) -> str:
        """Process comparison expressions (e.g., `a < b <= c`)."""
        # Process leftmost expression (e.g., "a" in "a < b < c")
        left_expr = self.visit(node.left)
        sub_conditions = []

        # Iterate over comparison operators and right-hand side expressions
        for op, comparator in zip(node.ops, node.comparators):
            # Get C++ operator string (e.g., Lt → "<")
            op_str = self._get_operator_str(op)
            # Process right-hand side expression
            right_expr = self.visit(comparator)
            # Wrap sub-condition in parentheses to preserve precedence
            sub_conditions.append(f"({left_expr} {op_str} {right_expr})")
            # Update left_expr for next iteration (for chained comparisons)
            left_expr = right_expr

        # If there's only one comparison, return it directly
        if len(sub_conditions) == 1:
            return sub_conditions[0]
        else:
            # For chained comparisons, combine with logical AND
            # e.g., "a < b < c" → "(a < b) && (b < c)"
            return f"({' && '.join(sub_conditions)})"

    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Process Python ternary expressions (conditional expressions)."""
        condition_cpp = self.visit(node.test)
        true_expr_cpp = self.visit(node.body)
        false_expr_cpp = self.visit(node.orelse)
        return f"({condition_cpp} ? {true_expr_cpp} : {false_expr_cpp})"

    def visit_Subscript(self, node: ast.Subscript) -> str:
        """Process subscript/indexing operations (e.g., A[i], B[0])."""
        base_str = self.visit(node.value)

        # Handle different index types
        if isinstance(node.slice, ast.Index):
            # Python 3.8 and earlier: ast.Index wrapper
            index_elem = node.slice.value
        elif isinstance(node.slice, ast.Tuple):
            # Multi-dimensional indexing (e.g., A[i, j])
            indices = [self._process_single_index(elem) for elem in node.slice.elts]
            return f"{base_str}[{']['.join(indices)}]"
        else:
            # Python 3.9+: direct value
            index_elem = node.slice

        # Single index
        index_str = self._process_single_index(index_elem)
        return f"{base_str}[{index_str}]"

    def _process_single_index(self, index_elem: ast.AST) -> str:
        """Helper to process a single index element."""

        # Reject slice indices
        if isinstance(index_elem, ast.Slice):
            message = "Slice indices (e.g., A[1:5]) are not supported yet"
            formatted = format_error_message(message, index_elem, ctx=self.ctx)
            raise NotImplementedError(formatted)

        # Visit the index element to get its string representation
        index_str = self.visit(index_elem)

        # Validate the index type (should be integer-compatible)
        # For constants, infer type from the value
        if isinstance(index_elem, ast.Constant):
            if not isinstance(index_elem.value, int):
                message = (f"Subscript index must be a constant integer, "
                           f"got {type(index_elem.value).__name__}: {index_elem.value}")
                formatted = format_error_message(message, index_elem, ctx=self.ctx)
                raise TypeError(formatted)
        else:
            # For non-constants, check type if available
            try:
                index_type = self.get_type(index_elem)
                if not (isinstance(index_type, ll.IntType) and not index_type.special):
                    message = (f"Subscript index must be a regular integer type (e.g., int32, uint64), "
                               f"got {index_type}")
                    formatted = format_error_message(message, index_elem, ctx=self.ctx)
                    raise TypeError(formatted)
            except (AssertionError, KeyError):
                # If type is not available, assume it's valid (might be a constant or already validated)
                pass

        return index_str

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        """Process f-strings (JoinedStr) → C++ string concatenation."""
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                # String literal part
                parts.append(f'"{value.value}"')
            elif isinstance(value, ast.FormattedValue):
                # Formatted value (e.g., {BLOCK_M})
                formatted_value_cpp = self.visit(value.value)
                if not isinstance(formatted_value_cpp, str):
                    formatted_value_cpp = str(formatted_value_cpp)
                # Check if the result is already a string literal
                if formatted_value_cpp.startswith('"') and formatted_value_cpp.endswith('"'):
                    parts.append(formatted_value_cpp)
                else:
                    # Convert to string using std::to_string
                    parts.append(f'std::to_string({formatted_value_cpp})')
            else:
                # Other types - visit and convert to string
                formatted_value = self.visit(value)
                if isinstance(formatted_value,
                              str) and formatted_value.startswith('"') and formatted_value.endswith('"'):
                    parts.append(formatted_value)
                else:
                    parts.append(f'std::to_string({formatted_value})')

        if len(parts) == 1:
            return parts[0]
        else:
            # Concatenate strings in C++
            return " + ".join(parts)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> str:
        """Process formatted value in f-string (e.g., {variable})."""
        return self.visit(node.value)

    def visit_List(self, node: ast.List) -> str:
        """Process list literals (e.g., [1, 2, 3] or [ArraySize]).
        
        For code generation, lists are typically used for:
        1. Shape parameters (should be folded to constants by const_fold pass)
        2. Array initializations (converted to C++ array initialization)
        
        If all elements are constants, convert to C++ array initialization.
        Otherwise, this might indicate an issue with const folding.
        """
        elements = []
        for elt in node.elts:
            elt_str = self.visit(elt)
            elements.append(elt_str)

        # For now, return as comma-separated values
        # This is typically used for shape parameters that should have been folded
        # If we reach here, it means const_fold didn't fold the list
        return f"{{{', '.join(elements)}}}"
