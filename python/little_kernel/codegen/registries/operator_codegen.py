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
from typing import Dict


class OperatorCodegenRegistry:
    """Registry for mapping Python AST operators to C++ operators."""

    def __init__(self):
        self._bin_op_map: Dict[type, str] = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "/",
            ast.Mod: "%",
            ast.LShift: "<<",
            ast.RShift: ">>",
            ast.BitOr: "|",
            ast.BitXor: "^",
            ast.BitAnd: "&",
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
            ast.Is: "==",
            ast.IsNot: "!=",
        }
        self._un_op_map: Dict[type, str] = {
            ast.Not: "!",
            ast.USub: "-",
            ast.UAdd: "+",
            ast.Invert: "~",
        }
        self._bool_op_map: Dict[type, str] = {
            ast.And: "&&",
            ast.Or: "||",
        }

    def register_bin_op(self, op_type: type, cpp_op: str):
        """Register a binary operator mapping."""
        self._bin_op_map[op_type] = cpp_op

    def register_un_op(self, op_type: type, cpp_op: str):
        """Register a unary operator mapping."""
        self._un_op_map[op_type] = cpp_op

    def register_bool_op(self, op_type: type, cpp_op: str):
        """Register a boolean operator mapping."""
        self._bool_op_map[op_type] = cpp_op

    def get_bin_op(self, op: ast.operator) -> str:
        """Get C++ operator string for binary operator."""
        op_type = type(op)
        if op_type in self._bin_op_map:
            return self._bin_op_map[op_type]
        raise NotImplementedError(f"Unsupported operator: {op_type.__name__}")

    def get_un_op(self, op: ast.unaryop) -> str:
        """Get C++ operator string for unary operator."""
        op_type = type(op)
        if op_type in self._un_op_map:
            return self._un_op_map[op_type]
        raise NotImplementedError(f"Unsupported unary operator: {op_type.__name__}")

    def get_bool_op(self, op: ast.boolop) -> str:
        """Get C++ operator string for boolean operator."""
        op_type = type(op)
        if op_type in self._bool_op_map:
            return self._bool_op_map[op_type]
        raise NotImplementedError(f"Unsupported boolean operator: {op_type.__name__}")


# Global registry instance
_operator_codegen_registry = OperatorCodegenRegistry()


def get_operator_codegen_registry() -> OperatorCodegenRegistry:
    """Get the global operator codegen registry."""
    return _operator_codegen_registry
