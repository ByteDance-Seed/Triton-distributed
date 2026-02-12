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
from typing import Dict, Callable, Optional
from little_kernel.core.type_system import LLType, bool_


class OperatorRegistry:
    """Registry for binary, unary, and boolean operators with type inference handlers."""

    def __init__(self):
        self._bin_op_handlers: Dict[type, Callable[[LLType, LLType], LLType]] = {}
        self._un_op_handlers: Dict[type, Callable[[LLType], LLType]] = {}
        self._bool_op_handlers: Dict[type, Callable[[], LLType]] = {}
        self._register_default_operators()

    def _register_default_operators(self):
        """Register default operators that return the left operand type."""
        # Binary operators: most return the type of the left operand
        for op_type in [
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift, ast.BitOr,
                ast.BitXor, ast.BitAnd, ast.MatMult
        ]:
            self.register_bin_op(op_type, lambda a, b: a)

        # Unary operators
        self.register_un_op(ast.UAdd, lambda a: a)
        self.register_un_op(ast.USub, lambda a: a)
        self.register_un_op(ast.Invert, lambda a: a)
        self.register_un_op(ast.Not, lambda a: bool_)  # Logical NOT always returns bool

        # Boolean operators
        self.register_bool_op(ast.And, lambda: bool_)
        self.register_bool_op(ast.Or, lambda: bool_)

    def register_bin_op(self, op_type: type, handler: Callable[[LLType, LLType], LLType]):
        """Register a binary operator handler."""
        self._bin_op_handlers[op_type] = handler

    def register_un_op(self, op_type: type, handler: Callable[[LLType], LLType]):
        """Register a unary operator handler."""
        self._un_op_handlers[op_type] = handler

    def register_bool_op(self, op_type: type, handler: Callable[[], LLType]):
        """Register a boolean operator handler."""
        self._bool_op_handlers[op_type] = handler

    def get_bin_op_handler(self, op_type: type) -> Optional[Callable[[LLType, LLType], LLType]]:
        """Get handler for a binary operator."""
        return self._bin_op_handlers.get(op_type)

    def get_un_op_handler(self, op_type: type) -> Optional[Callable[[LLType], LLType]]:
        """Get handler for a unary operator."""
        return self._un_op_handlers.get(op_type)

    def get_bool_op_handler(self, op_type: type) -> Optional[Callable[[], LLType]]:
        """Get handler for a boolean operator."""
        return self._bool_op_handlers.get(op_type)


# Global registry instance
_operator_registry = OperatorRegistry()


def get_operator_registry() -> OperatorRegistry:
    """Get the global operator registry."""
    return _operator_registry
