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
import sys
from typing import Dict, Optional, Any
from little_kernel.core.type_system import LLType
from ..error_report import format_error_message
# Import the unified ScopeManager
from ..scope_manager import ScopeManager


class TypeInferenceError(RuntimeError):
    """Base exception for type inference failures (with AST node context)"""

    def __init__(self, message: str, node: Optional[ast.AST] = None, ctx: Optional[Dict[str, Any]] = None):
        self.node = node
        if ctx is None:
            try:
                frame = sys._getframe(1)
                if 'self' in frame.f_locals:
                    inferencer = frame.f_locals['self']
                    if hasattr(inferencer, 'ctx'):
                        ctx = inferencer.ctx
            except (ValueError, AttributeError, KeyError):
                pass
        formatted_message = format_error_message(message, node, None, ctx)
        super().__init__(formatted_message)


class UndefinedVariableError(TypeInferenceError):
    """Raised when referencing a variable not in scope/context"""
    pass


class TypeMismatchError(TypeInferenceError):
    """Raised when types are incompatible (e.g., int + float)"""
    pass


class UnsupportedNodeError(TypeInferenceError):
    """Raised when an AST node type is not supported for inference"""
    pass


def get_variable_type_from_scope(scope_manager: ScopeManager, name: str, node: ast.AST, ctx: Dict[str, Any]) -> LLType:
    """
    Get type of a variable from scope/context using the unified ScopeManager.
    Raises UndefinedVariableError if not found.
    
    Priority:
    1. Check if name is local (defined in kernel) - get type from scope_manager
    2. Check global context (imports, top-level constants)
    """
    # Check if name is defined locally (in kernel)
    if scope_manager.is_local(name):
        # Try to get type from scope_manager's type tracking
        var_type = scope_manager.get_variable_type(name)
        if var_type is not None:
            return var_type
        # If we know it's local but don't have a type yet, it might be defined later
        # or inferred from usage. For now, we raise an error.
        raise UndefinedVariableError(f"Local variable '{name}' has no type annotation or inferred type yet", node=node)

    # Not local, check global context (imports, top-level constants)
    if name in ctx:
        ctx_value = ctx[name]
        if isinstance(ctx_value, LLType):
            return ctx_value
        elif isinstance(ctx_value, (bool, int, float, str)):
            # Note: bool must come before int in isinstance check since bool is subclass of int
            # Import here to avoid circular dependency
            from .type_inference_visitors import ConstantTypeInferencer
            inferencer = ConstantTypeInferencer()
            return inferencer.infer_constant_type(ast.Constant(value=ctx_value))
        elif isinstance(ctx_value, type(ast)) or hasattr(ctx_value, '__module__'):
            # Allow modules (like 'll') - they will be handled by visit_Attribute
            # when accessing attributes like ll.int64
            raise UndefinedVariableError(
                f"Global context value '{name}' is a module, access attributes like '{name}.int64' instead", node=node)
        else:
            raise UndefinedVariableError(
                f"Global context value '{name}' is not an LLType or primitive (got {type(ctx_value).__name__})",
                node=node)

    raise UndefinedVariableError(f"Variable '{name}' is not declared in scope or global context", node=node)
