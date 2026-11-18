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
from typing import Dict, Optional, Any
from little_kernel.core.type_system import LLType
from little_kernel.core.passes.utils.error_report import format_error_message
# Import the unified ScopeManager
from little_kernel.core.passes.utils.scope_manager import ScopeManager


class TypeInferenceError(RuntimeError):
    """Base exception for type inference failures (with AST node context).

    Diagnostic context is supplied explicitly: callers pass the failing AST
    node and, when available, the compilation context (used to locate the
    original kernel source).  The shared error formatter owns source
    resolution: it only accepts explicit ``source_lines`` or the registered
    compilation entry in ctx, and never inspects caller stack frames or
    picks an arbitrary callable, so an absent/empty ctx deterministically
    retains line/column/AST context with no unrelated snippet.
    """

    def __init__(self, message: str, node: Optional[ast.AST] = None, ctx: Optional[Dict[str, Any]] = None):
        self.node = node
        self.ctx = ctx
        # Keep the unformatted diagnosis so the owning inferencer can rebuild
        # the message later with its explicitly known compilation context.
        self.raw_message = message
        # The shared formatter sources snippets only from an explicit ctx
        # compilation entry or explicit source_lines -- never caller frames
        # -- so passing ctx here directly is sufficient and an absent/empty
        # ctx deterministically yields no snippet.
        formatted_message = format_error_message(message, node, None, ctx)
        super().__init__(formatted_message)

    def reformat_with_context(self, ctx: Dict[str, Any]) -> None:
        """Rebuild the formatted message once the owning context is known.

        Used by ``TypeInferencer`` at its single visit boundary so nested
        helpers can raise with a node alone while the inferencer attaches
        the compilation context explicitly.  Idempotent.
        """
        if ctx is None or self.ctx is not None:
            return
        self.ctx = ctx
        formatted_message = format_error_message(self.raw_message, self.node, None, ctx)
        self.args = (formatted_message, )


class UndefinedVariableError(TypeInferenceError):
    """Raised when referencing a variable not in scope/context"""
    pass


class TypeMismatchError(TypeInferenceError):
    """Raised when types are incompatible (e.g., int + float)"""
    pass


class UnsupportedNodeError(TypeInferenceError):
    """Raised when an AST node type is not supported for inference"""
    pass


def get_variable_type_from_scope(scope_manager: ScopeManager, name: str, node: ast.AST, ctx: Dict[str, Any], *,
                                 required: bool = True) -> Optional[LLType]:
    """
    Get type of a variable from scope/context using the unified ScopeManager.
    Raises UndefinedVariableError if not found, unless ``required`` is false.
    
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
        if not required:
            return None
        # If we know it's local but don't have a type yet, it might be defined later
        # or inferred from usage. For now, we raise an error.
        raise UndefinedVariableError(f"Local variable '{name}' has no type annotation or inferred type yet", node=node,
                                     ctx=ctx)

    # Not local, check global context (imports, top-level constants)
    if name in ctx:
        ctx_value = ctx[name]
        if isinstance(ctx_value, LLType):
            return ctx_value
        elif isinstance(ctx_value, (bool, int, float, str)):
            # Note: bool must come before int in isinstance check since bool is subclass of int
            # Import here to avoid circular dependency
            from little_kernel.core.passes.utils.type_inference.type_inference_visitors import ConstantTypeInferencer
            inferencer = ConstantTypeInferencer()
            return inferencer.infer_constant_type(ast.Constant(value=ctx_value))
        elif isinstance(ctx_value, type(ast)) or hasattr(ctx_value, '__module__'):
            # Allow modules (like 'll') - they will be handled by visit_Attribute
            # when accessing attributes like ll.int64
            if not required:
                return None
            raise UndefinedVariableError(
                f"Global context value '{name}' is a module, access attributes like '{name}.int64' instead", node=node,
                ctx=ctx)
        else:
            if not required:
                return None
            raise UndefinedVariableError(
                f"Global context value '{name}' is not an LLType or primitive (got {type(ctx_value).__name__})",
                node=node, ctx=ctx)

    if not required:
        return None
    raise UndefinedVariableError(f"Variable '{name}' is not declared in scope or global context", node=node, ctx=ctx)
