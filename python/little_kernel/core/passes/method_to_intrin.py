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
Pass to convert method-style calls to intrin function calls.

Converts:
    obj.to(dtype) -> ll.to(obj, dtype)
"""

import ast
import little_kernel.language as ll
from typing import Dict, Any
from .pass_base import CompleteASTMutator
from .utils.resolve_attribute import recursive_resolve_attribute
from .utils.update_context import update_context
from .utils.scope_manager import ScopeManager
from .utils.preserve_attributes import copy_node_with_attrs


class MethodToIntrinTransformer(CompleteASTMutator):
    """
    Transform method-style calls to intrin function calls.
    
    Example:
        scheduler.current_shape_k.to(ll.int64) -> ll.to(scheduler.current_shape_k, ll.int64)
    """

    def __init__(self, ctx: Dict[str, Any] = None):
        super().__init__()
        self.ctx = ctx if ctx is not None else {}
        self.scope_manager = ScopeManager(self.ctx)
        # Use a private name to avoid conflicts with user code
        # update_context will handle name conflicts by appending underscores
        self._ll_module_name = update_context(self.ctx, "_method_to_intrin_ll", ll)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.scope_manager.enter_scope()
        self.scope_manager.define_from_ast(node.args)
        self.scope_manager.scan_for_locals(node.body)

        # Manual copy of super().visit_FunctionDef logic because we need to wrap execution
        node_copy = self._copy_node(node)
        node_copy.name = node.name  # Preserve unless mutated
        node_copy.args = self.visit(node.args)
        node_copy.body = self._process_list(node.body)
        node_copy.decorator_list = self._process_list(node.decorator_list)
        node_copy.returns = self.visit(node.returns)
        node_copy.type_comment = node.type_comment  # Primitive, no mutation

        self.scope_manager.exit_scope()
        return node_copy

    def visit_Call(self, node: ast.Call) -> ast.AST:
        # Check if this is a method call: obj.to(dtype)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'to':
            # Check if this is already ll.to(...) by trying to resolve it
            # Use the context to resolve, which includes our private module name
            try:
                # Try to resolve the function to see if it's already ll.to
                resolved_func = recursive_resolve_attribute(node.func, self.ctx, self.scope_manager)
                if resolved_func is not None and hasattr(ll, 'to') and resolved_func is getattr(ll, 'to'):
                    # Already ll.to(...), process normally
                    return super().visit_Call(node)
            except Exception:
                # If resolution fails, continue with conversion
                pass

            # This is obj.to(...) where obj is not ll (or resolution failed)
            # Convert to ll.to(obj, ...)
            obj = node.func.value

            # Visit the object and arguments first
            new_obj = self.visit(obj)
            new_args = self._process_list(node.args)
            new_keywords = self._process_list(node.keywords)

            # Create ll.to call using the private module name from context
            ll_name = ast.Name(id=self._ll_module_name, ctx=ast.Load())
            ast.copy_location(ll_name, node.func)
            ll_to_attr = ast.Attribute(value=ll_name, attr='to', ctx=ast.Load())
            ast.copy_location(ll_to_attr, node.func)

            # ll.to(obj, *args, **kwargs)
            new_call = ast.Call(func=ll_to_attr, args=[new_obj] + new_args, keywords=new_keywords)
            ast.copy_location(new_call, node)
            # Preserve any special attributes from the original node
            return copy_node_with_attrs(node, new_call)

        # Not a .to() call, process normally
        return super().visit_Call(node)


def method_to_intrin(tree: ast.Module, ctx: dict = None) -> ast.Module:
    """
    Convert method-style calls to intrin function calls.
    
    Args:
        tree: AST module to transform
        ctx: Context dictionary (will be updated with private module name if needed)
    
    Returns:
        Transformed AST module
    """
    if ctx is None:
        ctx = {}
    transformer = MethodToIntrinTransformer(ctx)
    return transformer.visit(tree)
