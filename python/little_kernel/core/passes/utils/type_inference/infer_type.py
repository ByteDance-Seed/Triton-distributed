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
from typing import Dict, List, Optional, Any
from little_kernel.core.type_system import LLType
from .type_inference_core import (TypeInferenceError, get_variable_type_from_scope)
from ..scope_manager import ScopeManager
from .type_inference_visitors import AnnotationResolver
from .type_inference_ast_visitors import ASTNodeVisitors
from .type_inference_statements import StatementTypeInferencer
from .type_inference_call import CallTypeInferencer


class TypeInferencer(ast.NodeVisitor):
    """
    Centralized type inferencer for LLType system.
    Uses modular components for maintainability and extensibility.
    
    Uses the unified ScopeManager to properly handle symbol resolution:
    - Local (kernel-defined) symbols take priority over global (ctx) symbols
    - Prevents incorrect constant folding when local variables shadow global ones
    """

    def __init__(self, ctx: Dict[str, Any], scope_vars):
        """
        Initialize type inferencer with global context.
        
        Args:
            ctx: Global namespace (e.g., function.__globals__) with LLType imports/constants
            scope_vars: Initial scope variables (for backwards compatibility)
        """
        self.ctx = ctx
        # Use the unified ScopeManager - pass ctx as global context
        self.scope_manager = ScopeManager(ctx)
        # Enter initial scope with provided scope_vars types
        self.scope_manager.enter_scope(scope_vars if scope_vars is not None else {})
        self.annotation_resolver = AnnotationResolver(ctx, self._resolve_annotation, self.scope_manager)
        self.all_types: Dict[ast.AST, LLType] = {}

        # Initialize modular visitors
        self.ast_visitors = ASTNodeVisitors(self)
        self.statement_inferencer = StatementTypeInferencer(self)

    # ------------------------------ Scope Management (delegated to ScopeManager) ------------------------------
    @property
    def current_scope(self) -> Dict[str, LLType]:
        """Get the active scope's type map (for backwards compatibility)"""
        return self.scope_manager.current_scope_types

    @property
    def scope_stack(self) -> List[Dict[str, LLType]]:
        """Get the scope stack's type maps (for backwards compatibility)"""
        # Return list of type maps from each scope
        return [scope[2] for scope in self.scope_manager.scope_stack]

    def push_scope(self) -> None:
        """Create a new nested scope (e.g., for function bodies, loops)"""
        self.scope_manager.enter_scope()

    def pop_scope(self) -> None:
        """Remove the active scope (e.g., when exiting a function/loop)"""
        self.scope_manager.exit_scope()

    def add_variable_to_scope(self, name: str, type_: LLType, node: Optional[ast.AST] = None) -> None:
        """Add a variable to the active scope"""
        # Define the name as local and set its type
        self.scope_manager.define(name)
        self.scope_manager.set_variable_type(name, type_)

    def get_variable_type(self, name: str, node: ast.AST) -> LLType:
        """Get type of a variable from scope/context."""
        return get_variable_type_from_scope(self.scope_manager, name, node, self.ctx)

    # ------------------------------ Annotation Resolution ------------------------------
    def _resolve_annotation(self, ann_node: ast.AST) -> LLType:
        """Evaluate an AST annotation node to LLType."""
        return self.annotation_resolver.resolve_annotation(ann_node)

    # ------------------------------ AST Node Visitors ------------------------------
    def visit(self, node):
        """Override visit to cache results."""
        if node in self.all_types:
            return self.all_types[node]
        if node is None:
            return None
        ret = super().visit(node)
        if ret is not None:
            self.all_types[node] = ret
        return ret

    def visit_Module(self, node: ast.Module) -> None:
        """Process top-level module"""
        self.ast_visitors.visit_Module(node)

    def visit_Constant(self, node: ast.Constant) -> LLType:
        """Infer type of constants"""
        return self.ast_visitors.visit_Constant(node)

    def visit_Num(self, node: ast.Num) -> LLType:
        """Infer type of numeric constants (Python 3.7 and earlier)"""
        # Convert ast.Num to ast.Constant for compatibility
        const_node = ast.Constant(value=node.n)
        return self.ast_visitors.visit_Constant(const_node)

    def visit_Str(self, node: ast.Str) -> LLType:
        """Infer type of string constants (Python 3.7 and earlier)"""
        return self.ast_visitors.visit_Str(node)

    def visit_Name(self, node: ast.Name) -> LLType:
        """Infer type of variable references"""
        return self.ast_visitors.visit_Name(node)

    def visit_BinOp(self, node: ast.BinOp) -> LLType:
        """Infer type of binary operations"""
        return self.ast_visitors.visit_BinOp(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> LLType:
        """Infer type of unary operations"""
        return self.ast_visitors.visit_UnaryOp(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> LLType:
        """Infer type of boolean operations"""
        return self.ast_visitors.visit_BoolOp(node)

    def visit_Compare(self, node: ast.Compare) -> LLType:
        """Infer type of comparison operations"""
        return self.ast_visitors.visit_Compare(node)

    def visit_Subscript(self, node: ast.Subscript) -> LLType:
        """Infer type of subscript expressions"""
        return self.ast_visitors.visit_Subscript(node)

    def visit_List(self, node: ast.List) -> LLType:
        """Infer type of lists"""
        return self.ast_visitors.visit_List(node)

    def visit_Tuple(self, node: ast.Tuple) -> LLType:
        """Infer type of tuples"""
        return self.ast_visitors.visit_Tuple(node)

    def visit_ListComp(self, node: ast.ListComp) -> LLType:
        """Infer type of list comprehensions"""
        return self.ast_visitors.visit_ListComp(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Process expression statements"""
        self.ast_visitors.visit_Expr(node)

    def visit_Return(self, node: ast.Return) -> None:
        """Process return statements"""
        self.ast_visitors.visit_Return(node)

    def visit_IfExp(self, node: ast.IfExp) -> LLType:
        """Infer type of conditional expressions"""
        return self.ast_visitors.visit_IfExp(node)

    def visit_Attribute(self, node: ast.Attribute) -> LLType:
        """Infer type of attribute access"""
        return self.ast_visitors.visit_Attribute(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions"""
        self.statement_inferencer.visit_FunctionDef(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Process annotated assignments"""
        self.statement_inferencer.visit_AnnAssign(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Process assignment statements"""
        self.statement_inferencer.visit_Assign(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Process augmented assignment statements (e.g., x += 1)"""
        self.statement_inferencer.visit_AugAssign(node)

    def visit_For(self, node: ast.For) -> None:
        """Process for loops"""
        self.statement_inferencer.visit_For(node)

    def visit_Call(self, node: ast.Call) -> LLType:
        """Infer type of function calls"""
        call_inferencer = CallTypeInferencer(self.ctx, self.scope_manager, self._resolve_annotation,
                                             self.infer_node_type, self.all_types)
        return call_inferencer.infer_call_type(node)

    # ------------------------------ Public API ------------------------------
    def infer_node_type(self, node: ast.AST) -> LLType:
        """
        Public method to infer type of a single AST node.
        Delegates to the appropriate visit_* method.
        """
        if node in self.all_types:
            return self.all_types[node]
        result = self.visit(node)
        if not isinstance(result, LLType):
            raise TypeInferenceError(
                f"Node inference returned non-LLType: {type(result).__name__} (node: {ast.dump(node)})", node=node)
        self.all_types[node] = result
        return result

    def get_inferred_scopes(self) -> List[Dict[str, LLType]]:
        """Get all scopes with inferred variable types (for debugging/validation)"""
        return self.scope_stack.copy()


# ------------------------------ Top-Level API ------------------------------
def infer_type(tree: ast.AST, ctx: Dict[str, Any], scope_vars=None) -> Dict[str, LLType]:
    """
    Top-level function to run type inference on an AST.
    
    Args:
        tree: AST to infer types for
        ctx: Global context (e.g., function.__globals__)
        scope_vars: Initial scope variables (optional)
    
    Returns:
        Global scope with inferred variable/function types
    """
    scope_vars = scope_vars if scope_vars is not None else {}
    inferencer = TypeInferencer(ctx, scope_vars)
    inferencer.visit(tree)
    # Return global scope (first element of scope stack) - the type map
    if inferencer.scope_stack:
        return inferencer.scope_stack[0]
    return {}
