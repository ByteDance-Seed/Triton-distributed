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
from typing import Dict, Any, Set, List, Callable, Optional

from .pass_base import CompleteASTMutator
from .utils.resolve_attribute import recursive_resolve_attribute
from .utils.preserve_attributes import create_node_with_attrs
from .utils.scope_manager import ScopeManager
from little_kernel.language.builtin_base import (
    INLINE_FUNC_ATTR,
    ORIGINAL_PY_FUNC_ATTR,
)


class FunctionInliner(CompleteASTMutator):
    """
    AST transformer that inlines function calls.
    Supports:
    - Simple positional arguments
    - Local variable renaming to avoid conflicts
    - Return value substitution
    
    Uses the unified ScopeManager to properly resolve symbols:
    - Local (kernel-defined) variables are not resolved from ctx
    - Only global variables are resolved from ctx
    """

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.scope_manager = ScopeManager(ctx)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Process function definitions and manage scope."""
        # Enter new scope for this function
        self.scope_manager.enter_scope()
        # Define function parameters as local
        self.scope_manager.define_from_ast(node.args)
        # Scan body for local variable assignments
        self.scope_manager.scan_for_locals(node.body)

        # Process the function using parent class logic
        result = super().visit_FunctionDef(node)

        # Exit scope
        self.scope_manager.exit_scope()
        return result

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        """Handle side-effect function calls (e.g., 'func2(a, b)' as a standalone statement)"""
        if isinstance(node.value, ast.Call):
            call = node.value
            # Don't inline if this is a special struct method call
            if hasattr(call, '_is_special_struct_method') and call._is_special_struct_method:
                return super().visit_Expr(node)
            func = recursive_resolve_attribute(call.func, self.ctx, self.scope_manager)
            if isinstance(func, Callable):
                if hasattr(func, INLINE_FUNC_ATTR) and getattr(func, INLINE_FUNC_ATTR, False):
                    inline_func = getattr(func, INLINE_FUNC_ATTR)
                    return self._inline_call(call, inline_func, is_statement=True)
        return super().visit_Expr(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Handle function calls in expressions (e.g., 'cdiv(a, b)' in expressions)."""
        # Don't inline if this is a special struct method call
        if hasattr(node, '_is_special_struct_method') and node._is_special_struct_method:
            # Process arguments recursively (they might contain calls to inline)
            node_copy = self._copy_node(node)
            node_copy.func = self.visit(node.func)
            node_copy.args = self._process_list(node.args)
            node_copy.keywords = self._process_list(node.keywords)
            return node_copy

        # Try to resolve and inline the function
        try:
            func = recursive_resolve_attribute(node.func, self.ctx, self.scope_manager)
            if isinstance(func, Callable):
                if hasattr(func, INLINE_FUNC_ATTR) and getattr(func, INLINE_FUNC_ATTR, False):
                    original_func = getattr(func, ORIGINAL_PY_FUNC_ATTR, func)
                    # Inline the call - this will return a list of statements
                    # For expression context, we need to extract the return value
                    inlined_body = self._inline_call(node, original_func, is_statement=False)
                    if inlined_body:
                        # Find the return statement or Expr containing the return value
                        for stmt in inlined_body:
                            if isinstance(stmt, ast.Return) and stmt.value:
                                # Return the value from the return statement (recursively visit it)
                                return self.visit(stmt.value)
                            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.BinOp):
                                # _inline_call converts return to Expr when is_statement=False and no lhs
                                # The value is in stmt.value
                                return self.visit(stmt.value)
                        # If no return found, check for assignments (shouldn't happen in expression context)
                        for stmt in inlined_body:
                            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                                # This shouldn't happen, but handle it anyway
                                return self.visit(stmt.value)
        except Exception:
            # If resolution fails, continue with normal processing
            # Debug: uncomment to see inline errors
            # print(f"Warning: inline failed for {node.func if isinstance(node.func, ast.Name) else 'call'}: {e}\n{traceback.format_exc()}")
            pass

        # Process arguments recursively (they might contain calls to inline)
        node_copy = self._copy_node(node)
        node_copy.func = self.visit(node.func)
        node_copy.args = self._process_list(node.args)
        node_copy.keywords = self._process_list(node.keywords)
        return node_copy

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handle function calls with return values (e.g., 'x = func2(a, b)')"""
        if isinstance(node.value, ast.Call):
            call = node.value
            # Don't inline if this is a special struct method call
            if hasattr(call, '_is_special_struct_method') and call._is_special_struct_method:
                return super().visit_Assign(node)
            func = recursive_resolve_attribute(call.func, self.ctx, self.scope_manager)
            if isinstance(func, Callable):
                if hasattr(func, INLINE_FUNC_ATTR) and getattr(func, INLINE_FUNC_ATTR, False):
                    original_func = getattr(func, ORIGINAL_PY_FUNC_ATTR, func)
                    return self._inline_call(call, original_func, is_statement=False, lhs=node.targets[0])
        return super().visit_Assign(node)

    def _inline_call(self, call_node: ast.Call, func: Callable, is_statement: bool,
                     lhs: Optional[ast.expr] = None) -> List[ast.stmt]:
        """
        Core inlining logic:
        - is_statement: True if call is standalone (e.g., 'func()'), False if assigned (e.g., 'x = func()')
        - lhs: Left-hand side of assignment (for return value)
        """
        signature = inspect.signature(func)
        source = inspect.getsource(func)
        # Remove indentation for functions defined in non-global scope
        source = textwrap.dedent(source)
        func_ast = ast.parse(source)
        assert len(func_ast.body) == 1, "Function body must contain exactly one statement"

        # 1. Resolve arguments (map call args to function parameters)
        param_map = self._map_arguments(call_node, signature)

        # 2. Resolve variable conflicts (rename inlined locals to avoid clashes)
        rename_map = self._resolve_conflicts(call_node, func_ast.body[0], signature)

        # 3. Substitute parameters in the target function's body
        substituted_body = self._substitute_parameters(func_ast.body[0].body, param_map, rename_map)

        # 4. Handle return statements (replace with assignment if needed)
        if not is_statement and lhs:
            processed_body = self._replace_returns(substituted_body, lhs)
        else:
            if isinstance(substituted_body[-1], ast.Return):
                substituted_body[-1] = ast.Expr(value=substituted_body[-1].value)
            processed_body = substituted_body  # Keep returns as-is for side-effect functions

        return processed_body

    def _map_arguments(self, call_node: ast.Call, signature) -> Dict[str, ast.expr]:
        """Map call arguments (positional/keyword) to function parameters"""
        param_names = list(signature.parameters.keys())
        arg_map: Dict[str, ast.expr] = {}

        # Handle positional arguments
        for i, arg in enumerate(call_node.args):
            if i >= len(param_names):
                raise RuntimeError(f"Too many positional args for {signature}")
            arg_map[param_names[i]] = arg

        # Handle keyword arguments
        for kw in call_node.keywords:
            if kw.arg not in param_names:
                raise RuntimeError(f"Unexpected keyword arg {kw.arg} for {signature}")
            arg_map[kw.arg] = kw.value

        # Check for missing required parameters
        for param in signature.parameters.values():
            if param.default is inspect.Parameter.empty and param.name not in arg_map:
                raise RuntimeError(f"Missing required arg {param.name} for {signature}")

        return arg_map

    def _resolve_conflicts(self, call_context: ast.AST, target_ast: ast.AST, signature) -> Dict[str, str]:
        """
        Rename local variables in inlined function to avoid conflicts with:
        - Variables in the calling context
        - Parameters of the calling function
        """
        # Collect variables in the calling context
        context_vars = self._collect_variables(call_context)

        # Collect local variables in the target function (excluding its parameters)
        target_params = {p.arg for p in target_ast.args.args}
        target_locals = self._collect_local_vars(target_ast) - target_params

        # Generate rename map for conflicting variables
        rename_map = {}
        for var in target_locals:
            if var in context_vars:
                # Rename with prefix to avoid conflict (e.g., "tmp" → "func2_tmp")
                rename_map[var] = f"{signature.name}_{var}"
        return rename_map

    def _substitute_parameters(self, body: List[ast.stmt], param_map: Dict[str, ast.expr],
                               rename_map: Dict[str, str]) -> List[ast.stmt]:
        """Replace function parameters with call arguments and rename conflicting variables"""

        class ParamSubstituter(ast.NodeTransformer):

            def __init__(self, param_map: Dict[str, ast.expr], rename_map: Dict[str, str]):
                self.param_map = param_map
                self.rename_map = rename_map

            def visit_Name(self, node: ast.Name) -> ast.AST:
                # Replace parameters with actual arguments
                if node.id in self.param_map and isinstance(node.ctx, ast.Load):
                    return ast.copy_location(ast.parse(ast.unparse(self.param_map[node.id])).body[0].value, node)
                # Rename conflicting local variables
                if node.id in self.rename_map:
                    node.id = self.rename_map[node.id]
                return node

        substituter = ParamSubstituter(param_map, rename_map)
        return [substituter.visit(stmt) for stmt in body]

    def _replace_returns(self, body: List[ast.stmt], lhs: ast.expr) -> List[ast.stmt]:
        """Replace return statements with assignments to the LHS variable (for functions with return values)"""

        class ReturnReplacer(ast.NodeTransformer):

            def __init__(self, lhs: ast.expr):
                self.lhs = lhs

            def visit_Return(self, node: ast.Return) -> ast.AST:
                if node.value is None:
                    return None  # Ignore empty returns
                # Replace "return value" with "lhs = value"
                return create_node_with_attrs(ast.Assign, node, targets=[self.lhs], value=node.value)

        replacer = ReturnReplacer(lhs)
        return [replacer.visit(stmt) for stmt in body]

    def _collect_variables(self, node: ast.AST) -> Set[str]:
        """Collect all variable names in the calling context"""
        vars = set()

        class VarCollector(ast.NodeVisitor):

            def visit_Name(self, n: ast.Name) -> None:
                if isinstance(n.ctx, (ast.Load, ast.Store)):
                    vars.add(n.id)

        VarCollector().visit(node)
        return vars

    def _collect_local_vars(self, func_ast: ast.FunctionDef) -> Set[str]:
        """Collect local variables defined in the target function"""
        locals = set()

        class LocalVarCollector(ast.NodeVisitor):

            def visit_Name(self, n: ast.Name) -> None:
                if isinstance(n.ctx, ast.Store):  # Variables being assigned to
                    locals.add(n.id)

        LocalVarCollector().visit(func_ast)
        return locals


def inline(tree: ast.AST, ctx: Dict[str, Any]) -> ast.AST:
    inliner = FunctionInliner(ctx)
    new_ast = inliner.visit(tree)
    return new_ast
