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
Unified ScopeManager for all passes.

This module provides a centralized symbol resolution mechanism that:
1. Prioritizes local (kernel-defined) symbols over global (ctx) symbols
2. Manages nested scopes (function bodies, loops, etc.)
3. Is shared by all passes (constfold, type inference, inline, etc.)

Key design principle: Local definitions shadow global definitions.
"""

import ast
from typing import Dict, Any, List, Set, Tuple, Optional


class LocalScanner(ast.NodeVisitor):
    """
    Scans AST nodes for local variable definitions.
    Used by ScopeManager to populate the local scope.
    
    This follows Python's semantics: any assignment to a name in a function
    makes that name local throughout the entire function, not just after
    the assignment.
    """

    def __init__(self, scope_manager: 'ScopeManager'):
        self.scope_manager = scope_manager

    def visit_Assign(self, node):
        for target in node.targets:
            self.scope_manager.define_from_ast(target)
        # Note: We do NOT visit values, as they don't define names in THIS scope.

    def visit_AnnAssign(self, node):
        self.scope_manager.define_from_ast(node.target)

    def visit_AugAssign(self, node):
        self.scope_manager.define_from_ast(node.target)

    def visit_For(self, node):
        self.scope_manager.define_from_ast(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        self.scope_manager.define_from_ast(node.target)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self.scope_manager.define_from_ast(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node):
        for item in node.items:
            if item.optional_vars:
                self.scope_manager.define_from_ast(item.optional_vars)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Function name IS a definition in the CURRENT scope.
        self.scope_manager.define(node.name)
        # Do NOT recurse into body - it's a new scope.

    def visit_AsyncFunctionDef(self, node):
        self.scope_manager.define(node.name)
        # Do NOT recurse into body.

    def visit_ClassDef(self, node):
        self.scope_manager.define(node.name)
        # Do NOT recurse into body.

    def visit_Global(self, node):
        for name in node.names:
            self.scope_manager.mark_global(name)

    def visit_Nonlocal(self, node):
        # Nonlocal means it refers to outer scope, so it is NOT local to this scope.
        # We treat it similar to global for the purpose of "is it defined LOCALLY?"
        # i.e. it is not a NEW definition that shadows outer ones, it IS the outer one.
        for name in node.names:
            self.scope_manager.mark_global(name)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split('.')[0]
            self.scope_manager.define(name)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.scope_manager.define(name)


class ScopeManager:
    """
    Unified symbol scope manager for all passes.
    
    Manages symbol scopes for passes, ensuring that local definitions 
    take priority over global context (ctx) definitions.
    
    Key features:
    - Supports nested scopes (function bodies, loops, etc.)
    - Tracks local variable definitions from assignments
    - Handles global/nonlocal declarations
    - Provides unified API for all passes to query symbol resolution
    
    Usage example:
        scope_manager = ScopeManager(ctx)
        
        # When entering a function
        scope_manager.enter_scope()
        scope_manager.define_from_ast(func_node.args)  # Add function parameters
        scope_manager.scan_for_locals(func_node.body)  # Scan for local assignments
        
        # Query if a name is local (shadows global)
        if scope_manager.is_local("x"):
            # x is defined locally, don't resolve from ctx
            pass
        else:
            # x is not local, can safely resolve from ctx
            value = scope_manager.resolve("x")
        
        # When exiting the function
        scope_manager.exit_scope()
    """

    def __init__(self, global_ctx: Dict[str, Any]):
        """
        Initialize ScopeManager with global context.
        
        Args:
            global_ctx: The global context dictionary (usually py_func.__globals__)
                        This contains imported modules, global variables, etc.
        """
        self.global_ctx = global_ctx
        # Stack of scopes. Each scope is (locals_set, explicit_globals_set, type_map)
        # - locals_set: Set of names defined locally in this scope
        # - explicit_globals_set: Set of names marked as global/nonlocal
        # - type_map: Optional dict mapping names to their types (for type inference)
        self.scope_stack: List[Tuple[Set[str], Set[str], Dict[str, Any]]] = []

    def enter_scope(self, initial_types: Optional[Dict[str, Any]] = None):
        """
        Enter a new scope (e.g. function body, loop body).
        
        Args:
            initial_types: Optional initial type mappings for this scope
        """
        type_map = initial_types if initial_types is not None else {}
        self.scope_stack.append((set(), set(), type_map))

    def exit_scope(self):
        """Exit the current scope."""
        if self.scope_stack:
            self.scope_stack.pop()

    def define(self, name: str):
        """
        Define a name in the current scope.
        
        This marks the name as locally defined, which means it shadows
        any global definition with the same name.
        """
        if self.scope_stack:
            locals_set, globals_set, _ = self.scope_stack[-1]
            # If it's marked as global/nonlocal, don't add to locals
            if name not in globals_set:
                locals_set.add(name)

    def mark_global(self, name: str):
        """
        Mark a name as global/nonlocal in the current scope.
        
        This means the name refers to the outer scope, not a local definition.
        """
        if self.scope_stack:
            locals_set, globals_set, _ = self.scope_stack[-1]
            globals_set.add(name)
            locals_set.discard(name)

    def define_from_ast(self, node: ast.AST):
        """
        Helper to define variables from AST nodes (targets of assignments, args, etc.)
        
        Handles:
        - ast.Name: Simple variable name
        - ast.Tuple/ast.List: Unpacking assignments
        - ast.arg: Function argument
        - ast.arguments: All function arguments
        - ast.Starred: Starred expression in unpacking
        """
        if isinstance(node, ast.Name):
            self.define(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                self.define_from_ast(elt)
        elif isinstance(node, ast.arg):
            self.define(node.arg)
        elif isinstance(node, ast.arguments):
            for arg in node.args:
                self.define_from_ast(arg)
            for arg in node.posonlyargs:
                self.define_from_ast(arg)
            for arg in node.kwonlyargs:
                self.define_from_ast(arg)
            if node.vararg:
                self.define_from_ast(node.vararg)
            if node.kwarg:
                self.define_from_ast(node.kwarg)
        elif isinstance(node, ast.Starred):
            self.define_from_ast(node.value)

    def scan_for_locals(self, nodes: List[ast.AST]):
        """
        Scans a list of statements for variable definitions and adds them to current scope.
        
        This follows Python's semantics: any assignment to a name anywhere in a 
        function makes that name local throughout the entire function.
        """
        visitor = LocalScanner(self)
        for node in nodes:
            visitor.visit(node)

    def is_local(self, name: str) -> bool:
        """
        Check if name is defined in any local scope (shadows global).
        
        This is the key method for determining symbol resolution priority:
        - If True: The name is defined locally, DO NOT resolve from ctx
        - If False: The name is not local, can safely resolve from ctx
        
        Returns:
            True if the name is defined in any scope on the stack (local to kernel)
            False if the name should be resolved from global context
        """
        for locals_set, _, _ in reversed(self.scope_stack):
            if name in locals_set:
                return True
        return False

    def resolve(self, name: str) -> Any:
        """
        Resolve a name to its value.
        
        Priority:
        1. If name is local (defined in kernel), return None (don't resolve)
        2. If name is in global context, return the value
        3. Otherwise return None
        
        Returns:
            The resolved value from global context, or None if:
            - The name is shadowed by a local definition, OR
            - The name is not found in global context
        """
        if self.is_local(name):
            return None  # Shadowed by local definition

        return self.global_ctx.get(name)

    def set_variable_type(self, name: str, type_value: Any):
        """
        Set the type of a variable in the current scope.
        
        Used by type inference to track inferred types.
        """
        if self.scope_stack:
            _, _, type_map = self.scope_stack[-1]
            type_map[name] = type_value

    def get_variable_type(self, name: str) -> Optional[Any]:
        """
        Get the type of a variable from any scope.
        
        Searches from innermost scope to outermost.
        
        Returns:
            The type if found, None otherwise
        """
        for _, _, type_map in reversed(self.scope_stack):
            if name in type_map:
                return type_map[name]
        return None

    @property
    def current_scope_types(self) -> Dict[str, Any]:
        """
        Get the type map of the current (innermost) scope.
        
        Returns an empty dict if no scopes are active.
        """
        if self.scope_stack:
            return self.scope_stack[-1][2]
        return {}

    def get_all_local_names(self) -> Set[str]:
        """
        Get all locally defined names across all active scopes.
        
        Useful for debugging and for passes that need to know all local names.
        """
        all_locals = set()
        for locals_set, _, _ in self.scope_stack:
            all_locals.update(locals_set)
        return all_locals
