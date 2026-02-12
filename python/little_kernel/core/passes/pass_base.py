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
from typing import Any, List
from copy import copy
from .utils.preserve_attributes import copy_node_with_attrs


class CompleteASTMutator(ast.NodeTransformer):
    """
    A complete AST mutator that processes EVERY standard Python AST node type
    and all their attributes. Recursively creates modified copies of nodes while
    preserving structure. Override visit_<NodeType> to implement custom mutations.
    """

    def _copy_node(self, node: ast.AST) -> ast.AST:
        """Create a shallow copy of the node to avoid modifying the original"""
        if not isinstance(node, ast.AST):
            return node
        node_copy = copy(node)
        node_copy.__class__ = node.__class__
        node_copy.__dict__.update(node.__dict__)
        ast.copy_location(node_copy, node)
        # Ensure special attributes are preserved (in case __dict__.update missed any)
        return copy_node_with_attrs(node, node_copy)

    def _process_list(self, items: List[Any]) -> List[Any]:
        """Process a list of nodes/values (recursively visit nodes)"""
        ret = []
        tmp = [self.visit(item) if isinstance(item, ast.AST) else item for item in items]
        # remove None
        for item in tmp:
            if isinstance(item, (list, tuple)):
                ret.extend(item)
            elif item is not None:
                ret.append(item)
        return ret

    def visit(self, node):
        if node is None:
            return None
        # Preserve IR nodes (EnumDef, SpecialStructDef) - they should pass through unchanged
        # unless explicitly handled by subclasses
        from .utils.ir_nodes import is_ir_node
        if is_ir_node(node):
            # IR nodes should be preserved through passes unless explicitly handled
            return node
        else:
            return super().visit(node)

    # ------------------------------
    # Module and top-level nodes
    # ------------------------------
    def visit_Module(self, node: ast.Module) -> ast.AST:
        node_copy = self._copy_node(node)
        # Process body, but preserve IR nodes
        from .utils.ir_nodes import is_ir_node
        new_body = []
        for stmt in node.body:
            if is_ir_node(stmt):
                # Preserve IR nodes as-is (they will be handled in codegen)
                new_body.append(stmt)
            else:
                visited = self.visit(stmt)
                if visited is not None:
                    if isinstance(visited, list):
                        new_body.extend(visited)
                    else:
                        new_body.append(visited)
        node_copy.body = new_body
        node_copy.type_ignores = self._process_list(node.type_ignores)
        return node_copy

    def visit_Interactive(self, node: ast.Interactive) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.body = self._process_list(node.body)
        return node_copy

    def visit_Expression(self, node: ast.Expression) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.body = self.visit(node.body)
        return node_copy

    # ------------------------------
    # Statements
    # ------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.name = node.name  # Preserve unless mutated
        node_copy.args = self.visit(node.args)
        node_copy.body = self._process_list(node.body)
        node_copy.decorator_list = self._process_list(node.decorator_list)
        node_copy.returns = self.visit(node.returns)
        node_copy.type_comment = node.type_comment  # Primitive, no mutation
        return node_copy

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.name = node.name
        node_copy.args = self.visit(node.args)
        node_copy.body = self._process_list(node.body)
        node_copy.decorator_list = self._process_list(node.decorator_list)
        node_copy.returns = self.visit(node.returns)
        node_copy.type_comment = node.type_comment
        return node_copy

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.name = node.name
        node_copy.bases = self._process_list(node.bases)
        node_copy.keywords = self._process_list(node.keywords)
        node_copy.body = self._process_list(node.body)
        node_copy.decorator_list = self._process_list(node.decorator_list)
        return node_copy

    def visit_Return(self, node: ast.Return) -> ast.AST:
        node_copy = self._copy_node(node)
        if node.value is not None:
            node_copy.value = self.visit(node.value)
        return node_copy

    def visit_Delete(self, node: ast.Delete) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.targets = self._process_list(node.targets)
        return node_copy

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.targets = self._process_list(node.targets)
        node_copy.value = self.visit(node.value)
        return node_copy

    def visit_AugAssign(self, node: ast.AugAssign) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.target = self.visit(node.target)
        node_copy.op = self.visit(node.op)  # Operator (e.g., Add)
        node_copy.value = self.visit(node.value)
        return node_copy

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.target = self.visit(node.target)
        node_copy.annotation = self.visit(node.annotation)
        node_copy.value = self.visit(node.value) if node.value else None
        node_copy.simple = node.simple  # Boolean flag
        return node_copy

    def visit_For(self, node: ast.For) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.target = self.visit(node.target)
        node_copy.iter = self.visit(node.iter)
        node_copy.body = self._process_list(node.body)
        node_copy.orelse = self._process_list(node.orelse)
        return node_copy

    def visit_AsyncFor(self, node: ast.AsyncFor) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.target = self.visit(node.target)
        node_copy.iter = self.visit(node.iter)
        node_copy.body = self._process_list(node.body)
        node_copy.orelse = self._process_list(node.orelse)
        return node_copy

    def visit_While(self, node: ast.While) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.test = self.visit(node.test)
        node_copy.body = self._process_list(node.body)
        node_copy.orelse = self._process_list(node.orelse)
        return node_copy

    def visit_If(self, node: ast.If) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.test = self.visit(node.test)
        node_copy.body = self._process_list(node.body)
        node_copy.orelse = self._process_list(node.orelse)
        return node_copy

    def visit_With(self, node: ast.With) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.items = self._process_list(node.items)  # List of WithItem
        node_copy.body = self._process_list(node.body)
        return node_copy

    def visit_AsyncWith(self, node: ast.AsyncWith) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.items = self._process_list(node.items)
        node_copy.body = self._process_list(node.body)
        return node_copy

    # Match statement is only available in Python 3.10+
    # def visit_Match(self, node) -> ast.AST:
    #     node_copy = self._copy_node(node)
    #     node_copy.subject = self.visit(node.subject)
    #     node_copy.cases = self._process_list(node.cases)  # List of MatchCase
    #     return node_copy

    def visit_Raise(self, node: ast.Raise) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.exc = self.visit(node.exc) if node.exc else None
        node_copy.cause = self.visit(node.cause) if node.cause else None
        return node_copy

    def visit_Try(self, node: ast.Try) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.body = self._process_list(node.body)
        node_copy.handlers = self._process_list(node.handlers)  # List of ExceptHandler
        node_copy.orelse = self._process_list(node.orelse)
        node_copy.finalbody = self._process_list(node.finalbody)
        return node_copy

    def visit_Assert(self, node: ast.Assert) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.test = self.visit(node.test)
        node_copy.msg = self.visit(node.msg) if node.msg else None
        return node_copy

    def visit_Import(self, node: ast.Import) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.names = self._process_list(node.names)  # List of alias objects
        return node_copy

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.module = node.module  # String
        node_copy.names = self._process_list(node.names)  # List of alias
        node_copy.level = node.level  # Integer (relative import level)
        return node_copy

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.names = node.names  # List of strings (no nodes to process)
        return node_copy

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.names = node.names  # List of strings
        return node_copy

    def visit_Expr(self, node: ast.Expr) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        return node_copy

    def visit_Pass(self, node: ast.Pass) -> ast.AST:
        return self._copy_node(node)  # No attributes to process

    def visit_Break(self, node: ast.Break) -> ast.AST:
        return self._copy_node(node)

    def visit_Continue(self, node: ast.Continue) -> ast.AST:
        return self._copy_node(node)

    # ------------------------------
    # Expressions
    # ------------------------------
    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.op = self.visit(node.op)  # And/Or
        node_copy.values = self._process_list(node.values)
        return node_copy

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.left = self.visit(node.left)
        node_copy.op = self.visit(node.op)  # Add/Sub/etc.
        node_copy.right = self.visit(node.right)
        return node_copy

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.op = self.visit(node.op)  # Not/Invert/etc.
        node_copy.operand = self.visit(node.operand)
        return node_copy

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.args = self.visit(node.args)
        node_copy.body = self.visit(node.body)
        return node_copy

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.test = self.visit(node.test)
        node_copy.body = self.visit(node.body)
        node_copy.orelse = self.visit(node.orelse)
        return node_copy

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.keys = self._process_list(node.keys)
        node_copy.values = self._process_list(node.values)
        return node_copy

    def visit_Set(self, node: ast.Set) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.elts = self._process_list(node.elts)
        return node_copy

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.elt = self.visit(node.elt)
        node_copy.generators = self._process_list(node.generators)  # List of comprehension
        return node_copy

    def visit_SetComp(self, node: ast.SetComp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.elt = self.visit(node.elt)
        node_copy.generators = self._process_list(node.generators)
        return node_copy

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.key = self.visit(node.key)
        node_copy.value = self.visit(node.value)
        node_copy.generators = self._process_list(node.generators)
        return node_copy

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.elt = self.visit(node.elt)
        node_copy.generators = self._process_list(node.generators)
        return node_copy

    def visit_Await(self, node: ast.Await) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        return node_copy

    def visit_Yield(self, node: ast.Yield) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value) if node.value else None
        return node_copy

    def visit_YieldFrom(self, node: ast.YieldFrom) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        return node_copy

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.left = self.visit(node.left)
        node_copy.ops = self._process_list(node.ops)  # List of comparators (Eq/Lt/etc.)
        node_copy.comparators = self._process_list(node.comparators)
        return node_copy

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.func = self.visit(node.func)
        node_copy.args = self._process_list(node.args)
        node_copy.keywords = self._process_list(node.keywords)  # List of keyword
        return node_copy

    def visit_FormattedValue(self, node: ast.FormattedValue) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        node_copy.conversion = node.conversion  # Integer
        node_copy.format_spec = self.visit(node.format_spec) if node.format_spec else None
        return node_copy

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.values = self._process_list(node.values)  # List of FormattedValue/Constant
        return node_copy

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        node_copy = self._copy_node(node)
        # Constant values (int/str/float/etc.) are primitives - no mutation unless overridden
        node_copy.value = node.value
        node_copy.kind = node.kind
        return node_copy

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        node_copy.attr = node.attr  # String (attribute name)
        node_copy.ctx = self.visit(node.ctx)  # Load/Store/Del context
        return node_copy

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.id = node.id  # String (variable name)
        node_copy.ctx = self.visit(node.ctx)  # Context
        return node_copy

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        node_copy.slice = self.visit(node.slice)
        node_copy.ctx = self.visit(node.ctx)
        return node_copy

    def visit_Starred(self, node: ast.Starred) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.value = self.visit(node.value)
        node_copy.ctx = self.visit(node.ctx)
        return node_copy

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.elts = self._process_list(node.elts)
        node_copy.ctx = self.visit(node.ctx)
        return node_copy

    def visit_List(self, node: ast.List) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.elts = self._process_list(node.elts)
        node_copy.ctx = self.visit(node.ctx)
        return node_copy

    def visit_Slice(self, node: ast.Slice) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.lower = self.visit(node.lower) if node.lower else None
        node_copy.upper = self.visit(node.upper) if node.upper else None
        node_copy.step = self.visit(node.step) if node.step else None
        return node_copy

        # ------------------------------
        # Pattern matching (Python 3.10+)
        # ------------------------------
        # def visit_Match(self, node) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.subject = self.visit(node.subject)
        #        node_copy.cases = self._process_list(node.cases)
        #        return node_copy

        #    def visit_match_case(self, node) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.pattern = self.visit(node.pattern)
        #        node_copy.guard = self.visit(node.guard) if node.guard else None
        #        node_copy.body = self._process_list(node.body)
        #        return node_copy

        #    def visit_MatchValue(self, node: ast.MatchValue) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.value = self.visit(node.value)
        #        return node_copy

        #    def visit_MatchSingleton(self, node: ast.MatchSingleton) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.value = self.visit(node.value)
        #        return node_copy

        #    def visit_MatchSequence(self, node: ast.MatchSequence) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.patterns = self._process_list(node.patterns)
        #        return node_copy

        #    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.keys = self._process_list(node.keys)
        #        node_copy.patterns = self._process_list(node.patterns)
        #        node_copy.rest = self.visit(node.rest) if node.rest else None
        #        return node_copy

        #    def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.name = node.name  # String or None
        #        return node_copy

        #    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        #        node_copy = self._copy_node(node)
        #        node_copy.pattern = self.visit(node.pattern) if node.pattern else None
        #        node_copy.name = node.name  # String or None
        #        node_copy.guard = self.visit(node.guard) if node.guard else None
        #        return node_copy

        #    def visit_MatchOr(self, node: ast.MatchOr) -> ast.AST:
        #        node_copy = self._copy_node(node)
        node_copy.patterns = self._process_list(node.patterns)
        return node_copy

    # ------------------------------
    # Try/except related
    # ------------------------------
    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.type = self.visit(node.type) if node.type else None
        node_copy.name = node.name  # String or None
        node_copy.body = self._process_list(node.body)
        return node_copy

    # ------------------------------
    # With statement related (visit_With is already defined above)
    # ------------------------------
    def visit_withitem(self, node: ast.withitem) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.context_expr = self.visit(node.context_expr)
        node_copy.optional_vars = self.visit(node.optional_vars) if node.optional_vars else None
        return node_copy

    # ------------------------------
    # Function/class definition related
    # ------------------------------
    def visit_arguments(self, node: ast.arguments) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.args = self._process_list(node.args)  # List of arg
        node_copy.posonlyargs = self._process_list(node.posonlyargs)  # List of arg
        node_copy.vararg = self.visit(node.vararg) if node.vararg else None  # vararg object
        node_copy.kwonlyargs = self._process_list(node.kwonlyargs)  # List of arg
        node_copy.kw_defaults = self._process_list(node.kw_defaults)  # List of expr or None
        node_copy.kwarg = self.visit(node.kwarg) if node.kwarg else None  # kwarg object
        node_copy.defaults = self._process_list(node.defaults)  # List of expr
        return node_copy

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.arg = node.arg  # String (parameter name)
        node_copy.annotation = self.visit(node.annotation) if node.annotation else None
        node_copy.type_comment = node.type_comment
        return node_copy

    # ------------------------------
    # Other utility nodes
    # ------------------------------
    def visit_Keyword(self, node: ast.keyword) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.arg = node.arg  # String (keyword name)
        node_copy.value = self.visit(node.value)
        return node_copy

    def visit_alias(self, node: ast.alias) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.name = node.name  # String (import name)
        node_copy.asname = node.asname  # String or None (alias)
        return node_copy

    def visit_comprehension(self, node: ast.comprehension) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.target = self.visit(node.target)
        node_copy.iter = self.visit(node.iter)
        node_copy.ifs = self._process_list(node.ifs)
        node_copy.is_async = node.is_async  # Boolean
        return node_copy
