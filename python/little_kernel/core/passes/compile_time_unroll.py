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
"""Compile-time expansion of ``for x in ll.unroll(range(...))`` loops.

Each iteration becomes a copy of the body with ``x`` replaced by its value. Within a copy, locals
that are assigned once in the whole function and evaluate to a compile-time integer are propagated
into later statements, and ``if`` statements whose condition is then a constant keep only the taken
branch. A slot schedule written as one loop over slots therefore lowers to straight-line code that
contains, in every slot, only the operations of that slot (and every register-array index is a
literal). Locals declared in the body are renamed per copy.
"""

import ast
import copy
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

_SAFE_BUILTINS = {"min": min, "max": max, "abs": abs}


def _nested_scope_referenced(func: ast.FunctionDef) -> Set[str]:
    """Names loaded inside a nested FunctionDef / ClassDef / Lambda of ``func``.

    ``_Substitute`` only rewrites the statements of the block it walks; nested scopes are returned
    verbatim, so a constant propagated into ``consts`` is never substituted there. Erasing the
    declaration of such a name would leave a dangling reference, so these names are ineligible
    for fold-and-erase even when single-assigned."""
    names: Set[str] = set()

    def visit_scope(node: ast.AST, nested: bool) -> None:
        for child in ast.iter_child_nodes(node):
            is_scope = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda))
            if isinstance(child, ast.Name) and nested and isinstance(child.ctx, ast.Load):
                names.add(child.id)
            visit_scope(child, nested or is_scope)

    visit_scope(func, False)
    return names


def _copy_ast(node: Any) -> Any:
    """Fresh nodes for the syntax fields. Other attributes (locations, values attached by earlier
    passes) are shared, `.parent` links are dropped, and the Load / Store / Del singletons are kept:
    other passes may have left `.parent` on them, pointing into unrelated trees."""
    if isinstance(node, (ast.Load, ast.Store, ast.Del)):
        return node
    if isinstance(node, ast.AST):
        new = node.__class__()
        for key, value in node.__dict__.items():
            if key != "parent":
                setattr(new, key, _copy_ast(value) if key in node._fields else value)
        return new
    if isinstance(node, list):
        return [_copy_ast(item) for item in node]
    return node


def _is_unroll_range(node: ast.For) -> Optional[ast.Call]:
    """The range(...) call of ``for x in ll.unroll(range(...))``, else None."""
    call = node.iter
    if not (isinstance(call, ast.Call) and len(call.args) == 1):
        return None
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else None)
    if name != "unroll":
        return None
    inner = call.args[0]
    if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "range":
        return inner
    return None


def _assigned_names(target: ast.AST) -> List[str]:
    """Variables a store target writes: x, (x, y), and the root of x[i] / x.a (not names in the index)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [name for elt in target.elts for name in _assigned_names(elt)]
    if isinstance(target, ast.Starred):
        return _assigned_names(target.value)
    if isinstance(target, (ast.Subscript, ast.Attribute)):
        return _assigned_names(target.value)
    return []


def _assignment_counts(func: ast.FunctionDef) -> Counter:
    counts: Counter = Counter()
    for node in ast.walk(func):
        targets: List[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.For)):
            targets = [node.target]
        for target in targets:
            for name in _assigned_names(target):
                counts[name] += 1
    for arg in func.args.args:
        counts[arg.arg] += 1
    return counts


def _declared_names(stmts: List[ast.stmt]) -> Set[str]:
    names = set()
    for stmt in stmts:
        for node in ast.walk(stmt):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
    return names


class _Rename(ast.NodeTransformer):

    def __init__(self, mapping: Dict[str, str]):
        self.mapping = mapping

    def visit_Name(self, node: ast.Name):
        if node.id in self.mapping:
            return ast.copy_location(ast.Name(id=self.mapping[node.id], ctx=node.ctx), node)
        return node


class _Substitute(ast.NodeTransformer):
    """Replace loads of compile-time constants by literals."""

    def __init__(self, consts: Dict[str, Any]):
        self.consts = consts

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id in self.consts:
            return ast.copy_location(ast.Constant(self.consts[node.id]), node)
        return node


class CompileTimeUnroller:

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx
        self.local_names: Set[str] = set()
        self.single_assigned: Set[str] = set()
        self.nested_referenced: Set[str] = set()
        self._copy_id = 0

    # ------------------------------------------------------------------ evaluation
    def _eval(self, expr: ast.expr, consts: Dict[str, Any]) -> Tuple[bool, Any]:
        env: Dict[str, Any] = {}
        for node in ast.walk(expr):
            if isinstance(node, ast.Call):
                if not (isinstance(node.func, ast.Name) and node.func.id in _SAFE_BUILTINS):
                    return False, None
            elif isinstance(node, (ast.Attribute, ast.Subscript, ast.Lambda, ast.NamedExpr)):
                return False, None
            elif isinstance(node, ast.Name):
                if node.id in consts:
                    env[node.id] = consts[node.id]
                elif node.id in _SAFE_BUILTINS:
                    continue
                elif node.id not in self.local_names and isinstance(self.ctx.get(node.id), (bool, int)):
                    env[node.id] = self.ctx[node.id]
                else:
                    return False, None
        try:
            value = eval(compile(ast.Expression(body=_copy_ast(expr)), "<compile-time-unroll>", "eval"),
                         {"__builtins__": _SAFE_BUILTINS}, env)
        except Exception:
            return False, None
        if isinstance(value, (bool, int)):
            return True, value
        return False, None

    # ------------------------------------------------------------------ statements
    def run(self, func: ast.FunctionDef) -> ast.FunctionDef:
        counts = _assignment_counts(func)
        self.local_names = set(counts)
        self.single_assigned = {name for name, n in counts.items() if n == 1}
        self.nested_referenced = _nested_scope_referenced(func)
        func.body = self._block(func.body, {}) or [ast.Pass()]
        return func

    def _block(self, stmts: List[ast.stmt], consts: Dict[str, Any]) -> List[ast.stmt]:
        out: List[ast.stmt] = []
        for stmt in stmts:
            out.extend(self._stmt(stmt, consts))
        return out

    def _stmt(self, stmt: ast.stmt, consts: Dict[str, Any]) -> List[ast.stmt]:
        if isinstance(stmt, ast.For):
            rng = _is_unroll_range(stmt)
            if rng is not None and isinstance(stmt.target, ast.Name) and not stmt.orelse:
                bounds = [self._eval(arg, consts) for arg in rng.args]
                if all(ok for ok, _ in bounds):
                    return self._expand(stmt, range(*[v for _, v in bounds]), consts)
            stmt = copy.copy(stmt)
            stmt.iter = _Substitute(consts).visit(stmt.iter)
            stmt.body = self._block(stmt.body, dict(consts)) or [ast.Pass()]
            stmt.orelse = self._block(stmt.orelse, dict(consts))
            return [stmt]

        if isinstance(stmt, ast.If):
            test = _Substitute(consts).visit(stmt.test)
            ok, value = self._eval(test, consts)
            if ok:
                return self._block(stmt.body if value else stmt.orelse, consts)
            stmt = copy.copy(stmt)
            stmt.test = test
            stmt.body = self._block(stmt.body, dict(consts)) or [ast.Pass()]
            stmt.orelse = self._block(stmt.orelse, dict(consts))
            return [stmt]

        if isinstance(stmt, ast.While):
            stmt = copy.copy(stmt)
            stmt.test = _Substitute(consts).visit(stmt.test)
            stmt.body = self._block(stmt.body, dict(consts)) or [ast.Pass()]
            stmt.orelse = self._block(stmt.orelse, dict(consts))
            return [stmt]

        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            value = _Substitute(consts).visit(stmt.value)
            # Erase a single-assigned declaration whose value folds to an int/bool: every Load of
            # the name is then substituted, so no runtime slot is needed. A name still referenced
            # inside a nested scope (FunctionDef / ClassDef / Lambda) is NOT substituted there, so
            # its declaration must be kept or the reference dangles.
            if stmt.target.id in self.single_assigned and stmt.target.id not in self.nested_referenced:
                ok, const = self._eval(value, consts)
                if ok:
                    consts[stmt.target.id] = const
                    return []
            stmt = copy.copy(stmt)
            stmt.value = value
            return [stmt]

        if isinstance(stmt, (ast.FunctionDef, ast.ClassDef)):
            return [stmt]

        return [_Substitute(consts).visit(stmt)]

    def _expand(self, loop: ast.For, values, consts: Dict[str, Any]) -> List[ast.stmt]:
        var = loop.target.id
        declared = _declared_names(loop.body)
        out: List[ast.stmt] = []
        for value in values:
            self._copy_id += 1
            mapping = {name: f"{name}__u{self._copy_id}" for name in declared}
            body = [_Rename(mapping).visit(_copy_ast(s)) for s in loop.body]
            # the renamed locals are unique to this copy: single assignment is decided within it
            counts = _assignment_counts(
                ast.FunctionDef(name="_", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                                                             defaults=[]), body=body, decorator_list=[]))
            self.local_names |= set(mapping.values())
            self.single_assigned |= {name for name in mapping.values() if counts[name] == 1}
            # A renamed local still referenced inside a nested scope of the renamed body cannot be
            # erased: the nested scope is not substituted, so the reference would dangle.
            self.nested_referenced |= _nested_scope_referenced(
                ast.FunctionDef(name="_", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                                                             defaults=[]), body=body, decorator_list=[]))
            inner = dict(consts)
            inner[var] = value
            out.extend(self._block(body, inner))
        return out


def compile_time_unroll(tree: ast.AST, ctx: Dict[str, Any], *args, **kwargs) -> ast.AST:
    """Expand ``ll.unroll(range(const...))`` loops at compile time (see module docstring)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            CompileTimeUnroller(ctx).run(node)
    ast.fix_missing_locations(tree)
    return tree
