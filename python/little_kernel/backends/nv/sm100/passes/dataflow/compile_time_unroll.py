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
"""Expand compile-time loops that index static alias containers."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


def _call_name(node: ast.AST | None) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else None


def _clone(node):
    if isinstance(node, ast.AST):
        copied = type(node)(**{field: _clone(getattr(node, field)) for field in node._fields})
        return ast.copy_location(copied, node)
    if isinstance(node, list):
        return [_clone(item) for item in node]
    return node


def _eval_const(expr: ast.expr, env: dict[str, Any]) -> Any:
    return eval(compile(ast.Expression(_clone(expr)), "<compile-time-unroll>", "eval"), env)


def _range_values(iter_expr: ast.expr, env: dict[str, Any]) -> list[int] | None:
    call = iter_expr
    explicit_unroll = isinstance(call, ast.Call) and _call_name(call) == "unroll"
    if explicit_unroll:
        if len(call.args) != 1:
            raise ValueError("ll.unroll expects exactly one range(...) argument")
        call = call.args[0]
    if not isinstance(call, ast.Call) or _call_name(call) != "range":
        if explicit_unroll:
            raise ValueError("ll.unroll expects a range(...) argument")
        return None
    try:
        args = [_eval_const(arg, env) for arg in call.args]
    except (NameError, TypeError, ValueError) as exc:
        if explicit_unroll:
            raise ValueError("ll.unroll range bounds must be compile-time constants: "
                             f"{ast.unparse(call)}") from exc
        # An ordinary runtime range is not eligible for compile-time expansion.
        return None
    return list(range(*args))


def _target_names(target: ast.expr) -> list[str] | None:
    if not isinstance(target, (ast.Tuple, ast.List)):
        return None
    out: list[str] = []
    for item in target.elts:
        if not isinstance(item, ast.Name):
            return None
        out.append(item.id)
    return out


def _value_names(value: ast.expr) -> list[str] | None:
    if not isinstance(value, (ast.Tuple, ast.List)):
        return None
    out: list[str] = []
    for item in value.elts:
        if not isinstance(item, ast.Name):
            return None
        out.append(item.id)
    return out


def _slice_expr(node: ast.Subscript) -> ast.expr:
    return node.slice


def _const_int(expr: ast.expr, env: dict[str, Any]) -> int | None:
    try:
        value = _eval_const(expr, env)
    except Exception:
        return None
    return int(value) if isinstance(value, int) else None


@dataclass
class StaticAliases:
    by_container: dict[tuple[str, int], str]

    def containers(self) -> set[str]:
        return {name for name, _idx in self.by_container}

    def resolve(self, name: str, index: int) -> str | None:
        return self.by_container.get((name, index))


class StaticAliasCollector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.aliases: dict[tuple[str, int], str] = {}

    def visit_Assign(self, node: ast.Assign) -> None:
        if len(node.targets) != 1:
            return
        target = node.targets[0]
        if isinstance(node.value, ast.Name):
            names = _target_names(target)
            if names is not None:
                for idx, name in enumerate(names):
                    self.aliases[(node.value.id, idx)] = name
                return
        if isinstance(target, ast.Name):
            names = _value_names(node.value)
            if names is not None:
                for idx, name in enumerate(names):
                    self.aliases[(target.id, idx)] = name
                return
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name) and node.value is not None:
            names = _value_names(node.value)
            if names is not None:
                for idx, name in enumerate(names):
                    self.aliases[(node.target.id, idx)] = name
                return
        self.generic_visit(node)


class LoopVarSubstituter(ast.NodeTransformer):

    def __init__(self, var: str, value: int) -> None:
        self.var = var
        self.value = value

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load) and node.id == self.var:
            return ast.copy_location(ast.Constant(self.value), node)
        return node


class AnnAssignNameCollector(ast.NodeVisitor):

    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self.names.add(node.target.id)
        self.generic_visit(node)


class LocalNameRenamer(ast.NodeTransformer):

    def __init__(self, names: set[str], suffix: str) -> None:
        self.mapping = {name: f"{name}_{suffix}" for name in names}

    def visit_Name(self, node: ast.Name):
        renamed = self.mapping.get(node.id)
        if renamed is not None:
            return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)
        return node


class AliasIndexResolver(ast.NodeTransformer):

    def __init__(self, aliases: StaticAliases, env: dict[str, Any]) -> None:
        self.aliases = aliases
        self.env = env

    def visit_Subscript(self, node: ast.Subscript):
        node = self.generic_visit(node)
        if not isinstance(node.value, ast.Name):
            return node
        index = _const_int(_slice_expr(node), self.env)
        if index is None:
            return node
        resolved = self.aliases.resolve(node.value.id, index)
        if resolved is None:
            return node
        return ast.copy_location(ast.Name(id=resolved, ctx=node.ctx), node)


class AliasIndexUseFinder(ast.NodeVisitor):

    def __init__(self, aliases: StaticAliases, loop_var: str) -> None:
        self.aliases = aliases
        self.loop_var = loop_var
        self.found = False

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (isinstance(node.value, ast.Name) and node.value.id in self.aliases.containers() and any(
                isinstance(item, ast.Name) and item.id == self.loop_var for item in ast.walk(_slice_expr(node)))):
            self.found = True
            return
        self.generic_visit(node)


class CompileTimeUnroll(ast.NodeTransformer):

    def __init__(self, aliases: StaticAliases, env: dict[str, Any]) -> None:
        self.aliases = aliases
        self.env = env

    def visit_For(self, node: ast.For):
        node.iter = self.visit(node.iter)
        if not isinstance(node.target, ast.Name):
            return self.generic_visit(node)

        values = _range_values(node.iter, self.env)
        if values is None:
            return self.generic_visit(node)

        finder = AliasIndexUseFinder(self.aliases, node.target.id)
        for stmt in node.body:
            finder.visit(stmt)
        if not finder.found:
            return self.generic_visit(node)

        out: list[ast.stmt] = []
        for value in values:
            substituter = LoopVarSubstituter(node.target.id, value)
            locals = AnnAssignNameCollector()
            for stmt in node.body:
                locals.visit(stmt)
            renamer = LocalNameRenamer(locals.names, str(value))
            for stmt in node.body:
                expanded = renamer.visit(_clone(stmt))
                expanded = substituter.visit(expanded)
                expanded = self.visit(expanded)
                if isinstance(expanded, list):
                    out.extend(expanded)
                else:
                    out.append(expanded)
        return out


def _aliases_for(tree: ast.AST) -> StaticAliases:
    collector = StaticAliasCollector()
    collector.visit(tree)
    return StaticAliases(collector.aliases)


def compile_time_unroll_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    env = ctx if ctx is not None else {}
    aliases = _aliases_for(tree)
    CompileTimeUnroll(aliases, env).visit(tree)
    AliasIndexResolver(aliases, env).visit(tree)
    ast.fix_missing_locations(tree)
    return tree


__all__ = ["compile_time_unroll_pass"]
