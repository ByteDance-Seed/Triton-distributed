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
"""Coalesce adjacent equivalent role warp guards after resource lowering."""
from __future__ import annotations
import ast


def _expr_key(expr: ast.expr) -> str:
    return ast.dump(expr, include_attributes=False)


def _guarded_body(stmt: ast.stmt) -> tuple[ast.expr, list[ast.stmt], ast.expr | None] | None:
    if not isinstance(stmt, ast.If) or stmt.orelse:
        return None
    if (len(stmt.body) == 1 and isinstance(stmt.body[0], ast.If) and not stmt.body[0].orelse):
        return stmt.body[0].test, stmt.body[0].body, stmt.test
    return stmt.test, stmt.body, None


def _wrap_guarded(body: list[ast.stmt], guard: ast.expr | None) -> list[ast.stmt]:
    if guard is None:
        return body
    return [ast.If(test=guard, body=body, orelse=[])]


def _is_warp_guard(expr: ast.expr) -> bool:
    return any(isinstance(node, ast.Name) and node.id == "__lk_warp_idx" for node in ast.walk(expr))


def _merge_repeated_guards(body: list[ast.stmt]) -> list[ast.stmt]:
    out: list[ast.stmt] = []
    idx = 0
    while idx < len(body):
        first = _guarded_body(body[idx])
        if first is None or not _is_warp_guard(first[0]):
            out.append(body[idx])
            idx += 1
            continue

        guard_test, guard_body, inner_guard = first
        key = _expr_key(guard_test)
        merged = _wrap_guarded(guard_body, inner_guard)
        j = idx + 1
        while j < len(body):
            nxt = _guarded_body(body[j])
            if nxt is None or _expr_key(nxt[0]) != key:
                break
            merged.extend(_wrap_guarded(nxt[1], nxt[2]))
            j += 1

        if j == idx + 1:
            out.append(body[idx])
        else:
            out.append(ast.copy_location(ast.If(test=guard_test, body=merged, orelse=[]), body[idx]))
        idx = j
    return out


class RepeatedGuardMerger(ast.NodeTransformer):

    def _visit_block(self, body: list[ast.stmt]) -> list[ast.stmt]:
        visited: list[ast.stmt] = []
        for stmt in body:
            item = self.visit(stmt)
            if item is None:
                continue
            visited.extend(item if isinstance(item, list) else [item])
        return _merge_repeated_guards(visited)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node.body = self._visit_block(node.body)
        return node

    def visit_If(self, node: ast.If):
        node.test = self.visit(node.test)
        node.body = self._visit_block(node.body)
        node.orelse = self._visit_block(node.orelse)
        return node

    def visit_While(self, node: ast.While):
        node.test = self.visit(node.test)
        node.body = self._visit_block(node.body)
        node.orelse = self._visit_block(node.orelse)
        return node

    def visit_For(self, node: ast.For):
        node.target = self.visit(node.target)
        node.iter = self.visit(node.iter)
        node.body = self._visit_block(node.body)
        node.orelse = self._visit_block(node.orelse)
        return node


def merge_repeated_guards_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    del ctx
    tree = RepeatedGuardMerger().visit(tree)
    ast.fix_missing_locations(tree)
    return tree
