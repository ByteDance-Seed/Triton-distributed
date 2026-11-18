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
"""Frontend normalization before dataflow graph analysis."""
from __future__ import annotations
import ast
from little_kernel.core.passes.inline import inline as inline_functions
from little_kernel.backends.nv.sm100.passes.dataflow.analysis import _const_value


def inline_functions_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    tree.body = inline_functions(tree, ctx or {}).body
    ast.fix_missing_locations(tree)
    return tree


class _CompileTimeBranchPruner(ast.NodeTransformer):

    def __init__(self, ctx: dict) -> None:
        self.ctx = ctx

    def visit_If(self, node: ast.If):
        try:
            value = _const_value(node.test, self.ctx)
        except (TypeError, ValueError):
            return self.generic_visit(node)
        selected = node.body if bool(value) else node.orelse
        result = []
        for stmt in selected:
            visited = self.visit(stmt)
            if visited is not None:
                result.extend(visited if isinstance(visited, list) else [visited])
        return result


def compile_time_branch_prune_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    tree = _CompileTimeBranchPruner(ctx or {}).visit(tree)
    ast.fix_missing_locations(tree)
    return tree
