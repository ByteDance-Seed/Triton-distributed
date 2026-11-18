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
from typing import Optional, Tuple, Type

# CPython reuses one Load/Store/Del instance for every parsed AST. Attaching
# `.parent` to them poisons all later ast.parse() trees in the same process
# (e.g. CuTeDSL deepcopy during cute.compile after an LK kernel build).
_CTX_NODE_TYPES: Tuple[Type[ast.AST], ...] = (ast.Load, ast.Store, ast.Del)


def _reset_ctx_node_parents() -> None:
    """Clear `.parent` accidentally left on the global ctx singletons."""
    snippets = (
        ("x", lambda tree: tree.body[0].value.ctx),
        ("y = 1", lambda tree: tree.body[0].targets[0].ctx),
        ("del z", lambda tree: tree.body[0].targets[0].ctx),
    )
    for source, get_ctx in snippets:
        ctx = get_ctx(ast.parse(source))
        if hasattr(ctx, "parent"):
            del ctx.parent


def add_parent_references(node: ast.AST, parent: Optional[ast.AST] = None) -> None:
    """
    Recursively traverse the AST and add a `parent` attribute to each node,
    pointing to its parent node in the AST hierarchy.
    """
    if parent is None:
        _reset_ctx_node_parents()

    if parent is not None and not isinstance(node, _CTX_NODE_TYPES):
        node.parent = parent

    for child in ast.iter_child_nodes(node):
        add_parent_references(child, node)
