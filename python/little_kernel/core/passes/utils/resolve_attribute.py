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


def recursive_resolve_attribute(node, ctx, scope_manager=None):
    if isinstance(node, ast.Attribute):
        val = recursive_resolve_attribute(node.value, ctx, scope_manager)
        if isinstance(val, ast.AST):
            return node
        # otherwise, try to get the attribute
        try:
            attr = getattr(val, node.attr)
            return attr
        except AttributeError:
            # If attribute doesn't exist, return the AST node as-is
            return node
    if isinstance(node, ast.Name):
        # Check if the name is defined in local scope first
        if scope_manager is not None and scope_manager.is_local(node.id):
            return node

        # First try to resolve from ctx (which may include closure variables)
        if isinstance(ctx, dict) and node.id in ctx:
            return ctx[node.id]
        # If ctx is not a dict, try to get the attribute
        if not isinstance(ctx, dict):
            if hasattr(ctx, node.id):
                return getattr(ctx, node.id)
    if isinstance(node, ast.Subscript):
        new_value = recursive_resolve_attribute(node.value, ctx, scope_manager)
        new_slice = recursive_resolve_attribute(node.slice, ctx, scope_manager)
        if isinstance(new_value, ast.Constant):
            new_value = new_value.value
        if isinstance(new_slice, ast.Constant):
            new_slice = new_slice.value
        if not isinstance(new_value, ast.AST) and not isinstance(new_slice, ast.AST):
            return new_value[new_slice]
        else:
            return node
    return node
