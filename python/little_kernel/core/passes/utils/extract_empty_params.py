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
from typing import Optional, Tuple


def extract_empty_params(empty_call: ast.Call) -> Tuple[ast.AST, ast.AST, ast.AST]:
    """
    Extract shape, dtype, scope from ll.empty call (handles both positional and keyword args).
    Assumes ll.empty signature: empty(shape, dtype, scope)
    """
    # Default parameter positions (shape=0, dtype=1, scope=2)
    shape_expr: Optional[ast.AST] = None
    dtype_expr: Optional[ast.AST] = None
    scope_expr: Optional[ast.AST] = None

    # Process positional arguments
    if len(empty_call.args) >= 1:
        shape_expr = empty_call.args[0]
    if len(empty_call.args) >= 2:
        dtype_expr = empty_call.args[1]
    if len(empty_call.args) >= 3:
        scope_expr = empty_call.args[2]

    # Process keyword arguments (override positional if present)
    for kw in empty_call.keywords:
        if kw.arg == "shape":
            shape_expr = kw.value
        elif kw.arg == "dtype":
            dtype_expr = kw.value
        elif kw.arg == "scope":
            scope_expr = kw.value

    # Validate required parameters
    if not shape_expr:
        raise ValueError("ll.empty call missing 'shape' argument (positional or keyword)")
    if not dtype_expr:
        raise ValueError("ll.empty call missing 'dtype' argument (positional or keyword)")
    if not scope_expr:
        raise ValueError("ll.empty call missing 'scope' argument (positional or keyword)")

    return shape_expr, dtype_expr, scope_expr


def extract_empty_align(empty_call: ast.Call) -> Optional[int]:
    """
    Extract the optional per-buffer ``align`` keyword of an ll.empty /
    ll.zeros / ll.full call, or None when absent.  The value must be a
    positive power-of-two integer literal (byte alignment of this
    buffer's own start offset inside its scope's allocation order).
    """
    align_expr: Optional[ast.AST] = None
    for kw in empty_call.keywords:
        if kw.arg == "align":
            align_expr = kw.value
            break
    if align_expr is None:
        return None
    if not isinstance(align_expr, ast.Constant) or not isinstance(align_expr.value, int) or isinstance(
            align_expr.value, bool):
        raise ValueError("ll.empty align must be an integer constant, got "
                         f"{ast.dump(align_expr)}")
    value = align_expr.value
    if value <= 0 or (value & (value - 1)) != 0:
        raise ValueError(f"ll.empty align must be a positive power of two, got {value}")
    return value


def split_shape_dims(shape_expr: ast.AST) -> list:
    """
    Split an ll.empty shape list/tuple into its dimension values.

    Each dimension must be a constant: a nonnegative int (static extent) or a
    string in the launch shape-expression mini-language (dynamic extent,
    dynamic_shared scope only). Returns the list of dim values.
    """
    if not isinstance(shape_expr, (ast.List, ast.Tuple)):
        raise ValueError(f"ll.empty shape must be a list/tuple, got {type(shape_expr).__name__}")
    dims = []
    for s in shape_expr.elts:
        if not isinstance(s, ast.Constant):
            raise ValueError(f"ll.empty shape dimension must be a constant, got {type(s).__name__}")
        value = s.value
        if type(value) is int:
            if value < 0:
                raise ValueError("ll.empty shape dimension must be nonnegative")
            dims.append(value)
        elif type(value) is str:
            dims.append(value)
        else:
            raise ValueError(f"Shape dimension must be an integer or shape-expression string, "
                             f"got {type(value).__name__} (value: {value})")
    return dims


def extract_max_elems(empty_call: ast.Call, api_name: str):
    """Extract the optional max_elems constant from an ll.empty-like call."""
    max_elems_node = None
    for kw in empty_call.keywords:
        if kw.arg == "max_elems":
            max_elems_node = kw.value
            break
    if max_elems_node is None:
        return None
    if not (isinstance(max_elems_node, ast.Constant) and type(max_elems_node.value) is int
            and max_elems_node.value > 0):
        raise ValueError(f"{api_name} max_elems must be a positive integer constant, "
                         f"got {ast.dump(max_elems_node)}")
    return max_elems_node.value


def get_shape_size(shape_expr: ast.AST) -> int:
    """
    Calculate the total number of elements from the shape expression (product of all dimensions).
    e.g., shape [2, 4, 8] → 2*4*8 = 64 elements.
    """
    if not isinstance(shape_expr, (ast.List, ast.Tuple)):
        raise ValueError(f"ll.empty shape must be a list/tuple, got {type(shape_expr).__name__}")
    for s in shape_expr.elts:
        if not isinstance(s, ast.Constant):
            raise ValueError(f"ll.empty shape dimension must be a constant, got {type(s).__name__}")
    shape = [s.value for s in shape_expr.elts]
    if not isinstance(shape, (list, tuple)):
        raise ValueError(f"ll.empty shape must be a list/tuple, got {type(shape).__name__}")

    total_elements = 1
    for dim in shape:
        if not isinstance(dim, int):
            raise ValueError(f"Shape dimension must be an integer, got {type(dim).__name__} (value: {dim})")
        total_elements *= dim
    return total_elements
