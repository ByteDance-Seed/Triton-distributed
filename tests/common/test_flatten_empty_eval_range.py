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
"""CPU-only regression tests for FlattenEmpty's non-Constant range bounds.

``FlattenEmpty.visit_ListComp`` must evaluate a ``range()`` bound that is not
an ``ast.Constant`` (e.g. the ``BinOp`` ``N // 2``, or a plain module-level
name bound in ``ctx``) against the function's global context.  Previously
``_evaluate_expr`` compiled the expression as an ``ast.Module`` expression
statement in ``exec`` mode, which discards the value, then read a
``"__return__"`` key that is never inserted into the namespace - every
non-Constant range() bound therefore raised ``KeyError``, the enclosing
``try/except`` silently swallowed it, and the flatten optimization was skipped
for every evaluable non-Constant bound.  The fix evaluates the AST as an
``ast.Expression`` in ``eval`` mode so the value is returned directly.

These tests run the pass on a parsed module with ``unittest``-discoverable
cases; no backend or GPU is required.
"""

import ast
import textwrap
import unittest

import little_kernel.language as ll

from little_kernel.core.passes.flatten_empty import flatten_empty

_MODULE_TEMPLATE = """
def f() -> ll.void:
    xs = [ll.empty([8], ll.float32, "shared") for _ in range({bound})]
"""


def _flatten(bound: str, extra_ctx=None) -> ast.AST:
    source = textwrap.dedent(_MODULE_TEMPLATE.format(bound=bound))
    ctx = {"ll": ll, "N": 4}
    if extra_ctx:
        ctx.update(extra_ctx)
    return flatten_empty(ast.parse(source), ctx)


def _empty_array_binds(tree: ast.AST):
    """Collect (target_name, Assign.value) pairs whose value is the flattened
    ``flatten_empty_ll.empty([count], dtype=..., scope="local")`` call - i.e.
    a direct ``empty`` call taking a list shape and both keywords."""
    pairs = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == "empty" and len(call.args) == 1
                and isinstance(call.args[0], ast.List)):
            continue
        keywords = {kw.arg for kw in call.keywords}
        if keywords == {"dtype", "scope"}:
            pairs.append((node.targets[0].id, call))
    return pairs


def _elem_store_targets(tree: ast.AST):
    """Collect Subscript targets of assignments to ``name[i] = ll.empty(...)``."""
    indices = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript)
                and isinstance(node.targets[0].slice, ast.Constant)):
            continue
        indices.append(node.targets[0].slice.value)
    return indices


class TestFlattenEmptyNonConstantRangeBound(unittest.TestCase):

    def test_binop_range_bound_from_ctx_flattens(self):
        """`range(N // 2)` with N=4 in ctx flattens into an array plus stores."""
        tree = _flatten("N // 2")

        # No ListComp survives anywhere in the transformed module.
        self.assertEqual([n for n in ast.walk(tree) if isinstance(n, ast.ListComp)], [])

        # Exactly one tensor-array definition: xs = ll.empty([2], ...).
        arrays = _empty_array_binds(tree)
        self.assertEqual([name for name, _ in arrays], ["xs"])
        _, array_call = arrays[0]

        # Shape is the constant-count list [2].
        shape = array_call.args[0]
        self.assertEqual(len(shape.elts), 1)
        self.assertIsInstance(shape.elts[0], ast.Constant)
        self.assertEqual(shape.elts[0].value, 2)

        # dtype keyword is ll.Tensor[ll.float32] (const-folded to a TensorType
        # constant) and scope is "local".
        dtype_kw = next(kw for kw in array_call.keywords if kw.arg == "dtype")
        scope_kw = next(kw for kw in array_call.keywords if kw.arg == "scope")
        self.assertIsInstance(dtype_kw.value, ast.Constant)
        self.assertEqual(dtype_kw.value.value.element_type, ll.float32)
        self.assertIsInstance(scope_kw.value, ast.Constant)
        self.assertEqual(scope_kw.value.value, "local")

        # Two element stores xs[0] = ll.empty(...) and xs[1] = ll.empty(...).
        self.assertEqual(sorted(_elem_store_targets(tree)), [0, 1])

    def test_name_range_bound_from_ctx_flattens(self):
        """A plain ctx name bound in globals is a non-Constant evaluable arg."""
        tree = _flatten("COUNT", extra_ctx={"COUNT": 2})
        arrays = _empty_array_binds(tree)
        self.assertEqual([name for name, _ in arrays], ["xs"])
        shape = arrays[0][1].args[0]
        self.assertEqual(shape.elts[0].value, 2)
        self.assertEqual(sorted(_elem_store_targets(tree)), [0, 1])
        self.assertEqual([n for n in ast.walk(tree) if isinstance(n, ast.ListComp)], [])

    def test_undefined_name_range_bound_leaves_listcomp(self):
        """An unevaluable bound must skip the transform without raising."""
        tree = _flatten("MISSING // 2")
        listcomps = [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)]
        self.assertEqual(len(listcomps), 1)
        self.assertEqual(_empty_array_binds(tree), [])

    def test_non_evaluable_call_range_bound_leaves_listcomp(self):
        """A call to an unknown callable is not evaluable: no exception."""
        tree = _flatten("int(unknown_fn(N))")
        listcomps = [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)]
        self.assertEqual(len(listcomps), 1)
        self.assertEqual(_empty_array_binds(tree), [])

    def test_constant_range_bound_still_flattens(self):
        """The pre-existing Constant path keeps working unchanged."""
        tree = _flatten("2")
        arrays = _empty_array_binds(tree)
        self.assertEqual([name for name, _ in arrays], ["xs"])
        self.assertEqual(arrays[0][1].args[0].elts[0].value, 2)
        self.assertEqual(sorted(_elem_store_targets(tree)), [0, 1])

    def test_range_start_end_from_ctx_exprs_flattens(self):
        """Both bound args may be evaluable non-Constants (range(start, end))."""
        tree = _flatten("START, START + 2", extra_ctx={"START": 1})
        arrays = _empty_array_binds(tree)
        self.assertEqual([name for name, _ in arrays], ["xs"])
        # count = end - start = (1 + 2) - 1 = 2
        self.assertEqual(arrays[0][1].args[0].elts[0].value, 2)
        self.assertEqual(sorted(_elem_store_targets(tree)), [0, 1])

    def test_range_expression_names_are_looked_up_read_only(self):
        """Evaluating the bound in a copy of ctx leaves the caller's dict
        unchanged apart from the deliberate ``flatten_empty_ll`` injection
        performed by the pass constructor."""
        ctx = {"ll": ll, "N": 4}
        before = dict(ctx)
        tree = flatten_empty(ast.parse(textwrap.dedent(_MODULE_TEMPLATE.format(bound="N // 2"))), ctx)
        new_keys = {k for k in ctx if k not in before}
        self.assertEqual(new_keys, {"flatten_empty_ll"})
        for key, value in before.items():
            self.assertIs(ctx[key], value)
        self.assertEqual([name for name, _ in _empty_array_binds(tree)], ["xs"])


if __name__ == "__main__":
    unittest.main()
