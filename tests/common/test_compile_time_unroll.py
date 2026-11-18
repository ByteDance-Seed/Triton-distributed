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
"""CPU-only regression tests for ``compile_time_unroll``'s AnnAssign folding.

A single-assigned ``AnnAssign`` whose value folds to an int/bool is erased when
every ``Load`` of the name is substituted: the renamed per-copy local then needs
no runtime slot and disappears from the IR (an intended optimization the backend
tests rely on, e.g. ``j: ll.int32 = 2 * i + 1`` folding to a literal).

That erasure is unsound when the (renamed) name is still referenced inside a
nested scope — a nested ``FunctionDef``, ``ClassDef`` or ``Lambda`` — because
``_Substitute`` only rewrites the statements of the block it walks and returns
nested scopes verbatim.  Erasing the declaration there left a dangling
reference to ``x__uN``.  These tests pin both halves: full propagation still
erases, and a surviving nested-scope reference keeps the declaration.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.core.passes.compile_time_unroll import compile_time_unroll


def _unparse(source, ctx):
    return ast.unparse(compile_time_unroll(ast.parse(source), ctx))


def _ctx():
    return {"ll": ll, "const": ll.const}


def _ann_assign_stmts(source, ctx):
    tree = compile_time_unroll(ast.parse(source), ctx)
    return [node for node in ast.walk(tree) if isinstance(node, ast.AnnAssign)]


class TestFullPropagationErases(unittest.TestCase):
    """Intended optimization: a folded single-assigned local with no surviving
    reference is erased and substituted everywhere, regardless of annotation."""

    def test_int_annotated_local_is_folded_and_erased(self):
        out = _unparse("def f():\n"
                       "    for i in ll.unroll(range(2)):\n"
                       "        x: int = 1\n"
                       "        ll.foo(x)\n", _ctx())
        self.assertEqual(out.count("ll.foo(1)"), 2)
        self.assertNotIn("x__u1", out)

    def test_ll_int32_style_local_disappears(self):
        # The shape the HIP/CUDA backend tests depend on: a folded per-copy
        # index local leaves only literal indices in the emitted body.
        out = _unparse(
            "def f():\n"
            "    for i in ll.unroll(range(2)):\n"
            "        j: int = 2 * i + 1\n"
            "        ll.foo(j)\n", _ctx())
        self.assertIn("ll.foo(1)", out)
        self.assertIn("ll.foo(3)", out)
        self.assertNotIn("j__u", out)

    def test_const_annotated_local_still_erased(self):
        out = _unparse(
            "def f():\n"
            "    for i in ll.unroll(range(2)):\n"
            "        x: ll.const = 1\n"
            "        ll.foo(x)\n", _ctx())
        self.assertEqual(out.count("ll.foo(1)"), 2)
        self.assertNotIn("x__u1", out)


class TestNestedScopeKeepsDeclaration(unittest.TestCase):
    """Regression for the dangling reference: a folded single-assigned local
    still referenced inside a nested scope must keep its declaration."""

    def _assert_kept(self, source, name):
        out = _unparse(source, _ctx())
        for copy_id in (1, 2):
            renamed = f"{name}__u{copy_id}"
            self.assertIn(f"{renamed}: int = 1", out)
        return out

    def test_nested_function_def_keeps_declaration(self):
        out = self._assert_kept(
            "def f():\n"
            "    for i in ll.unroll(range(2)):\n"
            "        x: int = 1\n"
            "        def g():\n"
            "            return x\n", "x")
        self.assertIn("return x__u1", out)
        self.assertIn("return x__u2", out)

    def test_nested_class_def_keeps_declaration(self):
        out = self._assert_kept(
            "def f():\n"
            "    for i in ll.unroll(range(2)):\n"
            "        x: int = 1\n"
            "        class C:\n"
            "            y = x\n", "x")
        self.assertIn("y = x__u1", out)
        self.assertIn("y = x__u2", out)

    def test_lambda_keeps_declaration(self):
        out = self._assert_kept(
            "def f():\n"
            "    for i in ll.unroll(range(2)):\n"
            "        x: int = 1\n"
            "        g = lambda: x\n", "x")
        self.assertIn("lambda: x__u1", out)
        self.assertIn("lambda: x__u2", out)

    def test_no_dangling_reference_nested_def(self):
        # Every renamed load must resolve to a declaration in the emitted IR.
        out = _unparse(
            "def f():\n"
            "    for i in ll.unroll(range(2)):\n"
            "        x: int = 1\n"
            "        def g():\n"
            "            return x\n", _ctx())
        for copy_id in (1, 2):
            renamed = f"x__u{copy_id}"
            self.assertIn(f"{renamed}: int = 1", out)
            self.assertIn(f"return {renamed}", out)

    def test_outside_unroll_nested_reference_kept(self):
        out = _unparse("def f():\n"
                       "    x: int = 5\n"
                       "    def g():\n"
                       "        return x\n", _ctx())
        self.assertIn("x: int = 5", out)
        self.assertIn("return x", out)


class TestDeclarationsPreserved(unittest.TestCase):
    """Declarations that cannot be folded away are kept, one per copy."""

    def test_non_evaluable_value_keeps_declaration(self):
        out = _unparse(
            "def f(acc):\n"
            "    for i in ll.unroll(range(2)):\n"
            "        k: int = acc[i]\n"
            "        ll.foo(k)\n", _ctx())
        self.assertIn("k__u1: int = acc[0]", out)
        self.assertIn("k__u2: int = acc[1]", out)

    def test_reassigned_local_keeps_declaration(self):
        # assigned twice -> not single-assigned -> declaration stays
        out = _unparse(
            "def f(rt_slot):\n"
            "    for i in ll.unroll(range(2)):\n"
            "        x: int = 1\n"
            "        x = rt_slot[i]\n", _ctx())
        self.assertIn("x__u1: int = 1", out)
        self.assertIn("x__u2: int = 1", out)

    def test_per_copy_declarations_have_distinct_names(self):
        stmts = _ann_assign_stmts(
            "def f(acc):\n"
            "    for i in ll.unroll(range(3)):\n"
            "        k: int = acc[i]\n"
            "        ll.foo(k)\n", _ctx())
        names = [s.target.id for s in stmts]
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertNotIn("k", names)


class TestUnaffectedBehavior(unittest.TestCase):

    def test_plain_assign_statement_is_preserved(self):
        # A plain Assign is never erased by this pass; only AnnAssign folds.
        out = _unparse("def f():\n"
                       "    x = 7\n"
                       "    ll.foo(x)\n", _ctx())
        self.assertIn("x = 7", out)

    def test_unroll_loop_var_substituted(self):
        out = _unparse("def f():\n"
                       "    for i in ll.unroll(range(2)):\n"
                       "        ll.foo(i)\n", _ctx())
        self.assertIn("ll.foo(0)", out)
        self.assertIn("ll.foo(1)", out)


if __name__ == "__main__":
    unittest.main()
