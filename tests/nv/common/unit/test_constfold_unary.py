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
"""ConstantFolder folds unary operators on compile-time scalars.

`-5`, `~3`, `not False` and `+(2+1)` are genuinely static IR facts and must
collapse to a single ``ast.Constant`` (previously the missing ``visit_UnaryOp``
made ``x: const = -5`` raise "Can't assign non-const value to const variable").
Runtime operands (e.g. thread indices) must NOT be folded.
"""

import ast
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language import const
from little_kernel.core.passes.constfold import const_fold


def _fold(src, extra_ctx=None):
    ctx = {"const": const}
    if extra_ctx:
        ctx.update(extra_ctx)
    return ast.unparse(const_fold(ast.parse(src), ctx=ctx))


class ConstFoldUnaryTest(unittest.TestCase):

    def test_usub_const_annotation(self):
        self.assertIn("x: const = -5", _fold("x: const = -5"))

    def test_uadd_const_annotation(self):
        self.assertIn("x: const = 7", _fold("x: const = +7"))

    def test_invert_const_annotation(self):
        self.assertIn("w: const = -4", _fold("w: const = ~3"))

    def test_not_const_annotation(self):
        self.assertIn("b: const = True", _fold("b: const = not False"))

    def test_nested_unary_over_binop(self):
        # -(2*10+1) -> -21 (BinOp folds first, then the USub)
        self.assertIn("big: const = -21", _fold("big: const = -(2 * 10 + 1)"))

    def test_chained_fold_via_const_name(self):
        # y folds to 5, then -y folds to -5
        out = _fold("y: const = 2 + 3\nz: const = -y")
        self.assertIn("y: const = 5", out)
        self.assertIn("z: const = -5", out)

    def test_runtime_operand_not_folded(self):
        # A runtime thread index must keep its unary op (no constant to fold).
        ctx = dict(vars(ll))
        ctx["ll_kernel"] = lk.ll_kernel
        ctx["ll"] = ll
        src = ('@ll_kernel(backend="cuda", is_entry=True)\n'
               "def k(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32], "
               "n: ll.int32) -> ll.void:\n"
               "    i: ll.int32 = ll.threadIdx_x()\n"
               "    m: ll.int32 = -i\n"
               "    p: ll.int32 = -(i + 1)\n"
               "    c: ll.int32 = -(4 + 1)\n"
               "    if i < n:\n"
               "        out[i] = a[i] + m + p + c\n")
        out = ast.unparse(const_fold(ast.parse(src), ctx=ctx))
        # runtime unary preserved
        self.assertIn("m: ll.int32 = -i", out)
        self.assertIn("p: ll.int32 = -(i + 1)", out)
        # static unary folded
        self.assertIn("c: ll.int32 = -5", out)


if __name__ == "__main__":
    unittest.main()
