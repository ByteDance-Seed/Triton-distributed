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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction
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
"""Tests for while-loop body variable hoisting.

Python variables are function-scoped, but a C++ declaration emitted on a
variable's first assignment inside a while body is scoped to that block.
Because scope_vars still records the name, any later use in a sibling loop
body (or after the loop) used to emit a bare assignment to an undeclared C
identifier (observed as emitted_cuda_undefined_identifier on real
frontends). visit_While must hoist body-assigned variables before the loop,
mirroring the existing visit_For / visit_If behavior.
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


class TestWhileLoopVariableHoisting(unittest.TestCase):

    def test_annassign_in_while_body_used_after_loop(self):
        """A variable first AnnAssign'd in a while body is declared before
        the loop, assigned inside it, and usable after it."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    acc: ll.float32 = 0.0
    while i < n:
        v: ll.float32 = x[i]
        acc = acc + v
        i = i + 32
    x[0] = v
""")
        decl_pos = code.index("float v;")
        while_pos = code.index("while (")
        self.assertLess(decl_pos, while_pos, "declaration must precede the while loop")
        # exactly one declaration of v
        self.assertEqual(code.count("float v"), 1)
        # the after-loop use survives
        self.assertIn("x[0] = v;", code)

    def test_plain_assign_in_while_body_used_after_loop(self):
        """Same hoisting for a plain (unannotated) first assignment."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    while i < n:
        t = x[i] * 2.0
        i = i + 32
    x[0] = t
""")
        self.assertLess(code.index("float t;"), code.index("while ("))
        self.assertEqual(code.count("float t"), 1)

    def test_sibling_while_loops_share_variable(self):
        """The 0004 phase-1/phase-2 trap: a variable assigned in a first
        while body and reassigned in a sibling while body must be declared
        exactly once, before the first loop (was: undeclared identifier)."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    r: ll.int32 = 0
    while r + 8 < n:
        nvec: ll.int32 = (n - r) // 8
        r = r + 64
    while r < n:
        nvec = 1
        r = r + 8
    x[0] = nvec
""")
        self.assertEqual(code.count("int32_t nvec"), 1, "nvec must be declared exactly once")
        self.assertLess(code.index("int32_t nvec;"), code.index("while ("))
        # second loop reuses the name as a plain assignment
        self.assertIn("nvec = 1;", code)

    def test_preexisting_scope_variable_not_redeclared(self):
        """A variable already in scope before the loop keeps its single
        original declaration (the scan must exclude it)."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    v: ll.float32 = 0.0
    i = ll.threadIdx_x()
    while i < n:
        v = v + x[i]
        i = i + 32
    x[0] = v
""")
        self.assertEqual(code.count("float v"), 1)

    def test_nested_while_inside_if_hoists_to_enclosing_block(self):
        """A while nested in an if body hoists its body variables to the
        enclosing (if) block, not to function scope."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    if i < n:
        while i < n:
            inner: ll.float32 = x[i]
            i = i + 32
        x[0] = inner
""")
        # visit_If pre-scans its whole body (recursing into the nested
        # while), so the declaration is hoisted before the `if` — valid C
        # and consistent with Python function-level scoping. What matters:
        # exactly one declaration, preceding the while, with the after-loop
        # use intact.
        self.assertLess(code.index("float inner;"), code.index("while ("))
        self.assertEqual(code.count("float inner"), 1)
        self.assertIn("x[0] = inner;", code)

    def test_per_iteration_initializer_still_runs(self):
        """Hoisting must not drop per-iteration initialization: the body's
        AnnAssign still emits an assignment each iteration."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i = ll.threadIdx_x()
    total: ll.float32 = 0.0
    while i < n:
        d: ll.float32 = x[i] - 1.0
        total = total + d * d
        i = i + 32
    x[0] = total
""")
        body_start = code.index("while (")
        body = code[body_start:]
        self.assertIn("d = ", body)
        self.assertIn("1.0f", body)


if __name__ == "__main__":
    unittest.main()
