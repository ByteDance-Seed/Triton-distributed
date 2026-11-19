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
"""Regression tests for local (register-scope) arrays bound by an annotated
assignment (``rv: ll.Tensor[dtype] = ll.empty(..., scope="local")``).

``mem_analysis`` / ``insert_mem_alloc`` previously recognized only plain
``ast.Assign`` targets, so the annotated form raised
'll.empty call is not assigned to a variable ... Maybe should use flatten_empty
pass first.' at the insert_mem_alloc pass. An annotated assignment binds the
same single buffer name and must lower to a flat local C array whose unrolled
subscripts use per-iteration compile-time-constant indices, matching CUDA
local-array register promotion.
"""

import re
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES


def _compile(func, num_threads=128):
    kernel = lk.ll_kernel(backend="cuda", is_entry=True)(func)
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=num_threads)


class TestLocalArrayAnnotatedAssign(unittest.TestCase):

    def test_annotated_local_array_declares_register_array(self):
        """The pinned reproducer: AnnAssign of ll.empty(scope='local')."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            rv: ll.Tensor[ll.float32] = ll.empty([32], ll.float32, scope="local")
            tid: ll.int32 = ll.threadIdx_x()
            rv[0] = 3.0
            y[tid] = rv[0]

        cuda = _compile(k, num_threads=32)
        # The annotated binding becomes the same flat local C declaration as
        # a plain assignment; no ll.empty call survives into the output.
        self.assertIn("float rv[32];", cuda)
        self.assertNotIn("ll.empty", cuda)
        self.assertIn("rv[0] = 3.0f;", cuda)
        self.assertIn("y[tid] = rv[0];", cuda)

    def test_annotated_local_array_under_full_unroll(self):
        """Subscript access after ll.unroll uses compile-time-constant indices."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            rv: ll.Tensor[ll.float32] = ll.empty([8], ll.float32, scope="local")
            for i in ll.unroll(range(8)):
                rv[i] = ll.float32(i)
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = rv[7]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float rv[8];", cuda)
        # The loop is fully unrolled: each iteration binds its index to a
        # literal (0..7), so every rv[i] subscript is a compile-time constant
        # eligible for register promotion rather than a loop-variable index.
        for lit in range(8):
            self.assertRegex(cuda, r"\{\s*int32_t i = %d;\s*rv\[i\] = static_cast<float>\(i\);\s*\}" % lit)
        self.assertIn("y[tid] = rv[7];", cuda)
        # No loop construct survives full unrolling.
        self.assertNotIn("for (", cuda)

    def test_annotated_local_array_in_nested_block(self):
        """AnnAssign replacement also applies inside if/for bodies."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            if tid < 32:
                rv: ll.Tensor[ll.float32] = ll.empty([4], ll.float32, scope="local")
                for i in ll.unroll(range(4)):
                    rv[i] = ll.float32(i + 1)
                y[tid] = rv[tid & 3]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float rv[4];", cuda)
        self.assertNotIn("ll.empty", cuda)
        self.assertRegex(cuda, r"int32_t i = 0;")
        self.assertRegex(cuda, r"int32_t i = 3;")

    def test_annotated_zeros_local_matches_plain_path(self):
        """ll.zeros with an annotated target shares the alloc_local_zeros path."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            rv: ll.Tensor[ll.float32] = ll.zeros([4], dtype=ll.float32, scope="local")
            rv[2] = 5.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = rv[2]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float rv[4]", cuda)
        self.assertIn("rv[2] = 5.0f;", cuda)
        self.assertIn("y[tid] = rv[2];", cuda)

    def test_annotated_local_array_rank2_flattens(self):
        """Rank-2 flattening is independent of Assign vs AnnAssign."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a: ll.Tensor[ll.float32] = ll.empty([2, 4], dtype=ll.float32, scope="local")
            for i in ll.unroll(range(2)):
                for j in ll.unroll(range(4)):
                    a[i, j] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[1, 3]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float a[8];", cuda)
        self.assertIn("a[(i) * 4 + (j)] = 1.0f;", cuda)
        self.assertIn("y[tid] = a[(1) * 4 + (3)];", cuda)
        self.assertNotIn("][", cuda)


if __name__ == "__main__":
    unittest.main()
