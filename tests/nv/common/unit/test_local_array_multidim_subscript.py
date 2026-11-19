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
"""Regression tests for multi-dimensional local (register-scope) array codegen.

`insert_mem_alloc` flattens the declaration of `ll.empty([R, C], scope="local")`
(and zeros/full) to a 1-D C array (`float a[R * C];`). Multi-dimensional
subscripts of those arrays must lower to the matching flat row-major index
(`a[(i) * C + (j)]`), not the invalid nested subscript `a[i][j]` that NVCC
rejects with 'expression must have pointer-to-object type but it has type float'.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES


def _compile(func, num_threads=128):
    kernel = lk.ll_kernel(backend="cuda", is_entry=True)(func)
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=num_threads)


class TestLocalArrayMultidimSubscript(unittest.TestCase):

    def test_rank2_local_array_flattens_under_full_unroll(self):
        """The pinned reproducer: rank-2 local array, fully unrolled loops."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([2, 4], dtype=ll.float32, scope="local")
            for i in ll.unroll(range(2)):
                for j in ll.unroll(range(4)):
                    a[i, j] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[1, 3]

        cuda = _compile(k, num_threads=32)
        # Declaration stays a flat 1-D C array.
        self.assertIn("float a[8];", cuda)
        # Store and constant-index load are flattened row-major.
        self.assertIn("a[(i) * 4 + (j)] = 1.0f;", cuda)
        self.assertIn("y[tid] = a[(1) * 4 + (3)];", cuda)
        # No two-dimensional C subscript of the flat array survives.
        self.assertNotIn("a[i][j]", cuda)
        self.assertNotIn("a[1][3]", cuda)

    def test_rank2_local_array_rolled_loop(self):
        """Flattening applies without unrolling as well."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([3, 5], dtype=ll.float32, scope="local")
            for i in range(3):
                for j in range(5):
                    a[i, j] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[2, 4]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float a[15];", cuda)
        self.assertIn("a[(i) * 5 + (j)] = 1.0f;", cuda)
        self.assertIn("y[tid] = a[(2) * 5 + (4)];", cuda)
        self.assertNotIn("][", cuda)

    def test_rank3_local_array_flattens_with_trailing_strides(self):
        """Rank-3: index_k is scaled by the product of trailing extents."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([2, 3, 4], dtype=ll.float32, scope="local")
            for i in ll.unroll(range(2)):
                for j in ll.unroll(range(3)):
                    for l in ll.unroll(range(4)):
                        a[i, j, l] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[1, 2, 3]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float a[24];", cuda)
        self.assertIn("a[(i) * 12 + (j) * 4 + (l)] = 1.0f;", cuda)
        self.assertIn("y[tid] = a[(1) * 12 + (2) * 4 + (3)];", cuda)
        self.assertNotIn("][", cuda)

    def test_zeros_local_rank2_flattens(self):
        """ll.zeros(..., scope=\"local\") shares the same flat declaration."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.zeros([2, 4], dtype=ll.float32, scope="local")
            a[1, 2] = 3.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[1, 2]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float a[8]", cuda)
        self.assertIn("a[(1) * 4 + (2)] = 3.0f;", cuda)
        self.assertIn("y[tid] = a[(1) * 4 + (2)];", cuda)
        self.assertNotIn("][", cuda)

    def test_rank1_local_array_unchanged(self):
        """Rank-1 local arrays keep their existing 1-D subscript emission."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([8], dtype=ll.float32, scope="local")
            for i in range(8):
                a[i] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[3]

        cuda = _compile(k, num_threads=32)
        self.assertIn("float a[8];", cuda)
        self.assertIn("a[i] = 1.0f;", cuda)
        self.assertIn("y[tid] = a[3];", cuda)

    def test_rank2_shared_array_keeps_legacy_emission(self):
        """Shared-scope buffers are not register arrays: legacy path untouched.

        (Shared buffers of builtin dtype are addressed through a pointer whose
        2-D subscript handling is unchanged by this fix; this test pins that
        the local-array flattening does not leak across scopes by name.)
        """

        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([2, 4], dtype=ll.float32, scope="shared")
            tid: ll.int32 = ll.threadIdx_x()
            if tid < 8:
                a[tid // 4, tid % 4] = x[tid]
            if tid < 8:
                y[tid] = a[tid // 4, tid % 4]

        cuda = _compile(k, num_threads=32)
        # Shared buffer uses the existing addressing; no flat a[... * 4 ...]
        # register-array subscript is introduced for the shared name.
        self.assertNotIn("a[(tid / 4) * 4 + (tid % 4)]", cuda)

    def test_arity_mismatch_raises(self):
        """Wrong number of indices for a recorded local array is an error."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([2, 4], dtype=ll.float32, scope="local")
            a[0, 1, 2] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[0, 0]

        with self.assertRaises(IndexError):
            _compile(k, num_threads=32)

    def test_single_subscript_of_rank2_local_raises(self):
        """a[i] on a rank-2 local would silently read flat element i."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([2, 4], dtype=ll.float32, scope="local")
            for i in ll.unroll(range(2)):
                for j in ll.unroll(range(4)):
                    a[i, j] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[1]

        with self.assertRaises(IndexError):
            _compile(k, num_threads=32)

    def test_tuple_subscript_of_rank1_local_raises(self):
        """a[i, j] on a rank-1 local would emit invalid a[i][j] C code."""

        def k(y: ll.Tensor[ll.float32]) -> ll.void:
            a = ll.empty([8], dtype=ll.float32, scope="local")
            for i in range(8):
                a[i] = 1.0
            tid: ll.int32 = ll.threadIdx_x()
            y[tid] = a[1, 2]

        with self.assertRaises(IndexError):
            _compile(k, num_threads=32)


if __name__ == "__main__":
    unittest.main()
