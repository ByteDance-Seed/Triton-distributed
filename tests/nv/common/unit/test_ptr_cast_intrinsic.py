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
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
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

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
import little_kernel.language.intrin.dtype as dtype_intrin


class TestPtrCastIntrinsic(unittest.TestCase):
    """Regression tests for the generic ``ll.ptr_cast`` reinterpret intrinsic.

    ``ptr_cast`` must lower to a value-returning ``reinterpret_cast<T>(ptr)``
    expression via ``Builtin.return_val`` (the ``Builtin`` dataclass only has
    ``{body, includes, return_val}``).  A stale implementation passed a
    removed ``stmt=`` kwarg, so *any* ``ll.ptr_cast`` call failed CUDA codegen
    with ``TypeError: Builtin.__init__() got an unexpected keyword argument
    'stmt'``.  These tests pin the value-returning behaviour at both the
    codegen-function level and the full end-to-end emitter level.
    """

    def test_ptr_cast_is_builtin(self):
        self.assertTrue(getattr(dtype_intrin.ptr_cast, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.ptr_cast, BUILTIN_ATTR, False))
        self.assertTrue(callable(getattr(dtype_intrin.ptr_cast, CODEGEN_FUNC_ATTR)))

    def test_ptr_cast_codegen_returns_builtin_with_return_val(self):
        # At the emitter, call arguments are already visited to their C++
        # text, so the dtype argument arrives as a string.
        codegen = getattr(dtype_intrin.ptr_cast, CODEGEN_FUNC_ATTR)
        result = codegen("x", "uint32_t*")
        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.body, "")
        self.assertEqual(result.includes, [])
        self.assertEqual(result.return_val, "reinterpret_cast<uint32_t*>(x)")

    def test_ptr_cast_codegen_accepts_lltype_dtype(self):
        # Defensive path: a raw LLType argument (not yet visited to text) is
        # converted through the CUDA type mapper.
        codegen = getattr(dtype_intrin.ptr_cast, CODEGEN_FUNC_ATTR)
        result = codegen("x", ll.ptr[ll.uint32])
        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.return_val, "reinterpret_cast<uint32_t*>(x)")

    def test_annotated_assignment_emits_reinterpret_cast(self):
        """``p: ll.ptr[T] = ll.ptr_cast(x, ...)`` emits a reinterpret_cast."""

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.uint32], n: ll.int32) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            p32: ll.ptr[ll.uint32] = ll.ptr_cast(x, ll.ptr[ll.uint32])
            v: ll.uint32 = p32[tid]
            y[tid] = v

        code = k.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("uint32_t* p32 = reinterpret_cast<uint32_t*>(x);", code)

    def test_subscript_target_store_emits_reinterpret_cast(self):
        """``ll.ptr_cast(b, ll.ptr[U])[i] = v`` emits a reinterpret_cast."""

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k2(b: ll.Tensor[ll.uint8], v: ll.uint32, idx: ll.int32) -> ll.void:
            ll.ptr_cast(b, ll.ptr[ll.uint32])[idx] = v

        code = k2.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("reinterpret_cast<uint32_t*>(b)[idx] = v;", code)


if __name__ == "__main__":
    unittest.main()
