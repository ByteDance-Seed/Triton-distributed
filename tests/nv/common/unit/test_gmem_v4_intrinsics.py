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
# (the "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Unit tests for dtype-agnostic 128-bit GMEM intrinsics
(gmem_ld_v4_u32 / gmem_st_v4_u32).

Legality invariants under test:
  * codegen emits guaranteed-vectorized PTX (ld/st.global.v4.b32), not a
    reinterpret_cast that a compiler could split;
  * pointer operand uses the 64-bit "l" constraint (global addresses are
    64-bit) and carries a "memory" clobber (no illegal reordering);
  * the call site casts the typed element pointer to (const void*), so the
    builtin is usable on any element dtype (fp16/bf16/fp32/int...).
Alignment (16 B) remains a caller contract per docstring — mirror of the
in-tree smem_ld_128/smem_st_128 mechanism.
"""

import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.memory as memory_intrin
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin


class TestGmemV4U32Intrinsics(unittest.TestCase):
    """Tests for dtype-agnostic 128-bit global vector intrinsics."""

    def test_gmem_ld_v4_u32_is_builtin(self):
        self.assertTrue(getattr(memory_intrin.gmem_ld_v4_u32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.gmem_ld_v4_u32, BUILTIN_ATTR, False))

    def test_gmem_st_v4_u32_is_builtin(self):
        self.assertTrue(getattr(memory_intrin.gmem_st_v4_u32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.gmem_st_v4_u32, BUILTIN_ATTR, False))

    def test_gmem_ld_v4_u32_codegen(self):
        codegen_func = getattr(memory_intrin.gmem_ld_v4_u32, CODEGEN_FUNC_ATTR)
        result = codegen_func("t", "i", "a", "b", "c", "d")
        self.assertIsInstance(result, Builtin)
        self.assertIn("ld.global.v4.b32", result.body)
        self.assertIn('"l"(ptr)', result.body)
        self.assertIn('"memory"', result.body)
        self.assertIn("(const void*)(t + i)", result.return_val)
        self.assertIn("&a", result.return_val)
        self.assertIn("&d", result.return_val)

    def test_gmem_st_v4_u32_codegen(self):
        codegen_func = getattr(memory_intrin.gmem_st_v4_u32, CODEGEN_FUNC_ATTR)
        result = codegen_func("t", "i", "a", "b", "c", "d")
        self.assertIsInstance(result, Builtin)
        self.assertIn("st.global.v4.b32", result.body)
        self.assertIn('"l"(ptr)', result.body)
        self.assertIn('"memory"', result.body)
        self.assertIn("(void*)(t + i)", result.return_val)

    def test_generated_cuda_contains_v4_ptx(self):
        """End-to-end text check: a draft kernel using both intrinsics must
        emit ld/st.global.v4.b32 in the final CUDA source."""
        import little_kernel as lk
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def copy_v4(src: ll.Tensor[ll.bfloat16], dst: ll.Tensor[ll.bfloat16]) -> ll.void:
            base: ll.int32 = ll.threadIdx_x() * 8
            r0: ll.uint32 = 0
            r1: ll.uint32 = 0
            r2: ll.uint32 = 0
            r3: ll.uint32 = 0
            ll.gmem_ld_v4_u32(src, base, r0, r1, r2, r3)
            ll.gmem_st_v4_u32(dst, base, r0, r1, r2, r3)

        code = copy_v4.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("ld.global.v4.b32", code)
        self.assertIn("st.global.v4.b32", code)


if __name__ == "__main__":
    unittest.main()
