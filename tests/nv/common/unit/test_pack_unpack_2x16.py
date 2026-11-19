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
# publish, distribute, sublicense, and/or sell copies of the Software.
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
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
"""Unit tests for bit-level 16-bit lane pack/unpack intrinsics
(unpack_u32_2x16 / pack_u32_2x16).

Legality invariants under test:
  * codegen emits the exact lane-move PTX (mov.b32 {h0,h1}, r /
    mov.b32 r, {h0,h1}) with "h" (b16 register) constraints for lanes and
    "r" (b32 register) constraints for the packed word — register class
    matches dtype width at every operand;
  * unpack zero-extends lanes into uint32 outputs (no garbage upper bits);
  * pack masks inputs to 16 bits first, so callers keep full control of
    rounding/semantics (bit-level, no hidden conversion);
  * exported on the language surface (ll.unpack_u32_2x16 / ll.pack_u32_2x16).
"""

import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm_intrin
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin


class TestPackUnpack2x16Intrinsics(unittest.TestCase):
    """Tests for dtype-agnostic 16-bit lane pack/unpack builtins."""

    def test_unpack_is_builtin(self):
        self.assertTrue(getattr(cuda_asm_intrin.unpack_u32_2x16, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.unpack_u32_2x16, BUILTIN_ATTR, False))

    def test_pack_is_builtin(self):
        self.assertTrue(getattr(cuda_asm_intrin.pack_u32_2x16, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.pack_u32_2x16, BUILTIN_ATTR, False))

    def test_unpack_codegen(self):
        codegen_func = getattr(cuda_asm_intrin.unpack_u32_2x16, CODEGEN_FUNC_ATTR)
        result = codegen_func("bits", "lo", "hi")
        self.assertIsInstance(result, Builtin)
        self.assertIn("mov.b32 {%0, %1}, %2;", result.body)
        self.assertIn('"=h"(lo16)', result.body)
        self.assertIn('"=h"(hi16)', result.body)
        self.assertIn('"r"(bits)', result.body)
        # zero-extend into the uint32 outputs
        self.assertIn("*lo = (uint32_t)lo16;", result.body)
        self.assertIn("*hi = (uint32_t)hi16;", result.body)
        self.assertIn("unpack_u32_2x16_fn(bits, &lo, &hi)", result.return_val)

    def test_pack_codegen(self):
        codegen_func = getattr(cuda_asm_intrin.pack_u32_2x16, CODEGEN_FUNC_ATTR)
        result = codegen_func("lo", "hi")
        self.assertIsInstance(result, Builtin)
        self.assertIn("mov.b32 %0, {%1, %2};", result.body)
        self.assertIn('"=r"(bits)', result.body)
        self.assertIn('"h"(lo16)', result.body)
        self.assertIn('"h"(hi16)', result.body)
        # inputs are masked to 16 bits, no hidden conversion
        self.assertIn("lo & 0xFFFFu", result.body)
        self.assertIn("hi & 0xFFFFu", result.body)
        self.assertIn("pack_u32_2x16_fn(lo, hi)", result.return_val)

    def test_generated_cuda_contains_mov_b32(self):
        """End-to-end text check: a draft kernel unpacking and repacking a
        128-bit vector load must emit lane-move PTX in the final CUDA."""
        import little_kernel as lk
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def roundtrip_v4(src: ll.Tensor[ll.bfloat16], dst: ll.Tensor[ll.bfloat16]) -> ll.void:
            base: ll.int32 = ll.threadIdx_x() * 8
            r0: ll.uint32 = 0
            r1: ll.uint32 = 0
            r2: ll.uint32 = 0
            r3: ll.uint32 = 0
            ll.gmem_ld_v4_u32(src, base, r0, r1, r2, r3)
            lo0: ll.uint32 = 0
            hi0: ll.uint32 = 0
            ll.unpack_u32_2x16(r0, lo0, hi0)
            r0 = ll.pack_u32_2x16(lo0, hi0)
            ll.gmem_st_v4_u32(dst, base, r0, r1, r2, r3)

        code = roundtrip_v4.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("ld.global.v4.b32", code)
        self.assertIn("st.global.v4.b32", code)
        self.assertIn("mov.b32 {%0, %1}, %2;", code)
        self.assertIn("mov.b32 %0, {%1, %2};", code)


if __name__ == "__main__":
    unittest.main()
