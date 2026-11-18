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
"""Unit tests for the fp16 value pack/unpack intrinsics (pack_f16x2 /
unpack_f16x2_f32) and the .cs (evict-first) streaming variants of the
dtype-agnostic 16-byte GMEM intrinsics (gmem_ld_v4_u32_cs /
gmem_st_v4_u32_cs).

Legality invariants under test:
  * pack_f16x2 converts two float32 values to one uint32 of two fp16 lanes
    with round-to-nearest (__floats2half2_rn == cvt.rn.f16x2.f32 PTX); the
    fp32->fp16 value conversion lives here, never in the bit moves
    (pack_u32_2x16 stays bit-level);
  * unpack_f16x2_f32 converts both fp16 lanes of one uint32 to float32
    (__low2float/__high2float == cvt.f32.f16); fp16->fp32 is exact;
  * lane layout is the __half2 / v4.b32 stream layout: lo in bits [15:0],
    hi in bits [31:16], matching pack/unpack between the two families;
  * the _cs variants emit ld.global.cs.v4.b32 / st.global.cs.v4.b32 with
    identical operand shapes and the same (const void*) dtype-agnostic
    pointer form as the unhinted intrinsics — the cache op is the only
    difference, so they are semantics-preserving (values unchanged);
  * all four are exported on the language surface (ll.*).
"""

import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm_intrin
import little_kernel.backends.nv.common.language.intrin.memory as memory_intrin
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin


class TestF16x2CvtIntrinsics(unittest.TestCase):
    """fp16<->fp32 value conversions on u32 lane pairs."""

    def test_pack_f16x2_is_builtin(self):
        self.assertTrue(getattr(cuda_asm_intrin.pack_f16x2, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.pack_f16x2, BUILTIN_ATTR, False))

    def test_unpack_f16x2_f32_is_builtin(self):
        self.assertTrue(getattr(cuda_asm_intrin.unpack_f16x2_f32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.unpack_f16x2_f32, BUILTIN_ATTR, False))

    def test_pack_f16x2_codegen(self):
        codegen_func = getattr(cuda_asm_intrin.pack_f16x2, CODEGEN_FUNC_ATTR)
        result = codegen_func("a", "b")
        self.assertIsInstance(result, Builtin)
        # RNE fp32x2 -> fp16x2 conversion intrinsic (== cvt.rn.f16x2.f32)
        self.assertIn("__floats2half2_rn", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        # Call site passes both float operands by value, no extra masking
        self.assertEqual(result.return_val, "pack_f16x2_fn(a, b)")

    def test_unpack_f16x2_f32_codegen(self):
        codegen_func = getattr(cuda_asm_intrin.unpack_f16x2_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("bits", "lo", "hi")
        self.assertIsInstance(result, Builtin)
        # Value conversion per lane (cvt.f32.f16), lo from low half
        self.assertIn("__low2float", result.body)
        self.assertIn("__high2float", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertEqual(result.return_val, "unpack_f16x2_f32_fn(bits, &lo, &hi)")

    def test_pack_unpack_lane_layout_consistent(self):
        # pack takes (lo, hi) floats; unpack writes (lo, hi) with lo the
        # __half2 low half — the same bits [15:0] convention as
        # pack_u32_2x16 / unpack_u32_2x16.
        pack_body = getattr(cuda_asm_intrin.pack_f16x2, CODEGEN_FUNC_ATTR)("a", "b").body
        unpack_body = getattr(cuda_asm_intrin.unpack_f16x2_f32, CODEGEN_FUNC_ATTR)("bits", "lo", "hi").body
        self.assertIn("__floats2half2_rn(lo, hi)", pack_body)
        self.assertIn("*lo = __low2float(h2);", unpack_body)
        self.assertIn("*hi = __high2float(h2);", unpack_body)

    def test_pure_bit_moves_unchanged(self):
        # pack_u32_2x16 must remain bit-level (no cvt) so callers keep
        # rounding control; fp16 value moves route through pack_f16x2.
        bit_body = getattr(cuda_asm_intrin.pack_u32_2x16, CODEGEN_FUNC_ATTR)("lo", "hi").body
        self.assertIn("mov.b32 %0, {%1, %2};", bit_body)
        self.assertNotIn("cvt", bit_body)


class TestCsStreamingV4Intrinsics(unittest.TestCase):
    """.cs evict-first variants of the dtype-agnostic 16-byte GMEM ops."""

    def test_ld_cs_is_builtin(self):
        self.assertTrue(getattr(memory_intrin.gmem_ld_v4_u32_cs, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.gmem_ld_v4_u32_cs, BUILTIN_ATTR, False))

    def test_st_cs_is_builtin(self):
        self.assertTrue(getattr(memory_intrin.gmem_st_v4_u32_cs, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.gmem_st_v4_u32_cs, BUILTIN_ATTR, False))

    def test_ld_cs_codegen(self):
        codegen_func = getattr(memory_intrin.gmem_ld_v4_u32_cs, CODEGEN_FUNC_ATTR)
        result = codegen_func("x", "i", "r0", "r1", "r2", "r3")
        self.assertIsInstance(result, Builtin)
        self.assertIn("ld.global.cs.v4.b32", result.body)
        # dtype-agnostic pointer form, same as the unhinted load
        self.assertIn("const void*", result.body)
        self.assertEqual(result.return_val, "gmem_ld_v4_u32_cs_fn((const void*)(x + i), &r0, &r1, &r2, &r3)")

    def test_st_cs_codegen(self):
        codegen_func = getattr(memory_intrin.gmem_st_v4_u32_cs, CODEGEN_FUNC_ATTR)
        result = codegen_func("y", "i", "v0", "v1", "v2", "v3")
        self.assertIsInstance(result, Builtin)
        self.assertIn("st.global.cs.v4.b32", result.body)
        self.assertIn('"l"(ptr)', result.body)
        self.assertIn('"memory"', result.body)
        self.assertEqual(result.return_val, "gmem_st_v4_u32_cs_fn((void*)(y + i), v0, v1, v2, v3)")

    def test_cs_only_differs_in_cache_op(self):
        # Semantics-preserving claim: operand shapes and access size are
        # identical to the plain variants; only the PTX cache op changes.
        plain_ld = getattr(memory_intrin.gmem_ld_v4_u32, CODEGEN_FUNC_ATTR)("x", "i", "r0", "r1", "r2", "r3")
        cs_ld = getattr(memory_intrin.gmem_ld_v4_u32_cs, CODEGEN_FUNC_ATTR)("x", "i", "r0", "r1", "r2", "r3")
        self.assertEqual(
            plain_ld.body.replace("gmem_ld_v4_u32_fn", "FN").replace("ld.global.v4.b32", "OP"),
            cs_ld.body.replace("gmem_ld_v4_u32_cs_fn", "FN").replace("ld.global.cs.v4.b32", "OP"))
        plain_st = getattr(memory_intrin.gmem_st_v4_u32, CODEGEN_FUNC_ATTR)("x", "i", "v0", "v1", "v2", "v3")
        cs_st = getattr(memory_intrin.gmem_st_v4_u32_cs, CODEGEN_FUNC_ATTR)("x", "i", "v0", "v1", "v2", "v3")
        self.assertEqual(
            plain_st.body.replace("gmem_st_v4_u32_fn", "FN").replace("st.global.v4.b32", "OP"),
            cs_st.body.replace("gmem_st_v4_u32_cs_fn", "FN").replace("st.global.cs.v4.b32", "OP"))


if __name__ == "__main__":
    unittest.main()
