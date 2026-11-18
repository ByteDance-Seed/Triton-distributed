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

import little_kernel.language as ll
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.arith as arith


class TestF16x2Reinterpret(unittest.TestCase):
    """Unit tests for the no-convert fp16 bit-reinterpret family."""

    def _assert_builtin(self, fn):
        self.assertTrue(getattr(fn, BUILTIN_ATTR, False))
        self.assertTrue(callable(getattr(fn, CODEGEN_FUNC_ATTR)))

    def test_family_is_builtin_and_exported(self):
        for name in ("f16x2_reinterpret_u32", "reinterpret_f16x2_lo", "reinterpret_f16x2_hi"):
            self.assertTrue(hasattr(arith, name), name)
            self.assertTrue(hasattr(ll, name), f"ll.{name} not exported")
            self._assert_builtin(getattr(arith, name))

    def test_pack_codegen(self):
        codegen = getattr(arith.f16x2_reinterpret_u32, CODEGEN_FUNC_ATTR)
        result = codegen("LO", "HI")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__halves2half2(lo, hi)", result.body)
        self.assertIn("reinterpret_cast<const uint32_t*>", result.body)
        # No numeric conversion anywhere: no float/half cvt intrinsics.
        self.assertNotIn("2half_rn", result.body)
        self.assertNotIn("2float", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertEqual(result.return_val, "lk_f16x2_reinterpret_u32_fn(LO, HI)")

    def test_unpack_codegen(self):
        for lane, fn, cuda_fn in (("lo", arith.reinterpret_f16x2_lo, "__low2half"), ("hi", arith.reinterpret_f16x2_hi,
                                                                                     "__high2half")):
            codegen = getattr(fn, CODEGEN_FUNC_ATTR)
            result = codegen("BITS")
            self.assertIsInstance(result, Builtin)
            self.assertIn(f"{cuda_fn}(v)", result.body)
            self.assertIn("reinterpret_cast<const __half2*>", result.body)
            self.assertNotIn("2half_rn", result.body)
            self.assertNotIn("2float", result.body)
            self.assertIn("<cuda_fp16.h>", result.includes)
            self.assertEqual(result.return_val, f"lk_reinterpret_f16x2_{lane}_fn(BITS)")

    def test_helper_names_stable_across_sites(self):
        # Name-stable bodies are what lets writeln_builtin dedup collapse
        # multiple call sites into one helper definition.
        codegen = getattr(arith.f16x2_reinterpret_u32, CODEGEN_FUNC_ATTR)
        r1 = codegen("A", "B")
        r2 = codegen("C", "D")
        self.assertEqual(r1.body, r2.body)
        self.assertNotEqual(r1.return_val, r2.return_val)

    def test_lane_order_contract(self):
        # The doc contract: lo operand -> low 16 bits of the packed word,
        # matching the in-memory little-endian fp16 pair layout. Guarded by
        # the __halves2half2(lo, hi) argument order and the __low2half/
        # __high2half split on unpack.
        pack = getattr(arith.f16x2_reinterpret_u32, CODEGEN_FUNC_ATTR)("LO", "HI")
        self.assertIn("lk_f16x2_reinterpret_u32_fn(__half lo, __half hi)", pack.body)
        lo = getattr(arith.reinterpret_f16x2_lo, CODEGEN_FUNC_ATTR)("B")
        hi = getattr(arith.reinterpret_f16x2_hi, CODEGEN_FUNC_ATTR)("B")
        self.assertIn("__low2half", lo.body)
        self.assertIn("__high2half", hi.body)


if __name__ == "__main__":
    unittest.main()
