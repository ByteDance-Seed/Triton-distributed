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
"""Unit tests for the scalar fp8 -> float32 decode intrinsics
(e4m3_to_f32 / e5m2_to_f32).

Legality invariants under test:
  * both intrinsics are @builtin with eval_return_type=float32 and are
    exported on the language surface (ll.*) for the nv backends that
    register them (sm90, sm100);
  * codegen lowers to the header-only widening chain
    __half2float(__nv_cvt_fp8_to_halfraw((code & 0xFFu), __NV_E4M3/__NV_E5M2))
    inside a __device__ __forceinline__ helper — no PTX inline asm, so the
    intrinsic is legal wherever <cuda_fp8.h> ships;
  * only the low 8 bits of the argument are consumed (explicit 0xFFu mask),
    so a byte extracted from a wider word at ANY alignment is decoded
    identically to a fresh per-element u8 load;
  * <cuda_fp8.h> and <cuda_fp16.h> are declared as on-demand includes;
  * the emitted SASS-relevant constant distinguishes the two flavours
    (__NV_E4M3 vs __NV_E5M2) — a copy/paste mix-up across the twins is
    caught without a GPU.

No GPU is required: these checks inspect the Builtin/codegen metadata on
CPU, in the same style as the fp16 pack/unpack intrinsic tests.
"""

import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.arith as arith_intrin
from little_kernel.core.type_system import float32
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CODEGEN_FUNC_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    Builtin,
)


class TestE4m3ToF32Intrinsic(unittest.TestCase):
    """Scalar e4m3fn byte code -> float32 exact widening decode."""

    def test_is_builtin_with_f32_return(self):
        self.assertTrue(getattr(arith_intrin.e4m3_to_f32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.e4m3_to_f32, BUILTIN_ATTR, False))
        ret = getattr(arith_intrin.e4m3_to_f32, EVAL_RETURN_TYPE_ATTR)()
        self.assertIs(ret, float32)

    def test_codegen_widening_chain(self):
        codegen_func = getattr(arith_intrin.e4m3_to_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("code")
        self.assertIsInstance(result, Builtin)
        # Header-only widening chain: cvt to half raw, then exact f16->f32.
        self.assertIn("__nv_cvt_fp8_to_halfraw", result.body)
        self.assertIn("__half2float", result.body)
        self.assertIn("__NV_E4M3", result.body)
        # Only the low 8 bits are consumed.
        self.assertIn("code & 0xFFu", result.body)
        # No PTX inline asm: legal wherever cuda_fp8.h ships.
        self.assertNotIn("asm(", result.body)
        self.assertIn("__device__ __forceinline__", result.body)

    def test_codegen_declares_fp8_headers(self):
        codegen_func = getattr(arith_intrin.e4m3_to_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("code")
        self.assertIn("<cuda_fp8.h>", result.includes)
        self.assertIn("<cuda_fp16.h>", result.includes)

    def test_codegen_call_site(self):
        codegen_func = getattr(arith_intrin.e4m3_to_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("byte_reg")
        self.assertEqual(result.return_val, "lk_e4m3_to_f32_fn(byte_reg)")
        # Helper function name matches the call site prefix.
        self.assertIn("lk_e4m3_to_f32_fn(uint32_t code)", result.body)


class TestE5m2ToF32Intrinsic(unittest.TestCase):
    """Scalar e5m2 byte code -> float32 exact widening decode."""

    def test_is_builtin_with_f32_return(self):
        self.assertTrue(getattr(arith_intrin.e5m2_to_f32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.e5m2_to_f32, BUILTIN_ATTR, False))
        ret = getattr(arith_intrin.e5m2_to_f32, EVAL_RETURN_TYPE_ATTR)()
        self.assertIs(ret, float32)

    def test_codegen_widening_chain(self):
        codegen_func = getattr(arith_intrin.e5m2_to_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("code")
        self.assertIsInstance(result, Builtin)
        self.assertIn("__nv_cvt_fp8_to_halfraw", result.body)
        self.assertIn("__half2float", result.body)
        self.assertIn("__NV_E5M2", result.body)
        self.assertIn("code & 0xFFu", result.body)
        self.assertNotIn("asm(", result.body)

    def test_codegen_declares_fp8_headers(self):
        codegen_func = getattr(arith_intrin.e5m2_to_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("code")
        self.assertIn("<cuda_fp8.h>", result.includes)
        self.assertIn("<cuda_fp16.h>", result.includes)

    def test_codegen_call_site(self):
        codegen_func = getattr(arith_intrin.e5m2_to_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("byte_reg")
        self.assertEqual(result.return_val, "lk_e5m2_to_f32_fn(byte_reg)")
        self.assertIn("lk_e5m2_to_f32_fn(uint32_t code)", result.body)


class TestFp8ScalarTwinsAreNotCrossWired(unittest.TestCase):
    """The e4m3/e5m2 twins differ only in the flavour constant and helper
    name; a copy/paste regression must be caught."""

    def test_flavour_constants_match_intrinsic(self):
        e4_body = getattr(arith_intrin.e4m3_to_f32, CODEGEN_FUNC_ATTR)("c").body
        e5_body = getattr(arith_intrin.e5m2_to_f32, CODEGEN_FUNC_ATTR)("c").body
        self.assertNotIn("__NV_E5M2", e4_body)
        self.assertNotIn("__NV_E4M3", e5_body)
        self.assertNotIn("lk_e5m2_to_f32_fn", e4_body)
        self.assertNotIn("lk_e4m3_to_f32_fn", e5_body)

    def test_bodies_parallel_after_normalisation(self):
        # Semantics-preserving twin claim: after renaming the flavour token
        # and helper identifier, the two helpers are textually identical.
        e4_body = getattr(arith_intrin.e4m3_to_f32, CODEGEN_FUNC_ATTR)("c").body
        e5_body = getattr(arith_intrin.e5m2_to_f32, CODEGEN_FUNC_ATTR)("c").body
        normalised_e4 = (e4_body.replace("e4m3_to_f32", "FP8_TO_F32").replace("__NV_E4M3", "__NV_FLAVOUR"))
        normalised_e5 = (e5_body.replace("e5m2_to_f32", "FP8_TO_F32").replace("__NV_E5M2", "__NV_FLAVOUR"))
        self.assertEqual(normalised_e4, normalised_e5)


class TestBackendRegistration(unittest.TestCase):
    """Registered on the nv backends that ship cuda_fp8.h (sm90, sm100)."""

    def _backend_symbols(self, json_name):
        import json
        from pathlib import Path

        path = (Path(arith_intrin.__file__).resolve().parents[3] / json_name / "backend.json")
        with path.open() as f:
            data = json.load(f)
        return data["language_exports"]

    def test_sm90_registers_both(self):
        intrinsics = self._backend_symbols("sm90")
        self.assertEqual(intrinsics["e4m3_to_f32"],
                         "little_kernel.backends.nv.common.language.intrin.arith:e4m3_to_f32")
        self.assertEqual(intrinsics["e5m2_to_f32"],
                         "little_kernel.backends.nv.common.language.intrin.arith:e5m2_to_f32")

    def test_sm100_registers_both(self):
        intrinsics = self._backend_symbols("sm100")
        self.assertEqual(intrinsics["e4m3_to_f32"],
                         "little_kernel.backends.nv.common.language.intrin.arith:e4m3_to_f32")
        self.assertEqual(intrinsics["e5m2_to_f32"],
                         "little_kernel.backends.nv.common.language.intrin.arith:e5m2_to_f32")


if __name__ == "__main__":
    unittest.main()
