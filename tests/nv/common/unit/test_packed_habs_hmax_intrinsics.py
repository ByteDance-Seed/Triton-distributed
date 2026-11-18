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
"""CPU codegen coverage for packed BF16/FP16 abs and max intrinsics."""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm


class TestPackedHabsHmaxIntrinsics(unittest.TestCase):

    def test_bf16x2_intrinsics_are_typed_uint32_builtins(self):
        for intrinsic in (cuda_asm.habs2_bf16x2, cuda_asm.hmax2_bf16x2):
            self.assertTrue(getattr(intrinsic, BUILTIN_ATTR, False))
            return_type = getattr(intrinsic, EVAL_RETURN_TYPE_ATTR)
            self.assertIs(return_type(), ll.uint32)

    def test_f16x2_intrinsics_are_typed_uint32_builtins(self):
        for intrinsic in (cuda_asm.habs2_f16x2, cuda_asm.hmax2_f16x2):
            self.assertTrue(getattr(intrinsic, BUILTIN_ATTR, False))
            return_type = getattr(intrinsic, EVAL_RETURN_TYPE_ATTR)
            self.assertIs(return_type(), ll.uint32)

    def test_bf16x2_codegen_uses_hardware_intrinsics(self):
        abs_result = getattr(cuda_asm.habs2_bf16x2, CODEGEN_FUNC_ATTR)("bits")
        self.assertIsInstance(abs_result, Builtin)
        self.assertIn("const __nv_bfloat162 value", abs_result.body)
        self.assertIn("__habs2(value)", abs_result.body)
        self.assertIn("<cuda_bf16.h>", abs_result.includes)
        self.assertEqual(abs_result.return_val, "habs2_bf16x2_fn(bits)")

        max_result = getattr(cuda_asm.hmax2_bf16x2, CODEGEN_FUNC_ATTR)("lhs", "rhs")
        self.assertIsInstance(max_result, Builtin)
        self.assertIn("const __nv_bfloat162 a", max_result.body)
        self.assertIn("const __nv_bfloat162 b", max_result.body)
        self.assertIn("__hmax2(a, b)", max_result.body)
        self.assertIn("<cuda_bf16.h>", max_result.includes)
        self.assertEqual(max_result.return_val, "hmax2_bf16x2_fn(lhs, rhs)")

    def test_f16x2_codegen_uses_hardware_intrinsics(self):
        abs_result = getattr(cuda_asm.habs2_f16x2, CODEGEN_FUNC_ATTR)("bits")
        self.assertIsInstance(abs_result, Builtin)
        self.assertIn("const __half2 value", abs_result.body)
        self.assertIn("__habs2(value)", abs_result.body)
        self.assertIn("<cuda_fp16.h>", abs_result.includes)
        self.assertEqual(abs_result.return_val, "habs2_f16x2_fn(bits)")

        max_result = getattr(cuda_asm.hmax2_f16x2, CODEGEN_FUNC_ATTR)("lhs", "rhs")
        self.assertIsInstance(max_result, Builtin)
        self.assertIn("const __half2 a", max_result.body)
        self.assertIn("const __half2 b", max_result.body)
        self.assertIn("__hmax2(a, b)", max_result.body)
        self.assertIn("<cuda_fp16.h>", max_result.includes)
        self.assertEqual(max_result.return_val, "hmax2_f16x2_fn(lhs, rhs)")

    def test_language_exports(self):
        self.assertIs(ll.habs2_bf16x2, cuda_asm.habs2_bf16x2)
        self.assertIs(ll.hmax2_bf16x2, cuda_asm.hmax2_bf16x2)
        self.assertIs(ll.habs2_f16x2, cuda_asm.habs2_f16x2)
        self.assertIs(ll.hmax2_f16x2, cuda_asm.hmax2_f16x2)

    def test_full_kernel_codegen(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def packed_amax(x: ll.Tensor[ll.uint32], y: ll.Tensor[ll.uint32], out: ll.Tensor[ll.uint32]) -> ll.void:
            i: ll.int32 = ll.threadIdx_x()
            a: ll.uint32 = x[i]
            b: ll.uint32 = y[i]
            magnitude: ll.uint32 = ll.habs2_bf16x2(a)
            out[i] = ll.hmax2_bf16x2(magnitude, b)

        code = packed_amax.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("habs2_bf16x2_fn(a)", code)
        self.assertIn("hmax2_bf16x2_fn(magnitude, b)", code)
        self.assertIn("__habs2(value)", code)
        self.assertIn("__hmax2(a, b)", code)
        self.assertIn("<cuda_bf16.h>", code)


if __name__ == "__main__":
    unittest.main()
