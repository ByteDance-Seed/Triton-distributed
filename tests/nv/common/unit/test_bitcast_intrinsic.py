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
"""End-to-end CUDA emission regression for the generic ``ll.bitcast`` intrinsic.

The generic contract (width validation in ``return_type`` and the
``__builtin_bit_cast`` hook shape) is covered CPU-side in
``tests/common/test_bitcast.py``.  These tests pin the one part only a real
emitter run can regress: a full ``k.compile`` drive through the CUDA frontend
emits a value-returning ``__builtin_bit_cast(cpp_dst, (expr))`` with the
source operand parenthesized, and a width-mismatched call is rejected during
type inference before any C++ reaches disk.  No GPU is required: ``compile``
stops at source text.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes.utils.type_inference.type_inference_core import (
    TypeInferenceError, )


class TestBitcastCudaEmission(unittest.TestCase):
    """Full-pipeline CUDA codegen for ``ll.bitcast``."""

    def test_scalar_bitcast_emits_builtin_bit_cast(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.uint32], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            if tid < n:
                y[tid] = ll.bitcast(x[tid] + 1, ll.float32)

        code = k.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("__builtin_bit_cast(float, (x[tid] + 1))", code)

    def test_power_of_two_scale_idiom(self):
        # The docstring E8M0 idiom: a biased-exponent byte shifted into the
        # FP32 exponent field, then bitcast — exactly ldexp(1, b-127).
        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(b_exp: ll.Tensor[ll.uint32], out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            if tid < n:
                bits: ll.uint32 = b_exp[tid] << 23
                out[tid] = ll.bitcast(bits, ll.float32)

        code = k.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("bits = b_exp[tid] << 23;", code)
        self.assertIn("out[tid] = __builtin_bit_cast(float, (bits));", code)

    def test_float_to_uint_bitcast_preserves_width(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.uint32], n: ll.int32) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            if tid < n:
                y[tid] = ll.bitcast(x[tid], ll.uint32)

        code = k.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("y[tid] = __builtin_bit_cast(uint32_t, (x[tid]));", code)

    def test_mismatched_width_rejected_at_type_inference(self):
        # uint32 -> float16 must fail during lowering (type inference runs
        # return_type), never reaching C++ emission with an ill-formed cast.
        @lk.ll_kernel(backend="cuda", is_entry=True)
        def k(x: ll.Tensor[ll.uint32], y: ll.Tensor[ll.float16], n: ll.int32) -> ll.void:
            tid: ll.int32 = ll.threadIdx_x()
            if tid < n:
                y[tid] = ll.bitcast(x[tid], ll.float16)

        with self.assertRaises((TypeError, TypeInferenceError)) as ctx:
            k.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("equal bit width", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
