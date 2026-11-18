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
"""CPU regression: C++ integral promotion for narrow-int arithmetic (no GPU).

Capability ``ir.int8_to_fp16x2_cvt`` / dependent task diagnostic: on a
``Tensor[ll.int8]`` element read, ``x[i] & 255`` (or any scalar arithmetic
mixing an int8/uint8 element with int literals) used to fail type inference
at trace with an invalid operand-pair TypeError, because the narrow element
type was reconciled directly against the int32 literal instead of being
integral-promoted first as C++ assigns the expression type ``int``.

The fix lives in OperatorTypeInferencer: before the usual arithmetic
conversions (``promote_type``), each operand's *result* typing applies
[conv.integral.prom] — integer types of conversion rank below ``int``
(int8/uint8/int16/uint16) become int32; wider or non-integer types are
unchanged, and the special bool/binary types keep their identity so the
existing special->int32 path inside ``promote_type`` still keys off them.
Operand node types are intentionally untouched: codegen keeps the narrow
element reads and the emitted C++ applies the same promotion implicitly at
the expression level, so the trace-side type and the emitted arithmetic
agree.

This file covers:

* the promotion helper contract at unit level (narrow -> int32; wide,
  float and special types unchanged);
* binop/unop result typing with the promotion applied (the reproducer's
  ``int8 & int-literal`` mask; narrow+narrow computing in int; mixed
  int/float still float; wider pairs and pairs of specials unchanged);
* end-to-end CPU codegen of the reported minimal reproducer — an int8
  tensor element feeding a byte mask into ``cvt_e4m3x2_to_f16x2`` /
  ``f16x2_lo_f32`` — which must trace, lower and emit CUDA source without
  error (and must contain the masked byte load).
"""

import ast
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.core.type_system import (
    int8,
    uint8,
    int16,
    int32,
    uint32,
    int64,
    float32,
    bool_,
)
from little_kernel.core.passes.utils.type_inference.type_inference_visitors import (
    OperatorTypeInferencer,
    _integral_promote_operand,
)
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


@lk.ll_kernel(backend='cuda', is_entry=True)
def _int8_mask_cvt_repro(x: ll.Tensor[ll.int8], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    """The reported minimal reproducer: int8 element, byte mask, fp8 cvt."""
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        y[i] = ll.f16x2_lo_f32(ll.cvt_e4m3x2_to_f16x2(x[i] & 255))


class TestIntegralPromotionHelper(unittest.TestCase):

    def test_narrow_signed_and_unsigned_widen_to_int32(self):
        self.assertEqual(_integral_promote_operand(int8), int32)
        self.assertEqual(_integral_promote_operand(uint8), int32)
        self.assertEqual(_integral_promote_operand(int16), int32)

    def test_wide_types_unchanged(self):
        self.assertEqual(_integral_promote_operand(int32), int32)
        self.assertEqual(_integral_promote_operand(uint32), uint32)
        self.assertEqual(_integral_promote_operand(int64), int64)
        self.assertEqual(_integral_promote_operand(float32), float32)

    def test_special_bool_kept_for_promote_type_path(self):
        # bool has a dedicated int32-widening path inside promote_type that
        # keys off `special`; the helper must not rewrite it early.
        self.assertIs(_integral_promote_operand(bool_), bool_)


class TestBinOpPromotion(unittest.TestCase):

    def setUp(self):
        self.inf = OperatorTypeInferencer()
        self.node = ast.parse("a + b", mode="eval").body

    def _bin(self, l, r, op=ast.Add):
        return self.inf.infer_bin_op(l, r, op, self.node)

    def test_int8_and_literal_mask(self):
        # The capability reproducer: int8 element & int literal -> int32.
        self.assertEqual(self._bin(int8, int32, ast.BitAnd), int32)
        self.assertEqual(self._bin(uint8, int32, ast.BitAnd), int32)

    def test_narrow_narrow_arithmetic_computes_in_int(self):
        # C++ [expr.arith.conv]: int8 + int8 has type int, not int8.
        self.assertEqual(self._bin(int8, int8, ast.Add), int32)
        self.assertEqual(self._bin(uint8, uint8, ast.Mult), int32)
        self.assertEqual(self._bin(int16, int8, ast.Sub), int32)

    def test_int8_with_float_still_float(self):
        self.assertEqual(self._bin(int8, float32, ast.Add), float32)
        self.assertEqual(self._bin(float32, uint8, ast.Div), float32)

    def test_wider_int_pairs_unchanged(self):
        self.assertEqual(self._bin(int32, int64, ast.Add), int64)
        self.assertEqual(self._bin(uint32, int32, ast.BitOr), uint32)
        self.assertEqual(self._bin(int32, int32, ast.BitAnd), int32)

    def test_bool_pairs_keep_preexisting_typing(self):
        # Equal special-typed operands return early from promote_type;
        # that contract predates this change and is locked in here.
        self.assertEqual(self._bin(bool_, bool_, ast.Add), bool_)
        self.assertEqual(self._bin(bool_, int32, ast.BitAnd), int32)


class TestUnOpPromotion(unittest.TestCase):

    def setUp(self):
        self.inf = OperatorTypeInferencer()
        self.node = ast.parse("-a", mode="eval").body

    def _un(self, t, op=ast.USub):
        return self.inf.infer_un_op(t, op, self.node)

    def test_neg_narrow_is_int32(self):
        self.assertEqual(self._un(int8), int32)
        self.assertEqual(self._un(uint8), int32)

    def test_neg_wide_unchanged(self):
        self.assertEqual(self._un(int32), int32)
        self.assertEqual(self._un(float32), float32)


class TestInt8MaskCodegen(unittest.TestCase):

    def test_minimal_reproducer_lowers_end_to_end(self):
        # Trace + full CPU codegen of the reported reproducer; must not
        # raise the operand-pair TypeError from the capability diagnostic.
        src = _int8_mask_cvt_repro.compile(PASSES["cuda"], codegen_cuda)
        self.assertIsInstance(src, str)
        # The byte-masked load and both halves of the decode chain survive
        # into the emitted source.
        self.assertIn("& 255", src)
        self.assertIn("x[", src)


if __name__ == "__main__":
    unittest.main()
