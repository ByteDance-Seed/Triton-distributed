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
"""WGMMA accumulator-dtype support: init/fence/store keyed off the declared
accumulator dtype, not a hardcoded float.

Reproducer gap (ir.acc_dtype.int32): ll.empty(..., scope="local",
dtype=ll.int32) feeding wgmma_init_accum / wgmma_fence_acc /
wgmma_fence_operand_array / store_acc_to_global used to emit float
zero-init ("0.0f" on an int32_t array), "+f" constraints and float*
signatures on an int32 accumulator path -- a CUDA type error for integer
MMA accumulators (pairs with ir.mma_int8_acc32).

Invariants under test:
  * int32 accumulator: zero-init literal is integer 0 (never 0.0f), fence
    helpers use the integer "+r" register constraint with int32_t
    signatures, and store_acc_to_global copies through int32_t* with no
    float bitcast;
  * float accumulator: helper names and bodies stay byte-identical to the
    historical float-only templates (wgmma_fence_acc_fn etc., "+f",
    float*);
  * the dtype key is generic: any integer accumulator dtype (verified with
    uint32) gets the same integer treatment -- no per-task special case;
  * direct codegen calls without the emitter's _arg_dtypes channel fall
    back to float, preserving legacy behaviour for uninstrumented paths.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.type_system import int32, uint32, float32, Tensor
from little_kernel.language.builtin_base import CODEGEN_FUNC_ATTR, CODEGEN_NEEDS_TYPES_ATTR
import little_kernel.backends.nv.sm90.language.intrin.wgmma as wgmma_intrin


class TestInitAccumZeroLiteral(unittest.TestCase):
    """The zero literal keys off the C++ element type, generically."""

    def test_int32_zero_is_integer(self):
        b = wgmma_intrin.codegen_wgmma_init_accum("acc", 64, "int32_t")
        self.assertIn("acc[_i]=0", b.return_val)
        self.assertNotIn("0.0f", b.return_val)

    def test_uint32_zero_is_integer(self):
        b = wgmma_intrin.codegen_wgmma_init_accum("acc", 64, "uint32_t")
        self.assertIn("acc[_i]=0", b.return_val)
        self.assertNotIn("0.0f", b.return_val)

    def test_int64_zero_is_integer(self):
        b = wgmma_intrin.codegen_wgmma_init_accum("acc", 64, "int64_t")
        self.assertIn("acc[_i]=0", b.return_val)
        self.assertNotIn("0.0f", b.return_val)

    def test_float_zero_unchanged(self):
        b = wgmma_intrin.codegen_wgmma_init_accum("acc", 64, "float")
        self.assertIn("acc[_i]=0.0f", b.return_val)

    def test_double_zero_unchanged(self):
        b = wgmma_intrin.codegen_wgmma_init_accum("acc", 64, "double")
        self.assertIn("acc[_i]=0.0", b.return_val)
        self.assertNotIn("0.0f", b.return_val)


class TestFenceDtypeGeneric(unittest.TestCase):
    """Fence helpers pick "+r"/"+f" and the pointer type from _arg_dtypes."""

    def test_needs_types_attr_set(self):
        self.assertTrue(getattr(wgmma_intrin.wgmma_fence_acc, CODEGEN_NEEDS_TYPES_ATTR, False))
        self.assertTrue(getattr(wgmma_intrin.wgmma_fence_operand_array, CODEGEN_NEEDS_TYPES_ATTR, False))
        self.assertTrue(getattr(wgmma_intrin.wgmma_fence_operand, CODEGEN_NEEDS_TYPES_ATTR, False))
        self.assertTrue(getattr(wgmma_intrin.store_acc_to_global, CODEGEN_NEEDS_TYPES_ATTR, False))

    def test_fence_acc_int32(self):
        codegen = getattr(wgmma_intrin.wgmma_fence_acc, CODEGEN_FUNC_ATTR)
        b = codegen("acc", 64, _arg_dtypes=[Tensor[int32]])
        self.assertIn("int32_t* a", b.body)
        self.assertIn('"+r"(a[i])', b.body)
        self.assertNotIn('"+f"', b.body)
        self.assertIn("wgmma_fence_acc_int32_t_fn(acc, 64)", b.return_val)

    def test_fence_acc_unwraps_tensor_dtype(self):
        # The emitter hands the declared expression type (tensor[s32] for a
        # local accumulator); it must be unwrapped to the element type.
        codegen = getattr(wgmma_intrin.wgmma_fence_acc, CODEGEN_FUNC_ATTR)
        b = codegen("acc", 64, _arg_dtypes=[Tensor[int32]])
        self.assertIn("int32_t*", b.body)

    def test_fence_operand_array_int32(self):
        codegen = getattr(wgmma_intrin.wgmma_fence_operand_array, CODEGEN_FUNC_ATTR)
        b = codegen("acc", 64, _arg_dtypes=[Tensor[int32]])
        self.assertIn("int32_t* a", b.body)
        self.assertIn('"+r"(a[i])', b.body)
        self.assertIn("wgmma_fence_operand_array_int32_t_fn(acc, 64)", b.return_val)

    def test_fence_operand_int32(self):
        codegen = getattr(wgmma_intrin.wgmma_fence_operand, CODEGEN_FUNC_ATTR)
        b = codegen("acc[0]", _arg_dtypes=[int32])
        self.assertIn("int32_t& r", b.body)
        self.assertIn('"+r"(r)', b.body)

    def test_fence_acc_uint32_generic_int_path(self):
        # Any integer accumulator dtype gets the integer treatment, not a
        # task-specific int32 special case.
        codegen = getattr(wgmma_intrin.wgmma_fence_acc, CODEGEN_FUNC_ATTR)
        b = codegen("acc", 64, _arg_dtypes=[Tensor[uint32]])
        self.assertIn("uint32_t* a", b.body)
        self.assertIn('"+r"(a[i])', b.body)

    def test_fence_acc_float_unchanged(self):
        codegen = getattr(wgmma_intrin.wgmma_fence_acc, CODEGEN_FUNC_ATTR)
        b = codegen("acc", 64, _arg_dtypes=[Tensor[float32]])
        self.assertIn("float* a", b.body)
        self.assertIn('"+f"(a[i])', b.body)
        self.assertEqual(b.return_val, "wgmma_fence_acc_fn(acc, 64)")

    def test_fence_acc_no_channel_falls_back_to_float(self):
        # Direct codegen calls (no _arg_dtypes from the emitter) keep the
        # historical float-only template.
        codegen = getattr(wgmma_intrin.wgmma_fence_acc, CODEGEN_FUNC_ATTR)
        b = codegen("acc", 64)
        self.assertIn("float* a", b.body)
        self.assertIn('"+f"(a[i])', b.body)


class TestStoreAccToGlobalDtype(unittest.TestCase):
    """store_acc_* pointer/element types key off the declared dtypes."""

    def test_int32_store_signature(self):
        codegen = getattr(wgmma_intrin.store_acc_to_global, CODEGEN_FUNC_ATTR)
        b = codegen("C", "acc", 0, 0, 64, 64, "tid", 32, 64, _arg_dtypes=[Tensor[int32], Tensor[int32]])
        self.assertIn("int32_t* C, int32_t* acc", b.body)
        self.assertIn("C[(int64_t)gm * N + gn] = acc[r];", b.body)
        self.assertNotIn("float* C", b.body)
        self.assertNotIn("__float", b.body)
        self.assertIn("store_acc_global_64_int32_t_fn(C, acc", b.return_val)

    def test_int32_helper_names_dont_collide_with_float(self):
        codegen = getattr(wgmma_intrin.store_acc_to_global, CODEGEN_FUNC_ATTR)
        b_i = codegen("C", "acc", 0, 0, 64, 64, "tid", 32, 64, _arg_dtypes=[Tensor[int32], Tensor[int32]])
        b_f = codegen("C", "acc", 0, 0, 64, 64, "tid", 32, 64, _arg_dtypes=[Tensor[float32], Tensor[float32]])
        self.assertNotEqual(b_i.body, b_f.body)
        # Float keeps the historical unsuffixed helper names.
        self.assertIn("store_acc_global_64_fn(", b_f.return_val)
        self.assertIn("get_wgmma_coord_64_", b_f.body)
        self.assertNotIn("get_wgmma_coord_64_int32_t_", b_f.body)

    def test_float_store_unchanged(self):
        codegen = getattr(wgmma_intrin.store_acc_to_global, CODEGEN_FUNC_ATTR)
        b = codegen("C", "acc", 0, 0, 64, 64, "tid", 32, 64, _arg_dtypes=[Tensor[float32], Tensor[float32]])
        self.assertIn("float* C, float* acc", b.body)
        self.assertEqual(b.return_val, "store_acc_global_64_fn(C, acc, 0, 0, 64, 64, tid)")

    def test_no_channel_falls_back_to_float(self):
        codegen = getattr(wgmma_intrin.store_acc_to_global, CODEGEN_FUNC_ATTR)
        b = codegen("C", "acc", 0, 0, 64, 64, "tid", 32, 64)
        self.assertIn("float* C, float* acc", b.body)


class TestInt32AccumulatorRoundTripCodegen(unittest.TestCase):
    """End-to-end CUDA text: an int32 accumulator round-trips through
    init -> fence -> store with no float bitcast anywhere on the acc path."""

    def test_int32_round_trip(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def acc_i32(C: ll.Tensor[ll.int32]) -> ll.void:
            acc = ll.empty((64, ), scope="local", dtype=ll.int32)
            ll.wgmma_init_accum(acc, 64, ll.int32)
            ll.wgmma_fence_acc(acc, 64)
            ll.wgmma_fence_operand_array(acc, 64)
            tid: ll.int32 = ll.threadIdx_x() % 128
            ll.store_acc_to_global(C, acc, 0, 0, 64, 64, tid, 32, 64)

        code = acc_i32.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        # init: integer zero literal on an int32_t array
        self.assertIn("int32_t acc[64];", code)
        self.assertIn("acc[_i]=0;", code)
        self.assertNotIn("acc[_i]=0.0f;", code)
        # fences: int32_t signatures with the integer register constraint
        self.assertIn("wgmma_fence_acc_int32_t_fn(int32_t* a, int n)", code)
        self.assertIn('"+r"(a[i])', code)
        self.assertIn("wgmma_fence_acc_int32_t_fn(acc, 64);", code)
        self.assertIn("wgmma_fence_operand_array_int32_t_fn(int32_t* a, int n)", code)
        self.assertIn("wgmma_fence_operand_array_int32_t_fn(acc, 64);", code)
        # store: stores bitwise through int32_t pointers, no float path
        self.assertIn("store_acc_global_64_int32_t_fn(\n    int32_t* C, int32_t* acc", code)
        self.assertIn("store_acc_global_64_int32_t_fn(C, acc, 0, 0, 64, 64, tid);", code)
        self.assertNotIn("store_acc_global_64_fn(", code)

    def test_float_round_trip_unchanged(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def acc_f32(C: ll.Tensor[ll.float32]) -> ll.void:
            acc = ll.empty((32, ), scope="local", dtype=ll.float32)
            ll.wgmma_init_accum(acc, 32, ll.float32)
            ll.wgmma_fence_acc(acc, 32)
            ll.wgmma_fence_operand_array(acc, 32)
            tid: ll.int32 = ll.threadIdx_x() % 128
            ll.store_acc_to_global(C, acc, 0, 0, 64, 64, tid, 32, 64)

        code = acc_f32.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        self.assertIn("acc[_i]=0.0f;", code)
        self.assertIn("wgmma_fence_acc_fn(float* a, int n)", code)
        self.assertIn('"+f"(a[i])', code)
        self.assertIn("store_acc_global_64_fn(\n    float* C, float* acc", code)
        self.assertNotIn("int32_t_fn", code)


if __name__ == "__main__":
    unittest.main()
