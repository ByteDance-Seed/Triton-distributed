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
"""Regression coverage for the typed kernel launch ABI.

``python/little_kernel/runtime/abi.py``'s ``signature_from_annotations`` and
``pack_arguments`` carry the packing rules every backend launch relies on --
integer range checks, raw 16-bit half/bfloat16 bit patterns, mapping-vs-
sequence argument validation, and modifier-aware ``grid_constant`` metadata.
These contracts previously had no test importing the module, so a change to
packing or modifier unwrap would not have been caught by any included suite.
All cases are CPU-only: no device or driver is required.
"""

import ctypes
import struct
import unittest

import little_kernel.language as ll
from little_kernel.runtime.abi import (
    POINTER,
    SCALAR,
    STRUCTURE,
    KernelParameter,
    KernelSignature,
    bfloat16_bits,
    half_bits,
    pack_arguments,
    signature_from_annotations,
)


def _make_signed_struct():

    class Pair(ctypes.Structure):
        _fields_ = [("x", ctypes.c_int32), ("y", ctypes.c_float)]

    return Pair


class SignatureFromAnnotationsTest(unittest.TestCase):
    """Annotation-derived signatures classify scalar/pointer/structure kinds."""

    def test_scalar_pointer_struct_kinds_and_storage(self):
        pair_t = ll.Struct[(("x", ll.int32), ("y", ll.float32))]

        def kernel(a: ll.int32, buf: ll.Tensor[ll.float32], s: pair_t):
            pass

        sig = signature_from_annotations(kernel)
        self.assertIsInstance(sig, KernelSignature)
        self.assertEqual(sig.names, ("a", "buf", "s"))

        a, buf, s = sig.parameters
        self.assertEqual(a.kind, SCALAR)
        self.assertIs(a.c_type, ctypes.c_int32)
        self.assertEqual(buf.kind, POINTER)
        self.assertIs(buf.c_type, ctypes.c_void_p)
        self.assertEqual(s.kind, STRUCTURE)
        self.assertTrue(issubclass(s.c_type, ctypes.Structure))
        # (int32, float32) packs as a plain 8-byte C struct.
        self.assertEqual(s.size, 8)

    def test_unannotated_parameter_is_rejected(self):

        def kernel(a):
            pass

        with self.assertRaisesRegex(TypeError, "no type annotation"):
            signature_from_annotations(kernel)

    def test_struct_without_modifier_is_not_grid_constant(self):
        # Plain by-value struct: grid_constant must reflect the annotation,
        # not an unconditional default (modifier-unwrap regression).
        one_field_t = ll.Struct[(("x", ll.int32), )]

        def kernel(s: one_field_t):
            pass

        param = signature_from_annotations(kernel).parameter("s")
        self.assertEqual(param.kind, STRUCTURE)
        self.assertFalse(param.grid_constant)

    def test_grid_constant_modifier_sets_flag(self):
        one_field_t = ll.Struct[(("x", ll.int32), )]

        def kernel(s: ll.grid_constant[one_field_t]):
            pass

        param = signature_from_annotations(kernel).parameter("s")
        self.assertEqual(param.kind, STRUCTURE)
        self.assertTrue(param.grid_constant)


class PackArgumentsTest(unittest.TestCase):
    """pack_arguments validates values against the declared signature."""

    def _sig(self):
        Pair = _make_signed_struct()

        def kernel(a: ll.int32, buf: ll.Tensor[ll.float32], s: Pair):
            pass

        sig = signature_from_annotations(kernel, context={"Pair": Pair})
        return sig, Pair

    def test_positional_and_mapping_packing_round_trip(self):
        sig, Pair = self._sig()
        value = Pair(7, 1.5)
        for arguments in ([3, 0x1000, value], {"a": 3, "buf": 0x1000, "s": value}):
            with self.subTest(arguments=type(arguments).__name__):
                packed = pack_arguments(sig, arguments)
                self.assertEqual(len(packed.storage), 3)
                self.assertEqual(packed.storage[0].value, 3)
                self.assertEqual(packed.storage[1].value, 0x1000)
                self.assertEqual((packed.storage[2].x, packed.storage[2].y), (7, 1.5))
                # Pointers address the owned storage, suitable for void**.
                for i, obj in enumerate(packed.storage):
                    self.assertEqual(packed.pointers[i], ctypes.addressof(obj))

    def test_int32_rejects_out_of_range_value(self):
        sig, _ = self._sig()
        value = _make_signed_struct()(0, 0.0)
        with self.assertRaises(OverflowError):
            pack_arguments(sig, [2**40, 0x1000, value])
        with self.assertRaises(OverflowError):
            pack_arguments(sig, [-(1 << 31) - 1, 0x1000, value])
        # Boundary values at both ends of int32 remain accepted.
        for edge in (1 << 31) - 1, -(1 << 31):
            packed = pack_arguments(sig, [edge, 0x1000, value])
            self.assertEqual(packed.storage[0].value, edge)

    def test_mapping_with_unknown_name_raises_type_error(self):
        sig, _ = self._sig()
        value = _make_signed_struct()(0, 0.0)
        with self.assertRaisesRegex(TypeError, "Unknown kernel arguments"):
            pack_arguments(sig, {"a": 1, "buf": 0, "s": value, "bogus": 5})

    def test_mapping_missing_name_raises_type_error(self):
        sig, _ = self._sig()
        value = _make_signed_struct()(0, 0.0)
        with self.assertRaisesRegex(TypeError, "Missing kernel arguments"):
            pack_arguments(sig, {"a": 1, "s": value})

    def test_wrong_argument_count_raises_type_error(self):
        sig, _ = self._sig()
        value = _make_signed_struct()(0, 0.0)
        with self.assertRaisesRegex(TypeError, "Expected 3 arguments"):
            pack_arguments(sig, [1, 0x1000])
        with self.assertRaisesRegex(TypeError, "Expected 3 arguments"):
            pack_arguments(sig, [1, 0x1000, value, 9])

    def test_float_where_int_expected_raises_type_error(self):
        sig, _ = self._sig()
        value = _make_signed_struct()(0, 0.0)
        with self.assertRaisesRegex(TypeError, "got float"):
            pack_arguments(sig, [1.5, 0x1000, value])

    def test_scalar_int_rejects_non_numeric_value(self):
        sig, _ = self._sig()
        value = _make_signed_struct()(0, 0.0)
        with self.assertRaises(TypeError):
            pack_arguments(sig, ["three", 0x1000, value])

    def test_structure_with_wrong_layout_is_rejected(self):
        sig, _ = self._sig()

        class Other(ctypes.Structure):
            _fields_ = [("x", ctypes.c_int32)]

        with self.assertRaisesRegex(TypeError, "expects the declared"):
            pack_arguments(sig, [1, 0x1000, Other(3)])


class HalfBfloat16BitsTest(unittest.TestCase):
    """CUDA __half/__nv_bfloat16 parameters carry exact raw 16-bit patterns."""

    def _sig(self):

        def kernel(h: ll.float16, b: ll.bfloat16):
            pass

        return signature_from_annotations(kernel)

    def test_float_inputs_encode_to_exact_bits(self):
        packed = pack_arguments(self._sig(), [1.5, 2.0])
        self.assertEqual(packed.storage[0].value, struct.unpack("<H", struct.pack("<e", 1.5))[0])
        self.assertEqual(packed.storage[1].value, 0x4000)  # 2.0f in BF16.
        # Two bytes of storage each: never widened to float32.
        self.assertEqual(tuple(p.size for p in self._sig()), (2, 2))

    def test_raw_bit_patterns_are_round_tripped(self):
        packed = pack_arguments(self._sig(), [0x3C00, 0x3F80])
        self.assertEqual(packed.storage[0].value, 0x3C00)
        self.assertEqual(packed.storage[1].value, 0x3F80)
        self.assertEqual(packed.argument_bytes(), (struct.pack("<H", 0x3C00), struct.pack("<H", 0x3F80)))

    def test_raw_bits_outside_16_bits_raise_overflow(self):
        with self.assertRaises(OverflowError):
            pack_arguments(self._sig(), [0x1_0000, 0])
        with self.assertRaises(OverflowError):
            pack_arguments(self._sig(), [0, -1])

    def test_bfloat16_helpers_match_round_to_nearest_even(self):
        self.assertEqual(bfloat16_bits(1.0), 0x3F80)
        self.assertEqual(half_bits(1.0), 0x3C00)
        # NaN keeps its quiet bit through the narrowing.
        nan_bits = bfloat16_bits(float("nan"))
        self.assertEqual(nan_bits & 0x0040, 0x0040)


if __name__ == "__main__":
    unittest.main()
