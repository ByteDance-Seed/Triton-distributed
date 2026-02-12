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

import unittest
import little_kernel.language as ll
from little_kernel.core.type_system import (IntType, FloatType, StringType, Pointer, Const, GridConstant, TensorType,
                                            TmaDescriptorType, StructType, WgmmaType, LLType)


class TestScalarTypes(unittest.TestCase):
    """Tests for all scalar types (integers, floats, strings).
    
    Covers type creation, attribute validation, equality, hashing, and string representations.
    """

    def test_integer_types(self):
        """Test integer types (including special bool/binary and standard integer widths)."""
        # Test special integer types (1-bit)
        bool_type = ll.bool_
        binary_type = ll.binary

        self.assertIsInstance(bool_type, IntType)
        self.assertEqual(bool_type.bits, 1)
        self.assertEqual(bool_type.signed, False)
        self.assertEqual(bool_type.special, "bool")
        self.assertEqual(str(bool_type), "bool")

        self.assertIsInstance(binary_type, IntType)
        self.assertEqual(binary_type.bits, 1)
        self.assertEqual(binary_type.signed, True)
        self.assertEqual(binary_type.special, "binary")
        self.assertEqual(str(binary_type), "binary")

        # Test signed standard integers
        int4 = ll.int4
        int32 = ll.int32
        int64 = ll.int64

        self.assertEqual(int4.bits, 4)
        self.assertTrue(int4.signed)
        self.assertIsNone(int4.special)
        self.assertEqual(str(int4), "int4")

        # Test unsigned standard integers
        uint8 = ll.uint8
        _ = ll.uint128  # Exists check

        self.assertEqual(uint8.bits, 8)
        self.assertFalse(uint8.signed)
        self.assertEqual(str(uint8), "uint8")

        # Test equality and hashing
        self.assertEqual(int32, IntType(bits=32, signed=True))
        self.assertNotEqual(int32, ll.uint32)
        self.assertEqual(hash(int64), hash(IntType(bits=64, signed=True)))
        self.assertNotEqual(hash(int4), hash(ll.uint4))

    def test_float_types(self):
        """Test floating-point types (all predefined formats with different bit widths)."""
        # Test 4-bit float
        fp4 = ll.fp4_e2m1

        self.assertIsInstance(fp4, FloatType)
        self.assertEqual(fp4.fmt, "fp4_e2m1")
        self.assertEqual(fp4.bits, 4)
        self.assertEqual(str(fp4), "fp4_e2m1")

        # Test 8-bit floats (different formats)
        fp8_e5m2 = ll.fp8_e5m2
        fp8_e4m3 = ll.fp8_e4m3

        self.assertEqual(fp8_e5m2.bits, 8)
        self.assertEqual(fp8_e4m3.bits, 8)
        self.assertNotEqual(fp8_e5m2, fp8_e4m3)  # Same bits but different formats
        self.assertEqual(hash(fp8_e5m2), hash(FloatType("fp8_e5m2")))

        # Test 16-bit floats
        bfloat16 = ll.bfloat16
        float16 = ll.float16

        self.assertEqual(bfloat16.bits, 16)
        self.assertEqual(float16.fmt, "float16")
        self.assertNotEqual(bfloat16, float16)

        # Test 32/64-bit floats
        tfloat32 = ll.tfloat32
        float64 = ll.float64

        self.assertEqual(tfloat32.bits, 32)
        self.assertEqual(float64.fmt, "float64")
        self.assertEqual(str(tfloat32), "tfloat32")

    def test_string_type(self):
        """Test string scalar type (singleton instance)."""
        str_type = ll.str_

        self.assertIsInstance(str_type, StringType)
        self.assertEqual(str_type.kind, "str")
        self.assertEqual(str(str_type), "str")
        self.assertEqual(repr(str_type), "StringType()")

        # Test singleton property: multiple calls return the same instance
        str_type2 = StringType()
        self.assertIs(str_type, str_type2)
        self.assertEqual(hash(str_type), hash(str_type2))


class TestWrapperTypes(unittest.TestCase):
    """Tests for wrapper types (const, grid_constant) and their nesting."""

    def test_pointer_wrapper(self):
        """Test pointer wrapper type."""
        ptr_uint32 = ll.ptr[ll.uint32]

        self.assertIsInstance(ptr_uint32, Pointer)
        self.assertIsInstance(ptr_uint32.inner_type, IntType)
        self.assertEqual(ptr_uint32.inner_type, ll.uint32)
        self.assertEqual(str(ptr_uint32), "uint32_ptr")
        self.assertEqual(repr(ptr_uint32), "uint32_ptr")

    def test_const_wrapper(self):
        """Test const wrapper type for compile-time constants."""
        const_uint32 = ll.const[ll.uint32]

        self.assertIsInstance(const_uint32, Const)
        self.assertIsInstance(const_uint32.inner_type, IntType)
        self.assertEqual(const_uint32.inner_type, ll.uint32)
        self.assertEqual(str(const_uint32), "const[uint32]")
        self.assertEqual(repr(const_uint32), "const[uint32]")

        # Test equality
        const_uint32_2 = ll.const[ll.uint32]
        self.assertEqual(const_uint32, const_uint32_2)
        self.assertNotEqual(const_uint32, ll.const[ll.int32])

    def test_grid_const_wrapper(self):
        """Test grid_constant wrapper type for grid-scoped constants."""
        grid_const_tma = ll.grid_constant[ll.TmaDescriptor]

        self.assertIsInstance(grid_const_tma, GridConstant)
        self.assertIs(grid_const_tma.inner_type, ll.TmaDescriptor)
        self.assertEqual(str(grid_const_tma), "grid_constant[TmaDescriptor]")
        self.assertEqual(repr(grid_const_tma), "grid_constant[TmaDescriptor]")

    def test_nested_wrappers(self):
        """Test nested wrappers (e.g., const[grid_constant[TmaDescriptor]])."""
        nested = ll.const[ll.grid_constant[ll.TmaDescriptor]]

        self.assertIsInstance(nested, Const)
        self.assertIsInstance(nested.inner_type, GridConstant)
        self.assertIs(nested.inner_type.inner_type, ll.TmaDescriptor)
        self.assertEqual(str(nested), "const[grid_constant[TmaDescriptor]]")
        self.assertEqual(repr(nested), "const[grid_constant[TmaDescriptor]]")


class TestCompositeTypes(unittest.TestCase):
    """Tests for composite types (tensor) and special types (TmaDescriptor, LLType)."""

    def test_tensor_type(self):
        """Test tensor type parameterized by scalar element types."""
        tensor_bf16 = ll.Tensor[ll.bfloat16]

        self.assertIsInstance(tensor_bf16, TensorType)
        self.assertEqual(tensor_bf16.element_type, ll.bfloat16)
        self.assertEqual(str(tensor_bf16), "Tensor[bfloat16]")

        # Test inequality for different element types
        tensor_int32 = ll.Tensor[ll.int32]
        self.assertNotEqual(tensor_bf16, tensor_int32)
        self.assertEqual(hash(tensor_int32), hash(TensorType(ll.int32)))

    def test_tma_descriptor(self):
        """Test TmaDescriptor type (singleton for hardware descriptors)."""
        tma = ll.TmaDescriptor

        self.assertIsInstance(tma, TmaDescriptorType)
        self.assertEqual(str(tma), "TmaDescriptor")
        self.assertEqual(repr(tma), "TmaDescriptor")

        # Test singleton property
        tma2 = TmaDescriptorType()
        self.assertIs(tma, tma2)

    def test_wgmma(self):
        """Test Wgmma type (singleton for hardware mma)."""
        wgmma = ll.Wgmma

        self.assertIsInstance(wgmma, WgmmaType)
        self.assertEqual(str(wgmma), "Wgmma")
        self.assertEqual(repr(wgmma), "Wgmma")

    def test_struct(self):
        """Test Struct type parameterized by a tuple of types."""
        struct = ll.Struct[("a", ll.uint32), ("b", ll.bfloat16), ("c", ll.Tensor[ll.int32])]

        self.assertIsInstance(struct, StructType)
        self.assertEqual(struct.type_tuple, (("a", ll.uint32), ("b", ll.bfloat16), ("c", ll.Tensor[ll.int32])))
        self.assertEqual(repr(struct), "Struct[uint32, bfloat16, Tensor[int32]]")

    def test_type(self):
        """Test LLType (metatype representing 'type of types')."""
        type_meta = ll.type_

        self.assertEqual(type_meta, LLType())


class TestFunctionAnnotations(unittest.TestCase):
    """Tests for type usage in function annotations (mimicking user function signatures)."""

    def test_valid_annotations(self):
        """Verify all types can be correctly used as function parameter annotations."""

        # Simulate a user function with type annotations
        def test_func(
            # Scalar types
            a: ll.bool_,
            b: ll.uint32,
            c: ll.bfloat16,
            d: ll.str_,
            # Wrapper types
            e: ll.const[ll.int64],
            f: ll.grid_constant[ll.TmaDescriptor],
            g: ll.const[ll.grid_constant[ll.TmaDescriptor]],
            # Composite types
            h: ll.Tensor[ll.float32],
            # Special types
            i: ll.const[ll.type_],
        ):
            pass

        # Validate annotation types via __annotations__
        annotations = test_func.__annotations__
        self.assertIsInstance(annotations["a"], IntType)
        self.assertIsInstance(annotations["b"], IntType)
        self.assertIsInstance(annotations["c"], FloatType)
        self.assertIsInstance(annotations["d"], StringType)
        self.assertIsInstance(annotations["e"], Const)
        self.assertIsInstance(annotations["f"], GridConstant)
        self.assertIsInstance(annotations["g"].inner_type, GridConstant)
        self.assertIsInstance(annotations["h"], TensorType)
        self.assertIsInstance(annotations["i"].inner_type, LLType)


if __name__ == "__main__":
    unittest.main()
