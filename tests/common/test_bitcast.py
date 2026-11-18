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
"""CPU-only regression tests for the target-independent ``ll.bitcast`` builtin.

``bitcast`` reinterprets the bits of a scalar value as another dtype of equal
bit width (``__builtin_bit_cast`` at emission).  These tests exercise the
builtin protocol directly: the ``__eval_return_type__`` hook that the type
inferencer runs, and the ``__codegen_func__`` hook that the emitter runs.
No backend or GPU is required.
"""

import unittest

import little_kernel.language as ll
from little_kernel.core.type_system import (
    float32,
    float16,
    float64,
    int32,
    uint32,
    uint64,
    bfloat16,
    fp8_e4m3,
    uint8,
)
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CODEGEN_FUNC_ATTR,
    EVAL_RETURN_TYPE_ATTR,
)
from little_kernel.language.intrin.dtype import bitcast


class TestBitcastExport(unittest.TestCase):
    """The builtin must be reachable through the public ``ll`` namespace."""

    def test_exported_as_language_symbol(self):
        self.assertTrue(hasattr(ll, "bitcast"))
        self.assertIn("bitcast", ll.__all__)
        self.assertIs(getattr(ll, "bitcast"), bitcast)

    def test_is_builtin_with_inferencer_and_emitter_hooks(self):
        self.assertTrue(getattr(bitcast, BUILTIN_ATTR, False))
        self.assertTrue(callable(getattr(bitcast, EVAL_RETURN_TYPE_ATTR)))
        self.assertTrue(callable(getattr(bitcast, CODEGEN_FUNC_ATTR)))


class TestBitcastReturnType(unittest.TestCase):
    """``return_type`` is what the type inferencer calls; it validates width."""

    def setUp(self):
        self.eval_return_type = getattr(bitcast, EVAL_RETURN_TYPE_ATTR)

    def test_equal_width_returns_destination_dtype(self):
        self.assertIs(self.eval_return_type(uint32, float32), float32)
        self.assertIs(self.eval_return_type(float32, uint32), uint32)
        self.assertIs(self.eval_return_type(int32, float32), float32)
        self.assertIs(self.eval_return_type(uint64, float64), float64)
        self.assertIs(self.eval_return_type(float16, bfloat16), bfloat16)
        # Quantized-format boundary: an 8-bit byte and an FP8 format share the
        # same width, so the cast is accepted.
        self.assertEqual(fp8_e4m3.bits, 8)
        self.assertIs(self.eval_return_type(uint8, fp8_e4m3), fp8_e4m3)

    def test_dtype_keyword_form(self):
        self.assertIs(self.eval_return_type(uint32, dtype=float32), float32)

    def test_mismatched_width_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.eval_return_type(uint32, float16)
        with self.assertRaises(TypeError):
            self.eval_return_type(float64, uint32)
        with self.assertRaises(TypeError):
            self.eval_return_type(uint32, uint64)

    def test_error_message_names_types_and_widths(self):
        with self.assertRaises(TypeError) as ctx:
            self.eval_return_type(uint32, float16)
        msg = str(ctx.exception)
        self.assertIn("uint32", msg)
        self.assertIn("float16", msg)
        self.assertIn("32", msg)
        self.assertIn("16", msg)

    def test_unknown_destination_dtype_falls_back_to_source(self):
        # When the inferred dtype argument is not an LLType (e.g. still
        # unresolved), return the source type rather than crashing — matching
        # the val_cast/to family contract.
        self.assertIs(self.eval_return_type(uint32, None), uint32)
        self.assertIs(self.eval_return_type(uint32, "float"), uint32)

    def test_invalid_dtype_argument_type(self):
        # A non-type second argument (an int *value* never flows here, but a
        # stray LLType-less object must not be treated as a dtype).
        self.assertIs(self.eval_return_type(uint32, 12345), uint32)


class TestBitcastCodegen(unittest.TestCase):
    """Emission text uses ``__builtin_bit_cast`` with the resolved C++ type."""

    def setUp(self):
        self.codegen = getattr(bitcast, CODEGEN_FUNC_ATTR)

    def test_codegen_with_cpp_type_name_string(self):
        # The emitter visits arguments to C++ text before codegen runs, so the
        # common in-pipeline form is a destination type-name string.
        builtin = self.codegen("x", "float")
        self.assertEqual(builtin.return_val, "__builtin_bit_cast(float, (x))")
        self.assertEqual(builtin.body, "")
        self.assertEqual(builtin.includes, [])

    def test_codegen_with_lltype_destination(self):
        # Under an emission scope, the active target policy maps the LLType to
        # its C++ spelling (this fake policy doubles as the scope contract).
        from little_kernel.codegen.policy import emission_scope

        class _FakeEmitter:

            def _lltype_to_cpp(self, ll_type):
                return {float32: "float", uint32: "unsigned int"}[ll_type]

        with emission_scope(_FakeEmitter()):
            builtin = self.codegen("x", float32)
            builtin_u32 = self.codegen("y", uint32)
        self.assertEqual(builtin.return_val, "__builtin_bit_cast(float, (x))")
        self.assertEqual(builtin_u32.return_val, "__builtin_bit_cast(unsigned int, (y))")

    def test_source_expression_is_parenthesized(self):
        # Shifted exponent byte (the E8M0 idiom in the docstring) must bind
        # as a single operand regardless of operator precedence.
        builtin = self.codegen("(b << 23)", "unsigned int")
        self.assertEqual(builtin.return_val, "__builtin_bit_cast(unsigned int, ((b << 23)))")


if __name__ == "__main__":
    unittest.main()
