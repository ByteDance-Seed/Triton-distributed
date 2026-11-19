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
"""
Unit tests for @builtin class decorator with BuiltinOp base class.
"""

import unittest

from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp, BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR
from little_kernel.core.type_system import float32, int32, void


class TestBuiltinClassBasic(unittest.TestCase):
    """Test basic @builtin class decorator functionality."""

    def test_builtin_class_creates_callable(self):
        """Test that @builtin on BuiltinOp creates a callable."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a, b):
                return float32

            @staticmethod
            def codegen(a, b):
                return Builtin(body="", includes=[], return_val=f"({a} + {b})")

        self.assertTrue(callable(TestOp))

    def test_builtin_class_has_builtin_attr(self):
        """Test that @builtin sets BUILTIN_ATTR."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        self.assertTrue(hasattr(TestOp, BUILTIN_ATTR))
        self.assertTrue(getattr(TestOp, BUILTIN_ATTR))

    def test_builtin_class_has_codegen_func(self):
        """Test that @builtin sets CODEGEN_FUNC_ATTR."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        self.assertTrue(hasattr(TestOp, CODEGEN_FUNC_ATTR))

    def test_builtin_class_has_eval_return_type(self):
        """Test that @builtin sets EVAL_RETURN_TYPE_ATTR."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        self.assertTrue(hasattr(TestOp, EVAL_RETURN_TYPE_ATTR))

    def test_builtin_class_preserves_name(self):
        """Test that @builtin preserves the class name."""

        @builtin
        class MyCustomOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        self.assertEqual(MyCustomOp.__name__, "MyCustomOp")


class TestBuiltinClassCodegen(unittest.TestCase):
    """Test codegen functionality of @builtin class decorator."""

    def test_codegen_returns_builtin(self):
        """Test that codegen function returns Builtin object."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a, b):
                return float32

            @staticmethod
            def codegen(a, b):
                return Builtin(body="", includes=[], return_val=f"({a} * {b})")

        codegen_func = getattr(TestOp, CODEGEN_FUNC_ATTR)
        result = codegen_func("x", "y")

        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.return_val, "(x * y)")

    def test_codegen_with_body(self):
        """Test codegen with helper function body."""

        @builtin
        class FusedMulAdd(BuiltinOp):

            @staticmethod
            def return_type(a, b, c):
                return float32

            @staticmethod
            def codegen(a, b, c):
                body = """
__device__ __forceinline__ float fma_impl(float a, float b, float c) {
    return __fmaf_rn(a, b, c);
}
"""
                return Builtin(body=body, includes=[], return_val=f"fma_impl({a}, {b}, {c})")

        codegen_func = getattr(FusedMulAdd, CODEGEN_FUNC_ATTR)
        result = codegen_func("x", "y", "z")

        self.assertIn("fma_impl", result.body)
        self.assertEqual(result.return_val, "fma_impl(x, y, z)")

    def test_codegen_with_includes(self):
        """Test codegen with header includes."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=["<cuda_fp16.h>"], return_val=a)

        codegen_func = getattr(TestOp, CODEGEN_FUNC_ATTR)
        result = codegen_func("x")

        self.assertIn("<cuda_fp16.h>", result.includes)


class TestBuiltinClassReturnType(unittest.TestCase):
    """Test return type inference of @builtin class decorator."""

    def test_return_type_float32(self):
        """Test return type inference for float32."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        eval_return_type = getattr(TestOp, EVAL_RETURN_TYPE_ATTR)
        result = eval_return_type("x")

        self.assertEqual(result, float32)

    def test_return_type_int32(self):
        """Test return type inference for int32."""

        @builtin
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return int32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        eval_return_type = getattr(TestOp, EVAL_RETURN_TYPE_ATTR)
        result = eval_return_type("x")

        self.assertEqual(result, int32)


class TestBuiltinClassWithParentheses(unittest.TestCase):
    """Test @builtin() with parentheses on BuiltinOp class."""

    def test_builtin_with_empty_parens(self):
        """Test @builtin() with empty parentheses."""

        @builtin()
        class TestOp(BuiltinOp):

            @staticmethod
            def return_type(a):
                return float32

            @staticmethod
            def codegen(a):
                return Builtin(body="", includes=[], return_val=a)

        self.assertTrue(callable(TestOp))
        self.assertTrue(hasattr(TestOp, BUILTIN_ATTR))
        self.assertEqual(TestOp.__name__, "TestOp")


class TestBuiltinClassErrors(unittest.TestCase):
    """Test error handling for @builtin class decorator."""

    def test_builtin_empty_parens_on_non_builtinop(self):
        """Test that @builtin() on non-BuiltinOp raises error."""
        with self.assertRaises(ValueError):

            @builtin()
            def not_a_class():
                pass


if __name__ == '__main__':
    unittest.main()
