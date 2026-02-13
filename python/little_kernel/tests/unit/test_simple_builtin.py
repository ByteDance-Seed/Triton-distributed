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
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR


class TestSimpleBuiltinBasic(unittest.TestCase):
    """Basic tests for simple_builtin decorator."""

    def test_simple_builtin_creates_builtin(self):
        """Test that simple_builtin creates a builtin function."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r"], [x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))
        self.assertTrue(getattr(test_func, BUILTIN_ATTR, False))
        self.assertTrue(hasattr(test_func, CODEGEN_FUNC_ATTR))
        self.assertTrue(hasattr(test_func, EVAL_RETURN_TYPE_ATTR))

    def test_simple_builtin_with_ll_asm(self):
        """Test simple_builtin with ll.asm() call."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r"], [x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_with_asm_volatile(self):
        """Test simple_builtin with ll.asm_volatile() call."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm_volatile("mov.u32 %0, %1;", ["=r"], [x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_with_binary_operand(self):
        """Test simple_builtin with binary operation in operands."""

        @ll.simple_builtin
        def test_func(ptr: ll.Tensor[ll.uint64], offset: ll.int32) -> ll.uint64:
            ret: ll.uint64 = 0
            ll.asm("ld.shared.u64 %0, [%1];", ["=l", "l"], [ret, ptr + offset])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_void_return(self):
        """Test simple_builtin with void return type."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.void:
            ll.asm("mov.u32 %0, %1;", ["=r"], [x])

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_multiple_operands(self):
        """Test simple_builtin with multiple operands."""

        @ll.simple_builtin
        def test_func(a: ll.int32, b: ll.int32, c: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mad.lo.s32 %0, %1, %2, %3;", ["=r", "r", "r", "r"], [ret, a, b, c])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_output_constraint(self):
        """Test simple_builtin with output constraint (=r)."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r", "r"], [ret, x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_input_constraint(self):
        """Test simple_builtin with input constraint (r)."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r", "r"], [ret, x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_float_types(self):
        """Test simple_builtin with float types."""

        @ll.simple_builtin
        def test_func(x: ll.float32) -> ll.float32:
            ret: ll.float32 = 0.0
            ll.asm("mov.f32 %0, %1;", ["=f", "f"], [ret, x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))

    def test_simple_builtin_uint64_types(self):
        """Test simple_builtin with uint64 types."""

        @ll.simple_builtin
        def test_func(x: ll.uint64) -> ll.uint64:
            ret: ll.uint64 = 0
            ll.asm("mov.u64 %0, %1;", ["=l", "l"], [ret, x])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))


class TestSimpleBuiltinErrors(unittest.TestCase):
    """Test error cases for simple_builtin."""

    def test_simple_builtin_no_asm_call(self):
        """Test that simple_builtin raises error if no asm() call."""
        with self.assertRaises(ValueError):

            @ll.simple_builtin
            def test_func(x: ll.int32) -> ll.int32:
                return x

    def test_simple_builtin_multiple_asm_calls(self):
        """Test that simple_builtin handles multiple asm() calls (should use first one)."""
        # This should work - we use the first asm call
        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r"], [x])
            ll.asm("add.u32 %0, %0, 1;", ["+r"], [ret])
            return ret

        self.assertTrue(hasattr(test_func, BUILTIN_ATTR))


class TestSimpleBuiltinCodegen(unittest.TestCase):
    """Test code generation for simple_builtin functions."""

    def test_codegen_func_exists(self):
        """Test that codegen function is generated."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r", "r"], [ret, x])
            return ret

        codegen_func = getattr(test_func, CODEGEN_FUNC_ATTR)
        self.assertIsNotNone(codegen_func)
        self.assertTrue(callable(codegen_func))

    def test_codegen_func_returns_builtin(self):
        """Test that codegen function returns Builtin object."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r", "r"], [ret, x])
            return ret

        codegen_func = getattr(test_func, CODEGEN_FUNC_ATTR)
        result = codegen_func("arg1", "arg2")

        from little_kernel.language.builtin_base import Builtin
        self.assertIsInstance(result, Builtin)
        self.assertIn("asm volatile", result.body)
        self.assertIn("mov.u32", result.body)


class TestSimpleBuiltinEvalReturnType(unittest.TestCase):
    """Test eval_return_type generation for simple_builtin functions."""

    def test_eval_return_type_exists(self):
        """Test that eval_return_type function is generated."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r"], [x])
            return ret

        eval_return_type = getattr(test_func, EVAL_RETURN_TYPE_ATTR)
        self.assertIsNotNone(eval_return_type)
        self.assertTrue(callable(eval_return_type))

    def test_eval_return_type_int32(self):
        """Test eval_return_type for int32 return."""

        @ll.simple_builtin
        def test_func(x: ll.int32) -> ll.int32:
            ret: ll.int32 = 0
            ll.asm("mov.u32 %0, %1;", ["=r"], [x])
            return ret

        eval_return_type = getattr(test_func, EVAL_RETURN_TYPE_ATTR)
        # This will be called during type inference with actual argument types
        # For now, just check it's callable
        self.assertTrue(callable(eval_return_type))


if __name__ == "__main__":
    unittest.main()
