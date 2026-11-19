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


class NoSuchType:
    """Sentinel annotation target that is deliberately NOT an LLType.

    Using it as a parameter annotation is resolvable as a Python object by
    static analysis, but ``simple_builtin`` codegen cannot map it to an
    LLType, so the output-parameter guard raises ValueError.
    """


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

    def test_codegen_multi_output_params_become_references(self):
        """Params named by '=' output constraints lower as C++ references."""

        @ll.simple_builtin
        def ld_nc_f32x4(ptr: ll.Tensor[ll.float32], off: ll.int32, r0: ll.uint32, r1: ll.uint32, r2: ll.uint32,
                        r3: ll.uint32) -> ll.void:
            ll.asm("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];", ["=r", "=r", "=r", "=r", "l"],
                   [r0, r1, r2, r3, ptr + off])

        codegen_func = getattr(ld_nc_f32x4, CODEGEN_FUNC_ATTR)
        result = codegen_func("y + off", "off", "c0", "c1", "c2", "c3")

        for i in range(4):
            self.assertIn(f"uint32_t& r{i}", result.body)
            self.assertIn(f'"=r"(r{i})', result.body)
        self.assertIn("float* ptr", result.body)
        self.assertIn("int32_t off", result.body)  # input params stay by-value
        self.assertNotIn("uint32_t& off", result.body)

    def test_codegen_input_only_params_stay_by_value(self):
        """Store builtins (all-input constraints) keep by-value params."""

        @ll.simple_builtin
        def st_cs_f32x4(ptr: ll.Tensor[ll.float32], off: ll.int32, v0: ll.uint32, v1: ll.uint32, v2: ll.uint32,
                        v3: ll.uint32) -> ll.void:
            ll.asm("st.global.cs.v4.b32 [%0], {%1,%2,%3,%4};", ["l", "r", "r", "r", "r"], [ptr + off, v0, v1, v2, v3])

        codegen_func = getattr(st_cs_f32x4, CODEGEN_FUNC_ATTR)
        result = codegen_func("dx", "off2", "a", "b", "c", "d")

        for i in range(4):
            self.assertIn(f"uint32_t v{i}", result.body)
            self.assertNotIn(f"uint32_t& v{i}", result.body)

    def test_codegen_output_param_unresolved_type_raises(self):
        """An '=' output param with unresolvable type raises, never auto&."""

        # Deliberately unresolvable *type* annotation: the name exists here so
        # static analysis is happy, but `ll` has no such LLType attribute, so
        # codegen must raise instead of silently emitting an auto reference.
        NoSuchType = "not-an-lltype"

        @ll.simple_builtin
        def bad_out(r0: NoSuchType, x: ll.uint32) -> ll.void:
            ll.asm("mov.u32 %0, %1;", ["=r", "r"], [r0, x])

        codegen_func = getattr(bad_out, CODEGEN_FUNC_ATTR)
        with self.assertRaises(ValueError):
            codegen_func("c0", "1")

    def test_codegen_two_output_asm_params_become_references(self):
        """Two-output asm builtin emits one 'T&' declaration per output param.

        Regression guard for the output-reference contract described in
        simple_builtin.py: reverting the '{t}& {name}' emission to a by-value
        parameter silently discards both asm results (a miscompile that
        still compiles), so the emitted signature must be checked directly.
        """

        @ll.simple_builtin
        def ld_pair(ptr: ll.Tensor[ll.uint32], r0: ll.uint32, r1: ll.uint32) -> ll.void:
            ll.asm("ld.global.v2.u32 {%0,%1}, [%2];", ["=r", "=r", "l"], [r0, r1, ptr])

        codegen_func = getattr(ld_pair, CODEGEN_FUNC_ATTR)
        result = codegen_func("p", "a", "b")

        self.assertIn("uint32_t& r0", result.body)
        self.assertIn("uint32_t& r1", result.body)
        self.assertIn('"=r"(r0)', result.body)
        self.assertIn('"=r"(r1)', result.body)
        # The pointer input stays by value; no reference is emitted for it.
        self.assertIn("uint32_t* ptr", result.body)
        self.assertNotIn("uint32_t*& ptr", result.body)
        self.assertNotIn("uint32_t r0,", result.body)
        self.assertNotIn("uint32_t r1)", result.body)

    def test_codegen_two_output_with_input_param_references(self):
        """Output params take 'T&' while the input param stays by value."""

        @ll.simple_builtin
        def unpack_pair(x: ll.uint64, lo: ll.uint32, hi: ll.uint32) -> ll.void:
            ll.asm("mov.b64 {%0,%1}, %2;", ["=r", "=r", "l"], [lo, hi, x])

        codegen_func = getattr(unpack_pair, CODEGEN_FUNC_ATTR)
        result = codegen_func("v", "a", "b")

        self.assertIn("uint64_t x", result.body)  # input stays by value
        self.assertIn("uint32_t& lo", result.body)
        self.assertIn("uint32_t& hi", result.body)
        self.assertNotIn("uint64_t& x", result.body)

    def test_codegen_output_param_missing_annotation_raises(self):
        """An '=' output param with NO annotation raises, never 'auto&'."""

        @ll.simple_builtin
        def no_ann(r0, x: ll.uint32) -> ll.void:
            ll.asm("mov.u32 %0, %1;", ["=r", "r"], [r0, x])

        codegen_func = getattr(no_ann, CODEGEN_FUNC_ATTR)
        with self.assertRaises(ValueError) as ctx:
            codegen_func("c0", "1")
        self.assertIn("r0", str(ctx.exception))
        self.assertIn("reference", str(ctx.exception))


class TestSimpleBuiltinClobbers(unittest.TestCase):
    """Clobber lists must survive parsing in both positional and keyword form.

    Dropping a `clobbers` list (e.g. "memory") is silent and dangerous: the
    generated Builtin body loses the scheduler-visible barrier, letting the
    backend hoist memory operations across the asm statement.
    """

    def _codegen_body(self, func):
        codegen_func = getattr(func, CODEGEN_FUNC_ATTR)
        return codegen_func().body

    def test_clobbers_positional_kept(self):
        """Positional 4th-arg clobbers appear in the generated body."""

        @ll.simple_builtin
        def fence_pos() -> ll.void:
            ll.asm("bar.sync 0;", [], [], ["memory"])

        body = self._codegen_body(fence_pos)
        self.assertIn('"memory"', body)

    def test_clobbers_keyword_kept(self):
        """Keyword `clobbers=[...]` must not be silently dropped (regression)."""

        @ll.simple_builtin
        def fence_kw() -> ll.void:
            ll.asm("bar.sync 0;", [], [], clobbers=["memory"])

        body = self._codegen_body(fence_kw)
        self.assertIn('"memory"', body)

    def test_clobbers_keyword_with_input_operands(self):
        """Keyword clobbers plus input operands emit inputs then clobbers."""

        @ll.simple_builtin
        def fence_multi(x: ll.int32) -> ll.void:
            ll.asm("dummy;", ["r"], [x], clobbers=["memory", "cc"])

        codegen_func = getattr(fence_multi, CODEGEN_FUNC_ATTR)
        body = codegen_func("a").body
        self.assertIn('"memory", "cc"', body)
        self.assertIn('"r"(x)', body)

    def test_clobbers_keyword_and_positional_merge(self):
        """Keyword clobbers are honored in addition to positional ones."""

        @ll.simple_builtin
        def fence_both() -> ll.void:
            ll.asm("bar.sync 0;", [], [], ["memory"], clobbers=["cc"])

        body = self._codegen_body(fence_both)
        self.assertIn('"memory"', body)
        self.assertIn('"cc"', body)

    def test_clobbers_absent_emits_no_clobber_section(self):
        """Without clobbers there is no 4th asm section (baseline behavior)."""

        @ll.simple_builtin
        def no_clob() -> ll.void:
            ll.asm("bar.sync 0;", [], [])

        body = self._codegen_body(no_clob)
        self.assertNotIn('"memory"', body)


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
