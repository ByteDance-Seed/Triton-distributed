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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
#
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
Golden text tests for the atomic_sub / atomic_xor intrinsics
(language/intrin/atomic.py).

Contract under test:

- ll.atomic_sub / ll.atomic_xor (pointer form) emit the CUDA atomicSub /
  atomicXor builtins directly and return the old value with the pointer
  element type.
- ll.atomic_sub_global_{i32,u32} / ll.atomic_xor_global_{i32,u32}
  ((tensor, index) element form) emit ``atomicSub((tensor) + (index), v)``
  / ``atomicXor((tensor) + (index), v)`` with the tensor spelling
  parenthesized, matching the add/cas/exch element family.
- The element type is gated at trace time to int32/uint32: the CUDA C++
  atomic API has no float or 64-bit atomicSub/atomicXor overloads
  (device_atomic_functions.h), so float32/int64/uint64 pointer element
  types raise TypeError (legality before emission, wiki skill step 5).
- Every intrinsic declares the RW_GMEM_ATOMIC_SYNC memory effect, so
  effect-driven analyses never see them as side-effect free.

All assertions are on output text and trace-time typing only
(deterministic, no GPU).
"""

import unittest

import little_kernel.backends.nv.common.language.intrin.atomic as atomic
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, Builtin
from little_kernel.core.memory_effects import RW_GMEM_ATOMIC_SYNC
from little_kernel.core.type_system import int32, int64, float32, uint32, uint64


class _FakePointer:
    """Minimal pointer-like object for return-type inference tests."""

    def __init__(self, inner):
        self._inner = inner

    def is_pointer(self):
        return True

    @property
    def inner_type(self):
        return self._inner


class TestAtomicSubPointerForm(unittest.TestCase):
    """Intrinsic: 32-bit atomic decrement (CUDA atomicSub)."""

    def test_is_builtin(self):
        self.assertTrue(getattr(atomic.atomic_sub, BUILTIN_ATTR, False))

    def test_declares_atomic_rw_effect(self):
        effects = getattr(atomic.atomic_sub, MEMORY_EFFECTS_ATTR, ())
        self.assertTrue(effects)
        for eff in effects:
            self.assertIs(eff.kind, RW_GMEM_ATOMIC_SYNC)

    def test_return_type_int32_uint32(self):
        return_type_fn = getattr(atomic.atomic_sub, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(_FakePointer(int32), "1"), int32)
        self.assertIs(return_type_fn(_FakePointer(uint32), "1"), uint32)

    def test_return_type_rejects_float(self):
        # No float atomicSub overload in the CUDA C++ atomic API.
        return_type_fn = getattr(atomic.atomic_sub, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(float32), "1.0")

    def test_return_type_rejects_64_bit(self):
        return_type_fn = getattr(atomic.atomic_sub, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(int64), "1")
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(uint64), "1")

    def test_codegen_golden(self):
        codegen_func = getattr(atomic.atomic_sub, CODEGEN_FUNC_ATTR)
        result = codegen_func("budget + 0", "1")
        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.body, "")
        self.assertEqual(result.return_val, "atomicSub(budget + 0, 1)")

    def test_codegen_deterministic(self):
        codegen_func = getattr(atomic.atomic_sub, CODEGEN_FUNC_ATTR)
        first = codegen_func("c", "w")
        second = codegen_func("c", "w")
        self.assertEqual(first.return_val, second.return_val)


class TestAtomicXorPointerForm(unittest.TestCase):
    """Intrinsic: 32-bit atomic bitwise xor (CUDA atomicXor)."""

    def test_is_builtin(self):
        self.assertTrue(getattr(atomic.atomic_xor, BUILTIN_ATTR, False))

    def test_declares_atomic_rw_effect(self):
        effects = getattr(atomic.atomic_xor, MEMORY_EFFECTS_ATTR, ())
        self.assertTrue(effects)
        for eff in effects:
            self.assertIs(eff.kind, RW_GMEM_ATOMIC_SYNC)

    def test_return_type_int32_uint32(self):
        return_type_fn = getattr(atomic.atomic_xor, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(_FakePointer(int32), "mask"), int32)
        self.assertIs(return_type_fn(_FakePointer(uint32), "mask"), uint32)

    def test_return_type_rejects_float(self):
        return_type_fn = getattr(atomic.atomic_xor, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(float32), "mask")

    def test_return_type_rejects_64_bit(self):
        return_type_fn = getattr(atomic.atomic_xor, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(int64), "mask")
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(uint64), "mask")

    def test_codegen_golden(self):
        codegen_func = getattr(atomic.atomic_xor, CODEGEN_FUNC_ATTR)
        result = codegen_func("parity + 0", "mask")
        self.assertIsInstance(result, Builtin)
        self.assertEqual(result.body, "")
        self.assertEqual(result.return_val, "atomicXor(parity + 0, mask)")


class TestAtomicSubElementForm(unittest.TestCase):
    """(tensor, index) element forms: atomicSub on tensor[index]."""

    def test_is_builtin(self):
        self.assertTrue(getattr(atomic.atomic_sub_global_i32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(atomic.atomic_sub_global_u32, BUILTIN_ATTR, False))

    def test_return_types(self):
        self.assertIs(getattr(atomic.atomic_sub_global_i32, EVAL_RETURN_TYPE_ATTR)("t", "i", "v"), int32)
        self.assertIs(getattr(atomic.atomic_sub_global_u32, EVAL_RETURN_TYPE_ATTR)("t", "i", "v"), uint32)

    def test_declares_atomic_rw_effect(self):
        for fn in (atomic.atomic_sub_global_i32, atomic.atomic_sub_global_u32):
            effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
            self.assertTrue(effects)
            for eff in effects:
                self.assertIs(eff.kind, RW_GMEM_ATOMIC_SYNC)
                # One-element region derived from the "index" operand,
                # like the sibling add/cas/exch element forms.
                self.assertIsNotNone(eff.region)
                self.assertEqual(eff.region.offset_operand, "index")
                self.assertEqual(eff.region.extent, 1)

    def test_codegen_golden(self):
        for fn in (atomic.atomic_sub_global_i32, atomic.atomic_sub_global_u32):
            codegen_func = getattr(fn, CODEGEN_FUNC_ATTR)
            result = codegen_func("grads", "ticket", "-1")
            self.assertIsInstance(result, Builtin)
            self.assertEqual(result.body, "")
            self.assertEqual(result.return_val, "atomicSub((grads) + (ticket), -1)")


class TestAtomicXorElementForm(unittest.TestCase):
    """(tensor, index) element forms: atomicXor on tensor[index]."""

    def test_is_builtin(self):
        self.assertTrue(getattr(atomic.atomic_xor_global_i32, BUILTIN_ATTR, False))
        self.assertTrue(getattr(atomic.atomic_xor_global_u32, BUILTIN_ATTR, False))

    def test_return_types(self):
        self.assertIs(getattr(atomic.atomic_xor_global_i32, EVAL_RETURN_TYPE_ATTR)("t", "i", "v"), int32)
        self.assertIs(getattr(atomic.atomic_xor_global_u32, EVAL_RETURN_TYPE_ATTR)("t", "i", "v"), uint32)

    def test_declares_atomic_rw_effect(self):
        for fn in (atomic.atomic_xor_global_i32, atomic.atomic_xor_global_u32):
            effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
            self.assertTrue(effects)
            for eff in effects:
                self.assertIs(eff.kind, RW_GMEM_ATOMIC_SYNC)
                self.assertIsNotNone(eff.region)
                self.assertEqual(eff.region.offset_operand, "index")
                self.assertEqual(eff.region.extent, 1)

    def test_codegen_golden(self):
        for fn in (atomic.atomic_xor_global_i32, atomic.atomic_xor_global_u32):
            codegen_func = getattr(fn, CODEGEN_FUNC_ATTR)
            result = codegen_func("masks", "lane", "0x1")
            self.assertIsInstance(result, Builtin)
            self.assertEqual(result.body, "")
            self.assertEqual(result.return_val, "atomicXor((masks) + (lane), 0x1)")


if __name__ == "__main__":
    unittest.main()
