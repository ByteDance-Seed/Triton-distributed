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
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Golden text tests for the atomic_inc intrinsic (language/intrin/atomic.py).

Contract under test:

- ll.atomic_inc(addr, bound) emits an int32/uint32-overloaded
  atomic_inc_fn helper wrapping CUDA atomicInc (wrap-around ticket
  take; int32 counters are reinterpreted to unsigned) and returns the
  old value as uint32.
- The return type is legal only for int32 and uint32 pointer element
  types (counters commonly live in int32 workspaces); other element
  types raise TypeError (legality before emission, wiki skill step 5).

All assertions are on output text only (deterministic, no GPU).
"""

import unittest

import little_kernel.backends.nv.common.language.intrin.atomic as atomic
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, Builtin
from little_kernel.core.type_system import int32, float32, uint32


class _FakePointer:
    """Minimal pointer-like object for return-type inference tests."""

    def __init__(self, inner):
        self._inner = inner

    def is_pointer(self):
        return True

    @property
    def inner_type(self):
        return self._inner


class TestAtomicInc(unittest.TestCase):
    """Intrinsic: wrap-around atomic ticket (CUDA atomicInc)."""

    def test_is_builtin(self):
        self.assertTrue(getattr(atomic.atomic_inc, BUILTIN_ATTR, False))

    def test_return_type_uint32_pointer(self):
        return_type_fn = getattr(atomic.atomic_inc, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(_FakePointer(uint32), "bound"), uint32)

    def test_return_type_int32_pointer_is_uint32(self):
        # int32 counters are reinterpreted to unsigned at emission; the
        # ticket value itself is always uint32.
        return_type_fn = getattr(atomic.atomic_inc, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(_FakePointer(int32), "bound"), uint32)

    def test_return_type_rejects_float(self):
        return_type_fn = getattr(atomic.atomic_inc, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            return_type_fn(_FakePointer(float32), "bound")

    def test_return_type_untyped_addr_accepted_as_int32(self):
        # Untyped (None) address falls back to int32, matching the int32
        # counter path.
        return_type_fn = getattr(atomic.atomic_inc, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(None, "bound"), uint32)

    def test_codegen_golden(self):
        codegen_func = getattr(atomic.atomic_inc, CODEGEN_FUNC_ATTR)
        result = codegen_func("retired + 0", "bcount - 1")
        self.assertIsInstance(result, Builtin)
        self.assertIn("return atomicInc(ptr, bound);", result.body)
        self.assertIn("return atomicInc(reinterpret_cast<uint32_t*>(ptr), bound);", result.body)
        self.assertIn("<cstdint>", result.includes)
        self.assertEqual(result.return_val, "atomic_inc_fn(retired + 0, bcount - 1)")

    def test_codegen_deterministic(self):
        codegen_func = getattr(atomic.atomic_inc, CODEGEN_FUNC_ATTR)
        first = codegen_func("c", "w")
        second = codegen_func("c", "w")
        self.assertEqual(first.body, second.body)
        self.assertEqual(first.return_val, second.return_val)


if __name__ == "__main__":
    unittest.main()
