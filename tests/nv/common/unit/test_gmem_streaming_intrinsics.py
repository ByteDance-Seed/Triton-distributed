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
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Golden text tests for the streaming v4 / L2 prefetch / DSM scalar-store
intrinsic family (language/intrin/memory.py + barrier.py).

Contract under test:

- ll.ld_global_lu_v4_u32(mem, elem_offset, out) emits the same wrapper
  shape as the ld_global_v4_u32/cs/cg family with inst ld.global.lu.v4.u32.
- ll.prefetch_l2(mem, elem_offset) emits prefetch.global.L2 on the raw
  element-offset address; element offset arithmetic identical to the v4
  family.
- ll.dsm_store_f32(base, rank, byte_off, value) emits the
  mapa.shared::cluster.u32 + st.shared::cluster.f32 pair and returns void.

All assertions are on output text only (deterministic, no GPU).
"""

import unittest

import little_kernel.backends.nv.common.language.intrin.barrier as barrier
import little_kernel.backends.nv.common.language.intrin.memory as memory
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, Builtin
from little_kernel.core.type_system import void


class TestLdGlobalLuV4(unittest.TestCase):
    """Intrinsic: last-use-hint 16B global load (sibling of cs/cg)."""

    def test_is_builtin(self):
        self.assertTrue(getattr(memory.ld_global_lu_v4_u32, BUILTIN_ATTR, False))

    def test_return_type_is_void(self):
        return_type_fn = getattr(memory.ld_global_lu_v4_u32, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(), void)

    def test_codegen_golden(self):
        codegen_func = getattr(memory.ld_global_lu_v4_u32, CODEGEN_FUNC_ATTR)
        result = codegen_func("g", "off", "dst")
        self.assertIsInstance(result, Builtin)
        self.assertIn("ld.global.lu.v4.u32 {%0, %1, %2, %3}, [%4];", result.body)
        self.assertEqual(result.return_val, "gmem_ld_global_lu_v4_u32_fn(g + off, dst)")
        self.assertIn("<cstdint>", result.includes)


class TestPrefetchL2(unittest.TestCase):
    """Intrinsic: prefetch.global.L2 of the 128-byte line at mem+offset."""

    def test_is_builtin(self):
        self.assertTrue(getattr(memory.prefetch_l2, BUILTIN_ATTR, False))

    def test_return_type_is_void(self):
        return_type_fn = getattr(memory.prefetch_l2, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn(), void)

    def test_codegen_golden(self):
        codegen_func = getattr(memory.prefetch_l2, CODEGEN_FUNC_ATTR)
        result = codegen_func("gate", "line * 64")
        self.assertIsInstance(result, Builtin)
        self.assertIn('asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr));', result.body)
        self.assertEqual(result.return_val, "prefetch_l2_fn(gate + line * 64)")

    def test_codegen_deterministic(self):
        codegen_func = getattr(memory.prefetch_l2, CODEGEN_FUNC_ATTR)
        first = codegen_func("g", "0")
        second = codegen_func("g", "0")
        self.assertEqual(first.body, second.body)
        self.assertEqual(first.return_val, second.return_val)


class TestDsmStoreF32(unittest.TestCase):
    """Intrinsic: typed DSM scalar store (mapa + st.shared::cluster.f32)."""

    def test_is_builtin(self):
        self.assertTrue(getattr(barrier.dsm_store_f32, BUILTIN_ATTR, False))

    def test_return_type_is_void(self):
        return_type_fn = getattr(barrier.dsm_store_f32, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type_fn("b", "r", "o", "v"), void)

    def test_codegen_golden(self):
        codegen_func = getattr(barrier.dsm_store_f32, CODEGEN_FUNC_ATTR)
        result = codegen_func("base_u32", "tid", "rank * 4", "mine")
        self.assertIsInstance(result, Builtin)
        self.assertIn(
            '"mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr)'
            ' : "r"(smem_base_u32), "r"(target_rank)', result.body)
        self.assertIn('"st.shared::cluster.f32 [%0], %1;"'
                      ' :: "r"(remote_addr + byte_off), "f"(value) : "memory"', result.body)
        self.assertEqual(result.return_val, "dsm_store_f32_fn(base_u32, tid, rank * 4, mine)")
        self.assertIn("<cstdint>", result.includes)


if __name__ == "__main__":
    unittest.main()
