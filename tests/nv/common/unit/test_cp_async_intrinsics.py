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
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.memory as memory


class TestCpAsyncIntrinsics(unittest.TestCase):
    """Unit tests for the cp.async smem-staging intrinsic family (sm_80+)."""

    def _assert_builtin(self, fn):
        self.assertTrue(getattr(fn, BUILTIN_ATTR, False))
        self.assertTrue(callable(getattr(fn, CODEGEN_FUNC_ATTR)))

    def test_family_is_builtin_and_exported(self):
        for name in ("cp_async_cg_16", "cp_async_ca_16", "cp_async_ca_8", "cp_async_ca_4", "cp_async_cg_16_zfill",
                     "cp_async_commit_group", "cp_async_wait_group"):
            self.assertTrue(hasattr(memory, name), name)
            self.assertTrue(hasattr(ll, name), f"ll.{name} not exported")
            self._assert_builtin(getattr(memory, name))

    def test_cg_16_codegen(self):
        codegen = getattr(memory.cp_async_cg_16, CODEGEN_FUNC_ATTR)
        result = codegen("SM_A + 16", "SRC")
        self.assertIsInstance(result, Builtin)
        self.assertIn("cp.async.cg.shared.global [%0], [%1], 16;", result.body)
        self.assertIn("__cvta_generic_to_shared", result.body)
        self.assertIn('"memory"', result.body)
        self.assertEqual(result.return_val, "cp_async_cg_16_fn(SM_A + 16, SRC)")

    def test_ca_variants_codegen(self):
        for size, fn in ((16, memory.cp_async_ca_16), (8, memory.cp_async_ca_8), (4, memory.cp_async_ca_4)):
            codegen = getattr(fn, CODEGEN_FUNC_ATTR)
            result = codegen("SM_A", "SRC")
            self.assertIn(f"cp.async.ca.shared.global [%0], [%1], {size};", result.body)
            self.assertEqual(result.return_val, f"cp_async_ca_{size}_fn(SM_A, SRC)")

    def test_zfill_carries_src_size(self):
        codegen = getattr(memory.cp_async_cg_16_zfill, CODEGEN_FUNC_ATTR)
        result = codegen("SM_A", "SRC", "SS")
        self.assertIn("cp.async.cg.shared.global [%0], [%1], 16, %2;", result.body)
        self.assertIn('"r"(src_size)', result.body)
        self.assertEqual(result.return_val, "cp_async_cg_16_zfill_fn(SM_A, SRC, SS)")

    def test_commit_group_codegen(self):
        codegen = getattr(memory.cp_async_commit_group, CODEGEN_FUNC_ATTR)
        result = codegen()
        self.assertIn("cp.async.commit_group;", result.body)
        self.assertEqual(result.return_val, "cp_async_commit_group_fn()")

    def test_wait_group_templated_constant(self):
        codegen = getattr(memory.cp_async_wait_group, CODEGEN_FUNC_ATTR)
        result = codegen("0")
        self.assertIn("cp.async.wait_group %0;", result.body)
        self.assertIn('"n"(N)', result.body)
        self.assertIn("template <int N>", result.body)
        self.assertEqual(result.return_val, "cp_async_wait_group_fn<(0)>()")
        # A different constant reuses ONE template body (dedup by name).
        result2 = codegen("2")
        self.assertEqual(result.body, result2.body)
        self.assertEqual(result2.return_val, "cp_async_wait_group_fn<(2)>()")

    def test_wait_group_same_name_across_sites(self):
        # Helper-name stability is what lets the builtin-body dedup in
        # codegen_base.writeln_builtin collapse multiple call sites.
        codegen = getattr(memory.cp_async_cg_16, CODEGEN_FUNC_ATTR)
        r1 = codegen("A", "B")
        r2 = codegen("C", "D")
        self.assertEqual(r1.body, r2.body)
        self.assertNotEqual(r1.return_val, r2.return_val)


if __name__ == "__main__":
    unittest.main()
