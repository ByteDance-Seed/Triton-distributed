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
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Unit tests for the cycle-009 codegen-surface additions:

  * gmem_ld_v4_u32_nc         (ld.global.nc.v4.b32, read-only path)
  * cp_async_16               (cp.async.cg.shared.global, 16 B)
  * cp_async_commit_group     (cp.async.commit_group)
  * cp_async_wait_group(n)    (cp.async.wait_group, const or runtime n)
  * wgmma_m64n256k16_tnsp     (tnspA/tnspB-parameterized bf16 WGMMA)

Legality invariants under test:
  * cp.async ops carry a "memory" clobber; the global operand uses the 64-bit
    "l" constraint and the smem operand the 32-bit "r" constraint;
  * wait_group emits a bare PTX immediate for Python-int n (0..7) and a
    0..7 switch for runtime n; out-of-range Python ints are rejected;
  * the tnsp WGMMA emits the caller-selected immediates, is instruction- and
    constraint-identical to wgmma_m64n256k16 at tnsp=(0,0), and rejects
    non-{0,1} or non-compile-time tnsp operands.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.memory as memory_intrin
import little_kernel.backends.nv.sm90.language.intrin.wgmma as wgmma_intrin


class TestGmemV4U32Nc(unittest.TestCase):

    def test_is_builtin_and_exported(self):
        self.assertTrue(getattr(memory_intrin.gmem_ld_v4_u32_nc, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.gmem_ld_v4_u32_nc, BUILTIN_ATTR, False))

    def test_codegen(self):
        codegen_func = getattr(memory_intrin.gmem_ld_v4_u32_nc, CODEGEN_FUNC_ATTR)
        result = codegen_func("t", "i", "a", "b", "c", "d")
        self.assertIsInstance(result, Builtin)
        self.assertIn("ld.global.nc.v4.b32", result.body)
        self.assertIn('"l"(ptr)', result.body)
        self.assertIn('"memory"', result.body)
        self.assertIn("(const void*)(t + i)", result.return_val)


class TestCpAsyncIntrinsics(unittest.TestCase):

    def test_exported(self):
        self.assertTrue(getattr(ll.cp_async_16, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.cp_async_commit_group, BUILTIN_ATTR, False))
        self.assertTrue(getattr(ll.cp_async_wait_group, BUILTIN_ATTR, False))

    def test_cp_async_16_codegen(self):
        codegen_func = getattr(memory_intrin.cp_async_16, CODEGEN_FUNC_ATTR)
        result = codegen_func("saddr", "g", "off")
        self.assertIsInstance(result, Builtin)
        self.assertIn("cp.async.cg.shared.global", result.body)
        self.assertIn(", 16;", result.body)
        self.assertIn('"r"(smem_addr)', result.body)
        self.assertIn('"l"(gptr)', result.body)
        self.assertIn('"memory"', result.body)
        self.assertIn("(const void*)(g + off)", result.return_val)
        self.assertIn("cp_async_16_fn(saddr", result.return_val)

    def test_cp_async_commit_group_codegen(self):
        codegen_func = getattr(memory_intrin.cp_async_commit_group, CODEGEN_FUNC_ATTR)
        result = codegen_func()
        self.assertIn("cp.async.commit_group;", result.body)
        self.assertIn('"memory"', result.body)

    def test_wait_group_constant_immediate(self):
        codegen_func = getattr(memory_intrin.cp_async_wait_group, CODEGEN_FUNC_ATTR)
        result = codegen_func(2)
        self.assertIn("cp.async.wait_group 2;", result.body)
        self.assertNotIn("switch", result.body)
        result = codegen_func(0)
        self.assertIn("cp.async.wait_group 0;", result.body)

    def test_wait_group_runtime_switch(self):
        codegen_func = getattr(memory_intrin.cp_async_wait_group, CODEGEN_FUNC_ATTR)
        result = codegen_func("n_pending")
        self.assertIn("switch (n)", result.body)
        for k in range(8):
            self.assertIn(f"cp.async.wait_group {k};", result.body)
        self.assertIn("cp_async_wait_group_fn(n_pending)", result.return_val)

    def test_wait_group_rejects_out_of_range_constant(self):
        codegen_func = getattr(memory_intrin.cp_async_wait_group, CODEGEN_FUNC_ATTR)
        with self.assertRaises(ValueError):
            codegen_func(8)
        with self.assertRaises(ValueError):
            codegen_func(-1)

    def test_multiple_wait_variants_all_bodies_emitted(self):
        """Regression: builtin helper bodies must be emitted per distinct
        generated helper, not only on the first call of the builtin.
        wgmma_wait_group with two different constants in one kernel used to
        emit the second call with no definition (call_codegen builtin_cache
        keyed on the function object)."""
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def wait_twice(n: ll.int32) -> ll.void:
            ll.wgmma_wait_group(1)
            ll.wgmma_wait_group(2)
            ll.wgmma_wait_group(n)

        code = wait_twice.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("void wgmma_wait_group_1_fn()", code)
        self.assertIn("void wgmma_wait_group_2_fn()", code)
        self.assertIn("void wgmma_wait_group_fn(int n)", code)
        self.assertIn("wgmma_wait_group_1_fn();", code)
        self.assertIn("wgmma_wait_group_2_fn();", code)
        self.assertIn("wgmma_wait_group_fn(n);", code)

    def test_generated_cuda_contains_cp_async_ptx(self):
        """End-to-end text check: a draft kernel using the cp.async family must
        emit cp.async.cg.shared.global / commit_group / wait_group in the
        final CUDA source."""
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def cp_pipe(src: ll.Tensor[ll.float32], steps: ll.int32) -> ll.void:
            smem_addr: ll.uint32 = 0
            ll.cp_async_16(smem_addr, src, 0)
            ll.cp_async_commit_group()
            ll.cp_async_wait_group(0)
            ll.cp_async_wait_group(steps)

        code = cp_pipe.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertIn("cp.async.cg.shared.global", code)
        self.assertIn("cp.async.commit_group", code)
        self.assertIn("cp.async.wait_group 0;", code)
        self.assertIn("switch (n)", code)


class TestWgmmaTnsp(unittest.TestCase):

    def test_exported(self):
        self.assertTrue(getattr(ll.wgmma_m64n256k16_tnsp, BUILTIN_ATTR, False))

    def test_codegen_tnspb1(self):
        codegen_func = getattr(wgmma_intrin.wgmma_m64n256k16_tnsp, CODEGEN_FUNC_ATTR)
        result = codegen_func("acc", "da", "db", 0, 1)
        self.assertIsInstance(result, Builtin)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", result.body)
        self.assertIn("p,1,1,0,1;", result.body)
        self.assertEqual(result.body.count('"+f"(c['), 128)

    def test_tnsp00_matches_fixed_variant_body(self):
        """tnsp=(0,0) must be instruction-identical to wgmma_m64n256k16
        (only the helper fn name may differ)."""
        tnsp_func = getattr(wgmma_intrin.wgmma_m64n256k16_tnsp, CODEGEN_FUNC_ATTR)
        base_func = getattr(wgmma_intrin.wgmma_m64n256k16, CODEGEN_FUNC_ATTR)
        a = tnsp_func("acc", "da", "db", 0, 0).body.replace("_tnsp_fn", "_fn")
        b = base_func("acc", "da", "db").body
        self.assertEqual(a, b)

    def test_rejects_non_binary_or_runtime_tnsp(self):
        codegen_func = getattr(wgmma_intrin.wgmma_m64n256k16_tnsp, CODEGEN_FUNC_ATTR)
        with self.assertRaises(ValueError):
            codegen_func("acc", "da", "db", 0, 2)
        with self.assertRaises(ValueError):
            codegen_func("acc", "da", "db", 1, -1)
        # runtime expression strings are not PTX immediates -> rejected
        with self.assertRaises(ValueError):
            codegen_func("acc", "da", "db", 0, "tnsp_b")

    def test_generated_cuda_contains_tnsp_stem(self):
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def mma_nn(lo_in: ll.Tensor[ll.uint32]) -> ll.void:
            lo: ll.uint32 = lo_in[0]
            hi: ll.uint32 = lo_in[1]
            da: ll.uint64 = ll.make_desc(lo, hi)
            db: ll.uint64 = ll.make_desc(lo, hi)
            acc = ll.empty([128], dtype=ll.float32, scope="local")
            ll.wgmma_m64n256k16_tnsp(acc, da, db, 0, 1)

        code = mma_nn.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", code)
        self.assertIn("p,1,1,0,1;", code)


if __name__ == "__main__":
    unittest.main()
