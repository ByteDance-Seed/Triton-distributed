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
"""CPU regression tests for FP8 (e4m3/e5m2) WGMMA operand-dtype support.

Covers the typed-IR gap ir.wgmma.mma_async_ss.fp8_operands: before the
extension, any "e4m3"/"e5m2" token raised ValueError at trace time in
_wgmma_ptx_dtype/_wgmma_int_ptx_dtype because neither alias table contained
an FP8 entry. The FP8 family must now emit
`wgmma.mma_async.sync.aligned.m64nNk32.f32.{ta}.{tb}` with an f32 accumulator,
k32 depth, scale-d predicate semantics, and no scale/transpose immediates.

These tests compose no GPU code: they exercise codegen_func bodies directly
(the same call the emitter makes) and assert on the generated asm text. Paired
GPU validation is left to the queue.
"""

import re
import unittest

from little_kernel.backends.nv.sm90.language.intrin import wgmma as W


class TestFp8AliasTable(unittest.TestCase):
    """Whitelists accept FP8 tokens and reject everything else."""

    def test_e4m3_aliases(self):
        for token in ("e4m3", "fp8_e4m3", "float8_e4m3", "float8_e4m3fn", "E4M3", " e4m3 "):
            self.assertEqual(W._wgmma_fp8_ptx_dtype(token), "e4m3")

    def test_e5m2_aliases(self):
        for token in ("e5m2", "fp8_e5m2", "float8_e5m2", "E5M2", " e5m2 "):
            self.assertEqual(W._wgmma_fp8_ptx_dtype(token), "e5m2")

    def test_ll_dtype_object_name(self):

        class _FakeDtype:
            name = "fp8_e4m3"

        self.assertEqual(W._wgmma_fp8_ptx_dtype(_FakeDtype()), "e4m3")

    def test_rejects_non_fp8(self):
        for token in ("bf16", "f16", "s8", "u8", "tf32", "int4", ""):
            with self.assertRaises(ValueError):
                W._wgmma_fp8_ptx_dtype(token)

    def test_detects_fp8_family(self):
        self.assertTrue(W._wgmma_is_fp8_dtype("e4m3"))
        self.assertTrue(W._wgmma_is_fp8_dtype("fp8_e5m2"))
        self.assertFalse(W._wgmma_is_fp8_dtype("bf16"))
        self.assertFalse(W._wgmma_is_fp8_dtype("s8"))


class TestSsFp8Codegen(unittest.TestCase):
    """wgmma_mma_async_ss with FP8 dtypes emits the k32 f32 opcode."""

    def _gen(self, **kw):
        args = dict(acc="c", desc_a="da", desc_b="db", n=128)
        args.update(kw)
        return W.codegen_wgmma_mma_async_ss(**args)

    def test_e4m3_e4m3_opcode(self):
        b = self._gen(dtype="e4m3")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3", b.body)
        self.assertIn("float* c", b.body)
        self.assertIn('"+f"(c[0])', b.body)
        self.assertIn('"+f"(c[63])', b.body)
        # scale-d predicate path, accumulate by default
        self.assertIn("setp.ne.b32 p, 1, 0;", b.body)
        # helper-name uniqueness
        self.assertIn("wgmma_mma_ss_n128k32_f32_e4m3_e4m3_d1_fn", b.body)
        self.assertIn("wgmma_mma_ss_n128k32_f32_e4m3_e4m3_d1_fn(c, da, db)", b.return_val)

    def test_e5m2_e5m2_opcode(self):
        b = self._gen(dtype="e5m2")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e5m2.e5m2", b.body)

    def test_mixed_e4m3_e5m2_opcode(self):
        b = self._gen(dtype="e4m3", dtype_b="e5m2")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e5m2", b.body)
        self.assertIn("wgmma_mma_ss_n128k32_f32_e4m3_e5m2_d1_fn", b.body)

    def test_mixed_e5m2_e4m3_opcode(self):
        b = self._gen(dtype="e4m3", dtype_b="e5m2")
        c = self._gen(dtype="e5m2", dtype_b="e4m3")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e5m2.e4m3", c.body)
        self.assertIn("wgmma_mma_ss_n128k32_f32_e5m2_e4m3_d1_fn", c.body)
        self.assertNotEqual(b.body, c.body)

    def test_alias_token_normalizes_in_opcode(self):
        b = self._gen(dtype="fp8_e4m3")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3", b.body)

    def test_scale_d_overwrite_immediate(self):
        b = self._gen(dtype="e4m3", scale_d=0)
        self.assertIn("setp.ne.b32 p, 0, 0;", b.body)
        self.assertIn("wgmma_mma_ss_n128k32_f32_e4m3_e4m3_d0_fn", b.body)

    def test_scale_d_runtime_predicate(self):
        b = self._gen(dtype="e4m3", scale_d="sd_var")
        self.assertIn("setp.ne.b32 p, %66, 0;", b.body)
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("wgmma_mma_ss_n128k32_f32_e4m3_e4m3_dr_fn", b.body)
        self.assertIn("wgmma_mma_ss_n128k32_f32_e4m3_e4m3_dr_fn(c, da, db, sd_var)", b.return_val)

    def test_n_whitelist_applies(self):
        b = self._gen(dtype="e4m3", n=256)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3", b.body)
        self.assertIn('"+f"(c[127])', b.body)
        b = self._gen(dtype="e4m3", n=8)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3", b.body)
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", n=13)
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", n=264)

    def test_rejects_scale_a_b(self):
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", scale_a=-1)
        with self.assertRaises(ValueError):
            self._gen(dtype="e5m2", scale_b=-1)

    def test_rejects_transpose_immediates(self):
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", trans_a=1)
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", dtype_b="e5m2", trans_b=1)

    def test_no_scale_or_trans_operands_baked_into_asm(self):
        # FP8 wgmma has only the scale-d predicate after desc_a,desc_b: the
        # operand tail must not carry sa/sb/ta/tb immediates.
        b = self._gen(dtype="e4m3")
        m = re.search(r"m64n128k32\.f32\.e4m3\.e4m3[\s\S]*?;\\n", b.body)
        assert m is not None
        self.assertIn("%64,%65,p;", m.group(0))
        self.assertNotIn("1,1,0", m.group(0))

    def test_mixed_family_only_one_fp8_side_rejects(self):
        # dtype_b overrides must still be FP8 tokens.
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", dtype_b="bf16")
        with self.assertRaises(ValueError):
            self._gen(dtype="bf16", dtype_b="e4m3")


class TestDtypeBFamilyGuard(unittest.TestCase):
    """dtype_b is only legal for the FP8 family; never silently ignored.

    Before the guard, dtype='bf16', dtype_b='f16' (or any non-FP8 / mixed
    combination) fell through to the f16/bf16 or integer path with dtype_b
    silently dropped, emitting a same-type opcode the caller did not ask
    for. Every such combination must raise ValueError.
    """

    def test_non_fp8_dtype_b_rejected_for_float_family(self):
        for bad_b in ("f16", "bf16", "s8", "tf32"):
            with self.subTest(dtype_b=bad_b), self.assertRaises(ValueError):
                W.codegen_wgmma_mma_async_ss("c", "da", "db", 128, dtype="bf16", dtype_b=bad_b)
            with self.subTest(dtype_b=bad_b), self.assertRaises(ValueError):
                W.codegen_wgmma_mma_async_rs("c", "a0", "a1", "a2", "a3", "db", 128, dtype="f16", dtype_b=bad_b)

    def test_fp8_dtype_b_with_non_fp8_dtype_rejected(self):
        with self.assertRaises(ValueError):
            W.codegen_wgmma_mma_async_ss("c", "da", "db", 128, dtype="bf16", dtype_b="e4m3")
        with self.assertRaises(ValueError):
            W.codegen_wgmma_mma_async_rs("c", "a0", "a1", "a2", "a3", "db", 128, dtype="f16", dtype_b="e5m2")

    def test_fp8_dtype_b_with_int_dtype_rejected(self):
        # The integer family has no mixed-operand form; an FP8 dtype_b must
        # not be silently dropped by the int routing that runs first.
        with self.assertRaises(ValueError):
            W.codegen_wgmma_mma_async_ss("c", "da", "db", 128, dtype="s8", dtype_b="e4m3")
        with self.assertRaises(ValueError):
            W.codegen_wgmma_mma_async_rs("c", "a0", "a1", "a2", "a3", "db", 128, dtype="u8", dtype_b="e5m2")

    def test_fp8_mixed_pair_still_accepted(self):
        b = W.codegen_wgmma_mma_async_ss("c", "da", "db", 128, dtype="e4m3", dtype_b="e5m2")
        self.assertIn("f32.e4m3.e5m2", b.body)
        b = W.codegen_wgmma_mma_async_rs("c", "a0", "a1", "a2", "a3", "db", 128, dtype="e5m2", dtype_b="e4m3")
        self.assertIn("f32.e5m2.e4m3", b.body)

    def test_none_dtype_b_default_paths_unchanged(self):
        # Omitting dtype_b entirely keeps the legacy float/int routing.
        b = W.codegen_wgmma_mma_async_ss("c", "da", "db", 128, dtype="bf16")
        self.assertIn("k16.f32.bf16.bf16", b.body)
        b = W.codegen_wgmma_mma_async_ss("c", "da", "db", 128, dtype="s8")
        self.assertIn("k32.s32.s8.s8", b.body)


class TestRsFp8Codegen(unittest.TestCase):
    """wgmma_mma_async_rs with FP8 dtypes emits the k32 f32 opcode."""

    def _gen(self, **kw):
        args = dict(acc="c", a0="a0", a1="a1", a2="a2", a3="a3", desc_b="db", n=128)
        args.update(kw)
        return W.codegen_wgmma_mma_async_rs(**args)

    def test_e4m3_e4m3_opcode(self):
        b = self._gen(dtype="e4m3")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3", b.body)
        self.assertIn("float* c", b.body)
        self.assertIn('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b)', b.body)
        self.assertIn("wgmma_mma_rs_n128k32_f32_e4m3_e4m3_d1_fn", b.body)
        self.assertIn("wgmma_mma_rs_n128k32_f32_e4m3_e4m3_d1_fn(c, a0, a1, a2, a3, db)", b.return_val)

    def test_mixed_e5m2_e4m3_opcode(self):
        b = self._gen(dtype="e5m2", dtype_b="e4m3")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.f32.e5m2.e4m3", b.body)

    def test_scale_d_runtime_predicate(self):
        b = self._gen(dtype="e4m3", scale_d="sd_var")
        self.assertIn("setp.ne.b32 p, %69, 0;", b.body)
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("wgmma_mma_rs_n128k32_f32_e4m3_e4m3_dr_fn", b.body)

    def test_rejects_flags(self):
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", scale_a=-1)
        with self.assertRaises(ValueError):
            self._gen(dtype="e5m2", trans_b=1)

    def test_n_whitelist_applies(self):
        with self.assertRaises(ValueError):
            self._gen(dtype="e4m3", n=12)


class TestExistingFamiliesUnchanged(unittest.TestCase):
    """Regression guard: f16/bf16/int routing and text are not perturbed."""

    def test_bf16_ss_shape_unchanged(self):
        b = W.codegen_wgmma_mma_async_ss(acc="c", desc_a="da", desc_b="db", n=128, dtype="bf16")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", b.body)
        self.assertIn("wgmma_mma_ss_n128_bf16_sa1sb1d1ta0tb0_fn", b.body)

    def test_int_s8_ss_shape_unchanged(self):
        b = W.codegen_wgmma_mma_async_ss(acc="c", desc_a="da", desc_b="db", n=128, dtype="s8")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8", b.body)
        self.assertIn('"+r"(c[0])', b.body)

    def test_int_s8_rs_shape_unchanged(self):
        b = W.codegen_wgmma_mma_async_rs(acc="c", a0="a0", a1="a1", a2="a2", a3="a3", desc_b="db", n=128, dtype="u8")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k32.s32.u8.u8", b.body)

    def test_int_flags_still_rejected(self):
        with self.assertRaises(ValueError):
            W.codegen_wgmma_mma_async_ss(acc="c", desc_a="da", desc_b="db", n=128, dtype="s8", trans_a=1)


if __name__ == "__main__":
    unittest.main()
