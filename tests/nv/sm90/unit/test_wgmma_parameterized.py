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
"""Unit tests for the parameterized WGMMA m64n64k16 family.

Legality model (per cw.codegen.tilelang-ptx-inline-assembly, wiki commit
2e523d63b315bedc2adb022b750d9af0f42f7428): a (dtype, flag) whitelist is
checked BEFORE any asm text is composed; illegal configurations raise
ValueError, never a malformed template.
"""

import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.sm90.language.intrin.wgmma as wgmma
from little_kernel.language.builtin_base import BUILTIN_ATTR


class TestWgmmaParameterizedExports(unittest.TestCase):
    """New intrinsics are builtins and reachable from the ll namespace."""

    def test_ss_is_builtin(self):
        self.assertTrue(getattr(wgmma.wgmma_m64n64k16_ss, BUILTIN_ATTR, False))

    def test_rs_is_builtin(self):
        self.assertTrue(getattr(wgmma.wgmma_m64n64k16_rs, BUILTIN_ATTR, False))

    def test_ll_namespace_exports(self):
        self.assertIs(ll.wgmma_m64n64k16_ss, wgmma.wgmma_m64n64k16_ss)
        self.assertIs(ll.wgmma_m64n64k16_rs, wgmma.wgmma_m64n64k16_rs)
        self.assertTrue(hasattr(ll, "pack_f16_float"))


class TestWgmmaLegacyCodegenUnchanged(unittest.TestCase):
    """The legacy wgmma_m64n64k16 codegen must stay exactly the bf16,
    scale_d=1, trans=0,0 template it was before this change."""

    def test_legacy_template_pinned(self):
        b = wgmma.codegen_wgmma_m64n64k16("acc", "da", "db")
        self.assertIn("wgmma_m64n64k16_fn(", b.body)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", b.body)
        self.assertIn('"%32,%33,p,1,1,0,0;\\n}\\n"', b.body)
        self.assertEqual(b.return_val, "wgmma_m64n64k16_fn(acc, da, db)")

    def test_legacy_and_default_parameterized_agree(self):
        """SS with all defaults emits the same opcode/operand text as legacy."""
        legacy = wgmma.codegen_wgmma_m64n64k16("acc", "da", "db")
        param = wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db")
        for token in ("m64n64k16.f32.bf16.bf16", '"%32,%33,p,1,1,0,0;'):
            self.assertIn(token, legacy.body)
            self.assertIn(token, param.body)
        self.assertIn("wgmma_m64n64k16_ss_bf16_fn(", param.body)


class TestWgmmaParameterizedCodegen(unittest.TestCase):

    def test_ss_f16_template(self):
        b = wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", dtype="f16")
        self.assertIn("m64n64k16.f32.f16.f16", b.body)
        self.assertIn('"%32,%33,p,1,1,0,0;\\n}\\n"', b.body)
        self.assertIn("wgmma_m64n64k16_ss_f16_fn(", b.body)

    def test_ss_trans_flags(self):
        b = wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", dtype="bf16", trans_a=1, trans_b=1)
        self.assertIn('"%32,%33,p,1,1,1,1;\\n}\\n"', b.body)

    def test_ss_scale_d_zero(self):
        b = wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", scale_d=0)
        self.assertIn("setp.ne.b32 p, 0, 0;", b.body)

    def test_ss_runtime_scale_d(self):
        # A traced runtime value (not int-convertible) becomes a register pred
        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", scale_d=_RuntimeInt())
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("setp.ne.b32 p, %34, 0;", b.body)
        self.assertIn(", int scale_d)", b.body)

    def test_rs_bf16_default(self):
        b = wgmma.codegen_wgmma_m64n64k16_rs("acc", "a0", "a1", "a2", "a3", "db")
        self.assertIn("m64n64k16.f32.bf16.bf16", b.body)
        self.assertIn('"{%32,%33,%34,%35},%36,p,1,1,0;\\n}\\n"', b.body)
        self.assertIn("wgmma_m64n64k16_rs_bf16_fn(", b.body)

    def test_rs_f16_trans_b(self):
        b = wgmma.codegen_wgmma_m64n64k16_rs("acc", "a0", "a1", "a2", "a3", "db", dtype="f16", trans_b=1)
        self.assertIn("m64n64k16.f32.f16.f16", b.body)
        self.assertIn('"{%32,%33,%34,%35},%36,p,1,1,1;\\n}\\n"', b.body)
        self.assertIn("wgmma_m64n64k16_rs_f16_fn(", b.body)

    def test_rs_runtime_scale_d(self):

        class _RuntimeInt:
            pass

        b = wgmma.codegen_wgmma_m64n64k16_rs("acc", "a0", "a1", "a2", "a3", "db", scale_d=_RuntimeInt())
        self.assertIn('"r"(scale_d)', b.body)
        self.assertIn("setp.ne.b32 p, %37, 0;", b.body)


class TestWgmmaLegalityWhitelist(unittest.TestCase):
    """Counterexamples: illegal dtype/flags must raise before asm composition."""

    def test_tf32_dtype_rejected_ss(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", dtype="tf32")

    def test_tf32_dtype_rejected_rs(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_m64n64k16_rs("acc", "a0", "a1", "a2", "a3", "db", dtype="tf32")

    def test_f8_dtype_rejected(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", dtype="e4m3")

    def test_trans_flag_two_rejected(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", trans_b=2)

    def test_scale_d_two_rejected(self):
        with self.assertRaises(ValueError):
            wgmma.codegen_wgmma_m64n64k16_rs("acc", "a0", "a1", "a2", "a3", "db", scale_d=2)

    def test_dtype_aliases_accepted(self):
        for alias in ("bf16", "bfloat16"):
            self.assertIn("bf16", wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", dtype=alias).body)
        for alias in ("f16", "fp16", "float16", "half"):
            self.assertIn("f16", wgmma.codegen_wgmma_m64n64k16_ss("acc", "da", "db", dtype=alias).body)


if __name__ == "__main__":
    unittest.main()
