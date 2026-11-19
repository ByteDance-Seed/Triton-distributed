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
"""CPU unit tests for the SM100 kind::tf32 UMMA intrinsics.

These tests pin the instruction-descriptor encoding, the emitted PTX
instruction/cta_group signatures, the K-block descriptor stepping, the
accumulate predicate handling and the backend.json registrations. They do not
require an SM100 device; numerical correctness on hardware is out of scope.
"""

import json
import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.sm100.language.intrin.umma as umma
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CTA_GROUP_ATTR,
    CODEGEN_FUNC_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    Builtin,
)


class TestEncodeUmmaTf32Idesc(unittest.TestCase):
    """Tests for the kind::tf32 instruction-descriptor encoder."""

    def test_is_builtin_with_uint32_return(self):
        self.assertTrue(getattr(umma.encode_umma_tf32_idesc, BUILTIN_ATTR, False))
        return_type = getattr(umma.encode_umma_tf32_idesc, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type(None, None), ll.uint32)

    def test_direct_call_is_rejected(self):
        with self.assertRaises(RuntimeError):
            umma.encode_umma_tf32_idesc(128, 64)

    def test_codegen_descriptor_fields(self):
        result = getattr(umma.encode_umma_tf32_idesc, CODEGEN_FUNC_ATTR)("m", "n")
        self.assertIsInstance(result, Builtin)
        body = result.body
        # D is FP32 accumulator (idesc bit 4).
        self.assertIn("(1u << 4)", body)
        # A and B are TF32 (type code 2) at bit fields 7..9 and 10..12.
        self.assertIn("(2u << 7)", body)
        self.assertIn("(2u << 10)", body)
        # N/8 at bits 17..22 and M/16 at bits 24..28.
        self.assertIn("((n >> 3) << 17)", body)
        self.assertIn("((m >> 4) << 24)", body)
        # FP32 accumulate means the f16-style bit (negate/transpose-free
        # K-major layout) must not accidentally emit the f16 type code 0/1.
        self.assertNotIn("(0u << 7)", body)
        self.assertNotIn("(1u << 7)", body)
        self.assertEqual(result.return_val, "encode_umma_tf32_idesc_fn(m, n)")

    def test_idesc_tf32_type_code_is_2(self):
        # The kind::tf32 idesc uses operand type code 2 (TF32); kind::f16 uses
        # code 0/1. There is no f16 idesc encoder in this module, so pin the
        # exact emitted constants here as the distinctness guard.
        body = getattr(umma.encode_umma_tf32_idesc, CODEGEN_FUNC_ATTR)("m", "n").body
        import re
        a_codes = re.findall(r"\((\d+)u << 7\)", body)
        b_codes = re.findall(r"\((\d+)u << 10\)", body)
        self.assertEqual(a_codes, ["2"])
        self.assertEqual(b_codes, ["2"])


class TestUmmaTf32SsIntrinsics(unittest.TestCase):
    """Tests for the shared-memory-operand kind::tf32 UMMA variants."""

    def _check_ss_variant(self, op, cta_group):
        self.assertTrue(getattr(op, BUILTIN_ATTR, False))
        self.assertEqual(getattr(op, CTA_GROUP_ATTR, 1), cta_group)
        return_type = getattr(op, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type(), ll.void)
        result = getattr(op, CODEGEN_FUNC_ATTR)("tmem_c", "desc_a", "desc_b", "idesc", "accum")
        self.assertIsInstance(result, Builtin)
        self.assertIn(f"tcgen05.mma.cta_group::{cta_group}.kind::tf32", result.body)
        # SS form: A and B are 64-bit shared-memory descriptors.
        self.assertIn("[%0], %1, %2, %3, p;", result.body)
        # accum drives a predicate so accum=0 clears the accumulator.
        self.assertIn("setp.ne.b32 p, %4, 0;", result.body)
        return result

    def test_ss_1sm_codegen(self):
        result = self._check_ss_variant(umma.umma_tf32_ss_1sm, 1)
        self.assertEqual(result.return_val, "umma_tf32_1sm_fn(tmem_c, desc_a, desc_b, idesc, accum)")

    def test_ss_2sm_codegen(self):
        result = self._check_ss_variant(umma.umma_tf32_ss_2sm, 2)
        self.assertEqual(result.return_val, "umma_tf32_2sm_fn(tmem_c, desc_a, desc_b, idesc, accum)")

    def test_ss_variants_do_not_self_elect(self):
        # Like the f16 family, the issuing-thread election stays external.
        for op in (umma.umma_tf32_ss_1sm, umma.umma_tf32_ss_2sm):
            result = getattr(op, CODEGEN_FUNC_ATTR)("d", "a", "b", "idesc", "accum")
            self.assertNotIn("elect.sync", result.body)


class TestUmmaTf32SsKblock(unittest.TestCase):
    """Tests for the TF32 SS K-block helpers."""

    def _check_kblock(self, op, cta_group):
        self.assertTrue(getattr(op, BUILTIN_ATTR, False))
        self.assertEqual(getattr(op, CTA_GROUP_ATTR, 1), cta_group)
        result = getattr(op, CODEGEN_FUNC_ATTR)("d", "a", "b", "idesc", "64", "accum")
        self.assertIsInstance(result, Builtin)
        self.assertIn("#pragma unroll", result.body)
        # One kind::tf32 instruction covers K=8.
        self.assertIn("static_assert(K > 0 && K % 8 == 0", result.body)
        self.assertIn("constexpr int KTiles = K / 8", result.body)
        self.assertIn(f"tcgen05.mma.cta_group::{cta_group}.kind::tf32", result.body)
        # One K8 tf32 tile is 32 bytes = 2 x 16-byte descriptor units, applied
        # to the 14-bit address field of both descriptors.
        self.assertIn("((desc_a + k * 2) & 0x3FFFULL)", result.body)
        self.assertIn("((desc_b + k * 2) & 0x3FFFULL)", result.body)
        # Only the first tile honors accum; the rest accumulate unconditionally.
        self.assertIn("setp.ne.b32 p, %4, 0;", result.body)
        self.assertIn(", %3, 1;", result.body)
        # Election stays with the caller.
        self.assertNotIn("elect.sync", result.body)
        return result

    def test_1sm_kblock(self):
        result = self._check_kblock(umma.umma_tf32_ss_1sm_kblock, 1)
        self.assertEqual(result.return_val, "umma_tf32_1sm_kblock_fn<64>(d, a, b, idesc, accum)")

    def test_2sm_kblock(self):
        result = self._check_kblock(umma.umma_tf32_ss_2sm_kblock, 2)
        self.assertEqual(result.return_val, "umma_tf32_2sm_kblock_fn<64>(d, a, b, idesc, accum)")

    def test_kblock_step_differs_from_f16(self):
        # A K16 f16 tile is also 32 bytes but the tf32 tile granularity must be
        # K/8, not the f16 K/16; a copy-paste of the f16 helper would regress.
        tf32 = getattr(umma.umma_tf32_ss_1sm_kblock, CODEGEN_FUNC_ATTR)("d", "a", "b", "idesc", "64", "accum").body
        self.assertNotIn("K % 16 == 0", tf32)
        self.assertNotIn("K / 16", tf32)


class TestUmmaTf32TsIntrinsics(unittest.TestCase):
    """Tests for the tensor-memory-A kind::tf32 UMMA variants."""

    def _check_ts_variant(self, op, cta_group):
        self.assertTrue(getattr(op, BUILTIN_ATTR, False))
        self.assertEqual(getattr(op, CTA_GROUP_ATTR, 1), cta_group)
        return_type = getattr(op, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type(), ll.void)
        result = getattr(op, CODEGEN_FUNC_ATTR)("tmem_c", "tmem_a", "desc_b", "idesc", "accum")
        self.assertIsInstance(result, Builtin)
        self.assertIn(f"tcgen05.mma.cta_group::{cta_group}.kind::tf32", result.body)
        # TS form: A is a TMEM address operand ([%1]), B an smem descriptor.
        self.assertIn("[%0], [%1], %2, %3, p;", result.body)
        self.assertIn('"r"(tmem_a)', result.body)
        self.assertIn('"l"(desc_b)', result.body)
        self.assertIn("setp.ne.b32 p, %4, 0;", result.body)
        return result

    def test_ts_1sm_codegen(self):
        result = self._check_ts_variant(umma.umma_tf32_ts_1sm, 1)
        self.assertEqual(result.return_val, "umma_tf32_ts_1sm_fn(tmem_c, tmem_a, desc_b, idesc, accum)")

    def test_ts_2sm_codegen(self):
        result = self._check_ts_variant(umma.umma_tf32_ts_2sm, 2)
        self.assertEqual(result.return_val, "umma_tf32_ts_2sm_fn(tmem_c, tmem_a, desc_b, idesc, accum)")


class TestUmmaTf32Registration(unittest.TestCase):
    """The kind::tf32 intrinsics must be registered in the SM100 backend."""

    def test_backend_json_registers_all_tf32_intrinsics(self):
        import pathlib
        backend_json = (pathlib.Path(umma.__file__).resolve().parents[2] / "backend.json")
        with open(backend_json, "r", encoding="utf-8") as f:
            registry = json.load(f)
        self.assertIn("language_exports", registry)
        intrinsics = registry["language_exports"]
        prefix = "little_kernel.backends.nv.sm100.language.intrin.umma:"
        expected = (
            "encode_umma_tf32_idesc",
            "umma_tf32_ss_1sm",
            "umma_tf32_ss_1sm_kblock",
            "umma_tf32_ss_2sm",
            "umma_tf32_ss_2sm_kblock",
            "umma_tf32_ts_1sm",
            "umma_tf32_ts_2sm",
        )
        for name in expected:
            with self.subTest(name=name):
                self.assertEqual(intrinsics.get(name), prefix + name)
                target = getattr(umma, name, None)
                self.assertIsNotNone(target, f"registered symbol {name} must exist in umma.py")
                self.assertTrue(getattr(target, BUILTIN_ATTR, False))

    def test_codegen_integration(self):
        """A frontend call lowers to the tcgen05 kind::tf32 PTX sequence."""
        import ast as _ast
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
        for name in ("umma_tf32_ss_1sm", "umma_tf32_ss_2sm"):
            with self.subTest(name=name):
                src = ("def kern(tmem_c: uint32, desc_a: uint64, desc_b: uint64,"
                       " idesc: uint32) -> void:\n"
                       f"    {name}(tmem_c, desc_a, desc_b, idesc, 1)\n")
                ctx = dict(ll.__dict__)
                ctx[name] = getattr(umma, name)
                out = codegen_cuda(_ast.parse(src), ctx)
                self.assertIn(".kind::tf32", out)


if __name__ == "__main__":
    unittest.main()
