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
"""Unit tests for the typed SM90 WGMMA smem descriptor encoder.

encode_wgmma_smem_desc builds the 64-bit shared-memory matrix descriptor
consumed by wgmma.mma_async on sm_90a:

    bits [13:0]  = (addr      >> 4) & 0x3FFF   (start address / 16)
    bits [29:16] = (lbo_bytes >> 4) & 0x3FFF   (leading byte offset / 16)
    bits [45:32] = (sbo_bytes >> 4) & 0x3FFF   (stride byte offset / 16)
    bits [63:62] = layout code (0 none, 1 128B, 2 64B, 3 32B swizzle)

This closes the gap between the SM100-only encode_umma_smem_desc (version
bit 46, 3-bit layout at bit 61) and raw make_desc(lo, hi) bit-concat that
forced draft authors to hand-assemble descriptor bits. The Python reference
below mirrors a known-good sm_90a encoding independently of the emitted C++
so a bit-field regression is caught by string-level codegen tests (CPU only;
paired GPU validation is queued separately).
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
import little_kernel.backends.nv.sm90.language.intrin.wgmma as wgmma_intrin


def _reference_desc(addr, lbo_bytes, sbo_bytes, layout_code):
    """Reference SM90 WGMMA descriptor encoding (PTX matrix descriptor)."""
    d = ((addr >> 4) & 0x3FFF)
    d |= ((lbo_bytes >> 4) & 0x3FFF) << 16
    d |= ((sbo_bytes >> 4) & 0x3FFF) << 32
    d |= layout_code << 62
    return d


class TestWgmmaDescEncoderExport(unittest.TestCase):

    def test_is_builtin(self):
        self.assertTrue(getattr(wgmma_intrin.encode_wgmma_smem_desc, BUILTIN_ATTR, False))

    def test_ll_namespace_export(self):
        self.assertIs(ll.encode_wgmma_smem_desc, wgmma_intrin.encode_wgmma_smem_desc)

    def test_layout_whitelist_has_exactly_four_modes(self):
        self.assertEqual(sorted(wgmma_intrin._WGMMA_DESC_LAYOUTS.values()), [0, 1, 2, 3])


class TestWgmmaDescEncoderLayoutValidation(unittest.TestCase):

    def setUp(self):
        self.codegen = getattr(wgmma_intrin.encode_wgmma_smem_desc, CODEGEN_FUNC_ATTR)

    def test_int_layouts_accepted(self):
        for code in (0, 1, 2, 3):
            b = self.codegen("addr", "lbo", "sbo", code)
            self.assertIsInstance(b, Builtin)
            self.assertIn("<< 62;", b.body)
            self.assertIn(f"(uint64_t){code} << 62;", b.body)
            self.assertIn(f"encode_wgmma_smem_desc_l{code}_fn", b.body)
            self.assertEqual(b.return_val, f"encode_wgmma_smem_desc_l{code}_fn(addr, lbo, sbo)")

    def test_string_layouts_accepted(self):
        names = {0: "none", 1: "swizzle_128b", 2: "swizzle_64b", 3: "swizzle_32b"}
        for code, name in names.items():
            b = self.codegen("addr", "lbo", "sbo", name)
            self.assertIn(f"(uint64_t){code} << 62;", b.body)

    def test_default_layout_is_128b_swizzle(self):
        b = self.codegen("addr", "lbo", "sbo")
        self.assertIn("(uint64_t)1 << 62;", b.body)

    def test_illegal_layout_values_rejected(self):
        for bad in (4, 5, -1, 7, 100):
            with self.assertRaises(ValueError):
                self.codegen("addr", "lbo", "sbo", bad)

    def test_bool_layout_rejected(self):
        # bool is an int subclass; a shape-immediate field must not silently
        # accept True/False as 1/0.
        with self.assertRaises(ValueError):
            self.codegen("addr", "lbo", "sbo", True)
        with self.assertRaises(ValueError):
            self.codegen("addr", "lbo", "sbo", False)

    def test_float_and_unknown_string_rejected(self):
        with self.assertRaises(ValueError):
            self.codegen("addr", "lbo", "sbo", 1.0)
        with self.assertRaises(ValueError):
            self.codegen("addr", "lbo", "sbo", "128B")
        with self.assertRaises(ValueError):
            self.codegen("addr", "lbo", "sbo", "half")


class TestWgmmaDescEncoderBitFields(unittest.TestCase):
    """Bit-field shapes in the emitted C++ must match the exact reference
    encoding (the same expression the PTX ISA documents and the sm_90a
    teacher kernels use)."""

    def setUp(self):
        self.codegen = getattr(wgmma_intrin.encode_wgmma_smem_desc, CODEGEN_FUNC_ATTR)

    def test_field_expressions(self):
        b = self.codegen("addr", "lbo", "sbo", "swizzle_128b")
        self.assertIn("d |= (uint64_t)((addr >> 4) & 0x3FFF);", b.body)
        self.assertIn("d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;", b.body)
        self.assertIn("d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;", b.body)
        self.assertIn("d |= (uint64_t)1 << 62;", b.body)

    def test_reference_encoding_matches_c_semantics(self):
        # Cross-check the Python reference against representative inputs,
        # including the shape family used by real sm_90a GEMM pipelines
        # (SBO=1024 for an 8x128B atom, LBO=0 K-major / 8192 MN-major).
        cases = [
            (0, 0, 1024, 1),
            (8192, 0, 1024, 1),
            (32768, 8192, 1024, 1),
            (16384, 0, 512, 2),
            (16384, 0, 256, 3),
            (0, 0, 64, 0),
            (196608, 8192, 1024, 1),
        ]
        for addr, lbo, sbo, code in cases:
            d = _reference_desc(addr, lbo, sbo, code)
            # Fields must not overlap and the layout must land at [63:62].
            self.assertEqual(d & 0x3FFF, (addr >> 4) & 0x3FFF)
            self.assertEqual((d >> 16) & 0x3FFF, (lbo >> 4) & 0x3FFF)
            self.assertEqual((d >> 32) & 0x3FFF, (sbo >> 4) & 0x3FFF)
            self.assertEqual((d >> 62) & 0x3, code)
            # No stray bits outside the four fields.
            mask = (0x3FFF | (0x3FFF << 16) | (0x3FFF << 32) | (0x3 << 62))
            self.assertEqual(d & ~mask, 0)


class TestWgmmaDescEncoderFullKernelCodegen(unittest.TestCase):
    """Compile a full little-kernel pipeline that builds descriptors with the
    typed encoder and issues typed WGMMA m64n128k16 -> the exact instruction
    the sm_90a BF16 GEMM teacher uses. CPU codegen only (no GPU here)."""

    def test_typed_desc_feeds_wgmma_m64n128k16(self):
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def mma_with_typed_desc(addr_in: ll.Tensor[ll.uint32]) -> ll.void:
            base: ll.uint32 = addr_in[0]
            da: ll.uint64 = ll.encode_wgmma_smem_desc(base, 0, 1024, "swizzle_128b")
            db: ll.uint64 = ll.encode_wgmma_smem_desc(base, 8192, 1024, "swizzle_128b")
            acc = ll.empty([64], dtype=ll.float32, scope="local")
            ll.wgmma_mma_async_ss(acc, da, db, 128, dtype="bf16")

        code = mma_with_typed_desc.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        self.assertIn("encode_wgmma_smem_desc_l1_fn", code)
        self.assertIn("(uint64_t)1 << 62;", code)
        self.assertIn("((addr >> 4) & 0x3FFF)", code)
        self.assertIn("((sbo_bytes >> 4) & 0x3FFF) << 32;", code)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", code)

    def test_mixed_swizzle_modes_emit_distinct_helpers(self):
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def mma_two_modes(addr_in: ll.Tensor[ll.uint32]) -> ll.void:
            base: ll.uint32 = addr_in[0]
            da: ll.uint64 = ll.encode_wgmma_smem_desc(base, 0, 1024, "swizzle_128b")
            db: ll.uint64 = ll.encode_wgmma_smem_desc(base, 0, 512, "swizzle_64b")
            acc = ll.empty([32], dtype=ll.float32, scope="local")
            ll.wgmma_mma_async_ss(acc, da, db, 64, dtype="bf16")

        code = mma_two_modes.compile(PASSES["cuda"], codegen_cuda, num_threads=128)
        self.assertIn("encode_wgmma_smem_desc_l1_fn", code)
        self.assertIn("encode_wgmma_smem_desc_l2_fn", code)
        self.assertIn("(uint64_t)1 << 62;", code)
        self.assertIn("(uint64_t)2 << 62;", code)


if __name__ == "__main__":
    unittest.main()
