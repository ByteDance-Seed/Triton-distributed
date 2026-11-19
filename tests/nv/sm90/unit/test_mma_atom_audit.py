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
"""CPU regression tests for the atom.mma audit surface (coverage.mma_pipeline).

The atom-level WgMMA description answers coverage questions (accumulator
registers per thread, fragment sizes, SMEM bytes per K-step) that the
emitter-side intrinsics cannot answer because they only compose PTX text.
Every derived number is checked against the PTX contract the emitter tests
already pin (n//2 accumulator registers for m64nNk16 f32 WGMMA; full N
whitelist {8..256 step 8}; f16/bf16 operands only).
"""

import unittest

import little_kernel.core.type_system as ll_type
from little_kernel.backends.nv.sm90.atom.mma import tensor_core
import little_kernel.backends.nv.sm90.language.intrin.wgmma as _wgmma_emit


class TestLegalNWhitelistMatchesEmitter(unittest.TestCase):
    """The atom's N whitelist must be exactly the emitter-side PTX whitelist."""

    def test_same_set(self):
        self.assertEqual(tuple(tensor_core.WGMMA_F16_LEGAL_NCOLS), tuple(_wgmma_emit._WGMMA_LEGAL_NCOLS))

    def test_full_ptx_range(self):
        self.assertEqual(tensor_core.WGMMA_F16_LEGAL_NCOLS, tuple(range(8, 257, 8)))


class TestWgmmaAtomConstruction(unittest.TestCase):

    def test_defaults_bf16_ss(self):
        m = tensor_core.wgmma_m64nNk16_f32(64)
        self.assertEqual((m.M, m.N, m.K), (64, 64, 16))
        self.assertIs(m.A_dtype, ll_type.bfloat16)
        self.assertIs(m.B_dtype, ll_type.bfloat16)
        self.assertIs(m.ACC_dtype, ll_type.float32)
        self.assertTrue(m.A_in_SMEM)
        self.assertTrue(m.B_in_SMEM)

    def test_every_legal_n_constructs(self):
        for n in tensor_core.WGMMA_F16_LEGAL_NCOLS:
            m = tensor_core.wgmma_m64nNk16_f32(n)
            self.assertEqual(m.N, n, n)

    def test_f16_operands_accepted(self):
        m = tensor_core.wgmma_m64nNk16_f32(64, A_dtype=ll_type.float16)
        self.assertIs(m.A_dtype, ll_type.float16)

    def test_rs_path_flag(self):
        m = tensor_core.wgmma_m64nNk16_f32(64, A_in_SMEM=False)
        self.assertFalse(m.A_in_SMEM)
        self.assertTrue(m.B_in_SMEM)


class TestWgmmaAtomLegalityWhitelist(unittest.TestCase):
    """Illegal configurations must raise ValueError before any derived
    quantity is computed — same contract as the emitter-side dispatchers."""

    def test_n_below_range_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(4)

    def test_n_above_range_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(264)

    def test_n_not_multiple_of_8_rejected(self):
        for bad in (9, 15, 63, 100, 255):
            with self.assertRaises(ValueError, msg=str(bad)):
                tensor_core.wgmma_m64nNk16_f32(bad)

    def test_n_bool_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(True)

    def test_n_float_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(64.5)

    def test_n_string_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32("64")

    def test_tf32_operand_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(64, A_dtype=ll_type.tfloat32)

    def test_fp8_operand_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(64, B_dtype=ll_type.fp8_e4m3)

    def test_smem_flag_non_bool_rejected(self):
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(64, A_in_SMEM=1)
        with self.assertRaises(ValueError):
            tensor_core.wgmma_m64nNk16_f32(64, B_in_SMEM=0)


class TestWgmmaAccumulatorContract(unittest.TestCase):
    """acc_regs_per_thread must match the n//2 register count the emitter
    tests assert for every legal N (m64nNk16 f32 WGMMA, 128-thread
    warpgroup)."""

    def test_n8_smallest(self):
        m = tensor_core.wgmma_m64nNk16_f32(8)
        self.assertEqual(m.acc_regs_per_thread, 4)

    def test_n64_matches_emitter_reg_count(self):
        m = tensor_core.wgmma_m64nNk16_f32(64)
        self.assertEqual(m.acc_regs_per_thread, 32)
        body = _wgmma_emit.codegen_wgmma_mma_async_ss("acc", "da", "db", 64).body
        self.assertEqual(body.count('"+f"'), m.acc_regs_per_thread)

    def test_n256_largest(self):
        m = tensor_core.wgmma_m64nNk16_f32(256)
        self.assertEqual(m.acc_regs_per_thread, 128)

    def test_all_legal_n_agree_with_emitter(self):
        for n in tensor_core.WGMMA_F16_LEGAL_NCOLS:
            m = tensor_core.wgmma_m64nNk16_f32(n)
            body = _wgmma_emit.codegen_wgmma_mma_async_ss("acc", "da", "db", n).body
            self.assertEqual(m.acc_regs_per_thread, body.count('"+f"'), n)

    def test_non_f32_accumulator_rejects_reg_query(self):
        m = tensor_core.WgMMA(M=64, N=64, K=16, A_dtype=ll_type.bfloat16, B_dtype=ll_type.bfloat16,
                              ACC_dtype=ll_type.float16, A_in_SMEM=True, B_in_SMEM=True)
        with self.assertRaises(ValueError):
            _ = m.acc_regs_per_thread


class TestWgmmaFragmentAndSmemDerivations(unittest.TestCase):

    def test_fragment_totals(self):
        m = tensor_core.wgmma_m64nNk16_f32(64)
        self.assertEqual(m.acc_elems_total, 64 * 64)
        self.assertEqual(m.a_elems_total, 64 * 16)
        self.assertEqual(m.b_elems_total, 16 * 64)

    def test_smem_bytes_ss_bf16(self):
        # SS tile: both operands resident; 2 bytes per bf16 element.
        m = tensor_core.wgmma_m64nNk16_f32(64)
        expected = (64 * 16 + 16 * 64) * 2
        self.assertEqual(m.smem_bytes_per_k_step, expected)

    def test_smem_bytes_rs_only_b_resident(self):
        m = tensor_core.wgmma_m64nNk16_f32(64, A_in_SMEM=False)
        self.assertEqual(m.smem_bytes_per_k_step, 16 * 64 * 2)

    def test_smem_bytes_scales_with_n(self):
        m64 = tensor_core.wgmma_m64nNk16_f32(64)
        m128 = tensor_core.wgmma_m64nNk16_f32(128)
        # The B side doubles; the A side is fixed by M=64, K=16.
        delta = m128.smem_bytes_per_k_step - m64.smem_bytes_per_k_step
        self.assertEqual(delta, 16 * 64 * 2)

    def test_smem_bytes_f16_same_width_as_bf16(self):
        bf = tensor_core.wgmma_m64nNk16_f32(64)
        f16 = tensor_core.wgmma_m64nNk16_f32(64, A_dtype=ll_type.float16, B_dtype=ll_type.float16)
        self.assertEqual(bf.smem_bytes_per_k_step, f16.smem_bytes_per_k_step)


class TestLegacyFactoryPreserved(unittest.TestCase):

    def test_legacy_name_still_works(self):
        m = tensor_core.WgMMA_64_X_16_F32BF16BF16_SS(64)
        self.assertEqual((m.M, m.N, m.K), (64, 64, 16))
        self.assertIs(m.A_dtype, ll_type.bfloat16)
        self.assertIs(m.ACC_dtype, ll_type.float32)
        self.assertTrue(m.A_in_SMEM and m.B_in_SMEM)

    def test_legacy_full_range(self):
        for n in (8, 64, 128, 256):
            m = tensor_core.WgMMA_64_X_16_F32BF16BF16_SS(n)
            self.assertEqual(m.N, n)


if __name__ == "__main__":
    unittest.main()
