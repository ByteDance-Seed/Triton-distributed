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
"""CPU regression tests for atom.mma.loop_plan (coverage.mma_pipeline, the
composition axis over tile_plan x smem_plan).

``plan_pipelined_mma`` composes the accumulator/tiling plan
(``atom.mma.tile_plan``) with the SMEM pipeline ring (``atom.pipeline``),
deriving the cross-axis quantity a caller otherwise hand-derives: the
per-K-loop-step staged footprint of the WHOLE CTA operand slabs
(``cta_m * BK`` / ``cta_n * BK`` elements), as opposed to the per-instruction
``atom.a_elems_total`` / ``atom.b_elems_total`` that live at ``atom.K``
granularity. The worked numbers below are tied to the hand-derived constants
in ``benchmark/gemm_sm90/gemm_v5.py`` (BM=128, BN=128, BK=64, bf16, 2
warpgroups along M) and to the SM90 dynamic-SMEM budget the existing pipeline
tests pin, so the composition is checked against the same ground truth the
real kernel uses.
"""

import unittest

import little_kernel.core.type_system as ll_type
from little_kernel.backends.nv.sm90.atom.mma import tensor_core
from little_kernel.backends.nv.sm90.atom.mma import tile_plan
from little_kernel.backends.nv.sm90.atom.mma import loop_plan
from little_kernel.backends.nv.sm90.atom.mma.loop_plan import PipelinedMmaPlan, a_slab_bytes, b_slab_bytes, plan_pipelined_mma
from little_kernel.backends.nv.sm90.atom.pipeline import PIPELINE_ABI_ALIGNMENT_BYTES, plan_k_pipeline

ALIGN = PIPELINE_ABI_ALIGNMENT_BYTES
SM90_SMEM_BUDGET = 227 * 1024  # per-CTA dynamic SMEM carve-out used by tests

# gemm_v5's CTA tile (benchmark/gemm_sm90/gemm_v5.py BM/BN/BK).
GEMM_BM, GEMM_BN, GEMM_BK = 128, 128, 64


def _bf16_atom(n=GEMM_BN):
    return tensor_core.wgmma_m64nNk16_f32(n)  # bf16/bf16, SS


def _gemm_v5_plan(num_stages=2, budget=SM90_SMEM_BUDGET):
    # gemm_v5: BM=128 over 2 warpgroups (atom.M=64 each), BN=128 == atom.N,
    # BK=64 staging step, K extent 4*BK. 2-stage ring.
    return plan_pipelined_mma(_bf16_atom(), cta_m=GEMM_BM, cta_n=GEMM_BN, k_total=GEMM_BK * 4, loop_k=GEMM_BK,
                              smem_budget_bytes=budget, num_warpgroups_m=2, requested_stages=num_stages,
                              max_stages=num_stages)


class TestSlabByteDerivation(unittest.TestCase):
    """The composed A/B slab bytes match gemm_v5's hand-derived SMEM_A/B."""

    def test_a_slab_matches_gemm_v5_smem_a(self):
        # gemm_v5: SMEM_A = BM * BK * 2 (bf16)
        atom = _bf16_atom()
        self.assertEqual(a_slab_bytes(atom, GEMM_BM, GEMM_BK), GEMM_BM * GEMM_BK * 2)

    def test_b_slab_matches_gemm_v5_smem_b(self):
        # gemm_v5: SMEM_B = BN * BK * 2 (bf16)
        atom = _bf16_atom()
        self.assertEqual(b_slab_bytes(atom, GEMM_BN, GEMM_BK), GEMM_BN * GEMM_BK * 2)

    def test_slab_bytes_differ_from_per_instruction_bytes(self):
        # The cross-axis point: instruction-level operand bytes (atom.a/b
        # _elems_total) are at atom.K granularity, 4x/4x smaller for BK=64,
        # atom.K=16. The composed slab must NOT use them.
        atom = _bf16_atom()
        self.assertNotEqual(a_slab_bytes(atom, GEMM_BM, GEMM_BK), atom.a_elems_total * 2)
        self.assertNotEqual(b_slab_bytes(atom, GEMM_BN, GEMM_BK), atom.b_elems_total * 2)
        self.assertEqual(
            a_slab_bytes(atom, GEMM_BM, GEMM_BK) // (atom.a_elems_total * 2), (GEMM_BM * GEMM_BK) // (atom.M * atom.K))

    def test_slab_dtype_must_be_byte_sized(self):
        # a_slab_bytes/b_slab_bytes only read atom.{A,B}_dtype; use a duck-typed
        # stand-in carrying a non-byte-sized dtype so the frozen real atom is
        # never mutated.
        class _OddBits:
            bits = 4

        class _StandIn:
            A_dtype = _OddBits()
            B_dtype = _OddBits()

        with self.assertRaises(ValueError):
            a_slab_bytes(_StandIn(), 64, 16)
        with self.assertRaises(ValueError):
            b_slab_bytes(_StandIn(), 64, 16)

    def test_slab_dtype_must_have_int_bits(self):

        class _NoBits:
            pass

        class _StandIn:
            A_dtype = _NoBits()
            B_dtype = _NoBits()

        with self.assertRaises(ValueError):
            a_slab_bytes(_StandIn(), 64, 16)


class TestPlanLegality(unittest.TestCase):

    def test_non_atom_rejected(self):
        with self.assertRaises(ValueError):
            plan_pipelined_mma(object(), cta_m=64, cta_n=16, k_total=64, loop_k=16, smem_budget_bytes=SM90_SMEM_BUDGET)

    def test_loop_k_must_be_positive_int(self):
        for bad in (0, -16, 1.5, True, "64"):
            with self.assertRaises(ValueError, msg=bad):
                plan_pipelined_mma(_bf16_atom(), cta_m=64, cta_n=16, k_total=64, loop_k=bad,
                                   smem_budget_bytes=SM90_SMEM_BUDGET)

    def test_loop_k_must_be_multiple_of_atom_k(self):
        # atom.K=16; loop_k=32 is fine, loop_k=24 is not.
        plan_pipelined_mma(_bf16_atom(8), cta_m=64, cta_n=8, k_total=64, loop_k=32, smem_budget_bytes=SM90_SMEM_BUDGET)
        with self.assertRaises(ValueError):
            plan_pipelined_mma(_bf16_atom(8), cta_m=64, cta_n=8, k_total=64, loop_k=24,
                               smem_budget_bytes=SM90_SMEM_BUDGET)

    def test_k_total_must_be_multiple_of_loop_k(self):
        with self.assertRaises(ValueError):
            plan_pipelined_mma(_bf16_atom(8), cta_m=64, cta_n=8, k_total=72,  # not a multiple of 32
                               loop_k=32, smem_budget_bytes=SM90_SMEM_BUDGET)

    def test_inherits_tile_plan_legality(self):
        # cta_m must split exactly across warpgroups (2 warpgroups own 64 each
        # -> cta_m=128; cta_m=64 with 2 warpgroups is illegal).
        with self.assertRaises(ValueError):
            plan_pipelined_mma(_bf16_atom(), cta_m=64, cta_n=GEMM_BN, k_total=GEMM_BK * 4, loop_k=GEMM_BK,
                               smem_budget_bytes=SM90_SMEM_BUDGET, num_warpgroups_m=2)

    def test_inherits_pipeline_legality_when_one_stage_too_big(self):
        # A 64x64 bf16 tile's per-step ring is tiny; shrink the budget below
        # one aligned stage so even num_stages=1 cannot fit.
        with self.assertRaises(ValueError):
            plan_pipelined_mma(_bf16_atom(8), cta_m=64, cta_n=8, k_total=16, loop_k=16, smem_budget_bytes=ALIGN - 1)


class TestLoopDomainCoupling(unittest.TestCase):

    def test_num_k_loop_steps(self):
        p = _gemm_v5_plan()
        self.assertEqual(p.num_k_loop_steps, GEMM_BK * 4 // GEMM_BK)

    def test_loop_k_is_staging_granularity(self):
        # num_k_loop_steps counts BK steps, not atom.K instructions.
        p = _gemm_v5_plan()
        self.assertNotEqual(p.num_k_loop_steps, p.tile.num_k_steps)
        self.assertEqual(p.tile.num_k_steps, p.num_k_loop_steps * (GEMM_BK // p.tile.atom.K))


class TestRegisterBudgetCoupling(unittest.TestCase):

    def test_acc_regs_per_thread_matches_atom_audit(self):
        # one warpgroup holds num_n_blocks=1 atom of N=128 -> 128//2 = 64 regs.
        p = _gemm_v5_plan()
        self.assertEqual(p.acc_regs_per_thread_per_warpgroup, 128 // 2)
        self.assertEqual(p.acc_regs_per_thread_per_warpgroup, p.tile.acc_regs_per_warpgroup)

    def test_acc_bits_is_32bit_registers(self):
        p = _gemm_v5_plan()
        self.assertEqual(p.acc_bits_per_thread, p.acc_regs_per_thread_per_warpgroup * 32)

    def test_acc_regs_within_budget(self):
        p = _gemm_v5_plan()
        self.assertTrue(p.acc_regs_within(64))
        self.assertTrue(p.acc_regs_within(255))
        self.assertFalse(p.acc_regs_within(63))

    def test_acc_regs_within_rejects_illegal_budget(self):
        p = _gemm_v5_plan()
        for bad in (0, -1, 1.5, True, "64"):
            with self.assertRaises(ValueError, msg=bad):
                p.acc_regs_within(bad)


class TestRingCoupling(unittest.TestCase):

    def test_ring_depth_respects_budget_with_real_slab_bytes(self):
        # gemm_v5 (2-stage) is the ground truth: with the real CTA slab bytes
        # (16384 A + 16384 B = 32768 aligned per stage) the SM90 budget fits
        # 227KB // 32768 = 6 stages. The per-instruction undercount would have
        # reported far more.
        p = plan_pipelined_mma(_bf16_atom(), cta_m=GEMM_BM, cta_n=GEMM_BN, k_total=GEMM_BK * 4, loop_k=GEMM_BK,
                               smem_budget_bytes=SM90_SMEM_BUDGET, num_warpgroups_m=2, max_stages=64)
        self.assertEqual(p.pipeline.num_stages, SM90_SMEM_BUDGET // 32768)
        self.assertEqual(p.staged_total_bytes(), 32768 * p.pipeline.num_stages)

    def test_requested_stage_clamp_is_observable(self):
        p = _gemm_v5_plan(num_stages=2)
        self.assertEqual(p.pipeline.num_stages, 2)
        self.assertEqual(p.pipeline.requested_num_stages, 2)

    def test_slot_of_a_and_b_land_in_distinct_regions(self):
        p = _gemm_v5_plan(num_stages=2)
        for step in range(4):
            sa, sb = p.slot_of_a(step), p.slot_of_b(step)
            # same version slot, disjoint byte ranges.
            self.assertEqual(sa.version, sb.version)
            self.assertLessEqual(sa.end_offset, sb.offset)
            self.assertEqual(sa.bytes, GEMM_BM * GEMM_BK * 2)
            self.assertEqual(sb.bytes, GEMM_BN * GEMM_BK * 2)

    def test_iter_version_parity_wraps_with_ring(self):
        p = _gemm_v5_plan(num_stages=2)
        # steps 0,1 fill slots 0,1 at parity 0; steps 2,3 reuse slots at parity 1.
        self.assertEqual(p.iter_version_parity(0), (0, 0))
        self.assertEqual(p.iter_version_parity(1), (1, 0))
        self.assertEqual(p.iter_version_parity(2), (0, 1))
        self.assertEqual(p.iter_version_parity(3), (1, 1))

    def test_slot_rejects_negative_step(self):
        p = _gemm_v5_plan(num_stages=2)
        with self.assertRaises(ValueError):
            p.slot_of_a(-1)


class TestCompositionMatchesManualPlans(unittest.TestCase):
    """The composed plan must agree with each sub-plan built by hand."""

    def test_tile_subplan_matches_direct_plan_mma_tile(self):
        p = _gemm_v5_plan()
        direct = tile_plan.plan_mma_tile(_bf16_atom(), cta_m=GEMM_BM, cta_n=GEMM_BN, k_total=GEMM_BK * 4,
                                         num_warpgroups_m=2)
        self.assertEqual(p.tile, direct)

    def test_pipeline_subplan_matches_direct_plan_k_pipeline(self):
        p = _gemm_v5_plan(num_stages=2)
        direct = plan_k_pipeline([("A", GEMM_BM * GEMM_BK * 2), ("B", GEMM_BN * GEMM_BK * 2)],
                                 smem_budget_bytes=SM90_SMEM_BUDGET, max_stages=2, requested_stages=2)
        self.assertEqual(p.pipeline, direct)

    def test_composition_never_silently_uses_instruction_bytes(self):
        # Building a pipeline with instruction-level bytes yields a different
        # (over-deep) ring; the composed plan must NOT equal it.
        atom = _bf16_atom()
        wrong = plan_k_pipeline([("A", atom.a_elems_total * 2), ("B", atom.b_elems_total * 2)],
                                smem_budget_bytes=SM90_SMEM_BUDGET, max_stages=64)
        p = plan_pipelined_mma(atom, cta_m=GEMM_BM, cta_n=GEMM_BN, k_total=GEMM_BK * 4, loop_k=GEMM_BK,
                               smem_budget_bytes=SM90_SMEM_BUDGET, num_warpgroups_m=2, max_stages=64)
        self.assertNotEqual(p.pipeline.num_stages, wrong.num_stages)
        self.assertLess(p.pipeline.num_stages, wrong.num_stages)


class TestDataclassImmutability(unittest.TestCase):

    def test_plan_is_frozen(self):
        p = _gemm_v5_plan()
        with self.assertRaises(AttributeError):
            p.loop_k = 32


if __name__ == "__main__":
    unittest.main()
