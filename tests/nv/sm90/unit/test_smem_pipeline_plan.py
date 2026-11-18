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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""CPU regression tests for the SMEM pipeline stage planner
(coverage.mma_pipeline).

atom.pipeline.smem_plan answers the shared-memory pipeline questions that
neither the emitter-side cp.async/TMA/WGMMA intrinsics nor the single-shape
atom.mma.tensor_core audit can answer: stage depth under a byte budget,
version-multiplied buffer carve-out, floormod ring versions, expanded
barrier slots and phase parity, and the prologue/steady/epilogue partition
of the iteration domain. These are the invariants pinned for upstream
pipeline planners (cw.passes.tilelang-software-pipeline), expressed as pure
backend-agnostic arithmetic so they are CPU-testable with no GPU.

The buffer sizes in the MMA-composition tests below are derived from
atom.mma.tensor_core (bf16 WGMMA m64nNk16 K-step SMEM occupancy), tying the
pipeline axis to the instruction axis the existing audit already pins.
"""

import unittest

import little_kernel.core.type_system as ll_type
from little_kernel.backends.nv.sm90.atom.mma import tensor_core
from little_kernel.backends.nv.sm90.atom.pipeline import PIPELINE_ABI_ALIGNMENT_BYTES, PipelinePlan, SmemBufferSpec, plan_k_pipeline

ALIGN = PIPELINE_ABI_ALIGNMENT_BYTES


def _bf16_wgmma_buffers(n):
    """Staged SMEM buffers for one SS bf16 WGMMA K-step (A tile + B tile)."""
    m = tensor_core.wgmma_m64nNk16_f32(n)  # default: bf16/bf16, SS
    return [
        SmemBufferSpec("A", m.a_elems_total * 2),
        SmemBufferSpec("B", m.b_elems_total * 2),
    ]


class TestBufferSpecValidation(unittest.TestCase):

    def test_bytes_per_step_must_be_positive_int(self):
        with self.assertRaises(ValueError):
            SmemBufferSpec("A", 0)
        with self.assertRaises(ValueError):
            SmemBufferSpec("A", -4)

    def test_bytes_per_step_rejects_non_int(self):
        with self.assertRaises(ValueError):
            SmemBufferSpec("A", 1.5)
        with self.assertRaises(ValueError):
            SmemBufferSpec("A", True)
        with self.assertRaises(ValueError):
            SmemBufferSpec("A", "2048")

    def test_name_must_be_nonempty_str(self):
        with self.assertRaises(ValueError):
            SmemBufferSpec("", 16)
        with self.assertRaises(ValueError):
            SmemBufferSpec(None, 16)

    def test_aligned_bytes_rounds_up(self):
        self.assertEqual(SmemBufferSpec("A", 1).aligned_bytes_per_step, ALIGN)
        self.assertEqual(SmemBufferSpec("A", ALIGN).aligned_bytes_per_step, ALIGN)
        self.assertEqual(SmemBufferSpec("A", ALIGN + 1).aligned_bytes_per_step, 2 * ALIGN)

    def test_spec_is_frozen(self):
        s = SmemBufferSpec("A", 16)
        with self.assertRaises(AttributeError):
            s.bytes_per_step = 32  # frozen dataclass


class TestPlanLegality(unittest.TestCase):

    def test_empty_buffers_rejected(self):
        with self.assertRaises(ValueError):
            plan_k_pipeline([], smem_budget_bytes=1 << 16, max_stages=4)

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            plan_k_pipeline([("A", 16), ("A", 32)], smem_budget_bytes=1 << 16, max_stages=4)

    def test_budget_must_be_nonnegative_int(self):
        with self.assertRaises(ValueError):
            plan_k_pipeline([("A", 16)], smem_budget_bytes=-1, max_stages=4)
        with self.assertRaises(ValueError):
            plan_k_pipeline([("A", 16)], smem_budget_bytes=1.5, max_stages=4)

    def test_max_stages_must_be_positive(self):
        with self.assertRaises(ValueError):
            plan_k_pipeline([("A", 16)], smem_budget_bytes=1 << 16, max_stages=0)

    def test_bool_stages_rejected(self):
        with self.assertRaises(ValueError):
            plan_k_pipeline([("A", 16)], smem_budget_bytes=1 << 16, max_stages=True)

    def test_single_stage_over_budget_is_hard_error(self):
        # One aligned stage = ALIGN bytes; a smaller budget can never fit
        # even the as-is path, so the planner must not clip to fit.
        with self.assertRaises(ValueError):
            plan_k_pipeline([("A", 16)], smem_budget_bytes=ALIGN - 1, max_stages=4)

    def test_fill_stages_above_num_stages_rejected(self):
        p = plan_k_pipeline([("A", 16)], smem_budget_bytes=1 << 16, max_stages=3)
        with self.assertRaises(ValueError):
            p.steady_iters(8, fill_stages=p.num_stages + 1)


class TestStageDepthUnderBudget(unittest.TestCase):

    def test_budget_clamps_depth_and_is_observable(self):
        # per-stage aligned = 2*ALIGN = 2048; budget fits exactly 3 stages.
        p = plan_k_pipeline([("A", ALIGN), ("B", ALIGN)], smem_budget_bytes=3 * 2 * ALIGN, max_stages=8)
        self.assertEqual(p.num_stages, 3)
        self.assertEqual(p.requested_num_stages, 8)  # not silent
        self.assertEqual(p.total_smem_bytes, 3 * 2 * ALIGN)

    def test_requested_below_budget_fit(self):
        p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=64 * ALIGN, max_stages=8, requested_stages=3)
        self.assertEqual(p.num_stages, 3)
        self.assertEqual(p.requested_num_stages, 3)
        self.assertEqual(p.total_smem_bytes, 3 * ALIGN)

    def test_max_stages_is_hard_ceiling(self):
        p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=1 << 20, max_stages=2)
        self.assertEqual(p.num_stages, 2)

    def test_unaligned_steps_still_fit_by_aligned_footprint(self):
        # Two small buffers pack into one aligned slot each per stage.
        p = plan_k_pipeline([("A", 100), ("B", 100)], smem_budget_bytes=2 * 2 * ALIGN, max_stages=9)
        self.assertEqual(p.num_stages, 2)
        self.assertEqual(p.stage_bytes, 200)
        self.assertEqual(p.total_smem_bytes, 2 * 2 * ALIGN)


class TestRingLayout(unittest.TestCase):

    def setUp(self):
        # A: 2 aligned slots worth per step is intentionally irregular to
        # make offset arithmetic nontrivial.
        self.p = plan_k_pipeline([("Q", 3 * ALIGN // 2), ("R", ALIGN // 2)], smem_budget_bytes=64 * ALIGN, max_stages=4)

    def test_num_stages(self):
        self.assertEqual(self.p.num_stages, 4)
        # Per aligned stage: Q -> 2*ALIGN, R -> 1*ALIGN => 3*ALIGN; ring
        # depth 4 => 12*ALIGN total.
        self.assertEqual(self.p.total_smem_bytes, 12 * ALIGN)

    def test_versions_floormod_ring(self):
        self.assertEqual([self.p.version_of(i) for i in range(10)], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_buffer_base_offsets_and_stride(self):
        q = self.p.stage_plans_for("Q")
        r = self.p.stage_plans_for("R")
        self.assertEqual([s.offset for s in q], [0, 2 * ALIGN, 4 * ALIGN, 6 * ALIGN])
        # R starts after the whole Q ring (4 * 2*ALIGN).
        self.assertEqual([s.offset for s in r], [8 * ALIGN, 9 * ALIGN, 10 * ALIGN, 11 * ALIGN])

    def test_slots_do_not_overlap_and_cover_region(self):
        spans = []
        for name in ("Q", "R"):
            spans.extend((s.offset, s.offset + s.bytes) for s in self.p.stage_plans_for(name))
        spans.sort()
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0, (a0, a1, b0, b1))
        self.assertEqual(spans[-1][1] <= self.p.total_smem_bytes, True)

    def test_slot_of_matches_version(self):
        i = 7
        self.assertEqual(self.p.slot_of("R", i).version, i % 4)
        self.assertEqual(self.p.slot_of("R", i).offset, self.p.stage_plans_for("R")[i % 4].offset)

    def test_unknown_buffer_rejected(self):
        with self.assertRaises(ValueError):
            self.p.stage_plans_for("nope")

    def test_negative_iteration_rejected(self):
        with self.assertRaises(ValueError):
            self.p.version_of(-1)


class TestBarrierSlotsAndParity(unittest.TestCase):

    def setUp(self):
        self.p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=4 * ALIGN, max_stages=4, barrier_slots_per_stage=2)

    def test_slot_expansion_formula(self):
        # version * slots_per_stage + slot over the full expanded array.
        self.assertEqual(self.p.barrier_slot(0, 0), 0)
        self.assertEqual(self.p.barrier_slot(0, 1), 1)
        self.assertEqual(self.p.barrier_slot(1, 0), 2)
        self.assertEqual(self.p.barrier_slot(3, 1), 7)

    def test_expanded_slot_count(self):
        total = self.p.num_stages * self.p.barrier_slots_per_stage
        seen = {
            self.p.barrier_slot(v, s)
            for v in range(self.p.num_stages)
            for s in range(self.p.barrier_slots_per_stage)
        }
        self.assertEqual(seen, set(range(total)))

    def test_slot_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            self.p.barrier_slot(4, 0)
        with self.assertRaises(ValueError):
            self.p.barrier_slot(0, 2)

    def test_parity_flips_once_per_ring_wrap(self):
        parities = [self.p.barrier_parity(i) for i in range(12)]
        self.assertEqual(parities, [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])

    def test_iter_version_parity_pairs(self):
        p = self.p
        for i in range(9):
            v, ph = p.iter_version_parity(i)
            self.assertEqual(v, i % 4, i)
            self.assertEqual(ph, (i // 4) % 2, i)

    def test_single_stage_ring_arithmetic(self):
        # num_stages == 1 is the as-is path: codegen takes the synchronous
        # route with no barrier expansion, but the ring arithmetic is still
        # well-defined — the single version never moves, and the upstream
        # parity formula (i // num_stages) % 2 continues to apply in case a
        # backend still guards slot reuse with one mbarrier phase.
        p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=ALIGN, max_stages=1)
        self.assertEqual(p.num_stages, 1)
        self.assertEqual([p.version_of(i) for i in range(5)], [0] * 5)
        self.assertEqual([p.barrier_parity(i) for i in range(5)], [(i // 1) % 2 for i in range(5)])
        # Still exactly one slot in the (unexpanded) barrier group.
        self.assertEqual({p.barrier_slot(v, s) for v in range(1) for s in range(1)}, {0})


class TestIterationPartition(unittest.TestCase):
    """prologue/steady/epilogue must tile [0, trip+depth) without overlap or
    omission for every trip count, including the boundary shapes."""

    def setUp(self):
        self.p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=3 * ALIGN, max_stages=3)

    def _check_partition(self, trip, fill=None):
        p = self.p
        pro = p.prologue_iters(trip, fill)
        st = p.steady_iters(trip, fill)
        ep = p.epilogue_iters(trip, fill)
        covered = set(range(*pro)) | set(range(*st)) | set(range(*ep))
        depth = p.num_stages - 1 if fill is None else fill
        expected = set(range(trip + depth))
        self.assertEqual(covered, expected, (trip, pro, st, ep))
        # No overlap: sizes add up.
        self.assertEqual(sum(e - s for s, e in (pro, st, ep)), trip + depth, (trip, pro, st, ep))

    def test_boundary_trip_counts(self):
        for trip in (0, 1, 2, 3, 4, 5, 17):
            self._check_partition(trip)

    def test_trip_below_prologue_depth(self):
        # num_stages=3 => default prologue depth 2; trip=1 < depth.
        self.assertEqual(self.p.prologue_iters(1), (0, 1))
        self.assertEqual(self.p.steady_iters(1), (1, 1))
        self.assertEqual(self.p.epilogue_iters(1), (1, 3))

    def test_zero_trip(self):
        self.assertEqual(self.p.prologue_iters(0), (0, 0))
        self.assertEqual(self.p.steady_iters(0), (0, 0))
        self.assertEqual(self.p.epilogue_iters(0), (0, 2))

    def test_explicit_fill_stages(self):
        for fill in (0, 1, 2, 3):
            self._check_partition(7, fill)
        p = self.p
        self.assertEqual(p.prologue_iters(7, fill_stages=0), (0, 0))
        self.assertEqual(p.steady_iters(7, fill_stages=0), (0, 7))
        self.assertEqual(p.epilogue_iters(7, fill_stages=0), (7, 7))

    def test_single_stage_path_collapses_prologue_epilogue(self):
        p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=ALIGN, max_stages=1)
        self.assertEqual(p.prologue_iters(5), (0, 0))
        self.assertEqual(p.steady_iters(5), (0, 5))
        self.assertEqual(p.epilogue_iters(5), (5, 5))

    def test_negative_trip_rejected(self):
        with self.assertRaises(ValueError):
            self.p.steady_iters(-1)


class TestMmaComposition(unittest.TestCase):
    """The pipeline axis must compose with the instruction-axis audit:
    per-K-step SMEM occupancy from atom.mma.tensor_core feeds the planner,
    and the resulting depth is what a Hopper-class 227KB carve-out allows."""

    SM90_SMEM_BUDGET = 227 * 1024  # max dynamic SMEM per CTA on sm_90

    def test_bf16_n128_k_loop_depth(self):
        m = tensor_core.wgmma_m64nNk16_f32(128)
        self.assertEqual(m.smem_bytes_per_k_step, (64 * 16 + 16 * 128) * 2)
        buffers = _bf16_wgmma_buffers(128)
        p = plan_k_pipeline(buffers, smem_budget_bytes=self.SM90_SMEM_BUDGET, max_stages=8)
        # Per aligned stage: A 2048B, B 4096B (both already multiples of
        # ALIGN) => 6144B. 227KB fits 37 stages; max_stages=8 clamps.
        self.assertEqual(p.num_stages, 8)
        self.assertEqual(p.total_smem_bytes, 6144 * 8)
        self.assertLessEqual(p.total_smem_bytes, self.SM90_SMEM_BUDGET)

    def test_bf16_n256_k_loop_depth_budget_bound(self):
        buffers = _bf16_wgmma_buffers(256)
        # Per stage: A 2048B + B 8192B = 10240B; 227KB // 10240 = 22.
        p = plan_k_pipeline(buffers, smem_budget_bytes=self.SM90_SMEM_BUDGET, max_stages=64)
        self.assertEqual(p.num_stages, self.SM90_SMEM_BUDGET // 10240)
        self.assertLessEqual(p.total_smem_bytes, self.SM90_SMEM_BUDGET)

    def test_widening_n_shifts_bytes_to_b_ring(self):
        p64 = plan_k_pipeline(_bf16_wgmma_buffers(64), smem_budget_bytes=self.SM90_SMEM_BUDGET, max_stages=99)
        p128 = plan_k_pipeline(_bf16_wgmma_buffers(128), smem_budget_bytes=self.SM90_SMEM_BUDGET, max_stages=99)
        # A tile is fixed (2048B); B doubles from 4096B to 8192B aligned.
        q64, r64 = p64.stage_plans_for("A"), p64.stage_plans_for("B")
        q128, r128 = p128.stage_plans_for("A"), p128.stage_plans_for("B")
        self.assertEqual(q64[1].offset - q64[0].offset, q128[1].offset - q128[0].offset)
        self.assertEqual((r128[1].offset - r128[0].offset), 2 * (r64[1].offset - r64[0].offset))

    def test_plan_matches_mma_legal_n_whitelist(self):
        # Every legal WGMMA N feeds a legal pipeline buffer pair; the two
        # audit surfaces never disagree on what is constructible.
        for n in tensor_core.WGMMA_F16_LEGAL_NCOLS:
            m = tensor_core.wgmma_m64nNk16_f32(n)
            p = plan_k_pipeline([
                ("A", m.a_elems_total * (m.A_dtype.bits // 8)),
                ("B", m.b_elems_total * (m.B_dtype.bits // 8)),
            ], smem_budget_bytes=self.SM90_SMEM_BUDGET, max_stages=4)
            self.assertEqual(p.num_stages, 4, n)
            self.assertLessEqual(p.total_smem_bytes, self.SM90_SMEM_BUDGET, n)


class TestDataclassImmutability(unittest.TestCase):

    def test_plan_is_frozen(self):
        p = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=2 * ALIGN, max_stages=2)
        with self.assertRaises(AttributeError):
            p.num_stages = 7

    def test_buffers_exposed_as_tuple(self):
        p = plan_k_pipeline([("A", ALIGN), ["B", ALIGN]], smem_budget_bytes=4 * ALIGN, max_stages=2)
        self.assertIsInstance(p.buffers, tuple)
        self.assertEqual([b.name for b in p.buffers], ["A", "B"])

    def test_pair_and_spec_forms_are_equivalent(self):
        a = plan_k_pipeline([SmemBufferSpec("A", ALIGN)], smem_budget_bytes=2 * ALIGN, max_stages=2)
        b = plan_k_pipeline([("A", ALIGN)], smem_budget_bytes=2 * ALIGN, max_stages=2)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
