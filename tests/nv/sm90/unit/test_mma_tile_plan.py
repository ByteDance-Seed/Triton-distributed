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
"""CPU regression tests for atom.mma.tile_plan (coverage.mma_pipeline,
accumulator-tiling axis).

The tile plan answers the multi-atom accumulator questions a pipelined CTA
kernel asks before emitting PTX: how many atoms cover the CTA tile, how many
accumulator registers per thread the whole fragment set needs, which register
range / output-tile slice one atom owns, and the per-K-step ``scale_d``
zero-init/accumulate discipline. Every derived number is checked against the
PTX contract the wgmma emitter already pins (``N//2`` f32 accumulator
registers per thread for m64nNk16, full N whitelist) and against the real
kernel patterns (``benchmark/gemm_sm90/gemm_v5.py``'s hand-written
``acc_{wg_m}{n_blk}`` fragments).
"""

import unittest

import little_kernel.core.type_system as ll_type
from little_kernel.backends.nv.sm90.atom.mma import tensor_core
from little_kernel.backends.nv.sm90.atom.mma import tile_plan
import little_kernel.backends.nv.sm90.language.intrin.wgmma as _wgmma_emit


def _bf16_atom(n):
    return tensor_core.wgmma_m64nNk16_f32(n)


class TestPlanExports(unittest.TestCase):

    def test_ll_mma_namespace_exports(self):
        import little_kernel.backends.nv.sm90.atom.mma as mma
        self.assertIs(mma.tile_plan, tile_plan)
        self.assertIs(mma.MmaTilePlan, tile_plan.MmaTilePlan)
        self.assertIs(mma.plan_mma_tile, tile_plan.plan_mma_tile)

    def test_all_lists_new_symbols(self):
        import little_kernel.backends.nv.sm90.atom.mma as mma
        self.assertIn("tile_plan", mma.__all__)
        self.assertIn("MmaTilePlan", mma.__all__)
        self.assertIn("plan_mma_tile", mma.__all__)


class TestPlanLegality(unittest.TestCase):
    """Illegal tilings raise ValueError before any derived quantity."""

    def test_non_atom_rejected(self):
        with self.assertRaises(ValueError):
            tile_plan.plan_mma_tile(object(), 64, 64, 64)

    def test_non_positive_dims_rejected(self):
        atom = _bf16_atom(64)
        for bad in (0, -1):
            with self.assertRaises(ValueError, msg=f"cta_m={bad}"):
                tile_plan.plan_mma_tile(atom, bad, 64, 64)
            with self.assertRaises(ValueError, msg=f"cta_n={bad}"):
                tile_plan.plan_mma_tile(atom, 64, bad, 64)
            with self.assertRaises(ValueError, msg=f"k_total={bad}"):
                tile_plan.plan_mma_tile(atom, 64, 64, bad)

    def test_bool_dim_rejected(self):
        atom = _bf16_atom(64)
        with self.assertRaises(ValueError):
            tile_plan.plan_mma_tile(atom, True, 64, 64)

    def test_partial_m_tile_rejected(self):
        atom = _bf16_atom(64)  # atom.M == 64
        with self.assertRaises(ValueError):
            tile_plan.plan_mma_tile(atom, 96, 64, 64)

    def test_m_warpgroup_mismatch_rejected(self):
        atom = _bf16_atom(64)
        # cta_m=128 needs 2 warpgroups along M; asking for 1 is illegal.
        with self.assertRaises(ValueError):
            tile_plan.plan_mma_tile(atom, 128, 64, 64, num_warpgroups_m=1)

    def test_partial_n_tile_rejected(self):
        atom = _bf16_atom(64)  # atom.N == 64
        with self.assertRaises(ValueError):
            tile_plan.plan_mma_tile(atom, 64, 96, 64)

    def test_k_not_multiple_of_atom_k_rejected(self):
        atom = _bf16_atom(64)  # atom.K == 16
        with self.assertRaises(ValueError):
            tile_plan.plan_mma_tile(atom, 64, 64, 24)


class TestTilingDerivations(unittest.TestCase):

    def test_single_atom(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        self.assertEqual(p.num_warpgroups_m, 1)
        self.assertEqual(p.num_n_blocks, 1)
        self.assertEqual(p.num_atoms_total, 1)
        self.assertEqual(p.num_k_steps, 64 // 16)
        self.assertEqual(p.atoms_per_warpgroup, 1)
        self.assertEqual(p.num_issues_total, 1 * (64 // 16))

    def test_multi_n_block(self):
        # cta_n 128 with atom.N 64 -> 2 column blocks per warpgroup.
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 128, 32)
        self.assertEqual(p.num_n_blocks, 2)
        self.assertEqual(p.num_atoms_total, 2)
        self.assertEqual(p.atoms_per_warpgroup, 2)
        self.assertEqual(p.num_k_steps, 32 // 16)
        self.assertEqual(p.num_issues_total, 2 * 2)

    def test_multi_warpgroup_m(self):
        # cta_m 128 across 2 warpgroups (each owns atom.M=64 rows).
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 128, 128, 64, num_warpgroups_m=2)
        self.assertEqual(p.num_warpgroups_m, 2)
        self.assertEqual(p.cta_m_per_warpgroup, 64)
        self.assertEqual(p.num_n_blocks, 2)
        self.assertEqual(p.num_atoms_total, 2 * 2)
        self.assertEqual(p.num_issues_total, 4 * 4)

    def test_matches_gemm_v5_hand_tiling(self):
        # benchmark/gemm_sm90/gemm_v5.py hand-declares acc_00..acc_11 for a
        # CTA tile tiled by 64x64x16 WGMMA atoms: 2 warpgroups along M, 2
        # column blocks along N.
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 128, 128, 64, num_warpgroups_m=2)
        names = tuple(f"acc_{wg}{nb}" for wg in range(p.num_warpgroups_m) for nb in range(p.num_n_blocks))
        self.assertEqual(names, ("acc_00", "acc_01", "acc_10", "acc_11"))


class TestAccumulatorDerivations(unittest.TestCase):

    def test_acc_regs_per_atom_matches_atom_contract(self):
        for n in (8, 64, 128, 256):
            p = tile_plan.plan_mma_tile(_bf16_atom(n), 64, n, 16)
            self.assertEqual(p.acc_regs_per_atom, n // 2, n)

    def test_acc_regs_per_atom_matches_emitter_count(self):
        # The emitter's inline asm declares N//2 "+f" outputs; the plan must
        # agree for every legal N.
        for n in tensor_core.WGMMA_F16_LEGAL_NCOLS:
            p = tile_plan.plan_mma_tile(_bf16_atom(n), 64, n, 16)
            body = _wgmma_emit.codegen_wgmma_mma_async_ss("acc", "da", "db", n).body
            self.assertEqual(p.acc_regs_per_atom, body.count('"+f"'), n)

    def test_acc_regs_per_warpgroup_scales_with_n_blocks(self):
        p1 = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 16)
        p2 = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 128, 16)
        self.assertEqual(p1.acc_regs_per_warpgroup, 32)
        self.assertEqual(p2.acc_regs_per_warpgroup, 64)
        self.assertEqual(p2.acc_regs_per_warpgroup, 2 * p1.acc_regs_per_warpgroup)

    def test_acc_elems_total_covers_cta_tile(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 128, 128, 64, num_warpgroups_m=2)
        self.assertEqual(p.acc_elems_total, 128 * 128)

    def test_atom_index_row_major(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 128, 128, 64, num_warpgroups_m=2)
        self.assertEqual(p.atom_index(0, 0), 0)
        self.assertEqual(p.atom_index(0, 1), 1)
        self.assertEqual(p.atom_index(1, 0), 2)
        self.assertEqual(p.atom_index(1, 1), 3)

    def test_atom_index_out_of_range_rejected(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        with self.assertRaises(ValueError):
            p.atom_index(1, 0)  # only one warpgroup
        with self.assertRaises(ValueError):
            p.atom_index(0, 1)  # only one N block
        with self.assertRaises(ValueError):
            p.atom_index(-1, 0)

    def test_acc_reg_range_n_block_major(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 128, 16)
        self.assertEqual(p.acc_reg_range(0, 0), (0, 32))
        self.assertEqual(p.acc_reg_range(0, 1), (32, 64))

    def test_acc_reg_range_disjoint_covering(self):
        # Ranges across the N blocks must exactly partition
        # [0, acc_regs_per_warpgroup).
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 256, 16)
        covered = []
        for nb in range(p.num_n_blocks):
            lo, hi = p.acc_reg_range(0, nb)
            self.assertEqual(hi - lo, p.acc_regs_per_atom)
            covered.extend(range(lo, hi))
        self.assertEqual(covered, list(range(p.acc_regs_per_warpgroup)))

    def test_acc_tile_slice_places_atom_in_cta(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 128, 128, 64, num_warpgroups_m=2)
        self.assertEqual(p.acc_tile_slice(0, 0), ((0, 64), (0, 64)))
        self.assertEqual(p.acc_tile_slice(0, 1), ((0, 64), (64, 128)))
        self.assertEqual(p.acc_tile_slice(1, 0), ((64, 128), (0, 64)))
        self.assertEqual(p.acc_tile_slice(1, 1), ((64, 128), (64, 128)))

    def test_acc_tile_slices_tile_the_cta(self):
        # The per-atom slices must tile the CTA output with no overlap.
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 128, 256, 64, num_warpgroups_m=2)
        cells = set()
        for wg in range(p.num_warpgroups_m):
            for nb in range(p.num_n_blocks):
                (r0, r1), (c0, c1) = p.acc_tile_slice(wg, nb)
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        cells.add((r, c))
        self.assertEqual(len(cells), 128 * 256)


class TestKLoopDerivations(unittest.TestCase):

    def test_k_step_range(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        self.assertEqual(list(p.k_step_range()), [0, 16, 32, 48])
        self.assertEqual(len(p.k_step_range()), p.num_k_steps)

    def test_k_step_index_on_boundaries(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        self.assertEqual(p.k_step_index(0), 0)
        self.assertEqual(p.k_step_index(16), 1)
        self.assertEqual(p.k_step_index(48), 3)

    def test_k_step_index_mid_atom_rejected(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        with self.assertRaises(ValueError):
            p.k_step_index(8)
        with self.assertRaises(ValueError):
            p.k_step_index(17)

    def test_k_step_index_out_of_domain_rejected(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        with self.assertRaises(ValueError):
            p.k_step_index(64)  # one past the last boundary
        with self.assertRaises(ValueError):
            p.k_step_index(-16)

    def test_scale_d_zero_then_one(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        self.assertEqual([p.scale_d_for(k) for k in range(p.num_k_steps)], [0, 1, 1, 1])

    def test_scale_d_out_of_range_rejected(self):
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 64)
        with self.assertRaises(ValueError):
            p.scale_d_for(p.num_k_steps)
        with self.assertRaises(ValueError):
            p.scale_d_for(-1)

    def test_scale_d_matches_emitter_runtime_flag(self):
        # The emitter's runtime scale_d path composes setp.ne.b32 p, %N, 0
        # and passes scale_d as an "r" operand; the plan's 0/1 discipline is
        # exactly the predicate value that toggles it.
        class _RuntimeInt:
            pass

        body = _wgmma_emit.codegen_wgmma_mma_async_ss("acc", "da", "db", 64, scale_d=_RuntimeInt()).body
        self.assertIn('setp.ne.b32 p, %34, 0;', body)
        p = tile_plan.plan_mma_tile(_bf16_atom(64), 64, 64, 16)
        self.assertEqual(p.scale_d_for(0), 0)  # overwritten, not accumulated


class TestF32AccumulatorContract(unittest.TestCase):
    """Non-f32 accumulator atoms surface the atom's own ValueError."""

    def test_f16_accumulator_rejects_reg_query(self):
        atom = tensor_core.WgMMA(M=64, N=64, K=16, A_dtype=ll_type.bfloat16, B_dtype=ll_type.bfloat16,
                                 ACC_dtype=ll_type.float16, A_in_SMEM=True, B_in_SMEM=True)
        p = tile_plan.plan_mma_tile(atom, 64, 64, 64)
        with self.assertRaises(ValueError):
            _ = p.acc_regs_per_atom


if __name__ == "__main__":
    unittest.main()
