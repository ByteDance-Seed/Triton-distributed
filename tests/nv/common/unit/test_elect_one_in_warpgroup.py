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
"""CPU semantics regression for the warpgroup-level elect intrinsic.

Covers the contract of
``backends/nv/common/language/intrin/simt.py:elect_one_in_warpgroup``
without a GPU:

* Structural: the emitted helper must carry the election and the warp gate
  atomically -- one ``elect.sync`` with the full-warp mask 0xFFFFFFFF, the
  in-CTA warp-index read ``threadIdx.x >> 5``, and the conjunction
  ``warp_idx == warp_index`` gating the elected predicate, so threads of
  non-selected warps deterministically receive 0.  The builtin must type
  the result as uint32 and pass the warp-index argument through to the
  helper call.
* Semantic: a Python lockstep model that mirrors the emitted C++ statement
  for statement is checked against the intrinsic contract for a
  384-thread, three-warpgroup block (warps 0..11): for every selected warp
  index, exactly one elected lane of exactly the selected warp observes 1,
  all other 383 threads observe 0, and the non-elected decomposition
  (elect within the wrong warp, or gating without election) is kept as a
  documented counter-model.  A generative battery over the elected-lane
  position pins down "exactly one" rather than "lane 0".
* Frontend: the reproducer-shaped kernel (a warpgroup role gate feeding a
  store) lowered through the cuda backend must reference the fused helper
  call in its emitted source, so the typed IR composability gap (frontend
  materializing ``(threadIdx.x >> 5) == wg`` separately from ``elect_one``)
  stays closed.

No CUDA toolchain or GPU is needed; codegen output is consumed as source
text via the BuiltinOp ``codegen`` staticmethods and the source-only
``compile`` path.
"""

import unittest

from little_kernel.backends.nv.common.language.intrin import simt
from little_kernel.language import builtin_base as bb

import little_kernel as lk
import little_kernel.language as ll

WARPS_PER_CTA = 12  # 384-thread FA3-style block: three 128-thread warpgroups
LANES = 32


def _codegen(warp_index):
    fn = simt.elect_one_in_warpgroup
    codegen = getattr(fn, bb.CODEGEN_FUNC_ATTR)
    return codegen(warp_index)


# --------------------------------------------------------------------------
# Python lockstep models of the emitted helper body.
#
# elect.sync _|p, 0xFFFFFFFF is modeled per PTX: within the convergent full
# warp, p is true on exactly one lane (the "elected" lane; hardware picks
# the lowest active lane, but the contract only requires uniqueness, so the
# model is parameterized over the elected lane).
# --------------------------------------------------------------------------


def _model_emitted(tid, warp_index, elected_lane):
    """Mirror of the emitted C++ helper for a thread with id ``tid``."""
    warp_idx = tid >> 5
    lane = tid & 31
    pred = 1 if lane == elected_lane else 0  # elect.sync, full mask
    return pred if warp_idx == warp_index else 0


def _model_ungated(tid, warp_index, elected_lane):
    """Counter-model: elect without the warp gate (the composability gap)."""
    lane = tid & 31
    return 1 if lane == elected_lane else 0


class ElectOneInWarpgroupStructuralTest(unittest.TestCase):

    def test_body_fuses_elect_and_warp_gate(self):
        body = _codegen("8").body
        self.assertIn("elect.sync _|p, 0xFFFFFFFF", body)
        self.assertIn("threadIdx.x >> 5", body)
        self.assertIn("warp_idx == warp_index", body)
        # The elected predicate must be gated, not returned raw.
        self.assertIn("? pred : 0u", body)

    def test_return_val_forwards_warp_index(self):
        self.assertEqual(_codegen("8").return_val, "elect_one_in_warpgroup_fn(8)")
        self.assertEqual(_codegen("wg_expr + 1").return_val, "elect_one_in_warpgroup_fn(wg_expr + 1)")

    def test_eval_return_type_is_uint32(self):
        rt = getattr(simt.elect_one_in_warpgroup, bb.EVAL_RETURN_TYPE_ATTR)
        from little_kernel.core.type_system import uint32
        self.assertIs(rt(), uint32)

    def test_no_side_effect_builtins(self):
        # No store/barrier builtins may be emitted inline in the helper.
        body = _codegen("0").body
        for forbidden in ("st.global", "bar.sync", "mbarrier", "__syncthreads"):
            self.assertNotIn(forbidden, body)


class ElectOneInWarpgroupSemanticTest(unittest.TestCase):

    def test_exactly_one_thread_per_selected_warp(self):
        for warp_index in range(WARPS_PER_CTA):
            for elected_lane in (0, 1, 17, 31):
                winners = [
                    tid for tid in range(WARPS_PER_CTA * LANES) if _model_emitted(tid, warp_index, elected_lane) == 1
                ]
                self.assertEqual(len(winners), 1, f"warp {warp_index} elected lane {elected_lane}")
                winner = winners[0]
                self.assertEqual(winner >> 5, warp_index)
                self.assertEqual(winner & 31, elected_lane)

    def test_unselected_warps_observe_zero(self):
        for warp_index in range(WARPS_PER_CTA):
            for tid in range(WARPS_PER_CTA * LANES):
                if (tid >> 5) != warp_index:
                    self.assertEqual(_model_emitted(tid, warp_index, 0), 0)

    def test_ungated_elect_is_a_counter_model(self):
        # Without the warp gate, every warp of the CTA reports a winner:
        # 12 winners instead of 1.  This is precisely why the fused op
        # exists -- the frontend composed this form awkwardly by hand.
        for elected_lane in (0, 7, 31):
            winners = [tid for tid in range(WARPS_PER_CTA * LANES) if _model_ungated(tid, 8, elected_lane) == 1]
            self.assertEqual(len(winners), WARPS_PER_CTA)


@lk.ll_kernel(backend="cuda", is_entry=True)
def _wg_elect_single_kernel(flag: ll.Tensor[ll.int32]) -> ll.void:
    e: ll.uint32 = ll.elect_one_in_warpgroup(8)
    if e == 1:
        flag[ll.blockIdx_x() * 128 + ll.threadIdx_x()] = 1


@lk.ll_kernel(backend="cuda", is_entry=True)
def _wg_elect_pair_kernel(flag: ll.Tensor[ll.int32]) -> ll.void:
    a: ll.uint32 = ll.elect_one_in_warpgroup(8)
    b: ll.uint32 = ll.elect_one_in_warpgroup(9)
    if a == 1:
        flag[ll.threadIdx_x()] = 1
    if b == 1:
        flag[ll.threadIdx_x() + 1] = 1


class ElectOneInWarpgroupFrontendTest(unittest.TestCase):
    """End-to-end source-only lowering of the fused intrinsic."""

    def _lower(self, kern):
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        return kern.compile(PASSES["cuda"], codegen_cuda, num_threads=384)

    def test_fused_call_reaches_emitted_source(self):
        source = self._lower(_wg_elect_single_kernel)
        self.assertIn("elect_one_in_warpgroup_fn", source)

    def test_helper_definition_is_emitted_once(self):
        source = self._lower(_wg_elect_pair_kernel)
        # Deduplicated by the builtin-body cache.
        self.assertEqual(source.count("elect_one_in_warpgroup_fn(uint32_t warp_index)"), 1)
        self.assertIn("elect_one_in_warpgroup_fn(8)", source)
        self.assertIn("elect_one_in_warpgroup_fn(9)", source)


if __name__ == "__main__":
    unittest.main()
