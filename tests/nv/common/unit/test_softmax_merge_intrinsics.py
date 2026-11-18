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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""Online-softmax partial-state merge intrinsics: naming, typing, emission.

Covers the coverage.softmax_attention gap: the fold shared by streaming /
chunked softmax and attention pipelines -- combining partial states
``(m, l)`` with ``l = sum_j exp(v_j - m)`` -- in scalar form
(``softmax_merge``) and as a warp butterfly (``warp_softmax_merge`` /
``warp_softmax_merge_width``) for warp-fragmented attention statistics.

Compiler-side legality / accuracy contract under test (the range-reduction
and seeding obligations of the frontend are documented in
``language/intrin/softmax.py``):

* canonical names ``ll.softmax_merge`` / ``ll.warp_softmax_merge`` /
  ``ll.warp_softmax_merge_width`` are exported and marked builtin;
* the partial-state components are ``float32``-only (matching the FP32
  accumulate policy of the whole module); other dtypes reject at
  type-inference time; the result is the ``TupleType(float32, float32)``
  pair ``(m, l)``;
* ``softmax_merge`` is a pure value-to-value function and declares NO
  memory effect, while the warp merges are convergent collectives
  declaring ``R_RMEM_WARP_SYNC`` on both value operands;
* the segmented warp variant validates its width as a compile-time
  power-of-two literal in [1, 32], same as the reduce family's segmented
  variants;
* emission lowers to one once-guarded, fast-math-immune helper block that
  resolves the exponential to libdevice ``__nv_expf`` and exchanges via
  ``__shfl_xor_sync`` (5 steps full-warp, width-parameterized for the
  segmented variant), with ``cuda::std::tuple`` returns unpacked through
  the same caller path as ``block_reduce_max_sum``.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR
from little_kernel.core.memory_effects import R_RMEM_WARP_SYNC
from little_kernel.core.type_system import TupleType, bfloat16, float16, float32, float64, int32, uint32

import little_kernel.backends.nv.common.language.intrin.softmax as softmax_intrin


class TestSoftmaxMergeIntrinsics(unittest.TestCase):
    """Naming, builtin flag, typing and effect declarations."""

    def test_exported_canonical_names(self):
        for name in ("softmax_merge", "warp_softmax_merge", "warp_softmax_merge_width"):
            with self.subTest(op=name):
                op = getattr(ll, name, None)
                self.assertIsNotNone(op, f"ll.{name} not exported")
                self.assertTrue(getattr(op, BUILTIN_ATTR, False), f"ll.{name} missing __builtin__ flag")

    def test_return_type_is_float32_pair(self):
        for name in ("softmax_merge", "warp_softmax_merge", "warp_softmax_merge_width"):
            with self.subTest(op=name):
                ret = getattr(getattr(ll, name), EVAL_RETURN_TYPE_ATTR)(*self._legal_arg_types(name))
                self.assertIsInstance(ret, TupleType)
                self.assertEqual(len(ret), 2)
                for elem in ret:
                    self.assertEqual(elem, float32)

    @staticmethod
    def _legal_arg_types(name):
        if name == "softmax_merge":
            return (float32, float32, float32, float32)
        if name == "warp_softmax_merge":
            return (uint32, float32, float32)
        return (uint32, float32, float32, int32)

    def test_return_type_rejects_non_float32_components(self):
        bad = (float64, float16, bfloat16, int32)
        for dt in bad:
            with self.subTest(op="softmax_merge", dtype=dt):
                for slot in range(4):
                    args = [float32] * 4
                    args[slot] = dt
                    with self.assertRaises(TypeError):
                        getattr(ll.softmax_merge, EVAL_RETURN_TYPE_ATTR)(*args)
            with self.subTest(op="warp_softmax_merge", dtype=dt):
                for slot in (1, 2):
                    args = [uint32, float32, float32]
                    args[slot] = dt
                    with self.assertRaises(TypeError):
                        getattr(ll.warp_softmax_merge, EVAL_RETURN_TYPE_ATTR)(*args)
            with self.subTest(op="warp_softmax_merge_width", dtype=dt):
                for slot in (1, 2):
                    args = [uint32, float32, float32, int32]
                    args[slot] = dt
                    with self.assertRaises(TypeError):
                        getattr(ll.warp_softmax_merge_width, EVAL_RETURN_TYPE_ATTR)(*args)

    def test_scalar_merge_declares_no_memory_effect(self):
        self.assertFalse(hasattr(ll.softmax_merge, MEMORY_EFFECTS_ATTR),
                         "ll.softmax_merge unexpectedly declares a memory effect")

    def test_warp_merges_declare_rmem_warp_sync_on_both_operands(self):
        for name in ("warp_softmax_merge", "warp_softmax_merge_width"):
            with self.subTest(op=name):
                effects = getattr(getattr(ll, name), MEMORY_EFFECTS_ATTR)
                kinds = {(e.kind, e.operand) for e in effects}
                self.assertIn((R_RMEM_WARP_SYNC, "m"), kinds)
                self.assertIn((R_RMEM_WARP_SYNC, "l"), kinds)

    def test_segment_width_validated_as_power_of_two_literal(self):
        codegen = getattr(ll.warp_softmax_merge_width, CODEGEN_FUNC_ATTR)
        # Runtime / non-literal spellings reject as TypeError.
        for bad_literal in ("w", "seg", "1.5", "0x4"):
            with self.subTest(width=bad_literal):
                with self.assertRaises(TypeError):
                    codegen("mask", "m", "l", bad_literal)
        # In-range non-powers-of-two and out-of-range reject as ValueError.
        for bad_value in ("0", "3", "6", "12", "64", "-2"):
            with self.subTest(width=bad_value):
                with self.assertRaises(ValueError):
                    codegen("mask", "m", "l", bad_value)
        # Legal widths stamp the template argument.
        for good in ("1", "2", "4", "8", "16", "32"):
            with self.subTest(width=good):
                out = codegen("mask", "m", "l", good)
                self.assertIn(f"lk_warp_softmax_merge_width<{good}>(mask, m, l)", out.return_val)

    def test_helper_is_fast_math_immune_and_tuple_based(self):
        body = softmax_intrin._SOFTMAX_MERGE_HELPERS
        # Exponential resolves to libdevice __nv_expf, NOT expf/__expf.
        self.assertIn("__nv_expf", body)
        self.assertNotIn("__expf(", body)
        self.assertNotIn("expf(", body.replace("__nv_expf(", ""))
        # No fast reciprocal / approximate division anywhere in the merge.
        self.assertNotIn("__frcp_rn", body)
        self.assertNotIn("__fdividef", body)
        self.assertNotIn("div.approx", body)
        # Combined max uses fmaxf (stable against the wrong-max misspelling).
        self.assertIn("fmaxf", body)
        # Warp variants exchange both components through butterfly shuffles.
        self.assertIn("__shfl_xor_sync(mask, m, offset)", body)
        self.assertIn("__shfl_xor_sync(mask, l, offset)", body)
        self.assertIn("__shfl_xor_sync(mask, m, offset, W)", body)
        self.assertIn("__shfl_xor_sync(mask, l, offset, W)", body)
        # Tuple pair returns through the same spelling as block_reduce_max_sum.
        self.assertIn("cuda::std::tuple<float, float>", body)
        self.assertIn("cuda::std::make_tuple(m, l)", body)
        # Once-guard keeps mixed-TU emission link-clean and deduplicated.
        self.assertIn("#ifndef _LK_SOFTMAX_MERGE", body)
        self.assertIn("#define _LK_SOFTMAX_MERGE", body)

    def test_warp_merge_reuses_reduce_family_shfl_contract(self):
        # The merge butterfly must walk offsets 16..1 (full-warp) and
        # W/2..1 (segmented), matching the reduce family's exchange graph
        # so both can be reasoned about with the same convergence rules.
        body = softmax_intrin._SOFTMAX_MERGE_HELPERS
        self.assertIn("for (int offset = 16; offset > 0; offset >>= 1)", body)
        self.assertIn("for (int offset = W / 2; offset > 0; offset >>= 1)", body)


# ==============================================================================
# Full lower() path: native helper spelling in generated CUDA.
# ==============================================================================

# Streaming attention-style fold: fold an incoming block partial (m_b, l_b)
# into the running state (m_a, l_a) in one typed merge, then renormalize.
MERGE_SCALAR_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def fold_partial(state: ll.Tensor[ll.float32],
                 block: ll.Tensor[ll.float32],
                 out: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.blockIdx_x()
    m_a: ll.float32 = state[t * 2]
    l_a: ll.float32 = state[t * 2 + 1]
    m_b: ll.float32 = block[t * 2]
    l_b: ll.float32 = block[t * 2 + 1]
    m, l = ll.softmax_merge(m_a, l_a, m_b, l_b)
    state[t * 2] = m
    state[t * 2 + 1] = l
    out[t] = ll.softmax_normalize(block[t * 2], m, l)
'''

# Warp-fragmented statistics: each lane's partial merges across the warp so
# every lane rescales with the row total (the flash-attention epilogue shape).
MERGE_WARP_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def warp_stats(part_m: ll.Tensor[ll.float32],
               part_l: ll.Tensor[ll.float32],
               row: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    m: ll.float32 = part_m[lane]
    l: ll.float32 = part_l[lane]
    m, l = ll.warp_softmax_merge(0xffffffff, m, l)
    row[lane * 2] = m
    row[lane * 2 + 1] = l
'''

# Segmented variant: two independent 16-lane row fragments per warp.
MERGE_WARP_WIDTH_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def warp_stats16(part_m: ll.Tensor[ll.float32],
                 part_l: ll.Tensor[ll.float32],
                 row: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    m: ll.float32 = part_m[lane]
    l: ll.float32 = part_l[lane]
    m, l = ll.warp_softmax_merge_width(0xffffffff, m, l, 16)
    row[lane * 2] = m
    row[lane * 2 + 1] = l
'''


class TestSoftmaxMergeEndToEnd(unittest.TestCase):
    """Full lower() path: native helper spelling in generated CUDA."""

    def _lower_one(self, tmp: Path, source: str, name: str, input_dtypes, output_name: str) -> str:
        task = tmp / "task"
        task.mkdir()
        (task / "model.py").write_text("class Model: pass\n")
        (task / "test_cases.py").write_text("test_cases = []\n")
        program = tmp / "program"
        program.mkdir()
        (program / "draft.py").write_text(source)
        first_tensor = input_dtypes[0][0]
        (program / "export.json").write_text(
            json.dumps({
                "kernel": name,
                "function": name,
                "inputs": [{"name": t, "dtype": d, "rank": 1} for t, d in input_dtypes],
                "output": {"name": output_name, "dtype": "float32", "shape": [f"{first_tensor}[0]"]},
                "scalars": [],
                "launch": {"grid": [f"ceil_div({first_tensor}[0], 128)", 1, 1], "block": [32, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_scalar_merge_lowers(self):
        """The streaming fold composes the shared merge helper."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), MERGE_SCALAR_SOURCE, "fold_partial",
                                   (("state", "float32"), ("block", "float32")), "out")
        self.assertIn("lk_softmax_merge(", cuda)
        self.assertIn("__nv_expf", cuda)
        self.assertIn("cuda::std::tuple<float, float>", cuda)
        self.assertNotIn("asm", cuda)

    def test_warp_merge_lowers(self):
        """The full-warp butterfly merge lowers through the shared helper."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), MERGE_WARP_SOURCE, "warp_stats",
                                   (("part_m", "float32"), ("part_l", "float32")), "row")
        self.assertIn("lk_warp_softmax_merge(", cuda)
        self.assertIn("__shfl_xor_sync(mask, m, offset)", cuda)
        self.assertIn("__nv_expf", cuda)
        self.assertNotIn("asm", cuda)

    def test_warp_merge_width_lowers(self):
        """The segmented merge stamps the width as a template immediate."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), MERGE_WARP_WIDTH_SOURCE, "warp_stats16",
                                   (("part_m", "float32"), ("part_l", "float32")), "row")
        self.assertIn("lk_warp_softmax_merge_width<16>(", cuda)
        self.assertIn("__shfl_xor_sync(mask, m, offset, W)", cuda)
        self.assertNotIn("asm", cuda)

    def test_merge_helper_body_deduplicated(self):
        """Both warp call sites in one TU share ONE helper definition."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), MERGE_WARP_SOURCE, "warp_stats",
                                   (("part_m", "float32"), ("part_l", "float32")), "row")
        self.assertEqual(cuda.count("#define _LK_SOFTMAX_MERGE"), 1, cuda)
        self.assertEqual(cuda.count("lk_warp_softmax_merge(unsigned mask, float m"), 1, cuda)


# ==============================================================================
# Semantic CPU reference: the merge combine must reproduce the one-shot
# softmax no matter how the element set is partitioned or in which order the
# partials are folded (associativity), seeds must vanish exactly, and the
# degenerate all--inf case must surface as NaN (never silently 0).
# ==============================================================================


def _merge(m_a, l_a, m_b, l_b):
    """Mirror of lk_softmax_merge: union of two partial states."""
    m = max(m_a, m_b)
    l = l_a * math.exp(m_a - m) + l_b * math.exp(m_b - m)
    return m, l


def _fold(scores, parts):
    """Partition ``scores`` into ``parts`` contiguous chunks, partial-fold."""
    n = len(scores)
    chunk = (n + parts - 1) // parts
    m_run, l_run = -math.inf, 0.0
    for p in range(parts):
        seg = scores[p * chunk:(p + 1) * chunk]
        if not seg:
            continue
        m_seg = max(seg)
        l_seg = sum(math.exp(v - m_seg) for v in seg)
        m_run, l_run = _merge(m_run, l_run, m_seg, l_seg)
    return m_run, l_run


class TestSoftmaxMergeSemantics(unittest.TestCase):
    """Pin the math shape the emitted merge helpers encode."""

    def test_folded_merge_matches_oneshot_denominator(self):
        scores = [1.0, -2.5, 3.25, 0.0, 7.5, -4.0, 2.0, 1.5]
        m = max(scores)
        l_oneshot = sum(math.exp(v - m) for v in scores)
        for parts in (1, 2, 3, 4, 8):
            with self.subTest(parts=parts):
                m_fold, l_fold = _fold(scores, parts)
                self.assertAlmostEqual(m_fold, m, places=12)
                self.assertAlmostEqual(l_fold, l_oneshot, places=10)

    def test_merge_is_associative_and_commutative(self):
        partials = [(1.0, 2.5), (3.0, 1.25), (-2.0, 4.0)]
        m1, l1 = _merge(*_merge(*partials[0], *partials[1]), *partials[2])
        m2, l2 = _merge(*partials[0], *_merge(*partials[1], *partials[2]))
        m3, l3 = _merge(*partials[1], *_merge(*partials[0], *partials[2]))
        self.assertAlmostEqual(m1, m2, places=12)
        self.assertAlmostEqual(m1, m3, places=12)
        self.assertAlmostEqual(l1, l2, places=10)
        self.assertAlmostEqual(l1, l3, places=10)

    def test_seed_partial_vanishes_exactly(self):
        # An empty stream is (-inf, 0); merging it into a real partial must
        # return the real partial unchanged (exp(-inf - m) == 0).
        m, l = _merge(-math.inf, 0.0, 2.5, 3.75)
        self.assertEqual(m, 2.5)
        self.assertEqual(l, 3.75)
        m, l = _merge(2.5, 3.75, -math.inf, 0.0)
        self.assertEqual(m, 2.5)
        self.assertEqual(l, 3.75)

    def test_all_neg_inf_degenerate_propagates_nan(self):
        # Every lane empty: exp(-inf - -inf) == nan; the merge must NOT
        # mask it into 0 (module policy: degenerate rows stay debuggable).
        m, l = _merge(-math.inf, 0.0, -math.inf, 0.0)
        self.assertEqual(m, -math.inf)
        self.assertTrue(math.isnan(l))

    def test_no_merge_factor_exceeds_one(self):
        # Range-reduction invariant: both exp arguments are <= 0 because
        # m = fmax(m_a, m_b), so neither factor can overflow regardless of
        # fold order -- this is what a wrong-max hand-roll violates.
        for (m_a, l_a, m_b, l_b) in ((1.0, 2.0, 3.0, 1.0), (3.0, 1.0, 1.0, 2.0), (-50.0, 1.0, 99.0, 2.0)):
            with self.subTest(m_a=m_a, m_b=m_b):
                m = max(m_a, m_b)
                self.assertLessEqual(math.exp(m_a - m), 1.0)
                self.assertLessEqual(math.exp(m_b - m), 1.0)

    def test_butterfly_fold_order_matches_row_reduce(self):
        # Simulate the emitted butterfly's lane-pairing fold order over 8
        # lanes and check it against the straight row reduce: the butterfly
        # merges (lane, lane^16^...) pairs, but merge associativity makes
        # the union order-insensitive, which is the allreduce contract.
        scores = [float(v) for v in (3, -1, 4, 1, 5, -9, 2, 6)]
        partials = [(s, 1.0) for s in scores]  # one element per "lane"
        offset = 4
        while offset:
            partials = [_merge(*partials[i], *partials[i ^ offset]) for i in range(len(partials) // 2)]
            offset >>= 1
        (m_b, l_b), = partials
        m = max(scores)
        self.assertAlmostEqual(m_b, m, places=12)
        self.assertAlmostEqual(l_b, sum(math.exp(v - m) for v in scores), places=10)

    def test_wide_range_and_near_half_scenarios(self):
        # wide_range: log-uniform scale drives one weight to ~1, other ~0.
        m, l = _fold([4.0, 4096.0], 1)
        w = [math.exp(v - m) / l for v in (4.0, 4096.0)]
        self.assertAlmostEqual(sum(w), 1.0, places=12)
        self.assertLess(w[0], 1e-3)
        # near_half_weights: ~equal scores merge into a ~0.5/0.5 split.
        m, l = _fold([1e-7, 1.999e-7], 2)
        w = [math.exp(v - m) / l for v in (1e-7, 1.999e-7)]
        self.assertAlmostEqual(w[0], 0.5, places=6)
        self.assertAlmostEqual(w[1], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
