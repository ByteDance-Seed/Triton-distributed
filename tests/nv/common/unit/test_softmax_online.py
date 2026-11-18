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
"""CPU semantics regression for the online-softmax intrinsics.

Covers the generic online reduction / row-normalization contract of
``backends/nv/common/language/intrin/softmax.py`` without a GPU:

* Structural: every emitted helper body must go through libdevice
  ``__nv_expf`` (immune to ``--use_fast_math`` remapping onto the
  ``MUFU.EX2``-backed ``__expf`` approximation), keep the caller-side
  max-subtraction, avoid fast reciprocal division in the epilogue, and the
  warp merges must be butterfly ``__shfl_xor_sync`` collectives whose
  per-step combine is the scalar merge -- plus a fast-math countercheck
  proving the pinned token regimes actually discriminate.
* Semantic: a Python lockstep model that mirrors the emitted C++ merge
  bodies statement for statement (``fmax`` tie + both rescale factors taken
  against the freshly combined max) is checked against a brute-force
  union reference over scattered per-lane partial states: streaming
  scalar folds in both orders, the full-warp butterfly allreduce, and the
  width-segmented variant inside power-of-two lane segments.  Empty
  partials seeded ``(-inf, 0)`` must vanish exactly; an all-``-inf`` union
  propagates ``nan`` per the module's no-mask policy.  A documented
  counter-model (rescaling stale denominators against the pre-merge max,
  the hand-roll misspelling this module exists to remove) must diverge.
* Typing: all six builtins are strict-float32 (FP32-accumulate contract):
  wrong dtypes fail at return-type inference; the merge family returns the
  ``float32`` 2-tuple; the segmented width must be a literal power of two
  in [1, 32]; merge effects are ``R_RMEM_WARP_SYNC`` on both components.

No CUDA toolchain or GPU is needed; codegen output is consumed as source
text via the BuiltinOp ``codegen`` staticmethods.
"""

import math
import sys
import unittest
from pathlib import Path

# Locate the repository's python/ package root regardless of the working
# directory, so the module is importable under bare unittest discovery.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "python" / "little_kernel").is_dir():
        if str(_parent / "python") not in sys.path:
            sys.path.insert(0, str(_parent / "python"))
        break

from little_kernel.backends.nv.common.language.intrin import softmax as sm
from little_kernel.core.type_system import TupleType, float16, float32, int32
from little_kernel.language import builtin_base as bb

LANES = 32
NEG_INF = float("-inf")

# Numeric battery: finite mixes with -inf (a masked lane / empty block) and
# large magnitude gaps that would overflow exp() without max subtraction.
# The empty seed (-inf, 0) is exercised separately by the seeding tests and
# kept out of this battery: a real (-inf, 0) partial never meets another
# (-inf, 0) inside a nonempty segment (seeded lanes are folded before
# traffic), while a wall of seeds IS the degenerate all-empty case with its
# own documented nan policy.
PARTIALS = [
    (1.5, 3.0),
    (2.0, 0.5),
    (-30.0, 60.0),
    (2.0, 7.0),  # max tie with different denominators
    (-1.0, 4.0),
]


def _codegen(fn, *args):
    return getattr(fn, bb.CODEGEN_FUNC_ATTR)(*args)


def _return_type(fn, *types):
    return getattr(fn, bb.EVAL_RETURN_TYPE_ATTR)(*types)


# --------------------------------------------------------------------------
# Python lockstep models.
#
# The scalar-merge model mirrors lk_softmax_merge verbatim: fmax the two
# maxima, rescale BOTH denominators against the just-combined m, then sum.
# The butterfly models mirror lk_warp_softmax_merge(_width): two shuffles
# and one scalar merge per step.
# --------------------------------------------------------------------------


def _fmax(a, b):
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    return a if a >= b else b


def _exp(x):
    # IEEE exp: exp(-inf) == 0, exp(nan) == nan; a Python-level OverflowError
    # never occurs because every caller range-reduces (argument <= 0) exactly
    # like the emitted helpers.
    if math.isnan(x):
        return float("nan")
    if x == NEG_INF:
        return 0.0
    return math.exp(x)


def _mul_sub(operand, factor):
    """l * exp(m_partial - m_union): the seed's 0 * exp(-inf - -inf) term.
    exp(nan) == nan by IEEE, but the emitted helpers compute
    l * __nv_expf(m - m_u) where the zero denominator multiplies the nan
    factor straight into nan -- kept explicit so the model reproduces the
    documented all-empty-row nan propagation instead of papering over it.
    """
    return operand * factor


def _merge(m_a, l_a, m_b, l_b):
    """Model of lk_softmax_merge (emitted algorithm)."""
    m = _fmax(m_a, m_b)
    l = _mul_sub(l_a, _exp(m_a - m)) + _mul_sub(l_b, _exp(m_b - m))
    return m, l


def _merge_stale_max(m_a, l_a, m_b, l_b):
    """Counter-model: rescale b against the PRE-merge a-max (hand-roll bug).
    l overflows whenever m_b > m_a by a wide enough gap."""
    return _fmax(m_a, m_b), l_a + l_b * _exp(m_b - m_a)


def _reference_union(partials):
    """Brute force: expand each partial to its elements, take global max."""
    m = NEG_INF
    for (mi, li) in partials:
        m = _fmax(m, mi)
    if m == NEG_INF:
        # Only possible when every partial was seeded (-inf, 0).
        return NEG_INF, 0.0
    total = 0.0
    for (mi, li) in partials:
        if math.isnan(li):
            return m, float("nan")
        if mi == NEG_INF and m != NEG_INF:
            continue  # exp(-inf - m) == 0; folded exactly, no nan
        total += li * _exp(mi - m)
    return m, total


def _butterfly(partials, width, merge):
    """Mirror of lk_warp_softmax_merge_width: xor butterfly inside width
    segments (width == LANES is the full-warp variant)."""
    state = list(partials)
    offset = width // 2
    while offset > 0:
        nxt = list(state)
        for lane in range(LANES):
            src = lane ^ offset
            if src // width != lane // width:
                continue  # 4-arg shuffle clamps inside the segment
            m_u, l_u = merge(state[lane][0], state[lane][1], state[src][0], state[src][1])
            nxt[lane] = (m_u, l_u)
        state = nxt
        offset >>= 1
    return state


def _assert_close_pair(tc, got, want, ctx):
    gm, gl = got
    wm, wl = want
    tc.assertEqual(gm, wm, ctx)
    if math.isnan(wl):
        tc.assertTrue(math.isnan(gl), ctx)
    else:
        tc.assertAlmostEqual(gl, wl, places=5, msg=ctx)


class TestSoftmaxEpilogueStructure(unittest.TestCase):
    """Structural checks on the emitted epilogue helper source."""

    def test_libdevice_expf_not_fast_approx(self):
        body = _codegen(sm.softmax_exp_neg, "lk_x").body
        self.assertIn("__nv_expf", body)
        self.assertNotIn("__expf(", body.replace("__nv_expf", ""))
        self.assertIn('extern "C" __device__ float __nv_expf(float);', body)

    def test_all_epilogues_share_one_guard_block(self):
        b1 = _codegen(sm.softmax_exp_neg, "x").body
        b2 = _codegen(sm.softmax_normalize, "x", "m", "s").body
        b3 = _codegen(sm.softmax_rescale, "old", "new").body
        self.assertEqual(b1, b2)
        self.assertEqual(b2, b3)
        self.assertIn("#ifndef _LK_SOFTMAX_EPILOGUE", b1)
        self.assertEqual(b1.count("_LK_SOFTMAX_EPILOGUE"), 3)  # ifndef+def+endif

    def test_stable_form_and_ieee_divide(self):
        norm = _codegen(sm.softmax_normalize, "x", "m", "s")
        self.assertIn("lk_softmax_normalize(x, m, s)", norm.return_val)
        self.assertIn("__nv_expf(x - rowmax)", norm.body)
        self.assertNotIn("__fdividef", norm.body)
        self.assertNotIn("__frcp", norm.body)
        resc = _codegen(sm.softmax_rescale, "old", "new")
        self.assertIn("__nv_expf(old_max - new_max)", resc.body)
        neg = _codegen(sm.softmax_exp_neg, "x")
        self.assertIn("__nv_expf(-x)", neg.body)

    def test_epilogue_no_memory_effect(self):
        for fn in (sm.softmax_exp_neg, sm.softmax_normalize, sm.softmax_rescale, sm.softmax_merge):
            # Pure value-to-value: the class declares no ``effects`` tuple,
            # so the builtin wrapper carries no MEMORY_EFFECTS_ATTR at all.
            effects = getattr(fn, bb.MEMORY_EFFECTS_ATTR, ())
            self.assertFalse(effects, fn.__name__)


class TestSoftmaxMergeStructure(unittest.TestCase):
    """Structural checks on the scalar + warp merge helper source."""

    def test_merge_block_guard_and_tuple(self):
        body = _codegen(sm.softmax_merge, "ma", "la", "mb", "lb").body
        self.assertIn("#ifndef _LK_SOFTMAX_MERGE", body)
        self.assertEqual(body.count("_LK_SOFTMAX_MERGE"), 3)
        self.assertIn("cuda::std::tuple<float, float>", body)
        self.assertIn("cuda::std::make_tuple(m, l)", body)

    def test_scalar_merge_both_factors_against_combined_max(self):
        body = _codegen(sm.softmax_merge, "ma", "la", "mb", "lb").body
        self.assertIn("float m = fmaxf(m_a, m_b);", body)
        self.assertIn("float l = l_a * __nv_expf(m_a - m) + l_b * __nv_expf(m_b - m);", body)

    def test_warp_butterfly_is_xor_shfl_with_per_step_merge(self):
        full = _codegen(sm.warp_softmax_merge, "mask", "m", "l").body
        self.assertIn("__shfl_xor_sync(mask, m, offset)", full)
        self.assertIn("__shfl_xor_sync(mask, l, offset)", full)
        self.assertIn("for (int offset = 16; offset > 0; offset >>= 1)", full)
        seg = _codegen(sm.warp_softmax_merge_width, "mask", "m", "l", "8").body
        self.assertIn("__shfl_xor_sync(mask, m, offset, W)", seg)
        self.assertIn("for (int offset = W / 2; offset > 0; offset >>= 1)", seg)

    def test_warp_merge_declares_warp_sync_read_on_both_components(self):
        for fn in (sm.warp_softmax_merge, sm.warp_softmax_merge_width):
            effects = getattr(fn, bb.MEMORY_EFFECTS_ATTR)
            self.assertEqual(len(effects), 2, fn.__name__)
            ordered = sorted(effects, key=lambda e: e.operand)
            self.assertEqual([e.operand for e in ordered], ["l", "m"])
            for e in effects:
                self.assertEqual(e.kind.issuer.name, "WARP")
                self.assertEqual(e.kind.space.name, "RMEM")
                self.assertEqual(e.kind.access.name, "R")
        self.assertFalse(getattr(sm.softmax_merge, bb.MEMORY_EFFECTS_ATTR, ()))


class TestSoftmaxTyping(unittest.TestCase):
    """Strict float32 contract and tuple return at inference time."""

    def test_epilogues_return_float32(self):
        self.assertIs(_return_type(sm.softmax_exp_neg, float32), float32)
        self.assertIs(_return_type(sm.softmax_normalize, float32, float32, float32), float32)
        self.assertIs(_return_type(sm.softmax_rescale, float32, float32), float32)

    def test_epilogues_reject_non_float32(self):
        for bad in (float16, int32):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    _return_type(sm.softmax_exp_neg, bad)
                with self.assertRaises(TypeError):
                    _return_type(sm.softmax_normalize, float32, bad, float32)
                with self.assertRaises(TypeError):
                    _return_type(sm.softmax_rescale, bad, float32)

    def test_merge_returns_float32_pair(self):
        rt = _return_type(sm.softmax_merge, float32, float32, float32, float32)
        self.assertIsInstance(rt, TupleType)
        self.assertEqual(tuple(rt), (float32, float32))
        rt = _return_type(sm.warp_softmax_merge, int32, float32, float32)
        self.assertEqual(tuple(rt), (float32, float32))

    def test_merge_rejects_non_float32_components(self):
        with self.assertRaises(TypeError):
            _return_type(sm.softmax_merge, float32, float16, float32, float32)
        with self.assertRaises(TypeError):
            _return_type(sm.warp_softmax_merge, int32, float32, int32)
        with self.assertRaises(TypeError):
            _return_type(sm.warp_softmax_merge_width, int32, float32, float16, int32)

    def test_segmented_width_must_be_literal_power_of_two(self):
        good = _codegen(sm.warp_softmax_merge_width, "mask", "m", "l", "8")
        self.assertIn("lk_warp_softmax_merge_width<8>", good.return_val)
        for bad in ("3", "0", "33", "-4", "1.5", "w"):
            with self.subTest(width=bad):
                with self.assertRaises((TypeError, ValueError)):
                    _codegen(sm.warp_softmax_merge_width, "mask", "m", "l", bad)


class TestSoftmaxScalarMergeSemantics(unittest.TestCase):
    """Emitted scalar-merge model vs brute-force union reference."""

    def test_pairwise_merge_matches_reference(self):
        for i, a in enumerate(PARTIALS):
            for j, b in enumerate(PARTIALS):
                with self.subTest(pair=(i, j)):
                    _assert_close_pair(self, _merge(*a, *b), _reference_union([a, b]), (a, b))

    def test_merge_is_commutative(self):
        for i, a in enumerate(PARTIALS):
            for j, b in enumerate(PARTIALS):
                with self.subTest(pair=(i, j)):
                    _assert_close_pair(self, _merge(*a, *b), _merge(*b, *a), (a, b))

    def test_streaming_fold_order_independent(self):
        # Left fold in both orders must agree with the 4-way reference;
        # order blindness is the whole point of the shared merge.
        stream = [(1.0, 2.0), (3.0, 0.25), (NEG_INF, 0.0), (-10.0, 8.0)]
        m, l = NEG_INF, 0.0
        for p in stream:
            m, l = _merge(m, l, p[0], p[1])
        left = (m, l)
        m, l = NEG_INF, 0.0
        for p in reversed(stream):
            m, l = _merge(m, l, p[0], p[1])
        ref = _reference_union(stream)
        _assert_close_pair(self, left, ref, "forward fold")
        _assert_close_pair(self, (m, l), ref, "reverse fold")

    def test_empty_seed_vanishes(self):
        got = _merge(NEG_INF, 0.0, 2.0, 5.0)
        _assert_close_pair(self, got, (2.0, 5.0), "seed must be identity")
        got = _merge(2.0, 5.0, NEG_INF, 0.0)
        _assert_close_pair(self, got, (2.0, 5.0), "seed must be identity")

    def test_all_empty_union_propagates_nan(self):
        # Module no-mask policy: every partial seeded (-inf, 0) yields
        # 0 * exp(-inf - -inf) == 0 * nan == nan in l, propagated so a
        # degenerate all-masked row surfaces instead of reading back 0.
        m, l = _merge(NEG_INF, 0.0, NEG_INF, 0.0)
        self.assertEqual(m, NEG_INF)
        self.assertTrue(math.isnan(l))

    def test_counter_model_overflows_on_missing_rescale(self):
        # The stale-max hand-roll bug this module exists to remove: on a
        # 30-point max gap it must NOT match the reference (exp(30) leak).
        a, b = (NEG_INF, 0.0), (30.0, 1.0)
        got = _merge_stale_max(*a, *b)
        ref = _reference_union([a, b])
        self.assertNotAlmostEqual(got[1], ref[1], msg="counter-model unexpectedly correct")
        a, b = (0.0, 1.0), (40.0, 1.0)
        got = _merge_stale_max(*a, *b)
        ref = _reference_union([a, b])
        self.assertGreater(got[1], ref[1] * 1e10)


class TestSoftmaxWarpMergeSemantics(unittest.TestCase):
    """Butterfly allreduce model vs brute-force union, per segment."""

    def _lane_partials(self, fill):
        # Deterministic interleave: every battery member appears, and lanes
        # adjacent within a segment see the hardest mixes (large magnitude
        # gaps, the max tie).
        return [fill[(i * 3 + i // 5) % len(fill)] for i in range(LANES)]

    def test_full_warp_allreduce_matches_reference(self):
        partials = self._lane_partials(PARTIALS)
        states = _butterfly(partials, LANES, _merge)
        ref = _reference_union(partials)
        for lane in range(LANES):
            _assert_close_pair(self, states[lane], ref, f"lane {lane}")

    def test_full_warp_all_lanes_bit_identical(self):
        # The allreduce contract: identical (m, l) on every participating
        # lane, or the rescale epilogue sees divergent denominators.
        partials = self._lane_partials(PARTIALS)
        states = _butterfly(partials, LANES, _merge)
        first = states[0]
        for lane in range(1, LANES):
            gm, gl = states[lane]
            self.assertEqual(gm, first[0], f"lane {lane} max")
            if math.isnan(first[1]):
                self.assertTrue(math.isnan(gl), f"lane {lane} l")
            else:
                self.assertEqual(gl, first[1], f"lane {lane} l")

    def test_segmented_variants_match_per_segment_reference(self):
        for width in (1, 2, 4, 8, 16, 32):
            with self.subTest(width=width):
                partials = self._lane_partials(PARTIALS)
                states = _butterfly(partials, width, _merge)
                for lane in range(LANES):
                    base = (lane // width) * width
                    ref = _reference_union(partials[base:base + width])
                    _assert_close_pair(self, states[lane], ref, f"width={width} lane={lane}")

    def test_segmented_counter_model_leaks_across_segments(self):
        # A merge that rescales against the pre-merge lane max leaks the
        # neighbor state's unrescaled denominator; it must disagree with
        # the per-segment reference on a scattered battery.
        partials = self._lane_partials(PARTIALS)
        bad = _butterfly(partials, 8, _merge_stale_max)
        ref_states = _butterfly(partials, 8, _merge)
        mismatches = sum(1 for lane in range(LANES)
                         if not math.isnan(bad[lane][1]) and abs(bad[lane][1] - ref_states[lane][1]) > 1e-6)
        self.assertGreater(mismatches, 0, "stale-max counter-model unexpectedly correct")

    def test_all_neg_inf_segment_propagates_nan(self):
        # Per the module no-mask policy: a segment where every partial is
        # (-inf, 0) ends with nan in l (exp(-inf - -inf)), never masked.
        partials = [(NEG_INF, 0.0)] * 8 + self._lane_partials(PARTIALS)[8:]
        states = _butterfly(partials, 8, _merge)
        for lane in range(8):
            self.assertEqual(states[lane][0], NEG_INF)
            self.assertTrue(math.isnan(states[lane][1]), f"lane {lane} l must be nan")
        for lane in range(8, LANES):
            self.assertFalse(math.isnan(states[lane][1]), f"lane {lane} contaminates from dead segment")


class TestSoftmaxEpilogueSemantics(unittest.TestCase):
    """Value-range reasoning on the epilogue model formulas."""

    def test_normalize_quotient_bounded(self):

        def model(x, rowmax, rowsum):
            return _exp(x - rowmax) / rowsum

        row = [1.5, -2.0, 0.25, -30.0]
        m = max(row)
        denom = sum(_exp(v - m) for v in row)
        total = 0.0
        for v in row:
            q = model(v, m, denom)
            self.assertGreater(q, 0.0)
            self.assertLessEqual(q, 1.0)
            total += q
        self.assertAlmostEqual(total, 1.0, places=7)

    def test_rescale_keeps_running_denominator(self):
        # Streaming update: block A tracked alone, then folded with block B
        # via rescale, must equal the one-shot union denominator -- the
        # identity flash-attention's online update relies on.
        a_vals = [1.0, 3.0]
        b_vals = [2.5, -1.0]
        m_a = max(a_vals)
        l_a = sum(_exp(v - m_a) for v in a_vals)
        m_b = max(b_vals)
        l_b = sum(_exp(v - m_b) for v in b_vals)
        m_new = max(m_a, m_b)
        rescale_a = _exp(m_a - m_new)  # softmax_rescale(m_a, m_new)
        l_stream = l_a * rescale_a + l_b * _exp(m_b - m_new)
        l_ref = sum(_exp(v - m_new) for v in a_vals + b_vals)
        self.assertAlmostEqual(l_stream, l_ref, places=7)

    def test_exp_neg_model_range(self):
        # softmax_exp_neg(x) computes exp(-x); the caller guarantees x = m - s >= 0,
        # so the result is in (0, 1] and cannot overflow.
        def model_exp_neg(x):
            return _exp(-x)

        self.assertEqual(model_exp_neg(0.0), 1.0)  # max element itself -> 1
        for x in (1e-3, 1.0, 20.0, 80.0):
            self.assertLessEqual(model_exp_neg(x), 1.0)  # range-reduced: no overflow
            self.assertGreater(model_exp_neg(x), 0.0)  # no underflow to zero for tested range


if __name__ == "__main__":
    unittest.main()
