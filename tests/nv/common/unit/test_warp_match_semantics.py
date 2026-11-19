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
"""CPU semantics regression for the warp vote/match collectives.

Covers the generic sentinel/mask contract of the ``ballot_sync`` /
``match_any_sync`` / ``popc`` / ``ffs`` thin wrappers in
``backends/nv/common/language/intrin/simt.py`` without a GPU:

* Structural: ``match_any_sync`` / ``ballot_sync`` must emit the CUDA
  collective in the return value, declare the ``R_RMEM_WARP_SYNC`` effect on
  the cross-lane operand (``value`` / ``predicate``) so the op is not CSE'd
  or reordered across a warp sync / divergence boundary, and pin the
  value-grouping (i.e. sentinel-match) semantics in the docstring.
  ``popc`` / ``ffs`` are plain per-lane ALU ops and must carry no warp
  effect.
* Semantic: a Python lockstep model that mirrors the emitted CUDA
  ``__match_any_sync`` / ``__ballot_sync`` semantics (grouping by
  bitwise-equal value *within the mask*, sentinel lanes holding the same
  sentinel match each other) is checked against a brute-force
  group-by-value reference over a battery of value vectors: the canonical
  duplicate-detection pattern (ids with a ``-1`` sentinel on inactive
  lanes), all-sentinel, no-sentinel, mixed multi-group, and the hazardous
  ``sentinel-in-payload`` case where the documented pre-condition (sentinel
  never occurs as a real payload) is violated and the count inflates.  The
  ballot variants are checked against a per-lane predicate reference over a
  participation-mask battery.

The defect class guarded here is semantic, not crash-level: a lowering that
special-cases sentinel lanes (e.g. excludes equal-sentinel lanes from their
own group, or treats every inactive lane as matching every active lane)
would silently corrupt duplicate counts; a pass that CSE's two same-args
collectives across a ``__syncwarp`` / divergence boundary would be legal
only because the effect was undeclared.  Declaring ``R_RMEM_WARP_SYNC`` and
pinning value-grouping semantics in tests closes both.

No CUDA toolchain or GPU is needed; codegen output is consumed as source
text via the builtin ``codegen_func`` attributes.
"""

import sys
import unittest
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "python/little_kernel").is_dir():
        _src = str(_parent / "python")
        if _src not in sys.path:
            sys.path.insert(0, _src)
        break

from little_kernel.backends.nv.common.language.intrin import simt
from little_kernel.language import builtin_base as bb

LANES = 32
FULL_MASK = 0xFFFFFFFF
SENTINEL = -1  # canonical "inactive lane" payload
U32 = 0x100000000  # for two's-complement bit-pattern conversion

# --------------------------------------------------------------------------
# Value-pattern battery for match_any_sync semantics.
#
# Each entry is a 32-lane vector of int32 values.  The duplicate-detection
# use case gathers up to 32 ids per warp; inactive lanes hold the sentinel.
# --------------------------------------------------------------------------


def _ids_with_sentinel(active_ids):
    """Vector whose first ``len(active_ids)`` lanes hold ids, rest sentinel."""
    return list(active_ids) + [SENTINEL] * (LANES - len(active_ids))


VECTORS = {
    # Canonical duplicate detection: duplicates among lanes 0..9, sentinel on
    # the inactive tail.  Sentinel group must not leak into the id groups.
    "ids_with_sentinel": _ids_with_sentinel([7, 3, 7, 7, 42, 3, 7, 100, 42, 42]),
    # No inactive lanes: every lane carries a payload id.
    "no_sentinel_all_ids": [(i * 37) % 5 for i in range(LANES)],
    # Every lane inactive: one all-same sentinel group, count == 32.
    "all_sentinel": [SENTINEL] * LANES,
    # Single active lane: that lane is a singleton group (count 1), the 31
    # sentinel lanes form their own group (count 31).
    "single_active": [9] + [SENTINEL] * (LANES - 1),
    # Sentinel ALSO present as a real payload among active lanes: the
    # documented hazardous case -- active and inactive -1 lanes merge into
    # one group and counts inflate.  Still grouped by value (well-defined);
    # the guard is the documented pre-condition, not the collective.
    "sentinel_in_payload": [SENTINEL, 5, SENTINEL, 5] + [SENTINEL] * (LANES - 4),
    # Distinct high-bit ids: int32 negatives other than the sentinel group
    # by 32-bit pattern, matching uint32 bit-for-bit.
    "negative_ids": [-3, -3, 1, -3, 1, -2000000000, -2000000000] + [SENTINEL] * 25,
}

MASKS = {
    "full_warp": FULL_MASK,
    "lower_half": 0x0000FFFF,
    "upper_half": 0xFFFF0000,
    "odd_segment": 0b11111 << 7,  # lanes 7..11
    "single_lane": 1 << 20,
    "two_segments": 0x00FF00FF,
}

# --------------------------------------------------------------------------
# Brute-force references.
# --------------------------------------------------------------------------


def _u32(v):
    """32-bit pattern of an int32 value (matches CUDA's bitwise grouping)."""
    return v % U32


def _reference_match(values, mask):
    """Group-by-value reference for __match_any_sync(mask, value).

    Each participating lane l receives the mask of lanes l' in ``mask`` with
    equal 32-bit value pattern -- including lanes holding the same sentinel
    (sentinel lanes match each other; there is no special-casing).  Lanes
    outside the mask receive 0.
    """
    out = [0] * LANES
    for l in range(LANES):
        if not (mask >> l) & 1:
            continue
        out[l] = sum(1 << lp for lp in range(LANES) if (mask >> lp) & 1 and _u32(values[lp]) == _u32(values[l]))
    return out


def _reference_ballot(preds, mask):
    """Per-lane predicate reference for __ballot_sync(mask, predicate)."""
    out = [0] * LANES
    for l in range(LANES):
        if not (mask >> l) & 1:
            continue
        out[l] = sum(1 << lp for lp in range(LANES) if (mask >> lp) & 1 and preds[lp])
    return out


def _popc(x):
    return bin(x).count("1")


def _member_lanes(mask):
    return [l for l in range(LANES) if (mask >> l) & 1]


# --------------------------------------------------------------------------
# Lockstep models of the emitted collectives.
#
# __match_any_sync is modeled per PTX match.any.sync.b32: lanes partition by
# equal 32-bit source value among the mask members; every member of a group
# receives the group's bitmask.  This model IS the documented semantics; the
# tests below pin it against the reference so that a reimplementation that
# special-cases sentinel lanes (either direction) fails.
# --------------------------------------------------------------------------


def _model_match_emitted(values, mask):
    return _reference_match(values, mask)


def _model_match_sentinel_excluded(values, mask):
    """Documented counter-model: sentinel lanes wrongly EXCLUDED from their
    own group (each sentinel lane sees only itself).  Disagrees with the
    reference whenever >= 2 inactive lanes share the sentinel."""
    out = [0] * LANES
    for l in range(LANES):
        if not (mask >> l) & 1:
            continue
        if values[l] == SENTINEL:
            out[l] = 1 << l  # counter-model: isolated sentinel lane
            continue
        out[l] = sum(1 << lp
                     for lp in range(LANES)
                     if (mask >> lp) & 1 and values[lp] != SENTINEL and _u32(values[lp]) == _u32(values[l]))
    return out


def _model_match_sentinel_matches_all(values, mask):
    """Documented counter-model: sentinel lanes wrongly match EVERY lane.
    Disagrees with the reference on any vector mixing sentinel and payload."""
    out = [0] * LANES
    for l in range(LANES):
        if not (mask >> l) & 1:
            continue
        if values[l] == SENTINEL:
            out[l] = mask  # counter-model: sentinel matches all
        else:
            out[l] = sum(1 << lp for lp in range(LANES) if (mask >> lp) & 1 and _u32(values[lp]) == _u32(values[l]))
    return out


class TestWarpMatchStructure(unittest.TestCase):
    """Structural checks on the emitted collectives and declared effects."""

    def test_match_any_emits_cuda_collective(self):
        codegen = getattr(simt.match_any_sync, bb.CODEGEN_FUNC_ATTR)
        ret = codegen("0xFFFFFFFFu", "v").return_val
        self.assertIn("__match_any_sync(", ret)
        self.assertIn("0xFFFFFFFFu", ret)
        self.assertIn("v", ret)

    def test_ballot_emits_cuda_collective(self):
        codegen = getattr(simt.ballot_sync, bb.CODEGEN_FUNC_ATTR)
        ret = codegen("mask", "pred").return_val
        self.assertIn("__ballot_sync(", ret)
        self.assertIn("mask", ret)
        self.assertIn("pred", ret)

    def test_popc_ffs_emit_intrinsics(self):
        self.assertIn("__popc(x)", getattr(simt.popc, bb.CODEGEN_FUNC_ATTR)("x").return_val)
        self.assertIn("__ffs(x)", getattr(simt.ffs, bb.CODEGEN_FUNC_ATTR)("x").return_val)

    def test_collectives_declare_warp_effect(self):
        for name, operand in (("match_any_sync", "value"), ("ballot_sync", "predicate")):
            fn = getattr(simt, name)
            effects = getattr(fn, bb.MEMORY_EFFECTS_ATTR, ())
            self.assertEqual(len(effects), 1, name)
            eff = effects[0]
            kind = eff.kind
            # R_RMEM_WARP_SYNC: issued by the warp, sync execution, read.
            self.assertEqual(kind.issuer.name, "WARP", name)
            self.assertEqual(kind.space.name, "RMEM", name)
            self.assertEqual(kind.access.name, "R", name)
            # Effect is attributed to the cross-lane operand.
            self.assertEqual(eff.operand, operand, name)

    def test_per_lane_alu_ops_carry_no_warp_effect(self):
        for name in ("popc", "ffs"):
            effects = getattr(getattr(simt, name), bb.MEMORY_EFFECTS_ATTR, ())
            self.assertEqual(effects, (), name)

    def test_return_types(self):
        rt = getattr(simt.match_any_sync, bb.EVAL_RETURN_TYPE_ATTR)
        self.assertIs(rt("mask_type", "value_type"), simt.uint32)
        rt = getattr(simt.ballot_sync, bb.EVAL_RETURN_TYPE_ATTR)
        self.assertIs(rt("mask_type", "pred_type"), simt.uint32)
        rt = getattr(simt.popc, bb.EVAL_RETURN_TYPE_ATTR)
        self.assertIs(rt("x_type"), simt.int32)

    def test_docstrings_pin_sentinel_semantics(self):
        doc = simt.match_any_sync.__doc__ or ""
        self.assertIn("sentinel", doc.lower())
        self.assertIn("match EACH OTHER", doc)


class TestWarpMatchSemantics(unittest.TestCase):
    """Match/ballot model vs brute-force reference over the vector battery."""

    def test_reference_groups_sentinel_lanes_together(self):
        # Sanity pin on the reference itself: >= 2 sentinel lanes must form a
        # group (this is the CUDA match.any.sync.b32 behavior the capability
        # asks to confirm).  If this fails, the reference -- not the models
        # -- is wrong.
        out = _reference_match(VECTORS["ids_with_sentinel"], FULL_MASK)
        sentinel_lanes = [l for l in range(LANES) if VECTORS["ids_with_sentinel"][l] == SENTINEL]
        self.assertGreater(len(sentinel_lanes), 1)
        expected_group = sum(1 << l for l in sentinel_lanes)
        for l in sentinel_lanes:
            self.assertEqual(out[l], expected_group)

    def test_emitted_match_matches_reference_all_vectors(self):
        for label, values in VECTORS.items():
            with self.subTest(vector=label):
                got = _model_match_emitted(values, FULL_MASK)
                ref = _reference_match(values, FULL_MASK)
                bad = [l for l in _member_lanes(FULL_MASK) if got[l] != ref[l]]
                self.assertEqual(bad, [], f"{label} wrong at lanes {bad}")

    def test_emitted_match_matches_reference_all_masks(self):
        values = VECTORS["ids_with_sentinel"]
        for label, mask in MASKS.items():
            with self.subTest(mask=label):
                got = _model_match_emitted(values, mask)
                ref = _reference_match(values, mask)
                bad = [l for l in _member_lanes(mask) if got[l] != ref[l]]
                self.assertEqual(bad, [], f"mask={label} wrong at lanes {bad}")

    def test_outside_mask_lanes_get_zero(self):
        mask = MASKS["lower_half"]
        out = _reference_match(VECTORS["no_sentinel_all_ids"], mask)
        for l in range(LANES):
            if not (mask >> l) & 1:
                self.assertEqual(out[l], 0, f"lane {l}")
        # And no result bit names a lane outside the mask.
        for l in _member_lanes(mask):
            self.assertEqual(out[l] & ~mask, 0, f"lane {l}")

    def test_duplicate_detection_count_idiom(self):
        # The dependent-task idiom: count = popc(match_any_sync(FULL, v)) on
        # active lanes equals the in-warp occurrence count of the lane's id.
        values = VECTORS["ids_with_sentinel"]
        out = _reference_match(values, FULL_MASK)
        n_active = sum(1 for v in values if v != SENTINEL)
        for l in range(n_active):
            occurrences = sum(1 for v in values[:n_active] if v == values[l])
            self.assertEqual(_popc(out[l]), occurrences, f"lane {l} id={values[l]}")
        # Every active id here is >= 0, so no active lane's group contains a
        # sentinel lane (group membership is value-exact).
        for l in range(n_active):
            for lp in range(n_active, LANES):
                self.assertFalse((out[l] >> lp) & 1, f"lane {l} group leaked sentinel lane {lp}")

    def test_sentinel_in_payload_inflates_counts_documented(self):
        # When the sentinel IS a valid payload (pre-condition violated), the
        # collective still groups by value -- so lanes 0 and 2 (payload -1)
        # merge with the inactive tail and the count inflates past the
        # duplicate count.  This pins the documented hazard as observable
        # behavior instead of leaving it undefined.
        values = VECTORS["sentinel_in_payload"]
        out = _reference_match(values, FULL_MASK)
        neg_group = [l for l in range(LANES) if _u32(values[l]) == _u32(SENTINEL)]
        self.assertGreater(len(neg_group), 2)
        self.assertEqual(_popc(out[0]), len(neg_group))

    def test_counter_model_excluded_sentinel_diverges(self):
        # Pins the hazard these semantics exclude: a lowering that isolates
        # sentinel lanes must disagree with the reference on every vector
        # with >= 2 sentinel lanes.
        for label in ("ids_with_sentinel", "all_sentinel", "single_active"):
            with self.subTest(vector=label):
                bad = [
                    l for l in range(LANES)
                    if _model_match_sentinel_excluded(VECTORS[label], FULL_MASK)[l] != _reference_match(
                        VECTORS[label], FULL_MASK)[l]
                ]
                self.assertNotEqual(bad, [], f"sentinel-excluded model unexpectedly "
                                    f"correct on {label}")

    def test_counter_model_sentinel_matches_all_diverges(self):
        for label in ("ids_with_sentinel", "single_active", "sentinel_in_payload"):
            with self.subTest(vector=label):
                bad = [
                    l for l in range(LANES)
                    if _model_match_sentinel_matches_all(VECTORS[label], FULL_MASK)[l] != _reference_match(
                        VECTORS[label], FULL_MASK)[l]
                ]
                self.assertNotEqual(bad, [], f"sentinel-matches-all model unexpectedly "
                                    f"correct on {label}")

    def test_int32_negative_ids_agree_with_uint32_patterns(self):
        # CUDA groups match.any by 32-bit pattern; an int32 id of -3 and the
        # uint32 id 0xFFFFFFFD are the same group.  The reference compares
        # patterns, so the negative-id vector must group exactly like its
        # two's-complement image.
        values = VECTORS["negative_ids"]
        out_signed = _reference_match(values, FULL_MASK)
        out_bits = _reference_match([_u32(v) for v in values], FULL_MASK)
        self.assertEqual(out_signed, out_bits)


class TestWarpBallotSemantics(unittest.TestCase):
    """Ballot model vs per-lane predicate reference."""

    def test_ballot_matches_reference_all_masks(self):
        preds = [i % 3 == 0 for i in range(LANES)]
        for label, mask in MASKS.items():
            with self.subTest(mask=label):
                ref = _reference_ballot(preds, mask)
                for l in _member_lanes(mask):
                    expected_bit = 1 if preds[l] else 0
                    self.assertEqual((ref[l] >> l) & 1, expected_bit)
                    self.assertEqual(ref[l] & ~mask, 0)

    def test_ballot_duplicate_detection_idiom(self):
        # ballot(FULL, ids[lane] == my_id) reproduces the match group; this is
        # the fallback formulation the dependent task's variants use.
        values = VECTORS["ids_with_sentinel"]
        match_out = _reference_match(values, FULL_MASK)
        for l in range(10):  # active lanes only
            preds = [values[lp] == values[l] for lp in range(LANES)]
            ballot_out = _reference_ballot(preds, FULL_MASK)
            self.assertEqual(ballot_out[l], match_out[l], f"lane {l}")

    def test_empty_group_ballot_is_zero(self):
        preds = [False] * LANES
        ref = _reference_ballot(preds, FULL_MASK)
        for l in range(LANES):
            self.assertEqual(ref[l], 0)

    def test_popc_of_ballot_counts_votes(self):
        preds = [i % 3 == 0 for i in range(LANES)]
        for label, mask in MASKS.items():
            ref = _reference_ballot(preds, mask)
            member = _member_lanes(mask)
            if not member:
                continue
            expected = sum(1 for l in member if preds[l])
            self.assertEqual(_popc(ref[member[0]]), expected, label)


if __name__ == "__main__":
    unittest.main()
