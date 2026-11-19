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
"""CPU semantics regression for the warp prefix-scan intrinsics.

Covers the generic scan/masked-participation contract of
``backends/nv/common/language/intrin/scan.py`` without a GPU:

* Structural: every emitted helper body must compute the caller's mask
  segment base (``lk_base``), gate the Hillis-Steele accumulation on
  ``lk_laneid - lk_base >= delta`` (not raw ``lk_laneid >= delta``), keep the
  ``%laneid`` read, and emit the ``__shfl_up_sync`` collective.  The
  exclusive scan must seed the additive identity at the segment base, not at
  absolute lane 0.
* Semantic: a Python lockstep model that mirrors the emitted C++ statement
  for statement is checked against a brute-force per-segment prefix
  reference for all four operators (inclusive add/max/min, exclusive add)
  over a mask battery: full warp, half warps, quarter warps, odd-sized
  contiguous segments, single-lane masks, and a scattered mask.  The
  pre-fix algorithm (ungated segment clip, seed at absolute lane 0) is kept
  as a documented counter-model: it must disagree with the reference on the
  segmented masks, which pins down the defect these tests guard.

No CUDA toolchain or GPU is needed; codegen output is consumed as source
text via the BuiltinOp ``codegen`` staticmethods.
"""

import sys
import unittest
from pathlib import Path

# Mirror the repo-root bootstrap used by the other nv/common CPU tests: the
# package lives under <repo>/python (see setenv.sh), and tests must import it
# without requiring setenv.sh to be sourced first.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "python" / "little_kernel").is_dir():
        if str(_parent / "python") not in sys.path:
            sys.path.insert(0, str(_parent / "python"))
        break

from little_kernel.backends.nv.common.language.intrin import scan
from little_kernel.language import builtin_base as bb

LANES = 32
FULL_MASK = 0xFFFFFFFF

# Participation-mask battery.  Every mask names a convergent group; the
# brute-force reference treats each maximal run of contiguous set bits as an
# independent scan segment.
MASKS = {
    "full_warp": 0xFFFFFFFF,
    "upper_half": 0xFFFF0000,
    "lower_half": 0x0000FFFF,
    "quarter_1": 0x0000FF00,
    "odd_segment": 0b11111 << 7,  # lanes 7..11
    "single_lane": 1 << 20,
    "two_segments": 0x00FF00FF,
    "scattered": 0b10110101101001011010110100101101,
}

OPS = ("inclusive_add", "inclusive_max", "inclusive_min", "exclusive_add")


def _codegen_body(op_name):
    """Pull the emitted helper body out of the builtin wrapper."""
    fn = getattr(scan, "warp_scan_" + op_name)
    codegen = getattr(fn, bb.CODEGEN_FUNC_ATTR)
    return codegen("lk_mask", "lk_value").body


def _return_type_passthrough(op_name):
    fn = getattr(scan, "warp_scan_" + op_name)
    rt = getattr(fn, bb.EVAL_RETURN_TYPE_ATTR)
    sentinel = object()
    return rt("mask_type", sentinel) is sentinel


# --------------------------------------------------------------------------
# Python lockstep models of the emitted helper bodies.
#
# shfl_up_sync is modeled per PTX shfl.sync.up with clamp operand 0x1f:
# srcLane = lane - delta; out-of-range lanes (srcLane < 0) keep their own
# value.  Mask membership does not clip the source; the software guard in
# the helper is responsible for segment correctness.
# --------------------------------------------------------------------------


def _shfl_up_sync(acc, lane, delta):
    src = lane - delta
    return acc[src] if src >= 0 else acc[lane]


def _segment_base(mask, lane):
    """Lowest lane of the maximal run of mask bits containing ``lane``."""
    base = lane
    while base > 0 and (mask >> (base - 1)) & 1:
        base -= 1
    return base


def _combine_add(a, b):
    return a + b


def _combine_max(a, b):
    return b if b > a else a


def _combine_min(a, b):
    return b if b < a else a


def _model_emitted(values, mask, op):
    """Mirror of the emitted C++ (segment-gated, seed at segment base)."""
    combine = {
        "inclusive_add": _combine_add,
        "inclusive_max": _combine_max,
        "inclusive_min": _combine_min,
        "exclusive_add": _combine_add,
    }[op]
    acc = list(values)
    bases = [_segment_base(mask, l) for l in range(LANES)]
    for delta in (1, 2, 4, 8, 16):
        acc = [combine(acc[l], _shfl_up_sync(acc, l, delta)) if l - bases[l] >= delta else acc[l] for l in range(LANES)]
    if op == "exclusive_add":
        acc = [_shfl_up_sync(acc, l, 1) for l in range(LANES)]
        acc = [0 if l == bases[l] else acc[l] for l in range(LANES)]
    return acc


def _model_pre_fix(values, mask, op):
    """Documented counter-model: pre-fix helpers (raw lane guard, seed@0)."""
    combine = {
        "inclusive_add": _combine_add,
        "inclusive_max": _combine_max,
        "inclusive_min": _combine_min,
        "exclusive_add": _combine_add,
    }[op]
    acc = list(values)
    for delta in (1, 2, 4, 8, 16):
        acc = [combine(acc[l], _shfl_up_sync(acc, l, delta)) if l >= delta else acc[l] for l in range(LANES)]
    if op == "exclusive_add":
        acc = [_shfl_up_sync(acc, l, 1) for l in range(LANES)]
        acc[0] = 0
    return acc


def _reference(values, mask, op):
    """Brute force: lane l scans mask-member lanes of its own segment."""
    out = []
    for l in range(LANES):
        if not (mask >> l) & 1:
            out.append(None)  # non-participating lane: result not consumed
            continue
        base = _segment_base(mask, l)
        upto = l if op.startswith("inclusive") else l - 1
        members = [lp for lp in range(base, upto + 1)]
        if op.endswith("add"):
            out.append(sum(values[lp] for lp in members))
        elif op.endswith("max"):
            out.append(max(values[lp] for lp in members))
        else:
            out.append(min(values[lp] for lp in members))
    return out


def _member_lanes(mask):
    return [l for l in range(LANES) if (mask >> l) & 1]


class TestWarpScanStructure(unittest.TestCase):
    """Structural checks on the emitted helper source."""

    def test_segment_guard_present_all_ops(self):
        for op in OPS:
            body = _codegen_body(op)
            self.assertIn("lk_base", body, op)
            self.assertIn("lk_laneid - lk_base >= static_cast<unsigned int>(lk_delta)", body, op)
            self.assertNotIn("if (lk_laneid >= static_cast<unsigned int>(lk_delta))", body, op)

    def test_exclusive_seeds_at_segment_base(self):
        body = _codegen_body("exclusive_add")
        self.assertIn("if (lk_laneid == lk_base)", body)
        self.assertNotIn("if (lk_laneid == 0u)", body)

    def test_laneid_and_collective_present(self):
        for op in OPS:
            body = _codegen_body(op)
            self.assertIn("%laneid", body, op)
            self.assertIn("__shfl_up_sync(lk_mask, lk_acc, lk_delta)", body, op)

    def test_full_mask_fast_path(self):
        for op in OPS:
            body = _codegen_body(op)
            self.assertIn("if (lk_mask != 0xffffffffu)", body, op)

    def test_return_type_passthrough(self):
        for op in OPS:
            self.assertTrue(_return_type_passthrough(op), op)

    def test_effect_declared_as_warp_collective(self):
        for op in OPS:
            fn = getattr(scan, "warp_scan_" + op)
            effects = getattr(fn, bb.MEMORY_EFFECTS_ATTR)
            self.assertEqual(len(effects), 1, op)
            kind = effects[0].kind
            # R_RMEM_WARP_SYNC: issued by the warp, sync execution.
            self.assertEqual(kind.issuer.name, "WARP", op)
            self.assertEqual(kind.space.name, "RMEM", op)
            self.assertEqual(kind.access.name, "R", op)


class TestWarpScanSemantics(unittest.TestCase):
    """Emitted-algorithm model vs brute-force per-segment reference."""

    def _check_model(self, model, op, mask, should_match):
        if op.endswith("min"):
            # Ascending: a lower lane's value always beats a higher lane's
            # under min, so any cross-segment leak changes the result.
            values = list(range(1, LANES + 1))
        elif op.endswith("max"):
            # Descending: same property for max.
            values = [LANES - l for l in range(LANES)]
        else:
            values = list(range(1, LANES + 1))
        got = model(values, mask, op)
        ref = _reference(values, mask, op)
        bad = [l for l in _member_lanes(mask) if got[l] != ref[l]]
        if should_match:
            self.assertEqual(bad, [], f"{op} mask=0x{mask:08x} wrong at lanes {bad}")
        return bad

    def test_emitted_matches_reference_all_masks(self):
        for op in OPS:
            for label, mask in MASKS.items():
                with self.subTest(op=op, mask=label):
                    self._check_model(_model_emitted, op, mask, True)

    def test_emitted_full_warp_matches_pre_fix(self):
        # The canonical full-warp mask must be bit-identical to the pre-fix
        # behavior: the fix changes nothing on the fast path.
        for op in OPS:
            values = list(range(1, LANES + 1))
            self.assertEqual(_model_emitted(values, FULL_MASK, op), _model_pre_fix(values, FULL_MASK, op), op)

    def test_pre_fix_diverges_on_segmented_masks(self):
        # Pins down the defect the emitted helpers fix: on segmented masks
        # the old guard leaks lower-segment contributions (all ops) and the
        # old exclusive seed misses the segment base.
        for op in OPS:
            bad = self._check_model(_model_pre_fix, op, MASKS["upper_half"], should_match=False)
            self.assertNotEqual(bad, [], f"pre-fix {op} unexpectedly correct on "
                                "upper_half -- defect no longer reproduced")

    def test_exclusive_segment_base_gets_identity(self):
        mask = MASKS["upper_half"]
        base = _segment_base(mask, 16)
        values = list(range(1, LANES + 1))
        got = _model_emitted(values, mask, "exclusive_add")
        self.assertEqual(base, 16)
        self.assertEqual(got[16], 0)
        # Rank-within-segment property: lane base+k holds the sum of the
        # first k segment values.
        seg = _member_lanes(mask)
        for k, l in enumerate(seg):
            self.assertEqual(got[l], sum(values[lp] for lp in seg[:k]))


if __name__ == "__main__":
    unittest.main()
