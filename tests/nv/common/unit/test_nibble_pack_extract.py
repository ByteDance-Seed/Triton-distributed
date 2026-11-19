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
"""CPU semantics regression for the 4-bit field (nibble) pack/extract intrinsics.

Covers the generic contract of ``language/intrin/nibble.py`` without a GPU, in
the same two-layer style as ``test_warp_scan_intrinsics.py``:

* Structural: every intrinsic emits a fused, unsigned-domain shift/mask form.
  ``extract_field4`` must cast the lane to ``unsigned int`` before shifting so
  the sign bit of the most-significant nibble never smears into the result;
  ``insert_field4`` must clear the target nibble with a ``~(15u << shift)``
  mask; ``pack_field4_pair`` must mask both operands to 4 bits.
* Semantic: a Python lockstep model that mirrors each emitted C++ expression
  statement-for-statement is checked against a brute-force bit reference for
  all eight slots and a battery of packed lanes, including an 8-way extract
  driven through a *permuted* slot map (the Marlin 16x64 affine interleave
  reduced to its slot permutation) and the extract/insert inverse round-trip.

No CUDA toolchain or GPU is needed; codegen output is consumed as source text
via the BuiltinOp ``codegen`` staticmethods, exactly like the warp-scan tests.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.language import builtin_base as bb
from little_kernel.language.intrin import nibble
from little_kernel.core.type_system import int32, uint32, float32

FIELD4_BITS = nibble.FIELD4_BITS
N_SLOTS = nibble.FIELD4_PER_LANE  # 8 nibbles per 32-bit lane
MASK = nibble.FIELD4_MASK


@lk.ll_kernel(backend='cuda', is_entry=True)
def _unsigned_round_trip(lanes: ll.Tensor[ll.uint32], output: ll.Tensor[ll.uint32]) -> ll.void:
    i: ll.int32 = ll.threadIdx_x()
    slot: ll.int32 = i & 7
    field: ll.int32 = ll.extract_field4(lanes[i], slot)
    output[i] = ll.insert_field4(lanes[i], slot, field)


@lk.ll_kernel(backend='cuda', is_entry=True)
def _signed_round_trip(lanes: ll.Tensor[ll.int32], output: ll.Tensor[ll.int32]) -> ll.void:
    i: ll.int32 = ll.threadIdx_x()
    slot: ll.int32 = i & 7
    q: ll.int32 = lanes[i]
    field: ll.int32 = ll.extract_field4(q, slot)
    output[i] = ll.insert_field4(q, slot, field)


@lk.ll_kernel(backend='cuda', is_entry=True)
def _pack_dynamic_fields(lo: ll.Tensor[ll.int32], hi: ll.Tensor[ll.int32], output: ll.Tensor[ll.int32]) -> ll.void:
    i: ll.int32 = ll.threadIdx_x()
    output[i] = ll.pack_field4_pair(lo[i], hi[i])


class TestNibblePipeline(unittest.TestCase):
    """Exercise actual frontend tracing, type inference, passes and emission."""

    def test_unsigned_extract_insert_compiles_through_the_full_pipeline(self):
        source = _unsigned_round_trip.compile(PASSES['cuda'], codegen_cuda)
        self.assertIn('__global__', source)
        self.assertIn('unsigned int', source)
        self.assertIn('static_cast<unsigned int>', source)
        self.assertIn('15u', source)

    def test_dynamic_pair_compiles_through_the_full_pipeline(self):
        source = _pack_dynamic_fields.compile(PASSES['cuda'], codegen_cuda)
        self.assertIn('__global__', source)
        self.assertIn('static_cast<int>', source)
        self.assertIn('15u', source)


class TestInsertResultCastThroughEmitter(unittest.TestCase):
    """The emitter's dtype channel must drive insert_field4's result cast.

    insert_field4 opts into ``__codegen_needs_types__`` and selects its outer
    ``static_cast<{int|unsigned int}>`` spelling from ``_arg_dtypes[0]``.
    The direct-codegen twin
    (``TestNibbleStructure.test_insert_result_cast_matches_declared_lane_type``)
    passes ``_arg_dtypes`` by hand, so it cannot catch a regression where the
    emitter (``call_codegen``) stops forwarding the per-argument dtypes -- the
    cast would silently fall back to ``int`` and narrow the top nibble of a
    uint32 lane.  Compiling whole kernels through ``.compile()`` pins the real
    emitter path: a uint32 lane must yield ``static_cast<unsigned int>`` and an
    int32 lane ``static_cast<int>``.

    The insert expression is discriminated from extract_field4's (whose
    codegen always spells ``static_cast<unsigned int>(...)`` around the lane
    even for an int32 kernel) by insert's clear-mask ``& ~(15u << ...)``.
    """

    # Full body shape of the emitted insert expression (see nibble.py):
    #   static_cast<RESULT>((static_cast<unsigned int>(LANE) & ~(15u << ((slot) * 4))) | ...)
    # ``extract_field4`` never emits a ``~(`` clear-mask -- its body is a
    # ``>> ... & 15u)`` -- so matching the opening cast together with a
    # leading clear-mask, before any shift, attributes the cast to insert
    # and not to a neighbouring extract statement.
    INSERT_CLEAR_MASK = f"& ~({MASK}u <<"
    INSERT_BODY_U32 = ("static_cast<unsigned int>((static_cast<unsigned int>(lanes[i]) "
                       f"& ~({MASK}u <<")
    INSERT_BODY_I32 = ("static_cast<int>((static_cast<unsigned int>(q) "
                       f"& ~({MASK}u <<")

    def _count_insert_cast(self, source: str, result_cpp: str) -> int:
        prefix = f"static_cast<{result_cpp}>((static_cast<unsigned int>("
        kills = (" >> ", ">>", ";", "\n")
        count = 0
        idx = 0
        while True:
            idx = source.find(prefix, idx)
            if idx < 0:
                return count
            window = source[idx + len(prefix):idx + len(prefix) + 200]
            stop = min((window.find(k) for k in kills if window.find(k) >= 0), default=len(window))
            if self.INSERT_CLEAR_MASK in window[:stop]:
                count += 1
            idx += len(prefix)

    def test_wrapper_opts_into_typed_codegen(self):
        # The emitter consults this attribute to decide whether to build and
        # pass _arg_dtypes; dropping it must fail a test, not degrade quietly.
        self.assertTrue(getattr(nibble.insert_field4, bb.CODEGEN_NEEDS_TYPES_ATTR, False))
        self.assertTrue(getattr(ll.insert_field4, bb.CODEGEN_NEEDS_TYPES_ATTR, False))

    def test_uint32_lane_yields_unsigned_result_cast_in_emitted_kernel(self):
        source = _unsigned_round_trip.compile(PASSES['cuda'], codegen_cuda)
        self.assertEqual(self._count_insert_cast(source, "unsigned int"), 1)
        self.assertEqual(self._count_insert_cast(source, "int"), 0, msg="uint32 insert fell back to the int cast "
                         "(dtype channel not forwarded by the emitter?)")
        # The emitted statement ends up assigned into a uint32 output, so the
        # full insert body shape must appear verbatim.
        self.assertIn(self.INSERT_BODY_U32, source)

    def test_int32_lane_yields_int_result_cast_in_emitted_kernel(self):
        source = _signed_round_trip.compile(PASSES['cuda'], codegen_cuda)
        self.assertEqual(self._count_insert_cast(source, "int"), 1)
        self.assertEqual(self._count_insert_cast(source, "unsigned int"), 0)
        self.assertIn(self.INSERT_BODY_I32, source)

    def test_twin_kernels_differ_only_in_insert_result_cast(self):
        # Belt-and-braces: the same kernel shape with both lane types must
        # produce insert casts of opposite signedness; a hardcoded result
        # spelling in codegen (or a dropped dtype channel) breaks one side.
        src_u = _unsigned_round_trip.compile(PASSES['cuda'], codegen_cuda)
        src_i = _signed_round_trip.compile(PASSES['cuda'], codegen_cuda)
        self.assertIn("static_cast<unsigned int>((static_cast<unsigned int>", src_u)
        self.assertNotIn("static_cast<unsigned int>((static_cast<unsigned int>(q) & ~(", src_i)
        # extract_field4's unsigned lane cast is present in BOTH kernels and
        # is exactly why a bare assertIn('static_cast<unsigned int') cannot
        # substitute for these checks.
        self.assertIn("static_cast<unsigned int>(lanes[i]) >>", src_u)
        self.assertIn("static_cast<unsigned int>(q) >>", src_i)


def _codegen(fn, *args, _arg_dtypes=None):
    """Pull the emitted C++ return_val out of a builtin wrapper."""
    codegen = getattr(fn, bb.CODEGEN_FUNC_ATTR)
    if _arg_dtypes is not None:
        return codegen(*args, _arg_dtypes=_arg_dtypes).return_val
    return codegen(*args).return_val


def _return_type(fn, *arg_types):
    rt = getattr(fn, bb.EVAL_RETURN_TYPE_ATTR)
    return rt(*arg_types)


# ---------------------------------------------------------------------------
# Python lockstep models of the emitted C++ expressions.
#
# Models work on the uint32 view of the lane (matching the emitted
# static_cast<unsigned int>) and return the int32 reinterpret of the result.
# ---------------------------------------------------------------------------


def _u32(x):
    return x & 0xFFFFFFFF


def _model_extract(q, slot):
    """static_cast<int>((static_cast<unsigned int>(q) >> slot*4) & 15u)."""
    return (_u32(q) >> (slot * FIELD4_BITS)) & MASK


def _model_insert(q, slot, value):
    """(u32(q) & ~(15u << shift)) | ((u32(value) & 15u) << shift)."""
    shift = slot * FIELD4_BITS
    cleared = _u32(q) & (~(_u32(MASK) << shift) & 0xFFFFFFFF)
    return _u32(cleared | ((_u32(value) & MASK) << shift))


def _model_pack_pair(lo, hi):
    """(u32(lo) & 15u) | ((u32(hi) & 15u) << 4)."""
    return (_u32(lo) & MASK) | ((_u32(hi) & MASK) << FIELD4_BITS)


# ---------------------------------------------------------------------------
# Brute-force reference over the raw 32-bit pattern.
# ---------------------------------------------------------------------------


def _ref_unpack(q):
    """Lane -> list of its 8 nibbles, index 0 = least significant."""
    return [(_u32(q) >> (s * FIELD4_BITS)) & MASK for s in range(N_SLOTS)]


# Lane battery: alternating, all-ones, sign-bit-set, single top nibble, zero.
LANES = {
    "alternating": 0x12345678, "all_ones": 0xFFFFFFFF, "sign_bit": 0x80000000, "top_nibble_only": 0xF0000000, "zero":
    0x00000000, "low_byte": 0x000000AB, "marlin_pattern": 0x76543210,  # nibble s == s, easy to eyeball
}


class TestNibbleStructure(unittest.TestCase):
    """Structural checks on the emitted C++ source."""

    def test_extract_uses_unsigned_logical_shift(self):
        body = _codegen(nibble.extract_field4, "q", "slot_expr")
        self.assertIn("static_cast<unsigned int>(q)", body)
        self.assertIn(">>", body)
        self.assertIn(f"& {MASK}u", body)
        self.assertIn("static_cast<int>", body)

    def test_extract_scales_slot_by_field_width(self):
        body = _codegen(nibble.extract_field4, "q", "slot_expr")
        self.assertIn(f"(slot_expr) * {FIELD4_BITS}", body)

    def test_insert_clears_target_nibble(self):
        body = _codegen(nibble.insert_field4, "q", "slot_expr", "v")
        self.assertIn(f"~({MASK}u << ((slot_expr) * {FIELD4_BITS}))", body)
        self.assertIn(f"(static_cast<unsigned int>(v) & {MASK}u)", body)
        self.assertIn("|", body)

    def test_pack_masks_both_operands(self):
        body = _codegen(nibble.pack_field4_pair, "lo", "hi")
        self.assertIn(f"(static_cast<unsigned int>(lo) & {MASK}u)", body)
        self.assertIn(f"(static_cast<unsigned int>(hi) & {MASK}u) << {FIELD4_BITS}", body)

    def test_insert_result_cast_matches_declared_lane_type(self):
        # Regression: return_type declares the result as q_type, so the final
        # static_cast must spell the same C++ type.  Earlier codegen always
        # emitted static_cast<int>(...); for a uint32 lane the spliced
        # expression then had type int while the IR promised uint32, so a
        # downstream >> or comparison took signed semantics.  The intrinsic
        # opts into the emitter's _arg_dtypes channel and must spell
        # unsigned int iff the lane dtype is uint32.
        body_i32 = _codegen(nibble.insert_field4, "q", "slot_expr", "v", _arg_dtypes=[int32, int32, int32])
        body_u32 = _codegen(nibble.insert_field4, "q", "slot_expr", "v", _arg_dtypes=[uint32, int32, int32])
        self.assertTrue(body_i32.startswith("static_cast<int>("))
        self.assertTrue(body_u32.startswith("static_cast<unsigned int>("))
        # Fallback with no dtype channel: int32-in/int32-out common case.
        body_fallback = _codegen(nibble.insert_field4, "q", "slot_expr", "v")
        self.assertTrue(body_fallback.startswith("static_cast<int>("))


class TestNibbleTyping(unittest.TestCase):
    """return_type contract: lane must be int32/uint32; signedness of insert."""

    def test_extract_returns_int32(self):
        self.assertIs(_return_type(nibble.extract_field4, int32, 0), int32)
        self.assertIs(_return_type(nibble.extract_field4, uint32, 0), int32)

    def test_extract_rejects_float_lane(self):
        with self.assertRaises(TypeError):
            _return_type(nibble.extract_field4, float32, 0)

    def test_insert_preserves_lane_signedness(self):
        self.assertIs(_return_type(nibble.insert_field4, int32, 0, int32), int32)
        self.assertIs(_return_type(nibble.insert_field4, uint32, 0, int32), uint32)

    def test_insert_rejects_float_lane(self):
        with self.assertRaises(TypeError):
            _return_type(nibble.insert_field4, float32, 0, int32)

    def test_pack_returns_int32(self):
        self.assertIs(_return_type(nibble.pack_field4_pair, int32, int32), int32)
        self.assertIs(_return_type(nibble.pack_field4_pair, uint32, uint32), int32)

    def test_pack_rejects_float_operands(self):
        # Regression: pack_field4_pair used to validate nothing, so a float32
        # operand reached codegen as static_cast<unsigned int>(float_expr) and
        # silently truncated (3.9 -> 3) instead of failing in the frontend.
        with self.assertRaises(TypeError):
            _return_type(nibble.pack_field4_pair, float32, int32)
        with self.assertRaises(TypeError):
            _return_type(nibble.pack_field4_pair, int32, float32)

    def test_extract_rejects_float_slot(self):
        # A float-typed slot expression must fail as a typed frontend error,
        # not as an opaque nvcc error on the emitted C++ shift operand.
        with self.assertRaises(TypeError):
            _return_type(nibble.extract_field4, int32, float32)

    def test_insert_rejects_float_slot_and_value(self):
        with self.assertRaises(TypeError):
            _return_type(nibble.insert_field4, int32, float32, int32)
        with self.assertRaises(TypeError):
            _return_type(nibble.insert_field4, int32, int32, float32)

    def test_int_literal_slot_and_none_type_accepted(self):
        # Literal Python ints (unit-test invocation path) and None (pipeline
        # could not infer a type) must not trip the operand validation.
        self.assertIs(_return_type(nibble.extract_field4, int32, 3), int32)
        self.assertIs(_return_type(nibble.extract_field4, int32, None), int32)
        self.assertIs(_return_type(nibble.insert_field4, uint32, None, None), uint32)


class TestNibbleSemantics(unittest.TestCase):
    """Emitted-algorithm model vs brute-force per-slot reference."""

    def test_extract_matches_reference_all_slots_all_lanes(self):
        for label, q in LANES.items():
            ref = _ref_unpack(q)
            for slot in range(N_SLOTS):
                with self.subTest(lane=label, slot=slot):
                    self.assertEqual(_model_extract(q, slot), ref[slot])

    def test_extract_permuted_slot_map(self):
        # An 8-way extract driven through a permuted slot map (a reduced
        # Marlin-style affine interleave on the slot index).  The intrinsic
        # takes the permuted slot as an ordinary integer expression; the model
        # must still land on the correct nibble.
        perm = [3, 0, 5, 2, 7, 4, 1, 6]  # output position -> slot index
        for label, q in LANES.items():
            ref = _ref_unpack(q)
            for out_pos, slot in enumerate(perm):
                with self.subTest(lane=label, out_pos=out_pos, slot=slot):
                    self.assertEqual(_model_extract(q, slot), ref[slot])

    def test_insert_is_inverse_of_extract(self):
        # Rebuilding a lane nibble-by-nibble via insert must reproduce q.
        for label, q in LANES.items():
            rebuilt = 0
            for slot in range(N_SLOTS):
                rebuilt = _model_insert(rebuilt, slot, _model_extract(q, slot))
            with self.subTest(lane=label):
                self.assertEqual(rebuilt, _u32(q))

    def test_insert_only_touches_target_slot(self):
        for label, q in LANES.items():
            ref = _ref_unpack(q)
            for slot in range(N_SLOTS):
                new_ref = list(ref)
                new_ref[slot] = 0xA  # value we write (masked to 4 bits)
                out = _model_insert(q, slot, 0xA)
                got = _ref_unpack(out)
                with self.subTest(lane=label, slot=slot):
                    self.assertEqual(got, new_ref)

    def test_insert_masks_value_to_four_bits(self):
        # Values wider than 4 bits must be masked, not leak into neighbours.
        q = 0
        out = _model_insert(q, 0, 0x1F)  # 0b11111 -> stores 0b1111
        self.assertEqual(_ref_unpack(out)[0], 0xF)
        self.assertEqual(_ref_unpack(out)[1], 0x0)

    def test_pack_pair_inverse_of_extract_byte(self):
        q = 0xAB
        lo = _model_extract(q, 0)
        hi = _model_extract(q, 1)
        self.assertEqual(_model_pack_pair(lo, hi), 0xAB)

    def test_pack_8_nibbles_round_trip(self):
        # Fold all 8 nibbles of a lane back together pairwise and confirm the
        # rebuild equals the original: the full inverse-pack path.
        for label, q in LANES.items():
            nib = _ref_unpack(q)
            p0 = _model_pack_pair(nib[0], nib[1])  # bits [0:8)
            p1 = _model_pack_pair(nib[2], nib[3])  # bits [0:8)
            p2 = _model_pack_pair(nib[4], nib[5])
            p3 = _model_pack_pair(nib[6], nib[7])
            b0 = p0 | (p1 << 8)  # low 16
            b1 = p2 | (p3 << 8)  # high 16
            rebuilt = _u32(b0 | (b1 << 16))
            with self.subTest(lane=label):
                self.assertEqual(rebuilt, _u32(q))


if __name__ == "__main__":
    unittest.main()
