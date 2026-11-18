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
"""Typed 4-bit field (nibble) pack/extract intrinsics over a 32-bit lane.

Software FP4/FP8 dequantization kernels (E2M1, E4M3, Marlin's packed NVFP4
weight tiles) hold eight values in one 32-bit register lane.  Decoding them in
typed scalar IR normally forces either an unbounded chain of raw shift/mask
expressions the tracer cannot fuse, or a raw inline-CUDA body -- both of which
the frontend contract disallows (all algorithmic loads and arithmetic must
live in typed IR).

These intrinsics keep the decode in typed IR:

* :func:`extract_field4` -- read nibble ``slot`` of a 32-bit lane as an
  independently addressable ``int32`` (0..15), which the caller may
  ``val_cast`` to a float lane or reinterpret as E2M1.
* :func:`insert_field4` -- the inverse pack: return a copy of the lane with
  nibble ``slot`` replaced by ``value & 0xf``.
* :func:`pack_field4_pair` -- pack two nibbles into one byte. Combine those
  bytes with explicit wider shifts and ORs when assembling a full lane.

The ``slot`` may be any typed integer expression (a literal, or an affine
index transform such as the Marlin interleave below); codegen emits its C++
text inline as the shift operand.

Marlin 16x64 interleave
-----------------------
Marlin stores a 16(row) x 64(nibble) tile so that the nibble at logical
``(row, col)`` lives in slot ``perm`` of the packed lane for ``row`` via an
affine ``(warp, thread)`` interleave.  That permutation is an ordinary integer
index transform over the slot, so it does not need its own builtin: compose
``extract_field4`` with the caller's affine slot expression (for example
``((col & 3) + ((col >> 2) & 1) * 4) & 7``).  Because ``slot`` is emitted as a
plain shift operand, compile-time slot expressions can fold to constant
shifts. A runtime slot bounded by ``& 7`` remains a runtime shift.
"""
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp
from little_kernel.core.type_system import LLType, int32, uint32

# Width in bits of one field and the number of fields per 32-bit lane.  Kept
# as module constants so the docstring, the shift computation and the tests
# share a single source of truth.
FIELD4_BITS = 4
FIELD4_PER_LANE = 32 // FIELD4_BITS
FIELD4_MASK = (1 << FIELD4_BITS) - 1


def _field_mask_code() -> str:
    return str(FIELD4_MASK)


def _check_lane_type(q_type, who):
    if q_type not in (int32, uint32):
        raise TypeError(f"{who} expects an int32/uint32 lane, got {q_type}")


def _check_int_operand(op_type, name, who):
    """Reject non-integer typed operands (e.g. a float slot or value).

    ``op_type`` is the inferred LLType of the operand, a Python ``int`` for a
    literal argument under direct unit-test invocation, or ``None`` when the
    pipeline could not infer a type (codegen then still sees the C++ text, so
    there is nothing to validate).  Only a positively-known non-integer type
    is rejected.
    """
    if op_type is None or isinstance(op_type, int):
        return
    if op_type not in (int32, uint32):
        raise TypeError(f"{who} expects an int32/uint32 {name}, got {op_type}")


@builtin
class extract_field4(BuiltinOp):
    """Extract nibble ``slot`` of a packed 32-bit lane as an ``int32``.

    ``q`` is the packed lane (``int32`` or ``uint32``); ``slot`` selects the
    nibble (0 = least significant).  The return value is the zero-extended
    nibble in ``int32``, so ``extract_field4(q, s) == (q >> (4*s)) & 15``.
    Passing ``uint32`` keeps the lane unsigned so codegen emits a logical
    shift; ``int32`` lanes are first bit-widened to ``uint32`` so the sign bit
    of the most-significant nibble never smears into the result.

    The slot may be any typed integer expression (e.g. a Marlin affine
    interleave); it is emitted inline as the shift operand. Only a
    compile-time constant slot permits a constant shift. Callers must
    keep the slot in ``[0, 8)`` -- an out-of-range slot is a C++ shift wider
    than the lane, i.e. undefined behaviour.
    """

    @staticmethod
    def return_type(q_type, slot_type):
        _check_lane_type(q_type, "extract_field4")
        _check_int_operand(slot_type, "slot", "extract_field4")
        return int32

    @staticmethod
    def codegen(q, slot):
        # Emit the canonical fused form: one shift, one mask.  Cast to the
        # unsigned spelling first so an int32 lane takes a logical shift.
        shift = f"(({slot}) * {FIELD4_BITS})"
        body = (f"static_cast<int>((static_cast<unsigned int>({q}) >> "
                f"{shift}) & {_field_mask_code()}u)")
        return Builtin(body="", includes=[], return_val=body)


@builtin
class insert_field4(BuiltinOp):
    """Return a copy of the lane with nibble ``slot`` replaced.

    ``value`` is masked to its low 4 bits, so callers need not pre-mask:
    ``insert_field4(q, s, v) == (q & ~(15 << 4*s)) | ((v & 15) << 4*s)``.
    The result keeps the lane's signedness (``int32`` in, ``int32`` out;
    ``uint32`` in, ``uint32`` out); this is the exact inverse of
    :func:`extract_field4` per slot.  Codegen casts the spliced expression to
    the declared IR return type, so a ``uint32`` lane yields a C++
    ``unsigned int`` expression and downstream shifts/comparisons keep the
    logical (unsigned) semantics the IR type promises.
    """

    # Typed codegen: the final cast must spell the lane's declared IR type
    # (int vs unsigned int), so codegen needs the per-argument dtypes.
    __codegen_needs_types__ = True

    @staticmethod
    def return_type(q_type, slot_type, value_type):
        _check_lane_type(q_type, "insert_field4")
        _check_int_operand(slot_type, "slot", "insert_field4")
        _check_int_operand(value_type, "value", "insert_field4")
        return q_type

    @staticmethod
    def codegen(q, slot, value, _arg_dtypes=None):
        shift = f"(({slot}) * {FIELD4_BITS})"
        mask = _field_mask_code()
        # The result's IR type is the lane's type (return_type returns
        # q_type).  Spell the outer cast to match: unsigned int for a uint32
        # lane, int for an int32 lane.  When the dtype channel is unavailable
        # fall back to int, matching the int32-in/int32-out common case.
        lane_dtype = _arg_dtypes[0] if _arg_dtypes else None
        result_cpp = "unsigned int" if lane_dtype == uint32 else "int"
        # Clear the target nibble, then OR in the masked value.  Stay in the
        # unsigned domain so the clear-mask is well-defined at the top bit.
        body = (f"static_cast<{result_cpp}>((static_cast<unsigned int>({q}) & "
                f"~({mask}u << {shift})) | "
                f"((static_cast<unsigned int>({value}) & {mask}u) << {shift}))")
        return Builtin(body="", includes=[], return_val=body)


@builtin
class pack_field4_pair(BuiltinOp):
    """Pack two 4-bit values into the low byte of an ``int32``.

    ``lo`` occupies bits [0:4), ``hi`` bits [4:8); the upper 24 bits are zero.
    Both operands are masked to 4 bits. This is not a general concatenation
    primitive: assemble wider lanes by combining the resulting bytes with
    shifts and ORs.
    """

    @staticmethod
    def return_type(lo_type, hi_type):
        # Without this guard a float operand would reach codegen as
        # static_cast<unsigned int>(float_expr) and silently truncate.
        _check_int_operand(lo_type, "lo", "pack_field4_pair")
        _check_int_operand(hi_type, "hi", "pack_field4_pair")
        return int32

    @staticmethod
    def codegen(lo, hi):
        mask = _field_mask_code()
        body = (f"static_cast<int>((static_cast<unsigned int>({lo}) & {mask}u) | "
                f"((static_cast<unsigned int>({hi}) & {mask}u) << {FIELD4_BITS}))")
        return Builtin(body="", includes=[], return_val=body)
