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
"""First-class warp reduction intrinsics (butterfly allreduce family).

Row-parallel kernels (rowwise reductions, normalization statistics, softmax
denominators, attention score sums, dot-product style epilogues) reduce one
register value per lane into a warp-wide total.  Before these intrinsics the
only in-language route was for every frontend to hand-roll the same five-line
``__shfl_xor_sync`` butterfly chain (see design/flashcomm_compute.py for one
such copy) -- duplicated boilerplate with no shared typing contract and no
shared effect declaration.

``warp_reduce_sum`` lowers a full-warp butterfly sum to one shared,
deduplicated ``template <typename T>`` C++ helper emitted through the normal
Builtin body path; ``warp_reduce_sum_width`` is the segmented variant for
row layouts that pack multiple independent rows per warp (CUDA's 4-argument
``__shfl_xor_sync`` width parameter partitions the warp into power-of-two
segments, exactly matching a lanes-per-row tile).  ``warp_reduce_max`` and
``warp_reduce_min`` cover the ordered commutative combines needed by row
epilogues (softmax max-subtract, clamp bounds, best-so-far broadcasts) with
the same shared-helper structure; the scan module already ships
inclusive/exclusive max/min, so reduce now has operator parity.

Both ops are ALLREDUCEs: every participating lane receives the reduced
total, which is what per-row scaling epilogues need (each lane re-scales its
own elements by the row statistic without an extra broadcast).

Typing contract (declared in return_type, checked at type-inference time):

* The value operand must be one of the dtypes CUDA's shuffle family lowers
  natively: float32, float64, int32, uint32, int64, uint64, float16,
  bfloat16.  Anything else (including narrower integers and register
  arrays/Tensors) raises TypeError instead of silently emitting ill-formed
  device code.
* The mask and the segmented width follow the CUDA legality rules: every
  lane named in ``mask`` must execute the op together under uniform control
  flow; ``width`` must be a compile-time power of two in [1, 32] (validated
  in codegen, since it is spelled as a C++ integer literal).

Legality / memory-effect contract (same as language/intrin/shuffle.py):

* The declared effect is ``R_RMEM_WARP_SYNC`` on the ``value`` operand: the
  result is assembled from every lane's register, so the op must not be
  CSE'd or moved across warp synchronization or divergence boundaries.  It
  has NO GMEM/SMEM/TMEM effect and is idempotent under replay with the same
  operands.
* A cross-warp (block-level) reduction intentionally has no dedicated op
  here: it composes from this primitive plus the existing shared-memory and
  ``__syncthreads`` intrinsics (warp partial -> smem -> barrier -> first
  warp reduces), which keeps scratch allocation and barrier placement under
  frontend control instead of hiding them inside an opaque builtin.
"""

from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC
from little_kernel.core.type_system import LLType, bfloat16, float16, float32, float64, int32, int64, uint32, uint64
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp, CODEGEN_NEEDS_TYPES_ATTR

# Same warp-collective register-read effect as the shuffle family: the
# result is read back out of a DIFFERENT lane's register.
_SHFL_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "value"), )

# Dtypes CUDA's __shfl_*_sync family lowers without bit-slicing.  Narrower
# integers are rejected so a misspelled frontend cast fails loudly at type
# inference instead of producing an nvcc error deep in generated code.
_SUPPORTED_VALUE_TYPES = (float32, float64, int32, uint32, int64, uint64, float16, bfloat16)

# Full-warp butterfly: log2(32) == 5 exchange steps.  The helper is emitted
# once per translation unit (Builtin bodies are deduplicated by the
# emitter), so kernels with several reduction sites share one definition.
_WARP_REDUCE_SUM_BODY = r"""
template <typename T>
__device__ __forceinline__ T lk_warp_reduce_sum(unsigned mask, T v) {
    v += __shfl_xor_sync(mask, v, 16);
    v += __shfl_xor_sync(mask, v, 8);
    v += __shfl_xor_sync(mask, v, 4);
    v += __shfl_xor_sync(mask, v, 2);
    v += __shfl_xor_sync(mask, v, 1);
    return v;
}
"""

# Segmented butterfly: exchanges stay inside power-of-two lane segments of
# width W (CUDA's 4-argument shuffle form), so one warp can reduce W-lane
# rows in parallel (e.g. a [blockDim.y rows] x [16 lanes/H-tile] layout).
_WARP_REDUCE_SUM_WIDTH_BODY = r"""
template <int W, typename T>
__device__ __forceinline__ T lk_warp_reduce_sum_width(unsigned mask, T v) {
#pragma unroll
    for (int offset = W / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(mask, v, offset, W);
    }
    return v;
}
"""

_HALF_INCLUDES = ("<cuda_fp16.h>", "<cuda_bf16.h>")


def _value_type_or_raise(value_type) -> LLType:
    if not isinstance(value_type, LLType) or not any(value_type == supported for supported in _SUPPORTED_VALUE_TYPES):
        raise TypeError("warp reduce supports scalar value dtypes float32, float64, "
                        "int32, uint32, int64, uint64, float16, bfloat16; got "
                        f"{value_type!r}. Cast with an explicit conversion op first.")
    return value_type


def _half_includes_for(value_dtype) -> list:
    """Half-precision shuffles need the CUDA half/bf16 headers for +=.

    ``_arg_dtypes`` is an opt-in emitter channel; when it is unavailable
    (older emitter paths, direct codegen-func calls in tests) fall back to
    the safe superset of both headers.
    """
    if value_dtype is None:
        return list(_HALF_INCLUDES)
    includes = []
    if value_dtype == float16:
        includes.append(_HALF_INCLUDES[0])
    if value_dtype == bfloat16:
        includes.append(_HALF_INCLUDES[1])
    return includes


# ==============================================================================
# warp_reduce_sum -- full-warp butterfly allreduce
# ==============================================================================


@builtin
class warp_reduce_sum(BuiltinOp):
    """Warp-wide butterfly sum: every lane receives the 32-lane total.

    ``warp_reduce_sum(mask, value)`` emits log2(32) ``__shfl_xor_sync``
    exchange steps through a shared templated device helper.  Convergent
    warp collective: every lane named in ``mask`` must execute it together
    under uniform control flow (canonical full-warp mask ``0xffffffff``).
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return _value_type_or_raise(value_type)

    @staticmethod
    def codegen(mask, value, _arg_dtypes=None):
        value_dtype = _arg_dtypes[-1] if _arg_dtypes else None
        return Builtin(
            body=_WARP_REDUCE_SUM_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=f"lk_warp_reduce_sum({mask}, {value})",
        )


setattr(warp_reduce_sum, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# warp_reduce_sum_width -- segmented butterfly allreduce (lanes-per-row tiles)
# ==============================================================================


def _segment_width_or_raise(width) -> int:
    """The segment width is a C++ literal at codegen time; parse and check it.

    CUDA requires width to be a power of two in [1, 32].  A non-literal
    spelling means the frontend passed a runtime value, which cannot lower
    to the 4-argument shuffle's immediate operand.
    """
    try:
        parsed = int(str(width).strip(), 10)
    except ValueError:
        raise TypeError("warp_reduce_sum_width expects a compile-time integer literal "
                        f"segment width, got {width!r}") from None
    if parsed < 1 or parsed > 32 or (parsed & (parsed - 1)) != 0:
        raise ValueError("warp_reduce_sum_width segment width must be a power of two in "
                         f"[1, 32], got {parsed}")
    return parsed


@builtin
class warp_reduce_sum_width(BuiltinOp):
    """Segmented warp sum: reduce within power-of-two lane groups of ``width``.

    ``warp_reduce_sum_width(mask, value, width)`` reduces ``value`` across
    each group of ``width`` consecutive lanes independently (CUDA's
    4-argument ``__shfl_xor_sync``).  This matches row layouts that pack
    ``32 // width`` rows per warp; every lane inside a segment receives that
    segment's total.  ``width`` must be a power of two in [1, 32] spelled as
    an integer literal (compile-time checked).  Same convergence and memory
    effect contract as :class:`warp_reduce_sum`.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, width_type):
        return _value_type_or_raise(value_type)

    @staticmethod
    def codegen(mask, value, width, _arg_dtypes=None):
        segment = _segment_width_or_raise(width)
        value_dtype = _arg_dtypes[1] if _arg_dtypes and len(_arg_dtypes) == 3 else None
        return Builtin(
            body=_WARP_REDUCE_SUM_WIDTH_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=f"lk_warp_reduce_sum_width<{segment}>({mask}, {value})",
        )


setattr(warp_reduce_sum_width, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# warp_reduce_max / warp_reduce_min -- full-warp ordered allreduces
#
# ``warp_reduce_sum`` covered the commutative-with-add case, but row/segment
# epilogues routinely need an ORDERED warp total instead: max for online
# softmax numerators, min for coordinate clamping / present-value sends,
# argmax-style "best score so far" broadcasts.  Before these intrinsics the
# same frontends had to hand-roll the butterfly with a ``?:`` combine, which
# is a cut-and-paste of the sum butterfly with one operator swapped --
# exactly the mistake-prone boilerplate this module exists to remove.  The
# scan family already pairs add with max/min; this brings reduce to parity.
#
# Same convergence / effect / typing contract as ``warp_reduce_sum``: a
# convergent warp collective declared as ``R_RMEM_WARP_SYNC`` on ``value``;
# the value dtype is restricted to the same shuffle-native set (the helper
# body relies on a strict ``<``/``>`` compare, which every listed dtype
# supports natively).  Ties resolve to whichever lane wins the final
# exchange, matching the butterfly's associativity; if a deterministic
# lowest-lane-wins order matters, use ``warp_scan_inclusive_max`` /
# ``warp_scan_inclusive_min`` instead (they pick the lowest lane on ties).
# ==============================================================================

# IEEE-754 uniform extrema combine contract (documented; see builtin docs):
#
# The naive ``(o > v) ? o : v`` butterfly combine is NOT a well-defined
# allreduce for floating-point inputs containing NaN or signed zeros:
#
# * A NaN compares false against every operand, so a lane holding NaN keeps
#   NaN while its XOR partner holding a finite value keeps the finite value.
#   After one exchange step the warp disagrees about the "max", violating
#   the allreduce contract that every lane must observe the same result.
# * ``-0.0`` and ``+0.0`` compare equal under ``>``/``<`` but have
#   different bit patterns; the naive combine preserves whichever lane's
#   zero happened to arrive last, again leaving the warp in disagreement.
#
# The uniform shared policy used here is:
#
# * **non-NaN wins**: if exactly one operand is NaN the other (numeric) value
#   is the combine result.  A NaN only survives when every participating
#   lane supplied NaN (matching IEEE-754 ``fmax``/``fmin`` propagation).
# * **signed-zero canonicalization**: ``+0.0`` and ``-0.0`` are treated as
#   numerically equal; the butterfly picks a canonical representative
#   deterministically.  For ``max`` the canonical zero is ``+0.0``; for
#   ``min`` the canonical zero is ``-0.0``.  This matches the IEEE-754
#   ``maximum``/``minimum`` tie conventions and is lane-deterministic.
# * **infinities**: ``±inf`` compare correctly under ``>``/``<`` and need no
#   special casing; ``+inf`` wins max over every finite value, ``-inf``
#   wins min symmetrically.
# * **integers / non-float dtypes**: NaN and signed-zero cannot occur, so
#   the plain ``>``/``<`` select is already uniform; the helper handles
#   them through the same code path because ``isnan`` on an integer
#   compile-time folds to false (CUDA's ``isnan`` is overloaded for the
#   numeric dtypes in ``_SUPPORTED_VALUE_TYPES``).
#
# The predicate body is emitted in one shared templated helper per op so
# every frontend / call site shares exactly one lowering; per-op codegen
# below only names the entry point.

# Lane-symmetric per-step combine.  Helper predicates:
#
#   _lk_isnan_ieee(x):      (x != x) -- IEEE NaN test, always false for ints.
#   _lk_signbit_ieee(x):    float-promoted sign bit read.  The tie break
#                           only activates when ``o == v`` numerically,
#                           which for integers means identical bit patterns,
#                           so any deterministic choice is uniform.  For
#                           floats it distinguishes the +/-0.0 signs so the
#                           butterfly can canonicalize: max prefers +0.0,
#                           min prefers -0.0 (matching IEEE maximum/minimum).
#
# The same predicates are needed by the max, min, max_width, min_width
# helpers, and by-string body dedupe does not see across bodies of
# different length, so the block is wrapped in a preprocessor once-guard.
# That keeps emission-side dedupe (within one body) AND long-form
# multi-op kernels (across bodies) free of duplicate-definition errors.
_ORDERED_PREDICATES = r"""
#ifndef _LK_WARP_EXTREMA_PREDICATES
#define _LK_WARP_EXTREMA_PREDICATES
template <typename T>
__device__ __forceinline__ bool _lk_isnan_ieee(T x) {
    return x != x;
}
template <typename T>
__device__ __forceinline__ float _lk_to_f32(T x) {
    return static_cast<float>(x);
}
__device__ __forceinline__ bool _lk_signbit_f32(float x) {
    return signbit(x) != 0;
}
template <typename T>
__device__ __forceinline__ bool _lk_signbit_ieee(T x) {
    return _lk_signbit_f32(_lk_to_f32(x));
}
#endif  // _LK_WARP_EXTREMA_PREDICATES
"""


def _ordered_reduce_bodies():
    """Full-warp butterfly helpers with IEEE-uniform max/min combines.

    Each exchange step reads ``o`` (the value shuffled in from the XOR
    partner) and ``v`` (this lane's current partial).  The combine
    statement selects the new ``v`` under the documented non-NaN-wins,
    canonical-zero policy.  The same statement is re-used at every step so
    the lane-symmetric invariant is preserved by induction: if every lane
    enters step k with the same partial, every lane leaves step k with the
    same partial, regardless of which physical lane contributed the
    eventual winner.
    """

    # max: replace v with o iff
    #   * o is not NaN, AND
    #   * (v is NaN) OR (o > v strictly) OR (o == v AND o is the canonical
    #     zero sign: prefer +0.0 over -0.0, i.e. take o when v is the -0.0
    #     of the pair and o is the +0.0).
    # Equivalently with signed-zero canonicalization: on a +/-0.0 tie we
    # keep +0.0 for max.  The (o==v && signbit(v) && !signbit(o)) clause
    # only fires on that exact tie because for non-zero values equal signs
    # are the only way o==v, and equal signs make both signbit() calls
    # agree.
    max_combine = ("if (!_lk_isnan_ieee(o) && "
                   "(_lk_isnan_ieee(v) || o > v || "
                   "(o == v && _lk_signbit_ieee(v) && !_lk_signbit_ieee(o)))) { v = o; }")
    # min: mirror image; prefer -0.0 over +0.0 on a +/-0.0 tie.
    min_combine = ("if (!_lk_isnan_ieee(o) && "
                   "(_lk_isnan_ieee(v) || o < v || "
                   "(o == v && !_lk_signbit_ieee(v) && _lk_signbit_ieee(o)))) { v = o; }")

    def _ordered_helper(fn_name, combine_body):
        return (_ORDERED_PREDICATES + "\n" + "template <typename T>\n" +
                f"__device__ __forceinline__ T {fn_name}(unsigned mask, T v) {{\n" + "    T o;\n" +
                f"    o = __shfl_xor_sync(mask, v, 16); {combine_body}\n" +
                f"    o = __shfl_xor_sync(mask, v, 8);  {combine_body}\n" +
                f"    o = __shfl_xor_sync(mask, v, 4);  {combine_body}\n" +
                f"    o = __shfl_xor_sync(mask, v, 2);  {combine_body}\n" +
                f"    o = __shfl_xor_sync(mask, v, 1);  {combine_body}\n" + "    return v;\n" + "}")

    return (
        _ordered_helper("lk_warp_reduce_max", max_combine),
        _ordered_helper("lk_warp_reduce_min", min_combine),
    )


_WARP_REDUCE_MAX_BODY, _WARP_REDUCE_MIN_BODY = _ordered_reduce_bodies()

# Segmented (width-parameterised) ordered allreduce helpers.  Same
# IEEE-uniform combine policy as the full-warp variants, specialised to
# power-of-two lane segments via CUDA's 4-argument __shfl_xor_sync form.
# These are emitted as ``template <int W, typename T>`` so the segment
# width is stamped as an immediate at every exchange.


def _ordered_reduce_width_bodies():
    # Same combine statements as the full-warp variants (defined above);
    # the only difference at this layer is the segmented shuffle form.
    max_combine = ("if (!_lk_isnan_ieee(o) && "
                   "(_lk_isnan_ieee(v) || o > v || "
                   "(o == v && _lk_signbit_ieee(v) && !_lk_signbit_ieee(o)))) { v = o; }")
    min_combine = ("if (!_lk_isnan_ieee(o) && "
                   "(_lk_isnan_ieee(v) || o < v || "
                   "(o == v && !_lk_signbit_ieee(v) && _lk_signbit_ieee(o)))) { v = o; }")

    def _ordered_width_helper(fn_name, combine_body):
        return (_ORDERED_PREDICATES + "\n" + "template <int W, typename T>\n" +
                f"__device__ __forceinline__ T {fn_name}(unsigned mask, T v) {{\n" + "    T o;\n" + "#pragma unroll\n" +
                "    for (int offset = W / 2; offset > 0; offset >>= 1) {\n" +
                f"        o = __shfl_xor_sync(mask, v, offset, W); {combine_body}\n" + "    }\n" + "    return v;\n" +
                "}")

    return (
        _ordered_width_helper("lk_warp_reduce_max_width", max_combine),
        _ordered_width_helper("lk_warp_reduce_min_width", min_combine),
    )


_WARP_REDUCE_MAX_WIDTH_BODY, _WARP_REDUCE_MIN_WIDTH_BODY = (_ordered_reduce_width_bodies())


@builtin
class warp_reduce_max(BuiltinOp):
    """Warp-wide butterfly maximum: every lane receives the 32-lane maximum.

    ``warp_reduce_max(mask, value)`` emits log2(32) ``__shfl_xor_sync``
    exchanges through a shared templated device helper that applies a
    lane-symmetric IEEE-754-extrema combine.  Convergent warp collective:
    every lane named in ``mask`` must execute it together under uniform
    control flow.  Same typing and declared-effect contract as
    :class:`warp_reduce_sum`; the returned type equals the value dtype.

    Uniform-value contract (documented policy, encoded in the helper):

    * **Non-NaN wins**: if at least one participating lane supplies a
      non-NaN value, every participating lane observes the same numeric
      maximum.  NaN is returned by every lane iff every lane supplied NaN.
    * **Signed-zero canonicalization**: ``+0.0`` and ``-0.0`` are treated
      as numerically equal; when both appear, every lane observes ``+0.0``
      (the IEEE ``maximum``-style tie).  Bit-level uniformity is
      guaranteed across lanes.
    * **Infinities** behave as expected: ``+inf`` beats every finite
      value; ``-inf`` is returned only when no finite or ``+inf`` exists.
    * **Integer dtypes** have no NaN/signed-zero and reduce by strict
      ``>``, so every lane still observes an identical value.

    For deterministic lowest-lane-wins ordering on *ties among distinct
    values* use ``warp_scan_inclusive_max``; ties at the extrema here are
    resolved to a canonical representative, not a specific source lane.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return _value_type_or_raise(value_type)

    @staticmethod
    def codegen(mask, value, _arg_dtypes=None):
        value_dtype = _arg_dtypes[-1] if _arg_dtypes else None
        return Builtin(
            body=_WARP_REDUCE_MAX_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=f"lk_warp_reduce_max({mask}, {value})",
        )


setattr(warp_reduce_max, CODEGEN_NEEDS_TYPES_ATTR, True)


@builtin
class warp_reduce_min(BuiltinOp):
    """Warp-wide butterfly minimum: every lane receives the 32-lane minimum.

    ``warp_reduce_min(mask, value)`` emits log2(32) ``__shfl_xor_sync``
    exchanges through a shared templated device helper that applies a
    lane-symmetric IEEE-754-extrema combine.  Convergent warp collective:
    every lane named in ``mask`` must execute it together under uniform
    control flow.  Same typing and declared-effect contract as
    :class:`warp_reduce_sum`; the returned type equals the value dtype.

    Uniform-value contract, mirroring :class:`warp_reduce_max`:

    * **Non-NaN wins**: if any lane supplies a non-NaN value, every lane
      observes the same numeric minimum.  NaN is returned by every lane
      iff every lane supplied NaN.
    * **Signed-zero canonicalization**: on a ``+0.0``/``-0.0`` tie every
      lane observes ``-0.0`` (the IEEE ``minimum``-style tie).
    * **Infinities**: ``-inf`` wins over every finite value; ``+inf`` is
      returned only when no finite or ``-inf`` exists.
    * **Integer dtypes** reduce by strict ``<`` with bit-exact agreement.

    For deterministic lowest-lane-wins ordering on distinct-value ties use
    ``warp_scan_inclusive_min``; extrema ties here resolve canonically.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return _value_type_or_raise(value_type)

    @staticmethod
    def codegen(mask, value, _arg_dtypes=None):
        value_dtype = _arg_dtypes[-1] if _arg_dtypes else None
        return Builtin(
            body=_WARP_REDUCE_MIN_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=f"lk_warp_reduce_min({mask}, {value})",
        )


setattr(warp_reduce_min, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# warp_reduce_max_width / warp_reduce_min_width -- segmented ordered allreduce
#
# Segmented counterparts of :class:`warp_reduce_max` / :class:`warp_reduce_min`
# for row layouts that pack several independent rows per warp (matching
# :class:`warp_reduce_sum_width`).  Exchanges stay inside power-of-two lane
# segments via CUDA's 4-argument ``__shfl_xor_sync`` form.  The lane-symmetric
# IEEE-extrema combine policy is identical to the full-warp variants: non-NaN
# wins, signed zeros canonicalize (to ``+0.0`` for max, ``-0.0`` for min),
# infinities and integer dtypes behave naturally -- so within each segment,
# every lane observes the same extrema value bit-for-bit.
# ==============================================================================


@builtin
class warp_reduce_max_width(BuiltinOp):
    """Segmented warp maximum: reduce max within power-of-two lane groups.

    ``warp_reduce_max_width(mask, value, width)`` reduces ``value`` across
    each group of ``width`` consecutive lanes independently (CUDA's
    4-argument ``__shfl_xor_sync`` with a width parameter).  Useful when a
    warp packs ``32 // width`` independent rows/epilogue tiles.  ``width``
    must be a power-of-two integer literal in [1, 32] (compile-time
    checked).  Same NaN, signed-zero, infinity and integer uniformity
    guarantees as :class:`warp_reduce_max`, scoped to each segment.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, width_type):
        return _value_type_or_raise(value_type)

    @staticmethod
    def codegen(mask, value, width, _arg_dtypes=None):
        segment = _segment_width_or_raise(width)
        value_dtype = _arg_dtypes[1] if _arg_dtypes and len(_arg_dtypes) == 3 else None
        return Builtin(
            body=_WARP_REDUCE_MAX_WIDTH_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=f"lk_warp_reduce_max_width<{segment}>({mask}, {value})",
        )


setattr(warp_reduce_max_width, CODEGEN_NEEDS_TYPES_ATTR, True)


@builtin
class warp_reduce_min_width(BuiltinOp):
    """Segmented warp minimum: reduce min within power-of-two lane groups.

    ``warp_reduce_min_width(mask, value, width)`` reduces ``value`` across
    each group of ``width`` consecutive lanes independently.  Same width
    constraints, typing rules and IEEE-extrema uniformity policy as
    :class:`warp_reduce_max_width`, with the mirrored minimum combine.
    """

    effects = _SHFL_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type, width_type):
        return _value_type_or_raise(value_type)

    @staticmethod
    def codegen(mask, value, width, _arg_dtypes=None):
        segment = _segment_width_or_raise(width)
        value_dtype = _arg_dtypes[1] if _arg_dtypes and len(_arg_dtypes) == 3 else None
        return Builtin(
            body=_WARP_REDUCE_MIN_WIDTH_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=f"lk_warp_reduce_min_width<{segment}>({mask}, {value})",
        )


setattr(warp_reduce_min_width, CODEGEN_NEEDS_TYPES_ATTR, True)
