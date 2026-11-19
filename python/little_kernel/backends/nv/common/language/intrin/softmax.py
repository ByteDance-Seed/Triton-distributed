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
"""First-class online-softmax / row-normalize epilogue intrinsics.

Numerically-stable softmax-style epilogues appear in two recurring shapes
across the dependent workloads:

* **One-shot row normalize** (per-row top-K gate renormalization, small-K
  classifier heads, router logits): given a row value ``x``, its precomputed
  row max ``m`` and row sum-of-exps ``s = Σ_j exp(v_j - m)``, emit
  ``exp(x - m) / s``.  Kernels need this once per selected element, e.g. a
  K=2 MoE gate that gathers two scores per token and renormalizes them.
* **Streaming rescale** (flash-attention, chunked / online softmax): when a
  new block raises the running max from ``m_old`` to ``m_new``, the
  accumulated numerator and denominator must be multiplied by
  ``exp(m_old - m_new)`` before the new block is folded in.

Before this module every frontend hand-rolled the same three steps --
``fmax``-style max tracking, a ``__expf`` numerator, a reciprocal divide --
which is not only duplicated boilerplate but also a real accuracy hazard:
under ``--use_fast_math`` nvcc silently remaps every ``expf`` to the
``MUFU.EX2``-backed approximation, and the max-subtract guard is the only
thing keeping the quotient in ``[0, 1]``.  These intrinsics centralize the
composition into one shared, typed, fast-math-immune helper per shape.

Contract (declared at type-inference time):

* All operands and results are ``float32``.  The dependent workloads are
  FP32 accumulate pipelines (see the ``WHERE`` clause of the MoE gate
  task); a frontend holding a half/bf16 value must cast explicitly first,
  so an ill-typed call fails loudly here instead of emitting ill-formed
  device code.  (Higher-precision variants can be added the same way if a
  future workload needs them.)
* The helpers are pure value-to-value functions: they read and write no
  memory and perform no cross-lane communication, so they declare NO
  memory effect (unlike the warp collectives in ``intrin/reduce.py``).
  Frontends compose them with ``warp_reduce_max`` / ``warp_reduce_sum``
  (or simply a K-register folding loop) to build the full row reduction,
  then call these for the stable epilogue.

Accuracy / NaN contract (encoded in the emitted helper, matching the
IEEE-uniform policy used by ``warp_reduce_max``/``warp_reduce_min``):

* The exponential is libdevice ``__nv_expf`` (via the existing
  ``exp_precise_f32`` lowering), which ``--use_fast_math`` cannot remap to
  the ``__expf`` approximation, and a fast division is avoided in favour of
  the default IEEE-correct ``fdiv``.
* ``softmax_exp_neg(x)`` computes ``exp(-x)`` with the ``x >= 0`` range
  reduced by the caller's max-subtract, so the result is in ``(0, 1]`` for
  finite ``x``; it does NOT clamp -- the call site guarantees the range.
* ``softmax_normalize(x, rowmax, rowsum)`` is ``exp(x - rowmax) / rowsum``.
  ``x <= rowmax`` by construction, so ``exp(...) <= 1`` and the quotient is
  bounded; when ``rowsum`` is denormal/0 (all inputs ``-inf``) the IEEE
  result ``inf``/``nan`` propagates rather than being masked, so a degenerate
  row is debuggable instead of silently producing 0.
* ``softmax_rescale(new_max, old_max)`` is ``exp(old_max - new_max)``.
  ``old_max <= new_max`` by construction, so the factor is in ``(0, 1]``;
  on the very first block a frontend may pass ``old_max = -inf`` (likewise
  when a later block ties ``-inf``), which yields ``exp(-inf - -inf)``, i.e.
  ``nan`` -- the conventional fix is to seed the accumulator so the factor is
  only ever consumed after a finite addend has zeroed the term it scales.
  This NaN is intentionally NOT masked: masking it would hide the bug.

Online-softmax partial-state merge (the second recurring online shape):

* **Scalar merge** ``softmax_merge(m_a, l_a, m_b, l_b)``: combine two
  partial softmax states -- each a running max ``m_*`` with the
  max-rescaled denominator ``l_* = Σ_j exp(v_j - m_*)`` -- into the
  state of their union: ``m = max(m_a, m_b)``,
  ``l = l_a * exp(m_a - m) + l_b * exp(m_b - m)``.  This is the fold a
  streaming attention / chunked-softmax frontend applies for every
  incoming block partial, and the per-step combine of any tree reduction
  over softmax partials.  Hand-rolling it from ``softmax_rescale`` takes
  a compare, two rescale calls and an fma, and the misspellings are all
  silent: rescaling against the wrong (not-yet-combined) max overflows
  the numerator, adding the raw maxes instead of the rescaled
  denominators loses the normalization, and dividing instead of
  accumulating implements an average, not a union.
* **Warp merge** ``warp_softmax_merge(mask, m, l)``: butterfly allreduce
  whose per-step combine IS the scalar merge, so every lane finishes with
  the union state ``(row_max, Σ_j exp(v_j - row_max))`` of the full warp
  (or of each power-of-two lane segment for
  ``warp_softmax_merge_width``).  This is the cross-lane statistics
  reduction of warp-fragmented flash-attention pipelines, where each
  lane accumulates ``(m, l)`` over its fragment and the row total must
  land in every lane for the rescale epilogue.  Hand-rolling it means
  two shuffles plus the merge math per butterfly step -- exactly the
  duplication-and-drift hazard this module exists to remove.

Merge seeding / NaN contract (shared with the epilogue ops):

* Lanes or streams with no elements yet are seeded ``(-inf, 0)``: the
  merge then yields ``0 * exp(-inf - m) + l_other * exp(m_other - m)`'
  with ``-inf - m == -inf`` and ``exp(-inf) == 0`` for any finite ``m``,
  so the seed vanishes exactly.  Only when EVERY merged partial is
  ``(-inf, 0)`` does the union contain ``exp(-inf - -inf) == nan``;
  matching the epilogue policy, that NaN is propagated, never masked, so
  a degenerate all--inf row surfaces in the frontend instead of reading
  back a plausible-looking 0.

The scalar merge is a pure value-to-value function with NO memory
effect; the warp merges are convergent warp collectives declaring
``R_RMEM_WARP_SYNC`` on the value operands, same as the reduce family.

These are elementwise/warp primitives; nothing here depends on a task
id, a specific expert count, an attention head layout, or the
``T``/``K`` shape of any particular model.
"""

from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC
from little_kernel.core.type_system import TupleType, float32
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp

# Shared devices-side helper block.  All helpers are emitted through the
# normal Builtin body path, so they are deduplicated per translation unit:
# a kernel that normalizes many rows shares one definition.  The once-guard
# additionally keeps a body that a frontend also emits (e.g. one that mixes
# ``softmax_normalize`` with its own copy) link-clean.
_SOFTMAX_HELPERS = r"""
#ifndef _LK_SOFTMAX_EPILOGUE
#define _LK_SOFTMAX_EPILOGUE
extern "C" __device__ float __nv_expf(float);

// exp(x - rowmax), the stable softmax numerator.  x <= rowmax by the
// caller's range reduction, so the result is in (0, 1] for finite inputs.
__device__ __forceinline__ float lk_softmax_exp_neg(float x) {
    return __nv_expf(-x);
}

// Full one-element row-normalize epilogue: exp(x - rowmax) / rowsum.
__device__ __forceinline__ float lk_softmax_normalize(float x, float rowmax,
                                                      float rowsum) {
    return __nv_expf(x - rowmax) / rowsum;
}

// Streaming rescale factor: exp(old_max - new_max).  old_max <= new_max by
// the caller's running-max update, so the factor is in (0, 1].
__device__ __forceinline__ float lk_softmax_rescale(float old_max,
                                                    float new_max) {
    return __nv_expf(old_max - new_max);
}
#endif  // _LK_SOFTMAX_EPILOGUE
"""

# ==============================================================================
# Online-softmax partial-state merge helpers (scalar + warp butterfly).
#
# A softmax partial state is the pair (m, l) with row/running max m and
# max-rescaled denominator l = Σ_j exp(v_j - m).  The merge of two partials
# over disjoint element sets is the state of their union:
#
#     m = fmax(m_a, m_b)
#     l = l_a * exp(m_a - m) + l_b * exp(m_b - m)
#
# Both rescale exponents are <= 0 by the fmax, so every exp argument stays
# range-reduced and no factor can exceed 1 -- the same stability invariant
# the one-shot epilogue relies on, now maintained across arbitrary fold
# orders (streaming blocks, butterfly lanes, tree reductions), which is why
# the merge must go through one shared helper instead of per-frontend
# rescale call pairs.
#
# Seeding: an empty partial is (-inf, 0.0f).  With m finite,
# exp(-inf - m) == 0.0f, so the seed contributes nothing to either
# component.  The all-(-inf) degenerate case yields exp(nan) == nan by
# IEEE rules and is propagated, matching the epilogue ops' no-mask policy.
#
# The warp variants implement the butterfly allreduce whose per-step
# combine is this merge: two __shfl_xor_sync exchanges (one per
# component) plus one scalar merge per step.  Lane-symmetry follows from
# the merge being commutative and associative over partial states with
# the canonical fmax tie, so every lane of the mask finishes with the
# identical (m, l) bit pattern -- the allreduce contract row-rescale
# epilogues need.
# ==============================================================================
_SOFTMAX_MERGE_HELPERS = r"""
#ifndef _LK_SOFTMAX_MERGE
#define _LK_SOFTMAX_MERGE

// Scalar partial-state merge: union of (m_a, l_a) and (m_b, l_b).
// cuda::std::tuple keeps the two components in registers; callers unpack
// via the same tuple path as the block_reduce_max_sum statistics.
__device__ __forceinline__ cuda::std::tuple<float, float>
lk_softmax_merge(float m_a, float l_a, float m_b, float l_b) {
    float m = fmaxf(m_a, m_b);
    float l = l_a * __nv_expf(m_a - m) + l_b * __nv_expf(m_b - m);
    return cuda::std::make_tuple(m, l);
}

// Full-warp butterfly merge: every lane named in mask finishes with the
// union partial state of all participating lanes.  Convergent collective;
// see the reduce family's contract.  5 exchange steps for 32 lanes.
__device__ __forceinline__ cuda::std::tuple<float, float>
lk_warp_softmax_merge(unsigned mask, float m, float l) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float m_o = __shfl_xor_sync(mask, m, offset);
        float l_o = __shfl_xor_sync(mask, l, offset);
        float m_u = fmaxf(m, m_o);
        l = l * __nv_expf(m - m_u) + l_o * __nv_expf(m_o - m_u);
        m = m_u;
    }
    return cuda::std::make_tuple(m, l);
}

// Segmented variant: exchanges stay inside power-of-two lane segments of
// width W (CUDA's 4-argument shuffle form), so one warp merges 32 / W
// independent row-fragment partial states in parallel.
template <int W>
__device__ __forceinline__ cuda::std::tuple<float, float>
lk_warp_softmax_merge_width(unsigned mask, float m, float l) {
#pragma unroll
    for (int offset = W / 2; offset > 0; offset >>= 1) {
        float m_o = __shfl_xor_sync(mask, m, offset, W);
        float l_o = __shfl_xor_sync(mask, l, offset, W);
        float m_u = fmaxf(m, m_o);
        l = l * __nv_expf(m - m_u) + l_o * __nv_expf(m_o - m_u);
        m = m_u;
    }
    return cuda::std::make_tuple(m, l);
}
#endif  // _LK_SOFTMAX_MERGE
"""

# Same warp-collective register-read effect as the shuffle/reduce family:
# each component of the merged state is assembled from other lanes'
# registers.  Declared on both value operands of the warp merges.
_MERGE_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "m"), Effect(R_RMEM_WARP_SYNC, "l"))


def _require_float32(op_name, *types):
    """All operands must be float32; anything else fails at inference."""
    for t in types:
        if t != float32:
            raise TypeError(f"{op_name} expects float32 operands (the dependent "
                            "softmax/normalize workloads are FP32-accumulate); got "
                            f"{t!r}. Cast with an explicit conversion op first.")
    return float32


@builtin
class softmax_exp_neg(BuiltinOp):
    """Stable softmax numerator ``exp(-x)`` for a range-reduced ``x >= 0``.

    ``softmax_exp_neg(x)`` evaluates ``exp(-x)`` with libdevice ``__nv_expf``
    (fast-math immune).  The caller guarantees ``x = m - s >= 0`` where ``m``
    is the running/row max and ``s`` the element score, so the result is in
    ``(0, 1]``.  ``float32`` in, ``float32`` out; no memory effect.

    This is the shared building block for both shapes in the module docstring:
    a one-shot numerator ``exp(s_k - m)`` is ``softmax_exp_neg(m - s_k)``,
    while the per-element body of a flash-attention score tile is the direct
    ``exp(score - running_max)`` call.
    """

    @staticmethod
    def return_type(x_type):
        return _require_float32("softmax_exp_neg", x_type)

    @staticmethod
    def codegen(x):
        return Builtin(body=_SOFTMAX_HELPERS, includes=[], return_val=f"lk_softmax_exp_neg({x})")


@builtin
class softmax_normalize(BuiltinOp):
    """One-element row-normalize epilogue ``exp(x - rowmax) / rowsum``.

    ``softmax_normalize(x, rowmax, rowsum)`` evaluates the final rescale
    applied to every element of a stable softmax row: the numerator
    ``exp(x - rowmax)`` divided by the precomputed row denominator
    ``rowsum = Σ_j exp(v_j - rowmax)``.  All three operands are ``float32``;
    no memory effect.

    For the per-row top-K gate renormalization (the dependent MoE workload)
    a token's ``K`` selected scores each pass through this once, after a
    K-register (or warp-reduced) max and sum are computed up front.  The
    quotient is bounded by ``1`` because ``x <= rowmax`` and
    ``rowsum >= exp(x - rowmax)``; a degenerate all-``-inf`` row yields an
    IEEE ``inf``/``nan`` that is intentionally propagated rather than masked
    so the degenerate input surfaces in the frontend.
    """

    @staticmethod
    def return_type(x_type, rowmax_type, rowsum_type):
        return _require_float32("softmax_normalize", x_type, rowmax_type, rowsum_type)

    @staticmethod
    def codegen(x, rowmax, rowsum):
        return Builtin(body=_SOFTMAX_HELPERS, includes=[], return_val=f"lk_softmax_normalize({x}, {rowmax}, "
                       f"{rowsum})")


@builtin
class softmax_rescale(BuiltinOp):
    """Streaming rescale factor ``exp(old_max - new_max)``.

    ``softmax_rescale(old_max, new_max)`` evaluates the multiplicative
    correction applied to a running softmax numerator/denominator when a
    newly scanned block raises the running max from ``old_max`` to
    ``new_max``: both accumulated quantities are multiplied by this once
    before the new block's contribution is folded in (the flash-attention
    online-softmax update, and equivalently any chunked row-softmax).  Both
    operands and the result are ``float32``; no memory effect.

    ``old_max <= new_max`` by the caller's ``fmax``-style tracking, so the
    factor is in ``(0, 1]``.  The first block's ``old_max = -inf`` (and any
    all-``-inf`` tie) produces ``nan`` by IEEE rules, which is conventional
    and is left unmasked -- seeding the accumulator so the first real block
    zeroes the scaled term is the frontend's responsibility.
    """

    @staticmethod
    def return_type(old_max_type, new_max_type):
        return _require_float32("softmax_rescale", old_max_type, new_max_type)

    @staticmethod
    def codegen(old_max, new_max):
        return Builtin(body=_SOFTMAX_HELPERS, includes=[], return_val=f"lk_softmax_rescale({old_max}, {new_max})")


# ==============================================================================
# softmax_merge -- scalar online-softmax partial-state merge
# ==============================================================================


@builtin
class softmax_merge(BuiltinOp):
    """Merge two online-softmax partial states into their union state.

    ``softmax_merge(m_a, l_a, m_b, l_b)`` combines two partial softmax
    states -- each ``(m, l)`` with ``l = Σ_j exp(v_j - m)`` over its own
    element set -- into the state of the union:
    ``m = max(m_a, m_b)``,
    ``l = l_a * exp(m_a - m) + l_b * exp(m_b - m)``.  Returns the float32
    2-tuple ``(m, l)``.  Pure value-to-value: no memory effect.

    This is the fold a streaming attention or chunked-softmax frontend
    applies to each incoming block partial, and the per-node combine of
    any tree reduction over softmax partials (block reductions fuse
    max/sum instead; see ``block_reduce_max_sum`` for the staged
    two-barrier variant over raw partials).  Both rescale exponents are
    non-positive by the ``fmax``, so no factor exceeds 1 regardless of
    fold order.  Seed an empty partial with ``(-inf, 0)``: it then
    vanishes exactly; an all-``-inf`` merge yields ``nan`` by IEEE rules,
    propagated rather than masked (see module docstring).
    """

    @staticmethod
    def return_type(m_a_type, l_a_type, m_b_type, l_b_type):
        _require_float32("softmax_merge", m_a_type, l_a_type, m_b_type, l_b_type)
        return TupleType((float32, float32))

    @staticmethod
    def codegen(m_a, l_a, m_b, l_b):
        return Builtin(body=_SOFTMAX_MERGE_HELPERS, includes=[],
                       return_val=f"lk_softmax_merge({m_a}, {l_a}, {m_b}, {l_b})")


# ==============================================================================
# warp_softmax_merge / warp_softmax_merge_width -- butterfly partial merge
# ==============================================================================


def _merge_segment_width_or_raise(width) -> int:
    """Segment width must be a compile-time power-of-two literal in [1, 32].

    Same legality rule as the reduce family's segmented variants: the
    width is stamped as the immediate operand of the 4-argument shuffle,
    so a runtime value cannot lower.
    """
    try:
        parsed = int(str(width).strip(), 10)
    except ValueError:
        raise TypeError("warp_softmax_merge_width expects a compile-time integer "
                        f"literal segment width, got {width!r}") from None
    if parsed < 1 or parsed > 32 or (parsed & (parsed - 1)) != 0:
        raise ValueError("warp_softmax_merge_width segment width must be a power of two "
                         f"in [1, 32], got {parsed}")
    return parsed


@builtin
class warp_softmax_merge(BuiltinOp):
    """Warp-wide butterfly merge of per-lane softmax partial states.

    ``warp_softmax_merge(mask, m, l)`` reduces each lane's partial state
    ``(m, l)`` with the online-softmax merge combine across every lane
    named in ``mask`` and returns the union state ``(m_row, l_row)`` to
    EVERY participating lane (an allreduce).  This is the cross-lane
    statistics reduction of warp-fragmented flash-attention / online
    softmax pipelines: after the call, each lane can rescale its own
    fragment by ``exp(m_lane_old - m_row)`` and divide by ``l_row``
    without a second broadcast round.

    Convergent warp collective: every lane named in ``mask`` must execute
    the op together under uniform control flow (canonical mask
    ``0xffffffff``).  The combine is commutative and associative over
    partial states with an ``fmax`` tie, so all lanes finish with the
    identical ``(m, l)`` bit pattern; an all-``-inf`` row yields ``nan``
    in ``l``, propagated per the module policy.  Declared effect is
    ``R_RMEM_WARP_SYNC`` on both ``m`` and ``l`` (see reduce.py); no
    GMEM/SMEM/TMEM effect.
    """

    effects = _MERGE_EFFECTS

    @staticmethod
    def return_type(mask_type, m_type, l_type):
        _require_float32("warp_softmax_merge", m_type, l_type)
        return TupleType((float32, float32))

    @staticmethod
    def codegen(mask, m, l):
        return Builtin(body=_SOFTMAX_MERGE_HELPERS, includes=[], return_val=f"lk_warp_softmax_merge({mask}, {m}, {l})")


@builtin
class warp_softmax_merge_width(BuiltinOp):
    """Segmented warp merge: union partial state within lane groups.

    ``warp_softmax_merge_width(mask, m, l, width)`` applies the butterfly
    partial-state merge independently within each group of ``width``
    consecutive lanes (CUDA's 4-argument ``__shfl_xor_sync``), matching
    row layouts that pack ``32 // width`` row fragments per warp.
    ``width`` must be a power-of-two integer literal in [1, 32]
    (compile-time checked).  Same convergence, effect, seeding and NaN
    contract as :class:`warp_softmax_merge`, scoped to each segment.
    """

    effects = _MERGE_EFFECTS

    @staticmethod
    def return_type(mask_type, m_type, l_type, width_type):
        _require_float32("warp_softmax_merge_width", m_type, l_type)
        return TupleType((float32, float32))

    @staticmethod
    def codegen(mask, m, l, width):
        segment = _merge_segment_width_or_raise(width)
        return Builtin(body=_SOFTMAX_MERGE_HELPERS, includes=[], return_val=f"lk_warp_softmax_merge_width<{segment}>"
                       f"({mask}, {m}, {l})")
