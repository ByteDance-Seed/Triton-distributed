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
"""First-class block-level (cross-warp) reduction intrinsics.

Row-parallel kernels whose reduction axis exceeds one warp of lanes
(RMSNorm/variance statistics over H up to 8K, block-wide softmax
denominators, histogram totals, dot-product epilogues tiled across a whole
CTA) need a value reduced across EVERY thread of the block, not just one
warp.  The warp-level butterfly intrinsics in
``language/intrin/reduce.py`` deliberately stop at the warp boundary and
document the cross-warp composition as frontend work:

    warp partial -> shared memory -> __syncthreads -> first warp reduces
    -> shared memory broadcast -> __syncthreads

That composition is exactly the kind of duplicated, mistake-prone
boilerplate the warp intrinsics exist to remove: every frontend hand-rolls
the same lane-0 store / barrier / guarded reload / second butterfly /
broadcast choreography (see design/flashcomm_compute.py for one copy, with
``warp_sums`` scratch and two explicit ``__syncthreads`` sites).  Faults it
invites in third-party copies:

* forgetting the FIRST barrier, so warp 0 reloads stale or never-written
  partials;
* forgetting the SECOND (broadcast) barrier, so a fast warp proceeds with
  a stale broadcast slot while warp 0 is still reducing;
* reusing the scratch buffer for a second block reduction without a
  trailing barrier, letting a fast warp overwrite partials that slow warps
  have not yet consumed;
* a non-uniform ordered combine on the second butterfly, re-introducing
  the NaN / signed-zero lane disagreement that reduce.py's IEEE-extrema
  policy fixed at the warp level.

``block_reduce_sum`` / ``block_reduce_max`` / ``block_reduce_min`` emit the
entire composition through one shared, deduplicated C++ helper per
operator, reusing the SAME ``lk_warp_reduce_*`` device helpers the warp
intrinsics lower to (single source of truth for the butterfly sequence and
the IEEE-uniform max/min combine).  These are ALLREDUCEs: every thread of
the block receives the block-wide total, which is what per-row scaling
epilogues (RMSNorm ``hidden * rsqrt(var + eps)``) need without an extra
broadcast step.

Typing and legality contract:

* The ``value`` operand must be one of the same shuffle-native dtypes the
  warp intrinsics accept (float32, float64, int32, uint32, int64, uint64,
  float16, bfloat16); anything else raises TypeError at type-inference
  time.  Note these ops are typically composed over fp32 partials
  accumulated in registers even when the activations are bf16.
* ``smem_scratch`` must be a shared-memory pointer whose element type
  equals the value dtype.  It must name at least ``BLOCK_REDUCE_SCRATCH_ELEMS``
  (= 33) elements: up to 32 warp-partial slots (blockDim.x <= 1024 threads
  is 32 warps at CUDA's maximum CTA size) plus one broadcast slot at
  index 0 used AFTER the second barrier.  The frontend owns the
  allocation (``alloc_shared_memory`` or a dynamic-smem slice), keeping
  scratch placement, alignment and bank layout under explicit control.
* ``blockDim.x`` must be a multiple of 32 and at most 1024 at launch; the
  helper derives ``num_warps = (blockDim.x + 31) >> 5`` at runtime, so one
  compiled kernel serves every legal CTA size.  Launch geometry is a
  frontend obligation (the compiler cannot see the host launch config).
* Every thread of the block must execute the op together under uniform
  control flow: the emitted helper contains two ``__syncthreads()``
  barriers, so calling it from divergent code is undefined behavior in the
  same way a raw ``__syncthreads()`` is.

Declared memory effects:

* ``Effect(R_RMEM_WARP_SYNC, "value")``: the result is assembled from
  other lanes' registers via the embedded butterflies, so the op must not
  be CSE'd or moved across warp sync / divergence boundaries.
* ``Effect(W_SMEM_GENERIC_SYNC, "smem_scratch")`` and
  ``Effect(R_SMEM_GENERIC_SYNC, "smem_scratch")`` with a fixed region
  extent of ``BLOCK_REDUCE_SCRATCH_ELEMS``: the helper writes warp
  partials and the broadcast slot, and reads them back across the two
  barriers.  Effect-driven analyses (dataflow mem-event collection, race
  and write-only-init checks) must see that the scratch buffer is both
  read and written, and that the op must not be reordered across other
  accesses to the same allocation.

The two cross-warp ``__syncthreads()`` sites are emitted *inside* the
helper, so the op carries the explicit SMEM footprints above as its data
contract.  This composes with the accepted synchronization-boundary
model (``sync_effect(RW_SMEM_CTA_BARRIER)`` on the standalone barrier
intrinsics): when an effect analysis needs ordering-boundary visibility
between the staged partials and the broadcast, the helper's body keeps
the barrier ordering explicit and unchanged, and the scratch R/W effects
here pin the *data* footprint so the op is never treated as side-effect
free.  No barrier effect is folded into these ops: their contract is the
scratch dataflow, not a boundary event on a named operand.
"""

from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC, R_SMEM_GENERIC_SYNC, W_SMEM_GENERIC_SYNC, operand_region
from little_kernel.core.type_system import LLType, TupleType
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.backends.nv.common.language.intrin.reduce import _SUPPORTED_VALUE_TYPES, _WARP_REDUCE_MAX_BODY, _WARP_REDUCE_MIN_BODY, _WARP_REDUCE_SUM_BODY, _half_includes_for

# CUDA's maximum CTA size is 1024 threads == 32 warps, so a block reduction
# never needs more than 32 warp-partial slots.  One extra slot (index 0,
# reused after the second barrier) carries the broadcast total.
_BLOCK_REDUCE_MAX_WARPS = 32
BLOCK_REDUCE_SCRATCH_ELEMS = _BLOCK_REDUCE_MAX_WARPS + 1

_BLOCK_REDUCE_EFFECTS = (
    # Cross-lane register reads inside both embedded butterflies.
    Effect(R_RMEM_WARP_SYNC, "value"),
    # Warp partials and broadcast slot: written before/across the barriers.
    Effect(W_SMEM_GENERIC_SYNC, "smem_scratch", region=operand_region(None, BLOCK_REDUCE_SCRATCH_ELEMS)),
    # Warp partials and broadcast slot: read back across the barriers.
    Effect(R_SMEM_GENERIC_SYNC, "smem_scratch", region=operand_region(None, BLOCK_REDUCE_SCRATCH_ELEMS)),
)


def _value_type_or_raise(value_type, smem_scratch_type) -> LLType:
    """Value dtype must be shuffle-native; scratch must be a matching ptr."""
    if not isinstance(value_type, LLType) or not any(value_type == supported for supported in _SUPPORTED_VALUE_TYPES):
        raise TypeError("block reduce supports scalar value dtypes float32, float64, "
                        "int32, uint32, int64, uint64, float16, bfloat16; got "
                        f"{value_type!r}. Cast with an explicit conversion op first.")
    if not (hasattr(smem_scratch_type, "is_pointer") and smem_scratch_type.is_pointer()):
        raise TypeError("block reduce smem_scratch must be a shared-memory pointer to "
                        f"the value dtype (alloc_shared_memory / dynamic smem slice); "
                        f"got {smem_scratch_type!r}")
    inner = getattr(smem_scratch_type, "inner_type", None)
    if inner is not None and not inner == value_type:
        raise TypeError("block reduce smem_scratch element type must equal the value "
                        f"dtype; got ptr[{inner}] for value dtype {value_type!r}")
    return value_type


def _block_reduce_body(name, warp_body, warp_call, identity_fill):
    """Shared cross-warp composition helper, parameterised by combine op.

    The helper embeds the warp-level butterfly helper verbatim (as its
    first ``warp_body`` block) and calls it at both reduction stages, so
    the butterfly sequence and -- for max/min -- the IEEE-uniform ordered
    combine stay defined in exactly one place (language/intrin/reduce.py).

    ``identity_fill`` is the C++ expression used to pad lanes of warp 0
    whose index exceeds ``num_warps`` (blocks with fewer than 32 warps).
    For sum the fill is the additive identity ``T(0)``; for max/min the
    fill duplicates slot 0 -- duplicates are idempotent under a
    commutative/idempotent extrema combine, so no dtype-specific ``lowest``
    constant is needed (and the IEEE-extrema policy in the warp helper
    still governs NaN / signed-zero uniformity).
    """
    return (
        warp_body + "\n" + "template <typename T>\n" + f"__device__ __forceinline__ T {name}(T v, T* scratch) {{\n" +
        "    const int lane = threadIdx.x & 31;\n" + "    const int warp = threadIdx.x >> 5;\n" +
        "    const int nw = (blockDim.x + 31) >> 5;\n"
        # Stage 1: reduce within each warp; lane 0 of every warp stages its
        # partial into scratch[1 + warp] (slot 0 is reserved for the final
        # broadcast so a one-warp block never aliases the two roles).
        + f"    v = {warp_call}(0xffffffffu, v);\n" + "    if (lane == 0) {\n" + "        scratch[1 + warp] = v;\n" +
        "    }\n" + "    __syncthreads();\n"
        # Stage 2: warp 0 reduces the staged partials.  Lanes >= nw are
        # padded with a combine identity; duplicates are safe under the
        # idempotent extrema combines and T(0) is exact under addition.
        + "    if (warp == 0) {\n" + f"        T x = (lane < nw) ? scratch[1 + lane] : {identity_fill};\n" +
        f"        x = {warp_call}(0xffffffffu, x);\n" + "        if (lane == 0) {\n" + "            scratch[0] = x;\n" +
        "        }\n" + "    }\n" + "    __syncthreads();\n"
        # Broadcast: every thread reads the block total.  The trailing
        # barrier protects scratch reuse: without it a fast warp could
        # enter a second block_reduce on the same scratch and overwrite
        # partials before slow warps have read scratch[0].
        + "    T r = scratch[0];\n" + "    __syncthreads();\n" + "    return r;\n" + "}")


_BLOCK_REDUCE_SUM_BODY = _block_reduce_body(
    "lk_block_reduce_sum",
    _WARP_REDUCE_SUM_BODY,
    "lk_warp_reduce_sum",
    "T(0)",
)
_BLOCK_REDUCE_MAX_BODY = _block_reduce_body(
    "lk_block_reduce_max",
    _WARP_REDUCE_MAX_BODY,
    "lk_warp_reduce_max",
    "scratch[1]",
)
_BLOCK_REDUCE_MIN_BODY = _block_reduce_body(
    "lk_block_reduce_min",
    _WARP_REDUCE_MIN_BODY,
    "lk_warp_reduce_min",
    "scratch[1]",
)

# ==============================================================================
# block_reduce_max_sum -- fused max+sum block allreduce (softmax/RMSNorm pair)
#
# A staggering fraction of block-level epilogues (online softmax, RMSNorm /
# LayerNorm statistics, variance reduction, gating renormalization) need
# exactly TWO scalar statistics over the same block partials: a row max that
# stabilises the exponent, and the row sum that normalises it.  Expressing
# this as two independent allreduce calls works but performs the ENTIRE
# cross-warp choreography twice: two warp butterflies, two smem staging
# round-trips, and two full __syncthreads() barrier pairs (six barrier
# sites in total when composed with the trampoline above).  Scheduling the
# two reductions back-to-back also doubles the smem traffic on the staging
# buffer and doubles the broadcast read.
#
# ``block_reduce_max_sum`` fuses the pair into ONE schedule:
#
#   * one 32-lane warp butterfly per stat, sharing the same register reads
#     (each lane supplies its own partials directly; no partials are staged
#     through smem at the intra-warp level);
#   * one smem staging slot per warp per stat, written once;
#   * one pair of __syncthreads() that publishes warp partials AND the
#     first-warp cross-warp recombine for BOTH stats;
#   * one broadcast read per thread per stat, amortised behind the same
#     trailing barrier that protects the scratch from reuse.
#
# The emitted helper reuses the SAME ``lk_warp_reduce_max`` /
# ``lk_warp_reduce_sum`` device helpers the single-value block reductions
# (and the warp intrinsics themselves) lower to, so the numeric semantics
# are definitionally identical to sequencing ``block_reduce_max`` and
# ``block_reduce_sum`` on the same partials — including the IEEE-uniform
# extrema policy (non-NaN wins, signed-zero canonicalization) at the max
# combine, and the exact lane-associative add order at the sum combine.
#
# This is a pure scheduling/coalescing optimisation: it never changes which
# lanes participate, never reorders the butterfly steps, never changes the
# additive grouping or the extrema tie policy, and never silently drops a
# barrier that guards scratch reuse.  Its only effect is that the two
# independent results observe the same block totals bit-for-bit as the
# sequenced form, while issuing strictly fewer __syncthreads() sites and
# strictly fewer smem round-trips.
#
# Layout of the scratch buffer (which must hold
# ``BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS`` = 64 + 2 = 66 elements of the
# value dtype, compared to 33 for the single-value allreduce):
#
#   * ``scratch[0]``     -- broadcast slot for the final max;
#   * ``scratch[1]``     -- broadcast slot for the final sum;
#   * ``scratch[2 + w]`` -- staged warp partial MAX of warp ``w`` in
#                          ``[0, num_warps)`` (up to 32 warps);
#   * ``scratch[34 + w]``-- staged warp partial SUM of warp ``w``.
#
# The paired-region layout keeps the two stats in compact disjoint ranges
# so warp 0's cross-warp combine reads one coherent 32-lane vector per
# stat (max region and sum region do not interleave), avoiding the extra
# lane-strided indirection a fully interleaved layout would need on the
# second barrier path.
#
# Typing and legality contract is the same as the single-value family
# (same ``_SUPPORTED_VALUE_TYPES`` dtypes, same shared-memory pointer
# element type check, same ``blockDim.x`` multiple-of-32 upper bound,
# same uniform-control-flow requirement).  Both results are returned to
# every thread after reading the two shared broadcast slots; the final
# barrier protects those reads before scratch reuse.
#
# Declared memory effects extend the single-value set with the extra
# broadcast slot and larger staged-partials region: one R_RMEM_WARP_SYNC
# per input operand, one W_SMEM_GENERIC_SYNC covering the full 66-element
# scratch footprint, and one R_SMEM_GENERIC_SYNC covering the same region
# (so effect analyses never mistake the fused flow for side-effect free).
# ==============================================================================

# BLOCK_REDUCE_SCRATCH_ELEMS is 33 (32 warp-partial slots + 1 broadcast
# slot).  The fused max+sum form needs two disjoint 33-element regions
# (one per stat), so its scratch footprint is exactly double the
# single-value case.  Splitting into regions rather than interleaving
# keeps each stat's warp-partial range compact and lets warp 0's combine
# loop index naturally.
BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS = 2 * BLOCK_REDUCE_SCRATCH_ELEMS

_BLOCK_REDUCE_MAX_SUM_EFFECTS = (
    # Cross-lane register reads inside both embedded butterflies: one for
    # the max input, one for the sum input.
    Effect(R_RMEM_WARP_SYNC, "max_value"),
    Effect(R_RMEM_WARP_SYNC, "sum_value"),
    # Staged partials + broadcast slots for BOTH stats: written before and
    # across the two barriers, read back across them.
    Effect(W_SMEM_GENERIC_SYNC, "smem_scratch", region=operand_region(None, BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS)),
    Effect(R_SMEM_GENERIC_SYNC, "smem_scratch", region=operand_region(None, BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS)),
)


def _block_reduce_max_sum_body():
    """Fused two-stat helper: max + sum block allreduce on one smem buffer.

    The C++ body embeds the warp-level max and sum helpers verbatim (as its
    first two blocks, respectively) and calls each at both reduction
    stages, so every numeric detail (lane-symmetric IEEE-uniform extrema
    combine for max, exact butterfly add order for sum) is still defined
    in exactly one place.
    """
    return (
        _WARP_REDUCE_MAX_BODY + "\n" + _WARP_REDUCE_SUM_BODY + "\n" + "template <typename T>\n" +
        "__device__ __forceinline__ cuda::std::tuple<T, T>\n" +
        "lk_block_reduce_max_sum(T vmax, T vsum, T* scratch) {\n" + "    const int lane = threadIdx.x & 31;\n" +
        "    const int warp = threadIdx.x >> 5;\n" + "    const int nw = (blockDim.x + 31) >> 5;\n"
        # Stage 1: reduce both stats within each warp.  The two warp
        # butterflies are interleaved by the C++ compiler at no extra
        # communication cost; read-after-write hazards between the two
        # combine chains do not exist because each references its own
        # register accumulator.
        + "    vmax = lk_warp_reduce_max(0xffffffffu, vmax);\n" +
        "    vsum = lk_warp_reduce_sum(0xffffffffu, vsum);\n" + "    if (lane == 0) {\n"
        # Write partial max into [2, 2 + nw) and partial sum into
        # [34, 34 + nw).  One store per warp per stat, matching the total store count of
        # two independent allreduces; barriers are shared.
        + "        scratch[2 + warp]  = vmax;\n" + "        scratch[34 + warp] = vsum;\n" + "    }\n" +
        "    __syncthreads();\n"
        # Stage 2: warp 0 reduces both stats concurrently.  Identity fill
        # for lanes >= nw: for max, duplicate slot 2 (idempotent under the
        # max combine); for sum, use T(0) (additive identity, exact).
        + "    if (warp == 0) {\n" + "        T x_max = (lane < nw) ? scratch[2 + lane]  : scratch[2];\n" +
        "        T x_sum = (lane < nw) ? scratch[34 + lane] : T(0);\n" +
        "        x_max = lk_warp_reduce_max(0xffffffffu, x_max);\n" +
        "        x_sum = lk_warp_reduce_sum(0xffffffffu, x_sum);\n" + "        if (lane == 0) {\n" +
        "            scratch[0] = x_max;\n" + "            scratch[1] = x_sum;\n" + "        }\n" + "    }\n" +
        "    __syncthreads();\n"
        # Broadcast: every thread reads both block stats through the same
        # shared load path, amortised behind one barrier.  The trailing
        # barrier still protects scratch reuse: without it a fast warp
        # could enter another block_reduce on the same buffer and clobber
        # partials before slow warps have read scratch[0]/[1].
        + "    T out_max = scratch[0];\n" + "    T out_sum = scratch[1];\n" + "    __syncthreads();\n" +
        "    return cuda::std::make_tuple(out_max, out_sum);\n" + "}")


_BLOCK_REDUCE_MAX_SUM_BODY = _block_reduce_max_sum_body()


def _max_sum_value_type_or_raise(max_value_type, sum_value_type, smem_scratch_type):
    """Both value dtypes must match, be shuffle-native, and match scratch."""
    base = _value_type_or_raise(max_value_type, smem_scratch_type)
    if not max_value_type == sum_value_type:
        raise TypeError("block_reduce_max_sum requires max_value and sum_value to share "
                        "the same shuffle-native dtype (single staged partition layout); "
                        f"got max={max_value_type!r}, sum={sum_value_type!r}.  Cast "
                        "explicitly to a common dtype before calling.")
    return base


@builtin
class block_reduce_max_sum(BuiltinOp):
    """Fused block allreduce producing both row max and row sum in one pass.

    ``block_reduce_max_sum(max_value, sum_value, smem_scratch)`` reduces
    ``max_value`` and ``sum_value`` across all threads of the block with a
    single staging schedule and a single barrier pair, and returns the
    max AND the sum to every thread.

    Use this whenever a block epilogue needs BOTH statistics over the
    same partials (online softmax, RMSNorm/LayerNorm rsqrt(sum(x^2)/n),
    gating renormalization, variance estimation): sequencing the two
    single-value allreduces would pay for the entire cross-warp
    choreography twice (extra __syncthreads, extra smem staging and
    broadcast round-trips, extra register traffic through the butterfly).
    Fusing them is a pure scheduling/coalescing optimisation: numeric
    results are bit-identical to sequencing ``block_reduce_max`` then
    ``block_reduce_sum`` on the same partials, because both share the
    SAME warp combine helpers.

    ``smem_scratch`` must be a shared-memory pointer whose element type
    equals the common value dtype with at least
    ``BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS`` (66) elements of storage —
    exactly double the single-value scratch, one compact 33-element
    partition per stat.  Same launch-geometry and convergence contract
    as :class:`block_reduce_sum`.  The value dtypes are required to
    match because the C++ helper's return type is templated uniformly.
    """

    effects = _BLOCK_REDUCE_MAX_SUM_EFFECTS

    @staticmethod
    def return_type(max_value_type, sum_value_type, smem_scratch_type):
        common = _max_sum_value_type_or_raise(max_value_type, sum_value_type, smem_scratch_type)
        return TupleType((common, common))

    @staticmethod
    def codegen(max_value, sum_value, smem_scratch, _arg_dtypes=None):
        value_dtype = _arg_dtypes[0] if _arg_dtypes else None
        # The helper returns cuda::std::tuple<T, T>; the caller-side unpack
        # path in statement_codegen lowers `m, s = blk.reduce(...)` into
        # `auto _lk_tup_N = <call>; M m = cuda::std::get<0>(_lk_tup_N);
        # S s = cuda::std::get<1>(_lk_tup_N);`.  codegen_cuda's body-scan
        # adds <cuda/std/tuple> automatically when it sees the cuda::std
        # spelling in the helper body.
        return Builtin(
            body=_BLOCK_REDUCE_MAX_SUM_BODY,
            includes=_half_includes_for(value_dtype),
            return_val=(f"lk_block_reduce_max_sum({max_value}, {sum_value}, "
                        f"{smem_scratch})"),
        )


setattr(block_reduce_max_sum, CODEGEN_NEEDS_TYPES_ATTR, True)


def _block_reduce_codegen(body, helper_name, value, smem_scratch, _arg_dtypes):
    value_dtype = _arg_dtypes[0] if _arg_dtypes else None
    return Builtin(
        body=body,
        includes=_half_includes_for(value_dtype),
        return_val=f"{helper_name}({value}, {smem_scratch})",
    )


class _BlockReduceOp(BuiltinOp):
    """Shared typing/effect contract for the block reduction builtins."""

    effects = _BLOCK_REDUCE_EFFECTS

    @staticmethod
    def return_type(value_type, smem_scratch_type):
        return _value_type_or_raise(value_type, smem_scratch_type)


@builtin
class block_reduce_sum(_BlockReduceOp):
    """Block-wide allreduce sum: every thread receives the CTA total.

    ``block_reduce_sum(value, smem_scratch)`` reduces ``value`` across all
    threads of the block (two-stage warp butterfly + first-warp combine
    through the caller-owned shared scratch) and returns the identical
    total to every thread.  ``smem_scratch`` must be a ``ptr`` to the
    value dtype with at least ``BLOCK_REDUCE_SCRATCH_ELEMS`` (33) elements
    of shared storage; the block must be launched with ``blockDim.x`` a
    multiple of 32 and at most 1024, and every thread must execute the op
    under uniform control flow (the helper contains ``__syncthreads()``).
    """

    @staticmethod
    def codegen(value, smem_scratch, _arg_dtypes=None):
        return _block_reduce_codegen(_BLOCK_REDUCE_SUM_BODY, "lk_block_reduce_sum", value, smem_scratch, _arg_dtypes)


setattr(block_reduce_sum, CODEGEN_NEEDS_TYPES_ATTR, True)


@builtin
class block_reduce_max(_BlockReduceOp):
    """Block-wide allreduce maximum: every thread receives the CTA maximum.

    ``block_reduce_max(value, smem_scratch)`` reduces ``value`` across all
    threads of the block with the same IEEE-754-extrema policy as
    :class:`warp_reduce_max` (non-NaN wins; signed zeros canonicalize to
    ``+0.0``; infinities and integer dtypes behave naturally), applied at
    both butterfly stages -- so every thread of the block observes the same
    numeric maximum bit-for-bit.  Same scratch, launch-geometry and
    convergence contract as :class:`block_reduce_sum`.
    """

    @staticmethod
    def codegen(value, smem_scratch, _arg_dtypes=None):
        return _block_reduce_codegen(_BLOCK_REDUCE_MAX_BODY, "lk_block_reduce_max", value, smem_scratch, _arg_dtypes)


setattr(block_reduce_max, CODEGEN_NEEDS_TYPES_ATTR, True)


@builtin
class block_reduce_min(_BlockReduceOp):
    """Block-wide allreduce minimum: every thread receives the CTA minimum.

    ``block_reduce_min(value, smem_scratch)`` reduces ``value`` across all
    threads of the block with the mirrored IEEE-754-extrema policy of
    :class:`warp_reduce_min` (non-NaN wins; signed zeros canonicalize to
    ``-0.0``), applied at both butterfly stages.  Same scratch,
    launch-geometry and convergence contract as
    :class:`block_reduce_sum`.
    """

    @staticmethod
    def codegen(value, smem_scratch, _arg_dtypes=None):
        return _block_reduce_codegen(_BLOCK_REDUCE_MIN_BODY, "lk_block_reduce_min", value, smem_scratch, _arg_dtypes)


setattr(block_reduce_min, CODEGEN_NEEDS_TYPES_ATTR, True)
