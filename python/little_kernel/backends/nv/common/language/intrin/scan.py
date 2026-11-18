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
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
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
"""First-class warp prefix-scan intrinsics (inclusive/exclusive, add/max/min).

These are the language-level replacement for per-kernel hand-rolled
Hillis-Steele warp scans (the running acceptance-prefix count in speculative
decoding / rejection samplers, per-row logitsumexp-style running reductions,
cumsum offsets over ballot masks, token-order prefix bookkeeping, ...).

Before this module the compiler exposed only the raw
``language/intrin/shuffle.py`` building blocks (``shfl_up_sync`` etc.), so
every kernel author that needed a prefix had to rewrite the guarded
five-step shuffle loop -- including the ``lane >= delta`` guard, which is
easy to get wrong and silently produces over-counted prefixes.

Legality / convergence contract (declared, not just documented), identical
to the shuffle family in ``language/intrin/shuffle.py``:

* Every op emits ``__shfl_up_sync`` underneath, i.e. a convergent warp
  collective.  The kernel author must guarantee that every lane named in
  ``mask`` executes the call together under uniform control flow; the
  canonical full-warp mask is ``0xffffffff``.
* The declared memory effect is ``R_RMEM_WARP_SYNC`` on the ``value``
  operand (see little_kernel/core/memory_effects.py): the returned value may
  be derived from a DIFFERENT lane's register, so the op must not be CSE'd
  or reordered across a warp synchronization or a divergence boundary.  The
  op has NO GMEM/SMEM/TMEM effect and is pure under replay with the same
  mask/value arguments.
* The result type equals the ``value`` operand type.  Any 1/2/4/8-byte
  arithmetic type with ``+`` (add variants) or a strict ordering (max/min
  variants) works; wider aggregates are NOT expressible -- scan their
  components or keep them in shared memory.
* Scans run over the 32-lane warp but are segment-correct for arbitrary
  participation masks: a lane's prefix accumulates only contributions from
  lanes of its own mask segment (the maximal run of contiguous mask bits
  containing the lane), so partial masks such as ballot subgroups or
  half-warp masks produce exact per-segment prefixes instead of leaking
  lower-lane contributions across the segment boundary.  An aligned
  width-parameterized variant (``__shfl_up_sync`` ``width`` clip) is not
  offered here; the mask-driven software segment clip subsumes it, and
  ``shfl_up_sync_width`` remains available for hand composition.

To stay dtype-agnostic the generated helpers are C++ templates emitted once
per kernel module (deduplicated by the builtin-body cache), so no
dtype-spelling table has to be kept in sync with the type system.  The lane
index for the Hillis-Steele guard is read from the ``%laneid`` special
register inside the generated helper, so the intrinsics are correct for
arbitrary (multi-dimensional) block layouts -- unlike the common
``threadIdx.x & 31`` idiom, which is only valid for x-major 1-D warps.
"""

from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp

# Declared effect shared by the whole family: reads registers from
# arbitrary lower lanes of the same warp (same contract as shuffle.py).
_SCAN_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "value"), )

# Lane-index read used by every helper.  `%laneid` is the ISA-correct lane
# number independent of the block's dimension ordering, so the segment-guard
# arithmetic `laneid - base >= delta` behaves identically under 1-D and
# multi-D launches.
_LANEID_READ = ("    unsigned int lk_laneid = 0u;\n"
                '    asm("mov.u32 %0, %%laneid;" : "=r"(lk_laneid));')

# Segment-base computation shared by every scan helper.  A lane's scan
# segment is the maximal run of contiguous mask bits containing the lane --
# the group of lanes that actually exchange partial prefixes with it.  Lane
# ``l`` may accumulate from lane ``l - delta`` only when that source is also
# in the segment, i.e. when ``l - base >= delta`` where ``base`` is the
# segment's first lane.  The old ``l >= delta`` guard is the base==0 special
# case and silently leaks lower-segment values into partial-mask scans (e.g.
# mask 0xffff0000 gave every upper-half lane a prefix polluted by lanes
# 0-15).  ``__shfl_up_sync`` itself only clamps at lane 0 (its ``c`` operand
# is 0x1f), so the segment clip must be done in software here.  The full-warp
# canonical mask keeps the zero-overhead path (base == 0, no bit scan).
_SEGMENT_BASE_READ = ("    unsigned int lk_base = 0u;\n"
                      "    if (lk_mask != 0xffffffffu) {\n"
                      "        lk_base = lk_laneid;\n"
                      "        while (lk_base > 0u && ((lk_mask >> (lk_base - 1u)) & 1u) != 0u) {\n"
                      "            --lk_base;\n"
                      "        }\n"
                      "    }\n")


def _scan_helper(name, combine):
    """Emit the segment-guarded Hillis-Steele helper body for one operator.

    ``combine`` spells the guarded accumulation for in-segment lanes at
    distance >= delta, e.g. ``acc = acc + upstream;`` for add or a
    strict-order update for max/min.  The guard exploits
    __shfl_up_sync's own-value return for lanes whose source fell out of
    the lane's mask segment: skipping the update there keeps the prefix
    exact for arbitrary (partial) participation masks.
    """
    return ("template <typename T>\n"
            f"__device__ __forceinline__ T {name}_fn(const unsigned int lk_mask,\n"
            "                                      const T lk_value) {\n"
            f"{_LANEID_READ}\n"
            f"{_SEGMENT_BASE_READ}"
            "    T lk_acc = lk_value;\n"
            "#pragma unroll\n"
            "    for (int lk_delta = 1; lk_delta < 32; lk_delta <<= 1) {\n"
            "        const T lk_upstream = __shfl_up_sync(lk_mask, lk_acc, lk_delta);\n"
            "        if (lk_laneid - lk_base >= static_cast<unsigned int>(lk_delta)) {\n"
            f"            {combine}\n"
            "        }\n"
            "    }\n"
            "    return lk_acc;\n"
            "}")


_ADD_BODY = _scan_helper("warp_scan_inclusive_add", "lk_acc = lk_acc + lk_upstream;")

_MAX_BODY = _scan_helper(
    "warp_scan_inclusive_max",
    # Strict `>` keeps the first (lowest-lane) value on ties, matching the
    # first-index-wins tie rule used by sampling argmax kernels.
    "lk_acc = (lk_upstream > lk_acc) ? lk_upstream : lk_acc;",
)

_MIN_BODY = _scan_helper(
    "warp_scan_inclusive_min",
    "lk_acc = (lk_upstream < lk_acc) ? lk_upstream : lk_acc;",
)

# Exclusive add: shift the inclusive prefix up by one lane and re-seed the
# segment base with the additive identity.  The seed lane is ``lk_base`` (the
# first lane of the lane's mask segment), NOT absolute lane 0: with a partial
# mask such as 0xffff0000 the first participating lane is 16, and seeding at
# ``%laneid == 0`` left lane 16 holding its inclusive prefix while lane 0 --
# not even in the mask -- was the only zeroed lane.  Self-contained
# (duplicates the loop rather than referencing the inclusive helper) because
# helper bodies are emitted lazily and in call order, so a forward reference
# to a not-yet-emitted helper would leave the generated translation unit
# ill-formed.
_EXCL_ADD_BODY = ("template <typename T>\n"
                  "__device__ __forceinline__ T warp_scan_exclusive_add_fn(const unsigned int lk_mask,\n"
                  "                                      const T lk_value) {\n"
                  f"{_LANEID_READ}\n"
                  f"{_SEGMENT_BASE_READ}"
                  "    T lk_acc = lk_value;\n"
                  "#pragma unroll\n"
                  "    for (int lk_delta = 1; lk_delta < 32; lk_delta <<= 1) {\n"
                  "        const T lk_upstream = __shfl_up_sync(lk_mask, lk_acc, lk_delta);\n"
                  "        if (lk_laneid - lk_base >= static_cast<unsigned int>(lk_delta)) {\n"
                  "            lk_acc = lk_acc + lk_upstream;\n"
                  "        }\n"
                  "    }\n"
                  "    lk_acc = __shfl_up_sync(lk_mask, lk_acc, 1);\n"
                  "    if (lk_laneid == lk_base) {\n"
                  "        lk_acc = T{};  // additive identity of the element type\n"
                  "    }\n"
                  "    return lk_acc;\n"
                  "}")

# ==============================================================================
# warp_scan_inclusive_add -- Hillis-Steele inclusive prefix sum
# ==============================================================================


@builtin
class warp_scan_inclusive_add(BuiltinOp):
    """Inclusive warp prefix sum.

    Every participating lane ``l`` receives ``sum(value[l'] for l' <= l)``.
    Emits a guarded ``__shfl_up_sync`` Hillis-Steele loop via the templated
    helper ``warp_scan_inclusive_add_fn``.  Convergent warp collective --
    see module docstring.
    """

    effects = _SCAN_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return value_type

    @staticmethod
    def codegen(mask, value):
        return Builtin(
            body=_ADD_BODY,
            includes=[],
            return_val=f"warp_scan_inclusive_add_fn({mask}, ({value}))",
        )


# ==============================================================================
# warp_scan_exclusive_add -- exclusive prefix sum (lane 0 == identity)
# ==============================================================================


@builtin
class warp_scan_exclusive_add(BuiltinOp):
    """Exclusive warp prefix sum.

    Every participating lane ``l`` receives ``sum(value[l'] for l' < l)``
    over the lanes of ``l``'s mask segment; the segment's first lane
    receives the additive identity (``T{}``, value-initialized zero).
    Canonical use: rank-within-warp of set ballot bits, running offsets.
    Convergent warp collective -- see module docstring.
    """

    effects = _SCAN_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return value_type

    @staticmethod
    def codegen(mask, value):
        return Builtin(
            body=_EXCL_ADD_BODY,
            includes=[],
            return_val=f"warp_scan_exclusive_add_fn({mask}, ({value}))",
        )


# ==============================================================================
# warp_scan_inclusive_max / warp_scan_inclusive_min -- ordered prefix scans
# ==============================================================================


@builtin
class warp_scan_inclusive_max(BuiltinOp):
    """Inclusive warp prefix maximum.

    Every participating lane ``l`` receives ``max(value[l'] for l' <= l)``.
    Ties keep the lowest-lane value (strict ``>`` update), matching the
    first-index-wins convention of sampler argmax repair loops.  Convergent
    warp collective -- see module docstring.
    """

    effects = _SCAN_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return value_type

    @staticmethod
    def codegen(mask, value):
        return Builtin(
            body=_MAX_BODY,
            includes=[],
            return_val=f"warp_scan_inclusive_max_fn({mask}, ({value}))",
        )


@builtin
class warp_scan_inclusive_min(BuiltinOp):
    """Inclusive warp prefix minimum.

    Every participating lane ``l`` receives ``min(value[l'] for l' <= l)``.
    Ties keep the lowest-lane value (strict ``<`` update).  Convergent warp
    collective -- see module docstring.
    """

    effects = _SCAN_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return value_type

    @staticmethod
    def codegen(mask, value):
        return Builtin(
            body=_MIN_BODY,
            includes=[],
            return_val=f"warp_scan_inclusive_min_fn({mask}, ({value}))",
        )
