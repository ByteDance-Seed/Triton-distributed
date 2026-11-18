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
"""First-class warp first-index intrinsics (argmax/argmin/first-set/last-set lane).

These are the language-level replacement for the hand-rolled
"argmax with first-index-wins tie-break" butterfly and the
"lowest-lane-with-predicate" idiom every first-index sampler kernel
(Leviathan/vLLM rejection sampler, Gumbel-max resamplers, top-1 with
deterministic tie-breaking, segment-start gather, "find the first accepting
thread" loops, compact / ballot-prefix bookkeeping, ...) ends up re-writing
by pasting ``__ballot_sync`` + ``__ffs`` or by pairing a ``shfl_xor``
butterfly over (value, lane) with a custom tie rule.  The scan/reduce
families (``language/intrin/scan.py`` and ``language/intrin/reduce.py``)
cover the *value* axis; this module covers the **index** axis so that the
result of a callee carries its originating lane deterministically.

Collection of four ops, paired on two axes:

  * Ordered extremum -> winner lane
      ``warp_argmax_lane`` / ``warp_argmin_lane``: butterfly over (value,
      lane) with a first-index-wins tie rule, returning the winning lane id
      on every participating lane.
  * Ballot -> extreme set lane
      ``warp_first_set_lane`` / ``warp_last_set_lane``: read the extreme
      set bit of the participating-lane ballot (lowest / highest cooperating
      lane with the predicate true), returning its lane id, or -1 when the
      predicate is false on every participating lane.  Composition point for
      the common "elect the first accepting lane and let it publish its
      register" pattern.

Legality / convergence contract (identical to shuffle/scan/reduce):

* Every op emits a convergent warp collective (``__shfl_xor_sync`` or
  ``__ballot_sync``).  The caller must guarantee every lane named in
  ``mask`` executes the call together under uniform control flow; the
  canonical full-warp mask is ``0xffffffff``.
* The declared effect is ``Effect(R_RMEM_WARP_SYNC, "value")`` /
  ``Effect(R_RMEM_WARP_SYNC, "predicate")``: the result is derived from
  OTHER lanes' registers, so it must not be CSE'd or reordered across a
  warp synchronization or a divergence boundary.  No GMEM/SMEM/TMEM effect;
  idempotent under replay with identical operands.
* Result type is always ``int32``: a lane id in [0, 32) or -1 (the set-lane
  pair) / a lane id in [0, 32) (the arg pair).  Arg* on a 0-masked warp is
  ill-formed for the same reason any collective on a divergent warp is.
* First-index-wins tie rule, matching the spec used by spec-decode
  rejection samplers ("argmax_x ...  first index wins on a tie") and by the
  max/min scan helpers: when two (or more) lanes hold the same extremal
  value, the LOWEST lane id is reported.  ``warp_first_set_lane`` returns
  the lowest set lane; ``warp_last_set_lane`` the highest set lane.
* Value types: any 1/2/4/8-byte scalar arithmetic type accepted by
  ``__shfl_xor_sync``.  Wider aggregates are NOT expressible -- split them
  into component lanes or stage them in shared memory.

Helpers are ``template <typename T>`` C++ emitted once per kernel module
(deduplicated by the builtin-body cache), so no dtype table is maintained
in lockstep with the type system.  The lane id inside the helpers is read
from the ``%laneid`` special register, so the intrinsic is correct under
arbitrary multi-dimensional block layouts (unlike a ``threadIdx.x & 31``
idiom).
"""

from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC
from little_kernel.core.type_system import int32
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp

_VALUE_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "value"), )
_PRED_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "predicate"), )

_LANEID_READ = ("    unsigned int lk_laneid_ = 0u;\n"
                '    asm("mov.u32 %0, %%laneid;" : "=r"(lk_laneid_));')


def _arg_butterfly_body(name, first_pick):
    """Emit the paired (value, lane) butterfly helper for one extremum.

    ``first_pick`` is the C++ comparison body that replaces ``(bv, bl)``
    with the candidate ``(ov, ol)`` iff the candidate should win.  The
    update is written to honour first-index-wins: strict extremum
    comparison always wins, and an exact-tie comparison picks the smaller
    lane id.
    """
    return ("template <typename T>\n"
            f"__device__ __forceinline__ int {name}_fn(const unsigned int lk_mask_,\n"
            "                                       const T lk_value_) {\n"
            f"{_LANEID_READ}\n"
            "    int lk_lane_ = static_cast<int>(lk_laneid_);\n"
            "    T lk_best_ = lk_value_;\n"
            "#pragma unroll\n"
            "    for (int lk_off_ = 16; lk_off_ > 0; lk_off_ >>= 1) {\n"
            "        const T lk_other_v_ = __shfl_xor_sync(lk_mask_, lk_best_, lk_off_);\n"
            "        const int lk_other_l_ = __shfl_xor_sync(lk_mask_, lk_lane_, lk_off_);\n"
            f"{first_pick}\n"
            "    }\n"
            "    return lk_lane_;\n"
            "}")


_ARGMAX_BODY = _arg_butterfly_body(
    "warp_argmax_lane",
    # strict `>` wins outright; on an exact tie keep the lower lane id.
    "        if ((lk_other_v_ > lk_best_) ||\n"
    "            (lk_other_v_ == lk_best_ && lk_other_l_ < lk_lane_)) {\n"
    "            lk_best_ = lk_other_v_;\n"
    "            lk_lane_ = lk_other_l_;\n"
    "        }",
)

_ARGMIN_BODY = _arg_butterfly_body(
    "warp_argmin_lane",
    "        if ((lk_other_v_ < lk_best_) ||\n"
    "            (lk_other_v_ == lk_best_ && lk_other_l_ < lk_lane_)) {\n"
    "            lk_best_ = lk_other_v_;\n"
    "            lk_lane_ = lk_other_l_;\n"
    "        }",
)

_PREDICATE_BALLOT_BODY = ("__device__ __forceinline__ int"
                          " warp_first_set_lane_fn(const unsigned int lk_mask_,\n"
                          "                        const int lk_pred_) {\n"
                          "    const unsigned int lk_ballot_ = __ballot_sync(lk_mask_, lk_pred_);\n"
                          "    // __ffs is 1-indexed, 0 if no bit is set -> shift to a lane id or -1.\n"
                          "    return __ffs(lk_ballot_) - 1;\n"
                          "}\n"
                          "__device__ __forceinline__ int"
                          " warp_last_set_lane_fn(const unsigned int lk_mask_,\n"
                          "                       const int lk_pred_) {\n"
                          "    const unsigned int lk_ballot_ = __ballot_sync(lk_mask_, lk_pred_);\n"
                          "    // 31 - clz(0) is 32 by CUDA's definition of __clz on 0; the mask\n"
                          "    // makes the empty-ballot case explicit so an all-false predicate\n"
                          "    // still yields -1.\n"
                          "    return lk_ballot_ == 0u ? -1 : 31 - __clz(lk_ballot_);\n"
                          "}")

# ==============================================================================
# warp_argmax_lane -- lane id of the warp-argmax, first-index-wins tie-break
# ==============================================================================


@builtin
class warp_argmax_lane(BuiltinOp):
    """Lane id of the maximum warp value, first-index-wins on tie.

    Every participating lane ``l`` receives the same ``int32`` result:
    ``min { lane : value[lane] == max(value) }``.  Composes with
    ``__shfl_sync(mask, x, winner)`` when the caller also needs the winning
    value broadcast.  Convergent warp collective -- see module docstring.
    """

    effects = _VALUE_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return int32

    @staticmethod
    def codegen(mask, value):
        return Builtin(
            body=_ARGMAX_BODY,
            includes=[],
            return_val=f"warp_argmax_lane_fn({mask}, ({value}))",
        )


# ==============================================================================
# warp_argmin_lane -- lane id of the warp-argmin, first-index-wins tie-break
# ==============================================================================


@builtin
class warp_argmin_lane(BuiltinOp):
    """Lane id of the minimum warp value, first-index-wins on tie.

    Every participating lane ``l`` receives the same ``int32`` result:
    ``min { lane : value[lane] == min(value) }``.  Convergent warp
    collective -- see module docstring.
    """

    effects = _VALUE_EFFECTS

    @staticmethod
    def return_type(mask_type, value_type):
        return int32

    @staticmethod
    def codegen(mask, value):
        return Builtin(
            body=_ARGMIN_BODY,
            includes=[],
            return_val=f"warp_argmin_lane_fn({mask}, ({value}))",
        )


# ==============================================================================
# warp_first_set_lane / warp_last_set_lane -- extreme set-lane of a ballot
# ==============================================================================


@builtin
class warp_first_set_lane(BuiltinOp):
    """Lowest participating lane whose predicate is true, else -1.

    Composition point for "elect the first accepting lane" loops: the
    caller can combine the result with ``__shfl_sync(mask, x,
    first_lane)`` to have that lane publish its register.  Convergent
    warp collective -- see module docstring.
    """

    effects = _PRED_EFFECTS

    @staticmethod
    def return_type(mask_type, predicate_type):
        return int32

    @staticmethod
    def codegen(mask, predicate):
        return Builtin(
            body=_PREDICATE_BALLOT_BODY,
            includes=[],
            return_val=f"warp_first_set_lane_fn({mask}, ({predicate}))",
        )


@builtin
class warp_last_set_lane(BuiltinOp):
    """Highest participating lane whose predicate is true, else -1.

    Composition point for "elect the last accepting lane" loops (e.g.
    right-most set bit when the lane id encodes a position inside the
    warp).  Convergent warp collective -- see module docstring.
    """

    effects = _PRED_EFFECTS

    @staticmethod
    def return_type(mask_type, predicate_type):
        return int32

    @staticmethod
    def codegen(mask, predicate):
        return Builtin(
            body=_PREDICATE_BALLOT_BODY,
            includes=[],
            return_val=f"warp_last_set_lane_fn({mask}, ({predicate}))",
        )
