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
"""
SIMT intrinsics: thread/block/warp operations.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.core.memory_effects import Effect, R_RMEM_WARP_SYNC
from little_kernel.core.type_system import int32, uint32, void
from functools import partial

# Declared effect shared by the warp vote/match collectives below
# (ballot_sync / match_any_sync): the returned value is derived from the
# per-lane registers of EVERY lane named in the mask, exactly like a warp
# shuffle (see language/intrin/shuffle.py).  Declaring R_RMEM_WARP_SYNC pins
# the convergence contract: the op must not be CSE'd away, duplicated, or
# reordered across a warp synchronization or divergence boundary, and the
# result is pure only for replay with identical (mask, predicate/value)
# pairs.  popc/ffs are plain per-lane ALU ops and carry no effect.
_VOTE_MATCH_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "value"), )
_BALLOT_EFFECTS = (Effect(R_RMEM_WARP_SYNC, "predicate"), )

# ==============================================================================
# Thread/Block indices and dimensions
# ==============================================================================

# ==============================================================================
# SM identification
# ==============================================================================


def codegen_smid():
    body = """
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}
"""
    return Builtin(body=body, includes=[], return_val="get_smid()")


@builtin(eval_return_type=uint32, codegen_func=codegen_smid)
def smid():
    """Get the SM ID of the current thread."""
    raise RuntimeError("smid should never be called in compilation")


# ==============================================================================
# Warp-level operations
# ==============================================================================


def codegen_get_lane_idx():
    return Builtin(
        body="\n__device__ __forceinline__ uint32_t get_lane_idx() {\n"
        "    uint32_t lane_id;\n"
        '    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));\n'
        "    return lane_id;\n"
        "}\n",
        includes=[],
        return_val="get_lane_idx()",
    )


@builtin(eval_return_type=int32, codegen_func=codegen_get_lane_idx)
def get_lane_idx():
    """Get the lane index of the current thread."""
    raise RuntimeError("get_lane_idx should never be called in compilation")


def codegen_elect_one_sync():
    body = """
__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\\n"
        ".reg .pred p;\\n"
        "elect.sync _|p, 0xFFFFFFFF;\\n"
        "selp.b32 %0, 1, 0, p;\\n"
        "}\\n"
        : "=r"(pred));
    return pred != 0;
}
"""
    return Builtin(body=body, includes=[], return_val="elect_one_sync_fn()")


@builtin(eval_return_type=int32, codegen_func=codegen_elect_one_sync)
def elect_one_sync():
    """Elect one thread in the warp."""
    raise RuntimeError("elect_one_sync should not be called in compilation")


def codegen_elect_one_in_warpgroup(warp_index):
    # One typed op for "elected lane elect.sync && my warp index == warp_index".
    # The warp-index conjunction is materialized inside the same helper so the
    # typed IR carries the (elect, warp-gate) pair atomically instead of the
    # frontend composing `(threadIdx.x >> 5) == wg` with elect_one across
    # typed control flow.  Every thread of the calling block may execute the
    # op; threads outside the selected warp deterministically receive 0 and
    # exactly one thread of the selected warp receives 1.
    body = """
__device__ __forceinline__ uint32_t elect_one_in_warpgroup_fn(uint32_t warp_index) {
    const uint32_t warp_idx = threadIdx.x >> 5;
    uint32_t pred;
    asm volatile(
        "{\\n"
        ".reg .pred p;\\n"
        "elect.sync _|p, 0xFFFFFFFF;\\n"
        "selp.b32 %0, 1, 0, p;\\n"
        "}\\n"
        : "=r"(pred));
    return (warp_idx == warp_index) ? pred : 0u;
}
"""
    return Builtin(body=body, includes=[], return_val=f"elect_one_in_warpgroup_fn({warp_index})")


@builtin(eval_return_type=uint32, codegen_func=codegen_elect_one_in_warpgroup)
def elect_one_in_warpgroup(warp_index):
    """Elect a single thread within one warp of a multi-warpgroup block.

    Returns 1 on exactly one thread: the ``elect.sync``-elected lane of the
    warp whose in-CTA warp index equals ``warp_index``; every other thread
    returns 0.  This fuses the warp-index gate and the election so warpgroup
    roles (e.g. a TMA producer warpgroup) can be expressed as one typed op
    instead of the frontend materializing ``(threadIdx.x >> 5) == wg``
    separately and threading the conjunction through typed control flow.

    Convergence contract, mirroring PTX ``elect.sync``: all threads of the
    selected warp must execute the op convergently; other warps of the CTA
    may also execute it and are guaranteed to receive 0 (the op has no side
    effects beyond the predicate).  The block may contain several warpgroups;
    ``warp_index`` selects warps across the whole CTA, so a producer
    warpgroup at threads 256..383 of a 384-thread block is addressed with
    ``warp_index`` in 8..11 (typically the warpgroup's first warp).
    """
    raise RuntimeError("elect_one_in_warpgroup should not be called in compilation")


# ==============================================================================
# Warp voting and bit operations
# ==============================================================================


def codegen_ballot_sync(mask, predicate):
    return Builtin(body="", includes=[], return_val=f"__ballot_sync({mask}, {predicate})")


@builtin(eval_return_type=uint32, codegen_func=codegen_ballot_sync, effects=_BALLOT_EFFECTS)
def ballot_sync(mask, predicate):
    """Warp ballot -- returns bitmask of threads where predicate is true.

    Convergent warp collective (emits ``__ballot_sync``): every lane named
    in ``mask`` must execute the call together under uniform control flow.
    Each participating lane receives bit ``l`` set iff lane ``l`` is in
    ``mask`` and evaluated ``predicate`` true; bits for lanes outside the
    mask are zero in every participating lane's result.  Declared effect is
    ``R_RMEM_WARP_SYNC`` on ``predicate`` -- the result depends on peer
    lanes' registers, so the op must not be CSE'd or moved across a warp
    sync / divergence boundary.
    """
    raise RuntimeError("should not call ballot_sync in compilation")


def codegen_match_any_sync(mask, value):
    return Builtin(body="", includes=[], return_val=f"__match_any_sync({mask}, {value})")


@builtin(eval_return_type=uint32, codegen_func=codegen_match_any_sync, effects=_VOTE_MATCH_EFFECTS)
def match_any_sync(mask, value):
    """Warp match -- returns bitmask of threads holding the same value.

    Convergent warp collective (emits ``__match_any_sync``): every lane
    named in ``mask`` must execute the call together under uniform control
    flow.  Each participating lane receives the mask of lanes *within
    ``mask``* whose ``value`` is bitwise-equal to its own; bits for lanes
    outside ``mask`` are zero.

    Sentinel semantics (matches CUDA ``__match_any_sync`` exactly, no
    special-casing): grouping is by VALUE ONLY.  Lanes holding the same
    sentinel (e.g. ``-1`` for "inactive") match EACH OTHER and form their
    own group; they are never silently excluded.  The canonical duplicate-
    detection idiom -- initialize ``v`` to a sentinel on inactive lanes,
    ``m = match_any_sync(FULL_MASK, v)``, ``count = popc(m)`` -- is correct
    iff the sentinel value can never occur as a real payload among active
    lanes: active lanes then see only peers with the same real value, and
    each inactive (sentinel) lane's own result is discarded by its guarded
    store.  If the payload domain may contain the sentinel, pick a sentinel
    outside the domain or mask it out of the result explicitly.  ``int32``
    and ``uint32`` values compare bitwise-equal per CUDA; the two agree
    because ``__match_any_sync`` groups on the 32-bit pattern.

    Declared effect is ``R_RMEM_WARP_SYNC`` on ``value`` -- the result
    depends on peer lanes' registers, so the op must not be CSE'd or moved
    across a warp sync / divergence boundary.
    """
    raise RuntimeError("should not call match_any_sync in compilation")


def codegen_popc(x):
    return Builtin(body="", includes=[], return_val=f"__popc({x})")


@builtin(eval_return_type=int32, codegen_func=codegen_popc)
def popc(x):
    """Population count (number of set bits)."""
    raise RuntimeError("should not call popc in compilation")


def codegen_ffs(x):
    return Builtin(body="", includes=[], return_val=f"__ffs({x})")


@builtin(eval_return_type=int32, codegen_func=codegen_ffs)
def ffs(x):
    """Find first set bit (1-indexed, 0 if no bits set)."""
    raise RuntimeError("should not call ffs in compilation")


# ==============================================================================
# Grid dimensions
# ==============================================================================

# ==============================================================================
# Warpgroup register management
# ==============================================================================


def _codegen_setmaxnreg(mode, fn_name, num_registers):
    cases = "".join(f'        case {value}: asm volatile("setmaxnreg.{mode}.sync.aligned.u32 {value};\\n"); break;\n'
                    for value in range(24, 257, 8))
    body = f"\n__device__ __forceinline__ void {fn_name}(int value) {{\n    switch (value) {{\n{cases}    }}\n}}\n"
    return Builtin(body=body, includes=[], return_val=f"{fn_name}({num_registers})")


def codegen_warpgroup_reg_alloc(num_registers):
    return _codegen_setmaxnreg("inc", "warpgroup_reg_alloc_fn", num_registers)


@builtin(eval_return_type=void, codegen_func=codegen_warpgroup_reg_alloc)
def warpgroup_reg_alloc(num_registers):
    """Allocate registers for warpgroup (setmaxnreg.inc)."""
    raise RuntimeError("should not call warpgroup_reg_alloc in compilation")


def codegen_warpgroup_reg_dealloc(num_registers):
    return _codegen_setmaxnreg("dec", "warpgroup_reg_dealloc_fn", num_registers)


@builtin(eval_return_type=void, codegen_func=codegen_warpgroup_reg_dealloc)
def warpgroup_reg_dealloc(num_registers):
    """Deallocate registers for warpgroup (setmaxnreg.dec)."""
    raise RuntimeError("should not call warpgroup_reg_dealloc in compilation")


def codegen_warpgroup_reg_inc(num_registers):
    # Explicit inc name matching setmaxnreg.inc.sync: the consumer side of a
    # warp-specialized producer/consumer split raises its register ceiling
    # (e.g. inc<224> for WGMMA consumers) while the producer does dec<40>.
    return _codegen_setmaxnreg("inc", "warpgroup_reg_inc_fn", num_registers)


@builtin(eval_return_type=void, codegen_func=codegen_warpgroup_reg_inc)
def warpgroup_reg_inc(num_registers):
    """Raise the warpgroup register ceiling (setmaxnreg.inc).

    Pair with warpgroup_reg_dec on the sibling warp role: both are
    syncthreads-like warpgroup-collective, so every warp of a warpgroup must
    execute them with matching counts (legal values are 24..256 in steps of
    8)."""
    raise RuntimeError("should not call warpgroup_reg_inc in compilation")


def codegen_warpgroup_reg_dec(num_registers):
    return _codegen_setmaxnreg("dec", "warpgroup_reg_dec_fn", num_registers)


@builtin(eval_return_type=void, codegen_func=codegen_warpgroup_reg_dec)
def warpgroup_reg_dec(num_registers):
    """Lower the warpgroup register ceiling (setmaxnreg.dec).

    Producer-side counterpart of warpgroup_reg_inc; lowering the producer
    ceiling is what frees the register file for the high-register consumers
    to co-reside on the same SM in a cluster launch."""
    raise RuntimeError("should not call warpgroup_reg_dec in compilation")


# ==============================================================================
# Platform-agnostic block index/count (CUDA: blockIdx.x/gridDim.x)
# ==============================================================================
