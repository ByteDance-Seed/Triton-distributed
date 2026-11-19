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
"""Canonical SM100 role/dataflow intrinsics.

These compile-time-only operations declare precise memory effects over logical
resources for dataflow analysis and CUDA lowering.

Naming follows the PTX operation being wrapped:

* ``tma_{ld,st}_{rank}d`` moves a rank-``rank`` tensor between global memory (GMEM) and shared memory (SMEM).
* ``umma_<kind>_<sources>_<scope>`` writes a matrix-multiply accumulator to tensor memory (TMEM). ``ss`` means that both
  multiplicands come from SMEM; ``ts`` means A comes from TMEM and B from SMEM.
* ``tcgen05_cp_<lanes>x<bits>b[_warpx4]_<scope>`` copies SMEM to TMEM. The shape means ``lanes`` TMEM rows and ``bits``
  bits across the TMEM columns in each row. ``warpx4`` multicasts the source to all four warps.
* ``tmem_{ld,st}_<lanes>dp_<bits>bx<repeat>`` moves TMEM to or from registers. ``dp`` is the CUTLASS-style abbreviation
  for TMEM datapath lanes; ``xN`` is the PTX repeat count.
* ``1sm`` maps to ``cta_group::1`` and accesses only the current CTA's TMEM. ``2sm`` maps to ``cta_group::2`` and
  accesses the TMEM of the current and peer CTAs; a single elected thread in the CTA pair issues the instruction.

For example, a two-CTA block-scaled MMA first copies scale factors from SMEM to both CTAs' TMEM, then consumes those
TMEM addresses in the matching ``*_ss_2sm`` UMMA::

    elected = ll.elect_one()
    if elected == 1:
        sf_desc = ri.make_umma_smem_desc(sf_smem, (0, 0))
        ri.tcgen05_cp_32x128b_warpx4_2sm(sf_tmem, ll.uint32(sf_tmem), sf_desc)
        ri.umma_mxfp_e2m1_e2m1_ss_2sm(..., sf_tmem, ll.uint32(sf_tmem), ..., out_shape=(256, 256))
"""

from __future__ import annotations

from little_kernel.core.memory_effects import R_SMEM_TCGEN05_COMMIT, R_SMEM_TMA_BULK, R_TMEM_TCGEN05_COMMIT, R_TMEM_TCGEN05_WAIT_LD, RW_TMEM_TCGEN05_COMMIT, W_SMEM_TMA_TX, W_TMEM_TCGEN05_COMMIT, W_TMEM_TCGEN05_WAIT_ST, MS, effect, operand_region, r, w
from little_kernel.core.type_system import bfloat16, float32, uint8, uint32, uint64, void
from little_kernel.language.builtin_base import Builtin, MEMORY_EFFECTS_ATTR, ORIGINAL_PY_FUNC_ATTR, builtin as _builtin
from little_kernel.backends.nv.common.language.intrin.memory import gmem_ld_f32x4, gmem_st_f32x4


def builtin(*args, **kwargs):
    """Register logical dataflow operations without shadowing native intrinsics."""
    return _builtin(*args, namespace="dataflow", **kwargs)


from little_kernel.backends.nv.sm100.language.intrin.umma import codegen_tcgen05_cp_32x128b_warpx4_1sm, codegen_tcgen05_cp_32x128b_warpx4_2sm


def _compile_time_only(name: str) -> None:
    raise RuntimeError(f"{name} is compile-time only")


class Pipeline:
    """Compiler-managed producer/consumer pipeline backed by mbarriers and circular stage state."""

    def producer_acquire(self) -> None:
        """Wait until the current producer stage is available for reuse."""
        _compile_time_only("Pipeline.producer_acquire")

    def producer_prepare_transaction(self) -> None:
        """Arm the current stage's mbarrier with the compiler-planned TMA transaction byte count."""
        _compile_time_only("Pipeline.producer_prepare_transaction")

    def producer_get_barrier(self):
        """Return the current producer stage's mbarrier for an asynchronous transaction."""
        _compile_time_only("Pipeline.producer_get_barrier")

    def producer_stage(self) -> int:
        """Return the circular stage index currently owned by the producer."""
        _compile_time_only("Pipeline.producer_stage")

    def producer_commit(self) -> None:
        """Publish completion for non-TMA producer work in the current stage."""
        _compile_time_only("Pipeline.producer_commit")

    def producer_tail(self) -> None:
        """Drain outstanding producer stages before the role exits or storage is reused."""
        _compile_time_only("Pipeline.producer_tail")

    def consumer_wait(self) -> None:
        """Wait until producer work for the current consumer stage is complete."""
        _compile_time_only("Pipeline.consumer_wait")

    def consumer_stage(self) -> int:
        """Return the circular stage index currently owned by the consumer."""
        _compile_time_only("Pipeline.consumer_stage")

    def consumer_release(self) -> None:
        """Release the current stage so the producer may reuse it."""
        _compile_time_only("Pipeline.consumer_release")


# --- Static launch topology ----------------------------------------------------


@builtin(eval_return_type=uint32)
def cluster_coord(axis):
    """Physical CTA coordinate on literal axis ``"x"``, ``"y"``, or ``"z"``."""
    _compile_time_only("cluster_coord")


@builtin(eval_return_type=uint32)
def group_coord(axis):
    """CTA-group coordinate on literal group-grid axis ``"x"``, ``"y"``, or ``"z"``."""
    _compile_time_only("group_coord")


@builtin(eval_return_type=uint32)
def rank_in_group():
    """Linear participant rank within the kernel's declared CTA-group shape."""
    _compile_time_only("rank_in_group")


# --- TMA ld / UMMA / TMEM / SMEM (lowered to ll.* before codegen) ---------------


@builtin(
    eval_return_type=void,
    effects=(effect(W_SMEM_TMA_TX, "mem"), ),
    cta_group=2,
)
def tma_ld_2d(mem, desc, gmem_coord, plan=None, *, smem_coord=(0, 0), shape):
    """Compiler-planned 2D TMA load into local staged SMEM.

    ``mem`` is the logical local-CTA destination, ``desc`` is its 2D global
    tensor map, ``gmem_coord`` is the ``(c0, c1)`` group-tile base
    before any participant offset. ``plan=None`` selects a single-CTA,
    non-multicast transfer with no coordinate sharding when the kernel has no
    explicit topology. A named :class:`TMATransferPlan` configures clustered
    lowering. ``smem_coord`` and ``shape`` describe the logical destination
    region in ``mem``; lowering chooses the physical TMA boxes.
    """
    _compile_time_only("tma_ld_2d")


@builtin(eval_return_type=void, effects=(effect(W_SMEM_TMA_TX, "mem"), ), cta_group=2)
def tma_ld_3d(mem, desc, gmem_coord, plan=None, *, smem_coord=(0, 0), shape):
    """Compiler-planned 3D TMA load into local staged SMEM.

    ``mem`` is the logical local-CTA destination, ``desc`` is its 3D global
    tensor map, ``gmem_coord`` is the ``(c0, c1, c2)`` group-tile base
    before any participant offset. ``plan=None`` selects the default
    single-CTA transfer for kernels without explicit topology. ``smem_coord``
    and ``shape`` describe the logical destination region in ``mem``. No
    peer-SMEM destination or user-authored mask is accepted by this API.
    """
    _compile_time_only("tma_ld_3d")


@builtin(
    eval_return_type=void,
    effects=(effect(W_SMEM_TMA_TX, "mem", region=operand_region(offset_default=0)), ),
)
def tma_ld_1d(mem, src, num_bytes):
    """Copy ``num_bytes`` from raw GMEM pointer ``src`` to logical SMEM ``mem``.

    This is a bulk G2S copy, not a tensor-map operation; use ``tma_ld_2d`` or ``tma_ld_3d`` for tiled descriptors.
    """
    _compile_time_only("tma_ld_1d")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                    "d_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
    cta_group=2,
)
def umma_f16_ss_2sm(d_mem, tmem_d, desc_a, desc_b, accum, *, out_shape):
    """Issue one two-CTA ``kind::f16`` SS UMMA.

    ``f16`` names the PTX family, which accepts FP16/BF16 multiplicands. ``ss`` places both A and B in SMEM;
    ``2sm`` shares the operation across the current and peer CTAs. ``tmem_d`` addresses the FP32 TMEM accumulator and
    ``out_shape=(M, N)`` fixes its logical tile. ``accum=0`` computes ``D = A @ B``; nonzero computes ``D += A @ B``.
    """
    _compile_time_only("umma_f16_ss_2sm")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "d_mem"), effect(R_SMEM_TCGEN05_COMMIT,
                                                            "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
    cta_group=2,
)
def umma_f16_ss_2sm_init(d_mem, tmem_d, desc_a, desc_b, *, out_shape):
    """Initialize an FP32 TMEM tile with a two-CTA SS ``kind::f16`` MMA: ``D = A @ B``.

    Use this for the first K tile when A and B are in SMEM; subsequent K tiles should use the matching ``_accum`` form.
    """
    _compile_time_only("umma_f16_ss_2sm_init")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                    "d_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
    cta_group=2,
)
def umma_f16_ss_2sm_accum(d_mem, tmem_d, desc_a, desc_b, *, out_shape):
    """Accumulate a two-CTA SS ``kind::f16`` MMA into an existing FP32 TMEM tile: ``D += A @ B``."""
    _compile_time_only("umma_f16_ss_2sm_accum")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "d_mem"), effect(R_SMEM_TCGEN05_COMMIT,
                                                            "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
)
def umma_f16_ss_1sm_init(d_mem, tmem_d, desc_a, desc_b, *, out_shape):
    """Initialize an FP32 TMEM tile with a one-CTA SS ``kind::f16`` MMA: ``D = A @ B``.

    ``ss`` means that both operand descriptors address SMEM; ``1sm`` confines the result to the current CTA's TMEM.
    """
    _compile_time_only("umma_f16_ss_1sm_init")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                    "d_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
)
def umma_f16_ss_1sm_accum(d_mem, tmem_d, desc_a, desc_b, *, out_shape):
    """Accumulate a one-CTA SS ``kind::f16`` MMA into an existing FP32 TMEM tile: ``D += A @ B``."""
    _compile_time_only("umma_f16_ss_1sm_accum")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "d_mem"), effect(R_TMEM_TCGEN05_COMMIT,
                                                            "a_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
)
def umma_f16_ts_1sm_init(d_mem, a_mem, tmem_d, tmem_a, desc_b, *, out_shape):
    """Initialize an FP32 TMEM tile with a one-CTA TS ``kind::f16`` MMA: ``D = A @ B``.

    ``ts`` means A is read from ``tmem_a`` in TMEM while B is read through the SMEM descriptor ``desc_b``.
    """
    _compile_time_only("umma_f16_ts_1sm_init")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "d_mem"), effect(R_TMEM_TCGEN05_COMMIT,
                                                             "a_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
)
def umma_f16_ts_1sm_accum(d_mem, a_mem, tmem_d, tmem_a, desc_b, *, out_shape):
    """Accumulate a one-CTA TS ``kind::f16`` MMA into an existing FP32 TMEM tile: ``D += A @ B``."""
    _compile_time_only("umma_f16_ts_1sm_accum")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                    "d_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
    cta_group=2,
)
def umma_f16_ss_2sm_kblock(d_mem, desc_a, desc_b, k, *, out_shape, zero_init=False, tmem_d=None):
    """Expand a two-CTA SS ``kind::f16`` MMA over ``k`` logical elements.

    ``k`` is a compile-time positive multiple of 16. Lowering emits one K=16 UMMA per slice and advances both SMEM
    descriptors. ``zero_init=True`` initializes D on the first slice; otherwise every slice accumulates into D.
    ``tmem_d`` overrides D's physical TMEM address when writing a subtile; by default it is derived from ``d_mem``.
    """
    _compile_time_only("umma_f16_ss_2sm_kblock")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT,
                    "d_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
)
def umma_f16_ss_1sm_kblock(d_mem, desc_a, desc_b, k, *, out_shape, zero_init=False, tmem_d=None):
    """Expand a one-CTA SS ``kind::f16`` MMA over ``k`` logical elements.

    ``k`` is a compile-time positive multiple of 16. Lowering emits one K=16 UMMA per slice and advances both SMEM
    descriptors. ``zero_init=True`` initializes D on the first slice; otherwise every slice accumulates into D.
    ``tmem_d`` overrides D's physical TMEM address when writing a subtile; by default it is derived from ``d_mem``.
    """
    _compile_time_only("umma_f16_ss_1sm_kblock")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "tmem_d"), effect(R_TMEM_TCGEN05_COMMIT,
                                                              "tmem_a"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
)
def umma_f16_ts_1sm_kblock(
    tmem_d,
    tmem_a,
    desc_b,
    k,
    *,
    out_shape,
    zero_init=False,
    negate_a=False,
    negate_b=False,
):
    """Expand a one-CTA TS ``kind::f16`` MMA over ``k`` logical elements.

    Computes ``D = A @ B`` when ``zero_init=True`` and otherwise ``D += A @ B``. ``tmem_d`` is the FP32 TMEM output,
    ``tmem_a`` is the BF16/F16 TMEM input, and ``desc_b`` identifies the BF16/F16 SMEM input. Each K=16 slice advances
    A by eight TMEM columns and advances B's descriptor. ``negate_a`` and ``negate_b`` use the corresponding IDESC
    input-negate bits without materializing a negated operand.
    """
    _compile_time_only("umma_f16_ts_1sm_kblock")


@builtin(
    eval_return_type=void,
    effects=(effect(RW_TMEM_TCGEN05_COMMIT, "tmem_d"), effect(R_TMEM_TCGEN05_COMMIT,
                                                              "tmem_a"), effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem")),
    cta_group=2,
)
def umma_f16_ts_2sm_kblock(tmem_d, tmem_a, desc_b, k, *, out_shape, zero_init=False):
    """Expand a two-CTA TS ``kind::f16`` MMA over ``k`` logical elements.

    A is read from TMEM, B from SMEM, and D is distributed across the CTA pair's TMEM. Each K=16 slice advances both
    inputs; ``zero_init=True`` initializes D on the first slice.
    """
    _compile_time_only("umma_f16_ts_2sm_kblock")


@builtin(eval_return_type=uint64)
def make_umma_smem_desc(mem, coord):
    """Build an SM100 UMMA SMEM descriptor anchored at logical element coordinate ``mem[coord]``.

    The compiler derives swizzle, major mode, strides, and the physical stage address from ``mem``. Use this descriptor
    as an SS/TS UMMA operand or as the source of ``tcgen05_cp``; do not manually encode layout fields at the call site.
    """
    _compile_time_only("make_umma_smem_desc")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "dst_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc.mem")),
)
def tcgen05_cp_128x256b_1sm(dst_mem, tmem_dst, desc):
    """Copy a ``128x256b`` tile from SMEM to the current CTA's TMEM.

    ``128`` is the number of TMEM lanes (rows), and ``256b`` is the number of bits copied across columns per lane.
    ``dst_mem`` tracks the logical TMEM write, ``tmem_dst`` is its physical TMEM address, and ``desc`` describes SMEM.
    Use this shape to stage a full-width TMEM operand for a one-CTA TS UMMA.
    """
    _compile_time_only("tcgen05_cp_128x256b_1sm")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "dst_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc.mem")),
    cta_group=2,
)
def tcgen05_cp_128x256b_2sm(dst_mem, tmem_dst, desc):
    """Copy a ``128x256b`` tile from SMEM to both CTAs' TMEM in a CTA pair.

    The shape covers 128 TMEM lanes and 256 bits per lane. ``2sm`` maps to ``cta_group::2``: one elected thread issues
    the copy, and the destination covers the current and peer CTAs' TMEM. Use it to stage a two-CTA TS UMMA operand.
    """
    _compile_time_only("tcgen05_cp_128x256b_2sm")


@builtin(
    eval_return_type=void,
    effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "mem", region=operand_region("col", "regs")), ),
)
def tmem_ld_32dp_32bxn(mem, col, *regs):
    """Collectively load ``32dp x 32b x N`` from TMEM into registers.

    ``32dp`` means 32 TMEM datapath lanes, ``32b`` is the base width per lane, and the register count selects PTX repeat
    ``N`` from 1, 2, 4, 8, 16, or 32. Pass scalar outputs or ``(local_reg_array, N)``. ``mem`` is the logical TMEM read;
    ``col`` is the physical TMEM address. Use this for a 32-lane accumulator fragment.
    """
    _compile_time_only("tmem_ld_32dp_32bxn")


@builtin(
    eval_return_type=void,
    effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "mem", region=operand_region("col", "regs")), ),
)
def tmem_ld_16dp_32bxn(mem, col, *regs):
    """Collectively load PTX ``16x32bx2.xN`` data from TMEM into registers.

    ``16dp`` accesses 16 TMEM lanes. ``32bx2`` performs two 32-bit accesses per lane, at ``col`` and ``col + N``; the
    register count selects repeat ``N`` from 1 through 32. Use this split-lane layout for an M=64 UMMA accumulator.
    """
    _compile_time_only("tmem_ld_16dp_32bxn")


@builtin(eval_return_type=void, effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "mem"), ))
def tmem_ld_fence(mem) -> None:
    """Wait for prior ``tcgen05.ld`` operations and complete the logical TMEM read lifetime."""
    _compile_time_only("tmem_ld_fence")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_WAIT_ST, "mem", region=operand_region("col", "regs")), ),
)
def tmem_st_32dp_32bxn(mem, col, *regs):
    """Collectively store registers as ``32dp x 32b x N`` into TMEM.

    ``32dp`` means 32 TMEM lanes, ``32b`` is the base width per lane, and the register count selects repeat ``N`` from
    1, 2, 4, 8, 16, or 32. ``mem`` tracks the logical write and ``col`` is the physical TMEM destination address.
    """
    _compile_time_only("tmem_st_32dp_32bxn")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_WAIT_ST, "mem", region=operand_region("col", "regs")), ),
)
def tmem_st_16dp_32bxn(mem, col, *regs):
    """Collectively store registers with the PTX ``16x32bx2.xN`` split-lane layout.

    The store writes two 32-bit regions across 16 TMEM lanes, based at ``col`` and ``col + N``. Use it when producing
    the M=64 layout consumed by the matching ``tmem_ld_16dp_32bxn`` operation.
    """
    _compile_time_only("tmem_st_16dp_32bxn")


@builtin(
    eval_return_type=void,
    effects=(effect(W_TMEM_TCGEN05_WAIT_ST, "mem", region=operand_region("col", 8)), ),
)
def tmem_st_16dp_128bx2(mem, col, *regs):
    """Store a 16-datapath by 128-bit fragment with PTX repeat 2.

    The four registers per thread are the native destination of a transposed ``ldmatrix.x4`` BF16 fragment. The
    instruction writes 16 TMEM datapaths and eight packed-BF16 columns, so no warp shuffle is needed between the SMEM
    load and this store.
    """
    _compile_time_only("tmem_st_16dp_128bx2")


@builtin(eval_return_type=void, effects=())
def tmem_st_fence() -> None:
    """Wait for prior asynchronous ``tcgen05.st`` operations before reusing their source registers."""
    _compile_time_only("tmem_st_fence")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ))
def smem_st_128(mem, addr, v0, v1, v2, v3):
    """Store four 32-bit registers to byte address ``addr`` in logical SMEM ``mem`` with one ``st.shared.v4.b32``."""
    _compile_time_only("smem_st_128")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "mem"), ))
def smem_ld_128(mem, addr, *regs):
    """Load byte address ``addr`` in logical SMEM ``mem`` into exactly four 32-bit registers."""
    _compile_time_only("smem_ld_128")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ))
def stmatrix_x4_trans(mem, addr, v0, v1, v2, v3):
    """Collectively store four registers per thread as a transposed 16x16 B16 fragment at SMEM byte address ``addr``."""
    _compile_time_only("stmatrix_x4_trans")


@builtin(eval_return_type=void, effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "mem", region=operand_region("col", 16)), ))
def tmem_ld_16dp_256bx2(mem, col, *regs):
    """Collectively load a ``16dp x 256b x2`` TMEM fragment into eight registers per thread.

    ``16dp`` means 16 TMEM lanes, ``256b`` is the base width per lane, and ``x2`` repeats that width twice. ``mem``
    tracks the logical read and ``col`` is the physical TMEM address. Use this when eight registers per thread are
    preferable to multiple narrower loads, for example while reading a wide M=64 accumulator fragment.
    """
    _compile_time_only("tmem_ld_16dp_256bx2")


@builtin(eval_return_type=void, effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "mem", region=operand_region("col", 32)), ))
def tmem_ld_16dp_256bx4(mem, col, *regs):
    """Collectively load a ``16dp x 256b x4`` TMEM fragment into sixteen registers per thread.

    ``x4`` covers four adjacent 256-bit repeats from ``col``. Use it to amortize TMEM load fences when the consumer
    processes at least sixteen registers per thread before the next synchronization point.
    """
    _compile_time_only("tmem_ld_16dp_256bx4")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "mem"), ))
def ldmatrix_x1(mem, index, regs):
    """Collectively load one 8x8 BF16 SMEM fragment at element ``index`` into one register per thread."""
    _compile_time_only("ldmatrix_x1")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "mem"), ))
def ldmatrix_x1_trans(mem, index, regs):
    """Collectively load and transpose one 8x8 BF16 SMEM fragment into one register per thread."""
    _compile_time_only("ldmatrix_x1_trans")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "mem"), ))
def ldmatrix_x4(mem, index, regs):
    """Collectively load a 16x16 BF16 SMEM fragment at element ``index`` into four registers per thread."""
    _compile_time_only("ldmatrix_x4")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "mem"), ))
def ldmatrix_x4_trans(mem, index, regs):
    """Collectively load and transpose a 16x16 BF16 SMEM fragment into four registers per thread."""
    _compile_time_only("ldmatrix_x4_trans")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ))
def stmatrix_x4_regs(mem, index, regs):
    """Collectively store four registers per thread as a 16x16 BF16 fragment at SMEM element ``index``."""
    _compile_time_only("stmatrix_x4_regs")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ))
def stmatrix_x1_reg(mem, index, reg):
    """Collectively store one register per thread as an 8x8 BF16 SMEM fragment at element ``index``."""
    _compile_time_only("stmatrix_x1_reg")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "src_mem"), w(MS.SMEM, "dst_mem")))
def copy_bf16x16_linear_to_kmajor(src_mem, dst_mem, src_index, dst_base, row, group):
    """Copy 16 contiguous BF16 elements from ``src_mem[src_index:]`` into a 128-byte-swizzled K-major SMEM row.

    ``dst_base`` is a shared-memory byte address, ``row`` is the logical destination row, and ``group`` selects the
    first 16-byte group in that row. ``src_mem`` and ``dst_mem`` provide compiler-visible memory effects.
    """
    _compile_time_only("copy_bf16x16_linear_to_kmajor")


@builtin(eval_return_type=void, effects=(r(MS.SMEM, "src_mem"), w(MS.SMEM, "dst_mem"), r(MS.SMEM, "scales")))
def copy_bf16x16_linear_to_kmajor_scaled(src_mem, dst_mem, scales, src_index, dst_base, row, col, scale):
    """Scale and copy 16 contiguous BF16 elements into a 128-byte-swizzled K-major SMEM row.

    Element ``i`` is multiplied by ``scale * scales[col + i]`` before BF16 rounding. ``dst_base`` is a shared-memory
    byte address; ``row`` and ``col`` identify the logical destination coordinate.
    """
    _compile_time_only("copy_bf16x16_linear_to_kmajor_scaled")


def _codegen_smem_st_f32(mem, index, value):
    return Builtin(body="", includes=[], return_val=f"{mem}[{index}] = {value}")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_st_f32)
def smem_st_f32(mem, index, value):
    """Store ``value`` at float32 element ``index`` in logical SMEM ``mem``."""
    _compile_time_only("smem_st_f32")


def _codegen_smem_ld_f32(mem, index):
    return Builtin(body="", includes=[], return_val=f"{mem}[{index}]")


@builtin(eval_return_type=float32, effects=(r(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_ld_f32)
def smem_ld_f32(mem, index):
    """Load the float32 value at element ``index`` from logical SMEM ``mem``."""
    _compile_time_only("smem_ld_f32")


def _codegen_smem_ld_f32x2(mem, index):
    return Builtin(body="", includes=[], return_val=f"*reinterpret_cast<const uint64_t*>({mem} + {index})")


@builtin(eval_return_type=uint64, effects=(r(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_ld_f32x2)
def smem_ld_f32x2(mem, index):
    """Load two aligned consecutive float32 values as one packed FP32x2 value."""
    _compile_time_only("smem_ld_f32x2")


def _codegen_smem_st_bf16(mem, index, value):
    return Builtin(body="", includes=[], return_val=f"{mem}[{index}] = {value}")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_st_bf16)
def smem_st_bf16(mem, index, value):
    """Store ``value`` at BF16 element ``index`` in logical SMEM ``mem``."""
    _compile_time_only("smem_st_bf16")


def _codegen_smem_ld_bf16(mem, index):
    return Builtin(body="", includes=[], return_val=f"{mem}[{index}]")


@builtin(eval_return_type=bfloat16, effects=(r(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_ld_bf16)
def smem_ld_bf16(mem, index):
    """Load the BF16 value at element ``index`` from logical SMEM ``mem``."""
    _compile_time_only("smem_ld_bf16")


def _k_major_index(rows, row, col):
    return (f"(((({col}) >> 6) * ({rows}) + ({row})) * 64 + "
            f"((((({col}) & 63) >> 3) ^ (({row}) & 7)) * 8) + (({col}) & 7))")


def _codegen_smem_st_bf16_k_major(mem, rows, row, col, value):
    return Builtin(body="", includes=[], return_val=f"{mem}[{_k_major_index(rows, row, col)}] = {value}")


@builtin(eval_return_type=void, effects=(w(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_st_bf16_k_major)
def smem_st_bf16_k_major(mem, rows, row, col, value):
    """Store one BF16 value at logical ``(row, col)`` in canonical K-major SMEM.

    ``rows`` is the tile's row extent. Columns are grouped in 64-element atoms and 8-element subgroups are XOR-swizzled
    by the low three row bits.
    """
    _compile_time_only("smem_st_bf16_k_major")


def _codegen_smem_ld_bf16_k_major(mem, rows, row, col):
    return Builtin(body="", includes=[], return_val=f"{mem}[{_k_major_index(rows, row, col)}]")


@builtin(eval_return_type=bfloat16, effects=(r(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_ld_bf16_k_major)
def smem_ld_bf16_k_major(mem, rows, row, col):
    """Load one BF16 value at logical ``(row, col)`` from canonical K-major SMEM.

    ``rows`` is the tile's row extent; the helper applies the same 64-column, row-XOR layout as
    ``smem_st_bf16_k_major``.
    """
    _compile_time_only("smem_ld_bf16_k_major")


def _codegen_smem_ld_u8(mem, index):
    return Builtin(body="", includes=[], return_val=f"{mem}[{index}]")


@builtin(eval_return_type=uint8, effects=(r(MS.SMEM, "mem"), ), codegen_func=_codegen_smem_ld_u8)
def smem_ld_u8(mem, index):
    """Load the uint8 value at element ``index`` from logical SMEM ``mem``."""
    _compile_time_only("smem_ld_u8")


def _codegen_bf16_from_bits(bits):
    return Builtin(body="", includes=[], return_val=f"__nv_bfloat16_raw(static_cast<uint16_t>({bits}))")


@builtin(eval_return_type=bfloat16, codegen_func=_codegen_bf16_from_bits)
def bf16_from_bits(bits):
    """Reinterpret the low 16 bits of ``bits`` as BF16 without numeric conversion."""
    _compile_time_only("bf16_from_bits")


@builtin(
    eval_return_type=void,
    effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "src_mem", region=operand_region("src_col", 32)), w(MS.SMEM, "dst_mem")),
)
def tmem_ld_16dp_f32_to_smem_bf16_x16(
    src_mem,
    dst_mem,
    src_col,
    dst_base,
    dst_row,
    dst_group_base,
    dst_half,
    scale,
):
    """Load 16 FP32 values from a 16-datapath TMEM layout, scale them, and store BF16 to swizzled SMEM.

    This fused epilogue helper uses ``dst_row``, ``dst_group_base``, and ``dst_half`` to select two adjacent 16-byte
    groups in a 128-byte-swizzled destination row. Use it for one half of an M=64 accumulator fragment.
    """
    _compile_time_only("tmem_ld_16dp_f32_to_smem_bf16_x16")


@builtin(
    eval_return_type=void,
    effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "src_mem", region=operand_region("src_col", 8)), w(MS.SMEM, "dst_mem")),
)
def tmem_ld_32dp_f32_to_smem_bf16_x8(src_mem, dst_mem, src_col, dst_addr):
    """Load eight FP32 values from a 32-datapath TMEM layout, round to BF16, and store 128 bits to SMEM.

    ``src_col`` is the TMEM address and ``dst_addr`` is the SMEM byte address. Use this fused helper for an unscaled
    eight-element epilogue vector.
    """
    _compile_time_only("tmem_ld_32dp_f32_to_smem_bf16_x8")


@builtin(
    eval_return_type=void,
    effects=(effect(R_TMEM_TCGEN05_WAIT_LD, "src_mem", region=operand_region("src_col", 32)), w(MS.SMEM, "dst_mem")),
)
def tmem_ld_32dp_f32_to_smem_bf16_x32(
    src_mem,
    dst_mem,
    src_col,
    dst_base,
    dst_row,
    dst_group_base,
    scale,
):
    """Load 32 FP32 values from a 32-datapath TMEM layout, scale them, and store BF16 to swizzled SMEM.

    ``dst_row`` and ``dst_group_base`` select four adjacent 16-byte groups in a 128-byte-swizzled destination row. Use
    this fused helper when an epilogue needs one scalar scale for a 32-element accumulator vector.
    """
    _compile_time_only("tmem_ld_32dp_f32_to_smem_bf16_x32")


@builtin(
    eval_return_type=void,
    effects=(effect(R_SMEM_TMA_BULK, "mem"), ),
)
def tma_st_2d(mem, desc, gmem_coord, *, smem_coord=(0, 0), shape):
    """TMA 2D st from logical ``mem`` to global.

    ``gmem_coord`` is the ``(c0, c1)`` tensor-map coordinate. ``smem_coord``
    and ``shape`` describe the logical source region in ``mem``.
    """
    _compile_time_only("tma_st_2d")


@builtin(
    eval_return_type=void,
    effects=(effect(R_SMEM_TMA_BULK, "mem"), ),
)
def tma_st_3d(mem, desc, gmem_coord, *, smem_coord=(0, 0), shape):
    """Store a logical SMEM region through a rank-3 tensor map to GMEM.

    ``gmem_coord`` is the rank-3 tensor-map coordinate; ``smem_coord`` and ``shape`` identify the rank-2 source tile in
    ``mem``. The compiler chooses the physical TMA boxes for the logical region.
    """
    _compile_time_only("tma_st_3d")


@builtin(
    eval_return_type=void,
    effects=(effect(R_SMEM_TMA_BULK, "mem", region=operand_region(offset_default=0)), ),
)
def tma_st_1d(mem, dst, num_bytes):
    """Copy ``num_bytes`` from logical SMEM ``mem`` to raw GMEM pointer ``dst`` with a bulk S2G operation."""
    _compile_time_only("tma_st_1d")


# ── Blockscaled UMMA (MXFP8 / NVFP4 / MXFP4) ──


@builtin(
    eval_return_type=void,
    codegen_func=codegen_tcgen05_cp_32x128b_warpx4_1sm,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "dst_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc.mem")),
)
def tcgen05_cp_32x128b_warpx4_1sm(dst_mem, tmem_dst, desc):
    """Copy a ``32x128b`` SMEM tile into the current CTA's TMEM and multicast it across four warps.

    ``32`` means 32 TMEM lanes and ``128b`` means 128 bits across columns per lane. PTX requires ``warpx4`` for this
    shape; it gives each of the four warps the same source data. ``1sm`` targets only the current CTA. ``dst_mem``
    tracks the logical TMEM write, ``tmem_dst`` is its physical address, and ``desc`` describes the SMEM source. A
    common use is staging scale factors immediately before a one-CTA block-scaled UMMA.
    """
    _compile_time_only("tcgen05_cp_32x128b_warpx4_1sm")


@builtin(
    eval_return_type=void,
    codegen_func=codegen_tcgen05_cp_32x128b_warpx4_2sm,
    effects=(effect(W_TMEM_TCGEN05_COMMIT, "dst_mem"), effect(R_SMEM_TCGEN05_COMMIT, "desc.mem")),
    cta_group=2,
)
def tcgen05_cp_32x128b_warpx4_2sm(dst_mem, tmem_dst, desc):
    """Copy a ``32x128b`` SMEM tile into a CTA pair's TMEM and multicast it across four warps.

    ``32`` means 32 TMEM lanes and ``128b`` means 128 bits across columns per lane. PTX requires ``warpx4`` for this
    shape; it gives each of the four warps the same source data. ``2sm`` writes the current and peer CTAs' TMEM from one
    elected issuing thread. Use it to stage shared scale factors before a matching two-CTA block-scaled UMMA.
    """
    _compile_time_only("tcgen05_cp_32x128b_warpx4_2sm")


def _blockscaled_effects():
    return (
        effect(RW_TMEM_TCGEN05_COMMIT, "c_mem"),
        effect(R_SMEM_TCGEN05_COMMIT, "desc_a.mem"),
        effect(R_SMEM_TCGEN05_COMMIT, "desc_b.mem"),
        effect(R_TMEM_TCGEN05_COMMIT, "sfa_mem"),
        effect(R_TMEM_TCGEN05_COMMIT, "sfb_mem"),
    )


@builtin(eval_return_type=void, effects=_blockscaled_effects())
def umma_mxfp_e2m1_e2m1_ss_1sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a one-CTA MXFP4 block-scaled SS UMMA with E2M1 A/B and E8M0 scale factors.

    ``ss`` means A and B come from SMEM descriptors; SFA/SFB come from TMEM and scale blocks of 32 K elements. The
    instruction consumes K=64 and writes an ``out_shape=(M, N)`` FP32 TMEM tile. ``accum=0`` initializes C; nonzero adds
    into C. ``tsfa`` and ``tsfb`` contain the TMEM address in bits 0:29 and the scale-factor sub-column ID in bits
    30:31.
    """
    _compile_time_only("umma_mxfp_e2m1_e2m1_ss_1sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects(), cta_group=2)
def umma_mxfp_e2m1_e2m1_ss_2sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a two-CTA MXFP4 block-scaled SS UMMA with E2M1 A/B and E8M0 scale factors.

    A and B come from SMEM; SFA/SFB come from TMEM and scale blocks of 32 K elements. The instruction consumes K=64 and
    writes an ``out_shape=(M, N)`` FP32 accumulator across the CTA pair's TMEM. One elected thread in the pair issues
    it; ``accum=0`` initializes C and nonzero accumulates. Address bits 30:31 select the SF sub-column.
    """
    _compile_time_only("umma_mxfp_e2m1_e2m1_ss_2sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects())
def umma_mxfp_e4m3_e2m1_ss_1sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a one-CTA mixed MXFP8-by-MXFP4 SS UMMA with E8M0 scale factors.

    A uses E4M3, B uses E2M1, and each scale factor covers 32 K elements. The instruction consumes K=32 and writes an
    ``out_shape=(M, N)`` FP32 TMEM tile. ``accum=0`` initializes C; nonzero accumulates. Bits 30:31 of each scale
    address select its scale-factor sub-column.
    """
    _compile_time_only("umma_mxfp_e4m3_e2m1_ss_1sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects(), cta_group=2)
def umma_mxfp_e4m3_e2m1_ss_2sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a two-CTA mixed MXFP8-by-MXFP4 SS UMMA with E8M0 scale factors.

    A uses E4M3, B uses E2M1, and each scale factor covers 32 K elements. The instruction consumes K=32 and writes an
    ``out_shape=(M, N)`` FP32 accumulator across the CTA pair's TMEM. One elected thread issues it; ``accum=0``
    initializes C and nonzero accumulates. Bits 30:31 of each scale address select its scale-factor sub-column.
    """
    _compile_time_only("umma_mxfp_e4m3_e2m1_ss_2sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects())
def umma_mxfp_e4m3_e4m3_ss_1sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a one-CTA MXFP8 block-scaled SS UMMA with E4M3 A/B and E8M0 scale factors.

    A and B come from SMEM; SFA/SFB come from TMEM and scale blocks of 32 K elements. The instruction consumes K=32 and
    writes an ``out_shape=(M, N)`` FP32 TMEM tile. ``accum=0`` initializes C; nonzero accumulates. Address bits 30:31 of
    ``tsfa`` and ``tsfb`` select the scale-factor sub-column.
    """
    _compile_time_only("umma_mxfp_e4m3_e4m3_ss_1sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects(), cta_group=2)
def umma_mxfp_e4m3_e4m3_ss_2sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a two-CTA MXFP8 block-scaled SS UMMA with E4M3 A/B and E8M0 scale factors.

    A and B come from SMEM; SFA/SFB come from TMEM and scale blocks of 32 K elements. The instruction consumes K=32 and
    writes an ``out_shape=(M, N)`` FP32 accumulator across the CTA pair's TMEM. One elected thread issues the operation;
    ``accum=0`` initializes C and nonzero accumulates. Bits 30:31 of each scale address select its SF sub-column.
    """
    _compile_time_only("umma_mxfp_e4m3_e4m3_ss_2sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects())
def umma_nvfp4_ss_1sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a one-CTA NVFP4 block-scaled SS UMMA with E2M1 A/B and E4M3 scale factors.

    A and B come from SMEM; SFA/SFB come from TMEM and scale blocks of 16 K elements. The instruction consumes K=64 and
    writes an ``out_shape=(M, N)`` FP32 TMEM tile. ``accum=0`` initializes C; nonzero accumulates. ``tsfa``/``tsfb``
    bits 30:31 select the scale-factor sub-column.
    """
    _compile_time_only("umma_nvfp4_ss_1sm")


@builtin(eval_return_type=void, effects=_blockscaled_effects(), cta_group=2)
def umma_nvfp4_ss_2sm(
    c_mem,
    tmem_c,
    desc_a,
    desc_b,
    accum,
    sfa_mem,
    tsfa,
    sfb_mem,
    tsfb,
    *,
    out_shape,
):
    """Issue a two-CTA NVFP4 block-scaled SS UMMA with E2M1 A/B and E4M3 scale factors.

    A and B come from SMEM; SFA/SFB come from TMEM and scale blocks of 16 K elements. The instruction consumes K=64 and
    writes an ``out_shape=(M, N)`` FP32 accumulator across the CTA pair's TMEM. One elected thread issues the operation;
    ``accum=0`` initializes C and nonzero accumulates. Bits 30:31 of each scale address select its SF sub-column.
    """
    _compile_time_only("umma_nvfp4_ss_2sm")


__all__ = [
    "MEMORY_EFFECTS_ATTR",
    "ORIGINAL_PY_FUNC_ATTR",
    "Pipeline",
    "cluster_coord",
    "copy_bf16x16_linear_to_kmajor",
    "copy_bf16x16_linear_to_kmajor_scaled",
    "group_coord",
    "rank_in_group",
    "make_umma_smem_desc",
    "smem_ld_128",
    "ldmatrix_x1",
    "ldmatrix_x1_trans",
    "ldmatrix_x4",
    "ldmatrix_x4_trans",
    "smem_ld_bf16",
    "smem_ld_bf16_k_major",
    "smem_ld_f32",
    "smem_ld_f32x2",
    "smem_st_bf16",
    "smem_st_bf16_k_major",
    "bf16_from_bits",
    "smem_st_f32",
    "smem_ld_u8",
    "smem_st_128",
    "stmatrix_x4_trans",
    "stmatrix_x1_reg",
    "stmatrix_x4_regs",
    "tcgen05_cp_128x256b_1sm",
    "tcgen05_cp_128x256b_2sm",
    "tma_ld_1d",
    "tma_ld_2d",
    "tma_ld_3d",
    "tma_st_1d",
    "tma_st_2d",
    "tma_st_3d",
    "tmem_ld_16dp_f32_to_smem_bf16_x16",
    "tmem_ld_16dp_256bx2",
    "tmem_ld_16dp_256bx4",
    "tmem_ld_32dp_f32_to_smem_bf16_x8",
    "tmem_ld_32dp_f32_to_smem_bf16_x32",
    "tmem_ld_16dp_32bxn",
    "tmem_ld_32dp_32bxn",
    "tmem_ld_fence",
    "tmem_st_32dp_32bxn",
    "tmem_st_16dp_128bx2",
    "tmem_st_16dp_32bxn",
    "tmem_st_fence",
    "umma_f16_ss_2sm",
    "umma_f16_ss_2sm_accum",
    "umma_f16_ss_2sm_init",
    "umma_f16_ss_1sm_accum",
    "umma_f16_ss_1sm_init",
    "umma_f16_ss_1sm_kblock",
    "umma_f16_ss_2sm_kblock",
    "umma_f16_ts_1sm_accum",
    "umma_f16_ts_1sm_init",
    "umma_f16_ts_1sm_kblock",
    "umma_f16_ts_2sm_kblock",
    "tcgen05_cp_32x128b_warpx4_1sm",
    "tcgen05_cp_32x128b_warpx4_2sm",
    "umma_mxfp_e2m1_e2m1_ss_1sm",
    "umma_mxfp_e2m1_e2m1_ss_2sm",
    "umma_mxfp_e4m3_e2m1_ss_1sm",
    "umma_mxfp_e4m3_e2m1_ss_2sm",
    "umma_mxfp_e4m3_e4m3_ss_1sm",
    "umma_mxfp_e4m3_e4m3_ss_2sm",
    "umma_nvfp4_ss_1sm",
    "umma_nvfp4_ss_2sm",
]
