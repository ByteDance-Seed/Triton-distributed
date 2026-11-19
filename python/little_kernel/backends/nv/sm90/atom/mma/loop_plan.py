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
"""Jointly-validated pipelined-MMA K-loop plans (coverage.mma_pipeline,
the composition axis over tile_plan x smem_plan).

The three single-axis audit surfaces each answer one question:

* ``atom.mma.tensor_core`` — the *shape of one* tensor-core instruction
  (per-instruction operand bytes, accumulator registers per thread);
* ``atom.mma.tile_plan`` — the *accumulator-tiling* of a CTA output tile
  (atoms per warpgroup, per-thread fragment register ranges, the K-loop
  iteration domain at ``atom.K`` granularity);
* ``atom.pipeline.smem_plan`` — the *shared-memory pipeline* ring
  (stage depth under a byte budget, version slots, barrier parity).

A real pipelined CTA kernel never runs one instruction and never stages one
instruction's operands: its K loop walks in steps of ``BK`` (a whole CTA-tile
slice per step) and accumulates into a per-warpgroup fragment. The quantity
that couples the axes — and that a caller currently derives by hand — is the
**per-K-loop-step** staged-operand footprint of the *whole CTA tile*,
``cta_m * BK * opelem`` for A and ``cta_n * BK * opelem`` for B, NOT the
per-instruction ``atom.a_elems_total``/``atom.b_elems_total`` (which are at
``atom.K`` granularity and a single ``atom.M``/``atom.N`` tile). Using the
instruction-level numbers for a K-loop-step pipeline silently undercounts the
ring footprint (an 8x / 4x undercount for a 128x128x64 bf16 tile with a
m64n128k16 atom), over-reporting the stage depth the budget allows. See
``scratch/repro_mma_pipeline_gap.py``.

This module composes the axes so the coupling is derived and validated in one
place. It inherits the legality model of its inputs: every argument is
validated up front and illegal configurations raise ValueError, never a
clipped plan. It composes no PTX/CUDA strings and rewrites no IR — it is pure
backend-agnostic arithmetic, the same audit discipline as the three surfaces
it joins, recording the composition axis for the semantic key
``coverage.mma_pipeline``.

Diamond vs. plain operand staging: a warpgroup's A operand is shared across
all of its N blocks and its B operand across the M axis, so one CTA K-loop
step stages each CTA-tile operand slab exactly once. The A slab is
``cta_m * BK`` elements and the B slab ``cta_n * BK`` elements regardless of
how many warpgroups consume them; the warpgroup count affects the
accumulator-register question (axis 1), not the staged byte count.
"""

from dataclasses import dataclass

from little_kernel.backends.nv.sm90.atom.mma.tensor_core import MMA
from little_kernel.backends.nv.sm90.atom.mma.tile_plan import MmaTilePlan, plan_mma_tile
from little_kernel.backends.nv.sm90.atom.pipeline.smem_plan import PipelinePlan, SmemBufferSpec, plan_k_pipeline

# Register file granularity used for the accumulator-budget query. PTX
# accumulator registers are 32-bit; the SM90/SM100 per-thread register
# ceiling a kernel plans against is 255 (arch limit), but callers supply their
# own budget since epilogue/addressing registers share the file.
REGISTER_BITS = 32


def _check_positive_int(v, name, allow_zero=False):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name}: must be an int (got {v!r})")
    lo = 0 if allow_zero else 1
    if v < lo:
        cmp = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}: must be a {cmp} int (got {v!r})")
    return v


def _dtype_bytes(dtype, name):
    bits = getattr(dtype, "bits", None)
    if not isinstance(bits, int) or isinstance(bits, bool) or bits <= 0 or \
            bits % 8 != 0:
        raise ValueError(f"{name}: dtype must expose a positive multiple-of-8 ``bits`` "
                         f"(got {dtype!r}); byte-sized operand element widths are "
                         "required to derive the staged SMEM footprint")
    return bits // 8


def a_slab_bytes(atom, cta_m, loop_k):
    """Staged bytes of the A slab one CTA K-loop step writes.

    ``cta_m * loop_k * A_dtype_bytes``: the whole CTA tile's A slice at the
    loop's ``BK`` granularity — NOT the per-instruction
    ``atom.a_elems_total`` (which is one ``atom.M``-row x ``atom.K`` tile).
    """
    return cta_m * loop_k * _dtype_bytes(atom.A_dtype, "a_slab_bytes(A_dtype)")


def b_slab_bytes(atom, cta_n, loop_k):
    """Staged bytes of the B slab one CTA K-loop step writes.

    ``cta_n * loop_k * B_dtype_bytes``: the whole CTA tile's B slice at the
    loop's ``BK`` granularity — NOT the per-instruction
    ``atom.b_elems_total`` (which is one ``atom.K`` x ``atom.N`` tile).
    """
    return cta_n * loop_k * _dtype_bytes(atom.B_dtype, "b_slab_bytes(B_dtype)")


@dataclass(frozen=True)
class PipelinedMmaPlan:
    """A jointly-validated pipelined-MMA K-loop configuration.

    Fields:

    * ``tile`` — the :class:`MmaTilePlan` (accumulator/tiling axis); its
      ``k_total`` is the whole reduction extent at ``atom.K`` granularity.
    * ``loop_k`` — the K-loop step extent ``BK`` (the pipeline's staging
      granularity). Must be a positive multiple of ``atom.K`` and divide
      ``tile.k_total``.
    * ``pipeline`` — the :class:`PipelinePlan` (SMEM ring axis) over the two
      staged operand slabs ``("A", a_slab_bytes)`` / ``("B", b_slab_bytes)``.

    Derived quantities re-express the coupling so a caller never divides or
    re-derives them:

    * ``num_k_loop_steps`` — K-loop trip count at ``loop_k`` granularity
      (``tile.k_total // loop_k``), the domain the prologue/steady/epilogue
      partition and the ring versions index.
    * every accumulator / fragment question delegates to ``tile`` and every
      ring / barrier / partition question to ``pipeline``, with the loop-step
      trip count as the shared coordinate.
    """

    tile: MmaTilePlan
    loop_k: int
    pipeline: PipelinePlan

    # -- loop-domain coupling -------------------------------------------------

    @property
    def num_k_loop_steps(self) -> int:
        """K-loop trip count: the whole reduction extent in ``loop_k`` steps.

        This is the iteration domain the pipeline ring versions
        (``pipeline.version_of(i)``) and the prologue/steady/epilogue
        partition index — the shared coordinate between the two axes.
        """
        return self.tile.k_total // self.loop_k

    @property
    def loop_steps_per_stage_period(self) -> int:
        """Loop steps per full wrap of the ring (== ``pipeline.num_stages``).

        Exposed so an audit names the ring period in loop-step coordinates
        rather than re-reading the nested pipeline field.
        """
        return self.pipeline.num_stages

    # -- operand staging coupling --------------------------------------------

    @property
    def a_stage_bytes(self) -> int:
        """Staged bytes of the A slab in one ring slot (per K-loop step)."""
        return a_slab_bytes(self.tile.atom, self.tile.cta_m, self.loop_k)

    @property
    def b_stage_bytes(self) -> int:
        """Staged bytes of the B slab in one ring slot (per K-loop step)."""
        return b_slab_bytes(self.tile.atom, self.tile.cta_n, self.loop_k)

    def staged_total_bytes(self) -> int:
        """Aligned SMEM bytes the operand ring occupies across all stages.

        Delegates to ``pipeline.total_smem_bytes`` so the caller sees the
        same number codegen must reserve; equals
        ``(align(a_stage_bytes) + align(b_stage_bytes)) * num_stages``.
        """
        return self.pipeline.total_smem_bytes

    # -- accumulator / register coupling --------------------------------------

    @property
    def acc_regs_per_thread_per_warpgroup(self) -> int:
        """Per-thread accumulator registers one warpgroup holds.

        Delegates to ``tile.acc_regs_per_warpgroup``; this is the register
        that stays live for the whole K loop and so is the budget a pipeline
        feasibility query must check alongside SMEM.
        """
        return self.tile.acc_regs_per_warpgroup

    @property
    def acc_bits_per_thread(self) -> int:
        """Per-thread accumulator register bits (``acc_regs * 32``).

        Accumulator registers are 32-bit per the PTX f32-accumulator WGMMA
        contract the atom audit pins, so the register-file footprint is
        ``acc_regs_per_thread_per_warpgroup * 32``.
        """
        return self.acc_regs_per_thread_per_warpgroup * REGISTER_BITS

    def acc_regs_within(self, reg_budget_per_thread: int) -> bool:
        """Whether the per-thread accumulator fits a register budget.

        ``reg_budget_per_thread`` is the number of 32-bit registers the
        kernel reserves for the accumulator fragment; the query is a pure
        comparison so a caller can budget epilogue/addressing registers
        separately. Illegal (non-positive / non-int) budgets raise ValueError;
        a budget that is legal but too small returns False (it is a
        feasibility query, not a legality violation of the plan itself).
        """
        _check_positive_int(reg_budget_per_thread, "acc_regs_within(reg_budget_per_thread)")
        return self.acc_regs_per_thread_per_warpgroup <= reg_budget_per_thread

    # -- ring / partition delegation -------------------------------------------

    def slot_of_a(self, k_loop_step: int):
        """The ring slot (StagePlan) a K-loop step uses for the A slab."""
        _check_positive_int(k_loop_step, "slot_of_a(k_loop_step)", allow_zero=True)
        return self.pipeline.slot_of("A", k_loop_step)

    def slot_of_b(self, k_loop_step: int):
        """The ring slot (StagePlan) a K-loop step uses for the B slab."""
        _check_positive_int(k_loop_step, "slot_of_b(k_loop_step)", allow_zero=True)
        return self.pipeline.slot_of("B", k_loop_step)

    def iter_version_parity(self, k_loop_step: int) -> tuple:
        """(version, parity) guard pair for one K-loop step."""
        _check_positive_int(k_loop_step, "iter_version_parity(k_loop_step)", allow_zero=True)
        return self.pipeline.iter_version_parity(k_loop_step)


def plan_pipelined_mma(atom, cta_m, cta_n, k_total, loop_k, smem_budget_bytes, num_warpgroups_m=1, max_stages=64,
                       requested_stages=None, barrier_slots_per_stage=1):
    """Plan a jointly-legal pipelined-MMA K loop over a CTA output tile.

    Args:
        atom: An ``atom.mma.tensor_core.MMA`` — the single instruction shape
            every issue uses (e.g. ``wgmma_m64nNk16_f32(N)``).
        cta_m / cta_n: CTA output tile rows/cols (see ``plan_mma_tile`` for
            the exact-coverage legality).
        k_total: Whole reduction extent at ``atom.K`` granularity; must be a
            multiple of ``atom.K``.
        loop_k: The K-loop staging step ``BK``. Must be a positive multiple
            of ``atom.K`` (a step issues a whole number of instructions) and
            must divide ``k_total`` (the loop cannot tail off mid-step).
        smem_budget_bytes: Shared-memory byte budget for the operand ring
            (see ``plan_k_pipeline``).
        num_warpgroups_m: Warpgroups splitting the M axis (see
            ``plan_mma_tile``).
        max_stages / requested_stages / barrier_slots_per_stage: see
            ``plan_k_pipeline``.

    Returns:
        A validated :class:`PipelinedMmaPlan` whose ``tile`` and ``pipeline``
        sub-plans are themselves the single-axis validated plans.

    Raises:
        ValueError: On illegal arguments, non-exact tiling, an illegal
            ``loop_k``, or when the operand ring cannot fit even one stage in
            the budget (the composition inherits each sub-plan's legality —
            it never produces a clipped or best-effort configuration).
    """
    if not isinstance(atom, MMA):
        raise ValueError(f"plan_pipelined_mma: atom must be an "
                         f"atom.mma.tensor_core.MMA (got {atom!r}); build one via "
                         "wgmma_m64nNk16_f32 or the dataclass directly")

    tile = plan_mma_tile(atom, cta_m=cta_m, cta_n=cta_n, k_total=k_total, num_warpgroups_m=num_warpgroups_m)

    _check_positive_int(loop_k, "plan_pipelined_mma(loop_k)")
    if loop_k % atom.K != 0:
        raise ValueError(f"plan_pipelined_mma: loop_k {loop_k} is not a multiple of "
                         f"atom.K {atom.K}; a K-loop step must issue a whole number of "
                         "instructions (raise BK or pick an atom with a smaller K)")
    if k_total % loop_k != 0:
        raise ValueError(f"plan_pipelined_mma: k_total {k_total} is not a multiple of "
                         f"loop_k {loop_k}; the K loop cannot tail off mid-step (the "
                         "operand ring stages whole BK slices)")

    # Derive the CTA-tile operand slabs at loop granularity. This is the
    # cross-axis quantity a caller otherwise hand-derives (and can silently
    # undercount by using the per-instruction atom numbers instead).
    buffers = [
        SmemBufferSpec("A", a_slab_bytes(atom, cta_m, loop_k)),
        SmemBufferSpec("B", b_slab_bytes(atom, cta_n, loop_k)),
    ]
    pipeline = plan_k_pipeline(buffers, smem_budget_bytes=smem_budget_bytes, max_stages=max_stages,
                               requested_stages=requested_stages, barrier_slots_per_stage=barrier_slots_per_stage)

    return PipelinedMmaPlan(tile=tile, loop_k=loop_k, pipeline=pipeline)
