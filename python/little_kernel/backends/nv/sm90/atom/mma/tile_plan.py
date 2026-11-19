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
"""Queryable multi-atom MMA tile plans (coverage.mma_pipeline, accumulator axis).

``atom.mma.tensor_core`` describes the *shape of one* tensor-core
instruction: M/N/K dims, operand dtypes, per-thread accumulator register
count, SMEM bytes per K-step. ``atom.pipeline.smem_plan`` describes where the
staged shared-memory buffers a K-loop fills live. Neither answers the
*accumulator-tiling* questions a pipelined CTA kernel has to ask BEFORE any
PTX is emitted, because a real kernel never runs a single atom — it tiles a
CTA output ``(cta_m, cta_n)`` across one or more warpgroups along M and into
column blocks along N, then walks a K loop accumulating into a fixed register
fragment per atom. Today every kernel redoes that arithmetic by hand (e.g.
``benchmark/gemm_sm90/gemm_v5.py`` declares ``acc_00/acc_01/acc_10/acc_11``
and unrolls ``for ki in range(0, BK, WGMMA_K)`` with pinned fragment sizes).

This module derives the tile plan with pure, backend-agnostic arithmetic. It
composes no PTX/CUDA strings and rewrites no IR; it is the same kind of audit
surface as ``atom.mma.tensor_core`` and ``atom.pipeline.smem_plan``, one axis
over: the accumulator-tiling axis recorded for the semantic key
``coverage.mma_pipeline`` ("tensor-core tiling, accumulator and shared-memory
pipeline"). The contracts it pins mirror the invariants the wgmma emitter and
real kernels already enforce:

* the CTA output tile is covered exactly: ``cta_m`` must split evenly across
  warpgroups of ``atom.M`` rows each and ``cta_n`` must be a multiple of
  ``atom.N``; partial tiles raise ValueError rather than being clipped;
* every atom in the tile has the same fixed fragment size
  ``atom.acc_regs_per_thread``, so an epilogue can find the register range
  of atom ``(wg_m, n_blk)`` as a contiguous half-open interval inside the
  per-warp accumulator space;
* the K loop is described as ``ceil(k_total / atom.K)`` steps with a
  mandatory remainder check: a K extent that is not a multiple of
  ``atom.K`` is illegal (the tail would need a different instruction shape);
* the accumulator-ordering convention is C (row-major over (wg_m, n_blk)),
  matching the way kernels number their fragment variables
  (``acc_{wg_m}{n_blk}``).

Legality model (same shape as atom.mma.tensor_core): every argument is
validated up front; illegal plans raise ValueError, never a clipped or
"best-effort" plan.
"""

from dataclasses import dataclass

from little_kernel.backends.nv.sm90.atom.mma.tensor_core import MMA


def _check_positive_int(v, name, allow_zero=False):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name}: must be an int (got {v!r})")
    lo = 0 if allow_zero else 1
    if v < lo:
        cmp = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}: must be a {cmp} int (got {v!r})")
    return v


def _check_atom(atom):
    if not isinstance(atom, MMA):
        raise ValueError(f"plan_mma_tile: atom must be an atom.mma.tensor_core.MMA "
                         f"(got {atom!r}); build one via wgmma_m64nNk16_f32 or the "
                         "dataclass directly")
    return atom


@dataclass(frozen=True)
class MmaTilePlan:
    """A validated multi-atom tiling of a CTA output tile.

    Fields:

    * ``atom`` — the single instruction shape every issue uses.
    * ``cta_m``/``cta_n``/``k_total`` — the CTA output tile rows/cols and the
      total K extent the loop covers.
    * ``num_warpgroups_m`` — how many warpgroups split the M dimension
      (each owns ``atom.M`` rows).
    * ``num_n_blocks`` — column blocks one warpgroup owns along N
      (``cta_n // atom.N``).
    * ``num_k_steps`` — K-loop trip count (``k_total // atom.K``).

    Every derived quantity is exact; legality is established in
    ``plan_mma_tile`` before the dataclass is built.
    """

    atom: MMA
    cta_m: int
    cta_n: int
    k_total: int
    num_warpgroups_m: int
    num_n_blocks: int
    num_k_steps: int

    # -- tiling derivations --------------------------------------------------

    @property
    def cta_m_per_warpgroup(self) -> int:
        """Rows of the CTA tile one warpgroup owns (always ``atom.M``)."""
        return self.cta_m // self.num_warpgroups_m

    @property
    def atoms_per_warpgroup(self) -> int:
        """Atoms one warpgroup issues per K-step (``num_n_blocks``).

        A warpgroup owns exactly one M row of atoms (the atom's M already
        spans the full warpgroup row) and walks the N column blocks, so its
        per-K-step issue count is the N block count.
        """
        return self.num_n_blocks

    @property
    def num_atoms_total(self) -> int:
        """Atoms needed to cover the whole CTA tile once (one K-step, all
        warpgroups): ``num_warpgroups_m * num_n_blocks``."""
        return self.num_warpgroups_m * self.num_n_blocks

    @property
    def num_issues_total(self) -> int:
        """Total instruction issues the whole K loop performs:
        ``num_atoms_total * num_k_steps``."""
        return self.num_atoms_total * self.num_k_steps

    # -- accumulator register derivations ------------------------------------

    @property
    def acc_regs_per_atom(self) -> int:
        """Per-thread accumulator registers of one atom.

        Delegates to ``atom.acc_regs_per_thread``; for the PTX f32-accumulator
        WGMMA contract this is ``M * N / 128`` (``N // 2`` for M=64).
        """
        return self.atom.acc_regs_per_thread

    @property
    def acc_regs_per_warpgroup(self) -> int:
        """Per-thread accumulator registers one warpgroup holds across all
        of its N blocks: ``acc_regs_per_atom * num_n_blocks``.

        This is the register-budget question a pipeline audit asks BEFORE
        deciding whether a tile is feasible next to SMEM staging: the
        fragments are live for the whole K loop.
        """
        return self.acc_regs_per_atom * self.num_n_blocks

    @property
    def acc_elems_total(self) -> int:
        """Total accumulator elements across the whole CTA tile:
        ``cta_m * cta_n``. Sanity-checked against the per-atom layout
        (``acc_regs_per_atom * 128 * num_atoms_total``)."""
        per_atom_layout = (
            self.acc_regs_per_atom * 128  # threads per warpgroup
            * self.num_atoms_total)
        direct = self.cta_m * self.cta_n
        if per_atom_layout != direct:
            raise ValueError(f"acc_elems_total: atom layout ({per_atom_layout}) does not "
                             f"cover the CTA tile exactly ({direct}); the plan is not a "
                             "complete tiling")
        return direct

    def atom_index(self, wg_m: int, n_blk: int) -> int:
        """Flat (row-major over (wg_m, n_blk)) index of one atom.

        Atom ``(wg_m, n_blk)`` is ``wg_m * num_n_blocks + n_blk``; this is
        the numbering kernels use for their fragment variables
        (``acc_{wg_m}{n_blk}``) and the order an epilogue walks them.
        """
        _check_positive_int(wg_m, "atom_index(wg_m)", allow_zero=True)
        _check_positive_int(n_blk, "atom_index(n_blk)", allow_zero=True)
        if wg_m >= self.num_warpgroups_m:
            raise ValueError(f"atom_index: wg_m {wg_m} out of range for "
                             f"num_warpgroups_m={self.num_warpgroups_m}")
        if n_blk >= self.num_n_blocks:
            raise ValueError(f"atom_index: n_blk {n_blk} out of range for "
                             f"num_n_blocks={self.num_n_blocks}")
        return wg_m * self.num_n_blocks + n_blk

    def acc_reg_range(self, wg_m: int, n_blk: int) -> tuple:
        """Half-open ``[base, base + acc_regs_per_atom)`` accumulator register
        range atom ``(wg_m, n_blk)`` owns inside its warpgroup's fragment
        space.

        The fragment space is per warpgroup and laid out N-block-major (the
        same order as ``atom_index`` within one warpgroup), so atom
        ``(wg_m, n_blk)`` starts at ``n_blk * acc_regs_per_atom`` regardless
        of ``wg_m``; ``wg_m`` is validated for membership but does not offset
        the per-warpgroup space.
        """
        # Validate indices (raises on out-of-range).
        self.atom_index(wg_m, n_blk)
        base = n_blk * self.acc_regs_per_atom
        return (base, base + self.acc_regs_per_atom)

    def acc_tile_slice(self, wg_m: int, n_blk: int) -> tuple:
        """Half-open ``(row_range, col_range)`` of the CTA output tile that
        atom ``(wg_m, n_blk)`` computes: rows
        ``[wg_m*atom.M, (wg_m+1)*atom.M)`` and cols
        ``[n_blk*atom.N, (n_blk+1)*atom.N)``.

        This is the address an epilogue or a store-to-shared helper needs to
        place the atom's fragment into the CTA-global result tile.
        """
        self.atom_index(wg_m, n_blk)
        row0 = wg_m * self.atom.M
        col0 = n_blk * self.atom.N
        return ((row0, row0 + self.atom.M), (col0, col0 + self.atom.N))

    # -- K-loop derivations ----------------------------------------------------

    def k_step_range(self) -> range:
        """The K-loop iteration domain the tile walks: ``range(0, k_total,
        atom.K)``. Its length is ``num_k_steps``."""
        return range(0, self.k_total, self.atom.K)

    def k_step_index(self, k_offset: int) -> int:
        """K-loop step index of an absolute K offset: ``k_offset // atom.K``.

        ``k_offset`` must sit exactly on an atom boundary (a multiple of
        ``atom.K``) and inside the K extent; offsets into the middle of an
        atom are illegal because an instruction consumes a whole K step.
        """
        _check_positive_int(k_offset, "k_step_index(k_offset)", allow_zero=True)
        if k_offset >= self.k_total:
            raise ValueError(f"k_step_index: k_offset {k_offset} out of range for "
                             f"k_total={self.k_total}")
        if k_offset % self.atom.K != 0:
            raise ValueError(f"k_step_index: k_offset {k_offset} is not on an atom "
                             f"boundary (atom.K={self.atom.K}); an instruction consumes a "
                             "whole K step")
        return k_offset // self.atom.K

    def scale_d_for(self, k_step: int) -> int:
        """Accumulator-input scale (PTX ``scale_d``) for one K step.

        0 for the first K step (the instruction overwrites the accumulator,
        implementing the zero-init without a separate ``wgmma_init_accum``
        pass — the same discipline the emitter's runtime-``scale_d``
        predicate supports) and 1 for every later step (accumulate).
        Validated to be inside the K-loop domain.
        """
        _check_positive_int(k_step, "scale_d_for(k_step)", allow_zero=True)
        if k_step >= self.num_k_steps:
            raise ValueError(f"scale_d_for: k_step {k_step} out of range for "
                             f"num_k_steps={self.num_k_steps}")
        return 0 if k_step == 0 else 1


def plan_mma_tile(atom, cta_m, cta_n, k_total, num_warpgroups_m=1):
    """Plan the multi-atom tiling of a CTA output tile for one MMA atom.

    Args:
        atom: An ``atom.mma.tensor_core.MMA`` (e.g. the result of
            ``wgmma_m64nNk16_f32(N)``): the instruction shape and dtype of
            every issue in the tile.
        cta_m: CTA output tile rows. Must be exactly
            ``atom.M * num_warpgroups_m`` — partial M tiles are illegal.
        cta_n: CTA output tile cols. Must be a multiple of ``atom.N`` —
            partial N tiles are illegal.
        k_total: Total K extent the loop covers. Must be a multiple of
            ``atom.K`` — the K loop cannot tail off mid-instruction.
        num_warpgroups_m: Number of warpgroups that split the M dimension.
            Defaults to 1 (one warpgroup owns the whole CTA tile's rows).

    Returns:
        A validated :class:`MmaTilePlan`.

    Raises:
        ValueError: On illegal arguments or non-exact tiling.
    """
    _check_atom(atom)
    _check_positive_int(cta_m, "plan_mma_tile(cta_m)")
    _check_positive_int(cta_n, "plan_mma_tile(cta_n)")
    _check_positive_int(k_total, "plan_mma_tile(k_total)")
    _check_positive_int(num_warpgroups_m, "plan_mma_tile(num_warpgroups_m)")

    if cta_m % atom.M != 0:
        raise ValueError(f"plan_mma_tile: cta_m {cta_m} is not a multiple of atom.M "
                         f"{atom.M}; partial M tiles are illegal (the atom must cover "
                         "every output row)")
    if cta_m // atom.M != num_warpgroups_m:
        raise ValueError(f"plan_mma_tile: cta_m {cta_m} needs {cta_m // atom.M} atoms "
                         f"along M but num_warpgroups_m={num_warpgroups_m}; the M tile "
                         "must split exactly across warpgroups")
    if cta_n % atom.N != 0:
        raise ValueError(f"plan_mma_tile: cta_n {cta_n} is not a multiple of atom.N "
                         f"{atom.N}; partial N tiles are illegal (choose atom.N dividing "
                         "the column tile)")
    if k_total % atom.K != 0:
        raise ValueError(f"plan_mma_tile: k_total {k_total} is not a multiple of atom.K "
                         f"{atom.K}; the K loop cannot tail off mid-instruction")

    return MmaTilePlan(
        atom=atom,
        cta_m=cta_m,
        cta_n=cta_n,
        k_total=k_total,
        num_warpgroups_m=num_warpgroups_m,
        num_n_blocks=cta_n // atom.N,
        num_k_steps=k_total // atom.K,
    )
