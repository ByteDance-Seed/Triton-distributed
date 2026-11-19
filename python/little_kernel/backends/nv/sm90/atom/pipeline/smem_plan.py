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
"""Queryable shared-memory pipeline stage plans (coverage.mma_pipeline).

The emitter-side intrinsics (cp_async / TMA / WGMMA / UMMA) know how to
*emit* instruction text for one pipeline primitive; atom.mma.tensor_core
describes the *shape of* one tensor-core instruction. Neither answers the
planning questions a software-pipelined kernel has to ask BEFORE anything is
emitted:

* How many stages does a set of staged shared-memory buffers allow under a
  byte budget?
* Where does each stage of each version-multiplied buffer land in a carved
  shared-memory region (with alignment)?
* Which ring-buffer slot does loop iteration ``i`` use, and which barrier
  slot / phase parity guards it?
* How does the iteration domain split into prologue / steady / epilogue with
  no duplicated or dropped iteration?

This module answers those questions with pure, backend-agnostic
arithmetic. It composes no PTX/CUDA strings and rewrites no IR; it is the
same kind of audit surface as atom.mma.tensor_core, one layer up the
pipeline axis. The contracts it pins mirror the invariants recorded for
upstream pipeline passes (cw.passes.tilelang-software-pipeline):

* buffers that live across stages are multi-versioned, so SMEM footprint
  scales linearly with stage depth;
* versions are chosen by a floormod ring over the loop position;
* barrier slots expand from ``barrier_slots_per_stage`` to
  ``num_stages * barrier_slots_per_stage``, addressed as
  ``stage * barrier_slots_per_stage + slot`` with parity
  ``floormod(floordiv(i, num_stages), 2)``;
* num_stages == 1 takes the as-is path (codegen chooses the synchronous
  route with no barrier expansion), and prologue/epilogue intervals
  collapse; the ring arithmetic stays well-defined so audits can still
  reason about the degenerate plan uniformly;
* prologue/steady/epilogue intervals partition the iteration domain without
  overlap or omission (empty intervals are reported as empty, not clamped
  into each other).

Legality model (same shape as atom.mma.tensor_core): every argument is
validated up front; illegal plans raise ValueError, never a clipped or
"best-effort" plan.
"""

from dataclasses import dataclass

# Alignment applied to every staged buffer inside a carved SMEM region.
# 1024 matches the default dynamic-shared alignment the shared allocator
# accepts (tests/nv/common/unit/test_alloc_shared_memory_dtype_attr.py) and makes the
# byte-derivation examples exact powers-of-two stories.
PIPELINE_ABI_ALIGNMENT_BYTES = 1024


def _check_positive_int(v, name, allow_zero=False):
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"{name}: must be an int (got {v!r})")
    lo = 0 if allow_zero else 1
    if v < lo:
        cmp = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name}: must be a {cmp} int (got {v!r})")
    return v


def _align_up(n, align):
    return (n + align - 1) // align * align


@dataclass(frozen=True)
class SmemBufferSpec:
    """One shared-memory buffer that a pipelined loop stage produces.

    ``bytes_per_step`` is the number of bytes one iteration (one pipeline
    fill step) writes into this buffer. If the buffer's lifetime spans
    multiple stages (the usual producer case) it is multi-versioned, so its
    total footprint is ``bytes_per_step * num_stages``; callers account that
    through PipelinePlan.total_smem_bytes rather than hardcoding the factor.
    """

    name: str
    bytes_per_step: int

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"SmemBufferSpec: name must be a non-empty str "
                             f"(got {self.name!r})")
        _check_positive_int(self.bytes_per_step, f"SmemBufferSpec({self.name!r}).bytes_per_step")

    @property
    def aligned_bytes_per_step(self) -> int:
        """Per-step bytes padded to PIPELINE_ABI_ALIGNMENT_BYTES."""
        return _align_up(self.bytes_per_step, PIPELINE_ABI_ALIGNMENT_BYTES)


@dataclass(frozen=True)
class StagePlan:
    """Where one ring-buffer slot of one version-multiplied buffer lands.

    ``version`` is the floormod ring slot (0..num_stages-1) and ``offset``
    the byte offset of that slot inside the carved SMEM region. A
    ``StagePlan`` for the same buffer at successive versions is exactly
    ``aligned_bytes_per_step`` apart, which is the invariant codegen can
    rely on when indexing ``buffer_ptr + version * stride``.
    """

    buffer_name: str
    version: int
    offset: int
    bytes: int

    @property
    def end_offset(self) -> int:
        return self.offset + self.bytes


@dataclass(frozen=True)
class PipelinePlan:
    """A validated multi-stage pipeline layout over staged SMEM buffers.

    Fields:

    * ``num_stages`` — ring depth. 1 means the as-is, non-pipelined path:
      no version multiplication, no barrier expansion, and the iteration
      domain is entirely "steady".
    * ``requested_num_stages`` — the depth the caller asked for; a budget
      clamp is observable (``num_stages < requested_num_stages``) rather
      than silent.
    * ``buffers`` — the staged buffers, in caller order.
    * ``stage_bytes`` — total bytes ONE pipeline step writes across all
      buffers (unaligned sum; the per-buffer aligned carve-out lives in
      ``total_smem_bytes``).
    * ``total_smem_bytes`` — the full carved region
      (sum of aligned per-buffer bytes multiplied by the version count).
    * ``barrier_slots_per_stage`` — logical barrier slots one stage needs
      (e.g. one full/empty pair per producer group); expanded total is
      ``num_stages * barrier_slots_per_stage``.
    """

    num_stages: int
    requested_num_stages: int
    buffers: tuple
    stage_bytes: int
    total_smem_bytes: int
    barrier_slots_per_stage: int

    # -- ring / barrier indexing -------------------------------------------

    def version_of(self, iteration: int) -> int:
        """Ring slot a loop position lands in: floormod(i, num_stages)."""
        _check_positive_int(iteration, "version_of(iteration)", allow_zero=True)
        return iteration % self.num_stages

    def barrier_slot(self, version: int, slot: int) -> int:
        """Expanded barrier index: version * slots_per_stage + slot."""
        _check_positive_int(version, "barrier_slot(version)", allow_zero=True)
        _check_positive_int(slot, "barrier_slot(slot)", allow_zero=True)
        if version >= self.num_stages:
            raise ValueError(f"barrier_slot: version {version} out of range for "
                             f"num_stages={self.num_stages}")
        if slot >= self.barrier_slots_per_stage:
            raise ValueError(f"barrier_slot: slot {slot} out of range for "
                             f"barrier_slots_per_stage={self.barrier_slots_per_stage}")
        return version * self.barrier_slots_per_stage + slot

    def barrier_parity(self, iteration: int) -> int:
        """Phase parity at a loop position.

        Same period as the upstream contract: the parity flips once per
        full wrap of the ring, ``floormod(floordiv(i, num_stages), 2)``,
        so consecutive reuses of one slot see alternating phases.
        """
        _check_positive_int(iteration, "barrier_parity(iteration)", allow_zero=True)
        return (iteration // self.num_stages) % 2

    def iter_version_parity(self, iteration: int) -> tuple:
        """(version, parity) pair for one iteration, the guard triple minus
        the slot id a consumer picks from its own group."""
        return (self.version_of(iteration), self.barrier_parity(iteration))

    # -- layout -------------------------------------------------------------

    def stage_plans_for(self, buffer_name: str) -> tuple:
        """All ring slots of one buffer, in ascending version order."""
        for i, buf in enumerate(self.buffers):
            if buf.name == buffer_name:
                break
        else:
            raise ValueError(f"stage_plans_for: unknown buffer {buffer_name!r}; planned: "
                             f"{[b.name for b in self.buffers]}")
        base = self._buffer_base_offset(i)
        stride = buf.aligned_bytes_per_step
        return tuple(
            StagePlan(buffer_name=buf.name, version=v, offset=base + v * stride, bytes=buf.bytes_per_step)
            for v in range(self.num_stages))

    def slot_of(self, buffer_name: str, iteration: int) -> StagePlan:
        """The StagePlan a given loop iteration actually uses for a buffer."""
        v = self.version_of(iteration)
        return self.stage_plans_for(buffer_name)[v]

    def _buffer_base_offset(self, index: int) -> int:
        base = 0
        for buf in self.buffers[:index]:
            base += buf.aligned_bytes_per_step * self.num_stages
        return base

    # -- iteration-domain partition -----------------------------------------

    def prologue_iters(self, trip_count: int, fill_stages: int = None):
        """[0, min(n_prologue, trip_count)) — iterations that only fill.

        ``fill_stages`` defaults to ``num_stages - 1`` (the standard
        steady-state depth: every iteration overlaps one consumer with
        num_stages-1 fills). fill_stages == 0 means no prologue at all.
        """
        _check_positive_int(trip_count, "prologue_iters(trip_count)", allow_zero=True)
        n_prologue = self._resolve_fill_stages(fill_stages)
        end = min(n_prologue, trip_count)
        return (0, end)

    def steady_iters(self, trip_count: int, fill_stages: int = None):
        """[n_prologue, trip_count) — iterations with a full overlap."""
        _check_positive_int(trip_count, "steady_iters(trip_count)", allow_zero=True)
        n_prologue = self._resolve_fill_stages(fill_stages)
        return (min(n_prologue, trip_count), trip_count)

    def epilogue_iters(self, trip_count: int, fill_stages: int = None):
        """[trip_count, trip_count + n_prologue) — drain iterations.

        The three intervals [0,e_p), [e_p,trip_count), [trip_count,
        trip_count+n_prologue) partition [0, trip_count+n_prologue) with no
        overlap and no gap, for any trip_count (including 0 and values
        smaller than the prologue depth).
        """
        _check_positive_int(trip_count, "epilogue_iters(trip_count)", allow_zero=True)
        n_prologue = self._resolve_fill_stages(fill_stages)
        return (trip_count, trip_count + n_prologue)

    def _resolve_fill_stages(self, fill_stages):
        if fill_stages is None:
            return self.num_stages - 1
        _check_positive_int(fill_stages, "fill_stages", allow_zero=True)
        if fill_stages > self.num_stages:
            raise ValueError(f"fill_stages ({fill_stages}) must not exceed num_stages "
                             f"({self.num_stages}): a fill can never wait on a slot the "
                             "ring does not have")
        return fill_stages


def plan_k_pipeline(buffers, smem_budget_bytes, max_stages, requested_stages=None, barrier_slots_per_stage=1):
    """Plan the deepest legal K-loop SMEM pipeline under a byte budget.

    Args:
        buffers: Iterable of SmemBufferSpec (or (name, bytes_per_step)
            pairs) staged per K-step. Order is preserved in the carved
            layout.
        smem_budget_bytes: Total shared-memory bytes the pipeline may
            occupy (dynamic SMEM carve-out minus whatever non-staged
            buffers reserve). Must be a non-negative int.
        max_stages: Hard upper bound on ring depth (target/copy-queue
            limit the caller enforces, e.g. async-group depth). Must be >= 1.
        requested_stages: Optional desired depth; the plan is clamped to
            ``min(requested, budget_fit, max_stages)`` and the clamp is
            observable via plan.requested_num_stages. Defaults to
            ``max_stages``.
        barrier_slots_per_stage: Logical barrier slots each stage owns.

    Returns:
        A validated PipelinePlan.

    Raises:
        ValueError: On illegal arguments or when even num_stages == 1 does
            not fit the budget (the plan must be legal, never clipped into
            a region smaller than one stage).
    """
    specs = tuple(b if isinstance(b, SmemBufferSpec) else SmemBufferSpec(name=b[0], bytes_per_step=b[1])
                  for b in buffers)
    if not specs:
        raise ValueError("plan_k_pipeline: at least one staged buffer is required")
    names = [b.name for b in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"plan_k_pipeline: duplicate buffer names {names!r}")
    _check_positive_int(smem_budget_bytes, "plan_k_pipeline(smem_budget_bytes)", allow_zero=True)
    _check_positive_int(max_stages, "plan_k_pipeline(max_stages)")
    _check_positive_int(barrier_slots_per_stage, "plan_k_pipeline(barrier_slots_per_stage)")
    if requested_stages is None:
        requested_stages = max_stages
    _check_positive_int(requested_stages, "plan_k_pipeline(requested_stages)")

    per_stage_bytes = sum(b.aligned_bytes_per_step for b in specs)
    if per_stage_bytes > smem_budget_bytes:
        raise ValueError(f"plan_k_pipeline: one stage needs {per_stage_bytes} aligned "
                         f"bytes but the budget is {smem_budget_bytes}; even num_stages "
                         "= 1 cannot fit, so no legal pipeline plan exists (reduce the "
                         "staged tile or raise the carve-out)")

    budget_fit = smem_budget_bytes // per_stage_bytes
    num_stages = min(requested_stages, budget_fit, max_stages)
    if num_stages < 1:  # defensive; per_stage check above already excludes
        raise ValueError("plan_k_pipeline: no legal stage count")

    return PipelinePlan(num_stages=num_stages, requested_num_stages=requested_stages, buffers=specs,
                        stage_bytes=sum(b.bytes_per_step for b in specs), total_smem_bytes=per_stage_bytes * num_stages,
                        barrier_slots_per_stage=barrier_slots_per_stage)
