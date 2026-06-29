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
"""CUTLASS-style software pipeline abstractions for TMA.

Two ``@_aggregate`` types:

  - :class:`PipelineState` -- ring-buffer ``index`` / mbarrier ``phase`` pair
    with an ``advance()`` method.  Stores ``num_stages`` as a constexpr so the
    caller never needs to pass it explicitly.
  - :class:`TmaPipeline` -- owns shared-memory barriers and data buffers;
    exposes ``producer_*`` / ``consumer_*`` methods parameterised by a
    ``PipelineState``.  Stores ``num_stages`` as a constexpr.

Phase tracking
--------------
The pipeline uses mbarrier *parity*-based synchronisation.  Each mbarrier has
an internal phase counter that toggles (0 -> 1 -> 0 ...) every time all
expected arrivals complete.  ``producer_acquire`` waits on ``phase ^ 1``
because it needs the *previous* round's consumer release to have completed
before reusing the buffer.
"""
import triton
import triton.language as tl
from triton.language import core as tlc

from triton_dist.jit import jit

from triton_dist.language.extra.cuda.tma_language import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    mbarrier_arrive,
    mbarrier_expect_tx,
    mbarrier_init,
    mbarrier_invalidate,
    mbarrier_wait_parity,
    tma_load_2d,
    tma_store_2d,
)

from triton_dist.language.smem_ops import (
    SharedMemoryDesc,
    smem_get_ptr,
    smem_index,
)

# ===================================================================
# PipelineState  (@_aggregate)
# ===================================================================


@tl.core._aggregate
class PipelineState:
    """Ring-buffer state: ``index`` into pipeline stages, mbarrier ``phase``,
    and compile-time ``num_stages``.

    ``num_stages`` is a ``tl.constexpr`` field -- different stage counts produce
    distinct compiled specialisations, which is the intended behaviour.

    Usage::

        p = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
        for tile in range(...):
            pipeline.producer_acquire(p)
            ...
            p = p.advance()
    """
    index: tl.tensor
    phase: tl.tensor
    num_stages: tl.constexpr

    def __init__(self, index, phase, num_stages):
        self.index = index
        self.phase = phase
        self.num_stages = num_stages

    @triton.jit
    def advance(self):
        """Advance to the next stage; flip phase parity on wrap-around."""
        new_index = (self.index + 1) % self.num_stages
        self.phase = tl.where(new_index == 0, self.phase ^ 1, self.phase)
        self.index = new_index
        return self


# ===================================================================
# TmaPipeline  (@_aggregate)
# ===================================================================


@tl.core._aggregate
class TmaPipeline:
    """CUTLASS ``PipelineTmaAsync``-style object owning barriers and buffers.

    Construct after ``allocate_smem`` for ready/empty mbarriers and the data
    ring buffer.  Call ``init_barriers`` (thread 0) + ``__syncthreads`` before
    entering the producer/consumer loops.

    ``num_stages`` is stored as a ``tl.constexpr`` field so methods like
    ``init_barriers`` / ``invalidate_barriers`` can iterate stages without
    the caller passing it explicitly.

    Thread-gating contract:

      * ``producer_acquire`` / ``consumer_wait`` -- called by **all** threads
        in the warp (``mbarrier_wait_parity`` is per-thread).
      * ``producer_tma_load`` / ``consumer_tma_store`` / ``consumer_release``
        -- must be gated to a **single** elected thread (``elect_one_sync``).
    """
    ready_bars: SharedMemoryDesc
    empty_bars: SharedMemoryDesc
    bufs: SharedMemoryDesc
    num_stages: tl.constexpr

    def __init__(self, ready_bars, empty_bars, bufs, num_stages):
        self.ready_bars = ready_bars
        self.empty_bars = empty_bars
        self.bufs = bufs
        self.num_stages = num_stages

    @triton.jit
    def init_barriers(self, count):
        """Init ready/empty mbarriers for all stages. Call from tid==0."""
        for s in tl.static_range(self.num_stages):
            mbarrier_init(smem_get_ptr(self.ready_bars, [s]), count)
            mbarrier_init(smem_get_ptr(self.empty_bars, [s]), 1)

    @triton.jit
    def invalidate_barriers(self):
        """Invalidate all mbarriers. Call from tid==0 after final sync."""
        for s in tl.static_range(self.num_stages):
            mbarrier_invalidate(smem_get_ptr(self.ready_bars, [s]))
            mbarrier_invalidate(smem_get_ptr(self.empty_bars, [s]))

    @triton.jit
    def producer_acquire(self, state: PipelineState):
        """Wait until buffer ``state.index`` is released by the consumer."""
        mbarrier_wait_parity(smem_get_ptr(self.empty_bars, [state.index]), state.phase ^ 1)

    @triton.jit
    def producer_tma_load(self, state: PipelineState, tmap, coord0, coord1, tile_bytes):
        """``mbarrier.expect_tx`` + TMA G2S 2D load. Gate with ``elect_one_sync``."""
        buf_ptr = smem_get_ptr(smem_index(self.bufs, state.index))
        bar_ptr = smem_get_ptr(self.ready_bars, [state.index])
        mbarrier_expect_tx(bar_ptr, tile_bytes)
        tma_load_2d(tmap, buf_ptr, bar_ptr, coord0, coord1)

    @triton.jit
    def consumer_wait(self, state: PipelineState):
        """Wait for producer to fill buffer ``state.index`` at current phase."""
        mbarrier_wait_parity(smem_get_ptr(self.ready_bars, [state.index]), state.phase)

    @triton.jit
    def consumer_tma_store(self, state: PipelineState, tmap, coord0, coord1):
        """TMA S2G 2D store + ``commit_group``. Gate with ``elect_one_sync``."""
        buf_smem = smem_index(self.bufs, state.index)
        tma_store_2d(tmap, smem_get_ptr(buf_smem), coord0, coord1)
        cp_async_bulk_commit_group()

    @triton.jit
    def consumer_release(self, state: PipelineState):
        """Signal buffer ``state.index`` as empty. Gate with ``elect_one_sync``."""
        mbarrier_arrive(smem_get_ptr(self.empty_bars, [state.index]))


# ===================================================================
# Standalone convenience
# ===================================================================


@jit
def store_and_wait(tmap_gen, buf_smem, coord0, coord1, pending: tl.constexpr):
    """TMA 2D store + commit_group + wait_group (non-pipelined convenience)."""
    tma_store_2d(tmap_gen, smem_get_ptr(buf_smem), coord0, coord1)
    cp_async_bulk_commit_group()
    cp_async_bulk_wait_group(pending, read=tlc.constexpr(True))


__all__ = [
    "PipelineState",
    "TmaPipeline",
    "store_and_wait",
]
