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
"""TMA pipeline tests using :class:`TmaPipeline` and :class:`PipelineState`
aggregates from :mod:`triton_dist.language.tma`.

Tests cover:
  - PipelineState.advance correctness
  - Single-warp pipeline (producer-then-consumer in same thread)
  - Optional SIMT pass between TMA load and TMA store
  - Various stage/tile counts and edge cases
"""
from __future__ import annotations

import torch
import triton.language as tl
import triton_dist
from triton.language import core as tlc

from triton_dist.language.smem_ops import allocate_smem, smem_index, smem_load, smem_store
from triton_dist.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.language.extra.cuda.tma_language import fence_async_shared
from triton_dist.language.tma import (
    ELEM_BF16,
    SWIZZLE_128B,
    PipelineState,
    TmaPipeline,
    create_tmap_2d,
    store_and_wait,
)
from triton_dist.tma_utils import create_tmap_scratch

# ===================================================================
# PipelineState.advance unit test
# ===================================================================


@triton_dist.jit
def _state_advance_kernel(
    out_idx_ptr,
    out_phase_ptr,
    NUM_STAGES: tl.constexpr,
    NUM_ITERS: tl.constexpr,
):
    """Write (index, phase) after each advance to output tensors."""
    state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
    for i in tl.static_range(NUM_ITERS):
        tl.store(out_idx_ptr + i, state.index)
        tl.store(out_phase_ptr + i, state.phase)
        state = state.advance()


def test_pipeline_state_advance():
    """PipelineState.advance produces correct index/phase sequences."""
    for num_stages in (2, 3, 5):
        n = num_stages * 4
        out_idx = torch.zeros(n, dtype=torch.int32, device="cuda")
        out_phase = torch.zeros(n, dtype=torch.int32, device="cuda")
        _state_advance_kernel[(1, )](out_idx, out_phase, num_stages, n, num_warps=1)
        torch.cuda.synchronize()

        idx_cpu = out_idx.cpu().tolist()
        phase_cpu = out_phase.cpu().tolist()
        expected_idx, expected_phase = 0, 0
        for i in range(n):
            assert idx_cpu[i] == expected_idx, (
                f"stages={num_stages} i={i}: idx expected {expected_idx} got {idx_cpu[i]}")
            assert phase_cpu[i] == expected_phase, (
                f"stages={num_stages} i={i}: phase expected {expected_phase} got {phase_cpu[i]}")
            expected_idx = (expected_idx + 1) % num_stages
            if expected_idx == 0:
                expected_phase ^= 1


# ===================================================================
# Single-warp pipeline kernel (producer + consumer in same thread)
# ===================================================================


@triton_dist.jit
def pipeline_kernel(
    in_ptr,
    out_ptr,
    tmap_scratch,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    NUM_TILES: tl.constexpr,
    SIMT_ADD1: tl.constexpr,
    NUM_THREADS: tl.constexpr,
):
    TILE_BYTES: tl.constexpr = BM * BN * 2
    EPT: tl.constexpr = (BM * BN) // NUM_THREADS

    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bufs = allocate_smem(tl.bfloat16, [NUM_STAGES, BM, BN])
    smem_ready = allocate_smem(tl.int64, [NUM_STAGES])
    smem_empty = allocate_smem(tl.int64, [NUM_STAGES])
    pipeline = TmaPipeline(smem_ready, smem_empty, smem_bufs, NUM_STAGES)

    _t = tid(0)
    stride_b = tl.cast(N * 2, tl.int64)
    one_c = tl.cast(1, tl.int32)

    tg_in = tl.cast(tmap_scratch, tl.uint64, bitcast=True)
    in_tmap = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tg_out = tl.cast(tmap_scratch + 128, tl.uint64, bitcast=True)
    out_tmap = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )

    if _t == 0:
        pipeline.init_barriers(one_c)
    __syncthreads()

    state = PipelineState(tl.cast(0, tl.int32), tl.cast(0, tl.int32), NUM_STAGES)
    one_bf16 = tl.cast(1.0, tl.bfloat16)

    for tt in tl.static_range(NUM_TILES):
        if tt >= NUM_STAGES:
            pipeline.producer_acquire(state)

        coord_row = tt * BM
        buf_view = smem_index(pipeline.bufs, state.index)
        if _t == 0:
            pipeline.producer_tma_load(
                state,
                in_tmap,
                0,
                coord_row,
                TILE_BYTES,
            )
        pipeline.consumer_wait(state)

        if SIMT_ADD1:
            __syncthreads()
            for k in tl.static_range(EPT):
                lin = _t * EPT + k
                row = lin // BN
                col = lin % BN
                v = smem_load(buf_view, [row, col])
                smem_store(buf_view, v + one_bf16, [row, col])
            __syncthreads()

        fence_async_shared()
        if _t == 0:
            store_and_wait(out_tmap, buf_view, 0, coord_row, tlc.constexpr(0))
        fence_async_shared()
        if _t == 0:
            pipeline.consumer_release(state)

        state = state.advance()
        __syncthreads()

    if _t == 0:
        pipeline.invalidate_barriers()
    __syncthreads()


def _run(num_warps: int, stages: int, tiles: int, simt: bool, seed: int):
    torch.manual_seed(seed)
    BM, BN = 64, 64
    M = tiles * BM
    N = BN
    nt = num_warps * 32
    assert (BM * BN) % nt == 0
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    scratch = create_tmap_scratch(1, num_descs_per_cta=2)
    pipeline_kernel[(1, )](
        inp,
        out,
        scratch,
        M,
        N,
        BM,
        BN,
        stages,
        tiles,
        simt,
        nt,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()
    if simt:
        exp = inp + 1.0
        return torch.allclose(out.float(), exp.float(), rtol=0.02, atol=0.02)
    return torch.equal(out, inp)


def test_pipeline_deep_multistage():
    assert _run(4, 4, 16, False, 1), "4-stage 16-tile identity"


def test_pipeline_simt_under_load():
    assert _run(2, 3, 10, True, 2), "SIMT +1 with 3-stage ring"


def test_pipeline_single_warp_tight():
    assert _run(1, 2, 5, False, 3), "1-warp 2-stage"


def test_pipeline_stages_equals_tiles():
    """Edge case: NUM_STAGES == NUM_TILES — no producer_acquire needed."""
    assert _run(2, 3, 3, False, 10), "stages == tiles identity"


def test_pipeline_single_tile():
    """Minimal pipeline: one tile, two stages."""
    assert _run(1, 2, 1, False, 20), "single-tile pipeline"


def test_pipeline_simt_multi_warp():
    """4-warp SIMT +1 with deeper ring and more tiles."""
    assert _run(4, 4, 8, True, 42), "4-warp SIMT +1 deep"


_TESTS = (
    test_pipeline_state_advance,
    test_pipeline_single_warp_tight,
    test_pipeline_deep_multistage,
    test_pipeline_simt_under_load,
    test_pipeline_stages_equals_tiles,
    test_pipeline_single_tile,
    test_pipeline_simt_multi_warp,
)

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"{t.__name__}: PASS")
