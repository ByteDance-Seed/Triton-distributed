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
Persistent MXFP8 GEMM for gfx950 (MI355X): one work-group per CU walks over its output tiles.

Operands, scales and the 64-slot K step are those of mxfp8_gemm_gfx950. On top of that:
  - persistent tiles: work-group b takes tiles b' + t * grid, where b' renumbers b so that every
    XCD runs a contiguous range of tiles. The partial last round goes to work-groups
    0 .. rest - 1, i.e. rest / 8 CUs on every XCD: an XCD's clock depends on how many of its CUs
    are busy;
  - the K steps of consecutive tiles form one pipeline: the last two K steps of a tile load the
    first two of the next one. The first K step of a tile accumulates onto 0 (no zeroing), the
    last one does not read the next step's fragments (they would be live across the epilogue);
  - paired B fragments: B fragment f covers rows 32 (f // 2) + 4 (f % 2) + 8 a + b (a, b < 4) of
    the wave's 128, so A fragment i x B fragments 2p and 2p + 1 give a lane 8 consecutive columns;
  - full tiles go out through LDS: 32 rows x 256 columns at a time, then one buffer store per
    2 rows x 512 bytes. Partial tiles store 16 bytes (or single columns) per lane.

Requires K % 128 == 0, K >= 256, ldd % 8 == 0 and a 16-byte aligned D.
"""

import argparse

import torch

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.amd.cdna4.codegen.emitter import codegen_hip
from little_kernel.core.passes import PASSES
from benchmarks.amd.cdna4 import mxfp8_gemm_gfx950 as data_parallel
from benchmarks.amd.cdna4.mxfp8_gemm_gfx950 import A_LOADS, A_OFF, B_LOADS, B_OFF, BAR1, BAR2, BK, BM, BN, DMA_BEFORE_BAR2, DMA_BYTES, DMA_PER_TILE, FMT_FP8, FRAG_BYTES, GROUP_M, K_LANES, K_ROWS, K_WAVE_BYTES, NUM_THREADS, NUM_WAVES, NUM_XCDS, READS, SA_OFF, SB_OFF, SCALE_BYTES, SLOTS, SPAN, STAGE_BYTES, WAVE

EPI_OFF = 2 * STAGE_BYTES  # epilogue staging after the two stages: the next tile's DMAs are in flight
EPI_PITCH = 2 * BN + 16  # one staged row of bf16, padded against bank conflicts
LDS_BYTES = EPI_OFF + 32 * EPI_PITCH
assert LDS_BYTES <= 160 * 1024


def _dma_slot(j):
    return BAR1 + j * SPAN // DMA_PER_TILE


# A step's buffer descriptor is built in the slot before its first DMA: its scalar ops would delay the
# MFMA of a DMA slot.
RSRC_SLOT_A = _dma_slot(0) - 1
RSRC_SLOT_B = _dma_slot(A_LOADS) - 1
RSRC_SLOT_SA = _dma_slot(A_LOADS + B_LOADS) - 1
RSRC_SLOT_SB = _dma_slot(A_LOADS + B_LOADS + 1) - 1


@lk.ll_kernel(backend="hip", is_entry=False)
def xcd_remap(bid: ll.int32, count: ll.int32) -> ll.int32:
    """Work-group bid -> id in 0 .. count - 1 such that every XCD (work-groups go round-robin over the
    XCDs: XCD = bid % 8) gets a contiguous range."""
    per_xcd: ll.int32 = (count + NUM_XCDS - 1) // NUM_XCDS
    tall: ll.int32 = count % NUM_XCDS  # the first `tall` XCDs get per_xcd ids, the others one less
    if tall == 0:
        tall = NUM_XCDS
    xcd: ll.int32 = bid % NUM_XCDS
    if xcd < tall:
        return xcd * per_xcd + bid // NUM_XCDS
    return tall * per_xcd + (xcd - tall) * (per_xcd - 1) + bid // NUM_XCDS


@lk.ll_kernel(backend="hip", is_entry=False)
def tile_first_row(tile: ll.int32, tiles_m: ll.int32, tiles_n: ll.int32) -> ll.int32:
    """First output row of tile id `tile`: groups of GROUP_M M tiles walk down N together, so the tiles
    an XCD runs at the same time share rows of A and columns of B in its L2."""
    per_group: ll.int32 = GROUP_M * tiles_n
    first_m: ll.int32 = (tile // per_group) * GROUP_M
    group_rows: ll.int32 = tiles_m - first_m
    if group_rows > GROUP_M:
        group_rows = GROUP_M
    return (first_m + (tile % per_group) % group_rows) * BM


@lk.ll_kernel(backend="hip", is_entry=False)
def tile_first_col(tile: ll.int32, tiles_m: ll.int32, tiles_n: ll.int32) -> ll.int32:
    """First output column of tile id `tile` (see tile_first_row)."""
    per_group: ll.int32 = GROUP_M * tiles_n
    first_m: ll.int32 = (tile // per_group) * GROUP_M
    group_rows: ll.int32 = tiles_m - first_m
    if group_rows > GROUP_M:
        group_rows = GROUP_M
    return ((tile % per_group) // group_rows) * BN


@lk.ll_kernel(backend="hip", is_entry=True)
def mxfp8_gemm_gfx950_persistent(
    A: ll.Tensor[ll.uint8],
    B: ll.Tensor[ll.uint8],
    SFA: ll.Tensor[ll.int32],
    SFB: ll.Tensor[ll.int32],
    D: ll.Tensor[ll.bfloat16],
    M: ll.int32,
    N: ll.int32,
    K: ll.int32,
    lda: ll.int32,
    ldb: ll.int32,
    lda_sf: ll.int32,
    ldb_sf: ll.int32,
    ldd: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    smem = ll.empty([LDS_BYTES], dtype=ll.uint8, scope="dynamic_shared")

    tid: ll.int32 = ll.threadIdx_x()
    lane: ll.int32 = tid % WAVE
    wave: ll.int32 = ll.amdgcn_readfirstlane(tid // WAVE)
    wm: ll.int32 = wave // 2  # the wave's 128 x 128 quadrant: rows wm * 128, columns wn * 128
    wn: ll.int32 = wave % 2
    g: ll.int32 = lane // 16
    r16: ll.int32 = lane % 16
    k_steps: ll.int32 = K // BK

    # -- tiles of this work-group: first_tile + t * grid for t < full_rounds, then (bid < rest) one of
    #    the partial last round
    tiles_m: ll.int32 = (M + BM - 1) // BM
    tiles_n: ll.int32 = (N + BN - 1) // BN
    tiles: ll.int32 = tiles_m * tiles_n
    bid: ll.int32 = ll.blockIdx_x()
    grid: ll.int32 = ll.gridDim_x()
    full_rounds: ll.int32 = tiles // grid
    rest: ll.int32 = tiles - full_rounds * grid
    tile_count: ll.int32 = full_rounds + (1 if bid < rest else 0)
    first_tile: ll.int32 = xcd_remap(bid, grid)
    last_round_tile: ll.int32 = full_rounds * grid + xcd_remap(bid, rest)

    # -- global -> LDS: DMA j of a wave writes LDS rows K_ROWS * (j * NUM_WAVES + wave) .. + K_ROWS - 1,
    #    lane l fetches the logical chunk (l % K_LANES) ^ key(row) of its row (see mxfp8_gemm_gfx950).
    #    B rows use the key of the paired fragments: bits 1, 3 and 4 of the row.
    dma_row: ll.int32 = K_ROWS * wave + lane // K_LANES
    a_dma_key: ll.int32 = (dma_row >> 1) & 7
    b_dma_key: ll.int32 = ((dma_row >> 1) & 1) | (((dma_row >> 3) & 3) << 1)
    a_dma_offset: ll.int32 = dma_row * lda + ((lane % K_LANES) ^ a_dma_key) * DMA_BYTES
    b_dma_offset: ll.int32 = dma_row * ldb + ((lane % K_LANES) ^ b_dma_key) * DMA_BYTES
    a_dma_stride: ll.int32 = K_ROWS * NUM_WAVES * lda  # between DMA j and j + 1 of a wave
    b_dma_stride: ll.int32 = K_ROWS * NUM_WAVES * ldb
    scale_dma_offset: ll.int32 = (WAVE * wave + lane) * SCALE_BYTES

    # -- LDS -> registers. A fragment f: rows 16 f + r16 of the wave's 128. B fragment f: rows
    #    32 (f // 2) + 4 (f % 2) + b_row with b_row = 8 (r16 // 4) + r16 % 4. Lane (g, r16) reads
    #    logical chunks g and g + 4 of its row and byte g of the row's scale word. An address is
    #    (stage + lane part) + constant; the constant goes into the instruction's offset field.
    a_key: ll.int32 = (r16 >> 1) & 7
    b_row: ll.int32 = 8 * (r16 // 4) + r16 % 4
    b_key: ll.int32 = ((b_row >> 1) & 1) | (((b_row >> 3) & 3) << 1)
    a_read_lo: ll.int32 = (wm * 128 + r16) * BK + (g ^ a_key) * 16
    a_read_hi: ll.int32 = (wm * 128 + r16) * BK + ((g + 4) ^ a_key) * 16
    b_read_lo: ll.int32 = (wn * 128 + b_row) * BK + (g ^ b_key) * 16
    b_read_hi: ll.int32 = (wn * 128 + b_row) * BK + ((g + 4) ^ b_key) * 16
    a_scale_read: ll.int32 = SA_OFF + (wm * 128 + r16) * SCALE_BYTES + g
    b_scale_read: ll.int32 = SB_OFF + (wn * 128 + b_row) * SCALE_BYTES + g

    # -- epilogue through LDS: round i stages rows 16 i + r16 of both wave rows (32 rows x 256 columns;
    #    lane (g, r16) writes 8 columns per pair of B fragments); wave w then stores staged rows
    #    8 w .. 8 w + 7, two per buffer store (lane: row lane / 32, 16-byte chunk lane % 32)
    epi_write: ll.int32 = EPI_OFF + (wm * 16 + r16) * EPI_PITCH + (wn * 128 + 8 * g) * 2
    epi_read: ll.int32 = EPI_OFF + (8 * wave + lane // 32) * EPI_PITCH + (lane % 32) * 16
    epi_store: ll.int32 = ((wm * 128 + 8 * wn + lane // 32) * ldd + (lane % 32) * 8) * 2

    a_frag = ll.empty([16], dtype=ll.int32x4, scope="local")  # A fragment f, half h at 2 f + h
    b_frag = ll.empty([16], dtype=ll.int32x4, scope="local")
    a_scale = ll.empty([8], dtype=ll.uint8, scope="local")  # scale code of fragment f
    b_scale = ll.empty([8], dtype=ll.uint8, scope="local")
    acc = ll.empty([64], dtype=ll.float32x4, scope="local")  # A fragment i x B fragment j at 8 i + j
    zero = ll.zeros([1], dtype=ll.float32x4, scope="local")

    # -- prologue: K steps 0 and 1 of the first tile in flight (K >= 2 BK)
    tile_m0: ll.int32 = tile_first_row(first_tile, tiles_m, tiles_n)
    tile_n0: ll.int32 = tile_first_col(first_tile, tiles_m, tiles_n)
    next_t: ll.int32 = 1 if tile_count > 1 else 0
    next_tile: ll.int32 = last_round_tile if next_t == full_rounds else first_tile + next_t * grid
    next_m0: ll.int32 = tile_first_row(next_tile, tiles_m, tiles_n)
    next_n0: ll.int32 = tile_first_col(next_tile, tiles_m, tiles_n)
    # buffer descriptors at a tile's origin (loads past the matrix end return 0), rebuilt every K step
    rsrc_a: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(A, ll.int64(tile_m0) * lda, (M - tile_m0 - 1) * lda + K)
    rsrc_b: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(B, ll.int64(tile_n0) * ldb, (N - tile_n0 - 1) * ldb + K)
    rsrc_sa: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(SFA,
                                                           ll.int64(tile_m0) * SCALE_BYTES,
                                                           ((k_steps - 1) * lda_sf + M - tile_m0) * SCALE_BYTES)
    rsrc_sb: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(SFB,
                                                           ll.int64(tile_n0) * SCALE_BYTES,
                                                           ((k_steps - 1) * ldb_sf + N - tile_n0) * SCALE_BYTES)
    for stage in ll.unroll(range(2)):
        for j in ll.unroll(range(DMA_PER_TILE)):
            if j < A_LOADS:
                ll.amdgcn_buffer_load_lds(rsrc_a, smem,
                                          stage * STAGE_BYTES + A_OFF + (j * NUM_WAVES + wave) * K_WAVE_BYTES,
                                          DMA_BYTES, a_dma_offset, j * a_dma_stride + stage * BK)
            elif j < A_LOADS + B_LOADS:
                ll.amdgcn_buffer_load_lds(
                    rsrc_b, smem, stage * STAGE_BYTES + B_OFF + ((j - A_LOADS) * NUM_WAVES + wave) * K_WAVE_BYTES,
                    DMA_BYTES, b_dma_offset, (j - A_LOADS) * b_dma_stride + stage * BK)
            elif j == A_LOADS + B_LOADS:
                ll.amdgcn_buffer_load_lds(rsrc_sa, smem, stage * STAGE_BYTES + SA_OFF + wave * WAVE * SCALE_BYTES,
                                          SCALE_BYTES, scale_dma_offset, stage * lda_sf * SCALE_BYTES)
            else:
                ll.amdgcn_buffer_load_lds(rsrc_sb, smem, stage * STAGE_BYTES + SB_OFF + wave * WAVE * SCALE_BYTES,
                                          SCALE_BYTES, scale_dma_offset, stage * ldb_sf * SCALE_BYTES)
    ll.amdgcn_s_waitcnt_vmcnt(DMA_PER_TILE)
    ll.amdgcn_s_barrier()

    s: ll.int32 = 0  # K steps done by the work-group: step s sits in stage s % 2
    t: ll.int32 = 0  # tiles done
    while True:
        # -- fragments 0-3 of the tile's first K step, in the order its first quadrant needs them
        #    (B 0, A 0-3, B 1-3); every later step gets them from the step before
        first_stage: ll.int32 = (s % 2) * STAGE_BYTES
        for read in ll.unroll(range(READS)):
            item: ll.int32 = read // 3
            part: ll.int32 = read % 3
            is_b: ll.int32 = 1 if (item == 0 or item >= 5) else 0
            f: ll.int32 = 0 if item == 0 else (item - 1 if item <= 4 else item - 4)
            if is_b == 0:
                if part == 0:
                    a_frag[2 * f] = ll.amdgcn_lds_read_b128(smem, first_stage + a_read_lo + (A_OFF + f * FRAG_BYTES))
                elif part == 1:
                    a_frag[2 * f + 1] = ll.amdgcn_lds_read_b128(smem,
                                                                first_stage + a_read_hi + (A_OFF + f * FRAG_BYTES))
                else:
                    a_scale[f] = ll.amdgcn_lds_read_u8(smem, first_stage + a_scale_read + f * 16 * SCALE_BYTES)
            else:
                if part == 0:
                    b_frag[2 * f] = ll.amdgcn_lds_read_b128(
                        smem, first_stage + b_read_lo + (B_OFF + (32 * (f // 2) + 4 * (f % 2)) * BK))
                elif part == 1:
                    b_frag[2 * f + 1] = ll.amdgcn_lds_read_b128(
                        smem, first_stage + b_read_hi + (B_OFF + (32 * (f // 2) + 4 * (f % 2)) * BK))
                else:
                    b_scale[f] = ll.amdgcn_lds_read_u8(
                        smem, first_stage + b_scale_read + (32 * (f // 2) + 4 * (f % 2)) * SCALE_BYTES)
        # the compiler orders these reads freely; the K loop's waits assume its own order
        ll.amdgcn_s_waitcnt_lgkmcnt(0)

        # -- the tile's K steps: phase 0 is its first step, phase 1 the middle ones, phase 2 the last
        k: ll.int32 = 0
        for phase in ll.unroll(range(3)):
            phase_steps: ll.int32 = k_steps - 2
            if phase != 1:
                phase_steps = 1
            done: ll.int32 = 0
            while done < phase_steps:
                cur: ll.int32 = (s % 2) * STAGE_BYTES
                nxt: ll.int32 = STAGE_BYTES - cur
                # step s + 2, loaded into stage cur: K step k + 2 of this tile, or k + 2 - k_steps of the next
                # (after the last tile: of the same tile again, into a stage nobody reads)
                dma_m0: ll.int32 = tile_m0 if k + 2 < k_steps else next_m0
                dma_n0: ll.int32 = tile_n0 if k + 2 < k_steps else next_n0
                dma_k: ll.int32 = k + 2 if k + 2 < k_steps else k + 2 - k_steps
                for slot in ll.unroll(range(SLOTS)):
                    # MFMA of the slot; B goes into the MFMA's A operand, so acc[8 i + j] element e of lane
                    # (g, r16) is D[wm * 128 + 16 i + r16][wn * 128 + 32 (j // 2) + 4 (j % 2) + 8 g + e]
                    q: ll.int32 = slot // 16
                    mj: ll.int32 = (q >> 1) * 4 + (slot % 16) // 4
                    mi: ll.int32 = (q & 1) * 4 + slot % 4
                    if phase == 0:
                        acc[mi * 8 + mj] = ll.amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                            ll.amdgcn_concat_i32x4(b_frag[2 * mj], b_frag[2 * mj + 1]),
                            ll.amdgcn_concat_i32x4(a_frag[2 * mi], a_frag[2 * mi + 1]), zero[0], FMT_FP8, FMT_FP8, 0,
                            b_scale[mj], 0, a_scale[mi])
                    else:
                        acc[mi * 8 + mj] = ll.amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                            ll.amdgcn_concat_i32x4(b_frag[2 * mj], b_frag[2 * mj + 1]),
                            ll.amdgcn_concat_i32x4(a_frag[2 * mi], a_frag[2 * mi + 1]), acc[mi * 8 + mj], FMT_FP8,
                            FMT_FP8, 0, b_scale[mj], 0, a_scale[mi])
                    if slot < READS // 2:
                        # fragments 4-7 of this step (needed from the second quadrant on): A 4-7, then B 4-7
                        for h in ll.unroll(range(2)):
                            item4: ll.int32 = (2 * slot + h) // 3
                            part4: ll.int32 = (2 * slot + h) % 3
                            f4: ll.int32 = 4 + item4 % 4
                            if item4 < 4:
                                if part4 == 0:
                                    a_frag[2 * f4] = ll.amdgcn_lds_read_b128(
                                        smem, cur + a_read_lo + (A_OFF + f4 * FRAG_BYTES))
                                elif part4 == 1:
                                    a_frag[2 * f4 + 1] = ll.amdgcn_lds_read_b128(
                                        smem, cur + a_read_hi + (A_OFF + f4 * FRAG_BYTES))
                                else:
                                    a_scale[f4] = ll.amdgcn_lds_read_u8(smem,
                                                                        cur + a_scale_read + f4 * 16 * SCALE_BYTES)
                            else:
                                if part4 == 0:
                                    b_frag[2 * f4] = ll.amdgcn_lds_read_b128(
                                        smem, cur + b_read_lo + (B_OFF + (32 * (f4 // 2) + 4 * (f4 % 2)) * BK))
                                elif part4 == 1:
                                    b_frag[2 * f4 + 1] = ll.amdgcn_lds_read_b128(
                                        smem, cur + b_read_hi + (B_OFF + (32 * (f4 // 2) + 4 * (f4 % 2)) * BK))
                                else:
                                    b_scale[f4] = ll.amdgcn_lds_read_u8(
                                        smem, cur + b_scale_read + (32 * (f4 // 2) + 4 * (f4 % 2)) * SCALE_BYTES)
                    if slot == RSRC_SLOT_A:
                        rsrc_a = ll.amdgcn_make_buffer_rsrc(A, ll.int64(dma_m0) * lda, (M - dma_m0 - 1) * lda + K)
                    if slot == RSRC_SLOT_B:
                        rsrc_b = ll.amdgcn_make_buffer_rsrc(B, ll.int64(dma_n0) * ldb, (N - dma_n0 - 1) * ldb + K)
                    if slot == RSRC_SLOT_SA:
                        rsrc_sa = ll.amdgcn_make_buffer_rsrc(SFA,
                                                             ll.int64(dma_m0) * SCALE_BYTES,
                                                             ((k_steps - 1) * lda_sf + M - dma_m0) * SCALE_BYTES)
                    if slot == RSRC_SLOT_SB:
                        rsrc_sb = ll.amdgcn_make_buffer_rsrc(SFB,
                                                             ll.int64(dma_n0) * SCALE_BYTES,
                                                             ((k_steps - 1) * ldb_sf + N - dma_n0) * SCALE_BYTES)
                    if slot == BAR1:
                        # every wave has read stage cur: the DMAs of step s + 2 may overwrite it
                        ll.amdgcn_s_waitcnt_lgkmcnt(0)
                        ll.amdgcn_s_barrier()
                    # DMA j of step s + 2 in slot BAR1 + j * SPAN / DMA_PER_TILE
                    du: ll.int32 = slot - BAR1
                    dj: ll.int32 = (du * DMA_PER_TILE + SPAN - 1) // SPAN
                    if du >= 0 and dj < DMA_PER_TILE and (dj * SPAN) // DMA_PER_TILE == du:
                        if dj < A_LOADS:
                            ll.amdgcn_buffer_load_lds(rsrc_a, smem,
                                                      cur + A_OFF + (dj * NUM_WAVES + wave) * K_WAVE_BYTES, DMA_BYTES,
                                                      a_dma_offset, dj * a_dma_stride + dma_k * BK)
                        elif dj < A_LOADS + B_LOADS:
                            ll.amdgcn_buffer_load_lds(rsrc_b, smem,
                                                      cur + B_OFF + ((dj - A_LOADS) * NUM_WAVES + wave) * K_WAVE_BYTES,
                                                      DMA_BYTES, b_dma_offset,
                                                      (dj - A_LOADS) * b_dma_stride + dma_k * BK)
                        elif dj == A_LOADS + B_LOADS:
                            ll.amdgcn_buffer_load_lds(rsrc_sa, smem, cur + SA_OFF + wave * WAVE * SCALE_BYTES,
                                                      SCALE_BYTES, scale_dma_offset, dma_k * lda_sf * SCALE_BYTES)
                        else:
                            ll.amdgcn_buffer_load_lds(rsrc_sb, smem, cur + SB_OFF + wave * WAVE * SCALE_BYTES,
                                                      SCALE_BYTES, scale_dma_offset, dma_k * ldb_sf * SCALE_BYTES)
                    if slot == BAR2:
                        # step s + 1 has landed: only the DMAs of step s + 2 issued so far may be in flight
                        ll.amdgcn_s_waitcnt_vmcnt(DMA_BEFORE_BAR2)
                        ll.amdgcn_s_barrier()
                    if phase != 2 and slot >= BAR2 and slot < BAR2 + READS // 2:
                        # fragments 0-3 of the next step: B 0, A 0-3, B 1-3
                        for h in ll.unroll(range(2)):
                            item0: ll.int32 = (2 * (slot - BAR2) + h) // 3
                            part0: ll.int32 = (2 * (slot - BAR2) + h) % 3
                            is_b0: ll.int32 = 1 if (item0 == 0 or item0 >= 5) else 0
                            f0: ll.int32 = 0 if item0 == 0 else (item0 - 1 if item0 <= 4 else item0 - 4)
                            if is_b0 == 0:
                                if part0 == 0:
                                    a_frag[2 * f0] = ll.amdgcn_lds_read_b128(
                                        smem, nxt + a_read_lo + (A_OFF + f0 * FRAG_BYTES))
                                elif part0 == 1:
                                    a_frag[2 * f0 + 1] = ll.amdgcn_lds_read_b128(
                                        smem, nxt + a_read_hi + (A_OFF + f0 * FRAG_BYTES))
                                else:
                                    a_scale[f0] = ll.amdgcn_lds_read_u8(smem,
                                                                        nxt + a_scale_read + f0 * 16 * SCALE_BYTES)
                            else:
                                if part0 == 0:
                                    b_frag[2 * f0] = ll.amdgcn_lds_read_b128(
                                        smem, nxt + b_read_lo + (B_OFF + (32 * (f0 // 2) + 4 * (f0 % 2)) * BK))
                                elif part0 == 1:
                                    b_frag[2 * f0 + 1] = ll.amdgcn_lds_read_b128(
                                        smem, nxt + b_read_hi + (B_OFF + (32 * (f0 // 2) + 4 * (f0 % 2)) * BK))
                                else:
                                    b_scale[f0] = ll.amdgcn_lds_read_u8(
                                        smem, nxt + b_scale_read + (32 * (f0 // 2) + 4 * (f0 % 2)) * SCALE_BYTES)
                    ll.amdgcn_sched_barrier(0)
                k = k + 1
                s = s + 1
                done = done + 1

        # -- epilogue. Lane (g, r16) holds columns 32 p + 8 g .. + 7 of row 16 i + r16 of its quadrant in
        #    acc[8 i + 2 p] and acc[8 i + 2 p + 1].
        if tile_m0 + BM <= M and tile_n0 + BN <= N:
            rsrc_d: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(D,
                                                                  ll.int64(tile_m0) * ldd * 2, (M - tile_m0) * ldd * 2)
            for i in ll.unroll(range(8)):
                for p in ll.unroll(range(4)):
                    ll.amdgcn_lds_write_b128(smem, epi_write + 64 * p,
                                             ll.amdgcn_cvt_pk_bf16_f32x8(acc[8 * i + 2 * p], acc[8 * i + 2 * p + 1]))
                ll.amdgcn_s_waitcnt_lgkmcnt(0)
                ll.amdgcn_s_barrier()
                for row_pair in ll.unroll(range(4)):
                    ll.amdgcn_buffer_store_b128(ll.amdgcn_lds_read_b128(smem, epi_read + 2 * row_pair * EPI_PITCH),
                                                rsrc_d, epi_store, ((16 * i + 2 * row_pair) * ldd + tile_n0) * 2)
                ll.amdgcn_s_waitcnt_lgkmcnt(0)
                ll.amdgcn_s_barrier()
        else:
            for i in ll.unroll(range(8)):
                row: ll.int32 = tile_m0 + wm * 128 + 16 * i + r16
                for p in ll.unroll(range(4)):
                    col: ll.int32 = tile_n0 + wn * 128 + 32 * p + 8 * g
                    packed: ll.uint32x4 = ll.amdgcn_cvt_pk_bf16_f32x8(acc[8 * i + 2 * p], acc[8 * i + 2 * p + 1])
                    if row < M:
                        if col + 8 <= N:
                            ll.amdgcn_global_store_b128(D, (ll.int64(row) * ldd + col) * 2, packed)
                        else:
                            for e in ll.unroll(range(8)):
                                if col + e < N:
                                    ll.amdgcn_global_store_b16(D, (ll.int64(row) * ldd + col + e) * 2, packed[e // 2] >>
                                                               (16 * (e % 2)))
                    # keeps the scheduler from hoisting all 256 accumulator reads
                    ll.amdgcn_sched_barrier(0)

        t = t + 1
        if t >= tile_count:
            break
        tile_m0 = next_m0
        tile_n0 = next_n0
        next_t = t + 1 if t + 1 < tile_count else t
        next_tile = last_round_tile if next_t == full_rounds else first_tile + next_t * grid
        next_m0 = tile_first_row(next_tile, tiles_m, tiles_n)
        next_n0 = tile_first_col(next_tile, tiles_m, tiles_n)
    ll.amdgcn_s_waitcnt_vmcnt(0)


# ------------------------------------------------------------------------------ host side
def build_kernel(cache_dir=None, verbose=False):
    return mxfp8_gemm_gfx950_persistent.build(PASSES["hip"], codegen_hip, grid=(1, 1, 1), block=(NUM_THREADS, 1, 1),
                                              shared_mem_bytes=LDS_BYTES, arch="gfx950", cache_dir=cache_dir,
                                              verbose=verbose, disable_var_hoisting=True, waves_per_eu=1)


def run(kernel, a_q, sfa, b_q, sfb, d, num_cus=None):
    """One work-group per CU (at most one per tile)."""
    m, k = a_q.shape
    n = b_q.shape[0]
    assert k >= 2 * BK and k % BK == 0, f"K must be a multiple of {BK}, at least {2 * BK}"
    assert d.stride(0) % 8 == 0 and d.data_ptr() % 16 == 0, "rows of D must be 16-byte aligned"
    tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
    cus = num_cus or torch.cuda.get_device_properties(d.device).multi_processor_count
    kernel(a_q.view(torch.uint8), b_q.view(torch.uint8), sfa, sfb, d, m, n, k, a_q.stride(0), b_q.stride(0),
           sfa.stride(0), sfb.stride(0), d.stride(0), grid=(min(tiles, cus), 1, 1))


def same_as_data_parallel(kernel, reference_kernel, m, n, k):
    """Bitwise equal to mxfp8_gemm_gfx950 (same MFMAs in the same order)."""
    a_q, a_codes, b_q, b_codes = data_parallel.make_inputs(m, n, k, seed=1)
    sfa, sfb = data_parallel.pack_scales(a_codes), data_parallel.pack_scales(b_codes)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    d_ref = torch.empty_like(d)
    run(kernel, a_q, sfa, b_q, sfb, d)
    data_parallel.run(reference_kernel, a_q, sfa, b_q, sfb, d_ref)
    torch.cuda.synchronize()
    same = torch.equal(d, d_ref)
    print(f"  {m:5} x {n:5} x {k:5}: bitwise equal to mxfp8_gemm_gfx950: {same}")
    return same


def main():
    parser = argparse.ArgumentParser(description="LittleKernel persistent MXFP8 GEMM on gfx950")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    kernel = build_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    ok = True
    if not args.no_check:
        print("correctness (reference: dequantized operands, fp32 matmul):")
        # partial tiles in M and N; a column tail with ldd > N; a partial last round (289 tiles)
        for m, n, k, ldd in ((256, 256, 256, None), (512, 768, 1024, None), (1000, 1096, 2048, None),
                             (1000, 1100, 2048, 1104), (4352, 4352, 512, None), (4096, 4096, 4096, None)):
            ok &= data_parallel.check(kernel, m, n, k, launch=run, ldd=ldd)
        reference_kernel = data_parallel.build_kernel(cache_dir=args.cache_dir)
        for m, n, k in ((1000, 1096, 2048), (4352, 4352, 1024)):
            ok &= same_as_data_parallel(kernel, reference_kernel, m, n, k)
    if not args.no_bench:
        print("performance:")
        for m, n, k in ((4096, 4096, 7168), (8192, 8192, 8192)):
            data_parallel.bench(kernel, m, n, k, flush=True, launch=run)
            data_parallel.bench(kernel, m, n, k, flush=False, launch=run)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
