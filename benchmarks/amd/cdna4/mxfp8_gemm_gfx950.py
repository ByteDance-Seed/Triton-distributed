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
MXFP8 GEMM for gfx950 (MI355X) through the HIP backend.

    D[M][N] (bf16) = sum_k (A[m][k] * sa(m, k/32)) * (B[n][k] * sb(n, k/32))

A [M][K] and B [N][K] are FP8 E4M3, K contiguous (row strides lda / ldb bytes, multiples of 16).
Scales are UE8M0 codes, one per 32 K, packed four per int32 and stored MN-major: word w of row r
(K blocks 4w .. 4w + 3, byte i = block 4w + i) at SF[w * ld_sf + r] (DeepGEMM's packed layout).
Requires K % 128 == 0; M and N are arbitrary (rows past M / N read 0 and are not stored).

Structure (one work-group of 4 waves per 256 x 256 output tile; each wave a 128 x 128 quadrant):
  - global -> LDS with buffer_load ... lds DMAs (16 B per lane), XOR-swizzled 16 B chunks
    (chunk c of row r at c ^ ((r >> 1) & 7)); two LDS stages of one 128-wide K tile each;
  - one K step = 64 slots, one v_mfma_scale_f32_16x16x128_f8f6f4 per slot, in quadrant order
    (B 0-3 x A 0-3, B 0-3 x A 4-7, B 4-7 x A 0-3, B 4-7 x A 4-7). The wave holds one K step of
    operands (8 A + 8 B fragments): fragments 4-7 are read in slots 0-11, fragments 0-3 of the next
    step in slots 48-59, two LDS reads per slot;
  - the 18 DMAs of K step s + 2 go into the stage of step s once every wave has read it (barrier
    1, slot 16) and are spread over slots 16-61; barrier 2 (slot 48) waits for step s + 1.
"""

import argparse
import time

import torch

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.amd.cdna4.codegen.emitter import codegen_hip
from little_kernel.core.passes import PASSES

# ------------------------------------------------------------------------------ tile and schedule
BM = 256
BN = 256
BK = 128
NUM_WAVES = 4
NUM_THREADS = 64 * NUM_WAVES
WAVE = 64
DMA_BYTES = 16  # buffer_load_dwordx4 ... lds
K_LANES = BK // DMA_BYTES  # lanes per 128-byte K row
K_ROWS = WAVE // K_LANES  # rows per wave and DMA
K_WAVE_BYTES = K_ROWS * BK  # LDS bytes per wave and DMA
SCALE_BYTES = 4
A_LOADS = BM // (K_ROWS * NUM_WAVES)
B_LOADS = BN // (K_ROWS * NUM_WAVES)
DMA_PER_TILE = A_LOADS + B_LOADS + 2  # + one scale DMA for A and one for B
FRAG_BYTES = 16 * BK  # one 16-row fragment in LDS
A_OFF = 0
B_OFF = A_OFF + BM * BK
SA_OFF = B_OFF + BN * BK
SB_OFF = SA_OFF + BM * SCALE_BYTES
STAGE_BYTES = SB_OFF + BN * SCALE_BYTES
LDS_BYTES = 2 * STAGE_BYTES
SLOTS = 64  # MFMAs per K step and wave (8 x 8 fragments)
BAR1 = 16  # every wave has read its stage: the DMAs of step s + 2 may overwrite it
BAR2 = 48  # step s + 1 has landed
SPAN = 48  # slots over which the DMAs are spread
READS = 24  # LDS reads of one fragment group: 8 fragments x (2 operand halves + 1 scale code)
DMA_BEFORE_BAR2 = sum(1 for j in range(DMA_PER_TILE) if BAR1 + j * SPAN // DMA_PER_TILE < BAR2)
GROUP_M = 8  # tile order: groups of 8 M tiles walk down N together
NUM_XCDS = 8
FMT_FP8 = 0
assert BAR1 >= READS // 2 and BAR1 + SPAN <= SLOTS and SPAN >= DMA_PER_TILE


@lk.ll_kernel(backend="hip", is_entry=True)
def mxfp8_gemm_gfx950(
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
    wm: ll.int32 = wave // 2
    wn: ll.int32 = wave % 2
    g: ll.int32 = lane // 16
    r16: ll.int32 = lane % 16

    # -- output tile: work-groups go round-robin over the XCDs, so each XCD gets a contiguous range
    #    of tile ids, ordered in groups of GROUP_M M tiles (the tiles an XCD runs share A / B in L2)
    tiles_m: ll.int32 = (M + BM - 1) // BM
    tiles_n: ll.int32 = (N + BN - 1) // BN
    tiles: ll.int32 = tiles_m * tiles_n
    bid: ll.int32 = ll.blockIdx_x()
    per_xcd: ll.int32 = (tiles + NUM_XCDS - 1) // NUM_XCDS
    tall: ll.int32 = tiles % NUM_XCDS
    if tall == 0:
        tall = NUM_XCDS
    xcd: ll.int32 = bid % NUM_XCDS
    lid: ll.int32 = xcd * per_xcd + bid // NUM_XCDS
    if xcd >= tall:
        lid = tall * per_xcd + (xcd - tall) * (per_xcd - 1) + bid // NUM_XCDS
    per_group: ll.int32 = GROUP_M * tiles_n
    first_m: ll.int32 = (lid // per_group) * GROUP_M
    group_rows: ll.int32 = tiles_m - first_m
    if group_rows > GROUP_M:
        group_rows = GROUP_M
    rem: ll.int32 = lid % per_group
    m0: ll.int32 = (first_m + rem % group_rows) * BM
    n0: ll.int32 = (rem // group_rows) * BN
    kt_count: ll.int32 = K // BK

    # -- buffer descriptors at the tile origin; loads past the matrix end return 0
    ra: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(A, ll.int64(m0) * lda, (M - m0 - 1) * lda + K)
    rb: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(B, ll.int64(n0) * ldb, (N - n0 - 1) * ldb + K)
    rsa: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(SFA,
                                                       ll.int64(m0) * SCALE_BYTES,
                                                       ((kt_count - 1) * lda_sf + M - m0) * SCALE_BYTES)
    rsb: ll.AmdBufferRsrc = ll.amdgcn_make_buffer_rsrc(SFB,
                                                       ll.int64(n0) * SCALE_BYTES,
                                                       ((kt_count - 1) * ldb_sf + N - n0) * SCALE_BYTES)

    # -- DMA j of a wave writes LDS rows K_ROWS * (j * NUM_WAVES + wave) .. + K_ROWS - 1; lane l lands in
    #    LDS chunk l of that block (row l / K_LANES, physical chunk l % K_LANES), so it fetches the
    #    logical chunk (l % K_LANES) ^ key(row). The key does not depend on j: one offset per lane.
    dma_row: ll.int32 = K_ROWS * wave + lane // K_LANES
    dma_chunk: ll.int32 = (lane % K_LANES) ^ ((dma_row >> 1) & 7)
    a_voff: ll.int32 = dma_row * lda + dma_chunk * DMA_BYTES
    b_voff: ll.int32 = dma_row * ldb + dma_chunk * DMA_BYTES
    a_step: ll.int32 = K_ROWS * NUM_WAVES * lda
    b_step: ll.int32 = K_ROWS * NUM_WAVES * ldb
    s_voff: ll.int32 = (WAVE * wave + lane) * SCALE_BYTES

    # -- LDS -> registers: lane (g, r16) reads row r16 of a fragment, logical chunks g (half 0) and
    #    g + 4 (half 1), and byte g of the row's scale word (the code of K block g). A read address is
    #    (stage + lane part) + constant, the constant (operand offset + fragment offset < 64 KiB) going
    #    into the instruction's offset field.
    rkey: ll.int32 = (r16 >> 1) & 7
    a_lo: ll.int32 = (wm * 128 + r16) * BK + (g ^ rkey) * 16
    a_hi: ll.int32 = (wm * 128 + r16) * BK + ((g + 4) ^ rkey) * 16
    b_lo: ll.int32 = (wn * 128 + r16) * BK + (g ^ rkey) * 16
    b_hi: ll.int32 = (wn * 128 + r16) * BK + ((g + 4) ^ rkey) * 16
    sa_rd: ll.int32 = SA_OFF + (wm * 128 + r16) * SCALE_BYTES + g
    sb_rd: ll.int32 = SB_OFF + (wn * 128 + r16) * SCALE_BYTES + g

    fa = ll.empty([16], dtype=ll.int32x4, scope="local")  # A fragment f, half h at 2 f + h
    fb = ll.empty([16], dtype=ll.int32x4, scope="local")
    # scale code of fragment f. uint8, widened where the MFMA takes it: 32-bit copies of codes that live
    # across the loop's back edge would be narrowed by LLVM and re-widened at the loop head, which then
    # waits for all LDS reads of the previous step.
    sa = ll.empty([8], dtype=ll.uint8, scope="local")
    sb = ll.empty([8], dtype=ll.uint8, scope="local")
    acc = ll.zeros([64], dtype=ll.float32x4, scope="local")  # A fragment i x B fragment j at 8 i + j

    # -- prologue: K steps 0 and 1 in flight, wait for step 0, read its fragments 0-3
    for stage in ll.unroll(range(2)):
        for dj in ll.unroll(range(DMA_PER_TILE)):
            if dj < A_LOADS:
                ll.amdgcn_buffer_load_lds(ra, smem,
                                          stage * STAGE_BYTES + A_OFF + (dj * NUM_WAVES + wave) * K_WAVE_BYTES,
                                          DMA_BYTES, a_voff, dj * a_step + stage * BK)
            elif dj < A_LOADS + B_LOADS:
                ll.amdgcn_buffer_load_lds(
                    rb, smem, stage * STAGE_BYTES + B_OFF + ((dj - A_LOADS) * NUM_WAVES + wave) * K_WAVE_BYTES,
                    DMA_BYTES, b_voff, (dj - A_LOADS) * b_step + stage * BK)
            elif dj == A_LOADS + B_LOADS:
                ll.amdgcn_buffer_load_lds(rsa, smem, stage * STAGE_BYTES + SA_OFF + wave * WAVE * SCALE_BYTES,
                                          SCALE_BYTES, s_voff, stage * lda_sf * SCALE_BYTES)
            else:
                ll.amdgcn_buffer_load_lds(rsb, smem, stage * STAGE_BYTES + SB_OFF + wave * WAVE * SCALE_BYTES,
                                          SCALE_BYTES, s_voff, stage * ldb_sf * SCALE_BYTES)
    ll.amdgcn_s_waitcnt_vmcnt(DMA_PER_TILE)
    ll.amdgcn_s_barrier()
    for ri in ll.unroll(range(READS)):
        # fragments 0-3 in the order the first quadrant needs them: B 0, A 0-3, B 1-3
        item: ll.int32 = ri // 3
        part: ll.int32 = ri % 3
        is_b: ll.int32 = 1 if (item == 0 or item >= 5) else 0
        f: ll.int32 = 0 if item == 0 else (item - 1 if item <= 4 else item - 4)
        if is_b == 0:
            if part == 0:
                fa[2 * f] = ll.amdgcn_lds_read_b128(smem, a_lo + (A_OFF + f * FRAG_BYTES))
            elif part == 1:
                fa[2 * f + 1] = ll.amdgcn_lds_read_b128(smem, a_hi + (A_OFF + f * FRAG_BYTES))
            else:
                sa[f] = ll.amdgcn_lds_read_u8(smem, sa_rd + f * 16 * SCALE_BYTES)
        else:
            if part == 0:
                fb[2 * f] = ll.amdgcn_lds_read_b128(smem, b_lo + (B_OFF + f * FRAG_BYTES))
            elif part == 1:
                fb[2 * f + 1] = ll.amdgcn_lds_read_b128(smem, b_hi + (B_OFF + f * FRAG_BYTES))
            else:
                sb[f] = ll.amdgcn_lds_read_u8(smem, sb_rd + f * 16 * SCALE_BYTES)
    # the compiler orders these reads freely; the loop's waits for fragments 0-3 assume the loop's order
    ll.amdgcn_s_waitcnt_lgkmcnt(0)

    # -- main loop: K step s sits in stage s % 2 and loads step s + 2 into it (past the end: K tiles of
    #    this tile again, into a stage nobody reads). Every step reads the next step's fragments 0-3,
    #    the last one in vain: one loop body, no branch in the K loop.
    #    The body runs at least once (K >= BK) and the exit test sits at its end: with a zero-trip path,
    #    an accumulator would also be live past the loop, and the register allocator would shuffle
    #    accumulators between AGPRs instead of accumulating every MFMA in place.
    s: ll.int32 = 0
    while True:
        cur: ll.int32 = (s % 2) * STAGE_BYTES
        nxt: ll.int32 = STAGE_BYTES - cur
        kl: ll.int32 = s + 2
        if kl >= kt_count:
            kl = kl - kt_count
        for t in ll.unroll(range(SLOTS)):
            # MFMA of slot t; B goes into the MFMA's A operand, so acc[8 i + j] element e of lane l is
            # D[wm * 128 + 16 i + l % 16][wn * 128 + 16 j + 4 (l / 16) + e]
            q: ll.int32 = t // 16
            mj: ll.int32 = (q >> 1) * 4 + (t % 16) // 4
            mi: ll.int32 = (q & 1) * 4 + t % 4
            acc[mi * 8 + mj] = ll.amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
                ll.amdgcn_concat_i32x4(fb[2 * mj], fb[2 * mj + 1]), ll.amdgcn_concat_i32x4(fa[2 * mi], fa[2 * mi + 1]),
                acc[mi * 8 + mj], FMT_FP8, FMT_FP8, 0, sb[mj], 0, sa[mi])
            if t < READS // 2:
                # fragments 4-7 of this step (needed from the second quadrant on): A 4-7, then B 4-7
                for h in ll.unroll(range(2)):
                    item4: ll.int32 = (2 * t + h) // 3
                    part4: ll.int32 = (2 * t + h) % 3
                    f4: ll.int32 = 4 + item4 % 4
                    if item4 < 4:
                        if part4 == 0:
                            fa[2 * f4] = ll.amdgcn_lds_read_b128(smem, cur + a_lo + (A_OFF + f4 * FRAG_BYTES))
                        elif part4 == 1:
                            fa[2 * f4 + 1] = ll.amdgcn_lds_read_b128(smem, cur + a_hi + (A_OFF + f4 * FRAG_BYTES))
                        else:
                            sa[f4] = ll.amdgcn_lds_read_u8(smem, cur + sa_rd + f4 * 16 * SCALE_BYTES)
                    else:
                        if part4 == 0:
                            fb[2 * f4] = ll.amdgcn_lds_read_b128(smem, cur + b_lo + (B_OFF + f4 * FRAG_BYTES))
                        elif part4 == 1:
                            fb[2 * f4 + 1] = ll.amdgcn_lds_read_b128(smem, cur + b_hi + (B_OFF + f4 * FRAG_BYTES))
                        else:
                            sb[f4] = ll.amdgcn_lds_read_u8(smem, cur + sb_rd + f4 * 16 * SCALE_BYTES)
            if t == BAR1:
                ll.amdgcn_s_waitcnt_lgkmcnt(0)
                ll.amdgcn_s_barrier()
            # DMA j of step s + 2 in slot BAR1 + j * SPAN / DMA_PER_TILE
            du: ll.int32 = t - BAR1
            dj2: ll.int32 = (du * DMA_PER_TILE + SPAN - 1) // SPAN
            if du >= 0 and dj2 < DMA_PER_TILE and (dj2 * SPAN) // DMA_PER_TILE == du:
                if dj2 < A_LOADS:
                    ll.amdgcn_buffer_load_lds(ra, smem, cur + A_OFF + (dj2 * NUM_WAVES + wave) * K_WAVE_BYTES,
                                              DMA_BYTES, a_voff, dj2 * a_step + kl * BK)
                elif dj2 < A_LOADS + B_LOADS:
                    ll.amdgcn_buffer_load_lds(rb, smem,
                                              cur + B_OFF + ((dj2 - A_LOADS) * NUM_WAVES + wave) * K_WAVE_BYTES,
                                              DMA_BYTES, b_voff, (dj2 - A_LOADS) * b_step + kl * BK)
                elif dj2 == A_LOADS + B_LOADS:
                    ll.amdgcn_buffer_load_lds(rsa, smem, cur + SA_OFF + wave * WAVE * SCALE_BYTES, SCALE_BYTES, s_voff,
                                              kl * lda_sf * SCALE_BYTES)
                else:
                    ll.amdgcn_buffer_load_lds(rsb, smem, cur + SB_OFF + wave * WAVE * SCALE_BYTES, SCALE_BYTES, s_voff,
                                              kl * ldb_sf * SCALE_BYTES)
            if t == BAR2:
                # step s + 1 has landed: only the DMAs of step s + 2 issued so far may be in flight
                ll.amdgcn_s_waitcnt_vmcnt(DMA_BEFORE_BAR2)
                ll.amdgcn_s_barrier()
            if t >= BAR2 and t < BAR2 + READS // 2:
                # fragments 0-3 of the next step: B 0, A 0-3, B 1-3
                for h in ll.unroll(range(2)):
                    item0: ll.int32 = (2 * (t - BAR2) + h) // 3
                    part0: ll.int32 = (2 * (t - BAR2) + h) % 3
                    isb0: ll.int32 = 1 if (item0 == 0 or item0 >= 5) else 0
                    f0: ll.int32 = 0 if item0 == 0 else (item0 - 1 if item0 <= 4 else item0 - 4)
                    if isb0 == 0:
                        if part0 == 0:
                            fa[2 * f0] = ll.amdgcn_lds_read_b128(smem, nxt + a_lo + (A_OFF + f0 * FRAG_BYTES))
                        elif part0 == 1:
                            fa[2 * f0 + 1] = ll.amdgcn_lds_read_b128(smem, nxt + a_hi + (A_OFF + f0 * FRAG_BYTES))
                        else:
                            sa[f0] = ll.amdgcn_lds_read_u8(smem, nxt + sa_rd + f0 * 16 * SCALE_BYTES)
                    else:
                        if part0 == 0:
                            fb[2 * f0] = ll.amdgcn_lds_read_b128(smem, nxt + b_lo + (B_OFF + f0 * FRAG_BYTES))
                        elif part0 == 1:
                            fb[2 * f0 + 1] = ll.amdgcn_lds_read_b128(smem, nxt + b_hi + (B_OFF + f0 * FRAG_BYTES))
                        else:
                            sb[f0] = ll.amdgcn_lds_read_u8(smem, nxt + sb_rd + f0 * 16 * SCALE_BYTES)
            ll.amdgcn_sched_barrier(0)
        s = s + 1
        if s >= kt_count:
            break

    ll.amdgcn_s_waitcnt_vmcnt(0)
    ll.amdgcn_s_waitcnt_lgkmcnt(0)
    ll.amdgcn_s_barrier()

    # -- epilogue: lane (g, r16) stores columns 16 j + 4 g .. + 3 of row 16 i + r16 of its quadrant
    for i in ll.unroll(range(8)):
        row: ll.int32 = m0 + wm * 128 + 16 * i + r16
        for j in ll.unroll(range(8)):
            col: ll.int32 = n0 + wn * 128 + 16 * j + 4 * g
            if row < M and col < N:
                ll.amdgcn_global_store_b64(D, (ll.int64(row) * ldd + col) * 2,
                                           ll.amdgcn_cvt_pk_bf16_f32x4(acc[i * 8 + j]))


# ------------------------------------------------------------------------------ host side
E4M3_MAX = 448.0
SF_VEC = 32  # one scale per 32 K


def build_kernel(cache_dir=None, verbose=False):
    return mxfp8_gemm_gfx950.build(PASSES["hip"], codegen_hip, grid=(1, 1, 1), block=(NUM_THREADS, 1, 1),
                                   shared_mem_bytes=LDS_BYTES, arch="gfx950", cache_dir=cache_dir, verbose=verbose,
                                   disable_var_hoisting=True, waves_per_eu=1)


def quantize_mxfp8(x: torch.Tensor):
    """(E4M3 [mn, k], UE8M0 codes [mn, k / 32] as uint8): per 32 K, scale = 2^ceil(log2(amax / 448))."""
    mn, k = x.shape
    assert k % SF_VEC == 0
    blocks = x.float().reshape(mn, k // SF_VEC, SF_VEC)
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    exp = torch.ceil(torch.log2(amax / E4M3_MAX)).clamp(-127, 127)
    q = (blocks / torch.exp2(exp)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).reshape(mn, k)
    return q, (exp.squeeze(-1) + 127).to(torch.uint8)


def pack_scales(codes: torch.Tensor) -> torch.Tensor:
    """UE8M0 codes [mn, k / 32] -> int32 [k / 128][align(mn, 4)]: four codes per word, MN-major."""
    mn, kb = codes.shape
    assert kb % 4 == 0
    words = codes.contiguous().view(torch.int32)  # [mn, k / 128], byte i = K block 4 w + i
    ld = (mn + 3) // 4 * 4
    out = torch.zeros(kb // 4, ld, dtype=torch.int32, device=codes.device)
    out[:, :mn] = words.t()
    return out


def dequantize(q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    return q.float() * torch.exp2(codes.float() - 127).repeat_interleave(SF_VEC, dim=1)


def run(kernel, a_q, sfa, b_q, sfb, d):
    m, k = a_q.shape
    n = b_q.shape[0]
    assert k >= BK and k % BK == 0, f"K must be a positive multiple of {BK}"
    tiles = ((m + BM - 1) // BM) * ((n + BN - 1) // BN)
    kernel(a_q.view(torch.uint8), b_q.view(torch.uint8), sfa, sfb, d, m, n, k, a_q.stride(0), b_q.stride(0),
           sfa.stride(0), sfb.stride(0), d.stride(0), grid=(tiles, 1, 1))


def make_inputs(m, n, k, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn(m, k, device="cuda", generator=gen)
    b = torch.randn(n, k, device="cuda", generator=gen)
    (a_q, a_codes), (b_q, b_codes) = quantize_mxfp8(a), quantize_mxfp8(b)
    return a_q, a_codes, b_q, b_codes


def check(kernel, m, n, k, launch=None, ldd=None):
    """D against an fp32 matmul of the dequantized operands; D has row stride ldd (default n), the
    columns past n must stay untouched."""
    launch = launch or run
    a_q, a_codes, b_q, b_codes = make_inputs(m, n, k, seed=m + n + k)
    d_rows = torch.full((m, ldd or n), float("nan"), device="cuda", dtype=torch.bfloat16)
    d = d_rows[:, :n]
    launch(kernel, a_q, pack_scales(a_codes), b_q, pack_scales(b_codes), d)
    torch.cuda.synchronize()
    ref = dequantize(a_q, a_codes) @ dequantize(b_q, b_codes).t()
    ok = torch.allclose(d.float(), ref, atol=1e-2, rtol=1e-2) and bool(d_rows[:, n:].isnan().all())
    diff = 1 - 2 * (d.double() * ref.double()).sum() / (d.double().square() + ref.double().square()).sum()
    stride = f", ldd {ldd}" if ldd else ""
    print(f"  {m:5} x {n:5} x {k:5}{stride}: allclose(1e-2) {'ok' if ok else 'FAIL'}, calc_diff {float(diff):.2e}")
    return ok


def bench(kernel, m, n, k, flush=True, calls=30, rounds=5, warmup_s=1.0, launch=None):
    """Kernel time from the torch profiler, 256 MB written before every call (flush). Median of rounds."""
    launch = launch or run
    a_q, a_codes, b_q, b_codes = make_inputs(m, n, k)
    sfa, sfb = pack_scales(a_codes), pack_scales(b_codes)
    d = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty(int(256e6 // 4), dtype=torch.int, device="cuda")
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        launch(kernel, a_q, sfa, b_q, sfb, d)
        torch.cuda.synchronize()
    times = []
    for _ in range(rounds):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(calls):
                if flush:
                    scratch.zero_()
                launch(kernel, a_q, sfa, b_q, sfb, d)
            torch.cuda.synchronize()
        times.append(sum(e.device_time_total for e in prof.key_averages() if "mxfp8_gemm_gfx950" in e.key) / calls)
    us = sorted(times)[len(times) // 2]
    print(f"  {m:5} x {n:5} x {k:5} ({'flush' if flush else 'no flush'}): {us:7.1f} us, "
          f"{2 * m * n * k / us / 1e6:6.0f} TFLOPS")
    return us


def main():
    parser = argparse.ArgumentParser(description="LittleKernel MXFP8 GEMM on gfx950")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    kernel = build_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    ok = True
    if not args.no_check:
        print("correctness (reference: dequantized operands, fp32 matmul):")
        for m, n, k in ((256, 256, 128), (512, 768, 1024), (1000, 1100, 2048), (4096, 4096, 4096)):
            ok &= check(kernel, m, n, k)
    if not args.no_bench:
        print("performance:")
        for m, n, k in ((4096, 4096, 7168), (8192, 8192, 8192)):
            bench(kernel, m, n, k, flush=True)
            bench(kernel, m, n, k, flush=False)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
