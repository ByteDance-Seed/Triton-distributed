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
"""SM100 GEMM Level 9: Matching DeepGEMM Architecture.

D = A @ B^T  (A: M×K, B: N×K, D: M×N, all BF16 row-major)

Key changes from Level 8 (matching DeepGEMM):
1. BLOCK_M=256, BLOCK_N=256, NUM_STAGES=4 (larger tiles, better L2 reuse)
2. Separated A/B SMEM layout (A stages contiguous, then B stages)
   Enables constant-stride __shfl_sync descriptor caching.
3. MMA warp: all 32 threads participate in barrier_wait + __shfl_sync.
   elect_one() gates only UMMA issuance + commit.
4. tmem_empty: ALL 128 epilogue threads on BOTH CTAs arrive at leader's
   barrier via cluster addressing (init count = CLUSTER_SIZE × 128 = 256).
5. L2 promotion L2_256B in TMA descriptors.
6. Scheduler swizzle group = 16 (was 8).
7. NUM_EPILOGUE_STAGES = 1 (no TMEM double buffering, TMEM_COLS=512).
8. 8 TMA stores per tile (4 N-stores × 2 M-waves).
"""
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.runtime.tma_descriptor import create_tma_2d_descriptor
import torch

BM, BN, BK = 256, 256, 64
CLUSTER_SIZE = 2
NUM_STAGES = 4
NUM_EPI_STAGES = 1
WAVE_BM = 128
NUM_MWAVES = BM // WAVE_BM  # 2
LOAD_N_PER_CTA = BN // CLUSTER_SIZE  # 128
UMMA_M, UMMA_N, UMMA_K = 256, BN, 16
NUM_THREADS = 256
NUM_EPI_THREADS = 128
GROUP_M = 16  # swizzle group = 16
TMEM_COLS = NUM_EPI_STAGES * NUM_MWAVES * BN  # 512

SMEM_A = BM * BK * 2  # 32768
SMEM_B = LOAD_N_PER_CTA * BK * 2  # 16384
TX_BYTES = SMEM_A + SMEM_B  # 49152
UMMA_SBO = 8 * BK * 2  # 1024
WAVE_A_DESC_OFF = (WAVE_BM * BK * 2) // 16  # 1024

# Swizzled CD output
SWIZZLE_CD_BYTES = 128
STORE_BM = WAVE_BM  # 128
STORE_BN = SWIZZLE_CD_BYTES // 2  # 64
NUM_STORES = BN // STORE_BN  # 4
NUM_TMA_STORE_STAGES = 2
SMEM_CD_PER_STAGE = STORE_BM * SWIZZLE_CD_BYTES  # 16384 bytes

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2  # 8
BGS_PER_SWIZZLE = SWIZZLE_CD_BYTES // BANK_GROUP_BYTES  # 8

DESC_HI = (((UMMA_SBO >> 4) & 0x3FFF) << 32) | (1 << 46) | (2 << 61)
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((UMMA_N // 8) << 17) | ((UMMA_M // 16) << 24)

# SMEM layout: separated A/B
# [CD (32768)] [A0..A3 (131072)] [B0..B3 (65536)] [barriers + tmem_ptr]
SMEM_CD_TOTAL = NUM_TMA_STORE_STAGES * SMEM_CD_PER_STAGE
SMEM_TOTAL = SMEM_CD_TOTAL + NUM_STAGES * SMEM_A + NUM_STAGES * SMEM_B + 1024

passes = PASSES["cuda"]


@lk.ll_kernel(backend="cuda", is_entry=True)
def gemm_level9_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    # --- CD store buffers ---
    cd_smem_0 = ll.empty([STORE_BM, STORE_BN], dtype=ll.bfloat16, scope="dynamic_shared")
    cd_smem_1 = ll.empty([STORE_BM, STORE_BN], dtype=ll.bfloat16, scope="dynamic_shared")

    # --- Separated A/B SMEM layout ---
    # All A stages contiguous (for shfl descriptor caching)
    A_smem = [ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    # All B stages contiguous
    B_smem = [ll.empty([LOAD_N_PER_CTA, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]

    # --- Barriers ---
    full_bars = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    empty_bars = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
    tmem_full_bars = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_EPI_STAGES)]
    tmem_empty_bars = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_EPI_STAGES)]
    tmem_addr = ll.empty([1], dtype=ll.uint32, scope="dynamic_shared")
    _pad = ll.empty([50], dtype=ll.uint64, scope="dynamic_shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    lane_idx: ll.int32 = ll.get_lane_idx()
    cta: ll.uint32 = ll.cluster_rank()
    is_leader: ll.int32 = ll.val_cast(cta, ll.int32) == 0
    cta_id: ll.uint32 = ll.blockIdx_x()

    num_m_blocks: ll.uint32 = (M + BM - 1) // BM
    num_n_blocks: ll.uint32 = (N + BN - 1) // BN
    num_tiles: ll.uint32 = num_m_blocks * num_n_blocks
    num_k_blocks: ll.uint32 = (K + BK - 1) // BK

    if warp_idx == 0:
        el0: ll.uint32 = ll.elect_one()
        if el0 == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)
            ll.prefetch_tma_descriptor(dD)

    if warp_idx == 1:
        el1: ll.uint32 = ll.elect_one()
        if el1 == 1:
            for s in ll.unroll(range(NUM_STAGES)):
                ll.init_smem_barrier(full_bars[s], CLUSTER_SIZE)
                ll.init_smem_barrier(empty_bars[s], 1)
            for e in ll.unroll(range(NUM_EPI_STAGES)):
                ll.init_smem_barrier(tmem_full_bars[e], 1)
                # ALL 128 epilogue threads × 2 CTAs = 256 arrivals
                ll.init_smem_barrier(tmem_empty_bars[e], CLUSTER_SIZE * NUM_EPI_THREADS)
            ll.fence_smem_barrier_init()

    if warp_idx == 2:
        ll.tmem_alloc(tmem_addr, TMEM_COLS)
    ll.cluster_sync()

    # Base addresses for descriptor construction
    # Separated layout → A stages have stride SMEM_A between them
    base_a0: ll.uint32 = ll.cvta_generic_to_shared(A_smem[0]) >> 4
    base_b0: ll.uint32 = ll.cvta_generic_to_shared(B_smem[0]) >> 4
    cd_base_0: ll.uint32 = ll.cvta_generic_to_shared(cd_smem_0)
    cd_base_1: ll.uint32 = ll.cvta_generic_to_shared(cd_smem_1)

    # ======================== TMA WARP ========================
    if warp_idx == 0:
        el_tma: ll.uint32 = ll.elect_one()
        if el_tma == 1:
            tma_stage: ll.int32 = 0
            tma_phase: ll.uint32 = ll.uint32(0)
            tile_tma: ll.uint32 = cta_id
            while tile_tma < num_tiles:
                tpg_t: ll.uint32 = num_n_blocks * GROUP_M
                gi_t: ll.uint32 = tile_tma // tpg_t
                fm_t: ll.uint32 = gi_t * GROUP_M
                mg_t: ll.uint32 = ll.val_cast(
                    ll.min_val(ll.val_cast(GROUP_M, ll.int32), ll.val_cast(num_m_blocks - fm_t, ll.int32)), ll.uint32)
                ig_t: ll.uint32 = tile_tma % tpg_t
                mb_t: ll.uint32 = fm_t + ig_t % mg_t
                nb_t: ll.uint32 = ig_t // mg_t

                m_coord: ll.int32 = ll.val_cast(mb_t * BM, ll.int32)
                n_coord: ll.int32 = ll.val_cast(nb_t * BN + cta * LOAD_N_PER_CTA, ll.int32)

                tma_kb: ll.uint32 = ll.uint32(0)
                while tma_kb < num_k_blocks:
                    ll.mbarrier_wait(empty_bars[tma_stage], tma_phase ^ 1)
                    # Level 9: leader expects TX for BOTH CTAs
                    if is_leader == 1:
                        ll.mbarrier_arrive_and_expect_tx(full_bars[tma_stage], TX_BYTES * CLUSTER_SIZE)
                    if is_leader == 0:
                        ll.mbarrier_arrive_cluster(full_bars[tma_stage], 0)
                    tma_kc: ll.int32 = ll.val_cast(tma_kb * BK, ll.int32)
                    ll.tma_load_2d_cg2(dA, full_bars[tma_stage], A_smem[tma_stage], tma_kc, m_coord)
                    ll.tma_load_2d_cg2(dB, full_bars[tma_stage], B_smem[tma_stage], tma_kc, n_coord)
                    tma_stage = tma_stage + 1
                    if tma_stage == NUM_STAGES:
                        tma_stage = 0
                        tma_phase = tma_phase ^ 1
                    tma_kb = tma_kb + 1
                tile_tma = tile_tma + num_ctas

    # ======================== MMA WARP (leader, all 32 threads) ========================
    if warp_idx == 1:
        if is_leader == 1:
            # __shfl_sync descriptor caching:
            # Each lane i stores stage i's descriptor lo (separated layout = constant stride)
            a_lo_base: ll.uint32 = base_a0 & 0x3FFF
            b_lo_base: ll.uint32 = base_b0 & 0x3FFF
            # lane_idx < NUM_STAGES (4) stores valid descriptors
            a_stride_16: ll.uint32 = ll.uint32(SMEM_A // 16)
            b_stride_16: ll.uint32 = ll.uint32(SMEM_B // 16)
            my_a_lo: ll.uint32 = a_lo_base + ll.uint32(lane_idx) * a_stride_16
            my_b_lo: ll.uint32 = b_lo_base + ll.uint32(lane_idx) * b_stride_16

            mma_stage: ll.int32 = 0
            mma_phase: ll.uint32 = ll.uint32(0)
            tile_mma: ll.uint32 = cta_id
            tile_mma_iter: ll.int32 = 0

            while tile_mma < num_tiles:
                epi_idx: ll.int32 = tile_mma_iter % NUM_EPI_STAGES  # always 0
                epi_phase: ll.uint32 = ll.uint32(tile_mma_iter // NUM_EPI_STAGES) & 1

                # All 32 threads wait for epilogue to release TMEM
                ll.mbarrier_wait(tmem_empty_bars[epi_idx], epi_phase ^ 1)
                ll.tcgen05_fence_after()

                mma_kb: ll.uint32 = ll.uint32(0)
                while mma_kb < num_k_blocks:
                    # All 32 threads wait for TMA data
                    ll.mbarrier_wait(full_bars[mma_stage], mma_phase)
                    ll.tcgen05_fence_after()

                    # All 32 threads shuffle to get current stage's descriptor lo
                    cur_a_lo: ll.uint32 = ll.__shfl_sync(0xFFFFFFFF, my_a_lo, mma_stage)
                    cur_b_lo: ll.uint32 = ll.__shfl_sync(0xFFFFFFFF, my_b_lo, mma_stage)

                    # Only elected thread issues UMMA
                    el_umma: ll.uint32 = ll.elect_one()
                    if el_umma == 1:
                        for ki in ll.unroll(range(0, BK, UMMA_K)):
                            k_off = ki * 2 // 16
                            b_lo: ll.uint32 = cur_b_lo + k_off
                            bd: ll.uint64 = (ll.uint64(b_lo) & 0x3FFF) | DESC_HI
                            for w in ll.unroll(range(NUM_MWAVES)):
                                a_lo: ll.uint32 = cur_a_lo + w * WAVE_A_DESC_OFF + k_off
                                ad: ll.uint64 = (ll.uint64(a_lo) & 0x3FFF) | DESC_HI
                                tmem_off: ll.uint32 = ll.uint32(epi_idx * NUM_MWAVES * BN + w * BN)
                                if ki == 0:
                                    acc_flag: ll.uint32 = ll.uint32(mma_kb > 0)
                                    ll.umma_f16_cg2(tmem_off, ad, bd, IDESC, acc_flag)
                                else:
                                    ll.umma_f16_cg2(tmem_off, ad, bd, IDESC, ll.uint32(1))

                    # Only elected thread commits
                    el_commit: ll.uint32 = ll.elect_one()
                    if el_commit == 1:
                        ll.umma_commit_2sm(empty_bars[mma_stage])
                        if mma_kb == num_k_blocks - 1:
                            ll.umma_commit_2sm(tmem_full_bars[epi_idx])

                    mma_stage = mma_stage + 1
                    if mma_stage == NUM_STAGES:
                        mma_stage = 0
                        mma_phase = mma_phase ^ 1
                    mma_kb = mma_kb + 1

                tile_mma_iter = tile_mma_iter + 1
                tile_mma = tile_mma + num_ctas

            # Extra wait for safe barrier destruction
            if tile_mma_iter > 0:
                last_idx: ll.int32 = (tile_mma_iter - 1) % NUM_EPI_STAGES
                last_ph: ll.uint32 = ll.uint32((tile_mma_iter - 1) // NUM_EPI_STAGES) & 1
                ll.mbarrier_wait(tmem_empty_bars[last_idx], last_ph)

    # ======================== EPILOGUE WARPS (both CTAs) ========================
    if warp_idx >= 4:
        local_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = local_tid // 32
        tile_epi: ll.uint32 = cta_id
        tile_epi_iter: ll.int32 = 0

        while tile_epi < num_tiles:
            tpg_e: ll.uint32 = num_n_blocks * GROUP_M
            gi_e: ll.uint32 = tile_epi // tpg_e
            fm_e: ll.uint32 = gi_e * GROUP_M
            mg_e: ll.uint32 = ll.val_cast(
                ll.min_val(ll.val_cast(GROUP_M, ll.int32), ll.val_cast(num_m_blocks - fm_e, ll.int32)), ll.uint32)
            ig_e: ll.uint32 = tile_epi % tpg_e
            mb_e: ll.uint32 = fm_e + ig_e % mg_e
            nb_e: ll.uint32 = ig_e // mg_e

            epi_idx_e: ll.int32 = tile_epi_iter % NUM_EPI_STAGES  # always 0
            epi_phase_e: ll.uint32 = ll.uint32(tile_epi_iter // NUM_EPI_STAGES) & 1

            ll.mbarrier_wait(tmem_full_bars[epi_idx_e], epi_phase_e)
            ll.tcgen05_fence_after()

            # 2 M-waves × 4 N-stores = 8 TMA stores per tile
            # tma_store_stage cycles 0,1,0,1,... (8 stores mod 2 = 0, resets each tile)
            for w in ll.unroll(range(NUM_MWAVES)):
                for s in ll.unroll(range(NUM_STORES)):
                    # Static TMA store stage: (w * NUM_STORES + s) % 2
                    _stage_idx = (w * NUM_STORES + s) % NUM_TMA_STORE_STAGES
                    _cd_base = cd_base_0 if _stage_idx == 0 else cd_base_1

                    # Wait for TMA store stage to be free
                    if epi_warp == 0:
                        el_sw: ll.uint32 = ll.elect_one()
                        if el_sw == 1:
                            ll.tma_store_wait_n(NUM_TMA_STORE_STAGES - 1)
                    ll.named_barrier_sync(1, NUM_EPI_THREADS)

                    # TMEM -> SMEM with 128B swizzle
                    for i in ll.unroll(range(STORE_BN // ELEMS_PER_BG)):
                        tmem_col: ll.uint32 = ll.uint32(epi_idx_e * NUM_MWAVES * BN + w * BN + s * STORE_BN +
                                                        i * ELEMS_PER_BG)
                        r0: ll.uint32 = ll.uint32(0)
                        r1: ll.uint32 = ll.uint32(0)
                        r2: ll.uint32 = ll.uint32(0)
                        r3: ll.uint32 = ll.uint32(0)
                        r4: ll.uint32 = ll.uint32(0)
                        r5: ll.uint32 = ll.uint32(0)
                        r6: ll.uint32 = ll.uint32(0)
                        r7: ll.uint32 = ll.uint32(0)
                        ll.tmem_load_8x(tmem_col, r0, r1, r2, r3, r4, r5, r6, r7)
                        ll.tmem_load_fence()
                        p0: ll.uint32 = ll.pack_bf16(r0, r1)
                        p1: ll.uint32 = ll.pack_bf16(r2, r3)
                        p2: ll.uint32 = ll.pack_bf16(r4, r5)
                        p3: ll.uint32 = ll.pack_bf16(r6, r7)
                        swz_col: ll.uint32 = ll.uint32(i) ^ (ll.uint32(local_tid) % BGS_PER_SWIZZLE)
                        smem_addr: ll.uint32 = _cd_base + \
                            ll.uint32(local_tid) * SWIZZLE_CD_BYTES + swz_col * BANK_GROUP_BYTES
                        ll.st_shared_128(smem_addr, p0, p1, p2, p3)

                    # tmem_empty: ALL threads arrive via cluster addressing on last w/s
                    if w == NUM_MWAVES - 1 and s == NUM_STORES - 1:
                        ll.tcgen05_fence_before()
                        ll.mbarrier_arrive_cluster(tmem_empty_bars[epi_idx_e], 0)
                    ll.__syncwarp()

                    # TMA store
                    ll.tma_store_fence()
                    ll.named_barrier_sync(1, NUM_EPI_THREADS)
                    if epi_warp == 0:
                        el_st: ll.uint32 = ll.elect_one()
                        if el_st == 1:
                            n_idx: ll.int32 = ll.val_cast(nb_e * BN + s * STORE_BN, ll.int32)
                            m_idx: ll.int32 = ll.val_cast(mb_e * BM + w * WAVE_BM, ll.int32)
                            if _stage_idx == 0:
                                ll.tma_store_2d_sm100(dD, cd_smem_0, n_idx, m_idx)
                            else:
                                ll.tma_store_2d_sm100(dD, cd_smem_1, n_idx, m_idx)
                            ll.tma_store_commit()

            tile_epi_iter = tile_epi_iter + 1
            tile_epi = tile_epi + num_ctas

        if epi_warp == 0:
            el_final: ll.uint32 = ll.elect_one()
            if el_final == 1:
                ll.tma_store_wait()

    ll.__syncthreads()
    ll.cluster_sync()
    if warp_idx == 2:
        ll.tmem_dealloc(0, TMEM_COLS)


def build_kernel():
    return gemm_level9_kernel.build(passes, codegen_cuda, grid=(1, 1, 1), block=(NUM_THREADS, 1, 1),
                                    shared_mem_bytes=SMEM_TOTAL, arch="sm_100a", cluster_dim=(CLUSTER_SIZE, 1, 1))


def run_gemm(kernel, A, B, M, N, K):
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    # L2 promotion L2_256B for all descriptors
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True, l2_promotion=3)
    dB = create_tma_2d_descriptor(B, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=LOAD_N_PER_CTA, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True,
                                  l2_promotion=3)
    dD = create_tma_2d_descriptor(D, gmem_inner_dim=N, gmem_outer_dim=M, smem_inner_dim=STORE_BN,
                                  smem_outer_dim=STORE_BM, gmem_outer_stride=N, swizzle_mode=128, oob_fill=False,
                                  l2_promotion=3)
    num_m_blocks = (M + BM - 1) // BM
    num_n_blocks = (N + BN - 1) // BN
    num_tiles = num_m_blocks * num_n_blocks
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    nc = min(num_sms, num_tiles)
    nc = (nc // CLUSTER_SIZE) * CLUSTER_SIZE
    kernel(dA, dB, dD, M, N, K, nc, grid=(nc, 1, 1))
    return D


def test(kernel, M, N, K):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = run_gemm(kernel, A, B, M, N, K)
    torch.cuda.synchronize()
    D_ref = torch.matmul(A.float(), B.float().t())
    cos = torch.nn.functional.cosine_similarity(D.float().flatten().unsqueeze(0), D_ref.flatten().unsqueeze(0)).item()
    ok = cos > 0.98
    print("  {}x{}x{}: cos={:.6f} {}".format(M, N, K, cos, "PASS" if ok else "FAIL"))
    return ok


def bench(kernel, M, N, K, iters=20):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    D = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    dA = create_tma_2d_descriptor(A, gmem_inner_dim=K, gmem_outer_dim=M, smem_inner_dim=BK, smem_outer_dim=BM,
                                  gmem_outer_stride=K, swizzle_mode=128, oob_fill=True, l2_promotion=3)
    dB = create_tma_2d_descriptor(B, gmem_inner_dim=K, gmem_outer_dim=N, smem_inner_dim=BK,
                                  smem_outer_dim=LOAD_N_PER_CTA, gmem_outer_stride=K, swizzle_mode=128, oob_fill=True,
                                  l2_promotion=3)
    dD = create_tma_2d_descriptor(D, gmem_inner_dim=N, gmem_outer_dim=M, smem_inner_dim=STORE_BN,
                                  smem_outer_dim=STORE_BM, gmem_outer_stride=N, swizzle_mode=128, oob_fill=False,
                                  l2_promotion=3)
    num_m_blocks = (M + BM - 1) // BM
    num_n_blocks = (N + BN - 1) // BN
    num_tiles = num_m_blocks * num_n_blocks
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    nc = min(num_sms, num_tiles)
    nc = (nc // CLUSTER_SIZE) * CLUSTER_SIZE
    for _ in range(3):
        kernel(dA, dB, dD, M, N, K, nc, grid=(nc, 1, 1))
    torch.cuda.synchronize()
    s_ev, e_ev = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s_ev.record()
    for _ in range(iters):
        kernel(dA, dB, dD, M, N, K, nc, grid=(nc, 1, 1))
    e_ev.record()
    torch.cuda.synchronize()
    ms = s_ev.elapsed_time(e_ev) / iters
    tflops = 2.0 * M * N * K / (ms * 1e9)
    print("  {}x{}x{}: {:.3f}ms, {:.1f} TFLOPS".format(M, N, K, ms, tflops))


if __name__ == "__main__":
    print("=" * 60)
    print("SM100 GEMM Level 9: DeepGEMM Architecture")
    print("=" * 60)
    kernel = build_kernel()
    print("Kernel built OK\n")
    print("=== Correctness ===")
    all_pass = True
    for size in [512, 1024, 2048, 4096]:
        ok = test(kernel, size, size, size)
        all_pass = all_pass and ok
    print("\nAll tests {}!".format("PASSED" if all_pass else "FAILED"))
    print("\n=== Benchmark ===")
    for size in [1024, 2048, 4096, 8192]:
        bench(kernel, size, size, size)
