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

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_grouped_m
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

# MXFP4 GEMM kernel using tcgen05.mma.blockscaled (cta_group::2)
# A: (M, K) E2M1 packed as (M, K/2) uint8, K-major
# B: (N, K) E2M1 packed as (N, K/2) uint8, K-major
# SFA: (M*K/128, 32) E8M0 uint8, CUTLASS interleaved layout
# SFB: (N*K/128, 32) E8M0 uint8, CUTLASS interleaved layout
# D: (M, N) BF16
#
# Uses two native K=64 MXFP4 MMAs per CUTLASS-style K=128 mainloop stage.

PAIR_TILE_M = 256
PAIR_TILE_N = 256
CLUSTER_SIZE = 2
CTA_GROUP_SHAPE = (CLUSTER_SIZE, 1, 1)
CTA_TILE_M = PAIR_TILE_M // CLUSTER_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CLUSTER_SIZE
K_TILE = 128
K_TILE_PACKED = K_TILE // 2
A_SMEM_K = K_TILE_PACKED
NUM_STAGES = 7
NUM_EPI_STAGES = 1
GROUP_M = 16

# SF: CUTLASS interleaved layout — 32 m1-rows × 32 bytes per K=128 stage
# (128 M-rows packed into 32 rows × 16 bytes, padded to 32 for TMA alignment)
# With VS=32, four SF vectors are adjacent in each row's 4-byte region.
SF_SMEM_M = 32
SF_SMEM_K = 32

SMEM_A = CTA_TILE_M * A_SMEM_K  # uint8
SMEM_B = CTA_TILE_N_SHARD * A_SMEM_K  # uint8
SMEM_SFA = SF_SMEM_M * SF_SMEM_K
SMEM_SFB = SF_SMEM_M * CLUSTER_SIZE * SF_SMEM_K

SWIZZLE_CD_BYTES = 128
ST_TILE_M = CTA_TILE_M  # 128
ST_TILE_N = SWIZZLE_CD_BYTES // 2  # 64 (bf16)
NUM_STORES = PAIR_TILE_N // ST_TILE_N  # 4
NUM_TMA_STORE_STAGES = 2

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2  # 8


@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    frontend=dataflow_frontend,
    cluster_shape=(CLUSTER_SIZE, 1, 1),
    cta_group_shape=CTA_GROUP_SHAPE,
)
def mxfp4_gemm_sm100_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dSFA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dSFB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    with dataflow() as g:
        tma = g.role("tma", name="load_ab", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_TMA_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, A_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, A_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        SFA_smem = g.smem("SFA_smem", (SF_SMEM_M, SF_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        SFB0_smem = g.smem("SFB0_smem", (SF_SMEM_M, SF_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        SFB1_smem = g.smem("SFB1_smem", (SF_SMEM_M, SF_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_EPI_STAGES)
        SFA_tmem = g.tmem("SFA_tmem", (128, 8), dtype=ll.uint8, stages=1)
        SFB_tmem = g.tmem("SFB_tmem", (128, 16), dtype=ll.uint8, stages=1)

        stage_ab = g.depends(src=tma, dst=mma, via=[A_smem, B_smem, SFA_smem, SFB0_smem, SFB1_smem])
        stage_acc = g.depends(src=mma, dst=epi, via=[C_tmem])
        stage_store = g.depends(src=epi, dst=epi, via=[C_smem])
        A_plan = g.tma_transfer(inter_cta_coord_dim=1, inter_cta_stride=CTA_TILE_M)
        B_plan = g.tma_transfer(inter_cta_coord_dim=1, inter_cta_stride=CTA_TILE_N_SHARD)
        SFA_plan = g.tma_transfer(inter_cta_coord_dim=1, inter_cta_stride=SF_SMEM_M)
        SFB0_plan = g.tma_transfer(inter_cta_coord_dim=1, inter_cta_stride=SF_SMEM_M * CLUSTER_SIZE)
        SFB1_plan = g.tma_transfer(inter_cta_coord_dim=1, inter_cta_stride=SF_SMEM_M * CLUSTER_SIZE)

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta: ll.uint32 = ri.rank_in_group()
    cta_id: ll.uint32 = ll.blockIdx_x()
    cluster_id: ll.uint32 = cta_id // CLUSTER_SIZE
    num_clusters: ll.uint32 = num_ctas // CLUSTER_SIZE

    num_m_blocks: ll.uint32 = (M + PAIR_TILE_M - 1) // PAIR_TILE_M
    num_n_blocks: ll.uint32 = (N + PAIR_TILE_N - 1) // PAIR_TILE_N
    num_tiles: ll.uint32 = num_m_blocks * num_n_blocks
    num_k_blocks: ll.uint32 = (K + K_TILE - 1) // K_TILE

    if warp_idx == 0:
        el0: ll.uint32 = ll.elect_one()
        if el0 == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)
            ll.prefetch_tma_descriptor(dSFA)
            ll.prefetch_tma_descriptor(dSFB)
            ll.prefetch_tma_descriptor(dD)

    # ---- TMA role: load A, B, SFA, SFB ----
    with tma:
        el_tma: ll.uint32 = ll.elect_one()
        tile_tma: ll.uint32 = cluster_id
        while tile_tma < num_tiles:
            coord_m_tma, coord_n_tma = get_tile_coord_grouped_m(tile_tma, num_m_blocks, num_n_blocks, GROUP_M)

            offset_m_tma: ll.int32 = ll.int32(coord_m_tma * PAIR_TILE_M)
            offset_n_tma: ll.int32 = ll.int32(coord_n_tma * PAIR_TILE_N)

            kb: ll.uint32 = ll.uint32(0)
            while kb < num_k_blocks:
                stage_ab.bind(kb)
                offset_k_tma: ll.int32 = ll.int32(kb * A_SMEM_K)
                if el_tma == 1:
                    ri.tma_ld_2d(A_smem, dA, (offset_k_tma, offset_m_tma), A_plan, shape=(CTA_TILE_M, A_SMEM_K))
                    ri.tma_ld_2d(B_smem, dB, (offset_k_tma, offset_n_tma), B_plan, shape=(CTA_TILE_N_SHARD, A_SMEM_K))
                    sf_base: ll.int32 = ll.int32(coord_m_tma * num_k_blocks * SF_SMEM_M * CLUSTER_SIZE * CLUSTER_SIZE +
                                                 kb * SF_SMEM_M * CLUSTER_SIZE * CLUSTER_SIZE)
                    sf_base_b: ll.int32 = ll.int32(coord_n_tma * num_k_blocks * SF_SMEM_M * CLUSTER_SIZE *
                                                   CLUSTER_SIZE + kb * SF_SMEM_M * CLUSTER_SIZE * CLUSTER_SIZE)
                    ri.tma_ld_2d(SFA_smem, dSFA, (0, sf_base), SFA_plan, shape=(SF_SMEM_M, SF_SMEM_K))
                    ri.tma_ld_2d(SFB0_smem, dSFB, (0, sf_base_b), SFB0_plan, shape=(SF_SMEM_M, SF_SMEM_K))
                    ri.tma_ld_2d(SFB1_smem, dSFB, (0, sf_base_b + SF_SMEM_M), SFB1_plan, shape=(SF_SMEM_M, SF_SMEM_K))
                kb = kb + 1
            tile_tma = tile_tma + num_clusters

    # ---- MMA role: SF load + blockscaled MMA ----
    with mma:
        tile_mma: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0
        while tile_mma < num_tiles:
            stage_acc.bind(tile_iter)
            kb: ll.uint32 = ll.uint32(0)
            while kb < num_k_blocks:
                stage_ab.bind(kb)
                el_umma: ll.uint32 = ll.elect_one()
                if el_umma == 1:
                    sfa_desc: ll.uint64 = ri.make_umma_smem_desc(SFA_smem, (0, 0))
                    sfb_desc_0: ll.uint64 = ri.make_umma_smem_desc(SFB0_smem, (0, 0))
                    sfb_desc_1: ll.uint64 = ri.make_umma_smem_desc(SFB1_smem, (0, 0))

                    tsfa_addr: ll.uint32 = ll.uint32(SFA_tmem)
                    tsfb_addr: ll.uint32 = ll.uint32(SFB_tmem)

                    ri.tcgen05_cp_32x128b_warpx4_2sm(SFA_tmem, ll.uint32(SFA_tmem), sfa_desc)
                    ri.tcgen05_cp_32x128b_warpx4_2sm(SFB_tmem, ll.uint32(SFB_tmem), sfb_desc_0)
                    ri.tcgen05_cp_32x128b_warpx4_2sm(SFB_tmem, ll.uint32(SFB_tmem + 4), sfb_desc_1)
                    for ki in ll.unroll(range(2)):
                        ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, ki * 32))
                        bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, ki * 32))
                        accum: ll.uint32 = ll.uint32(1)
                        if kb == 0 and ki == 0:
                            accum = ll.uint32(0)
                        sf_addr_offset: ll.uint32 = ll.uint32(ki * 0x80000000)
                        ri.umma_mxfp_e2m1_e2m1_ss_2sm(
                            C_tmem,
                            ll.uint32(C_tmem),
                            ad,
                            bd,
                            accum,
                            SFA_tmem,
                            tsfa_addr + sf_addr_offset,
                            SFB_tmem,
                            tsfb_addr + sf_addr_offset,
                            out_shape=(PAIR_TILE_M, PAIR_TILE_N),
                        )
                kb = kb + 1
            tile_iter = tile_iter + 1
            tile_mma = tile_mma + num_clusters

    # ---- Epilogue: TMEM → SMEM → TMA store ----
    with epi:
        local_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = local_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0

        while tile_epi < num_tiles:
            stage_acc.bind(tile_iter)
            coord_m_epi, coord_n_epi = get_tile_coord_grouped_m(tile_epi, num_m_blocks, num_n_blocks, GROUP_M)

            for s in ll.unroll(range(NUM_STORES)):
                stage_store.bind((tile_iter, s))
                cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                    tmem_col: ll.uint32 = ll.uint32(C_tmem + s * ST_TILE_N + i * ELEMS_PER_BG)
                    smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(local_tid), ll.uint32(i))
                    ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                if epi_warp == 0:
                    el_st: ll.uint32 = ll.elect_one()
                    if el_st == 1:
                        offset_n_epi: ll.int32 = ll.int32(coord_n_epi * PAIR_TILE_N + s * ST_TILE_N)
                        offset_m_epi: ll.int32 = ll.int32(coord_m_epi * PAIR_TILE_M + cta * CTA_TILE_M)
                        ri.tma_st_2d(C_smem, dD, (offset_n_epi, offset_m_epi), shape=(ST_TILE_M, ST_TILE_N))

            tile_iter = tile_iter + 1
            tile_epi = tile_epi + num_clusters
