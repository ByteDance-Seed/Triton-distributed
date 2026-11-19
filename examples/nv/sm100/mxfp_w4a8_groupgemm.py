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
"""Persistent SM100 grouped GEMM with MXFP8 activations and MXFP4 weights.

A is E4M3 with one E8M0 scale per 32 K elements. B is packed E2M1 with the same scale granularity. Each two-CTA group
either reads one host work item or derives its uniform-expert coordinates online.
"""

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_along_n
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

PAIR_TILE_M = 256
PAIR_TILE_N = 256
CLUSTER_SIZE = 2
CTA_TILE_M = PAIR_TILE_M // CLUSTER_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CLUSTER_SIZE
K_TILE = 128
A_SMEM_K = K_TILE
B_SMEM_K = K_TILE
NUM_STAGES = 5
NUM_ACC_STAGES = 1

SF_SMEM_M = 32
SF_SMEM_K = 32

SMEM_A = CTA_TILE_M * A_SMEM_K
SMEM_B = CTA_TILE_N_SHARD * B_SMEM_K
SMEM_SFA = SF_SMEM_M * SF_SMEM_K
SMEM_SFB = SF_SMEM_M * CLUSTER_SIZE * SF_SMEM_K

ST_TILE_M = CTA_TILE_M
ST_TILE_N = 64
NUM_STORES = PAIR_TILE_N // ST_TILE_N
NUM_STORE_STAGES = 2
ELEMS_PER_BANK_GROUP = 8

NUM_THREADS = 8 * 32
SHARED_MEM_BYTES = NUM_STORE_STAGES * ST_TILE_M * ST_TILE_N * 2 + NUM_STAGES * (SMEM_A + SMEM_B + SMEM_SFA +
                                                                                SMEM_SFB) + 1024


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend, cluster_shape=(CLUSTER_SIZE, 1, 1),
              cta_group_shape=(2, 1, 1))
def mxfp_w4a8_groupgemm_sm100_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dSFA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dSFB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    work_experts: ll.ptr[ll.int32],
    work_m_blocks: ll.ptr[ll.int32],
    work_n_blocks: ll.ptr[ll.int32],
    K: ll.uint32,
    num_work: ll.uint32,
    num_clusters: ll.uint32,
    blocks_per_expert: ll.uint32,
    num_n_blocks: ll.uint32,
    use_worklist: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        tma = g.role("tma", name="load_ab", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, A_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, B_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K",
                        tma_transaction_bits=4)
        SFA_smem = g.smem("SFA_smem", (SF_SMEM_M, SF_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        SFB0_smem = g.smem("SFB0_smem", (SF_SMEM_M, SF_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        SFB1_smem = g.smem("SFB1_smem", (SF_SMEM_M, SF_SMEM_K), dtype=ll.uint8, stages=NUM_STAGES, major="K")
        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_ACC_STAGES)
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
    cluster_id: ll.uint32 = ll.blockIdx_x() // CLUSTER_SIZE
    num_k_blocks: ll.uint32 = K // K_TILE

    if warp_idx == 0 and ll.elect_one():
        ll.prefetch_tma_descriptor(dA)
        ll.prefetch_tma_descriptor(dB)
        ll.prefetch_tma_descriptor(dSFA)
        ll.prefetch_tma_descriptor(dSFB)
        ll.prefetch_tma_descriptor(dD)

    with tma:
        tma_issuer: ll.uint32 = ll.elect_one()
        work_tma: ll.uint32 = cluster_id
        while work_tma < num_work:
            expert_tma: ll.uint32 = ll.uint32(0)
            coord_m_tma: ll.uint32 = ll.uint32(0)
            coord_n_tma: ll.uint32 = ll.uint32(0)
            if use_worklist == 1:
                expert_tma = ll.uint32(work_experts[work_tma])
                coord_m_tma = ll.uint32(work_m_blocks[work_tma])
                coord_n_tma = ll.uint32(work_n_blocks[work_tma])
            else:
                coord_m_tma, coord_n_tma = get_tile_coord_along_n(work_tma, num_n_blocks)
                expert_tma = coord_m_tma // blocks_per_expert
            kb_tma: ll.uint32 = ll.uint32(0)
            while kb_tma < num_k_blocks:
                stage_ab.bind((work_tma, kb_tma))
                if tma_issuer == 1:
                    a_k: ll.int32 = ll.int32(kb_tma * A_SMEM_K)
                    b_k: ll.int32 = ll.int32(kb_tma * K_TILE)
                    offset_m: ll.int32 = ll.int32(coord_m_tma * PAIR_TILE_M)
                    offset_n: ll.int32 = ll.int32(coord_n_tma * PAIR_TILE_N)
                    sf: ll.int32 = ll.int32(kb_tma * SF_SMEM_M * 4)
                    sf_a: ll.int32 = ll.int32(coord_m_tma * num_k_blocks * SF_SMEM_M * 4 + sf)
                    sf_b: ll.int32 = ll.int32(coord_n_tma * num_k_blocks * SF_SMEM_M * 4 + sf)
                    ri.tma_ld_2d(A_smem, dA, (a_k, offset_m), A_plan, shape=(CTA_TILE_M, A_SMEM_K))
                    ri.tma_ld_3d(B_smem, dB, (b_k, offset_n, ll.int32(expert_tma)), B_plan,
                                 shape=(CTA_TILE_N_SHARD, B_SMEM_K))
                    ri.tma_ld_2d(SFA_smem, dSFA, (0, sf_a), SFA_plan, shape=(SF_SMEM_M, SF_SMEM_K))
                    ri.tma_ld_3d(SFB0_smem, dSFB, (0, sf_b, ll.int32(expert_tma)), SFB0_plan,
                                 shape=(SF_SMEM_M, SF_SMEM_K))
                    ri.tma_ld_3d(SFB1_smem, dSFB, (0, sf_b + SF_SMEM_M, ll.int32(expert_tma)), SFB1_plan,
                                 shape=(SF_SMEM_M, SF_SMEM_K))
                kb_tma = kb_tma + 1
            work_tma = work_tma + num_clusters

    with mma:
        mma_issuer: ll.uint32 = ll.elect_one()
        work_mma: ll.uint32 = cluster_id
        work_iter: ll.int32 = 0
        while work_mma < num_work:
            stage_acc.bind(work_iter)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_blocks:
                stage_ab.bind((work_mma, kb_mma))
                if mma_issuer == 1:
                    sfa_desc: ll.uint64 = ri.make_umma_smem_desc(SFA_smem, (0, 0))
                    sfb_desc_0: ll.uint64 = ri.make_umma_smem_desc(SFB0_smem, (0, 0))
                    sfb_desc_1: ll.uint64 = ri.make_umma_smem_desc(SFB1_smem, (0, 0))
                    sfa_base: ll.uint32 = ll.uint32(SFA_tmem)
                    sfb_base: ll.uint32 = ll.uint32(SFB_tmem)
                    ri.tcgen05_cp_32x128b_warpx4_2sm(SFA_tmem, sfa_base, sfa_desc)
                    ri.tcgen05_cp_32x128b_warpx4_2sm(SFB_tmem, sfb_base, sfb_desc_0)
                    ri.tcgen05_cp_32x128b_warpx4_2sm(SFB_tmem, sfb_base + 4, sfb_desc_1)

                    for ki in ll.unroll(range(4)):
                        ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, ki * 32))
                        bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, ki * 32))
                        accum: ll.uint32 = ll.uint32(1)
                        if kb_mma == 0 and ki == 0:
                            accum = ll.uint32(0)
                        sf_offset: ll.uint32 = ll.uint32(ki) << 30
                        ri.umma_mxfp_e4m3_e2m1_ss_2sm(
                            C_tmem,
                            ll.uint32(C_tmem),
                            ad,
                            bd,
                            accum,
                            SFA_tmem,
                            sfa_base + sf_offset,
                            SFB_tmem,
                            sfb_base + sf_offset,
                            out_shape=(PAIR_TILE_M, PAIR_TILE_N),
                        )
                kb_mma = kb_mma + 1
            work_iter = work_iter + 1
            work_mma = work_mma + num_clusters

    with epi:
        epi_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = epi_tid // 32
        work_epi: ll.uint32 = cluster_id
        work_iter: ll.int32 = 0
        while work_epi < num_work:
            stage_acc.bind(work_iter)
            coord_m_epi: ll.uint32 = ll.uint32(0)
            coord_n_epi: ll.uint32 = ll.uint32(0)
            if use_worklist == 1:
                coord_m_epi = ll.uint32(work_m_blocks[work_epi])
                coord_n_epi = ll.uint32(work_n_blocks[work_epi])
            else:
                coord_m_epi, coord_n_epi = get_tile_coord_along_n(work_epi, num_n_blocks)
            for store in ll.unroll(range(NUM_STORES)):
                stage_store.bind((work_iter, store))
                c_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BANK_GROUP)):
                    tmem_col: ll.uint32 = ll.uint32(C_tmem + store * ST_TILE_N + i * ELEMS_PER_BANK_GROUP)
                    smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(c_base, ll.uint32(epi_tid), ll.uint32(i))
                    ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)
                if epi_warp == 0:
                    store_issuer: ll.uint32 = ll.elect_one()
                    if store_issuer == 1:
                        offset_m: ll.int32 = ll.int32(coord_m_epi * PAIR_TILE_M + cta * CTA_TILE_M)
                        offset_n: ll.int32 = ll.int32(coord_n_epi * PAIR_TILE_N + store * ST_TILE_N)
                        ri.tma_st_2d(C_smem, dD, (offset_n, offset_m), shape=(ST_TILE_M, ST_TILE_N))
            work_iter = work_iter + 1
            work_epi = work_epi + num_clusters
