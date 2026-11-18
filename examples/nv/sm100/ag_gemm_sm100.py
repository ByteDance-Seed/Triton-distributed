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
import os

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_grouped_m
from examples.nv.sm100.ag_tma_push_sm100 import CHUNK_ROWS
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

# PAIR_TILE_M/PAIR_TILE_N are one cooperative 2-CTA group's logical
# output tile. Each participant owns CTA_TILE_M A rows and
# CTA_TILE_N_SHARD B columns.
PAIR_TILE_M = 256
PAIR_TILE_N = 256
K_TILE = 64
CLUSTER_SIZE = 2
CLUSTER_SHAPE = (CLUSTER_SIZE, 1, 1)
CTA_GROUP_SHAPE = (CLUSTER_SIZE, 1, 1)
CTA_TILE_M = PAIR_TILE_M // CLUSTER_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CLUSTER_SIZE
MMA_WAVE_M = CTA_TILE_M
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
NUM_A_STAGES = int(os.environ.get("LK_AG_GEMM_A_STAGES", "6"))
NUM_B_STAGES = int(os.environ.get("LK_AG_GEMM_B_STAGES", "6"))
NUM_ACC_STAGES = int(os.environ.get("LK_AG_GEMM_ACC_STAGES", "2"))
NUM_STORE_STAGES = int(os.environ.get("LK_AG_GEMM_STORE_STAGES", "2"))
GROUP_M = int(os.environ.get("LK_AG_GEMM_GROUP_M", "8"))

SMEM_A = CTA_TILE_M * K_TILE * 2
SMEM_B = CTA_TILE_N_SHARD * K_TILE * 2

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // 2
NUM_STORES = PAIR_TILE_N // ST_TILE_N

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend, cluster_shape=CLUSTER_SHAPE,
              cta_group_shape=CTA_GROUP_SHAPE)
def ag_gemm_sm100_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready: ll.ptr[ll.int32],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    m_per_rank: ll.uint32,
    rank: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load_a = g.role("load_a", name="load_a", hw=HW.TMA, warps=[0])
        load_b = g.role("load_b", name="load_b", hw=HW.TMA, warps=[2])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_A_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_B_STAGES, major="K")
        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_ACC_STAGES)

        stage_a = g.depends(src=load_a, dst=mma, via=[A_smem])
        stage_b = g.depends(src=load_b, dst=mma, via=[B_smem])
        stage_acc = g.depends(src=mma, dst=epi, via=[C_tmem])
        stage_store = g.depends(src=epi, dst=epi, via=[C_smem])
        A_plan = g.tma_transfer(
            mcast_group_dim=None,
            inter_cta_coord_dim=1,
            inter_cta_stride=CTA_TILE_M,
        )
        B_plan = g.tma_transfer(
            mcast_group_dim=None,
            inter_cta_coord_dim=1,
            inter_cta_stride=CTA_TILE_N_SHARD,
        )

    tid: ll.int32 = ll.threadIdx_x()
    cta: ll.uint32 = ri.rank_in_group()
    cta_id: ll.uint32 = ll.blockIdx_x()
    cluster_id: ll.uint32 = cta_id // CLUSTER_SIZE
    num_clusters: ll.uint32 = num_ctas // CLUSTER_SIZE

    num_m_blocks: ll.uint32 = (M + PAIR_TILE_M - 1) // PAIR_TILE_M
    num_n_blocks: ll.uint32 = (N + PAIR_TILE_N - 1) // PAIR_TILE_N
    num_tiles: ll.uint32 = num_m_blocks * num_n_blocks
    num_k_blocks: ll.uint32 = (K + K_TILE - 1) // K_TILE
    m_blocks_per_rank: ll.uint32 = m_per_rank // PAIR_TILE_M

    with load_a:
        el_a: ll.uint32 = ll.elect_one()
        tile_a: ll.uint32 = cluster_id
        while tile_a < num_tiles:
            coord_m_a, _ = get_tile_coord_grouped_m(tile_a, num_m_blocks, num_n_blocks, GROUP_M)
            coord_m_a = (coord_m_a + rank * m_blocks_per_rank) % num_m_blocks
            offset_m_a: ll.int32 = ll.int32(coord_m_a * PAIR_TILE_M)

            if el_a == 1:
                offset_m_local: ll.int32 = offset_m_a + ll.int32(cta * CTA_TILE_M)
                first_chunk: ll.int32 = offset_m_local // CHUNK_ROWS
                last_row: ll.int32 = ll.min_val(offset_m_local + CTA_TILE_M, ll.int32(M))
                last_chunk: ll.int32 = (last_row + CHUNK_ROWS - 1) // CHUNK_ROWS
                chunk: ll.int32 = first_chunk
                while chunk < last_chunk:
                    while ll.ld_acquire_system(ready + chunk) < epoch:
                        pass
                    chunk = chunk + 1
            ll.__syncwarp()

            kb_a: ll.uint32 = ll.uint32(0)
            while kb_a < num_k_blocks:
                stage_a.bind(kb_a)
                if el_a == 1:
                    ri.tma_ld_2d(A_smem, dA, (ll.int32(kb_a * K_TILE), offset_m_a), A_plan, shape=(CTA_TILE_M, K_TILE))
                kb_a = kb_a + 1
            tile_a = tile_a + num_clusters

    with load_b:
        el_b: ll.uint32 = ll.elect_one()
        tile_b: ll.uint32 = cluster_id
        while tile_b < num_tiles:
            _, coord_n_b = get_tile_coord_grouped_m(tile_b, num_m_blocks, num_n_blocks, GROUP_M)
            offset_n_b: ll.int32 = ll.int32(coord_n_b * PAIR_TILE_N)

            kb_b: ll.uint32 = ll.uint32(0)
            while kb_b < num_k_blocks:
                stage_b.bind(kb_b)
                if el_b == 1:
                    ri.tma_ld_2d(
                        B_smem,
                        dB,
                        (ll.int32(kb_b * K_TILE), offset_n_b),
                        B_plan,
                        shape=(CTA_TILE_N_SHARD, K_TILE),
                    )
                kb_b = kb_b + 1
            tile_b = tile_b + num_clusters

    with mma:
        tile_mma: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0
        while tile_mma < num_tiles:
            stage_acc.bind(tile_iter)
            kb: ll.uint32 = ll.uint32(0)
            while kb < num_k_blocks:
                stage_a.bind(kb)
                stage_b.bind(kb)
                el_umma: ll.uint32 = ll.elect_one()
                if el_umma == 1:
                    ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, 0))
                    bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                    ri.umma_f16_ss_2sm_kblock(
                        C_tmem,
                        ad,
                        bd,
                        K_TILE,
                        out_shape=(PAIR_TILE_M, PAIR_TILE_N),
                        zero_init=kb == 0,
                    )
                kb = kb + 1

            tile_iter = tile_iter + 1
            tile_mma = tile_mma + num_clusters

    with epi:
        local_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = local_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0

        while tile_epi < num_tiles:
            stage_acc.bind(tile_iter)
            coord_m_epi, coord_n_epi = get_tile_coord_grouped_m(tile_epi, num_m_blocks, num_n_blocks, GROUP_M)
            coord_m_epi = (coord_m_epi + rank * m_blocks_per_rank) % num_m_blocks

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
