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
"""SM100 UMMA QKV projection producer with tile-ready publication for Ulysses A2A."""

import os

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_grouped_m
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

# PAIR_TILE_M/PAIR_TILE_N are one cooperative 2-CTA group's logical
# output tile. Each participant owns CTA_TILE_M A rows and
# CTA_TILE_N_SHARD B columns.
PAIR_TILE_M = 256
PAIR_TILE_N = 256
K_TILE = 64
CLUSTER_SIZE = 2
CTA_GROUP_SHAPE = (CLUSTER_SIZE, 1, 1)
CTA_TILE_M = PAIR_TILE_M // CLUSTER_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CLUSTER_SIZE
MMA_WAVE_M = CTA_TILE_M
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
NUM_AB_STAGES = int(os.environ.get("LK_GEMM_A2A_AB_STAGES", "6"))
NUM_ACC_STAGES = int(os.environ.get("LK_GEMM_A2A_ACC_STAGES", "2"))
NUM_STORE_STAGES = int(os.environ.get("LK_GEMM_A2A_STORE_STAGES", "2"))
NUM_THREADS = 256

SMEM_A = CTA_TILE_M * K_TILE * 2
SMEM_B = CTA_TILE_N_SHARD * K_TILE * 2

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // 2
NUM_STORES = PAIR_TILE_N // ST_TILE_N

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2

SHARED_MEM_BYTES = (NUM_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_AB_STAGES * (SMEM_A + SMEM_B) + 1024)


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend, cluster_shape=(CLUSTER_SIZE, 1, 1),
              cta_group_shape=CTA_GROUP_SHAPE)
def gemm_a2a_sm100_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready: ll.ptr[ll.int32],
    counters: ll.ptr[ll.int32],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    q_features: ll.uint32,
    kv_features: ll.uint32,
    world_size: ll.uint32,
    group_m_tiles: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        tma = g.role("tma", name="load_ab", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_AB_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_AB_STAGES, major="K")
        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_ACC_STAGES)

        stage_ab = g.depends(src=tma, dst=mma, via=[A_smem, B_smem])
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
    # Strip-interleaved N order: emit each SP rank's Q/K/V tiles in lockstep so
    # remote A2A consumers see early ready flags instead of a late contiguous band.
    q_local_bn: ll.uint32 = q_features // PAIR_TILE_N
    kv_local_bn: ll.uint32 = kv_features // PAIR_TILE_N
    q_n_blocks: ll.uint32 = world_size * q_local_bn
    kv_n_blocks: ll.uint32 = world_size * kv_local_bn

    with tma:
        load_issuer: ll.uint32 = ll.elect_one()
        tile_load: ll.uint32 = cluster_id
        while tile_load < num_tiles:
            coord_m_load, coord_n_sched = get_tile_coord_grouped_m(tile_load, num_m_blocks, num_n_blocks, group_m_tiles)
            coord_n_load: ll.uint32 = coord_n_sched
            if coord_n_sched < q_n_blocks:
                coord_n_load = (coord_n_sched % world_size) * q_local_bn + coord_n_sched // world_size
            elif coord_n_sched < q_n_blocks + kv_n_blocks:
                j_k: ll.uint32 = coord_n_sched - q_n_blocks
                coord_n_load = q_n_blocks + (j_k % world_size) * kv_local_bn + j_k // world_size
            else:
                j_v: ll.uint32 = coord_n_sched - q_n_blocks - kv_n_blocks
                coord_n_load = q_n_blocks + kv_n_blocks + (j_v % world_size) * kv_local_bn + j_v // world_size
            offset_m_load: ll.int32 = ll.int32(coord_m_load * PAIR_TILE_M)
            offset_n_load: ll.int32 = ll.int32(coord_n_load * PAIR_TILE_N)

            kb_load: ll.uint32 = ll.uint32(0)
            while kb_load < num_k_blocks:
                stage_ab.bind(kb_load)
                offset_k_load: ll.int32 = ll.int32(kb_load * K_TILE)
                if load_issuer == 1:
                    ri.tma_ld_2d(A_smem, dA, (offset_k_load, offset_m_load), A_plan, shape=(CTA_TILE_M, K_TILE))
                    ri.tma_ld_2d(B_smem, dB, (offset_k_load, offset_n_load), B_plan, shape=(CTA_TILE_N_SHARD, K_TILE))
                kb_load = kb_load + 1
            tile_load = tile_load + num_clusters

    with mma:
        tile_mma: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0
        while tile_mma < num_tiles:
            stage_acc.bind(tile_iter)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_blocks:
                stage_ab.bind(kb_mma)
                mma_issuer: ll.uint32 = ll.elect_one()
                if mma_issuer == 1:
                    ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, 0))
                    bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                    ri.umma_f16_ss_2sm_kblock(
                        C_tmem,
                        ad,
                        bd,
                        K_TILE,
                        out_shape=(PAIR_TILE_M, PAIR_TILE_N),
                        zero_init=kb_mma == 0,
                    )
                kb_mma = kb_mma + 1

            tile_iter = tile_iter + 1
            tile_mma = tile_mma + num_clusters

    with epi:
        epi_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = epi_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0

        while tile_epi < num_tiles:
            stage_acc.bind(tile_iter)
            coord_m_epi, coord_n_sched = get_tile_coord_grouped_m(tile_epi, num_m_blocks, num_n_blocks, group_m_tiles)
            coord_n_epi: ll.uint32 = coord_n_sched
            if coord_n_sched < q_n_blocks:
                coord_n_epi = (coord_n_sched % world_size) * q_local_bn + coord_n_sched // world_size
            elif coord_n_sched < q_n_blocks + kv_n_blocks:
                j_k: ll.uint32 = coord_n_sched - q_n_blocks
                coord_n_epi = q_n_blocks + (j_k % world_size) * kv_local_bn + j_k // world_size
            else:
                j_v: ll.uint32 = coord_n_sched - q_n_blocks - kv_n_blocks
                coord_n_epi = q_n_blocks + kv_n_blocks + (j_v % world_size) * kv_local_bn + j_v // world_size

            for store in ll.unroll(range(NUM_STORES)):
                stage_store.bind((tile_iter, store))
                cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                    tmem_col: ll.uint32 = ll.uint32(C_tmem + store * ST_TILE_N + i * ELEMS_PER_BG)
                    smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(epi_tid), ll.uint32(i))
                    ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                if epi_warp == 0:
                    store_issuer: ll.uint32 = ll.elect_one()
                    if store_issuer == 1:
                        offset_n_epi: ll.int32 = ll.int32(coord_n_epi * PAIR_TILE_N + store * ST_TILE_N)
                        offset_m_epi: ll.int32 = ll.int32(coord_m_epi * PAIR_TILE_M + cta * CTA_TILE_M)
                        ri.tma_st_2d(C_smem, dD, (offset_n_epi, offset_m_epi), shape=(ST_TILE_M, ST_TILE_N))

            if epi_warp == 0:
                ll.tma_store_wait_group(0)
                ll.fence_proxy_async()
                ll.__syncwarp()
                publish_issuer: ll.uint32 = ll.elect_one()
                if publish_issuer == 1:
                    ready_idx: ll.uint32 = coord_m_epi * num_n_blocks + coord_n_epi
                    old: ll.int32 = ll.atomic_add_acq_rel_gpu(counters + ready_idx, 1)
                    if old + 1 == CLUSTER_SIZE:
                        ll.st_release_gpu(counters + ready_idx, 0)
                        ll.st_release_system(ready + ready_idx, epoch)

            tile_iter = tile_iter + 1
            tile_epi = tile_epi + num_clusters
