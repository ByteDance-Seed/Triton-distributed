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
"""Single-kernel SM100 GEMM plus TMA-pull reduce-scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_along_n, get_tile_coord_grouped_m
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

# PAIR_TILE_M/PAIR_TILE_N are one cooperative 2-CTA group's logical
# output tile. Each participant owns CTA_TILE_M A rows and
# CTA_TILE_N_SHARD B columns.
PAIR_TILE_M = 512
PAIR_TILE_N = 256
K_TILE = 64
CLUSTER_SIZE = 2
CTA_GROUP_SHAPE = (CLUSTER_SIZE, 1, 1)
CTA_TILE_M = PAIR_TILE_M // CLUSTER_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CLUSTER_SIZE
NUM_A_STAGES = 2
NUM_B_STAGES = 2
NUM_EPI_STAGES = 1
MMA_WAVE_M = 128
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
PAIR_GROUP_M = 8

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // 2
NUM_STORES = PAIR_TILE_N // ST_TILE_N
NUM_TMA_STORE_STAGES = 2

COMM_TILE_M = 64
COMM_TILE_N = 256
NUM_RS_LOAD_STAGES = 2
NUM_RS_STORE_STAGES = 1
NUM_RS_REDUCE_WARPS = 8
NUM_RS_REDUCE_THREADS = NUM_RS_REDUCE_WARPS * 32
RS_ELEMS_PER_THREAD = COMM_TILE_M * COMM_TILE_N // NUM_RS_REDUCE_THREADS

NUM_THREADS = 17 * 32
ELE_BYTES = 2
SMEM_A = CTA_TILE_M * K_TILE * ELE_BYTES
SMEM_B = CTA_TILE_N_SHARD * K_TILE * ELE_BYTES
SMEM_RS_TILE = COMM_TILE_M * COMM_TILE_N * ELE_BYTES

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // ELE_BYTES

SHARED_MEM_BYTES = (NUM_TMA_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_A_STAGES * SMEM_A +
                    NUM_B_STAGES * SMEM_B + (NUM_RS_LOAD_STAGES + NUM_RS_STORE_STAGES) * SMEM_RS_TILE + 1024)


@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    frontend=dataflow_frontend,
    cluster_shape=(CLUSTER_SIZE, 1, 1),
    cta_group_shape=CTA_GROUP_SHAPE,
)
def gemm_rs_fuse_v2_kernel(
    dPeers: ll.ptr[ll.TmaDescriptor],
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dPartial: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dOut: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready_local: ll.ptr[ll.int32],
    ready_peers: ll.ptr[ll.ptr[ll.int32]],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    m_per_rank: ll.uint32,
    rank: ll.uint32,
    world_size: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
    comm_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load_ab = g.role("load_ab", name="load_ab", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        rs_load = g.role("rs_load", name="rs_load", hw=HW.TMA, warps=[3])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])
        rs_reduce = g.role("rs_reduce", name="rs_reduce", hw=HW.CUDACore, warps=[8, 9, 10, 11, 12, 13, 14, 15])
        rs_store = g.role("rs_store", name="rs_store", hw=HW.TMA, warps=[16])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_TMA_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_A_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_B_STAGES, major="K")
        rs_peer_smem = g.smem("rs_peer_smem", (COMM_TILE_M, COMM_TILE_N), dtype=ll.bfloat16, stages=NUM_RS_LOAD_STAGES)
        rs_out_smem = g.smem("rs_out_smem", (COMM_TILE_M, COMM_TILE_N), dtype=ll.bfloat16, stages=NUM_RS_STORE_STAGES)
        C_tmem = g.tmem("C_tmem", (CTA_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_EPI_STAGES)

        stage_ab = g.depends(src=load_ab, dst=mma, via=[A_smem, B_smem])
        stage_acc = g.depends(src=mma, dst=epi, via=[C_tmem])
        stage_store = g.depends(src=epi, dst=epi, via=[C_smem])
        stage_rs_peer = g.depends(src=rs_load, dst=rs_reduce, via=[rs_peer_smem])
        stage_rs_out = g.depends(src=rs_reduce, dst=rs_store, via=[rs_out_smem])
        A_plan = g.tma_transfer(mcast_group_dim=None, inter_cta_coord_dim=1, inter_cta_stride=CTA_TILE_M)
        B_plan = g.tma_transfer(mcast_group_dim=None, inter_cta_coord_dim=1, inter_cta_stride=CTA_TILE_N_SHARD)

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta: ll.uint32 = ri.rank_in_group()
    cta_id: ll.uint32 = ll.blockIdx_x()
    cluster_id: ll.uint32 = cta_id // CLUSTER_SIZE
    num_clusters: ll.uint32 = num_ctas // CLUSTER_SIZE

    num_pair_m_tiles: ll.uint32 = M // PAIR_TILE_M
    num_n_tiles: ll.uint32 = N // PAIR_TILE_N
    num_pair_tiles: ll.uint32 = num_pair_m_tiles * num_n_tiles
    num_k_tiles: ll.uint32 = (K + K_TILE - 1) // K_TILE
    pair_m_tiles_per_rank: ll.uint32 = m_per_rank // PAIR_TILE_M

    num_rs_m_tiles: ll.uint32 = m_per_rank // COMM_TILE_M
    num_rs_n_tiles: ll.uint32 = N // COMM_TILE_N
    num_rs_jobs: ll.uint32 = num_rs_m_tiles * num_rs_n_tiles

    if warp_idx == 0:
        prefetch_issuer: ll.uint32 = ll.elect_one()
        if prefetch_issuer == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)
            ll.prefetch_tma_descriptor(dPartial)

            if cta_id < comm_ctas:
                for peer_prefetch in range(world_size):
                    ll.prefetch_tma_descriptor(dPeers[peer_prefetch])
                ll.prefetch_tma_descriptor(dOut)

    with load_ab:
        load_issuer: ll.uint32 = ll.elect_one()
        tile_load: ll.uint32 = cluster_id
        while tile_load < num_pair_tiles:
            coord_pair_m_load, coord_n_load = get_tile_coord_grouped_m(
                tile_load,
                num_pair_m_tiles,
                num_n_tiles,
                PAIR_GROUP_M,
            )
            coord_pair_m_load = ((coord_pair_m_load % world_size) * pair_m_tiles_per_rank +
                                 coord_pair_m_load // world_size)
            offset_m_load: ll.int32 = ll.int32(coord_pair_m_load * PAIR_TILE_M)
            offset_n_load: ll.int32 = ll.int32(coord_n_load * PAIR_TILE_N)

            kt: ll.uint32 = ll.uint32(0)
            while kt < num_k_tiles:
                stage_ab.bind(kt)
                offset_k_load: ll.int32 = ll.int32(kt * K_TILE)
                if load_issuer == 1:
                    ri.tma_ld_2d(A_smem, dA, (offset_k_load, offset_m_load), A_plan, shape=(CTA_TILE_M, K_TILE))
                    ri.tma_ld_2d(B_smem, dB, (offset_k_load, offset_n_load), B_plan, shape=(CTA_TILE_N_SHARD, K_TILE))
                kt = kt + 1
            tile_load = tile_load + num_clusters

    with mma:
        mma_issuer: ll.uint32 = ll.elect_one()
        tile_mma: ll.uint32 = cluster_id
        tile_iter_mma: ll.int32 = 0

        while tile_mma < num_pair_tiles:
            stage_acc.bind(tile_iter_mma)
            kt_mma: ll.uint32 = ll.uint32(0)
            while kt_mma < num_k_tiles:
                stage_ab.bind(kt_mma)
                bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                for wave in ll.unroll(range(NUM_MMA_M_WAVES)):
                    ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (wave * MMA_WAVE_M, 0))
                    tmem_off: ll.uint32 = ll.uint32(C_tmem + wave * PAIR_TILE_N)
                    if mma_issuer == 1:
                        ri.umma_f16_ss_2sm_kblock(
                            C_tmem,
                            ad,
                            bd,
                            K_TILE,
                            out_shape=(CTA_TILE_M, PAIR_TILE_N),
                            zero_init=kt_mma == 0,
                            tmem_d=tmem_off,
                        )
                kt_mma = kt_mma + 1

            tile_iter_mma = tile_iter_mma + 1
            tile_mma = tile_mma + num_clusters

    with epi:
        epi_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = epi_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter_epi: ll.int32 = 0

        while tile_epi < num_pair_tiles:
            stage_acc.bind(tile_iter_epi)
            coord_pair_m_epi, coord_n_epi = get_tile_coord_grouped_m(
                tile_epi,
                num_pair_m_tiles,
                num_n_tiles,
                PAIR_GROUP_M,
            )
            coord_pair_m_epi = ((coord_pair_m_epi % world_size) * pair_m_tiles_per_rank +
                                coord_pair_m_epi // world_size)
            coord_cta_m_epi: ll.uint32 = coord_pair_m_epi * CLUSTER_SIZE + cta

            for wave in ll.unroll(range(NUM_MMA_M_WAVES)):
                for store in ll.unroll(range(NUM_STORES)):
                    stage_store.bind((tile_iter_epi, wave, store))
                    cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                    for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                        tmem_col: ll.uint32 = ll.uint32(C_tmem + wave * PAIR_TILE_N + store * ST_TILE_N +
                                                        i * ELEMS_PER_BG)
                        smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(epi_tid), ll.uint32(i))
                        ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                    if epi_warp == 0:
                        partial_issuer: ll.uint32 = ll.elect_one()
                        if partial_issuer == 1:
                            offset_n_partial: ll.int32 = ll.int32(coord_n_epi * PAIR_TILE_N + store * ST_TILE_N)
                            offset_m_partial: ll.int32 = ll.int32(coord_cta_m_epi * CTA_TILE_M + wave * MMA_WAVE_M)
                            ri.tma_st_2d(
                                C_smem,
                                dPartial,
                                (offset_n_partial, offset_m_partial),
                                shape=(ST_TILE_M, ST_TILE_N),
                            )

            if epi_warp == 0:
                ll.tma_store_wait_group(0)
                ll.__syncwarp()
                publish_issuer: ll.uint32 = ll.elect_one()
                if publish_issuer == 1:
                    ready_idx: ll.uint32 = coord_cta_m_epi * num_n_tiles + coord_n_epi
                    ll.st_release_system(ready_local + ready_idx, epoch)

            tile_iter_epi = tile_iter_epi + 1
            tile_epi = tile_epi + num_clusters

    with rs_load:
        rs_load_issuer: ll.uint32 = ll.elect_one()
        rs_load_job: ll.uint32 = cta_id
        if cta_id >= comm_ctas:
            rs_load_job = num_rs_jobs

        while rs_load_job < num_rs_jobs:
            coord_m_rs_load, coord_n_rs_load = get_tile_coord_along_n(rs_load_job, num_rs_n_tiles)
            offset_m_rs_load: ll.uint32 = rank * m_per_rank + coord_m_rs_load * COMM_TILE_M
            offset_n_rs_load: ll.uint32 = coord_n_rs_load * COMM_TILE_N
            rs_ready_idx: ll.uint32 = (offset_m_rs_load // CTA_TILE_M) * num_n_tiles + offset_n_rs_load // PAIR_TILE_N

            for rs_step in range(world_size):
                rs_src: ll.uint32 = (rank + cta_id + rs_step) % world_size
                if rs_load_issuer == 1:
                    while ll.ld_acquire_system(ready_peers[rs_src] + rs_ready_idx) != epoch:
                        pass
                ll.__syncwarp()
                stage_rs_peer.bind((rs_load_job, rs_step))
                if rs_load_issuer == 1:
                    ri.tma_ld_2d(
                        rs_peer_smem,
                        dPeers[rs_src],
                        (ll.int32(offset_n_rs_load), ll.int32(offset_m_rs_load)),
                        shape=(COMM_TILE_M, COMM_TILE_N),
                    )
            rs_load_job = rs_load_job + comm_ctas

    with rs_reduce:
        rs_reduce_tid: ll.int32 = tid - 256
        rs_reduce_job: ll.uint32 = cta_id
        if cta_id >= comm_ctas:
            rs_reduce_job = num_rs_jobs

        while rs_reduce_job < num_rs_jobs:
            accum = ll.empty([64], dtype=ll.float32, scope="local")
            for init_i in ll.unroll(range(RS_ELEMS_PER_THREAD)):
                accum[init_i] = ll.float32(0.0)

            for rs_reduce_src in range(world_size):
                stage_rs_peer.bind((rs_reduce_job, rs_reduce_src))
                for reduce_i in ll.unroll(range(RS_ELEMS_PER_THREAD)):
                    rs_elem: ll.int32 = rs_reduce_tid + reduce_i * NUM_RS_REDUCE_THREADS
                    rs_value: ll.bfloat16 = ri.smem_ld_bf16(rs_peer_smem, rs_elem)
                    accum[reduce_i] = accum[reduce_i] + ll.bf16_to_float(rs_value)

            stage_rs_out.bind(rs_reduce_job)
            for rs_store_i in ll.unroll(range(RS_ELEMS_PER_THREAD)):
                rs_out_elem: ll.int32 = rs_reduce_tid + rs_store_i * NUM_RS_REDUCE_THREADS
                ri.smem_st_bf16(rs_out_smem, rs_out_elem, ll.float_to_bf16(accum[rs_store_i]))
            rs_reduce_job = rs_reduce_job + comm_ctas

    with rs_store:
        rs_store_job: ll.uint32 = cta_id
        if cta_id >= comm_ctas:
            rs_store_job = num_rs_jobs

        while rs_store_job < num_rs_jobs:
            rs_store_issuer: ll.uint32 = ll.elect_one()
            coord_m_rs_store, coord_n_rs_store = get_tile_coord_along_n(rs_store_job, num_rs_n_tiles)
            stage_rs_out.bind(rs_store_job)
            if rs_store_issuer == 1:
                ri.tma_st_2d(rs_out_smem, dOut,
                             (ll.int32(coord_n_rs_store * COMM_TILE_N), ll.int32(coord_m_rs_store * COMM_TILE_M)),
                             shape=(COMM_TILE_M, COMM_TILE_N))
            rs_store_job = rs_store_job + comm_ctas


class GemmRSFuseV2:
    """Launcher for fused local GEMM and owner-side TMA-pull RS."""

    MAX_RANKS = 64
    INT32_MAX = (1 << 31) - 1
    TMA_DESCRIPTOR_BYTES = 128
    NUM_THREADS = NUM_THREADS
    CLUSTER_SIZE = CLUSTER_SIZE
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    CTA_TILE_M = CTA_TILE_M
    PAIR_TILE_M = PAIR_TILE_M
    PAIR_TILE_N = PAIR_TILE_N
    K_TILE = K_TILE
    CTA_TILE_N_SHARD = CTA_TILE_N_SHARD
    MMA_WAVE_M = MMA_WAVE_M
    ST_TILE_M = ST_TILE_M
    ST_TILE_N = ST_TILE_N
    COMM_TILE_M = COMM_TILE_M
    COMM_TILE_N = COMM_TILE_N
    kernel = gemm_rs_fuse_v2_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "gemm_rs_fuse_v2_kernel",
            grid=(1, 1, 1),
            block=(cls.NUM_THREADS, 1, 1),
            shared_mem_bytes=cls.SHARED_MEM_BYTES,
            arch="sm_100a",
            cache_dir=cache_dir,
            verbose=verbose,
            include_paths=[Path(__file__).resolve().parents[2]],
            cluster_shape=cls.kernel.cluster_shape,
            cta_group_shape=cls.kernel.cta_group_shape,
            allow_non_portable_cluster=True,
        )

    @classmethod
    def create(
        cls,
        compiled_kernel,
        *,
        workspace_handle,
        partial: torch.Tensor,
        ready_handle,
        ready_local: torch.Tensor,
        dA,
        dB,
        m: int,
        n: int,
        k: int,
        rank: int,
        world_size: int,
        comm_ctas: int | None = None,
    ):
        import torch
        if not 1 <= world_size <= cls.MAX_RANKS:
            raise ValueError(f"GemmRSFuseV2 supports 1..{cls.MAX_RANKS} ranks")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if min(m, n, k) <= 0:
            raise ValueError("M, N, and K must be positive")
        if m % world_size:
            raise ValueError("M must be divisible by world_size")
        if m % cls.PAIR_TILE_M or n % cls.PAIR_TILE_N:
            raise ValueError(f"M and N must be divisible by fused pair tile {cls.PAIR_TILE_M}x{cls.PAIR_TILE_N}")

        m_per_rank = m // world_size
        if m_per_rank % cls.PAIR_TILE_M or k % cls.K_TILE:
            raise ValueError(f"M/world_size and K must be divisible by fused tiles {cls.PAIR_TILE_M} and {cls.K_TILE}")
        if m_per_rank % cls.COMM_TILE_M or n % cls.COMM_TILE_N:
            raise ValueError(f"M/world_size and N must be divisible by RS tile {cls.COMM_TILE_M}x{cls.COMM_TILE_N}")
        if partial.ndim != 2 or tuple(partial.shape) != (m, n) or partial.dtype != torch.bfloat16:
            raise ValueError(f"partial must be a two-dimensional BF16 tensor of shape ({m}, {n})")
        if not partial.is_cuda or not partial.is_contiguous():
            raise ValueError("partial must be a contiguous CUDA tensor")

        expected_ready = (m // cls.CTA_TILE_M) * (n // cls.PAIR_TILE_N)
        if ready_local.dtype != torch.int32 or ready_local.numel() != expected_ready:
            raise ValueError(f"ready_local must be int32[{expected_ready}]")
        if ready_local.device != partial.device or not ready_local.is_contiguous():
            raise ValueError("ready_local must be contiguous and on partial's device")
        if len(ready_handle.buffer_ptrs) < world_size:
            raise ValueError("ready_handle does not contain all peer pointers")

        peers = [
            workspace_handle.get_buffer(peer_rank, partial.shape, partial.dtype) for peer_rank in range(world_size)
        ]

        def rs_descriptor(tensor):
            return create_tma_2d_descriptor(
                tensor,
                gmem_inner_dim=n,
                gmem_outer_dim=m,
                smem_inner_dim=cls.COMM_TILE_N,
                smem_outer_dim=cls.COMM_TILE_M,
                gmem_outer_stride=n,
                swizzle_mode=0,
                oob_fill=False,
                l2_promotion=3,
            )

        peer_descs = [rs_descriptor(peer) for peer in peers]
        descriptor_bytes = bytearray().join(bytes(desc) for desc in peer_descs)
        peer_descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=partial.device)
        if peer_descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")

        d_partial = create_tma_2d_descriptor(
            partial,
            gmem_inner_dim=n,
            gmem_outer_dim=m,
            smem_inner_dim=cls.ST_TILE_N,
            smem_outer_dim=cls.ST_TILE_M,
            gmem_outer_stride=n,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )
        out = torch.empty((m_per_rank, n), dtype=partial.dtype, device=partial.device)
        d_out = create_tma_2d_descriptor(
            out,
            gmem_inner_dim=n,
            gmem_outer_dim=m_per_rank,
            smem_inner_dim=cls.COMM_TILE_N,
            smem_outer_dim=cls.COMM_TILE_M,
            gmem_outer_stride=n,
            swizzle_mode=0,
            oob_fill=False,
            l2_promotion=3,
        )
        ready_peers = torch.tensor(ready_handle.buffer_ptrs[:world_size], dtype=torch.int64, device=partial.device)

        pair_tiles = (m // cls.PAIR_TILE_M) * (n // cls.PAIR_TILE_N)
        if pair_tiles < 2:
            raise ValueError("fused GEMM-RS requires at least two pair output tiles")
        sms = torch.cuda.get_device_properties(partial.device).multi_processor_count
        num_ctas = (min(sms, pair_tiles * cls.CLUSTER_SIZE) // cls.CLUSTER_SIZE) * cls.CLUSTER_SIZE
        if num_ctas == 0:
            raise ValueError("problem is too small for a two-CTA cluster")

        rs_jobs = (m_per_rank // cls.COMM_TILE_M) * (n // cls.COMM_TILE_N)
        requested_comm_ctas = num_ctas if comm_ctas is None else comm_ctas
        if requested_comm_ctas <= 0:
            raise ValueError("comm_ctas must be positive")
        comm_ctas = max(1, min(requested_comm_ctas, num_ctas, rs_jobs))
        ready_local.zero_()

        return cls(
            compiled_kernel,
            peer_descriptors,
            dA,
            dB,
            d_partial,
            d_out,
            ready_local,
            ready_peers,
            ready_handle,
            partial,
            out,
            peers,
            m=m,
            n=n,
            k=k,
            m_per_rank=m_per_rank,
            rank=rank,
            world_size=world_size,
            num_ctas=num_ctas,
            comm_ctas=comm_ctas,
        )

    def __init__(
        self,
        compiled_kernel,
        peer_descriptors,
        d_a,
        d_b,
        d_partial,
        d_out,
        ready_local,
        ready_peers,
        ready_handle,
        partial,
        out,
        peer_buffers,
        *,
        m,
        n,
        k,
        m_per_rank,
        rank,
        world_size,
        num_ctas,
        comm_ctas,
    ):
        peer_count = peer_descriptors.numel() // self.TMA_DESCRIPTOR_BYTES
        if peer_count != world_size:
            raise ValueError(f"expected {world_size} peer descriptors, got {peer_count}")

        self.compiled_kernel = compiled_kernel
        self.peer_descriptors = peer_descriptors
        self.d_a = d_a
        self.d_b = d_b
        self.d_partial = d_partial
        self.d_out = d_out
        self.ready_local = ready_local
        self.ready_peers = ready_peers
        self.ready_handle = ready_handle
        self.partial = partial
        self.out = out
        self.peer_buffers = tuple(peer_buffers)
        self.args = (m, n, k, m_per_rank, rank, world_size)
        self.num_ctas = num_ctas
        self.comm_ctas = comm_ctas
        self.epoch = 0

    def launch(self, stream=None):
        self.epoch = self.epoch % self.INT32_MAX + 1
        self.compiled_kernel(
            self.peer_descriptors,
            self.d_a,
            self.d_b,
            self.d_partial,
            self.d_out,
            self.ready_local,
            self.ready_peers,
            *self.args,
            self.epoch,
            self.num_ctas,
            self.comm_ctas,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def release(self):
        self.peer_descriptors = None
        self.d_a = None
        self.d_b = None
        self.d_partial = None
        self.d_out = None
        self.ready_local = None
        self.ready_peers = None
        self.ready_handle = None
        self.partial = None
        self.peer_buffers = ()
