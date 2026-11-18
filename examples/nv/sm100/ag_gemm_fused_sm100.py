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
"""Push-based fused all-gather GEMM with tile-ready publication on SM100."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.distributed._symmetric_memory as symm_mem

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_grouped_m
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
NUM_A_STAGES = 4
NUM_B_STAGES = 3
NUM_COMM_STAGES = 2
NUM_EPI_STAGES = 1
MMA_WAVE_M = 128
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
PAIR_MMA_WAVE_M = CLUSTER_SIZE * MMA_WAVE_M
PAIR_GROUP_M_TILES = 8

ELEMENT_BYTES = 2
COMM_TILE_M = 64
COMM_TILE_K = 64
CHUNKS_PER_CTA_READY = CTA_TILE_M // COMM_TILE_M
SMEM_COMM_BYTES = COMM_TILE_M * COMM_TILE_K * ELEMENT_BYTES
SMEM_A_BYTES = CTA_TILE_M * K_TILE * ELEMENT_BYTES
SMEM_B_BYTES = CTA_TILE_N_SHARD * K_TILE * ELEMENT_BYTES

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // ELEMENT_BYTES
NUM_N_STORES = PAIR_TILE_N // ST_TILE_N
NUM_TMA_STORE_STAGES = 2

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // ELEMENT_BYTES

SHARED_MEM_BYTES = (NUM_TMA_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_A_STAGES * SMEM_A_BYTES +
                    NUM_B_STAGES * SMEM_B_BYTES + NUM_COMM_STAGES * SMEM_COMM_BYTES + 1024)


@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    frontend=dataflow_frontend,
    cluster_shape=(CLUSTER_SIZE, 1, 1),
    cta_group_shape=CTA_GROUP_SHAPE,
)
def ag_gemm_fuse_kernel(
    dPeers: ll.ptr[ll.TmaDescriptor],
    dAComm: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready: ll.ptr[ll.int32],
    ready_ptrs: ll.ptr[ll.ptr[ll.int32]],
    counters: ll.ptr[ll.int32],
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
        load_a = g.role("load_a", name="load_a", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        load_b = g.role("load_b", name="load_b", hw=HW.TMA, warps=[2])
        comm = g.role("comm", name="comm", hw=HW.TMA, warps=[3])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_TMA_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_A_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_B_STAGES, major="K")
        comm_smem = g.smem("comm_smem", (COMM_TILE_M, COMM_TILE_K), dtype=ll.bfloat16, stages=NUM_COMM_STAGES)
        C_tmem = g.tmem("C_tmem", (CTA_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_EPI_STAGES)

        stage_comm = g.depends(src=comm, dst=comm, via=[comm_smem])
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
    warp_idx: ll.int32 = tid // 32
    cta_rank: ll.uint32 = ri.rank_in_group()
    cta_id: ll.uint32 = ll.blockIdx_x()
    cluster_id: ll.uint32 = cta_id // CLUSTER_SIZE
    num_clusters: ll.uint32 = num_ctas // CLUSTER_SIZE

    num_pair_m_tiles: ll.uint32 = M // PAIR_TILE_M
    num_pair_n_tiles: ll.uint32 = N // PAIR_TILE_N
    num_pair_tiles: ll.uint32 = num_pair_m_tiles * num_pair_n_tiles
    num_k_tiles: ll.uint32 = (K + K_TILE - 1) // K_TILE
    cta_m_tiles_per_rank: ll.uint32 = m_per_rank // CTA_TILE_M
    pair_m_tiles_per_rank: ll.uint32 = m_per_rank // PAIR_TILE_M

    comm_k_tiles: ll.uint32 = K // COMM_TILE_K
    comm_m_chunks_per_rank: ll.uint32 = m_per_rank // COMM_TILE_M

    if warp_idx == 0 and ll.elect_one():
        ll.prefetch_tma_descriptor(dA)
        ll.prefetch_tma_descriptor(dD)
        ll.prefetch_tma_descriptor(dB)
        if cta_id < comm_ctas:
            peer_rank: ll.uint32 = ll.uint32(0)
            while peer_rank < world_size:
                if rank != peer_rank:
                    ll.prefetch_tma_descriptor(dPeers[peer_rank])
                peer_rank = peer_rank + 1
        ll.prefetch_tma_descriptor(dAComm)

    with comm:
        chunk_job_comm: ll.uint32 = cta_id
        # early exit
        if cta_id >= comm_ctas:
            chunk_job_comm = comm_m_chunks_per_rank
        while chunk_job_comm < comm_m_chunks_per_rank:
            # 1. push the comm tile to the peers
            row_comm: ll.int32 = ll.int32(rank * m_per_rank + chunk_job_comm * COMM_TILE_M)
            comm_kb: ll.uint32 = ll.uint32(0)
            while comm_kb < comm_k_tiles:
                # bind drives the two-stage TMA load/store ring.
                stage_comm.bind((chunk_job_comm, comm_kb))
                comm_issuer: ll.uint32 = ll.elect_one()
                k_comm: ll.int32 = ll.int32(comm_kb * COMM_TILE_K)
                if comm_issuer == 1:
                    ri.tma_ld_2d(comm_smem, dAComm, (k_comm, row_comm), shape=(COMM_TILE_M, COMM_TILE_K))
                    for peer_step in range(1, world_size):
                        peer_rank: ll.uint32 = (rank + ll.uint32(peer_step)) % world_size
                        ri.tma_st_2d(comm_smem, dPeers[peer_rank], (k_comm, row_comm), shape=(COMM_TILE_M, COMM_TILE_K))
                comm_kb = comm_kb + 1

            # 2. complete the push; then notify peers
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            publish_issuer: ll.uint32 = ll.elect_one()
            if publish_issuer == 1:
                publish_cta_mt: ll.uint32 = chunk_job_comm // CHUNKS_PER_CTA_READY
                old: ll.int32 = ll.atomic_add_acq_rel_gpu(counters + publish_cta_mt, 1)
                if old + 1 == CHUNKS_PER_CTA_READY:
                    # The last chunk resets the per-tile counter for reuse.
                    ll.st_release_gpu(counters + publish_cta_mt, 0)
                    signal_cta_mt: ll.uint32 = rank * cta_m_tiles_per_rank + publish_cta_mt
                    for signal_step in range(1, world_size):
                        signal_rank: ll.uint32 = (rank + ll.uint32(signal_step)) % world_size
                        ll.st_release_system(ready_ptrs[signal_rank] + signal_cta_mt, epoch)
            chunk_job_comm = chunk_job_comm + comm_ctas

    with load_a:
        load_issuer: ll.uint32 = ll.elect_one()
        tile_load: ll.uint32 = cluster_id
        while tile_load < num_pair_tiles:
            coord_pair_m_load, _ = get_tile_coord_grouped_m(
                tile_load,
                num_pair_m_tiles,
                num_pair_n_tiles,
                PAIR_GROUP_M_TILES,
            )
            coord_pair_m_load = (coord_pair_m_load + rank * pair_m_tiles_per_rank) % num_pair_m_tiles
            coord_cta_m_a: ll.uint32 = coord_pair_m_load * CLUSTER_SIZE + cta_rank
            offset_m_a: ll.int32 = ll.int32(coord_pair_m_load * PAIR_TILE_M)

            if load_issuer == 1:
                owner_rank: ll.uint32 = coord_cta_m_a // cta_m_tiles_per_rank
                if owner_rank != rank:
                    # Each participant waits for its own 256-row A tile.
                    wait_cta_mt: ll.int32 = ll.int32(coord_cta_m_a)
                    while ll.ld_acquire_system(ready + wait_cta_mt) != epoch:
                        pass
            ll.__syncwarp()

            kb_load: ll.uint32 = ll.uint32(0)
            while kb_load < num_k_tiles:
                stage_a.bind(kb_load)
                offset_k_a: ll.int32 = ll.int32(kb_load * K_TILE)
                if load_issuer == 1:
                    ri.tma_ld_2d(A_smem, dA, (offset_k_a, offset_m_a), A_plan, shape=(CTA_TILE_M, K_TILE))
                kb_load = kb_load + 1
            tile_load = tile_load + num_clusters

    with load_b:
        b_issuer: ll.uint32 = ll.elect_one()
        tile_b: ll.uint32 = cluster_id
        while tile_b < num_pair_tiles:
            _, coord_pair_n_b = get_tile_coord_grouped_m(
                tile_b,
                num_pair_m_tiles,
                num_pair_n_tiles,
                PAIR_GROUP_M_TILES,
            )
            offset_n_b: ll.int32 = ll.int32(coord_pair_n_b * PAIR_TILE_N)

            kb_b: ll.uint32 = ll.uint32(0)
            while kb_b < num_k_tiles:
                stage_b.bind(kb_b)
                offset_k_b: ll.int32 = ll.int32(kb_b * K_TILE)
                if b_issuer == 1:
                    ri.tma_ld_2d(B_smem, dB, (offset_k_b, offset_n_b), B_plan, shape=(CTA_TILE_N_SHARD, K_TILE))
                kb_b = kb_b + 1
            tile_b = tile_b + num_clusters

    with mma:
        tile_mma: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0
        mma_issuer: ll.uint32 = ll.elect_one()
        while tile_mma < num_pair_tiles:
            stage_acc.bind(tile_iter)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_tiles:
                stage_a.bind(kb_mma)
                stage_b.bind(kb_mma)
                bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                for m in ll.unroll(range(NUM_MMA_M_WAVES)):
                    ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (m * MMA_WAVE_M, 0))
                    tmem_off: ll.uint32 = ll.uint32(C_tmem + m * PAIR_TILE_N)
                    if mma_issuer == 1:
                        ri.umma_f16_ss_2sm_kblock(
                            C_tmem,
                            ad,
                            bd,
                            K_TILE,
                            out_shape=(PAIR_MMA_WAVE_M, PAIR_TILE_N),
                            zero_init=kb_mma == 0,
                            tmem_d=tmem_off,
                        )
                kb_mma = kb_mma + 1

            tile_iter = tile_iter + 1
            tile_mma = tile_mma + num_clusters

    with epi:
        local_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = local_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0

        while tile_epi < num_pair_tiles:
            stage_acc.bind(tile_iter)
            coord_pair_m_epi, coord_pair_n_epi = get_tile_coord_grouped_m(
                tile_epi,
                num_pair_m_tiles,
                num_pair_n_tiles,
                PAIR_GROUP_M_TILES,
            )
            coord_pair_m_epi = (coord_pair_m_epi + rank * pair_m_tiles_per_rank) % num_pair_m_tiles
            coord_cta_m_epi: ll.uint32 = coord_pair_m_epi * CLUSTER_SIZE + cta_rank

            for m in ll.unroll(range(NUM_MMA_M_WAVES)):
                for n in ll.unroll(range(NUM_N_STORES)):
                    stage_store.bind((tile_iter, m, n))
                    cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                    for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                        tmem_col: ll.uint32 = ll.uint32(C_tmem + m * PAIR_TILE_N + n * ST_TILE_N + i * ELEMS_PER_BG)
                        smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(local_tid), ll.uint32(i))
                        ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                    if epi_warp == 0:
                        output_issuer: ll.uint32 = ll.elect_one()
                        if output_issuer == 1:
                            offset_n_epi: ll.int32 = ll.int32(coord_pair_n_epi * PAIR_TILE_N + n * ST_TILE_N)
                            offset_m_epi: ll.int32 = ll.int32(coord_cta_m_epi * CTA_TILE_M + m * MMA_WAVE_M)
                            ri.tma_st_2d(C_smem, dD, (offset_n_epi, offset_m_epi), shape=(ST_TILE_M, ST_TILE_N))

            tile_iter = tile_iter + 1
            tile_epi = tile_epi + num_clusters


class AGGemmFuse:
    """Launcher for fused peer push, chunk publication, and GEMM."""

    MAX_RANKS = 64
    INT32_MAX = (1 << 31) - 1
    TMA_DESCRIPTOR_BYTES = 128
    DEFAULT_COMM_CTAS = 32
    NUM_THREADS = 256
    CLUSTER_SIZE = CLUSTER_SIZE
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    CTA_TILE_M = CTA_TILE_M
    PAIR_TILE_M = PAIR_TILE_M
    PAIR_TILE_N = PAIR_TILE_N
    CTA_TILE_N_SHARD = CTA_TILE_N_SHARD
    K_TILE = K_TILE
    MMA_WAVE_M = MMA_WAVE_M
    COMM_TILE_M = COMM_TILE_M
    COMM_TILE_K = COMM_TILE_K
    ST_TILE_M = ST_TILE_M
    ST_TILE_N = ST_TILE_N
    kernel = ag_gemm_fuse_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from pathlib import Path

        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source),
            "ag_gemm_fuse_kernel",
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
        workspace_handle: symm_mem._SymmetricMemory,
        workspace: torch.Tensor,
        dB,
        out_shape,
        m,
        n,
        k,
        m_per_rank,
        rank,
        world_size,
        comm_ctas=None,
    ):

        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        if not 1 <= world_size <= cls.MAX_RANKS:
            raise ValueError(f"AGGemmFuse supports 1..{cls.MAX_RANKS} ranks")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if min(m, n, k, m_per_rank) <= 0:
            raise ValueError("M, N, K, and m_per_rank must be positive")
        if m != m_per_rank * world_size:
            raise ValueError("M must equal m_per_rank * world_size")
        if m % cls.PAIR_TILE_M or n % cls.PAIR_TILE_N:
            raise ValueError(f"M and N must be divisible by pair tile {cls.PAIR_TILE_M}x{cls.PAIR_TILE_N}")
        if m_per_rank % cls.PAIR_TILE_M:
            raise ValueError(f"m_per_rank must be divisible by pair tile M={cls.PAIR_TILE_M}")
        if m_per_rank % cls.COMM_TILE_M or k % cls.COMM_TILE_K:
            raise ValueError("m_per_rank and K must be divisible by the communication tile")

        peers = [
            workspace_handle.get_buffer(peer_rank, workspace.shape, workspace.dtype) for peer_rank in range(world_size)
        ]

        def descriptor(tensor, inner, outer, *, oob_fill):
            return create_tma_2d_descriptor(
                tensor,
                gmem_inner_dim=k,
                gmem_outer_dim=m,
                smem_inner_dim=inner,
                smem_outer_dim=outer,
                gmem_outer_stride=k,
                swizzle_mode=128,
                oob_fill=oob_fill,
                l2_promotion=3,
            )

        d_peer = [descriptor(peer, cls.COMM_TILE_K, cls.COMM_TILE_M, oob_fill=False) for peer in peers]
        # Device allocations are aligned; a 128-byte stride preserves alignment.
        descriptor_bytes = bytearray().join(bytes(desc) for desc in d_peer)
        peer_descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=workspace.device)
        if peer_descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")
        d_a_comm = descriptor(workspace, cls.COMM_TILE_K, cls.COMM_TILE_M, oob_fill=False)
        d_a = descriptor(workspace, cls.K_TILE, cls.CTA_TILE_M, oob_fill=True)

        out = torch.empty(out_shape, dtype=workspace.dtype, device=workspace.device)
        d_d = create_tma_2d_descriptor(
            out,
            gmem_inner_dim=n,
            gmem_outer_dim=m,
            smem_inner_dim=cls.ST_TILE_N,
            smem_outer_dim=cls.ST_TILE_M,
            gmem_outer_stride=n,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )

        tiles = (m // cls.PAIR_TILE_M) * (n // cls.PAIR_TILE_N)
        sms = torch.cuda.get_device_properties(workspace.device).multi_processor_count
        num_ctas = (min(sms, tiles * cls.CLUSTER_SIZE) // cls.CLUSTER_SIZE) * cls.CLUSTER_SIZE
        if num_ctas == 0:
            raise ValueError("problem is too small for a two-CTA cluster")
        requested_comm_ctas = cls.DEFAULT_COMM_CTAS if comm_ctas is None else comm_ctas
        comm_ctas = max(cls.CLUSTER_SIZE, min(requested_comm_ctas, num_ctas) // cls.CLUSTER_SIZE * cls.CLUSTER_SIZE)
        ready = symm_mem.empty((m // cls.CTA_TILE_M, ), dtype=torch.int32, device=workspace.device)
        ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
        ready_ptrs = torch.tensor(ready_handle.buffer_ptrs, dtype=torch.int64, device=workspace.device)
        ready.zero_()
        counters = torch.zeros((m_per_rank // cls.CTA_TILE_M, ), dtype=torch.int32, device=workspace.device)

        return cls(
            compiled_kernel,
            peer_descriptors,
            d_a_comm,
            d_a,
            dB,
            d_d,
            ready,
            ready_ptrs,
            ready_handle,
            counters,
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
        d_a_comm,
        d_a,
        d_b,
        d_d,
        ready,
        ready_ptrs,
        ready_handle,
        counters,
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
        if peer_count != world_size or not 1 <= peer_count <= self.MAX_RANKS:
            raise ValueError(f"expected 1..{self.MAX_RANKS} peer descriptors, got {peer_count}")
        self.compiled_kernel = compiled_kernel
        self.peer_descriptors = peer_descriptors
        self.d_a_comm = d_a_comm
        self.d_a = d_a
        self.d_b = d_b
        self.d_d = d_d
        self.ready = ready
        self.ready_ptrs = ready_ptrs
        self.ready_handle = ready_handle
        self.counters = counters
        self.out = out
        self.peer_buffers = tuple(peer_buffers)
        self.args = (m, n, k, m_per_rank, rank, world_size)
        self.num_ctas = num_ctas
        self.comm_ctas = comm_ctas
        self.epoch = 0

    def launch(self, stream=None):
        # Keep zero as the initial sentinel and cycle through positive int32 IDs.
        self.epoch = self.epoch % self.INT32_MAX + 1
        self.compiled_kernel(
            self.peer_descriptors,
            self.d_a_comm,
            self.d_a,
            self.d_b,
            self.d_d,
            self.ready,
            self.ready_ptrs,
            self.counters,
            *self.args,
            self.epoch,
            self.num_ctas,
            self.comm_ctas,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def release(self):
        self.peer_descriptors = None
        self.ready_handle = None
        self.ready = None
        self.ready_ptrs = None
        self.counters = None
