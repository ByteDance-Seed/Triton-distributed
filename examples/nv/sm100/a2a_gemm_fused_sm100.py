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
"""Push-based fused Ulysses post-attention all-to-all GEMM on SM100."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

import os

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
CTA_GROUP_SIZE = 2
CLUSTER_SIZE = CTA_GROUP_SIZE
CTA_GROUP_SHAPE = (CTA_GROUP_SIZE, 1, 1)
CTA_TILE_M = PAIR_TILE_M // CTA_GROUP_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CTA_GROUP_SIZE
NUM_A_STAGES = int(os.environ.get("LK_A2A_FUSED_A_STAGES", "3"))
NUM_B_STAGES = int(os.environ.get("LK_A2A_FUSED_B_STAGES", "3"))
NUM_COMM_STAGES = int(os.environ.get("LK_A2A_FUSED_COMM_STAGES", "2"))
NUM_EPI_STAGES = 1
MMA_WAVE_M = 128
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
PAIR_GROUP_M = 8

ELE_BYTES = 2
COMM_TILE_M = int(os.environ.get("LK_A2A_FUSED_COMM_M", "128"))
COMM_TILE_K = 64
SMEM_COMM = COMM_TILE_M * COMM_TILE_K * ELE_BYTES
SMEM_A = CTA_TILE_M * K_TILE * ELE_BYTES
SMEM_B = CTA_TILE_N_SHARD * K_TILE * ELE_BYTES

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // 2
NUM_STORES = PAIR_TILE_N // ST_TILE_N
NUM_TMA_STORE_STAGES = 2

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2

SHARED_MEM_BYTES = (NUM_TMA_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_A_STAGES * SMEM_A +
                    NUM_B_STAGES * SMEM_B + NUM_COMM_STAGES * SMEM_COMM + 1024)


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend, cluster_shape=(CLUSTER_SIZE, 1, 1),
              cta_group_shape=CTA_GROUP_SHAPE)
def a2a_gemm_fuse_sm100_kernel(
    dPeers: ll.ptr[ll.TmaDescriptor],
    dAComm: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready: ll.ptr[ll.int32],
    ready_ptrs: ll.ptr[ll.ptr[ll.int32]],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    band_features: ll.uint32,
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
    cta: ll.uint32 = ri.rank_in_group()
    cta_id: ll.uint32 = ll.blockIdx_x()
    cluster_id: ll.uint32 = cta_id // CLUSTER_SIZE
    num_clusters: ll.uint32 = num_ctas // CLUSTER_SIZE

    num_cta_m_blocks: ll.uint32 = M // CTA_TILE_M
    num_pair_m_blocks: ll.uint32 = M // PAIR_TILE_M
    num_n_blocks: ll.uint32 = N // PAIR_TILE_N
    num_tiles: ll.uint32 = num_pair_m_blocks * num_n_blocks
    num_k_blocks: ll.uint32 = K // K_TILE
    band_blocks: ll.uint32 = band_features // K_TILE

    m_chunks: ll.uint32 = CTA_TILE_M // COMM_TILE_M
    band_k_tiles: ll.uint32 = band_features // COMM_TILE_K
    num_jobs: ll.uint32 = world_size * num_cta_m_blocks

    if warp_idx == 0 and ll.elect_one():
        if cta_id < comm_ctas:
            peer_rank: ll.uint32 = ll.uint32(0)
            while peer_rank < world_size:
                ll.prefetch_tma_descriptor(dPeers[peer_rank])
                peer_rank = peer_rank + 1
            ll.prefetch_tma_descriptor(dAComm)

    with comm:
        job: ll.uint32 = cta_id
        # early exit
        if cta_id >= comm_ctas:
            job = num_jobs
        while job < num_jobs:
            # Interleave destinations for every M block.  A destination-major
            # order publishes every block for rank 0 before rank 1, creating a
            # rank-dependent GEMM start delay in the fused pipeline.
            mb: ll.uint32 = job // world_size
            dest_rank: ll.uint32 = job % world_size
            cm: ll.uint32 = ll.uint32(0)
            while cm < m_chunks:
                kt: ll.uint32 = ll.uint32(0)
                while kt < band_k_tiles:
                    # bind drives the two-stage TMA load/store ring.
                    stage_comm.bind((job, cm, kt))
                    comm_issuer: ll.uint32 = ll.elect_one()
                    k_src: ll.int32 = ll.int32(kt * COMM_TILE_K)
                    row_src: ll.int32 = ll.int32(dest_rank * M + mb * CTA_TILE_M + cm * COMM_TILE_M)
                    k_dst: ll.int32 = ll.int32(rank * band_features + kt * COMM_TILE_K)
                    row_dst: ll.int32 = ll.int32(mb * CTA_TILE_M + cm * COMM_TILE_M)
                    if comm_issuer == 1:
                        ri.tma_ld_2d(comm_smem, dAComm, (k_src, row_src), shape=(COMM_TILE_M, COMM_TILE_K))
                        ri.tma_st_2d(comm_smem, dPeers[dest_rank], (k_dst, row_dst), shape=(COMM_TILE_M, COMM_TILE_K))
                    kt = kt + 1
                cm = cm + 1
            # Complete the push; then notify the destination rank that this
            # rank's K band for the m-block is consumable.
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            publish_issuer: ll.uint32 = ll.elect_one()
            if publish_issuer == 1:
                ll.st_release_system(ready_ptrs[dest_rank] + rank * num_cta_m_blocks + mb, epoch)
            job = job + comm_ctas

    with load_a:
        el_a: ll.uint32 = ll.elect_one()
        tile_a: ll.uint32 = cluster_id
        while tile_a < num_tiles:
            coord_pair_m_a, _ = get_tile_coord_grouped_m(tile_a, num_pair_m_blocks, num_n_blocks, PAIR_GROUP_M)
            coord_cta_m_a: ll.uint32 = coord_pair_m_a * CTA_GROUP_SIZE + cta
            offset_m_a: ll.int32 = ll.int32(coord_pair_m_a * PAIR_TILE_M)

            # The A operand gathers one K band per source rank; consume each
            # band only after the owning rank published it for this M block.
            kb_a: ll.uint32 = ll.uint32(0)
            while kb_a < num_k_blocks:
                if kb_a % band_blocks == 0:
                    band: ll.uint32 = kb_a // band_blocks
                    if el_a == 1:
                        while ll.ld_acquire_system(ready + band * num_cta_m_blocks + coord_cta_m_a) != epoch:
                            pass
                    ll.__syncwarp()
                stage_a.bind(kb_a)
                if el_a == 1:
                    ri.tma_ld_2d(A_smem, dA, (ll.int32(kb_a * K_TILE), offset_m_a), A_plan, shape=(CTA_TILE_M, K_TILE))
                kb_a = kb_a + 1
            tile_a = tile_a + num_clusters

    with load_b:
        el_b: ll.uint32 = ll.elect_one()
        tile_b: ll.uint32 = cluster_id
        while tile_b < num_tiles:
            _, coord_n_b = get_tile_coord_grouped_m(tile_b, num_pair_m_blocks, num_n_blocks, PAIR_GROUP_M)
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
        mma_issuer: ll.uint32 = ll.elect_one()
        while tile_mma < num_tiles:
            stage_acc.bind(tile_iter)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_blocks:
                stage_a.bind(kb_mma)
                stage_b.bind(kb_mma)
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

        while tile_epi < num_tiles:
            stage_acc.bind(tile_iter)
            coord_pair_m_epi, coord_n_epi = get_tile_coord_grouped_m(
                tile_epi,
                num_pair_m_blocks,
                num_n_blocks,
                PAIR_GROUP_M,
            )
            coord_cta_m_epi: ll.uint32 = coord_pair_m_epi * CTA_GROUP_SIZE + cta

            for wave in ll.unroll(range(NUM_MMA_M_WAVES)):
                for s in ll.unroll(range(NUM_STORES)):
                    stage_store.bind((tile_iter, wave, s))
                    cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                    for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                        tmem_col: ll.uint32 = ll.uint32(C_tmem + wave * PAIR_TILE_N + s * ST_TILE_N + i * ELEMS_PER_BG)
                        smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(local_tid), ll.uint32(i))
                        ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                    if epi_warp == 0:
                        output_issuer: ll.uint32 = ll.elect_one()
                        if output_issuer == 1:
                            offset_n_epi: ll.int32 = ll.int32(coord_n_epi * PAIR_TILE_N + s * ST_TILE_N)
                            offset_m_epi: ll.int32 = ll.int32(coord_cta_m_epi * CTA_TILE_M + wave * MMA_WAVE_M)
                            ri.tma_st_2d(C_smem, dD, (offset_n_epi, offset_m_epi), shape=(ST_TILE_M, ST_TILE_N))

            tile_iter = tile_iter + 1
            tile_epi = tile_epi + num_clusters


class A2AGemmFuse:
    """Launcher for fused A2A push, band publication, and o-proj GEMM."""

    MAX_RANKS = 64
    INT32_MAX = (1 << 31) - 1
    TMA_DESCRIPTOR_BYTES = 128
    DEFAULT_COMM_CTAS = int(os.environ.get("LK_A2A_FUSED_COMM_CTAS", "64"))
    NUM_THREADS = 256
    CLUSTER_SIZE = CLUSTER_SIZE
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    CTA_TILE_M = CTA_TILE_M
    PAIR_TILE_M = PAIR_TILE_M
    PAIR_TILE_N = PAIR_TILE_N
    K_TILE = K_TILE
    COMM_TILE_M = COMM_TILE_M
    COMM_TILE_K = COMM_TILE_K
    ST_TILE_M = ST_TILE_M
    ST_TILE_N = ST_TILE_N
    kernel = a2a_gemm_fuse_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from pathlib import Path

        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source),
            "a2a_gemm_fuse_sm100_kernel",
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
        attn_out: torch.Tensor,
        workspace_handle,
        workspace: torch.Tensor,
        dB,
        out_shape,
        m,
        n,
        k,
        rank,
        world_size,
        comm_ctas=None,
    ):
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        if not 1 <= world_size <= cls.MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"A2AGemmFuse supports 1..{cls.MAX_RANKS} ranks")
        if m % cls.PAIR_TILE_M or n % cls.PAIR_TILE_N or k % cls.K_TILE:
            raise ValueError(f"M, N, and K must be divisible by pair tiles "
                             f"{cls.PAIR_TILE_M}, {cls.PAIR_TILE_N}, and {cls.K_TILE}")
        if k % world_size:
            raise ValueError("K must be divisible by world_size")
        band_features = k // world_size
        if band_features % cls.K_TILE or band_features % cls.COMM_TILE_K:
            raise ValueError(
                f"per-rank K band must be divisible by K_TILE={cls.K_TILE} and COMM_TILE_K={cls.COMM_TILE_K}")
        if cls.CTA_TILE_M % cls.COMM_TILE_M:
            raise ValueError(f"CTA_TILE_M={cls.CTA_TILE_M} must be divisible by "
                             f"COMM_TILE_M={cls.COMM_TILE_M}")
        if attn_out.shape != (m * world_size, band_features):
            raise ValueError("attn_out must have shape (M * world_size, K / world_size)")

        peers = [
            workspace_handle.get_buffer(peer_rank, workspace.shape, workspace.dtype) for peer_rank in range(world_size)
        ]

        def comm_descriptor(tensor, rows, cols):
            return create_tma_2d_descriptor(
                tensor,
                gmem_inner_dim=cols,
                gmem_outer_dim=rows,
                smem_inner_dim=cls.COMM_TILE_K,
                smem_outer_dim=cls.COMM_TILE_M,
                gmem_outer_stride=cols,
                swizzle_mode=128,
                oob_fill=False,
                l2_promotion=3,
            )

        d_peer = [comm_descriptor(peer, m, k) for peer in peers]
        # Device allocations are aligned; a 128-byte stride preserves alignment.
        descriptor_bytes = bytearray().join(bytes(desc) for desc in d_peer)
        peer_descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=workspace.device)
        if peer_descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")
        d_a_comm = comm_descriptor(attn_out, m * world_size, band_features)
        d_a = create_tma_2d_descriptor(
            workspace,
            gmem_inner_dim=k,
            gmem_outer_dim=m,
            smem_inner_dim=cls.K_TILE,
            smem_outer_dim=cls.CTA_TILE_M,
            gmem_outer_stride=k,
            swizzle_mode=128,
            oob_fill=True,
            l2_promotion=3,
        )

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

        pair_tiles = (m // cls.PAIR_TILE_M) * (n // cls.PAIR_TILE_N)
        if pair_tiles < 2:
            raise ValueError("A2A-GEMM fused requires at least two pair tiles")
        sms = torch.cuda.get_device_properties(workspace.device).multi_processor_count
        num_ctas = (min(sms, pair_tiles * cls.CLUSTER_SIZE) // cls.CLUSTER_SIZE) * cls.CLUSTER_SIZE
        if num_ctas == 0:
            raise ValueError("problem is too small for a two-CTA cluster")
        requested_comm_ctas = cls.DEFAULT_COMM_CTAS if comm_ctas is None else comm_ctas
        comm_ctas = max(cls.CLUSTER_SIZE, min(requested_comm_ctas, num_ctas) // cls.CLUSTER_SIZE * cls.CLUSTER_SIZE)

        num_m_blocks = m // cls.CTA_TILE_M
        ready = symm_mem.empty((world_size * num_m_blocks, ), dtype=torch.int32, device=workspace.device)
        ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
        ready_ptrs = torch.tensor(ready_handle.buffer_ptrs, dtype=torch.int64, device=workspace.device)
        ready.zero_()

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
            out,
            peers,
            m=m,
            n=n,
            k=k,
            band_features=band_features,
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
        out,
        peer_buffers,
        *,
        m,
        n,
        k,
        band_features,
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
        self.out = out
        self.peer_buffers = tuple(peer_buffers)
        self.args = (m, n, k, band_features, rank, world_size)
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
            *self.args,
            self.epoch,
            self.num_ctas,
            self.comm_ctas,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def release(self):
        self.peer_descriptors = None
        self.d_a_comm = None
        self.d_a = None
        self.d_b = None
        self.d_d = None
        self.ready_handle = None
        self.ready = None
        self.ready_ptrs = None
        self.peer_buffers = ()
