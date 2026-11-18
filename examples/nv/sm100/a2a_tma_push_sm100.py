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
"""Persistent intra-node TMA push Ulysses post-attention all-to-all for SM100."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

import os
from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

COMM_M = int(os.environ.get("LK_A2A_PUSH_COMM_M", "64"))
COMM_K = 64
PUSH_BM = 256
NUM_COMM_STAGES = int(os.environ.get("LK_A2A_PUSH_STAGES", "3"))
NUM_COMM_WARPS = 4
KSPLIT = int(os.environ.get("LK_A2A_PUSH_KSPLIT", "2"))
DEST_MODE = int(os.environ.get("LK_A2A_PUSH_DEST_MODE", "0"))
NUM_THREADS = NUM_COMM_WARPS * 32
ELE_BYTES = 2
SMEM_COMM = COMM_M * COMM_K * ELE_BYTES
SHARED_MEM_BYTES = NUM_COMM_WARPS * NUM_COMM_STAGES * SMEM_COMM + 1024


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def a2a_tma_push_sm100_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dPeers: ll.ptr[ll.TmaDescriptor],
    ready_peers: ll.ptr[ll.ptr[ll.int32]],
    rank: ll.uint32,
    world_size: ll.uint32,
    local_m: ll.uint32,
    local_features: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        comm0 = g.role("comm0", name="push_peer0", hw=HW.TMA, warps=[0])
        comm_smem0 = g.smem("comm_smem0", (COMM_M, COMM_K), dtype=ll.bfloat16, stages=NUM_COMM_STAGES)
        stage_comm0 = g.depends(src=comm0, dst=comm0, via=[comm_smem0])
        comm1 = g.role("comm1", name="push_peer1", hw=HW.TMA, warps=[1])
        comm_smem1 = g.smem("comm_smem1", (COMM_M, COMM_K), dtype=ll.bfloat16, stages=NUM_COMM_STAGES)
        stage_comm1 = g.depends(src=comm1, dst=comm1, via=[comm_smem1])
        comm2 = g.role("comm2", name="push_peer2", hw=HW.TMA, warps=[2])
        comm_smem2 = g.smem("comm_smem2", (COMM_M, COMM_K), dtype=ll.bfloat16, stages=NUM_COMM_STAGES)
        stage_comm2 = g.depends(src=comm2, dst=comm2, via=[comm_smem2])
        comm3 = g.role("comm3", name="push_peer3", hw=HW.TMA, warps=[3])
        comm_smem3 = g.smem("comm_smem3", (COMM_M, COMM_K), dtype=ll.bfloat16, stages=NUM_COMM_STAGES)
        stage_comm3 = g.depends(src=comm3, dst=comm3, via=[comm_smem3])

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta_id: ll.uint32 = ll.blockIdx_x()

    num_m_blocks: ll.uint32 = local_m // PUSH_BM
    m_chunks: ll.uint32 = PUSH_BM // COMM_M
    band_k_tiles: ll.uint32 = local_features // COMM_K
    part_k_tiles: ll.uint32 = band_k_tiles // KSPLIT
    if DEST_MODE == 1:
        dest_count: ll.uint32 = world_size - 1
    else:
        dest_count: ll.uint32 = world_size
    jobs_per_m_block: ll.uint32 = dest_count * KSPLIT
    num_jobs: ll.uint32 = num_m_blocks * jobs_per_m_block
    num_workers: ll.uint32 = num_ctas * NUM_COMM_WARPS

    if warp_idx == 0:
        prefetch_issuer: ll.uint32 = ll.elect_one()
        if prefetch_issuer == 1:
            ll.prefetch_tma_descriptor(dA)
            peer: ll.uint32 = ll.uint32(0)
            while peer < world_size:
                ll.prefetch_tma_descriptor(dPeers[peer])
                peer = peer + 1

    with comm0:
        job: ll.uint32 = cta_id
        while job < num_jobs:
            mb: ll.uint32 = job // jobs_per_m_block
            rem: ll.uint32 = job % jobs_per_m_block
            ks: ll.uint32 = rem // dest_count
            dest_rank: ll.uint32 = rem % dest_count
            if DEST_MODE == 1:
                # Diagnostic: remote destinations only.
                dest_rank = (rank + 1 + dest_rank) % world_size
            elif DEST_MODE == 2:
                # Diagnostic: local destination only.
                dest_rank = rank
            cm: ll.uint32 = ll.uint32(0)
            while cm < m_chunks:
                kp: ll.uint32 = ll.uint32(0)
                while kp < part_k_tiles:
                    kt: ll.uint32 = ks * part_k_tiles + kp
                    stage_comm0.bind((job, cm, kp))
                    comm_issuer: ll.uint32 = ll.elect_one()
                    k_src: ll.int32 = ll.int32(kt * COMM_K)
                    row_src: ll.int32 = ll.int32(dest_rank * local_m + mb * PUSH_BM + cm * COMM_M)
                    k_dst: ll.int32 = ll.int32(rank * local_features + kt * COMM_K)
                    row_dst: ll.int32 = ll.int32(mb * PUSH_BM + cm * COMM_M)
                    if comm_issuer == 1:
                        ri.tma_ld_2d(comm_smem0, dA, (k_src, row_src), shape=(COMM_M, COMM_K))
                        ri.tma_st_2d(comm_smem0, dPeers[dest_rank], (k_dst, row_dst), shape=(COMM_M, COMM_K))
                    kp = kp + 1
                cm = cm + 1
            # Publish only after every store for this (dest, m-block) is visible.
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            publish_issuer: ll.uint32 = ll.elect_one()
            if publish_issuer == 1:
                signal_idx: ll.uint32 = (rank * num_m_blocks + mb) * KSPLIT + ks
                ll.st_release_system(ready_peers[dest_rank] + signal_idx, epoch)
            job = job + num_workers

    with comm1:
        job1: ll.uint32 = num_ctas + cta_id
        while job1 < num_jobs:
            mb1: ll.uint32 = job1 // jobs_per_m_block
            rem1: ll.uint32 = job1 % jobs_per_m_block
            ks1: ll.uint32 = rem1 // dest_count
            dest_rank1: ll.uint32 = rem1 % dest_count
            if DEST_MODE == 1:
                dest_rank1 = (rank + 1 + dest_rank1) % world_size
            elif DEST_MODE == 2:
                dest_rank1 = rank
            cm1: ll.uint32 = ll.uint32(0)
            while cm1 < m_chunks:
                kp1: ll.uint32 = ll.uint32(0)
                while kp1 < part_k_tiles:
                    kt1: ll.uint32 = ks1 * part_k_tiles + kp1
                    stage_comm1.bind((job1, cm1, kp1))
                    comm_issuer1: ll.uint32 = ll.elect_one()
                    k_src1: ll.int32 = ll.int32(kt1 * COMM_K)
                    row_src1: ll.int32 = ll.int32(dest_rank1 * local_m + mb1 * PUSH_BM + cm1 * COMM_M)
                    k_dst1: ll.int32 = ll.int32(rank * local_features + kt1 * COMM_K)
                    row_dst1: ll.int32 = ll.int32(mb1 * PUSH_BM + cm1 * COMM_M)
                    if comm_issuer1 == 1:
                        ri.tma_ld_2d(comm_smem1, dA, (k_src1, row_src1), shape=(COMM_M, COMM_K))
                        ri.tma_st_2d(comm_smem1, dPeers[dest_rank1], (k_dst1, row_dst1), shape=(COMM_M, COMM_K))
                    kp1 = kp1 + 1
                cm1 = cm1 + 1
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            publish_issuer1: ll.uint32 = ll.elect_one()
            if publish_issuer1 == 1:
                signal_idx1: ll.uint32 = (rank * num_m_blocks + mb1) * KSPLIT + ks1
                ll.st_release_system(ready_peers[dest_rank1] + signal_idx1, epoch)
            job1 = job1 + num_workers

    with comm2:
        job2: ll.uint32 = 2 * num_ctas + cta_id
        while job2 < num_jobs:
            mb2: ll.uint32 = job2 // jobs_per_m_block
            rem2: ll.uint32 = job2 % jobs_per_m_block
            ks2: ll.uint32 = rem2 // dest_count
            dest_rank2: ll.uint32 = rem2 % dest_count
            if DEST_MODE == 1:
                dest_rank2 = (rank + 1 + dest_rank2) % world_size
            elif DEST_MODE == 2:
                dest_rank2 = rank
            cm2: ll.uint32 = ll.uint32(0)
            while cm2 < m_chunks:
                kp2: ll.uint32 = ll.uint32(0)
                while kp2 < part_k_tiles:
                    kt2: ll.uint32 = ks2 * part_k_tiles + kp2
                    stage_comm2.bind((job2, cm2, kp2))
                    comm_issuer2: ll.uint32 = ll.elect_one()
                    k_src2: ll.int32 = ll.int32(kt2 * COMM_K)
                    row_src2: ll.int32 = ll.int32(dest_rank2 * local_m + mb2 * PUSH_BM + cm2 * COMM_M)
                    k_dst2: ll.int32 = ll.int32(rank * local_features + kt2 * COMM_K)
                    row_dst2: ll.int32 = ll.int32(mb2 * PUSH_BM + cm2 * COMM_M)
                    if comm_issuer2 == 1:
                        ri.tma_ld_2d(comm_smem2, dA, (k_src2, row_src2), shape=(COMM_M, COMM_K))
                        ri.tma_st_2d(comm_smem2, dPeers[dest_rank2], (k_dst2, row_dst2), shape=(COMM_M, COMM_K))
                    kp2 = kp2 + 1
                cm2 = cm2 + 1
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            publish_issuer2: ll.uint32 = ll.elect_one()
            if publish_issuer2 == 1:
                signal_idx2: ll.uint32 = (rank * num_m_blocks + mb2) * KSPLIT + ks2
                ll.st_release_system(ready_peers[dest_rank2] + signal_idx2, epoch)
            job2 = job2 + num_workers

    with comm3:
        job3: ll.uint32 = 3 * num_ctas + cta_id
        while job3 < num_jobs:
            mb3: ll.uint32 = job3 // jobs_per_m_block
            rem3: ll.uint32 = job3 % jobs_per_m_block
            ks3: ll.uint32 = rem3 // dest_count
            dest_rank3: ll.uint32 = rem3 % dest_count
            if DEST_MODE == 1:
                dest_rank3 = (rank + 1 + dest_rank3) % world_size
            elif DEST_MODE == 2:
                dest_rank3 = rank
            cm3: ll.uint32 = ll.uint32(0)
            while cm3 < m_chunks:
                kp3: ll.uint32 = ll.uint32(0)
                while kp3 < part_k_tiles:
                    kt3: ll.uint32 = ks3 * part_k_tiles + kp3
                    stage_comm3.bind((job3, cm3, kp3))
                    comm_issuer3: ll.uint32 = ll.elect_one()
                    k_src3: ll.int32 = ll.int32(kt3 * COMM_K)
                    row_src3: ll.int32 = ll.int32(dest_rank3 * local_m + mb3 * PUSH_BM + cm3 * COMM_M)
                    k_dst3: ll.int32 = ll.int32(rank * local_features + kt3 * COMM_K)
                    row_dst3: ll.int32 = ll.int32(mb3 * PUSH_BM + cm3 * COMM_M)
                    if comm_issuer3 == 1:
                        ri.tma_ld_2d(comm_smem3, dA, (k_src3, row_src3), shape=(COMM_M, COMM_K))
                        ri.tma_st_2d(comm_smem3, dPeers[dest_rank3], (k_dst3, row_dst3), shape=(COMM_M, COMM_K))
                    kp3 = kp3 + 1
                cm3 = cm3 + 1
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            publish_issuer3: ll.uint32 = ll.elect_one()
            if publish_issuer3 == 1:
                signal_idx3: ll.uint32 = (rank * num_m_blocks + mb3) * KSPLIT + ks3
                ll.st_release_system(ready_peers[dest_rank3] + signal_idx3, epoch)
            job3 = job3 + num_workers


class TmaPushA2A:
    """Launcher for owner-side TMA push Ulysses post-attention all-to-all."""

    MAX_RANKS = 64
    TMA_DESCRIPTOR_BYTES = 128
    DEFAULT_COMM_CTAS = 32
    NUM_THREADS = NUM_THREADS
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    COMM_M = COMM_M
    COMM_K = COMM_K
    PUSH_BM = PUSH_BM
    KSPLIT = KSPLIT
    kernel = a2a_tma_push_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "a2a_tma_push_sm100_kernel",
            grid=(1, 1, 1),
            block=(cls.NUM_THREADS, 1, 1),
            shared_mem_bytes=cls.SHARED_MEM_BYTES,
            arch="sm_100a",
            cache_dir=cache_dir,
            verbose=verbose,
            include_paths=[Path(__file__).resolve().parents[2]],
        )

    @staticmethod
    def _descriptor(tensor: torch.Tensor, *, rows: int, cols: int, inner: int, outer: int):
        import torch
        return create_tma_2d_descriptor(
            tensor,
            gmem_inner_dim=cols,
            gmem_outer_dim=rows,
            smem_inner_dim=inner,
            smem_outer_dim=outer,
            gmem_outer_stride=cols,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )

    @classmethod
    def create(
        cls,
        compiled_kernel,
        *,
        attn_out: torch.Tensor,
        workspace_handle,
        workspace: torch.Tensor,
        ready_handle,
        ready: torch.Tensor,
        rank: int,
        world_size: int,
        comm_ctas: int | None = None,
    ):
        import torch
        if attn_out.ndim != 2 or attn_out.dtype != torch.bfloat16:
            raise ValueError("attn_out must be a two-dimensional BF16 tensor")
        if not attn_out.is_cuda or not attn_out.is_contiguous():
            raise ValueError("attn_out must be a contiguous CUDA tensor")
        if workspace.ndim != 2 or workspace.dtype != torch.bfloat16:
            raise ValueError("workspace must be a two-dimensional BF16 tensor")
        if not 1 <= world_size <= cls.MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"TMA push A2A supports 1..{cls.MAX_RANKS} ranks")

        global_m, local_features = attn_out.shape
        local_m, k = workspace.shape
        if global_m != local_m * world_size:
            raise ValueError("attn_out rows must equal local_m * world_size")
        if k != local_features * world_size:
            raise ValueError("workspace cols must equal local_features * world_size")
        if local_m % cls.PUSH_BM:
            raise ValueError(f"local_m must be divisible by PUSH_BM={cls.PUSH_BM}")
        if local_features % (cls.COMM_K * cls.KSPLIT):
            raise ValueError(f"local features must be divisible by COMM_K*KSPLIT={cls.COMM_K * cls.KSPLIT}")

        num_m_blocks = local_m // cls.PUSH_BM
        expected_ready = world_size * num_m_blocks * cls.KSPLIT
        if ready.dtype != torch.int32 or ready.numel() != expected_ready:
            raise ValueError(f"ready must be int32[{expected_ready}]")
        if len(ready_handle.buffer_ptrs) < world_size:
            raise ValueError("ready_handle does not contain all peer pointers")

        d_a = cls._descriptor(attn_out, rows=global_m, cols=local_features, inner=cls.COMM_K, outer=cls.COMM_M)
        peers = [
            workspace_handle.get_buffer(peer_rank, workspace.shape, workspace.dtype) for peer_rank in range(world_size)
        ]
        peer_descs = [cls._descriptor(peer, rows=local_m, cols=k, inner=cls.COMM_K, outer=cls.COMM_M) for peer in peers]
        descriptor_bytes = bytearray().join(bytes(desc) for desc in peer_descs)
        peer_descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=workspace.device)
        if peer_descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")
        ready_peers = torch.tensor(ready_handle.buffer_ptrs[:world_size], dtype=torch.int64, device=workspace.device)

        jobs = world_size * num_m_blocks * cls.KSPLIT
        sms = torch.cuda.get_device_properties(workspace.device).multi_processor_count
        requested = cls.DEFAULT_COMM_CTAS if comm_ctas is None else comm_ctas
        if requested <= 0:
            raise ValueError("comm_ctas must be positive")
        num_ctas = max(1, min(requested, jobs, sms))

        return cls(
            compiled_kernel,
            d_a,
            peer_descriptors,
            ready_peers,
            peers,
            local_m=local_m,
            local_features=local_features,
            rank=rank,
            world_size=world_size,
            num_ctas=num_ctas,
        )

    def __init__(
        self,
        compiled_kernel,
        d_a,
        peer_descriptors,
        ready_peers,
        peer_buffers,
        *,
        local_m,
        local_features,
        rank,
        world_size,
        num_ctas,
    ):
        self.compiled_kernel = compiled_kernel
        self.d_a = d_a
        self.peer_descriptors = peer_descriptors
        self.ready_peers = ready_peers
        self.peer_buffers = tuple(peer_buffers)
        self.args = (rank, world_size, local_m, local_features)
        self.num_ctas = num_ctas

    def launch(self, epoch: int, stream=None):
        self.compiled_kernel(
            self.d_a,
            self.peer_descriptors,
            self.ready_peers,
            *self.args,
            epoch,
            self.num_ctas,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def release(self):
        self.d_a = None
        self.peer_descriptors = None
        self.ready_peers = None
        self.peer_buffers = ()
