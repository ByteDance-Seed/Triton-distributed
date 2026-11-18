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
"""Persistent intra-node TMA pull reduce-scatter for SM100."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_along_n
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

TILE_M = 128
TILE_N = 256
READY_BM = 256
READY_BN = 256
NUM_LOAD_STAGES = 2
NUM_STORE_STAGES = 1
NUM_REDUCE_WARPS = 8
NUM_REDUCE_THREADS = NUM_REDUCE_WARPS * 32
ELEMS_PER_THREAD = TILE_M * TILE_N // NUM_REDUCE_THREADS
NUM_THREADS = (1 + NUM_REDUCE_WARPS + 1) * 32
TILE_BYTES = TILE_M * TILE_N * 2
SHARED_MEM_BYTES = (NUM_LOAD_STAGES + NUM_STORE_STAGES) * TILE_BYTES + 1024


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def rs_tma_pull_sm100_kernel(
    dPeers: ll.ptr[ll.TmaDescriptor],
    dOut: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready_ptrs: ll.ptr[ll.ptr[ll.int32]],
    M: ll.uint32,
    N: ll.uint32,
    m_per_rank: ll.uint32,
    rank: ll.uint32,
    world_size: ll.uint32,
    ready_n_blocks: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load = g.role("load", name="load_peers", hw=HW.TMA, warps=[0])
        reduce = g.role("reduce", name="reduce_peers", hw=HW.CUDACore, warps=[1, 2, 3, 4, 5, 6, 7, 8])
        store = g.role("store", name="store_output", hw=HW.TMA, warps=[9])

        peer_smem = g.smem("peer_smem", (TILE_M, TILE_N), dtype=ll.bfloat16, stages=NUM_LOAD_STAGES)
        out_smem = g.smem("out_smem", (TILE_M, TILE_N), dtype=ll.bfloat16, stages=NUM_STORE_STAGES)

        stage_peer = g.depends(src=load, dst=reduce, via=[peer_smem])
        stage_out = g.depends(src=reduce, dst=store, via=[out_smem])

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta_id: ll.uint32 = ll.blockIdx_x()
    num_m_tiles: ll.uint32 = m_per_rank // TILE_M
    num_n_tiles: ll.uint32 = N // TILE_N
    num_jobs: ll.uint32 = num_m_tiles * num_n_tiles

    if warp_idx == 0:
        descriptor_issuer: ll.uint32 = ll.elect_one()
        if descriptor_issuer == 1:
            src_prefetch: ll.uint32 = ll.uint32(0)
            while src_prefetch < world_size:
                ll.prefetch_tma_descriptor(dPeers[src_prefetch])
                src_prefetch = src_prefetch + 1
            ll.prefetch_tma_descriptor(dOut)

    with load:
        load_issuer: ll.uint32 = ll.elect_one()
        load_job: ll.uint32 = cta_id
        while load_job < num_jobs:
            coord_m_load, coord_n_load = get_tile_coord_along_n(load_job, num_n_tiles)
            offset_m_load: ll.uint32 = rank * m_per_rank + coord_m_load * TILE_M
            offset_n_load: ll.uint32 = coord_n_load * TILE_N
            ready_idx: ll.uint32 = (offset_m_load // READY_BM) * ready_n_blocks + offset_n_load // READY_BN

            load_step: ll.uint32 = ll.uint32(0)
            while load_step < world_size:
                stage_peer.bind((load_job, load_step))
                load_src: ll.uint32 = (rank + cta_id + load_step) % world_size
                if load_issuer == 1:
                    while ll.ld_acquire_system(ready_ptrs[load_src] + ready_idx) != epoch:
                        pass
                ll.__syncwarp()
                if load_issuer == 1:
                    ri.tma_ld_2d(
                        peer_smem,
                        dPeers[load_src],
                        (ll.int32(offset_n_load), ll.int32(offset_m_load)),
                        shape=(TILE_M, TILE_N),
                    )
                load_step = load_step + 1
            load_job = load_job + num_ctas

    with reduce:
        reduce_tid: ll.int32 = tid - 32
        reduce_job: ll.uint32 = cta_id
        while reduce_job < num_jobs:
            accum = ll.empty([128], dtype=ll.float32, scope="local")
            for init_i in ll.unroll(range(ELEMS_PER_THREAD)):
                accum[init_i] = ll.float32(0.0)

            reduce_src: ll.uint32 = ll.uint32(0)
            while reduce_src < world_size:
                stage_peer.bind((reduce_job, reduce_src))
                for reduce_i in ll.unroll(range(ELEMS_PER_THREAD)):
                    elem_idx: ll.int32 = reduce_tid + reduce_i * NUM_REDUCE_THREADS
                    value: ll.bfloat16 = ri.smem_ld_bf16(peer_smem, elem_idx)
                    accum[reduce_i] = accum[reduce_i] + ll.bf16_to_float(value)
                reduce_src = reduce_src + 1

            stage_out.bind(reduce_job)
            for store_i in ll.unroll(range(ELEMS_PER_THREAD)):
                out_idx: ll.int32 = reduce_tid + store_i * NUM_REDUCE_THREADS
                ri.smem_st_bf16(out_smem, out_idx, ll.float_to_bf16(accum[store_i]))
            reduce_job = reduce_job + num_ctas

    with store:
        store_job: ll.uint32 = cta_id
        while store_job < num_jobs:
            store_issuer: ll.uint32 = ll.elect_one()
            coord_m_store, coord_n_store = get_tile_coord_along_n(store_job, num_n_tiles)
            stage_out.bind(store_job)
            if store_issuer == 1:
                ri.tma_st_2d(
                    out_smem,
                    dOut,
                    (ll.int32(coord_n_store * TILE_N), ll.int32(coord_m_store * TILE_M)),
                    shape=(TILE_M, TILE_N),
                )
            store_job = store_job + num_ctas


class TmaPullReduceScatter:
    """Launcher for owner-side TMA pull plus FP32 local reduction."""

    MAX_RANKS = 64
    TMA_DESCRIPTOR_BYTES = 128
    DEFAULT_COMM_CTAS = 32
    NUM_THREADS = NUM_THREADS
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    TILE_M = TILE_M
    TILE_N = TILE_N
    READY_BM = READY_BM
    READY_BN = READY_BN
    kernel = rs_tma_pull_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "rs_tma_pull_sm100_kernel",
            grid=(1, 1, 1),
            block=(cls.NUM_THREADS, 1, 1),
            shared_mem_bytes=cls.SHARED_MEM_BYTES,
            arch="sm_100a",
            cache_dir=cache_dir,
            verbose=verbose,
            include_paths=[Path(__file__).resolve().parents[2]],
        )

    @classmethod
    def create(
        cls,
        compiled_kernel,
        *,
        workspace_handle,
        partial: torch.Tensor,
        ready_handle,
        ready: torch.Tensor,
        rank: int,
        world_size: int,
        comm_ctas: int | None = None,
    ):
        import torch
        if partial.ndim != 2 or partial.dtype != torch.bfloat16:
            raise ValueError("partial must be a two-dimensional BF16 tensor")
        if not 1 <= world_size <= cls.MAX_RANKS:
            raise ValueError(f"TMA pull RS supports 1..{cls.MAX_RANKS} ranks")

        m, n = partial.shape
        if m % world_size:
            raise ValueError("M must be divisible by world_size")
        m_per_rank = m // world_size
        if m_per_rank % cls.TILE_M or n % cls.TILE_N:
            raise ValueError(f"M/world_size and N must be divisible by {cls.TILE_M} and {cls.TILE_N}")
        if m % cls.READY_BM or n % cls.READY_BN:
            raise ValueError(f"M and N must be divisible by ready tile {cls.READY_BM}x{cls.READY_BN}")

        expected_ready = (m // cls.READY_BM) * (n // cls.READY_BN)
        if ready.numel() != expected_ready or ready.dtype != torch.int32:
            raise ValueError(f"ready must be int32[{expected_ready}]")

        peers = [
            workspace_handle.get_buffer(peer_rank, partial.shape, partial.dtype) for peer_rank in range(world_size)
        ]

        def descriptor(tensor):
            return create_tma_2d_descriptor(
                tensor,
                gmem_inner_dim=n,
                gmem_outer_dim=m,
                smem_inner_dim=cls.TILE_N,
                smem_outer_dim=cls.TILE_M,
                gmem_outer_stride=n,
                swizzle_mode=0,
                oob_fill=False,
                l2_promotion=3,
            )

        peer_descs = [descriptor(peer) for peer in peers]
        descriptor_bytes = bytearray().join(bytes(desc) for desc in peer_descs)
        peer_descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=partial.device)
        if peer_descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")

        out = torch.empty((m_per_rank, n), dtype=partial.dtype, device=partial.device)
        d_out = create_tma_2d_descriptor(
            out,
            gmem_inner_dim=n,
            gmem_outer_dim=m_per_rank,
            smem_inner_dim=cls.TILE_N,
            smem_outer_dim=cls.TILE_M,
            gmem_outer_stride=n,
            swizzle_mode=0,
            oob_fill=False,
            l2_promotion=3,
        )
        ready_ptrs = torch.tensor(ready_handle.buffer_ptrs, dtype=torch.int64, device=partial.device)

        jobs = (m_per_rank // cls.TILE_M) * (n // cls.TILE_N)
        sms = torch.cuda.get_device_properties(partial.device).multi_processor_count
        requested = cls.DEFAULT_COMM_CTAS if comm_ctas is None else comm_ctas
        if requested <= 0:
            raise ValueError("comm_ctas must be positive")
        num_ctas = max(1, min(requested, jobs, sms))

        return cls(
            compiled_kernel,
            peer_descriptors,
            d_out,
            ready_ptrs,
            out,
            peers,
            m=m,
            n=n,
            m_per_rank=m_per_rank,
            rank=rank,
            world_size=world_size,
            num_ctas=num_ctas,
        )

    def __init__(
        self,
        compiled_kernel,
        peer_descriptors,
        d_out,
        ready_ptrs,
        out,
        peer_buffers,
        *,
        m,
        n,
        m_per_rank,
        rank,
        world_size,
        num_ctas,
    ):
        self.compiled_kernel = compiled_kernel
        self.peer_descriptors = peer_descriptors
        self.d_out = d_out
        self.ready_ptrs = ready_ptrs
        self.out = out
        self.peer_buffers = tuple(peer_buffers)
        self.args = (m, n, m_per_rank, rank, world_size, n // self.READY_BN)
        self.num_ctas = num_ctas

    def launch(self, epoch: int, stream=None):
        self.compiled_kernel(
            self.peer_descriptors,
            self.d_out,
            self.ready_ptrs,
            *self.args,
            epoch,
            self.num_ctas,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def release(self):
        self.peer_descriptors = None
        self.ready_ptrs = None
        self.peer_buffers = ()
