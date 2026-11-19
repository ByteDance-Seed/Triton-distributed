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
"""Persistent intra-node TMA-pull Ulysses QKV all-to-all for SM100."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

TILE_M = 128
TILE_N = 128
Q_TILE_N = 256
LOAD_TILE_N = 256
READY_BM = 256
READY_BN = 256
NUM_LOAD_STAGES = 2
NUM_COPY_WARPS = 8
NUM_COPY_THREADS = NUM_COPY_WARPS * 32
Q_ELEMS_PER_THREAD = TILE_M * Q_TILE_N // NUM_COPY_THREADS
KV_ELEMS_PER_THREAD = TILE_M * TILE_N // NUM_COPY_THREADS
NUM_THREADS = (1 + NUM_COPY_WARPS) * 32
LOAD_TILE_BYTES = TILE_M * LOAD_TILE_N * 2
SHARED_MEM_BYTES = NUM_LOAD_STAGES * LOAD_TILE_BYTES + 1024


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def a2a_tma_pull_sm100_kernel(
    dPeers: ll.ptr[ll.TmaDescriptor],
    qOut: ll.ptr[ll.bfloat16],
    kOut: ll.ptr[ll.bfloat16],
    vOut: ll.ptr[ll.bfloat16],
    ready_ptrs: ll.ptr[ll.ptr[ll.int32]],
    local_m: ll.uint32,
    q_features: ll.uint32,
    kv_features: ll.uint32,
    sp_rank: ll.uint32,
    world_size: ll.uint32,
    ready_n_blocks: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load = g.role("load", name="pull_peer", hw=HW.TMA, warps=[0])
        copy = g.role("copy", name="store_qkv", hw=HW.CUDACore, warps=[1, 2, 3, 4, 5, 6, 7, 8])
        peer_smem = g.smem("peer_smem", (TILE_M, LOAD_TILE_N), dtype=ll.bfloat16, stages=NUM_LOAD_STAGES)
        stage_peer = g.depends(src=load, dst=copy, via=[peer_smem])

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta_id: ll.uint32 = ll.blockIdx_x()
    num_m_tiles: ll.uint32 = local_m // TILE_M
    q_tiles: ll.uint32 = q_features // Q_TILE_N
    kv_tiles: ll.uint32 = kv_features // TILE_N
    feature_tiles: ll.uint32 = q_tiles + 2 * kv_tiles
    # Schedule: feature -> M -> peer (self first). Matches GEMM strip publication
    # and starts remote pulls as soon as each feature/M becomes ready, instead of
    # draining the entire local peer before touching remotes.
    jobs_per_feature: ll.uint32 = world_size * num_m_tiles
    num_jobs: ll.uint32 = feature_tiles * jobs_per_feature
    total_q_features: ll.uint32 = world_size * q_features
    total_k_features: ll.uint32 = world_size * kv_features

    if warp_idx == 0:
        prefetch_issuer: ll.uint32 = ll.elect_one()
        if prefetch_issuer == 1:
            peer: ll.uint32 = ll.uint32(0)
            while peer < world_size:
                ll.prefetch_tma_descriptor(dPeers[peer])
                peer = peer + 1

    with load:
        load_issuer: ll.uint32 = ll.elect_one()
        load_job: ll.uint32 = cta_id
        while load_job < num_jobs:
            feature_tile: ll.uint32 = load_job // jobs_per_feature
            rem: ll.uint32 = load_job % jobs_per_feature
            load_mt: ll.uint32 = rem // world_size
            peer_slot: ll.uint32 = rem % world_size
            source_rank: ll.uint32 = (sp_rank + peer_slot) % world_size
            source_n: ll.uint32 = ll.uint32(0)
            if feature_tile < q_tiles:
                source_n = sp_rank * q_features + feature_tile * Q_TILE_N
            elif feature_tile < q_tiles + kv_tiles:
                source_n = total_q_features + sp_rank * kv_features + (feature_tile - q_tiles) * TILE_N
            else:
                source_n = (total_q_features + total_k_features + sp_rank * kv_features +
                            (feature_tile - q_tiles - kv_tiles) * TILE_N)

            ready_idx: ll.uint32 = (load_mt * TILE_M // READY_BM) * ready_n_blocks + source_n // READY_BN
            if load_issuer == 1:
                while ll.ld_acquire_system(ready_ptrs[source_rank] + ready_idx) != epoch:
                    pass
            ll.__syncwarp()
            stage_peer.bind(load_job)
            if load_issuer == 1:
                ri.tma_ld_2d(peer_smem, dPeers[source_rank],
                             (ll.int32(source_n // LOAD_TILE_N * LOAD_TILE_N), ll.int32(load_mt * TILE_M)),
                             shape=(TILE_M, LOAD_TILE_N))
            load_job = load_job + num_ctas

    with copy:
        copy_tid: ll.int32 = tid - 32
        copy_job: ll.uint32 = cta_id
        while copy_job < num_jobs:
            feature_tile: ll.uint32 = copy_job // jobs_per_feature
            rem: ll.uint32 = copy_job % jobs_per_feature
            copy_mt: ll.uint32 = rem // world_size
            peer_slot: ll.uint32 = rem % world_size
            source_rank: ll.uint32 = (sp_rank + peer_slot) % world_size
            source_n: ll.uint32 = ll.uint32(0)
            output_n: ll.uint32 = ll.uint32(0)
            output_kind: ll.uint32 = ll.uint32(0)
            if feature_tile < q_tiles:
                source_n = sp_rank * q_features + feature_tile * Q_TILE_N
                output_n = feature_tile * Q_TILE_N
            elif feature_tile < q_tiles + kv_tiles:
                source_n = total_q_features + sp_rank * kv_features + (feature_tile - q_tiles) * TILE_N
                output_kind = ll.uint32(1)
                output_n = (feature_tile - q_tiles) * TILE_N
            else:
                source_n = (total_q_features + total_k_features + sp_rank * kv_features +
                            (feature_tile - q_tiles - kv_tiles) * TILE_N)
                output_kind = ll.uint32(2)
                output_n = (feature_tile - q_tiles - kv_tiles) * TILE_N
            output_m: ll.uint32 = source_rank * local_m + copy_mt * TILE_M
            stage_peer.bind(copy_job)
            if feature_tile < q_tiles:
                for copy_i in ll.unroll(range(Q_ELEMS_PER_THREAD)):
                    elem: ll.int32 = copy_tid + copy_i * NUM_COPY_THREADS
                    row: ll.int32 = elem // Q_TILE_N
                    col: ll.int32 = elem % Q_TILE_N
                    value: ll.bfloat16 = ri.smem_ld_bf16(peer_smem, elem)
                    qOut[(output_m + row) * q_features + output_n + col] = value
            else:
                source_subtile: ll.int32 = ll.int32(source_n % LOAD_TILE_N)
                for copy_i in ll.unroll(range(KV_ELEMS_PER_THREAD)):
                    elem: ll.int32 = copy_tid + copy_i * NUM_COPY_THREADS
                    row: ll.int32 = elem // TILE_N
                    col: ll.int32 = elem % TILE_N
                    value: ll.bfloat16 = ri.smem_ld_bf16(peer_smem, row * LOAD_TILE_N + source_subtile + col)
                    if output_kind == 1:
                        kOut[(output_m + row) * kv_features + output_n + col] = value
                    else:
                        vOut[(output_m + row) * kv_features + output_n + col] = value
            copy_job = copy_job + num_ctas


class TmaPullQKVA2A:
    """Launcher for owner-side TMA pull and Ulysses QKV layout conversion."""

    MAX_RANKS = 64
    TMA_DESCRIPTOR_BYTES = 128
    DEFAULT_COMM_CTAS = 36
    NUM_THREADS = NUM_THREADS
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    TILE_M = TILE_M
    TILE_N = TILE_N
    Q_TILE_N = Q_TILE_N
    LOAD_TILE_N = LOAD_TILE_N
    READY_BM = READY_BM
    READY_BN = READY_BN
    kernel = a2a_tma_pull_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "a2a_tma_pull_sm100_kernel",
            grid=(1, 1, 1),
            block=(cls.NUM_THREADS, 1, 1),
            shared_mem_bytes=cls.SHARED_MEM_BYTES,
            arch="sm_100a",
            cache_dir=cache_dir,
            verbose=verbose,
            include_paths=[Path(__file__).resolve().parents[2]],
        )

    @staticmethod
    def _descriptor(tensor: torch.Tensor, *, inner: int, outer: int):
        import torch
        rows, cols = tensor.shape
        return create_tma_2d_descriptor(
            tensor,
            gmem_inner_dim=cols,
            gmem_outer_dim=rows,
            smem_inner_dim=inner,
            smem_outer_dim=outer,
            gmem_outer_stride=cols,
            swizzle_mode=0,
            oob_fill=False,
            l2_promotion=3,
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
        head_dim: int,
        q_heads: int,
        kv_heads: int,
        comm_ctas: int | None = None,
    ):
        import torch
        if partial.ndim != 2 or partial.dtype != torch.bfloat16:
            raise ValueError("partial must be a two-dimensional BF16 tensor")
        if not partial.is_cuda or not partial.is_contiguous():
            raise ValueError("partial must be a contiguous CUDA tensor")
        if not 1 <= world_size <= cls.MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"TMA pull A2A supports 1..{cls.MAX_RANKS} ranks")
        if q_heads <= 0 or kv_heads <= 0 or head_dim <= 0:
            raise ValueError("q_heads, kv_heads, and head_dim must be positive")
        if q_heads % world_size or kv_heads % world_size:
            raise ValueError("q_heads and kv_heads must be divisible by world_size")

        local_m, n = partial.shape
        local_q_heads, local_kv_heads = q_heads // world_size, kv_heads // world_size
        q_features, kv_features = local_q_heads * head_dim, local_kv_heads * head_dim
        expected_n = (q_heads + 2 * kv_heads) * head_dim
        if n != expected_n:
            raise ValueError(f"partial N must be (q_heads + 2 * kv_heads) * head_dim = {expected_n}")
        if local_m % cls.TILE_M or q_features % cls.Q_TILE_N or kv_features % cls.TILE_N:
            raise ValueError(f"local M and local Q/KV features must be divisible by A2A tile {cls.TILE_M}x{cls.TILE_N}")
        if local_m % cls.READY_BM or n % cls.READY_BN:
            raise ValueError(f"local M and N must be divisible by ready tile {cls.READY_BM}x{cls.READY_BN}")

        expected_ready = (local_m // cls.READY_BM) * (n // cls.READY_BN)
        if ready.dtype != torch.int32 or ready.numel() != expected_ready:
            raise ValueError(f"ready must be int32[{expected_ready}]")
        if len(ready_handle.buffer_ptrs) < world_size:
            raise ValueError("ready_handle does not contain all peer pointers")

        peers = [
            workspace_handle.get_buffer(peer_rank, partial.shape, partial.dtype) for peer_rank in range(world_size)
        ]
        peer_descs = [cls._descriptor(peer, inner=cls.LOAD_TILE_N, outer=cls.TILE_M) for peer in peers]
        descriptor_bytes = bytearray().join(bytes(desc) for desc in peer_descs)
        peer_descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=partial.device)
        if peer_descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")

        global_m = local_m * world_size
        q_out = torch.empty((global_m, local_q_heads, head_dim), dtype=partial.dtype, device=partial.device)
        k_out = torch.empty((global_m, local_kv_heads, head_dim), dtype=partial.dtype, device=partial.device)
        v_out = torch.empty_like(k_out)
        ready_ptrs = torch.tensor(ready_handle.buffer_ptrs[:world_size], dtype=torch.int64, device=partial.device)

        feature_tiles = q_features // cls.Q_TILE_N + 2 * (kv_features // cls.TILE_N)
        jobs = world_size * (local_m // cls.TILE_M) * feature_tiles
        sms = torch.cuda.get_device_properties(partial.device).multi_processor_count
        requested = cls.DEFAULT_COMM_CTAS if comm_ctas is None else comm_ctas
        if requested <= 0:
            raise ValueError("comm_ctas must be positive")
        num_ctas = max(1, min(requested, jobs, sms))

        return cls(
            compiled_kernel,
            peer_descriptors,
            ready_ptrs,
            (q_out, k_out, v_out),
            peers,
            local_m=local_m,
            q_features=q_features,
            kv_features=kv_features,
            rank=rank,
            world_size=world_size,
            ready_n_blocks=n // cls.READY_BN,
            num_ctas=num_ctas,
        )

    def __init__(
        self,
        compiled_kernel,
        peer_descriptors,
        ready_ptrs,
        outputs,
        peer_buffers,
        *,
        local_m,
        q_features,
        kv_features,
        rank,
        world_size,
        ready_n_blocks,
        num_ctas,
    ):
        self.compiled_kernel = compiled_kernel
        self.peer_descriptors = peer_descriptors
        self.ready_ptrs = ready_ptrs
        self.outputs = tuple(outputs)
        self.peer_buffers = tuple(peer_buffers)
        self.args = (local_m, q_features, kv_features, rank, world_size, ready_n_blocks)
        self.num_ctas = num_ctas

    def launch(self, epoch: int, stream=None):
        self.compiled_kernel(
            self.peer_descriptors,
            self.outputs[0],
            self.outputs[1],
            self.outputs[2],
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
        self.outputs = ()
        self.peer_buffers = ()
