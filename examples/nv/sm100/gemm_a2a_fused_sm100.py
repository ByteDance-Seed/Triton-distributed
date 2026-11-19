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
"""Single-kernel SM100 QKV GEMM with epilogue-push Ulysses A2A."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

from pathlib import Path

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
NUM_STAGES = 4
NUM_EPI_STAGES = 1
MMA_WAVE_M = 128
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
GROUP_PAIR_M_TILES = 8

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // 2
NUM_N_STORES = PAIR_TILE_N // ST_TILE_N
NUM_TMA_STORE_STAGES = 2

NUM_COMPLETION_WARPS = 1
NUM_COMPLETION_THREADS = NUM_COMPLETION_WARPS * 32
NUM_THREADS = 256
ELE_BYTES = 2
SMEM_A = CTA_TILE_M * K_TILE * ELE_BYTES
SMEM_B = CTA_TILE_N_SHARD * K_TILE * ELE_BYTES

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // ELE_BYTES

SHARED_MEM_BYTES = (NUM_TMA_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_STAGES * (SMEM_A + SMEM_B) + 1024)


@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    frontend=dataflow_frontend,
    cluster_shape=(CLUSTER_SIZE, 1, 1),
    cta_group_shape=CTA_GROUP_SHAPE,
)
def gemm_a2a_fuse_push_sm100_kernel(
    dOutQPeers: ll.ptr[ll.TmaDescriptor],
    dOutKPeers: ll.ptr[ll.TmaDescriptor],
    dOutVPeers: ll.ptr[ll.TmaDescriptor],
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    ready_local: ll.ptr[ll.int32],
    ready_peers: ll.ptr[ll.ptr[ll.int32]],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    q_features: ll.uint32,
    kv_features: ll.uint32,
    rank: ll.uint32,
    world_size: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load_ab = g.role("load_ab", name="load_ab", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        complete = g.role("complete", name="wait_incoming", hw=HW.CUDACore, warps=[2])
        epi = g.role("epi", name="push_epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_TMA_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_STAGES, major="K")
        C_tmem = g.tmem("C_tmem", (CTA_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_EPI_STAGES)

        stage_ab = g.depends(src=load_ab, dst=mma, via=[A_smem, B_smem])
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

    num_cta_m_tiles: ll.uint32 = M // CTA_TILE_M
    num_pair_m_tiles: ll.uint32 = M // PAIR_TILE_M
    num_n_tiles: ll.uint32 = N // PAIR_TILE_N
    num_gemm_tiles: ll.uint32 = num_pair_m_tiles * num_n_tiles
    num_k_tiles: ll.uint32 = K // K_TILE
    total_q_features: ll.uint32 = world_size * q_features
    total_k_features: ll.uint32 = world_size * kv_features
    local_output_tiles: ll.uint32 = (q_features + 2 * kv_features) // PAIR_TILE_N
    num_ready: ll.uint32 = world_size * num_cta_m_tiles * local_output_tiles

    if warp_idx == 0:
        prefetch_issuer: ll.uint32 = ll.elect_one()
        if prefetch_issuer == 1:
            ll.prefetch_tma_descriptor(dA)
            ll.prefetch_tma_descriptor(dB)
            peer: ll.uint32 = ll.uint32(0)
            while peer < world_size:
                ll.prefetch_tma_descriptor(dOutQPeers[peer])
                ll.prefetch_tma_descriptor(dOutKPeers[peer])
                ll.prefetch_tma_descriptor(dOutVPeers[peer])
                peer = peer + 1

    with load_ab:
        load_issuer: ll.uint32 = ll.elect_one()
        tile_load: ll.uint32 = cluster_id
        while tile_load < num_gemm_tiles:
            coord_pair_m_load, coord_n_load = get_tile_coord_grouped_m(
                tile_load,
                num_pair_m_tiles,
                num_n_tiles,
                GROUP_PAIR_M_TILES,
            )
            offset_m_load: ll.int32 = ll.int32(coord_pair_m_load * PAIR_TILE_M)
            offset_n_load: ll.int32 = ll.int32(coord_n_load * PAIR_TILE_N)

            kb_load: ll.uint32 = ll.uint32(0)
            while kb_load < num_k_tiles:
                stage_ab.bind(kb_load)
                offset_k_load: ll.int32 = ll.int32(kb_load * K_TILE)
                if load_issuer == 1:
                    ri.tma_ld_2d(A_smem, dA, (offset_k_load, offset_m_load), A_plan, shape=(CTA_TILE_M, K_TILE))
                    ri.tma_ld_2d(B_smem, dB, (offset_k_load, offset_n_load), B_plan, shape=(CTA_TILE_N_SHARD, K_TILE))
                kb_load = kb_load + 1
            tile_load = tile_load + num_clusters

    with mma:
        mma_issuer: ll.uint32 = ll.elect_one()
        tile_mma: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0
        while tile_mma < num_gemm_tiles:
            stage_acc.bind(tile_iter)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_tiles:
                stage_ab.bind(kb_mma)
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
                            out_shape=(CTA_TILE_M, PAIR_TILE_N),
                            zero_init=kb_mma == 0,
                            tmem_d=tmem_off,
                        )
                kb_mma = kb_mma + 1

            tile_iter = tile_iter + 1
            tile_mma = tile_mma + num_clusters

    with epi:
        epi_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = epi_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter: ll.int32 = 0

        while tile_epi < num_gemm_tiles:
            stage_acc.bind(tile_iter)
            coord_pair_m_epi, coord_n_epi = get_tile_coord_grouped_m(
                tile_epi,
                num_pair_m_tiles,
                num_n_tiles,
                GROUP_PAIR_M_TILES,
            )
            coord_cta_m_epi: ll.uint32 = coord_pair_m_epi * CLUSTER_SIZE + cta
            offset_n_global: ll.uint32 = coord_n_epi * PAIR_TILE_N
            target_rank: ll.uint32 = ll.uint32(0)
            offset_n_local: ll.uint32 = ll.uint32(0)
            output_kind: ll.uint32 = ll.uint32(0)
            coord_n_signal: ll.uint32 = ll.uint32(0)
            if offset_n_global < total_q_features:
                target_rank = offset_n_global // q_features
                offset_n_local = offset_n_global % q_features
                coord_n_signal = offset_n_local // PAIR_TILE_N
            elif offset_n_global < total_q_features + total_k_features:
                output_kind = ll.uint32(1)
                relative_n: ll.uint32 = offset_n_global - total_q_features
                target_rank = relative_n // kv_features
                offset_n_local = relative_n % kv_features
                coord_n_signal = q_features // PAIR_TILE_N + offset_n_local // PAIR_TILE_N
            else:
                output_kind = ll.uint32(2)
                relative_n: ll.uint32 = offset_n_global - total_q_features - total_k_features
                target_rank = relative_n // kv_features
                offset_n_local = relative_n % kv_features
                coord_n_signal = ((q_features + kv_features) // PAIR_TILE_N + offset_n_local // PAIR_TILE_N)

            for m in ll.unroll(range(NUM_MMA_M_WAVES)):
                for n in ll.unroll(range(NUM_N_STORES)):
                    stage_store.bind((tile_iter, m, n))
                    cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                    for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                        tmem_col: ll.uint32 = ll.uint32(C_tmem + m * PAIR_TILE_N + n * ST_TILE_N + i * ELEMS_PER_BG)
                        smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(epi_tid), ll.uint32(i))
                        ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                    if epi_warp == 0:
                        store_issuer: ll.uint32 = ll.elect_one()
                        if store_issuer == 1:
                            offset_m_output: ll.int32 = ll.int32(rank * M + coord_cta_m_epi * CTA_TILE_M +
                                                                 m * MMA_WAVE_M)
                            offset_n_output: ll.int32 = ll.int32(offset_n_local + n * ST_TILE_N)
                            if output_kind == 0:
                                ri.tma_st_2d(
                                    C_smem,
                                    dOutQPeers[target_rank],
                                    (offset_n_output, offset_m_output),
                                    shape=(ST_TILE_M, ST_TILE_N),
                                )
                            elif output_kind == 1:
                                ri.tma_st_2d(
                                    C_smem,
                                    dOutKPeers[target_rank],
                                    (offset_n_output, offset_m_output),
                                    shape=(ST_TILE_M, ST_TILE_N),
                                )
                            else:
                                ri.tma_st_2d(
                                    C_smem,
                                    dOutVPeers[target_rank],
                                    (offset_n_output, offset_m_output),
                                    shape=(ST_TILE_M, ST_TILE_N),
                                )

            if epi_warp == 0:
                ll.tma_store_wait_group(0)
                ll.fence_proxy_async()
                ll.__syncwarp()
                signal_issuer: ll.uint32 = ll.elect_one()
                if signal_issuer == 1:
                    signal_idx: ll.uint32 = ((rank * num_cta_m_tiles + coord_cta_m_epi) * local_output_tiles +
                                             coord_n_signal)
                    ll.st_release_system(ready_peers[target_rank] + signal_idx, epoch)

            tile_iter = tile_iter + 1
            tile_epi = tile_epi + num_clusters

    with complete:
        wait_tid: ll.uint32 = ll.uint32(tid - 64)
        wait_idx: ll.uint32 = wait_tid
        if cta_id != 0:
            wait_idx = num_ready
        while wait_idx < num_ready:
            while ll.ld_acquire_system(ready_local + wait_idx) != epoch:
                pass
            wait_idx = wait_idx + NUM_COMPLETION_THREADS


class GemmA2AFusePush:
    """Launcher for GEMM epilogue-push A2A plus incoming completion."""

    MAX_RANKS = 64
    TMA_DESCRIPTOR_BYTES = 128
    FUSED_EPOCH_BASE = 1 << 30
    FUSED_EPOCH_RANGE = (1 << 30) - 2
    NUM_THREADS = NUM_THREADS
    CLUSTER_SIZE = CLUSTER_SIZE
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    CTA_TILE_M = CTA_TILE_M
    PAIR_TILE_M = PAIR_TILE_M
    PAIR_TILE_N = PAIR_TILE_N
    CTA_TILE_N_SHARD = CTA_TILE_N_SHARD
    K_TILE = K_TILE
    MMA_WAVE_M = MMA_WAVE_M
    ST_TILE_M = ST_TILE_M
    ST_TILE_N = ST_TILE_N
    kernel = gemm_a2a_fuse_push_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "gemm_a2a_fuse_push_sm100_kernel",
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
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )

    @classmethod
    def _peer_descriptors(cls, handle, tensor: torch.Tensor, world_size: int, features: int):
        import torch
        global_m = tensor.shape[0]
        peers = [
            handle.get_buffer(peer_rank, tensor.shape, tensor.dtype).view(global_m, features)
            for peer_rank in range(world_size)
        ]
        descs = [cls._descriptor(peer, inner=cls.ST_TILE_N, outer=cls.ST_TILE_M) for peer in peers]
        descriptor_bytes = bytearray().join(bytes(desc) for desc in descs)
        descriptors = torch.frombuffer(descriptor_bytes, dtype=torch.uint8).to(device=tensor.device)
        if descriptors.data_ptr() % cls.TMA_DESCRIPTOR_BYTES:
            raise RuntimeError("device TMA descriptor array is not 128-byte aligned")
        return descriptors, peers

    @classmethod
    def create(
        cls,
        compiled_kernel,
        *,
        d_a,
        d_b,
        m: int,
        n: int,
        k: int,
        rank: int,
        world_size: int,
        head_dim: int,
        q_heads: int,
        kv_heads: int,
    ):
        import torch
        import torch.distributed as dist
        import torch.distributed._symmetric_memory as symm_mem
        if not 1 <= world_size <= cls.MAX_RANKS or not 0 <= rank < world_size:
            raise ValueError(f"GemmA2AFusePush supports 1..{cls.MAX_RANKS} ranks")
        if min(m, n, k) <= 0:
            raise ValueError("M, N, and K must be positive")
        if q_heads % world_size or kv_heads % world_size:
            raise ValueError("q_heads and kv_heads must be divisible by world_size")
        if m % cls.PAIR_TILE_M or n % cls.PAIR_TILE_N or k % cls.K_TILE:
            raise ValueError("M, N, and K must be divisible by fused pair tiles "
                             f"{cls.PAIR_TILE_M}, {cls.PAIR_TILE_N}, and {cls.K_TILE}")

        local_q_heads, local_kv_heads = q_heads // world_size, kv_heads // world_size
        q_features, kv_features = local_q_heads * head_dim, local_kv_heads * head_dim
        expected_n = (q_heads + 2 * kv_heads) * head_dim
        if n != expected_n:
            raise ValueError(f"N must be (q_heads + 2 * kv_heads) * head_dim = {expected_n}")
        if q_features % cls.PAIR_TILE_N or kv_features % cls.PAIR_TILE_N:
            raise ValueError(f"local Q/KV features must be divisible by PAIR_TILE_N={cls.PAIR_TILE_N}")

        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.bfloat16
        global_m = m * world_size
        q_out = symm_mem.empty((global_m, local_q_heads, head_dim), dtype=dtype, device=device)
        q_handle = symm_mem.rendezvous(q_out, group=dist.group.WORLD)
        k_out = symm_mem.empty((global_m, local_kv_heads, head_dim), dtype=dtype, device=device)
        k_handle = symm_mem.rendezvous(k_out, group=dist.group.WORLD)
        v_out = symm_mem.empty((global_m, local_kv_heads, head_dim), dtype=dtype, device=device)
        v_handle = symm_mem.rendezvous(v_out, group=dist.group.WORLD)

        d_out_q_peers, q_peers = cls._peer_descriptors(q_handle, q_out, world_size, q_features)
        d_out_k_peers, k_peers = cls._peer_descriptors(k_handle, k_out, world_size, kv_features)
        d_out_v_peers, v_peers = cls._peer_descriptors(v_handle, v_out, world_size, kv_features)

        local_output_tiles = (q_features + 2 * kv_features) // cls.PAIR_TILE_N
        num_ready = world_size * (m // cls.CTA_TILE_M) * local_output_tiles
        ready = symm_mem.empty((num_ready, ), dtype=torch.int32, device=device)
        ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
        ready_peers = torch.tensor(ready_handle.buffer_ptrs[:world_size], dtype=torch.int64, device=device)
        ready.zero_()

        pair_tiles = (m // cls.PAIR_TILE_M) * (n // cls.PAIR_TILE_N)
        if pair_tiles < 2:
            raise ValueError("GEMM-A2A push requires at least two fused pair tiles")
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        num_ctas = (min(sms, pair_tiles * cls.CLUSTER_SIZE) // cls.CLUSTER_SIZE) * cls.CLUSTER_SIZE

        return cls(
            compiled_kernel,
            d_out_q_peers,
            d_out_k_peers,
            d_out_v_peers,
            d_a,
            d_b,
            ready,
            ready_peers,
            ready_handle,
            (q_out, k_out, v_out),
            (q_handle, k_handle, v_handle),
            tuple(q_peers + k_peers + v_peers),
            args=(m, n, k, q_features, kv_features, rank, world_size),
            num_ctas=num_ctas,
        )

    def __init__(
        self,
        compiled_kernel,
        d_out_q_peers,
        d_out_k_peers,
        d_out_v_peers,
        d_a,
        d_b,
        ready,
        ready_peers,
        ready_handle,
        outputs,
        output_handles,
        peer_buffers,
        *,
        args,
        num_ctas,
    ):
        self.compiled_kernel = compiled_kernel
        self.d_out_q_peers = d_out_q_peers
        self.d_out_k_peers = d_out_k_peers
        self.d_out_v_peers = d_out_v_peers
        self.d_a = d_a
        self.d_b = d_b
        self.ready = ready
        self.ready_peers = ready_peers
        self.ready_handle = ready_handle
        self.outputs = tuple(outputs)
        self.output_handles = tuple(output_handles)
        self.peer_buffers = tuple(peer_buffers)
        self.args = tuple(args)
        self.num_ctas = num_ctas
        self.iteration = 0

    def launch(self, stream=None):
        self.iteration = self.iteration % self.FUSED_EPOCH_RANGE + 1
        epoch = self.FUSED_EPOCH_BASE + self.iteration
        self.compiled_kernel(
            self.d_out_q_peers,
            self.d_out_k_peers,
            self.d_out_v_peers,
            self.d_a,
            self.d_b,
            self.ready,
            self.ready_peers,
            *self.args,
            epoch,
            self.num_ctas,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def release(self):
        self.d_out_q_peers = self.d_out_k_peers = self.d_out_v_peers = None
        self.d_a = self.d_b = None
        # get_buffer() views must die before any symmetric-memory handle.
        self.peer_buffers = ()
        self.ready_peers = None
        self.ready_handle = None
        self.ready = None
        self.output_handles = ()
        self.outputs = ()
