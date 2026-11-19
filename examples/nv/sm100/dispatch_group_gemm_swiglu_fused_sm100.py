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
"""Role-fused pull dispatch, grouped FC1 GEMM, and SwiGLU on SM100."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

import os
from pathlib import Path
from typing import Sequence

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.atom.scheduler import get_tile_coord_grouped_m
from little_kernel.backends.nv.common.language.intrin.cuda_asm import rcp_f32
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor, create_tma_3d_descriptor

# A physical 2x2 cluster contains a 1x2 grid of 2-CTA UMMA groups.
# Group-grid X advances M; group-grid Y advances N.  This makes A reusable
# along Y and expert-local B reusable along X.
# PAIR_TILE_M/PAIR_TILE_N are one cooperative 2-CTA group's logical
# output tile. Each participant owns CTA_TILE_M A rows and
# CTA_TILE_N_SHARD B columns.
PAIR_TILE_M = 256
PAIR_TILE_N = 256
K_TILE = 64
CTA_GROUP_SIZE = 2
CTA_GROUP_SHAPE = (CTA_GROUP_SIZE, 1, 1)
CTA_TILE_M = PAIR_TILE_M // CTA_GROUP_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CTA_GROUP_SIZE
MMA_WAVE_M = CTA_TILE_M
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
CLUSTER_SHAPE = (2, 2, 1)
GROUP_GRID = tuple(c // g for c, g in zip(CLUSTER_SHAPE, CTA_GROUP_SHAPE))
CLUSTER_M, CLUSTER_N = GROUP_GRID[0], GROUP_GRID[1]
CLUSTER_SIZE = CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1] * CLUSTER_SHAPE[2]
EXPERT_ALIGNMENT = PAIR_TILE_M * CLUSTER_M
NUM_A_STAGES = int(os.environ.get("LK_DGG_SWIGLU_A_STAGES", "5"))
NUM_B_STAGES = int(os.environ.get("LK_DGG_SWIGLU_B_STAGES", "5"))
NUM_DISPATCH_STAGES = int(os.environ.get("LK_DGG_SWIGLU_DISPATCH_STAGES", "1"))
NUM_ACC_STAGES = 2
SCHEDULER_GROUP_M = int(os.environ.get("LK_DGG_SWIGLU_GROUP_M", "16"))

ELE_BYTES = 2
COMM_TILE_K = int(os.environ.get("LK_DGG_SWIGLU_DISPATCH_K", "8192"))
SMEM_DISPATCH = COMM_TILE_K * ELE_BYTES
SMEM_A = CTA_TILE_M * K_TILE * ELE_BYTES
SMEM_B = CTA_TILE_N_SHARD * K_TILE * ELE_BYTES

# The accumulator is PAIR_TILE_N columns, but adjacent 32-column chunks are
# (gate, up), so the fused output tile has PAIR_TILE_N / 2 columns.
SWIGLU_EPI_TILE_N = 32
OUTPUT_TILE_N = PAIR_TILE_N // 2
# CuTe uses two 128x32 SW64 store stages. Keep the same 16 KiB footprint
# as one 128x64 SW128 stage until LK has the matching x32 TMEM copy layout.
ST_TILE_N = 2 * SWIGLU_EPI_TILE_N
SWIZZLE_CD_BYTES = ST_TILE_N * ELE_BYTES
ST_TILE_M = MMA_WAVE_M
NUM_OUTPUT_STORES = OUTPUT_TILE_N // ST_TILE_N
NUM_STORE_STAGES = int(os.environ.get("LK_DGG_SWIGLU_STORE_STAGES", "1"))
DEFAULT_RESERVED_SMS = int(os.environ.get("LK_DGG_SWIGLU_RESERVED_SMS", "0"))

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // ELE_BYTES

LOG2_E = 1.4426950408889634

NUM_THREADS = 8 * 32
SHARED_MEM_BYTES = (NUM_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_A_STAGES * SMEM_A + NUM_B_STAGES * SMEM_B +
                    NUM_DISPATCH_STAGES * SMEM_DISPATCH + 2048)


@lk.ll_kernel(backend="cuda", is_entry=False)
def swiglu_f32(gate: ll.float32, up: ll.float32) -> ll.float32:
    return gate * rcp_f32(ll.exp2_approx_f32(gate * ll.float32(-LOG2_E)) + 1.0) * up


@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    frontend=dataflow_frontend,
    cluster_shape=CLUSTER_SHAPE,
    cta_group_shape=CTA_GROUP_SHAPE,
)
def dispatch_group_gemm_swiglu_fuse_sm100_kernel(
    input_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],
    dispatch_output: ll.Tensor[ll.bfloat16],
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    token_src_rank_topk_and_indices: ll.ptr[ll.int64],
    expert_counts: ll.ptr[ll.int32],
    expert_ready: ll.ptr[ll.int32],
    expert_counters: ll.ptr[ll.int32],
    M: ll.uint32,
    N: ll.uint32,
    K: ll.uint32,
    num_experts: ll.uint32,
    epoch: ll.int32,
    num_ctas: ll.uint32,
) -> ll.void:
    """Pull valid token rows, then consume each ready expert in grouped GEMM."""
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        epi = g.role("epi", name="swiglu_epilogue", hw=HW.CUDACore, warps=[0, 1, 2, 3])
        dispatch_load = g.role("dispatch_load", name="dispatch_g2s", hw=HW.TMA, warps=[4])
        dispatch_store = g.role("dispatch_store", name="dispatch_s2g", hw=HW.TMA, warps=[5])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[6])
        load_ab = g.role("load_ab", name="load_ab", hw=HW.TMA, warps=[7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_A_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_B_STAGES, major="K")
        dispatch_smem = g.smem("dispatch_smem", (COMM_TILE_K, ), dtype=ll.bfloat16, stages=NUM_DISPATCH_STAGES)
        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_ACC_STAGES)

        stage_ab = g.depends(src=load_ab, dst=mma, via=[A_smem, B_smem])
        stage_dispatch = g.depends(src=dispatch_load, dst=dispatch_store, via=[dispatch_smem])
        stage_acc = g.depends(src=mma, dst=epi, via=[C_tmem])
        stage_store = g.depends(src=epi, dst=epi, via=[C_smem])
        A_plan = g.tma_transfer(
            mcast_group_dim="y",
            inter_cta_coord_dim=1,
            inter_cta_stride=CTA_TILE_M,
        )
        B_plan = g.tma_transfer(
            mcast_group_dim="x",
            inter_cta_coord_dim=1,
            inter_cta_stride=CTA_TILE_N_SHARD,
        )

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    lane_idx: ll.int32 = ll.get_lane_idx()
    cluster_rank: ll.uint32 = ll.cluster_rank()
    cta_m: ll.uint32 = ri.rank_in_group()
    group_m: ll.uint32 = ri.group_coord("x")
    group_n: ll.uint32 = ri.group_coord("y")
    cluster_id: ll.uint32 = ll.blockIdx_z()
    cta_id: ll.uint32 = cluster_id * CLUSTER_SIZE + cluster_rank
    num_clusters: ll.uint32 = num_ctas // CLUSTER_SIZE

    num_m_blocks: ll.uint32 = M // PAIR_TILE_M
    num_n_blocks: ll.uint32 = N // PAIR_TILE_N
    num_m_clusters: ll.uint32 = num_m_blocks // CLUSTER_M
    num_n_clusters: ll.uint32 = num_n_blocks // CLUSTER_N
    num_tiles: ll.uint32 = num_m_clusters * num_n_clusters
    num_k_blocks: ll.uint32 = K // K_TILE

    if warp_idx == 7 and ll.elect_one():
        ll.prefetch_tma_descriptor(dA)
        ll.prefetch_tma_descriptor(dB)
        ll.prefetch_tma_descriptor(dD)

    # Relay one contiguous token row with raw-pointer bulk G2S/S2G
    with dispatch_load:
        cur_expert: ll.uint32 = ll.uint32(0)
        dA_local_offset: ll.uint32 = ll.uint32(0)
        while cur_expert < num_experts:
            num_tok_cur_expert: ll.uint32 = ll.uint32(expert_counts[cur_expert])
            dispatch_batch: ll.uint32 = cta_id
            while dispatch_batch < num_tok_cur_expert:
                lane_token: ll.uint32 = dispatch_batch + lane_idx * num_ctas
                src_token: ll.int32 = ll.int32(0)
                src_rank: ll.int32 = ll.int32(0)
                if lane_token < num_tok_cur_expert:
                    encoded: ll.int64 = token_src_rank_topk_and_indices[dA_local_offset + lane_token]
                    src_token = ll.int32(encoded & ll.int64(0xFFFFFFFF))
                    src_rank = ll.int32((encoded >> 48) & ll.int64(0xFFFF))
                for lane in ll.unroll(range(32)):
                    dispatch_relative: ll.uint32 = dispatch_batch + lane * num_ctas
                    while dispatch_relative < num_tok_cur_expert:
                        dispatch_token: ll.uint32 = dA_local_offset + dispatch_relative
                        shuffled_src_token: ll.int32 = ll.__shfl_sync(ll.uint32(0xFFFFFFFF), src_token, lane)
                        shuffled_src_rank: ll.int32 = ll.__shfl_sync(ll.uint32(0xFFFFFFFF), src_rank, lane)
                        src_offset: ll.int64 = ll.int64(shuffled_src_token) * ll.int64(K)
                        load_issuer: ll.uint32 = ll.elect_one()
                        stage_dispatch.bind((cur_expert, dispatch_token))
                        if load_issuer == 1:
                            ri.tma_ld_1d(dispatch_smem, input_ptrs[shuffled_src_rank] + src_offset,
                                         COMM_TILE_K * ELE_BYTES)
                        dispatch_relative = num_tok_cur_expert
                dispatch_batch = dispatch_batch + num_ctas * 32
            dA_local_offset = dA_local_offset + ll.align_up_u32(num_tok_cur_expert, EXPERT_ALIGNMENT)
            cur_expert = cur_expert + 1

    with dispatch_store:
        cur_expert: ll.uint32 = ll.uint32(0)
        dA_local_offset: ll.uint32 = ll.uint32(0)
        while cur_expert < num_experts:
            num_tok_cur_expert: ll.uint32 = ll.uint32(expert_counts[cur_expert])
            dispatch_batch: ll.uint32 = cta_id
            while dispatch_batch < num_tok_cur_expert:
                for lane in ll.unroll(range(32)):
                    dispatch_relative: ll.uint32 = dispatch_batch + lane * num_ctas
                    while dispatch_relative < num_tok_cur_expert:
                        dispatch_token: ll.uint32 = dA_local_offset + dispatch_relative
                        dst_offset: ll.int64 = ll.int64(dispatch_token) * ll.int64(K)
                        store_issuer: ll.uint32 = ll.elect_one()
                        stage_dispatch.bind((cur_expert, dispatch_token))
                        if store_issuer == 1:
                            ri.tma_st_1d(dispatch_smem, dispatch_output + dst_offset, COMM_TILE_K * ELE_BYTES)
                        dispatch_relative = num_tok_cur_expert
                dispatch_batch = dispatch_batch + num_ctas * 32
            ll.tma_store_wait_group(0)
            ll.__syncwarp()
            ll.fence_proxy_async()
            ll.__syncwarp()
            publish_issuer: ll.uint32 = ll.elect_one()
            if publish_issuer == 1:
                old: ll.int32 = ll.atomic_add_acq_rel_gpu(expert_counters + cur_expert, 1)
                if old + 1 == ll.int32(num_ctas):
                    ll.st_release_gpu(expert_counters + cur_expert, 0)
                    ll.st_release_gpu(expert_ready + cur_expert, epoch)

            dA_local_offset = dA_local_offset + ll.align_up_u32(num_tok_cur_expert, EXPERT_ALIGNMENT)
            cur_expert = cur_expert + 1

    with load_ab:
        ab_issuer: ll.uint32 = ll.elect_one()
        tile_ab: ll.uint32 = cluster_id
        while tile_ab < num_tiles:
            coord_cluster_m_ab, coord_cluster_n_ab = get_tile_coord_grouped_m(
                tile_ab,
                num_m_clusters,
                num_n_clusters,
                SCHEDULER_GROUP_M,
            )
            coord_m_ab: ll.uint32 = coord_cluster_m_ab * CLUSTER_M + group_m
            coord_n_ab: ll.uint32 = coord_cluster_n_ab * CLUSTER_N + group_n

            expert_ab: ll.uint32 = ll.uint32(0)
            expert_first_mb_ab: ll.uint32 = ll.uint32(0)
            found_ab: ll.uint32 = ll.uint32(0)
            while expert_ab < num_experts and found_ab == 0:
                expert_rows_ab: ll.uint32 = ll.align_up_u32(ll.uint32(expert_counts[expert_ab]), EXPERT_ALIGNMENT)
                expert_blocks_ab: ll.uint32 = expert_rows_ab // PAIR_TILE_M
                if coord_m_ab < expert_first_mb_ab + expert_blocks_ab:
                    found_ab = ll.uint32(1)
                else:
                    expert_first_mb_ab = expert_first_mb_ab + expert_blocks_ab
                    expert_ab = expert_ab + 1

            if ab_issuer == 1:
                while ll.ld_acquire_gpu(expert_ready + expert_ab) != epoch:
                    pass
            ll.__syncwarp()

            kb_ab: ll.uint32 = ll.uint32(0)
            while kb_ab < num_k_blocks:
                stage_ab.bind(kb_ab)
                if ab_issuer == 1:
                    offset_k_ab: ll.int32 = ll.int32(kb_ab * K_TILE)
                    offset_m_a: ll.int32 = ll.int32(coord_m_ab * PAIR_TILE_M)
                    offset_n_b: ll.int32 = ll.int32(coord_n_ab * PAIR_TILE_N)
                    ri.tma_ld_2d(A_smem, dA, (offset_k_ab, offset_m_a), A_plan, shape=(CTA_TILE_M, K_TILE))
                    ri.tma_ld_3d(
                        B_smem,
                        dB,
                        (offset_k_ab, offset_n_b, ll.int32(expert_ab)),
                        B_plan,
                        shape=(CTA_TILE_N_SHARD, K_TILE),
                    )
                kb_ab = kb_ab + 1
            tile_ab = tile_ab + num_clusters

    with mma:
        mma_issuer: ll.uint32 = ll.elect_one()
        tile_mma: ll.uint32 = cluster_id
        tile_iter_mma: ll.int32 = 0
        while tile_mma < num_tiles:
            stage_acc.bind(tile_iter_mma)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_blocks:
                stage_ab.bind(kb_mma)
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
            tile_iter_mma = tile_iter_mma + 1
            tile_mma = tile_mma + num_clusters

    with epi:
        epi_tid: ll.int32 = tid
        epi_warp: ll.int32 = epi_tid // 32
        tile_epi: ll.uint32 = cluster_id
        tile_iter_epi: ll.int32 = 0

        while tile_epi < num_tiles:
            stage_acc.bind(tile_iter_epi)
            coord_cluster_m_epi, coord_cluster_n_epi = get_tile_coord_grouped_m(
                tile_epi,
                num_m_clusters,
                num_n_clusters,
                SCHEDULER_GROUP_M,
            )
            coord_m_epi: ll.uint32 = coord_cluster_m_epi * CLUSTER_M + group_m
            coord_n_epi: ll.uint32 = coord_cluster_n_epi * CLUSTER_N + group_n
            offset_m_store: ll.int32 = ll.int32(coord_m_epi * PAIR_TILE_M + cta_m * CTA_TILE_M)

            for s in ll.unroll(range(NUM_OUTPUT_STORES)):
                stage_store.bind((tile_iter_epi, s))
                cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                    # Every output 32-column chunk consumes two adjacent
                    # 32-column accumulator chunks: gate then up.
                    pair = s * (ST_TILE_N // SWIGLU_EPI_TILE_N) + (i * ELEMS_PER_BG) // SWIGLU_EPI_TILE_N
                    within_pair = (i * ELEMS_PER_BG) % SWIGLU_EPI_TILE_N
                    gate_col: ll.uint32 = ll.uint32(C_tmem + pair * (2 * SWIGLU_EPI_TILE_N) + within_pair)
                    up_col: ll.uint32 = gate_col + SWIGLU_EPI_TILE_N

                    g_ = ll.empty([8], dtype=ll.uint32, scope="local")
                    u_ = ll.empty([8], dtype=ll.uint32, scope="local")
                    ri.tmem_ld_32dp_32bxn(C_tmem, gate_col, g_, 8)
                    ri.tmem_ld_32dp_32bxn(C_tmem, up_col, u_, 8)
                    ri.tmem_ld_fence()

                    f_ = ll.empty([8], dtype=ll.float32, scope="local")
                    for j in ll.unroll(range(8)):
                        f_[j] = swiglu_f32(ll.uint_as_float(g_[j]), ll.uint_as_float(u_[j]))

                    p0: ll.uint32 = ll.pack_bf16_float(f_[0], f_[1])
                    p1: ll.uint32 = ll.pack_bf16_float(f_[2], f_[3])
                    p2: ll.uint32 = ll.pack_bf16_float(f_[4], f_[5])
                    p3: ll.uint32 = ll.pack_bf16_float(f_[6], f_[7])
                    smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(epi_tid), ll.uint32(i))
                    ri.smem_st_128(C_smem, smem_addr, p0, p1, p2, p3)

                output_issuer: ll.uint32 = ll.elect_one()
                if output_issuer == 1:
                    if epi_warp == 0:
                        offset_n_store: ll.int32 = ll.int32(coord_n_epi * OUTPUT_TILE_N + s * ST_TILE_N)
                        ri.tma_st_2d(C_smem, dD, (offset_n_store, offset_m_store), shape=(ST_TILE_M, ST_TILE_N))

            tile_iter_epi = tile_iter_epi + 1
            tile_epi = tile_epi + num_clusters


class DispatchGroupGemmSwigluFuse:
    """Host launcher for the 4-CTA clustered, 2-CTA UMMA role-fused kernel."""

    MAX_RANKS = 128
    INT32_MAX = (1 << 31) - 1
    NUM_THREADS = NUM_THREADS
    CLUSTER_SHAPE = CLUSTER_SHAPE
    CLUSTER_SIZE = CLUSTER_SIZE
    CLUSTER_M = CLUSTER_M
    CLUSTER_N = CLUSTER_N
    EXPERT_ALIGNMENT = EXPERT_ALIGNMENT
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    PAIR_TILE_M = PAIR_TILE_M
    CTA_TILE_M = CTA_TILE_M
    PAIR_TILE_N = PAIR_TILE_N
    K_TILE = K_TILE
    OUTPUT_TILE_N = OUTPUT_TILE_N
    COMM_TILE_K = COMM_TILE_K
    ST_TILE_M = ST_TILE_M
    ST_TILE_N = ST_TILE_N
    DEFAULT_RESERVED_SMS = DEFAULT_RESERVED_SMS
    SWIGLU_EPI_TILE_N = SWIGLU_EPI_TILE_N
    kernel = dispatch_group_gemm_swiglu_fuse_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "dispatch_group_gemm_swiglu_fuse_sm100_kernel",
            grid=cls.CLUSTER_SHAPE,
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
        input_peers: Sequence[torch.Tensor],
        workspace: torch.Tensor,
        weights_interleaved: torch.Tensor,
        token_src_rank_topk_and_indices: torch.Tensor,
        expert_counts: torch.Tensor,
        num_ctas: int | None = None,
    ) -> "DispatchGroupGemmSwigluFuse":
        import torch
        if not input_peers or len(input_peers) > cls.MAX_RANKS:
            raise ValueError(f"input_peers must contain 1..{cls.MAX_RANKS} tensors")
        if (workspace.ndim != 2 or workspace.dtype != torch.bfloat16 or not workspace.is_cuda
                or not workspace.is_contiguous()):
            raise ValueError("workspace must be a contiguous two-dimensional CUDA BF16 tensor")
        if (weights_interleaved.ndim != 3 or weights_interleaved.dtype != torch.bfloat16
                or not weights_interleaved.is_cuda or not weights_interleaved.is_contiguous()):
            raise ValueError("weights_interleaved must be contiguous CUDA BF16 (E, 2*N_out, K)")
        if (token_src_rank_topk_and_indices.dtype != torch.int64 or not token_src_rank_topk_and_indices.is_cuda
                or not token_src_rank_topk_and_indices.is_contiguous()):
            raise ValueError("token_src_rank_topk_and_indices must be contiguous CUDA int64")
        if (expert_counts.ndim != 1 or expert_counts.dtype != torch.int32 or not expert_counts.is_cuda
                or not expert_counts.is_contiguous()):
            raise ValueError("expert_counts must be a contiguous one-dimensional CUDA int32 tensor")

        m, k = (int(x) for x in workspace.shape)
        num_experts, n_full, weight_k = (int(x) for x in weights_interleaved.shape)
        if weight_k != k:
            raise ValueError(f"weight K={weight_k} does not match workspace K={k}")
        if expert_counts.numel() != num_experts:
            raise ValueError("expert_counts length must match weights_interleaved.shape[0]")
        if (m % cls.EXPERT_ALIGNMENT or n_full % (cls.PAIR_TILE_N * cls.CLUSTER_N) or k % cls.K_TILE):
            raise ValueError(f"M, 2*N_out, and K must be divisible by "
                             f"{cls.EXPERT_ALIGNMENT}, "
                             f"{cls.PAIR_TILE_N * cls.CLUSTER_N}, and {cls.K_TILE}")
        if k != cls.COMM_TILE_K:
            raise ValueError(f"K={k} must equal the compiled full-token dispatch width "
                             f"{cls.COMM_TILE_K}; set LK_DGG_SWIGLU_DISPATCH_K before "
                             f"starting the benchmark")
        n_out = n_full // 2
        if n_out % cls.SWIGLU_EPI_TILE_N:
            raise ValueError(f"N_out must be divisible by {cls.SWIGLU_EPI_TILE_N}")

        counts = [int(x) for x in expert_counts.cpu().tolist()]
        if any(x < 0 for x in counts):
            raise ValueError("expert counts must be non-negative")
        expected_m = sum((x + cls.EXPERT_ALIGNMENT - 1) // cls.EXPERT_ALIGNMENT * cls.EXPERT_ALIGNMENT for x in counts)
        if expected_m != m:
            raise ValueError(f"workspace M={m} does not match padded expert rows {expected_m}")
        if token_src_rank_topk_and_indices.numel() < m:
            raise ValueError("token metadata is shorter than the padded workspace")

        device = workspace.device
        input_peer_refs = tuple(input_peers)
        for peer in input_peer_refs:
            if (peer.ndim != 2 or peer.dtype != torch.bfloat16 or peer.device != device or not peer.is_contiguous()
                    or int(peer.shape[1]) != k):
                raise ValueError("every input peer must be contiguous CUDA BF16 (max_tokens, K)")

        input_ptrs = torch.tensor([peer.data_ptr() for peer in input_peer_refs], dtype=torch.int64, device=device)
        d_a = create_tma_2d_descriptor(
            workspace,
            gmem_inner_dim=k,
            gmem_outer_dim=m,
            smem_inner_dim=cls.K_TILE,
            smem_outer_dim=cls.CTA_TILE_M,
            gmem_outer_stride=k,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )
        d_b = create_tma_3d_descriptor(
            weights_interleaved,
            dim0=k,
            dim1=n_full,
            dim2=num_experts,
            stride1_bytes=k * weights_interleaved.element_size(),
            stride2_bytes=n_full * k * weights_interleaved.element_size(),
            tile_dim0=cls.K_TILE,
            tile_dim1=CTA_TILE_N_SHARD,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )

        out = torch.empty((m, n_out), dtype=workspace.dtype, device=device)
        d_d = create_tma_2d_descriptor(
            out,
            gmem_inner_dim=n_out,
            gmem_outer_dim=m,
            smem_inner_dim=cls.ST_TILE_N,
            smem_outer_dim=cls.ST_TILE_M,
            gmem_outer_stride=n_out,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )

        m_clusters = m // (cls.PAIR_TILE_M * cls.CLUSTER_M)
        n_clusters = n_full // (cls.PAIR_TILE_N * cls.CLUSTER_N)
        work_ctas = m_clusters * n_clusters * cls.CLUSTER_SIZE
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        resident_ctas = compiled_kernel.max_active_clusters() * cls.CLUSTER_SIZE
        if num_ctas is None:
            available_sms = max(cls.CLUSTER_SIZE, sms - cls.DEFAULT_RESERVED_SMS)
            active = min(
                available_sms,
                resident_ctas,
                max(cls.CLUSTER_SIZE, work_ctas),
            )
        else:
            if num_ctas <= 0:
                raise ValueError("num_ctas must be positive")
            active = min(sms, resident_ctas, int(num_ctas))
        active = active // cls.CLUSTER_SIZE * cls.CLUSTER_SIZE
        if active == 0:
            raise ValueError("problem has no active GEMM CTA")

        expert_ready = torch.zeros((num_experts, ), dtype=torch.int32, device=device)
        expert_counters = torch.zeros_like(expert_ready)
        return cls(
            compiled_kernel,
            input_ptrs,
            d_a,
            d_b,
            d_d,
            token_src_rank_topk_and_indices,
            expert_counts,
            expert_ready,
            expert_counters,
            workspace,
            weights_interleaved,
            out,
            input_peer_refs,
            m=m,
            n=n_full,
            k=k,
            num_ctas=active,
        )

    def __init__(
        self,
        compiled_kernel,
        input_ptrs,
        d_a,
        d_b,
        d_d,
        token_metadata,
        expert_counts,
        expert_ready,
        expert_counters,
        workspace,
        weights,
        out,
        input_peer_refs,
        *,
        m,
        n,
        k,
        num_ctas,
    ):
        self.compiled_kernel = compiled_kernel
        self.input_ptrs = input_ptrs
        self.d_a = d_a
        self.d_b = d_b
        self.d_d = d_d
        self.token_metadata = token_metadata
        self.expert_counts = expert_counts
        self.expert_ready = expert_ready
        self.expert_counters = expert_counters
        self.workspace = workspace
        self.weights = weights
        self.out = out
        self.input_peer_refs = tuple(input_peer_refs)
        self.args = (m, n, k, int(expert_counts.numel()))
        self.num_ctas = num_ctas
        self.epoch = 0

    def launch(self, stream=None) -> None:
        self.epoch = self.epoch % self.INT32_MAX + 1
        self.compiled_kernel(
            self.input_ptrs,
            self.workspace,
            self.d_a,
            self.d_b,
            self.d_d,
            self.token_metadata,
            self.expert_counts,
            self.expert_ready,
            self.expert_counters,
            *self.args,
            self.epoch,
            self.num_ctas,
            grid=(self.CLUSTER_SHAPE[0], self.CLUSTER_SHAPE[1], self.num_ctas // self.CLUSTER_SIZE),
            stream=stream,
        )

    def release(self) -> None:
        self.input_ptrs = None
        self.d_a = self.d_b = self.d_d = None
        self.token_metadata = self.expert_counts = None
        self.expert_ready = self.expert_counters = None
        self.workspace = self.weights = self.out = None
        self.input_peer_refs = ()
