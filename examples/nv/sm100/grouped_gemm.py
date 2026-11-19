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
"""Cluster-level persistent grouped GEMM for SM100.

The host builds one work item per ``(expert, 256-row M-block, N-block)``.
Both CTAs consume that item; each CTA owns one 128-row half of the M tile.
"""

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
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor, create_tma_3d_descriptor

# PAIR_TILE_M/PAIR_TILE_N are one cooperative 2-CTA group's logical
# output tile. Each participant owns CTA_TILE_M A rows and
# CTA_TILE_N_SHARD B columns.
PAIR_TILE_M = 256
PAIR_TILE_N = 256
K_TILE = 64
CLUSTER_SIZE = 2
CTA_TILE_M = PAIR_TILE_M // CLUSTER_SIZE
CTA_TILE_N_SHARD = PAIR_TILE_N // CLUSTER_SIZE
MMA_WAVE_M = CTA_TILE_M
NUM_MMA_M_WAVES = CTA_TILE_M // MMA_WAVE_M
EXPERT_ALIGNMENT = PAIR_TILE_M
NUM_A_STAGES = int(os.environ.get("LK_GROUPGEMM_A_STAGES", "6"))
NUM_B_STAGES = int(os.environ.get("LK_GROUPGEMM_B_STAGES", "6"))
NUM_ACC_STAGES = 2
NUM_STORE_STAGES = int(os.environ.get("LK_GROUPGEMM_STORE_STAGES", "2"))
NUM_THREADS = 8 * 32

ELE_BYTES = 2
SMEM_A = CTA_TILE_M * K_TILE * ELE_BYTES
SMEM_B = CTA_TILE_N_SHARD * K_TILE * ELE_BYTES

SWIZZLE_CD_BYTES = 128
ST_TILE_M = MMA_WAVE_M
ST_TILE_N = SWIZZLE_CD_BYTES // ELE_BYTES
NUM_N_STORES = PAIR_TILE_N // ST_TILE_N

BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // ELE_BYTES

SHARED_MEM_BYTES = NUM_STORE_STAGES * ST_TILE_M * SWIZZLE_CD_BYTES + NUM_A_STAGES * SMEM_A + NUM_B_STAGES * SMEM_B + 1024


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend, cluster_shape=(CLUSTER_SIZE, 1, 1),
              cta_group_shape=(2, 1, 1))
def groupgemm_sm100_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    work_experts: ll.ptr[ll.int32],
    work_m_blocks: ll.ptr[ll.int32],
    work_n_blocks: ll.ptr[ll.int32],
    K: ll.uint32,
    num_work: ll.uint32,
    num_clusters: ll.uint32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load_ab = g.role("load_ab", name="load_ab", hw=HW.TMA, warps=[0])
        mma = g.role("mma", name="mma", hw=HW.TensorCore, warps=[1])
        epi = g.role("epi", name="epilogue", hw=HW.CUDACore, warps=[4, 5, 6, 7])

        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=NUM_STORE_STAGES, major="K")
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_A_STAGES, major="K")
        B_smem = g.smem("B_smem", (CTA_TILE_N_SHARD, K_TILE), dtype=ll.bfloat16, stages=NUM_B_STAGES, major="K")
        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32, stages=NUM_ACC_STAGES)

        stage_ab = g.depends(src=load_ab, dst=mma, via=[A_smem, B_smem])
        stage_acc = g.depends(src=mma, dst=epi, via=[C_tmem])
        stage_store = g.depends(src=epi, dst=epi, via=[C_smem])
        A_plan = g.tma_transfer(mcast_group_dim=None, inter_cta_coord_dim=1, inter_cta_stride=CTA_TILE_M)
        B_plan = g.tma_transfer(mcast_group_dim=None, inter_cta_coord_dim=1, inter_cta_stride=CTA_TILE_N_SHARD)

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid // 32
    cta: ll.uint32 = ri.rank_in_group()
    cta_id: ll.uint32 = ll.blockIdx_x()
    cluster_id: ll.uint32 = cta_id // CLUSTER_SIZE
    num_k_blocks: ll.uint32 = K // K_TILE

    if warp_idx == 0 and ll.elect_one():
        ll.prefetch_tma_descriptor(dA)
        ll.prefetch_tma_descriptor(dD)

    with load_ab:
        ab_issuer: ll.uint32 = ll.elect_one()
        work_ab: ll.uint32 = cluster_id
        while work_ab < num_work:
            expert_ab: ll.uint32 = ll.uint32(work_experts[work_ab])
            coord_m_ab: ll.uint32 = ll.uint32(work_m_blocks[work_ab])
            coord_n_ab: ll.uint32 = ll.uint32(work_n_blocks[work_ab])
            kb_ab: ll.uint32 = ll.uint32(0)
            offset_m: ll.int32 = ll.int32(coord_m_ab * PAIR_TILE_M)
            offset_n: ll.int32 = ll.int32(coord_n_ab * PAIR_TILE_N)
            while kb_ab < num_k_blocks:
                stage_ab.bind((work_ab, kb_ab))
                offset_k: ll.int32 = ll.int32(kb_ab * K_TILE)
                if ab_issuer == 1:
                    ri.tma_ld_2d(A_smem, dA, (offset_k, offset_m), A_plan, shape=(CTA_TILE_M, K_TILE))
                    ri.tma_ld_3d(B_smem, dB, (offset_k, offset_n, ll.int32(expert_ab)), B_plan,
                                 shape=(CTA_TILE_N_SHARD, K_TILE))
                kb_ab = kb_ab + 1
            work_ab = work_ab + num_clusters

    with mma:
        mma_issuer: ll.uint32 = ll.elect_one()
        work_mma: ll.uint32 = cluster_id
        work_iter_mma: ll.int32 = 0
        while work_mma < num_work:
            stage_acc.bind(work_iter_mma)
            kb_mma: ll.uint32 = ll.uint32(0)
            while kb_mma < num_k_blocks:
                stage_ab.bind((work_mma, kb_mma))
                if mma_issuer == 1:
                    ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, 0))
                    bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                    ri.umma_f16_ss_2sm_kblock(C_tmem, ad, bd, K_TILE, out_shape=(PAIR_TILE_M, PAIR_TILE_N),
                                              zero_init=kb_mma == 0)
                kb_mma = kb_mma + 1
            work_iter_mma = work_iter_mma + 1
            work_mma = work_mma + num_clusters

    with epi:
        epi_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = epi_tid // 32
        work_epi: ll.uint32 = cluster_id
        work_iter_epi: ll.int32 = 0
        while work_epi < num_work:
            stage_acc.bind(work_iter_epi)
            coord_m_epi: ll.uint32 = ll.uint32(work_m_blocks[work_epi])
            coord_n_epi: ll.uint32 = ll.uint32(work_n_blocks[work_epi])

            offset_m: ll.int32 = ll.int32(coord_m_epi * PAIR_TILE_M + cta * CTA_TILE_M)
            for store in ll.unroll(range(NUM_N_STORES)):
                stage_store.bind((work_iter_epi, store))
                offset_n: ll.int32 = ll.int32(coord_n_epi * PAIR_TILE_N + store * ST_TILE_N)
                cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
                for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                    tmem_col: ll.uint32 = ll.uint32(C_tmem + store * ST_TILE_N + i * ELEMS_PER_BG)
                    smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(epi_tid), ll.uint32(i))
                    ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)

                if epi_warp == 0:
                    store_issuer: ll.uint32 = ll.elect_one()
                    if store_issuer == 1:
                        ri.tma_st_2d(C_smem, dD, (offset_n, offset_m), shape=(ST_TILE_M, ST_TILE_N))

            work_iter_epi = work_iter_epi + 1
            work_epi = work_epi + num_clusters


class GroupGemmV2:
    """Launcher for the cluster-level SM100 grouped GEMM."""

    PAIR_TILE_M = PAIR_TILE_M
    PAIR_TILE_N = PAIR_TILE_N
    K_TILE = K_TILE
    EXPERT_ALIGNMENT = EXPERT_ALIGNMENT
    CLUSTER_SIZE = CLUSTER_SIZE
    NUM_THREADS = NUM_THREADS
    SHARED_MEM_BYTES = SHARED_MEM_BYTES
    kernel = groupgemm_sm100_kernel

    @classmethod
    def build(cls, *, cache_dir=None, verbose=False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")
        return CompiledKernel(
            compile_dataflow_cuda(source, num_threads=cls.NUM_THREADS),
            "groupgemm_sm100_kernel",
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
        inputs: torch.Tensor,
        weights: torch.Tensor,
        expert_counts: Sequence[int] | torch.Tensor,
        num_clusters: int | None = None,
    ) -> "GroupGemmV2":
        import torch
        if inputs.ndim != 2 or inputs.dtype != torch.bfloat16 or not inputs.is_cuda or not inputs.is_contiguous():
            raise ValueError("inputs must be contiguous CUDA BF16 (sum_M, K)")
        if weights.ndim != 3 or weights.dtype != torch.bfloat16 or not weights.is_cuda or not weights.is_contiguous():
            raise ValueError("weights must be contiguous CUDA BF16 (E, N, K)")
        if isinstance(expert_counts, torch.Tensor):
            counts = [int(x) for x in expert_counts.cpu().tolist()]
        else:
            counts = [int(x) for x in expert_counts]

        num_experts, n, weight_k = (int(x) for x in weights.shape)
        compact_m, k = (int(x) for x in inputs.shape)
        if len(counts) != num_experts or sum(counts) != compact_m:
            raise ValueError("expert_counts must match weights and sum to inputs.shape[0]")
        if any(count < 0 for count in counts):
            raise ValueError("expert counts must be non-negative")
        if weight_k != k:
            raise ValueError("inputs and weights must have the same K")
        if n % cls.PAIR_TILE_N or k % cls.K_TILE:
            raise ValueError(f"N and K must be divisible by {cls.PAIR_TILE_N} and {cls.K_TILE}")

        compact_offsets = [0]
        padded_offsets = [0]
        for count in counts:
            compact_offsets.append(compact_offsets[-1] + count)
            padded_offsets.append(padded_offsets[-1] +
                                  (count + cls.EXPERT_ALIGNMENT - 1) // cls.EXPERT_ALIGNMENT * cls.EXPERT_ALIGNMENT)
        padded_m = padded_offsets[-1]
        if padded_m == 0:
            raise ValueError("grouped GEMM requires at least one row")

        device = inputs.device
        a_padded = torch.zeros((padded_m, k), dtype=inputs.dtype, device=device)
        for expert, count in enumerate(counts):
            if count:
                a_padded[padded_offsets[expert]:padded_offsets[expert] + count].copy_(
                    inputs[compact_offsets[expert]:compact_offsets[expert + 1]])

        work_experts: list[int] = []
        work_m_blocks: list[int] = []
        work_n_blocks: list[int] = []
        num_n_blocks = n // cls.PAIR_TILE_N
        for expert, count in enumerate(counts):
            padded_rows = (count + cls.EXPERT_ALIGNMENT - 1) // cls.EXPERT_ALIGNMENT * cls.EXPERT_ALIGNMENT
            first_m_block = padded_offsets[expert] // cls.PAIR_TILE_M
            num_m_blocks = padded_rows // cls.PAIR_TILE_M
            for m_block in range(num_m_blocks):
                for nb in range(num_n_blocks):
                    work_experts.append(expert)
                    work_m_blocks.append(first_m_block + m_block)
                    work_n_blocks.append(nb)
        num_work = len(work_experts)

        work_experts_t = torch.tensor(work_experts, dtype=torch.int32, device=device)
        work_m_blocks_t = torch.tensor(work_m_blocks, dtype=torch.int32, device=device)
        work_n_blocks_t = torch.tensor(work_n_blocks, dtype=torch.int32, device=device)

        d_a = create_tma_2d_descriptor(
            a_padded,
            gmem_inner_dim=k,
            gmem_outer_dim=padded_m,
            smem_inner_dim=cls.K_TILE,
            smem_outer_dim=CTA_TILE_M,
            gmem_outer_stride=k,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )
        d_b = create_tma_3d_descriptor(
            weights,
            dim0=k,
            dim1=n,
            dim2=num_experts,
            stride1_bytes=k * weights.element_size(),
            stride2_bytes=n * k * weights.element_size(),
            tile_dim0=cls.K_TILE,
            tile_dim1=CTA_TILE_N_SHARD,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )
        output = torch.empty((padded_m, n), dtype=inputs.dtype, device=device)
        d_d = create_tma_2d_descriptor(
            output,
            gmem_inner_dim=n,
            gmem_outer_dim=padded_m,
            smem_inner_dim=ST_TILE_N,
            smem_outer_dim=ST_TILE_M,
            gmem_outer_stride=n,
            swizzle_mode=128,
            oob_fill=False,
            l2_promotion=3,
        )

        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_clusters = max(1, sms // cls.CLUSTER_SIZE)
        active_clusters = (min(max_clusters, num_work) if num_clusters is None else min(
            max_clusters, num_work, int(num_clusters)))
        if active_clusters <= 0:
            raise ValueError("num_clusters must be positive")

        return cls(
            compiled_kernel,
            d_a,
            d_b,
            d_d,
            work_experts_t,
            work_m_blocks_t,
            work_n_blocks_t,
            a_padded,
            weights,
            output,
            tuple(counts),
            tuple(compact_offsets),
            tuple(padded_offsets),
            k=k,
            num_work=num_work,
            num_clusters=active_clusters,
        )

    def __init__(
        self,
        compiled_kernel,
        d_a,
        d_b,
        d_d,
        work_experts,
        work_m_blocks,
        work_n_blocks,
        a_padded,
        weights,
        output,
        expert_counts,
        compact_offsets,
        padded_offsets,
        *,
        k,
        num_work,
        num_clusters,
    ):
        self.compiled_kernel = compiled_kernel
        self.d_a = d_a
        self.d_b = d_b
        self.d_d = d_d
        self.work_experts = work_experts
        self.work_m_blocks = work_m_blocks
        self.work_n_blocks = work_n_blocks
        self.a_padded = a_padded
        self.weights = weights
        self.output = output
        self.expert_counts = expert_counts
        self.compact_offsets = compact_offsets
        self.padded_offsets = padded_offsets
        self.k = k
        self.num_work = num_work
        self.num_clusters = num_clusters
        self.num_ctas = num_clusters * self.CLUSTER_SIZE

    def copy_inputs(self, inputs: torch.Tensor) -> None:
        import torch
        expected = (self.compact_offsets[-1], self.k)
        if (tuple(inputs.shape) != expected or inputs.dtype != self.a_padded.dtype
                or inputs.device != self.a_padded.device or not inputs.is_contiguous()):
            raise ValueError(
                f"inputs must remain contiguous {expected} on {self.a_padded.device} with dtype {self.a_padded.dtype}")
        for expert, count in enumerate(self.expert_counts):
            if count:
                self.a_padded[self.padded_offsets[expert]:self.padded_offsets[expert] + count].copy_(
                    inputs[self.compact_offsets[expert]:self.compact_offsets[expert + 1]])

    def launch(self, stream=None) -> None:
        self.compiled_kernel(
            self.d_a,
            self.d_b,
            self.d_d,
            self.work_experts,
            self.work_m_blocks,
            self.work_n_blocks,
            self.k,
            self.num_work,
            self.num_clusters,
            grid=(self.num_ctas, 1, 1),
            stream=stream,
        )

    def valid_output(self) -> torch.Tensor:
        import torch
        parts = [
            self.output[self.padded_offsets[expert]:self.padded_offsets[expert] + count]
            for expert, count in enumerate(self.expert_counts)
            if count
        ]
        return torch.cat(parts, dim=0)

    def release(self) -> None:
        self.d_a = self.d_b = self.d_d = None
        self.work_experts = self.work_m_blocks = None
        self.work_n_blocks = None
        self.a_padded = self.weights = self.output = None
