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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Fixed-shape SM100 FlashAttention forward kernel.

This benchmark targets the constrained bring-up case:
Q heads = 128, KV heads = 2, head_dim = 128, causal, block_q = block_kv = 128,
one-CTA tcgen05 MMA, TMA load Q/K/V, TMA store O, no varlen and no LSE.

The kernel uses online softmax per Q tile. Each K/V block is loaded once, QK
updates the running row max/sum, P is written to TMEM unnormalized, and
TMEM-source UMMA accumulates P @ V. The final epilogue normalizes O by the row
sum before TMA store.
"""

from __future__ import annotations

import math
import os
import sys
import types
import warnings
from typing import Iterable

import torch

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.language.builtin_base import Builtin, BuiltinOp, builtin
from little_kernel.backends.nv.common.language.intrin.cuda_asm import rcp_f32
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor, create_tma_3d_descriptor

NEG_INF = -math.inf

H_Q = 128
H_KV = 2
KV_GROUP = H_Q // H_KV
HEAD_DIM = 128
BLOCK_M = 128
BLOCK_N = 128
TMA_COLS = 64

NUM_THREADS = 512
NUM_EPI_THREADS = 128
CORR_EPI_THREADS = 160
LOAD_WARP = 14
MMA_WARP = 12
EPI_STORE_WARP = 13
EMPTY_WARP = 15
CORRECTION_WARP_START = 0
CORRECTION_WARP_END = 4
SOFTMAX1_WARP_START = 8
Q_STAGE = 2
V_STAGE = 2
SOFTMAX_TID_START = 128
SOFTMAX_WARP_START = SOFTMAX_TID_START // 32
SOFTMAX_STAGE1_TID_START = 256
SOFTMAX_WARP_END = 12
P_READY_BARRIER_STAGE0 = 5
EPI_BARRIER_STAGE0 = 1
O_READY_BARRIER = 3
UMMA_K = 16
STORE_BN = 64
NUM_STORES = HEAD_DIM // STORE_BN
SWIZZLE_BYTES = 128
BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2

SMEM_Q_TILE_BYTES = BLOCK_M * HEAD_DIM * 2
SMEM_Q_BYTES = Q_STAGE * SMEM_Q_TILE_BYTES
SMEM_K_BYTES = BLOCK_N * HEAD_DIM * 2
SMEM_VT_STAGE_ELEMS = HEAD_DIM * BLOCK_N
SMEM_VT_STAGE_BYTES = SMEM_VT_STAGE_ELEMS * 2
SMEM_VT_BYTES = V_STAGE * SMEM_VT_STAGE_BYTES
SMEM_O_HALF_ELEMS = BLOCK_M * STORE_BN
SMEM_O_HALF_BYTES = SMEM_O_HALF_ELEMS * 2
SMEM_O_STAGE_ELEMS = BLOCK_M * HEAD_DIM
SMEM_O_STAGE_BYTES = SMEM_O_STAGE_ELEMS * 2
SMEM_O_BYTES = Q_STAGE * SMEM_O_STAGE_BYTES
SMEM_TOTAL = SMEM_Q_BYTES + SMEM_K_BYTES + SMEM_VT_BYTES + 4096

TMA_Q_BYTES = SMEM_Q_BYTES
TMA_Q_STAGE_BYTES = SMEM_Q_TILE_BYTES
TMA_K_BYTES = SMEM_K_BYTES
TMA_VT_BYTES = SMEM_VT_STAGE_BYTES
TMA_KV_BYTES = SMEM_K_BYTES + SMEM_VT_BYTES

# UMMA K-major descriptor for a 128x64 bf16 swizzled tile. The second 64-wide
# half is addressed by adding HALF_TILE_16B to the descriptor low bits.
UMMA_SBO_64 = 8 * TMA_COLS * 2
DESC_HI_64 = (((UMMA_SBO_64 >> 4) & 0x3FFF) << 32) | (1 << 46) | (2 << 61)
HALF_TILE_16B = (BLOCK_M * TMA_COLS * 2) >> 4
VT_STAGE_16B = SMEM_VT_STAGE_BYTES >> 4
Q_TILE_16B = (BLOCK_M * HEAD_DIM * 2) >> 4
K_STEP_16B = (UMMA_K * 2) >> 4
IDESC_128 = (1 << 4) | (1 << 7) | (1 << 10) | ((BLOCK_N // 8) << 17) | ((BLOCK_M // 16) << 24)
TMEM_S0_COL = 0
TMEM_S1_COL = 128
TMEM_P0_COL = TMEM_S0_COL + 64
TMEM_P1_COL = TMEM_S1_COL + 64
TMEM_P0_ADDR = TMEM_P0_COL
TMEM_P1_ADDR = TMEM_P1_COL
TMEM_O0_COL = 256
TMEM_O1_COL = 384
TMEM_S_STAGE_STRIDE = TMEM_S1_COL - TMEM_S0_COL
TMEM_P_STAGE_STRIDE = TMEM_P1_COL - TMEM_P0_COL
TMEM_O_STAGE_STRIDE = TMEM_O1_COL - TMEM_O0_COL
TMEM_COLS = 512

passes = PASSES["cuda"]
__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ = 1


@builtin
class tmem_o_to_smem_32x(BuiltinOp):

    @staticmethod
    def return_type(*_):
        return ll.void

    @staticmethod
    def codegen(tcol, smem_base, row, group_base, inv_sum):
        body = """
#include <cuda_bf16.h>
__device__ __forceinline__ uint32_t lk_pack_scaled_bf16_fn(uint32_t lo, uint32_t hi, float inv_sum) {
    auto bf2 = __floats2bfloat162_rn(__uint_as_float(lo) * inv_sum, __uint_as_float(hi) * inv_sum);
    return *reinterpret_cast<uint32_t*>(&bf2);
}

__device__ __forceinline__ void lk_store_o_group_fn(
    uint32_t smem_base, uint32_t row, uint32_t group, uint32_t p0, uint32_t p1, uint32_t p2, uint32_t p3) {
    uint32_t swz = group ^ (row & 7u);
    uint32_t addr = smem_base + row * 128u + swz * 16u;
    asm volatile(
        "st.shared.v4.b32 [%0], {%1,%2,%3,%4};"
        :: "r"(addr), "r"(p0), "r"(p1), "r"(p2), "r"(p3)
        : "memory");
}

__device__ __forceinline__ void lk_tmem_o_to_smem_32x_fn(
    uint32_t tcol, uint32_t smem_base, uint32_t row, uint32_t group_base, float inv_sum) {
    uint32_t r0, r1, r2, r3, r4, r5, r6, r7;
    uint32_t r8, r9, r10, r11, r12, r13, r14, r15;
    uint32_t r16, r17, r18, r19, r20, r21, r22, r23;
    uint32_t r24, r25, r26, r27, r28, r29, r30, r31;
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3),
          "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7),
          "=r"(r8), "=r"(r9), "=r"(r10), "=r"(r11),
          "=r"(r12), "=r"(r13), "=r"(r14), "=r"(r15),
          "=r"(r16), "=r"(r17), "=r"(r18), "=r"(r19),
          "=r"(r20), "=r"(r21), "=r"(r22), "=r"(r23),
          "=r"(r24), "=r"(r25), "=r"(r26), "=r"(r27),
          "=r"(r28), "=r"(r29), "=r"(r30), "=r"(r31)
        : "r"(tcol));
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

    lk_store_o_group_fn(
        smem_base, row, group_base + 0u,
        lk_pack_scaled_bf16_fn(r0, r1, inv_sum),
        lk_pack_scaled_bf16_fn(r2, r3, inv_sum),
        lk_pack_scaled_bf16_fn(r4, r5, inv_sum),
        lk_pack_scaled_bf16_fn(r6, r7, inv_sum));
    lk_store_o_group_fn(
        smem_base, row, group_base + 1u,
        lk_pack_scaled_bf16_fn(r8, r9, inv_sum),
        lk_pack_scaled_bf16_fn(r10, r11, inv_sum),
        lk_pack_scaled_bf16_fn(r12, r13, inv_sum),
        lk_pack_scaled_bf16_fn(r14, r15, inv_sum));
    lk_store_o_group_fn(
        smem_base, row, group_base + 2u,
        lk_pack_scaled_bf16_fn(r16, r17, inv_sum),
        lk_pack_scaled_bf16_fn(r18, r19, inv_sum),
        lk_pack_scaled_bf16_fn(r20, r21, inv_sum),
        lk_pack_scaled_bf16_fn(r22, r23, inv_sum));
    lk_store_o_group_fn(
        smem_base, row, group_base + 3u,
        lk_pack_scaled_bf16_fn(r24, r25, inv_sum),
        lk_pack_scaled_bf16_fn(r26, r27, inv_sum),
        lk_pack_scaled_bf16_fn(r28, r29, inv_sum),
        lk_pack_scaled_bf16_fn(r30, r31, inv_sum));
}
"""
        return Builtin(
            body=body,
            includes=["<cuda_bf16.h>"],
            return_val=f"lk_tmem_o_to_smem_32x_fn({tcol}, {smem_base}, {row}, {group_base}, {inv_sum})",
        )


@lk.ll_kernel(backend="cuda", is_entry=True)
def flash_fwd_sm100_kernel(
    tma_QP: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_K: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_VT: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_OP: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    batch: ll.int32,
    seqlen: ll.int32,
    seqlen_tma: ll.int32,
    m_pack_tma: ll.int32,
    softmax_scale_log2: ll.float32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    smem_q = ll.empty([Q_STAGE * BLOCK_M * HEAD_DIM], dtype=ll.bfloat16, scope="dynamic_shared")
    smem_k = ll.empty([BLOCK_N * HEAD_DIM], dtype=ll.bfloat16, scope="dynamic_shared")
    smem_vt = ll.empty([V_STAGE * HEAD_DIM * BLOCK_N], dtype=ll.bfloat16, scope="dynamic_shared")
    smem_row_sum = ll.empty([Q_STAGE * BLOCK_M], dtype=ll.float32, scope="dynamic_shared")

    q_bar = ll.empty([Q_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    k_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    v_bar = ll.empty([V_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    v_issue_bar = ll.empty([V_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    s_bar = ll.empty([Q_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    p_bar = ll.empty([Q_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    p_split_bar = ll.empty([Q_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    row_sum_bar = ll.empty([Q_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    o_bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    o_stage_bar = ll.empty([Q_STAGE], dtype=ll.uint64, scope="dynamic_shared")
    tmem_addr = ll.empty([1], dtype=ll.uint32, scope="dynamic_shared")

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid >> 5
    local_tid: ll.int32 = tid - SOFTMAX_TID_START
    row_local: ll.int32 = local_tid
    soft_stage: ll.int32 = ll.val_cast(0, ll.int32)
    m_block_thread: ll.int32 = ll.val_cast(0, ll.int32)

    m_pack: ll.int32 = seqlen * KV_GROUP
    num_m_blocks: ll.int32 = (m_pack + BLOCK_M - 1) >> 7
    num_q_stage_blocks: ll.int32 = (num_m_blocks + Q_STAGE - 1) >> 1
    q_stage_linear: ll.int32 = ll.val_cast(ll.blockIdx_x(), ll.int32)
    # Causal LPT scheduling: issue the longest row groups first, matching FA4.
    q_stage_block: ll.int32 = num_q_stage_blocks - 1 - q_stage_linear
    kv_head_idx: ll.int32 = ll.val_cast(ll.blockIdx_y(), ll.int32)
    batch_idx: ll.int32 = ll.val_cast(ll.blockIdx_z(), ll.int32)
    m_block0: ll.int32 = q_stage_block << 1
    m_block1: ll.int32 = m_block0 + 1
    m_block_thread = m_block0
    if tid >= SOFTMAX_STAGE1_TID_START:
        row_local = tid - SOFTMAX_STAGE1_TID_START
        soft_stage = 1
        m_block_thread = m_block1
    batch_offset: ll.int32 = batch_idx * seqlen_tma
    pack_batch_offset: ll.int32 = (batch_idx * H_KV + kv_head_idx) * m_pack_tma

    # Warp-specialized 16-warp layout aligned with FA4:
    #   W0-W3: correction/final O normalization into the Q shared tiles
    #   W4-W7: softmax stage 0
    #   W8-W11: softmax stage 1
    #   W12: QK/PV UMMA issue and TMEM lifetime
    #   W13: O TMA store
    #   W14: Q/K/V TMA load
    #   W15: empty
    if warp_idx == 0:
        el_init: ll.uint32 = ll.elect_one()
        if el_init == 1:
            for qs_bar_init in ll.unroll(range(Q_STAGE)):
                ll.init_smem_barrier(q_bar + qs_bar_init, 1)
            ll.init_smem_barrier(k_bar, 1)
            ll.init_smem_barrier(v_bar, 1)
            ll.init_smem_barrier(v_bar + 1, 1)
            ll.init_smem_barrier(v_issue_bar, 1)
            ll.init_smem_barrier(v_issue_bar + 1, 1)
            ll.init_smem_barrier(s_bar, 1)
            ll.init_smem_barrier(s_bar + 1, 1)
            ll.init_smem_barrier(p_bar, 4)
            ll.init_smem_barrier(p_bar + 1, 4)
            ll.init_smem_barrier(p_split_bar, 4)
            ll.init_smem_barrier(p_split_bar + 1, 4)
            ll.init_smem_barrier(row_sum_bar, 4)
            ll.init_smem_barrier(row_sum_bar + 1, 4)
            ll.init_smem_barrier(o_bar, 1)
            ll.init_smem_barrier(o_stage_bar, 1)
            ll.init_smem_barrier(o_stage_bar + 1, 1)
            ll.fence_smem_barrier_init()
            ll.prefetch_tma_descriptor(tma_QP)
            ll.prefetch_tma_descriptor(tma_K)
            ll.prefetch_tma_descriptor(tma_VT)
            ll.prefetch_tma_descriptor(tma_OP)

    if warp_idx == MMA_WARP:
        ll.tmem_alloc_1sm(tmem_addr, TMEM_COLS)

    ll.__syncthreads()

    q_base_16: ll.uint32 = ll.cvta_generic_to_shared(smem_q) >> 4
    k_base_16: ll.uint32 = ll.cvta_generic_to_shared(smem_k) >> 4
    vt_base_16: ll.uint32 = ll.cvta_generic_to_shared(smem_vt) >> 4
    q_base_bytes: ll.uint32 = ll.cvta_generic_to_shared(smem_q)

    packed_row_idx: ll.int32 = m_block_thread * BLOCK_M + row_local
    row_idx: ll.int32 = packed_row_idx // KV_GROUP
    row_idx_compute: ll.int32 = ll.min_val(row_idx, seqlen - 1)
    row_max: ll.float32 = ll.val_cast(NEG_INF, ll.float32)
    row_sum: ll.float32 = ll.val_cast(0.0, ll.float32)

    num_k_blocks: ll.int32 = (seqlen + BLOCK_N - 1) >> 7
    packed_row_last: ll.int32 = m_block1 * BLOCK_M + BLOCK_M - 1
    row_idx_last: ll.int32 = packed_row_last // KV_GROUP
    row_idx_last = ll.min_val(row_idx_last, seqlen - 1)
    max_k_block: ll.int32 = (row_idx_last + BLOCK_N) >> 7
    if max_k_block > num_k_blocks:
        max_k_block = num_k_blocks

    # FA4-style warp role dispatch (flash_fwd_sm100_gqa.py ~841).
    if warp_idx == EMPTY_WARP:
        pass

    if warp_idx == LOAD_WARP:
        # load() aligned with FA4 flash_fwd_sm100_gqa.py:1057-1076:
        #   load_K(first) -> pipeline_q stage0/1 -> load_V(first) -> K/V prefetch loop.
        if max_k_block > 0:
            n_block_first: ll.int32 = max_k_block - 1
            n_block_base: ll.int32 = n_block_first << 7
            el_load_first: ll.uint32 = ll.elect_one()
            if el_load_first == 1:
                k_row: ll.int32 = batch_offset + n_block_base
                ll.mbarrier_arrive_and_expect_tx(k_bar, TMA_K_BYTES)
                ll.tma_load_3d_1sm(tma_K, k_bar, smem_k, 0, k_row, kv_head_idx)
                ll.tma_load_3d_1sm(tma_K, k_bar, smem_k + BLOCK_N * TMA_COLS, TMA_COLS, k_row, kv_head_idx)

                q_row0: ll.int32 = pack_batch_offset + m_block0 * BLOCK_M
                q_row1: ll.int32 = pack_batch_offset + m_block1 * BLOCK_M
                ll.mbarrier_arrive_and_expect_tx(q_bar + 0, TMA_Q_STAGE_BYTES)
                ll.tma_load_2d_1sm(tma_QP, q_bar + 0, smem_q, 0, q_row0)
                ll.tma_load_2d_1sm(tma_QP, q_bar + 0, smem_q + BLOCK_M * TMA_COLS, TMA_COLS, q_row0)
                ll.mbarrier_arrive_and_expect_tx(q_bar + 1, TMA_Q_STAGE_BYTES)
                ll.tma_load_2d_1sm(tma_QP, q_bar + 1, smem_q + BLOCK_M * HEAD_DIM, 0, q_row1)
                ll.tma_load_2d_1sm(tma_QP, q_bar + 1, smem_q + BLOCK_M * (HEAD_DIM + TMA_COLS), TMA_COLS, q_row1)

                ll.mbarrier_arrive_and_expect_tx(v_bar, TMA_VT_BYTES)
                ll.tma_load_3d_1sm(tma_VT, v_bar, smem_vt, k_row, 0, kv_head_idx)
                ll.tma_load_3d_1sm(tma_VT, v_bar, smem_vt + HEAD_DIM * TMA_COLS, k_row + TMA_COLS, 0, kv_head_idx)
                ll.mbarrier_arrive(v_issue_bar)

            n_block: ll.int32 = n_block_first - 1
            k_iter: ll.int32 = 0
            s_phase: ll.uint32 = ll.uint32(0)
            while n_block >= 0:
                el_load: ll.uint32 = ll.elect_one()
                if el_load == 1:
                    ll.mbarrier_wait(s_bar + 1, s_phase)
                    k_row_next: ll.int32 = batch_offset + (n_block << 7)
                    ll.mbarrier_arrive_and_expect_tx(k_bar, TMA_K_BYTES)
                    ll.tma_load_3d_1sm(tma_K, k_bar, smem_k, 0, k_row_next, kv_head_idx)
                    ll.tma_load_3d_1sm(tma_K, k_bar, smem_k + BLOCK_N * TMA_COLS, TMA_COLS, k_row_next, kv_head_idx)
                    v_prefetch_stage: ll.int32 = (k_iter + 1) & 1
                    if k_iter > 0:
                        v_reuse_phase: ll.uint32 = ll.uint32((k_iter - 1) & 1)
                        ll.mbarrier_wait(o_bar, v_reuse_phase)
                    v_prefetch_smem: ll.ptr[ll.bfloat16] = smem_vt + v_prefetch_stage * SMEM_VT_STAGE_ELEMS
                    v_prefetch_bar: ll.ptr[ll.uint64] = v_bar + v_prefetch_stage
                    ll.mbarrier_arrive_and_expect_tx(v_prefetch_bar, TMA_VT_BYTES)
                    ll.tma_load_3d_1sm(tma_VT, v_prefetch_bar, v_prefetch_smem, k_row_next, 0, kv_head_idx)
                    ll.tma_load_3d_1sm(
                        tma_VT,
                        v_prefetch_bar,
                        v_prefetch_smem + HEAD_DIM * TMA_COLS,
                        k_row_next + TMA_COLS,
                        0,
                        kv_head_idx,
                    )
                    ll.mbarrier_arrive(v_issue_bar + v_prefetch_stage)

                s_phase = s_phase ^ 1
                n_block = n_block - 1
                k_iter = k_iter + 1

    if warp_idx == MMA_WARP:
        # mma(): QK/PV UMMA main loop + final PV tail.
        k_phase: ll.uint32 = ll.uint32(0)
        s_phase: ll.uint32 = ll.uint32(0)
        k_iter: ll.int32 = 0
        n_block: ll.int32 = max_k_block - 1
        while n_block >= 0:
            n_block_base: ll.int32 = n_block << 7
            el_mma: ll.uint32 = ll.elect_one()
            if el_mma == 1:
                if k_iter == 0:
                    ll.mbarrier_wait(k_bar, k_phase)
                    k_phase = k_phase ^ 1
                    ll.tcgen05_fence_after()
                    for qs_init in ll.unroll(range(Q_STAGE)):
                        ll.mbarrier_wait(q_bar + qs_init, ll.uint32(0))
                        q_stage_base_init: ll.uint32 = q_base_16 + ll.uint32(qs_init * Q_TILE_16B)
                        s_col_base_init: ll.uint32 = ll.uint32(TMEM_S0_COL + qs_init * TMEM_S_STAGE_STRIDE)
                        for qk_half_init in ll.unroll(range(2)):
                            q_half_base_init: ll.uint32 = q_stage_base_init + ll.uint32(qk_half_init * HALF_TILE_16B)
                            k_half_base_init: ll.uint32 = k_base_16 + ll.uint32(qk_half_init * HALF_TILE_16B)
                            for ki_init in ll.unroll(range(0, TMA_COLS, UMMA_K)):
                                k_off_init: ll.uint32 = ll.uint32(ki_init // 8)
                                qd_init: ll.uint64 = (ll.uint64(q_half_base_init + k_off_init) & 0x3FFF) | DESC_HI_64
                                kd_init: ll.uint64 = (ll.uint64(k_half_base_init + k_off_init) & 0x3FFF) | DESC_HI_64
                                acc_flag_init: ll.uint32 = ll.uint32(1)
                                if qk_half_init == 0:
                                    if ki_init == 0:
                                        acc_flag_init = ll.uint32(0)
                                ll.umma_f16_ss_1sm(s_col_base_init, qd_init, kd_init, IDESC_128, acc_flag_init)
                        ll.umma_commit_1sm(s_bar + qs_init)
                else:
                    v_read_stage: ll.int32 = (k_iter - 1) & 1
                    v_read_phase: ll.uint32 = ll.uint32(((k_iter - 1) // 2) & 1)
                    ll.mbarrier_wait(v_issue_bar + v_read_stage, v_read_phase)
                    ll.mbarrier_wait(v_bar + v_read_stage, v_read_phase)
                    ll.tcgen05_fence_after()
                    for qs_qk in ll.unroll(range(Q_STAGE)):
                        p_stage_phase: ll.uint32 = ll.uint32((k_iter - 1) & 1)
                        p_read_addr: ll.uint32 = ll.uint32(TMEM_P0_ADDR + qs_qk * TMEM_P_STAGE_STRIDE)
                        o_col_base: ll.uint32 = ll.uint32(TMEM_O0_COL + qs_qk * TMEM_O_STAGE_STRIDE)
                        zero_prev: ll.uint32 = ll.uint32(0)
                        if k_iter == 1:
                            zero_prev = ll.uint32(1)
                        ll.mbarrier_wait(p_split_bar + qs_qk, p_stage_phase)
                        p_high_addr_prev: ll.uint32 = ll.uint32(p_read_addr + (TMA_COLS // 2))
                        vt_stage_base_prev: ll.uint32 = vt_base_16 + ll.uint32(v_read_stage * VT_STAGE_16B)
                        vt_high_base_prev: ll.uint32 = vt_stage_base_prev + ll.uint32(HALF_TILE_16B)
                        for pki_hi_prev in ll.unroll(range(0, TMA_COLS, UMMA_K)):
                            p_tmem_hi_prev: ll.uint32 = ll.uint32(p_high_addr_prev + pki_hi_prev // 2)
                            v_off_hi_prev: ll.uint32 = ll.uint32(pki_hi_prev // 8)
                            vd_hi_prev: ll.uint64 = (ll.uint64(vt_high_base_prev + v_off_hi_prev) & 0x3FFF) | DESC_HI_64
                            pv_hi_acc_prev: ll.uint32 = ll.uint32(1)
                            if pki_hi_prev == 0:
                                if zero_prev == 1:
                                    pv_hi_acc_prev = ll.uint32(0)
                            ll.umma_f16_ts_1sm(o_col_base, p_tmem_hi_prev, vd_hi_prev, IDESC_128, pv_hi_acc_prev)
                        p_low_addr_prev: ll.uint32 = p_read_addr
                        vt_low_base_prev: ll.uint32 = vt_stage_base_prev
                        for pki_lo_split_prev in ll.unroll(range(0, TMA_COLS // 2, UMMA_K)):
                            p_tmem_lo_split_prev: ll.uint32 = ll.uint32(p_low_addr_prev + pki_lo_split_prev // 2)
                            v_off_lo_split_prev: ll.uint32 = ll.uint32(pki_lo_split_prev // 8)
                            vd_lo_split_prev: ll.uint64 = (ll.uint64(vt_low_base_prev + v_off_lo_split_prev)
                                                           & 0x3FFF) | DESC_HI_64
                            ll.umma_f16_ts_1sm(o_col_base, p_tmem_lo_split_prev, vd_lo_split_prev, IDESC_128,
                                               ll.uint32(1))
                        ll.mbarrier_wait(p_bar + qs_qk, p_stage_phase)
                        for pki_lo_prev in ll.unroll(range(TMA_COLS // 2, TMA_COLS, UMMA_K)):
                            p_tmem_lo_prev: ll.uint32 = ll.uint32(p_low_addr_prev + pki_lo_prev // 2)
                            v_off_lo_prev: ll.uint32 = ll.uint32(pki_lo_prev // 8)
                            vd_lo_prev: ll.uint64 = (ll.uint64(vt_low_base_prev + v_off_lo_prev) & 0x3FFF) | DESC_HI_64
                            ll.umma_f16_ts_1sm(o_col_base, p_tmem_lo_prev, vd_lo_prev, IDESC_128, ll.uint32(1))
                        if qs_qk == 1:
                            ll.umma_commit_1sm(o_bar)
                        if qs_qk == 0:
                            ll.mbarrier_wait(k_bar, k_phase)
                            k_phase = k_phase ^ 1
                        ll.tcgen05_fence_after()
                        q_stage_base: ll.uint32 = q_base_16 + ll.uint32(qs_qk * Q_TILE_16B)
                        s_col_base: ll.uint32 = ll.uint32(TMEM_S0_COL + qs_qk * TMEM_S_STAGE_STRIDE)
                        for qk_half in ll.unroll(range(2)):
                            q_half_base: ll.uint32 = q_stage_base + ll.uint32(qk_half * HALF_TILE_16B)
                            k_half_base: ll.uint32 = k_base_16 + ll.uint32(qk_half * HALF_TILE_16B)
                            for ki in ll.unroll(range(0, TMA_COLS, UMMA_K)):
                                k_off: ll.uint32 = ll.uint32(ki // 8)
                                qd: ll.uint64 = (ll.uint64(q_half_base + k_off) & 0x3FFF) | DESC_HI_64
                                kd: ll.uint64 = (ll.uint64(k_half_base + k_off) & 0x3FFF) | DESC_HI_64
                                acc_flag: ll.uint32 = ll.uint32(1)
                                if qk_half == 0:
                                    if ki == 0:
                                        acc_flag = ll.uint32(0)
                                ll.umma_f16_ss_1sm(s_col_base, qd, kd, IDESC_128, acc_flag)
                        ll.umma_commit_1sm(s_bar + qs_qk)

            s_phase = s_phase ^ 1
            n_block = n_block - 1
            k_iter = k_iter + 1

        el_pv_final: ll.uint32 = ll.elect_one()
        if el_pv_final == 1:
            v_final_stage: ll.int32 = (k_iter - 1) & 1
            v_final_phase: ll.uint32 = ll.uint32(((k_iter - 1) // 2) & 1)
            ll.mbarrier_wait(v_issue_bar + v_final_stage, v_final_phase)
            ll.mbarrier_wait(v_bar + v_final_stage, v_final_phase)
            ll.tcgen05_fence_after()
        for qs_final in ll.unroll(range(Q_STAGE)):
            if el_pv_final == 1:
                final_p_phase: ll.uint32 = ll.uint32((max_k_block - 1) & 1)
                p_final_addr: ll.uint32 = ll.uint32(TMEM_P0_ADDR + qs_final * TMEM_P_STAGE_STRIDE)
                o_final_col: ll.uint32 = ll.uint32(TMEM_O0_COL + qs_final * TMEM_O_STAGE_STRIDE)
                zero_final: ll.uint32 = ll.uint32(0)
                if max_k_block == 1:
                    zero_final = ll.uint32(1)
                ll.mbarrier_wait(p_split_bar + qs_final, final_p_phase)
                p_high_addr_final: ll.uint32 = ll.uint32(p_final_addr + (TMA_COLS // 2))
                vt_stage_base_final: ll.uint32 = vt_base_16 + ll.uint32(v_final_stage * VT_STAGE_16B)
                vt_high_base_final: ll.uint32 = vt_stage_base_final + ll.uint32(HALF_TILE_16B)
                for pki_hi_final in ll.unroll(range(0, TMA_COLS, UMMA_K)):
                    p_tmem_hi_final: ll.uint32 = ll.uint32(p_high_addr_final + pki_hi_final // 2)
                    v_off_hi_final: ll.uint32 = ll.uint32(pki_hi_final // 8)
                    vd_hi_final: ll.uint64 = (ll.uint64(vt_high_base_final + v_off_hi_final) & 0x3FFF) | DESC_HI_64
                    pv_hi_acc_final: ll.uint32 = ll.uint32(1)
                    if pki_hi_final == 0:
                        if zero_final == 1:
                            pv_hi_acc_final = ll.uint32(0)
                    ll.umma_f16_ts_1sm(o_final_col, p_tmem_hi_final, vd_hi_final, IDESC_128, pv_hi_acc_final)
                p_low_addr_final: ll.uint32 = p_final_addr
                vt_low_base_final: ll.uint32 = vt_stage_base_final
                for pki_lo_split_final in ll.unroll(range(0, TMA_COLS // 2, UMMA_K)):
                    p_tmem_lo_split_final: ll.uint32 = ll.uint32(p_low_addr_final + pki_lo_split_final // 2)
                    v_off_lo_split_final: ll.uint32 = ll.uint32(pki_lo_split_final // 8)
                    vd_lo_split_final: ll.uint64 = (ll.uint64(vt_low_base_final + v_off_lo_split_final)
                                                    & 0x3FFF) | DESC_HI_64
                    ll.umma_f16_ts_1sm(o_final_col, p_tmem_lo_split_final, vd_lo_split_final, IDESC_128, ll.uint32(1))
                ll.mbarrier_wait(p_bar + qs_final, final_p_phase)
                for pki_lo_final in ll.unroll(range(TMA_COLS // 2, TMA_COLS, UMMA_K)):
                    p_tmem_lo_final: ll.uint32 = ll.uint32(p_low_addr_final + pki_lo_final // 2)
                    v_off_lo_final: ll.uint32 = ll.uint32(pki_lo_final // 8)
                    vd_lo_final: ll.uint64 = (ll.uint64(vt_low_base_final + v_off_lo_final) & 0x3FFF) | DESC_HI_64
                    ll.umma_f16_ts_1sm(o_final_col, p_tmem_lo_final, vd_lo_final, IDESC_128, ll.uint32(1))
                ll.umma_commit_1sm(o_stage_bar + qs_final)
            ll.named_barrier_sync(O_READY_BARRIER + qs_final, CORR_EPI_THREADS)

    # Stage 0 uses W0-W3; stage 1 reuses W8-W11 after softmax completes.

    if warp_idx >= SOFTMAX_WARP_START and warp_idx < SOFTMAX_WARP_END:
        # softmax_loop(): online softmax + row_sum publish.
        s_phase: ll.uint32 = ll.uint32(0)
        k_iter: ll.int32 = 0
        n_block: ll.int32 = max_k_block - 1
        while n_block >= 0:
            n_block_base: ll.int32 = n_block << 7
            ll.mbarrier_wait(s_bar + soft_stage, s_phase)
            ll.tcgen05_fence_after()
            s_base = TMEM_S0_COL + soft_stage * TMEM_S_STAGE_STRIDE
            p_base = TMEM_P0_COL + soft_stage * TMEM_P_STAGE_STRIDE
            p_ready_bar: ll.ptr[ll.uint64] = p_bar + soft_stage
            p_split_ready_bar: ll.ptr[ll.uint64] = p_split_bar + soft_stage
            new_max: ll.float32 = row_max
            if k_iter == 0:
                new_max = ll.val_cast(0.0, ll.float32)
            block_sum: ll.float32 = ll.val_cast(0.0, ll.float32)
            new_max_scaled: ll.float32 = new_max * softmax_scale_log2
            if k_iter == 0:
                for group_hi in ll.unroll(range(7, -1, -1)):
                    tcol_hi: ll.uint32 = ll.uint32(s_base + 64 + group_hi * 8)
                    a0: ll.uint32 = ll.uint32(0)
                    a1: ll.uint32 = ll.uint32(0)
                    a2: ll.uint32 = ll.uint32(0)
                    a3: ll.uint32 = ll.uint32(0)
                    a4: ll.uint32 = ll.uint32(0)
                    a5: ll.uint32 = ll.uint32(0)
                    a6: ll.uint32 = ll.uint32(0)
                    a7: ll.uint32 = ll.uint32(0)
                    ll.tmem_ld_32dp_32bxn(tcol_hi, a0, a1, a2, a3, a4, a5, a6, a7)
                    ll.tmem_ld_fence()
                    pvals_hi = ll.empty([8], dtype=ll.float32, scope="local")
                    pvals_hi[0] = ll.uint_as_float(a0)
                    pvals_hi[1] = ll.uint_as_float(a1)
                    pvals_hi[2] = ll.uint_as_float(a2)
                    pvals_hi[3] = ll.uint_as_float(a3)
                    pvals_hi[4] = ll.uint_as_float(a4)
                    pvals_hi[5] = ll.uint_as_float(a5)
                    pvals_hi[6] = ll.uint_as_float(a6)
                    pvals_hi[7] = ll.uint_as_float(a7)
                    for pe_hi in ll.unroll(range(8)):
                        p_col_hi: ll.int32 = n_block_base + 64 + group_hi * 8 + pe_hi
                        if (p_col_hi >= seqlen) | (p_col_hi > row_idx_compute):
                            pvals_hi[pe_hi] = ll.val_cast(0.0, ll.float32)
                        else:
                            pvals_hi[pe_hi] = ll.exp2f(pvals_hi[pe_hi] * softmax_scale_log2 - new_max_scaled)
                            block_sum = block_sum + pvals_hi[pe_hi]
                    pack_hi0: ll.uint32 = ll.pack_bf16_float(pvals_hi[0], pvals_hi[1])
                    pack_hi1: ll.uint32 = ll.pack_bf16_float(pvals_hi[2], pvals_hi[3])
                    pack_hi2: ll.uint32 = ll.pack_bf16_float(pvals_hi[4], pvals_hi[5])
                    pack_hi3: ll.uint32 = ll.pack_bf16_float(pvals_hi[6], pvals_hi[7])
                    p_tcol_hi: ll.uint32 = ll.uint32(p_base + 32 + group_hi * 4)
                    ll.tmem_st_32dp_32bxn(p_tcol_hi, pack_hi0, pack_hi1, pack_hi2, pack_hi3)
                for group_lo in ll.unroll(range(8)):
                    tcol_lo: ll.uint32 = ll.uint32(s_base + group_lo * 8)
                    b0: ll.uint32 = ll.uint32(0)
                    b1: ll.uint32 = ll.uint32(0)
                    b2: ll.uint32 = ll.uint32(0)
                    b3: ll.uint32 = ll.uint32(0)
                    b4: ll.uint32 = ll.uint32(0)
                    b5: ll.uint32 = ll.uint32(0)
                    b6: ll.uint32 = ll.uint32(0)
                    b7: ll.uint32 = ll.uint32(0)
                    ll.tmem_ld_32dp_32bxn(tcol_lo, b0, b1, b2, b3, b4, b5, b6, b7)
                    ll.tmem_ld_fence()
                    pvals_lo = ll.empty([8], dtype=ll.float32, scope="local")
                    pvals_lo[0] = ll.uint_as_float(b0)
                    pvals_lo[1] = ll.uint_as_float(b1)
                    pvals_lo[2] = ll.uint_as_float(b2)
                    pvals_lo[3] = ll.uint_as_float(b3)
                    pvals_lo[4] = ll.uint_as_float(b4)
                    pvals_lo[5] = ll.uint_as_float(b5)
                    pvals_lo[6] = ll.uint_as_float(b6)
                    pvals_lo[7] = ll.uint_as_float(b7)
                    for pe_lo in ll.unroll(range(8)):
                        p_col_lo: ll.int32 = n_block_base + group_lo * 8 + pe_lo
                        if (p_col_lo >= seqlen) | (p_col_lo > row_idx_compute):
                            pvals_lo[pe_lo] = ll.val_cast(0.0, ll.float32)
                        else:
                            pvals_lo[pe_lo] = ll.exp2f(pvals_lo[pe_lo] * softmax_scale_log2 - new_max_scaled)
                            block_sum = block_sum + pvals_lo[pe_lo]
                    pack_lo0: ll.uint32 = ll.pack_bf16_float(pvals_lo[0], pvals_lo[1])
                    pack_lo1: ll.uint32 = ll.pack_bf16_float(pvals_lo[2], pvals_lo[3])
                    pack_lo2: ll.uint32 = ll.pack_bf16_float(pvals_lo[4], pvals_lo[5])
                    pack_lo3: ll.uint32 = ll.pack_bf16_float(pvals_lo[6], pvals_lo[7])
                    p_tcol_lo: ll.uint32 = ll.uint32(p_base + group_lo * 4)
                    ll.tmem_st_32dp_32bxn(p_tcol_lo, pack_lo0, pack_lo1, pack_lo2, pack_lo3)
                    if group_lo == 3:
                        ll.tmem_st_fence()
                        if (row_local % 32) == 0:
                            ll.mbarrier_arrive(p_split_ready_bar)
            else:
                for pair_fhi in ll.unroll(range(3, -1, -1)):
                    tcol_fhi: ll.uint32 = ll.uint32(s_base + 64 + pair_fhi * 16)
                    af0: ll.uint32 = ll.uint32(0)
                    af1: ll.uint32 = ll.uint32(0)
                    af2: ll.uint32 = ll.uint32(0)
                    af3: ll.uint32 = ll.uint32(0)
                    af4: ll.uint32 = ll.uint32(0)
                    af5: ll.uint32 = ll.uint32(0)
                    af6: ll.uint32 = ll.uint32(0)
                    af7: ll.uint32 = ll.uint32(0)
                    af8: ll.uint32 = ll.uint32(0)
                    af9: ll.uint32 = ll.uint32(0)
                    af10: ll.uint32 = ll.uint32(0)
                    af11: ll.uint32 = ll.uint32(0)
                    af12: ll.uint32 = ll.uint32(0)
                    af13: ll.uint32 = ll.uint32(0)
                    af14: ll.uint32 = ll.uint32(0)
                    af15: ll.uint32 = ll.uint32(0)
                    ll.tmem_ld_32dp_32bxn(tcol_fhi, af0, af1, af2, af3, af4, af5, af6, af7, af8, af9, af10, af11, af12,
                                          af13, af14, af15)
                    ll.tmem_ld_fence()
                    pf0: ll.float32 = ll.exp2f(ll.uint_as_float(af0) * softmax_scale_log2 - new_max_scaled)
                    pf1: ll.float32 = ll.exp2f(ll.uint_as_float(af1) * softmax_scale_log2 - new_max_scaled)
                    pf2: ll.float32 = ll.exp2f(ll.uint_as_float(af2) * softmax_scale_log2 - new_max_scaled)
                    pf3: ll.float32 = ll.exp2f(ll.uint_as_float(af3) * softmax_scale_log2 - new_max_scaled)
                    pf4: ll.float32 = ll.exp2f(ll.uint_as_float(af4) * softmax_scale_log2 - new_max_scaled)
                    pf5: ll.float32 = ll.exp2f(ll.uint_as_float(af5) * softmax_scale_log2 - new_max_scaled)
                    pf6: ll.float32 = ll.exp2f(ll.uint_as_float(af6) * softmax_scale_log2 - new_max_scaled)
                    pf7: ll.float32 = ll.exp2f(ll.uint_as_float(af7) * softmax_scale_log2 - new_max_scaled)
                    pf8: ll.float32 = ll.exp2f(ll.uint_as_float(af8) * softmax_scale_log2 - new_max_scaled)
                    pf9: ll.float32 = ll.exp2f(ll.uint_as_float(af9) * softmax_scale_log2 - new_max_scaled)
                    pf10: ll.float32 = ll.exp2f(ll.uint_as_float(af10) * softmax_scale_log2 - new_max_scaled)
                    pf11: ll.float32 = ll.exp2f(ll.uint_as_float(af11) * softmax_scale_log2 - new_max_scaled)
                    pf12: ll.float32 = ll.exp2f(ll.uint_as_float(af12) * softmax_scale_log2 - new_max_scaled)
                    pf13: ll.float32 = ll.exp2f(ll.uint_as_float(af13) * softmax_scale_log2 - new_max_scaled)
                    pf14: ll.float32 = ll.exp2f(ll.uint_as_float(af14) * softmax_scale_log2 - new_max_scaled)
                    pf15: ll.float32 = ll.exp2f(ll.uint_as_float(af15) * softmax_scale_log2 - new_max_scaled)
                    if pair_fhi == -1:
                        pfp67: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(af6) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(af7) * softmax_scale_log2 - new_max_scaled,
                        )
                        pfp89: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(af8) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(af9) * softmax_scale_log2 - new_max_scaled,
                        )
                        pf6 = ll.uint_as_float(ll.uint32(pfp67))
                        pf7 = ll.uint_as_float(ll.uint32(pfp67 >> 32))
                        pf8 = ll.uint_as_float(ll.uint32(pfp89))
                        pf9 = ll.uint_as_float(ll.uint32(pfp89 >> 32))
                    if pair_fhi == -1:
                        pfp01: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(af0) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(af1) * softmax_scale_log2 - new_max_scaled,
                        )
                        pfp23: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(af2) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(af3) * softmax_scale_log2 - new_max_scaled,
                        )
                        pfp1011: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(af10) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(af11) * softmax_scale_log2 - new_max_scaled,
                        )
                        pfp1213: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(af12) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(af13) * softmax_scale_log2 - new_max_scaled,
                        )
                        pf0 = ll.uint_as_float(ll.uint32(pfp01))
                        pf1 = ll.uint_as_float(ll.uint32(pfp01 >> 32))
                        pf2 = ll.uint_as_float(ll.uint32(pfp23))
                        pf3 = ll.uint_as_float(ll.uint32(pfp23 >> 32))
                        pf10 = ll.uint_as_float(ll.uint32(pfp1011))
                        pf11 = ll.uint_as_float(ll.uint32(pfp1011 >> 32))
                        pf12 = ll.uint_as_float(ll.uint32(pfp1213))
                        pf13 = ll.uint_as_float(ll.uint32(pfp1213 >> 32))
                    block_sum = block_sum + pf0 + pf1 + pf2 + pf3 + pf4 + pf5 + pf6 + pf7
                    block_sum = block_sum + pf8 + pf9 + pf10 + pf11 + pf12 + pf13 + pf14 + pf15
                    pk_fhi0: ll.uint32 = ll.pack_bf16_float(pf0, pf1)
                    pk_fhi1: ll.uint32 = ll.pack_bf16_float(pf2, pf3)
                    pk_fhi2: ll.uint32 = ll.pack_bf16_float(pf4, pf5)
                    pk_fhi3: ll.uint32 = ll.pack_bf16_float(pf6, pf7)
                    pk_fhi4: ll.uint32 = ll.pack_bf16_float(pf8, pf9)
                    pk_fhi5: ll.uint32 = ll.pack_bf16_float(pf10, pf11)
                    pk_fhi6: ll.uint32 = ll.pack_bf16_float(pf12, pf13)
                    pk_fhi7: ll.uint32 = ll.pack_bf16_float(pf14, pf15)
                    p_tcol_fhi: ll.uint32 = ll.uint32(p_base + 32 + pair_fhi * 8)
                    ll.tmem_st_32dp_32bxn(p_tcol_fhi, pk_fhi0, pk_fhi1, pk_fhi2, pk_fhi3, pk_fhi4, pk_fhi5, pk_fhi6,
                                          pk_fhi7)
                for pair_flo in ll.unroll(range(4)):
                    tcol_flo: ll.uint32 = ll.uint32(s_base + pair_flo * 16)
                    cf0: ll.uint32 = ll.uint32(0)
                    cf1: ll.uint32 = ll.uint32(0)
                    cf2: ll.uint32 = ll.uint32(0)
                    cf3: ll.uint32 = ll.uint32(0)
                    cf4: ll.uint32 = ll.uint32(0)
                    cf5: ll.uint32 = ll.uint32(0)
                    cf6: ll.uint32 = ll.uint32(0)
                    cf7: ll.uint32 = ll.uint32(0)
                    cf8: ll.uint32 = ll.uint32(0)
                    cf9: ll.uint32 = ll.uint32(0)
                    cf10: ll.uint32 = ll.uint32(0)
                    cf11: ll.uint32 = ll.uint32(0)
                    cf12: ll.uint32 = ll.uint32(0)
                    cf13: ll.uint32 = ll.uint32(0)
                    cf14: ll.uint32 = ll.uint32(0)
                    cf15: ll.uint32 = ll.uint32(0)
                    ll.tmem_ld_32dp_32bxn(tcol_flo, cf0, cf1, cf2, cf3, cf4, cf5, cf6, cf7, cf8, cf9, cf10, cf11, cf12,
                                          cf13, cf14, cf15)
                    ll.tmem_ld_fence()
                    qf0: ll.float32 = ll.exp2f(ll.uint_as_float(cf0) * softmax_scale_log2 - new_max_scaled)
                    qf1: ll.float32 = ll.exp2f(ll.uint_as_float(cf1) * softmax_scale_log2 - new_max_scaled)
                    qf2: ll.float32 = ll.exp2f(ll.uint_as_float(cf2) * softmax_scale_log2 - new_max_scaled)
                    qf3: ll.float32 = ll.exp2f(ll.uint_as_float(cf3) * softmax_scale_log2 - new_max_scaled)
                    qf4: ll.float32 = ll.exp2f(ll.uint_as_float(cf4) * softmax_scale_log2 - new_max_scaled)
                    qf5: ll.float32 = ll.exp2f(ll.uint_as_float(cf5) * softmax_scale_log2 - new_max_scaled)
                    qf6: ll.float32 = ll.exp2f(ll.uint_as_float(cf6) * softmax_scale_log2 - new_max_scaled)
                    qf7: ll.float32 = ll.exp2f(ll.uint_as_float(cf7) * softmax_scale_log2 - new_max_scaled)
                    qf8: ll.float32 = ll.exp2f(ll.uint_as_float(cf8) * softmax_scale_log2 - new_max_scaled)
                    qf9: ll.float32 = ll.exp2f(ll.uint_as_float(cf9) * softmax_scale_log2 - new_max_scaled)
                    qf10: ll.float32 = ll.exp2f(ll.uint_as_float(cf10) * softmax_scale_log2 - new_max_scaled)
                    qf11: ll.float32 = ll.exp2f(ll.uint_as_float(cf11) * softmax_scale_log2 - new_max_scaled)
                    qf12: ll.float32 = ll.exp2f(ll.uint_as_float(cf12) * softmax_scale_log2 - new_max_scaled)
                    qf13: ll.float32 = ll.exp2f(ll.uint_as_float(cf13) * softmax_scale_log2 - new_max_scaled)
                    qf14: ll.float32 = ll.exp2f(ll.uint_as_float(cf14) * softmax_scale_log2 - new_max_scaled)
                    qf15: ll.float32 = ll.exp2f(ll.uint_as_float(cf15) * softmax_scale_log2 - new_max_scaled)
                    if pair_flo == -1:
                        qfp67: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(cf6) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(cf7) * softmax_scale_log2 - new_max_scaled,
                        )
                        qfp89: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(cf8) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(cf9) * softmax_scale_log2 - new_max_scaled,
                        )
                        qf6 = ll.uint_as_float(ll.uint32(qfp67))
                        qf7 = ll.uint_as_float(ll.uint32(qfp67 >> 32))
                        qf8 = ll.uint_as_float(ll.uint32(qfp89))
                        qf9 = ll.uint_as_float(ll.uint32(qfp89 >> 32))
                    if pair_flo == -1:
                        qfp01: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(cf0) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(cf1) * softmax_scale_log2 - new_max_scaled,
                        )
                        qfp23: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(cf2) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(cf3) * softmax_scale_log2 - new_max_scaled,
                        )
                        qfp1011: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(cf10) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(cf11) * softmax_scale_log2 - new_max_scaled,
                        )
                        qfp1213: ll.uint64 = ll.exp2_emu_f32x2(
                            ll.uint_as_float(cf12) * softmax_scale_log2 - new_max_scaled,
                            ll.uint_as_float(cf13) * softmax_scale_log2 - new_max_scaled,
                        )
                        qf0 = ll.uint_as_float(ll.uint32(qfp01))
                        qf1 = ll.uint_as_float(ll.uint32(qfp01 >> 32))
                        qf2 = ll.uint_as_float(ll.uint32(qfp23))
                        qf3 = ll.uint_as_float(ll.uint32(qfp23 >> 32))
                        qf10 = ll.uint_as_float(ll.uint32(qfp1011))
                        qf11 = ll.uint_as_float(ll.uint32(qfp1011 >> 32))
                        qf12 = ll.uint_as_float(ll.uint32(qfp1213))
                        qf13 = ll.uint_as_float(ll.uint32(qfp1213 >> 32))
                    block_sum = block_sum + qf0 + qf1 + qf2 + qf3 + qf4 + qf5 + qf6 + qf7
                    block_sum = block_sum + qf8 + qf9 + qf10 + qf11 + qf12 + qf13 + qf14 + qf15
                    pk_flo0: ll.uint32 = ll.pack_bf16_float(qf0, qf1)
                    pk_flo1: ll.uint32 = ll.pack_bf16_float(qf2, qf3)
                    pk_flo2: ll.uint32 = ll.pack_bf16_float(qf4, qf5)
                    pk_flo3: ll.uint32 = ll.pack_bf16_float(qf6, qf7)
                    pk_flo4: ll.uint32 = ll.pack_bf16_float(qf8, qf9)
                    pk_flo5: ll.uint32 = ll.pack_bf16_float(qf10, qf11)
                    pk_flo6: ll.uint32 = ll.pack_bf16_float(qf12, qf13)
                    pk_flo7: ll.uint32 = ll.pack_bf16_float(qf14, qf15)
                    p_tcol_flo: ll.uint32 = ll.uint32(p_base + pair_flo * 8)
                    ll.tmem_st_32dp_32bxn(p_tcol_flo, pk_flo0, pk_flo1, pk_flo2, pk_flo3, pk_flo4, pk_flo5, pk_flo6,
                                          pk_flo7)
                    if pair_flo == 1:
                        ll.tmem_st_fence()
                        if (row_local % 32) == 0:
                            ll.mbarrier_arrive(p_split_ready_bar)

            if k_iter == 0:
                row_sum = block_sum
            else:
                row_sum = row_sum + block_sum
            row_max = new_max
            ll.tmem_st_fence()
            if (row_local % 32) == 0:
                ll.mbarrier_arrive(p_ready_bar)

            s_phase = s_phase ^ 1
            n_block = n_block - 1
            k_iter = k_iter + 1
        smem_row_sum[soft_stage * BLOCK_M + row_local] = row_sum
        ll.__syncwarp()
        if (row_local % 32) == 0:
            ll.mbarrier_arrive(row_sum_bar + soft_stage)

    if (warp_idx >= CORRECTION_WARP_START and warp_idx < CORRECTION_WARP_END) or (warp_idx >= SOFTMAX1_WARP_START
                                                                                  and warp_idx < SOFTMAX_WARP_END):
        # correction_loop(): normalize O into shared Q tiles.
        corr_lane: ll.int32 = tid & 31
        corr_stage: ll.int32 = ll.val_cast(0, ll.int32)
        corr_warp: ll.int32 = warp_idx
        if warp_idx >= 8:
            corr_stage = 1
            corr_warp = warp_idx - 8
        corr_row: ll.int32 = (corr_warp << 5) + corr_lane
        corr_row_u: ll.uint32 = ll.uint32(corr_row)
        corr_dp_addr: ll.uint32 = ll.uint32(corr_row - corr_lane) << 16
        ll.mbarrier_wait(o_stage_bar + corr_stage, ll.uint32(0))
        ll.named_barrier_sync(O_READY_BARRIER + corr_stage, CORR_EPI_THREADS)
        ll.tcgen05_fence_after()
        ll.mbarrier_wait(row_sum_bar + corr_stage, ll.uint32(0))
        row_sum_final: ll.float32 = smem_row_sum[corr_stage * BLOCK_M + corr_row]
        inv_sum_final: ll.float32 = ll.val_cast(0.0, ll.float32)
        if row_sum_final != ll.val_cast(0.0, ll.float32):
            inv_sum_final = rcp_f32(row_sum_final)
        o_stage_col: ll.uint32 = ll.uint32(TMEM_O0_COL) + ll.uint32(corr_stage) * ll.uint32(TMEM_O_STAGE_STRIDE)
        o_stage_base_bytes: ll.uint32 = q_base_bytes + ll.uint32(corr_stage) * ll.uint32(SMEM_Q_TILE_BYTES)
        for half_o in ll.unroll(range(NUM_STORES)):
            o_half_base_bytes: ll.uint32 = o_stage_base_bytes + ll.uint32(half_o * SMEM_O_HALF_BYTES)
            for block32 in ll.unroll(range(2)):
                o_tcol: ll.uint32 = o_stage_col + ll.uint32(half_o * STORE_BN + block32 * 32) + corr_dp_addr
                tmem_o_to_smem_32x(
                    o_tcol,
                    o_half_base_bytes,
                    corr_row_u,
                    ll.uint32(block32 * 4),
                    inv_sum_final,
                )
        ll.named_barrier_sync(EPI_BARRIER_STAGE0 + corr_stage, CORR_EPI_THREADS)

    # W13 performs the TMA stores after correction has materialized each stage.
    # W13: epilogue_s2g / TMA store O.
    if warp_idx == EPI_STORE_WARP:
        for qs_store in ll.unroll(range(Q_STAGE)):
            ll.named_barrier_sync(EPI_BARRIER_STAGE0 + qs_store, CORR_EPI_THREADS)
            el_st: ll.uint32 = ll.elect_one()
            if el_st == 1:
                m_store_block: ll.int32 = m_block0 + qs_store
                o_row: ll.int32 = pack_batch_offset + m_store_block * BLOCK_M
                ll.fence_async_shared()
                for half_store in ll.unroll(range(NUM_STORES)):
                    o_col: ll.int32 = half_store * STORE_BN
                    o_stage_smem: ll.ptr[ll.bfloat16] = (smem_q + qs_store * BLOCK_M * HEAD_DIM +
                                                         half_store * SMEM_O_HALF_ELEMS)
                    ll.tma_store_2d(tma_OP, o_stage_smem, o_col, o_row)
                ll.tma_store_commit()
                ll.tma_store_wait_group(0)

    ll.__syncthreads()
    if warp_idx == MMA_WARP:
        ll.tmem_relinquish_alloc_permit_1sm()
        ll.tmem_dealloc_1sm(ll.uint32(0), TMEM_COLS)


def build_kernel(cache_dir: str | None = None):
    if cache_dir is None:
        import tempfile
        cache_dir = os.path.join(tempfile.gettempdir(), 'lk_cache_sm100_flash')
    return flash_fwd_sm100_kernel.build(
        passes,
        codegen_cuda,
        grid=(1, 1, 1),
        block=(NUM_THREADS, 1, 1),
        shared_mem_bytes=SMEM_TOTAL,
        arch="sm_100a",
        cache_dir=cache_dir,
        extra_nvcc_options=["-DNDEBUG", "--use_fast_math", "--maxrregcount=136"],
        disable_var_hoisting=True,
    )


def _flatten_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if q.dim() == 4:
        batch, seqlen, hq, d = q.shape
        _, _, hkv, _ = k.shape
        qf = q.reshape(batch * seqlen, hq, d)
        kf = k.reshape(batch * seqlen, hkv, d)
        vf = v.reshape(batch * seqlen, hkv, d)
        return qf, kf, vf, batch, seqlen
    if q.dim() == 3:
        raise ValueError("flattened q/k/v require explicit batch and seqlen")
    raise ValueError("q/k/v must be [B, T, H, D]")


def _validate_fixed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16
    assert v.dtype == torch.bfloat16
    assert q.shape[-2:] == (H_Q, HEAD_DIM)
    assert k.shape[-2:] == (H_KV, HEAD_DIM)
    assert v.shape[-2:] == (H_KV, HEAD_DIM)
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()


def prepare_flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    _validate_fixed(q, k, v)
    qf, kf, vf, batch, seqlen = _flatten_qkv(q, k, v)
    seqlen_tma = ((seqlen + BLOCK_N - 1) // BLOCK_N) * BLOCK_N
    total_tma = batch * seqlen_tma
    m_pack_tma = seqlen_tma * KV_GROUP

    if seqlen_tma == seqlen:
        q_src = q
        k_tma = kf
        v_tma = vf
    else:
        q_pad = torch.empty((batch, seqlen_tma, H_Q, HEAD_DIM), device=q.device, dtype=q.dtype)
        k_pad = torch.empty((batch, seqlen_tma, H_KV, HEAD_DIM), device=k.device, dtype=k.dtype)
        v_pad = torch.empty((batch, seqlen_tma, H_KV, HEAD_DIM), device=v.device, dtype=v.dtype)
        q_pad[:, :seqlen] = q
        k_pad[:, :seqlen] = k
        v_pad[:, :seqlen] = v
        q_src = q_pad
        k_tma = k_pad.reshape(total_tma, H_KV, HEAD_DIM)
        v_tma = v_pad.reshape(total_tma, H_KV, HEAD_DIM)

    q_pack = torch.empty((batch, H_KV, m_pack_tma, HEAD_DIM), device=q.device, dtype=q.dtype)
    for kvh in range(H_KV):
        h0 = kvh * KV_GROUP
        h1 = h0 + KV_GROUP
        q_pack[:, kvh] = q_src[:, :, h0:h1, :].reshape(batch, m_pack_tma, HEAD_DIM)
    out_pack = torch.empty_like(q_pack)

    # V is loaded as B[D, K] for PV, so make T contiguous in the innermost dim.
    vt = v_tma.view(total_tma, H_KV, HEAD_DIM).permute(1, 2, 0).contiguous()

    tma_q = create_tma_2d_descriptor(
        q_pack.reshape(batch * H_KV * m_pack_tma, HEAD_DIM),
        gmem_inner_dim=HEAD_DIM,
        gmem_outer_dim=batch * H_KV * m_pack_tma,
        smem_inner_dim=TMA_COLS,
        smem_outer_dim=BLOCK_M,
        gmem_outer_stride=HEAD_DIM,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    tma_k = create_tma_3d_descriptor(
        k_tma,
        dim0=HEAD_DIM,
        dim1=total_tma,
        dim2=H_KV,
        stride1_bytes=H_KV * HEAD_DIM * 2,
        stride2_bytes=HEAD_DIM * 2,
        tile_dim0=TMA_COLS,
        tile_dim1=BLOCK_N,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    tma_vt = create_tma_3d_descriptor(
        vt,
        dim0=total_tma,
        dim1=HEAD_DIM,
        dim2=H_KV,
        stride1_bytes=total_tma * 2,
        stride2_bytes=HEAD_DIM * total_tma * 2,
        tile_dim0=TMA_COLS,
        tile_dim1=HEAD_DIM,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    tma_o = create_tma_2d_descriptor(
        out_pack.reshape(batch * H_KV * m_pack_tma, HEAD_DIM),
        gmem_inner_dim=HEAD_DIM,
        gmem_outer_dim=batch * H_KV * m_pack_tma,
        smem_inner_dim=STORE_BN,
        smem_outer_dim=BLOCK_M,
        gmem_outer_stride=HEAD_DIM,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )
    scale_log2 = (1.0 / math.sqrt(HEAD_DIM)) * math.log2(math.e)
    num_m_pack_blocks = (seqlen * KV_GROUP + BLOCK_M - 1) // BLOCK_M
    num_q_stage_blocks = (num_m_pack_blocks + Q_STAGE - 1) // Q_STAGE
    grid = (num_q_stage_blocks, H_KV, batch)
    args = (tma_q, tma_k, tma_vt, tma_o, batch, seqlen, seqlen_tma, m_pack_tma, scale_log2)
    return args, grid, out_pack, batch, seqlen, seqlen_tma


def run_prepared_flash_attn(kernel, prepared):
    args, grid, out_pack, batch, seqlen, seqlen_tma = prepared
    kernel(*args, grid=grid)
    out_view = out_pack.view(batch, H_KV, seqlen_tma, KV_GROUP, HEAD_DIM)[:, :, :seqlen]
    out = torch.empty((batch, seqlen, H_Q, HEAD_DIM), device=out_pack.device, dtype=out_pack.dtype)
    for kvh in range(H_KV):
        h0 = kvh * KV_GROUP
        h1 = h0 + KV_GROUP
        out[:, :, h0:h1, :] = out_view[:, kvh]
    return out.contiguous()


def run_flash_attn(kernel, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    return run_prepared_flash_attn(kernel, prepare_flash_attn(q, k, v))


def torch_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    qf = q.float().transpose(1, 2)
    kf = k.float().transpose(1, 2)
    vf = v.float().transpose(1, 2)
    kf = kf.repeat_interleave(KV_GROUP, dim=1)
    vf = vf.repeat_interleave(KV_GROUP, dim=1)
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * (1.0 / math.sqrt(HEAD_DIM))
    causal = torch.ones((q.shape[1], q.shape[1]), device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal, float("-inf"))
    out = torch.matmul(torch.softmax(scores, dim=-1), vf)
    return out.transpose(1, 2).to(torch.bfloat16).contiguous()


def sdpa_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    import torch.nn.functional as F

    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                         enable_gqa=True)
    return out.transpose(1, 2).contiguous()


def flashattention_cute_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor | None = None,
):
    # use `FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE` to install FA
    # so that we do not need to build FA2/FA3 CUDA source
    # But we need a way to bypass the import of `__init__.py`:
    if "flash_attn.flash_attn_interface" not in sys.modules:
        stub = types.ModuleType("flash_attn.flash_attn_interface")
        for name in (
                "flash_attn_func",
                "flash_attn_varlen_func",
                "flash_attn_qkvpacked_func",
                "flash_attn_varlen_qkvpacked_func",
                "flash_attn_kvpacked_func",
                "flash_attn_varlen_kvpacked_func",
                "flash_attn_with_kvcache",
        ):
            setattr(stub, name, None)
        sys.modules["flash_attn.flash_attn_interface"] = stub

    warn_orig = warnings.warn

    def warn_filtered(message, *args, **kwargs):
        if str(message).startswith("Use explicit `struct.scalar.ptr` for pointer instead"):
            return
        return warn_orig(message, *args, **kwargs)

    try:
        warnings.warn = warn_filtered
        from flash_attn.cute.interface import _flash_attn_fwd

        out_ref, _, _, _ = _flash_attn_fwd(q, k, v, causal=True, pack_gqa=True, return_lse=False, out=out)
    finally:
        warnings.warn = warn_orig
    return out_ref.contiguous()


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    try:
        return flashattention_cute_reference(q, k, v)
    except Exception as exc:
        print(f"FlashAttention CuTe reference unavailable: {exc}; using SDPA")
        return sdpa_reference(q, k, v)


TEST_SHAPES = [(1, 8192), (4, 16384), (8, 527), (17, 1023)]
BENCH_SHAPES = TEST_SHAPES


def correctness(shapes: Iterable[tuple[int, int]] = TEST_SHAPES):
    kernel = build_kernel()
    for batch, seqlen in shapes:
        torch.manual_seed(0)
        q = torch.randn((batch, seqlen, H_Q, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, seqlen, H_KV, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((batch, seqlen, H_KV, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        ref = reference(q, k, v)
        out = run_flash_attn(kernel, q, k, v)
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
        print(f"PASS B={batch} T={seqlen}")


def _time_cuda(fn, iters: int = 20, warmup: int = 3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def benchmark(shapes: Iterable[tuple[int, int]] = BENCH_SHAPES, iters: int = 20):
    inputs = []
    fa4_times = []
    for batch, seqlen in shapes:
        torch.manual_seed(0)
        q = torch.randn((batch, seqlen, H_Q, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((batch, seqlen, H_KV, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        v = torch.randn((batch, seqlen, H_KV, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
        fa4_out = torch.empty_like(q)
        fa4_ms = _time_cuda(lambda: flashattention_cute_reference(q, k, v, out=fa4_out), iters=iters)
        inputs.append((batch, seqlen, q, k, v))
        fa4_times.append(fa4_ms)

    kernel = build_kernel()
    for (batch, seqlen, q, k, v), fa4_ms in zip(inputs, fa4_times):
        prepared = prepare_flash_attn(q, k, v)
        args, grid = prepared[0], prepared[1]
        lk_ms = _time_cuda(lambda: kernel(*args, grid=grid), iters=iters)
        flops = 2.0 * batch * H_Q * seqlen * seqlen * HEAD_DIM
        lk_tflops = flops / (lk_ms / 1e3) / 1e12
        fa4_tflops = flops / (fa4_ms / 1e3) / 1e12
        print(f"B={batch} T={seqlen}: LK {lk_tflops:.1f} TFLOP/s, {lk_ms:.3f} ms; "
              f"FA4 {fa4_tflops:.1f} TFLOP/s, {fa4_ms:.3f} ms; speedup={fa4_ms / lk_ms:.3f}x")


if __name__ == "__main__":
    mode = os.environ.get("LK_SM100_FA_MODE", "correctness")
    if mode == "bench":
        benchmark()
    else:
        correctness()
