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
"""Role/dataflow skeleton for the fixed-shape SM100 FA4 GQA forward kernel.

This file is intentionally compiler-driven: it declares roles, logical buffers,
data channels, stage regions, and logical memory intrinsics. It must not
hand-write pipeline barriers; ``pass_auto_pipe`` owns those insertions.
"""

from __future__ import annotations
from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.builtin_base import inline_func
from little_kernel.backends.nv.common.language.intrin.cuda_asm import rcp_f32
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

NEG_INF_BITS = 0xFF800000

H_Q = 128
H_KV = 2
KV_GROUP = H_Q // H_KV
HEAD_DIM = 128
BM = 128
BN = 128
TMA_COLS = 64
Q_STAGE = 2
KV_STAGE = 2
EPI_N = 64

NUM_THREADS = 512
NUM_EPI_THREADS = 128
CORR_EPI_THREADS = 160

QK_WARP = 12
EPI_STORE_WARP = 13
LOAD_WARP = 14
PV_WARP = 15
CORRECTION_WARP_START = 0
CORRECTION_WARP_END = 4
SOFTMAX_TID_START = 128
SOFTMAX_WARP_START = SOFTMAX_TID_START // 32
SOFTMAX_STAGE1_TID_START = 256
SOFTMAX_WARP_END = 12
CORRECTION_WARPS = [0, 1, 2, 3]
SOFTMAX0_WARPS = [4, 5, 6, 7]
SOFTMAX1_WARPS = [8, 9, 10, 11]
QK_WARPS = [QK_WARP]
EPI_STORE_WARPS = [EPI_STORE_WARP]
LOAD_WARPS = [LOAD_WARP]
PV_WARPS = [PV_WARP]

STORE_BN = 64
NUM_STORES = HEAD_DIM // STORE_BN
SWIZZLE_BYTES = 128
BANK_GROUP_BYTES = 16
ELEMS_PER_BG = BANK_GROUP_BYTES // 2

SMEM_Q_TILE_BYTES = BM * HEAD_DIM * 2
SMEM_Q_BYTES = Q_STAGE * SMEM_Q_TILE_BYTES
SMEM_K_BYTES = BN * HEAD_DIM * 2
SMEM_V_STAGE_ELEMS = HEAD_DIM * BN
SMEM_V_STAGE_BYTES = SMEM_V_STAGE_ELEMS * 2
SMEM_V_BYTES = KV_STAGE * SMEM_V_STAGE_BYTES
SMEM_ROW_SUM_BYTES = Q_STAGE * BM * 4
SMEM_O_HALF_ELEMS = BM * STORE_BN
SMEM_O_HALF_BYTES = SMEM_O_HALF_ELEMS * 2
SMEM_O_STAGE_ELEMS = BM * HEAD_DIM
SMEM_O_STAGE_BYTES = SMEM_O_STAGE_ELEMS * 2
SMEM_TOTAL = SMEM_Q_BYTES + SMEM_K_BYTES + SMEM_V_BYTES + 4096

TMEM_S0_COL = 0
TMEM_S1_COL = 128
TMEM_P0_COL = TMEM_S0_COL + 64
TMEM_P1_COL = TMEM_S1_COL + 64
TMEM_O0_COL = 256
TMEM_O1_COL = 384
TMEM_S_STAGE_STRIDE = TMEM_S1_COL - TMEM_S0_COL
TMEM_P_STAGE_STRIDE = TMEM_P1_COL - TMEM_P0_COL
TMEM_O_STAGE_STRIDE = TMEM_O1_COL - TMEM_O0_COL
TMEM_COLS = 512


@inline_func
def softmax_fn(
    tid,
    tid_start,
    m_block,
    seqlen,
    max_k_block,
    sfm_scale_log2,
    score,
    prob_hi,
    prob_lo_first,
    prob_lo_rest,
    row_sum_smem,
    stage_score,
    stage_prob_split,
    stage_prob_lo_rest,
    stage_sum,
) -> ll.void:
    local_row: ll.int32 = tid - tid_start
    packed_row: ll.int32 = m_block * BM + local_row
    row_idx: ll.int32 = packed_row // KV_GROUP
    causal_row: ll.int32 = ll.min_val(row_idx, seqlen - 1)
    row_sum: ll.float32 = ll.float32(0.0)
    key_iteration: ll.int32 = 0
    key_block: ll.int32 = max_k_block - 1
    while key_block >= 0:
        key_block_start: ll.int32 = key_block << 7
        stage_score.bind(key_iteration)
        stage_prob_split.bind(key_iteration)
        stage_prob_lo_rest.bind(key_iteration)
        block_sum: ll.float32 = ll.float32(0.0)
        if key_iteration == 0:
            # The newest key block may cross the causal boundary.
            valid_columns: ll.int32 = causal_row - key_block_start + 1
            # Walk the score/probability overlay backward to avoid clobbering unread scores.
            for chunk_hi in ll.unroll(range(1, -1, -1)):
                chunk_start_hi: ll.int32 = 64 + chunk_hi * 32
                if valid_columns > chunk_start_hi:
                    mask_hi: ll.uint32 = ll.prefix_lsb_mask_i32(valid_columns - chunk_start_hi)
                    for half_hi_masked in ll.unroll(range(1, -1, -1)):
                        scores_hi_masked = ll.empty([16], dtype=ll.uint32, scope="local")
                        probs_hi_masked = ll.empty([16], dtype=ll.float32, scope="local")
                        packed_hi_masked = ll.empty([8], dtype=ll.uint32, scope="local")
                        score_col_hi_masked: ll.uint32 = ll.uint32(score + chunk_start_hi + half_hi_masked * 16)
                        ri.tmem_ld_32dp_32bxn(score, score_col_hi_masked, scores_hi_masked, 16)
                        ri.tmem_ld_fence()
                        for i in ll.unroll(range(16)):
                            valid_hi: ll.uint32 = mask_hi & (ll.uint32(1) << (half_hi_masked * 16 + i))
                            score_hi: ll.uint32 = ll.selp_b32(valid_hi, scores_hi_masked[i], ll.uint32(NEG_INF_BITS))
                            probs_hi_masked[i] = ll.exp2f(ll.uint_as_float(score_hi) * sfm_scale_log2)
                        block_sum += ll.sum(probs_hi_masked)
                        for i in ll.unroll(range(8)):
                            packed_hi_masked[i] = ll.pack_bf16_float(probs_hi_masked[i * 2], probs_hi_masked[i * 2 + 1])
                        ri.tmem_st_32dp_32bxn(prob_hi, ll.uint32(prob_hi + chunk_hi * 16 + half_hi_masked * 8),
                                              packed_hi_masked, 8)
                else:
                    z_ = ll.zeros([8], dtype=ll.uint32, scope="local")
                    for half_hi_masked in ll.unroll(range(2)):
                        ri.tmem_st_32dp_32bxn(prob_hi, ll.uint32(prob_hi + chunk_hi * 16 + half_hi_masked * 8), z_, 8)
            mask_lo0: ll.uint32 = ll.prefix_lsb_mask_i32(valid_columns)
            if valid_columns > 0:
                for half_lo0 in ll.unroll(range(2)):
                    scores_lo0 = ll.empty([16], dtype=ll.uint32, scope="local")
                    probs_lo0 = ll.empty([16], dtype=ll.float32, scope="local")
                    packed_lo0 = ll.empty([8], dtype=ll.uint32, scope="local")
                    score_col_lo0: ll.uint32 = ll.uint32(score + half_lo0 * 16)
                    ri.tmem_ld_32dp_32bxn(score, score_col_lo0, scores_lo0, 16)
                    ri.tmem_ld_fence()
                    for i in ll.unroll(range(16)):
                        valid_lo0: ll.uint32 = mask_lo0 & (ll.uint32(1) << (half_lo0 * 16 + i))
                        score_lo0: ll.uint32 = ll.selp_b32(valid_lo0, scores_lo0[i], ll.uint32(NEG_INF_BITS))
                        probs_lo0[i] = ll.exp2f(ll.uint_as_float(score_lo0) * sfm_scale_log2)
                    block_sum += ll.sum(probs_lo0)
                    for i in ll.unroll(range(8)):
                        packed_lo0[i] = ll.pack_bf16_float(probs_lo0[i * 2], probs_lo0[i * 2 + 1])
                    ri.tmem_st_32dp_32bxn(prob_lo_first, ll.uint32(prob_lo_first + half_lo0 * 8), packed_lo0, 8)
            else:
                z0 = ll.zeros([8], dtype=ll.uint32, scope="local")
                for half_lo0 in ll.unroll(range(2)):
                    ri.tmem_st_32dp_32bxn(prob_lo_first, ll.uint32(prob_lo_first + half_lo0 * 8), z0, 8)

            mask_lo1: ll.uint32 = ll.prefix_lsb_mask_i32(valid_columns - 32)
            if valid_columns > 32:
                for half_lo1 in ll.unroll(range(2)):
                    scores_lo1 = ll.empty([16], dtype=ll.uint32, scope="local")
                    probs_lo1 = ll.empty([16], dtype=ll.float32, scope="local")
                    packed_lo1 = ll.empty([8], dtype=ll.uint32, scope="local")
                    score_col_lo1: ll.uint32 = ll.uint32(score + 32 + half_lo1 * 16)
                    ri.tmem_ld_32dp_32bxn(score, score_col_lo1, scores_lo1, 16)
                    ri.tmem_ld_fence()
                    for i in ll.unroll(range(16)):
                        valid_lo1: ll.uint32 = mask_lo1 & (ll.uint32(1) << (half_lo1 * 16 + i))
                        score_lo1: ll.uint32 = ll.selp_b32(valid_lo1, scores_lo1[i], ll.uint32(NEG_INF_BITS))
                        probs_lo1[i] = ll.exp2f(ll.uint_as_float(score_lo1) * sfm_scale_log2)
                    block_sum += ll.sum(probs_lo1)
                    for i in ll.unroll(range(8)):
                        packed_lo1[i] = ll.pack_bf16_float(probs_lo1[i * 2], probs_lo1[i * 2 + 1])
                    ri.tmem_st_32dp_32bxn(prob_lo_rest, ll.uint32(prob_lo_rest + half_lo1 * 8), packed_lo1, 8)
            else:
                z1 = ll.zeros([8], dtype=ll.uint32, scope="local")
                for half_lo1 in ll.unroll(range(2)):
                    ri.tmem_st_32dp_32bxn(prob_lo_rest, ll.uint32(prob_lo_rest + half_lo1 * 8), z1, 8)
        else:
            # Older key blocks are entirely visible and need no causal mask.
            for slice_hi in ll.unroll(range(3, -1, -1)):
                score_col_hi: ll.uint32 = ll.uint32(score + 64 + slice_hi * 16)
                scores_hi = ll.empty([16], dtype=ll.uint32, scope="local")
                probs_hi = ll.empty([16], dtype=ll.float32, scope="local")
                packed_hi = ll.empty([8], dtype=ll.uint32, scope="local")
                ri.tmem_ld_32dp_32bxn(score, score_col_hi, scores_hi, 16)
                ri.tmem_ld_fence()
                for i in ll.unroll(range(16)):
                    probs_hi[i] = ll.exp2f(ll.uint_as_float(scores_hi[i]) * sfm_scale_log2)
                block_sum += ll.sum(probs_hi)
                for i in ll.unroll(range(8)):
                    packed_hi[i] = ll.pack_bf16_float(probs_hi[i * 2], probs_hi[i * 2 + 1])
                ri.tmem_st_32dp_32bxn(prob_hi, ll.uint32(prob_hi + slice_hi * 8), packed_hi, 8)

            for slice_lo_first in ll.unroll(range(2)):
                score_col_lo_first: ll.uint32 = ll.uint32(score + slice_lo_first * 16)
                scores_lo_first = ll.empty([16], dtype=ll.uint32, scope="local")
                probs_lo_first = ll.empty([16], dtype=ll.float32, scope="local")
                packed_lo_first = ll.empty([8], dtype=ll.uint32, scope="local")
                ri.tmem_ld_32dp_32bxn(score, score_col_lo_first, scores_lo_first, 16)
                ri.tmem_ld_fence()
                for i in ll.unroll(range(16)):
                    probs_lo_first[i] = ll.exp2f(ll.uint_as_float(scores_lo_first[i]) * sfm_scale_log2)
                block_sum += ll.sum(probs_lo_first)
                for i in ll.unroll(range(8)):
                    packed_lo_first[i] = ll.pack_bf16_float(probs_lo_first[i * 2], probs_lo_first[i * 2 + 1])
                ri.tmem_st_32dp_32bxn(prob_lo_first, ll.uint32(prob_lo_first + slice_lo_first * 8), packed_lo_first, 8)

            for slice_lo_rest in ll.unroll(range(2, 4)):
                score_col_lo_rest: ll.uint32 = ll.uint32(score + slice_lo_rest * 16)
                scores_lo_rest = ll.empty([16], dtype=ll.uint32, scope="local")
                probs_lo_rest = ll.empty([16], dtype=ll.float32, scope="local")
                packed_lo_rest = ll.empty([8], dtype=ll.uint32, scope="local")
                ri.tmem_ld_32dp_32bxn(score, score_col_lo_rest, scores_lo_rest, 16)
                ri.tmem_ld_fence()
                for i in ll.unroll(range(16)):
                    probs_lo_rest[i] = ll.exp2f(ll.uint_as_float(scores_lo_rest[i]) * sfm_scale_log2)
                block_sum += ll.sum(probs_lo_rest)
                for i in ll.unroll(range(8)):
                    packed_lo_rest[i] = ll.pack_bf16_float(probs_lo_rest[i * 2], probs_lo_rest[i * 2 + 1])
                ri.tmem_st_32dp_32bxn(prob_lo_rest, ll.uint32(prob_lo_rest + (slice_lo_rest - 2) * 8), packed_lo_rest,
                                      8)

        row_sum = row_sum + block_sum
        key_iteration = key_iteration + 1
        key_block = key_block - 1
    stage_sum.bind(0)
    ri.smem_st_f32(row_sum_smem, local_row, row_sum)


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def fa4_sm100_kernel(
    tma_QP: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_K: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_V: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    tma_OP: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    batch: ll.int32,
    seqlen: ll.int32,
    seqlen_tma: ll.int32,
    m_pack_tma: ll.int32,
    softmax_scale_log2: ll.float32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as g:
        load_q = g.role("load_q", name="tma.q", hw=HW.TMA, warps=LOAD_WARPS)
        load_kv = g.role("load_kv", name="tma.kv", hw=HW.TMA, warps=LOAD_WARPS)
        qk_role = g.role("qk_role", name="mma.qk", hw=HW.TensorCore, warps=QK_WARPS)
        pv_role = g.role("pv_role", name="mma.pv", hw=HW.TensorCore, warps=PV_WARPS)
        sfm0 = g.role("sfm0", name="softmax0", hw=HW.CUDACore, warps=SOFTMAX0_WARPS)
        sfm1 = g.role("sfm1", name="softmax1", hw=HW.CUDACore, warps=SOFTMAX1_WARPS)
        correction = g.role("correction", name="correction", hw=HW.CUDACore, warps=CORRECTION_WARPS)
        store = g.role("store", name="epilogue.store", hw=HW.TMA, warps=EPI_STORE_WARPS)

        sQ = [g.smem(f"sQ{s}", (BM, HEAD_DIM), dtype=ll.bfloat16, major="K") for s in range(Q_STAGE)]
        sK = g.smem("sK", (BN, HEAD_DIM), dtype=ll.bfloat16, major="K")
        sV = g.smem("sV", (HEAD_DIM, BN), dtype=ll.bfloat16, stages=KV_STAGE, major="MN")
        sO = [g.smem(f"sO{s}", (BM, HEAD_DIM), dtype=ll.bfloat16, major="K") for s in range(Q_STAGE)]
        sRowSum = [g.smem(f"sRowSum{s}", (BM, ), dtype=ll.float32) for s in range(Q_STAGE)]
        tS = [g.tmem(f"score{s}", (BM, BN), dtype=ll.float32, stages=1) for s in range(Q_STAGE)]
        tP_hi = [g.tmem(f"prob{s}_hi", (BM, BN // 2), dtype=ll.bfloat16, stages=1) for s in range(Q_STAGE)]
        tP_lo_first = [g.tmem(f"prob{s}_lo_first", (BM, BN // 4), dtype=ll.bfloat16, stages=1) for s in range(Q_STAGE)]
        tP_lo_rest = [g.tmem(f"prob{s}_lo_rest", (BM, BN // 4), dtype=ll.bfloat16, stages=1) for s in range(Q_STAGE)]
        tO = [g.tmem(f"out{s}", (BM, HEAD_DIM), dtype=ll.float32, stages=1, copies=1) for s in range(Q_STAGE)]

        sQ0, sQ1 = sQ
        sO0, sO1 = sO
        sRowSum0, sRowSum1 = sRowSum
        score0, score1 = tS
        prob0_hi, prob1_hi = tP_hi
        prob0_lo_first, prob1_lo_first = tP_lo_first
        prob0_lo_rest, prob1_lo_rest = tP_lo_rest
        out0, out1 = tO

        g.memory_mapping_plan = [
            MemoryMappingEntry("prob0_hi", "score0", space="tmem", offset=BN // 2 + BN // 4, extent=BN // 4,
                               stage_stride=BN),
            MemoryMappingEntry("prob0_lo_first", "score0", space="tmem", offset=BN // 2, extent=BN // 8,
                               stage_stride=BN),
            MemoryMappingEntry("prob0_lo_rest", "score0", space="tmem", offset=BN // 2 + BN // 8, extent=BN // 8,
                               stage_stride=BN),
            MemoryMappingEntry("prob1_hi", "score1", space="tmem", offset=BN // 2 + BN // 4, extent=BN // 4,
                               stage_stride=BN),
            MemoryMappingEntry("prob1_lo_first", "score1", space="tmem", offset=BN // 2, extent=BN // 8,
                               stage_stride=BN),
            MemoryMappingEntry("prob1_lo_rest", "score1", space="tmem", offset=BN // 2 + BN // 8, extent=BN // 8,
                               stage_stride=BN),
            MemoryMappingEntry("out0", "out0", space="tmem", offset=2 * BN, extent=HEAD_DIM, stage_stride=HEAD_DIM),
            MemoryMappingEntry("out1", "out1", space="tmem", offset=3 * BN, extent=HEAD_DIM, stage_stride=HEAD_DIM),
            MemoryMappingEntry("sO0", "sQ0", space="smem", offset=0, extent=SMEM_O_STAGE_BYTES,
                               stage_stride=SMEM_Q_TILE_BYTES),
            MemoryMappingEntry("sO1", "sQ1", space="smem", offset=0, extent=SMEM_O_STAGE_BYTES,
                               stage_stride=SMEM_Q_TILE_BYTES),
        ]

        stage_k = g.depends(src=load_kv, dst=qk_role, via=[sK])
        stage_v = g.depends(src=load_kv, dst=pv_role, via=[sV])
        stage_o_store0 = g.depends(src=correction, dst=store, via=[sO0])
        stage_o_store1 = g.depends(src=correction, dst=store, via=[sO1])
        stage_q0 = g.depends(src=load_q, dst=qk_role, via=[sQ0])
        stage_q1 = g.depends(src=load_q, dst=qk_role, via=[sQ1])
        stage_score0 = g.depends(src=qk_role, dst=sfm0, via=[score0])
        stage_score1 = g.depends(src=qk_role, dst=sfm1, via=[score1])
        stage_prob0_split = g.depends(src=sfm0, dst=pv_role, via=[prob0_hi, prob0_lo_first])
        stage_prob0_lo_rest = g.depends(src=sfm0, dst=pv_role, via=[prob0_lo_rest])
        stage_prob1_split = g.depends(src=sfm1, dst=pv_role, via=[prob1_hi, prob1_lo_first])
        stage_prob1_lo_rest = g.depends(src=sfm1, dst=pv_role, via=[prob1_lo_rest])
        stage_sum0 = g.depends(src=sfm0, dst=correction, via=[sRowSum0])
        stage_sum1 = g.depends(src=sfm1, dst=correction, via=[sRowSum1])
        stage_out0 = g.depends(src=pv_role, dst=correction, via=[out0])
        stage_out1 = g.depends(src=pv_role, dst=correction, via=[out1])

        stage_score = [stage_score0, stage_score1]
        stage_prob_split = [stage_prob0_split, stage_prob1_split]
        stage_prob_lo_rest = [stage_prob0_lo_rest, stage_prob1_lo_rest]

    tid: ll.int32 = ll.threadIdx_x()
    warp_idx: ll.int32 = tid >> 5
    m_pack: ll.int32 = seqlen * KV_GROUP
    num_m_blocks: ll.int32 = (m_pack + BM - 1) // BM
    num_q_stage_blocks: ll.int32 = (num_m_blocks + Q_STAGE - 1) >> 1
    total_work_tiles: ll.int32 = num_q_stage_blocks * H_KV * batch
    work_idx: ll.int32 = ll.int32(ll.blockIdx_x())
    work_stride: ll.int32 = ll.int32(ll.gridDim_x())

    while work_idx < total_work_tiles:
        q_stage_linear: ll.int32 = work_idx % num_q_stage_blocks
        work_tmp: ll.int32 = work_idx // num_q_stage_blocks
        kv_head_idx: ll.int32 = work_tmp % H_KV
        batch_idx: ll.int32 = work_tmp // H_KV
        q_stage_block: ll.int32 = num_q_stage_blocks - 1 - q_stage_linear
        m_block0: ll.int32 = q_stage_block << 1
        m_block1: ll.int32 = m_block0 + 1
        batch_offset: ll.int32 = batch_idx * seqlen_tma
        pack_batch_offset: ll.int32 = (batch_idx * H_KV + kv_head_idx) * m_pack_tma
        num_k_blocks: ll.int32 = (seqlen + BN - 1) // BN
        packed_row_last: ll.int32 = m_block1 * BM + BM - 1
        row_idx_last: ll.int32 = ll.min_val(packed_row_last // KV_GROUP, seqlen - 1)
        max_k_block: ll.int32 = (row_idx_last + BN) // BN
        if max_k_block > num_k_blocks:
            max_k_block = num_k_blocks

        with load_q:
            el_q_load: ll.uint32 = ll.elect_one()
            stage_q0.bind(0)
            stage_q1.bind(0)
            if el_q_load == 1:
                q_row0: ll.int32 = pack_batch_offset + m_block0 * BM
                q_row1: ll.int32 = pack_batch_offset + m_block1 * BM
                ri.tma_ld_2d(sQ0, tma_QP, (0, q_row0), shape=(BM, HEAD_DIM))
                ri.tma_ld_2d(sQ1, tma_QP, (0, q_row1), shape=(BM, HEAD_DIM))

        with load_kv:
            el_kv_load: ll.uint32 = ll.elect_one()
            n_block_load: ll.int32 = max_k_block - 1
            while n_block_load >= 0:
                stage_k.bind(n_block_load)
                stage_v.bind(n_block_load)
                k_row_load: ll.int32 = batch_offset + (n_block_load << 7)
                if el_kv_load == 1:
                    ri.tma_ld_3d(sK, tma_K, (0, k_row_load, kv_head_idx), shape=(BN, HEAD_DIM))
                    ri.tma_ld_3d(sV, tma_V, (0, k_row_load, kv_head_idx), shape=(HEAD_DIM, BN))
                n_block_load = n_block_load - 1

        with qk_role:
            el_qk_mma: ll.uint32 = ll.elect_one()
            stage_q0.bind(0)
            stage_q1.bind(0)
            qk_iter: ll.int32 = 0
            n_block_qk: ll.int32 = max_k_block - 1
            while n_block_qk >= 0:
                stage_k.bind(qk_iter)
                for q in ll.unroll(range(Q_STAGE)):
                    stage_score[q].bind(qk_iter)

                    if el_qk_mma == 1:
                        for qk_half in ll.unroll(range(2)):
                            qd: ll.uint64 = ri.make_umma_smem_desc(sQ[q], (0, qk_half * TMA_COLS))
                            kd: ll.uint64 = ri.make_umma_smem_desc(sK, (0, qk_half * TMA_COLS))
                            ri.umma_f16_ss_1sm_kblock(
                                tS[q],
                                qd,
                                kd,
                                TMA_COLS,
                                out_shape=(BM, BN),
                                zero_init=qk_half == 0,
                            )

                qk_iter = qk_iter + 1
                n_block_qk = n_block_qk - 1

        with pv_role:
            el_pv_mma: ll.uint32 = ll.elect_one()
            stage_out0.bind(0)
            stage_out1.bind(0)
            pv_iter: ll.int32 = 0
            n_block_pv: ll.int32 = max_k_block - 1
            while n_block_pv >= 0:
                stage_v.bind(pv_iter)
                for q in ll.unroll(range(Q_STAGE)):
                    stage_prob_split[q].bind(pv_iter)
                    stage_prob_lo_rest[q].bind(pv_iter)

                    if el_pv_mma == 1:
                        vd_hi: ll.uint64 = ri.make_umma_smem_desc(sV, (0, TMA_COLS))
                        ri.umma_f16_ts_1sm_kblock(tO[q], tP_hi[q], vd_hi, TMA_COLS, out_shape=(BM, HEAD_DIM),
                                                  zero_init=pv_iter == 0)

                        vd_lo_first: ll.uint64 = ri.make_umma_smem_desc(sV, (0, 0))
                        ri.umma_f16_ts_1sm_kblock(tO[q], tP_lo_first[q], vd_lo_first, TMA_COLS // 2,
                                                  out_shape=(BM, HEAD_DIM))

                        vd_lo_rest: ll.uint64 = ri.make_umma_smem_desc(sV, (0, TMA_COLS // 2))
                        ri.umma_f16_ts_1sm_kblock(tO[q], tP_lo_rest[q], vd_lo_rest, TMA_COLS // 2,
                                                  out_shape=(BM, HEAD_DIM))

                pv_iter = pv_iter + 1
                n_block_pv = n_block_pv - 1

        with store:
            el_store: ll.uint32 = ll.elect_one()
            stage_o_store0.bind(0)
            stage_o_store1.bind(0)
            if el_store == 1:
                o_row0: ll.int32 = pack_batch_offset + m_block0 * BM
                o_row1: ll.int32 = pack_batch_offset + m_block1 * BM

                ri.tma_st_2d(sO0, tma_OP, (0, o_row0), shape=(BM, HEAD_DIM))
                ri.tma_st_2d(sO1, tma_OP, (0, o_row1), shape=(BM, HEAD_DIM))

        with sfm0:
            softmax_fn(
                tid,
                SOFTMAX_TID_START,
                m_block0,
                seqlen,
                max_k_block,
                softmax_scale_log2,
                score0,
                prob0_hi,
                prob0_lo_first,
                prob0_lo_rest,
                sRowSum0,
                stage_score0,
                stage_prob0_split,
                stage_prob0_lo_rest,
                stage_sum0,
            )

        with sfm1:
            softmax_fn(
                tid,
                SOFTMAX_STAGE1_TID_START,
                m_block1,
                seqlen,
                max_k_block,
                softmax_scale_log2,
                score1,
                prob1_hi,
                prob1_lo_first,
                prob1_lo_rest,
                sRowSum1,
                stage_score1,
                stage_prob1_split,
                stage_prob1_lo_rest,
                stage_sum1,
            )

        with correction:
            corr_lane0: ll.int32 = tid & 31
            corr_warp0: ll.int32 = warp_idx
            corr_row0: ll.int32 = (corr_warp0 << 5) + corr_lane0
            corr_row0_u: ll.uint32 = ll.uint32(corr_row0)
            corr_dp0_addr: ll.uint32 = ll.uint32(corr_row0 - corr_lane0) << 16
            stage_sum0.bind(0)
            stage_out0.bind(0)
            stage_o_store0.bind(0)
            row_sum_final0: ll.float32 = ri.smem_ld_f32(sRowSum0, corr_row0)
            inv_sum_final0: ll.float32 = ll.float32(0.0)
            if row_sum_final0 != ll.float32(0.0):
                inv_sum_final0 = rcp_f32(row_sum_final0)
            o0_stage_base_bytes: ll.uint32 = ll.cvta_generic_to_shared(sO0)
            for n in ll.unroll(range(NUM_STORES)):
                o0_half_base_bytes: ll.uint32 = o0_stage_base_bytes + ll.uint32(n * SMEM_O_HALF_BYTES)
                for block0_32 in ll.unroll(range(2)):
                    o0_tcol: ll.uint32 = out0 + ll.uint32(n * STORE_BN + block0_32 * 32) + corr_dp0_addr
                    ri.tmem_ld_32dp_f32_to_smem_bf16_x32(
                        out0,
                        sO0,
                        o0_tcol,
                        o0_half_base_bytes,
                        corr_row0_u,
                        ll.uint32(block0_32 * 4),
                        inv_sum_final0,
                    )

            corr_lane1: ll.int32 = tid & 31
            corr_warp1: ll.int32 = warp_idx
            corr_row1: ll.int32 = (corr_warp1 << 5) + corr_lane1
            corr_row1_u: ll.uint32 = ll.uint32(corr_row1)
            corr_dp1_addr: ll.uint32 = ll.uint32(corr_row1 - corr_lane1) << 16
            stage_sum1.bind(0)
            stage_out1.bind(0)
            stage_o_store1.bind(0)
            row_sum_final1: ll.float32 = ri.smem_ld_f32(sRowSum1, corr_row1)
            inv_sum_final1: ll.float32 = ll.float32(0.0)
            if row_sum_final1 != ll.float32(0.0):
                inv_sum_final1 = rcp_f32(row_sum_final1)
            o1_stage_base_bytes: ll.uint32 = ll.cvta_generic_to_shared(sO1)
            for n in ll.unroll(range(NUM_STORES)):
                o1_half_base_bytes: ll.uint32 = o1_stage_base_bytes + ll.uint32(n * SMEM_O_HALF_BYTES)
                for block1_32 in ll.unroll(range(2)):
                    o1_tcol: ll.uint32 = out1 + ll.uint32(n * STORE_BN + block1_32 * 32) + corr_dp1_addr
                    ri.tmem_ld_32dp_f32_to_smem_bf16_x32(
                        out1,
                        sO1,
                        o1_tcol,
                        o1_half_base_bytes,
                        corr_row1_u,
                        ll.uint32(block1_32 * 4),
                        inv_sum_final1,
                    )

        ll.__syncthreads()
        work_idx = work_idx + work_stride
