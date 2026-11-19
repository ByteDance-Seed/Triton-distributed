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
"""Fused SM100 varlen gated-delta forward for H=64, K=V=128.

One CTA owns one ``(sequence, head)`` pair and walks 64-token chunks in
sequence order.  All cross-role synchronization is compiler generated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from little_kernel.backends.nv.sm100.passes.dataflow import dataflow_frontend

import math
from dataclasses import dataclass
from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.builtin_base import inline_func
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

from little_kernel.backends.nv.sm100.passes.dataflow.buffer_overlay import MemoryMappingEntry

H = 64
D = 128
BT = 64
QK_PAIR_STAGES = 2
V_STAGES = 1
OUTPUT_STAGES = 2
GATE_STAGES = 5
SCORE_STAGES = 2
LONG_SEQUENCE_SEGMENTS = 3
PERSISTENT_TASKS_PER_CTA = 7
ASSUME_FULL_CHUNKS = False
ASSUME_SINGLE_SEGMENT = False
ALIAS_VNEW_BF_WITH_STATE_BF = False
FUSE_STATE_VNEW_BF = False
LOG2E = 1.4426950408889634

NUM_THREADS = 512
STATE_WARPS = list(range(8))
SOLVE_WARPS = [8, 9, 10, 11]
MMA1_WARPS = [12]
IO_WARPS = [13]
MMA0_WARPS = [14]
STORE_WARPS = [15]

STATE_TID = 0
SOLVE_TID = 256
MMA1_TID = 384
IO_TID = 416
STORE_TID = 480

SMEM_INPUT_BYTES = (QK_PAIR_STAGES * 4 + V_STAGES) * BT * D * 2
SMEM_STATE_BYTES = D * D * 2
SMEM_TRANSPOSE_BYTES = D * BT * 2
SMEM_TOTAL = 227 * 1024


@inline_func
def tma_ld_qkv_g_fn(
    tid,
    dQ,
    dK,
    seq_start,
    seqlen,
    head,
    num_pairs,
    scale,
    g_ptr,
    beta_ptr,
    sQK,
    sG,
    sGInv,
    sGOut,
    sBeta,
    stage_qk,
    stage_gate,
) -> ll.void:
    issuer: ll.uint32 = ll.elect_one()
    lane: ll.int32 = tid & 31
    pair: ll.int32 = 0
    while pair < num_pairs:
        stage_qk.bind(pair)
        if issuer == 1:
            token0: ll.int32 = seq_start + pair * 2 * BT
            token1: ll.int32 = token0 + BT
            ri.tma_ld_3d(sQK, dK, (0, token0, head), smem_coord=(0, 0), shape=(BT, D))
            ri.tma_ld_3d(sQK, dQ, (0, token0, head), smem_coord=(BT, 0), shape=(BT, D))
            ri.tma_ld_3d(sQK, dK, (0, token1, head), smem_coord=(BT * 2, 0), shape=(BT, D))
            ri.tma_ld_3d(sQK, dQ, (0, token1, head), smem_coord=(BT * 3, 0), shape=(BT, D))
        gate_scan_fn(lane, pair * 2, seq_start, seqlen, head, scale, g_ptr, beta_ptr, sG, sGInv, sGOut, sBeta,
                     stage_gate)
        gate_scan_fn(lane, pair * 2 + 1, seq_start, seqlen, head, scale, g_ptr, beta_ptr, sG, sGInv, sGOut, sBeta,
                     stage_gate)
        pair = pair + 1


@inline_func
def gate_scan_fn(
    lane,
    chunk,
    seq_start,
    seqlen,
    head,
    scale,
    g_ptr,
    beta_ptr,
    sG,
    sGInv,
    sGOut,
    sBeta,
    stage_gate,
) -> ll.void:
    stage_gate.bind(chunk)
    chunk_start: ll.int32 = chunk * BT
    valid: ll.int32 = BT if ASSUME_FULL_CHUNKS else ll.min_val(BT, seqlen - chunk_start)
    row0: ll.int32 = lane
    row1: ll.int32 = lane + 32
    g0: ll.float32 = ll.float32(0.0)
    g1: ll.float32 = ll.float32(0.0)
    beta0: ll.float32 = ll.float32(0.0)
    beta1: ll.float32 = ll.float32(0.0)
    if row0 < valid:
        token0: ll.int64 = ll.int64(seq_start + chunk_start + row0)
        offset0: ll.int64 = token0 * ll.int64(H) + ll.int64(head)
        g0 = g_ptr[offset0] * ll.float32(LOG2E)
        beta0 = beta_ptr[offset0]
    if row1 < valid:
        token1: ll.int64 = ll.int64(seq_start + chunk_start + row1)
        offset1: ll.int64 = token1 * ll.int64(H) + ll.int64(head)
        g1 = g_ptr[offset1] * ll.float32(LOG2E)
        beta1 = beta_ptr[offset1]
    for scan_step in ll.unroll(range(5)):
        delta: ll.int32 = 1 << scan_step
        prefix0: ll.float32 = ll.shfl_up_sync(0xFFFFFFFF, g0, delta)
        prefix1: ll.float32 = ll.shfl_up_sync(0xFFFFFFFF, g1, delta)
        if lane >= delta:
            g0 = g0 + prefix0
            g1 = g1 + prefix1
    g1 = g1 + ll.__shfl_sync(ll.uint32(0xFFFFFFFF), g0, 31)
    gate0: ll.float32 = ll.exp2f(g0)
    gate1: ll.float32 = ll.exp2f(g1)
    ri.smem_st_f32(sG, row0, gate0)
    ri.smem_st_f32(sG, row1, gate1)
    ri.smem_st_f32(sGInv, row0, ll.float32(1.0) / gate0)
    ri.smem_st_f32(sGInv, row1, ll.float32(1.0) / gate1)
    ri.smem_st_f32(sGOut, row0, scale * gate0)
    ri.smem_st_f32(sGOut, row1, scale * gate1)
    ri.smem_st_f32(sBeta, row0, beta0)
    ri.smem_st_f32(sBeta, row1, beta1)


@inline_func
def scale_state_fn(tState, dp_addr, decay, col_start) -> ll.void:
    decay2: ll.uint64 = ll.pack_f32x2(decay, decay)
    for col_offset in ll.unroll(range(0, D // 2, 32)):
        col: ll.int32 = col_start + col_offset
        state_values = ll.empty([32], dtype=ll.uint32, scope="local")
        ri.tmem_ld_32dp_32bxn(tState, ll.uint32(tState + col) + dp_addr, state_values, 32)
        ri.tmem_ld_fence(tState)
        for pair in ll.unroll(range(16)):
            scaled: ll.uint64 = ll.mul_f32x2(
                ll.pack_f32x2(ll.uint_as_float(state_values[pair * 2]), ll.uint_as_float(state_values[pair * 2 + 1])),
                decay2)
            state_values[pair * 2] = ll.uint32(scaled)
            state_values[pair * 2 + 1] = ll.uint32(scaled >> 32)
        ri.tmem_st_32dp_32bxn(tState, ll.uint32(tState + col) + dp_addr, state_values, 32)
    ri.tmem_st_fence()


@inline_func
def stage_state_to_smem_all_fn(tState, sStateView, tid) -> ll.void:
    state_group: ll.int32 = tid >> 7
    state_tid: ll.int32 = tid & 127
    lane: ll.int32 = state_tid & 31
    col: ll.int32 = state_group * 32
    dp_addr: ll.uint32 = ll.uint32(state_tid - lane) << 16
    values = ll.empty([32], dtype=ll.uint32, scope="local")
    ll.tmem_ld_32dp_32bxn(
        ll.uint32(tState + col) + dp_addr, values[0], values[1], values[2], values[3], values[4], values[5], values[6],
        values[7], values[8], values[9], values[10], values[11], values[12], values[13], values[14], values[15],
        values[16], values[17], values[18], values[19], values[20], values[21], values[22], values[23], values[24],
        values[25], values[26], values[27], values[28], values[29], values[30], values[31])
    ll.tmem_ld_fence()
    base: ll.uint32 = ll.cvta_generic_to_shared(sStateView)
    for x in ll.unroll(range(0, 32, 4)):
        group: ll.uint32 = ll.uint32((col + x) >> 2)
        atom: ll.uint32 = group >> ll.uint32(3)
        group_in_atom: ll.uint32 = group & ll.uint32(7)
        addr: ll.uint32 = base + (atom * ll.uint32(D) + ll.uint32(state_tid)) * ll.uint32(128)
        addr = addr + (group_in_atom ^ ll.uint32(state_tid & 7)) * ll.uint32(16)
        ll.smem_st_128(addr, values[x], values[x + 1], values[x + 2], values[x + 3])


@inline_func
def rhs_full_fn(tPred, tRhs, sV, sG, state_tid, token_start) -> ll.void:
    lane: ll.int32 = state_tid & 31
    warp: ll.int32 = state_tid >> 5
    for dim_half in ll.unroll(range(2)):
        dim_base: ll.int32 = warp * 32 + dim_half * 16
        for token_offset in ll.unroll(range(0, BT // 2, 16)):
            token_base: ll.int32 = token_start + token_offset
            v_row: ll.int32 = token_base + (lane & 7) + (lane >> 4) * 8
            v_col: ll.int32 = dim_base + ((lane >> 3) & 1) * 8
            v_index: ll.int32 = ((v_col >> 6) * BT + v_row) * 64 + ((((v_col & 63) >> 3) ^
                                                                     (v_row & 7)) * 8) + (v_col & 7)
            rhs_values = ll.empty([4], dtype=ll.uint32, scope="local")
            rhs_predictions = ll.empty([8], dtype=ll.uint32, scope="local")
            rhs_residual = ll.empty([4], dtype=ll.uint32, scope="local")
            ri.ldmatrix_x4_trans(sV, v_index, rhs_values)
            pred_addr: ll.uint32 = ll.uint32(tPred + token_base) + (ll.uint32(dim_base) << 16)
            ri.tmem_ld_16dp_256bx2(tPred, pred_addr, rhs_predictions, 8)
            ri.tmem_ld_fence(tPred)
            for pair in ll.unroll(range(4)):
                row0: ll.int32 = token_base + (lane & 3) * 2
                if pair >= 2:
                    row0 = row0 + 8
                value2: ll.uint64 = ll.pack_f32x2(ll.uint_as_float(rhs_values[pair] << 16),
                                                  ll.uint_as_float(rhs_values[pair] & ll.uint32(0xFFFF0000)))
                pred2: ll.uint64 = ll.pack_f32x2(ll.uint_as_float(rhs_predictions[pair * 2]),
                                                 ll.uint_as_float(rhs_predictions[pair * 2 + 1]))
                rhs_residual[pair] = ll.pack_bf16_f32x2(ll.fma_f32x2(ri.smem_ld_f32x2(sG, row0), pred2, value2), True,
                                                        True)
            rhs_addr: ll.uint32 = ll.uint32(tRhs + token_base // 2) + (ll.uint32(dim_base) << 16)
            ri.tmem_st_16dp_128bx2(tRhs, rhs_addr, rhs_residual[0], rhs_residual[1], rhs_residual[2], rhs_residual[3])
    ri.tmem_st_fence()


@inline_func
def output_full_fn(tO, sO, sGOut, state_tid, state_group) -> ll.void:
    lane: ll.int32 = state_tid & 31
    warp: ll.int32 = state_tid >> 5
    output_base: ll.uint32 = ll.cvta_generic_to_shared(sO)
    for tile_n in ll.unroll(range(0, 4, 2)):
        output_values = ll.empty([16], dtype=ll.uint32, scope="local")
        tmem_addr: ll.uint32 = ll.uint32(tO + tile_n * 16) + (ll.uint32(warp * 32 + state_group * 16) << 16)
        ri.tmem_ld_16dp_256bx4(tO, tmem_addr, output_values, 16)
        ri.tmem_ld_fence(tO)
        for tile_offset in ll.unroll(range(2)):
            packed = ll.empty([4], dtype=ll.uint32, scope="local")
            value_start: ll.int32 = tile_offset * 8
            output_row: ll.int32 = (tile_n + tile_offset) * 16 + (lane & 3) * 2
            scale01x2: ll.uint64 = ri.smem_ld_f32x2(sGOut, output_row)
            scale81x2: ll.uint64 = ri.smem_ld_f32x2(sGOut, output_row + 8)
            for pair in ll.unroll(range(2)):
                value01: ll.uint64 = ll.pack_f32x2(ll.uint_as_float(output_values[value_start + pair * 2]),
                                                   ll.uint_as_float(output_values[value_start + pair * 2 + 1]))
                value81: ll.uint64 = ll.pack_f32x2(ll.uint_as_float(output_values[value_start + pair * 2 + 4]),
                                                   ll.uint_as_float(output_values[value_start + pair * 2 + 5]))
                packed[pair] = ll.pack_bf16_f32x2(ll.mul_f32x2(value01, scale01x2), True, True)
                packed[pair + 2] = ll.pack_bf16_f32x2(ll.mul_f32x2(value81, scale81x2), True, True)
            out_row: ll.int32 = (tile_n + tile_offset) * 16 + (lane & 15)
            out_col: ll.int32 = (warp * 2 + state_group) * 16 + (lane >> 4) * 8
            linear: ll.int32 = (out_col >> 6) * 4096 + out_row * 64 + (out_col & 63)
            byte_offset: ll.uint32 = ll.uint32(linear * 2) ^ ll.uint32((((linear * 2) >> 7) & 7) << 4)
            ri.stmatrix_x4_trans(sO, output_base + byte_offset, packed[0], packed[2], packed[1], packed[3])


@inline_func
def state_role_fn(
    tid,
    dO,
    h0,
    ht,
    o,
    seq,
    seq_start,
    seqlen,
    head,
    num_pairs,
    scale,
    tState,
    tStateBF,
    tPred,
    tRhs,
    tVNew,
    tVNewBF,
    tO,
    sV,
    sG,
    sGInv,
    sGOut,
    sO,
    stage_v,
    stage_gate,
    stage_state_input,
    stage_pred,
    stage_vnew,
    stage_vnew_bf,
    stage_rhs,
    stage_state_fwd,
    stage_state_back,
    stage_tout,
    stage_output,
    sStateDone,
) -> ll.void:
    local_tid: ll.int32 = tid - STATE_TID
    state_group: ll.int32 = local_tid >> 7
    state_tid: ll.int32 = local_tid & 127
    lane: ll.int32 = state_tid & 31
    dp_addr: ll.uint32 = ll.uint32(state_tid - lane) << 16
    col_start: ll.int32 = state_group * (D // 2)
    token_start: ll.int32 = state_group * (BT // 2)
    init_probe = ll.empty([4], dtype=ll.uint32, scope="local")
    ll.tmem_ld_32dp_32bxn(dp_addr, init_probe[0], init_probe[1], init_probe[2], init_probe[3])
    ll.tmem_ld_fence()
    ri.tmem_st_32dp_32bxn(tState, ll.uint32(tState) + dp_addr, init_probe, 4)
    ri.tmem_st_fence()

    chunk: ll.int32 = 0
    while chunk < num_pairs * 2:
        stage_state_input.bind(chunk)
        for input_offset in ll.unroll(range(0, D // 2, 32)):
            input_col: ll.int32 = col_start + input_offset
            state_f32 = ll.empty([32], dtype=ll.uint32, scope="local")
            state_bf16 = ll.empty([16], dtype=ll.uint32, scope="local")
            ri.tmem_ld_32dp_32bxn(tState, ll.uint32(tState + input_col) + dp_addr, state_f32, 32)
            ri.tmem_ld_fence(tState)
            for input_x in ll.unroll(range(16)):
                state_bf16[input_x] = ll.pack_bf16_float(ll.uint_as_float(state_f32[input_x * 2]),
                                                         ll.uint_as_float(state_f32[input_x * 2 + 1]))
            ri.tmem_st_32dp_32bxn(tStateBF, ll.uint32(tStateBF + input_col // 2) + dp_addr, state_bf16, 16)
        ri.tmem_st_fence()
        stage_gate.bind(chunk)
        decay: ll.float32 = ri.smem_ld_f32(sG, BT - 1)
        valid: ll.int32 = BT if ASSUME_FULL_CHUNKS else ll.min_val(BT, seqlen - chunk * BT)
        stage_v.bind(chunk)
        stage_pred.bind(chunk)
        stage_rhs.bind(chunk)
        if valid == BT:
            rhs_full_fn(tPred, tRhs, sV, sG, state_tid, token_start)
        else:
            for projection_offset in ll.unroll(range(0, BT // 2, 8)):
                projection_block: ll.int32 = token_start + projection_offset
                predictions = ll.empty([8], dtype=ll.uint32, scope="local")
                residual = ll.empty([4], dtype=ll.uint32, scope="local")
                ri.tmem_ld_32dp_32bxn(tPred, ll.uint32(tPred + projection_block) + dp_addr, predictions, 8)
                ri.tmem_ld_fence(tPred)
                for projection_x in ll.unroll(range(8)):
                    token_row: ll.int32 = projection_block + projection_x
                    gate_row: ll.float32 = ri.smem_ld_f32(sG, token_row)
                    rhs: ll.float32 = ll.float32(0.0)
                    if token_row < valid:
                        vv: ll.float32 = ll.bf16_to_float(ri.smem_ld_bf16_k_major(sV, BT, token_row, state_tid))
                        rhs = vv + gate_row * ll.uint_as_float(predictions[projection_x])
                    predictions[projection_x] = ll.float_as_uint(rhs)
                for residual_x in ll.unroll(range(4)):
                    residual[residual_x] = ll.pack_bf16_float(ll.uint_as_float(predictions[residual_x * 2]),
                                                              ll.uint_as_float(predictions[residual_x * 2 + 1]))
                ri.tmem_st_32dp_32bxn(tRhs, ll.uint32(tRhs + projection_block // 2) + dp_addr, residual, 4)
        ri.tmem_st_fence()
        stage_state_fwd.bind(chunk)
        scale_state_fn(tState, dp_addr, decay, col_start)
        stage_vnew.bind(chunk)
        if not FUSE_STATE_VNEW_BF:
            stage_vnew_bf.bind(chunk)
        for solution_offset in ll.unroll(range(0, BT // 2, 32)):
            solution_block: ll.int32 = token_start + solution_offset
            solution_f32 = ll.empty([32], dtype=ll.uint32, scope="local")
            solution_bf16 = ll.empty([16], dtype=ll.uint32, scope="local")
            ri.tmem_ld_32dp_32bxn(tVNew, ll.uint32(tVNew + solution_block) + dp_addr, solution_f32, 32)
            ri.tmem_ld_fence(tVNew)
            for solution_x in ll.unroll(range(16)):
                solution0: ll.float32 = ll.uint_as_float(solution_f32[solution_x * 2])
                solution1: ll.float32 = ll.uint_as_float(solution_f32[solution_x * 2 + 1])
                solution_bf16[solution_x] = ll.pack_bf16_float(solution0, solution1)
            ri.tmem_st_32dp_32bxn(tVNewBF, ll.uint32(tVNewBF + solution_block // 2) + dp_addr, solution_bf16, 16)
        ri.tmem_st_fence()
        stage_tout.bind(chunk)
        stage_output.bind(chunk)
        if valid == BT:
            output_full_fn(tO, sO, sGOut, state_tid, state_group)
        else:
            for output_offset_in_group in ll.unroll(range(0, BT // 2, 16)):
                output_block: ll.int32 = token_start + output_offset_in_group
                output = ll.empty([16], dtype=ll.uint32, scope="local")
                output_scales = ll.empty([16], dtype=ll.float32, scope="local")
                for output_x in ll.unroll(range(16)):
                    output_scales[output_x] = ri.smem_ld_f32(sGOut, output_block + output_x)
                ri.tmem_ld_32dp_32bxn(tO, ll.uint32(tO + output_block) + dp_addr, output, 16)
                ri.tmem_ld_fence(tO)
                for output_x in ll.unroll(range(16)):
                    output_row: ll.int32 = output_block + output_x
                    if output_row < valid:
                        output_token: ll.int64 = ll.int64(seq_start + chunk * BT + output_row)
                        output_offset: ll.int64 = output_token * ll.int64(H * D) + ll.int64(head * D + state_tid)
                        o[output_offset] = ll.float_to_bf16(
                            ll.uint_as_float(output[output_x]) * output_scales[output_x])
        if valid == BT:
            if local_tid == 0:
                token: ll.int32 = seq_start + chunk * BT
                ri.tma_st_3d(sO, dO, (0, token, head), shape=(BT, D))
        stage_state_back.bind(chunk)
        ri.tmem_ld_fence(tState)
        chunk = chunk + 1
    if local_tid == 0:
        ll.mbarrier_arrive(sStateDone)


@inline_func
def mma_qk_kk_fn(num_pairs, sQK, tKKT, tQK, stage_qk, stage_kkt, stage_qk_acc) -> ll.void:
    issuer: ll.uint32 = ll.elect_one()
    pair: ll.int32 = 0
    while pair < num_pairs:
        chunk: ll.int32 = pair * 2
        stage_qk.bind(pair)
        stage_kkt.bind(chunk)
        if issuer == 1:
            k0d: ll.uint64 = ri.make_umma_smem_desc(sQK, (0, 0))
            ri.umma_f16_ss_1sm_kblock(tKKT, k0d, k0d, D, out_shape=(BT, BT), zero_init=True)
        stage_kkt.bind(chunk + 1)
        if issuer == 1:
            k1d: ll.uint64 = ri.make_umma_smem_desc(sQK, (BT * 2, 0))
            ri.umma_f16_ss_1sm_kblock(tKKT, k1d, k1d, D, out_shape=(BT, BT), zero_init=True)
        stage_qk_acc.bind(chunk)
        if issuer == 1:
            qk0_kd: ll.uint64 = ri.make_umma_smem_desc(sQK, (0, 0))
            q0d: ll.uint64 = ri.make_umma_smem_desc(sQK, (BT, 0))
            ri.umma_f16_ss_1sm_kblock(tQK, q0d, qk0_kd, D, out_shape=(BT, BT), zero_init=True,
                                      tmem_d=ll.uint32(tQK) + (ll.uint32(16) << 16))
        stage_qk_acc.bind(chunk + 1)
        if issuer == 1:
            qk1_kd: ll.uint64 = ri.make_umma_smem_desc(sQK, (BT * 2, 0))
            q1d: ll.uint64 = ri.make_umma_smem_desc(sQK, (BT * 3, 0))
            ri.umma_f16_ss_1sm_kblock(tQK, q1d, qk1_kd, D, out_shape=(BT, BT), zero_init=True,
                                      tmem_d=ll.uint32(tQK) + (ll.uint32(16) << 16))
        pair = pair + 1


@inline_func
def kkt_epilogue_fn(tKKT, sL, sBeta, dp_addr, valid, row, half) -> ll.void:
    # Factor L = diag(exp(g)) L0 diag(exp(-g)). Inverting L0 lets
    # publish_inverse apply exp(g_last - g_j) * beta_j by column.
    row_scale: ll.float32 = ri.smem_ld_f32(sBeta, row)
    row_scale2: ll.uint64 = ll.pack_f32x2(row_scale, row_scale)
    base: ll.uint32 = ll.cvta_generic_to_shared(sL)
    for block in ll.unroll(range(0, BT, 32)):
        values = ll.empty([16], dtype=ll.uint32, scope="local")
        packed = ll.empty([8], dtype=ll.uint32, scope="local")
        ri.tmem_ld_16dp_32bxn(tKKT, ll.uint32(tKKT + block) + dp_addr, values, 16)
        ri.tmem_ld_fence(tKKT)
        for pair in ll.unroll(range(8)):
            col0: ll.int32 = block + half * 16 + pair * 2
            col1: ll.int32 = col0 + 1
            value: ll.uint64 = ll.pack_f32x2(ll.uint_as_float(values[pair * 2]), ll.uint_as_float(values[pair * 2 + 1]))
            value = ll.mul_f32x2(value, row_scale2)
            packed[pair] = ll.pack_bf16_f32x2(value, row < valid and col0 < row, col1 < row)
        group: ll.uint32 = ll.uint32((block + half * 16) >> 3)
        addr0: ll.uint32 = ll.smem_swizzle_128b_addr(base, ll.uint32(row), group)
        addr1: ll.uint32 = ll.smem_swizzle_128b_addr(base, ll.uint32(row), group + ll.uint32(1))
        ri.smem_st_128(sL, addr0, packed[0], packed[1], packed[2], packed[3])
        ri.smem_st_128(sL, addr1, packed[4], packed[5], packed[6], packed[7])


@inline_func
def qk_epilogue_fn(tQK, sP, score_scale, dp_addr, valid, row, half) -> ll.void:
    base: ll.uint32 = ll.cvta_generic_to_shared(sP)
    score_scale2: ll.uint64 = ll.pack_f32x2(score_scale, score_scale)
    for block in ll.unroll(range(0, BT, 32)):
        values = ll.empty([16], dtype=ll.uint32, scope="local")
        packed = ll.empty([8], dtype=ll.uint32, scope="local")
        ri.tmem_ld_16dp_32bxn(tQK, ll.uint32(tQK + block) + dp_addr, values, 16)
        ri.tmem_ld_fence(tQK)
        for pair in ll.unroll(range(8)):
            col0: ll.int32 = block + half * 16 + pair * 2
            col1: ll.int32 = col0 + 1
            value: ll.uint64 = ll.pack_f32x2(ll.uint_as_float(values[pair * 2]), ll.uint_as_float(values[pair * 2 + 1]))
            value = ll.mul_f32x2(value, score_scale2)
            packed[pair] = ll.pack_bf16_f32x2(value, row < valid and col0 <= row, col1 <= row)
        group: ll.uint32 = ll.uint32((block + half * 16) >> 3)
        addr0: ll.uint32 = ll.smem_swizzle_128b_addr(base, ll.uint32(row), group)
        addr1: ll.uint32 = ll.smem_swizzle_128b_addr(base, ll.uint32(row), group + ll.uint32(1))
        ri.smem_st_128(sP, addr0, packed[0], packed[1], packed[2], packed[3])
        ri.smem_st_128(sP, addr1, packed[4], packed[5], packed[6], packed[7])


@inline_func
def inverse_index(row, col) -> ll.int32:
    return row * BT + (((col >> 3) ^ (row & 7)) << 3) + (col & 7)


@inline_func
def inv_blk64_pair_fn(sL0, sL1, local_tid) -> ll.void:
    lane: ll.int32 = local_tid & 31
    warp: ll.int32 = local_tid >> 5
    inverse_group: ll.int32 = warp >> 1
    inverse_warp: ll.int32 = warp & 1
    inverse_tid: ll.int32 = inverse_warp * 32 + lane
    work_base: ll.uint32 = ll.cvta_generic_to_shared(sL0)
    if inverse_group == 1:
        work_base = ll.cvta_generic_to_shared(sL1)
    sL = ll.smem_bf16_ptr_from_addr(work_base)
    ll.named_barrier_sync(15, 128)

    block8: ll.int32 = inverse_tid >> 3
    row8: ll.int32 = inverse_tid & 7
    base8: ll.int32 = block8 << 3
    row_addr: ll.uint32 = ll.smem_swizzle_128b_addr(work_base, ll.uint32(base8 + row8), ll.uint32(block8))
    row_packed = ll.empty([4], dtype=ll.uint32, scope="local")
    row_values = ll.empty([8], dtype=ll.float32, scope="local")
    ri.smem_ld_128(sL, row_addr, row_packed, 4)
    for col in ll.unroll(range(8)):
        row_values[col] = ll.uint_as_float((row_packed[col >> 1] >> ((col & 1) * 16)) << 16)
        if row8 == col:
            row_values[col] = ll.float32(1.0)
    for src in ll.unroll(range(7)):
        row_scale: ll.float32 = -row_values[src]
        for col in ll.unroll(range(7)):
            if col < src:
                pivot: ll.float32 = ll.__shfl_sync(ll.uint32(0xFFFFFFFF), row_values[col], (lane & ~7) + src)
                if row8 > src:
                    row_values[col] = row_scale * pivot + row_values[col]
        if row8 > src:
            row_values[src] = row_scale
    for pair in ll.unroll(range(4)):
        row_packed[pair] = ll.pack_bf16_float(row_values[pair * 2], row_values[pair * 2 + 1])
    ri.smem_st_128(sL, row_addr, row_packed[0], row_packed[1], row_packed[2], row_packed[3])
    ll.named_barrier_sync(15, 128)

    ld_row16: ll.int32 = lane & 7
    stage16_d = ll.empty([1], dtype=ll.uint32, scope="local")
    stage16_b = ll.empty([1], dtype=ll.uint32, scope="local")
    stage16_a = ll.empty([2], dtype=ll.uint32, scope="local")
    stage16_acc0 = ll.empty([4], dtype=ll.float32, scope="local")
    stage16_acc1 = ll.empty([4], dtype=ll.float32, scope="local")
    base16: ll.int32 = warp << 4
    ri.ldmatrix_x1(sL0, inverse_index(base16 + 8 + ld_row16, base16 + 8), stage16_d)
    stage16_a[0] = stage16_d[0]
    stage16_a[1] = stage16_d[0]
    ri.ldmatrix_x1_trans(sL0, inverse_index(base16 + 8 + ld_row16, base16), stage16_b)
    ll.mma_m16n8k8(stage16_acc0, stage16_a, stage16_b[0], False)
    ri.ldmatrix_x1(sL1, inverse_index(base16 + 8 + ld_row16, base16 + 8), stage16_d)
    stage16_a[0] = stage16_d[0]
    stage16_a[1] = stage16_d[0]
    ri.ldmatrix_x1_trans(sL1, inverse_index(base16 + 8 + ld_row16, base16), stage16_b)
    ll.mma_m16n8k8(stage16_acc1, stage16_a, stage16_b[0], False)
    stage16_a[0] = ll.pack_bf16_float(-stage16_acc0[0], -stage16_acc0[1])
    stage16_a[1] = ll.pack_bf16_float(-stage16_acc0[2], -stage16_acc0[3])
    ri.ldmatrix_x1_trans(sL0, inverse_index(base16 + ld_row16, base16), stage16_b)
    ll.mma_m16n8k8(stage16_acc0, stage16_a, stage16_b[0], False)
    stage16_d[0] = ll.pack_bf16_float(stage16_acc0[0], stage16_acc0[1])
    ri.stmatrix_x1_reg(sL0, inverse_index(base16 + 8 + ld_row16, base16), stage16_d[0])
    stage16_a[0] = ll.pack_bf16_float(-stage16_acc1[0], -stage16_acc1[1])
    stage16_a[1] = ll.pack_bf16_float(-stage16_acc1[2], -stage16_acc1[3])
    ri.ldmatrix_x1_trans(sL1, inverse_index(base16 + ld_row16, base16), stage16_b)
    ll.mma_m16n8k8(stage16_acc1, stage16_a, stage16_b[0], False)
    stage16_d[0] = ll.pack_bf16_float(stage16_acc1[0], stage16_acc1[1])
    ri.stmatrix_x1_reg(sL1, inverse_index(base16 + 8 + ld_row16, base16), stage16_d[0])
    ll.named_barrier_sync(15, 128)

    base32: ll.int32 = inverse_warp << 5
    ld_row32: ll.int32 = lane & 15
    ld_col32: ll.int32 = (lane >> 4) << 3
    stage32_d = ll.empty([4], dtype=ll.uint32, scope="local")
    stage32_c = ll.empty([4], dtype=ll.uint32, scope="local")
    stage32_a = ll.empty([4], dtype=ll.uint32, scope="local")
    stage32_dc = ll.empty([4], dtype=ll.uint32, scope="local")
    stage32_acc = ll.empty([8], dtype=ll.float32, scope="local")
    ri.ldmatrix_x4(sL, inverse_index(base32 + 16 + ld_row32, base32 + 16 + ld_col32), stage32_d)
    ri.ldmatrix_x4_trans(sL, inverse_index(base32 + 16 + ld_row32, base32 + ld_col32), stage32_c)
    ll.mma_m16n16k16(stage32_acc, stage32_d, stage32_c, False)
    ll.acc_to_bf16x4_scaled(stage32_acc, stage32_dc, ll.float32(-1.0))
    ri.ldmatrix_x4_trans(sL, inverse_index(base32 + ld_row32, base32 + ld_col32), stage32_a)
    ll.mma_m16n16k16(stage32_acc, stage32_dc, stage32_a, False)
    ll.acc_to_bf16x4_scaled(stage32_acc, stage32_dc, ll.float32(1.0))
    ri.stmatrix_x4_regs(sL, inverse_index(base32 + 16 + ld_row32, base32 + ld_col32), stage32_dc)
    ll.named_barrier_sync(15, 128)

    row_base64: ll.int32 = 32 + inverse_warp * 16
    ld_row64: ll.int32 = lane & 15
    ld_col64: ll.int32 = (lane >> 4) << 3
    stage64_d = ll.empty([4], dtype=ll.uint32, scope="local")
    stage64_b = ll.empty([4], dtype=ll.uint32, scope="local")
    stage64_dc0 = ll.empty([4], dtype=ll.uint32, scope="local")
    stage64_dc1 = ll.empty([4], dtype=ll.uint32, scope="local")
    stage64_out = ll.empty([4], dtype=ll.uint32, scope="local")
    stage64_temp0 = ll.empty([8], dtype=ll.float32, scope="local")
    stage64_temp1 = ll.empty([8], dtype=ll.float32, scope="local")
    stage64_result = ll.empty([8], dtype=ll.float32, scope="local")
    for kb in ll.unroll(range(2)):
        # Bottom-left inverse block is -D^-1 C A^-1; D starts at column 32.
        ri.ldmatrix_x4(sL, inverse_index(row_base64 + ld_row64, 32 + kb * 16 + ld_col64), stage64_d)
        ri.ldmatrix_x4_trans(sL, inverse_index(32 + kb * 16 + ld_row64, ld_col64), stage64_b)
        ll.mma_m16n16k16(stage64_temp0, stage64_d, stage64_b, kb != 0)
        ri.ldmatrix_x4_trans(sL, inverse_index(32 + kb * 16 + ld_row64, 16 + ld_col64), stage64_b)
        ll.mma_m16n16k16(stage64_temp1, stage64_d, stage64_b, kb != 0)
    ll.acc_to_bf16x4_scaled(stage64_temp0, stage64_dc0, ll.float32(-1.0))
    ll.acc_to_bf16x4_scaled(stage64_temp1, stage64_dc1, ll.float32(-1.0))
    for nb in ll.unroll(range(2)):
        ri.ldmatrix_x4_trans(sL, inverse_index(ld_row64, nb * 16 + ld_col64), stage64_b)
        ll.mma_m16n16k16(stage64_result, stage64_dc0, stage64_b, False)
        ri.ldmatrix_x4_trans(sL, inverse_index(16 + ld_row64, nb * 16 + ld_col64), stage64_b)
        ll.mma_m16n16k16(stage64_result, stage64_dc1, stage64_b, True)
        ll.acc_to_bf16x4_scaled(stage64_result, stage64_out, ll.float32(1.0))
        ri.stmatrix_x4_regs(sL, inverse_index(row_base64 + ld_row64, nb * 16 + ld_col64), stage64_out)
    ll.named_barrier_sync(15, 128)


@inline_func
def publish_inverse_fn(sL, sD, scale_lo2, scale_hi2, local_tid) -> ll.void:
    warp: ll.int32 = local_tid >> 5
    lane: ll.int32 = local_tid & 31
    half: ll.int32 = lane >> 4
    row: ll.int32 = warp * 16 + (lane & 15)
    l_base: ll.uint32 = ll.cvta_generic_to_shared(sL)
    d_base: ll.uint32 = ll.cvta_generic_to_shared(sD)
    for segment in ll.unroll(range(2)):
        col: ll.int32 = segment * 32 + half * 16
        for group_offset in ll.unroll(range(2)):
            group_col: ll.int32 = col + group_offset * 8
            group: ll.uint32 = ll.uint32(group_col >> 3)
            packed_scale: ll.uint32 = scale_lo2
            if group_col >= 32:
                packed_scale = scale_hi2
            inverse_values = ll.empty([4], dtype=ll.uint32, scope="local")
            src_addr: ll.uint32 = ll.smem_swizzle_128b_addr(l_base, ll.uint32(row), group)
            dst_addr: ll.uint32 = ll.smem_swizzle_128b_addr(d_base, ll.uint32(row), group)
            ri.smem_ld_128(sL, src_addr, inverse_values, 4)
            for pair in ll.unroll(range(4)):
                scale2: ll.uint32 = ll.__shfl_sync(ll.uint32(0xFFFFFFFF), packed_scale, (group_col + pair * 2) & 31)
                inverse_values[pair] = ll.mul_bf16x2(inverse_values[pair], scale2)
            ri.smem_st_128(sD, dst_addr, inverse_values[0], inverse_values[1], inverse_values[2], inverse_values[3])


@inline_func
def solve_role_fn(
    tid,
    num_pairs,
    seqlen,
    sG,
    sGInv,
    sBeta,
    sL0,
    sL1,
    sD,
    sP,
    tKKT,
    tQK,
    stage_gate,
    stage_kkt,
    stage_qk_acc,
    stage_inverse,
    stage_score,
) -> ll.void:
    local_tid: ll.int32 = tid - SOLVE_TID
    warp: ll.int32 = local_tid >> 5
    lane: ll.int32 = local_tid & 31
    half: ll.int32 = lane >> 4
    row_in_group: ll.int32 = lane & 15
    row: ll.int32 = warp * 16 + row_in_group
    dp_addr: ll.uint32 = ll.uint32(warp * 32) << 16
    qk_dp_addr: ll.uint32 = ll.uint32(warp * 32 + 16) << 16
    chunk: ll.int32 = 0
    while chunk < num_pairs * 2:
        stage_gate.bind(chunk)
        stage_kkt.bind(chunk)
        valid0: ll.int32 = BT if ASSUME_FULL_CHUNKS else ll.min_val(BT, seqlen - chunk * BT)
        kkt_epilogue_fn(tKKT, sL0, sBeta, dp_addr, valid0, row, half)
        decay0: ll.float32 = ri.smem_ld_f32(sG, BT - 1)
        scale0_lo: ll.float32 = decay0 * ri.smem_ld_f32(sGInv, lane) * ri.smem_ld_f32(sBeta, lane)
        scale0_hi: ll.float32 = decay0 * ri.smem_ld_f32(sGInv, lane + 32) * ri.smem_ld_f32(sBeta, lane + 32)
        scale0_lo2: ll.uint32 = ll.pack_bf16_float(scale0_lo,
                                                   ll.__shfl_sync(ll.uint32(0xFFFFFFFF), scale0_lo, (lane & ~1) + 1))
        scale0_hi2: ll.uint32 = ll.pack_bf16_float(scale0_hi,
                                                   ll.__shfl_sync(ll.uint32(0xFFFFFFFF), scale0_hi, (lane & ~1) + 1))

        stage_gate.bind(chunk + 1)
        stage_kkt.bind(chunk + 1)
        valid1: ll.int32 = BT if ASSUME_FULL_CHUNKS else ll.min_val(BT, seqlen - (chunk + 1) * BT)
        kkt_epilogue_fn(tKKT, sL1, sBeta, dp_addr, valid1, row, half)
        decay1: ll.float32 = ri.smem_ld_f32(sG, BT - 1)
        scale1_lo: ll.float32 = decay1 * ri.smem_ld_f32(sGInv, lane) * ri.smem_ld_f32(sBeta, lane)
        scale1_hi: ll.float32 = decay1 * ri.smem_ld_f32(sGInv, lane + 32) * ri.smem_ld_f32(sBeta, lane + 32)
        scale1_lo2: ll.uint32 = ll.pack_bf16_float(scale1_lo,
                                                   ll.__shfl_sync(ll.uint32(0xFFFFFFFF), scale1_lo, (lane & ~1) + 1))
        scale1_hi2: ll.uint32 = ll.pack_bf16_float(scale1_hi,
                                                   ll.__shfl_sync(ll.uint32(0xFFFFFFFF), scale1_hi, (lane & ~1) + 1))

        inv_blk64_pair_fn(sL0, sL1, local_tid)
        stage_inverse.bind(chunk)
        publish_inverse_fn(sL0, sD, scale0_lo2, scale0_hi2, local_tid)

        stage_score.bind(chunk)
        stage_qk_acc.bind(chunk)
        qk_epilogue_fn(tQK, sP, ll.float32(1.0) / decay0, qk_dp_addr, valid0, row, half)

        stage_inverse.bind(chunk + 1)
        publish_inverse_fn(sL1, sD, scale1_lo2, scale1_hi2, local_tid)

        stage_score.bind(chunk + 1)
        stage_qk_acc.bind(chunk + 1)
        qk_epilogue_fn(tQK, sP, ll.float32(1.0) / decay1, qk_dp_addr, valid1, row, half)

        chunk = chunk + 2


@inline_func
def mma_output_state_fn(
    tid,
    num_pairs,
    seqlen,
    sQK,
    sQKTrans,
    sD,
    sP,
    tState,
    tStateBF,
    tPred,
    tRhs,
    tVNew,
    tVNewBF,
    tO,
    stage_qk,
    stage_state_input,
    stage_pred,
    stage_inverse,
    stage_vnew,
    stage_vnew_bf,
    stage_score,
    stage_rhs,
    stage_state_fwd,
    stage_state_back,
    stage_tout,
) -> ll.void:
    issuer: ll.uint32 = ll.elect_one()
    lane: ll.int32 = tid - MMA1_TID
    pair: ll.int32 = 0
    while pair < num_pairs:
        stage_qk.bind(pair)
        for pair_chunk in ll.unroll(range(2)):
            chunk: ll.int32 = pair * 2 + pair_chunk
            valid: ll.int32 = BT if ASSUME_FULL_CHUNKS else ll.min_val(BT, seqlen - chunk * BT)
            kd: ll.uint64 = ri.make_umma_smem_desc(sQK, (pair_chunk * BT * 2, 0))
            qd: ll.uint64 = ri.make_umma_smem_desc(sQK, (pair_chunk * BT * 2 + BT, 0))
            stage_state_input.bind(chunk)
            stage_pred.bind(chunk)
            if issuer == 1:
                ri.umma_f16_ts_1sm_kblock(tPred, tStateBF, kd, D, out_shape=(D, BT), zero_init=True, negate_a=True)
            if issuer == 1:
                ri.umma_f16_ts_1sm_kblock(tO, tStateBF, qd, D, out_shape=(D, BT), zero_init=True)

            stage_rhs.bind(chunk)
            stage_inverse.bind(chunk)
            stage_vnew.bind(chunk)
            dd: ll.uint64 = ri.make_umma_smem_desc(sD, (0, 0))
            if issuer == 1:
                ri.umma_f16_ts_1sm_kblock(tVNew, tRhs, dd, BT, out_shape=(D, BT), zero_init=True)

            stage_score.bind(chunk)
            stage_state_fwd.bind(chunk)
            if not FUSE_STATE_VNEW_BF:
                stage_vnew_bf.bind(chunk)
            stage_tout.bind(chunk)
            if valid < BT:
                for token_row in ll.unroll(range(BT)):
                    if token_row >= valid:
                        for dim_block in ll.unroll(range(0, D, 32)):
                            dim: ll.int32 = dim_block + lane
                            ri.smem_st_bf16_k_major(sQK, BT * 4, pair_chunk * BT * 2 + token_row, dim,
                                                    ll.float_to_bf16(ll.float32(0.0)))
                ll.fence_async_shared()
                ll.__syncwarp()
            if issuer == 1:
                ri.umma_f16_ts_1sm_kblock(tO, tVNewBF, ri.make_umma_smem_desc(sP, (0, 0)), BT, out_shape=(D, BT))
            stage_state_back.bind(chunk)
            if issuer == 1:
                ri.umma_f16_ts_1sm_kblock(tState, tVNewBF, ri.make_umma_smem_desc(sQKTrans, (0, pair_chunk * BT * 2)),
                                          BT, out_shape=(D, D))
        pair = pair + 1


@inline_func
def store_role_fn(
    tid,
    dV,
    seq_start,
    head,
    num_pairs,
    sV,
    stage_v,
) -> ll.void:
    chunk: ll.int32 = 0
    while chunk < num_pairs * 2:
        issuer: ll.uint32 = ll.elect_one()
        stage_v.bind(chunk)
        if issuer == 1:
            token: ll.int32 = seq_start + chunk * BT
            ri.tma_ld_3d(sV, dV, (0, token, head), shape=(BT, D))
        chunk = chunk + 1


@lk.ll_kernel(backend="cuda", is_entry=True, frontend=dataflow_frontend)
def gdn_varlen_sm100_kernel(
    dQ: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dK: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dV: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dO: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dH0: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dHt: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    q: ll.ptr[ll.bfloat16],
    k: ll.ptr[ll.bfloat16],
    v: ll.ptr[ll.bfloat16],
    g_ptr: ll.ptr[ll.float32],
    beta_ptr: ll.ptr[ll.float32],
    h0: ll.ptr[ll.float32],
    o: ll.ptr[ll.bfloat16],
    ht: ll.ptr[ll.float32],
    cu_seqlens: ll.ptr[ll.int64],
    scale: ll.float32,
    segment_ready: ll.ptr[ll.int32],
    num_heads: ll.int32,
    num_segments: ll.int32,
    epoch: ll.int32,
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")

    with dataflow() as flow:
        state = flow.role("state", name="state.prepare", hw=HW.CUDACore, warps=STATE_WARPS)
        mma0 = flow.role("mma0", name="mma.qk_kk", hw=HW.TensorCore, warps=MMA0_WARPS)
        mma1 = flow.role("mma1", name="mma.output_state", hw=HW.TensorCore, warps=MMA1_WARPS)
        solve = flow.role("solve", name="triangular.solve_mask", hw=HW.CUDACore, warps=SOLVE_WARPS)
        io = flow.role("io", name="qkv.tma_gate.scan", hw=HW.TMA, warps=IO_WARPS)
        store = flow.role("store", name="output.tma", hw=HW.TMA, warps=STORE_WARPS)

        sQK = flow.smem("sQK", (BT * 4, D), dtype=ll.bfloat16, stages=QK_PAIR_STAGES, major="K")
        sQKTrans = flow.smem("sQKTrans", (D, BT * 4), dtype=ll.bfloat16, stages=QK_PAIR_STAGES, major="MN")
        sState = flow.smem("sState", (D * 4, 32), dtype=ll.float32)  # noqa: F841
        sStateView = flow.smem("sStateView", (D, D * 4), dtype=ll.uint8, major="K")
        sStateReady = flow.smem("sStateReady", (1, ), dtype=ll.uint64)
        sStateCopy = flow.smem("sStateCopy", (1, ), dtype=ll.uint64)
        sStateDone = flow.smem("sStateDone", (1, ), dtype=ll.uint64)
        sV = flow.smem("sV", (BT, D), dtype=ll.bfloat16, stages=V_STAGES, major="K")
        sL0 = flow.smem("sL0", (BT, BT), dtype=ll.bfloat16, major="K")
        sL1 = flow.smem("sL1", (BT, BT), dtype=ll.bfloat16, major="K")
        sD = flow.smem("sD", (BT, BT), dtype=ll.bfloat16, major="K")
        sP = flow.smem("sP", (BT, BT), dtype=ll.bfloat16, stages=SCORE_STAGES, major="K")
        sO = flow.smem("sO", (BT, D), dtype=ll.bfloat16, stages=OUTPUT_STAGES, major="K")
        sG = flow.smem("sG", (BT, ), dtype=ll.float32, stages=GATE_STAGES)
        sGInv = flow.smem("sGInv", (BT, ), dtype=ll.float32, stages=GATE_STAGES)
        sGOut = flow.smem("sGOut", (BT, ), dtype=ll.float32, stages=GATE_STAGES)
        sBeta = flow.smem("sBeta", (BT, ), dtype=ll.float32, stages=GATE_STAGES)
        tState = flow.tmem("tState", (D, D), dtype=ll.float32, copies=1)
        tStateBF = flow.tmem("tStateBF", (D, D), dtype=ll.bfloat16)
        tAcc = flow.tmem("tAcc", (BT * 2, BT), dtype=ll.float32, stages=2)  # noqa: F841
        tKKT = flow.tmem("tKKT", (BT, BT), dtype=ll.float32, stages=2)
        tQK = flow.tmem("tQK", (BT, BT), dtype=ll.float32, stages=2)
        tPred = flow.tmem("tPred", (D, BT), dtype=ll.float32)
        tVNew = flow.tmem("tVNew", (D, BT), dtype=ll.float32)
        if ALIAS_VNEW_BF_WITH_STATE_BF:
            tRhs = flow.tmem("tRhs", (D, BT), dtype=ll.bfloat16, stages=2)
            tVNewBF = flow.tmem("tVNewBF", (D, BT), dtype=ll.bfloat16)
        else:
            tRhs = flow.tmem("tRhs", (D, BT), dtype=ll.bfloat16, stages=2)
            tVNewBF = flow.tmem("tVNewBF", (D, BT), dtype=ll.bfloat16, stages=2)
        tO = flow.tmem("tO", (D, BT), dtype=ll.float32)

        flow.memory_mapping_plan = [
            MemoryMappingEntry("tVNew", "tPred", space="tmem"),
            MemoryMappingEntry("sQKTrans", "sQK", space="smem"),
            MemoryMappingEntry("sState", "sQK", space="smem", extent=0),
            MemoryMappingEntry("sStateView", "sQK", space="smem", extent=0),
            MemoryMappingEntry("tKKT", "tAcc", space="tmem", extent=0),
            MemoryMappingEntry("tQK", "tAcc", space="tmem", extent=0),
        ]
        if ALIAS_VNEW_BF_WITH_STATE_BF:
            # StateBF is dead before VNewBF is published, leaving a two-stage RHS ring within 512 TMEM columns.
            flow.memory_mapping_plan.append(
                MemoryMappingEntry("tVNewBF", "tStateBF", space="tmem", extent=BT // 2, stage_stride=D // 2))
        else:
            flow.memory_mapping_plan.append(MemoryMappingEntry("tVNewBF", "tRhs", space="tmem"))

        stage_qk = flow.depends(src=io, dst=[mma0, mma1], via=[sQK])
        stage_v = flow.depends(src=store, dst=state, via=[sV])
        stage_gate = flow.depends(src=io, dst=[state, solve], via=[sG, sGInv, sGOut, sBeta])
        stage_state_input = flow.depends(src=state, dst=mma1, via=[tStateBF])
        stage_pred = flow.depends(src=mma1, dst=state, via=[tPred])
        stage_kkt = flow.depends(src=mma0, dst=solve, via=[tKKT])
        stage_qk_acc = flow.depends(src=mma0, dst=solve, via=[tQK])
        stage_inverse = flow.depends(src=solve, dst=mma1, via=[sD])
        stage_score = flow.depends(src=solve, dst=mma1, via=[sP])
        stage_vnew = flow.depends(src=mma1, dst=state, via=[tVNew])
        stage_rhs = flow.depends(src=state, dst=mma1, via=[tRhs])
        if FUSE_STATE_VNEW_BF:
            stage_state_fwd = flow.depends(src=state, dst=mma1, via=[tState, tVNewBF])
            stage_vnew_bf = stage_state_fwd
        else:
            stage_state_fwd = flow.depends(src=state, dst=mma1, via=[tState])
            stage_vnew_bf = flow.depends(src=state, dst=mma1, via=[tVNewBF])
        stage_state_back = flow.depends(src=mma1, dst=state, via=[tState])
        stage_tout = flow.depends(src=mma1, dst=state, via=[tO])
        stage_output = flow.depends(src=state, dst=state, via=[sO])

    tid: ll.int32 = ll.threadIdx_x()
    if tid == 0:
        ll.init_smem_barrier(sStateReady, 1)
        ll.init_smem_barrier(sStateCopy, 1)
        ll.init_smem_barrier(sStateDone, 1)
        ll.fence_smem_barrier_init()
    ll.__syncthreads()
    state_phase: ll.uint32 = ll.uint32(0)
    copy_phase: ll.uint32 = ll.uint32(0)
    state_done_phase: ll.uint32 = ll.uint32(0)
    task: ll.int32 = ll.int32(ll.blockIdx_x())
    task_stride: ll.int32 = ll.int32(ll.gridDim_x())
    while task < num_heads * num_segments:
        segment: ll.int32 = ll.int32(0) if ASSUME_SINGLE_SEGMENT else task // num_heads
        work: ll.int32 = task if ASSUME_SINGLE_SEGMENT else task - segment * num_heads
        seq: ll.int32 = work // H
        head: ll.int32 = work - seq * H
        seq_base: ll.int32 = ll.int32(cu_seqlens[seq])
        seq_end: ll.int32 = ll.int32(cu_seqlens[seq + 1])
        total_chunks: ll.int32 = (seq_end - seq_base) // BT if ASSUME_FULL_CHUNKS else (seq_end - seq_base + BT -
                                                                                        1) // BT
        total_pairs: ll.int32 = total_chunks // 2 if ASSUME_FULL_CHUNKS else (total_chunks + 1) // 2
        segment_capacity: ll.int32 = total_pairs if ASSUME_SINGLE_SEGMENT else (total_pairs + num_segments -
                                                                                1) // num_segments
        pair_start: ll.int32 = ll.int32(0) if ASSUME_SINGLE_SEGMENT else segment * segment_capacity
        num_pairs: ll.int32 = total_pairs if ASSUME_SINGLE_SEGMENT else ll.max_val(
            0, ll.min_val(segment_capacity, total_pairs - pair_start))
        seq_start: ll.int32 = seq_base + pair_start * 2 * BT
        seqlen: ll.int32 = seq_end - seq_base if ASSUME_SINGLE_SEGMENT else ll.min_val(
            num_pairs * 2 * BT, seq_end - seq_start)
        state_input: ll.ptr[ll.float32] = h0
        if not ASSUME_SINGLE_SEGMENT and segment != 0:
            if tid == 0:
                while ll.ld_acquire_gpu(segment_ready + work) != epoch * num_segments + segment:
                    pass
            ll.__syncthreads()
            state_input = ht

        if tid == 0:
            ll.mbarrier_arrive_and_expect_tx(sStateReady, D * D * 4)
            for state_tile in ll.unroll(range(4)):
                if ASSUME_SINGLE_SEGMENT or segment == 0:
                    ll.tma_load_3d_1sm(dH0, sStateReady, ll.ptr_byte_offset(sStateView, state_tile * D * 32 * 4),
                                       state_tile * 32, 0, work)
                else:
                    ll.tma_load_3d_1sm(dHt, sStateReady, ll.ptr_byte_offset(sStateView, state_tile * D * 32 * 4),
                                       state_tile * 32, 0, work)
            ll.mbarrier_wait(sStateReady, state_phase)
            for state_tile in ll.unroll(range(4)):
                for state_subtile in ll.unroll(range(4)):
                    state_desc: ll.uint64 = ri.make_umma_smem_desc(sStateView,
                                                                   (0, state_tile * 128 + state_subtile * 32))
                    ll.tcgen05_cp_128x256b_1sm(ll.uint32(tState + state_tile * 32 + state_subtile * 8), state_desc)
            ll.umma_commit_1sm(sStateCopy)
            ll.mbarrier_wait(sStateCopy, copy_phase)
            state_phase = state_phase ^ ll.uint32(1)
            copy_phase = copy_phase ^ ll.uint32(1)
        ll.__syncthreads()
        ll.tcgen05_fence_after()

        with io:
            tma_ld_qkv_g_fn(tid, dQ, dK, seq_start, seqlen, head, num_pairs, scale, g_ptr, beta_ptr, sQK, sG, sGInv,
                            sGOut, sBeta, stage_qk, stage_gate)
        with mma0:
            mma_qk_kk_fn(num_pairs, sQK, tKKT, tQK, stage_qk, stage_kkt, stage_qk_acc)
        with state:
            state_role_fn(
                tid,
                dO,
                state_input,
                ht,
                o,
                seq,
                seq_start,
                seqlen,
                head,
                num_pairs,
                scale,
                tState,
                tStateBF,
                tPred,
                tRhs,
                tVNew,
                tVNewBF,
                tO,
                sV,
                sG,
                sGInv,
                sGOut,
                sO,
                stage_v,
                stage_gate,
                stage_state_input,
                stage_pred,
                stage_vnew,
                stage_vnew_bf,
                stage_rhs,
                stage_state_fwd,
                stage_state_back,
                stage_tout,
                stage_output,
                sStateDone,
            )
        with mma1:
            mma_output_state_fn(
                tid,
                num_pairs,
                seqlen,
                sQK,
                sQKTrans,
                sD,
                sP,
                tState,
                tStateBF,
                tPred,
                tRhs,
                tVNew,
                tVNewBF,
                tO,
                stage_qk,
                stage_state_input,
                stage_pred,
                stage_inverse,
                stage_vnew,
                stage_vnew_bf,
                stage_score,
                stage_rhs,
                stage_state_fwd,
                stage_state_back,
                stage_tout,
            )
        with solve:
            solve_role_fn(
                tid,
                num_pairs,
                seqlen,
                sG,
                sGInv,
                sBeta,
                sL0,
                sL1,
                sD,
                sP,
                tKKT,
                tQK,
                stage_gate,
                stage_kkt,
                stage_qk_acc,
                stage_inverse,
                stage_score,
            )
        with store:
            store_role_fn(tid, dV, seq_start, head, num_pairs, sV, stage_v)

        if (tid & 31) == 0:
            ll.mbarrier_wait(sStateDone, state_done_phase)
        # The branch reconverges each warp before its per-thread TMEM proxy fence below.
        state_done_phase = state_done_phase ^ ll.uint32(1)
        ll.tcgen05_fence_after()
        stage_state_to_smem_all_fn(tState, sStateView, tid)
        ll.__syncthreads()
        if tid == 0:
            ll.fence_async_shared()
            for state_tile in ll.unroll(range(4)):
                ll.tma_store_3d(dHt, ll.ptr_byte_offset(sStateView, state_tile * D * 32 * 4), state_tile * 32, 0, work)
            ll.tma_store_commit()
            ll.tma_store_wait_group(0)
            if not ASSUME_SINGLE_SEGMENT:
                ll.st_release_gpu(segment_ready + work, epoch * num_segments + segment + 1)
        ll.__syncthreads()
        task = task + task_stride


@dataclass(frozen=True)
class GDNVarlenV2Kernels:
    ragged: object
    full_single_segment: object


class GDNVarlenV2:
    """Minimal FLA-shaped launcher for the fixed SM100 kernel."""

    @classmethod
    def build(cls, *, cache_dir: str | None = None, verbose: bool = False):
        from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
        from little_kernel.runtime import CompiledKernel

        source = Path(__file__).read_text(encoding="utf-8")

        def compile_variant(assume_full_chunks: bool, assume_single_segment: bool, alias_vnew_bf_with_state_bf: bool):
            code = compile_dataflow_cuda(
                source,
                {
                    "ASSUME_FULL_CHUNKS": int(assume_full_chunks),
                    "ASSUME_SINGLE_SEGMENT": int(assume_single_segment),
                    "ALIAS_VNEW_BF_WITH_STATE_BF": int(alias_vnew_bf_with_state_bf),
                    "FUSE_STATE_VNEW_BF": int(assume_full_chunks),
                    "V_STAGES": 1,
                    "OUTPUT_STAGES": 2,
                    "GATE_STAGES": 5,
                    "SCORE_STAGES": 2,
                },
                num_threads=NUM_THREADS,
                disable_var_hoisting=True,
            )
            return CompiledKernel(
                code,
                "gdn_varlen_sm100_kernel",
                grid=(1, 1, 1),
                block=(NUM_THREADS, 1, 1),
                shared_mem_bytes=SMEM_TOTAL,
                arch="sm_100a",
                cache_dir=cache_dir,
                verbose=verbose,
                include_paths=[Path(__file__).resolve().parents[2]],
                extra_nvcc_options=["-DNDEBUG", "--use_fast_math", "--maxrregcount=192", "-lineinfo"],
            )

        return GDNVarlenV2Kernels(compile_variant(False, False, False), compile_variant(True, True, True))

    @classmethod
    def create(
        cls,
        compiled_kernel,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None = None,
        initial_state: torch.Tensor,
        output_final_state: bool = True,
        cu_seqlens: torch.Tensor,
        head_first: bool = False,
        state_v_first: bool = True,
    ) -> "GDNVarlenV2":
        import torch
        if head_first or not state_v_first or not output_final_state:
            raise ValueError("v2_gdn supports only head_first=False, state_v_first=True, output_final_state=True")
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[2:] != (H, D):
            raise ValueError(f"q/k/v must be [1, T, {H}, {D}]")
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q, k, and v must have identical shapes")
        if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError("q, k, and v must be BF16")
        if g.shape != q.shape[:3] or beta.shape != q.shape[:3]:
            raise ValueError("g and beta must be [1, T, H]")
        if g.dtype != torch.float32 or beta.dtype != torch.float32:
            raise ValueError("g and beta must be FP32")
        if cu_seqlens.dtype != torch.int64 or cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a one-dimensional int64 tensor")
        nseq = cu_seqlens.numel() - 1
        if initial_state.shape != (nseq, H, D, D) or initial_state.dtype != torch.float32:
            raise ValueError(f"initial_state must be FP32 [{nseq}, {H}, {D}, {D}] in V-first layout")
        tensors = (q, k, v, g, beta, initial_state, cu_seqlens)
        if not all(t.is_cuda and t.is_contiguous() for t in tensors):
            raise ValueError("all inputs must be contiguous CUDA tensors")
        cu_cpu = cu_seqlens.cpu()
        if int(cu_cpu[0]) != 0 or int(cu_cpu[-1]) != q.shape[1]:
            raise ValueError("cu_seqlens must start at zero and end at total_T")
        lengths = cu_cpu[1:] - cu_cpu[:-1]
        if bool(torch.any(lengths <= 0)):
            raise ValueError("v2_gdn does not support empty sequences")
        out = torch.empty_like(q)
        final_state = torch.empty_like(initial_state)
        segment_ready = torch.zeros(nseq * H, device=q.device, dtype=torch.int32)
        from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_3d_descriptor

        def make_descriptor(tensor, *, oob_fill=True):
            return create_tma_3d_descriptor(
                tensor,
                dim0=D,
                dim1=q.shape[1],
                dim2=H,
                stride1_bytes=H * D * 2,
                stride2_bytes=D * 2,
                tile_dim0=64,
                tile_dim1=BT,
                swizzle_mode=128,
                oob_fill=oob_fill,
                l2_promotion=3,
            )

        def make_state_descriptor(tensor, *, oob_fill=True):
            return create_tma_3d_descriptor(
                tensor,
                dim0=D,
                dim1=D,
                dim2=nseq * H,
                stride1_bytes=D * 4,
                stride2_bytes=D * D * 4,
                tile_dim0=32,
                tile_dim1=D,
                swizzle_mode=128,
                oob_fill=oob_fill,
                l2_promotion=3,
            )

        descriptors = tuple(make_descriptor(tensor) for tensor in (q, k, v)) + (
            make_descriptor(out, oob_fill=False),
            make_state_descriptor(initial_state),
            make_state_descriptor(final_state, oob_fill=False),
        )
        actual_scale = float(scale if scale is not None else 1.0 / math.sqrt(D))
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
        max_chunks = (int(lengths.max()) + BT - 1) // BT
        num_segments = LONG_SEQUENCE_SEGMENTS if nseq * H < sm_count * 4 and max_chunks >= 64 else 1
        if num_segments == 1:
            persistent_waves = 2 if nseq * H <= sm_count * 8 else 3
        else:
            persistent_waves = max(1, math.ceil(nseq * H / (sm_count * PERSISTENT_TASKS_PER_CTA)))
        num_ctas = min(nseq * H, persistent_waves * sm_count) if max_chunks >= 64 else nseq * H
        if isinstance(compiled_kernel, GDNVarlenV2Kernels):
            full_single_segment = num_segments == 1 and bool(torch.all(lengths % (BT * 2) == 0))
            compiled_kernel = compiled_kernel.full_single_segment if full_single_segment else compiled_kernel.ragged
        return cls(
            compiled_kernel,
            descriptors,
            (q, k, v, g, beta, initial_state, out, final_state, cu_seqlens, actual_scale),
            out,
            final_state,
            nseq,
            segment_ready,
            num_ctas,
            num_segments,
        )

    def __init__(
        self,
        compiled_kernel,
        descriptors,
        args,
        output,
        final_state,
        nseq,
        segment_ready,
        num_ctas,
        num_segments,
    ):
        self.compiled_kernel = compiled_kernel
        self.descriptors = descriptors
        self.args = args
        self.output = output
        self.final_state = final_state
        self.nseq = nseq
        self.segment_ready = segment_ready
        self.num_ctas = num_ctas
        self.num_segments = num_segments
        self.epoch = 0

    def launch(self, stream=None):
        self.epoch += 1
        self.compiled_kernel(*self.descriptors, *self.args, self.segment_ready, self.nseq * H, self.num_segments,
                             self.epoch, grid=(self.num_ctas, 1, 1), stream=stream)
        return self.output, self.final_state

    def __call__(self, stream=None):
        return self.launch(stream)


__all__ = ["GDNVarlenV2", "gdn_varlen_sm100_kernel"]
