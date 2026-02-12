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
"""
FlashComm combine kernels:
- kernel_combine_preprocess_inplace
- kernel_combine_intranode

Fixed config: bf16 tokens, float32 weights, int32 offsets
kHiddenSize=7168
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"
WARP_SIZE = 32
HIDDEN_SIZE = 7168
TOPK = 8
NUM_WARPS_PREPROCESS = 32
BLOCK_SIZE_PREPROCESS = NUM_WARPS_PREPROCESS * WARP_SIZE  # 1024


# ============================================================================
# kernel_combine_preprocess_inplace
# For each received token, check which topk slots are valid.
# If multiple slots scattered from same source, reduce (sum) bf16 data.
# Simplified: uses per-thread scalar loop instead of int4 vectorization.
# ============================================================================
@ll_kernel(backend=backend, is_entry=True)
def kernel_combine_preprocess_inplace(
    x: ll.Tensor[ll.bfloat16],  # [num_recv_worst_token, hidden_size]
    weight: ll.Tensor[ll.float32],  # [num_recv_worst_token]
    recv_token_count: ll.Tensor[ll.int32],  # [num_ranks]
    recv_topk_scatter_indices: ll.Tensor[ll.int32],  # [num_recv_worst_token, topk]
    recv_topk_weight: ll.Tensor[ll.float32],  # [num_recv_worst_token, topk]
    topk: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    lane_id: ll.int32 = thread_id % WARP_SIZE

    ll.__syncthreads()
    num_recv_token: ll.int32 = recv_token_count[rank]

    i: ll.int32 = block_id
    while i < num_recv_token:
        # Check validity of each topk slot
        is_valid_lane: ll.int32 = 0
        if lane_id < topk:
            dst_idx: ll.int32 = recv_topk_scatter_indices[i * topk + lane_id]
            if dst_idx != -1:
                is_valid_lane = 1

        ll.__syncwarp()
        valid_mask: ll.uint32 = ll.ballot_sync(0xFFFFFFFF, is_valid_lane)
        reduce_cnt: ll.int32 = ll.popc(valid_mask)

        # Scatter weight to per-destination weight buffer
        if lane_id < topk:
            dst_idx2: ll.int32 = recv_topk_scatter_indices[i * topk + lane_id]
            if dst_idx2 != -1:
                w_val: ll.float32 = weight[dst_idx2]
                recv_topk_weight[dst_idx2 * topk + lane_id] = w_val

        if reduce_cnt > 1:
            # Multiple valid slots: need to reduce (accumulate) bf16 data
            # Each thread handles a portion of hidden_size
            elem_idx: ll.int32 = thread_id
            while elem_idx < HIDDEN_SIZE:
                acc: ll.float32 = 0.0
                j: ll.int32 = 0
                while j < topk:
                    dst_idx3: ll.int32 = recv_topk_scatter_indices[i * topk + j]
                    if dst_idx3 != -1:
                        src_val: ll.bfloat16 = x[dst_idx3 * HIDDEN_SIZE + elem_idx]
                        acc = acc + ll.bf16_to_float(src_val)
                    j = j + 1
                # Write reduced result to position i
                x[i * HIDDEN_SIZE + elem_idx] = ll.float_to_bf16(acc)
                elem_idx = elem_idx + BLOCK_SIZE_PREPROCESS
        ll.__syncthreads()
        i = i + num_block


# ============================================================================
# kernel_combine_intranode
# Producer-consumer TMA pipeline for combine phase.
# Each consumer warp processes one topk slot, loads remote token data via TMA,
# and accumulates into local buffer.
# ============================================================================

NUM_LOAD_STAGES = 2
NUM_STORE_STAGES = 0  # v0: no store pipeline, direct accumulation


@ll_kernel(backend=backend, is_entry=True)
def kernel_combine_intranode(
    x_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],  # [num_ranks] -> [num_token, hidden_size]
    weight_ptrs: ll.ptr[ll.ptr[ll.float32]],  # [num_ranks] -> [num_token, topk]
    topk_send_mask: ll.Tensor[ll.int32],  # [num_token, topk]
    topk_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    token_dst_scatter_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    recv_x: ll.Tensor[ll.bfloat16],  # [num_recv_worst_token, hidden_size]
    recv_weight: ll.Tensor[ll.float32],  # [num_recv_worst_token, topk]
    num_token: ll.int32,
    hidden_size: ll.int32,
    topk: ll.int32,
    num_experts_per_rank: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE

    # Shared memory
    ll.align_memory(1024, scope="dynamic_shared")
    mbar_full = ll.empty([NUM_LOAD_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty = ll.empty([NUM_LOAD_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    tma_load_buffer = ll.empty([NUM_LOAD_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    is_producer_warp: ll.int32 = 1 if warp_id == TOPK else 0
    is_consumer_warp: ll.int32 = 1 if warp_id < TOPK else 0
    response_expert_idx: ll.int32 = warp_id % TOPK

    if warp_id == 0:
        if ll.elect_one_sync():
            for i_s in range(NUM_LOAD_STAGES):
                ll.init_smem_barrier(mbar_full + i_s, 1)
            for i_s2 in range(NUM_LOAD_STAGES):
                ll.init_smem_barrier(mbar_empty + i_s2, TOPK)

    ll.__syncthreads()

    num_bytes_per_token: ll.int32 = hidden_size * 2  # bf16

    # ---- Producer: load tokens from local buffer ----
    if is_producer_warp == 1:
        prod_idx: ll.int32 = 0
        prod_phase: ll.int32 = 1
        token_offset: ll.int32 = block_id
        while token_offset < num_token:
            ll.mbarrier_wait(mbar_empty + prod_idx, prod_phase)
            if ll.elect_one_sync():
                # Load token from recv_x at this token's position
                ll.tma_copy_1d_g2s(
                    recv_x + token_offset * hidden_size,
                    mbar_full + prod_idx,
                    tma_load_buffer + prod_idx * HIDDEN_SIZE,
                    num_bytes_per_token,
                )
                ll.mbarrier_arrive_and_expect_tx(mbar_full + prod_idx, num_bytes_per_token)
            ll.__syncwarp()
            prod_idx = prod_idx + 1
            if prod_idx == NUM_LOAD_STAGES:
                prod_idx = 0
                prod_phase = prod_phase ^ 1
            token_offset = token_offset + num_block

    # ---- Consumer: scatter token data to remote ranks ----
    if is_consumer_warp == 1:
        cons_idx: ll.int32 = 0
        cons_phase: ll.int32 = 0
        is_leader: ll.int32 = ll.elect_one_sync()
        token_offset2: ll.int32 = block_id

        while token_offset2 < num_token:
            expert_idx: ll.int32 = topk_indices[token_offset2 * TOPK + response_expert_idx]
            target_rank: ll.int32 = expert_idx / num_experts_per_rank
            is_need_send: ll.int32 = topk_send_mask[token_offset2 * TOPK + response_expert_idx]

            ll.mbarrier_wait(mbar_full + cons_idx, cons_phase)

            if target_rank < num_ranks:
                if is_need_send == 1:
                    store_idx: ll.int32 = token_dst_scatter_indices[token_offset2 * TOPK + response_expert_idx]
                    # TMA store to remote rank
                    if is_leader:
                        ll.tma_copy_1d_s2g(
                            tma_load_buffer + cons_idx * HIDDEN_SIZE,
                            x_ptrs[target_rank] + store_idx * hidden_size,
                            num_bytes_per_token,
                        )
                        ll.tma_store_arrive()

            ll.tma_store_wait_n(0)
            ll.__syncwarp()
            if is_leader:
                ll.mbarrier_arrive(mbar_empty + cons_idx)
            ll.__syncwarp()
            cons_idx = cons_idx + 1
            if cons_idx == NUM_LOAD_STAGES:
                cons_idx = 0
                cons_phase = cons_phase ^ 1
            token_offset2 = token_offset2 + num_block


# ============================================================================
# kernel_combine_intranode_v1
# Optimized combine: producer uses ballot/shuffle to iterate valid slots.
# Consumer does scalar bf16→float accumulation + TMA store back.
# ============================================================================

V1_NUM_LOAD_STAGES = 2
V1_NUM_STORE_STAGES = 1
V1_WARPS_PER_WG = 4  # 1 producer + 3 consumer warps per warp group
V1_NUM_WG = 4  # 4 warp groups => 16 warps total


@ll_kernel(backend=backend, is_entry=True)
def kernel_combine_intranode_v1(
    x_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],  # [num_ranks]
    topk_send_mask: ll.Tensor[ll.int32],  # [num_token, topk]
    topk_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    token_dst_scatter_indices: ll.Tensor[ll.int32],
    recv_x: ll.Tensor[ll.bfloat16],  # [num_recv_worst_token, hidden_size]
    num_token: ll.int32,
    num_experts_per_rank: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE
    lane_id: ll.int32 = thread_id % WARP_SIZE

    # Shared memory
    ll.align_memory(1024, scope="dynamic_shared")
    # Per-warp-group TMA barriers: [NUM_WG * NUM_LOAD_STAGES] full/empty
    mbar_full = ll.empty([V1_NUM_WG * V1_NUM_LOAD_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty = ll.empty([V1_NUM_WG * V1_NUM_LOAD_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    # Per-warp-group TMA load buffers
    tma_load_buf = ll.empty([V1_NUM_WG * V1_NUM_LOAD_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")
    # Per-warp-group TMA store buffer (1 per WG)
    tma_store_buf = ll.empty([V1_NUM_WG * V1_NUM_STORE_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    wg_id: ll.int32 = warp_id / V1_WARPS_PER_WG
    warp_in_wg: ll.int32 = warp_id % V1_WARPS_PER_WG
    global_wg_id: ll.int32 = block_id * V1_NUM_WG + wg_id
    total_wg: ll.int32 = num_block * V1_NUM_WG
    is_producer: ll.int32 = 1 if warp_in_wg == 0 else 0
    num_bytes_per_token: ll.int32 = HIDDEN_SIZE * 2

    # Init barriers
    if warp_in_wg == 0:
        if ll.elect_one_sync():
            for i_s in range(V1_NUM_LOAD_STAGES):
                ll.init_smem_barrier(mbar_full + wg_id * V1_NUM_LOAD_STAGES + i_s, 1)
                ll.init_smem_barrier(mbar_empty + wg_id * V1_NUM_LOAD_STAGES + i_s, V1_WARPS_PER_WG - 1)

    ll.__syncthreads()

    # ---- Producer warp: uses ballot/shuffle for efficient TMA loads ----
    if is_producer == 1:
        prod_idx: ll.int32 = 0
        prod_phase: ll.int32 = 1
        wg_mbar_full: ll.ptr[ll.uint64] = mbar_full + wg_id * V1_NUM_LOAD_STAGES
        wg_mbar_empty: ll.ptr[ll.uint64] = mbar_empty + wg_id * V1_NUM_LOAD_STAGES
        wg_load_buf: ll.ptr[ll.bfloat16] = tma_load_buf + wg_id * V1_NUM_LOAD_STAGES * HIDDEN_SIZE

        token_offset: ll.int32 = global_wg_id
        while token_offset < num_token:
            # Each lane < TOPK loads its expert info
            expert_rank_lane: ll.int32 = 0
            scatter_idx_lane: ll.int32 = 0
            is_valid_lane: ll.int32 = 0
            if lane_id < TOPK:
                eidx: ll.int32 = topk_indices[token_offset * TOPK + lane_id]
                expert_rank_lane = eidx / num_experts_per_rank
                is_need: ll.int32 = topk_send_mask[token_offset * TOPK + lane_id]
                scatter_idx_lane = token_dst_scatter_indices[token_offset * TOPK + lane_id]
                if expert_rank_lane < num_ranks:
                    if is_need == 1:
                        is_valid_lane = 1
            ll.__syncwarp()

            valid_mask: ll.uint32 = ll.ballot_sync(0xFFFFFFFF, is_valid_lane)
            remaining: ll.uint32 = valid_mask
            while remaining != 0:
                j: ll.int32 = ll.ffs(remaining) - 1
                remaining = remaining & (remaining - 1)
                erank: ll.int32 = ll.__shfl_sync(0xFFFFFFFF, expert_rank_lane, j)
                sidx: ll.int32 = ll.__shfl_sync(0xFFFFFFFF, scatter_idx_lane, j)

                ll.mbarrier_wait(wg_mbar_empty + prod_idx, prod_phase)
                if ll.elect_one_sync():
                    ll.mbarrier_arrive_and_expect_tx(wg_mbar_full + prod_idx, num_bytes_per_token)
                    ll.tma_copy_1d_g2s(
                        x_ptrs[erank] + sidx * HIDDEN_SIZE,
                        wg_mbar_full + prod_idx,
                        wg_load_buf + prod_idx * HIDDEN_SIZE,
                        num_bytes_per_token,
                    )
                prod_idx = prod_idx + 1
                if prod_idx == V1_NUM_LOAD_STAGES:
                    prod_idx = 0
                    prod_phase = prod_phase ^ 1
                ll.__syncwarp()
            token_offset = token_offset + total_wg

    # ---- Consumer warps: scalar accumulate + write back ----
    if is_producer == 0:
        if wg_id < V1_NUM_WG:
            cons_idx: ll.int32 = 0
            cons_phase: ll.int32 = 0
            wg_mbar_full2: ll.ptr[ll.uint64] = mbar_full + wg_id * V1_NUM_LOAD_STAGES
            wg_mbar_empty2: ll.ptr[ll.uint64] = mbar_empty + wg_id * V1_NUM_LOAD_STAGES
            wg_load_buf2: ll.ptr[ll.bfloat16] = tma_load_buf + wg_id * V1_NUM_LOAD_STAGES * HIDDEN_SIZE
            wg_store_buf: ll.ptr[ll.bfloat16] = tma_store_buf + wg_id * V1_NUM_STORE_STAGES * HIDDEN_SIZE
            is_leader2: ll.int32 = ll.elect_one_sync()
            consumer_threads: ll.int32 = (V1_WARPS_PER_WG - 1) * WARP_SIZE
            cons_tid: ll.int32 = (thread_id % (WARP_SIZE * V1_WARPS_PER_WG)) - WARP_SIZE

            token_offset2: ll.int32 = global_wg_id
            while token_offset2 < num_token:
                # Count valid topk slots
                is_valid2: ll.int32 = 0
                if lane_id < TOPK:
                    eidx2: ll.int32 = topk_indices[token_offset2 * TOPK + lane_id]
                    erank2: ll.int32 = eidx2 / num_experts_per_rank
                    is_need2: ll.int32 = topk_send_mask[token_offset2 * TOPK + lane_id]
                    if erank2 < num_ranks:
                        if is_need2 == 1:
                            is_valid2 = 1
                valid_mask2: ll.uint32 = ll.ballot_sync(0xFFFFFFFF, is_valid2)
                valid_count: ll.int32 = ll.popc(valid_mask2)

                # Accumulate from each valid TMA-loaded buffer
                j2: ll.int32 = 0
                while j2 < valid_count:
                    ll.mbarrier_wait(wg_mbar_full2 + cons_idx, cons_phase)
                    # Scalar reduce: each consumer thread handles a range of elements
                    elem: ll.int32 = cons_tid
                    while elem < HIDDEN_SIZE:
                        src_val: ll.bfloat16 = wg_load_buf2[cons_idx * HIDDEN_SIZE + elem]
                        # Accumulate into store buffer (first iteration init, later add)
                        if j2 == 0:
                            wg_store_buf[elem] = src_val
                        else:
                            old_val: ll.bfloat16 = wg_store_buf[elem]
                            wg_store_buf[elem] = ll.float_to_bf16(ll.bf16_to_float(old_val) + ll.bf16_to_float(src_val))
                        elem = elem + consumer_threads
                    ll.__syncwarp()
                    if is_leader2:
                        ll.mbarrier_arrive(wg_mbar_empty2 + cons_idx)
                    cons_idx = cons_idx + 1
                    if cons_idx == V1_NUM_LOAD_STAGES:
                        cons_idx = 0
                        cons_phase = cons_phase ^ 1
                    j2 = j2 + 1

                # TMA store accumulated result back to recv_x
                ll.fence_async_shared()
                ll.named_barrier_sync(wg_id + 1, consumer_threads)
                if warp_in_wg == 1:
                    if is_leader2:
                        ll.tma_copy_1d_s2g(
                            wg_store_buf,
                            recv_x + token_offset2 * HIDDEN_SIZE,
                            num_bytes_per_token,
                        )
                        ll.tma_store_arrive()
                    ll.__syncwarp()

                token_offset2 = token_offset2 + total_wg

    ll.tma_store_wait_n(0)
    ll.__syncthreads()


# ============================================================================
# kernel_combine_intranode_v2
# Adds weight support: extra warp combines weights from remote ranks.
# ============================================================================
@ll_kernel(backend=backend, is_entry=True)
def kernel_combine_intranode_v2(
    x_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],
    weight_ptrs: ll.ptr[ll.ptr[ll.float32]],
    topk_send_mask: ll.Tensor[ll.int32],
    topk_indices: ll.Tensor[ll.int32],
    token_dst_scatter_indices: ll.Tensor[ll.int32],
    recv_x: ll.Tensor[ll.bfloat16],
    recv_weight: ll.Tensor[ll.float32],
    num_token: ll.int32,
    num_experts_per_rank: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE
    lane_id: ll.int32 = thread_id % WARP_SIZE

    # Weight combining: last warp handles all weights
    NUM_WARPS_TOTAL: ll.int32 = 24  # 768 threads / 32
    if warp_id == NUM_WARPS_TOTAL - 1:
        total_weight_threads: ll.int32 = num_block * WARP_SIZE
        global_wt: ll.int32 = lane_id + block_id * WARP_SIZE
        num_weights: ll.int32 = num_token * TOPK
        while global_wt < num_weights:
            tok_off: ll.int32 = global_wt / TOPK
            topk_idx: ll.int32 = global_wt % TOPK
            eidx: ll.int32 = topk_indices[tok_off * TOPK + topk_idx]
            sidx: ll.int32 = token_dst_scatter_indices[tok_off * TOPK + topk_idx]
            erank: ll.int32 = eidx / num_experts_per_rank
            valid_w: ll.float32 = 0.0
            if erank < num_ranks:
                if sidx != -1:
                    valid_w = weight_ptrs[erank][sidx * TOPK + topk_idx]
            recv_weight[global_wt] = valid_w
            global_wt = global_wt + total_weight_threads
        return

    # Remaining warps: same TMA pipeline as v1 for token data
    # (Simplified: use same pipeline structure as basic combine)
    ll.align_memory(1024, scope="dynamic_shared")
    mbar_full3 = ll.empty([NUM_LOAD_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty3 = ll.empty([NUM_LOAD_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    tma_load_buf3 = ll.empty([NUM_LOAD_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    is_producer3: ll.int32 = 1 if warp_id == TOPK else 0
    is_consumer3: ll.int32 = 1 if warp_id < TOPK else 0
    response_idx: ll.int32 = warp_id % TOPK

    if warp_id == 0:
        if ll.elect_one_sync():
            for i_s3 in range(NUM_LOAD_STAGES):
                ll.init_smem_barrier(mbar_full3 + i_s3, 1)
            for i_s4 in range(NUM_LOAD_STAGES):
                ll.init_smem_barrier(mbar_empty3 + i_s4, TOPK)

    ll.__syncthreads()
    num_bytes3: ll.int32 = HIDDEN_SIZE * 2

    if is_producer3 == 1:
        prod3_idx: ll.int32 = 0
        prod3_phase: ll.int32 = 1
        tok3: ll.int32 = block_id
        while tok3 < num_token:
            ll.mbarrier_wait(mbar_empty3 + prod3_idx, prod3_phase)
            if ll.elect_one_sync():
                ll.tma_copy_1d_g2s(
                    recv_x + tok3 * HIDDEN_SIZE,
                    mbar_full3 + prod3_idx,
                    tma_load_buf3 + prod3_idx * HIDDEN_SIZE,
                    num_bytes3,
                )
                ll.mbarrier_arrive_and_expect_tx(mbar_full3 + prod3_idx, num_bytes3)
            ll.__syncwarp()
            prod3_idx = prod3_idx + 1
            if prod3_idx == NUM_LOAD_STAGES:
                prod3_idx = 0
                prod3_phase = prod3_phase ^ 1
            tok3 = tok3 + num_block

    if is_consumer3 == 1:
        cons3_idx: ll.int32 = 0
        cons3_phase: ll.int32 = 0
        leader3: ll.int32 = ll.elect_one_sync()
        tok4: ll.int32 = block_id
        while tok4 < num_token:
            eidx3: ll.int32 = topk_indices[tok4 * TOPK + response_idx]
            trank3: ll.int32 = eidx3 / num_experts_per_rank
            need3: ll.int32 = topk_send_mask[tok4 * TOPK + response_idx]
            ll.mbarrier_wait(mbar_full3 + cons3_idx, cons3_phase)
            if trank3 < num_ranks:
                if need3 == 1:
                    sidx3: ll.int32 = token_dst_scatter_indices[tok4 * TOPK + response_idx]
                    if leader3:
                        ll.tma_copy_1d_s2g(
                            tma_load_buf3 + cons3_idx * HIDDEN_SIZE,
                            x_ptrs[trank3] + sidx3 * HIDDEN_SIZE,
                            num_bytes3,
                        )
                        ll.tma_store_arrive()
            ll.tma_store_wait_n(0)
            ll.__syncwarp()
            if leader3:
                ll.mbarrier_arrive(mbar_empty3 + cons3_idx)
            ll.__syncwarp()
            cons3_idx = cons3_idx + 1
            if cons3_idx == NUM_LOAD_STAGES:
                cons3_idx = 0
                cons3_phase = cons3_phase ^ 1
            tok4 = tok4 + num_block

    ll.tma_store_wait_n(0)
    ll.__syncthreads()


# ============================================================================
# Codegen / build helpers
# ============================================================================
SMEM_PREPROCESS = 4096  # minimal for preprocess (no TMA buffers)
SMEM_COMBINE = 1024 + (NUM_LOAD_STAGES * 2) * 8 + NUM_LOAD_STAGES * HIDDEN_SIZE * 2 + 4096

SMEM_V1 = 1024 + (V1_NUM_WG * V1_NUM_LOAD_STAGES * 2) * 8 + V1_NUM_WG * (V1_NUM_LOAD_STAGES +
                                                                         V1_NUM_STORE_STAGES) * HIDDEN_SIZE * 2 + 4096
SMEM_V2 = SMEM_COMBINE  # v2 reuses basic combine smem layout


def show_generated_code(kernel_name="preprocess"):
    kernels = {
        "preprocess": (kernel_combine_preprocess_inplace, BLOCK_SIZE_PREPROCESS),
        "combine": (kernel_combine_intranode, 1024),
        "v1": (kernel_combine_intranode_v1, 1024),
        "v2": (kernel_combine_intranode_v2, 768),
    }
    kernel, threads = kernels[kernel_name]
    cuda_code = kernel.compile(passes=PASSES["cuda"], codegen_func=codegen_cuda, need_header=True, num_threads=threads)
    print("=" * 80)
    print(f"Generated CUDA code ({kernel_name}):")
    print("=" * 80)
    print(cuda_code)
    return cuda_code


def build_kernel(kernel_name="preprocess"):
    configs = {
        "preprocess": (kernel_combine_preprocess_inplace, BLOCK_SIZE_PREPROCESS, SMEM_PREPROCESS),
        "combine": (kernel_combine_intranode, 1024, SMEM_COMBINE),
        "v1": (kernel_combine_intranode_v1, 1024, SMEM_V1),
        "v2": (kernel_combine_intranode_v2, 768, SMEM_V2),
    }
    kernel, threads, smem = configs[kernel_name]
    compiled = kernel.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(128, 1, 1), block=(threads, 1, 1),
                            shared_mem_bytes=smem, verbose=True)
    return compiled


if __name__ == "__main__":
    import sys
    kernel_name = "preprocess"
    if "--combine" in sys.argv:
        kernel_name = "combine"
    elif "--v1" in sys.argv:
        kernel_name = "v1"
    elif "--v2" in sys.argv:
        kernel_name = "v2"
    if "--codegen" in sys.argv:
        show_generated_code(kernel_name)
    else:
        print(f"=== Generating CUDA code ({kernel_name}) ===")
        cuda_code = show_generated_code(kernel_name)
        print(f"\n=== Compiling kernel ({kernel_name}) ===")
        compiled = build_kernel(kernel_name)
        print("Kernel compiled successfully!")
