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
FlashComm dispatch kernel: kernel_dispatch_intranode
Producer-consumer pattern with TMA 1D copies.

Fixed configuration: bf16 tokens, float32 weights, int32 offsets
kHiddenSize=7168, kTopk=8, kNumStages=2, kNumConsumerGroups=1
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"
WARP_SIZE = 32

# Compile-time template parameters (fixed for bf16/float/int32 config)
HIDDEN_SIZE = 7168
TOPK = 8
NUM_STAGES = 2
NUM_CONSUMER_GROUPS = 1
MAX_WORLD_SIZE = 72

# Shared memory layout offsets (manual, matching DispatchIntraNodeSmem)
# mbar_full: NUM_STAGES * 8 bytes = 16
# mbar_empty: NUM_STAGES * 8 bytes = 16
# recv_x_ptrs: MAX_WORLD_SIZE * 8 bytes = 576
# recv_weights_ptrs: MAX_WORLD_SIZE * 8 bytes = 576
# recv_topk_scatter_indices_ptrs: MAX_WORLD_SIZE * 8 bytes = 576
# align to 128 for TMA
# tma_buffer: NUM_STAGES * HIDDEN_SIZE * 2 bytes (bf16)

MBAR_FULL_OFFSET = 0
MBAR_EMPTY_OFFSET = MBAR_FULL_OFFSET + NUM_STAGES * 8
RECV_X_PTRS_OFFSET = MBAR_EMPTY_OFFSET + NUM_STAGES * 8
RECV_WEIGHTS_PTRS_OFFSET = RECV_X_PTRS_OFFSET + MAX_WORLD_SIZE * 8
RECV_SCATTER_PTRS_OFFSET = RECV_WEIGHTS_PTRS_OFFSET + MAX_WORLD_SIZE * 8
TMA_BUF_UNALIGNED = RECV_SCATTER_PTRS_OFFSET + MAX_WORLD_SIZE * 8
TMA_BUF_OFFSET = ((TMA_BUF_UNALIGNED + 127) // 128) * 128
TMA_BUF_SIZE = NUM_STAGES * HIDDEN_SIZE * 2  # bf16
TOTAL_SMEM = TMA_BUF_OFFSET + TMA_BUF_SIZE

NUM_PRODUCER_WARPS = 1
NUM_CONSUMER_WARPS = TOPK * NUM_CONSUMER_GROUPS
TOTAL_WARPS = NUM_CONSUMER_WARPS + NUM_PRODUCER_WARPS + 1  # +1 for init warp overlap
TOTAL_THREADS = 1024


# ============================================================================
# Pipeline state helpers (inline, replacing PipelineState struct)
# ============================================================================
@ll_kernel(backend=backend, is_entry=False)
def pipeline_advance(
    pipe_index: ll.int32,
    pipe_phase: ll.int32,
    num_stages: ll.int32,
) -> ll.int32:
    """Advance pipeline by 1 step. Returns new index. Updates phase via side effect."""
    pipe_index = pipe_index + 1
    if pipe_index == num_stages:
        pipe_index = 0
    return pipe_index


@ll_kernel(backend=backend, is_entry=True)
def kernel_dispatch_intranode(
    x: ll.Tensor[ll.bfloat16],  # [num_token, hidden_size]
    topk_send_mask: ll.Tensor[ll.int32],  # [num_token, topk]
    topk_weights: ll.Tensor[ll.float32],  # [num_token, topk]
    topk_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    token_dst_scatter_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    num_token: ll.int32,
    hidden_size: ll.int32,
    num_experts_per_rank: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
    recv_x_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],
    recv_weights_ptrs: ll.ptr[ll.ptr[ll.float32]],
    recv_topk_scatter_indices_ptrs: ll.ptr[ll.ptr[ll.int32]],
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE

    # Allocate shared memory regions (1024-byte aligned covers 128-byte TMA requirement)
    ll.align_memory(1024, scope="dynamic_shared")
    mbar_full = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    tma_buffer = ll.empty([NUM_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    # Init barriers (warp 0, elected thread)
    if warp_id == 0:
        if ll.elect_one_sync():
            for i_stage in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_full + i_stage, 1)
            for i_stage2 in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_empty + i_stage2, TOPK)

    ll.__syncthreads()

    # Pipeline state for producer and consumer
    is_producer_warp: ll.int32 = 1 if warp_id == TOPK else 0
    is_consumer_warp: ll.int32 = 1 if warp_id < TOPK else 0

    num_bytes_per_token: ll.int32 = hidden_size * 2  # bf16 = 2 bytes

    # Producer warp
    if is_producer_warp == 1:
        prod_idx: ll.int32 = 0
        prod_phase: ll.int32 = 1  # producer starts with phase=1
        token_offset: ll.int32 = block_id
        while token_offset < num_token:
            # Wait for empty slot
            ll.mbarrier_wait(mbar_empty + prod_idx, prod_phase)
            # Issue TMA copy
            if ll.elect_one_sync():
                ll.tma_copy_1d_g2s(
                    x + token_offset * hidden_size,
                    mbar_full + prod_idx,
                    tma_buffer + prod_idx * HIDDEN_SIZE,
                    num_bytes_per_token,
                )
                ll.mbarrier_arrive_and_expect_tx(mbar_full + prod_idx, num_bytes_per_token)
            ll.__syncwarp()
            # Advance pipeline
            prod_idx = prod_idx + 1
            if prod_idx == NUM_STAGES:
                prod_idx = 0
                prod_phase = prod_phase ^ 1
            token_offset = token_offset + num_block

    # Consumer warps: each warp handles one topk slot
    if is_consumer_warp == 1:
        response_expert_idx: ll.int32 = warp_id % TOPK
        cons_idx: ll.int32 = 0
        cons_phase: ll.int32 = 0  # consumer starts with phase=0
        is_leader: ll.int32 = ll.elect_one_sync()
        token_offset2: ll.int32 = block_id
        while token_offset2 < num_token:
            expert_idx: ll.int32 = topk_indices[token_offset2 * TOPK + response_expert_idx]
            target_rank: ll.int32 = expert_idx / num_experts_per_rank
            is_need_send: ll.int32 = topk_send_mask[token_offset2 * TOPK + response_expert_idx]

            # Wait for data
            ll.mbarrier_wait(mbar_full + cons_idx, cons_phase)

            if target_rank < num_ranks:
                if is_need_send == 1:
                    store_idx: ll.int32 = token_dst_scatter_indices[token_offset2 * TOPK + response_expert_idx]
                    # TMA store token data
                    if is_leader:
                        ll.tma_copy_1d_s2g(
                            tma_buffer + cons_idx * HIDDEN_SIZE,
                            recv_x_ptrs[target_rank] + store_idx * hidden_size,
                            num_bytes_per_token,
                        )
                        ll.tma_store_arrive()

            ll.tma_store_wait_n(0)
            ll.__syncwarp()
            if is_leader:
                ll.mbarrier_arrive(mbar_empty + cons_idx)
            ll.__syncwarp()
            # Advance consumer pipeline
            cons_idx = cons_idx + 1
            if cons_idx == NUM_STAGES:
                cons_idx = 0
                cons_phase = cons_phase ^ 1
            token_offset2 = token_offset2 + num_block


# ============================================================================
# kernel_dispatch_intranode_v1
# Compared to dispatch_intranode:
# 1. each consumer group uses only 1 warp (instead of kTopk warps)
# 2. single warp sends up to kTopk TMA stores per token using ballot/shuffle
# 3. in-flight TMA stores (kStorePipe=2) for better performance and less warps
# ============================================================================

# v1 constants: 1 producer + kNumConsumerGroups consumer warps
V1_NUM_CONSUMER_GROUPS = 1
V1_STORE_PIPE = 2
V1_TOTAL_WARPS = 1 + V1_NUM_CONSUMER_GROUPS + 1  # producer + consumers + init overlap
V1_TOTAL_THREADS = 1024


@ll_kernel(backend=backend, is_entry=True)
def kernel_dispatch_intranode_v1(
    x: ll.Tensor[ll.bfloat16],  # [num_token, hidden_size]
    topk_send_mask: ll.Tensor[ll.int32],  # [num_token, topk]
    topk_weights: ll.Tensor[ll.float32],  # [num_token, topk]
    topk_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    token_dst_scatter_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    num_token: ll.int32,
    hidden_size: ll.int32,
    num_experts_per_rank: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
    recv_x_ptrs: ll.ptr[ll.ptr[ll.bfloat16]],
    recv_weights_ptrs: ll.ptr[ll.ptr[ll.float32]],
    recv_topk_scatter_indices_ptrs: ll.ptr[ll.ptr[ll.int32]],
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE
    lane_id: ll.int32 = thread_id % WARP_SIZE

    # Allocate shared memory
    ll.align_memory(1024, scope="dynamic_shared")
    mbar_full = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    tma_buffer = ll.empty([NUM_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    # Init barriers (warp 0)
    if warp_id == 0:
        if ll.elect_one_sync():
            for i_stage in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_full + i_stage, 1)
            for i_stage2 in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_empty + i_stage2, 1)  # v1: 1 consumer warp per stage

    ll.__syncthreads()

    is_producer_warp: ll.int32 = 1 if warp_id == 0 else 0
    is_consumer_warp: ll.int32 = 1 if warp_id == 1 else 0  # v1: only 1 consumer warp

    num_bytes_per_token: ll.int32 = hidden_size * 2  # bf16 = 2 bytes

    # ---- Producer warp ----
    if is_producer_warp == 1:
        prod_idx: ll.int32 = 0
        prod_phase: ll.int32 = 1
        token_offset: ll.int32 = block_id
        while token_offset < num_token:
            ll.mbarrier_wait(mbar_empty + prod_idx, prod_phase)
            if ll.elect_one_sync():
                ll.tma_copy_1d_g2s(
                    x + token_offset * hidden_size,
                    mbar_full + prod_idx,
                    tma_buffer + prod_idx * HIDDEN_SIZE,
                    num_bytes_per_token,
                )
                ll.mbarrier_arrive_and_expect_tx(mbar_full + prod_idx, num_bytes_per_token)
            ll.__syncwarp()
            prod_idx = prod_idx + 1
            if prod_idx == NUM_STAGES:
                prod_idx = 0
                prod_phase = prod_phase ^ 1
            token_offset = token_offset + num_block

    # ---- Consumer warp: single warp sends all topk slots via ballot/shuffle ----
    if is_consumer_warp == 1:
        cons_idx: ll.int32 = 0
        cons_phase: ll.int32 = 0
        rel_idx: ll.int32 = 0
        rel_phase: ll.int32 = 0
        is_leader: ll.int32 = ll.elect_one_sync()
        tokens_processed: ll.int32 = 0
        token_offset2: ll.int32 = block_id

        while token_offset2 < num_token:
            # Load per-lane metadata: each lane < kTopk loads its expert info
            my_expert_idx: ll.int32 = -1
            my_target_rank: ll.int32 = -1
            my_is_need_send: ll.int32 = 0
            my_store_idx: ll.int32 = -1
            my_weight: ll.float32 = 0.0

            if lane_id < TOPK:
                my_expert_idx = topk_indices[token_offset2 * TOPK + lane_id]
                my_target_rank = my_expert_idx / num_experts_per_rank
                my_is_need_send = topk_send_mask[token_offset2 * TOPK + lane_id]
                my_store_idx = token_dst_scatter_indices[token_offset2 * TOPK + lane_id]
                my_weight = topk_weights[token_offset2 * TOPK + lane_id]
            ll.__syncwarp()

            # Wait for in-flight TMA stores from kStorePipe tokens ago
            ll.tma_store_wait_n(V1_STORE_PIPE - 1)
            ll.__syncwarp()

            # Release empty barrier for the finished stage
            if tokens_processed >= V1_STORE_PIPE:
                if is_leader:
                    ll.mbarrier_arrive(mbar_empty + rel_idx)
                rel_idx = rel_idx + 1
                if rel_idx == NUM_STAGES:
                    rel_idx = 0
                    rel_phase = rel_phase ^ 1

            # Wait for producer to fill the stage
            ll.mbarrier_wait(mbar_full + cons_idx, cons_phase)

            # Ballot: which lanes need to send?
            should_send: ll.int32 = 1 if (my_target_rank >= 0 and my_target_rank < num_ranks
                                          and my_is_need_send == 1) else 0
            send_mask: ll.uint32 = ll.ballot_sync(0xFFFFFFFF, should_send)

            # Iterate over set bits in send_mask
            remaining_mask: ll.uint32 = send_mask
            while remaining_mask != 0:
                send_lane: ll.int32 = ll.ffs(remaining_mask) - 1
                remaining_mask = remaining_mask & (remaining_mask - 1)

                # Broadcast target_rank and store_idx from the sending lane
                target_rank_bcast: ll.int32 = ll.__shfl_sync(0xFFFFFFFF, my_target_rank, send_lane)
                store_idx_bcast: ll.int32 = ll.__shfl_sync(0xFFFFFFFF, my_store_idx, send_lane)

                # TMA store token data to remote
                if is_leader:
                    ll.tma_copy_1d_s2g(
                        tma_buffer + cons_idx * HIDDEN_SIZE,
                        recv_x_ptrs[target_rank_bcast] + store_idx_bcast * hidden_size,
                        num_bytes_per_token,
                    )

                # Write weights and scatter indices for all topk slots
                if lane_id < TOPK:
                    # Weight
                    dst_weight_ptr: ll.ptr[
                        ll.float32] = recv_weights_ptrs[target_rank_bcast] + store_idx_bcast * TOPK + lane_id
                    dst_weight_ptr[0] = my_weight

                    # Scatter index: only valid if this lane's target_rank matches
                    cur_index: ll.int32 = my_store_idx if my_target_rank == target_rank_bcast else -1
                    dst_index_ptr: ll.ptr[
                        ll.int32] = recv_topk_scatter_indices_ptrs[target_rank_bcast] + store_idx_bcast * TOPK + lane_id
                    dst_index_ptr[0] = cur_index
                ll.__syncwarp()

            ll.tma_store_arrive()
            ll.__syncwarp()

            # Advance consumer pipeline
            cons_idx = cons_idx + 1
            if cons_idx == NUM_STAGES:
                cons_idx = 0
                cons_phase = cons_phase ^ 1
            tokens_processed = tokens_processed + 1
            token_offset2 = token_offset2 + num_block

        # Drain remaining in-flight TMA stores
        ll.tma_store_wait_n(0)


# ============================================================================
# Codegen / build helpers
# ============================================================================
def show_generated_code(kernel_name="basic"):
    kernels = {"basic": kernel_dispatch_intranode, "v1": kernel_dispatch_intranode_v1}
    kernel = kernels[kernel_name]
    threads = TOTAL_THREADS if kernel_name == "basic" else V1_TOTAL_THREADS
    cuda_code = kernel.compile(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        need_header=True,
        num_threads=threads,
    )
    print("=" * 80)
    print(f"Generated CUDA code ({kernel_name}):")
    print("=" * 80)
    print(cuda_code)
    return cuda_code


def build_kernel(kernel_name="basic"):
    kernels = {"basic": kernel_dispatch_intranode, "v1": kernel_dispatch_intranode_v1}
    kernel = kernels[kernel_name]
    threads = TOTAL_THREADS if kernel_name == "basic" else V1_TOTAL_THREADS
    compiled = kernel.build(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        grid=(128, 1, 1),
        block=(threads, 1, 1),
        shared_mem_bytes=TOTAL_SMEM + 4096,
        verbose=True,
    )
    return compiled


if __name__ == "__main__":
    import sys
    kernel_name = "basic"
    if "--v1" in sys.argv:
        kernel_name = "v1"
    if "--codegen" in sys.argv:
        show_generated_code(kernel_name)
    else:
        print(f"=== Generating CUDA code ({kernel_name}) ===")
        cuda_code = show_generated_code(kernel_name)
        print(f"\n=== Compiling kernel ({kernel_name}) ===")
        compiled = build_kernel(kernel_name)
        print("Kernel compiled successfully!")
