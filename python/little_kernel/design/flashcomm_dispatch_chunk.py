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
FlashComm dispatch kernel: kernel_dispatch_intranode_chunk
Chunked traversal with meta prefetch ping-pong in shared memory.

Fixed configuration: bf16 tokens, float32 weights, int32 offsets
kHiddenSize=7168, kTopk=8, kNumStages=2, kNumConsumerGroups=1, kChunkSize=128
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"
WARP_SIZE = 32

# Compile-time template parameters
HIDDEN_SIZE = 7168
TOPK = 8
NUM_STAGES = 2
NUM_CONSUMER_GROUPS = 1
CHUNK_SIZE = 128
MAX_WORLD_SIZE = 72

# Warp layout: kTopk consumer warps + 1 producer warp
NUM_PRODUCER_WARPS = 1
NUM_CONSUMER_WARPS = TOPK * NUM_CONSUMER_GROUPS
TOTAL_THREADS = 1024

# Meta buffer sizes (per ping)
META_ELEMS = CHUNK_SIZE * TOPK  # 128 * 8 = 1024 elements per meta array


# ============================================================================
# issue_meta_prefetch_one_thread
# Only warp 0 lane 0 of consumer warps issues TMA 1D copies for metadata.
# ============================================================================
@ll_kernel(backend=backend, is_entry=False)
def issue_meta_prefetch(
    meta_mbar_full: ll.ptr[ll.uint64],  # &meta_mbar_full[buf]
    meta_send_mask: ll.ptr[ll.int32],  # &meta_topk_send_mask[buf * META_ELEMS]
    meta_weights: ll.ptr[ll.float32],  # &meta_topk_weights[buf * META_ELEMS]
    meta_indices: ll.ptr[ll.int32],  # &meta_topk_indices[buf * META_ELEMS]
    meta_scatter: ll.ptr[ll.int32],  # &meta_token_dst_scatter_indices[buf * META_ELEMS]
    topk_send_mask: ll.Tensor[ll.int32],
    topk_weights: ll.Tensor[ll.float32],
    topk_indices: ll.Tensor[ll.int32],
    token_dst_scatter_indices: ll.Tensor[ll.int32],
    chunk_base: ll.int32,
    chunk_len: ll.int32,
    topk: ll.int32,
    warp_id: ll.int32,
    lane_id: ll.int32,
) -> ll.void:
    if chunk_len <= 0:
        return
    # Only warp 0 lane 0 issues prefetch
    if warp_id != 0:
        return
    if lane_id != 0:
        return

    elems: ll.int32 = chunk_len * topk
    bytes_send_mask: ll.int32 = elems * 4  # int32 = 4 bytes
    bytes_weights: ll.int32 = elems * 4  # float32 = 4 bytes
    bytes_indices: ll.int32 = elems * 4  # int32 = 4 bytes
    bytes_scatter: ll.int32 = elems * 4  # int32 = 4 bytes
    total_bytes: ll.int32 = bytes_send_mask + bytes_weights + bytes_indices + bytes_scatter

    ll.mbarrier_arrive_and_expect_tx(meta_mbar_full, total_bytes)
    ll.tma_copy_1d_g2s(topk_send_mask + chunk_base * topk, meta_mbar_full, meta_send_mask, bytes_send_mask)
    ll.tma_copy_1d_g2s(topk_weights + chunk_base * topk, meta_mbar_full, meta_weights, bytes_weights)
    ll.tma_copy_1d_g2s(topk_indices + chunk_base * topk, meta_mbar_full, meta_indices, bytes_indices)
    ll.tma_copy_1d_g2s(token_dst_scatter_indices + chunk_base * topk, meta_mbar_full, meta_scatter, bytes_scatter)


# ============================================================================
# kernel_dispatch_intranode_chunk
# ============================================================================
@ll_kernel(backend=backend, is_entry=True)
def kernel_dispatch_intranode_chunk(
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

    # ---- Shared memory layout ----
    ll.align_memory(1024, scope="dynamic_shared")
    mbar_full = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    # Meta prefetch barriers (2 for ping-pong)
    meta_mbar_full = ll.empty([2], dtype=ll.uint64, scope="dynamic_shared")
    meta_mbar_empty = ll.empty([2], dtype=ll.uint64, scope="dynamic_shared")
    # Meta ping-pong buffers: [2 * CHUNK_SIZE * TOPK] per type
    meta_topk_send_mask = ll.empty([2 * META_ELEMS], dtype=ll.int32, scope="dynamic_shared")
    meta_topk_weights = ll.empty([2 * META_ELEMS], dtype=ll.float32, scope="dynamic_shared")
    meta_topk_indices = ll.empty([2 * META_ELEMS], dtype=ll.int32, scope="dynamic_shared")
    meta_token_dst_scatter_indices = ll.empty([2 * META_ELEMS], dtype=ll.int32, scope="dynamic_shared")
    # TMA buffer for token data
    tma_buffer = ll.empty([NUM_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    # Warp roles
    is_consumer_warp: ll.int32 = 1 if warp_id < TOPK else 0
    is_producer_warp: ll.int32 = 1 if warp_id == TOPK else 0
    consumer_group_id: ll.int32 = warp_id / TOPK
    response_expert_idx: ll.int32 = warp_id % TOPK
    consumer_arr_cnt: ll.int32 = TOPK

    # Init barriers (warp 0, elected thread)
    if warp_id == 0:
        if ll.elect_one_sync():
            for i_s in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_full + i_s, 1)
            for i_s2 in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_empty + i_s2, consumer_arr_cnt)
            # Meta prefetch barriers
            ll.init_smem_barrier(meta_mbar_full, 1)
            ll.init_smem_barrier(meta_mbar_full + 1, 1)
            ll.init_smem_barrier(meta_mbar_empty, NUM_CONSUMER_WARPS)
            ll.init_smem_barrier(meta_mbar_empty + 1, NUM_CONSUMER_WARPS)

    ll.__syncthreads()

    num_bytes_per_token: ll.int32 = hidden_size * 2  # bf16 = 2 bytes

    # ---- Producer warp: chunked token traversal ----
    if is_producer_warp == 1:
        prod_idx: ll.int32 = 0
        prod_phase: ll.int32 = 1
        chunk_base: ll.int32 = block_id * CHUNK_SIZE
        while chunk_base < num_token:
            chunk_len: ll.int32 = num_token - chunk_base
            if chunk_len > CHUNK_SIZE:
                chunk_len = CHUNK_SIZE
            i_tok: ll.int32 = 0
            while i_tok < chunk_len:
                token_offset: ll.int32 = chunk_base + i_tok
                ll.mbarrier_wait(mbar_empty + prod_idx, prod_phase)
                if ll.elect_one_sync():
                    ll.mbarrier_arrive_and_expect_tx(mbar_full + prod_idx, num_bytes_per_token)
                    ll.tma_copy_1d_g2s(
                        x + token_offset * hidden_size,
                        mbar_full + prod_idx,
                        tma_buffer + prod_idx * HIDDEN_SIZE,
                        num_bytes_per_token,
                    )
                ll.__syncwarp()
                prod_idx = prod_idx + 1
                if prod_idx == NUM_STAGES:
                    prod_idx = 0
                    prod_phase = prod_phase ^ 1
                i_tok = i_tok + 1
            chunk_base = chunk_base + num_block * CHUNK_SIZE

    # ---- Consumer warps: chunked with meta prefetch ----
    if is_consumer_warp == 1:
        cons_idx: ll.int32 = 0
        cons_phase: ll.int32 = 0
        is_leader: ll.int32 = ll.elect_one_sync()

        # Meta prefetch phase tracking (ping-pong)
        meta_phase_0: ll.int32 = 0
        meta_phase_1: ll.int32 = 0
        meta_empty_phase_0: ll.int32 = 1
        meta_empty_phase_1: ll.int32 = 1
        buf: ll.int32 = 0

        # Prefetch first chunk's meta
        first_chunk_base: ll.int32 = block_id * CHUNK_SIZE
        first_chunk_len: ll.int32 = num_token - first_chunk_base
        if first_chunk_len > CHUNK_SIZE:
            first_chunk_len = CHUNK_SIZE
        if first_chunk_base < num_token:
            ll.mbarrier_wait(meta_mbar_empty, meta_empty_phase_0)
            meta_empty_phase_0 = meta_empty_phase_0 ^ 1
            issue_meta_prefetch(
                meta_mbar_full,
                meta_topk_send_mask,
                meta_topk_weights,
                meta_topk_indices,
                meta_token_dst_scatter_indices,
                topk_send_mask,
                topk_weights,
                topk_indices,
                token_dst_scatter_indices,
                first_chunk_base,
                first_chunk_len,
                TOPK,
                warp_id,
                lane_id,
            )

        chunk_base2: ll.int32 = first_chunk_base
        chunk_len2: ll.int32 = first_chunk_len

        while chunk_base2 < num_token:
            # Wait for current meta buffer
            if buf == 0:
                ll.mbarrier_wait(meta_mbar_full, meta_phase_0)
                meta_phase_0 = meta_phase_0 ^ 1
            else:
                ll.mbarrier_wait(meta_mbar_full + 1, meta_phase_1)
                meta_phase_1 = meta_phase_1 ^ 1

            # Prefetch next chunk's meta
            next_chunk_base: ll.int32 = chunk_base2 + num_block * CHUNK_SIZE
            next_chunk_len: ll.int32 = num_token - next_chunk_base
            if next_chunk_len > CHUNK_SIZE:
                next_chunk_len = CHUNK_SIZE
            next_buf: ll.int32 = buf ^ 1
            if next_chunk_base < num_token:
                if next_buf == 0:
                    ll.mbarrier_wait(meta_mbar_empty, meta_empty_phase_0)
                    meta_empty_phase_0 = meta_empty_phase_0 ^ 1
                else:
                    ll.mbarrier_wait(meta_mbar_empty + 1, meta_empty_phase_1)
                    meta_empty_phase_1 = meta_empty_phase_1 ^ 1
                issue_meta_prefetch(
                    meta_mbar_full + next_buf,
                    meta_topk_send_mask + next_buf * META_ELEMS,
                    meta_topk_weights + next_buf * META_ELEMS,
                    meta_topk_indices + next_buf * META_ELEMS,
                    meta_token_dst_scatter_indices + next_buf * META_ELEMS,
                    topk_send_mask,
                    topk_weights,
                    topk_indices,
                    token_dst_scatter_indices,
                    next_chunk_base,
                    next_chunk_len,
                    TOPK,
                    warp_id,
                    lane_id,
                )

            # Process tokens in this chunk
            i_cons: ll.int32 = consumer_group_id
            while i_cons < chunk_len2:
                expert_idx: ll.int32 = meta_topk_indices[buf * META_ELEMS + i_cons * TOPK + response_expert_idx]
                target_rank: ll.int32 = expert_idx / num_experts_per_rank
                is_need_send: ll.int32 = meta_topk_send_mask[buf * META_ELEMS + i_cons * TOPK + response_expert_idx]

                # Wait for producer to fill TMA buffer
                ll.mbarrier_wait(mbar_full + cons_idx, cons_phase)

                if target_rank < num_ranks:
                    if is_need_send == 1:
                        store_idx: ll.int32 = meta_token_dst_scatter_indices[buf * META_ELEMS + i_cons * TOPK +
                                                                             response_expert_idx]
                        # TMA store token data to remote
                        if is_leader:
                            ll.tma_copy_1d_s2g(
                                tma_buffer + cons_idx * HIDDEN_SIZE,
                                recv_x_ptrs[target_rank] + store_idx * hidden_size,
                                num_bytes_per_token,
                            )
                            ll.tma_store_arrive()
                        # Write weights
                        if lane_id < TOPK:
                            dst_weight_ptr: ll.ptr[
                                ll.float32] = recv_weights_ptrs[target_rank] + store_idx * TOPK + lane_id
                            cur_weight: ll.float32 = meta_topk_weights[buf * META_ELEMS + i_cons * TOPK + lane_id]
                            dst_weight_ptr[0] = cur_weight
                        # Write scatter indices (for lanes kTopk..2*kTopk-1)
                        if lane_id >= TOPK:
                            if lane_id < 2 * TOPK:
                                li: ll.int32 = lane_id - TOPK
                                dst_index_ptr: ll.ptr[
                                    ll.int32] = recv_topk_scatter_indices_ptrs[target_rank] + store_idx * TOPK + li
                                cur_index: ll.int32 = meta_token_dst_scatter_indices[buf * META_ELEMS + i_cons * TOPK +
                                                                                     li]
                                cur_expert_idx: ll.int32 = meta_topk_indices[buf * META_ELEMS + i_cons * TOPK + li]
                                if cur_expert_idx / num_experts_per_rank != target_rank:
                                    cur_index = -1
                                dst_index_ptr[0] = cur_index

                ll.__syncwarp()
                ll.tma_store_wait_n(0)
                ll.__syncwarp()
                if is_leader:
                    ll.mbarrier_arrive(mbar_empty + cons_idx)
                ll.__syncwarp()
                # Advance pipeline
                cons_idx = cons_idx + 1
                if cons_idx == NUM_STAGES:
                    cons_idx = 0
                    cons_phase = cons_phase ^ 1
                i_cons = i_cons + NUM_CONSUMER_GROUPS

            # Release meta buffer
            if lane_id == 0:
                if buf == 0:
                    ll.mbarrier_arrive(meta_mbar_empty)
                else:
                    ll.mbarrier_arrive(meta_mbar_empty + 1)
            ll.__syncwarp()

            buf = next_buf
            chunk_len2 = next_chunk_len
            chunk_base2 = chunk_base2 + num_block * CHUNK_SIZE


# ============================================================================
# Shared memory size calculation
# ============================================================================
SMEM_MBAR = (NUM_STAGES + NUM_STAGES + 2 + 2) * 8  # mbar_full, mbar_empty, meta_mbar_full/empty
SMEM_META = 2 * META_ELEMS * 4 * 4  # 4 meta arrays, each 2*1024 elements of int32/float32
SMEM_TMA = NUM_STAGES * HIDDEN_SIZE * 2  # bf16
TOTAL_SMEM = 1024 + SMEM_MBAR + SMEM_META + SMEM_TMA + 4096  # 1024 alignment + padding


def show_generated_code():
    kernel = kernel_dispatch_intranode_chunk
    cuda_code = kernel.compile(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        need_header=True,
        num_threads=TOTAL_THREADS,
    )
    print("=" * 80)
    print("Generated CUDA code (dispatch_chunk):")
    print("=" * 80)
    print(cuda_code)
    return cuda_code


def build_kernel():
    kernel = kernel_dispatch_intranode_chunk
    compiled = kernel.build(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        grid=(128, 1, 1),
        block=(TOTAL_THREADS, 1, 1),
        shared_mem_bytes=TOTAL_SMEM,
        verbose=True,
    )
    return compiled


if __name__ == "__main__":
    import sys
    if "--codegen" in sys.argv:
        show_generated_code()
    else:
        print("=== Generating CUDA code (dispatch_chunk) ===")
        cuda_code = show_generated_code()
        print("\n=== Compiling kernel (dispatch_chunk) ===")
        compiled = build_kernel()
        print("Kernel compiled successfully!")
