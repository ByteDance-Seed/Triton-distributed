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
FlashComm compute kernels
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"
WARP_SIZE = 32


@ll_kernel(backend=backend, is_entry=False)
def warp_reduce_sum(value: ll.int32) -> ll.int32:
    value = value + ll.shfl_xor_sync(0xFFFFFFFF, value, 16)
    value = value + ll.shfl_xor_sync(0xFFFFFFFF, value, 8)
    value = value + ll.shfl_xor_sync(0xFFFFFFFF, value, 4)
    value = value + ll.shfl_xor_sync(0xFFFFFFFF, value, 2)
    value = value + ll.shfl_xor_sync(0xFFFFFFFF, value, 1)
    return value


@ll_kernel(backend=backend, is_entry=False)
def warp_scan_inclusive(value: ll.int32) -> ll.int32:
    lane_id: ll.int32 = ll.threadIdx_x() % WARP_SIZE
    val1: ll.int32 = ll.shfl_up_sync(0xFFFFFFFF, value, 1)
    if lane_id >= 1:
        value = value + val1
    val2: ll.int32 = ll.shfl_up_sync(0xFFFFFFFF, value, 2)
    if lane_id >= 2:
        value = value + val2
    val4: ll.int32 = ll.shfl_up_sync(0xFFFFFFFF, value, 4)
    if lane_id >= 4:
        value = value + val4
    val8: ll.int32 = ll.shfl_up_sync(0xFFFFFFFF, value, 8)
    if lane_id >= 8:
        value = value + val8
    val16: ll.int32 = ll.shfl_up_sync(0xFFFFFFFF, value, 16)
    if lane_id >= 16:
        value = value + val16
    return value


@ll_kernel(backend=backend, is_entry=False)
def calc_rank_in_warp_and_accumulate(value: ll.int32, hist_ptr: ll.ptr[ll.int32]) -> ll.int32:
    lane_id: ll.int32 = ll.threadIdx_x() % WARP_SIZE
    m: ll.uint32 = ll.match_any_sync(0xFFFFFFFF, value)
    mask: ll.uint32 = (1 << lane_id) - 1
    rank_in_warp: ll.int32 = ll.popc(m & mask)
    if value != -1:
        ll.atomic_add(hist_ptr + value, 1)
    ll.__syncwarp()
    return rank_in_warp


@ll_kernel(backend=backend, is_entry=False)
def block_scan_inclusive(value: ll.int32, warp_sums: ll.ptr[ll.int32]) -> ll.int32:
    lane: ll.int32 = ll.threadIdx_x() & 31
    warp: ll.int32 = ll.threadIdx_x() >> 5
    num_warps: ll.int32 = (ll.blockDim_x() + 31) >> 5
    value = warp_scan_inclusive(value)
    if lane == WARP_SIZE - 1:
        warp_sums[warp] = value
    ll.__syncthreads()
    if warp == 0:
        x: ll.int32 = warp_sums[lane] if lane < num_warps else 0
        x = warp_scan_inclusive(x)
        if lane < num_warps:
            warp_sums[lane] = x
    ll.__syncthreads()
    warp_prefix: ll.int32 = 0 if warp == 0 else warp_sums[warp - 1]
    return warp_prefix + value


@ll_kernel(backend=backend, is_entry=False)
def barrier_all_block(barrier_ptrs: ll.ptr[ll.ptr[ll.int32]], rank: ll.int32, num_ranks: ll.int32) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    if thread_id < num_ranks:
        ll.threadfence_system()
        remote_ptr: ll.ptr[ll.int32] = barrier_ptrs[thread_id] + rank
        old: ll.int32 = ll.atomic_cas_system(remote_ptr, 0, 1)
        while old != 0:
            old = ll.atomic_cas_system(remote_ptr, 0, 1)
        ll.threadfence_system()
        wait_ptr: ll.ptr[ll.int32] = barrier_ptrs[rank] + thread_id
        old2: ll.int32 = ll.atomic_cas_system(wait_ptr, 1, 0)
        while old2 != 1:
            old2 = ll.atomic_cas_system(wait_ptr, 1, 0)
        ll.threadfence_system()
    ll.__syncthreads()


NUM_WARPS = 32
BLOCK_SIZE = NUM_WARPS * WARP_SIZE
MAX_EXPERTS_PLUS_1 = 385  # Must be >= max(num_experts) + 1, supports up to 384 experts


@ll_kernel(backend=backend, is_entry=True)
def kernel_compute_offset(
    topk_indices: ll.Tensor[ll.int32],
    num_token: ll.int32,
    topk: ll.int32,
    num_experts: ll.int32,
    block_cumsum_hist: ll.Tensor[ll.int32],
    token_within_expert_offset: ll.Tensor[ll.int32],
    expert_counts: ll.Tensor[ll.int32],
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE
    lane_id: ll.int32 = thread_id % WARP_SIZE
    ll.align_memory(1024, scope="dynamic_shared")
    smem = ll.empty([NUM_WARPS * MAX_EXPERTS_PLUS_1], dtype=ll.int32, scope="dynamic_shared")
    warp_hist: ll.ptr[ll.int32] = smem + warp_id * (num_experts + 1)
    for i in range(0, MAX_EXPERTS_PLUS_1, WARP_SIZE):
        if lane_id + i < num_experts + 1:
            warp_hist[lane_id + i] = 0
    ll.__syncthreads()
    num_tiles: ll.int32 = ll.cdiv(num_token * topk, BLOCK_SIZE)
    tile_id: ll.int32 = block_id
    while tile_id < num_tiles:
        tile_start: ll.int32 = tile_id * BLOCK_SIZE
        tile_end: ll.int32 = ll.min_val(tile_start + BLOCK_SIZE, num_token * topk)
        j: ll.int32 = thread_id
        while j < tile_end - tile_start:
            idx: ll.int32 = tile_start + j
            value: ll.int32 = topk_indices[idx] if idx < tile_end else -1
            if value != -1:
                ll.atomic_add(warp_hist + value, 1)
            j = j + BLOCK_SIZE
        ll.__syncthreads()
        j_exp: ll.int32 = warp_id
        while j_exp < num_experts + 1:
            v: ll.int32 = smem[j_exp + lane_id * (num_experts + 1)] if lane_id < NUM_WARPS else 0
            s: ll.int32 = warp_reduce_sum(v)
            if lane_id == 0:
                block_cumsum_hist[tile_id * (num_experts + 1) + j_exp] = s
            j_exp = j_exp + NUM_WARPS
        ll.__syncthreads()
        for i2 in range(0, MAX_EXPERTS_PLUS_1, WARP_SIZE):
            if lane_id + i2 < num_experts + 1:
                warp_hist[lane_id + i2] = 0
        tile_id = tile_id + num_block
    ll.__syncthreads()
    ll.grid_sync()
    global_warp_id: ll.int32 = block_id * NUM_WARPS + warp_id
    exp_idx: ll.int32 = global_warp_id
    while exp_idx < num_experts + 1:
        global_prefix_sum: ll.int32 = 0
        num_tiles_padding: ll.int32 = ll.cdiv(num_tiles, WARP_SIZE) * WARP_SIZE
        j2: ll.int32 = lane_id
        while j2 < num_tiles_padding:
            v2: ll.int32 = block_cumsum_hist[exp_idx + j2 * (num_experts + 1)] if j2 < num_tiles else 0
            warp_pre_sum: ll.int32 = warp_scan_inclusive(v2)
            if j2 < num_tiles:
                block_cumsum_hist[exp_idx + j2 * (num_experts + 1)] = global_prefix_sum + warp_pre_sum - v2
            warp_sum: ll.int32 = ll.__shfl_sync(0xFFFFFFFF, warp_pre_sum, WARP_SIZE - 1)
            global_prefix_sum = global_prefix_sum + warp_sum
            j2 = j2 + WARP_SIZE
        if expert_counts != 0:
            if lane_id == 0:
                expert_counts[exp_idx] = global_prefix_sum
        exp_idx = exp_idx + NUM_WARPS * num_block
    ll.__syncthreads()
    ll.grid_sync()
    for i3 in range(0, MAX_EXPERTS_PLUS_1, WARP_SIZE):
        if lane_id + i3 < num_experts + 1:
            warp_hist[lane_id + i3] = 0
    ll.__syncthreads()
    tile_id2: ll.int32 = block_id
    while tile_id2 < num_tiles:
        tile_start2: ll.int32 = tile_id2 * BLOCK_SIZE
        tile_end2: ll.int32 = ll.min_val(tile_start2 + BLOCK_SIZE, num_token * topk)
        val: ll.int32 = topk_indices[tile_start2 + thread_id] if tile_start2 + thread_id < tile_end2 else -1
        rank_in_warp: ll.int32 = calc_rank_in_warp_and_accumulate(val, warp_hist)
        ll.__syncthreads()
        j3: ll.int32 = warp_id
        while j3 < num_experts + 1:
            v3: ll.int32 = smem[j3 + lane_id * (num_experts + 1)] if lane_id < NUM_WARPS else 0
            wp: ll.int32 = warp_scan_inclusive(v3)
            ll.__syncwarp()
            if lane_id < NUM_WARPS:
                smem[j3 + lane_id * (num_experts + 1)] = wp - v3
            ll.__syncwarp()
            j3 = j3 + NUM_WARPS
        ll.__syncthreads()
        if tile_start2 + thread_id < tile_end2:
            token_within_expert_offset[tile_start2 + thread_id] = block_cumsum_hist[tile_id2 * (num_experts + 1) +
                                                                                    val] + warp_hist[val] + rank_in_warp
        ll.__syncthreads()
        for i4 in range(0, MAX_EXPERTS_PLUS_1, WARP_SIZE):
            if lane_id + i4 < num_experts + 1:
                warp_hist[lane_id + i4] = 0
        tile_id2 = tile_id2 + num_block


# ============================================================================
# kernel_compute_dispatch_layout
# 1. all gather local_splits to full_splits (pull mode)
# 2. cumsum full_splits to get recv_base_offset
# 3. calc token destination index
# 4. calc token send mask
# ============================================================================
@ll_kernel(backend=backend, is_entry=True)
def kernel_compute_dispatch_layout(
    topk_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    token_within_expert_offset: ll.Tensor[ll.int32],  # [num_token, topk]
    local_splits: ll.Tensor[ll.int32],  # [num_experts + 1]
    full_splits_ptrs: ll.ptr[ll.ptr[ll.int32]],  # [num_ranks] -> [num_ranks * (num_experts + 1)]
    barrier_ptrs: ll.ptr[ll.ptr[ll.int32]],  # [num_ranks]
    recv_base_offset: ll.Tensor[ll.int32],  # [num_ranks, num_experts_per_rank, num_ranks]
    token_dst_scatter_indices: ll.Tensor[ll.int32],  # [num_token, topk]
    token_topk_send_mask: ll.Tensor[ll.int32],  # [num_token, topk]
    recv_token_count_cpu: ll.ptr[ll.int32],  # [num_ranks] (optional, pinned)
    recv_token_count: ll.ptr[ll.int32],  # [num_ranks] (optional, device)
    num_token: ll.int32,
    topk: ll.int32,
    num_experts: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    num_experts_per_rank: ll.int32 = num_experts / num_ranks

    ll.align_memory(1024, scope="dynamic_shared")
    scan_warp_prefix_sum = ll.empty([NUM_WARPS], dtype=ll.int32, scope="dynamic_shared")

    # ---- Step 1: Pull mode all-gather ----
    # Each block copies local_splits into remote full_splits
    i_rank: ll.int32 = block_id
    while i_rank < num_ranks:
        remote_full_splits: ll.ptr[ll.int32] = full_splits_ptrs[i_rank] + rank * (num_experts + 1)
        j: ll.int32 = thread_id
        while j < num_experts + 1:
            remote_full_splits[j] = local_splits[j]
            j = j + BLOCK_SIZE
        i_rank = i_rank + num_block

    ll.grid_sync()

    # Cross-rank barrier (only block 0)
    if block_id == 0:
        barrier_all_block(barrier_ptrs, rank, num_ranks)

    ll.grid_sync()

    # ---- Step 2: Cumsum full_splits -> recv_base_offset ----
    local_full_splits_ptr: ll.ptr[ll.int32] = full_splits_ptrs[rank]
    dst_rank: ll.int32 = block_id
    while dst_rank < num_ranks:
        src_rank: ll.int32 = thread_id % num_ranks
        local_expert_idx: ll.int32 = thread_id / num_ranks
        value: ll.int32 = 0
        if thread_id < num_experts:
            value = local_full_splits_ptr[src_rank * (num_experts + 1) + dst_rank * num_experts_per_rank +
                                          local_expert_idx]
        prefix_sum: ll.int32 = block_scan_inclusive(value, scan_warp_prefix_sum)
        if thread_id < num_experts:
            recv_base_offset[dst_rank * num_experts + local_expert_idx * num_ranks + src_rank] = prefix_sum - value

        if thread_id == num_experts - 1:
            if recv_token_count != 0:
                recv_token_count[dst_rank] = prefix_sum
            if recv_token_count_cpu != 0:
                recv_token_count_cpu[dst_rank] = prefix_sum
            ll.threadfence_system()
        ll.__syncthreads()
        dst_rank = dst_rank + num_block

    ll.grid_sync()

    # ---- Step 3: Token scatter indices and send mask ----
    i: ll.int32 = block_id * BLOCK_SIZE + thread_id
    while i < num_token * topk:
        expert_idx: ll.int32 = topk_indices[i]
        target_rank: ll.int32 = expert_idx / num_experts_per_rank
        target_local_expert_idx: ll.int32 = expert_idx % num_experts_per_rank
        num_pre_expert: ll.int32 = i % topk
        need_send: ll.int32 = 1 if target_rank < num_ranks else 0
        ll.__syncwarp()
        if need_send == 1:
            j2: ll.int32 = 0
            while j2 < num_pre_expert:
                cur_expert_idx: ll.int32 = topk_indices[i / topk * topk + j2]
                cur_target_rank: ll.int32 = cur_expert_idx / num_experts_per_rank
                if cur_target_rank == target_rank:
                    need_send = 0
                j2 = j2 + 1
        ll.__syncwarp()
        scatter_idx: ll.int32 = -1
        if target_rank < num_ranks:
            scatter_idx = recv_base_offset[target_rank * num_experts + target_local_expert_idx * num_ranks +
                                           rank] + token_within_expert_offset[i]
        token_dst_scatter_indices[i] = scatter_idx
        token_topk_send_mask[i] = need_send
        i = i + BLOCK_SIZE * num_block


# ============================================================================
# Codegen / build helpers
# ============================================================================
def show_generated_code(kernel_name="offset"):
    if kernel_name == "layout":
        kernel = kernel_compute_dispatch_layout
    else:
        kernel = kernel_compute_offset
    cuda_code = kernel.compile(passes=PASSES["cuda"], codegen_func=codegen_cuda, need_header=True,
                               num_threads=BLOCK_SIZE)
    print("=" * 80)
    print(f"Generated CUDA code for {kernel_name}:")
    print("=" * 80)
    print(cuda_code)
    return cuda_code


def build_kernel(kernel_name="offset"):
    if kernel_name == "layout":
        kernel = kernel_compute_dispatch_layout
        smem = NUM_WARPS * 4  # scan_warp_prefix_sum: 32 int32s
    else:
        kernel = kernel_compute_offset
        smem = NUM_WARPS * MAX_EXPERTS_PLUS_1 * 4
    compiled = kernel.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(128, 1, 1),
                            block=(BLOCK_SIZE, 1, 1), shared_mem_bytes=smem, verbose=True)
    return compiled


if __name__ == "__main__":
    import sys
    kernel_name = "offset"
    if "--layout" in sys.argv:
        kernel_name = "layout"
    if "--codegen" in sys.argv:
        show_generated_code(kernel_name)
    else:
        print(f"=== Generating CUDA code ({kernel_name}) ===")
        cuda_code = show_generated_code(kernel_name)
        print(f"\n=== Compiling kernel ({kernel_name}) ===")
        compiled = build_kernel(kernel_name)
        print("Kernel compiled successfully!")
