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
FlashComm kernel: kernel_dispatch_postprocess_tma
Scatter received tokens to correct positions via TMA pipeline.

Fixed config: bf16 tokens, float32 weights, int32 offsets
kHiddenSize=7168, kNumStages=2, kWritePipeCount=1
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

backend = "cuda"
WARP_SIZE = 32
HIDDEN_SIZE = 7168
NUM_STAGES = 2
WRITE_PIPE_COUNT = 1
TOTAL_THREADS = 1024


@ll_kernel(backend=backend, is_entry=True)
def kernel_dispatch_postprocess_tma(
    recv_x: ll.Tensor[ll.bfloat16],  # [num_recv_worst_token, hidden_size]
    recv_topk_weights: ll.Tensor[ll.float32],  # [num_recv_worst_token, topk]
    recv_topk_scatter_indices_comm_buffer: ll.Tensor[ll.int32],  # [num_recv_worst_token, topk]
    recv_token_count: ll.Tensor[ll.int32],  # [num_ranks]
    dispatch_weights: ll.Tensor[ll.float32],  # [num_recv_worst_token]
    recv_topk_scatter_indices: ll.Tensor[ll.int32],  # [num_recv_worst_token, topk]
    hidden_size: ll.int32,
    topk: ll.int32,
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    block_id: ll.int32 = ll.blockIdx_x()
    num_block: ll.int32 = ll.gridDim_x()
    warp_id: ll.int32 = thread_id / WARP_SIZE
    lane_id: ll.int32 = thread_id % WARP_SIZE
    num_recv_token: ll.int32 = recv_token_count[rank]
    bytes_per_token: ll.int32 = hidden_size * 2  # bf16

    ll.align_memory(1024, scope="dynamic_shared")
    mbar_full = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    mbar_empty = ll.empty([NUM_STAGES], dtype=ll.uint64, scope="dynamic_shared")
    tma_buffer = ll.empty([NUM_STAGES * HIDDEN_SIZE], dtype=ll.bfloat16, scope="dynamic_shared")

    if warp_id == 0:
        if ll.elect_one_sync():
            for i_s in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_full + i_s, 1)
            for i_s2 in range(NUM_STAGES):
                ll.init_smem_barrier(mbar_empty + i_s2, 1)

    # Pipeline state for producer (warp 0)
    prod_idx: ll.int32 = 0
    prod_phase: ll.int32 = 1
    # Pipeline states for consumer (warp 1)
    cons_read_idx: ll.int32 = 0
    cons_read_phase: ll.int32 = 0
    cons_read_count: ll.int32 = 0
    cons_write_idx: ll.int32 = 0
    cons_write_phase: ll.int32 = 0

    ll.__syncthreads()

    # ---- Producer warp (warp 0) ----
    if warp_id == 0:
        token_offset: ll.int32 = block_id
        while token_offset < num_recv_token:
            j: ll.int32 = 0
            while j < topk:
                dst_idx: ll.int32 = recv_topk_scatter_indices_comm_buffer[token_offset * topk + j]
                if dst_idx != -1:
                    if dst_idx != token_offset:
                        # Need to copy: src = recv_x[token_offset], dst = recv_x[dst_idx]
                        ll.mbarrier_wait(mbar_empty + prod_idx, prod_phase)
                        if ll.elect_one_sync():
                            ll.mbarrier_arrive_and_expect_tx(mbar_full + prod_idx, bytes_per_token)
                            ll.tma_copy_1d_g2s(
                                recv_x + token_offset * hidden_size,
                                mbar_full + prod_idx,
                                tma_buffer + prod_idx * HIDDEN_SIZE,
                                bytes_per_token,
                            )
                        ll.__syncwarp()
                        prod_idx = prod_idx + 1
                        if prod_idx == NUM_STAGES:
                            prod_idx = 0
                            prod_phase = prod_phase ^ 1
                j = j + 1
            token_offset = token_offset + num_block

    # ---- Consumer warp (warp 1) ----
    if warp_id == 1:
        is_leader: ll.int32 = ll.elect_one_sync()
        token_offset2: ll.int32 = block_id
        while token_offset2 < num_recv_token:
            j2: ll.int32 = 0
            while j2 < topk:
                dst_idx2: ll.int32 = recv_topk_scatter_indices_comm_buffer[token_offset2 * topk + j2]
                if dst_idx2 != -1:
                    if dst_idx2 != token_offset2:
                        # Consumer wait for producer
                        ll.mbarrier_wait(mbar_full + cons_read_idx, cons_read_phase)
                        if is_leader:
                            ll.tma_copy_1d_s2g(
                                tma_buffer + cons_read_idx * HIDDEN_SIZE,
                                recv_x + dst_idx2 * hidden_size,
                                bytes_per_token,
                            )
                            ll.tma_store_arrive()
                        ll.tma_store_wait_n(WRITE_PIPE_COUNT)
                        ll.__syncwarp()
                        # Release empty barrier when enough stores are in-flight
                        if cons_read_count >= WRITE_PIPE_COUNT:
                            if is_leader:
                                ll.mbarrier_arrive(mbar_empty + cons_write_idx)
                            cons_write_idx = cons_write_idx + 1
                            if cons_write_idx == NUM_STAGES:
                                cons_write_idx = 0
                                cons_write_phase = cons_write_phase ^ 1
                        cons_read_idx = cons_read_idx + 1
                        if cons_read_idx == NUM_STAGES:
                            cons_read_idx = 0
                            cons_read_phase = cons_read_phase ^ 1
                        cons_read_count = cons_read_count + 1
                ll.__syncwarp()
                # Reset comm buffer and write final indices + weights
                if lane_id == 0:
                    recv_topk_scatter_indices_comm_buffer[token_offset2 * topk + j2] = -1
                    recv_topk_scatter_indices[token_offset2 * topk + j2] = dst_idx2
                    # Scatter weight
                    weight_val: ll.float32 = recv_topk_weights[token_offset2 * topk + j2]
                    if dst_idx2 != -1:
                        dispatch_weights[dst_idx2] = weight_val
                ll.__syncwarp()
                j2 = j2 + 1
            token_offset2 = token_offset2 + num_block

    ll.tma_store_wait_n(0)
    ll.__syncthreads()


# Shared memory size
TOTAL_SMEM = 1024 + (NUM_STAGES * 2) * 8 + NUM_STAGES * HIDDEN_SIZE * 2 + 4096


def show_generated_code():
    kernel = kernel_dispatch_postprocess_tma
    cuda_code = kernel.compile(passes=PASSES["cuda"], codegen_func=codegen_cuda, need_header=True,
                               num_threads=TOTAL_THREADS)
    print("=" * 80)
    print("Generated CUDA code (dispatch_postprocess_tma):")
    print("=" * 80)
    print(cuda_code)
    return cuda_code


def build_kernel():
    compiled = kernel_dispatch_postprocess_tma.build(passes=PASSES["cuda"], codegen_func=codegen_cuda, grid=(128, 1, 1),
                                                     block=(TOTAL_THREADS, 1, 1), shared_mem_bytes=TOTAL_SMEM,
                                                     verbose=True)
    return compiled


if __name__ == "__main__":
    import sys
    if "--codegen" in sys.argv:
        show_generated_code()
    else:
        print("=== Generating CUDA code ===")
        cuda_code = show_generated_code()
        print("\n=== Compiling kernel ===")
        compiled = build_kernel()
        print("Kernel compiled successfully!")
