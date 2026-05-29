/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include "flash_comm/common.h"
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_comm {
namespace ep {
namespace intranode {

// Compute stable local token-within-expert offsets and per-tile per-expert
// exclusive prefixes. If expert_counts is non-null, it will be filled with
// per-expert token counts (including drop token): expert_counts: [num_experts +
// 1] int32
void compute_stable_local_token_within_expert_offset_cuda(
    int32_t *topk_indices, int32_t num_token, int32_t topk, int32_t num_experts,
    int32_t *block_cumsum_hist, int32_t *token_within_expert_offset,
    int32_t *expert_counts, int32_t num_sm, cudaStream_t stream);

void compute_dispatch_layout_cuda(
    int32_t *topk_indices, int32_t *token_within_expert_offset,
    int32_t *local_splits, int32_t **full_splits_ptrs, int32_t **barrier_ptrs,
    int32_t *recv_base_offset, int32_t *token_dst_scatter_indices,
    int32_t *token_topk_send_mask, int32_t *recv_token_count_cpu,
    int32_t *recv_token_count, int32_t *recv_aligned_token_count_cpu,
    int32_t *recv_aligned_token_count, int32_t *recv_expert_counts,
    int64_t **token_src_rank_topk_and_indices_ptrs, int32_t num_token,
    int32_t topk, int32_t num_experts, int32_t rank, int32_t num_ranks,
    int32_t num_sm, int32_t expert_alignment, cudaStream_t stream);

void dispatch_intranode_cuda(
    void *x, void *topk_send_mask, void *topk_weights, void *topk_indices,
    void *token_dst_scatter_indices, void *recv_x_ptrs,
    void **recv_weights_ptrs, void **recv_topk_scatter_indices_ptrs,
    int32_t num_token, int32_t hidden_size, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks, int32_t num_sm, FlashCommDType dtype,
    FlashCommDType weight_dtype, FlashCommDType offset_dtype, int32_t topk,
    cudaStream_t stream);

void dispatch_postprocess_cuda(
    void *recv_x, void *recv_topk_scatter_indices_comm_buffer,
    void *recv_topk_weights, int32_t *recv_token_count, void *dispatch_weights,
    void *recv_topk_scatter_indices, int32_t num_recv_worst_token,
    int32_t hidden_size, int32_t topk, int32_t rank, int32_t num_ranks,
    int32_t num_sm, FlashCommDType dtype, FlashCommDType weight_dtype,
    FlashCommDType offset_dtype, cudaStream_t stream);

void combine_intranode_cuda(void *x_ptrs, void *weight_ptrs,
                            void *topk_send_mask, void *topk_indices,
                            void *token_dst_scatter_indices, void *recv_x,
                            void *recv_weight, int32_t num_token,
                            int32_t hidden_size, int32_t topk,
                            int32_t num_experts_per_rank, int32_t rank,
                            int32_t num_ranks, int32_t num_sm,
                            FlashCommDType dtype, FlashCommDType weight_dtype,
                            FlashCommDType offset_dtype, cudaStream_t stream);

void combine_preprocess_inplace_cuda(
    void *x, void *weight_ptrs, int32_t *recv_token_count,
    void *recv_topk_scatter_indices, void *recv_topk_weight,
    int32_t num_recv_worst_token, int32_t hidden_size, int32_t topk,
    int32_t rank, int32_t num_ranks, int32_t num_sm, FlashCommDType dtype,
    FlashCommDType weight_dtype, FlashCommDType offset_dtype,
    cudaStream_t stream);

void barrier_all_on_stream_cuda(void **barrier_ptrs, int32_t rank,
                                int32_t num_ranks, FlashCommDType dtype,
                                cudaStream_t stream);

} // namespace intranode
} // namespace ep
} // namespace flash_comm
