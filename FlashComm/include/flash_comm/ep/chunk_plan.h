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

/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_comm {
namespace ep {
namespace chunk_plan {

size_t workspace_numel(int32_t num_chunks, int32_t num_experts,
                       int32_t num_ranks);

void build_ep_chunk_plan_cuda(
    const int32_t *topk_indices, int32_t num_token, int32_t topk,
    int32_t num_experts, int32_t chunk_size, int32_t max_num_tokens,
    int32_t recv_capacity_tokens, int32_t expert_alignment,
    int32_t launch_num_sms, int32_t rank, int32_t num_ranks,
    int32_t lsa_world_size, int32_t gin_context, uintptr_t workspace_win_handle,
    const void *dev_comm_host, int32_t *logical_token_ranges,
    int32_t *rank_chunk_prefix, cudaStream_t stream);

void build_ep_chunk_layouts_cuda(
    const int32_t *topk_indices, const int32_t *token_within_expert_offset,
    const int32_t *logical_token_ranges, const int32_t *rank_chunk_prefix,
    int32_t num_token, int32_t topk, int32_t num_experts, int32_t chunk_size,
    int32_t num_chunks, int32_t rank, int32_t num_ranks,
    int32_t expert_alignment, int32_t *recv_base_offset,
    int32_t *token_dst_scatter_indices, int32_t *token_topk_send_mask,
    int32_t *recv_token_count, int32_t *recv_aligned_token_count,
    int32_t *recv_expert_counts, int32_t *num_tokens_per_rank,
    cudaStream_t stream);

} // namespace chunk_plan
} // namespace ep
} // namespace flash_comm
