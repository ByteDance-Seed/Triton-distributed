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
namespace internode {
namespace detail {

struct DispatchInternodeArgs {
  uintptr_t rdma_rail_send_win_handle;
  int32_t *num_tokens_per_rank;
  int32_t *node_topk_indices;
  int32_t *node_topk_send_mask;
  int32_t *node_token_dst_scatter_indices;
  bool has_weight;
  void *recv_x_ptrs;
  void **recv_weights_ptrs;
  void **recv_topk_scatter_indices_ptrs;
  int32_t max_slot_num_token;
  int32_t num_experts_per_rank;
  int32_t rank;
  int32_t num_ranks;
  int32_t local_world_size;
  int32_t max_recv_tokens;
  int32_t num_sm;
  const void *dev_comm_host;
  int32_t num_qps;
  FlashCommDType dtype;
  FlashCommDType weight_dtype;
  FlashCommDType offset_dtype;
  int32_t topk;
  int32_t dispatch_pipeline_chunks;
  const int32_t *logical_token_range;
  cudaStream_t stream;
};

struct CombineInternodeArgs {
  void *combine_x_ptrs;
  void *combine_weight_ptrs;
  uintptr_t rdma_rail_send_win_handle;
  void *output;
  void *output_weight;
  int32_t *local_topk_indices;
  int32_t *local_topk_send_mask;
  int32_t *local_token_dst_scatter_indices;
  int32_t *num_tokens_per_rank;
  bool has_weight;
  int32_t num_token;
  int32_t max_slot_num_token;
  int32_t topk;
  int32_t num_experts_per_rank;
  int32_t rank;
  int32_t num_ranks;
  int32_t local_world_size;
  int32_t num_sm;
  const void *dev_comm_host;
  int32_t num_qps;
  FlashCommDType dtype;
  FlashCommDType weight_dtype;
  FlashCommDType offset_dtype;
  int32_t combine_pipeline_chunks;
  const int32_t *logical_token_range;
  cudaStream_t stream;
};

template <int32_t kHiddenSize>
void dispatch_internode_cuda_hidden(const DispatchInternodeArgs &args);

template <int32_t kHiddenSize>
void combine_internode_cuda_hidden(const CombineInternodeArgs &args);

} // namespace detail
} // namespace internode
} // namespace ep
} // namespace flash_comm
