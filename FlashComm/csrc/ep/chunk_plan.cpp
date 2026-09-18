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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "flash_comm/buffer/nccl_gin.h"
#include "flash_comm/ep/chunk_plan.h"
#include "flash_comm/nccl_utils.h"
#include "flash_comm/torch_utils.h"

namespace flash_comm {
namespace ep {
namespace chunk_plan {

namespace {

void check_topk_indices(const torch::Tensor &topk_indices) {
  check_tensor_common(topk_indices, "topk_indices", true, torch::kInt32, 2);
  FLASH_CHECK(topk_indices.size(0) <= std::numeric_limits<int32_t>::max() &&
              topk_indices.size(1) <= std::numeric_limits<int32_t>::max())
      << "topk_indices dimensions must fit int32";
}

} // namespace

void build_ep_chunk_plan_out(torch::Tensor topk_indices, int32_t num_experts,
                             int32_t chunk_size, int32_t max_num_tokens,
                             int32_t recv_capacity_tokens,
                             int32_t expert_alignment, int32_t launch_num_sms,
                             torch::Tensor workspace,
                             uintptr_t workspace_win_handle,
                             torch::Tensor logical_token_ranges,
                             torch::Tensor rank_chunk_prefix) {
  check_topk_indices(topk_indices);
  FLASH_CHECK(buffer::nccl_gin_is_initialized());
  auto &gin = buffer::nccl_gin_require_state();
  const int32_t rank = gin.rank;
  const int32_t num_ranks = gin.nranks;
  const int32_t num_token = static_cast<int32_t>(topk_indices.size(0));
  const int32_t topk = static_cast<int32_t>(topk_indices.size(1));
  FLASH_CHECK(max_num_tokens > 0 && chunk_size > 0);
  FLASH_CHECK(num_token >= 0 && num_token <= max_num_tokens);
  FLASH_CHECK(topk > 0);
  FLASH_CHECK(num_experts > 0 && num_experts <= 1024);
  FLASH_CHECK(num_ranks > 0 && num_ranks <= 32);
  FLASH_CHECK(num_experts % num_ranks == 0);
  FLASH_CHECK(recv_capacity_tokens > 0 && expert_alignment > 0);
  FLASH_CHECK(launch_num_sms >= 0) << "launch_num_sms must be non-negative";
  const int64_t routes_per_token = static_cast<int64_t>(num_ranks) * topk;
  FLASH_CHECK(routes_per_token > 0 &&
              max_num_tokens <=
                  std::numeric_limits<int32_t>::max() / routes_per_token)
      << "maximum global route count must fit int32";
  const int64_t max_global_routes =
      routes_per_token * static_cast<int64_t>(max_num_tokens);
  const int64_t max_chunk_tokens =
      std::min<int64_t>(chunk_size, max_num_tokens);
  const int64_t assignments =
      static_cast<int64_t>(num_ranks) * max_chunk_tokens * topk;
  const int64_t experts_per_rank = num_experts / num_ranks;
  const int64_t guaranteed_capacity =
      assignments + std::min(experts_per_rank, assignments) *
                        (static_cast<int64_t>(expert_alignment) - 1);
  const int64_t max_aligned_footprint =
      max_global_routes + std::min(experts_per_rank, max_global_routes) *
                              (static_cast<int64_t>(expert_alignment) - 1);
  FLASH_CHECK(max_aligned_footprint <= std::numeric_limits<int32_t>::max())
      << "maximum aligned receive footprint must fit int32";
  FLASH_CHECK(recv_capacity_tokens >= guaranteed_capacity)
      << "recv_capacity_tokens cannot guarantee one logical chunk fits: got "
      << recv_capacity_tokens << ", require at least " << guaranteed_capacity;
  FLASH_CHECK(workspace_win_handle != 0);
  FLASH_CHECK(gin.gin_contexts > gin.ep_num_qps)
      << "EP chunk planning requires one GIN context in addition to the "
         "EP communication contexts";
  const int32_t num_chunks = 1 + (max_num_tokens - 1) / chunk_size;
  FLASH_CHECK(static_cast<int64_t>(2) * num_chunks + 2 <=
                  std::numeric_limits<int32_t>::max() &&
              static_cast<int64_t>(num_chunks + 1) * (num_experts + 1) <=
                  std::numeric_limits<int32_t>::max())
      << "chunk-plan metadata exceeds int32 indexing";

  check_tensor_common(workspace, "workspace", true, torch::kInt32, 1);
  FLASH_CHECK(static_cast<size_t>(workspace.numel()) >=
              workspace_numel(num_chunks, num_experts, num_ranks));
  void *workspace_ptr = nullptr;
  NCCL_CHECK(ncclWinGetUserPtr(
      buffer::nccl_gin_comm(),
      reinterpret_cast<ncclWindow_t>(workspace_win_handle), &workspace_ptr));
  FLASH_CHECK(workspace_ptr == workspace.data_ptr())
      << "workspace does not own workspace_win_handle";

  check_tensor_common(logical_token_ranges, "logical_token_ranges", true,
                      torch::kInt32, 2);
  check_tensor_common(rank_chunk_prefix, "rank_chunk_prefix", true,
                      torch::kInt32, 3);
  check_tensor_shape(logical_token_ranges, "logical_token_ranges",
                     {num_chunks, 2});
  check_tensor_shape(rank_chunk_prefix, "rank_chunk_prefix",
                     {num_ranks, num_chunks + 1, num_experts + 1});

  build_ep_chunk_plan_cuda(
      topk_indices.data_ptr<int32_t>(), num_token, topk, num_experts,
      chunk_size, max_num_tokens, recv_capacity_tokens, expert_alignment,
      launch_num_sms, rank, num_ranks, gin.lsa_size, gin.ep_num_qps,
      workspace_win_handle,
      static_cast<const void *>(buffer::nccl_gin_dev_comm()),
      logical_token_ranges.data_ptr<int32_t>(),
      rank_chunk_prefix.data_ptr<int32_t>(), at::cuda::getCurrentCUDAStream());
}

void build_ep_chunk_layouts_out(
    torch::Tensor topk_indices, torch::Tensor token_within_expert_offset,
    torch::Tensor logical_token_ranges, torch::Tensor rank_chunk_prefix,
    int32_t chunk_size, int32_t expert_alignment,
    torch::Tensor recv_base_offset, torch::Tensor token_dst_scatter_indices,
    torch::Tensor token_topk_send_mask, torch::Tensor recv_token_count,
    torch::Tensor recv_aligned_token_count, torch::Tensor recv_expert_counts,
    c10::optional<torch::Tensor> optional_num_tokens_per_rank) {
  check_topk_indices(topk_indices);
  check_topk_indices(token_within_expert_offset);
  FLASH_CHECK(topk_indices.sizes() == token_within_expert_offset.sizes());
  FLASH_CHECK(buffer::nccl_gin_is_initialized());
  auto &gin = buffer::nccl_gin_require_state();
  const int32_t rank = gin.rank;
  const int32_t num_ranks = gin.nranks;
  const int32_t num_token = static_cast<int32_t>(topk_indices.size(0));
  const int32_t topk = static_cast<int32_t>(topk_indices.size(1));
  FLASH_CHECK(chunk_size > 0 && expert_alignment > 0);

  check_tensor_common(logical_token_ranges, "logical_token_ranges", true,
                      torch::kInt32, 2);
  check_tensor_common(rank_chunk_prefix, "rank_chunk_prefix", true,
                      torch::kInt32, 3);
  FLASH_CHECK(logical_token_ranges.size(0) > 0 &&
              logical_token_ranges.size(0) <=
                  std::numeric_limits<int32_t>::max() - 1)
      << "logical_token_ranges dim0 must fit a positive int32";
  FLASH_CHECK(rank_chunk_prefix.size(2) > 1 &&
              rank_chunk_prefix.size(2) <=
                  static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1)
      << "rank_chunk_prefix expert dimension must fit int32";
  const int32_t num_chunks = static_cast<int32_t>(logical_token_ranges.size(0));
  FLASH_CHECK(logical_token_ranges.size(1) == 2);
  const int32_t num_experts =
      static_cast<int32_t>(rank_chunk_prefix.size(2) - 1);
  FLASH_CHECK(num_experts > 0 && num_experts <= 1024);
  FLASH_CHECK(num_experts % num_ranks == 0);
  const int32_t experts_per_rank = num_experts / num_ranks;
  check_tensor_shape(rank_chunk_prefix, "rank_chunk_prefix",
                     {num_ranks, num_chunks + 1, num_experts + 1});

  check_tensor_common(recv_base_offset, "recv_base_offset", true, torch::kInt32,
                      4);
  check_tensor_common(token_dst_scatter_indices, "token_dst_scatter_indices",
                      true, torch::kInt32, 3);
  check_tensor_common(token_topk_send_mask, "token_topk_send_mask", true,
                      torch::kInt32, 3);
  check_tensor_common(recv_token_count, "recv_token_count", true, torch::kInt32,
                      2);
  check_tensor_common(recv_aligned_token_count, "recv_aligned_token_count",
                      true, torch::kInt32, 2);
  check_tensor_common(recv_expert_counts, "recv_expert_counts", true,
                      torch::kInt32, 2);
  check_tensor_shape(recv_base_offset, "recv_base_offset",
                     {num_chunks, num_ranks, experts_per_rank, num_ranks});
  check_tensor_shape(token_dst_scatter_indices, "token_dst_scatter_indices",
                     {num_chunks, num_token, topk});
  check_tensor_shape(token_topk_send_mask, "token_topk_send_mask",
                     {num_chunks, num_token, topk});
  check_tensor_shape(recv_token_count, "recv_token_count",
                     {num_chunks, num_ranks});
  check_tensor_shape(recv_aligned_token_count, "recv_aligned_token_count",
                     {num_chunks, num_ranks});
  check_tensor_shape(recv_expert_counts, "recv_expert_counts",
                     {num_chunks, experts_per_rank});

  int32_t *num_tokens_per_rank = nullptr;
  if (optional_num_tokens_per_rank.has_value() &&
      optional_num_tokens_per_rank->defined()) {
    auto tensor = *optional_num_tokens_per_rank;
    check_tensor_common(tensor, "num_tokens_per_rank", true, torch::kInt32, 2);
    check_tensor_shape(tensor, "num_tokens_per_rank", {num_chunks, num_ranks});
    num_tokens_per_rank = tensor.data_ptr<int32_t>();
  }

  build_ep_chunk_layouts_cuda(
      topk_indices.data_ptr<int32_t>(),
      token_within_expert_offset.data_ptr<int32_t>(),
      logical_token_ranges.data_ptr<int32_t>(),
      rank_chunk_prefix.data_ptr<int32_t>(), num_token, topk, num_experts,
      chunk_size, num_chunks, rank, num_ranks, expert_alignment,
      recv_base_offset.data_ptr<int32_t>(),
      token_dst_scatter_indices.data_ptr<int32_t>(),
      token_topk_send_mask.data_ptr<int32_t>(),
      recv_token_count.data_ptr<int32_t>(),
      recv_aligned_token_count.data_ptr<int32_t>(),
      recv_expert_counts.data_ptr<int32_t>(), num_tokens_per_rank,
      at::cuda::getCurrentCUDAStream());
}

} // namespace chunk_plan
} // namespace ep
} // namespace flash_comm

void bind_ep_chunk_plan_ops(py::module &m) {
  m.def("workspace_numel", &flash_comm::ep::chunk_plan::workspace_numel,
        py::arg("num_chunks"), py::arg("num_experts"), py::arg("num_ranks"));
  m.def("build_ep_chunk_plan_out",
        &flash_comm::ep::chunk_plan::build_ep_chunk_plan_out,
        py::arg("topk_indices"), py::arg("num_experts"), py::arg("chunk_size"),
        py::arg("max_num_tokens"), py::arg("recv_capacity_tokens"),
        py::arg("expert_alignment"), py::arg("launch_num_sms") = 0,
        py::arg("workspace"), py::arg("workspace_win_handle"),
        py::arg("logical_token_ranges"), py::arg("rank_chunk_prefix"),
        "Build a caller-owned EP chunk plan with one NCCL device kernel.");
  m.def("build_ep_chunk_layouts_out",
        &flash_comm::ep::chunk_plan::build_ep_chunk_layouts_out,
        py::arg("topk_indices"), py::arg("token_within_expert_offset"),
        py::arg("logical_token_ranges"), py::arg("rank_chunk_prefix"),
        py::arg("chunk_size"), py::arg("expert_alignment"),
        py::arg("recv_base_offset"), py::arg("token_dst_scatter_indices"),
        py::arg("token_topk_send_mask"), py::arg("recv_token_count"),
        py::arg("recv_aligned_token_count"), py::arg("recv_expert_counts"),
        py::arg("num_tokens_per_rank") = c10::nullopt,
        "Build all EP chunk layouts with one local device kernel.");
}
