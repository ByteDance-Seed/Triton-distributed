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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <mutex>

#include "flash_comm/common.h"
#include "flash_comm/ep/intranode.h"
#include "flash_comm/quantization/mxfp8.h"
#include "flash_comm/torch_utils.h"

namespace flash_comm {
namespace ep {
namespace intranode {

using flash_comm::quantization::kMXFP8BlockSize;
using flash_comm::quantization::kMXFP8MaxIntranodeRanks;
using flash_comm::quantization::mxfp8_packed_row_bytes;

#define WARP_SIZE 32

static inline void check_uva_enabled_for_current_device() {
  static std::once_flag once;
  std::call_once(once, []() {
    int dev = -1;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    FLASH_CHECK(prop.unifiedAddressing == 1)
        << "Current CUDA device does not support unified virtual addressing "
           "(unifiedAddressing==0). "
           "Passing a pinned CPU pointer directly to a CUDA kernel requires "
           "UVA. "
           "Please run on Ampere+ GPUs / enable UVA, or change the "
           "implementation to use a device buffer + D2H copy.";
  });
}

static inline void check_topk_indices(const torch::Tensor &topk_indices) {
  check_tensor_common(topk_indices, "topk_indices", /*expect_cuda=*/true,
                      torch::kInt32, /*expect_dim=*/2);
}

static inline void check_mxfp8_dispatch_args(int32_t rank, int32_t num_ranks,
                                             int32_t num_experts_per_rank,
                                             int32_t num_sm) {
  FLASH_CHECK(num_ranks > 0 && num_ranks <= kMXFP8MaxIntranodeRanks)
      << "MXFP8 intranode dispatch supports 1.." << kMXFP8MaxIntranodeRanks
      << " ranks";
  FLASH_CHECK(rank >= 0 && rank < num_ranks)
      << "rank must be between 0 and num_ranks - 1, " << rank << " not in [0, "
      << num_ranks << ")";
  FLASH_CHECK(num_experts_per_rank > 0)
      << "num_experts_per_rank must be positive";
  FLASH_CHECK(num_sm > 0) << "num_sm must be positive for MXFP8 dispatch";
}

static inline int32_t check_mxfp8_dispatch_layout_tensors(
    const torch::Tensor &topk_send_mask, const torch::Tensor &topk_indices,
    const torch::Tensor &token_dst_scatter_indices, int32_t num_token) {
  check_tensor_common(topk_send_mask, "topk_send_mask", true, torch::kInt32, 2);
  check_topk_indices(topk_indices);
  check_tensor_common(token_dst_scatter_indices, "token_dst_scatter_indices",
                      true, torch::kInt32, 2);
  int32_t topk = topk_indices.size(1);
  check_tensor_shape(topk_send_mask, "topk_send_mask", {num_token, topk});
  check_tensor_shape(topk_indices, "topk_indices", {num_token, topk});
  check_tensor_shape(token_dst_scatter_indices, "token_dst_scatter_indices",
                     {num_token, topk});
  return topk;
}

static inline void check_expert_counts(const torch::Tensor &expert_counts,
                                       int32_t num_experts) {
  check_tensor_common(expert_counts, "expert_counts", /*expect_cuda=*/true,
                      torch::kInt32, /*expect_dim=*/1);
  check_tensor_shape(expert_counts, "expert_counts", {num_experts + 1});
}

std::tuple<torch::Tensor, torch::Tensor>
compute_stable_local_token_within_expert_offset(torch::Tensor topk_indices,
                                                int32_t num_experts,
                                                int32_t num_sm) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_topk_indices(topk_indices);
  auto offset_dtype = get_flash_comm_dtype(topk_indices.scalar_type());
  int32_t topk = topk_indices.size(1);
  int32_t num_token = topk_indices.size(0);

  // token_within_expert_offset: [num_token, topk], int32 (stable within-expert
  // ordinal)
  torch::Tensor token_within_expert_offset =
      torch::empty({num_token, topk}, topk_indices.options());

  // block_cumsum_hist: [num_tiles, num_experts + 1], int32
  // NOTE: kernel uses tile size = 32 warps * 32 lanes = 1024 elements.
  constexpr int32_t kTileSize = 32 * WARP_SIZE;
  int32_t num_tiles = (num_token * topk + kTileSize - 1) / kTileSize;
  torch::Tensor block_cumsum_hist =
      torch::empty({num_tiles, num_experts + 1}, topk_indices.options());

  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";

  compute_stable_local_token_within_expert_offset_cuda(
      topk_indices.data_ptr<int32_t>(), num_token, topk, num_experts,
      block_cumsum_hist.data_ptr<int32_t>(),
      token_within_expert_offset.data_ptr<int32_t>(), /*expert_count=*/nullptr,
      num_sm, stream);
  return {token_within_expert_offset, block_cumsum_hist};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
compute_stable_local_token_within_expert_offset_and_expert_counts(
    torch::Tensor topk_indices, int32_t num_experts, int32_t num_sm,
    torch::Tensor expert_counts) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_topk_indices(topk_indices);
  auto offset_dtype = get_flash_comm_dtype(topk_indices.scalar_type());
  int32_t topk = topk_indices.size(1);
  int32_t num_token = topk_indices.size(0);

  torch::Tensor token_within_expert_offset =
      torch::empty({num_token, topk}, topk_indices.options());
  constexpr int32_t kTileSize = 32 * WARP_SIZE;
  int32_t num_tiles = (num_token * topk + kTileSize - 1) / kTileSize;
  torch::Tensor block_cumsum_hist =
      torch::empty({num_tiles, num_experts + 1}, topk_indices.options());
  if (expert_counts.defined()) {
    check_expert_counts(expert_counts, num_experts);
  } else {
    expert_counts = torch::empty({num_experts + 1}, topk_indices.options());
  }

  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";

  compute_stable_local_token_within_expert_offset_cuda(
      topk_indices.data_ptr<int32_t>(), num_token, topk, num_experts,
      block_cumsum_hist.data_ptr<int32_t>(),
      token_within_expert_offset.data_ptr<int32_t>(),
      expert_counts.data_ptr<int32_t>(), num_sm, stream);
  return {token_within_expert_offset, block_cumsum_hist, expert_counts};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_dispatch_layout(
    torch::Tensor topk_indices, torch::Tensor token_within_expert_offset,
    torch::Tensor local_splits, torch::Tensor full_splits_ptrs,
    torch::Tensor barrier_ptrs, int32_t num_experts, int32_t rank,
    int32_t num_ranks, int32_t num_sm,
    c10::optional<torch::Tensor> optional_recv_token_count_cpu,
    c10::optional<torch::Tensor> optional_recv_token_count,
    c10::optional<torch::Tensor> optional_token_src_rank_topk_and_indices_ptrs,
    int32_t expert_alignment) {
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  check_topk_indices(topk_indices);
  check_topk_indices(token_within_expert_offset);
  FLASH_CHECK(local_splits.is_cuda() && local_splits.is_contiguous() &&
              local_splits.scalar_type() == torch::kInt32)
      << "local_splits must be contiguous CUDA int32";
  FLASH_CHECK(local_splits.dim() == 1) << "local_splits must be 1D";
  FLASH_CHECK(local_splits.size(0) == (num_experts + 1))
      << "local_splits shape mismatch, expected [" << (num_experts + 1) << "]";

  FLASH_CHECK(topk_indices.sizes() == token_within_expert_offset.sizes())
      << "token_within_expert_offset must have same shape as topk_indices";

  FLASH_CHECK(num_experts % num_ranks == 0)
      << "num_experts must be divisible by num_ranks";

  check_ptrs_tensor_i64(full_splits_ptrs, num_ranks, "full_splits_ptrs");
  check_ptrs_tensor_i64(barrier_ptrs, num_ranks, "barrier_ptrs");

  int32_t topk = topk_indices.size(1);
  int32_t num_token = topk_indices.size(0);
  int32_t experts_per_rank = num_experts / num_ranks;

  // Outputs
  auto opts_i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_indices.device());
  torch::Tensor recv_base_offset =
      torch::empty({num_ranks, experts_per_rank, num_ranks}, opts_i32);
  torch::Tensor token_dst_scatter_indices =
      torch::empty({num_token, topk}, opts_i32);
  torch::Tensor token_topk_send_mask =
      torch::empty({num_token, topk}, opts_i32);

  torch::Tensor recv_token_count_cpu, recv_token_count;
  // recv_token_count_cpu: pinned CPU int32 [num_ranks] (optional)
  if (optional_recv_token_count_cpu.has_value()) {
    check_pinned_cpu_i32_vector(optional_recv_token_count_cpu.value(),
                                num_ranks, "recv_token_count_cpu");
    optional_recv_token_count_cpu.value().fill_(-1);
    recv_token_count_cpu = optional_recv_token_count_cpu.value();
    check_tensor_shape(recv_token_count_cpu, "recv_token_count_cpu",
                       {num_ranks});
  } else {
    auto host_opts = torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(torch::kCPU)
                         .pinned_memory(true);
    recv_token_count_cpu = torch::full({num_ranks}, -1, host_opts);
  }

  if (optional_recv_token_count.has_value()) {
    recv_token_count = optional_recv_token_count.value();
    check_tensor_common(recv_token_count, "recv_token_count", true,
                        torch::kInt32, 1);
    check_tensor_shape(recv_token_count, "recv_token_count", {num_ranks});
  } else {
    auto device_opts = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .device(topk_indices.device());
    recv_token_count = torch::empty({num_ranks}, device_opts);
  }

  // Aligned token counts: when expert_alignment > 1, the aligned total
  // (including padding) is needed for buffer allocation and
  // dispatch_postprocess/ combine_preprocess iteration, while recv_token_count
  // reports the original count.
  torch::Tensor recv_aligned_token_count_cpu, recv_aligned_token_count;
  int32_t *recv_aligned_token_count_cpu_ptr = nullptr;
  int32_t *recv_aligned_token_count_ptr = nullptr;
  if (expert_alignment > 1) {
    auto host_opts = torch::TensorOptions()
                         .dtype(torch::kInt32)
                         .device(torch::kCPU)
                         .pinned_memory(true);
    recv_aligned_token_count_cpu = torch::full({num_ranks}, -1, host_opts);
    auto device_opts = torch::TensorOptions()
                           .dtype(torch::kInt32)
                           .device(topk_indices.device());
    recv_aligned_token_count = torch::empty({num_ranks}, device_opts);
    recv_aligned_token_count_cpu_ptr =
        recv_aligned_token_count_cpu.data_ptr<int32_t>();
    recv_aligned_token_count_ptr = recv_aligned_token_count.data_ptr<int32_t>();
  }

  // Per-expert actual token counts for group GEMM
  auto device_opts_i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_indices.device());
  torch::Tensor recv_expert_counts =
      torch::empty({experts_per_rank}, device_opts_i32);
  int32_t *recv_expert_counts_ptr = recv_expert_counts.data_ptr<int32_t>();

  check_uva_enabled_for_current_device();

  int32_t *recv_token_count_cpu_ptr = recv_token_count_cpu.data_ptr<int32_t>();
  int32_t *recv_token_count_ptr = recv_token_count.data_ptr<int32_t>();

  int64_t **token_src_rank_topk_and_indices_ptrs_data = nullptr;
  if (optional_token_src_rank_topk_and_indices_ptrs.has_value() &&
      optional_token_src_rank_topk_and_indices_ptrs.value().defined()) {
    const auto &t = optional_token_src_rank_topk_and_indices_ptrs.value();
    check_ptrs_tensor_i64(t, num_ranks, "token_src_rank_topk_and_indices_ptrs");
    token_src_rank_topk_and_indices_ptrs_data =
        reinterpret_cast<int64_t **>(t.data_ptr<int64_t>());
  }

  compute_dispatch_layout_cuda(
      topk_indices.data_ptr<int32_t>(),
      token_within_expert_offset.data_ptr<int32_t>(),
      local_splits.data_ptr<int32_t>(),
      reinterpret_cast<int32_t **>(full_splits_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<int32_t **>(barrier_ptrs.data_ptr<int64_t>()),
      recv_base_offset.data_ptr<int32_t>(),
      token_dst_scatter_indices.data_ptr<int32_t>(),
      token_topk_send_mask.data_ptr<int32_t>(), recv_token_count_cpu_ptr,
      recv_token_count_ptr, recv_aligned_token_count_cpu_ptr,
      recv_aligned_token_count_ptr, recv_expert_counts_ptr,
      token_src_rank_topk_and_indices_ptrs_data, num_token, topk, num_experts,
      rank, num_ranks, num_sm, expert_alignment, stream);

  record_pinned_tensor(recv_token_count_cpu, stream);
  if (expert_alignment > 1) {
    record_pinned_tensor(recv_aligned_token_count_cpu, stream);
  }

  return {recv_base_offset,         token_dst_scatter_indices,
          token_topk_send_mask,     recv_token_count_cpu,
          recv_token_count,         recv_aligned_token_count_cpu,
          recv_aligned_token_count, recv_expert_counts};
}

static void dispatch_intranode_impl(
    torch::Tensor x,              // [num_token, hidden_size]
    torch::Tensor topk_send_mask, // [num_token, topk]
    c10::optional<torch::Tensor> optional_topk_weights, // [num_token, topk]
    torch::Tensor topk_indices,                         // [num_token, topk]
    torch::Tensor token_dst_scatter_indices,            // [num_token, topk]
    // outputs
    torch::Tensor recv_x_ptrs, // [num_ranks]
    torch::Tensor
        recv_weights_ptrs, // [num_ranks], recv_weights [num_recv_token, topk]
    torch::Tensor recv_topk_scatter_indices_ptrs, // [num_ranks],
                                                  // recv_topk_scatter_indices
                                                  // [num_recv_token, topk]
    int32_t rank, int32_t num_ranks, int32_t num_experts_per_rank,
    int32_t num_sm, c10::optional<torch::Tensor> optional_logical_token_range) {
  // check num_ranks <= nvlink domain size
  FLASH_CHECK(x.dim() == 2 && topk_indices.dim() == 2);
  const int32_t num_token = static_cast<int32_t>(x.size(0));
  const int32_t hidden_size = static_cast<int32_t>(x.size(1));
  const int32_t topk = static_cast<int32_t>(topk_indices.size(1));
  if (optional_topk_weights.has_value() &&
      optional_topk_weights.value().defined()) {
    check_tensor_shape(optional_topk_weights.value(), "topk_weights",
                       {num_token, topk});
  }
  FLASH_CHECK(topk_indices.sizes() == token_dst_scatter_indices.sizes());
  FLASH_CHECK(topk_indices.sizes() == topk_send_mask.sizes());
  FLASH_CHECK(topk_indices.size(0) == num_token);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto dtype = get_flash_comm_dtype(x.scalar_type());
  const auto offset_dtype = get_flash_comm_dtype(topk_indices.scalar_type());
  auto weight_dtype = FlashCommDType::Float32;
  void *topk_weights_ptr = nullptr;
  if (optional_topk_weights.has_value() &&
      optional_topk_weights.value().defined()) {
    weight_dtype =
        get_flash_comm_dtype(optional_topk_weights.value().scalar_type());
    topk_weights_ptr = optional_topk_weights.value().data_ptr();
  }
  FLASH_CHECK(dtype == FlashCommDType::BFloat16)
      << "Only BFloat16 token type is currently supported";
  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";
  if (topk_weights_ptr != nullptr) {
    FLASH_CHECK(weight_dtype == FlashCommDType::Float32)
        << "Only Float32 weight type is currently supported";
  }

  FLASH_CHECK(topk * 2 <= WARP_SIZE);
  FLASH_CHECK(num_ranks * 3 * 8 <= 1024);
  const int32_t *logical_token_range_ptr = nullptr;
  if (optional_logical_token_range.has_value() &&
      optional_logical_token_range.value().defined()) {
    auto logical_token_range = optional_logical_token_range.value();
    check_tensor_common(logical_token_range, "logical_token_range", true,
                        torch::kInt32, 1);
    check_tensor_shape(logical_token_range, "logical_token_range", {2});
    logical_token_range_ptr = logical_token_range.data_ptr<int32_t>();
  }

  dispatch_intranode_cuda(
      x.data_ptr(), topk_send_mask.data_ptr(), topk_weights_ptr,
      topk_indices.data_ptr(), token_dst_scatter_indices.data_ptr(),
      recv_x_ptrs.data_ptr(),
      reinterpret_cast<void **>(recv_weights_ptrs.data_ptr()),
      reinterpret_cast<void **>(recv_topk_scatter_indices_ptrs.data_ptr()),
      num_token, hidden_size, num_experts_per_rank, rank, num_ranks, num_sm,
      dtype, weight_dtype, offset_dtype, topk, logical_token_range_ptr, stream);
}

void dispatch_intranode(torch::Tensor x, torch::Tensor topk_send_mask,
                        c10::optional<torch::Tensor> optional_topk_weights,
                        torch::Tensor topk_indices,
                        torch::Tensor token_dst_scatter_indices,
                        torch::Tensor recv_x_ptrs,
                        torch::Tensor recv_weights_ptrs,
                        torch::Tensor recv_topk_scatter_indices_ptrs,
                        int32_t rank, int32_t num_ranks,
                        int32_t num_experts_per_rank, int32_t num_sm) {
  dispatch_intranode_impl(x, topk_send_mask, optional_topk_weights,
                          topk_indices, token_dst_scatter_indices, recv_x_ptrs,
                          recv_weights_ptrs, recv_topk_scatter_indices_ptrs,
                          rank, num_ranks, num_experts_per_rank, num_sm,
                          c10::nullopt);
}

void dispatch_intranode_range(
    torch::Tensor x, torch::Tensor topk_send_mask,
    c10::optional<torch::Tensor> optional_topk_weights,
    torch::Tensor topk_indices, torch::Tensor token_dst_scatter_indices,
    torch::Tensor logical_token_range, torch::Tensor recv_x_ptrs,
    torch::Tensor recv_weights_ptrs,
    torch::Tensor recv_topk_scatter_indices_ptrs, int32_t rank,
    int32_t num_ranks, int32_t num_experts_per_rank, int32_t num_sm) {
  dispatch_intranode_impl(x, topk_send_mask, optional_topk_weights,
                          topk_indices, token_dst_scatter_indices, recv_x_ptrs,
                          recv_weights_ptrs, recv_topk_scatter_indices_ptrs,
                          rank, num_ranks, num_experts_per_rank, num_sm,
                          logical_token_range);
}

void dispatch_mxfp8_quant_fused_intranode(
    torch::Tensor x, torch::Tensor topk_send_mask,
    c10::optional<torch::Tensor> optional_topk_weights,
    torch::Tensor topk_indices, torch::Tensor token_dst_scatter_indices,
    torch::Tensor recv_x_ptrs, torch::Tensor recv_weights_ptrs,
    torch::Tensor recv_topk_scatter_indices_ptrs, int32_t rank,
    int32_t num_ranks, int32_t num_experts_per_rank, int32_t num_sm) {
  check_mxfp8_dispatch_args(rank, num_ranks, num_experts_per_rank, num_sm);
  check_tensor_common(x, "x", true, torch::kBFloat16, 2);
  int32_t num_token = x.size(0);
  int32_t hidden_size = x.size(1);
  FLASH_CHECK(hidden_size % kMXFP8BlockSize == 0)
      << "hidden_size must be divisible by " << kMXFP8BlockSize;
  int32_t topk = check_mxfp8_dispatch_layout_tensors(
      topk_send_mask, topk_indices, token_dst_scatter_indices, num_token);
  check_ptrs_tensor_i64(recv_x_ptrs, num_ranks, "recv_x_ptrs");
  check_ptrs_tensor_i64(recv_weights_ptrs, num_ranks, "recv_weights_ptrs");
  check_ptrs_tensor_i64(recv_topk_scatter_indices_ptrs, num_ranks,
                        "recv_topk_scatter_indices_ptrs");
  void *topk_weights = nullptr;
  auto weight_dtype = FlashCommDType::Float32;
  if (optional_topk_weights.has_value() &&
      optional_topk_weights.value().defined()) {
    check_tensor_common(optional_topk_weights.value(), "topk_weights", true,
                        torch::kFloat32, 2);
    check_tensor_shape(optional_topk_weights.value(), "topk_weights",
                       {num_token, topk});
    topk_weights = optional_topk_weights.value().data_ptr();
  }
  auto offset_dtype = get_flash_comm_dtype(topk_indices.scalar_type());
  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";
  FLASH_CHECK(topk * 2 <= WARP_SIZE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_mxfp8_quant_fused_intranode_cuda(
      x.data_ptr(), topk_send_mask.data_ptr(), topk_weights,
      topk_indices.data_ptr(), token_dst_scatter_indices.data_ptr(),
      recv_x_ptrs.data_ptr(),
      reinterpret_cast<void **>(recv_weights_ptrs.data_ptr()),
      reinterpret_cast<void **>(recv_topk_scatter_indices_ptrs.data_ptr()),
      num_token, hidden_size, num_experts_per_rank, rank, num_ranks, num_sm,
      weight_dtype, offset_dtype, topk, stream);
}

void dispatch_mxfp8_prequantized_intranode(
    torch::Tensor x, torch::Tensor topk_send_mask,
    c10::optional<torch::Tensor> optional_topk_weights,
    torch::Tensor topk_indices, torch::Tensor token_dst_scatter_indices,
    torch::Tensor recv_x_ptrs, torch::Tensor recv_weights_ptrs,
    torch::Tensor recv_topk_scatter_indices_ptrs, int32_t hidden_size,
    int32_t rank, int32_t num_ranks, int32_t num_experts_per_rank,
    int32_t num_sm) {
  check_mxfp8_dispatch_args(rank, num_ranks, num_experts_per_rank, num_sm);
  check_tensor_common(x, "x", true, torch::kBFloat16, 2);
  int32_t num_token = x.size(0);
  // The caller states the logical hidden size and the packed row length is
  // derived from it, so a mis-sized packed buffer is rejected here instead of
  // silently selecting a differently shaped kernel.
  int32_t packed_row_bytes = mxfp8_packed_row_bytes(hidden_size);
  check_tensor_shape(x, "x", {num_token, packed_row_bytes / 2});
  int32_t topk = check_mxfp8_dispatch_layout_tensors(
      topk_send_mask, topk_indices, token_dst_scatter_indices, num_token);
  check_ptrs_tensor_i64(recv_x_ptrs, num_ranks, "recv_x_ptrs");
  check_ptrs_tensor_i64(recv_weights_ptrs, num_ranks, "recv_weights_ptrs");
  check_ptrs_tensor_i64(recv_topk_scatter_indices_ptrs, num_ranks,
                        "recv_topk_scatter_indices_ptrs");
  void *topk_weights = nullptr;
  auto weight_dtype = FlashCommDType::Float32;
  if (optional_topk_weights.has_value() &&
      optional_topk_weights.value().defined()) {
    check_tensor_common(optional_topk_weights.value(), "topk_weights", true,
                        torch::kFloat32, 2);
    check_tensor_shape(optional_topk_weights.value(), "topk_weights",
                       {num_token, topk});
    topk_weights = optional_topk_weights.value().data_ptr();
  }
  auto offset_dtype = get_flash_comm_dtype(topk_indices.scalar_type());
  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_mxfp8_prequantized_intranode_cuda(
      x.data_ptr(), topk_send_mask.data_ptr(), topk_weights,
      topk_indices.data_ptr(), token_dst_scatter_indices.data_ptr(),
      recv_x_ptrs.data_ptr(),
      reinterpret_cast<void **>(recv_weights_ptrs.data_ptr()),
      reinterpret_cast<void **>(recv_topk_scatter_indices_ptrs.data_ptr()),
      num_token, hidden_size, num_experts_per_rank, rank, num_ranks, num_sm,
      weight_dtype, offset_dtype, topk, stream);
}

// dispatch_postprocess host function:
//  1.inplace update the recv_x with the local dispatch
//  2.reset the recv_topk_scatter_indices to -1 (in comm buffer)
//  3.copy the recv_weights to the dispatch_weights
//  4.copy the recv_topk_scatter_indices from comm buffer to torch tensor
void dispatch_postprocess(
    torch::Tensor recv_x, // [num_worst_recv_token, hidden_size], contiguous, in
                          // symm/shared memory or CUDA malloc
    torch::Tensor
        recv_topk_scatter_indices_comm_buffer, // [num_worst_recv_token,
                                               // topk], int32_t
    c10::optional<torch::Tensor>
        optional_recv_topk_weights, // [num_worst_recv_token, topk], float32
    torch::Tensor recv_token_count, // [num_ranks], int32
    c10::optional<torch::Tensor>
        optional_dispatch_weights, // output: [num_worst_recv_token], float32
    torch::Tensor recv_topk_scatter_indices, // output: [num_worst_recv_token,
                                             // topk], int32/offset_t
    int32_t hidden_size, int32_t topk, int32_t rank, int32_t num_ranks,
    int32_t num_sm) {
  // Assumes inputs are properly checked for shapes, types & device
  auto num_worst_recv_token = recv_x.size(0);
  auto dtype = get_flash_comm_dtype(recv_x.scalar_type());
  auto weight_dtype = FlashCommDType::Float32;
  auto offset_dtype =
      get_flash_comm_dtype(recv_topk_scatter_indices.scalar_type());
  bool has_weight = (optional_recv_topk_weights.has_value() &&
                     optional_recv_topk_weights.value().defined());
  bool has_dispatch_weights = (optional_dispatch_weights.has_value() &&
                               optional_dispatch_weights.value().defined());
  FLASH_CHECK(has_weight == has_dispatch_weights)
      << "recv_topk_weights and dispatch_weights must be both provided or both "
         "None";
  void *recv_topk_weights_ptr = nullptr;
  void *dispatch_weights_ptr = nullptr;
  if (has_weight) {
    weight_dtype =
        get_flash_comm_dtype(optional_recv_topk_weights.value().scalar_type());
    recv_topk_weights_ptr = optional_recv_topk_weights.value().data_ptr();
    dispatch_weights_ptr = optional_dispatch_weights.value().data_ptr();
  }

  FLASH_CHECK(num_ranks == recv_token_count.size(0))
      << "dim0 of recv_token_count must be equal to num_ranks, " << num_ranks
      << " != " << recv_token_count.size(0);
  FLASH_CHECK(rank >= 0 && rank < num_ranks)
      << "rank must be between 0 and num_ranks - 1, " << rank << " not in [0, "
      << num_ranks << ")";
  FLASH_CHECK(num_worst_recv_token == recv_topk_scatter_indices.size(0))
      << "dim0 of recv_x and recv_topk_scatter_indices must be equal, "
      << num_worst_recv_token << " != " << recv_topk_scatter_indices.size(0);
  if (has_weight) {
    FLASH_CHECK(num_worst_recv_token ==
                optional_recv_topk_weights.value().size(0))
        << "dim0 of recv_x and recv_topk_weights must be equal, "
        << num_worst_recv_token
        << " != " << optional_recv_topk_weights.value().size(0);
    FLASH_CHECK(num_worst_recv_token ==
                optional_dispatch_weights.value().size(0))
        << "dim0 of recv_x and dispatch_weights must be equal, "
        << num_worst_recv_token
        << " != " << optional_dispatch_weights.value().size(0);
  }
  FLASH_CHECK(topk == recv_topk_scatter_indices.size(1))
      << "dim1 of recv_topk_scatter_indices must be equal to topk, " << topk
      << " != " << recv_topk_scatter_indices.size(1);
  if (has_weight) {
    FLASH_CHECK(topk == optional_recv_topk_weights.value().size(1))
        << "dim1 of recv_topk_weights must be equal to topk, " << topk
        << " != " << optional_recv_topk_weights.value().size(1);
  }
  FLASH_CHECK(hidden_size == recv_x.size(1))
      << "dim1 of recv_x must be equal to hidden_size, " << hidden_size
      << " != " << recv_x.size(1);

  check_tensor_common(recv_token_count, "recv_token_count", true, torch::kInt32,
                      1);
  check_tensor_shape(recv_token_count, "recv_token_count", {num_ranks});
  // set max dynamic shared memory if necessary (not needed for this kernel)
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  FLASH_CHECK(dtype == FlashCommDType::BFloat16)
      << "Only BFloat16 token type is currently supported";
  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";
  if (has_weight) {
    FLASH_CHECK(weight_dtype == FlashCommDType::Float32)
        << "Only Float32 weight type is currently supported";
  }

  dispatch_postprocess_cuda(
      recv_x.data_ptr(), recv_topk_scatter_indices_comm_buffer.data_ptr(),
      recv_topk_weights_ptr, recv_token_count.data_ptr<int32_t>(),
      dispatch_weights_ptr, recv_topk_scatter_indices.data_ptr(),
      num_worst_recv_token, hidden_size, topk, rank, num_ranks, num_sm, dtype,
      weight_dtype, offset_dtype, stream);
}

void dispatch_mxfp8_postprocess_unpack(
    torch::Tensor recv_packed,
    torch::Tensor recv_topk_scatter_indices_comm_buffer,
    c10::optional<torch::Tensor> optional_recv_topk_weights,
    torch::Tensor recv_token_count, torch::Tensor dispatch_data,
    torch::Tensor dispatch_scales,
    c10::optional<torch::Tensor> optional_dispatch_weights,
    torch::Tensor recv_topk_scatter_indices, int32_t hidden_size, int32_t topk,
    int32_t rank, int32_t num_ranks, int32_t num_sm) {
  FLASH_CHECK(num_ranks > 0 && num_ranks <= kMXFP8MaxIntranodeRanks)
      << "MXFP8 intranode postprocess supports 1.." << kMXFP8MaxIntranodeRanks
      << " ranks";
  FLASH_CHECK(rank >= 0 && rank < num_ranks)
      << "rank must be between 0 and num_ranks - 1, " << rank << " not in [0, "
      << num_ranks << ")";
  FLASH_CHECK(hidden_size > 0 && hidden_size % kMXFP8BlockSize == 0)
      << "hidden_size must be positive and divisible by " << kMXFP8BlockSize;
  check_tensor_common(recv_packed, "recv_packed", true, torch::kBFloat16, 2);
  int32_t num_token = recv_packed.size(0);
  int32_t packed_row_bytes = mxfp8_packed_row_bytes(hidden_size);
  check_tensor_shape(recv_packed, "recv_packed",
                     {num_token, packed_row_bytes / 2});
  check_tensor_common(recv_topk_scatter_indices_comm_buffer,
                      "recv_topk_scatter_indices_comm_buffer", true,
                      torch::kInt32, 2);
  check_tensor_shape(recv_topk_scatter_indices_comm_buffer,
                     "recv_topk_scatter_indices_comm_buffer",
                     {num_token, topk});
  check_tensor_common(recv_token_count, "recv_token_count", true, torch::kInt32,
                      1);
  check_tensor_shape(recv_token_count, "recv_token_count", {num_ranks});
  FLASH_CHECK(dispatch_data.is_cuda() && dispatch_data.is_contiguous() &&
              dispatch_data.scalar_type() == torch::ScalarType::Float8_e4m3fn)
      << "dispatch_data must be a contiguous CUDA float8_e4m3fn tensor";
  check_tensor_shape(dispatch_data, "dispatch_data", {num_token, hidden_size});
  check_tensor_common(dispatch_scales, "dispatch_scales", true, torch::kUInt8,
                      2);
  check_tensor_shape(dispatch_scales, "dispatch_scales",
                     {num_token, hidden_size / kMXFP8BlockSize});
  check_tensor_common(recv_topk_scatter_indices, "recv_topk_scatter_indices",
                      true, torch::kInt32, 2);
  check_tensor_shape(recv_topk_scatter_indices, "recv_topk_scatter_indices",
                     {num_token, topk});

  bool has_weight = optional_recv_topk_weights.has_value() &&
                    optional_recv_topk_weights.value().defined();
  bool has_dispatch_weight = optional_dispatch_weights.has_value() &&
                             optional_dispatch_weights.value().defined();
  FLASH_CHECK(has_weight == has_dispatch_weight)
      << "recv weights and dispatch weights must both be provided or omitted";
  void *recv_weights = nullptr;
  void *dispatch_weights = nullptr;
  if (has_weight) {
    check_tensor_common(optional_recv_topk_weights.value(), "recv_topk_weights",
                        true, torch::kFloat32, 2);
    check_tensor_shape(optional_recv_topk_weights.value(), "recv_topk_weights",
                       {num_token, topk});
    check_tensor_common(optional_dispatch_weights.value(), "dispatch_weights",
                        true, torch::kFloat32, 1);
    check_tensor_shape(optional_dispatch_weights.value(), "dispatch_weights",
                       {num_token});
    recv_weights = optional_recv_topk_weights.value().data_ptr();
    dispatch_weights = optional_dispatch_weights.value().data_ptr();
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dispatch_mxfp8_postprocess_unpack_cuda(
      recv_packed.data_ptr(), recv_topk_scatter_indices_comm_buffer.data_ptr(),
      recv_weights, recv_token_count.data_ptr<int32_t>(),
      dispatch_data.data_ptr(), dispatch_scales.data_ptr(), dispatch_weights,
      recv_topk_scatter_indices.data_ptr(), num_token, hidden_size, topk, rank,
      num_ranks, num_sm, FlashCommDType::Float32, FlashCommDType::Int32,
      stream);
}

static void combine_intranode_impl(
    torch::Tensor x_ptrs,         // [num_ranks], shape of x is [num_recv_token,
                                  // hidden_size], need in symmetric memory
    torch::Tensor topk_send_mask, // [num_recv_token, topk]
    torch::Tensor topk_indices,   // [num_token, topk]
    torch::Tensor token_dst_scatter_indices, // [num_token, topk]
    torch::Tensor recv_x,                    // [num_token, hidden_size]
    int32_t rank, int32_t num_ranks, int32_t num_experts_per_rank,
    int32_t num_sm, c10::optional<torch::Tensor> optional_weight_ptrs,
    c10::optional<torch::Tensor> optional_recv_weight,
    c10::optional<torch::Tensor> optional_logical_token_range) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const auto dtype = get_flash_comm_dtype(recv_x.scalar_type());
  const auto offset_dtype = get_flash_comm_dtype(topk_indices.scalar_type());
  const int token_size = recv_x.element_size();

  FLASH_CHECK(x_ptrs.size(0) == num_ranks)
      << "dim0 of x_ptrs must be equal to num_ranks, " << x_ptrs.size(0)
      << " != " << num_ranks;
  FLASH_CHECK(recv_x.size(0) == token_dst_scatter_indices.size(0))
      << "dim0 of recv_x and token_dst_scatter_indices must be equal, "
      << recv_x.size(0) << " != " << token_dst_scatter_indices.size(0);
  FLASH_CHECK(topk_indices.size(1) == token_dst_scatter_indices.size(1))
      << "dim1 of topk_indices and token_dst_scatter_indices must be equal, "
      << topk_indices.size(1) << " != " << token_dst_scatter_indices.size(1);
  const int32_t num_token = topk_indices.size(0);
  const int32_t hidden_size = recv_x.size(1);
  const int32_t topk = topk_indices.size(1);
  FLASH_CHECK(topk_send_mask.sizes() == topk_indices.sizes());
  FLASH_CHECK(token_dst_scatter_indices.sizes() == topk_indices.sizes());

  FLASH_CHECK(hidden_size % (128 / token_size) == 0)
      << "hidden_size must be divisible by 128 / token_size, " << hidden_size
      << " % " << (128 / token_size) << " != " << 0;

  void *weight_ptrs = nullptr;
  void *recv_weight = nullptr;
  bool has_weight_ptrs = false;
  bool has_recv_weight = false;
  auto weight_dtype = FlashCommDType::Float32;
  if (optional_weight_ptrs.has_value() and
      optional_weight_ptrs.value().defined()) {
    weight_ptrs = optional_weight_ptrs.value().data_ptr();
    has_weight_ptrs = true;
    check_tensor_shape(optional_weight_ptrs.value(), "weight_ptrs",
                       {num_ranks});
  }
  if (optional_recv_weight.has_value() and
      optional_recv_weight.value().defined()) {
    recv_weight = optional_recv_weight.value().data_ptr();
    has_recv_weight = true;
    weight_dtype =
        get_flash_comm_dtype(optional_recv_weight.value().scalar_type());
    check_tensor_shape(optional_recv_weight.value(), "recv_weight",
                       {num_token, topk});
  }
  FLASH_CHECK(has_weight_ptrs == has_recv_weight);

  FLASH_CHECK(dtype == FlashCommDType::BFloat16)
      << "Only BFloat16 token type is currently supported";
  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";
  FLASH_CHECK(weight_dtype == FlashCommDType::Float32)
      << "Only Float32 weight type is currently supported";
  const int32_t *logical_token_range_ptr = nullptr;
  if (optional_logical_token_range.has_value() &&
      optional_logical_token_range.value().defined()) {
    auto logical_token_range = optional_logical_token_range.value();
    check_tensor_common(logical_token_range, "logical_token_range", true,
                        torch::kInt32, 1);
    check_tensor_shape(logical_token_range, "logical_token_range", {2});
    logical_token_range_ptr = logical_token_range.data_ptr<int32_t>();
  }

  combine_intranode_cuda(
      x_ptrs.data_ptr(), weight_ptrs, topk_send_mask.data_ptr(),
      topk_indices.data_ptr(), token_dst_scatter_indices.data_ptr(),
      recv_x.data_ptr(), recv_weight, has_recv_weight, num_token, hidden_size,
      topk, num_experts_per_rank, rank, num_ranks, num_sm, dtype, weight_dtype,
      offset_dtype, logical_token_range_ptr, stream);
}

void combine_intranode(torch::Tensor x_ptrs, torch::Tensor topk_send_mask,
                       torch::Tensor topk_indices,
                       torch::Tensor token_dst_scatter_indices,
                       torch::Tensor recv_x, int32_t rank, int32_t num_ranks,
                       int32_t num_experts_per_rank, int32_t num_sm,
                       c10::optional<torch::Tensor> optional_weight_ptrs,
                       c10::optional<torch::Tensor> optional_recv_weight) {
  combine_intranode_impl(x_ptrs, topk_send_mask, topk_indices,
                         token_dst_scatter_indices, recv_x, rank, num_ranks,
                         num_experts_per_rank, num_sm, optional_weight_ptrs,
                         optional_recv_weight, c10::nullopt);
}

void combine_intranode_range(
    torch::Tensor x_ptrs, torch::Tensor topk_send_mask,
    torch::Tensor topk_indices, torch::Tensor token_dst_scatter_indices,
    torch::Tensor logical_token_range, torch::Tensor recv_x, int32_t rank,
    int32_t num_ranks, int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<torch::Tensor> optional_weight_ptrs,
    c10::optional<torch::Tensor> optional_recv_weight) {
  combine_intranode_impl(x_ptrs, topk_send_mask, topk_indices,
                         token_dst_scatter_indices, recv_x, rank, num_ranks,
                         num_experts_per_rank, num_sm, optional_weight_ptrs,
                         optional_recv_weight, logical_token_range);
}

void combine_preprocess_inplace(
    torch::Tensor x,                // [num_recv_worst_token, hidden_size]
    torch::Tensor recv_token_count, // [num_ranks], int32
    torch::Tensor recv_topk_scatter_indices, // [num_recv_worst_token, topk]
    int32_t rank, int32_t num_ranks, int32_t num_sm,
    c10::optional<torch::Tensor>
        optional_weight, // optional weight [num_recv_worst_token,]
    c10::optional<torch::Tensor>
        optional_recv_topk_weight) { // optional recv_topk_weight
                                     // [num_recv_worst_token, topk]
  int64_t hidden_size = x.size(1);
  int64_t topk = recv_topk_scatter_indices.size(1);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto dtype = get_flash_comm_dtype(x.scalar_type());
  auto offset_dtype =
      get_flash_comm_dtype(recv_topk_scatter_indices.scalar_type());
  int token_size = x.element_size();

  int32_t num_recv_worst_token = x.size(0);
  FLASH_CHECK(num_recv_worst_token == recv_topk_scatter_indices.size(0))
      << "dim0 of x and recv_topk_scatter_indices must be equal, "
      << num_recv_worst_token << " != " << recv_topk_scatter_indices.size(0);
  FLASH_CHECK(hidden_size == x.size(1))
      << "dim1 of x must be equal to hidden_size, " << hidden_size
      << " != " << x.size(1);

  check_tensor_common(recv_token_count, "recv_token_count", true, torch::kInt32,
                      1);
  check_tensor_shape(recv_token_count, "recv_token_count", {num_ranks});
  constexpr int32_t kElemsPerThread = 32;
  int32_t kVec = 16 / token_size;
  FLASH_CHECK(hidden_size % kVec == 0)
      << "hidden_size must be divisible by kVec, " << hidden_size << " % "
      << kVec << " != 0";
  const int32_t hidden_size_int4 = hidden_size / kVec;
  const int32_t preprocess_num_warps = hidden_size >= 12288 ? 32 : 16;
  const int32_t preprocess_capacity_int4 =
      kElemsPerThread / kVec * preprocess_num_warps * WARP_SIZE;
  FLASH_CHECK(preprocess_capacity_int4 >= hidden_size_int4)
      << "combine_preprocess int4 capacity must cover hidden_size_int4, "
      << preprocess_capacity_int4 << " < " << hidden_size_int4;

  void *weight_ptr = nullptr;
  void *recv_topk_weight_ptr = nullptr;
  auto weight_dtype = FlashCommDType::Float32;
  if (optional_weight.has_value() and optional_weight.value().defined()) {
    weight_ptr = optional_weight.value().data_ptr();
    weight_dtype = get_flash_comm_dtype(optional_weight.value().scalar_type());
    check_tensor_shape(optional_weight.value(), "weight",
                       {num_recv_worst_token});
  }
  if (optional_recv_topk_weight.has_value() and
      optional_recv_topk_weight.value().defined()) {
    recv_topk_weight_ptr = optional_recv_topk_weight.value().data_ptr();
    weight_dtype =
        get_flash_comm_dtype(optional_recv_topk_weight.value().scalar_type());
    check_tensor_shape(optional_recv_topk_weight.value(), "recv_topk_weight",
                       {num_recv_worst_token, topk});
  }
  FLASH_CHECK((weight_ptr == nullptr) == (recv_topk_weight_ptr == nullptr))
      << "weight and recv_topk_weight must be both nullptr or both not nullptr";
  FLASH_CHECK(weight_dtype == FlashCommDType::Float32)
      << "Only Float32 weight type is currently supported";
  FLASH_CHECK(dtype == FlashCommDType::BFloat16)
      << "Only BFloat16 token type is currently supported";
  FLASH_CHECK(offset_dtype == FlashCommDType::Int32)
      << "Only Int32 offset type is currently supported";

  combine_preprocess_inplace_cuda(
      x.data_ptr(), weight_ptr, recv_token_count.data_ptr<int32_t>(),
      recv_topk_scatter_indices.data_ptr(), recv_topk_weight_ptr,
      num_recv_worst_token, hidden_size, topk, rank, num_ranks, num_sm, dtype,
      weight_dtype, offset_dtype, stream);
}

void barrier_all_on_stream(torch::Tensor barrier_ptrs, int32_t rank,
                           int32_t num_ranks) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  barrier_all_on_stream_cuda(reinterpret_cast<void **>(barrier_ptrs.data_ptr()),
                             rank, num_ranks, FlashCommDType::Int32, stream);
}

void barrier_all_on_stream_range(torch::Tensor barrier_ptrs, int32_t rank,
                                 int32_t num_ranks,
                                 torch::Tensor logical_token_range) {
  check_ptrs_tensor_i64(barrier_ptrs, num_ranks, "barrier_ptrs");
  check_tensor_common(logical_token_range, "logical_token_range", true,
                      torch::kInt32, 1);
  check_tensor_shape(logical_token_range, "logical_token_range", {2});
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  barrier_all_on_stream_range_cuda(
      reinterpret_cast<void **>(barrier_ptrs.data_ptr()), rank, num_ranks,
      logical_token_range.data_ptr<int32_t>(), stream);
}

} // namespace intranode
} // namespace ep
} // namespace flash_comm

void bind_intranode_ops(py::module &m) {
  m.def("compute_stable_local_token_within_expert_offset",
        &flash_comm::ep::intranode::
            compute_stable_local_token_within_expert_offset,
        "Compute stable local token_within_expert_offset + per-tile prefixes.");
  m.def(
      "compute_stable_local_token_within_expert_offset_and_expert_counts",
      [](torch::Tensor topk_indices, int32_t num_experts, int32_t num_sm,
         py::object expert_counts_obj) {
        torch::Tensor expert_counts;
        if (!expert_counts_obj.is_none()) {
          expert_counts = expert_counts_obj.cast<torch::Tensor>();
        }
        return flash_comm::ep::intranode::
            compute_stable_local_token_within_expert_offset_and_expert_counts(
                topk_indices, num_experts, num_sm, expert_counts);
      },
      py::arg("topk_indices"), py::arg("num_experts"), py::arg("num_sm"),
      py::arg("expert_counts") = py::none(),
      "Compute stable local token_within_expert_offset + per-tile prefixes + "
      "per-expert total counts (bincount, "
      "including drop token).\n"
      "If expert_counts is provided, it must be a contiguous CUDA int32 tensor "
      "of shape [num_experts + 1].");

  m.def("compute_dispatch_layout",
        &flash_comm::ep::intranode::compute_dispatch_layout,
        py::arg("topk_indices"), py::arg("token_within_expert_offset"),
        py::arg("local_splits"), py::arg("full_splits_ptrs"),
        py::arg("barrier_ptrs"), py::arg("num_experts"), py::arg("rank"),
        py::arg("num_ranks"), py::arg("num_sm"),
        py::arg("recv_token_count_cpu") = py::none(),
        py::arg("recv_token_count") = py::none(),
        py::arg("token_src_rank_topk_and_indices_ptrs") = py::none(),
        py::arg("expert_alignment") = 1,
        "Compute dispatch layout:\n"
        "- all-gather local_splits into full_splits (via symmetric buffers)\n"
        "- compute recv_base_offset[dst_rank, local_expert, src_rank]\n"
        "- compute token_dst_scatter_indices and token_topk_send_mask\n"
        "- compute recv_token_count_cpu[dst_rank] (drop excluded) into a "
        "pinned CPU int32 tensor\n"
        "- compute recv_token_count[dst_rank] (drop excluded) into a CUDA "
        "int32 tensor\n"
        "Inputs full_splits_ptrs/barrier_ptrs are int64 CUDA tensors of device "
        "pointers (len = num_ranks).\n"
        "If recv_token_count_cpu is provided, it must be a contiguous pinned "
        "CPU int32 tensor of shape [num_ranks].\n"
        "If recv_token_count is provided, it must be a contiguous CUDA int32 "
        "tensor of shape [num_ranks].\n"
        "Returns: (recv_base_offset, token_dst_scatter_indices, "
        "token_topk_send_mask, recv_token_count_cpu, "
        "recv_token_count, recv_aligned_token_count_cpu, "
        "recv_aligned_token_count, recv_expert_counts).\n"
        "recv_expert_counts is shape [experts_per_rank] with per-expert "
        "actual token counts for the local rank.\n"
        "When expert_alignment > 1, recv_token_count holds the original "
        "(unpadded) count, while recv_aligned_token_count holds the padded "
        "count for buffer allocation.");
  m.def("dispatch_intranode", &flash_comm::ep::intranode::dispatch_intranode,
        py::arg("x"), py::arg("topk_send_mask"),
        py::arg("optional_topk_weights"), py::arg("topk_indices"),
        py::arg("token_dst_scatter_indices"), py::arg("recv_x_ptrs"),
        py::arg("recv_weights_ptrs"), py::arg("recv_topk_scatter_indices_ptrs"),
        py::arg("rank"), py::arg("num_ranks"), py::arg("num_experts_per_rank"),
        py::arg("num_sm"), "intranode dispatch for ep");
  m.def("dispatch_intranode_range",
        &flash_comm::ep::intranode::dispatch_intranode_range, py::arg("x"),
        py::arg("topk_send_mask"), py::arg("optional_topk_weights"),
        py::arg("topk_indices"), py::arg("token_dst_scatter_indices"),
        py::arg("logical_token_range"), py::arg("recv_x_ptrs"),
        py::arg("recv_weights_ptrs"), py::arg("recv_topk_scatter_indices_ptrs"),
        py::arg("rank"), py::arg("num_ranks"), py::arg("num_experts_per_rank"),
        py::arg("num_sm"),
        "Intranode dispatch for one device-selected token range.");
  m.def("dispatch_mxfp8_quant_fused_intranode",
        &flash_comm::ep::intranode::dispatch_mxfp8_quant_fused_intranode,
        py::arg("x"), py::arg("topk_send_mask"),
        py::arg("optional_topk_weights"), py::arg("topk_indices"),
        py::arg("token_dst_scatter_indices"), py::arg("recv_x_ptrs"),
        py::arg("recv_weights_ptrs"), py::arg("recv_topk_scatter_indices_ptrs"),
        py::arg("rank"), py::arg("num_ranks"), py::arg("num_experts_per_rank"),
        py::arg("num_sm"), "fused BF16-to-MXFP8 intranode dispatch");
  m.def("dispatch_mxfp8_prequantized_intranode",
        &flash_comm::ep::intranode::dispatch_mxfp8_prequantized_intranode,
        py::arg("x"), py::arg("topk_send_mask"),
        py::arg("optional_topk_weights"), py::arg("topk_indices"),
        py::arg("token_dst_scatter_indices"), py::arg("recv_x_ptrs"),
        py::arg("recv_weights_ptrs"), py::arg("recv_topk_scatter_indices_ptrs"),
        py::arg("hidden_size"), py::arg("rank"), py::arg("num_ranks"),
        py::arg("num_experts_per_rank"), py::arg("num_sm"),
        "dispatch a pre-quantized packed MXFP8 row");
  m.def("dispatch_postprocess",
        &flash_comm::ep::intranode::dispatch_postprocess, py::arg("recv_x"),
        py::arg("recv_topk_scatter_indices_comm_buffer"),
        py::arg("optional_recv_topk_weights"), py::arg("recv_token_count"),
        py::arg("optional_dispatch_weights"),
        py::arg("recv_topk_scatter_indices"), py::arg("hidden_size"),
        py::arg("topk"), py::arg("rank"), py::arg("num_ranks"),
        py::arg("num_sm"), "intranode dispatch postprocess for ep");
  m.def("dispatch_mxfp8_postprocess_unpack",
        &flash_comm::ep::intranode::dispatch_mxfp8_postprocess_unpack,
        py::arg("recv_packed"),
        py::arg("recv_topk_scatter_indices_comm_buffer"),
        py::arg("optional_recv_topk_weights"), py::arg("recv_token_count"),
        py::arg("dispatch_data"), py::arg("dispatch_scales"),
        py::arg("optional_dispatch_weights"),
        py::arg("recv_topk_scatter_indices"), py::arg("hidden_size"),
        py::arg("topk"), py::arg("rank"), py::arg("num_ranks"),
        py::arg("num_sm"), "Fused MXFP8 scatter and data/scale separation");
  m.def("combine_preprocess_inplace",
        &flash_comm::ep::intranode::combine_preprocess_inplace, py::arg("x"),
        py::arg("recv_token_count"), py::arg("recv_topk_scatter_indices"),
        py::arg("rank"), py::arg("num_ranks"), py::arg("num_sm"),
        py::arg("weight") = py::none(),
        py::arg("recv_topk_weight") = py::none(),
        "intranode combine preprocess inplace for ep\n");
  m.def("combine_intranode", &flash_comm::ep::intranode::combine_intranode,
        py::arg("x_ptrs"), py::arg("topk_send_mask"), py::arg("topk_indices"),
        py::arg("token_dst_scatter_indices"), py::arg("recv_x"),
        py::arg("rank"), py::arg("num_ranks"), py::arg("num_experts_per_rank"),
        py::arg("num_sm"), py::arg("weight_ptrs") = py::none(),
        py::arg("recv_weight") = py::none(),
        "intranode combine for ep\n"
        "If weight_ptrs is provided, it must be a contiguous CUDA int64 tensor "
        "of shape [num_ranks].\n"
        "If recv_weight is provided, it must be a contiguous CUDA float32 "
        "tensor of shape [num_token, topk]");
  m.def("combine_intranode_range",
        &flash_comm::ep::intranode::combine_intranode_range, py::arg("x_ptrs"),
        py::arg("topk_send_mask"), py::arg("topk_indices"),
        py::arg("token_dst_scatter_indices"), py::arg("logical_token_range"),
        py::arg("recv_x"), py::arg("rank"), py::arg("num_ranks"),
        py::arg("num_experts_per_rank"), py::arg("num_sm"),
        py::arg("weight_ptrs") = py::none(),
        py::arg("recv_weight") = py::none(),
        "Intranode combine for one device-selected token range.");
  m.def("barrier_all_on_stream",
        &flash_comm::ep::intranode::barrier_all_on_stream,
        "intranode barrier all on stream");
  m.def("barrier_all_on_stream_range",
        &flash_comm::ep::intranode::barrier_all_on_stream_range,
        py::arg("barrier_ptrs"), py::arg("rank"), py::arg("num_ranks"),
        py::arg("logical_token_range"),
        "Intranode stream barrier gated by a common device token range.");
}
