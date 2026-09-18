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
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <torch/extension.h>

#include "flash_comm/buffer/nccl_gin.h"
#include "flash_comm/ep/internode.h"
#include "flash_comm/ep/intranode.h"
#include "flash_comm/nccl_utils.h"
#include "flash_comm/torch_utils.h"

namespace flash_comm {
namespace ep {
namespace internode {

namespace {

// kMaxDispatchPipelineChunks / kMaxCombinePipelineChunks come from
// flash_comm/ep/internode.h (single source of truth for the signal space).

static inline void check_uva_enabled_for_current_device() {
  static std::once_flag once;
  std::call_once(once, []() {
    int dev = -1;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    FLASH_CHECK(prop.unifiedAddressing == 1)
        << "UVA required for pinned recv_token_count_cpu in layout kernel";
  });
}

static inline void check_topk_indices(const torch::Tensor &topk_indices) {
  check_tensor_common(topk_indices, "topk_indices", true, torch::kInt32, 2);
}

static inline void *window_user_ptr(ncclComm_t comm, uintptr_t win_handle) {
  void *ptr = nullptr;
  NCCL_CHECK(ncclWinGetUserPtr(comm, reinterpret_cast<ncclWindow_t>(win_handle),
                               &ptr));
  FLASH_CHECK(ptr != nullptr);
  return ptr;
}

static inline void *checked_window_user_ptr(ncclComm_t comm,
                                            uintptr_t win_handle,
                                            const torch::Tensor &owner,
                                            const char *owner_name) {
  void *ptr = window_user_ptr(comm, win_handle);
  FLASH_CHECK(ptr == owner.data_ptr())
      << owner_name << " data_ptr does not match ncclWinGetUserPtr(win)";
  return ptr;
}

static inline int32_t
optional_chunk_env(const char *name, int32_t default_value, int32_t max_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  long value = std::strtol(raw, &end, 10);
  FLASH_CHECK(end != raw && *end == '\0') << name << " must be an integer";
  FLASH_CHECK(value >= 1 && value <= max_value)
      << name << " must be in [1, " << max_value << "]";
  FLASH_CHECK(value <= std::numeric_limits<int32_t>::max());
  return static_cast<int32_t>(value);
}

} // namespace

static void dispatch_internode_impl(
    torch::Tensor recv_x_ptrs, torch::Tensor recv_weights_ptrs,
    torch::Tensor recv_topk_scatter_indices_ptrs, int32_t max_recv_tokens,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    torch::Tensor num_tokens_per_rank, torch::Tensor node_topk_indices,
    torch::Tensor node_topk_send_mask,
    torch::Tensor node_token_dst_scatter_indices, int32_t max_slot_num_token,
    int32_t local_num_token, int32_t hidden_size, bool has_weight,
    int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<int64_t> opt_num_qps,
    c10::optional<torch::Tensor> optional_logical_token_range) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  auto &ep_state = buffer::nccl_gin_require_state();
  const int32_t rank = ep_state.rank;
  const int32_t num_ranks = ep_state.nranks;
  const int32_t local_world_size = ep_state.local_world_size;
  const int32_t max_qps = ep_state.ep_num_qps;
  const int64_t requested_qps = opt_num_qps.value_or(max_qps);
  FLASH_CHECK(requested_qps >= 1 && requested_qps <= max_qps)
      << "num_qps must be in [1, " << max_qps
      << "] (the QP count configured at NCCL GIN init), got " << requested_qps;
  const int32_t num_qps = static_cast<int32_t>(requested_qps);
  FLASH_CHECK(num_ranks % local_world_size == 0);
  FLASH_CHECK(!ep_state.ep_dispatch_needs_barrier)
      << "internode dispatch requires reset_signals_barrier_all_on_stream "
         "covering the dispatch signal range after the previous dispatch "
         "(it resets the GIN signals and protects RDMA slot reuse)";
  const int32_t nnodes = num_ranks / local_world_size;
  FLASH_CHECK(rdma_rail_send_buf.is_cuda() &&
              rdma_rail_send_buf.is_contiguous());
  FLASH_CHECK(num_tokens_per_rank.is_cuda() &&
              num_tokens_per_rank.scalar_type() == torch::kInt32);
  FLASH_CHECK(num_tokens_per_rank.numel() == num_ranks);
  FLASH_CHECK(node_topk_indices.is_cuda() &&
              node_topk_indices.is_contiguous() &&
              node_topk_indices.scalar_type() == torch::kInt32);
  FLASH_CHECK(node_topk_send_mask.is_cuda() &&
              node_topk_send_mask.is_contiguous() &&
              node_topk_send_mask.scalar_type() == torch::kInt32);
  FLASH_CHECK(node_token_dst_scatter_indices.is_cuda() &&
              node_token_dst_scatter_indices.is_contiguous() &&
              node_token_dst_scatter_indices.scalar_type() == torch::kInt32);
  FLASH_CHECK(node_topk_indices.dim() == 3);
  const int32_t topk = node_topk_indices.size(2);
  FLASH_CHECK(node_topk_indices.numel() >=
              static_cast<int64_t>(nnodes) * max_slot_num_token * topk);
  FLASH_CHECK(node_topk_send_mask.numel() >=
              static_cast<int64_t>(nnodes) * max_slot_num_token * topk);
  FLASH_CHECK(node_token_dst_scatter_indices.numel() >=
              static_cast<int64_t>(nnodes) * max_slot_num_token * topk);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const void *dev_comm = static_cast<const void *>(&ep_state.dev_comm);

  FLASH_CHECK(hidden_size > 0);
  FLASH_CHECK(topk > 0);
  auto dtype = FlashCommDType::BFloat16;
  auto weight_dtype = FlashCommDType::Float32;
  auto offset_dtype = FlashCommDType::Int32;

  check_ptrs_tensor_i64(recv_x_ptrs, local_world_size, "recv_x_ptrs");
  check_ptrs_tensor_i64(recv_weights_ptrs, local_world_size,
                        "recv_weights_ptrs");
  check_ptrs_tensor_i64(recv_topk_scatter_indices_ptrs, local_world_size,
                        "recv_topk_scatter_indices_ptrs");
  FLASH_CHECK(local_num_token >= 0 && local_num_token <= max_slot_num_token);
  const size_t slot_stride =
      rdma_rail_send_slot_stride_bytes(max_slot_num_token, hidden_size, topk);
  const size_t rdma_rail_send_bytes = slot_stride * static_cast<size_t>(nnodes);
  FLASH_CHECK(static_cast<size_t>(rdma_rail_send_buf.numel()) *
                  rdma_rail_send_buf.element_size() >=
              rdma_rail_send_bytes);
  // Dispatch tokens are currently materialized as BF16 in the RDMA rail slot.
  static const int32_t configured_dispatch_pipeline_chunks = optional_chunk_env(
      "FLASH_COMM_EP_DISPATCH_PIPELINE_CHUNKS", 1, kMaxDispatchPipelineChunks);
  const int32_t dispatch_pipeline_chunks = configured_dispatch_pipeline_chunks;
  FLASH_CHECK(dispatch_pipeline_chunks > 0 &&
              dispatch_pipeline_chunks <= kMaxDispatchPipelineChunks)
      << "dispatch_pipeline_chunks must be in [1, "
      << kMaxDispatchPipelineChunks << "]";
  checked_window_user_ptr(ep_state.comm, rdma_rail_send_win_handle,
                          rdma_rail_send_buf, "rdma_rail_send_buf");
  const int32_t *logical_token_range_ptr = nullptr;
  if (optional_logical_token_range.has_value() &&
      optional_logical_token_range.value().defined()) {
    auto logical_token_range = optional_logical_token_range.value();
    check_tensor_common(logical_token_range, "logical_token_range", true,
                        torch::kInt32, 1);
    check_tensor_shape(logical_token_range, "logical_token_range", {2});
    logical_token_range_ptr = logical_token_range.data_ptr<int32_t>();
  }
  dispatch_internode_cuda(
      rdma_rail_send_win_handle, num_tokens_per_rank.data_ptr<int32_t>(),
      node_topk_indices.data_ptr<int32_t>(),
      node_topk_send_mask.data_ptr<int32_t>(),
      node_token_dst_scatter_indices.data_ptr<int32_t>(), has_weight,
      recv_x_ptrs.data_ptr(),
      reinterpret_cast<void **>(recv_weights_ptrs.data_ptr()),
      reinterpret_cast<void **>(recv_topk_scatter_indices_ptrs.data_ptr()),
      max_slot_num_token, hidden_size, num_experts_per_rank, rank, num_ranks,
      local_world_size, max_recv_tokens, num_sm, dev_comm, num_qps, dtype,
      weight_dtype, offset_dtype, topk, dispatch_pipeline_chunks,
      logical_token_range_ptr, stream);
  ep_state.ep_dispatch_needs_barrier = true;
}

void dispatch_internode(
    torch::Tensor recv_x_ptrs, torch::Tensor recv_weights_ptrs,
    torch::Tensor recv_topk_scatter_indices_ptrs, int32_t max_recv_tokens,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    torch::Tensor num_tokens_per_rank, torch::Tensor node_topk_indices,
    torch::Tensor node_topk_send_mask,
    torch::Tensor node_token_dst_scatter_indices, int32_t max_slot_num_token,
    int32_t local_num_token, int32_t hidden_size, bool has_weight,
    int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<int64_t> opt_num_qps) {
  dispatch_internode_impl(
      recv_x_ptrs, recv_weights_ptrs, recv_topk_scatter_indices_ptrs,
      max_recv_tokens, rdma_rail_send_buf, rdma_rail_send_win_handle,
      num_tokens_per_rank, node_topk_indices, node_topk_send_mask,
      node_token_dst_scatter_indices, max_slot_num_token, local_num_token,
      hidden_size, has_weight, num_experts_per_rank, num_sm, opt_num_qps,
      c10::nullopt);
}

void dispatch_internode_range(
    torch::Tensor recv_x_ptrs, torch::Tensor recv_weights_ptrs,
    torch::Tensor recv_topk_scatter_indices_ptrs, int32_t max_recv_tokens,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    torch::Tensor num_tokens_per_rank, torch::Tensor node_topk_indices,
    torch::Tensor node_topk_send_mask,
    torch::Tensor node_token_dst_scatter_indices, int32_t max_slot_num_token,
    int32_t local_num_token, int32_t hidden_size, bool has_weight,
    int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<int64_t> opt_num_qps, torch::Tensor logical_token_range) {
  dispatch_internode_impl(
      recv_x_ptrs, recv_weights_ptrs, recv_topk_scatter_indices_ptrs,
      max_recv_tokens, rdma_rail_send_buf, rdma_rail_send_win_handle,
      num_tokens_per_rank, node_topk_indices, node_topk_send_mask,
      node_token_dst_scatter_indices, max_slot_num_token, local_num_token,
      hidden_size, has_weight, num_experts_per_rank, num_sm, opt_num_qps,
      logical_token_range);
}

void barrier_all_on_stream() {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto &ep_state = buffer::nccl_gin_require_state();
  const void *dev_comm = static_cast<const void *>(&ep_state.dev_comm);
  internode_barrier_on_stream_cuda(dev_comm, ep_state.ep_num_qps, stream);
}

void barrier_all_on_stream_if_active(torch::Tensor logical_token_range) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  check_tensor_common(logical_token_range, "logical_token_range", true,
                      torch::kInt32, 1);
  check_tensor_shape(logical_token_range, "logical_token_range", {2});
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto &ep_state = buffer::nccl_gin_require_state();
  const void *dev_comm = static_cast<const void *>(&ep_state.dev_comm);
  internode_barrier_on_stream_if_active_cuda(
      dev_comm, ep_state.ep_num_qps, logical_token_range.data_ptr<int32_t>(),
      stream);
}

// Fused reset-signals-then-barrier over an explicit EP signal id range. Must
// follow each internode dispatch (with the dispatch signal range) and each
// combine (with the combine signal range) before the next call of the same
// protocol; the plain barrier_all_on_stream does not touch signals.
void reset_signals_barrier_all_on_stream(int64_t signal_begin,
                                         int64_t signal_end) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto &ep_state = buffer::nccl_gin_require_state();
  const int32_t max_qps = ep_state.ep_num_qps;
  const int32_t nnodes = ep_state.nnodes;
  const int32_t total = ep_required_gin_signal_count(nnodes);
  FLASH_CHECK(signal_begin >= 0 && signal_begin <= signal_end &&
              signal_end <= total)
      << "signal reset range [" << signal_begin << ", " << signal_end
      << ") must lie within [0, " << total << ")";
  const void *dev_comm = static_cast<const void *>(&ep_state.dev_comm);
  internode_reset_signals_barrier_on_stream_cuda(
      dev_comm, max_qps, static_cast<int32_t>(signal_begin),
      static_cast<int32_t>(signal_end), stream);
  // Only a reset covering a protocol's full signal range re-arms that
  // protocol for its next call.
  if (signal_begin == 0 && signal_end >= ep_dispatch_signal_count(nnodes)) {
    ep_state.ep_dispatch_needs_barrier = false;
  }
  if (signal_begin <= ep_combine_signal_base(nnodes) && signal_end == total) {
    ep_state.ep_combine_needs_barrier = false;
  }
}

void reset_signals_barrier_all_on_stream_if_active(
    int64_t signal_begin, int64_t signal_end,
    torch::Tensor logical_token_range) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  check_tensor_common(logical_token_range, "logical_token_range", true,
                      torch::kInt32, 1);
  check_tensor_shape(logical_token_range, "logical_token_range", {2});
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto &ep_state = buffer::nccl_gin_require_state();
  const int32_t nnodes = ep_state.nnodes;
  const int32_t total = ep_required_gin_signal_count(nnodes);
  FLASH_CHECK(signal_begin >= 0 && signal_begin <= signal_end &&
              signal_end <= total);
  const void *dev_comm = static_cast<const void *>(&ep_state.dev_comm);
  internode_reset_signals_barrier_on_stream_if_active_cuda(
      dev_comm, ep_state.ep_num_qps, static_cast<int32_t>(signal_begin),
      static_cast<int32_t>(signal_end), logical_token_range.data_ptr<int32_t>(),
      stream);
  // Host enqueue state follows the fixed call sequence, while the device
  // decides whether this plan row owns any protocol state to reset.
  if (signal_begin == 0 && signal_end >= ep_dispatch_signal_count(nnodes)) {
    ep_state.ep_dispatch_needs_barrier = false;
  }
  if (signal_begin <= ep_combine_signal_base(nnodes) && signal_end == total) {
    ep_state.ep_combine_needs_barrier = false;
  }
}

static void combine_internode_impl(
    torch::Tensor combine_x_ptrs,
    c10::optional<torch::Tensor> optional_combine_weight_ptrs,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    torch::Tensor num_tokens_per_rank, torch::Tensor output,
    torch::Tensor local_topk_indices, torch::Tensor local_topk_send_mask,
    torch::Tensor local_token_dst_scatter_indices,
    c10::optional<torch::Tensor> optional_output_weight,
    int32_t max_slot_num_token, int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<int64_t> opt_num_qps,
    c10::optional<torch::Tensor> optional_logical_token_range) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  auto &ep_state = buffer::nccl_gin_require_state();
  const int32_t rank = ep_state.rank;
  const int32_t num_ranks = ep_state.nranks;
  const int32_t local_world_size = ep_state.local_world_size;
  const int32_t max_qps = ep_state.ep_num_qps;
  const int64_t requested_qps = opt_num_qps.value_or(max_qps);
  FLASH_CHECK(requested_qps >= 1 && requested_qps <= max_qps)
      << "num_qps must be in [1, " << max_qps
      << "] (the QP count configured at NCCL GIN init), got " << requested_qps;
  const int32_t num_qps = static_cast<int32_t>(requested_qps);
  FLASH_CHECK(num_ranks % local_world_size == 0);
  FLASH_CHECK(!ep_state.ep_combine_needs_barrier)
      << "internode combine requires reset_signals_barrier_all_on_stream "
         "covering the combine signal range after the previous combine "
         "(it resets the GIN signals and protects RDMA slot reuse)";
  check_ptrs_tensor_i64(combine_x_ptrs, local_world_size, "combine_x_ptrs");
  FLASH_CHECK(rdma_rail_send_buf.is_cuda() &&
              rdma_rail_send_buf.is_contiguous());
  FLASH_CHECK(num_tokens_per_rank.is_cuda() &&
              num_tokens_per_rank.scalar_type() == torch::kInt32);
  FLASH_CHECK(num_tokens_per_rank.numel() == num_ranks);
  FLASH_CHECK(output.is_cuda() && output.is_contiguous());
  FLASH_CHECK(local_topk_indices.is_cuda() &&
              local_topk_indices.is_contiguous() &&
              local_topk_indices.scalar_type() == torch::kInt32);
  FLASH_CHECK(local_topk_send_mask.is_cuda() &&
              local_topk_send_mask.is_contiguous() &&
              local_topk_send_mask.scalar_type() == torch::kInt32);
  FLASH_CHECK(local_token_dst_scatter_indices.is_cuda() &&
              local_token_dst_scatter_indices.is_contiguous() &&
              local_token_dst_scatter_indices.scalar_type() == torch::kInt32);
  const int32_t num_token = output.size(0);
  const int32_t hidden_size = output.size(1);
  void *combine_weight_ptrs = nullptr;
  void *output_weight = nullptr;
  bool has_combine_weight = false;
  bool has_output_weight = false;
  bool empty_output_weight = false;
  auto weight_dtype = FlashCommDType::Float32;
  if (optional_combine_weight_ptrs.has_value() &&
      optional_combine_weight_ptrs.value().defined()) {
    check_ptrs_tensor_i64(optional_combine_weight_ptrs.value(),
                          local_world_size, "combine_weight_ptrs");
    combine_weight_ptrs = optional_combine_weight_ptrs.value().data_ptr();
    has_combine_weight = true;
  }
  if (optional_output_weight.has_value() &&
      optional_output_weight.value().defined()) {
    auto out_weight = optional_output_weight.value();
    FLASH_CHECK(out_weight.is_cuda() && out_weight.is_contiguous());
    empty_output_weight = out_weight.numel() == 0;
    FLASH_CHECK(!empty_output_weight || num_token == 0)
        << "empty output_weight requires an empty output";
    output_weight = empty_output_weight ? nullptr : out_weight.data_ptr();
    weight_dtype = get_flash_comm_dtype(out_weight.scalar_type());
    has_output_weight = !empty_output_weight;
  }
  if (empty_output_weight) {
    // There is no output row that can consume the weight path.  Disable both
    // sides together so kHasWeight=true is never launched with a null output.
    combine_weight_ptrs = nullptr;
    has_combine_weight = false;
  }
  FLASH_CHECK(has_combine_weight == has_output_weight)
      << "combine_weight_ptrs and output_weight must be both set or both null";
  const bool has_weight = has_combine_weight;

  FLASH_CHECK(local_topk_indices.dim() == 3);
  const int32_t topk = local_topk_indices.size(2);
  FLASH_CHECK(topk > 0);
  FLASH_CHECK(num_token <= max_slot_num_token);
  const int32_t nnodes = num_ranks / local_world_size;
  FLASH_CHECK(local_topk_indices.numel() >=
              static_cast<int64_t>(nnodes) * max_slot_num_token * topk);
  FLASH_CHECK(local_topk_send_mask.numel() >=
              static_cast<int64_t>(nnodes) * max_slot_num_token * topk);
  FLASH_CHECK(local_token_dst_scatter_indices.numel() >=
              static_cast<int64_t>(nnodes) * max_slot_num_token * topk);
  const size_t slot_stride =
      rdma_rail_send_slot_stride_bytes(max_slot_num_token, hidden_size, topk);
  const size_t min_rdmabuf_bytes =
      slot_stride * static_cast<size_t>(nnodes) * 3;
  FLASH_CHECK(static_cast<size_t>(rdma_rail_send_buf.numel()) *
                  rdma_rail_send_buf.element_size() >=
              min_rdmabuf_bytes);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const void *dev_comm = static_cast<const void *>(&ep_state.dev_comm);
  const auto dtype = get_flash_comm_dtype(output.scalar_type());
  const auto offset_dtype = FlashCommDType::Int32;
  static const int32_t configured_combine_pipeline_chunks = optional_chunk_env(
      "FLASH_COMM_EP_COMBINE_PIPELINE_CHUNKS", 1, kMaxCombinePipelineChunks);
  const int32_t combine_pipeline_chunks = configured_combine_pipeline_chunks;
  FLASH_CHECK(combine_pipeline_chunks > 0 &&
              combine_pipeline_chunks <= kMaxCombinePipelineChunks)
      << "combine_pipeline_chunks must be in [1, " << kMaxCombinePipelineChunks
      << "]";
  checked_window_user_ptr(ep_state.comm, rdma_rail_send_win_handle,
                          rdma_rail_send_buf, "rdma_rail_send_buf");
  const int32_t *logical_token_range_ptr = nullptr;
  if (optional_logical_token_range.has_value() &&
      optional_logical_token_range.value().defined()) {
    auto logical_token_range = optional_logical_token_range.value();
    check_tensor_common(logical_token_range, "logical_token_range", true,
                        torch::kInt32, 1);
    check_tensor_shape(logical_token_range, "logical_token_range", {2});
    logical_token_range_ptr = logical_token_range.data_ptr<int32_t>();
  }
  combine_internode_cuda(
      combine_x_ptrs.data_ptr(), combine_weight_ptrs, rdma_rail_send_win_handle,
      output.data_ptr(), output_weight, local_topk_indices.data_ptr<int32_t>(),
      local_topk_send_mask.data_ptr<int32_t>(),
      local_token_dst_scatter_indices.data_ptr<int32_t>(),
      num_tokens_per_rank.data_ptr<int32_t>(), has_weight, num_token,
      max_slot_num_token, hidden_size, topk, num_experts_per_rank, rank,
      num_ranks, local_world_size, num_sm, dev_comm, num_qps, dtype,
      weight_dtype, offset_dtype, combine_pipeline_chunks,
      logical_token_range_ptr, stream);
  ep_state.ep_combine_needs_barrier = true;
}

void combine_internode(
    torch::Tensor combine_x_ptrs,
    c10::optional<torch::Tensor> optional_combine_weight_ptrs,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    torch::Tensor num_tokens_per_rank, torch::Tensor output,
    torch::Tensor local_topk_indices, torch::Tensor local_topk_send_mask,
    torch::Tensor local_token_dst_scatter_indices,
    c10::optional<torch::Tensor> optional_output_weight,
    int32_t max_slot_num_token, int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<int64_t> opt_num_qps) {
  combine_internode_impl(
      combine_x_ptrs, optional_combine_weight_ptrs, rdma_rail_send_buf,
      rdma_rail_send_win_handle, num_tokens_per_rank, output,
      local_topk_indices, local_topk_send_mask, local_token_dst_scatter_indices,
      optional_output_weight, max_slot_num_token, num_experts_per_rank, num_sm,
      opt_num_qps, c10::nullopt);
}

void combine_internode_range(
    torch::Tensor combine_x_ptrs,
    c10::optional<torch::Tensor> optional_combine_weight_ptrs,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    torch::Tensor num_tokens_per_rank, torch::Tensor output,
    torch::Tensor local_topk_indices, torch::Tensor local_topk_send_mask,
    torch::Tensor local_token_dst_scatter_indices,
    c10::optional<torch::Tensor> optional_output_weight,
    int32_t max_slot_num_token, int32_t num_experts_per_rank, int32_t num_sm,
    c10::optional<int64_t> opt_num_qps, torch::Tensor logical_token_range) {
  combine_internode_impl(
      combine_x_ptrs, optional_combine_weight_ptrs, rdma_rail_send_buf,
      rdma_rail_send_win_handle, num_tokens_per_rank, output,
      local_topk_indices, local_topk_send_mask, local_token_dst_scatter_indices,
      optional_output_weight, max_slot_num_token, num_experts_per_rank, num_sm,
      opt_num_qps, logical_token_range);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
compute_dispatch_layout(
    torch::Tensor topk_indices, torch::Tensor token_within_expert_offset,
    torch::Tensor local_splits, uintptr_t full_splits_win_handle,
    torch::Tensor num_tokens_per_rank, int32_t num_experts, int32_t num_sm,
    c10::optional<torch::Tensor> optional_recv_token_count_cpu,
    c10::optional<torch::Tensor> optional_recv_token_count,
    int32_t expert_alignment) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  auto &ep_state = buffer::nccl_gin_require_state();
  const int32_t rank = ep_state.rank;
  const int32_t num_ranks = ep_state.nranks;
  const int32_t local_world_size = ep_state.local_world_size;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
  const void *dev_comm = static_cast<const void *>(buffer::nccl_gin_dev_comm());

  check_topk_indices(topk_indices);
  check_topk_indices(token_within_expert_offset);
  FLASH_CHECK(local_splits.is_cuda() && local_splits.is_contiguous() &&
              local_splits.scalar_type() == torch::kInt32);
  FLASH_CHECK(local_splits.dim() == 1);
  FLASH_CHECK(local_splits.size(0) == (num_experts + 1));
  FLASH_CHECK(topk_indices.sizes() == token_within_expert_offset.sizes());
  FLASH_CHECK(num_experts % num_ranks == 0);
  FLASH_CHECK(num_ranks % local_world_size == 0);
  FLASH_CHECK(full_splits_win_handle != 0);
  FLASH_CHECK(num_tokens_per_rank.is_cuda() &&
              num_tokens_per_rank.scalar_type() == torch::kInt32);
  FLASH_CHECK(num_tokens_per_rank.numel() == num_ranks);

  int32_t topk = topk_indices.size(1);
  int32_t num_token = topk_indices.size(0);
  int32_t experts_per_rank = num_experts / num_ranks;

  auto opts_i32 =
      torch::TensorOptions().dtype(torch::kInt32).device(topk_indices.device());
  torch::Tensor recv_base_offset =
      torch::empty({num_ranks, experts_per_rank, num_ranks}, opts_i32);
  torch::Tensor token_dst_scatter_indices =
      torch::empty({num_token, topk}, opts_i32);
  torch::Tensor token_topk_send_mask =
      torch::empty({num_token, topk}, opts_i32);

  torch::Tensor recv_token_count_cpu, recv_token_count;
  if (optional_recv_token_count_cpu.has_value()) {
    check_pinned_cpu_i32_vector(optional_recv_token_count_cpu.value(),
                                num_ranks, "recv_token_count_cpu");
    optional_recv_token_count_cpu.value().fill_(-1);
    recv_token_count_cpu = optional_recv_token_count_cpu.value();
  } else {
    recv_token_count_cpu = torch::full({num_ranks}, -1,
                                       torch::TensorOptions()
                                           .dtype(torch::kInt32)
                                           .device(torch::kCPU)
                                           .pinned_memory(true));
  }
  if (optional_recv_token_count.has_value()) {
    recv_token_count = optional_recv_token_count.value();
  } else {
    recv_token_count = torch::empty({num_ranks}, opts_i32);
  }

  torch::Tensor recv_aligned_token_count_cpu, recv_aligned_token_count;
  int32_t *recv_aligned_token_count_cpu_ptr = nullptr;
  int32_t *recv_aligned_token_count_ptr = nullptr;
  if (expert_alignment > 1) {
    recv_aligned_token_count_cpu = torch::full({num_ranks}, -1,
                                               torch::TensorOptions()
                                                   .dtype(torch::kInt32)
                                                   .device(torch::kCPU)
                                                   .pinned_memory(true));
    recv_aligned_token_count = torch::empty({num_ranks}, opts_i32);
    recv_aligned_token_count_cpu_ptr =
        recv_aligned_token_count_cpu.data_ptr<int32_t>();
    recv_aligned_token_count_ptr = recv_aligned_token_count.data_ptr<int32_t>();
  }

  torch::Tensor recv_expert_counts = torch::empty({experts_per_rank}, opts_i32);

  check_uva_enabled_for_current_device();

  // compute_dispatch_layout is a pure layout function. Staging the RDMA rail
  // source slot is the caller's responsibility (done every dispatch, so a
  // reused layout still refreshes the slot); the layout kernel no longer
  // writes it.
  compute_dispatch_layout_cuda(
      topk_indices.data_ptr<int32_t>(),
      token_within_expert_offset.data_ptr<int32_t>(),
      local_splits.data_ptr<int32_t>(), num_tokens_per_rank.data_ptr<int32_t>(),
      recv_base_offset.data_ptr<int32_t>(),
      token_dst_scatter_indices.data_ptr<int32_t>(),
      token_topk_send_mask.data_ptr<int32_t>(),
      recv_token_count_cpu.data_ptr<int32_t>(),
      recv_token_count.data_ptr<int32_t>(), recv_aligned_token_count_cpu_ptr,
      recv_aligned_token_count_ptr, recv_expert_counts.data_ptr<int32_t>(),
      num_token, topk, num_experts, rank, num_ranks, num_sm, expert_alignment,
      local_world_size, dev_comm,
      reinterpret_cast<void *>(full_splits_win_handle), stream);

  record_pinned_tensor(recv_token_count_cpu, stream);
  if (expert_alignment > 1) {
    record_pinned_tensor(recv_aligned_token_count_cpu, stream);
  }

  return {recv_base_offset,         token_dst_scatter_indices,
          token_topk_send_mask,     recv_token_count_cpu,
          recv_token_count,         recv_aligned_token_count_cpu,
          recv_aligned_token_count, recv_expert_counts};
}

void stage_dispatch_internode_range(
    torch::Tensor x, torch::Tensor topk_indices, torch::Tensor topk_send_mask,
    torch::Tensor token_dst_scatter_indices,
    c10::optional<torch::Tensor> optional_topk_weights,
    torch::Tensor rdma_rail_send_buf, uintptr_t rdma_rail_send_win_handle,
    int32_t max_slot_num_token, torch::Tensor logical_token_range) {
  if (!buffer::nccl_gin_is_initialized()) {
    throw std::runtime_error(
        "NCCL GIN not initialized; call buffer.nccl_gin_init first");
  }
  auto &ep_state = buffer::nccl_gin_require_state();
  FLASH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
              x.scalar_type() == torch::kBFloat16);
  check_topk_indices(topk_indices);
  FLASH_CHECK(topk_indices.sizes() == topk_send_mask.sizes() &&
              topk_indices.sizes() == token_dst_scatter_indices.sizes() &&
              topk_indices.size(0) == x.size(0));
  FLASH_CHECK(topk_send_mask.is_cuda() && topk_send_mask.is_contiguous() &&
              topk_send_mask.scalar_type() == torch::kInt32);
  FLASH_CHECK(token_dst_scatter_indices.is_cuda() &&
              token_dst_scatter_indices.is_contiguous() &&
              token_dst_scatter_indices.scalar_type() == torch::kInt32);
  const void *weights_ptr = nullptr;
  if (optional_topk_weights.has_value() &&
      optional_topk_weights.value().defined()) {
    auto weights = optional_topk_weights.value();
    FLASH_CHECK(weights.is_cuda() && weights.is_contiguous() &&
                weights.scalar_type() == torch::kFloat32 &&
                weights.sizes() == topk_indices.sizes());
    weights_ptr = weights.data_ptr();
  }
  FLASH_CHECK(max_slot_num_token >= x.size(0));
  check_tensor_common(logical_token_range, "logical_token_range", true,
                      torch::kInt32, 1);
  check_tensor_shape(logical_token_range, "logical_token_range", {2});
  checked_window_user_ptr(ep_state.comm, rdma_rail_send_win_handle,
                          rdma_rail_send_buf, "rdma_rail_send_buf");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  stage_dispatch_internode_range_cuda(
      rdma_rail_send_win_handle, x.data_ptr(), topk_indices.data_ptr<int32_t>(),
      topk_send_mask.data_ptr<int32_t>(),
      token_dst_scatter_indices.data_ptr<int32_t>(), weights_ptr,
      static_cast<int32_t>(x.size(0)), max_slot_num_token,
      static_cast<int32_t>(x.size(1)),
      static_cast<int32_t>(topk_indices.size(1)), ep_state.rank,
      ep_state.local_world_size, logical_token_range.data_ptr<int32_t>(),
      stream);
}

void bind_internode_ops(py::module &m) {
  // Single source of truth for the GIN signal-space layout
  // (flash_comm/ep/internode.h). Python must not redefine these values.
  m.attr("MAX_DISPATCH_PIPELINE_CHUNKS") = py::int_(kMaxDispatchPipelineChunks);
  m.attr("MAX_COMBINE_PIPELINE_CHUNKS") = py::int_(kMaxCombinePipelineChunks);
  m.def(
      "ep_required_gin_signal_count",
      [](int nnodes) { return ep_required_gin_signal_count(nnodes); },
      py::arg("nnodes"),
      "Minimum gin_signals needed for the context-local per-chunk dispatch "
      "and combine signals (independent of the QP count).");
  m.def(
      "ep_dispatch_signal_count",
      [](int nnodes) { return ep_dispatch_signal_count(nnodes); },
      py::arg("nnodes"), "Size of the dispatch signal id range [0, count).");
  m.def(
      "ep_combine_signal_base",
      [](int nnodes) { return ep_combine_signal_base(nnodes); },
      py::arg("nnodes"),
      "First combine signal id; the combine range is [base, "
      "ep_required_gin_signal_count).");

  m.def("barrier_all_on_stream", &barrier_all_on_stream);
  m.def("barrier_all_on_stream_if_active", &barrier_all_on_stream_if_active,
        py::arg("logical_token_range"));
  m.def("reset_signals_barrier_all_on_stream",
        &reset_signals_barrier_all_on_stream, py::arg("signal_begin"),
        py::arg("signal_end"),
        "Reset EP GIN signals in [signal_begin, signal_end) at the quiescent "
        "point, then world barrier. Required after each internode "
        "dispatch/combine with that protocol's signal range.");
  m.def("reset_signals_barrier_all_on_stream_if_active",
        &reset_signals_barrier_all_on_stream_if_active, py::arg("signal_begin"),
        py::arg("signal_end"), py::arg("logical_token_range"));

  m.def("compute_dispatch_layout", &compute_dispatch_layout,
        py::arg("topk_indices"), py::arg("token_within_expert_offset"),
        py::arg("local_splits"), py::arg("full_splits_win_handle"),
        py::arg("num_tokens_per_rank"), py::arg("num_experts"),
        py::arg("num_sm"), py::arg("recv_token_count_cpu") = c10::nullopt,
        py::arg("recv_token_count") = c10::nullopt,
        py::arg("expert_alignment") = 1);
  m.def("stage_dispatch_internode_range", &stage_dispatch_internode_range,
        py::arg("x"), py::arg("topk_indices"), py::arg("topk_send_mask"),
        py::arg("token_dst_scatter_indices"), py::arg("topk_weights"),
        py::arg("rdma_rail_send_buf"), py::arg("rdma_rail_send_win_handle"),
        py::arg("max_slot_num_token"), py::arg("logical_token_range"));

  m.def(
      "rdma_rail_send_slot_stride_bytes",
      [](int num_token, int hidden, int topk) {
        return static_cast<int64_t>(
            rdma_rail_send_slot_stride_bytes(num_token, hidden, topk));
      },
      py::arg("num_token"), py::arg("hidden_size"), py::arg("topk"));

  m.def(
      "rdma_rail_send_layout_desc",
      [](int max_slot_num_token, int hidden, int topk, int nnodes) {
        const auto desc = rdma_rail_send_layout_desc(max_slot_num_token, hidden,
                                                     topk, nnodes);
        py::dict out;
        out["max_slot_num_token"] = desc.max_slot_num_token;
        out["hidden_size"] = desc.hidden_size;
        out["topk"] = desc.topk;
        out["nnodes"] = desc.nnodes;
        out["token_region_bytes"] =
            static_cast<int64_t>(desc.token_region_bytes);
        out["meta_region_bytes"] = static_cast<int64_t>(desc.meta_region_bytes);
        out["topk_indices_offset"] =
            static_cast<int64_t>(desc.topk_indices_offset);
        out["topk_send_mask_offset"] =
            static_cast<int64_t>(desc.topk_send_mask_offset);
        out["token_dst_scatter_offset"] =
            static_cast<int64_t>(desc.token_dst_scatter_offset);
        out["topk_weights_offset"] =
            static_cast<int64_t>(desc.topk_weights_offset);
        out["slot_payload_bytes"] =
            static_cast<int64_t>(desc.slot_payload_bytes);
        out["slot_stride_bytes"] = static_cast<int64_t>(desc.slot_stride_bytes);
        out["source_slot_base"] = desc.source_slot_base;
        out["outgoing_slot_base"] = desc.outgoing_slot_base;
        out["incoming_slot_base"] = desc.incoming_slot_base;
        out["num_slots"] = desc.num_slots;
        out["buffer_bytes"] = static_cast<int64_t>(desc.buffer_bytes);
        return out;
      },
      py::arg("max_slot_num_token"), py::arg("hidden_size"), py::arg("topk"),
      py::arg("nnodes"));

  m.def("dispatch_internode", &dispatch_internode, py::arg("recv_x_ptrs"),
        py::arg("recv_weights_ptrs"), py::arg("recv_topk_scatter_indices_ptrs"),
        py::arg("max_recv_tokens"), py::arg("rdma_rail_send_buf"),
        py::arg("rdma_rail_send_win_handle"), py::arg("num_tokens_per_rank"),
        py::arg("node_topk_indices"), py::arg("node_topk_send_mask"),
        py::arg("node_token_dst_scatter_indices"),
        py::arg("max_slot_num_token"), py::arg("local_num_token"),
        py::arg("hidden_size"), py::arg("has_weight"),
        py::arg("num_experts_per_rank"), py::arg("num_sm"),
        py::arg("num_qps") = c10::nullopt);
  m.def("dispatch_internode_range", &dispatch_internode_range,
        py::arg("recv_x_ptrs"), py::arg("recv_weights_ptrs"),
        py::arg("recv_topk_scatter_indices_ptrs"), py::arg("max_recv_tokens"),
        py::arg("rdma_rail_send_buf"), py::arg("rdma_rail_send_win_handle"),
        py::arg("num_tokens_per_rank"), py::arg("node_topk_indices"),
        py::arg("node_topk_send_mask"),
        py::arg("node_token_dst_scatter_indices"),
        py::arg("max_slot_num_token"), py::arg("local_num_token"),
        py::arg("hidden_size"), py::arg("has_weight"),
        py::arg("num_experts_per_rank"), py::arg("num_sm"), py::arg("num_qps"),
        py::arg("logical_token_range"));

  m.def("combine_internode", &combine_internode, py::arg("combine_x_ptrs"),
        py::arg("combine_weight_ptrs"), py::arg("rdma_rail_send_buf"),
        py::arg("rdma_rail_send_win_handle"), py::arg("num_tokens_per_rank"),
        py::arg("output"), py::arg("local_topk_indices"),
        py::arg("local_topk_send_mask"),
        py::arg("local_token_dst_scatter_indices"), py::arg("output_weight"),
        py::arg("max_slot_num_token"), py::arg("num_experts_per_rank"),
        py::arg("num_sm"), py::arg("num_qps") = c10::nullopt);
  m.def("combine_internode_range", &combine_internode_range,
        py::arg("combine_x_ptrs"), py::arg("combine_weight_ptrs"),
        py::arg("rdma_rail_send_buf"), py::arg("rdma_rail_send_win_handle"),
        py::arg("num_tokens_per_rank"), py::arg("output"),
        py::arg("local_topk_indices"), py::arg("local_topk_send_mask"),
        py::arg("local_token_dst_scatter_indices"), py::arg("output_weight"),
        py::arg("max_slot_num_token"), py::arg("num_experts_per_rank"),
        py::arg("num_sm"), py::arg("num_qps"), py::arg("logical_token_range"));
}

} // namespace internode
} // namespace ep
} // namespace flash_comm
