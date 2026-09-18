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
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace flash_comm {
namespace ep {
namespace internode {

// ---------------------------------------------------------------------------
// EP GIN signal-space layout -- single source of truth.
//
// Shared by the device kernels (csrc/ep/kernels/internode_cuda.cu), host-side
// resource validation (csrc/buffer/nccl_gin.cpp) and Python (exported through
// pybind in csrc/ep/internode.cpp as MAX_*_PIPELINE_CHUNKS /
// ep_required_gin_signal_count). Do not redefine these values elsewhere.
//
// Signal id space (disjoint per protocol). GIN signal storage is
// per-context: every context owns an independent array indexed by the same id
// numbering, and FlashComm binds one GIN context per QP (context qp carries
// exactly QP qp's traffic). Signal ids are therefore context-local and do not
// encode the QP -- the same id on context qp0 and qp1 are distinct physical
// counters:
//   dispatch: node * kMaxDispatchPipelineChunks + chunk,
//             chunk in [0, kMaxDispatchPipelineChunks)
//   combine : ep_combine_signal_base(nnodes) + node
// The required gin_signals count is independent of the QP count; a call
// running with num_qps = N only touches contexts qp < N.
//
// Signal lifecycle: after each dispatch (resp. combine) a fused
// reset-signals-then-barrier kernel resets that protocol's signal id range to
// 0 on every context at a globally quiescent point, and each dispatch/combine
// call emits exactly one increment per participating (context, signal), so
// waiters always target the constant value 1. No epoch state is kept
// anywhere. The reset must precede the barrier arrival: once peers pass the
// barrier they immediately issue the next step's puts, and a reset racing
// with an incoming SignalInc would lose the increment and hang the next
// waiter.
// ---------------------------------------------------------------------------
constexpr int32_t kMaxDispatchPipelineChunks = 32;
constexpr int32_t kMaxCombinePipelineChunks = 32;

constexpr int32_t ep_dispatch_signal_count(int32_t nnodes) {
  return nnodes * kMaxDispatchPipelineChunks;
}

constexpr int32_t ep_combine_signal_base(int32_t nnodes) {
  return ep_dispatch_signal_count(nnodes);
}

constexpr int32_t ep_combine_signal_count(int32_t nnodes) { return nnodes; }

constexpr int32_t ep_required_gin_signal_count(int32_t nnodes) {
  return ep_dispatch_signal_count(nnodes) + ep_combine_signal_count(nnodes);
}

struct RDMARailSendLayoutDesc {
  int32_t max_slot_num_token = 0;
  int32_t hidden_size = 0;
  int32_t topk = 0;
  int32_t nnodes = 0;
  size_t token_region_bytes = 0;
  size_t meta_region_bytes = 0;
  size_t topk_indices_offset = 0;
  size_t topk_send_mask_offset = 0;
  size_t token_dst_scatter_offset = 0;
  size_t topk_weights_offset = 0;
  size_t slot_payload_bytes = 0;
  size_t slot_stride_bytes = 0;
  int32_t source_slot_base = 0;
  int32_t outgoing_slot_base = 0;
  int32_t incoming_slot_base = 0;
  int32_t num_slots = 0;
  size_t buffer_bytes = 0;
};

inline size_t align_gin_stride_bytes(size_t bytes) {
  constexpr size_t kGinStrideAlign = 4096;
  return (bytes + kGinStrideAlign - 1) / kGinStrideAlign * kGinStrideAlign;
}

inline RDMARailSendLayoutDesc
rdma_rail_send_layout_desc(int32_t max_slot_num_token, int32_t hidden_size,
                           int32_t topk, int32_t nnodes) {
  RDMARailSendLayoutDesc desc{};
  desc.max_slot_num_token = max_slot_num_token;
  desc.hidden_size = hidden_size;
  desc.topk = topk;
  desc.nnodes = nnodes;
  desc.token_region_bytes =
      static_cast<size_t>(max_slot_num_token) * hidden_size * sizeof(uint16_t);
  desc.meta_region_bytes =
      static_cast<size_t>(max_slot_num_token) * topk * sizeof(int32_t);
  desc.topk_indices_offset = desc.token_region_bytes;
  desc.topk_send_mask_offset =
      desc.topk_indices_offset + desc.meta_region_bytes;
  desc.token_dst_scatter_offset =
      desc.topk_send_mask_offset + desc.meta_region_bytes;
  desc.topk_weights_offset =
      desc.token_dst_scatter_offset + desc.meta_region_bytes;
  desc.slot_payload_bytes =
      desc.topk_weights_offset +
      static_cast<size_t>(max_slot_num_token) * topk * sizeof(float);
  desc.slot_stride_bytes = align_gin_stride_bytes(desc.slot_payload_bytes);
  desc.source_slot_base = 0;
  desc.outgoing_slot_base = nnodes;
  desc.incoming_slot_base = 2 * nnodes;
  desc.num_slots = 3 * nnodes;
  desc.buffer_bytes =
      desc.slot_stride_bytes * static_cast<size_t>(desc.num_slots);
  return desc;
}

inline size_t rdma_rail_send_slot_stride_bytes(int32_t max_slot_num_token,
                                               int32_t hidden_size,
                                               int32_t topk) {
  return rdma_rail_send_layout_desc(max_slot_num_token, hidden_size, topk, 1)
      .slot_stride_bytes;
}

void dispatch_internode_cuda(
    uintptr_t rdma_rail_send_win_handle, int32_t *num_tokens_per_rank,
    int32_t *node_topk_indices, int32_t *node_topk_send_mask,
    int32_t *node_token_dst_scatter_indices, bool has_weight, void *recv_x_ptrs,
    void **recv_weights_ptrs, void **recv_topk_scatter_indices_ptrs,
    int32_t max_slot_num_token, int32_t hidden_size,
    int32_t num_experts_per_rank, int32_t rank, int32_t num_ranks,
    int32_t local_world_size, int32_t max_recv_tokens, int32_t num_sm,
    const void *dev_comm_host, int32_t num_qps, FlashCommDType dtype,
    FlashCommDType weight_dtype, FlashCommDType offset_dtype, int32_t topk,
    int32_t dispatch_pipeline_chunks, const int32_t *logical_token_range,
    cudaStream_t stream);

void stage_dispatch_internode_range_cuda(
    uintptr_t rdma_rail_send_win_handle, const void *x,
    const int32_t *topk_indices, const int32_t *topk_send_mask,
    const int32_t *token_dst_scatter_indices, const void *topk_weights,
    int32_t num_token, int32_t max_slot_num_token, int32_t hidden_size,
    int32_t topk, int32_t rank, int32_t local_world_size,
    const int32_t *logical_token_range, cudaStream_t stream);

void combine_internode_cuda(
    void *combine_x_ptrs, void *combine_weight_ptrs,
    uintptr_t rdma_rail_send_win_handle, void *output, void *output_weight,
    int32_t *local_topk_indices, int32_t *local_topk_send_mask,
    int32_t *local_token_dst_scatter_indices, int32_t *num_tokens_per_rank,
    bool has_weight, int32_t num_token, int32_t max_slot_num_token,
    int32_t hidden_size, int32_t topk, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks, int32_t local_world_size, int32_t num_sm,
    const void *dev_comm_host, int32_t num_qps, FlashCommDType dtype,
    FlashCommDType weight_dtype, FlashCommDType offset_dtype,
    int32_t combine_pipeline_chunks, const int32_t *logical_token_range,
    cudaStream_t stream);

// Flushes all payload contexts and performs the world barrier. Does not touch
// any GIN signal.
void internode_barrier_on_stream_cuda(const void *dev_comm_host,
                                      int32_t max_qps, cudaStream_t stream);

void internode_barrier_on_stream_if_active_cuda(
    const void *dev_comm_host, int32_t max_qps,
    const int32_t *logical_token_range, cudaStream_t stream);

// Fused variant: resets the context-local EP GIN signals in
// [signal_begin, signal_end) to 0 on every context in [0, max_qps) and then
// performs the world barrier. The reset runs before the barrier arrival, when
// those signals are globally quiescent (see the signal lifecycle note above).
void internode_reset_signals_barrier_on_stream_cuda(const void *dev_comm_host,
                                                    int32_t max_qps,
                                                    int32_t signal_begin,
                                                    int32_t signal_end,
                                                    cudaStream_t stream);

void internode_reset_signals_barrier_on_stream_if_active_cuda(
    const void *dev_comm_host, int32_t max_qps, int32_t signal_begin,
    int32_t signal_end, const int32_t *logical_token_range,
    cudaStream_t stream);

void compute_dispatch_layout_cuda(
    int32_t *topk_indices, int32_t *token_within_expert_offset,
    int32_t *local_splits, int32_t *num_tokens_per_rank,
    int32_t *recv_base_offset, int32_t *token_dst_scatter_indices,
    int32_t *token_topk_send_mask, int32_t *recv_token_count_cpu,
    int32_t *recv_token_count, int32_t *recv_aligned_token_count_cpu,
    int32_t *recv_aligned_token_count, int32_t *recv_expert_counts,
    int32_t num_token, int32_t topk, int32_t num_experts, int32_t rank,
    int32_t num_ranks, int32_t num_sm, int32_t expert_alignment,
    int32_t local_world_size, const void *dev_comm_host,
    void *full_splits_win_ptr, cudaStream_t stream);

} // namespace internode
} // namespace ep
} // namespace flash_comm
