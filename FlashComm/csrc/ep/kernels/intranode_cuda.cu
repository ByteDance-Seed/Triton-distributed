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

#include <cooperative_groups.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "flash_comm/common.h"
#include "flash_comm/copy.cuh"
#include "flash_comm/ep/intranode.h"
#include "flash_comm/launch_utils.cuh"
#include "flash_comm/utils.cuh"

namespace flash_comm {
namespace ep {
namespace intranode {

namespace kernels {

namespace smem {
constexpr int32_t kTMAAlignment = 128;
// Original dispatch smem (stage-per-token, no token-chunk meta prefetch).
template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kNumStages>
struct DispatchIntraNodeSmem {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each stage must be kTMAAlignment-byte aligned for TMA");

  uint64_t mbar_full[kNumStages];
  uint64_t mbar_empty[kNumStages];

  token_t *recv_x_ptrs[flash_comm::kMaxWorldSize];
  weight_t *recv_weights_ptrs[flash_comm::kMaxWorldSize];
  offset_t *recv_topk_scatter_indices_ptrs[flash_comm::kMaxWorldSize];

  alignas(kTMAAlignment) token_t tma_buffer[kNumStages][kHiddenSize];
};

// Chunked dispatch smem: token-chunk meta prefetch via TMA into ping-pong
// buffers.
template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kNumStages, int32_t kTopk,
          int32_t kChunkSize>
struct DispatchIntraNodeChunkSmem {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each stage must be kTMAAlignment-byte aligned for TMA");

  uint64_t mbar_full[kNumStages];
  uint64_t mbar_empty[kNumStages];

  token_t *recv_x_ptrs[flash_comm::kMaxWorldSize];
  weight_t *recv_weights_ptrs[flash_comm::kMaxWorldSize];
  offset_t *recv_topk_scatter_indices_ptrs[flash_comm::kMaxWorldSize];

  uint64_t meta_mbar_full[2];
  uint64_t meta_mbar_empty[2];
  alignas(16) int32_t meta_topk_send_mask[2][kChunkSize][kTopk];
  alignas(16) weight_t meta_topk_weights[2][kChunkSize][kTopk];
  alignas(16) offset_t meta_topk_indices[2][kChunkSize][kTopk];
  alignas(16) offset_t meta_token_dst_scatter_indices[2][kChunkSize][kTopk];

  alignas(kTMAAlignment) token_t tma_buffer[kNumStages][kHiddenSize];
};

template <typename token_t, int32_t kHiddenSize, int32_t kNumStages>
struct DispatchPostprocessSmem {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each stage must be kTMAAlignment-byte aligned for TMA");

  uint64_t mbar_full[kNumStages];
  uint64_t mbar_empty[kNumStages];

  alignas(kTMAAlignment) token_t tma_buffer[kNumStages][kHiddenSize];
};

// kNumStoreStages > 0
template <typename token_t, typename weight_t, int32_t kHiddenSize,
          int32_t kNumLoadStages, int32_t kNumStoreStages,
          int32_t kNumWGPerBlock>
struct CombineIntraNodeSmem {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each stage must be kTMAAlignment-byte aligned for TMA");
  struct WarpGroupSmem {
    uint64_t mbar_full[kNumLoadStages];
    uint64_t mbar_empty[kNumLoadStages];

    alignas(kTMAAlignment) token_t tma_load_buffer[kNumLoadStages][kHiddenSize];
    alignas(kTMAAlignment) token_t
        tma_store_buffer[kNumStoreStages][kHiddenSize];
  };
  token_t *x_ptrs[flash_comm::kMaxWorldSize];
  weight_t *weight_ptrs[flash_comm::kMaxWorldSize];
  WarpGroupSmem warp_group_smem[kNumWGPerBlock];
};

// kNumStoreStages = 0
template <typename token_t, typename weight_t, int32_t kHiddenSize,
          int32_t kNumLoadStages, int32_t kNumWGPerBlock>
struct CombineIntraNodeSmem<token_t, weight_t, kHiddenSize, kNumLoadStages, 0,
                            kNumWGPerBlock> {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each stage must be kTMAAlignment-byte aligned for TMA");
  struct WarpGroupSmem {
    uint64_t mbar_full[kNumLoadStages];
    uint64_t mbar_empty[kNumLoadStages];

    alignas(kTMAAlignment) token_t tma_load_buffer[kNumLoadStages][kHiddenSize];
  };
  token_t *x_ptrs[flash_comm::kMaxWorldSize];
  weight_t *weight_ptrs[flash_comm::kMaxWorldSize];
  WarpGroupSmem warp_group_smem[kNumWGPerBlock];
};

} // namespace smem

template <typename weight_t, typename offset_t, int32_t kTopk,
          int32_t kChunkSize, typename SmemT, bool kHasWeight>
__device__ __forceinline__ void issue_meta_prefetch_one_thread(
    SmemT &smem, int warp_id, int lane_id,
    int32_t *topk_send_mask,             // [num_token, kTopk]
    void *topk_weights,                  // [num_token, kTopk]
    offset_t *topk_indices,              // [num_token, kTopk]
    offset_t *token_dst_scatter_indices, // [num_token, kTopk]
    int buf, int chunk_base, int chunk_len) {
  if (chunk_len <= 0)
    return;
  // Pick a single issuer thread from *consumer warps* (warp 0 lane 0).
  if (!(warp_id == 0 && lane_id == 0))
    return;

  const int32_t elems = chunk_len * kTopk;
  static_assert(kTopk * sizeof(int32_t) % 16 == 0,
                "kTopk * sizeof(int32_t) must be 16-byte aligned");
  static_assert(kTopk * sizeof(weight_t) % 16 == 0,
                "kTopk * sizeof(weight_t) must be 16-byte aligned");
  static_assert(kTopk * sizeof(offset_t) % 16 == 0,
                "kTopk * sizeof(offset_t) must be 16-byte aligned");

  const int32_t bytes_send_mask = elems * sizeof(int32_t);
  const int32_t bytes_weights = kHasWeight ? (elems * sizeof(weight_t)) : 0;
  const int32_t bytes_indices = elems * sizeof(offset_t);
  const int32_t bytes_scatter = elems * sizeof(offset_t);
  const int32_t total_bytes =
      bytes_send_mask + bytes_weights + bytes_indices + bytes_scatter;

  uint64_t *mbar = &smem.meta_mbar_full[buf];
  const void *g_send_mask =
      reinterpret_cast<const int32_t *>(topk_send_mask) + chunk_base * kTopk;
  const void *g_weights =
      kHasWeight ? (reinterpret_cast<const weight_t *>(topk_weights) +
                    chunk_base * kTopk)
                 : nullptr;
  const void *g_indices =
      reinterpret_cast<const offset_t *>(topk_indices) + chunk_base * kTopk;
  const void *g_scatter =
      reinterpret_cast<const offset_t *>(token_dst_scatter_indices) +
      chunk_base * kTopk;

  void *s_send_mask = &smem.meta_topk_send_mask[buf][0][0];
  void *s_weights = &smem.meta_topk_weights[buf][0][0];
  void *s_indices = &smem.meta_topk_indices[buf][0][0];
  void *s_scatter = &smem.meta_token_dst_scatter_indices[buf][0][0];

  mbar_arrive_and_set_barrier_transaction_bytes(mbar, total_bytes);
  tma_copy_1d_g2s(g_send_mask, mbar, s_send_mask, bytes_send_mask);
  if constexpr (kHasWeight) {
    tma_copy_1d_g2s(g_weights, mbar, s_weights, bytes_weights);
  }
  tma_copy_1d_g2s(g_indices, mbar, s_indices, bytes_indices);
  tma_copy_1d_g2s(g_scatter, mbar, s_scatter, bytes_scatter);
}

// if value is not -1, accumulate 1 to hist_ptr[value]
int32_t __device__ __forceinline__
calc_rank_in_warp_and_accumulate(int32_t value, int32_t *hist_ptr) {
  const int32_t thread_id = threadIdx.x;
  const int32_t lane_id = thread_id % WARP_SIZE;
  unsigned m = __match_any_sync(0xffffffff, value);
  int rank_in_warp = __popc(m & ((1u << lane_id) - 1));

  if (value != -1) {
    atomicAdd(hist_ptr + value, 1);
  }
  __syncwarp();
  return rank_in_warp;
}

template <typename T> T __device__ __forceinline__ warp_reduce_sum(T value) {
  value += __shfl_xor_sync(~0, value, 16);
  value += __shfl_xor_sync(~0, value, 8);
  value += __shfl_xor_sync(~0, value, 4);
  value += __shfl_xor_sync(~0, value, 2);
  value += __shfl_xor_sync(~0, value, 1);
  return value;
}

template <typename T>
T __device__ __forceinline__ warp_scan_inclusive(T value) {
  const int32_t thread_id = threadIdx.x;
  const int32_t lane_id = thread_id % WARP_SIZE;
  for (int32_t i = 1; i < WARP_SIZE; i <<= 1) {
    T val = __shfl_up_sync(~0, value, i);
    if (lane_id >= i)
      value += val;
  }
  return value;
}

template <typename T>
__device__ __forceinline__ T block_scan_inclusive(T value, T *warp_sums) {
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int num_warps = (blockDim.x + 31) >> 5;

  value = warp_scan_inclusive(value);

  if (lane == WARP_SIZE - 1)
    warp_sums[warp] = value;
  __syncthreads();

  if (warp == 0) {
    T x = (lane < num_warps) ? warp_sums[lane] : T(0);
    x = warp_scan_inclusive(x);
    if (lane < num_warps)
      warp_sums[lane] = x;
  }
  __syncthreads();

  T warp_prefix = (warp == 0) ? T(0) : warp_sums[warp - 1];
  return warp_prefix + value;
}

template <typename T>
void __device__ __forceinline__ barrier_all_block(T **barrier_ptrs,
                                                  int32_t rank,
                                                  int32_t num_ranks) {
  const int32_t thread_id = threadIdx.x;
  if (thread_id < num_ranks) {
    __threadfence_system();
    T *remote_ptr = barrier_ptrs[thread_id] + rank;
    while (atomicCAS_system(remote_ptr, 0, 1) != 0) {
    }
    __threadfence_system();
    T *wait_ptr = barrier_ptrs[rank] + thread_id;
    while (atomicCAS_system(wait_ptr, 1, 0) != 1) {
    }
    __threadfence_system();
  }
  __syncthreads();
}

template <typename T>
void __global__ __launch_bounds__(128, 1)
    kernel_barrier_all_on_stream(T **barrier_ptrs, int32_t rank,
                                 int32_t num_ranks) {
  barrier_all_block<T>(reinterpret_cast<T **>(barrier_ptrs), rank, num_ranks);
}

template <int32_t kNumWarps>
void __global__ __launch_bounds__(1024, 1)
    kernel_compute_stable_local_token_within_expert_offset(
        int32_t *topk_indices, // [num_token, topk]
        int32_t num_token, int32_t topk, int32_t num_experts,
        int32_t *block_cumsum_hist,          // [num_tiles, num_experts + 1]
        int32_t *token_within_expert_offset, // [num_token, topk]
        int32_t *expert_counts // [num_experts + 1] (optional, can be nullptr)
    ) {
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;
  constexpr int32_t kBlockSize = kNumWarps * WARP_SIZE;
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  int32_t *warp_hist_all =
      reinterpret_cast<int32_t *>(smem_buffer); // [kNumWarps, num_experts + 1]
  int32_t *warp_hist = warp_hist_all + warp_id * (num_experts + 1);

  for (int32_t i = lane_id; i < num_experts + 1; i += WARP_SIZE) {
    warp_hist[i] = 0;
  }
  __syncthreads();

  int32_t num_tiles = (num_token * topk + kBlockSize - 1) / kBlockSize;
  for (int32_t tile_id = block_id; tile_id < num_tiles; tile_id += num_block) {
    int32_t tile_start = tile_id * kBlockSize;
    int32_t tile_end = min(tile_start + kBlockSize, num_token * topk);
    for (int32_t j = thread_id; j < tile_end - tile_start; j += kBlockSize) {
      int32_t idx = tile_start + j;
      int32_t value = idx < tile_end ? topk_indices[idx] : -1;
      if (value != -1) {
        atomicAdd(warp_hist + value, 1);
      }
    }
    __syncthreads();
    // reduce along kNumWarps dimension: [kNumWarps, num_experts + 1] -> [1,
    // num_experts + 1]
    for (int32_t j = warp_id; j < num_experts + 1; j += kNumWarps) {
      int32_t value = lane_id < kNumWarps
                          ? warp_hist_all[j + lane_id * (num_experts + 1)]
                          : 0;
      int32_t sum = warp_reduce_sum(value);
      if (lane_id == 0) {
        block_cumsum_hist[tile_id * (num_experts + 1) + j] = sum;
      }
    }
    __syncthreads();
    // reset warp_hist
    for (int32_t i = lane_id; i < num_experts + 1; i += WARP_SIZE) {
      warp_hist[i] = 0;
    }
  }
  __syncthreads();

  // grid sync to wait for all tiles to finish
  cooperative_groups::this_grid().sync();

  // scan along num_tiles dimension(exclusive scan): [num_tiles, num_experts +
  // 1] -> [num_tiles, num_experts + 1]
  int32_t global_warp_id = block_id * kNumWarps + warp_id;
  for (int32_t exp_idx = global_warp_id; exp_idx < num_experts + 1;
       exp_idx += kNumWarps * num_block) {
    int32_t global_prefix_sum = 0;
    int32_t num_tiles_padding =
        (num_tiles + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    for (int32_t j = lane_id; j < num_tiles_padding; j += WARP_SIZE) {
      int32_t value = j < num_tiles
                          ? block_cumsum_hist[exp_idx + j * (num_experts + 1)]
                          : 0;
      int32_t warp_pre_sum = warp_scan_inclusive(value);
      if (j < num_tiles) {
        block_cumsum_hist[exp_idx + j * (num_experts + 1)] =
            global_prefix_sum + warp_pre_sum - value;
      }
      int32_t warp_sum = __shfl_sync(~0, warp_pre_sum, WARP_SIZE - 1);
      global_prefix_sum += warp_sum;
    }
    if (expert_counts != nullptr && lane_id == 0) {
      // total count for this expert across all tiles
      expert_counts[exp_idx] = global_prefix_sum;
    }
  }
  __syncthreads();

  // grid sync to wait block_cumsum_hist to be ready
  cooperative_groups::this_grid().sync();
  for (int32_t i = lane_id; i < num_experts + 1; i += WARP_SIZE) {
    warp_hist[i] = 0;
  }
  __syncthreads();

  for (int32_t tile_id = block_id; tile_id < num_tiles; tile_id += num_block) {
    int32_t tile_start = tile_id * kBlockSize;
    int32_t tile_end = min(tile_start + kBlockSize, num_token * topk);
    int32_t value = tile_start + thread_id < tile_end
                        ? topk_indices[tile_start + thread_id]
                        : -1;
    int32_t rank_in_warp = calc_rank_in_warp_and_accumulate(value, warp_hist);
    __syncthreads();

    // scan along kNumWarps dimension: [kNumWarps, num_experts + 1] -> [1,
    // num_experts + 1]
    for (int32_t j = warp_id; j < num_experts + 1; j += kNumWarps) {
      int32_t value = lane_id < kNumWarps
                          ? warp_hist_all[j + lane_id * (num_experts + 1)]
                          : 0;
      int32_t warp_pre_sum = warp_scan_inclusive(value);
      __syncwarp();
      if (lane_id < kNumWarps) {
        warp_hist_all[j + lane_id * (num_experts + 1)] = warp_pre_sum - value;
      }
      __syncwarp();
    }
    __syncthreads();

    if (tile_start + thread_id < tile_end) {
      token_within_expert_offset[tile_start + thread_id] =
          block_cumsum_hist[tile_id * (num_experts + 1) + value] +
          warp_hist[value] + rank_in_warp;
    }
    __syncthreads();
    // reset
    for (int32_t i = lane_id; i < num_experts + 1; i += WARP_SIZE) {
      warp_hist[i] = 0;
    }
  }
}

// 1. all gather local_splits to full_splits
// 2. cumsum full_splits to get recv_base_offset
// 3. calc token destination index
// 4. calc token send mask
// NOTE: token_within_expert_offset and local_splits are computed by
// kernel_compute_stable_local_token_within_expert_offset
template <int32_t kNumWarps>
void __global__ __launch_bounds__(1024, 1) kernel_compute_dispatch_layout(
    int32_t *topk_indices,               // [num_token, topk]
    int32_t *token_within_expert_offset, // [num_token, topk]
    int32_t *local_splits,               // [num_experts + 1]
    int32_t **full_splits_ptrs,          // [num_ranks, num_experts + 1]
    int32_t **barrier_ptrs,              // [num_ranks]
    int32_t *recv_base_offset, // [num_ranks, experts_per_rank, num_ranks],
                               // dst_rank, local_expert_idx, src_rank
    int32_t *token_dst_scatter_indices, // [num_token, topk]
    int32_t *token_topk_send_mask,      // [num_token, topk]
    int32_t *recv_token_count_cpu,      // [num_ranks] (optional, pinned memory)
    int32_t *recv_token_count,          // [num_ranks] (optional, device memory)
    int32_t num_token, int32_t topk, int32_t num_experts, int32_t rank,
    int32_t num_ranks) {
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  constexpr int32_t kBlockSize = kNumWarps * WARP_SIZE;
  const int32_t num_experts_per_rank = num_experts / num_ranks;
  __shared__ __align__(1024) int32_t scan_warp_prefix_sum[kNumWarps];

  // push-mode all gather: each block writes local_splits to a remote rank's
  // symmetric buffer. The writes go across NVLink to peer GPUs, so a
  // system-scope fence is required to guarantee they are visible to remote
  // ranks before the subsequent barrier signals readiness.
  // grid().sync() alone only orders within the local GPU.
  for (int32_t i = block_id; i < num_ranks; i += num_block) {
    int32_t *remote_full_splits =
        full_splits_ptrs[i] + rank * (num_experts + 1);
    for (int32_t j = thread_id; j < num_experts + 1; j += kBlockSize) {
      remote_full_splits[j] = local_splits[j];
    }
  }
  __threadfence_system();

  cooperative_groups::this_grid().sync();
  if (block_id == 0) {
    barrier_all_block<int32_t>(barrier_ptrs, rank, num_ranks);
  }
  __threadfence_system();
  cooperative_groups::this_grid().sync();

  int32_t *local_full_splits_ptr = full_splits_ptrs[rank];
  // cumsum along num_experts dimension: [num_ranks, num_ranks,
  // experts_per_rank] -> [num_ranks, num_ranks, experts_per_rank]
  for (int32_t dst_rank = block_id; dst_rank < num_ranks;
       dst_rank += num_block) {
    int32_t src_rank = thread_id % num_ranks;
    int32_t local_expert_idx = thread_id / num_ranks;
    // assume num_expert <= kBlockSize
    int32_t value =
        thread_id < num_experts
            ? local_full_splits_ptr[src_rank * (num_experts + 1) +
                                    dst_rank * num_experts_per_rank +
                                    local_expert_idx]
            : 0;
    int32_t prefix_sum = block_scan_inclusive(value, scan_warp_prefix_sum);
    if (thread_id < num_experts) {
      recv_base_offset[dst_rank * num_experts + local_expert_idx * num_ranks +
                       src_rank] = prefix_sum - value;
    }

    if (thread_id == num_experts - 1) {
      if (recv_token_count != nullptr) {
        recv_token_count[dst_rank] = prefix_sum;
      }
      if (recv_token_count_cpu != nullptr) {
        recv_token_count_cpu[dst_rank] = prefix_sum;
      }
      __threadfence_system();
    }
    __syncthreads();
  }

  cooperative_groups::this_grid().sync();

  // token scatter indices and send mask
  for (int32_t i = block_id * kBlockSize + thread_id; i < num_token * topk;
       i += kBlockSize * num_block) {
    int32_t expert_idx = topk_indices[i];
    int32_t target_rank = expert_idx / num_experts_per_rank;
    int32_t target_local_expert_idx = expert_idx % num_experts_per_rank;
    int32_t num_pre_expert = i % topk;
    int32_t need_send = target_rank < num_ranks ? 1 : 0;
    __syncwarp();
    if (need_send) {
      for (int32_t j = 0; j < num_pre_expert; ++j) {
        int32_t cur_expert_idx = topk_indices[i / topk * topk + j];
        int32_t cur_target_rank = cur_expert_idx / num_experts_per_rank;
        if (cur_target_rank == target_rank) {
          need_send = 0;
          break;
        }
      }
    }
    __syncwarp();
    int32_t scatter_idx = -1;
    if (target_rank < num_ranks) {
      scatter_idx =
          recv_base_offset[target_rank * num_experts +
                           target_local_expert_idx * num_ranks + rank] +
          token_within_expert_offset[i];
    }
    token_dst_scatter_indices[i] = scatter_idx;
    token_topk_send_mask[i] = need_send;
  }
}

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kTopk, int32_t kNumStages,
          int32_t kNumConsumerGroups>
void __global__ __launch_bounds__(1024, 1) kernel_dispatch_intranode(
    void *x,                             // [num_token, hidden_size]
    int32_t *topk_send_mask,             // [num_token, topk]
    void *topk_weights,                  // [num_token, topk]
    offset_t *topk_indices,              // [num_token, topk]
    offset_t *token_dst_scatter_indices, // [num_token, topk]
    int32_t num_token, int32_t hidden_size, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks,
    // outputs
    void *recv_x_ptrs, // [num_ranks], recv_x [num_recv_token, hidden_size]
    void *
        *recv_weights_ptrs, // [num_ranks], recv_weights [num_recv_token, topk]
    offset_t **
        recv_topk_scatter_indices_ptrs // [num_ranks], recv_topk_scatter_indices
                                       // [num_recv_token, topk]
) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  using smem_t =
      kernels::smem::DispatchIntraNodeSmem<token_t, weight_t, offset_t,
                                           kHiddenSize, kNumStages>;
  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  static_assert(kNumConsumerGroups > 0, "kNumConsumerGroups must be > 0");
  // With multi consumer groups, stages must be partitionable to avoid different
  // groups contending for the same stage+parity and breaking mbarrier arrival
  // counts.
  static_assert(kNumStages % kNumConsumerGroups == 0,
                "kNumStages must be divisible by kNumConsumerGroups");

  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;

  uint64_t *mbar_full_ptr = smem.mbar_full;
  uint64_t *mbar_empty_ptr = smem.mbar_empty;
  token_t **smem_recv_x_ptrs = smem.recv_x_ptrs;
  weight_t **smem_recv_weights_ptrs = smem.recv_weights_ptrs;
  offset_t **smem_recv_topk_scatter_indices_ptrs =
      smem.recv_topk_scatter_indices_ptrs;
  bool is_init_warp = warp_id == 0;
  const bool is_consumer_warp = (warp_id < kTopk * kNumConsumerGroups);
  const bool is_producer_warp = (warp_id == kTopk * kNumConsumerGroups);
  const int32_t consumer_group_id = warp_id / kTopk;
  const int32_t response_expert_idx = warp_id % kTopk;

  int32_t consumer_arr_cnt = kTopk;
  if (is_init_warp && elect_one_sync()) {
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
    }
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_empty_ptr + i, consumer_arr_cnt);
    }
  }

  if (thread_id < num_ranks) {
    smem_recv_x_ptrs[thread_id] =
        reinterpret_cast<token_t **>(recv_x_ptrs)[thread_id];
    smem_recv_weights_ptrs[thread_id] =
        reinterpret_cast<weight_t **>(recv_weights_ptrs)[thread_id];
    smem_recv_topk_scatter_indices_ptrs[thread_id] =
        recv_topk_scatter_indices_ptrs[thread_id];
  }

  auto producer_pipe_state = PipelineState<kNumStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumStages>(0, 0, 0);
  consumer_pipe_state += consumer_group_id;
  __syncthreads();

  int32_t num_bytes_per_token = hidden_size * sizeof(token_t);
  if (is_producer_warp) {
    for (int token_offset = block_id; token_offset < num_token;
         token_offset += num_block) {
      void *src_gmem_ptr =
          reinterpret_cast<token_t *>(x) + token_offset * hidden_size;
      uint64_t *cur_mbar_empty_ptr =
          mbar_empty_ptr + producer_pipe_state.index();
      uint64_t *cur_mbar_full_ptr = mbar_full_ptr + producer_pipe_state.index();

      wait_barrier(cur_mbar_empty_ptr, producer_pipe_state.phase());

      void *dst_smem_ptr = smem.tma_buffer[producer_pipe_state.index()];
      if (elect_one_sync()) {
        tma_copy_1d_g2s(src_gmem_ptr, cur_mbar_full_ptr, dst_smem_ptr,
                        num_bytes_per_token);
        mbar_arrive_and_set_barrier_transaction_bytes(cur_mbar_full_ptr,
                                                      num_bytes_per_token);
      }
      __syncwarp();
      ++producer_pipe_state;
    }
  } else if (is_consumer_warp) {
    // the token set of each consumer group is disjoint, and each token is only
    // consumed by one consumer group. the token set of all consumer groups must
    // be the same as the token set of producer.
    const uint32_t is_leader_lane = elect_one_sync();
    for (int token_offset = block_id + consumer_group_id * num_block;
         token_offset < num_token;
         token_offset += num_block * kNumConsumerGroups) {
      // store to peer
      int32_t expert_idx =
          topk_indices[token_offset * kTopk + response_expert_idx];
      int32_t target_rank = expert_idx / num_experts_per_rank;
      int32_t is_need_send =
          topk_send_mask[token_offset * kTopk + response_expert_idx];

      uint64_t *cur_mbar_empty_ptr =
          mbar_empty_ptr + consumer_pipe_state.index();
      uint64_t *cur_mbar_full_ptr = mbar_full_ptr + consumer_pipe_state.index();

      // all consumer warp need to wait this barrier, otherwise drop token may
      // cause dead lock.
      wait_barrier(cur_mbar_full_ptr, consumer_pipe_state.phase());
      if (target_rank < num_ranks && is_need_send) {
        int32_t store_idx = token_dst_scatter_indices[token_offset * kTopk +
                                                      response_expert_idx];
        void *dst_gmem_ptr =
            reinterpret_cast<token_t *>(smem_recv_x_ptrs[target_rank]) +
            store_idx * hidden_size;

        void *src_smem_ptr = smem.tma_buffer[consumer_pipe_state.index()];
        if (is_leader_lane) {
          tma_copy_1d_s2g(src_smem_ptr, dst_gmem_ptr, num_bytes_per_token);
          tma_store_arrive();
        }

        if (lane_id < kTopk) {
          weight_t *dst_weight_ptr = reinterpret_cast<weight_t *>(
                                         smem_recv_weights_ptrs[target_rank]) +
                                     store_idx * kTopk + lane_id;
          weight_t cur_weight = reinterpret_cast<weight_t *>(
              topk_weights)[token_offset * kTopk + lane_id];
          *dst_weight_ptr = cur_weight;
        } else if (lane_id < 2 * kTopk) {
          offset_t *dst_index_ptr =
              reinterpret_cast<offset_t *>(
                  smem_recv_topk_scatter_indices_ptrs[target_rank]) +
              store_idx * kTopk + lane_id - kTopk;
          offset_t cur_index = reinterpret_cast<offset_t *>(
              token_dst_scatter_indices)[token_offset * kTopk + lane_id -
                                         kTopk];
          offset_t cur_expert_idx = reinterpret_cast<offset_t *>(
              topk_indices)[token_offset * kTopk + lane_id - kTopk];
          cur_index = cur_expert_idx / num_experts_per_rank == target_rank
                          ? cur_index
                          : -1;
          *dst_index_ptr = cur_index;
        }
      }

      tma_store_wait<0>();
      // ensure that the thread which issued the tma_store_wait has completed
      __syncwarp();
      if (is_leader_lane) {
        arrive_barrier(cur_mbar_empty_ptr);
      }
      __syncwarp();
      consumer_pipe_state += kNumConsumerGroups;
    }
  }
}

// kernel_dispatch_intranode_v1 compared to dispatch_intranode:
// 1. each consumer group uses only 1 warp (instead of kTopk warps)
// 2. single warp sends up to kTopk TMA stores per token using ballot/shuffle
// 3. in-flight TMA stores for better performance and less warps
template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kTopk, int32_t kNumStages,
          int32_t kNumConsumerGroups, bool kHasWeight>
void __global__ __launch_bounds__(1024, 1) kernel_dispatch_intranode_v1(
    void *x,                             // [num_token, hidden_size]
    int32_t *topk_send_mask,             // [num_token, topk]
    void *topk_weights,                  // [num_token, topk]
    offset_t *topk_indices,              // [num_token, topk]
    offset_t *token_dst_scatter_indices, // [num_token, topk]
    int32_t num_token, int32_t hidden_size, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks,
    // outputs
    void *recv_x_ptrs, // [num_ranks], recv_x [num_recv_token, hidden_size]
    void *
        *recv_weights_ptrs, // [num_ranks], recv_weights [num_recv_token, topk]
    offset_t **
        recv_topk_scatter_indices_ptrs // [num_ranks], recv_topk_scatter_indices
                                       // [num_recv_token, topk]
) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  using smem_t =
      kernels::smem::DispatchIntraNodeSmem<token_t, weight_t, offset_t,
                                           kHiddenSize, kNumStages>;
  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);

  static_assert(kTopk <= WARP_SIZE,
                "kTopk must be <= WARP_SIZE for warp shuffle");
  static_assert(kNumConsumerGroups > 0, "kNumConsumerGroups must be > 0");
  static_assert(kNumStages % kNumConsumerGroups == 0,
                "kNumStages must be divisible by kNumConsumerGroups");
  static_assert(kTopk <= WARP_SIZE,
                "kTopk must be <= WARP_SIZE for warp shuffle");

  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;
  // TMA store pipeline depth
  constexpr int32_t kStorePipe = 2;

  // 1 producer warp + kNumConsumerGroups consumer warps
  const bool is_producer_warp = (warp_id == 0);
  const bool is_consumer_warp = (warp_id >= 1 && warp_id <= kNumConsumerGroups);
  const int32_t consumer_group_id = warp_id - 1;

  uint64_t *mbar_full_ptr = smem.mbar_full;
  uint64_t *mbar_empty_ptr = smem.mbar_empty;
  token_t **smem_recv_x_ptrs = smem.recv_x_ptrs;
  weight_t **smem_recv_weights_ptrs = smem.recv_weights_ptrs;
  offset_t **smem_recv_topk_scatter_indices_ptrs =
      smem.recv_topk_scatter_indices_ptrs;

  // Initialize barriers (only need warp 0)
  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
      // each stage has 1 consumer warp
      initialize_barrier(mbar_empty_ptr + i, 1);
    }
  }

  if (thread_id < num_ranks) {
    smem_recv_x_ptrs[thread_id] =
        reinterpret_cast<token_t **>(recv_x_ptrs)[thread_id];
    smem_recv_weights_ptrs[thread_id] =
        reinterpret_cast<weight_t **>(recv_weights_ptrs)[thread_id];
    smem_recv_topk_scatter_indices_ptrs[thread_id] =
        recv_topk_scatter_indices_ptrs[thread_id];
  }

  auto producer_pipe_state = PipelineState<kNumStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumStages>(0, 0, 0);
  consumer_pipe_state += consumer_group_id;

  __syncthreads();

  const int32_t num_bytes_per_token = hidden_size * sizeof(token_t);

  if (is_producer_warp) {
    for (int token_offset = block_id; token_offset < num_token;
         token_offset += num_block) {
      void *src_gmem_ptr =
          reinterpret_cast<token_t *>(x) + token_offset * hidden_size;
      uint64_t *cur_mbar_empty_ptr =
          mbar_empty_ptr + producer_pipe_state.index();
      uint64_t *cur_mbar_full_ptr = mbar_full_ptr + producer_pipe_state.index();

      wait_barrier(cur_mbar_empty_ptr, producer_pipe_state.phase());

      void *dst_smem_ptr = smem.tma_buffer[producer_pipe_state.index()];
      if (elect_one_sync()) {
        tma_copy_1d_g2s(src_gmem_ptr, cur_mbar_full_ptr, dst_smem_ptr,
                        num_bytes_per_token);
        mbar_arrive_and_set_barrier_transaction_bytes(cur_mbar_full_ptr,
                                                      num_bytes_per_token);
      }
      __syncwarp();
      ++producer_pipe_state;
    }
  } else if (is_consumer_warp) {
    // consumer_pipe_state: tracks current token being processed (for full
    // barrier) release_pipe_state: tracks which stage's empty barrier to
    // release (for empty barrier)
    auto release_pipe_state = PipelineState<kNumStages>(0, 0, 0);
    release_pipe_state += consumer_group_id;

    int32_t tokens_processed = 0;

    const uint32_t is_leader_lane = elect_one_sync();
    for (int token_offset = block_id + consumer_group_id * num_block;
         token_offset < num_token;
         token_offset += num_block * kNumConsumerGroups) {
      uint64_t *cur_mbar_full_ptr = mbar_full_ptr + consumer_pipe_state.index();

      int32_t my_expert_idx = -1;
      int32_t my_target_rank = -1;
      int32_t my_is_need_send = 0;
      int32_t my_store_idx = -1;
      weight_t my_weight = 0;

      if (lane_id < kTopk) {
        my_expert_idx = topk_indices[token_offset * kTopk + lane_id];
        my_target_rank = my_expert_idx / num_experts_per_rank;
        my_is_need_send = topk_send_mask[token_offset * kTopk + lane_id];
        my_store_idx =
            token_dst_scatter_indices[token_offset * kTopk + lane_id];
        if constexpr (kHasWeight) {
          my_weight = reinterpret_cast<weight_t *>(
              topk_weights)[token_offset * kTopk + lane_id];
        }
      }
      __syncwarp();

      // tma stores from kStorePipe tokens ago to complete
      tma_store_wait<kStorePipe - 1>();
      // ensure that all threads have completed the tma_store_wait
      __syncwarp();

      // consumer release
      if (tokens_processed >= kStorePipe) {
        uint64_t *release_mbar_empty_ptr =
            mbar_empty_ptr + release_pipe_state.index();
        if (is_leader_lane) {
          arrive_barrier(release_mbar_empty_ptr);
        }
        release_pipe_state += kNumConsumerGroups;
      }

      // consumer wait for producer
      wait_barrier(cur_mbar_full_ptr, consumer_pipe_state.phase());

      int32_t should_send = (my_target_rank >= 0 &&
                             my_target_rank < num_ranks && my_is_need_send);
      uint32_t send_mask = __ballot_sync(0xffffffff, should_send);

      void *src_smem_ptr = smem.tma_buffer[consumer_pipe_state.index()];

      uint32_t remaining_mask = send_mask;
      while (remaining_mask) {
        int32_t send_lane = __ffs(remaining_mask) - 1;
        remaining_mask &= (remaining_mask - 1);

        int32_t target_rank =
            __shfl_sync(0xffffffff, my_target_rank, send_lane);
        int32_t store_idx = __shfl_sync(0xffffffff, my_store_idx, send_lane);

        void *dst_gmem_ptr =
            reinterpret_cast<token_t *>(smem_recv_x_ptrs[target_rank]) +
            store_idx * hidden_size;
        if (is_leader_lane) {
          tma_copy_1d_s2g(src_smem_ptr, dst_gmem_ptr, num_bytes_per_token);
        }

        // write recv_topk_weights and recv_topk_scatter_indices
        if (lane_id < kTopk) {
          if constexpr (kHasWeight) {
            weight_t *dst_weight_ptr =
                reinterpret_cast<weight_t *>(
                    smem_recv_weights_ptrs[target_rank]) +
                store_idx * kTopk + lane_id;
            *dst_weight_ptr = my_weight;
          }
          offset_t cur_index = my_store_idx;
          cur_index = (my_target_rank == target_rank) ? cur_index : -1;
          offset_t *dst_index_ptr =
              reinterpret_cast<offset_t *>(
                  smem_recv_topk_scatter_indices_ptrs[target_rank]) +
              store_idx * kTopk + lane_id;
          *dst_index_ptr = cur_index;
        }
        __syncwarp();
      }

      // ensure the thread that issued TMA stores commits tma stores
      tma_store_arrive();

      __syncwarp();
      consumer_pipe_state += kNumConsumerGroups;
      tokens_processed++;
    }
    tma_store_wait<0>();
  }
}

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kTopk, int32_t kNumStages,
          int32_t kNumConsumerGroups, bool kHasWeight,
          int32_t kNumDispatchChunkSize = 128>
void __global__ __launch_bounds__(1024, 1) kernel_dispatch_intranode_chunk(
    void *x,                             // [num_token, hidden_size]
    int32_t *topk_send_mask,             // [num_token, topk]
    void *topk_weights,                  // [num_token, topk]
    offset_t *topk_indices,              // [num_token, topk]
    offset_t *token_dst_scatter_indices, // [num_token, topk]
    int32_t num_token, int32_t hidden_size, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks,
    // outputs
    void *recv_x_ptrs, // [num_ranks], recv_x [num_recv_token, hidden_size]
    void *
        *recv_weights_ptrs, // [num_ranks], recv_weights [num_recv_token, topk]
    offset_t **
        recv_topk_scatter_indices_ptrs // [num_ranks], recv_topk_scatter_indices
                                       // [num_recv_token, topk]
) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  using smem_t =
      kernels::smem::DispatchIntraNodeChunkSmem<token_t, weight_t, offset_t,
                                                kHiddenSize, kNumStages, kTopk,
                                                kNumDispatchChunkSize>;
  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  static_assert(kNumConsumerGroups > 0, "kNumConsumerGroups must be > 0");
  // With multi consumer groups, stages must be partitionable to avoid different
  // groups contending for the same stage+parity and breaking mbarrier arrival
  // counts.
  static_assert(kNumStages % kNumConsumerGroups == 0,
                "kNumStages must be divisible by kNumConsumerGroups");

  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;

  uint64_t *mbar_full_ptr = smem.mbar_full;
  uint64_t *mbar_empty_ptr = smem.mbar_empty;
  token_t **smem_recv_x_ptrs = smem.recv_x_ptrs;
  weight_t **smem_recv_weights_ptrs = smem.recv_weights_ptrs;
  offset_t **smem_recv_topk_scatter_indices_ptrs =
      smem.recv_topk_scatter_indices_ptrs;
  bool is_init_warp = warp_id == 0;
  const bool is_consumer_warp = (warp_id < kTopk * kNumConsumerGroups);
  const bool is_producer_warp = (warp_id == kTopk * kNumConsumerGroups);
  const int32_t consumer_group_id = warp_id / kTopk;
  const int32_t response_expert_idx = warp_id % kTopk;

  int32_t consumer_arr_cnt = kTopk;
  if (is_init_warp and elect_one_sync()) {
    constexpr int32_t kNumConsumerWarps = kTopk * kNumConsumerGroups;
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
    }
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_empty_ptr + i, consumer_arr_cnt);
    }
    // Meta prefetch barriers (ping-pong).
    initialize_barrier(&smem.meta_mbar_full[0], 1);
    initialize_barrier(&smem.meta_mbar_full[1], 1);
    initialize_barrier(&smem.meta_mbar_empty[0], kNumConsumerWarps);
    initialize_barrier(&smem.meta_mbar_empty[1], kNumConsumerWarps);
  }

  if (thread_id < num_ranks) {
    smem_recv_x_ptrs[thread_id] =
        reinterpret_cast<token_t **>(recv_x_ptrs)[thread_id];
    smem_recv_weights_ptrs[thread_id] =
        reinterpret_cast<weight_t **>(recv_weights_ptrs)[thread_id];
    smem_recv_topk_scatter_indices_ptrs[thread_id] =
        recv_topk_scatter_indices_ptrs[thread_id];
  }
  auto producer_pipe_state = PipelineState<kNumStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumStages>(0, 0, 0);
  consumer_pipe_state += consumer_group_id;

  __syncthreads();

  int32_t num_bytes_per_token = hidden_size * sizeof(token_t);
  if (is_producer_warp) {
    // Producer keeps stage-per-token behavior. Token chunking only changes
    // traversal order.
    for (int chunk_base = block_id * kNumDispatchChunkSize;
         chunk_base < num_token;
         chunk_base += num_block * kNumDispatchChunkSize) {
      int chunk_len = num_token - chunk_base;
      if (chunk_len > kNumDispatchChunkSize)
        chunk_len = kNumDispatchChunkSize;
      for (int i = 0; i < chunk_len; ++i) {
        const int token_offset = chunk_base + i;
        void *src_gmem_ptr =
            reinterpret_cast<token_t *>(x) + token_offset * hidden_size;
        uint64_t *cur_mbar_empty_ptr =
            mbar_empty_ptr + producer_pipe_state.index();
        uint64_t *cur_mbar_full_ptr =
            mbar_full_ptr + producer_pipe_state.index();

        wait_barrier(cur_mbar_empty_ptr, producer_pipe_state.phase());

        void *dst_smem_ptr = smem.tma_buffer[producer_pipe_state.index()];
        if (elect_one_sync()) {
          mbar_arrive_and_set_barrier_transaction_bytes(cur_mbar_full_ptr,
                                                        num_bytes_per_token);
          tma_copy_1d_g2s(src_gmem_ptr, cur_mbar_full_ptr, dst_smem_ptr,
                          num_bytes_per_token);
        }
        __syncwarp();
        ++producer_pipe_state;
      }
    }
  } else if (is_consumer_warp) {
    uint32_t meta_phase[2] = {0u, 0u};
    uint32_t meta_empty_phase[2] = {1u, 1u};
    int buf = 0;

    int chunk_base = block_id * kNumDispatchChunkSize;
    int chunk_len = num_token - chunk_base;
    if (chunk_len > kNumDispatchChunkSize)
      chunk_len = kNumDispatchChunkSize;
    if (chunk_base < num_token) {
      wait_barrier(&smem.meta_mbar_empty[buf], meta_empty_phase[buf]);
      meta_empty_phase[buf] ^= 1u;
      issue_meta_prefetch_one_thread<weight_t, offset_t, kTopk,
                                     kNumDispatchChunkSize, smem_t, kHasWeight>(
          smem, warp_id, lane_id, topk_send_mask, topk_weights, topk_indices,
          token_dst_scatter_indices, buf, chunk_base, chunk_len);
    }

    for (; chunk_base < num_token;
         chunk_base += num_block * kNumDispatchChunkSize) {
      // consumer wait for meta buffer
      wait_barrier(&smem.meta_mbar_full[buf], meta_phase[buf]);
      meta_phase[buf] ^= 1u;

      // prefetch next chunk's meta
      const int next_chunk_base =
          chunk_base + num_block * kNumDispatchChunkSize;
      int next_chunk_len = num_token - next_chunk_base;
      if (next_chunk_len > kNumDispatchChunkSize)
        next_chunk_len = kNumDispatchChunkSize;
      const int next_buf = buf ^ 1;
      if (next_chunk_base < num_token) {
        // producer acquire meta buffer
        wait_barrier(&smem.meta_mbar_empty[next_buf],
                     meta_empty_phase[next_buf]);
        meta_empty_phase[next_buf] ^= 1u;
        issue_meta_prefetch_one_thread<weight_t, offset_t, kTopk,
                                       kNumDispatchChunkSize, smem_t,
                                       kHasWeight>(
            smem, warp_id, lane_id, topk_send_mask, topk_weights, topk_indices,
            token_dst_scatter_indices, next_buf, next_chunk_base,
            next_chunk_len);
      }

      const uint32_t is_leader_lane = elect_one_sync();
      // each consumer group takes a disjoint subset of tokens within the chunk
      // via round-robin.
      for (int i = consumer_group_id; i < chunk_len; i += kNumConsumerGroups) {
        const int32_t expert_idx =
            smem.meta_topk_indices[buf][i][response_expert_idx];
        const int32_t target_rank = expert_idx / num_experts_per_rank;
        const int32_t is_need_send =
            smem.meta_topk_send_mask[buf][i][response_expert_idx];

        uint64_t *cur_mbar_empty_ptr =
            mbar_empty_ptr + consumer_pipe_state.index();
        uint64_t *cur_mbar_full_ptr =
            mbar_full_ptr + consumer_pipe_state.index();

        // all consumer warp need to wait this barrrier, otherwise drop token
        // may cause dead lock.
        wait_barrier(cur_mbar_full_ptr, consumer_pipe_state.phase());
        if (target_rank < num_ranks and is_need_send) {
          const int32_t store_idx =
              smem.meta_token_dst_scatter_indices[buf][i][response_expert_idx];
          void *dst_gmem_ptr =
              reinterpret_cast<token_t *>(smem_recv_x_ptrs[target_rank]) +
              store_idx * hidden_size;

          void *src_smem_ptr = smem.tma_buffer[consumer_pipe_state.index()];
          if (is_leader_lane) {
            tma_copy_1d_s2g(src_smem_ptr, dst_gmem_ptr, num_bytes_per_token);
            tma_store_arrive();
          }

          if (lane_id < kTopk) {
            if constexpr (kHasWeight) {
              weight_t *dst_weight_ptr =
                  reinterpret_cast<weight_t *>(
                      smem_recv_weights_ptrs[target_rank]) +
                  store_idx * kTopk + lane_id;
              weight_t cur_weight = smem.meta_topk_weights[buf][i][lane_id];
              *dst_weight_ptr = cur_weight;
            }
          } else if (lane_id < 2 * kTopk) {
            // TODO:some token may be not belong to target rank, so we need to
            // set the index to a invalid value.
            offset_t *dst_index_ptr =
                reinterpret_cast<offset_t *>(
                    smem_recv_topk_scatter_indices_ptrs[target_rank]) +
                store_idx * kTopk + lane_id - kTopk;

            offset_t cur_index =
                smem.meta_token_dst_scatter_indices[buf][i][lane_id - kTopk];
            offset_t cur_expert_idx =
                smem.meta_topk_indices[buf][i][lane_id - kTopk];
            cur_index = cur_expert_idx / num_experts_per_rank == target_rank
                            ? cur_index
                            : -1;
            *dst_index_ptr = cur_index;
          }
        }

        __syncwarp();
        tma_store_wait<0>();
        __syncwarp();
        if (is_leader_lane) {
          arrive_barrier(cur_mbar_empty_ptr);
        }
        __syncwarp();
        consumer_pipe_state += kNumConsumerGroups;
      }

      // consumer release for meta buffer
      // lane 0 issue meta tma load
      if (lane_id == 0) {
        arrive_barrier(&smem.meta_mbar_empty[buf]);
      }
      __syncwarp();

      buf = next_buf;
      chunk_len = next_chunk_len;
    }
  }
}

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kNumStages, int32_t kWritePipeCount,
          bool kHasWeight>
__global__ void kernel_dispatch_postprocess_tma(
    void *recv_x,                // [num_recv_worst_token, hidden_size]
    weight_t *recv_topk_weights, // [num_recv_worst_token, topk]   (not used,
                                 // but for complete signature)
    offset_t
        *recv_topk_scatter_indices_comm_buffer, // [num_recv_worst_token, topk]
    int32_t *recv_token_count,                  // [num_ranks], int32
    weight_t *dispatch_weights,          // [num_recv_worst_token], float32
    offset_t *recv_topk_scatter_indices, // [num_recv_worst_token, topk],
                                         // int32/offset_t
    int32_t hidden_size, int32_t topk, int32_t rank, int32_t num_ranks) {
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;
  const int32_t num_recv_token = recv_token_count[rank];

  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  using smem_t =
      kernels::smem::DispatchPostprocessSmem<token_t, kHiddenSize, kNumStages>;
  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  const int32_t bytes_per_token = hidden_size * sizeof(token_t);
  uint64_t *mbar_full_ptr = smem.mbar_full;
  uint64_t *mbar_empty_ptr = smem.mbar_empty;

  if (warp_id == 0 and elect_one_sync()) {
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
    }
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_empty_ptr + i, 1);
    }
  }
  auto producer_pipe_state = PipelineState<kNumStages>(0, 1, 0);
  auto consumer_read_pipe_state = PipelineState<kNumStages>(0, 0, 0);
  auto consumer_write_pipe_state = PipelineState<kNumStages>(0, 0, 0);
  __syncthreads();

  if (warp_id == 0) {
    for (int32_t token_offset = block_id; token_offset < num_recv_token;
         token_offset += num_block) {
      for (int32_t j = 0; j < topk; ++j) {
        int32_t dst_idx =
            recv_topk_scatter_indices_comm_buffer[token_offset * topk + j];
        if (dst_idx != -1 && dst_idx != token_offset) {
          token_t *src_ptr =
              reinterpret_cast<token_t *>(recv_x) + token_offset * hidden_size;
          token_t *dst_smem_ptr = smem.tma_buffer[producer_pipe_state.index()];
          // producer acquire
          uint64_t *cur_mbar_empty_ptr =
              mbar_empty_ptr + producer_pipe_state.index();
          uint64_t *cur_mbar_full_ptr =
              mbar_full_ptr + producer_pipe_state.index();
          wait_barrier(cur_mbar_empty_ptr, producer_pipe_state.phase());

          if (elect_one_sync()) {
            mbar_arrive_and_set_barrier_transaction_bytes(cur_mbar_full_ptr,
                                                          bytes_per_token);
            tma_copy_1d_g2s(src_ptr, cur_mbar_full_ptr, dst_smem_ptr,
                            bytes_per_token);
          }
          __syncwarp();
          ++producer_pipe_state;
        }
      }
    }
  } else if (warp_id == 1) {
    const uint32_t is_leader_lane = elect_one_sync();
    for (int32_t token_offset = block_id; token_offset < num_recv_token;
         token_offset += num_block) {
      for (int32_t j = 0; j < topk; ++j) {
        int32_t dst_idx =
            recv_topk_scatter_indices_comm_buffer[token_offset * topk + j];
        if (dst_idx != -1 && dst_idx != token_offset) {
          token_t *src_smem_ptr =
              smem.tma_buffer[consumer_read_pipe_state.index()];
          token_t *dst_ptr =
              reinterpret_cast<token_t *>(recv_x) + dst_idx * hidden_size;
          // consumer wait
          uint64_t *cur_mbar_full_ptr =
              mbar_full_ptr + consumer_read_pipe_state.index();
          wait_barrier(cur_mbar_full_ptr, consumer_read_pipe_state.phase());
          if (is_leader_lane) {
            tma_copy_1d_s2g(src_smem_ptr, dst_ptr, bytes_per_token);
            tma_store_arrive();
          }
          tma_store_wait<kWritePipeCount>();
          __syncwarp();
          // consumer release
          if (consumer_read_pipe_state.count() >= kWritePipeCount) {
            uint64_t *cur_mbar_empty_ptr =
                mbar_empty_ptr + consumer_write_pipe_state.index();
            if (is_leader_lane) {
              arrive_barrier(cur_mbar_empty_ptr);
            }
            ++consumer_write_pipe_state;
          }
          ++consumer_read_pipe_state;
        }
        __syncwarp();
        if (lane_id == 0) {
          // reset the scatter index to -1
          recv_topk_scatter_indices_comm_buffer[token_offset * topk + j] = -1;
          recv_topk_scatter_indices[token_offset * topk + j] = dst_idx;
          if constexpr (kHasWeight) {
            // scatter the weight to dispatch_weights
            weight_t weight = recv_topk_weights[token_offset * topk + j];
            if (dst_idx != -1) {
              dispatch_weights[dst_idx] = weight;
            }
          }
        }
        __syncwarp();
      }
    }
  }
  tma_store_wait<0>();
  __syncthreads();
}

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kNumWarps, int32_t kElemsPerThread,
          bool kHasWeight = false>
__global__ void kernel_combine_preprocess_inplace(
    void *x,      // [num_recv_worst_token, hidden_size]
    void *weight, // [num_recv_worst_token], float32 (aligned with x rows)
    int32_t *recv_token_count,           // [num_ranks], int32
    offset_t *recv_topk_scatter_indices, // [num_recv_worst_token, topk]
    void *recv_topk_weight, // [num_recv_worst_token, topk], float32
    int32_t topk, int32_t rank, int32_t num_ranks) {
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int lane_id = thread_id % WARP_SIZE;
  // const int warp_id = thread_id / WARP_SIZE;
  constexpr int32_t kElemsPerInt4 = sizeof(int4) / sizeof(token_t);
  constexpr int32_t kNumThreadsPerBlock = kNumWarps * WARP_SIZE;

  constexpr int32_t kHiddenSizeInt4 = kHiddenSize / kElemsPerInt4;
  constexpr int32_t kInt4PerThread =
      (kHiddenSizeInt4 + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock;

  float acc[kElemsPerThread];
  static_assert((kHiddenSizeInt4 + kNumThreadsPerBlock - 1) /
                        kNumThreadsPerBlock <=
                    kElemsPerThread / kElemsPerInt4,
                "kElemsPerThread too small");

  PRAGMA_UNROLL
  for (int32_t i = 0; i < kElemsPerThread; ++i) {
    acc[i] = 0.0f;
  }

  __syncthreads();
  int32_t num_recv_token = recv_token_count[rank];

  for (int32_t i = block_id; i < num_recv_token; i += num_block) {
    int32_t is_valid_lane = 0;
    if (lane_id < topk) {
      int32_t dst_idx = recv_topk_scatter_indices[i * topk + lane_id];
      is_valid_lane = (dst_idx != -1);
    }

    __syncwarp();
    uint32_t valid_mask = __ballot_sync(0xffffffff, is_valid_lane);
    int32_t reduce_cnt = __popc(valid_mask);

    if constexpr (kHasWeight) {
      if (lane_id < topk) {
        int32_t dst_idx = recv_topk_scatter_indices[i * topk + lane_id];
        if (dst_idx != -1) {
          weight_t val = reinterpret_cast<weight_t *>(weight)[dst_idx];
          reinterpret_cast<weight_t *>(
              recv_topk_weight)[dst_idx * topk + lane_id] = val;
        }
      }
    }

    if (reduce_cnt > 1) {
      static_assert(kElemsPerInt4 % 2 == 0, "kElemsPerInt4 must be even");
      for (int32_t j = 0; j < topk; ++j) {
        int32_t dst_idx = recv_topk_scatter_indices[i * topk + j];
        if (dst_idx != -1) {
          token_t *src_ptr =
              reinterpret_cast<token_t *>(x) + dst_idx * kHiddenSize;

          PRAGMA_UNROLL
          for (int32_t idx = 0; idx < kInt4PerThread; ++idx) {
            if (idx * kNumThreadsPerBlock + thread_id < kHiddenSizeInt4) {
              int4 data = reinterpret_cast<int4 *>(
                  src_ptr)[idx * kNumThreadsPerBlock + thread_id];
              union {
                int4 vec;
                __nv_bfloat162 bf162[4];
              } converter;
              converter.vec = data;
              PRAGMA_UNROLL
              for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
                float2 v = __bfloat1622float2(converter.bf162[k]);
                acc[idx * kElemsPerInt4 + k * 2] += v.x;
                acc[idx * kElemsPerInt4 + k * 2 + 1] += v.y;
              }
            }
          }
        }
      }

      token_t *dst_ptr = reinterpret_cast<token_t *>(x) + i * kHiddenSize;
      PRAGMA_UNROLL
      for (int32_t idx = 0; idx < kInt4PerThread; ++idx) {
        if (idx * kNumThreadsPerBlock + thread_id < kHiddenSizeInt4) {
          union {
            int4 vec;
            __nv_bfloat162 bf162[4];
          } converter;
          PRAGMA_UNROLL
          for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
            converter.bf162[k] = __float22bfloat162_rn(
                make_float2(acc[idx * kElemsPerInt4 + k * 2],
                            acc[idx * kElemsPerInt4 + k * 2 + 1]));
            // clear the accumulator
            acc[idx * kElemsPerInt4 + k * 2] = 0.0f;
            acc[idx * kElemsPerInt4 + k * 2 + 1] = 0.0f;
          }
          reinterpret_cast<int4 *>(
              dst_ptr)[idx * kNumThreadsPerBlock + thread_id] = converter.vec;
        }
      }
    }
  }
}

template <typename token_t, typename offset_t, int32_t kTopk,
          int32_t kHiddenSize, int32_t kNumStages, int32_t kNumWarps,
          int32_t kWarpsPerWG, int32_t kElemsPerThread>
void __global__ __launch_bounds__(1024, 1) kernel_combine_intranode(
    void *x_ptrs,                        // [num_ranks]
    offset_t *topk_send_mask,            // [num_token, topk]
    offset_t *topk_indices,              // [num_token, topk]
    offset_t *token_dst_scatter_indices, // [num_token, topk]
    void *recv_x,                        // [num_token, hidden_size]
    int32_t num_token, int32_t num_experts_per_rank, int32_t rank,
    int32_t num_ranks) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  constexpr int32_t kNumWGPerBlock = kNumWarps / kWarpsPerWG;
  constexpr int32_t kElemsPerInt4 = sizeof(int4) / sizeof(token_t);

  using smem_t = smem::CombineIntraNodeSmem<token_t, float, kHiddenSize,
                                            kNumStages, 0, kNumWGPerBlock>;

  const int32_t warp_group_id = warp_id / kWarpsPerWG;
  const int32_t global_warp_group_id =
      block_id * kNumWGPerBlock + warp_group_id;
  const int32_t total_warp_groups = num_block * kNumWGPerBlock;
  const int32_t num_bytes_per_token = kHiddenSize * sizeof(token_t);
  constexpr int32_t kHiddenSizeInt4 = kHiddenSize / kElemsPerInt4;

  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  auto &wg_smem = smem.warp_group_smem[warp_group_id];
  uint64_t *mbar_full_ptr = wg_smem.mbar_full;
  uint64_t *mbar_empty_ptr = wg_smem.mbar_empty;
  token_t **symm_buffer_ptrs = smem.x_ptrs;
  // TODO: store use tma/tma reduce

  // the first warp in each warpgroup as leader to initialize the barrier
  bool is_init_warp = (warp_id % kWarpsPerWG == 0);
  int32_t consumer_arr_cnt = (kWarpsPerWG - 1);
  if (is_init_warp and elect_one_sync()) {
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
    }
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_empty_ptr + i, consumer_arr_cnt);
    }
  }

  if (thread_id < num_ranks) {
    symm_buffer_ptrs[thread_id] =
        reinterpret_cast<token_t **>(x_ptrs)[thread_id];
  }

  auto producer_pipe_state = PipelineState<kNumStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumStages>(0, 0, 0);

  __syncthreads();

  float acc[kElemsPerThread];
  PRAGMA_UNROLL
  for (int32_t i = 0; i < kElemsPerThread; ++i) {
    acc[i] = 0;
  }

  bool is_tma_load_warp = warp_id % kWarpsPerWG == 0;
  constexpr int32_t kNumConsumerThreadsPerWG = (kWarpsPerWG - 1) * WARP_SIZE;
  const int32_t consumer_tid_in_wg =
      thread_id % (WARP_SIZE * kWarpsPerWG) - WARP_SIZE;
  const uint32_t is_leader_lane = elect_one_sync();
  for (int token_offset = global_warp_group_id; token_offset < num_token;
       token_offset += total_warp_groups) {
    for (int32_t j = 0; j < kTopk; ++j) {
      int32_t expert_idx = topk_indices[token_offset * kTopk + j];
      int32_t expert_rank = expert_idx / num_experts_per_rank;
      int32_t is_need_send = topk_send_mask[token_offset * kTopk + j];

      if (expert_rank < num_ranks and is_need_send) {
        // produer issue tma load
        if (is_tma_load_warp) {
          uint64_t *cur_mbar_empty_ptr =
              mbar_empty_ptr + producer_pipe_state.index();
          uint64_t *cur_mbar_full_ptr =
              mbar_full_ptr + producer_pipe_state.index();
          // producer acquire
          wait_barrier(cur_mbar_empty_ptr, producer_pipe_state.phase());
          void *src_gmem_ptr =
              reinterpret_cast<token_t *>(symm_buffer_ptrs[expert_rank]) +
              token_dst_scatter_indices[token_offset * kTopk + j] * kHiddenSize;
          void *smem_ptr = wg_smem.tma_load_buffer[producer_pipe_state.index()];
          if (is_leader_lane) {
            mbar_arrive_and_set_barrier_transaction_bytes(cur_mbar_full_ptr,
                                                          num_bytes_per_token);
            tma_copy_1d_g2s(src_gmem_ptr, cur_mbar_full_ptr, smem_ptr,
                            num_bytes_per_token);
          }
          ++producer_pipe_state;
          __syncwarp();
        } else {
          // consumer perform reduction
          uint64_t *cur_mbar_empty_ptr =
              mbar_empty_ptr + consumer_pipe_state.index();
          uint64_t *cur_mbar_full_ptr =
              mbar_full_ptr + consumer_pipe_state.index();

          // consumer wait
          wait_barrier(cur_mbar_full_ptr, consumer_pipe_state.phase());
          token_t *smem_ptr =
              wg_smem.tma_load_buffer[consumer_pipe_state.index()];
          int4 *smem_ptr_int4 = reinterpret_cast<int4 *>(smem_ptr);

          for (int32_t smem_idx = consumer_tid_in_wg, reg_idx = 0;
               smem_idx < kHiddenSizeInt4;
               smem_idx += kNumConsumerThreadsPerWG, reg_idx += kElemsPerInt4) {
            // acc[reg_idx] += static_cast<float>(smem_ptr[smem_idx]);
            int4 data = smem_ptr_int4[smem_idx];
            union {
              int4 vec;
              __nv_bfloat162 bf162[4];
              __nv_bfloat16 bf16[8];
            } converter;

            converter.vec = data;
            // for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
            //   float2 data_fp32_vec2 = __bfloat1622float2(converter.bf162[k]);
            //   acc[reg_idx + k * 2] += data_fp32_vec2.x;
            //   acc[reg_idx + k * 2 + 1] += data_fp32_vec2.y;
            // }
            for (int32_t k = 0; k < kElemsPerInt4; ++k) {
              acc[reg_idx + k] += __bfloat162float(converter.bf16[k]);
            }
          }
          __syncwarp();

          // consumer release
          if (is_leader_lane) {
            arrive_barrier(cur_mbar_empty_ptr);
          }
          __syncwarp();
          ++consumer_pipe_state;
        }
      }
    }

    // consumer store to gmem
    if (!is_tma_load_warp) {
      token_t *dst_gmem_ptr =
          reinterpret_cast<token_t *>(recv_x) + token_offset * kHiddenSize;

      int4 *dst_gmem_ptr_int4 = reinterpret_cast<int4 *>(dst_gmem_ptr);
      PRAGMA_UNROLL
      for (int32_t gmem_idx = consumer_tid_in_wg, reg_idx = 0;
           gmem_idx < kHiddenSizeInt4;
           gmem_idx += kNumConsumerThreadsPerWG, reg_idx += kElemsPerInt4) {
        // acc[reg_idx] += static_cast<float>(smem_ptr[smem_idx]);
        union {
          int4 vec;
          __nv_bfloat162 bf162[4];
        } converter_dst;

        union {
          float val[2];
          float2 fp32_vec2;
        } converter_ori;
        for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
          converter_ori.val[0] = acc[reg_idx + k * 2];
          converter_ori.val[1] = acc[reg_idx + k * 2 + 1];
          converter_dst.bf162[k] =
              __float22bfloat162_rn(converter_ori.fp32_vec2);
        }
        dst_gmem_ptr_int4[gmem_idx] = converter_dst.vec;
      }

      PRAGMA_UNROLL
      for (int j = 0; j < kElemsPerThread; ++j) {
        acc[j] = 0;
      }
    }
  }
}

template <typename token_t, typename weight_t, typename offset_t, int32_t kTopk,
          int32_t kHiddenSize, int32_t kNumLoadStages, int32_t kNumStoreStages,
          int32_t kNumWarps, int32_t kWarpsPerWG, int32_t kElemsPerThread>
void __global__ __launch_bounds__(1024, 1) kernel_combine_intranode_v1(
    void *x_ptrs,                        // [num_ranks]
    offset_t *topk_send_mask,            // [num_token, topk]
    offset_t *topk_indices,              // [num_token, topk]
    offset_t *token_dst_scatter_indices, // [num_token, topk]
    void *recv_x,                        // [num_token, hidden_size]
    int32_t num_token, int32_t num_experts_per_rank, int32_t rank,
    int32_t num_ranks) {
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;
  constexpr int32_t kNumWGPerBlock = kNumWarps / kWarpsPerWG;
  constexpr int32_t kElemsPerInt4 = sizeof(int4) / sizeof(token_t);
  using smem_t =
      smem::CombineIntraNodeSmem<token_t, weight_t, kHiddenSize, kNumLoadStages,
                                 kNumStoreStages, kNumWGPerBlock>;

  const int32_t warp_group_id = warp_id / kWarpsPerWG;
  const int32_t global_warp_group_id =
      block_id * kNumWGPerBlock + warp_group_id;
  const int32_t total_warp_groups = num_block * kNumWGPerBlock;
  const int32_t num_bytes_per_token = kHiddenSize * sizeof(token_t);
  constexpr int32_t kHiddenSizeInt4 = kHiddenSize / kElemsPerInt4;

  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  auto &wg_smem = smem.warp_group_smem[warp_group_id];
  uint64_t *mbar_full_ptr = wg_smem.mbar_full;
  uint64_t *mbar_empty_ptr = wg_smem.mbar_empty;

  token_t **symm_buffer_ptrs = smem.x_ptrs;

  // the first warp in each warpgroup as leader to initialize the barrier
  bool is_init_warp = (warp_id % kWarpsPerWG == 0);
  int32_t consumer_arr_cnt = (kWarpsPerWG - 1);
  if (is_init_warp and elect_one_sync()) {
    for (int32_t i = 0; i < kNumLoadStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
    }
    for (int32_t i = 0; i < kNumLoadStages; ++i) {
      initialize_barrier(mbar_empty_ptr + i, consumer_arr_cnt);
    }
  }

  if (thread_id < num_ranks) {
    symm_buffer_ptrs[thread_id] =
        reinterpret_cast<token_t **>(x_ptrs)[thread_id];
  }

  auto producer_pipe_state = PipelineState<kNumLoadStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumLoadStages>(0, 0, 0);

  __syncthreads();

  float acc[kElemsPerThread];
  PRAGMA_UNROLL
  for (int32_t i = 0; i < kElemsPerThread; ++i) {
    acc[i] = 0;
  }

  int32_t store_buffer_idx = 0;
  bool is_tma_load_warp = warp_id % kWarpsPerWG == 0;
  constexpr int32_t kNumConsumerThreadsPerWG = (kWarpsPerWG - 1) * WARP_SIZE;
  const int32_t consumer_tid_in_wg =
      thread_id % (WARP_SIZE * kWarpsPerWG) - WARP_SIZE;
  if (is_tma_load_warp) {
    for (int token_offset = global_warp_group_id; token_offset < num_token;
         token_offset += total_warp_groups) {

      int32_t expert_rank_lane = 0;
      int32_t scatter_idx_lane = 0;
      int32_t is_valid_lane = 0;

      static_assert(kTopk <= WARP_SIZE,
                    "kTopk must be less than or equal to WARP_SIZE");
      if (lane_id < kTopk) {
        int32_t expert_idx = topk_indices[token_offset * kTopk + lane_id];
        expert_rank_lane = expert_idx / num_experts_per_rank;
        int32_t is_need_send = topk_send_mask[token_offset * kTopk + lane_id];
        scatter_idx_lane =
            token_dst_scatter_indices[token_offset * kTopk + lane_id];
        is_valid_lane = expert_rank_lane < num_ranks and is_need_send;
      }

      __syncwarp();

      // bit i==1 means lane i has a valid token
      uint32_t valid_mask = __ballot_sync(0xffffffff, is_valid_lane);

      while (valid_mask) {
        int32_t j = __ffs(valid_mask) - 1; // next valid lane
        valid_mask &= (valid_mask - 1);    // clear lowest set bit

        int32_t expert_rank = __shfl_sync(0xffffffff, expert_rank_lane, j);
        int32_t scatter_idx = __shfl_sync(0xffffffff, scatter_idx_lane, j);

        uint64_t *empty = mbar_empty_ptr + producer_pipe_state.index();
        uint64_t *full = mbar_full_ptr + producer_pipe_state.index();

        wait_barrier(empty, producer_pipe_state.phase());

        void *src_gmem_ptr = (void *)(reinterpret_cast<token_t *>(
                                          symm_buffer_ptrs[expert_rank]) +
                                      scatter_idx * kHiddenSize);
        void *dst_smem_ptr =
            (void *)wg_smem.tma_load_buffer[producer_pipe_state.index()];

        if (elect_one_sync()) {
          mbar_arrive_and_set_barrier_transaction_bytes(full,
                                                        num_bytes_per_token);
          tma_copy_1d_g2s(src_gmem_ptr, full, dst_smem_ptr,
                          num_bytes_per_token);
        }

        ++producer_pipe_state;
        __syncwarp();
      }
    }
  } else {

    union {
      int4 vec;
      __nv_bfloat162 bf162[4];
    } converter;

    static_assert(kElemsPerInt4 % 2 == 0, "kElemsPerInt4 must be even");
    const uint32_t is_leader_lane = elect_one_sync();
    int32_t token_iter = 0;
    for (int token_offset = global_warp_group_id; token_offset < num_token;
         token_offset += total_warp_groups, ++token_iter) {
      int32_t is_valid_lane = 0;

      static_assert(kTopk <= WARP_SIZE,
                    "kTopk must be less than or equal to WARP_SIZE");
      if (lane_id < kTopk) {
        int32_t expert_idx = topk_indices[token_offset * kTopk + lane_id];
        int32_t expert_rank = expert_idx / num_experts_per_rank;
        int32_t is_need_send = topk_send_mask[token_offset * kTopk + lane_id];
        is_valid_lane = (expert_rank < num_ranks) && is_need_send;
      }

      uint32_t valid_mask = __ballot_sync(0xffffffff, is_valid_lane);
      int32_t valid_count = __popc(valid_mask);

      for (int32_t j = 0; j < valid_count; ++j) {
        uint64_t *empty = mbar_empty_ptr + consumer_pipe_state.index();
        uint64_t *full = mbar_full_ptr + consumer_pipe_state.index();

        // consumer wait
        wait_barrier(full, consumer_pipe_state.phase());

        // reduce from the current stage buffer
        token_t *smem_ptr = reinterpret_cast<token_t *>(
            wg_smem.tma_load_buffer[consumer_pipe_state.index()]);
        int4 *smem_ptr_int4 = reinterpret_cast<int4 *>(smem_ptr);

        constexpr int32_t kInt4PerThread =
            (kHiddenSizeInt4 + kNumConsumerThreadsPerWG - 1) /
            kNumConsumerThreadsPerWG;
        static_assert(kElemsPerInt4 % 2 == 0, "kElemsPerInt4 must be even");
        static_assert(kInt4PerThread <= kElemsPerThread / kElemsPerInt4,
                      "kElemsPerThread too small");
        PRAGMA_UNROLL
        for (int32_t idx = 0; idx < kInt4PerThread; ++idx) {
          int32_t smem_idx =
              consumer_tid_in_wg + idx * kNumConsumerThreadsPerWG;
          if (smem_idx < kHiddenSizeInt4) {
            int4 data = smem_ptr_int4[smem_idx];
            converter.vec = data;
            PRAGMA_UNROLL
            for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
              float2 fp32_vec2 = __bfloat1622float2(converter.bf162[k]);
              acc[idx * kElemsPerInt4 + k * 2] += fp32_vec2.x;
              acc[idx * kElemsPerInt4 + k * 2 + 1] += fp32_vec2.y;
            }
          }
        }
        __syncwarp();

        // consumer release
        if (is_leader_lane) {
          arrive_barrier(empty);
        }
        __syncwarp();
        ++consumer_pipe_state;
      }

      // consumer store to gmem
      token_t *dst_gmem_ptr =
          reinterpret_cast<token_t *>(recv_x) + token_offset * kHiddenSize;

      // ensure the store staging buffer we're about to overwrite is no longer
      // being read by a prior TMA store.
      int32_t warp_id_in_consumer_wg = consumer_tid_in_wg / WARP_SIZE;
      if (warp_id_in_consumer_wg == 0) {
        if (token_iter >= kNumStoreStages) {
          tma_store_wait<kNumStoreStages - 1>();
        }
      }
      named_barrier_arrive_and_wait(kNumConsumerThreadsPerWG,
                                    warp_group_id + 1);

      int4 *tma_store_ptr_int4 =
          reinterpret_cast<int4 *>(wg_smem.tma_store_buffer[store_buffer_idx]);
      constexpr int32_t kInt4PerThreadStore =
          (kHiddenSizeInt4 + kNumConsumerThreadsPerWG - 1) /
          kNumConsumerThreadsPerWG;
      static_assert(kElemsPerInt4 % 2 == 0, "kElemsPerInt4 must be even");
      static_assert(kInt4PerThreadStore <= kElemsPerThread / kElemsPerInt4,
                    "kElemsPerThread too small");
      PRAGMA_UNROLL
      for (int32_t idx = 0; idx < kInt4PerThreadStore; ++idx) {
        int32_t dst_idx = consumer_tid_in_wg + idx * kNumConsumerThreadsPerWG;
        if (dst_idx < kHiddenSizeInt4) {
          PRAGMA_UNROLL
          for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
            converter.bf162[k] = __float22bfloat162_rn(
                make_float2(acc[idx * kElemsPerInt4 + k * 2],
                            acc[idx * kElemsPerInt4 + k * 2 + 1]));
          }
          tma_store_ptr_int4[dst_idx] = converter.vec;
        }
      }

      PRAGMA_UNROLL
      for (int32_t j = 0; j < kElemsPerThread; ++j) {
        acc[j] = 0;
      }

      fence_async_shared();
      named_barrier_arrive_and_wait(kNumConsumerThreadsPerWG,
                                    warp_group_id + 1);
      if (warp_id_in_consumer_wg == 0) {
        if (is_leader_lane) {
          tma_copy_1d_s2g(reinterpret_cast<void *>(tma_store_ptr_int4),
                          dst_gmem_ptr, num_bytes_per_token);
          tma_store_arrive();
        }
        __syncwarp();
      }
      store_buffer_idx = (store_buffer_idx + 1) % kNumStoreStages;
    }
  }

  tma_store_wait<0>();
  __syncthreads();
}

template <typename token_t, typename weight_t, typename offset_t, int32_t kTopk,
          int32_t kHiddenSize, int32_t kNumLoadStages, int32_t kNumStoreStages,
          int32_t kNumWarps, int32_t kWarpsPerWG, int32_t kElemsPerThread,
          bool kHasWeight = false>
void __global__ __launch_bounds__(768, 1) kernel_combine_intranode_v2(
    void *x_ptrs,                        // [num_ranks]
    void *weight_ptrs,                   // [num_ranks]
    offset_t *topk_send_mask,            // [num_token, topk]
    offset_t *topk_indices,              // [num_token, topk]
    offset_t *token_dst_scatter_indices, // [num_token, topk]
    void *recv_x,                        // [num_token, hidden_size]
    void *recv_weight,                   // [num_token, topk]
    int32_t num_token, int32_t num_experts_per_rank, int32_t rank,
    int32_t num_ranks) {
  static_assert(kWarpsPerWG > 1, "kWarpsPerWG must be greater than 1");
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;
  constexpr int32_t kNumWGPerBlock =
      kHasWeight ? (kNumWarps - 1) / kWarpsPerWG : kNumWarps / kWarpsPerWG;
  constexpr int32_t kElemsPerInt4 = sizeof(int4) / sizeof(token_t);
  using smem_t =
      smem::CombineIntraNodeSmem<token_t, weight_t, kHiddenSize, kNumLoadStages,
                                 kNumStoreStages, kNumWGPerBlock>;

  // When kHasWeight is true, the last warp is a dedicated weight-combine warp
  // that does not belong to any warp group. Its warp_group_id would be
  // kNumWGPerBlock (out of bounds for warp_group_smem[kNumWGPerBlock]).
  // Clamp to a valid index so we never form an out-of-bounds reference.
  // The weight warp never enters the TMA-load / consumer paths, so the
  // particular wg_smem / mbar pointers it gets are never dereferenced.
  const int32_t warp_group_id = warp_id / kWarpsPerWG;
  const int32_t safe_wg_id = (warp_group_id < kNumWGPerBlock)
                                  ? warp_group_id
                                  : (kNumWGPerBlock - 1);
  const int32_t global_warp_group_id =
      block_id * kNumWGPerBlock + warp_group_id;
  const int32_t total_warp_groups = num_block * kNumWGPerBlock;
  const int32_t num_bytes_per_token = kHiddenSize * sizeof(token_t);
  constexpr int32_t kHiddenSizeInt4 = kHiddenSize / kElemsPerInt4;

  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  auto &wg_smem = smem.warp_group_smem[safe_wg_id];
  uint64_t *mbar_full_ptr = wg_smem.mbar_full;
  uint64_t *mbar_empty_ptr = wg_smem.mbar_empty;

  token_t **symm_buffer_ptrs = smem.x_ptrs;
  weight_t **smem_symm_weight_ptrs = smem.weight_ptrs;

  // Only the leading warp of each real warp group initializes barriers.
  // The weight warp (when kHasWeight) is excluded by the range check.
  bool is_init_warp =
      (warp_id % kWarpsPerWG == 0) && warp_id < kNumWGPerBlock * kWarpsPerWG;
  int32_t consumer_arr_cnt = (kWarpsPerWG - 1);
  if (is_init_warp and elect_one_sync()) {
    for (int32_t i = 0; i < kNumLoadStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
    }
    for (int32_t i = 0; i < kNumLoadStages; ++i) {
      initialize_barrier(mbar_empty_ptr + i, consumer_arr_cnt);
    }
  }

  if (thread_id < num_ranks) {
    symm_buffer_ptrs[thread_id] =
        reinterpret_cast<token_t **>(x_ptrs)[thread_id];
    if constexpr (kHasWeight) {
      smem_symm_weight_ptrs[thread_id] =
          reinterpret_cast<weight_t **>(weight_ptrs)[thread_id];
    }
  }

  auto producer_pipe_state = PipelineState<kNumLoadStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumLoadStages>(0, 0, 0);

  __syncthreads();

  float acc[kElemsPerThread];
  PRAGMA_UNROLL
  for (int32_t i = 0; i < kElemsPerThread; ++i) {
    acc[i] = 0;
  }

  if constexpr (kHasWeight) {
    static_assert((kNumWarps - 1) % kWarpsPerWG == 0,
                  "when kHasWeight is true, (kNumWarps - 1) must be divisible "
                  "by kWarpsPerWG");
    // only use the last warp of each block to combine weight
    if (warp_id == kNumWarps - 1) {
      int32_t total_weight_warps = num_block;
      int32_t total_weight_threads = total_weight_warps * WARP_SIZE;
      int32_t num_weights = num_token * kTopk;
      int32_t global_weight_thread_id = lane_id + block_id * WARP_SIZE;
      for (int i = global_weight_thread_id; i < num_weights;
           i += total_weight_threads) {
        int32_t token_offset = i / kTopk;
        int32_t topk_idx = i % kTopk;
        int32_t expert_idx = topk_indices[token_offset * kTopk + topk_idx];
        int32_t scatter_idx =
            token_dst_scatter_indices[token_offset * kTopk + topk_idx];
        int32_t expert_rank = expert_idx / num_experts_per_rank;
        bool is_valid_lane = (expert_rank < num_ranks) && (scatter_idx != -1);
        weight_t valid_weight = 0;
        if (is_valid_lane) {
          valid_weight = reinterpret_cast<weight_t *>(
              smem_symm_weight_ptrs[expert_rank])[scatter_idx * kTopk +
                                                  topk_idx];
        }
        reinterpret_cast<weight_t *>(recv_weight)[i] = valid_weight;
      }
    }
  } else {
    static_assert(
        kNumWarps % kWarpsPerWG == 0,
        "use the first kWarpsPerWG warps of each block to combine weight");
  }

  int32_t store_buffer_idx = 0;
  bool is_tma_load_warp =
      (warp_id % kWarpsPerWG == 0) && (warp_id < kNumWGPerBlock * kWarpsPerWG);
  bool is_consumer_warp =
      (warp_id < kNumWGPerBlock * kWarpsPerWG) && !is_tma_load_warp;
  constexpr int32_t kNumConsumerThreadsPerWG = (kWarpsPerWG - 1) * WARP_SIZE;
  const int32_t consumer_tid_in_wg =
      thread_id % (WARP_SIZE * kWarpsPerWG) - WARP_SIZE;
  if (is_tma_load_warp) {
    for (int token_offset = global_warp_group_id; token_offset < num_token;
         token_offset += total_warp_groups) {

      int32_t expert_rank_lane = 0;
      int32_t scatter_idx_lane = 0;
      int32_t is_valid_lane = 0;

      static_assert(kTopk <= WARP_SIZE,
                    "kTopk must be less than or equal to WARP_SIZE");
      if (lane_id < kTopk) {
        int32_t expert_idx = topk_indices[token_offset * kTopk + lane_id];
        expert_rank_lane = expert_idx / num_experts_per_rank;
        int32_t is_need_send = topk_send_mask[token_offset * kTopk + lane_id];
        scatter_idx_lane =
            token_dst_scatter_indices[token_offset * kTopk + lane_id];
        is_valid_lane = expert_rank_lane < num_ranks and is_need_send;
      }

      __syncwarp();

      // bit i==1 means lane i has a valid token
      uint32_t valid_mask = __ballot_sync(0xffffffff, is_valid_lane);

      while (valid_mask) {
        int32_t j = __ffs(valid_mask) - 1; // next valid lane
        valid_mask &= (valid_mask - 1);    // clear lowest set bit

        int32_t expert_rank = __shfl_sync(0xffffffff, expert_rank_lane, j);
        int32_t scatter_idx = __shfl_sync(0xffffffff, scatter_idx_lane, j);

        uint64_t *empty = mbar_empty_ptr + producer_pipe_state.index();
        uint64_t *full = mbar_full_ptr + producer_pipe_state.index();

        wait_barrier(empty, producer_pipe_state.phase());

        void *src_gmem_ptr = (void *)(reinterpret_cast<token_t *>(
                                          symm_buffer_ptrs[expert_rank]) +
                                      scatter_idx * kHiddenSize);
        void *dst_smem_ptr =
            (void *)wg_smem.tma_load_buffer[producer_pipe_state.index()];

        if (elect_one_sync()) {
          mbar_arrive_and_set_barrier_transaction_bytes(full,
                                                        num_bytes_per_token);
          tma_copy_1d_g2s(src_gmem_ptr, full, dst_smem_ptr,
                          num_bytes_per_token);
        }

        ++producer_pipe_state;
        __syncwarp();
      }
    }
  } else if (is_consumer_warp) {

    union {
      int4 vec;
      __nv_bfloat162 bf162[4];
    } converter;

    static_assert(kElemsPerInt4 % 2 == 0, "kElemsPerInt4 must be even");
    static_assert(kNumStoreStages > 0,
                  "kNumStoreStages must be > 0 for TMA store pipeline");
    static_assert(kTopk <= WARP_SIZE,
                  "kTopk must be <= WARP_SIZE for warp shuffle");

    const uint32_t is_leader_lane = elect_one_sync();
    int32_t token_iter = 0;
    for (int token_offset = global_warp_group_id; token_offset < num_token;
         token_offset += total_warp_groups, ++token_iter) {
      int32_t is_valid_lane = 0;

      if (lane_id < kTopk) {
        int32_t expert_idx = topk_indices[token_offset * kTopk + lane_id];
        int32_t expert_rank = expert_idx / num_experts_per_rank;
        int32_t is_need_send = topk_send_mask[token_offset * kTopk + lane_id];
        is_valid_lane = (expert_rank < num_ranks) && is_need_send;
      }

      uint32_t valid_mask = __ballot_sync(0xffffffff, is_valid_lane);
      int32_t valid_count = __popc(valid_mask);

      for (int32_t j = 0; j < valid_count; ++j) {
        uint64_t *empty = mbar_empty_ptr + consumer_pipe_state.index();
        uint64_t *full = mbar_full_ptr + consumer_pipe_state.index();

        // consumer wait
        wait_barrier(full, consumer_pipe_state.phase());

        // reduce from the current stage buffer
        token_t *smem_ptr = reinterpret_cast<token_t *>(
            wg_smem.tma_load_buffer[consumer_pipe_state.index()]);
        int4 *smem_ptr_int4 = reinterpret_cast<int4 *>(smem_ptr);

        constexpr int32_t kInt4PerThread =
            (kHiddenSizeInt4 + kNumConsumerThreadsPerWG - 1) /
            kNumConsumerThreadsPerWG;
        static_assert(kInt4PerThread <= kElemsPerThread / kElemsPerInt4,
                      "kElemsPerThread too small for load");
        PRAGMA_UNROLL
        for (int32_t idx = 0; idx < kInt4PerThread; ++idx) {
          int32_t smem_idx =
              consumer_tid_in_wg + idx * kNumConsumerThreadsPerWG;
          if (smem_idx < kHiddenSizeInt4) {
            int4 data = smem_ptr_int4[smem_idx];
            converter.vec = data;
            PRAGMA_UNROLL
            for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
              float2 fp32_vec2 = __bfloat1622float2(converter.bf162[k]);
              acc[idx * kElemsPerInt4 + k * 2] += fp32_vec2.x;
              acc[idx * kElemsPerInt4 + k * 2 + 1] += fp32_vec2.y;
            }
          }
        }
        __syncwarp();

        // consumer release
        if (is_leader_lane) {
          arrive_barrier(empty);
        }
        __syncwarp();
        ++consumer_pipe_state;
      }

      // consumer store to gmem
      token_t *dst_gmem_ptr =
          reinterpret_cast<token_t *>(recv_x) + token_offset * kHiddenSize;
      int32_t warp_id_in_consumer_wg = consumer_tid_in_wg / WARP_SIZE;

      // Wait for prior TMA store using this buffer to complete
      // All threads in the first warp must call tma_store_wait (per-thread bulk
      // async-group semantics) This ensures the thread that issued the prior
      // TMA store has completed
      if (warp_id_in_consumer_wg == 0) {
        if (token_iter >= kNumStoreStages) {
          tma_store_wait<kNumStoreStages - 1>();
        }
      }

      // sync all consumer threads before writing to store buffer
      named_barrier_arrive_and_wait(kNumConsumerThreadsPerWG,
                                    warp_group_id + 1);

      // write accumulated result to store buffer
      int4 *tma_store_ptr_int4 =
          reinterpret_cast<int4 *>(wg_smem.tma_store_buffer[store_buffer_idx]);
      constexpr int32_t kInt4PerThreadStore =
          (kHiddenSizeInt4 + kNumConsumerThreadsPerWG - 1) /
          kNumConsumerThreadsPerWG;
      static_assert(kInt4PerThreadStore <= kElemsPerThread / kElemsPerInt4,
                    "kElemsPerThread too small for store");
      PRAGMA_UNROLL
      for (int32_t idx = 0; idx < kInt4PerThreadStore; ++idx) {
        int32_t dst_idx = consumer_tid_in_wg + idx * kNumConsumerThreadsPerWG;
        if (dst_idx < kHiddenSizeInt4) {
          PRAGMA_UNROLL
          for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
            converter.bf162[k] = __float22bfloat162_rn(
                make_float2(acc[idx * kElemsPerInt4 + k * 2],
                            acc[idx * kElemsPerInt4 + k * 2 + 1]));
            acc[idx * kElemsPerInt4 + k * 2] = 0;
            acc[idx * kElemsPerInt4 + k * 2 + 1] = 0;
          }
          tma_store_ptr_int4[dst_idx] = converter.vec;
        }
      }

      // ensure all smem writes are visible before TMA store reads them
      fence_async_shared();
      named_barrier_arrive_and_wait(kNumConsumerThreadsPerWG,
                                    warp_group_id + 1);
      if (warp_id_in_consumer_wg == 0) {
        if (is_leader_lane) {
          tma_copy_1d_s2g(reinterpret_cast<void *>(tma_store_ptr_int4),
                          dst_gmem_ptr, num_bytes_per_token);
          tma_store_arrive();
        }
        __syncwarp();
      }
      store_buffer_idx = (store_buffer_idx + 1) % kNumStoreStages;
    }
  }

  tma_store_wait<0>();
  __syncthreads();
}

} // namespace kernels

void compute_stable_local_token_within_expert_offset_cuda(
    int32_t *topk_indices, int32_t num_token, int32_t topk, int32_t num_experts,
    int32_t *block_cumsum_hist, int32_t *token_within_expert_offset,
    int32_t *expert_counts, int32_t num_sm, cudaStream_t stream) {
  constexpr int32_t kNumWarps = 32;
  constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
  static int device_sm_count = 0;
  if (device_sm_count == 0) {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&device_sm_count,
                                      cudaDevAttrMultiProcessorCount, device));
  }
  if (num_sm <= 0) {
    num_sm = (num_token * topk + kNumThreads - 1) / kNumThreads;
  }
  num_sm = (num_sm < device_sm_count) ? num_sm : device_sm_count;

  dim3 block_dim(kNumThreads);
  dim3 grid_dim(num_sm);
  size_t smem_size = (num_experts + 1) * sizeof(int32_t) * kNumWarps;
  CUDA_CHECK(cudaFuncSetAttribute(
      kernels::kernel_compute_stable_local_token_within_expert_offset<
          kNumWarps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  void *kernel_args[] = {
      &topk_indices, &num_token,         &topk,
      &num_experts,  &block_cumsum_hist, &token_within_expert_offset,
      &expert_counts};
  CUDA_CHECK(cudaLaunchCooperativeKernel(
      (void *)kernels::kernel_compute_stable_local_token_within_expert_offset<
          kNumWarps>,
      grid_dim, block_dim, kernel_args, smem_size, stream));
  CUDA_CHECK(cudaGetLastError());
}

void compute_dispatch_layout_cuda(
    int32_t *topk_indices, int32_t *token_within_expert_offset,
    int32_t *local_splits, int32_t **full_splits_ptrs, int32_t **barrier_ptrs,
    int32_t *recv_base_offset, int32_t *token_dst_scatter_indices,
    int32_t *token_topk_send_mask, int32_t *recv_token_count_cpu,
    int32_t *recv_token_count, int32_t num_token, int32_t topk,
    int32_t num_experts, int32_t rank, int32_t num_ranks, int32_t num_sm,
    cudaStream_t stream) {
  constexpr int32_t kNumWarps = 32;
  constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
  dim3 block_dim(kNumThreads);
  dim3 grid_dim(num_sm);
  size_t smem_size = sizeof(int32_t) * kNumWarps;
  FLASH_CHECK(num_experts <= kNumThreads)
      << "num_experts must be less than or equal to kNumThreads, "
      << num_experts << " > " << kNumThreads;
  FLASH_CHECK(num_sm > 0) << "num_sm must be greater than 0, " << num_sm
                          << " <= 0";
  CUDA_CHECK(cudaFuncSetAttribute(
      kernels::kernel_compute_dispatch_layout<kNumWarps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  void *kernel_args[] = {&topk_indices,
                         &token_within_expert_offset,
                         &local_splits,
                         &full_splits_ptrs,
                         &barrier_ptrs,
                         &recv_base_offset,
                         &token_dst_scatter_indices,
                         &token_topk_send_mask,
                         &recv_token_count_cpu,
                         &recv_token_count,
                         &num_token,
                         &topk,
                         &num_experts,
                         &rank,
                         &num_ranks};
  flash_comm::launch_kernel_ex(
      (void *)kernels::kernel_compute_dispatch_layout<kNumWarps>, grid_dim,
      block_dim, kernel_args, smem_size, stream,
      flash_comm::internal::get_cga_cluster_size(), true);
  CUDA_CHECK(cudaGetLastError());
}

void dispatch_intranode_cuda(
    void *x, void *topk_send_mask, void *topk_weights, void *topk_indices,
    void *token_dst_scatter_indices, void *recv_x_ptrs,
    void **recv_weights_ptrs, void **recv_topk_scatter_indices_ptrs,
    int32_t num_token, int32_t hidden_size, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks, int32_t num_sm,
    flash_comm::FlashCommDType dtype, flash_comm::FlashCommDType weight_dtype,
    flash_comm::FlashCommDType offset_dtype, int32_t topk,
    cudaStream_t stream) {
  // check alignment
  constexpr int32_t kNumStages = 12;
  constexpr int32_t kNumConsumerGroups = 3;
  constexpr int32_t kMaxSmemSize = 226 * 1024;

  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
      DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
        DISPATCH_HIDDEN_SIZE(hidden_size, kHiddenSize, {
          DISPATCH_TOPK(topk, kTopk, {
            bool has_weight = (topk_weights != nullptr);
            // v1 kernel: fewer warps (1 + kNumConsumerGroups), in-flight TMA
            // store groups
            constexpr int32_t kNumWarps = 1 + kNumConsumerGroups;
            constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
            dim3 block_dim(kNumThreads);
            dim3 grid_dim(num_sm);
            using smem_t = kernels::smem::DispatchIntraNodeSmem<
                token_t, weight_t, offset_t, kHiddenSize, kNumStages>;
            constexpr int32_t smem_size = sizeof(smem_t);
            FLASH_CHECK(smem_size <= kMaxSmemSize);
            DISPATCH_BOOL(has_weight, kHasWeight, {
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernels::kernel_dispatch_intranode_v1<
                      token_t, weight_t, offset_t, kHiddenSize, kTopk,
                      kNumStages, kNumConsumerGroups, kHasWeight>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
              flash_comm::launch_kernel_ex(
                  kernels::kernel_dispatch_intranode_v1<
                      token_t, weight_t, offset_t, kHiddenSize, kTopk,
                      kNumStages, kNumConsumerGroups, kHasWeight>,
                  grid_dim, block_dim, smem_size, stream,
                  flash_comm::internal::get_cga_cluster_size(), x,
                  reinterpret_cast<int32_t *>(topk_send_mask),
                  kHasWeight ? topk_weights : nullptr,
                  reinterpret_cast<offset_t *>(topk_indices),
                  reinterpret_cast<offset_t *>(token_dst_scatter_indices),
                  num_token, hidden_size, num_experts_per_rank, rank, num_ranks,
                  recv_x_ptrs, recv_weights_ptrs,
                  reinterpret_cast<offset_t **>(
                      recv_topk_scatter_indices_ptrs));
            });

            // constexpr int32_t kNumWarpPerConsumerGroup = kTopk;
            // constexpr int32_t kNumWarps = kNumConsumerGroups *
            // kNumWarpPerConsumerGroup + 1; constexpr int32_t kNumThreads =
            // kNumWarps * WARP_SIZE; dim3 block_dim(kNumThreads); dim3
            // grid_dim(num_sm); using smem_t =
            // kernels::smem::DispatchIntraNodeSmem<token_t, weight_t, offset_t,
            // kHiddenSize, kNumStages>; constexpr int32_t smem_size =
            // sizeof(smem_t); FLASH_CHECK(smem_size <= kMaxSmemSize);
            // CUDA_CHECK(cudaFuncSetAttribute(
            //     kernels::kernel_dispatch_intranode<token_t, weight_t,
            //     offset_t, kHiddenSize, kTopk, kNumStages,
            //     kNumConsumerGroups>,
            //     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            // kernels::kernel_dispatch_intranode<token_t, weight_t, offset_t,
            // kHiddenSize, kTopk, kNumStages, kNumConsumerGroups>
            //     <<<grid_dim, block_dim, smem_size, stream>>>(
            //         x, reinterpret_cast<int32_t *>(topk_send_mask),
            //         topk_weights, reinterpret_cast<offset_t *>(topk_indices),
            //         reinterpret_cast<offset_t
            //         *>(token_dst_scatter_indices), num_token, hidden_size,
            //         num_experts_per_rank, rank, num_ranks, recv_x_ptrs,
            //         recv_weights_ptrs, reinterpret_cast<offset_t
            //         **>(recv_topk_scatter_indices_ptrs));
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

void dispatch_intranode_chunk_cuda(
    void *x, void *topk_send_mask, void *topk_weights, void *topk_indices,
    void *token_dst_scatter_indices, void *recv_x_ptrs,
    void **recv_weights_ptrs, void **recv_topk_scatter_indices_ptrs,
    int32_t num_token, int32_t hidden_size, int32_t num_experts_per_rank,
    int32_t rank, int32_t num_ranks, int32_t num_sm,
    flash_comm::FlashCommDType dtype, flash_comm::FlashCommDType weight_dtype,
    flash_comm::FlashCommDType offset_dtype, int32_t topk,
    cudaStream_t stream) {
  // Chunk variant uses extra smem for meta ping-pong buffers, so we run fewer
  // stages to stay within 226KB.
  constexpr int32_t kNumStages = 13;
  constexpr int32_t kNumConsumerGroups = 1;
  constexpr int32_t kMaxSmemSize = 226 * 1024;

  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
      DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
        DISPATCH_HIDDEN_SIZE(hidden_size, kHiddenSize, {
          DISPATCH_TOPK(topk, kTopk, {
            bool has_weight = (topk_weights != nullptr);
            constexpr bool isMetaAligned = kTopk * sizeof(int32_t) % 16 == 0 &&
                                           kTopk * sizeof(weight_t) % 16 == 0 &&
                                           kTopk * sizeof(offset_t) % 16 == 0;

            if constexpr (isMetaAligned) {
              constexpr int32_t kNumWarpPerConsumerGroup = kTopk;
              constexpr int32_t kNumWarps =
                  kNumConsumerGroups * kNumWarpPerConsumerGroup + 1;
              constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
              dim3 block_dim(kNumThreads);
              dim3 grid_dim(num_sm);
              constexpr int32_t kNumDispatchChunkSize = 64;
              using smem_t = kernels::smem::DispatchIntraNodeChunkSmem<
                  token_t, weight_t, offset_t, kHiddenSize, kNumStages, kTopk,
                  kNumDispatchChunkSize>;
              constexpr int32_t smem_size = sizeof(smem_t);
              FLASH_CHECK(smem_size <= kMaxSmemSize);
              DISPATCH_BOOL(has_weight, kHasWeight, {
                CUDA_CHECK(cudaFuncSetAttribute(
                    kernels::kernel_dispatch_intranode_chunk<
                        token_t, weight_t, offset_t, kHiddenSize, kTopk,
                        kNumStages, kNumConsumerGroups, kHasWeight,
                        kNumDispatchChunkSize>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                flash_comm::launch_kernel_ex(
                    kernels::kernel_dispatch_intranode_chunk<
                        token_t, weight_t, offset_t, kHiddenSize, kTopk,
                        kNumStages, kNumConsumerGroups, kHasWeight,
                        kNumDispatchChunkSize>,
                    grid_dim, block_dim, smem_size, stream,
                    flash_comm::internal::get_cga_cluster_size(), x,
                    reinterpret_cast<int32_t *>(topk_send_mask),
                    kHasWeight ? topk_weights : nullptr,
                    reinterpret_cast<offset_t *>(topk_indices),
                    reinterpret_cast<offset_t *>(token_dst_scatter_indices),
                    num_token, hidden_size, num_experts_per_rank, rank,
                    num_ranks, recv_x_ptrs, recv_weights_ptrs,
                    reinterpret_cast<offset_t **>(
                        recv_topk_scatter_indices_ptrs));
              });
            } else {
              dispatch_intranode_cuda(
                  x, topk_send_mask, topk_weights, topk_indices,
                  token_dst_scatter_indices, recv_x_ptrs, recv_weights_ptrs,
                  recv_topk_scatter_indices_ptrs, num_token, hidden_size,
                  num_experts_per_rank, rank, num_ranks, num_sm, dtype,
                  weight_dtype, offset_dtype, topk, stream);
            }
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

void dispatch_postprocess_cuda(
    void *recv_x, void *recv_topk_scatter_indices_comm_buffer,
    void *recv_topk_weights, int32_t *recv_token_count, void *dispatch_weights,
    void *recv_topk_scatter_indices, int32_t num_recv_worst_token,
    int32_t hidden_size, int32_t topk, int32_t rank, int32_t num_ranks,
    int32_t num_sm, flash_comm::FlashCommDType dtype,
    flash_comm::FlashCommDType weight_dtype,
    flash_comm::FlashCommDType offset_dtype, cudaStream_t stream) {
  constexpr int32_t kNumWarps = 4; // config: can be tuned
  const int32_t kNumThreads = kNumWarps * WARP_SIZE;
  size_t num_blocks = num_sm;
  if (num_sm <= 0) {
    // too many empty blocks, performance degradation
    num_blocks = num_recv_worst_token;
  }
  dim3 block_dim(kNumThreads);
  dim3 grid_dim(num_blocks);

  constexpr int32_t kNumStages = 5;
  constexpr int32_t kWritePipeCount = 2;

  bool has_weight =
      (recv_topk_weights != nullptr && dispatch_weights != nullptr);
  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
      DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
        DISPATCH_HIDDEN_SIZE(hidden_size, kHiddenSize, {
          using smem_t =
              kernels::smem::DispatchPostprocessSmem<token_t, kHiddenSize,
                                                     kNumStages>;
          constexpr int32_t smem_size = sizeof(smem_t);
          DISPATCH_BOOL(has_weight, kHasWeight, {
            CUDA_CHECK(cudaFuncSetAttribute(
                kernels::kernel_dispatch_postprocess_tma<
                    token_t, weight_t, offset_t, kHiddenSize, kNumStages,
                    kWritePipeCount, kHasWeight>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            kernels::kernel_dispatch_postprocess_tma<
                token_t, weight_t, offset_t, kHiddenSize, kNumStages,
                kWritePipeCount, kHasWeight>
                <<<grid_dim, block_dim, smem_size, stream>>>(
                    recv_x,
                    kHasWeight ? reinterpret_cast<weight_t *>(recv_topk_weights)
                               : nullptr,
                    reinterpret_cast<offset_t *>(
                        recv_topk_scatter_indices_comm_buffer),
                    recv_token_count,
                    kHasWeight ? reinterpret_cast<weight_t *>(dispatch_weights)
                               : nullptr,
                    reinterpret_cast<offset_t *>(recv_topk_scatter_indices),
                    hidden_size, topk, rank, num_ranks);
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

void combine_intranode_cuda(
    void *x_ptrs, void *weight_ptrs, void *topk_send_mask, void *topk_indices,
    void *token_dst_scatter_indices, void *recv_x, void *recv_weight,
    int32_t num_token, int32_t hidden_size, int32_t topk,
    int32_t num_experts_per_rank, int32_t rank, int32_t num_ranks,
    int32_t num_sm, flash_comm::FlashCommDType dtype,
    flash_comm::FlashCommDType weight_dtype,
    flash_comm::FlashCommDType offset_dtype, cudaStream_t stream) {
  constexpr int32_t kNumLoadStages = 6;
  constexpr int32_t kNumStoreStages = 2;
  constexpr int32_t kElemsPerThread = 64;

  constexpr int32_t kWarpsPerWG = 9;
  constexpr int32_t kNumWGPerBlock = 2;

  constexpr int32_t kMaxSmemSize = 226 * 1024;
  FLASH_CHECK((weight_ptrs != nullptr) == (recv_weight != nullptr))
      << "weight_ptrs and recv_weight must be both nullptr or both not nullptr";
  bool has_weight = weight_ptrs != nullptr;
  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
      DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
        DISPATCH_HIDDEN_SIZE(hidden_size, kHiddenSize, {
          DISPATCH_TOPK(topk, kTopk, {
            DISPATCH_BOOL(has_weight, kHasWeight, {
              constexpr int32_t kNumWarps =
                  kNumWGPerBlock * kWarpsPerWG + int(kHasWeight);
              constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
              using smem_t = kernels::smem::CombineIntraNodeSmem<
                  token_t, weight_t, kHiddenSize, kNumLoadStages,
                  kNumStoreStages, kNumWGPerBlock>;
              constexpr int32_t smem_size = sizeof(smem_t);
              FLASH_CHECK(smem_size <= kMaxSmemSize);
              // CUDA_CHECK(cudaFuncSetAttribute(kernels::kernel_combine_intranode_v1<token_t,
              // offset_t, kTopk, kHiddenSize, kNumLoadStages, kNumStoreStages,
              //                                                                   kNumWarps, kWarpsPerWG,
              //                                                                   kElemsPerThread>,
              //                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
              //                                 smem_size));
              // dim3 block_dim(kNumThreads);
              // dim3 grid_dim(num_sm);
              // kernels::kernel_combine_intranode_v1<token_t, offset_t, kTopk,
              // kHiddenSize, kNumLoadStages, kNumStoreStages, kNumWarps,
              // kWarpsPerWG,
              //                                   kElemsPerThread><<<grid_dim,
              //                                   block_dim, smem_size,
              //                                   stream>>>(
              //     x_ptrs, reinterpret_cast<offset_t *>(topk_send_mask),
              //     reinterpret_cast<offset_t *>(topk_indices),
              //     reinterpret_cast<offset_t *>(token_dst_scatter_indices),
              //     recv_x, num_token, num_experts_per_rank, rank, num_ranks);
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernels::kernel_combine_intranode_v2<
                      token_t, weight_t, offset_t, kTopk, kHiddenSize,
                      kNumLoadStages, kNumStoreStages, kNumWarps, kWarpsPerWG,
                      kElemsPerThread, kHasWeight>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
              dim3 block_dim(kNumThreads);
              dim3 grid_dim(num_sm);
              flash_comm::launch_kernel_ex(
                  kernels::kernel_combine_intranode_v2<
                      token_t, weight_t, offset_t, kTopk, kHiddenSize,
                      kNumLoadStages, kNumStoreStages, kNumWarps, kWarpsPerWG,
                      kElemsPerThread, kHasWeight>,
                  grid_dim, block_dim, smem_size, stream,
                  flash_comm::internal::get_cga_cluster_size(), x_ptrs,
                  weight_ptrs, reinterpret_cast<offset_t *>(topk_send_mask),
                  reinterpret_cast<offset_t *>(topk_indices),
                  reinterpret_cast<offset_t *>(token_dst_scatter_indices),
                  recv_x, recv_weight, num_token, num_experts_per_rank, rank,
                  num_ranks);
            });
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

void combine_preprocess_inplace_cuda(
    void *x, void *weight_ptrs, int32_t *recv_token_count,
    void *recv_topk_scatter_indices, void *recv_topk_weight,
    int32_t num_recv_worst_token, int32_t hidden_size, int32_t topk,
    int32_t rank, int32_t num_ranks, int32_t num_sm,
    flash_comm::FlashCommDType dtype, flash_comm::FlashCommDType weight_dtype,
    flash_comm::FlashCommDType offset_dtype, cudaStream_t stream) {
  constexpr int32_t kNumWarps = 16;
  constexpr int32_t kElemsPerThread = 32;

  constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
  dim3 block_dim(kNumThreads);
  dim3 grid_dim(num_sm);
  size_t smem_size = 0;
  bool has_weight = weight_ptrs != nullptr;
  FLASH_CHECK((has_weight == (recv_topk_weight != nullptr)))
      << "has_weight and recv_topk_weight must be both nullptr or both not "
         "nullptr";
  FLASH_CHECK(topk <= WARP_SIZE)
      << "topk must be less than or equal to WARP_SIZE, " << topk << " > "
      << WARP_SIZE;
  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
      DISPATCH_HIDDEN_SIZE(hidden_size, kHiddenSize, {
        DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
          DISPATCH_BOOL(has_weight, kHasWeight, {
            kernels::kernel_combine_preprocess_inplace<
                token_t, weight_t, offset_t, kHiddenSize, kNumWarps,
                kElemsPerThread, kHasWeight>
                <<<grid_dim, block_dim, smem_size, stream>>>(
                    x, weight_ptrs, recv_token_count,
                    reinterpret_cast<offset_t *>(recv_topk_scatter_indices),
                    recv_topk_weight, topk, rank, num_ranks);
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

void barrier_all_on_stream_cuda(void **barrier_ptrs, int32_t rank,
                                int32_t num_ranks,
                                flash_comm::FlashCommDType dtype,
                                cudaStream_t stream) {
  constexpr int32_t kNumThreads = 128;
  FLASH_CHECK(num_ranks <= kNumThreads);
  dim3 block_dim(kNumThreads);
  dim3 grid_dim(1);
  FLASH_CHECK(dtype == flash_comm::FlashCommDType::Int32)
      << "Only Int32 barrier type is currently supported";
  kernels::kernel_barrier_all_on_stream<int32_t>
      <<<grid_dim, block_dim, 0, stream>>>(
          reinterpret_cast<int32_t **>(barrier_ptrs), rank, num_ranks);
  CUDA_CHECK(cudaGetLastError());
}

} // namespace intranode
} // namespace ep
} // namespace flash_comm