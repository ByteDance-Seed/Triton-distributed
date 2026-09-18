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

#include "ep_world_barrier.cuh"
#include "flash_comm/common.h"
#include "flash_comm/copy.cuh"
#include "flash_comm/ep/internode.h"
#include "flash_comm/ep/intranode.h"
#include "flash_comm/launch_utils.cuh"
#include "flash_comm/utils.cuh"
#include "internode_cuda_launch.cuh"
#include <cooperative_groups.h>
#include <cstdint>
#include <cuda_bf16.h>
#include <nccl_device.h>

namespace flash_comm {
namespace ep {
namespace internode {
namespace kernels {

constexpr int WARP_SIZE = 32;
// Payload transfers use contexts [0, num_qps). Layout publication and world
// barriers share context 0 and barrier slot 0; their ordering requirements are
// expressed by the barrier memory order below.
constexpr int kGinLayoutPutCtx = 0;
constexpr int kGinDispatchPutCtx = 0;
constexpr int kGinCombinePutCtx = 0;
constexpr int kGinBarrierCtx = 0;
constexpr int kGinBarrierId = 0;

// GIN signal ids are partitioned by protocol, node and dispatch chunk
// (kMaxDispatchPipelineChunks and the id-space layout live in
// flash_comm/ep/internode.h). Ids are context-local: signal storage is
// per-context and one context carries exactly one QP's traffic, so the QP is
// selected by the ncclGin context index, never encoded in the id. Per-chunk
// dispatch signals avoid relying on completion ordering across independently
// pipelined RDMA puts.
__device__ __forceinline__ ncclGinSignal_t dispatch_node_signal(int32_t node,
                                                                int32_t chunk) {
  return static_cast<ncclGinSignal_t>(node * kMaxDispatchPipelineChunks +
                                      chunk);
}

__device__ __forceinline__ ncclGinSignal_t combine_node_signal(int32_t nnodes,
                                                               int32_t node) {
  return static_cast<ncclGinSignal_t>(ep_combine_signal_base(nnodes) + node);
}

// No epoch state anywhere: the internode barrier kernel resets every EP
// signal to 0 at a globally quiescent point (see kernel_internode_gin_barrier)
// and each dispatch/combine emits exactly one increment per participating
// signal, so every wait targets the constant value below.
constexpr uint64_t kGinSignalStepTarget = 1;

__device__ __forceinline__ void split_qp_range(size_t bytes, int32_t num_qps,
                                               int32_t qp, size_t &off,
                                               size_t &len) {
  if (num_qps == 1) {
    off = 0;
    len = bytes;
    return;
  }
  const size_t base = bytes / static_cast<size_t>(num_qps);
  const size_t rem = bytes - base * static_cast<size_t>(num_qps);
  const size_t extra = static_cast<size_t>(qp) < rem ? 1 : 0;
  off = base * static_cast<size_t>(qp) +
        (static_cast<size_t>(qp) < rem ? static_cast<size_t>(qp) : rem);
  len = base + extra;
}

// Put one scalar region and attach the completion signal when the remaining
// count reaches zero. Callers handle the all-empty case separately; an
// intermediate pipeline step can suppress signaling by adding one sentinel to
// remaining_regions.
// Keep this force-inlined and scalar-only: runtime-indexed local region arrays
// are lowered to thread-local memory by CUDA.
template <typename Coop>
__device__ __forceinline__ void gin_put_region_tail_signal(
    ncclGin &gin, ncclDevComm dev_comm, int32_t peer, ncclWindow_t win,
    size_t dst_offset, size_t src_offset, size_t bytes,
    ncclGinSignal_t completion_signal, int32_t &remaining_regions, Coop coop) {
  if (bytes == 0) {
    return;
  }
  const bool is_last_region = (--remaining_regions == 0);
  if (is_last_region) {
    gin.put(ncclTeamWorld(dev_comm), peer, win, dst_offset, win, src_offset,
            bytes, ncclGin_SignalInc{completion_signal}, ncclGin_None{}, coop,
            ncclGin_None{}, cuda::thread_scope_thread,
            cuda::thread_scope_system);
  } else {
    gin.put(ncclTeamWorld(dev_comm), peer, win, dst_offset, win, src_offset,
            bytes, ncclGin_None{}, ncclGin_None{}, coop, ncclGin_None{},
            cuda::thread_scope_thread, cuda::thread_scope_system);
  }
}

__device__ __forceinline__ void gin_world_barrier_release(ncclDevComm dev_comm,
                                                          int32_t num_qps) {
  if (blockIdx.x != 0) {
    return;
  }
  ncclGin gin{dev_comm, kGinBarrierCtx};
  for (int32_t qp = static_cast<int32_t>(threadIdx.x); qp < num_qps;
       qp += static_cast<int32_t>(blockDim.x)) {
    ncclGin payload_gin{dev_comm, kGinDispatchPutCtx + qp};
    payload_gin.flush(ncclCoopThread(), cuda::memory_order_release);
  }
  __syncthreads();
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin,
                                      kGinBarrierId};
  bar.sync(ncclCoopCta(), cuda::memory_order_release,
           ncclGinFenceLevel::Relaxed);
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
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
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

#if !defined(FLASH_COMM_INTERNODE_INSTANTIATION)

// Standard EP layout kernel kept unchanged. Range EP uses the independent
// kernel below so the default path and its launch behavior remain stable.
template <int32_t kNumWarps>
void __global__ __launch_bounds__(kNumWarps *WARP_SIZE, 1)
    kernel_compute_dispatch_layout(
        int32_t *topk_indices,               // [num_token, topk]
        int32_t *token_within_expert_offset, // [num_token, topk]
        int32_t *local_splits,               // [num_experts + 1]
        int32_t *num_tokens_per_rank,        // local [num_ranks]
        int32_t *recv_base_offset, // [num_ranks, experts_per_rank, num_ranks]
        int32_t *token_dst_scatter_indices, // [num_token, topk]
        int32_t *token_topk_send_mask,      // [num_token, topk]
        int32_t *recv_token_count_cpu, // [num_ranks] (optional, pinned memory)
        int32_t *recv_token_count,     // [num_ranks] (optional, device memory)
        int32_t *recv_aligned_token_count_cpu, // [num_ranks] (optional, pinned)
        int32_t *recv_aligned_token_count,     // [num_ranks] (optional, device)
        int32_t *recv_expert_counts, // [experts_per_rank] (optional, device,
                                     // local rank only)
        int32_t num_token, int32_t topk, int32_t num_experts, int32_t rank,
        int32_t num_ranks, int32_t expert_alignment, int32_t local_world_size,
        ncclDevComm dev_comm, ncclWindow_t full_splits_win) {
  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  constexpr int32_t kBlockSize = kNumWarps * WARP_SIZE;
  const int32_t num_experts_per_rank = num_experts / num_ranks;
  __shared__ __align__(1024) int32_t scan_warp_prefix_sum[kNumWarps];
  __shared__ int32_t alignment_values[kBlockSize];

  const int32_t my_node = rank / local_world_size;
  const int32_t row_stride = num_experts + 2;
  const size_t splits_row_bytes =
      static_cast<size_t>(row_stride) * sizeof(int32_t);
  int32_t *full_splits =
      reinterpret_cast<int32_t *>(ncclGetLocalPointer(full_splits_win, 0));

  // Publish this rank's row once. The NCCL window row is the GIN put source;
  // local peers receive the same row through their LSA pointer to this window.
  if (block_id == 0) {
    int32_t *self_full_row =
        full_splits + static_cast<int64_t>(rank) * row_stride;
    for (int32_t j = thread_id; j < num_experts + 1; j += blockDim.x) {
      self_full_row[j] = local_splits[j];
    }
    if (thread_id == 0) {
      self_full_row[num_experts + 1] = num_token;
    }
    __syncthreads();
    __threadfence_system();
  }
  cooperative_groups::this_grid().sync();

  // One-shot fanout: every rank sends its own row to every rank. Same-node
  // destinations use NVL peer stores; cross-node destinations use one GIN put.
  ncclGin gin_put{dev_comm, kGinLayoutPutCtx};
  for (int32_t dst_rank = block_id; dst_rank < num_ranks;
       dst_rank += num_block) {
    if (dst_rank == rank) {
      continue;
    }
    const int32_t dst_node = dst_rank / local_world_size;
    const size_t row_offset = static_cast<size_t>(rank) * splits_row_bytes;
    if (dst_node == my_node) {
      const int32_t *src_row =
          full_splits + static_cast<int64_t>(rank) * row_stride;
      // Use the global peer rank here; NCCL maps it to the correct local LSA
      // slot.
      int32_t *dst_row = reinterpret_cast<int32_t *>(
          ncclGetPeerPointer(full_splits_win, row_offset, dst_rank));
      for (int32_t j = thread_id; j < row_stride; j += blockDim.x) {
        dst_row[j] = src_row[j];
      }
    } else {
      if (thread_id == 0) {
        gin_put.put(ncclTeamWorld(dev_comm), dst_rank, full_splits_win,
                    row_offset, full_splits_win, row_offset, splits_row_bytes,
                    ncclGin_None{}, ncclGin_None{}, ncclCoopThread(),
                    ncclGin_None{}, cuda::thread_scope_thread,
                    cuda::thread_scope_system);
      }
    }
  }
  __threadfence_system();
  cooperative_groups::this_grid().sync();

  // Block 0 flushes the shared put context and performs the world barrier
  // after every CTA has finished issuing puts.
  if (block_id == 0) {
    ::flash_comm::ep::kernels::ep_world_barrier_after_puts(
        dev_comm, kGinBarrierCtx, kGinLayoutPutCtx, true);
  }
  cooperative_groups::this_grid().sync();

  for (int32_t dst_rank = block_id; dst_rank < num_ranks;
       dst_rank += num_block) {
    int32_t src_rank = thread_id % num_ranks;
    int32_t local_expert_idx = thread_id / num_ranks;
    int32_t value = 0;
    if (thread_id < num_experts) {
      value = full_splits[src_rank * row_stride +
                          dst_rank * num_experts_per_rank + local_expert_idx];
    }

    int32_t prefix_sum = block_scan_inclusive(value, scan_warp_prefix_sum);
    int32_t aligned_prefix_sum = prefix_sum;

    bool need_per_expert =
        (expert_alignment > 1) || (recv_expert_counts != nullptr);
    if (need_per_expert) {
      alignment_values[thread_id] = prefix_sum;
      __syncthreads();

      int32_t expert_total = 0;
      int32_t my_padding = 0;
      if (thread_id < num_experts_per_rank) {
        int32_t le = thread_id;
        int32_t cum_end = alignment_values[(le + 1) * num_ranks - 1];
        int32_t cum_start = (le > 0) ? alignment_values[le * num_ranks - 1] : 0;
        expert_total = cum_end - cum_start;
        if (expert_alignment > 1) {
          int32_t aligned_total = (expert_total + expert_alignment - 1) /
                                  expert_alignment * expert_alignment;
          my_padding = aligned_total - expert_total;
        }
      }

      if (dst_rank == rank && recv_expert_counts != nullptr &&
          thread_id < num_experts_per_rank) {
        recv_expert_counts[thread_id] = expert_total;
      }

      if (expert_alignment > 1) {
        __syncthreads();

        int32_t padding_val =
            (thread_id < num_experts_per_rank) ? my_padding : 0;
        int32_t cum_padding =
            block_scan_inclusive(padding_val, scan_warp_prefix_sum);

        if (thread_id < num_experts_per_rank) {
          alignment_values[thread_id] = cum_padding;
        }
        __syncthreads();

        if (thread_id < num_experts) {
          int32_t le = thread_id / num_ranks;
          aligned_prefix_sum =
              prefix_sum + ((le > 0) ? alignment_values[le - 1] : 0);
        }
      }
    }

    if (thread_id < num_experts) {
      recv_base_offset[dst_rank * num_experts + local_expert_idx * num_ranks +
                       src_rank] = aligned_prefix_sum - value;
    }

    if (thread_id == num_experts - 1) {
      if (recv_token_count != nullptr) {
        recv_token_count[dst_rank] = prefix_sum;
      }
      if (recv_token_count_cpu != nullptr) {
        recv_token_count_cpu[dst_rank] = prefix_sum;
      }
      if (expert_alignment > 1) {
        int32_t aligned_count =
            prefix_sum + alignment_values[num_experts_per_rank - 1];
        if (recv_aligned_token_count != nullptr) {
          recv_aligned_token_count[dst_rank] = aligned_count;
        }
        if (recv_aligned_token_count_cpu != nullptr) {
          recv_aligned_token_count_cpu[dst_rank] = aligned_count;
        }
      }
      __threadfence_system();
    }
    __syncthreads();
  }

  cooperative_groups::this_grid().sync();

  for (int32_t i = block_id * kBlockSize + thread_id; i < num_token * topk;
       i += kBlockSize * num_block) {
    int32_t expert_idx = topk_indices[i];
    int32_t target_rank = expert_idx / num_experts_per_rank;
    int32_t target_local_expert_idx = expert_idx % num_experts_per_rank;
    int32_t num_pre_expert = i % topk;
    int32_t need_send = target_rank < num_ranks ? 1 : 0;
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

  cooperative_groups::this_grid().sync();

  for (int32_t g = block_id; g < num_ranks; g += num_block) {
    if (thread_id == 0) {
      num_tokens_per_rank[g] =
          full_splits[static_cast<int64_t>(g) * row_stride + num_experts + 1];
    }
  }
}

void __global__ __launch_bounds__(512, 1)
    kernel_internode_gin_barrier(ncclDevComm dev_comm, int32_t max_qps) {
  // The barrier is also used after stream-ordered D2D copies populate the
  // registered RDMA window. Publish those writes at system scope before peers
  // can issue GIN puts that read the window in the next kernel.
  if (threadIdx.x == 0) {
    __threadfence_system();
  }
  __syncthreads();
  gin_world_barrier_release(dev_comm, max_qps);
}

void __global__ kernel_stage_dispatch_range(
    ncclWindow_t rdma_win, size_t source_slot_offset, size_t token_region_bytes,
    size_t weight_region_offset, const uint4 *x, const int32_t *topk_indices,
    const int32_t *topk_send_mask, const int32_t *token_dst_scatter_indices,
    const float *topk_weights, int32_t num_token, int32_t hidden_size,
    int32_t topk, const int32_t *logical_token_range) {
  const int32_t logical_begin = logical_token_range[0];
  const int32_t logical_end = logical_token_range[1];
  if (logical_end <= logical_begin) {
    return;
  }
  const int32_t begin = max(0, min(logical_begin, num_token));
  const int32_t end = max(0, min(logical_end, num_token));
  if (end <= begin) {
    return;
  }
  const int32_t active = end - begin;
  char *slot = reinterpret_cast<char *>(
      ncclGetLocalPointer(rdma_win, source_slot_offset));

  const int64_t row_bytes =
      static_cast<int64_t>(hidden_size) * sizeof(uint16_t);
  const int64_t x_vecs_per_row = row_bytes / sizeof(uint4);
  const int64_t x_vec_count = static_cast<int64_t>(active) * x_vecs_per_row;
  uint4 *dst_x = reinterpret_cast<uint4 *>(slot);
  const uint4 *src_x = x + static_cast<int64_t>(begin) * x_vecs_per_row;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < x_vec_count;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    __stcg(dst_x + i, __ldcg(src_x + i));
  }

  const int64_t meta_count = static_cast<int64_t>(active) * topk;
  const int64_t src_meta_begin = static_cast<int64_t>(begin) * topk;
  int32_t *dst_topk = reinterpret_cast<int32_t *>(slot + token_region_bytes);
  int32_t *dst_mask = dst_topk + meta_count;
  int32_t *dst_scatter = dst_mask + meta_count;
  float *dst_weights = reinterpret_cast<float *>(slot + weight_region_offset);
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < meta_count;
       i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t src_i = src_meta_begin + i;
    // Routing/weights are full-input; mask/scatter are step-local compact.
    dst_topk[i] = topk_indices[src_i];
    dst_mask[i] = topk_send_mask[i];
    dst_scatter[i] = token_dst_scatter_indices[i];
    if (topk_weights != nullptr) {
      dst_weights[i] = topk_weights[src_i];
    }
  }
}

void __global__ __launch_bounds__(512, 1)
    kernel_internode_gin_barrier_if_active(ncclDevComm dev_comm,
                                           int32_t max_qps,
                                           const int32_t *logical_token_range) {
  if (logical_token_range[1] <= logical_token_range[0]) {
    return;
  }
  if (threadIdx.x == 0) {
    __threadfence_system();
  }
  __syncthreads();
  gin_world_barrier_release(dev_comm, max_qps);
}

// Fused reset-signals-then-barrier. Signal ids are context-local, so the
// range [signal_begin, signal_end) is reset on every context in [0, max_qps).
// The order is load-bearing: the reset must complete before this rank arrives
// at the barrier. Pre-arrival those signals are quiescent, which resetSignal
// requires: all increments expected by the preceding dispatch/combine were
// both emitted and drained (their kernels wait for every emitted signal
// before completing, and this kernel is stream-ordered after them), while
// peers cannot issue the next step's puts until this rank arrives. Resetting
// after the barrier would instead race with the next step's incoming
// SignalInc and could lose an increment (hanging the next waiter). Subsequent
// dispatch/combine kernels wait for the constant kGinSignalStepTarget.
void __global__ __launch_bounds__(512, 1)
    kernel_internode_gin_reset_signals_barrier(ncclDevComm dev_comm,
                                               int32_t max_qps,
                                               int32_t signal_begin,
                                               int32_t signal_end) {
  const int32_t range = signal_end - signal_begin;
  const int32_t total = range * max_qps;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < total;
       i += static_cast<int32_t>(blockDim.x)) {
    // Dispatch and combine share contexts [0, max_qps); signal-id ranges keep
    // their protocol state disjoint.
    const int32_t ctx = i / range;
    const ncclGinSignal_t sig =
        static_cast<ncclGinSignal_t>(signal_begin + i % range);
    ncclGin sig_gin{dev_comm, ctx};
    sig_gin.resetSignal(sig);
  }
  // Publish every thread's resets (and any stream-ordered D2D writes into the
  // RDMA window) at system scope before arriving at the barrier: sync first so
  // thread 0's fence covers all reset stores.
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
  }
  __syncthreads();
  gin_world_barrier_release(dev_comm, max_qps);
}

void __global__ __launch_bounds__(512, 1)
    kernel_internode_gin_reset_signals_barrier_if_active(
        ncclDevComm dev_comm, int32_t max_qps, int32_t signal_begin,
        int32_t signal_end, const int32_t *logical_token_range) {
  if (logical_token_range[1] <= logical_token_range[0]) {
    return;
  }
  const int32_t range = signal_end - signal_begin;
  const int32_t total = range * max_qps;
  for (int32_t i = static_cast<int32_t>(threadIdx.x); i < total;
       i += static_cast<int32_t>(blockDim.x)) {
    const int32_t ctx = i / range;
    const ncclGinSignal_t sig =
        static_cast<ncclGinSignal_t>(signal_begin + i % range);
    ncclGin sig_gin{dev_comm, ctx};
    sig_gin.resetSignal(sig);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
  }
  __syncthreads();
  gin_world_barrier_release(dev_comm, max_qps);
}

} // namespace kernels

void internode_barrier_on_stream_cuda(const void *dev_comm_host,
                                      int32_t max_qps, cudaStream_t stream) {
  FLASH_CHECK(max_qps > 0);
  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  dim3 block_dim(512);
  dim3 grid_dim(1);
  void *kernel_args[] = {&dev_comm, &max_qps};
  flash_comm::launch_kernel_ex((void *)kernels::kernel_internode_gin_barrier,
                               grid_dim, block_dim, kernel_args, 0, stream,
                               flash_comm::internal::get_cga_cluster_size(),
                               false);
  CUDA_CHECK(cudaGetLastError());
}

void internode_barrier_on_stream_if_active_cuda(
    const void *dev_comm_host, int32_t max_qps,
    const int32_t *logical_token_range, cudaStream_t stream) {
  FLASH_CHECK(max_qps > 0);
  FLASH_CHECK(logical_token_range != nullptr);
  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  dim3 block_dim(512);
  dim3 grid_dim(1);
  void *kernel_args[] = {&dev_comm, &max_qps, &logical_token_range};
  flash_comm::launch_kernel_ex(
      (void *)kernels::kernel_internode_gin_barrier_if_active, grid_dim,
      block_dim, kernel_args, 0, stream,
      flash_comm::internal::get_cga_cluster_size(), false);
  CUDA_CHECK(cudaGetLastError());
}

void internode_reset_signals_barrier_on_stream_cuda(const void *dev_comm_host,
                                                    int32_t max_qps,
                                                    int32_t signal_begin,
                                                    int32_t signal_end,
                                                    cudaStream_t stream) {
  FLASH_CHECK(max_qps > 0);
  FLASH_CHECK(signal_begin >= 0 && signal_begin <= signal_end)
      << "invalid EP signal reset range [" << signal_begin << ", " << signal_end
      << ")";
  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  dim3 block_dim(512);
  dim3 grid_dim(1);
  void *kernel_args[] = {&dev_comm, &max_qps, &signal_begin, &signal_end};
  flash_comm::launch_kernel_ex(
      (void *)kernels::kernel_internode_gin_reset_signals_barrier, grid_dim,
      block_dim, kernel_args, 0, stream,
      flash_comm::internal::get_cga_cluster_size(), false);
  CUDA_CHECK(cudaGetLastError());
}

void internode_reset_signals_barrier_on_stream_if_active_cuda(
    const void *dev_comm_host, int32_t max_qps, int32_t signal_begin,
    int32_t signal_end, const int32_t *logical_token_range,
    cudaStream_t stream) {
  FLASH_CHECK(max_qps > 0);
  FLASH_CHECK(signal_begin >= 0 && signal_begin <= signal_end);
  FLASH_CHECK(logical_token_range != nullptr);
  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  dim3 block_dim(512);
  dim3 grid_dim(1);
  void *kernel_args[] = {&dev_comm, &max_qps, &signal_begin, &signal_end,
                         &logical_token_range};
  flash_comm::launch_kernel_ex(
      (void *)kernels::kernel_internode_gin_reset_signals_barrier_if_active,
      grid_dim, block_dim, kernel_args, 0, stream,
      flash_comm::internal::get_cga_cluster_size(), false);
  CUDA_CHECK(cudaGetLastError());
}

void stage_dispatch_internode_range_cuda(
    uintptr_t rdma_rail_send_win_handle, const void *x,
    const int32_t *topk_indices, const int32_t *topk_send_mask,
    const int32_t *token_dst_scatter_indices, const void *topk_weights,
    int32_t num_token, int32_t max_slot_num_token, int32_t hidden_size,
    int32_t topk, int32_t rank, int32_t local_world_size,
    const int32_t *logical_token_range, cudaStream_t stream) {
  FLASH_CHECK(hidden_size > 0 && topk > 0);
  FLASH_CHECK(logical_token_range != nullptr);
  FLASH_CHECK((static_cast<int64_t>(hidden_size) * sizeof(uint16_t)) %
                  sizeof(uint4) ==
              0);
  const int32_t nnodes = 1;
  const auto layout =
      rdma_rail_send_layout_desc(max_slot_num_token, hidden_size, topk, nnodes);
  const int32_t node = rank / local_world_size;
  const size_t source_slot_offset =
      static_cast<size_t>(node) * layout.slot_stride_bytes;
  ncclWindow_t rdma_win =
      reinterpret_cast<ncclWindow_t>(rdma_rail_send_win_handle);
  const float *weights = static_cast<const float *>(topk_weights);
  constexpr int32_t threads = 256;
  const int64_t max_x_vecs = static_cast<int64_t>(max_slot_num_token) *
                             hidden_size * sizeof(uint16_t) / sizeof(uint4);
  const int64_t max_meta = static_cast<int64_t>(max_slot_num_token) * topk;
  const int64_t max_work = max_x_vecs > max_meta ? max_x_vecs : max_meta;
  const int64_t requested_blocks = (max_work + threads - 1) / threads;
  const int32_t blocks =
      static_cast<int32_t>(requested_blocks < 2048 ? requested_blocks : 2048);
  kernels::kernel_stage_dispatch_range<<<blocks, threads, 0, stream>>>(
      rdma_win, source_slot_offset, layout.token_region_bytes,
      layout.topk_weights_offset, static_cast<const uint4 *>(x), topk_indices,
      topk_send_mask, token_dst_scatter_indices, weights, num_token,
      hidden_size, topk, logical_token_range);
  CUDA_CHECK(cudaGetLastError());
}

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
    void *full_splits_win_ptr, cudaStream_t stream) {
  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  ncclWindow_t full_splits_win =
      reinterpret_cast<ncclWindow_t>(full_splits_win_ptr);
  constexpr int32_t kNumWarps = 32;
  constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
  dim3 block_dim(kNumThreads);
  dim3 grid_dim(num_sm);
  size_t smem_size = sizeof(int32_t) * kNumWarps;
  FLASH_CHECK(num_experts <= kNumThreads);
  FLASH_CHECK(num_sm > 0);
  CUDA_CHECK(cudaFuncSetAttribute(
      kernels::kernel_compute_dispatch_layout<kNumWarps>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  void *kernel_args[] = {&topk_indices,
                         &token_within_expert_offset,
                         &local_splits,
                         &num_tokens_per_rank,
                         &recv_base_offset,
                         &token_dst_scatter_indices,
                         &token_topk_send_mask,
                         &recv_token_count_cpu,
                         &recv_token_count,
                         &recv_aligned_token_count_cpu,
                         &recv_aligned_token_count,
                         &recv_expert_counts,
                         &num_token,
                         &topk,
                         &num_experts,
                         &rank,
                         &num_ranks,
                         &expert_alignment,
                         &local_world_size,
                         &dev_comm,
                         &full_splits_win};
  flash_comm::launch_kernel_ex(
      (void *)kernels::kernel_compute_dispatch_layout<kNumWarps>, grid_dim,
      block_dim, kernel_args, smem_size, stream,
      flash_comm::internal::get_cga_cluster_size(), true);
  CUDA_CHECK(cudaGetLastError());
}

#define FLASH_COMM_DISPATCH_HIDDEN_CASE(HIDDEN, ARGS)                          \
  case HIDDEN:                                                                 \
    detail::dispatch_internode_cuda_hidden<HIDDEN>(ARGS);                      \
    break;

#define FLASH_COMM_COMBINE_HIDDEN_CASE(HIDDEN, ARGS)                           \
  case HIDDEN:                                                                 \
    detail::combine_internode_cuda_hidden<HIDDEN>(ARGS);                       \
    break;

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
    cudaStream_t stream) {
  FLASH_CHECK(num_ranks % local_world_size == 0);
  FLASH_CHECK(num_qps > 0);
  FLASH_CHECK(num_sm > 0);
  FLASH_CHECK(dispatch_pipeline_chunks > 0 &&
              dispatch_pipeline_chunks <= kMaxDispatchPipelineChunks)
      << "dispatch_pipeline_chunks must be in [1, "
      << kMaxDispatchPipelineChunks << "]";

  const detail::DispatchInternodeArgs args{rdma_rail_send_win_handle,
                                           num_tokens_per_rank,
                                           node_topk_indices,
                                           node_topk_send_mask,
                                           node_token_dst_scatter_indices,
                                           has_weight,
                                           recv_x_ptrs,
                                           recv_weights_ptrs,
                                           recv_topk_scatter_indices_ptrs,
                                           max_slot_num_token,
                                           num_experts_per_rank,
                                           rank,
                                           num_ranks,
                                           local_world_size,
                                           max_recv_tokens,
                                           num_sm,
                                           dev_comm_host,
                                           num_qps,
                                           dtype,
                                           weight_dtype,
                                           offset_dtype,
                                           topk,
                                           dispatch_pipeline_chunks,
                                           logical_token_range,
                                           stream};
  switch (hidden_size) {
    SUPPORTED_HIDDEN_SIZES(FLASH_COMM_DISPATCH_HIDDEN_CASE, args)
  default:
    throw std::runtime_error("Unsupported hidden size: " +
                             std::to_string(hidden_size));
  }
}

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
    cudaStream_t stream) {
  FLASH_CHECK(num_ranks % local_world_size == 0);
  FLASH_CHECK(num_qps > 0);
  FLASH_CHECK(num_sm > 0);
  FLASH_CHECK(combine_pipeline_chunks > 0 &&
              combine_pipeline_chunks <= kMaxCombinePipelineChunks)
      << "combine_pipeline_chunks must be in [1, " << kMaxCombinePipelineChunks
      << "]";

  const detail::CombineInternodeArgs args{combine_x_ptrs,
                                          combine_weight_ptrs,
                                          rdma_rail_send_win_handle,
                                          output,
                                          output_weight,
                                          local_topk_indices,
                                          local_topk_send_mask,
                                          local_token_dst_scatter_indices,
                                          num_tokens_per_rank,
                                          has_weight,
                                          num_token,
                                          max_slot_num_token,
                                          topk,
                                          num_experts_per_rank,
                                          rank,
                                          num_ranks,
                                          local_world_size,
                                          num_sm,
                                          dev_comm_host,
                                          num_qps,
                                          dtype,
                                          weight_dtype,
                                          offset_dtype,
                                          combine_pipeline_chunks,
                                          logical_token_range,
                                          stream};
  switch (hidden_size) {
    SUPPORTED_HIDDEN_SIZES(FLASH_COMM_COMBINE_HIDDEN_CASE, args)
  default:
    throw std::runtime_error("Unsupported hidden size: " +
                             std::to_string(hidden_size));
  }
}

#undef FLASH_COMM_COMBINE_HIDDEN_CASE
#undef FLASH_COMM_DISPATCH_HIDDEN_CASE

} // namespace internode
} // namespace ep
} // namespace flash_comm

#else

template <typename T>
__device__ __forceinline__ T *offset_ptr(T *base, int32_t index,
                                         int32_t stride) {
  return base + static_cast<int64_t>(index) * stride;
}

struct RDMARailWindowDesc {
  // One NCCL symmetric-memory window is shared by dispatch and combine on one
  // rail (same local_rank across nodes). The window has 3 * nnodes fixed-size
  // slots:
  //
  //   slot [0, nnodes):
  //     dispatch source slots, indexed by source node.
  //     Each slot contains:
  //       x                         : [max_slot_num_token, hidden] bf16
  //       topk_indices              : [max_slot_num_token, topk] int32
  //       topk_send_mask            : [max_slot_num_token, topk] int32
  //       token_dst_scatter_indices : [max_slot_num_token, topk] int32
  //       topk_weights              : [max_slot_num_token, topk] fp32
  //
  //     dispatch source slots. Dispatch metadata is copied into layout_desc
  //     before combine, so combine does not depend on this region in-place.
  //
  //   slot [nnodes, 2 * nnodes):
  //     outgoing combine scratch, indexed by owner node. A contributor writes
  //     the partial it will send to owner_node into:
  //       nnodes + owner_node
  //
  //   slot [2 * nnodes, 3 * nnodes):
  //     incoming combine partials for local reduce, indexed by contributor
  //     node. A contributor_node writes into the owner rank's:
  //       2 * nnodes + contributor_node
  //
  //     Outgoing and incoming state are separate regions. This keeps
  //     send/compute overlap without flush and gives reduce one isolated slot
  //     per contributor while keeping the mapping directly tied to ownership.
  //
  // slot_stride_bytes is the allocation-time stride and always includes the
  // weight region. Runtime has_weight only controls whether kernels read/write
  // the weight payload, never the physical slot spacing.
  ncclWindow_t win;
  char *local_base_cache;
  size_t slot_stride_bytes;
  int32_t max_slot_num_token;
  int32_t nnodes;
  size_t token_region_bytes;
  size_t meta_region_bytes;
  size_t weight_region_off;

  __host__ __device__ __forceinline__ int32_t source_slot(int32_t node) const {
    return node;
  }

  __host__ __device__ __forceinline__ int32_t
  outgoing_partial_slot(int32_t owner_node) const {
    return nnodes + owner_node;
  }

  __host__ __device__ __forceinline__ int32_t
  incoming_partial_slot(int32_t contributor_node) const {
    return 2 * nnodes + contributor_node;
  }

  __host__ __device__ __forceinline__ int32_t
  local_partial_slot(int32_t owner_node) const {
    return incoming_partial_slot(owner_node);
  }

  __device__ __forceinline__ void set_local_base(char *base) {
    local_base_cache = base;
  }

  __device__ __forceinline__ char *local_base() const {
    return local_base_cache;
  }

  __device__ __forceinline__ char *slot_ptr(int32_t slot) const {
    return local_base() + static_cast<size_t>(slot) * slot_stride_bytes;
  }

  __device__ __forceinline__ char *source_slot_ptr(int32_t node) const {
    return slot_ptr(source_slot(node));
  }

  __device__ __forceinline__ char *
  reduce_partial_slot_ptr(int32_t contributor_node) const {
    return slot_ptr(incoming_partial_slot(contributor_node));
  }
};

static inline RDMARailWindowDesc
make_rdma_rail_window_desc(ncclWindow_t win, int32_t max_slot_num_token,
                           int32_t hidden_size, int32_t topk, int32_t nnodes) {
  const auto layout = ::flash_comm::ep::internode::rdma_rail_send_layout_desc(
      max_slot_num_token, hidden_size, topk, nnodes);
  return RDMARailWindowDesc{win,
                            nullptr,
                            layout.slot_stride_bytes,
                            max_slot_num_token,
                            nnodes,
                            layout.token_region_bytes,
                            layout.meta_region_bytes,
                            layout.topk_weights_offset};
}

__device__ __forceinline__ void
init_rdma_desc_local_base(RDMARailWindowDesc &desc) {
  desc.set_local_base(
      reinterpret_cast<char *>(ncclGetLocalPointer(desc.win, 0)));
}

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kTopk, bool kHasWeight>
__device__ __forceinline__ void
rdma_rail_slot_ptrs(const RDMARailWindowDesc &desc, int32_t slot, token_t *&x,
                    offset_t *&topk_indices, int32_t *&topk_send_mask,
                    offset_t *&token_dst_scatter, weight_t *&topk_weights) {
  char *base = desc.slot_ptr(slot);
  x = reinterpret_cast<token_t *>(base);
  base += desc.token_region_bytes;
  topk_indices = reinterpret_cast<offset_t *>(base);
  base += desc.meta_region_bytes;
  topk_send_mask = reinterpret_cast<int32_t *>(base);
  base += desc.meta_region_bytes;
  token_dst_scatter = reinterpret_cast<offset_t *>(base);
  base += desc.meta_region_bytes;
  topk_weights = kHasWeight ? reinterpret_cast<weight_t *>(base) : nullptr;
}

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kTopk, bool kHasWeight>
__device__ __forceinline__ void rdma_dispatch_source_slot_ptrs(
    const RDMARailWindowDesc &desc, int32_t slot, int32_t num_tokens,
    token_t *&x, offset_t *&topk_indices, int32_t *&topk_send_mask,
    offset_t *&token_dst_scatter, weight_t *&topk_weights) {
  char *base = desc.slot_ptr(slot);
  x = reinterpret_cast<token_t *>(base);
  base += desc.token_region_bytes;
  const size_t meta_bytes =
      static_cast<size_t>(num_tokens) * kTopk * sizeof(int32_t);
  topk_indices = reinterpret_cast<offset_t *>(base);
  base += meta_bytes;
  topk_send_mask = reinterpret_cast<int32_t *>(base);
  base += meta_bytes;
  token_dst_scatter = reinterpret_cast<offset_t *>(base);
  topk_weights = kHasWeight ? reinterpret_cast<weight_t *>(
                                  desc.slot_ptr(slot) + desc.weight_region_off)
                            : nullptr;
}

// Ring prefetch: this rank always sends its own source slot. The byte ranges
// must therefore be based on this rank's local input token count, not on the
// source node that this rank is currently consuming in the ring.
//
// Exactly one dispatch signal per (dst_node, qp, chunk) is emitted per call
// (bare signal when the payload slice is empty), so waiters can target
// snapshot + 1 for any num_qps / chunk config.
template <typename token_t, typename weight_t, int32_t kHiddenSize,
          int32_t kTopk, bool kHasWeight, typename Coop>
__device__ __forceinline__ void internode_producer_ring_put(
    ncclDevComm dev_comm, const RDMARailWindowDesc &desc, int32_t rank,
    int32_t local_world_size, int32_t dst_node, int32_t local_num_tokens,
    int32_t num_qps, int32_t qp_to_send, Coop coop,
    int32_t dispatch_pipeline_chunks) {
  const int32_t local_rank = rank % local_world_size;
  const int32_t my_node = rank / local_world_size;
  if (dst_node == my_node) {
    return;
  }
  if (qp_to_send < 0 || qp_to_send >= num_qps) {
    return;
  }
  const int32_t qp = qp_to_send;
  const ncclGinSignal_t completion_signal = dispatch_node_signal(my_node, 0);
  ncclGin qp_gin{dev_comm, kGinDispatchPutCtx + qp};
  const size_t src_off =
      static_cast<size_t>(desc.source_slot(my_node)) * desc.slot_stride_bytes;
  const int32_t peer = dst_node * local_world_size + local_rank;
  const size_t token_bytes =
      static_cast<size_t>(local_num_tokens) * kHiddenSize * sizeof(token_t);
  const size_t meta_bytes =
      static_cast<size_t>(local_num_tokens) * kTopk * sizeof(int32_t);
  const size_t meta_bundle_bytes = 3 * meta_bytes;
  const size_t weight_bytes =
      kHasWeight
          ? static_cast<size_t>(local_num_tokens) * kTopk * sizeof(weight_t)
          : 0;
  dispatch_pipeline_chunks =
      dispatch_pipeline_chunks < 1 ? 1 : dispatch_pipeline_chunks;

  if (dispatch_pipeline_chunks > 1) {
    // Chunked pipeline: tokens are split into chunks and each chunk is
    // striped across QPs at token granularity, so a chunk's slice of every
    // sub-region (token / 3x compact meta / weight) stays contiguous per QP.
    // One per-chunk signal releases consumers for exactly that token range.
    const size_t topk_indices_region_off = desc.token_region_bytes;
    const size_t topk_send_mask_region_off =
        topk_indices_region_off + meta_bytes;
    const size_t token_dst_scatter_region_off =
        topk_send_mask_region_off + meta_bytes;

    for (int32_t chunk = 0; chunk < dispatch_pipeline_chunks; ++chunk) {
      const int32_t token_begin = static_cast<int32_t>(
          (static_cast<int64_t>(local_num_tokens) * chunk) /
          dispatch_pipeline_chunks);
      const int32_t token_end = static_cast<int32_t>(
          (static_cast<int64_t>(local_num_tokens) * (chunk + 1)) /
          dispatch_pipeline_chunks);
      const int32_t chunk_num_tokens = token_end - token_begin;
      const int32_t qp_token_base = chunk_num_tokens / num_qps;
      const int32_t qp_token_rem = chunk_num_tokens - qp_token_base * num_qps;
      const int32_t qp_token_begin = token_begin + qp_token_base * qp +
                                     (qp < qp_token_rem ? qp : qp_token_rem);
      const int32_t qp_token_count =
          qp_token_base + (qp < qp_token_rem ? 1 : 0);
      const size_t token_region_off =
          static_cast<size_t>(qp_token_begin) * kHiddenSize * sizeof(token_t);
      const size_t token_slice_bytes =
          static_cast<size_t>(qp_token_count) * kHiddenSize * sizeof(token_t);
      const size_t meta_region_off =
          static_cast<size_t>(qp_token_begin) * kTopk * sizeof(int32_t);
      const size_t meta_slice_bytes =
          static_cast<size_t>(qp_token_count) * kTopk * sizeof(int32_t);
      const size_t weight_region_off =
          static_cast<size_t>(qp_token_begin) * kTopk * sizeof(weight_t);
      const size_t weight_slice_bytes =
          kHasWeight
              ? static_cast<size_t>(qp_token_count) * kTopk * sizeof(weight_t)
              : 0;

      const ncclGinSignal_t chunk_completion_signal =
          dispatch_node_signal(my_node, chunk);
      int32_t remaining_regions = (token_slice_bytes > 0 ? 1 : 0) +
                                  (meta_slice_bytes > 0 ? 3 : 0) +
                                  (weight_slice_bytes > 0 ? 1 : 0);
      if (remaining_regions == 0) {
        qp_gin.signal(ncclTeamWorld(dev_comm), peer,
                      ncclGin_SignalInc{chunk_completion_signal}, coop);
        continue;
      }

      gin_put_region_tail_signal(
          qp_gin, dev_comm, peer, desc.win, src_off + token_region_off,
          src_off + token_region_off, token_slice_bytes,
          chunk_completion_signal, remaining_regions, coop);
      gin_put_region_tail_signal(
          qp_gin, dev_comm, peer, desc.win,
          src_off + topk_indices_region_off + meta_region_off,
          src_off + topk_indices_region_off + meta_region_off, meta_slice_bytes,
          chunk_completion_signal, remaining_regions, coop);
      gin_put_region_tail_signal(
          qp_gin, dev_comm, peer, desc.win,
          src_off + topk_send_mask_region_off + meta_region_off,
          src_off + topk_send_mask_region_off + meta_region_off,
          meta_slice_bytes, chunk_completion_signal, remaining_regions, coop);
      gin_put_region_tail_signal(
          qp_gin, dev_comm, peer, desc.win,
          src_off + token_dst_scatter_region_off + meta_region_off,
          src_off + token_dst_scatter_region_off + meta_region_off,
          meta_slice_bytes, chunk_completion_signal, remaining_regions, coop);
      if constexpr (kHasWeight) {
        gin_put_region_tail_signal(
            qp_gin, dev_comm, peer, desc.win,
            src_off + desc.weight_region_off + weight_region_off,
            src_off + desc.weight_region_off + weight_region_off,
            weight_slice_bytes, chunk_completion_signal, remaining_regions,
            coop);
      }
    }
    return;
  }

  size_t token_slice_off = 0;
  size_t token_slice_bytes = token_bytes;
  size_t weight_slice_off = 0;
  size_t weight_slice_bytes = weight_bytes;
  size_t meta_slice_off = 0;
  size_t meta_slice_bytes = meta_bundle_bytes;
  split_qp_range(token_bytes, num_qps, qp, token_slice_off, token_slice_bytes);
  split_qp_range(weight_bytes, num_qps, qp, weight_slice_off,
                 weight_slice_bytes);
  split_qp_range(meta_bundle_bytes, num_qps, qp, meta_slice_off,
                 meta_slice_bytes);
  int32_t remaining_regions = (token_slice_bytes > 0 ? 1 : 0) +
                              (weight_slice_bytes > 0 ? 1 : 0) +
                              (meta_slice_bytes > 0 ? 1 : 0);
  if (remaining_regions == 0) {
    qp_gin.signal(ncclTeamWorld(dev_comm), peer,
                  ncclGin_SignalInc{completion_signal}, coop);
    return;
  }

  gin_put_region_tail_signal(qp_gin, dev_comm, peer, desc.win,
                             src_off + token_slice_off,
                             src_off + token_slice_off, token_slice_bytes,
                             completion_signal, remaining_regions, coop);
  if constexpr (kHasWeight) {
    gin_put_region_tail_signal(
        qp_gin, dev_comm, peer, desc.win,
        src_off + desc.weight_region_off + weight_slice_off,
        src_off + desc.weight_region_off + weight_slice_off, weight_slice_bytes,
        completion_signal, remaining_regions, coop);
  }
  gin_put_region_tail_signal(qp_gin, dev_comm, peer, desc.win,
                             src_off + desc.token_region_bytes + meta_slice_off,
                             src_off + desc.token_region_bytes + meta_slice_off,
                             meta_slice_bytes, completion_signal,
                             remaining_regions, coop);
}
namespace smem {
constexpr int32_t kTMAAlignment = 128;

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kNumStages>
struct DispatchIntraNodeSmem {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each TMA slot must be kTMAAlignment-byte aligned");

  uint64_t mbar_full[kNumStages];
  uint64_t mbar_empty[kNumStages];

  token_t *recv_x_ptrs[flash_comm::kMaxWorldSize];
  weight_t *recv_weights_ptrs[flash_comm::kMaxWorldSize];
  offset_t *recv_topk_scatter_indices_ptrs[flash_comm::kMaxWorldSize];

  alignas(kTMAAlignment) token_t tma_buffer[kNumStages][kHiddenSize];
};

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kMaxSmemSize, int32_t Lo = 1,
          int32_t Hi = 64>
struct MaxDispatchStages {
  static constexpr int32_t Mid = (Lo + Hi + 1) / 2;
  static constexpr bool fits =
      sizeof(DispatchIntraNodeSmem<token_t, weight_t, offset_t, kHiddenSize,
                                   Mid>) <= kMaxSmemSize;
  static constexpr int32_t value =
      (Lo >= Hi)
          ? Lo
          : (fits ? MaxDispatchStages<token_t, weight_t, offset_t, kHiddenSize,
                                      kMaxSmemSize, Mid, Hi>::value
                  : MaxDispatchStages<token_t, weight_t, offset_t, kHiddenSize,
                                      kMaxSmemSize, Lo, Mid - 1>::value);
};

template <typename token_t, typename weight_t, int32_t kHiddenSize,
          int32_t kNumLoadStages, int32_t kNumStoreStages,
          int32_t kNumWGPerBlock>
struct CombineIntraNodeSmem {
  static_assert((kHiddenSize * sizeof(token_t)) % kTMAAlignment == 0,
                "Each TMA slot must be kTMAAlignment-byte aligned");
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

template <typename token_t, typename weight_t, int32_t kHiddenSize,
          int32_t kMaxSmemSize, int32_t kNumStoreStages, int32_t kNumWGPerBlock,
          int32_t Lo = 1, int32_t Hi = 64>
struct MaxCombineLoadStages {
  static constexpr int32_t Mid = (Lo + Hi + 1) / 2;
  static constexpr bool fits =
      sizeof(CombineIntraNodeSmem<token_t, weight_t, kHiddenSize, Mid,
                                  kNumStoreStages, kNumWGPerBlock>) <=
      kMaxSmemSize;
  static constexpr int32_t value =
      (Lo >= Hi)
          ? Lo
          : (fits ? MaxCombineLoadStages<token_t, weight_t, kHiddenSize,
                                         kMaxSmemSize, kNumStoreStages,
                                         kNumWGPerBlock, Mid, Hi>::value
                  : MaxCombineLoadStages<token_t, weight_t, kHiddenSize,
                                         kMaxSmemSize, kNumStoreStages,
                                         kNumWGPerBlock, Lo, Mid - 1>::value);
};
} // namespace smem

template <typename token_t, typename weight_t, typename offset_t,
          int32_t kHiddenSize, int32_t kTopk, int32_t kNumStages,
          int32_t kNumConsumerGroups, bool kHasWeight, bool kRanged>
void __global__ __launch_bounds__((1 + kNumConsumerGroups) * WARP_SIZE, 1)
    kernel_dispatch_internode(
        ncclDevComm dev_comm, RDMARailWindowDesc rdma_desc,
        int32_t *num_tokens_per_rank, offset_t *node_topk_indices,
        int32_t *node_topk_send_mask, offset_t *node_token_dst_scatter_indices,
        int32_t num_experts_per_rank, int32_t rank, int32_t num_ranks,
        int32_t local_world_size, int32_t max_recv_tokens, int32_t num_qps,
        void *recv_x_ptrs, void **recv_weights_ptrs,
        offset_t **recv_topk_scatter_indices_ptrs,
        int32_t dispatch_pipeline_chunks, const int32_t *logical_token_range) {
  if constexpr (kRanged) {
    if (logical_token_range[1] <= logical_token_range[0]) {
      return;
    }
  }
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  using smem_t = smem::DispatchIntraNodeSmem<token_t, weight_t, offset_t,
                                             kHiddenSize, kNumStages>;
  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);

  static_assert(kTopk <= WARP_SIZE, "kTopk must be <= WARP_SIZE");
  static_assert(kNumConsumerGroups > 0, "kNumConsumerGroups must be > 0");
  static_assert(kNumStages % kNumConsumerGroups == 0,
                "kNumStages must be divisible by kNumConsumerGroups");

  const int thread_id = threadIdx.x;
  const int block_id = blockIdx.x;
  const int num_block = gridDim.x;
  const int warp_id = thread_id / WARP_SIZE;
  const int lane_id = thread_id % WARP_SIZE;
  constexpr int32_t kStorePipe = 2;
  const int32_t local_rank = rank % local_world_size;
  const int32_t my_node = rank / local_world_size;
  const int32_t nnodes = num_ranks / local_world_size;
  const int32_t dispatch_chunks =
      dispatch_pipeline_chunks < 1 ? 1 : dispatch_pipeline_chunks;
  init_rdma_desc_local_base(rdma_desc);

  const bool is_producer_warp = (warp_id == 0);
  const bool is_consumer_warp = (warp_id >= 1 && warp_id <= kNumConsumerGroups);
  const int32_t consumer_group_id = warp_id - 1;

  uint64_t *mbar_full_ptr = smem.mbar_full;
  uint64_t *mbar_empty_ptr = smem.mbar_empty;
  token_t **smem_recv_x_ptrs = smem.recv_x_ptrs;
  weight_t **smem_recv_weights_ptrs = smem.recv_weights_ptrs;
  offset_t **smem_recv_topk_scatter_indices_ptrs =
      smem.recv_topk_scatter_indices_ptrs;

  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < kNumStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
      initialize_barrier(mbar_empty_ptr + i, 1);
    }
  }

  if (thread_id < local_world_size) {
    smem_recv_x_ptrs[thread_id] =
        reinterpret_cast<token_t **>(recv_x_ptrs)[thread_id];
    smem_recv_weights_ptrs[thread_id] =
        reinterpret_cast<weight_t **>(recv_weights_ptrs)[thread_id];
    smem_recv_topk_scatter_indices_ptrs[thread_id] =
        recv_topk_scatter_indices_ptrs[thread_id];
  }

  __syncthreads();

  constexpr int32_t num_bytes_per_token = kHiddenSize * sizeof(token_t);
  auto producer_pipe_state = PipelineState<kNumStages>(0, 1, 0);
  const uint32_t is_leader_lane = elect_one_sync();

  auto wait_remote_source_node = [&](ncclCoopWarpSpan &gin_warp,
                                     int32_t src_node) {
    if (src_node == my_node) {
      return;
    }
    const ncclGinSignal_t wait_sig = dispatch_node_signal(src_node, 0);
    for (int32_t qp = 0; qp < num_qps; ++qp) {
      ncclGin wait_gin{dev_comm, kGinDispatchPutCtx + qp};
      wait_gin.waitSignal(gin_warp, wait_sig, kGinSignalStepTarget);
    }
    __threadfence_system();
    __syncwarp();
  };
  auto wait_remote_source_chunk = [&](ncclCoopWarpSpan &gin_warp,
                                      int32_t src_node, int32_t chunk) {
    if (src_node == my_node) {
      return;
    }
    chunk = chunk < 0
                ? 0
                : (chunk >= dispatch_chunks ? dispatch_chunks - 1 : chunk);
    const ncclGinSignal_t wait_sig = dispatch_node_signal(src_node, chunk);
    for (int32_t qp = 0; qp < num_qps; ++qp) {
      ncclGin wait_gin{dev_comm, kGinDispatchPutCtx + qp};
      wait_gin.waitSignal(gin_warp, wait_sig, kGinSignalStepTarget);
    }
    __threadfence_system();
    __syncwarp();
  };

  if (is_producer_warp) {
    ncclCoopWarpSpan gin_warp(0, 1, 0);
    for (int32_t node_offset = 0; node_offset < nnodes; ++node_offset) {
      const int32_t src_node = (my_node + node_offset) % nnodes;
      const int32_t src_global_rank = src_node * local_world_size + local_rank;
      const int32_t src_num_token = num_tokens_per_rank[src_global_rank];
      const int32_t local_num_token = num_tokens_per_rank[rank];
      const int32_t prefetch_dst_node =
          (my_node + nnodes - ((node_offset + 1) % nnodes)) % nnodes;

      for (int32_t qp = block_id; qp < num_qps; qp += num_block) {
        internode_producer_ring_put<token_t, weight_t, kHiddenSize, kTopk,
                                    kHasWeight>(
            dev_comm, rdma_desc, rank, local_world_size, prefetch_dst_node,
            local_num_token, num_qps, qp, gin_warp, dispatch_chunks);
      }
      if (dispatch_chunks == 1) {
        wait_remote_source_node(gin_warp, src_node);
      }

      token_t *cur_x = nullptr;
      offset_t *cur_topk = nullptr;
      int32_t *cur_mask = nullptr;
      offset_t *cur_scatter = nullptr;
      weight_t *cur_weights = nullptr;
      rdma_dispatch_source_slot_ptrs<token_t, weight_t, offset_t, kHiddenSize,
                                     kTopk, kHasWeight>(
          rdma_desc, rdma_desc.source_slot(src_node), src_num_token, cur_x,
          cur_topk, cur_mask, cur_scatter, cur_weights);

      int32_t waited_chunk = -1;
      for (int token_offset = block_id; token_offset < src_num_token;
           token_offset += num_block) {
        if (dispatch_chunks > 1 && src_node != my_node) {
          // Producer chunks are [floor(n*c/C), floor(n*(c+1)/C)).  Use the
          // right edge so boundary tokens wait on the chunk that owns them.
          const int32_t chunk = static_cast<int32_t>(
              ((static_cast<int64_t>(token_offset) + 1) * dispatch_chunks - 1) /
              src_num_token);
          if (chunk != waited_chunk) {
            wait_remote_source_chunk(gin_warp, src_node, chunk);
            waited_chunk = chunk;
          }
        }
        token_t *src_gmem_ptr = offset_ptr(cur_x, token_offset, kHiddenSize);
        uint64_t *cur_mbar_empty_ptr =
            mbar_empty_ptr + producer_pipe_state.index();
        uint64_t *cur_mbar_full_ptr =
            mbar_full_ptr + producer_pipe_state.index();

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
    }

    // Chunked waits above only cover chunks whose tokens this CTA consumed,
    // so some of this step's signal increments may still be in flight when
    // the CTAs exit. Block 0 drains every (node, qp, chunk) signal emitted
    // this step so the counters are settled before the stream-ordered barrier
    // kernel resets them. (With dispatch_chunks == 1 the per-node waits above
    // already cover every emitted signal.)
    if (dispatch_chunks > 1 && block_id == 0) {
      for (int32_t src_node = 0; src_node < nnodes; ++src_node) {
        if (src_node == my_node) {
          continue;
        }
        for (int32_t qp = 0; qp < num_qps; ++qp) {
          ncclGin wait_gin{dev_comm, kGinDispatchPutCtx + qp};
          for (int32_t chunk = 0; chunk < dispatch_chunks; ++chunk) {
            const ncclGinSignal_t wait_sig =
                dispatch_node_signal(src_node, chunk);
            wait_gin.waitSignal(gin_warp, wait_sig, kGinSignalStepTarget);
          }
        }
      }
    }
  } else if (is_consumer_warp) {
    int32_t node_base_count = 0;
    auto consumer_pipe_state = PipelineState<kNumStages>(0, 0, 0);
    auto release_pipe_state = PipelineState<kNumStages>(0, 0, 0);
    consumer_pipe_state += consumer_group_id;
    release_pipe_state += consumer_group_id;
    int32_t tokens_processed = 0;

    for (int32_t node_offset = 0; node_offset < nnodes; ++node_offset) {
      const int32_t src_node = (my_node + node_offset) % nnodes;
      const int32_t src_global_rank = src_node * local_world_size + local_rank;
      const int32_t src_num_token = num_tokens_per_rank[src_global_rank];
      const int32_t node_block_tokens =
          (block_id < src_num_token)
              ? ((src_num_token - 1 - block_id) / num_block + 1)
              : 0;

      token_t *cur_x = nullptr;
      offset_t *cur_topk = nullptr;
      int32_t *cur_mask = nullptr;
      offset_t *cur_scatter = nullptr;
      weight_t *cur_weights = nullptr;
      rdma_dispatch_source_slot_ptrs<token_t, weight_t, offset_t, kHiddenSize,
                                     kTopk, kHasWeight>(
          rdma_desc, rdma_desc.source_slot(src_node), src_num_token, cur_x,
          cur_topk, cur_mask, cur_scatter, cur_weights);

      // CTA partitioning is identical on producer and consumer: CTA block_id
      // owns token offsets block_id + node_ordinal * num_block for every source
      // node, and both sides visit src_node in the same ring order.  Producer
      // uses one continuous PipelineState across nodes, so the mbar slot for a
      // CTA-local token is keyed by global ordinal:
      //   global_ordinal = node_base_count + node_ordinal.
      //
      // Consumer groups split that same global sequence by modulo.  The first
      // token for this node must satisfy
      //   global_ordinal % kNumConsumerGroups == consumer_group_id.
      // The offset is non-zero only when the previous node's token count is not
      // a multiple of kNumConsumerGroups; node boundaries do not reset the
      // pipe.
      const int32_t first_node_ordinal =
          (consumer_group_id + kNumConsumerGroups -
           (node_base_count % kNumConsumerGroups)) %
          kNumConsumerGroups;

      for (int32_t node_ordinal = first_node_ordinal;
           node_ordinal < node_block_tokens;
           node_ordinal += kNumConsumerGroups) {
        const int32_t token_offset = block_id + node_ordinal * num_block;
        uint64_t *cur_mbar_full_ptr =
            mbar_full_ptr + consumer_pipe_state.index();

        int32_t my_expert_idx = -1;
        int32_t my_target_rank = -1;
        int32_t my_is_need_send = 0;
        int32_t my_store_idx = -1;
        weight_t my_weight = 0;

        tma_store_wait<kStorePipe - 1>();
        __syncwarp();

        if (tokens_processed >= kStorePipe) {
          uint64_t *release_mbar_empty_ptr =
              mbar_empty_ptr + release_pipe_state.index();
          if (is_leader_lane) {
            arrive_barrier(release_mbar_empty_ptr);
          }
          release_pipe_state += kNumConsumerGroups;
        }

        wait_barrier(cur_mbar_full_ptr, consumer_pipe_state.phase());

        if (lane_id < kTopk) {
          my_expert_idx = cur_topk[token_offset * kTopk + lane_id];
          my_target_rank = my_expert_idx / num_experts_per_rank;
          my_is_need_send = cur_mask[token_offset * kTopk + lane_id];
          my_store_idx = cur_scatter[token_offset * kTopk + lane_id];
          const int64_t meta_idx =
              (static_cast<int64_t>(src_node) * rdma_desc.max_slot_num_token +
               token_offset) *
                  kTopk +
              lane_id;
          node_topk_indices[meta_idx] = my_expert_idx;
          node_topk_send_mask[meta_idx] = my_is_need_send;
          node_token_dst_scatter_indices[meta_idx] = my_store_idx;
          if constexpr (kHasWeight) {
            my_weight = cur_weights[token_offset * kTopk + lane_id];
          }
        }
        __syncwarp();

        const int32_t my_target_local_rank =
            (my_target_rank >= 0) ? (my_target_rank % local_world_size) : -1;
        const int32_t dst_node =
            (my_target_rank >= 0) ? (my_target_rank / local_world_size) : -1;
        int32_t should_send =
            (my_target_rank >= 0 && my_target_rank < num_ranks &&
             my_is_need_send && my_store_idx >= 0 &&
             my_store_idx < max_recv_tokens && dst_node == my_node &&
             my_target_local_rank >= 0 &&
             my_target_local_rank < local_world_size);
        uint32_t send_mask = __ballot_sync(0xffffffff, should_send);

        void *src_smem_ptr = smem.tma_buffer[consumer_pipe_state.index()];
        uint32_t remaining_mask = send_mask;
        while (remaining_mask) {
          int32_t send_lane = __ffs(remaining_mask) - 1;
          remaining_mask &= (remaining_mask - 1);

          int32_t target_local_rank =
              __shfl_sync(0xffffffff, my_target_local_rank, send_lane);
          int32_t target_global_rank =
              __shfl_sync(0xffffffff, my_target_rank, send_lane);
          int32_t store_idx = __shfl_sync(0xffffffff, my_store_idx, send_lane);

          token_t *dst_gmem_ptr = offset_ptr(
              smem_recv_x_ptrs[target_local_rank], store_idx, kHiddenSize);
          if (is_leader_lane) {
            tma_copy_1d_s2g(src_smem_ptr, dst_gmem_ptr, num_bytes_per_token);
          }

          if (lane_id < kTopk) {
            if constexpr (kHasWeight) {
              weight_t *dst_weight_ptr =
                  offset_ptr(smem_recv_weights_ptrs[target_local_rank],
                             store_idx, kTopk) +
                  lane_id;
              *dst_weight_ptr = my_weight;
            }
            offset_t cur_index = my_store_idx;
            cur_index = (my_target_rank == target_global_rank) ? cur_index : -1;
            offset_t *dst_index_ptr =
                offset_ptr(
                    smem_recv_topk_scatter_indices_ptrs[target_local_rank],
                    store_idx, kTopk) +
                lane_id;
            *dst_index_ptr = cur_index;
          }
          __syncwarp();
        }

        tma_store_arrive();
        __syncwarp();
        consumer_pipe_state += kNumConsumerGroups;
        tokens_processed++;
      }

      node_base_count += node_block_tokens;
    }
    tma_store_wait<0>();
    __syncwarp();
  }
}

} // namespace kernels

template <int32_t kHiddenSize>
void detail::dispatch_internode_cuda_hidden(
    const detail::DispatchInternodeArgs &args) {
  auto rdma_rail_send_win_handle = args.rdma_rail_send_win_handle;
  auto num_tokens_per_rank = args.num_tokens_per_rank;
  auto node_topk_indices = args.node_topk_indices;
  auto node_topk_send_mask = args.node_topk_send_mask;
  auto node_token_dst_scatter_indices = args.node_token_dst_scatter_indices;
  auto has_weight = args.has_weight;
  auto recv_x_ptrs = args.recv_x_ptrs;
  auto recv_weights_ptrs = args.recv_weights_ptrs;
  auto recv_topk_scatter_indices_ptrs = args.recv_topk_scatter_indices_ptrs;
  auto max_slot_num_token = args.max_slot_num_token;
  auto num_experts_per_rank = args.num_experts_per_rank;
  auto rank = args.rank;
  auto num_ranks = args.num_ranks;
  auto local_world_size = args.local_world_size;
  auto max_recv_tokens = args.max_recv_tokens;
  auto num_sm = args.num_sm;
  auto dev_comm_host = args.dev_comm_host;
  auto num_qps = args.num_qps;
  auto dtype = args.dtype;
  auto weight_dtype = args.weight_dtype;
  auto offset_dtype = args.offset_dtype;
  auto topk = args.topk;
  auto dispatch_pipeline_chunks = args.dispatch_pipeline_chunks;
  auto logical_token_range = args.logical_token_range;
  auto stream = args.stream;
  const int32_t nnodes = num_ranks / local_world_size;

  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  ncclWindow_t rdma_rail_send_win =
      reinterpret_cast<ncclWindow_t>(rdma_rail_send_win_handle);
  const kernels::RDMARailWindowDesc rdma_desc =
      kernels::make_rdma_rail_window_desc(
          rdma_rail_send_win, max_slot_num_token, kHiddenSize, topk, nnodes);

  // Empirically tuned on H800: 3 consumer warps hide the per-token scatter
  // cost behind one TMA g2s producer warp (4 warps/CTA keeps occupancy high
  // enough for the RDMA warp to make progress), and 12 pipeline stages cover
  // the g2s latency at hidden sizes up to 8K while still fitting in smem
  // (MaxDispatchStages clamps when they do not fit).
  constexpr int32_t kNumConsumerGroups = 3;
  constexpr int32_t kMaxSmemSize = flash_comm::kMaxSmemBytes;
  constexpr int32_t kPreferredStages = 12;

  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
      DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
        DISPATCH_TOPK(topk, kTopk, {
          constexpr int32_t kMaxFitStages = kernels::smem::MaxDispatchStages<
              token_t, weight_t, offset_t, kHiddenSize, kMaxSmemSize>::value;
          constexpr int32_t kCapped = kMaxFitStages < kPreferredStages
                                          ? kMaxFitStages
                                          : kPreferredStages;
          constexpr int32_t kNumStages =
              (kCapped / kNumConsumerGroups) * kNumConsumerGroups;
          static_assert(kNumStages >= kNumConsumerGroups,
                        "kNumStages too small");

          constexpr int32_t kNumWarps = 1 + kNumConsumerGroups;
          constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
          dim3 block_dim(kNumThreads);
          dim3 grid_dim(num_sm);
          using smem_t =
              kernels::smem::DispatchIntraNodeSmem<token_t, weight_t, offset_t,
                                                   kHiddenSize, kNumStages>;
          constexpr int32_t smem_size = sizeof(smem_t);

          const bool ranged = logical_token_range != nullptr;
          DISPATCH_BOOL(has_weight, kHasWeight, {
            DISPATCH_BOOL(ranged, kRanged, {
              auto kernel_fn = kernels::kernel_dispatch_internode<
                  token_t, weight_t, offset_t, kHiddenSize, kTopk, kNumStages,
                  kNumConsumerGroups, kHasWeight, kRanged>;
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_size));
              kernels::RDMARailWindowDesc rdma_desc_arg = rdma_desc;
              offset_t *node_topk_indices_arg =
                  reinterpret_cast<offset_t *>(node_topk_indices);
              int32_t *node_topk_send_mask_arg = node_topk_send_mask;
              offset_t *node_token_dst_scatter_indices_arg =
                  reinterpret_cast<offset_t *>(node_token_dst_scatter_indices);
              int32_t num_experts_per_rank_arg = num_experts_per_rank;
              int32_t rank_arg = rank;
              int32_t num_ranks_arg = num_ranks;
              int32_t local_world_size_arg = local_world_size;
              int32_t max_recv_tokens_arg = max_recv_tokens;
              int32_t num_qps_arg = num_qps;
              int32_t dispatch_pipeline_chunks_arg = dispatch_pipeline_chunks;
              const int32_t *logical_token_range_arg = logical_token_range;
              offset_t **recv_topk_ptrs_arg =
                  reinterpret_cast<offset_t **>(recv_topk_scatter_indices_ptrs);
              void *kernel_args[] = {&dev_comm,
                                     &rdma_desc_arg,
                                     &num_tokens_per_rank,
                                     &node_topk_indices_arg,
                                     &node_topk_send_mask_arg,
                                     &node_token_dst_scatter_indices_arg,
                                     &num_experts_per_rank_arg,
                                     &rank_arg,
                                     &num_ranks_arg,
                                     &local_world_size_arg,
                                     &max_recv_tokens_arg,
                                     &num_qps_arg,
                                     &recv_x_ptrs,
                                     &recv_weights_ptrs,
                                     &recv_topk_ptrs_arg,
                                     &dispatch_pipeline_chunks_arg,
                                     &logical_token_range_arg};
              flash_comm::launch_kernel_ex(
                  reinterpret_cast<const void *>(kernel_fn), grid_dim,
                  block_dim, kernel_args, smem_size, stream,
                  flash_comm::internal::get_cga_cluster_size(), false);
            });
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

namespace kernels {

template <typename token_t, typename weight_t, typename offset_t, int32_t kTopk,
          int32_t kHiddenSize, int32_t kNumLoadStages, int32_t kNumStoreStages,
          int32_t kNumWarps, int32_t kWarpsPerWG, int32_t kElemsPerThread,
          bool kHasWeight, bool kRanged>
void __global__ __launch_bounds__(kNumWarps *WARP_SIZE, 1)
    kernel_combine_internode(ncclDevComm dev_comm, RDMARailWindowDesc rdma_desc,
                             int32_t *local_topk_indices,
                             int32_t *local_topk_send_mask,
                             int32_t *local_token_dst_scatter_indices,
                             int32_t *num_tokens_per_rank, void *combine_x_ptrs,
                             void *combine_weight_ptrs, int32_t num_token,
                             int32_t num_experts_per_rank, int32_t rank,
                             int32_t num_ranks, int32_t local_world_size,
                             int32_t num_qps, int32_t combine_pipeline_chunks,
                             const int32_t *logical_token_range) {
  if constexpr (kRanged) {
    if (logical_token_range[1] <= logical_token_range[0]) {
      return;
    }
  }
  static_assert(kWarpsPerWG > 1, "kWarpsPerWG must be greater than 1");
  extern __shared__ __align__(1024) uint8_t smem_buffer[];
  using smem_t =
      smem::CombineIntraNodeSmem<token_t, weight_t, kHiddenSize, kNumLoadStages,
                                 kNumStoreStages, kNumWarps / kWarpsPerWG>;
  constexpr int32_t kNumWGPerBlock = kNumWarps / kWarpsPerWG;
  constexpr int32_t kElemsPerInt4 = sizeof(int4) / sizeof(token_t);
  constexpr int32_t kHiddenSizeInt4 = kHiddenSize / kElemsPerInt4;
  constexpr int32_t kNumConsumerThreadsPerWG = (kWarpsPerWG - 1) * WARP_SIZE;

  const int32_t thread_id = threadIdx.x;
  const int32_t block_id = blockIdx.x;
  const int32_t num_block = gridDim.x;
  const int32_t warp_id = thread_id / WARP_SIZE;
  const int32_t lane_id = thread_id % WARP_SIZE;
  const int32_t warp_group_id = warp_id / kWarpsPerWG;
  constexpr int32_t kNumComputeWarps = kNumWGPerBlock * kWarpsPerWG;
  constexpr int32_t kWeightWarpId = kNumComputeWarps;
  constexpr int32_t kRdmaWarpId = kNumComputeWarps + int(kHasWeight);
  const int32_t consumer_tid_in_wg =
      thread_id % (WARP_SIZE * kWarpsPerWG) - WARP_SIZE;
  const int32_t num_bytes_per_token = kHiddenSize * sizeof(token_t);
  const int32_t local_rank = rank % local_world_size;
  const int32_t my_node = rank / local_world_size;
  const int32_t nnodes = num_ranks / local_world_size;
  const int32_t global_warp_group_id =
      block_id * kNumWGPerBlock + warp_group_id;
  const int32_t total_warp_groups = num_block * kNumWGPerBlock;
  init_rdma_desc_local_base(rdma_desc);
  const int32_t my_node_rank_begin = my_node * local_world_size;
  const int32_t my_node_rank_end = my_node_rank_begin + local_world_size;

  auto &smem = *reinterpret_cast<smem_t *>(smem_buffer);
  auto &wg_smem =
      smem.warp_group_smem[warp_group_id < kNumWGPerBlock ? warp_group_id : 0];
  uint64_t *mbar_full_ptr = wg_smem.mbar_full;
  uint64_t *mbar_empty_ptr = wg_smem.mbar_empty;
  token_t **x_ptrs = smem.x_ptrs;
  weight_t **weight_ptrs = smem.weight_ptrs;

  const bool is_tma_load_warp =
      (warp_id % kWarpsPerWG == 0) && (warp_id < kNumWGPerBlock * kWarpsPerWG);
  const bool is_consumer_warp =
      (warp_id < kNumWGPerBlock * kWarpsPerWG) && !is_tma_load_warp;
  const int32_t num_rdma_warps = num_qps < num_block ? num_qps : num_block;
  const int32_t rdma_warp_id = block_id;
  const bool is_rdma_warp =
      (rdma_warp_id < num_rdma_warps && warp_id == kRdmaWarpId);

  if (is_tma_load_warp && elect_one_sync()) {
    for (int32_t i = 0; i < kNumLoadStages; ++i) {
      initialize_barrier(mbar_full_ptr + i, 1);
      initialize_barrier(mbar_empty_ptr + i, kWarpsPerWG - 1);
    }
  }
  if (thread_id < local_world_size) {
    x_ptrs[thread_id] = reinterpret_cast<token_t **>(combine_x_ptrs)[thread_id];
    if constexpr (kHasWeight) {
      weight_ptrs[thread_id] =
          reinterpret_cast<weight_t **>(combine_weight_ptrs)[thread_id];
    }
  }
  __syncthreads();

  auto send_partial_range = [&](int32_t owner_node, int32_t token_begin,
                                int32_t token_end, int32_t weight_begin,
                                int32_t weight_end, bool signal_completion) {
    if (!is_rdma_warp || owner_node == my_node) {
      return;
    }
    // Use a GIN coop barrier id after the TMA consumer barriers [1,
    // kNumWGPerBlock] owned by named_barrier_arrive_and_wait(warp_group_id +
    // 1).
    ncclCoopWarpSpan gin_warp(kRdmaWarpId, 1, kNumWGPerBlock);
    const int32_t src_rank = owner_node * local_world_size + local_rank;
    const int32_t src_num_token = num_tokens_per_rank[src_rank];
    token_begin = token_begin < 0 ? 0 : token_begin;
    token_end = token_end > src_num_token ? src_num_token : token_end;
    token_end = token_end < token_begin ? token_begin : token_end;
    const int32_t chunk_num_token = token_end - token_begin;
    weight_begin = weight_begin < 0 ? 0 : weight_begin;
    weight_end = weight_end > src_num_token ? src_num_token : weight_end;
    weight_end = weight_end < weight_begin ? weight_begin : weight_end;
    const int32_t weight_num_token = weight_end - weight_begin;
    const int32_t peer = owner_node * local_world_size + local_rank;
    const int32_t transfer_slot = rdma_desc.outgoing_partial_slot(owner_node);
    const int32_t dst_slot = rdma_desc.incoming_partial_slot(my_node);
    const size_t src_slot_off =
        static_cast<size_t>(transfer_slot) * rdma_desc.slot_stride_bytes;
    const size_t dst_slot_off =
        static_cast<size_t>(dst_slot) * rdma_desc.slot_stride_bytes;
    const size_t token_region_off =
        static_cast<size_t>(token_begin) * kHiddenSize * sizeof(token_t);
    const size_t token_bytes =
        static_cast<size_t>(chunk_num_token) * kHiddenSize * sizeof(token_t);
    const size_t weight_region_slice_off =
        static_cast<size_t>(weight_begin) * kTopk * sizeof(weight_t);
    const size_t weight_bytes =
        kHasWeight
            ? static_cast<size_t>(weight_num_token) * kTopk * sizeof(weight_t)
            : 0;

    for (int32_t rdma_qp = rdma_warp_id; rdma_qp < num_qps;
         rdma_qp += num_rdma_warps) {
      const ncclGinSignal_t completion_signal =
          combine_node_signal(nnodes, my_node);
      ncclGin qp_gin{dev_comm, kGinCombinePutCtx + rdma_qp};
      size_t token_slice_off = 0;
      size_t token_slice_bytes = token_bytes;
      size_t weight_slice_off = 0;
      size_t weight_slice_bytes = weight_bytes;
      split_qp_range(token_bytes, num_qps, rdma_qp, token_slice_off,
                     token_slice_bytes);
      split_qp_range(weight_bytes, num_qps, rdma_qp, weight_slice_off,
                     weight_slice_bytes);
      const int32_t nonempty_regions =
          (token_slice_bytes > 0 ? 1 : 0) + (weight_slice_bytes > 0 ? 1 : 0);
      if (nonempty_regions == 0) {
        if (signal_completion) {
          qp_gin.signal(ncclTeamWorld(dev_comm), peer,
                        ncclGin_SignalInc{completion_signal}, gin_warp);
        }
        continue;
      }
      int32_t remaining_regions =
          nonempty_regions + (signal_completion ? 0 : 1);
      gin_put_region_tail_signal(
          qp_gin, dev_comm, peer, rdma_desc.win,
          dst_slot_off + token_region_off + token_slice_off,
          src_slot_off + token_region_off + token_slice_off, token_slice_bytes,
          completion_signal, remaining_regions, gin_warp);
      if constexpr (kHasWeight) {
        gin_put_region_tail_signal(
            qp_gin, dev_comm, peer, rdma_desc.win,
            dst_slot_off + rdma_desc.weight_region_off +
                weight_region_slice_off + weight_slice_off,
            src_slot_off + rdma_desc.weight_region_off +
                weight_region_slice_off + weight_slice_off,
            weight_slice_bytes, completion_signal, remaining_regions, gin_warp);
      }
    }
  };

  auto send_partial = [&](int32_t owner_node) {
    const int32_t src_rank = owner_node * local_world_size + local_rank;
    const int32_t src_num_token = num_tokens_per_rank[src_rank];
    send_partial_range(owner_node, 0, src_num_token, 0, src_num_token, true);
  };

  auto producer_pipe_state = PipelineState<kNumLoadStages>(0, 1, 0);
  auto consumer_pipe_state = PipelineState<kNumLoadStages>(0, 0, 0);

  auto combine_one_node_range = [&](int32_t owner_node, int32_t token_begin,
                                    int32_t token_end) {
    const int32_t partial_slot =
        (owner_node == my_node) ? rdma_desc.local_partial_slot(my_node)
                                : rdma_desc.outgoing_partial_slot(owner_node);
    char *partial_slot_ptr = rdma_desc.slot_ptr(partial_slot);
    const int64_t meta_off =
        static_cast<int64_t>(owner_node) * rdma_desc.max_slot_num_token * kTopk;
    offset_t *cur_topk =
        reinterpret_cast<offset_t *>(local_topk_indices) + meta_off;
    int32_t *cur_mask = local_topk_send_mask + meta_off;
    offset_t *cur_scatter =
        reinterpret_cast<offset_t *>(local_token_dst_scatter_indices) +
        meta_off;
    token_t *partial_x = reinterpret_cast<token_t *>(partial_slot_ptr);
    weight_t *partial_weight =
        kHasWeight ? reinterpret_cast<weight_t *>(partial_slot_ptr +
                                                  rdma_desc.weight_region_off)
                   : nullptr;
    const int32_t src_rank = owner_node * local_world_size + local_rank;
    const int32_t src_num_token = num_tokens_per_rank[src_rank];
    token_begin = token_begin < 0 ? 0 : token_begin;
    token_end = token_end > src_num_token ? src_num_token : token_end;
    token_end = token_end < token_begin ? token_begin : token_end;

    if constexpr (kHasWeight) {
      if (warp_id == kWeightWarpId) {
        const int32_t total_weight_threads = num_block * WARP_SIZE;
        const int32_t global_weight_thread_id =
            token_begin * kTopk + lane_id + block_id * WARP_SIZE;
        for (int32_t i = global_weight_thread_id; i < token_end * kTopk;
             i += total_weight_threads) {
          const int32_t token_offset = i / kTopk;
          const int32_t topk_idx = i % kTopk;
          const int32_t expert_idx = cur_topk[token_offset * kTopk + topk_idx];
          const int32_t expert_rank = expert_idx / num_experts_per_rank;
          const int32_t expert_local_rank = expert_rank - my_node_rank_begin;
          const int32_t scatter = cur_scatter[token_offset * kTopk + topk_idx];
          weight_t value = 0;
          if (expert_rank >= my_node_rank_begin &&
              expert_rank < my_node_rank_end && scatter >= 0) {
            value =
                weight_ptrs[expert_local_rank]
                           [static_cast<int64_t>(scatter) * kTopk + topk_idx];
          }
          partial_weight[token_offset * kTopk + topk_idx] = value;
        }
      }
    }

    float acc[kElemsPerThread];
    PRAGMA_UNROLL
    for (int32_t i = 0; i < kElemsPerThread; ++i) {
      acc[i] = 0.0f;
    }

    if (is_tma_load_warp) {
      for (int32_t token_offset = token_begin + global_warp_group_id;
           token_offset < token_end; token_offset += total_warp_groups) {
        int32_t expert_rank_lane = 0;
        int32_t scatter_lane = 0;
        int32_t is_valid_lane = 0;
        if (lane_id < kTopk) {
          const int32_t expert_idx = cur_topk[token_offset * kTopk + lane_id];
          expert_rank_lane = expert_idx / num_experts_per_rank;
          scatter_lane = cur_scatter[token_offset * kTopk + lane_id];
          is_valid_lane =
              (expert_rank_lane >= my_node_rank_begin &&
               expert_rank_lane < my_node_rank_end &&
               cur_mask[token_offset * kTopk + lane_id] && scatter_lane >= 0);
        }
        uint32_t valid_mask = __ballot_sync(0xffffffff, is_valid_lane);
        while (valid_mask) {
          const int32_t send_lane = __ffs(valid_mask) - 1;
          valid_mask &= (valid_mask - 1);
          const int32_t expert_rank =
              __shfl_sync(0xffffffff, expert_rank_lane, send_lane);
          const int32_t scatter =
              __shfl_sync(0xffffffff, scatter_lane, send_lane);
          const int32_t expert_local_rank = expert_rank - my_node_rank_begin;
          uint64_t *empty = mbar_empty_ptr + producer_pipe_state.index();
          uint64_t *full = mbar_full_ptr + producer_pipe_state.index();
          wait_barrier(empty, producer_pipe_state.phase());
          void *src_ptr =
              offset_ptr(x_ptrs[expert_local_rank], scatter, kHiddenSize);
          void *dst_smem_ptr =
              wg_smem.tma_load_buffer[producer_pipe_state.index()];
          if (elect_one_sync()) {
            mbar_arrive_and_set_barrier_transaction_bytes(full,
                                                          num_bytes_per_token);
            tma_copy_1d_g2s(src_ptr, full, dst_smem_ptr, num_bytes_per_token);
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
      const uint32_t is_leader_lane = elect_one_sync();
      int32_t token_iter = 0;
      int32_t store_buffer_idx = 0;
      for (int32_t token_offset = token_begin + global_warp_group_id;
           token_offset < token_end;
           token_offset += total_warp_groups, ++token_iter) {
        int32_t scatter_lane = -1;
        int32_t is_valid_lane = 0;
        if (lane_id < kTopk) {
          const int32_t expert_idx = cur_topk[token_offset * kTopk + lane_id];
          const int32_t expert_rank = expert_idx / num_experts_per_rank;
          scatter_lane = cur_scatter[token_offset * kTopk + lane_id];
          is_valid_lane =
              (expert_rank >= my_node_rank_begin &&
               expert_rank < my_node_rank_end &&
               cur_mask[token_offset * kTopk + lane_id] && scatter_lane >= 0);
        }
        const int32_t valid_count =
            __popc(__ballot_sync(0xffffffff, is_valid_lane));
        for (int32_t j = 0; j < valid_count; ++j) {
          uint64_t *empty = mbar_empty_ptr + consumer_pipe_state.index();
          uint64_t *full = mbar_full_ptr + consumer_pipe_state.index();
          wait_barrier(full, consumer_pipe_state.phase());
          token_t *smem_ptr = reinterpret_cast<token_t *>(
              wg_smem.tma_load_buffer[consumer_pipe_state.index()]);
          int4 *smem_ptr_int4 = reinterpret_cast<int4 *>(smem_ptr);
          constexpr int32_t kInt4PerThread =
              (kHiddenSizeInt4 + kNumConsumerThreadsPerWG - 1) /
              kNumConsumerThreadsPerWG;
          PRAGMA_UNROLL
          for (int32_t idx = 0; idx < kInt4PerThread; ++idx) {
            const int32_t smem_idx =
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
          if (is_leader_lane) {
            arrive_barrier(empty);
          }
          ++consumer_pipe_state;
          __syncwarp();
        }

        const int32_t warp_id_in_consumer_wg = consumer_tid_in_wg / WARP_SIZE;
        if (warp_id_in_consumer_wg == 0 && token_iter >= kNumStoreStages) {
          tma_store_wait<kNumStoreStages - 1>();
        }
        named_barrier_arrive_and_wait(kNumConsumerThreadsPerWG,
                                      warp_group_id + 1);
        int4 *tma_store_ptr_int4 = reinterpret_cast<int4 *>(
            wg_smem.tma_store_buffer[store_buffer_idx]);
        constexpr int32_t kInt4PerThreadStore =
            (kHiddenSizeInt4 + kNumConsumerThreadsPerWG - 1) /
            kNumConsumerThreadsPerWG;
        PRAGMA_UNROLL
        for (int32_t idx = 0; idx < kInt4PerThreadStore; ++idx) {
          const int32_t dst_idx =
              consumer_tid_in_wg + idx * kNumConsumerThreadsPerWG;
          if (dst_idx < kHiddenSizeInt4) {
            PRAGMA_UNROLL
            for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
              converter.bf162[k] = __float22bfloat162_rn(
                  make_float2(acc[idx * kElemsPerInt4 + k * 2],
                              acc[idx * kElemsPerInt4 + k * 2 + 1]));
              acc[idx * kElemsPerInt4 + k * 2] = 0.0f;
              acc[idx * kElemsPerInt4 + k * 2 + 1] = 0.0f;
            }
            tma_store_ptr_int4[dst_idx] = converter.vec;
          }
        }
        fence_async_shared();
        named_barrier_arrive_and_wait(kNumConsumerThreadsPerWG,
                                      warp_group_id + 1);
        if (warp_id_in_consumer_wg == 0) {
          if (is_leader_lane) {
            token_t *dst_ptr = offset_ptr(partial_x, token_offset, kHiddenSize);
            tma_copy_1d_s2g(reinterpret_cast<void *>(tma_store_ptr_int4),
                            dst_ptr, num_bytes_per_token);
            tma_store_arrive();
          }
          __syncwarp();
        }
        store_buffer_idx = (store_buffer_idx + 1) % kNumStoreStages;
      }
      tma_store_wait<0>();
    }
  };

  auto combine_one_node = [&](int32_t owner_node) {
    const int32_t src_rank = owner_node * local_world_size + local_rank;
    const int32_t src_num_token = num_tokens_per_rank[src_rank];
    combine_one_node_range(owner_node, 0, src_num_token);
  };

  auto combine_remote_node_chunked = [&](int32_t owner_node) {
    const int32_t src_rank = owner_node * local_world_size + local_rank;
    const int32_t src_num_token = num_tokens_per_rank[src_rank];
    int32_t node_chunks = combine_pipeline_chunks;
    if (src_num_token > 0 && node_chunks > src_num_token) {
      node_chunks = src_num_token;
    }
    if (node_chunks <= 1) {
      return false;
    }

    int32_t prev_begin = 0;
    int32_t prev_end = 0;
    bool have_prev_chunk = false;
    for (int32_t chunk = 0; chunk < node_chunks; ++chunk) {
      const int32_t chunk_begin = static_cast<int32_t>(
          (static_cast<int64_t>(src_num_token) * chunk) / node_chunks);
      const int32_t chunk_end = static_cast<int32_t>(
          (static_cast<int64_t>(src_num_token) * (chunk + 1)) / node_chunks);
      if (have_prev_chunk) {
        send_partial_range(owner_node, prev_begin, prev_end, 0, 0, false);
      }
      combine_one_node_range(owner_node, chunk_begin, chunk_end);
      __threadfence_system();
      cooperative_groups::this_grid().sync();
      prev_begin = chunk_begin;
      prev_end = chunk_end;
      have_prev_chunk = true;
    }

    if (have_prev_chunk) {
      send_partial_range(owner_node, prev_begin, prev_end, 0, src_num_token,
                         true);
    }
    return true;
  };

  int32_t prev_node = -1;
  for (int32_t node_iter = 0; node_iter < nnodes; ++node_iter) {
    const int32_t owner_node = (my_node + 1 + node_iter) % nnodes;
    if (owner_node != my_node && combine_pipeline_chunks > 1) {
      if (prev_node >= 0) {
        send_partial(prev_node);
        prev_node = -1;
      }
      if (combine_remote_node_chunked(owner_node)) {
        continue;
      }
    }
    if (prev_node >= 0) {
      send_partial(prev_node);
    }

    combine_one_node(owner_node);
    // Publish partial writes before the previous owner-node partial is sent in
    // the next iteration. The RDMA warp is separate from TMA producer/consumer
    // warps, so GIN progress can overlap with the next combine phase.
    __threadfence_system();
    cooperative_groups::this_grid().sync();
    prev_node = owner_node;
  }

  if (is_rdma_warp) {
    ncclCoopWarpSpan gin_warp(kRdmaWarpId, 1, kNumWGPerBlock);
    for (int32_t node = 0; node < nnodes; ++node) {
      if (node == my_node) {
        continue;
      }
      // Each contributor signals with its node id after putting its partial
      // into our incoming reduce slot. Only after all contributors arrive can
      // we reduce across node slots.
      const ncclGinSignal_t wait_sig = combine_node_signal(nnodes, node);
      for (int32_t qp = 0; qp < num_qps; ++qp) {
        ncclGin wait_gin{dev_comm, kGinCombinePutCtx + qp};
        wait_gin.waitSignal(gin_warp, wait_sig, kGinSignalStepTarget);
      }
    }
    __threadfence_system();
    __syncwarp();
  }
}

template <typename token_t, typename weight_t, int32_t kHiddenSize,
          int32_t kTopk, bool kHasWeight, bool kRanged>
void __global__ kernel_combine_internode_reduce(
    RDMARailWindowDesc rdma_desc, void *output, void *output_weight,
    int32_t num_token, int32_t *num_tokens_per_rank, int32_t rank,
    int32_t num_ranks, int32_t local_world_size,
    const int32_t *logical_token_range) {
  constexpr int32_t kElemsPerInt4 = sizeof(int4) / sizeof(token_t);
  constexpr int32_t kHiddenSizeInt4 = kHiddenSize / kElemsPerInt4;
  const int32_t nnodes = num_ranks / local_world_size;
  int32_t token_output_begin = 0;
  if constexpr (kRanged) {
    if (logical_token_range[1] <= logical_token_range[0]) {
      return;
    }
    token_output_begin = max(0, min(logical_token_range[0], num_token));
    num_token = num_tokens_per_rank[rank];
  }
  init_rdma_desc_local_base(rdma_desc);
  const int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride = blockDim.x * gridDim.x;
  const int32_t token_vec_count = num_token * kHiddenSizeInt4;
  const int32_t weight_count = kHasWeight ? num_token * kTopk : 0;
  const int32_t total_work =
      token_vec_count > weight_count ? token_vec_count : weight_count;

  union {
    int4 vec;
    __nv_bfloat162 bf162[4];
  } converter;
  union {
    int4 vec;
    __nv_bfloat162 bf162[4];
  } out_converter;

  for (int32_t linear = tid; linear < total_work; linear += stride) {
    if (linear < token_vec_count) {
      const int32_t token = linear / kHiddenSizeInt4;
      const int32_t vec_idx = linear - token * kHiddenSizeInt4;
      float acc[kElemsPerInt4];
      PRAGMA_UNROLL
      for (int32_t i = 0; i < kElemsPerInt4; ++i) {
        acc[i] = 0.0f;
      }
      for (int32_t node = 0; node < nnodes; ++node) {
        char *slot = rdma_desc.reduce_partial_slot_ptr(node);
        int4 *src_vec =
            reinterpret_cast<int4 *>(reinterpret_cast<token_t *>(slot) +
                                     static_cast<int64_t>(token) * kHiddenSize);
        converter.vec = src_vec[vec_idx];
        PRAGMA_UNROLL
        for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
          float2 fp32_vec2 = __bfloat1622float2(converter.bf162[k]);
          acc[k * 2] += fp32_vec2.x;
          acc[k * 2 + 1] += fp32_vec2.y;
        }
      }
      PRAGMA_UNROLL
      for (int32_t k = 0; k < kElemsPerInt4 / 2; ++k) {
        out_converter.bf162[k] =
            __float22bfloat162_rn(make_float2(acc[k * 2], acc[k * 2 + 1]));
      }
      int4 *dst_vec = reinterpret_cast<int4 *>(
          reinterpret_cast<token_t *>(output) +
          static_cast<int64_t>(token_output_begin + token) * kHiddenSize);
      dst_vec[vec_idx] = out_converter.vec;
    }

    if constexpr (kHasWeight) {
      if (linear < weight_count) {
        const int32_t token = linear / kTopk;
        const int32_t topk_idx = linear - token * kTopk;
        const int64_t weight_offset =
            static_cast<int64_t>(token) * kTopk + topk_idx;
        float acc = 0.0f;
        for (int32_t node = 0; node < nnodes; ++node) {
          char *slot = rdma_desc.reduce_partial_slot_ptr(node);
          weight_t *partial_weight =
              reinterpret_cast<weight_t *>(slot + rdma_desc.weight_region_off);
          acc += partial_weight[weight_offset];
        }
        const int64_t output_weight_offset =
            static_cast<int64_t>(token_output_begin + token) * kTopk + topk_idx;
        reinterpret_cast<weight_t *>(output_weight)[output_weight_offset] =
            static_cast<weight_t>(acc);
      }
    }
  }
}

} // namespace kernels

template <int32_t kHiddenSize>
void detail::combine_internode_cuda_hidden(
    const detail::CombineInternodeArgs &args) {
  auto combine_x_ptrs = args.combine_x_ptrs;
  auto combine_weight_ptrs = args.combine_weight_ptrs;
  auto rdma_rail_send_win_handle = args.rdma_rail_send_win_handle;
  auto output = args.output;
  auto output_weight = args.output_weight;
  auto local_topk_indices = args.local_topk_indices;
  auto local_topk_send_mask = args.local_topk_send_mask;
  auto local_token_dst_scatter_indices = args.local_token_dst_scatter_indices;
  auto num_tokens_per_rank = args.num_tokens_per_rank;
  auto has_weight = args.has_weight;
  auto num_token = args.num_token;
  auto max_slot_num_token = args.max_slot_num_token;
  auto topk = args.topk;
  auto num_experts_per_rank = args.num_experts_per_rank;
  auto rank = args.rank;
  auto num_ranks = args.num_ranks;
  auto local_world_size = args.local_world_size;
  auto num_sm = args.num_sm;
  auto dev_comm_host = args.dev_comm_host;
  auto num_qps = args.num_qps;
  auto dtype = args.dtype;
  auto weight_dtype = args.weight_dtype;
  auto offset_dtype = args.offset_dtype;
  auto combine_pipeline_chunks = args.combine_pipeline_chunks;
  auto logical_token_range = args.logical_token_range;
  auto stream = args.stream;
  const int32_t nnodes = num_ranks / local_world_size;
  ncclDevComm dev_comm = *static_cast<const ncclDevComm *>(dev_comm_host);
  ncclWindow_t rdma_win =
      reinterpret_cast<ncclWindow_t>(rdma_rail_send_win_handle);
  const kernels::RDMARailWindowDesc rdma_desc =
      kernels::make_rdma_rail_window_desc(rdma_win, max_slot_num_token,
                                          kHiddenSize, topk, nnodes);

  // Empirically tuned on H800: 2 warp groups of (1 TMA load + 8 consumer)
  // warps saturate smem bandwidth for the fp32 accumulate; 2 store stages
  // double-buffer the s2g path. kElemsPerThread sizes the per-thread fp32
  // accumulator: the 8 consumer warps (256 threads) of one warp group split
  // kHiddenSize elements, so 64 elems/thread covers hidden sizes up to
  // 256 * 64 = 16K before register pressure forces spills.
  constexpr int32_t kNumStoreStages = 2;
  constexpr int32_t kElemsPerThread = 64;
  constexpr int32_t kWarpsPerWG = 9;
  constexpr int32_t kNumWGPerBlock = 2;
  constexpr int32_t kMaxSmemSize = flash_comm::kMaxSmemBytes;

  DISPATCH_TOKEN_DTYPE(dtype, token_t, {
    DISPATCH_WEIGHT_DTYPE(weight_dtype, weight_t, {
      DISPATCH_OFFSET_TYPE(offset_dtype, offset_t, {
        DISPATCH_TOPK(topk, kTopk, {
          const bool ranged = logical_token_range != nullptr;
          DISPATCH_BOOL(has_weight, kHasWeight, {
            DISPATCH_BOOL(ranged, kRanged, {
              constexpr int32_t kPreferredLoadStages = 6;
              constexpr int32_t kMaxFitLoadStages =
                  kernels::smem::MaxCombineLoadStages<
                      token_t, weight_t, kHiddenSize, kMaxSmemSize,
                      kNumStoreStages, kNumWGPerBlock>::value;
              constexpr int32_t kNumLoadStages =
                  kMaxFitLoadStages < kPreferredLoadStages
                      ? kMaxFitLoadStages
                      : kPreferredLoadStages;
              static_assert(kNumLoadStages >= 2,
                            "combine_internode: shared memory too small for at "
                            "least two load stages");
              static_assert(
                  (kHiddenSize * static_cast<int32_t>(sizeof(token_t))) %
                          kernels::smem::kTMAAlignment ==
                      0,
                  "combine_internode: one token row must be TMA aligned");
              constexpr int32_t kNumComputeWarps = kNumWGPerBlock * kWarpsPerWG;
              constexpr int32_t kNumWarps =
                  kNumComputeWarps + int(kHasWeight) + 1;
              static_assert(
                  kNumComputeWarps % kWarpsPerWG == 0,
                  "combine_internode: compute warp groups must be complete");
              constexpr int32_t kNumThreads = kNumWarps * WARP_SIZE;
              using smem_t = kernels::smem::CombineIntraNodeSmem<
                  token_t, weight_t, kHiddenSize, kNumLoadStages,
                  kNumStoreStages, kNumWGPerBlock>;
              constexpr int32_t smem_size = sizeof(smem_t);
              static_assert(
                  smem_size <= kMaxSmemSize,
                  "combine_internode: smem_size exceeds kMaxSmemSize");
              auto kernel_fn = kernels::kernel_combine_internode<
                  token_t, weight_t, offset_t, kTopk, kHiddenSize,
                  kNumLoadStages, kNumStoreStages, kNumWarps, kWarpsPerWG,
                  kElemsPerThread, kHasWeight, kRanged>;
              CUDA_CHECK(cudaFuncSetAttribute(
                  kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                  smem_size));
              kernels::RDMARailWindowDesc rdma_desc_arg = rdma_desc;
              int32_t num_token_arg = num_token;
              int32_t num_experts_per_rank_arg = num_experts_per_rank;
              int32_t rank_arg = rank;
              int32_t num_ranks_arg = num_ranks;
              int32_t local_world_size_arg = local_world_size;
              int32_t num_qps_arg = num_qps;
              int32_t combine_pipeline_chunks_arg = combine_pipeline_chunks;
              const int32_t *logical_token_range_arg = logical_token_range;
              dim3 block_dim(kNumThreads);
              dim3 grid_dim(num_sm);
              void *kernel_args[] = {&dev_comm,
                                     &rdma_desc_arg,
                                     &local_topk_indices,
                                     &local_topk_send_mask,
                                     &local_token_dst_scatter_indices,
                                     &num_tokens_per_rank,
                                     &combine_x_ptrs,
                                     &combine_weight_ptrs,
                                     &num_token_arg,
                                     &num_experts_per_rank_arg,
                                     &rank_arg,
                                     &num_ranks_arg,
                                     &local_world_size_arg,
                                     &num_qps_arg,
                                     &combine_pipeline_chunks_arg,
                                     &logical_token_range_arg};
              flash_comm::launch_kernel_ex(
                  reinterpret_cast<const void *>(kernel_fn), grid_dim,
                  block_dim, kernel_args, smem_size, stream, 0, true);

              constexpr int32_t kReduceThreads = 256;
              constexpr int32_t kReduceElemsPerInt4 =
                  sizeof(int4) / sizeof(token_t);
              constexpr int32_t kReduceHiddenSizeInt4 =
                  kHiddenSize / kReduceElemsPerInt4;
              const int32_t launch_num_token = num_token;
              const int64_t token_vec_work =
                  static_cast<int64_t>(launch_num_token) *
                  kReduceHiddenSizeInt4;
              const int64_t weight_work =
                  kHasWeight ? static_cast<int64_t>(launch_num_token) * kTopk
                             : 0;
              const int64_t reduce_work =
                  token_vec_work > weight_work ? token_vec_work : weight_work;
              if (reduce_work > 0) {
                const int64_t min_grid =
                    (reduce_work + kReduceThreads - 1) / kReduceThreads;
                // The reduce is HBM-bandwidth bound; 2048 CTAs of 256 threads
                // already oversubscribe every current part (H800: 132 SMs),
                // so larger grids only add launch/tail overhead. Beyond this
                // the grid-stride loop covers the remaining work.
                constexpr int32_t max_hbm_grid = 2048;
                const int32_t reduce_grid = static_cast<int32_t>(
                    min_grid < max_hbm_grid ? min_grid : max_hbm_grid);
                kernels::kernel_combine_internode_reduce<
                    token_t, weight_t, kHiddenSize, kTopk, kHasWeight, kRanged>
                    <<<reduce_grid, kReduceThreads, 0, stream>>>(
                        rdma_desc, output, output_weight, num_token,
                        num_tokens_per_rank, rank, num_ranks, local_world_size,
                        logical_token_range);
              }
            });
          });
        });
      });
    });
  });
  CUDA_CHECK(cudaGetLastError());
}

} // namespace internode
} // namespace ep
} // namespace flash_comm

#define FLASH_COMM_INSTANTIATE_INTERNODE_HIDDEN_SIZE(HIDDEN)                   \
  namespace flash_comm {                                                       \
  namespace ep {                                                               \
  namespace internode {                                                        \
  template void detail::dispatch_internode_cuda_hidden<HIDDEN>(                \
      const detail::DispatchInternodeArgs &);                                  \
  template void detail::combine_internode_cuda_hidden<HIDDEN>(                 \
      const detail::CombineInternodeArgs &);                                   \
  }                                                                            \
  }                                                                            \
  }

#endif
