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

#include <cuda_runtime.h>
#include <nccl_device.h>

#include <algorithm>
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>
#include <limits>

#include "ep_world_barrier.cuh"
#include "flash_comm/common.h"
#include "flash_comm/ep/chunk_plan.h"

namespace flash_comm {
namespace ep {
namespace chunk_plan {
namespace {

constexpr int32_t kWarpSize = 32;
constexpr int32_t kGinPublicationSignal = 0;

struct WorkspaceLayout {
  size_t prefix_numel;
  size_t local_prefix;
  size_t gathered_prefix;
  size_t publish_prefix;
  size_t global_prefix;
  size_t schedule;
  size_t total;
};

__host__ __device__ WorkspaceLayout get_workspace_layout(int32_t num_chunks,
                                                         int32_t num_experts,
                                                         int32_t num_ranks) {
  const size_t prefix_numel =
      static_cast<size_t>(num_chunks + 1) * (num_experts + 1);
  const size_t local_prefix = 0;
  const size_t gathered_prefix = local_prefix + prefix_numel;
  const size_t publish_prefix =
      gathered_prefix + static_cast<size_t>(num_ranks) * prefix_numel;
  const size_t global_prefix =
      publish_prefix + static_cast<size_t>(num_ranks) * prefix_numel;
  const size_t schedule = global_prefix + prefix_numel;
  const size_t total = schedule + static_cast<size_t>(2 * num_chunks + 2);
  return {prefix_numel,   local_prefix,  gathered_prefix,
          publish_prefix, global_prefix, schedule,
          total};
}

template <typename T> __device__ __forceinline__ T warp_reduce_sum(T value) {
  value += __shfl_xor_sync(0xffffffffu, value, 16);
  value += __shfl_xor_sync(0xffffffffu, value, 8);
  value += __shfl_xor_sync(0xffffffffu, value, 4);
  value += __shfl_xor_sync(0xffffffffu, value, 2);
  value += __shfl_xor_sync(0xffffffffu, value, 1);
  return value;
}

__device__ __forceinline__ void
snapshot_gathered_prefix(const int32_t *gathered_prefix,
                         int32_t *rank_chunk_prefix, int32_t *publish_prefix,
                         size_t count, size_t first, size_t stride,
                         bool has_remote_peers) {
  for (size_t index = first; index < count; index += stride) {
    const int32_t value = gathered_prefix[index];
    rank_chunk_prefix[index] = value;
    if (has_remote_peers) {
      publish_prefix[index] = value;
    }
  }
}

__device__ __forceinline__ bool
uniform_range_fits_rank(const int32_t *global_prefix, int32_t begin,
                        int32_t end, int32_t num_experts, int32_t num_ranks,
                        int32_t target_rank, int32_t recv_capacity_tokens,
                        int32_t expert_alignment) {
  const int32_t lane = static_cast<int32_t>(threadIdx.x) & (kWarpSize - 1);
  const int32_t experts_per_rank = num_experts / num_ranks;
  const int32_t bins = num_experts + 1;
  const bool power_of_two = (expert_alignment & (expert_alignment - 1)) == 0;
  const int64_t mask = static_cast<int64_t>(expert_alignment - 1);
  int64_t aligned_recv = 0;
  for (int32_t local_expert = lane; local_expert < experts_per_rank;
       local_expert += kWarpSize) {
    const int32_t expert = target_rank * experts_per_rank + local_expert;
    const int32_t count =
        global_prefix[static_cast<int64_t>(end) * bins + expert] -
        global_prefix[static_cast<int64_t>(begin) * bins + expert];
    const int64_t value = static_cast<int64_t>(count) + mask;
    aligned_recv += power_of_two
                        ? (value & ~mask)
                        : (value / expert_alignment) * expert_alignment;
  }
  aligned_recv = warp_reduce_sum(aligned_recv);
  return __shfl_sync(0xffffffffu, aligned_recv, 0) <= recv_capacity_tokens;
}

__device__ __forceinline__ int32_t uniform_range_max_end(
    const int32_t *global_prefix, int32_t begin, int32_t num_chunks,
    int32_t num_experts, int32_t num_ranks, int32_t target_rank,
    int32_t recv_capacity_tokens, int32_t expert_alignment,
    int32_t suggested_count) {
  int32_t low = begin;
  int32_t high = num_chunks;
  if (suggested_count > 0) {
    int64_t distance = suggested_count;
    int32_t first =
        static_cast<int32_t>(min(static_cast<int64_t>(num_chunks),
                                 static_cast<int64_t>(begin) + distance));
    if (uniform_range_fits_rank(global_prefix, begin, first, num_experts,
                                num_ranks, target_rank, recv_capacity_tokens,
                                expert_alignment)) {
      low = first;
      if (first == num_chunks) {
        return first;
      }
      while (true) {
        distance *= 2;
        const int32_t candidate =
            static_cast<int32_t>(min(static_cast<int64_t>(num_chunks),
                                     static_cast<int64_t>(begin) + distance));
        if (uniform_range_fits_rank(global_prefix, begin, candidate,
                                    num_experts, num_ranks, target_rank,
                                    recv_capacity_tokens, expert_alignment)) {
          low = candidate;
          if (candidate == num_chunks) {
            return candidate;
          }
        } else {
          high = candidate - 1;
          break;
        }
      }
    } else {
      high = first - 1;
    }
  }
  int32_t left = low + 1;
  while (left <= high) {
    const int32_t mid = left + (high - left) / 2;
    if (uniform_range_fits_rank(global_prefix, begin, mid, num_experts,
                                num_ranks, target_rank, recv_capacity_tokens,
                                expert_alignment)) {
      low = mid;
      left = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return low;
}

__device__ void build_schedule(const int32_t *global_prefix, int32_t num_chunks,
                               int32_t num_experts, int32_t num_ranks,
                               int32_t recv_capacity_tokens,
                               int32_t expert_alignment, int32_t *step_begin,
                               int32_t *step_count, int32_t *schedule_valid,
                               int32_t *control) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t lane = tid & (kWarpSize - 1);
  const int32_t warp_id = tid / kWarpSize;
  const int32_t num_warps = blockDim.x / kWarpSize;
  int32_t *rank_max_end = control;
  int32_t *common_end = control + num_ranks;

  for (int32_t step = tid; step < num_chunks; step += blockDim.x) {
    step_begin[step] = 0;
    step_count[step] = 0;
  }
  if (tid == 0) {
    schedule_valid[0] = 1;
    bool has_routes = false;
    const int32_t bins = num_experts + 1;
    const int32_t *last =
        global_prefix + static_cast<int64_t>(num_chunks) * bins;
    for (int32_t expert = 0; expert < bins; ++expert) {
      has_routes |= last[expert] != 0;
    }
    common_end[0] = has_routes ? num_chunks : 0;
  }
  __syncthreads();

  const int32_t schedule_num_chunks = common_end[0];
  int32_t begin = 0;
  int32_t step = 0;
  int32_t previous_step_chunks = 0;
  while (begin < schedule_num_chunks) {
    // A 512-thread CTA has fewer warps than the maximum EP size.  Process
    // target ranks in round-robin order so every rank contributes its bound.
    for (int32_t target_rank = warp_id; target_rank < num_ranks;
         target_rank += num_warps) {
      const int32_t end = uniform_range_max_end(
          global_prefix, begin, schedule_num_chunks, num_experts, num_ranks,
          target_rank, recv_capacity_tokens, expert_alignment,
          previous_step_chunks);
      if (lane == 0) {
        rank_max_end[target_rank] = end;
      }
    }
    __syncthreads();
    if (tid == 0) {
      int32_t end = num_chunks;
      for (int32_t target = 0; target < num_ranks; ++target) {
        end = min(end, rank_max_end[target]);
      }
      common_end[0] = end;
      if (end > begin) {
        step_begin[step] = begin;
        step_count[step] = end - begin;
      }
    }
    __syncthreads();
    const int32_t end = common_end[0];
    if (end <= begin) {
      if (tid == 0) {
        schedule_valid[0] = 0;
      }
      break;
    }
    previous_step_chunks = end - begin;
    ++step;
    begin = end;
  }
}

__global__ __launch_bounds__(512, 1) void kernel_build_ep_chunk_plan(
    const int32_t *topk_indices, int32_t num_token, int32_t topk,
    int32_t num_experts, int32_t chunk_size, int32_t max_num_tokens,
    int32_t recv_capacity_tokens, int32_t expert_alignment, int32_t rank,
    int32_t num_ranks, int32_t lsa_world_size, int32_t gin_context,
    ncclDevComm dev_comm, ncclWindow_t workspace_win,
    int32_t *logical_token_ranges, int32_t *rank_chunk_prefix) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  extern __shared__ int32_t control[];
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t block = static_cast<int32_t>(blockIdx.x);
  const int32_t global_tid = block * static_cast<int32_t>(blockDim.x) + tid;
  const int32_t global_threads =
      static_cast<int32_t>(gridDim.x) * static_cast<int32_t>(blockDim.x);
  const int32_t num_chunks = 1 + (max_num_tokens - 1) / chunk_size;
  const int32_t bins = num_experts + 1;
  const WorkspaceLayout layout =
      get_workspace_layout(num_chunks, num_experts, num_ranks);
  int32_t *workspace =
      reinterpret_cast<int32_t *>(ncclGetLocalPointer(workspace_win, 0));
  int32_t *local_prefix = workspace + layout.local_prefix;
  int32_t *gathered_prefix = workspace + layout.gathered_prefix;
  int32_t *publish_prefix = workspace + layout.publish_prefix;
  int32_t *global_prefix = workspace + layout.global_prefix;
  int32_t *schedule = workspace + layout.schedule;
  int32_t *step_begin = schedule;
  int32_t *step_count = schedule + num_chunks;
  int32_t *schedule_valid = schedule + 2 * num_chunks;
  int32_t *schedule_ready = schedule_valid + 1;

  if (global_tid == 0 && num_ranks > lsa_world_size) {
    // Every rank resets before the gather world barrier below.  Root therefore
    // cannot publish the next call until all receivers have re-armed signal 0.
    ncclGin gin{dev_comm, gin_context};
    gin.resetSignal(kGinPublicationSignal);
  }

  for (size_t index = static_cast<size_t>(global_tid);
       index < layout.prefix_numel; index += global_threads) {
    local_prefix[index] = 0;
  }
  if (global_tid == 0) {
    schedule_ready[0] = 0;
  }
  grid.sync();

  // Route work is distributed across the full cooperative grid.  This keeps
  // large logical chunks from being limited to one CTA while preserving one
  // independent histogram row per chunk.
  const int64_t local_route_count = static_cast<int64_t>(num_token) * topk;
  for (int64_t route = global_tid; route < local_route_count;
       route += global_threads) {
    const int32_t expert = topk_indices[route];
    if (expert >= 0 && expert <= num_experts) {
      const int32_t token = static_cast<int32_t>(route / topk);
      const int32_t chunk = token / chunk_size;
      atomicAdd(local_prefix + static_cast<int64_t>(chunk + 1) * bins + expert,
                1);
    }
  }
  grid.sync();

  const size_t prefix_bytes = layout.prefix_numel * sizeof(int32_t);
  const size_t local_prefix_bytes = layout.local_prefix * sizeof(int32_t);
  const size_t root_row_bytes =
      (layout.gathered_prefix +
       static_cast<size_t>(rank) * layout.prefix_numel) *
      sizeof(int32_t);
  int32_t *root_row = nullptr;
  if (rank == 0) {
    root_row = gathered_prefix;
  } else if (rank < lsa_world_size) {
    root_row = reinterpret_cast<int32_t *>(
        ncclGetPeerPointer(workspace_win, root_row_bytes, 0));
  }
  for (int32_t expert = global_tid; expert < bins; expert += global_threads) {
    int32_t prefix = 0;
    if (root_row != nullptr) {
      root_row[expert] = 0;
    }
    for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
      const int64_t index = static_cast<int64_t>(chunk + 1) * bins + expert;
      prefix += local_prefix[index];
      local_prefix[index] = prefix;
      if (root_row != nullptr) {
        root_row[index] = prefix;
      }
    }
  }
  // LSA ranks publish their root row directly while computing the cumsum.
  // Remote ranks retain the local row as the source of one GIN put.
  __threadfence_system();
  grid.sync();

  if (block == 0) {
    if (rank >= lsa_world_size && tid == 0) {
      ncclGin gin{dev_comm, gin_context};
      gin.put(ncclTeamWorld(dev_comm), 0, workspace_win, root_row_bytes,
              workspace_win, local_prefix_bytes, prefix_bytes, ncclGin_None{},
              ncclGin_None{}, ncclCoopThread(), ncclGin_None{},
              cuda::thread_scope_thread, cuda::thread_scope_system);
    }
    ::flash_comm::ep::kernels::ep_world_barrier_after_puts(
        dev_comm, gin_context, gin_context, num_ranks > lsa_world_size);
  }
  grid.sync();

  if (rank == 0) {
    for (size_t index = static_cast<size_t>(global_tid);
         index < layout.prefix_numel; index += global_threads) {
      int32_t total = 0;
      for (int32_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
        total += gathered_prefix[static_cast<size_t>(src_rank) *
                                     layout.prefix_numel +
                                 index];
      }
      global_prefix[index] = total;
    }
  }
  grid.sync();

  if (rank == 0) {
    const size_t gathered_count =
        static_cast<size_t>(num_ranks) * layout.prefix_numel;
    const bool has_remote_peers = num_ranks > lsa_world_size;
    if (gridDim.x == 1) {
      build_schedule(global_prefix, num_chunks, num_experts, num_ranks,
                     recv_capacity_tokens, expert_alignment, step_begin,
                     step_count, schedule_valid, control);
      snapshot_gathered_prefix(
          gathered_prefix, rank_chunk_prefix, publish_prefix, gathered_count,
          static_cast<size_t>(tid), static_cast<size_t>(blockDim.x),
          has_remote_peers);
    } else if (block == 0) {
      build_schedule(global_prefix, num_chunks, num_experts, num_ranks,
                     recv_capacity_tokens, expert_alignment, step_begin,
                     step_count, schedule_valid, control);
    } else {
      const size_t copy_tid = static_cast<size_t>(block - 1) * blockDim.x + tid;
      const size_t copy_threads =
          static_cast<size_t>(gridDim.x - 1) * blockDim.x;
      // Peers can start the next build as soon as their publication completes
      // and overwrite their row in gathered_prefix.  Remote puts therefore
      // read an immutable root-owned snapshot which is not reused until the
      // root's release flush completes.
      snapshot_gathered_prefix(gathered_prefix, rank_chunk_prefix,
                               publish_prefix, gathered_count, copy_tid,
                               copy_threads, has_remote_peers);
    }
  }
  grid.sync();

  if (rank == 0) {
    const size_t schedule_bytes =
        static_cast<size_t>(2 * num_chunks + 1) * sizeof(int32_t);
    const size_t schedule_offset = layout.schedule * sizeof(int32_t);
    const size_t gathered_bytes = static_cast<size_t>(num_ranks) * prefix_bytes;
    const size_t gathered_offset = layout.gathered_prefix * sizeof(int32_t);
    const size_t publish_offset = layout.publish_prefix * sizeof(int32_t);
    const size_t ready_offset =
        (layout.schedule + static_cast<size_t>(2 * num_chunks + 1)) *
        sizeof(int32_t);
    const int32_t *local_publication_prefix =
        num_ranks > lsa_world_size ? publish_prefix : gathered_prefix;
    // The whole grid cooperates on every local peer copy.  Publication cost is
    // bandwidth-bound instead of one-CTA-per-peer latency-bound.
    for (int32_t dst_rank = 1; dst_rank < lsa_world_size; ++dst_rank) {
      int32_t *dst_schedule = reinterpret_cast<int32_t *>(
          ncclGetPeerPointer(workspace_win, schedule_offset, dst_rank));
      for (int32_t index = global_tid; index < 2 * num_chunks + 1;
           index += global_threads) {
        dst_schedule[index] = schedule[index];
      }
      int32_t *dst_gathered = reinterpret_cast<int32_t *>(
          ncclGetPeerPointer(workspace_win, gathered_offset, dst_rank));
      for (size_t index = static_cast<size_t>(global_tid);
           index < static_cast<size_t>(num_ranks) * layout.prefix_numel;
           index += global_threads) {
        dst_gathered[index] = local_publication_prefix[index];
      }
    }
    __threadfence_system();
    if (block == 0 && tid == 0) {
      schedule_ready[0] = 1;
      ncclGin gin{dev_comm, gin_context};
      for (int32_t dst_rank = lsa_world_size; dst_rank < num_ranks;
           ++dst_rank) {
        gin.put(ncclTeamWorld(dev_comm), dst_rank, workspace_win,
                schedule_offset, workspace_win, schedule_offset, schedule_bytes,
                ncclGin_None{}, ncclGin_None{}, ncclCoopThread(),
                ncclGin_None{}, cuda::thread_scope_thread,
                cuda::thread_scope_system);
        gin.put(ncclTeamWorld(dev_comm), dst_rank, workspace_win,
                gathered_offset, workspace_win, publish_offset, gathered_bytes,
                ncclGin_SignalInc{kGinPublicationSignal}, ncclGin_None{},
                ncclCoopThread(), ncclGin_None{}, cuda::thread_scope_thread,
                cuda::thread_scope_system);
      }
    }
    grid.sync();
    if (block == 0 && tid == 0) {
      // Remote completion signals publish both payload puts.  The flush is
      // still required independently before root reuses schedule/publish_prefix
      // as local GIN sources in the next build.
      ncclGin gin{dev_comm, gin_context};
      if (num_ranks > lsa_world_size) {
        gin.flush(ncclCoopThread(), cuda::memory_order_release);
      }
      for (int32_t dst_rank = 1; dst_rank < lsa_world_size; ++dst_rank) {
        int32_t *dst_ready = reinterpret_cast<int32_t *>(
            ncclGetPeerPointer(workspace_win, ready_offset, dst_rank));
        cuda::atomic_ref<int32_t, cuda::thread_scope_system>(*dst_ready)
            .store(1, cuda::memory_order_release);
      }
    }
  }
  if (rank != 0) {
    if (block == 0 && tid == 0) {
      if (rank < lsa_world_size) {
        cuda::atomic_ref<int32_t, cuda::thread_scope_system> ready(
            schedule_ready[0]);
        while (ready.load(cuda::memory_order_acquire) == 0) {
        }
      } else {
        ncclGin gin{dev_comm, gin_context};
        gin.waitSignal(ncclCoopThread(), kGinPublicationSignal, 1);
      }
    }
    grid.sync();
  }

  // Capacity validation makes this unreachable for valid inputs.  If an
  // internal regression ever prevents progress, every rank observes the
  // published failure before trapping, so no peer remains stuck waiting for
  // publication while another rank silently returns a truncated schedule.
  if (global_tid == 0 && schedule_valid[0] == 0) {
    asm volatile("trap;");
  }

  if (rank != 0) {
    for (size_t index = static_cast<size_t>(global_tid);
         index < static_cast<size_t>(num_ranks) * layout.prefix_numel;
         index += global_threads) {
      rank_chunk_prefix[index] = gathered_prefix[index];
    }
  }

  for (int32_t step = global_tid; step < num_chunks; step += global_threads) {
    const int32_t count = step_count[step];
    if (count == 0) {
      logical_token_ranges[step * 2] = 0;
      logical_token_ranges[step * 2 + 1] = 0;
    } else {
      const int32_t begin = step_begin[step];
      logical_token_ranges[step * 2] =
          static_cast<int32_t>(min(static_cast<int64_t>(begin) * chunk_size,
                                   static_cast<int64_t>(max_num_tokens)));
      logical_token_ranges[step * 2 + 1] = static_cast<int32_t>(
          min(static_cast<int64_t>(begin + count) * chunk_size,
              static_cast<int64_t>(max_num_tokens)));
    }
  }

  // Retire this build before the planner workspace can be reused.  The
  // published prefix copy is part of the plan result, so the barrier belongs
  // to this kernel rather than the later EP execution path.
  grid.sync();
  if (block == 0) {
    ::flash_comm::ep::kernels::ep_world_barrier(dev_comm, gin_context,
                                                num_ranks > lsa_world_size);
  }
}

__global__ void kernel_build_ep_chunk_layouts(
    const int32_t *topk_indices, const int32_t *token_within_expert_offset,
    const int32_t *logical_token_ranges, const int32_t *rank_chunk_prefix,
    int32_t num_token, int32_t topk, int32_t num_experts, int32_t chunk_size,
    int32_t num_chunks, int32_t rank, int32_t num_ranks,
    int32_t expert_alignment, int32_t *recv_base_offset,
    int32_t *token_dst_scatter_indices, int32_t *token_topk_send_mask,
    int32_t *recv_token_count, int32_t *recv_aligned_token_count,
    int32_t *recv_expert_counts, int32_t *num_tokens_per_rank) {
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  const int32_t block = static_cast<int32_t>(blockIdx.x);
  const int32_t global_tid = block * static_cast<int32_t>(blockDim.x) + tid;
  const int32_t global_threads =
      static_cast<int32_t>(gridDim.x) * static_cast<int32_t>(blockDim.x);
  const int32_t bins = num_experts + 1;
  const int32_t experts_per_rank = num_experts / num_ranks;

  // Phase 1: one logical step is owned by one CTA at a time.  All metadata is
  // ready before any route reads recv_base_offset in phase 2.
  for (int32_t step = block; step < num_chunks; step += gridDim.x) {
    const int32_t logical_begin = logical_token_ranges[step * 2];
    const int32_t logical_end = logical_token_ranges[step * 2 + 1];
    if (logical_end <= logical_begin) {
      if (tid < num_ranks) {
        recv_token_count[static_cast<int64_t>(step) * num_ranks + tid] = 0;
        recv_aligned_token_count[static_cast<int64_t>(step) * num_ranks + tid] =
            0;
        if (num_tokens_per_rank != nullptr) {
          num_tokens_per_rank[static_cast<int64_t>(step) * num_ranks + tid] = 0;
        }
      }
      for (int32_t local_expert = tid; local_expert < experts_per_rank;
           local_expert += static_cast<int32_t>(blockDim.x)) {
        recv_expert_counts[static_cast<int64_t>(step) * experts_per_rank +
                           local_expert] = 0;
      }
      continue;
    }
    const int32_t begin_chunk = logical_begin / chunk_size;
    const int32_t end_chunk = (logical_end + chunk_size - 1) / chunk_size;

    if (tid < num_ranks) {
      const int32_t dst_rank = tid;
      int32_t unaligned_total = 0;
      int32_t aligned_base = 0;
      for (int32_t local_expert = 0; local_expert < experts_per_rank;
           ++local_expert) {
        const int32_t expert = dst_rank * experts_per_rank + local_expert;
        int32_t expert_total = 0;
        for (int32_t src_rank = 0; src_rank < num_ranks; ++src_rank) {
          const int64_t prefix_base =
              static_cast<int64_t>(src_rank) * (num_chunks + 1) * bins;
          const int32_t count =
              rank_chunk_prefix[prefix_base +
                                static_cast<int64_t>(end_chunk) * bins +
                                expert] -
              rank_chunk_prefix[prefix_base +
                                static_cast<int64_t>(begin_chunk) * bins +
                                expert];
          const int64_t base_index =
              ((static_cast<int64_t>(step) * num_ranks + dst_rank) *
                   experts_per_rank +
               local_expert) *
                  num_ranks +
              src_rank;
          recv_base_offset[base_index] = aligned_base + expert_total;
          expert_total += count;
        }
        if (dst_rank == rank) {
          recv_expert_counts[static_cast<int64_t>(step) * experts_per_rank +
                             local_expert] = expert_total;
        }
        unaligned_total += expert_total;
        aligned_base +=
            ((expert_total + expert_alignment - 1) / expert_alignment) *
            expert_alignment;
      }
      recv_token_count[static_cast<int64_t>(step) * num_ranks + dst_rank] =
          unaligned_total;
      recv_aligned_token_count[static_cast<int64_t>(step) * num_ranks +
                               dst_rank] = aligned_base;

      if (num_tokens_per_rank != nullptr) {
        const int64_t prefix_base =
            static_cast<int64_t>(dst_rank) * (num_chunks + 1) * bins;
        int32_t route_count = 0;
        for (int32_t expert = 0; expert < bins; ++expert) {
          route_count +=
              rank_chunk_prefix[prefix_base +
                                static_cast<int64_t>(num_chunks) * bins +
                                expert];
        }
        const int32_t rank_num_token = route_count / topk;
        const int32_t rank_begin = min(logical_begin, rank_num_token);
        const int32_t rank_end = min(logical_end, rank_num_token);
        num_tokens_per_rank[static_cast<int64_t>(step) * num_ranks + dst_rank] =
            rank_end - rank_begin;
      }
    }
  }
  grid.sync();

  // Phase 2: each local route is visited exactly once by the full grid.  The
  // fixed-Q schedule is an active prefix followed by zero rows; one scan per
  // CTA finds that prefix and each route binary-searches its owning step.
  __shared__ int32_t active_steps;
  if (tid == 0) {
    int32_t active = 0;
    while (active < num_chunks && logical_token_ranges[active * 2 + 1] >
                                      logical_token_ranges[active * 2]) {
      ++active;
    }
    active_steps = active;
  }
  __syncthreads();

  const int64_t route_count = static_cast<int64_t>(num_token) * topk;
  for (int64_t route = global_tid; route < route_count;
       route += global_threads) {
    const int32_t token = static_cast<int32_t>(route / topk);
    int32_t left = 0;
    int32_t right = active_steps;
    while (left < right) {
      const int32_t mid = left + (right - left) / 2;
      if (token < logical_token_ranges[mid * 2 + 1]) {
        right = mid;
      } else {
        left = mid + 1;
      }
    }
    if (left >= active_steps) {
      continue;
    }
    const int32_t step = left;
    const int32_t logical_begin = logical_token_ranges[step * 2];
    if (token < logical_begin) {
      continue;
    }
    const int32_t begin_chunk = logical_begin / chunk_size;
    const int32_t expert = topk_indices[route];
    const int32_t target_rank = expert / experts_per_rank;
    const int32_t target_local_expert = expert % experts_per_rank;
    const int32_t topk_index = static_cast<int32_t>(route % topk);
    int32_t need_send = target_rank < num_ranks ? 1 : 0;
    if (need_send) {
      const int64_t token_route = route - topk_index;
      for (int32_t previous = 0; previous < topk_index; ++previous) {
        const int32_t previous_expert = topk_indices[token_route + previous];
        if (previous_expert / experts_per_rank == target_rank) {
          need_send = 0;
          break;
        }
      }
    }
    int32_t scatter = -1;
    if (target_rank < num_ranks) {
      const int64_t local_prefix_base =
          static_cast<int64_t>(rank) * (num_chunks + 1) * bins +
          static_cast<int64_t>(begin_chunk) * bins;
      const int64_t base_index =
          ((static_cast<int64_t>(step) * num_ranks + target_rank) *
               experts_per_rank +
           target_local_expert) *
              num_ranks +
          rank;
      scatter = recv_base_offset[base_index] +
                token_within_expert_offset[route] -
                rank_chunk_prefix[local_prefix_base + expert];
    }
    const int64_t output_base = static_cast<int64_t>(step) * num_token * topk;
    const int64_t chunk_route =
        output_base + route - static_cast<int64_t>(logical_begin) * topk;
    token_dst_scatter_indices[chunk_route] = scatter;
    token_topk_send_mask[chunk_route] = need_send;
  }
}

} // namespace

size_t workspace_numel(int32_t num_chunks, int32_t num_experts,
                       int32_t num_ranks) {
  FLASH_CHECK(num_chunks > 0);
  FLASH_CHECK(num_experts > 0 && num_experts <= 1024);
  FLASH_CHECK(num_ranks > 0 && num_ranks <= 32);
  FLASH_CHECK(static_cast<int64_t>(2) * num_chunks + 2 <=
                  std::numeric_limits<int32_t>::max() &&
              static_cast<int64_t>(num_chunks + 1) * (num_experts + 1) <=
                  std::numeric_limits<int32_t>::max())
      << "chunk-plan metadata exceeds int32 indexing";
  return get_workspace_layout(num_chunks, num_experts, num_ranks).total;
}

void build_ep_chunk_plan_cuda(
    const int32_t *topk_indices, int32_t num_token, int32_t topk,
    int32_t num_experts, int32_t chunk_size, int32_t max_num_tokens,
    int32_t recv_capacity_tokens, int32_t expert_alignment,
    int32_t launch_num_sms, int32_t rank, int32_t num_ranks,
    int32_t lsa_world_size, int32_t gin_context, uintptr_t workspace_win_handle,
    const void *dev_comm_host, int32_t *logical_token_ranges,
    int32_t *rank_chunk_prefix, cudaStream_t stream) {
  FLASH_CHECK(topk_indices != nullptr || num_token == 0);
  FLASH_CHECK(dev_comm_host != nullptr);
  FLASH_CHECK(launch_num_sms >= 0) << "launch_num_sms must be non-negative";
  FLASH_CHECK(rank >= 0 && rank < num_ranks);
  constexpr int32_t threads = 512;
  const int32_t num_chunks = 1 + (max_num_tokens - 1) / chunk_size;
  const size_t shared_bytes =
      static_cast<size_t>(num_ranks + 1) * sizeof(int32_t);
  ncclDevComm dev_comm = *reinterpret_cast<const ncclDevComm *>(dev_comm_host);
  int32_t device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  static thread_local int32_t cached_device = -1;
  static thread_local int32_t cached_threads = 0;
  static thread_local size_t cached_shared_bytes = 0;
  static thread_local int32_t cached_sm_count = 0;
  static thread_local int32_t cached_blocks_per_sm = 0;
  if (cached_device != device || cached_threads != threads ||
      cached_shared_bytes != shared_bytes) {
    CUDA_CHECK(cudaDeviceGetAttribute(&cached_sm_count,
                                      cudaDevAttrMultiProcessorCount, device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &cached_blocks_per_sm, kernel_build_ep_chunk_plan, threads,
        shared_bytes));
    cached_device = device;
    cached_threads = threads;
    cached_shared_bytes = shared_bytes;
  }
  FLASH_CHECK(cached_sm_count > 0);
  FLASH_CHECK(cached_blocks_per_sm > 0);
  if (launch_num_sms > 0) {
    FLASH_CHECK(launch_num_sms <= cached_sm_count)
        << "launch_num_sms exceeds device SM count: got " << launch_num_sms
        << ", device has " << cached_sm_count;
  }
  const int32_t effective_sm_count =
      (launch_num_sms > 0) ? launch_num_sms : cached_sm_count;
  const int32_t cooperative_limit = effective_sm_count * cached_blocks_per_sm;
  FLASH_CHECK(cooperative_limit > 0);
  constexpr int32_t kRoutesPerThread = 8;
  const int64_t local_routes = static_cast<int64_t>(num_token) * topk;
  const int64_t route_blocks_64 =
      (local_routes + static_cast<int64_t>(threads) * kRoutesPerThread - 1) /
      (static_cast<int64_t>(threads) * kRoutesPerThread);
  const int32_t route_blocks = static_cast<int32_t>(
      std::min<int64_t>(route_blocks_64, cooperative_limit));
  const int32_t desired_blocks = std::max(num_chunks, route_blocks);
  const int32_t blocks = std::min(desired_blocks, cooperative_limit);
  ncclWindow_t workspace_win =
      reinterpret_cast<ncclWindow_t>(workspace_win_handle);
  void *kernel_args[] = {
      &topk_indices,
      &num_token,
      &topk,
      &num_experts,
      &chunk_size,
      &max_num_tokens,
      &recv_capacity_tokens,
      &expert_alignment,
      &rank,
      &num_ranks,
      &lsa_world_size,
      &gin_context,
      &dev_comm,
      &workspace_win,
      &logical_token_ranges,
      &rank_chunk_prefix,
  };
  CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(kernel_build_ep_chunk_plan), blocks, threads,
      kernel_args, shared_bytes, stream));
  CUDA_CHECK(cudaGetLastError());
}

void build_ep_chunk_layouts_cuda(
    const int32_t *topk_indices, const int32_t *token_within_expert_offset,
    const int32_t *logical_token_ranges, const int32_t *rank_chunk_prefix,
    int32_t num_token, int32_t topk, int32_t num_experts, int32_t chunk_size,
    int32_t num_chunks, int32_t rank, int32_t num_ranks,
    int32_t expert_alignment, int32_t *recv_base_offset,
    int32_t *token_dst_scatter_indices, int32_t *token_topk_send_mask,
    int32_t *recv_token_count, int32_t *recv_aligned_token_count,
    int32_t *recv_expert_counts, int32_t *num_tokens_per_rank,
    cudaStream_t stream) {
  FLASH_CHECK(topk_indices != nullptr || num_token == 0);
  FLASH_CHECK(token_within_expert_offset != nullptr || num_token == 0);
  FLASH_CHECK(logical_token_ranges != nullptr);
  FLASH_CHECK(rank_chunk_prefix != nullptr);
  FLASH_CHECK(num_chunks > 0 && chunk_size > 0);
  FLASH_CHECK(num_experts > 0 && num_experts <= 1024);
  FLASH_CHECK(num_ranks > 0 && num_ranks <= 32);
  FLASH_CHECK(rank >= 0 && rank < num_ranks);
  constexpr int32_t threads = 256;
  int32_t device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  static thread_local int32_t cached_device = -1;
  static thread_local int32_t cached_cooperative_limit = 0;
  if (cached_device != device) {
    int32_t sm_count = 0;
    int32_t blocks_per_sm = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                                      device));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, kernel_build_ep_chunk_layouts, threads, 0));
    cached_device = device;
    cached_cooperative_limit = sm_count * blocks_per_sm;
  }
  FLASH_CHECK(cached_cooperative_limit > 0);
  constexpr int32_t kRoutesPerThread = 4;
  const int64_t routes = static_cast<int64_t>(num_token) * topk;
  const int64_t route_blocks_64 =
      (routes + static_cast<int64_t>(threads) * kRoutesPerThread - 1) /
      (static_cast<int64_t>(threads) * kRoutesPerThread);
  const int32_t route_blocks = static_cast<int32_t>(
      std::min<int64_t>(route_blocks_64, cached_cooperative_limit));
  const int32_t desired_blocks = std::max(num_chunks, route_blocks);
  const int32_t blocks = std::min(desired_blocks, cached_cooperative_limit);
  void *kernel_args[] = {
      &topk_indices,
      &token_within_expert_offset,
      &logical_token_ranges,
      &rank_chunk_prefix,
      &num_token,
      &topk,
      &num_experts,
      &chunk_size,
      &num_chunks,
      &rank,
      &num_ranks,
      &expert_alignment,
      &recv_base_offset,
      &token_dst_scatter_indices,
      &token_topk_send_mask,
      &recv_token_count,
      &recv_aligned_token_count,
      &recv_expert_counts,
      &num_tokens_per_rank,
  };
  CUDA_CHECK(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(kernel_build_ep_chunk_layouts), blocks, threads,
      kernel_args, 0, stream));
  CUDA_CHECK(cudaGetLastError());
}

} // namespace chunk_plan
} // namespace ep
} // namespace flash_comm
