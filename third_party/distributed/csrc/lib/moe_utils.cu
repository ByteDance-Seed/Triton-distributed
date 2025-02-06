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
#include "apis.cuh"
#include <assert.h>

#include <cstdint>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

#undef CUDA_CHECK
#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

namespace distributed {

namespace ops {

__device__ __forceinline__ int32_t index(int32_t total_col, int32_t row,
                                         int32_t col) {
  // don't worry about overflow because num_experts is relatively small
  return row * total_col + col;
}

template <typename scalar_t>
__global__ void moe_ag_scatter_align_block_size_kernel(
    scalar_t *__restrict__ topk_ids, int32_t *sorted_token_ids,
    int32_t *expert_ids, int32_t *block_barrier_ids, int32_t *rank_block_num,
    int32_t *total_tokens_post_pad, int32_t num_experts, int32_t num_iterations,
    int32_t num_tokens_per_iteration, int32_t block_size, size_t numel) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  extern __shared__ int32_t shared_mem[];
  const size_t tokens_per_thread = CEILDIV(numel, nthreads);
  const size_t start_idx = tid * tokens_per_thread;

  int last_pad_tokens = 0;
  *total_tokens_post_pad = 0;
  for (int iter = 0; iter < num_iterations; ++iter) {
    sorted_token_ids += last_pad_tokens;
    expert_ids += last_pad_tokens / block_size;
    block_barrier_ids += last_pad_tokens / block_size;

    int32_t *tokens_cnts =
        shared_mem; // 2d tensor with shape (num_experts + 1, num_experts)
    int32_t *cumsum =
        shared_mem + (num_experts + 1) *
                         num_experts; // 1d tensor with shape (num_experts + 1)

    for (int i = 0; i < num_experts; ++i) {
      tokens_cnts[index(num_experts, tid + 1, i)] = 0;
    }

    /**
     * In the first step we compute token_cnts[thread_index + 1][expert_index],
     * which counts how many tokens in the token shard of thread_index are
     * assigned to expert expert_index.
     */
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread;
         ++i) {
      ++tokens_cnts[index(num_experts, tid + 1, topk_ids[i])];
    }

    __syncthreads();

    // For each expert we accumulate the token counts from the different
    // threads.
    tokens_cnts[index(num_experts, 0, tid)] = 0;
    for (int i = 1; i <= nthreads; ++i) {
      tokens_cnts[index(num_experts, i, tid)] +=
          tokens_cnts[index(num_experts, i - 1, tid)];
    }
    __syncthreads();

    // We accumulate the token counts of all experts in thread 0.
    if (tid == 0) {
      cumsum[0] = 0;
      for (int i = 1; i <= num_experts; ++i) {
        cumsum[i] = cumsum[i - 1] +
                    CEILDIV(tokens_cnts[index(num_experts, nthreads, i - 1)],
                            block_size) *
                        block_size;
      }

      *total_tokens_post_pad += cumsum[num_experts];
      rank_block_num[iter] = cumsum[num_experts] / block_size;
    }

    __syncthreads();
    last_pad_tokens = cumsum[num_experts];

    /**
     * For each expert, each thread processes the tokens of the corresponding
     * blocks and stores the corresponding expert_id for each block.
     */
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid;
      block_barrier_ids[i / block_size] = iter;
    }

    /**
     * Each thread processes a token shard, calculating the index of each token
     * after sorting by expert number. Given the example topk_ids =
     * [0,1,2,1,2,3,0,3,4] and block_size = 4, then the output would be [0, 6,
     * *,
     * *, 1, 3, *, *, 2, 4, *, *, 5, 7, *, *, 8, *, *, *], where * represents a
     * padding value(preset in python).
     */
    for (int i = start_idx; i < numel && i < start_idx + tokens_per_thread;
         ++i) {
      int32_t expert_id = topk_ids[i];
      /** The cumsum[expert_id] stores the starting index of the tokens that the
       * expert with expert_id needs to process, and
       * tokens_cnts[tid][expert_id] stores the indices of the tokens
       * processed by the expert with expert_id within the current thread's
       * token shard.
       */
      int32_t rank_post_pad =
          tokens_cnts[index(num_experts, tid, expert_id)] + cumsum[expert_id];
      sorted_token_ids[rank_post_pad] = i + iter * num_tokens_per_iteration;
      ++tokens_cnts[index(num_experts, tid, expert_id)];
    }

    topk_ids += num_tokens_per_iteration;

    __syncthreads();
  }
}

void moe_ag_scatter_align_block_size_op(
    torch::Tensor topk_ids, int32_t num_experts, int32_t num_iterations,
    int32_t num_tokens_per_iteration, int32_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor block_barrier_ids, torch::Tensor rank_block_num,
    torch::Tensor num_tokens_post_pad, intptr_t moe_stream) {

  const int32_t shared_mem =
      ((num_experts + 1) * num_experts + (num_experts + 1)) * sizeof(int32_t);
  int THREAD_NUM = (int)num_experts;
  constexpr int CTA_NUM = 1;
  dim3 grid_dim(CTA_NUM);
  dim3 block_dim(THREAD_NUM);
  moe_ag_scatter_align_block_size_kernel<int64_t>
      <<<grid_dim, block_dim, shared_mem, (cudaStream_t)moe_stream>>>(
          (int64_t *)topk_ids.data_ptr(),
          (int32_t *)sorted_token_ids.data_ptr(),
          (int32_t *)experts_ids.data_ptr(),
          (int32_t *)block_barrier_ids.data_ptr(),
          (int32_t *)rank_block_num.data_ptr(),
          (int32_t *)num_tokens_post_pad.data_ptr(), num_experts,
          num_iterations, num_tokens_per_iteration, block_size,
          num_tokens_per_iteration);
  CUDA_CHECK(cudaGetLastError());
}

template <int kNumThreads, int kNumMaxExperts>
__global__ void __launch_bounds__(kNumThreads, 1)
    moe_ag_scatter_align_block_size_parallel_kernel(
        int64_t *__restrict__ topk_ids, int32_t num_topk, int32_t num_experts,
        int32_t num_ranks, int32_t num_tokens_per_rank, int32_t block_size,
        unsigned int *expert_cumsum_per_rank, volatile unsigned int *block_sync,
        int32_t *sorted_token_ids, int32_t *expert_ids,
        int32_t *block_barrier_ids, int32_t *rank_block_num,
        int32_t *total_tokens_post_pad) {
  int nblocks = gridDim.x;
  int nthreads = blockDim.x;
  int bid = blockIdx.x;
  int tid = (threadIdx.x * blockDim.y * blockDim.z + threadIdx.y * blockDim.z +
             threadIdx.z);
  unsigned int counter;
  __shared__ int32_t tokens_cnts_per_expert[kNumMaxExperts];
  __shared__ int32_t tokens_cnts_per_thread[kNumThreads][kNumMaxExperts];
  __shared__ int32_t tokens_cumsum_per_thread[kNumThreads][kNumMaxExperts];

  const size_t tokens_per_thread = CEILDIV(num_tokens_per_rank, nthreads);
  // each block calculate one rank
  if (bid >= num_ranks) {
    return;
  }
  int topk_start_idx = bid * num_tokens_per_rank;

#pragma unroll
  for (int i = 0; i < num_experts; ++i)
    tokens_cnts_per_thread[tid][i] = 0;

#pragma unroll
  for (int i = tid; i < tokens_per_thread; i += nthreads) {
    auto shifted_topk_idx = topk_ids + topk_start_idx + i * num_topk;
#pragma unroll
    for (int j = 0; j < num_topk; ++j) {
      ++tokens_cnts_per_thread[tid][shifted_topk_idx[j]];
    }
  }
  __syncthreads();

  if (tid < num_experts) {
    int sum = tokens_cnts_per_thread[0][tid];
    tokens_cumsum_per_thread[0][tid] = 0;
#pragma unroll
    for (int i = 1; i < nthreads; ++i) {
      sum += tokens_cnts_per_thread[i][tid];
      tokens_cumsum_per_thread[i][tid] =
          tokens_cumsum_per_thread[i - 1][tid] + tokens_cnts_per_thread[i][tid];
    }
    tokens_cnts_per_expert[tid] = sum;
  }
  __syncthreads();

  int global_token_start_idx = 0;

  if (!tid) {
    expert_cumsum_per_rank[bid * num_experts] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      expert_cumsum_per_rank[bid * num_experts + i] =
          expert_cumsum_per_rank[bid * num_experts + i - 1] +
          CEILDIV(tokens_cnts_per_expert[i - 1], block_size) * block_size;
    }
    rank_block_num[bid] =
        expert_cumsum_per_rank[bid * num_experts + num_experts - 1] /
        block_size;

    __threadfence();
    counter = atomicInc((unsigned int *)block_sync, UINT_MAX);
    if (counter == (num_ranks - 1)) {
      *(block_sync + 1) += 1;
    }
    while (*(block_sync + 1) != 1)
      ;

    for (int i = 0; i < bid; i++) {
      global_token_start_idx +=
          expert_cumsum_per_rank[i * num_experts + num_experts - 1];
    }
    int token_end_idx =
        global_token_start_idx +
        expert_cumsum_per_rank[bid * num_experts + num_experts - 1];
    if (bid == (num_ranks - 1)) {
      *total_tokens_post_pad = token_end_idx;
    }
  }
  __syncthreads();

  int global_block_start_idx = global_token_start_idx / block_size;
  if (tid < num_experts) {
#pragma unroll
    for (int i = expert_cumsum_per_rank[bid * num_experts + tid];
         i < expert_cumsum_per_rank[bid * num_experts + tid + 1];
         i += block_size) {
      expert_ids[global_block_start_idx + i / block_size] = tid;
      block_barrier_ids[global_block_start_idx + i / block_size] = bid;
    }
  }

#pragma unroll
  for (int i = tid; i < tokens_per_thread; i += nthreads) {
    auto shifted_topk_idx = topk_ids + topk_start_idx + i * num_topk;
#pragma unroll
    for (int j = 0; j < num_topk; ++j) {
      int32_t expert_id = shifted_topk_idx[j];
      int32_t pos = expert_cumsum_per_rank[bid * num_experts + expert_id] +
                    tokens_cumsum_per_thread[tid][expert_id];
      sorted_token_ids[global_token_start_idx + pos] =
          topk_start_idx + i * num_topk + j;
      ++tokens_cumsum_per_thread[tid][expert_id];
    }
  }
}

void moe_ag_scatter_align_block_size_parallel_op(
    torch::Tensor topk_ids, int32_t topk, int32_t num_experts,
    int32_t num_ranks, int32_t num_tokens_per_rank, int32_t block_size,
    torch::Tensor sorted_token_ids, torch::Tensor experts_ids,
    torch::Tensor block_barrier_ids, torch::Tensor rank_block_num,
    torch::Tensor num_tokens_post_pad, intptr_t moe_stream) {

  constexpr int THREAD_NUM = 64;
  constexpr int MAX_EXPERT = 256;
  int cta_num = num_ranks;
  dim3 grid_dim(cta_num);
  dim3 block_dim(THREAD_NUM);
  unsigned int *counter_d, *expert_cumsum_per_rank;

  cudaMalloc((void **)&counter_d, sizeof(unsigned int) * 2);
  cudaMemset(counter_d, 0, sizeof(unsigned int) * 2);
  cudaMalloc((void **)&expert_cumsum_per_rank,
             sizeof(unsigned int) * num_ranks * (num_experts + 1));
  cudaMemset(expert_cumsum_per_rank, 0,
             sizeof(unsigned int) * num_ranks * (num_experts + 1));
  CUDA_CHECK(cudaDeviceSynchronize());

  moe_ag_scatter_align_block_size_parallel_kernel<THREAD_NUM, MAX_EXPERT>
      <<<grid_dim, block_dim, 0, (cudaStream_t)moe_stream>>>(
          (int64_t *)topk_ids.data_ptr(), topk, num_experts, num_ranks,
          num_tokens_per_rank, block_size, expert_cumsum_per_rank, counter_d,
          (int32_t *)sorted_token_ids.data_ptr(),
          (int32_t *)experts_ids.data_ptr(),
          (int32_t *)block_barrier_ids.data_ptr(),
          (int32_t *)rank_block_num.data_ptr(),
          (int32_t *)num_tokens_post_pad.data_ptr());
}

} // namespace ops
} // namespace distributed
