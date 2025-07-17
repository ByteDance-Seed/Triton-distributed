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
// compile with NVCC with extra flag: --expt-relaxed-constexpr
#include <algorithm>
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ostream>
#include <random>
#include <tuple>
#include <vector>

using namespace std;

// CUDA error checking macros
#define CUDA_CHECK(ans)                                                        \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

#define CHECK(ans)                                                             \
  {                                                                            \
    bool value = (ans);                                                        \
    if (!value) {                                                              \
      fprintf(stderr, "[%s:%d] " #ans "\n", __FILE__, __LINE__);               \
      throw std::runtime_error("CHECK failed");                                \
    }                                                                          \
  }

constexpr int kWarpSize = 32;

template <typename T> __device__ __host__ T cdiv(T a, int b) {
  return (a + b - 1) / b;
}

template <typename T> __device__ __host__ T pad_to(T x, int alignment) {
  return cdiv(x, alignment) * alignment;
}

// Input: each thread in a warp call this function
//        with its `id` (lane_id) and its `cnt` to be summed.
// Output: each thread get a presum of all threads' `cnt`
//        that have `id` less than or equal to its own `id`
template <class T> __inline__ __device__ T warp_prefix_sum(int id, T cnt) {
  for (int i = 1; i < kWarpSize; i <<= 1) {
    T val = __shfl_up_sync(0xffffffff, cnt, i);
    if (id >= i)
      cnt += val;
  }
  return cnt;
}

template <class T>
__inline__ __device__ void
aligned_warp_prefix_sum(const T *data_in, T *data_out, int cnt, int align,
                        int in_stride = 1) {
  int cur_offset = 0;
  int lane_idx = threadIdx.x % kWarpSize;
  for (int i = lane_idx; i < pad_to(cnt, kWarpSize); i += kWarpSize) {
    int value = i < cnt ? data_in[i * in_stride] : 0;
    value = pad_to(value, align);
    int temp_offset = warp_prefix_sum(lane_idx, value);
    if (i < cnt) {
      data_out[i] = cur_offset + temp_offset;
    }
    cur_offset += __shfl_sync(0xffffffff, temp_offset, kWarpSize - 1);
  }
}

template <class T>
__inline__ __device__ void warp_prefix_sum(const T *data_in, T *data_out,
                                           int cnt, int in_stride = 1) {
  aligned_warp_prefix_sum(data_in, data_out, cnt, 1, in_stride);
}

template <class T>
__inline__ __device__ void
aligned_block_prefix_sum(const T *data_in, T *data_out, int cnt, int align,
                         int in_stride = 1) {
  int warp_idx = threadIdx.x / kWarpSize;
  if (warp_idx == 0) {
    aligned_warp_prefix_sum(data_in, data_out, cnt, align, in_stride);
  }
}

template <class T>
__inline__ __device__ void block_prefix_sum(const T *data_in, T *data_out,
                                            int cnt, int in_stride = 1) {
  aligned_block_prefix_sum(data_in, data_out, cnt, 1, in_stride);
}

/**
 * @brief a CUDA version std::upper_bound like this:
 * https://cplusplus.com/reference/algorithm/upper_bound/
 */
template <class T>
__inline__ __device__ const T *upper_bound_kernel(const T *first, const T *last,
                                                  const T &value) {
  const T *it;
  size_t cnt, step;
  cnt = std::distance(first, last);

  while (cnt > 0) {
    it = first;
    step = cnt / 2;
    it += step;

    if (*it <= value) {
      first = ++it;
      cnt -= step + 1;
    } else
      cnt = step;
  }

  return first;
}

template <class T>
__inline__ __device__ int bisect_left(const T *first, const T *last,
                                      const T &value) {
  return std::distance(first, lower_bound_kernel(first, last, value));
}

template <class T>
__inline__ __device__ int bisect_right(const T *first, const T *last,
                                       const T &value) {
  return std::distance(first, upper_bound_kernel(first, last, value));
}

struct TileInfo {
  int expert_id;
  int tiled_m;
  int stage;
  int segment_end;
  int segment_start;
};

// for each tile each expert
__device__ std::tuple<int, int, int>
get_tile_stage(int tiled_m, const int rank, int tp_size, int block_size_m,
               const int *token_cnt, const int *token_cnt_acc) {

  int ntokens = token_cnt_acc[tp_size - 1];
  int global_m_start = token_cnt_acc[rank] - token_cnt[rank];
  int global_tiled_m_start = cdiv(global_m_start, block_size_m);

  int m_start = tiled_m * block_size_m;
  int m_end = min((tiled_m + 1) * block_size_m, ntokens) - 1;
  int segment_start =
      bisect_right(token_cnt_acc, token_cnt_acc + tp_size, m_start);
  int segment_end = bisect_right(token_cnt_acc, token_cnt_acc + tp_size, m_end);

  int stage = (segment_end - rank + tp_size) % tp_size;

  if (tiled_m == global_tiled_m_start - 1 &&
      global_m_start % block_size_m != 0) {
    int m_segment_end =
        bisect_right(token_cnt_acc, token_cnt_acc + rank, m_end);
    stage = (m_segment_end - rank + tp_size) % tp_size;
  }
  return std::tuple<int, int, int>{stage, segment_start, segment_end};
}

// contingunous at dim1
template <typename T>
vector<T> transpose2d(const vector<T> &arr_2d, int dim0, int dim1) {
  CHECK(arr_2d.size() == dim0 * dim1);

  vector<T> out(dim0 * dim1);
  for (size_t i = 0; i < dim0 * dim1; i++) {
    int row = i / dim1;
    int col = i % dim1;
    out[row + col * dim0] = arr_2d[i];
  }
  return out;
}

/**
 * @brief out(j, i) = in(i, j).
 *
 * @param in : shape (dim0, dim1). in(i, j) = in + i * dim1 + j
 * @param out : shape (dim1, dim0). out(j, i) = out + j * dim0 + i
 */
template <typename T>
__device__ __forceinline__ void transpose2d_block(const T *in, T *out, int dim0,
                                                  int dim1) {
  for (int i = threadIdx.x; i < dim0 * dim1; i += blockDim.x) {
    int row = i / dim1;
    int col = i % dim1;
    out[col * dim0 + row] = in[i];
  }
}

// contiguous at dim1
template <typename T>
__device__ __forceinline__ void
aligned_block_prefix_sum_2d(const T *src, T *dst, int dim0, int dim1,
                            int align) {
  int wid = threadIdx.x / kWarpSize;
  int num_warps = blockDim.x / kWarpSize;

  for (int i = wid; i < dim0; i += num_warps) {
    warp_prefix_sum(src + i * dim1, dst + i * dim1, dim1);
  }
}

// Main GPU swizzle kernel
// clang-format off
// @params[in] token_cnt_by_rank_by_expert[eid, rank] => tokens for expert `eid` at `rank`. with shape shape [tp_size, n_experts]
// @param[out] tiles_by_expert_by_stage organized as
//    [
//      Tile(eid=0, stage=0, ...), Tile(eid=0, stage=1, ...), Tile(eid=0, stage=7, ...),
//      Tile(eid=1, stage=0, ...), Tile(eid=1, stage=1, ...), Tile(eid=1, stage=7, ...),
//            ...
//      Tile(eid=31, stage=0, ...), Tile(eid=31, stage=1, ...), Tile(eid=31, stage=7, ...),
//    ].  in which Tile(eid=0, stage=0, ...) mean all tiles for expert 0 at stage 0. maybe 0 tiles, maybe more than 1 tiles.
// clang-format on
template <bool kDebug = false>
__global__ void threadblock_swizzle_ag_moe_kernel(
    int rank, int tp_size, int n_experts, int block_size_m,
    const int *token_cnt_by_rank_by_expert,
    // as workspace buffer. no use after the kernel. should be zeroed before
    // kernel launch.
    int *num_tiles_ptr,                       // of 1
    int *token_cnt_by_expert_by_rank,         // of tp_size * n_experts.
    int *token_cnt_by_expert_by_rank_cumsum,  // of tp_size * n_experts.
    int *stage_index_by_stage_by_expert,      // of tp_size * n_experts
    int *num_tiles_by_stage_by_expert,        // of tp_size * n_experts
    int *num_tiles_by_stage_by_expert_cumsum, // of tp_size * n_experts
    int *num_tiles_by_expert_by_stage,        // of tp_size * n_experts
    int *num_tiles_by_expert_by_stage_cumsum, // of tp_size * n_experts
    int *num_tiles_by_stage,                  // of tp_size
    int *num_tiles_by_stage_cumsum,           // of tp_size
    int *num_tiles_by_expert,                 // of n_experts
    int *num_tiles_by_expert_cumsum,          // of n_experts
    TileInfo *tiles_by_expert_by_tiled_m,     // of num_tiles
    int *tile_index_by_expert_by_stage,       // of num_tiles
    // output
    TileInfo *output_tiles) {

  auto DBG = [](auto &&...args) {
    if constexpr (kDebug) {
      printf(args...);
    }
  };

  if (threadIdx.x == 0) {
    DBG("threadblock_swizzle_ag_moe_kernel kernel: rank: %d, tp_size: %d, "
        "n_experts: %d, block_size_m: %d\n",
        rank, tp_size, n_experts, block_size_m);
  }

  transpose2d_block(token_cnt_by_rank_by_expert, token_cnt_by_expert_by_rank,
                    tp_size, n_experts);
  __syncthreads();

  aligned_block_prefix_sum_2d(token_cnt_by_expert_by_rank,
                              token_cnt_by_expert_by_rank_cumsum, n_experts,
                              tp_size, 1);
  __syncthreads();

  for (int i = threadIdx.x; i < n_experts; i += blockDim.x) {
    num_tiles_by_expert[i] =
        cdiv(token_cnt_by_expert_by_rank_cumsum[i * tp_size + tp_size - 1],
             block_size_m);
  }
  __syncthreads();
  aligned_block_prefix_sum(num_tiles_by_expert, num_tiles_by_expert_cumsum,
                           n_experts, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    *num_tiles_ptr = num_tiles_by_expert_cumsum[n_experts - 1];
  }
  __syncthreads();

  int num_tiles = *num_tiles_ptr;

  // (expert, tiled_m) => which stage should it be? which segment does it cross?
  // organized as this to fully partition the job into many threadIdx.x
  for (int i = threadIdx.x; i < num_tiles; i += blockDim.x) {
    // expert id
    int eid = bisect_right(num_tiles_by_expert_cumsum,
                           num_tiles_by_expert_cumsum + n_experts, i);
    int tiled_m_eid_offset =
        num_tiles_by_expert_cumsum[eid] - num_tiles_by_expert[eid];
    int tiled_m_in_expert = i - tiled_m_eid_offset;

    const int *token_cnt_this_expert =
        token_cnt_by_expert_by_rank + eid * tp_size;
    const int *token_cnt_this_expert_cumsum =
        token_cnt_by_expert_by_rank_cumsum + eid * tp_size;
    auto [stage, segment_start, segment_end] =
        get_tile_stage(tiled_m_in_expert, rank, tp_size, block_size_m,
                       token_cnt_this_expert, token_cnt_this_expert_cumsum);
    tiles_by_expert_by_tiled_m[i].tiled_m = tiled_m_in_expert;
    tiles_by_expert_by_tiled_m[i].stage = stage;
    tiles_by_expert_by_tiled_m[i].expert_id = eid;
    tiles_by_expert_by_tiled_m[i].segment_start = segment_end;
    tiles_by_expert_by_tiled_m[i].segment_end = segment_start;
    DBG("i: %d (eid: %d, tiled_m: %d, stage: %d, segment_start: %d, "
        "segment_end: %d)\n",
        i, eid, tiled_m_in_expert, stage, segment_start, segment_end);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < num_tiles; i += blockDim.x) {
    const auto &tile = tiles_by_expert_by_tiled_m[i];
    int eid = tile.expert_id;
    int stage = tile.stage;
    atomicAdd(num_tiles_by_expert_by_stage + eid * tp_size + stage, 1);
    atomicAdd(num_tiles_by_stage_by_expert + eid + stage * n_experts, 1);
    atomicAdd(num_tiles_by_stage + stage, 1);
  }
  __syncthreads();
  aligned_block_prefix_sum_2d(num_tiles_by_expert_by_stage,
                              num_tiles_by_expert_by_stage_cumsum, n_experts,
                              tp_size, 1);
  aligned_block_prefix_sum_2d(num_tiles_by_stage_by_expert,
                              num_tiles_by_stage_by_expert_cumsum, tp_size,
                              n_experts, 1);
  block_prefix_sum(num_tiles_by_stage, num_tiles_by_stage_cumsum, tp_size);
  __syncthreads();

  // tiles_by_expert_by_tileid => tiles_by_expert_by_stage
  for (int i = threadIdx.x; i < num_tiles; i += blockDim.x) {
    const auto &tile = tiles_by_expert_by_tiled_m[i];
    int eid = tile.expert_id;
    int stage = tile.stage;
    int stage_index = atomicAdd_block(stage_index_by_stage_by_expert + eid +
                                          stage * n_experts,
                                      1); // stage index does not matter
    // new index: expert_id_offset + stage_offset + stage_index
    int expert_id_offset =
        num_tiles_by_expert_cumsum[eid] - num_tiles_by_expert[eid];
    int stage_offset =
        num_tiles_by_expert_by_stage_cumsum[eid * tp_size + stage] -
        num_tiles_by_expert_by_stage[eid * tp_size + stage];
    tile_index_by_expert_by_stage[expert_id_offset + stage_offset +
                                  stage_index] = i;
    DBG("reorg: from %d(E%d, S%d) => (%d, %d, %d) => %d\n", i, eid, stage,
        expert_id_offset, stage_offset, stage_index,
        expert_id_offset + stage_offset + stage_index);
  }

  __syncthreads();

  for (int i = threadIdx.x; i < num_tiles; i += blockDim.x) {
    int tiled_m = i;
    // threadIdx.x => stage
    int stage = bisect_right(num_tiles_by_stage_cumsum,
                             num_tiles_by_stage_cumsum + tp_size, tiled_m);
    int stage_offset =
        num_tiles_by_stage_cumsum[stage] - num_tiles_by_stage[stage];
    int tiled_m_this_stage = tiled_m - stage_offset;

    const int *num_tiles_by_expert_this_stage =
        num_tiles_by_stage_by_expert + stage * n_experts;
    const int *num_tiles_by_expert_this_stage_cumsum =
        num_tiles_by_stage_by_expert_cumsum + stage * n_experts;
    int expert_id = bisect_right(
        num_tiles_by_expert_this_stage_cumsum,
        num_tiles_by_expert_this_stage_cumsum + n_experts, tiled_m_this_stage);
    int expert_offset_this_stage =
        num_tiles_by_expert_this_stage_cumsum[expert_id] -
        num_tiles_by_expert_this_stage[expert_id];
    int tiled_m_this_stage_this_expert =
        tiled_m_this_stage - expert_offset_this_stage;

    // mapping back to stage info
    int expert_offset =
        num_tiles_by_expert_cumsum[expert_id] - num_tiles_by_expert[expert_id];
    int stage_offset_this_expert =
        num_tiles_by_expert_by_stage_cumsum[expert_id * tp_size + stage] -
        num_tiles_by_expert_by_stage[expert_id * tp_size + stage];
    int index = expert_offset + stage_offset_this_expert +
                tiled_m_this_stage_this_expert;
    output_tiles[i] =
        tiles_by_expert_by_tiled_m[tile_index_by_expert_by_stage[index]];
    DBG("i = %d => (%d, %d, %d) (%d, %d, %d) => %d\n", i, expert_id, stage,
        tiled_m_this_stage_this_expert, expert_offset, stage_offset_this_expert,
        tiled_m_this_stage_this_expert, index);
  }
}

vector<int> generate_uniform_counts_by_rank_by_expert(int ntokens_by_rank,
                                                      int nexperts,
                                                      int tp_size) {
  return vector<int>(tp_size * nexperts, ntokens_by_rank / nexperts);
}

vector<int> generate_random_counts_by_rank_by_expert(int ntokens_by_rank,
                                                     int nexperts,
                                                     int tp_size) {
  random_device rd;
  mt19937 gen(12345);
  vector<int> counts(tp_size * nexperts);

  for (int i = 0; i < tp_size; i++) {
    int *row = counts.data() + i * nexperts;
    discrete_distribution<> d(nexperts, 0, nexperts - 1,
                              [=](int) { return 1.0 / nexperts; });
    for (int i = 0; i < ntokens_by_rank; i++)
      row[d(gen)]++;
  }
  return counts;
}

vector<int> generate_random_counts_with_zeros_by_rank_by_expert(
    int ntokens_by_rank, int nexperts, int tp_size, float zero_rate) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dist(0.0, 1.0);

  vector<int> counts(tp_size * nexperts, 0);

  for (int i = 0; i < tp_size; i++) {
    int *row = counts.data() + i * nexperts;
    vector<double> weights;
    for (int i = 0; i < nexperts; i++) {
      weights.push_back(dist(gen) > zero_rate ? 1.0 : 1e-5);
    }
    double sum = accumulate(weights.begin(), weights.end(), 0.0);
    for (auto &w : weights)
      w /= sum;

    discrete_distribution<> d(weights.begin(), weights.end());
    for (int i = 0; i < ntokens_by_rank; i++)
      row[d(gen)]++;
  }
  return counts;
}

void check_swizzled(const vector<tuple<int, int>> &swizzled,
                    const vector<int> &token_cnts_by_rank_by_expert,
                    int n_experts, int tp_size) {
  auto token_cnt_by_expert_by_rank =
      transpose2d(token_cnts_by_rank_by_expert, n_experts, tp_size);
  for (size_t expert_id = 0; expert_id < n_experts; expert_id++) {
    int *token_cnt = token_cnt_by_expert_by_rank.data() + expert_id * tp_size;
    int total = accumulate(token_cnt, token_cnt + tp_size, 0);
    vector<int> tiles;
    for (auto [eid, tiled_m] : swizzled)
      if (eid == expert_id) {
        tiles.push_back(tiled_m);
      }

    sort(tiles.begin(), tiles.end());
    for (size_t i = 0; i < tiles.size(); i++) {
      CHECK(tiles[i] == i);
    }
  }
}

// dim1 is the contiguous dimension
void print_vector_2d(const int *vec, int dim0, int dim1) {
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      cout << vec[i * dim1 + j] << " ";
    }
    cout << endl;
  }
}

// dim1 is the contiguous dimension
void print_vector_2d(const vector<int> &vec, int dim0, int dim1) {
  for (int i = 0; i < dim0; i++) {
    for (int j = 0; j < dim1; j++) {
      cout << vec[i * dim1 + j] << " ";
    }
    cout << endl;
  }
}

std::vector<std::tuple<int, int>> threadblock_swizzle_ag_moe_cuda(
    int rank, int n_experts, int tp_size, int block_size_m,
    const vector<int> &token_cnts_by_rank_by_expert, int ntokens,
    int ntiles_total, cudaStream_t stream, int verbose = 0) {
  int alignment = 32;
  char *workspace = nullptr;
  size_t workspace_size =
      pad_to(sizeof(int), alignment) +
      pad_to(tp_size * n_experts * sizeof(int), alignment) * 7 +
      pad_to(tp_size * sizeof(int), alignment) * 2 +
      pad_to(n_experts * sizeof(int), alignment) * 2 +
      pad_to(ntiles_total * sizeof(TileInfo), alignment) +
      pad_to(ntiles_total * sizeof(int), alignment);
  // CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
  char *buffer = workspace;
  CUDA_CHECK(cudaMemset(workspace, 0, workspace_size));

  int *num_tiles_ptr = (int *)(workspace);
  workspace += pad_to(sizeof(int), alignment);
  int *token_cnt_by_expert_by_rank = (int *)workspace;
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *token_cnt_by_expert_by_rank_cumsum = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *stage_index_by_stage_by_expert = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *num_tiles_by_stage_by_expert = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *num_tiles_by_stage_by_expert_cumsum = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *num_tiles_by_expert_by_stage = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *num_tiles_by_expert_by_stage_cumsum = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
  int *num_tiles_by_stage = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size, alignment);
  int *num_tiles_by_stage_cumsum = (int *)(workspace);
  workspace += pad_to(sizeof(int) * tp_size, alignment);
  int *num_tiles_by_expert = (int *)(workspace);
  workspace += pad_to(sizeof(int) * n_experts, alignment);
  int *num_tiles_by_expert_cumsum = (int *)(workspace);
  workspace += pad_to(sizeof(int) * n_experts, alignment);
  TileInfo *tiles_by_expert_by_tiled_m = (TileInfo *)(workspace);
  workspace += pad_to(sizeof(TileInfo) * ntiles_total, alignment);
  int *tile_index_by_expert_by_stage = (int *)(workspace);
  workspace += pad_to(sizeof(int) * ntiles_total, alignment);

  int *d_token_cnt_by_rank_by_expert = nullptr;
  CUDA_CHECK(cudaMalloc(&d_token_cnt_by_rank_by_expert,
                        token_cnts_by_rank_by_expert.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_token_cnt_by_rank_by_expert,
                        token_cnts_by_rank_by_expert.data(),
                        token_cnts_by_rank_by_expert.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  void *d_output_tiles = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output_tiles, sizeof(TileInfo) * ntiles_total));

  if (verbose >= 1) {
    cout << "threadblock_swizzle_ag_moe_kernel start with tp_size: " << tp_size
         << " experts: " << n_experts << "\n";
    flush(cout);
  }

  (verbose >= 4
       ? threadblock_swizzle_ag_moe_kernel<true>
       : threadblock_swizzle_ag_moe_kernel<false>)<<<1, 1024, 0, stream>>>(
      rank, tp_size, n_experts, block_size_m,
      d_token_cnt_by_rank_by_expert, // of tp_size * n_experts.
      // as workspace buffer. no use after the kernel. should be zeroed before
      // kernel launch.
      num_tiles_ptr,
      token_cnt_by_expert_by_rank,         // of tp_size * n_experts.
      token_cnt_by_expert_by_rank_cumsum,  // of tp_size * n_experts.
      stage_index_by_stage_by_expert,      // of tp_size * n_experts
      num_tiles_by_stage_by_expert,        // of tp_size * n_experts
      num_tiles_by_stage_by_expert_cumsum, // of tp_size * n_experts
      num_tiles_by_expert_by_stage,        // of tp_size * n_experts
      num_tiles_by_expert_by_stage_cumsum, // of tp_size * n_experts
      num_tiles_by_stage,                  // of tp_size
      num_tiles_by_stage_cumsum,           // of tp_size
      num_tiles_by_expert,                 // of n_experts
      num_tiles_by_expert_cumsum,          // of n_experts
      tiles_by_expert_by_tiled_m, tile_index_by_expert_by_stage,
      // output
      (TileInfo *)d_output_tiles);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  if (verbose >= 1) {
    cout << "threadblock_swizzle_ag_moe_kernel done\n";
    flush(cout);
  }

  std::vector<TileInfo> output_tiles(ntiles_total);
  CUDA_CHECK(cudaMemcpy(output_tiles.data(), d_output_tiles,
                        sizeof(TileInfo) * ntiles_total,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (verbose >= 3) {
    std::vector<char> h_workspace(workspace_size);
    CUDA_CHECK(cudaMemcpy(h_workspace.data(), buffer, workspace_size,
                          cudaMemcpyDeviceToHost));

    char *workspace = h_workspace.data();
    int *num_tiles_ptr = (int *)(workspace);
    workspace += pad_to(sizeof(int), alignment);
    int *token_cnt_by_expert_by_rank = (int *)workspace;
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *token_cnt_by_expert_by_rank_cumsum = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *stage_index_by_stage_by_expert = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *num_tiles_by_stage_by_expert = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *num_tiles_by_stage_by_expert_cumsum = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *num_tiles_by_expert_by_stage = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *num_tiles_by_expert_by_stage_cumsum = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size * n_experts, alignment);
    int *num_tiles_by_stage = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size, alignment);
    int *num_tiles_by_stage_cumsum = (int *)(workspace);
    workspace += pad_to(sizeof(int) * tp_size, alignment);
    int *num_tiles_by_expert = (int *)(workspace);
    workspace += pad_to(sizeof(int) * n_experts, alignment);
    int *num_tiles_by_expert_cumsum = (int *)(workspace);
    workspace += pad_to(sizeof(int) * n_experts, alignment);
    TileInfo *tiles_by_expert_by_tiled_m = (TileInfo *)(workspace);
    workspace += pad_to(sizeof(TileInfo) * ntiles_total, alignment);
    int *tile_index_by_expert_by_stage = (int *)(workspace);
    workspace += pad_to(sizeof(int) * ntiles_total, alignment);

    printf("token_cnt_by_rank_by_expert\n");
    print_vector_2d(token_cnts_by_rank_by_expert, tp_size, n_experts);

    printf("token_cnt_by_expert_by_rank\n");
    print_vector_2d(token_cnt_by_expert_by_rank, n_experts, tp_size);

    printf("token_cnt_by_expert_by_rank_cumsum\n");
    print_vector_2d(token_cnt_by_expert_by_rank_cumsum, n_experts, tp_size);

    printf("stage_index_by_stage_by_expert\n");
    print_vector_2d(stage_index_by_stage_by_expert, tp_size, n_experts);

    printf("num_tiles_by_stage_by_expert\n");
    print_vector_2d(num_tiles_by_stage_by_expert, tp_size, n_experts);

    printf("num_tiles_by_stage_by_expert_cumsum\n");
    print_vector_2d(num_tiles_by_stage_by_expert_cumsum, tp_size, n_experts);

    printf("num_tiles_by_expert_by_stage\n");
    print_vector_2d(num_tiles_by_expert_by_stage, n_experts, tp_size);

    printf("num_tiles_by_expert_by_stage_cumsum\n");
    print_vector_2d(num_tiles_by_expert_by_stage_cumsum, n_experts, tp_size);

    printf("num_tiles_by_stage\n");
    print_vector_2d(num_tiles_by_stage, 1, tp_size);

    printf("num_tiles_by_stage_cumsum\n");
    print_vector_2d(num_tiles_by_stage_cumsum, 1, tp_size);

    printf("num_tiles_by_expert\n");
    print_vector_2d(num_tiles_by_expert, 1, n_experts);

    printf("num_tiles_by_expert_cumsum\n");
    print_vector_2d(num_tiles_by_expert_cumsum, 1, n_experts);

    printf("tile_index_by_expert_by_stage\n");
    print_vector_2d(tile_index_by_expert_by_stage, 1, ntiles_total);
  }

  CUDA_CHECK(cudaFree(buffer));

  std::vector<std::tuple<int, int>> swizzled;
  for (int i = 0; i < ntiles_total; i++) {
    swizzled.emplace_back(output_tiles[i].expert_id, output_tiles[i].tiled_m);
  }
  return swizzled;
}

// verbose: 0 for no verbose, 1 for start/end log, 2 for output swizzled. 3 for
// host log. 4 for kernel log
void check_with_token_cnts(const vector<int> &token_cnts_by_rank_by_expert,
                           int n_experts, int tp_size, int block_size_m,
                           int verbose = 0) {
  auto token_cnts_by_expert_by_rank =
      transpose2d(token_cnts_by_rank_by_expert, tp_size, n_experts);
  int ntokens_total = 0;
  for (const int expert : token_cnts_by_expert_by_rank) {
    ntokens_total += expert;
  }

  int ntiles_total = 0;
  for (int i = 0; i < n_experts; i++) {
    int tokens_this_expert = 0;
    for (int j = 0; j < tp_size; j++) {
      tokens_this_expert += token_cnts_by_expert_by_rank[i * tp_size + j];
    }
    ntiles_total += cdiv(tokens_this_expert, block_size_m);
  }

  for (int rank = 0; rank < tp_size; rank++) {
    vector<tuple<int, int>> swizzled = threadblock_swizzle_ag_moe_cuda(
        rank, n_experts, tp_size, block_size_m, token_cnts_by_rank_by_expert,
        ntokens_total, ntiles_total, nullptr, verbose);
    if (verbose >= 1) {
      cout << "Rank " << rank << " swizzled: ";
      for (auto [eid, tm] : swizzled)
        cout << "(" << eid << "," << tm << ") ";
      cout << endl;
    }

    try {
      check_swizzled(swizzled, token_cnts_by_rank_by_expert, n_experts,
                     tp_size);
    } catch (const exception e) {
      cerr << "token cnts: \n";
      print_vector_2d(token_cnts_by_rank_by_expert, tp_size, n_experts);
      cerr << "swizzled: \n";
      for (auto [eid, tm] : swizzled) {
        cerr << "(" << eid << "," << tm << ") ";
      }
      cerr << "Test failed\n";
      flush(cout);
      throw;
    }
  }
}

int main() {
  int nexperts = 2;
  int tp_size = 4;
  int BLOCK_SIZE_M = 128;

  // Test uniform distributions
  cout << "Testing uniform distributions...\n";
  check_with_token_cnts(generate_uniform_counts_by_rank_by_expert(
                            128 * nexperts, nexperts, tp_size),
                        nexperts, tp_size, BLOCK_SIZE_M, 4);
  check_with_token_cnts(generate_uniform_counts_by_rank_by_expert(
                            127 * nexperts, nexperts, tp_size),
                        nexperts, tp_size, BLOCK_SIZE_M, 3);
  check_with_token_cnts(generate_uniform_counts_by_rank_by_expert(
                            129 * nexperts, nexperts, tp_size),
                        nexperts, tp_size, BLOCK_SIZE_M, 2);

  nexperts = 32;
  tp_size = 8;

  for (int j = 0; j < 10; j++) {
    // Test random distributions
    cout << "\nTesting random distributions...\n";
    for (int i = 0; i < 1000; i++) {
      check_with_token_cnts(generate_random_counts_by_rank_by_expert(
                                128 * nexperts, nexperts, tp_size),
                            nexperts, tp_size, BLOCK_SIZE_M, 0);
    }

    // Test random distributions with many zeros
    cout << "\nTesting random distributions with many zeros...\n";
    for (int i = 0; i < 1000; i++) {
      check_with_token_cnts(generate_random_counts_with_zeros_by_rank_by_expert(
                                128 * nexperts, nexperts, tp_size, 0.3),
                            nexperts, tp_size, BLOCK_SIZE_M, 0);
    }
  }

  cout << "\nAll tests passed successfully!\n";

  return 0;
}
