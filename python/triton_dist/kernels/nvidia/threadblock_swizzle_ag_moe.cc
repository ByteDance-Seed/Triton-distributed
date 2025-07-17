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
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

class LazyLogger {
public:
  LazyLogger(bool no_error = false) {
    _no_print = no_error;
    _no_error = no_error;
  };

  ~LazyLogger() noexcept(false) {
    if (!_no_print) {
      std::cerr << _message.str() << std::endl;
    }
    if (!_no_error) {
      throw std::runtime_error(_message.str());
    }
  }

  template <typename T> LazyLogger &operator<<(const T &value) {
    _message << value;
    return *this;
  }

private:
  bool _no_print = false;
  bool _no_error = false;
  std::ostringstream _message;
};

// Base CHECK macro with stream support
#define CHECK(cond)                                                            \
  LazyLogger(cond) << __FILE__ << ":" << __LINE__                              \
                   << " Check failed: " #cond ". "

using namespace std;

// Comparison macros
#define CHECK_EQ(a, b) CHECK((a) == (b)) << " (" << (a) << " vs " << (b) << ") "
#define CHECK_NE(a, b) CHECK((a) != (b))
#define CHECK_LT(a, b) CHECK((a) < (b))
#define CHECK_LE(a, b) CHECK((a) <= (b))
#define CHECK_GT(a, b) CHECK((a) > (b))
#define CHECK_GE(a, b) CHECK((a) >= (b))

// Helper class for stream accumulation
class CheckMessage {
public:
  std::ostringstream &stream() { return ss_; }

  ~CheckMessage() noexcept(false) { throw std::runtime_error(ss_.str()); }

private:
  std::ostringstream ss_;
};

struct Tile {
  int expert_id;
  int tiled_m;
  int segment_start;
  int segment_end;

  Tile(int eid, int tm, int ss, int se)
      : expert_id(eid), tiled_m(tm), segment_start(ss), segment_end(se) {}
};

int cdiv(int x, int y) { return (x - 1 + y) / y; }

vector<int> cumsum(const vector<int> &x) {
  vector<int> y;
  int s = 0;
  for (int num : x) {
    s += num;
    y.push_back(s);
  }
  return y;
}

vector<vector<Tile>>
_split_tiles_for_each_segment(int expert_id, int rank, int tp_size,
                              int block_size_m, const vector<int> &token_cnts) {

  vector<vector<Tile>> tiles(tp_size);
  vector<int> token_cnts_acc = cumsum(token_cnts);
  int ntokens = token_cnts_acc.back();
  int global_m_start = token_cnts_acc[rank] - token_cnts[rank];
  int global_tiled_m_start = cdiv(global_m_start, block_size_m);
  int n_tiles = cdiv(ntokens, block_size_m);

  vector<int> tids;
  for (int i = global_tiled_m_start; i < n_tiles; i++)
    tids.push_back(i);
  for (int i = 0; i < global_tiled_m_start; i++)
    tids.push_back(i);

  for (int tid : tids) {
    int m_start = tid * block_size_m;
    int m_end = min((tid + 1) * block_size_m, ntokens) - 1;

    auto it_start =
        upper_bound(token_cnts_acc.begin(), token_cnts_acc.end(), m_start);
    int segment_start = it_start - token_cnts_acc.begin();

    auto it_end =
        upper_bound(token_cnts_acc.begin(), token_cnts_acc.end(), m_end);
    int segment_end = it_end - token_cnts_acc.begin();

    int stage = (segment_end - rank + tp_size) % tp_size;

    if (tid == global_tiled_m_start - 1 && global_m_start % block_size_m != 0) {
      auto it = upper_bound(token_cnts_acc.begin(),
                            token_cnts_acc.begin() + rank, m_end);
      int m_segment_end = it - token_cnts_acc.begin();
      stage = (m_segment_end - rank + tp_size) % tp_size;
    }

    tiles[stage].emplace_back(expert_id, tid, segment_start, segment_end);
  }
  return tiles;
}

template <typename T>
vector<vector<T>> transpose2d(const vector<vector<T>> &arr_2d) {
  if (arr_2d.empty())
    return {};
  size_t dim0 = arr_2d.size();
  if (arr_2d[0].empty()) {
    return {};
  }
  size_t dim1 = arr_2d[0].size();

  vector<T> flatten;
  for (const auto &inner : arr_2d)
    for (const auto &item : inner)
      flatten.push_back(item);

  vector<vector<T>> reshaped(dim1);
  for (size_t i = 0; i < flatten.size(); i++)
    reshaped[i % dim1].push_back(flatten[i]);

  return reshaped;
}

tuple<int, int, Tile> threadblock_swizzle_ag_moe(
    int tiled_m, int rank, int nexperts, int tp_size, int block_size_m,
    const vector<vector<int>> &token_cnts_per_rank_per_expert) {

  auto token_cnts_per_expert_per_rank =
      transpose2d(token_cnts_per_rank_per_expert);
  vector<vector<vector<Tile>>> tiles_by_expert_by_segment;

  for (int expert_id = 0; expert_id < nexperts; expert_id++) {
    auto tiles = _split_tiles_for_each_segment(
        expert_id, rank, tp_size, block_size_m,
        token_cnts_per_expert_per_rank[expert_id]);
    tiles_by_expert_by_segment.push_back(tiles);
  }

  auto tiles_by_segment_by_expert = transpose2d(tiles_by_expert_by_segment);
  vector<vector<int>> ntiles_by_segment_by_expert;
  for (auto &vec : tiles_by_segment_by_expert) {
    vector<int> sizes;
    for (auto &t : vec)
      sizes.push_back(t.size());
    ntiles_by_segment_by_expert.push_back(sizes);
  }

  vector<vector<int>> ntiles_acc_by_segment_by_expert;

  for (auto &vec : tiles_by_segment_by_expert) {
    vector<int> sizes;
    for (auto &t : vec)
      sizes.push_back(t.size());
    ntiles_acc_by_segment_by_expert.push_back(cumsum(sizes));
  }

  vector<int> ntiles_by_segment;
  for (auto &vec : ntiles_by_segment_by_expert)
    ntiles_by_segment.push_back(accumulate(vec.begin(), vec.end(), 0));

  vector<int> ntiles_acc_by_segment = cumsum(ntiles_by_segment);

  int stage = upper_bound(ntiles_acc_by_segment.begin(),
                          ntiles_acc_by_segment.end(), tiled_m) -
              ntiles_acc_by_segment.begin();

  int tiled_m_in_rank =
      tiled_m - (stage > 0 ? ntiles_acc_by_segment[stage - 1] : 0);
  auto &acc = ntiles_acc_by_segment_by_expert[stage];
  int expert_id =
      upper_bound(acc.begin(), acc.end(), tiled_m_in_rank) - acc.begin();

  int tiled_m_in_problem =
      tiled_m_in_rank - (expert_id > 0 ? acc[expert_id - 1] : 0);

  return make_tuple(
      stage, expert_id,
      tiles_by_segment_by_expert[stage][expert_id][tiled_m_in_problem]);
}

vector<vector<int>>
generate_uniform_counts_per_rank_per_expert(int ntokens_per_rank, int nexperts,
                                            int tp_size) {
  CHECK(ntokens_per_rank % nexperts == 0);
  vector<vector<int>> counts(
      tp_size, vector<int>(nexperts, ntokens_per_rank / nexperts));
  return counts;
}

vector<vector<int>>
generate_random_counts_per_rank_per_expert(int ntokens_per_rank, int nexperts,
                                           int tp_size) {
  random_device rd;
  mt19937 gen(rd());
  vector<vector<int>> counts(tp_size, vector<int>(nexperts));

  for (auto &row : counts) {
    discrete_distribution<> d(nexperts, 0, nexperts - 1,
                              [=](int) { return 1.0 / nexperts; });
    for (int i = 0; i < ntokens_per_rank; i++)
      row[d(gen)]++;
  }
  return counts;
}

vector<vector<int>> generate_random_counts_with_zeros_per_rank_per_expert(
    int ntokens_per_rank, int nexperts, int tp_size, float zero_rate) {
  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dist(0.0, 1.0);

  vector<vector<int>> counts(tp_size, vector<int>(nexperts, 0));

  for (auto &row : counts) {
    vector<double> weights;
    for (int i = 0; i < nexperts; i++) {
      weights.push_back(dist(gen) > zero_rate ? 1.0 : 1e-5);
    }
    double sum = accumulate(weights.begin(), weights.end(), 0.0);
    for (auto &w : weights)
      w /= sum;

    discrete_distribution<> d(weights.begin(), weights.end());
    for (int i = 0; i < ntokens_per_rank; i++)
      row[d(gen)]++;
  }
  return counts;
}

void check_swizzled(const vector<pair<int, int>> &swizzled,
                    const vector<vector<int>> &token_cnts) {
  auto reshaped = transpose2d(token_cnts);
  for (size_t expert_id = 0; expert_id < reshaped.size(); expert_id++) {
    int total =
        accumulate(reshaped[expert_id].begin(), reshaped[expert_id].end(), 0);
    vector<int> tiles;
    for (auto [eid, tm] : swizzled)
      if (eid == expert_id)
        tiles.push_back(tm);

    sort(tiles.begin(), tiles.end());
    for (size_t i = 0; i < tiles.size(); i++)
      assert(tiles[i] == i);
  }
}

void print_vector_2d(const vector<vector<int>> &vec) {
  for (const auto &row : vec) {
    for (int num : row)
      cout << num << " ";
    cout << endl;
  }
}

void check_with_token_cnts(const vector<vector<int>> &token_cnts, int tp_size,
                           int n_experts, int BLOCK_SIZE_M,
                           bool verbose = true) {

  auto token_cnts_per_expert = transpose2d(token_cnts);
  int ntokens_total = 0;
  for (const auto &expert : token_cnts_per_expert)
    ntokens_total += accumulate(expert.begin(), expert.end(), 0);

  int ntiles_total = 0;
  for (const auto &expert : token_cnts_per_expert) {
    int tokens = accumulate(expert.begin(), expert.end(), 0);
    ntiles_total += cdiv(tokens, BLOCK_SIZE_M);
  }

  for (int rank = 0; rank < tp_size; rank++) {
    vector<pair<int, int>> swizzled;
    for (int tiled_m = 0; tiled_m < ntiles_total; tiled_m++) {
      auto [stage, expert_id, tile] = threadblock_swizzle_ag_moe(
          tiled_m, rank, n_experts, tp_size, BLOCK_SIZE_M, token_cnts);
      swizzled.emplace_back(expert_id, tile.tiled_m);
    }

    if (verbose) {
      cout << "Rank " << rank << " swizzled: ";
      for (auto [eid, tm] : swizzled)
        cout << "(" << eid << "," << tm << ") ";
      cout << endl;
    }

    try {
      check_swizzled(swizzled, token_cnts);
    } catch (const exception &e) {
      cerr << "Test failed for rank " << rank << " with token counts:\n";
      print_vector_2d(token_cnts);
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
  check_with_token_cnts(generate_uniform_counts_per_rank_per_expert(
                            BLOCK_SIZE_M * nexperts, nexperts, tp_size),
                        tp_size, nexperts, BLOCK_SIZE_M);
  check_with_token_cnts(generate_uniform_counts_per_rank_per_expert(
                            (BLOCK_SIZE_M - 1) * nexperts, nexperts, tp_size),
                        tp_size, nexperts, BLOCK_SIZE_M);
  check_with_token_cnts(generate_uniform_counts_per_rank_per_expert(
                            (BLOCK_SIZE_M + 1) * nexperts, nexperts, tp_size),
                        tp_size, nexperts, BLOCK_SIZE_M);

  nexperts = 32;
  tp_size = 8;
  for (int j = 0; j < 100; j++) {
    // Test random distributions
    cout << "\nTesting random distributions...\n";
    for (int i = 0; i < 100; i++) {
      check_with_token_cnts(generate_random_counts_per_rank_per_expert(
                                BLOCK_SIZE_M * nexperts, nexperts, tp_size),
                            tp_size, nexperts, BLOCK_SIZE_M, false);
    }

    // Test random distributions with many zeros
    cout << "\nTesting distributions with zeros...\n";
    for (int i = 0; i < 100; i++) {
      check_with_token_cnts(
          generate_random_counts_with_zeros_per_rank_per_expert(
              BLOCK_SIZE_M * nexperts, nexperts, tp_size, 0.3),
          tp_size, nexperts, BLOCK_SIZE_M, false);
    }
  }

  cout << "\nAll tests passed successfully!\n";

  return 0;
}
