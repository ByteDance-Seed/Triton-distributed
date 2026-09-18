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

#include <cstddef>
#include <cstdint>
#include <nccl.h>
#include <vector>

namespace flash_comm {
namespace buffer {

// NCCL-registered symmetric buffer/window for device GIN put/get.
class NcclSymmetricMemory {
public:
  // Uses the initialized NCCL GIN comm.
  explicit NcclSymmetricMemory(size_t size,
                               int win_flags = NCCL_WIN_COLL_SYMMETRIC);
  NcclSymmetricMemory(ncclComm_t comm, size_t size, int win_flags);
  NcclSymmetricMemory(ncclComm_t comm, void *ptr, size_t size, int win_flags);
  ~NcclSymmetricMemory();

  NcclSymmetricMemory(const NcclSymmetricMemory &) = delete;
  NcclSymmetricMemory &operator=(const NcclSymmetricMemory &) = delete;

  void *get_local_ptr() const { return ptr_; }
  void *get_peer_ptr(int peer_rank) const;
  ncclWindow_t get_window() const { return win_; }
  size_t get_size() const { return size_; }

  // Serialized window descriptor for peers (opaque bytes).
  std::vector<char> export_descriptor() const;

  // Import peer descriptor (same size buffer on all ranks).
  void import_peer(int peer_rank, const std::vector<char> &desc);

  int peer_rank_for_team(int team_rank) const {
    return peer_team_ranks_[team_rank];
  }

private:
  ncclComm_t comm_ = nullptr;
  void *ptr_ = nullptr;
  ncclWindow_t win_ = nullptr;
  size_t size_ = 0;
  bool owns_ptr_ = false;
  std::vector<int> peer_team_ranks_;
};

} // namespace buffer
} // namespace flash_comm
