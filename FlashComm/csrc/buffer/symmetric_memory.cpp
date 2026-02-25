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

#include "flash_comm/buffer/symmetric_memory.h"
#include <iostream>
#include <stdexcept>

namespace flash_comm {
namespace buffer {

SymmetricMemory::SymmetricMemory(size_t size, BlockBackend backend)
    : size_(size), backend_(backend) {

  block_ = std::make_unique<ShareableBlock>(size, backend_);
  size_ = block_->get_size();
}

SymmetricMemory::~SymmetricMemory() {
  for (auto &pair : peer_ptrs_) {
    int rank = pair.first;
    void *ptr = pair.second;
    try {
      block_->unmap_remote(ptr, size_);
    } catch (const std::exception &e) {
      std::cerr << "[FlashComm Warning] Failed to unmap peer ptr for rank "
                << rank << ": " << e.what() << std::endl;
    } catch (...) {
      std::cerr << "[FlashComm Warning] Failed to unmap peer ptr for rank "
                << rank << ": Unknown error" << std::endl;
    }
  }
}

void *SymmetricMemory::get_local_ptr() const { return block_->get_ptr(); }

std::pair<BlockBackend, std::vector<char>> SymmetricMemory::get_handle() const {
  return {block_->get_backend(), block_->get_handle_data()};
}

void SymmetricMemory::register_peer(int peer_rank,
                                    const std::vector<char> &handle_data) {
  if (peer_ptrs_.find(peer_rank) != peer_ptrs_.end()) {
    return;
  }

  void *ptr = block_->map_remote(handle_data, size_);
  peer_ptrs_[peer_rank] = ptr;
}

void *SymmetricMemory::get_peer_ptr(int peer_rank) const {
  auto it = peer_ptrs_.find(peer_rank);
  if (it == peer_ptrs_.end()) {
    return nullptr;
  }
  return it->second;
}

} // namespace buffer
} // namespace flash_comm
