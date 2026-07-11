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

#include "flash_comm/buffer/shareable_block.h"
#include <memory>
#include <unordered_map>
#include <vector>

namespace flash_comm {
namespace buffer {

class SymmetricMemory {
public:
  // backend: CUDA_IPC or VMM.
  SymmetricMemory(size_t size, BlockBackend backend);
  ~SymmetricMemory();

  void *get_local_ptr() const;

  // Returns {backend, handle_bytes}
  std::pair<BlockBackend, std::vector<char>> get_handle() const;

  // Register a peer's handle
  void register_peer(int peer_rank, const std::vector<char> &handle_data);

  void *get_peer_ptr(int peer_rank) const;

private:
  size_t size_;
  BlockBackend backend_;

  std::unique_ptr<ShareableBlock> block_;

  std::unordered_map<int, void *> peer_ptrs_;
};

} // namespace buffer
} // namespace flash_comm
