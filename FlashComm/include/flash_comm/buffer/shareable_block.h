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

#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace flash_comm {
namespace buffer {

enum class BlockBackend {
  CUDA_IPC, // cudaMalloc + cudaIpcGetMemHandle
  VMM       // cuMemCreate + cuMemExportToShareableHandle (Fabric only)
};

class ShareableBlock {
public:
  // For VMM backend, Fabric handle is always used.
  // For CUDA_IPC backend, cudaIpcMemHandle is used.
  ShareableBlock(size_t size, BlockBackend backend);
  ~ShareableBlock();

  void *get_ptr() const { return ptr_; }
  size_t get_size() const { return size_; }
  BlockBackend get_backend() const { return backend_; }

  // CUDA_IPC: returns cudaIpcMemHandle_t bytes.
  // VMM: returns Fabric handle bytes.
  std::vector<char> get_handle_data() const;

  // Map a peer's handle, returns the mapped pointer.
  void *map_remote(const std::vector<char> &handle, size_t size);

  // Unmap a previously mapped remote pointer.
  void unmap_remote(void *ptr, size_t size);

  static bool is_fabric_supported(int device);
  static bool is_vmm_supported(int device);

private:
  void allocate_cuda_ipc(size_t size);
  void allocate_vmm(size_t size);

  void *ptr_ = nullptr;
  size_t size_;
  BlockBackend backend_;

  CUmemGenericAllocationHandle vmm_handle_ = 0;
  bool has_vmm_handle_ = false;
};

} // namespace buffer
} // namespace flash_comm
