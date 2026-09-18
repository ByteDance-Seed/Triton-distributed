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

#include "flash_comm/buffer/nccl_symmetric_memory.h"
#include "flash_comm/buffer/nccl_gin.h"
#include "flash_comm/common.h"
#include "flash_comm/nccl_utils.h"
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>

namespace flash_comm {
namespace buffer {

namespace {

struct NcclWinDesc {
  int rank;
  int64_t offset;
};

} // namespace

NcclSymmetricMemory::NcclSymmetricMemory(size_t size, int win_flags)
    : NcclSymmetricMemory(nccl_gin_comm(), size, win_flags) {}

NcclSymmetricMemory::NcclSymmetricMemory(ncclComm_t comm, size_t size,
                                         int win_flags)
    : comm_(comm), size_(size) {
  if (comm_ == nullptr) {
    throw std::runtime_error("NcclSymmetricMemory: comm is null");
  }
  NCCL_CHECK(ncclMemAlloc(&ptr_, size_));
  owns_ptr_ = true;
  NCCL_CHECK(ncclCommWindowRegister(comm_, ptr_, size_, &win_, win_flags));
}

NcclSymmetricMemory::NcclSymmetricMemory(ncclComm_t comm, void *ptr,
                                         size_t size, int win_flags)
    : comm_(comm), ptr_(ptr), size_(size), owns_ptr_(false) {
  if (comm_ == nullptr) {
    throw std::runtime_error("NcclSymmetricMemory: comm is null");
  }
  if (ptr_ == nullptr) {
    throw std::runtime_error("NcclSymmetricMemory: ptr is null");
  }
  NCCL_CHECK(ncclCommWindowRegister(comm_, ptr_, size_, &win_, win_flags));
}

NcclSymmetricMemory::~NcclSymmetricMemory() {
  if (comm_ != nullptr && win_ != nullptr) {
    ncclCommWindowDeregister(comm_, win_);
  }
  if (owns_ptr_ && ptr_ != nullptr) {
    ncclMemFree(ptr_);
  }
}

void *NcclSymmetricMemory::get_peer_ptr(int peer_rank) const {
  if (win_ == nullptr) {
    throw std::runtime_error("NcclSymmetricMemory: window is null");
  }
  // peer_rank is a node-local rank. The LSA team may span more than one node
  // (init allows lsa_size > local_world_size), so translate it to the
  // LSA-domain rank before indexing the flat LSA base: init also guarantees
  // lsa_rank % local_world_size == local_rank on every rank with node-major
  // rank order, so this node's GPUs occupy the contiguous LSA range starting
  // at lsa_rank - local_rank.
  const NcclGinState &st = nccl_gin_require_state();
  const int lsa_peer_rank = (st.lsa_rank - st.local_rank) + peer_rank;
  if (peer_rank < 0 || peer_rank >= st.local_world_size ||
      lsa_peer_rank >= st.lsa_size) {
    throw std::runtime_error("NcclSymmetricMemory: peer rank out of LSA range");
  }
  // Public NCCL headers expose only ncclWinGetUserPtr for the local user
  // pointer.  The host-side LSA peer helper exists in NCCL internals but is not
  // installed, so use the public device window descriptor and mirror
  // ncclGetLsaPointer(win, 0, lsa_peer_rank) for the local LSA domain.
  ncclWindow_vidmem win_desc{};
  CUDA_CHECK(
      cudaMemcpy(&win_desc, win_, sizeof(win_desc), cudaMemcpyDeviceToHost));
  const uintptr_t stride_bytes = static_cast<uintptr_t>(win_desc.stride4G)
                                 << 32;
  return reinterpret_cast<void *>(
      reinterpret_cast<uintptr_t>(win_desc.lsaFlatBase) +
      static_cast<uintptr_t>(lsa_peer_rank) * stride_bytes);
}

std::vector<char> NcclSymmetricMemory::export_descriptor() const {
  NcclWinDesc desc{};
  int rank = 0;
  NCCL_CHECK(ncclCommUserRank(comm_, &rank));
  desc.rank = rank;
  desc.offset = 0;
  std::vector<char> bytes(sizeof(desc));
  std::memcpy(bytes.data(), &desc, sizeof(desc));
  return bytes;
}

void NcclSymmetricMemory::import_peer(int peer_rank,
                                      const std::vector<char> &desc) {
  if (desc.size() != sizeof(NcclWinDesc)) {
    throw std::runtime_error(
        "NcclSymmetricMemory: invalid peer descriptor size");
  }
  NcclWinDesc parsed{};
  std::memcpy(&parsed, desc.data(), sizeof(parsed));
  if (parsed.rank != peer_rank) {
    throw std::runtime_error(
        "NcclSymmetricMemory: peer rank mismatch in descriptor");
  }
  peer_team_ranks_.push_back(peer_rank);
}

} // namespace buffer
} // namespace flash_comm
