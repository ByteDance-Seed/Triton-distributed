/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the
 * following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 * Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
 */

#pragma once

#include <cooperative_groups.h>
#include <cstdint>
#include <nccl_device.h>

namespace flash_comm {
namespace ep {
namespace kernels {

constexpr int32_t kEpWorldBarrierId = 0;

__device__ __forceinline__ void ep_world_barrier(ncclDevComm dev_comm,
                                                 int32_t barrier_context,
                                                 bool has_remote_peers) {
  if (has_remote_peers) {
    ncclGin gin{dev_comm, barrier_context};
    ncclBarrierSession<ncclCoopCta> barrier{ncclCoopCta(), ncclTeamTagWorld(),
                                            gin, kEpWorldBarrierId};
    barrier.sync(ncclCoopCta(), cuda::memory_order_acquire,
                 ncclGinFenceLevel::Relaxed);
  } else {
    ncclLsaBarrierSession<ncclCoopCta> barrier{
        ncclCoopCta(), dev_comm, ncclTeamTagLsa(), kEpWorldBarrierId};
    barrier.sync(ncclCoopCta(), cuda::memory_order_acquire);
  }
}

__device__ __forceinline__ void
ep_world_barrier_after_puts(ncclDevComm dev_comm, int32_t barrier_context,
                            int32_t put_context, bool has_remote_peers) {
  if (has_remote_peers) {
    ncclGin gin{dev_comm, put_context};
    if (threadIdx.x == 0) {
      gin.flush(ncclCoopThread(), cuda::memory_order_release);
    }
  }
  __syncthreads();
  ep_world_barrier(dev_comm, barrier_context, has_remote_peers);
}

} // namespace kernels
} // namespace ep
} // namespace flash_comm
