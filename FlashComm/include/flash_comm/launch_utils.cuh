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

#include "flash_comm/common.h"

namespace flash_comm {

inline void launch_kernel_ex(const void *kernel, dim3 grid_dim, dim3 block_dim,
                             void **kernel_args, size_t smem_size,
                             cudaStream_t stream, int32_t cluster_size = 0,
                             bool cooperative = false) {
#if __CUDACC_VER_MAJOR__ >= 12
  cudaLaunchConfig_t config = {};
  config.gridDim = grid_dim;
  config.blockDim = block_dim;
  config.dynamicSmemBytes = smem_size;
  config.stream = stream;

  cudaLaunchAttribute attrs[2];
  int num_attrs = 0;
  if (cooperative) {
    attrs[num_attrs].id = cudaLaunchAttributeCooperative;
    attrs[num_attrs++].val.cooperative = 1;
  }
  if (cluster_size > 0) {
    attrs[num_attrs].id = cudaLaunchAttributeClusterDimension;
    attrs[num_attrs].val.clusterDim.x = cluster_size;
    attrs[num_attrs].val.clusterDim.y = 1;
    attrs[num_attrs++].val.clusterDim.z = 1;
  }
  config.attrs = num_attrs > 0 ? attrs : nullptr;
  config.numAttrs = num_attrs;

  CUDA_CHECK(cudaLaunchKernelExC(&config, kernel, kernel_args));
#else
  if (cooperative) {
    CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, grid_dim, block_dim,
                                           kernel_args, smem_size, stream));
  } else {
    CUDA_CHECK(cudaLaunchKernel(kernel, grid_dim, block_dim, kernel_args,
                                smem_size, stream));
  }
#endif
}

template <typename KernelFunc, typename... Args>
inline void launch_kernel_ex(KernelFunc kernel, dim3 grid_dim, dim3 block_dim,
                             size_t smem_size, cudaStream_t stream,
                             int32_t cluster_size, Args... args) {
  void *kernel_args[] = {
      const_cast<void *>(static_cast<const void *>(&args))...};
  launch_kernel_ex(reinterpret_cast<const void *>(kernel), grid_dim, block_dim,
                   kernel_args, smem_size, stream, cluster_size, false);
}

} // namespace flash_comm
