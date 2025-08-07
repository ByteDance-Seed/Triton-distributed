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
#include <hip/hip_runtime.h>

#include <rocshmem/rocshmem.hpp>
using namespace rocshmem;

extern "C" {

__device__ int __attribute__((visibility("default"))) rocshmem_my_pe_wrapper() {
  return rocshmem_my_pe();
}


__device__ void __attribute__((visibility("default"))) rocshmem_set_rocshmem_ctx(
  void *ctx) {
  ROCSHMEM_CTX_DEFAULT.ctx_opaque = ctx;
}

__device__ int __attribute__((visibility("default"))) rocshmem_n_pes_wrapper() {
  return rocshmem_n_pes();
}

__device__ void * __attribute__((visibility("default"))) rocshmem_ptr_wrapper(void *dest,
                                                                     int pe) {
  return rocshmem_ptr(dest, pe);
}

__device__ void __attribute__((visibility("default"))) rocshmem_int_p_wrapper(
    int *dest, int value, int pe) {
  rocshmem_int_p(dest, value, pe);
}

}

