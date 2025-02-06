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
#include "nvshmem_wrapper.h"

extern "C" {

__device__ int nvshmem_my_pe_wrapper() { return nvshmem_my_pe(); }

__device__ int nvshmem_n_pes_wrapper() { return nvshmem_n_pes(); }

__device__ void nvshmem_int_p_wrapper(int *destination, int mype, int peer) {
  nvshmem_int_p(destination, mype, peer);
}

__device__ void *nvshmem_ptr_wrapper(void *ptr, int peer) {
  return nvshmem_ptr(ptr, peer);
}

__device__ void nvshmemx_signal_op_wrapper(uint64_t *sig_addr, uint64_t signal,
                                           int sig_op, int pe) {
  return nvshmemx_signal_op(sig_addr, signal, sig_op, pe);
}
}
