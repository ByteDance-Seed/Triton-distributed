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

__device__ void __attribute__((visibility("default")))
rocshmem_set_rocshmem_ctx(void *ctx) {
  ROCSHMEM_CTX_DEFAULT.ctx_opaque = ctx;
}

__device__ int __attribute__((visibility("default"))) rocshmem_n_pes_wrapper() {
  return rocshmem_n_pes();
}

__device__ void *__attribute__((visibility("default")))
rocshmem_ptr_wrapper(void *dest, int pe) {
  return rocshmem_ptr(dest, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_int_p_wrapper(int *dest, int value, int pe) {
  rocshmem_int_p(dest, value, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_signal_wrapper(void *dest, const void *source, size_t nbytes, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem_putmem_signal(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_signal_wg_wrapper(void *dest, const void *source, size_t nbytes, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem_putmem_signal_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_signal_wave_wrapper(void *dest, const void *source, size_t nbytes, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem_putmem_signal_wave(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_signal_nbi_wrapper(void *dest, const void *source, size_t nbytes, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem_putmem_signal_nbi(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_signal_nbi_wg_wrapper(void *dest, const void *source, size_t nbytes, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem_putmem_signal_nbi_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_signal_nbi_wave_wrapper(void *dest, const void *source, size_t nbytes, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) {
  rocshmem_putmem_signal_nbi_wave(dest, source, nbytes, sig_addr, signal, sig_op, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_getmem_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_getmem(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_getmem_wave_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_getmem_wave(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_getmem_wg_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_getmem_wg(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_getmem_nbi_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_getmem_nbi(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_getmem_nbi_wave_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_getmem_nbi_wave(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_getmem_nbi_wg_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_getmem_nbi_wg(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_putmem(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_wave_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_putmem_wave(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_wg_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_putmem_wg(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_nbi_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_putmem_nbi(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_nbi_wave_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_putmem_nbi_wave(dest, source, nbytes, pe);
}

__device__ void __attribute__((visibility("default")))
rocshmem_putmem_nbi_wg_wrapper(void *dest, const void *source, size_t nbytes, int pe) {
  rocshmem_putmem_nbi_wg(dest, source, nbytes, pe);
}


__device__ void __attribute__((visibility("default")))
rocshmem_wait_until_wrapper(void *sig_addr, int cmp, uint64_t cmp_val) {
  rocshmem_wait_until(sig_addr, cmp, cmp_val);
}

__device__ void __attribute__((visibility("default")))
rocshmem_barrier_all_wrapper() {
  rocshmem_barrier_all();
}

__device__ void __attribute__((visibility("default")))
rocshmem_barrier_all_wg_wrapper() {
  rocshmem_barrier_all_wg();
}

__device__ void __attribute__((visibility("default")))
rocshmem_barrier_all_wave_wrapper() {
  rocshmem_barrier_all_wave();
}

__device__ void __attribute__((visibility("default")))
rocshmem_fence_wave_wrapper() {
  rocshmem_fence();
}
}
