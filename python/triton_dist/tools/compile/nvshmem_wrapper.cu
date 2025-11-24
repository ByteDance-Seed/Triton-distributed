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
#include <nvshmem.h>
#include <nvshmemx.h>

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

__device__ void *nvshmemx_mc_ptr_wrapper(nvshmem_team_t team, const void *ptr) {
  return nvshmemx_mc_ptr(team, ptr);
}

__device__ void nvshmem_barrier_all_wrapper() { nvshmem_barrier_all(); }

__device__ void nvshmemx_barrier_all_warp_wrapper() {
  nvshmemx_barrier_all_warp();
}

__device__ void nvshmemx_barrier_all_block_wrapper() {
  nvshmemx_barrier_all_block();
}

__device__ void nvshmemx_sync_all_warp_wrapper() {
  nvshmemx_sync_all_warp_wrapper();
}

__device__ void nvshmemx_sync_all_block_wrapper() { nvshmemx_sync_all_block(); }

__device__ void nvshmem_quiet_wrapper() { nvshmem_quiet(); }

__device__ void nvshmem_fence_wrapper() { nvshmem_fence(); }

__device__ void nvshmemx_getmem_nbi_block_wrapper(void *dest,
                                                  const void *source,
                                                  size_t bytes, int pe) {
  nvshmemx_getmem_nbi_block(dest, source, bytes, pe);
}

__device__ void nvshmemx_getmem_block_wrapper(void *dest, const void *source,
                                              size_t bytes, int pe) {
  nvshmemx_getmem_block(dest, source, bytes, pe);
}

__device__ void nvshmemx_getmem_nbi_warp_wrapper(void *dest, const void *source,
                                                 size_t bytes, int pe) {
  nvshmemx_getmem_nbi_warp(dest, source, bytes, pe);
}

__device__ void nvshmemx_getmem_warp_wrapper(void *dest, const void *source,
                                             size_t bytes, int pe) {
  nvshmemx_getmem_warp(dest, source, bytes, pe);
}

__device__ void nvshmem_getmem_nbi_wrapper(void *dest, const void *source,
                                           size_t bytes, int pe) {
  nvshmem_getmem_nbi(dest, source, bytes, pe);
}

__device__ void nvshmem_getmem_wrapper(void *dest, const void *source,
                                       size_t bytes, int pe) {
  nvshmem_getmem(dest, source, bytes, pe);
}

__device__ void nvshmemx_putmem_warp_wrapper(void *dest, const void *source,
                                             size_t bytes, int pe) {
  nvshmemx_putmem_warp(dest, source, bytes, pe);
}
__device__ void nvshmemx_putmem_block_wrapper(void *dest, const void *source,
                                              size_t bytes, int pe) {
  nvshmemx_putmem_block(dest, source, bytes, pe);
}

__device__ void nvshmemx_putmem_nbi_warp_wrapper(void *dest, const void *source,
                                                 size_t bytes, int pe) {
  nvshmemx_putmem_nbi_warp(dest, source, bytes, pe);
}

__device__ void nvshmemx_putmem_nbi_block_wrapper(void *dest,
                                                  const void *source,
                                                  size_t nelems, int pe) {
  return nvshmemx_putmem_nbi_block(dest, source, nelems, pe);
}

__device__ void nvshmem_putmem_wrapper(void *dest, const void *source,
                                       size_t bytes, int pe) {
  nvshmem_putmem(dest, source, bytes, pe);
}

__device__ void nvshmem_putmem_nbi_wrapper(void *dest, const void *source,
                                           size_t bytes, int pe) {
  nvshmem_putmem_nbi(dest, source, bytes, pe);
}

__device__ void nvshmem_putmem_signal_wrapper(void *dest, const void *source,
                                              size_t bytes, uint64_t *sig_addr,
                                              uint64_t signal, int sig_op,
                                              int pe) {
  nvshmem_putmem_signal(dest, source, bytes, sig_addr, signal, sig_op, pe);
}

__device__ void
nvshmem_putmem_signal_nbi_wrapper(void *dest, const void *source, size_t bytes,
                                  uint64_t *sig_addr, uint64_t signal,
                                  int sig_op, int pe) {
  nvshmem_putmem_signal_nbi(dest, source, bytes, sig_addr, signal, sig_op, pe);
}

__device__ void
nvshmemx_putmem_signal_block_wrapper(void *dest, const void *source,
                                     size_t nelems, uint64_t *sig_addr,
                                     uint64_t signal, int sig_op, int pe) {
  nvshmemx_putmem_signal_block(dest, source, nelems, sig_addr, signal, sig_op,
                               pe);
}

__device__ void
nvshmemx_putmem_signal_nbi_block_wrapper(void *dest, const void *source,
                                         size_t nelems, uint64_t *sig_addr,
                                         uint64_t signal, int sig_op, int pe) {
  nvshmemx_putmem_signal_nbi_block(dest, source, nelems, sig_addr, signal,
                                   sig_op, pe);
}

__device__ void
nvshmemx_putmem_signal_warp_wrapper(void *dest, const void *source,
                                    size_t nelems, uint64_t *sig_addr,
                                    uint64_t signal, int sig_op, int pe) {
  nvshmemx_putmem_signal_warp(dest, source, nelems, sig_addr, signal, sig_op,
                              pe);
}

__device__ void
nvshmemx_putmem_signal_nbi_warp_wrapper(void *dest, const void *source,
                                        size_t nelems, uint64_t *sig_addr,
                                        uint64_t signal, int sig_op, int pe) {
  nvshmemx_putmem_signal_nbi_warp(dest, source, nelems, sig_addr, signal,
                                  sig_op, pe);
}

__device__ uint64_t nvshmem_signal_wait_until_wrapper(uint64_t *sig_addr,
                                                      int cmp,
                                                      uint64_t cmp_val) {
  return nvshmem_signal_wait_until(sig_addr, cmp, cmp_val);
}

__device__ void nvshmem_sync_all_wrapper() { nvshmem_sync_all(); }

__device__ int nvshmemx_int8_broadcast_warp_wrapper(nvshmem_team_t team,
                                                    int8_t *dest,
                                                    const int8_t *source,
                                                    size_t nelems,
                                                    int PE_root) {
  return nvshmemx_int8_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int16_broadcast_warp_wrapper(nvshmem_team_t team,
                                                     int16_t *dest,
                                                     const int16_t *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_int16_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int32_broadcast_warp_wrapper(nvshmem_team_t team,
                                                     int32_t *dest,
                                                     const int32_t *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_int32_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int64_broadcast_warp_wrapper(nvshmem_team_t team,
                                                     int64_t *dest,
                                                     const int64_t *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_int64_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint8_broadcast_warp_wrapper(nvshmem_team_t team,
                                                     uint8_t *dest,
                                                     const uint8_t *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_uint8_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint16_broadcast_warp_wrapper(nvshmem_team_t team,
                                                      uint16_t *dest,
                                                      const uint16_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_uint16_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint32_broadcast_warp_wrapper(nvshmem_team_t team,
                                                      uint32_t *dest,
                                                      const uint32_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_uint32_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint64_broadcast_warp_wrapper(nvshmem_team_t team,
                                                      uint64_t *dest,
                                                      const uint64_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_uint64_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_half_broadcast_warp_wrapper(nvshmem_team_t team,
                                                    half *dest,
                                                    const half *source,
                                                    size_t nelems,
                                                    int PE_root) {
  return nvshmemx_half_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_bfloat16_broadcast_warp_wrapper(
    nvshmem_team_t team, __nv_bfloat16 *dest, const __nv_bfloat16 *source,
    size_t nelems, int PE_root) {
  return nvshmemx_bfloat16_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_float_broadcast_warp_wrapper(nvshmem_team_t team,
                                                     float *dest,
                                                     const float *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_float_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_double_broadcast_warp_wrapper(nvshmem_team_t team,
                                                      double *dest,
                                                      const double *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_double_broadcast_warp(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_int8_broadcast_wrapper(nvshmem_team_t team, int8_t *dest,
                                              const int8_t *source,
                                              size_t nelems, int PE_root) {
  return nvshmem_int8_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_int16_broadcast_wrapper(nvshmem_team_t team,
                                               int16_t *dest,
                                               const int16_t *source,
                                               size_t nelems, int PE_root) {
  return nvshmem_int16_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_int32_broadcast_wrapper(nvshmem_team_t team,
                                               int32_t *dest,
                                               const int32_t *source,
                                               size_t nelems, int PE_root) {
  return nvshmem_int32_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_int64_broadcast_wrapper(nvshmem_team_t team,
                                               int64_t *dest,
                                               const int64_t *source,
                                               size_t nelems, int PE_root) {
  return nvshmem_int64_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_uint8_broadcast_wrapper(nvshmem_team_t team,
                                               uint8_t *dest,
                                               const uint8_t *source,
                                               size_t nelems, int PE_root) {
  return nvshmem_uint8_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_uint16_broadcast_wrapper(nvshmem_team_t team,
                                                uint16_t *dest,
                                                const uint16_t *source,
                                                size_t nelems, int PE_root) {
  return nvshmem_uint16_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_uint32_broadcast_wrapper(nvshmem_team_t team,
                                                uint32_t *dest,
                                                const uint32_t *source,
                                                size_t nelems, int PE_root) {
  return nvshmem_uint32_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_uint64_broadcast_wrapper(nvshmem_team_t team,
                                                uint64_t *dest,
                                                const uint64_t *source,
                                                size_t nelems, int PE_root) {
  return nvshmem_uint64_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_half_broadcast_wrapper(nvshmem_team_t team, half *dest,
                                              const half *source, size_t nelems,
                                              int PE_root) {
  return nvshmem_half_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_bfloat16_broadcast_wrapper(nvshmem_team_t team,
                                                  __nv_bfloat16 *dest,
                                                  const __nv_bfloat16 *source,
                                                  size_t nelems, int PE_root) {
  return nvshmem_bfloat16_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_float_broadcast_wrapper(nvshmem_team_t team, float *dest,
                                               const float *source,
                                               size_t nelems, int PE_root) {
  return nvshmem_float_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_double_broadcast_wrapper(nvshmem_team_t team,
                                                double *dest,
                                                const double *source,
                                                size_t nelems, int PE_root) {
  return nvshmem_double_broadcast(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int8_broadcast_block_wrapper(nvshmem_team_t team,
                                                     int8_t *dest,
                                                     const int8_t *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_int8_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int16_broadcast_block_wrapper(nvshmem_team_t team,
                                                      int16_t *dest,
                                                      const int16_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_int16_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int32_broadcast_block_wrapper(nvshmem_team_t team,
                                                      int32_t *dest,
                                                      const int32_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_int32_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_int64_broadcast_block_wrapper(nvshmem_team_t team,
                                                      int64_t *dest,
                                                      const int64_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_int64_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint8_broadcast_block_wrapper(nvshmem_team_t team,
                                                      uint8_t *dest,
                                                      const uint8_t *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_uint8_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint16_broadcast_block_wrapper(nvshmem_team_t team,
                                                       uint16_t *dest,
                                                       const uint16_t *source,
                                                       size_t nelems,
                                                       int PE_root) {
  return nvshmemx_uint16_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint32_broadcast_block_wrapper(nvshmem_team_t team,
                                                       uint32_t *dest,
                                                       const uint32_t *source,
                                                       size_t nelems,
                                                       int PE_root) {
  return nvshmemx_uint32_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_uint64_broadcast_block_wrapper(nvshmem_team_t team,
                                                       uint64_t *dest,
                                                       const uint64_t *source,
                                                       size_t nelems,
                                                       int PE_root) {
  return nvshmemx_uint64_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_half_broadcast_block_wrapper(nvshmem_team_t team,
                                                     half *dest,
                                                     const half *source,
                                                     size_t nelems,
                                                     int PE_root) {
  return nvshmemx_half_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_bfloat16_broadcast_block_wrapper(
    nvshmem_team_t team, __nv_bfloat16 *dest, const __nv_bfloat16 *source,
    size_t nelems, int PE_root) {
  return nvshmemx_bfloat16_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_float_broadcast_block_wrapper(nvshmem_team_t team,
                                                      float *dest,
                                                      const float *source,
                                                      size_t nelems,
                                                      int PE_root) {
  return nvshmemx_float_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmemx_double_broadcast_block_wrapper(nvshmem_team_t team,
                                                       double *dest,
                                                       const double *source,
                                                       size_t nelems,
                                                       int PE_root) {
  return nvshmemx_double_broadcast_block(team, dest, source, nelems, PE_root);
}

__device__ int nvshmem_int8_fcollect_wrapper(nvshmem_team_t team, int8_t *dest,
                                             const int8_t *source,
                                             size_t nelems) {
  return nvshmem_int8_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_int16_fcollect_wrapper(nvshmem_team_t team,
                                              int16_t *dest,
                                              const int16_t *source,
                                              size_t nelems) {
  return nvshmem_int16_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_int32_fcollect_wrapper(nvshmem_team_t team,
                                              int32_t *dest,
                                              const int32_t *source,
                                              size_t nelems) {
  return nvshmem_int32_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_int64_fcollect_wrapper(nvshmem_team_t team,
                                              int64_t *dest,
                                              const int64_t *source,
                                              size_t nelems) {
  return nvshmem_int64_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_uint8_fcollect_wrapper(nvshmem_team_t team,
                                              uint8_t *dest,
                                              const uint8_t *source,
                                              size_t nelems) {
  return nvshmem_uint8_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_uint16_fcollect_wrapper(nvshmem_team_t team,
                                               uint16_t *dest,
                                               const uint16_t *source,
                                               size_t nelems) {
  return nvshmem_uint16_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_uint32_fcollect_wrapper(nvshmem_team_t team,
                                               uint32_t *dest,
                                               const uint32_t *source,
                                               size_t nelems) {
  return nvshmem_uint32_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_uint64_fcollect_wrapper(nvshmem_team_t team,
                                               uint64_t *dest,
                                               const uint64_t *source,
                                               size_t nelems) {
  return nvshmem_uint64_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_half_fcollect_wrapper(nvshmem_team_t team, half *dest,
                                             const half *source,
                                             size_t nelems) {
  return nvshmem_half_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_bfloat16_fcollect_wrapper(nvshmem_team_t team,
                                                 __nv_bfloat16 *dest,
                                                 const __nv_bfloat16 *source,
                                                 size_t nelems) {
  return nvshmem_bfloat16_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_float_fcollect_wrapper(nvshmem_team_t team, float *dest,
                                              const float *source,
                                              size_t nelems) {
  return nvshmem_float_fcollect(team, dest, source, nelems);
}

__device__ int nvshmem_double_fcollect_wrapper(nvshmem_team_t team,
                                               double *dest,
                                               const double *source,
                                               size_t nelems) {
  return nvshmem_double_fcollect(team, dest, source, nelems);
}

__device__ int nvshmemx_int8_fcollect_warp_wrapper(nvshmem_team_t team,
                                                   int8_t *dest,
                                                   const int8_t *source,
                                                   size_t nelems) {
  return nvshmemx_int8_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_int16_fcollect_warp_wrapper(nvshmem_team_t team,
                                                    int16_t *dest,
                                                    const int16_t *source,
                                                    size_t nelems) {
  return nvshmemx_int16_fcollect_warp(team, dest, source, nelems);
}
__device__ int nvshmemx_int32_fcollect_warp_wrapper(nvshmem_team_t team,
                                                    int32_t *dest,
                                                    const int32_t *source,
                                                    size_t nelems) {
  return nvshmemx_int32_fcollect_warp(team, dest, source, nelems);
}
__device__ int nvshmemx_int64_fcollect_warp_wrapper(nvshmem_team_t team,
                                                    int64_t *dest,
                                                    const int64_t *source,
                                                    size_t nelems) {
  return nvshmemx_int64_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_uint8_fcollect_warp_wrapper(nvshmem_team_t team,
                                                    uint8_t *dest,
                                                    const uint8_t *source,
                                                    size_t nelems) {
  return nvshmemx_uint8_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_uint16_fcollect_warp_wrapper(nvshmem_team_t team,
                                                     uint16_t *dest,
                                                     const uint16_t *source,
                                                     size_t nelems) {
  return nvshmemx_uint16_fcollect_warp(team, dest, source, nelems);
}
__device__ int nvshmemx_uint32_fcollect_warp_wrapper(nvshmem_team_t team,
                                                     uint32_t *dest,
                                                     const uint32_t *source,
                                                     size_t nelems) {
  return nvshmemx_uint32_fcollect_warp(team, dest, source, nelems);
}
__device__ int nvshmemx_uint64_fcollect_warp_wrapper(nvshmem_team_t team,
                                                     uint64_t *dest,
                                                     const uint64_t *source,
                                                     size_t nelems) {
  return nvshmemx_uint64_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_half_fcollect_warp_wrapper(nvshmem_team_t team,
                                                   half *dest,
                                                   const half *source,
                                                   size_t nelems) {
  return nvshmemx_half_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_bfloat16_fcollect_warp_wrapper(
    nvshmem_team_t team, __nv_bfloat16 *dest, const __nv_bfloat16 *source,
    size_t nelems) {
  return nvshmemx_bfloat16_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_float_fcollect_warp_wrapper(nvshmem_team_t team,
                                                    float *dest,
                                                    const float *source,
                                                    size_t nelems) {
  return nvshmemx_float_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_double_fcollect_warp_wrapper(nvshmem_team_t team,
                                                     double *dest,
                                                     const double *source,
                                                     size_t nelems) {
  return nvshmemx_double_fcollect_warp(team, dest, source, nelems);
}

__device__ int nvshmemx_int8_fcollect_block_wrapper(nvshmem_team_t team,
                                                    int8_t *dest,
                                                    const int8_t *source,
                                                    size_t nelems) {
  return nvshmemx_int8_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_int16_fcollect_block_wrapper(nvshmem_team_t team,
                                                     int16_t *dest,
                                                     const int16_t *source,
                                                     size_t nelems) {
  return nvshmemx_int16_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_int32_fcollect_block_wrapper(nvshmem_team_t team,
                                                     int32_t *dest,
                                                     const int32_t *source,
                                                     size_t nelems) {
  return nvshmemx_int32_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_int64_fcollect_block_wrapper(nvshmem_team_t team,
                                                     int64_t *dest,
                                                     const int64_t *source,
                                                     size_t nelems) {
  return nvshmemx_int64_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_uint8_fcollect_block_wrapper(nvshmem_team_t team,
                                                     uint8_t *dest,
                                                     const uint8_t *source,
                                                     size_t nelems) {
  return nvshmemx_uint8_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_uint16_fcollect_block_wrapper(nvshmem_team_t team,
                                                      uint16_t *dest,
                                                      const uint16_t *source,
                                                      size_t nelems) {
  return nvshmemx_uint16_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_uint32_fcollect_block_wrapper(nvshmem_team_t team,
                                                      uint32_t *dest,
                                                      const uint32_t *source,
                                                      size_t nelems) {
  return nvshmemx_uint32_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_uint64_fcollect_block_wrapper(nvshmem_team_t team,
                                                      uint64_t *dest,
                                                      const uint64_t *source,
                                                      size_t nelems) {
  return nvshmemx_uint64_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_half_fcollect_block_wrapper(nvshmem_team_t team,
                                                    half *dest,
                                                    const half *source,
                                                    size_t nelems) {
  return nvshmemx_half_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_bfloat16_fcollect_block_wrapper(
    nvshmem_team_t team, __nv_bfloat16 *dest, const __nv_bfloat16 *source,
    size_t nelems) {
  return nvshmemx_bfloat16_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_float_fcollect_block_wrapper(nvshmem_team_t team,
                                                     float *dest,
                                                     const float *source,
                                                     size_t nelems) {
  return nvshmemx_float_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmemx_double_fcollect_block_wrapper(nvshmem_team_t team,
                                                      double *dest,
                                                      const double *source,
                                                      size_t nelems) {
  return nvshmemx_double_fcollect_block(team, dest, source, nelems);
}

__device__ int nvshmem_team_my_pe_wrapper(nvshmem_team_t team) {
  return nvshmem_team_my_pe(team);
}

__device__ int nvshmem_team_n_pes_wrapper(nvshmem_team_t team) {
  return nvshmem_team_n_pes(team);
}

__device__ int nvshmemx_barrier_warp_wrapper(nvshmem_team_t team) {
  return nvshmemx_barrier_warp(team);
}

__device__ int nvshmemx_barrier_warpgroup_wrapper(nvshmem_team_t team) {
  return nvshmemx_barrier_warpgroup(team);
}

__device__ int nvshmemx_barrier_block_wrapper(nvshmem_team_t team) {
  return nvshmemx_barrier_block(team);
}

__device__ int nvshmemx_team_sync_warp_wrapper(nvshmem_team_t team) {
  return nvshmemx_team_sync_warp(team);
}

__device__ int nvshmemx_team_sync_block_wrapper(nvshmem_team_t team) {
  return nvshmemx_team_sync_block(team);
}

__device__ int nvshmem_barrier_wrapper(nvshmem_team_t team) {
  return nvshmem_barrier(team);
}
}
