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
// clang-format off
/*
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See COPYRIGHT for license information
 */

#ifndef TRANSFER_DEVICE_CUH
#define TRANSFER_DEVICE_CUH
#include <assert.h>
#include <cuda_runtime.h>
#include "non_abi/nvshmem_build_options.h"
#include "non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh"
#include "device_host/nvshmem_common.cuh"
#include "device/nvshmem_device_macros.h"
#include "non_abi/device/pt-to-pt/proxy_device.cuh"

#ifdef NVSHMEM_IBGDA_SUPPORT
#include "non_abi/device/pt-to-pt/ibgda_device.cuh"
#endif

#if defined __clang_llvm_bitcode_lib__
#define NVSHMEMI_TRANSFER_INLINE \
    __attribute__((noinline, section(".text.compute"), not_tail_called))
#define NVSHMEMI_TRANSFER_STATIC
#elif defined NVSHMEM_ENABLE_ALL_DEVICE_INLINING
#define NVSHMEMI_TRANSFER_INLINE inline
#define NVSHMEMI_TRANSFER_STATIC static
#else
#define NVSHMEMI_TRANSFER_INLINE __noinline__
#define NVSHMEMI_TRANSFER_STATIC
#endif

#ifdef __CUDA_ARCH__

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_TRANSFER_STATIC __device__ NVSHMEMI_TRANSFER_INLINE void nvshmemi_transfer_rma(
    void *rptr, void *lptr, size_t bytes, int pe) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_rma<SCOPE, channel_op>(rptr, lptr, bytes, pe);
    } else
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, channel_op);
            nvshmemi_proxy_quiet(false);
            if (SCOPE == nvshmemi_threadgroup_thread)
                __threadfence_block(); /* to prevent reuse of src buffer before quiet completion;
                                    for warp/block scope, following sync op will accomplish that */
        }
    }
}

template <threadgroup_t SCOPE, nvshmemi_op_t channel_op>
NVSHMEMI_TRANSFER_STATIC __device__ NVSHMEMI_TRANSFER_INLINE void nvshmemi_transfer_rma_nbi(
    void *rptr, void *lptr, size_t bytes, int pe) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_rma_nbi<SCOPE, channel_op>(rptr, lptr, bytes, pe);
    } else
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, channel_op);
    }
}

template <threadgroup_t SCOPE>
NVSHMEMI_TRANSFER_STATIC __device__ NVSHMEMI_TRANSFER_INLINE void nvshmemi_transfer_put_signal(
    void *rptr, void *lptr, size_t bytes, void *sig_addr, uint64_t signal, nvshmemi_amo_t sig_op,
    int pe, bool is_nbi) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        nvshmemi_ibgda_put_signal<SCOPE>(rptr, lptr, bytes, sig_addr, signal, sig_op, pe, is_nbi);
    } else
#endif
    {
        int myIdx = nvshmemi_thread_id_in_threadgroup<SCOPE>();
        if (!myIdx) {
            nvshmemi_proxy_rma_nbi(rptr, lptr, bytes, pe, NVSHMEMI_OP_PUT);
            nvshmemi_proxy_fence();
            nvshmemi_proxy_amo_nonfetch<uint64_t>(sig_addr, signal, pe, sig_op);
            if (is_nbi == 0) {
                nvshmemi_proxy_quiet(false);
                if (SCOPE == nvshmemi_threadgroup_thread)
                    __threadfence_block(); /* to prevent reuse of src buffer before quiet completion
                                        for warp/block scope, following sync op will accomplish that
                                      */
            }
        }
    }
}

template <typename T>
NVSHMEMI_TRANSFER_STATIC __device__ NVSHMEMI_TRANSFER_INLINE T
nvshmemi_transfer_amo_fetch(void *rptr, T value, T compare, int pe, nvshmemi_amo_t op) {
#ifdef NVSHMEM_IBGDA_SUPPORT
    if (nvshmemi_device_state_d.ibgda_is_initialized) {
        return nvshmemi_ibgda_amo_fetch<T>(rptr, value, compare, pe, op);
    } else
#endif
    {
        T retval;
        nvshmemi_proxy_amo_fetch<T>(rptr, (void *)&retval, value, compare, pe, op);
        return retval;
    }
}

// clang-format on
#if defined __cplusplus || defined __clang_llvm_bitcode_lib__
extern "C" {
#endif
#define TRANSFER_DECL_RMA_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)              \
  NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                    \
      nvshmemi_transfer_rma_put##SC_SUFFIX(void *rptr, void *lptr,             \
                                           size_t bytes, int pe) {             \
    nvshmemi_transfer_rma<nvshmemi_threadgroup_##SCOPE, NVSHMEMI_OP_PUT>(      \
        rptr, lptr, bytes, pe);                                                \
  }                                                                            \
  NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                    \
      nvshmemi_transfer_rma_get##SC_SUFFIX(void *rptr, void *lptr,             \
                                           size_t bytes, int pe) {             \
    nvshmemi_transfer_rma<nvshmemi_threadgroup_##SCOPE, NVSHMEMI_OP_GET>(      \
        rptr, lptr, bytes, pe);                                                \
  }

TRANSFER_DECL_RMA_SCOPE_IMPL(thread, , x)
TRANSFER_DECL_RMA_SCOPE_IMPL(warp, _warp, x)
TRANSFER_DECL_RMA_SCOPE_IMPL(block, _block, x)
#undef TRANSFER_DECL_RMA_SCOPE_IMPL

#define TRANSFER_DECL_RMA_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)              \
  NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                    \
      nvshmemi_transfer_rma_put_nbi##SC_SUFFIX(void *rptr, void *lptr,         \
                                               size_t bytes, int pe) {         \
    nvshmemi_transfer_rma_nbi<nvshmemi_threadgroup_##SCOPE, NVSHMEMI_OP_PUT>(  \
        rptr, lptr, bytes, pe);                                                \
  }                                                                            \
  NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                    \
      nvshmemi_transfer_rma_nbi_get##SC_SUFFIX(void *rptr, void *lptr,         \
                                               size_t bytes, int pe) {         \
    nvshmemi_transfer_rma_nbi<nvshmemi_threadgroup_##SCOPE, NVSHMEMI_OP_GET>(  \
        rptr, lptr, bytes, pe);                                                \
  }

TRANSFER_DECL_RMA_SCOPE_IMPL(thread, , x)
TRANSFER_DECL_RMA_SCOPE_IMPL(warp, _warp, x)
TRANSFER_DECL_RMA_SCOPE_IMPL(block, _block, x)
#undef TRANSFER_DECL_RMA_SCOPE_IMPL

#define TRANSFER_DECL_RMA_SCOPE_IMPL(SCOPE, SC_SUFFIX, SC_PREFIX)              \
  NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                    \
      nvshmemi_transfer_put_signal##SC_SUFFIX(                                 \
          void *rptr, void *lptr, size_t bytes, void *sig_addr,                \
          uint64_t signal, nvshmemi_amo_t sig_op, int pe, bool is_nbi) {       \
    nvshmemi_transfer_put_signal<nvshmemi_threadgroup_##SCOPE>(                \
        rptr, lptr, bytes, sig_addr, signal, sig_op, pe, false);               \
  }                                                                            \
  NVSHMEMI_DEVICE_PREFIX NVSHMEMI_DEVICE_ALWAYS_INLINE void                    \
      nvshmemi_transfer_put_signal_nbi##SC_SUFFIX(                             \
          void *rptr, void *lptr, size_t bytes, void *sig_addr,                \
          uint64_t signal, nvshmemi_amo_t sig_op, int pe, bool is_nbi) {       \
    nvshmemi_transfer_put_signal<nvshmemi_threadgroup_##SCOPE>(                \
        rptr, lptr, bytes, sig_addr, signal, sig_op, pe, true);                \
  }

TRANSFER_DECL_RMA_SCOPE_IMPL(thread, , x)
TRANSFER_DECL_RMA_SCOPE_IMPL(warp, _warp, x)
TRANSFER_DECL_RMA_SCOPE_IMPL(block, _block, x)
#undef TRANSFER_DECL_RMA_SCOPE_IMPL

// clang-format off
#if defined __cplusplus || defined __clang_llvm_bitcode_lib__
}
#endif


#endif /* __CUDA_ARCH__ */

#endif /* TRANSFER_DEVICE_CUH */
