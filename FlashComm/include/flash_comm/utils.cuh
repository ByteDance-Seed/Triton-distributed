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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#error "FlashComm CUDA kernels require SM90 or newer."
#endif

__forceinline__ __device__ uint32_t
cast_smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Initialize barrier present in shared memory
__forceinline__ __device__ void initialize_barrier(
    uint64_t *smem_barrier_ptr, // 64 bits user-managed barrier in smem
    int thread_count =
        1) // Thread count expected to arrive/wait on this barrier
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_barrier_ptr);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
               "r"(thread_count));
}

// Set the number of bytes transferred per transaction and perform an arrive
// operation as well
__forceinline__ __device__ void mbar_arrive_and_set_barrier_transaction_bytes(
    uint64_t *smem_barrier_ptr, // 64 bits user-managed barrier in smem
    uint32_t bytes) // Number of bytes transferred by per TMA transaction
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_barrier_ptr);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
                   smem_int_ptr),
               "r"(bytes));
}

// Barrier wait
__forceinline__ __device__ void
wait_barrier(uint64_t *smem_barrier_ptr, // 64 bits user-managed barrier in smem
             int phase_bit) // Current phase bit the barrier waiting to flip
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_barrier_ptr);
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra DONE;\n"
               "bra                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(smem_int_ptr),
               "r"(phase_bit));
}

// Barrier arrive
__forceinline__ __device__ void arrive_barrier(
    uint64_t *smem_barrier_ptr) // 64 bits user-managed barrier in smem
{
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_barrier_ptr);
  asm volatile("{\n"
               ".reg .b64 state; \n"
               "mbarrier.arrive.shared::cta.b64   state, [%0];\n"
               "}\n" ::"r"(smem_int_ptr));
}

__forceinline__ __device__ void
named_barrier_arrive_and_wait(uint32_t num_threads, uint32_t barrier_id) {
  asm volatile("bar.sync %0, %1;"
               :
               : "r"(barrier_id), "r"(num_threads)
               : "memory");
}

__forceinline__ __device__ void fence_async_shared() {
  asm volatile("fence.proxy.async.shared::cta;");
}

__forceinline__ __device__ uint32_t elect_one_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile("{\n"
               ".reg .b32 %%rx;\n"
               ".reg .pred %%px;\n"
               "     elect.sync %%rx|%%px, %2;\n"
               "@%%px mov.s32 %1, 1;\n"
               "     mov.s32 %0, %%rx;\n"
               "}\n"
               : "+r"(laneid), "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
#elif defined(__CUDA_ARCH__)
  return (threadIdx.x % 32) == 0;
#else
  return true;
#endif
}

constexpr int WARP_SIZE = 32;

#define PRAGMA_UNROLL _Pragma("unroll")

namespace flash_comm {
namespace detail {

template <int kUnrollFactor, typename VecT> struct UnrolledLoad {
  __device__ __forceinline__ static void apply(const VecT *src, VecT *regs,
                                               int base_idx) {
    PRAGMA_UNROLL
    for (int i = 0; i < kUnrollFactor; ++i) {
      regs[i] = __ldg(src + base_idx + i * WARP_SIZE);
    }
  }
};

template <int kUnrollFactor, typename VecT> struct UnrolledStore {
  __device__ __forceinline__ static void apply(VecT *dst, const VecT *regs,
                                               int base_idx) {
    PRAGMA_UNROLL
    for (int i = 0; i < kUnrollFactor; ++i) {
      dst[base_idx + i * WARP_SIZE] = regs[i];
    }
  }
};

} // namespace detail
} // namespace flash_comm

template <int kUnrollFactor = 4, typename VecT = int4>
__device__ __forceinline__ void warp_copy_unrolled(const VecT *__restrict__ src,
                                                   VecT *__restrict__ dst,
                                                   int num_vecs, int lane_id) {
  static_assert(kUnrollFactor >= 1 && kUnrollFactor <= 8,
                "kUnrollFactor must be between 1 and 8");

  VecT regs[kUnrollFactor];
  int idx = lane_id;

  const int unroll_stride = WARP_SIZE * kUnrollFactor;
  for (; idx + (kUnrollFactor - 1) * WARP_SIZE < num_vecs;
       idx += unroll_stride) {
    flash_comm::detail::UnrolledLoad<kUnrollFactor, VecT>::apply(src, regs,
                                                                 idx);
    flash_comm::detail::UnrolledStore<kUnrollFactor, VecT>::apply(dst, regs,
                                                                  idx);
  }

  // remainder loop
  for (; idx < num_vecs; idx += WARP_SIZE) {
    dst[idx] = __ldg(src + idx);
  }
}

template <int kUnrollFactor = 4, typename TokenT>
__device__ __forceinline__ void warp_copy_token(const TokenT *__restrict__ src,
                                                TokenT *__restrict__ dst,
                                                int hidden_size, int lane_id) {
  constexpr int kElemsPerInt4 = sizeof(int4) / sizeof(TokenT);
  const int num_int4s = hidden_size / kElemsPerInt4;

  warp_copy_unrolled<kUnrollFactor, int4>(reinterpret_cast<const int4 *>(src),
                                          reinterpret_cast<int4 *>(dst),
                                          num_int4s, lane_id);
}
