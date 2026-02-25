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
#include "flash_comm/utils.cuh"

__forceinline__ __device__ void tma_copy_1d_g2s(void const *gmem_ptr,
                                                uint64_t *mbar_ptr,
                                                void *smem_ptr,
                                                int32_t load_bytes) {
#if defined(ARCH_SM90_ENABLED)
  uint32_t smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr);
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];\n"
               :
               : "r"(smem_int_ptr), "l"(gmem_ptr), "r"(load_bytes),
                 "r"(smem_int_mbar)
               : "memory");
#endif
}

__forceinline__ __device__ void
tma_copy_1d_s2g(void const *smem_ptr, void *gmem_ptr, int32_t store_bytes) {
#if defined(ARCH_SM90_ENABLED)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
               : "memory");
#endif
};

// Wait until at most Count committed TMA_STOREs are pending and all prior
// commits are complete
template <int Count> __forceinline__ __device__ void tma_store_wait() {
#if defined(ARCH_SM90_ENABLED)
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
#endif
}

__forceinline__ __device__ void tma_store_arrive() {
#if defined(ARCH_SM90_ENABLED)
  asm volatile("cp.async.bulk.commit_group;");
#endif
}

template <uint32_t Stages_> struct PipelineState {
  static constexpr uint32_t Stages = Stages_;

  int index_ = 0;
  uint32_t phase_ = 0;
  uint32_t count_ = 0;

  __forceinline__ __device__ PipelineState() : index_{}, phase_{}, count_{} {}

  __forceinline__ __device__ PipelineState(int index, uint32_t phase,
                                           uint32_t count)
      : index_(index), phase_(phase), count_(count) {}

  __forceinline__ __device__ int index() const { return index_; }

  __forceinline__ __device__ uint32_t phase() const { return phase_; }

  __forceinline__ __device__ uint32_t count() const { return count_; }

  __forceinline__ __device__ void operator++() {
    if constexpr (Stages > 0) {
      ++index_;
      ++count_;
      if (index_ == Stages) {
        index_ = 0;
        phase_ ^= 1;
      }
    }
  }

  __forceinline__ __device__ PipelineState &
  operator+=(uint32_t num_iterations) {
    return advance(num_iterations);
  }

  __forceinline__ __device__ PipelineState &
  operator=(PipelineState const &other) {
    index_ = other.index();
    phase_ = other.phase();
    count_ = other.count();
    return *this;
  }

  __forceinline__ __device__ PipelineState &advance(uint32_t num_iterations) {
    if constexpr (Stages > 0) {
      // Number of iterations cross over the stage boundary => flipped phase
      if ((num_iterations < Stages) && (index_ + num_iterations) >= Stages) {
        phase_ ^= 1;
      }
      // How many times number of iterations cross over the stage boundary and
      // end up on a odd number => flipped phase
      if ((num_iterations >= Stages) &&
          (((index_ + num_iterations) / Stages) % 2) == 1) {
        phase_ ^= 1;
      }
      index_ = (index_ + num_iterations) % Stages;
      count_ += num_iterations;
    }
    return *this;
  }

  __forceinline__ __device__ static PipelineState
  make_pipeline_state(PipelineState start_state, uint32_t num_iterations) {
    return start_state.advance(num_iterations);
  }
};
