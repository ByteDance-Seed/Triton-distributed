/*
 * MIT License
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
//
// Created by Maksim Levental (Advanced Micro Devices, Inc) on 3/20/25.
// Modified by Wenlei Bao, Xuegui zheng and Chenhui Huang (ByteDance Ltd).
//

#include <cstdint>

// Instructions for updating:
// 1. Change impl here
// 2. Compile using clang:
// /opt/rocm-6.3.0/lib/llvm/bin/clang-18  -o comms.ll \
//        --no-gpu-bundle-output -S -x hip --cuda-device-only -O3 \
//        -D__HIP_PLATFORM_AMD__=1 -D__HIP_PLATFORM_HCC__=1 -fcolor-diagnostics
//        \
//        -fno-crash-diagnostics -S -emit-llvm --offload-arch=gfx942
//        -mno-tgsplit \
//        Triton-distributed/third_party/amd/language/hip/comms.cpp
// 3. Copy resulting llvm ir to comms.ll

// 4. Translate llvm ir to
// third_party\amd\lib\TritonAMDGPUToLLVM\BuiltinFuncToLLVM.cpp:124
//    - common patterns collected in buildAtomicLoad, buildAtomicStore,
//    buildAtomicFetchAdd

// =============================================================================
// ========================================   #1 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
load_acquire_workgroup(uint64_t [[clang::opencl_global]] * input) {
  return __hip_atomic_load(input, __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_WORKGROUP);
}

// =============================================================================
// ========================================   #2 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
load_relaxed_workgroup(uint64_t [[clang::opencl_global]] * input) {
  return __hip_atomic_load(input, __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_WORKGROUP);
}

// =============================================================================
// ========================================   #3 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
load_acquire_agent(uint64_t [[clang::opencl_global]] * input) {
  return __hip_atomic_load(input, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
// ========================================   #4 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
load_relaxed_agent(uint64_t [[clang::opencl_global]] * input) {
  return __hip_atomic_load(input, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
// ========================================   #5 ===============================
// =============================================================================

__attribute__((used, device)) int
load_acquire_system(int [[clang::opencl_global]] * input) {
  return __hip_atomic_load(input, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
// ========================================   #6 ===============================
// =============================================================================

__attribute__((used, device)) int
load_relaxed_system(int [[clang::opencl_global]] * input) {
  return __hip_atomic_load(input, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
// ========================================   #7 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
store_release_workgroup(uint64_t [[clang::opencl_global]] * input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELEASE,
                     __HIP_MEMORY_SCOPE_WORKGROUP);
  return *input;
}

// =============================================================================
// ========================================   #8 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
store_relaxed_workgroup(uint64_t [[clang::opencl_global]] * input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELAXED,
                     __HIP_MEMORY_SCOPE_WORKGROUP);
  return *input;
}

// =============================================================================
// ========================================   #9 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
store_release_agent(uint64_t [[clang::opencl_global]] * input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
  return *input;
}

// =============================================================================
/// =====================================   #10 ===============================
// =============================================================================

__attribute__((used, device)) uint64_t
store_relaxed_agent(uint64_t [[clang::opencl_global]] * input) {
  uint64_t value{1};
  __hip_atomic_store(input, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  return *input;
}

// =============================================================================
/// =====================================   #11 ===============================
// =============================================================================

__attribute__((used, device)) int
store_release_system(int [[clang::opencl_global]] * input, int value) {
  __hip_atomic_store(input, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  return *input;
}

// =============================================================================
/// =====================================   #12 ===============================
// =============================================================================

__attribute__((used, device)) int
store_relaxed_system(int [[clang::opencl_global]] * input, int value) {
  __hip_atomic_store(input, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  return *input;
}

// =============================================================================
/// =====================================   #15 ===============================
// =============================================================================

__attribute__((used, device)) uint32_t
red_add_release_agent(int [[clang::opencl_global]] * atomic_address,
                      int [[clang::opencl_global]] * value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #16 ===============================
// =============================================================================

__attribute__((used, device)) int
red_add_release_system(int [[clang::opencl_global]] * atomic_address,
                       int value) {
  return __hip_atomic_fetch_add(atomic_address, value, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #17 ===============================
// =============================================================================

__attribute__((used, device)) uint32_t
atom_add_acquire_agent(int [[clang::opencl_global]] * atomic_address,
                       int [[clang::opencl_global]] * value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #19 ===============================
// =============================================================================

__attribute__((used, device)) uint32_t
atom_add_relaxed_agent(int [[clang::opencl_global]] * atomic_address,
                       int [[clang::opencl_global]] * value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #20 ===============================
// =============================================================================

__attribute__((used, device)) uint32_t
atom_add_acqrel_agent(int [[clang::opencl_global]] * atomic_address,
                      int [[clang::opencl_global]] * value) {
  return __hip_atomic_fetch_add(atomic_address, *value, __ATOMIC_ACQ_REL,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #21 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_add_acquire_system(int [[clang::opencl_global]] * atomic_address,
                        int value) {
  return __hip_atomic_fetch_add(atomic_address, value, __ATOMIC_ACQUIRE,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #23 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_add_relaxed_system(int [[clang::opencl_global]] * atomic_address,
                        int value) {
  return __hip_atomic_fetch_add(atomic_address, value, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #24 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_add_acqrel_system(int [[clang::opencl_global]] * atomic_address,
                       int value) {
  return __hip_atomic_fetch_add(atomic_address, value, __ATOMIC_ACQ_REL,
                                __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #25 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_acquire_relaxed_agent(int [[clang::opencl_global]] * atomic_address,
                               int [[clang::opencl_global]] * compare,
                               int [[clang::opencl_global]] * value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #26 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_release_relaxed_agent(int [[clang::opencl_global]] * atomic_address,
                               int [[clang::opencl_global]] * compare,
                               int [[clang::opencl_global]] * value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #27 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_relaxed_relaxed_agent(int [[clang::opencl_global]] * atomic_address,
                               int [[clang::opencl_global]] * compare,
                               int [[clang::opencl_global]] * value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #28 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_acqrel_relaxed_agent(int [[clang::opencl_global]] * atomic_address,
                              int [[clang::opencl_global]] * compare,
                              int [[clang::opencl_global]] * value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, *value, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_AGENT);
}

// =============================================================================
/// =====================================   #29 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_acquire_relaxed_system(int [[clang::opencl_global]] * atomic_address,
                                int [[clang::opencl_global]] * compare,
                                int value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, value, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #30 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_release_relaxed_system(int [[clang::opencl_global]] * atomic_address,
                                int [[clang::opencl_global]] * compare,
                                int value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, value, __ATOMIC_RELEASE, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #31 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_relaxed_relaxed_system(int [[clang::opencl_global]] * atomic_address,
                                int [[clang::opencl_global]] * compare,
                                int value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, value, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #32 ===============================
// =============================================================================

__attribute__((used, device)) int
atom_cas_acqrel_relaxed_system(int [[clang::opencl_global]] * atomic_address,
                               int [[clang::opencl_global]] * compare,
                               int value) {
  return __hip_atomic_compare_exchange_strong(
      atomic_address, compare, value, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED,
      __HIP_MEMORY_SCOPE_SYSTEM);
}

// =============================================================================
/// =====================================   #14 ===============================
// =============================================================================

// __attribute__((used, device)) int syncthreads() {
//   syncthreads();
//   return 0;
// }
