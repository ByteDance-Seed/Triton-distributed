################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
TMA (Tensor Memory Accelerator) intrinsics for SM90+.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import void

# ==============================================================================
# TMA Load
# ==============================================================================


def codegen_tma_load_2d(desc, bar_ptr, smem_ptr, c0, c1):
    body = """
__device__ __forceinline__ void tma_load_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "r"(c0), "r"(c1) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_load_2d_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_2d)
def tma_load_2d(desc, bar_ptr, smem_ptr, c0, c1):
    """TMA 2D load from global to shared memory."""
    raise RuntimeError("should not call tma_load_2d in compilation")


def codegen_tma_load_multicast_2d(desc, bar_ptr, smem_ptr, c0, c1, multicast_mask):
    body = """
__device__ __forceinline__ void tma_load_multicast_2d_fn(const CUtensorMap* d, uint64_t* bar, void* smem, int32_t c0, int32_t c1, uint16_t mask) {
    uint64_t cache_hint = 0;
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint [%0], [%1, {%4, %5}], [%2], %3, %6;"
        :: "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(&bar[0])),
           "h"(mask), "r"(c0), "r"(c1), "l"(cache_hint) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"],
                   return_val=f"tma_load_multicast_2d_fn(&{desc}, {bar_ptr}, {smem_ptr}, {c0}, {c1}, {multicast_mask})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_load_multicast_2d)
def tma_load_multicast_2d(desc, bar_ptr, smem_ptr, c0, c1, multicast_mask):
    """TMA 2D multicast load from global to shared memory."""
    raise RuntimeError("should not call tma_load_multicast_2d in compilation")


# ==============================================================================
# TMA Store
# ==============================================================================


def codegen_tma_store_2d(desc, smem_ptr, c0, c1):
    body = """
__device__ __forceinline__ void tma_store_2d_fn(const CUtensorMap* d, void* smem, int32_t c0, int32_t c1) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%2, %3}], [%1];"
        :: "l"((uint64_t)d),
           "r"((uint32_t)__cvta_generic_to_shared(smem)),
           "r"(c0), "r"(c1) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"], return_val=f"tma_store_2d_fn(&{desc}, {smem_ptr}, {c0}, {c1})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_2d)
def tma_store_2d(desc, smem_ptr, c0, c1):
    """TMA 2D store from shared to global memory."""
    raise RuntimeError("should not call tma_store_2d in compilation")


def codegen_tma_store_fence():
    body = """
__device__ __forceinline__ void tma_store_fence_fn() {
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tma_store_fence_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_fence)
def tma_store_fence():
    """TMA store fence."""
    raise RuntimeError("should not call tma_store_fence in compilation")


def codegen_tma_store_arrive():
    body = """
__device__ __forceinline__ void tma_store_arrive_fn() {
    asm volatile("cp.async.bulk.commit_group;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tma_store_arrive_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_arrive)
def tma_store_arrive():
    """TMA store arrive/commit."""
    raise RuntimeError("should not call tma_store_arrive in compilation")


def codegen_tma_store_wait():
    body = """
__device__ __forceinline__ void tma_store_wait_fn() {
    asm volatile("cp.async.bulk.wait_group 0;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="tma_store_wait_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_wait)
def tma_store_wait():
    """TMA store wait."""
    raise RuntimeError("should not call tma_store_wait in compilation")


# ==============================================================================
# TMA 1D Bulk Copy (used by FlashComm EP kernels)
# ==============================================================================


def codegen_tma_copy_1d_g2s(gmem_ptr, mbar_ptr, smem_ptr, load_bytes):
    body = """
__device__ __forceinline__ void tma_copy_1d_g2s_fn(void const* gmem, uint64_t* mbar, void* smem, int32_t bytes) {
    uint32_t smem_mbar = (uint32_t)__cvta_generic_to_shared(mbar);
    uint32_t smem_ptr  = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\\n"
        :: "r"(smem_ptr), "l"(gmem), "r"(bytes), "r"(smem_mbar) : "memory");
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"tma_copy_1d_g2s_fn({gmem_ptr}, {mbar_ptr}, {smem_ptr}, {load_bytes})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_copy_1d_g2s)
def tma_copy_1d_g2s(gmem_ptr, mbar_ptr, smem_ptr, load_bytes):
    """TMA 1D bulk copy from global to shared memory."""
    raise RuntimeError("should not call tma_copy_1d_g2s in compilation")


def codegen_tma_copy_1d_s2g(smem_ptr, gmem_ptr, store_bytes):
    body = """
__device__ __forceinline__ void tma_copy_1d_s2g_fn(void const* smem, void* gmem, int32_t bytes) {
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\\n"
        :: "l"(gmem), "r"(smem_int), "r"(bytes) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"tma_copy_1d_s2g_fn({smem_ptr}, {gmem_ptr}, {store_bytes})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_copy_1d_s2g)
def tma_copy_1d_s2g(smem_ptr, gmem_ptr, store_bytes):
    """TMA 1D bulk copy from shared to global memory."""
    raise RuntimeError("should not call tma_copy_1d_s2g in compilation")


def codegen_tma_store_wait_n(count):
    body = """
__device__ __forceinline__ void tma_store_wait_n_fn(int count) {
    // cp.async.bulk.wait_group requires immediate operand, use switch for common values
    switch (count) {
        case 0: asm volatile("cp.async.bulk.wait_group 0;\\n" ::: "memory"); break;
        case 1: asm volatile("cp.async.bulk.wait_group 1;\\n" ::: "memory"); break;
        case 2: asm volatile("cp.async.bulk.wait_group 2;\\n" ::: "memory"); break;
        case 3: asm volatile("cp.async.bulk.wait_group 3;\\n" ::: "memory"); break;
        case 4: asm volatile("cp.async.bulk.wait_group 4;\\n" ::: "memory"); break;
        case 5: asm volatile("cp.async.bulk.wait_group 5;\\n" ::: "memory"); break;
        case 6: asm volatile("cp.async.bulk.wait_group 6;\\n" ::: "memory"); break;
        case 7: asm volatile("cp.async.bulk.wait_group 7;\\n" ::: "memory"); break;
        default: asm volatile("cp.async.bulk.wait_group 0;\\n" ::: "memory"); break;
    }
}
"""
    return Builtin(body=body, includes=[], return_val=f"tma_store_wait_n_fn({count})")


@builtin(eval_return_type=void, codegen_func=codegen_tma_store_wait_n)
def tma_store_wait_n(count):
    """TMA store wait with pipeline depth count (wait until at most `count` stores pending)."""
    raise RuntimeError("should not call tma_store_wait_n in compilation")


# ==============================================================================
# TMA Descriptor
# ==============================================================================


def codegen_prefetch_tma_descriptor(tensor_map):
    body = """
__device__ __forceinline__ void prefetch_tma_descriptor_fn(const CUtensorMap* d) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(d) : "memory");
}
"""
    return Builtin(body=body, includes=["<cuda.h>"], return_val=f"prefetch_tma_descriptor_fn(&{tensor_map})")


@builtin(eval_return_type=void, codegen_func=codegen_prefetch_tma_descriptor)
def prefetch_tma_descriptor(tensor_map):
    """Prefetch TMA descriptor."""
    raise RuntimeError("should not call prefetch_tma_descriptor in compilation")
