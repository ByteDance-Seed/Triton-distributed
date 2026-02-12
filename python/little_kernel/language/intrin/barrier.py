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
Barrier and synchronization intrinsics.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import void, uint32

# ==============================================================================
# Basic synchronization
# ==============================================================================


def codegen_syncwarp():
    return Builtin(body="", includes=[], return_val="__syncwarp()")


@builtin(eval_return_type=void, codegen_func=codegen_syncwarp)
def __syncwarp():
    """Sync all threads in the warp."""
    raise RuntimeError("__syncwarp should never be called in compilation")


def codegen_block_sync():
    return Builtin(body="", includes=[], return_val="__syncthreads()")


@builtin(eval_return_type=void, codegen_func=codegen_block_sync)
def block_sync():
    """Block synchronization (__syncthreads)."""
    raise RuntimeError("should not call block_sync in compilation")


@builtin(eval_return_type=void, codegen_func=codegen_block_sync)
def __syncthreads():
    """CUDA __syncthreads() barrier synchronization."""
    raise RuntimeError("should not call __syncthreads in compilation")


# ==============================================================================
# Cluster synchronization
# ==============================================================================


def codegen_cluster_arrive():
    body = """
__device__ __forceinline__ void cluster_arrive_fn() {
    asm volatile("barrier.cluster.arrive;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="cluster_arrive_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_cluster_arrive)
def cluster_arrive():
    """Arrive at cluster barrier."""
    raise RuntimeError("should not call cluster_arrive in compilation")


def codegen_cluster_wait():
    body = """
__device__ __forceinline__ void cluster_wait_fn() {
    asm volatile("barrier.cluster.wait;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="cluster_wait_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_cluster_wait)
def cluster_wait():
    """Wait on cluster barrier."""
    raise RuntimeError("should not call cluster_wait in compilation")


def codegen_cluster_sync():
    body = """
__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\\nbarrier.cluster.wait;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="cluster_sync_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_cluster_sync)
def cluster_sync():
    """Full cluster barrier sync."""
    raise RuntimeError("should not call cluster_sync in compilation")


def codegen_cluster_rank():
    body = """
__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
"""
    return Builtin(body=body, includes=[], return_val="cluster_rank_fn()")


@builtin(eval_return_type=uint32, codegen_func=codegen_cluster_rank)
def cluster_rank():
    """Get cluster CTA rank."""
    raise RuntimeError("should not call cluster_rank in compilation")


# ==============================================================================
# MBarrier operations (standalone PTX, uses __cvta_generic_to_shared)
# ==============================================================================


def codegen_init_smem_barrier(smem_bar_ptr, arrive_cnt):
    body = """
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
"""
    return Builtin(body=body, includes=[], return_val=f"init_smem_barrier_fn({smem_bar_ptr}, {arrive_cnt})")


@builtin(eval_return_type=void, codegen_func=codegen_init_smem_barrier)
def init_smem_barrier(smem_bar_ptr, arrive_cnt):
    """Initialize mbarrier at shared memory pointer."""
    raise RuntimeError("should not call init_smem_barrier in compilation")


def codegen_fence_smem_barrier_init():
    body = """
__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="fence_smem_barrier_init_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_fence_smem_barrier_init)
def fence_smem_barrier_init():
    """Fence for mbarrier initialization."""
    raise RuntimeError("should not call fence_smem_barrier_init in compilation")


def codegen_mbarrier_arrive(smem):
    body = """
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_fn({smem})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive)
def mbarrier_arrive(smem):
    """Arrive on a mbarrier."""
    raise RuntimeError("should not call mbarrier_arrive in compilation")


def codegen_mbarrier_arrive_and_expect_tx(smem, transaction_bytes):
    body = """
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_and_expect_tx_fn({smem}, {transaction_bytes})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive_and_expect_tx)
def mbarrier_arrive_and_expect_tx(smem, transaction_bytes):
    """Arrive and expect transaction bytes on a mbarrier."""
    raise RuntimeError("should not call mbarrier_arrive_and_expect_tx in compilation")


def codegen_mbarrier_wait(smem, phase):
    body = """
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    asm volatile(
        "{\\n"
        ".reg .pred P;\\n"
        "WAIT_%=:\\n"
        "mbarrier.try_wait.parity.shared.b64 P, [%0], %1;\\n"
        "@!P bra WAIT_%=;\\n"
        "}\\n"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(phase));
}
"""
    return Builtin(body=body, includes=[], return_val=f"mbarrier_wait_fn({smem}, {phase})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_wait)
def mbarrier_wait(smem, phase):
    """Wait on a mbarrier parity."""
    raise RuntimeError("should not call mbarrier_wait in compilation")


def codegen_mbarrier_arrive_remote(bar_ptr, target_cta):
    body = """
__device__ __forceinline__ void mbarrier_arrive_remote_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t smem_addr = __cvta_generic_to_shared(&bar[0]);
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote_addr) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_remote_fn({bar_ptr}, {target_cta})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive_remote)
def mbarrier_arrive_remote(bar_ptr, target_cta):
    """Arrive on mbarrier at a remote CTA in cluster."""
    raise RuntimeError("should not call mbarrier_arrive_remote in compilation")


# ==============================================================================
# Memory fences
# ==============================================================================


def codegen_threadfence():
    return Builtin(body="", includes=[], return_val="__threadfence()")


@builtin(eval_return_type=void, codegen_func=codegen_threadfence)
def threadfence():
    """Device-scope memory fence (__threadfence)."""
    raise RuntimeError("should not call threadfence in compilation")


def codegen_threadfence_system():
    return Builtin(body="", includes=[], return_val="__threadfence_system()")


@builtin(eval_return_type=void, codegen_func=codegen_threadfence_system)
def threadfence_system():
    """System-scope memory fence (__threadfence_system)."""
    raise RuntimeError("should not call threadfence_system in compilation")


def codegen_fence_proxy_async():
    body = """
__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="fence_proxy_async_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_fence_proxy_async)
def fence_proxy_async():
    """Generic async proxy fence (fence.proxy.async, without shared::cta qualifier).
    Required after manual shared memory writes before WGMMA consumption."""
    raise RuntimeError("should not call fence_proxy_async in compilation")


def codegen_fence_async_shared():
    body = """
__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="fence_async_shared_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_fence_async_shared)
def fence_async_shared():
    """Async proxy fence on shared memory (fence.proxy.async.shared::cta)."""
    raise RuntimeError("should not call fence_async_shared in compilation")


# ==============================================================================
# Named barrier
# ==============================================================================


def codegen_named_barrier_sync(bar_id, count):
    body = """
__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count));
}
"""
    return Builtin(body=body, includes=[], return_val=f"named_barrier_sync_fn({bar_id}, {count})")


@builtin(eval_return_type=void, codegen_func=codegen_named_barrier_sync)
def named_barrier_sync(bar_id, count):
    """Named barrier synchronization."""
    raise RuntimeError("should not call named_barrier_sync in compilation")


def codegen_named_barrier_arrive(bar_id, count):
    body = """
__device__ __forceinline__ void named_barrier_arrive_fn(int bar_id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar_id), "r"(count));
}
"""
    return Builtin(body=body, includes=[], return_val=f"named_barrier_arrive_fn({bar_id}, {count})")


@builtin(eval_return_type=void, codegen_func=codegen_named_barrier_arrive)
def named_barrier_arrive(bar_id, count):
    """Named barrier arrive (without wait)."""
    raise RuntimeError("should not call named_barrier_arrive in compilation")


# ==============================================================================
# Grid-level synchronization (atomic-based, no cooperative_groups needed)
# ==============================================================================


def codegen_grid_sync():
    body = """
// Grid-level barrier using atomic operations (no cooperative_groups needed)
__device__ unsigned int __grid_sync_count = 0;
__device__ volatile int __grid_sync_sense = 0;

__device__ __forceinline__ void grid_sync_fn() {
    __syncthreads();
    __threadfence();
    // Only (0,0,0) thread of each block participates in the grid barrier
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Atomically increment the arrival counter
        unsigned int arrived = atomicAdd(&__grid_sync_count, 1);
        if (arrived == num_blocks - 1) {
            // Last block: reset counter and flip sense
            __grid_sync_count = 0;
            __threadfence();
            __grid_sync_sense ^= 1;
        } else {
            // Wait for the sense to flip
            int expected = __grid_sync_sense ^ 1;
            while (__grid_sync_sense != expected) {
                // Spin-wait
            }
        }
    }
    __syncthreads();
}
"""
    return Builtin(body=body, includes=[], return_val="grid_sync_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_grid_sync)
def grid_sync():
    """Grid-level synchronization via atomic barrier (no cooperative_groups)."""
    raise RuntimeError("should not call grid_sync in compilation")
