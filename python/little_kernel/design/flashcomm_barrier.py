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
FlashComm kernel: kernel_barrier_all_on_stream
Cross-rank barrier using system-scope atomics over symmetric memory.

Original source: Triton-distributed/FlashComm/csrc/ep/kernels/intranode_cuda.cu
"""
import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.codegen.codegen_cuda import codegen_cuda

# ============================================================================
# Kernel: barrier_all_block (device function)
# ============================================================================
backend = "cuda"


@ll_kernel(backend=backend, is_entry=False)
def barrier_all_block(
    barrier_ptrs: ll.ptr[ll.ptr[ll.int32]],
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    thread_id: ll.int32 = ll.threadIdx_x()
    if thread_id < num_ranks:
        ll.threadfence_system()
        # Write 1 to remote rank's barrier slot for this rank
        remote_ptr: ll.ptr[ll.int32] = barrier_ptrs[thread_id] + rank
        # Spin until we successfully set remote_ptr from 0 to 1
        old: ll.int32 = ll.atomic_cas_system(remote_ptr, 0, 1)
        while old != 0:
            old = ll.atomic_cas_system(remote_ptr, 0, 1)
        ll.threadfence_system()
        # Wait for our own slot to be set to 1 by the other rank
        wait_ptr: ll.ptr[ll.int32] = barrier_ptrs[rank] + thread_id
        old2: ll.int32 = ll.atomic_cas_system(wait_ptr, 1, 0)
        while old2 != 1:
            old2 = ll.atomic_cas_system(wait_ptr, 1, 0)
        ll.threadfence_system()
    ll.__syncthreads()


# ============================================================================
# Kernel: kernel_barrier_all_on_stream (entry point)
# ============================================================================
@ll_kernel(backend=backend, is_entry=True)
def kernel_barrier_all_on_stream(
    barrier_ptrs: ll.ptr[ll.ptr[ll.int32]],
    rank: ll.int32,
    num_ranks: ll.int32,
) -> ll.void:
    barrier_all_block(barrier_ptrs, rank, num_ranks)


# ============================================================================
# Build and test
# ============================================================================
NUM_THREADS = 128


def build_kernel():
    """Build the barrier kernel."""
    kernel = kernel_barrier_all_on_stream

    compiled = kernel.build(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        grid=(1, 1, 1),
        block=(NUM_THREADS, 1, 1),
        shared_mem_bytes=0,
        verbose=True,
    )
    return compiled


def show_generated_code():
    """Show generated CUDA code."""
    kernel = kernel_barrier_all_on_stream
    cuda_code = kernel.compile(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        need_header=True,
        num_threads=NUM_THREADS,
    )
    print("=" * 80)
    print("Generated CUDA code:")
    print("=" * 80)
    print(cuda_code)
    return cuda_code


if __name__ == "__main__":
    import sys
    if "--codegen" in sys.argv:
        show_generated_code()
    else:
        print("=== Building kernel_barrier_all_on_stream ===")
        cuda_code = show_generated_code()

        print("\n=== Compiling kernel ===")
        compiled = build_kernel()
        print("Kernel compiled successfully!")
        print(f"Kernel name: {compiled.kernel_name}")
