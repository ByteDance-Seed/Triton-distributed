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
from typing import Optional

from cuda import cuda, cudart
import nvshmem.bindings.nvshmem as pynvshmem
import nvshmem.core
import torch

import triton
import triton.language as tl
from triton_dist.utils import CUDA_CHECK, NVSHMEM_SIGNAL_DTYPE
import triton_dist.language as dl
from triton.language.extra.cuda.language_extra import (__syncthreads, atomic_add, atomic_cas, ld, ld_acquire, st, tid)
from triton_dist.utils import (supports_p2p_native_atomic, nvshmem_barrier_all_on_stream, nvshmem_create_tensor)


@triton.jit
def _is_cta_master():
    thread_idx_x = tid(0)
    thread_idx_y = tid(1)
    thread_idx_z = tid(2)
    return (thread_idx_x + thread_idx_y + thread_idx_z) == 0


@triton.jit
def _is_gpu_master():
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    pid_z = tl.program_id(axis=2)
    return (pid_x + pid_y + pid_z) == 0


@triton.jit
def unsafe_barrier_on_this_grid(ptr):
    """ triton implementation of cooperative_group::thid_grid().sync()
    WARNING: use with care. better launch triton with launch_cooperative_grid=True to throw an explicit error instead of hang without notice.
    """
    __syncthreads()
    pid_size_x = tl.num_programs(axis=0)
    pid_size_y = tl.num_programs(axis=1)
    pid_size_z = tl.num_programs(axis=2)
    expected = pid_size_x * pid_size_y * pid_size_z
    if _is_cta_master():
        nb = tl.where(
            _is_gpu_master(),
            tl.cast(0x80000000, tl.uint32, bitcast=True) - (expected - 1),
            1,
        )
        old_arrive = atomic_add(ptr.to(tl.pointer_type(tl.uint32)), nb, scope="gpu", semantic="release")
    else:
        old_arrive = tl.cast(0, tl.uint32)

    if _is_cta_master():
        current_arrive = ld_acquire(ptr)
        while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
            current_arrive = ld_acquire(ptr, scope=tl.constexpr("gpu"))

    __syncthreads()


@triton.jit
def load_envreg(val: tl.constexpr):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %envreg{val};",
        constraints=("=r"),
        args=[],
        dtype=(tl.uint32),
        is_pure=True,
        pack=1,
    )


@triton.jit
def load_grid_ws_abi_address():
    envreg1 = load_envreg(tl.constexpr(1))
    envreg2 = load_envreg(tl.constexpr(2))
    grid_ws_abi_address = (tl.cast(envreg1, tl.uint64) << 32) | tl.cast(envreg2, tl.uint64)
    return tl.cast(grid_ws_abi_address, tl.pointer_type(tl.uint32), bitcast=True)


@triton.jit
def cooperative_barrier_on_this_grid():
    """ triton implementation of cooperative_group::thid_grid().sync()
    WARNING: use with care. better launch triton with launch_cooperative_grid=True to throw an explicit error instead of hang without notice.
    """
    ptr = load_grid_ws_abi_address() + 1
    __syncthreads()
    pid_size_x = tl.num_programs(axis=0)
    pid_size_y = tl.num_programs(axis=1)
    pid_size_z = tl.num_programs(axis=2)
    expected = pid_size_x * pid_size_y * pid_size_z
    if _is_cta_master():
        nb = tl.where(
            _is_gpu_master(),
            tl.cast(0x80000000, tl.uint32, bitcast=True) - (expected - 1),
            1,
        )
        old_arrive = atomic_add(ptr.to(tl.pointer_type(tl.uint32)), nb, scope="gpu", semantic="release")
    else:
        old_arrive = tl.cast(0, tl.uint32)

    if _is_cta_master():
        current_arrive = ld_acquire(ptr)
        while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
            current_arrive = ld_acquire(ptr, scope=tl.constexpr("gpu"))

    __syncthreads()


@triton.jit
def barrier_on_this_grid(ptr, use_cooperative: tl.constexpr):
    if use_cooperative:
        cooperative_barrier_on_this_grid()
    else:
        unsafe_barrier_on_this_grid(ptr)


@triton.jit(do_not_specialize=["local_rank", "rank", "local_world_size"])
def barrier_all_intra_node_atomic_cas_block(local_rank, rank, local_world_size, symm_flag_ptr):
    """ NOTE: this function should only be called with atomic support. memory over PCI-e does not support atomic r/w. DON'T use this function on such platforms.
    """
    with dl.simt_exec_region() as (thread_idx, block_size):
        local_rank_offset = rank - local_rank
        if thread_idx < local_world_size:  # thread_idx => local_rank
            remote_ptr = dl.symm_at(symm_flag_ptr + local_rank, thread_idx + local_rank_offset)
            while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
                pass

        if thread_idx < local_world_size:  # thread_idx => local_rank
            while (atomic_cas(symm_flag_ptr + thread_idx, 1, 0, "sys", "acquire") != 1):
                pass
        __syncthreads()


@triton.jit
def _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, local_world_size, symm_flags, target_value):
    with dl.simt_exec_region() as (thread_idx, block_size):
        if thread_idx < local_world_size:  # thread_idx => local_rank
            local_rank_offset = rank - local_rank
            remote_ptr = dl.symm_at(symm_flags + local_rank, thread_idx + local_rank_offset)
            st(remote_ptr, target_value, scope="sys", semantic="release")
            while ld(symm_flags + thread_idx, scope="sys", semantic="acquire") != target_value:
                pass
        __syncthreads()


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "target_value"])
def barrier_all_intra_node_non_atomic_block(local_rank, rank, num_ranks, symm_flags, target_value):
    """ symm_flags is expected to:
        1. of int32 dtype
        2. has at least num_ranks * 2 elements
        3. of symmetric pointer

        symm_flags [0, num_ranks * 2) is used to sync all ranks.
    """
    tl.static_assert(symm_flags.dtype.element_ty == tl.int32 or symm_flags.dtype.element_ty == tl.int64)
    _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flags, target_value)
    # next iter
    _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flags + num_ranks, target_value)


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "target_value"])
def barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_flags, target_value,
                                      use_cooperative: tl.constexpr):
    """ symm_flags is expected to:
        1. of int32 dtype
        2. has at least num_ranks * 2 + 1 elements
        3. of symmetric pointer

        symm_flags [0, num_ranks * 2) is used to sync all ranks.
        symm_flags[num_ranks * 2] is used to sync all CTAs
    """
    tl.static_assert(symm_flags.dtype.element_ty == tl.int32)
    pid = tl.program_id(axis=0)
    if pid == 0:
        _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flags, target_value)

    # barrier all CTAs
    barrier_on_this_grid(symm_flags + 2 * num_ranks, use_cooperative)

    # next iter
    if pid == 0:
        _barrier_all_intra_node_non_atomic_once_block(local_rank, rank, num_ranks, symm_flags + num_ranks, target_value)

    # barrier all CTAs
    barrier_on_this_grid(symm_flags + 2 * num_ranks, use_cooperative)


class BarrierAllContext:
    """
    You may use this to barrier all ranks in global, or just in intra-node team.

    NOTE: nvshmem_barrier_all is slower for intra-node only.
    """

    def __init__(self, is_intra_node):
        self.is_intra_node = is_intra_node
        self.target_value = 1
        if self.is_intra_node:
            self.rank = pynvshmem.my_pe()
            self.local_rank = pynvshmem.team_my_pe(nvshmem.core.Teams.TEAM_NODE)
            self.num_local_ranks = pynvshmem.team_n_pes(nvshmem.core.Teams.TEAM_NODE)
            self.symm_barrier = nvshmem_create_tensor((self.num_local_ranks, ), torch.int32)
            self.symm_barrier.fill_(0)
            nvshmem_barrier_all_on_stream(torch.cuda.current_stream())


def barrier_all_on_stream(ctx: BarrierAllContext, stream: Optional[torch.cuda.Stream] = None):
    """
    barrier_all_on_stream does not support CUDAGraph
    """
    if ctx is None or not ctx.is_intra_node:
        return nvshmem_barrier_all_on_stream(stream)

    if supports_p2p_native_atomic():
        barrier_all_intra_node_atomic_cas_block[(1, )](ctx.local_rank, ctx.rank, ctx.num_local_ranks, ctx.symm_barrier)
    else:
        barrier_all_intra_node_non_atomic_block[(1, )](ctx.local_rank, ctx.rank, ctx.num_local_ranks, ctx.symm_barrier,
                                                       ctx.target_value)
        ctx.target_value += 1


@tl.constexpr_function
def log2(n):
    return len(bin(n)) - 3


@tl.constexpr_function
def next_power_of_2(n: tl.constexpr):
    return triton.next_power_of_2(n)


@triton.jit
def bisect_left_kernel_aligned(sorted_values_ptr,  # Pointer to sorted input array of K
                               target_values,  # Pointer to search values of N
                               N: tl.constexpr,  # K should be power of 2
                               ):
    # Binary search initialization
    low = tl.full((target_values.numel, ), 0, dtype=tl.int32)
    high = tl.full((target_values.numel, ), N, dtype=tl.int32)  # Length of the sorted array

    N_LOG2 = log2(N)
    # Binary search loop
    for n in tl.range(N_LOG2):
        mid = (low + high) // 2
        mid_val = tl.load(sorted_values_ptr + mid)
        # Update search bounds
        low = tl.where(mid_val < target_values, mid + 1, low)
        high = tl.where(mid_val >= target_values, mid, high)

    low = tl.where(
        low != high and tl.load(sorted_values_ptr + low) < target_values,
        low + 1,
        low,
    )
    return low


@triton.jit
def bisect_left_kernel(
    sorted_values_ptr,  # Pointer to sorted input array of M
    target_values,  # Pointer to search values of L
    N: tl.constexpr,
):
    # Binary search initialization
    index = tl.full((target_values.numel, ), -1, dtype=tl.int32)

    # Binary search loop
    for i in tl.range(N):
        x = tl.load(sorted_values_ptr + i)
        # if index > 0 => index
        # if x > target_value => i
        # else => -1
        index = tl.where(index >= 0, index, tl.where(x >= target_values, i, -1))
    index = tl.where(index == -1, N, index)

    return index


@triton.jit
def bisect_right_kernel(sorted_values_ptr,  # Pointer to sorted input array (1D)
                        target_values,  # Pointer to search values (1D)
                        N: tl.constexpr,  # Length of sorted array
                        ):
    # Binary search initialization
    index = tl.full((target_values.numel, ), -1, dtype=tl.int32)

    # Binary search loop
    for i in tl.range(N):
        x = tl.load(sorted_values_ptr + i)
        # if index > 0 => index
        # if x > target_value => i
        # else => -1
        index = tl.where(index >= 0, index, tl.where(x > target_values, i, -1))
    index = tl.where(index == -1, N, index)
    return index


@triton.jit
def bisect_right_kernel_aligned(
    sorted_values_ptr,  # Pointer to sorted input array (1D)
    target_values,
    N: tl.constexpr,
):
    # Binary search initialization
    low = tl.full((target_values.numel, ), 0, dtype=tl.int32)
    high = tl.full((target_values.numel, ), N, dtype=tl.int32)  # Length of the sorted array

    N_LOG2 = log2(N)
    # Binary search loop
    for _ in tl.range(N_LOG2):
        mid = (low + high) // 2
        mid_val = tl.load(sorted_values_ptr + mid)

        # Update search bounds
        low = tl.where(mid_val <= target_values, mid + 1, low)
        high = tl.where(mid_val > target_values, mid, high)

    low = tl.where(low != high and tl.load(sorted_values_ptr + low) <= target_values, low + 1, low)
    # Store result
    return low


def _wait_eq_cuda(signal_tensor: torch.Tensor, signal: int, stream: Optional[torch.cuda.Stream] = None,
                  require_i64=False):
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        (err, ) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)
    elif signal_tensor.dtype == NVSHMEM_SIGNAL_DTYPE:
        (err, ) = cuda.cuStreamWaitValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")


def _set_signal_cuda(signal_tensor: torch.Tensor, signal: int, stream: Optional[torch.cuda.Stream] = None):
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        (err, ) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)
    elif signal_tensor.dtype == NVSHMEM_SIGNAL_DTYPE:
        (err, ) = cuda.cuStreamWriteValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
        CUDA_CHECK(err)
    else:
        raise Exception(f"Unsupported signal dtype {signal_tensor.dtype}")


def _memcpy_async_cuda(dst: torch.Tensor, src: torch.Tensor, nbytes: int, stream: Optional[torch.cuda.Stream] = None):
    stream = stream or torch.cuda.current_stream()
    (err, ) = cudart.cudaMemcpyAsync(dst.data_ptr(), src.data_ptr(), nbytes, cudart.cudaMemcpyKind.cudaMemcpyDefault,
                                     stream.cuda_stream)

    CUDA_CHECK(err)
