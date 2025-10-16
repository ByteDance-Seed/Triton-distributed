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
import triton
import torch
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from typing import Optional

from triton.language.extra.hip.libdevice import (thread_idx, load_acquire_system)
from hip import hip
from triton_dist.utils import HIP_CHECK


@triton.jit
def wait_eq_sys(barrier_ptr, value):
    tid = thread_idx(axis=0)
    if tid == 0:
        while load_acquire_system(barrier_ptr) != value:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all_kernel(rank, num_ranks, comm_buf_base_ptrs):
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all_with_ctx_kernel(ctx, rank, num_ranks, comm_buf_ptr):
    libshmem_device.set_rocshmem_ctx(ctx)
    for i in range(num_ranks):
        remote_base_ptr = dl.symm_at(comm_buf_ptr, i)
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = dl.symm_at(comm_buf_ptr, rank)
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream: torch.cuda.Stream,
):
    with torch.cuda.stream(stream):
        barrier_all_kernel[(1, )](rank, num_ranks, sync_bufs_ptr)


def barrier_all_with_ctx_on_stream(
    ctx,
    rank,
    num_ranks,
    comm_buf_ptr,
    stream: torch.cuda.Stream,
):
    with torch.cuda.stream(stream):
        barrier_all_with_ctx_kernel[(1, )](ctx, rank, num_ranks, comm_buf_ptr)


def _wait_eq_hip(signal_tensor: torch.Tensor, val: int, stream: Optional[torch.cuda.Stream] = None):
    # This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    # please refer to: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html#ga9ef06d564d19ef9afc11d60d20c9c541
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        call_result = hip.hipStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            hip.hipStreamWaitValueEq,
            0xFFFFFFFF,
        )
    else:
        call_result = hip.hipStreamWaitValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            hip.hipStreamWaitValueEq,
            0xFFFFFFFFFFFFFFFF,
        )
    HIP_CHECK(call_result)


def _set_signal_hip(signal_tensor: torch.Tensor, val: int, stream: Optional[torch.cuda.Stream] = None):
    # This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    # https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html#ga2520d4e1e57697edff2a85a3c03d652b
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        call_result = hip.hipStreamWriteValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            0,
        )
    else:
        call_result = hip.hipStreamWriteValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            0,
        )
    HIP_CHECK(call_result)
