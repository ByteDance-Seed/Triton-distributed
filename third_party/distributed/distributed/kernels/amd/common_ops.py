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

from triton.distributed.utils import (
    HIP_CHECK, )
from hip import hip

from triton.language.extra.hip import libdevice


@triton.jit
def wait_eq_sys(barrier_ptr, value):
    thread_idx = libdevice.thread_idx(axis=0)
    if thread_idx == 0:
        while libdevice.load_acquire_system(barrier_ptr) != value:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all_ipc(rank, num_ranks, comm_buf_base_ptrs):
    tid = libdevice.thread_idx(axis=0)  # noqa: F841
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        # tl.device_print("remote_base_ptr", remote_base_ptr)
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr,
    stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc[(1, )](rank, num_ranks, sync_bufs_ptr)


def wait_eq(ptr: int, val: int, stream: torch.cuda.Stream, require_i64=False):
    mask = 0xFFFFFFFF
    if not require_i64:
        call_result = hip.hipStreamWaitValue32(
            stream.cuda_stream,
            ptr,
            val,
            hip.hipStreamWaitValueEq,
            mask,
        )
    else:
        call_result = hip.hipStreamWaitValue64(
            stream.cuda_stream,
            ptr,
            val,
            hip.hipStreamWaitValueEq,
            mask,
        )
    HIP_CHECK(call_result)


def set_signal(ptr: int, val: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        call_result = hip.hipStreamWriteValue32(
            stream.cuda_stream,
            ptr,
            val,
            0,
        )
    else:
        call_result = hip.hipStreamWriteValue64(
            stream.cuda_stream,
            ptr,
            val,
            0,
        )
    HIP_CHECK(call_result)
