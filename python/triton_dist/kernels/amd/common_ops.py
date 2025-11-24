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
import triton_dist
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from typing import Optional

from triton_dist.language.extra.hip.language_extra import load, atomic_add, sync_grid, atomic_cas, tid, __syncthreads
from hip import hip
from triton_dist.utils import HIP_CHECK


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
        old_arrive = atomic_add(ptr.to(tl.pointer_type(tl.uint32)), nb, scope="agent", semantic="release")
    else:
        old_arrive = tl.cast(0, tl.uint32)

    if _is_cta_master():
        current_arrive = load(ptr, semantic="acquire", scope="agent")
        while ((old_arrive ^ current_arrive) & 0x80000000) == 0:
            current_arrive = load(ptr, semantic="acquire", scope="agent")

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
    sync_grid()  # use __ockl_grid_sync


@triton.jit
def barrier_on_this_grid(ptr, use_cooperative: tl.constexpr):
    if use_cooperative:
        cooperative_barrier_on_this_grid()
    else:
        unsafe_barrier_on_this_grid(ptr)


@triton.jit(do_not_specialize=["rank"])
def barrier_all_ipc_kernel(rank, num_ranks, comm_buf_base_ptrs):
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass


@triton.jit(do_not_specialize=["rank"])
def barrier_all_ipc_kernel_v2(rank, num_ranks, comm_buf_base_ptrs):
    if tid(0) < num_ranks:
        remote_base_ptr = tl.load(comm_buf_base_ptrs + tid(0)).to(tl.pointer_type(tl.int32))
        while atomic_cas(remote_base_ptr + rank, 0, 1, semantic="release", scope="system") != 0:
            pass
    __syncthreads()

    if tid(0) < num_ranks:
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while atomic_cas(local_base_ptr + tid(0), 1, 0, semantic="acquire", scope="system") != 1:
            pass
    __syncthreads()


@triton_dist.jit(do_not_specialize=["rank"])
def barrier_all_kernel(rank, num_ranks, comm_buf_ptr):
    for i in range(num_ranks):
        remote_base_ptr = dl.symm_at(comm_buf_ptr, i)
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = dl.symm_at(comm_buf_ptr, rank)
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


@triton_dist.jit
def barrier_all_with_ctx_kernel(ctx, rank, num_ranks, comm_buf_ptr):
    libshmem_device.set_rocshmem_ctx(ctx)
    barrier_all_kernel(rank, num_ranks, comm_buf_ptr)


def barrier_all_on_stream(
    rank,
    num_ranks,
    sync_bufs_ptr: torch.Tensor,
    stream: torch.cuda.Stream,
):
    with torch.cuda.stream(stream):
        barrier_all_ipc_kernel[(1, )](rank, num_ranks, sync_bufs_ptr)


def barrier_all_with_ctx_on_stream(
    ctx,
    rank,
    num_ranks,
    comm_buf: torch.Tensor,
    stream: torch.cuda.Stream,
):
    with torch.cuda.stream(stream):
        barrier_all_with_ctx_kernel[(1, )](ctx, rank, num_ranks, comm_buf)


def _wait_eq_hip(signal_tensor: torch.Tensor, val: int, stream: Optional[torch.cuda.Stream] = None):
    # This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    # please refer to: https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html#ga9ef06d564d19ef9afc11d60d20c9c541
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        err = hip.hipStreamWaitValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            hip.hipStreamWaitValueEq,
            0xFFFFFFFF,
        )
    else:
        err = hip.hipStreamWaitValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            hip.hipStreamWaitValueEq,
            0xFFFFFFFFFFFFFFFF,
        )
    HIP_CHECK(err)


def _set_signal_hip(signal_tensor: torch.Tensor, val: int, stream: Optional[torch.cuda.Stream] = None):
    # This API is marked as Beta. While this feature is complete, it can change and might have outstanding issues.
    # https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/group___stream_m.html#ga2520d4e1e57697edff2a85a3c03d652b
    stream = stream or torch.cuda.current_stream()
    if signal_tensor.dtype == torch.int32:
        err = hip.hipStreamWriteValue32(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            0,
        )
    else:
        err = hip.hipStreamWriteValue64(
            stream.cuda_stream,
            signal_tensor.data_ptr(),
            val,
            0,
        )
    HIP_CHECK(err)
