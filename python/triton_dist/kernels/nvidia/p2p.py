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
import triton.language as tl
import triton_dist
import triton_dist.language as tdl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import tid, __syncthreads


@triton_dist.jit
def p2p_set_signal(
    signal_ptr,
    remote_pe,
    signal_value,
    num_barriers=1,
):
    pid = tl.program_id(0)
    thread_id = tid(0)
    if pid == 0 and thread_id < num_barriers:
        libshmem_device.signal_op(signal_ptr + thread_id, signal_value, libshmem_device.NVSHMEM_SIGNAL_SET, remote_pe)
    __syncthreads()


@triton_dist.jit
def p2p_wait_signal(
    signal_ptr,
    remote_pe,
    signal_value,
    num_barriers=1,
):
    pid = tl.program_id(0)
    thread_id = tid(0)
    signal_ptr = tdl.symm_at(signal_ptr + thread_id, remote_pe)
    if pid == 0 and thread_id < num_barriers:
        libshmem_device.signal_wait_until(
            signal_ptr,
            libshmem_device.NVSHMEM_CMP_EQ,
            signal_value,
        )
    __syncthreads()


@triton_dist.jit
def p2p_copy_kernel(
    src_ptr,
    src_pe,
    data_size_byte,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.int8))
    NUM_SMS = tl.num_programs(0)
    pid = tl.program_id(0)
    data_size_byte_per_pid = tl.cdiv(data_size_byte, NUM_SMS)
    data_copy_begin = pid * data_size_byte_per_pid
    data_copy_end = min(data_copy_begin + data_size_byte_per_pid, data_size_byte)

    data_copy_real_size = data_copy_end - data_copy_begin

    libshmem_device.getmem_block(
        src_ptr + data_copy_begin,
        src_ptr + data_copy_begin,
        data_copy_real_size,
        src_pe,
    )


@triton_dist.jit
def p2p_put_kernel(
    src_ptr,
    dst_ptr,
    dst_pe,
    data_size_byte,
    signal_ptr,
    signal_value,
):
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.int8))
    src_ptr = src_ptr.to(tl.pointer_type(tl.int8))
    NUM_SMS = tl.num_programs(0)
    pid = tl.program_id(0)
    data_size_byte_per_pid = tl.cdiv(data_size_byte, NUM_SMS)
    data_copy_begin = pid * data_size_byte_per_pid
    data_copy_end = min(data_copy_begin + data_size_byte_per_pid, data_size_byte)

    data_copy_real_size = data_copy_end - data_copy_begin

    libshmem_device.putmem_signal_nbi_block(
        dst_ptr + data_copy_begin,
        src_ptr + data_copy_begin,
        data_copy_real_size,
        signal_ptr + pid,
        signal_value,
        libshmem_device.NVSHMEM_SIGNAL_SET,
        dst_pe,
    )


@triton_dist.jit
def p2p_copy_remote_to_local_kernel(
    src_ptr,
    src_pe,
    dst_ptr,
    data_size_byte,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    src_ptr = src_ptr.to(tl.pointer_type(tl.int8))
    dst_ptr = dst_ptr.to(tl.pointer_type(tl.int8))
    pid = tl.program_id(0)
    data_size_byte_per_pid = tl.cdiv(data_size_byte, NUM_SMS)
    data_copy_begin = pid * data_size_byte_per_pid
    data_copy_end = min(data_copy_begin + data_size_byte_per_pid, data_size_byte)

    data_copy_real_size = data_copy_end - data_copy_begin

    # for this chunk of data, it is blocking
    libshmem_device.getmem_block(
        src_ptr + data_copy_begin,
        src_ptr + data_copy_begin,
        data_copy_real_size,
        src_pe,
    )

    offs = tl.arange(0, BLOCK_SIZE)
    num_iters = tl.cdiv(data_copy_real_size, BLOCK_SIZE)
    for i in range(0, num_iters):
        data = tl.load(src_ptr + data_copy_begin + i * BLOCK_SIZE + offs, mask=i * BLOCK_SIZE + offs
                       < data_copy_real_size)
        tl.store(dst_ptr + data_copy_begin + i * BLOCK_SIZE + offs, data, mask=i * BLOCK_SIZE + offs
                 < data_copy_real_size)
