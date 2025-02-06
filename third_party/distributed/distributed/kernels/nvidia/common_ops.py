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
from triton.language.extra import libshmem_device
from triton.distributed.utils import (
    CUDA_CHECK, )
from cuda import cuda
from triton.language.extra.cuda.language_extra import (
    __syncthreads, )


@tl.core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: tl.constexpr,
    semantic: tl.constexpr,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.cas.b32 $0, [$1], $2, $3;",
        constraints=("=r,l,r,r"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def thread_id(axis: tl.constexpr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %tid.{axis.value};",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_ptr):
    tid = thread_id(axis="x")
    sm_id = tl.program_id(axis=0)
    if tid < num_ranks:
        remote_ptr = libshmem_device.remote_ptr(comm_buf_ptr + sm_id * num_ranks + rank,
                                                tid.to(tl.int32)).to(tl.pointer_type(tl.int32))
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass
        while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


@tl.core.extern
def red_release(barrier_ptr, value, scope: tl.constexpr = "gpu", _builder=None):
    tl.inline_asm_elementwise(
        asm=f"""{{
        mov.u32         $0, %tid.x;
        red.release.{scope.value}.global.add.s32 [$1], $2;
        }}""",
        constraints=("=r,"
                     "l,r"),  # no use output, which is threadId.x
        args=[barrier_ptr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def ld_acquire(barrier_ptr, scope: tl.constexpr = "gpu", _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"""{{
        ld.global.acquire.{scope.value}.b32 $0, [$1];
        }}
        """,
        constraints=("=r,l"),
        args=[barrier_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


def wait_eq(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = cuda.cuStreamWaitValue32(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
    else:
        (err, ) = cuda.cuStreamWaitValue64(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ,
        )
    CUDA_CHECK(err)


def set_signal(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = cuda.cuStreamWriteValue32(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    else:
        (err, ) = cuda.cuStreamWriteValue64(
            stream.cuda_stream,
            ptr,
            signal,
            cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
        )
    CUDA_CHECK(err)
