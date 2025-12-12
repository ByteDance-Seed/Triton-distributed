# Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
import triton
import torch
import triton.language as tl
from triton.language.extra.maca import libshmem_device
from triton_dist.utils import (
    MACA_CHECK, )
# from cuda import cuda
from triton.language.extra.maca.language_extra import (
    __syncthreads,)
from maca import maca


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
    return tl.inline_intrinsic_elementwise(
        intrinsic=f"llvm.mxc.thread.id.{axis.value}",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        _builder=_builder,
    )


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_ptr):
    tid = thread_id(axis="x")
    sm_id = tl.program_id(axis=0)
    if tid < num_ranks:
        remote_ptr = libshmem_device.remote_ptr(comm_buf_ptr + sm_id * num_ranks + rank,
                                                tid.to(tl.int32)).to(tl.pointer_type(tl.int32))
        #while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
        #    pass
        #while (atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
        #    pass
        while tl.atomic_cas(remote_ptr, 0, 1, "release", "sys", True) != 0: # use intrinsic
            pass
        while (tl.atomic_cas(comm_buf_ptr + sm_id * num_ranks + tid, 1, 0, "acquire", "sys", True) != 1): # use intrinsic
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


def _wait_eq_maca(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = maca.mcStreamWaitValue32(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWaitValue_flags.MC_STREAM_WAIT_VALUE_EQ,
        )
    else:
        (err, ) = maca.mcStreamWaitValue64(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWaitValue_flags.MC_STREAM_WAIT_VALUE_EQ,
        )
    MACA_CHECK(err)


def _set_signal_maca(ptr: int, signal: int, stream: torch.cuda.Stream, require_i64=False):
    if not require_i64:
        (err, ) = maca.mcStreamWriteValue32(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWriteValue_flags.MC_STREAM_WRITE_VALUE_DEFAULT,
        )
    else:
        (err, ) = maca.mcStreamWriteValue64(
            stream.cuda_stream,
            ptr,
            signal,
            maca.mcStreamWriteValue_flags.MC_STREAM_WRITE_VALUE_DEFAULT,
        )
    MACA_CHECK(err)
