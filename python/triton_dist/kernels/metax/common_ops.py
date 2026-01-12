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
import torch
import triton.language as tl
from triton_dist.utils import (
    MACA_CHECK, )
from maca import maca


@tl.core.extern
def thread_id(axis: tl.constexpr, _builder=None):
    return tl.inline_intrinsic_elementwise(
        intrinsic=f"llvm.mxc.thread.id.{axis.value}",
        args=[],
        dtype=tl.int32,
        is_pure=True,
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
