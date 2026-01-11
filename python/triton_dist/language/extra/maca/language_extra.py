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
import triton.language as tl
from triton.language import core
# from triton.language.extra.maca.libdevice import ffs

@core.extern
def __syncthreads(_builder=None):
    return tl.debug_barrier(_builder=_builder)


@core.extern
def __tid__(axis: core.constexpr, _builder=None):
    return tl.inline_intrinsic_elementwise(
        intrinsic=f"llvm.mxc.thread.id.{axis.value}",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        _builder=_builder,
    )


@core.extern
def tid(axis: core.constexpr, _builder=None):
    if axis == 0:
        return __tid__(core.constexpr("x"), _builder=_builder)
    elif axis == 1:
        return __tid__(core.constexpr("y"), _builder=_builder)
    elif axis == 2:
        return __tid__(core.constexpr("z"), _builder=_builder)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


__all__ = [
    "__syncthreads",
    "tid",
]
