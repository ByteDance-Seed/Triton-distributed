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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Fixture kernels for test_typed_int_cast_codegen.

Kernels are defined in a real module (not inline in the test) because the
frontend resolves kernel bodies via ``inspect.getsource``.
"""

import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend='cuda', is_entry=True)
def u32_to_i64(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.int64(ll.uint32(x[i]))


@lk.ll_kernel(backend='cuda', is_entry=True)
def i64_to_u32(x: ll.Tensor[ll.int64], y: ll.Tensor[ll.uint32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.uint32(ll.int64(x[i]))


@lk.ll_kernel(backend='cuda', is_entry=True)
def vc_u32_i64(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.val_cast(ll.val_cast(x[i], ll.uint32), ll.int64)


@lk.ll_kernel(backend='cuda', is_entry=True)
def vc_i64_u32(x: ll.Tensor[ll.int64], y: ll.Tensor[ll.uint32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.val_cast(ll.val_cast(x[i], ll.int64), ll.uint32)


@lk.ll_kernel(backend='cuda', is_entry=True)
def to_u32_i64(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.to(ll.val_cast(x[i], ll.uint32), ll.int64)


@lk.ll_kernel(backend='cuda', is_entry=True)
def nested(x: ll.Tensor[ll.int64], y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.int64(ll.uint32(ll.int64(x[i])))


@lk.ll_kernel(backend='cuda', is_entry=True)
def const_operand(y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1:
        y[i] = ll.int64(ll.uint32(11))


@lk.ll_kernel(backend='cuda', is_entry=True)
def i32_u32(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.uint32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.uint32(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def u32_i32(x: ll.Tensor[ll.uint32], y: ll.Tensor[ll.int32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.int32(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def i16_u16(x: ll.Tensor[ll.int16], y: ll.Tensor[ll.uint16]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.uint16(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def u16_i16(x: ll.Tensor[ll.uint16], y: ll.Tensor[ll.int16]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.int16(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def i8_u8(x: ll.Tensor[ll.int8], y: ll.Tensor[ll.uint8]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.uint8(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def u8_i8(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.int8]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.int8(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def i64_u64(x: ll.Tensor[ll.int64], y: ll.Tensor[ll.uint64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.uint64(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def u64_i64(x: ll.Tensor[ll.uint64], y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = ll.int64(x[i])


@lk.ll_kernel(backend='cuda', is_entry=True)
def paren_binding(y: ll.Tensor[ll.int64]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[i] = (ll.int64(ll.uint32(11)) + 2) * 3
