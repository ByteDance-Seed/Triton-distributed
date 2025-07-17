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
from triton.language import core as tlc
from typing import List


class simt_exec_region:

    def __init__(self, _builder=None):
        self._builder = _builder

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass


# Extension of dist triton: create op to load scalar from tile
def extract(input: tlc.tensor, indices: List, _semantic) -> tlc.tensor:
    dst_indices = []
    for idx in indices:
        if isinstance(idx, tlc.tensor):
            dst_indices.append(idx.handle)
        elif isinstance(idx, tlc.constexpr):
            dst_indices.append(_semantic._convert_elem_to_ir_value(idx, require_i64=False))
        else:
            raise ValueError(f"unsupported tensor index: {idx}")
    ret = _semantic.builder.create_extract(input.handle, dst_indices)
    return tlc.tensor(ret, input.dtype)


# Extension of dist triton: create op to store scalar to tile
def insert(input: tlc.tensor, scalar, indices, _semantic) -> tlc.tensor:
    if isinstance(indices, (tlc.tensor, tlc.constexpr)):
        indices = [indices]
    dst_indices = []
    for idx in indices:
        if isinstance(idx, tlc.tensor):
            dst_indices.append(idx.handle)
        elif isinstance(idx, tlc.constexpr):
            dst_indices.append(_semantic._convert_elem_to_ir_value(idx, require_i64=False))
        else:
            raise ValueError(f"unsupported tensor index: {idx}")
    return tlc.tensor(_semantic.builder.create_insert(scalar.handle, input.handle, dst_indices), input.type)
