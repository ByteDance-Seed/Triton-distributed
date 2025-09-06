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
from typing import List, Tuple
import torch


def build_tile_desc(full_shape: List[int], tile_sizes: List[int], tile_id: int,
                    return_valid_size=False) -> Tuple[List[int], List[int]]:
    num_tiles = []
    ndim = len(full_shape)
    assert len(tile_sizes) == ndim
    for x, y in zip(full_shape, tile_sizes):
        num_tiles.append(cdiv(x, y))
    strides = [1 for i in range(len(num_tiles))]
    for j in range(ndim - 1, 0, -1):
        strides[j - 1] *= strides[j]
    start_indices = []
    data_sizes = []
    for i in range(0, ndim):
        v = tile_id // strides[i]
        tile_id = tile_id % strides[i]
        st = v * tile_sizes[i]
        start_indices.append(st)
        valid_size = tile_sizes[i]
        if return_valid_size:
            valid_size = min(full_shape[i] - st, valid_size)
        data_sizes.append(valid_size)
    return start_indices, data_sizes


def cdiv(x: int, y: int) -> int:
    assert x > 0 and y > 0
    return (x + y - 1) // y


def torch_dtype_to_triton_dtype_str(torch_dtype):
    DTYPE_MAP = {
        torch.float16: "tl.float16",
        torch.bfloat16: "tl.bfloat16",
        torch.float: "tl.float32",
        torch.int32: "tl.int32",
    }
    assert torch_dtype in DTYPE_MAP, f"{torch_dtype} is not in DTYPE_MAP"
    return DTYPE_MAP[torch_dtype]
