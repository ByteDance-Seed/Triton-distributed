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
import math


def all_zeros(data_sizes):
    for val in data_sizes:
        if val != 0:
            return False
    return True


def _has_slice_intersection_for_diff_shape(start_indices1, data_sizes1, shape1, start_indices2, data_sizes2, shape2):
    nelems1 = math.prod(shape1)
    nelems2 = math.prod(shape2)
    if nelems1 != nelems2:
        raise ValueError(
            f"shape1 and shape2 must have the same number of elements. shape1 ={shape1}, shape2 = {shape2}")

    tensor = torch.zeros(shape1, dtype=torch.int32, device=torch.cuda.current_device())
    slices1 = []
    for idx, (l, len) in enumerate(zip(start_indices1, data_sizes1)):
        r = l + len
        if l < 0 or r > shape1[idx]:
            raise ValueError(
                f"illega desc, start_indices = {start_indices1}, data_sizes = {data_sizes1}, shape = {shape1}")
        slices1.append(slice(l, r))
    tensor[tuple(slices1)] = 1
    tensor_reshaped = tensor.reshape(shape2)

    slices2 = []
    for idx, (l, len) in enumerate(zip(start_indices2, data_sizes2)):
        r = l + len
        slices2.append(slice(l, r))
    tile2_sum = tensor_reshaped[tuple(slices2)].sum().item()
    return tile2_sum > 0


def has_slice_intersection(start_indices1, data_sizes1, shape1, start_indices2, data_sizes2, shape2):
    assert len(start_indices1) == len(data_sizes1)
    assert len(start_indices2) == len(data_sizes2)
    if all_zeros(data_sizes1) or all_zeros(data_sizes2):
        return False
    if len(data_sizes1) != len(data_sizes2):
        return _has_slice_intersection_for_diff_shape(start_indices1, data_sizes1, shape1, start_indices2, data_sizes2,
                                                      shape2)

    for i in range(len(shape1)):
        start1 = start_indices1[i]
        end1 = start1 + data_sizes1[i] - 1
        if end1 >= shape1[i]:
            end1 = shape1[i] - 1

        start2 = start_indices2[i]
        end2 = start2 + data_sizes2[i] - 1
        if end2 >= shape1[i]:
            end2 = shape1[i] - 1

        if start1 > end2 or start2 > end1:
            return False
    return True
