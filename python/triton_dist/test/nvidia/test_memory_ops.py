################################################################################
#
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import pytest
import torch
from triton_dist.kernels.nvidia.memory_ops import copy_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("large_stride_operand", ["source", "destination"])
def test_copy_2d_persistent_large_row_stride(large_stride_operand):
    row_stride = 1 << 30
    storage_offset = 1 << 31
    allocation_size = storage_offset + 2 * row_stride + 8
    free_bytes, _ = torch.cuda.mem_get_info()
    safety_margin = 1 << 30
    if free_bytes < allocation_size + safety_margin:
        pytest.skip(f"requires at least {(allocation_size + safety_margin) / 2**30:.1f} GiB free GPU memory")

    storage = torch.empty(allocation_size, dtype=torch.int8, device="cuda")
    storage[:8].fill_(7)
    strided_tensor = storage.as_strided((3, 8), (row_stride, 1), storage_offset)
    expected = torch.full((3, 8), 42, dtype=torch.int8, device="cuda")
    if large_stride_operand == "source":
        strided_tensor.copy_(expected)
        src = strided_tensor
        persistent_out = torch.empty_like(expected)
        tilewise_out = torch.empty_like(expected)
    else:
        strided_tensor.zero_()
        src = expected
        persistent_out = strided_tensor
        tilewise_out = strided_tensor

    copy_tensor(persistent_out, src, num_sms=1, persistent=True)
    persistent_result = persistent_out.clone()

    if large_stride_operand == "destination":
        storage[:8].fill_(7)
        tilewise_out.zero_()
    copy_tensor(tilewise_out, src, persistent=False)
    tilewise_result = tilewise_out.clone()

    torch.testing.assert_close(tilewise_result, expected)
    torch.testing.assert_close(persistent_result, expected)
    torch.testing.assert_close(storage[:8], torch.full_like(storage[:8], 7))
