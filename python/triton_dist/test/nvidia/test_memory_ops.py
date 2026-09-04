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
def test_copy_2d_persistent_large_row_stride():
    row_stride = 1 << 30
    storage_offset = 1 << 31
    allocation_size = storage_offset + 2 * row_stride + 8
    free_bytes, _ = torch.cuda.mem_get_info()
    safety_margin = 1 << 30
    if free_bytes < allocation_size + safety_margin:
        pytest.skip(f"requires at least {(allocation_size + safety_margin) / 2**30:.1f} GiB free GPU memory")

    storage = torch.empty(allocation_size, dtype=torch.int8, device="cuda")
    storage[:8].fill_(7)
    src = storage.as_strided((3, 8), (row_stride, 1), storage_offset)
    src.fill_(42)

    persistent_out = torch.empty_like(src, memory_format=torch.contiguous_format)
    tilewise_out = torch.empty_like(src, memory_format=torch.contiguous_format)
    copy_tensor(persistent_out, src, num_sms=1, persistent=True)
    copy_tensor(tilewise_out, src, persistent=False)

    expected = torch.full_like(persistent_out, 42)
    torch.testing.assert_close(tilewise_out, expected)
    torch.testing.assert_close(persistent_out, expected)
