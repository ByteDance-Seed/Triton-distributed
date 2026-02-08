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
import triton
import triton.language as tl


@triton.jit
def memcpy_async_kernel(src_ptr, dst_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    num_blocks = tl.cdiv(N, BLOCK_SIZE)

    for bid in tl.range(pid, num_blocks, npid):
        offs = bid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        val = tl.load(src_ptr + offs, mask)
        tl.store(dst_ptr + offs, val, mask)


def memcpy_async(src: torch.Tensor, dst: torch.Tensor, num_workgroups: int, block_size: int, num_warps: int):
    assert src.is_contiguous() and src.is_cuda
    assert dst.is_contiguous() and dst.is_cuda
    assert src.dtype == dst.dtype and src.numel() == dst.numel()
    memcpy_async_kernel[(num_workgroups, )](src, dst, src.numel(), block_size, num_warps=num_warps)
