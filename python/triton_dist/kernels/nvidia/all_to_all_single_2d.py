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

from triton_dist.language.extra import libshmem_device
from triton_dist.utils import nvshmem_free_tensor_sync, nvshmem_create_tensor, nvshmem_barrier_all_on_stream


@triton.jit
def get_element_at(tensor: tl.tensor, idx: int):
    tl.static_assert(len(tensor.shape) == 1)
    size: tl.constexpr = tensor.shape[0]
    return tl.sum(tl.where(tl.arange(0, size) == idx, tensor, 0))


@triton.jit
def all_to_all_single_2d_kernel(
    data_src_ptr,
    data_dst_ptr,
    input_split_ptr,
    output_split_ptr,
    rank: int,
    WORLD_SIZE: tl.constexpr,
    MAX_M: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr = 2,
):
    pid = tl.program_id(0)
    npid = tl.num_programs(0)
    offset_pos = tl.arange(0, WORLD_SIZE)
    input_splits = tl.load(input_split_ptr + offset_pos)
    input_splits_cumsum = tl.cumsum(input_splits, axis=0)
    for tgt_rank in range(pid, WORLD_SIZE, npid):
        input_row_start = 0
        if tgt_rank > 0:
            input_row_start = get_element_at(input_splits_cumsum, tgt_rank - 1)
        input_row_end = get_element_at(input_splits_cumsum, tgt_rank)
        data_size = (input_row_end - input_row_start) * HIDDEN_DIM * ELEMENT_SIZE
        src_ptr = data_src_ptr + input_row_start * HIDDEN_DIM
        dst_ptr = data_dst_ptr + rank * MAX_M * HIDDEN_DIM
        libshmem_device.putmem_nbi_block(
            dst_ptr,
            src_ptr,
            data_size,
            tgt_rank,
        )


@triton.jit
def cp_from_recv_buf(
    output_ptr,
    recv_buf_ptr,
    output_split_ptr,
    WORLD_SIZE: tl.constexpr,
    MAX_M: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset_pos = tl.arange(0, WORLD_SIZE)
    output_splits = tl.load(output_split_ptr + offset_pos)
    output_splits_cumsum = tl.cumsum(output_splits, axis=0)
    n_row = tl.load(output_split_ptr + pid)
    output_row_offset = 0
    if pid > 0:
        output_row_offset = get_element_at(output_splits_cumsum, pid - 1)
    mask = tl.arange(0, BLOCK_SIZE) < HIDDEN_DIM
    offset = tl.arange(0, BLOCK_SIZE)

    for bid in range(0, n_row):
        data = tl.load(recv_buf_ptr + pid * MAX_M * HIDDEN_DIM + bid * HIDDEN_DIM + offset, mask=mask)
        tl.store(output_ptr + (output_row_offset + bid) * HIDDEN_DIM + offset, data, mask=mask)


class AllToAllSingle2DContext:

    def __init__(
        self,
        max_m: int,
        hidden_dim: int,
        rank: int,
        world_size: int,
        dtype=torch.bfloat16,
    ):
        self.send_buf = nvshmem_create_tensor((max_m, hidden_dim), dtype)
        self.recv_buf = nvshmem_create_tensor((max_m * world_size, hidden_dim), dtype)
        self.max_m = max_m
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.element_size = torch.tensor([], dtype=dtype).element_size()

    def finalize(self):
        nvshmem_free_tensor_sync(self.send_buf)
        nvshmem_free_tensor_sync(self.recv_buf)


def create_all_to_all_single_2d_context(
    max_m: int,
    hidden_dim: int,
    rank: int,
    world_size: int,
    dtype=torch.bfloat16,
):
    return AllToAllSingle2DContext(
        max_m,
        hidden_dim,
        rank,
        world_size,
        dtype,
    )


def all_to_all_single_2d(
    ctx: AllToAllSingle2DContext,
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    input_splits: torch.Tensor,
    output_splits: torch.Tensor,
    num_sm: int,
):
    assert input_splits.numel() == ctx.world_size, "input_splits must have world_size elements"
    assert output_splits.numel() == ctx.world_size, "output_splits must have world_size elements"
    assert input_tensor.dim() == 2, "input tensor dim() != 2 "
    assert output_tensor.dim() == 2, "output tensor dim() != 2"

    ctx.send_buf[:input_tensor.size(0)].data.copy_(input_tensor)

    grid = (num_sm, )
    all_to_all_single_2d_kernel[grid](ctx.send_buf, ctx.recv_buf, input_splits, output_splits, rank=ctx.rank,
                                      WORLD_SIZE=ctx.world_size, MAX_M=ctx.max_m, HIDDEN_DIM=ctx.hidden_dim,
                                      ELEMENT_SIZE=ctx.element_size, num_warps=32)
    nvshmem_barrier_all_on_stream()
    block_size = triton.next_power_of_2(ctx.hidden_dim)
    grid = (ctx.world_size, )
    cp_from_recv_buf[grid](
        output_ptr=output_tensor,
        recv_buf_ptr=ctx.recv_buf,
        output_split_ptr=output_splits,
        WORLD_SIZE=ctx.world_size,
        MAX_M=ctx.max_m,
        HIDDEN_DIM=ctx.hidden_dim,
        BLOCK_SIZE=block_size,
        num_warps=32,
    )
    return output_tensor
