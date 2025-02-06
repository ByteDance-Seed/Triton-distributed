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
import pynvshmem

from typing import Optional
from triton.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid


@triton.jit
def all_to_all_kernel(
    data_src,
    data_dst,
    splits_src,
    splits_dst,
    signal,
    splits_cumsum,
    scale_src,
    scale_dst,
    rank: int,
    call_count: int,
    WITH_SCALE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    EXPERTS_PER_RANK: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    ELEMENT_SIZE: tl.constexpr = 2,
    SCALE_ELEMENT_SIZE: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    threadidx = tid(axis=0)

    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK

    m_st = tl.load(splits_cumsum + exp_st)
    m_ed = tl.load(splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    src_off = m_st
    dst_off = rank * MAX_M

    split_src_ptr = splits_src + exp_st
    off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
    off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
    cumsum_sts = tl.load(splits_cumsum + off0)
    cumsum_eds = tl.load(splits_cumsum + off1)
    tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK), cumsum_eds - cumsum_sts)

    act_pos = call_count % 2
    data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
    split_dst_ptr = splits_dst + act_pos * NUM_TOT_EXPERTS + rank * EXPERTS_PER_RANK
    signal_ptr = signal + act_pos * WORLD_SIZE + rank

    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src + src_off * HIDDEN,
        num_rows_cur_block * HIDDEN * ELEMENT_SIZE,
        pid,
    )
    libshmem_device.putmem_nbi_block(
        split_dst_ptr,
        split_src_ptr,
        EXPERTS_PER_RANK * 4,  # now we use `int32` for splits
        pid,
    )
    if WITH_SCALE:
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M + dst_off
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src + src_off,
            num_rows_cur_block * SCALE_ELEMENT_SIZE,
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )

    libshmem_device.fence()
    if threadidx == 0:
        if not WITH_SCALE:
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )


def dtype_size_in_bytes(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


class AllToAllContext:

    def __init__(
        self,
        max_m: int,
        hidden: int,
        rank: int,
        num_tot_experts: int,
        WORLD_SIZE: int,
        experts_per_rank: int,
        dtype=torch.bfloat16,
        scale_dtype=torch.float,
    ):
        """
        max_m: max number of tokens per rank
        """
        self.send_buf = pynvshmem.nvshmem_create_tensor([max_m, hidden], dtype)
        self.recv_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * max_m * 2, hidden], dtype)
        self.scale_send_buf = pynvshmem.nvshmem_create_tensor([max_m], scale_dtype)
        self.scale_recv_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * max_m * 2], scale_dtype)
        self.split_send_buf = pynvshmem.nvshmem_create_tensor([num_tot_experts], torch.int32)
        self.split_recv_buf = pynvshmem.nvshmem_create_tensor([num_tot_experts * 2], torch.int32)
        self.signal_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * 2], torch.uint64)

        self.max_m = max_m
        self.hidden = hidden
        self.dtype = dtype
        self.scale_dtype = scale_dtype
        self.ele_size = dtype_size_in_bytes(self.dtype)
        self.scale_ele_size = dtype_size_in_bytes(self.scale_dtype)

        self.num_tot_experts = num_tot_experts
        self.experts_per_rank = experts_per_rank

        self.WORLD_SIZE = WORLD_SIZE
        self.rank = rank

        # start from 1, becase the initial values of signal buffer is 0
        self.call_count = 1
        self.MOD_VALUE = 1000000


def create_all_to_all_context(
    max_m: int,
    hidden: int,
    rank: int,
    num_tot_experts: int,
    WORLD_SIZE: int,
    experts_per_rank: int,
    dtype=torch.bfloat16,
    scale_dtype=torch.float,
):
    return AllToAllContext(
        max_m,
        hidden,
        rank,
        num_tot_experts,
        WORLD_SIZE,
        experts_per_rank,
        dtype,
        scale_dtype,
    )


def fast_all_to_all(
    ctx: AllToAllContext,
    send_tensor: torch.Tensor,
    send_split_cumsum: torch.Tensor,
    send_scale: Optional[torch.Tensor],
):
    """
    low-latency all-to-all communication
    """
    with_scale = send_scale is not None

    act_pos = ctx.call_count % 2

    split_buf_st = act_pos * ctx.num_tot_experts
    split_buf_ed = split_buf_st + ctx.num_tot_experts

    data_buf_st = act_pos * ctx.WORLD_SIZE * ctx.max_m
    data_buf_ed = data_buf_st + ctx.WORLD_SIZE * ctx.max_m

    scale_buf_st = act_pos * ctx.WORLD_SIZE * ctx.max_m
    scale_buf_ed = scale_buf_st + ctx.WORLD_SIZE * ctx.max_m

    num_tokens = send_tensor.shape[0]
    assert num_tokens <= ctx.max_m
    ctx.send_buf[:num_tokens, :] = send_tensor
    if with_scale:
        ctx.scale_send_buf[:num_tokens] = send_scale

    grid = (ctx.WORLD_SIZE, )
    all_to_all_kernel[grid](
        data_src=ctx.send_buf,
        data_dst=ctx.recv_buf,
        splits_src=ctx.split_send_buf,
        splits_dst=ctx.split_recv_buf,
        signal=ctx.signal_buf,
        splits_cumsum=send_split_cumsum,
        scale_src=ctx.scale_send_buf,
        scale_dst=ctx.scale_recv_buf,
        rank=ctx.rank,
        call_count=ctx.call_count,
        WITH_SCALE=with_scale,
        WORLD_SIZE=ctx.WORLD_SIZE,
        HIDDEN=ctx.hidden,
        MAX_M=ctx.max_m,
        EXPERTS_PER_RANK=ctx.experts_per_rank,
        NUM_TOT_EXPERTS=ctx.num_tot_experts,
        ELEMENT_SIZE=ctx.ele_size,
        SCALE_ELEMENT_SIZE=ctx.scale_ele_size,
    )

    ctx.call_count = (ctx.call_count + 1) % ctx.MOD_VALUE
    out_lis: list[torch.Tensor] = []
    out_lis.append(ctx.split_recv_buf[split_buf_st:split_buf_ed])
    out_lis.append(ctx.recv_buf[data_buf_st:data_buf_ed, :])
    if with_scale:
        out_lis.append(ctx.scale_recv_buf[scale_buf_st:scale_buf_ed])
    else:
        out_lis.append(None)

    return out_lis


def all_to_all_post_process(
    ctx: AllToAllContext,
    input_splits: torch.Tensor,
    recv_buffer: torch.Tensor,
    scale_buffer: Optional[torch.Tensor] = None,
):
    with_scale = scale_buffer is not None
    world_size = ctx.WORLD_SIZE
    num_tokens_from_each_rank = input_splits.reshape(world_size, -1).sum(dim=1)
    data_vec, scale_vec = [], []
    for i in range(world_size):
        n_token_from_tgt_rank = num_tokens_from_each_rank[i]
        _start = i * ctx.max_m
        data_vec.append(recv_buffer[_start:_start + n_token_from_tgt_rank])
        if with_scale:
            scale_vec.append(scale_buffer[_start:_start + n_token_from_tgt_rank])
    output = torch.concat(data_vec)
    output_scale = torch.concat(scale_vec) if with_scale else None

    return output, output_scale
