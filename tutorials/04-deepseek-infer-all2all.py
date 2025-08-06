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
"""
Low Latency All-to-All Communication
====================================
In this tutorial, we demonstrate how to implement the All-to-All communication
paradigm in Expert Parallelism (EP) for MoE models using Triton-distributed.

First, let's quickly review the EP workflow:
In MoE, the `E` experts are distributed across `N` devices (EP ranks).
For simplicity, we assume that `N` divides `E` evenly, so experts are distributed
uniformly. For example, when `E = 128` and `N = 32`, each device will handle 4 experts.

During inference with EP, each device is assigned a subset of tokens, as determined
by the MoE router module. The router on each device generates a tensor of
shape `[num_tokens, topk]`, containing the indices of the top `k` experts selected
for each token. The experts chosen for a token may reside on other devices,
necessitating communication to send the tokens to the appropriate devices.
Similarly, if other devices have tokens that select experts located on the current device,
those tokens need to be sent to the current device as well. This process is called Dispatch.

After the tokens are processed by their corresponding experts, they need to be
returned to their original devices. This operation mirrors Dispatch and is referred
to as Combine. From a communication perspective, both Dispatch and Combine are
essentially All-to-All collective communication operations.

Next, we demonstrate how to implement an efficient All-to-All operation in Triton-distributed with minimal code.

Triton-distributed provides a programming model that allows fine-grained control
over data movement between devices, optimizing hardware utilization.
At the core of our implementation are low-level primitives that manage the communication logic.

.. code-block:: bash

    # To run this tutorial
    source ./scripts/sentenv.sh
    bash ./scripts/launch.sh tutorials/04-deepseek-infer-all2all.py

"""
import os
import torch
import torch.distributed
import triton
import triton.language as tl
import random
import argparse

import nvshmem.core
from triton_dist.language.extra import libshmem_device
from triton.language.extra.cuda.language_extra import tid
from triton_dist.utils import dist_print, initialize_distributed, NVSHMEM_SIGNAL_DTYPE, nvshmem_free_tensor_sync, nvshmem_create_tensor


@triton.jit
def ceil_div(a, b):
    return (a + b - 1) // b


FP8_MAX = tl.constexpr(torch.finfo(torch.float8_e4m3fn).max)
FP8_MAX_INV = tl.constexpr(1 / 448.)
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


# use static config for fast compile in this tutorial
@triton.autotune(configs=[
    triton.Config(kwargs={'BM': BM}, num_warps=w) for BM in [16] for w in [16]
],
                 key=[])
@triton.jit
def all_to_all_kernel(
    send_tensor,
    data_src,
    data_dst,
    scale_src,
    scale_dst,
    splits_src,
    splits_dst,
    signal,
    send_splits_cumsum,
    recv_offset,
    rank: int,
    call_count: int,
    act_pos: int,
    MODE: tl.constexpr,
    ONLINE_QUANT_FP8: tl.constexpr,
    FP8_GSIZE: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HIDDEN: tl.constexpr,
    MAX_M: tl.constexpr,
    NUM_TOT_EXPERTS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """
    All-to-All kernel for the Dispatch and Combine phases.

    - send_tensor: The tokens to be sent.
    - data/scale/splits_src/dst: The source and destination symmetric buffers for communication.
    - signal: signal buffer for communication.
    - send_splits_cumsum: Cumulative sum of the token splits (expert-level) for the current rank.
    - recv_offset: only used in combine mode, the base offset of the received tokens.
    - call_count: as the unique ID used for signal operation.
    - act_pos: The position of the active buffer (0 or 1) for double buffering.

    - MODE: Determines whether the operation is Dispatch (0) or Combine (1).
    - ONLINE_QUANT_FP8: A flag indicating whether FP8 quantization is used.
    - FP8_GSIZE: The group size for FP8 quantization.
    - WORLD_SIZE: number of EP ranks.
    - HIDDEN: The hidden size for each token.
    - MAX_M: The maximum number of tokens that can be processed per rank.
    - EXPERTS_PER_RANK: The number of experts handled by each rank.
    - NUM_TOT_EXPERTS: The total number of experts.
    - BM, BN: Block size used to copy data to send buffer
    """
    pid = tl.program_id(0)
    # Triton-distributed exposes `tid` that can be used to identify the thread index within a CTA
    threadidx = tid(axis=0)
    NUM_GROUPS: tl.constexpr = HIDDEN // FP8_GSIZE
    EXPERTS_PER_RANK: tl.constexpr = NUM_TOT_EXPERTS // WORLD_SIZE

    # 1. Calculate the token range for the current program (rank), get the corresponding pointer
    exp_st = pid * EXPERTS_PER_RANK
    exp_ed = exp_st + EXPERTS_PER_RANK
    m_st = tl.load(send_splits_cumsum + exp_st)
    m_ed = tl.load(send_splits_cumsum + exp_ed)
    num_rows_cur_block = m_ed - m_st

    # Signal pointer to communicate when data is ready
    signal_ptr = signal + act_pos * WORLD_SIZE + rank
    if MODE == 0:  # dispatch mode
        # Calculate source and destination offsets based on the expert-level token number cumsum
        split_src_ptr = splits_src + (exp_st + pid)
        split_dst_ptr = splits_dst + act_pos * (
            NUM_TOT_EXPERTS + WORLD_SIZE) + rank * (EXPERTS_PER_RANK + 1)

        off0 = exp_st + tl.arange(0, EXPERTS_PER_RANK)
        off1 = exp_st + tl.arange(0, EXPERTS_PER_RANK) + 1
        cumsum_sts = tl.load(send_splits_cumsum + off0)
        cumsum_eds = tl.load(send_splits_cumsum + off1)
        tl.store(split_src_ptr + tl.arange(0, EXPERTS_PER_RANK),
                 cumsum_eds - cumsum_sts)
        tl.store(split_src_ptr + EXPERTS_PER_RANK, m_st)

        # Calculate the source and destination data offsets for the dispatch operation
        src_off = m_st
        dst_off = rank * MAX_M
        data_src_ptr = data_src + src_off * HIDDEN
        data_dst_ptr = data_dst + act_pos * WORLD_SIZE * MAX_M * HIDDEN + dst_off * HIDDEN
        scale_src_ptr = scale_src + src_off * NUM_GROUPS
        scale_dst_ptr = scale_dst + act_pos * WORLD_SIZE * MAX_M * NUM_GROUPS + dst_off * NUM_GROUPS
    else:  # combine mode
        # For the combine phase, source and destination offsets are updated accordingly
        src_off = pid * MAX_M
        dst_off = tl.load(recv_offset + pid)
        data_src_ptr = data_src + act_pos * WORLD_SIZE * MAX_M * HIDDEN + src_off * HIDDEN
        data_dst_ptr = data_dst + dst_off * HIDDEN
        scale_src_ptr = scale_src + act_pos * WORLD_SIZE * MAX_M * NUM_GROUPS + src_off * NUM_GROUPS
        scale_dst_ptr = scale_dst + dst_off * NUM_GROUPS

    # 2. Copy the data (may be online quantized to FP8) to send buffer
    off_m = tl.arange(0, BM)
    if ONLINE_QUANT_FP8 and MODE == 0:
        # TODO: adaptive UNROLL_FACTOR
        UNROLL_FACTOR: tl.constexpr = 4
        group_offs = off_m[:, None] * HIDDEN + tl.arange(
            0, FP8_GSIZE * UNROLL_FACTOR)[None, :]
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + group_offs
        data_src_ptrs = tl.cast(data_src_ptr, tl.pointer_type(
            tl.float8e4nv)) + group_offs
        scale_src_ptrs = scale_src_ptr + off_m[:,
                                               None] * NUM_GROUPS + tl.arange(
                                                   0, UNROLL_FACTOR)[None, :]
        # online quant the input data to FP8
        for i in tl.range(ceil_div(num_rows_cur_block, BM)):
            group_mask = off_m[:, None] < num_rows_cur_block - i * BM
            for _ in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
                group = tl.reshape(tl.load(send_tensor_ptrs, group_mask),
                                   (BM * UNROLL_FACTOR, FP8_GSIZE))
                scale = tl.max(tl.abs(group), 1, keep_dims=True).to(
                    tl.float32) * FP8_MAX_INV
                quant = tl.reshape(
                    (group.to(tl.float32) / scale).to(tl.float8e4nv),
                    (BM, UNROLL_FACTOR * FP8_GSIZE))
                tl.store(data_src_ptrs, quant, group_mask)
                tl.store(scale_src_ptrs,
                         tl.reshape(scale, (BM, UNROLL_FACTOR)), group_mask)
                send_tensor_ptrs += UNROLL_FACTOR * FP8_GSIZE
                data_src_ptrs += UNROLL_FACTOR * FP8_GSIZE
                scale_src_ptrs += UNROLL_FACTOR
            send_tensor_ptrs += (BM - 1) * HIDDEN
            data_src_ptrs += (BM - 1) * HIDDEN
            scale_src_ptrs += (BM - 1) * NUM_GROUPS
    else:
        off_n = tl.arange(0, BN)
        send_tensor_ptrs = send_tensor + m_st * HIDDEN + off_m[:,
                                                               None] * HIDDEN + off_n[
                                                                   None, :]
        data_src_ptrs = data_src_ptr + off_m[:, None] * HIDDEN + off_n[None, :]
        for i in tl.range(ceil_div(num_rows_cur_block, BM)):
            data_mask = (off_m[:, None] < num_rows_cur_block -
                         i * BM) & (off_n[None, :] < HIDDEN)
            tl.store(data_src_ptrs, tl.load(send_tensor_ptrs, data_mask),
                     data_mask)
            send_tensor_ptrs += BM * HIDDEN
            data_src_ptrs += BM * HIDDEN

    # 3. Perform the memory copy operation using shared memory for inter-rank communication.
    #   the last argument is the peer id (id of target rank)
    libshmem_device.putmem_nbi_block(
        data_dst_ptr,
        data_src_ptr,
        num_rows_cur_block * HIDDEN *
        (1 if (ONLINE_QUANT_FP8 and MODE == 0) else 2),
        pid,
    )
    if MODE == 0:
        # Dispatch mode: send split information to the target rank
        libshmem_device.putmem_nbi_block(
            split_dst_ptr,
            split_src_ptr,
            (EXPERTS_PER_RANK + 1) * 4,  # now we use `int32` for splits
            pid,
        )
    # If online quantization is enbaled, signal the target rank with the scale data
    if ONLINE_QUANT_FP8:
        libshmem_device.putmem_signal_nbi_block(
            scale_dst_ptr,
            scale_src_ptr,
            num_rows_cur_block * NUM_GROUPS * 4,  # assume `float32` for scale
            signal_ptr,
            call_count,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            pid,
        )

    # 4. fence data transfer. Then wait for signal
    libshmem_device.fence()
    if threadidx == 0:
        # notify the target rank (here is the `pid`-th rank) that the data is ready by setting the signal
        if not ONLINE_QUANT_FP8:
            libshmem_device.signal_op(
                signal_ptr,
                call_count,
                libshmem_device.NVSHMEM_SIGNAL_SET,
                pid,
            )
        # wait for the signal from the source rank (here is the `pid`-th rank)
        libshmem_device.signal_wait_until(
            signal + act_pos * WORLD_SIZE + pid,
            libshmem_device.NVSHMEM_CMP_EQ,
            call_count,
        )


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.empty(splits.shape[0] + 1,
                      dtype=splits.dtype,
                      device=splits.device)
    out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


def calc_gather_index(
    exp_indices: torch.Tensor,
    row_start: int,
    row_end: int,
    BLOCK_SIZE: int = 1024,
):

    @triton.jit
    def _kernel(
        scatter_index: torch.Tensor,
        gather_index: torch.Tensor,
        ntokens: int,
        topk: int,
        row_start: int,
        row_end: int,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < ntokens * topk
        scatter_idx = tl.load(scatter_index + offset, mask=mask, other=-1)
        token_idx = offset // topk
        token_idx_mask = (scatter_idx >= row_start) & (scatter_idx < row_end)
        tl.store(gather_index + scatter_idx - row_start,
                 token_idx,
                 mask=token_idx_mask)

    scatter_index = exp_indices.flatten().argsort(
        stable=True).argsort().int().view(exp_indices.shape)
    ntokens, topk = scatter_index.shape
    gather_index = torch.zeros(row_end - row_start,
                               dtype=torch.int32,
                               device=scatter_index.device)
    grid = lambda META: (triton.cdiv(ntokens * topk, META["BLOCK_SIZE"]), )
    _kernel[grid](
        scatter_index,
        gather_index,
        ntokens,
        topk,
        row_start,
        row_end,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return gather_index


@triton.jit
def _quant_kernel(out,
                  out_scale,
                  t,
                  m,
                  N: tl.constexpr,
                  FP8_GSIZE: tl.constexpr = 128,
                  BM: tl.constexpr = 16):
    pid = tl.program_id(0)
    FP8_MAX_INV = tl.constexpr(1 / 448.)
    NUM_GROUPS: tl.constexpr = N // FP8_GSIZE
    UNROLL_FACTOR: tl.constexpr = 4
    off_m = pid * BM + tl.arange(0, BM)
    off_n = tl.arange(0, UNROLL_FACTOR * FP8_GSIZE)
    input_ptrs = t + off_m[:, None] * N + off_n[None, :]
    out_ptrs = tl.cast(out, tl.pointer_type(
        tl.float8e4nv)) + off_m[:, None] * N + off_n[None, :]
    out_scale_ptrs = out_scale + off_m[:, None] * NUM_GROUPS + tl.arange(
        0, UNROLL_FACTOR)[None, :]
    for i in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
        group_mask = off_m[:, None] < m and (off_n[None, :]
                                             < N - i * FP8_GSIZE)
        scale_mask = tl.arange(0, UNROLL_FACTOR)[None, :] < NUM_GROUPS - i
        group = tl.reshape(tl.load(input_ptrs, group_mask, 0.),
                           (BM * UNROLL_FACTOR, FP8_GSIZE))
        scale = tl.max(tl.abs(group), 1, keep_dims=True).to(
            tl.float32) * FP8_MAX_INV
        quant = (group.to(tl.float32) / scale).to(tl.float8e4nv)
        tl.store(out_ptrs,
                 tl.reshape(quant, (BM, UNROLL_FACTOR * FP8_GSIZE)),
                 mask=group_mask)
        tl.store(out_scale_ptrs,
                 tl.reshape(scale, (BM, UNROLL_FACTOR)),
                 mask=scale_mask)
        input_ptrs += UNROLL_FACTOR * FP8_GSIZE
        out_ptrs += UNROLL_FACTOR * FP8_GSIZE
        out_scale_ptrs += UNROLL_FACTOR


@triton.jit
def _dequant_kernel(out,
                    input,
                    scales,
                    m,
                    N: tl.constexpr,
                    FP8_GSIZE: tl.constexpr = 128,
                    BM: tl.constexpr = 16):
    pid = tl.program_id(0)
    NUM_GROUPS: tl.constexpr = N // FP8_GSIZE
    UNROLL_FACTOR: tl.constexpr = 4
    off_m = pid * BM + tl.arange(0, BM)
    off_n = tl.arange(0, UNROLL_FACTOR * FP8_GSIZE)
    input_ptrs = tl.cast(input, tl.pointer_type(
        tl.float8e4nv)) + off_m[:, None] * N + off_n[None, :]
    input_scale_ptrs = scales + off_m[:, None] * NUM_GROUPS + tl.arange(
        0, UNROLL_FACTOR)[None, :]
    out_ptrs = out + off_m[:, None] * N + off_n[None, :]
    for i in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
        group_mask = off_m[:, None] < m and (off_n[None, :]
                                             < N - i * FP8_GSIZE)
        scale_mask = off_m[:, None] < m and (tl.arange(
            0, UNROLL_FACTOR)[None, :] < NUM_GROUPS - i)
        group = tl.reshape(tl.load(input_ptrs, group_mask, 0.),
                           (BM * UNROLL_FACTOR, FP8_GSIZE))
        scale = tl.reshape(tl.load(input_scale_ptrs, scale_mask, 0.),
                           (BM * UNROLL_FACTOR, 1))
        deq = (group.to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(out_ptrs,
                 tl.reshape(deq, (BM, UNROLL_FACTOR * FP8_GSIZE)),
                 mask=group_mask)
        input_ptrs += UNROLL_FACTOR * FP8_GSIZE
        input_scale_ptrs += UNROLL_FACTOR
        out_ptrs += UNROLL_FACTOR * FP8_GSIZE


def quant_bf16_fp8(tensor: torch.Tensor,
                   gsize: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    m, N = tensor.shape
    grid = (triton.cdiv(m, 16), )
    out = torch.empty((m, N // 2), dtype=torch.bfloat16, device=tensor.device)
    out_scale = torch.empty(m,
                            N // gsize,
                            dtype=torch.float32,
                            device=tensor.device)
    _quant_kernel[grid](out, out_scale, tensor, m, N)
    return out, out_scale


def dequant_fp8_bf16(q_tensor: torch.Tensor, scales: torch.Tensor):
    m, N = q_tensor.shape
    grid = (triton.cdiv(m, 16), )
    out = torch.empty([m, N * 2], dtype=torch.bfloat16, device=q_tensor.device)
    _dequant_kernel[grid](out, q_tensor, scales, m, N * 2)
    return out


def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for t in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        help="number of tokens per rank",
                        type=int,
                        default=8)
    parser.add_argument("-N", help="hidden size", type=int, default=7168)
    parser.add_argument("-G", help="number of experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--online_quant_fp8", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert (
        args.G % WORLD_SIZE == 0
    ), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"
    initialize_distributed()
    EXPERTS_PER_RANK = args.G // WORLD_SIZE
    MAX_NUM_TOKENS = args.M * args.topk
    ONLINE_QUANT = args.online_quant_fp8
    DTYPE = torch.bfloat16
    GROUP_SIZE = 128
    assert args.N % GROUP_SIZE == 0, f"N:{args.N} should be divisible by 128 for online FP8 quantization"
    NUM_GROUPS = args.N // GROUP_SIZE

    #####################
    # Prepare the input data:
    #   1. `input`: [num_tokens, HIDDEN] in each rank, which will be sent to other ranks
    #   2. `exp_indices`: [num_tokens, topk] in each rank, which indicates the experts that each token will be sent to
    #
    # num_tokens = random.randint(args.M // 2, args.M)
    num_tokens = args.M
    exp_indices = generate_random_exp_indices(num_tokens, args.G,
                                              args.topk).to("cuda")
    input_tokens_cur_rank = torch.rand(
        MAX_NUM_TOKENS, args.N, dtype=torch.float32).to(DTYPE).to("cuda")

    # The number of tokens for each expert; we use cumsum of the splits in the kernel
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1),
                                         minlength=args.G).to(torch.int32)
    split_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

    # gather `num_tokens * topk` input tokens according to `exp_indices`
    gather_idx_cur_rank = calc_gather_index(exp_indices, 0,
                                            num_tokens * args.topk)
    scattered_input_cur_rank = torch.empty(num_tokens * args.topk,
                                           args.N,
                                           dtype=DTYPE,
                                           device="cuda")
    scattered_input_cur_rank.copy_(
        torch.index_select(input_tokens_cur_rank,
                           dim=0,
                           index=gather_idx_cur_rank))
    #####################

    #####################
    # 1. we pre-define the max number of tokens that can be sent from one device `MAX_NUM_TOKENS`,
    #   which is typically 128 or 256, and reserve corresponding send/receive buffer size.
    #
    # NOTE: the size of receive buffer should be `WORLD_SIZE * MAX_NUM_TOKENS` because for the
    #   extremely imbalanced case: one single rank receive the data from all other ranks.
    #   We use *double buffer* for recieve buffer to avoid possible race condition, so the total size
    #   of receive buffer is `WORLD_SIZE * MAX_NUM_TOKENS * 2`.
    #
    #
    # 2. We also need to allocate split_buffer and send splits information to record
    #   the number of tokens received by each expert for subsequent calculations and communication.
    #
    # 3. The signal buffer is used to notify the target rank that the data is already ready.
    #   `nvshmem_create_tensor` is the low-level API to create shared memory
    #   between different devices (see https://docs.nvidia.com/nvshmem/api/gen/mem-model.html#memory-model).
    #   `WORLD_SIZE * 2` is for double buffer.
    #
    send_buf: torch.Tensor = nvshmem_create_tensor([MAX_NUM_TOKENS, args.N],
                                                   DTYPE)
    recv_buf: torch.Tensor = nvshmem_create_tensor(
        [WORLD_SIZE * MAX_NUM_TOKENS * 2, args.N], DTYPE)
    scale_send_buf: torch.Tensor = nvshmem_create_tensor(
        [MAX_NUM_TOKENS, NUM_GROUPS], torch.float32)
    scale_recv_buf: torch.Tensor = nvshmem_create_tensor(
        [WORLD_SIZE * MAX_NUM_TOKENS * 2, NUM_GROUPS], torch.float32)
    split_send_buf: torch.Tensor = nvshmem_create_tensor([args.G + WORLD_SIZE],
                                                         torch.int32)
    split_recv_buf: torch.Tensor = nvshmem_create_tensor(
        [(args.G + WORLD_SIZE) * 2], torch.int32)
    signal_buf: torch.Tensor = nvshmem_create_tensor([WORLD_SIZE * 2],
                                                     NVSHMEM_SIGNAL_DTYPE)
    #####################

    act_pos = 1
    # test 3 rounds in this example
    for round in range(1, 4):
        # flip the `act_pos` in each iteration for double buffer
        act_pos ^= 1
        # we launch the kernel with `WORLD_SIZE` blocks, each block handles the traffic of one rank
        grid = (WORLD_SIZE, )
        kwargs = {
            "ONLINE_QUANT_FP8":
            ONLINE_QUANT,  # whether to use online FP8 quantization
            "FP8_GSIZE":
            GROUP_SIZE,  # fixed quatization group size to 128 in this example
            "WORLD_SIZE": WORLD_SIZE,
            "HIDDEN": args.N,
            "MAX_M": MAX_NUM_TOKENS,
            "NUM_TOT_EXPERTS": args.G,  #
            "BN": 1 << (args.N - 1).bit_length(
            ),  # block size for copy data to send buffer; next_power_of_2(N)
        }
        #####################
        # 1. Dispatch
        all_to_all_kernel[grid](
            scattered_input_cur_rank,
            send_buf,
            recv_buf,
            scale_send_buf,
            scale_recv_buf,
            split_send_buf,
            split_recv_buf,
            signal_buf,
            split_cumsum,
            recv_offset=None,  # not used in dispatch mode
            rank=RANK,
            call_count=round *
            2,  # as we call the kernel twice in each iteration
            act_pos=act_pos,  # flip for double buffer
            MODE=0,  # dispatch mode
            **kwargs,
        )
        # 1.1. Post-process the dispatch output
        # this is for double buffer
        split_buf_st, split_buf_size = act_pos * (
            args.G + WORLD_SIZE), args.G + WORLD_SIZE
        data_buf_st, data_buf_size = act_pos * (
            WORLD_SIZE * MAX_NUM_TOKENS), WORLD_SIZE * MAX_NUM_TOKENS
        dis_splits_buf = split_recv_buf[split_buf_st:split_buf_st +
                                        split_buf_size]
        dis_tokens_buf = recv_buf[data_buf_st:data_buf_st + data_buf_size, :]
        dis_scales_buf = scale_recv_buf[data_buf_st:data_buf_st +
                                        data_buf_size, :]

        combine_offset = dis_splits_buf[torch.arange(1, WORLD_SIZE + 1) *
                                        (EXPERTS_PER_RANK + 1) - 1]
        combine_send_splits = dis_splits_buf.reshape(
            WORLD_SIZE, -1)[:, :EXPERTS_PER_RANK].flatten()
        num_tokens_from_each_rank = combine_send_splits.reshape(
            WORLD_SIZE, -1).sum(dim=1).tolist()
        off, token_vec, scale_vec = 0, [], []
        for ntk in num_tokens_from_each_rank:
            if ONLINE_QUANT:
                token_vec.append(
                    dis_tokens_buf.reshape(-1,
                                           args.N // 2)[off * 2:off * 2 + ntk])
                scale_vec.append(dis_scales_buf[off:off + ntk])
            else:
                token_vec.append(dis_tokens_buf[off:off + ntk])
            off += MAX_NUM_TOKENS
        dispatched_tokens, s = torch.concat(token_vec), torch.concat(
            scale_vec) if ONLINE_QUANT else None
        #####################
        # 2. compute, e.g. GroupGEMM(dispatched_tokens, disaptched_scales, ...) ...
        if ONLINE_QUANT:
            dispatched_tokens = dequant_fp8_bf16(dispatched_tokens, s)
        #####################
        # 3. Combine
        combine_splits_cumsum = splits_to_cumsum(combine_send_splits)
        all_to_all_kernel[grid](
            dispatched_tokens,
            recv_buf,
            send_buf,  # we re-use the recv/send buffer in combine mode:
            scale_recv_buf,
            scale_send_buf,  # the recv buffer is used to store the input
            split_recv_buf,
            split_send_buf,  # and the send buffer is used to store the output
            signal_buf,
            combine_splits_cumsum,
            recv_offset=
            combine_offset,  # get the base offset of the received tokens in combine phase from dispatch
            rank=RANK,
            call_count=round * 2 +
            1,  # as we call the kernel twice in each iteration
            act_pos=act_pos,  # flip for double buffer
            MODE=1,  # combine mode
            **kwargs,
        )
        combined_tokens = send_buf[:scattered_input_cur_rank.shape[0]]
        #####################
        # check the correctness of dispatch-combine
        combine_ref = (dequant_fp8_bf16(
            *quant_bf16_fp8(scattered_input_cur_rank))
                       if ONLINE_QUANT else scattered_input_cur_rank)
        try:
            torch.testing.assert_close(combined_tokens.float(),
                                       combine_ref.float(),
                                       rtol=1e-5,
                                       atol=1e-5)
            dist_print(f"✅ Round-{round} combine check passed!")
        except Exception as e:
            dist_print(f"❌ Round-{round} combine check failed! {e}")
            raise e

    nvshmem_free_tensor_sync(send_buf)
    nvshmem_free_tensor_sync(recv_buf)
    nvshmem_free_tensor_sync(scale_send_buf)
    nvshmem_free_tensor_sync(scale_recv_buf)
    nvshmem_free_tensor_sync(split_send_buf)
    nvshmem_free_tensor_sync(split_recv_buf)
    nvshmem_free_tensor_sync(signal_buf)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()

# To run this tutorial
# source ./scripts/sentenv.sh
# bash ./scripts/launch.sh tutorials/04-deepseek-infer-all2all.py
