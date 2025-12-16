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
def _quant_kernel(out, out_scale, t, m, N: tl.constexpr, FP8_GSIZE: tl.constexpr = 128, BM: tl.constexpr = 16):
    pid = tl.program_id(0)
    FP8_MAX_INV = tl.constexpr(1 / 448.)
    NUM_GROUPS: tl.constexpr = N // FP8_GSIZE
    UNROLL_FACTOR: tl.constexpr = 4
    off_m = pid * BM + tl.arange(0, BM)
    off_n = tl.arange(0, UNROLL_FACTOR * FP8_GSIZE)
    input_ptrs = t + off_m[:, None] * N + off_n[None, :]
    out_ptrs = tl.cast(out, tl.pointer_type(tl.float8e4nv)) + off_m[:, None] * N + off_n[None, :]
    out_scale_ptrs = out_scale + off_m[:, None] * NUM_GROUPS + tl.arange(0, UNROLL_FACTOR)[None, :]
    for i in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
        group_mask = (off_m[:, None] < m) & (off_n[None, :] < N - i * FP8_GSIZE)
        scale_mask = (off_m[:, None] < m) & (tl.arange(0, UNROLL_FACTOR)[None, :] < NUM_GROUPS - i)
        group = tl.reshape(tl.load(input_ptrs, group_mask, 0.), (BM * UNROLL_FACTOR, FP8_GSIZE))
        scale = tl.max(tl.abs(group), 1, keep_dims=True).to(tl.float32) * FP8_MAX_INV
        quant = (group.to(tl.float32) / scale).to(tl.float8e4nv)
        tl.store(out_ptrs, tl.reshape(quant, (BM, UNROLL_FACTOR * FP8_GSIZE)), mask=group_mask)
        tl.store(out_scale_ptrs, tl.reshape(scale, (BM, UNROLL_FACTOR)), mask=scale_mask)
        input_ptrs += UNROLL_FACTOR * FP8_GSIZE
        out_ptrs += UNROLL_FACTOR * FP8_GSIZE
        out_scale_ptrs += UNROLL_FACTOR


@triton.jit
def _dequant_kernel(out, input, scales, m, N: tl.constexpr, FP8_GSIZE: tl.constexpr = 128, BM: tl.constexpr = 16):
    pid = tl.program_id(0)
    NUM_GROUPS: tl.constexpr = N // FP8_GSIZE
    UNROLL_FACTOR: tl.constexpr = 4
    off_m = pid * BM + tl.arange(0, BM)
    off_n = tl.arange(0, UNROLL_FACTOR * FP8_GSIZE)
    input_ptrs = tl.cast(input, tl.pointer_type(tl.float8e4nv)) + off_m[:, None] * N + off_n[None, :]
    input_scale_ptrs = scales + off_m[:, None] * NUM_GROUPS + tl.arange(0, UNROLL_FACTOR)[None, :]
    out_ptrs = out + off_m[:, None] * N + off_n[None, :]
    for i in tl.static_range(0, NUM_GROUPS, UNROLL_FACTOR):
        group_mask = (off_m[:, None] < m) & (off_n[None, :] < N - i * FP8_GSIZE)
        scale_mask = (off_m[:, None] < m) & (tl.arange(0, UNROLL_FACTOR)[None, :] < NUM_GROUPS - i)
        group = tl.reshape(tl.load(input_ptrs, group_mask, 0.), (BM * UNROLL_FACTOR, FP8_GSIZE))
        scale = tl.reshape(tl.load(input_scale_ptrs, scale_mask, 0.), (BM * UNROLL_FACTOR, 1))
        deq = (group.to(tl.float32) * scale).to(tl.bfloat16)
        tl.store(out_ptrs, tl.reshape(deq, (BM, UNROLL_FACTOR * FP8_GSIZE)), mask=group_mask)
        input_ptrs += UNROLL_FACTOR * FP8_GSIZE
        input_scale_ptrs += UNROLL_FACTOR
        out_ptrs += UNROLL_FACTOR * FP8_GSIZE


def quant_bf16_fp8(tensor: torch.Tensor, gsize: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    assert tensor.dtype == torch.bfloat16, f"dtype = {tensor.dtype}"
    m, N = tensor.shape
    grid = (triton.cdiv(m, 16), )
    out = torch.empty((m, N), dtype=torch.float8_e4m3fn, device=tensor.device)
    out_scale = torch.empty(m, N // gsize, dtype=torch.float32, device=tensor.device)
    _quant_kernel[grid](out, out_scale, tensor, m, N)
    out = out.view(torch.float8_e4m3fn)
    return out, out_scale


def dequant_fp8_bf16(q_tensor: torch.Tensor, scales: torch.Tensor):
    if scales is None:
        return q_tensor

    assert q_tensor.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float32
    ori_shape = q_tensor.shape
    q_tensor = q_tensor.reshape(-1, q_tensor.shape[-1])
    m, N = q_tensor.shape
    grid = (triton.cdiv(m, 16), )
    out = torch.empty([m, N], dtype=torch.bfloat16, device=q_tensor.device)
    _dequant_kernel[grid](out, q_tensor, scales, m, N)
    out = out.reshape(ori_shape[:-1] + (N, ))
    return out


def torch_ll_dispatch(EP_GROUP, input: torch.Tensor, exp_indices: torch.Tensor, num_experts: int, quant_group_size: int,
                      online_quant_fp8: bool):
    num_tokens, hidden = input.shape
    topk = exp_indices.shape[1]
    assert hidden % quant_group_size == 0
    NUM_GROUPS = hidden // quant_group_size
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter and gather idx
    _, index_choosed_experts = exp_indices.flatten().sort(stable=True)
    gather_idx_cur_rank = index_choosed_experts.to(torch.int32) // topk
    scattered_input = torch.empty((input.size(0) * topk, input.size(1)), dtype=input.dtype,
                                  device=input.device).copy_(torch.index_select(input, dim=0,
                                                                                index=gather_idx_cur_rank))

    if online_quant_fp8:
        send_tensor, send_scale = quant_bf16_fp8(scattered_input)
    else:
        send_tensor, send_scale = scattered_input, None

    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    num_recv_tokens = a2a_splits_cpu.sum()
    dis_tokens: torch.Tensor = torch.empty([num_recv_tokens, send_tensor.size(1)], dtype=send_tensor.dtype,
                                           device="cuda")
    dis_scales: torch.Tensor = torch.empty([num_recv_tokens, NUM_GROUPS], dtype=torch.float32, device="cuda")

    # 1. Dispatch
    world_size = EP_GROUP.size()
    ori_dtype = None
    if send_tensor.dtype.itemsize == 1:
        ori_dtype = dis_tokens.dtype
        dis_tokens = dis_tokens.view(torch.int8)
        send_tensor = send_tensor.view(torch.int8)

    torch.distributed.all_to_all_single(
        output=dis_tokens,
        input=send_tensor,
        output_split_sizes=a2a_splits_cpu.reshape(world_size, -1).sum(dim=-1).tolist(),
        input_split_sizes=splits_cpu_cur_rank.reshape(world_size, -1).sum(-1).tolist(),
        group=EP_GROUP,
    )

    if online_quant_fp8:
        torch.distributed.all_to_all_single(
            output=dis_scales,
            input=send_scale,
            output_split_sizes=a2a_splits_cpu.reshape(world_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=splits_cpu_cur_rank.reshape(world_size, -1).sum(-1).tolist(),
            group=EP_GROUP,
        )

    # postprocess: sort by (expert, rank)
    num_experts_per_rank = num_experts // world_size
    assert num_experts % world_size == 0
    a2a_expert_input_list = torch.split(dis_tokens, a2a_splits_cpu.tolist())

    a2a_dispatch_scale_list = torch.split(dis_scales, a2a_splits_cpu.tolist())

    permute_a2a_expert_input_list = list()
    permute_a2a_dispatch_scale_list = list()
    for idx in range(num_experts_per_rank):
        for idy in range(world_size):
            permute_a2a_expert_input_list.append(a2a_expert_input_list[idy * num_experts_per_rank + idx])
            permute_a2a_dispatch_scale_list.append(a2a_dispatch_scale_list[idy * num_experts_per_rank + idx])

    permute_a2a_expert_input = torch.cat(permute_a2a_expert_input_list, dim=0)
    permute_a2a_expert_scale = torch.cat(permute_a2a_dispatch_scale_list, dim=0)

    # cat/a2a_single not implemented for fp8
    if ori_dtype is not None:
        permute_a2a_expert_input = permute_a2a_expert_input.view(ori_dtype)

    if not online_quant_fp8:
        permute_a2a_expert_scale = None
    return permute_a2a_expert_input, permute_a2a_expert_scale


def torch_ll_combine(EP_GROUP, input, exp_indices, weight, num_experts):
    topk = exp_indices.size(1)
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)
    # drop token logic :only need the splits information for the non-dropped tokenes
    splits_gpu_cur_rank = splits_gpu_cur_rank[:num_experts]
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter idx

    _, new_index = exp_indices.flatten().sort(stable=True)
    new_index = new_index.to(torch.int32)
    # calculate the gather idx accordingly
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    ep_size = EP_GROUP.size()
    num_experts_per_rank = num_experts // ep_size
    a2a_splits_cpu = a2a_splits.cpu()
    permute_a2a_splits_cpu = (a2a_splits_cpu.reshape(-1, num_experts_per_rank).permute(-1, -2).flatten())
    count_before_drop = exp_indices.numel()
    count_after_drop = splits_cpu_cur_rank.sum()

    permute_a2a_expert_output_list = torch.split(input, permute_a2a_splits_cpu.tolist())
    a2a_expert_output_list = list()
    for idy in range(ep_size):
        for idx in range(num_experts_per_rank):
            a2a_expert_output_list.append(permute_a2a_expert_output_list[idx * ep_size + idy])
    a2a_expert_output = torch.cat(a2a_expert_output_list, dim=0)
    all2all_out = torch.empty([splits_cpu_cur_rank.sum(), input.shape[-1]], device=input.device, dtype=input.dtype)
    torch.distributed.all_to_all_single(
        output=all2all_out,
        input=a2a_expert_output,
        output_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(dim=-1).tolist(),
        input_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
        group=EP_GROUP,
    )
    all2all_out_padded = torch.zeros(
        (count_before_drop, all2all_out.size(1)),
        device=all2all_out.device,
        dtype=all2all_out.dtype,
    )
    all2all_out_padded.data[:count_after_drop] = all2all_out
    gather_output = torch.zeros_like(all2all_out_padded)
    gather_output[new_index] = all2all_out_padded
    assert len(weight.shape) == 2
    assert weight.shape[0] == gather_output.size(0) // topk
    assert weight.shape[1] == topk
    weight = weight.unsqueeze(-1)
    topk_reduce = (gather_output.view((gather_output.size(0) // topk, topk, gather_output.size(-1))) * weight).sum(1)
    topk_reduce = topk_reduce.to(input.dtype)
    return topk_reduce
