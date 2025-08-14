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
import nvshmem.core
import torch
import torch.distributed
import triton
import triton.language as tl
from triton_dist.utils import finalize_distributed, initialize_distributed, perf_func, get_torch_prof_ctx
from functools import partial

import argparse
import random
import os
import numpy as np

from triton_dist.layers.nvidia import EPAll2AllLayer

EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def _check(out: torch.Tensor, ref: torch.Tensor, msg: str = "Triton"):
    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        print(f"✅ RANK[{RANK}] check {msg} passed")
    except Exception as e:
        print(f"❌ RANK[{RANK}] check {msg} failed")
        raise e


def generate_random_exp_indices(token_num, total_num_experts, topk, drop_ratio=0.0):
    exp_indices = []
    exp_list = list(range(total_num_experts))

    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        for i, _ in enumerate(top_selected):
            if random.uniform(0, 1) < drop_ratio:
                # current topk choice will be dropped
                top_selected[i] = total_num_experts
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.zeros(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    # out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


def sort_by_vectors(x):
    assert len(x.shape) <= 2
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    M, K = x.shape
    current_order = torch.arange(M, device=x.device)
    for k in reversed(range(K)):
        current_col = x[current_order, k]
        _, sorted_indices = torch.sort(current_col, stable=True)
        current_order = current_order[sorted_indices]
    sorted_x = x[current_order]
    return sorted_x


def calc_gather_index(
    scatter_index: torch.Tensor,
    row_start: int,
    row_end: int,
    BLOCK_SIZE: int = 1024,
):

    @triton.jit
    def _kernel(
        scatter_index: torch.Tensor,
        gather_index: torch.Tensor,
        topk_index: torch.Tensor,
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
        topk_idx = offset % topk
        token_idx_mask = (scatter_idx >= row_start) & (scatter_idx < row_end)
        tl.store(gather_index + scatter_idx - row_start, token_idx, mask=token_idx_mask)
        tl.store(topk_index + scatter_idx - row_start, topk_idx, mask=token_idx_mask)

    ntokens, topk = scatter_index.shape
    gather_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    topk_index = torch.zeros(row_end - row_start, dtype=torch.int32, device=scatter_index.device)
    grid = lambda META: (triton.cdiv(ntokens * topk, META["BLOCK_SIZE"]), )
    _kernel[grid](
        scatter_index,
        gather_index,
        topk_index,
        ntokens,
        topk,
        row_start,
        row_end,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32,
    )
    return gather_index, topk_index


def calc_scatter_index_stable(chosen_experts: torch.Tensor):
    return (chosen_experts.flatten().argsort(stable=True).argsort().int().view(chosen_experts.shape))


def calc_full_scatter_indices(exp_indices):
    n_token_cur_ep_rank = exp_indices.size(0)
    input_len = torch.tensor([n_token_cur_ep_rank], dtype=torch.int32, device=exp_indices.device)
    ag_input_len = torch.zeros(EP_GROUP.size(), dtype=torch.int32, device=exp_indices.device)
    torch.distributed.all_gather_into_tensor(ag_input_len, input_len, group=EP_GROUP)
    ag_input_len_cpu = ag_input_len.cpu()
    ag_input_len_list = ag_input_len_cpu.tolist()
    padded_indices = torch.empty([args.M, args.topk], dtype=torch.int32, device=exp_indices.device)
    padded_indices[
        :exp_indices.size(0),
    ] = exp_indices
    ag_padded_indices = [torch.empty_like(padded_indices) for _ in range(EP_GROUP.size())]
    # concat the exp_indices from all the rank
    torch.distributed.all_gather(ag_padded_indices, padded_indices, group=EP_GROUP)
    ag_indices = torch.concat([t[:ag_input_len_list[i], :] for i, t in enumerate(ag_padded_indices)])
    ag_scatter_idx = calc_scatter_index_stable(ag_indices)
    return ag_scatter_idx


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
    "float32": torch.float32,
}


def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=5, profile="full")
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=7168)
    parser.add_argument("-G", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", default=3, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=5, type=int)
    parser.add_argument("--bench_iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--drop_ratio", default=0.1, type=float, help="the token drop ratio")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--weight_dtype", default="float32", help="weight type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--has_weight", action="store_true")
    parser.add_argument("--with-scatter-indices", action="store_true")
    return parser.parse_args()


def calc_gather_index_stable(choosed_experts: torch.Tensor, topk, ntokens):
    _, index_choosed_experts = choosed_experts.flatten().sort(stable=True)
    gather_index = index_choosed_experts.to(torch.int32) // topk
    topk_index = torch.arange(0, topk, dtype=torch.int32, device="cuda").repeat(ntokens)[index_choosed_experts]
    return gather_index, topk_index


def torch_forward_preprocess(input, weight, exp_indices, num_experts):
    token_num, topk = exp_indices.size(0), exp_indices.size(1)
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)
    # drop token logic :only need the splits information for the non-dropped tokenes
    splits_gpu_cur_rank = splits_gpu_cur_rank[:num_experts]
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    count_after_drop = torch.sum(splits_cpu_cur_rank)
    count_origin = exp_indices.numel()
    # calculate the scatter idx
    # scatter_idx_cur_rank = calc_scatter_index(exp_indices, splits_gpu_cur_rank)
    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    # calculate the gather idx accordingly
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, token_num * topk)
    # use torch native scatter forward(will not be included in the e2e time measurement)
    assert count_origin == input.size(0) * topk
    scattered_input = torch.empty(input.size(0) * topk, input.size(1), dtype=input.dtype, device=input.device)
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    if weight is not None:
        scattered_weight = torch.empty_like(weight.flatten())
        scattered_weight[scatter_idx_cur_rank.flatten()] = weight.flatten()
        scattered_weight = scattered_weight[:count_after_drop]
    else:
        scattered_weight = None
    # print(f"exp_indices = {exp_indices}, weight = {weight}, scatter_idx_cur_rank = {scatter_idx_cur_rank}, scattered_weight = {scattered_weight}")
    # drop token logic: drop the token here
    scattered_input = scattered_input[:count_after_drop]
    return scattered_input, scattered_weight


def torch_forward_comm(scattered_input, scattered_weight, a2a_splits_cpu, splits_cpu_cur_rank):
    ep_size = EP_GROUP.size()
    input_splits = splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist()
    output_splits = a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist()
    a2a_dispatch_output = torch.empty([a2a_splits_cpu.sum(), input.size(1)], dtype=input.dtype, device=input.device)

    torch.distributed.all_to_all_single(
        output=a2a_dispatch_output,
        input=scattered_input,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=EP_GROUP,
    )

    if scattered_weight is not None:
        a2a_dispatch_weight = torch.empty([a2a_splits_cpu.sum()], dtype=scattered_weight.dtype,
                                          device=scattered_weight.device)
        torch.distributed.all_to_all_single(
            output=a2a_dispatch_weight,
            input=scattered_weight,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=EP_GROUP,
        )
    else:
        a2a_dispatch_weight = None

    return a2a_dispatch_output, a2a_dispatch_weight


def torch_forward_single(input, weight, exp_indices, num_experts):
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)
    # drop token logic :only need the splits information for the non-dropped tokenes
    splits_gpu_cur_rank = splits_gpu_cur_rank[:num_experts]
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    scattered_input, scattered_weight = torch_forward_preprocess(input, weight, exp_indices, num_experts)
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    ep_size = EP_GROUP.size()

    a2a_dispatch_output, a2a_dispatch_weight = torch_forward_comm(scattered_input, scattered_weight, a2a_splits_cpu,
                                                                  splits_cpu_cur_rank)

    # postprocess: sort by (expert, rank)
    num_experts_per_rank = num_experts // ep_size
    assert num_experts % ep_size == 0
    a2a_expert_input_list = torch.split(a2a_dispatch_output, a2a_splits_cpu.tolist())

    if a2a_dispatch_weight is not None:
        a2a_dispatch_weight_list = torch.split(a2a_dispatch_weight, a2a_splits_cpu.tolist())

    permute_a2a_expert_input_list = list()
    permute_a2a_expert_weight_list = list()
    for idx in range(num_experts_per_rank):
        for idy in range(ep_size):
            permute_a2a_expert_input_list.append(a2a_expert_input_list[idy * num_experts_per_rank + idx])
            if a2a_dispatch_weight is not None:
                permute_a2a_expert_weight_list.append(a2a_dispatch_weight_list[idy * num_experts_per_rank + idx])

    permute_a2a_expert_input = torch.cat(permute_a2a_expert_input_list, dim=0)
    if a2a_dispatch_weight is not None:
        permute_a2a_expert_weight = torch.cat(permute_a2a_expert_weight_list, dim=0)
    else:
        permute_a2a_expert_weight = None
    return permute_a2a_expert_input, permute_a2a_expert_weight


def torch_backward_single(input, exp_indices, num_experts):
    topk = exp_indices.size(1)
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)
    # drop token logic :only need the splits information for the non-dropped tokenes
    splits_gpu_cur_rank = splits_gpu_cur_rank[:num_experts]
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter idx

    gather_index, topk_index = calc_gather_index_stable(exp_indices, topk, exp_indices.size(0))
    new_index = topk * gather_index + topk_index
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
    # if args.drop_ratio > 0:
    #     print(f"Drop token enabled {count_before_drop} -> {count_after_drop}")
    # if args.drop_ratio == 0:
    #     assert count_before_drop == count_after_drop

    permute_a2a_expert_output_list = torch.split(input, permute_a2a_splits_cpu.tolist())
    # print(f"Len: {len(permute_a2a_expert_output_list)}")
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
    topk_reduce = gather_output.view((gather_output.size(0) // topk, topk, gather_output.size(-1))).sum(1)

    return topk_reduce


def perf_torch(args, input, weight, exp_indices):
    # prepare the indexes
    token_num, _ = input.shape
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
    # calculate the scatter idx
    scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
    # calculate the gather idx accordingly
    gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, token_num * args.topk)
    # use torch native scatter forward(will not be included in the e2e time measurement)
    scattered_input = torch.empty(input.size(0) * args.topk, input.size(1), dtype=input.dtype, device=input.device)
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    ep_size = EP_GROUP.size()
    a2a_dispatch_output = torch.empty([a2a_splits_cpu.sum(), input.size(1)], dtype=input.dtype, device=input.device)
    if weight is not None:
        scattered_weight = torch.empty_like(weight.flatten())
        scattered_weight[scatter_idx_cur_rank.flatten()] = weight.flatten()
        a2a_dispatch_weight = torch.empty([a2a_splits_cpu.sum()], dtype=weight.dtype, device=weight.device)
    else:
        a2a_dispatch_weight = None
    torch.cuda.synchronize()

    def fwd():
        torch.distributed.all_to_all_single(
            output=a2a_dispatch_output,
            input=scattered_input,
            output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
            group=EP_GROUP,
        )

        if weight is not None:
            torch.distributed.all_to_all_single(
                output=a2a_dispatch_weight,
                input=scattered_weight,
                output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
                input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
                group=EP_GROUP,
            )

    # warmup
    for _ in range(10):
        fwd()
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    # bench
    st.record()
    for _ in range(args.bench_iters):
        fwd()
    ed.record()
    torch.cuda.synchronize()
    avg_time = st.elapsed_time(ed) / args.bench_iters
    return a2a_dispatch_output, a2a_dispatch_weight, avg_time


def straggler(rank):
    clock_rate = torch.cuda.clock_rate() * 1e6
    cycles = random.randint(0, clock_rate * 0.0001) * (rank + 1)
    torch.cuda._sleep(cycles)


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    assert (args.G % WORLD_SIZE == 0), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"
    # if LOCAL_RANK == 0:
    #     print(f"LOCAL_RANK = {LOCAL_RANK}")
    #     os.environ["MLIR_ENABLE_DUMP"]="1"
    experts_per_rank = args.G // WORLD_SIZE
    input_dtype = DTYPE_MAP[args.dtype]
    weight_dtype = DTYPE_MAP[args.weight_dtype]
    triton_a2a_op = EPAll2AllLayer(EP_GROUP, args.M, args.N, args.topk, RANK, args.G, LOCAL_WORLD_SIZE, WORLD_SIZE,
                                   input_dtype, weight_dtype=weight_dtype)

    def _make_data(token_num):
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk, args.drop_ratio)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = (torch.rand(token_num, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
        if args.has_weight:
            weight = torch.randn(token_num, args.topk, dtype=torch.float32).to("cuda")
            weight = torch.nn.functional.softmax(weight, dim=1).to(weight_dtype)
        else:
            weight = None
        if args.with_scatter_indices:
            full_scatter_indices = calc_full_scatter_indices(exp_indices)
        else:
            full_scatter_indices = None
        return input, weight, exp_indices, full_scatter_indices

    if args.check:
        for n in range(args.iters):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            input_list = [_make_data(random.randint(1, args.M)) for _ in range(args.verify_iters)]
            combine_out_list, dispatch_out_list, torch_input_list, torch_dispatch_out_list, torch_combine_out_list = [], [], [], [], []

            # torch impl
            for input, weight, exp_indices, full_scatter_indices in input_list:
                # ref_dispatch_out, ref_weight, ref_scale, ref_time = perf_torch(args, input, weight, scale_tensor, exp_indices)
                ref_dispatch_out, ref_dispatch_weight = torch_forward_single(input, weight, exp_indices, args.G)
                if args.has_weight:
                    torch_combine_input = (ref_dispatch_weight.reshape(-1, 1) * ref_dispatch_out).to(input_dtype)
                else:
                    torch_combine_input = ref_dispatch_out
                ref_combine_out = torch_backward_single(torch_combine_input, exp_indices, args.G)
                torch_dispatch_out_list.append((ref_dispatch_out, ref_dispatch_weight))
                torch_combine_out_list.append(ref_combine_out)

            # dist triton impl
            for input, weight, exp_indices, full_scatter_indices in input_list:
                straggler(RANK)
                dispatch_out, dispatch_weight, ep_a2a_layout_desc = triton_a2a_op.dispatch(
                    input, exp_indices, weight=weight, full_scatter_indices=full_scatter_indices)
                dispatch_out_list.append((dispatch_out, dispatch_weight))
                # torch.cuda.synchronize()
                if args.has_weight:
                    triton_combine_input = (dispatch_weight.reshape(-1, 1) * dispatch_out).to(input_dtype)
                else:
                    triton_combine_input = dispatch_out
                combined_out = triton_a2a_op.combine(triton_combine_input, ep_a2a_layout_desc)
                combine_out_list.append(combined_out)

            shape_only = False
            # torch.cuda.synchronize()
            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_dispatch_out_list, dispatch_out_list)):
                torch_dispatch_out, torch_dispatch_weight = torch_out
                triton_dispatch_out, triton_dispatch_weight = dist_out

                if RANK == 0:
                    print(f"dispatch: shape = {torch_dispatch_out.shape}, {triton_dispatch_out.shape}")
                    # print(f"dispatch weight: triton_dispatch_weight = {triton_dispatch_weight.sum()}, torch_dispatch_weight = {torch_dispatch_weight.sum()}")
                try:
                    if not shape_only:
                        if not args.with_scatter_indices:
                            sorted_triton_dispatch_out = sort_by_vectors(triton_dispatch_out)
                            sorted_torch_dispatch_out = sort_by_vectors(torch_dispatch_out)
                        else:
                            sorted_triton_dispatch_out = triton_dispatch_out
                            sorted_torch_dispatch_out = torch_dispatch_out
                        # print(f"sorted_triton_dispatch_weight = {sorted_triton_dispatch_weight}, sorted_torch_dispatch_weight = {sorted_torch_dispatch_weight}")

                        torch.testing.assert_close(sorted_triton_dispatch_out, sorted_torch_dispatch_out, atol=0,
                                                   rtol=0)
                        if args.has_weight:
                            sorted_triton_dispatch_weight = sort_by_vectors(triton_dispatch_weight)
                            sorted_torch_dispatch_weight = sort_by_vectors(torch_dispatch_weight)
                            torch.testing.assert_close(sorted_triton_dispatch_weight, sorted_torch_dispatch_weight,
                                                       atol=0, rtol=0)
                        else:
                            assert torch_dispatch_weight is None and triton_dispatch_weight is None
                    else:
                        assert triton_dispatch_out.shape == torch_dispatch_out.shape, f"triton_dispatch_out.shape = {triton_dispatch_out.shape}, torch_dispatch_out.shape = {torch_dispatch_out.shape}"
                except Exception as e:
                    raise e
                # torch.distributed.barrier()

            for idx, (torch_combine_out, triton_combine_out) in enumerate(zip(torch_combine_out_list,
                                                                              combine_out_list)):
                if RANK == 0:
                    print(
                        f"combine: shape = {torch_combine_out.shape} {torch_combine_out.dtype}, {triton_combine_out.shape} {triton_combine_out.dtype}"
                    )
                try:
                    if not shape_only:
                        torch.testing.assert_close(torch_combine_out, triton_combine_out, atol=1e-2, rtol=1e-2)
                    else:
                        assert torch_combine_out.shape == triton_combine_out.shape, f"torch_combine_out.shape = {torch_combine_out.shape}, triton_combine_out.shape = {triton_combine_out.shape}"
                except Exception as e:
                    raise e

        print(f"RANK[{RANK}]: pass.")
        triton_a2a_op.finalize()
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group(EP_GROUP)
        exit(0)

    for rid in range(args.rounds):
        # random simulate token received from dataloader
        L = args.M // 2 if not args.profile else args.M

        token_num = random.randint(L, args.M)

        print(f"Rank-{RANK}: Received {token_num} tokens")

        input, weight, exp_indices, full_scatter_indices = _make_data(token_num)
        ctx = get_torch_prof_ctx(args.profile)
        with ctx:
            (ref_dispatch_out,
             ref_dispatch_weight), _ = perf_func(partial(torch_forward_single, input, weight, exp_indices, args.G),
                                                 iters=100, warmup_iters=20)
            ref_combine_out, _ = perf_func(partial(torch_backward_single, ref_dispatch_out, exp_indices, args.G),
                                           iters=100, warmup_iters=20)

            (triton_dispatch_out, triton_dispatch_weight, ep_a2a_layout_desc), triton_perf = perf_func(
                partial(triton_a2a_op.dispatch, input, exp_indices, weight, full_scatter_indices), iters=100,
                warmup_iters=20)
            combined_out, triton_combine_perf = perf_func(
                partial(triton_a2a_op.combine, triton_dispatch_out, ep_a2a_layout_desc), iters=100, warmup_iters=20)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        torch.distributed.barrier()  # wait all rank dispatch
        if not args.with_scatter_indices:
            sorted_triton_dispatch_out = sort_by_vectors(triton_dispatch_out)
            sorted_ref_dispatch_out = sort_by_vectors(ref_dispatch_out)
        else:
            sorted_triton_dispatch_out = triton_dispatch_out
            sorted_ref_dispatch_out = ref_dispatch_out

        torch.cuda.synchronize()
        torch.distributed.barrier()

        if args.profile:
            run_id = os.environ["TORCHELASTIC_RUN_ID"]
            prof_dir = f"prof/{run_id}"
            os.makedirs(prof_dir, exist_ok=True)
            ctx.export_chrome_trace(f"{prof_dir}/trace_rank{EP_GROUP.rank()}.json.gz")

        _check(sorted_triton_dispatch_out, sorted_ref_dispatch_out)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.testing.assert_close(ref_combine_out, combined_out, rtol=1e-2, atol=1e-2)

        print(f"RANK {RANK}: triton dispatch perf = {triton_perf}ms, triton_combine_perf = {triton_combine_perf}ms")

    triton_a2a_op.finalize()
    finalize_distributed()
