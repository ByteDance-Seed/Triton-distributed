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
import datetime
from functools import partial

import argparse
import random
import os

from contextlib import nullcontext
from flash_comm.ep import EPKernels
from flash_comm.ep.ep_kernels import EPCommLayoutDesc

EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def init_seed(seed=0):
    import numpy as np
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)


def bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    assert a.is_contiguous() and b.is_contiguous()
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        return False
    a8 = a.view(torch.uint8)
    b8 = b.view(torch.uint8)
    return torch.equal(a8, b8)


def get_torch_prof_ctx(do_prof: bool):
    ctx = (torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
    ) if do_prof else nullcontext())
    return ctx


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(warmup_iters):
        _ = func()
    torch.cuda.synchronize()
    # Benchmark
    start_event.record()
    for _ in range(iters):
        output = func()
    stop_event.record()
    torch.cuda.synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


torch.set_printoptions(precision=4)


def _check(out: torch.Tensor, ref: torch.Tensor, msg: str = "FlashComm"):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=5120)
    parser.add_argument("-G", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", default=3000, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=20, type=int)
    parser.add_argument("--bench_iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--drop_ratio", default=0.1, type=float, help="the token drop ratio")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--num_sm", default=16, type=int, help="number of sm")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--weight_dtype", default="float32", help="weight type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true", help="profile the kernel")
    parser.add_argument("--check", action="store_true", help="check the result")
    parser.add_argument("--enable-local-combine", action="store_true", help="enable local combine")
    parser.add_argument("--num_worst_tokens", default=-1, type=int, help="num worst tokens")
    parser.add_argument("--disable-weight-combine", action="store_true", help="disable weight combine")
    parser.add_argument("--disable-weight-dispatch", action="store_true", help="disable weight dispatch")
    return parser.parse_args()


def torch_forward_preprocess(input, weight, exp_indices, num_experts):
    _, topk = exp_indices.size(0), exp_indices.size(1)
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
    _, index_choosed_experts = exp_indices.flatten().sort(stable=True)
    gather_idx_cur_rank = index_choosed_experts.to(torch.int32) // topk
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


def torch_forward_comm(input, scattered_input, scattered_weight, a2a_splits_cpu, splits_cpu_cur_rank):
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

    a2a_dispatch_output, a2a_dispatch_weight = torch_forward_comm(input, scattered_input, scattered_weight,
                                                                  a2a_splits_cpu, splits_cpu_cur_rank)

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


def torch_backward_single(input, exp_indices, num_experts, enable_local_combine=False):
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

    gather_output_view = gather_output.view((gather_output.size(0) // topk, topk, gather_output.size(-1)))
    if enable_local_combine:
        num_experts_per_rank = num_experts // EP_GROUP.size()
        group_ids = exp_indices // num_experts_per_rank

        # Calculate target indices: map each position to the index of the first occurrence of its group
        # group_ids: (N, topk) -> matches: (N, topk, topk)
        matches = group_ids.unsqueeze(2) == group_ids.unsqueeze(1)
        # argmax returns the first index where match is True, indicating the first occurrence of the group
        target_indices = matches.to(torch.int32).argmax(dim=2)

        # Accumulate into the first occurrence position (fp32)
        # Other positions remain 0
        accum_buffer = torch.zeros_like(gather_output_view, dtype=torch.float32)

        target_indices_expanded = target_indices.unsqueeze(-1).expand(-1, -1, gather_output_view.size(-1))
        gather_output_fp32 = gather_output_view.to(torch.float32)

        accum_buffer.scatter_add_(1, target_indices_expanded, gather_output_fp32)

        # Cast back to original dtype and sum along k dimension
        # This sums the group accumulators (and zeros) in the order of their first appearance

        # Cast back to original dtype to align the local reduce.
        accum_buffer = accum_buffer.to(input.dtype).to(torch.float32)
        topk_reduce = torch.zeros((accum_buffer.shape[0], accum_buffer.shape[2]), dtype=torch.float32,
                                  device=accum_buffer.device)
        for i in range(accum_buffer.size(1)):
            topk_reduce[:, :] += accum_buffer[:, i, :]
        topk_reduce = topk_reduce.to(input.dtype)
    else:
        # topk_reduce = gather_output_view.to(torch.float32).sum(1).to(input.dtype)
        topk_reduce = torch.zeros((gather_output_view.shape[0], gather_output_view.shape[2]), dtype=torch.float32,
                                  device=gather_output_view.device)
        for i in range(gather_output_view.size(1)):
            topk_reduce[:, :] += gather_output_view[:, i, :]
        topk_reduce = topk_reduce.to(input.dtype)

    return topk_reduce


def straggler(rank):
    clock_rate = torch.cuda.clock_rate() * 1e6
    cycles = random.randint(0, clock_rate * 0.0001) * (rank + 1)
    torch.cuda._sleep(cycles)


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    assert args.num_sm > 0
    # assert args.enable_local_combine == True

    init_seed(RANK)

    # use all ranks as tp group
    EP_GROUP: torch.distributed.ProcessGroup = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)),
                                                                           backend="nccl")
    torch.distributed.barrier(group=EP_GROUP)
    torch.cuda.synchronize()

    assert (args.G % WORLD_SIZE == 0), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"
    experts_per_rank = args.G // WORLD_SIZE
    input_dtype = DTYPE_MAP[args.dtype]
    weight_dtype = DTYPE_MAP[args.weight_dtype]
    assert weight_dtype == torch.float32, "Only float32 weight type is supported"
    assert input_dtype == torch.bfloat16, "Only bfloat16 input type is supported"

    def _make_data(token_num):
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk, args.drop_ratio)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = (torch.rand(token_num, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
        if args.disable_weight_dispatch:
            weight = None
        else:
            weight = torch.randn(token_num, args.topk, dtype=torch.float32).to("cuda")
            weight = torch.nn.functional.softmax(weight, dim=1).to(weight_dtype)
        return input, weight, exp_indices

    ep_kernels = EPKernels(max_m=args.M, hidden=args.N, topk=args.topk, num_experts=args.G, local_world_size=WORLD_SIZE,
                           ep_group=EP_GROUP, num_sm=args.num_sm, num_worst_tokens=args.num_worst_tokens)

    def _run_dispatch(input, weight, exp_indices, copy_out=False):
        token_within_expert_offset, expert_counts = ep_kernels.compute_stable_local_token_within_expert_offset_and_expert_counts(
            exp_indices)
        layout_desc = EPCommLayoutDesc(token_within_expert_offset=token_within_expert_offset,
                                       expert_counts=expert_counts)
        dispatch_out, dispatch_weights, layout_desc = ep_kernels.dispatch(input, exp_indices, weight, layout_desc)
        if copy_out:
            postprocess_dispatch_out = dispatch_out.clone()
        else:
            postprocess_dispatch_out = dispatch_out
        final_dispatch_out, final_dispatch_weights, layout_desc = ep_kernels.dispatch_postprocess(
            postprocess_dispatch_out, dispatch_weights, layout_desc)
        return final_dispatch_out, final_dispatch_weights, layout_desc

    def _prepare_combine_input(input):
        combine_input_buf = ep_kernels.get_combine_buffer(input.shape[0])
        combine_input_buf.copy_(input)
        return combine_input_buf

    def _run_combine(combine_input_buf, layout_desc, weight, zero_copy=False):
        if args.enable_local_combine:
            combine_input_buf, combine_weight_buf = ep_kernels.combine_preprocess(combine_input_buf, layout_desc,
                                                                                  weight=weight, zero_copy=zero_copy)
        else:
            if not zero_copy:
                combine_input_buf = _prepare_combine_input(combine_input_buf)
            layout_desc.token_topk_send_mask.fill_(1)
            combine_weight_buf = None
        combine_out, combine_out_weight = ep_kernels.combine(input_preprocessed=combine_input_buf,
                                                             layout_desc=layout_desc,
                                                             weight_preprocessed=combine_weight_buf)
        return combine_out, combine_out_weight

    if args.disable_weight_dispatch:
        assert args.disable_weight_combine, "disable-weight-dispatch requires --disable-weight-combine"
    if RANK == 0:
        print(f"args = {args}")

    if args.check:
        for n in range(args.iters):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            input_list = [_make_data(random.randint(1, args.M)) for _ in range(args.verify_iters)]
            flash_dispatch_out_list, flash_combine_out_list = [], []
            ref_dispatch_out_list, ref_combine_out_list = [], []

            # flash_comm impl
            for input, weight, exp_indices in input_list:
                straggler(RANK)
                torch.distributed.barrier()
                flash_dispatch_out, flash_dispatch_weight, flash_layout_desc = _run_dispatch(
                    input, weight, exp_indices, copy_out=True)
                # enable zero copy
                if flash_dispatch_weight is None:
                    flash_combine_input = flash_dispatch_out.to(input_dtype)
                    flash_combine_weight = None
                else:
                    flash_combine_input = (flash_dispatch_weight.reshape(-1, 1) * flash_dispatch_out).to(input_dtype)
                    if not args.disable_weight_combine:
                        flash_combine_weight = flash_dispatch_weight
                    else:
                        flash_combine_weight = None
                flash_combine_input_buf = _prepare_combine_input(flash_combine_input)
                flash_combine_out, flash_combine_weight = _run_combine(flash_combine_input_buf, flash_layout_desc,
                                                                       weight=flash_combine_weight, zero_copy=True)
                flash_dispatch_out_list.append((flash_dispatch_out, flash_dispatch_weight))
                flash_combine_out_list.append((flash_combine_out, flash_combine_weight))

            # torch impl
            for input, weight, exp_indices in input_list:
                ref_dispatch_out, ref_dispatch_weight = torch_forward_single(input, weight, exp_indices, args.G)
                if ref_dispatch_weight is None:
                    ref_combine_input = ref_dispatch_out.to(input_dtype)
                else:
                    ref_combine_input = (ref_dispatch_weight.reshape(-1, 1) * ref_dispatch_out).to(input_dtype)
                ref_combine_out = torch_backward_single(ref_combine_input, exp_indices, args.G,
                                                        enable_local_combine=args.enable_local_combine)
                ref_dispatch_out_list.append((ref_dispatch_out, ref_dispatch_weight))
                if not args.disable_weight_combine:
                    if weight is None:
                        ref_combine_weight = None
                    else:
                        ref_combine_weight = weight.clone()
                        ref_combine_weight[exp_indices == args.G] = 0.0
                else:
                    ref_combine_weight = None
                ref_combine_out_list.append((ref_combine_out, ref_combine_weight))

            # verify dispatch
            for idx, (ref_out, flash_out) in enumerate(zip(ref_dispatch_out_list, flash_dispatch_out_list)):
                ref_dispatch_out, ref_dispatch_weight = ref_out
                num_recv_token = ref_dispatch_out.shape[0]
                flash_dispatch_out, flash_dispatch_weight = flash_out
                flash_dispatch_out = flash_dispatch_out[:num_recv_token]
                if flash_dispatch_weight is not None:
                    flash_dispatch_weight = flash_dispatch_weight[:num_recv_token]
                if RANK == 0:
                    print(
                        f"dispatch: shape = {ref_dispatch_out.shape}, {flash_dispatch_out.shape}, ref_sum = {ref_dispatch_out.to(torch.float32).sum()}, flash_sum = {flash_dispatch_out.to(torch.float32).sum()}"
                    )
                    # mask = ref_dispatch_out != flash_dispatch_out
                    # print(f"mask = {mask.nonzero()}, sum = {mask.sum()}, x[mask] = {ref_dispatch_out[mask]}, y[mask] = {flash_dispatch_out[mask]}")

                torch.testing.assert_close(ref_dispatch_out, flash_dispatch_out, rtol=0, atol=0)
                assert (ref_dispatch_weight is None) == (flash_dispatch_weight is None), \
                    f"Weight None-ness mismatch: ref={ref_dispatch_weight is None}, flash={flash_dispatch_weight is None}"
                if ref_dispatch_weight is not None:
                    torch.testing.assert_close(ref_dispatch_weight, flash_dispatch_weight, rtol=0, atol=0)
                    bitwise_equal(ref_dispatch_weight, flash_dispatch_weight)
                bitwise_equal(ref_dispatch_out, flash_dispatch_out)
                # assert_close(ref_dispatch_out, flash_dispatch_out, rtol=0, atol=0)

            # verify combine
            for idx, ((ref_combine_out, ref_combine_weight),
                      (flash_combine_out,
                       flash_combine_weight)) in enumerate(zip(ref_combine_out_list, flash_combine_out_list)):
                atol, rtol = 0, 0
                if RANK == 0:
                    print(
                        f"combine: shape = {ref_combine_out.shape} {ref_combine_out.dtype}, {flash_combine_out.shape} {flash_combine_out.dtype}, atol={atol}, rtol={rtol}, ref_sum = {ref_combine_out.sum()}, flash_sum = {flash_combine_out.sum()}"
                    )
                assert (ref_combine_weight is None) == (flash_combine_weight is None), \
                    f"Combine weight None-ness mismatch: ref={ref_combine_weight is None}, flash={flash_combine_weight is None}"
                if not args.disable_weight_combine and ref_combine_weight is not None:
                    torch.testing.assert_close(ref_combine_weight, flash_combine_weight, rtol=0, atol=0)
                    bitwise_equal(ref_combine_weight, flash_combine_weight)
                torch.testing.assert_close(ref_combine_out, flash_combine_out, rtol=0, atol=0)
                bitwise_equal(ref_combine_out, flash_combine_out)

        print(f"RANK[{RANK}]: pass.")
        torch.distributed.destroy_process_group(EP_GROUP)
        exit(0)

    for rid in range(args.rounds):
        # random simulate token received from dataloader
        L = args.M // 2 if not args.profile else args.M

        token_num = random.randint(L, args.M)

        print(f"Rank-{RANK}: Received {token_num} tokens")

        input, weight, exp_indices = _make_data(token_num)
        ctx = get_torch_prof_ctx(args.profile)
        copy_dispatch_out = False
        with ctx:
            (ref_dispatch_out,
             ref_dispatch_weight), _ = perf_func(partial(torch_forward_single, input, weight, exp_indices, args.G),
                                                 iters=10, warmup_iters=2)
            if ref_dispatch_weight is None:
                ref_combine_input = ref_dispatch_out.to(input_dtype)
            else:
                ref_combine_input = (ref_dispatch_weight.reshape(-1, 1) * ref_dispatch_out).to(input_dtype)

            ref_combine_out, _ = perf_func(
                partial(torch_backward_single, ref_combine_input, exp_indices, args.G,
                        enable_local_combine=args.enable_local_combine), iters=10, warmup_iters=2)

            (dispatch_out, dispatch_weights, layout_desc), flash_comm_dispatch_perf = perf_func(
                partial(_run_dispatch, input, weight, exp_indices, copy_dispatch_out), iters=100, warmup_iters=20)
            # dispatch_out_buf may reuse as the combine input buffer, the value will be overwritten by the combine function, so we need to clone it
            if not copy_dispatch_out:
                dispatch_out = dispatch_out.clone()
            if dispatch_weights is None:
                combine_input = dispatch_out.to(input_dtype)
                combine_weight = None
            else:
                combine_input = (dispatch_weights.reshape(-1, 1) * dispatch_out).to(input_dtype)
                if not args.disable_weight_combine:
                    combine_weight = dispatch_weights
                else:
                    combine_weight = None
            # why not use zero_copy in profile mode?
            # local_combine inplace update the combine input buffer, the result will be incorrect when run combine multiple times.
            (combine_out, combine_out_weight), flash_comm_combine_perf = perf_func(
                partial(_run_combine, combine_input, layout_desc, weight=combine_weight), iters=100, warmup_iters=20)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        if args.profile:
            run_id = os.environ["TORCHELASTIC_RUN_ID"]
            prof_dir = f"prof/{run_id}"
            os.makedirs(prof_dir, exist_ok=True)
            ctx.export_chrome_trace(f"{prof_dir}/trace_rank{EP_GROUP.rank()}.json.gz")
        if RANK == 0:
            print(
                f"input = {input.shape}, weight = {None if weight is None else weight.shape}, ref_dispatch_out = {ref_dispatch_out.shape}, dispatch_out = {dispatch_out.shape}, combine_out = {combine_out.shape}, combine_out_weight is not None = {combine_out_weight is not None}"
            )
        print(
            f"RANK {RANK}: flash_comm_dispatch_perf = {flash_comm_dispatch_perf}ms, flash_comm_combine_perf = {flash_comm_combine_perf}ms"
        )
        num_recv_token = ref_dispatch_out.shape[0]
        dispatch_out = dispatch_out[:num_recv_token]
        if dispatch_weights is not None:
            dispatch_weights = dispatch_weights[:num_recv_token]
        torch.testing.assert_close(dispatch_out, ref_dispatch_out, rtol=0, atol=0)
        torch.testing.assert_close(combine_out, ref_combine_out, rtol=0, atol=0)
        assert (ref_dispatch_weight is None) == (dispatch_weights is None), \
            f"Weight None-ness mismatch: ref={ref_dispatch_weight is None}, flash={dispatch_weights is None}"
        if ref_dispatch_weight is not None:
            torch.testing.assert_close(dispatch_weights, ref_dispatch_weight, rtol=0, atol=0)
            bitwise_equal(dispatch_weights, ref_dispatch_weight)
        bitwise_equal(dispatch_out, ref_dispatch_out)
        bitwise_equal(combine_out, ref_combine_out)
        if not args.disable_weight_combine:
            drop_token_mask = exp_indices == args.G
            ref_weight = weight.clone()
            ref_weight[drop_token_mask] = 0.0
            assert combine_out_weight is not None, "combine_out_weight should not be None when weight combine is enabled"
            torch.testing.assert_close(ref_weight, combine_out_weight, rtol=0, atol=0)
            bitwise_equal(ref_weight, combine_out_weight)
        else:
            # when weight combine is disabled, combine_out_weight should be None
            assert combine_out_weight is None, f"combine_out_weight should be None, got {type(combine_out_weight)}"
        torch.cuda.synchronize()
        torch.distributed.barrier()
