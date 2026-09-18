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

import argparse
import datetime
import gzip
import json
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist

from flash_comm.ep import EPKernels
from flash_comm.ep.ep_kernels import EPCommLayoutDesc

import test_ep_intranode as ep_ref
from test_ep_intranode import (  # noqa: E402
    DTYPE_MAP, bitwise_equal, calc_combine_preprocess_hbm_bytes, calc_dispatch_postprocess_hbm_bytes,
    densify_aligned_output, format_gb, format_gbps, format_ms, generate_random_exp_indices, init_seed, perf_func,
    straggler,
)

RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=5120)
    parser.add_argument("-G", type=int, default=384)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", default=3000, type=int, help="correctness iterations")
    parser.add_argument("--verify-iters", default=20, type=int)
    parser.add_argument("--bench-iters", default=100, type=int)
    parser.add_argument("--warmup-iters", default=20, type=int)
    parser.add_argument("--drop_ratio", default=0.1, type=float)
    parser.add_argument("--rounds", default=1, type=int)
    parser.add_argument("--num_sm", default=16, type=int)
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--weight_dtype", default="float32", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prof-dir", default="prof")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--enable-local-combine", action="store_true")
    parser.add_argument("--num_worst_tokens", default=-1, type=int)
    parser.add_argument("--disable-weight-combine", action="store_true")
    parser.add_argument("--disable-weight-dispatch", action="store_true")
    parser.add_argument("--expert-alignment", default=1, type=int)
    parser.add_argument(
        "--local-world-size",
        type=int,
        default=0,
        help="GPUs per node. Defaults to EP_LOCAL_WORLD_SIZE/LOCAL_WORLD_SIZE/world_size.",
    )
    parser.add_argument(
        "--qp-schedule",
        type=str,
        default="",
        help="Comma-separated per-call QP counts cycled across iterations for internode "
        "dispatch/combine (e.g. '1,2,4'); each value must be <= FLASH_COMM_EP_NUM_QPS. "
        "Dispatch and combine cycle with different phases to stress dynamic QP switching.",
    )
    return parser.parse_args()


def resolve_local_world_size(args) -> int:
    if args.local_world_size > 0:
        return args.local_world_size
    env_value = int(os.environ.get("EP_LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "0")))
    return env_value if env_value > 0 else WORLD_SIZE


def get_torch_prof_ctx(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )


def load_profile_kernel_averages(trace_path: str, is_internode: bool):
    if is_internode:
        patterns = [
            ("dispatch_index", "kernel_compute_stable_local_token_within_expert_offset"),
            ("dispatch_layout", "kernel_compute_dispatch_layout"),
            ("combine_reduce", "kernel_combine_internode_reduce"),
            ("dispatch_nvl", "kernel_dispatch_internode"),
            ("dispatch_postprocess", "kernel_dispatch_postprocess"),
            ("combine_preprocess", "kernel_combine_preprocess"),
            ("combine_nvl", "kernel_combine_internode"),
            ("gin_barrier", "kernel_internode_gin_barrier"),
        ]
    else:
        patterns = [
            ("dispatch_nvl", "kernel_dispatch_intranode"),
            ("dispatch_postprocess", "kernel_dispatch_postprocess"),
            ("combine_preprocess", "kernel_combine_preprocess"),
            ("combine_nvl", "kernel_combine_intranode"),
        ]
    totals = {key: 0.0 for key, _ in patterns}
    counts = {key: 0 for key, _ in patterns}
    with gzip.open(trace_path, "rt") as f:
        events = json.load(f).get("traceEvents", [])
    for event in events:
        name = event.get("name", "")
        duration_us = event.get("dur")
        if duration_us is None:
            continue
        for key, needle in patterns:
            if needle in name:
                totals[key] += duration_us / 1000.0
                counts[key] += 1
                break
    return {key: (totals[key] / counts[key] if counts[key] else 0.0) for key in totals}


def rows_by_rank(mask: torch.Tensor, target_ranks: torch.Tensor) -> torch.Tensor:
    rows = torch.zeros(WORLD_SIZE, dtype=torch.float64, device=target_ranks.device)
    selected = target_ranks[mask].to(torch.long)
    rows.scatter_add_(0, selected, torch.ones_like(selected, dtype=torch.float64))
    dist.all_reduce(rows, op=dist.ReduceOp.SUM)
    return rows


def payload_stats(layout_desc, exp_indices, input_tensor, dispatch_weight, combine_weight, experts_per_rank: int,
                  is_internode: bool, local_world_size: int):
    target_ranks = torch.div(exp_indices, experts_per_rank, rounding_mode="floor")
    valid_target = target_ranks < WORLD_SIZE
    token_bytes = input_tensor.shape[1] * input_tensor.element_size()
    offset_bytes = exp_indices.element_size()
    topk = exp_indices.shape[1]
    dispatch_meta_bytes_per_row = topk * offset_bytes
    if dispatch_weight is not None:
        dispatch_meta_bytes_per_row += topk * dispatch_weight.element_size()

    send_mask = layout_desc.token_topk_send_mask
    if send_mask is None:
        raise ValueError("layout_desc.token_topk_send_mask is required for profiling")
    if is_internode and send_mask.dim() == 3:
        send_mask = send_mask[RANK // local_world_size, :exp_indices.size(0)]
    send_mask = send_mask.to(torch.bool)

    if is_internode:
        physical_dispatch = valid_target & send_mask
        local_postprocess_copy = valid_target & (~send_mask)
        final_rows = rows_by_rank(valid_target, target_ranks)
        local_copy_rows = rows_by_rank(local_postprocess_copy, target_ranks)
        dispatch_rows_avg = (final_rows - local_copy_rows).mean().item()
        combine_rows_avg = rows_by_rank(physical_dispatch, target_ranks).mean().item()
        combine_weight_entries = torch.tensor([int(valid_target.sum().item())], dtype=torch.float64,
                                              device=input_tensor.device)
        dist.all_reduce(combine_weight_entries, op=dist.ReduceOp.SUM)
        combine_weight_entries_avg = combine_weight_entries.item() / WORLD_SIZE
    else:
        valid_remote = valid_target & (target_ranks != RANK)
        dispatch_rows_avg = float((send_mask & valid_remote).sum().item())
        combine_rows_avg = float((layout_desc.token_topk_send_mask.to(torch.bool) & valid_remote).sum().item())
        combine_weight_entries_avg = float(valid_remote.sum().item())

    dispatch_token_bytes = int(dispatch_rows_avg * token_bytes)
    dispatch_bytes_with_meta = int(dispatch_rows_avg * (token_bytes + dispatch_meta_bytes_per_row))
    combine_token_bytes = int(combine_rows_avg * token_bytes)
    combine_bytes_with_weight = combine_token_bytes
    if combine_weight is not None:
        combine_bytes_with_weight += int(combine_weight_entries_avg * combine_weight.element_size())
    remote_node_count = 0
    combine_rdma_token_count = 0
    if is_internode:
        num_nodes = WORLD_SIZE // local_world_size
        my_node = RANK // local_world_size
        local_rank = RANK % local_world_size
        remote_node_count = num_nodes - 1
        if layout_desc.num_tokens_per_rank is None:
            raise ValueError("layout_desc.num_tokens_per_rank is required for RDMA profiling")
        remote_source_ranks = [node * local_world_size + local_rank for node in range(num_nodes) if node != my_node]
        combine_rdma_token_count = int(layout_desc.num_tokens_per_rank[remote_source_ranks].sum().item())

    dispatch_rdma_bytes = remote_node_count * input_tensor.shape[0] * (
        token_bytes + 3 * topk * offset_bytes +
        (topk * dispatch_weight.element_size() if dispatch_weight is not None else 0))
    combine_rdma_bytes = combine_rdma_token_count * token_bytes
    if combine_weight is not None:
        combine_rdma_bytes += combine_rdma_token_count * topk * combine_weight.element_size()
    return {
        "dispatch_rows_avg": dispatch_rows_avg,
        "combine_rows_avg": combine_rows_avg,
        "dispatch_token_bytes": dispatch_token_bytes,
        "dispatch_bytes_with_meta": dispatch_bytes_with_meta,
        "combine_token_bytes": combine_token_bytes,
        "combine_bytes_with_weight": combine_bytes_with_weight,
        "total_token_bytes": dispatch_token_bytes + combine_token_bytes,
        "dispatch_rdma_bytes": dispatch_rdma_bytes,
        "combine_rdma_bytes": combine_rdma_bytes,
    }


def main():
    args = parse_args()
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        timeout=datetime.timedelta(seconds=1800),
    )
    init_seed(RANK)
    ep_group = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    ep_ref.EP_GROUP = ep_group

    local_world_size = resolve_local_world_size(args)
    if WORLD_SIZE % local_world_size != 0:
        raise ValueError(f"WORLD_SIZE must be divisible by local_world_size, got {WORLD_SIZE} and {local_world_size}")
    is_internode = WORLD_SIZE > local_world_size
    if RANK == 0:
        print(
            f"EP test initialized: world_size={WORLD_SIZE}, local_world_size={local_world_size}, "
            f"is_internode={is_internode}",
            flush=True,
        )

    assert args.G % WORLD_SIZE == 0
    experts_per_rank = args.G // WORLD_SIZE
    input_dtype = DTYPE_MAP[args.dtype]
    weight_dtype = DTYPE_MAP[args.weight_dtype]
    assert input_dtype == torch.bfloat16
    assert weight_dtype == torch.float32

    def make_data(token_num):
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk, args.drop_ratio).to("cuda")
        input_tensor = torch.rand(token_num, args.N, dtype=torch.float32).to(input_dtype).cuda()
        if args.disable_weight_dispatch:
            return input_tensor, None, exp_indices
        weight = torch.randn(token_num, args.topk, dtype=torch.float32, device="cuda")
        weight = torch.nn.functional.softmax(weight, dim=1).to(weight_dtype)
        return input_tensor, weight, exp_indices

    def torch_backward_internode_flash_order(input_tensor, exp_indices, layout_desc):
        topk = exp_indices.size(1)
        ep_size = ep_group.size()
        num_experts_per_rank = args.G // ep_size
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)[:args.G]
        splits_cpu_cur_rank = splits_gpu_cur_rank.cpu()
        _, new_index = exp_indices.flatten().sort(stable=True)
        new_index = new_index.to(torch.int32)

        a2a_splits = torch.empty_like(splits_gpu_cur_rank)
        dist.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=ep_group)
        a2a_splits_cpu = a2a_splits.cpu()
        permute_splits = a2a_splits_cpu.reshape(-1, num_experts_per_rank).permute(-1, -2).flatten()
        permute_outputs = torch.split(input_tensor, permute_splits.tolist())
        a2a_expert_output = torch.cat([
            permute_outputs[expert_offset * ep_size + src_rank]
            for src_rank in range(ep_size)
            for expert_offset in range(num_experts_per_rank)
        ], dim=0)
        all2all_out = torch.empty([splits_cpu_cur_rank.sum(), input_tensor.shape[-1]], device=input_tensor.device,
                                  dtype=input_tensor.dtype)
        dist.all_to_all_single(
            output=all2all_out,
            input=a2a_expert_output,
            output_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            group=ep_group,
        )
        padded = torch.zeros((exp_indices.numel(), all2all_out.size(1)), device=all2all_out.device,
                             dtype=all2all_out.dtype)
        padded[:splits_cpu_cur_rank.sum()] = all2all_out
        gather_output = torch.zeros_like(padded)
        gather_output[new_index] = padded
        gather_view = gather_output.view((gather_output.size(0) // topk, topk, gather_output.size(-1)))

        target_ranks = torch.div(exp_indices, num_experts_per_rank, rounding_mode="floor")
        matches = target_ranks.unsqueeze(2) == target_ranks.unsqueeze(1)
        target_indices = matches.to(torch.int32).argmax(dim=2)
        accum = torch.zeros_like(gather_view, dtype=torch.float32)
        accum.scatter_add_(1,
                           target_indices.unsqueeze(-1).expand(-1, -1, gather_view.size(-1)),
                           gather_view.to(torch.float32))
        accum = accum.to(input_tensor.dtype).to(torch.float32)

        target_nodes = torch.div(target_ranks, local_world_size, rounding_mode="floor")
        valid_target = target_ranks < ep_size
        send_mask = layout_desc.token_topk_send_mask
        if send_mask.dim() == 3:
            send_mask = send_mask[RANK // local_world_size, :exp_indices.size(0)]
        send_mask = send_mask.to(torch.bool) & valid_target
        out = torch.zeros((exp_indices.size(0), input_tensor.shape[-1]), dtype=torch.float32,
                          device=input_tensor.device)
        for node_id in range(ep_size // local_world_size):
            partial = torch.zeros_like(out)
            for topk_idx in range(topk):
                lane_mask = send_mask[:, topk_idx] & (target_nodes[:, topk_idx] == node_id)
                partial[lane_mask] += accum[:, topk_idx, :][lane_mask]
            out += partial.to(input_tensor.dtype).to(torch.float32)
        return out.to(input_tensor.dtype)

    ep_kernels = EPKernels(
        max_m=args.M,
        hidden=args.N,
        topk=args.topk,
        num_experts=args.G,
        local_world_size=local_world_size,
        ep_group=ep_group,
        num_sm=args.num_sm,
        num_worst_tokens=args.num_worst_tokens,
        expert_alignment=args.expert_alignment,
    )

    qp_schedule = [int(v) for v in args.qp_schedule.split(",") if v.strip()]
    if qp_schedule:
        assert is_internode, "--qp-schedule only applies to internode runs"
        assert all(v >= 1 for v in qp_schedule), f"invalid --qp-schedule: {qp_schedule}"
    # Per-protocol call counters; every rank runs the same call sequence, so
    # the derived per-call QP count stays identical across the EP group.
    qp_call_idx = {"dispatch": 0, "combine": 0}

    def next_num_qps(kind):
        if not qp_schedule:
            return None
        idx = qp_call_idx[kind]
        qp_call_idx[kind] = idx + 1
        # Offset combine by one so dispatch/combine of the same step usually
        # run with different QP counts.
        return qp_schedule[(idx + (1 if kind == "combine" else 0)) % len(qp_schedule)]

    def run_dispatch(input_tensor, weight, exp_indices, copy_out=False):
        token_offset, expert_counts = ep_kernels.compute_stable_local_token_within_expert_offset_and_expert_counts(
            exp_indices)
        layout_desc = EPCommLayoutDesc(token_within_expert_offset=token_offset, expert_counts=expert_counts)
        dispatch_out, dispatch_weights, layout_desc = ep_kernels.dispatch(input_tensor, exp_indices, weight,
                                                                          layout_desc, num_qps=next_num_qps("dispatch"))
        return ep_kernels.dispatch_postprocess(dispatch_out.clone() if copy_out else dispatch_out, dispatch_weights,
                                               layout_desc)

    def prepare_combine_input(input_tensor):
        combine_input_buf = ep_kernels.get_combine_buffer(input_tensor.shape[0])
        combine_input_buf.copy_(input_tensor)
        return combine_input_buf

    def run_combine(combine_input, layout_desc, weight, zero_copy=False):
        if is_internode:
            if not zero_copy:
                combine_input = prepare_combine_input(combine_input)
            combine_input, weight = ep_kernels.combine_preprocess(combine_input, layout_desc, weight=weight,
                                                                  zero_copy=True)
        elif args.enable_local_combine:
            combine_input, weight = ep_kernels.combine_preprocess(combine_input, layout_desc, weight=weight,
                                                                  zero_copy=zero_copy)
        else:
            if not zero_copy:
                combine_input = prepare_combine_input(combine_input)
            layout_desc.token_topk_send_mask.fill_(1)
            weight = None
        return ep_kernels.combine(combine_input, layout_desc, weight, num_qps=next_num_qps("combine"))

    def make_combine_input(dispatch_out, dispatch_weight):
        if dispatch_weight is None:
            return dispatch_out.to(input_dtype), None
        combine_input = (dispatch_weight.reshape(-1, 1) * dispatch_out).to(input_dtype)
        return combine_input, None if args.disable_weight_combine else dispatch_weight

    def align_dispatch(dispatch_out, dispatch_weight, ref_token_count, ref_expert_counts):
        if args.expert_alignment > 1:
            dispatch_out = densify_aligned_output(dispatch_out, ref_expert_counts, args.expert_alignment)
            if dispatch_weight is not None:
                dispatch_weight = densify_aligned_output(dispatch_weight, ref_expert_counts, args.expert_alignment)
        else:
            dispatch_out = dispatch_out[:ref_token_count]
            if dispatch_weight is not None:
                dispatch_weight = dispatch_weight[:ref_token_count]
        return dispatch_out, dispatch_weight

    def reference_combine(ref_dispatch_out, ref_dispatch_weight, exp_indices, layout_desc, original_weight):
        ref_combine_input, _ = make_combine_input(ref_dispatch_out, ref_dispatch_weight)
        if is_internode:
            ref_out = torch_backward_internode_flash_order(ref_combine_input, exp_indices, layout_desc)
        else:
            ref_out = ep_ref.torch_backward_single(ref_combine_input, exp_indices, args.G,
                                                   enable_local_combine=args.enable_local_combine)
        ref_weight = None
        if not args.disable_weight_combine and original_weight is not None:
            ref_weight = original_weight.clone()
            ref_weight[exp_indices == args.G] = 0.0
        return ref_out, ref_weight

    def check_dispatch(ref_out, ref_weight, ref_counts, flash_out, flash_weight, layout_desc, idx):
        flash_out, flash_weight = align_dispatch(flash_out, flash_weight, ref_out.shape[0], ref_counts)
        dispatch_mismatch_msg = (f"dispatch mismatch[{idx}], ref_shape={tuple(ref_out.shape)}, "
                                 f"flash_shape={tuple(flash_out.shape)}")
        if ref_out.shape == flash_out.shape:
            mismatch_rows = torch.nonzero((ref_out != flash_out).any(dim=1), as_tuple=False).flatten()
            if mismatch_rows.numel() > 0:
                dispatch_mismatch_msg += (f", mismatch_rows={mismatch_rows.numel()}, "
                                          f"first_row={int(mismatch_rows[0])}, last_row={int(mismatch_rows[-1])}")
        torch.testing.assert_close(ref_out, flash_out, rtol=0, atol=0, msg=dispatch_mismatch_msg)
        assert bitwise_equal(ref_out, flash_out)
        assert (ref_weight is None) == (flash_weight is None), f"dispatch weight mismatch[{idx}]"
        if ref_weight is not None:
            torch.testing.assert_close(
                ref_weight, flash_weight, rtol=0, atol=0,
                msg=f"dispatch weight mismatch[{idx}], ref_shape={tuple(ref_weight.shape)}, "
                f"flash_shape={tuple(flash_weight.shape)}")
            assert bitwise_equal(ref_weight, flash_weight)
        if layout_desc.recv_expert_counts is not None:
            torch.testing.assert_close(layout_desc.recv_expert_counts.cpu(), ref_counts.to(torch.int32), rtol=0, atol=0)
        return flash_out, flash_weight

    def check_combine(ref_out, ref_weight, flash_out, flash_weight, idx):
        torch.testing.assert_close(
            ref_out, flash_out, rtol=0, atol=0,
            msg=f"combine mismatch[{idx}], ref_shape={tuple(ref_out.shape)}, flash_shape={tuple(flash_out.shape)}")
        assert bitwise_equal(ref_out, flash_out)
        assert (ref_weight is None) == (flash_weight is None), f"combine weight mismatch[{idx}]"
        if ref_weight is not None:
            torch.testing.assert_close(
                ref_weight, flash_weight, rtol=0, atol=0,
                msg=f"combine weight mismatch[{idx}], ref_shape={tuple(ref_weight.shape)}, "
                f"flash_shape={tuple(flash_weight.shape)}")
            assert bitwise_equal(ref_weight, flash_weight)

    try:
        if args.disable_weight_dispatch:
            assert args.disable_weight_combine, "disable-weight-dispatch requires --disable-weight-combine"
        if RANK == 0:
            print(f"args = {args}", flush=True)

        if args.check:
            for n in range(args.iters):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                inputs = [make_data(random.randint(1, args.M)) for _ in range(args.verify_iters)]
                flash_dispatches, flash_combines, ref_dispatches, ref_combines = [], [], [], []
                for input_tensor, weight, exp_indices in inputs:
                    straggler(RANK)
                    dist.barrier(group=ep_group)
                    flash_out, flash_weight, layout_desc = run_dispatch(input_tensor, weight, exp_indices,
                                                                        copy_out=True)
                    combine_input, combine_weight = make_combine_input(flash_out, flash_weight)
                    if is_internode:
                        flash_combine_out, flash_combine_weight = run_combine(combine_input, layout_desc,
                                                                              combine_weight, zero_copy=False)
                    else:
                        flash_combine_out, flash_combine_weight = run_combine(prepare_combine_input(combine_input),
                                                                              layout_desc, combine_weight,
                                                                              zero_copy=True)
                    flash_dispatches.append((flash_out, flash_weight, layout_desc))
                    flash_combines.append((flash_combine_out, flash_combine_weight))

                for input_tensor, weight, exp_indices in inputs:
                    ref_out, ref_weight, ref_counts = ep_ref.torch_forward_single(input_tensor, weight, exp_indices,
                                                                                  args.G)
                    layout_desc = flash_dispatches[len(ref_dispatches)][2]
                    ref_combine_out, ref_combine_weight = reference_combine(ref_out, ref_weight, exp_indices,
                                                                            layout_desc, weight)
                    ref_dispatches.append((ref_out, ref_weight, ref_counts))
                    ref_combines.append((ref_combine_out, ref_combine_weight))

                for idx, (ref_item, flash_item) in enumerate(zip(ref_dispatches, flash_dispatches)):
                    check_dispatch(*ref_item, *flash_item, idx)
                for idx, (ref_item, flash_item) in enumerate(zip(ref_combines, flash_combines)):
                    check_combine(*ref_item, *flash_item, idx)
                if RANK == 0:
                    print(f"check iter {n}: {args.verify_iters} inputs bitwise OK", flush=True)
            if RANK == 0:
                print("✅ EP dispatch+combine check passed.", flush=True)
            return

        for rid in range(args.rounds):
            token_num = args.M if args.profile else random.randint(args.M // 2, args.M)
            input_tensor, weight, exp_indices = make_data(token_num)
            if RANK == 0:
                print(f"round {rid}: token_num={token_num}", flush=True)
            ctx = get_torch_prof_ctx(args.profile)
            with ctx:
                (ref_out, ref_weight, ref_counts), _ = perf_func(
                    partial(ep_ref.torch_forward_single, input_tensor, weight, exp_indices, args.G), iters=10,
                    warmup_iters=2)
                (dispatch_out, dispatch_weights, layout_desc), dispatch_perf = perf_func(
                    partial(run_dispatch, input_tensor, weight, exp_indices, False), iters=args.bench_iters,
                    warmup_iters=args.warmup_iters)
                dispatch_out = dispatch_out.clone()
                if dispatch_weights is not None:
                    dispatch_weights = dispatch_weights.clone()
                dispatch_out, dispatch_weights = check_dispatch(ref_out, ref_weight, ref_counts, dispatch_out,
                                                                dispatch_weights, layout_desc, rid)
                combine_input, combine_weight = make_combine_input(dispatch_out, dispatch_weights)
                (combine_out, combine_out_weight), combine_perf = perf_func(
                    partial(run_combine, combine_input, layout_desc, combine_weight, False), iters=args.bench_iters,
                    warmup_iters=args.warmup_iters)
            ref_combine_out, ref_combine_weight = reference_combine(ref_out, ref_weight, exp_indices, layout_desc,
                                                                    weight)
            check_combine(ref_combine_out, ref_combine_weight, combine_out, combine_out_weight, rid)
            dist.barrier(group=ep_group)
            if RANK == 0:
                print(
                    f"round {rid}: dispatch_perf={dispatch_perf:.4f} ms, "
                    f"combine_perf={combine_perf:.4f} ms, bitwise OK", flush=True)

            if args.profile:
                run_id = os.environ.get("TORCHELASTIC_RUN_ID", "none")
                prof_dir = Path(args.prof_dir) / run_id
                prof_dir.mkdir(parents=True, exist_ok=True)
                trace_path = prof_dir / f"trace_rank{ep_group.rank()}.json.gz"
                ctx.export_chrome_trace(str(trace_path))
                averages = load_profile_kernel_averages(str(trace_path), is_internode)
                stats = payload_stats(layout_desc, exp_indices, input_tensor, weight, combine_weight, experts_per_rank,
                                      is_internode, local_world_size)
                dispatch_kernel = averages.get("dispatch_nvl", 0.0)
                combine_kernel = averages.get("combine_nvl", 0.0)
                dispatch_index = averages.get("dispatch_index", 0.0)
                dispatch_layout = averages.get("dispatch_layout", 0.0)
                gin_barrier = averages.get("gin_barrier", 0.0)
                dispatch_post = averages.get("dispatch_postprocess", 0.0)
                combine_pre = averages.get("combine_preprocess", 0.0)
                combine_reduce = averages.get("combine_reduce", 0.0)
                dispatch_hbm = calc_dispatch_postprocess_hbm_bytes(layout_desc.recv_topk_scatter_indices, dispatch_out,
                                                                   dispatch_weights)
                combine_hbm = calc_combine_preprocess_hbm_bytes(layout_desc.recv_topk_scatter_indices, combine_input,
                                                                combine_weight)
                print(
                    f"RANK {RANK}: trace_path = {trace_path}, "
                    f"dispatch_rows_avg = {stats['dispatch_rows_avg']:.1f}, "
                    f"combine_rows_avg = {stats['combine_rows_avg']:.1f}", flush=True)
                print(
                    f"RANK {RANK}: dispatch_token_payload = {format_gb(stats['dispatch_token_bytes'])}, "
                    f"dispatch_payload_with_meta = {format_gb(stats['dispatch_bytes_with_meta'])}, "
                    f"dispatch_index_time = {format_ms(dispatch_index)}, "
                    f"dispatch_layout_time = {format_ms(dispatch_layout)}, "
                    f"gin_barrier_time = {format_ms(gin_barrier)}, "
                    f"dispatch_kernel_time = {format_ms(dispatch_kernel)}, "
                    f"dispatch_token_bw = {format_gbps(stats['dispatch_token_bytes'], dispatch_kernel)}, "
                    f"dispatch_bw_with_meta = {format_gbps(stats['dispatch_bytes_with_meta'], dispatch_kernel)}, "
                    f"dispatch_rdma_total_payload = {format_gb(stats['dispatch_rdma_bytes'])}, "
                    f"dispatch_rdma_bw = {format_gbps(stats['dispatch_rdma_bytes'], dispatch_kernel)}, "
                    f"dispatch_postprocess_time = {format_ms(dispatch_post)}, "
                    f"dispatch_postprocess_hbm_payload = {format_gb(dispatch_hbm)}, "
                    f"dispatch_postprocess_hbm_bw = {format_gbps(dispatch_hbm, dispatch_post)}", flush=True)
                print(
                    f"RANK {RANK}: combine_token_payload = {format_gb(stats['combine_token_bytes'])}, "
                    f"combine_payload_with_weight = {format_gb(stats['combine_bytes_with_weight'])}, "
                    f"combine_kernel_time = {format_ms(combine_kernel)}, "
                    f"combine_token_bw = {format_gbps(stats['combine_token_bytes'], combine_kernel)}, "
                    f"combine_bw_with_weight = {format_gbps(stats['combine_bytes_with_weight'], combine_kernel)}, "
                    f"combine_rdma_total_payload = {format_gb(stats['combine_rdma_bytes'])}, "
                    f"combine_rdma_bw = {format_gbps(stats['combine_rdma_bytes'], combine_kernel)}, "
                    f"combine_reduce_time = {format_ms(combine_reduce)}, "
                    f"combine_preprocess_time = {format_ms(combine_pre)}, "
                    f"combine_preprocess_hbm_payload = {format_gb(combine_hbm)}, "
                    f"combine_preprocess_hbm_bw = {format_gbps(combine_hbm, combine_pre)}", flush=True)
                total_time = dispatch_kernel + combine_kernel
                print(
                    f"RANK {RANK}: total_token_payload = {format_gb(stats['total_token_bytes'])}, "
                    f"total_comm_time = {format_ms(total_time)}, "
                    f"total_token_bw = {format_gbps(stats['total_token_bytes'], total_time)}", flush=True)
        if RANK == 0:
            print(f"✅ EP dispatch test passed ({args.rounds} rounds, bitwise).", flush=True)
    finally:
        ep_kernels.finalize()
        if dist.is_initialized():
            dist.destroy_process_group(ep_group)
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
