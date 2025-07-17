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
from triton_dist.utils import (perf_func, get_torch_prof_ctx, init_nvshmem_by_torch_process_group)
from functools import partial

import argparse
import random
import os
import datetime
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


def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.zeros(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    # out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


def sort_by_vectors(x):
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


def calc_scatter_index_stable(choosed_experts: torch.Tensor):
    return (choosed_experts.flatten().argsort(stable=True).argsort().int().view(choosed_experts.shape))


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


def initialize_distributed():
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP has already been initialized"

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    # use all ranks as tp group
    EP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(EP_GROUP)

    init_seed(seed=RANK)

    torch.cuda.synchronize()
    init_nvshmem_by_torch_process_group(EP_GROUP)
    return EP_GROUP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=7168)
    parser.add_argument("-G", type=int, default=256)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", default=3, type=int, help="perf iterations")
    parser.add_argument("--verify-iters", default=5, type=int)
    parser.add_argument("--bench_iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--with_scale", action="store_true")
    return parser.parse_args()


def perf_torch(args, input, scale_tensor, exp_indices):
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
    scattered_scale_tensor = torch.empty(
        (scale_tensor.size(0) * args.topk),
        dtype=scale_tensor.dtype,
        device=scale_tensor.device,
    )
    scattered_scale_tensor.copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank))
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    # following are the all2all
    a2a_splits = torch.empty_like(splits_gpu_cur_rank)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu_cur_rank, group=EP_GROUP)
    a2a_splits_cpu = a2a_splits.cpu()
    ep_size = EP_GROUP.size()
    a2a_dispatch_output = torch.empty([a2a_splits_cpu.sum(), input.size(1)], dtype=input.dtype, device=input.device)
    a2a_dispatch_scale = torch.empty([a2a_splits_cpu.sum()], dtype=scale_tensor.dtype, device=scale_tensor.device)
    torch.cuda.synchronize()

    def fwd():
        torch.distributed.all_to_all_single(
            output=a2a_dispatch_output,
            input=scattered_input,
            output_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
            input_split_sizes=splits_cpu_cur_rank.reshape(ep_size, -1).sum(-1).tolist(),
            group=EP_GROUP,
        )
        if args.with_scale:
            torch.distributed.all_to_all_single(
                output=a2a_dispatch_scale,
                input=scattered_scale_tensor,
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
    return a2a_dispatch_output, a2a_dispatch_scale, avg_time


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    assert (args.G % WORLD_SIZE == 0), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"

    experts_per_rank = args.G // WORLD_SIZE
    input_dtype = DTYPE_MAP[args.dtype]
    triton_a2a_op = EPAll2AllLayer(EP_GROUP, args.M, args.N, args.topk, RANK, args.G, LOCAL_WORLD_SIZE, WORLD_SIZE,
                                   input_dtype)

    def _make_data(token_num):
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = (torch.rand(token_num, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
        scale_tensor = torch.rand(token_num, dtype=torch.float32).to("cuda")
        return input, scale_tensor, exp_indices

    if args.check:
        for n in range(args.iters):
            torch.cuda.empty_cache()

            input_list = [_make_data(random.randint(args.M // 2, args.M)) for _ in range(args.verify_iters)]
            combine_out_list, dispatch_out_list, torch_input_list, torch_out_list = [], [], [], []

            # torch impl
            for input, scale_tensor, exp_indices in input_list:
                ref_out, ref_scale, ref_time = perf_torch(args, input, scale_tensor, exp_indices)
                torch_out_list.append(ref_out)
                torch_input_list.append(input * args.topk)

            # dist triton impl
            for input, scale_tensor, exp_indices in input_list:
                dispatch_out = triton_a2a_op.dispatch(input, exp_indices)
                dispatch_out_list.append(dispatch_out)
                torch.cuda.synchronize()
                combined_out = triton_a2a_op.combine(dispatch_out)
                combine_out_list.append(combined_out)

            # torch.cuda.synchronize()
            # verify
            for idx, (torch_out, dist_out) in enumerate(zip(torch_out_list, dispatch_out_list)):
                if RANK == 0:
                    print(f"dispatch: shape = {torch_out.shape}, {dist_out.shape}")
                try:
                    sorted_dist_out = sort_by_vectors(dist_out)
                    sorted_ref_out = sort_by_vectors(torch_out)
                    torch.testing.assert_close(sorted_dist_out, sorted_ref_out, atol=0, rtol=0)
                except Exception as e:
                    raise e

            for idx, (input, triton_combine_out) in enumerate(zip(torch_input_list, combine_out_list)):
                if RANK == 0:
                    print(f"combine: shape = {input.shape}, {triton_combine_out.shape}")
                try:
                    torch.testing.assert_close(input, triton_combine_out, atol=1e-2, rtol=1e-2)
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

        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)

        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = (torch.rand(token_num, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
        scale_tensor = torch.rand(token_num, dtype=torch.float32).to("cuda")

        ctx = get_torch_prof_ctx(args.profile)
        with ctx:
            ref_out, ref_scale, ref_time = perf_torch(args, input, scale_tensor, exp_indices)
            triton_dispatch_out, triton_perf = perf_func(partial(triton_a2a_op.dispatch, input, exp_indices), iters=100,
                                                         warmup_iters=20)
            combined_out, triton_combine_perf = perf_func(partial(triton_a2a_op.combine, triton_dispatch_out),
                                                          iters=100, warmup_iters=20)

        torch.cuda.synchronize()
        torch.distributed.barrier()

        torch.distributed.barrier()  # wait all rank dispatch
        sorted_triton_dispatch_out = sort_by_vectors(triton_dispatch_out)
        sorted_ref_out = sort_by_vectors(ref_out)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        if args.profile:
            run_id = os.environ["TORCHELASTIC_RUN_ID"]
            prof_dir = f"prof/{run_id}"
            os.makedirs(prof_dir, exist_ok=True)
            ctx.export_chrome_trace(f"{prof_dir}/trace_rank{EP_GROUP.rank()}.json.gz")

        _check(sorted_triton_dispatch_out, sorted_ref_out)

        combined_out, triton_combine_perf = perf_func(partial(triton_a2a_op.combine, triton_dispatch_out), iters=100,
                                                      warmup_iters=20)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        torch.testing.assert_close(combined_out, input * args.topk, rtol=1e-2, atol=1e-2)

        print(f"RANK {RANK}: triton dispatch perf = {triton_perf}ms, triton_combine_perf = {triton_combine_perf}ms")

    triton_a2a_op.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group(EP_GROUP)
