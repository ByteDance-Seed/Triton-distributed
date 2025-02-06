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
import torch.distributed
import triton
import triton.language as tl
import triton.distributed
import pynvshmem

import argparse
import random
import os
import datetime
import numpy as np
from tabulate import tabulate

from triton.distributed.utils import group_profile
from triton.distributed.kernels.nvidia import create_all_to_all_context, fast_all_to_all, all_to_all_post_process


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.empty(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    out[0] = 0
    _ = torch.cumsum(splits, 0, out=out[1:])
    return out


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
    torch.use_deterministic_algorithms(True, warn_only=True)
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


EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))


def initialize_distributed(enable_flux=False):
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

    init_seed(seed=RANK)

    if enable_flux:
        flux.init_flux_shm(EP_GROUP)
    else:
        pynvshmem.init_nvshmem_by_uniqueid(EP_GROUP)

    torch.cuda.synchronize()
    return EP_GROUP


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-N", type=int, default=3584)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bench_iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--sm_margin", default=16, type=int, help="sm margin")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--with_scale", action="store_true")
    parser.add_argument("--enable_flux", action="store_true", help="enable flux")
    return parser.parse_args()


def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


def perf_torch(input, scale_tensor, exp_indices):
    # prepare the indexes
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


def perf_flux(input, scale_tensor, exp_indices):
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    splits_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

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
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    scattered_scale_tensor.copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank))

    # the last op should directly write the output buffer into the comm buffer
    flux_input_tensors = flux_op.get_input_buffer(scattered_input.size(), 2, args.with_scale)
    flux_input_tensors[0].copy_(scattered_input)
    assert torch.allclose(scattered_input, flux_input_tensors[0])
    if args.with_scale:
        flux_input_tensors[1].copy_(scattered_scale_tensor)
    ep_size = EP_GROUP.size()

    def fwd():
        flux_out = flux_op.forward(
            [scattered_input.size(0), scattered_input.size(1)],
            splits_cumsum,
            2,
            args.with_scale,
        )
        return flux_out

    torch.cuda._sleep(1000000000)
    # warmup
    for _ in range(10):
        flux_out = fwd()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    # bench
    st.record()
    for _ in range(args.bench_iters):
        flux_out = fwd()
    ed.record()
    torch.cuda.synchronize()
    avg_time = st.elapsed_time(ed) / args.bench_iters

    data_vec = []
    scale_vec = []
    output_splits = flux_out[0].cpu().reshape(ep_size, -1).sum(dim=-1)

    for i in range(WORLD_SIZE):
        n_token_from_tgt_rank = output_splits[i]
        _start = i * args.M * args.topk
        data_vec.append(flux_out[1][_start:_start + n_token_from_tgt_rank])
        if args.with_scale:
            assert len(flux_out) == 3
            scale_vec.append(flux_out[2][_start:_start + n_token_from_tgt_rank])

    output = torch.concat(data_vec)
    scale_output = torch.concat(scale_vec) if args.with_scale else None
    return output, scale_output, avg_time


def perf_triton(input: torch.Tensor, scale_tensor: torch.Tensor, exp_indices: torch.Tensor):
    # prepare the indexes
    splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=args.G).to(torch.int32)
    split_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

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
    scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))
    scattered_scale_tensor.copy_(torch.index_select(scale_tensor, dim=0, index=gather_idx_cur_rank))

    def fwd():
        return fast_all_to_all(all_to_all_ctx, scattered_input, split_cumsum,
                               scattered_scale_tensor if args.with_scale else None)

    torch.cuda._sleep(1000000000)
    # warmup
    for _ in range(20):
        fwd()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    # bench
    st.record()
    for _ in range(args.bench_iters):
        _ = fwd()
    ed.record()
    torch.cuda.synchronize()
    avg_time = st.elapsed_time(ed) / args.bench_iters

    # 1. dispatch
    dispatch_splits, dispatch_token, dispatch_scale = fast_all_to_all(
        all_to_all_ctx, scattered_input, split_cumsum, scattered_scale_tensor if args.with_scale else None)
    dispatch_token, dispatch_scale = all_to_all_post_process(all_to_all_ctx, dispatch_splits, dispatch_token,
                                                             dispatch_scale if args.with_scale else None)

    # 2. compute: moe_compute(dispatch_token, dispatch_scale, moe_weight, ...)
    # ...

    # 3. combine
    combine_splits, combine_token, combine_scale = fast_all_to_all(all_to_all_ctx, dispatch_token,
                                                                   splits_to_cumsum(dispatch_splits), dispatch_scale)
    combine_token, combine_scale = all_to_all_post_process(all_to_all_ctx, combine_splits, combine_token,
                                                           combine_scale if args.with_scale else None)

    # 3.1. reduce: [num_tokens_local_rank * topk] => [num_tokens_local_rank]
    combine_reduced_out = torch.zeros_like(input)
    combine_reduced_out.index_add_(0, gather_idx_cur_rank, combine_token)

    # check the output of `dispatch => => combine`
    torch.testing.assert_close(combine_reduced_out, input * args.topk, rtol=1e-2, atol=1e-2)

    return dispatch_token, dispatch_scale, avg_time


if __name__ == "__main__":
    args = parse_args()
    if args.enable_flux:
        try:
            import flux
        except ImportError:
            raise ImportError("flux is not successfully imported")
    EP_GROUP = initialize_distributed(args.enable_flux)
    assert (args.G % WORLD_SIZE == 0), f"args.G:{args.G} should be divisible by WORLD_SIZE:{WORLD_SIZE}"

    experts_per_rank = args.G // WORLD_SIZE

    all_to_all_ctx = create_all_to_all_context(
        args.M * args.topk,
        args.N,
        RANK,
        args.G,
        WORLD_SIZE,
        experts_per_rank,
        DTYPE_MAP[args.dtype],
        torch.float,
    )
    if args.enable_flux:
        flux_op = flux.All2AllInference(
            args.M * args.topk,
            args.N,
            RANK,
            args.G,
            WORLD_SIZE,
            LOCAL_WORLD_SIZE,
            2,
        )
    for rid in range(args.rounds):
        # random simulate token received from dataloader
        token_num = random.randint(args.M // 2, args.M)

        print(f"Rank-{RANK}: Received {token_num} tokens")

        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk)
        assert exp_indices.size(0) == token_num and exp_indices.size(1) == args.topk
        exp_indices = exp_indices.to("cuda")
        input = (torch.rand(token_num, args.N, dtype=torch.float32).to(DTYPE_MAP[args.dtype]).to("cuda"))
        scale_tensor = torch.rand(token_num, dtype=torch.float32).to("cuda")

        with group_profile(name="all2all_inference", group=EP_GROUP, do_prof=args.profile):
            ref_out, ref_scale, ref_time = perf_torch(input, scale_tensor, exp_indices)
            torch.cuda.synchronize()
            triton_out, triton_scale, triton_time = perf_triton(input, scale_tensor, exp_indices)
            torch.cuda.synchronize()
            if args.enable_flux:
                flux_out, flux_scale, flux_time = perf_flux(input, scale_tensor, exp_indices)
                torch.cuda.synchronize()
        torch.distributed.barrier()

        def gather_benchmark(time_value):
            tensor = torch.tensor(time_value, device="cuda")
            gather_list = ([torch.zeros_like(tensor) for _ in range(WORLD_SIZE)] if RANK == 0 else None)
            torch.distributed.gather(tensor, gather_list, dst=0)
            return [t.item() for t in gather_list] if RANK == 0 else None

        torch_times = gather_benchmark(ref_time)
        triton_times = gather_benchmark(triton_time)
        flux_times = gather_benchmark(flux_time) if args.enable_flux else None

        if RANK == 0:
            print(f"\n=== Round {rid + 1}/{args.rounds} ===")
            headers = ["Rank", "Torch (ms)", "Triton (ms)"]
            if args.enable_flux:
                headers.append("Flux (ms)")
            rows = []
            for rank in range(WORLD_SIZE):
                row = [rank, f"{torch_times[rank]:.3f}", f"{triton_times[rank]:.3f}"]
                if args.enable_flux:
                    row.append(f"{flux_times[rank]:.3f}")
                rows.append(row)
            avg_row = ["Avg"]
            avg_row.append(f"{sum(torch_times)/WORLD_SIZE:.3f}")
            avg_row.append(f"{sum(triton_times)/WORLD_SIZE:.3f}")
            if args.enable_flux:
                avg_row.append(f"{sum(flux_times)/WORLD_SIZE:.3f}")
            rows.append(avg_row)

            print(tabulate(rows, headers=headers, floatfmt=".3f", tablefmt="grid"))

        torch.distributed.barrier()

        def _check(out: torch.Tensor, ref: torch.Tensor, msg: str = "Triton"):
            try:
                torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
                print(f"✅ RANK[{RANK}] check {msg} passed")
            except Exception as e:
                print(f"❌ RANK[{RANK}] check {msg} failed")
                raise e

        _check(triton_out, ref_out, "Triton out")
        if args.enable_flux:
            _check(flux_out, ref_out, "Flux out")
        if args.with_scale:
            _check(triton_scale, ref_scale, "Triton scale")
            if args.enable_flux:
                _check(flux_scale, ref_scale, "Flux scale")

    torch.distributed.destroy_process_group(EP_GROUP)
