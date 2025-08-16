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
import os

import triton
import torch
from triton_dist.autotuner import contextual_autotune
from triton_dist.kernels.nvidia.moe_reduce_ar import run_moe_reduce_ar_triton_non_overlap, run_moe_reduce_ar, create_moe_ar_context
from triton_dist.utils import dist_print, group_profile, perf_func, initialize_distributed, assert_allclose, sleep_async, finalize_distributed


def create_rand_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return (-2 * torch.rand(shape, dtype=dtype, device=device) + 1) / 10 * (rank + 1)


def select_experts(pg: torch.distributed.ProcessGroup, num_ranks: int, topk: int, dtype: torch.dtype,
                   router_logits_shard: torch.Tensor):
    score = torch.softmax(router_logits_shard, dim=-1)
    local_topk_weight, local_topk_ids = torch.topk(score, topk)
    # do all-gather
    ntokens_per_rank = router_logits_shard.shape[0]
    ntokens = ntokens_per_rank * num_ranks
    full_topk_ids = torch.zeros((ntokens, topk), dtype=torch.int32, device="cuda")
    full_topk_weight = torch.zeros((ntokens, topk), dtype=dtype, device="cuda")
    torch.distributed.all_gather_into_tensor(full_topk_weight, local_topk_weight, group=pg)
    torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids.to(torch.int32), group=pg)
    return full_topk_ids, full_topk_weight


THRESHOLD_MAP = {torch.float16: (1e-2, 1e-2), torch.bfloat16: (1e-2, 1e-2)}


def moe_reduce_ar_torch(x: torch.Tensor, w: torch.Tensor, chosen_experts: torch.Tensor, expert_weight: torch.Tensor,
                        pg: torch.distributed.ProcessGroup):
    M, _ = x.shape
    ntokens, topk = expert_weight.shape
    num_experts, _, hidden_dim = w.shape
    assert ntokens * topk == M
    assert x.shape[1] == w.shape[1]
    grouped_gemm_out = torch.zeros((M, hidden_dim), dtype=x.dtype, device="cuda")
    chosen_experts = chosen_experts.view(-1)
    expert_weight = expert_weight.view(-1)
    for i in range(num_experts):
        mask = chosen_experts == i
        if mask.sum():
            grouped_gemm_out[mask] = (x[mask] @ w[i]) * expert_weight[mask, None]
    out_reduce_topk = torch.sum(grouped_gemm_out.reshape(ntokens, topk, hidden_dim), dim=1, keepdim=False)
    torch.distributed.all_reduce(out_reduce_topk)
    return out_reduce_topk


class MoEReduceARTensorParallel(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        local_world_size: int,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
    ):
        super(MoEReduceARTensorParallel, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.local_world_size = local_world_size
        self.local_rank = self.rank % self.local_world_size
        self.max_token_num = max_token_num
        assert (max_token_num % self.world_size == 0)
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.world_size == 0)
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.input_dtype = input_dtype

        self.ctx = create_moe_ar_context(
            self.rank,
            self.world_size,
            self.local_world_size,
            self.max_token_num,
            self.hidden_dim,
            self.num_experts,
            self.topk,
            self.input_dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        chosen_experts: torch.Tensor,
        expert_weight: torch.Tensor,
    ):
        output = run_moe_reduce_ar(
            x,
            w,
            chosen_experts,
            expert_weight,
            ctx=self.ctx,
            n_chunks=4,
        )
        return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)  # num_tokens
    parser.add_argument("N", type=int)  # hidden_size
    parser.add_argument("K", type=int)  # intermediate_size
    parser.add_argument("E", type=int)  # num_experts
    parser.add_argument("TOPK", type=int)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autotune", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.autotune:
        configs = [
            triton.Config({"BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK}, num_stages=s, num_warps=w)
            for BN in [128]
            for BK in [32, 64]
            for s in [3, 4]
            for w in [4, 8]
        ]
        from triton_dist.kernels.nvidia import moe_reduce_ar
        moe_reduce_ar.moe_grouped_gemm_kernel = triton.autotune(configs=configs,
                                                                key=["M", "N",
                                                                     "K"])(moe_reduce_ar.moe_grouped_gemm_kernel)
        run_moe_reduce_ar_triton_non_overlap = contextual_autotune(is_dist=True)(run_moe_reduce_ar_triton_non_overlap)
        moe_reduce_ar.moe_gather_ar_grouped_gemm_kernel = triton.autotune(configs=configs, key=["M", "N", "K"])(
            moe_reduce_ar.moe_gather_ar_grouped_gemm_kernel)
        run_moe_reduce_ar = contextual_autotune(is_dist=True)(run_moe_reduce_ar)

    tp_group = initialize_distributed(args.seed)
    RANK = tp_group.rank()
    WORLD_SIZE = tp_group.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))

    ntokens = args.M  # this is actually the tokens
    ntokens_per_rank = ntokens // WORLD_SIZE
    hidden_size = args.N
    intermediate_size = args.K
    num_experts = args.E
    topk = args.TOPK

    max_token_num = 64 * 1024
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    iters = args.iters
    warmup_iters = args.warmup

    router_logits = create_rand_tensor(RANK, (ntokens_per_rank, num_experts), device="cuda", dtype=dtype)

    M = ntokens * topk
    K_per_rank = intermediate_size // WORLD_SIZE
    intermediate_states = create_rand_tensor(RANK, (M, K_per_rank), device="cuda", dtype=dtype)
    w = create_rand_tensor(RANK, (num_experts, K_per_rank, hidden_size), device="cuda", dtype=dtype)
    if args.debug:
        intermediate_states.fill_((RANK + 1) / 10)
        for n in range(num_experts):
            w.fill_((n + 1) // hidden_size)
        router_logits = torch.arange(0, num_experts, device="cuda", dtype=router_logits.dtype).repeat(
            (ntokens_per_rank, 1))

    choosed_expert, expert_weight = select_experts(tp_group, WORLD_SIZE, topk, dtype, router_logits)

    module = MoEReduceARTensorParallel(
        tp_group,
        LOCAL_WORLD_SIZE,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        max_token_num,
        input_dtype=dtype,
    )

    func_torch = lambda: moe_reduce_ar_torch(intermediate_states, w, choosed_expert, expert_weight, tp_group)
    func_triton_non_overlap = lambda: run_moe_reduce_ar_triton_non_overlap(intermediate_states, w, choosed_expert,
                                                                           expert_weight)
    func_triton = lambda: module.forward(intermediate_states, w, choosed_expert, expert_weight)

    # runs
    output_torch = func_torch()
    output_triton_non_overlap = func_triton_non_overlap()
    output_triton = func_triton()

    atol, rtol = THRESHOLD_MAP[dtype]
    assert_allclose(output_triton_non_overlap, output_torch, atol=atol, rtol=rtol)
    assert_allclose(output_triton, output_torch, atol=atol, rtol=rtol)

    sleep_async(200)  # in case CPU bound
    torch_output, duration_ms_torch = perf_func(func_torch, iters=iters, warmup_iters=warmup_iters)

    with group_profile("moe_ar_non_overlap", do_prof=args.profile, group=tp_group):
        sleep_async(100)
        output, duration_ms_triton_non_overlap = perf_func(func_triton_non_overlap, iters=iters,
                                                           warmup_iters=warmup_iters)

    with group_profile("moe_ar", do_prof=args.profile, group=tp_group):
        sleep_async(100)
        output, duration_ms_triton = perf_func(func_triton, iters=iters, warmup_iters=warmup_iters)

    dist_print(f"torch #{RANK} {duration_ms_torch:0.2f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"triton non-overlap #{RANK} {duration_ms_triton_non_overlap:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"triton #{RANK} {duration_ms_triton:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))

    module.ctx.finalize()
    finalize_distributed()
