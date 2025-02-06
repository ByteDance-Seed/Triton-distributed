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

import os
from typing import Optional
import torch
import torch.distributed as dist

import triton
import triton.language as tl
import numpy as np
import random

import datetime
import pynvshmem

from triton.distributed.kernels.nvidia import fast_all_to_all, create_all_to_all_context, all_to_all_post_process

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-G", type=int, default=128)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--bench_iters", default=1, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=1, type=int, help="random data round")
    parser.add_argument("--dtype", default="bfloat16", help="data type", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--with_scale", action="store_true")
    return parser.parse_args()


EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))


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


def initialize_distributed():
    global EP_GROUP
    assert EP_GROUP is None, "EP_GROUP has already been initialized"

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert dist.is_initialized()

    EP_GROUP = dist.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    init_seed(seed=RANK)
    pynvshmem.init_nvshmem_by_uniqueid(EP_GROUP)
    torch.cuda.synchronize()
    return EP_GROUP


def generate_random_exp_indices(token_num: int, total_num_experts: int, topk: int):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for tid in range(token_num):
        top_selected = random.sample(exp_list, topk)
        exp_indices.append(top_selected)
    return torch.Tensor(exp_indices).int()


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
    return choosed_experts.flatten().argsort(stable=True).argsort().int().view(choosed_experts.shape)


@triton.jit
def moe_groupgemm_kernel(
    A,
    B,
    C,
    scatter_idx,
    expert_idx,
    M,
    N,
    K,
    E,
    num_valid_tokens,
    A_stride_m,
    A_stride_k,
    B_stride_e,
    B_stride_n,
    B_stride_k,
    C_stride_m,
    C_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)

    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(scatter_idx + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_token[:, None] * A_stride_m + offs_k[None, :] * A_stride_k

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_be = tl.load(expert_idx + pid_m)
    b_ptrs = B + offs_be * B_stride_e + offs_k[:, None] * B_stride_k + offs_bn[None, :] * B_stride_n

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * A_stride_k
        b_ptrs += BLOCK_K * B_stride_k

    accumulator = accumulator.to(tl.float16)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_token[:, None] * C_stride_m + offs_cn[None, :] * C_stride_n
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def groupgemm_torch(
    input: torch.Tensor,
    weight: torch.Tensor,
    scatter_indices: torch.Tensor,
    exp_indices: torch.Tensor,
    with_scale: bool,
    BM: int,
):
    output = torch.empty([input.shape[0], weight.shape[1]], dtype=input.dtype, device=input.device)
    inp_offs, block_offs = 0, 0
    for e in exp_indices.tolist():
        w = weight[e]
        indices = scatter_indices[block_offs:block_offs + BM].tolist()
        a, gather_indices = [], []
        for i in indices:
            if i < input.shape[0]:
                a.append(input[i])
                gather_indices.append(i)

        gather_indices = torch.tensor(gather_indices, dtype=torch.int, device=input.device)
        a = torch.stack(a, dim=0)

        cur_num_tokens = a.shape[0]
        output[gather_indices, :] = a @ w.T

        inp_offs += cur_num_tokens
        block_offs += BM

    return output


def splits_to_cumsum(splits: torch.Tensor):
    out = torch.empty(splits.shape[0] + 1, dtype=splits.dtype, device=splits.device)
    torch.cumsum(splits, 0, out=out[1:])
    out[0] = 0
    return out


def cdiv(a, b):
    return (a + b - 1) // b


def group_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scatter_indices: torch.Tensor,
    exp_indices: torch.Tensor,
    with_scale: bool = False,
    BLOCK_M: int = 128,
):
    """
    scatter_idx: [M, topk]
    """
    assert len(weight.shape) == 3, f"shape of weight should be [LOCAL_E, N, K], but got {weight.shape}"
    assert len(input.shape) == 2, f"shape of input should be [M, K], but got {input.shape}"
    assert input.shape[1] == weight.shape[2], f"input.shape[1]({input.shape[1]}) != weight.shape[2]({weight.shape[2]})"
    assert not with_scale, "with_scale is not supported yet"

    M, K = input.shape
    E, N, K = weight.shape
    EM = scatter_indices.shape[0]

    output = torch.zeros([M, N], dtype=input.dtype, device=input.device)

    # rank_print(f"MNK: {[M, N, K]}")
    # rank_print(f"scatter_indices({scatter_indices.shape}):\n{scatter_indices}")
    # rank_print(f"exp_indices({exp_indices.shape}):\n{exp_indices}")
    # rank_print(f"input({input.shape}):\n{input}")

    grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
    BLOCK_N = 128
    BLOCK_K = 32
    GROUP_M = 8
    moe_groupgemm_kernel[grid](
        input,
        weight,
        output,
        scatter_indices,
        exp_indices,
        EM,
        N,
        K,
        E,
        M,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        GROUP_M,
    )
    return output


def rank_print(s):
    dist.barrier(EP_GROUP)
    for r in range(WORLD_SIZE):
        if r == RANK:
            print(f"RANK-{RANK}(local {LOCAL_RANK}): {s}", flush=True)
        dist.barrier(EP_GROUP)


class DistributedMoELayer:

    def __init__(
        self,
        n_experts: int,
        topk: int,
        hidden: int,  # K
        intermediate: int,  # N
        dtype: torch.dtype,
        with_scale: bool,
        PG: dist.ProcessGroup,
    ):
        assert n_experts % WORLD_SIZE == 0, f"n_experts({n_experts}) must be divisible by WORLD_SIZE({WORLD_SIZE})"
        self.dtype = dtype
        self.tot_experts = n_experts
        self.n_experts = n_experts // WORLD_SIZE
        self.topk = topk
        self.hidden = hidden
        self.intermediate = intermediate
        self.with_scale = with_scale
        self.gate_up_proj = torch.randn([self.n_experts, 2 * intermediate, hidden], dtype=torch.float32,
                                        device="cuda").to(dtype)
        self.down_proj = torch.randn([self.n_experts, hidden, intermediate], dtype=torch.float32,
                                     device="cuda").to(dtype)

        if self.with_scale:
            # assume per-channel quant
            self.gate_up_scale = torch.randn([self.n_experts, 2 * intermediate], dtype=torch.float32, device="cuda")
            self.down_scale = torch.randn([self.n_experts, hidden], dtype=torch.float32, device="cuda")
        else:
            self.gate_up_scale, self.down_scale = None, None

        self.MAX_M = 128 * topk
        self.all2all_ctx = create_all_to_all_context(self.MAX_M, self.hidden, RANK, self.tot_experts, WORLD_SIZE,
                                                     self.n_experts, dtype, torch.float32)

    def simulate_input(self, num_tokens: int,  # local tokens
                       ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        exp_indices = generate_random_exp_indices(num_tokens, self.tot_experts, self.topk).cuda()
        assert exp_indices.size(0) == num_tokens and exp_indices.size(1) == self.topk
        # rank_print(f"exp_indices:\n{exp_indices}")

        # prepare the indexes
        splits_gpu_cur_rank = torch.bincount(exp_indices.view(-1), minlength=self.tot_experts).to(torch.int32)
        split_cumsum = splits_to_cumsum(splits_gpu_cur_rank)

        # calculate the scatter and the gather idx
        scatter_idx_cur_rank = calc_scatter_index_stable(exp_indices)
        gather_idx_cur_rank, _ = calc_gather_index(scatter_idx_cur_rank, 0, num_tokens * self.topk)

        input = torch.randn([num_tokens, self.hidden], dtype=torch.float32, device="cuda").to(self.dtype)
        scattered_input = torch.empty(num_tokens * self.topk, input.size(1), dtype=self.dtype, device=input.device)
        scattered_input.copy_(torch.index_select(input, dim=0, index=gather_idx_cur_rank))

        if self.with_scale:
            scale = torch.randn([num_tokens], dtype=torch.float32, device="cuda")
            scattered_scale = torch.empty(num_tokens * self.topk, dtype=self.dtype, device=scale.device)
            scattered_scale.copy_(torch.index_select(scale, dim=0, index=gather_idx_cur_rank))
        else:
            scattered_scale = None

        return scattered_input, split_cumsum, gather_idx_cur_rank, scattered_scale

    def naive_forward(
        self,
        input: torch.Tensor,
        dispatch_split_cumsum: torch.Tensor,
        gather_idx_cur_rank: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ):
        """
        no-overlap
        """
        assert input.shape[0] <= self.MAX_M, f"input.shape[0]({input.shape[0]}) > MAX_M({self.MAX_M})"
        assert scale is None, "scale is not supported yet"

        # 1. dispatch
        splits, recv_buf, scale_buf = fast_all_to_all(self.all2all_ctx, input, dispatch_split_cumsum, scale)
        dispatched_tokens, dispatched_scale = all_to_all_post_process(self.all2all_ctx, splits, recv_buf, scale_buf)

        # rank_print(f"dispatch_split_cumsum: {dispatch_split_cumsum}")
        # rank_print(f"splits: {splits}")
        # rank_print(f"send tokens: {input.shape}, recv tokens: {dispatched_tokens.shape}")

        # 2. compute
        input_splits = splits.reshape(WORLD_SIZE, -1)
        input_splits_cumsum = splits_to_cumsum(splits)
        input_splits_cumsum_cpu = input_splits_cumsum.cpu().tolist()
        BM = 128
        M_list = input_splits.sum(dim=0).tolist()
        M_list_pad = [cdiv(x, BM) * BM for x in M_list]
        EM = sum(M_list_pad)
        num_blocks = cdiv(EM, BM)

        exp_indices = torch.zeros([num_blocks], dtype=torch.int32, device="cuda")
        expert_idx_full = torch.zeros((EM), dtype=torch.int32, device="cuda")
        scatter_indices = torch.full([EM], EM, dtype=torch.int32, device="cuda")

        # TODO: optimize
        offs_pad, offs = 0, 0
        for e, num_tokens_pad in zip(range(self.n_experts), M_list_pad):
            for i in range(WORLD_SIZE):
                idx = 0 + e + i * self.n_experts
                x = torch.arange(input_splits_cumsum_cpu[idx], input_splits_cumsum_cpu[idx + 1])
                step = x.shape[0]
                scatter_indices[offs:offs + step] = x
                offs += step
            expert_idx_full[offs_pad:offs_pad + num_tokens_pad] = e
            offs_pad += num_tokens_pad
            offs = offs_pad
        exp_indices[torch.arange(num_blocks)] = expert_idx_full[torch.arange(num_blocks) * BM]

        # rank_print(f"splits:\n{input_splits}")
        # rank_print(f"input_splits_cumsum:\n{input_splits_cumsum}")
        # rank_print(f"scatter_indices ({scatter_indices.shape}):\n{scatter_indices}")
        # rank_print(f"exp_indices ({exp_indices.shape}):\n{exp_indices}")

        tmp0 = group_gemm(dispatched_tokens, self.gate_up_proj, scatter_indices, exp_indices, self.with_scale, BM)
        tmp1, tmp1_scale = DistributedMoELayer.act(tmp0, self.with_scale)
        local_out = group_gemm(tmp1, self.down_proj, scatter_indices, exp_indices, self.with_scale, BM)

        _tmp0 = groupgemm_torch(dispatched_tokens, self.gate_up_proj, scatter_indices, exp_indices, self.with_scale, BM)
        # rank_print(f"ref_output({_tmp0.shape}):\n{_tmp0[:16]}")
        _tmp1, _tmp1_scale = DistributedMoELayer.act(_tmp0, self.with_scale)
        _local_out = groupgemm_torch(_tmp1, self.down_proj, scatter_indices, exp_indices, self.with_scale, BM)
        try:
            torch.testing.assert_close(tmp0, _tmp0, rtol=1e-1, atol=1e-1)
            print(f"✅ RANK-{RANK} check pass")
        except Exception as e:
            print(f"❌ RANK-{RANK} check failed")

        # 3. combine
        splits, recv_buf, scale_buf = fast_all_to_all(self.all2all_ctx, local_out, input_splits_cumsum, scale)
        combined_tokens, combined_scale = all_to_all_post_process(self.all2all_ctx, splits, recv_buf, scale_buf)
        # 3.1. reduce: [num_tokens_local_rank * topk] => [num_tokens_local_rank]
        combine_reduced_out = torch.zeros_like(input)
        combine_reduced_out.index_add_(0, gather_idx_cur_rank, combined_tokens)

        return combine_reduced_out

    def forward(
        self,
        input: torch.Tensor,
        dispatch_split_cumsum: torch.Tensor,
        gather_idx_cur_rank: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        impl="naive",
    ):
        if impl == "naive":
            return self.naive_forward(input, dispatch_split_cumsum, gather_idx_cur_rank, scale)
        elif impl == "fused":
            raise NotImplementedError()

    @staticmethod
    def act(t: torch.Tensor, quant: bool):

        def test(t):
            d = t.shape[1] // 2
            return t[:, d:]

        def silu_mul_quant(t):
            pass

        return test(t), None

    @staticmethod
    def quant(t: torch.Tensor):
        assert len(t.shape) == 2
        return torch.randn(t.shape[0], dtype=torch.float32, device=t.device)


if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()

    layer = DistributedMoELayer(args.G, args.topk, args.K, args.N, DTYPE_MAP[args.dtype], args.with_scale, EP_GROUP)
    input = layer.simulate_input(args.M)

    rank_print(
        f"G={args.G}, [K,N]=[{args.K}, {args.N}], tokens_per_rank={args.M}, topk={args.topk}, dtype: {args.dtype} with_scale: {args.with_scale}"
    )

    layer.forward(*input)

    torch.distributed.destroy_process_group(EP_GROUP)
