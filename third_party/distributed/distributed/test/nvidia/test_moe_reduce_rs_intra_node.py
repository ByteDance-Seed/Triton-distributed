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
import random

import argparse
import os
import datetime
import numpy as np

from typing import List

from functools import partial

import pynvshmem

from triton.distributed.kernels.nvidia import create_moe_rs_context, get_dataflowconfig, moe_reduce_rs_intra_node

from triton.distributed.utils import (
    get_torch_prof_ctx,
    perf_func,
    dist_print,
)


def create_ones_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return torch.ones(shape, dtype=dtype, device=device)


def create_rand_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return (-2 * torch.rand(shape, dtype=dtype, device=device) + 1) / 100 * (rank + 1)


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = pynvshmem.nvshmemx_get_uniqueid()
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    pynvshmem.nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    # torch.bfloat16: 1e-2,
    # torch.float8_e4m3fn: 1e-2,
    # torch.float8_e5m2: 1e-2,
}


class TorchGroupGemmReduceRS(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        full_topk_ids: torch.Tensor,
        full_topk_weight: torch.Tensor,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
    ):
        super(TorchGroupGemmReduceRS, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.num_ranks = pg.size()
        self.max_token_num = max_token_num
        assert (
            max_token_num %
            self.num_ranks == 0), f"max_token_num({max_token_num}) should be multiple of num_ranks({self.num_ranks})"
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.num_ranks == 0
                ), f"intermediate_size({intermediate_size}) should be multiple of num_ranks({self.num_ranks})"
        self.intermediate_size_per_rank = intermediate_size // self.num_ranks

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.full_topk_ids = full_topk_ids.clone()
        self.full_topk_weight = full_topk_weight.clone()

        self.rs_buffer: torch.Tensor = torch.zeros(
            [self.max_token_num // self.num_ranks, self.hidden_dim],
            dtype=self.output_dtype,
            device=self.device,
        )

    def forward(self, intermediate_states, w):
        final_output_buffer = torch.zeros(
            self.max_token_num * self.topk,
            self.hidden_dim,
            dtype=self.output_dtype,
            device=self.device,
        )
        num_tokens_topk, intermediate_size_per_rank = intermediate_states.shape
        topk_ids = self.full_topk_ids[:num_tokens_topk // self.topk].view(-1)
        topk_weight = self.full_topk_weight[:num_tokens_topk // self.topk].view(-1)
        out = final_output_buffer[:num_tokens_topk, :]
        for i in range(self.num_experts):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = (intermediate_states[mask] @ w[i]) * topk_weight[mask, None]
        output = torch.sum(
            out.reshape(num_tokens_topk // self.topk, self.topk, -1),
            dim=1,
            keepdim=False,
        )
        torch.distributed.reduce_scatter_tensor(
            self.rs_buffer[:num_tokens_topk // self.topk // self.num_ranks, :],
            output,
            group=self.pg,
        )

        return self.rs_buffer[:num_tokens_topk // self.topk // self.num_ranks, :]


class MoEReduceRSTensorParallelIntraNode(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        ctx,
        full_topk_ids: torch.Tensor,
        full_topk_weight: torch.Tensor,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        moe_block_size=128,
        no_intermediate_scatter=False,
        debug_sync=False,
    ):
        super(MoEReduceRSTensorParallelIntraNode, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.num_ranks = pg.size()
        self.max_token_num = max_token_num
        assert (
            max_token_num %
            self.num_ranks == 0), f"max_token_num({max_token_num}) should be multiple of num_ranks({self.num_ranks})"
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size % self.num_ranks == 0
                ), f"intermediate_size({intermediate_size}) should be multiple of num_ranks({self.num_ranks})"
        self.intermediate_size_per_rank = intermediate_size // self.num_ranks

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.moe_block_size = moe_block_size
        self.no_intermediate_scatter = no_intermediate_scatter
        self.debug_sync = debug_sync

        self.ctx = ctx
        self.full_topk_ids = full_topk_ids.clone()
        self.full_topk_weight = full_topk_weight.clone()

        # From prepare_kernels

        self.rs_buffers: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
            [self.max_token_num, self.hidden_dim], self.input_dtype)
        self.rs_buffer_ptrs: torch.Tensor = torch.tensor([t.data_ptr() for t in self.rs_buffers], device=self.device)

        self.max_blocks = 65536
        self.sync_buf = pynvshmem.nvshmem_create_tensor([self.max_blocks * self.num_ranks], torch.int32)
        self.sync_buf.fill_(0)

        # stream
        self.scatter_stream = torch.cuda.Stream()
        self.reduce_stream = torch.cuda.Stream()

        # Setup metadata for kernel launch
        RS_BLOCK_M = self.max_token_num // self.num_ranks
        RS_BLOCK_N = self.hidden_dim
        GEMM_BLOCK_M = self.moe_block_size
        GEMM_BLOCK_N = 128
        GEMM_BLOCK_K = 32

        self.dataflow_config = get_dataflowconfig(GEMM_BLOCK_M, GEMM_BLOCK_N, GEMM_BLOCK_K, 8, 4, 4, RS_BLOCK_M,
                                                  RS_BLOCK_N)

        # initialize barriers
        with torch.device(torch.cuda.current_device()):

            # gemm_scatter

            self.barriers_gemm_scatter_counter: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
                [self.num_ranks, 1], torch.int32)

            self.barriers_gemm_scatter_counter_ptrs = torch.tensor(
                [ptr.data_ptr() for ptr in self.barriers_gemm_scatter_counter]).cuda()

            self.barriers_gemm_scatter_ready: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
                [self.num_ranks, 1], torch.uint64)

            self.barriers_gemm_scatter_ready_ptrs = torch.tensor(
                [ptr.data_ptr() for ptr in self.barriers_gemm_scatter_ready]).cuda()

            self.barrier_gemm_scatter_counter = self.barriers_gemm_scatter_counter[self.rank]
            self.barrier_gemm_scatter_ready = self.barriers_gemm_scatter_ready[self.rank]

            self.barrier_gemm_scatter_counter.zero_()
            self.barrier_gemm_scatter_ready.zero_()

            # scatter_reduce

            self.barriers_scatter_reduce_counter: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
                [self.num_ranks, 1], torch.int32)

            self.barriers_scatter_reduce_counter_ptrs = torch.tensor(
                [ptr.data_ptr() for ptr in self.barriers_scatter_reduce_counter]).cuda()

            self.barriers_scatter_reduce_ready: List[torch.Tensor] = pynvshmem.nvshmem_create_tensor_list_intra_node(
                [self.num_ranks, 1], torch.uint64)

            self.barriers_scatter_reduce_ready_ptrs = torch.tensor(
                [ptr.data_ptr() for ptr in self.barriers_scatter_reduce_ready]).cuda()

            self.barrier_scatter_reduce_counter = self.barriers_scatter_reduce_counter[self.rank]
            self.barrier_scatter_reduce_ready = self.barriers_scatter_reduce_ready[self.rank]

            self.barrier_scatter_reduce_counter.zero_()
            self.barrier_scatter_reduce_ready.zero_()

    def forward(self, intermediate_states, w):
        final_output_buffer = torch.zeros(
            (self.max_token_num * self.topk, self.hidden_dim),
            dtype=self.output_dtype,
            device=self.device,
        )
        assert hasattr(self, "ctx") and self.ctx is not None
        num_tokens_per_rank = self.ctx.num_tokens_per_rank
        num_tokens = num_tokens_per_rank * self.num_ranks

        self.dataflow_config.RS_BLOCK_M = num_tokens // self.num_ranks

        moe_reduce_rs_intra_node(
            self.rank,
            self.num_ranks,
            intermediate_states,
            self.full_topk_ids,
            self.full_topk_weight,
            w,
            final_output_buffer,
            self.rs_buffers,
            self.rs_buffer_ptrs,
            self.scatter_stream,
            self.reduce_stream,
            self.barrier_gemm_scatter_counter,
            self.barriers_gemm_scatter_ready_ptrs,
            self.barrier_gemm_scatter_ready,
            self.barrier_scatter_reduce_counter,
            self.barriers_scatter_reduce_ready_ptrs,
            self.barrier_scatter_reduce_ready,
            self.sync_buf,
            self.dataflow_config,
            self.ctx,
            dump_ir=False,
            debug_sync=self.debug_sync,
            bypass_comm=False,
            no_input_scatter=self.no_intermediate_scatter,
        )
        # barrier_all_on_stream(self.rank, self.num_ranks, self.sync_buf, torch.cuda.current_stream())
        return final_output_buffer[self.rank * num_tokens_per_rank:(self.rank + 1) * num_tokens_per_rank, :]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("M", type=int)  # num_tokens
    parser.add_argument("N", type=int)  # hidden_size
    parser.add_argument("K", type=int)  # intermediate_size
    parser.add_argument("E", type=int)  # num_experts
    parser.add_argument("TOPK", type=int)
    parser.add_argument("--warmup", default=10, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=100, type=int, help="perf iterations")

    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--check", default=False, action="store_true", help="correctness check")
    parser.add_argument("--debug_sync", default=False, action="store_true", help="sync between compute and comm")
    parser.add_argument("--no_intermediate_scatter", default=False, action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    NNODES = WORLD_SIZE // LOCAL_WORLD_SIZE

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    os.environ["NCCL_DEBUG"] = "ERROR"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)  # True or False
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)
    random.seed(args.seed)

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    # bench_moe_reduce_rs_tensor_parallel(TP_GROUP, args.M // WORLD_SIZE, args.N, args.K, args.E, args.TOPK, args)
    num_tokens_per_rank = args.M // WORLD_SIZE
    hidden_size = args.N
    intermediate_size = args.K
    num_experts = args.E
    topk = args.TOPK

    max_token_num = 16 * 1024
    input_dtype = torch.float16
    output_dtype = torch.float16
    device = "cuda"

    iters = args.iters
    warmup_iters = args.warmup

    rank = TP_GROUP.rank()
    world_size = TP_GROUP.size()

    no_intermediate_scatter = args.no_intermediate_scatter
    debug_sync = args.debug_sync
    check = args.check

    with torch.no_grad():
        router_logits = create_rand_tensor(
            rank,
            (num_tokens_per_rank, num_experts),
            device="cuda",
            dtype=input_dtype,
        )

        moe_block_size = 128

        ctx = create_moe_rs_context(
            TP_GROUP,
            world_size,
            num_experts,
            topk,
            input_dtype,
            device,
            moe_block_size,
            router_logits,
        )

        module = MoEReduceRSTensorParallelIntraNode(
            TP_GROUP,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            ctx,
            ctx.full_topk_ids,
            ctx.full_topk_weight,
            max_token_num,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            device=device,
            no_intermediate_scatter=no_intermediate_scatter,
            debug_sync=debug_sync,
        )

        torch_module = TorchGroupGemmReduceRS(
            TP_GROUP,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            ctx.full_topk_ids,
            ctx.full_topk_weight,
            max_token_num,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            device=device,
        )

        if not check:
            intermediate_states = create_rand_tensor(
                rank,
                (num_tokens_per_rank * world_size * topk, intermediate_size // world_size),
                device="cuda",
                dtype=input_dtype,
            )
            down_weight = create_rand_tensor(
                rank,
                (num_experts, intermediate_size // world_size, hidden_size),
                device="cuda",
                dtype=input_dtype,
            )
        else:
            intermediate_states = create_ones_tensor(
                rank,
                (num_tokens_per_rank * world_size * topk, intermediate_size // world_size),
                device="cuda",
                dtype=input_dtype,
            )
            down_weight = create_ones_tensor(
                rank,
                (num_experts, intermediate_size // world_size, hidden_size),
                device="cuda",
                dtype=input_dtype,
            )

        prof_ctx = get_torch_prof_ctx(args.profile)
        with prof_ctx:
            torch_output, torch_perf = perf_func(partial(torch_module.forward, intermediate_states, down_weight),
                                                 iters=iters, warmup_iters=warmup_iters)

            pynvshmem.nvshmem_barrier_all()
            torch.cuda.synchronize()

            output, perf = perf_func(partial(module.forward, intermediate_states, down_weight), iters=iters,
                                     warmup_iters=warmup_iters)

        pynvshmem.nvshmem_barrier_all()
        torch.cuda.synchronize()

    if args.profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        prof_ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    if check:
        atol = THRESHOLD_MAP[output_dtype]
        rtol = THRESHOLD_MAP[output_dtype]
        torch.testing.assert_close(output, torch_output, atol=atol, rtol=rtol)
        torch.cuda.synchronize()

    dist_print(f"dist-triton #{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK}", torch_perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    torch.distributed.destroy_process_group()
