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
from functools import partial

import torch
import nvshmem.core
from triton_dist.kernels.nvidia import (create_moe_rs_context, create_moe_rs_context_colwise, moe_reduce_rs_rowise,
                                        select_experts)
from triton_dist.kernels.nvidia.comm_perf_model import estimate_reduce_scatter_time_ms, get_nic_gbps_per_gpu
from triton_dist.kernels.nvidia.gemm_perf_model import get_dram_gbps, get_tensorcore_tflops
from triton_dist.kernels.nvidia.moe_reduce_rs import moe_reduce_rs_colwise
from triton_dist.utils import dist_print, get_intranode_max_speed, group_profile, perf_func, initialize_distributed, TP_GROUP, assert_allclose, sleep_async


def create_rand_tensor(rank, shape, dtype=torch.float16, device="cuda"):
    return (-2 * torch.rand(shape, dtype=dtype, device=device) + 1) / 10 * (rank + 1)


THRESHOLD_MAP = {
    torch.float16: 1e-2,
    torch.bfloat16: 1e-2,
    # torch.float8_e4m3fn: 1e-2,
    # torch.float8_e5m2: 1e-2,
}


def print_sol_time_estimate(M, K, N, E, topk, dtype, WORLD_SIZE, LOCAL_WORLD_SIZE):
    K_per_rank = K // WORLD_SIZE
    flops_per_rank = 2 * M * K * N * topk // WORLD_SIZE
    tflops = get_tensorcore_tflops(dtype)
    tensorcore_ms = flops_per_rank / tflops / 1e9
    dram_gbps = get_dram_gbps()
    memory_read_per_rank = M * K_per_rank * dtype.itemsize + N * K_per_rank * E * dtype.itemsize
    memory_write_per_rank = M * N * dtype.itemsize
    memory_read_ms = memory_read_per_rank / 1e6 / dram_gbps
    memory_write_ms = memory_write_per_rank / 1e6 / dram_gbps
    moe_sol_ms = max(tensorcore_ms, memory_read_ms + memory_write_ms)
    print("  MOE perf estimate")
    print(f"   TensorCore: {flops_per_rank/1e12:0.2f} TFLOPs {tensorcore_ms:0.2f} ms expected")
    print(f"   Memory read: {memory_read_per_rank/1e9:0.2f} GB, {memory_read_ms:0.2f} ms expected")
    print(f"   Memory write: {memory_write_per_rank/1e9:0.2f} GB, {memory_write_ms:0.2f} ms expected")
    print(f"   SOL time: {moe_sol_ms:0.2f} ms")
    print("  ReduceScatter perf estimate")
    intranode_bw = get_intranode_max_speed()
    internode_bw = get_nic_gbps_per_gpu()
    reduce_scatter_sol_ms = estimate_reduce_scatter_time_ms(M * N * dtype.itemsize, WORLD_SIZE, LOCAL_WORLD_SIZE,
                                                            intranode_bw, internode_bw)
    print(f"   SOL time: {reduce_scatter_sol_ms:0.2f} ms")
    print(f" MOE+RS SOL time: {max(reduce_scatter_sol_ms, moe_sol_ms):0.2f} ms")


class TorchGroupGemmReduceRS(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        router_logits: torch.Tensor,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
    ):
        super(TorchGroupGemmReduceRS, self).__init__()
        self.pg = pg
        self.rank = pg.rank()
        self.world_size = pg.size()
        self.max_token_num = max_token_num
        assert (max_token_num %
                self.world_size == 0), f"max_token_num({max_token_num}) % world_size({self.world_size}) != 0"
        self.hidden_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.topk = topk

        assert (intermediate_size %
                self.world_size == 0), f"intermediate_size({intermediate_size}) % world_size({self.world_size}) != 0"
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.router_logits = router_logits

        self.full_topk_ids, self.full_topk_weight = select_experts(self.pg, self.world_size, self.topk,
                                                                   self.input_dtype, self.device, self.router_logits)

        self.rs_buffer: torch.Tensor = torch.zeros(
            [self.max_token_num // self.world_size, self.hidden_dim],
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
        ntokens = num_tokens_topk // self.topk
        topk_ids = self.full_topk_ids[:ntokens].view(-1)
        topk_weight = self.full_topk_weight[:ntokens].view(-1)
        out = final_output_buffer[:num_tokens_topk, :]
        for i in range(self.num_experts):
            mask = topk_ids == i
            if mask.sum():
                out[mask] = (intermediate_states[mask] @ w[i]) * topk_weight[mask, None]
        output = torch.sum(
            out.reshape(ntokens, self.topk, -1),
            dim=1,
            keepdim=False,
        )
        # torch.save(out, f"grouped_gemm_out_torch_{self.rank}.pt")
        # torch.save(output, f"reduced_torch_{self.rank}.pt")
        torch.distributed.reduce_scatter_tensor(
            self.rs_buffer[:ntokens // self.world_size, :],
            output,
            group=self.pg,
        )

        return self.rs_buffer[:ntokens // self.world_size, :]


class MoEReduceRSTensorParallel(torch.nn.Module):

    def __init__(
        self,
        pg: torch.distributed.ProcessGroup,
        local_world_size: int,
        hidden_dim: int,
        intermediate_size: int,
        num_experts: int,
        topk: int,
        router_logits: torch.Tensor,
        max_token_num: int = 16 * 1024,
        input_dtype=torch.float16,
        output_dtype=torch.float16,
        device="cuda",
        moe_block_size=128,
    ):
        super(MoEReduceRSTensorParallel, self).__init__()
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
        self.output_dtype = output_dtype
        assert self.input_dtype == self.output_dtype
        self.device = device

        self.moe_block_size = moe_block_size

        self.router_logits = router_logits

        self.ctx = create_moe_rs_context(
            self.pg,
            self.local_world_size,
            self.max_token_num,
            self.hidden_dim,
            self.num_experts,
            self.topk,
            self.input_dtype,
            self.output_dtype,
            self.device,
            self.moe_block_size,
            self.router_logits,
        )

        self.ctx_colwise = create_moe_rs_context_colwise(
            self.rank,
            self.world_size,
            self.local_world_size,
            self.max_token_num,
            self.hidden_dim,
            self.num_experts,
            self.topk,
            self.input_dtype,
        )

    def forward(self, intermediate_states, w: torch.Tensor):
        assert hasattr(self, "ctx") and self.ctx is not None
        num_tokens_per_rank = self.ctx.precompute_ctx.num_tokens_per_rank
        num_tokens = num_tokens_per_rank * self.world_size

        self.ctx.dataflow_config.RS_BLOCK_M = num_tokens // self.world_size

        output = moe_reduce_rs_rowise(
            self.rank,
            self.world_size,
            self.local_world_size,
            intermediate_states,
            w,
            self.ctx,
        )

        return output

    def forward_colwise(self, intermediate_states, w: torch.Tensor):
        assert hasattr(self, "ctx") and self.ctx is not None
        num_tokens_per_rank = self.ctx.precompute_ctx.num_tokens_per_rank
        num_tokens = num_tokens_per_rank * self.world_size

        self.ctx.dataflow_config.RS_BLOCK_M = num_tokens // self.world_size

        output = moe_reduce_rs_colwise(
            intermediate_states,
            w,
            self.ctx.precompute_ctx.full_topk_ids,
            self.ctx.precompute_ctx.full_topk_weight,
            ctx=self.ctx_colwise,
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

        import triton
        from triton_dist.autotuner import contextual_autotune
        from triton_dist.kernels.nvidia import moe_reduce_rs

        configs = [
            triton.Config({"BLOCK_N": BN, "BLOCK_K": BK}, num_stages=s, num_warps=w)
            for BN in [128, 256]
            for BK in [32, 64]
            for s in [3, 4]
            for w in [4, 8]
        ]
        moe_reduce_rs.kernel_producer_group_gemm_tp_scatter_input = triton.autotune(
            configs=configs, key=["EM", "N", "K_per_rank"])(moe_reduce_rs.kernel_producer_group_gemm_tp_scatter_input)
        moe_reduce_rs_rowise = contextual_autotune(is_dist=True)(moe_reduce_rs_rowise)

    initialize_distributed()
    tp_group = TP_GROUP()
    RANK = tp_group.rank()
    WORLD_SIZE = tp_group.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))

    num_tokens_per_rank = args.M // WORLD_SIZE
    hidden_size = args.N
    intermediate_size = args.K
    num_experts = args.E
    topk = args.TOPK

    max_token_num = 16 * 1024
    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    input_dtype = DTYPE_MAP[args.dtype]
    output_dtype = DTYPE_MAP[args.dtype]
    device = "cuda"

    iters = args.iters
    warmup_iters = args.warmup

    with torch.no_grad():
        router_logits = create_rand_tensor(
            RANK,
            (num_tokens_per_rank, num_experts),
            device="cuda",
            dtype=input_dtype,
        )

        M = num_tokens_per_rank * WORLD_SIZE * topk
        K_per_rank = intermediate_size // WORLD_SIZE
        intermediate_states = create_rand_tensor(
            RANK,
            (M, K_per_rank),
            device="cuda",
            dtype=input_dtype,
        )
        down_weight = create_rand_tensor(
            RANK,
            (num_experts, K_per_rank, hidden_size),
            device="cuda",
            dtype=input_dtype,
        )
        if args.debug:
            intermediate_states.copy_(
                torch.arange(0, M * K_per_rank, device="cuda", dtype=torch.float32).view(M, K_per_rank) // K_per_rank +
                (RANK + 1) / 10)
            intermediate_states.fill_((RANK + 1) / 10)
            for n in range(num_experts):
                down_weight.fill_(n // hidden_size)
            router_logits = torch.arange(0, num_experts, device="cuda", dtype=router_logits.dtype).repeat(
                (num_tokens_per_rank, 1))

        moe_block_size = 128

        module = MoEReduceRSTensorParallel(
            tp_group,
            LOCAL_WORLD_SIZE,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            router_logits,
            max_token_num,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            device=device,
        )

        torch_module = TorchGroupGemmReduceRS(
            tp_group,
            hidden_size,
            intermediate_size,
            num_experts,
            topk,
            router_logits,
            max_token_num,
            input_dtype=input_dtype,
            output_dtype=output_dtype,
            device=device,
        )

        # runs
        output_torch = torch_module.forward(intermediate_states, down_weight)
        output_triton = module.forward(intermediate_states, down_weight)
        output_triton_colwise = module.forward_colwise(intermediate_states, down_weight)
        # import sys
        # sys.exit(0)
        atol = THRESHOLD_MAP[output_dtype]
        rtol = THRESHOLD_MAP[output_dtype]
        assert_allclose(output_triton, output_torch, atol=atol, rtol=rtol)
        assert_allclose(output_triton_colwise, output_torch, atol=atol, rtol=rtol)

        # don't care torch profile
        sleep_async(200)  # in case CPU bound
        torch_output, duration_ms_torch = perf_func(partial(torch_module.forward, intermediate_states, down_weight),
                                                    iters=iters, warmup_iters=warmup_iters)

        with group_profile(f"moe_rs_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile, group=tp_group):
            sleep_async(100)  # in case CPU bound
            output, duration_ms_triton = perf_func(partial(module.forward, intermediate_states, down_weight),
                                                   iters=iters, warmup_iters=warmup_iters)

            sleep_async(100)  # in case CPU bound
            output, duration_ms_triton_colwise = perf_func(
                partial(module.forward_colwise, intermediate_states, down_weight), iters=iters,
                warmup_iters=warmup_iters)

    print_sol_time_estimate(args.M, args.K, args.N, args.E, args.TOPK, input_dtype, WORLD_SIZE, LOCAL_WORLD_SIZE)
    dist_print(f"dist-triton #{RANK} {duration_ms_triton:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"dist-triton-colwise #{RANK} {duration_ms_triton_colwise:0.2f} ms/iter", need_sync=True,
               allowed_ranks=list(range(WORLD_SIZE)))
    dist_print(f"torch #{RANK} {duration_ms_torch:0.2f} ms/iter", need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    module.ctx.finalize()
    module.ctx_colwise.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
