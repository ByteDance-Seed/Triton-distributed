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
from pathlib import Path
import torch
import torch.distributed

import argparse
import os
import nvshmem.core

from triton_dist.kernels.nvidia import (ag_gemm, create_ag_gemm_context, gemm_persistent)
from triton_dist.kernels.nvidia.allgather import cp_engine_producer_all_gather_intra_node, cp_engine_producer_all_gather_inter_node
from triton_dist.utils import (get_torch_prof_ctx, initialize_distributed, perf_func, dist_print,
                               nvshmem_barrier_all_on_stream)

dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument("--M", type=int, default=8192)
parser.add_argument("--autotune", action="store_true", default=False)
parser.add_argument("--profile", action="store_true", default=False)
parser.add_argument("--dump_csv", action="store_true", default=False)
parser.add_argument("--plot_figure", action="store_true", default=False)
args = parser.parse_args()


def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output


def perf_test(M, config, pg):
    N = config["N"]
    K = config["K"]
    rank = pg.rank()
    world_size = pg.size()

    if rank == 0:
        print(f"test shape: M {M}, N {N}, K {K}")

    assert M % world_size == 0
    assert N % world_size == 0
    M_per_rank = M // world_size
    N_per_rank = N // world_size

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    A_gathered = torch.randn([M, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda")
    torch_ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")

    def _torch_func():
        return torch_ag_gemm(pg, A, B.T, torch_ag_buffer)

    ctx = create_ag_gemm_context(
        A,
        B,
        rank,
        world_size,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        stages=config["stage"],
    )

    def _triton_ag_func():  # this does not include the local copy latency, which is included in ag_gemm
        current_stream = torch.cuda.current_stream()
        nvshmem_barrier_all_on_stream(current_stream)

        if ctx.is_multinode:
            ctx.ag_internode_stream.wait_stream(current_stream)
        ctx.ag_intranode_stream.wait_stream(current_stream)

        if not ctx.is_multinode:
            cp_engine_producer_all_gather_intra_node(
                ctx.rank,
                ctx.num_ranks,
                A,
                ctx.symm_workspaces,
                ctx.symm_barriers,
                ctx.ag_intranode_stream,
                for_correctness=ctx.for_correctness,
                all_gather_method=ctx.all_gather_method,
            )
        else:
            cp_engine_producer_all_gather_inter_node(A, ctx.symm_workspaces, ctx.symm_barriers, ctx.barrier_target,
                                                     ctx.rank, ctx.num_local_ranks, ctx.num_ranks,
                                                     ctx.ag_intranode_stream, ctx.ag_internode_stream,
                                                     for_correctness=ctx.for_correctness,
                                                     all_gather_method=ctx.all_gather_method)

        if ctx.is_multinode:
            current_stream.wait_stream(ctx.ag_internode_stream)
        current_stream.wait_stream(ctx.ag_intranode_stream)

    def _triton_gemm_func():
        return gemm_persistent(A_gathered, B, ctx=ctx, autotune=args.autotune)

    def _triton_func():
        return ag_gemm(A, B, ctx=ctx, autotune=args.autotune)

    for i in range(5):
        A.copy_(torch.randn([M_per_rank, K], dtype=dtype, device="cuda"))
        B.copy_(torch.randn([N_per_rank, K], dtype=dtype, device="cuda"))
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        C = _triton_func()

    C_golden = _torch_func()

    for i in range(world_size):
        torch.distributed.barrier(pg)
        if rank == i:
            if not torch.allclose(C_golden, C, atol=1e-3, rtol=1e-3):
                print(f"Rank {rank}")
                print("Golden")
                print(C_golden)
                print("Output")
                print(C)
                print("Max diff", torch.max(torch.abs(C_golden - C)))
                print("Avg diff", torch.mean(torch.abs(C_golden - C)))
                print("Wrong Answer!")

    profile_ctx = get_torch_prof_ctx(args.profile)
    with profile_ctx:
        perf_func(_triton_func, iters=10, warmup_iters=10)
        perf_func(_torch_func, iters=10, warmup_iters=10)

    if args.profile:
        prof_dir = "prof"
        os.makedirs(prof_dir, exist_ok=True)
        profile_ctx.export_chrome_trace(f"{prof_dir}/trace_ag_gemm_rank_{pg.rank()}_m_{M}_n_{N}_k_{K}.json")

    _, triton_perf = perf_func(_triton_func, iters=100, warmup_iters=500)
    _, triton_ag_perf = perf_func(_triton_ag_func, iters=100, warmup_iters=500)
    _, triton_gemm_perf = perf_func(_triton_gemm_func, iters=100, warmup_iters=500)
    _, torch_perf = perf_func(_torch_func, iters=100, warmup_iters=500)

    dist_print(
        f"Rank {rank} latency (ms): triton={triton_perf:.2f}, ag_only={triton_ag_perf:.2f}, gemm_only={triton_gemm_perf:.2f}, torch={torch_perf:.2f}; speedup {torch_perf/triton_perf:.2f}",
        need_sync=True, allowed_ranks=list(range(world_size)))

    return triton_perf, torch_perf


layer_configs = {
    "LLaMA-7B": {"N": 11008, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "stage": 5},
    "LLaMA-3.1-8B": {"N": 14336, "K": 4096, "BM": 128, "BN": 128, "BK": 64, "stage": 5},
    "LLaMA-3.1-70B": {"N": 28672, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
    "Qwen2-72B": {"N": 29568, "K": 8192, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
    "GPT-3-175B": {"N": 49152, "K": 12288, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
    "LLaMA-3.1-405B": {"N": 53248, "K": 16384, "BM": 128, "BN": 256, "BK": 64, "stage": 3},
}

if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    perf_res = []

    for _, config in layer_configs.items():
        triton_perf, torch_perf = perf_test(args.M, config, TP_GROUP)
        perf_res.append([triton_perf, torch_perf])

    if args.dump_csv and TP_GROUP.rank() == 0:
        if not os.path.exists("csv"):
            os.makedirs("csv")
        csv_file = Path("csv") / f"perf_ag_gemm_{TP_GROUP.size()}_ranks.csv"

        with open(csv_file, "w") as fout:
            print(
                ",".join(
                    map(
                        str,
                        [
                            "Model", "M", "N", "K", "dist-triton ag gemm latency (ms)", "torch ag gemm latency (ms)",
                            "speed up"
                        ],
                    )),
                file=fout,
            )
            for model, config in layer_configs.items():
                index = list(layer_configs.keys()).index(model)
                print(
                    ",".join([model] + list(map(
                        "{:d}".format,
                        [
                            args.M,
                            config["N"],
                            config["K"],
                        ],
                    )) + list(
                        map(
                            "{:02f}".format,
                            [perf_res[index][0], perf_res[index][1], perf_res[index][1] / perf_res[index][0]],
                        ))),
                    file=fout,
                    flush=True,
                )
        print(f"csv file is dumped into {csv_file}")

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
