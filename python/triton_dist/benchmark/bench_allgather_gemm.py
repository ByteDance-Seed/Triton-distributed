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
from pathlib import Path

import torch
import torch.distributed

from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import LAYER_CONFIGS, assert_allclose
from triton_dist.kernels.nvidia import (ag_gemm, create_ag_gemm_context, gemm_persistent, gemm_non_persistent)
from triton_dist.kernels.nvidia.allgather import (cp_engine_producer_all_gather_inter_node,
                                                  cp_engine_producer_all_gather_intra_node)
from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream,
                               wait_until_max_gpu_clock_or_warning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--autotune", action="store_true", default=False)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--dump_csv", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--trans_b", default=True, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args


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
    if args.trans_b:
        B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda").T
    else:
        B = torch.randn([K, N_per_rank], dtype=dtype, device="cuda")

    # in case zero GEMM runs faster than randn GEMM
    torch_ag_buffer = torch.randn([M, K], dtype=dtype, device="cuda")

    def _torch_func():
        return torch_ag_gemm(pg, A, B, torch_ag_buffer)

    ctx = create_ag_gemm_context(A, B, rank, world_size, M, LOCAL_WORLD_SIZE)

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
                all_gather_method=ctx.all_gather_method,
                debug=args.debug,
            )
        else:
            cp_engine_producer_all_gather_inter_node(
                A,
                ctx.symm_workspaces,
                ctx.symm_barriers,
                ctx.barrier_target,
                ctx.rank,
                ctx.num_local_ranks,
                ctx.num_ranks,
                ctx.ag_intranode_stream,
                ctx.ag_internode_stream,
                all_gather_method=ctx.all_gather_method,
                debug=args.debug,
            )

        if ctx.is_multinode:
            current_stream.wait_stream(ctx.ag_internode_stream)
        current_stream.wait_stream(ctx.ag_intranode_stream)

    persistent = torch.cuda.get_device_capability()[0] >= 9

    def _triton_gemm_func():
        if persistent:
            return gemm_persistent(A_gathered, B, ctx=ctx, autotune=args.autotune)
        else:
            return gemm_non_persistent(A_gathered, B, ctx=ctx, autotune=args.autotune)

    def _triton_func():
        return ag_gemm(A, B, ctx=ctx, autotune=args.autotune)

    for i in range(5):
        A.copy_(torch.randn_like(A))
        B.copy_(torch.randn_like(B))
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
        torch.cuda.synchronize()
        C = _triton_func()

    C_golden = _torch_func()

    for i in range(world_size):
        torch.distributed.barrier(pg)
        if rank == i:
            assert_allclose(C_golden, C, atol=1e-3, rtol=1e-3)

    with group_profile(f"ag_gemm_perf_m_{M}_n_{N}_k_{K}_{os.environ['TORCHELASTIC_RUN_ID']}", args.profile,
                       group=TP_GROUP):
        perf_func(_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
        perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, triton_duration_ms = perf_func(_triton_func, iters=args.iters, warmup_iters=args.warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, triton_ag_duration_ms = perf_func(_triton_ag_func, iters=args.iters, warmup_iters=args.warmup_iters)
    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, triton_gemm_duration_ms = perf_func(_triton_gemm_func, iters=args.iters, warmup_iters=args.warmup_iters)
    wait_until_max_gpu_clock_or_warning()
    _, torch_ag_duration_ms = perf_func(lambda: torch.distributed.all_gather_into_tensor(torch_ag_buffer, A, pg),
                                        iters=args.iters, warmup_iters=args.warmup_iters)
    wait_until_max_gpu_clock_or_warning()
    _, torch_gemm_duration_ms = perf_func(lambda: torch.matmul(torch_ag_buffer, B), iters=args.iters,
                                          warmup_iters=args.warmup_iters)

    wait_until_max_gpu_clock_or_warning(torch.cuda.current_device())
    _, torch_duration_ms = perf_func(_torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    dist_print(
        f"Rank {rank} latency (ms): " \
        f"triton total={triton_duration_ms:.2f}, ag_only={triton_ag_duration_ms:.2f}, triton_gemm_only={triton_gemm_duration_ms:.2f}, " \
        f"torch total={torch_duration_ms:.2f}, ag_only={torch_ag_duration_ms:0.2f}, torch_gemm_only={torch_gemm_duration_ms:0.2f} " \
        f"speedup {torch_duration_ms/triton_duration_ms:.2f}",
        need_sync=True, allowed_ranks=list(range(world_size)))

    return triton_duration_ms, torch_duration_ms


if __name__ == "__main__":
    args = parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    TP_GROUP = initialize_distributed()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 8))
    perf_res = []

    for _, config in LAYER_CONFIGS.items():
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
            for model, config in LAYER_CONFIGS.items():
                index = list(LAYER_CONFIGS.keys()).index(model)
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

    finalize_distributed()
