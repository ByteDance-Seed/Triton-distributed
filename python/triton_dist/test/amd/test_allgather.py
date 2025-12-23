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
from triton_dist.kernels.amd.allgather import allgather, allgather_chunked, allgather_chunked_pull, allgather_chunked_pull_fused, allgather_chunked_pull_packed_fused
import torch
import os
from triton_dist.utils import (
    sleep_async,
    initialize_distributed,
    finalize_distributed,
    rand_tensor,
)
from triton_dist.profiler_utils import group_profile, perf_func
from triton_dist.test.utils import assert_bitwise_equal
import torch.distributed
import pyrocshmem
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nbytes_per_rank",
        type=str,
        default="32M",
        help="Data size (int/K/M/G, default 32M)",
    )
    parser.add_argument("--iters", type=int, default=10, help="Number of iterations (default 10)")
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=5,
        help="Number of warmup iterations (default 5)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Enable profiling (default False)",
    )
    parser.add_argument("--stream_priority", "--stream_priority", default=0, type=int,
                        help="stream priority, 0 or -1 supported")
    parser.add_argument("--N", "-N", type=int, default=1376)
    parser.add_argument("--workgroup_per_rank", default=2, type=int)
    return parser.parse_args()


def parse_nbytes(nbytes: str):
    try:
        val = int(nbytes)
        return val
    except Exception:
        nbytes = nbytes.upper()
        # nbytes maybe 1k or 2M or 3g like this
        if nbytes.endswith("K"):
            return int(nbytes[:-1]) * 1024
        elif nbytes.endswith("M"):
            return int(nbytes[:-1]) * 1024 * 1024
        elif nbytes.endswith("G"):
            return int(nbytes[:-1]) * 1024 * 1024 * 1024
        else:
            raise ValueError(f"Unsupported nbytes format: {nbytes}")


if __name__ == "__main__":
    TP_GROUP = initialize_distributed()

    args = parse_args()
    nbytes_per_rank: int = parse_nbytes(args.nbytes_per_rank)
    nbytes_per_rank = (nbytes_per_rank // args.N) * args.N
    N = nbytes_per_rank * TP_GROUP.size()

    stream_priority_min, stream_priority_max = torch.cuda.Stream.priority_range()
    assert stream_priority_max <= args.stream_priority <= stream_priority_min
    stream = torch.cuda.Stream(priority=args.stream_priority)
    torch.cuda.set_stream(stream)

    # this seems does not help
    from hip import hip
    err, = hip.hipSetDeviceFlags(hip.hipDeviceScheduleSpin)
    assert err == 0

    group_barrier = pyrocshmem.rocshmem_create_tensor((TP_GROUP.size(), ), dtype=torch.int32)
    group_barrier.zero_()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    A = rand_tensor((nbytes_per_rank, ), device="cuda", dtype=torch.int8)
    A.fill_(TP_GROUP.rank())
    A_full = pyrocshmem.rocshmem_create_tensor((N, ), dtype=torch.int8)
    A_full_torch = torch.empty_like(A_full)
    grid_barrier = torch.zeros((1024, ), device="cuda", dtype=torch.int32)

    torch.distributed.barrier(TP_GROUP)

    # as warmup
    fn_torch = lambda: torch.distributed.all_gather_into_tensor(A_full_torch, A, TP_GROUP)
    fn_triton = lambda: allgather(A_full, A, group_barrier)
    fn_triton_chunked = lambda: allgather_chunked(A_full.view(-1, args.N), A.view(-1, args.N), group_barrier,
                                                  grid_barrier, SPLIT_N=2)
    fn_triton_chunked_pull = lambda: allgather_chunked_pull(A_full.view(-1, args.N), A.view(-1, args.N), group_barrier,
                                                            grid_barrier, SPLIT_N=2)
    fn_triton_chunkerd_pull_fused = lambda: allgather_chunked_pull_fused(A_full.view(-1, args.N), A.view(-1, args.N),
                                                                         group_barrier, grid_barrier, SPLIT_N=2)
    fn_triton_chunkerd_pull_advanced_fused = lambda: allgather_chunked_pull_packed_fused(
        A_full.view(-1, args.N), A.view(-1, args.N), group_barrier, grid_barrier, SPLIT_N=2)
    triton_fns = {
        "all_gather_push": fn_triton,
        "all_gather_chunked": fn_triton_chunked,
        "all_gather_chunked_read": fn_triton_chunked_pull,
        "all_gather_chunked_read_fused": fn_triton_chunkerd_pull_fused,
        "all_gather_chunked_read_fused_advance": fn_triton_chunkerd_pull_advanced_fused,
    }

    fn_torch()
    for name, fn_triton in triton_fns.items():
        torch.distributed.barrier(TP_GROUP)
        fn_triton()
        assert_bitwise_equal(A_full.view(8, -1, args.N), A_full_torch.view(8, -1, args.N))
        torch.distributed.barrier(TP_GROUP)
        A_full.random_()

    _run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    exp_name = f"allgather_{_run_id}"
    with group_profile(exp_name, do_prof=args.profile, merge_group=TP_GROUP):
        sleep_async(10)
        _, duration_ms_torch = perf_func(fn_torch, args.iters, args.warmup_iters)
        triton_durations_ms = {}
        for name, fn_triton in triton_fns.items():
            sleep_async(10)
            _, triton_durations_ms[name] = perf_func(fn_triton, args.iters, args.warmup_iters)

    bw_gpbs_torch = nbytes_per_rank * (TP_GROUP.size() - 1) / 2**30 / duration_ms_torch * 1e3

    print(f"RANK #{TP_GROUP.rank()} AllGather torch: {duration_ms_torch:0.3f} ms/iter {bw_gpbs_torch:0.1f} GB/s.")
    for name, duration_ms in triton_durations_ms.items():
        bw_gpbs_fn = (nbytes_per_rank * (TP_GROUP.size() - 1) / 2**30 / duration_ms * 1e3)
        print(f" triton {duration_ms:0.3f} ms/iter {bw_gpbs_fn:0.1f} GB/s.")

    finalize_distributed()
