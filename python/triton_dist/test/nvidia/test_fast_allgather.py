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

import nvshmem.core
import torch

from triton_dist.layers.nvidia import AllGatherLayer
from triton_dist.profiler_utils import group_profile
from triton_dist.utils import (initialize_distributed, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync,
                               nvshmem_create_tensor, sleep_async)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--stepfactor", default=2, type=int)
    parser.add_argument("-b", "--minbytes", type=int, default=1024 * 4)
    parser.add_argument("-e", "--maxbytes", type=int, default=1024 * 1024 * 32)
    parser.add_argument("--dtype", type=str, default="int32")
    parser.add_argument("--warmup_iters", type=int, default=30)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--verify", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--mode",
        default="pull_1d",
        choices=[
            "push_numa_2d_ll", "push_numa_2d", "push_2d_ll_multimem", "push_2d_ll", "push_2d", "pull_1d",
            "push_numa_2d_ll_multinode", "push_3d", "push_3d_ll"
        ],
    )
    args = parser.parse_args()
    return args


def perf_ag(ag_op: AllGatherLayer, ag_buffer: torch.Tensor, nbytes: int, do_verify: bool):
    nbytes_per_rank = nbytes // WORLD_SIZE
    ref_tensor = torch.arange(nbytes, dtype=dtype).cuda()

    # local copy
    index_start, index_end = nbytes_per_rank * RANK, nbytes_per_rank * (RANK + 1)
    ag_buffer[ag_op.signal_target % ag_op.stages][index_start:index_end].copy_(ref_tensor[index_start:index_end])

    # ag_buffer = ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes]  # only keeps the needed part

    def _run_with_ag_op():
        if args.mode == "push_2d":
            return ag_op.forward_push_2d(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        if args.mode == "push_3d":
            return ag_op.forward_push_3d(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        if args.mode == "push_3d_ll":
            return ag_op.forward_push_3d_ll(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        elif args.mode == "push_2d_ll":
            return ag_op.forward_push_2d_ll(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        elif args.mode == "push_numa_2d":
            return ag_op.forward_push_numa_2d(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        elif args.mode == "push_numa_2d_ll":
            return ag_op.forward_push_numa_2d_ll(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        elif args.mode == "push_numa_2d_ll_multinode":
            return ag_op.forward_push_numa_2d_ll_multinode(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        elif args.mode == "push_2d_ll_multimem":
            return ag_op.forward_push_2d_ll_multimem(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        elif args.mode == "pull_1d":
            return ag_op.forward_pull(ag_buffer[ag_op.signal_target % ag_op.stages][:nbytes])
        else:
            raise ValueError(f"Unknown mode {args.mode}")

    def _verify():
        for i in range(100):
            ref_tensor = torch.randint(0, 9999999, [nbytes // 4], dtype=torch.int32).view(dtype).cuda()
            torch.distributed.broadcast(ref_tensor, src=0)
            ag_buffer[ag_op.signal_target % ag_op.stages][index_start:index_end].copy_(
                ref_tensor[index_start:index_end])
            result = _run_with_ag_op()

            try:
                torch.testing.assert_close(result[:nbytes], ref_tensor, atol=0, rtol=0)
            except Exception as e:
                print(result[:nbytes].view(WORLD_SIZE, -1))
                print(ref_tensor.view(WORLD_SIZE, -1))
                print(f"❌ RANK[{RANK}] check failed")
                raise e
        print(f"✅ RANK[{RANK}] check passed")

    if do_verify:
        _verify()
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    from triton_dist.profiler_utils import perf_func

    with group_profile(f"all_gather_op_{nbytes//1024}KB", do_prof=args.profile, group=TP_GROUP):
        sleep_async(1000)  # in case CPU bound
        _, ag_time_ms = perf_func(
            _run_with_ag_op,
            warmup_iters=warmup_iters,
            iters=iters,
        )

    gbps = (lambda ms: nbytes * 1e-9 / (ms * 1e-3) * (WORLD_SIZE - 1) / WORLD_SIZE)
    print(
        f"RANK = {RANK}, {nbytes // 1024} KB, Latency = {ag_time_ms * 1000:0.2f} us, Bandwith = {gbps(ag_time_ms):0.2f} GB/s"
    )


def align_to(value, alignment):
    return (value + alignment - 1) // alignment * alignment


if __name__ == "__main__":
    args = parse_args()
    dtype = torch.int8
    warmup_iters = args.warmup_iters
    iters = args.iters

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()

    stages = 2

    ag_buffer = nvshmem_create_tensor((stages, args.maxbytes), dtype)

    nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE
    ag_op = AllGatherLayer(nnodes, WORLD_SIZE, RANK, max_buffer_size=args.maxbytes * 2, stages=stages)

    minbytes = align_to(args.minbytes, 16)
    maxbytes = align_to(args.maxbytes, 16)
    nbytes = minbytes
    while nbytes < maxbytes:
        perf_ag(ag_op, ag_buffer, nbytes, args.verify)
        nbytes = args.stepfactor * nbytes

    ag_op.finalize()
    nvshmem_free_tensor_sync(ag_buffer)
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
