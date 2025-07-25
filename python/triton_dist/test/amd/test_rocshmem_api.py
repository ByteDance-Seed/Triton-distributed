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
# from triton.language.extra import libshmem_device

import argparse
import os
from typing import Optional
import datetime

from mpi4py import MPI
import numpy as np
from hip  import hip

from functools import partial

# from hip import hip
import triton
import torch
import triton.language as tl
import torch.distributed as dist
import triton_dist.language as dl
from triton.language.extra import libdevice
from triton.language.extra.hip import libdevice  # noqa: F811
from triton_dist.language.extra import libshmem_device
import time
import pyrocshmem
import random

from triton_dist.utils import (get_torch_prof_ctx)

def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result

def test_rocshmem_basic():
    @triton.jit
    def _rocshmem_basic(comm_buf, ctx, mype, npes):
        tl.store(comm_buf, mype)
        comm_buf+=1
        tl.store(comm_buf, npes)

    print("**rocshmem basic start!")
    pyrocshmem.rocshmem_init()

    mype = pyrocshmem.rocshmem_my_pe()

    npes =  pyrocshmem.rocshmem_n_pes()
    peer = (mype + 1) % npes

    print('mype: {} -- num_pes: {}'.format(mype, npes))

    ctx = pyrocshmem.rocshmem_get_device_ctx()
    comm_buf = pyrocshmem.rocshmem_create_tensor((2,), torch.int32)

    _rocshmem_basic[(1, )](comm_buf, ctx, mype, npes)
    print(f"_rocshmem_basic [dl.rank , dl.num_ranks] from pe#{mype}: {comm_buf}")

    pyrocshmem.rocshmem_barrier_all()

    try:
        torch.testing.assert_close(
            comm_buf,
            torch.tensor([mype, npes], dtype=torch.int32,
                         device="cuda")), comm_buf
    except Exception as e:
        print(f" _rocshmem_basic #{mype} failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_basic #{mype} pass")

    nelems_per_rank = 4
    nelems = nelems_per_rank * npes

    comm_buffs = pyrocshmem.rocshmem_create_tensor_list_intra_node([nelems_per_rank],torch.int32)

    comm_buffs[rank].fill_(0)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_buffs], device=torch.cuda.current_device(),
                                requires_grad=False)

    pyrocshmem.rocshmem_barrier_all()

    one = torch.arange(nelems_per_rank, dtype=torch.int32, device=torch.cuda.current_device())
    stream = torch.cuda.current_stream()

    with torch.cuda.stream(stream):
        for i in range(npes):
            remote_rank = (i + mype) %npes
            if remote_rank == i:
                continue
            dst_ptr = comm_buffs[remote_rank].data_ptr()
            src_ptr = one.data_ptr()
            nbytes = nelems_per_rank * one.element_size()
            cp_res = hip.hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                nbytes,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                stream.cuda_stream,
            )

            hip_check(cp_res)
    pyrocshmem.rocshmem_barrier_all()

    print(f"mype#: {mype} comm_buffs: {comm_buffs}")

    try:
        torch.testing.assert_close(
            comm_buffs[peer],
            one)
    except Exception as e:
        print(f" _rocshmem_basic #{mype} - Check tensor_list failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_basic #{mype} - Check tensor_list pass")

    pyrocshmem.rocshmem_finalize()

def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return duration_ms / iters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")

    return parser.parse_args()


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return duration_ms / iters

if __name__ == "__main__":
    # init
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    os.environ["RANK"]  = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
            backend="nccl", init_method="env://")

    TP_GROUP = torch.distributed.new_group(ranks=list(range(torch.distributed.get_world_size())), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    ctx = get_torch_prof_ctx(args.profile)
    
    with ctx:
        perf = perf_func(test_rocshmem_basic, iters=args.iters,
            warmup_iters=args.warmup)

    if args.profile:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", f"manual_run_{os.getpid()}")    
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    print(f"torch #{rank}", perf)
    
    torch.distributed.destroy_process_group()