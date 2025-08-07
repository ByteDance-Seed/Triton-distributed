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
from typing import Optional
import datetime

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

def test_rocshmem_device():
    @triton.jit
    def _rocshmem_device(comm_buf, ctx, ptr):
        libshmem_device.set_rocshmem_ctx(ctx)

        mype = dl.rank()
        npes = dl.num_ranks()

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        tl.store(comm_buf, mype)
        comm_buf+=1
        tl.store(comm_buf, npes)
    
    @triton.jit
    def _rocshmem_put(ptr,ctx):
        libshmem_device.set_rocshmem_ctx(ctx)

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        peer = (mype + 1) % npes

        libshmem_device.int_p(ptr, mype, peer)

    @triton.jit
    def _rocshmem_get_put_symm_at(local_ptr, ctx):
        libshmem_device.set_rocshmem_ctx(ctx)

        mype = libshmem_device.my_pe()
        npes = libshmem_device.n_pes()
        pid = tl.program_id(axis=0)
        boffset = pid + tl.arange(0, 4)

        for i in range (1, npes):
            src_rank = (mype + i) % npes
            remote_ptr = dl.symm_at(local_ptr, src_rank)
            rank_offset = src_rank * 4
            val = tl.load(remote_ptr + rank_offset + boffset)
            tl.store(local_ptr + rank_offset + boffset, val)
    
    print("**test_rocshmem_device start!")

    mype = pyrocshmem.rocshmem_my_pe()

    npes =  pyrocshmem.rocshmem_n_pes()
    peer = (mype + 1) % npes

    ctx = pyrocshmem.rocshmem_get_device_ctx()
    comm_buf = pyrocshmem.rocshmem_create_tensor((2,), torch.int32)
    torch.cuda.synchronize()
    _rocshmem_device[(1, )](comm_buf, ctx, comm_buf.data_ptr())

    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    print(f"mype#: {mype} comm_buffs: {comm_buf}")

    try:
        torch.testing.assert_close(
            comm_buf,
            torch.tensor([mype, npes], dtype=torch.int32,
                         device="cuda")), comm_buf
    except Exception as e:
        print(f" _rocshmem_device #{mype} failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_device #{mype} pass")

    put_buf = pyrocshmem.rocshmem_create_tensor((1,), torch.int32)
    torch.cuda.synchronize()
    _rocshmem_put[(1, )](put_buf, ctx)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    print(f"put_buf from pe#{mype}: {put_buf}")
    
    nelems_per_rank = 4
    n_elements = npes*nelems_per_rank
    dtype = torch.int32

    put_bufs = pyrocshmem.rocshmem_create_tensor((n_elements,), torch.int32)
    ref_tensor = torch.arange(n_elements, dtype=dtype).cuda()
    put_bufs[nelems_per_rank * mype : nelems_per_rank *(mype+1)].copy_(ref_tensor[nelems_per_rank * mype : nelems_per_rank *(mype+1)])
    torch.cuda.synchronize()
    torch.distributed.barrier()    
    _rocshmem_get_put_symm_at[(1, )](put_bufs, ctx)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    print(f"put_buf remote_ptr from pe#{mype}: {put_bufs}")

    try:
        torch.testing.assert_close(put_bufs, ref_tensor, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{mype}] check failed")
        raise e
    else:
        print(f"✅ RANK[{mype}] check passed")

def test_rocshmem_basic():
    @triton.jit
    def _rocshmem_basic(comm_buf, ctx, mype, npes):
        tl.store(comm_buf, mype)
        comm_buf+=1
        tl.store(comm_buf, npes)

    print("**rocshmem basic start!")

    mype = pyrocshmem.rocshmem_my_pe()

    npes =  pyrocshmem.rocshmem_n_pes()
    peer = (mype + 1) % npes

    ctx = pyrocshmem.rocshmem_get_device_ctx()
    comm_buf = pyrocshmem.rocshmem_create_tensor((2,), torch.int32)
    torch.cuda.synchronize()
    _rocshmem_basic[(1, )](comm_buf, ctx, mype, npes)

    torch.cuda.synchronize()
    torch.distributed.barrier()
    
    print(f"mype#: {mype} comm_buffs: {comm_buf}")

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

def test_rocshmem_memcpy():
    print("**rocshmem memcpy start!")

    mype = pyrocshmem.rocshmem_my_pe()

    npes =  pyrocshmem.rocshmem_n_pes()
    peer = (mype + 1) % npes

    nelems_per_rank = 4
    nelems = nelems_per_rank * npes
    
    comm_buffs = pyrocshmem.rocshmem_create_tensor_list_intra_node([nelems_per_rank],torch.int32)
    comm_buffs[mype].fill_(0)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_buffs], device=torch.cuda.current_device(),
                                requires_grad=False)

    torch.cuda.synchronize()

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
    torch.cuda.synchronize()
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
    
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(torch.distributed.get_world_size())), backend="nccl")

    torch.distributed.barrier(TP_GROUP)

    num_ranks = torch.distributed.get_world_size()
    rank_id = torch.distributed.get_rank()

    if rank_id==0:
        uid = pyrocshmem.rocshmem_get_uniqueid()
        bcast_obj = [uid]
    else:
        bcast_obj = [None]

    torch.distributed.broadcast_object_list(bcast_obj, src=0)
    torch.distributed.barrier()

    pyrocshmem.rocshmem_init_attr(rank_id, num_ranks, bcast_obj[0])
    
    torch.cuda.synchronize()
    torch.distributed.barrier()

    test_rocshmem_basic()
    
    test_rocshmem_device()

    ctx = get_torch_prof_ctx(args.profile)
    
    with ctx:
        perf = perf_func(partial(test_rocshmem_memcpy), iters=10,
            warmup_iters=5)
    
    torch.cuda.synchronize()
    torch.distributed.barrier()

    if args.profile:
        run_id = os.environ.get("TORCHELASTIC_RUN_ID", f"manual_run_{os.getpid()}")    
        prof_dir = "prof_rshmem"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    print(f"rocSHMEM #{RANK} ", perf)

    pyrocshmem.rocshmem_finalize()

    torch.distributed.destroy_process_group()