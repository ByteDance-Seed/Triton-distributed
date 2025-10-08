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
import datetime
from functools import partial

import triton
import torch
import pyrocshmem
from hip import hip
import triton.language as tl
from triton_dist.utils import (HIP_CHECK, get_torch_prof_ctx, group_profile, perf_func)

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def test_rocshmem_basic():

    @triton.jit
    def _rocshmem_basic(comm_buf, ctx, mype, npes):
        tl.store(comm_buf, mype)
        comm_buf += 1
        tl.store(comm_buf, npes)

    print("**rocshmem basic start!")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()

    ctx = pyrocshmem.rocshmem_get_device_ctx()
    comm_buf = pyrocshmem.rocshmem_create_tensor((2, ), torch.int32)
    torch.cuda.synchronize()
    _rocshmem_basic[(1, )](comm_buf, ctx, mype, npes)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    print(f"mype#: {mype} comm_buffs: {comm_buf}")

    try:
        torch.testing.assert_close(comm_buf, torch.tensor([mype, npes], dtype=torch.int32, device="cuda")), comm_buf
    except Exception as e:
        print(f" _rocshmem_basic #{mype} failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_basic #{mype} pass")


def test_rocshmem_memcpy():
    print("**rocshmem memcpy start!")

    mype = pyrocshmem.rocshmem_my_pe()
    npes = pyrocshmem.rocshmem_n_pes()
    peer = (mype + 1) % npes

    nelems_per_rank = 1024*1024

    comm_buffs = pyrocshmem.rocshmem_create_tensor_list_intra_node([nelems_per_rank], torch.int32)
    comm_buffs[mype].fill_(0)

    torch.cuda.synchronize()

    one = torch.arange(nelems_per_rank, dtype=torch.int32, device=torch.cuda.current_device())
    cur_stream = torch.cuda.current_stream()
    ag_streams = [torch.cuda.Stream(priority=-1) for i in range(npes)]
    
    torch.cuda.synchronize()
    pyrocshmem.rocshmem_barrier_all_on_stream(cur_stream.cuda_stream)

    with torch.cuda.stream(ag_streams[mype]):
        for i in range(npes):
            remote_rank = (i + mype) % npes
            if remote_rank == i:
                continue
            stream = ag_streams[i]
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

            HIP_CHECK(cp_res)

    pyrocshmem.rocshmem_barrier_all_on_stream(cur_stream.cuda_stream)
    
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(comm_buffs[peer], one)
    except Exception as e:
        print(f" _rocshmem_memcpy #{mype} - Check tensor_list failed")
        raise (e)
    else:
        print(f"✅ _rocshmem_memcpy #{mype} - Check tensor_list pass")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")
    parser.add_argument("--warmup", default=5, type=int, help="warmup iterations")
    parser.add_argument("--iters", default=10, type=int, help="perf iterations")

    return parser.parse_args()

if __name__ == "__main__":
    # init
    args = parse_args()
    nbytes = 1024*1024 * 4

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
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    test_rocshmem_basic()
    ctx = group_profile("rshmem_api_profile", args.profile, group=TP_GROUP)

    with ctx:
        perf = perf_func(partial(test_rocshmem_memcpy), iters=10, warmup_iters=5)
   
    torch.cuda.synchronize()
    torch.distributed.barrier()

    print(f"rocSHMEM #{RANK} ", perf)

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()
