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
from hip import hip
import os
import triton
import torch
import torch.distributed as dist
import time
import pyrocshmem
import triton.language as tl
from triton.language.extra.hip import libdevice

from contextlib import nullcontext

GLOBAL_RANK = int(os.environ["RANK"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', 1))
LOCAL_WORLD_SIZE = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
torch.cuda.set_device(LOCAL_RANK)
# nccl is recommended by pytorch for distributed GPU training
dist.init_process_group(backend="nccl")
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend='nccl')


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


def get_torch_prof_ctx(do_prof: bool):
    ctx = (torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) if do_prof else nullcontext())
    return ctx


@triton.jit
def barrier_all(rank, num_ranks, comm_buf_base_ptrs):
    tid = libdevice.thread_idx(axis=0)  # noqa: F841
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
            pass

    tl.debug_barrier()


@triton.jit
def barrier_all2(rank, num_ranks, comm_buf_base_ptrs, zero_ptr, one_ptr):
    tid = libdevice.thread_idx(axis=0)  # noqa: F841
    for i in range(num_ranks):
        remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
        while libdevice.atom_cas_release_relaxed_system(remote_base_ptr + rank, zero_ptr, one_ptr) == 0:
            pass

    for i in range(num_ranks):
        local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
        while libdevice.atom_cas_acquire_relaxed_system(local_base_ptr + i, one_ptr, zero_ptr) == 0:
            pass

    tl.debug_barrier()


if __name__ == "__main__":
    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)

    profile = True
    torch.cuda.set_device(LOCAL_RANK)
    torch.cuda.synchronize()
    dtype = torch.float16
    signals = pyrocshmem.rocshmem_create_tensor_list_intra_node([
        WORLD_SIZE,
    ], torch.int32)
    if LOCAL_RANK % 4 == 0:
        print(f"dbg signals: {signals}")
    signals[LOCAL_RANK].fill_(0)
    torch.cuda.synchronize()
    dist.barrier()
    ctx = get_torch_prof_ctx(profile)
    zero = torch.zeros([
        8,
    ], dtype=torch.int32).cuda()
    one = torch.ones([
        8,
    ], dtype=torch.int32).cuda()

    signals_ptr = torch.tensor([t.data_ptr() for t in signals]).cuda()
    dist.barrier()
    with ctx:
        dist.barrier()
        time.sleep(LOCAL_RANK)
        barrier_all[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, signals_ptr)
        # barrier_all2[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, signals_ptr, zero, one)

    if profile:
        run_id = os.environ["TORCHELASTIC_RUN_ID"]
        prof_dir = f"prof/{run_id}"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/trace_rank{TP_GROUP.rank()}.json.gz")

    torch.cuda.synchronize()
    dist.barrier()
    print("after sync:", signals[LOCAL_RANK])

    pyrocshmem.rocshmem_finalize()
    dist.destroy_process_group()
