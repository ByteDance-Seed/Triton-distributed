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
import os
import torch
import torch.distributed
import pyrocshmem
from triton_dist.utils import initialize_distributed, finalize_distributed, sleep_async
from triton_dist.kernels.amd.common_ops import barrier_all_kernel, barrier_all_kernel_v2
from triton_dist.profiler_utils import group_profile, perf_func

if __name__ == "__main__":
    TP_GROUP = initialize_distributed()
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE"))

    profile = False
    dtype = torch.float16

    comm = pyrocshmem.rocshmem_create_tensor((WORLD_SIZE, ), torch.int32)
    comm.fill_(0)
    torch.distributed.barrier(group=TP_GROUP)
    torch.cuda.synchronize()

    with group_profile("barrier", do_prof=profile, group=TP_GROUP):
        barrier_all_kernel[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, comm)
        torch.cuda.synchronize()
        print("barrier all passed")
        print(comm)
        torch.distributed.barrier(group=TP_GROUP)

        barrier_all_kernel_v2[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, comm)

        torch.cuda.synchronize()
        print("barrier all v2 passed")
        print(comm)
        torch.distributed.barrier(group=TP_GROUP)

    sleep_async(10)
    fn = lambda: barrier_all_kernel[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, comm)
    _, duration_ms = perf_func(fn, iters=10, warmup_iters=5)
    print(f"barrier_all {duration_ms * 1000:0.2f} us/iter")

    barrier_all_kernel_v2[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, comm)
    torch.cuda.synchronize()

    sleep_async(10)
    fn = lambda: barrier_all_kernel_v2[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, comm)
    _, duration_ms = perf_func(fn, iters=10, warmup_iters=5)
    print(f"barrier_all v2 {duration_ms * 1000:0.2f} us/iter")

    # Explicitly delete rocSHMEM-backed tensors before finalization
    # without explicit cleanup, rocshmem barrier_all collective operation
    # is called during python shutdown when some ranks may already have exited,
    # which may cause segfaults.
    del comm
    finalize_distributed()
