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

import nvshmem.core
import torch

import triton
import triton_dist.language as dl
from triton_dist.utils import (NVSHMEM_SIGNAL_DTYPE, initialize_distributed, nvshmem_barrier_all_on_stream,
                               nvshmem_create_tensor)


@triton.jit
def test_notify_set(ptr):
    mype = dl.rank()
    npes = dl.num_ranks()
    peer = (mype + 1) % npes
    dl.notify(ptr, peer, signal=mype, sig_op="set", comm_scope="inter_node")


@triton.jit
def test_notify_add(ptr):
    dl.notify(ptr, 0, signal=1, sig_op="add", comm_scope="intra_node")


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()

    t = nvshmem_create_tensor((8, ), NVSHMEM_SIGNAL_DTYPE)
    t.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    test_notify_set[(1, )](t)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    assert t[0].item() == (RANK + WORLD_SIZE - 1) % WORLD_SIZE

    t.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    test_notify_add[(1, )](t)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    ref = WORLD_SIZE if RANK == 0 else 0
    assert t[0].item() == ref

    print(f"RANK {RANK}: pass.")
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
