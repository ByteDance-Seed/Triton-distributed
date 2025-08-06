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
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor


@triton.jit
def ring_put(ptr):
    mype = libshmem_device.my_pe()
    npes = libshmem_device.n_pes()
    peer = (mype + 1) % npes
    libshmem_device.int_p(ptr, mype, peer)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    initialize_distributed()

    t = nvshmem_create_tensor((32, ), dtype=torch.int32)
    ring_put[(1, )](t)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    print(f"RANK {RANK}: {t}")
    nvshmem_free_tensor_sync(t)
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
