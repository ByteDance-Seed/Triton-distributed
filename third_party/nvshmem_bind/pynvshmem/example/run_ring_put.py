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
# torchrun --nproc_per_node=8 --nnodes=1 run_ring_put.py
import datetime
import os

import pynvshmem
import torch
import torch.distributed


def ring_put():
    t = pynvshmem.nvshmem_create_tensor([1024], torch.int)
    print("create torch tensor with nvshmem")
    torch.cuda.synchronize()
    print(t)
    pynvshmem.nvshmem_int_p(t.data_ptr(), TP_GROUP.rank(), (RANK + 1) % WORLD_SIZE)
    print("after put_rank_to_next")
    print(t.to(torch.int32))


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
# use all ranks as tp group
TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")

torch.cuda.synchronize()
pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
ring_put()
