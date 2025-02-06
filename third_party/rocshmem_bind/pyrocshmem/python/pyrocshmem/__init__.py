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
import sys

import torch
import torch.distributed

try:
    from _pyrocshmem import *  # noqa: F403
except Exception as e:
    print(
        "please add ROCSHMEM library path to LD_LIBRARY_PATH and try again",
        flush=True,
        file=sys.stderr,
    )
    raise e


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


# def init_rocshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
#     rank, nranks = group.rank(), group.size()
#     if rank == 0:
#         unique_id: bytes = rocshmemx_get_uniqueid()  # noqa: F405
#         unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
#     else:
#         unique_id = torch.empty(128, dtype=torch.uint8)
#
#     broadcast_cpu(tensor=unique_id, group=group, src=0)
#
#     unique_id = unique_id.numpy().tobytes()
#     rocshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)  # noqa: F405
