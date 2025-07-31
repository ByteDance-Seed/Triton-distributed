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
from typing import Sequence

import torch
import torch.distributed
from hip import hip

try:
    from _pyrocshmem import rocshmem_malloc, rocshmem_my_pe, rocshmem_free
    from _pyrocshmem import *  # noqa: F403
except Exception as e:
    print(
        "please add ROCSHMEM library path to LD_LIBRARY_PATH and try again",
        flush=True,
        file=sys.stderr,
    )
    raise e

class SymmHeap:
    def __init__(self, ptr, nbytes, dtype: torch.dtype,own_data: bool = True):
        self.ptr = ptr
        self.nbytes = nbytes
        self.dtype = dtype
        self._device = torch.cuda.current_device()
        self.own_data = own_data
        self.__cuda_array_interface__ = {
            "data": (self.ptr, False),
            "shape": tuple((self.nbytes, )),
            "typestr": "<i1",  # uint8 data type
            "strides": None,  # Contiguous memory
            "version": 3,
        }
        self.hip = hip
        self.rocshmem_free = rocshmem_free

    # def __del__(self):
        # if self.own_data:

        #     err, device = self.hip.hipGetDevice()
        #     assert err == self.hip.hipError_t.hipSuccess, f"hipError: {err}"
        #     if device != self._device:
        #         err, = self.hip.hipSetDevice(self._device)
        #         assert err == self.hip.hipError_t.hipSuccess, f"hipError: {err}"
        #     # sync
        #     err, = self.hip.hipDeviceSynchronize()
        #     assert err == self.hip.hipError_t.hipSuccess, f"hipError: {err}"

        #     self.rocshmem_free(self.ptr)

        #     # sync
        #     err, = self.hip.hipDeviceSynchronize()
        #     assert err == self.hip.hipError_t.hipSuccess
        #     # set device back
        #     if device != self._device:
        #         err, = self.hip.hipSetDevice(device)
        #         assert err == self.hip.hipError_t.hipSuccess, f"hipError: {err}"

        #     self.own_data = False

def symm_heap_tensor(tensor: torch.Tensor, peer: int) -> torch.Tensor:
    if peer == rocshmem_my_pe():
        return tensor
    ptr=rocshmem_ptr(tensor.data_ptr(), peer)
    buffer = SymmHeap(ptr,nbytes=tensor.nbytes, dtype=tensor.dtype, own_data=False)
    t = torch.as_tensor(buffer, device="cuda").view(tensor.dtype).view(tensor.shape)
    return  t


def rocshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    nbytes = torch.Size(shape).numel() * dtype.itemsize
    ptr=rocshmem_malloc(nbytes)
    buffer = SymmHeap(ptr, nbytes=nbytes, dtype=dtype, own_data=True)
    t = torch.as_tensor(buffer,device="cuda").view(dtype).view(shape)
    setattr(t, "__symm_tensor__", True)
    return t

# ROCSHMEM_TEAM_WORLD = 2
def rocshmem_create_tensor_list_intra_node(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    t = rocshmem_create_tensor(shape, dtype)
    # local_rank = rocshmem_team_my_pe(ROCSHMEM_TEAM_WORLD)
    rank = rocshmem_my_pe()
    npes = rocshmem_n_pes()
    # rank_offset = rank - local_rank
    return [symm_heap_tensor(t, i) for i in range(npes)]

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
