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
from typing import List, Sequence

import numpy as np
import torch

ROCSHMEM_TEAM_INVALID = -1
ROCSHMEM_TEAM_WORLD = 0


class symm_rocshmem_buffer:

    def __init__(self, nbytes: int):
        ...

    def data_ptr(self) -> np.intp:
        ...

    def nbytes(self) -> int:
        ...

    def symm_at(self, rank) -> symm_rocshmem_buffer:
        ...

    def __cuda_array_interface__(self) -> dict:
        ...


def rocshmem_n_pes() -> np.int32:
    ...


def rocshmem_my_pe() -> np.int32:
    ...


def rocshmem_team_n_pes(team: np.uintp) -> np.int32:
    ...


def rocshmem_team_my_pe(team: np.int32) -> np.int32:
    ...


def rocshmem_malloc(size: np.uint) -> np.intp:
    ...


def rocshmem_free(ptr: np.intp) -> None:
    ...


def rocshmem_get_uniqueid() -> bytes:
    ...


def rocshmem_init_attr(rank: np.int32, nranks: np.int32,
                       unique_id: bytes) -> None:
    ...


def rocshmem_int_p(ptr: np.intp, src: np.int32, dst: np.int32) -> None:
    ...


def rocshmem_barrier_all():
    ...


def rocshmem_ptr(dest: np.intp, pe: np.int32) -> np.intp:
    ...


## for device side api
def rocshmem_get_device_ctx() -> np.intp:
    ...


# torch related


def rocshmem_create_tensor_list_intra_node(
        shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    ...


def rocshmem_getmem(dest: np.intp, source: np.intp, nelems: int, pe: int):
    ...


def rocshmem_putmem(dest: np.intp, source: np.intp, nelems: int, pe: int):
    ...


## TODO: add host side API
# def rocshmem_putmem_signal(dest: np.intp, source: np.intp, nelems: int, sig_addr: np.intp, signal: int, sig_op: int, pe: int):
#     ...
