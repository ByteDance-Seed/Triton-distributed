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

import sys
import numpy as np
import torch

NVSHMEM_TEAM_INVALID = -1
NVSHMEM_TEAM_WORLD = 0
NVSHMEM_TEAM_WORLD_INDEX = 0
NVSHMEM_TEAM_SHARED = 1
NVSHMEM_TEAM_SHARED_INDEX = 1
NVSHMEMX_TEAM_NODE = 2
NVSHMEM_TEAM_NODE_INDEX = 2
NVSHMEMX_TEAM_SAME_MYPE_NODE = 3
NVSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3
NVSHMEMI_TEAM_SAME_GPU = 4
NVSHMEM_TEAM_SAME_GPU_INDEX = 4
NVSHMEMI_TEAM_GPU_LEADERS = 5
NVSHMEM_TEAM_GPU_LEADERS_INDEX = 5
NVSHMEM_TEAMS_MIN = 6
NVSHMEM_TEAM_INDEX_MAX = sys.maxsize


def nvshmemx_cumodule_init(module: np.intp) -> None:
    ...


def nvshmemx_cumodule_finalize(module: np.intp) -> None:
    ...


def nvshmem_malloc(size: np.uint) -> np.intp:
    ...


def nvshmemx_get_uniqueid() -> bytes:
    ...


def nvshmemx_init_attr_with_uniqueid(rank: np.int32, nranks: np.int32, unique_id: bytes) -> None:
    ...


def nvshmem_int_p(ptr: np.intp, src: np.int32, dst: np.int32) -> None:
    ...


def nvshmem_barrier_all():
    ...


def nvshmem_barrier_all_on_stream():
    ...


def nvshmem_ptr(ptr, peer):
    ...


def nvshmemx_mc_ptr(team, ptr):
    """ DON'T CALL this function if NVLS is not used on NVSHMEM 3.2.5!!!
    even nvshmem official doc say that it returns a nullptr(https://docs.nvidia.com/nvshmem/api/gen/api/setup.html?highlight=nvshmemx_mc_ptr#nvshmemx-mc-ptr), it actually core dump without any notice. use this function only when you are sure NVLS is used.
    here is an issue: https://forums.developer.nvidia.com/t/how-do-i-query-if-nvshmemx-mc-ptr-is-supported-nvshmemx-mc-ptr-core-dump-if-nvls-is-not-used/328986
    """
    ...


# torch related
def nvshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    ...


def nvshmem_create_tensor_list_intra_node(shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    ...
