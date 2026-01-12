from typing import List, Sequence

import sys
import numpy as np
import torch

MXSHMEM_TEAM_INVALID = -1
MXSHMEM_TEAM_WORLD = 0
MXSHMEM_TEAM_WORLD_INDEX = 0
MXSHMEM_TEAM_SHARED = 1
MXSHMEM_TEAM_SHARED_INDEX = 1
MXSHMEMX_TEAM_NODE = 2
MXSHMEM_TEAM_NODE_INDEX = 2
MXSHMEMX_TEAM_SAME_MYPE_NODE = 3
MXSHMEM_TEAM_SAME_MYPE_NODE_INDEX = 3
MXSHMEMI_TEAM_SAME_GPU = 4
MXSHMEM_TEAM_SAME_GPU_INDEX = 4
MXSHMEMI_TEAM_GPU_LEADERS = 5
MXSHMEM_TEAM_GPU_LEADERS_INDEX = 5
MXSHMEM_TEAMS_MIN = 6
MXSHMEM_TEAM_INDEX_MAX = sys.maxsize


def mxshmemx_mcmodule_init(module: np.intp) -> None:
    ...


def mxshmemx_mcmodule_finalize(module: np.intp) -> None:
    ...


def mxshmem_malloc(size: np.uint) -> np.intp:
    ...


def mxshmemx_get_uniqueid() -> bytes:
    ...


def mxshmemx_init_attr_with_uniqueid(rank: np.int32, nranks: np.int32,
                                     unique_id: bytes) -> None:
    ...


def mxshmem_int_p(ptr: np.intp, src: np.int32, dst: np.int32) -> None:
    ...


def mxshmem_barrier_all():
    ...


def mxshmem_barrier_all_on_stream():
    ...


def mxshmem_ptr(ptr, peer):
    ...


# torch related
def mxshmem_create_tensor(shape: Sequence[int],
                          dtype: torch.dtype) -> torch.Tensor:
    ...


def mxshmem_create_tensor_list_intra_node(
        shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    ...
