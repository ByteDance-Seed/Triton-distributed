import sys

import torch
import torch.distributed

try:
    from _pymxshmem import *  # noqa: F403
except Exception as e:
    print(
        "please add mxshmem library path to LD_LIBRARY_PATH and try again",
        flush=True,
        file=sys.stderr,
    )
    raise e


def broadcast_cpu(tensor: torch.Tensor, src: int,
                  group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_mxshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = mxshmemx_get_uniqueid()  # noqa: F405
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    mxshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)  # noqa: F405


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
