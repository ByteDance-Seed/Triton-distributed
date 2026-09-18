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
"""Shared helpers for internode EP tests (2+ nodes × local GPUs)."""

import os

import torch.distributed as dist

import flash_comm._C.buffer as _buffer
import flash_comm._C.ep_internode as _ep_inter
# Reuse the production env parsing so tests cannot drift from the runtime
# validation rules.
from flash_comm.ep.nccl_gin import _optional_positive_int_env


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def resolve_local_world_size(world_size: int) -> int:
    lws = int(os.environ.get("EP_LOCAL_WORLD_SIZE", "0"))
    if lws <= 0:
        return world_size
    return lws


def init_ep_nccl(ep_group: dist.ProcessGroup, local_world_size: int) -> None:
    rank = dist.get_rank(group=ep_group)
    world = dist.get_world_size(group=ep_group)
    uid = [_buffer.nccl_gin_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(uid, src=0, group=ep_group)
    nnodes = world // local_world_size
    ep_num_qps = _optional_positive_int_env("FLASH_COMM_EP_NUM_QPS") or 1
    gin_contexts = ep_num_qps
    gin_signals = max(
        16,
        _round_up(int(_ep_inter.ep_required_gin_signal_count(nnodes)), 16),
    )
    gin_rail_barriers = 16
    gin_queue_depth = 4096
    _buffer.nccl_gin_init(uid[0], rank, world, local_world_size, gin_contexts=gin_contexts, gin_signals=gin_signals,
                          rail_barriers=gin_rail_barriers, gin_queue_depth=gin_queue_depth,
                          gin_connection_type=int(_buffer.NCCL_GIN_CONNECTION_FULL), ep_num_qps=ep_num_qps)


def destroy_ep_nccl() -> None:
    if _buffer.nccl_gin_is_initialized():
        _buffer.nccl_gin_destroy()
