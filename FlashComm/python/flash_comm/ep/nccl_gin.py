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

################################################################################
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
################################################################################

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import flash_comm._C.buffer as _buffer
import flash_comm._C.ep_internode as _ep_inter

_MAX_DISPATCH_PIPELINE_CHUNKS = int(_ep_inter.MAX_DISPATCH_PIPELINE_CHUNKS)
_MAX_COMBINE_PIPELINE_CHUNKS = int(_ep_inter.MAX_COMBINE_PIPELINE_CHUNKS)


def _optional_positive_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}")
    return parsed


def _resolve_pipeline_chunks(name: str, max_value: int) -> int:
    chunks = _optional_positive_int_env(name)
    if chunks is None:
        return 1
    if chunks > max_value:
        raise ValueError(f"{name} must be in [1, {max_value}], got {chunks}")
    return chunks


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def acquire_nccl_gin(group: dist.ProcessGroup, local_world_size: int, *, min_rail_barriers: int,
                     require_gin: bool) -> tuple[tuple[int, int], tuple[int, int]]:
    """Acquire one process-local lease on the EP group's NCCL device communicator.

    Initialization is collective.  Reuse validates the process-global EP
    rank/topology contract; one EP group per process is supported.  The returned
    ranges are the existing EP signal partitions.  One additional GIN context is
    reserved for chunk-plan publication, so its completion signal cannot collide
    with dispatch/combine signals.
    """
    rank = group.rank()
    world_size = group.size()
    if local_world_size <= 0 or world_size % local_world_size != 0:
        raise ValueError("local_world_size must divide the EP world size")

    ep_num_qps = _optional_positive_int_env("FLASH_COMM_EP_NUM_QPS") or 1
    ep_pipeline_config = ()
    if require_gin:
        ep_pipeline_config = (
            _resolve_pipeline_chunks("FLASH_COMM_EP_DISPATCH_PIPELINE_CHUNKS", _MAX_DISPATCH_PIPELINE_CHUNKS),
            _resolve_pipeline_chunks("FLASH_COMM_EP_COMBINE_PIPELINE_CHUNKS", _MAX_COMBINE_PIPELINE_CHUNKS),
        )
    root_global_rank = dist.get_global_rank(group, 0)
    root_config = [(ep_num_qps, *ep_pipeline_config) if rank == 0 else None]
    dist.broadcast_object_list(root_config, src=root_global_rank, group=group)
    local_config = (ep_num_qps, *ep_pipeline_config)
    if local_config != root_config[0]:
        raise ValueError("NCCL device communicator settings must be identical across the "
                         f"EP group: rank {rank} has {local_config}, root has "
                         f"{root_config[0]}")

    if _buffer.nccl_gin_is_initialized():
        actual = (
            _buffer.nccl_gin_rank(),
            _buffer.nccl_gin_nranks(),
            _buffer.nccl_gin_local_world_size(),
        )
        expected = (rank, world_size, local_world_size)
        if actual != expected:
            raise ValueError("Existing NCCL GIN communicator does not match this EP group: "
                             f"rank/nranks/local_world_size={actual}, expected {expected}")
        _buffer.nccl_gin_retain()
    else:
        uid = [_buffer.nccl_gin_get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(uid, src=root_global_rank, group=group)
        nnodes = world_size // local_world_size
        gin_signals = max(
            16,
            _round_up(int(_ep_inter.ep_required_gin_signal_count(nnodes)), 16),
        )
        _buffer.nccl_gin_init(
            uid[0],
            rank,
            world_size,
            local_world_size,
            gin_contexts=ep_num_qps + 1,
            gin_signals=gin_signals,
            rail_barriers=max(16, min_rail_barriers),
            gin_queue_depth=4096,
            gin_connection_type=int(_buffer.NCCL_GIN_CONNECTION_FULL),
            ep_num_qps=ep_num_qps,
        )

    if require_gin and _buffer.nccl_gin_type() == 0:
        _buffer.nccl_gin_release()
        raise RuntimeError("NCCL GIN is required for internode EP but unavailable on this communicator")

    lsa_size = _buffer.nccl_gin_lsa_size()
    lsa_rank = _buffer.nccl_gin_lsa_rank()
    if (lsa_size < local_world_size or (lsa_rank % local_world_size) != (rank % local_world_size)):
        _buffer.nccl_gin_release()
        raise ValueError("NCCL GIN LSA team must cover local_world_size/local_rank")

    nnodes = world_size // local_world_size
    dispatch_range = (0, int(_ep_inter.ep_dispatch_signal_count(nnodes)))
    combine_range = (
        int(_ep_inter.ep_combine_signal_base(nnodes)),
        int(_ep_inter.ep_required_gin_signal_count(nnodes)),
    )
    return dispatch_range, combine_range


def release_nccl_gin(group: dist.ProcessGroup) -> None:
    """Collectively release one lease after the owner has freed its windows."""
    torch.cuda.synchronize()
    dist.barrier(group=group)
    if _buffer.nccl_gin_is_initialized():
        _buffer.nccl_gin_release()
    torch.cuda.synchronize()
    dist.barrier(group=group)
