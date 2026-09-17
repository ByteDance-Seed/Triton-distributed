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

import dataclasses
import os

import torch
import torch.distributed as dist

import flash_comm._C.ep_chunk_plan as _chunk_plan

from flash_comm.buffer import SymmetricTensor

from .nccl_gin import acquire_nccl_gin, release_nccl_gin


@dataclasses.dataclass(frozen=True)
class EPChunkPlan:
    """Caller-owned fixed-Q plan and exchanged routing prefixes."""

    logical_token_ranges: torch.Tensor  # [Q, 2]
    rank_chunk_prefix: torch.Tensor  # [W, Q + 1, E + 1]
    chunk_size: int
    max_num_tokens: int
    recv_capacity_tokens: int
    expert_alignment: int
    world_size: int
    topk: int
    num_experts: int

    @property
    def max_steps(self) -> int:
        return self.logical_token_ranges.shape[0]


def _validate_build_inputs(topk_indices: torch.Tensor, *, max_num_tokens: int, chunk_size: int,
                           recv_capacity_tokens: int, num_experts: int, expert_alignment: int,
                           world_size: int) -> tuple[int, int]:
    if (topk_indices.dtype != torch.int32 or not topk_indices.is_cuda or topk_indices.dim() != 2
            or not topk_indices.is_contiguous()):
        raise ValueError("topk_indices must be a contiguous rank-2 CUDA int32 tensor")
    if max_num_tokens <= 0 or chunk_size <= 0 or recv_capacity_tokens <= 0:
        raise ValueError("max_num_tokens, chunk_size, and recv_capacity_tokens must be positive")
    if topk_indices.shape[0] > max_num_tokens:
        raise ValueError("topk_indices exceeds max_num_tokens")

    int32_max = torch.iinfo(torch.int32).max
    if (max_num_tokens > int32_max or chunk_size > int32_max or recv_capacity_tokens > int32_max
            or expert_alignment > int32_max):
        raise ValueError("chunk-plan scalar inputs must fit int32")

    topk = topk_indices.shape[1]
    if world_size <= 0 or world_size > 32:
        raise ValueError("chunk plan requires an EP world size in [1, 32]")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if (num_experts <= 0 or num_experts > 1024 or num_experts % world_size != 0):
        raise ValueError("num_experts must be in [1, 1024] and divisible by EP world size")
    if expert_alignment <= 0:
        raise ValueError("expert_alignment must be positive")

    num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
    if (2 * num_chunks + 2 > int32_max or (num_chunks + 1) * (num_experts + 1) > int32_max):
        raise ValueError("chunk-plan metadata exceeds int32 indexing")

    max_chunk_tokens = min(chunk_size, max_num_tokens)
    assignments = world_size * max_chunk_tokens * topk
    experts_per_rank = num_experts // world_size
    # Zero-count experts add no padding.  With at most ``assignments`` routes
    # delivered to one target rank, at most min(experts_per_rank, assignments)
    # local experts can be non-empty; each contributes at most alignment - 1
    # padding tokens.  Using experts_per_rank unconditionally is safe but would
    # reject valid small-chunk capacities unnecessarily.
    guaranteed_capacity = (assignments + min(experts_per_rank, assignments) * (expert_alignment - 1))
    if recv_capacity_tokens < guaranteed_capacity:
        raise ValueError("recv_capacity_tokens cannot guarantee one logical chunk fits: "
                         f"got {recv_capacity_tokens}, require at least {guaranteed_capacity}")

    max_global_routes = world_size * max_num_tokens * topk
    max_aligned_footprint = (max_global_routes + min(experts_per_rank, max_global_routes) * (expert_alignment - 1))
    if max_global_routes > int32_max or max_aligned_footprint > int32_max:
        raise ValueError("maximum receive footprint must fit int32")

    return num_chunks, topk


def _resolve_launch_num_sms(launch_num_sms: int | None, *, device: torch.device) -> int:
    if launch_num_sms is None:
        env_value = os.environ.get("FLASH_COMM_CHUNK_PLAN_NUM_SMS")
        if env_value:
            try:
                launch_num_sms = int(env_value)
            except ValueError as exc:
                raise ValueError("FLASH_COMM_CHUNK_PLAN_NUM_SMS must be an integer when set") from exc
    if launch_num_sms is None:
        return 0
    if launch_num_sms <= 0:
        raise ValueError("launch_num_sms must be positive when specified")
    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if launch_num_sms > device_sms:
        raise ValueError(f"launch_num_sms exceeds device SM count: got {launch_num_sms}, device has {device_sms}")
    return int(launch_num_sms)


class EPChunkPlanner:
    """Reusable NCCL communication workspace for stateless chunk-plan builds.

    The planner owns only a fixed-shape symmetric scratch buffer and one lease
    on the process-global NCCL device communicator.  Every ``build`` allocates
    and returns independent output tensors; no routing value, schedule, epoch,
    or returned plan is retained by the planner.  Builds are collective and
    must be issued in the same serialized order on every rank because they
    reuse the one communication workspace.
    """

    def __init__(self, *, max_num_tokens: int, chunk_size: int, num_experts: int, group: dist.ProcessGroup,
                 local_world_size: int = 0):
        if max_num_tokens <= 0 or chunk_size <= 0:
            raise ValueError("max_num_tokens and chunk_size must be positive")
        int32_max = torch.iinfo(torch.int32).max
        if max_num_tokens > int32_max or chunk_size > int32_max:
            raise ValueError("max_num_tokens and chunk_size must fit int32")
        self.group = group
        self.world_size = group.size()
        if self.world_size <= 0 or self.world_size > 32:
            raise ValueError("chunk plan requires an EP world size in [1, 32]")
        env_local_world_size = int(os.environ.get("EP_LOCAL_WORLD_SIZE", os.environ.get("LOCAL_WORLD_SIZE", "0")))
        self.local_world_size = (local_world_size or env_local_world_size or self.world_size)
        if (self.local_world_size <= 0 or self.world_size % self.local_world_size != 0):
            raise ValueError("local_world_size must divide the EP world size")
        if (num_experts <= 0 or num_experts > 1024 or num_experts % self.world_size != 0):
            raise ValueError("num_experts must be in [1, 1024] and divisible by EP world size")

        self.max_num_tokens = max_num_tokens
        self.chunk_size = chunk_size
        self.num_experts = num_experts
        self.num_chunks = (max_num_tokens + chunk_size - 1) // chunk_size
        if (2 * self.num_chunks + 2 > int32_max or (self.num_chunks + 1) * (num_experts + 1) > int32_max):
            raise ValueError("chunk-plan metadata exceeds int32 indexing")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._finalized = False
        self._has_nccl_gin_lease = False
        self._workspace: SymmetricTensor | None = None
        self._workspace_tensor: torch.Tensor | None = None
        self._workspace_win_handle = 0

        # The status tensor is allocated before the large symmetric workspace so
        # a rank-local workspace OOM can still coordinate teardown with peers.
        init_status = torch.ones((1, ), dtype=torch.int32, device=self.device)
        device_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        acquire_nccl_gin(group, self.local_world_size, min_rail_barriers=device_sms, require_gin=False)
        self._has_nccl_gin_lease = True
        init_error: Exception | None = None
        try:
            workspace_numel = int(_chunk_plan.workspace_numel(self.num_chunks, self.num_experts, self.world_size))
            self._workspace = SymmetricTensor((workspace_numel, ), torch.int32, group, backend="nccl",
                                              local_world_size=self.local_world_size)
            self._workspace_tensor = self._workspace.get_local_tensor()
            self._workspace_win_handle = self._workspace.get_window_handle()
        except Exception as exc:
            init_error = exc

        init_status.fill_(0 if init_error is not None else 1)
        dist.all_reduce(init_status, op=dist.ReduceOp.MIN, group=group)
        if int(init_status.item()) == 0:
            # Every rank has now observed the same failure state.  Synchronize
            # before freeing any successfully allocated peer-visible window,
            # then release the NCCL GIN lease collectively.
            torch.cuda.synchronize()
            dist.barrier(group=group)
            if self._workspace is not None:
                self._workspace_tensor = None
                self._workspace.free(group)
                self._workspace = None
                self._workspace_win_handle = 0
            if self._has_nccl_gin_lease:
                release_nccl_gin(group)
                self._has_nccl_gin_lease = False
            if init_error is not None:
                raise init_error
            raise RuntimeError("EPChunkPlanner workspace initialization failed on another rank")

    def build(self, topk_indices: torch.Tensor, *, recv_capacity_tokens: int, expert_alignment: int = 1,
              launch_num_sms: int | None = None) -> EPChunkPlan:
        """Build one newly allocated plan from the current rank's routing."""
        if self._finalized or self._workspace_tensor is None:
            raise RuntimeError("EPChunkPlanner has been finalized")
        if topk_indices.device != self.device:
            raise ValueError(f"topk_indices must be on planner device {self.device}, got {topk_indices.device}")
        num_chunks, topk = _validate_build_inputs(
            topk_indices,
            max_num_tokens=self.max_num_tokens,
            chunk_size=self.chunk_size,
            recv_capacity_tokens=recv_capacity_tokens,
            num_experts=self.num_experts,
            expert_alignment=expert_alignment,
            world_size=self.world_size,
        )

        launch_num_sms = _resolve_launch_num_sms(launch_num_sms, device=self.device)

        logical_token_ranges = torch.empty((num_chunks, 2), dtype=torch.int32, device=self.device)
        rank_chunk_prefix = torch.empty((self.world_size, num_chunks + 1, self.num_experts + 1), dtype=torch.int32,
                                        device=self.device)
        _chunk_plan.build_ep_chunk_plan_out(topk_indices, self.num_experts, self.chunk_size, self.max_num_tokens,
                                            recv_capacity_tokens, expert_alignment, launch_num_sms,
                                            self._workspace_tensor, self._workspace_win_handle, logical_token_ranges,
                                            rank_chunk_prefix)

        return EPChunkPlan(
            logical_token_ranges=logical_token_ranges,
            rank_chunk_prefix=rank_chunk_prefix,
            chunk_size=self.chunk_size,
            max_num_tokens=self.max_num_tokens,
            recv_capacity_tokens=recv_capacity_tokens,
            expert_alignment=expert_alignment,
            world_size=self.world_size,
            topk=topk,
            num_experts=self.num_experts,
        )

    def finalize(self) -> None:
        """Collectively release the planner workspace and its NCCL lease."""
        if self._finalized:
            return
        if self._workspace is not None:
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            self._workspace_tensor = None
            self._workspace.free(self.group)
            self._workspace = None
            self._workspace_win_handle = 0
        if self._has_nccl_gin_lease:
            release_nccl_gin(self.group)
            self._has_nccl_gin_lease = False
        self._finalized = True

    close = finalize

    def __enter__(self) -> "EPChunkPlanner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.finalize()


def build_ep_chunk_plan(planner: EPChunkPlanner, topk_indices: torch.Tensor, *, recv_capacity_tokens: int,
                        expert_alignment: int = 1, launch_num_sms: int | None = None) -> EPChunkPlan:
    """Functional spelling of :meth:`EPChunkPlanner.build`."""
    return planner.build(
        topk_indices,
        recv_capacity_tokens=recv_capacity_tokens,
        expert_alignment=expert_alignment,
        launch_num_sms=launch_num_sms,
    )
