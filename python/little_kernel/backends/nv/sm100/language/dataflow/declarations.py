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
"""Build a kernel's role/memory graph inside ``with dataflow() as g:``.

* ``g.role("tma", name="tma_loader", ...)`` — create a role; ``"tma"`` is the graph key
* ``g.smem("A_smem", shape, dtype=...)`` — create SMEM (registered by buffer name)
* ``stage = g.depends(src=tma, dst=[mma0, mma1], via=[A_smem])`` — create a channel key
* ``stage.bind(k)`` — mark one logical stage transaction in a role body
"""

from __future__ import annotations

import logging
import pprint
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass
from enum import StrEnum
from typing import Any, Iterator


class HW(StrEnum):
    TMA = "TMA"
    TensorCore = "TensorCore"
    CUDACore = "CUDACore"
    SFU = "SFU"


@dataclass
class Role:
    var: str
    name: str
    hw: HW
    _: KW_ONLY
    warps: list[int]


@dataclass
class SMem:
    name: str
    shape: Any
    _: KW_ONLY
    dtype: Any
    stages: int = 1
    copies: int | None = None
    major: str | None = None
    tma_transaction_bits: int | None = None

    def __post_init__(self) -> None:
        if self.major not in {None, "K", "MN"}:
            raise ValueError("SMEM major must be 'K' or 'MN'")
        if self.tma_transaction_bits is not None and self.tma_transaction_bits <= 0:
            raise ValueError("TMA transaction bits must be positive")


@dataclass(frozen=True)
class SMemLayoutPlan:
    logical_shape: tuple[int, ...]
    dtype: Any
    major: str
    swizzle_bytes: int


class TMem(SMem):
    pass


@dataclass(frozen=True)
class TMATransferPlan:
    """Compile-time coordinate sharding and group-grid multicast contract.

    ``cluster_shape`` is divided by ``cta_group_shape`` to form the logical
    group grid.  The fields below deliberately distinguish axes in that group
    grid from indices in the TMA tensor-coordinate tuple.
    """

    # Group-grid axis over which each CTA participant's local SMEM shard is
    # replicated.  Keep the other group coordinates and rank-in-group fixed.
    # None means that every CTA group loads its own shard without multicast.
    mcast_group_dim: str | None = None

    # Zero-based index in the 2D/3D TMA coordinate tuple that is partitioned
    # between participants of one CTA group.  None means no participant offset.
    inter_cta_coord_dim: int | None = None

    # Coordinate increment per rank-in-group:
    # coords[inter_cta_coord_dim] += rank_in_group * inter_cta_stride.
    # It must be zero when inter_cta_coord_dim is None and positive otherwise.
    inter_cta_stride: int = 0

    def __post_init__(self) -> None:
        if self.mcast_group_dim not in {None, "x", "y", "z"}:
            raise ValueError("mcast_group_dim must be one of None, 'x', 'y', or 'z'")
        if self.inter_cta_coord_dim is None:
            if self.inter_cta_stride != 0:
                raise ValueError("inter_cta_stride must be zero when inter_cta_coord_dim is None")
        elif self.inter_cta_coord_dim < 0 or self.inter_cta_stride <= 0:
            raise ValueError("inter_cta_coord_dim must be non-negative and inter_cta_stride positive")


@dataclass(frozen=True)
class StageKey:
    id: int

    def bind(self, value: Any) -> None:
        del value


class Flow:

    def __init__(self) -> None:
        self.roles: dict[str, Role] = {}
        self.memories: dict[str, SMem | TMem] = {}
        self.channels: list[dict[str, Any]] = []
        self.stage_key_vars: dict[str, int] = {}

    def _summary(self) -> dict[str, Any]:
        return {
            "roles": self.roles,
            "memories": self.memories,
            "channels": self.channels,
            "stage_key_vars": self.stage_key_vars,
        }

    def print(self, logger: logging.Logger | None = None, *, level: int = logging.INFO) -> str:
        text = pprint.pformat(self._summary(), sort_dicts=False, width=120)
        if logger is None:
            print(text)
        else:
            logger.log(level, "%s", text)
        return text

    def role(self, var: str, *, name: str | None = None, hw: HW, warps: list[int]) -> Role:
        role = Role(var=var, name=name or var, hw=hw, warps=warps)
        self.roles[var] = role
        return role

    def smem(self, name: str, shape: Any, *, dtype: Any, stages: int = 1, copies: int | None = None,
             major: str | None = None, tma_transaction_bits: int | None = None) -> SMem:
        mem = SMem(name, shape, dtype=dtype, stages=stages, copies=copies, major=major,
                   tma_transaction_bits=tma_transaction_bits)
        self.memories[name] = mem
        return mem

    def tmem(self, name: str, shape: Any, *, dtype: Any, stages: int = 1, copies: int | None = None) -> TMem:
        mem = TMem(name, shape, dtype=dtype, stages=stages, copies=copies)
        self.memories[name] = mem
        return mem

    def tma_transfer(
        self,
        *,
        mcast_group_dim: str | None = None,
        inter_cta_coord_dim: int | None = None,
        inter_cta_stride: int = 0,
    ) -> TMATransferPlan:
        """Declare compiler-owned TMA sharding and multicast.

        Args:
            mcast_group_dim: ``"x"``, ``"y"``, or ``"z"`` in group-grid
                coordinates; the selected axis is reused by multicast.  Use
                ``None`` when each CTA group loads independently.
            inter_cta_coord_dim: Index in the TMA tensor-coordinate tuple to
                shard between CTA-group participants, or ``None`` when all
                participants use the same coordinate.
            inter_cta_stride: Per-participant increment on that tensor
                coordinate.  It is zero with no sharded coordinate and
                positive otherwise.
        """
        return TMATransferPlan(
            mcast_group_dim,
            inter_cta_coord_dim,
            inter_cta_stride,
        )

    def depends(
        self,
        *,
        src: Role | list[Role] | tuple[Role, ...],
        dst: Role | list[Role] | tuple[Role, ...],
        via: list[SMem | TMem],
    ) -> StageKey:
        return self._channel(src=src, dst=dst, via=via)

    def _channel(
        self,
        *,
        src: Role | list[Role] | tuple[Role, ...],
        dst: Role | list[Role] | tuple[Role, ...],
        via: list[SMem | TMem],
    ) -> StageKey:
        producers = (src, ) if isinstance(src, Role) else tuple(src)
        consumers = (dst, ) if isinstance(dst, Role) else tuple(dst)
        if not producers or not consumers:
            raise ValueError("depends requires at least one producer and one consumer")
        if len(set(role.var for role in producers)) != len(producers):
            raise ValueError("depends producer roles must be unique")
        if len(set(role.var for role in consumers)) != len(consumers):
            raise ValueError("depends consumer roles must be unique")
        if not via:
            raise ValueError("depends requires at least one logical memory")

        key = StageKey(len(self.channels))
        self.channels.append({
            "key": key.id,
            "producers": tuple(role.var for role in producers),
            "consumers": tuple(role.var for role in consumers),
            "via": [m.name for m in via],
        })
        return key


@contextmanager
def dataflow() -> Iterator[Flow]:
    yield Flow()
