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
"""Memory effects attached to LittleKernel builtins."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MemorySpace(Enum):
    RMEM = "rmem"
    TMEM = "tmem"
    SMEM = "smem"
    GMEM = "gmem"
    CMEM = "cmem"


class MemoryProxy(Enum):
    GENERIC = "generic"
    ASYNC = "async"
    TCGEN05 = "tcgen05"


class Access(Enum):
    R = "r"
    W = "w"
    RW = "rw"


class Execution(Enum):
    SYNC = "sync"
    ASYNC = "async"


class IssuerSet(Enum):
    THREADS = "threads"
    WARP = "warp"
    CTA_GROUP = "cta_group"
    ELECTED_ONE = "elected_one"
    ISSUER_DOMAIN = "issuer_domain"


class Completion(Enum):
    SCOREBOARD = "scoreboard"
    MBARRIER_TX = "mbarrier_tx"
    TMA_BULK = "tma_bulk"
    TCGEN05_COMMIT = "tcgen05_commit"
    TCGEN05_WAIT_ST = "tcgen05_wait_st"
    TCGEN05_WAIT_LD = "tcgen05_wait_ld"


@dataclass(frozen=True)
class MemoryEffectKind:
    access: Access
    space: MemorySpace
    execution: Execution
    proxy: MemoryProxy
    issuer: IssuerSet
    completion: Completion
    # ``atomic=True`` marks a hardware-atomic read-modify-write (or
    # fire-and-forget reduction): concurrent atomic accesses to the same
    # region are serialized by the memory system and do NOT form a data
    # race, unlike plain (non-atomic) accesses whose write value or read
    # result would otherwise be undefined under conflicting concurrent
    # access.  The flag carries no memory-ordering promise (relaxed,
    # acq/rel and system-scope variants all map to the same region-access
    # kind); it only classifies the access nature for race/conflict and
    # write-only-init analyses.  Non-atomic kinds keep ``atomic=False`` and
    # their ``.value`` rendering is unchanged.
    atomic: bool = False

    def for_access(self, access: Access) -> "MemoryEffectKind":
        if self.access != Access.RW and self.access != access:
            raise ValueError(f"cannot use {self.value} as {access.value}")
        return MemoryEffectKind(access, self.space, self.execution, self.proxy, self.issuer, self.completion,
                                self.atomic)

    @property
    def value(self) -> str:
        suffix = {
            Completion.SCOREBOARD: f"{self.proxy.value}.{self.execution.value}",
            Completion.MBARRIER_TX: "tma.tx",
            Completion.TMA_BULK: "tma.bulk",
            Completion.TCGEN05_COMMIT: "tcgen05.commit",
            Completion.TCGEN05_WAIT_ST: "tcgen05.wait_st",
            Completion.TCGEN05_WAIT_LD: "tcgen05.wait_ld",
        }[self.completion]
        return (f"{self.access.value}.{self.space.value}.{suffix}"
                f"{'.atomic' if self.atomic else ''}")


@dataclass(frozen=True)
class OperandRegion:
    """Logical region derived from intrinsic operands.

    ``offset_operand=None`` denotes a fixed default offset. ``extent`` is
    either a fixed element count or the name of an operand whose value/tuple
    length supplies the count. ``offset_default`` also supplies the fallback
    for optional offset operands such as TMA's ``smem_offset``.
    """

    offset_operand: str | None
    extent: int | str | None = None
    offset_default: int | None = None


@dataclass(frozen=True)
class Effect:
    kind: MemoryEffectKind
    operand: str
    region: OperandRegion | None = None

    @property
    def access(self) -> Access:
        return self.kind.access

    @property
    def space(self) -> MemorySpace:
        return self.kind.space

    @property
    def proxy(self) -> MemoryProxy:
        return self.kind.proxy


MS = MemorySpace
MP = MemoryProxy

W_SMEM_GENERIC_SYNC = MemoryEffectKind(Access.W, MS.SMEM, Execution.SYNC, MP.GENERIC, IssuerSet.THREADS,
                                       Completion.SCOREBOARD)
R_SMEM_GENERIC_SYNC = MemoryEffectKind(Access.R, MS.SMEM, Execution.SYNC, MP.GENERIC, IssuerSet.THREADS,
                                       Completion.SCOREBOARD)
W_SMEM_TMA_TX = MemoryEffectKind(Access.W, MS.SMEM, Execution.ASYNC, MP.ASYNC, IssuerSet.ELECTED_ONE,
                                 Completion.MBARRIER_TX)
R_SMEM_TMA_BULK = MemoryEffectKind(Access.R, MS.SMEM, Execution.ASYNC, MP.ASYNC, IssuerSet.ISSUER_DOMAIN,
                                   Completion.TMA_BULK)
R_SMEM_TCGEN05_COMMIT = MemoryEffectKind(Access.R, MS.SMEM, Execution.ASYNC, MP.ASYNC, IssuerSet.CTA_GROUP,
                                         Completion.TCGEN05_COMMIT)
W_TMEM_TCGEN05_COMMIT = MemoryEffectKind(Access.W, MS.TMEM, Execution.ASYNC, MP.TCGEN05, IssuerSet.CTA_GROUP,
                                         Completion.TCGEN05_COMMIT)
RW_TMEM_TCGEN05_COMMIT = MemoryEffectKind(Access.RW, MS.TMEM, Execution.ASYNC, MP.TCGEN05, IssuerSet.CTA_GROUP,
                                          Completion.TCGEN05_COMMIT)
R_TMEM_TCGEN05_COMMIT = MemoryEffectKind(Access.R, MS.TMEM, Execution.ASYNC, MP.TCGEN05, IssuerSet.CTA_GROUP,
                                         Completion.TCGEN05_COMMIT)
W_TMEM_TCGEN05_WAIT_ST = MemoryEffectKind(Access.W, MS.TMEM, Execution.ASYNC, MP.TCGEN05, IssuerSet.WARP,
                                          Completion.TCGEN05_WAIT_ST)
R_TMEM_TCGEN05_WAIT_LD = MemoryEffectKind(Access.R, MS.TMEM, Execution.ASYNC, MP.TCGEN05, IssuerSet.WARP,
                                          Completion.TCGEN05_WAIT_LD)

# TMEM ordering boundary for ``tcgen05.fence::before/after_thread_sync``.
# These fences touch no TMEM data themselves; their force is ordering
# between asynchronous tcgen05 accesses (UMMA commits, tcgen05.ld/st) and
# surrounding thread synchronization.  Like RW_SMEM_CTA_BARRIER this is a
# conservative RW boundary so effect-driven motion/merge analyses keep
# TMEM accesses on their own side of the fence; recorded via sync_effect.
RW_TMEM_TCGEN05_FENCE = MemoryEffectKind(Access.RW, MS.TMEM, Execution.SYNC, MP.TCGEN05, IssuerSet.THREADS,
                                         Completion.SCOREBOARD)

# Cross-lane register read issued collectively by a warp (warp shuffles,
# vote/match collectives).  The read operand is a per-lane register whose
# returned value may come from a DIFFERENT lane, so the op is a convergent
# warp collective: it must be executed by every lane named in its mask under
# uniform control flow, and it must not be reordered across a warp sync or a
# divergence boundary.  The effect captures that the "region read" is the
# warp-wide register file, not the issuing thread's SSA value alone.
R_RMEM_WARP_SYNC = MemoryEffectKind(Access.R, MS.RMEM, Execution.SYNC, MP.GENERIC, IssuerSet.WARP,
                                    Completion.SCOREBOARD)

# Synchronization-boundary effects (barriers and fences).
#
# A barrier itself touches no data: its observable force is ORDERING — no
# access issued before the barrier may be observed after accesses issued
# after it by the synchronized issuer set.  The conservative RW access on the
# ordered space encodes exactly that: effect-driven motion/merge analyses
# that treat any R/W/RW effect of a space as a use/definition barrier will
# keep accesses on their own side of the op.  No operand names a region;
# the effect records the boundary EXISTENCE via the distinguished empty
# operand "" (see sync_effect).  Issuer sets select who syncs: WARP for
# warp-level barriers, THREADS for CTA-partial/named barriers arrived at by
# individual threads (the union of arriving threads is the synchronized
# set), and THREADS with GMEM scope for device/system fences which order
# global memory visibility of the issuing thread against observers at the
# fence's scope.
RW_RMEM_WARP_BARRIER = MemoryEffectKind(Access.RW, MS.RMEM, Execution.SYNC, MP.GENERIC, IssuerSet.WARP,
                                        Completion.SCOREBOARD)
RW_SMEM_CTA_BARRIER = MemoryEffectKind(Access.RW, MS.SMEM, Execution.SYNC, MP.GENERIC, IssuerSet.THREADS,
                                       Completion.SCOREBOARD)
RW_GMEM_DEVICE_FENCE = MemoryEffectKind(Access.RW, MS.GMEM, Execution.SYNC, MP.GENERIC, IssuerSet.THREADS,
                                        Completion.SCOREBOARD)

# Generic-address hardware-atomic read-modify-write on global memory
# (atomicAdd/CAS/Min/Max/Or/And/Exch/Inc, red.global vector reductions).
# Access is RW even for fire-and-forget reductions that return no value:
# the stored value depends on the prior memory contents, so such an op can
# never serve as a write-only init origin for a buffer state cycle.  The
# atomic flag marks hardware serialization (concurrent atomics to the same
# region are not a data race); device-scope, acquire-release and system
# scope variants share this one conservative kind because the model carries
# no ordering vocabulary.  The generic address space is modeled as GMEM (the
# dominant use); shared-memory-only specializations SHOULD declare a
# narrower smem kind instead of reusing this one.
RW_GMEM_ATOMIC_SYNC = MemoryEffectKind(Access.RW, MS.GMEM, Execution.SYNC, MP.GENERIC, IssuerSet.THREADS,
                                       Completion.SCOREBOARD, atomic=True)


def operand_region(
    offset_operand: str | None = None,
    extent: int | str | None = None,
    *,
    offset_default: int | None = None,
) -> OperandRegion:
    return OperandRegion(offset_operand, extent, offset_default)


def effect(
    kind: MemoryEffectKind,
    operand: str,
    *,
    region: OperandRegion | None = None,
) -> Effect:
    return Effect(kind, operand, region)


def sync_effect(kind: MemoryEffectKind) -> Effect:
    """Boundary effect for a synchronization op (barrier/fence).

    Returns an Effect with the distinguished empty operand ``""``: the op
    has no data operand naming a region — the effect records the ORDERING
    boundary itself, not a memory footprint.  Consumers that map effect
    operands to call parameters must treat ``""`` as "no operand" (it never
    names a parameter, so per-parameter read/write attribution stays empty,
    which is correct: a barrier neither reads nor writes any argument).
    """
    return Effect(kind, "", None)


def r(
    space: MemorySpace,
    operand: str,
    *,
    proxy: MemoryProxy = MP.GENERIC,
    region: OperandRegion | None = None,
) -> Effect:
    return Effect(
        MemoryEffectKind(Access.R, space, Execution.SYNC, proxy, IssuerSet.THREADS, Completion.SCOREBOARD),
        operand,
        region,
    )


def w(
    space: MemorySpace,
    operand: str,
    *,
    proxy: MemoryProxy = MP.GENERIC,
    region: OperandRegion | None = None,
) -> Effect:
    return Effect(
        MemoryEffectKind(Access.W, space, Execution.SYNC, proxy, IssuerSet.THREADS, Completion.SCOREBOARD),
        operand,
        region,
    )


def rw(
    space: MemorySpace,
    operand: str,
    *,
    proxy: MemoryProxy = MP.GENERIC,
    region: OperandRegion | None = None,
) -> Effect:
    return Effect(
        MemoryEffectKind(Access.RW, space, Execution.SYNC, proxy, IssuerSet.THREADS, Completion.SCOREBOARD),
        operand,
        region,
    )
