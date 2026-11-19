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
"""Compiler-owned role pipeline templates."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from little_kernel.core.memory_effects import R_SMEM_GENERIC_SYNC, R_SMEM_TCGEN05_COMMIT, R_SMEM_TMA_BULK, R_TMEM_TCGEN05_COMMIT, R_TMEM_TCGEN05_WAIT_LD, W_SMEM_GENERIC_SYNC, W_SMEM_TMA_TX, W_TMEM_TCGEN05_COMMIT, W_TMEM_TCGEN05_WAIT_ST


class ArriveKind(StrEnum):
    """How a role arrives an mbarrier (named-barrier pipelines do not use this)."""

    MBAR_THREADS = "threads"  # mbarrier.arrive
    MBAR_TX = "tx"  # mbarrier.arrive.expect_tx
    MBAR_UMMA = "umma"  # tcgen05.commit
    STORE = "store"  # tma-store completion, then mbarrier.arrive


class CountBasis(StrEnum):
    THREADS = "threads"
    WARPS = "warps"
    COMPLETION_DOMAINS = "completion_domains"


@dataclass(frozen=True)
class BarrierSide:
    arrive: ArriveKind
    count: CountBasis
    cluster_scale: bool = False


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    has_barriers: bool = True
    producer: BarrierSide | None = None
    consumer: BarrierSide | None = None
    named_barrier: bool = False


#  Except named barrier, both sides arrive an mbarrier:
#    threads  mbarrier.arrive
#    tx       mbarrier.arrive.expect_tx
#    umma     tcgen05.commit
#    store    tma-store completion, then mbarrier.arrive
#
#  producer -> consumer   spec                       producer  consumer
#  ---------------------  -------------------------  --------  --------
#  TMA_LOAD -> UMMA       PipelineTmaUmma            tx        umma
#  TMA_LOAD -> GENERIC    PipelineTmaGeneric         tx        threads*
#  TMA_LOAD -> TMA_STORE  PipelineTmaTmaStore        tx        store
#  same-role TMA relay    PipelineTmaTmaStoreSameRole tx        wait_group.read
#  UMMA -> UMMA           PipelineUmmaUmma           umma      umma
#  UMMA -> TMEM load      PipelineUmmaTmemLoad       umma      threads*
#  TMEM store -> UMMA     PipelineTmemStoreUmma      threads   umma
#  TMEM store -> load     PipelineTmemStoreTmemLoad  threads   threads*
#  GENERIC -> UMMA        PipelineGenericUmma        threads   umma
#  GENERIC -> GENERIC     PipelineGenericGeneric     threads   threads
#  GENERIC -> TMA_STORE   PipelineTmaStoreCrossRole  threads   store
#  same-role TMA_STORE    PipelineTmaStoreSameRole   named barrier
#
#  * thread count * dest CTAs (cta_group on a pair, multicast component otherwise).
PIPELINE_TMA_UMMA = PipelineSpec(
    "PipelineTmaUmma",
    producer=BarrierSide(ArriveKind.MBAR_TX, CountBasis.WARPS),
    consumer=BarrierSide(ArriveKind.MBAR_UMMA, CountBasis.WARPS),
)
PIPELINE_TMA_GENERIC = PipelineSpec(
    "PipelineTmaGeneric",
    producer=BarrierSide(ArriveKind.MBAR_TX, CountBasis.WARPS),
    consumer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS, cluster_scale=True),
)
PIPELINE_TMA_TMA_STORE = PipelineSpec(
    "PipelineTmaTmaStore",
    producer=BarrierSide(ArriveKind.MBAR_TX, CountBasis.WARPS),
    consumer=BarrierSide(ArriveKind.STORE, CountBasis.COMPLETION_DOMAINS),
)
PIPELINE_TMA_TMA_STORE_SAME_ROLE = PipelineSpec(
    "PipelineTmaTmaStoreSameRole",
    producer=BarrierSide(ArriveKind.MBAR_TX, CountBasis.WARPS),
)
PIPELINE_UMMA_UMMA = PipelineSpec(
    "PipelineUmmaUmma",
    producer=BarrierSide(ArriveKind.MBAR_UMMA, CountBasis.WARPS),
    consumer=BarrierSide(ArriveKind.MBAR_UMMA, CountBasis.WARPS),
)
PIPELINE_UMMA_TMEM_LOAD = PipelineSpec(
    "PipelineUmmaTmemLoad",
    producer=BarrierSide(ArriveKind.MBAR_UMMA, CountBasis.WARPS),
    consumer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS, cluster_scale=True),
)
PIPELINE_TMEM_STORE_UMMA = PipelineSpec(
    "PipelineTmemStoreUmma",
    producer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS),
    consumer=BarrierSide(ArriveKind.MBAR_UMMA, CountBasis.WARPS),
)
PIPELINE_TMEM_STORE_TMEM_LOAD = PipelineSpec(
    "PipelineTmemStoreTmemLoad",
    producer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS),
    consumer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS, cluster_scale=True),
)
PIPELINE_GENERIC_UMMA = PipelineSpec(
    "PipelineGenericUmma",
    producer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS),
    consumer=BarrierSide(ArriveKind.MBAR_UMMA, CountBasis.WARPS),
)
PIPELINE_TMA_STORE_CROSS_ROLE = PipelineSpec(
    "PipelineTmaStoreCrossRole",
    producer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS),
    consumer=BarrierSide(ArriveKind.STORE, CountBasis.COMPLETION_DOMAINS),
)
PIPELINE_TMA_STORE_SAME_ROLE = PipelineSpec(
    "PipelineTmaStoreSameRole",
    has_barriers=False,
    named_barrier=True,
)
PIPELINE_GENERIC_GENERIC = PipelineSpec(
    "PipelineGenericGeneric",
    producer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS),
    consumer=BarrierSide(ArriveKind.MBAR_THREADS, CountBasis.THREADS),
)

# Same-role TMA stores are selected by topology after this semantic lookup.
PIPELINE_BY_EFFECT_KINDS = {
    (W_SMEM_GENERIC_SYNC, R_SMEM_GENERIC_SYNC): PIPELINE_GENERIC_GENERIC,
    (W_SMEM_GENERIC_SYNC, R_SMEM_TMA_BULK): PIPELINE_TMA_STORE_CROSS_ROLE,
    (W_SMEM_GENERIC_SYNC, R_SMEM_TCGEN05_COMMIT): PIPELINE_GENERIC_UMMA,
    (W_SMEM_TMA_TX, R_SMEM_GENERIC_SYNC): PIPELINE_TMA_GENERIC,
    (W_SMEM_TMA_TX, R_SMEM_TMA_BULK): PIPELINE_TMA_TMA_STORE,
    (W_SMEM_TMA_TX, R_SMEM_TCGEN05_COMMIT): PIPELINE_TMA_UMMA,
    (W_TMEM_TCGEN05_COMMIT, R_TMEM_TCGEN05_COMMIT): PIPELINE_UMMA_UMMA,
    (W_TMEM_TCGEN05_COMMIT, R_TMEM_TCGEN05_WAIT_LD): PIPELINE_UMMA_TMEM_LOAD,
    (W_TMEM_TCGEN05_WAIT_ST, R_TMEM_TCGEN05_COMMIT): PIPELINE_TMEM_STORE_UMMA,
    (W_TMEM_TCGEN05_WAIT_ST, R_TMEM_TCGEN05_WAIT_LD): PIPELINE_TMEM_STORE_TMEM_LOAD,
}

__all__ = [
    "ArriveKind",
    "BarrierSide",
    "CountBasis",
    "PIPELINE_BY_EFFECT_KINDS",
    "PIPELINE_GENERIC_GENERIC",
    "PIPELINE_TMA_STORE_SAME_ROLE",
    "PIPELINE_TMA_STORE_CROSS_ROLE",
    "PIPELINE_TMA_GENERIC",
    "PIPELINE_TMA_TMA_STORE",
    "PIPELINE_TMA_TMA_STORE_SAME_ROLE",
    "PIPELINE_TMA_UMMA",
    "PIPELINE_GENERIC_UMMA",
    "PIPELINE_TMEM_STORE_TMEM_LOAD",
    "PIPELINE_TMEM_STORE_UMMA",
    "PIPELINE_UMMA_TMEM_LOAD",
    "PIPELINE_UMMA_UMMA",
    "PipelineSpec",
]
