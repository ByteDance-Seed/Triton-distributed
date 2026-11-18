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
"""TMEM epilogue ordering validation for role/dataflow kernels (RF05 follow-on).

A 2-CTA (or 1-CTA) UMMA mainloop produces an FP32 TMEM accumulator through
asynchronous ``tcgen05.commit`` completion.  An epilogue that drains that
accumulator with ``tcgen05.ld`` (modeled as R_TMEM_TCGEN05_WAIT_LD reads)
outside the MMA role is only correct under a precise ordering contract:

* ``tcgen05.fence::after_thread_sync`` (``ll.tcgen05_fence_after``) must
  separate the producer commit from the first drain read in program order;
* once the drain is complete, ``tcgen05.fence::before_thread_sync``
  (``ll.tcgen05_fence_before``), a CTA barrier (``__syncthreads``) and — for
  multi-CTA clusters — a ``cluster_sync`` must separate the last drain read
  from the next asynchronous write of the same accumulator (tile reuse) or
  from kernel exit.

Until now nothing validated that contract: the tcgen05 fences carried no
typed boundary effect and the only failure surface was a generic pre-codegen
refusal or a silent miscompile.  This pass checks the contract on the typed
role/dataflow graph and reports the first violated marker by name.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

from little_kernel.backends.nv.sm100.passes.dataflow.common import call_name as _call_name
from little_kernel.backends.nv.sm100.passes.dataflow.constants import DATAFLOW_CTX_KEY
from little_kernel.backends.nv.sm100.passes.dataflow.intrinsics_registry import get_intrin_mem_effects
from little_kernel.core.memory_effects import Access, MemorySpace

#: Intrinsic names that complete a prior ``tcgen05.ld`` register drain.
_TMEM_LD_FENCE = "tmem_ld_fence"
_FENCE_AFTER = "tcgen05_fence_after"
_FENCE_BEFORE = "tcgen05_fence_before"
_CTA_BARRIER = "__syncthreads"
_CLUSTER_BARRIER = "cluster_sync"


@dataclass
class _Event:
    """One program-ordered TMEM-relevant marker in the epilogue region."""
    order: int
    kind: str  # "write" | "read" | "fence_after" | "fence_before" | "cta_barrier" | "cluster_barrier"
    mems: frozenset[str] = frozenset()
    context: str = ""  # role name for in-role events, else "epilogue"


@dataclass
class _Scan:
    events: list[_Event] = field(default_factory=list)
    roles: frozenset[str] = frozenset()
    counter: int = 0

    def emit(self, kind: str, mems: frozenset[str], context: str) -> None:
        self.events.append(_Event(self.counter, kind, mems, context))
        self.counter += 1


def _classify_call(node: ast.Call) -> tuple[str | None, frozenset[str]]:
    """Classify one call by its typed memory effects; return (kind, mems)."""
    name = _call_name(node)
    if name is None:
        return None, frozenset()
    if name == _FENCE_AFTER:
        return "fence_after", frozenset()
    if name == _FENCE_BEFORE:
        return "fence_before", frozenset()
    if name == _CTA_BARRIER:
        return "cta_barrier", frozenset()
    if name == _CLUSTER_BARRIER:
        return "cluster_barrier", frozenset()
    mems: set[str] = set()
    kind: str | None = None
    for effect in get_intrin_mem_effects(name):
        if effect.space != MemorySpace.TMEM:
            continue
        # Every TMEM intrinsic takes its logical TMEM memory as the first
        # positional argument, so resolve the effect operand to that name.
        operand = node.args[0].id if node.args and isinstance(node.args[0], ast.Name) \
            else effect.operand.split(".", 1)[0]
        if effect.access in {Access.W, Access.RW} and name.startswith(("umma_", "tmem_st_", "tcgen05_cp_")):
            kind = "write" if kind is None else kind
            mems.add(operand)
        elif effect.access == Access.R and name.startswith("tmem_ld_") \
                and name != _TMEM_LD_FENCE:
            kind = "read" if kind is None else kind
            mems.add(operand)
    return kind, frozenset(mems)


def _scan_block(scan: _Scan, body: list[ast.stmt], context: str) -> None:
    for stmt in body:
        if isinstance(stmt, ast.With) and len(stmt.items) == 1 and \
                isinstance(stmt.items[0].context_expr, ast.Name) and \
                stmt.items[0].context_expr.id in scan.roles:
            role = stmt.items[0].context_expr.id
            _scan_block(scan, stmt.body, role)
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            kind, mems = _classify_call(stmt.value)
            if kind is not None:
                scan.emit(kind, mems, context)
        # Recurse so events inside loops/conditionals keep source order; this
        # validates one canonical iteration of tile loops.
        for field_name in ("body", "orelse", "finalbody"):
            child = getattr(stmt, field_name, None)
            if isinstance(child, list) and child and isinstance(child[0], ast.stmt):
                _scan_block(scan, child, context)


def _violation(graph_name: str, message: str) -> ValueError:
    return ValueError(f"{graph_name}: TMEM epilogue ordering violation: {message}")


def _validate_epilogue_ordering(graph_name: str, graph: dict, scan: _Scan) -> None:
    cluster_ctas = math.prod(graph.get("cluster_shape") or (1, 1, 1))
    events = scan.events
    umma_written = frozenset(mem for event in events if event.kind == "write" for mem in event.mems)
    drained = frozenset(mem for event in events if event.kind == "read" and event.context == "epilogue"
                        for mem in event.mems)
    for mem in sorted(umma_written & drained):
        reads = [
            event for event in events if event.kind == "read" and mem in event.mems and event.context == "epilogue"
        ]
        first_read, last_read = reads[0], reads[-1]
        prior = [event for event in events if event.order < first_read.order and mem in event.mems]
        prior_writes = [event for event in prior if event.kind == "write"]
        if prior_writes:
            fences = [
                event for event in events
                if event.kind == "fence_after" and prior_writes[-1].order < event.order < first_read.order
            ]
            if not fences:
                raise _violation(
                    graph_name, f"TMEM accumulator {mem!r}: drain read must be preceded by "
                    "ll.tcgen05_fence_after() after the producing UMMA")
        following = [event for event in events if event.order > last_read.order]
        next_write = next((event for event in following if event.kind == "write" and mem in event.mems), None)
        boundary_end = next_write.order if next_write is not None else math.inf
        tail = [event for event in following if event.order < boundary_end]
        fence_before = next((event for event in tail if event.kind == "fence_before"), None)
        if fence_before is None:
            raise _violation(
                graph_name, f"TMEM accumulator {mem!r}: drain must be closed by "
                "ll.tcgen05_fence_before() before accumulator reuse or kernel exit")
        cta_barrier = next(
            (event for event in tail if event.kind == "cta_barrier" and event.order > fence_before.order), None)
        if cta_barrier is None:
            raise _violation(
                graph_name, f"TMEM accumulator {mem!r}: ll.tcgen05_fence_before() must be "
                "followed by ll.__syncthreads() before accumulator reuse or kernel exit")
        if cluster_ctas > 1:
            cluster_barrier = next(
                (event for event in tail if event.kind == "cluster_barrier" and event.order > cta_barrier.order), None)
            if cluster_barrier is None:
                raise _violation(
                    graph_name, f"TMEM accumulator {mem!r}: both CTAs of the pair drain the "
                    "shared accumulator, so ll.__syncthreads() must be followed by "
                    "ll.cluster_sync() before accumulator reuse or kernel exit")


def tmem_epilogue_ordering_pass(tree: ast.Module, ctx: dict | None = None) -> ast.Module:
    """Validate fence/barrier ordering of TMEM accumulator drain epilogues."""
    ctx = ctx if ctx is not None else {}
    graphs = ctx.get(DATAFLOW_CTX_KEY) or {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in graphs:
            continue
        graph = graphs[node.name]
        scan = _Scan(roles=frozenset(graph.get("roles", {})))
        _scan_block(scan, node.body, "epilogue")
        _validate_epilogue_ordering(node.name, graph, scan)
    return tree
