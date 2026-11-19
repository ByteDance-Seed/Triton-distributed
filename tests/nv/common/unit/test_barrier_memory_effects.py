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
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
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
"""Synchronization-boundary effects for the barrier/fence intrinsics.

This pins the "synchronization semantics" axis of the scan/indexing
coverage audit (semantic key coverage.scan_indexing).  Before these
declarations the barrier intrinsics carried NO memory effects, so an
effect-driven analysis walking a kernel via MEMORY_EFFECTS_ATTR could not
distinguish ``__syncthreads()`` from a side-effect-free op: the boundary
that protects the scan family's "no reorder across a warp sync" contract
was itself invisible.

Contract under test (all assertions are on function attributes only —
deterministic, no GPU):

- every listed barrier/fence builtin declares exactly one effect via
  MEMORY_EFFECTS_ATTR (see little_kernel/core/memory_effects.py);
- each effect is a synchronization-boundary effect produced by
  ``sync_effect``: the distinguished operand "" names NO parameter (a
  barrier reads/writes no argument; per-parameter read/write attribution
  must stay empty) and carries no region;
- the effect kind is the conservative boundary kind that matches the op's
  synchronized issuer set and ordered space:
    * __syncwarp          -> RW_RMEM_WARP_BARRIER (warp-issued, RMEM view);
    * __syncthreads, named_barrier_sync/arrive, cluster_arrive/wait/sync
                          -> RW_SMEM_CTA_BARRIER (thread-arrived CTA
                             boundary over SMEM; the named/cluster kinds
                             conservatively share it because the effect
                             model carries no barrier-id or cluster-issuer
                             vocabulary);
    * threadfence, threadfence_system
                          -> RW_GMEM_DEVICE_FENCE (thread-issued fence
                             over GMEM; system scope shares the kind — the
                             model has no fence-scope vocabulary);
- declared effects change no emission: the zero-parameter codegen return
  spellings (__syncwarp() / __syncthreads() / __threadfence()) and the
  named-barrier validity checks are byte-for-byte unchanged.
"""

import inspect
import unittest

from little_kernel.core.memory_effects import Access, Completion, Execution, IssuerSet, MemoryProxy, MemorySpace, RW_GMEM_DEVICE_FENCE, RW_RMEM_WARP_BARRIER, RW_SMEM_CTA_BARRIER
from little_kernel.language.builtin_base import CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, ORIGINAL_PY_FUNC_ATTR
import little_kernel.backends.nv.common.language.intrin.barrier as barrier
from little_kernel.core.type_system import void

# Every barrier/fence builtin that declares a synchronization-boundary
# effect, mapped to its required boundary kind.  Keeping the list explicit
# makes the coverage property testable: adding a new boundary op requires
# adding it HERE (with its declared kind), so a silently un-effected
# intrinsic fails this suite.
BOUNDARY_DECLARATIONS = (
    # func, kind
    ("__syncwarp", RW_RMEM_WARP_BARRIER),
    ("__syncthreads", RW_SMEM_CTA_BARRIER),
    ("named_barrier_sync", RW_SMEM_CTA_BARRIER),
    ("named_barrier_arrive", RW_SMEM_CTA_BARRIER),
    ("cluster_arrive", RW_SMEM_CTA_BARRIER),
    ("cluster_wait", RW_SMEM_CTA_BARRIER),
    ("cluster_sync", RW_SMEM_CTA_BARRIER),
    ("threadfence", RW_GMEM_DEVICE_FENCE),
    ("threadfence_system", RW_GMEM_DEVICE_FENCE),
)


class TestBarrierBoundaryEffects(unittest.TestCase):

    def test_every_boundary_declares_one_effect(self):
        for name, kind in BOUNDARY_DECLARATIONS:
            fn = getattr(barrier, name)
            with self.subTest(intrinsic=name):
                effects = getattr(fn, MEMORY_EFFECTS_ATTR, None)
                self.assertIsNotNone(effects, f"{name}: no MEMORY_EFFECTS_ATTR declared")
                self.assertEqual(len(effects), 1)
                self.assertIs(effects[0].kind, kind)

    def test_boundary_effect_operand_names_no_parameter(self):
        for name, _kind in BOUNDARY_DECLARATIONS:
            fn = getattr(barrier, name)
            cls = getattr(fn, ORIGINAL_PY_FUNC_ATTR, None)
            self.assertIsNotNone(cls)
            params = set(inspect.signature(cls.codegen).parameters)
            for eff in getattr(fn, MEMORY_EFFECTS_ATTR):
                with self.subTest(intrinsic=name, operand=eff.operand):
                    # The "" boundary operand is the ONLY legal operand of a
                    # boundary effect; it names no parameter and binds no
                    # region, so per-argument read/write attribution stays
                    # empty (a barrier reads/writes no argument).
                    self.assertEqual(eff.operand, "")
                    self.assertNotIn(eff.operand, params)
                    self.assertIsNone(eff.region)

    def test_boundary_kind_shape(self):
        # All boundary kinds are synchronous, generic-proxy, scoreboard-
        # completed, conservative RW accesses; they differ only in space
        # and issuer set.  This pins the shape so a future kind edit that
        # silently weakens the claim (R-only, async, ...) fails here.
        for name, kind in BOUNDARY_DECLARATIONS:
            with self.subTest(intrinsic=name):
                self.assertIs(kind.access, Access.RW)
                self.assertIs(kind.execution, Execution.SYNC)
                self.assertIs(kind.proxy, MemoryProxy.GENERIC)
                self.assertIs(kind.completion, Completion.SCOREBOARD)
                self.assertFalse(kind.atomic)

    def test_boundary_kind_space_and_issuer(self):
        expected = {
            "__syncwarp": (MemorySpace.RMEM, IssuerSet.WARP),
            "__syncthreads": (MemorySpace.SMEM, IssuerSet.THREADS),
            "named_barrier_sync": (MemorySpace.SMEM, IssuerSet.THREADS),
            "named_barrier_arrive": (MemorySpace.SMEM, IssuerSet.THREADS),
            "cluster_arrive": (MemorySpace.SMEM, IssuerSet.THREADS),
            "cluster_wait": (MemorySpace.SMEM, IssuerSet.THREADS),
            "cluster_sync": (MemorySpace.SMEM, IssuerSet.THREADS),
            "threadfence": (MemorySpace.GMEM, IssuerSet.THREADS),
            "threadfence_system": (MemorySpace.GMEM, IssuerSet.THREADS),
        }
        for name, (space, issuer) in expected.items():
            fn = getattr(barrier, name)
            (eff, ) = getattr(fn, MEMORY_EFFECTS_ATTR)
            with self.subTest(intrinsic=name):
                self.assertIs(eff.kind.space, space)
                self.assertIs(eff.kind.issuer, issuer)

    def test_return_type_and_codegen_unchanged(self):
        # Zero-parameter ops: declaring effects must not perturb the emitted
        # spelling or the return typing.
        spelling = {
            "__syncwarp": "__syncwarp()",
            "__syncthreads": "__syncthreads()",
            "threadfence": "__threadfence()",
            "threadfence_system": "__threadfence_system()",
        }
        for name, text in spelling.items():
            fn = getattr(barrier, name)
            with self.subTest(intrinsic=name):
                ret = getattr(fn, EVAL_RETURN_TYPE_ATTR)()
                self.assertIs(ret, void)
                out = getattr(fn, CODEGEN_FUNC_ATTR)()
                self.assertEqual(out.return_val, text)
                self.assertEqual(out.includes, [])
        # Parameterized named barriers keep their legality checks verbatim:
        # a bad count still raises, a good call still renders bar.sync.
        good = getattr(barrier.named_barrier_sync, CODEGEN_FUNC_ATTR)(1, 64)
        self.assertEqual(good.return_val, "named_barrier_sync_fn(1, 64)")
        with self.assertRaises(ValueError):
            getattr(barrier.named_barrier_sync, EVAL_RETURN_TYPE_ATTR)(1, 33)
        with self.assertRaises(ValueError):
            getattr(barrier.named_barrier_arrive, CODEGEN_FUNC_ATTR)(16, 32)

    def test_cluster_and_fence_effects_are_the_shared_kinds(self):
        # The shared-kind claims of the docstrings are load-bearing: a
        # cluster op must not silently drift onto a distinct kind, because
        # consumers reason "CTA-barrier kind => keep SMEM accesses on their
        # own side" for the whole family.
        for name in ("cluster_arrive", "cluster_wait", "cluster_sync", "named_barrier_sync", "named_barrier_arrive",
                     "__syncthreads"):
            (eff, ) = getattr(getattr(barrier, name), MEMORY_EFFECTS_ATTR)
            with self.subTest(intrinsic=name):
                self.assertIs(eff.kind, RW_SMEM_CTA_BARRIER)
        (fence, ) = getattr(barrier.threadfence, MEMORY_EFFECTS_ATTR)
        (sysfence, ) = getattr(barrier.threadfence_system, MEMORY_EFFECTS_ATTR)
        self.assertIs(fence.kind, sysfence.kind)
        self.assertIs(fence.kind, RW_GMEM_DEVICE_FENCE)


SCAN_BARRIER_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def scan_barrier(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    # Warp scan (cross-lane RMEM effect) on both sides of a CTA barrier
    # (SMEM boundary effect): the canonical composition the effect model
    # must keep ordered (accepted-prefix walk -> barrier -> repair write).
    a: ll.int32 = ll.warp_scan_inclusive_add(0xffffffff, x[i])
    ll.__syncthreads()
    b: ll.int32 = ll.warp_scan_exclusive_add(0xffffffff, x[i])
    out[i] = a + b
'''


class TestBarrierEffectsEndToEnd(unittest.TestCase):
    """Full lower() path: barrier + scan composition lowers with effects
    declared; emitted spelling is unchanged by the declarations."""

    def test_scan_barrier_composition_lowers(self):
        import json
        import tempfile
        from pathlib import Path
        from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            task = tmp / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = tmp / "program"
            program.mkdir()
            (program / "draft.py").write_text(SCAN_BARRIER_SOURCE)
            (program / "export.json").write_text(
                json.dumps({
                    "kernel": "scan_barrier",
                    "function": "scan_barrier",
                    "inputs": [{"name": "x", "dtype": "int32", "rank": 1}],
                    "output": {"name": "out", "dtype": "int32", "shape": ["x[0]"]},
                    "scalars": [],
                    "launch": {"grid": ["ceil_div(x[0], 32)", 1, 1], "block": [32, 1, 1]},
                }))
            output = tmp / "output"
            lower(task, program, output, 0)
            kernels = sorted((output / "kernels").glob("*.cu"))
            self.assertEqual(len(kernels), 1, kernels)
            cuda = kernels[0].read_text()
        # The barrier and both scan calls survived lowering.  The templated
        # helper DEFINITIONS hoist to the module top; within the kernel BODY
        # the issued order must be inclusive-scan call, then __syncthreads(),
        # then exclusive-scan call (declaring effects changed nothing about
        # emission, and nothing reordered the boundary relative to the
        # collectives it protects).
        body = cuda[cuda.index("__global__"):]
        self.assertIn("__syncthreads();", body)
        incl_call = body.index("= warp_scan_inclusive_add_fn(")
        barrier_pos = body.index("__syncthreads();")
        excl_call = body.index("= warp_scan_exclusive_add_fn(")
        self.assertLess(incl_call, barrier_pos)
        self.assertLess(barrier_pos, excl_call)


if __name__ == "__main__":
    unittest.main()
