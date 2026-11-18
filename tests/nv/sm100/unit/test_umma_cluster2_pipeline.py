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
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""CPU regression tests for the sm_100a 2-CTA-cluster UMMA pipeline composition.

Capability: ir.cuda.tcgen05_umma_cluster2_pipeline

Covers, without GPU access:

1. ``alloc_dynamic_shared_memory`` is a pass-inserted internal builtin with
   signature ``(size_bytes, align_bytes, name, dyn_exprs)``. Calling it
   directly from a kernel (dtype-first, as an unwitting frontend would to
   allocate a typed shared buffer) must raise a *typed* TypeInferenceError
   that names the public dynamic-shared idiom, not an opaque Python
   TypeError wrapped in "Failed to compute return type".

2. The canonical typed single-kernel composition for the 2-CTA-cluster
   GEMM barrier schedule lowers to CUDA source for nv/sm100 on CPU:
   per-stage full barrier with arrival count 2 where the leader arrives
   with expect_tx and the peer arrives on the LEADER's barrier through the
   cluster-scoped arrive (mapa.shared::cluster), with mbarrier phase state
   carried across a persistent output-tile loop (no re-init per tile).
"""

from __future__ import annotations

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.core.passes.utils.type_inference.type_inference_core import (
    TypeInferenceError, )


def _compile_to_source(kernel, num_threads=128) -> str:
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=num_threads)


class TestAllocDynamicSharedMemoryDiagnostic(unittest.TestCase):

    def test_direct_call_dtype_first_raises_typed_diagnostic(self):
        with self.assertRaises(TypeInferenceError) as ctx:

            @lk.ll_kernel(backend="nv/sm100", is_entry=True)
            def bad(a: ll.Tensor[ll.bfloat16]) -> ll.void:
                # The dtype-first user-facing form is not a valid call: this
                # builtin is inserted by insert_mem_alloc, not authored.
                smem: ll.Tensor[ll.uint8] = ll.alloc_dynamic_shared_memory(ll.uint8, 4096)  # noqa: F841

            _compile_to_source(bad)

        msg = str(ctx.exception)
        self.assertIn("internal builtin", msg)
        self.assertIn("insert_mem_alloc", msg)
        # The diagnostic must name the public idiom so frontends can fix the
        # call site without reading the pass source.
        self.assertIn("dynamic_shared", msg)

    def test_missing_name_raises_typed_diagnostic(self):
        # Positional (size, align) without name hits the same misuse check:
        # a bare two-argument call cannot be a pass-inserted allocation.
        with self.assertRaises(TypeInferenceError) as ctx:

            @lk.ll_kernel(backend="nv/sm100", is_entry=True)
            def bad(a: ll.Tensor[ll.bfloat16]) -> ll.void:
                smem: ll.Tensor[ll.uint8] = ll.alloc_dynamic_shared_memory(4096, 16)  # noqa: F841

            _compile_to_source(bad)

        self.assertIn("internal builtin", str(ctx.exception))

    def test_pass_inserted_shape_type_checks(self):
        # The internal call shape (size, align, name, dyn_exprs) must remain
        # accepted by type inference: insert_mem_alloc generates exactly this.
        from little_kernel.backends.common.gpu.language.intrin.allocation import (
            alloc_dynamic_shared_memory_eval_arg_type, )
        type_ctx = {}
        alloc_dynamic_shared_memory_eval_arg_type(type_ctx, 4096, 16, "_dyn_smem", None)
        from little_kernel.core.type_system import uint8
        self.assertIs(type_ctx["_dyn_smem"], uint8)


def _cluster2_barrier_pipeline():

    @lk.ll_kernel(backend="nv/sm100", is_entry=True, cluster_shape=(2, 1, 1))
    def pipe(a: ll.Tensor[ll.bfloat16], out: ll.Tensor[ll.float32]) -> ll.void:
        # One full-barrier slot in static shared memory.
        s_bar_full = ll.alloc_shared_memory("s_bar_full", ll.uint64, 1, 8)
        bar_full: ll.uint32 = ll.cvta_generic_to_shared(s_bar_full)
        rank: ll.uint32 = ll.cluster_rank()
        tid: ll.uint32 = ll.threadIdx_x()

        # Leader elected thread initializes the full barrier with arrival
        # count 2: leader expect_tx arrival + peer cluster-scoped arrival.
        if tid == 0 and rank == 0:
            ll.init_smem_barrier(bar_full, 2)
        ll.fence_smem_barrier_init()
        ll.cluster_sync()

        # Persistent output-tile loop: phase carries across tiles.
        phase: ll.uint32 = ll.uint32(0)
        for tile in range(4):
            if tid == 0:
                # Peer CTA arrives on the LEADER's full barrier:
                # mbarrier_arrive_cluster applies mapa.shared::cluster to
                # retarget the slot to CTA `target_cta` (rank 0).
                ll.mbarrier_arrive_cluster(bar_full, 0)
                if rank == 0:
                    # Leader arrives with expect_tx for cta_group::2 TMA.
                    ll.mbarrier_arrive_and_expect_tx(bar_full, 256)
            ll.mbarrier_wait(bar_full, phase)
            phase = phase ^ 1

        if tid == 0:
            ll.mbarrier_arrive(bar_full)
        ll.cluster_sync()

    return pipe


class TestCluster2BarrierPipeline(unittest.TestCase):

    def test_lowers_to_cuda_source(self):
        src = _compile_to_source(_cluster2_barrier_pipeline())

        # Barrier init with arrival count 2 and the cluster-scope init fence.
        self.assertIn("mbarrier.init.shared.b64", src)
        self.assertIn("fence.mbarrier_init.release.cluster", src)
        # Peer arrive retargeted to the leader through the cluster address map.
        self.assertIn("mapa.shared::cluster.u32", src)
        self.assertIn("mbarrier.arrive.shared::cluster.b64", src)
        # Leader-side arrival carrying the TMA transaction byte count.
        self.assertIn("mbarrier.arrive.expect_tx.shared::cta.b64", src)
        # Phase-parity wait (persistent ring; phase toggled by the loop).
        self.assertIn("mbarrier.try_wait.parity.shared::cta.b64", src)
        # Cluster-wide sync bracketing init and drain.
        self.assertIn("barrier.cluster.arrive", src)
        self.assertIn("barrier.cluster.wait", src)


if __name__ == "__main__":
    unittest.main()
