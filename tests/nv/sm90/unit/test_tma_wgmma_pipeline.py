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
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
################################################################################
"""CPU lowering regression for the reusable sm90a bf16 GEMM pipeline
(ir.pipeline.tma_wgmma_gemm).

A single typed-IR kernel exercises the full producer/consumer chain the
frontend teacher decomposition needs:

  * TMA descriptor kernel parameters (``ll.TmaDescriptor``) consumed by
    ``tma_load_2d_1sm`` and ``tma_load_2d_1sm_mcast`` (cluster multicast);
  * the mbarrier lifecycle -- init / fence-init / arrive_and_expect_tx /
    wait / arrive_remote -- across a 2-CTA cluster bracketed by
    ``cluster_sync`` (barrier.cluster.arrive/.wait);
  * ``wgmma_m64n256k16`` f32.bf16.bf16 accumulation over shared-memory
    descriptors assembled with ``make_desc`` from cvta addresses;
  * the ``stmatrix.x2`` epilogue plus a named barrier.

CPU codegen only: we assert the emitted CUDA carries every required PTX
construct. Paired GPU correctness is left to the controller queue.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.memory as memory_intrin

BM, BK, BN = 64, 16, 256
NUM_STAGES = 2
CLUSTER_SIZE = 2
TX_BYTES = BM * BK * 2 + BN * BK * 2


class TestStmatrixX2(unittest.TestCase):
    """stmatrix.sync.aligned.x2.m8n8 is the two-register member of the
    stmatrix family (8x16 b16 tile); it was missing while x1/x4 existed,
    which blocked the draft GEMM epilogue at type inference."""

    def test_is_builtin_and_exported(self):
        self.assertTrue(getattr(memory_intrin.stmatrix_x2, BUILTIN_ATTR, False))
        self.assertIs(ll.stmatrix_x2, memory_intrin.stmatrix_x2)

    def test_codegen_emits_x2_ptx(self):
        codegen_func = getattr(memory_intrin.stmatrix_x2, CODEGEN_FUNC_ATTR)
        b = codegen_func("smem", "r0", "r1")
        self.assertIsInstance(b, Builtin)
        self.assertIn("stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};", b.body)
        # 32-bit shared address + exactly two source registers.
        self.assertIn('"r"(addr)', b.body)
        self.assertIn('"r"(r0)', b.body)
        self.assertIn('"r"(r1)', b.body)
        self.assertNotIn('"r"(r2)', b.body)
        # smem address goes through cvta like the x4 variant.
        self.assertIn("__cvta_generic_to_shared(smem_ptr)", b.body)
        self.assertEqual(b.return_val, "stmatrix_x2_fn(smem, r0, r1)")

    def test_distinct_from_x4_helper(self):
        x2 = getattr(memory_intrin.stmatrix_x2, CODEGEN_FUNC_ATTR)("p", "a", "b")
        x4 = memory_intrin.codegen_stmatrix_x4("p", "a", "b", "c", "d")
        self.assertIn("stmatrix_x2_fn", x2.return_val)
        self.assertIn("stmatrix_x4_fn", x4.return_val)
        self.assertIn(".x2.", x2.body)
        self.assertIn(".x4.", x4.body)


def _define_pipeline_kernel():

    @lk.ll_kernel(backend="cuda", is_entry=True)
    def tma_wgmma_pipe_kernel(
        dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
        dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
        C: ll.Tensor[ll.bfloat16],
    ) -> ll.void:
        ll.align_memory(1024, scope="dynamic_shared")
        A_smem = [ll.empty([BM, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
        B_smem = [ll.empty([BN, BK], dtype=ll.bfloat16, scope="dynamic_shared") for _ in range(NUM_STAGES)]
        C_smem = ll.empty([BM, BN], dtype=ll.bfloat16, scope="dynamic_shared")
        full = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]
        empty = [ll.empty([1], dtype=ll.uint64, scope="dynamic_shared") for _ in range(NUM_STAGES)]

        tid: ll.int32 = ll.threadIdx_x()
        lane: ll.int32 = tid % 32
        ll.cluster_rank()

        # ---- barrier init across the 2-CTA cluster ----
        if tid == 0:
            for s in ll.unroll(range(NUM_STAGES)):
                ll.init_smem_barrier(full[s], 1)
                ll.init_smem_barrier(empty[s], CLUSTER_SIZE * 8)
            ll.fence_smem_barrier_init()
        ll.cluster_sync()

        # ---- 2-stage TMA producer (B is cluster-multicast) ----
        if tid == 0:
            for s in ll.unroll(range(NUM_STAGES)):
                ll.tma_load_2d_1sm(dA, full[s], A_smem[s], 0, 0)
                ll.tma_load_2d_1sm_mcast(dB, full[s], B_smem[s], 0, 0, 3)
                ll.mbarrier_arrive_and_expect_tx(full[s], TX_BYTES)

        # ---- WGMMA m64n256k16 consumer ----
        base_a: ll.uint32 = ll.cvta_generic_to_shared(A_smem[0]) >> 4
        base_b: ll.uint32 = ll.cvta_generic_to_shared(B_smem[0]) >> 4
        acc = ll.zeros([128], dtype=ll.float32, scope="local")
        for s in ll.unroll(range(NUM_STAGES)):
            ll.mbarrier_wait(full[s], 0)
            ll.__syncwarp()
            ll.wgmma_fence()
            da: ll.uint64 = ll.make_desc(base_a + s * 64, 1 << 62)
            db: ll.uint64 = ll.make_desc(base_b + s * 256, 1 << 62)
            ll.wgmma_m64n256k16(acc, da, db)
            ll.wgmma_commit()
            ll.wgmma_wait()
            if lane < CLUSTER_SIZE:
                ll.mbarrier_arrive_remote(empty[s], lane)
            ll.__syncwarp()

        # ---- stmatrix.x2 epilogue + named barrier ----
        ll.named_barrier_sync(0, 256)
        r0: ll.uint32 = ll.uint32(acc[0])
        r1: ll.uint32 = ll.uint32(acc[1])
        ll.stmatrix_x2(C_smem, r0, r1)
        ll.named_barrier_sync(0, 256)
        ll.cluster_sync()

    return tma_wgmma_pipe_kernel


class TestTmaWgmmaGemmLowering(unittest.TestCase):
    """The whole sm90a pipeline must lower through type inference and the
    CUDA passes to a compilable artifact on CPU."""

    @classmethod
    def setUpClass(cls):
        from little_kernel.core.passes import PASSES
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
        kernel = _define_pipeline_kernel()
        cls.code = kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=256)

    def test_tma_descriptor_kernel_params(self):
        # Descriptor params keep their CUtensorMap identity (no uint64 cast).
        self.assertIn("CUtensorMap", self.code)

    def test_tma_2d_load_and_cluster_multicast_mask(self):
        self.assertIn("cp.async.bulk.tensor.2d", self.code)
        self.assertIn("multicast::cluster", self.code)
        # The draft's 2-CTA multicast mask (0b11) reaches the call site.
        self.assertIn("tma_load_2d_1sm_mcast_fn", self.code)
        self.assertRegex(self.code, r"tma_load_2d_1sm_mcast_fn\([^;]*, 3,")

    def test_mbarrier_lifecycle(self):
        self.assertIn("mbarrier.init", self.code)
        self.assertIn("mbarrier.arrive.expect_tx", self.code)
        self.assertIn("mbarrier.try_wait", self.code)
        # Remote arrive targets a peer CTA in the cluster.
        self.assertIn("mbarrier.arrive", self.code)

    def test_cluster_sync_brackets(self):
        self.assertIn("barrier.cluster.arrive", self.code)
        self.assertIn("barrier.cluster.wait", self.code)

    def test_wgmma_descriptors_and_instruction(self):
        # Descriptors assembled from cvta + offsets via mov.b64.
        self.assertIn("__cvta_generic_to_shared", self.code)
        self.assertIn("mov.b64", self.code)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", self.code)

    def test_stmatrix_x2_epilogue_and_named_barrier(self):
        self.assertIn("stmatrix.sync.aligned.x2.m8n8.shared.b16", self.code)
        self.assertIn("stmatrix_x2_fn", self.code)
        self.assertIn("named_barrier_sync_fn(0, 256)", self.code)


if __name__ == "__main__":
    unittest.main()
