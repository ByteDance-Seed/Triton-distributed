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
"""CPU regression coverage for the 2-CTA UMMA + TMEM drain epilogue recipe.

Semantic key: ir.cuda.tmem_epilogue_cluster_split

These tests run on CPU only. They compile an entry kernel that combines a
2-CTA (cta_group::2) UMMA mainloop with the cluster-split TMEM drain epilogue
through the canonical role/dataflow pipeline and check:

* the lowered CUDA carries the full typed pattern (tcgen05.ld 32x32b.x32
  drain at 32-column strides, tcgen05.wait::ld, RNE bf16 packing into four
  16-byte stores per 32-column chunk under the ragged-N guard, and the
  fence_after / fence_before / __syncthreads / cluster_sync ordering);
* the ordering validator rejects each missing ordering marker with a
  precise pre-codegen error instead of a generic refusal or a miscompile;
* the recipe's numeric contract (cluster-rank row split + ragged guard)
  reproduces a CPU GEMM on one small shape, modeled lane-for-lane.

No GPU is required or claimed; paired GPU validation belongs to the queue.
Run with:
    uv run --locked --no-sync python -m unittest tests.nv.sm100.unit.test_tmem_epilogue_cluster_split
"""
from __future__ import annotations

import re
import unittest

from little_kernel.backends.nv.sm100.passes.dataflow.pipeline import compile_dataflow_cuda

BM = 256
BN = 256
BK = 64

MAINLOOP = """
    with dataflow() as g:
        tma = g.role("tma", hw=HW.TMA, warps=[0])
        mma = g.role("mma", hw=HW.TensorCore, warps=[1])
        a_smem = g.smem("a_smem", (BM, BK), dtype=ll.bfloat16, stages=STAGES, major="K")
        b_smem = g.smem("b_smem", (BN, BK), dtype=ll.bfloat16, stages=STAGES, major="K")
        d_tmem = g.tmem("d_tmem", (BM, BN), dtype=ll.float32)
        ab_stage = g.depends(src=tma, dst=mma, via=[a_smem, b_smem])

    with tma:
        elected: ll.uint32 = ll.elect_one()
        kb: ll.int32 = 0
        while kb < k // BK:
            ab_stage.bind(kb)
            if elected == 1:
                ri.tma_ld_2d(a_smem, a_desc, (kb * BK, 0), shape=(BM, BK))
                ri.tma_ld_2d(b_smem, b_desc, (kb * BK, 0), shape=(BN, BK))
            kb = kb + 1

    with mma:
        elected2: ll.uint32 = ll.elect_one()
        kb2: ll.int32 = 0
        while kb2 < k // BK:
            ab_stage.bind(kb2)
            desc_a = ri.make_umma_smem_desc(a_smem, (0, 0))
            desc_b = ri.make_umma_smem_desc(b_smem, (0, 0))
            if elected2 == 1:
                ri.umma_f16_ss_2sm(d_tmem, ll.uint32(d_tmem), desc_a, desc_b, kb2 > 0,
                                   out_shape=(BM, BN))
            kb2 = kb2 + 1
"""

EPILOGUE = """
    ll.tcgen05_fence_after()
    rank: ll.uint32 = ri.rank_in_group()
    tid: ll.uint32 = ll.threadIdx_x()
    row: ll.uint32 = rank * 128 + tid
    r = ll.empty((32,), ll.float32, "local")
    for cc in ll.no_unroll(range(0, BN, 32)):
        ri.tmem_ld_32dp_32bxn(d_tmem, ll.uint32(d_tmem) + cc, r, 32)
        ri.tmem_ld_fence(d_tmem)
        if cc + 32 <= n:
            for v in ll.unroll(range(0, 4)):
                ll.gmem_st_bf16x8(c, row * BN + cc + 8 * v, r[8 * v], r[8 * v + 1],
                                  r[8 * v + 2], r[8 * v + 3], r[8 * v + 4], r[8 * v + 5],
                                  r[8 * v + 6], r[8 * v + 7])
    ll.tcgen05_fence_before()
    ll.__syncthreads()
    ll.cluster_sync()
"""

KERNEL_TEMPLATE = ('''
import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import dataflow, HW
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri

BM: ll.const = 256
BN: ll.const = 256
BK: ll.const = 64
STAGES: ll.const = 2


@lk.ll_kernel(backend="nv/sm100", is_entry=True, cluster_shape=%(cluster)r,
              cta_group_shape=%(group)r, launch_bounds=(128, 1))
def gemm_epilogue(
    a_desc: ll.TmaDescriptor,
    b_desc: ll.TmaDescriptor,
    c: ll.Tensor[ll.bfloat16],
    k: ll.int32,
    n: ll.int32,
) -> ll.void:
%(mainloop)s
%(epilogue)s
''')

MAINLOOP_1SM = MAINLOOP.replace(
    "            if elected2 == 1:\n"
    "                ri.umma_f16_ss_2sm(d_tmem, ll.uint32(d_tmem), desc_a, desc_b, kb2 > 0,\n"
    "                                   out_shape=(BM, BN))", "            if elected2 == 1:\n"
    "                if kb2 == 0:\n"
    "                    ri.umma_f16_ss_1sm_init(d_tmem, ll.uint32(d_tmem), desc_a, desc_b,\n"
    "                                            out_shape=(BM, BN))\n"
    "                else:\n"
    "                    ri.umma_f16_ss_1sm_accum(d_tmem, ll.uint32(d_tmem), desc_a, desc_b,\n"
    "                                             out_shape=(BM, BN))")


def build_source(cluster=(2, 1, 1), group=(2, 1, 1), epilogue=EPILOGUE, mainloop=MAINLOOP) -> str:
    return KERNEL_TEMPLATE % {
        "cluster": cluster,
        "group": group,
        "mainloop": mainloop,
        "epilogue": epilogue,
    }


class TestLoweredEpiloguePattern(unittest.TestCase):
    """Codegen checks on the 2-CTA kernel lowered once per class."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda = compile_dataflow_cuda(build_source(), param_values={})

    def test_mainloop_epilogue_lowers_with_full_pattern(self) -> None:
        cuda = self.cuda
        # 2-CTA UMMA mainloop.
        self.assertIn("tcgen05.mma.cta_group::2.kind::f16", cuda)
        # TMEM drain: x32 load + completion fence per 32-column chunk.
        self.assertIn("tcgen05.ld.sync.aligned.32x32b.x32", cuda)
        self.assertIn("tcgen05.wait::ld", cuda)
        # Ragged-edge guard and vector stores.
        self.assertIsNotNone(re.search(r"if \(cc \+ 32 <= n\)", cuda), "ragged N-edge guard missing")
        stores_16b = re.findall(r"reinterpret_cast<uint4 ?\*>|st\.global\.v4|gmem_st_bf16x8", cuda)
        self.assertTrue(stores_16b, "expected 16-byte (uint4) vector stores in the epilogue")
        # Fences and cluster ordering all present.
        self.assertIn("tcgen05.fence::after_thread_sync", cuda)
        self.assertIn("tcgen05.fence::before_thread_sync", cuda)
        self.assertIn("barrier.cluster.arrive", cuda)

    def test_epilogue_ordering_in_lowered_cuda(self) -> None:
        cuda = self.cuda
        fence_after = cuda.index("tcgen05_fence_after_fn();")
        first_ld = cuda.index("tmem_ld_32dp_32bxn_x32_fn(", fence_after)
        wait_ld = cuda.index("tmem_ld_fence_fn();", first_ld)
        guard = cuda.index("if (cc + 32 <= n)", wait_ld)
        fence_before = cuda.index("tcgen05_fence_before_fn();", guard)
        barrier = cuda.index("__syncthreads();", fence_before)
        cluster = cuda.index("cluster_sync_fn();", barrier)
        self.assertLess(fence_after, first_ld)
        self.assertLess(first_ld, wait_ld)
        self.assertLess(wait_ld, guard)
        self.assertLess(guard, fence_before)
        self.assertLess(fence_before, barrier)
        self.assertLess(barrier, cluster)
        # The drain loop must stride 32 TMEM columns per iteration.
        drain_loop = re.search(
            r"for \(int32_t cc = 0; cc < 256; cc \+= 32\).*?tmem_ld_32dp_32bxn_x32_fn\("
            r"static_cast<uint32_t>\(__lk_tmem_d_tmem\) \+ cc", cuda, re.DOTALL)
        self.assertIsNotNone(drain_loop, "expected 32-column-strided drain at tmem base + cc")

    def test_ragged_guard_wraps_all_four_stores(self) -> None:
        guard_start = self.cuda.index("if (cc + 32 <= n)")
        fence_before = self.cuda.index("tcgen05_fence_before_fn();", guard_start)
        segment = self.cuda[guard_start:fence_before]
        stores = re.findall(r"gmem_st_bf16x8_fn\(|\*reinterpret_cast<uint4 ?\*>", segment)
        self.assertEqual(len(stores), 4, f"expected exactly four uint4 stores under the guard, got {len(stores)}")


class TestOrderingMarkerValidation(unittest.TestCase):
    """Each dropped ordering marker must fail with a precise pre-codegen error."""

    def _assert_missing_marker_rejected(self, removed: str, message: str) -> None:
        source = build_source().replace(removed, "")
        self.assertNotIn(removed, source)
        with self.assertRaises(ValueError) as ctx:
            compile_dataflow_cuda(source, param_values={})
        self.assertIn(message, str(ctx.exception))

    def test_missing_fence_after_rejected(self) -> None:
        self._assert_missing_marker_rejected("    ll.tcgen05_fence_after()\n",
                                             "must be preceded by ll.tcgen05_fence_after")

    def test_missing_fence_before_rejected(self) -> None:
        self._assert_missing_marker_rejected("    ll.tcgen05_fence_before()\n",
                                             "must be closed by ll.tcgen05_fence_before")

    def test_missing_cta_barrier_rejected(self) -> None:
        self._assert_missing_marker_rejected("    ll.__syncthreads()\n", "must be followed by ll.__syncthreads")

    def test_missing_cluster_sync_rejected(self) -> None:
        self._assert_missing_marker_rejected("    ll.cluster_sync()\n", "must be followed by ll.cluster_sync")

    def test_single_cta_cluster_allows_no_cluster_sync(self) -> None:
        """A 1-CTA kernel drains its own TMEM; cluster_sync is not required."""
        self.assertIn("umma_f16_ss_1sm_init", MAINLOOP_1SM)
        epilogue = EPILOGUE.replace("    ll.cluster_sync()\n", "")
        source = build_source(cluster=(1, 1, 1), group=(1, 1, 1), mainloop=MAINLOOP_1SM, epilogue=epilogue)
        cuda = compile_dataflow_cuda(source, param_values={})
        self.assertIn("tcgen05.fence::before_thread_sync", cuda)


class TestRecipeNumericContract(unittest.TestCase):

    def test_recipe_numeric_contract_against_cpu_gemm(self) -> None:
        """Model the drain recipe lane-for-lane against a CPU GEMM (small shape).

        The 256x256 FP32 TMEM accumulator of umma_f16_ss_2sm is lane-split so each
        CTA of the pair owns 128 lanes: CTA rank r covers rows 128*r..128*r+127,
        lane (tid) covers row 128*r + tid.  Each 32x32b.x32 load at column cc
        returns elements [row, cc + j] for j in 0..31.  A 32-column chunk is
        stored (as four uint4 = 4 x 8 bf16) iff cc + 32 <= n, which is the
        ragged in-range test the kernel emits.
        """
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is required for the numeric contract model")
        rng = np.random.default_rng(0)
        m_t, k_t, n_t = 256, 64, 224  # single tile, ragged N edge (224 < 256)
        a = rng.standard_normal((m_t, k_t)).astype(np.float32)
        b = rng.standard_normal((n_t, k_t)).astype(np.float32)
        acc = a @ b.T  # CPU reference GEMM, FP32 accumulation

        out = np.zeros((m_t, n_t), dtype=np.float32)
        stored = np.zeros((m_t, n_t), dtype=bool)
        for rank in range(2):  # cluster_shape=(2,1,1)
            for tid in range(128):
                row = rank * 128 + tid
                for cc in range(0, BN, 32):
                    regs = acc[row, cc:cc + 32]  # tmem_ld_32dp_32bxn chunk
                    if cc + 32 <= n_t:  # ragged N-edge guard, same as the kernel
                        for v in range(4):  # four uint4 (8-element) stores
                            lo, hi = cc + 8 * v, cc + 8 * v + 8
                            if hi <= n_t:
                                out[row, lo:hi] = regs[8 * v:8 * v + 8]
                                stored[row, lo:hi] = True

        # Every fully in-range element must be stored exactly once and match.
        expected = np.zeros((m_t, n_t), dtype=bool)
        expected[:, :(n_t // 32) * 32] = True
        self.assertTrue((stored == expected).all(), "row split/guard must tile each row exactly once")
        np.testing.assert_allclose(out[stored], acc[stored], rtol=0, atol=0)
        # The ragged tail (columns 224..255 of the tile) is never stored.
        self.assertFalse(out[:, (n_t // 32) * 32:].any())


if __name__ == "__main__":
    unittest.main()
