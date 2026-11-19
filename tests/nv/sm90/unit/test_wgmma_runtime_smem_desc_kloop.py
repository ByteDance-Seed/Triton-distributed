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
"""Regression coverage for runtime (loop-varying) WGMMA smem descriptors.

Semantic key: ir.wgmma_mma_async_ss.runtime_smem_desc.

``ll.wgmma_mma_async_ss`` (and the ``wgmma_m64nNk16`` family) must accept
runtime uint64 smem descriptor values built in-kernel -- e.g. an
``encode_wgmma_smem_desc`` of a shared-memory base offset by a pipeline-stage
loop variable -- instead of requiring compile-time static descriptors. A
multi-stage producer/consumer WGMMA pipeline re-encodes its descriptors once
per K tile, so the desc_a/desc_b operands must reach the inline asm as ``l``
register operands (uint64_t helper parameters), never as values folded into
the asm template at compile time.

These are CPU codegen tests only: they assert the lowered CUDA source carries
the runtime descriptor expressions through the K-loop verbatim. Paired GPU
numerical validation is queued separately by the controller.
"""

import re
import unittest

import little_kernel as lk
import little_kernel.language as ll


def _compile(fn):
    from little_kernel.core.passes import PASSES
    from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
    return fn.compile(PASSES["cuda"], codegen_cuda, num_threads=128)


class TestWgmmaRuntimeSmemDescKLoop(unittest.TestCase):
    """A K-loop that re-encodes descriptors per stage must lower end to end."""

    def test_mma_async_ss_kloop_reencodes_descs_per_stage(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def wgmma_loop_desc(addr_in: ll.Tensor[ll.uint32]) -> ll.void:
            base: ll.uint32 = addr_in[0]
            acc = ll.empty([128], dtype=ll.float32, scope="local")
            for k in range(2):
                da = ll.encode_wgmma_smem_desc(base + k * 32, 0, 1024)
                db = ll.encode_wgmma_smem_desc(base + k * 2048, 8192, 1024)
                ll.wgmma_mma_async_ss(acc, da, db, 256)
                ll.wgmma_commit()
            ll.wgmma_wait_group(0)

        # Must not raise "operand count mismatch / non-constexpr descriptor".
        code = _compile(wgmma_loop_desc)

        # The descriptor values are declared once and recomputed per iteration
        # from the loop-varying shared base.
        self.assertIn("uint64_t da;", code)
        self.assertIn("uint64_t db;", code)
        self.assertIn("encode_wgmma_smem_desc_l1_fn(base + k * 32, 0, 1024)", code)
        self.assertIn("encode_wgmma_smem_desc_l1_fn(base + k * 2048, 8192, 1024)", code)

        # desc_a/desc_b reach the WGMMA helper as runtime arguments ...
        self.assertIn("wgmma_mma_ss_n256_bf16_sa1sb1d1ta0tb0_fn(acc, da, db)", code)
        # ... and the helper forwards them through 64-bit register constraints,
        # not as compile-time immediates baked into the asm template.
        self.assertIn("void wgmma_mma_ss_n256_bf16_sa1sb1d1ta0tb0_fn("
                      "float* c, uint64_t desc_a, uint64_t desc_b)", code)
        self.assertIn('"l"(desc_a), "l"(desc_b)', code)
        # The per-stage call site lives inside the K-loop body.
        loop = re.search(r"for \(int32_t k = 0;.*?\n    \}", code, re.S)
        self.assertIsNotNone(loop)
        self.assertIn("da = encode_wgmma_smem_desc", loop.group(0))
        self.assertIn("wgmma_commit_fn()", loop.group(0))

    def test_m64n256k16_direct_accepts_runtime_descs(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def wgmma_direct_loop(addr_in: ll.Tensor[ll.uint32]) -> ll.void:
            base: ll.uint32 = addr_in[0]
            acc = ll.empty([128], dtype=ll.float32, scope="local")
            for k in range(2):
                da = ll.encode_wgmma_smem_desc(base + k * 32, 0, 1024)
                db = ll.encode_wgmma_smem_desc(base + k * 2048, 8192, 1024)
                ll.wgmma_m64n256k16(acc, da, db)
                ll.wgmma_commit()
            ll.wgmma_wait_group(0)

        code = _compile(wgmma_direct_loop)
        self.assertIn("wgmma_m64n256k16_fn(acc, da, db)", code)
        self.assertIn("void wgmma_m64n256k16_fn(float* c, uint64_t desc_a, "
                      "uint64_t desc_b)", code)
        self.assertIn('"l"(desc_a), "l"(desc_b)', code)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", code)

    def test_inline_descriptor_expressions_need_no_temp(self):

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def wgmma_inline_desc(addr_in: ll.Tensor[ll.uint32]) -> ll.void:
            base: ll.uint32 = addr_in[0]
            acc = ll.empty([32], dtype=ll.float32, scope="local")
            for k in range(2):
                ll.wgmma_mma_async_ss(acc, ll.encode_wgmma_smem_desc(base + k * 32, 0, 1024),
                                      ll.encode_wgmma_smem_desc(base + k * 2048, 8192, 1024), 64)
                ll.wgmma_commit()
            ll.wgmma_wait_group(0)

        code = _compile(wgmma_inline_desc)
        # The fixed-n helper is still reached through uint64 runtime operands.
        self.assertIn("void wgmma_mma_ss_n64_bf16_sa1sb1d1ta0tb0_fn("
                      "float* c, uint64_t desc_a, uint64_t desc_b)", code)
        self.assertIn('"l"(desc_a), "l"(desc_b)', code)
        self.assertIn("base + k * 32", code)
        self.assertIn("base + k * 2048", code)


if __name__ == "__main__":
    unittest.main()
