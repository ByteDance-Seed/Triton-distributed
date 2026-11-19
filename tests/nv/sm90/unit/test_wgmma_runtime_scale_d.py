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
"""Regression coverage for runtime (traced) WGMMA ``scale_d``.

Semantic key: ``ir.sm90wgmma.bf16_ss_runtime_scaled``.

``ll.wgmma_mma_async_ss`` accepts a *traced runtime* int32 ``scale_d`` operand
(accumulate-vs-overwrite) in addition to a compile-time 0/1 immediate. The
runtime form must lower to a register predicate -- a ``setp.ne.b32 p, %r, 0``
inside the helper, with the value passed via an ``"r"(scale_d)`` constraint --
and must never be baked as a PTX immediate. A traced runtime value is the
only way to vary accumulate-vs-overwrite inside a single loop body without
unrolling it, which is the canonical K-loop shape of any bf16 GEMM.

These are CPU codegen tests only: they assert the lowered CUDA source carries
the runtime scale-d through the register-predicate form, that a dynamic-trip
loop zeroes an fp32 accumulator reborn per iteration, and that the composed
2-tile chained m64n128k16 bf16 GEMM (K-major first mma, MN-major second mma
via ``trans_a=1``) can drain through vectorized ``red.global.add.v2.f32``.
Paired GPU numerical validation is queued separately by the controller.
"""

import re
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _compile(fn):
    return fn.compile(PASSES["cuda"], codegen_cuda, num_threads=128)


@lk.ll_kernel(backend="cuda", is_entry=True)
def _bf16_ss_dynamic_loop_runtime_sd(x: ll.Tensor[ll.bfloat16], n: ll.int32) -> ll.void:
    """One bf16 GEMM K-loop with runtime (traced) scale_d.

    A single fp32 accumulator (loop-carried) is zeroed on the first k-step
    (``scale_d=0``) and accumulated afterwards (``scale_d=1``), so the loop
    body uses both the zero-init and accumulate forms with the *same* opcode.
    Shared-address descriptors are rebuilt per k-step from a runtime cvta'd
    base plus a byte offset so no operand is trace-time folded.
    """
    acc = ll.empty([64], ll.float32, scope="local")
    ll.wgmma_init_accum(acc, 64, "float")
    # Same-name assignment binds the array directly (no pointer alias).
    s_a = ll.alloc_shared_memory("s_a", ll.bfloat16, 64 * 128, 1024)
    s_b = ll.alloc_shared_memory("s_b", ll.bfloat16, 128 * 128, 1024)
    a_addr: ll.uint32 = ll.cvta_generic_to_shared(s_a)
    b_addr: ll.uint32 = ll.cvta_generic_to_shared(s_b)
    for it in range(n):  # dynamic trip: scale_d varies per iteration
        ll.wgmma_fence()
        for ks in ll.unroll(range(8)):
            sd: ll.int32 = 0
            if it > 0 or ks > 0:
                sd = 1
            da: ll.uint64 = ll.encode_wgmma_smem_desc(a_addr + ks * 32, 0, 1024, "swizzle_128b")
            db: ll.uint64 = ll.encode_wgmma_smem_desc(b_addr + ks * 32, 0, 1024, "swizzle_128b")
            ll.wgmma_mma_async_ss(acc, da, db, 128, 'bf16', 1, 1, sd, 0, 0)
        ll.wgmma_commit()
        ll.wgmma_wait_group(0)


@lk.ll_kernel(backend="cuda", is_entry=True)
def _bf16_ss_loop_carried_and_reborn(n: ll.int32) -> ll.void:
    """Loop-carried ``dq`` (declared above the loop) plus an accumulator
    ``acc`` reborn per iteration (declared inside the loop body)."""
    dq = ll.empty([64], ll.float32, scope="local")
    ll.wgmma_init_accum(dq, 64, "float")
    # Same-name assignment binds the array directly (no pointer alias).
    s_a = ll.alloc_shared_memory("s_a", ll.bfloat16, 64 * 128, 1024)
    s_b = ll.alloc_shared_memory("s_b", ll.bfloat16, 128 * 128, 1024)
    a_addr: ll.uint32 = ll.cvta_generic_to_shared(s_a)
    b_addr: ll.uint32 = ll.cvta_generic_to_shared(s_b)
    da: ll.uint64 = ll.encode_wgmma_smem_desc(a_addr, 0, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(b_addr, 0, 1024, "swizzle_128b")
    one: ll.int32 = 1
    for it in range(n):
        acc = ll.empty([64], ll.float32, scope="local")
        ll.wgmma_init_accum(acc, 64, "float")
        ll.wgmma_fence()
        ll.wgmma_mma_async_ss(acc, da, db, 128, 'bf16', 1, 1, one, 0, 0)
        ll.wgmma_mma_async_ss(dq, da, db, 128, 'bf16', 1, 1, one, 0, 0)
        ll.wgmma_commit()
        ll.wgmma_wait_group(0)


@lk.ll_kernel(backend="cuda", is_entry=True)
def _bf16_ss_chained_2tile_with_red(out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    """2-tile chained m64n128k16 bf16 GEMM: first mma K-major; second mma
    consumes the first accumulator re-staged as an MN-major (``trans_a=1``)
    operand. Two independent runtime ``scale_d`` predicates; fp32 result
    drained by vectorized ``red.global.add.v2.f32``."""
    acc1 = ll.empty([64], ll.float32, scope="local")
    acc2 = ll.empty([64], ll.float32, scope="local")
    ll.wgmma_init_accum(acc1, 64, "float")
    ll.wgmma_init_accum(acc2, 64, "float")
    # Shared tiles are compile-test stand-ins only (never loaded on CPU);
    # keep static shared under the 48KB ptxas limit: 8KB + 16KB + 8KB.
    s_a = ll.alloc_shared_memory("s_a", ll.bfloat16, 64 * 64, 1024)
    s_b = ll.alloc_shared_memory("s_b", ll.bfloat16, 128 * 64, 1024)
    s_c = ll.alloc_shared_memory("s_c", ll.bfloat16, 64 * 64, 1024)
    a_addr: ll.uint32 = ll.cvta_generic_to_shared(s_a)
    b_addr: ll.uint32 = ll.cvta_generic_to_shared(s_b)
    c_addr: ll.uint32 = ll.cvta_generic_to_shared(s_c)
    dc: ll.uint64 = ll.encode_wgmma_smem_desc(c_addr, 0, 1024, "swizzle_128b")
    i: ll.int32 = ll.threadIdx_x()
    for it in range(n):
        sd1: ll.int32 = 0
        if it > 0:
            sd1 = 1
        ll.wgmma_fence()
        for ks in ll.unroll(range(8)):
            da: ll.uint64 = ll.encode_wgmma_smem_desc(a_addr + ks * 32, 0, 1024, "swizzle_128b")
            db: ll.uint64 = ll.encode_wgmma_smem_desc(b_addr + ks * 32, 0, 1024, "swizzle_128b")
            # K-major accumulate into acc1, zero on first k-step
            ll.wgmma_mma_async_ss(acc1, da, db, 128, 'bf16', 1, 1, sd1, 0, 0)
        ll.wgmma_commit()
        ll.wgmma_wait_group(0)
        # (stage acc1 through 128B-swizzled shared as MN-major operand s_c)
        sd2: ll.int32 = 0
        if it > 0:
            sd2 = 1
        ll.wgmma_fence()
        db2: ll.uint64 = ll.encode_wgmma_smem_desc(b_addr, 0, 1024, "swizzle_128b")
        # MN-major chained mma: A operand column-wise from shared
        ll.wgmma_mma_async_ss(acc2, dc, db2, 128, 'bf16', 1, 1, sd2, 1, 0)
        ll.wgmma_commit()
        ll.wgmma_wait_group(0)
        for j in ll.unroll(range(0, 64, 2)):
            v0: ll.float32 = acc2[j]
            v1: ll.float32 = acc2[j + 1]
            ll.atomic_add_f32x2(out, i * 64 + j, v0, v1)


class TestRuntimeScaleDRegisterPredicate(unittest.TestCase):
    """Traced runtime ``scale_d`` must lower to a register predicate."""

    def test_runtime_scale_d_is_register_predicate_not_immediate(self):
        src = _compile(_bf16_ss_dynamic_loop_runtime_sd)
        # Value passed to helper via an "r" register operand...
        self.assertIn('"l"(desc_a), "l"(desc_b), "r"(scale_d)', src)
        # ...tested by a setp into a predicate register...
        self.assertRegex(src, r"setp\.ne\.b32 p, %\d+, 0;")
        # ...by a helper that takes an int scale_d parameter.
        self.assertIn(", int scale_d)", src)
        # The bf16 m64n128k16 SS opcode is emitted.
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", src)
        # 64 read-write fp32 accumulator constraints for m64n128k16.
        self.assertGreaterEqual(src.count('"+f"'), 64)
        # The runtime-scale-d helper form (drt) is used, not a baked
        # immediate helper (d0/d1).
        self.assertIn("drta0tb0_fn", src)


class TestDynamicLoopAndRuntimeDescriptors(unittest.TestCase):
    """The dynamic-trip loop survives lowering with runtime descs."""

    def test_dynamic_loop_preserved_with_runtime_descriptor_offsets(self):
        src = _compile(_bf16_ss_dynamic_loop_runtime_sd)
        # Dynamic-trip loop: a comparison against n guards a backward branch
        # (not an unrolled constant trip count).
        has_n_bound = (re.search(r"for\s*\([^;]*;[^;]*<\s*n\b", src) or re.search(r"<\s*n[;) ]", src))
        self.assertIsNotNone(has_n_bound, "expected runtime loop bound on n")
        # Byte-offset arithmetic on cvta'd shared addresses stays runtime:
        # the descriptor encoder is invoked on an address expression inside
        # the loop body, not a trace-time-folded constant.
        self.assertRegex(src, r"encode_wgmma_smem_desc\w*\([^)]*\+")

    def test_accumulator_rebirth_inside_loop_and_carry_above(self):
        src = _compile(_bf16_ss_loop_carried_and_reborn)
        # Both accumulators present as 64-float locals.
        self.assertRegex(src, r"float\s+dq\[64\];")
        self.assertRegex(src, r"float\s+acc\[64\];")
        # dq (declared above the loop) is declared-and-zeroed before the
        # loop; acc (declared inside) is reborn per iteration.
        loop_m = re.search(r"for\s*\(", src)
        self.assertIsNotNone(loop_m, "expected a loop in emitted source")
        dq_pos = src.index("dq[64]")
        acc_pos = src.index("acc[64]")
        self.assertLess(dq_pos, loop_m.start(), "loop-carried dq must be declared above the loop")
        self.assertGreater(acc_pos, loop_m.start(), "per-iteration acc must be declared inside the loop")


class TestChained2TileKMajorMNMajor(unittest.TestCase):
    """K-major first mma chained into an MN-major second mma."""

    def test_chained_2tile_emits_kmajor_and_mnmajor_with_two_predicates(self):
        src = _compile(_bf16_ss_chained_2tile_with_red)
        # K-major form: trans immediates 0,0 -> helper token ta0tb0.
        self.assertIn("drta0tb0_fn", src)
        # MN-major form: trans_a=1 -> helper token ta1tb0.
        self.assertIn("drta1tb0_fn", src)
        # Both are m64n128k16 bf16 SS SS ops.
        self.assertGreaterEqual(src.count("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16"), 2)
        # Two independent runtime predicates: two "r"(scale_d) operand lists.
        self.assertGreaterEqual(src.count('"l"(desc_a), "l"(desc_b), "r"(scale_d)'), 2)
        # trans_a=1 must appear as an instruction immediate in the MN-major
        # asm body.
        m = re.search(r"wgmma\.mma_async\.sync\.aligned\.m64n128k16\.f32\.bf16\.bf16\\n\"\n"
                      r".*?%\d+,%\d+,p,1,1,1,0;", src, re.DOTALL)
        self.assertIsNotNone(m, "expected trans_a=1, trans_b=0 immediates in MN-major wgmma")

    def test_vectorized_red_global_add_epilogue(self):
        src = _compile(_bf16_ss_chained_2tile_with_red)
        self.assertIn("red.global.add.v2.f32", src)


if __name__ == "__main__":
    unittest.main()
