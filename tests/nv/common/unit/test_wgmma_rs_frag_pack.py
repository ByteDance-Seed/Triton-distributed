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
"""CPU regression tests for the sm90 RS WGMMA A-fragment pack capability
(``ir.wgmma.rs.fragment_pack_from_accum``).

Before this capability the only typed spellings that fed the
``wgmma_mma_async_rs`` / ``wgmma_m64n64k16_rs`` A operand were raw asm
(forbidden to frontends) or a shared-memory round trip of the P tile (extra
traffic plus barriers, breaking the flash-attention epilogue layout). The
fragment pack family:

* ``cvt_bf16x2_f32_u32`` / ``cvt_f16x2_f32_u32``: scalar two-value pair pack
  through the single PTX ``cvt.rn.{bf16,f16}x2.f32`` instruction;
* ``wgmma_pack_rs_frag``: fragment-level op, fills a ``[4]uint32`` local with
  the A-fragment words of one k16 step from FP32 accumulator values in
  registers, paired ``frag[j] = {hi = acc[8*ks+2*j+1], lo = acc[8*ks+2*j]}``.

These tests compose no GPU code: codegen_func bodies and full-kernel CPU
codegen only. Paired GPU validation is left to the queue.
"""

import re
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.sm90.language.intrin import wgmma as W


class _TestScalarPackCodegen(unittest.TestCase):
    """The pair-pack builtins emit exactly one cvt instruction each."""

    def test_bf16_pair_pack_shape(self):
        b = W.codegen_cvt_bf16x2_f32_u32(lo="x", hi="y")
        # One rounding boundary: the literal single PTX instruction.
        self.assertEqual(b.body.count("cvt.rn.bf16x2.f32"), 1)
        self.assertNotIn("cvt.rn.f16x2.f32", b.body)
        # cvt.rn.bf16x2.f32 d, a, b: a -> HIGH lane, b -> LOW lane, so the
        # (lo, hi) spelling binds operand a=hi (%2) and b=lo (%1).
        self.assertIn('"cvt.rn.bf16x2.f32 %0, %2, %1;"', b.body)
        self.assertIn('"=r"(d)', b.body)
        self.assertIn('"f"(lo), "f"(hi)', b.body)
        self.assertIn("lk_cvt_bf16x2_f32_u32_fn(x, y)", b.return_val)

    def test_f16_pair_pack_shape(self):
        b = W.codegen_cvt_f16x2_f32_u32(lo="x", hi="y")
        self.assertEqual(b.body.count("cvt.rn.f16x2.f32"), 1)
        self.assertNotIn("cvt.rn.bf16x2.f32", b.body)
        self.assertIn("lk_cvt_f16x2_f32_u32_fn(x, y)", b.return_val)


class _TestFragPackCodegen(unittest.TestCase):
    """wgmma_pack_rs_frag emits the 4 cvt pairs with the documented pairing."""

    def test_bf16_default(self):
        b = W.codegen_wgmma_pack_rs_frag(frag="frag", acc="acc", ks=1)
        self.assertEqual(b.body.count("cvt.rn.bf16x2.f32"), 4)
        # Documented pairing: frag[j] = {hi = acc[8*ks+2*j+1], lo = acc[8*ks+2*j]}
        # and cvt's a operand -> HIGH lane, b operand -> LOW lane.
        for j in range(4):
            self.assertIn(f': "=r"(frag[{j}]) : "f"(acc[8 * ks + {2*j}]), '
                          f'"f"(acc[8 * ks + {2*j+1}]));', b.body)
        self.assertIn("wgmma_pack_rs_frag_bf16_fn(uint32_t* frag, const float* acc, "
                      "int ks)", b.body)
        self.assertIn("wgmma_pack_rs_frag_bf16_fn(frag, acc, 1)", b.return_val)

    def test_f16_variant(self):
        b = W.codegen_wgmma_pack_rs_frag(frag="frag", acc="acc", ks=0, dtype="f16")
        self.assertEqual(b.body.count("cvt.rn.f16x2.f32"), 4)
        self.assertIn("wgmma_pack_rs_frag_f16_fn(frag, acc, 0)", b.return_val)
        # dtype normalization accepts aliases, e.g. "half" -> "f16".
        b2 = W.codegen_wgmma_pack_rs_frag(frag="frag", acc="acc", ks=0, dtype="half")
        self.assertIn("cvt.rn.f16x2.f32", b2.body)

    def test_legality_whitelist(self):
        # dtype passes through the shared f16/bf16 whitelist: illegal tokens
        # raise before any asm text is composed.
        for bad in ("tf32", "fp8", "s8", "", "bf16x", 0, None):
            with self.assertRaises(ValueError):
                W.codegen_wgmma_pack_rs_frag(frag="f", acc="a", ks=0, dtype=bad)

    def test_no_smem_traffic_in_pack(self):
        # The pack is a pure register-to-register conversion: no shared/global
        # memory instructions, no barriers, no accumulator copies.
        b = W.codegen_wgmma_pack_rs_frag(frag="frag", acc="acc", ks=2)
        for forbidden in ("ld.shared", "st.shared", "bar.sync", "__syncthreads", "cp.async", "memcpy"):
            self.assertNotIn(forbidden, b.body)


@lk.ll_kernel(backend='cuda', is_entry=True)
def _rs_pack_pv_repro(o: ll.Tensor[ll.float32], v_desc: ll.uint64) -> ll.void:
    """16x64 P tile in FP32 accumulators -> pack -> RS WGMMA (PV stage shape).

    Four k16 steps of m64n64k16: each step packs its 8-accumulator window
    into the [4]uint32 fragment and feeds wgmma_mma_async_rs, with
    wgmma.fence separating the accumulator-dependent reads — the
    flash-attention P@V epilogue data flow. Written as trace-time-constant
    k-step indices so scale_d is a PTX immediate per step.
    """
    acc = ll.empty([32], ll.float32, scope="local")
    frag = ll.empty([4], ll.uint32, scope="local")
    ll.wgmma_init_accum(acc, 32, ll.float32)
    ll.wgmma_pack_rs_frag(frag, acc, 0)
    ll.wgmma_fence()
    ll.wgmma_mma_async_rs(acc, frag[0], frag[1], frag[2], frag[3], v_desc, 64, dtype="bf16", scale_d=0)
    ll.wgmma_commit()
    ll.wgmma_pack_rs_frag(frag, acc, 1)
    ll.wgmma_fence()
    ll.wgmma_mma_async_rs(acc, frag[0], frag[1], frag[2], frag[3], v_desc, 64, dtype="bf16", scale_d=1)
    ll.wgmma_commit()
    ll.wgmma_pack_rs_frag(frag, acc, 2)
    ll.wgmma_fence()
    ll.wgmma_mma_async_rs(acc, frag[0], frag[1], frag[2], frag[3], v_desc, 64, dtype="bf16", scale_d=1)
    ll.wgmma_commit()
    ll.wgmma_pack_rs_frag(frag, acc, 3)
    ll.wgmma_fence()
    ll.wgmma_mma_async_rs(acc, frag[0], frag[1], frag[2], frag[3], v_desc, 64, dtype="bf16", scale_d=1)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    o[i] = acc[0]


@lk.ll_kernel(backend='cuda', is_entry=True)
def _rs_pack_pv_loop(o: ll.Tensor[ll.float32], v_desc: ll.uint64) -> ll.void:
    """Runtime-k spelling: the pack also composes inside a dynamic k loop.

    ks flows to the pack helper as a runtime int argument and scale_d through
    the register-predicate path, proving the fragment local stays addressable
    as an lvalue across loop-carried asm calls.
    """
    acc = ll.empty([32], ll.float32, scope="local")
    frag = ll.empty([4], ll.uint32, scope="local")
    ll.wgmma_init_accum(acc, 32, ll.float32)
    for ks in range(4):
        ll.wgmma_pack_rs_frag(frag, acc, ks)
        ll.wgmma_fence()
        ll.wgmma_mma_async_rs(acc, frag[0], frag[1], frag[2], frag[3], v_desc, 64, dtype="bf16", scale_d=(ks > 0))
        ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    o[i] = acc[0]


@lk.ll_kernel(backend='cuda', is_entry=True)
def _scalar_pack_repro(o: ll.Tensor[ll.uint32]) -> ll.void:
    """Per-lane spelling: the same packing spelled with the scalar builtin."""
    lo: ll.float32 = ll.f32(1.5)
    hi: ll.float32 = ll.f32(-2.25)
    word: ll.uint32 = ll.cvt_bf16x2_f32_u32(lo, hi)
    i: ll.int32 = ll.threadIdx_x()
    o[i] = word


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda)


class _TestFrontendComposition(unittest.TestCase):
    """Full-frontend lowering: accumulator locals -> pack -> RS wgmma."""

    def test_pv_epilogue_composition(self):
        src = _compile(_rs_pack_pv_repro)
        # One pack helper holding the 4 cvt.rn.bf16x2.f32 pairs is emitted;
        # it is called once per k-step trace-time index (0..3), for a total
        # of 16 pack instructions per 16x64 bf16 P tile at runtime.
        self.assertEqual(src.count("cvt.rn.bf16x2.f32"), 4)
        for ks in range(4):
            self.assertIn(f"wgmma_pack_rs_frag_bf16_fn(frag, acc, {ks})", src)
        # The first k-step overwrites (scale_d=0) then accumulates (1):
        # distinct helper bodies per scale_d immediate, each defined once
        # and called per matching k-step.
        self.assertEqual(src.count("wgmma_mma_rs_n64_bf16_sa1sb1d0tb0_fn("), 2)  # def + 1 call
        self.assertEqual(src.count("wgmma_mma_rs_n64_bf16_sa1sb1d1tb0_fn("), 4)  # def + 3 calls
        # RS wgmma issues with the register fragment {a0..a3} operand group.
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", src)
        # Register-A operand group: four .b32 A registers then desc_b + the
        # p predicate, the two operand scales and the trans_b immediate
        # (no trans_a on the RS float path).
        m = re.search(r"\{%\d+,%\d+,%\d+,%\d+\},%\d+,p,1,1,0;\\n\}", src)
        self.assertIsNotNone(m)
        # RS float path takes no trans_a immediate; trans_b=0 default.
        self.assertIn(",0;\\n}", src)
        # Synchronization spine present.
        for token in ("wgmma.fence.sync.aligned", "wgmma.commit_group.sync.aligned", "wgmma.wait_group.sync.aligned 0"):
            self.assertIn(token, src)
        # In the kernel body each pack call precedes its wgmma invocation
        # (definition order in the helper preamble is irrelevant).
        body = src[src.find("_rs_pack_pv_repro(float* o"):]
        for ks in range(4):
            pack_at = body.find(f"wgmma_pack_rs_frag_bf16_fn(frag, acc, {ks})")
            self.assertNotEqual(pack_at, -1)
            tail = body[pack_at:]
            self.assertLess(tail.find("wgmma_fence_fn()"), tail.find("wgmma_mma_rs_n64_bf16"))

    def test_runtime_ks_loop_composition(self):
        # The pack also composes inside a dynamic k-loop: ks flows as a
        # runtime int argument and scale_d via the register-predicate path.
        src = _compile(_rs_pack_pv_loop)
        body = src[src.find("_rs_pack_pv_loop(float* o"):]
        self.assertIn("wgmma_pack_rs_frag_bf16_fn(frag, acc, ks)", body)
        self.assertIn("wgmma_mma_rs_n64_bf16_sa1sb1drtb0_fn(", src)
        self.assertIn('"r"(scale_d)', src)

    def test_pack_consumes_fronted_frag_words(self):
        # The fragment local must be addressable as an lvalue across the asm
        # call: codegen passes the array, and a0..a3 reads type-check as
        # uint32 scalars fed to the RS builtin ("r" constraints).
        src = _compile(_rs_pack_pv_repro)
        self.assertIn("uint32_t frag[4]", src)
        self.assertIn("float acc[32]", src)

    def test_scalar_pack_frontend(self):
        src = _compile(_scalar_pack_repro)
        self.assertIn("lk_cvt_bf16x2_f32_u32_fn", src)
        self.assertIn("cvt.rn.bf16x2.f32", src)


class _TestExistingFamiliesUnchanged(unittest.TestCase):
    """Regression guard: adding the pack family must not perturb RS/SS asm."""

    def test_rs_bf16_shape_unchanged(self):
        b = W.codegen_wgmma_mma_async_rs(acc="c", a0="a0", a1="a1", a2="a2", a3="a3", desc_b="db", n=64, dtype="bf16")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", b.body)
        self.assertIn("wgmma_mma_rs_n64_bf16_sa1sb1d1tb0_fn", b.body)
        self.assertIn('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b)', b.body)

    def test_ss_bf16_shape_unchanged(self):
        b = W.codegen_wgmma_mma_async_ss(acc="c", desc_a="da", desc_b="db", n=64, dtype="bf16")
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16", b.body)


if __name__ == "__main__":
    unittest.main()
