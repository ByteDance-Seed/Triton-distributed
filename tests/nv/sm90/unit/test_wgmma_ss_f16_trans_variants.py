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
"""SM90 WGMMA-SS typed-IR CPU regression tests (no GPU required).

Semantic key: ir.wgmma.m64_multi_n16_f16_trans_variants.

Covers the sm90a capability ``ir.cuda.wgmma_ss_f16_m64nnk16_accum32f``:

* the typed WGMMA-SS m64nNk16 f16/bf16 family with 64-bit shared-memory
  matrix descriptor operands and fence/commit/wait_group warpgroup ops
  lowers to whitelisted PTX via CPU codegen;
* ``encode_wgmma_smem_desc`` accepts the documented compile-time layout
  selectors both as Python ints and as the C++ source text codegen hands
  positional literals (e.g. ``"1"``), and rejects illegal selectors with
  ``ValueError`` before any asm text is composed;
* the composed producer/consumer pipeline (mbarrier init /
  arrive_expect_tx / wait_parity + cp.async.bulk.tensor.2d TMA global ->
  shared + WGMMA-SS over the staged tile) lowers end-to-end.

When ``nvcc`` is on PATH, the emitted sources are additionally compiled
with ``-gencode arch=compute_90a,code=sm_90a`` as a pure CPU-side nvcc
acceptance gate (ptxas legality of the inline PTX); no device is used.
"""

import atexit
import importlib.util
import os
import re
import shutil
import subprocess
import tempfile
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.sm90.language.intrin.wgmma import (
    _wgmma_desc_layout_code, )


@lk.ll_kernel(backend='cuda', is_entry=True)
def _wgmma_ss_f16_repro(q: ll.Tensor[ll.float16], out: ll.Tensor[ll.float32]) -> ll.void:
    acc = ll.empty([32], ll.float32, scope="local")
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, 1)
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, 1)
    ll.wgmma_zero_accum(acc, 32, ll.float32)
    ll.wgmma_fence()
    ll.wgmma_mma_async_ss(acc, da, db, 64, dtype='f16', scale_d=1, trans_a=0, trans_b=0)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc[0]
    out[i] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _wgmma_ss_f16_qkT(out: ll.Tensor[ll.float32]) -> ll.void:
    """Q @ K^T form: B operand transposed via the trans_b immediate."""
    acc = ll.empty([32], ll.float32, scope="local")
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_zero_accum(acc, 32, ll.float32)
    ll.wgmma_fence()
    ll.wgmma_mma_async_ss(acc, da, db, 64, dtype='f16', scale_d=1, trans_a=0, trans_b=1)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc[0]
    out[i] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _tma_mbarrier_wgmma_pipe(tma_q: ll.const[ll.grid_constant[ll.TmaDescriptor]],
                             out: ll.Tensor[ll.float32]) -> ll.void:
    smem_q = ll.empty([64, 64], dtype=ll.float16, scope="dynamic_shared")
    smem_k = ll.empty([64, 64], dtype=ll.float16, scope="dynamic_shared")
    bar = ll.empty([1], dtype=ll.uint64, scope="dynamic_shared")
    acc = ll.empty([32], ll.float32, scope="local")

    tid: ll.int32 = ll.threadIdx_x()
    if tid == 0:
        ll.init_smem_barrier(bar, 1)
    ll.fence_smem_barrier_init()
    ll.__syncthreads()

    if tid == 0:
        ll.mbarrier_arrive_and_expect_tx(bar, 8192)
        ll.tma_load_2d_1sm(tma_q, bar, smem_q, 0, 0)
        ll.tma_load_2d_1sm(tma_q, bar, smem_k, 0, 64)
    ll.mbarrier_wait(bar, 0)

    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_zero_accum(acc, 32, ll.float32)
    ll.wgmma_fence()
    ll.wgmma_mma_async_ss(acc, da, db, 64, dtype='f16', scale_d=1, trans_a=0, trans_b=0)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    v: ll.float32 = acc[0]
    out[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _wgmma_m64n128k16_bf16_direct_empty(out: ll.Tensor[ll.float32]) -> ll.void:
    """Direct m64n128k16_bf16 builtin into a 64-element ``ll.empty`` local.

    Covers the m64n128k16 accumulator contract: a 64-element float32 local
    array addressable as a stable lvalue across the asm call, fed from two
    shared-memory descriptors. Mirrors the shape spellings on the
    m64nN256-side examples (``empty`` + ``wgmma_zero_accum``).
    """
    acc = ll.empty([64], ll.float32, scope="local")
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, 1)
    db: ll.uint64 = ll.encode_wgmma_smem_desc(8192, 16, 1024, 1)
    ll.wgmma_zero_accum(acc, 64, ll.float32)
    ll.wgmma_fence()
    ll.wgmma_m64n128k16_bf16(acc, da, db, 0, 1, 1)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc[0]
    out[i] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _wgmma_m64n128k16_bf16_direct_zeros(out: ll.Tensor[ll.float32]) -> ll.void:
    """Direct m64n128k16_bf16 into an accumulator produced by ``ll.zeros``.

    ``ll.zeros(..., scope="local")`` is rewritten by the mem-alloc pass
    into ``alloc_local_zeros(<name>, dtype, 64)``; this test pins that the
    rewritten form is also acceptable to the direct m64n128k16 builtins
    (i.e. their accumulator contract is not specific to ``ll.empty``).
    """
    acc = ll.zeros([64], ll.float32, scope="local")
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, 1)
    db: ll.uint64 = ll.encode_wgmma_smem_desc(8192, 16, 1024, 1)
    ll.wgmma_fence()
    ll.wgmma_m64n128k16_bf16(acc, da, db, 0, 1, 1)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc[0]
    out[i] = v


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda)


def _maybe_nvcc_sm90a(test_case: unittest.TestCase, source: str, label: str) -> str:
    """Compile with nvcc for sm_90a if available; return 'ran' or 'skipped'."""
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return "skipped"
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, f"{label}.cu")
        obj = os.path.join(td, f"{label}.o")
        with open(src, "w") as f:
            f.write(source)
        proc = subprocess.run([nvcc, "-c", "-gencode", "arch=compute_90a,code=sm_90a", "-std=c++17", src, "-o", obj],
                              capture_output=True, text=True, check=False)
        test_case.assertEqual(proc.returncode, 0, f"nvcc sm_90a rejected {label}:\n{proc.stderr}")
        test_case.assertTrue(os.path.exists(obj))
        return "ran"


def _make_f16_trans_variant_kernel(n_tile, trans_a, trans_b):
    """Materialise an SS f16 k-step repro kernel with constant immediates.

    The tracer resolves kernel source via ``inspect.getsource`` and does not
    capture Python closure variables, so each (n_tile, trans_a, trans_b)
    variant is emitted as source text with the constants inlined and written
    to a generated module on disk before import; the resulting kernel body is
    the same shape as the static kernels above. The fp16 tiles come from
    ``ll.alloc_shared_memory`` and the accumulator is a 32- or 64-element
    ``ll.empty`` local, matching the draft decomposition (``acc`` is fixed
    at Python level because its width is n_tile-dependent).

    For trans_a=1 (dV / dK FA-style operands) PTX requires A column-wise
    from shared memory; for trans_b=1 (QK^T scores) B column-wise. Both
    immediates are baked into the helper name and the trailing asm operands
    (``...,p,scale_a,scale_b,trans_a,trans_b;``).
    """
    label = f"n{n_tile}_ta{trans_a}_tb{trans_b}"
    src = f'''
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend='cuda', is_entry=True)
def kern(out: ll.Tensor[ll.float32]) -> ll.void:
    smem_a = ll.alloc_shared_memory("smem_a", ll.float16, 64 * 64)
    smem_b = ll.alloc_shared_memory("smem_b", ll.float16, 64 * 128)
    acc = ll.empty([{n_tile // 2}], ll.float32, scope="local")
    ll.wgmma_zero_accum(acc, {n_tile // 2}, ll.float32)
    ll.wgmma_fence()
    for ks in ll.unroll(range(4)):
        da: ll.uint64 = ll.encode_wgmma_smem_desc(
            0 + ks * 2048, 8192, 1024, "swizzle_128b")
        db: ll.uint64 = ll.encode_wgmma_smem_desc(
            32768 + ks * 2048, 8192, 1024, "swizzle_128b")
        ll.wgmma_mma_async_ss(acc, da, db, {n_tile}, dtype='f16',
                              scale_d=1, trans_a={trans_a},
                              trans_b={trans_b})
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    i: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc[0]
    out[i] = v
'''
    # Write the generated module to a unique per-process temp file
    # created with O_EXCL (never follows pre-planted symlinks or truncates
    # an existing path) and removed at interpreter exit; the returned
    # kernel must stay importable after this function returns, so a
    # TemporaryDirectory block is too narrow here.
    fd, path = tempfile.mkstemp(prefix=f"_wgmma_ss_f16_trans_variant_{label}_", suffix=".py")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location(f"_wgmma_ss_f16_trans_variant_{label}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.kern
    finally:
        atexit.register(_silently_remove, path)


def _silently_remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class TestWgmmaSsF16Codegen(unittest.TestCase):
    """Static f16 SS repro kernels lower to the whitelisted PTX forms."""

    def test_wgmma_ss_f16_lowering(self):
        src = _compile(_wgmma_ss_f16_repro)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16", src)
        self.assertIn("wgmma.fence.sync.aligned", src)
        self.assertIn("wgmma.commit_group.sync.aligned", src)
        self.assertIn("wgmma.wait_group.sync.aligned 0", src)
        # 32-float f32 accumulator with read-write constraints
        # (scale_d=1 reads D)
        self.assertEqual(src.count('"+f"'), 32)
        # descriptor helpers encode layout 1 (128B swizzle) in bits [63:62]
        self.assertIn("encode_wgmma_smem_desc_l1_fn", src)
        self.assertIn("(uint64_t)1 << 62", src)

    def test_wgmma_ss_f16_qkT_trans_b(self):
        src = _compile(_wgmma_ss_f16_qkT)
        # trans_b=1 must appear as the last immediate of the wgmma
        # instruction
        m = re.search(r"wgmma\.mma_async\.sync\.aligned\.m64n64k16\.f32\.f16\.f16\\n\"\n"
                      r".*?%\d+,%\d+,p,1,1,0,1;", src, re.DOTALL)
        self.assertIsNotNone(m, "expected trans_a=0, trans_b=1 immediates in wgmma asm")

    def test_tma_mbarrier_wgmma_pipeline(self):
        src = _compile(_tma_mbarrier_wgmma_pipe)
        for token in ("mbarrier.init.shared.b64", "mbarrier.arrive.expect_tx.shared::cta.b64",
                      "mbarrier.try_wait.parity.shared::cta.b64", "cp.async.bulk.tensor.2d.shared::cluster.global."
                      "mbarrier::complete_tx::bytes", "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16"):
            self.assertIn(token, src, f"missing {token}")
        # the TMA descriptor arrives by grid-constant kernel parameter
        self.assertIn("__grid_constant__ TmaDescriptor tma_q", src)


class TestEncodeWgmmaSmemDescLayoutLegality(unittest.TestCase):
    """The layout selector is validated before any asm text is composed."""

    def test_legal_selectors(self):
        # Python ints
        for code in range(4):
            self.assertEqual(_wgmma_desc_layout_code(code), code)
        # named selectors
        self.assertEqual(_wgmma_desc_layout_code("none"), 0)
        self.assertEqual(_wgmma_desc_layout_code("swizzle_128b"), 1)
        self.assertEqual(_wgmma_desc_layout_code("swizzle_64b"), 2)
        self.assertEqual(_wgmma_desc_layout_code("swizzle_32b"), 3)
        # decimal text (how codegen delivers positional int literals)
        for code in range(4):
            self.assertEqual(_wgmma_desc_layout_code(str(code)), code)

    def test_illegal_selectors_raise_value_error(self):
        # everything else is a hard compile error, never malformed asm
        for bad in ("bogus", "4", "-1", "1.5", 4, -1, True, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _wgmma_desc_layout_code(bad)


class TestWgmmaSsF16TransVariants(unittest.TestCase):
    """f16 SS wgmma m64n{64,128}k16 lowers for all (trans_a, trans_b) combos.

    Pins the teacher-decomposition descriptor forms: Q@K^T (0,0 / S=QK^T),
    dV=P^T dO and dK=dS^T Q (1,1), dQ=dS K (0,1); each against
    ``encode_wgmma_smem_desc(lbo=8192, sbo=1024, 'swizzle_128b')`` rebased
    per k-step by a runtime byte offset inside ``ll.unroll(range(4))``.
    """

    def test_trans_variants_lowering(self):
        for n_tile in (64, 128):
            mnm = f"m64n{n_tile}k16.f32.f16.f16"
            for trans_a in (0, 1):
                for trans_b in (0, 1):
                    with self.subTest(n_tile=n_tile, trans_a=trans_a, trans_b=trans_b):
                        kern = _make_f16_trans_variant_kernel(n_tile, trans_a, trans_b)
                        src = _compile(kern)
                        label = f"n{n_tile} ta{trans_a} tb{trans_b}"
                        # CPU lowering of the whitelisted
                        # m64nNk16.f32.f16.f16 form
                        self.assertIn(mnm, src, f"{label}: missing {mnm}")
                        # accumulator width n//2 read-write float constraints
                        self.assertEqual(src.count('"+f"'), n_tile // 2, f"{label}: expected {n_tile // 2} '+f' "
                                         f"constraints")
                        # immediate tail p,scale_a,scale_b,trans_a,trans_b
                        # in the asm
                        m = re.search(
                            rf"wgmma\.mma_async\.sync\.aligned\.{mnm}\\n\"\n"
                            rf".*?%\d+,%\d+,p,1,1,{trans_a},{trans_b};", src, re.DOTALL)
                        self.assertIsNotNone(m, f"{label}: expected immediates "
                                             f"1,1,{trans_a},{trans_b} in wgmma asm")
                        # the unrolled k-loop emits four rebased SS calls per
                        # operand and four mma invocations
                        self.assertEqual(src.count("encode_wgmma_smem_desc_l1_fn("), 9,
                                         f"{label}: expected 1 helper decl + 8 rebased "
                                         f"encode call sites")
                        fn = (f"wgmma_mma_ss_n{n_tile}_f16_sa1sb1d1"
                              f"ta{trans_a}tb{trans_b}_fn")
                        self.assertEqual(src.count(f"{fn}(acc, da, db);"), 4, f"{label}: expected four unrolled {fn} "
                                         f"invocations")
                        # runtime per-k-step byte-offset rebasing survives
                        # codegen
                        self.assertIn("encode_wgmma_smem_desc_l1_fn(0 + ks * 2048, "
                                      "8192, 1024)", src, f"{label}: A rebase missing")
                        self.assertIn("encode_wgmma_smem_desc_l1_fn(32768 + ks * 2048, "
                                      "8192, 1024)", src, f"{label}: B rebase missing")

    def test_trans_variants_nvcc_acceptance(self):
        nvcc_status = []
        for n_tile in (64, 128):
            for trans_a in (0, 1):
                for trans_b in (0, 1):
                    kern = _make_f16_trans_variant_kernel(n_tile, trans_a, trans_b)
                    nvcc_status.append(
                        _maybe_nvcc_sm90a(self, _compile(kern), f"wgmma_ss_f16_n{n_tile}_ta{trans_a}tb{trans_b}"))
        print(f"nvcc sm_90a f16 trans-variant acceptance: {nvcc_status}")


class TestWgmmaM64n128k16Bf16Direct(unittest.TestCase):
    """Direct m64n128k16_bf16 builtins accept ``empty``/``zeros`` locals."""

    def assert_m64n128k16_bf16_lowering(self, src: str) -> None:
        # m64n128k16 PTX, bf16 elf-ty operand pair, 64-element f32 accumulator
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", src)
        # The inline-asm helper declares the accumulator as float* and lists
        # exactly 64 '+f' read-write register constraints (one per lane).
        self.assertIn("float* c", src)
        self.assertEqual(src.count('"+f"'), 64)
        # The helper is invoked (return_val uses the helper, not raw asm
        # text).
        self.assertIn("wgmma_m64n128k16_bf16_a0b1d1_fn(", src)

    def test_direct_empty_lowering(self):
        src = _compile(_wgmma_m64n128k16_bf16_direct_empty)
        self.assert_m64n128k16_bf16_lowering(src)
        self.assertIn("wgmma.fence.sync.aligned", src)
        self.assertIn("wgmma.commit_group.sync.aligned", src)
        self.assertIn("wgmma.wait_group.sync.aligned 0", src)

    def test_direct_zeros_lowering(self):
        src = _compile(_wgmma_m64n128k16_bf16_direct_zeros)
        self.assert_m64n128k16_bf16_lowering(src)


class TestNvccSm90aAcceptance(unittest.TestCase):
    """CPU-side ptxas legality gate over the static kernels (nvcc on PATH)."""

    def test_nvcc_sm90a_acceptance(self):
        nvcc_status = []
        nvcc_status.append(_maybe_nvcc_sm90a(self, _compile(_wgmma_ss_f16_repro), "wgmma_ss_repro"))
        nvcc_status.append(_maybe_nvcc_sm90a(self, _compile(_tma_mbarrier_wgmma_pipe), "tma_mbarrier_wgmma_pipe"))
        nvcc_status.append(
            _maybe_nvcc_sm90a(self, _compile(_wgmma_m64n128k16_bf16_direct_empty), "wgmma_m64n128k16_bf16_empty"))
        nvcc_status.append(
            _maybe_nvcc_sm90a(self, _compile(_wgmma_m64n128k16_bf16_direct_zeros), "wgmma_m64n128k16_bf16_zeros"))
        print(f"nvcc sm_90a acceptance: {nvcc_status}")


if __name__ == "__main__":
    unittest.main()
