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
# Fragment NAMES (acc_s, acc_p, ...) are compile-time bindings created by the
# ll.wgmma_acc_fragment_f32("name", N) declaration builtin in each kernel;
# they intentionally have no Python-level definition (same mechanism as the
# typing-scope registration used by declaration intrinsics), so every bare
# use carries a per-line "noqa: F821" suppression (per-line form, as in
# tests/nv/common/unit/test_mbarrier_byte_address_idiom.py).
"""SM90 WGMMA register-fragment typed-IR CPU regression tests (no GPU).

Covers the sm90a capability ``ir.wgmma.acc_fragment_epilogue``:

* ``wgmma_acc_fragment_f32`` declares a WGMMA accumulator as N INDEPENDENT
  per-thread float register scalars (never an address-taken array);
* the existing fragment-aware SS family (wgmma_m64n64k16_{f16,bf16},
  wgmma_m64n128k16_{f16,bf16}, wgmma_m64n256k16_{f16,bf16}) binds each
  fragment scalar as a separate ``"+f"`` asm operand — the teacher's
  hand-CUDA operand shape — when its acc operand names a declared fragment;
* fragment elements are readable AND writable at compile-time-unrolled
  indices (``ll.unroll(range(N))`` counters, literals, provably-static int
  locals), lowering to the per-element scalar ``<name>_<i>`` so a
  multi-consumer elementwise epilogue (online-softmax stats) keeps the
  fragment in registers;
* ``wgmma_frag_pack_bf16x2`` packs two fragment scalars into one bf16x2
  u32 word in registers;
* ``smem_st_frag_swizzle_128b`` restages a fragment via stmatrix into the
  128B-swizzled bf16 smem tile layout a subsequent WGMMA consumes;
* the n64/n128 accumulator-coordinate helpers implement the documented
  generic fragment layout;
* legality: dynamic (non-constant) fragment indices, undeclared fragment
  operands, and size-mismatched fragment/instruction pairings are hard
  compile errors, never silent local-memory demotions.

When ``nvcc`` is on PATH, the emitted sources are additionally compiled
with ``-gencode arch=compute_90a,code=sm_90a`` as a pure CPU-side nvcc
acceptance gate (ptxas legality of the inline PTX); no device is used.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.sm90.language.intrin.wgmma import (
    wgmma_m64n64k16_accumulator_coordinate,
    wgmma_m64n128k16_accumulator_coordinate,
    wgmma_m64n256k16_accumulator_coordinate,
)

# ---------------------------------------------------------------------------
# Kernels under test
# ---------------------------------------------------------------------------


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_softmax_epilogue(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.float32]) -> ll.void:
    """QK^T fragment -> register online-softmax stats -> elementwise rewrite.

    Declares an m64n128k16 fragment (64 floats/thread), accumulates one SS
    WGMMA into it, then runs a three-pass elementwise epilogue (max, exp,
    store) whose subscripts must all lower to per-element register scalars.
    """
    ll.wgmma_acc_fragment_f32("acc_s", 64)
    tid: ll.int32 = ll.threadIdx_x()
    warp: ll.int32 = tid / 32
    lane: ll.int32 = tid % 32
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_fence()
    ll.wgmma_m64n128k16_bf16(acc_s, da, db, 0, 0, 1)  # noqa: F821 - bound by wgmma_acc_fragment_f32 declaration
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    # Elementwise multi-consumer epilogue on the SAME fragment object.
    m: ll.float32 = ll.neg_infinity()
    for i in ll.unroll(range(64)):
        m = ll.max_val(m, acc_s[i])  # noqa: F821 - fragment binding from declaration builtin
    for i in ll.unroll(range(64)):
        acc_s[i] = ll.expf(acc_s[i] - m)  # noqa: F821 - fragment binding
    for i in ll.unroll(range(64)):
        row: ll.int32 = 16 * warp + 8 * ((i // 2) % 2) + (lane // 4) % 8
        col: ll.int32 = (i // 4) * 8 + 2 * (lane % 4) + (i % 2)
        y[row * 128 + col] = acc_s[i]  # noqa: F821 - fragment binding
    v: ll.float32 = x[0]  # keep x live  # noqa: F841


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_two_mma_scale_gating(out: ll.Tensor[ll.float32]) -> ll.void:
    """Same fragment across two MMA calls with per-call scale_d gating."""
    ll.wgmma_acc_fragment_f32("acc_p", 64)
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_fence()
    ll.wgmma_m64n128k16_bf16(acc_p, da, db, 0, 0, 0)  # noqa: F821 - fragment binding; overwrite (compute 0)
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    ll.wgmma_fence()
    ll.wgmma_m64n128k16_bf16(acc_p, da, db, 0, 0, 1)  # noqa: F821 - fragment binding; accumulate
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc_p[0]  # noqa: F821 - fragment binding
    out[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_pack_and_restage(out: ll.Tensor[ll.uint32]) -> ll.void:
    """Register pack of fragment pairs + stmatrix restage into a swzz tile."""
    smem = ll.empty([16384], dtype=ll.uint8, scope="dynamic_shared")
    ll.wgmma_acc_fragment_f32("acc_p", 64)
    tid: ll.int32 = ll.threadIdx_x()
    warp: ll.int32 = tid / 32
    lane: ll.int32 = tid % 32
    r: ll.uint32 = 0
    for j in ll.unroll(range(32)):
        r = r | ll.wgmma_frag_pack_bf16x2(acc_p[2 * j],  # noqa: F821 - fragment binding
                                          acc_p[2 * j + 1])  # noqa: F821 - fragment binding
    ll.smem_st_frag_swizzle_128b(acc_p, smem, 0, warp, lane)  # noqa: F821 - fragment binding
    ll.__syncthreads()
    out[tid] = r


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_two_size_restage(out: ll.Tensor[ll.uint32]) -> ll.void:
    """Two differently-sized fragments restaged through stmatrix in one kernel.

    The FA-backward composition stages a 64-elem P fragment and a 128-elem
    dP fragment; both size-specialized helper definitions must be emitted.
    """
    smem = ll.empty([65536], dtype=ll.uint8, scope="dynamic_shared")
    ll.wgmma_acc_fragment_f32("acc_a", 64)
    ll.wgmma_acc_fragment_f32("acc_b", 128)
    tid: ll.int32 = ll.threadIdx_x()
    warp: ll.int32 = tid / 32
    lane: ll.int32 = tid % 32
    ll.smem_st_frag_swizzle_128b(acc_a, smem, 0, warp, lane)  # noqa: F821 - fragment binding
    ll.smem_st_frag_swizzle_128b(acc_b, smem, 16384, warp, lane)  # noqa: F821 - fragment binding
    ll.__syncthreads()
    out[tid] = 0


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_dynamic_index(out: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: fragment element access at a runtime (non-constant) index."""
    ll.wgmma_acc_fragment_f32("acc_d", 64)
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc_d[tid]  # noqa: F821 - fragment binding
    out[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_undeclared_mma(out: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: acc operand names no declared fragment (caught as array)."""
    acc = ll.empty([64], ll.float32, scope="local")
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_m64n64k16_bf16(acc, da, db, 0, 0, 1)  # 32-elem op on 64-elem arr
    v: ll.float32 = acc[0]
    tid: ll.int32 = ll.threadIdx_x()
    out[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_size_mismatch(out: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: 64-elem fragment fed to the 32-elem m64n64k16 instruction."""
    ll.wgmma_acc_fragment_f32("acc_m", 64)
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_m64n64k16_bf16(acc_m, da, db, 0, 0, 1)  # noqa: F821 - fragment binding
    tid: ll.int32 = ll.threadIdx_x()
    out[tid] = acc_m[0]  # noqa: F821 - fragment binding


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_n256_family(out: ll.Tensor[ll.float32]) -> ll.void:
    """Fragment path of the m64n256k16 family (128 scalars)."""
    ll.wgmma_acc_fragment_f32("acc_w", 128)
    da: ll.uint64 = ll.encode_wgmma_smem_desc(0, 16, 1024, "swizzle_128b")
    db: ll.uint64 = ll.encode_wgmma_smem_desc(16384, 16, 1024, "swizzle_128b")
    ll.wgmma_fence()
    ll.wgmma_m64n256k16_bf16(acc_w, da, db, 0, 0, 1)  # noqa: F821 - fragment binding
    ll.wgmma_commit()
    ll.wgmma_wait_group(0)
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc_w[127]  # noqa: F821 - fragment binding
    out[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_mixed_static_unroll_index(out: ll.Tensor[ll.float32]) -> ll.void:
    """Index expressions mixing a static-int base with an unroll counter."""
    ll.wgmma_acc_fragment_f32("acc_v", 32)
    KMAX: ll.int32 = 32
    base: ll.int32 = 16
    for i in ll.unroll(range(16)):
        acc_v[base + i] = out[i] * 2.0  # noqa: F821 - fragment binding
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc_v[KMAX - 1]  # noqa: F821 - fragment binding
    out[tid] = v


@lk.ll_kernel(backend='cuda', is_entry=True)
def _frag_nested_unroll_index(out: ll.Tensor[ll.float32]) -> ll.void:
    """Illegal: subscript depends on a NESTED (inner) ll.unroll counter.

    Only the outermost source-level full-unroll loop is expanded into
    per-iteration scoped copies whose counter is published to the codegen
    constant env; an inner ``ll.unroll`` body is emitted as a C ``#pragma
    unroll`` loop whose counter has no compile-time binding. A fragment
    subscript that mixes the outer (foldable) and inner (runtime) counters
    therefore does NOT fold and must be a hard compile error, never a silent
    local-memory demotion.
    """
    ll.wgmma_acc_fragment_f32("acc_n", 32)
    for i in ll.unroll(range(8)):
        for j in ll.unroll(range(4)):
            acc_n[i * 4 + j] = out[i * 4 + j]  # noqa: F821 - fragment binding
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = acc_n[31]  # noqa: F821 - fragment binding
    out[tid] = v


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda)


def _maybe_nvcc_sm90a(source: str, label: str) -> str:
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
        assert proc.returncode == 0, (f"nvcc sm_90a rejected {label}:\n{proc.stderr}")
        assert os.path.exists(obj)
        return "ran"


# ---------------------------------------------------------------------------
# Lowering tests
# ---------------------------------------------------------------------------


class TestFragmentDeclaration(unittest.TestCase):
    """The declaration emits independent scalars, never an array."""

    def test_declaration_emits_scalars_not_array(self):
        src = _compile(_frag_softmax_epilogue)
        for i in range(64):
            self.assertIn(f"float acc_s_{i} = 0.0f;", src)
        self.assertNotIn("float acc_s[", src)
        self.assertNotIn("float acc_s;", src)


class TestFragmentMmaOperandBinding(unittest.TestCase):
    """Fragment-aware MMA binds per-scalar "+f" operands."""

    def test_mma_binds_per_scalar_operands(self):
        src = _compile(_frag_softmax_epilogue)
        self.assertIn("wgmma_m64n128k16_bf16_a0b0d1_frag_fn", src)
        self.assertIn("_frag_fn(acc_s_0", src)
        self.assertGreaterEqual(src.count('"+f"(c'), 64)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16", src)
        # The legacy whole-array helper for these immediates must NOT
        # appear: taking acc's address is exactly the local-memory demotion
        # this capability eliminates.
        self.assertNotIn("wgmma_m64n128k16_bf16_a0b0d1_fn(", src.replace("wgmma_m64n128k16_bf16_a0b0d1_frag_fn(", ""))

    def test_element_subscripts_lower_to_scalars(self):
        src = _compile(_frag_softmax_epilogue)
        # exp rewrite loop: acc_s[i] lowers to acc_s_<i> on BOTH sides.
        self.assertIn("acc_s_5 = __nv_expf(acc_s_5 - m)", src)
        # store loop reads the scalar; no address-taken subscript remains
        self.assertNotIn("acc_s[", src)
        self.assertNotIn("&acc_s", src)

    def test_two_mma_scale_gating(self):
        src = _compile(_frag_two_mma_scale_gating)
        self.assertIn("wgmma_m64n128k16_bf16_a0b0d0_frag_fn", src)
        self.assertIn("wgmma_m64n128k16_bf16_a0b0d1_frag_fn", src)
        helper_d0 = src.split("wgmma_m64n128k16_bf16_a0b0d0_frag_fn", 1)[1]
        self.assertIn("setp.ne.b32 p, 0, 0", helper_d0)
        helper_d1 = src.split("wgmma_m64n128k16_bf16_a0b0d1_frag_fn", 1)[1]
        self.assertIn("setp.ne.b32 p, 1, 0", helper_d1)

    def test_pack_and_restage_lowering(self):
        src = _compile(_frag_pack_and_restage)
        # register-to-register pack of fragment scalars
        self.assertIn("cvt.rn.bf16x2.f32", src)
        # stmatrix staging helper with per-scalar float params (no float*);
        # the helper name is specialized by fragment size so two differently
        # sized fragments in one kernel never collide on the dedup-by-name
        # builtin cache (writeln_builtin).
        self.assertIn("smem_st_frag_swizzle_128b_n64_fn", src)
        self.assertIn("stmatrix.sync.aligned.x2.m8n8.shared.b16", src)
        helper_body = src.split("smem_st_frag_swizzle_128b_n64_fn", 1)[1]
        self.assertNotIn("float* acc", helper_body)
        # swizzle XOR is present with the 128B (8-chunk) modulus
        self.assertIn("^= row_ % 8;", src)

    def test_two_fragment_sizes_both_helpers_emitted(self):
        # Regression: writeln_builtin dedups builtin bodies by function NAME,
        # so staging a 64-elem and a 128-elem fragment from one kernel must
        # emit TWO size-specialized helper definitions (n64/n128), each
        # called with its own arity (68 and 132 arguments respectively).
        src = _compile(_frag_two_size_restage)
        self.assertIn("smem_st_frag_swizzle_128b_n64_fn", src)
        self.assertIn("smem_st_frag_swizzle_128b_n128_fn", src)
        # the stmatrix inner helper is size-suffixed too: a shared name
        # would be re-emitted with the second size (writeln_builtin only
        # skips a body when EVERY extracted function name is already
        # emitted) and redefine the function in the same TU
        self.assertEqual(src.count("void stsm_x2_frag_n64_("), 1)
        self.assertEqual(src.count("void stsm_x2_frag_n128_("), 1)
        self.assertNotIn("void stsm_x2_frag_(", src)
        # the n64 call site passes the 64 fragment scalars; the n128 call
        # site passes 128 — spot-check first/last scalar arguments
        n64_call = src.split("smem_st_frag_swizzle_128b_n64_fn(", 1)[1]
        self.assertIn("acc_a_0, ", n64_call)
        self.assertIn("acc_a_63)", n64_call)
        n128_call = src.split("smem_st_frag_swizzle_128b_n128_fn(", 1)[1]
        self.assertIn("acc_b_127)", n128_call)

    def test_n256_family(self):
        src = _compile(_frag_n256_family)
        self.assertIn("wgmma_m64n256k16_bf16_a0b0d1_frag_fn", src)
        for i in range(128):
            self.assertIn(f"float acc_w_{i} = 0.0f;", src)
        self.assertNotIn("float acc_w[", src)
        self.assertIn("wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16", src)
        self.assertIn("acc_w_127", src)


class TestAccumulatorCoordinateHelpers(unittest.TestCase):
    """The n64/n128 helpers implement the documented fragment layout."""

    def test_coordinate_formulas_and_bounds(self):
        # n64: 32-elem fragment; layout
        # row=16*(t>>5)+8*((i>>1)&1)+((t>>2)&7), col=(i>>2)*8+2*(t&3)+(i&1).
        t = 5  # warp 0, lane 5
        warp, lane = 0, 5
        for i in range(32):
            row, col = wgmma_m64n64k16_accumulator_coordinate(t, i)
            self.assertEqual(row, 16 * warp + 8 * ((i // 2) % 2) + (lane // 4) % 8)
            self.assertEqual(col, (i // 4) * 8 + 2 * (lane % 4) + (i % 2))
        # a non-warp-0 case
        t = 96 + 7  # warp 3, lane 7
        warp, lane = 3, 7
        row, col = wgmma_m64n64k16_accumulator_coordinate(t, 9)
        self.assertEqual(row, 16 * warp + 8 * ((9 // 2) % 2) + (lane // 4) % 8)
        self.assertEqual(col, (9 // 4) * 8 + 2 * (lane % 4) + (9 % 2))
        # n128: 64-elem fragment, same layout formula.
        for i in range(64):
            row, col = wgmma_m64n128k16_accumulator_coordinate(37, i)
            warp, lane = 1, 5
            self.assertEqual(row, 16 * warp + 8 * ((i // 2) % 2) + (lane // 4) % 8)
            self.assertEqual(col, (i // 4) * 8 + 2 * (lane % 4) + (i % 2))

    def test_n128_column_coverage(self):
        # n128 column coverage: 16 col-blocks of stride 8, each contributing
        # the lane-fixed pair {2*(lane%4), 2*(lane%4)+1}.
        lane = 37 % 32
        base_pair = {2 * (lane % 4), 2 * (lane % 4) + 1}
        cols = {wgmma_m64n128k16_accumulator_coordinate(37, i)[1] for i in range(64)}
        self.assertEqual(cols, {8 * b + p for b in range(16) for p in base_pair})
        self.assertLessEqual(max(cols), 127)

    def test_bounds_rejected(self):
        with self.assertRaises(ValueError):
            wgmma_m64n64k16_accumulator_coordinate(0, 32)
        with self.assertRaises(ValueError):
            wgmma_m64n128k16_accumulator_coordinate(0, 64)
        with self.assertRaises(ValueError):
            wgmma_m64n128k16_accumulator_coordinate(128, 0)

    def test_consistency_with_n256_helper(self):
        # consistency with the pre-existing n256 helper on its overlap
        for i in range(0, 64, 4):
            r_new, c_new = wgmma_m64n128k16_accumulator_coordinate(7, i)
            r_old, c_old = wgmma_m64n256k16_accumulator_coordinate(7, i)
            self.assertEqual((r_new, c_new), (r_old, c_old))


class TestMixedStaticUnrollIndices(unittest.TestCase):
    """Index expressions mixing a static-int base with an unroll counter."""

    def test_mixed_static_unroll_indices_fold(self):
        src = _compile(_frag_mixed_static_unroll_index)
        self.assertIn("acc_v_16", src)
        self.assertIn("acc_v_31", src)
        self.assertNotIn("acc_v[", src)


# ---------------------------------------------------------------------------
# Legality tests (compile errors, never silent demotions)
# ---------------------------------------------------------------------------


class TestFragmentLegality(unittest.TestCase):
    """Illegal fragment uses are hard compile errors."""

    def test_dynamic_index_rejected(self):
        with self.assertRaisesRegex(IndexError, "compile-time-unrolled"):
            _compile(_frag_dynamic_index)

    def test_nested_unroll_inner_counter_index_rejected(self):
        # Boundary case for the fix's constant-env mechanism: a fragment
        # subscript that mixes the OUTER (foldable) ll.unroll counter with an
        # INNER ll.unroll counter. Only the outermost source-unrolled loop is
        # expanded into scoped copies whose counter is published to the
        # codegen constant env; the inner loop is emitted as a C `#pragma
        # unroll` loop whose counter has no compile-time binding, so the
        # combined index i * 4 + j does not fold and must be a hard compile
        # error, never a silent local-memory demotion.
        with self.assertRaisesRegex(IndexError, "compile-time-unrolled"):
            _compile(_frag_nested_unroll_index)

    def test_size_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "exactly 32"):
            _compile(_frag_size_mismatch)

    def test_legacy_array_operand_still_array_path(self):
        # A plain ll.empty local keeps the legacy float* array helper: the
        # fragment path must be strictly opt-in via the declaration builtin.
        src = _compile(_frag_undeclared_mma)
        self.assertIn("wgmma_m64n64k16_bf16_a0b0d1_fn(", src)
        self.assertNotIn("_frag_fn", src)


# ---------------------------------------------------------------------------
# nvcc sm_90a acceptance gates (skipped when nvcc is unavailable)
# ---------------------------------------------------------------------------


@unittest.skipUnless(shutil.which("nvcc"), "nvcc not on PATH")
class TestNvccSm90aAcceptance(unittest.TestCase):
    """nvcc sm_90a acceptance for the emitted source (CPU-side only)."""

    def test_nvcc_accepts_frag_softmax_epilogue(self):
        src = _compile(_frag_softmax_epilogue)
        _maybe_nvcc_sm90a(src, "frag_softmax_epilogue")

    def test_nvcc_accepts_frag_two_mma(self):
        src = _compile(_frag_two_mma_scale_gating)
        _maybe_nvcc_sm90a(src, "frag_two_mma")

    def test_nvcc_accepts_frag_pack_restage(self):
        src = _compile(_frag_pack_and_restage)
        _maybe_nvcc_sm90a(src, "frag_pack_restage")

    def test_nvcc_accepts_frag_n256(self):
        src = _compile(_frag_n256_family)
        _maybe_nvcc_sm90a(src, "frag_n256")

    def test_nvcc_accepts_frag_two_size_restage(self):
        src = _compile(_frag_two_size_restage)
        _maybe_nvcc_sm90a(src, "frag_two_size_restage")


if __name__ == "__main__":
    unittest.main()
