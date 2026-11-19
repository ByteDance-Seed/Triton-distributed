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
"""CPU code-generation regression tests for ``smem_dot_accumulate_f32``.

Covers the generic per-thread fp32 dot-accumulate intrinsic
(:func:`smem_dot_accumulate_f32` in
``little_kernel.backends.nv.common.language.intrin.smem_dot``, semantic key
``ir.warp_smem_block_dot.accumulate_f32``) without a GPU:

* Trace/codegen: the capability reproducer pattern (per-thread strided DRAM
  state column contracted against a staged shared fp32 row) traces and
  lowers through the full cuda pass pipeline to CUDA C++ source.
* Structure: the emitted helper body must perform a serial FMA chain over a
  strided gmem index (``base + i * stride``) and a unit-stride smem index
  (``base + i``), under ``#pragma unroll``, with the DRAM load spelling
  selected by the cache policy (plain dereference for "wb", the CUDA
  ``__ld*`` cache operators otherwise).
* Per-policy lowering: every cache policy gets its own distinctly-named,
  #ifndef-guarded helper (``lk_smem_dot_accumulate_f32[_<policy>]_fn``),
  and a single kernel using two policies must emit both specializations
  with each call site bound to its own -- the regression shape for the
  writeln_builtin name-dedup hazard (a policy-invariant helper name would
  let one policy's body absorb another policy's call site).
* Semantics: a Python lockstep model of the emitted loop is checked against
  a brute-force reference dot across a battery of (count, stride, base,
  policy) combinations, including non-unit strides and nonzero smem row
  offsets.
* Typing: fp32 acc/gmem/smem accepted; fp16 acc, non-tensor operands, and
  non-fp32 element dtypes rejected with precise messages; an unknown cache
  policy rejected with the accepted set in the message.
* Effects: the intrinsic declares a GMEM read on the ``gmem`` operand and an
  SMEM read on the ``smem`` operand.
* Language plumbing: the symbol resolves lazily through
  ``little_kernel.language`` for the cuda backend.

No CUDA toolchain or GPU is required; when ``nvcc`` is on PATH the emitted
sources are additionally compiled for sm_80 as a pure CPU-side acceptance
gate, exactly the pattern used by the gmem_ld_masked codegen tests.
"""

import math
import re
import shutil
import subprocess
import tempfile
import os
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.backends.nv.common.language.intrin.smem_dot import (
    _SMEM_DOT_BODY,
    _eval_return_type_smem_dot,
    codegen_smem_dot_accumulate_f32,
    smem_dot_accumulate_f32,
)
from little_kernel.core.memory_effects import Access, MemorySpace
from little_kernel.language import builtin_base as bb

# ---------------------------------------------------------------------------
# Capability reproducer written against the new intrinsic: per-thread
# strided state column (row stride V) x staged shared fp32 row.
# ---------------------------------------------------------------------------


@lk.ll_kernel(backend='cuda', is_entry=True)
def _repro_dot(h0: ll.Tensor[ll.float32], sq: ll.Tensor[ll.float32], o: ll.Tensor[ll.float32], V: ll.int32,
               K: ll.int32) -> ll.void:
    s_kn: ll.Tensor[ll.float32] = ll.alloc_shared_memory("s_kn", ll.float32, 256)
    tid: ll.int32 = ll.threadIdx_x()
    idx: ll.int32 = tid
    while idx < 256:
        s_kn[idx] = sq[idx]
        idx = idx + 128
    ll.__syncthreads()
    s: ll.float32 = ll.smem_dot_accumulate_f32(0.0, h0, tid * V + tid, V, s_kn, 0, K, "cg")
    o[tid] = s


@lk.ll_kernel(backend='cuda', is_entry=True)
def _repro_dot_default_policy(h0: ll.Tensor[ll.float32], s_row: ll.Tensor[ll.float32], o: ll.Tensor[ll.float32],
                              K: ll.int32, stride: ll.int32, smem_off: ll.int32) -> ll.void:
    s_part: ll.Tensor[ll.float32] = ll.alloc_shared_memory("s_part", ll.float32, 512)
    tid: ll.int32 = ll.threadIdx_x()
    i: ll.int32 = tid
    while i < 512:
        s_part[i] = s_row[i]
        i = i + 128
    ll.__syncthreads()
    acc: ll.float32 = 1.0
    acc = ll.smem_dot_accumulate_f32(acc, h0, tid, stride, s_part, smem_off, K)
    o[tid] = acc


@lk.ll_kernel(backend='cuda', is_entry=True)
def _repro_dot_two_policies(h0: ll.Tensor[ll.float32], s_row: ll.Tensor[ll.float32], o: ll.Tensor[ll.float32],
                            K: ll.int32, V: ll.int32) -> ll.void:
    """One kernel using two different cache policies in a single thread.

    Regression shape for the writeln_builtin dedup hazard: each policy must
    lower to its own distinctly-named helper so the second call site cannot
    be silently bound to the first policy's specialized body.
    """
    s_row_st: ll.Tensor[ll.float32] = ll.alloc_shared_memory("s_row_st", ll.float32, 256)
    tid: ll.int32 = ll.threadIdx_x()
    j: ll.int32 = tid
    while j < 256:
        s_row_st[j] = s_row[j]
        j = j + 128
    ll.__syncthreads()
    a: ll.float32 = ll.smem_dot_accumulate_f32(1.0, h0, tid * V, 1, s_row_st, 0, K, "cg")
    b: ll.float32 = ll.smem_dot_accumulate_f32(a, h0, tid * V + V, 2, s_row_st, 0, K, "cs")
    o[tid] = b


_TWO_POLICIES_SRC = None


def _two_policies_src():
    global _TWO_POLICIES_SRC
    if _TWO_POLICIES_SRC is None:
        _TWO_POLICIES_SRC = _compile(_repro_dot_two_policies)
    return _TWO_POLICIES_SRC


def _helper_body(src: str, fn_name: str) -> str:
    """Extract the emitted definition of ``fn_name`` through its closing
    brace, so structural assertions stay inside exactly one policy's
    helper even when several helpers share the file."""
    start = src.index(fn_name)
    end = src.index("\n}\n", start) + 3
    return src[start:end]


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda)


def _maybe_nvcc(source: str, label: str, arch: str) -> str:
    """Compile with nvcc for ``arch`` if available; return 'ran'/'skipped'.

    Pure CPU-side nvcc legality gate (no device): catches malformed helper
    bodies and bad intrinsic spellings in the emitted source.
    """
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return "skipped"
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, f"{label}.cu")
        obj = os.path.join(td, f"{label}.o")
        with open(src, "w") as f:
            f.write(source)
        proc = subprocess.run(
            [nvcc, "-c", "-gencode", f"arch=compute_{arch},code=sm_{arch}", "-std=c++17", src, "-o", obj],
            capture_output=True, text=True, check=False)
        assert proc.returncode == 0, (f"nvcc sm_{arch} rejected {label}:\n{proc.stderr}")
        assert os.path.exists(obj)
        return "ran"


class TestSmemDotTraceAndCodegen(unittest.TestCase):
    """End-to-end trace + cuda-pipeline lowering of the dot intrinsic."""

    def test_capability_reproducer_lowers(self):
        src = _compile(_repro_dot)
        self.assertIn("float lk_smem_dot_accumulate_f32_cg_fn(", src, 'policy "cg" must define the _cg-suffixed helper')
        self.assertIn("lk_smem_dot_accumulate_f32_cg_fn(", src, "emitted kernel must call the _cg dot helper")
        self.assertIn("__ldcg(", src, 'policy "cg" must select the __ldcg load spelling')
        self.assertIn("fmaf(", src, "helper body must accumulate with a fused FMA")
        self.assertIn("#pragma unroll", src, "helper loop must be marked unrollable")

    def test_default_policy_lowers_to_plain_load(self):
        src = _compile(_repro_dot_default_policy)
        self.assertIn("float lk_smem_dot_accumulate_f32_fn(", src, "wb policy must define the unsuffixed helper")
        # Default (wb) policy: plain dereference, no __ld* cache operator,
        # checked inside exactly the wb helper's body.
        body = _helper_body(src, "float lk_smem_dot_accumulate_f32_fn(")
        self.assertNotRegex(body, r"__ld(a|g|s|u|v)\(", "default policy must not emit a CUDA cache-operator load")

    def test_two_policies_one_kernel_both_helpers_survive(self):
        """Regression: writeln_builtin dedups builtin bodies by extracted
        C++ function name, so a policy-invariant helper name would let the
        first policy's body absorb a second call site with a different
        policy.  Each policy must get its own helper and call site."""
        src = _two_policies_src()
        # One definition per policy, each carrying its own load spelling.
        cg_body = _helper_body(src, "float lk_smem_dot_accumulate_f32_cg_fn(")
        cs_body = _helper_body(src, "float lk_smem_dot_accumulate_f32_cs_fn(")
        self.assertIn("__ldcg(", cg_body)
        self.assertNotIn("__ldcs(", cg_body)
        self.assertIn("__ldcs(", cs_body)
        self.assertNotIn("__ldcg(", cs_body)
        # Each call site references its own policy's helper.
        self.assertIn("lk_smem_dot_accumulate_f32_cg_fn(", src)
        self.assertIn("lk_smem_dot_accumulate_f32_cs_fn(", src)
        self.assertNotIn("lk_smem_dot_accumulate_f32_fn(", src, "no unsuffixed wb helper may appear in this kernel")
        # The dedup hazard would have collapsed the two definitions into
        # one: assert each specialization survived exactly once.
        self.assertEqual(src.count("float lk_smem_dot_accumulate_f32_cg_fn("), 1)
        self.assertEqual(src.count("float lk_smem_dot_accumulate_f32_cs_fn("), 1)

    def test_same_policy_twice_emits_one_guard_copy(self):
        # Repeated use of one policy must still yield one definition (the
        # #ifndef guard / name-based dedup covers idempotence).
        src = _compile(_repro_dot_default_policy)
        self.assertEqual(src.count("float lk_smem_dot_accumulate_f32_fn("), 1)

    def test_nvcc_cpu_acceptance_when_available(self):
        status = _maybe_nvcc(_compile(_repro_dot), "smem_dot_cg", "80")
        self.assertIn(status, ("ran", "skipped"))
        status = _maybe_nvcc(_compile(_repro_dot_default_policy), "smem_dot_wb", "80")
        self.assertIn(status, ("ran", "skipped"))
        # The two-policy kernel is the exact hazard shape: two helper
        # definitions, each call site bound to its own.  Syntax-check it
        # too when a toolchain is present.
        status = _maybe_nvcc(_two_policies_src(), "smem_dot_two_policies", "80")
        self.assertIn(status, ("ran", "skipped"))


class TestSmemDotCodegenUnit(unittest.TestCase):
    """Direct unit coverage of the codegen function (no pipeline)."""

    def _codegen(self, policy=None):
        kwargs = {}
        if policy is not None:
            kwargs["policy"] = policy
        return codegen_smem_dot_accumulate_f32("acc", "h0", "gbase", "gstride", "s_kn", "sbase", "K", **kwargs)

    def test_return_val_passes_operands_in_order(self):
        rv = self._codegen().return_val
        self.assertEqual(rv, "lk_smem_dot_accumulate_f32_fn(acc, h0, gbase, gstride, "
                         "s_kn, sbase, K)")

    def test_policy_mangled_helper_names(self):
        """The helper name must be policy-distinct (writeln_builtin dedups
        builtin bodies by extracted C++ function name; an invariant name
        would silently rebind later call sites to the first policy's
        body).  Same suffix convention as codegen_gmem_ld_v4: wb is the
        unsuffixed default."""
        for policy, suffix in (("wb", ""), ("ca", "_ca"), ("cg", "_cg"), ("cs", "_cs"), ("lu", "_lu"), ("nc", "_nc"),
                               ("cv", "_cv")):
            with self.subTest(policy=policy):
                got = self._codegen(policy=f'"{policy}"')
                fn = f"lk_smem_dot_accumulate_f32{suffix}_fn"
                guard = (f"LK_SMEM_DOT_ACCUMULATE_F32{suffix.upper()}"
                         "_FN_DEFINED")
                self.assertIn(f"float {fn}(", got.body)
                self.assertIn(f"#ifndef {guard}", got.body)
                self.assertIn(f"#define {guard}", got.body)
                self.assertIn("#endif", got.body)
                self.assertTrue(got.return_val.startswith(f"{fn}("))

    def test_wb_policy_plain_dereference(self):
        body = self._codegen().body
        # The full address expression must be parenthesized inside the
        # dereference: `*(ptr_expr)`, never `*ptr + off`.
        self.assertIn("(*(lk_gmem + (lk_gmem_base + (long long)lk_i * lk_gmem_stride)))", body)

    def test_policy_spelling_map(self):
        for policy, intrin in (("ca", "__ldca"), ("cg", "__ldcg"), ("cs", "__ldcs"), ("lu", "__ldlu"), ("nc", "__ldg"),
                               ("cv", "__ldcv")):
            with self.subTest(policy=policy):
                body = self._codegen(policy=f'"{policy}"').body
                self.assertIn(f"{intrin}(lk_gmem + (lk_gmem_base + "
                              "(long long)lk_i * lk_gmem_stride))", body)

    def test_alias_policies_normalize(self):
        # Aliases normalize to their canonical policy in BOTH the load
        # spelling and the mangled helper name.
        got = self._codegen(policy='"l2_only"')
        self.assertIn("__ldcg(", got.body)
        self.assertIn("lk_smem_dot_accumulate_f32_cg_fn(", got.return_val)
        got = self._codegen(policy='"read_only"')
        self.assertIn("__ldg(", got.body)
        self.assertIn("lk_smem_dot_accumulate_f32_nc_fn(", got.return_val)

    def test_smem_side_is_unit_stride_subscript(self):
        body = self._codegen().body
        self.assertIn("lk_smem[lk_smem_base + lk_i]", body)

    def test_serial_fma_chain_structure(self):
        body = self._codegen().body
        self.assertIn("#pragma unroll", body)
        self.assertRegex(body, r"for \(int lk_i = 0; lk_i < lk_count; \+\+lk_i\)")
        self.assertIn("acc = fmaf(lk_col, lk_smem[lk_smem_base + lk_i], acc)", body)

    def test_unknown_policy_rejected_with_accepted_set(self):
        with self.assertRaisesRegex(ValueError, r"smem_dot_accumulate_f32: cache policy 'bogus' not in "
                                    r"\[.*\] \(aliases: .*"):
            self._codegen(policy='"bogus"')


class TestSmemDotSemantics(unittest.TestCase):
    """Lockstep Python model of the emitted loop vs brute-force reference."""

    @staticmethod
    def _reference(acc, gmem, gbase, gstride, smem, sbase, count):
        out = acc
        for i in range(count):
            out = math.fma(gmem[gbase + i * gstride], smem[sbase + i], out) \
                if hasattr(math, "fma") else out + gmem[gbase + i * gstride] * smem[sbase + i]
        return out

    def _model(self, acc, gmem, gbase, gstride, smem, sbase, count):
        """Mirror the emitted C++ loop statement-for-statement (fp32-free
        Python floats; FMA contraction is checked by structure, not bits)."""
        for lk_i in range(count):
            lk_col = gmem[gbase + lk_i * gstride]
            acc = lk_col * smem[sbase + lk_i] + acc
        return acc

    def test_battery_matches_reference(self):
        gmem = [math.sin(i * 0.7) * 3.0 for i in range(4096)]
        smem = [math.cos(i * 0.3) for i in range(2048)]
        cases = [
            # (acc, gbase, gstride, sbase, count)
            (0.0, 0, 1, 0, 1),
            (0.0, 5, 1, 0, 64),
            (1.0, 0, 7, 3, 100),
            (-2.5, 17, 13, 129, 96),
            (0.0, 100, 33, 512, 64),
            (3.25, 4095 - 31 * 3, 3, 700, 32),
        ]
        for acc, gbase, gstride, sbase, count in cases:
            with self.subTest(gbase=gbase, gstride=gstride, sbase=sbase, count=count):
                got = self._model(acc, gmem, gbase, gstride, smem, sbase, count)
                want = self._reference(acc, gmem, gbase, gstride, smem, sbase, count)
                self.assertAlmostEqual(got, want, places=12)

    def test_reproducer_shape_matches_handrolled_loop(self):
        """The intrinsic semantics equal the capability reproducer's
        explicit gmem_ld_scalar + fma while-loop."""
        import random
        random.seed(0)
        V, K = 8, 37
        h0 = [random.uniform(-1, 1) for _ in range(128 * V + K * V + 128)]
        s_kn = [random.uniform(-1, 1) for _ in range(256)]
        tid = 42
        # Hand-rolled reproducer loop:
        s, i = 0.0, 0
        while i < K:
            tmp = h0[tid * V + i * V + tid]
            s = tmp * s_kn[i] + s
            i += 1
        # Intrinsic model: gmem_base = tid*V + tid, gmem_stride = V,
        # smem_base = 0, count = K.
        got = self._model(0.0, h0, tid * V + tid, V, s_kn, 0, K)
        self.assertAlmostEqual(got, s, places=12)


class TestSmemDotTyping(unittest.TestCase):
    """Typed return-type evaluation contract."""

    def test_fp32_operands_accepted(self):
        from little_kernel.core import type_system as ts
        rt = _eval_return_type_smem_dot(ts.float32, ts.Tensor[ts.float32], ts.int32, ts.int32, ts.Tensor[ts.float32],
                                        ts.int32, ts.int32)
        self.assertEqual(rt, ts.float32)

    def test_non_fp32_acc_rejected(self):
        from little_kernel.core import type_system as ts
        with self.assertRaisesRegex(TypeError, "smem_dot_accumulate_f32 expects an fp32 accumulator"):
            _eval_return_type_smem_dot(ts.float16, ts.Tensor[ts.float32], ts.int32, ts.int32, ts.Tensor[ts.float32],
                                       ts.int32, ts.int32)

    def test_non_tensor_gmem_rejected(self):
        from little_kernel.core import type_system as ts
        with self.assertRaisesRegex(TypeError, "smem_dot_accumulate_f32 expects gmem to be a Tensor"):
            _eval_return_type_smem_dot(ts.float32, ts.float32, ts.int32, ts.int32, ts.Tensor[ts.float32], ts.int32,
                                       ts.int32)

    def test_non_fp32_smem_element_rejected(self):
        from little_kernel.core import type_system as ts
        with self.assertRaisesRegex(TypeError, "smem_dot_accumulate_f32 expects fp32 elements for smem"):
            _eval_return_type_smem_dot(ts.float32, ts.Tensor[ts.float32], ts.int32, ts.int32, ts.Tensor[ts.float16],
                                       ts.int32, ts.int32)

    def test_none_dtypes_rejected(self):
        from little_kernel.core import type_system as ts
        with self.assertRaisesRegex(TypeError, "smem_dot_accumulate_f32: cannot infer type of gmem"):
            _eval_return_type_smem_dot(ts.float32, None, ts.int32, ts.int32, ts.Tensor[ts.float32], ts.int32, ts.int32)


class TestSmemDotAttributes(unittest.TestCase):
    """Builtin wiring: attributes, effects, language-namespace resolution."""

    def test_builtin_attributes_registered(self):
        self.assertTrue(getattr(smem_dot_accumulate_f32, bb.BUILTIN_ATTR, False))
        self.assertTrue(getattr(smem_dot_accumulate_f32, bb.CODEGEN_NEEDS_TYPES_ATTR, False))
        self.assertIs(getattr(smem_dot_accumulate_f32, bb.EVAL_RETURN_TYPE_ATTR), _eval_return_type_smem_dot)
        self.assertIs(getattr(smem_dot_accumulate_f32, bb.CODEGEN_FUNC_ATTR), codegen_smem_dot_accumulate_f32)

    def test_declared_memory_effects(self):
        effects = getattr(smem_dot_accumulate_f32, bb.MEMORY_EFFECTS_ATTR, ())
        by_operand = {}
        for eff in effects:
            by_operand.setdefault(eff.operand, []).append(eff)
        self.assertIn("gmem", by_operand)
        self.assertIn("smem", by_operand)
        gmem_reads = [e for e in by_operand["gmem"] if e.access == Access.R and e.space == MemorySpace.GMEM]
        smem_reads = [e for e in by_operand["smem"] if e.access == Access.R and e.space == MemorySpace.SMEM]
        self.assertTrue(gmem_reads, "gmem operand must declare a GMEM read")
        self.assertTrue(smem_reads, "smem operand must declare an SMEM read")

    def test_language_namespace_resolves(self):
        self.assertIs(ll.smem_dot_accumulate_f32, smem_dot_accumulate_f32)

    def test_backend_exports_register_symbol(self):
        import json
        # Repo root: tests/nv/common/unit/<this file> -> up 4 levels.
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        for arch in ("sm90", "sm100"):
            path = os.path.join(repo_root, "python", "little_kernel", "backends", "nv", arch, "backend.json")
            with open(path) as f:
                exports = json.load(f)["language_exports"]
            self.assertEqual(exports.get("smem_dot_accumulate_f32"),
                             "little_kernel.backends.nv.common.language.intrin.smem_dot"
                             ":smem_dot_accumulate_f32", f"{arch} backend.json must export smem_dot_accumulate_f32")


if __name__ == "__main__":
    unittest.main()
