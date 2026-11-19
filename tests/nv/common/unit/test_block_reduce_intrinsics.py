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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""First-class block-level reduction intrinsics: naming, typing, effects,
emission and CPU-simulated semantics.

Legality contract under test (compiler side; launch geometry and uniform
control flow are frontend obligations documented in
language/intrin/block_reduce.py):

* canonical names ``ll.block_reduce_sum`` / ``ll.block_reduce_max`` /
  ``ll.block_reduce_min`` are exported and marked builtin;
* the returned type equals the reduced value's type for every dtype CUDA's
  shuffle family lowers natively (f32, f64, s32, u32, s64, u64, f16, bf16);
  anything else (e.g. int16) raises TypeError at type-inference time;
* ``smem_scratch`` must be a pointer whose element type equals the value
  dtype; non-pointer or mismatched-element arguments raise TypeError;
* each op declares the warp-collective register-read effect on ``value``
  plus shared-memory read/write effects on ``smem_scratch`` (region extent
  BLOCK_REDUCE_SCRATCH_ELEMS), so effect analyses never see the cross-warp
  dataflow as side-effect free;
* emission lowers to one shared templated helper per operator that embeds
  the SAME ``lk_warp_reduce_*`` butterfly helper the warp intrinsics use
  (single source of truth for the combine), contains both cross-warp
  barriers, and reuses slot 0 only after the second barrier.

The two-stage allreduce semantics are validated by a Python harness that
mirrors the emitted helper's schedule: per-warp butterfly, lane-0 staging,
first-warp combine with identity padding, broadcast -- for block sizes
from 1 to 32 warps, including a fused-residual-RMSNorm-style fp32 sum of
squares.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.memory_effects import Access, IssuerSet, MemorySpace, R_RMEM_WARP_SYNC
from little_kernel.core.type_system import Pointer
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.block_reduce as block_reduce_intrin


class TestBlockReduceIntrinsics(unittest.TestCase):

    def test_exported_canonical_names(self):
        for name in ("block_reduce_sum", "block_reduce_max", "block_reduce_min"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name), name)
                self.assertTrue(getattr(getattr(ll, name), BUILTIN_ATTR, False))
        self.assertEqual(block_reduce_intrin.BLOCK_REDUCE_SCRATCH_ELEMS, 33)

    def test_return_type_equals_value_type(self):
        for name in ("block_reduce_sum", "block_reduce_max", "block_reduce_min"):
            ret = getattr(getattr(block_reduce_intrin, name), EVAL_RETURN_TYPE_ATTR)
            for dtype in (ll.float32, ll.float64, ll.int32, ll.uint32, ll.int64, ll.uint64, ll.float16, ll.bfloat16):
                with self.subTest(op=name, dtype=dtype):
                    self.assertEqual(ret(dtype, Pointer(dtype)), dtype)

    def test_return_type_rejects_unsupported_dtype(self):
        ret = getattr(block_reduce_intrin.block_reduce_sum, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            ret(ll.int16, Pointer(ll.int16))
        with self.assertRaises(TypeError):
            ret(ll.int8, Pointer(ll.int8))

    def test_return_type_rejects_non_pointer_scratch(self):
        ret = getattr(block_reduce_intrin.block_reduce_sum, EVAL_RETURN_TYPE_ATTR)
        # A bare scalar local / element value is not a scratch pointer.
        with self.assertRaises(TypeError):
            ret(ll.float32, ll.float32)

    def test_return_type_rejects_mismatched_scratch_element(self):
        for name in ("block_reduce_sum", "block_reduce_max", "block_reduce_min"):
            ret = getattr(getattr(block_reduce_intrin, name), EVAL_RETURN_TYPE_ATTR)
            with self.subTest(op=name):
                with self.assertRaises(TypeError):
                    ret(ll.float32, Pointer(ll.int32))

    def test_declared_effects(self):
        for fn in (block_reduce_intrin.block_reduce_sum, block_reduce_intrin.block_reduce_max,
                   block_reduce_intrin.block_reduce_min):
            with self.subTest(name=fn.__name__):
                effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
                self.assertEqual(len(effects), 3)
                warp_eff = effects[0]
                self.assertIs(warp_eff.kind, R_RMEM_WARP_SYNC)
                self.assertIs(warp_eff.kind.access, Access.R)
                self.assertIs(warp_eff.kind.space, MemorySpace.RMEM)
                self.assertIs(warp_eff.kind.issuer, IssuerSet.WARP)
                self.assertEqual(warp_eff.operand, "value")
                smem_by_access = {e.access: e for e in effects[1:]}
                self.assertEqual(set(smem_by_access), {Access.W, Access.R})
                for eff in smem_by_access.values():
                    self.assertIs(eff.space, MemorySpace.SMEM)
                    self.assertEqual(eff.operand, "smem_scratch")
                    self.assertIsNotNone(eff.region)
                    self.assertEqual(eff.region.extent, block_reduce_intrin.BLOCK_REDUCE_SCRATCH_ELEMS)

    def test_codegen_spelling(self):
        cases = (
            (block_reduce_intrin.block_reduce_sum, "lk_block_reduce_sum", "lk_warp_reduce_sum(0xffffffffu, v)", "T(0)"),
            (block_reduce_intrin.block_reduce_max, "lk_block_reduce_max", "lk_warp_reduce_max(0xffffffffu, v)",
             "scratch[1]"),
            (block_reduce_intrin.block_reduce_min, "lk_block_reduce_min", "lk_warp_reduce_min(0xffffffffu, v)",
             "scratch[1]"),
        )
        for fn, helper, warp_call, fill in cases:
            with self.subTest(name=fn.__name__):
                out = getattr(fn, CODEGEN_FUNC_ATTR)("v", "scratch")
                self.assertIsInstance(out, Builtin)
                self.assertEqual(out.return_val, f"{helper}(v, scratch)")
                # One shared cross-warp helper embedding the warp butterfly
                # helper verbatim (combine defined in exactly one place).
                self.assertIn(f"T {helper}(T v, T* scratch)", out.body)
                self.assertIn(warp_call, out.body)
                # Two-stage choreography: stage partials, barrier, first
                # warp recombines with identity padding, broadcast, final
                # barrier protecting scratch reuse.
                self.assertIn("scratch[1 + warp] = v;", out.body)
                self.assertIn(f": {fill};", out.body)
                self.assertIn("scratch[0] = x;", out.body)
                self.assertIn("T r = scratch[0];", out.body)
                self.assertEqual(out.body.count("__syncthreads();"), 3)
                # num_warps derives from the runtime launch config.
                self.assertIn("(blockDim.x + 31) >> 5", out.body)
                self.assertNotIn("asm", out.body)

    def test_codegen_half_includes_only_when_needed(self):
        codegen = getattr(block_reduce_intrin.block_reduce_sum, CODEGEN_FUNC_ATTR)
        fp32 = codegen("v", "scratch", _arg_dtypes=[ll.float32, Pointer(ll.float32)])
        self.assertEqual(fp32.includes, [])
        f16 = codegen("v", "scratch", _arg_dtypes=[ll.float16, Pointer(ll.float16)])
        self.assertEqual(f16.includes, ["<cuda_fp16.h>"])
        bf16 = codegen("v", "scratch", _arg_dtypes=[ll.bfloat16, Pointer(ll.bfloat16)])
        self.assertEqual(bf16.includes, ["<cuda_bf16.h>"])


# Canonical use: RMSNorm-style block statistics.  One CTA owns one output
# row; each lane accumulates its hidden-state slice in fp32, the block
# allreduce produces the row's sum (of squares), and every thread rescales
# its own slice by the shared statistic -- the pattern all twelve dependent
# kernels hand-roll today.
RMSNORM_STATS_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def rms_stats(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    acc: ll.float32 = x[tid] * x[tid] + x[256 + tid] * x[256 + tid]
    ll.alloc_shared_memory("s_red", ll.float32, 33)
    total: ll.float32 = ll.block_reduce_sum(acc, s_red + 0)
    out[tid] = total
'''

# Ordered combines across the whole block: softmax numerator scale (max)
# and clamp/best-so-far bound (min) in one translation unit, also the
# dedup tripwire for the embedded warp helpers and predicate once-guard.
BLOCK_EXTREMA_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def block_extrema(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = x[tid]
    ll.alloc_shared_memory("s_red", ll.float32, 33)
    out[tid] = ll.block_reduce_max(v, s_red + 0)
    out[256 + tid] = ll.block_reduce_min(v, s_red + 0)
'''


class TestBlockReduceEndToEnd(unittest.TestCase):
    """Full lower() path: native helper spelling in generated CUDA."""

    def _lower_one(self, tmp: Path, source: str, name: str, tensor_names, dtype: str = "float32") -> str:
        task = tmp / "task"
        task.mkdir()
        (task / "model.py").write_text("class Model: pass\n")
        (task / "test_cases.py").write_text("test_cases = []\n")
        program = tmp / "program"
        program.mkdir()
        (program / "draft.py").write_text(source)
        (program / "export.json").write_text(
            json.dumps({
                "kernel": name,
                "function": name,
                "inputs": [{"name": t, "dtype": dtype, "rank": 1} for t in tensor_names],
                "output": {"name": "out", "dtype": dtype, "shape": [f"{tensor_names[0]}[0]"]},
                "scalars": [],
                "launch": {"grid": [f"ceil_div({tensor_names[0]}[0], 128)", 1, 1], "block": [128, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_rmsnorm_stats_lowers_to_shared_helper(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), RMSNORM_STATS_SOURCE, "rms_stats", ("x", ))
        self.assertIn("T lk_block_reduce_sum(T v, T* scratch)", cuda)
        self.assertIn("lk_block_reduce_sum(", cuda)
        # The embedded warp butterfly is defined (deduplicated) once.
        self.assertEqual(cuda.count("T lk_warp_reduce_sum(unsigned mask, T v)"), 1)
        # Cross-warp barriers are inside the helper, not hand-rolled at the
        # call site: the kernel body shows a single expression call.
        self.assertNotIn("asm", cuda)

    def test_block_extrema_max_min_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), BLOCK_EXTREMA_SOURCE, "block_extrema", ("x", ))
        self.assertEqual(cuda.count("T lk_block_reduce_max(T v, T* scratch)"), 1)
        self.assertEqual(cuda.count("T lk_block_reduce_min(T v, T* scratch)"), 1)
        self.assertEqual(cuda.count("T lk_warp_reduce_max(unsigned mask, T v)"), 1)
        self.assertEqual(cuda.count("T lk_warp_reduce_min(unsigned mask, T v)"), 1)
        # IEEE-extrema policy carries into the block helpers via the
        # embedded warp butterflies, guarded once per translation unit.
        self.assertIn("_lk_isnan_ieee", cuda)
        self.assertNotIn("(o > v) ? o : v", cuda)
        self.assertIn("lk_block_reduce_max(", cuda)
        self.assertIn("lk_block_reduce_min(", cuda)
        self.assertNotIn("asm", cuda)


# ==============================================================================
# Semantic CPU regression: simulate the two-stage block allreduce exactly as
# emitted (per-warp butterfly, lane-0 staging, first-warp combine with
# identity padding, broadcast) and assert the allreduce uniformity + numeric
# value invariants across block sizes, NaN/signed-zero corners and a
# fused-residual-RMSNorm style sum of squares.
# ==============================================================================


def _interp_lk_isnan(x):
    return isinstance(x, float) and x != x


def _interp_lk_signbit(x):
    return math.copysign(1.0, float(x)) < 0


def _interp_combine_max(o, v):
    if not _interp_lk_isnan(o) and (_interp_lk_isnan(v) or o > v or
                                    (o == v and _interp_lk_signbit(v) and not _interp_lk_signbit(o))):
        return o
    return v


def _interp_combine_min(o, v):
    if not _interp_lk_isnan(o) and (_interp_lk_isnan(v) or o < v or
                                    (o == v and not _interp_lk_signbit(v) and _interp_lk_signbit(o))):
        return o
    return v


def _interp_combine_sum(o, v):
    return o + v


def _warp_allreduce(values, combine):
    """Full-warp XOR butterfly over a length-32 lane array."""
    result = list(values)
    step = 16
    while step > 0:
        result = [combine(result[i ^ step], result[i]) for i in range(32)]
        step >>= 1
    return result


def _simulate_block_reduce(values, combine, fill):
    """Mirror of the emitted lk_block_reduce_* helper schedule.

    ``values`` holds one element per thread, len(values) == 32 * num_warps;
    ``fill`` maps the staged partial array to the identity used for lanes
    >= num_warps of the first warp.  Returns the per-thread broadcast.
    """
    n = len(values)
    assert n % 32 == 0 and 32 <= n <= 1024
    nw = n // 32
    scratch = [None] * 33
    # Stage 1: per-warp butterfly; lane 0 stages into scratch[1 + warp].
    for w in range(nw):
        partials = _warp_allreduce(values[w * 32:(w + 1) * 32], combine)
        scratch[1 + w] = partials[0]
    # Stage 2: first warp combines partials; lanes >= nw take the fill.
    stage2 = [scratch[1 + lane] if lane < nw else fill(scratch) for lane in range(32)]
    stage2 = _warp_allreduce(stage2, combine)
    scratch[0] = stage2[0]
    return [scratch[0]] * n


def _fill_sum(scratch):
    return 0.0


def _fill_extrema(scratch):
    # Idempotent duplicate of a valid staged partial (slot 1 == warp 0).
    return scratch[1]


class TestBlockReduceSemantics(unittest.TestCase):

    NAN = float("nan")
    INF = float("inf")
    NINF = float("-inf")

    def _run(self, values, combine, fill):
        broadcast = _simulate_block_reduce(values, combine, fill)
        first = broadcast[0]
        for r in broadcast:
            if _interp_lk_isnan(first) and _interp_lk_isnan(r):
                continue
            self.assertEqual(first, r, "block allreduce lanes disagree")
            self.assertEqual(_interp_lk_signbit(first), _interp_lk_signbit(r), "block allreduce zero-sign disagreement")
        return first

    def test_sum_block_sizes(self):
        for nw in (1, 2, 4, 8, 16, 32):
            with self.subTest(num_warps=nw):
                values = [float(i % 7 - 3) for i in range(32 * nw)]
                total = self._run(values, _interp_combine_sum, _fill_sum)
                self.assertEqual(total, sum(values))

    def test_sum_rmsnorm_style(self):
        # Block-wide fp32 sum of squares from bf16-magnitude values, as an
        # RMSNorm variance numerator needs it (mean = total / H downstream).
        values = [0.25 * ((i % 13) - 6) for i in range(256)]
        total = self._run([v * v for v in values], _interp_combine_sum, _fill_sum)
        self.assertAlmostEqual(total, sum(v * v for v in values))

    def test_max_block_sizes_and_corners(self):
        cases = (
            ([self.NAN] + [1.0] * 31, 1.0),
            ([1.0] * 31 + [self.NAN], 1.0),
            ([-0.0, +0.0] + [-2.0] * 30, +0.0),
            ([self.INF] + [1.0] * 31, self.INF),
            ([self.NINF] + [1.0] * 31, 1.0),
            ([7, 3] + [5] * 30, 7),
        )
        for nw in (1, 2, 4, 32):
            for vals, expected in cases:
                with self.subTest(op="max", nw=nw, expected=expected):
                    values = (vals * nw)[:32 * nw]
                    got = self._run(values, _interp_combine_max, _fill_extrema)
                    self.assertEqual(got, expected)
                    if expected == +0.0:
                        self.assertFalse(_interp_lk_signbit(got), "max must canonicalize to +0.0")

    def test_min_block_sizes_and_corners(self):
        cases = (
            ([self.NAN] + [1.0] * 31, 1.0),
            ([-0.0, +0.0] + [2.0] * 30, -0.0),
            ([self.NINF] + [1.0] * 31, self.NINF),
            ([self.INF] + [1.0] * 31, 1.0),
            ([7, 3] + [5] * 30, 3),
        )
        for nw in (1, 2, 4, 32):
            for vals, expected in cases:
                with self.subTest(op="min", nw=nw, expected=expected):
                    values = (vals * nw)[:32 * nw]
                    got = self._run(values, _interp_combine_min, _fill_extrema)
                    self.assertEqual(got, expected)
                    if expected == -0.0:
                        self.assertTrue(_interp_lk_signbit(got), "min must canonicalize to -0.0")

    def test_all_nan_returns_nan(self):
        for nw in (1, 4):
            with self.subTest(nw=nw):
                got_max = self._run([self.NAN] * 32 * nw, _interp_combine_max, _fill_extrema)
                self.assertTrue(_interp_lk_isnan(got_max))
                got_min = self._run([self.NAN] * 32 * nw, _interp_combine_min, _fill_extrema)
                self.assertTrue(_interp_lk_isnan(got_min))

    def test_cross_warp_extrema_visible(self):
        # The block maximum must see values outside the calling thread's
        # own warp (the whole point of the cross-warp composition).
        values = [1.0] * 128
        values[96 + 17] = 42.0  # lane 17 of warp 3
        got = self._run(values, _interp_combine_max, _fill_extrema)
        self.assertEqual(got, 42.0)
        values = [1.0] * 128
        values[64 + 3] = -42.0  # lane 3 of warp 2
        got = self._run(values, _interp_combine_min, _fill_extrema)
        self.assertEqual(got, -42.0)

    def test_identity_fill_correct_for_partial_first_warp(self):
        # Blocks with num_warps < 32 pad the first-warp combine; padding
        # must never change the result for ANY combine.
        for nw in (1, 2, 3, 5, 31):
            n = 32 * nw
            with self.subTest(nw=nw):
                values = [float(i % 5) for i in range(n)]
                got = self._run(values, _interp_combine_sum, _fill_sum)
                self.assertEqual(got, sum(values))
                got = self._run(values, _interp_combine_max, _fill_extrema)
                self.assertEqual(got, max(values))
                got = self._run(values, _interp_combine_min, _fill_extrema)
                self.assertEqual(got, min(values))


# Canonical fused max+sum use: softmax-style row statistics.  One CTA owns
# one output row; each lane supplies its own fp32 partial; the fused
# block allreduce produces BOTH the row max and the row sum amortised over
# one staging schedule, one barrier pair, and one broadcast round-trip.
FUSED_SOFTMAX_STATS_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def fused_softmax_stats(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = x[tid]
    ll.alloc_shared_memory("s_red", ll.float32, 66)
    m, s = ll.block_reduce_max_sum(v, v, s_red + 0)
    out[tid] = m
    out[128 + tid] = s
'''


class TestBlockReduceMaxSum(unittest.TestCase):
    """block_reduce_max_sum: fused two-stat block allreduce.

    Scheduling/coalescing win over sequencing ``block_reduce_max`` +
    ``block_reduce_sum``: same numeric results, strictly fewer
    ``__syncthreads`` sites, one shared smem staging/broadcast round-trip.
    """

    def test_exported_canonical_name(self):
        self.assertTrue(hasattr(ll, "block_reduce_max_sum"))
        self.assertTrue(getattr(ll.block_reduce_max_sum, BUILTIN_ATTR, False))
        self.assertEqual(block_reduce_intrin.BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS, 66)

    def test_return_type_is_tuple_pair(self):
        from little_kernel.core.type_system import TupleType
        ret = getattr(block_reduce_intrin.block_reduce_max_sum, EVAL_RETURN_TYPE_ATTR)
        for dtype in (ll.float32, ll.float64, ll.int32, ll.uint32, ll.int64, ll.uint64, ll.float16, ll.bfloat16):
            with self.subTest(dtype=dtype):
                t = ret(dtype, dtype, Pointer(dtype))
                self.assertIsInstance(t, TupleType)
                self.assertEqual(tuple(t.element_types), (dtype, dtype))

    def test_return_type_rejects_unsupported_dtype(self):
        ret = getattr(block_reduce_intrin.block_reduce_max_sum, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            ret(ll.int16, ll.int16, Pointer(ll.int16))
        with self.assertRaises(TypeError):
            ret(ll.int8, ll.int8, Pointer(ll.int8))

    def test_return_type_rejects_mismatched_operand_dtypes(self):
        # The helper is a single C++ template, so both value operands must
        # share the same dtype.  A mixed (f32 max, s32 sum) spelling must
        # fail loudly here instead of producing an nvcc error downstream.
        ret = getattr(block_reduce_intrin.block_reduce_max_sum, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            ret(ll.float32, ll.int32, Pointer(ll.float32))
        with self.assertRaises(TypeError):
            ret(ll.int32, ll.float32, Pointer(ll.int32))

    def test_return_type_rejects_non_pointer_scratch(self):
        ret = getattr(block_reduce_intrin.block_reduce_max_sum, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            ret(ll.float32, ll.float32, ll.float32)

    def test_return_type_rejects_mismatched_scratch_element(self):
        ret = getattr(block_reduce_intrin.block_reduce_max_sum, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            ret(ll.float32, ll.float32, Pointer(ll.int32))

    def test_declared_effects(self):
        effects = getattr(block_reduce_intrin.block_reduce_max_sum, MEMORY_EFFECTS_ATTR, ())
        self.assertEqual(len(effects), 4)
        # Two cross-lane register reads (one per value operand).
        warp_effs = effects[:2]
        self.assertEqual({e.operand for e in warp_effs}, {"max_value", "sum_value"})
        for eff in warp_effs:
            self.assertIs(eff.kind, R_RMEM_WARP_SYNC)
        # One smem R/W pair covering BOTH partition regions.
        smem = {e.access: e for e in effects[2:]}
        self.assertEqual(set(smem), {Access.W, Access.R})
        for eff in smem.values():
            self.assertIs(eff.space, MemorySpace.SMEM)
            self.assertEqual(eff.operand, "smem_scratch")
            self.assertIsNotNone(eff.region)
            self.assertEqual(eff.region.extent, block_reduce_intrin.BLOCK_REDUCE_MAX_SUM_SCRATCH_ELEMS)

    def test_codegen_spelling(self):
        out = getattr(block_reduce_intrin.block_reduce_max_sum, CODEGEN_FUNC_ATTR)("vm", "vs", "scratch")
        self.assertIsInstance(out, Builtin)
        self.assertEqual(out.return_val, "lk_block_reduce_max_sum(vm, vs, scratch)")
        # One shared C++ helper emitting a cuda::std::tuple<T, T> result.
        self.assertIn("cuda::std::tuple<T, T>", out.body)
        self.assertIn("lk_block_reduce_max_sum(T vmax, T vsum, T* scratch)", out.body)
        # Same warp helpers as the single-value block reductions: numeric
        # semantics are defined once, in reduce.py.
        self.assertIn("lk_warp_reduce_max(0xffffffffu, vmax)", out.body)
        self.assertIn("lk_warp_reduce_sum(0xffffffffu, vsum)", out.body)
        # Two partition regions: max in [2, 34), sum in [34, 66).
        self.assertIn("scratch[2 + warp]  = vmax;", out.body)
        self.assertIn("scratch[34 + warp] = vsum;", out.body)
        self.assertIn("scratch[2 + lane]  : scratch[2];", out.body)
        self.assertIn("scratch[34 + lane] : T(0);", out.body)
        self.assertIn("scratch[0] = x_max;", out.body)
        self.assertIn("scratch[1] = x_sum;", out.body)
        # Three barriers (stage partials / publish broadcast / protect
        # reuse) vs SIX for two sequenced single-value allreduces.
        self.assertEqual(out.body.count("__syncthreads();"), 3)
        # Return is materialised through cuda::std::make_tuple so the
        # caller-side unpack path (cuda::std::get<i>) type-checks.
        self.assertIn("cuda::std::make_tuple(out_max, out_sum)", out.body)

    def test_codegen_half_includes_only_when_needed(self):
        codegen = getattr(block_reduce_intrin.block_reduce_max_sum, CODEGEN_FUNC_ATTR)
        fp32 = codegen("vm", "vs", "scratch", _arg_dtypes=[ll.float32, ll.float32, Pointer(ll.float32)])
        self.assertEqual(fp32.includes, [])
        f16 = codegen("vm", "vs", "scratch", _arg_dtypes=[ll.float16, ll.float16, Pointer(ll.float16)])
        self.assertEqual(f16.includes, ["<cuda_fp16.h>"])
        bf16 = codegen("vm", "vs", "scratch", _arg_dtypes=[ll.bfloat16, ll.bfloat16, Pointer(ll.bfloat16)])
        self.assertEqual(bf16.includes, ["<cuda_bf16.h>"])

    def test_fused_lowers_with_fewer_barrier_sites_than_sequenced(self):
        """End-to-end: fused form has exactly 3 barrier sites; sequenced
        (max + sum on separate allreduces) has 6.  Same numeric value."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)

            def do(source):
                task = d / "task"
                if task.exists():
                    import shutil
                    shutil.rmtree(task)
                task.mkdir()
                (task / "model.py").write_text("class Model: pass\n")
                (task / "test_cases.py").write_text("test_cases = []\n")
                program = d / "program"
                if program.exists():
                    import shutil
                    shutil.rmtree(program)
                program.mkdir()
                (program / "draft.py").write_text(source)
                (program / "export.json").write_text(
                    json.dumps({
                        "kernel": "k",
                        "function": "k",
                        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
                        "output": {"name": "out", "dtype": "float32", "shape": ["x[0]"]},
                        "scalars": [],
                        "launch": {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": [128, 1, 1]},
                    }))
                output = d / "output"
                if output.exists():
                    import shutil
                    shutil.rmtree(output)
                lower(task, program, output, 0)
                kernels = sorted((output / "kernels").glob("*.cu"))
                self.assertEqual(len(kernels), 1, kernels)
                return kernels[0].read_text()

            fused_src = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def k(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = x[tid]
    ll.alloc_shared_memory("s_red", ll.float32, 66)
    m, s = ll.block_reduce_max_sum(v, v, s_red + 0)
    out[tid] = m
    out[128 + tid] = s
'''
            sep_src = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def k(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    v: ll.float32 = x[tid]
    ll.alloc_shared_memory("s_red", ll.float32, 66)
    m: ll.float32 = ll.block_reduce_max(v, s_red + 0)
    s: ll.float32 = ll.block_reduce_sum(v, s_red + 33)
    out[tid] = m
    out[128 + tid] = s
'''
            fused_cuda = do(fused_src)
            sep_cuda = do(sep_src)

        # Fused: exactly 3 barrier sites (stage / publish / protect-reuse).
        self.assertEqual(fused_cuda.count("__syncthreads"), 3)
        # Sequenced: 3 barriers per helper invocation, two helpers = 6.
        self.assertEqual(sep_cuda.count("__syncthreads"), 6)
        # Fused emits one fused helper; sequenced emits the two
        # single-value helpers.
        self.assertIn("lk_block_reduce_max_sum(", fused_cuda)
        self.assertIn("cuda::std::tuple<T, T>", fused_cuda)
        self.assertIn("lk_block_reduce_max(", sep_cuda)
        self.assertIn("lk_block_reduce_sum(", sep_cuda)
        # Tuple unpack materialises at the call site through cuda::std::get.
        self.assertIn("cuda::std::get<0>(", fused_cuda)
        self.assertIn("cuda::std::get<1>(", fused_cuda)

    def test_fused_semantics_match_sequenced_simulation(self):
        """The fused two-stat simulation produces EXACTLY the same block
        totals as sequencing the two single-stat simulations on the same
        inputs (bit-level uniformity of the IEEE-extrema policy is carried
        through by construction)."""

        def simulate_fused(values):
            """Same schedule as lk_block_reduce_max_sum: per-warp butterfly
            for both stats, stage 2+w and 34+w, warp 0 combines both,
            broadcast via slots 0 and 1."""
            n = len(values)
            assert n % 32 == 0 and 32 <= n <= 1024
            nw = n // 32
            scratch = [None] * 66
            for w in range(nw):
                seg = values[w * 32:(w + 1) * 32]
                part_max = _warp_allreduce(seg, _interp_combine_max)
                part_sum = _warp_allreduce(seg, _interp_combine_sum)
                scratch[2 + w] = part_max[0]
                scratch[34 + w] = part_sum[0]
            # Stage 2: warp 0 with identity fills; max pads duplicate slot
            # 2 (idempotent), sum pads 0.
            stage2_max = [scratch[2 + lane] if lane < nw else scratch[2] for lane in range(32)]
            stage2_sum = [scratch[34 + lane] if lane < nw else 0.0 for lane in range(32)]
            stage2_max = _warp_allreduce(stage2_max, _interp_combine_max)
            stage2_sum = _warp_allreduce(stage2_sum, _interp_combine_sum)
            scratch[0] = stage2_max[0]
            scratch[1] = stage2_sum[0]
            return scratch[0], scratch[1]

        for nw in (1, 2, 4, 8, 16, 32):
            with self.subTest(num_warps=nw):
                values = [float((i % 11) - 5) * 0.5 for i in range(32 * nw)]
                fused_max, fused_sum = simulate_fused(values)
                sep_max = _simulate_block_reduce(values, _interp_combine_max, _fill_extrema)[0]
                sep_sum = _simulate_block_reduce(values, _interp_combine_sum, _fill_sum)[0]
                self.assertEqual(fused_max, sep_max)
                self.assertEqual(fused_sum, sep_sum)


if __name__ == "__main__":
    unittest.main()
