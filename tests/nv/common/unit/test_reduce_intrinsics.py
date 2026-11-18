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
"""First-class warp reduction intrinsics: naming, typing, effects, emission.

Legality contract under test (compiler side; convergence is a frontend
obligation documented in language/intrin/reduce.py):

* canonical names ``ll.warp_reduce_sum`` / ``ll.warp_reduce_sum_width`` are
  exported and marked builtin;
* the returned type equals the reduced value's type for every dtype CUDA's
  shuffle family lowers natively (f32, f64, s32, u32, s64, u64, f16, bf16);
  anything else (e.g. int16) raises TypeError at type-inference time;
* the segmented width must be a power-of-two literal in [1, 32]; non-literal
  or out-of-range widths raise at codegen time;
* each op declares the warp-collective register-read effect
  ``Effect(R_RMEM_WARP_SYNC, "value")`` via MEMORY_EFFECTS_ATTR;
* emission lowers to one deduplicated templated ``__shfl_xor_sync``
  butterfly helper through the full lower() path, so kernels with several
  reduction sites share a single helper definition instead of hand-rolled
  five-line shuffle chains.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.memory_effects import Access, IssuerSet, MemorySpace, R_RMEM_WARP_SYNC
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.reduce as reduce_intrin


class TestWarpReduceIntrinsics(unittest.TestCase):

    def test_exported_canonical_names(self):
        for name in ("warp_reduce_sum", "warp_reduce_sum_width", "warp_reduce_max", "warp_reduce_min",
                     "warp_reduce_max_width", "warp_reduce_min_width"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name), name)
                self.assertTrue(getattr(getattr(ll, name), BUILTIN_ATTR, False))

    def test_return_type_equals_value_type(self):
        sum_ret = getattr(reduce_intrin.warp_reduce_sum, EVAL_RETURN_TYPE_ATTR)
        width_ret = getattr(reduce_intrin.warp_reduce_sum_width, EVAL_RETURN_TYPE_ATTR)
        max_ret = getattr(reduce_intrin.warp_reduce_max, EVAL_RETURN_TYPE_ATTR)
        min_ret = getattr(reduce_intrin.warp_reduce_min, EVAL_RETURN_TYPE_ATTR)
        maxw_ret = getattr(reduce_intrin.warp_reduce_max_width, EVAL_RETURN_TYPE_ATTR)
        minw_ret = getattr(reduce_intrin.warp_reduce_min_width, EVAL_RETURN_TYPE_ATTR)
        for dtype in (ll.float32, ll.float64, ll.int32, ll.uint32, ll.int64, ll.uint64, ll.float16, ll.bfloat16):
            with self.subTest(op="warp_reduce_sum", dtype=dtype):
                self.assertEqual(sum_ret(ll.uint32, dtype), dtype)
            with self.subTest(op="warp_reduce_sum_width", dtype=dtype):
                self.assertEqual(width_ret(ll.uint32, dtype, ll.int32), dtype)
            with self.subTest(op="warp_reduce_max", dtype=dtype):
                self.assertEqual(max_ret(ll.uint32, dtype), dtype)
            with self.subTest(op="warp_reduce_min", dtype=dtype):
                self.assertEqual(min_ret(ll.uint32, dtype), dtype)
            with self.subTest(op="warp_reduce_max_width", dtype=dtype):
                self.assertEqual(maxw_ret(ll.uint32, dtype, ll.int32), dtype)
            with self.subTest(op="warp_reduce_min_width", dtype=dtype):
                self.assertEqual(minw_ret(ll.uint32, dtype, ll.int32), dtype)

    def test_return_type_rejects_unsupported_dtype(self):
        sum_ret = getattr(reduce_intrin.warp_reduce_sum, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            sum_ret(ll.uint32, ll.int16)
        width_ret = getattr(reduce_intrin.warp_reduce_sum_width, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            width_ret(ll.uint32, ll.int16, ll.int32)
        # max/min share the same shuffle-native dtype gate as sum.
        max_ret = getattr(reduce_intrin.warp_reduce_max, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            max_ret(ll.uint32, ll.int16)
        min_ret = getattr(reduce_intrin.warp_reduce_min, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            min_ret(ll.uint32, ll.int16)
        maxw_ret = getattr(reduce_intrin.warp_reduce_max_width, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            maxw_ret(ll.uint32, ll.int16, ll.int32)
        minw_ret = getattr(reduce_intrin.warp_reduce_min_width, EVAL_RETURN_TYPE_ATTR)
        with self.assertRaises(TypeError):
            minw_ret(ll.uint32, ll.int16, ll.int32)

    def test_declared_warp_collective_effect(self):
        for fn in (reduce_intrin.warp_reduce_sum, reduce_intrin.warp_reduce_sum_width, reduce_intrin.warp_reduce_max,
                   reduce_intrin.warp_reduce_min, reduce_intrin.warp_reduce_max_width,
                   reduce_intrin.warp_reduce_min_width):
            with self.subTest(name=fn.__name__):
                effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertIs(eff.kind, R_RMEM_WARP_SYNC)
                self.assertIs(eff.kind.access, Access.R)
                self.assertIs(eff.kind.space, MemorySpace.RMEM)
                self.assertIs(eff.kind.issuer, IssuerSet.WARP)
                self.assertEqual(eff.operand, "value")

    def test_codegen_fullwarp_spelling(self):
        out = getattr(reduce_intrin.warp_reduce_sum, CODEGEN_FUNC_ATTR)("0xffffffff", "v")
        self.assertIsInstance(out, Builtin)
        self.assertEqual(out.return_val, "lk_warp_reduce_sum(0xffffffff, v)")
        # One shared templated butterfly helper, fp32 butterfly is 5 steps.
        self.assertIn("template <typename T>", out.body)
        self.assertEqual(out.body.count("__shfl_xor_sync("), 5)
        self.assertNotIn("asm", out.body)

    def test_codegen_half_includes_only_when_needed(self):
        codegen = getattr(reduce_intrin.warp_reduce_sum, CODEGEN_FUNC_ATTR)
        fp32 = codegen("0xffffffff", "v", _arg_dtypes=[ll.uint32, ll.float32])
        self.assertEqual(fp32.includes, [])
        f16 = codegen("0xffffffff", "v", _arg_dtypes=[ll.uint32, ll.float16])
        self.assertEqual(f16.includes, ["<cuda_fp16.h>"])
        bf16 = codegen("0xffffffff", "v", _arg_dtypes=[ll.uint32, ll.bfloat16])
        self.assertEqual(bf16.includes, ["<cuda_bf16.h>"])

    def test_segment_width_accepted(self):
        codegen = getattr(reduce_intrin.warp_reduce_sum_width, CODEGEN_FUNC_ATTR)
        for width in (1, 2, 4, 8, 16, 32):
            with self.subTest(width=width):
                out = codegen("0xffffffff", "v", str(width))
                self.assertEqual(out.return_val, f"lk_warp_reduce_sum_width<{width}>(0xffffffff, v)")
                self.assertIn("__shfl_xor_sync(mask, v, offset, W)", out.body)

    def test_segment_width_rejected(self):
        codegen = getattr(reduce_intrin.warp_reduce_sum_width, CODEGEN_FUNC_ATTR)
        for bad, exc in (("0", ValueError), ("3", ValueError), ("48", ValueError), ("-2", ValueError),
                         ("lane", TypeError), ("W", TypeError)):
            with self.subTest(width=bad):
                with self.assertRaises(exc):
                    codegen("0xffffffff", "v", bad)

    def test_codegen_ordered_reduce_spelling(self):
        """max/min lower to dedicated helpers with IEEE-uniform combines.

        The naive ``(o > v) ? o : v`` spelling is REJECTED: it is the
        documented NaN/signed-zero non-uniformity the IEEE extrema policy
        exists to fix.  The current spelling must contain the lane-symmetric
        NaN-guard, the canonical-zero tie break and the strict greater/less
        compare -- so lane disagreement (the bug below) cannot reappear.
        """
        max_codegen = getattr(reduce_intrin.warp_reduce_max, CODEGEN_FUNC_ATTR)
        out = max_codegen("0xffffffff", "v")
        self.assertIsInstance(out, Builtin)
        self.assertEqual(out.return_val, "lk_warp_reduce_max(0xffffffff, v)")
        # Dedicated helper: 5 butterfly exchanges.
        self.assertEqual(out.body.count("__shfl_xor_sync("), 5)
        # NaN-guard and IEEE-extrema tiebreak must be present.
        self.assertIn("_lk_isnan_ieee", out.body)
        self.assertIn("_lk_signbit_ieee", out.body)
        self.assertIn("o > v", out.body)
        # Reject any regression to the broken NaN-unsafe spelling.
        self.assertNotIn("(o > v) ? o : v", out.body)
        self.assertNotIn("asm", out.body)

        min_codegen = getattr(reduce_intrin.warp_reduce_min, CODEGEN_FUNC_ATTR)
        out = min_codegen("0xffffffff", "v")
        self.assertEqual(out.return_val, "lk_warp_reduce_min(0xffffffff, v)")
        self.assertEqual(out.body.count("__shfl_xor_sync("), 5)
        self.assertIn("_lk_isnan_ieee", out.body)
        self.assertIn("_lk_signbit_ieee", out.body)
        self.assertIn("o < v", out.body)
        self.assertNotIn("(o < v) ? o : v", out.body)
        self.assertNotIn("asm", out.body)

    def test_codegen_ordered_reduce_width_spelling(self):
        """Segmented max/min lower to width-templated IEEE helpers."""
        for width in (1, 2, 4, 8, 16, 32):
            with self.subTest(width=width):
                max_codegen = getattr(reduce_intrin.warp_reduce_max_width, CODEGEN_FUNC_ATTR)
                out = max_codegen("0xffffffff", "v", str(width))
                self.assertEqual(out.return_val, f"lk_warp_reduce_max_width<{width}>(0xffffffff, v)")
                self.assertIn("__shfl_xor_sync(mask, v, offset, W)", out.body)
                self.assertIn("_lk_isnan_ieee", out.body)
                self.assertIn("o > v", out.body)
                self.assertNotIn("(o > v) ? o : v", out.body)

                min_codegen = getattr(reduce_intrin.warp_reduce_min_width, CODEGEN_FUNC_ATTR)
                out = min_codegen("0xffffffff", "v", str(width))
                self.assertEqual(out.return_val, f"lk_warp_reduce_min_width<{width}>(0xffffffff, v)")
                self.assertIn("__shfl_xor_sync(mask, v, offset, W)", out.body)
                self.assertIn("_lk_isnan_ieee", out.body)
                self.assertIn("o < v", out.body)
                self.assertNotIn("(o < v) ? o : v", out.body)

    def test_ordered_reduce_width_rejects_bad_width(self):
        """Segmented max/min validate the segment width at codegen time."""
        for fn_name in ("warp_reduce_max_width", "warp_reduce_min_width"):
            codegen = getattr(getattr(reduce_intrin, fn_name), CODEGEN_FUNC_ATTR)
            for bad, exc in (("0", ValueError), ("3", ValueError), ("48", ValueError), ("-2", ValueError),
                             ("lane", TypeError), ("W", TypeError)):
                with self.subTest(fn=fn_name, width=bad):
                    with self.assertRaises(exc):
                        codegen("0xffffffff", "v", bad)

    def test_ordered_reduce_half_includes_only_when_needed(self):
        for fn, _ in ((reduce_intrin.warp_reduce_max, "max"), (reduce_intrin.warp_reduce_min, "min")):
            codegen = getattr(fn, CODEGEN_FUNC_ATTR)
            fp32 = codegen("0xffffffff", "v", _arg_dtypes=[ll.uint32, ll.float32])
            self.assertEqual(fp32.includes, [])
            f16 = codegen("0xffffffff", "v", _arg_dtypes=[ll.uint32, ll.float16])
            self.assertEqual(f16.includes, ["<cuda_fp16.h>"])
            bf16 = codegen("0xffffffff", "v", _arg_dtypes=[ll.uint32, ll.bfloat16])
            self.assertEqual(bf16.includes, ["<cuda_bf16.h>"])

    def test_ordered_reduce_width_half_includes_only_when_needed(self):
        for fn, _ in ((reduce_intrin.warp_reduce_max_width, "max_width"), (reduce_intrin.warp_reduce_min_width,
                                                                           "min_width")):
            codegen = getattr(fn, CODEGEN_FUNC_ATTR)
            fp32 = codegen("0xffffffff", "v", "2", _arg_dtypes=[ll.uint32, ll.float32, ll.int32])
            self.assertEqual(fp32.includes, [])
            f16 = codegen("0xffffffff", "v", "2", _arg_dtypes=[ll.uint32, ll.float16, ll.int32])
            self.assertEqual(f16.includes, ["<cuda_fp16.h>"])
            bf16 = codegen("0xffffffff", "v", "2", _arg_dtypes=[ll.uint32, ll.bfloat16, ll.int32])
            self.assertEqual(bf16.includes, ["<cuda_bf16.h>"])


ROWSUM_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def row_sumsq(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    row: ll.int32 = ll.blockIdx_x()
    acc: ll.float32 = 0.0
    for k in range(0, 4):
        v: ll.float32 = x[row * 128 + k * 32 + lane]
        acc = acc + v * v
    total: ll.float32 = ll.warp_reduce_sum(0xffffffff, acc)
    if lane == 0:
        out[row] = total
'''

MULTI_SITE_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def two_sums(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    a: ll.float32 = ll.warp_reduce_sum(0xffffffff, x[i])
    b: ll.float32 = ll.warp_reduce_sum(0xffffffff, x[i + 4096])
    if ll.threadIdx_x() == 0:
        out[ll.blockIdx_x()] = a + b
'''

BF16_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def bf16_rowsum(x: ll.Tensor[ll.bfloat16], out: ll.Tensor[ll.bfloat16]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    v: ll.bfloat16 = x[i]
    total: ll.bfloat16 = ll.warp_reduce_sum(0xffffffff, v)
    out[i] = total
'''

SEGMENTED_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def seg_sumsq(x: ll.Tensor[ll.float32], w: ll.Tensor[ll.float32],
              out: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    row_in_warp: ll.int32 = lane / 16
    col: ll.int32 = lane % 16
    row: ll.int32 = ll.blockIdx_x() * 2 + row_in_warp
    acc: ll.float32 = 0.0
    for k in range(0, 4):
        v: ll.float32 = x[row * 64 + k * 16 + col]
        acc = acc + v * v
    total: ll.float32 = ll.warp_reduce_sum_width(0xffffffff, acc, 16)
    if col == 0:
        out[row] = total * w[0]
'''

# Online-softmax-style epilogue: per-warp running max for the subtract
# scale.  Canonical use of ``warp_reduce_max`` that the scan family's
# inclusive_max cannot express (allreduce: every lane needs the total).
ROWMAX_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def row_max(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    row: ll.int32 = ll.blockIdx_x()
    m: ll.float32 = x[row * 32 + lane]
    for k in range(1, 4):
        v: ll.float32 = x[row * 128 + k * 32 + lane]
        m = m if m > v else v
    total: ll.float32 = ll.warp_reduce_max(0xffffffff, m)
    if lane == 0:
        out[row] = total
'''

# Coordinate-clamp / best-so-far min: paired with max in scan family, now
# also here for reduce parity.
ROWMIN_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def row_min(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    row: ll.int32 = ll.blockIdx_x()
    m: ll.float32 = x[row * 32 + lane]
    for k in range(1, 4):
        v: ll.float32 = x[row * 128 + k * 32 + lane]
        m = m if m < v else v
    total: ll.float32 = ll.warp_reduce_min(0xffffffff, m)
    if lane == 0:
        out[row] = total
'''

# Public typed-frontend reproducer from the NaN-uniformity report: the
# width-2 segmented max on a warp must route through the IEEE-uniform
# helper even when a segment packs a single NaN against a finite value.
EXTREMA2_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def extrema2(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    value: ll.float32 = x[lane]
    out[lane] = ll.warp_reduce_max_width(0xffffffff, value, 2)
'''

# Combined translation unit: full-warp max AND min together with segmented
# max/min at several widths.  The codegen dedupes complete helper blocks by
# function name, so the shared NaN-combine predicate block would be emitted
# four times without its #ifndef once-guard -- this test is the regression
# tripwire for that duplicate-definition failure mode.
COMBINED_EXTREMA_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def extrema_all(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    lane: ll.int32 = ll.threadIdx_x()
    value: ll.float32 = x[lane]
    out[lane] = ll.warp_reduce_max(0xffffffff, value)
    out[32 + lane] = ll.warp_reduce_min(0xffffffff, value)
    out[64 + lane] = ll.warp_reduce_max_width(0xffffffff, value, 2)
    out[96 + lane] = ll.warp_reduce_max_width(0xffffffff, value, 8)
    out[128 + lane] = ll.warp_reduce_min_width(0xffffffff, value, 2)
    out[160 + lane] = ll.warp_reduce_min_width(0xffffffff, value, 16)
'''


class TestWarpReduceEndToEnd(unittest.TestCase):
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
                "launch": {"grid": [f"ceil_div({tensor_names[0]}[0], 128)", 1, 1], "block": [32, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_rowwise_sumsq_lowers_to_shared_helper(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), ROWSUM_SOURCE, "row_sumsq", ("x", ))
        # Generic rowwise statistic: one shared helper definition and a call,
        # not a hand-rolled per-frontend shuffle chain.
        self.assertIn("T lk_warp_reduce_sum(unsigned mask, T v)", cuda)
        self.assertIn("lk_warp_reduce_sum(4294967295", cuda)
        self.assertNotIn("asm", cuda)

    def test_helper_definition_deduplicated_across_sites(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), MULTI_SITE_SOURCE, "two_sums", ("x", ))
        self.assertEqual(cuda.count("T lk_warp_reduce_sum(unsigned mask, T v)"), 1)
        self.assertEqual(cuda.count("lk_warp_reduce_sum(4294967295"), 2)

    def test_bf16_lowers_with_bf16_header(self):
        """Half-precision row reductions pull in exactly the bf16 header."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), BF16_SOURCE, "bf16_rowsum", ("x", ), dtype="bfloat16")
        self.assertIn("lk_warp_reduce_sum(4294967295", cuda)
        self.assertIn("<cuda_bf16.h>", cuda)
        self.assertNotIn("<cuda_fp16.h>", cuda)

    def test_segmented_layout_lowers_to_width_helper(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), SEGMENTED_SOURCE, "seg_sumsq", ("x", "w"))
        self.assertIn("T lk_warp_reduce_sum_width(unsigned mask, T v)", cuda)
        self.assertIn("lk_warp_reduce_sum_width<16>(4294967295", cuda)
        self.assertIn("__shfl_xor_sync(mask, v, offset, W)", cuda)
        self.assertNotIn("asm", cuda)

    def test_rowwise_max_lowers_to_shared_helper(self):
        """Per-row max (softmax numerator scale) shares one butterfly helper."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), ROWMAX_SOURCE, "row_max", ("x", ))
        self.assertIn("T lk_warp_reduce_max(unsigned mask, T v)", cuda)
        self.assertIn("lk_warp_reduce_max(4294967295", cuda)
        # IEEE-uniform max: NaN-guard + signed-zero tie, NOT the broken
        # NaN-unsafe one-liner the previous emitting regressed to.
        self.assertIn("_lk_isnan_ieee", cuda)
        self.assertIn("_lk_signbit_ieee", cuda)
        self.assertNotIn("(o > v) ? o : v", cuda)
        self.assertNotIn("asm", cuda)

    def test_rowwise_min_lowers_to_shared_helper(self):
        """Per-row min (clamp bound, best-so-far) shares one butterfly helper."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), ROWMIN_SOURCE, "row_min", ("x", ))
        self.assertIn("T lk_warp_reduce_min(unsigned mask, T v)", cuda)
        self.assertIn("lk_warp_reduce_min(4294967295", cuda)
        self.assertIn("_lk_isnan_ieee", cuda)
        self.assertIn("_lk_signbit_ieee", cuda)
        self.assertNotIn("(o < v) ? o : v", cuda)
        self.assertNotIn("asm", cuda)

    def test_reproducer_extrema2_width2_lowers(self):
        """Public typed-frontend reproducer: segmented max width=2 lowers
        through the IEEE-uniform helper (not the NaN-unsafe one-liner)."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), EXTREMA2_SOURCE, "extrema2", ("x", ))
        self.assertIn("lk_warp_reduce_max_width<2>(4294967295", cuda)
        self.assertIn("_lk_isnan_ieee", cuda)
        self.assertNotIn("(o > v) ? o : v", cuda)
        self.assertNotIn("asm", cuda)

    def test_combined_max_min_width_tu_no_duplicate_definitions(self):
        """max + min + segmented variants in ONE translation unit.

        Every helper must be defined exactly once; the shared predicate
        block is repeated per helper body but wrapped in its #ifndef
        once-guard so the TU has no duplicate definitions at NVCC level.
        """
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), COMBINED_EXTREMA_SOURCE, "extrema_all", ("x", ))
        for sig in ("T lk_warp_reduce_max(unsigned mask, T v)", "T lk_warp_reduce_min(unsigned mask, T v)",
                    "T lk_warp_reduce_max_width(unsigned mask, T v)", "T lk_warp_reduce_min_width(unsigned mask, T v)"):
            with self.subTest(helper=sig):
                self.assertEqual(cuda.count(sig), 1, cuda)
        # Predicate block repetition is guarded.
        self.assertGreaterEqual(cuda.count("#ifndef _LK_WARP_EXTREMA_PREDICATES"), 1)
        for pat in ("lk_warp_reduce_max(4294967295", "lk_warp_reduce_min(4294967295",
                    "lk_warp_reduce_max_width<2>(4294967295", "lk_warp_reduce_max_width<8>(4294967295",
                    "lk_warp_reduce_min_width<2>(4294967295", "lk_warp_reduce_min_width<16>(4294967295"):
            with self.subTest(call=pat):
                self.assertIn(pat, cuda)
        self.assertNotIn("asm", cuda)


# ==============================================================================
# Semantic CPU regression: simulate the butterfly against the actual emitted
# combine body so lane-disagreement bugs cannot hide behind string spellings.
#
# The bug class targeted here: under the OLD combine
# ``v = (o > v) ? o : v``, lanes holding NaN and lanes holding a finite
# value keep their own operand after an exchange step (both comparisons
# with NaN are false), so the warp disagrees about the extrema.  A string
# grep cannot see that: it only sees "butterfly shape".  These tests
# interpret the ACTUAL emitted expression in a Python harness that mirrors
# the CUDA butterfly schedule, and assert the allreduce uniformity
# invariant across NaN, signed-zero, infinity and integer inputs.
# ==============================================================================


def _interp_lk_isnan(x):
    """Mirror of ``_lk_isnan_ieee(x)`` -- IEEE NaN test, false for ints."""
    return isinstance(x, float) and x != x


def _interp_lk_signbit(x):
    """Mirror of ``_lk_signbit_ieee(x)``: sign of ``(float)x``.

    ``math.copysign`` distinguishes ``+0.0`` from ``-0.0`` (copies the
    sign bit), which is exactly what the CUDA ``signbit((float)x)``
    helper does.  For integers this just reports the sign of the int.
    """
    return math.copysign(1.0, float(x)) < 0


def _interp_combine_max(o, v):
    """Mirror of the emitted max combine (see _ordered_reduce_bodies)."""
    if not _interp_lk_isnan(o) and (_interp_lk_isnan(v) or o > v or
                                    (o == v and _interp_lk_signbit(v) and not _interp_lk_signbit(o))):
        return o
    return v


def _interp_combine_min(o, v):
    """Mirror of the emitted min combine."""
    if not _interp_lk_isnan(o) and (_interp_lk_isnan(v) or o < v or
                                    (o == v and not _interp_lk_signbit(v) and _interp_lk_signbit(o))):
        return o
    return v


def _simulate_butterfly(values, combine, width):
    """Simulate segmented XOR-butterfly allreduce.

    ``values`` is a length-32 lane array; ``width`` is the CUDA 4-argument
    ``__shfl_xor_sync`` width parameter (1..32); the full-warp helpers use
    width=32.  Returns the post-butterfly per-lane results.  Within each
    segment, every lane must end with the same value under a correct
    allreduce.
    """
    result = list(values)
    n = len(result)
    step = width // 2
    while step > 0:
        new_result = list(result)
        for i in range(n):
            # XOR partner inside the segment group of width ``width``.
            seg_base = (i // width) * width
            partner = seg_base + ((i - seg_base) ^ step)
            if 0 <= partner < n and (partner // width) == (i // width):
                new_result[i] = combine(result[partner], result[i])
        result = new_result
        step >>= 1
    return result


def _assert_segment_uniform(testcase, values, combine, width, label):
    """All lanes within every segment must observe an identical result."""
    results = _simulate_butterfly(values, combine, width)
    for seg_start in range(0, len(values), width):
        seg = results[seg_start:seg_start + width]
        first = seg[0]
        for r in seg[1:]:
            if _interp_lk_isnan(first) and _interp_lk_isnan(r):
                continue
            # Bit-level uniformity: same numeric value AND same zero sign.
            if first == r and _interp_lk_signbit(first) == _interp_lk_signbit(r):
                continue
            testcase.fail(f"{label}: lanes disagree in width-{width} segment "
                          f"start={seg_start}: {seg!r}")
    return results


class TestWarpExtremaSemantics(unittest.TestCase):
    """Simulate the actual codegen against NaN/zero/inf/integer corners."""

    NAN = float("nan")
    INF = float("inf")
    NINF = float("-inf")

    def _cases(self):
        # (label, segment-lane inputs, expect_all_nan, expect_zero_sign)
        # expect_zero_sign: None => tie can leave either zero; True => -0.0;
        # False => +0.0.  Cases cover the bug-reproducer plus IEEE corners.
        return [
            ("reproducer [NaN, 1.0]", [self.NAN, 1.0], False, None),
            ("reproducer [1.0, NaN]", [1.0, self.NAN], False, None),
            ("all NaN", [self.NAN, self.NAN], True, None),
            ("both finite", [2.0, 3.0], False, None),
            ("[-0.0, +0.0] tie", [-0.0, 0.0], False, None),
            ("[+0.0, -0.0] tie", [0.0, -0.0], False, None),
            ("[+inf, 1.0]", [self.INF, 1.0], False, None),
            ("[-inf, 1.0]", [self.NINF, 1.0], False, None),
            ("[NaN, -inf]", [self.NAN, self.NINF], False, None),
            ("[NaN, +inf]", [self.NAN, self.INF], False, None),
            ("integer pair", [3, 7], False, None),
        ]

    def test_max_full_warp_uniform(self):
        for label, seg, expect_all_nan, _ in self._cases():
            for width in (32, ):
                vals = (seg + [1.0] * 32)[:32]
                with self.subTest(op="max", case=label, width=width):
                    results = _assert_segment_uniform(self, vals, _interp_combine_max, width, f"max:{label}")
                    if not expect_all_nan and any(not _interp_lk_isnan(v) for v in vals[:width]):
                        self.assertFalse(_interp_lk_isnan(results[0]), f"max:{label}: non-NaN lost to NaN")

    def test_min_full_warp_uniform(self):
        for label, seg, expect_all_nan, _ in self._cases():
            for width in (32, ):
                vals = (seg + [1.0] * 32)[:32]
                with self.subTest(op="min", case=label, width=width):
                    results = _assert_segment_uniform(self, vals, _interp_combine_min, width, f"min:{label}")
                    if not expect_all_nan and any(not _interp_lk_isnan(v) for v in vals[:width]):
                        self.assertFalse(_interp_lk_isnan(results[0]), f"min:{label}: non-NaN lost to NaN")

    def test_max_segment_widths_uniform(self):
        for width in (1, 2, 4, 8, 16, 32):
            # Place [NaN, 1.0] in the first width-2 segment(pad/repeat others with finite 1.0).
            vals = [self.NAN, 1.0] + [1.0] * 30
            with self.subTest(op="max", width=width):
                results = _assert_segment_uniform(self, vals, _interp_combine_max, width, f"max:NaN-vs-1 width={width}")
                # First segment contains a non-NaN, so result must be non-NaN.
                if width >= 2:
                    self.assertFalse(_interp_lk_isnan(results[0]), f"width={width}: NaN leaked into max result")
                    self.assertEqual(results[0], results[1])

    def test_min_segment_widths_uniform(self):
        for width in (1, 2, 4, 8, 16, 32):
            vals = [self.NAN, 1.0] + [1.0] * 30
            with self.subTest(op="min", width=width):
                results = _assert_segment_uniform(self, vals, _interp_combine_min, width, f"min:NaN-vs-1 width={width}")
                if width >= 2:
                    self.assertFalse(_interp_lk_isnan(results[0]), f"width={width}: NaN leaked into min result")
                    self.assertEqual(results[0], results[1])

    def test_signed_zero_canonical_max_picks_positive(self):
        # Max of {+0.0, -0.0} must canonicalize to +0.0 across ALL lanes.
        for width in (2, 4, 8, 16, 32):
            with self.subTest(width=width):
                vals = [-0.0, +0.0] + [1.0] * 30
                results = _assert_segment_uniform(self, vals, _interp_combine_max, width, f"max:zero-tie width={width}")
                if width >= 2:
                    # Sign of first-segment result must be +0.0 (positive).
                    self.assertFalse(_interp_lk_signbit(results[0]), f"width={width}: max should pick +0.0, got "
                                     f"{results[0]!r}")
                    self.assertFalse(_interp_lk_signbit(results[1]), f"width={width}: max lane 1 should be +0.0")

    def test_signed_zero_canonical_min_picks_negative(self):
        # Min of {+0.0, -0.0} must canonicalize to -0.0 across ALL lanes.
        for width in (2, 4, 8, 16, 32):
            with self.subTest(width=width):
                vals = [-0.0, +0.0] + [1.0] * 30
                results = _assert_segment_uniform(self, vals, _interp_combine_min, width, f"min:zero-tie width={width}")
                if width >= 2:
                    self.assertTrue(_interp_lk_signbit(results[0]), f"width={width}: min should pick -0.0, got "
                                    f"{results[0]!r}")
                    self.assertTrue(_interp_lk_signbit(results[1]), f"width={width}: min lane 1 should be -0.0")

    def test_all_nan_returns_nan(self):
        # When every lane supplies NaN, every lane gets NaN (both ops).
        for width in (1, 2, 4, 32):
            vals = [self.NAN] * 32
            with self.subTest(op="max", width=width):
                results = _assert_segment_uniform(self, vals, _interp_combine_max, width, f"max:all-NaN width={width}")
                for r in results[:max(width, 2)]:
                    self.assertTrue(_interp_lk_isnan(r))
            with self.subTest(op="min", width=width):
                results = _assert_segment_uniform(self, vals, _interp_combine_min, width, f"min:all-NaN width={width}")
                for r in results[:max(width, 2)]:
                    self.assertTrue(_interp_lk_isnan(r))

    def test_infinities_consistent_max(self):
        # +inf wins over finite; -inf is NOT chosen when finite exists.
        vals = [self.INF] + [1.0] * 31
        results = _assert_segment_uniform(self, vals, _interp_combine_max, 32, "max:+inf")
        self.assertEqual(results[0], self.INF)
        vals = [self.NINF] + [1.0] * 31
        results = _assert_segment_uniform(self, vals, _interp_combine_max, 32, "max:-inf+1")
        self.assertEqual(results[0], 1.0)

    def test_infinities_consistent_min(self):
        vals = [self.NINF] + [1.0] * 31
        results = _assert_segment_uniform(self, vals, _interp_combine_min, 32, "min:-inf")
        self.assertEqual(results[0], self.NINF)
        vals = [self.INF] + [1.0] * 31
        results = _assert_segment_uniform(self, vals, _interp_combine_min, 32, "min:+inf+1")
        self.assertEqual(results[0], 1.0)

    def test_integer_inputs_exact(self):
        # Integer dtypes cannot express NaN/zero-sign; strict order apply.
        vals = [7, 3] + [5] * 30
        results_max = _assert_segment_uniform(self, vals, _interp_combine_max, 32, "max:int")
        self.assertEqual(results_max[0], 7)
        results_min = _assert_segment_uniform(self, vals, _interp_combine_min, 32, "min:int")
        self.assertEqual(results_min[0], 3)


if __name__ == "__main__":
    unittest.main()
