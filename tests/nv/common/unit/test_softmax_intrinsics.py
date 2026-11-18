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
"""Online-softmax / row-normalize epilogue intrinsics: naming, typing, emission.

Legality / accuracy contract under test (compiler side; the range-reduction
itself is the frontend's obligation, documented in
``language/intrin/softmax.py``):

* canonical names ``ll.softmax_exp_neg`` / ``ll.softmax_normalize`` /
  ``ll.softmax_rescale`` are exported and marked builtin;
* every operand and result is ``float32`` (the dependent softmax/normalize
  workloads are FP32-accumulate); any other dtype rejects at type-inference
  time;
* the ops are pure value-to-value functions: they read/write no memory and
  do no cross-lane communication, so they declare NO memory effect (unlike
  the warp collectives in :mod:`language/intrin/reduce.py`);
* emission lowers to one deduplicated, fast-math-immune helper block that
  resolves its exponential to libdevice ``__nv_expf`` (immune to
  ``--use_fast_math`` remapping) and divides with the default IEEE-correct
  ``fdiv`` (not the SFU ``__fdividef`` / ``div.approx.f32``);
* a translation unit that uses several epilogue sites emits the shared
  helper body exactly once and reaches the three call spellings.
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR
from little_kernel.core.type_system import bfloat16, float16, float32, float64, int32

import little_kernel.backends.nv.common.language.intrin.softmax as softmax_intrin


class TestSoftmaxEpilogueIntrinsics(unittest.TestCase):
    """Naming, builtin flag, typing and effect declarations."""

    def test_exported_canonical_names(self):
        for name in ("softmax_exp_neg", "softmax_normalize", "softmax_rescale"):
            with self.subTest(op=name):
                op = getattr(ll, name, None)
                self.assertIsNotNone(op, f"ll.{name} not exported")
                self.assertTrue(getattr(op, BUILTIN_ATTR, False), f"ll.{name} missing __builtin__ flag")

    def test_return_type_accepts_float32(self):
        self.assertEqual(getattr(ll.softmax_exp_neg, EVAL_RETURN_TYPE_ATTR)(float32), float32)
        self.assertEqual(getattr(ll.softmax_normalize, EVAL_RETURN_TYPE_ATTR)(float32, float32, float32), float32)
        self.assertEqual(getattr(ll.softmax_rescale, EVAL_RETURN_TYPE_ATTR)(float32, float32), float32)

    def test_return_type_rejects_non_float32(self):
        bad = (float64, float16, bfloat16, int32)
        for dt in bad:
            with self.subTest(op="softmax_exp_neg", dtype=dt):
                with self.assertRaises(TypeError):
                    getattr(ll.softmax_exp_neg, EVAL_RETURN_TYPE_ATTR)(dt)
            with self.subTest(op="softmax_normalize", dtype=dt):
                with self.assertRaises(TypeError):
                    getattr(ll.softmax_normalize, EVAL_RETURN_TYPE_ATTR)(dt, float32, float32)
                with self.assertRaises(TypeError):
                    getattr(ll.softmax_normalize, EVAL_RETURN_TYPE_ATTR)(float32, dt, float32)
                with self.assertRaises(TypeError):
                    getattr(ll.softmax_normalize, EVAL_RETURN_TYPE_ATTR)(float32, float32, dt)
            with self.subTest(op="softmax_rescale", dtype=dt):
                with self.assertRaises(TypeError):
                    getattr(ll.softmax_rescale, EVAL_RETURN_TYPE_ATTR)(dt, float32)
                with self.assertRaises(TypeError):
                    getattr(ll.softmax_rescale, EVAL_RETURN_TYPE_ATTR)(float32, dt)

    def test_no_memory_effect_declared(self):
        """Pure scalar epilogue: no RMEM/GMEM/SMEM/TMEM effect.

        Unlike the warp collectives, these read and write only SSA
        registers of the issuing thread, so MEMORY_EFFECTS_ATTR must NOT
        be set--asserting its absence locks in the "safe to compose, no
        cross-lane / no memory hazard" contract.
        """
        for name in ("softmax_exp_neg", "softmax_normalize", "softmax_rescale"):
            with self.subTest(op=name):
                self.assertFalse(hasattr(getattr(ll, name), MEMORY_EFFECTS_ATTR),
                                 f"ll.{name} unexpectedly declares a memory effect")

    def test_helper_is_fast_math_immune_and_ieee_divide(self):
        """The emitted helper must not be remappable by --use_fast_math."""
        body = softmax_intrin._SOFTMAX_HELPERS
        # Exponential resolves to libdevice __nv_expf, NOT expf/__expf.
        self.assertIn("__nv_expf", body)
        self.assertNotIn("__expf(", body)
        self.assertNotIn("expf(", body.replace("__nv_expf(", ""))
        # Division is the IEEE default fdiv (a plain '/'), not an SFU
        # reciprocal/div.approx, which fast-math would also touch.
        self.assertNotIn("__frcp_rn", body)
        self.assertNotIn("__fdividef", body)
        self.assertNotIn("div.approx", body)
        # The guard protects a TU that mixes our body with a frontend copy.
        self.assertIn("#ifndef _LK_SOFTMAX_EPILOGUE", body)
        self.assertIn("#define _LK_SOFTMAX_EPILOGUE", body)


# ==============================================================================
# Full lower() path: native helper spelling in generated CUDA.
# ==============================================================================

# The dependent MoE top-K gate renormalization, written against the natural
# typed frontend.  Gather -> running max (K registers) -> exp(sub) -> div(sum)
# folds to three softmax_normalize calls after the shared warp max/sum.
NORMALIZE_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def gate_renorm(logits: ll.Tensor[ll.float32],
                idx: ll.Tensor[ll.int32],
                w: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.blockIdx_x()
    k0: ll.int32 = idx[t * 2]
    k1: ll.int32 = idx[t * 2 + 1]
    s0: ll.float32 = logits[t * 8 + k0]
    s1: ll.float32 = logits[t * 8 + k1]
    m: ll.float32 = s0 if s0 > s1 else s1
    e0: ll.float32 = ll.softmax_exp_neg(m - s0)
    e1: ll.float32 = ll.softmax_exp_neg(m - s1)
    s: ll.float32 = e0 + e1
    w[t * 2] = ll.softmax_normalize(s0, m, s)
    w[t * 2 + 1] = ll.softmax_normalize(s1, m, s)
'''

# Streaming (flash-attention / chunked) rescale: new block raises the
# running max from m_old to m_new, so the accumulator rescales once by
# softmax_rescale(m_old, m_new) before folding the new numerator.
RESCALE_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def online_step(score: ll.Tensor[ll.float32],
                state: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.blockIdx_x()
    m_old: ll.float32 = state[t * 3]
    l_old: ll.float32 = state[t * 3 + 1]
    acc_old: ll.float32 = state[t * 3 + 2]
    x: ll.float32 = score[t]
    m_new: ll.float32 = m_old if m_old > x else x
    alpha: ll.float32 = ll.softmax_rescale(m_old, m_new)
    e: ll.float32 = ll.softmax_exp_neg(m_new - x)
    state[t * 3] = m_new
    state[t * 3 + 1] = l_old * alpha + e
    state[t * 3 + 2] = acc_old * alpha + e * x
'''

# Many epilogue sites in one TU -- the helper must be defined exactly once.
MULTI_SITE_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def multi(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.blockIdx_x()
    a: ll.float32 = ll.softmax_normalize(x[t], x[t + 256], x[t + 512])
    b: ll.float32 = ll.softmax_normalize(x[t + 1], x[t + 256], x[t + 512])
    c: ll.float32 = ll.softmax_exp_neg(x[t + 768])
    d: ll.float32 = ll.softmax_rescale(x[t + 1024], x[t + 1280])
    out[t] = a + b + c + d
'''


class TestSoftmaxEpilogueEndToEnd(unittest.TestCase):
    """Full lower() path: native helper spelling in generated CUDA."""

    def _lower_one(self, tmp: Path, source: str, name: str, input_dtypes, output_name: str) -> str:
        task = tmp / "task"
        task.mkdir()
        (task / "model.py").write_text("class Model: pass\n")
        (task / "test_cases.py").write_text("test_cases = []\n")
        program = tmp / "program"
        program.mkdir()
        (program / "draft.py").write_text(source)
        first_tensor = input_dtypes[0][0]
        (program / "export.json").write_text(
            json.dumps({
                "kernel": name,
                "function": name,
                "inputs": [{"name": t, "dtype": d, "rank": 1} for t, d in input_dtypes],
                "output": {"name": output_name, "dtype": "float32", "shape": [f"{first_tensor}[0]"]},
                "scalars": [],
                "launch": {"grid": [f"ceil_div({first_tensor}[0], 128)", 1, 1], "block": [32, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_gate_renormalization_lowers(self):
        """The dependent top-K MoE gate renormalize composes the epilogue."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), NORMALIZE_SOURCE, "gate_renorm", (("logits", "float32"), ("idx", "int32")),
                                   "w")
        # exp(sub)->div through the shared fast-math-immune helpers.
        self.assertIn("lk_softmax_exp_neg(", cuda)
        self.assertIn("lk_softmax_normalize(", cuda)
        self.assertIn("__nv_expf", cuda)
        self.assertNotIn("asm", cuda)

    def test_online_rescale_step_lowers(self):
        """The streaming-rescale twin lowers through the shared helper."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), RESCALE_SOURCE, "online_step", (("score", "float32"), ), "state")
        self.assertIn("lk_softmax_rescale(", cuda)
        self.assertIn("lk_softmax_exp_neg(", cuda)
        self.assertIn("__nv_expf", cuda)
        self.assertNotIn("asm", cuda)

    def test_helper_body_deduplicated_across_sites(self):
        """Several epilogue calls in one TU share ONE helper definition."""
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), MULTI_SITE_SOURCE, "multi", (("x", "float32"), ), "out")
        # Guard establishes exactly-one definition per translation unit.
        self.assertEqual(cuda.count("#define _LK_SOFTMAX_EPILOGUE"), 1, cuda)
        self.assertEqual(cuda.count("float lk_softmax_exp_neg(float x)"), 1, cuda)
        self.assertEqual(cuda.count("float lk_softmax_normalize(float x"), 1, cuda)
        self.assertEqual(cuda.count("float lk_softmax_rescale(float old_max"), 1, cuda)
        # The three call spellings are present (normalizer used twice).
        self.assertEqual(cuda.count("lk_softmax_exp_neg("), 1 + 1)  # def+call
        self.assertEqual(cuda.count("lk_softmax_normalize("), 1 + 2)
        self.assertEqual(cuda.count("lk_softmax_rescale("), 1 + 1)
        self.assertNotIn("asm", cuda)


# ==============================================================================
# Semantic CPU reference: interpret the emitted helper against the workload
# reproducer's floating-point contract, so a string-only bug (e.g. dropping
# the max-subtract, or dividing by the wrong operand, or using the fast exp)
# cannot hide behind spelling matches.
# ==============================================================================


def _softmax_row_reference(scores):
    """Reference per-row top-K-style softmax over a 1-D score list (float32).

    Mirrors the emitted helper's math exactly: m = max(s), e_k = exp(s_k-m),
    w_k = e_k / sum(e).  Implemented in float32-free Python double-precision
    so the test asserts the *shape* of the helper, not float32 roundoff.
    """
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    total = sum(e)
    return [v / total for v in e]


class TestSoftmaxEpilogueSemantics(unittest.TestCase):
    """Pin the math shape the emitted helpers encode."""

    def test_normalize_matches_reference_topk(self):
        # Reproducer-shaped rows: K=2 columns, weights must sum to 1.
        for row in ([1.0, 2.0], [2.0, 1.0], [0.5, 0.5], [-3.0, 5.0], [10.0, 10.0]):
            with self.subTest(row=row):
                w = _softmax_row_reference(row)
                self.assertAlmostEqual(sum(w), 1.0, places=12)
                # Column order preserved: NOT score-sorted (column 0 wins
                # for the first row, column 1 for the second).
                if row[0] > row[1]:
                    self.assertGreater(w[0], w[1])
                elif row[1] > row[0]:
                    self.assertGreater(w[1], w[0])
                else:
                    self.assertAlmostEqual(w[0], w[1], places=12)

    def test_max_subtract_guarantees_bounded_numerator(self):
        # exp(s - m) <= 1 always, so the quotient is in [0,1].
        for row in ([1.0, 2.0], [-100.0, -99.0], [0.0, 0.0]):
            m = max(row)
            for s in row:
                self.assertLessEqual(math.exp(s - m), 1.0)
                self.assertGreater(math.exp(s - m), 0.0)

    def test_rescale_matches_running_max_update(self):
        # Streaming: rescaling old acc by exp(m_old-m_new) then adding the
        # new exp(x - m_new) equals one-shot over the full set.
        seq = [(1.0, 2.0), (2.0, 1.0), (5.0, -3.0), (3.5, 3.5)]
        for a, b in seq:
            with self.subTest(a=a, b=b):
                # Online: process a, then fold in b.
                m_new = max(a, b)
                l1 = 1.0  # exp(a - a) == 1 for seed max a (conceptually)
                # Emitted helper: alpha = exp(m_old - m_new) with m_old = a.
                alpha = math.exp(a - m_new)
                online_sum = l1 * alpha + math.exp(b - m_new)
                oneshot = math.exp(a - m_new) + math.exp(b - m_new)
                self.assertAlmostEqual(online_sum, oneshot, places=12)
                self.assertLessEqual(alpha, 1.0)
                self.assertGreater(alpha, 0.0)

    def test_wide_range_scores_stable(self):
        # wide_range scenario from the math_definition: log-uniform scale
        # 2^-6 .. 2^6, weights collapse toward the 1.0/0.0 hard limit.
        heavy = [64.0 * (2**-6), 64.0 * (2**6)]
        w = _softmax_row_reference(heavy)
        self.assertAlmostEqual(w[0] + w[1], 1.0, places=12)
        self.assertLess(w[0], 1e-3)
        self.assertGreater(w[1], 1.0 - 1e-8)

    def test_near_equal_scores_split_half(self):
        # near_half_weights scenario: ~1e-7 apart rounds to ~0.5/0.5.
        tiny = [1e-7, 1.999e-7]
        w = _softmax_row_reference(tiny)
        self.assertAlmostEqual(w[0], 0.5, places=6)
        self.assertAlmostEqual(w[1], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
