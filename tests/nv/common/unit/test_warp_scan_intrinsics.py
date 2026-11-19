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
"""First-class warp prefix-scan intrinsics: naming, typing, effects, emission.

These cover the "generic scan" axis of the scan/indexing coverage audit:
before ll.warp_scan_* only the raw shfl_up_sync building block existed and
every kernel hand-rolled (and frequently mis-guarded) its Hillis-Steele loop.

Legality contract under test (compiler side; convergence is a frontend
obligation documented in language/intrin/scan.py):

* canonical clean names ``ll.warp_scan_inclusive_add`` /
  ``ll.warp_scan_exclusive_add`` / ``ll.warp_scan_inclusive_max`` /
  ``ll.warp_scan_inclusive_min``;
* the returned type equals the scanned value's type (f32 -> f32, s32 -> s32,
  f64 -> f64) independent of the mask type;
* each op declares the warp-collective register-read effect
  ``Effect(R_RMEM_WARP_SYNC, "value")`` via MEMORY_EFFECTS_ATTR;
* emission is a dtype-agnostic templated helper built on native
  ``__shfl_up_sync`` through the full lower() path, with the ISA lane index
  (``%laneid``) for the segment-gated Hillis-Steele guard so the scan is
  correct under multi-dimensional block layouts and under partial
  participation masks (each lane accumulates only within its own
  contiguous mask segment);
* helper bodies are emitted exactly once per module even with several call
  sites, and the exclusive helper is self-contained (no forward reference
  to the inclusive helper, which may not be emitted at all);
* generic indirect (gathered) indexing lowers through the same path, so a
  gathered per-lane value can seed a warp scan in one expression.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.memory_effects import Access, IssuerSet, MemorySpace, R_RMEM_WARP_SYNC
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.scan as scan

_SCAN_OPS = (
    (scan.warp_scan_inclusive_add, "warp_scan_inclusive_add_fn"),
    (scan.warp_scan_exclusive_add, "warp_scan_exclusive_add_fn"),
    (scan.warp_scan_inclusive_max, "warp_scan_inclusive_max_fn"),
    (scan.warp_scan_inclusive_min, "warp_scan_inclusive_min_fn"),
)


class TestWarpScanIntrinsics(unittest.TestCase):

    def test_exported_canonical_names(self):
        for name in ("warp_scan_inclusive_add", "warp_scan_exclusive_add", "warp_scan_inclusive_max",
                     "warp_scan_inclusive_min"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name), name)
                self.assertTrue(getattr(getattr(ll, name), BUILTIN_ATTR, False))

    def test_return_type_equals_value_type(self):
        for fn, _helper in _SCAN_OPS:
            ret = getattr(fn, EVAL_RETURN_TYPE_ATTR)
            with self.subTest(name=fn.__name__, dtype="float32"):
                self.assertEqual(ret(ll.uint32, ll.float32), ll.float32)
            with self.subTest(name=fn.__name__, dtype="int32"):
                self.assertEqual(ret(ll.uint32, ll.int32), ll.int32)
            with self.subTest(name=fn.__name__, dtype="float64"):
                self.assertEqual(ret(ll.uint32, ll.float64), ll.float64)

    def test_declared_warp_collective_effect(self):
        for fn, _helper in _SCAN_OPS:
            with self.subTest(name=fn.__name__):
                effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertIs(eff.kind, R_RMEM_WARP_SYNC)
                self.assertIs(eff.kind.access, Access.R)
                self.assertIs(eff.kind.space, MemorySpace.RMEM)
                self.assertIs(eff.kind.issuer, IssuerSet.WARP)
                self.assertEqual(eff.operand, "value")

    def test_codegen_spelling(self):
        for fn, helper in _SCAN_OPS:
            with self.subTest(name=fn.__name__):
                out = getattr(fn, CODEGEN_FUNC_ATTR)("m", "v")
                self.assertIsInstance(out, Builtin)
                self.assertEqual(out.includes, [])
                self.assertEqual(out.return_val, f"{helper}(m, (v))")
                self.assertIn(f"T {helper}(", out.body)
                self.assertIn("__shfl_up_sync(", out.body)
                # Segment-gated Hillis-Steele: the accumulation is gated on
                # the lane's in-segment distance (laneid - segment base),
                # using the ISA lane index, never a threadIdx.x % 32 idiom
                # that breaks under multi-D blocks.  The base read stays on
                # the zero-cost path for the canonical full-warp mask.
                self.assertIn("%laneid", out.body)
                self.assertIn("lk_laneid - lk_base >= static_cast<unsigned int>(lk_delta)", out.body)
                self.assertIn("if (lk_mask != 0xffffffffu)", out.body)
                self.assertNotIn("if (lk_laneid >= static_cast<unsigned int>(lk_delta))", out.body)

    def test_exclusive_add_helper_is_self_contained(self):
        """Exclusive add must not forward-reference the inclusive helper."""
        out = getattr(scan.warp_scan_exclusive_add, CODEGEN_FUNC_ATTR)("m", "v")
        self.assertNotIn("warp_scan_inclusive_add_fn(", out.body)
        # Shift-by-one seed plus identity re-seed of the segment base lane --
        # under a partial mask the first participating lane is the segment
        # base, not absolute lane 0.
        self.assertIn("__shfl_up_sync(lk_mask, lk_acc, 1)", out.body)
        self.assertIn("lk_laneid == lk_base", out.body)
        self.assertNotIn("lk_laneid == 0u", out.body)


GATHER_SCAN_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def gather_scan(x: ll.Tensor[ll.int32], idx: ll.Tensor[ll.int32],
                out: ll.Tensor[ll.int32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    # Indirect (gathered) indexing: the load address is itself a loaded
    # scalar -- the masked-indexing axis of the audit.
    v: ll.int32 = x[idx[i]]
    incl: ll.int32 = ll.warp_scan_inclusive_add(0xffffffff, v)
    incl = incl + ll.warp_scan_inclusive_add(0xffffffff, v)
    excl: ll.int32 = ll.warp_scan_exclusive_add(0xffffffff, v)
    hi: ll.int32 = ll.warp_scan_inclusive_max(0xffffffff, v)
    lo: ll.int32 = ll.warp_scan_inclusive_min(0xffffffff, v)
    out[i] = incl + excl + hi + lo
'''


class TestWarpScanEndToEnd(unittest.TestCase):
    """Full lower() path: templated helper + native shuffle spelling."""

    def _lower_one(self, tmp: Path, source: str, name: str) -> str:
        task = tmp / "task"
        task.mkdir()
        (task / "model.py").write_text("class Model: pass\n")
        (task / "test_cases.py").write_text("test_cases = []\n")
        program = tmp / "program"
        program.mkdir()
        (program / "draft.py").write_text(source)
        (program / "export.json").write_text(
            json.dumps({
                "kernel":
                name,
                "function":
                name,
                "inputs": [{"name": "x", "dtype": "int32", "rank": 1}, {"name": "idx", "dtype": "int32", "rank": 1}],
                "output": {"name": "out", "dtype": "int32", "shape": ["x[0]"]},
                "scalars": [],
                "launch": {"grid": ["ceil_div(x[0], 32)", 1, 1], "block": [32, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_gather_scan_emits_helpers_and_shuffles(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), GATHER_SCAN_SOURCE, "gather_scan")
        # Each helper body is emitted exactly once even for two call sites
        # of warp_scan_inclusive_add.
        self.assertEqual(cuda.count("T warp_scan_inclusive_add_fn("), 1)
        self.assertEqual(cuda.count("warp_scan_inclusive_add_fn("), 3)
        self.assertEqual(cuda.count("T warp_scan_exclusive_add_fn("), 1)
        self.assertEqual(cuda.count("T warp_scan_inclusive_max_fn("), 1)
        self.assertEqual(cuda.count("T warp_scan_inclusive_min_fn("), 1)
        # Native convergent collective underneath; no per-frontend asm is
        # required from the kernel author (the %laneid read is emitted by
        # the compiler inside the helper).
        self.assertIn("__shfl_up_sync(", cuda)
        self.assertIn("%laneid", cuda)
        # Exclusive scan shift + identity seed.
        self.assertIn("__shfl_up_sync(lk_mask, lk_acc, 1)", cuda)
        self.assertIn("lk_acc = T{};", cuda)


if __name__ == "__main__":
    unittest.main()
