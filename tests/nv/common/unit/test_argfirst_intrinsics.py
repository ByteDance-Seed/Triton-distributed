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
"""First-class warp first-index intrinsics: naming, typing, effects, emission.

These cover the "masked indexing" and "synchronization" axes of the
scan_indexing coverage audit:

* masked indexing     --> ``warp_first_set_lane`` / ``warp_last_set_lane``
  (ballot + ffs/clz, the natural primitive whenever the mask is itself the
  result of a per-lane guard);
* synchronization     --> ``warp_argmax_lane`` / ``warp_argmin_lane``
  (a butterfly that must not be CSE'd or reordered across divergence or
  barrier boundaries, expressed via the R_RMEM_WARP_SYNC effect).

Legality contract under test (compiler side; convergence is a frontend
obligation documented in language/intrin/argfirst.py):

* canonical names ``ll.warp_argmax_lane`` / ``ll.warp_argmin_lane`` /
  ``ll.warp_first_set_lane`` / ``ll.warp_last_set_lane`` are exported and
  marked builtin;
* the return type is always ``int32`` (a lane id or -1), independent of
  the value/mask dtype;
* each op declares the warp-collective register-read effect on its data
  operand (``value`` for arg*, ``predicate`` for set-lane*) via
  MEMORY_EFFECTS_ATTR, so the scheduler keeps it across barriers and
  divergent branches;
* codegen emits one self-contained templated or untemplated CUDA helper per
  kernel module, deduplicated by the builtin-body cache, with:
    - a 5-step ``__shfl_xor_sync`` butterfly over (value, lane) for arg*,
    - a strict-extremum update plus an exact-tie ``other_lane < lane``
      comparison so first-index-wins is structurally preserved
      (rejecting any regression to a plain greater-only update),
    - the ISA lane index ``%laneid``, never the ``threadIdx.x & 31``
      idiom that is wrong under multi-dimensional blocks;
    - a single ``__ballot_sync`` for set-lane*, with the result mapped
      through ``__ffs`` (first) or ``31 - __clz`` (last), plus an
      explicit all-false -> -1 guard on the last-set helper;
* a gathered per-lane value can seed ``warp_argmax_lane`` in one
  expression (generic indirect indexing composes through the same path).
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.memory_effects import Access, IssuerSet, MemorySpace, R_RMEM_WARP_SYNC
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.argfirst as argfirst

_VALUE_OPS = (
    (argfirst.warp_argmax_lane, "warp_argmax_lane_fn"),
    (argfirst.warp_argmin_lane, "warp_argmin_lane_fn"),
)
_PRED_OPS = (
    (argfirst.warp_first_set_lane, "warp_first_set_lane_fn"),
    (argfirst.warp_last_set_lane, "warp_last_set_lane_fn"),
)


class TestWarpArgFirstIntrinsics(unittest.TestCase):

    def test_exported_canonical_names(self):
        for name in ("warp_argmax_lane", "warp_argmin_lane", "warp_first_set_lane", "warp_last_set_lane"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name), name)
                self.assertTrue(getattr(getattr(ll, name), BUILTIN_ATTR, False))

    def test_return_type_is_int32_for_any_value_dtype(self):
        # Arg* index the value's home lane, not the value itself: the
        # return type is independent of the value dtype so int32/float32/
        # float64 all land on int32.
        for fn, _helper in _VALUE_OPS:
            ret = getattr(fn, EVAL_RETURN_TYPE_ATTR)
            for value_dtype in (ll.float32, ll.float64, ll.int32, ll.uint32, ll.int64):
                with self.subTest(name=fn.__name__, dtype=value_dtype):
                    self.assertEqual(ret(ll.uint32, value_dtype), ll.int32)
        for fn, _helper in _PRED_OPS:
            ret = getattr(fn, EVAL_RETURN_TYPE_ATTR)
            for pred_dtype in (ll.int32, ll.uint32):
                with self.subTest(name=fn.__name__, dtype=pred_dtype):
                    self.assertEqual(ret(ll.uint32, pred_dtype), ll.int32)

    def test_declared_warp_collective_effect(self):
        # Arg* read the value operand across lanes.
        for fn, _helper in _VALUE_OPS:
            with self.subTest(name=fn.__name__):
                effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertIs(eff.kind, R_RMEM_WARP_SYNC)
                self.assertIs(eff.kind.access, Access.R)
                self.assertIs(eff.kind.space, MemorySpace.RMEM)
                self.assertIs(eff.kind.issuer, IssuerSet.WARP)
                self.assertEqual(eff.operand, "value")
        # Set-lane* read the predicate operand across lanes.
        for fn, _helper in _PRED_OPS:
            with self.subTest(name=fn.__name__):
                effects = getattr(fn, MEMORY_EFFECTS_ATTR, ())
                self.assertEqual(len(effects), 1)
                eff = effects[0]
                self.assertIs(eff.kind, R_RMEM_WARP_SYNC)
                self.assertIs(eff.kind.access, Access.R)
                self.assertIs(eff.kind.space, MemorySpace.RMEM)
                self.assertIs(eff.kind.issuer, IssuerSet.WARP)
                self.assertEqual(eff.operand, "predicate")

    def test_codegen_argmax_spelling_first_index_wins(self):
        out = getattr(argfirst.warp_argmax_lane, CODEGEN_FUNC_ATTR)("m", "v")
        self.assertIsInstance(out, Builtin)
        self.assertEqual(out.includes, [])
        self.assertEqual(out.return_val, "warp_argmax_lane_fn(m, (v))")
        # Return type is int (lane id); only the value parameter is templated.
        self.assertIn("template <typename T>", out.body)
        self.assertIn("int warp_argmax_lane_fn(", out.body)
        # Rolled butterfly body: the value- and lane-shuffles appear once
        # each inside the `for lk_off_ in 16..1` loop.
        self.assertEqual(out.body.count("__shfl_xor_sync("), 2)
        # First-index-wins tie-break: strict maximum OR exact tie with the
        # smaller lane id -- a plain strict-greater would drop the tie rule
        # and break first-index-wins registration.
        self.assertIn("lk_other_v_ > lk_best_", out.body)
        self.assertIn("lk_other_v_ == lk_best_", out.body)
        self.assertIn("lk_other_l_ < lk_lane_", out.body)
        # Lane id from the ISA register, never threadIdx.x % 32.
        self.assertIn("%laneid", out.body)

    def test_codegen_argmin_spelling_first_index_wins(self):
        out = getattr(argfirst.warp_argmin_lane, CODEGEN_FUNC_ATTR)("m", "v")
        self.assertEqual(out.return_val, "warp_argmin_lane_fn(m, (v))")
        self.assertIn("int warp_argmin_lane_fn(", out.body)
        self.assertEqual(out.body.count("__shfl_xor_sync("), 2)
        self.assertIn("lk_other_v_ < lk_best_", out.body)
        self.assertIn("lk_other_v_ == lk_best_", out.body)
        self.assertIn("lk_other_l_ < lk_lane_", out.body)
        self.assertIn("%laneid", out.body)

    def test_codegen_first_set_lane_spelling(self):
        out = getattr(argfirst.warp_first_set_lane, CODEGEN_FUNC_ATTR)("m", "p")
        self.assertIsInstance(out, Builtin)
        self.assertEqual(out.includes, [])
        self.assertEqual(out.return_val, "warp_first_set_lane_fn(m, (p))")
        # One ballot + __ffs -> 1-based index shifted to a lane id or -1.
        self.assertIn("__ballot_sync(lk_mask_, lk_pred_)", out.body)
        self.assertIn("__ffs(lk_ballot_) - 1", out.body)

    def test_codegen_last_set_lane_spelling(self):
        out = getattr(argfirst.warp_last_set_lane, CODEGEN_FUNC_ATTR)("m", "p")
        self.assertEqual(out.return_val, "warp_last_set_lane_fn(m, (p))")
        self.assertIn("__ballot_sync(lk_mask_, lk_pred_)", out.body)
        # Highest set lane via 31 - clz, and an explicit -1 on empty ballot
        # (CUDA's __clz(0)==32 would otherwise alias to -1 by accident --
        # the explicit guard pins the contract).
        self.assertIn("__clz(lk_ballot_)", out.body)
        self.assertIn("lk_ballot_ == 0u ? -1 :", out.body)


ARGFIRST_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def first_index(x: ll.Tensor[ll.int32], idx: ll.Tensor[ll.int32],
                out: ll.Tensor[ll.int32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    # Indirect (gathered) indexing -- the masked-indexing axis.
    v: ll.int32 = x[idx[i]]
    rel_mask: ll.int32 = v
    lane_argmax: ll.int32 = ll.warp_argmax_lane(0xffffffff, v)
    lane_argmax2: ll.int32 = ll.warp_argmax_lane(0xffffffff, v)
    lane_argmin: ll.int32 = ll.warp_argmin_lane(0xffffffff, v)
    lane_first: ll.int32 = ll.warp_first_set_lane(0xffffffff, rel_mask)
    lane_last: ll.int32 = ll.warp_last_set_lane(0xffffffff, rel_mask)
    out[i] = lane_argmax + lane_argmax2 + lane_argmin + lane_first + lane_last
'''


class TestWarpArgFirstEndToEnd(unittest.TestCase):
    """Full lower() path: templated + untemplated helpers spelling."""

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

    def test_gather_argfirst_lowers_to_shared_helpers(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), ARGFIRST_SOURCE, "first_index")
        # Each builtin helper emitted exactly once even for repeated use of
        # warp_argmax_lane.  The helper signature reads
        # ``int warp_argmax_lane_fn(const unsigned int lk_mask_, const T)``,
        # so count the definitional " int NAME_fn(" form (not the call site).
        self.assertEqual(cuda.count("int warp_argmax_lane_fn("), 1)
        self.assertEqual(cuda.count("warp_argmax_lane_fn("), 3)
        self.assertEqual(cuda.count("int warp_argmin_lane_fn("), 1)
        # Two untemplated ballot helpers share one body region.
        self.assertIn("warp_first_set_lane_fn(", cuda)
        self.assertIn("warp_last_set_lane_fn(", cuda)
        self.assertIn("__ballot_sync(", cuda)
        self.assertIn("__ffs(", cuda)
        self.assertIn("__clz(", cuda)
        # Native convergent collectives, one %laneid read for the arg*
        # helpers (the ballot helpers do not need one).
        self.assertIn("__shfl_xor_sync(", cuda)
        self.assertIn("%laneid", cuda)
        # All-false-ballot -> -1 guard intact.
        self.assertIn("lk_ballot_ == 0u ? -1 :", cuda)


if __name__ == "__main__":
    unittest.main()
