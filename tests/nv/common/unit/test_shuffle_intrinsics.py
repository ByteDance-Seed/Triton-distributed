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
"""First-class warp shuffle intrinsics: naming, typing, effects, emission.

Legality contract under test (compiler side; convergence is a frontend
obligation documented in language/intrin/shuffle.py):

* canonical clean names ``ll.shfl_sync`` / ``ll.shfl_xor_sync`` /
  ``ll.shfl_up_sync`` / ``ll.shfl_down_sync`` plus the backward-compatible
  alias ``ll.__shfl_sync``;
* the returned type equals the shuffled value's type (f32 -> f32, s32 -> s32,
  f64 -> f64);
* each op declares the warp-collective register-read effect
  ``Effect(R_RMEM_WARP_SYNC, "value")`` via MEMORY_EFFECTS_ATTR -- the
  returned value may come from a different lane, so the op must not be
  reordered across warp sync / divergence boundaries;
* emission is the native CUDA intrinsic through the full lower() path, with
  no per-frontend asm/inline-CUDA escape hatch required.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.memory_effects import Access, IssuerSet, MemorySpace, R_RMEM_WARP_SYNC
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, MEMORY_EFFECTS_ATTR, Builtin
import little_kernel.backends.nv.common.language.intrin.shuffle as shuffle


class TestShuffleIntrinsics(unittest.TestCase):

    def test_exported_canonical_names(self):
        for name in ("shfl_sync", "shfl_xor_sync", "shfl_up_sync", "shfl_down_sync"):
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name), name)
                self.assertTrue(getattr(getattr(ll, name), BUILTIN_ATTR, False))

    def test_compat_alias_identity(self):
        """ll.__shfl_sync is the same builtin object as ll.shfl_sync."""
        self.assertIs(getattr(ll, "__shfl_sync"), ll.shfl_sync)

    def test_return_type_equals_value_type(self):
        cases = (
            (shuffle.shfl_sync, ("mask", "value", "src")),
            (shuffle.shfl_xor_sync, ("mask", "value", "lane_mask")),
            (shuffle.shfl_up_sync, ("mask", "value", "delta")),
            (shuffle.shfl_down_sync, ("mask", "value", "delta")),
        )
        for fn, _arg_names in cases:
            ret = getattr(fn, EVAL_RETURN_TYPE_ATTR)
            with self.subTest(name=fn.__name__, dtype="float32"):
                self.assertEqual(ret(ll.uint32, ll.float32, ll.int32), ll.float32)
            with self.subTest(name=fn.__name__, dtype="int32"):
                self.assertEqual(ret(ll.uint32, ll.int32, ll.int32), ll.int32)
            with self.subTest(name=fn.__name__, dtype="float64"):
                self.assertEqual(ret(ll.uint32, ll.float64, ll.int32), ll.float64)

    def test_declared_warp_collective_effect(self):
        for fn in (shuffle.shfl_sync, shuffle.shfl_xor_sync, shuffle.shfl_up_sync, shuffle.shfl_down_sync):
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
        cases = (
            (shuffle.shfl_sync, ("m", "v", "8"), "__shfl_sync(m, v, 8)"),
            (shuffle.shfl_xor_sync, ("m", "v", "16"), "__shfl_xor_sync(m, v, 16)"),
            (shuffle.shfl_up_sync, ("m", "v", "1"), "__shfl_up_sync(m, v, 1)"),
            (shuffle.shfl_down_sync, ("m", "v", "2"), "__shfl_down_sync(m, v, 2)"),
        )
        for fn, args, expected in cases:
            with self.subTest(name=fn.__name__):
                out = getattr(fn, CODEGEN_FUNC_ATTR)(*args)
                self.assertIsInstance(out, Builtin)
                self.assertEqual(out.body, "")
                self.assertEqual(out.includes, [])
                self.assertEqual(out.return_val, expected)


WARP_REDUCE_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def warp_sum(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    v: ll.float32 = x[i]
    v = v + ll.shfl_xor_sync(0xffffffff, v, 16)
    v = v + ll.shfl_xor_sync(0xffffffff, v, 8)
    v = v + ll.shfl_xor_sync(0xffffffff, v, 4)
    v = v + ll.shfl_xor_sync(0xffffffff, v, 2)
    v = v + ll.shfl_xor_sync(0xffffffff, v, 1)
    lane0: ll.int32 = ll.blockIdx_x() * 32
    if ll.threadIdx_x() == 0:
        out[ll.blockIdx_x()] = v
'''

PARTNER_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def partner(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 4096:
        v: ll.float32 = x[i]
        pv: ll.float32 = ll.shfl_sync(0xffffffff, v, 7)
        out[i] = v + pv
'''


class TestShuffleEndToEnd(unittest.TestCase):
    """Full lower() path: native intrinsic spelling in generated CUDA."""

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
                "kernel": name,
                "function": name,
                "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
                "output": {"name": "out", "dtype": "float32", "shape": ["x[0]"]},
                "scalars": [],
                "launch": {"grid": ["ceil_div(x[0], 32)", 1, 1], "block": [32, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_warp_reduce_emits_native_xors(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), WARP_REDUCE_SOURCE, "warp_sum")
        for lane_mask in ("16", "8", "4", "2", "1"):
            self.assertIn("__shfl_xor_sync(", cuda)
        self.assertEqual(cuda.count("__shfl_xor_sync("), 5)
        self.assertNotIn("asm", cuda)

    def test_indexed_shuffle_emits_native_intrinsic(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), PARTNER_SOURCE, "partner")
        self.assertIn("__shfl_sync(", cuda)
        self.assertEqual(cuda.count("__shfl_sync("), 1)


if __name__ == "__main__":
    unittest.main()
