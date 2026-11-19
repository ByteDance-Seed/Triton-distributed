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
"""Exact int8->int32 dot accumulate intrinsic: ``ll.dp4a_s8u8_s32``.

The capability question was whether ``ll.viaddmin_s32(a, w, acc)`` is a
``__dp4a``-style a*w+acc accumulator.  It is not: it emits
``__viaddmin_s32`` == ``min(a + b, c)``, so using it as a running dot
accumulator silently clamps every step by the accumulator.  These tests
pin the dedicated replacement:

* canonical name ``ll.dp4a_s8u8_s32`` is exported and marked builtin;
* return type is always ``int32``;
* codegen emits CUDA ``__dp4a`` (PTX ``dp4a.s32.s8.u8``) with explicit
  casts selecting the signed-lane / unsigned-lane overload;
* the constant folder computes the exact four signed-int8 x unsigned-int8
  products plus the accumulator, with int32 wraparound and **no**
  min/clamp -- including a case where viaddmin's clamp would diverge;
* the full lower() path emits ``__dp4a`` in the .cu source (CPU check;
  paired GPU numeric validation is left to the queue).
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.type_system import int32, uint32
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR, Builtin
from little_kernel.backends.nv.common.language.intrin.arith import dp4a_s8u8_s32


def _reference_dp4a(a, b, c):
    a &= 0xFFFFFFFF
    b &= 0xFFFFFFFF
    dot = 0
    for i in range(4):
        sa = (a >> (8 * i)) & 0xFF
        if sa >= 0x80:
            sa -= 0x100
        ub = (b >> (8 * i)) & 0xFF
        dot += sa * ub
    r = (dot + c) & 0xFFFFFFFF
    return r - (1 << 32) if r & 0x80000000 else r


class TestDp4aIntrinsic(unittest.TestCase):

    def test_exported_canonical_name(self):
        self.assertTrue(hasattr(ll, "dp4a_s8u8_s32"))
        self.assertIs(ll.dp4a_s8u8_s32, dp4a_s8u8_s32)
        self.assertTrue(getattr(dp4a_s8u8_s32, BUILTIN_ATTR, False))

    def test_return_type_is_int32(self):
        # Packed lanes may ride in signed or unsigned 32-bit containers;
        # the accumulated dot is always signed int32.
        ret = getattr(dp4a_s8u8_s32, EVAL_RETURN_TYPE_ATTR)
        for dtypes in ((int32, int32, int32), (int32, uint32, int32), (uint32, uint32, uint32)):
            with self.subTest(dtypes=dtypes):
                resolved = ret(*dtypes) if callable(ret) else ret
                self.assertIs(resolved, int32)

    def test_codegen_spelling_pins_s8u8_overload(self):
        out = getattr(dp4a_s8u8_s32, CODEGEN_FUNC_ATTR)("a", "w", "acc")
        self.assertIsInstance(out, Builtin)
        self.assertEqual(out.includes, [])
        # Explicit casts pin int __dp4a(int, unsigned int, int), i.e.
        # dp4a.s32.s8.u8: first operand signed int8 lanes, second unsigned.
        self.assertIn("__dp4a(", out.return_val)
        self.assertIn("static_cast<int>(a)", out.return_val)
        self.assertIn("static_cast<unsigned int>(w)", out.return_val)
        self.assertIn("static_cast<int>(acc)", out.return_val)
        # Must not be misspelled as the min-clamp video intrinsic.
        self.assertNotIn("viaddmin", out.return_val)

    def test_constant_fold_matches_reference(self):
        cases = [
            # lanes [4,3,2,1] (signed) x [1,1,1,1] -> 10
            (0x01020304, 0x01010101, 0),
            # all-negative signed lanes [-1,-1,-1,-1] x ones -> -4
            (0xFFFFFFFF, 0x01010101, 0),
            # mixed signed lanes [0, -1, 127, -128] x [0, 255, 2, 1]
            (0x807FFF00, 0x0102FF00, 7),
            # running accumulator: add onto a non-zero base
            (0x01020304, 0x01010101, 100000),
        ]
        for a, b, c in cases:
            with self.subTest(a=a, b=b, c=c):
                self.assertEqual(dp4a_s8u8_s32(a, b, c), _reference_dp4a(a, b, c))

    def test_accumulate_is_exact_no_min_clamp(self):
        # dot = 1*1 + 2*1 + 3*1 + 4*1 = 10; accumulator c = 3.
        # An exact accumulator returns 13.  viaddmin's min(a+b, c)
        # contract would clamp to min(10, 3) = 3 and corrupt the dot.
        self.assertEqual(dp4a_s8u8_s32(0x01020304, 0x01010101, 3), 13)
        self.assertEqual(ll.viaddmin_s32(10, 0, 3), 3)  # documents the gap

    def test_chained_accumulation_is_an_exact_running_sum(self):
        # Two packed pairs folded the way a dot-product loop would chain
        # them must equal the direct sum over all eight lane products.
        a1, b1 = 0x01020304, 0x05060708
        a2, b2 = 0xFFFFFFFF, 0x01010101  # [-1]*4 x [1]*4
        step1 = dp4a_s8u8_s32(a1, b1, 0)
        step2 = dp4a_s8u8_s32(a2, b2, step1)
        direct = (_reference_dp4a(a1, b1, 0) + _reference_dp4a(a2, b2, 0))
        self.assertEqual(step2, direct)

    def test_result_wraps_as_signed_int32(self):
        # 127*255*4 = 129540, added to INT32_MAX wraps into the negatives.
        a = 0x7F7F7F7F  # signed lanes [127]*4
        b = 0xFFFFFFFF  # unsigned lanes [255]*4
        c = 0x7FFFFFFF
        self.assertEqual(dp4a_s8u8_s32(a, b, c), _reference_dp4a(a, b, c))
        self.assertLess(dp4a_s8u8_s32(a, b, c), 0)


DP4A_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def probe_dp4a(x: ll.Tensor[ll.int32], out: ll.Tensor[ll.int32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    if t == 0:
        a: ll.int32 = x[0]
        w: ll.int32 = x[1]
        acc: ll.int32 = x[2]
        out[0] = ll.dp4a_s8u8_s32(a, w, acc)
'''


class TestDp4aEndToEnd(unittest.TestCase):
    """Full lower() path: the intrinsic reaches .cu codegen."""

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
                "inputs": [{"name": "x", "dtype": "int32", "rank": 1}],
                "output": {"name": "out", "dtype": "int32", "shape": ["x[0]"]},
                "scalars": [],
                "launch": {"grid": [1, 1, 1], "block": [1, 1, 1]},
            }))
        output = tmp / "output"
        lower(task, program, output, 0)
        kernels = sorted((output / "kernels").glob("*.cu"))
        self.assertEqual(len(kernels), 1, kernels)
        return kernels[0].read_text()

    def test_dp4a_lowers_to_cuda_builtin(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = self._lower_one(Path(d), DP4A_SOURCE, "probe_dp4a")
        self.assertIn("__dp4a(", cuda)
        # Exact dot must not route through the min-clamp video intrinsic.
        self.assertNotIn("__viaddmin_s32", cuda)


if __name__ == "__main__":
    unittest.main()
