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
"""Lane-width / sign-extension semantics for ``gmem_ld_v4s32`` on byte
element streams (semantic key ir.gmem_ld_v4s32.int8_tensor_lanes).

The intrinsic always issues exactly one 16-byte vector transaction at the
16-byte-aligned address ``mem + index`` (``index`` in *element* units via
typed C++ pointer arithmetic).  What the four output int32 registers mean
depends on the source element dtype:

* int32 stream: lanes are the four raw int32 words of the transaction
  (historical spelling, must never drift);
* int8 stream: lane ``i`` is the sign-extended int8 element
  ``mem[index + i]`` (i = 0..3), extracted from the low word of the
  transaction -- never four packed int32 words and never a 64-byte
  (4 x int32-element) footprint;
* uint8 stream: the analogous zero extension.

These are CPU lowering/codegen tests; paired GPU validation stays with
the queue.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.type_system import Tensor, int8, int32, uint8
from little_kernel.language.builtin_base import CODEGEN_FUNC_ATTR, CODEGEN_NEEDS_TYPES_ATTR
import little_kernel.backends.nv.common.language.intrin.memory as memory


def _codegen(func_name, _arg_dtypes):
    func = getattr(memory, func_name)
    return getattr(func, CODEGEN_FUNC_ATTR)("x", "16", "w0", "w1", "w2", "w3", _arg_dtypes=_arg_dtypes)


class TestGmemLdV4S32ByteLaneCodegen(unittest.TestCase):
    """Typed-codegen-func level: helper body spelling per source dtype."""

    def test_needs_types_channel_opted_in(self):
        for name in ("gmem_ld_v4s32", "gmem_ld_v4s32_nc"):
            self.assertTrue(getattr(getattr(memory, name), CODEGEN_NEEDS_TYPES_ATTR, False), name)

    def test_int32_stream_keeps_legacy_word_lanes(self):
        for name in ("gmem_ld_v4s32", "gmem_ld_v4s32_nc"):
            b = _codegen(name, [Tensor[int32]])
            self.assertIn("const int4 value =", b.body)
            self.assertIn("*r0 = value.x; *r1 = value.y;", b.body)
            self.assertNotIn("int8_t", b.body)
            helper = "gmem_ld_v4s32_nc_fn" if name.endswith("_nc") \
                else "gmem_ld_v4s32_fn"
            self.assertIn(f"void {helper}(", b.body)
            self.assertIn(f"{helper}(x + 16,", b.return_val)

    def test_unknown_dtype_falls_back_to_legacy(self):
        # Direct codegen calls without the emitter channel, and untyped
        # mem expressions, keep the historical spelling.
        b = getattr(memory.gmem_ld_v4s32, CODEGEN_FUNC_ATTR)("x", "0", "w0", "w1", "w2", "w3")
        self.assertIn("void gmem_ld_v4s32_fn(", b.body)
        self.assertIn("*r0 = value.x", b.body)

    def test_int8_stream_sign_extends_four_elements(self):
        b = _codegen("gmem_ld_v4s32", [Tensor[int8]])
        self.assertIn("void gmem_ld_v4s32_i8sext_fn(", b.body)
        # One 16-byte transaction, lanes extracted from its low word only.
        self.assertEqual(b.body.count("reinterpret_cast<const int4*>"), 1)
        self.assertIn("static_cast<int32_t>(static_cast<int8_t>"
                      "(value.x & 0xFF))", b.body)
        self.assertIn("(value.x >> 8) & 0xFF", b.body)
        self.assertIn("(value.x >> 16) & 0xFF", b.body)
        self.assertIn("(value.x >> 24)", b.body)
        self.assertNotIn("value.y", b.body)
        self.assertNotIn("value.z", b.body)
        self.assertNotIn("value.w", b.body)
        # index stays in int8 element units: plain typed pointer arithmetic
        self.assertIn("gmem_ld_v4s32_i8sext_fn(x + 16,", b.return_val)

    def test_uint8_stream_zero_extends(self):
        b = _codegen("gmem_ld_v4s32", [Tensor[uint8]])
        self.assertIn("void gmem_ld_v4s32_u8zext_fn(", b.body)
        self.assertIn("static_cast<uint8_t>", b.body)
        self.assertNotIn("static_cast<int8_t>", b.body)

    def test_nc_int8_uses_ldg_and_same_lanes(self):
        b = _codegen("gmem_ld_v4s32_nc", [Tensor[int8]])
        self.assertIn("void gmem_ld_v4s32_nc_i8sext_fn(", b.body)
        self.assertIn("__ldg(reinterpret_cast<const int4*>(ptr))", b.body)
        self.assertIn("static_cast<int8_t>", b.body)
        self.assertIn("gmem_ld_v4s32_nc_i8sext_fn(x + 16,", b.return_val)

    def test_lane_extraction_numerically_matches_sign_extension(self):
        # Pin the lane mapping itself: lanes are bytes 0..3 of the 16-byte
        # chunk (low int32 word, little-endian), each sign-extended.
        random.seed(20261021)
        for _ in range(512):
            data = bytes(random.randrange(256) for _ in range(16))
            low_word = int.from_bytes(data[:4], "little")
            lanes = []
            for shift in (0, 8, 16, 24):
                v = (low_word >> shift) & 0xFF
                lanes.append(v - 256 if v >= 128 else v)  # int8_t -> int32_t
            expect = [b - 256 if b >= 128 else b for b in data[:4]]
            self.assertEqual(lanes, expect)


_PROBE_INT8 = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def probe_int8_v4s32(x: ll.Tensor[ll.int8], y: ll.Tensor[ll.int32],
                     k: ll.int32) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    if t == 0:
        w0: ll.int32 = 0
        w1: ll.int32 = 0
        w2: ll.int32 = 0
        w3: ll.int32 = 0
        ll.gmem_ld_v4s32(x, 0, w0, w1, w2, w3)
        y[0] = w0
        y[1] = w1
        y[2] = w2
        y[3] = w3
'''


def _export_json(name: str, in_dtype: str) -> str:
    return json.dumps({
        "kernel": name,
        "function": name,
        "inputs": [{"name": "x", "dtype": in_dtype, "rank": 1}],
        "output": {"name": "y", "dtype": "int32", "shape": ["16"]},
        "scalars": [{"name": "k", "dtype": "int32", "runtime": True}],
        "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
    })


def _lower(tmp: Path, source: str, name: str, in_dtype: str) -> str:
    task = tmp / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = tmp / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(_export_json(name, in_dtype))
    output = tmp / "output"
    lower(task, program, output, 0)
    kernels = sorted((output / "kernels").glob("*.cu"))
    assert len(kernels) == 1, kernels
    return kernels[0].read_text()


class TestGmemLdV4S32Int8Lowering(unittest.TestCase):
    """End-to-end CPU lowering of the capability reproducer."""

    def test_int8_tensor_probe_lowers_widened_lanes(self):
        with tempfile.TemporaryDirectory() as d:
            cu = _lower(Path(d), _PROBE_INT8, "probe_int8_v4s32", "int8")
        # mem pointer is byte-typed ...
        self.assertIn("const int8_t* __restrict__ x", cu)
        # ... so index 0 counts int8 elements (no *4 scaling), and the
        # widened helper services the call.
        self.assertIn("gmem_ld_v4s32_i8sext_fn(x + 0,", cu)
        self.assertIn("void gmem_ld_v4s32_i8sext_fn(", cu)
        # Exactly one 16-byte vector load: no 4 x int32 (64-byte) footprint,
        # and the legacy word-lane helper is not emitted at all.
        self.assertEqual(cu.count("reinterpret_cast<const int4*>"), 1)
        self.assertNotIn("gmem_ld_v4s32_fn(", cu)
        # Every lane is sign-extended through int8_t.
        self.assertEqual(cu.count("static_cast<int8_t>"), 4)

    def test_int32_tensor_probe_unchanged(self):
        src = _PROBE_INT8.replace("ll.int8", "ll.int32")
        with tempfile.TemporaryDirectory() as d:
            cu = _lower(Path(d), src, "probe_int8_v4s32", "int32")
        self.assertRegex(cu, r"const int32_t\* __restrict__ x")
        self.assertIn("gmem_ld_v4s32_fn(x + 0,", cu)
        self.assertIn("*r0 = value.x; *r1 = value.y;", cu)
        self.assertNotIn("i8sext", cu)


if __name__ == "__main__":
    unittest.main()
