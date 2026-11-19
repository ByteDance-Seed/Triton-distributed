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
"""By-reference output-lane type validation for the gmem_ld_* load family
(kernel-03 request R1: traced-scope name reuse fed an int32 local to a
``float*`` lane and surfaced only as an nvcc error).

Legality contract under test:

* every ``gmem_ld_*`` load writes through ``&lane`` into a typed C++
  pointer parameter, so an output-lane local must be declared with exactly
  the lane element type (v4f32/v4f32_nc -> float32, v4s32/v4s32_nc ->
  int32, v8f16*/v8bf16* -> float32, widened lanes; f32x4 -> uint32, the
  raw bit-pattern family -- float-lane value loads use gmem_vld_f32x4);
* a mismatched lane local of a statically known different type raises a
  ``TypeInferenceError`` naming the intrinsic, the lane, and both types at
  frontend compile time -- instead of an nvcc invalid-pointer-conversion
  error from generated CUDA;
* lane locals of the correct type lower unchanged (no codegen drift), and
  lanes that are not statically typeable (non-Name arguments or names not
  yet in the scope-type map) keep legacy behavior.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel as lk
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError


def _export_json(name: str) -> str:
    return json.dumps({
        "kernel": name,
        "function": name,
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
        "output": {"name": "out", "dtype": "float32", "shape": ["x[0]"]},
        "scalars": [],
        "launch": {"grid": ["ceil_div(x[0], 32)", 1, 1], "block": [32, 1, 1]},
    })


def _lower(tmp: Path, source: str, name: str) -> str:
    task = tmp / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = tmp / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(_export_json(name))
    output = tmp / "output"
    lower(task, program, output, 0)
    kernels = sorted((output / "kernels").glob("*.cu"))
    assert len(kernels) == 1, kernels
    return kernels[0].read_text()


R1_REPRODUCER = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def lane_mismatch(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    # phase 1: an int local named d
    d: ll.int32 = 1 << 2
    # phase 2: the same name is (erroneously) reused as a float lane temp
    ll.gmem_ld_v4f32_nc(x, i, d, d, d, d)
    out[i] = x[i] + d
'''

S32_LANE_MISMATCH = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def s32_lane_mismatch(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    a: ll.float32 = x[i]
    b: ll.float32 = x[i]
    c: ll.float32 = x[i]
    e: ll.float32 = x[i]
    ll.gmem_ld_v4s32_nc(x, i, a, b, c, e)
    out[i] = x[i] + a
'''

F32_LANES_OK = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f32_lanes_ok(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    step: ll.int32 = 1 << 2
    r0: ll.float32 = 0.0
    r1: ll.float32 = 0.0
    r2: ll.float32 = 0.0
    r3: ll.float32 = 0.0
    ll.gmem_ld_v4f32_nc(x, i, r0, r1, r2, r3)
    out[i] = r0 + r1 + r2 + r3 + step
'''

S32_LANES_OK = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def s32_lanes_ok(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 32 + ll.threadIdx_x()
    q0: ll.int32 = 0
    q1: ll.int32 = 0
    q2: ll.int32 = 0
    q3: ll.int32 = 0
    ll.gmem_ld_v4s32(x, i, q0, q1, q2, q3)
    out[i] = q0 + q1 + q2 + q3
'''

F32X4_U32_LANES_OK = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f32x4_u32_lanes_ok(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = (ll.blockIdx_x() * 32 + ll.threadIdx_x()) * 4
    q0: ll.uint32 = 0
    q1: ll.uint32 = 0
    q2: ll.uint32 = 0
    q3: ll.uint32 = 0
    ll.gmem_ld_f32x4(x, i, q0, q1, q2, q3)
    out[i] = ll.uint_as_float(q0)
'''

F32X4_F32_LANES_MISMATCH = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f32x4_f32_lanes_mismatch(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = (ll.blockIdx_x() * 32 + ll.threadIdx_x()) * 4
    r0: ll.float32 = 0.0
    ll.gmem_ld_f32x4(x, i, r0, r0, r0, r0)
    out[i] = r0
'''


class TestByRefLaneTypeValidation(unittest.TestCase):

    def test_r1_reproducer_raises_clean_diagnostic(self):
        """int32 local reused as a float lane temp -> compile-time error
        instead of an nvcc failure."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeInferenceError) as cm:
                _lower(Path(d), R1_REPRODUCER, "lane_mismatch")
        msg = str(cm.exception)
        self.assertIn("gmem_ld_v4f32_nc", msg)
        self.assertIn("'d'", msg)
        self.assertIn("int32", msg)
        self.assertIn("float32", msg)
        self.assertNotIn("Failed to compute return type", msg)

    def test_s32_intrinsic_rejects_float_lanes(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeInferenceError) as cm:
                _lower(Path(d), S32_LANE_MISMATCH, "s32_lane_mismatch")
        msg = str(cm.exception)
        self.assertIn("gmem_ld_v4s32_nc", msg)
        self.assertIn("int32", msg)
        self.assertIn("float32", msg)

    def test_correct_f32_lanes_lower_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = _lower(Path(d), F32_LANES_OK, "f32_lanes_ok")
        self.assertIn("gmem_ld_v4f32_nc_fn", cuda)
        self.assertIn("&r0", cuda)
        self.assertIn("&r3", cuda)
        # The merged compiler's f32-literal codegen emits an explicit `f`
        # suffix (pinned by tests/nv/common/unit/test_float_literal_codegen.py); the
        # lane-type contract this test guards is unchanged.
        self.assertIn("float r0 = 0.0f;", cuda)

    def test_correct_s32_lanes_lower_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            cuda = _lower(Path(d), S32_LANES_OK, "s32_lanes_ok")
        self.assertIn("gmem_ld_v4s32_fn", cuda)
        self.assertIn("&q0", cuda)
        self.assertIn("int32_t q0 = 0;", cuda)

    def test_f32x4_uint32_lanes_lower_unchanged(self):
        """f32x4 belongs to the raw bit-pattern family: its codegen writes
        through ``uint32_t*`` lane pointers (docstring "four raw uint32
        lanes; bit-level, dtype-agnostic", same as gmem_ld_cs_f32x4), so
        uint32 lanes are the legal Name-lane form."""
        with tempfile.TemporaryDirectory() as d:
            cuda = _lower(Path(d), F32X4_U32_LANES_OK, "f32x4_u32_lanes_ok")
        self.assertIn("gmem_ld_f32x4_fn", cuda)
        self.assertIn("&q0", cuda)
        self.assertIn("uint32_t q0", cuda)

    def test_f32x4_rejects_float_lanes_at_compile_time(self):
        """A float32 lane under f32x4 passed the stale check and failed only
        at nvcc (``float*`` into a ``uint32_t*`` parameter); it must now be
        the honest compile-time diagnostic. Float-lane value loads use
        ``gmem_vld_f32x4`` instead."""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeInferenceError) as cm:
                _lower(Path(d), F32X4_F32_LANES_MISMATCH, "f32x4_f32_lanes_mismatch")
        msg = str(cm.exception)
        self.assertIn("gmem_ld_f32x4", msg)
        self.assertIn("'r0'", msg)
        self.assertIn("float32", msg)
        self.assertIn("uint32", msg)


if __name__ == "__main__":
    unittest.main()
