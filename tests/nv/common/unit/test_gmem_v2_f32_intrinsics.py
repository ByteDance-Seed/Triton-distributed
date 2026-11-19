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

import ast
import unittest

import little_kernel.language as ll
import little_kernel.backends.nv.common.language.intrin.memory as memory
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, Builtin
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    return codegen_cuda(ast.parse(source), ctx={"ll": ll}, emit_header=False)


class TestGmemV2F32Intrinsics(unittest.TestCase):
    """Tests for the typed FP32 8-byte (float2) GMEM load/store pair.

    ``gmem_st_v2_f32`` / ``gmem_ld_v2_f32`` are the typed-lane twins of the
    raw bit-pattern ``gmem_st_f32x2`` / ``gmem_ld_f32x2``: the lanes are C++
    ``float`` values, so FP32 float2 traffic needs no ``float_as_uint``
    bitcasts at the call site, and lane type mismatches are diagnosed at
    frontend compile time (mirroring the v4f32 load-side check).
    """

    NAMES = ("gmem_st_v2_f32", "gmem_ld_v2_f32")

    def test_are_builtins(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertTrue(getattr(getattr(memory, name), BUILTIN_ATTR, False))

    def test_exported_from_language(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                self.assertTrue(hasattr(ll, name))

    def test_st_v2_f32_codegen(self):
        codegen = getattr(memory.gmem_st_v2_f32, CODEGEN_FUNC_ATTR)
        result = codegen("gate_weights", "t * 2", "w0", "w1")
        self.assertIsInstance(result, Builtin)
        self.assertIn("*reinterpret_cast<float2*>(ptr) = make_float2(r0, r1);", result.body)
        self.assertNotIn("uint2", result.body)
        self.assertEqual(
            "gmem_st_v2_f32_fn(gate_weights + t * 2, w0, w1)",
            result.return_val,
        )

    def test_ld_v2_f32_codegen(self):
        codegen = getattr(memory.gmem_ld_v2_f32, CODEGEN_FUNC_ATTR)
        result = codegen("gate_weights", "t * 2", "w0", "w1")
        self.assertIsInstance(result, Builtin)
        self.assertIn("*reinterpret_cast<const float2*>(ptr)", result.body)
        self.assertEqual(
            "gmem_ld_v2_f32_fn(gate_weights + t * 2, &w0, &w1)",
            result.return_val,
        )

    def test_frontend_lowers_float2_store(self):
        """The exact reproducer shape: two FP32 lanes stored as one float2."""
        code = _emit("""
def kern(gate_weights: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    g0: ll.float32 = gate_weights[t * 2]
    g1: ll.float32 = gate_weights[t * 2 + 1]
    s: ll.float32 = g0 + g1
    w0: ll.float32 = g0 / s
    w1: ll.float32 = g1 / s
    ll.gmem_st_v2_f32(gate_weights, t * 2, w0, w1)
""")
        self.assertIn("gmem_st_v2_f32_fn(gate_weights + t * 2, w0, w1);", code)
        self.assertIn("*reinterpret_cast<float2*>(ptr) = make_float2(r0, r1);", code)
        self.assertNotIn("float_as_uint", code)

    def test_frontend_lowers_float2_load(self):
        code = _emit("""
def kern(gate_weights: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    w0: ll.float32 = 0.0
    w1: ll.float32 = 0.0
    ll.gmem_ld_v2_f32(gate_weights, t * 2, w0, w1)
    gate_weights[t * 2] = w0 + w1
""")
        self.assertIn("gmem_ld_v2_f32_fn(gate_weights + t * 2, &w0, &w1);", code)

    def test_store_rejects_non_float_lane(self):
        """A uint32 lane fed to the typed store is a frontend compile error."""
        from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
        with self.assertRaises(TypeInferenceError) as cm:
            _emit("""
def kern(out: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    u: ll.uint32 = 0
    v: ll.uint32 = 1
    ll.gmem_st_v2_f32(out, t * 2, u, v)
""")
        self.assertIn("gmem_st_v2_f32", str(cm.exception))
        self.assertIn("float32", str(cm.exception))

    def test_load_rejects_non_float_lane(self):
        from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
        with self.assertRaises(TypeInferenceError) as cm:
            _emit("""
def kern(out: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    u: ll.uint32 = 0
    v: ll.uint32 = 1
    ll.gmem_ld_v2_f32(out, t * 2, u, v)
""")
        self.assertIn("gmem_ld_v2_f32", str(cm.exception))

    def test_raw_f32x2_api_unchanged(self):
        """The raw uint32 bit-pattern API must stay bit-exact for its bit
        pattern use cases (e.g. float_as_uint plumbing)."""
        codegen = getattr(memory.gmem_st_f32x2, CODEGEN_FUNC_ATTR)
        result = codegen("out", "i", "a", "b")
        self.assertIn("*reinterpret_cast<uint2*>(ptr) = uint2{r0, r1};", result.body)
        code = _emit("""
def kern(out: ll.Tensor[ll.float32]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    f0: ll.float32 = 1.0
    f1: ll.float32 = 2.0
    ll.gmem_st_f32x2(out, t * 2, ll.float_as_uint(f0), ll.float_as_uint(f1))
""")
        self.assertIn("gmem_st_f32x2_fn(out + t * 2", code)


if __name__ == "__main__":
    unittest.main()
