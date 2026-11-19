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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Regressions for 16-bit-float val_cast/to/type-constructor lowering.

Torch CUDA extension builds define __CUDA_NO_HALF_CONVERSIONS__ and
__CUDA_NO_BFLOAT16_CONVERSIONS__, deleting the converting constructors and
conversion operators of __half/__nv_bfloat16.  Casts that involve a 16-bit
float dtype must therefore be lowered to CUDA conversion intrinsics; these
tests assert the emitted expressions and their on-demand includes, so a
regression to static_cast<__nv_bfloat16>(...) is caught without a GPU.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval
from little_kernel.core.type_system import bfloat16, float16, float32, float64, int32, uint64, bool_
from little_kernel.backends.nv.common.language.intrin.dtype import half_float_cast_code

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def casts(x: ll.Tensor[ll.float32], o: ll.Tensor[ll.float32],
          n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        v: ll.float32 = x[i]
        b: ll.bfloat16 = v.to(ll.bfloat16)
        h: ll.float16 = ll.val_cast(v + 1.0, ll.float16)
        h2: ll.float16 = ll.float16(v * 2.0)
        fb: ll.float32 = ll.val_cast(b, ll.float32)
        fh: ll.float32 = h.to(ll.float32)
        o[i] = fb + fh + ll.val_cast(h2, ll.float32) * 0.5
'''

SPEC = {
    "kernel": "casts",
    "function": "f",
    "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
    "outputs": [{"name": "o", "dtype": "float32", "shape": ["x[0]"]}],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
    "kernel_args": ["x", "o", "n"],
    "forward_args": ["x"],
    "launch": {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": [128, 1, 1]},
}


class HalfFloatCastCodeTests(unittest.TestCase):
    """Unit-level checks of the (code, includes) helper itself."""

    def test_f32_to_bf16_intrinsic(self):
        code, includes = half_float_cast_code("v", float32, bfloat16)
        self.assertEqual(code, "__float2bfloat16_rn(v)")
        self.assertIn("<cuda_bf16.h>", includes)

    def test_f32_to_f16_intrinsic(self):
        code, _ = half_float_cast_code("v", float32, float16)
        self.assertEqual(code, "__float2half_rn(v)")

    def test_bf16_to_f32_widen_intrinsic(self):
        code, _ = half_float_cast_code("b", bfloat16, float32)
        self.assertEqual(code, "__bfloat162float(b)")

    def test_f16_to_f32_widen_intrinsic(self):
        code, _ = half_float_cast_code("h", float16, float32)
        self.assertEqual(code, "__half2float(h)")

    def test_f16_to_f64_double_widen(self):
        code, _ = half_float_cast_code("h", float16, float64)
        self.assertEqual(code, "static_cast<double>(__half2float(h))")

    def test_f64_to_bf16_stages_through_float(self):
        code, _ = half_float_cast_code("d", float64, bfloat16)
        self.assertEqual(code, "__float2bfloat16_rn(static_cast<float>(d))")

    def test_int_to_f16_matches_header_constructor_sequence(self):
        code, _ = half_float_cast_code("i", int32, float16)
        self.assertEqual(code, "__float2half_rn(static_cast<float>(i))")

    def test_u64_to_bf16_stages_through_float(self):
        code, _ = half_float_cast_code("u", uint64, bfloat16)
        self.assertEqual(code, "__float2bfloat16_rn(static_cast<float>(u))")

    def test_bool_to_bf16_stages_through_float(self):
        code, _ = half_float_cast_code("t", bool_, bfloat16)
        self.assertEqual(code, "__float2bfloat16_rn(static_cast<float>(t))")

    def test_cross_16bit_goes_through_widen_intrinsic(self):
        code, _ = half_float_cast_code("b", bfloat16, float16)
        self.assertEqual(code, "__float2half_rn(__bfloat162float(b))")

    def test_same_type_returns_none(self):
        self.assertIsNone(half_float_cast_code("v", bfloat16, bfloat16))

    def test_unknown_source_returns_none(self):
        self.assertIsNone(half_float_cast_code("v", None, bfloat16))

    def test_non_16bit_casts_untouched(self):
        self.assertIsNone(half_float_cast_code("v", float32, int32))
        self.assertIsNone(half_float_cast_code("v", int32, float64))


class HalfFloatCastCodegenTests(unittest.TestCase):
    """End-to-end export codegen checks: no ill-formed static_cast survives."""

    def _export_cuda(self, source=KERNEL_SOURCE):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(source)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(SPEC))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        return paths[0].read_text()

    def test_to_bf16_uses_intrinsic(self):
        cuda = self._export_cuda()
        self.assertIn("__float2bfloat16_rn(v)", cuda)
        self.assertNotIn("static_cast<__nv_bfloat16>", cuda)

    def test_val_cast_to_f16_uses_intrinsic(self):
        cuda = self._export_cuda()
        self.assertIn("__float2half_rn(", cuda)
        self.assertNotIn("static_cast<__half>", cuda)

    def test_type_constructor_to_f16_uses_intrinsic(self):
        cuda = self._export_cuda()
        self.assertIn("__float2half_rn(v * 2.0f", cuda)

    def test_widen_back_to_f32_uses_intrinsics(self):
        cuda = self._export_cuda()
        self.assertIn("__bfloat162float(", cuda)
        self.assertIn("__half2float(", cuda)

    def test_half_headers_emitted_on_demand(self):
        cuda = self._export_cuda()
        self.assertEqual(cuda.count("#include <cuda_bf16.h>"), 1)
        self.assertEqual(cuda.count("#include <cuda_fp16.h>"), 1)

    def test_plain_f32_kernel_keeps_static_cast(self):
        src = KERNEL_SOURCE.replace("b: ll.bfloat16 = v.to(ll.bfloat16)", "k: ll.int32 = v.to(ll.int32)").replace(
            "h: ll.float16 = ll.val_cast(v + 1.0, ll.float16)", "h: ll.float32 = v + 1.0").replace(
                "h2: ll.float16 = ll.float16(v * 2.0)",
                "h2: ll.float32 = v * 2.0").replace("fb: ll.float32 = ll.val_cast(b, ll.float32)",
                                                    "fb: ll.float32 = ll.val_cast(k, ll.float32)").replace(
                                                        "fh: ll.float32 = h.to(ll.float32)",
                                                        "fh: ll.float32 = h").replace(
                                                            "o[i] = fb + fh + ll.val_cast(h2, ll.float32) * 0.5",
                                                            "o[i] = fb + fh + h2 * 0.5")
        cuda = self._export_cuda(source=src)
        self.assertIn("static_cast<int32_t>", cuda)
        self.assertNotIn("__float2bfloat16", cuda)


if __name__ == "__main__":
    unittest.main()
