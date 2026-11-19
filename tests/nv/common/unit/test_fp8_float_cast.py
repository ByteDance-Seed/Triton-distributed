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
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""Regressions for fp8_e4m3/fp8_e5m2 val_cast/to/type-constructor lowering.

Torch CUDA extension builds define __CUDA_NO_FP8_CONVERSIONS__, deleting the
converting constructors and conversion operators of __nv_fp8_e4m3 /
__nv_fp8_e5m2 in <cuda_fp8.h> (the same family of macro that motivated the
16-bit half_float_cast_code lowering).  Casts that involve an 8-bit float
dtype must therefore be lowered to the __nv_cvt_* conversion intrinsics,
which exist regardless of the macros; these tests assert the emitted
expressions and their on-demand includes, so a regression to
static_cast<__nv_fp8_e4m3>(...) is caught without a GPU.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval
from little_kernel.core.type_system import bfloat16, bool_, fp8_e4m3, fp8_e5m2, float16, float32, float64, int32, uint32
from little_kernel.backends.nv.common.language.intrin.dtype import half_float_cast_code

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def casts(x: ll.Tensor[ll.float32], packed: ll.Tensor[ll.uint8],
          o: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    # fp8 payload bytes move through uint8 tensor storage (the kernel-level
    # fp8 tensor ABI); scalar fp8 values appear as locals carrying the e4m3 /
    # e5m2 element type.
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        v: ll.float32 = x[i]
        e: ll.fp8_e4m3 = v.to(ll.fp8_e4m3)
        e5: ll.fp8_e5m2 = ll.val_cast(v + 1.0, ll.fp8_e5m2)
        e2: ll.fp8_e4m3 = ll.fp8_e4m3(v * 2.0)
        # Numeric (not bit-reinterpret) fp8 -> uint8: follows the CUDA
        # header conversion-operator sequence through float.
        packed[i] = e2.to(ll.uint8)
        fv: ll.float32 = ll.val_cast(e, ll.float32)
        o[i] = fv + ll.val_cast(e5, ll.float32)
'''

SPEC = {
    "kernel":
    "casts",
    "function":
    "f",
    "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
    "outputs": [
        {"name": "packed", "dtype": "float8_e4m3fn", "shape": ["x[0]"]},
        {"name": "o", "dtype": "float32", "shape": ["x[0]"]},
    ],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
    "kernel_args": ["x", "packed", "o", "n"],
    "forward_args": ["x"],
    "launch": {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": [128, 1, 1]},
}


class Fp8FloatCastCodeTests(unittest.TestCase):
    """Unit-level checks of the (code, includes) helper's fp8 paths."""

    def test_f32_to_e4m3_intrinsic(self):
        code, includes = half_float_cast_code("v", float32, fp8_e4m3)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)")
        self.assertIn("<cuda_fp8.h>", includes)

    def test_f32_to_e5m2_intrinsic(self):
        code, includes = half_float_cast_code("v", float32, fp8_e5m2)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E5M2)")

    def test_f16_to_e4m3_widens_first(self):
        code, _ = half_float_cast_code("h", float16, fp8_e4m3)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(__half2float(h), __NV_SATFINITE, __NV_E4M3)")

    def test_bf16_to_e4m3_widens_first(self):
        code, includes = half_float_cast_code("b", bfloat16, fp8_e4m3)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(__bfloat162float(b), __NV_SATFINITE, "
                         "__NV_E4M3)")
        self.assertIn("<cuda_bf16.h>", includes)

    def test_f64_to_e4m3_stages_through_float(self):
        code, _ = half_float_cast_code("d", float64, fp8_e4m3)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(static_cast<float>(d), __NV_SATFINITE, "
                         "__NV_E4M3)")

    def test_int_to_e4m3_stages_through_float(self):
        code, _ = half_float_cast_code("i", int32, fp8_e4m3)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(static_cast<float>(i), __NV_SATFINITE, "
                         "__NV_E4M3)")

    def test_bool_to_e4m3_stages_through_float(self):
        code, _ = half_float_cast_code("t", bool_, fp8_e4m3)
        self.assertEqual(code, "__nv_cvt_float_to_fp8(static_cast<float>(t), __NV_SATFINITE, "
                         "__NV_E4M3)")

    def test_e4m3_to_f32_widens_through_halfraw(self):
        code, includes = half_float_cast_code("e", fp8_e4m3, float32)
        self.assertEqual(code, "__half2float(__nv_cvt_fp8_to_halfraw((e).__x, __NV_E4M3))")
        self.assertIn("<cuda_fp16.h>", includes)

    def test_e5m2_to_f32_widens_through_halfraw(self):
        code, _ = half_float_cast_code("e", fp8_e5m2, float32)
        self.assertEqual(code, "__half2float(__nv_cvt_fp8_to_halfraw((e).__x, __NV_E5M2))")

    def test_e4m3_to_f16_goes_through_exact_widen(self):
        code, _ = half_float_cast_code("e", fp8_e4m3, float16)
        self.assertEqual(code, "__float2half_rn(__half2float(__nv_cvt_fp8_to_halfraw((e).__x, "
                         "__NV_E4M3)))")

    def test_e4m3_to_bf16_goes_through_exact_widen(self):
        code, includes = half_float_cast_code("e", fp8_e4m3, bfloat16)
        self.assertEqual(code, "__float2bfloat16_rn(__half2float(__nv_cvt_fp8_to_halfraw("
                         "(e).__x, __NV_E4M3)))")
        self.assertIn("<cuda_bf16.h>", includes)

    def test_e4m3_to_f64_exact_widen_then_double(self):
        code, _ = half_float_cast_code("e", fp8_e4m3, float64)
        self.assertEqual(code, "static_cast<double>(__half2float(__nv_cvt_fp8_to_halfraw("
                         "(e).__x, __NV_E4M3)))")

    def test_e4m3_to_int_goes_through_exact_widen(self):
        code, _ = half_float_cast_code("e", fp8_e4m3, int32)
        self.assertEqual(code, "static_cast<int32_t>(__half2float(__nv_cvt_fp8_to_halfraw("
                         "(e).__x, __NV_E4M3)))")

    def test_e5m2_to_uint32_goes_through_exact_widen(self):
        code, _ = half_float_cast_code("e", fp8_e5m2, uint32)
        self.assertEqual(code, "static_cast<uint32_t>(__half2float(__nv_cvt_fp8_to_halfraw("
                         "(e).__x, __NV_E5M2)))")

    def test_cross_flavour_e4m3_to_e5m2_double_narrows(self):
        code, _ = half_float_cast_code("e", fp8_e4m3, fp8_e5m2)
        self.assertEqual(
            code, "__nv_cvt_float_to_fp8(__half2float(__nv_cvt_fp8_to_halfraw("
            "(e).__x, __NV_E4M3)), __NV_SATFINITE, __NV_E5M2)")

    def test_same_type_returns_none(self):
        self.assertIsNone(half_float_cast_code("e", fp8_e4m3, fp8_e4m3))
        self.assertIsNone(half_float_cast_code("e", fp8_e5m2, fp8_e5m2))

    def test_unknown_source_returns_none(self):
        self.assertIsNone(half_float_cast_code("v", None, fp8_e4m3))

    def test_non_fp8_casts_untouched(self):
        self.assertIsNone(half_float_cast_code("v", float32, int32))
        self.assertIsNone(half_float_cast_code("v", int32, float64))


class Fp8FloatCastCodegenTests(unittest.TestCase):
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

    def test_to_e4m3_uses_intrinsic(self):
        cuda = self._export_cuda()
        self.assertIn("__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)", cuda)
        self.assertNotIn("static_cast<__nv_fp8_e4m3>", cuda)

    def test_val_cast_to_e5m2_uses_intrinsic(self):
        cuda = self._export_cuda()
        self.assertIn("__NV_SATFINITE, __NV_E5M2", cuda)
        self.assertNotIn("static_cast<__nv_fp8_e5m2>", cuda)

    def test_type_constructor_to_e4m3_uses_intrinsic(self):
        cuda = self._export_cuda()
        self.assertIn("__nv_cvt_float_to_fp8(v * 2.0f", cuda)

    def test_store_uses_byte_conversion_without_static_cast(self):
        cuda = self._export_cuda()
        # The byte store goes through the fp8 -> uint8 widening conversion;
        # no fp8 static_cast may appear anywhere in the translation unit.
        self.assertIn("__nv_cvt_float_to_fp8(", cuda)
        self.assertNotIn("static_cast<__nv_fp8_", cuda)

    def test_widen_back_to_f32_uses_intrinsics(self):
        cuda = self._export_cuda()
        self.assertIn("__nv_cvt_fp8_to_halfraw(", cuda)

    def test_fp8_headers_emitted_on_demand(self):
        cuda = self._export_cuda()
        self.assertEqual(cuda.count("#include <cuda_fp8.h>"), 1)
        self.assertEqual(cuda.count("#include <cuda_fp16.h>"), 1)

    def test_plain_f32_kernel_keeps_static_cast(self):
        # Variant with no fp8 involvement at all: the intrinsic must not
        # appear and int casts keep the historical static_cast form.
        src = KERNEL_SOURCE.replace("e: ll.fp8_e4m3 = v.to(ll.fp8_e4m3)", "k: ll.int32 = v.to(ll.int32)").replace(
            "e5: ll.fp8_e5m2 = ll.val_cast(v + 1.0, ll.fp8_e5m2)", "e5: ll.float32 = v + 1.0").replace(
                "e2: ll.fp8_e4m3 = ll.fp8_e4m3(v * 2.0)",
                "e2: ll.float32 = v * 2.0").replace("        packed[i] = e2.to(ll.uint8)\n", "").replace(
                    "fv: ll.float32 = ll.val_cast(e, ll.float32)",
                    "fv: ll.float32 = ll.val_cast(k, ll.float32)").replace("o[i] = fv + ll.val_cast(e5, ll.float32)",
                                                                           "o[i] = fv + e5")
        cuda = self._export_cuda(source=src)
        self.assertIn("static_cast<int32_t>", cuda)
        self.assertNotIn("__nv_cvt_float_to_fp8(", cuda)


if __name__ == "__main__":
    unittest.main()
