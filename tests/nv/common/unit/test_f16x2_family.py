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
"""FP16x2 intrinsic family: pack_f16_float / mul_f16x2 / add_f16x2 /
unpack_f16x2_to_f32.

Cycle-12 request audit (/exchange/requests/cycle-012-kernel-01.md, task
0003, sampled task): the requester's fp16-twin check resolves to
"present in the accepted source" - these tests make that "present AND
regression-tested".  The family is the exact dtype-twin of the
regression-tested BF16x2 family: bit-move between one uint32 container
and two fp32/fp16 lanes, lane order preserved (lo = low-address
element), RNE on pack.  The first level asserts the builtin registry
and emitted helper text; the second drives the real
draft -> PASSES -> codegen -> export() renderer (no nvcc) and asserts
source snapshots.  No task, shape, or launch-legality assumption: any
fp16 streaming kernel with a scalar/ragged path may use these.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations import vibe_cuda_eval
from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_RETURN_TYPE_ATTR
from little_kernel.language.builtin_base import Builtin
import little_kernel.backends.nv.common.language.intrin.cuda_asm as cuda_asm

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f16x2fma(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
             out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    j: ll.int32 = i * 2
    if j + 1 < n:
        a: ll.uint32 = ll.pack_f16_float(x[j], x[j + 1])
        b: ll.uint32 = ll.pack_f16_float(y[j], y[j + 1])
        prod: ll.uint32 = ll.mul_f16x2(a, b)
        acc: ll.uint32 = ll.add_f16x2(prod, a)
        lanes = ll.empty([2], dtype=ll.float32, scope="local")
        ll.unpack_f16x2_to_f32(acc, lanes[0], lanes[1])
        out[j] = lanes[0]
        out[j + 1] = lanes[1]
'''

SPEC = {
    "kernel": "f16x2fma",
    "function": "f",
    "inputs": [{"name": "x", "dtype": "float32", "rank": 1}, {"name": "y", "dtype": "float32", "rank": 1}],
    "outputs": [{"name": "out", "dtype": "float32", "shape": ["x[0]"]}],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
    "kernel_args": ["x", "y", "out", "n"],
    "forward_args": ["x", "y"],
    "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [128, 1, 1]},
}


class F16x2RegistryTests(unittest.TestCase):

    def _assert_registered(self, name):
        fn = getattr(cuda_asm, name)
        self.assertTrue(getattr(fn, BUILTIN_ATTR, False))
        self.assertIs(getattr(ll, name), fn)

    def test_all_four_registered_and_exported(self):
        for name in ("pack_f16_float", "mul_f16x2", "add_f16x2", "unpack_f16x2_to_f32"):
            self._assert_registered(name)

    def test_pack_f16_float_codegen_rne_and_header(self):
        result = getattr(cuda_asm.pack_f16_float, CODEGEN_FUNC_ATTR)("fp32_a", "fp32_b")
        self.assertIsInstance(result, Builtin)
        # Round-to-nearest-even pack of two f32 into one __half2.
        self.assertIn("__floats2half2_rn(fp32_a, fp32_b)", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertEqual(result.return_val, "pack_f16_float_fn(fp32_a, fp32_b)")
        return_type = getattr(cuda_asm.pack_f16_float, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(return_type(None, None), ll.uint32)

    def test_mul_f16x2_codegen_lane_wise(self):
        result = getattr(cuda_asm.mul_f16x2, CODEGEN_FUNC_ATTR)("lhs", "rhs")
        self.assertIn("__hmul2(a, b)", result.body)
        self.assertIn("__half2", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertEqual(result.return_val, "mul_f16x2_fn(lhs, rhs)")

    def test_add_f16x2_codegen_lane_wise(self):
        result = getattr(cuda_asm.add_f16x2, CODEGEN_FUNC_ATTR)("lhs", "rhs")
        self.assertIn("__hadd2(a, b)", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertEqual(result.return_val, "add_f16x2_fn(lhs, rhs)")

    def test_unpack_f16x2_to_f32_preserves_lane_order(self):
        result = getattr(cuda_asm.unpack_f16x2_to_f32, CODEGEN_FUNC_ATTR)("bits", "lo", "hi")
        self.assertIn("__half22float2", result.body)
        # lo is the low-address element, mirroring the bf16 twin.
        self.assertIn("*lo = pair.x; *hi = pair.y;", result.body)
        self.assertIn("<cuda_fp16.h>", result.includes)
        self.assertEqual(result.return_val, "unpack_f16x2_to_f32_fn(bits, &lo, &hi)")


class F16x2ExportPipelineTests(unittest.TestCase):

    def _export_cuda(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(KERNEL_SOURCE)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(SPEC))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        return {p.name: p.read_text() for p in paths}

    def test_all_four_intrinsics_render_through_export(self):
        cuda = next(v for k, v in self._export_cuda().items() if k.endswith(".cu"))
        self.assertIn("pack_f16_float_fn(", cuda)
        self.assertIn("__floats2half2_rn", cuda)
        self.assertIn("mul_f16x2_fn(", cuda)
        self.assertIn("__hmul2(", cuda)
        self.assertIn("add_f16x2_fn(", cuda)
        self.assertIn("__hadd2(", cuda)
        self.assertIn("unpack_f16x2_to_f32_fn(", cuda)
        self.assertIn("__half22float2", cuda)
        self.assertIn("#include <cuda_fp16.h>", cuda)

    def test_binding_keeps_float32_guards(self):
        cpp = next(v for k, v in self._export_cuda().items() if k.endswith("_binding.cpp"))
        self.assertIn("at::kFloat", cpp)


if __name__ == "__main__":
    unittest.main()
