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
"""Regression for ir.scalar.f32_to_f16_bits.

Packing fp16 header words into a uint8 body used to require
``float_to_half`` followed by ``val_cast(h, ll.uint16)``, which lowers to
the ill-formed ``static_cast<uint16_t>((__half)h)`` — __half has no
conversion to uint16_t in a __CUDA_NO_HALF_CONVERSIONS__ build
(nvcc: "no suitable conversion function from __half to uint16_t exists").
``ll.float_to_half_bits_u16`` returns the IEEE fp16 storage word (RNE)
without the frontend ever naming __half. These checks are codegen-only:
they need no GPU.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval
from little_kernel.core.type_system import float32, uint16
from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR
from little_kernel.backends.nv.common.language.intrin.arith import codegen_float_to_half_bits_u16, float_to_half_bits_u16

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def pack_header(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.uint8],
                n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        v: ll.float32 = x[i]
        bits: ll.uint16 = ll.float_to_half_bits_u16(v)
        y[2 * i] = ll.val_cast(bits & 0xFF, ll.uint8)
        y[2 * i + 1] = ll.val_cast((bits >> 8) & 0xFF, ll.uint8)
'''

SPEC = {
    "kernel": "pack_header",
    "function": "f",
    "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
    "outputs": [{"name": "y", "dtype": "uint8", "shape": ["2*x[0]"]}],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
    "kernel_args": ["x", "y", "n"],
    "forward_args": ["x"],
    "launch": {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": [128, 1, 1]},
}


class FloatToHalfBitsBuiltinTests(unittest.TestCase):
    """Unit-level checks of the codegen helper itself."""

    def test_helper_body_rounds_then_reinterprets(self):
        b = codegen_float_to_half_bits_u16("v")
        self.assertEqual(b.return_val, "lk_f32_to_f16_bits_u16_fn(v)")
        self.assertIn("__float2half_rn(val)", b.body)
        self.assertIn("__half_as_ushort(", b.body)
        self.assertIn("uint16_t", b.body)
        self.assertIn("<cuda_fp16.h>", b.includes)

    def test_eval_return_type_is_uint16(self):
        # Downstream inference and the byte-lane masks depend on the result
        # being a 16-bit uint rather than an fp16 value.
        fn = getattr(float_to_half_bits_u16, EVAL_RETURN_TYPE_ATTR)
        self.assertIs(fn(float32), uint16)


class FloatToHalfBitsCodegenTests(unittest.TestCase):
    """End-to-end export: fp16 bits packed into a uint8 body, no __half named."""

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

    def test_helper_and_call_emitted(self):
        cuda = self._export_cuda()
        self.assertIn("lk_f32_to_f16_bits_u16_fn", cuda)
        self.assertIn("__half_as_ushort(__float2half_rn", cuda)
        # Defined once even though called from one site; guards duplicates.
        self.assertEqual(cuda.count("lk_f32_to_f16_bits_u16_fn(float"), 1)

    def test_fp16_header_included_once(self):
        cuda = self._export_cuda()
        self.assertEqual(cuda.count("#include <cuda_fp16.h>"), 1)

    def test_no_illformed_half_to_uint_cast(self):
        cuda = self._export_cuda()
        # The exact failure from the reproducer must never reappear.
        self.assertNotIn("static_cast<uint16_t>((__half)", cuda)
        self.assertNotIn("static_cast<uint16_t>(__half", cuda)

    def test_byte_lanes_stored_to_uint8_body(self):
        cuda = self._export_cuda()
        self.assertIn("& 255", cuda)
        self.assertIn(">> 8", cuda)


if __name__ == "__main__":
    unittest.main()
