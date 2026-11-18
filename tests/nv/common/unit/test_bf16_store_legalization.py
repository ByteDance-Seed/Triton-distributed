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
"""Regressions for typed BF16/FP16 subscript-store legalization (CUDA).

Typed store validation accepts C++-style promotion of the stored value onto
the tensor element type, so ``out[i] = 0.0`` on a ``Tensor[bfloat16]``
reaches codegen still typed float32.  Under the pinned torch extension flags
(__CUDA_NO_HALF_CONVERSIONS__ / __CUDA_NO_BFLOAT16_CONVERSIONS__) the naive
emission ``out[i] = 0.0f;`` is ill-formed:
``no operator '=' matches these operands; operand types are __nv_bfloat16 = float``.

The CUDA backend routes mismatched subscript-store values through
``half_float_cast_code`` so they are lowered with explicit round-to-nearest-
even conversion intrinsics.  These tests pin the emitted code strings for
tail/masked stores of zero/nonzero literals, FP32 expressions and explicit
casts, plus FP16 and FP32 regressions — all without a GPU.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval
from little_kernel.codegen.codegen_base import CppEmitter
from little_kernel.core.type_system import bfloat16, float32


def _make_kernel(body: str, out_dtype: str = "bfloat16", x_dtype: str = "float32") -> str:
    return f'''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def store_tail(x: ll.Tensor[ll.{x_dtype}], out: ll.Tensor[ll.{out_dtype}],
               n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        {body}
'''


def _make_spec(kernel: str, out_dtype: str = "bfloat16", x_dtype: str = "float32") -> dict:
    return {
        "kernel": kernel,
        "function": "f",
        "inputs": [{"name": "x", "dtype": x_dtype, "rank": 1}],
        "outputs": [{"name": "out", "dtype": out_dtype, "shape": ["x[0]"]}],
        "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
        "kernel_args": ["x", "out", "n"],
        "forward_args": ["x"],
        "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
    }


class SubscriptStoreLegalizationCodegenTests(unittest.TestCase):
    """End-to-end export codegen checks for typed subscript stores."""

    def _export_cuda(self, body, out_dtype="bfloat16", x_dtype="float32"):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(_make_kernel(body, out_dtype, x_dtype))
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(_make_spec("store_tail", out_dtype, x_dtype)))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        return paths[0].read_text()

    # ---------------------- BF16 targets: must be legalized -------------------

    def test_bf16_store_zero_literal_masked(self):
        cuda = self._export_cuda("out[i] = 0.0")
        self.assertIn("__float2bfloat16_rn(0.0f)", cuda)
        self.assertNotIn("= 0.0f;", cuda)
        self.assertNotIn("static_cast<__nv_bfloat16>", cuda)
        self.assertEqual(cuda.count("#include <cuda_bf16.h>"), 1)

    def test_bf16_store_nonzero_literal(self):
        cuda = self._export_cuda("out[i] = 1.5")
        self.assertIn("__float2bfloat16_rn(1.5f)", cuda)
        self.assertNotIn("= 1.5f;", cuda)
        self.assertNotIn("static_cast<__nv_bfloat16>", cuda)

    def test_bf16_store_fp32_expression(self):
        cuda = self._export_cuda("out[i] = x[i] * 2.0 + 1.0")
        self.assertIn("__float2bfloat16_rn(", cuda)
        self.assertNotIn("= x[i] * 2.0f + 1.0f;", cuda)
        self.assertNotIn("static_cast<__nv_bfloat16>", cuda)

    def test_bf16_store_explicit_float_to_bf16_preserved(self):
        cuda = self._export_cuda("out[i] = ll.float_to_bf16(0.0)")
        # Already explicitly converted: single RNE lowering, no double wrap.
        self.assertEqual(cuda.count("__float2bfloat16_rn("), 1)
        self.assertNotIn("static_cast<__nv_bfloat16>", cuda)

    def test_bf16_store_bf16_value_unchanged(self):
        # Storing a value already typed bf16 (load + explicit cast chain)
        # must not be re-converted.
        body = "v: ll.bfloat16 = ll.float_to_bf16(x[i])\n        out[i] = v"
        cuda = self._export_cuda(body)
        self.assertNotIn("__float2bfloat16_rn(__float2bfloat16_rn", cuda)
        self.assertNotIn("static_cast<__nv_bfloat16>", cuda)

    # --------------------------- FP16 target regression -----------------------

    def test_f16_store_zero_literal(self):
        cuda = self._export_cuda("out[i] = 0.0", out_dtype="float16")
        self.assertIn("__float2half_rn(0.0f)", cuda)
        self.assertNotIn("static_cast<__half>", cuda)
        self.assertEqual(cuda.count("#include <cuda_fp16.h>"), 1)

    def test_f16_store_fp32_expression(self):
        cuda = self._export_cuda("out[i] = x[i] + 1.0", out_dtype="float16")
        self.assertIn("__float2half_rn(", cuda)
        self.assertNotIn("static_cast<__half>", cuda)

    # --------------------------- FP32 target regression -----------------------

    def test_f32_store_literal_unchanged(self):
        cuda = self._export_cuda("out[i] = 0.0", out_dtype="float32")
        self.assertIn("out[i] = 0.0f;", cuda)
        self.assertNotIn("__float2bfloat16_rn", cuda)
        self.assertNotIn("__float2half_rn", cuda)

    def test_f32_store_expression_unchanged(self):
        cuda = self._export_cuda("out[i] = x[i] * 2.0", out_dtype="float32")
        self.assertIn("out[i] = x[i] * 2.0f;", cuda)
        self.assertNotIn("__float2bfloat16_rn", cuda)
        self.assertNotIn("__float2half_rn", cuda)


class StoreLegalizationHookTests(unittest.TestCase):
    """The base backend hook is a strict no-op (non-CUDA backends unchanged)."""

    def test_base_hook_is_noop(self):
        # The hook is instance-state-free; skip the ctx validation in __init__.
        emitter = CppEmitter.__new__(CppEmitter)
        self.assertEqual(emitter._convert_value_for_store("0.0f", float32, bfloat16), "0.0f")

    def test_base_hook_noop_matching_types(self):
        emitter = CppEmitter.__new__(CppEmitter)
        self.assertEqual(emitter._convert_value_for_store("v", bfloat16, bfloat16), "v")


if __name__ == "__main__":
    unittest.main()
