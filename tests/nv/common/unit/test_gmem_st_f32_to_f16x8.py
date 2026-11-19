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
"""f32 -> f16/bf16 widening vector GMEM stores (facade re-export regression).

The codegen for the gmem_st_f32_to_{f16,bf16}x{4,8} family lives in
language/intrin/memory.py, but the pinned facade export list omitted the
names, so drafts hit:
    AttributeError: module 'little_kernel.language' has no attribute
    'gmem_st_f32_to_f16x8'

Render-layer legality tests (no GPU):
  * the four names are importable from little_kernel.language;
  * gmem_st_f32_to_f16x8 RNE-packs eight fp32 lanes into eight FP16 and
    stores one aligned 16-byte block (uint4 cast + __floats2half2_rn);
  * the fp16 include is emitted exactly when an fp16 helper body is used;
  * the bf16 twins emit __floats2bfloat162_rn and <cuda_bf16.h>.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def store8(x: ll.Tensor[ll.float16], y: ll.Tensor[ll.float16]) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    off: ll.int32 = c * 8
    f: ll.float32 = 1.0
    g: ll.float32 = 2.0
    ll.gmem_st_f32_to_f16x8(y, off, f, g, f, g, f, g, f, g)
    ll.gmem_st_f32_to_f16x4(y, off, f, g, f, g)
'''


def _spec():
    return {
        "kernel": "store8",
        "function": "store8",
        "inputs": [{"name": "x", "dtype": "float16", "rank": 1}],
        "output": {"name": "y", "dtype": "float16", "shape": ["x[0]"]},
        "scalars": [],
        "launch": {"grid": ["ceil_div(x[0], 2048)", 1, 1], "block": [256, 1, 1]},
    }


class GmemStF32ToF16x8Tests(unittest.TestCase):

    def test_facade_exports(self):
        for name in ("gmem_st_f32_to_f16x8", "gmem_st_f32_to_f16x4", "gmem_st_f32_to_bf16x8", "gmem_st_f32_to_bf16x4"):
            self.assertTrue(hasattr(ll, name), f"ll.{name} missing from facade")

    def _render(self, directory, source=SOURCE):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "store8")
        return _render_cuda(spec, kernel)

    def test_f16x8_store_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        self.assertIn("gmem_st_f32_to_f16x8_fn(", cuda)
        self.assertIn("gmem_st_f32_to_f16x4_fn(", cuda)
        self.assertIn("__floats2half2_rn(f0, f1)", cuda)
        self.assertIn("*reinterpret_cast<uint4*>(ptr) = v;", cuda)
        self.assertIn("*reinterpret_cast<uint2*>(ptr) = v;", cuda)
        self.assertIn("<cuda_fp16.h>", cuda)

    def test_bf16_store_renders(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def store8(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16]) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    off: ll.int32 = c * 8
    f: ll.float32 = 1.0
    g: ll.float32 = 2.0
    ll.gmem_st_f32_to_bf16x8(y, off, f, g, f, g, f, g, f, g)
    ll.gmem_st_f32_to_bf16x4(y, off, f, g, f, g)
'''
        spec = _spec()
        spec["inputs"] = [{"name": "x", "dtype": "bfloat16", "rank": 1}]
        spec["output"] = {"name": "y", "dtype": "bfloat16", "shape": ["x[0]"]}

        def render(directory):
            root = Path(directory)
            (root / "export.json").write_text(json.dumps(spec))
            kernel_path = root / "kernel.py"
            kernel_path.write_text(source)
            loaded_spec = _load_spec(root / "export.json")
            kernel = _load_kernel(kernel_path, "store8")
            return _render_cuda(loaded_spec, kernel)

        with tempfile.TemporaryDirectory() as directory:
            cuda = render(directory)
        self.assertIn("gmem_st_f32_to_bf16x8_fn(", cuda)
        self.assertIn("gmem_st_f32_to_bf16x4_fn(", cuda)
        self.assertIn("__floats2bfloat162_rn(f0, f1)", cuda)
        self.assertIn("<cuda_bf16.h>", cuda)

    def test_no_include_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def store8(x: ll.Tensor[ll.float16], y: ll.Tensor[ll.float16]) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    y[c] = x[c]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, source)
        self.assertNotIn("gmem_st_f32_to_f16x8", cuda)


if __name__ == "__main__":
    unittest.main()
