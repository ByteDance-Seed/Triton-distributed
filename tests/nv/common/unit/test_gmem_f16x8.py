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
"""f16x8 (16-byte) typed vector GMEM twins with cache-policy variants.

Render-layer legality tests (no GPU), mirroring the bf16x8/f32x4 policy set
contract:
  * gmem_ld_f16x8[/_cg/_streaming] move 16 bytes into two uint64
    register bit patterns (r0 = elements 0-3, r1 = elements 4-7), the
    policy twins moving bits identical to the default load;
  * gmem_st_f16x8[/_cg/_streaming] RNE-pack eight fp32 values into eight
    FP16 and store one aligned 16-byte block;
  * vec_f16x8_lane(lo, hi, k) extracts compile-time literal lane k in
    0..7 as fp32 and rejects dynamic lanes;
  * the <cuda_fp16.h> include is emitted exactly when an fp16 helper body
    is used, and programs that use none of the new intrinsics render
    without the new helper functions.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda
from little_kernel.backends.nv.common.language.intrin.memory import codegen_vec_f16x8_lane

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy8(x: ll.Tensor[ll.float16], y: ll.Tensor[ll.float16],
          nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        off: ll.int32 = c * 8
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        lo2: ll.uint64 = 0
        hi2: ll.uint64 = 0
        lo3: ll.uint64 = 0
        hi3: ll.uint64 = 0
        ll.gmem_ld_f16x8(x, off, lo, hi)
        ll.gmem_ld_f16x8_cg(x, off, lo2, hi2)
        ll.gmem_ld_f16x8_streaming(x, off, lo3, hi3)
        f0: ll.float32 = ll.vec_f16x8_lane(lo, hi, 0)
        f1: ll.float32 = ll.vec_f16x8_lane(lo2, hi2, 1)
        f2: ll.float32 = ll.vec_f16x8_lane(lo3, hi3, 2)
        f3: ll.float32 = ll.vec_f16x8_lane(lo, hi, 3)
        f4: ll.float32 = ll.vec_f16x8_lane(lo2, hi2, 4)
        f5: ll.float32 = ll.vec_f16x8_lane(lo3, hi3, 5)
        f6: ll.float32 = ll.vec_f16x8_lane(lo, hi, 6)
        f7: ll.float32 = ll.vec_f16x8_lane(lo2, hi2, 7)
        ll.gmem_st_f16x8(y, off, f0, f1, f2, f3, f4, f5, f6, f7)
        ll.gmem_st_f16x8_cg(y, off, f0, f1, f2, f3, f4, f5, f6, f7)
        ll.gmem_st_f16x8_streaming(y, off, f0, f1, f2, f3, f4, f5, f6, f7)
'''


def _spec():
    return {
        "kernel": "copy8",
        "function": "copy8",
        "inputs": [{"name": "x", "dtype": "float16", "rank": 1}],
        "output": {"name": "y", "dtype": "float16", "shape": ["x[0]"]},
        "scalars": [{"name": "nvec", "expr": "x[0]"}],
        "launch": {"grid": ["ceil_div(nvec, 256)", 1, 1], "block": [256, 1, 1]},
    }


class GmemF16x8Tests(unittest.TestCase):

    def _render(self, directory, source=SOURCE):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "copy8")
        return _render_cuda(spec, kernel)

    def test_all_policies_render(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Load helpers: default dereference, .cg, and .cs policy variants.
        self.assertIn("gmem_ld_f16x8_fn(", cuda)
        self.assertIn("gmem_ld_f16x8_cg_fn(", cuda)
        self.assertIn("gmem_ld_f16x8_streaming_fn(", cuda)
        self.assertIn("__ldcg(reinterpret_cast<const uint4*>(ptr))", cuda)
        self.assertIn("__ldcs(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Store helpers: RNE pack + default/.cg/.cs stores.
        self.assertIn("gmem_st_f16x8_fn(", cuda)
        self.assertIn("gmem_st_f16x8_cg_fn(", cuda)
        self.assertIn("gmem_st_f16x8_streaming_fn(", cuda)
        self.assertIn("__floats2half2_rn(f0, f1)", cuda)
        self.assertIn("__stcg(reinterpret_cast<uint4*>(ptr), value)", cuda)
        self.assertIn("__stcs(reinterpret_cast<uint4*>(ptr), value)", cuda)
        # Lane extractor and its fp16 include.
        self.assertIn("vec_f16x8_lane_fn(", cuda)
        self.assertIn("<cuda_fp16.h>", cuda)
        # Out-param wiring: address-of locals, pointer arithmetic at the
        # call site (element index on the fp16 tensor pointer).
        self.assertIn("&lo", cuda)

    def test_no_include_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy8(x: ll.Tensor[ll.float16], y: ll.Tensor[ll.float16],
          nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        y[c] = x[c]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, source)
        self.assertNotIn("f16x8", cuda)

    def test_lane_literal_legality(self):
        for lane in ("0", "7"):
            builtin = codegen_vec_f16x8_lane("lo", "hi", lane)
            self.assertIn(f", {lane})", builtin.return_val)
        for bad in ("8", "-1", "k", "lane + 1"):
            with self.assertRaises(ValueError):
                codegen_vec_f16x8_lane("lo", "hi", bad)


if __name__ == "__main__":
    unittest.main()
