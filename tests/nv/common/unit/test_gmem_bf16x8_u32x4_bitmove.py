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
"""bf16x8 / f16x8 raw 4xuint32 bit-move GMEM pair (no float round trip).

Capability ir.gmem_st_v4_bf16_bits: a no-scale bf16 repack row copy must be
expressible as a pure 128-bit bit move -- ld.global.v4.b32 /
st.global.v4.b32 of four uint32 words reinterpreted as bf16x8 -- with the
element dtype preserved end to end and *no* bf16->f32 widening / f32->bf16
RNE narrowing.  The render-level pair is:

  * gmem_ld_bf16x8_u32x4(mem, index, r0..r3)
  * gmem_st_bf16x8_u32x4(mem, index, r0..r3)

(index is in 16-bit elements; 16-byte alignment is the caller's contract,
identical to gmem_ld/st_f32x4.)  The f16 pair
(gmem_ld/st_f16x8_u32x4) is the same bit-move spelling for half tensors.

These tests drive the real draft -> PASSES -> codegen -> render path (no
nvcc) and assert that the load half is reachable from the frontend
namespace (regression: it was defined in intrin.memory but never exported,
so a pure-bit-move copy could not be named and had to detour through the
float-converting family).
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

BF16_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy8(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16],
          nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        off: ll.int32 = c * 8
        w0: ll.uint32 = 0
        w1: ll.uint32 = 0
        w2: ll.uint32 = 0
        w3: ll.uint32 = 0
        ll.gmem_ld_bf16x8_u32x4(x, off, w0, w1, w2, w3)
        ll.gmem_st_bf16x8_u32x4(y, off, w0, w1, w2, w3)
'''

F16_SOURCE = BF16_SOURCE.replace("ll.bfloat16", "ll.float16").replace("bf16x8_u32x4", "f16x8_u32x4")


def _spec(dtype="bfloat16"):
    return {
        "kernel": "copy8",
        "function": "copy8",
        "inputs": [{"name": "x", "dtype": dtype, "rank": 1}],
        "output": {"name": "y", "dtype": dtype, "shape": ["x[0]"]},
        "scalars": [{"name": "nvec", "expr": "x[0]"}],
        "launch": {"grid": ["ceil_div(nvec, 256)", 1, 1], "block": [256, 1, 1]},
    }


class GmemBf16x8U32x4BitMoveTests(unittest.TestCase):

    def _render(self, source, dtype="bfloat16"):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "export.json").write_text(json.dumps(_spec(dtype)))
            kernel_path = root / "kernel.py"
            kernel_path.write_text(source)
            spec = _load_spec(root / "export.json")
            kernel = _load_kernel(kernel_path, "copy8")
            return _render_cuda(spec, kernel)

    def test_pair_exported_from_frontend_namespace(self):
        # The load half of each pair must be nameable as ll.<intrinsic>;
        # the store halves already were.
        for name in ("gmem_ld_bf16x8_u32x4", "gmem_st_bf16x8_u32x4", "gmem_ld_f16x8_u32x4", "gmem_st_f16x8_u32x4"):
            self.assertTrue(hasattr(ll, name), f"{name} not exported")

    def test_bf16_copy_renders_pure_bitmove(self):
        cuda = self._render(BF16_SOURCE)
        # Both halves render as uint4 reinterpret moves over bf16 pointers.
        self.assertIn("void gmem_ld_bf16x8_u32x4_fn(", cuda)
        self.assertIn("const __nv_bfloat16* ptr", cuda)
        self.assertIn("const uint4 value = *reinterpret_cast<const uint4*>(ptr);", cuda)
        self.assertIn("void gmem_st_bf16x8_u32x4_fn(", cuda)
        self.assertIn("*reinterpret_cast<uint4*>(ptr) = uint4{", cuda)
        # Element-indexed addressing on the bf16 tensor (index in elements).
        self.assertIn("gmem_ld_bf16x8_u32x4_fn(x + off, &w0, &w1, &w2, &w3);", cuda)
        self.assertIn("gmem_st_bf16x8_u32x4_fn(y + off, w0, w1, w2, w3);", cuda)

    def test_bf16_bitmove_has_no_float_round_trip(self):
        cuda = self._render(BF16_SOURCE)
        # The exact-widen then RNE-narrow family must not appear: the four
        # words move bits only.
        for marker in ("__bfloat162float", "__float2bfloat16", "__floats2bfloat162_rn", "__floats2bfloat16_rn",
                       "gmem_st_f32_to_bf16x8"):
            self.assertNotIn(marker, cuda, f"unexpected conversion: {marker}")

    def test_f16_pair_renders_pure_bitmove(self):
        cuda = self._render(F16_SOURCE, dtype="float16")
        self.assertIn("void gmem_ld_f16x8_u32x4_fn(", cuda)
        self.assertIn("void gmem_st_f16x8_u32x4_fn(", cuda)
        self.assertIn("gmem_ld_f16x8_u32x4_fn(x + off, &w0, &w1, &w2, &w3);", cuda)
        self.assertIn("gmem_st_f16x8_u32x4_fn(y + off, w0, w1, w2, w3);", cuda)
        for marker in ("__half2float", "__float2half_rn", "gmem_st_f32_to_f16x8"):
            self.assertNotIn(marker, cuda, f"unexpected conversion: {marker}")


if __name__ == "__main__":
    unittest.main()
