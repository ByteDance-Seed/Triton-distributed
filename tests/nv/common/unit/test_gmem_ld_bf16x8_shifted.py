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
"""HEAD-shifted aligned bf16x8 window load (GEMV idiom) render tests.

Render-layer legality tests (no GPU), mirroring the gmem_ld_bf16x8 policy
family contract:
  * gmem_ld_bf16x8_shifted[/_cg/_streaming] each emit one aligned 16-byte
    load (default dereference / __ldcg / __ldcs) as a uint4;
  * the raw four words are shuffled from lane+1 with four
    __shfl_down_sync(..., 1) full-mask ops (the warp-edge lane's tail is
    the caller-guaranteed "extra aligned block" case);
  * shifted-lane selection then unpacks raw bf16x2 words to FP32 with
    __bfloat162float, bit-exact to eight scalar bf16_to_float loads;
  * out-registers are wired by address at the call site and ``head`` is a
    runtime int parameter.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def shiftwin(vec: ll.Tensor[ll.bfloat16], out: ll.Tensor[ll.float32],
             n: ll.int32, head: ll.int32) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    # logical window [head + tid*8, head + tid*8 + 8)
    win: ll.int32 = head + tid * 8
    base: ll.int32 = win - (win % 8)
    shift: ll.int32 = win % 8
    s0: ll.float32 = 0.0
    s1: ll.float32 = 0.0
    s2: ll.float32 = 0.0
    s3: ll.float32 = 0.0
    s4: ll.float32 = 0.0
    s5: ll.float32 = 0.0
    s6: ll.float32 = 0.0
    s7: ll.float32 = 0.0
    t0: ll.float32 = 0.0
    t1: ll.float32 = 0.0
    t2: ll.float32 = 0.0
    t3: ll.float32 = 0.0
    t4: ll.float32 = 0.0
    t5: ll.float32 = 0.0
    t6: ll.float32 = 0.0
    t7: ll.float32 = 0.0
    u0: ll.float32 = 0.0
    u1: ll.float32 = 0.0
    u2: ll.float32 = 0.0
    u3: ll.float32 = 0.0
    u4: ll.float32 = 0.0
    u5: ll.float32 = 0.0
    u6: ll.float32 = 0.0
    u7: ll.float32 = 0.0
    ll.gmem_ld_bf16x8_shifted(vec, base, shift,
                              s0, s1, s2, s3, s4, s5, s6, s7)
    ll.gmem_ld_bf16x8_shifted_cg(vec, base, shift,
                                 t0, t1, t2, t3, t4, t5, t6, t7)
    ll.gmem_ld_bf16x8_shifted_streaming(vec, base, shift,
                                        u0, u1, u2, u3, u4, u5, u6, u7)
    if tid == 0:
        out[0] = s0 + s1 + s2 + s3 + s4 + s5 + s6 + s7
        out[1] = t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7
        out[2] = u0 + u1 + u2 + u3 + u4 + u5 + u6 + u7
'''


def _spec():
    return {
        "kernel": "shiftwin",
        "function": "shiftwin",
        "inputs": [{"name": "vec", "dtype": "bfloat16", "rank": 1}],
        "output": {"name": "out", "dtype": "float32", "shape": [3]},
        "scalars": [{"name": "n", "expr": "vec[0]"}, {"name": "head", "expr": "vec[0]"}],
        "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
    }


class GmemLdBf16x8ShiftedTests(unittest.TestCase):

    def _render(self, directory, source=SOURCE):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "shiftwin")
        return _render_cuda(spec, kernel)

    def test_all_policies_render(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Helper definitions and call sites for all three cache policies.
        self.assertIn("gmem_ld_bf16x8_shifted_fn(", cuda)
        self.assertIn("gmem_ld_bf16x8_shifted_cg_fn(", cuda)
        self.assertIn("gmem_ld_bf16x8_shifted_streaming_fn(", cuda)
        # One aligned 16-byte load per policy variant.
        self.assertIn("*reinterpret_cast<const uint4*>(ptr)", cuda)
        self.assertIn("__ldcg(reinterpret_cast<const uint4*>(ptr))", cuda)
        self.assertIn("__ldcs(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Four full-mask lane+1 shuffles fetch the next aligned block.
        self.assertEqual(cuda.count("__shfl_down_sync(0xffffffffu, value."), 12)
        self.assertIn("__shfl_down_sync(0xffffffffu, value.x, 1)", cuda)
        self.assertIn("__shfl_down_sync(0xffffffffu, value.w, 1)", cuda)
        # Register recombination selects own vs next-block word by (head+j).
        self.assertIn("const int g = head + j;", cuda)
        self.assertIn("(g < 8) ? own_w[g >> 1] : nxt[(g - 8) >> 1]", cuda)
        # Bit-exact bf16 -> fp32 widening of the packed word halves.
        self.assertIn("__bfloat162float(pair.y)", cuda)
        self.assertIn("__bfloat162float(pair.x)", cuda)
        self.assertIn("<cuda_bf16.h>", cuda)
        # head and out-registers are wired at the call site.
        self.assertIn("&s0", cuda)
        self.assertIn("&u7", cuda)

    def test_no_helper_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def plain(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16],
          n: ll.int32) -> ll.void:
    c: ll.int32 = ll.threadIdx_x()
    if c < n:
        y[c] = x[c]
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "export.json").write_text(
                json.dumps({
                    "kernel": "plain",
                    "function": "plain",
                    "inputs": [{"name": "x", "dtype": "bfloat16", "rank": 1}],
                    "output": {"name": "y", "dtype": "bfloat16", "shape": ["x[0]"]},
                    "scalars": [{"name": "n", "expr": "x[0]"}],
                    "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
                }))
            kernel_path = root / "kernel.py"
            kernel_path.write_text(source)
            spec = _load_spec(root / "export.json")
            kernel = _load_kernel(kernel_path, "plain")
            cuda = _render_cuda(spec, kernel)
        self.assertNotIn("gmem_ld_bf16x8_shifted_fn", cuda)
        self.assertNotIn("__shfl_down_sync(0xffffffffu, value.x, 1)", cuda)


if __name__ == "__main__":
    unittest.main()
