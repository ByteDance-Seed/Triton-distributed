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
"""Read-only (.nc/__ldg) twin of gmem_ld_bf16x8_f32x8: render-layer tests.

CPU-only legality coverage (no GPU), mirroring test_gmem_bf16x8 and
test_gmem_u8x16_nc:

  * gmem_ld_bf16x8_f32x8_nc loads 8 contiguous 16-byte-aligned bf16 lanes
    through __ldg(reinterpret_cast<const uint4*>(ptr)) — the LDG.CONSTANT
    payload policy — and widens each __nv_bfloat162 lane with exact
    __bfloat162float into eight FP32 registers (same deliverable dtype and
    out-param wiring as the default gmem_ld_bf16x8_f32x8);
  * the default (non-nc) path stays bit-identical to the historical helper
    (plain dereference, unsuffixed helper name), so existing frontends are
    unaffected;
  * the read-only legality contract matches the gmem_ld_*_nc family: the
    loaded bytes must not be written by any thread for the duration of the
    launch; the frontend asserts it, the compiler does no aliasing analysis;
  * factory rejects unknown 16-bit typed load policies at codegen time
    rather than emitting invalid C++.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda
from little_kernel.backends.nv.common.language.intrin.memory import _gen_gmem_ld_16bit

NC_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy8_nc(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16],
             nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        off: ll.int32 = c * 8
        f0: ll.float32 = 0.0
        f1: ll.float32 = 0.0
        f2: ll.float32 = 0.0
        f3: ll.float32 = 0.0
        f4: ll.float32 = 0.0
        f5: ll.float32 = 0.0
        f6: ll.float32 = 0.0
        f7: ll.float32 = 0.0
        ll.gmem_ld_bf16x8_f32x8_nc(x, off, f0, f1, f2, f3, f4, f5, f6, f7)
        ll.gmem_st_bf16x8(y, off, f0, f1, f2, f3, f4, f5, f6, f7)
'''

DEFAULT_SOURCE = NC_SOURCE.replace("ll.gmem_ld_bf16x8_f32x8_nc(x, off, f0, f1, f2, f3, f4, f5, f6, f7)",
                                   "ll.gmem_ld_bf16x8_f32x8(x, off, f0, f1, f2, f3, f4, f5, f6, f7)")

MIXED_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy8_nc(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16],
             nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        off: ll.int32 = c * 8
        f0: ll.float32 = 0.0
        f1: ll.float32 = 0.0
        f2: ll.float32 = 0.0
        f3: ll.float32 = 0.0
        f4: ll.float32 = 0.0
        f5: ll.float32 = 0.0
        f6: ll.float32 = 0.0
        f7: ll.float32 = 0.0
        g0: ll.float32 = 0.0
        g1: ll.float32 = 0.0
        g2: ll.float32 = 0.0
        g3: ll.float32 = 0.0
        g4: ll.float32 = 0.0
        g5: ll.float32 = 0.0
        g6: ll.float32 = 0.0
        g7: ll.float32 = 0.0
        ll.gmem_ld_bf16x8_f32x8_nc(x, off, f0, f1, f2, f3, f4, f5, f6, f7)
        ll.gmem_ld_bf16x8_f32x8(x, off + 8, g0, g1, g2, g3, g4, g5, g6, g7)
        ll.gmem_st_bf16x8(y, off, f0 + g0, f1 + g1, f2 + g2, f3 + g3,
                          f4 + g4, f5 + g5, f6 + g6, f7 + g7)
'''


def _spec():
    return {
        "kernel": "copy8_nc",
        "function": "copy8_nc",
        "inputs": [{"name": "x", "dtype": "bfloat16", "rank": 1}],
        "output": {"name": "y", "dtype": "bfloat16", "shape": ["x[0]"]},
        "scalars": [{"name": "nvec", "expr": "x[0]"}],
        "launch": {"grid": ["ceil_div(nvec, 256)", 1, 1], "block": [256, 1, 1]},
    }


class GmemLdBf16x8F32x8NcTests(unittest.TestCase):

    def _render(self, directory, source=NC_SOURCE):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(_spec()))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, "copy8_nc")
        return _render_cuda(spec, kernel)

    def test_language_namespace_exports_twin(self):
        self.assertTrue(hasattr(ll, "gmem_ld_bf16x8_f32x8_nc"))

    def test_nc_policy_renders_ldg(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory)
        # Read-only helper under its own suffixed name.
        self.assertIn("gmem_ld_bf16x8_f32x8_nc_fn(", cuda)
        # LDG.CONSTANT policy via __ldg on the vector load.
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Same exact fp32 widening and fp16-free include set as the default.
        self.assertIn("__bfloat162float", cuda)
        self.assertIn("<cuda_bf16.h>", cuda)
        self.assertNotIn("<cuda_fp16.h>", cuda)
        self.assertNotIn("__half2float", cuda)
        # Out-param wiring identical to the default twin: address-of locals.
        self.assertIn("&f0", cuda)
        # One helper definition + one call site.
        self.assertEqual(cuda.count("gmem_ld_bf16x8_f32x8_nc_fn("), 2)
        # The plain default load helper must not leak in.
        self.assertNotIn("gmem_ld_bf16x8_f32x8_fn(", cuda)
        # Store twin pairs without any extra policy.
        self.assertIn("gmem_st_bf16x8_fn(", cuda)

    def test_default_path_unchanged(self):
        """The non-nc twin renders bit-identically to the historical helper."""
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, DEFAULT_SOURCE)
        self.assertIn("gmem_ld_bf16x8_f32x8_fn(", cuda)
        self.assertIn("*reinterpret_cast<const uint4*>(ptr)", cuda)
        self.assertNotIn("__ldg", cuda)
        self.assertNotIn("gmem_ld_bf16x8_f32x8_nc_fn(", cuda)

    def test_both_variants_coexist(self):
        """Default and .nc spellings in one kernel emit disjoint helpers."""
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, MIXED_SOURCE)
        self.assertIn("gmem_ld_bf16x8_f32x8_fn(", cuda)
        self.assertIn("gmem_ld_bf16x8_f32x8_nc_fn(", cuda)
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Exactly one __ldg site (the .nc load) — the default stays plain.
        self.assertEqual(cuda.count("__ldg(reinterpret_cast"), 1)
        self.assertEqual(cuda.count("gmem_ld_bf16x8_f32x8_nc_fn("), 2)
        self.assertEqual(cuda.count("gmem_ld_bf16x8_f32x8_fn("), 2)

    def test_no_helpers_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy8_nc(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16],
             nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        y[c] = x[c]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, source)
        self.assertNotIn("bf16x8_f32x8", cuda)

    def test_factory_rejects_unknown_policy(self):
        with self.assertRaises(ValueError):
            _gen_gmem_ld_16bit("gmem_ld_bf16x8_f32x8_bogus", "__nv_bfloat16", "uint4", 8, "__bfloat162float",
                               load_policy="bogus")


if __name__ == "__main__":
    unittest.main()
