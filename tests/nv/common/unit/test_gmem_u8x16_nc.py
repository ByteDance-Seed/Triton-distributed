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
"""u8x16 read-only (.nc/__ldg) load twin: render-layer legality tests (no GPU).

Completes the u8x16 cache-policy set (default/_cg/_streaming/_nc) so the
16-byte opaque move family for one-byte storage dtypes (fp8_e4m3/fp8_e5m2/
uint8/int8) matches the _nc twins of v4f32/v4s32/v8f16/v8bf16/f64x2. The
motivating pattern is a bit-exact gather from a table that is read-only for
the whole launch (duplicate row ids re-read the same lines): the frontend
asserts the read-only contract, the compiler performs no aliasing analysis.

  * gmem_ld_u8x16_nc moves 16 bytes into two uint64 register bit patterns
    through __ldg with NO numeric conversion; the packing (r0 = bytes 0-7,
    r1 = bytes 8-15) is identical to gmem_ld_u8x16, so an ld.nc -> st pair
    is a bit-exact copy;
  * index is in elements (== bytes) and stays frontend-computed, so dynamic
    row/layout arithmetic propagates to the aligned 16-byte access without
    any IR-level layout metadata;
  * programs that use none of the u8x16 intrinsics render without the u8x16
    helper functions.
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
def copy16_ro(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.uint8],
              nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        off: ll.int32 = c * 16
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        ll.gmem_ld_u8x16_nc(x, off, lo, hi)
        ll.gmem_st_u8x16(y, off, lo, hi)
        b0: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 0)
        b15: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 15)
        y[off] = (b0 ^ b15) % 256
'''

# Gather idiom: read-only source table rows (dynamic row id from an index
# table), 16-byte opaque moves with a frontend-computed, provably aligned
# element offset (row * 16).  This is the dependent-tasks' core pattern.
GATHER_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def gather16(table: ll.Tensor[ll.uint8], idx: ll.Tensor[ll.int32],
             out: ll.Tensor[ll.uint8], nrows: ll.int32) -> ll.void:
    r: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if r < nrows:
        row: ll.int32 = idx[r]
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        ll.gmem_ld_u8x16_nc(table, row * 16, lo, hi)
        ll.gmem_st_u8x16(out, r * 16, lo, hi)
'''


def _spec(name, extra_inputs=()):
    inputs = [{"name": "x", "dtype": "uint8", "rank": 1}]
    if extra_inputs:
        inputs = list(extra_inputs)
    return {
        "kernel": name,
        "function": name,
        "inputs": inputs,
        "output": {"name": "y", "dtype": "uint8", "shape": ["x[0]"]},
        "scalars": [{"name": "nvec", "expr": "x[0]"}],
        "launch": {"grid": ["ceil_div(nvec, 256)", 1, 1], "block": [256, 1, 1]},
    }


def _gather_spec():
    return {
        "kernel": "gather16",
        "function": "gather16",
        "inputs": [
            {"name": "table", "dtype": "uint8", "rank": 1},
            {"name": "idx", "dtype": "int32", "rank": 1},
        ],
        "output": {"name": "out", "dtype": "uint8", "shape": ["idx[0]"]},
        "scalars": [{"name": "nrows", "expr": "idx[0]"}],
        "launch": {"grid": ["ceil_div(nrows, 256)", 1, 1], "block": [256, 1, 1]},
    }


class GmemU8x16NcTests(unittest.TestCase):

    def _render(self, directory, name, source, spec):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(spec))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        loaded_spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, name)
        return _render_cuda(loaded_spec, kernel)

    def test_language_namespace_exports_twin(self):
        self.assertTrue(hasattr(ll, "gmem_ld_u8x16_nc"))

    def test_nc_policy_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "copy16_ro", SOURCE, _spec("copy16_ro"))
        # Read-only helper and __ldg body.
        self.assertIn("gmem_ld_u8x16_nc_fn(", cuda)
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Packing identical to the default twin: lo = lanes 0-7, hi = 8-15.
        self.assertIn("*r0 = (uint64_t)value.x | ((uint64_t)value.y << 32);", cuda)
        self.assertIn("*r1 = (uint64_t)value.z | ((uint64_t)value.w << 32);", cuda)
        # Store twin pairs bit-exactly, and the lane extractor still works.
        self.assertIn("gmem_st_u8x16_fn(", cuda)
        self.assertIn("vec_u8x16_lane_fn(", cuda)
        # No numeric conversion anywhere: no fp8/half headers or cvt asm.
        self.assertNotIn("cvt.rn.satfinite", cuda)
        self.assertNotIn("<cuda_fp8.h>", cuda)
        # Only the policies actually used are emitted.
        self.assertNotIn("gmem_ld_u8x16_cg_fn(", cuda)
        self.assertNotIn("gmem_ld_u8x16_streaming_fn(", cuda)

    def test_gather_idiom_dynamic_offsets_render(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "gather16", GATHER_SOURCE, _gather_spec())
        # Frontend-computed dynamic row offsets propagate to the aligned
        # 16-byte read-only access: index arithmetic stays in the source.
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        self.assertIn("gmem_ld_u8x16_nc_fn(", cuda)
        self.assertIn("row * 16", cuda)

    def test_no_helpers_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy16_ro(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.uint8],
              nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        y[c] = x[c]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "copy16_ro", source, _spec("copy16_ro"))
        self.assertNotIn("u8x16", cuda)


if __name__ == "__main__":
    unittest.main()
