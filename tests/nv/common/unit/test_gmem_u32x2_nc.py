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
"""u32x2 (8-byte) read-only (.nc/__ldg) load twin: render-layer tests (no GPU).

Completes the register-pair u32x2 cache-policy set (default/_cg/_cs/_nc) so
the 8-byte dtype-generic vector load matches the _nc twins of the 16-byte
families (gmem_ld_v4_u32_nc, gmem_ld_u8x16_nc, v4f32/v4s32/v8f16/v8bf16).
A gathered read-only table with an 8-byte row stride — e.g. a packed int32
scale-factor table of shape [S, 2] (two packs per entry) whose dynamic rows
are selected by an index table, with duplicate indices allowed — could take
the 16-byte _nc path only by over-fetching two rows or the default .ca path
by giving up the read-only data cache; this twin closes that coverage hole.

  * gmem_ld_u32x2_nc moves 8 bytes into two uint32 registers through the
    read-only data cache (``__ldg`` on ``const uint2*``), legality contract
    identical to gmem_ld_v4_u32_nc: the loaded bytes are read-only for the
    whole launch, the frontend asserts no aliasing; bits moved are the same
    as the default twin, only the cache path differs;
  * the frontend-computed dynamic element offset (``row * 2`` for an
    int32 [S, 2] table) propagates to the aligned 8-byte access, with the
    8-byte alignment invariant stated in the docstring;
  * programs that use none of the u32x2 register-pair intrinsics render
    without the u32x2 helper functions.
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
def copy2_ro(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int32],
             nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        off: ll.int32 = c * 2
        r0: ll.uint32 = 0
        r1: ll.uint32 = 0
        s0: ll.uint32 = 0
        s1: ll.uint32 = 0
        t0: ll.uint32 = 0
        t1: ll.uint32 = 0
        ll.gmem_ld_u32x2_nc(x, off, r0, r1)
        ll.gmem_ld_u32x2(x, off, s0, s1)
        ll.gmem_ld_u32x2_cs(x, off, t0, t1)
        ll.gmem_st_u32x2(y, off, r0 ^ s0 ^ t0, r1 ^ s1 ^ t1)
'''

# Gather idiom: read-only packed scale-factor table [S, 2] of int32 (8-byte
# rows), dynamic row id from an index table, duplicates allowed; the same
# pattern the 16-byte u8x16 gather test uses, at 8-byte row granularity.
GATHER_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def gather2(table: ll.Tensor[ll.int32], idx: ll.Tensor[ll.int32],
            out: ll.Tensor[ll.int32], nrows: ll.int32) -> ll.void:
    r: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if r < nrows:
        row: ll.int32 = idx[r]
        r0: ll.uint32 = 0
        r1: ll.uint32 = 0
        ll.gmem_ld_u32x2_nc(table, row * 2, r0, r1)
        ll.gmem_st_u32x2(out, r * 2, r0, r1)
'''


def _spec(name):
    return {
        "kernel": name,
        "function": name,
        "inputs": [{"name": "x", "dtype": "int32", "rank": 1}],
        "output": {"name": "y", "dtype": "int32", "shape": ["x[0]"]},
        "scalars": [{"name": "nvec", "expr": "x[0]"}],
        "launch": {"grid": ["ceil_div(nvec, 256)", 1, 1], "block": [256, 1, 1]},
    }


def _gather_spec():
    return {
        "kernel": "gather2",
        "function": "gather2",
        "inputs": [
            {"name": "table", "dtype": "int32", "rank": 1},
            {"name": "idx", "dtype": "int32", "rank": 1},
        ],
        "output": {"name": "out", "dtype": "int32", "shape": ["idx[0]"]},
        "scalars": [{"name": "nrows", "expr": "idx[0]"}],
        "launch": {"grid": ["ceil_div(nrows, 256)", 1, 1], "block": [256, 1, 1]},
    }


class GmemU32x2NcTests(unittest.TestCase):

    def _render(self, directory, name, source, spec):
        root = Path(directory)
        (root / "export.json").write_text(json.dumps(spec))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        loaded_spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, name)
        return _render_cuda(loaded_spec, kernel)

    def test_language_namespace_exports_twin(self):
        self.assertTrue(hasattr(ll, "gmem_ld_u32x2_nc"))

    def test_nc_policy_renders(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "copy2_ro", SOURCE, _spec("copy2_ro"))
        # Read-only helper and __ldg body on const uint2*.
        self.assertIn("gmem_ld_u32x2_nc_fn(", cuda)
        self.assertIn("__ldg(reinterpret_cast<const uint2*>(ptr))", cuda)
        # Lane unpacking identical to the default twin: r0 = lane 0, r1 = 1.
        self.assertIn("*r0 = value.x; *r1 = value.y;", cuda)
        # The other policy twins coexist in the same kernel.
        self.assertIn("gmem_ld_u32x2_fn(", cuda)
        self.assertIn("gmem_ld_u32x2_cs_fn(", cuda)
        self.assertIn("__ldcs(reinterpret_cast<const uint2*>(ptr))", cuda)
        # Store twin pairs bit-exactly.
        self.assertIn("gmem_st_u32x2_fn(", cuda)

    def test_gather_idiom_dynamic_offsets_render(self):
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "gather2", GATHER_SOURCE, _gather_spec())
        # Frontend-computed dynamic row offsets propagate to the aligned
        # 8-byte read-only access: index arithmetic stays in the source.
        self.assertIn("__ldg(reinterpret_cast<const uint2*>(ptr))", cuda)
        self.assertIn("gmem_ld_u32x2_nc_fn(", cuda)
        self.assertIn("row * 2", cuda)

    def test_no_helpers_when_unused(self):
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy2_ro(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int32],
             nvec: ll.int32) -> ll.void:
    c: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if c < nvec:
        y[c] = x[c]
'''
        with tempfile.TemporaryDirectory() as directory:
            cuda = self._render(directory, "copy2_ro", source, _spec("copy2_ro"))
        self.assertNotIn("u32x2", cuda)


if __name__ == "__main__":
    unittest.main()
