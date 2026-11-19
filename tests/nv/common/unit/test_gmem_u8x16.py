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
"""u8x16 (16-byte) opaque vector GMEM moves for one-byte element dtypes.

Render-layer legality tests (no GPU) for the policy set contract.  These
tests consume the shared executable RF12 fixtures in
``examples/nv/common/testing/vector_fixtures.py`` through the public compile API
(``PASSES['cuda']`` + ``codegen_cuda``) instead of the private evaluator
parser/renderer helpers (``_load_kernel`` / ``_load_spec`` /
``_render_cuda``), and they pin the corrected geometry:

  * ``nvec`` is a *vector* count (one 16-byte vector per iteration), not the
    element count; the offset multiplies by 16, so bounds are expressed in
    vectors (see ``docs/behavioral-fixtures.md``);
  * gmem_ld_u8x16[/_cg/_streaming/_nc] move 16 bytes into two uint64 register
    bit patterns (r0 = elements 0-7, r1 = elements 8-15), policy twins moving
    bits identical to the default load;
  * gmem_st_u8x16[/_cg/_streaming] store those two uint64 bit patterns as one
    aligned 16-byte block, so an ld -> st pair is a bit-exact copy;
  * vec_u8x16_lane(lo, hi, k) extracts compile-time literal lane k in 0..15 as
    the raw byte widened to uint32 and rejects dynamic lanes.
"""

import sys
import unittest
from pathlib import Path

# Locate the repository root (which holds examples/nv/common/testing) regardless of
# whether this file is at python/little_kernel/tests/unit or the migrated
# tests/unit location.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "examples/nv/common/testing").is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from examples.nv.common.testing import vector_fixtures as vf
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.language.intrin.memory import codegen_vec_u8x16_lane


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=32)


class GmemU8x16Tests(unittest.TestCase):

    def test_vector_count_geometry_is_in_bounds(self):
        # The reviewed defect used the element count as nvec while offsetting
        # by 16.  The shared geometry describes vector counts with explicit
        # tail policy and proves every vector is in bounds.
        for geometry in vf.U8X16_CASES:
            self.assertEqual(geometry.nvec, vf.ceil_div(geometry.nelem, geometry.lanes))
            self.assertTrue(geometry.vector_bounds_are_safe())

    def test_all_policies_render(self):
        cuda = _compile(vf.copy_u8x16_policies)
        # Load helpers: default dereference, .cg, and .cs policy variants.
        self.assertIn("gmem_ld_u8x16_fn(", cuda)
        self.assertIn("gmem_ld_u8x16_cg_fn(", cuda)
        self.assertIn("gmem_ld_u8x16_streaming_fn(", cuda)
        self.assertIn("__ldcg(reinterpret_cast<const uint4*>(ptr))", cuda)
        self.assertIn("__ldcs(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Store helpers: opaque uint64-pair repack + default/.cg/.cs stores.
        self.assertIn("gmem_st_u8x16_fn(", cuda)
        self.assertIn("gmem_st_u8x16_cg_fn(", cuda)
        self.assertIn("gmem_st_u8x16_streaming_fn(", cuda)
        self.assertIn("(uint32_t)(r0 >> 32)", cuda)
        self.assertIn("__stcg(reinterpret_cast<uint4*>(ptr), value)", cuda)
        self.assertIn("__stcs(reinterpret_cast<uint4*>(ptr), value)", cuda)
        # r0 carries bytes 0-7, r1 bytes 8-15 (packed lane order).
        self.assertIn("(uint64_t)value.x | ((uint64_t)value.y << 32)", cuda)
        # No numeric conversion anywhere: no fp8/half headers or cvt asm.
        self.assertNotIn("cvt.rn.satfinite", cuda)
        self.assertNotIn("<cuda_fp8.h>", cuda)

    def test_nc_policy_renders(self):
        cuda = _compile(vf.copy_u8x16_nc)
        self.assertIn("gmem_ld_u8x16_nc_fn(", cuda)
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        self.assertNotIn("gmem_ld_u8x16_cg_fn(", cuda)
        self.assertNotIn("gmem_ld_u8x16_streaming_fn(", cuda)

    def test_lane_extractor_and_out_param_wiring(self):
        cuda = _compile(vf.u8x16_lane_canary)
        # Lane extractor and out-param wiring (address-of locals, element
        # index on the u8 pointer at the call site).
        self.assertIn("vec_u8x16_lane_fn(", cuda)
        self.assertIn("&lo", cuda)
        self.assertIn("&hi", cuda)
        # All sixteen literal lanes are extracted in memory order.
        lanes = sorted(int(token) for token in __import__("re").findall(r"vec_u8x16_lane_fn\(lo, hi, (\d+)\)", cuda))
        self.assertEqual(lanes, list(range(16)))

    def test_masked_tail_never_uses_a_partial_vector(self):
        cuda = _compile(vf.copy_u8x16_masked_tail)
        # The full-vector branch is guarded against the ragged last vector;
        # the tail copies scalar bytes only.
        self.assertIn("if (tid + 1 < nvec)", cuda)
        self.assertIn("if (idx < nelem)", cuda)
        self.assertEqual(cuda.count("gmem_st_u8x16_fn(dst + off"), 1)
        self.assertIn("x[idx]", cuda)

    def test_no_helpers_when_unused(self):
        import little_kernel as lk
        import little_kernel.language as ll

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def scalar_only(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.uint8], n: ll.int32) -> ll.void:
            c: ll.int32 = ll.threadIdx_x()
            if c < n:
                y[c] = x[c]

        cuda = scalar_only.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertNotIn("u8x16", cuda)

    def test_lane_literal_legality(self):
        for lane in ("0", "7", "8", "15"):
            builtin = codegen_vec_u8x16_lane("lo", "hi", lane)
            self.assertIn(f", {lane})", builtin.return_val)
        for bad in ("16", "-1", "k", "lane + 1"):
            with self.assertRaises(ValueError):
                codegen_vec_u8x16_lane("lo", "hi", bad)


if __name__ == "__main__":
    unittest.main()
