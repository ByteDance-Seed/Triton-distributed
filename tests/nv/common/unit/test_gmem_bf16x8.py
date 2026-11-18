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
"""bf16x8 (16-byte) typed vector GMEM twins with cache-policy variants.

Render-layer legality tests (no GPU) for the policy set contract.  These
tests consume the shared executable RF12 fixtures in
``examples/nv/common/testing/vector_fixtures.py`` through the public compile API
(``PASSES['cuda']`` + ``codegen_cuda``) instead of the private evaluator
parser/renderer helpers, and they pin the corrected geometry: ``nvec`` is a
*vector* count (eight BF16 lanes per vector), not the element count.

  * gmem_ld_bf16x8[/_cg/_streaming/_nc] move 16 bytes into two uint64 register
    bit patterns (r0 = elements 0-3, r1 = elements 4-7);
  * gmem_st_bf16x8[/_cg/_streaming] RNE-pack eight FP32 values into eight BF16
    and store one aligned 16-byte block;
  * vec_bf16x8_lane(lo, hi, k) extracts compile-time literal lane k in 0..7 as
    fp32 and rejects dynamic lanes;
  * the widening f32x8 _nc/_cs twins fetch the same 16 bytes through __ldg /
    __ldcs and widen every lane with __bfloat162float.
"""

import sys
import unittest
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "examples/nv/common/testing").is_dir():
        if str(_parent) not in sys.path:
            sys.path.insert(0, str(_parent))
        break

from examples.nv.common.testing import vector_fixtures as vf
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.language.intrin.memory import codegen_vec_bf16x8_lane


def _compile(kernel):
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=32)


class GmemBf16x8Tests(unittest.TestCase):

    def test_vector_count_geometry_is_in_bounds(self):
        for geometry in vf.BF16X8_CASES:
            self.assertEqual(geometry.nvec, vf.ceil_div(geometry.nelem, geometry.lanes))
            self.assertTrue(geometry.vector_bounds_are_safe())

    def test_all_policies_render(self):
        cuda = _compile(vf.copy_bf16x8_roundtrip)
        # Load helper: default opaque dereference.
        self.assertIn("gmem_ld_bf16x8_fn(", cuda)
        self.assertIn("*reinterpret_cast<const uint4*>(ptr)", cuda)
        # Store helpers: RNE pack + default/.cg/.cs stores.
        self.assertIn("gmem_st_bf16x8_fn(", cuda)
        self.assertIn("gmem_st_bf16x8_cg_fn(", cuda)
        self.assertIn("gmem_st_bf16x8_streaming_fn(", cuda)
        self.assertIn("__floats2bfloat162_rn(f0, f1)", cuda)
        self.assertIn("__stcg(reinterpret_cast<uint4*>(ptr), value)", cuda)
        self.assertIn("__stcs(reinterpret_cast<uint4*>(ptr), value)", cuda)
        # Lane extractor and its bf16 include.
        self.assertIn("vec_bf16x8_lane_fn(", cuda)
        self.assertIn("<cuda_bf16.h>", cuda)
        # Out-param wiring: address-of locals at the load call site.
        self.assertIn("&lo", cuda)

    def test_masked_tail_uses_scalar_rne_store(self):
        cuda = _compile(vf.copy_bf16x8_masked_tail)
        # Full vectors use the pack helper; the ragged tail uses the scalar
        # BF16 RNE store, never a second vector transaction.
        self.assertIn("if (tid + 1 < nvec)", cuda)
        self.assertIn("if (idx < nelem)", cuda)
        self.assertIn("st_global_cs_bf16_f32_rn_fn(", cuda)
        self.assertEqual(cuda.count("gmem_st_bf16x8_fn(dst + off"), 1)

    def test_no_include_when_unused(self):
        import little_kernel as lk
        import little_kernel.language as ll

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def scalar_only(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16], n: ll.int32) -> ll.void:
            c: ll.int32 = ll.threadIdx_x()
            if c < n:
                y[c] = x[c]

        cuda = scalar_only.compile(PASSES["cuda"], codegen_cuda, num_threads=32)
        self.assertNotIn("bf16x8", cuda)

    def test_lane_literal_legality(self):
        for lane in ("0", "7"):
            builtin = codegen_vec_bf16x8_lane("lo", "hi", lane)
            self.assertIn(f", {lane})", builtin.return_val)
        for bad in ("8", "-1", "k", "lane + 1"):
            with self.assertRaises(ValueError):
                codegen_vec_bf16x8_lane("lo", "hi", bad)


class GmemLdBf16x8F32x8PolicyTests(unittest.TestCase):
    """Cache-policy twins of the FP32-lane widening load.

    ll.gmem_ld_bf16x8_f32x8_nc / _cs emit the same 16-byte bf16x8 fetch +
    __bfloat162float widening as the plain form, but the vector fetch goes
    through __ldg (read-only) or __ldcs (evict-first streaming) instead of a
    plain dereference.
    """

    def test_policy_twins_render(self):
        cuda = _compile(vf.copy_bf16x8_f32x8_policy)
        # Both twin helpers are emitted and called.
        self.assertIn("gmem_ld_bf16x8_f32x8_nc_fn(", cuda)
        self.assertIn("gmem_ld_bf16x8_f32x8_cs_fn(", cuda)
        # The .nc twin fetches via __ldg, the .cs twin via __ldcs.
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        self.assertIn("__ldcs(reinterpret_cast<const uint4*>(ptr))", cuda)
        # Both still widen every bf16 lane to FP32 and carry the bf16 include.
        self.assertIn("__bfloat162float(p[0].x)", cuda)
        self.assertIn("__bfloat162float(p[3].y)", cuda)
        self.assertIn("<cuda_bf16.h>", cuda)
        # Out-param wiring: address-of FP32 locals at the call site.
        self.assertIn("&a0", cuda)
        self.assertIn("&b7", cuda)


if __name__ == "__main__":
    unittest.main()
