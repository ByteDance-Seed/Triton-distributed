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
"""CPU regression for rank-4 TMA descriptor manifests and 4-D bulk-tensor builtins.

Covers the shared rank-4 tensor-map capability used by warp-specialized
SM90/SM100 attention pipelines (a BSHD operand viewed innermost-out as
(D, S, H, B)) without a GPU:

* Manifest validation: ``export._spec._resolve`` must accept a rank-4
  TMA descriptor over a rank-4 tensor, resolve the dim3/stride3 host shape
  expressions, accept a 4-element partial-tile box, and default a per-batch
  rank-4 bundle's slice stride to ``dim3 * stride3`` (the contiguous
  leading-axis volume). It must keep rejecting rank-5 descriptors, unknown
  descriptor fields and mismatched box ranks.
* Host binding codegen: the rendered binding must encode a rank-4 map with
  ``agent_lk_make_tma_4d`` and carry the dim3/stride3/box3 arguments.
* Device builtins: ``tma_load_4d_1sm`` / ``tma_load_4d_1sm_mcast`` /
  ``tma_store_4d`` must emit the ``cp.async.bulk.tensor.4d`` PTX forms with
  four coordinate operands.

No CUDA toolchain or GPU is needed; the builtin codegens are consumed as
source text via the BuiltinOp ``codegen`` staticmethods.
"""

import json
import os
import unittest

from little_kernel.backends.nv.common.export import _bindings, _spec
from little_kernel.backends.nv.common.language.intrin import tma


def _manifest(tma_descriptors):
    return {
        "kernel": "lk_kernel",
        "inputs": [
            {"name": "q", "dtype": "bfloat16", "rank": 4},
        ],
        "output": {"name": "out", "dtype": "bfloat16", "shape": ["q[0]", "q[1]", "q[2]", "q[3]"]},
        "tma_descriptors": tma_descriptors,
        "launch": {"grid": [1, 1, 1], "block": [128, 1, 1]},
    }


class TestTmaRank4Spec(unittest.TestCase):

    def _bshd_descriptor(self, **extra):
        item = {
            "name": "qd",
            "tensor": "q",
            "rank": 4,
            # BSHD view innermost-out: (D, H, S, B). Strides in elements.
            "dim0": "q[3]",
            "dim1": "q[2]",
            "dim2": "q[1]",
            "dim3": "q[0]",
            "stride1": "q.stride(2)",
            "stride2": "q.stride(1)",
            "stride3": "q.stride(0)",
            "box": [64, 1, 128, 1],
            "swizzle_bytes": 128,
        }
        item.update(extra)
        return item

    def test_rank4_descriptor_resolves(self):
        spec = _spec._resolve(_manifest([self._bshd_descriptor()]))
        item = spec["tma_descriptors"][0]
        self.assertEqual(item["_rank"], 4)
        self.assertEqual(item["_box"], [64, 1, 128, 1])
        # Host C++ shape expressions must be filled for every rank-4 field.
        for field in ("dim0", "dim1", "dim2", "dim3", "stride1", "stride2", "stride3"):
            self.assertIn(f"_{field}_cpp", item)
            self.assertTrue(item[f"_{field}_cpp"])
        self.assertEqual(item["_swizzle"], "CU_TENSOR_MAP_SWIZZLE_128B")

    def test_rank4_batch_bundle_default_stride(self):
        # A per-batch bundle needs a leading batch axis beyond the four mapped
        # dims, so the tensor is rank 5 (leading batch over the BSHD volume).
        manifest = _manifest([self._bshd_descriptor(batch_count=2)])
        manifest["inputs"][0]["rank"] = 5
        manifest["output"]["shape"].append("q[4]")
        spec = _spec._resolve(manifest)
        item = spec["tma_descriptors"][0]
        self.assertTrue(item["_bundle"])
        # Contiguous leading-axis slice = full dim0*dim1*dim2 volume, i.e.
        # dim3 * stride3 elements.
        self.assertEqual(item["_batch_stride_cpp"], f"(({item['_dim3_cpp']}) * ({item['_stride3_cpp']}))")

    def test_rank5_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank must be 2, 3 or 4"):
            _spec._resolve(_manifest([self._bshd_descriptor(rank=5)]))

    def test_box_rank_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "box must have one dimension"):
            _spec._resolve(_manifest([self._bshd_descriptor(box=[64, 1, 128])]))

    def test_binding_emits_make_tma_4d(self):
        spec = _spec._resolve(_manifest([self._bshd_descriptor()]))
        source = _bindings._render_binding(spec)
        self.assertIn("agent_lk_make_tma_4d(&qd", source)
        self.assertIn("static CUresult agent_lk_make_tma_4d(", source)
        self.assertIn("encode(map, dtype, 4, ptr, dims, strides, box", source)


class TestTma4dBuiltins(unittest.TestCase):

    def test_load_4d_1sm_emits_bulk_tensor_4d(self):
        from little_kernel.language import builtin_base as bb
        codegen = getattr(tma.tma_load_4d_1sm, bb.CODEGEN_FUNC_ATTR)
        out = codegen("qd", "bar", "smem", 0, 1, 2, 3)
        self.assertIn("cp.async.bulk.tensor.4d.shared::cluster.global."
                      "mbarrier::complete_tx::bytes", out.body)
        for operand in ('"r"(c0)', '"r"(c1)', '"r"(c2)', '"r"(c3)'):
            self.assertIn(operand, out.body)
        self.assertIn("tma_load_4d_1sm_fn(&qd, bar, smem, 0, 1, 2, 3, 0)", out.return_val)

    def test_load_4d_1sm_mcast_emits_multicast_4d(self):
        out = tma.codegen_tma_load_4d_1sm_mcast("qd", "bar", "smem", 0, 1, 2, 3, 0x3)
        self.assertIn("cp.async.bulk.tensor.4d.shared::cluster.global.tile", out.body)
        self.assertIn(".multicast::cluster", out.body)
        self.assertIn("{%2, %3, %4, %5}", out.body)

    def test_store_4d_emits_bulk_tensor_4d(self):
        out = tma.codegen_tma_store_4d("od", "smem", 0, 1, 2, 3)
        self.assertIn("cp.async.bulk.tensor.4d.global.shared::cta.tile."
                      "bulk_group", out.body)
        self.assertIn("tma_store_4d_fn(&od, smem, 0, 1, 2, 3)", out.return_val)

    def test_builtins_registered_on_sm90_and_sm100(self):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        for arch in ("sm90", "sm100"):
            path = os.path.join(repo_root, "python", "little_kernel", "backends", "nv", arch, "backend.json")
            with open(path) as handle:
                builtins = json.load(handle)["language_exports"]
            for name in ("tma_load_4d_1sm", "tma_load_4d_1sm_mcast", "tma_store_4d"):
                self.assertIn(name, builtins, f"{name} missing on {arch}")
                self.assertEqual(builtins[name], f"little_kernel.backends.nv.common.language.intrin.tma:{name}")


if __name__ == "__main__":
    unittest.main()
