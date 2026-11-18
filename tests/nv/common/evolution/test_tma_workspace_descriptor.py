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
"""CPU compile regression for TMA descriptors over workspace tensors.

Covers the export-layer contract that a ``tma_descriptors`` record may name a
workspace tensor (not only model inputs/outputs), so a host-encoded
CUtensorMap bundle can reference freshly allocated transposed storage after
a transpose kernel.  The pattern is the per-expert B-weight bundle used in
per-expert grouped GEMM: a rank-3 fp8 workspace ``wT`` with shape
``[E, N, K]`` written by a transpose kernel, then one rank-2 TMA descriptor
per leading expert slice staged to device memory.

No CUDA toolchain or GPU is needed; the test consumes the resolved manifest
view and the rendered pybind11 binding as source text:

* Validation (``_resolve`` via ``validate_spec``): the workspace-backed
  descriptor validates with the same rules as an input-backed one -- dtype
  and rank recorded from the workspace's ``tensor_types``/``ranks`` entry,
  batch count honored as a bundle, extents/strides resolved from the input
  shape expressions.
* Codegen (``_render_binding``): the workspace ``at::empty`` allocation and
  its zero-init ``cudaMemsetAsync`` are emitted before the descriptor
  encode loop, and each per-batch encode drives its base pointer from the
  workspace allocation's ``data_ptr()`` offset by ``batch * batch_stride``.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from little_kernel.backends.nv.common.export._validation import validate_spec
from little_kernel.backends.nv.common.export._spec import _resolve
from little_kernel.backends.nv.common.export._bindings import _render_binding


def _manifest():
    """Exact capability manifest: batched descriptor over a zero-init workspace."""
    return {
        "function":
        "f",
        "kernel":
        "f",
        "inputs": [
            {"name": "w_q", "dtype": "float8_e4m3fn", "rank": 3},
        ],
        "outputs": [
            {"name": "out", "dtype": "bfloat16", "shape": ["w_q[0]", "w_q[2]"]},
        ],
        "workspaces": [
            {"name": "wT", "dtype": "float8_e4m3fn", "shape": ["w_q[0]", "w_q[2]", "w_q[1]"], "zero": True},
        ],
        "tma_descriptors": [
            {
                "name": "dB", "tensor": "wT", "rank": 2, "inner": "w_q[1]", "outer": "w_q[2]", "outer_stride": "w_q[1]",
                "batch_count": "w_q[0]", "box": [128, 128], "swizzle_bytes": 128
            },
        ],
        "launch": {"grid": [1, 1, 1], "block": [32, 1, 1]},
    }


class TestTmaWorkspaceDescriptorValidation(unittest.TestCase):

    def test_validate_spec_accepts_workspace_backed_descriptor(self):
        # The whole point of the capability: validation must not reject the
        # manifest merely because the descriptor tensor is a workspace.
        validate_spec(_manifest())

    def test_resolved_descriptor_facts(self):
        view = _resolve(dict(_manifest()))
        (item, ) = view["tma_descriptors"]
        self.assertEqual(item["tensor"], "wT")
        self.assertEqual(item["_dtype"], "float8_e4m3fn")
        self.assertTrue(item["_bundle"])
        self.assertEqual(item["_inner_cpp"], "w_q.size(1)")
        self.assertEqual(item["_outer_cpp"], "w_q.size(2)")
        self.assertEqual(item["_outer_stride_cpp"], "w_q.size(1)")
        self.assertEqual(item["_batch_count_cpp"], "w_q.size(0)")
        # Contiguous leading-axis layout: one expert slice spans outer*stride.
        self.assertEqual(item["_batch_stride_cpp"], "((w_q.size(2)) * (w_q.size(1)))")


class TestTmaWorkspaceDescriptorCodegen(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.binding = _render_binding(_resolve(dict(_manifest())))

    def test_workspace_allocated_and_zeroed_before_encode(self):
        alloc = self.binding.index("torch::Tensor wT = at::empty(")
        zero = self.binding.index("cudaMemsetAsync(wT.data_ptr()")
        # Anchor on the per-batch encode *call* (it takes &dB_host[lk_b]),
        # not the static agent_lk_make_tma_2d helper defined earlier.
        encode = self.binding.index("agent_lk_make_tma_2d(&dB_host[lk_b],")
        self.assertLess(alloc, zero)
        self.assertLess(zero, encode)

    def test_zero_init_uses_workspace_stream_and_size(self):
        self.assertIn("cudaMemsetAsync(wT.data_ptr(), 0, "
                      "wT.numel() * wT.element_size(), lk_stream)", self.binding)

    def test_encode_base_pointer_drives_from_workspace(self):
        # Per-batch encode: base = workspace data_ptr + batch * stride.
        self.assertIn("static_cast<char*>((wT).data_ptr()) + "
                      "lk_b * (((w_q.size(2)) * (w_q.size(1)))) * 1", self.binding)

    def test_bundle_staged_to_device_and_freed(self):
        self.assertIn("std::vector<CUtensorMap> dB_host", self.binding)
        self.assertIn("CUtensorMap* dB = nullptr", self.binding)
        self.assertIn("cudaMemcpy(dB, dB_host.data()", self.binding)
        self.assertIn("cudaFree(dB)", self.binding)


if __name__ == "__main__":
    unittest.main()
