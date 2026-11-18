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
"""TMA descriptor box-legality compile-time validation tests (no GPU required).

Locks in the capability ``ir.tma.scale_descriptor``: manifest-declared TMA
descriptors are checked against the two driver legality rules that apply to
every ``agent_lk_make_tma_*`` emission (all of which encode with
``CU_TENSOR_MAP_INTERLEAVE_NONE``):

* the innermost box extent must cover whole 16-byte TMA atoms, which puts a
  dtype-dependent floor on ``box[0]`` — 4 elements for float32/int32, 8 for
  float16/bfloat16, 16 for the 1-byte dtypes.  A float32 small-box no-swizzle
  form with a 4-column inner box over a rank-2 scale tensor is exactly one
  atom and is legal (this is the dual-K GEMM A_scale descriptor form);
* with swizzling enabled the innermost box byte extent must fit the selected
  swizzle span (32/64/128 bytes).

Both rules previously surfaced only as an opaque ``cuTensorMapEncodeTiled``
failure at kernel launch; they are now precise compile-time ``ValueError``s,
and the error names the offending descriptor.

These tests validate manifests and render binding text only; nothing compiles
CUDA or touches a device.
"""

import unittest

from little_kernel.backends.nv.common.export._bindings import _render_binding
from little_kernel.backends.nv.common.export._validation import validate_spec


def _manifest(box, dtype="float32", swizzle_bytes=0, rank=2):
    """A rank-2 descriptor over a 2-D tensor of the given dtype, with the
    ``box``/``swizzle_bytes`` under test (the A_scale staging form)."""
    descriptor = {"name": "ds", "tensor": "A_scale", "rank": rank, "box": box, "swizzle_bytes": swizzle_bytes}
    if rank == 2:
        descriptor.update({"inner": "A_scale[1]", "outer": "A_scale[0]", "outer_stride": "A_scale[1]"})
    else:
        descriptor.update({
            "dim0": "A_scale[1]", "dim1": "A_scale[0]", "dim2": 1, "stride1": "A_scale[1]", "stride2":
            "A_scale[0]*A_scale[1]"
        })
    return {
        "function": "fwd",
        "inputs": [{"name": "A_scale", "dtype": dtype, "rank": 2}],
        "kernel": "kernel",
        "kernel_args": ["ds"],
        "launch": {"block": [128, 1, 1], "grid": [1, 1, 1]},
        "tma_descriptors": [descriptor],
        "outputs": [{"name": "C", "dtype": dtype, "shape": ["A_scale[0]", "A_scale[1]"]}],
    }


def _validate(manifest):
    return validate_spec(manifest).legacy_mapping()


class TestTmaBoxLegality(unittest.TestCase):

    def test_f32_small_box_no_swizzle_accepted_and_emitted(self):
        """The capability reproducer: rank-2 float32 descriptor with a
        4-column inner box and swizzle_bytes=0 (one 16-byte TMA atom)
        validates and the host binding emits the grid_constant descriptor
        encode."""
        view = _validate(_manifest([4, 128], dtype="float32", swizzle_bytes=0))
        binding = _render_binding(view)
        self.assertIn("agent_lk_make_tma_2d(&ds, A_scale.data_ptr()", binding)
        self.assertIn("CU_TENSOR_MAP_SWIZZLE_NONE", binding)
        self.assertIn("CU_TENSOR_MAP_DATA_TYPE_FLOAT32", binding)

    def test_whole_atom_inner_boxes_accepted(self):
        """Inner boxes spanning whole 16-byte atoms stay legal for every
        dtype."""
        cases = [
            ("float32", [4, 128]),  # one 16-byte atom, the capability form
            ("float32", [16, 32]),  # four atoms
            ("int32", [4, 128]),
            ("bfloat16", [8, 128]),
            ("float16", [8, 128]),
            ("uint8", [16, 128]),
            ("float8_e4m3fn", [16, 128]),
        ]
        for dtype, box in cases:
            with self.subTest(dtype=dtype, box=box):
                _validate(_manifest(box, dtype=dtype, swizzle_bytes=0))

    def test_sub_atom_inner_boxes_rejected(self):
        """Inner boxes that do not cover whole 16-byte TMA atoms fail at
        compile time with a precise, descriptor-naming error."""
        cases = [("float32", [3, 128]),  # 12 bytes: not a whole atom
                 ("int32", [1, 128]),  # 4 bytes
                 ("bfloat16", [4, 128]),  # 8 bytes
                 ("float16", [7, 128]),  # 14 bytes
                 ("uint8", [8, 128]),  # 8 bytes
                 ]
        for dtype, box in cases:
            with self.subTest(dtype=dtype, box=box):
                with self.assertRaisesRegex(ValueError, r"TMA descriptor ds: inner box .* "
                                            r"not a multiple of 16 bytes"):
                    _validate(_manifest(box, dtype=dtype, swizzle_bytes=0))

    def test_swizzle_span_fit_accepted(self):
        """An inner box exactly filling the swizzle span is legal (bf16
        swizzle64 over a 64-byte inner box of 32 elements)."""
        _validate(_manifest([32, 32], dtype="bfloat16", swizzle_bytes=64))

    def test_inner_box_exceeding_swizzle_span_rejected(self):
        """A swizzled inner box wider than the selected swizzle span fails
        at compile time (bf16 swizzle64 over a 256-byte inner box)."""
        with self.assertRaisesRegex(
                ValueError, r"TMA descriptor ds: inner box of 256 bytes exceeds the "
                r"selected swizzle span of 64 bytes"):
            _validate(_manifest([128, 32], dtype="bfloat16", swizzle_bytes=64))

    def test_no_swizzle_skips_span_check(self):
        """With swizzle_bytes=0 there is no span cap: a wide unswizzled
        inner box is constrained only by the 16-byte atom rule."""
        _validate(_manifest([64, 16], dtype="float32", swizzle_bytes=0))

    def test_legality_applies_to_rank3_descriptors(self):
        """The same two rules guard rank-3 descriptor boxes (box[0] is still
        the innermost extent)."""
        _validate(_manifest([4, 8, 1], dtype="float32", swizzle_bytes=0, rank=3))
        with self.assertRaisesRegex(ValueError, "not a multiple of 16 bytes"):
            _validate(_manifest([2, 8, 1], dtype="float32", swizzle_bytes=0, rank=3))


if __name__ == "__main__":
    unittest.main()
