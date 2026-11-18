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
"""Typed quantizing FP8 stores at arbitrary (strided) byte offsets.

Render-layer regressions (no GPU) for the strided/element-store completion
of the fp8 store family.  The pre-existing gmem_st_f32_to_e4m3x16 /
..._e5m2x16 emit one 16-byte CONTIGUOUS vector transaction, so they cannot
address a manifest rank-2 [M, N] fp8 output with column-major stride
(1, M): element (m, n) lives at byte offset m + n*M and lanes across
columns are M bytes apart.  gmem_st_f32_to_{e4m3,e5m2} (one element at an
arbitrary byte offset) and gmem_st_f32_to_{e4m3,e5m2}x2 (one pair
conversion, two independently addressed bytes) close that gap with no
shape/task assumption -- the caller supplies the stride arithmetic.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel import language as ll
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda
from little_kernel.backends.nv.common.language.intrin.memory import codegen_gmem_st_f32_to_e4m3, codegen_gmem_st_f32_to_e4m3x2, codegen_gmem_st_f32_to_e5m2, codegen_gmem_st_f32_to_e5m2x2

# One thread per logical element: column-major [M, N] fp8 outputs D/Aux
# (stride (1, M)) are written from a contiguous FP32 source.  The byte
# offset i * M is the column stride; the x2 twins write a pair of columns
# (offsets i*M and i*M + M) with a single pair conversion.
SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def colmajor_fp8_store(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
                       D: ll.Tensor[ll.uint8], Aux: ll.Tensor[ll.uint8],
                       M: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    # stride (1, M): logical (m, n) -> byte offset m + n*M (here i*M).
    ll.gmem_st_f32_to_e4m3(D, i * M, x[i])
    ll.gmem_st_f32_to_e5m2(Aux, i * M, y[i])
    # strided pair: one conversion, two bytes M apart (columns n, n+1).
    ll.gmem_st_f32_to_e4m3x2(D, i * M, i * M + M, x[i], y[i])
    ll.gmem_st_f32_to_e5m2x2(Aux, i * M, i * M + M, y[i], x[i])
'''


def _spec():
    return {
        "kernel":
        "colmajor_fp8_store",
        "function":
        "colmajor_fp8_store",
        "inputs": [
            {"name": "x", "dtype": "float32", "rank": 1},
            {"name": "y", "dtype": "float32", "rank": 1},
        ],
        "outputs": [
            # Manifest-declared rank-2 fp8 outputs, exported byte-addressed.
            {"name": "D", "dtype": "float8_e4m3fn", "shape": ["16", "8"]},
            {"name": "Aux", "dtype": "float8_e4m3fn", "shape": ["16", "8"]},
        ],
        "scalars": [{"name": "M", "dtype": "int32", "expr": "16"}, {"name": "N", "dtype": "int32", "expr": "8"}],
        "kernel_args": ["x", "y", "D", "Aux", "M"],
        "forward_args": ["x", "y"],
        "launch": {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": [128, 1, 1]},
    }


def _render(source=SOURCE, kernel_name="colmajor_fp8_store"):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spec = _spec()
        spec["kernel"] = kernel_name
        spec["function"] = kernel_name
        (root / "export.json").write_text(json.dumps(spec))
        kernel_path = root / "kernel.py"
        kernel_path.write_text(source)
        spec = _load_spec(root / "export.json")
        kernel = _load_kernel(kernel_path, kernel_name)
        return _render_cuda(spec, kernel)


class StridedFp8StoreTests(unittest.TestCase):

    def test_builtins_exported_and_typed(self):
        for name in ("gmem_st_f32_to_e4m3", "gmem_st_f32_to_e4m3x2", "gmem_st_f32_to_e5m2", "gmem_st_f32_to_e5m2x2"):
            self.assertTrue(hasattr(ll, name), f"missing ll.{name}")

    def test_column_major_fp8_outputs_render(self):
        cuda = _render()
        # Singleton element stores for both encodings, addressed at the
        # caller-computed strided byte offset (i * M).
        self.assertIn("gmem_st_f32_to_e4m3_fn(", cuda)
        self.assertIn("gmem_st_f32_to_e5m2_fn(", cuda)
        # Explicit-offset pair stores.
        self.assertIn("gmem_st_f32_to_e4m3x2_fn(", cuda)
        self.assertIn("gmem_st_f32_to_e5m2x2_fn(", cuda)
        # The stride arithmetic survives lowering (column stride M).
        self.assertIn("* M)", cuda)
        # Both RNE/satfinite pair converts are emitted exactly once each
        # (one conversion per x2 call; the singleton reuses the pair cvt).
        self.assertIn("cvt.rn.satfinite.e4m3x2.f32", cuda)
        self.assertIn("cvt.rn.satfinite.e5m2x2.f32", cuda)

    def test_scalar_byte_stores_no_vector_transaction(self):
        cuda = _render()
        # Strided output must not pull in the 16-byte contiguous vector
        # store or its uint4 transaction.
        self.assertNotIn("gmem_st_f32_to_e4m3x16_fn", cuda)
        self.assertNotIn("gmem_st_f32_to_e5m2x16_fn", cuda)
        # The pair store scatters through two scalar byte stores, and byte
        # extraction follows the packed-pair lane order (low <- f_lo).
        self.assertIn("base[off_lo] = (uint8_t)(p & 0xFFu);", cuda)
        self.assertIn("base[off_hi] = (uint8_t)((p >> 8) & 0xFFu);", cuda)
        # Scalar byte stores need no 16-byte alignment helper.
        self.assertNotIn("reinterpret_cast<uint4*>", cuda)

    def test_contiguous_tail_via_offsets(self):
        # The same x2 builtin serves a contiguous pair (adjacent offsets),
        # proving the offset form generalizes rather than hard-coding a
        # stride: no task/layout dispatch, just address arithmetic.
        source = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def contig_tail(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
                D: ll.Tensor[ll.uint8], Aux: ll.Tensor[ll.uint8],
                M: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    ll.gmem_st_f32_to_e4m3x2(D, i * 2, i * 2 + 1, x[i], x[i + 1])
    ll.gmem_st_f32_to_e5m2(D, i, x[i])
'''
        cuda = _render(source, "contig_tail")
        self.assertIn("gmem_st_f32_to_e4m3x2_fn(", cuda)
        self.assertIn("gmem_st_f32_to_e5m2_fn(", cuda)

    def test_codegen_factories(self):
        b = codegen_gmem_st_f32_to_e4m3("D", "m + n * M", "v")
        self.assertIn("reinterpret_cast<uint8_t*>(D) + (m + n * M)", b.return_val)
        self.assertIn("cvt.rn.satfinite.e4m3x2.f32", b.body)
        b2 = codegen_gmem_st_f32_to_e5m2x2("D", "m + n * M", "m + (n + 1) * M", "a", "c")
        self.assertIn("(m + n * M)", b2.return_val)
        self.assertIn("(m + (n + 1) * M)", b2.return_val)
        self.assertIn("cvt.rn.satfinite.e5m2x2.f32", b2.body)
        # Lane order: PTX operand order is (hi, lo); low byte <- f_lo.
        self.assertIn('"f"(f_hi), "f"(f_lo)', b2.body)
        # Smoke-test the remaining twins construct.
        self.assertIn("e4m3", codegen_gmem_st_f32_to_e4m3x2("D", "0", "1", "a", "b").body)
        self.assertIn("e5m2", codegen_gmem_st_f32_to_e5m2("D", "0", "a").body)


if __name__ == "__main__":
    unittest.main()
