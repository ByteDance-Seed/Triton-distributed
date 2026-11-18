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
"""CPU regression for the export ABI byte-view transport contract.

A model tensor whose torch contract dtype is fp16/bf16 may be transported
to/from the entry kernel as an opaque byte view: the wrapper allocates and
compares at the declared contract scalar type while the kernel signature
receives a uint8_t element pointer (the word-indexing pattern used by
bitwise 2:4 sparsity pack/unpack kernels that address fp16/bf16 storage
through ld.global.v4.b32 words).

The split is key-driven, not manifest-field-driven:

* the manifest ``dtype`` selects the torch-level allocation scalar type
  (``_DTYPES[d][2]``), the comparison scalar type (``_DTYPES[d][1]``) and
  the element unit of every shape/scalar expression;
* the kernel pointer element type follows the *frontend-declared* tensor
  annotation (``ll.Tensor[ll.uint8]``), resolved by
  ``_frontend._declared_tensor_dtypes`` from the same py_func annotations
  the code generator used for the entry signature, so the launcher's
  ``reinterpret_cast`` and the generated signature agree by construction.
  C++ has no implicit conversion between distinct one-byte pointer types,
  so keying the cast anywhere else (e.g. on a manifest view flag that
  could contradict the annotation) can emit a translation unit nvcc
  rejects.

These tests render the full export bundle on CPU (no CUDA toolchain, no
GPU) and pin both halves of the contract for every tensor role: inputs,
freshly allocated outputs, aliased (in-place) outputs and workspaces, for
both float16 and bfloat16 contract dtypes.
"""

import sys
import tempfile
import unittest
from pathlib import Path

for _parent in Path(__file__).resolve().parents:
    if (_parent / "python/little_kernel/backends/nv/common/export/_render.py").is_file():
        if str(_parent / "python") not in sys.path:
            sys.path.insert(0, str(_parent / "python"))
        break
else:
    raise RuntimeError("could not locate repository root containing python/little_kernel")

from little_kernel.backends.nv.common.export._render import render_bundle

# Byte-view entry: parameters are declared uint8 tensors while the manifest
# below declares the torch contract dtype.  The body copies one 16-byte word
# per block through v4.b32 lanes; indexing is byte-unit, which is the
# frontend's responsibility under this contract.
_BYTE_VIEW_DRAFT = '''
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend='cuda', is_entry=True)
def byte_view_copy(x: ll.Tensor[ll.uint8], packed: ll.Tensor[ll.uint8]) -> ll.void:
    item: ll.int32 = ll.blockIdx_x()
    n: ll.int32 = item * 16
    r0: ll.uint32 = 0
    r1: ll.uint32 = 0
    r2: ll.uint32 = 0
    r3: ll.uint32 = 0
    ll.gmem_ld_v4_u32(x, n * 2, r0, r1, r2, r3)
    ll.gmem_st_v4_u32(packed, n, r0, r1, r2, r3)
'''

_ROLES_DRAFT = '''
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend='cuda', is_entry=True)
def byte_view_roles(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.uint8],
                    scratch: ll.Tensor[ll.uint8]) -> ll.void:
    item: ll.int32 = ll.blockIdx_x()
    y[item] = x[item]
    scratch[item] = x[item]
'''


def _render(draft_text, spec):
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "draft.py"
        src.write_text(draft_text)
        return render_bundle(src, spec)


class ByteViewTransportTest(unittest.TestCase):
    """Contract dtype and kernel pointer element type are independent."""

    def _spec(self, dtype):
        return {
            "kernel": "byte_view_copy",
            "function": "byte_view_copy",
            "inputs": [{"name": "x", "dtype": dtype, "rank": 2}],
            "outputs": [
                {"name": "packed", "dtype": dtype, "shape": ["x[0]", "x[1] // 2"]},
            ],
            "launch": {"grid": ["x[0] * x[1] // 64", 1, 1], "block": [1, 1, 1]},
        }

    def _check_bundle(self, dtype, torch_scalar):
        bundle = _render(_BYTE_VIEW_DRAFT, self._spec(dtype))
        cuda, binding = bundle.cuda, bundle.binding
        # Kernel side: both parameters are uint8 element pointers and the
        # launcher transport casts the void* data pointers to uint8_t*.
        self.assertIn("const uint8_t* __restrict__ x", cuda)
        self.assertIn("uint8_t* __restrict__ packed", cuda)
        self.assertIn("reinterpret_cast<uint8_t*>(x)", cuda)
        self.assertIn("reinterpret_cast<uint8_t*>(packed)", cuda)
        # Torch ABI side: allocation and dtype comparison stay at the
        # contract scalar type, and the shape expression counts contract
        # elements (halving fp16/bf16 columns, not bytes).
        self.assertIn(f"TORCH_CHECK(x.scalar_type() == at::{torch_scalar}", binding)
        self.assertIn(
            f"torch::Tensor packed = at::empty({{x.size(0), (x.size(1) / 2)}}, "
            f"x.options().dtype(at::{torch_scalar}));", binding)

    def test_float16_contract(self):
        self._check_bundle("float16", "kHalf")

    def test_bfloat16_contract(self):
        self._check_bundle("bfloat16", "kBFloat16")

    def test_all_tensor_roles(self):
        """Aliased outputs and workspaces ride the same transport split."""
        bundle = _render(
            _ROLES_DRAFT, {
                "kernel": "byte_view_roles",
                "function": "byte_view_roles",
                "inputs": [{"name": "x", "dtype": "float16", "rank": 1}],
                "outputs": [{"name": "y", "dtype": "float16", "alias_of": "x"}],
                "workspaces": [
                    {"name": "scratch", "dtype": "float16", "shape": ["x[0]"]},
                ],
                "kernel_args": ["x", "y", "scratch"],
                "launch": {"grid": ["x[0]", 1, 1], "block": [1, 1, 1]},
            })
        cuda, binding = bundle.cuda, bundle.binding
        for name in ("x", "y", "scratch"):
            self.assertIn(f"reinterpret_cast<uint8_t*>({name})", cuda)
        self.assertIn("at::empty({x.size(0)}, x.options().dtype(at::kHalf));", binding)
        self.assertIn("TORCH_CHECK(x.scalar_type() == at::kHalf", binding)


if __name__ == "__main__":
    unittest.main()
