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
"""Two-byte tensor ABI: int16/uint16 through the real export() path.

The eval integration's _DTYPES table is the single source of truth for the
torch <-> CUDA transport of each tensor dtype.  The device side already
supports 16-bit integer element types (codegen_cuda's IntType bit_map and
core/type_system's IntType(bits=16)); these tests drive the real spec
validation + binding/codegen renderers (no nvcc) and assert the host ABI:

  * ``uint16`` maps to C++ ``uint16_t*`` guarded by ``at::kUInt16`` and
    ``int16`` to ``int16_t*`` guarded by ``at::kShort``;
  * fresh outputs allocate with the matching torch dtype;
  * the declared-annotation channel (_declared_tensor_dtypes) resolves
    ``ll.Tensor[ll.uint16]`` / ``ll.Tensor[ll.int16]`` so the launcher cast
    equals the entry signature's element type;
  * the two two-byte torch dtypes stay distinct scalar types.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def copy16(x: ll.Tensor[ll.{in_ll}], y: ll.Tensor[ll.{out_ll}], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        y[i] = x[i]
'''


def _spec(in_dtype, out_dtype):
    return {
        "kernel": "copy16",
        "function": "copy",
        "inputs": [{"name": "x", "dtype": in_dtype, "rank": 1}],
        "outputs": [{"name": "y", "dtype": out_dtype, "shape": ["x[0]"]}],
        "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
        "kernel_args": ["x", "y", "n"],
        "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
    }


class _ExportCase(unittest.TestCase):

    def _export(self, spec, in_ll, out_ll):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(KERNEL_SOURCE.format(in_ll=in_ll, out_ll=out_ll))
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(spec))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        # (generated CUDA translation unit, torch binding translation unit)
        return paths[0].read_text(), paths[1].read_text()


class TwoByteTensorAbiTests(_ExportCase):

    def test_uint16_input_guard_and_launcher_cast(self):
        cuda, binding = self._export(_spec("uint16", "uint16"), "uint16", "uint16")
        self.assertIn("TORCH_CHECK(x.scalar_type() == at::kUInt16", binding)
        self.assertIn("reinterpret_cast<uint16_t*>(x)", cuda)
        # Entry-kernel parameters may carry the merged const/__restrict__
        # alias contract, so tolerate those qualifiers between the storage
        # type and the parameter name.
        self.assertRegex(cuda, r"\buint16_t\*\s+(?:__restrict__\s+)?x\b")

    def test_int16_input_guard_and_launcher_cast(self):
        cuda, binding = self._export(_spec("int16", "int16"), "int16", "int16")
        self.assertIn("TORCH_CHECK(x.scalar_type() == at::kShort", binding)
        self.assertIn("reinterpret_cast<int16_t*>(x)", cuda)
        self.assertRegex(cuda, r"\bint16_t\*\s+(?:__restrict__\s+)?x\b")

    def test_uint16_output_allocation(self):
        cuda, binding = self._export(_spec("uint16", "uint16"), "uint16", "uint16")
        self.assertIn("dtype(at::kUInt16)", binding)
        self.assertIn("reinterpret_cast<uint16_t*>(y)", cuda)

    def test_int16_output_allocation(self):
        cuda, binding = self._export(_spec("int16", "int16"), "int16", "int16")
        self.assertIn("dtype(at::kShort)", binding)
        self.assertIn("reinterpret_cast<int16_t*>(y)", cuda)

    def test_declared_dtype_resolves_from_annotation(self):
        # The launcher cast keys on the frontend annotation channel
        # (_declared_tensor_dtypes); ll uint16/int16 must resolve to the same
        # spec dtype name so the cast equals the generated kernel signature's
        # element type.  Drive the channel directly on a live kernel.
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        src = Path(tmp.name) / "k.py"
        src.write_text(KERNEL_SOURCE.format(in_ll="uint16", out_ll="int16"))
        kernel = vibe_cuda_eval._load_kernel(src, "copy16")
        declared = vibe_cuda_eval._declared_tensor_dtypes(kernel)
        self.assertEqual(declared.get("x"), "uint16")
        self.assertEqual(declared.get("y"), "int16")

    def test_mixing_16bit_with_8bit_params(self):
        cuda, binding = self._export(
            {
                "kernel": "copy16",
                "function": "copy",
                "inputs": [{"name": "x", "dtype": "uint16", "rank": 1}],
                "outputs": [{"name": "y", "dtype": "int16", "shape": ["x[0]"]}],
                "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
                "kernel_args": ["x", "y", "n"],
                "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
            }, "uint16", "int16")
        self.assertIn("TORCH_CHECK(x.scalar_type() == at::kUInt16", binding)
        self.assertIn("dtype(at::kShort)", binding)
        self.assertIn("reinterpret_cast<uint16_t*>(x)", cuda)
        self.assertIn("reinterpret_cast<int16_t*>(y)", cuda)

    def test_two_byte_dtypes_stay_distinct(self):
        self.assertEqual(vibe_cuda_eval._DTYPES["uint16"][0], "uint16_t")
        self.assertEqual(vibe_cuda_eval._DTYPES["int16"][0], "int16_t")
        # Signed and unsigned two-byte torch tensors carry distinct scalar
        # types, so the guards cannot accept each other's tensors.
        self.assertNotEqual(vibe_cuda_eval._DTYPES["uint16"][1], vibe_cuda_eval._DTYPES["int16"][1])
        # ... and both differ from the 8-bit guards.
        self.assertNotEqual(vibe_cuda_eval._DTYPES["uint16"][1], vibe_cuda_eval._DTYPES["uint8"][1])
        self.assertNotEqual(vibe_cuda_eval._DTYPES["int16"][1], vibe_cuda_eval._DTYPES["uint8"][1])


if __name__ == "__main__":
    unittest.main()
