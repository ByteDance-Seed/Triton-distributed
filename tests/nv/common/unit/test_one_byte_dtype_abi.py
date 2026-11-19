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
"""One-byte tensor ABI: bool/int8/uint8 through the real export() path.

The eval integration's _DTYPES table is the single source of truth for the
torch <-> CUDA transport of each tensor dtype.  These tests drive the real
spec validation + binding/codegen renderers (no nvcc) and assert:

  * ``bool`` maps to C++ ``bool*`` guarded by ``at::kBool`` (genuine
    torch.bool storage semantics, true/false only -- never a numeric
    reinterpretation);
  * bool outputs are allocated with ``at::empty(..., dtype(at::kBool))`` and
    may alias a bool input in place;
  * bool stays distinct from uint8 (at::kByte) and from the FP8 storage
    dtypes (at::kFloat8_e4m3fn over uint8_t*);
  * unsupported dtypes are still rejected at spec load.
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
def flag_copy(a: ll.Tensor[ll.float32], flag: ll.Tensor[ll.bool_],
              out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        if flag[0]:
            out[i] = a[i]
'''

KERNEL_SOURCE_OUT = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def flag_copy(a: ll.Tensor[ll.float32], flag: ll.Tensor[ll.bool_],
              out: ll.Tensor[ll.float32], flag_out: ll.Tensor[ll.bool_],
              n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        if flag[0]:
            out[i] = a[i]
        flag_out[0] = flag[0]
'''


def _spec(inputs, outputs, kernel_args, scalars=None):
    return {
        "kernel": "flag_copy",
        "function": "gated",
        "inputs": inputs,
        "outputs": outputs,
        "scalars": scalars if scalars is not None else [{"name": "n", "dtype": "int32", "expr": "a[0]"}],
        "kernel_args": kernel_args,
        "forward_args": ["a", "flag"],
        "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
    }


BASE_INPUTS = [
    {"name": "a", "dtype": "float32", "rank": 1},
    {"name": "flag", "dtype": "bool", "rank": 1},
]


class _ExportCase(unittest.TestCase):

    def _export(self, spec, source=KERNEL_SOURCE):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(source)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(spec))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        # (generated CUDA translation unit, torch binding translation unit)
        return paths[0].read_text(), paths[1].read_text()


class BoolTensorAbiTests(_ExportCase):

    def test_bool_input_guard_and_launcher_cast(self):
        cuda, binding = self._export(
            _spec(
                BASE_INPUTS,
                [{"name": "out", "dtype": "float32", "shape": ["a[0]"]}],
                ["a", "flag", "out", "n"],
            ))
        self.assertIn('TORCH_CHECK(flag.scalar_type() == at::kBool', binding)
        # The CUDA-side launcher casts the void* transport to the genuine
        # bool* storage type; the device kernel signature carries it too.
        # Entry-kernel parameters additionally carry the merged const/
        # __restrict__ alias contract (asserted verbatim in
        # test_eval_alias_contract.py and test_restrict_params.py), so the
        # storage-type check tolerates those qualifiers between `bool*` and
        # the parameter name.
        self.assertIn("reinterpret_cast<bool*>(flag)", cuda)
        self.assertRegex(cuda, r"\bbool\*\s+(?:__restrict__\s+)?flag\b")
        # The non-bool input keeps its own guard.
        self.assertIn('TORCH_CHECK(a.scalar_type() == at::kFloat', binding)

    def test_bool_output_allocation_and_guard(self):
        cuda, binding = self._export(
            _spec(
                BASE_INPUTS,
                [{"name": "out", "dtype": "float32", "shape": ["a[0]"]},
                 {"name": "flag_out", "dtype": "bool", "shape": ["flag[0]"]}],
                ["a", "flag", "out", "flag_out", "n"],
            ), source=KERNEL_SOURCE_OUT)
        self.assertIn("dtype(at::kBool)", binding)
        self.assertIn("reinterpret_cast<bool*>(flag_out)", cuda)

    def test_bool_inplace_alias(self):
        cuda, binding = self._export(
            _spec(
                BASE_INPUTS,
                [{"name": "out", "dtype": "float32", "shape": ["a[0]"]},
                 {"name": "flag_out", "dtype": "bool", "alias_of": "flag"}],
                ["a", "flag", "out", "flag_out", "n"],
            ), source=KERNEL_SOURCE_OUT)
        self.assertIn("torch::Tensor flag_out = flag;", binding)

    def test_alias_dtype_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            self._export(
                _spec(
                    BASE_INPUTS,
                    [{"name": "flag_out", "dtype": "int8", "alias_of": "flag"}],
                    ["a", "flag", "flag_out", "n"],
                ), source=KERNEL_SOURCE_OUT)


class OneByteDistinctionTests(_ExportCase):

    def test_bool_stays_distinct_from_uint8_and_fp8(self):
        self.assertEqual(vibe_cuda_eval._DTYPES["bool"][0], "bool")
        self.assertEqual(vibe_cuda_eval._DTYPES["uint8"][0], "uint8_t")
        # A torch.bool tensor must fail an at::kByte guard and vice versa:
        # the guards carry distinct scalar types.
        self.assertNotEqual(vibe_cuda_eval._DTYPES["bool"][1], vibe_cuda_eval._DTYPES["uint8"][1])
        # FP8 keeps its own identity: uint8_t storage but FP8 torch dtype.
        self.assertEqual(vibe_cuda_eval._DTYPES["float8_e4m3fn"][0], "uint8_t")
        self.assertIn("Float8", vibe_cuda_eval._DTYPES["float8_e4m3fn"][1])

    def test_unsupported_dtype_still_rejected(self):
        # uint32 remains outside the export dtype table (int16/uint16 joined
        # it in the two-byte ABI completion; see test_two_byte_dtype_abi.py),
        # so it still exercises the spec-load rejection path.
        spec = _spec(
            [{"name": "a", "dtype": "float32", "rank": 1}, {"name": "flag", "dtype": "uint32", "rank": 1}],
            [{"name": "out", "dtype": "float32", "shape": ["a[0]"]}],
            ["a", "flag", "out", "n"],
        )
        with self.assertRaisesRegex(ValueError, "unsupported dtype"):
            self._export(spec)


if __name__ == "__main__":
    unittest.main()
