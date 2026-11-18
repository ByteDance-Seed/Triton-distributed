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
"""Export-contract alias legality for entry-kernel __restrict__.

The vibe-cuda-eval export contract can hand one device allocation to two
entry-kernel parameters (an "alias_of" output rebinds its input in place; an
"alias" transform rebinds its source tensor). Codegen must then drop the
__restrict__ qualifier on every transitively linked parameter while keeping it
on parameters the contract can never alias. These tests assert that ABI
legality contract on the generated CUDA text through the full lower() path.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

PLAIN_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
            out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        out[i] = a[i] + b[i]
'''

ALIAS_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def spot(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
         y: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        xi: ll.float32 = x[i]
        out[i] = 2.0 * xi
        y[i] = xi + 1.0
'''

VIEW_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def seen(x: ll.Tensor[ll.float32], xv: ll.Tensor[ll.float32],
         out: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        out[i] = xv[i] * 2.0
        y[i] = x[i] + 1.0
'''

ESCAPE_SOURCE = '''\
__LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__ = ("a",)

import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def esc(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
        out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        out[i] = a[i] + b[i]
'''


def _export(tmp: Path, source: str, spec: dict) -> str:
    task = tmp / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = tmp / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(json.dumps(spec))
    output = tmp / "output"
    lower(task, program, output, 0)
    kernels = list((output / "kernels").glob("*.cu"))
    assert len(kernels) == 1, kernels
    return kernels[0].read_text()


class AliasContractTests(unittest.TestCase):

    def test_plain_program_keeps_full_restrict(self):
        """No aliasing constructs: every entry tensor param stays __restrict__."""
        spec = {
            "kernel": "vec_add",
            "function": "vec_add",
            "inputs": [{"name": "a", "dtype": "float32", "rank": 1}, {"name": "b", "dtype": "float32", "rank": 1}],
            "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
            "scalars": [],
            "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            cuda = _export(Path(directory), PLAIN_SOURCE, spec)
        self.assertIn("float* __restrict__ a", cuda)
        self.assertIn("float* __restrict__ b", cuda)
        self.assertIn("float* __restrict__ out", cuda)
        # No opt-out active: no audit line, artifact unchanged.
        self.assertNotIn("alias-policy:", cuda)

    def test_alias_of_output_opts_out_both_sides(self):
        """out alias_of x: x and out lose __restrict__; fresh y keeps it."""
        spec = {
            "kernel":
            "spot",
            "function":
            "spot",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
            "outputs": [{"name": "out", "dtype": "float32", "alias_of": "x"},
                        {"name": "y", "dtype": "float32", "shape": ["x[0]"]}],
            "scalars": [],
            "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            cuda = _export(Path(directory), ALIAS_SOURCE, spec)
        self.assertNotIn("float* __restrict__ x", cuda)
        self.assertNotIn("float* __restrict__ out", cuda)
        self.assertIn("float* __restrict__ y", cuda)
        # Opt-out active: the audit line records the exact split.
        self.assertIn(
            "// alias-policy: __restrict__ (noalias) asserted on [y]; "
            "not asserted on [x, out] (contract/frontend-declared "
            "may-alias).", cuda)

    def test_alias_transform_group_is_transitive(self):
        """alias transform xv of x plus out alias_of x: all three opt out."""
        spec = {
            "kernel":
            "seen",
            "function":
            "seen",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
            "transforms": [{"op": "alias", "name": "xv", "source": "x"}],
            "outputs": [{"name": "out", "dtype": "float32", "alias_of": "x"},
                        {"name": "y", "dtype": "float32", "shape": ["x[0]"]}],
            "kernel_args": ["x", "xv", "out", "y"],
            "scalars": [],
            "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            cuda = _export(Path(directory), VIEW_SOURCE, spec)
        self.assertNotIn("float* __restrict__ x", cuda)
        self.assertNotIn("float* __restrict__ xv", cuda)
        self.assertNotIn("float* __restrict__ out", cuda)
        self.assertIn("float* __restrict__ y", cuda)

    def test_frontend_escape_hatch_unions_with_contract(self):
        """A draft-declared MAY_ALIAS set is honored and unioned, not replaced."""
        spec = {
            "kernel": "esc",
            "function": "esc",
            "inputs": [{"name": "a", "dtype": "float32", "rank": 1}, {"name": "b", "dtype": "float32", "rank": 1}],
            "output": {"name": "out", "dtype": "float32", "alias_of": "b"},
            "scalars": [],
            "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            cuda = _export(Path(directory), ESCAPE_SOURCE, spec)
        self.assertNotIn("float* __restrict__ a", cuda)  # frontend-declared
        self.assertNotIn("float* __restrict__ b", cuda)  # contract alias_of
        self.assertNotIn("float* __restrict__ out", cuda)

    def test_single_launch_foreach_propagates_opt_out(self):
        """The fused foreach table copies the entry param's restrict qualifier."""
        spec = {
            "kernel":
            "spot",
            "function":
            "spot",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
            "outputs": [{"name": "out", "dtype": "float32", "alias_of": "x"},
                        {"name": "y", "dtype": "float32", "shape": ["x[0]"]}],
            "foreach": {"inputs": ["x"], "mode": "single_launch"},
            "scalars": [],
            "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            cuda = _export(Path(directory), ALIAS_SOURCE, spec)
        self.assertIn("float* x = tbl.x[lk_e];", cuda)
        self.assertIn("float* out = tbl.out[lk_e];", cuda)
        self.assertIn("float* __restrict__ y = tbl.y[lk_e];", cuda)
        self.assertNotIn("__restrict__ x = tbl.x[lk_e];", cuda)
        self.assertNotIn("__restrict__ out = tbl.out[lk_e];", cuda)


ESCAPE_TYPO_SOURCE = '''\
__LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__ = ("aa",)

import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def esc(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
        out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        out[i] = a[i] + b[i]
'''


class AliasRejectionTests(unittest.TestCase):
    """Malformed/ambiguous alias declarations must be rejected, not ignored
    (an ignored alias declaration leaves a parameter under __restrict__
    while the ABI can hand it a shared allocation)."""

    def _spec(self, **overrides):
        spec = {
            "kernel":
            "spot",
            "function":
            "spot",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
            "outputs": [{"name": "out", "dtype": "float32", "alias_of": "x"},
                        {"name": "y", "dtype": "float32", "shape": ["x[0]"]}],
            "scalars": [],
            "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        spec.update(overrides)
        return spec

    def test_alias_of_unknown_input_rejected(self):
        spec = self._spec(outputs=[{"name": "out", "dtype": "float32", "alias_of": "nope"}])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _export(Path(directory), ALIAS_SOURCE, spec)

    def test_alias_of_dtype_mismatch_rejected(self):
        spec = self._spec(outputs=[{"name": "out", "dtype": "bfloat16", "alias_of": "x"}])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _export(Path(directory), ALIAS_SOURCE, spec)

    def test_alias_of_rank_mismatched_shape_rejected(self):
        """An aliased output IS the input tensor; an explicit shape of a
        different rank is ambiguous metadata and must not load."""
        spec = self._spec(outputs=[{"name": "out", "dtype": "float32", "alias_of": "x", "shape": ["x[0]", "x[0]"]}])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _export(Path(directory), ALIAS_SOURCE, spec)

    def test_alias_transform_unknown_source_rejected(self):
        spec = self._spec(transforms=[{"op": "alias", "name": "xv", "source": "nope"}])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _export(Path(directory), ALIAS_SOURCE, spec)

    def test_frontend_escape_hatch_unknown_param_rejected(self):
        """A typo'd may-alias name would silently leave the intended
        parameter restricted: reject the declaration."""
        spec = {
            "kernel": "esc",
            "function": "esc",
            "inputs": [{"name": "a", "dtype": "float32", "rank": 1}, {"name": "b", "dtype": "float32", "rank": 1}],
            "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
            "scalars": [],
            "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                _export(Path(directory), ESCAPE_TYPO_SOURCE, spec)


if __name__ == "__main__":
    unittest.main()
