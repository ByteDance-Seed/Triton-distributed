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
"""Self-aliased (canvas) outputs: an output whose name IS an input name.

The KL-divergence backward, gradient-accumulation canvases, and other
caller-owned-buffer operators hand the kernel a pre-allocated input it must
fully overwrite and return. The export contract expresses this in-place
canvas as an output whose ``alias_of`` names ITSELF (``alias_of == name``,
where ``name`` is already a declared input): the binding threads the input
parameter straight through, no ``at::empty`` is allocated, no shadowing
``torch::Tensor name = name;`` rebind is emitted (that redeclaration would
be a compile error), and ``Model.forward`` returns the input tensor object.

Previously the spec loader rejected self-alias at the uniqueness check
("output name must be unique"), and even if the check were skipped, the
binding emitter would have generated uncompilable shadowing declarations.

Contract under test (through the full lower() path):

- ``_load_spec`` accepts an output whose name is one of the declared tensor
  inputs when ``alias_of == name`` (self-alias); the shape defaults to the
  input's shape expressions.
- The generated binding does NOT emit a shadowing
  ``torch::Tensor name = name;`` line; it does emit TORCH_CHECKs, launch,
  and ``return name;`` against the input parameter.
- The generated CUDA entry kernel still qualifies every pointer with
  ``__restrict__`` — a self-alias alone cannot make two *distinct* kernel
  parameters overlap (the alias is identity, not a view of a different
  parameter), so the noalias assertion on each pointer parameter stays
  justified.
- ``model_new.py`` returns the input tensor object itself (identity), not
  a rebound local.
- Self-alias with a mismatched dtype, a self-alias targeting a non-input
  name, and a self-alias with an explicit mismatched-rank shape are still
  rejected.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_spec

CANVAS_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def negate_into(canvas: ll.Tensor[ll.float32],
                target: ll.Tensor[ll.float32],
                n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        t: ll.float32 = target[i]
        canvas[i] = -t
'''


def _spec():
    return {
        "kernel": "negate_into",
        "function": "negate_into",
        "inputs": [
            {"name": "canvas", "dtype": "float32", "rank": 1},
            {"name": "target", "dtype": "float32", "rank": 1},
        ],
        # Self-alias: the canvas input is fully overwritten and returned.
        "outputs": [{"name": "canvas", "dtype": "float32", "alias_of": "canvas"}],
        "scalars": [{"name": "n", "dtype": "int32", "expr": "target[0]"}],
        "kernel_args": ["canvas", "target", "n"],
        "forward_args": ["canvas", "target"],
        "launch": {"grid": ["ceil_div(n, 256)", 1, 1], "block": [256, 1, 1]},
    }


def _export(tmp: Path, source: str, spec: dict) -> tuple[str, str, str]:
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
    bindings = list((output / "kernels").glob("*_binding.cpp"))
    assert len(bindings) == 1, bindings
    return (kernels[0].read_text(), bindings[0].read_text(), (output / "model_new.py").read_text())


class SelfAliasLoadTests(unittest.TestCase):
    """The spec loader must accept a self-aliased output and prefill shape."""

    def test_self_alias_loads_and_inherits_input_shape(self):
        with tempfile.TemporaryDirectory() as d:
            spec_path = Path(d) / "export.json"
            spec_path.write_text(json.dumps(_spec()))
            loaded = _load_spec(spec_path)
        outputs = loaded["_outputs"]
        self.assertEqual(len(outputs), 1)
        out = outputs[0]
        self.assertEqual(out["name"], "canvas")
        self.assertEqual(out["alias_of"], "canvas")
        # Shape inherited from the aliased input's dimensions.
        self.assertEqual(out["shape"], ["canvas[0]"])
        self.assertEqual(out["_shape_cpp"], ["canvas.size(0)"])


class SelfAliasCodegenTests(unittest.TestCase):
    """Binding/CUDA emission must handle self-alias: no shadow, no empty."""

    def test_binding_does_not_shadow_self_alias(self):
        with tempfile.TemporaryDirectory() as d:
            cuda, binding, model = _export(Path(d), CANVAS_SOURCE, _spec())
        # No shadowing rebind (would be a C++ compile error: redeclaration
        # of a function parameter).
        self.assertNotIn("torch::Tensor canvas = canvas;", binding)
        # No fresh at::empty allocation for the canvas output.
        self.assertNotIn("at::empty({canvas.size(0)}", binding)
        # Kernel launch uses the input parameter directly.
        self.assertIn("agent_lk_launch_negate_into(canvas.data_ptr()", binding)
        # The binding returns the input object.
        self.assertIn("return canvas;", binding)
        # Model returns the input tensor (identity pass-back).
        self.assertIn("return canvas", model)
        self.assertIn("cuda_extension.agent_negate_into(canvas, target)", model)
        # Entry kernel: canvas is the ONLY kernel arg that aliases the ABI
        # slot of the same name; a lone pointer parameter cannot overlap
        # any *other* parameter, so __restrict__ remains justified.
        self.assertIn("float* __restrict__ canvas", cuda)
        self.assertIn("const float* __restrict__ target", cuda)

    def test_no_alias_policy_line_emitted_for_self_alias(self):
        """A self-alias is an identity aliasing group (singleton). It does
        not trip the cross-parameter __restrict__ opt-out (that mechanism
        guards against ONE allocation reaching TWO kernel parameters), so
        no alias-policy audit line should be emitted."""
        with tempfile.TemporaryDirectory() as d:
            cuda, _, _ = _export(Path(d), CANVAS_SOURCE, _spec())
        self.assertNotIn("alias-policy:", cuda)


class SelfAliasRejectionTests(unittest.TestCase):
    """Malformed self-alias declarations must still fail."""

    def _reject(self, mutate):
        spec = _spec()
        mutate(spec)
        with tempfile.TemporaryDirectory() as d:
            spec_path = Path(d) / "export.json"
            spec_path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                _load_spec(spec_path)

    def test_self_alias_dtype_mismatch_rejected(self):

        def mutate(spec):
            spec["outputs"] = [{"name": "canvas", "dtype": "bfloat16", "alias_of": "canvas"}]

        self._reject(mutate)

    def test_self_alias_rank_mismatched_explicit_shape_rejected(self):

        def mutate(spec):
            spec["outputs"] = [{
                "name": "canvas", "dtype": "float32", "alias_of": "canvas", "shape": ["canvas[0]", "canvas[0]"]
            }]

        self._reject(mutate)

    def test_alias_of_not_an_input_rejected(self):

        def mutate(spec):
            spec["outputs"] = [{"name": "canvas", "dtype": "float32", "alias_of": "nope"}]

        self._reject(mutate)

    def test_unaliased_output_name_colliding_with_input_rejected(self):
        """Without alias_of, an output naming an existing input must still
        fail the uniqueness check (no implicit in-place rebinding)."""

        def mutate(spec):
            spec["outputs"] = [{"name": "canvas", "dtype": "float32", "shape": ["canvas[0]"]}]

        self._reject(mutate)

    def test_alias_of_cross_input_still_binds(self):
        """Cross-input alias (out alias_of x) remains the documented way
        to rebind a distinct input and must keep working (C++ emission
        DOES spell the rebind for this case)."""
        src = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < 1024:
        out[i] = x[i] * 2.0
'''
        spec = {
            "kernel": "f",
            "function": "f",
            "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
            "outputs": [{"name": "out", "dtype": "float32", "alias_of": "x"}],
            "scalars": [],
            "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
        }
        with tempfile.TemporaryDirectory() as d:
            _, binding, _ = _export(Path(d), src, spec)
        self.assertIn("torch::Tensor out = x;", binding)


if __name__ == "__main__":
    unittest.main()
