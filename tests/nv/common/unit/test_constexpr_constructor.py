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
"""Constexpr constructor scalars bake per-case literals into the generated CUDA.

An export.json constructor entry may carry `"constexpr": <value>`; the frontend
entry still declares the parameter, but every read becomes the literal at trace
time and the name leaves the runtime kernel ABI. Runtime scalars are untouched.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def chunk_scale(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
                n: ll.int32, step: ll.int32) -> ll.void:
    i: ll.int32 = (ll.blockIdx_x() * 256 + ll.threadIdx_x()) * step
    for c in range(0, step):
        if i + c < n:
            out[i + c] = a[i + c] * 2.0
'''

SPEC = {
    "kernel": "chunk_scale",
    "function": "scale",
    "inputs": [{"name": "a", "dtype": "float32", "rank": 1}],
    "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
    "constructor": [{"name": "step", "dtype": "int32", "constexpr": 32}],
    "scalars": [{"name": "n", "expr": "a[0]"}],
    "kernel_args": ["a", "out", "n", "step"],
    "launch": {"grid": ["ceil_div(a[0], step * 256)", 1, 1], "block": [256, 1, 1]},
}


def _build(tmp_root, source=SOURCE, spec=SPEC):
    root = Path(tmp_root)
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(json.dumps(spec))
    output = root / "out"
    lower(task, program, output, 0)
    return {
        "cuda": (output / "kernels/agent_lk_scale.cu").read_text(),
        "binding": (output / "kernels/agent_lk_scale_binding.cpp").read_text(),
        "model": (output / "model_new.py").read_text(),
    }


class ConstexprConstructorTests(unittest.TestCase):

    def test_literal_baked_and_runtime_abi_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = _build(directory)
            cuda = generated["cuda"]
            # The kernel signature drops the constexpr constructor parameter.
            entry = cuda[cuda.index("void agent_chunk_scale"):cuda.index("{", cuda.index("void agent_chunk_scale"))]
            self.assertIn("int32_t n", entry)
            self.assertNotIn("step", entry)
            # Static trip count and static chunk stride appear as literals.
            self.assertIn("c < 32", cuda)
            self.assertIn("* 32", cuda)
            # The launcher and binding drop the runtime argument entirely.
            self.assertNotIn("int64_t step", cuda)
            binding = generated["binding"]
            self.assertNotIn('pybind11::arg("step")', binding)
            self.assertNotIn(", step,", binding)
            # The host-side literal alias keeps grid/equality expressions working.
            self.assertIn("const int64_t step = 32;", binding)
            self.assertIn("lk_ceil_div(a.size(0), (step * 256))", binding)
            # The model still accepts the argument but refuses a mismatched value.
            model = generated["model"]
            compile(model, "model_new.py", "exec")
            self.assertIn("def __init__(self, step)", model)
            self.assertIn("if step != 32:", model)
            self.assertNotIn("self.step", model)
            self.assertIn("cuda_extension.agent_scale(a)", model)

    def test_runtime_scalars_stay_in_abi(self):
        with tempfile.TemporaryDirectory() as directory:
            generated = _build(directory)
            # Runtime scalar n remains a real kernel parameter end to end.
            self.assertIn("int64_t n", generated["cuda"])
            self.assertIn("const int64_t n = a.size(0);", generated["binding"])

    def test_spec_diagnostics(self):
        cases = [
            ({"name": "step", "dtype": "ignored", "constexpr":
              32}, "ignored constructor arguments cannot be constexpr"),
            ({"name": "step", "dtype": "int32", "constexpr": 32, "default":
              32}, "constexpr constructor arguments must not declare a default"),
            ({"name": "step", "dtype": "int32", "constexpr":
              1.5}, "constructor constexpr value does not match its dtype"),
            ({"name": "step", "dtype": "bool", "constexpr": 1}, "constructor constexpr value does not match its dtype"),
            ({"name": "step", "dtype": "float32", "constexpr":
              float("inf")}, "constructor constexpr value does not match its dtype"),
        ]
        for entry, message in cases:
            with tempfile.TemporaryDirectory() as directory:
                spec = {**SPEC, "constructor": [entry]}
                with self.assertRaisesRegex(ValueError, message):
                    _build(directory, spec=spec)

    def test_assignment_to_constexpr_name_is_a_compile_error(self):
        bad = SOURCE.replace("n: ll.int32, step: ll.int32",
                             "n: ll.int32, step: ll.int32").replace("    i: ll.int32 =",
                                                                    "    step = 7\n    i: ll.int32 =")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "constexpr parameter 'step' cannot be assigned"):
                _build(directory, source=bad)

    def test_cache_dump_keeps_specialization_identity(self):
        """Consecutive compiles of different constexpr values must not overwrite each
        other in a shared LK_CACHE_DIR debug dump."""
        import os
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache"
            cache.mkdir()
            previous = os.environ.get("LK_CACHE_DIR")
            os.environ["LK_CACHE_DIR"] = str(cache)
            try:
                with tempfile.TemporaryDirectory() as build:
                    _build(build, spec={**SPEC})  # step = 32
                with tempfile.TemporaryDirectory() as build:
                    _build(build, spec={**SPEC, "constructor": [{"name": "step", "dtype": "int32", "constexpr": 16}]})
            finally:
                if previous is None:
                    os.environ.pop("LK_CACHE_DIR", None)
                else:
                    os.environ["LK_CACHE_DIR"] = previous
            dumps = sorted(cache.glob("chunk_scale__cx*.cu"))
            self.assertEqual(len(dumps), 2, f"expected 2 specialized dumps, got {dumps}")
            bodies = {dump.read_text() for dump in dumps}
            self.assertTrue(any("* 32" in body for body in bodies))
            self.assertTrue(any("* 16" in body for body in bodies))

    def test_plain_specs_render_without_constexpr_plumbing(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = {**SPEC, "constructor": [{"name": "step", "dtype": "int32"}]}
            generated = _build(directory, spec=spec)
            # Without the field every artifact keeps the runtime ABI unchanged.
            cuda = generated["cuda"]
            entry = cuda[cuda.index("void agent_chunk_scale"):cuda.index("{", cuda.index("void agent_chunk_scale"))]
            self.assertIn("step", entry)
            self.assertNotIn("constexpr", generated["model"])
            self.assertNotIn("const int64_t step = ", generated["binding"])
            self.assertIn('pybind11::arg("step")', generated["binding"])

    def test_case_export_specializes_a_shared_draft(self):
        """A case-NNN/ directory with only export.json reuses the root draft.py."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            program.mkdir()
            (program / "draft.py").write_text(SOURCE)
            (program / "export.json").write_text(
                json.dumps({**SPEC, "constructor": [{"name": "step", "dtype": "int32"}]}))
            case = program / "case-000"
            case.mkdir()
            (case / "export.json").write_text(json.dumps(SPEC))
            output = root / "out"
            lower(task, program, output, 0)
            cuda = (output / "kernels/agent_lk_scale.cu").read_text()
            entry = cuda[cuda.index("void agent_chunk_scale"):cuda.index("{", cuda.index("void agent_chunk_scale"))]
            # Case 0 compiled the shared draft with the case literal.
            self.assertNotIn("step", entry)
            self.assertIn("c < 32", cuda)

    def test_pipeline_stage_constexpr_drops_forwarded_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            stage = program / "stages/scale"
            stage.mkdir(parents=True)
            (program / "pipeline.json").write_text(
                json.dumps({
                    "inputs": ["a"],
                    "outputs": ["out"],
                    "stages": [{"name": "scale"}],
                    "constructor": [{"name": "step", "dtype": "int32"}],
                }))
            (stage / "draft.py").write_text(SOURCE)
            (stage / "export.json").write_text(json.dumps(SPEC))
            output = root / "out"
            lower(task, program, output, 0)
            model = (output / "model_new.py").read_text()
            compile(model, "model_new.py", "exec")
            # The generated stage call omits the baked constructor argument.
            self.assertIn("cuda_extension.agent_scale(a)", model)
            self.assertNotIn("agent_scale(a, self.step)", model)
