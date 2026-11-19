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
"""A shared LittleKernel compiler accepts arbitrary authored DSL programs."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
            out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        out[i] = a[i] + b[i]
'''

SPEC = {
    "kernel": "vec_add",
    "function": "add",
    "inputs": [
        {"name": "a", "dtype": "float32", "rank": 1},
        {"name": "b", "dtype": "float32", "rank": 1},
    ],
    "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
    "equalities": [["a[0]", "b[0]"]],
    "scalars": [{"name": "n", "expr": "a[0]"}],
    "launch": {"grid": ["ceil_div(a[0], 256)", 1, 1], "block": [256, 1, 1]},
}


class ProgramBridgeTests(unittest.TestCase):

    def test_same_compiler_accepts_another_program(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            program.mkdir()
            (program / "draft.py").write_text(
                SOURCE.replace("vec_add", "vec_mul").replace("a[i] + b[i]", "a[i] * b[i]"))
            (program / "export.json").write_text(json.dumps({
                **SPEC,
                "kernel": "vec_mul",
                "function": "mul",
            }))
            manifest = lower(task, program, root / "output", 0)
            self.assertEqual(manifest["compiler"], "LittleKernel")
            self.assertIn("a[i] * b[i]", (root / "output/kernels/agent_lk_mul.cu").read_text())

    def test_generic_frontend_emits_cuda_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            program.mkdir()
            (program / "draft.py").write_text(SOURCE)
            (program / "export.json").write_text(json.dumps(SPEC))
            output = root / "output"
            manifest = lower(task, program, output, 0)
            self.assertEqual(manifest["compiler"], "LittleKernel")
            self.assertEqual(manifest["backend"], "cuda-extension")
            self.assertIn("__global__", (output / "kernels/agent_lk_add.cu").read_text())
            self.assertIn("REGISTER_BINDING", (output / "kernels/agent_lk_add_binding.cpp").read_text())
            self.assertIn("cuda_extension.agent_add", (output / "model_new.py").read_text())

    def test_missing_program_is_honest_unsupported_failure_and_case_specific(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            program.mkdir()
            # No artifact for case 0: source policy requires an honest
            # unsupported failure, never a delegation to the task reference
            # model or a no-op CUDA placeholder labelled as compiled.
            with self.assertRaises(NotImplementedError):
                lower(task, program, root / "missing", 0)
            self.assertFalse((root / "missing").exists())
            case = program / "case-001"
            case.mkdir()
            (case / "draft.py").write_text(SOURCE)
            (case / "export.json").write_text(json.dumps(SPEC))
            self.assertEqual(lower(task, program, root / "out", 1)["case_index"], 1)

    def test_constructor_scalar_is_forwarded_to_cuda(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            program.mkdir()
            (program / "draft.py").write_text(
                SOURCE.replace("vec_add", "scaled_add").replace("n: ll.int32",
                                                                "n: ll.int32, scale: ll.float32").replace(
                                                                    "a[i] + b[i]", "a[i] + scale * b[i]"))
            (program / "export.json").write_text(
                json.dumps({
                    **SPEC,
                    "kernel": "scaled_add",
                    "function": "scaled_add",
                    "constructor": [{"name": "scale", "dtype": "float32", "default": 1.0}],
                    "kernel_args": ["a", "b", "out", "n", "scale"],
                }))
            output = root / "out"
            lower(task, program, output, 0)
            model = (output / "model_new.py").read_text()
            compile(model, "model_new.py", "exec")
            self.assertIn("def __init__(self, scale=1.0)", model)
            self.assertIn("cuda_extension.agent_scaled_add(a, b, self.scale)", model)
            binding = (output / "kernels/agent_lk_scaled_add_binding.cpp").read_text()
            self.assertIn("float scale", binding)
            self.assertIn('pybind11::arg("scale")', binding)
            cuda = (output / "kernels/agent_lk_scaled_add.cu").read_text()
            self.assertIn("static_cast<float>(scale)", cuda)

    def test_pipeline_compiles_two_authored_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            add = program / "stages/add"
            mul = program / "stages/mul"
            add.mkdir(parents=True)
            mul.mkdir(parents=True)
            (program / "pipeline.json").write_text(
                json.dumps({
                    "inputs": ["a", "b"],
                    "outputs": ["out"],
                    "stages": [{"name": "add"}, {"name": "mul"}],
                }))
            (add / "draft.py").write_text(SOURCE.replace("out", "tmp"))
            (add / "export.json").write_text(
                json.dumps({
                    **SPEC,
                    "output": {"name": "tmp", "dtype": "float32", "shape": ["a[0]"]},
                }))
            (mul / "draft.py").write_text(
                SOURCE.replace("vec_add",
                               "vec_mul").replace("a: ll.Tensor",
                                                  "tmp: ll.Tensor").replace("out[i] = a[i] + b[i]",
                                                                            "out[i] = tmp[i] * b[i]"))
            (mul / "export.json").write_text(
                json.dumps({
                    **SPEC,
                    "kernel":
                    "vec_mul",
                    "function":
                    "mul",
                    "inputs": [{"name": "tmp", "dtype": "float32", "rank": 1},
                               {"name": "b", "dtype": "float32", "rank": 1}],
                    "output": {"name": "out", "dtype": "float32", "shape": ["tmp[0]"]},
                    "equalities": [["tmp[0]", "b[0]"]],
                    "scalars": [{"name": "n", "expr": "tmp[0]"}],
                    "launch": {"grid": ["ceil_div(tmp[0], 256)", 1, 1], "block": [256, 1, 1]},
                }))
            output = root / "out"
            artifact = lower(task, program, output, 0)
            self.assertEqual(artifact["frontend"], "agent-authored-littlekernel-pipeline")
            self.assertEqual(len(list((output / "kernels").glob("*.cu"))), 2)
            self.assertEqual(len(list((output / "kernels").glob("*.cpp"))), 2)
            model = (output / "model_new.py").read_text()
            compile(model, "model_new.py", "exec")
            self.assertIn("tmp = cuda_extension.agent_add(a, b)", model)
            self.assertIn("out = cuda_extension.agent_mul(tmp, b)", model)

    def test_pipeline_forwards_runtime_scalars_in_stage_binding_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            stage = program / "stages/add"
            stage.mkdir(parents=True)
            (program / "pipeline.json").write_text(
                json.dumps({
                    "inputs": ["a", "b"],
                    "outputs": ["out"],
                    "runtime_scalars": ["alpha", "beta"],
                    "stages": [{"name": "add"}],
                }))
            (stage / "draft.py").write_text(
                SOURCE.replace("n: ll.int32", "n: ll.int32, alpha: ll.int32, beta: ll.int32").replace(
                    "a[i] + b[i]", "a[i] * alpha + b[i] * beta"))
            (stage / "export.json").write_text(
                json.dumps({
                    **SPEC,
                    # The stage ABI order is independent of the pipeline's ABI.
                    "scalars":
                    SPEC["scalars"] + [
                        {"name": "beta", "dtype": "int32", "runtime": True},
                        {"name": "alpha", "dtype": "int32", "runtime": True},
                    ],
                    "kernel_args": ["a", "b", "out", "n", "alpha", "beta"],
                }))
            output = root / "out"
            lower(task, program, output, 0)
            tree = ast.parse((output / "model_new.py").read_text())
            # Execute the generated wrapper with a stand-in binding; this
            # tests positional ABI semantics without importing Torch or CUDA.
            tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
            namespace = {
                "torch": SimpleNamespace(nn=SimpleNamespace(Module=object)),
                "cuda_extension": SimpleNamespace(agent_add=lambda a, b, beta, alpha: (alpha, beta)),
            }
            exec(compile(tree, "model_new.py", "exec"), namespace)
            self.assertEqual(namespace["Model"]().forward(None, None, 3, 7), (3, 7))

    def test_ignored_constructor_argument_stays_out_of_binding(self):
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
                json.dumps({
                    **SPEC,
                    "constructor": [{"name": "dtype", "dtype": "ignored"}],
                }))
            output = root / "out"
            lower(task, program, output, 0)
            model = (output / "model_new.py").read_text()
            compile(model, "model_new.py", "exec")
            self.assertIn("def __init__(self, dtype)", model)
            self.assertIn("cuda_extension.agent_add(a, b)", model)
            self.assertNotIn('pybind11::arg("dtype")', (output / "kernels/agent_lk_add_binding.cpp").read_text())

    def test_cuda_emitter_preserves_readable_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.mkdir()
            (task / "model.py").write_text("class Model: pass\n")
            (task / "test_cases.py").write_text("test_cases = []\n")
            program = root / "program"
            program.mkdir()
            (program / "draft.py").write_text('''\
"""A readable vector add with alternating signs."""

import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def vec_add(a: ll.Tensor[ll.float32], b: ll.Tensor[ll.float32],
            out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    """Add adjacent values with a lane-dependent sign."""
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        if (i & 1) == 0:
            out[i] = a[i] + b[i]
        else:
            out[i] = a[i] - b[i]
''')
            (program / "export.json").write_text(json.dumps(SPEC))
            output = root / "out"
            lower(task, program, output, 0)
            cuda = (output / "kernels/agent_lk_add.cu").read_text()
            self.assertIn("// A readable vector add with alternating signs.", cuda)
            self.assertIn("// Add adjacent values with a lane-dependent sign.", cuda)
            self.assertIn('extern "C" __global__', cuda)
            self.assertIn("int32_t i = blockIdx.x * 256 + threadIdx.x;", cuda)
            self.assertIn("(i & 1) == 0", cuda)
            self.assertNotIn("#include <tuple>", cuda)
            self.assertNotIn("using TmaDescriptor", cuda)


if __name__ == "__main__":
    unittest.main()
