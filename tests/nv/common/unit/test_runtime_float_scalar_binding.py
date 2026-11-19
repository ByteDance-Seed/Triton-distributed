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
"""CPU regression for runtime float32 forward scalars.

A manifest may accept plain host fp32 values (for example alpha and beta) in
Model.forward and pass them by value through the host binding and launcher ABI.
These checks generate and inspect Python/C++/CUDA source text only; they do not
invoke nvcc or use a GPU.
"""

import json
import tempfile
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.backends.nv.common.integrations import vibe_cuda_eval as vibe


def _scalar_spec():
    return {
        "kernel":
        "axpy",
        "function":
        "axpy",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}],
        "output": {"name": "y", "dtype": "float32", "shape": ["x[0]"]},
        "scalars": [
            {"name": "alpha", "dtype": "float32", "runtime": True},
            {"name": "beta", "dtype": "float32", "runtime": True},
        ],
        "kernel_args": ["x", "y", "alpha", "beta"],
        "launch": {"grid": ["ceil_div(x[0], 128)", "1", "1"], "block": [128, 1, 1]},
    }


def _render():
    with tempfile.TemporaryDirectory() as tmp:
        spec_path = Path(tmp) / "export.json"
        spec_path.write_text(json.dumps(_scalar_spec()))
        spec = vibe._load_spec(spec_path)
        return spec, vibe._render_model(spec), vibe._render_binding(spec)


def test_float32_runtime_scalars_follow_tensor_inputs():
    spec, model, binding = _render()

    assert spec["_scalar_types"] == {"alpha": "float32", "beta": "float32"}
    assert spec["_runtime_kernel_args"] == ["x", "y", "alpha", "beta"]

    assert "def forward(self, x, alpha, beta):" in model
    assert "cuda_extension.agent_axpy(x, alpha, beta)" in model

    # The host wrapper receives Python floats by value, after tensor inputs.
    assert ("static torch::Tensor agent_axpy("
            "torch::Tensor x, float alpha, float beta)") in binding
    assert ('m.def("agent_axpy", &agent_axpy, pybind11::arg("x"), '
            'pybind11::arg("alpha"), pybind11::arg("beta"));') in binding

    # The launcher ABI keeps the same scalar order and C++ float value type.
    assert ('extern "C" cudaError_t agent_lk_launch_axpy('
            "void* x, void* y, float alpha, float beta,") in binding
    assert "agent_lk_launch_axpy(x.data_ptr(), y.data_ptr(), alpha, beta," in binding
    assert vibe._launcher_arg_decl(spec, "alpha") == "float alpha"
    assert vibe._cuda_kernel_arg(spec, "beta") == "static_cast<float>(beta)"


def test_float32_scalars_reach_generated_kernel_signature():
    source = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def axpy(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
         n: ll.int32, alpha: ll.float32, beta: ll.float32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        y[i] = alpha * x[i] + beta
'''
    spec = _scalar_spec()
    spec["scalars"].insert(0, {"name": "n", "expr": "x[0]"})
    spec["kernel_args"] = ["x", "y", "n", "alpha", "beta"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        task = root / "task"
        task.mkdir()
        (task / "model.py").write_text("class Model: pass\n")
        (task / "test_cases.py").write_text("test_cases = []\n")
        program = root / "program"
        program.mkdir()
        (program / "draft.py").write_text(source)
        (program / "export.json").write_text(json.dumps(spec))
        output = root / "output"
        lower(task, program, output, 0)
        cuda = (output / "kernels/agent_lk_axpy.cu").read_text()
        binding = (output / "kernels/agent_lk_axpy_binding.cpp").read_text()
        model = (output / "model_new.py").read_text()

    # alpha and beta remain real fp32 kernel parameters, while the derived
    # length n stays an internal host/launch value and never enters forward.
    assert "float alpha,\n    float beta\n) {" in cuda
    assert "static_cast<float>(alpha),\n        static_cast<float>(beta)" in cuda
    assert "const int64_t n = x.size(0);" in binding
    assert ("static torch::Tensor agent_axpy("
            "torch::Tensor x, float alpha, float beta)") in binding
    assert "def forward(self, x, alpha, beta):" in model
    assert "cuda_extension.agent_axpy(x, alpha, beta)" in model
