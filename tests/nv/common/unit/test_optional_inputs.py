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
"""Validation and render contracts for optional-tensor operands in export specs.

An input with ``"optional": true`` is a tensor argument that may be None at
runtime: the generated Model.forward gives it a None default, the pybind
binding takes ``const c10::optional<torch::Tensor>&``, and the kernel ABI
carries the raw pointer (nullptr when absent) plus an injected int32 presence
scalar ``<name>_present`` right after the tensor in ``kernel_args``. Optional
inputs must be trailing model inputs and trailing forward arguments, and their
sizes may never feed shape/equality/scalar/launch-grid expressions.
"""
import copy
import json

import pytest

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval

BASE_SPEC = {
    "kernel":
    "k",
    "inputs": [
        {"name": "x", "dtype": "float16", "rank": 2},
        {"name": "w", "dtype": "float32", "rank": 1},
        {"name": "bias", "dtype": "float32", "rank": 1, "optional": True},
    ],
    "outputs": [{"name": "out", "dtype": "float16", "shape": ["x[0]", "x[1]"]}],
    "scalars": [],
    "constructor": [{"name": "eps", "dtype": "float32"}],
    "launch": {"grid": ["x[0]", 1, 1], "block": [256, 1, 1]},
}


def _load(spec):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "export.json"
        path.write_text(json.dumps(spec))
        return vibe_cuda_eval._load_spec(path)


def _spec():
    return copy.deepcopy(BASE_SPEC)


def test_presence_scalar_injected_after_optional_kernel_arg():
    spec = _load(_spec())
    assert spec["_optional_inputs"] == ["bias"]
    assert spec["_presence_args"] == {"bias_present": "bias"}
    assert spec["_kernel_args"] == ["x", "w", "bias", "bias_present", "out"]
    names = vibe_cuda_eval._kernel_param_names(spec)
    assert names == ["x", "w", "bias", "bias_present", "out"]
    assert vibe_cuda_eval._launcher_arg_decl(spec, "bias_present") == "int32_t bias_present"
    assert vibe_cuda_eval._cuda_kernel_arg(spec, "bias_present") == "static_cast<int32_t>(bias_present)"


def test_model_forward_defaults_none_and_calls_by_keyword():
    model = vibe_cuda_eval._render_model(_load(_spec()))
    assert "def forward(self, x, w, bias=None):" in model
    assert "bias=bias" in model
    # Positional non-optional inputs precede the constructor values.
    assert "cuda_extension.agent_lk_forward(x, w, self.eps, bias=bias)" in model


def test_binding_optional_signature_checks_and_launch_args():
    spec = _load(_spec())
    spec["launch"]["_shared"] = 0
    binding = vibe_cuda_eval._render_binding(spec)
    # pybind11 requires defaults trailing, so the optional tensor is declared
    # after runtime scalars and constructor arguments.
    assert ("agent_lk_forward(torch::Tensor x, torch::Tensor w, float eps, "
            "const c10::optional<torch::Tensor>& bias)") in binding
    assert 'pybind11::arg("bias") = pybind11::none()' in binding
    # TORCH_CHECKs of an optional tensor run only when it is defined.
    assert "if (bias.has_value()) {" in binding
    assert "bias->is_cuda()" in binding
    assert "(bias.has_value() ? bias->data_ptr() : nullptr)" in binding
    assert "(bias.has_value() ? 1 : 0)" in binding


def test_forward_args_must_place_optional_last():
    spec = _spec()
    spec["forward_args"] = ["bias", "x", "w"]
    with pytest.raises(ValueError, match="trailing forward arguments"):
        _load(spec)
    spec = _spec()
    spec["forward_args"] = ["x", "w", "bias"]
    assert "bias=None" in vibe_cuda_eval._render_model(_load(spec))


def test_first_input_cannot_be_optional():
    spec = _spec()
    spec["inputs"][0]["optional"] = True
    with pytest.raises(ValueError, match="first input cannot be optional"):
        _load(spec)


def test_optional_inputs_must_be_trailing_inputs():
    spec = _spec()
    spec["inputs"].append({"name": "z", "dtype": "int32", "rank": 1})
    with pytest.raises(ValueError, match="trailing model inputs"):
        _load(spec)


def test_optional_flag_must_be_boolean():
    spec = _spec()
    spec["inputs"][2]["optional"] = "yes"
    with pytest.raises(ValueError, match="optional flag for bias must be boolean"):
        _load(spec)


@pytest.mark.parametrize("mutate", [
    lambda s: s["outputs"][0].update(shape=["x[0]", "bias[0]"]),
    lambda s: s.update(equalities=[["bias[0]", "x[1]"]]),
    lambda s: s["launch"].update(grid=["bias[0]", 1, 1]),
    lambda s: s.update(scalars=[{"name": "C", "dtype": "int32", "expr": "bias[0]"}]),
    lambda s: s.update(workspaces=[{"name": "ws", "dtype": "float32", "shape": ["bias[0]"]}]),
])
def test_optional_sizes_rejected_in_expressions(mutate):
    spec = _spec()
    mutate(spec)
    with pytest.raises(ValueError, match="invalid tensor dimension"):
        _load(spec)


def test_optional_cannot_alias_an_output():
    spec = _spec()
    spec["outputs"][0].update(alias_of="bias", shape=None, dtype="float32")
    with pytest.raises(ValueError, match="cannot alias an output"):
        _load(spec)


def test_optional_cannot_be_transform_source():
    spec = _spec()
    spec["transforms"] = [{"op": "alias", "name": "t", "source": "bias"}]
    with pytest.raises(ValueError, match="cannot be a transform source"):
        _load(spec)


def test_optional_cannot_back_a_tma_descriptor():
    spec = _spec()
    spec["tma_descriptors"] = [{"name": "d", "tensor": "bias", "rank": 2, "inner": 1, "outer": 1, "box": [1, 1]}]
    with pytest.raises(ValueError, match="rank-1/2/3/4/5 tensor"):
        _load(spec)


def test_optional_sizes_rejected_in_tma_descriptor_fields():
    # The descriptor tensor itself is non-optional, but its extents still
    # must not come from an optional tensor's sizes (an optional tensor may
    # be absent at runtime, so its sizes are filtered out of every
    # expression's ranks map).
    spec = _spec()
    spec["tma_descriptors"] = [{
        "name": "d", "tensor": "x", "rank": 2, "inner": "bias[0]", "outer": "x[0]", "outer_stride": "x[1]", "box":
        [1, 1]
    }]
    with pytest.raises(ValueError, match="invalid tensor dimension"):
        _load(spec)


def test_tma_descriptor_fields_unaffected_without_optional_refs():
    spec = _spec()
    spec["tma_descriptors"] = [{
        "name": "d", "tensor": "x", "rank": 2, "inner": "x[1]", "outer": "x[0]", "outer_stride": "x[1]", "box":
        [16, 16]
    }]
    loaded = _load(spec)
    assert loaded["_optional_inputs"] == ["bias"]
    item = [d for d in loaded["tma_descriptors"] if d["name"] == "d"][0]
    assert item["_inner_cpp"] == "x.size(1)"
    assert item["_outer_cpp"] == "x.size(0)"


def test_foreach_rejects_optional_inputs():
    spec = _spec()
    spec["outputs"] = [{"name": "out", "dtype": "float32", "shape": ["x[0]"]}]
    spec["foreach"] = {"inputs": ["bias"]}
    with pytest.raises(ValueError, match="foreach cannot be combined"):
        _load(spec)


def test_presence_name_collision_rejected():
    spec = _spec()
    spec["scalars"] = [{"name": "bias_present", "dtype": "int32", "expr": "x[0]"}]
    with pytest.raises(ValueError, match="collides with a declared name"):
        _load(spec)


def test_specs_without_optional_inputs_are_unchanged():
    spec = _spec()
    spec["inputs"][2].pop("optional")
    loaded = _load(spec)
    assert loaded["_optional_inputs"] == []
    assert loaded["_presence_args"] == {}
    assert loaded["_kernel_args"] == ["x", "w", "bias", "out"]
    model = vibe_cuda_eval._render_model(loaded)
    assert "bias=None" not in model
    assert "bias=bias" not in model
    loaded["launch"]["_shared"] = 0
    binding = vibe_cuda_eval._render_binding(loaded)
    assert "c10::optional" not in binding
    assert "pybind11::none()" not in binding
