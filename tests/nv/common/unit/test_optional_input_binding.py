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
"""Unit tests for optional tensor inputs in the cuda-extension binding.

An input declared with "optional": true is exposed to pybind as
c10::optional<torch::Tensor>; the harness may pass None for a feature-disabled
tensor. The generated binding must accept None, skip value checks when the
tensor is absent, and pass nullptr to the kernel launcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_spec, _render_binding


def _write_spec(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "export.json"
    path.write_text(json.dumps(data))
    return path


def _base_spec(**overrides) -> dict:
    data = {
        "kernel":
        "k",
        "function":
        "copy",
        "inputs": [
            {"name": "x", "dtype": "float32", "rank": 1},
            {"name": "z", "dtype": "float32", "rank": 1, "optional": True},
        ],
        "output": {"name": "y", "dtype": "float32", "shape": ["x[0]"]},
        "scalars": [],
        "workspaces": [],
        "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [256, 1, 1]},
    }
    data.update(overrides)
    return data


def test_optional_binding_renders_none_capable_signature(tmp_path):
    spec = _load_spec(_write_spec(tmp_path, _base_spec()))
    binding = _render_binding(spec)
    assert "const c10::optional<torch::Tensor>& lk_opt_z" in binding
    # non-optional input keeps its plain tensor parameter
    assert "torch::Tensor x" in binding
    assert "const c10::optional<torch::Tensor>& lk_opt_x" not in binding
    # local alias keeps downstream rendering unchanged
    assert "torch::Tensor z = lk_opt_z.value_or(torch::Tensor());" in binding


def test_optional_binding_guards_checks_and_launch_arg(tmp_path):
    spec = _load_spec(_write_spec(tmp_path, _base_spec()))
    binding = _render_binding(spec)
    assert "if (z.defined()) {" in binding
    assert 'TORCH_CHECK(z.is_cuda(), "z must be on CUDA");' in binding
    assert 'TORCH_CHECK(z.scalar_type() == at::kFloat, "z dtype mismatch");' in binding
    # launcher receives nullptr when the tensor is absent
    assert "z.defined() ? z.data_ptr() : nullptr" in binding
    # non-optional input is passed directly
    assert "x.data_ptr()," in binding
    # include for the c10::optional spelling
    assert "#include <c10/util/Optional.h>" in binding


def test_no_optional_inputs_renders_unchanged(tmp_path):
    data = _base_spec()
    data["inputs"] = [
        {"name": "x", "dtype": "float32", "rank": 1},
        {"name": "z", "dtype": "float32", "rank": 1},
    ]
    spec = _load_spec(_write_spec(tmp_path, data))
    binding = _render_binding(spec)
    assert "c10::optional" not in binding
    assert "lk_opt_" not in binding
    assert "z.data_ptr()," in binding


def test_first_input_must_not_be_optional(tmp_path):
    data = _base_spec()
    data["inputs"][0]["optional"] = True
    with pytest.raises(ValueError, match="first input must not be optional"):
        _load_spec(_write_spec(tmp_path, data))


def test_optional_flag_must_be_boolean(tmp_path):
    data = _base_spec()
    data["inputs"][1]["optional"] = "yes"
    with pytest.raises(ValueError, match="optional flag for z must be boolean"):
        _load_spec(_write_spec(tmp_path, data))


def test_alias_of_must_not_target_optional_input(tmp_path):
    data = _base_spec()
    data["output"] = {"name": "y", "dtype": "float32", "alias_of": "z"}
    with pytest.raises(ValueError, match="must not alias an optional input"):
        _load_spec(_write_spec(tmp_path, data))


def test_foreach_rejects_optional_inputs(tmp_path):
    data = _base_spec()
    data["foreach"] = {"inputs": ["x"]}
    with pytest.raises(ValueError, match="foreach cannot be combined with optional inputs"):
        _load_spec(_write_spec(tmp_path, data))
