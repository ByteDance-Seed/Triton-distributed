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
"""CPU spec/render tests for the forward_unpack ABI and runtime-int shape exprs.

These tests exercise little_kernel.backends.nv.common.integrations.vibe_cuda_eval's spec loading
and model/binding rendering only (no kernel compile, no GPU). The feature:
export.json may declare an extra forward argument that is a Python tuple of
ints, unpacked in the generated model forward into declared `runtime: true`
int32/int64 scalars, and runtime int scalars may appear in output/workspace
shape expressions and shape equalities (they already worked in launch grids).
"""

import json
import tempfile
from pathlib import Path

import pytest

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval as vibe


def _base_spec():
    return {
        "kernel":
        "probe_kernel",
        "function":
        "probe",
        "inputs": [
            {"name": "dy", "dtype": "float32", "rank": 4},
            {"name": "mask", "dtype": "float32", "rank": 4},
        ],
        "output": {"name": "dx", "dtype": "float32", "shape": ["x_n", "x_c", "x_h", "x_w"]},
        "constructor": [
            {"name": "kernel_h", "dtype": "int32"},
            {"name": "kernel_w", "dtype": "int32"},
        ],
        "scalars": [
            {"name": "x_n", "dtype": "int32", "runtime": True},
            {"name": "x_c", "dtype": "int32", "runtime": True},
            {"name": "x_h", "dtype": "int32", "runtime": True},
            {"name": "x_w", "dtype": "int32", "runtime": True},
        ],
        "launch": {"grid": ["ceil_div(x_h * x_w, 128)", "x_c", "x_n"], "block": [128, 1, 1]},
    }


def _load(spec):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.json"
        path.write_text(json.dumps(spec))
        return vibe._load_spec(path)


def _render(spec_dict):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "export.json"
        path.write_text(json.dumps(spec_dict))
        spec = vibe._load_spec(path)
        return spec, vibe._render_model(spec), vibe._render_binding(spec)


def test_forward_unpack_renders_tuple_forward():
    spec_dict = _base_spec()
    spec_dict["forward_unpack"] = {"x_shape": ["x_n", "x_c", "x_h", "x_w"]}
    spec, model, binding = _render(spec_dict)
    assert "def forward(self, dy, mask, x_shape):" in model
    assert "        x_n, x_c, x_h, x_w = x_shape\n" in model
    # binding order is unchanged: inputs, then all runtime scalars, then ctor
    ext_call = model.strip().splitlines()[-1]
    assert ("cuda_extension.agent_probe(dy, mask, x_n, x_c, x_h, x_w, "
            "self.kernel_h, self.kernel_w)") in ext_call
    # output shape lowered from the runtime scalar names
    assert spec["_outputs"][0]["_shape_cpp"] == ["x_n", "x_c", "x_h", "x_w"]
    # pybind still receives the plain members
    assert 'pybind11::arg("x_w")' in binding
    # launch grid may use the runtime scalars
    assert "lk_ceil_div((x_h * x_w), 128)" in binding


def test_single_member_tuple_trailing_comma():
    spec_dict = _base_spec()
    spec_dict["forward_unpack"] = {"x_shape": ["x_n"]}
    spec, model, _ = _render(spec_dict)
    assert "def forward(self, dy, mask, x_shape, x_c, x_h, x_w):" in model
    assert "        x_n, = x_shape\n" in model


def test_forward_args_permute_pool_with_unpack():
    spec_dict = _base_spec()
    spec_dict["forward_unpack"] = {"x_shape": ["x_n", "x_c"]}
    spec_dict["forward_args"] = ["x_shape", "mask", "x_h", "dy", "x_w"]
    _, model, _ = _render(spec_dict)
    assert "def forward(self, x_shape, mask, x_h, dy, x_w):" in model
    assert "        x_n, x_c = x_shape\n" in model


def test_forward_args_must_permute_pool_not_members():
    spec_dict = _base_spec()
    spec_dict["forward_unpack"] = {"x_shape": ["x_n", "x_c", "x_h", "x_w"]}
    spec_dict["forward_args"] = ["dy", "mask", "x_n"]  # covered member: illegal
    with pytest.raises(ValueError, match="permutation"):
        _load(spec_dict)


def test_unpack_rejects_name_collisions_and_bad_members():
    spec_dict = _base_spec()
    spec_dict["forward_unpack"] = {"dy": ["x_n"]}
    with pytest.raises(ValueError, match="duplicate forward_unpack"):
        _load(spec_dict)
    spec_dict["forward_unpack"] = {"x_shape": ["x_n", "dy"]}
    with pytest.raises(ValueError, match="distinct runtime scalar"):
        _load(spec_dict)
    spec_dict["forward_unpack"] = {"a": ["x_n"], "b": ["x_n"]}
    with pytest.raises(ValueError, match="distinct runtime scalar"):
        _load(spec_dict)
    spec_dict["forward_unpack"] = {"x_shape": ["x_n", "x_n"]}
    with pytest.raises(ValueError, match="distinct runtime scalar"):
        _load(spec_dict)
    spec_dict["forward_unpack"] = {}
    with pytest.raises(ValueError, match="nonempty"):
        _load(spec_dict)


def test_runtime_scalar_in_workspace_and_equality():
    spec_dict = _base_spec()
    spec_dict["workspaces"] = [{"name": "buf", "dtype": "float32", "shape": ["x_n", "x_c"], "zero": True}]
    spec_dict["equalities"] = [["dy[0]", "x_n"], ["dy[1]", "x_c"]]
    spec, _, binding = _render(spec_dict)
    assert "TORCH_CHECK(dy.size(0) == x_n" in binding
    assert "at::empty({x_n, x_c}" in binding


def test_float_runtime_scalar_rejected_in_int_shape():
    spec_dict = _base_spec()
    spec_dict["scalars"].append({"name": "alpha", "dtype": "float32", "runtime": True})
    spec_dict["output"]["shape"] = ["x_n", "x_c", "x_h", "alpha"]
    with pytest.raises(ValueError, match="unsupported shape expression"):
        _load(spec_dict)


def test_backward_compat_without_unpack():
    spec_dict = _base_spec()
    spec_dict["scalars"].append({"name": "extra", "dtype": "int64", "runtime": True})
    spec, model, _ = _render(spec_dict)
    assert "= x_shape" not in model
    assert "def forward(self, dy, mask, x_n, x_c, x_h, x_w, extra):" in model
    assert spec["_forward_unpack"] == {}
    assert spec["_forward_pool"] == ["dy", "mask", "x_n", "x_c", "x_h", "x_w", "extra"]
