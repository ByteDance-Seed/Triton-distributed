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

import copy
import json
import tempfile
from pathlib import Path

import pytest

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_spec, _render_binding

BASE_SPEC = {
    "kernel": "k",
    "inputs": [
        {"name": "w", "dtype": "bfloat16", "rank": 3},
    ],
    "outputs": [{"name": "out", "dtype": "float32", "shape": ["w[0]"]}],
    "scalars": [],
    "launch": {"grid": ["w[0]", 1, 1], "block": [128, 1, 1]},
}


def _load(spec):
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "export.json"
        path.write_text(json.dumps(spec))
        return _load_spec(path)


def test_rank3_tensor_can_declare_flattened_leading_2d_tma():
    spec = copy.deepcopy(BASE_SPEC)
    spec["tma_descriptors"] = [{
        "name": "dW",
        "tensor": "w",
        "rank": 2,
        "flatten_leading": True,
        "box": [64, 256],
        "swizzle_bytes": 128,
    }]

    loaded = _load(spec)
    (item, ) = loaded["tma_descriptors"]
    assert item["_inner_cpp"] == "w.size(2)"
    assert item["_outer_cpp"] == "(w.size(0) * w.size(1))"
    assert item["_outer_stride_cpp"] == "w.size(2)"

    binding = _render_binding(loaded)
    assert ("agent_lk_make_tma_2d(&dW, w.data_ptr(), w.size(2), "
            "(w.size(0) * w.size(1)), w.size(2), 64, 256, "
            "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, CU_TENSOR_MAP_SWIZZLE_128B") in binding


def test_rank4_tensor_flattens_all_leading_dimensions():
    spec = copy.deepcopy(BASE_SPEC)
    spec["inputs"] = [{"name": "w", "dtype": "bfloat16", "rank": 4}]
    spec["outputs"] = [{"name": "out", "dtype": "float32", "shape": ["w[0]"]}]
    spec["launch"] = {"grid": ["w[0]", 1, 1], "block": [128, 1, 1]}
    spec["tma_descriptors"] = [{
        "name": "dW",
        "tensor": "w",
        "rank": 2,
        "flatten_leading": True,
        "box": [64, 128],
    }]

    loaded = _load(spec)
    (item, ) = loaded["tma_descriptors"]
    assert item["_inner_cpp"] == "w.size(3)"
    assert item["_outer_cpp"] == "((w.size(0) * w.size(1)) * w.size(2))"
    assert item["_outer_stride_cpp"] == "w.size(3)"


def test_flatten_leading_rejects_noncanonical_view():
    spec = copy.deepcopy(BASE_SPEC)
    spec["tma_descriptors"] = [{
        "name": "dW",
        "tensor": "w",
        "rank": 2,
        "flatten_leading": True,
        "inner": "w[2]",
        "outer": "w[0] * w[1]",
        "outer_stride": "w[1] * w[2]",
        "box": [64, 256],
    }]

    with pytest.raises(ValueError, match="flatten_leading TMA descriptor outer_stride"):
        _load(spec)


def test_flatten_leading_requires_rank2_descriptor():
    spec = copy.deepcopy(BASE_SPEC)
    spec["tma_descriptors"] = [{
        "name": "dW",
        "tensor": "w",
        "rank": 3,
        "flatten_leading": True,
        "box": [64, 8, 1],
    }]

    with pytest.raises(ValueError, match="flatten_leading requires"):
        _load(spec)
