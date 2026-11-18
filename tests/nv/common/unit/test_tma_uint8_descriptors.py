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
"""1-byte TMA tensor-map dtypes (uint8 / fp8 storage).

TMA descriptor validation and codegen in
integrations/vibe_cuda_eval.py (_load_spec TMA descriptor pass +
_render_binding descriptor emission). FP8 tensors (torch float8_e5m2 /
float8_e4m3fn, exposed to kernels as uint8_t* by the storage table) move
through TMA as raw 1-byte elements: the consumer (e.g. WGMMA) interprets the
bytes, so the descriptor carries CU_TENSOR_MAP_DATA_TYPE_UINT8 with
element_size 1. Keeping element_size=1 means the emitted
`stride * element_size` byte multiplication in agent_lk_make_tma_2d/3d stays
correct, and a 128-column fp8 box is 128 bytes, matching the existing
CU_TENSOR_MAP_SWIZZLE_128B mode.
"""

import copy
import json
from pathlib import Path

import pytest

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval
from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _TMA_DTYPES, _load_spec, _render_binding

BASE_SPEC = {
    "kernel":
    "k",
    "inputs": [
        {"name": "A", "dtype": "float8_e5m2", "rank": 2},
        {"name": "B", "dtype": "float8_e4m3fn", "rank": 2},
        {"name": "scale", "dtype": "float32", "rank": 1},
    ],
    "outputs": [{"name": "out", "dtype": "float32", "shape": ["A[0]", "B[0]"]}],
    "scalars": [],
    "launch": {"grid": ["A[0]", 1, 1], "block": [128, 1, 1]},
}


def _load(spec):
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "export.json"
        path.write_text(json.dumps(spec))
        return _load_spec(path)


def _spec():
    return copy.deepcopy(BASE_SPEC)


def test_one_byte_entries_in_tma_dtype_table():
    for name in ("uint8", "float8_e4m3fn", "float8_e5m2"):
        dtype, element_size = _TMA_DTYPES[name]
        assert dtype == "CU_TENSOR_MAP_DATA_TYPE_UINT8"
        assert element_size == 1


def test_fp8_and_uint8_tensors_accept_tma_descriptors():
    spec = _spec()
    spec["inputs"].append({"name": "mask", "dtype": "uint8", "rank": 2})
    spec["tma_descriptors"] = [
        {
            "name": "dA", "tensor": "A", "rank": 2, "inner": "A[1]", "outer": "A[0]", "outer_stride": "A[1]", "box":
            [128, 8], "swizzle_bytes": 128
        },
        {
            "name": "dB", "tensor": "B", "rank": 2, "inner": "B[1]", "outer": "B[0]", "outer_stride": "B[1]", "box":
            [128, 8], "swizzle_bytes": 128
        },
        {
            "name": "dM", "tensor": "mask", "rank": 2, "inner": "mask[1]", "outer": "mask[0]", "outer_stride":
            "mask[1]", "box": [64, 4]
        },
    ]
    loaded = _load(spec)
    items = {item["name"]: item for item in loaded["tma_descriptors"]}
    assert items["dA"]["_dtype"] == "float8_e5m2"
    assert items["dB"]["_dtype"] == "float8_e4m3fn"
    assert items["dM"]["_dtype"] == "uint8"

    binding = _render_binding(loaded)
    # element_size stays 1 so the helper's byte-stride multiplication holds;
    # the 128-byte swizzle fences the 128-byte fp8 box without a dtype change.
    assert ("agent_lk_make_tma_2d(&dA, A.data_ptr(), A.size(1), A.size(0), "
            "A.size(1), 128, 8, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, "
            "CU_TENSOR_MAP_SWIZZLE_128B") in binding
    assert ("agent_lk_make_tma_2d(&dM, mask.data_ptr(), mask.size(1), "
            "mask.size(0), mask.size(1), 64, 4, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, "
            "CU_TENSOR_MAP_SWIZZLE_NONE") in binding


def test_uint8_tensor_accepts_rank3_tma_descriptor():
    spec = _spec()
    spec["inputs"] = [{"name": "Q", "dtype": "float8_e5m2", "rank": 4}]
    spec["outputs"] = [{"name": "out", "dtype": "float32", "shape": ["Q[0]", "Q[1]"]}]
    spec["launch"] = {"grid": ["Q[0]", 1, 1], "block": [128, 1, 1]}
    spec["tma_descriptors"] = [
        {
            "name": "dQ", "tensor": "Q", "rank": 3, "dim0": "Q[3]", "dim1": "Q[2]", "dim2": "Q[0] * Q[1]", "stride1":
            "Q[3]", "stride2": "Q[2] * Q[3]", "box": [128, 8, 1], "swizzle_bytes": 128
        },
    ]
    loaded = _load(spec)
    binding = _render_binding(loaded)
    assert ("agent_lk_make_tma_3d(&dQ, Q.data_ptr(), Q.size(3), Q.size(2), "
            "(Q.size(0) * Q.size(1)), Q.size(3), (Q.size(2) * Q.size(3)), "
            "128, 8, 1, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, "
            "CU_TENSOR_MAP_SWIZZLE_128B") in binding


def test_unsupported_tma_dtypes_still_rejected():
    spec = _spec()
    spec["inputs"] = [{"name": "wide", "dtype": "float64", "rank": 2}]
    spec["outputs"] = [{"name": "out", "dtype": "float32", "shape": ["wide[0]", "wide[1]"]}]
    spec["launch"] = {"grid": ["wide[0]", 1, 1], "block": [128, 1, 1]}
    spec["tma_descriptors"] = [{
        "name": "dW", "tensor": "wide", "rank": 2, "inner": "wide[1]", "outer": "wide[0]", "outer_stride": "wide[1]",
        "box": [8, 8]
    }]
    with pytest.raises(ValueError, match="unsupported TMA tensor dtype"):
        _load(spec)


def test_int8_not_backfilled_into_tma_table():
    # int8 tensors keep binding as int8 for kernels, but were never TMA-
    # described; this capability adds only the 1-byte raw-byte view.
    assert "int8" not in _TMA_DTYPES
