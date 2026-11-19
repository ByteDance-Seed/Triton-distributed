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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
"""Typed 16 x int8 lane GMEM vector moves (gmem_ld/st_v16s8).

Semantic key: ir.gmem_ld_v16s8.typed_lanes.

One 16-byte transaction exchanges SIXTEEN int8-typed lane locals through the
caller's own registers, mirroring the gmem_ld_v4s32 four-int32-lane family
for one-byte signed elements.  This differs from the opaque
gmem_ld_u8x16 twin, which stages the same bytes as two uint64 bit patterns
whose vec_u8x16_lane extractor zero-extends to uint32 (0..255) and forces
per-lane unpacking; v16s8 delivers signed int8 lanes directly.

Contract pinned here (CPU IR/codegen regression; no GPU required):

* ``gmem_ld_v16s8`` / ``gmem_ld_v16s8_nc`` write 16 int8 locals through
  ``&lane``; ``_nc`` routes the 16-byte load through ``__ldg`` (.nc);
* ``gmem_st_v16s8`` / ``gmem_st_v16s8_cs`` consume 16 int8 value lanes;
  ``_cs`` is the streaming (``__stcs``, evict-first) cache hint;
* lane locals must be declared ``ll.int8`` -- a mismatched statically-typed
  local raises TypeInferenceError naming the intrinsic and both types,
  same discipline as the v4s32 family;
* by-ref load lanes are never emitted ``static constexpr`` (their address
  is taken), matching test_static_int_extents' v4s32 pin.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
import little_kernel as lk
from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
import little_kernel.backends.nv.common.language.intrin.memory as memory
from little_kernel.language.builtin_base import CODEGEN_FUNC_ATTR


def _export_json(name: str) -> str:
    return json.dumps({
        "kernel": name,
        "function": name,
        "inputs": [{"name": "x", "dtype": "int8", "rank": 1}],
        "output": {"name": "out", "dtype": "int8", "shape": ["x[0]"]},
        "scalars": [],
        "launch": {"grid": ["ceil_div(x[0], 32)", 1, 1], "block": [32, 1, 1]},
    })


def _lower(tmp: Path, source: str, name: str, dtype: str = "int8") -> str:
    task = tmp / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = tmp / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(_export_json(name))
    if dtype != "int8":
        spec = json.loads((program / "export.json").read_text())
        spec["inputs"][0]["dtype"] = dtype
        spec["output"]["dtype"] = dtype
        (program / "export.json").write_text(json.dumps(spec))
    output = tmp / "output"
    lower(task, program, output, 0)
    kernels = sorted((output / "kernels").glob("*.cu"))
    assert len(kernels) == 1, kernels
    return kernels[0].read_text()


def _lane_decls(prefix: str = "b", ann: str = "ll.int8 = 0") -> str:
    return "\n    ".join(f"{prefix}{k}: {ann}" for k in range(16))


def _lane_args(prefix: str = "b") -> str:
    return ", ".join(f"{prefix}{k}" for k in range(16))


def _kernel(load: str, store: str, name: str) -> str:
    return f"""\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def {name}(x: ll.Tensor[ll.int8], out: ll.Tensor[ll.int8]) -> ll.void:
    i: ll.int32 = (ll.blockIdx_x() * 32 + ll.threadIdx_x()) * 16
    {_lane_decls()}
    ll.{load}(x, i, {_lane_args()})
    ll.{store}(out, i, {_lane_args()})
"""


class TestGmemV16s8Api(unittest.TestCase):

    def test_family_is_exported(self):
        for name in ("gmem_ld_v16s8", "gmem_ld_v16s8_nc", "gmem_st_v16s8", "gmem_st_v16s8_cs"):
            self.assertTrue(hasattr(ll, name), name)
            self.assertTrue(callable(getattr(ll, name)), name)

    def test_codegen_helper_unit(self):
        """The codegen funcs render the typed 16-lane call site directly."""
        result = getattr(memory.gmem_ld_v16s8, CODEGEN_FUNC_ATTR)("x", "i", *[f"b{k}" for k in range(16)])
        self.assertIn(
            "gmem_ld_v16s8_fn(x + i, &b0, &b1, &b2, &b3, &b4, &b5, &b6, "
            "&b7, &b8, &b9, &b10, &b11, &b12, &b13, &b14, &b15)", result.return_val)
        self.assertIn("int8_t* r15", result.body)
        self.assertIn("*r15 = b[15];", result.body)
        # One 16-byte vector transaction, not scalar loads.
        self.assertIn("const uint4 value = "
                      "*reinterpret_cast<const uint4*>(ptr);", result.body)

        result = getattr(memory.gmem_st_v16s8, CODEGEN_FUNC_ATTR)("out", "i", *[f"b{k}" for k in range(16)])
        self.assertIn("gmem_st_v16s8_fn(out + i, ", result.return_val)
        self.assertIn("b15)", result.return_val)
        self.assertIn("int8_t r15", result.body)
        self.assertIn("b[15] = r15;", result.body)


class TestGmemV16s8Lowering(unittest.TestCase):

    def test_typed_int8_lanes_lower_to_one_16b_move(self):
        source = _kernel("gmem_ld_v16s8_nc", "gmem_st_v16s8_cs", "v16s8_copy")
        with tempfile.TemporaryDirectory() as d:
            cuda = _lower(Path(d), source, "v16s8_copy")
        # 16 int8-typed by-ref output lanes ...
        self.assertIn("gmem_ld_v16s8_nc_fn", cuda)
        self.assertIn("&b0", cuda)
        self.assertIn("&b15", cuda)
        self.assertIn("int8_t* r0", cuda)
        self.assertIn("int8_t* r15", cuda)
        # ... via the read-only cache, one uint4 (16-byte) transaction ...
        self.assertIn("__ldg(reinterpret_cast<const uint4*>(ptr))", cuda)
        # ... and the symmetric streaming store of 16 int8 value lanes.
        self.assertIn("gmem_st_v16s8_cs_fn", cuda)
        self.assertIn("int8_t r0", cuda)
        self.assertIn("__stcs(reinterpret_cast<uint4*>(ptr), value)", cuda)
        # By-ref lanes keep an address: never forced static constexpr.
        self.assertNotIn("static constexpr", cuda)
        # int8 locals are declared with the signed C spelling.
        self.assertIn("int8_t b0 =", cuda)

    def test_default_policy_twins(self):
        source = _kernel("gmem_ld_v16s8", "gmem_st_v16s8", "v16s8_plain")
        with tempfile.TemporaryDirectory() as d:
            cuda = _lower(Path(d), source, "v16s8_plain")
        self.assertIn("gmem_ld_v16s8_fn(x + i,", cuda)
        self.assertIn("*reinterpret_cast<const uint4*>(ptr)", cuda)
        self.assertNotIn("gmem_ld_v16s8_nc_fn", cuda)
        self.assertIn("gmem_st_v16s8_fn(out + i,", cuda)
        self.assertIn("*reinterpret_cast<uint4*>(ptr) = value", cuda)
        self.assertNotIn("__stcs", cuda)

    def test_wrong_load_lane_type_rejected(self):
        source = _kernel("gmem_ld_v16s8", "gmem_st_v16s8",
                         "v16s8_bad_load").replace("b0: ll.int8 = 0", "b0: ll.int32 = 0")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeInferenceError) as cm:
                _lower(Path(d), source, "v16s8_bad_load")
        msg = str(cm.exception)
        self.assertIn("gmem_ld_v16s8", msg)
        self.assertIn("b0", msg)
        self.assertIn("int8", msg)
        self.assertIn("int32", msg)

    def test_wrong_store_lane_type_rejected(self):
        # Load lanes (a0..a15) are all int8; the store mixes in z, a uint32
        # local, so the load-side validator cannot fire first.
        load_decls = "\n    ".join(f"a{k}: ll.int8 = 0" for k in range(16))
        load_args = ", ".join(f"a{k}" for k in range(16))
        store_args = ", ".join((f"a{k}" if k != 15 else "z") for k in range(16))
        source = f"""\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def v16s8_bad_store(x: ll.Tensor[ll.int8], out: ll.Tensor[ll.int8]) -> ll.void:
    i: ll.int32 = (ll.blockIdx_x() * 32 + ll.threadIdx_x()) * 16
    {load_decls}
    z: ll.uint32 = 0
    ll.gmem_ld_v16s8(x, i, {load_args})
    ll.gmem_st_v16s8(out, i, {store_args})
"""
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeInferenceError) as cm:
                _lower(Path(d), source, "v16s8_bad_store")
        msg = str(cm.exception)
        self.assertIn("gmem_st_v16s8", msg)
        self.assertIn("z", msg)
        self.assertIn("int8", msg)
        self.assertIn("uint32", msg)


if __name__ == "__main__":
    unittest.main()
