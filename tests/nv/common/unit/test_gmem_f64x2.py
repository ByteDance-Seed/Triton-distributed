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
"""gmem_ld_f64x2 / gmem_st_f64x2: dtype-twin 128-bit f64 vector GMEM access.

Cycle-11 request (/exchange/requests/cycle-011-kernel-02.md, task 0000 f64
cases): for 64-bit dtypes there was no typed way to reach LDG.E.128 /
STG.E.128 - the generic byte-lane family ``gmem_ld_v4`` delivers four
uint32 bit lanes that must be re-assembled with shifts before
``u64_as_f64`` can make them arithmetic values.  These builtins are the
exact dtype-twin of ``gmem_ld_f32x4`` / ``gmem_st_f32x4`` for 64-bit
elements: one 16-byte ``double2`` move, two lanes delivered/accepted as
typed float64 registers.

Legality contract (identical wording to the f32x4 / v4_u32 families): the
byte address of ``mem + index`` must be 16-byte aligned (equivalently,
``index`` even relative to the f64 tensor base) and both elements
in-bounds; ragged tails and unaligned offsets go through a separate scalar
path.  Violating alignment is a fault (misaligned address), the same
precondition a CUDA ``double2`` dereference has.  The API mirrors the
address-alignment precondition of PTX ``ld/st.global.v2.f64``; it carries
no task, shape, or launch-legality assumption, so it is generic for any
f64 (or bitcast int64/uint64) streaming kernel.

These tests drive the real draft -> PASSES -> codegen -> export() renderer
(no nvcc) and assert source snapshots.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations import vibe_cuda_eval

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def f64x2copy(x: ll.Tensor[ll.float64], y: ll.Tensor[ll.float64],
              n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    base: ll.int32 = i * 2
    if base + 1 < n:
        vec = ll.empty([2], dtype=ll.float64, scope="local")
        ll.gmem_ld_f64x2(x, base, vec[0], vec[1])
        ll.gmem_st_f64x2(y, base, vec[0], vec[1])
'''

SPEC = {
    "kernel": "f64x2copy",
    "function": "f",
    "inputs": [{"name": "x", "dtype": "float64", "rank": 1}],
    "outputs": [{"name": "y", "dtype": "float64", "shape": ["x[0]"]}],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
    "kernel_args": ["x", "y", "n"],
    "forward_args": ["x"],
    "launch": {"grid": ["ceil_div(x[0], 256)", 1, 1], "block": [128, 1, 1]},
}


class GmemF64x2Tests(unittest.TestCase):

    def _export_cuda(self, source=KERNEL_SOURCE):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(source)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(SPEC))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        return {p.name: p.read_text() for p in paths}

    def test_load_helper_emits_one_double2_move(self):
        cuda = next(v for k, v in self._export_cuda().items() if k.endswith(".cu"))
        self.assertIn("__device__ __forceinline__ void gmem_ld_f64x2_fn(", cuda)
        self.assertIn("const double2 value = *reinterpret_cast<const double2*>(ptr);", cuda)
        # Lane order matches memory order: r0 <- ptr[0], r1 <- ptr[1].
        self.assertIn("*r0 = value.x; *r1 = value.y;", cuda)

    def test_store_helper_emits_one_double2_move(self):
        cuda = next(v for k, v in self._export_cuda().items() if k.endswith(".cu"))
        self.assertIn("__device__ __forceinline__ void gmem_st_f64x2_fn(", cuda)
        self.assertIn("*reinterpret_cast<double2*>(ptr) = make_double2(v0, v1);", cuda)

    def test_call_sites_use_element_index_and_register_refs(self):
        cuda = next(v for k, v in self._export_cuda().items() if k.endswith(".cu"))
        self.assertIn("gmem_ld_f64x2_fn(x + base, &", cuda)
        self.assertIn("gmem_st_f64x2_fn(y + base, ", cuda)

    def test_binding_declares_float64_guards(self):
        files = self._export_cuda()
        cuda = next(v for k, v in files.items() if k.endswith("_binding.cpp"))
        # The surrounding ABI must keep f64 storage typed end to end.
        self.assertIn("at::kDouble", cuda)


if __name__ == "__main__":
    unittest.main()
