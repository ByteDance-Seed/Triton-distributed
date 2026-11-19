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
"""Runtime-sized (shape-expression) dynamic_shared allocations.

Legality contract for `ll.empty([...," expr "...], scope="dynamic_shared",
max_elems=N)`:
  * at most one runtime-sized dynamic_shared allocation, and it must be the
    last dynamic_shared allocation (later offsets unknown at compile time);
  * the static shared floor does not advance for it;
  * the launcher host-evaluates `elems * elem_size`, rejects values outside
    [0, max_elems*elem_size], and rejects a reservation below
    static_floor + dyn_bytes;
  * `shared_memory_bytes="auto"` is a compile-time error;
  * compile is shape-independent: the same program lowered twice produces
    byte-identical kernel bodies.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

DRAFT_A = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def stage_scale(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
                n: ll.int32) -> ll.void:
    ll.align_memory(16, "dynamic_shared")
    buf = ll.empty(["a[0]"], dtype=ll.float32, scope="dynamic_shared",
                   max_elems=4096)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < n:
        buf[tid] = a[tid]
    ll.__syncthreads()
    if tid < n:
        out[tid] = buf[tid] * 2.0
'''

DRAFT_B = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def row_stage(y: ll.Tensor[ll.float32], out: ll.Tensor[ll.bfloat16],
              s: ll.int32) -> ll.void:
    ll.align_memory(16, "dynamic_shared")
    buf = ll.empty(["2 * (((y[3] + 15) // 16) * 16 + 16)"], dtype=ll.bfloat16,
                   scope="dynamic_shared", max_elems=29056)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < s:
        out[tid] = buf[tid]
'''

SPEC_A = {
    "kernel": "stage_scale",
    "function": "stage_scale",
    "inputs": [{"name": "a", "dtype": "float32", "rank": 1}],
    "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
    "equalities": [],
    "scalars": [{"name": "n", "dtype": "int32", "expr": "a[0]"}],
    "launch": {"grid": [1, 1, 1], "block": [256, 1, 1], "shared_memory_bytes": "4 * a[0]"},
}

SPEC_B = {
    "kernel": "row_stage",
    "function": "row_stage",
    "inputs": [{"name": "y", "dtype": "float32", "rank": 4}],
    "output": {"name": "out", "dtype": "bfloat16", "shape": ["y[0]"]},
    "equalities": [],
    "scalars": [{"name": "s", "dtype": "int32", "expr": "y[3]"}],
    "launch": {
        "grid": ["y[0] * y[1] * y[2]", 1, 1],
        "block": ["min(512, 256 * ((y[3] + 8191) // 8192))", 1, 1],
        "block_max_threads": 512,
        "shared_memory_bytes": "min(58112, 4 * (((y[3] + 15) // 16) * 16 + 16))",
    },
}


def _lower(root: Path, source: str, spec: dict) -> Path:
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    (program / "export.json").write_text(json.dumps(spec))
    out = root / "output"
    lower(task, program, out, 0)
    return out


def _binding(out: Path) -> str:
    return next(out.glob("kernels/*_binding.cpp")).read_text()


def _kernel_cu(out: Path) -> str:
    return next(out.glob("kernels/*.cu")).read_text()


class RuntimeSizedDynSharedTests(unittest.TestCase):

    def test_host_evaluated_bytes_bound_and_floor(self):
        with tempfile.TemporaryDirectory() as d:
            out = _lower(Path(d), DRAFT_A, SPEC_A)
            binding = _binding(out)
            # host-evaluated bytes, max_elems legality bound (4096 f32),
            # and reservation >= dyn bytes (static floor is 0 here)
            self.assertIn("lk_shdyn_0 = (a.size(0) * 4)", binding)
            self.assertIn("lk_shdyn_0 >= 0 && lk_shdyn_0 <= 16384", binding)
            self.assertIn("lk_shared >= lk_shdyn_0", binding)
            cu = _kernel_cu(out)
            self.assertIn("extern __shared__", cu)
            # slice sits at the compile-time static offset 0
            self.assertIn("__INTERNAL_DYN_SHMEM__ + 0", cu)

    def test_reservation_not_baked_at_compile_time(self):
        # The launch reservation and the dyn floor stay expressions, so one
        # program covers every S; two lowers are byte-identical.
        with tempfile.TemporaryDirectory() as d:
            out1 = _lower(Path(d), DRAFT_B, SPEC_B)
            binding1 = _binding(out1)
            cu1 = _kernel_cu(out1)
            self.assertIn("lk_min(58112, (4 *", binding1)
            self.assertIn("lk_shdyn_0 <= 58112", binding1)
            self.assertNotIn("58112u", cu1)
        with tempfile.TemporaryDirectory() as d:
            out2 = _lower(Path(d), DRAFT_B, SPEC_B)
            self.assertEqual(cu1, _kernel_cu(out2))
            self.assertEqual(binding1, _binding(out2))

    def test_constant_reservation_still_checked(self):
        spec = {**SPEC_A, "launch": {**SPEC_A["launch"], "shared_memory_bytes": 20000}}
        with tempfile.TemporaryDirectory() as d:
            binding = _binding(_lower(Path(d), DRAFT_A, spec))
            self.assertIn("lk_shdyn_0 <= 20000", binding)

    def test_auto_shared_memory_rejected(self):
        spec = {**SPEC_A, "launch": {**SPEC_A["launch"], "shared_memory_bytes": "auto"}}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "auto"):
                _lower(Path(d), DRAFT_A, spec)

    def test_expr_dim_requires_max_elems(self):
        src = DRAFT_A.replace(",\n                   max_elems=4096)", ")")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "max_elems"):
                _lower(Path(d), src, SPEC_A)

    def test_max_elems_requires_expr_dim(self):
        src = DRAFT_A.replace('ll.empty(["a[0]"]', 'll.empty([4096]')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "shape-expression"):
                _lower(Path(d), src, SPEC_A)

    def test_expr_dim_rejected_outside_dynamic_shared(self):
        src = DRAFT_A.replace('scope="dynamic_shared"', 'scope="shared"')
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "dynamic_shared"):
                _lower(Path(d), src, SPEC_A)

    def test_static_alloc_before_runtime_sized_offsets(self):
        src = DRAFT_A.replace(
            "    buf = ll.empty(", '    head = ll.empty([64], dtype=ll.float32,'
            ' scope="dynamic_shared")\n'
            "    buf = ll.empty(")
        with tempfile.TemporaryDirectory() as d:
            out = _lower(Path(d), src, SPEC_A)
            cu = _kernel_cu(out)
            # static buffer at offset 0; runtime-sized buffer starts after
            # the static region (64 f32 = 256 bytes)
            self.assertIn("__INTERNAL_DYN_SHMEM__ + 0", cu)
            self.assertIn("__INTERNAL_DYN_SHMEM__ + 256", cu)
            binding = _binding(out)
            # floor check covers the static 256 bytes AND the dyn bytes
            self.assertIn("lk_shared >= 256 + lk_shdyn_0", binding)

    def test_second_runtime_sized_alloc_rejected(self):
        src = DRAFT_A.replace(
            "    tid: ll.int32 = ll.threadIdx_x()\n", '    buf2 = ll.empty(["a[0]"], dtype=ll.float32,'
            ' scope="dynamic_shared", max_elems=4096)\n'
            "    tid: ll.int32 = ll.threadIdx_x()\n")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "only one dynamic_shared"):
                _lower(Path(d), src, SPEC_A)

    def test_static_alloc_after_runtime_sized_rejected(self):
        src = DRAFT_A.replace(
            "    tid: ll.int32 = ll.threadIdx_x()\n", '    buf2 = ll.empty([16], dtype=ll.float32,'
            ' scope="dynamic_shared")\n'
            "    tid: ll.int32 = ll.threadIdx_x()\n")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, "may not follow"):
                _lower(Path(d), src, SPEC_A)


if __name__ == "__main__":
    unittest.main()
