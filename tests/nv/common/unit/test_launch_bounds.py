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
"""Optional export.json launch.min_blocks_per_sm -> __launch_bounds__ surface.

Cycle-13 request (/exchange/requests/cycle-013-kernel-00.md, task 0000,
sampled task): an explicit typed frontend surface for the already-shipped
``__launch_bounds__(num_threads, min_blocks)`` codegen branch
(``codegen/codegen_cuda.py`` entry-kernel signature emission).  Before this
surface the ctx key ``__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__`` was
only settable by an undocumented module-level global in the draft
(``py_func.__globals__`` -> ``LLKernel.ctx``).  The contract is generic and
task-agnostic: an optional positive ``launch.min_blocks_per_sm`` integer is
validated in ``integrations/vibe_cuda_eval.py`` (against the 2048
threads/SM hardware bound) and injected into that ctx key before compile;
absent => generated CUDA is byte-identical to the previous behavior
(no ``__launch_bounds__`` attribute).

Tests are CPU-only (draft -> PASSES -> codegen -> export() renderer, no
nvcc): one generic two-operand elementwise kernel rendered with and
without the field, plus negative validation cases.  Legality of the
occupancy claim itself (regs/thread actually fitting the bound) is the
frontend author's burden, identical wording to the raw-CUDA contract.
"""

import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll
from little_kernel.backends.nv.common.integrations import vibe_cuda_eval

KERNEL_SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def scale_add(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32],
              out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    if i < n:
        out[i] = x[i] + y[i]
'''


def _spec(min_blocks=None, block=(128, 1, 1)):
    launch = {"grid": ["ceil_div(x[0], 128)", 1, 1], "block": list(block)}
    if min_blocks is not None:
        launch["min_blocks_per_sm"] = min_blocks
    return {
        "kernel": "scale_add",
        "function": "f",
        "inputs": [{"name": "x", "dtype": "float32", "rank": 1}, {"name": "y", "dtype": "float32", "rank": 1}],
        "outputs": [{"name": "out", "dtype": "float32", "shape": ["x[0]"]}],
        "scalars": [{"name": "n", "dtype": "int32", "expr": "x[0]"}],
        "kernel_args": ["x", "y", "out", "n"],
        "forward_args": ["x", "y"],
        "launch": launch,
    }


class LaunchBoundsExportTests(unittest.TestCase):

    def _export_cuda(self, spec):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(KERNEL_SOURCE)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(spec))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        return {p.name: p.read_text() for p in paths}

    def test_field_emits_launch_bounds(self):
        cuda = next(v for k, v in self._export_cuda(_spec(min_blocks=4)).items() if k.endswith(".cu"))
        # block is [128,1,1] -> num_threads 128, min_blocks 4.
        self.assertIn("__launch_bounds__(128, 4)", cuda)

    def test_field_absent_emits_no_launch_bounds(self):
        cuda = next(v for k, v in self._export_cuda(_spec()).items() if k.endswith(".cu"))
        self.assertNotIn("__launch_bounds__", cuda)

    def test_field_overrides_module_global(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text("__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ = 2\n" + KERNEL_SOURCE)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(_spec(min_blocks=8)))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        cuda = next(p.read_text() for p in paths if p.name.endswith(".cu"))
        self.assertIn("__launch_bounds__(128, 8)", cuda)

    def test_module_global_path_still_works(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text("__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ = 6\n" + KERNEL_SOURCE)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(_spec()))
        paths = vibe_cuda_eval.export(draft, spec_path, root, force=True)
        cuda = next(p.read_text() for p in paths if p.name.endswith(".cu"))
        self.assertIn("__launch_bounds__(128, 6)", cuda)

    def _assert_rejected(self, spec_update):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "task").mkdir()
        (root / "agent" / "kernels").mkdir(parents=True)
        draft = root / "agent" / "draft.py"
        draft.write_text(KERNEL_SOURCE)
        spec = _spec()
        spec["launch"].update(spec_update)
        spec_path = root / "agent" / "export.json"
        spec_path.write_text(json.dumps(spec))
        with self.assertRaises(ValueError):
            vibe_cuda_eval.export(draft, spec_path, root, force=True)

    def test_rejects_zero_and_negative(self):
        self._assert_rejected({"min_blocks_per_sm": 0})
        self._assert_rejected({"min_blocks_per_sm": -2})

    def test_rejects_bool_and_float(self):
        self._assert_rejected({"min_blocks_per_sm": True})
        self._assert_rejected({"min_blocks_per_sm": 2.5})

    def test_rejects_threads_over_hardware_bound(self):
        # block 256 * 9 = 2304 > 2048 threads/SM must fail loudly.
        self._assert_rejected({"block": [256, 1, 1], "min_blocks_per_sm": 9})

    def test_accepts_exact_hardware_bound(self):
        cuda = next(v for k, v in self._export_cuda(_spec(min_blocks=8, block=(256, 1, 1))).items()
                    if k.endswith(".cu"))
        self.assertIn("__launch_bounds__(256, 8)", cuda)


if __name__ == "__main__":
    unittest.main()
