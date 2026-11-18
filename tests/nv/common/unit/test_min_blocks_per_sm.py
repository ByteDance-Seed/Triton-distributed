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
"""Vibe manifest launch.min_blocks_per_sm -> __launch_bounds__ emission.

Compatibility-repair coverage (integrations/vibe_cuda_eval.py::_render_cuda):

- The validated manifest field launch.min_blocks_per_sm is bridged to the
  single ctx key codegen_cuda consumes
  (__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__; codegen/codegen_cuda.py
  MIN_BLOCKS_CTX_KEY), which is emitted as
  __launch_bounds__(num_threads, min_blocks) on the entry kernel only when
  the key is present. Before the repair the manifest field was validated
  and stored but never consumed, so frontend occupancy requests were
  silently dropped.
- Invariant (fallback): when the field is absent the validator stores None,
  no ctx key is written, and the generated kernel is byte-identical to a
  plain __global__ entry. The legacy draft.py module global keeps working
  for existing programs because it lands in the same ctx (LLKernel copies
  py_func.__globals__). Both request channels and the
  ll_kernel(min_blocks=...) decorator API converge on one consumer.
- Legality: the validator rejects non-positive/non-integer values, so
  invalid requests fail before any CUDA is generated.

Expected codegen: __launch_bounds__(<block product>, <min_blocks>) inside
the entry-kernel signature prefix. All assertions are golden-text (no GPU,
no nvcc).
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower

DRAFT = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_scale(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
               n: ll.int32) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    if tid < n:
        out[tid] = a[tid] * 2.0
'''

DRAFT_LEGACY_GLOBAL = '''\
import little_kernel as lk
import little_kernel.language as ll

__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ = 2

@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_scale(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32],
               n: ll.int32) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    if tid < n:
        out[tid] = a[tid] * 2.0
'''


def _spec(block_x=128, min_blocks=None):
    launch = {"grid": [1, 1, 1], "block": [block_x, 1, 1]}
    if min_blocks is not None:
        launch["min_blocks_per_sm"] = min_blocks
    return {
        "kernel": "copy_scale",
        "function": "copy_scale",
        "inputs": [{"name": "a", "dtype": "float32", "rank": 1}],
        "output": {"name": "out", "dtype": "float32", "shape": ["a[0]"]},
        "equalities": [],
        "scalars": [{"name": "n", "dtype": "int32", "expr": "a[0]"}],
        "launch": launch,
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


def _kernel_cu(out: Path) -> str:
    return next(out.glob("kernels/*.cu")).read_text()


class MinBlocksPerSmTests(unittest.TestCase):

    def test_manifest_field_emits_launch_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT, _spec(min_blocks=2)))
            self.assertIn("__launch_bounds__(128, 2)", cu)

    def test_absent_field_keeps_plain_global(self):
        # Fallback invariant: no field -> no ctx key -> plain entry kernel.
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT, _spec()))
            self.assertNotIn("__launch_bounds__", cu)

    def test_legacy_module_global_still_works(self):
        # The pre-existing request channel shares the same consumer ctx key.
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT_LEGACY_GLOBAL, _spec()))
            self.assertIn("__launch_bounds__(128, 2)", cu)

    def test_perturbed_block_and_value(self):
        # A different block size and min_blocks value track independently.
        with tempfile.TemporaryDirectory() as d:
            cu = _kernel_cu(_lower(Path(d), DRAFT, _spec(block_x=64, min_blocks=3)))
            self.assertIn("__launch_bounds__(64, 3)", cu)

    def test_nonpositive_value_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                _lower(Path(d), DRAFT, _spec(min_blocks=0))

    def test_noninteger_value_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(Exception):
                _lower(Path(d), DRAFT, _spec(min_blocks="2"))


if __name__ == "__main__":
    unittest.main()
