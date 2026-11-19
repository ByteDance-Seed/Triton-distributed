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
"""prefix_sum1d transform helper bounds-guard regression (no GPU required).

The emitted agent_lk_prefix_sum1d_kernel is launched with
dim3((rows + 128) / 128) blocks of 128 threads, so whenever rows % 128 != 0
a thread with index i == rows exists. A guard of `if (i > rows) return;`
lets that thread through, and it then writes dst[rows] — one element past
the rows-element output — while reading src[rows - 1]. The guard must be
`if (i >= rows) return;` so only indices 0 .. rows - 1 touch the buffers.

These tests render the CUDA translation unit through the real export path
(_load_spec + _load_kernel + _render_cuda) and assert on the emitted
helper source and its launch configuration.
"""

import json
import tempfile
import unittest
from pathlib import Path

from little_kernel.backends.nv.common.integrations.vibe_cuda_eval import _load_kernel, _load_spec, _render_cuda

SOURCE = '''\
import little_kernel as lk
import little_kernel.language as ll


@lk.ll_kernel(backend="cuda", is_entry=True)
def probe(counts: ll.Tensor[ll.int32], offsets: ll.Tensor[ll.int32],
          n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        offsets[i] = counts[i]
'''


def _spec():
    return {
        "kernel": "probe",
        "function": "probe",
        "inputs": [{"name": "counts", "dtype": "int32", "rank": 1}],
        "output": {"name": "offsets", "dtype": "int32", "shape": ["counts[0]"]},
        "transforms": [
            {"op": "prefix_sum1d", "name": "count_ps", "source": "counts"},
        ],
        "scalars": [{"name": "n", "expr": "counts[0]"}],
        "launch": {"grid": ["ceil_div(n, 256)", 1, 1], "block": [256, 1, 1]},
    }


class PrefixSum1DBoundsTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec_path = root / "export.json"
            spec_path.write_text(json.dumps(_spec()))
            source = root / "kernel.py"
            source.write_text(SOURCE)
            spec = _load_spec(spec_path)
            kernel = _load_kernel(source, "probe")
            cls.cuda = _render_cuda(spec, kernel)

    def test_helper_guard_excludes_index_rows(self):
        # The exact bounds check: i == rows must return before dst[i].
        self.assertIn("if (i >= rows) return;", self.cuda)
        self.assertNotIn("if (i > rows) return;", self.cuda)

    def test_helper_writes_only_guarded_indices(self):
        helper_start = self.cuda.index("agent_lk_prefix_sum1d_kernel(")
        helper_end = self.cuda.index("}", self.cuda.index("dst[i] = total;", helper_start))
        helper = self.cuda[helper_start:helper_end]
        # Guard precedes both the src[j] accumulation loop and dst[i] store.
        guard_pos = helper.index("if (i >= rows) return;")
        self.assertLess(guard_pos, helper.index("total += src[j];"))
        self.assertLess(guard_pos, helper.index("dst[i] = total;"))

    def test_host_wrapper_launch_can_produce_index_rows(self):
        # The launch rounds up to whole 128-thread blocks, so a thread with
        # i == rows exists exactly when rows % 128 != 0; the device guard
        # above is the only thing protecting dst[rows].
        self.assertIn("dim3((rows + 128) / 128)", self.cuda)
        self.assertIn("dim3(128)", self.cuda)


if __name__ == "__main__":
    unittest.main()
