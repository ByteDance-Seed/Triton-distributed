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
"""Static versus dynamic shared-memory resource legality.

Regression for ir.shared_memory.static_resource_legality: independent
scope="shared" allocations are summed against ptxas's 48 KiB static .shared
cap and diagnosed at compile time on CPU. A launch dynamic-shared byte
reservation (scope="dynamic_shared") is a separate resource: it is never
subtracted from that sum and cannot rewrite independent static allocations,
but it may itself exceed 48 KiB through the launch-time opt-in carveout.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

import little_kernel.language as ll

from little_kernel.backends.nv.common.integrations.ai_compiler_eval import lower
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.backends.common.gpu.passes.mem_analysis import STATIC_SHARED_MEMORY_LIMIT_BYTES, MemoryAnalyzer
from little_kernel.core.passes.utils.add_parent_reference import add_parent_references


class _ConstDtypeTransformer(ast.NodeTransformer):
    """Resolve `ll.<dtype>` into ast.Constant(LLType), as the pipeline does."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "ll":
            value = getattr(ll, node.attr, None)
            if node.attr in ("float32", "bfloat16", "int32"):
                return ast.copy_location(ast.Constant(value=value), node)
        return node


def _run_analyzer(body: str) -> MemoryAnalyzer:
    tree = ast.parse("def main():\n" + body)
    tree = _ConstDtypeTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    add_parent_references(tree)

    def main():
        pass

    ctx = {"ll": ll, __LITTLE_KERNEL_ENTRY__: main}
    analyzer = MemoryAnalyzer(ctx)
    analyzer.visit(tree)
    return analyzer


class StaticSharedLegalityAnalyzerTests(unittest.TestCase):

    def test_independent_static_allocations_are_summed(self):
        analyzer = _run_analyzer('    sA = ll.empty([1056], dtype=ll.float32, scope="shared")\n'
                                 '    sB = ll.empty([1056], dtype=ll.float32, scope="shared")\n'
                                 '    sE = ll.empty([20480], dtype=ll.float32, scope="shared")\n')
        report = analyzer.static_shared_memory_report()
        # 2 * 1056 * 4 + 20480 * 4 == 0x16100, the exact ptxas figure of the
        # reported failure (two 8x132 mainloop stages + 128x160 epilogue).
        self.assertEqual(report.static_bytes, 0x16100)
        self.assertEqual([name for name, _, _ in report.buffers], ["sA", "sB", "sE"])
        self.assertEqual([off for _, off, _ in report.buffers], [0, 1056 * 4, 2 * 1056 * 4])
        self.assertFalse(report.legal)

    def test_exact_limit_is_legal(self):
        analyzer = _run_analyzer('    sA = ll.empty([8192], dtype=ll.float32, scope="shared")\n'
                                 '    sB = ll.empty([4096], dtype=ll.float32, scope="shared")\n')
        report = analyzer.audit_static_shared_memory()
        self.assertEqual(report.static_bytes, STATIC_SHARED_MEMORY_LIMIT_BYTES)
        self.assertTrue(report.legal)

    def test_over_limit_raises_actionable_diagnostic(self):
        analyzer = _run_analyzer('    sA = ll.empty([8192], dtype=ll.float32, scope="shared")\n'
                                 '    sB = ll.empty([4097], dtype=ll.float32, scope="shared")\n')
        with self.assertRaises(ValueError) as cm:
            analyzer.audit_static_shared_memory()
        msg = str(cm.exception)
        self.assertIn("ir_shared_memory_static_resource_legality", msg)
        self.assertIn("0xc000", msg)
        # Every independent buffer is itemized with bytes and offset.
        self.assertIn("sA: 32768 bytes at offset 0", msg)
        self.assertIn("sB: 16388 bytes at offset 32768", msg)
        # The launch dynamic reservation must not be mistaken for a rewrite.
        self.assertIn("summed by the compiler", msg)
        self.assertIn("dynamic", msg)
        self.assertIn("scope=\"dynamic_shared\"", msg)

    def test_dynamic_shared_floor_does_not_count_toward_static_sum(self):
        # 64 KiB dynamic_shared (legal via launch opt-in) beside a small
        # static buffer: static sum stays small, no diagnostic.
        analyzer = _run_analyzer('    ll.align_memory(16, "dynamic_shared")\n'
                                 '    dA = ll.empty([16384], dtype=ll.float32,'
                                 ' scope="dynamic_shared")\n'
                                 '    sA = ll.empty([256], dtype=ll.float32, scope="shared")\n')
        report = analyzer.audit_static_shared_memory()
        self.assertEqual(report.static_bytes, 1024)
        self.assertEqual(report.dynamic_floor_bytes, 65536)
        self.assertTrue(report.legal)


DRAFT_OVER = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def k(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    sA = ll.empty([8192], dtype=ll.float32, scope="shared")
    sB = ll.empty([4097], dtype=ll.float32, scope="shared")
    tid: ll.int32 = ll.threadIdx_x()
    sA[tid] = a[tid]
    sB[tid] = a[tid + 1]
    ll.__syncthreads()
    out[tid] = sA[tid] + sB[tid]
'''

DRAFT_DYN = '''\
import little_kernel as lk
import little_kernel.language as ll

@lk.ll_kernel(backend="cuda", is_entry=True)
def k(a: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32]) -> ll.void:
    ll.align_memory(16, "dynamic_shared")
    sA = ll.empty([16384], dtype=ll.float32, scope="dynamic_shared")
    tid: ll.int32 = ll.threadIdx_x()
    sA[tid] = a[tid]
    ll.__syncthreads()
    out[tid] = sA[tid]
'''


def _lower(root: Path, source: str, shared_bytes: int) -> str:
    task = root / "task"
    task.mkdir()
    (task / "model.py").write_text("class Model: pass\n")
    (task / "test_cases.py").write_text("test_cases = []\n")
    program = root / "program"
    program.mkdir()
    (program / "draft.py").write_text(source)
    spec = {
        "kernel": "k",
        "function": "k",
        "inputs": [{"name": "a", "dtype": "float32", "rank": 1}],
        "outputs": [{"name": "out", "dtype": "float32", "shape": ["a[0]"]}],
        "launch": {"grid": [1, 1, 1], "block": [256, 1, 1], "shared_memory_bytes": shared_bytes},
    }
    (program / "export.json").write_text(json.dumps(spec))
    out = root / "output"
    lower(task, program, out, 0)
    return next(out.glob("kernels/*.cu")).read_text()


class StaticSharedLegalityLowerTests(unittest.TestCase):

    def test_over_limit_rejected_during_lower(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as cm:
                _lower(Path(d), DRAFT_OVER, 0)
            self.assertIn("ir_shared_memory_static_resource_legality", str(cm.exception))

    def test_large_dynamic_shared_still_lowers(self):
        with tempfile.TemporaryDirectory() as d:
            cu = _lower(Path(d), DRAFT_DYN, 65536)
            self.assertIn("extern __shared__", cu)


if __name__ == "__main__":
    unittest.main()
