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
"""Unit tests for MemoryAnalyzer duplicate-allocation diagnostics.

Covers the ir_memory_analysis_duplicate failure mode observed in trusted
controller data ("ll.empty call for g has already been recorded"): the
raised error must stay a ValueError whose leading text is unchanged, and
must additionally report both call sites plus the descriptor diff so the
frontend author can act on it without re-running the pipeline by hand.
"""

import ast
import unittest

import little_kernel.language as ll

from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.backends.common.gpu.passes.mem_analysis import MemoryAnalyzer
from little_kernel.core.passes.utils.add_parent_reference import add_parent_references


class _ConstDtypeTransformer(ast.NodeTransformer):
    """Resolve `ll.<dtype>` Attribute nodes into ast.Constant(LLType), the
    same form the production pipeline hands to MemoryAnalyzer."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "ll":
            value = getattr(ll, node.attr, None)
            if node.attr in ("float32", "uint32", "bfloat16", "int32"):
                return ast.copy_location(ast.Constant(value=value), node)
        return node


def _run_analyzer(source: str) -> MemoryAnalyzer:
    tree = ast.parse(source)
    tree = _ConstDtypeTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    add_parent_references(tree)

    def main():  # only its __name__ is inspected by the analyzer
        pass

    ctx = {"ll": ll, __LITTLE_KERNEL_ENTRY__: main}
    analyzer = MemoryAnalyzer(ctx)
    analyzer.visit(tree)
    return analyzer


class TestMemAnalysisDuplicateDiagnostics(unittest.TestCase):

    def test_single_declaration_records_call_site(self):
        analyzer = _run_analyzer("def main():\n"
                                 "    red = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n")
        self.assertIn("red", analyzer.variables)
        self.assertEqual(analyzer.variables["red"]["scope"], "shared")
        self.assertIn("call_site", analyzer.variables["red"])
        self.assertIn("ll.empty([32]", analyzer.variables["red"]["call_site"])

    def test_duplicate_identical_reports_both_sites(self):
        with self.assertRaises(ValueError) as cm:
            _run_analyzer("def main():\n"
                          "    red = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n"
                          "    red = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n")
        msg = str(cm.exception)
        # Leading text + exception type unchanged for downstream dashboards.
        self.assertIn("ll.empty call for red has already been recorded", msg)
        # Both call sites are shown.
        self.assertIn("first definition:", msg)
        self.assertIn("second definition:", msg)
        # Identical descriptors are called out explicitly.
        self.assertIn("identical scope/dtype/shape/bytes", msg)

    def test_duplicate_different_dtype_reports_diff(self):
        with self.assertRaises(ValueError) as cm:
            _run_analyzer("def main():\n"
                          "    red = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n"
                          "    red = ll.empty([32], dtype=ll.uint32, scope=\"shared\")\n")
        msg = str(cm.exception)
        self.assertIn("ll.empty call for red has already been recorded", msg)
        self.assertIn("dtype: first=", msg)

    def test_duplicate_different_shape_reports_diff(self):
        with self.assertRaises(ValueError) as cm:
            _run_analyzer("def main():\n"
                          "    red = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n"
                          "    red = ll.empty([64], dtype=ll.float32, scope=\"shared\")\n")
        msg = str(cm.exception)
        self.assertIn("total_elements: first=32 second=64", msg)

    def test_ll_zeros_duplicate_also_reports(self):
        with self.assertRaises(ValueError) as cm:
            _run_analyzer("def main():\n"
                          "    buf = ll.zeros([8], dtype=ll.float32, scope=\"local\")\n"
                          "    buf = ll.zeros([8], dtype=ll.float32, scope=\"local\")\n")
        self.assertIn("ll.zeros call for buf has already been recorded", str(cm.exception))

    def test_distinct_names_still_pass(self):
        analyzer = _run_analyzer("def main():\n"
                                 "    a = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n"
                                 "    b = ll.empty([16], dtype=ll.float32, scope=\"shared\")\n")
        self.assertEqual(set(analyzer.variables), {"a", "b"})
        # 32 + 16 float32 elements = 192 bytes total in the shared scope.
        self.assertEqual(analyzer.scope_offsets["shared"], 192)


if __name__ == "__main__":
    unittest.main()
