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

################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
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
"""Unit tests for the rebind_alloc_names pass.

Covers the generic capability requested by cycle-011 kernel-03 (P3) and
measured as the `ir_memory_analysis_duplicate` validity failures in frozen
controller full2 data: re-declared ll.empty/ll.zeros/ll.full buffer names
must compile with exact Python re-binding semantics whenever the binding is
statically resolvable, and must keep the old failure contract otherwise.
"""

import ast
import unittest

import little_kernel.language as ll

from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.backends.common.gpu.passes.mem_analysis import MemoryAnalyzer
from little_kernel.backends.common.gpu.passes.rebind_alloc_names import rebind_alloc_names
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


def _parse(source: str) -> ast.AST:
    tree = ast.parse(source)
    tree = _ConstDtypeTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return tree


def _run_pass(source: str) -> ast.AST:

    def main():  # only its __name__ is inspected
        pass

    ctx = {"ll": ll, __LITTLE_KERNEL_ENTRY__: main}
    tree = _parse(source)
    return rebind_alloc_names(tree, ctx)


def _run_pass_then_analyzer(source: str) -> MemoryAnalyzer:

    def main():
        pass

    ctx = {"ll": ll, __LITTLE_KERNEL_ENTRY__: main}
    tree = _parse(source)
    tree = rebind_alloc_names(tree, ctx)
    add_parent_references(tree)
    analyzer = MemoryAnalyzer(ctx)
    analyzer.visit(tree)
    return analyzer


class TestNoDuplicateBackwardCompat(unittest.TestCase):
    """Programs without duplicated names must be byte-identical after the
    pass (pure backward-compat guarantee, invariant 1)."""

    def test_single_declaration_unchanged(self):
        src = ("def main():\n"
               "    red = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n"
               "    use(red)\n")
        self.assertEqual(ast.dump(_run_pass(src)), ast.dump(_parse(src)))

    def test_distinct_names_unchanged(self):
        src = ("def main():\n"
               "    a = ll.empty([32], dtype=ll.float32, scope=\"shared\")\n"
               "    b = ll.zeros([16], dtype=ll.float32, scope=\"local\")\n"
               "    acc = a[0]\n"
               "    use(a, b, acc)\n")
        self.assertEqual(ast.dump(_run_pass(src)), ast.dump(_parse(src)))


class TestStraightLineVersioning(unittest.TestCase):

    def test_duplicate_empty_versions_and_analyzer_accepts(self):
        analyzer = _run_pass_then_analyzer("def main():\n"
                                           "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                                           "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n")
        self.assertEqual(set(analyzer.variables), {"g__lkv1", "g__lkv2"})
        # 8 float32 elements = 32 bytes per version in the local scope.
        self.assertEqual(analyzer.scope_offsets["local"], 64)
        self.assertEqual(analyzer.variables["g__lkv1"]["offset"], 0)
        self.assertEqual(analyzer.variables["g__lkv2"]["offset"], 32)

    def test_references_bind_before_and_after_rebinding(self):
        tree = _run_pass("def main():\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    first = g\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    second = g\n")
        text = ast.unparse(tree)
        # Uses before the re-declaration bind the first buffer; uses after
        # bind the second (Python re-binding semantics).
        self.assertIn("first = g__lkv1", text)
        self.assertIn("second = g__lkv2", text)

    def test_alias_escape_binds_old_version(self):
        # `y = x` binds y to the buffer object current at that point; the
        # later re-declaration of x must not retro-bind y.
        tree = _run_pass("def main():\n"
                         "    x = ll.empty([4], dtype=ll.float32, scope=\"local\")\n"
                         "    y = x\n"
                         "    x = ll.empty([4], dtype=ll.float32, scope=\"local\")\n"
                         "    use(y)\n")
        text = ast.unparse(tree)
        self.assertIn("y = x__lkv1", text)
        self.assertIn("use(y)", text)
        self.assertNotIn("y = x__lkv2", text)

    def test_mixed_apis_and_descriptors_each_get_own_buffer(self):
        analyzer = _run_pass_then_analyzer("def main():\n"
                                           "    buf = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                                           "    buf = ll.zeros([4], dtype=ll.uint32, scope=\"local\")\n"
                                           "    buf = ll.empty([16], dtype=ll.float32, scope=\"shared\")\n")
        self.assertEqual(set(analyzer.variables), {"buf__lkv1", "buf__lkv2", "buf__lkv3"})
        self.assertFalse(analyzer.variables["buf__lkv1"]["is_zeros"])
        self.assertTrue(analyzer.variables["buf__lkv2"]["is_zeros"])
        self.assertEqual(analyzer.variables["buf__lkv3"]["scope"], "shared")

    def test_version_suffix_avoids_user_name_collision(self):
        analyzer = _run_pass_then_analyzer("def main():\n"
                                           "    g__lkv1 = 5\n"
                                           "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                                           "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                                           "    use(g__lkv1)\n")
        # The pre-existing user name g__lkv1 must not be overwritten by
        # versioning; versions start at g__lkv2.
        self.assertEqual(set(analyzer.variables), {"g__lkv2", "g__lkv3"})


class TestControlFlowLegality(unittest.TestCase):

    def test_branch_redeclaration_then_use_raises_old_contract(self):
        with self.assertRaises(ValueError) as cm:
            _run_pass("def main():\n"
                      "    c = 1\n"
                      "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                      "    if c:\n"
                      "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                      "    use(g)\n")
        msg = str(cm.exception)
        self.assertIsInstance(cm.exception, ValueError)
        self.assertIn("has already been recorded", msg)
        self.assertIn("ir_memory_analysis_duplicate", msg)

    def test_dominating_redeclaration_after_branch_clears_ambiguity(self):
        tree = _run_pass("def main():\n"
                         "    c = 1\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    if c:\n"
                         "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    use(g)\n")
        text = ast.unparse(tree)
        # The use after the dominating re-declaration binds the third
        # version (the branch its own version was visible only inside).
        self.assertIn("use(g__lkv3)", text)

    def test_branch_internal_uses_bind_branch_version(self):
        tree = _run_pass("def main():\n"
                         "    c = 1\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    if c:\n"
                         "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "        keep(g)\n"
                         "    else:\n"
                         "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "        keep(g)\n")
        text = ast.unparse(tree)
        self.assertIn("keep(g__lkv2)", text)
        self.assertIn("keep(g__lkv3)", text)

    def test_loop_body_versions_with_loop_local_uses(self):
        tree = _run_pass("def main():\n"
                         "    for i in range(3):\n"
                         "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "        consume(g)\n")
        text = ast.unparse(tree)
        self.assertIn("consume(g__lkv2)", text)

    def test_use_after_loop_of_loop_rebound_name_raises(self):
        with self.assertRaises(ValueError):
            _run_pass("def main():\n"
                      "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                      "    for i in range(3):\n"
                      "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                      "    use(g)\n")

    def test_single_declaration_inside_branch_still_compiles(self):
        # One textual site inside a loop body is one allocation record
        # (unchanged behaviour; the pass is a no-op for it).
        analyzer = _run_pass_then_analyzer("def main():\n"
                                           "    for i in range(3):\n"
                                           "        g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                                           "        consume(g)\n")
        self.assertEqual(set(analyzer.variables), {"g"})


class TestNonAllocPatterns(unittest.TestCase):

    def test_subscript_keyed_declarations_untouched(self):
        # Subscript-keyed allocs (flatten_empty output) keep their previous
        # behaviour: identical duplicate keys still fail in the analyzer.
        with self.assertRaises(ValueError) as cm:
            _run_pass_then_analyzer("def main():\n"
                                    "    a = ll.empty([2], dtype=ll.float32, scope=\"local\")\n"
                                    "    a[0] = ll.empty([8], dtype=ll.float32, scope=\"shared\")\n"
                                    "    a[0] = ll.empty([8], dtype=ll.float32, scope=\"shared\")\n")
        self.assertIn("has already been recorded", str(cm.exception))

    def test_plain_rebinding_between_declarations(self):
        tree = _run_pass("def main():\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    g = 5\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    use(g)\n")
        text = ast.unparse(tree)
        # The plain re-binding keeps its textual name; the use after the
        # second declaration binds the second buffer version.
        self.assertIn("g = 5", text)
        self.assertIn("use(g__lkv2)", text)

    def test_bare_name_expressions_are_renamed(self):
        tree = _run_pass("def main():\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    g[0]\n"
                         "    g = ll.empty([8], dtype=ll.float32, scope=\"local\")\n"
                         "    g[0]\n")
        text = ast.unparse(tree)
        # Both bare expression statements resolve against the binding in
        # scope at that point (regresses: rewriter results must be
        # assigned back on the generic statement path).
        self.assertIn("g__lkv1[0]", text)
        self.assertIn("g__lkv2[0]", text)

    def test_shape_expressions_rename_tracked_names(self):
        tree = _run_pass("def main():\n"
                         "    n = ll.empty([8], dtype=ll.int32, scope=\"local\")\n"
                         "    n = ll.empty([8], dtype=ll.int32, scope=\"local\")\n"
                         "    m = ll.empty([n[0]], dtype=ll.int32, scope=\"local\")\n")
        text = ast.unparse(tree)
        # The otherwise-unique buffer `m` is not versioned, but its shape
        # expression references the *current* version of duplicated `n`.
        self.assertIn("m = ll.empty([n__lkv2[0]]", text)


class TestFullPipelineIntegration(unittest.TestCase):
    """End-to-end: a duplicated local-buffer declaration must lower through
    the whole backend pass pipeline to two independent CUDA arrays."""

    def test_duplicate_local_buffers_codegen_two_arrays(self):
        import little_kernel as lk
        from little_kernel.core.passes.pass_config import get_passes_for_backend
        from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def dup_buffers(x: ll.Tensor[ll.float32], out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
            g = ll.empty([8], dtype=ll.float32, scope="local")
            t: ll.int32 = ll.threadIdx_x()
            for i in range(8):
                g[i] = x[t * 8 + i] * 2.0
            acc: ll.float32 = 0.0
            for i in range(8):
                acc = acc + g[i]
            g = ll.empty([8], dtype=ll.float32, scope="local")
            for i in range(8):
                g[i] = acc + x[t * 8 + i]
            out[t] = g[7] + g[0]

        code = dup_buffers.compile(get_passes_for_backend("cuda"), codegen_cuda, need_header=True, num_threads=128)
        # Two independent arrays, references resolved per binding.
        self.assertIn("float g__lkv1[8];", code)
        self.assertIn("float g__lkv2[8];", code)
        self.assertIn("= x[t * 8 + i] * 2.0", code)
        self.assertNotIn("float g[8];", code)
        # GPU-run numeric check (nvcc + launch, fixed input, exact result)
        # was executed on H800 by the cycle-013 compiler author
        # (/exchange/releases/cycle-013.md): PASS.


if __name__ == "__main__":
    unittest.main()
