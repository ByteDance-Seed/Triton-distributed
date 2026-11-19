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
"""CPU regression for inline-callsite provenance diagnostics.

When one outlined helper (e.g. an ``@lk.ll_kernel(is_entry=False)`` device
subroutine, or any ``@inline_func`` helper) is inlined at multiple
callsites, every downstream pass sees N copies of the same body.  Any
diagnostic raised inside one of those copies must name *which* copy it
came from; otherwise users have to comment out callsites one at a time
(divide and conquer) to find the culprit -- the exact failure mode of the
pinned reproducer (two inlined callsites of ``_leaf`` where the
merge-predicate diagnostic referenced the erased device-only subtree
instead of the failing copy).

These tests cover the generic mechanism without a GPU:

* ``FunctionInliner`` stamps an :class:`InlineProvenance` on every node of
  every inlined expansion, with a distinct ``copy_index`` per callsite and
  the callee name recorded.
* The stamp survives node rewrites that go through
  ``preserve_attributes`` (the attribute is registered in
  ``AST_SPECIAL_ATTRS``).
* ``format_error_message`` / ``PassError`` route through the provenance
  map so the final diagnostic reads e.g. "in inlined copy 2 of _leaf"
  when the offending node belongs to the second inlined expansion, and
  stays silent (no provenance line) for nodes that were never inlined.
* CPython-interned leaf singletons (``Load``/``Store``/operators) are
  never stamped, so provenance cannot leak into unrelated parse trees.

Run:  uv run --locked --no-sync python -m unittest \
      tests.nv.common.unit.test_inline_provenance -v
"""

import unittest

from little_kernel.core.passes.inline import inline as run_inline_pass
from little_kernel.core.passes.utils.error_report import (
    PassError,
    format_error_message,
)
from little_kernel.core.passes.utils.inline_provenance import (
    INLINE_PROVENANCE_ATTR,
    find_inline_provenance,
    get_inline_provenance,
)
from little_kernel.core.passes.utils.preserve_attributes import (
    create_node_with_attrs, )
from little_kernel.language.builtin_base import (
    AST_SPECIAL_ATTRS,
    inline_func,
)


def _run_inliner(kernel_source: str, ctx_funcs: dict):
    """Parse ``kernel_source`` and run the real FunctionInliner over it."""
    import ast

    tree = ast.parse(kernel_source)
    return run_inline_pass(tree, dict(ctx_funcs))


class TestInlineProvenanceStamping(unittest.TestCase):
    """The inliner must tag every inlined copy with its callsite identity."""

    def _make_leaf(self):

        @inline_func
        def _leaf(x):
            y = x + 1
            return y

        return _leaf

    def _two_callsite_source(self):
        # Mirrors the pinned reproducer shape: two callsites of one
        # outlined helper inside one entry body.
        return ("def entry(a, b):\n"
                "    r0 = _leaf(a)\n"
                "    r1 = _leaf(b)\n"
                "    return r0 + r1\n")

    def test_two_callsites_get_distinct_copy_indices(self):
        import ast

        _leaf = self._make_leaf()
        new_tree = _run_inliner(self._two_callsite_source(), {"_leaf": _leaf})

        # Collect every provenance tag present on the inlined tree.
        provs = []
        for node in ast.walk(new_tree):
            prov = get_inline_provenance(node)
            if prov is not None and all(p is not prov for p in provs):
                provs.append(prov)

        self.assertEqual(len(provs), 2, f"expected exactly 2 distinct inlined copies, got {provs}")
        copy_indices = sorted(p.copy_index for p in provs)
        self.assertEqual(copy_indices, [1, 2])
        for prov in provs:
            self.assertIn("_leaf", prov.function_name)
            # Callsite line info must be captured for diagnostics.
            self.assertIsNotNone(prov.callsite_lineno)

    def test_every_node_of_each_copy_is_tagged(self):
        import ast

        _leaf = self._make_leaf()
        new_tree = _run_inliner(self._two_callsite_source(), {"_leaf": _leaf})

        # The renamed locals of each copy (_leaf_1_y / _leaf_2_y per the
        # inliner's suffix scheme) must carry their own copy's tag on both
        # the Store target and the Load use.
        seen = {}
        for node in ast.walk(new_tree):
            if isinstance(node, ast.Name) and "_leaf_" in node.id:
                prov = get_inline_provenance(node)
                self.assertIsNotNone(prov, f"inlined local {node.id!r} lost its provenance tag")
                seen.setdefault(node.id, set()).add(prov.copy_index)

        self.assertTrue(any("1" in name or "_leaf_1_" in name for name in seen),
                        f"first copy locals missing: {sorted(seen)}")
        self.assertTrue(any("_leaf_2_" in name for name in seen), f"second copy locals missing: {sorted(seen)}")
        for name, indices in seen.items():
            self.assertEqual(len(indices), 1, f"local {name!r} carries tags from multiple copies: {indices}")

    def test_attr_registered_in_ast_special_attrs(self):
        self.assertIn(INLINE_PROVENANCE_ATTR, AST_SPECIAL_ATTRS)

    def test_interned_singleton_leaf_kinds_are_never_stamped(self):
        """CPython interns Load/Store/operator singletons across ast.parse
        calls; stamping one would leak provenance into unrelated trees.
        After running the inliner, no expr_context/operator node of the
        inlined tree may carry the attribute, and a freshly parsed tree
        must be completely clean."""
        import ast as ast_mod

        _leaf = self._make_leaf()
        new_tree = _run_inliner(self._two_callsite_source(), {"_leaf": _leaf})

        singleton_kinds = (ast_mod.expr_context, ast_mod.operator, ast_mod.unaryop, ast_mod.cmpop, ast_mod.boolop)
        saw_singleton = False
        for node in ast_mod.walk(new_tree):
            if isinstance(node, singleton_kinds):
                saw_singleton = True
                self.assertFalse(
                    hasattr(node, INLINE_PROVENANCE_ATTR), f"interned {type(node).__name__} singleton was stamped; "
                    "provenance would leak across parse trees")
        self.assertTrue(saw_singleton, "inlined tree unexpectedly had no context/operator "
                        "leaf nodes; the guard above is vacuous")

        # A tree parsed after inlining must not observe any leaked tag,
        # regardless of object identity shared with the inlined tree.
        fresh = ast_mod.parse("def other(a):\n    return a + 1\n")
        for node in ast_mod.walk(fresh):
            self.assertFalse(hasattr(node, INLINE_PROVENANCE_ATTR),
                             f"provenance leaked onto {type(node).__name__} of an "
                             "independently parsed tree")

    def test_provenance_survives_node_rewrite(self):
        """create_node_with_attrs (used by passes when rebuilding nodes)
        must carry the provenance over to the replacement node."""
        import ast

        _leaf = self._make_leaf()
        new_tree = _run_inliner(self._two_callsite_source(), {"_leaf": _leaf})
        tagged = None
        for node in ast.walk(new_tree):
            if get_inline_provenance(node) is not None:
                tagged = node
                break
        self.assertIsNotNone(tagged)

        rebuilt = create_node_with_attrs(ast.Assign, tagged, targets=[], value=None)
        self.assertIs(get_inline_provenance(rebuilt), get_inline_provenance(tagged))


class TestInlineProvenanceDiagnostics(unittest.TestCase):
    """Error reporting must name the inlined copy the failure came from."""

    def _make_leaf(self):

        @inline_func
        def _leaf(x):
            y = x + 1
            return y

        return _leaf

    def _nodes_by_copy(self, new_tree):
        import ast

        by_copy = {}
        for node in ast.walk(new_tree):
            prov = get_inline_provenance(node)
            if prov is not None:
                by_copy.setdefault(prov.copy_index, []).append(node)
        return by_copy

    def test_error_names_second_copy(self):
        source = ("def entry(a, b):\n"
                  "    r0 = _leaf(a)\n"
                  "    r1 = _leaf(b)\n"
                  "    return r0 + r1\n")
        new_tree = _run_inliner(source, {"_leaf": self._make_leaf()})
        by_copy = self._nodes_by_copy(new_tree)
        self.assertEqual(sorted(by_copy), [1, 2])

        # Simulate a downstream pass (e.g. merge-predicate) raising
        # against a node deep inside the SECOND inlined expansion.
        failing_node = by_copy[2][-1]
        msg = format_error_message("Internal error: getDefinitionNodeId called on device-only subtree", failing_node)
        self.assertIn("copy 2 of _leaf", msg)
        self.assertNotIn("copy 1 of _leaf", msg)

    def test_error_names_first_copy(self):
        source = ("def entry(a, b):\n"
                  "    r0 = _leaf(a)\n"
                  "    r1 = _leaf(b)\n"
                  "    return r0 + r1\n")
        new_tree = _run_inliner(source, {"_leaf": self._make_leaf()})
        by_copy = self._nodes_by_copy(new_tree)

        failing_node = by_copy[1][-1]
        with self.assertRaises(PassError) as cm:
            raise PassError("Internal error: getDefinitionNodeId called on device-only subtree", failing_node)
        self.assertIn("copy 1 of _leaf", str(cm.exception))
        self.assertNotIn("copy 2 of _leaf", str(cm.exception))

    def test_error_on_untagged_node_has_no_provenance_line(self):
        import ast

        tree = ast.parse("def entry(a):\n    return a\n")
        func_node = tree.body[0]
        msg = format_error_message("some pass error", func_node)
        self.assertNotIn("Inline provenance", msg)
        self.assertIn("some pass error", msg)

    def test_find_inline_provenance_tolerates_untagged(self):
        import ast

        node = ast.parse("x = 1").body[0]
        self.assertIsNone(find_inline_provenance(node))
        self.assertIsNone(find_inline_provenance(None))

    def test_callsite_location_appears_in_message(self):
        source = (
            "def entry(a, b):\n"
            "    r0 = _leaf(a)\n"
            "    r1 = _leaf(b)\n"  # <- callsite of copy 2, line 3
            "    return r0 + r1\n")
        new_tree = _run_inliner(source, {"_leaf": self._make_leaf()})
        by_copy = self._nodes_by_copy(new_tree)
        msg = format_error_message("boom", by_copy[2][-1])
        self.assertIn("callsite line 3", msg)


if __name__ == "__main__":
    unittest.main()
