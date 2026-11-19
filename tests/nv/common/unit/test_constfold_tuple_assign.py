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
"""ConstantFolder handles tuple-unpack targets mixing Name and Subscript.

Regression: the tuple-value branch of ``ConstantFolder.visit_Assign`` compared
``v.id`` with ``t.id`` without checking that the target element ``t`` is an
``ast.Name``. For ``(b[0], a) = (a, 1)`` the Subscript target element has no
``.id``, so the pass crashed with ``AttributeError``. Mixed targets must be
preserved element-wise: only genuine Name-to-Name self-assignments are dropped.

These tests exercise ``ConstantFolder.visit_Assign`` directly: the end-to-end
``const_fold`` entry point first runs the TypeInferencer, which independently
rejects tuple unpacking into non-Name targets, so the folder's defective branch
is only reachable (and only testable) at the unit level.
"""

import ast
import unittest

from little_kernel.core.passes.constfold import ConstantFolder


def _assign_from(src):
    """Extract the single Assign statement from a one-statement function body."""
    return ast.parse(src).body[0].body[0]


def _folder_in_kernel():
    # in_kernel_function=True keeps constants from being folded into defines,
    # so the tuple branch exercises pure structural handling.
    folder = ConstantFolder(ctx={})
    folder.in_kernel_function = True
    return folder


def _assign_elts(node):
    """Return (target elements, value elements) of a tuple-unpack Assign."""
    assert isinstance(node, ast.Assign) and len(node.targets) == 1
    target = node.targets[0]
    value = node.value
    assert isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple)
    return target.elts, value.elts


class ConstFoldTupleAssignTest(unittest.TestCase):

    def test_subscript_target_name_value_does_not_crash(self):
        # Before the fix: t=Subscript(b[0]) has no .id -> AttributeError.
        folder = _folder_in_kernel()
        out = folder.visit_Assign(_assign_from("def k():\n    (b[0], a) = (a, 1)"))
        self.assertIsNotNone(out)
        targets, values = _assign_elts(out)
        # Both elements are real assignments and must be retained element-wise.
        self.assertIsInstance(targets[0], ast.Subscript)
        self.assertIsInstance(targets[1], ast.Name)
        self.assertEqual(targets[1].id, "a")
        self.assertEqual(len(targets), 2)
        self.assertEqual(len(values), 2)

    def test_pure_name_self_assignment_is_dropped(self):
        # (a, b) = (a, b): every element is a Name self-assign -> whole assign
        # is removed (visit returns None).
        folder = _folder_in_kernel()
        out = folder.visit_Assign(_assign_from("def k():\n    (a, b) = (a, b)"))
        self.assertIsNone(out)

    def test_partial_self_assignment_keeps_real_element(self):
        # (a, c) = (a, 1): element 0 is a self-assign (dropped), element 1 is
        # real and must survive.
        folder = _folder_in_kernel()
        out = folder.visit_Assign(_assign_from("def k():\n    (a, c) = (a, 1)"))
        self.assertIsNotNone(out)
        targets, values = _assign_elts(out)
        self.assertEqual(len(targets), 1)
        self.assertIsInstance(targets[0], ast.Name)
        self.assertEqual(targets[0].id, "c")
        self.assertIsInstance(values[0], ast.Constant)
        self.assertEqual(values[0].value, 1)

    def test_attribute_target_element_preserved(self):
        # Attribute targets (obj.x) also lack .id; they must not crash and the
        # assignment must be preserved.
        folder = _folder_in_kernel()
        out = folder.visit_Assign(_assign_from("def k():\n    (obj.x, a) = (a, 1)"))
        self.assertIsNotNone(out)
        targets, _ = _assign_elts(out)
        self.assertIsInstance(targets[0], ast.Attribute)
        self.assertEqual(targets[0].attr, "x")
        self.assertIsInstance(targets[1], ast.Name)

    def test_swapped_order_mixed_targets(self):
        # Name value paired with a Subscript target is the exact reported crash.
        folder = _folder_in_kernel()
        out = folder.visit_Assign(_assign_from("def k():\n    (a, b[0]) = (1, c)"))
        self.assertIsNotNone(out)
        targets, _ = _assign_elts(out)
        self.assertIsInstance(targets[0], ast.Name)
        self.assertIsInstance(targets[1], ast.Subscript)


if __name__ == "__main__":
    unittest.main()
