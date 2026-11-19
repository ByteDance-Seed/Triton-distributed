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
"""
Unit tests for register_by_value_arguments().

These guard the stale-metadata defect: re-registering the SAME function object
with an identical parameter set is a legal no-op, but re-registering it with a
DIFFERENT parameter set must raise ValueError rather than silently keeping the
first (stale) registration.

Registry isolation: by_value_argument_primitives() reads a module-level list.
Each test registers a fresh locally-defined builtin function and removes any
entries it appended in finally, so pre-existing real registrations (e.g.
alloc_shared_memory) remain untouched and unaffected by our checks.
"""

import unittest

from little_kernel.language import builtin_base
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    by_value_argument_primitives,
    register_by_value_arguments,
)


def _make_builtin(name):
    """Create a fresh function object carrying BUILTIN_ATTR."""

    def fn(*args, **kwargs):
        raise RuntimeError("placeholder builtin")

    fn.__name__ = name
    setattr(fn, BUILTIN_ATTR, True)
    return fn


def _prune(functions):
    """Remove all registry entries whose function object is in ``functions``."""
    keep = [(fn, names) for fn, names in builtin_base._BY_VALUE_ARGUMENTS if fn not in functions]
    builtin_base._BY_VALUE_ARGUMENTS[:] = keep


def _names_for(function):
    for candidate, names in by_value_argument_primitives():
        if candidate is function:
            return names
    return None


class TestRegisterByValueArguments(unittest.TestCase):

    def setUp(self):
        self._created = []

    def tearDown(self):
        _prune(set(self._created))
        self._created = []

    def _fn(self, name):
        fn = _make_builtin(name)
        self._created.append(fn)
        return fn

    def test_first_registration_appends(self):
        fn = self._fn("fresh_first")
        register_by_value_arguments(fn, ("a", "b"))
        self.assertEqual(_names_for(fn), frozenset({"a", "b"}))

    def test_same_set_reregistration_is_noop(self):
        fn = self._fn("fresh_same_set")
        register_by_value_arguments(fn, {"a"})
        register_by_value_arguments(fn, {"a"})
        self.assertEqual(_names_for(fn), frozenset({"a"}))

    def test_same_set_reregistration_order_independent(self):
        fn = self._fn("fresh_order")
        register_by_value_arguments(fn, ("a", "b"))
        register_by_value_arguments(fn, ("b", "a"))
        self.assertEqual(_names_for(fn), frozenset({"a", "b"}))

    def test_same_set_reregistration_leaves_registry_size_unchanged(self):
        fn = self._fn("fresh_size")
        before = len(by_value_argument_primitives())
        register_by_value_arguments(fn, {"a"})
        mid = len(by_value_argument_primitives())
        register_by_value_arguments(fn, {"a"})
        after = len(by_value_argument_primitives())
        self.assertEqual(mid, before + 1)
        self.assertEqual(after, mid)

    def test_differing_set_reregistration_raises(self):
        fn = self._fn("fresh_differing")
        register_by_value_arguments(fn, {"a"})
        with self.assertRaises(ValueError):
            register_by_value_arguments(fn, {"a", "b"})

    def test_differing_set_reregistration_keeps_original(self):
        fn = self._fn("fresh_keep")
        register_by_value_arguments(fn, {"a"})
        try:
            register_by_value_arguments(fn, {"b"})
        except ValueError:
            pass
        else:
            self.fail("expected ValueError")
        self.assertEqual(_names_for(fn), frozenset({"a"}))

    def test_differing_set_removal_raises(self):
        fn = self._fn("fresh_removal")
        register_by_value_arguments(fn, {"a", "b"})
        with self.assertRaises(ValueError):
            register_by_value_arguments(fn, {"a"})
        self.assertEqual(_names_for(fn), frozenset({"a", "b"}))

    def test_distinct_function_objects_register_independently(self):
        fn1 = self._fn("distinct_one")
        fn2 = self._fn("distinct_two")
        register_by_value_arguments(fn1, {"a"})
        register_by_value_arguments(fn2, {"a", "c"})
        self.assertEqual(_names_for(fn1), frozenset({"a"}))
        self.assertEqual(_names_for(fn2), frozenset({"a", "c"}))

    def test_non_builtin_rejected(self):

        def plain():
            pass

        with self.assertRaises(TypeError):
            register_by_value_arguments(plain, {"a"})

    def test_empty_name_rejected(self):
        fn = self._fn("fresh_empty_name")
        with self.assertRaises(TypeError):
            register_by_value_arguments(fn, {""})

    def test_non_string_name_rejected(self):
        fn = self._fn("fresh_nonstring_name")
        with self.assertRaises(TypeError):
            register_by_value_arguments(fn, {1})


if __name__ == '__main__':
    unittest.main()
