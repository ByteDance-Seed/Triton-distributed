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
# including without limitation the rights to use, copy, modify, merge
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""CPU codegen regression for same-width signed-int typed casts.

Pins the emission contract of explicit LL typed casts in the NV/CUDA
codegen (``backends/nv/common/codegen/emitter.py`` +
``codegen/visitors/call_codegen.py``): an explicit cast along either
direction of a same-width signed/unsigned integer pair — written either in
type-constructor form (``ll.int64(ll.uint32(x))``) or in free-function form
(``ll.val_cast(x, ll.int64)`` / ``ll.to(x, ll.int64)``) — must be emitted as
``static_cast<dst>(src)`` with the operand fully wrapped.

The fully-parenthesized form matters: emitting parentheses-free text would
let an outer accumulator expression bind across the cast and silently
change both the type and the value of the result.

No CUDA toolchain or GPU is needed; generated source is consumed as text.
"""

import importlib.util
import os
import unittest

from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES


def _compile(module_globals, fn_name):
    """Compile a fixture kernel from cast_fixtures.py to CUDA source."""
    return module_globals[fn_name].compile(passes=PASSES["cuda"], codegen_func=codegen_cuda)


def _load_fixtures():
    path = os.path.join(os.path.dirname(__file__), "cast_fixtures.py")
    spec = importlib.util.spec_from_file_location("cast_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__dict__


FIXTURES = _load_fixtures()


class TestTypedIntCastCodegen(unittest.TestCase):

    def assert_has_static_cast(self, src, dst_cpp, src_text):
        needle = f"static_cast<{dst_cpp}>({src_text})"
        self.assertIn(needle, src)

    # --- both directions of the pinned int64<->uint32 pair -------------

    def test_uint32_to_int64_type_call(self):
        src = _compile(FIXTURES, "u32_to_i64")
        self.assert_has_static_cast(src, "int64_t", "static_cast<uint32_t>(x[i])")

    def test_int64_to_uint32_type_call(self):
        src = _compile(FIXTURES, "i64_to_u32")
        self.assert_has_static_cast(src, "uint32_t", "static_cast<int64_t>(x[i])")

    def test_uint32_to_int64_val_cast(self):
        src = _compile(FIXTURES, "vc_u32_i64")
        self.assert_has_static_cast(src, "int64_t", "static_cast<uint32_t>(x[i])")

    def test_int64_to_uint32_val_cast(self):
        src = _compile(FIXTURES, "vc_i64_u32")
        self.assert_has_static_cast(src, "uint32_t", "static_cast<int64_t>(x[i])")

    def test_uint32_to_int64_to_fn(self):
        src = _compile(FIXTURES, "to_u32_i64")
        self.assert_has_static_cast(src, "int64_t", "static_cast<uint32_t>(x[i])")

    # --- nested casts ----------------------------------------------------

    def test_nested_casts(self):
        src = _compile(FIXTURES, "nested")
        self.assert_has_static_cast(
            src,
            "int64_t",
            "static_cast<uint32_t>(static_cast<int64_t>(x[i]))",
        )

    def test_constant_operand(self):
        src = _compile(FIXTURES, "const_operand")
        self.assert_has_static_cast(src, "int64_t", "static_cast<uint32_t>(11)")

    # --- other same-width signed pairs -----------------------------------

    def test_signed_pairs_all_widths(self):
        cases = [
            ("i64_u64", "uint64_t", "x[i]"),
            ("u64_i64", "int64_t", "x[i]"),
            ("i32_u32", "uint32_t", "x[i]"),
            ("u32_i32", "int32_t", "x[i]"),
            ("i16_u16", "uint16_t", "x[i]"),
            ("u16_i16", "int16_t", "x[i]"),
            ("i8_u8", "uint8_t", "x[i]"),
            ("u8_i8", "int8_t", "x[i]"),
        ]
        for fn_name, dst_cpp, src_text in cases:
            with self.subTest(fn=fn_name):
                src = _compile(FIXTURES, fn_name)
                self.assert_has_static_cast(src, dst_cpp, src_text)

    # --- operand wrapping / accumulator binding --------------------------

    def test_cast_wraps_operand_in_outer_expression(self):
        src = _compile(FIXTURES, "paren_binding")
        # The cast must close before the surrounding arithmetic binds: the
        # accumulator adds 2 to the int64_t-typed cast result, and the whole
        # sum is multiplied by 3.  A parentheses-free emission would change
        # both the type and the value here.
        self.assertIn(
            "(static_cast<int64_t>(static_cast<uint32_t>(11)) + 2) * 3",
            src,
        )


if __name__ == "__main__":
    unittest.main()
