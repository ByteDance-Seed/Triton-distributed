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
# publish, distribute, sublicense, and/or sell copies of the Software.
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
"""Unit tests for the opt-in __LITTLE_KERNEL_CUDA_RESTRICT_PARAMS__ knob.

The knob lists function parameter names the frontend asserts are
non-aliasing; codegen renders them as `T* __restrict__ name`. Unlisted
parameters keep the plain spelling. No GPU required.
"""
import ast
import textwrap
import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda

_SRC = textwrap.dedent('''\
    def kern(src: Tensor[uint8], dst: Tensor[uint8], n: int32) -> void:
        pass
''')

CTX = dict(ll.__dict__)
KEY = "__LITTLE_KERNEL_CUDA_RESTRICT_PARAMS__"


def _compile(names=None):
    ctx = dict(CTX)
    if names is not None:
        ctx[KEY] = names
    return codegen_cuda(ast.parse(_SRC), ctx)


class TestRestrictParams(unittest.TestCase):

    def test_knob_absent_plain_signature(self):
        out = _compile()
        self.assertNotIn("__restrict__", out)
        self.assertIn("uint8_t* src", out)

    def test_listed_params_get_restrict(self):
        out = _compile(("src", "dst"))
        self.assertIn("uint8_t* __restrict__ src", out)
        self.assertIn("uint8_t* __restrict__ dst", out)

    def test_unlisted_params_unchanged(self):
        out = _compile(("src", ))
        self.assertIn("uint8_t* __restrict__ src", out)
        self.assertIn("uint8_t* dst", out)
        self.assertNotIn("__restrict__ dst", out)

    def test_scalars_never_annotated(self):
        out = _compile(("src", "dst"))
        self.assertNotIn("__restrict__ n", out)
        self.assertIn("int32_t n", out)

    def test_scalar_listed_raises_type_error(self):
        with self.assertRaisesRegex(TypeError, "'n'"):
            _compile(("n", ))

    def test_unknown_name_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "scr"):
            _compile(("scr", ))

    def test_non_string_entries_raise_type_error(self):
        with self.assertRaises(TypeError):
            _compile((1, 2))

    def test_empty_knob_same_as_absent(self):
        self.assertEqual(_compile(()), _compile())


if __name__ == "__main__":
    unittest.main()
