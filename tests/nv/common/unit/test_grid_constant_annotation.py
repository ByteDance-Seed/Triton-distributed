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
"""no_alias / read_only annotations on by-value grid_constant TMA descriptor
entry parameters (CPU codegen legality regression).

The descriptor transport ABI in ``backends/nv/common/export/_cuda.py``
mandates that a singleton TMA descriptor bind to a
``const[grid_constant[TmaDescriptor]]`` entry parameter. That parameter does
not lower to a pointer type: ``_lltype_to_cpp`` renders it as the by-value
``const __grid_constant__ TmaDescriptor``. Before this fix, the codegen
legality check ``annotated and not arg_cpp.rstrip().endswith("*")`` rejected
explicit read_only/no_alias annotations naming such a descriptor parameter,
so the ABI-validated spec was unexportable whenever the frontend also passed
a pointer-decoration annotation for the descriptor name.

These CPU tests pin the acceptance rules:
  * read_only / no_alias on a grid_constant TmaDescriptor entry param renders
    ``const __grid_constant__ TmaDescriptor <name>`` without raising;
  * no ``__restrict__`` is ever appended to the by-value descriptor (illegal
    under the CUDA grid_constant ABI);
  * annotation on a genuinely non-pointer scalar param still raises;
  * an unannotated descriptor param renders identically to before.
"""

import ast
import unittest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import (
    NO_ALIAS_CTX_KEY,
    READ_ONLY_CTX_KEY,
    codegen_cuda,
)
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__

SOURCE = '''
def tma_copy(dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
             out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.threadIdx_x()
    if i < n:
        out[i] = 0.0
'''

SCALAR_ANNOTATED_SOURCE = '''
def scalar_bad(out: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.threadIdx_x()
    if i < n:
        out[i] = 0.0
'''


def _entry_named(name):

    def entry():
        pass

    entry.__name__ = name
    return entry


def _render(source, entry_name, ctx_keys=None):
    tree = ast.parse(source)
    ctx = {"ll": ll, __LITTLE_KERNEL_ENTRY__: _entry_named(entry_name)}
    ctx.update(ctx_keys or {})
    return codegen_cuda(tree, ctx=ctx, emit_header=False)


class GridConstantAnnotationTests(unittest.TestCase):

    def _render_tma_copy(self, ctx_keys=None):
        return _render(SOURCE, "tma_copy", ctx_keys)

    def test_read_only_on_grid_constant_descriptor_param(self):
        cuda = self._render_tma_copy({READ_ONLY_CTX_KEY: frozenset({"dA"})})
        # The by-value descriptor parameter renders as the ABI spelling and
        # the annotation channel no longer raises.
        self.assertIn("const __grid_constant__ TmaDescriptor dA", cuda)

    def test_no_alias_on_grid_constant_descriptor_param(self):
        cuda = self._render_tma_copy({NO_ALIAS_CTX_KEY: frozenset({"dA"})})
        self.assertIn("const __grid_constant__ TmaDescriptor dA", cuda)

    def test_no_restrict_appended_to_byvalue_grid_constant(self):
        # A by-value __grid_constant__ struct parameter must never carry
        # __restrict__: the CUDA ABI forbids pointer qualifiers on it.
        for key in (READ_ONLY_CTX_KEY, NO_ALIAS_CTX_KEY):
            with self.subTest(key=key):
                cuda = self._render_tma_copy({key: frozenset({"dA"})})
                self.assertNotIn("TmaDescriptor __restrict__", cuda)
                self.assertNotIn("TmaDescriptor* ", cuda)

    def test_read_only_also_decorates_pointer_params(self):
        # read_only covering both the descriptor and the tensor param: the
        # pointer param still gets its const qualifier; only the descriptor
        # acceptance is new.
        cuda = self._render_tma_copy({READ_ONLY_CTX_KEY: frozenset({"dA", "out"})})
        self.assertIn("const __grid_constant__ TmaDescriptor dA", cuda)
        self.assertIn("const float* out", cuda)

    def test_unannotated_descriptor_renders_unchanged(self):
        cuda = self._render_tma_copy()
        self.assertIn("const __grid_constant__ TmaDescriptor dA", cuda)
        self.assertNotIn("__restrict__", cuda)

    def test_scalar_param_annotation_still_raises(self):
        # The legality diagnostic is preserved for genuinely non-pointer,
        # non-grid-constant params (e.g. a scalar int).
        with self.assertRaisesRegex(ValueError, "does not lower to a pointer type"):
            _render(SCALAR_ANNOTATED_SOURCE, "scalar_bad", {READ_ONLY_CTX_KEY: frozenset({"n"})})


if __name__ == "__main__":
    unittest.main()
