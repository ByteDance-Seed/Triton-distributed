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
"""CPU codegen regression: element-annotation local array declaration.

Capability: ``ir.allocation.local_array_scalar_annotation``.  A scalar-
annotated assignment ``v: ll.float32 = ll.alloc_local_memory(
ll.float32, N)`` (and the alloc_local_zeros / alloc_local_full siblings)
declares an N-element local (register/local-memory) array whose elements
are addressable as ``v[i]``; the assignment target supplies the buffer
name and the scalar annotation declares the element type.

Previously the family's ``eval_arg_type`` validator was invoked with only
the resolved ``(dtype, elems)`` arguments and aborted type inference with
``alloc_local_memory_eval_arg_type() missing 1 required positional
argument: 'elems'`` before codegen, so no fixed-size local array could be
declared from frontend typed IR.

The fix marks the family with ELEMENT_ANNOTATION_ALLOC_ATTR; statement
type inference normalizes the direct spelling into the builtin's internal
(var_name, dtype, elems) form (idempotently, across repeated inference
passes over the same AST), and a scalar annotation matching the returned
Tensor[element] value type binds the variable to the array type.

Run with:  PYTHONPATH=python uv run --locked --no-sync python -m unittest \\
    discover -s tests/nv/common/unit -p "test_local_array_scalar_annotation.py" -v
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(kernel):
    return kernel.compile(None, codegen_cuda)


# --- kernels (traced once per compile call; no GPU involved) -----------------


@lk.ll_kernel(backend="cuda", is_entry=True)
def _annotated_local_array(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    v: ll.float32 = ll.alloc_local_memory(ll.float32, 4)
    if i < n:
        v[0] = x[i]
        v[1] = v[0] * 2.0
        y[i] = v[1]


@lk.ll_kernel(backend="cuda", is_entry=True)
def _annotated_local_zeros(x: ll.Tensor[ll.float32]) -> ll.void:
    acc: ll.float32 = ll.alloc_local_zeros(ll.float32, 2)
    acc[0] = x[0]
    x[1] = acc[0] + acc[1]


@lk.ll_kernel(backend="cuda", is_entry=True)
def _annotated_local_full(x: ll.Tensor[ll.float32]) -> ll.void:
    buf: ll.float32 = ll.alloc_local_full(ll.float32, 2, 3.5)
    buf[0] = buf[1] + x[0]
    x[0] = buf[0]


@lk.ll_kernel(backend="cuda", is_entry=True)
def _bare_assign_local_array(x: ll.Tensor[ll.float32]) -> ll.void:
    w = ll.alloc_local_memory(ll.float32, 2)
    w[0] = x[0]
    x[1] = w[0]


class LocalArrayScalarAnnotation(unittest.TestCase):

    def test_annotated_local_array_decl_and_element_rw(self):
        """The capability's reproducer: scalar-annotated alloc lowers to a
        C local array with subscript reads/writes."""
        cuda = _emit(_annotated_local_array)
        self.assertIn("float v[4];", cuda)
        self.assertIn("v[0] = x[i];", cuda)
        self.assertIn("v[1] = v[0] * 2.0f;", cuda)
        self.assertIn("y[i] = v[1];", cuda)

    def test_annotated_local_zeros(self):
        """alloc_local_zeros in the same spelling declares + zero-inits.
        (The unaliased local lets const-fold propagate element values, so
        only the declaration and first store are asserted verbatim.)"""
        cuda = _emit(_annotated_local_zeros)
        self.assertIn("float acc[2] = {0.0f, 0.0f};", cuda)
        self.assertIn("acc[0] = x[0];", cuda)
        self.assertNotIn('"acc"', cuda)

    def test_annotated_local_full(self):
        """alloc_local_full in the same spelling declares + fills."""
        cuda = _emit(_annotated_local_full)
        self.assertIn("float buf[2] = {3.5f, 3.5f};", cuda)
        self.assertIn("buf[0] = buf[1] + x[0];", cuda)
        self.assertNotIn('"buf"', cuda)

    def test_bare_assign_also_normalizes(self):
        """The unannotated direct spelling binds the target name too."""
        cuda = _emit(_bare_assign_local_array)
        self.assertIn("float w[2];", cuda)
        self.assertIn("w[0] = x[0];", cuda)

    def test_internal_spelling_unchanged(self):
        """The internal form (leading name literal, assignment-bound) still
        declares the array under that name."""

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def _internal_form(x: ll.Tensor[ll.float32]) -> ll.void:
            s = ll.alloc_local_memory("s", ll.float32, 2)
            s[0] = x[0]
            x[1] = s[0]

        cuda = _emit(_internal_form)
        self.assertIn("float s[2];", cuda)
        self.assertIn("s[0] = x[0];", cuda)

    def test_scalar_annotation_mismatch_still_an_error(self):
        """A non-alloc scalar annotation mismatch remains a hard type error
        (the new acceptance is restricted to marked declaration allocs)."""

        @lk.ll_kernel(backend="cuda", is_entry=True)
        def _bad(x: ll.Tensor[ll.float32]) -> ll.void:
            v: ll.float32 = x  # tensor into scalar annotation  # noqa: F841

        with self.assertRaises(Exception):
            _emit(_bad)


if __name__ == "__main__":
    unittest.main()
