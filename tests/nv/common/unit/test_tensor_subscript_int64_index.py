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
"""CPU codegen regression for 64-bit tensor subscript offsets in the NV
backend (``codegen/visitors/expression_codegen.py`` +
``backends/nv/common/codegen/emitter.py``).

A tensor subscript ``x[i]`` whose index expression is IR-typed 64-bit
(int64/uint64 — e.g. ``row * vocab`` for a large 2-D tile) must keep 64-bit
addressing all the way to the pointer arithmetic: the CUDA emitter wraps
such an index in an explicit ``(ptrdiff_t)`` element offset so the C
usual-arithmetic conversions can never truncate the subscript through a
32-bit intermediate on the way to the address (the historical failure mode
was ``size_t((int)(i)) * sizeof(element)``, which silently re-addressed any
element index with magnitude > INT_MAX). The decision is purely
type-directed: the expression tree is never inspected for a recoverable
64-bit product — any int64-typed index expression is widened, and constant
Python ints are typed by the frontend before codegen.

Indices typed 32-bit (int32/uint32) keep the historical plain ``x[i]``
emission byte-for-byte, so existing programs are unaffected.

Each case compiles a small kernel to CUDA *source text* via
``LLKernel.compile`` (no GPU, no nvcc, no launch ABI) and asserts on the
emitted code.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(kernel):
    return kernel.compile(None, codegen_cuda)


# --- kernels (traced once per compile call; no GPU involved) -----------------


@lk.ll_kernel(backend="cuda", is_entry=False)
def _const_i64_index_kernel(x: ll.Tensor[ll.bfloat16], v: ll.int32) -> ll.void:
    # 823177216 > INT_MAX/2 with bf16 elements; the byte offset overflows a
    # 32-bit int outright, and even the element index is addressed wrongly by
    # any truncation to int through the usual arithmetic conversions.
    i: ll.int64 = 823177216
    if v < -1:
        x[i] = ll.float_to_bf16(1.0)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _row_vocab_store_kernel(x: ll.Tensor[ll.bfloat16], row: ll.int64, col: ll.int32) -> ll.void:
    # Index expression whose *type* is int64 although the tree mixes an
    # int32 operand: type-directed widening must cover the whole expression.
    x[row * 50257 + col] = ll.float_to_bf16(1.0)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _i64_load_kernel(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16], row: ll.int64) -> ll.void:
    y[0] = x[row * 50257]


@lk.ll_kernel(backend="cuda", is_entry=False)
def _u64_index_kernel(x: ll.Tensor[ll.bfloat16], row: ll.uint64) -> ll.void:
    x[row * 50257] = ll.float_to_bf16(1.0)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _2d_store_kernel(x: ll.Tensor[ll.bfloat16], row: ll.int64, col: ll.int32) -> ll.void:
    x[row, col] = ll.float_to_bf16(1.0)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _2d_load_kernel(x: ll.Tensor[ll.bfloat16], y: ll.Tensor[ll.bfloat16], row: ll.int64, col: ll.int64) -> ll.void:
    y[0] = x[row, col]


@lk.ll_kernel(backend="cuda", is_entry=False)
def _2d_i32_kernel(x: ll.Tensor[ll.float32], i: ll.int32, j: ll.int32) -> ll.void:
    x[i, j] = 1.0


@lk.ll_kernel(backend="cuda", is_entry=False)
def _i32_index_kernel(x: ll.Tensor[ll.float32], i: ll.int32) -> ll.void:
    x[i] = 1.0


@lk.ll_kernel(backend="cuda", is_entry=False)
def _small_const_index_kernel(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32]) -> ll.void:
    y[0] = 1.0


# --- checks ------------------------------------------------------------------


class TestInt64SubscriptOffset(unittest.TestCase):

    def test_const_index_above_int_max_keeps_64bit_offset(self):
        src = _emit(_const_i64_index_kernel)
        squashed = src.replace(" ", "")
        # The 64-bit element offset is explicit: base + ptrdiff_t arithmetic.
        self.assertIn("x[(ptrdiff_t)(i)]", squashed, src)
        # No 32-bit narrowing of the index survives anywhere in the emission.
        self.assertNotIn("(int)(i)", squashed, src)
        self.assertNotIn("size_t((int)", squashed, src)
        # ptrdiff_t requires <cstddef>, emitted as a conditional header.
        self.assertIn("#include <cstddef>", src, src)

    def test_int64_index_expression_is_wrapped_as_a_whole(self):
        src = _emit(_row_vocab_store_kernel)
        squashed = src.replace(" ", "")
        self.assertIn("x[(ptrdiff_t)(row*50257+col)]", squashed, src)

    def test_int64_load_subscript_is_wrapped(self):
        src = _emit(_i64_load_kernel)
        squashed = src.replace(" ", "")
        self.assertIn("x[(ptrdiff_t)(row*50257)]", squashed, src)

    def test_uint64_index_is_wrapped(self):
        src = _emit(_u64_index_kernel)
        squashed = src.replace(" ", "")
        self.assertIn("x[(ptrdiff_t)(row*50257)]", squashed, src)

    def test_2d_mixed_width_subscript_wraps_only_the_wide_index(self):
        src = _emit(_2d_store_kernel)
        squashed = src.replace(" ", "")
        # Per-element typing: the int64 row index is widened, the int32
        # column index keeps the plain historical emission.
        self.assertIn("x[(ptrdiff_t)(row)][col]", squashed, src)

    def test_2d_both_wide_indices_are_wrapped(self):
        src = _emit(_2d_load_kernel)
        squashed = src.replace(" ", "")
        self.assertIn("x[(ptrdiff_t)(row)][(ptrdiff_t)(col)]", squashed, src)


class TestWideIndexIsBoundaryDirected(unittest.TestCase):
    """The wrap decision follows the index expression's LLType width."""

    def test_2d_all_i32_indices_stay_plain(self):
        src = _emit(_2d_i32_kernel)
        squashed = src.replace(" ", "")
        self.assertIn("x[i][j]", squashed, src)
        self.assertNotIn("ptrdiff_t", src, src)


class TestNarrowIndicesUnchanged(unittest.TestCase):
    """32-bit-typed indices and small constants keep the legacy emission."""

    def test_i32_index_is_not_wrapped(self):
        src = _emit(_i32_index_kernel)
        squashed = src.replace(" ", "")
        self.assertIn("x[i]", squashed, src)
        self.assertNotIn("ptrdiff_t", src, src)
        self.assertNotIn("<cstddef>", src, src)

    def test_small_constant_index_is_not_wrapped(self):
        src = _emit(_small_const_index_kernel)
        self.assertNotIn("ptrdiff_t", src, src)


if __name__ == "__main__":
    unittest.main()
