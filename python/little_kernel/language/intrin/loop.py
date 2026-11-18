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

from little_kernel.language.builtin_base import builtin, BuiltinOp
from little_kernel.core.type_system import void


@builtin
class unroll(BuiltinOp):
    """Unroll the loop over the inner range."""

    @staticmethod
    def return_type(inner_range):
        return void

    @staticmethod
    def codegen(inner_range):
        raise RuntimeError("unroll intrinsic should never be called in code generation")


@builtin
class no_unroll(BuiltinOp):
    """Keep the loop rolled even when the compiler can infer its trip count."""

    @staticmethod
    def return_type(inner_range):
        return void

    @staticmethod
    def codegen(inner_range):
        raise RuntimeError("no_unroll intrinsic should never be called in code generation")


@builtin
class unroll_dyn(BuiltinOp):
    """Unroll the loop over the inner range by a fixed count.

    Unlike ll.unroll (a full-unroll request that nvcc silently ignores on
    dynamic trip counts), this emits ``#pragma unroll <count>`` which nvcc
    honors on runtime-bound loops: the body is duplicated <count> times with
    a residue loop, so the target compiler can batch independent memory
    operations (e.g. predicated streaming loads) across the copies. This is
    the generic mechanism for software-pipelining-style load front-loading
    on dynamic-trip loops without hand-duplicating the body.

    Usage: ``for i in ll.unroll_dyn(range(lo, hi, step), 4): ...``
    The count must be a positive integer literal at trace time.
    """

    @staticmethod
    def return_type(inner_range, count):
        return void

    @staticmethod
    def codegen(inner_range, count):
        raise RuntimeError("unroll_dyn intrinsic should never be called in code generation")
