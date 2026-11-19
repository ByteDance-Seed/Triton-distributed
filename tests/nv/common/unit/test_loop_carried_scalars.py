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
"""Regression tests for the loop-carried mutable scalar contract.

Language contract (documented in little_kernel/README.md):
an annotated scalar declared ABOVE a runtime-bounds
``for i in range(start, stop, step)`` loop and reassigned inside the loop
body (plain ``=``, augmented ``+=``-style, or conditionally under a nested
runtime ``if``) is emitted exactly once as a C++ declaration before the
loop and mutated in place on every iteration — i.e. registers carry across
iterations with the semantics of a mutable CUDA variable.  This is the
software-prefetch formulation (resolve iteration i+stride's indices while
computing iteration i).  The frontend pipeline is AST -> passes -> codegen
(core/compile.py), so there is no SSA restriction; these tests pin the
emission contract so a future emitter change cannot silently redeclare a
carried variable inside the loop (which would break the carry).

Discipline pinned here:
* annotate the carried variable ONCE, before the loop;
* reassignment sites inside the loop are PLAIN assignments, never
  re-annotated (re-annotation would shadow/redeclare);
* loop-local temporaries ARE annotated inside the loop body and must not
  leak declarations of carried names.
"""

import re
import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES


def _compile(fn, disable_var_hoisting=False):
    return fn.compile(PASSES["cuda"], codegen_cuda, num_threads=128, disable_var_hoisting=disable_var_hoisting)


def _loop_excerpt(source):
    """Return the for-header and loop body from the emitted kernel."""
    idx = source.find("for (int32_t i = tid;")
    if idx < 0:
        raise AssertionError("no runtime for-loop in emitted CUDA:\n" + source)
    # The header contains nested parens (range-with-ternary cond), so find
    # the opening brace after the header instead of matching `([^)]*)`.
    start = source.index("{", idx)
    depth = 0
    for pos in range(start, len(source)):
        if source[pos] == "{":
            depth += 1
        elif source[pos] == "}":
            depth -= 1
            if depth == 0:
                return source[idx:pos]
    raise AssertionError("unbalanced braces in emitted CUDA")


@lk.ll_kernel(backend="cuda", is_entry=True)
def _plain_carry(src: ll.Tensor[ll.int32], dst: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    tid: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    stride: ll.int32 = ll.gridDim_x() * 128
    prev: ll.int32 = 0
    for i in range(tid, n, stride):
        cur: ll.int32 = prev
        v: ll.int32 = ll.int32(src[i])
        prev = v
        dst[i] = cur + v


@lk.ll_kernel(backend="cuda", is_entry=True)
def _prefetch_group(src: ll.Tensor[ll.int32], dst: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    tid: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    stride: ll.int32 = ll.gridDim_x() * 128
    nxt: ll.int32 = ll.int32(src[tid])
    total: ll.int32 = 0
    for i in range(tid, n, stride):
        cur: ll.int32 = nxt
        ahead: ll.int32 = i + stride
        if ahead < n:
            nxt = ll.int32(src[ahead])
            total = total + 1
        dst[i] = cur * 3 + total


@lk.ll_kernel(backend="cuda", is_entry=True)
def _aug_carry(src: ll.Tensor[ll.int32], dst: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    tid: ll.int32 = ll.blockIdx_x() * 128 + ll.threadIdx_x()
    stride: ll.int32 = ll.gridDim_x() * 128
    total: ll.int32 = 0
    for i in range(tid, n, stride):
        total += ll.int32(src[i])
    dst[tid] = total


class TestLoopCarriedScalars(unittest.TestCase):

    def test_plain_reassignment_carries_across_iterations(self):
        source = _compile(_plain_carry)
        # Single declaration of the carried name, before the loop.
        decls = re.findall(r"int32_t prev\b", source)
        self.assertEqual(len(decls), 1, source)
        self.assertIn("int32_t prev = 0;", source)
        self.assertLess(source.index("int32_t prev = 0;"), source.index("for (int32_t i = tid;"))
        loop = _loop_excerpt(source)
        # Plain assignment inside the loop, never a redeclaration.
        self.assertIn("prev = v;", loop)
        self.assertNotIn("int32_t prev", loop)

    def test_prefetch_group_with_conditional_update(self):
        source = _compile(_prefetch_group)
        for name in ("nxt", "total"):
            self.assertEqual(len(re.findall(r"int32_t %s\b" % name, source)), 1, source)
        self.assertLess(source.index("int32_t nxt = "), source.index("for (int32_t i = tid;"))
        loop = _loop_excerpt(source)
        # Runtime-guarded in-place update of the carried registers.
        self.assertIn("if (ahead < n) {", loop)
        self.assertIn("nxt = static_cast<int32_t>(src[ahead]);", loop)
        self.assertIn("total = total + 1;", loop)
        self.assertNotIn("int32_t nxt", loop)
        self.assertNotIn("int32_t total", loop)

    def test_augmented_assignment_carries(self):
        source = _compile(_aug_carry)
        self.assertEqual(len(re.findall(r"int32_t total\b", source)), 1, source)
        loop = _loop_excerpt(source)
        self.assertIn("total += static_cast<int32_t>(src[i]);", loop)
        self.assertNotIn("int32_t total", loop)

    def test_disable_var_hoisting_preserves_carry(self):
        for disabling in (False, True):
            source = _compile(_prefetch_group, disable_var_hoisting=disabling)
            loop = _loop_excerpt(source)
            # Carried names are declared outside the loop in both modes and
            # mutated in place; semantics identical by construction.
            self.assertIn("nxt = static_cast<int32_t>(src[ahead]);", loop)
            self.assertNotIn("int32_t nxt", loop)
            self.assertNotIn("int32_t total", loop)

    def test_loop_local_temporaries_are_not_carried_declarations(self):
        source = _compile(_prefetch_group)
        # Loop-local annotated temporaries are hoisted as UNINITIALIZED
        # declarations (never re-declared inside the loop either).
        self.assertEqual(len(re.findall(r"int32_t cur\b", source)), 1, source)
        self.assertIn("int32_t cur;", source)
        loop = _loop_excerpt(source)
        self.assertIn("cur = nxt;", loop)
        self.assertNotIn("int32_t cur", loop)
        self.assertNotIn("int32_t ahead", loop)


if __name__ == "__main__":
    unittest.main()
