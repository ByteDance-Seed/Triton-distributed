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
"""Tests for stock ll.alloc_shared_memory with Attribute dtype arguments.

Two stacked bugs used to make the stock builtin unusable from frontends
(cycle-004/006/007/008 kernel-00 requests; frontends shipped private
per-dtype smem builtins as a workaround):

1. Type inference: ``_infer_builtin_call`` forwarded the raw ``ast.Attribute``
   node for a dtype argument like ``ll.float32`` to the builtin's
   ``eval_arg_type`` hook, whose ``Tensor[dtype]`` indexing then raised
   TypeInferenceError.  The hook must receive the resolved LLType.
2. Codegen: string arguments arrive at codegen functions as codegened C
   string literals (with quotes); ``codegen_alloc_shared_memory`` inlined the
   quoted name, emitting ``__shared__ int32_t "s_rank"[192];`` (invalid C++).
   Following the ``codegen_alloc_local_memory`` convention, the name is
   stripped of quotes before emission.
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


class TestAllocSharedMemoryDtypeAttr(unittest.TestCase):

    def test_shared_alloc_resolves_dtype_attribute(self):
        """stock alloc_shared_memory accepts ll.<dtype> Attribute args."""
        code = _emit("""
def kern(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
    ll.alloc_shared_memory("s_buf", ll.float32, 256, 16)
    i = ll.blockIdx_x() * 256 + ll.threadIdx_x()
    if i < n:
        s_buf[i % 256] = x[i]
    ll.__syncthreads()
    if i < n:
        y[i] = s_buf[i % 256] * 1.0
""")
        # Valid C declaration: dtype resolved, name unquoted.
        self.assertIn("__shared__ __align__(16) float s_buf[256]", code)
        self.assertNotIn('"s_buf"[', code)

    def test_shared_alloc_unaligned_and_int64(self):
        """No-align variant and a second dtype (int64)."""
        code = _emit("""
def kern(p: ll.Tensor[ll.int64], q: ll.Tensor[ll.int64], n: ll.int32) -> ll.void:
    ll.alloc_shared_memory("s_ptr", ll.int64, 64)
    i = ll.threadIdx_x()
    if i < 64:
        s_ptr[i] = p[i]
    ll.__syncthreads()
    if i < 64:
        q[i] = s_ptr[i]
""")
        self.assertIn("__shared__ int64_t s_ptr[64]", code)
        self.assertNotIn('"s_ptr"[', code)

    def test_shared_alloc_int32_int64_align16(self):
        """int32 + int64, both align-16: the smem-table decl shapes consumed
        by frontends replacing private per-dtype smem builtins.

        Locks the bare-statement call convention: the ``__shared__``
        declaration is emitted by the call itself and the name is registered
        in the type context for later indexing.  (Assigning the call result
        would wrap the declaration in a pointer initializer, which is
        invalid C++ — stock usage is the bare statement.)"""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32], y: ll.Tensor[ll.int64]) -> ll.void:
    ll.alloc_shared_memory("s_rank", ll.int32, 192, 16)
    ll.alloc_shared_memory("s_ptr", ll.int64, 576, 16)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < 192:
        s_rank[tid] = x[tid]
    if tid < 128:
        s_ptr[tid] = y[tid]
    ll.__syncthreads()
""")
        self.assertIn("__shared__ __align__(16) int32_t s_rank[192]", code)
        self.assertIn("__shared__ __align__(16) int64_t s_ptr[576]", code)
        self.assertNotIn('"s_rank"[', code)
        self.assertNotIn('"s_ptr"[', code)

    def test_swapped_dtype_elems_order_fails_loudly(self):
        """alloc_shared_memory(name, elems, dtype) is NOT the stock signature
        (dtype is the second argument); the swapped order must fail at type
        inference with a clear error, not emit wrong C++."""
        with self.assertRaises(Exception) as cm:
            _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    ll.alloc_shared_memory("s_i", 192, ll.int32)
""")
        self.assertIn("LLType", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
