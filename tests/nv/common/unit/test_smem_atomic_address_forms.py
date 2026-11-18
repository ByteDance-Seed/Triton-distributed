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
"""Address-form contract for scalar atomics into static shared memory.

The natural idiom ``ll.atomic_add(smem[idx], v)`` on an array declared with
``ll.alloc_shared_memory`` passes an element *lvalue* whose inferred type is
the element type.  CUDA atomics take a pointer, so emitting the subscript
verbatim (``atomicAdd(smem[idx], v)`` — an ``int32_t``, not an
``int32_t*``) fails NVCC with "no instance of overloaded function atomicAdd
matches the argument list".  The intrinsics in language/intrin/atomic.py are
typed-address ops: an element-typed address operand is taken by address
(``&(smem[idx])``) via ``_atomic_addr_ptr``, while a pointer-typed operand
(``smem + idx``) is already in address position and passes through.

These tests lock the end-to-end codegen contract for both address forms on
both static shared arrays and global tensors, across the atomic family that
shares the ``_atomic_addr_ptr`` normalization (add / cas / inc / min).  All
assertions are on emitted text only (deterministic, no GPU).
"""

import unittest
import ast

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(source: str) -> str:
    tree = ast.parse(source)
    return codegen_cuda(tree, ctx={"ll": ll}, emit_header=False)


class TestSharedMemoryAtomicAddressForms(unittest.TestCase):
    """Subscript (lvalue) vs additive (pointer) address forms on smem."""

    def test_subscript_form_takes_address_of_element(self):
        """atomic_add(cnt[tid], 1) emits atomicAdd(&(cnt[tid]), 1)."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    ll.alloc_shared_memory("cnt", ll.int32, 64)
    tid: ll.int32 = ll.threadIdx_x()
    ll.atomic_add(cnt[tid], 1)
    ll.__syncthreads()
    x[tid] = cnt[tid]
""")
        self.assertIn("__shared__ int32_t cnt[64];", code)
        self.assertIn("atomicAdd(&(cnt[tid]), 1);", code)
        # The raw-lvalue form that NVCC rejects must not appear.
        self.assertNotIn("atomicAdd(cnt[tid], 1);", code)

    def test_pointer_form_passes_through_unchanged(self):
        """atomic_add(cnt + tid, 1) stays atomicAdd(cnt + tid, 1)."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    ll.alloc_shared_memory("cnt", ll.int32, 64)
    tid: ll.int32 = ll.threadIdx_x()
    ll.atomic_add(cnt + tid, 1)
    ll.__syncthreads()
    x[tid] = cnt[tid]
""")
        self.assertIn("atomicAdd(cnt + tid, 1);", code)
        self.assertNotIn("atomicAdd(&(cnt + tid), 1);", code)

    def test_atomic_family_shares_the_normalization(self):
        """cas / min / inc take subscript addresses on smem the same way."""
        code = _emit("""
def kern(x: ll.Tensor[ll.int32]) -> ll.void:
    ll.alloc_shared_memory("cnt", ll.int32, 64)
    tid: ll.int32 = ll.threadIdx_x()
    ll.atomic_cas(cnt[tid], 0, 1)
    ll.atomic_min(cnt[tid], 7)
    ll.atomic_inc(cnt[tid], 63)
    ll.__syncthreads()
    x[tid] = cnt[tid]
""")
        self.assertIn("atomicCAS(&(cnt[tid]), 0, 1);", code)
        self.assertIn("atomicMin(&(cnt[tid]), 7);", code)
        self.assertIn("atomic_inc_fn(&(cnt[tid]), 63);", code)
        self.assertNotIn("atomicCAS(cnt[tid], 0, 1);", code)

    def test_global_tensor_subscript_and_pointer_forms(self):
        """The same contract holds on a global tensor parameter: the
        subscript lvalue is taken by address, ``y + tid`` passes through."""
        code = _emit("""
def kern(y: ll.Tensor[ll.int32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    ll.atomic_add(y[tid], 2)
    ll.atomic_add(y + tid, 3)
""")
        self.assertIn("atomicAdd(&(y[tid]), 2);", code)
        self.assertIn("atomicAdd(y + tid, 3);", code)
        self.assertNotIn("atomicAdd(y[tid], 2);", code)

    def test_atomic_family_global_pointer_forms_unchanged(self):
        """Pointer operands are never double-addressed, global or shared."""
        code = _emit("""
def kern(y: ll.Tensor[ll.int32]) -> ll.void:
    tid: ll.int32 = ll.threadIdx_x()
    ll.atomic_cas(y + tid, 0, 1)
    ll.atomic_inc(y + tid, 63)
""")
        self.assertIn("atomicCAS(y + tid, 0, 1);", code)
        self.assertIn("atomic_inc_fn(y + tid, 63);", code)
        self.assertNotIn("atomicCAS(&(y + tid)", code)
        self.assertNotIn("atomic_inc_fn(&(y + tid)", code)


if __name__ == "__main__":
    unittest.main()
