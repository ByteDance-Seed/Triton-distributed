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
"""CPU regression: ll.tensor_view typed element view (no GPU).

Capability ``ir.elementwise_tensor_retyped_view`` / dependent task diagnostic:
viewing a narrow kernel argument as wider element storage for dtype-keyed
intrinsics used to fail type inference at trace with

    gmem_ld_scalar expects Tensor as first argument, got IntType: uint16

because ``ll.ptr_cast(x, ll.uint16)`` is typed by the tracer as the target
*scalar* type (it exists for raw pointer expressions such as TMA descriptors),
not as a tensor.  ``ll.tensor_view(src, dtype)`` closes that gap: it keeps the
same underlying storage, returns a genuine ``TensorType[dtype]`` so the typed
subscript and ``gmem_ld_scalar`` cache-policy paths accept it, and lowers to a
``reinterpret_cast<dtype*>(src)`` at emission — for a shared-memory array the
identifier decays to its base pointer, so ``view[i]`` type-checks exactly as a
native ``Tensor[dtype]`` argument.

This file covers:

* the return-type contract at unit level (tensor source -> TensorType of the
  requested dtype; scalar source or non-type dtype raise);
* end-to-end CPU codegen of the reported minimal reproducer — a uint8 kernel
  argument viewed as uint16 feeding the 'nc' cache-policy gmem_ld_scalar —
  which must trace, lower and emit CUDA source without the element-vs-tensor
  TypeError (and must contain the uint16 reinterpret and __ldg);
* end-to-end CPU codegen of a retyped shared-memory allocation: a uint32
  shared array viewed as uint8 with byte subscript stores, which must emit
  the shared declaration plus its uint8 reinterpret.
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.core.passes import PASSES
from little_kernel.core.type_system import (
    TensorType,
    bfloat16,
    uint8,
    uint16,
    uint32,
)
from little_kernel.language.intrin.dtype import tensor_view
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


@lk.ll_kernel(backend='cuda', is_entry=True)
def _gmem_arg_view_repro(x: ll.Tensor[ll.uint8], y: ll.Tensor[ll.uint8]) -> ll.void:
    """The reported minimal reproducer: kernel-argument element retype."""
    xin = ll.tensor_view(x, ll.uint16)
    v: ll.uint16 = ll.gmem_ld_scalar(xin, ll.threadIdx_x(), 'nc')
    ll.gmem_st_u8(ll.ptr_byte_offset(y, ll.threadIdx_x()), ll.val_cast(v, ll.uint32) & 255)


@lk.ll_kernel(backend='cuda', is_entry=True)
def _shared_view_repro(y: ll.Tensor[ll.uint8]) -> ll.void:
    """Retyped shared-memory allocation: uint32 array staged as bytes."""
    s32 = ll.alloc_shared_memory("s_view", ll.uint32, 64, 16)
    s8 = ll.tensor_view(s32, ll.uint8)
    i: ll.int32 = ll.threadIdx_x()
    s8[i] = ll.val_cast(i, ll.uint32) & 255
    y[i] = s8[i]


class TestTensorViewReturnType(unittest.TestCase):
    """return_type contract, reached via the builtin's __eval_return_type__.

    The @builtin decorator wraps the BuiltinOp subclass into a plain
    function whose type-inference entry point lives under
    __eval_return_type__, so the unit checks drive that entry point
    directly (the same path the tracer calls).
    """

    @staticmethod
    def _rt(*args):
        return tensor_view.__eval_return_type__(*args)

    def test_tensor_source_retypes_to_requested_dtype(self):
        self.assertEqual(self._rt(TensorType(uint8), uint16), TensorType(uint16))
        self.assertEqual(self._rt(TensorType(uint32), uint8), TensorType(uint8))
        self.assertEqual(self._rt(TensorType(bfloat16), uint16), TensorType(uint16))

    def test_scalar_source_rejected(self):
        with self.assertRaises(TypeError):
            self._rt(uint16, uint8)

    def test_non_type_dtype_rejected(self):
        with self.assertRaises(TypeError):
            self._rt(TensorType(uint8), "uint16_t")


class TestKernelArgViewCodegen(unittest.TestCase):

    def test_minimal_reproducer_lowers_end_to_end(self):
        # Trace + full CPU codegen of the capability reproducer; must not
        # raise the "expects Tensor as first argument" TypeError that
        # ptr_cast's scalar typing produces on this kernel shape.
        src = _gmem_arg_view_repro.compile(PASSES["cuda"], codegen_cuda)
        self.assertIsInstance(src, str)
        # The uint8 argument is reinterpreted as uint16 elements for the
        # cache-policy ('nc' -> __ldg) scalar load.
        self.assertIn("uint16_t", src)
        self.assertIn("__ldg", src)
        # The byte store path through ptr_byte_offset survives.
        self.assertIn("lk_ptr_byte_offset", src)

    def test_retyped_shared_allocation_lowers_end_to_end(self):
        src = _shared_view_repro.compile(PASSES["cuda"], codegen_cuda)
        self.assertIsInstance(src, str)
        # The native uint32 shared declaration is emitted, and the view
        # reinterprets the shared array's base as uint8 elements; typed byte
        # subscripts read/write through the view.
        self.assertIn("__shared__", src)
        self.assertIn("uint32_t s_view[64]", src)
        self.assertIn("reinterpret_cast<uint8_t*>", src)
        self.assertIn("uint8_t* s8", src)


if __name__ == "__main__":
    unittest.main()
