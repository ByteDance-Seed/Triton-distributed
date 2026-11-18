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
"""CPU codegen regression: scalar static array extents in container form.

Capability: ``ir.alloc_shared_memory.static_extent``.  A static array's
element count may be spelled as a bare integer expression (``512``) or as
a one-element shape container (``[512]`` / ``(512,)``) — frontend code
conventionally uses the shape form even for a 1-D scratch buffer.  Both
name the same compile-time constant bound on backends that require one
(CUDA: ``__shared__ T xs[512];`` and the local-array equivalent), so the
validator must accept a foldable 1-element container and the emitter must
unwrap it to the scalar expression — the generic ``visit_List`` rendering
``{512}`` is braced-aggregate-init syntax and is not a valid C++ array
bound inside a declarator.

Equally important are the conservative negative cases: a multi-element
container is a real shape, not a scalar extent, and must keep failing the
constant-extent check; so must a one-element container that wraps a
runtime value.

Run with:  uv run --locked --no-sync python -m unittest \\
    tests.nv.common.unit.test_static_extent_container -v
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(kernel):
    return kernel.compile(None, codegen_cuda)


# --- kernels (CPU lowering only; no GPU involved) ----------------------------


@lk.ll_kernel(backend="cuda", is_entry=True)
def _shared_literal_list_extent(x: ll.Tensor[ll.float16]) -> ll.void:
    xs = ll.alloc_shared_memory("xs", ll.float16, [512], 16)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < 512:
        xs[tid] = x[tid]
    ll.__syncthreads()


@lk.ll_kernel(backend="cuda", is_entry=True)
def _shared_literal_tuple_extent(x: ll.Tensor[ll.float16]) -> ll.void:
    xs = ll.alloc_shared_memory("xs", ll.float16, (256, ), 16)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < 256:
        xs[tid] = x[tid]
    ll.__syncthreads()


@lk.ll_kernel(backend="cuda", is_entry=True)
def _shared_arith_over_typed_local_in_list(x: ll.Tensor[ll.float16]) -> ll.void:
    TILE: ll.int32 = 128
    xs = ll.alloc_shared_memory("xs", ll.float16, [TILE * 4], 16)
    tid: ll.int32 = ll.threadIdx_x()
    if tid < 512:
        xs[tid] = x[tid]
    ll.__syncthreads()


@lk.ll_kernel(backend="cuda", is_entry=True)
def _local_literal_list_extent(x: ll.Tensor[ll.int32]) -> ll.void:
    buf = ll.alloc_local_memory("buf", ll.int32, [8])
    tid: ll.int32 = ll.threadIdx_x()
    if tid < 8:
        buf[tid] = x[tid]


@lk.ll_kernel(backend="cuda", is_entry=True)
def _shared_multi_element_list_rejected(x: ll.Tensor[ll.int32]) -> ll.void:
    buf = ll.alloc_shared_memory("buf", ll.int32, [4, 4], 16)  # noqa: F841


@lk.ll_kernel(backend="cuda", is_entry=True)
def _shared_runtime_extent_rejected(x: ll.Tensor[ll.int32], n: ll.int32) -> ll.void:
    buf = ll.alloc_shared_memory("buf", ll.int32, [n], 16)  # noqa: F841


class StaticExtentContainer(unittest.TestCase):

    def test_shared_literal_list_extent(self):
        """``[512]`` literal extent emits ``__shared__ __half xs[512]``."""
        cuda = _emit(_shared_literal_list_extent)
        self.assertIn("__shared__ __align__(16) __half xs[512];", cuda)
        self.assertIn("xs[tid] = x[tid];", cuda)

    def test_shared_literal_tuple_extent(self):
        """A 1-element tuple extent ``(256,)`` unwraps the same way."""
        cuda = _emit(_shared_literal_tuple_extent)
        self.assertIn("__shared__ __align__(16) __half xs[256];", cuda)

    def test_shared_arith_over_typed_local_in_list(self):
        """Arithmetic over a typed static local inside a 1-element list
        emits the original expression against the emitted ``constexpr``."""
        cuda = _emit(_shared_arith_over_typed_local_in_list)
        self.assertIn("static constexpr int32_t TILE = 128;", cuda)
        self.assertIn("__shared__ __align__(16) __half xs[TILE * 4];", cuda)

    def test_local_literal_list_extent(self):
        """``alloc_local_memory`` accepts the same scalar container form."""
        cuda = _emit(_local_literal_list_extent)
        self.assertIn("int32_t buf[8];", cuda)

    def test_shared_multi_element_list_rejected(self):
        """A 2-element list is a shape, not a scalar extent: still rejected
        honestly as a non-constant scalar bound."""
        with self.assertRaisesRegex(ValueError, r"alloc_shared_memory.*non-constant element count"):
            _emit(_shared_multi_element_list_rejected)

    def test_shared_runtime_extent_rejected(self):
        """A container wrapping a runtime expression remains rejected."""
        with self.assertRaisesRegex(ValueError, r"alloc_shared_memory.*non-constant element count"):
            _emit(_shared_runtime_extent_rejected)


if __name__ == "__main__":
    unittest.main()
