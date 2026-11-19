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
"""Tests for per-buffer ll.empty(..., align=...) and align_memory legality."""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES


def _compile(func):
    kernel = lk.ll_kernel(backend="cuda", is_entry=True)(func)
    return kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=128)


class TestEmptyAlignDynamicShared(unittest.TestCase):
    """Per-buffer align pads the buffer's offset in the dynamic_shared pool."""

    def test_align_pads_dynamic_shared_offset(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            ll.align_memory(128, scope="dynamic_shared")
            a = ll.empty([10], dtype=ll.bfloat16, scope="dynamic_shared")
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared", align=16)
            tid: ll.int32 = ll.threadIdx_x()
            if tid < 10:
                a[tid] = x[tid]
            if tid < 16:
                o[tid] = ll.bf16_to_float(b[tid])

        cuda = _compile(k)
        # a: 10 * 2 = 20 bytes at offset 0; b is padded to offset 32 (not 20).
        self.assertIn("+ 32); /*size = 32 bytes*/", cuda)
        # total pool: 20 -> pad 32, + 16*2 = 64 bytes.
        self.assertIn("uint8_t __INTERNAL_DYN_SHMEM__[]", cuda)

    def test_total_pool_size_reflects_padding(self):
        import ast
        from little_kernel.core.passes import PASSES as _PASSES

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            ll.align_memory(128, scope="dynamic_shared")
            a = ll.empty([10], dtype=ll.bfloat16, scope="dynamic_shared")
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared", align=16)
            o[ll.threadIdx_x()] = ll.bf16_to_float(a[0]) + ll.bf16_to_float(b[0])

        kernel = lk.ll_kernel(backend="cuda", is_entry=True)(k)
        lowered = kernel.lower(_PASSES["cuda"])
        allocs = [
            call.args[0].value for call in ast.walk(lowered) if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name) and call.func.id == "alloc_dynamic_shared_memory"
        ]
        self.assertEqual(allocs, [64])

    def test_align_is_power_of_two(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared", align=12)
            o[ll.threadIdx_x()] = ll.bf16_to_float(b[0])

        with self.assertRaises(ValueError):
            _compile(k)

    def test_align_rejected_for_local_scope(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            b = ll.empty([16], dtype=ll.bfloat16, scope="local", align=16)
            o[ll.threadIdx_x()] = ll.bf16_to_float(b[0])

        with self.assertRaises(ValueError):
            _compile(k)

    def test_align_must_not_exceed_base_alignment(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            ll.align_memory(16, scope="dynamic_shared")
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared", align=128)
            o[ll.threadIdx_x()] = ll.bf16_to_float(b[0])

        with self.assertRaises(ValueError):
            _compile(k)

    def test_no_align_memory_decl_defaults_to_16(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared")
            o[ll.threadIdx_x()] = ll.bf16_to_float(b[0])

        cuda = _compile(k)
        self.assertIn("extern __shared__ __align__(16) uint8_t __INTERNAL_DYN_SHMEM__[]", cuda)


class TestAlignMemoryLegality(unittest.TestCase):
    """align_memory re-application rules."""

    def test_idempotent_reapplication_allowed(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            ll.align_memory(128, scope="dynamic_shared")
            a = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared")
            ll.align_memory(128, scope="dynamic_shared")
            o[ll.threadIdx_x()] = ll.bf16_to_float(a[0])

        _compile(k)  # must not raise

    def test_raise_before_any_allocation_allowed(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            ll.align_memory(16, scope="dynamic_shared")
            ll.align_memory(128, scope="dynamic_shared")
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared")
            o[ll.threadIdx_x()] = ll.bf16_to_float(b[0])

        cuda = _compile(k)
        self.assertIn("__align__(128) uint8_t __INTERNAL_DYN_SHMEM__[]", cuda)

    def test_change_after_allocation_rejected(self):

        def k(x: ll.Tensor[ll.bfloat16], o: ll.Tensor[ll.float32]) -> ll.void:
            ll.align_memory(16, scope="dynamic_shared")
            b = ll.empty([16], dtype=ll.bfloat16, scope="dynamic_shared")
            ll.align_memory(128, scope="dynamic_shared")
            o[ll.threadIdx_x()] = ll.bf16_to_float(b[0])

        with self.assertRaises(RuntimeError):
            _compile(k)

    def test_static_shared_decl_carries_per_buffer_align(self):

        def k(x: ll.Tensor[ll.float32], o: ll.Tensor[ll.float32]) -> ll.void:
            b = ll.empty([8], dtype=ll.float32, scope="shared", align=16)
            b[0] = x[ll.threadIdx_x()]
            o[ll.threadIdx_x()] = b[0]

        cuda = _compile(k)
        self.assertIn("__shared__ __align__(16) float", cuda)


if __name__ == "__main__":
    unittest.main()
