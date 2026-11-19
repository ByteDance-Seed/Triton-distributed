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
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""CPU regression tests for the byte-addressed mbarrier idiom
(ir.smem_mbarrier.byte_addressed_idiom).

Every Hopper TMA-producer/WGMMA-consumer ring forms mbarrier instructions
on a *uint32 shared-window byte address*: PTX ``mbarrier.*.shared.b64 [%0]``
takes a ``.shared`` state-space address directly.  Two typed slot idioms
must lower cleanly:

  1. shared tensor pointer (legacy):
       bar = ll.empty([1], ll.uint64, scope="shared")
       ll.init_smem_barrier(bar, 1)
     The helper takes ``uint64_t*`` and runs __cvta_generic_to_shared on
     ``&bar[0]`` itself.

  2. uint32 byte address (canonical PTX form):
       ll.alloc_shared_memory("s_bar", ll.uint64, 1, 8)
       addr: ll.uint32 = ll.cvta_generic_to_shared(s_bar)
       ll.init_smem_barrier(addr, 1)
     The helper takes ``uint32_t`` and uses it directly in [%0]; it must
     NOT apply __cvta_generic_to_shared a second time (an address already
     in the .shared window re-run through cvta is invalid PTX).

Before typed dispatch, form 2 emitted ``..._fn(addr, 1)`` against a
``uint64_t*`` parameter followed by a second cvta -- invalid C++/PTX that
failed late, and a scalar load of the slot contents (uint64) was accepted
silently.  These tests pin lowering of one full init -> expect_tx -> wait
parity -> arrive launch, with no GPU.
"""

import unittest

import little_kernel.language as ll
from little_kernel.core.compile import ll_kernel
from little_kernel.core.passes import PASSES
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError


def _compile(fn):
    return fn.compile(PASSES["cuda"], codegen_cuda)


def _byte_address_kernel():

    @ll_kernel(backend="cuda", is_entry=True)
    def mbar_byte_addr(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
        ll.alloc_shared_memory("s_bar", ll.uint64, 1, 8)
        bar: ll.uint32 = ll.cvta_generic_to_shared(s_bar)  # noqa: F821 - bound by alloc_shared_memory during lowering
        if ll.threadIdx_x() == 0:
            ll.init_smem_barrier(bar, 1)
        ll.__syncthreads()
        phase: ll.int32 = 0
        ll.mbarrier_arrive_and_expect_tx(bar, 256)
        ll.mbarrier_wait(bar, phase)
        ready = ll.mbarrier_try_wait_parity(bar, phase ^ 1)
        ll.mbarrier_wait_if_needed(bar, phase ^ 1, ready)
        ll.mbarrier_arrive(bar)
        i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
        if i < n:
            y[i] = x[i]

    return mbar_byte_addr


def _pointer_kernel():

    @ll_kernel(backend="cuda", is_entry=True)
    def mbar_pointer(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
        bar = ll.empty([1], dtype=ll.uint64, scope="shared")
        if ll.threadIdx_x() == 0:
            ll.init_smem_barrier(bar, 1)
        ll.__syncthreads()
        ll.mbarrier_arrive_and_expect_tx(bar, 256)
        ll.mbarrier_wait(bar, 0)
        ll.mbarrier_arrive(bar)
        i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
        if i < n:
            y[i] = x[i]

    return mbar_pointer


class TestMbarrierByteAddressIdiom(unittest.TestCase):

    def test_alloc_shared_memory_barename_slot_lowers(self):
        """The originally reported form: named alloc + bare slot name."""

        @ll_kernel(backend="cuda", is_entry=True)
        def probe(x: ll.Tensor[ll.float32], y: ll.Tensor[ll.float32], n: ll.int32) -> ll.void:
            ll.alloc_shared_memory("s_bar", ll.uint64, 2, 8)
            ll.alloc_shared_memory("s_i", ll.int32, 1, 4)
            if ll.threadIdx_x() == 0:
                ll.init_smem_barrier(s_bar, 1)  # noqa: F821 - DSL allocation binds this symbol
                s_i[0] = 0  # noqa: F821 - DSL allocation binds this symbol
            ll.__syncthreads()
            i: ll.int32 = ll.blockIdx_x() * 256 + ll.threadIdx_x()
            if i < n:
                y[i] = x[i]

        code = _compile(probe)
        self.assertIn("__shared__ __align__(8) uint64_t s_bar[2];", code)
        self.assertIn("init_smem_barrier_fn(s_bar, 1);", code)

    def test_uint32_address_form_uses_direct_ptx_address(self):
        code = _compile(_byte_address_kernel())
        # Slot declaration and exactly one cvta, at address formation.
        self.assertIn("uint32_t bar = (uint32_t)__cvta_generic_to_shared(s_bar);", code)
        # Address-form helpers take uint32 and call sites use them.
        for call in ("init_smem_barrier_addr_fn(bar, 1);", "mbarrier_arrive_and_expect_tx_addr_fn(bar, 256);",
                     "mbarrier_wait_addr_fn(bar, phase);", "mbarrier_try_wait_parity_addr_fn(bar, phase ^ 1);",
                     "mbarrier_wait_if_needed_addr_fn(bar, phase ^ 1, ready);", "mbarrier_arrive_addr_fn(bar);"):
            with self.subTest(call=call):
                self.assertIn(call, code)
        for sig in ("void init_smem_barrier_addr_fn(uint32_t bar, uint32_t count)",
                    "void mbarrier_arrive_and_expect_tx_addr_fn(uint32_t bar, uint32_t tx_bytes)",
                    "void mbarrier_wait_addr_fn(uint32_t smem_addr, uint32_t phase)",
                    "bool mbarrier_try_wait_parity_addr_fn(uint32_t smem_addr, uint32_t phase)",
                    "void mbarrier_arrive_addr_fn(uint32_t bar)"):
            with self.subTest(sig=sig):
                self.assertIn(sig, code)
        # The whole point of the address form: no cvta inside ANY address
        # helper (the incoming value is already a .shared address).
        addr_helper_section = code.split("init_smem_barrier_addr_fn", 1)[1].split('extern "C" __global__', 1)[0]
        self.assertNotIn("__cvta_generic_to_shared", addr_helper_section)

    def test_pointer_form_is_unchanged(self):
        code = _compile(_pointer_kernel())
        # Legacy pointer helpers still derive the address from &bar[0].
        self.assertIn("void init_smem_barrier_fn(uint64_t* bar, uint32_t count)", code)
        self.assertIn('"r"((uint32_t)__cvta_generic_to_shared(&bar[0]))', code)
        self.assertIn("init_smem_barrier_fn(bar, 1);", code)
        self.assertIn("mbarrier_arrive_and_expect_tx_fn(bar, 256);", code)
        self.assertIn("mbarrier_wait_fn(bar, 0);", code)
        self.assertIn("mbarrier_arrive_fn(bar);", code)
        # No address-form spelling leaks into pointer-only kernels.
        self.assertNotIn("_addr_fn", code)

    def test_both_forms_coexist_with_distinct_helpers(self):

        @ll_kernel(backend="cuda", is_entry=True)
        def both(x: ll.Tensor[ll.float32]) -> ll.void:
            ll.alloc_shared_memory("s_a", ll.uint64, 1, 8)
            a: ll.uint32 = ll.cvta_generic_to_shared(s_a)  # noqa: F821 - DSL allocation binds this symbol
            b = ll.empty([1], dtype=ll.uint64, scope="shared")
            ll.init_smem_barrier(a, 1)
            ll.init_smem_barrier(b, 1)
            ll.mbarrier_arrive(a)
            ll.mbarrier_arrive(b)

        code = _compile(both)
        self.assertIn("init_smem_barrier_addr_fn(a, 1);", code)
        self.assertIn("init_smem_barrier_fn(b, 1);", code)
        self.assertIn("mbarrier_arrive_addr_fn(a);", code)
        self.assertIn("mbarrier_arrive_fn(b);", code)

    def test_scalar_slot_contents_rejected(self):

        @ll_kernel(backend="cuda", is_entry=True)
        def bad() -> ll.void:
            ll.alloc_shared_memory("s_bar", ll.uint64, 1, 8)
            v: ll.uint64 = s_bar[0]  # noqa: F821 - DSL allocation binds this symbol
            ll.init_smem_barrier(v, 1)

        with self.assertRaises(TypeInferenceError) as cm:
            _compile(bad)
        msg = str(cm.exception)
        self.assertIn("init_smem_barrier", msg)
        # Diagnosis must name both canonical idioms.
        self.assertIn("cvta_generic_to_shared", msg)
        self.assertIn("uint32", msg)

    def test_indexed_barrier_ring_keeps_pointer_semantics(self):

        @ll_kernel(backend="cuda", is_entry=True)
        def ring(stage: ll.int32) -> ll.void:
            barriers = ll.empty([2], dtype=ll.uint64, scope="shared")
            ll.init_smem_barrier(barriers + stage, 1)
            ll.mbarrier_arrive(barriers + stage)
            ll.mbarrier_wait(barriers + stage, 0)

        code = _compile(ring)
        self.assertIn("init_smem_barrier_fn(barriers + stage, 1)", code)
        self.assertIn("mbarrier_arrive_fn(barriers + stage)", code)

    def test_non_uint32_scalar_address_rejected(self):
        # An int32 address is not a .shared u32 address and is not a slot
        # pointer either; reject instead of emitting an implicit cast.
        @ll_kernel(backend="cuda", is_entry=True)
        def bad(x: ll.Tensor[ll.int32]) -> ll.void:
            a: ll.int32 = x[0]
            ll.mbarrier_wait(a, 0)

        with self.assertRaises(TypeInferenceError) as cm:
            _compile(bad)
        self.assertIn("mbarrier_wait", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
