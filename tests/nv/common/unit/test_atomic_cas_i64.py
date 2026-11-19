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
"""CPU codegen regression for the 64-bit pointer/lvalue atomicCAS capability
(``backends/nv/common/language/intrin/atomic.py: atomic_cas_i64``).

Each case compiles a small typed-frontend kernel to CUDA *source text* via
``LLKernel.compile`` (no GPU, no nvcc, no launch ABI) and asserts on the
emitted code.  The capability is generic, not a task shape:

* a 64-bit atomicCAS on the address of a **shared-memory** tensor element
  lvalue (``Tensor[ll.uint64]`` from ``alloc_shared_memory``), legalized to
  PTX ``atom.cas.b64`` on the generic address.  CUDA's ``atomicCAS`` has no
  overload for ``long*`` (LP64 ``uint64_t`` is ``unsigned long``; the
  ``unsigned long long*`` overload is a distinct type), so the emitted
  ``&(smem[i])`` form would fail NVCC overload resolution without this
  legalization — the exact LP64 gap ``atomic_add_i64`` documents on the
  global side, here closed for the shared-element CAS path;
* the same intrinsic on a global ``Tensor[ll.uint64]`` element lvalue;
* typed legality: a 32-bit element lvalue is rejected at type inference,
  steering callers to the plain ``atomic_cas`` intrinsic.

The concurrent-CAS regression target is a per-block last-wins hash table
whose 64-bit entry packs (order-tag << 32 | payload bits) so claim + payload
commit is one CAS word instead of a serialized single-thread probe loop.

Run with:  uv run --locked --no-sync python -m unittest discover \\
    -s tests/nv/common/unit -p "test_atomic_cas_i64.py" -v
"""

import unittest

import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda


def _emit(kernel):
    return kernel.compile(None, codegen_cuda)


# --- kernels (traced once per compile call; no GPU involved) -----------------


@lk.ll_kernel(backend="cuda", is_entry=False)
def _smem_cas_i64_kernel(out: ll.Tensor[ll.uint32]) -> ll.void:
    ht: ll.Tensor[ll.uint64] = ll.alloc_shared_memory("ht", ll.uint64, 4, 8)
    t: ll.int32 = ll.threadIdx_x()
    if t == 0:
        ht[0] = 0
    ll.__syncthreads()
    old: ll.uint64 = ll.atomic_cas_i64(ht[0], 0, ll.uint64(t))
    if t == 0:
        out[0] = ll.val_cast(old, ll.uint32)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _smem_cas_generic_kernel(out: ll.Tensor[ll.uint32]) -> ll.void:
    # Same shape through the generic ll.atomic_cas entry point: a 64-bit
    # shared element lvalue must be legalized to the PTX b64 helper just the
    # same (type-driven dispatch in codegen_atomic_cas).
    ht: ll.Tensor[ll.uint64] = ll.alloc_shared_memory("ht", ll.uint64, 4, 8)
    t: ll.int32 = ll.threadIdx_x()
    if t == 0:
        ht[0] = 0
    ll.__syncthreads()
    old: ll.uint64 = ll.atomic_cas(ht[0], 0, ll.uint64(t))
    if t == 0:
        out[0] = ll.val_cast(old, ll.uint32)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _smem_cas_generic_u32_kernel(out: ll.Tensor[ll.uint32]) -> ll.void:
    # A 32-bit element lvalue through ll.atomic_cas keeps the CUDA
    # atomicCAS path (no helper body, no PTX legalization).
    smem: ll.Tensor[ll.uint32] = ll.alloc_shared_memory("smem", ll.uint32, 4, 4)
    t: ll.int32 = ll.threadIdx_x()
    if t == 0:
        smem[0] = 0
    ll.__syncthreads()
    old: ll.uint32 = ll.atomic_cas(smem[0], 0, ll.uint32(t))
    if t == 0:
        out[0] = old


@lk.ll_kernel(backend="cuda", is_entry=False)
def _smem_last_wins_kernel(key: ll.Tensor[ll.uint32], val: ll.Tensor[ll.uint32]) -> ll.void:
    # A slot-claim CAS over a packed u64 entry: high32 = claimant order/tag,
    # low32 = payload bits.  Every thread concurrently CASes its probe slot;
    # the last successful CAS wins the whole 64-bit entry atomically, so no
    # serialized probe loop is needed.
    ht: ll.Tensor[ll.uint64] = ll.alloc_shared_memory("ht_pack", ll.uint64, 16, 8)
    t: ll.int32 = ll.threadIdx_x()
    if t < 16:
        ht[t] = 0
    ll.__syncthreads()
    k: ll.uint32 = key[t] % 16
    entry: ll.uint64 = ll.uint64(t)
    packed: ll.uint64 = (entry << 32) | ll.uint64(val[t])
    old: ll.uint64 = ll.atomic_cas_i64(ht[k], 0, packed)  # noqa: F841


@lk.ll_kernel(backend="cuda", is_entry=False)
def _gmem_cas_i64_kernel(ht: ll.Tensor[ll.uint64], out: ll.Tensor[ll.uint64]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    old: ll.uint64 = ll.atomic_cas_i64(ht[t % 4], 0, ll.uint64(t))
    out[t] = old


@lk.ll_kernel(backend="cuda", is_entry=False)
def _gmem_int64_cas_kernel(ht: ll.Tensor[ll.int64], out: ll.Tensor[ll.int64]) -> ll.void:
    t: ll.int32 = ll.threadIdx_x()
    old: ll.int64 = ll.atomic_cas_i64(ht[t % 4], 0, ll.int64(t))
    out[t] = old


# --- checks ------------------------------------------------------------------


class TestAtomicCasSharedU64Generic(unittest.TestCase):
    """The capability-spec reproducer: generic ll.atomic_cas on a u64
    shared-memory element lvalue legalizes to atom.cas.b64."""

    def test_u64_shared_lvalue_legalized_via_generic_atomic_cas(self):
        src = _emit(_smem_cas_generic_kernel)
        self.assertIn("atom.cas.b64", src, src)
        self.assertIn("lk_atomic_cas_i64_fn(&(ht[0])", src.replace(" ", ""), src)

    def test_u16_shared_lvalue_keeps_cuda_atomicCAS(self):
        # 16-bit element lvalues type-check at their own element type (the
        # pre-change int32 fallback would fail the ll.uint16 assignment at
        # trace) and keep CUDA atomicCAS — its `unsigned short` overload is
        # real, unlike the LP64 `long*` gap the u64 legalization closes.
        src = _emit(_smem_cas_generic_u16_kernel)
        self.assertNotIn("atomic_cas_i64_fn", src, src)
        self.assertIn("atomicCAS(&(smem[0])", src.replace(" ", ""), src)

    def test_u32_shared_lvalue_keeps_cuda_atomicCAS(self):
        src = _emit(_smem_cas_generic_u32_kernel)
        self.assertNotIn("atomic_cas_i64_fn", src, src)
        self.assertIn("atomicCAS(&(smem[0])", src.replace(" ", ""), src)


class TestAtomicCasI64Shared(unittest.TestCase):
    """The added capability: 64-bit CAS on a shared-memory element lvalue."""

    def test_declared_memory_effects(self):
        # An atomic must never look side-effect free to effect-driven
        # analyses; the intrinsic declares RW_GMEM_ATOMIC_SYNC on "addr".
        from little_kernel.language import builtin_base as bb
        from little_kernel.core.memory_effects import RW_GMEM_ATOMIC_SYNC
        from little_kernel.backends.nv.common.language.intrin.atomic import (
            atomic_cas_i64, )
        effects = getattr(atomic_cas_i64, bb.MEMORY_EFFECTS_ATTR)
        self.assertTrue(any(e.kind is RW_GMEM_ATOMIC_SYNC for e in effects))

    def test_emits_ptx_cas_b64_helper(self):
        src = _emit(_smem_cas_i64_kernel)
        # Legalization is the PTX b64 form, not a CUDA atomicCAS overload
        # that does not exist for LP64 uint64_t* (= unsigned long*).
        self.assertIn("atom.cas.b64", src, src)
        self.assertIn("lk_atomic_cas_i64_fn", src, src)

    def test_shared_lvalue_address_is_taken(self):
        src = _emit(_smem_cas_i64_kernel)
        # Element-typed subscript operand -> &(ht[0]); a bare "ht[0]" rvalue
        # here would be an integer, not a pointer destination.
        self.assertIn("lk_atomic_cas_i64_fn(&(ht[0])", src.replace(" ", ""), src)

    def test_packed_last_wins_cas(self):
        # The motivating concurrent shape: many threads CAS packed 64-bit
        # entries into a shared hash table with no probe-loop serialization.
        src = _emit(_smem_last_wins_kernel)
        self.assertIn("atom.cas.b64", src, src)
        stripped = src.replace(" ", "")
        self.assertIn("lk_atomic_cas_i64_fn(&(ht[k])", stripped, src)
        # Entry construction is the packed form: order-tag << 32 | payload.
        self.assertIn("entry<<32", stripped, src)


@lk.ll_kernel(backend="cuda", is_entry=False)
def _smem_cas_generic_u16_kernel(out: ll.Tensor[ll.uint32]) -> ll.void:
    # A 16-bit element lvalue through ll.atomic_cas infers the 16-bit element
    # type and keeps the CUDA atomicCAS path (its `unsigned short` overload
    # exists, so no PTX legalization is needed).
    smem: ll.Tensor[ll.uint16] = ll.alloc_shared_memory("smem", ll.uint16, 4, 2)
    t: ll.int32 = ll.threadIdx_x()
    if t == 0:
        smem[0] = 0
    ll.__syncthreads()
    old: ll.uint16 = ll.atomic_cas(smem[0], 0, ll.uint16(t))
    if t == 0:
        out[0] = ll.val_cast(old, ll.uint32)


class TestAtomicCasI64Global(unittest.TestCase):

    def test_global_u64_element_lvalue(self):
        src = _emit(_gmem_cas_i64_kernel)
        self.assertIn("atom.cas.b64", src, src)
        self.assertIn("lk_atomic_cas_i64_fn(&(ht[t%4])", src.replace(" ", ""), src)

    def test_signed_int64_element_lvalue_uses_same_ptx_form(self):
        # Two's-complement compare-exchange is bit-identical signed/unsigned;
        # the int64 overload exists so a Tensor[ll.int64] lvalue type-checks.
        src = _emit(_gmem_int64_cas_kernel)
        self.assertIn("atom.cas.b64", src, src)
        self.assertIn("lk_atomic_cas_i64_fn(&(ht[t%4])", src.replace(" ", ""), src)


class TestAtomicCasI64Legality(unittest.TestCase):

    def test_signed_int16_lvalue_legalized_to_ptx_b16(self):
        # Legality regression (review finding da89b495): CUDA atomicCAS has
        # no signed short* overload (int*/uint*/unsigned short*/unsigned
        # long long* only) and pointer types never convert implicitly, so an
        # atomic_cas on a Tensor[ll.int16] lvalue must not reach codegen as
        # the overloadless `atomicCAS(&(smem[0]), ...)`; it is legalized to
        # PTX atom.cas.b16 on the generic address, mirroring the int64 path.
        @lk.ll_kernel(backend="cuda", is_entry=False)
        def _i16(out: ll.Tensor[ll.int16]) -> ll.void:
            smem: ll.Tensor[ll.int16] = ll.alloc_shared_memory("smem", ll.int16, 4, 2)
            smem[0] = 0
            ll.__syncthreads()
            old: ll.int16 = ll.atomic_cas(smem[0], 0, ll.int16(1))
            out[0] = old

        src = _emit(_i16)
        self.assertIn("atom.cas.b16", src, src)
        self.assertIn("lk_atomic_cas_i16_fn(&(smem[0])", src.replace(" ", ""), src)
        self.assertNotIn("atomicCAS(&(smem[0])", src.replace(" ", ""), src)

    def test_fp32_lvalue_legalized_to_ptx_b32(self):
        # Legality regression (review finding 466b43c5): CUDA atomicCAS has
        # no float* overload (int*/uint*/unsigned short*/unsigned long long*
        # only) and pointer types never convert implicitly, so an atomic_cas
        # on a Tensor[ll.float32] element lvalue must not reach codegen as
        # the overloadless `atomicCAS(&(smem[0]), ...)`; it is legalized to
        # PTX atom.cas.b32 on the generic address (fp32 bit-pattern equality
        # is the natural semantics for fp32 compare-exchange).
        @lk.ll_kernel(backend="cuda", is_entry=False)
        def _f32(out: ll.Tensor[ll.float32]) -> ll.void:
            smem: ll.Tensor[ll.float32] = ll.alloc_shared_memory("smem", ll.float32, 4, 4)
            smem[0] = 0.0
            ll.__syncthreads()
            old: ll.float32 = ll.atomic_cas(smem[0], 0.0, ll.float32(1.0))
            out[0] = old

        src = _emit(_f32)
        self.assertIn("atom.cas.b32", src, src)
        self.assertIn("lk_atomic_cas_f32_fn(&(smem[0])", src.replace(" ", ""), src)
        self.assertNotIn("atomicCAS(&(smem[0])", src.replace(" ", ""), src)

    def test_rejects_fp32_element_lvalue(self):
        # Invalid-type boundary: the 64-bit intrinsic must not silently
        # truncate or reinterpret an fp32 element lvalue; it raises at trace
        # time with the same steering message as the 32-bit integer case.
        @lk.ll_kernel(backend="cuda", is_entry=False)
        def _bad_f32(out: ll.Tensor[ll.float32]) -> ll.void:
            smem: ll.Tensor[ll.float32] = ll.alloc_shared_memory("smem", ll.float32, 4, 4)
            smem[0] = 0.0
            ll.__syncthreads()
            old: ll.float32 = ll.atomic_cas_i64(smem[0], 0.0, 1.0)  # noqa: F841

        with self.assertRaises(Exception) as ctx:
            _emit(_bad_f32)
        self.assertIn("atomic_cas_i64 requires an int64 or uint64", str(ctx.exception))

    def test_rejects_32bit_element_lvalue(self):

        @lk.ll_kernel(backend="cuda", is_entry=False)
        def _bad(out: ll.Tensor[ll.int32]) -> ll.void:
            smem: ll.Tensor[ll.int32] = ll.alloc_shared_memory("smem", ll.int32, 4, 4)
            smem[0] = 0
            ll.__syncthreads()
            old: ll.int32 = ll.atomic_cas_i64(smem[0], 0, 1)  # noqa: F841

        with self.assertRaises(Exception) as ctx:
            _emit(_bad)
        self.assertIn("atomic_cas_i64 requires an int64 or uint64", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
