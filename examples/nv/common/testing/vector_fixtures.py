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
"""Shared executable minimal-DSL fixtures for the 16-byte global vector family.

The reviewed fixtures (``tests/nv/common/unit/test_gmem_u8x16.py`` and
``tests/nv/common/unit/test_gmem_bf16x8.py``) set the loop bound *element* count
(``nvec = x[0]``) while multiplying each lane's offset by 16 / 8, so a real
launch with one thread per element would multiply ``off`` up to
``16 * nelement`` and run far out of bounds.  These fixtures are the
correct replacement and are executable as real LittleKernel programs:

``u8x16`` covers the opaque byte pair (16 bytes in two ``uint64`` register
halves; no numeric conversion); ``bf16x8`` covers the typed lane family
(eight BF16 lanes widened to FP32 and packed back with RNE).  Both families
share one geometry contract:

* **Vector count, not element count.**  ``nvec = ceil_div(nelem, lanes)`` is
  the number of 16-byte vectors; the per-lane offset is ``tid * lanes``.
* **Tails are explicit.**  Vectors that are not fully backed by ``nelem``
  lanes never issue a 16-byte vector access.  The last partial vector is
  handled either by a guarded scalar tail (``masked`` tail policy) or by a
  rounded allocation (``padded`` tail policy).  There is no implicit
  fallback and no out-of-bounds vector transaction.
* **One vector / several vectors / tail** all have checked bounds:
  ``(nvec - 1) * lanes + lanes <= bytes - elems_before`` for every vector
  access a thread can issue.

The same kernels are consumed by the CPU codegen tests, the NVCC stage and
the GPU semantic tests in ``tests/nv/common/refactor/test_rf12_vector_fixtures.py``;
they are deliberately written with the public compile API
(``lk.ll_kernel`` + ``PASSES['cuda']`` + ``codegen_cuda``) instead of the
private evaluator parser/renderer helpers.

Hardware gates
--------------
* CPU codegen stage: always runs (no torch, no GPU).
* NVCC stage: runs when ``nvcc`` is on ``PATH``; it cross-compiles PTX for
  ``sm_90a`` and never launches anything.
* GPU semantic stage: requires a real CUDA GPU with compute capability
  >= 9.0; it is skipped (not silently passed) otherwise.  The packed
  bf16x8 widening additionally requires SM90-class hardware for the
  ``ld.global.nc/cs.v4`` policy paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import little_kernel as lk
import little_kernel.language as ll

# Lane counts (elements per 16-byte vector) for the two families.
U8X16_LANES = 16
BF16X8_LANES = 8

# Thread block width the fixtures launch with.  Small on purpose: these are
# semantic minimal programs, not performance kernels.
THREADS_PER_VECTOR_CTA = 32


def ceil_div(value: int, divisor: int) -> int:
    """Integer ceiling, shared by host geometry checks and docs."""
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class VectorGeometry:
    """Checked, launch-safe geometry for one vector fixture case.

    Attributes
    ----------
    nelem:
        Number of *elements* logically present in the source/destination.
    lanes:
        Elements per 16-byte vector (16 for u8x16, 8 for bf16x8).
    tail_policy:
        ``"padded"`` allocates a multiple of ``lanes`` (so every vector is
        fully backed); ``"masked"`` issues scalar stores for the final
        partial vector and never lets it become a vector transaction.
    """

    nelem: int
    lanes: int
    tail_policy: str = "padded"

    def __post_init__(self) -> None:
        if self.lanes not in (U8X16_LANES, BF16X8_LANES):
            raise ValueError(f"unsupported lanes: {self.lanes}")
        if self.tail_policy not in ("padded", "masked"):
            raise ValueError(f"unsupported tail policy: {self.tail_policy!r}")
        if self.nelem <= 0:
            raise ValueError("nelem must be positive")
        if self.tail_policy == "masked" and self.nelem % self.lanes == 0:
            # A masked policy with no ragged tail is legal but pointless;
            # reject it so the tail case actually exercises a tail.
            raise ValueError("masked tail policy requires a ragged nelem")
        if self.tail_policy == "padded" and self.nelem % self.lanes != 0:
            raise ValueError("padded tail policy requires nelem % lanes == 0")

    @property
    def nvec(self) -> int:
        """Number of 16-byte vectors that exist (full + partial)."""
        return ceil_div(self.nelem, self.lanes)

    @property
    def allocated_elems(self) -> int:
        """Elements actually allocated so every issued access stays in bounds."""
        if self.tail_policy == "padded":
            return self.nvec * self.lanes
        return self.nelem

    def vector_bounds_are_safe(self) -> bool:
        """True iff every *full-vector* lane interval lies in the allocation."""
        full_vectors = self.nvec if self.tail_policy == "padded" else self.nvec - 1
        if full_vectors <= 0:
            return True
        last_vector_end = (full_vectors - 1) * self.lanes + self.lanes
        return last_vector_end <= self.allocated_elems


# Canonical geometry cases: exactly one vector, several vectors, and a tail.
U8X16_CASES = (
    VectorGeometry(16, U8X16_LANES),  # one vector
    VectorGeometry(32 * 16, U8X16_LANES),  # several vectors
    VectorGeometry(32 * 16 - 7, U8X16_LANES, "masked"),  # ragged tail
)
BF16X8_CASES = (
    VectorGeometry(8, BF16X8_LANES),
    VectorGeometry(32 * 8, BF16X8_LANES),
    VectorGeometry(32 * 8 - 3, BF16X8_LANES, "masked"),
)

# ---------------------------------------------------------------------------
# u8x16: opaque 16-byte pair, default / .cg / .streaming(.cs) / .nc policies
# ---------------------------------------------------------------------------


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_u8x16_policies(
    x: ll.Tensor[ll.uint8],
    dst: ll.Tensor[ll.uint8],
    dst_cg: ll.Tensor[ll.uint8],
    dst_cs: ll.Tensor[ll.uint8],
    nvec: ll.int32,
) -> ll.void:
    """One thread copies one 16-byte vector through each cache policy.

    ``nvec`` is a *vector* count; ``off = tid * 16`` stays in bounds because
    the host allocates ``nvec * 16`` bytes (padded case) or supplies the
    guarded-tail kernel for the ragged case.
    """
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 16
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        lo_cg: ll.uint64 = 0
        hi_cg: ll.uint64 = 0
        lo_cs: ll.uint64 = 0
        hi_cs: ll.uint64 = 0
        ll.gmem_ld_u8x16(x, off, lo, hi)
        ll.gmem_ld_u8x16_cg(x, off, lo_cg, hi_cg)
        ll.gmem_ld_u8x16_streaming(x, off, lo_cs, hi_cs)
        ll.gmem_st_u8x16(dst, off, lo, hi)
        ll.gmem_st_u8x16_cg(dst_cg, off, lo_cg, hi_cg)
        ll.gmem_st_u8x16_streaming(dst_cs, off, lo_cs, hi_cs)


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_u8x16_nc(
    x: ll.Tensor[ll.uint8],
    dst_nc: ll.Tensor[ll.uint8],
    nvec: ll.int32,
) -> ll.void:
    """Read-only non-coherent (.nc) 16-byte copy; ``x`` is never written."""
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 16
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        ll.gmem_ld_u8x16_nc(x, off, lo, hi)
        ll.gmem_st_u8x16(dst_nc, off, lo, hi)


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_u8x16_masked_tail(
    x: ll.Tensor[ll.uint8],
    dst: ll.Tensor[ll.uint8],
    nvec: ll.int32,
    nelem: ll.int32,
) -> ll.void:
    """Vector copy for full vectors; guarded scalar tail for the last partial.

    The final partial vector (``tid == nvec - 1``) never issues a 16-byte
    vector load/store: each lane is bounds-checked against ``nelem``.
    """
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 16
        if tid + 1 < nvec:
            lo: ll.uint64 = 0
            hi: ll.uint64 = 0
            ll.gmem_ld_u8x16(x, off, lo, hi)
            ll.gmem_st_u8x16(dst, off, lo, hi)
        else:
            k: ll.int32 = 0
            while k < 16:
                idx: ll.int32 = off + k
                if idx < nelem:
                    dst[idx] = x[idx]
                k = k + 1


@lk.ll_kernel(backend="cuda", is_entry=True)
def u8x16_lane_canary(
    x: ll.Tensor[ll.uint8],
    lane_order: ll.Tensor[ll.uint8],
    nvec: ll.int32,
) -> ll.void:
    """Lane-order canary: extracted bytes are written one-per-slot in order.

    Every lane 0..15 is extracted with a compile-time literal (the register
    pair has no addressable storage, so a dynamic lane is illegal) and written
    to its own output slot ``lane_order[off + k]``.  This oracle is
    **order-sensitive**: a bit-exact vector copy cannot detect a swapped half
    or a reversed/permuted lane, and neither can a sum (addition is
    commutative).  Comparing the full 16-byte output sequence against the
    literal memory order 0,1,...,15 catches any such permutation.
    """
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 16
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        ll.gmem_ld_u8x16(x, off, lo, hi)
        b0: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 0)
        b1: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 1)
        b2: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 2)
        b3: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 3)
        b4: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 4)
        b5: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 5)
        b6: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 6)
        b7: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 7)
        b8: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 8)
        b9: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 9)
        b10: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 10)
        b11: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 11)
        b12: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 12)
        b13: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 13)
        b14: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 14)
        b15: ll.uint32 = ll.vec_u8x16_lane(lo, hi, 15)
        lane_order[off + 0] = ll.uint8(b0)
        lane_order[off + 1] = ll.uint8(b1)
        lane_order[off + 2] = ll.uint8(b2)
        lane_order[off + 3] = ll.uint8(b3)
        lane_order[off + 4] = ll.uint8(b4)
        lane_order[off + 5] = ll.uint8(b5)
        lane_order[off + 6] = ll.uint8(b6)
        lane_order[off + 7] = ll.uint8(b7)
        lane_order[off + 8] = ll.uint8(b8)
        lane_order[off + 9] = ll.uint8(b9)
        lane_order[off + 10] = ll.uint8(b10)
        lane_order[off + 11] = ll.uint8(b11)
        lane_order[off + 12] = ll.uint8(b12)
        lane_order[off + 13] = ll.uint8(b13)
        lane_order[off + 14] = ll.uint8(b14)
        lane_order[off + 15] = ll.uint8(b15)


# ---------------------------------------------------------------------------
# bf16x8: typed eight-lane family, opaque load + lane widen + RNE pack store
# ---------------------------------------------------------------------------


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_bf16x8_roundtrip(
    x: ll.Tensor[ll.bfloat16],
    dst: ll.Tensor[ll.bfloat16],
    dst_cg: ll.Tensor[ll.bfloat16],
    dst_cs: ll.Tensor[ll.bfloat16],
    nvec: ll.int32,
) -> ll.void:
    """BF16 bit-exact round trip through default/.cg/.cs stores.

    The opaque load moves 16 raw bytes; ``vec_bf16x8_lane`` widens each lane
    to FP32 (the documented representation), and ``gmem_st_bf16x8*`` packs
    them back to BF16 with round-to-nearest-even.  Inputs already in BF16
    must come back bit-identical under every policy.
    """
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 8
        lo: ll.uint64 = 0
        hi: ll.uint64 = 0
        ll.gmem_ld_bf16x8(x, off, lo, hi)
        f0: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 0)
        f1: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 1)
        f2: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 2)
        f3: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 3)
        f4: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 4)
        f5: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 5)
        f6: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 6)
        f7: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 7)
        ll.gmem_st_bf16x8(dst, off, f0, f1, f2, f3, f4, f5, f6, f7)
        ll.gmem_st_bf16x8_cg(dst_cg, off, f0, f1, f2, f3, f4, f5, f6, f7)
        ll.gmem_st_bf16x8_streaming(dst_cs, off, f0, f1, f2, f3, f4, f5, f6, f7)


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_bf16x8_masked_tail(
    x: ll.Tensor[ll.bfloat16],
    dst: ll.Tensor[ll.bfloat16],
    nvec: ll.int32,
    nelem: ll.int32,
) -> ll.void:
    """Vector RNE copy for full vectors; guarded scalar BF16 tail."""
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 8
        if tid + 1 < nvec:
            lo: ll.uint64 = 0
            hi: ll.uint64 = 0
            ll.gmem_ld_bf16x8(x, off, lo, hi)
            f0: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 0)
            f1: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 1)
            f2: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 2)
            f3: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 3)
            f4: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 4)
            f5: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 5)
            f6: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 6)
            f7: ll.float32 = ll.vec_bf16x8_lane(lo, hi, 7)
            ll.gmem_st_bf16x8(dst, off, f0, f1, f2, f3, f4, f5, f6, f7)
        else:
            k: ll.int32 = 0
            while k < 8:
                idx: ll.int32 = off + k
                if idx < nelem:
                    v: ll.float32 = x[idx]
                    ll.st_global_cs_bf16_f32_rn(dst, idx, v)
                k = k + 1


@lk.ll_kernel(backend="cuda", is_entry=True)
def copy_bf16x8_f32x8_policy(
    x: ll.Tensor[ll.bfloat16],
    dst_nc: ll.Tensor[ll.bfloat16],
    dst_cs: ll.Tensor[ll.bfloat16],
    nvec: ll.int32,
) -> ll.void:
    """FP32-lane widening loads under .nc / .cs, packed back via streaming store.

    Exercises the policy variants of the widening family (``__ldg`` /
    ``__ldcs`` on the 16-byte vector pointer) and proves policy never alters
    widened lane values.
    """
    tid: ll.int32 = ll.threadIdx_x()
    if tid < nvec:
        off: ll.int32 = tid * 8
        a0: ll.float32 = 0.0
        a1: ll.float32 = 0.0
        a2: ll.float32 = 0.0
        a3: ll.float32 = 0.0
        a4: ll.float32 = 0.0
        a5: ll.float32 = 0.0
        a6: ll.float32 = 0.0
        a7: ll.float32 = 0.0
        b0: ll.float32 = 0.0
        b1: ll.float32 = 0.0
        b2: ll.float32 = 0.0
        b3: ll.float32 = 0.0
        b4: ll.float32 = 0.0
        b5: ll.float32 = 0.0
        b6: ll.float32 = 0.0
        b7: ll.float32 = 0.0
        ll.gmem_ld_bf16x8_f32x8_nc(x, off, a0, a1, a2, a3, a4, a5, a6, a7)
        ll.gmem_ld_bf16x8_f32x8_cs(x, off, b0, b1, b2, b3, b4, b5, b6, b7)
        ll.gmem_st_bf16x8_streaming(dst_nc, off, a0, a1, a2, a3, a4, a5, a6, a7)
        ll.gmem_st_bf16x8_streaming(dst_cs, off, b0, b1, b2, b3, b4, b5, b6, b7)
