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
"""Targeted coverage for :mod:`triton_dist.language.smem_ops` (memdesc, subview, indexed u32,
dealloc, multi-thread scatter/gather, validation)."""
from __future__ import annotations

import pytest
import torch
import triton.language as tl
import triton_dist

from triton_dist.language.smem_ops import (
    SharedMemDescType,
    allocate_smem,
    get_smem_shared_address_u32,
    smem_dealloc,
    smem_get_ptr,
    smem_index,
    smem_load,
    smem_store,
)
from triton_dist.language.extra.cuda.language_extra import __syncthreads, st, tid


@triton_dist.jit
def kernel_smem_index_row(out_row, R: tl.constexpr, C: tl.constexpr, pick: tl.constexpr):
    smem = allocate_smem(tl.int32, [R, C])
    row_desc = smem_index(smem, pick)
    if tid(0) == 0:
        for c in tl.static_range(C):
            smem_store(row_desc, pick * 1000 + c, [c])
    __syncthreads()
    _t = tid(0)
    if _t < C:
        v = smem_load(row_desc, [_t])
        st(out_row + _t, v)


def test_smem_index_subview_row():
    R, C, pick = 4, 8, 2
    out = torch.zeros(C, dtype=torch.int32, device="cuda")
    kernel_smem_index_row[(1, )](out, R, C, pick, num_warps=1)
    torch.cuda.synchronize()
    exp = torch.tensor([pick * 1000 + c for c in range(C)], dtype=torch.int32, device="cuda")
    assert torch.equal(out, exp), "smem_index row subview load/store mismatch"


@triton_dist.jit
def kernel_smem_index_row_swizzled(out_row, R: tl.constexpr, C: tl.constexpr, pick: tl.constexpr):
    smem = allocate_smem(tl.int32, [R, C], vec=2, per_phase=2, max_phase=2)
    row_desc = smem_index(smem, pick)
    if tid(0) == 0:
        for c in tl.static_range(C):
            smem_store(row_desc, pick * 1000 + c, [c])
    __syncthreads()
    _t = tid(0)
    if _t < C:
        v = smem_load(row_desc, [_t])
        st(out_row + _t, v)


def test_smem_index_subview_row_swizzled_layout():
    R, C, pick = 4, 8, 1
    out = torch.zeros(C, dtype=torch.int32, device="cuda")
    kernel_smem_index_row_swizzled[(1, )](out, R, C, pick, num_warps=1)
    torch.cuda.synchronize()
    exp = torch.tensor([pick * 1000 + c for c in range(C)], dtype=torch.int32, device="cuda")
    assert torch.equal(out, exp), "smem_index row subview with swizzle mismatch"


@triton_dist.jit
def kernel_u32_adjacent_cells(out_lo, out_hi):
    da = allocate_smem(tl.int32, [4])
    if tid(0) == 0:
        tl.store(out_lo, get_smem_shared_address_u32(da, [0]))
        tl.store(out_hi, get_smem_shared_address_u32(da, [1]))


def test_get_smem_shared_address_u32_indexed_adjacent():
    lo = torch.zeros(1, dtype=torch.int32, device="cuda")
    hi = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel_u32_adjacent_cells[(1, )](lo, hi, num_warps=1)
    torch.cuda.synchronize()
    a, b = lo.item(), hi.item()
    assert a != b, f"bad u32 addresses, a = {a}, b = {b}"
    delta = (b - a) & 0xFFFFFFFF
    assert delta == 4, "expected +4 byte offset between int32 cells"


@triton_dist.jit
def kernel_smem_bool_as_int32(out_ptr):
    smem = allocate_smem(tl.int32, [2])
    if tid(0) == 0:
        smem_store(smem, False, [0])
        smem_store(smem, True, [1])
    __syncthreads()
    if tid(0) == 0:
        tl.store(out_ptr + 0, smem_load(smem, [0]))
        tl.store(out_ptr + 1, smem_load(smem, [1]))


def test_smem_store_bool_constants():
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    kernel_smem_bool_as_int32[(1, )](out, num_warps=1)
    torch.cuda.synchronize()
    assert out[0].item() == 0 and out[1].item() == 1


@triton_dist.jit
def kernel_smem_get_ptr_base_matches_index0(out_same):
    smem = allocate_smem(tl.int32, [8])
    if tid(0) == 0:
        pb = smem_get_ptr(smem)
        p0 = smem_get_ptr(smem, [0])
        b = tl.cast(pb, tl.uint64, bitcast=True)
        z = tl.cast(p0, tl.uint64, bitcast=True)
        tl.store(out_same, tl.where(b == z, tl.cast(1, tl.uint64), tl.cast(0, tl.uint64)))


def test_smem_get_ptr_empty_indices_same_as_zero():
    """Base memdesc pointer must match explicit zero-dimensional index."""
    out = torch.zeros(1, dtype=torch.int64, device="cuda")
    kernel_smem_get_ptr_base_matches_index0[(1, )](out, num_warps=1)
    torch.cuda.synchronize()
    assert out.item() == 1


@triton_dist.jit
def kernel_smem_dealloc_reuse(out_ptr):
    """Allocate, use, dealloc, allocate again — verifies lifetime management."""
    smem_a = allocate_smem(tl.int32, [4])
    if tid(0) == 0:
        smem_store(smem_a, 100, [0])
    __syncthreads()
    if tid(0) == 0:
        va = smem_load(smem_a, [0])
        st(out_ptr, va)
    __syncthreads()
    smem_dealloc(smem_a)

    smem_b = allocate_smem(tl.int32, [4])
    if tid(0) == 0:
        smem_store(smem_b, 200, [0])
    __syncthreads()
    if tid(0) == 0:
        vb = smem_load(smem_b, [0])
        st(out_ptr + 1, vb)
    __syncthreads()
    smem_dealloc(smem_b)


def test_smem_dealloc_and_realloc():
    """Dealloc then re-alloc must produce correct results from each allocation."""
    out = torch.zeros(2, dtype=torch.int32, device="cuda")
    kernel_smem_dealloc_reuse[(1, )](out, num_warps=1)
    torch.cuda.synchronize()
    assert out[0].item() == 100, f"first alloc value wrong: {out[0].item()}"
    assert out[1].item() == 200, f"second alloc value wrong: {out[1].item()}"


@triton_dist.jit
def kernel_smem_scatter_gather(out_ptr, N: tl.constexpr, NT: tl.constexpr):
    """All threads write their tid, then read back in reverse order."""
    smem = allocate_smem(tl.int32, [N])
    _t = tid(0)
    if _t < N:
        smem_store(smem, tl.cast(_t * 7 + 3, tl.int32), [_t])
    __syncthreads()
    if _t < N:
        rev = N - 1 - _t
        v = smem_load(smem, [rev])
        st(out_ptr + _t, v)


def test_smem_multi_thread_scatter_gather():
    """32 threads scatter distinct values, read in reverse — covers parallel smem access."""
    N = 32
    out = torch.zeros(N, dtype=torch.int32, device="cuda")
    kernel_smem_scatter_gather[(1, )](out, N, N, num_warps=1)
    torch.cuda.synchronize()
    exp = torch.tensor([(N - 1 - t) * 7 + 3 for t in range(N)], dtype=torch.int32, device="cuda")
    assert torch.equal(out, exp), f"scatter/gather mismatch: {out.tolist()} vs {exp.tolist()}"


@triton_dist.jit
def kernel_smem_3d_index(out_ptr, D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr):
    """3D smem allocation: index dim-0 twice, verify sub-subview."""
    smem = allocate_smem(tl.int32, [D0, D1, D2])
    if tid(0) == 0:
        for r in tl.static_range(D1):
            for c in tl.static_range(D2):
                smem_store(smem_index(smem, 1), 10 * r + c, [r, c])
    __syncthreads()
    _t = tid(0)
    if _t < D1 * D2:
        r = _t // D2
        c = _t % D2
        v = smem_load(smem_index(smem, 1), [r, c])
        st(out_ptr + _t, v)


def test_smem_3d_index_subview():
    """3D allocation with smem_index along dim-0 produces correct 2D subview."""
    D0, D1, D2 = 3, 4, 8
    out = torch.zeros(D1 * D2, dtype=torch.int32, device="cuda")
    kernel_smem_3d_index[(1, )](out, D0, D1, D2, num_warps=1)
    torch.cuda.synchronize()
    exp = torch.tensor([10 * r + c for r in range(D1) for c in range(D2)], dtype=torch.int32, device="cuda")
    assert torch.equal(out, exp), "3D subview mismatch"


def test_shared_mem_desc_type_repr():
    """SharedMemDescType.__repr__ returns a human-readable string."""
    ty = SharedMemDescType(tl.int32, (4, 8), (4, 8), 1, 1, 1)
    s = repr(ty)
    assert "SharedMemDescType" in s
    assert "(4, 8)" in s


def test_shared_mem_desc_type_equality():
    """Equality and hash for SharedMemDescType."""
    a = SharedMemDescType(tl.int32, (4, 8), (4, 8), 1, 1, 1)
    b = SharedMemDescType(tl.int32, (4, 8), (4, 8), 1, 1, 1)
    c = SharedMemDescType(tl.bfloat16, (4, 8), (4, 8), 1, 1, 1)
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    assert hash(a) != hash(c)


def test_allocate_smem_validation():
    """allocate_smem rejects empty or zero-dimensioned shapes at trace time."""
    caught_empty = False
    try:

        @triton_dist.jit
        def _k1(p):
            allocate_smem(tl.int32, [])

        _k1[(1, )](torch.zeros(1, device="cuda"), num_warps=1)
    except Exception as e:
        cause = e.__cause__ if e.__cause__ else e
        assert "non-empty" in str(cause), f"unexpected error: {cause}"
        caught_empty = True
    assert caught_empty, "allocate_smem([]) should raise"

    caught_zero = False
    try:

        @triton_dist.jit
        def _k2(p):
            allocate_smem(tl.int32, [0])

        _k2[(1, )](torch.zeros(1, device="cuda"), num_warps=1)
    except Exception as e:
        cause = e.__cause__ if e.__cause__ else e
        assert "positive" in str(cause), f"unexpected error: {cause}"
        caught_zero = True
    assert caught_zero, "allocate_smem([0]) should raise"


def test_create_tmap_scratch_validation():
    """create_tmap_scratch rejects non-positive inputs."""
    from triton_dist.tma_utils import create_tmap_scratch
    with pytest.raises(ValueError, match="num_ctas"):
        create_tmap_scratch(0)
    with pytest.raises(ValueError, match="num_descs_per_cta"):
        create_tmap_scratch(1, num_descs_per_cta=0)


_TESTS = (
    test_smem_index_subview_row,
    test_smem_index_subview_row_swizzled_layout,
    test_get_smem_shared_address_u32_indexed_adjacent,
    test_smem_store_bool_constants,
    test_smem_get_ptr_empty_indices_same_as_zero,
    test_smem_dealloc_and_realloc,
    test_smem_multi_thread_scatter_gather,
    test_smem_3d_index_subview,
    test_shared_mem_desc_type_repr,
    test_shared_mem_desc_type_equality,
    test_allocate_smem_validation,
    test_create_tmap_scratch_validation,
)

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"{t.__name__}: PASS")
