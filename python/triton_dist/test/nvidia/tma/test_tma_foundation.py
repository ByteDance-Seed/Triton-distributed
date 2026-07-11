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
"""Baseline coverage: smem memdesc, mbarrier + device tensormap TMA, predicated
loads, L2 + tensormap prefetch, warp/barrier, and PTX fence helpers. In-kernel
``tmap_update_address`` stress lives in ``tma_old/`` (see below).

Standalone ``cp_async_bulk_{global_to_shared,shared_to_global}`` are not called with
raw memdesc pointers here (Triton FE rejects two ``pointer<uint8>`` operands in the
same inline_asm bundle); bulk groups are still exercised via TMA store +
``cp_async_bulk_commit_group`` / ``wait_group``.

``tmap_update_address`` is exercised in the pipeline and mixed-SIMT test suites;
a dedicated multi-phase in-kernel test was flaky (illegal access) on some GPUs.

Coverage inventory (this package vs :mod:`triton_dist.language.smem_ops`):
``allocate_smem``, ``smem_load``/``store``, ``smem_get_ptr``, ``get_smem_shared_address_u32``,
``smem_dealloc`` (smoke). ``smem_index`` / indexed u32 / bool store: ``test_smem_ops``; pipeline: ``test_tma_pipeline``. Raw
``cp_async_bulk_{g2s,s2g}`` and in-kernel ``tmap_update_address`` stress: ``tma_old/``.

Assertions are bitwise or exact numerical checks on global data."""
from __future__ import annotations

import torch
import triton.language as tl
import triton_dist
from triton.language import core as tlc

from triton_dist.language.smem_ops import (
    allocate_smem,
    get_smem_shared_address_u32,
    smem_dealloc,
    smem_get_ptr,
    smem_load,
    smem_store,
)
from triton_dist.language.extra.cuda.language_extra import __syncthreads, st, tid
from triton_dist.language.extra.cuda.tma_language import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    cvta_generic_to_shared,
    elect_one_sync,
    fence_async_shared,
    fence_proxy_async,
    mbarrier_arrive_expect_tx,
    mbarrier_expect_tx,
    mbarrier_init,
    mbarrier_invalidate,
    mbarrier_try_wait_parity,
    mbarrier_wait_parity,
    named_barrier_sync,
    prefetch_tensormap,
    prefetch_tensor_2d_l2,
    tma_load_2d,
    tma_load_2d_pred,
    tma_store_2d,
    warp_idx,
)
from triton_dist.language.tma import (
    ELEM_BF16,
    SWIZZLE_128B,
    create_tmap_2d,
)
from triton_dist.tma_utils import create_tmap_scratch, create_tensormap_2d


@triton_dist.jit
def kernel_smem_pattern(out_ptr, BM: tl.constexpr, BN: tl.constexpr):
    _t = tid(0)
    THREADS: tl.constexpr = 128
    ELEMS: tl.constexpr = BM * BN
    EPT: tl.constexpr = ELEMS // THREADS
    smem = allocate_smem(tl.int32, [BM, BN])
    for i in range(EPT):
        idx = _t * EPT + i
        row = idx // BN
        col = idx % BN
        smem_store(smem, idx, [row, col])
    __syncthreads()
    for i in range(EPT):
        idx = _t * EPT + i
        row = idx // BN
        col = idx % BN
        v = smem_load(smem, [row, col])
        st(out_ptr + idx, v)


def test_smem_pattern():
    BM, BN = 32, 32
    out = torch.zeros(BM * BN, dtype=torch.int32, device="cuda")
    kernel_smem_pattern[(1, )](out, BM, BN, num_warps=4)
    torch.cuda.synchronize()
    exp = torch.arange(BM * BN, dtype=torch.int32, device="cuda")
    assert torch.equal(out, exp), "smem scatter/gather pattern mismatch"


@triton_dist.jit
def kernel_smem_dealloc(out_ptr):
    """Explicit ``local_dealloc`` path; all threads must reach dealloc after last use."""
    smem = allocate_smem(tl.int32, [1])
    if tid(0) == 0:
        smem_store(smem, 42, [0])
    __syncthreads()
    if tid(0) == 0:
        v = smem_load(smem, [0])
        tl.store(out_ptr, v)
    __syncthreads()
    smem_dealloc(smem)


def test_smem_dealloc_smoke():
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel_smem_dealloc[(1, )](out, num_warps=1)
    torch.cuda.synchronize()
    assert out.item() == 42


@triton_dist.jit
def kernel_shared_u32_addrs(oa, ob):
    da = allocate_smem(tl.int32, [64])
    db = allocate_smem(tl.int32, [64])
    if tid(0) == 0:
        tl.store(oa, get_smem_shared_address_u32(da))
        tl.store(ob, get_smem_shared_address_u32(db))


def test_shared_address_u32_distinct():
    outa = torch.zeros(1, dtype=torch.int32, device="cuda")
    outb = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel_shared_u32_addrs[(1, )](outa, outb, num_warps=1)
    torch.cuda.synchronize()
    ua, ub = outa.item(), outb.item()
    assert ua != 0 and ub != 0 and ua != ub, f"bad shared u32 {ua:#x} {ub:#x}"


@triton_dist.jit
def kernel_tmap_identity(
    in_ptr,
    out_ptr,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    use_arrive_expect: tl.constexpr,
):
    pid = tl.program_id(0)
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg_in = tl.cast(scratch_ptr + pid * 256, tl.uint64, bitcast=True)
    tg_out = tl.cast(scratch_ptr + pid * 256 + 128, tl.uint64, bitcast=True)
    tmap_in = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tmap_out = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    fence_proxy_async()
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    row = pid * BM
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            if use_arrive_expect:
                mbarrier_arrive_expect_tx(bar_ptr, tile_b)
            else:
                mbarrier_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_in, buf_ptr, bar_ptr, 0, row)
    mbarrier_wait_parity(bar_ptr, 0)
    fence_async_shared()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            tma_store_2d(tmap_out, buf_ptr, 0, row)
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(True))
    __syncthreads()
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_device_tmap_identity_both_expect_variants():
    M, N, BM, BN = 128, 64, 64, 64
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    scr = create_tmap_scratch(M // BM, num_descs_per_cta=2)
    for use_ae in (False, True):
        out = torch.zeros_like(inp)
        kernel_tmap_identity[(M // BM, )](
            inp,
            out,
            scr,
            M,
            N,
            BM,
            BN,
            use_ae,
            num_warps=4,
        )
        torch.cuda.synchronize()
        assert torch.equal(out, inp), f"identity copy failed use_arrive_expect={use_ae}"


@triton_dist.jit
def kernel_try_wait_flag(in_ptr, scratch_ptr, out_flag, M, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg = tl.cast(scratch_ptr, tl.uint64, bitcast=True)
    tmap_g = create_tmap_2d(
        smem_tmap,
        tg,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_g, buf_ptr, bar_ptr, 0, 0)
    mbarrier_wait_parity(bar_ptr, 0)
    ok = mbarrier_try_wait_parity(bar_ptr, 0)
    if _x == 0:
        tl.store(out_flag, ok)
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_mbarrier_try_wait_nonzero_after_completion():
    M, N, BM, BN = 64, 64, 64, 64
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    scr = create_tmap_scratch(1, 1)
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel_try_wait_flag[(1, )](inp, scr, flag, M, N, BM, BN, num_warps=1)
    torch.cuda.synchronize()
    assert flag.item() != 0, "try_wait_parity should succeed after barrier phase completes"


@triton_dist.jit
def kernel_cvta_smoke(out_hi, out_lo, BM: tl.constexpr, BN: tl.constexpr):
    smem = allocate_smem(tl.bfloat16, [BM, BN])
    p = smem_get_ptr(smem)
    su = cvta_generic_to_shared(p)
    if tid(0) == 0:
        tl.store(out_lo, tl.cast(su, tl.uint32))
        tl.store(out_hi, tl.cast(su >> 32, tl.uint32))


def test_cvta_generic_to_shared_nonzero():
    hi = torch.zeros(1, dtype=torch.int32, device="cuda")
    lo = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel_cvta_smoke[(1, )](hi, lo, 8, 8, num_warps=1)
    torch.cuda.synchronize()
    assert (hi.item() | lo.item()) != 0, "cvta_generic_to_shared produced zero"


@triton_dist.jit
def kernel_pred_load(
    in_ptr,
    out_ptr,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg_in = tl.cast(scratch_ptr, tl.uint64, bitcast=True)
    tg_out = tl.cast(scratch_ptr + 128, tl.uint64, bitcast=True)
    tmap_in = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tmap_out = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    sent = tl.cast(0x7FC0, tl.bfloat16)
    ept: tl.constexpr = (BM * BN) // 32
    for k in tl.static_range(ept):
        lin = _x * ept + k
        r = lin // BN
        c = lin % BN
        smem_store(smem_buf, sent, [r, c])
    __syncthreads()
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    pred_skip = tl.cast(1, tl.int32)
    pred_take = tl.cast(1, tl.int32)
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_expect_tx(bar_ptr, tile_b)
            tma_load_2d_pred(tmap_in, buf_ptr, bar_ptr, 0, BM, pred_skip)
    mbarrier_wait_parity(bar_ptr, 0)
    if _x == 0:
        mbarrier_invalidate(bar_ptr)
    __syncthreads()
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_expect_tx(bar_ptr, tile_b)
            tma_load_2d_pred(tmap_in, buf_ptr, bar_ptr, 0, 0, pred_take)
    mbarrier_wait_parity(bar_ptr, 0)
    fence_async_shared()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            tma_store_2d(tmap_out, buf_ptr, 0, 0)
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(True))
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_tma_load_pred_skip_then_take():
    """First predicated load is skipped (smem keeps NaN sentinel); second loads row 0."""
    M, N, BM, BN = 128, 64, 64, 64
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    scr = create_tmap_scratch(1, num_descs_per_cta=2)
    kernel_pred_load[(1, )](inp, out, scr, M, N, BM, BN, num_warps=1)
    torch.cuda.synchronize()
    assert torch.equal(out[0:BM], inp[0:BM])
    tail = out[BM:]
    assert torch.all(tail == 0), "rows never stored must stay zero"


@triton_dist.jit
def kernel_prefetch_then_copy(
    in_ptr,
    out_ptr,
    tm_bytes,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg_in = tl.cast(scratch_ptr, tl.uint64, bitcast=True)
    tg_out = tl.cast(scratch_ptr + 128, tl.uint64, bitcast=True)
    if _x == 0:
        prefetch_tensormap(tm_bytes)
        prefetch_tensor_2d_l2(tm_bytes, tl.cast(0, tl.uint32), tl.cast(0, tl.uint32))
    __syncthreads()
    tmap_in = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tmap_out = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_in, buf_ptr, bar_ptr, 0, 0)
    mbarrier_wait_parity(bar_ptr, 0)
    fence_async_shared()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            tma_store_2d(tmap_out, buf_ptr, 0, 0)
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(True))
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_prefetch_host_tmap_then_device_copy():
    M, N, BM, BN = 64, 64, 64, 64
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    raw = create_tensormap_2d(inp, (BM, BN))
    tm_dev = raw.to(device="cuda")
    scr = create_tmap_scratch(1, num_descs_per_cta=2)
    kernel_prefetch_then_copy[(1, )](inp, out, tm_dev, scr, M, N, BM, BN, num_warps=2)
    torch.cuda.synchronize()
    assert torch.equal(out[0:BM], inp[0:BM])


@triton_dist.jit
def kernel_warp_named_barrier(out_warp, out_ok, NT: tl.constexpr):
    named_barrier_sync(tlc.constexpr(1), NT)
    if tid(0) == 0:
        tl.store(out_warp, warp_idx())
        tl.store(out_ok, tl.cast(1, tl.int32))


def test_warp_idx_and_named_barrier():
    NT = 128
    w = torch.zeros(1, dtype=torch.int32, device="cuda")
    ok = torch.zeros(1, dtype=torch.int32, device="cuda")
    kernel_warp_named_barrier[(1, )](w, ok, NT, num_warps=4)
    torch.cuda.synchronize()
    assert ok.item() == 1
    assert w.item() == 0


@triton_dist.jit
def kernel_multi_cta_identity(
    in_ptr,
    out_ptr,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Each CTA copies one tile — exercises per-CTA tensormap slot addressing."""
    pid = tl.program_id(0)
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg_in = tl.cast(scratch_ptr + pid * 256, tl.uint64, bitcast=True)
    tg_out = tl.cast(scratch_ptr + pid * 256 + 128, tl.uint64, bitcast=True)
    tmap_in = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tmap_out = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    row = pid * BM
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_arrive_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_in, buf_ptr, bar_ptr, 0, row)
    mbarrier_wait_parity(bar_ptr, 0)
    fence_async_shared()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            tma_store_2d(tmap_out, buf_ptr, 0, row)
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(True))
    __syncthreads()
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_multi_cta_tma_identity():
    """4 CTAs each copy a 64x64 tile — verifies per-CTA scratch slot isolation."""
    M, N, BM, BN = 256, 64, 64, 64
    num_ctas = M // BM
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    scr = create_tmap_scratch(num_ctas, num_descs_per_cta=2)
    kernel_multi_cta_identity[(num_ctas, )](inp, out, scr, M, N, BM, BN, num_warps=2)
    torch.cuda.synchronize()
    assert torch.equal(out, inp), "multi-CTA identity copy failed"


@triton_dist.jit
def kernel_double_wait(
    in_ptr,
    out_ptr,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """Calls mbarrier_wait_parity twice — regression test for PTX label uniqueness."""
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg_in = tl.cast(scratch_ptr, tl.uint64, bitcast=True)
    tg_out = tl.cast(scratch_ptr + 128, tl.uint64, bitcast=True)
    tmap_in = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tmap_out = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()

    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_arrive_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_in, buf_ptr, bar_ptr, 0, 0)
    mbarrier_wait_parity(bar_ptr, 0)
    if _x == 0:
        mbarrier_invalidate(bar_ptr)

    __syncthreads()
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()

    fence_async_shared()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            tma_store_2d(tmap_out, buf_ptr, 0, 0)
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(True))

    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_arrive_expect_tx(bar_ptr, 0)
    mbarrier_wait_parity(bar_ptr, 0)
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_double_mbarrier_wait_unique_labels():
    """Two mbarrier_wait_parity calls compile without PTX label collision."""
    M, N, BM, BN = 64, 64, 64, 64
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    scr = create_tmap_scratch(1, num_descs_per_cta=2)
    kernel_double_wait[(1, )](inp, out, scr, M, N, BM, BN, num_warps=1)
    torch.cuda.synchronize()
    assert torch.equal(out[:BM], inp[:BM]), "double-wait identity copy failed"


@triton_dist.jit
def kernel_multi_warp_tma(
    in_ptr,
    out_ptr,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """4-warp kernel: only warp 0 issues TMA, all warps wait."""
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_buf = allocate_smem(tl.bfloat16, [BM, BN])
    buf_ptr = smem_get_ptr(smem_buf)
    bar_ptr = smem_get_ptr(smem_bar)
    stride_b = tl.cast(N * 2, tl.int64)
    tg_in = tl.cast(scratch_ptr, tl.uint64, bitcast=True)
    tg_out = tl.cast(scratch_ptr + 128, tl.uint64, bitcast=True)
    tmap_in = create_tmap_2d(
        smem_tmap,
        tg_in,
        in_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    tmap_out = create_tmap_2d(
        smem_tmap,
        tg_out,
        out_ptr,
        N,
        M,
        stride_b,
        BN,
        BM,
        ELEM_BF16,
        SWIZZLE_128B,
    )
    if _x == 0:
        mbarrier_init(bar_ptr, 1)
    __syncthreads()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            mbarrier_arrive_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_in, buf_ptr, bar_ptr, 0, 0)
    mbarrier_wait_parity(bar_ptr, 0)
    fence_async_shared()
    if _x < 32:
        e = elect_one_sync()
        if e != 0:
            tma_store_2d(tmap_out, buf_ptr, 0, 0)
            cp_async_bulk_commit_group()
            cp_async_bulk_wait_group(tlc.constexpr(0), read=tlc.constexpr(True))
    __syncthreads()
    if _x == 0:
        mbarrier_invalidate(bar_ptr)


def test_multi_warp_tma_identity():
    """4-warp CTA where warp 0 drives TMA — ensures warp scoping is correct."""
    M, N, BM, BN = 64, 64, 64, 64
    inp = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    scr = create_tmap_scratch(1, num_descs_per_cta=2)
    kernel_multi_warp_tma[(1, )](inp, out, scr, M, N, BM, BN, num_warps=4)
    torch.cuda.synchronize()
    assert torch.equal(out, inp), "multi-warp TMA identity failed"


_TESTS = (
    test_smem_pattern,
    test_smem_dealloc_smoke,
    test_shared_address_u32_distinct,
    test_device_tmap_identity_both_expect_variants,
    test_mbarrier_try_wait_nonzero_after_completion,
    test_cvta_generic_to_shared_nonzero,
    test_tma_load_pred_skip_then_take,
    test_prefetch_host_tmap_then_device_copy,
    test_warp_idx_and_named_barrier,
    test_multi_cta_tma_identity,
    test_double_mbarrier_wait_unique_labels,
    test_multi_warp_tma_identity,
)

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"{t.__name__}: PASS")
