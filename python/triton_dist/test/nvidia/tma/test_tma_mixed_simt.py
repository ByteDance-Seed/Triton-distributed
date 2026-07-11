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
"""TMA load to smem, SIMT smem_load/store scale, TMA store out."""
from __future__ import annotations

import torch
import triton.language as tl
import triton_dist
from triton.language import core as tlc

from triton_dist.language.smem_ops import allocate_smem, smem_get_ptr, smem_load, smem_store
from triton_dist.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.language.extra.cuda.tma_language import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
    elect_one_sync,
    fence_async_shared,
    mbarrier_expect_tx,
    mbarrier_init,
    mbarrier_invalidate,
    mbarrier_wait_parity,
    tma_load_2d,
    tma_store_2d,
)
from triton_dist.language.tma import ELEM_BF16, SWIZZLE_128B, create_tmap_2d
from triton_dist.tma_utils import create_tmap_scratch


@triton_dist.jit
def kernel_tma_simt_scale(
    in_ptr,
    out_ptr,
    scratch_ptr,
    M,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    NUM_THREADS: tl.constexpr,
):
    pid = tl.program_id(0)
    _x = tid(0)
    tile_b: tl.constexpr = BM * BN * 2
    ept: tl.constexpr = (BM * BN) // NUM_THREADS

    smem_tmap = allocate_smem(tl.int32, [32])
    smem_bar = allocate_smem(tl.int64, [1])
    smem_tile = allocate_smem(tl.bfloat16, [BM, BN])

    buf_ptr = smem_get_ptr(smem_tile)
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
            mbarrier_expect_tx(bar_ptr, tile_b)
            tma_load_2d(tmap_in, buf_ptr, bar_ptr, 0, row)
    mbarrier_wait_parity(bar_ptr, 0)

    __syncthreads()
    for k in tl.static_range(ept):
        lin = _x * ept + k
        r = lin // BN
        c = lin % BN
        v = smem_load(smem_tile, [r, c])
        smem_store(smem_tile, v + v, [r, c])
    __syncthreads()

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


def test_tma_then_simt_scale_then_tma_store():
    # Single CTA, single warp: all lanes participate in elect_one_sync on the TMA path.
    M, N, BM, BN = 64, 64, 64, 64
    num_threads = 32
    assert (BM * BN) % num_threads == 0
    inp = torch.ones(M, N, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(inp)
    scr = create_tmap_scratch(M // BM, num_descs_per_cta=2)
    kernel_tma_simt_scale[(M // BM, )](
        inp,
        out,
        scr,
        M,
        N,
        BM,
        BN,
        num_threads,
        num_warps=1,
    )
    torch.cuda.synchronize()
    exp = inp + inp
    assert torch.equal(out, exp), "TMA+SIMT smem v+v must match 2.0 tile copy"


def test_tma_simt_scale_multi_cta():
    """Multi-CTA (4 tiles): each CTA loads, doubles, stores back."""
    M, N, BM, BN = 256, 64, 64, 64
    num_tiles = M // BM
    num_threads = 128
    assert (BM * BN) % num_threads == 0
    inp = torch.ones(M, N, dtype=torch.bfloat16, device="cuda") * 3.0
    out = torch.zeros_like(inp)
    scr = create_tmap_scratch(num_tiles, num_descs_per_cta=2)
    kernel_tma_simt_scale[(num_tiles, )](
        inp,
        out,
        scr,
        M,
        N,
        BM,
        BN,
        num_threads,
        num_warps=4,
    )
    torch.cuda.synchronize()
    exp = inp + inp
    assert torch.equal(out, exp), "multi-CTA TMA+SIMT v+v must equal 6.0"


_TESTS = (
    test_tma_then_simt_scale_then_tma_store,
    test_tma_simt_scale_multi_cta,
)

if __name__ == "__main__":
    for t in _TESTS:
        t()
        print(f"{t.__name__}: PASS")
