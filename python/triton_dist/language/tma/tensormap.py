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
"""Device-side tensormap construction via PTX ``tensormap.replace`` intrinsics.

Provides constants for CUtensorMap encoding (element type, swizzle mode, etc.)
and ``@jit`` helpers that build or patch a 2-D tensormap entirely in-kernel
using shared-memory workspace + a global slot.

These are independent of the pipeline abstractions in ``pipeline.py`` and can
be used by any kernel that needs device-side TMA descriptors.
"""
import triton.language as tl

from triton_dist.jit import jit

from triton_dist.language.extra.cuda.language_extra import __syncthreads, tid
from triton_dist.language.extra.cuda.tma_language import (
    bar_warp_sync,
    cvta_global_to_generic_u64,
    fence_proxy_tensormap_acquire,
    tensormap_cp_fenceproxy,
    tensormap_replace_box_dim,
    tensormap_replace_element_stride,
    tensormap_replace_elemtype,
    tensormap_replace_fill_mode,
    tensormap_replace_global_address,
    tensormap_replace_global_dim,
    tensormap_replace_global_stride,
    tensormap_replace_interleave_layout,
    tensormap_replace_rank,
    tensormap_replace_swizzle_mode,
    tensormap_smem_zero_first_warp,
)

from triton_dist.language.smem_ops import get_smem_shared_address_u32

# ---------------------------------------------------------------------------
# CUtensorMap / PTX tensormap immediates
# ---------------------------------------------------------------------------
ELEM_BF16 = tl.constexpr(0x0A)
ELEM_F32 = tl.constexpr(0x07)
SWIZZLE_128B = tl.constexpr(0x03)
SWIZZLE_NONE = tl.constexpr(0x00)
INTERLEAVE_NONE = tl.constexpr(0x00)
FILL_NONE = tl.constexpr(0x00)
TMAP_RANK_2D = tl.constexpr(1)
_GD0 = tl.constexpr(0)
_GD1 = tl.constexpr(1)
_BOX0 = tl.constexpr(0)
_BOX1 = tl.constexpr(1)
_ES0 = tl.constexpr(0)
_ES1 = tl.constexpr(1)
_GS0 = tl.constexpr(0)

# ===================================================================
# Tensormap build / update helpers
# ===================================================================


@jit
def _tensormap_build_publish_2d(
    tmap_smem_u32,
    tmap_global_ptr,
    data_ptr,
    N,
    M,
    stride_bytes,
    BN,
    BM,
    elemtype: tl.constexpr,
    swizzle: tl.constexpr,
):
    """Build a 2D tensormap in shared memory and publish to a global slot."""
    tensormap_smem_zero_first_warp(tmap_smem_u32)
    bar_warp_sync()
    s_u64 = tl.cast(tmap_smem_u32, tl.uint64)
    _t = tid(0)
    one = tl.cast(1, tl.int32)
    data_i64 = tl.cast(data_ptr, tl.uint64, bitcast=True)
    tg_i64 = tl.cast(tmap_global_ptr, tl.uint64, bitcast=True)
    if _t == 0:
        tensormap_replace_global_address(s_u64, data_i64)
        tensormap_replace_rank(s_u64, TMAP_RANK_2D)
        tensormap_replace_box_dim(s_u64, _BOX0, BN)
        tensormap_replace_box_dim(s_u64, _BOX1, BM)
        tensormap_replace_global_dim(s_u64, _GD0, N)
        tensormap_replace_global_dim(s_u64, _GD1, M)
        tensormap_replace_global_stride(s_u64, _GS0, stride_bytes)
        tensormap_replace_element_stride(s_u64, _ES0, one)
        tensormap_replace_element_stride(s_u64, _ES1, one)
        tensormap_replace_elemtype(s_u64, elemtype)
        tensormap_replace_interleave_layout(s_u64, INTERLEAVE_NONE)
        tensormap_replace_swizzle_mode(s_u64, swizzle)
        tensormap_replace_fill_mode(s_u64, FILL_NONE)
    if _t < 32:
        tensormap_cp_fenceproxy(tg_i64, s_u64)
        fence_proxy_tensormap_acquire(tg_i64)
    __syncthreads()
    return cvta_global_to_generic_u64(tg_i64)


@jit
def create_tmap_2d(
    smem_tmap,
    tmap_global_ptr,
    data_ptr,
    N,
    M,
    stride_bytes,
    BN,
    BM,
    elemtype: tl.constexpr,
    swizzle: tl.constexpr,
):
    """Create and publish a 2D tensormap (128 B smem workspace + global slot)."""
    s_u32 = get_smem_shared_address_u32(smem_tmap)
    return _tensormap_build_publish_2d(
        s_u32,
        tmap_global_ptr,
        data_ptr,
        N,
        M,
        stride_bytes,
        BN,
        BM,
        elemtype,
        swizzle,
    )


@jit
def tmap_update_address(smem_tmap, tmap_global_ptr, new_data_ptr):
    """Update only the global_address field of an existing tensormap."""
    s_u32 = get_smem_shared_address_u32(smem_tmap)
    s_u64 = tl.cast(s_u32, tl.uint64)
    _t = tid(0)
    new_i64 = tl.cast(new_data_ptr, tl.uint64, bitcast=True)
    tg_i64 = tl.cast(tmap_global_ptr, tl.uint64, bitcast=True)
    if _t == 0:
        tensormap_replace_global_address(s_u64, new_i64)
    if _t < 32:
        tensormap_cp_fenceproxy(tg_i64, s_u64)
        fence_proxy_tensormap_acquire(tg_i64)
    __syncthreads()
    return cvta_global_to_generic_u64(tg_i64)


__all__ = [
    "ELEM_BF16",
    "ELEM_F32",
    "SWIZZLE_128B",
    "SWIZZLE_NONE",
    "INTERLEAVE_NONE",
    "FILL_NONE",
    "create_tmap_2d",
    "tmap_update_address",
]
