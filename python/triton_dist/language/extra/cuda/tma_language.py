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
"""
TMA (Tensor Memory Accelerator) language primitives for Triton-distributed.

All operations use inline PTX and operate on raw pointers obtained from
compiler-allocated shared memory (via simt.memdesc_to_ptr). This ensures
compatibility with the compiler's shared memory allocation pass.

Naming conventions (mbarrier family):
  - ``mbarrier_arrive_expect_tx``:  arrive **and** set expected tx bytes
    (PTX ``mbarrier.arrive.expect_tx``).  Canonical for TMA producer pattern.
  - ``mbarrier_expect_transaction``:  set expected tx bytes **without**
    arriving (PTX ``mbarrier.expect_tx``).
  - ``mbarrier_arrive``:  arrive without setting tx bytes.
"""

import itertools

import triton.language as tl
from triton.language import core
from triton.language.core import builtin

_label_counter = itertools.count()


@builtin
def _ptr_to_shared_u32(ptr, _semantic=None):
    """Generic pointer → u32 ``.shared::cta`` address (PTX); internal helper.

    Callers that have a ``SharedMemoryDesc`` should use
    ``get_smem_shared_address_u32`` from ``smem_ops`` instead. This stays for
    primitives that take an already-built generic pointer (e.g. mbarrier/TMA
    wrappers).
    """
    ptr_i64 = tl.cast(ptr, tl.uint64, bitcast=True, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u64 %generic_addr;
            mov.u64 %generic_addr, $1;
            cvta.to.shared.u64 %generic_addr, %generic_addr;
            cvt.u32.u64 $0, %generic_addr;
        }
        """,
        constraints="=r,l",
        args=[ptr_i64],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_init(mbar_ptr, count, _semantic=None):
    """Initialize an mbarrier with the given arrival count.

    Call from **one thread per CTA** (then block-wide sync). No ``elect.sync``:
    the whole warp must not execute this unless you intentionally use a
    warp-collective elect pattern elsewhere.
    """
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        mbarrier.init.shared::cta.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,r,r",
        args=[saddr, count],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_expect_tx(mbar_ptr, tx_count, _semantic=None):
    """Arrive **and** set expected tx bytes (identical to :func:`mbarrier_arrive_expect_tx`).

    Issues PTX ``mbarrier.arrive.expect_tx``.  Despite the short name this
    **does include an arrive**.  Use :func:`mbarrier_expect_transaction` for
    the pure ``mbarrier.expect_tx`` (no arrive).
    """
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,r,r",
        args=[saddr, tx_count],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_expect_transaction(mbar_ptr, tx_count, _semantic=None):
    """Increment expected transaction bytes without an arrive (``mbarrier.expect_tx``).

    Matches CUTLASS ``ClusterTransactionBarrier::expect_transaction``. Use when the
    barrier state already reflects arrivals and you only need to register byte count.
    """
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        mbarrier.expect_tx.shared::cta.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,r,r",
        args=[saddr, tx_count],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_arrive(mbar_ptr, _semantic=None):
    """Arrive at an mbarrier (decrement count by 1)."""
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b64 %state;
            mbarrier.arrive.shared::cta.b64 %state, [$1];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[saddr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_arrive_expect_tx(mbar_ptr, tx_count, _semantic=None):
    """Combined arrive + set expected tx count (for producer pattern)."""
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,r,r",
        args=[saddr, tx_count],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_try_wait_parity(mbar_ptr, phase, _semantic=None):
    """Single try: returns non-zero if the given parity phase is complete."""
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p;
            mbarrier.try_wait.parity.shared::cta.b64 %p, [$1], $2;
            selp.u32 $0, 1, 0, %p;
        }
        """,
        constraints="=r,r,r",
        args=[saddr, phase],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_wait_parity(mbar_ptr, phase, _semantic=None):
    """Spin-wait until parity phase completes (CUTLASS ``ClusterBarrier::wait`` pattern).

    Uses ``mbarrier.try_wait.parity`` with a timeout tick operand (``0x989680``) and
    retries until the predicate is true, matching ``cutlass/arch/barrier.h``.
    """
    uid = next(_label_counter)
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    ticks = tl.cast(0x989680, tl.uint32, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm=f"""
        {{
            .reg .pred %p;
            MBAR_WAIT_LOOP_{uid}:
            mbarrier.try_wait.parity.shared::cta.b64 %p, [$1], $2, $3;
            @%p bra MBAR_WAIT_DONE_{uid};
            bra MBAR_WAIT_LOOP_{uid};
            MBAR_WAIT_DONE_{uid}:
            mov.u32 $0, 0;
        }}
        """,
        constraints="=r,r,r,r",
        args=[saddr, phase, ticks],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def mbarrier_invalidate(mbar_ptr, _semantic=None):
    """Invalidate an mbarrier (must be called when done).

    Call from **one thread per CTA** (then sync as needed), same contract as init.
    """
    saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        mbarrier.inval.shared::cta.b64 [$1];
        mov.u32 $0, 0;
        """,
        constraints="=r,r",
        args=[saddr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def tma_load_2d(tmap_ptr, smem_ptr, mbar_ptr, coord_0, coord_1, _semantic=None):
    """Async TMA load: global -> shared memory (2D tensor descriptor).

    Issues CUTE Hopper-style ``...complete_tx::bytes.L2::cache_hint`` with hint ``0``
    (same as passing a zero cache-hint register in ``copy_sm90_tma.hpp``).
    The mbarrier tracks completion via transaction count.
    **Only one thread per CTA** may issue this instruction; pair with
    ``mbarrier_expect_tx`` from that same thread. All threads that read smem
    after the load should call ``mbarrier_wait_parity``. The caller must gate
    issuance (e.g. narrow to one warp with ``tid(0) // 32 == 0`` then pick one
    lane, or ``tid(0) == 0`` if the whole CTA is a single warp).
    """
    smem_saddr = _ptr_to_shared_u32(smem_ptr, _semantic=_semantic)
    mbar_saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tmap_ptr_i64 = tl.cast(tmap_ptr, tl.uint64, bitcast=True, _semantic=_semantic)
    cache_hint_u64 = tl.cast(0, tl.uint64, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$1], [$2, {$4, $5}], [$3], $6;
        mov.u32 $0, 0;
        """,
        constraints="=r,r,l,r,r,r,l",
        args=[smem_saddr, tmap_ptr_i64, mbar_saddr, coord_0, coord_1, cache_hint_u64],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def tma_load_2d_pred(tmap_ptr, smem_ptr, mbar_ptr, coord_0, coord_1, pred, _semantic=None):
    """Predicated async TMA load: global -> shared memory (2D).

    Same operand kinds as ``tma_load_2d``; ``pred`` is i32 0/1 in ``$7``.
    """
    smem_saddr = _ptr_to_shared_u32(smem_ptr, _semantic=_semantic)
    mbar_saddr = _ptr_to_shared_u32(mbar_ptr, _semantic=_semantic)
    tmap_ptr_i64 = tl.cast(tmap_ptr, tl.uint64, bitcast=True, _semantic=_semantic)
    cache_hint_u64 = tl.cast(0, tl.uint64, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p;
            setp.eq.s32 %p, $7, 1;
            @%p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$1], [$2, {$4, $5}], [$3], $6;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r,l,r,r,r,l,r",
        args=[smem_saddr, tmap_ptr_i64, mbar_saddr, coord_0, coord_1, cache_hint_u64, pred],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def tma_store_2d(tmap_ptr, smem_ptr, coord_0, coord_1, _semantic=None):
    """Async TMA store: shared memory -> global (2D tensor descriptor).

    Issues cp.async.bulk.tensor.2d.global.shared::cta.
    Completion is tracked via commit groups (cp_async_bulk_commit_group).
    **Only one thread per CTA** may issue this instruction; use
    ``cp_async_bulk_commit_group`` / ``cp_async_bulk_wait_group`` from the same
    thread (or ensure correct bulk-group semantics).
    """
    smem_saddr = _ptr_to_shared_u32(smem_ptr, _semantic=_semantic)
    tmap_ptr_i64 = tl.cast(tmap_ptr, tl.uint64, bitcast=True, _semantic=_semantic)
    tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [$1, {$3, $4}], [$2];
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r,r,r",
        args=[tmap_ptr_i64, smem_saddr, coord_0, coord_1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def cp_async_bulk_global_to_shared(dst_smem_ptr, src_global_ptr, size, mbar_ptr, _semantic=None):
    """Bulk async copy global → shared; mbarrier tracks completion (non-tensor path).

    Operand layout matches CUTE ``SM90_BULK_COPY_G2S`` (``shared::cluster`` dst).
    """
    return tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], $3, [$4];
        mov.u32 $0, 0;
        """,
        constraints="=r,r,l,r,r",
        args=[dst_smem_ptr, src_global_ptr, size, mbar_ptr],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def cp_async_bulk_shared_to_global(dst_global_ptr, src_smem_ptr, size, _semantic=None):
    """Bulk async copy shared → global (bulk_group tracking)."""
    return tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.global.shared::cta.bulk_group [$1], [$2], $3;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,l,r",
        args=[dst_global_ptr, src_smem_ptr, size],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def cp_async_bulk_commit_group(_semantic=None):
    """Commit outstanding bulk async copies into a group for tracking."""
    return tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.commit_group;
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def cp_async_bulk_wait_group(N: core.constexpr, read: core.constexpr = core.constexpr(False), _semantic=None):
    """Wait for bulk async copy groups to complete.

    N: number of most recent groups allowed to be outstanding (0 = wait all).
    read: if True, use .read variant (data visible for read but may not be
          globally visible yet - use for smem->gmem stores where you need to
          read the data back).
    """
    read_suffix = ".read" if read.value else ""
    return tl.inline_asm_elementwise(
        asm=f"""
        cp.async.bulk.wait_group{read_suffix} {N.value};
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


# ============================================================================
# Tensormap management (device-side)
#
# TMA tensor descriptors (tensormaps) are 128-byte structures. On device,
# individual fields can be updated using tensormap.replace.tile.* instructions.
# This is used when different CTAs need to point tensormaps at different
# memory regions (e.g., different remote ranks in all-to-all).
#
# The workflow:
# 1. Host creates initial tensormap via cuTensorMapEncodeTiled
# 2. Copy tensormap to device global memory
# 3. On device, copy tensormap to shared memory for modification
# 4. Use tensormap.replace.tile.* to update fields in shared memory
# 5. Use tensormap.cp_fenceproxy to copy back to global memory
# 6. Use fence.proxy.tensormap to make it visible for TMA
# ============================================================================


@core.extern
def tensormap_replace_global_address(tmap_smem_ptr, new_global_addr, _semantic=None):
    """Replace the global_address field of a tensormap in shared memory."""
    return tl.inline_asm_elementwise(
        asm="""
        tensormap.replace.tile.global_address.shared::cta.b1024.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,l",
        args=[tmap_smem_ptr, new_global_addr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_global_dim(tmap_smem_ptr, ord: core.constexpr, new_val, _semantic=None):
    """Replace a global_dim field of a tensormap in shared memory."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [$1], {ord.value}, $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r",
        args=[tmap_smem_ptr, new_val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_global_stride(tmap_smem_ptr, ord: core.constexpr, new_val, _semantic=None):
    """Replace a global_stride field of a tensormap in shared memory."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [$1], {ord.value}, $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,l",
        args=[tmap_smem_ptr, new_val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_cp_fenceproxy(tmap_global_ptr, tmap_smem_ptr, _semantic=None):
    """Copy tensormap from shared to global memory with fenceproxy.

    This atomically publishes the modified tensormap so TMA can use it.
    """
    return tl.inline_asm_elementwise(
        asm="""
        tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [$1], [$2], 128;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,l",
        args=[tmap_global_ptr, tmap_smem_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def fence_proxy_tensormap_acquire(tmap_global_ptr, _semantic=None):
    """Acquire fence for tensormap - makes prior tensormap updates visible."""
    return tl.inline_asm_elementwise(
        asm="""
        fence.proxy.tensormap::generic.acquire.gpu [$1], 128;
        mov.u32 $0, 0;
        """,
        constraints="=r,l",
        args=[tmap_global_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_rank(tmap_smem_ptr, rank: core.constexpr, _semantic=None):
    """Replace tensor rank field (immediate) of a tensormap in shared memory."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.rank.shared::cta.b1024.b32 [$1], {rank.value};
        mov.u32 $0, 0;
        """,
        constraints="=r,l",
        args=[tmap_smem_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_box_dim(tmap_smem_ptr, ord: core.constexpr, new_val, _semantic=None):
    """Replace a box_dim field of a tensormap in shared memory."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.box_dim.shared::cta.b1024.b32 [$1], {ord.value}, $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r",
        args=[tmap_smem_ptr, new_val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_element_stride(tmap_smem_ptr, ord: core.constexpr, new_val, _semantic=None):
    """Replace an element_stride field of a tensormap in shared memory."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.element_stride.shared::cta.b1024.b32 [$1], {ord.value}, $2;
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r",
        args=[tmap_smem_ptr, new_val],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_elemtype(tmap_smem_ptr, elemtype: core.constexpr, _semantic=None):
    """Replace elemtype field (CUtensorMapDataType encoding)."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.elemtype.shared::cta.b1024.b32 [$1], {elemtype.value};
        mov.u32 $0, 0;
        """,
        constraints="=r,l",
        args=[tmap_smem_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_interleave_layout(tmap_smem_ptr, val: core.constexpr, _semantic=None):
    """Replace interleave_layout field."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [$1], {val.value};
        mov.u32 $0, 0;
        """,
        constraints="=r,l",
        args=[tmap_smem_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_swizzle_mode(tmap_smem_ptr, mode: core.constexpr, _semantic=None):
    """Replace swizzle_mode field."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32 [$1], {mode.value};
        mov.u32 $0, 0;
        """,
        constraints="=r,l",
        args=[tmap_smem_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tensormap_replace_fill_mode(tmap_smem_ptr, mode: core.constexpr, _semantic=None):
    """Replace fill_mode field."""
    return tl.inline_asm_elementwise(
        asm=f"""
        tensormap.replace.tile.fill_mode.shared::cta.b1024.b32 [$1], {mode.value};
        mov.u32 $0, 0;
        """,
        constraints="=r,l",
        args=[tmap_smem_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def bar_warp_sync(_semantic=None):
    """PTX ``bar.warp.sync -1`` — full warp must execute."""
    return tl.inline_asm_elementwise(
        asm="bar.warp.sync -1; mov.u32 $0, 0;",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def cvta_global_to_generic_u64(global_addr_u64, _semantic=None):
    """``cvta.global.u64`` — global device pointer to generic address for TMA descriptor use."""
    return tl.inline_asm_elementwise(
        asm="cvta.global.u64 $0, $1;",
        constraints="=l,l",
        args=[global_addr_u64],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@builtin
def tensormap_smem_zero_first_warp(smem_u32_base, _semantic=None):
    """Zero 128 bytes of shared tensormap scratch using the first warp (32× u32).

    Control flow stays inside one PTX block: Python ``if`` on thread-id tensors
    does not propagate ``_semantic`` through tensor compares when this is marked
    ``@builtin``.
    """
    tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u32 %tx;
            .reg .u32 %addr;
            .reg .pred %p;
            mov.u32 %tx, %tid.x;
            mul.lo.u32 %addr, %tx, 4;
            add.u32 %addr, $1, %addr;
            setp.lt.u32 %p, %tx, 32;
            @%p st.shared.b32 [%addr], 0;
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,r",
        args=[smem_u32_base],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


# ============================================================================
# Fence primitives
# ============================================================================


@core.extern
def fence_async_shared(_semantic=None):
    """Fence to order async proxy (TMA) and generic proxy (ld/st) on shared memory.

    Required between:
    - shared memory store (generic) followed by TMA store from same location (async)
    - shared memory load (generic) followed by TMA load to same location (async)
    """
    return tl.inline_asm_elementwise(
        asm="""
        fence.proxy.async.shared::cta;
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def fence_proxy_async(_semantic=None):
    """General async proxy fence."""
    return tl.inline_asm_elementwise(
        asm="""
        fence.proxy.async;
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


# ============================================================================
# Prefetch TMA descriptor
# ============================================================================


@core.extern
def prefetch_tensormap(tmap_global_ptr, _semantic=None):
    """2D tensor-map prefetch into L2 (CUTE ``SM90_TMA_LOAD_2D::PREFETCH`` at coords 0,0)."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u32 %zc0;
            .reg .u32 %zc1;
            mov.u32 %zc0, 0;
            mov.u32 %zc1, 0;
            cp.async.bulk.prefetch.tensor.2d.L2.global [$1, {%zc0, %zc1}];
            mov.u32 $0, 0;
        }
        """,
        constraints="=r,l",
        args=[tmap_global_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def prefetch_tensor_2d_l2(tmap_global_ptr, coord_0, coord_1, _semantic=None):
    """``cp.async.bulk.prefetch.tensor.2d.L2.global`` — matches CUTE PREFETCH::copy."""
    return tl.inline_asm_elementwise(
        asm="""
        cp.async.bulk.prefetch.tensor.2d.L2.global [$1, {$2, $3}];
        mov.u32 $0, 0;
        """,
        constraints="=r,l,r,r",
        args=[tmap_global_ptr, coord_0, coord_1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


# ============================================================================
# Dynamic shared memory + address conversion (legacy / non-memdesc paths)
# ============================================================================


@core.extern
def get_dynamic_smem_pointer(_semantic=None):
    """Generic pointer to the start of dynamic shared memory (``dynamic_smem``)."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .extern .shared .align 128 .b8 dynamic_smem[];
            cvta.shared.to.generic.u64 $0, dynamic_smem;
        }
        """,
        constraints="=l",
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def cvta_generic_to_shared(generic_ptr, _semantic=None):
    """Convert generic address to shared (u64); for PTX that needs ``.shared::`` operands."""
    return tl.inline_asm_elementwise(
        asm="""
        cvta.to.shared.u64 $0, $1;
        """,
        constraints="=l,l",
        args=[generic_ptr],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


# ============================================================================
# Warp helpers (TMA / mbarrier issuing patterns)
# ============================================================================


@core.extern
def warp_idx(_semantic=None):
    """Warp index within the CTA (``threadIdx.x / 32``)."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .u32 %tx;
            mov.u32 %tx, %tid.x;
            shr.u32 $0, %tx, 5;
        }
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def named_barrier_arrive(barrier_id: core.constexpr, num_threads: core.constexpr, _semantic=None):
    """PTX ``bar.arrive``."""
    return tl.inline_asm_elementwise(
        asm=f"""
        bar.arrive {barrier_id.value}, {num_threads.value};
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def named_barrier_sync(barrier_id: core.constexpr, num_threads: core.constexpr, _semantic=None):
    """PTX ``bar.sync``."""
    return tl.inline_asm_elementwise(
        asm=f"""
        bar.sync {barrier_id.value}, {num_threads.value};
        mov.u32 $0, 0;
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def elect_one_sync(_semantic=None):
    """One elected lane per warp (``elect.sync``); all lanes must execute this."""
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p;
            elect.sync _|%p, 0xFFFFFFFF;
            selp.u32 $0, 1, 0, %p;
        }
        """,
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


__all__ = [
    "mbarrier_init",
    "mbarrier_expect_tx",
    "mbarrier_expect_transaction",
    "mbarrier_arrive",
    "mbarrier_arrive_expect_tx",
    "mbarrier_try_wait_parity",
    "mbarrier_wait_parity",
    "mbarrier_invalidate",
    "tma_load_2d",
    "tma_load_2d_pred",
    "tma_store_2d",
    "cp_async_bulk_global_to_shared",
    "cp_async_bulk_shared_to_global",
    "cp_async_bulk_commit_group",
    "cp_async_bulk_wait_group",
    "tensormap_replace_global_address",
    "tensormap_replace_global_dim",
    "tensormap_replace_global_stride",
    "tensormap_replace_rank",
    "tensormap_replace_box_dim",
    "tensormap_replace_element_stride",
    "tensormap_replace_elemtype",
    "tensormap_replace_interleave_layout",
    "tensormap_replace_swizzle_mode",
    "tensormap_replace_fill_mode",
    "tensormap_cp_fenceproxy",
    "bar_warp_sync",
    "cvta_global_to_generic_u64",
    "tensormap_smem_zero_first_warp",
    "_ptr_to_shared_u32",
    "fence_proxy_tensormap_acquire",
    "fence_async_shared",
    "fence_proxy_async",
    "prefetch_tensormap",
    "prefetch_tensor_2d_l2",
    "get_dynamic_smem_pointer",
    "cvta_generic_to_shared",
    "warp_idx",
    "named_barrier_arrive",
    "named_barrier_sync",
    "elect_one_sync",
]
