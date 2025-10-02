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
'''
This file demonstrates how we use Triton-distributed to write custom communication kernels
that require many external calls for NVSHMEM and many inline asm functions to wrap up various PTX instructions.
And moreover, these communications kernels often require both intra-node and inter-node support, which add further
complexity to the kernel implementation.
'''

import triton
import triton.language as tl
from triton.language import core
import triton_dist.language as dl

from triton_dist.language.core import extern_call

################################## nvshmem ##################################
void_ptr = core.pointer_type(core.void)

@core.extern
def _putmem_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                 _semantic=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                f"nvshmem{'x' if SCOPE_SUFFIX.value else ''}_putmem{NBI.value}{SCOPE_SUFFIX.value}",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def putmem_nbi_warp(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_warp"), core.constexpr("_nbi"), _semantic=_semantic)

@core.extern
def remote_mc_ptr(team, ptr, _semantic=None):
    tl.static_assert(ptr.type.is_ptr(), "remote_mc_ptr(team, ptr) should be a pointer", _semantic=_semantic)
    return extern_call(
        "libnvshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic),
         tl.cast(ptr, void_ptr, _semantic=_semantic)],
        {(tl.int32, void_ptr): (
             "nvshmemx_mc_ptr", (ptr.type),  # of the same pointer type like ptr
         )},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "libnvshmem_device",
        "",
        [],
        {
            (): ("nvshmem_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )

#############################################################################

#################################### ptx ####################################
@core.extern
def __syncthreads(_semantic=None):
    return tl.tensor(_semantic.builder.create_barrier(), tl.void)

@core.extern
def _tid_wrapper(axis: core.constexpr, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                f"llvm.nvvm.read.ptx.sreg.tid.{axis.value}",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def tid(axis: core.constexpr, _semantic=None):
    if axis == 0:
        return _tid_wrapper(core.constexpr("x"), _semantic=_semantic)
    elif axis == 1:
        return _tid_wrapper(core.constexpr("y"), _semantic=_semantic)
    elif axis == 2:
        return _tid_wrapper(core.constexpr("z"), _semantic=_semantic)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2", _semantic=_semantic)


@core.extern
def multimem_st_b64(ptr, val0, _semantic=None):
    val_type: core.constexpr = tl.int64
    c: core.constexpr = core.constexpr("l")
    return tl.inline_asm_elementwise(
        asm=f"""
        multimem.st.global.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value}"),  # no use output
        args=[ptr, val0],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def pack_b32_v2(val0, val1, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.b64 $0, {$1, $2};",
        constraints=("=l,r,r"),
        args=[val0, val1],
        dtype=tl.uint64,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def st_v2_u32(ptr, val0, val1, _semantic=None):
    return tl.inline_asm_elementwise(
        asm=f"""
        st.volatile.global.v2.u32 [$1], {{$2,$3}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,r,r"),  # no use output
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def load_v4_u32(ptr, _semantic=None):
    val_type: core.constexpr = tl.uint32
    return tl.inline_asm_elementwise(
        asm=f"ld.volatile.global.v4.u32 {{$0,$1,$2,$3}}, [$4];",
        constraints=(f"=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(val_type, val_type, val_type, val_type),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def load_v2_b64(ptr, _semantic=None):
    val_type: core.constexpr = tl.int64
    c: core.constexpr = core.constexpr("l")
    return tl.inline_asm_elementwise(
        asm=f"ld.volatile.global.v2.b64 {{$0,$1}}, [$2];",
        constraints=(f"={c.value},={c.value},l"),
        args=[ptr],
        dtype=(val_type, val_type),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@triton.jit
def broadcast_naive_block(dst_ptr, src_ptr, nbytes):
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int8))
    dst_ptr = tl.cast(dst_ptr, tl.pointer_type(tl.int8))
    NVSHMEMX_TEAM_NODE = 2
    dst_mc_ptr = remote_mc_ptr(NVSHMEMX_TEAM_NODE, dst_ptr)
    num_int4 = nbytes // 16
    with dl.simt_exec_region() as (thread_idx, block_dim):
        for n in range(thread_idx, num_int4, block_dim):
            val0, val1 = load_v2_b64(src_ptr + 16 * n)
            multimem_st_b64(dst_mc_ptr + n * 16, val0)
            multimem_st_b64(dst_mc_ptr + n * 16 + 8, val1)


@triton.jit(do_not_specialize=["ll_flag"])
def _pack_ll_block(dest_ptr, src_ptr, num_ints, ll_flag, BLOCK_SIZE: tl.constexpr):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    nbytes: of the pre-LL-packed bytes.
    BLOCK_SIZE: count by ints, not bytes.
    """
    iters = tl.cdiv(num_ints, BLOCK_SIZE)
    src_ptr = tl.cast(src_ptr, dtype=tl.pi32_t)
    dest_ptr = tl.cast(dest_ptr, dtype=tl.pi32_t)
    for n in range(iters):
        src_offsets = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < num_ints
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        flags = tl.full((BLOCK_SIZE, ), ll_flag, tl.int32)
        dst = tl.interleave(src, flags)
        dest_offset = n * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
        dest_mask = dest_offset < num_ints * 2
        tl.store(dest_ptr + dest_offset, dst, mask=dest_mask)


@triton.jit
def _recv_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    num_ints: of the pre-LL-packed num_ints.
    """
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    # manual load per vec
    with dl.simt_exec_region() as (thread_idx, block_size):
        for n in range(thread_idx, num_ints // 2, block_size):
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
            while flag1 != ll_flag or flag2 != ll_flag:
                data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
            st_v2_u32(dest_ptr + n * 2, data1, data2)


@triton.jit(do_not_specialize=["ll_flag"])
def _recv_ll_and_multimem_st_ll_block(dest_ptr, src_ptr, num_ints, ll_flag):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    num_ints: of the pre-LL-packed num_ints.
    """
    src_ptr = tl.cast(src_ptr, tl.pointer_type(tl.int32))
    dest_ptr = tl.cast(dest_ptr, tl.pointer_type(tl.int32))
    NVSHMEMX_TEAM_NODE = 2
    dest_mc_ptr = remote_mc_ptr(NVSHMEMX_TEAM_NODE, dest_ptr)
    # manual load per vec
    with dl.simt_exec_region() as (thread_idx, block_size):
        for n in range(thread_idx, num_ints // 2, block_size):
            data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
            while flag1 != ll_flag or flag2 != ll_flag:
                data1, flag1, data2, flag2 = load_v4_u32(src_ptr + n * 4)
            multimem_st_b64(dest_mc_ptr + n * 4, pack_b32_v2(data1, flag1))
            multimem_st_b64(dest_mc_ptr + n * 4 + 2, pack_b32_v2(data2, flag2))



@triton.jit(do_not_specialize=["ll_flag"])
def _pack_ll_block(dest_ptr, src_ptr, num_ints, ll_flag, BLOCK_SIZE: tl.constexpr):
    """split src/dest outside of _recv_ll. this function is designed for a threadblock

    nbytes: of the pre-LL-packed bytes.
    BLOCK_SIZE: count by ints, not bytes.
    """
    iters = tl.cdiv(num_ints, BLOCK_SIZE)
    src_ptr = tl.cast(src_ptr, dtype=tl.pi32_t)
    dest_ptr = tl.cast(dest_ptr, dtype=tl.pi32_t)
    for n in range(iters):
        src_offsets = n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        src_mask = src_offsets < num_ints
        src = tl.load(src_ptr + src_offsets, mask=src_mask)
        flags = tl.full((BLOCK_SIZE, ), ll_flag, tl.int32)
        dst = tl.interleave(src, flags)
        dest_offset = n * BLOCK_SIZE * 2 + tl.arange(0, BLOCK_SIZE * 2)
        dest_mask = dest_offset < num_ints * 2
        tl.store(dest_ptr + dest_offset, dst, mask=dest_mask)
#############################################################################


@triton.jit(do_not_specialize=["rank", "signal_target"])
def forward_push_2d_ll_multimem_kernel(
    symm_ptr,
    bytes_per_rank,
    symm_ll_buffer,
    nnodes: tl.constexpr,
    signal_target,
):
    """
    pack_ll and nvshmem_putmem_nbi, then recv_ll and multimem.st
    """
    rank = my_pe()
    world_size = n_pes()
    local_world_size = world_size // nnodes
    local_rank = rank % local_world_size
    nid = rank // local_world_size

    pid = tl.program_id(0)
    peer_nid = pid // local_world_size
    peer_local_rank = pid % local_world_size
    num_ints = bytes_per_rank // 4
    thread_idx = tid(axis=0)

    ll_buffer_int8 = tl.cast(symm_ll_buffer, tl.pointer_type(tl.int8))
    symm_ptr = tl.cast(symm_ptr, tl.pointer_type(tl.int8))

    if peer_local_rank == local_rank:
        if nid != peer_nid:
            segment = peer_nid * local_world_size + local_rank
            _recv_ll_and_multimem_st_ll_block(
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
            _recv_ll_block(
                symm_ptr + segment * bytes_per_rank,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                num_ints,
                signal_target,
            )  # magic number here
        else:  # already has data. pack only
            _pack_ll_block(
                ll_buffer_int8 + rank * bytes_per_rank * 2,
                symm_ptr + rank * bytes_per_rank,
                num_ints,
                signal_target,
                2048,
            )  # magic number here
            __syncthreads()
            wid = thread_idx // 32
            # send
            if wid < nnodes and wid != nid:
                peer_to = wid * local_world_size + local_rank
                putmem_nbi_warp(
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    ll_buffer_int8 + rank * bytes_per_rank * 2,
                    bytes_per_rank * 2,
                    peer_to,
                )  # write and tell peer remote that remote copy is done

            segment = peer_nid * local_world_size + local_rank
            broadcast_naive_block(
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                ll_buffer_int8 + segment * bytes_per_rank * 2,
                bytes_per_rank * 2,
            )
    else:
        segment_recv_local = peer_nid * local_world_size + peer_local_rank
        _recv_ll_block(
            symm_ptr + segment_recv_local * bytes_per_rank,
            ll_buffer_int8 + segment_recv_local * bytes_per_rank * 2,
            num_ints,
            signal_target,
        )  # magic number here

