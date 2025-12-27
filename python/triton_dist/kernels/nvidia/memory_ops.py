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

import torch
import triton
import triton.language as tl
from triton_dist.language.extra.cuda.language_extra import tid, st, ld
from triton.language import core
from typing import Any


@core.extern
def _ptx_suffix_to_constraint(suffix: core.constexpr, _semantic=None):
    if suffix == "f64":
        return core.constexpr("d")
    elif suffix == "f32":
        return core.constexpr("f")
    elif suffix == "f16x2":
        return core.constexpr("r")
    elif suffix == "bf16x2":
        return core.constexpr("r")
    elif suffix == "b32":
        return core.constexpr("r")
    elif suffix == "s32":
        return core.constexpr("r")
    elif suffix == "u32":
        return core.constexpr("r")
    else:
        tl.static_assert(False, "unsupported dtype", _semantic=_semantic)


@core.extern
def _ptx_suffix_to_tl_type(suffix: core.constexpr, _semantic=None):
    if suffix == "u32":
        return tl.uint32
    elif suffix == "s32":
        return tl.int32
    elif suffix == "b32":
        return tl.int32
    elif suffix == "b64":
        return tl.int64
    elif suffix == "f16":
        return tl.float16
    elif suffix == "bf16":
        return tl.bfloat16
    elif suffix == "f16x2":
        return tl.int32
    elif suffix == "bf16x2":
        return tl.int32
    elif suffix == "f32":
        return tl.float32
    elif suffix == "f64":
        return tl.float64
    else:
        tl.static_assert(False, "unsupported suffix", _semantic=_semantic)


@core.extern
def load_v2(ptr, suffix: core.constexpr, _semantic=None):
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"ld.global.v2.{suffix.value} {{$0,$1}}, [$2];",
        constraints=(f"={c.value},={c.value},l"),
        args=[ptr],
        dtype=(val_type, val_type),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def load_v4(ptr, suffix: core.constexpr, _semantic=None):
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"ld.global.v4.{suffix.value} {{$0,$1,$2,$3}}, [$4];",
        constraints=(f"={c.value},={c.value},={c.value},={c.value},l"),
        args=[ptr],
        dtype=(val_type, val_type, val_type, val_type),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def store_v2(ptr, val0, val1, suffix: core.constexpr, _semantic=None):
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        st.global.v2.{suffix.value} [$1], {{$2,$3}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value},{c.value}"),  # no use output
        args=[
            tl.cast(ptr, dtype=tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(val0, dtype=tl.uint32, bitcast=True, _semantic=_semantic),
            tl.cast(val1, dtype=tl.uint32, bitcast=True, _semantic=_semantic),
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def store_v4(ptr, val0, val1, val2, val3, suffix: core.constexpr, _semantic=None):
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        st.global.v4.{suffix.value} [$1], {{$2,$3,$4,$5}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value},{c.value},{c.value},{c.value}"),  # no use output
        args=[
            tl.cast(ptr, dtype=tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(val0, dtype=tl.uint32, bitcast=True, _semantic=_semantic),
            tl.cast(val1, dtype=tl.uint32, bitcast=True, _semantic=_semantic),
            tl.cast(val2, dtype=tl.uint32, bitcast=True, _semantic=_semantic),
            tl.cast(val3, dtype=tl.uint32, bitcast=True, _semantic=_semantic)
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def zero_vec_f32(vec_size: tl.constexpr, _semantic=None):
    if vec_size == 1:
        return tl.inline_asm_elementwise(
            asm="mov.b32 $0, 0;",
            constraints=("=r"),
            args=[],
            dtype=tl.float32,
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    elif vec_size == 2:
        return tl.inline_asm_elementwise(
            asm="mov.b32 $0, 0; mov.b32 $1, 0;",
            constraints=("=r,=r"),
            args=[],
            dtype=(tl.float32, tl.float32),
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    elif vec_size == 4:
        return tl.inline_asm_elementwise(
            asm="mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;",
            constraints=("=r,=r,=r,=r"),
            args=[],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    elif vec_size == 8:
        return tl.inline_asm_elementwise(
            asm=
            "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0; mov.b32 $4, 0; mov.b32 $5, 0; mov.b32 $6, 0; mov.b32 $7, 0;",
            constraints=("=r,=r,=r,=r,=r,=r,=r,=r"),
            args=[],
            dtype=(tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32),
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )
    else:
        tl.static_assert(False, "unsupported vec_size", _semantic=_semantic)


@core.extern
def unpack_bf16x2_f32(v1, v2, v3, v4, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .b16 unpackres<8>;
            mov.b32 {unpackres0, unpackres1}, $8;
            cvt.f32.bf16 $0, unpackres0;
            cvt.f32.bf16 $1, unpackres1;
            mov.b32 {unpackres2, unpackres3}, $9;
            cvt.f32.bf16 $2, unpackres2;
            cvt.f32.bf16 $3, unpackres3;
            mov.b32 {unpackres4, unpackres5}, $10;
            cvt.f32.bf16 $4, unpackres4;
            cvt.f32.bf16 $5, unpackres5;
            mov.b32 {unpackres6, unpackres7}, $11;
            cvt.f32.bf16 $6, unpackres6;
            cvt.f32.bf16 $7, unpackres7;
        }
        """,
        constraints=("=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r"),
        args=[v1, v2, v3, v4],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def pack_f32_bf16x2(vec, _semantic=None):
    v1, v2, v3, v4, v5, v6, v7, v8 = vec
    return tl.inline_asm_elementwise(
        asm="""
        cvt.rn.bf16x2.f32 $0, $5, $4;
        cvt.rn.bf16x2.f32 $1, $7, $6;
        cvt.rn.bf16x2.f32 $2, $9, $8;
        cvt.rn.bf16x2.f32 $3, $11, $10;
        """,
        constraints=("=r,=r,=r,=r,r,r,r,r,r,r,r,r"),
        args=[v1, v2, v3, v4, v5, v6, v7, v8],
        dtype=(tl.float32, tl.float32, tl.float32, tl.float32),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@triton.jit
def copy_warp(
    dst_ptr,
    src_ptr,
    nbytes,
):
    WARP_SIZE = 32
    thread_idx = tid(0)
    lane_idx = thread_idx % WARP_SIZE

    src_ptr = tl.cast(src_ptr, dtype=tl.pointer_type(tl.uint8), bitcast=True)
    dst_ptr = tl.cast(dst_ptr, dtype=tl.pointer_type(tl.uint8), bitcast=True)

    for vec_idx in range(lane_idx, nbytes // 16, WARP_SIZE):
        t1, t2, t3, t4 = load_v4(src_ptr + vec_idx * 16, "b32")
        store_v4(dst_ptr + vec_idx * 16, t1, t2, t3, t4, "b32")

    src_ptr = src_ptr + nbytes // 16 * 16
    dst_ptr = dst_ptr + nbytes // 16 * 16
    nbytes = nbytes % 16

    if nbytes != 0:
        for vec_idx in range(lane_idx, nbytes // 8, WARP_SIZE):
            t1, t2 = load_v2(src_ptr + vec_idx * 8, "b32")
            store_v2(dst_ptr + vec_idx * 8, t1, t2, "b32")
        src_ptr = src_ptr + nbytes // 8 * 8
        dst_ptr = dst_ptr + nbytes // 8 * 8
        nbytes = nbytes % 8

    if nbytes != 0:
        for vec_idx in range(lane_idx, nbytes // 4, WARP_SIZE):
            t = ld(src_ptr.to(tl.pointer_type(tl.uint32)) + vec_idx)
            st(dst_ptr.to(tl.pointer_type(tl.uint32)) + vec_idx, t)
        src_ptr = src_ptr + nbytes // 4 * 4
        dst_ptr = dst_ptr + nbytes // 4 * 4
        nbytes = nbytes % 4

    if nbytes != 0:
        for vec_idx in range(lane_idx, nbytes // 2, WARP_SIZE):
            t = ld(src_ptr.to(tl.pointer_type(tl.uint16)) + vec_idx)
            st(dst_ptr.to(tl.pointer_type(tl.uint16)) + vec_idx, t)
        src_ptr = src_ptr + nbytes // 2 * 2
        dst_ptr = dst_ptr + nbytes // 2 * 2
        nbytes = nbytes % 2

    if nbytes != 0:
        t = ld(src_ptr.to(tl.pointer_type(tl.uint8)))
        st(dst_ptr.to(tl.pointer_type(tl.uint8)), t)


@triton.jit(do_not_specialize=["nelems"])
def copy_1d_tilewise_kernel(dst_ptr, src_ptr,  #
                            nelems,  #
                            BLOCK_SIZE: tl.constexpr,  #
                            ):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = nelems // BLOCK_SIZE

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        data = tl.load(src_ptr + offs, volatile=True)
        tl.store(dst_ptr + offs, data)

    if nelems % BLOCK_SIZE:
        if pid == NUM_COPY_SMS - 1:
            offs = num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < nelems
            data = tl.load(src_ptr + offs, mask=mask, volatile=True)
            tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit(do_not_specialize=["nelems"])
def copy_1d_persistent_kernel(dst_ptr, src_ptr,  #
                              nelems,  #
                              BLOCK_SIZE: tl.constexpr,  #
                              ):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = nelems // BLOCK_SIZE
    elem_size: tl.constexpr = tl.constexpr(dst_ptr.dtype.element_ty.primitive_bitwidth) // 8
    vec_size: tl.constexpr = tl.constexpr(16 // elem_size)

    if BLOCK_SIZE >= vec_size:
        tl.static_assert(BLOCK_SIZE % vec_size == 0, "BLOCK_SIZE must be divisible by vec_size")
        BLOCK_VEC: tl.constexpr = tl.constexpr(BLOCK_SIZE // vec_size)

        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_VEC) * vec_size
            v0, v1, v2, v3 = load_v4((src_ptr + offs).to(tl.pointer_type(tl.uint32)), suffix=tl.constexpr("u32"))
            store_v4((dst_ptr + offs).to(tl.pointer_type(tl.uint32)), v0, v1, v2, v3, suffix=tl.constexpr("u32"))

    else:
        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            data = tl.load(src_ptr + offs)
            tl.store(dst_ptr + offs, data)

    if nelems % BLOCK_SIZE:
        if pid == 0:
            offs = num_tiles * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offs < nelems
            data = tl.load(src_ptr + offs, mask=mask)
            tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit(do_not_specialize=["M"])
def copy_2d_persistent_kernel(
    dst_ptr,
    src_ptr,  #
    M,  #
    N,
    stride_m,
    stride_n,
    stride_dst_m,
    stride_dst_n,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        data = tl.load(src_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, mask=mask)
        tl.store(dst_ptr + offs_m[:, None] * stride_dst_m + offs_n[None, :] * stride_dst_n, data, mask=mask)


@triton.jit(do_not_specialize=["M"])
def copy_2d_kernel(
    dst_ptr,
    src_ptr,  #
    M,  #
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    stride_dst_m: tl.constexpr,
    stride_dst_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_tiles_n
    pid_n = pid % num_tiles_n
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m[:, None] & mask_n[None, :]
    data = tl.load(src_ptr + offs_m[:, None].to(tl.int64) * stride_m + offs_n[None, :] * stride_n, mask=mask)
    tl.store(dst_ptr + offs_m[:, None].to(tl.int64) * stride_dst_m + offs_n[None, :] * stride_dst_n, data, mask=mask)


@triton.jit(do_not_specialize=["nelems"])
def copy_2d_tma_kernel(
    dst_ptr,
    src_ptr,  #
    M,  #
    N: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    stride_dst_m: tl.constexpr,
    stride_dst_n: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    src_desc = tl.make_tensor_descriptor(
        src_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    dst_desc = tl.make_tensor_descriptor(
        dst_ptr,
        shape=[M, N],
        strides=[stride_m, stride_n],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        data = src_desc.load([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N])
        dst_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], data)


def copy_tensor(dst_tensor: torch.Tensor, src_tensor: torch.Tensor, num_sms: int = -1, eager=False, persistent=True):
    if dst_tensor.numel() == 0 or src_tensor.numel() == 0:
        return
    if eager:
        dst_tensor.copy_(src_tensor)
        return

    blocksizes = []
    choices = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_elems = 256 * 128

    assert dst_tensor.shape == src_tensor.shape, f"dst_tensor.shape: {dst_tensor.shape}, src_tensor.shape: {src_tensor.shape}"

    for dim in reversed(dst_tensor.shape):
        for i, choice in enumerate(choices):
            if total_elems % choice == 0 and dim >= choice:
                blocksizes.append(choice)
                total_elems //= choice
                break
    blocksizes = blocksizes[::-1]
    assert len(blocksizes) == dst_tensor.ndim

    if dst_tensor.ndim == 1:
        assert src_tensor.is_contiguous()
        assert dst_tensor.is_contiguous()
        BLOCK_SIZE_M = blocksizes[0]
        if persistent:
            assert num_sms > 0, "num_sms must be provided for persistent copy"
            copy_1d_persistent_kernel[(num_sms, )](dst_tensor, src_tensor, dst_tensor.shape[0], BLOCK_SIZE_M)
        else:
            grid = (triton.cdiv(dst_tensor.shape[0], BLOCK_SIZE_M), )
            copy_1d_tilewise_kernel[grid](dst_tensor, src_tensor, dst_tensor.shape[0], BLOCK_SIZE_M)
    elif dst_tensor.ndim == 2:
        BLOCK_SIZE_M = blocksizes[0]
        BLOCK_SIZE_N = blocksizes[1]
        if persistent:
            assert num_sms > 0, "num_sms must be provided for persistent copy"
            copy_2d_persistent_kernel[(num_sms, )](dst_tensor, src_tensor, dst_tensor.shape[0], dst_tensor.shape[1],
                                                   src_tensor.stride(0), src_tensor.stride(1), dst_tensor.stride(0),
                                                   dst_tensor.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)
        else:
            grid = (triton.cdiv(dst_tensor.shape[0], BLOCK_SIZE_M) * triton.cdiv(dst_tensor.shape[1], BLOCK_SIZE_N), )
            copy_2d_kernel[grid](dst_tensor, src_tensor, dst_tensor.shape[0], dst_tensor.shape[1], src_tensor.stride(0),
                                 src_tensor.stride(1), dst_tensor.stride(0), dst_tensor.stride(1), BLOCK_SIZE_M,
                                 BLOCK_SIZE_N)
    else:
        raise ValueError(f"Unsupported tensor dimension: {dst_tensor.ndim}")


@triton.jit
def fill_1d_persistent_kernel(
    dst_ptr,
    value,
    M,
    stride_m,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M)

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        offs = tile_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offs < M
        data = tl.full([BLOCK_SIZE_M], value, dtype=dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs, data, mask=mask)


@triton.jit
def fill_2d_persistent_kernel(
    dst_ptr,
    value,
    M,
    N,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        data = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], value, dtype=dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, data, mask=mask)


@triton.jit
def fill_3d_persistent_kernel(
    dst_ptr,
    value,
    B,
    M,
    N,
    stride_b,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_COPY_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for batch in range(B):
        for tile_id in range(pid, num_tiles, NUM_COPY_SMS):
            pid_m = tile_id // num_tiles_n
            pid_n = tile_id % num_tiles_n
            offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            mask_m = offs_m < M
            mask_n = offs_n < N
            mask = mask_m[:, None] & mask_n[None, :]
            data = tl.full([BLOCK_SIZE_M, BLOCK_SIZE_N], value, dtype=dst_ptr.dtype.element_ty)
            tl.store(dst_ptr + batch * stride_b + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n, data,
                     mask=mask)


def fill_tensor(tensor: torch.Tensor, value: Any, num_sms: int = -1, eager=False):
    if tensor.numel() == 0:
        return
    if eager:
        tensor.fill_(value)
        return

    if num_sms == -1:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count

    blocksizes = []
    choices = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_elems = 256 * 128

    for dim in reversed(tensor.shape):
        for i, choice in enumerate(choices):
            if total_elems % choice == 0 and dim >= choice:
                blocksizes.append(choice)
                total_elems //= choice
                break
    blocksizes = blocksizes[::-1]
    assert len(blocksizes) == tensor.ndim, f"blocksizes: {blocksizes}, tensor.shape: {tensor.shape}"

    if tensor.ndim == 1:
        BLOCK_SIZE = blocksizes[0]
        fill_1d_persistent_kernel[(num_sms, )](tensor, value, tensor.shape[0], tensor.stride(0), BLOCK_SIZE)
    elif tensor.ndim == 2:
        BLOCK_SIZE_M = blocksizes[0]
        BLOCK_SIZE_N = blocksizes[1]
        fill_2d_persistent_kernel[(num_sms, )](tensor, value, tensor.shape[0], tensor.shape[1], tensor.stride(0),
                                               tensor.stride(1), BLOCK_SIZE_M, BLOCK_SIZE_N)
    elif tensor.ndim == 3:
        BLOCK_SIZE_M = blocksizes[1]
        BLOCK_SIZE_N = blocksizes[2]
        fill_3d_persistent_kernel[(num_sms, )](tensor, value, tensor.shape[0], tensor.shape[1], tensor.shape[2],
                                               tensor.stride(0), tensor.stride(1), tensor.stride(2), BLOCK_SIZE_M,
                                               BLOCK_SIZE_N)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")


@triton.jit
def reduce_1d_persistent_kernel(
    src_ptr,
    dst_ptr,
    reduce_dim,
    stride_reduce,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(reduce_dim, BLOCK_SIZE)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < reduce_dim
        data = tl.load(src_ptr + offs * stride_reduce, mask=mask).to(tl.float32)
        accum = tl.sum(data).to(dst_ptr.dtype.element_ty)
        tl.atomic_add(dst_ptr, accum)


@triton.jit
def reduce_2d_persistent_kernel(
    src_ptr,
    dst_ptr,
    reduce_dim,
    M,
    stride_reduce,
    stride_m,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        offs = tile_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        mask = offs < M
        accum = tl.zeros((BLOCK_SIZE_M), dtype=tl.float32)
        for i in range(reduce_dim):
            data = tl.load(src_ptr + i * stride_reduce + offs * stride_m, mask=mask).to(tl.float32)
            accum += data
        accum = accum.to(dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs, accum, mask=mask)


@triton.jit
def reduce_3d_persistent_kernel(
    src_ptr,
    dst_ptr,
    reduce_dim,
    M,
    N,
    stride_reduce,
    stride_m,
    stride_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_SMS = tl.num_programs(0)
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_tiles_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_tiles_m * num_tiles_n

    for tile_id in range(pid, num_tiles, NUM_SMS):
        pid_m = tile_id // num_tiles_n
        pid_n = tile_id % num_tiles_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask = mask_m[:, None] & mask_n[None, :]
        accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for i in range(reduce_dim):
            data = tl.load(src_ptr + i * stride_reduce + offs_m[:, None] * stride_m + offs_n[None, :] * stride_n,
                           mask=mask).to(tl.float32)
            accum += data
        accum = accum.to(dst_ptr.dtype.element_ty)
        tl.store(dst_ptr + offs_m[:, None] * N + offs_n[None, :], accum, mask=mask)


def reduce_tensor(tensor: torch.Tensor, num_sms: int, dim=0, acc_dtype=torch.float32, eager=False):
    if tensor.numel() == 0:
        return torch.empty([1], dtype=tensor.dtype, device=tensor.device).fill_(0)
    assert acc_dtype == torch.float32
    if eager:
        return tensor.to(acc_dtype).sum(dim=dim).to(tensor.dtype)

    ndim = tensor.ndim
    dim = dim % ndim
    blocksizes = []
    choices = [256, 128, 64, 32, 16, 8, 4, 2, 1]
    total_elems = 256 * 128

    for d in reversed(tensor.shape):
        if d == dim:
            blocksizes.append(1)
            continue
        for i, choice in enumerate(choices):
            if total_elems % choice == 0 and d >= choice:
                blocksizes.append(choice)
                total_elems //= choice
                break
    blocksizes = blocksizes[::-1]
    assert len(blocksizes) == tensor.ndim

    if tensor.ndim == 1:
        reduce_dim = tensor.shape[dim]
        stride_reduce = tensor.stride(dim)
        output = torch.empty([1], dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE = 1024
        reduce_1d_persistent_kernel[(num_sms, )](tensor, output, reduce_dim, stride_reduce, BLOCK_SIZE)
    elif tensor.ndim == 2:
        reduce_dim = tensor.shape[dim]
        stride_reduce = tensor.stride(dim)
        M = tensor.shape[1 - dim]
        stride_m = tensor.stride(1 - dim)
        output = torch.empty([M], dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE_M = blocksizes[1 - dim]
        reduce_2d_persistent_kernel[(num_sms, )](tensor, output, reduce_dim, M, stride_reduce, stride_m, BLOCK_SIZE_M)
    elif tensor.ndim == 3:
        reduce_dim = tensor.shape[dim]
        stride_reduce = tensor.stride(dim)
        shapes = []
        strides = []
        blocks = []
        for i in range(ndim):
            if i != dim:
                shapes.append(tensor.shape[i])
                strides.append(tensor.stride(i))
                blocks.append(blocksizes[i])
        M, N = shapes
        stride_m, stride_n = strides
        output = torch.empty([M, N], dtype=tensor.dtype, device=tensor.device)
        BLOCK_SIZE_M, BLOCK_SIZE_N = blocks
        reduce_3d_persistent_kernel[(num_sms, )](tensor, output, reduce_dim, M, N, stride_reduce, stride_m, stride_n,
                                                 BLOCK_SIZE_M, BLOCK_SIZE_N)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")
    return output
