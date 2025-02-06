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

import triton
import triton.language as tl
from triton.language import core
from triton.language.extra.cuda.libdevice import ffs

# def patch_triton_module(func):
# func.__module__ = f"triton.{func.__module__}"
# return func


# @patch_triton_module
@core.extern
def __syncthreads(_builder=None):
    return core.inline_asm_elementwise(
        asm="""
        bar.sync 0;
        mov.u32 $0, 0;
        """,
        constraints="=r",  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=False,  # no optimize this!
        pack=1,
        _builder=_builder,
    )


# @patch_triton_module
@tl.core.extern
def __tid__(axis: core.constexpr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %tid.{axis.value};",
        constraints="=r",
        args=[],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def load_v4_u32(ptr, _builder=None):
    return tl.inline_asm_elementwise(
        asm="""
        ld.volatile.global.v4.u32 {$0,$1,$2,$3}, [$4];
        """,
        constraints=("=r,=r,=r,=r,l"),  # no use output, which is threadId.x
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def load_v2_b64(ptr, _builder=None):
    return tl.inline_asm_elementwise(
        asm="""
        ld.volatile.global.v2.b64 {$0,$1}, [$2];
        """,
        constraints=("=l,=l,l"),  # no use output, which is threadId.x
        args=[ptr],
        dtype=(tl.int64, tl.int64),
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def store_v2_u32(ptr, val0, val1, _builder=None):
    return tl.inline_asm_elementwise(
        asm="""
        st.volatile.global.v2.u32 [$1], {$2,$3};
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r,r"),  # no use output
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def multimem_st_b64(ptr, val0, _builder=None):
    return tl.inline_asm_elementwise(
        asm="""
        multimem.st.global.b64 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,l"),  # no use output
        args=[ptr, val0],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def multimem_st_b32(ptr, val0, _builder=None):
    return tl.inline_asm_elementwise(
        asm="""
        multimem.st.global.b32 [$1], $2;
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r"),  # no use output
        args=[ptr, val0],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def multimem_st_v2_b32(ptr, val0, val1, _builder=None):
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b64 r_combined;
        mov.b64 r_combined, {$2, $3};
        multimem.st.global.b64 [$1], r_combined;
        mov.u32 $0, 0;
        }""",
        constraints=("=r,l,r,r"),  # no use output
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


# TODO(houqi.1993) this is for reduce_scatter
@tl.core.extern
def multimem_ld_reduce(ptr, op, _builder=None):
    tl.static_assert(ptr.is_ptr(), "multimem_ld_reduce(ptr) expect ptr is a pointer_type")
    if ptr.dtype == tl.int32:
        return tl.inline_asm_elementwise(
            asm="""
            multimem.ld_reduce.global.b32 [$1], $2;
            mov.u32 $0, 0;
            """,
            constraints=("=r,l,r"),  # no use output
            args=[ptr],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
            _builder=_builder,
        )


@triton.jit
def tid(axis: core.constexpr):
    if axis == 0:
        return __tid__("x")
    elif axis == 1:
        return __tid__("y")
    elif axis == 2:
        return __tid__("z")
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


# @patch_triton_module
@tl.core.extern
def __ntid__(axis: core.constexpr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %ntid.{axis.value};",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def ntid(axis: core.constexpr):
    if axis == 0:
        return __ntid__("x")
    elif axis == 1:
        return __ntid__("y")
    elif axis == 2:
        return __ntid__("z")
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2")


# @patch_triton_module
@tl.core.extern
def red_release(barrier_ptr, value, scope: core.constexpr = core.constexpr("gpu"), _builder=None):
    tl.inline_asm_elementwise(
        asm=f"""{{
        mov.u32         $0, %tid.x;
        red.release.{scope.value}.global.add.s32 [$1], $2;
        }}""",
        constraints=("=r,"
                     "l,r"),  # no use output, which is threadId.x
        args=[barrier_ptr, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def arrive_inc(barrier_ptr, thread_idx, value, scope: core.constexpr):
    __syncthreads()
    if thread_idx == 0:
        red_release(barrier_ptr, value, scope)


# @patch_triton_module
@tl.core.extern
def arrive_inc_asm(barrier_ptr, thread_idx, value, scope: core.constexpr = "gpu", _builder=None):
    tl.inline_asm_elementwise(
        asm=f"""{{
        bar.sync        0;
        mov.u32         $0, %tid.x;
        setp.eq.s32     %p1, $2, 0;
        @%p1            red.release.{scope.value}.global.add.s32 [$1], $3;
        }}""",
        constraints=("=r,"
                     "l,r,r"),  # no use output
        args=[barrier_ptr, thread_idx, value],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


# @patch_triton_module
@tl.core.extern
def ld_acquire(barrier_ptr, scope: core.constexpr = "gpu", _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"""{{
        ld.global.acquire.{scope.value}.b32 $0, [$1];
        }}
        """,
        constraints=("=r,l"),
        args=[barrier_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def ld_u32_acquire(barrier_ptr, scope: core.constexpr = core.constexpr("gpu"), _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"""{{
        ld.global.acquire.{scope.value}.u32 $0, [$1];
        }}
        """,  # for older triton, scope maybe scope.value
        constraints=("=r,l"),
        args=[barrier_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


# @patch_triton_module
@tl.core.extern
def __atomic_add(
    ptr,
    value,
    scope: core.constexpr = "gpu",
    semantic: core.constexpr = "relaxed",
    _builder=None,
):
    if ptr.dtype.element_ty == tl.int32:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.int32,
                "value must be of the same dtype with ptr: int32",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"atom.{semantic.value}.{scope.value}.global.add.s32 $0, [$1], $2;",
            constraints=("=r,l,r"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.int32,
            _builder=_builder,
        )
    if ptr.dtype.element_ty == tl.uint32:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.uint32,
                "value must be of the same dtype with ptr: uint32",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"atom.{semantic.value}.{scope.value}.global.add.u32 $0, [$1], $2;",
            constraints=("=r,l,r"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.uint32,
            _builder=_builder,
        )
    elif ptr.dtype.element_ty == tl.int64:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.int64,
                "value must be of the same dtype with ptr: int64",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"atom.{semantic.value}.{scope.value}.global.add.s64 $0, [$1], $2;",
            constraints=("=l,l,l"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.int64,
            _builder=_builder,
        )
    elif ptr.dtype.element_ty == tl.uint64:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.uint64,
                "value must be of the same dtype with ptr: uint64",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"atom.{semantic.value}.{scope.value}.global.add.u64 $0, [$1], $2;",
            constraints=("=l,l,l"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.uint64,
            _builder=_builder,
        )
    else:
        raise ValueError("unsupported dtype")


@triton.jit
def atomic_add(barrier_ptr, value, scope: core.constexpr, semantic: core.constexpr):
    """custom atomic_add implementation using extern_elementwise

    :param scope: one of "gpu", "sys". default to "gpu"
    :param semantic: one of "release", "acquire", "relaxed", "acq_rel". default to "relaxed"
    :returns: the result of atomic_add
    :rtype: int
    """
    return __atomic_add(barrier_ptr, value, scope, semantic)


# @patch_triton_module
@tl.core.extern
def __atomic_store(
    ptr,
    value,
    scope: core.constexpr = "gpu",
    semantic: core.constexpr = "relaxed",
    _builder=None,
):
    if ptr.dtype.element_ty == tl.int32:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.int32,
                "value must be of the same dtype with ptr: int32",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"st.{semantic.value}.{scope.value}.global.s32 [$1], $2;\n mov.u32 $0, 0;",
            constraints=("=r,l,r"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.int32,
            _builder=_builder,
        )
    if ptr.dtype.element_ty == tl.uint32:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.uint32,
                "value must be of the same dtype with ptr: uint32",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"st.{semantic.value}.{scope.value}.global.u32 [$1], $2;\n mov.u32 $0, 0;",
            constraints=("=r,l,r"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.int32,
            _builder=_builder,
        )
    elif ptr.dtype.element_ty == tl.int64:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.int64,
                "value must be of the same dtype with ptr: int64",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"st.{semantic.value}.{scope.value}.global.s64 [$1], $2;\n mov.u32 $0, 0;",
            constraints=("=r,l,l"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.int32,
            _builder=_builder,
        )
    elif ptr.dtype.element_ty == tl.uint64:
        if not isinstance(value, tl.constexpr):
            tl.static_assert(
                value.dtype == tl.uint64,
                "value must be of the same dtype with ptr: uint64",
                _builder=_builder,
            )
        return tl.inline_asm_elementwise(
            asm=f"st.{semantic.value}.{scope.value}.global.u64 [$1], $2;\n mov.u32 $0, 0;",
            constraints=("=r,l,l"),
            args=[
                ptr,
                value,
            ],
            is_pure=False,
            pack=1,
            dtype=core.int32,
            _builder=_builder,
        )
    else:
        raise ValueError("unsupported dtype")


@triton.jit
def atomic_store(barrier_ptr, value, scope: core.constexpr, semantic: core.constexpr):
    """custom atomic_store implementation using extern_elementwise

    :param scope: one of "gpu", "sys". default to "gpu"
    :param semantic: one of "release", "acquire", "relaxed", "acq_rel". default to "relaxed"
    :returns: the result of atomic_store
    :rtype: int
    """
    return __atomic_store(barrier_ptr, value, scope, semantic)


@triton.jit
def wait_eq(barrier_ptr, thread_idx, value, scope: core.constexpr):
    if thread_idx == 0:
        while ld_acquire(barrier_ptr, scope) != value:
            pass
    __syncthreads()


# @patch_triton_module
@tl.core.extern
def __shfl_sync_with_mode_i32(
    mask: tl.core.uint32,
    value: tl.core.int32,
    delta: tl.core.uint32,
    mode: core.constexpr = "up",
    c: core.constexpr = 31,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"shfl.sync.{mode.value}.b32 $0, $1, $2, {c.value}, $3;",
        constraints="=r,r,r,r",
        args=[value, delta, mask],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def __shfl_sync_i32(mask, value, laneid):
    return __shfl_sync_with_mode_i32(mask, value, laneid, "idx", 31)


@triton.jit
def __shfl_up_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "up", 0)


@triton.jit
def __shfl_down_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "down", 0)


# @patch_triton_module
@tl.core.extern
def __ballot_sync(
    mask: tl.core.uint32,
    predicate: tl.core.int32,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm="{.reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2;}",
        constraints="=r,r,r",
        args=[predicate, mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


# @patch_triton_module
@tl.core.extern
def __ld(ptr, scope: core.constexpr = "gpu", nbit: core.constexpr = 32, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"ld.global.relaxed.{scope.value}.b{nbit.value} $0, [$1];",
        constraints="=r,l",
        args=[ptr],
        dtype=(tl.int32 if nbit.value == 32 else (tl.int64 if nbit.value == 64 else tl.int16)),
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def ld(ptr, scope: core.constexpr = "gpu"):
    if ptr.dtype == tl.pointer_type(tl.int32):
        return __ld(ptr, scope, 32)
    elif ptr.dtype == tl.pointer_type(tl.int64):
        return __ld(ptr, scope, 64)
    elif ptr.dtype == tl.pointer_type(tl.int16):
        return __ld(ptr, scope, 16)
    else:
        tl.static_assert(False, "unsupported dtype")


@tl.core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: core.constexpr,
    semantic: core.constexpr,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.cas.b32 $0, [$1], $2, $3;",
        constraints=("=r,l,r,r"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


__all__ = [
    "__syncthreads",
    "tid",
    "ntid",
    "wait_eq",
    "arrive_inc",
    "red_release",
    "ld_acquire",
    "ld_u32_acquire",
    "atomic_add",
    "atomic_store",
    "__shfl_sync_i32",
    "__shfl_up_sync_i32",
    "__shfl_down_sync_i32",
    "__ballot_sync",
    "ld",
    "ffs",
    "atomic_cas",
]
