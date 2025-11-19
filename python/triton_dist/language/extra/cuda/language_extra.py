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
from triton_dist.language import vector, make_vector


@core.extern
def _int_constraint(bitwidth: core.constexpr, _semantic=None):
    # https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints
    # PTX has no constraint for int8. use "r"
    if bitwidth.value == 128:
        return core.constexpr("q")
    if bitwidth.value == 64:
        return core.constexpr("l")
    elif bitwidth.value == 32:
        return core.constexpr("r")
    elif bitwidth.value == 16:
        return core.constexpr("h")
    elif bitwidth.value == 8:
        return core.constexpr("r")
    else:
        tl.static_assert(False, "unsupported dtype", _semantic=_semantic)


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


@core.constexpr_function
def tl_type_to_ptx_suffix(dtype: core.constexpr):
    if dtype == tl.uint32:
        return "u32"
    elif dtype == tl.int32:
        return "s32"
    elif dtype == tl.int64:
        return "b64"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


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
def __syncthreads(_semantic=None):
    return tl.tensor(_semantic.builder.create_barrier(), tl.void)


@core.extern
def __fence(scope: core.constexpr = core.constexpr("gpu"), _semantic=None):
    return core.inline_asm_elementwise(
        asm=f"""
        fence.sc.{scope.value};
        """,
        constraints="=r",  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=False,  # no optimize this!
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def tma_sync(N: core.constexpr = core.constexpr(0), _semantic=None):
    return core.inline_asm_elementwise(
        asm=f"""
        cp.async.bulk.wait_group {N.value};
        """,
        constraints="=r",  # force have a return value, even not used.
        args=[],
        dtype=tl.uint32,
        is_pure=False,  # no optimize this!
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def _load_v4_impl(ptr, suffix: core.constexpr, scope: core.constexpr = "", semantic: core.constexpr = "",
                  _semantic=None):
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    scope = core._unwrap_if_constexpr(scope)
    semantic = core._unwrap_if_constexpr(semantic)
    if scope != "":
        scope = f".{scope}"
    if semantic != "":
        semantic = f".{semantic}"
    return tl.inline_asm_elementwise(
        asm=f"ld.global{semantic}{scope}.v4.{suffix.value} {{$0,$1,$2,$3}}, [$4];",
        constraints=(f"={c.value},={c.value},={c.value},={c.value},l"),
        args=[ptr],
        dtype=(val_type, val_type, val_type, val_type),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def _load_v2_impl(ptr, suffix: core.constexpr, _semantic=None):
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"ld.volatile.global.v2.{suffix.value} {{$0,$1}}, [$2];",
        constraints=(f"={c.value},={c.value},l"),
        args=[ptr],
        dtype=(val_type, val_type),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def load_v4_u32(ptr, _semantic=None):
    return _load_v4_impl(ptr, core.constexpr("u32"), _semantic=_semantic)


@core.extern
def load_v4_b32(ptr, scope: core.constexpr = "", semantic: core.constexpr = "", _semantic=None):
    return _load_v4_impl(ptr, core.constexpr("b32"), scope, semantic, _semantic=_semantic)


@core.extern
def load_v4_s32(ptr, _semantic=None):
    return _load_v4_impl(ptr, core.constexpr("s32"), _semantic=_semantic)


@core.extern
def load_v2_b64(ptr, _semantic=None):
    return _load_v2_impl(ptr, core.constexpr("b64"), _semantic=_semantic)


@core.extern
def _store_v4_impl(ptr, val0, val1, val2, val3, suffix: core.constexpr, scope="gpu", semantic="relaxed",
                   _semantic=None):
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    scope = core._unwrap_if_constexpr(scope)
    semantic = core._unwrap_if_constexpr(semantic)
    if scope != "":
        scope = f".{scope}"
    if semantic != "":
        semantic = f".{semantic}"
    return tl.inline_asm_elementwise(
        asm=f"""
        st.global{semantic}{scope}.v4.{suffix.value} [$1], {{$2,$3,$4,$5}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value},{c.value},{c.value},{c.value}"),  # no use output
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def _store_v2_impl(ptr, val0, val1, suffix: core.constexpr, _semantic=None):
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        st.volatile.global.v2.{suffix.value} [$1], {{$2,$3}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value},{c.value}"),  # no use output
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def st_v4_u32(ptr, val0, val1, val2, val3, _semantic=None):
    return _store_v4_impl(tl.cast(ptr, tl.pi32_t, _semantic=_semantic), val0, val1, val2, val3, core.constexpr("u32"),
                          _semantic=_semantic)


@core.extern
def st_v4_b32(ptr, val0, val1, val2, val3, scope="", semantic="", _semantic=None):
    return _store_v4_impl(tl.cast(ptr, tl.pi32_t, _semantic=_semantic), val0, val1, val2, val3, core.constexpr("b32"),
                          scope=scope, semantic=semantic, _semantic=_semantic)


@core.extern
def st_v2_u32(ptr, val0, val1, _semantic=None):
    return _store_v2_impl(tl.cast(ptr, tl.pi32_t, _semantic=_semantic), val0, val1, core.constexpr("u32"),
                          _semantic=_semantic)


@core.extern
def _multimem_st_impl(ptr, val0, suffix: core.constexpr, _semantic=None):
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    c: core.constexpr = _int_constraint(core.constexpr(val_type.primitive_bitwidth), _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        multimem.st.global.{suffix.value} [$1], $2;
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
def multimem_st_b64(ptr, val0, _semantic=None):
    return _multimem_st_impl(ptr, val0, core.constexpr("b64"), _semantic=_semantic)


@core.extern
def multimem_st_b32(ptr, val0, _semantic=None):
    return _multimem_st_impl(ptr, val0, core.constexpr("b32"), _semantic=_semantic)


@core.extern
def _multimem_st_v2_impl(ptr, val0, val1, suffix: core.constexpr, _semantic=None):
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        multimem.st.global.v2.{suffix.value} [$1], {{$2, $3}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value},{c.value}"),  # no use output
        args=[ptr, val0, val1],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def multimem_st_v2(ptr, val0, val1, _semantic=None):
    # it seems that multimem.st does not support v2, when doc say so: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-multimem
    tl.static_assert(val0.dtype.primitive_bitwidth == 32, _semantic=_semantic)
    tl.static_assert(val1.dtype.primitive_bitwidth == 32, _semantic=_semantic)
    if ptr.dtype.element_ty == tl.float32:
        return _multimem_st_v2_impl(ptr, val0, val1, core.constexpr("f32"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.bfloat16:
        return _multimem_st_v2_impl(ptr, val0, val1, core.constexpr("bf16x2"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.float16:
        return _multimem_st_v2_impl(ptr, val0, val1, core.constexpr("f16x2"), _semantic=_semantic)
    else:
        tl.static_assert(False, "multimem.st.v2 only support f32 and fp16 and bf16", _semantic=_semantic)


@core.extern
def _multimem_st_v4_impl(ptr, val0, val1, val2, val3, suffix: core.constexpr, _semantic=None):
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        multimem.st.global.v4.{suffix.value} [$1], {{$2, $3, $4, $5}};
        mov.u32 $0, 0;
        """,
        constraints=(f"=r,l,{c.value},{c.value},{c.value},{c.value}"),  # no use output
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def multimem_st_v4(ptr, val0, val1, val2, val3, _semantic=None):
    tl.static_assert(ptr.dtype.is_ptr(), _semantic=_semantic)
    tl.static_assert(val0.dtype.primitive_bitwidth == 32, _semantic=_semantic)
    tl.static_assert(val1.dtype.primitive_bitwidth == 32, _semantic=_semantic)
    tl.static_assert(val2.dtype.primitive_bitwidth == 32, _semantic=_semantic)
    tl.static_assert(val3.dtype.primitive_bitwidth == 32, _semantic=_semantic)
    if ptr.dtype.element_ty == tl.float32:
        return _multimem_st_v4_impl(ptr, val0, val1, val2, val3, core.constexpr("f32"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.bfloat16:
        return _multimem_st_v4_impl(ptr, val0, val1, val2, val3, core.constexpr("bf16x2"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.float16:
        return _multimem_st_v4_impl(ptr, val0, val1, val2, val3, core.constexpr("f16x2"), _semantic=_semantic)
    else:
        tl.static_assert(False, "unsupported type", _semantic=_semantic)


# TODO(houqi.1993) multimem does not work with @%p.
@triton.jit
def multimem_st_p_b32(ptr, val, mask):
    tl.static_assert(val.dtype.primitive_bitwidth == 32)
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $3, 1;
            @%p0 multimem.st.global.b32 [$1], $2;
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,r,r"),
        args=[ptr.to(tl.pointer_type(tl.uint32)), val,
              tl.cast(mask, tl.uint32)],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )


@core.extern
def _multimem_ld_reduce_p_128bit(ptr, mask, suffix: core.constexpr, _semantic=None):
    # Use with caution: @p and multicast instructions are incompatible;
    # Mask may not control instruction execution correctly with imperfect tiling.
    # May be replaced or deprecated in the future. TODO(lsy.314)
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"""
        {{
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @%p0 multimem.ld_reduce.acquire.sys.global.add.v4.{suffix.value} {{$0,$1,$2,$3}}, [$4];
        }}
        """,
        constraints=(f"={c.value},={c.value},={c.value},={c.value},l,r"),
        args=[ptr, mask.to(tl.int32, _semantic=_semantic)],
        dtype=[val_type, val_type, val_type, val_type],
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


# TODO(houqi.1993) there is a BUG with ptxas that multimem does not work with @%p. so mask does not work actually. use with care.
@core.extern
def multimem_ld_reduce_p_v4(ptr, mask, _semantic=None):
    if ptr.dtype.element_ty == tl.bfloat16:
        return _multimem_ld_reduce_p_128bit(ptr, mask, core.constexpr("bf16x2"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.float16:
        return _multimem_ld_reduce_p_128bit(ptr, mask, core.constexpr("f16x2"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.float32:
        return _multimem_ld_reduce_p_128bit(ptr, mask, core.constexpr("f32"), _semantic=_semantic)
    else:
        tl.static_assert(False, "Unsupported dtype, only fp16/bf16/fp32 is supported", _semantic=_semantic)


@core.extern
def _multimem_ld_reduce_128bit(ptr, acc_dtype: core.constexpr, suffix: core.constexpr, _semantic=None):
    # Use with caution: @p and multicast instructions are incompatible;
    # Mask may not control instruction execution correctly with imperfect tiling.
    # May be replaced or deprecated in the future. TODO(lsy.314)
    c: core.constexpr = _ptx_suffix_to_constraint(suffix, _semantic=_semantic)
    val_type: core.constexpr = _ptx_suffix_to_tl_type(suffix, _semantic=_semantic)
    acc_prec = ""
    if acc_dtype == tl.float32:
        acc_prec = ".acc::f32"
    else:
        tl.static_assert(False, "Unsupported dtype, acc::f16 is used for fp8", _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm=f"multimem.ld_reduce.global.add{acc_prec}.v4.{suffix.value} {{$0,$1,$2,$3}}, [$4];",
        constraints=(f"={c.value},={c.value},={c.value},={c.value},l"),
        args=[ptr],
        dtype=[val_type, val_type, val_type, val_type],
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def multimem_ld_reduce_v4(ptr, acc_dtype=None, _semantic=None):
    """
    Load data from global memory with PTX instructions multimem.ld_reduce
    Args:
        ptr: Pointer to the global memory
        acc_prec: Accumulation precision, can be "auto", ".acc::f32" or "". for "auto", prefer high precision.
    Returns:
        Loaded data of 32-bit register tuple (val0, val1, val2, val3)
    """
    tl.static_assert(ptr.dtype.element_ty.kind() == core.dtype.KIND.FLOATING, _semantic=_semantic)
    if acc_dtype is not None:
        tl.static_assert(acc_dtype == tl.float32 or acc_dtype == tl.float16,
                         "Unsupported acc dtype, only fp16/fp32 is supported", _semantic=_semantic)
    if ptr.dtype.element_ty == tl.bfloat16:
        return _multimem_ld_reduce_128bit(ptr, core.float32 if acc_dtype is None else acc_dtype,
                                          core.constexpr("bf16x2"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.float16:
        return _multimem_ld_reduce_128bit(ptr, core.float32 if acc_dtype is None else acc_dtype,
                                          core.constexpr("f16x2"), _semantic=_semantic)
    elif ptr.dtype.element_ty == tl.float32:
        return _multimem_ld_reduce_128bit(ptr, acc_dtype, core.constexpr("f32"), _semantic=_semantic)
    else:
        tl.static_assert(False, "Unsupported dtype, only fp16/bf16/fp32 is supported", _semantic=_semantic)


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
def laneid(_semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                "llvm.nvvm.read.ptx.sreg.laneid",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def _ntid_wrapper(axis: core.constexpr, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                f"llvm.nvvm.read.ptx.sreg.ntid.{axis.value}",
                core.dtype("int32"),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ntid(axis: core.constexpr, _semantic=None):
    if axis == 0:
        return _ntid_wrapper(core.constexpr("x"), _semantic=_semantic)
    elif axis == 1:
        return _ntid_wrapper(core.constexpr("y"), _semantic=_semantic)
    elif axis == 2:
        return _ntid_wrapper(core.constexpr("z"), _semantic=_semantic)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2", _semantic=_semantic)


# @patch_triton_module
@core.extern
def red_release(barrier_ptr, value, scope: core.constexpr = core.constexpr("gpu"), _semantic=None):
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
        _semantic=_semantic,
    )


@triton.jit
def arrive_inc(barrier_ptr, thread_idx, value, scope: core.constexpr):
    __syncthreads()
    if thread_idx == 0:
        red_release(barrier_ptr, value, scope)


# @patch_triton_module
@core.extern
def arrive_inc_asm(barrier_ptr, thread_idx, value, scope: core.constexpr = "gpu", _semantic=None):
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
        _semantic=_semantic,
    )


@core.extern
def ld(
    ptr,
    scope: core.constexpr = "gpu",
    semantic: core.constexpr = "relaxed",
    _semantic=None,
):
    tl.static_assert(ptr.dtype.is_ptr(), "ld(ptr, scope) should be a pointer", _semantic=_semantic)
    if isinstance(scope, core.constexpr):
        scope = scope.value
    tl.static_assert(scope in ["gpu", "sys"], "scope should be gpu or sys", _semantic=_semantic)
    if isinstance(semantic, core.constexpr):
        semantic = semantic.value
    tl.static_assert(
        semantic in ["relaxed", "acquire"],
        "semantic should be relaxed or acquire",
        _semantic=_semantic,
    )
    element_ty: tl.dtype = ptr.dtype.element_ty
    constraint = _int_constraint(core.constexpr(element_ty.primitive_bitwidth), _semantic=_semantic)
    if semantic != "":
        semantic = f".{semantic}"
    if scope != "":
        scope = f".{scope}"
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld
    return tl.inline_asm_elementwise(
        asm=f"ld.global{semantic}{scope}.b{element_ty.primitive_bitwidth} $0, [$1];",
        constraints=f"={constraint.value},l",
        args=[ptr],
        dtype=ptr.dtype.element_ty,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def ld_vector(
    ptr,
    vec_size: core.constexpr = 1,
    scope: core.constexpr = "",
    semantic: core.constexpr = "",
    _semantic=None,
):
    assert isinstance(vec_size, tl.constexpr)

    elems_bits = ptr.dtype.element_ty.primitive_bitwidth
    total_bits: tl.constexpr = elems_bits * vec_size
    if total_bits % 128 == 0:
        # currently user need to guarantee the alignment of `ptr`.
        ptr_i32 = tl.cast(ptr, tl.pi32_t, _semantic=_semantic)
        vals_i32 = []
        num_iters = total_bits // 128
        for idx in range(num_iters):
            val0, val1, val2, val3 = load_v4_b32(tl.add(ptr_i32, idx * 4, _semantic=_semantic), scope=scope,
                                                 semantic=semantic, _semantic=_semantic)
            vals_i32 += [val0, val1, val2, val3]
        return make_vector(vals_i32, _semantic=_semantic).recast(ptr.dtype.element_ty, _semantic=_semantic)

    else:
        ret = []
        for i in range(vec_size):
            ret.append(ld(tl.add(ptr, i, _semantic=_semantic), scope=scope, semantic=semantic, _semantic=_semantic))
        return make_vector(ret, _semantic=_semantic)


@core.extern
def st_vector(
        ptr,
        vec,
        scope: core.constexpr = core.constexpr(""),
        semantic: core.constexpr = core.constexpr(""),
        _semantic=None,
):
    assert ptr.dtype.element_ty == vec.type.elem_type, f"ptr.dtype.element_ty {ptr.dtype.element_ty} != vec.type.elem_type {vec.type.elem_type}"
    assert isinstance(vec, vector)
    total_bits = vec.type.vector_nbits
    if total_bits % 128 == 0:
        vec_i32 = vec.recast(tl.uint32, _semantic=_semantic)
        ptr_i32 = tl.cast(ptr, tl.pointer_type(tl.uint32), _semantic=_semantic)
        vals_i32 = [tl.cast(val, tl.uint32, _semantic=_semantic) for val in vec_i32]
        num_iters = total_bits // 128
        for idx in range(num_iters):
            st_v4_b32(tl.add(ptr_i32, idx * 4, _semantic=_semantic), *vals_i32[idx * 4:(idx + 1) * 4], scope=scope,
                      semantic=semantic, _semantic=_semantic)
    else:
        for idx, v in enumerate(vec):
            st(tl.add(ptr, idx, _semantic=_semantic), v, scope=scope, semantic=semantic, _semantic=_semantic)


@core.extern
def ld_b32(ptr, _semantic=None):
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int32(),
        "ld_b32(ptr) argument 0 `ptr` should be a pointer of int type",
        _semantic=_semantic,
    )
    return ld(ptr, scope="gpu", semantic="relaxed", _semantic=_semantic)


@core.extern
def ld_acquire(ptr, scope: core.constexpr = "gpu", _semantic=None):
    return ld(ptr, scope, "acquire", _semantic=_semantic)


@core.extern
def st(
        ptr,
        val,
        scope: core.constexpr = core.constexpr("gpu"),
        semantic: core.constexpr = core.constexpr("relaxed"),
        _semantic=None,
):
    if isinstance(val, vector):
        st_vector(ptr, val, scope, semantic, _semantic=_semantic)
    else:
        int_dtype = core.get_int_dtype(ptr.dtype.element_ty.primitive_bitwidth, signed=False)
        if isinstance(val, core.constexpr) and ptr.dtype.element_ty.is_int():
            val = tl.cast(val.value, int_dtype, _semantic=_semantic)
        elif val.dtype.primitive_bitwidth == ptr.dtype.element_ty.primitive_bitwidth:
            val = tl.cast(val, int_dtype, bitcast=True, _semantic=_semantic)
        ptr = tl.cast(ptr, core.pointer_type(int_dtype), _semantic=_semantic)
        tl.static_assert(
            ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int(),
            "st(ptr, val) argument 0 `ptr` should be a pointer of int type",
            _semantic=_semantic,
        )
        dtype = ptr.dtype.element_ty
        if isinstance(val, core.constexpr):
            val = tl.cast(val.value, dtype, _semantic=_semantic)
        else:
            val = tl.cast(val, dtype, _semantic=_semantic)

        tl.static_assert(val.dtype.is_int(), "st(ptr, val) argument `val` should be of int type", _semantic=_semantic)

        if isinstance(scope, core.constexpr):
            scope = scope.value
        if isinstance(semantic, core.constexpr):
            semantic = semantic.value
        tl.static_assert(scope in ["gpu", "sys"], "scope should be gpu or sys", _semantic=_semantic)
        tl.static_assert(
            semantic in ["relaxed", "release"],
            "semantic should be relaxed or release",
            _semantic=_semantic,
        )
        constraint = _int_constraint(core.constexpr(dtype.primitive_bitwidth), _semantic=_semantic)
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st
        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation
        if semantic != "":
            semantic = f".{semantic}"
        if scope != "":
            scope = f".{scope}"
        return tl.inline_asm_elementwise(
            asm=f"""
            st.global{semantic}{scope}.b{dtype.primitive_bitwidth} [$1], $2;
            mov.u32 $0, 0;
            """,
            constraints=(f"=r,l,{constraint.value}"),  # no use output
            args=[ptr, val],
            dtype=tl.int32,  # never used
            is_pure=False,
            pack=1,
            _semantic=_semantic,
        )


@core.extern
def st_b32(ptr, val0, _semantic=None):
    return st(ptr, val0, scope="gpu", semantic="relaxed", _semantic=_semantic)


@core.extern
def atomic_add(
    ptr,
    value,
    scope: core.constexpr = "gpu",
    semantic: core.constexpr = "relaxed",
    _semantic=None,
):
    """custom atomic_add implementation using extern_elementwise

    :param scope: one of "gpu", "sys". default to "gpu"
    :param semantic: one of "release", "acquire", "relaxed", "acq_rel". default to "relaxed"
    :returns: the result of atomic_add
    :rtype: int
    """
    tl.static_assert(
        ptr.dtype.is_ptr() and ptr.dtype.element_ty.is_int(),
        "ptr must be a pointer of int",  # PTX support atom add float, but tl.inline_asm_elementwise does not like it
        _semantic=_semantic,
    )
    if isinstance(scope, core.constexpr):
        scope = scope.value
    tl.static_assert(scope in ["gpu", "sys"], "scope should be gpu or sys", _semantic=_semantic)
    if isinstance(semantic, core.constexpr):
        semantic = semantic.value
    tl.static_assert(
        semantic in ["release", "acquire", "relaxed", "acq_rel"],
        "semantic should be release, acquire, relaxed or acq_rel",
        _semantic=_semantic,
    )
    constraint = _int_constraint(core.constexpr(ptr.dtype.element_ty.primitive_bitwidth), _semantic=_semantic).value

    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom
    # .add requires .u32 or .s32 or .u64 or .f64 or f16 or f16x2 or .f32 or .bf16 or .bf16x2 type for instruction 'atom'
    return tl.inline_asm_elementwise(
        asm=
        f"atom.{semantic}.{scope}.global.add.{'s' if ptr.dtype.element_ty.is_int_signed() else 'u'}{ptr.dtype.element_ty.primitive_bitwidth} $0, [$1], $2;",
        constraints=(f"={constraint},l,{constraint}"),
        args=[
            ptr,
            value,
        ],
        is_pure=False,
        pack=1,
        dtype=ptr.dtype.element_ty,
        _semantic=_semantic,
    )


@triton.jit
def atomic_add_per_warp(barrier_ptr, value, scope: core.constexpr, semantic: core.constexpr):
    _laneid = laneid()
    x = tl.cast(0, barrier_ptr.dtype.element_ty)
    if _laneid == 0:
        x = atomic_add(barrier_ptr, value, scope, semantic)
    return __shfl_sync_i32(0xFFFFFFFF, x, 0)


@triton.jit
def wait_eq(barrier_ptr, thread_idx, value, scope: core.constexpr):
    if thread_idx == 0:
        while ld_acquire(barrier_ptr, scope) != value:
            pass
    __syncthreads()


@core.extern
def __shfl_sync_with_mode_i32(
    mask,
    value,
    delta,
    mode: core.constexpr = "up",
    c: core.constexpr = 31,
    _semantic=None,
):
    tl.static_assert(value.dtype == tl.int32 or value.dtype == tl.uint32,
                     "__shfl_sync_i32 only support int32 or uint32", _semantic=_semantic)
    # refer to https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync
    return tl.inline_asm_elementwise(
        asm=f"shfl.sync.{mode.value}.b32 $0, $1, $2, {c.value}, $3;",
        constraints="=r,r,r,r",
        args=[value, delta, mask],
        dtype=value.dtype,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@triton.jit
def __shfl_sync_i32(mask, value, laneid):
    return __shfl_sync_with_mode_i32(mask, value, laneid, "idx", 31)


@triton.jit
def __shfl_up_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "up", 0)


@triton.jit
def __shfl_down_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "down", 31)


@triton.jit
def __shfl_xor_sync_i32(mask, value, delta):
    return __shfl_sync_with_mode_i32(mask, value, delta, "bfly", 31)


# @patch_triton_module
@core.extern
def __ballot_sync(
    mask,
    predicate,
    _semantic=None,
):
    return tl.inline_asm_elementwise(
        asm="{.reg .pred p; setp.ne.b32 p, $1, 0; vote.sync.ballot.b32 $0, p, $2;}",
        constraints="=r,r,r",
        args=[predicate, mask],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: core.constexpr,
    semantic: core.constexpr,
    _semantic=None,
):
    constraint = _int_constraint(core.constexpr(ptr.dtype.element_ty.primitive_bitwidth), _semantic=_semantic).value
    return tl.inline_asm_elementwise(
        asm=
        f"atom.{semantic.value}.{scope.value}.global.cas.b{ptr.dtype.element_ty.primitive_bitwidth} $0, [$1], $2, $3;",
        constraints=(f"={constraint},l,{constraint},{constraint}"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=ptr.dtype.element_ty,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def globaltimer_lo(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %globaltimer_lo;",
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def globaltimer_hi(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %globaltimer_hi;",
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def globaltimer(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.u64 $0, %globaltimer;",
        constraints=("=l"),
        args=[],
        dtype=tl.uint64,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def smid(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %smid;",
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def membar(scope: core.constexpr = core.constexpr("cta"), _semantic=None):
    return tl.inline_asm_elementwise(
        asm=f"""
        membar.{scope.value};
        mov.u32 $0, 0;
        """,
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def fence(
        semantic: core.constexpr = core.constexpr("sc"), scope: core.constexpr = core.constexpr("gpu"), _semantic=None):
    return tl.inline_asm_elementwise(
        asm=f"""
        fence.{core._unwrap_if_constexpr(semantic)}.{core._unwrap_if_constexpr(scope)};
        mov.u32 $0, 0;
        """,
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
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
def pack(src: vector, dst_type, _semantic=None):
    assert isinstance(src, vector)
    dst_nbits = dst_type.primitive_bitwidth
    src_elem_dtype = src.type.elem_type
    assert src.type.vector_nbits == dst_type.primitive_bitwidth, f"src.type.vector_nbits {src.type.vector_nbits} != dst_type.primitive_bitwidth {dst_type.primitive_bitwidth}"
    src_elem_int_ty = core.get_int_dtype(src_elem_dtype.primitive_bitwidth, False)
    src = src.recast(src_elem_int_ty, _semantic=_semantic)

    dst_constraint = _int_constraint(core.constexpr(dst_nbits), _semantic=_semantic).value
    src_constraint = _int_constraint(core.constexpr(src_elem_dtype.primitive_bitwidth), _semantic=_semantic).value

    src_operands = [f'${i + 1}' for i in range(src.type.vec_size)]
    src_constraints = [src_constraint for i in range(src.type.vec_size)]
    asm = f"mov.b{dst_nbits} $0, {{{', '.join(src_operands)}}};"
    constraints = f"={dst_constraint},{', '.join(src_constraints)}"
    return tl.inline_asm_elementwise(
        asm=asm,
        constraints=constraints,
        args=src.values,
        dtype=dst_type,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def unpack(src, dst_type, _semantic=None):
    src_nbits = src.dtype.primitive_bitwidth
    dst_nbits = dst_type.primitive_bitwidth
    assert src_nbits % dst_nbits == 0, f"src_nbits {src_nbits} %dst_nbits {dst_nbits}!= 0"
    num_elements = src_nbits // dst_nbits
    dst_constraint = _int_constraint(core.constexpr(dst_nbits), _semantic=_semantic).value
    src_constraint = _int_constraint(core.constexpr(src_nbits), _semantic=_semantic).value

    dst_operands = [f'${i}' for i in range(num_elements)]
    dst_constraints = ['=' + dst_constraint for i in range(num_elements)]
    asm = f"mov.b{src_nbits} {{{', '.join(dst_operands)}}}, ${num_elements};"
    constraints = f"{','.join(dst_constraints)}, {src_constraint}"
    vals = tl.inline_asm_elementwise(
        asm=asm,
        constraints=constraints,
        args=[src],
        dtype=tuple([dst_type for _ in range(num_elements)]),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    ).values
    return vector(vals)


__all__ = [
    "__syncthreads",
    "__fence",
    "tid",
    "ntid",
    "laneid",
    "wait_eq",
    "arrive_inc",
    "red_release",
    "ld_acquire",
    "atomic_add",
    "atomic_add_per_warp",
    "__shfl_sync_i32",
    "__shfl_up_sync_i32",
    "__shfl_down_sync_i32",
    "__shfl_xor_sync_i32",
    "__ballot_sync",
    "ld",
    "atomic_cas",
    "ld_b32",
    "st_b32",
    "globaltimer",
    "globaltimer_lo",
    "globaltimer_hi",
    "smid",
    "membar",
    "pack_b32_v2",
]
