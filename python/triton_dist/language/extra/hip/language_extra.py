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
from triton_dist.language import core as dist_core
from triton_dist.language import vector, make_vector


def _translate_scope(scope):
    scope = core._unwrap_if_constexpr(scope)
    if scope in ["workgroup", "agent", "system"]:
        return scope
    cuda_to_hip_scope_map = {"cta": "workgroup", "gpu": "agent", "sys": "system"}
    return cuda_to_hip_scope_map.get(scope, None)


def _translate_semantic(semantic):
    semantic = core._unwrap_if_constexpr(semantic)
    if semantic == "relaxed":
        return "monotonic"
    elif semantic in ["monotonic", "acquire", "release", "acq_rel"]:
        return semantic
    else:
        raise ValueError(f"Unsupported semantic: {semantic}")


# Keyed by Triton dtype str → LLVM type-width suffix for callee names.
# signed/unsigned share the same suffix because LLVM does not distinguish
# signedness in integer types (both map to iN).  Float types (fp32, fp64)
# are absent here because scalar ld/st bitcasts them to integer equivalents.
_DTYPE_SUFFIX = {
    "int16": "16",
    "uint16": "16",
    "fp16": "fp16",
    "bf16": "bf16",
    "int32": "32",
    "uint32": "32",
    "int64": "64",
    "uint64": "64",
}


def _dtype_callee_suffix(dtype):
    """Unique LLVM function name suffix per underlying LLVM type."""
    return _DTYPE_SUFFIX[str(dtype)]


@core.extern
def _tid_wrapper(axis: core.constexpr, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (f"llvm.amdgcn.workitem.id.{axis.value}", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def tid(axis, _semantic=None):
    if axis == 0:
        return _tid_wrapper(core.constexpr("x"), _semantic=_semantic)
    elif axis == 1:
        return _tid_wrapper(core.constexpr("y"), _semantic=_semantic)
    elif axis == 2:
        return _tid_wrapper(core.constexpr("z"), _semantic=_semantic)
    else:
        tl.static_assert(False, "axis must be 0, 1 or 2", _semantic=_semantic)


@core.extern
def _ntid_wrapper(axis: core.constexpr, _semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): (
                f"llvm.amdgcn.workgroup.size.{axis.value}",
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


@core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope="agent",
    semantic="relaxed",
    _semantic=None,
):
    """
    semantic should be one of ["monotonic", "release", "acquire", "acq_rel]
    scope should be one of ["workgroup", "agent", "system]
    semantic only works when compare `old == val` success(success oreder), otherwise it's always "relaxed"(failure order)
    """
    semantic = _translate_semantic(semantic)
    scope = _translate_scope(scope)
    failure_order = "relaxed"  # equal to monotonic
    assert core._unwrap_if_constexpr(semantic) in [
        "monotonic",
        "release",
        "acquire",
        "acq_rel",
        "relaxed",
    ], "semantic should be one of ['monotonic', 'release', 'acquire', 'acq_rel', 'relaxed']"
    assert core._unwrap_if_constexpr(scope) in [
        "workgroup",
        "agent",
        "system",
    ], "scope should be one of ['workgroup', 'agent', 'system']"

    _sem = core._unwrap_if_constexpr(semantic)
    _fail = core._unwrap_if_constexpr(failure_order)
    _scp = core._unwrap_if_constexpr(scope)

    return dist_core.extern_elementwise(
        "",
        "",
        [
            ptr,
            core.cast(value, dtype=ptr.dtype.element_ty, _semantic=_semantic),
            core.cast(target_value, dtype=ptr.dtype.element_ty, _semantic=_semantic),
        ],
        {(core.pointer_type(dtype), dtype, dtype): (
             f"__triton_hip_atom_cas_{dtype.primitive_bitwidth}_{_sem}_{_fail}_{_scp}",
             dtype,
         )
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
         ]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_add(
    ptr,
    value,
    scope="agent",
    semantic="relaxed",
    _semantic=None,
):
    """
    semantic should be one of ["monotonic", "release", "acquire", "acq_rel]
    scope should be one of ["workgroup", "agent", "system]
    """
    semantic = _translate_semantic(semantic)
    scope = _translate_scope(scope)
    assert core._unwrap_if_constexpr(semantic) in [
        "monotonic",
        "release",
        "acquire",
        "acq_rel",
        "relaxed",
    ], "semantic should be one of ['monotonic', 'release', 'acquire', 'acq_rel']"
    assert core._unwrap_if_constexpr(scope) in [
        "workgroup",
        "agent",
        "system",
    ], "scope should be one of ['workgroup', 'agent', 'system']"

    _sem = core._unwrap_if_constexpr(semantic)
    _scp = core._unwrap_if_constexpr(scope)

    return dist_core.extern_elementwise(
        "",
        "",
        [
            ptr,
            core.cast(value, dtype=ptr.dtype.element_ty, _semantic=_semantic),
        ],
        {(core.pointer_type(dtype), dtype): (
             f"__triton_hip_atom_add_{dtype.primitive_bitwidth}_{_sem}_{_scp}",
             dtype,
         )
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
         ]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ld(
    ptr,
    scope="agent",
    semantic="monotonic",
    _semantic=None,
):
    """
    semantic should be one of ["monotonic", "accquire"]
    scope should be one of ["workgroup", "agent", "system]
    """
    semantic = _translate_semantic(semantic)
    scope = _translate_scope(scope)
    assert core._unwrap_if_constexpr(semantic) in [
        "monotonic",
        "acquire",
    ], "load only supports 'monotonic' and 'acquire' semantics"
    assert core._unwrap_if_constexpr(scope) in [
        "workgroup",
        "agent",
        "system",
    ], "scope should be one of ['workgroup', 'agent', 'system']"

    elem_ty = ptr.dtype.element_ty
    if elem_ty.is_floating():
        int_ty = core.get_int_dtype(elem_ty.primitive_bitwidth, True)
        ptr_int = tl.cast(ptr, core.pointer_type(int_ty), _semantic=_semantic)
        raw = ld(ptr_int, scope=scope, semantic=semantic, _semantic=_semantic)
        return tl.cast(raw, elem_ty, bitcast=True, _semantic=_semantic)

    sem = core._unwrap_if_constexpr(semantic)
    scp = core._unwrap_if_constexpr(scope)
    return dist_core.extern_elementwise(
        "",
        "",
        [ptr],
        {(core.pointer_type(dtype), ): (f"__triton_hip_load_{_dtype_callee_suffix(dtype)}_{sem}_{scp}", dtype)
         for dtype in [
             core.dtype("int16"),
             core.dtype("uint16"),
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
         ]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def st(
    ptr,
    val,
    scope="agent",
    semantic="monotonic",
    _semantic=None,
):
    """
    semantic should be one of ["monotonic", "release"]
    scope should be one of ["workgroup", "agent", "system]
    """
    semantic = _translate_semantic(semantic)
    scope = _translate_scope(scope)
    assert core._unwrap_if_constexpr(semantic) in [
        "monotonic",
        "release",
    ], "store only supports 'monotonic' and 'release' semantics"
    assert core._unwrap_if_constexpr(scope) in [
        "workgroup",
        "agent",
        "system",
    ], "scope should be one of ['workgroup', 'agent', 'system']"

    elem_ty = ptr.dtype.element_ty
    if elem_ty.is_floating():
        int_ty = core.get_int_dtype(elem_ty.primitive_bitwidth, True)
        ptr_int = tl.cast(ptr, core.pointer_type(int_ty), _semantic=_semantic)
        val_int = tl.cast(val, int_ty, bitcast=True, _semantic=_semantic)
        return st(ptr_int, val_int, scope=scope, semantic=semantic, _semantic=_semantic)

    sem = core._unwrap_if_constexpr(semantic)
    scp = core._unwrap_if_constexpr(scope)
    return dist_core.extern_elementwise(
        "",
        "",
        [ptr, core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic)],
        {(core.pointer_type(dtype), dtype): (f"__triton_hip_store_{_dtype_callee_suffix(dtype)}_{sem}_{scp}", dtype)
         for dtype in [
             core.dtype("int16"),
             core.dtype("uint16"),
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
         ]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def __syncthreads(_semantic=None):
    return core.extern_elementwise(
        "",
        "",
        [],
        {
            (): ("__triton_hip_syncthreads", core.uint64),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def load(ptr, scope="agent", semantic="monotonic", _semantic=None):
    return ld(
        ptr,
        scope=scope,
        semantic=semantic,
        _semantic=_semantic,
    )


@core.extern
def store(ptr, val, scope="agent", semantic="monotonic", _semantic=None):
    return st(
        ptr,
        val,
        semantic=semantic,
        scope=scope,
        _semantic=_semantic,
    )


@core.extern
def sync_grid(_semantic=None):
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__ockl_grid_sync", core.dtype("int32")),  # does not return
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def smid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__extra_smid", core.dtype("int32")),  # does not return
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def seid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__extra_seid", core.dtype("int32")),  # does not return
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def cuid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__extra_cuid", core.dtype("int32")),  # does not return
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def xccid(_semantic=None):
    # now only support GFX942
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__extra_xccid", core.dtype("int32")),  # does not return
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def clock(_semantic=None):
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__extra_clock", core.dtype("uint64")),  # does not return
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def wallclock(_semantic=None):
    """ strange that wallclock is not in unit of nanosecond, but 10*nanosecond. no doc found for that """
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): ("__extra_wallclock", core.dtype("uint64")),  # does not return
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence(semantic="monotonic", scope="agent", _semantic=None):
    return dist_core.extern_elementwise(
        "",
        "",
        [],
        {
            tuple(): (f"__extra_fence_{core._unwrap_if_constexpr(semantic)}_{core._unwrap_if_constexpr(scope)}",
                      core.dtype("int32")),  # does not return
        },
        is_pure=False,
        _semantic=_semantic,
    )


def _str_to_gpu_shfl_mode(mode_str):
    # Must match mlir::gpu::ShuffleMode enum order:
    #   XOR = 0, DOWN = 1, UP = 2, IDX = 3
    ALL_SHFL_MODES = ["xor", "down", "up", "idx"]

    if mode_str not in ALL_SHFL_MODES:
        raise RuntimeError(f"unexpected gpu shuffle mode, expected: {ALL_SHFL_MODES}, but got: {mode_str}")

    return ALL_SHFL_MODES.index(mode_str)


@core.extern
def laneid(_semantic=None):
    return core.tensor(_semantic.builder.create_laneid(), core.int32)


@core.extern
def __shfl_sync_with_mode_i32(
    value,
    offset,
    mode: core.constexpr = "up",
    width: int = 64,
    _semantic=None,
):
    shfl_mode = _str_to_gpu_shfl_mode(mode.value)
    if isinstance(offset, core.constexpr):
        offset = core.to_tensor(offset, _semantic=_semantic)

    return core.tensor(
        _semantic.builder.create_warp_shuffle(
            value.handle,
            offset.handle,
            core.to_tensor(width, _semantic=_semantic).handle,
            shfl_mode,
        ), value.dtype)


@triton.jit
def __shfl_sync_i32(value, laneid):
    return __shfl_sync_with_mode_i32(value, laneid, "idx", 64)


@triton.jit
def __shfl_up_sync_i32(value, offset):
    # CUDA semantics: lane i reads from lane (i - offset) if i >= offset,
    # else returns its own value (clamp).

    lid = laneid()
    src = lid - offset
    clamped = tl.where(src >= 0, src, lid)
    return __shfl_sync_with_mode_i32(value, clamped, "idx", 64)


@triton.jit
def __shfl_down_sync_i32(value, offset):
    # CUDA semantics: lane i reads from lane (i + offset) if (i + offset) < warp_size,
    # else returns its own value (clamp).
    lid = laneid()
    src = lid + offset
    clamped = tl.where(src < 64, src, lid)
    return __shfl_sync_with_mode_i32(value, clamped, "idx", 64)


@triton.jit
def __shfl_xor_sync_i32(value, offset):
    return __shfl_sync_with_mode_i32(value, offset, "xor", 64)


@triton.jit
def pack_b32_v2(val0, val1):
    """Pack two 32-bit values into one 64-bit value (val0: low 32, val1: high 32)."""
    lo = tl.cast(tl.cast(val0, tl.uint32), tl.uint64)
    hi = tl.cast(tl.cast(val1, tl.uint32), tl.uint64)
    return lo | (hi << 32)


@triton.jit
def atomic_add_per_warp(barrier_ptr, value, scope: core.constexpr, semantic: core.constexpr):
    """Warp-level atomic add: lane 0 performs atomic_add, result is broadcast to all lanes.

    AMD equivalent of the NVIDIA atomic_add_per_warp. Uses wavefront64 shuffle
    (no mask needed — AMD wavefronts are always fully synchronized).
    """
    _laneid = laneid()
    x = tl.cast(0, barrier_ptr.dtype.element_ty)
    if _laneid == 0:
        x = atomic_add(barrier_ptr, value, scope, semantic)
    return __shfl_sync_i32(x, 0)


@core.extern
def load_v4_b32(ptr, _semantic=None):

    i32_ir = tl.int32.to_ir(_semantic.builder)
    ptr_val = _semantic.to_tensor(ptr)
    op = _semantic.builder.create_extern_call("", "", "__triton_hip_load_v4_b32", [ptr_val.handle], [i32_ir] * 4, False)
    return (
        tl.tensor(op.get_result(0), tl.int32),
        tl.tensor(op.get_result(1), tl.int32),
        tl.tensor(op.get_result(2), tl.int32),
        tl.tensor(op.get_result(3), tl.int32),
    )


@core.extern
def st_v4_b32(ptr, val0, val1, val2, val3, _semantic=None):

    return dist_core.extern_elementwise(
        "",
        "",
        [
            ptr,
            tl.cast(val0, tl.int32, _semantic=_semantic),
            tl.cast(val1, tl.int32, _semantic=_semantic),
            tl.cast(val2, tl.int32, _semantic=_semantic),
            tl.cast(val3, tl.int32, _semantic=_semantic)
        ],
        {(core.pointer_type(core.dtype("int32")), core.dtype("int32"), core.dtype("int32"), core.dtype("int32"),
          core.dtype("int32")): ("__triton_hip_store_v4_b32", core.dtype("int32"))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ld_vector(
    ptr,
    vec_size: core.constexpr = 1,
    scope: core.constexpr = "agent",
    semantic: core.constexpr = "monotonic",
    _semantic=None,
):
    assert isinstance(vec_size, tl.constexpr)
    elems_bits = ptr.dtype.element_ty.primitive_bitwidth
    total_bits = elems_bits * vec_size
    if total_bits % 128 == 0:
        assert scope is None or core._unwrap_if_constexpr(scope) in (None, "agent"), \
            f"ld_vector v4 path: AMDGPU vector load is non-atomic, scope must be None or 'agent', got {scope}"
        assert semantic is None or core._unwrap_if_constexpr(semantic) in (None, "monotonic", "relaxed"), \
            f"ld_vector v4 path: AMDGPU vector load is non-atomic, semantic must be None/'monotonic'/'relaxed', got {semantic}"
        ptr_i32 = tl.cast(ptr, tl.pi32_t, _semantic=_semantic)
        vals_i32 = []
        num_iters = total_bits // 128
        for idx in range(num_iters):
            v0, v1, v2, v3 = load_v4_b32(tl.add(ptr_i32, idx * 4, _semantic=_semantic), _semantic=_semantic)
            vals_i32 += [v0, v1, v2, v3]
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
    scope: core.constexpr = "agent",
    semantic: core.constexpr = "monotonic",
    _semantic=None,
):
    assert isinstance(vec, vector), "st_vector: vec must be a vector"
    total_bits = vec.type.vector_nbits
    if total_bits % 128 == 0:
        assert scope is None or core._unwrap_if_constexpr(scope) in (None, "agent"), \
            f"st_vector v4 path: AMDGPU vector store is non-atomic, scope must be None or 'agent', got {scope}"
        assert semantic is None or core._unwrap_if_constexpr(semantic) in (None, "monotonic", "relaxed"), \
            f"st_vector v4 path: AMDGPU vector store is non-atomic, semantic must be None/'monotonic'/'relaxed', got {semantic}"
        vec_i32 = vec.recast(tl.int32, _semantic=_semantic)
        ptr_i32 = tl.cast(ptr, tl.pi32_t, _semantic=_semantic)
        vals_i32 = [tl.cast(val, tl.int32, _semantic=_semantic) for val in vec_i32]
        num_iters = total_bits // 128
        for idx in range(num_iters):
            st_v4_b32(tl.add(ptr_i32, idx * 4, _semantic=_semantic), *vals_i32[idx * 4:(idx + 1) * 4],
                      _semantic=_semantic)
    else:
        for idx, v in enumerate(vec):
            st(tl.add(ptr, idx, _semantic=_semantic), v, scope=scope, semantic=semantic, _semantic=_semantic)


@core.extern
def pack(src: vector, dst_type, _semantic=None):
    """Pack a vector of smaller-bitwidth elements into one larger scalar.
    E.g. vector([i32_lo, i32_hi]) -> i64 via shift+or (no PTX needed)."""
    assert isinstance(src, vector)
    dst_nbits = dst_type.primitive_bitwidth
    src_elem_dtype = src.type.elem_type
    assert src.type.vector_nbits == dst_nbits, (f"src.type.vector_nbits {src.type.vector_nbits} != "
                                                f"dst_type.primitive_bitwidth {dst_nbits}")
    src_nbits = src_elem_dtype.primitive_bitwidth
    src_int_ty = core.get_int_dtype(src_nbits, False)
    dst_int_ty = core.get_int_dtype(dst_nbits, False)
    combined = tl.cast(tl.constexpr(0), dst_int_ty, _semantic=_semantic)
    for j in range(src.type.vec_size):
        bits = tl.cast(src.values[j], src_int_ty, bitcast=True, _semantic=_semantic)
        bits = tl.cast(bits, dst_int_ty, _semantic=_semantic)
        shifted = bits.__lshift__(j * src_nbits, _semantic=_semantic)
        combined = combined.__or__(shifted, _semantic=_semantic)
    return tl.cast(combined, dst_type, bitcast=True, _semantic=_semantic)


@core.extern
def unpack(src, dst_type, _semantic=None):
    """Unpack a larger scalar into multiple smaller-bitwidth elements.
    E.g. i64 -> (i32_lo, i32_hi) via shift+mask."""
    src_nbits = src.dtype.primitive_bitwidth
    dst_nbits = dst_type.primitive_bitwidth
    assert src_nbits % dst_nbits == 0
    ratio = src_nbits // dst_nbits
    src_int_ty = core.get_int_dtype(src_nbits, False)
    dst_int_ty = core.get_int_dtype(dst_nbits, False)
    int_val = tl.cast(src, src_int_ty, bitcast=True, _semantic=_semantic)
    mask_val = (1 << dst_nbits) - 1
    results = []
    for j in range(ratio):
        shifted = int_val.__rshift__(j * dst_nbits, _semantic=_semantic)
        masked = shifted.__and__(mask_val, _semantic=_semantic)
        elem = tl.cast(masked, dst_int_ty, _semantic=_semantic)
        elem = tl.cast(elem, dst_type, bitcast=True, _semantic=_semantic)
        results.append(elem)
    return results


__all__ = [
    "__syncthreads",
    "tid",
    "ntid",
    "laneid",
    "atomic_cas",
    "atomic_add",
    "pack_b32_v2",
    "atomic_add_per_warp",
    "__shfl_sync_i32",
    "__shfl_up_sync_i32",
    "__shfl_down_sync_i32",
    "__shfl_xor_sync_i32",
    "load",
    "store",
    "load_v4_b32",
    "st_v4_b32",
    "ld_vector",
    "st_vector",
    "pack",
    "unpack",
    "sync_grid",
    "smid",
    "sync_grid",
    "smid",
    "seid",
    "cuid",
    "xccid",
    "clock",
    "wallclock",
    "fence",
]
