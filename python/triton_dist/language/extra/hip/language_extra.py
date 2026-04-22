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

    _sem = core._unwrap_if_constexpr(semantic)
    _scp = core._unwrap_if_constexpr(scope)

    return dist_core.extern_elementwise(
        "",
        "",
        [ptr],
        {(core.pointer_type(dtype), ): (
             f"__triton_hip_load_{dtype.primitive_bitwidth}_{_sem}_{_scp}",
             dtype,
         )
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
             core.dtype("fp16"),
             core.dtype("bf16"),
             core.dtype("fp32"),
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

    _sem = core._unwrap_if_constexpr(semantic)
    _scp = core._unwrap_if_constexpr(scope)

    return dist_core.extern_elementwise(
        "",
        "",
        [ptr, core.cast(val, dtype=ptr.dtype.element_ty, _semantic=_semantic)],
        {(core.pointer_type(dtype), dtype): (
             f"__triton_hip_store_{dtype.primitive_bitwidth}_{_sem}_{_scp}",
             dtype,
         )
         for dtype in [
             core.dtype("int32"),
             core.dtype("uint32"),
             core.dtype("int64"),
             core.dtype("uint64"),
             core.dtype("fp16"),
             core.dtype("bf16"),
             core.dtype("fp32"),
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
    # The order of shfl modes is from (llvm-project/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td)
    ALL_SHFL_MODES = ["xor", "up", "down", "idx"]

    if mode_str not in ALL_SHFL_MODES:
        raise RuntimeError(f"unexpected gpu shuffle mode, expecte: {ALL_SHFL_MODES}, but got: {mode_str}")

    return ALL_SHFL_MODES.index(mode_str)


@core.extern
def laneid(_semantic=None):
    return core.tensor(_semantic.builder.create_laneid(), core.int32)


# ---------------------------------------------------------------------------
# Warp shuffle primitives via GCN inline assembly (ds_bpermute_b32).
#
# gpu.shuffle (mlir::gpu::ShuffleOp) does not reliably lower to LLVM for
# wavefront-64 targets on the current Triton AMD backend.  We emit
# ds_bpermute_b32 directly instead, which reads the source VGPR from an
# arbitrary lane specified by a byte-offset (lane_id * 4).
# ---------------------------------------------------------------------------


@core.extern
def _ds_bpermute_b32(value, byte_offset, _semantic=None):
    """Low-level ds_bpermute_b32: read *value* from the lane at *byte_offset/4*."""
    tl.static_assert(value.dtype == tl.int32 or value.dtype == tl.uint32, "_ds_bpermute_b32 only supports int32/uint32",
                     _semantic=_semantic)
    return tl.inline_asm_elementwise(
        asm="ds_bpermute_b32 $0, $1, $2\ns_waitcnt lgkmcnt(0)",
        constraints="=v,v,v",
        args=[byte_offset, value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
        _semantic=_semantic,
    )


@triton.jit
def __shfl_sync_i32(value, lane):
    """Shuffle idx: read *value* from *lane* (broadcast if lane is constant)."""
    byte_offset = tl.cast(lane, tl.int32) * 4
    return _ds_bpermute_b32(tl.cast(value, tl.int32), byte_offset)


@triton.jit
def __shfl_up_sync_i32(value, delta):
    """Shuffle up: each lane reads from (laneid - delta), clamped to 0."""
    _lid = laneid()
    src_lane = _lid - tl.cast(delta, tl.int32)
    if src_lane < 0:
        src_lane = _lid
    byte_offset = src_lane * 4
    return _ds_bpermute_b32(tl.cast(value, tl.int32), byte_offset)


@triton.jit
def __shfl_down_sync_i32(value, delta):
    """Shuffle down: each lane reads from (laneid + delta), clamped to 63."""
    WARP_SIZE: tl.constexpr = 64
    _lid = laneid()
    src_lane = _lid + tl.cast(delta, tl.int32)
    if src_lane >= WARP_SIZE:
        src_lane = _lid
    byte_offset = src_lane * 4
    return _ds_bpermute_b32(tl.cast(value, tl.int32), byte_offset)


@triton.jit
def __shfl_xor_sync_i32(value, mask):
    """Shuffle xor (butterfly): each lane reads from (laneid ^ mask)."""
    _lid = laneid()
    src_lane = _lid ^ tl.cast(mask, tl.int32)
    byte_offset = src_lane * 4
    return _ds_bpermute_b32(tl.cast(value, tl.int32), byte_offset)


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
