# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))

void_ptr = core.pointer_type(core.void)


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    # force have a return value, even not used.
    return extern_call(
        "libshmem_device",
        "",
        [dest, value, pe],
        {
            (
                core.pointer_type(core.dtype("int32")),
                core.dtype("int32"),
                core.dtype("int32"),
            ): (
                "aclshmem_int32_p",
                (),
            ),  # void return type
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [local_ptr, pe],
        {
            (core.pointer_type(core.dtype(core_dtype)), core.dtype(pe_dtype)): (
                "aclshmem_ptr",
                core.pointer_type(core.dtype(core_dtype)),  # of the same dtype
            )
            for core_dtype in core.dtype.SINT_TYPES
            + core.dtype.UINT_TYPES
            + core.dtype.FP_TYPES
            + core.dtype.OTHER_TYPES
            for pe_dtype in ["int32", "uint32"]
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def barrier_all(_semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_barrier_all", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def barrier_all_vec(_semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmemx_barrier_all_vec", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
