# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
pi_i32_t = tl.core.pointer_type(tl.core.dtype("int32"))

void_ptr = core.pointer_type(core.void)

# The subset of dtypes supported by Ascend SHMEM RMA ops.
RMA_DTYPES = [
    "fp16",
    "fp32",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bf16",
]

DTYPE_TO_KERNEL_SUFFIX = {
    "fp16": "half",
    "fp32": "float",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bf16": "bfloat16",
}


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    elem_dtype_list = (tl.core.dtype.SINT_TYPES + tl.core.dtype.UINT_TYPES + tl.core.dtype.FP_TYPES +
                       tl.core.dtype.OTHER_TYPES)
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(tl.core.pointer_type(tl.core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


# Keep pointer_type hashable for Triton extern signature dict keys.
patch_hash_method_for_pointer_type()


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
def team_my_pe(team, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("aclshmem_team_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def team_n_pes(team, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("aclshmem_team_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def team_translate_pe(src_team, pe_in_src_team, dest_team, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [
            tl.cast(src_team, tl.int32, _semantic=_semantic),
            tl.cast(pe_in_src_team, tl.int32, _semantic=_semantic),
            tl.cast(dest_team, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.int32, tl.int32, tl.int32): (
                "aclshmem_team_translate_pe",
                core.dtype("int32"),
            ),
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
        {(
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
        {(core.pointer_type(core.dtype(core_dtype)), core.dtype(pe_dtype)): (
             "aclshmem_ptr", core.pointer_type(core.dtype(core_dtype)),  # of the same dtype
         )
         for core_dtype in core.dtype.SINT_TYPES + core.dtype.UINT_TYPES + core.dtype.FP_TYPES + core.dtype.OTHER_TYPES
         for pe_dtype in ["int32", "uint32"]},
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


@core.extern
def barrier(team, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("aclshmem_barrier", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def barrier_vec(team, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [tl.cast(team, tl.int32, _semantic=_semantic)],
        {
            (tl.int32, ): ("aclshmemx_barrier_vec", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def _rma_mem_impl(dest, source, elem_size, pe, op: core.constexpr, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [
            dest,
            source,
            tl.cast(elem_size, tl.uint32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {(
            core.pointer_type(core.dtype(core_dtype)),
            core.pointer_type(core.dtype(core_dtype)),
            tl.uint32,
            tl.int32,
        ): (
             f"aclshmem_{DTYPE_TO_KERNEL_SUFFIX[core_dtype]}_{op.value}",
             (),
         )
         for core_dtype in RMA_DTYPES},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def getmem(dest, source, bytes, pe, _semantic=None):
    return _rma_mem_impl(dest, source, bytes, pe, core.constexpr("getmem"), _semantic=_semantic)


@core.extern
def putmem(dest, source, bytes, pe, _semantic=None):
    return _rma_mem_impl(dest, source, bytes, pe, core.constexpr("putmem"), _semantic=_semantic)


@core.extern
def getmem_nbi(dest, source, bytes, pe, qp_id=0, _semantic=None):
    return _rma_mem_impl(dest, source, bytes, pe, core.constexpr("getmem_nbi"), _semantic=_semantic)


@core.extern
def putmem_nbi(dest, source, bytes, pe, qp_id=0, _semantic=None):
    return _rma_mem_impl(dest, source, bytes, pe, core.constexpr("putmem_nbi"), _semantic=_semantic)


@core.extern
def _putmem_signal_impl(
    dest,
    source,
    nbytes,
    sig_addr,
    signal,
    sig_op,
    pe,
    op: core.constexpr,
    _semantic=None,
):
    return extern_call(
        "libshmem_device",
        "",
        [
            dest,
            source,
            tl.cast(nbytes, tl.uint32, _semantic=_semantic),
            sig_addr,
            tl.cast(signal, tl.int32, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {(
            core.pointer_type(core.dtype(core_dtype)),
            core.pointer_type(core.dtype(core_dtype)),
            tl.uint32,
            core.pointer_type(core.dtype("int32")),
            tl.int32,
            tl.int32,
            tl.int32,
        ): (
             f"aclshmem_{DTYPE_TO_KERNEL_SUFFIX[core_dtype]}_{op.value}",
             (),
         )
         for core_dtype in RMA_DTYPES},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem_signal(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("putmem_signal"),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("putmem_signal_nbi"),
        _semantic=_semantic,
    )


@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [
            sig_addr,
            tl.cast(signal, tl.int32, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_i32_t, tl.int32, tl.int32, tl.int32): (
                "aclshmemx_signal_op",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [
            sig_addr,
            tl.cast(cmp_, tl.int32, _semantic=_semantic),
            tl.cast(cmp_val, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_i32_t, tl.int32, tl.int32): (
                "aclshmem_signal_wait_until",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def quiet(_semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_quiet", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence(_semantic=None):
    return extern_call(
        "libshmem_device",
        "",
        [],
        {
            (): ("aclshmem_fence", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
