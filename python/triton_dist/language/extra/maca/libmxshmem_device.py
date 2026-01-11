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
# TODO:(MACA UPGRADE) upgrade to support import libmxshmem_device from triton_dist.language.extra.libshmem_device
from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call
import sys

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))

# class mxshmemi_cmp_type(Enum):
MXSHMEM_CMP_EQ = 0
MXSHMEM_CMP_NE = 1
MXSHMEM_CMP_GT = 2
MXSHMEM_CMP_LE = 3
MXSHMEM_CMP_LT = 4
MXSHMEM_CMP_GE = 5
MXSHMEM_CMP_SENTINEL = sys.maxsize

MXSHMEM_SIGNAL_SET = 9
MXSHMEM_SIGNAL_ADD = 10

void_ptr = core.pointer_type(core.void)


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "libmxshmem_device",
        "",
        [],
        {
            (): ("mxshmem_my_pe", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "libmxshmem_device",
        "",
        [],
        {
            (): ("mxshmem_n_pes", core.dtype("int32")),
        },
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    # force have a return value, even not used.
    return extern_call(
        "libmxshmem_device",
        "",
        [dest, value, pe],
        {(
            core.pointer_type(core.dtype("int32")),
            core.dtype("int32"),
            core.dtype("int32"),
        ): ("mxshmem_int_p", ()),  # void return type
         },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    return extern_call(
        "libmxshmem_device",
        "",
        [local_ptr, pe],
        {(core.pointer_type(core.dtype(core_dtype)), core.dtype(pe_dtype)): (
             "mxshmem_ptr", core.pointer_type(core.dtype(core_dtype)),  # of the same dtype
         )
         for core_dtype in core.dtype.SINT_TYPES + core.dtype.UINT_TYPES + core.dtype.FP_TYPES + core.dtype.OTHER_TYPES
         for pe_dtype in ["int32", "uint32"]},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_block(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        "libmxshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int8), _builder=_semantic.builder),
            tl.cast(source, tl.pointer_type(tl.int8), _builder=_semantic.builder),
            tl.cast(nbytes, tl.uint64, _builder=_semantic.builder),
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_semantic.builder),
            tl.cast(sig_op, tl.int32, _builder=_semantic.builder),
            tl.cast(pe, tl.int32, _builder=_semantic.builder),
        ],
        {
            (tl.pointer_type(tl.int8), tl.pointer_type(tl.int8), tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "mxshmemx_putmem_signal_block",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def signal_op(sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        "libmxshmem_device",
        "",
        [
            sig_addr,  # no cast: pointer type should be aligned
            tl.cast(signal, tl.uint64, _builder=_semantic.builder),
            tl.cast(sig_op, tl.int32, _builder=_semantic.builder),
            tl.cast(pe, tl.int32,_builder=_semantic.builder),
        ],
        {
            (pi_u64_t, tl.uint64, tl.int32, tl.int32): (
                "mxshmemx_signal_op",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    return extern_call(
        "libmxshmem_device",
        "",
        [
            sig_addr,
            tl.cast(cmp_, tl.int32, _builder=_semantic.builder),
            tl.cast(cmp_val, tl.uint64, _builder=_semantic.builder),
        ],  # no cast
        {
            (pi_u64_t, tl.int32, tl.uint64): (
                "mxshmem_signal_wait_until",
                tl.int32,
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )
