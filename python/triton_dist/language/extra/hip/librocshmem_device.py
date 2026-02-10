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
from triton.language import core
import triton.language as tl
from triton_dist.language.core import extern_call

pi_u64_t = tl.core.pointer_type(tl.core.dtype("uint64"))
pi_i64_t = tl.core.pointer_type(tl.core.dtype("int64"))

# ROCSHMEM_CMPS (enum)
ROCSHMEM_CMP_EQ = 0
ROCSHMEM_CMP_NE = 1
ROCSHMEM_CMP_GT = 2
ROCSHMEM_CMP_GE = 3
ROCSHMEM_CMP_LT = 4
ROCSHMEM_CMP_LE = 5

# ROCSHMEM_SIGNAL_OPS (enum)
ROCSHMEM_SIGNAL_SET = 0
ROCSHMEM_SIGNAL_ADD = 1


void_ptr = core.pointer_type(core.void)


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {(): ("rocshmem_my_pe_wrapper", (tl.int32))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {(): ("rocshmem_n_pes_wrapper", (tl.int32))},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def int_p(dest, value, pe, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(value, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.int32, tl.int32): ("rocshmem_int_p_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def remote_ptr(local_ptr, pe, _semantic=None):
    return tl.cast(
        extern_call(
            "librocshmem_device",
            "",
            [
                tl.cast(local_ptr, tl.pointer_type(tl.void), _semantic=_semantic),
                tl.cast(pe, tl.int32, _semantic=_semantic),
            ],
            {
                (tl.pointer_type(tl.void), tl.int32): (
                    "rocshmem_ptr_wrapper",
                    tl.pointer_type(tl.void),
                ),
            },
            is_pure=False,
            _semantic=_semantic,
        ),
        local_ptr.dtype,
        _semantic=_semantic,
    )


@core.extern
def _getmem_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                 _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                f"rocshmem_getmem{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def getmem(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr(""), _semantic=_semantic)


@core.extern
def getmem_wave(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_wave"), core.constexpr(""), _semantic=_semantic)


@core.extern
def getmem_wg(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_wg"), core.constexpr(""), _semantic=_semantic)


@core.extern
def getmem_nbi(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def getmem_nbi_wave(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_wave"), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def getmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return _getmem_impl(dest, source, nbytes, pe, core.constexpr("_wg"), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def _putmem_impl(dest, source, nbytes, pe, SCOPE_SUFFIX: core.constexpr, NBI: core.constexpr = core.constexpr(""),
                 _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32): (
                f"rocshmem_putmem{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def putmem(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_wave(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_wave"), core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_wg(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_wg"), core.constexpr(""), _semantic=_semantic)


@core.extern
def putmem_nbi(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr(""), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def putmem_nbi_wave(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_wave"), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def putmem_nbi_wg(dest, source, nbytes, pe, _semantic=None):
    return _putmem_impl(dest, source, nbytes, pe, core.constexpr("_wg"), core.constexpr("_nbi"), _semantic=_semantic)


@core.extern
def _putmem_signal_impl(dest, source, nbytes, sig_addr, signal, sig_op, pe, SCOPE_SUFFIX: core.constexpr,
                        NBI: core.constexpr = core.constexpr(""), _semantic=None):
    tl.static_assert(sig_addr.dtype == tl.pointer_type(tl.int64), "sig_addr should be a pointer of uint64_t",
                     _semantic=_semantic)
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(sig_addr, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.pointer_type(tl.int64), tl.uint64, tl.int32, tl.int32):
            (
                f"rocshmem_putmem_signal{NBI.value}{SCOPE_SUFFIX.value}_wrapper",
                (),
            ),
        },
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
        core.constexpr(""),
        core.constexpr(""),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wg"),
        core.constexpr(""),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_wave(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wave"),
        core.constexpr(""),
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
        core.constexpr(""),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi_wg(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wg"),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )


@core.extern
def putmem_signal_nbi_wave(dest, source, nbytes, sig_addr, signal, sig_op, pe, _semantic=None):
    return _putmem_signal_impl(
        dest,
        source,
        nbytes,
        sig_addr,
        signal,
        sig_op,
        pe,
        core.constexpr("_wave"),
        core.constexpr("_nbi"),
        _semantic=_semantic,
    )


@core.extern
def signal_wait_until(sig_addr, cmp_, cmp_val, _semantic=None):
    tl.static_assert(sig_addr.dtype == pi_u64_t or sig_addr.dtype == pi_i64_t,
                     "sig_addr should be a pointer of uint64_t/int64_t",
                     _semantic=_semantic)
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(sig_addr, pi_u64_t, _semantic=_semantic),
            tl.cast(cmp_, tl.int32, _semantic=_semantic),
            tl.cast(cmp_val, tl.uint64, _semantic=_semantic),
        ],  # no cast
        {
            (pi_u64_t, tl.int32, tl.uint64): (
                "rocshmem_ulong_wait_until_wrapper",
                (),
            ),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ulong_put_signal(dest, source, nelems, sig_addr, signal, sig_op, pe, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [
            tl.cast(dest, pi_u64_t, _semantic=_semantic),
            tl.cast(source, pi_u64_t, _semantic=_semantic),
            tl.cast(nelems, tl.uint64, _semantic=_semantic),
            tl.cast(sig_addr, pi_u64_t, _semantic=_semantic),
            tl.cast(signal, tl.uint64, _semantic=_semantic),
            tl.cast(sig_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
        ],
        {
            (pi_u64_t, pi_u64_t, tl.uint64, pi_u64_t, tl.uint64, tl.int32, tl.int32): (
               "rocshmem_ulong_put_signal_wrapper",
               (),
            ),
        }, is_pure=False, _semantic=_semantic)


@core.extern
def _barrier_all_impl(SCOPE_SUFFIX: core.constexpr, _semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {
            (): (f"rocshmem_barrier_all{SCOPE_SUFFIX.value}_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def barrier_all(_semantic=None):
    return _barrier_all_impl(core.constexpr(""), _semantic=_semantic)


@core.extern
def barrier_all_wave(_semantic=None):
    return _barrier_all_impl(core.constexpr("_wave"), _semantic=_semantic)


@core.extern
def barrier_all_wg(_semantic=None):
    return _barrier_all_impl(core.constexpr("_wg"), _semantic=_semantic)


@core.extern
def fence(_semantic=None):
    return extern_call(
        "librocshmem_device",
        "",
        [],
        {
            (): ("rocshmem_fence_wrapper", ()),
        },
        is_pure=False,
        _semantic=_semantic,
    )
