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


@core.extern
def my_pe(_semantic=None):
    return extern_call(
        "libmori_shmem_device",
        "",
        [],
        {(): ("mori_shmem_my_pe", (tl.int32))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def n_pes(_semantic=None):
    return extern_call(
        "libmori_shmem_device",
        "",
        [],
        {(): ("mori_shmem_n_pes", (tl.int32))},
        is_pure=True,
        _semantic=_semantic,
    )

@core.extern
def remote_ptr(destPtr, myPe, destPe, _semantic=None):
    """Get P2P pointer to remote memory."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(destPtr, tl.uint64, _semantic=_semantic),
            tl.cast(myPe, tl.int32, _semantic=_semantic),
            tl.cast(destPe, tl.int32, _semantic=_semantic),
        ],
        {(tl.uint64, tl.int32, tl.int32): ("mori_shmem_ptr_p2p", (tl.uint64,))},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def quiet(_semantic=None):
    """Quiet operation for thread scope."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [],
        {(): ("mori_shmem_quiet_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def quiet_pe(pe, _semantic=None):
    """Quiet operation for thread scope to specific PE."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [tl.cast(pe, tl.int32, _semantic=_semantic)],
        {(tl.int32,): ("mori_shmem_quiet_thread_pe", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def quiet_pe_qp(pe, qp_id, _semantic=None):
    """Quiet operation for thread scope to specific PE and QP.
    
    Args:
        pe: Target PE number
        qp_id: Queue Pair ID
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.int32, tl.int32): ("mori_shmem_quiet_thread_pe_qp", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence(_semantic=None):
    """Fence operation for thread scope."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [],
        {(): ("mori_shmem_fence_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence_pe(pe, _semantic=None):
    """Fence operation for thread scope to specific PE."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [tl.cast(pe, tl.int32, _semantic=_semantic)],
        {(tl.int32,): ("mori_shmem_fence_thread_pe", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def fence_pe_qp(pe, qp_id, _semantic=None):
    """Fence operation for thread scope to specific PE and QP.
    
    Args:
        pe: Target PE number
        qp_id: Queue Pair ID
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.int32, tl.int32): ("mori_shmem_fence_thread_pe_qp", ())},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def putmem_nbi(dest, source, nbytes, pe, qp_id=0, _semantic=None):
    """Non-blocking put memory operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        source: Source pointer on local PE
        nbytes: Number of bytes to transfer
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(nbytes, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.int32, tl.int32): ("mori_shmem_putmem_nbi_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def put_uint32_nbi(dest, source, nelems, pe, qp_id=0, _semantic=None):
    """Non-blocking put uint32 operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        source: Source pointer on local PE
        nelems: Number of uint32 elements to transfer
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(nelems, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint32), tl.pointer_type(tl.uint32), tl.uint64, tl.int32, tl.int32): ("mori_shmem_put_uint32_nbi_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def put_uint64_nbi(dest, source, nelems, pe, qp_id=0, _semantic=None):
    """Non-blocking put uint64 operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        source: Source pointer on local PE
        nelems: Number of uint64 elements to transfer
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(nelems, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_put_uint64_nbi_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def put_float_nbi(dest, source, nelems, pe, qp_id=0, _semantic=None):
    """Non-blocking put float operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        source: Source pointer on local PE
        nelems: Number of float elements to transfer
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.float32), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.float32), _semantic=_semantic),
            tl.cast(nelems, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.float32), tl.pointer_type(tl.float32), tl.uint64, tl.int32, tl.int32): ("mori_shmem_put_float_nbi_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def put_double_nbi(dest, source, nelems, pe, qp_id=0, _semantic=None):
    """Non-blocking put double operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        source: Source pointer on local PE
        nelems: Number of double elements to transfer
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.float64), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.float64), _semantic=_semantic),
            tl.cast(nelems, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.float64), tl.pointer_type(tl.float64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_put_double_nbi_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def putmem_nbi_signal(dest, source, bytes, sig_addr, signal_value, signal_op, pe, qp_id=0, _semantic=None):
    """Non-blocking put memory with signal operation (thread scope).
    
    Performs a non-blocking memory transfer and atomically updates a signal value 
    at the destination PE after the data transfer completes.
    
    Args:
        dest: Symmetric address on target PE for data
        source: Source pointer on local PE
        bytes: Number of bytes to transfer
        sig_addr: Symmetric address on target PE for signal
        signal_value: Signal value to write
        signal_op: Signal operation type (e.g., MORI_SIGNAL_SET, MORI_SIGNAL_ADD)
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(source, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(bytes, tl.uint64, _semantic=_semantic),
            tl.cast(sig_addr, tl.pointer_type(tl.void), _semantic=_semantic),
            tl.cast(signal_value, tl.uint64, _semantic=_semantic),
            tl.cast(signal_op, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.void), tl.pointer_type(tl.void), tl.uint64, tl.pointer_type(tl.void), tl.uint64, tl.int32, tl.int32, tl.int32): ("mori_shmem_putmem_nbi_signal_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def int_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single int value.
    
    Args:
        dest: Symmetric address on target PE
        value: Integer value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int32), _semantic=_semantic),
            tl.cast(value, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int32), tl.int32, tl.int32, tl.int32): ("mori_shmem_int_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def long_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single long value.
    
    Args:
        dest: Symmetric address on target PE
        value: Long value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(value, tl.int64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int32, tl.int32): ("mori_shmem_long_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def longlong_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single long long value.
    
    Args:
        dest: Symmetric address on target PE
        value: Long long value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(value, tl.int64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int32, tl.int32): ("mori_shmem_longlong_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def float_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single float value.
    
    Args:
        dest: Symmetric address on target PE
        value: Float value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.float32), _semantic=_semantic),
            tl.cast(value, tl.float32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.float32), tl.float32, tl.int32, tl.int32): ("mori_shmem_float_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def double_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single double value.
    
    Args:
        dest: Symmetric address on target PE
        value: Double value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.float64), _semantic=_semantic),
            tl.cast(value, tl.float64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.float64), tl.float64, tl.int32, tl.int32): ("mori_shmem_double_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def char_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single char value.
    
    Args:
        dest: Symmetric address on target PE
        value: Char value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int8), _semantic=_semantic),
            tl.cast(value, tl.int8, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int8), tl.int8, tl.int32, tl.int32): ("mori_shmem_char_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def short_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single short value.
    
    Args:
        dest: Symmetric address on target PE
        value: Short value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int16), _semantic=_semantic),
            tl.cast(value, tl.int16, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int16), tl.int16, tl.int32, tl.int32): ("mori_shmem_short_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uint_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single unsigned int value.
    
    Args:
        dest: Symmetric address on target PE
        value: Unsigned int value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(value, tl.uint32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint32), tl.uint32, tl.int32, tl.int32): ("mori_shmem_uint_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ulong_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single unsigned long value.
    
    Args:
        dest: Symmetric address on target PE
        value: Unsigned long value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(value, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_ulong_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ulonglong_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single unsigned long long value.
    
    Args:
        dest: Symmetric address on target PE
        value: Unsigned long long value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(value, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_ulonglong_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uchar_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single unsigned char value.
    
    Args:
        dest: Symmetric address on target PE
        value: Unsigned char value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint8), _semantic=_semantic),
            tl.cast(value, tl.uint8, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint8), tl.uint8, tl.int32, tl.int32): ("mori_shmem_uchar_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def ushort_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single unsigned short value.
    
    Args:
        dest: Symmetric address on target PE
        value: Unsigned short value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint16), _semantic=_semantic),
            tl.cast(value, tl.uint16, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint16), tl.uint16, tl.int32, tl.int32): ("mori_shmem_ushort_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int32_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single int32_t value.
    
    Args:
        dest: Symmetric address on target PE
        value: Int32 value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int32), _semantic=_semantic),
            tl.cast(value, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int32), tl.int32, tl.int32, tl.int32): ("mori_shmem_int32_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int64_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single int64_t value.
    
    Args:
        dest: Symmetric address on target PE
        value: Int64 value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(value, tl.int64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int32, tl.int32): ("mori_shmem_int64_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uint32_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single uint32_t value.
    
    Args:
        dest: Symmetric address on target PE
        value: Uint32 value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(value, tl.uint32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint32), tl.uint32, tl.int32, tl.int32): ("mori_shmem_uint32_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uint64_p(dest, value, pe, qp_id=0, _semantic=None):
    """Put single uint64_t value.
    
    Args:
        dest: Symmetric address on target PE
        value: Uint64 value to write
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(value, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_uint64_p", ())},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def atomic_uint64_nonfetch(dest, val, amoType, pe, qp_id=0, _semantic=None):
    """Atomic non-fetch uint64 operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value to be used in atomic operation
        amoType: Atomic nonfetch operation type (AMO_ADD, etc.)
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(val, tl.uint64, _semantic=_semantic),
            tl.cast(amoType, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32, tl.int32): ("mori_shmem_atomic_uint64_nonfetch_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_uint64_fetch(dest, val, compare, amoType, pe, qp_id=0, _semantic=None):
    """Atomic fetch uint64 operation (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value to be used in atomic operation
        compare: Compare value for CAS operations
        amoType: Atomic fetch operation type (AMO_FETCH_ADD, AMO_FETCH_CAS, etc.)
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    
    Returns:
        The original value at dest before the operation
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(val, tl.uint64, _semantic=_semantic),
            tl.cast(compare, tl.uint64, _semantic=_semantic),
            tl.cast(amoType, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.uint64, tl.int32, tl.int32, tl.int32): ("mori_shmem_atomic_uint64_fetch_thread", (tl.uint64,))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_add_uint64(dest, val, pe, qp_id=0, _semantic=None):
    """Atomic add uint64 (thread scope, no fetch).
    
    Args:
        dest: Symmetric address on target PE
        val: Value to add
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(val, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_uint64_atomic_add_thread", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def atomic_fetch_add_uint64(dest, val, pe, qp_id=0, _semantic=None):
    """Atomic fetch-add uint64 (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value to add
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    
    Returns:
        The original value at dest before the add operation
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(val, tl.uint64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64, tl.int32, tl.int32): ("mori_shmem_uint64_atomic_fetch_add_thread", (tl.uint64,))},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def atomic_int64_nonfetch(dest, val, amo_type, pe, qp_id=0, _semantic=None):
    """Generic atomic non-fetch operation on int64 (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value for atomic operation
        amo_type: Atomic operation type (e.g., MORI_AMO_ADD, MORI_AMO_SET)
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(val, tl.int64, _semantic=_semantic),
            tl.cast(amo_type, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int32, tl.int32, tl.int32): ("mori_shmem_atomic_int64_nonfetch_thread", (tl.void,))},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def atomic_int64_fetch(dest, val, compare, amo_type, pe, qp_id=0, _semantic=None):
    """Generic atomic fetch operation on int64 (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value for atomic operation
        compare: Compare value (used for compare-swap)
        amo_type: Atomic operation type (e.g., MORI_AMO_FETCH_ADD, MORI_AMO_SWAP)
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    
    Returns:
        The value at dest before the atomic operation
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(val, tl.int64, _semantic=_semantic),
            tl.cast(compare, tl.int64, _semantic=_semantic),
            tl.cast(amo_type, tl.int32, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int64, tl.int32, tl.int32, tl.int32): ("mori_shmem_atomic_int64_fetch_thread", (tl.int64,))},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def atomic_add_int64(dest, val, pe, qp_id=0, _semantic=None):
    """Atomic add int64 (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value to add
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    """
    extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(val, tl.int64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int32, tl.int32): ("mori_shmem_int64_atomic_add_thread", (tl.void,))},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def atomic_fetch_add_int64(dest, val, pe, qp_id=0, _semantic=None):
    """Atomic fetch-add int64 (thread scope).
    
    Args:
        dest: Symmetric address on target PE
        val: Value to add
        pe: Target PE number
        qp_id: Queue Pair ID (default: 0)
    
    Returns:
        The original value at dest before the add operation
    """
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(dest, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(val, tl.int64, _semantic=_semantic),
            tl.cast(pe, tl.int32, _semantic=_semantic),
            tl.cast(qp_id, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64, tl.int32, tl.int32): ("mori_shmem_int64_atomic_fetch_add_thread", (tl.int64,))},
        is_pure=False,
        _semantic=_semantic,
    )

@core.extern
def uint32_wait_until_greater_than(addr, val, _semantic=None):
    """Wait until uint32 value is greater than specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(val, tl.uint32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint32), tl.uint32): ("mori_shmem_uint32_wait_until_greater_than", (tl.uint32,))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uint32_wait_until_equals(addr, val, _semantic=None):
    """Wait until uint32 value equals specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.uint32), _semantic=_semantic),
            tl.cast(val, tl.uint32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint32), tl.uint32): ("mori_shmem_uint32_wait_until_equals", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uint64_wait_until_greater_than(addr, val, _semantic=None):
    """Wait until uint64 value is greater than specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(val, tl.uint64, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64): ("mori_shmem_uint64_wait_until_greater_than", (tl.uint64,))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def uint64_wait_until_equals(addr, val, _semantic=None):
    """Wait until uint64 value equals specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.uint64), _semantic=_semantic),
            tl.cast(val, tl.uint64, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.uint64), tl.uint64): ("mori_shmem_uint64_wait_until_equals", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int32_wait_until_greater_than(addr, val, _semantic=None):
    """Wait until int32 value is greater than specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.int32), _semantic=_semantic),
            tl.cast(val, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int32), tl.int32): ("mori_shmem_int32_wait_until_greater_than", (tl.int32,))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int32_wait_until_equals(addr, val, _semantic=None):
    """Wait until int32 value equals specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.int32), _semantic=_semantic),
            tl.cast(val, tl.int32, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int32), tl.int32): ("mori_shmem_int32_wait_until_equals", ())},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int64_wait_until_greater_than(addr, val, _semantic=None):
    """Wait until int64 value is greater than specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(val, tl.int64, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64): ("mori_shmem_int64_wait_until_greater_than", (tl.int64,))},
        is_pure=False,
        _semantic=_semantic,
    )


@core.extern
def int64_wait_until_equals(addr, val, _semantic=None):
    """Wait until int64 value equals specified value."""
    return extern_call(
        "libmori_shmem_device",
        "",
        [
            tl.cast(addr, tl.pointer_type(tl.int64), _semantic=_semantic),
            tl.cast(val, tl.int64, _semantic=_semantic),
        ],
        {(tl.pointer_type(tl.int64), tl.int64): ("mori_shmem_int64_wait_until_equals", ())},
        is_pure=False,
        _semantic=_semantic,
    )
