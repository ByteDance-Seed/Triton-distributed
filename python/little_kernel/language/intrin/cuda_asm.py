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
"""
CUDA intrinsics implemented using simple_builtin.

This module provides commonly used CUDA intrinsics implemented using the
simple_builtin decorator, which allows for concise definition of inline
assembly functions.
"""

import little_kernel.language as ll
from little_kernel.language.simple_builtin import simple_builtin

# ==============================================================================
# Shared Memory Operations
# ==============================================================================


@simple_builtin
def ld_shared_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32) -> ll.uint32:
    """Load 32-bit unsigned integer from shared memory."""
    ret: ll.uint32 = 0
    ll.asm("ld.shared.u32 %0, [%1];", ["=r", "r"], [ret, ptr + offset])
    return ret


@simple_builtin
def ld_shared_u64(ptr: ll.Tensor[ll.uint64], offset: ll.int32) -> ll.uint64:
    """Load 64-bit unsigned integer from shared memory."""
    ret: ll.uint64 = 0
    ll.asm("ld.shared.u64 %0, [%1];", ["=l", "l"], [ret, ptr + offset])
    return ret


@simple_builtin
def ld_shared_f32(ptr: ll.Tensor[ll.float32], offset: ll.int32) -> ll.float32:
    """Load 32-bit float from shared memory."""
    ret: ll.float32 = 0.0
    ll.asm("ld.shared.f32 %0, [%1];", ["=f", "r"], [ret, ptr + offset])
    return ret


@simple_builtin
def st_shared_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32, value: ll.uint32) -> ll.void:
    """Store 32-bit unsigned integer to shared memory."""
    ll.asm("st.shared.u32 [%0], %1;", ["r", "r"], [ptr + offset, value])


@simple_builtin
def st_shared_u64(ptr: ll.Tensor[ll.uint64], offset: ll.int32, value: ll.uint64) -> ll.void:
    """Store 64-bit unsigned integer to shared memory."""
    ll.asm("st.shared.u64 [%0], %1;", ["l", "l"], [ptr + offset, value])


@simple_builtin
def st_shared_f32(ptr: ll.Tensor[ll.float32], offset: ll.int32, value: ll.float32) -> ll.void:
    """Store 32-bit float to shared memory."""
    ll.asm("st.shared.f32 [%0], %1;", ["r", "f"], [ptr + offset, value])


# ==============================================================================
# Global Memory Operations
# ==============================================================================


@simple_builtin
def ld_global_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32) -> ll.uint32:
    """Load 32-bit unsigned integer from global memory."""
    ret: ll.uint32 = 0
    ll.asm("ld.global.u32 %0, [%1];", ["=r", "l"], [ret, ptr + offset])
    return ret


@simple_builtin
def ld_global_u64(ptr: ll.Tensor[ll.uint64], offset: ll.int32) -> ll.uint64:
    """Load 64-bit unsigned integer from global memory."""
    ret: ll.uint64 = 0
    ll.asm("ld.global.u64 %0, [%1];", ["=l", "l"], [ret, ptr + offset])
    return ret


@simple_builtin
def st_global_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32, value: ll.uint32) -> ll.void:
    """Store 32-bit unsigned integer to global memory."""
    ll.asm("st.global.u32 [%0], %1;", ["l", "r"], [ptr + offset, value])


@simple_builtin
def st_global_u64(ptr: ll.Tensor[ll.uint64], offset: ll.int32, value: ll.uint64) -> ll.void:
    """Store 64-bit unsigned integer to global memory."""
    ll.asm("st.global.u64 [%0], %1;", ["l", "l"], [ptr + offset, value])


# ==============================================================================
# Arithmetic Operations
# ==============================================================================


@simple_builtin
def fma_f32(a: ll.float32, b: ll.float32, c: ll.float32) -> ll.float32:
    """Fused multiply-add: a * b + c."""
    ret: ll.float32 = 0.0
    ll.asm("fma.rn.f32 %0, %1, %2, %3;", ["=f", "f", "f", "f"], [ret, a, b, c])
    return ret


@simple_builtin
def mul_lo_u32(a: ll.uint32, b: ll.uint32) -> ll.uint32:
    """Multiply low 32 bits: (a * b) & 0xFFFFFFFF."""
    ret: ll.uint32 = 0
    ll.asm("mul.lo.u32 %0, %1, %2;", ["=r", "r", "r"], [ret, a, b])
    return ret


@simple_builtin
def mul_hi_u32(a: ll.uint32, b: ll.uint32) -> ll.uint32:
    """Multiply high 32 bits: (a * b) >> 32."""
    ret: ll.uint32 = 0
    ll.asm("mul.hi.u32 %0, %1, %2;", ["=r", "r", "r"], [ret, a, b])
    return ret


@simple_builtin
def mad_lo_u32(a: ll.uint32, b: ll.uint32, c: ll.uint32) -> ll.uint32:
    """Multiply-add low: (a * b + c) & 0xFFFFFFFF."""
    ret: ll.uint32 = 0
    ll.asm("mad.lo.u32 %0, %1, %2, %3;", ["=r", "r", "r", "r"], [ret, a, b, c])
    return ret


# ==============================================================================
# Bit Operations
# ==============================================================================


@simple_builtin
def popc_u32(x: ll.uint32) -> ll.uint32:
    """Population count: number of set bits in x."""
    ret: ll.uint32 = 0
    ll.asm("popc.u32 %0, %1;", ["=r", "r"], [ret, x])
    return ret


@simple_builtin
def clz_u32(x: ll.uint32) -> ll.uint32:
    """Count leading zeros."""
    ret: ll.uint32 = 0
    ll.asm("clz.b32 %0, %1;", ["=r", "r"], [ret, x])
    return ret


@simple_builtin
def bfind_u32(x: ll.uint32) -> ll.uint32:
    """Find most significant bit position."""
    ret: ll.uint32 = 0
    ll.asm("bfind.u32 %0, %1;", ["=r", "r"], [ret, x])
    return ret


# ==============================================================================
# Warp Operations
# ==============================================================================


@simple_builtin
def ballot_sync(mask: ll.uint32, pred: ll.uint32) -> ll.uint32:
    """Warp ballot: returns mask of threads where pred is non-zero."""
    ret: ll.uint32 = 0
    ll.asm("vote.ballot.sync %0, %1, %2;", ["=r", "r", "r"], [ret, mask, pred])
    return ret


@simple_builtin
def shfl_sync_up(mask: ll.uint32, value: ll.uint32, delta: ll.uint32) -> ll.uint32:
    """Warp shuffle up: get value from thread with lower lane ID."""
    ret: ll.uint32 = 0
    ll.asm("shfl.sync.up.b32 %0, %1, %2, %3, %4;", ["=r", "r", "r", "r", "r"], [ret, value, delta, 0, mask])
    return ret


@simple_builtin
def shfl_sync_down(mask: ll.uint32, value: ll.uint32, delta: ll.uint32) -> ll.uint32:
    """Warp shuffle down: get value from thread with higher lane ID."""
    ret: ll.uint32 = 0
    ll.asm("shfl.sync.down.b32 %0, %1, %2, %3, %4;", ["=r", "r", "r", "r", "r"], [ret, value, delta, 0, mask])
    return ret


@simple_builtin
def shfl_sync_bfly(mask: ll.uint32, value: ll.uint32, delta: ll.uint32) -> ll.uint32:
    """Warp shuffle butterfly: XOR shuffle pattern."""
    ret: ll.uint32 = 0
    ll.asm("shfl.sync.bfly.b32 %0, %1, %2, %3, %4;", ["=r", "r", "r", "r", "r"], [ret, value, delta, 0, mask])
    return ret


# ==============================================================================
# Math Functions
# ==============================================================================


@simple_builtin
def sqrt_f32(x: ll.float32) -> ll.float32:
    """Square root."""
    ret: ll.float32 = 0.0
    ll.asm("sqrt.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def rsqrt_f32(x: ll.float32) -> ll.float32:
    """Reciprocal square root."""
    ret: ll.float32 = 0.0
    ll.asm("rsqrt.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def sin_f32(x: ll.float32) -> ll.float32:
    """Sine function (approximate)."""
    ret: ll.float32 = 0.0
    ll.asm("sin.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def cos_f32(x: ll.float32) -> ll.float32:
    """Cosine function (approximate)."""
    ret: ll.float32 = 0.0
    ll.asm("cos.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def exp2_f32(x: ll.float32) -> ll.float32:
    """Base-2 exponential (approximate, SFU)."""
    ret: ll.float32 = 0.0
    ll.asm("ex2.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def log2_f32(x: ll.float32) -> ll.float32:
    """Base-2 logarithm (approximate, SFU)."""
    ret: ll.float32 = 0.0
    ll.asm("lg2.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def rcp_f32(x: ll.float32) -> ll.float32:
    """Reciprocal (approximate, SFU)."""
    ret: ll.float32 = 0.0
    ll.asm("rcp.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


# ==============================================================================
# Cache Hint Loads
# ==============================================================================


@simple_builtin
def ldlu_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32) -> ll.uint32:
    """Load with last-use cache hint (evict-first)."""
    ret: ll.uint32 = 0
    ll.asm("ld.global.lu.u32 %0, [%1];", ["=r", "l"], [ret, ptr + offset])
    return ret


@simple_builtin
def ldcs_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32) -> ll.uint32:
    """Load with streaming cache hint (evict normally)."""
    ret: ll.uint32 = 0
    ll.asm("ld.global.cs.u32 %0, [%1];", ["=r", "l"], [ret, ptr + offset])
    return ret


@simple_builtin
def ldcv_u32(ptr: ll.Tensor[ll.uint32], offset: ll.int32) -> ll.uint32:
    """Load volatile (bypass L1 cache)."""
    ret: ll.uint32 = 0
    ll.asm("ld.global.cv.u32 %0, [%1];", ["=r", "l"], [ret, ptr + offset])
    return ret


# ==============================================================================
# Warp Shuffle (idx variant)
# ==============================================================================


@simple_builtin
def shfl_sync_idx(mask: ll.uint32, value: ll.uint32, srcLane: ll.uint32) -> ll.uint32:
    """Warp shuffle idx: get value from specific lane."""
    ret: ll.uint32 = 0
    ll.asm("shfl.sync.idx.b32 %0, %1, %2, %3, %4;", ["=r", "r", "r", "r", "r"], [ret, value, srcLane, 31, mask])
    return ret


# ==============================================================================
# Integer Arithmetic (for compute benchmarks)
# ==============================================================================


@simple_builtin
def add_u32(a: ll.uint32, b: ll.uint32) -> ll.uint32:
    """32-bit unsigned add."""
    ret: ll.uint32 = 0
    ll.asm("add.u32 %0, %1, %2;", ["=r", "r", "r"], [ret, a, b])
    return ret


@simple_builtin
def add_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """32-bit float add."""
    ret: ll.float32 = 0.0
    ll.asm("add.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret


@simple_builtin
def mul_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """32-bit float mul."""
    ret: ll.float32 = 0.0
    ll.asm("mul.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret
