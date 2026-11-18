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
from little_kernel.language.builtin_base import Builtin, builtin
from little_kernel.language.simple_builtin import simple_builtin
from little_kernel.core.type_system import void, uint8, uint16, uint32, uint64, float32, float64
from little_kernel.core.compile import ll_kernel
from little_kernel.backends.nv.common.language.intrin.simt import codegen_elect_one_sync

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


@simple_builtin
def st_global_cs_u64(ptr: ll.Tensor[ll.uint64], offset: ll.int32, value: ll.uint64) -> ll.void:
    """Store 64-bit unsigned integer to global memory, evict-first (.cs).

    Same 8-byte alignment legality precondition as ``st_global_u64`` (PTX
    ``st.global.u64``); the ``.cs`` cache policy is bit-exact and is purely
    a performance hint for single-touch data, mirroring the accepted
    ``st_global_cs_v4_u32`` family in intrin/memory.py.
    """
    ll.asm("st.global.cs.u64 [%0], %1;", ["l", "l"], [ptr + offset, value])


# ==============================================================================
# Arithmetic Operations
# ==============================================================================


def codegen_pack_f32x2(lo, hi):
    body = r'''
__device__ __forceinline__ uint64_t pack_f32x2_fn(float lo, float hi) {
    uint64_t value;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(value) : "f"(lo), "f"(hi));
    return value;
}
'''
    return Builtin(body=body, includes=[], return_val=f"pack_f32x2_fn({lo}, {hi})")


@builtin(eval_return_type=ll.uint64, codegen_func=codegen_pack_f32x2)
def pack_f32x2(lo, hi):
    """Pack two FP32 registers for PTX packed arithmetic."""
    raise RuntimeError("should not call pack_f32x2 in compilation")


def codegen_mul_f32x2(a, b):
    body = r'''
__device__ __forceinline__ uint64_t mul_f32x2_fn(uint64_t a, uint64_t b) {
    uint64_t value;
    asm volatile("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(value) : "l"(a), "l"(b));
    return value;
}
'''
    return Builtin(body=body, includes=[], return_val=f"mul_f32x2_fn({a}, {b})")


@builtin(eval_return_type=ll.uint64, codegen_func=codegen_mul_f32x2)
def mul_f32x2(a, b):
    """Multiply two packed FP32x2 values."""
    raise RuntimeError("should not call mul_f32x2 in compilation")


def codegen_sub_f32x2(a, b):
    body = r'''
__device__ __forceinline__ uint64_t sub_f32x2_fn(uint64_t a, uint64_t b) {
    uint64_t value;
    asm volatile("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(value) : "l"(a), "l"(b));
    return value;
}
'''
    return Builtin(body=body, includes=[], return_val=f"sub_f32x2_fn({a}, {b})")


@builtin(eval_return_type=ll.uint64, codegen_func=codegen_sub_f32x2)
def sub_f32x2(a, b):
    """Subtract two packed FP32x2 values."""
    raise RuntimeError("should not call sub_f32x2 in compilation")


def codegen_fma_f32x2(a, b, c):
    body = r'''
__device__ __forceinline__ uint64_t fma_f32x2_fn(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t value;
    asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(value) : "l"(a), "l"(b), "l"(c));
    return value;
}
'''
    return Builtin(body=body, includes=[], return_val=f"fma_f32x2_fn({a}, {b}, {c})")


@builtin(eval_return_type=ll.uint64, codegen_func=codegen_fma_f32x2)
def fma_f32x2(a, b, c):
    """Compute ``a * b + c`` on two packed FP32 lanes."""
    raise RuntimeError("should not call fma_f32x2 in compilation")


@simple_builtin
def fma_f32(a: ll.float32, b: ll.float32, c: ll.float32) -> ll.float32:
    """Fused multiply-add: a * b + c."""
    ret: ll.float32 = 0.0
    ll.asm("fma.rn.f32 %0, %1, %2, %3;", ["=f", "f", "f", "f"], [ret, a, b, c])
    return ret


@simple_builtin
def rsqrt_f32(x: ll.float32) -> ll.float32:
    """Reciprocal square root via rsqrt.approx.f32 (MUFU.RSQ).

    Approximate: max relative error ~2^-23 per the PTX ISA, NOT
    IEEE-correctly-rounded.  Suitable where the consumer tolerance admits it
    (e.g. 1e-4 layer-statistic budgets); do not use where a bitwise contract
    demands correctly-rounded sqrt/division.
    """
    ret: ll.float32 = 0.0
    ll.asm("rsqrt.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
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


@simple_builtin
def prmt(a: ll.uint32, b: ll.uint32, sel: ll.uint32) -> ll.uint32:
    """Byte-permute across the 8-byte pair {b, a}: prmt.b32 (mode 0).

    Each of the four low nibbles of ``sel`` selects one result byte from the
    8-byte pool formed by a's 4 bytes (indices 0-3) followed by b's 4 bytes
    (indices 4-7); nibble bit 3 is the sign-replicate mode and is unused in
    the default gather mode. This is the generic two-register byte gather
    used for register-level 4x4 byte transposes in transposing-store
    epilogues (e.g. fp8 quantize+transpose staging), replacing an SMEM byte
    round trip. A constant selector literal folds into the asm operand as an
    immediate through the usual constant-propagation path."""
    ret: ll.uint32 = 0
    ll.asm("prmt.b32 %0, %1, %2, %3;", ["=r", "r", "r", "r"], [ret, a, b, sel])
    return ret


@simple_builtin
def prefix_lsb_mask_i32(count: ll.int32) -> ll.uint32:
    """Set the lowest clamp(count, 0, 32) bits."""
    ret: ll.uint32 = 0
    ll.asm(
        "{\n"
        ".reg .pred p_le_zero;\n"
        ".reg .b32 start, mask;\n"
        "setp.le.s32 p_le_zero, %1, 0;\n"
        "mov.u32 start, 0;\n"
        "bmsk.clamp.b32 mask, start, %1;\n"
        "selp.b32 %0, 0, mask, p_le_zero;\n"
        "}",
        ["=r", "r"],
        [ret, count],
    )
    return ret


@simple_builtin
def selp_b32(
    pred: ll.uint32,
    on_true: ll.uint32,
    on_false: ll.uint32,
) -> ll.uint32:
    """Select one 32-bit value according to whether pred is nonzero."""
    ret: ll.uint32 = 0
    ll.asm(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.u32 p, %1, 0;\n"
        "selp.b32 %0, %2, %3, p;\n"
        "}",
        ["=r", "r", "r", "r"],
        [ret, pred, on_true, on_false],
    )
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
    """Square root (correctly rounded, IEEE RN; body emits sqrt.rn.f32)."""
    ret: ll.float32 = 0.0
    ll.asm("sqrt.rn.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def sin_f32(x: ll.float32) -> ll.float32:
    """Sine function (approximate)."""
    ret: ll.float32 = 0.0
    ll.asm("sin.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


# ==============================================================================
# Bare non-ftz round-to-nearest fp32 arithmetic
#
# The ``__fadd_rn`` / ``__fmul_rn`` / ``__fsub_rn`` / ``__fdiv_rn`` device
# intrinsics pin round-to-nearest even under ``--use_fast_math`` but are
# still compiled with the module-wide ftz mode, so their PTX is
# ``add.rn.ftz.f32`` etc. and denormal results flush to zero.  Kernels with a
# bitwise contract on exact denormal FP32 values (e.g. softmax weights with
# gaps in 88..104 or subnormal gradients) need the bare non-ftz form.  These
# builtins emit ``add.rn.f32`` / ``mul.rn.f32`` / ``sub.rn.f32`` /
# ``div.rn.f32`` via inline asm, which the compiler cannot tag with ``.ftz``
# and cannot contract into an fma, regardless of ``--use_fast_math``.
# ==============================================================================


@simple_builtin
def add_rn_noftz_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """Non-ftz correctly-rounded (RN) float32 addition; denormal outputs
    preserved, not fma-fusable, immune to ``--use_fast_math``."""
    ret: ll.float32 = 0.0
    ll.asm("add.rn.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret


@simple_builtin
def sub_rn_noftz_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """Non-ftz correctly-rounded (RN) float32 subtraction; denormal outputs
    preserved, not fma-fusable, immune to ``--use_fast_math``."""
    ret: ll.float32 = 0.0
    ll.asm("sub.rn.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret


@simple_builtin
def mul_rn_noftz_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """Non-ftz correctly-rounded (RN) float32 multiplication; denormal
    outputs preserved, not fma-fusable, immune to ``--use_fast_math``."""
    ret: ll.float32 = 0.0
    ll.asm("mul.rn.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret


@simple_builtin
def div_rn_noftz_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """Non-ftz correctly-rounded (RN) float32 division; denormal outputs
    preserved, never lowered to ``div.approx.f32`` or ``div.rn.ftz.f32``
    under ``--use_fast_math``."""
    ret: ll.float32 = 0.0
    ll.asm("div.rn.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret


@simple_builtin
def cos_f32(x: ll.float32) -> ll.float32:
    """Cosine function (approximate)."""
    ret: ll.float32 = 0.0
    ll.asm("cos.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def exp2_approx_f32(x: ll.float32) -> ll.float32:
    """Base-2 exponential (approximate, SFU)."""
    ret: ll.float32 = 0.0
    ll.asm("ex2.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def tanh_approx(x: ll.float32) -> ll.float32:
    """Hyperbolic tangent (approximate, SFU ``tanh.approx.f32``).

    Same instruction ``--use_fast_math`` maps ``tanhf`` to; one MUFU op.
    """
    ret: ll.float32 = 0.0
    ll.asm("tanh.approx.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


def codegen_exp2f(x):
    return Builtin(body="", includes=[], return_val=f"exp2f({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_exp2f)
def exp2f(x):
    """Base-2 exponential using CUDA ``exp2f``."""
    raise RuntimeError("should not call exp2f in compilation")


def codegen_expf(x):
    return Builtin(body="", includes=[], return_val=f"expf({x})")


# ==============================================================================
# Accurate libdevice math entry points
#
# ``tanhf``/``expf``/``logf`` bind directly to libdevice's accurate entry
# points (__nv_*), immune to ``--use_fast_math`` remapping.  The generic
# round-to-nearest arithmetic builtins (fadd_rn/fmul_rn/fdiv_rn) live in the
# single _codegen_rn_binary family defined later in this module, so they are
# not duplicated here; this section defines only the unary functions that do
# not already have a canonical owner.
# ==============================================================================
def codegen_tanhf_precise(x):
    # libdevice's accurate entry points are resolved from libdevice.10.bc at
    # link time; nvcc needs an explicit device prototype (same pattern the
    # Kernel Wiki fused-LSTM-cell reference uses).
    body = '\nextern "C" __device__ float __nv_tanhf(float);\n'
    return Builtin(body=body, includes=[], return_val=f"__nv_tanhf({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_tanhf_precise)
def tanhf(x):
    """Hyperbolic tangent, libdevice accurate ``__nv_tanhf`` (RN-exact, never
    lowered to a fast-math approximation)."""
    raise RuntimeError("should not call tanhf in compilation")


def codegen_expf_precise(x):
    body = '\nextern "C" __device__ float __nv_expf(float);\n'
    return Builtin(body=body, includes=[], return_val=f"__nv_expf({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_expf_precise)
def expf(x):
    """Natural exponential, libdevice accurate ``__nv_expf`` (RN-exact, never
    lowered to a fast-math approximation)."""
    raise RuntimeError("should not call expf in compilation")


def codegen_logf_precise(x):
    body = '\nextern "C" __device__ float __nv_logf(float);\n'
    return Builtin(body=body, includes=[], return_val=f"__nv_logf({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_logf_precise)
def logf(x):
    """Natural logarithm, libdevice accurate ``__nv_logf`` (RN-exact, never
    lowered to a fast-math approximation)."""
    raise RuntimeError("should not call logf in compilation")


def codegen_bf16x2_lane_to_float(value, lane):
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in ("0", "1"):
        raise ValueError("bf16x2_lane_to_float requires a compile-time literal lane in "
                         f"{{0,1}} (got {lane!r}); bf16x2 lanes live in one register and "
                         "cannot be indexed dynamically")
    body = r"""
__device__ __forceinline__ float bf16x2_lane_to_float_fn(uint32_t value, int lane) {
    const __nv_bfloat162 pair = *reinterpret_cast<const __nv_bfloat162*>(&value);
    return __bfloat162float(lane ? pair.y : pair.x);
}
"""
    return Builtin(body=body, includes=[], return_val=f"bf16x2_lane_to_float_fn({value}, {lane_str})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_bf16x2_lane_to_float)
def bf16x2_lane_to_float(value, lane):
    """Unpack one lane (compile-time literal 0=lo, 1=hi) of a BF16x2-packed
    uint32 to FP32. Complements ``pack_bf16`` for vectorized BF16 pipelines
    with FP32 math."""
    raise RuntimeError("should not call bf16x2_lane_to_float in compilation")


# ==============================================================================
# FP16 (half2) pair helpers — exact mirrors of the BF16 helpers above, for
# 16-bit elementwise/reduction pipelines on fp16 storage: after a 128/64-bit
# vector load each u32 lane holds a packed __half2 (lo=x, hi=y).
# ==============================================================================


def codegen_half2_lane_to_float(value, lane):
    lane_str = str(lane).strip().strip("()").strip()
    if lane_str not in ("0", "1"):
        raise ValueError("half2_lane_to_float requires a compile-time literal lane in "
                         f"{{0,1}} (got {lane!r}); half2 lanes live in one register and "
                         "cannot be indexed dynamically")
    body = r"""
#include <cuda_fp16.h>
__device__ __forceinline__ float half2_lane_to_float_fn(uint32_t value, int lane) {
    const __half2 pair = *reinterpret_cast<const __half2*>(&value);
    return __half2float(lane ? pair.y : pair.x);
}
"""
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"half2_lane_to_float_fn({value}, {lane_str})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_half2_lane_to_float)
def half2_lane_to_float(value, lane):
    """Unpack one lane (compile-time literal 0=lo, 1=hi) of a HALF2-packed
    uint32 to FP32. Complements ``pack_half2`` for vectorized FP16 pipelines
    with FP32 math."""
    raise RuntimeError("should not call half2_lane_to_float in compilation")


def codegen_pack_half2(fp32_a, fp32_b):
    body = """
#include <cuda_fp16.h>
__device__ __forceinline__ uint32_t pack_half2_fn(uint32_t fp32_a, uint32_t fp32_b) {
    float2 f2;
    f2.x = __uint_as_float(fp32_a);
    f2.y = __uint_as_float(fp32_b);
    __half2 h2 = __floats2half2_rn(f2.x, f2.y);
    return *reinterpret_cast<uint32_t*>(&h2);
}
"""
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"pack_half2_fn({fp32_a}, {fp32_b})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_half2)
def pack_half2(fp32_a, fp32_b):
    """Pack two FP32 values (as uint32 bit patterns) into one uint32 of two
    IEEE fp16 (RNE). FP16 twin of ``pack_bf16``; only the low fp16 pair
    pattern differs."""
    raise RuntimeError("should not call pack_half2 in compilation")


def codegen_erff(x):
    return Builtin(body="", includes=[], return_val=f"erff({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_erff)
def erff(x):
    """Error function using CUDA ``erff`` (exact, not a tanh approximation)."""
    raise RuntimeError("should not call erff in compilation")


# Fast-math-immune accurate fp32 math.
#
# The task build contract uses ``-O3 --use_fast_math``, under which nvcc silently
# remaps ``expf``/``logf``/``sinf``/... to the fast SFU approximations.  The only
# way to reach libdevice accuracy from generated source is to call the manglers
# directly (``__nv_expf`` & friends cannot be touched by -use_fast_math) or to use
# the explicit round-to-nearest compiler intrinsics (``__fadd_rn`` & friends, which
# are immune to fast-math contraction).  These builtins follow the intrinsic
# registry discipline: one canonical op per name, fp32 lane type, no string
# matching on task or tensor names.
#
# ``exp_precise_f32`` is a published alias of ``expf_rn``: both lower to the same
# libdevice ``__nv_expf`` semantics and both names are retained so frontends keep
# compiling unchanged.  Co-emission is safe: each builtin body declares the same
# ``extern "C" __device__ float __nv_expf(float);`` prototype, and duplicate
# identical extern-"C" declarations are legal in C++; the body-cache symbols
# (``exp_precise_f32_fn`` vs. inline declarations) do not collide.


def _codegen_libdevice_unary(mangled):

    def codegen(x):
        body = f'extern "C" __device__ float {mangled}(float);'
        return Builtin(body=body, includes=[], return_val=f"{mangled}({x})")

    return codegen


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_libdevice_unary("__nv_expf"))
def expf_rn(x):
    """Accurate IEEE fp32 exponential via libdevice (immune to --use_fast_math)."""
    raise RuntimeError("should not call expf_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_libdevice_unary("__nv_logf"))
def logf_rn(x):
    """Accurate IEEE fp32 natural log via libdevice (immune to --use_fast_math)."""
    raise RuntimeError("should not call logf_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_libdevice_unary("__nv_log1pf"))
def log1pf_rn(x):
    """Accurate IEEE fp32 log(1+x) via libdevice (immune to --use_fast_math)."""
    raise RuntimeError("should not call log1pf_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_libdevice_unary("__nv_expm1f"))
def expm1f_rn(x):
    """Accurate IEEE fp32 exp(x)-1 via libdevice (immune to --use_fast_math)."""
    raise RuntimeError("should not call expm1f_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_libdevice_unary("__nv_sinf"))
def sinf_rn(x):
    """Accurate IEEE fp32 sine via libdevice (immune to --use_fast_math)."""
    raise RuntimeError("should not call sinf_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_libdevice_unary("__nv_cosf"))
def cosf_rn(x):
    """Accurate IEEE fp32 cosine via libdevice (immune to --use_fast_math)."""
    raise RuntimeError("should not call cosf_rn in compilation")


def codegen_sincosf_rn(x, s, c):
    body = 'extern "C" __device__ void __nv_sincosf(float, float*, float*);'
    return Builtin(body=body, includes=[], return_val=f"__nv_sincosf({x}, &{s}, &{c})")


@builtin(eval_return_type=void, codegen_func=codegen_sincosf_rn)
def sincosf_rn(x, s, c):
    """Accurate IEEE fp32 simultaneous sine/cosine, written into local floats s, c."""
    raise RuntimeError("should not call sincosf_rn in compilation")


def _codegen_rn_binary(name):

    def codegen(a, b):
        return Builtin(body="", includes=[], return_val=f"{name}({a}, {b})")

    return codegen


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_rn_binary("__fadd_rn"))
def fadd_rn(a, b):
    """IEEE round-to-nearest fp32 add (immune to --use_fast_math contraction)."""
    raise RuntimeError("should not call fadd_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_rn_binary("__fsub_rn"))
def fsub_rn(a, b):
    """IEEE round-to-nearest fp32 subtract (immune to --use_fast_math)."""
    raise RuntimeError("should not call fsub_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_rn_binary("__fmul_rn"))
def fmul_rn(a, b):
    """IEEE round-to-nearest fp32 multiply (immune to --use_fast_math)."""
    raise RuntimeError("should not call fmul_rn in compilation")


@builtin(eval_return_type=ll.float32, codegen_func=_codegen_rn_binary("__fdiv_rn"))
def fdiv_rn(a, b):
    """IEEE round-to-nearest fp32 divide (immune to --use_fast_math)."""
    raise RuntimeError("should not call fdiv_rn in compilation")


def codegen_fmaf_rn(a, b, c):
    return Builtin(body="", includes=[], return_val=f"__fmaf_rn({a}, {b}, {c})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_fmaf_rn)
def fmaf_rn(a, b, c):
    """IEEE round-to-nearest fp32 fused multiply-add (immune to --use_fast_math)."""
    raise RuntimeError("should not call fmaf_rn in compilation")


# --------------------------------------------------------------------------------------
# Packed fp8 conversions (OCP e4m3fn / e5m2, SATFINITE semantics).
#
# These lower through cuda_fp8.h's `__nv_cvt_float2_to_fp8x2(..., __NV_SATFINITE,
# __NV_E4M3|__NV_E5M2)`, which on sm_89+ is exactly one
# `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` instruction per call (clamp-to-max-finite,
# RNE tie-to-even, Inf saturation, NaN canonicalization in a single step) and on
# older targets an exact header-emulated sequence with the same semantics.  A
# *conversion* instruction is never rewritten by `--use_fast_math`, so these are
# immune to the task build's fast-math flag (unlike arithmetic on the way in,
# which the caller shields with the `_rn` family above where bit-exactness is
# required).
#
# Byte order contract (verified against cuda_fp8.hpp:267-292 at the pinned
# toolchain: asm operand `%2 = x.y` lands in the high byte): the first argument
# `hi` occupies bits [15:8] of the returned uint16, the second argument `lo`
# bits [7:0], so a little-endian 16-bit store lays the pair down in element
# order `lo, hi`.
#
# Registry discipline (compiler-wiki `cw.codegen.tvm-cuda-source-visitor`,
# item 5): one canonical entry per (op, source dtype, destination dtype)
# triple, no task-name or string matching; legality (sm_89 vs emulation) is
# decided by the header from the target attribute, not by the frontend.


def _codegen_cvt_satfinite_fp8x2(fn_name, fp8_enum):

    def codegen(hi, lo):
        body = f'''
__device__ __forceinline__ uint16_t {fn_name}_fn(float hi, float lo) {{
    const float2 v = make_float2(lo, hi);
    return static_cast<uint16_t>(__nv_cvt_float2_to_fp8x2(v, __NV_SATFINITE, {fp8_enum}));
}}
'''
        return Builtin(body=body, includes=["<cuda_fp8.h>"], return_val=f"{fn_name}_fn({hi}, {lo})")

    return codegen


@builtin(eval_return_type=uint16, codegen_func=_codegen_cvt_satfinite_fp8x2("cvt_satfinite_e4m3x2_f32", "__NV_E4M3"))
def cvt_satfinite_e4m3x2_f32(hi, lo):
    """Convert two fp32 values to packed e4m3fn SATFINITE codes.

    Returns a uint16 whose high byte is the SATFINITE e4m3 encoding of `hi`
    and whose low byte is the encoding of `lo`.
    """
    raise RuntimeError("should not call cvt_satfinite_e4m3x2_f32 in compilation")


@builtin(eval_return_type=uint16, codegen_func=_codegen_cvt_satfinite_fp8x2("cvt_satfinite_e5m2x2_f32", "__NV_E5M2"))
def cvt_satfinite_e5m2x2_f32(hi, lo):
    """Convert two fp32 values to packed e5m2 SATFINITE codes.

    Returns a uint16 whose high byte is the SATFINITE e5m2 encoding of `hi`
    and whose low byte is the encoding of `lo`.
    """
    raise RuntimeError("should not call cvt_satfinite_e5m2x2_f32 in compilation")


def codegen_exp_f32(x):
    return Builtin(body="", includes=[], return_val=f"expf({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_exp_f32)
def exp_f32(x):
    """Base-e exponential using CUDA ``expf`` (libm-exact, not ``__expf``)."""
    raise RuntimeError("should not call exp_f32 in compilation")


def codegen_log1p_f32(x):
    return Builtin(body="", includes=[], return_val=f"log1pf({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_log1p_f32)
def log1p_f32(x):
    """``log(1+x)`` using CUDA ``log1pf`` (libm-exact near zero)."""
    raise RuntimeError("should not call log1p_f32 in compilation")


def codegen_pow_f32(base, exponent):
    return Builtin(body="", includes=[], return_val=f"powf({base}, {exponent})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_pow_f32)
def pow_f32(base, exponent):
    """Power using CUDA ``powf`` (libm semantics, e.g. beta**step bias terms)."""
    raise RuntimeError("should not call pow_f32 in compilation")


# Published alias: semantically identical to ``expf_rn`` (same libdevice
# ``__nv_expf`` lowering), retained for source compatibility; see the
# fast-math-immune math section above.
def codegen_exp_precise_f32(x):
    body = r"""
extern "C" __device__ float __nv_expf(float);
__device__ __forceinline__ float exp_precise_f32_fn(float x) { return __nv_expf(x); }
"""
    return Builtin(body=body, includes=[], return_val=f"exp_precise_f32_fn({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_exp_precise_f32)
def exp_precise_f32(x):
    """Precise single-precision exponential via libdevice ``__nv_expf``.

    Immune to ``--use_fast_math`` (which remaps ``expf`` to the fast
    ``__expf``); use when a frontend needs ``expf``-grade accuracy under a
    fast-math build. The helper body is emitted exactly once per generated
    file via the builtin body cache.
    """
    raise RuntimeError("should not call exp_precise_f32 in compilation")


def codegen_exp2_emu_f32x2(x, y):
    body = r"""
__device__ __forceinline__ uint64_t exp2_emu_f32x2_fn(float x, float y) {
    uint64_t out;
    asm volatile(
        "{\n\t"
        ".reg .f32 f1, f2, f3, f4, f5, f6, f7;\n\t"
        ".reg .b64 l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;\n\t"
        ".reg .s32 r1, r2, r3, r4, r5, r6, r7, r8;\n\t"
        "max.ftz.f32 f1, %1, 0fC2FE0000;\n\t"
        "max.ftz.f32 f2, %2, 0fC2FE0000;\n\t"
        "mov.b64 l1, {f1, f2};\n\t"
        "mov.f32 f3, 0f4B400000;\n\t"
        "mov.b64 l2, {f3, f3};\n\t"
        "add.rm.ftz.f32x2 l7, l1, l2;\n\t"
        "sub.rn.ftz.f32x2 l8, l7, l2;\n\t"
        "sub.rn.ftz.f32x2 l9, l1, l8;\n\t"
        "mov.f32 f7, 0f3D9DF09D;\n\t"
        "mov.b64 l6, {f7, f7};\n\t"
        "mov.f32 f6, 0f3E6906A4;\n\t"
        "mov.b64 l5, {f6, f6};\n\t"
        "mov.f32 f5, 0f3F31F519;\n\t"
        "mov.b64 l4, {f5, f5};\n\t"
        "mov.f32 f4, 0f3F800000;\n\t"
        "mov.b64 l3, {f4, f4};\n\t"
        "fma.rn.ftz.f32x2 l10, l9, l6, l5;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l4;\n\t"
        "fma.rn.ftz.f32x2 l10, l10, l9, l3;\n\t"
        "mov.b64 {r1, r2}, l7;\n\t"
        "mov.b64 {r3, r4}, l10;\n\t"
        "shl.b32 r5, r1, 23;\n\t"
        "add.s32 r7, r5, r3;\n\t"
        "shl.b32 r6, r2, 23;\n\t"
        "add.s32 r8, r6, r4;\n\t"
        "mov.b64 %0, {r7, r8};\n\t"
        "}\n"
        : "=l"(out) : "f"(x), "f"(y));
    return out;
}
"""
    return Builtin(body=body, includes=[], return_val=f"exp2_emu_f32x2_fn({x}, {y})")


@builtin(eval_return_type=ll.uint64, codegen_func=codegen_exp2_emu_f32x2)
def exp2_emu_f32x2(x, y):
    """Approximate two FP32 base-2 exponentials packed in one uint64."""
    raise RuntimeError("should not call exp2_emu_f32x2 in compilation")


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
# Integer Arithmetic (for compute benchmarks.nv.common)
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


# ==============================================================================
# SM100 Generic Utilities (moved from umma.py)
# ==============================================================================

# elect_one: elect a single thread from the warp
# ==============================================================================


@builtin(eval_return_type=uint32, codegen_func=codegen_elect_one_sync)
def elect_one():
    """Elect a single thread in the warp. Returns 1 for elected, 0 for others."""
    raise RuntimeError("should not call elect_one in compilation")


# BF16 packing utility
# ==============================================================================


def codegen_pack_bf16(fp32_a, fp32_b):
    # Use __float22bfloat162_rn for efficient vectorized conversion (FA3 style)
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ uint32_t pack_bf16_fn(uint32_t fp32_a, uint32_t fp32_b) {
    // FA3-style: use __float22bfloat162_rn for efficient vectorized conversion
    float2 f2;
    f2.x = __uint_as_float(fp32_a);
    f2.y = __uint_as_float(fp32_b);
    __nv_bfloat162 bf2 = __float22bfloat162_rn(f2);
    return *reinterpret_cast<uint32_t*>(&bf2);
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"pack_bf16_fn({fp32_a}, {fp32_b})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_bf16)
def pack_bf16(fp32_a, fp32_b):
    """Pack two FP32 values (as uint32 bit patterns) into one uint32 of two BF16."""
    raise RuntimeError("should not call pack_bf16 in compilation")


def codegen_pack_bf16_f32x2(value, keep_pair, keep_hi):
    body = r'''
__device__ __forceinline__ uint32_t pack_bf16_f32x2_fn(uint64_t value, bool keep_pair, bool keep_hi) {
    float lo, hi;
    asm volatile("mov.b64 {%0, %1}, %2;" : "=f"(lo), "=f"(hi) : "l"(value));
    const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, keep_hi ? hi : 0.0f);
    return keep_pair ? *reinterpret_cast<const uint32_t*>(&packed) : 0;
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"pack_bf16_f32x2_fn({value}, {keep_pair}, {keep_hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_bf16_f32x2)
def pack_bf16_f32x2(value, keep_pair, keep_hi):
    """Convert packed FP32x2 to BF16x2 with pair and high-lane masks."""
    raise RuntimeError("should not call pack_bf16_f32x2 in compilation")


# BF16 packing utility (direct float input)
# ==============================================================================


def codegen_pack_bf16_float(fp32_a, fp32_b):
    # Use __floats2bfloat162_rn for efficient vectorized conversion (FA3 style)
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ uint32_t pack_bf16_float_fn(float fp32_a, float fp32_b) {
    // FA3-style: use __floats2bfloat162_rn for efficient vectorized conversion
    auto tmp = __floats2bfloat162_rn(fp32_a, fp32_b);
    return *reinterpret_cast<uint32_t*>(&tmp);
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"pack_bf16_float_fn({fp32_a}, {fp32_b})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_bf16_float)
def pack_bf16_float(fp32_a, fp32_b):
    """Pack two float32 values directly into one uint32 of two BF16.
    More efficient than pack_bf16 as it avoids uint32<->float conversions."""
    raise RuntimeError("should not call pack_bf16_float in compilation")


def codegen_pack_bf16_float4(acc, j, out0, out1):
    """Convert 4 floats to 2 uint32 bf16 pairs using float4 vectorized load."""
    body = """
#include <cuda_bf16.h>
__device__ __forceinline__ void pack_bf16_float4_fn(const float* acc, int j, uint32_t* out0, uint32_t* out1) {
    // FA3-style: use float4 vectorized load + __float22bfloat162_rn
    float4 v = *reinterpret_cast<const float4*>(acc + j * 4);
    __nv_bfloat162 bf2_0 = __float22bfloat162_rn(*reinterpret_cast<float2*>(&v.x));
    __nv_bfloat162 bf2_1 = __float22bfloat162_rn(*reinterpret_cast<float2*>(&v.z));
    *out0 = *reinterpret_cast<uint32_t*>(&bf2_0);
    *out1 = *reinterpret_cast<uint32_t*>(&bf2_1);
}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"pack_bf16_float4_fn({acc}, {j}, &{out0}, &{out1})")


@builtin(eval_return_type=void, codegen_func=codegen_pack_bf16_float4)
def pack_bf16_float4(acc, j, out0, out1):
    """Convert 4 floats at acc[j*4:j*4+4] to 2 uint32 bf16 pairs.
    Uses float4 vectorized load for better performance."""
    raise RuntimeError("should not call pack_bf16_float4 in compilation")


def codegen_pack_2xbf16(lo, hi, out):
    body = r'''
#ifndef LK_PACK_2XBF16_FN_DEFINED
#define LK_PACK_2XBF16_FN_DEFINED
__device__ __forceinline__ void pack_2xbf16_fn(float lo, float hi, uint32_t* out) {
    // Single-instruction RNE pack: d.lo = cvt(lo), d.hi = cvt(hi).
    asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(*out) : "f"(lo), "f"(hi));
}
#endif
'''
    return Builtin(body=body, includes=[], return_val=f"pack_2xbf16_fn({lo}, {hi}, &{out})")


@builtin(eval_return_type=void, codegen_func=codegen_pack_2xbf16)
def pack_2xbf16(lo, hi, out):
    """Convert two float32 registers into one uint32 of two BF16 lanes (RNE).

    Out-param spelling of the single-instruction PTX ``cvt.rn.bf16x2.f32``
    idiom (round-to-nearest-even on both lanes; ``lo`` -> low lane,
    ``hi`` -> high lane), matching the drain style of the ``tmem_ld``
    family: ``pack_2xbf16(r0, r1, p)`` writes the packed word into the
    already-declared uint32 register ``p``, ready to feed a vectorized
    ``st.shared.v4`` (``smem_st_128``) alongside three sibling packs.
    Complements the value-returning ``pack_bf16_float`` (libdevice
    ``__floats2bfloat162_rn``) for epilogues that stage results through
    named out registers. Subtype-restricted companion: if either input
    holds uint32 bit patterns rather than float32 values, use
    ``pack_bf16`` instead.

    Args:
        lo: float32 value for the LOW bf16 lane (bits 15:0 of ``out``)
        hi: float32 value for the HIGH bf16 lane (bits 31:16 of ``out``)
        out: uint32 register written with the packed bf16x2 word
    """
    raise RuntimeError("should not call pack_2xbf16 in compilation")


# Packed 2-byte pair codecs (uint32 <-> two float32 lanes)
# ==============================================================================


def codegen_unpack_bf16x2_to_f32(bits, lo, hi):
    body = """
#ifndef LK_UNPACK_BF16X2_TO_F32_FN_DEFINED
#define LK_UNPACK_BF16X2_TO_F32_FN_DEFINED
__device__ __forceinline__ void unpack_bf16x2_to_f32_fn(uint32_t bits, float* lo, float* hi) {
    const float2 pair = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&bits));
    *lo = pair.x; *hi = pair.y;
}
#endif
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"unpack_bf16x2_to_f32_fn({bits}, &{lo}, &{hi})")


@builtin(eval_return_type=void, codegen_func=codegen_unpack_bf16x2_to_f32)
def unpack_bf16x2_to_f32(bits, lo, hi):
    """Unpack one uint32 of two BF16 lanes into two float32 out-values
    (lo = low-address element).  Pair with ``gmem_ld_v4`` lanes and
    ``pack_bf16_float``."""
    raise RuntimeError("should not call unpack_bf16x2_to_f32 in compilation")


def codegen_unpack_f16x2_to_f32(bits, lo, hi):
    body = """
#ifndef LK_UNPACK_F16X2_TO_F32_FN_DEFINED
#define LK_UNPACK_F16X2_TO_F32_FN_DEFINED
__device__ __forceinline__ void unpack_f16x2_to_f32_fn(uint32_t bits, float* lo, float* hi) {
    const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&bits));
    *lo = pair.x; *hi = pair.y;
}
#endif
"""
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"unpack_f16x2_to_f32_fn({bits}, &{lo}, &{hi})")


@builtin(eval_return_type=void, codegen_func=codegen_unpack_f16x2_to_f32)
def unpack_f16x2_to_f32(bits, lo, hi):
    """Unpack one uint32 of two FP16 lanes into two float32 out-values
    (lo = low-address element).  Pair with ``gmem_ld_v4`` lanes and
    ``pack_f16_float``."""
    raise RuntimeError("should not call unpack_f16x2_to_f32 in compilation")


def codegen_pack_f16_float(fp32_a, fp32_b):
    body = """
#ifndef LK_PACK_F16_FLOAT_FN_DEFINED
#define LK_PACK_F16_FLOAT_FN_DEFINED
__device__ __forceinline__ uint32_t pack_f16_float_fn(float fp32_a, float fp32_b) {
    const __half2 tmp = __floats2half2_rn(fp32_a, fp32_b);
    return *reinterpret_cast<const uint32_t*>(&tmp);
}
#endif
"""
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"pack_f16_float_fn({fp32_a}, {fp32_b})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_f16_float)
def pack_f16_float(fp32_a, fp32_b):
    """Pack two float32 values into one uint32 of two FP16 (RNE), the FP16
    counterpart of ``pack_bf16_float``."""
    raise RuntimeError("should not call pack_f16_float in compilation")


# 16-bit lane pack/unpack (bit-level, dtype-agnostic)
# ==============================================================================


def codegen_unpack_u32_2x16(bits, lo, hi):
    body = r'''
__device__ __forceinline__ void unpack_u32_2x16_fn(uint32_t bits, uint32_t* lo, uint32_t* hi) {
    uint16_t lo16, hi16;
    asm volatile("mov.b32 {%0, %1}, %2;" : "=h"(lo16), "=h"(hi16) : "r"(bits));
    *lo = (uint32_t)lo16; *hi = (uint32_t)hi16;
}
'''
    return Builtin(body=body, includes=[], return_val=f"unpack_u32_2x16_fn({bits}, &{lo}, &{hi})")


@builtin(eval_return_type=void, codegen_func=codegen_unpack_u32_2x16)
def unpack_u32_2x16(bits, lo, hi):
    """Split one uint32 into its two 16-bit lanes via mov.b32 {h0,h1}, r.

    Bit-level and dtype-agnostic: the lanes may be fp16, bf16, or uint16 bit
    patterns; both outputs are written zero-extended as uint32. lo receives
    bits [15:0], hi receives bits [31:16]. Counterpart of pack_u32_2x16."""
    raise RuntimeError("should not call unpack_u32_2x16 in compilation")


def codegen_pack_u32_2x16(lo, hi):
    body = r'''
__device__ __forceinline__ uint32_t pack_u32_2x16_fn(uint32_t lo, uint32_t hi) {
    const uint16_t lo16 = (uint16_t)(lo & 0xFFFFu);
    const uint16_t hi16 = (uint16_t)(hi & 0xFFFFu);
    uint32_t bits;
    asm volatile("mov.b32 %0, {%1, %2};" : "=r"(bits) : "h"(lo16), "h"(hi16));
    return bits;
}
'''
    return Builtin(body=body, includes=[], return_val=f"pack_u32_2x16_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_u32_2x16)
def pack_u32_2x16(lo, hi):
    """Pack the low 16 bits of two uint32 values via mov.b32 r, {h0,h1}.

    Bit-level and dtype-agnostic (no value conversion): lo supplies result
    bits [15:0], hi supplies result bits [31:16]; upper input bits are
    ignored, so callers masking/rounding (e.g. float_to_bf16) stay in
    control. Counterpart of unpack_u32_2x16."""
    raise RuntimeError("should not call pack_u32_2x16 in compilation")


# FP16 value pack/unpack (RNE cvt on u32 lanes; fp16 counterparts of the
# pack_bf16_float / scale_bf16x2 patterns — fp16<->fp32 is NOT a bit shift,
# so it needs cvt instructions, unlike bf16)
# ==============================================================================


def codegen_pack_f16x2(lo, hi):
    body = r'''
__device__ __forceinline__ uint32_t pack_f16x2_fn(float lo, float hi) {
    __half2 h2 = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h2);
}
'''
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"pack_f16x2_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_pack_f16x2)
def pack_f16x2(lo, hi):
    """Pack two float32 values into one uint32 of two fp16 lanes with RNE.

    Emits the same cvt.rn.f16x2.f32 PTX as __floats2half2_rn; fp16 analog of
    pack_bf16_float. Result lane layout matches the __half2 / v4.b32 stream
    layout: lo value in bits [15:0], hi value in bits [31:16], ready for
    gmem_st_v4_u32 or pack-free smem staging."""
    raise RuntimeError("should not call pack_f16x2 in compilation")


def codegen_unpack_f16x2_f32(bits, lo, hi):
    body = r'''
__device__ __forceinline__ void unpack_f16x2_f32_fn(uint32_t bits, float* lo, float* hi) {
    const __half2 h2 = *reinterpret_cast<const __half2*>(&bits);
    *lo = __low2float(h2);
    *hi = __high2float(h2);
}
'''
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"unpack_f16x2_f32_fn({bits}, &{lo}, &{hi})")


@builtin(eval_return_type=void, codegen_func=codegen_unpack_f16x2_f32)
def unpack_f16x2_f32(bits, lo, hi):
    """Convert the two fp16 lanes of one uint32 to float32 (cvt.f32.f16 each).

    Lane layout matches pack_f16x2 (lo = bits [15:0], hi = bits [31:16]).
    Use after gmem_ld_v4_u32 on an fp16 stream; bf16 streams do not need
    this (shift-left-16 / mask is exact) and bit-only lane moves belong to
    unpack_u32_2x16. Rounding: none — fp16->fp32 is exact."""
    raise RuntimeError("should not call unpack_f16x2_f32 in compilation")


# 128-bit shared memory store
# ==============================================================================


def codegen_smem_st_128(addr, v0, v1, v2, v3):
    body = """
__device__ __forceinline__ void smem_st_128_fn(uint32_t addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "r"(addr), "r"(v0), "r"(v1), "r"(v2), "r"(v3) : "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"smem_st_128_fn({addr}, {v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=void, codegen_func=codegen_smem_st_128)
def smem_st_128(addr, v0, v1, v2, v3):
    """128-bit vectorized store to shared memory (st.shared.v4.b32)."""
    raise RuntimeError("should not call smem_st_128 in compilation")


def codegen_smem_ld_128(addr, *regs):
    if len(regs) != 4:
        raise ValueError(f"smem_ld_128 requires 4 registers, got {len(regs)}")
    args = ", ".join(f"&{reg}" for reg in regs)
    body = r'''
__device__ __forceinline__ void smem_ld_128_fn(uint32_t addr, uint32_t* r0, uint32_t* r1, uint32_t* r2, uint32_t* r3) {
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(*r0), "=r"(*r1), "=r"(*r2), "=r"(*r3) : "r"(addr) : "memory");
}
'''
    return Builtin(body=body, includes=[], return_val=f"smem_ld_128_fn({addr}, {args})")


@builtin(eval_return_type=void, codegen_func=codegen_smem_ld_128)
def smem_ld_128(addr, *regs):
    """Load four uint32 values (128 bits) from shared memory."""
    raise RuntimeError("should not call smem_ld_128 in compilation")


def codegen_smem_to_gmem_st_128(mem, index, smem_addr):
    body = r'''
__device__ __forceinline__ void smem_to_gmem_st_128_fn(void* gptr, uint32_t saddr) {
    uint4 value;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "r"(saddr) : "memory");
    *reinterpret_cast<uint4*>(gptr) = value;
}
'''
    return Builtin(body=body, includes=[], return_val=f"smem_to_gmem_st_128_fn((void*)({mem} + {index}), {smem_addr})")


@builtin(eval_return_type=void, codegen_func=codegen_smem_to_gmem_st_128)
def smem_to_gmem_st_128(mem, index, smem_addr):
    """Copy 16 raw bytes from shared memory to global memory (ld.shared.v4.b32 + st.global.v4.b32).

    Fused form of `smem_ld_128` + a vectorized gmem store, one per call site;
    dtype-agnostic (bit patterns are moved unconverted), so `mem` may be any
    tensor element type and `index` is in `mem` elements.

    Legality contract (caller-guaranteed, as in the tilelang vectorize-loop
    alignment rule): `(mem + index)` in element units is 16-byte aligned,
    `smem_addr` is a 16-byte-aligned shared-window u32 address (as produced by
    the smem *_ptr_from_addr helpers), and all 16 bytes are in bounds on both
    sides. The shared->global ordering obligations (e.g. __syncthreads before
    reading smem written by other threads) stay with the caller, exactly as
    for `smem_ld_128`/`gmem_st_bf16x8` used back to back."""
    raise RuntimeError("should not call smem_to_gmem_st_128 in compilation")


def codegen_scale_bf16x2(bits, scale):
    body = r'''
__device__ __forceinline__ uint32_t scale_bf16x2_fn(uint32_t bits, float scale) {
    uint32_t lo = bits & 0xFFFFu;
    uint32_t hi = bits >> 16;
    float f0, f1;
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f0) : "h"((uint16_t)lo));
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(f1) : "h"((uint16_t)hi));
    f0 *= scale;
    f1 *= scale;
    uint32_t result;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result) : "f"(f1), "f"(f0));
    return result;
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"scale_bf16x2_fn({bits}, {scale})")


@builtin(eval_return_type=uint32, codegen_func=codegen_scale_bf16x2)
def scale_bf16x2(bits, scale):
    """Scale one packed BF16x2 value by an FP32 scalar."""
    raise RuntimeError("should not call scale_bf16x2 in compilation")


def codegen_mul_bf16x2(lhs, rhs):
    body = r'''
__device__ __forceinline__ uint32_t mul_bf16x2_fn(uint32_t lhs, uint32_t rhs) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&lhs);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&rhs);
    const __nv_bfloat162 result = __hmul2(a, b);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"mul_bf16x2_fn({lhs}, {rhs})")


@builtin(eval_return_type=uint32, codegen_func=codegen_mul_bf16x2)
def mul_bf16x2(lhs, rhs):
    """Multiply two packed BF16x2 values lane-wise."""
    raise RuntimeError("should not call mul_bf16x2 in compilation")


def codegen_add_bf16x2(lhs, rhs):
    body = r'''
__device__ __forceinline__ uint32_t add_bf16x2_fn(uint32_t lhs, uint32_t rhs) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&lhs);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&rhs);
    const __nv_bfloat162 result = __hadd2(a, b);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"add_bf16x2_fn({lhs}, {rhs})")


@builtin(eval_return_type=uint32, codegen_func=codegen_add_bf16x2)
def add_bf16x2(lhs, rhs):
    """Add two packed BF16x2 values lane-wise (``__hadd2``, RN per lane).

    Bitwise identical to widening both lanes to F32, adding, and rounding
    each sum once to BF16 with round-to-nearest-even."""
    raise RuntimeError("should not call add_bf16x2 in compilation")


def codegen_habs2_bf16x2(bits):
    body = r'''
__device__ __forceinline__ uint32_t habs2_bf16x2_fn(uint32_t bits) {
    const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&bits);
    const __nv_bfloat162 result = __habs2(value);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"habs2_bf16x2_fn({bits})")


@builtin(eval_return_type=uint32, codegen_func=codegen_habs2_bf16x2)
def habs2_bf16x2(bits):
    """Absolute value of both packed BF16 lanes using hardware ``__habs2``.

    The raw uint32 container is reinterpreted as ``__nv_bfloat162``; no lane
    is widened to F32, so there is no extra conversion or rounding boundary."""
    raise RuntimeError("should not call habs2_bf16x2 in compilation")


def codegen_hmax2_bf16x2(lhs, rhs):
    body = r'''
__device__ __forceinline__ uint32_t hmax2_bf16x2_fn(uint32_t lhs, uint32_t rhs) {
    const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&lhs);
    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&rhs);
    const __nv_bfloat162 result = __hmax2(a, b);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"hmax2_bf16x2_fn({lhs}, {rhs})")


@builtin(eval_return_type=uint32, codegen_func=codegen_hmax2_bf16x2)
def hmax2_bf16x2(lhs, rhs):
    """Lane-wise maximum of two packed BF16x2 values using NaN-suppressing ``__hmax2``.

    This deliberately maps to ``__hmax2`` rather than ``__hmax2_nan`` and
    avoids unpacking both lanes to F32 before recombining them."""
    raise RuntimeError("should not call hmax2_bf16x2 in compilation")


# ==============================================================================
# FP8 (e4m3) saturating RNE conversions
# ==============================================================================
#
# ``cvt.rn.satfinite.e4m3x2.f32 d, a, b;`` packs RNE_satfinite(b) into the
# low byte of d and RNE_satfinite(a) into the high byte.  satfinite: inputs
# beyond +-448 map to the largest-magnitude finite e4m3 value, NaN stays NaN
# (0x7f).  One hardware rounding per value -- the bitwise reference for any
# fp8-quant epilogue that rounds exactly once.


def codegen_cvt_rn_satfinite_e4m3x2_f32(lo, hi):
    body = r'''
__device__ __forceinline__ uint32_t cvt_rn_satfinite_e4m3x2_f32_fn(float lo, float hi) {
    unsigned short d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;"
                 : "=h"(d) : "f"(lo), "f"(hi));
    return (uint32_t)d;
}
'''
    return Builtin(body=body, includes=[], return_val=f"cvt_rn_satfinite_e4m3x2_f32_fn(({lo}), ({hi}))")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvt_rn_satfinite_e4m3x2_f32)
def cvt_rn_satfinite_e4m3x2_f32(lo, hi):
    """Saturating RNE e4m3 cast of two F32 values, packed: ``lo`` -> bits
    [0:8), ``hi`` -> bits [8:16) of the returned uint32 (upper 16 bits are 0)."""
    raise RuntimeError("should not call cvt_rn_satfinite_e4m3x2_f32 in compilation")


def codegen_cvt_rn_satfinite_e4m3_f32(v):
    body = r'''
__device__ __forceinline__ uint8_t cvt_rn_satfinite_e4m3_f32_fn(float v) {
    unsigned short d;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;"
                 : "=h"(d) : "f"(v));
    return (uint8_t)(d & 0xff);
}
'''
    return Builtin(body=body, includes=[], return_val=f"cvt_rn_satfinite_e4m3_f32_fn(({v}))")


@builtin(eval_return_type=uint8, codegen_func=codegen_cvt_rn_satfinite_e4m3_f32)
def cvt_rn_satfinite_e4m3_f32(v):
    """Saturating RNE e4m3 cast of one F32 value (single hardware rounding)."""
    raise RuntimeError("should not call cvt_rn_satfinite_e4m3_f32 in compilation")


def codegen_scale_bf16x2_pair(bits, scale0, scale1):
    body = r'''
__device__ __forceinline__ uint32_t scale_bf16x2_pair_fn(uint32_t bits, float scale0, float scale1) {
    const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&bits);
    const float2 pair = __bfloat1622float2(value);
    const __nv_bfloat162 scaled = __floats2bfloat162_rn(pair.x * scale0, pair.y * scale1);
    return *reinterpret_cast<const uint32_t*>(&scaled);
}
'''
    return Builtin(
        body=body,
        includes=["<cuda_bf16.h>"],
        return_val=f"scale_bf16x2_pair_fn({bits}, {scale0}, {scale1})",
    )


@builtin(eval_return_type=uint32, codegen_func=codegen_scale_bf16x2_pair)
def scale_bf16x2_pair(bits, scale0, scale1):
    """Scale the low and high BF16 lanes by separate FP32 scalars."""
    raise RuntimeError("should not call scale_bf16x2_pair in compilation")


def codegen_copy_bf16x16_linear_to_kmajor(src, src_index, dst_base, row, group):
    body = r'''
__device__ __forceinline__ void copy_bf16x16_linear_to_kmajor_fn(
        const __nv_bfloat16* src, int src_index, uint32_t dst_base, uint32_t row, uint32_t group) {
    const uint4 v0 = *reinterpret_cast<const uint4*>(src + src_index);
    const uint4 v1 = *reinterpret_cast<const uint4*>(src + src_index + 8);
    const uint32_t addr0 = dst_base + row * 128 + ((group ^ (row & 7)) * 16);
    const uint32_t addr1 = dst_base + row * 128 + (((group + 1) ^ (row & 7)) * 16);
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr0), "r"(v0.x), "r"(v0.y), "r"(v0.z), "r"(v0.w) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr1), "r"(v1.x), "r"(v1.y), "r"(v1.z), "r"(v1.w) : "memory");
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"copy_bf16x16_linear_to_kmajor_fn({src}, {src_index}, {dst_base}, {row}, {group})")


@builtin(eval_return_type=void, codegen_func=codegen_copy_bf16x16_linear_to_kmajor)
def copy_bf16x16_linear_to_kmajor(src, src_index, dst_base, row, group):
    """Copy 16 linear BF16 values into one row of K-major shared memory."""
    raise RuntimeError("should not call copy_bf16x16_linear_to_kmajor in compilation")


def codegen_copy_bf16x16_linear_to_kmajor_scaled(src, dst, scales, src_index, dst_base, row, col, scale):
    body = r'''
__device__ __forceinline__ uint32_t lk_scale_bf16x2(uint32_t bits, float scale0, float scale1) {
    const __nv_bfloat162 value = *reinterpret_cast<const __nv_bfloat162*>(&bits);
    const float2 pair = __bfloat1622float2(value);
    uint64_t packed_pair, packed_scale, packed_result;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(packed_pair) : "f"(pair.x), "f"(pair.y));
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(packed_scale) : "f"(scale0), "f"(scale1));
    asm volatile("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(packed_result) : "l"(packed_pair), "l"(packed_scale));
    const float2 result = *reinterpret_cast<const float2*>(&packed_result);
    const __nv_bfloat162 scaled = __floats2bfloat162_rn(result.x, result.y);
    return *reinterpret_cast<const uint32_t*>(&scaled);
}

__device__ __forceinline__ void copy_bf16x16_linear_to_kmajor_scaled_fn(
        const __nv_bfloat16* src, const float* scales, int src_index, uint32_t dst_base,
        uint32_t row, uint32_t col, float scale) {
    const uint32_t group = col >> 3;
    const uint32_t addr0 = dst_base + row * 128 + ((group ^ (row & 7)) * 16);
    const uint32_t addr1 = dst_base + row * 128 + (((group + 1) ^ (row & 7)) * 16);
    uint4 value = *reinterpret_cast<const uint4*>(src + src_index);
    value.x = lk_scale_bf16x2(value.x, scale * scales[col], scale * scales[col + 1]);
    value.y = lk_scale_bf16x2(value.y, scale * scales[col + 2], scale * scales[col + 3]);
    value.z = lk_scale_bf16x2(value.z, scale * scales[col + 4], scale * scales[col + 5]);
    value.w = lk_scale_bf16x2(value.w, scale * scales[col + 6], scale * scales[col + 7]);
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr0), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
    value = *reinterpret_cast<const uint4*>(src + src_index + 8);
    value.x = lk_scale_bf16x2(value.x, scale * scales[col + 8], scale * scales[col + 9]);
    value.y = lk_scale_bf16x2(value.y, scale * scales[col + 10], scale * scales[col + 11]);
    value.z = lk_scale_bf16x2(value.z, scale * scales[col + 12], scale * scales[col + 13]);
    value.w = lk_scale_bf16x2(value.w, scale * scales[col + 14], scale * scales[col + 15]);
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" :: "r"(addr1), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w) : "memory");
}
'''
    return Builtin(
        body=body, includes=["<cuda_bf16.h>"], return_val=
        f"copy_bf16x16_linear_to_kmajor_scaled_fn({src}, {scales}, {src_index}, {dst_base}, {row}, {col}, {scale})")


@builtin(eval_return_type=void, codegen_func=codegen_copy_bf16x16_linear_to_kmajor_scaled)
def copy_bf16x16_linear_to_kmajor_scaled(src, dst, scales, src_index, dst_base, row, col, scale):
    """Scale 16 linear BF16 values by columns while copying to K-major SMEM."""
    raise RuntimeError("should not call copy_bf16x16_linear_to_kmajor_scaled in compilation")


# float_as_uint: bitcast float32 -> uint32
# ==============================================================================


def codegen_float_as_uint(val):
    body = ""
    return Builtin(body=body, includes=[], return_val=f"__float_as_uint({val})")


@builtin(eval_return_type=uint32, codegen_func=codegen_float_as_uint)
def float_as_uint(val):
    """Bitcast float32 to uint32 (__float_as_uint)."""
    raise RuntimeError("should not call float_as_uint in compilation")


def codegen_cvt_e4m3x2_rn(hi, lo):
    body = r"""
__device__ __forceinline__ uint32_t cvt_e4m3x2_rn_fn(float hi, float lo) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(packed) : "f"(hi), "f"(lo));
    return (uint32_t)packed;
}
"""
    return Builtin(body=body, includes=[], return_val=f"cvt_e4m3x2_rn_fn({hi}, {lo})")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvt_e4m3x2_rn)
def cvt_e4m3x2_rn(hi, lo):
    """Convert two FP32 values to e4m3 (RNE, satfinite) packed in one u16,
    zero-extended to uint32. Operand order: `hi` lands in the HIGH byte of
    the packed u16, `lo` in the low byte (PTX cvt.e4m3x2.f32 d, a, b puts a
    in the upper half). Combine four lanes as w01 | (w23 << 16)."""
    raise RuntimeError("should not call cvt_e4m3x2_rn in compilation")


# uint_as_float: bitcast uint32 -> float32
# ==============================================================================


def codegen_uint_as_float(val):
    body = ""
    return Builtin(body=body, includes=[], return_val=f"__uint_as_float({val})")


@builtin(eval_return_type=float32, codegen_func=codegen_uint_as_float)
def uint_as_float(val):
    """Bitcast uint32 to float32 (__uint_as_float)."""
    raise RuntimeError("should not call uint_as_float in compilation")


# Packed FP8 (e4m3 / e5m2) conversions
#
# Encoding uses cvt.rn.satfinite.<fp8x2>.f32: round-to-nearest-even with
# saturation to +/-max-finite on overflow (NaN maps to canonical NaN), so the
# caller needs no explicit clamp. PTX operand order is d, a, b with cvt(a) in
# bits 15:8 and cvt(b) in bits 7:0; the intrinsics below take (lo, hi) so the
# returned 16-bit pair has lo in the low byte, hi in the high byte. Results are
# raw bit patterns carried in uint32 (16-bit intrinsic inputs zero-extend);
# legal on sm_89+. Requires no CUDA header.
# ==============================================================================


def codegen_fp8x2_e4m3_from_f32x2(lo, hi):
    body = """
__device__ __forceinline__ uint32_t fp8x2_e4m3_from_f32x2_fn(float lo, float hi) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(packed) : "f"(hi), "f"(lo));
    return static_cast<uint32_t>(packed);
}
"""
    return Builtin(body=body, includes=[], return_val=f"fp8x2_e4m3_from_f32x2_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_fp8x2_e4m3_from_f32x2)
def fp8x2_e4m3_from_f32x2(lo, hi):
    """Convert two FP32 values to two packed e4m3 bytes (lo in bits 7:0, hi in bits 15:8).

    RN rounding, satfinite overflow clamp. Requires sm_89+."""
    raise RuntimeError("should not call fp8x2_e4m3_from_f32x2 in compilation")


def codegen_fp8x4_e4m3_from_f32x4(v0, v1, v2, v3):
    body = r"""
__device__ __forceinline__ uint32_t fp8x4_e4m3_from_f32x4_fn(float v0, float v1, float v2, float v3) {
    uint32_t packed;
    asm volatile("{\n\t.reg .b16 t0, t1;\n\t"
                 "cvt.rn.satfinite.e4m3x2.f32 t0, %1, %2;\n\t"
                 "cvt.rn.satfinite.e4m3x2.f32 t1, %3, %4;\n\t"
                 "mov.b32 %0, {t0, t1};\n\t}"
                 : "=r"(packed) : "f"(v1), "f"(v0), "f"(v3), "f"(v2));
    return packed;
}
"""
    return Builtin(body=body, includes=[], return_val=f"fp8x4_e4m3_from_f32x4_fn({v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=uint32, codegen_func=codegen_fp8x4_e4m3_from_f32x4)
def fp8x4_e4m3_from_f32x4(v0, v1, v2, v3):
    """Convert four FP32 values to one uint32 of four e4m3 bytes (v0 in bits 7:0 .. v3 in bits 31:24).

    RN rounding, satfinite overflow clamp. Requires sm_89+."""
    raise RuntimeError("should not call fp8x4_e4m3_from_f32x4 in compilation")


def codegen_fp8x2_e5m2_from_f32x2(lo, hi):
    body = """
__device__ __forceinline__ uint32_t fp8x2_e5m2_from_f32x2_fn(float lo, float hi) {
    uint16_t packed;
    asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;"
                 : "=h"(packed) : "f"(hi), "f"(lo));
    return static_cast<uint32_t>(packed);
}
"""
    return Builtin(body=body, includes=[], return_val=f"fp8x2_e5m2_from_f32x2_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_fp8x2_e5m2_from_f32x2)
def fp8x2_e5m2_from_f32x2(lo, hi):
    """Convert two FP32 values to two packed e5m2 bytes (lo in bits 7:0, hi in bits 15:8).

    RN rounding, satfinite overflow clamp. Requires sm_89+."""
    raise RuntimeError("should not call fp8x2_e5m2_from_f32x2 in compilation")


def codegen_fp8x4_e5m2_from_f32x4(v0, v1, v2, v3):
    body = r"""
__device__ __forceinline__ uint32_t fp8x4_e5m2_from_f32x4_fn(float v0, float v1, float v2, float v3) {
    uint32_t packed;
    asm volatile("{\n\t.reg .b16 t0, t1;\n\t"
                 "cvt.rn.satfinite.e5m2x2.f32 t0, %1, %2;\n\t"
                 "cvt.rn.satfinite.e5m2x2.f32 t1, %3, %4;\n\t"
                 "mov.b32 %0, {t0, t1};\n\t}"
                 : "=r"(packed) : "f"(v1), "f"(v0), "f"(v3), "f"(v2));
    return packed;
}
"""
    return Builtin(body=body, includes=[], return_val=f"fp8x4_e5m2_from_f32x4_fn({v0}, {v1}, {v2}, {v3})")


@builtin(eval_return_type=uint32, codegen_func=codegen_fp8x4_e5m2_from_f32x4)
def fp8x4_e5m2_from_f32x4(v0, v1, v2, v3):
    """Convert four FP32 values to one uint32 of four e5m2 bytes (v0 in bits 7:0 .. v3 in bits 31:24).

    RN rounding, satfinite overflow clamp. Requires sm_89+."""
    raise RuntimeError("should not call fp8x4_e5m2_from_f32x4 in compilation")


def codegen_f16x2_from_fp8x2_e4m3(bits):
    body = """
__device__ __forceinline__ uint32_t f16x2_from_fp8x2_e4m3_fn(uint32_t bits) {
    const uint16_t low = static_cast<uint16_t>(bits);
    uint32_t halves;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(low));
    return halves;
}
"""
    return Builtin(body=body, includes=[], return_val=f"f16x2_from_fp8x2_e4m3_fn({bits})")


@builtin(eval_return_type=uint32, codegen_func=codegen_f16x2_from_fp8x2_e4m3)
def f16x2_from_fp8x2_e4m3(bits):
    """Decode two e4m3 bytes (bits 7:0 / 15:8 of bits) to packed f16x2 bit pattern.

    Byte 0 lands in the low half. Requires sm_89+."""
    raise RuntimeError("should not call f16x2_from_fp8x2_e4m3 in compilation")


def codegen_f16x2_from_fp8x2_e5m2(bits):
    body = """
__device__ __forceinline__ uint32_t f16x2_from_fp8x2_e5m2_fn(uint32_t bits) {
    const uint16_t low = static_cast<uint16_t>(bits);
    uint32_t halves;
    asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(halves) : "h"(low));
    return halves;
}
"""
    return Builtin(body=body, includes=[], return_val=f"f16x2_from_fp8x2_e5m2_fn({bits})")


@builtin(eval_return_type=uint32, codegen_func=codegen_f16x2_from_fp8x2_e5m2)
def f16x2_from_fp8x2_e5m2(bits):
    """Decode two e5m2 bytes (bits 7:0 / 15:8 of bits) to packed f16x2 bit pattern.

    Byte 0 lands in the low half. Requires sm_89+."""
    raise RuntimeError("should not call f16x2_from_fp8x2_e5m2 in compilation")


# f64 <-> uint64 bitcasts: pure register reinterpret, no conversion.
# __double_as_longlong/__longlong_as_double operate on the identical 64-bit
# pattern; the uint64 view carries the same bits.
# ==============================================================================


def codegen_f64_as_u64(val):
    body = ""
    return Builtin(body=body, includes=[], return_val=f"__double_as_longlong({val})")


@builtin(eval_return_type=uint64, codegen_func=codegen_f64_as_u64)
def f64_as_u64(val):
    """Bitcast float64 to its 64-bit pattern (uint64 view)."""
    raise RuntimeError("should not call f64_as_u64 in compilation")


def codegen_u64_as_f64(val):
    body = ""
    return Builtin(body=body, includes=[], return_val=f"__longlong_as_double(static_cast<long long>({val}))")


@builtin(eval_return_type=float64, codegen_func=codegen_u64_as_f64)
def u64_as_f64(val):
    """Bitcast a uint64 pattern back to float64 (exact inverse of f64_as_u64)."""
    raise RuntimeError("should not call u64_as_f64 in compilation")


# Packed FP16x2 arithmetic on raw uint32 containers (f16 companions of the
# BF16x2 family): lane-wise __hmul2/__hadd2 on __half2, bit-move in/out.
# ==============================================================================


def codegen_mul_f16x2(lhs, rhs):
    body = r'''
__device__ __forceinline__ uint32_t mul_f16x2_fn(uint32_t lhs, uint32_t rhs) {
    const __half2 a = *reinterpret_cast<const __half2*>(&lhs);
    const __half2 b = *reinterpret_cast<const __half2*>(&rhs);
    const __half2 result = __hmul2(a, b);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"mul_f16x2_fn({lhs}, {rhs})")


@builtin(eval_return_type=uint32, codegen_func=codegen_mul_f16x2)
def mul_f16x2(lhs, rhs):
    """Multiply two packed FP16x2 values lane-wise."""
    raise RuntimeError("should not call mul_f16x2 in compilation")


def codegen_add_f16x2(lhs, rhs):
    body = r'''
__device__ __forceinline__ uint32_t add_f16x2_fn(uint32_t lhs, uint32_t rhs) {
    const __half2 a = *reinterpret_cast<const __half2*>(&lhs);
    const __half2 b = *reinterpret_cast<const __half2*>(&rhs);
    const __half2 result = __hadd2(a, b);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"add_f16x2_fn({lhs}, {rhs})")


@builtin(eval_return_type=uint32, codegen_func=codegen_add_f16x2)
def add_f16x2(lhs, rhs):
    """Add two packed FP16x2 values lane-wise."""
    raise RuntimeError("should not call add_f16x2 in compilation")


def codegen_habs2_f16x2(bits):
    body = r'''
__device__ __forceinline__ uint32_t habs2_f16x2_fn(uint32_t bits) {
    const __half2 value = *reinterpret_cast<const __half2*>(&bits);
    const __half2 result = __habs2(value);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"habs2_f16x2_fn({bits})")


@builtin(eval_return_type=uint32, codegen_func=codegen_habs2_f16x2)
def habs2_f16x2(bits):
    """Absolute value of both packed FP16 lanes using hardware ``__habs2``.

    The uint32 bit container is reinterpreted without an F32 round trip."""
    raise RuntimeError("should not call habs2_f16x2 in compilation")


def codegen_hmax2_f16x2(lhs, rhs):
    body = r'''
__device__ __forceinline__ uint32_t hmax2_f16x2_fn(uint32_t lhs, uint32_t rhs) {
    const __half2 a = *reinterpret_cast<const __half2*>(&lhs);
    const __half2 b = *reinterpret_cast<const __half2*>(&rhs);
    const __half2 result = __hmax2(a, b);
    return *reinterpret_cast<const uint32_t*>(&result);
}
'''
    return Builtin(body=body, includes=["<cuda_fp16.h>"], return_val=f"hmax2_f16x2_fn({lhs}, {rhs})")


@builtin(eval_return_type=uint32, codegen_func=codegen_hmax2_f16x2)
def hmax2_f16x2(lhs, rhs):
    """Lane-wise maximum of two packed FP16x2 values using NaN-suppressing ``__hmax2``."""
    raise RuntimeError("should not call hmax2_f16x2 in compilation")


@ll_kernel(backend="cuda", is_entry=False)
def smem_swizzle_128b_addr(base: uint32, row: uint32, bank_group: uint32) -> uint32:
    """Byte address of a 16-byte bank group in a 128-byte-swizzled SMEM row."""
    return base + row * uint32(128) + (bank_group ^ (row & uint32(7))) * uint32(16)


# ``log2f`` is exported by ``language/__init__.py`` and must remain defined
# alongside the disjoint approximate cp.async/math intrinsics in this section.
# It is the accurately-rounded CUDA base-2 logarithm and the twin of
# ``exp2f``; historical provenance for its restoration is recorded in
# docs/adr/0003-comment-provenance.md.
def codegen_log2f(x):
    return Builtin(body="", includes=[], return_val=f"log2f({x})")


@builtin(eval_return_type=ll.float32, codegen_func=codegen_log2f)
def log2f(x):
    """Base-2 logarithm using CUDA ``log2f`` (correctly-rounded-ish twin of
    ``exp2f``; use ``log2_f32`` when an SFU lg2.approx is sufficient)."""
    raise RuntimeError("should not call log2f in compilation")


# ==============================================================================
# Approximate math with explicit flush-to-zero / fast-division flavours
# ==============================================================================


@simple_builtin
def exp2_approx_ftz_f32(x: ll.float32) -> ll.float32:
    """Base-2 exponential (approximate, SFU, flush-to-zero inputs/outputs)."""
    ret: ll.float32 = 0.0
    ll.asm("ex2.approx.ftz.f32 %0, %1;", ["=f", "f"], [ret, x])
    return ret


@simple_builtin
def div_full_f32(a: ll.float32, b: ll.float32) -> ll.float32:
    """Fast range-scaled-reciprocal division using ``div.full.f32``.

    Much cheaper than IEEE div.rn.f32 (no FFMA Newton-Raphson tail) but not
    correctly rounded; intended for normalization kernels (softmax, rmsnorm,
    layernorm) whose tolerance covers the approximation.
    """
    ret: ll.float32 = 0.0
    ll.asm("div.full.f32 %0, %1, %2;", ["=f", "f", "f"], [ret, a, b])
    return ret


# ==============================================================================
# cp.async (LDGSTS) asynchronous GMEM->SMEM copy
# ==============================================================================
# Pure inline-PTX wrappers, same emission style as the gmem_ld/st_* helpers:
# the helper is a {const void*}-typed function so frontends of any element
# dtype share one intrinsic (pointer arithmetic happens in the typed call-site
# expression, then converts to const void*). sm_80+; this GPU is sm_90a.


def codegen_cp_async16(dst_smem, src, index, src_bytes):
    body = """
__device__ __forceinline__ void cp_async16_fn(
        uint32_t dst, const void* src, uint32_t src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src), "r"(src_bytes));
}
"""
    return Builtin(body=body, includes=[], return_val=f"cp_async16_fn({dst_smem}, {src} + {index}, {src_bytes})")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async16)
def cp_async16(dst_smem, src, index, src_bytes):
    """Copy 16 bytes GMEM->SMEM asynchronously (cp.async.cg, 16-byte tier).

    dst_smem is the SMEM byte address as uint32 (e.g. from a cvta-shared
    helper), src/index index the source tensor in its own element units
    (caller guarantees 16-byte alignment of src + index), and src_bytes is 16
    or 0 (0 writes zero-fill, the LDGSTS out-of-bounds masking idiom).
    Requires sm_80+; does not commit a group by itself."""
    raise RuntimeError("should not call cp_async16 in compilation")


def codegen_cp_async_commit_group():
    body = """
__device__ __forceinline__ void cp_async_commit_group_fn() {
    asm volatile("cp.async.commit_group;");
}
"""
    return Builtin(body=body, includes=[], return_val="cp_async_commit_group_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_commit_group)
def cp_async_commit_group():
    """Commit all prior cp.async ops of this thread into a new async group."""
    raise RuntimeError("should not call cp_async_commit_group in compilation")


def codegen_cp_async_wait_group(n):
    body = """
template<int N>
__device__ __forceinline__ void cp_async_wait_group_fn() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}
"""
    return Builtin(body=body, includes=[], return_val=f"cp_async_wait_group_fn<{n}>()")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_wait_group)
def cp_async_wait_group(n):
    """Block until at most n async groups of this thread are in flight.

    n must be a compile-time-constant expression (it becomes a C++ template
    argument, matching the PTX immediate-operand rule for cp.async.wait_group).
    """
    raise RuntimeError("should not call cp_async_wait_group in compilation")


def codegen_cp_async_mbar_arrive(bar_smem):
    body = """
__device__ __forceinline__ void cp_async_mbar_arrive_fn(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" :: "r"(bar));
}
"""
    return Builtin(body=body, includes=[], return_val=f"cp_async_mbar_arrive_fn({bar_smem})")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_mbar_arrive)
def cp_async_mbar_arrive(bar_smem):
    """Make this thread's prior cp.async ops arrive on an mbarrier.

    bar_smem is the mbarrier's SMEM byte address as uint32. The barrier trips
    once the thread's outstanding cp.async bytes have landed in SMEM.
    """
    raise RuntimeError("should not call cp_async_mbar_arrive in compilation")


def codegen_cp_async_mbar_arrive_noinc(bar_smem):
    body = """
__device__ __forceinline__ void cp_async_mbar_arrive_noinc_fn(uint32_t bar) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(bar));
}
"""
    return Builtin(body=body, includes=[], return_val=f"cp_async_mbar_arrive_noinc_fn({bar_smem})")


@builtin(eval_return_type=void, codegen_func=codegen_cp_async_mbar_arrive_noinc)
def cp_async_mbar_arrive_noinc(bar_smem):
    """.noinc form of cp_async_mbar_arrive: does not increment the mbarrier's
    expected-arrival count, so the caller must have registered every arrival
    up front (e.g. mbarrier.init with the full producer thread count). Use the
    plain cp_async_mbar_arrive for self-balancing barriers instead; mixing the
    two forms on one barrier is invalid."""
    raise RuntimeError("should not call cp_async_mbar_arrive_noinc in compilation")
