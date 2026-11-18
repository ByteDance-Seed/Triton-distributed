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
Matrix-move/tensor-core intrinsics (stmatrix/ldmatrix, mma.sync, STSM).

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4

# mma.sync operand-dtype legality: whitelist checked BEFORE any asm text is
# composed (same contract as the WGMMA family in intrin/wgmma.py).
_MMA_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "fp16": "f16",
    "float16": "f16",
    "half": "f16",
    "f64": "f64",
    "fp64": "f64",
    "float64": "f64",
    "double": "f64",
}

# tf32 is a legal mma.sync operand type only on the 16x8x8 (and 16x8x4) shapes
# PTX defines; the 16-bit atoms (m16n8k16, m16n16k16) have no tf32 opcode, so
# it is kept out of the shared whitelist and opted in per atom.
_MMA_TF32_DTYPE_ALIASES = {
    "tf32": "tf32",
    "tfloat32": "tf32",
}


def _mma_ptx_dtype(dtype, allow_tf32=False):
    """Normalize a frontend dtype token to the PTX mma.sync operand type.

    Accepts "bf16"/"bfloat16", "f16"/"fp16"/"float16"/"half", and
    "f64"/"fp64"/"float64"/"double" (also ll dtype objects exposing .name).
    Raises ValueError for anything else: mma.sync opcodes are only defined for
    these element types (16-bit f16/bf16 with an f32 accumulator, and
    binary64 f64 with a binary64 accumulator on the dedicated m16n8k4 atom),
    so legality is enforced by the whitelist, never by strings.

    With allow_tf32=True, "tf32"/"tfloat32" is additionally accepted; only
    atoms whose PTX shape has a tf32 form (m16n8k8, f32 accumulator) opt in.
    The tf32 path takes f32 register operands that the tensor core truncates
    to tf32 internally (no author-side pre-rounding).

    Note: f64 is legal as a PTX operand token here; the mixed 16-bit codegen
    paths only use the f16/bf16 names and keep their fixed f32 accumulator.
    The f64 tensor-core atom is exposed through mma_m16n8k4_f64, the only
    FP64 mma.sync shape PTX defines for f64.
    """
    token = dtype
    if not isinstance(token, str):
        token = getattr(token, "name", None) or str(token)
    token = str(token).strip().strip("'\"").lower()
    ptx = _MMA_DTYPE_ALIASES.get(token)
    if ptx is None and allow_tf32:
        ptx = _MMA_TF32_DTYPE_ALIASES.get(token)
    if ptx is None:
        aliases = sorted(_MMA_DTYPE_ALIASES)
        if allow_tf32:
            aliases += sorted(_MMA_TF32_DTYPE_ALIASES)
        raise ValueError(f"mma.sync operand dtype must be one of {aliases}, "
                         f"got {dtype!r}")
    return ptx


# ==============================================================================
# STSM (Store Matrix) with type conversion
# ==============================================================================


def codegen_stsm_x2_from_floats(f0, f1, f2, f3, smem_ptr):
    body = ("\n__device__ __forceinline__ void stsm_x2_from_floats_fn("
            "float f0, float f1, float f2, float f3, void* smem_ptr) {\n"
            "    __nv_bfloat162 v0 = __floats2bfloat162_rn(f0, f1);\n"
            "    __nv_bfloat162 v1 = __floats2bfloat162_rn(f2, f3);\n"
            "    uint32_t src0 = *reinterpret_cast<uint32_t*>(&v0);\n"
            "    uint32_t src1 = *reinterpret_cast<uint32_t*>(&v1);\n"
            '    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16 '
            '[%0], {%1, %2};\\n"\n'
            '                 :: "r"((uint32_t)__cvta_generic_to_shared(smem_ptr)),'
            ' "r"(src0), "r"(src1));\n'
            "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"stsm_x2_from_floats_fn({f0}, {f1}, {f2}, {f3}, {smem_ptr})")


@builtin(eval_return_type=void, codegen_func=codegen_stsm_x2_from_floats)
def stsm_x2_from_floats(f0, f1, f2, f3, smem_ptr):
    """Convert 4 floats to 2 bfloat162 and store via STSM x2."""
    raise RuntimeError("should not be called in compilation")


def codegen_stmatrix_x1(smem_ptr, r0):
    """Generate stmatrix.x1 inline asm for storing 1 uint32 register to shared memory."""
    body = ("\n__device__ __forceinline__ void stmatrix_x1_fn("
            "void* smem_ptr, uint32_t r0) {\n"
            "    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));\n"
            '    asm volatile(\n'
            '        "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\\n"\n'
            '        :: "r"(addr), "r"(r0)\n'
            '    );\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"stmatrix_x1_fn({smem_ptr}, {r0})")


@builtin(eval_return_type=void, codegen_func=codegen_stmatrix_x1)
def stmatrix_x1(smem_ptr, r0):
    """Store 1 uint32 register to shared memory via stmatrix.x1.

    stmatrix.x1 stores an 8x8 b16 tile using one warp (32 threads).
    Each thread provides 1 uint32 register (2 bf16 values); the
    per-warp register fragments land in the shared-memory matrix
    layout required by ldmatrix/wgmma consumers.

    Args:
        smem_ptr: Pointer to shared memory (must be 16-byte aligned)
        r0: uint32 register to store
    """
    raise RuntimeError("should not be called in compilation")


def codegen_stmatrix_x2(smem_ptr, r0, r1):
    """Generate stmatrix.x2 inline asm for storing 2 uint32 registers to shared memory."""
    body = ("\n__device__ __forceinline__ void stmatrix_x2_fn("
            "void* smem_ptr, uint32_t r0, uint32_t r1) {\n"
            "    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));\n"
            '    asm volatile(\n'
            '        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\\n"\n'
            '        :: "r"(addr), "r"(r0), "r"(r1)\n'
            '    );\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"stmatrix_x2_fn({smem_ptr}, {r0}, {r1})")


@builtin(eval_return_type=void, codegen_func=codegen_stmatrix_x2)
def stmatrix_x2(smem_ptr, r0, r1):
    """Store 2 uint32 registers to shared memory via stmatrix.x2.

    stmatrix.x2 stores an 8x16 b16 tile using one warp (32 threads).
    Each thread provides 2 uint32 registers (4 bf16 values).

    Args:
        smem_ptr: Pointer to shared memory (must be 16-byte aligned)
        r0, r1: Two uint32 registers to store
    """
    raise RuntimeError("should not be called in compilation")


def codegen_stmatrix_x4(smem_ptr, r0, r1, r2, r3):
    """Generate stmatrix.x4 inline asm for storing 4 uint32 registers to shared memory."""
    body = ("\n__device__ __forceinline__ void stmatrix_x4_fn("
            "void* smem_ptr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {\n"
            "    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));\n"
            '    asm volatile(\n'
            '        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\\n"\n'
            '        :: "r"(addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3)\n'
            '    );\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"stmatrix_x4_fn({smem_ptr}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_stmatrix_x4)
def stmatrix_x4(smem_ptr, r0, r1, r2, r3):
    """Store 4 uint32 registers to shared memory via stmatrix.x4.

    stmatrix.x4 stores a 16x16 tile using 32 threads (one warp).
    Each thread provides 4 uint32 registers (8 bf16 values).

    Args:
        smem_ptr: Pointer to shared memory (must be 16-byte aligned)
        r0, r1, r2, r3: Four uint32 registers to store
    """
    raise RuntimeError("should not be called in compilation")


def codegen_stmatrix_x4_trans(smem_addr, r0, r1, r2, r3):
    body = ("\n__device__ __forceinline__ void stmatrix_x4_trans_fn("
            "uint32_t smem_addr, uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {\n"
            '    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};" '
            ':: "r"(smem_addr), "r"(r0), "r"(r1), "r"(r2), "r"(r3) : "memory");\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"stmatrix_x4_trans_fn({smem_addr}, {r0}, {r1}, {r2}, {r3})")


@builtin(eval_return_type=void, codegen_func=codegen_stmatrix_x4_trans)
def stmatrix_x4_trans(smem_addr, r0, r1, r2, r3):
    """Store one transposed 16x16 BF16 fragment to shared memory."""
    raise RuntimeError("should not call stmatrix_x4_trans in compilation")


def codegen_ldmatrix_x4(mem, index, regs):
    body = r'''
__device__ __forceinline__ void ldmatrix_x4_regs_fn(const __nv_bfloat16* ptr, uint32_t (&regs)[4]) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(addr) : "memory");
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"ldmatrix_x4_regs_fn({mem} + {index}, {regs})")


@builtin(eval_return_type=void, codegen_func=codegen_ldmatrix_x4)
def ldmatrix_x4(mem, index, regs):
    """Load one 16x16 BF16 fragment from shared memory."""
    raise RuntimeError("should not call ldmatrix_x4 in compilation")


def codegen_ldmatrix_x4_trans(mem, index, regs):
    body = r'''
__device__ __forceinline__ void ldmatrix_x4_trans_regs_fn(const __nv_bfloat16* ptr, uint32_t (&regs)[4]) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.trans.shared.b16 {%0, %1, %2, %3}, [%4];" : "=r"(regs[0]), "=r"(regs[1]), "=r"(regs[2]), "=r"(regs[3]) : "r"(addr) : "memory");
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"ldmatrix_x4_trans_regs_fn({mem} + {index}, {regs})")


@builtin(eval_return_type=void, codegen_func=codegen_ldmatrix_x4_trans)
def ldmatrix_x4_trans(mem, index, regs):
    """Load one transposed 16x16 BF16 fragment from shared memory."""
    raise RuntimeError("should not call ldmatrix_x4_trans in compilation")


def codegen_ldmatrix_x1(mem, index, regs):
    body = r'''
__device__ __forceinline__ void ldmatrix_x1_regs_fn(const __nv_bfloat16* ptr, uint32_t (&regs)[1]) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];" : "=r"(regs[0]) : "r"(addr) : "memory");
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"ldmatrix_x1_regs_fn({mem} + {index}, {regs})")


@builtin(eval_return_type=void, codegen_func=codegen_ldmatrix_x1)
def ldmatrix_x1(mem, index, regs):
    """Load one 8x8 BF16 fragment from shared memory."""
    raise RuntimeError("should not call ldmatrix_x1 in compilation")


def codegen_ldmatrix_x1_trans(mem, index, regs):
    body = r'''
__device__ __forceinline__ void ldmatrix_x1_trans_regs_fn(const __nv_bfloat16* ptr, uint32_t (&regs)[1]) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.trans.shared.b16 {%0}, [%1];" : "=r"(regs[0]) : "r"(addr) : "memory");
}
'''
    return Builtin(
        body=body,
        includes=["<cuda_bf16.h>"],
        return_val=f"ldmatrix_x1_trans_regs_fn({mem} + {index}, {regs})",
    )


@builtin(eval_return_type=void, codegen_func=codegen_ldmatrix_x1_trans)
def ldmatrix_x1_trans(mem, index, regs):
    """Load and transpose one 8x8 BF16 fragment from shared memory."""
    raise RuntimeError("should not call ldmatrix_x1_trans in compilation")


def codegen_stmatrix_x4_regs(mem, index, regs):
    body = r'''
__device__ __forceinline__ void stmatrix_x4_regs_fn(__nv_bfloat16* ptr, const uint32_t (&regs)[4]) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};" :: "r"(addr), "r"(regs[0]), "r"(regs[1]), "r"(regs[2]), "r"(regs[3]) : "memory");
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"stmatrix_x4_regs_fn({mem} + {index}, {regs})")


@builtin(eval_return_type=void, codegen_func=codegen_stmatrix_x4_regs)
def stmatrix_x4_regs(mem, index, regs):
    """Store one 16x16 BF16 fragment to shared memory."""
    raise RuntimeError("should not call stmatrix_x4_regs in compilation")


def codegen_stmatrix_x1_reg(mem, index, reg):
    body = r'''
__device__ __forceinline__ void stmatrix_x1_reg_fn(__nv_bfloat16* ptr, uint32_t reg) {
    const uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};" :: "r"(addr), "r"(reg) : "memory");
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"stmatrix_x1_reg_fn({mem} + {index}, {reg})")


@builtin(eval_return_type=void, codegen_func=codegen_stmatrix_x1_reg)
def stmatrix_x1_reg(mem, index, reg):
    """Store one register per thread as an 8x8 BF16 fragment."""
    raise RuntimeError("should not call stmatrix_x1_reg in compilation")


def codegen_mma_m16n8k8(acc, a, b, accumulate, dtype="bf16"):
    # dtype is whitelisted BEFORE any asm text is composed; the bf16 default
    # emits exactly the pre-existing body (same helper name, same asm string).
    # m16n8k8 is one of the shapes PTX defines a tf32 opcode for, so this atom
    # opts into the tf32 whitelist.
    ptx_t = _mma_ptx_dtype(dtype, allow_tf32=True)
    if ptx_t == "tf32":
        # fp32 register operands (A: 4 regs/lane, B: 2 regs/lane) truncated to
        # tf32 inside the tensor core; nothing pre-rounds here.  Distinct
        # helper name so dedup-by-name keeps every dtype body used in one
        # kernel.
        body = r'''
__device__ __forceinline__ void mma_m16n8k8_tf32_fn(
        float (&c)[4], const float (&a)[4], const float (&b)[2], bool accumulate) {
    if (!accumulate) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "f"(a[0]), "f"(a[1]), "f"(a[2]), "f"(a[3]), "f"(b[0]), "f"(b[1]));
}
'''
        return Builtin(body=body, includes=[], return_val=f"mma_m16n8k8_tf32_fn({acc}, {a}, {b}, {accumulate})")
    if ptx_t == "bf16":
        body = r'''
__device__ __forceinline__ void mma_m16n8k8_fn(
        float (&c)[4], const uint32_t (&a)[2], uint32_t b, bool accumulate) {
    if (!accumulate) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}
'''
        return Builtin(body=body, includes=[], return_val=f"mma_m16n8k8_fn({acc}, {a}, {b}, {accumulate})")
    # f16: distinct helper name so dedup-by-name never drops one dtype's body
    # when both dtypes are used in a single kernel.
    body = r'''
__device__ __forceinline__ void mma_m16n8k8_f16_fn(
        float (&c)[4], const uint32_t (&a)[2], uint32_t b, bool accumulate) {
    if (!accumulate) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}
'''
    return Builtin(body=body, includes=[], return_val=f"mma_m16n8k8_f16_fn({acc}, {a}, {b}, {accumulate})")


@builtin(eval_return_type=void, codegen_func=codegen_mma_m16n8k8)
def mma_m16n8k8(acc, a, b, accumulate, dtype="bf16"):
    """One warp-level 16x8x8 MMA into four FP32 accumulators.

    dtype selects the PTX operand element type:

    - "bf16" (default) or "f16" (aliases fp16/half/float16 also accepted):
      A is 2 packed uint32 registers, B is 1 packed uint32 register.
    - "tf32" (alias tfloat32): A is 4 f32 registers and B is 2 f32
      registers, truncated to tf32 inside the tensor core (no pre-rounding).

    The accumulator is 4 FP32 registers in every case. Any other dtype value
    raises ValueError before asm is composed.
    """
    raise RuntimeError("should not call mma_m16n8k8 in compilation")


def codegen_mma_m16n16k16(acc, a, b, accumulate, dtype="bf16"):
    ptx_t = _mma_ptx_dtype(dtype)
    if ptx_t == "bf16":
        body = r'''
__device__ __forceinline__ void mma_m16n16k16_fn(float (&c)[8], const uint32_t (&a)[4], const uint32_t (&b)[4], bool accumulate) {
    if (!accumulate) {
#pragma unroll
        for (int i = 0; i < 8; ++i) c[i] = 0.0f;
    }
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" : "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]));
}
'''
        return Builtin(body=body, includes=[], return_val=f"mma_m16n16k16_fn({acc}, {a}, {b}, {accumulate})")
    body = r'''
__device__ __forceinline__ void mma_m16n16k16_f16_fn(float (&c)[8], const uint32_t (&a)[4], const uint32_t (&b)[4], bool accumulate) {
    if (!accumulate) {
#pragma unroll
        for (int i = 0; i < 8; ++i) c[i] = 0.0f;
    }
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};" : "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[2]), "r"(b[3]));
}
'''
    return Builtin(body=body, includes=[], return_val=f"mma_m16n16k16_f16_fn({acc}, {a}, {b}, {accumulate})")


@builtin(eval_return_type=void, codegen_func=codegen_mma_m16n16k16)
def mma_m16n16k16(acc, a, b, accumulate, dtype="bf16"):
    """One warp-level 16x16x16 MMA (2x m16n8k16) into eight FP32 accumulators.

    dtype selects the PTX operand element type ("bf16" default, or "f16";
    aliases fp16/half/float16 also accepted). Any other value raises
    ValueError before asm is composed.
    """
    raise RuntimeError("should not call mma_m16n16k16 in compilation")


def codegen_mma_m16n8k16(acc, a, b, accumulate, dtype="bf16"):
    """Single m16n8k16 MMA atom (4+2 register operands), dtype parameterized.

    Fragment-aware path (ir.mma.m16n8k16.bf16.per_thread_fragment): when
    acc/a/b name typed mma.sync register fragments, emit a helper taking
    per-scalar references (float& c0..c3, const uint32_t& a0..a3, ...) with
    one asm operand per element — the same operand shape hand CUDA uses —
    so the fragments stay in registers across MMAs and any elementwise
    prologue/epilogue. A PARTIAL fragment operand set (e.g. fragment acc
    with an array A) is a compile error: mixing an address-taken array with
    register scalars for one atom cannot express the PTX operand list.
    Legacy float[4]/uint32[4]/uint32[2] array operands keep the original
    array path byte-identical.
    """
    ptx_t = _mma_ptx_dtype(dtype)
    frag_acc = _try_resolve_mma_fragment_arg(acc, 4, float32, "acc")
    frag_a = _try_resolve_mma_fragment_arg(a, 4, uint32, "a")
    frag_b = _try_resolve_mma_fragment_arg(b, 2, uint32, "b")
    n_frag = sum(x is not None for x in (frag_acc, frag_a, frag_b))
    if n_frag == 3:
        builtin_name = f"mma_m16n8k16(dtype={ptx_t})"
        acc_args = _resolve_mma_fragment_arg(frag_acc, 4, float32, builtin_name, "acc")
        a_args = _resolve_mma_fragment_arg(frag_a, 4, uint32, builtin_name, "a")
        b_args = _resolve_mma_fragment_arg(frag_b, 2, uint32, builtin_name, "b")
        c_params = ", ".join(f"float& c{i}" for i in range(4))
        a_params = ", ".join(f"const uint32_t a{i}" for i in range(4))
        b_params = ", ".join(f"const uint32_t b{i}" for i in range(2))
        helper = f"mma_m16n8k16_{ptx_t}_frag_fn"
        body = f'''
__device__ __forceinline__ void {helper}({c_params}, {a_params}, {b_params}, bool accumulate) {{
    if (!accumulate) {{
        c0 = 0.0f; c1 = 0.0f; c2 = 0.0f; c3 = 0.0f;
    }}
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{ptx_t}.{ptx_t}.f32 {{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};" : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}}
'''
        return Builtin(body=body, includes=[], return_val=(f"{helper}({acc_args}, {a_args}, "
                                                           f"{b_args}, {accumulate})"))
    if n_frag:
        raise ValueError("mma_m16n8k16: partial fragment operand set (acc fragment: "
                         f"{frag_acc is not None}, a fragment: {frag_a is not None}, "
                         f"b fragment: {frag_b is not None}); either ALL THREE of acc/a/b "
                         "must name typed mma.sync register fragments declared with "
                         "ll.mma_acc_fragment_f32 / ll.mma_operand_fragment_u32, or none "
                         "may (the legacy per-thread array path cannot be mixed with "
                         "register fragments for one atom)")
    body = f'''
__device__ __forceinline__ void mma_m16n8k16_{ptx_t}_fn(float (&c)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], bool accumulate) {{
    if (!accumulate) {{
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }}
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.{ptx_t}.{ptx_t}.f32 {{%0, %1, %2, %3}}, {{%4, %5, %6, %7}}, {{%8, %9}}, {{%0, %1, %2, %3}};" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}}
'''
    return Builtin(body=body, includes=[], return_val=f"mma_m16n8k16_{ptx_t}_fn({acc}, {a}, {b}, {accumulate})")


@builtin(eval_return_type=void, codegen_func=codegen_mma_m16n8k16)
def mma_m16n8k16(acc, a, b, accumulate, dtype="bf16"):
    """One warp-level 16x8x16 MMA atom into four FP32 accumulators.

    Args:
        acc: 4-element float accumulator array, OR the name of a typed
            fragment declared with ll.mma_acc_fragment_f32(name, 4) — see
            the fragment note below.
        a: 4 uint32 registers (8 x 16-bit A values), OR a typed fragment
            declared with ll.mma_operand_fragment_u32(name, 4).
        b: 2 uint32 registers (4 x 16-bit B values), OR a typed fragment
            declared with ll.mma_operand_fragment_u32(name, 2).
        accumulate: 1 to accumulate into acc, 0 to overwrite
        dtype: Operand element type, "bf16" (default) or "f16"

    Fragment path (ir.mma.m16n8k16.bf16.per_thread_fragment): when acc, a
    AND b all name typed mma.sync register fragments of the matching
    shapes, the atom binds each per-thread scalar as an individual asm
    register operand, so fragments stay in registers across MMA calls and
    interleaved elementwise subscripts (``acc[i] = f(acc[j])`` at
    compile-time-unrolled indices). Use mma_m16n8k16_a_coordinate /
    mma_m16n8k16_b_coordinate to pack A/B from a shared-memory tile per the
    PTX lane mapping and mma_m16n8k16_accumulator_coordinate to consume
    accumulators portably. Mixing fragment and array operands in one call
    is a compile error.
    """
    raise RuntimeError("should not call mma_m16n8k16 in compilation")


def codegen_mma_m16n8k4_f64(acc, a, b, accumulate, dtype="f64"):
    """Single m16n8k4 FP64 tensor-core MMA atom (4+2+1 f64 register operands).

    PTX defines exactly one FP64 mma.sync shape, m16n8k4, with row.col layout
    and binary64 operands/accumulator. Per warp lane on sm_90:

      * acc/c: four f64 registers (IEEE-754 binary64 accumulator lanes)
          c0 = C[cg][2t], c1 = C[cg][2t+1], c2 = C[cg+8][2t], c3 = C[cg+8][2t+1]
      * a: two f64 registers (m16k4 row-major A fragment)
          a0 = A[cg][t], a1 = A[cg+8][t]
      * b: one f64 register (k4n8 column-major B fragment)
          b0 = B[t][cg]   (stored column-major, element (k=t, n=cg))
      where cg = lane >> 2 and t = lane & 3.

    All arithmetic is IEEE-754 binary64 with binary64 accumulation. Empty
    dtypes (None) are treated as the "f64" default; any non-f64 token raises
    ValueError before asm text is composed.
    """
    ptx_t = _mma_ptx_dtype(dtype)
    if ptx_t != "f64":
        raise ValueError(f"mma_m16n8k4_f64 supports only the f64 operand dtype, got {dtype!r}")
    body = r'''
__device__ __forceinline__ void mma_m16n8k4_f64_fn(
        double (&c)[4], const double (&a)[2], const double (&b)[1], bool accumulate) {
    if (!accumulate) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0;
    }
    asm volatile(
        "mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+d"(c[0]), "+d"(c[1]), "+d"(c[2]), "+d"(c[3])
        : "d"(a[0]), "d"(a[1]), "d"(b[0]));
}
'''
    return Builtin(body=body, includes=[], return_val=f"mma_m16n8k4_f64_fn({acc}, {a}, {b}, {accumulate})")


@builtin(eval_return_type=void, codegen_func=codegen_mma_m16n8k4_f64)
def mma_m16n8k4_f64(acc, a, b, accumulate, dtype="f64"):
    """One warp-level 16x8x4 FP64 tensor-core MMA into four binary64 accumulators.

    Emits PTX mma.sync.aligned.m16n8k4.row.col.f64.f64.f64.f64: the only FP64
    mma.sync encoding PTX defines for f64 (sm_89/sm_90). Operands are fixed-size
    typed local float64 arrays following the same fragment-array calling
    convention as the bf16/f16 mma family:

    Args:
        acc: 4-element float64 accumulator fragment array (IEEE-754 binary64)
        a: 2-element float64 m16k4 row-major A fragment
        b: 1-element float64 k4n8 column-major B fragment
        accumulate: 1 to accumulate into acc, 0 to overwrite
        dtype: operand element type; must be "f64" (aliases
            fp64/float64/double accepted). Any other value raises ValueError
            before asm is composed.
    """
    raise RuntimeError("should not call mma_m16n8k4_f64 in compilation")


# ==============================================================================
# Warp-level mma.sync per-thread register fragments
# (semantic key ir.mma.m16n8k16.bf16.per_thread_fragment)
#
# Motivation: ``ll.mma_m16n8k16(acc, a, b, accumulate)`` requires the caller
# to hand-pack the PTX fragment mapping across the 32 lanes of a warp (acc:
# four f32/lane, a: four u32, b: two u32). Plain per-thread local arrays in
# the frontend cannot construct or index those fragments portably: a dynamic
# lane-dependent subscript on ``ll.alloc_local`` storage takes the array's
# address and demotes it to local memory, and the caller still has to know
# which (row_pair, col_parity) slice of the tile each lane owns.
#
# This section implements the typed-fragment model (same shape as the SM90
# WGMMA accumulator fragment family):
#   * ``ll.mma_acc_fragment_f32(name, num_elems=4)`` declares
#     ``float <name>_0 .. <name>_{N-1}`` as N INDEPENDENT per-thread register
#     scalars (zero-initialized), registered so ``<name>[i]`` with a
#     compile-time-unrolled index lowers to ``<name>_<i>``.
#   * ``ll.mma_operand_fragment_u32(name, num_elems)`` does the same for the
#     packed 16-bit A/B operand registers (uint32 scalars).
#   * ``ll.mma_m16n8k16`` is FRAGMENT-AWARE: when its acc/a/b operands name
#     registered fragments of the m16n8k16 shapes (4 f32 + 4 u32 + 2 u32),
#     codegen emits a helper taking per-scalar references, one asm operand
#     per element — so the same fragment object flows through several MMA
#     calls and interleaved elementwise reads/writes without any address
#     being taken. Legacy ``float acc[4]`` / ``uint32_t a[4]`` / ``uint32_t
#     b[2]`` operands keep the original array path.
#   * ``mma_m16n8k16_accumulator_coordinate`` /
#     ``mma_m16n8k16_a_coordinate`` / ``mma_m16n8k16_b_coordinate`` expose
#     the PTX lane-coordinate mapping so a frontend can build fragments from
#     shared-memory tiles and consume accumulators portably, without
#     hand-encoding per-lane register fragments.
#
# Legality: fragment element indices must be compile-time constants at
# codegen (a literal, a provably-static int local, or an ll.unroll(range(N))
# counter binding inside the unrolled copy). Anything else is a compile
# error, never a silent local-memory demotion.
# ==============================================================================

# Accepted mma.sync fragment sizes, keyed by role: the m16n8k16 atom owns
# 4 per-thread f32 accumulators, 4 per-thread packed A registers and 2
# per-thread packed B registers. The sets are deliberately closed so a typo
# is a diagnosing compile error rather than a malformed asm operand list.
_MMA_ACC_FRAGMENT_ELEMS = (4, )
_MMA_OPERAND_FRAGMENT_ELEMS = (2, 4)


def mma_m16n8k16_accumulator_coordinate(lane, register):
    """Map one FP32 accumulator fragment element of m16n8k16 to (row, col).

    PTX fragment layout for mma.sync.aligned.m16n8k16.row.col .f32 D/C:
    with groupID g = lane >> 2 and threadID_in_group t = lane & 3,
      c0 = C[g][2t],  c1 = C[g][2t+1],
      c2 = C[g+8][2t], c3 = C[g+8][2t+1].
    The same fragment layout applies to f16 and bf16 A/B operand dtypes.
    """
    if (not isinstance(lane, int) or isinstance(lane, bool) or not 0 <= lane < 32):
        raise ValueError(f"lane must be in [0, 32), got {lane!r}")
    if (not isinstance(register, int) or isinstance(register, bool) or not 0 <= register < 4):
        raise ValueError(f"m16n8k16 accumulator register must be in [0, 4), got "
                         f"{register!r}")
    group = lane >> 2
    tid_in_group = lane & 3
    row = group + 8 * (register >> 1)
    col = 2 * tid_in_group + (register & 1)
    return row, col


def mma_m16n8k16_a_coordinate(lane, register, element):
    """Map one 16-bit A fragment element of m16n8k16 to (row, col) of A.

    PTX fragment layout for the 16x16 row-major A operand (four u32
    registers per lane, two 16-bit elements each), with groupID
    g = lane >> 2 and threadID_in_group t = lane & 3:
      reg 0: rows g,    cols 2t, 2t+1
      reg 1: rows g+8,  cols 2t, 2t+1
      reg 2: rows g,    cols 2t+8, 2t+9
      reg 3: rows g+8,  cols 2t+8, 2t+9
    ``element`` is the 16-bit lane within the packed register (0 = low).
    """
    if (not isinstance(lane, int) or isinstance(lane, bool) or not 0 <= lane < 32):
        raise ValueError(f"lane must be in [0, 32), got {lane!r}")
    if (not isinstance(register, int) or isinstance(register, bool) or not 0 <= register < 4):
        raise ValueError(f"m16n8k16 A register must be in [0, 4), got {register!r}")
    if (not isinstance(element, int) or isinstance(element, bool) or not 0 <= element < 2):
        raise ValueError(f"A register element must be in [0, 2), got {element!r}")
    group = lane >> 2
    tid_in_group = lane & 3
    row = group + 8 * (register & 1)
    col = 2 * tid_in_group + element + 8 * (register >> 1)
    return row, col


def mma_m16n8k16_b_coordinate(lane, register, element):
    """Map one 16-bit B fragment element of m16n8k16 to (row, col) of B.

    PTX fragment layout for the 16x8 column-major B operand (two u32
    registers per lane, two 16-bit elements each) with groupID
    g = lane >> 2 and threadID_in_group t = lane & 3; coordinates are
    (k, n) in the KxN B tile:
      reg 0: k = 2t, 2t+1, n = g
      reg 1: k = 2t+8, 2t+9, n = g
    ``element`` is the 16-bit lane within the packed register (0 = low).
    """
    if (not isinstance(lane, int) or isinstance(lane, bool) or not 0 <= lane < 32):
        raise ValueError(f"lane must be in [0, 32), got {lane!r}")
    if (not isinstance(register, int) or isinstance(register, bool) or not 0 <= register < 2):
        raise ValueError(f"m16n8k16 B register must be in [0, 2), got {register!r}")
    if (not isinstance(element, int) or isinstance(element, bool) or not 0 <= element < 2):
        raise ValueError(f"B register element must be in [0, 2), got {element!r}")
    group = lane >> 2
    tid_in_group = lane & 3
    k = 2 * tid_in_group + element + 8 * register
    n = group
    return k, n


def _mma_sync_fragment_registry(emitter):
    """Codegen-side warp-mma fragment registry {name: {"num_elems", "dtype"}}.

    Lives in the emitter's compile ctx under __MMA_SYNC_FRAGMENTS_CTX_KEY__;
    created on first declaration. Returns None when no emitter is active
    (defensive: a codegen function always runs inside emission_scope).
    """
    if emitter is None:
        return None
    from little_kernel.core.internal import __MMA_SYNC_FRAGMENTS_CTX_KEY__
    ctx = getattr(emitter, "ctx", None)
    if ctx is None:
        return None
    return ctx.setdefault(__MMA_SYNC_FRAGMENTS_CTX_KEY__, {})


def _mma_bind_fragment_name(ctx, name, dtype, decl_builtin):
    """Bind a fragment NAME in the typing scope as a tensor of its dtype.

    The binding lets the type inferencer accept element subscripts
    (``name[i]`` loads/stores yield the element dtype). Codegen NEVER
    treats the name as an array: the fragment map (written by the
    declaration's codegen, which always runs before any use) intercepts
    every subscript and MMA operand reference and lowers them to the
    per-element register scalars.
    """
    if isinstance(name, str):
        from little_kernel.core.type_system import Tensor
        existing = ctx.get(name)
        if existing is not None and existing != Tensor[dtype]:
            # Rebinding an existing typed name (e.g. another tensor shape or
            # dtype) as a fragment is almost certainly a name collision bug;
            # diagnose it rather than silently shadowing.
            raise TypeError(f"{decl_builtin}: name {name!r} already bound to "
                            f"{existing}; choose a fresh fragment identifier")
        ctx[name] = Tensor[dtype]


def _mma_acc_fragment_eval_arg_type(ctx, name, num_elems):
    from little_kernel.core.type_system import float32 as _f32
    _mma_bind_fragment_name(ctx, name, _f32, "mma_acc_fragment_f32")


def _mma_operand_fragment_eval_arg_type(ctx, name, num_elems):
    _mma_bind_fragment_name(ctx, name, uint32, "mma_operand_fragment_u32")


def _codegen_mma_fragment_decl(name, num_elems, dtype_ll, c_type, init_zero, allowed_elems, decl_builtin):
    """Shared declaration codegen for the warp-mma fragment family.

    Emits ``<ctype> <name>_0 = 0; ...`` — N INDEPENDENT per-thread register
    scalars, never an array — and registers the fragment so subsequent
    ``<name>[i]`` subscripts (constant index) and the fragment-aware
    mma.sync builtins resolve against the individual scalars.
    """
    from little_kernel.codegen.policy import current_emitter
    try:
        count = int(num_elems)
    except (TypeError, ValueError):
        raise ValueError(f"{decl_builtin}: num_elems must be a compile-time integer, "
                         f"got {num_elems!r}")
    if count not in allowed_elems:
        raise ValueError(f"{decl_builtin}: num_elems must be one of {allowed_elems} "
                         f"(PTX mma.sync fragment sizes), got {count}")
    # Codegen hands string literals through as their quoted source text.
    if isinstance(name, str):
        name = name.strip().strip("'\"")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"{decl_builtin}: fragment name must be a C identifier string, "
                         f"got {name!r}")

    registry = _mma_sync_fragment_registry(current_emitter())
    if registry is not None:
        prev = registry.get(name)
        if prev is not None and (prev["num_elems"] != count or prev["dtype"] != dtype_ll):
            raise ValueError(f"{decl_builtin}: fragment {name!r} redeclared with a "
                             f"different shape ({prev['num_elems']} x {prev['dtype']} vs "
                             f"{count} x {dtype_ll})")
        registry[name] = {"num_elems": count, "dtype": dtype_ll}

    suffix = "f" if init_zero == "0.0" else "u" if dtype_ll == uint32 else ""
    decls = "\n".join(f"{c_type} {name}_{i} = {init_zero}{suffix};" for i in range(count))
    # return_val is inserted as a statement; the declaration text IS the
    # statement (semicolon-terminated lines), so the builtin contributes no
    # trailing expression.
    return Builtin(body="", includes=[], return_val=decls)


def codegen_mma_acc_fragment_f32(name, num_elems=4):
    """Declare a typed warp-mma float32 accumulator register fragment."""
    from little_kernel.core.type_system import float32 as _f32
    return _codegen_mma_fragment_decl(name, num_elems, _f32, "float", "0.0", _MMA_ACC_FRAGMENT_ELEMS,
                                      "mma_acc_fragment_f32")


@builtin(
    eval_return_type=void,
    eval_arg_type=_mma_acc_fragment_eval_arg_type,
    codegen_func=codegen_mma_acc_fragment_f32,
)
def mma_acc_fragment_f32(name, num_elems=4):
    """Declare a typed float32 warp-level mma.sync accumulator fragment.

    Args:
        name: C identifier (string literal) for the fragment; element access
            in kernel source is ``<name>[i]`` with a compile-time-unrolled
            index (literal, provably-static int, or ll.unroll counter),
            lowered to the per-element register scalar ``<name>_<i>``.
        num_elems: accumulator elements per thread; 4 for the m16n8k16 atom.

    The fragment owns ``num_elems`` independent float scalars per thread;
    pass the SAME name as the acc operand of the fragment-aware
    ``mma_m16n8k16`` builtin to accumulate in registers across calls, and
    read/write elements between MMAs for elementwise epilogues without any
    local-memory traffic. Dynamic (non-constant) element indices are a
    compile error.
    """
    raise RuntimeError("should not be called in compilation")


def codegen_mma_operand_fragment_u32(name, num_elems):
    """Declare a typed warp-mma packed-operand (u32) register fragment."""
    return _codegen_mma_fragment_decl(name, num_elems, uint32, "uint32_t", "0", _MMA_OPERAND_FRAGMENT_ELEMS,
                                      "mma_operand_fragment_u32")


@builtin(
    eval_return_type=void,
    eval_arg_type=_mma_operand_fragment_eval_arg_type,
    codegen_func=codegen_mma_operand_fragment_u32,
)
def mma_operand_fragment_u32(name, num_elems):
    """Declare a typed uint32 warp-level mma.sync operand register fragment.

    Holds the packed 16-bit A or B operand registers of one warp-level
    mma.sync atom: two 16-bit elements per register laid out per the PTX
    fragment mapping (see mma_m16n8k16_a_coordinate /
    mma_m16n8k16_b_coordinate, which a frontend uses to pack elements from
    a shared-memory tile at compile-time-unrolled indices).

    Args:
        name: C identifier (string literal) for the fragment; element access
            in kernel source is ``<name>[i]`` with a compile-time-unrolled
            index, lowered to the per-element register scalar ``<name>_<i>``.
        num_elems: registers per thread: 4 (m16n8k16 A) or 2 (m16n8k16 B).

    Dynamic (non-constant) element indices are a compile error.
    """
    raise RuntimeError("should not be called in compilation")


def _resolve_mma_fragment_arg(frag, n_elems, want_dtype, builtin_name, role):
    """Resolve an mma.sync builtin's fragment operand to fragment scalars.

    Returns the comma-separated per-element scalar argument list
    ``name_0, name_1, ...`` for the per-scalar-helper call. Raises
    ValueError when the name is not a registered fragment or its size/dtype
    does not match the operand role.
    """
    from little_kernel.codegen.policy import current_emitter
    token = frag.strip().strip("'\"") if isinstance(frag, str) else str(frag)
    registry = _mma_sync_fragment_registry(current_emitter())
    rec = registry.get(token) if registry else None
    if rec is None:
        raise ValueError(f"{builtin_name}: {role} operand {frag!r} is not a typed "
                         f"mma.sync register fragment; declare it first with "
                         f"ll.mma_acc_fragment_f32 / ll.mma_operand_fragment_u32 before "
                         f"the MMA call")
    if rec["num_elems"] != n_elems or rec["dtype"] != want_dtype:
        raise ValueError(f"{builtin_name}: fragment {token!r} has shape "
                         f"{rec['num_elems']} x {rec['dtype']} but the {role} operand of "
                         f"this atom needs exactly {n_elems} x {want_dtype}; declare the "
                         f"fragment with the matching size and dtype")
    return ", ".join(f"{token}_{i}" for i in range(n_elems))


def _try_resolve_mma_fragment_arg(frag, n_elems, want_dtype, role):
    """Probe whether an mma.sync operand names a register fragment.

    Returns the unquoted fragment identifier when the operand names a
    registered fragment (the caller then resolves/validates per-role via
    _resolve_mma_fragment_arg); returns None when the operand is plainly
    not a fragment name so the legacy array path stays available.
    """
    from little_kernel.codegen.policy import current_emitter
    token = frag.strip().strip("'\"") if isinstance(frag, str) else str(frag)
    if not token.isidentifier():
        return None
    emitter = current_emitter()
    if emitter is None:
        return None
    ctx = getattr(emitter, "ctx", None) or {}
    from little_kernel.core.internal import __MMA_SYNC_FRAGMENTS_CTX_KEY__
    registry = ctx.get(__MMA_SYNC_FRAGMENTS_CTX_KEY__)
    if not registry or token not in registry:
        return None
    return token


def codegen_acc_to_bf16x4_scaled(acc, out, scale):
    body = r'''
__device__ __forceinline__ void acc_to_bf16x4_scaled_fn(const float (&acc)[8], uint32_t (&out)[4], float scale) {
    const __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[0] * scale, acc[1] * scale);
    const __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[2] * scale, acc[3] * scale);
    const __nv_bfloat162 v2 = __floats2bfloat162_rn(acc[4] * scale, acc[5] * scale);
    const __nv_bfloat162 v3 = __floats2bfloat162_rn(acc[6] * scale, acc[7] * scale);
    out[0] = *reinterpret_cast<const uint32_t*>(&v0);
    out[1] = *reinterpret_cast<const uint32_t*>(&v1);
    out[2] = *reinterpret_cast<const uint32_t*>(&v2);
    out[3] = *reinterpret_cast<const uint32_t*>(&v3);
}
'''
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"acc_to_bf16x4_scaled_fn({acc}, {out}, {scale})")


@builtin(eval_return_type=void, codegen_func=codegen_acc_to_bf16x4_scaled)
def acc_to_bf16x4_scaled(acc, out, scale):
    """Scale eight FP32 accumulators and pack them into four BF16x2 registers."""
    raise RuntimeError("should not call acc_to_bf16x4_scaled in compilation")
