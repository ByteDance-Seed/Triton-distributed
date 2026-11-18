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
WGMMA (Warp Group Matrix Multiply-Accumulate) intrinsics for SM90+.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import uint32, uint64, void

# ==============================================================================
# WGMMA legality helpers (whitelist checked BEFORE any asm text is composed)
# ==============================================================================

_WGMMA_DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "f16": "f16",
    "fp16": "f16",
    "float16": "f16",
    "half": "f16",
}

# Integer WGMMA operand types (m64nNk32 with s32 accumulator). Kept separate
# from the float whitelist: integer opcodes have a different shape (k32 vs
# k16), accumulator element type (s32 vs f32) and operand model (no scale_a/
# scale_b immediates, no tnsp transpose immediates), so the dispatchers route
# on the normalized token instead of widening one flat alias table.
_WGMMA_INT_DTYPE_ALIASES = {
    "s8": "s8",
    "int8": "s8",
    "int8_t": "s8",
    "char": "s8",
    "u8": "u8",
    "uint8": "u8",
    "uint8_t": "u8",
}

# FP8 WGMMA operand types (m64nNk32 with f32 accumulator). Kept separate from
# both the 16-bit float whitelist and the integer whitelist: the FP8 opcodes
# have a k32 depth, an f32 accumulator, allow the two operand element types
# to differ (e4m3/e5m2 in any mix), and — like the integer family — have no
# scale_a/scale_b operand-scale immediates and no tnsp transpose immediates
# (the only immediate is scale-d). The dispatchers route FP8 tokens to a
# dedicated branch instead of widening one flat alias table.
_WGMMA_FP8_DTYPE_ALIASES = {
    "e4m3": "e4m3",
    "fp8_e4m3": "e4m3",
    "float8_e4m3": "e4m3",
    "float8_e4m3fn": "e4m3",
    "e4m3fn": "e4m3",
    "e5m2": "e5m2",
    "fp8_e5m2": "e5m2",
    "float8_e5m2": "e5m2",
}


def _wgmma_ptx_dtype(dtype):
    """Normalize a frontend dtype token to the PTX operand type string.

    Accepts "bf16"/"bfloat16" and "f16"/"fp16"/"float16"/"half" (also ll dtype
    objects exposing .name). Raises ValueError for anything else: these wgmma
    opcodes are only defined for 16-bit f16/bf16 operands with an f32
    accumulator, so legality is enforced by the whitelist, never by strings.
    """
    token = dtype
    if not isinstance(token, str):
        token = getattr(token, "name", None) or str(token)
    token = str(token).strip().strip("'\"").lower()
    ptx = _WGMMA_DTYPE_ALIASES.get(token)
    if ptx is None:
        raise ValueError(f"wgmma operand dtype must be one of {sorted(_WGMMA_DTYPE_ALIASES)}, "
                         f"got {dtype!r}")
    return ptx


def _wgmma_int_ptx_dtype(dtype):
    """Normalize a frontend dtype token to an integer WGMMA operand type.

    Accepts "s8"/"int8"/"int8_t" and "u8"/"uint8"/"uint8_t" (also ll dtype
    objects exposing .name), returning the PTX token "s8" or "u8". Raises
    ValueError for anything else: the integer wgmma opcodes are only defined
    for 8-bit integer operands with an s32 accumulator, so legality is
    enforced by the whitelist, never by strings.
    """
    token = dtype
    if not isinstance(token, str):
        token = getattr(token, "name", None) or str(token)
    token = str(token).strip().strip("'\"").lower()
    ptx = _WGMMA_INT_DTYPE_ALIASES.get(token)
    if ptx is None:
        raise ValueError("wgmma integer operand dtype must be one of "
                         f"{sorted(_WGMMA_INT_DTYPE_ALIASES)}, got {dtype!r}")
    return ptx


def _wgmma_fp8_ptx_dtype(dtype):
    """Normalize a frontend dtype token to an FP8 WGMMA operand type string.

    Accepts "e4m3"/"fp8_e4m3"/"float8_e4m3"/"float8_e4m3fn" and
    "e5m2"/"fp8_e5m2"/"float8_e5m2" (also ll dtype objects exposing .name),
    returning the PTX token "e4m3" or "e5m2". Raises ValueError for anything
    else: the FP8 wgmma opcodes are only defined for these two operand types
    with an f32 accumulator, so legality is enforced by the whitelist, never
    by strings.
    """
    token = dtype
    if not isinstance(token, str):
        token = getattr(token, "name", None) or str(token)
    token = str(token).strip().strip("'\"").lower()
    ptx = _WGMMA_FP8_DTYPE_ALIASES.get(token)
    if ptx is None:
        raise ValueError("wgmma FP8 operand dtype must be one of "
                         f"{sorted(_WGMMA_FP8_DTYPE_ALIASES)}, got {dtype!r}")
    return ptx


def _wgmma_is_fp8_dtype(dtype):
    """True iff a frontend dtype token names an FP8 wgmma operand type."""
    token = dtype
    if not isinstance(token, str):
        token = getattr(token, "name", None) or str(token)
    token = str(token).strip().strip("'\"").lower()
    return token in _WGMMA_FP8_DTYPE_ALIASES


def _wgmma_check_dtype_b_family(dtype, dtype_b, builtin_name):
    """Reject dtype_b overrides that cannot select a PTX-legal operand pair.

    dtype_b is only meaningful for the FP8 family, where the two operand
    element types may differ (.e4m3.e5m2 / .e5m2.e4m3). Every other use is
    a ValueError, never a silently ignored argument: a non-FP8 dtype_b, or
    an FP8 dtype_b paired with a non-FP8 dtype (f16/bf16/int), names a wgmma
    form PTX does not define.
    """
    if dtype_b is None:
        return
    if _wgmma_is_fp8_dtype(dtype_b) and _wgmma_is_fp8_dtype(dtype):
        return
    raise ValueError(f"{builtin_name}: dtype_b={dtype_b!r} is only accepted together "
                     "with an FP8 dtype (e4m3/e5m2); the f16/bf16 and integer wgmma "
                     "families require both operand element types to be dtype and have "
                     "no mixed-operand form, and a one-sided FP8 mix is not PTX-legal")


def _wgmma_is_compile_time_int(value):
    """True iff value can be interpreted as a compile-time integer (i.e. it is
    not a traced runtime value). Used to select between an immediate baked
    into the asm template and a register operand feeding a predicate."""
    try:
        int(value)
    except (TypeError, ValueError):
        return False
    return True


def _wgmma_imm_flag(value, name):
    """Normalize a compile-time 0/1 immediate flag (scale_d/trans_a/trans_b)."""
    try:
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"wgmma immediate flag {name} must be a compile-time 0/1 integer, "
                         f"got {value!r}")
    if iv not in (0, 1):
        raise ValueError(f"wgmma immediate flag {name} must be 0 or 1, got {value!r}")
    return iv


# ==============================================================================
# WGMMA Synchronization
# ==============================================================================


def codegen_wgmma_fence():
    body = """
__device__ __forceinline__ void wgmma_fence_fn() {
    asm volatile("wgmma.fence.sync.aligned;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="wgmma_fence_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence)
def wgmma_fence():
    """WGMMA fence synchronization."""
    raise RuntimeError("should not be called in compilation")


def codegen_wgmma_commit():
    body = """
__device__ __forceinline__ void wgmma_commit_fn() {
    asm volatile("wgmma.commit_group.sync.aligned;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="wgmma_commit_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_commit)
def wgmma_commit():
    """WGMMA commit group."""
    raise RuntimeError("should not be called in compilation")


def codegen_wgmma_wait():
    body = """
__device__ __forceinline__ void wgmma_wait_fn() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val="wgmma_wait_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_wait)
def wgmma_wait():
    """WGMMA wait group."""
    raise RuntimeError("should not be called in compilation")


def codegen_wgmma_wait_group(n):
    # Check if n is a compile-time constant (integer literal)
    try:
        n_val = int(n)
        # Generate direct inline asm without function call for constant n
        body = f"""
__device__ __forceinline__ void wgmma_wait_group_{n_val}_fn() {{
    asm volatile("wgmma.wait_group.sync.aligned {n_val};\\n" ::: "memory");
}}
"""
        return Builtin(body=body, includes=[], return_val=f"wgmma_wait_group_{n_val}_fn()")
    except (ValueError, TypeError):
        # Fall back to switch statement for runtime values
        body = """
__device__ __forceinline__ void wgmma_wait_group_fn(int n) {
    switch (n) {
        case 0: asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory"); break;
        case 1: asm volatile("wgmma.wait_group.sync.aligned 1;\\n" ::: "memory"); break;
        case 2: asm volatile("wgmma.wait_group.sync.aligned 2;\\n" ::: "memory"); break;
        case 3: asm volatile("wgmma.wait_group.sync.aligned 3;\\n" ::: "memory"); break;
        case 4: asm volatile("wgmma.wait_group.sync.aligned 4;\\n" ::: "memory"); break;
        case 5: asm volatile("wgmma.wait_group.sync.aligned 5;\\n" ::: "memory"); break;
        case 6: asm volatile("wgmma.wait_group.sync.aligned 6;\\n" ::: "memory"); break;
        case 7: asm volatile("wgmma.wait_group.sync.aligned 7;\\n" ::: "memory"); break;
        default: asm volatile("wgmma.wait_group.sync.aligned 0;\\n" ::: "memory"); break;
    }
}
"""
        return Builtin(body=body, includes=[], return_val=f"wgmma_wait_group_fn({n})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_wait_group)
def wgmma_wait_group(n):
    """WGMMA wait group with specified count (wait until at most n groups pending)."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# WGMMA m64n256k16 instruction
# ==============================================================================


def wgmma_m64n256k16_accumulator_coordinate(warpgroup_tid, register):
    """Map one FP32 accumulator fragment to its ``(row, column)`` tile element.

    The mapping is for a four-warp warpgroup (``warpgroup_tid`` in [0, 128))
    and the 128 per-thread FP32 accumulators of ``m64n256k16``.  Adjacent
    register pairs hold adjacent columns; this helper therefore accepts a
    scalar register index and returns the coordinate of that individual
    accumulator.  The same fragment layout applies to f16 and bf16 operands.
    """
    if (not isinstance(warpgroup_tid, int) or isinstance(warpgroup_tid, bool) or not 0 <= warpgroup_tid < 128):
        raise ValueError("warpgroup_tid must be in [0, 128), got "
                         f"{warpgroup_tid!r}")
    if (not isinstance(register, int) or isinstance(register, bool) or not 0 <= register < 128):
        raise ValueError("m64n256k16 register must be in [0, 128), got "
                         f"{register!r}")

    warp = warpgroup_tid // 32
    lane = warpgroup_tid % 32
    pair = register // 2
    row = 16 * warp + 8 * ((register % 4) // 2) + lane // 4
    column = (pair // 2) * 8 + 2 * (lane % 4) + register % 2
    return row, column


def codegen_wgmma_m64n256k16(acc, desc_a, desc_b):
    """Generate the WGMMA m64n256k16 inline asm (128 output registers)."""
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(128)])
    reg_lines = []
    for row_start in range(0, 128, 16):
        regs = ",".join([f"%{i}" for i in range(row_start, row_start + 16)])
        reg_lines.append(regs)
    reg_block = ",\\n".join(reg_lines)

    body = ("\n__device__ __forceinline__ void wgmma_m64n256k16_fn("
            "float* c, uint64_t desc_a, uint64_t desc_b) {\n"
            "    asm volatile(\n"
            '        "{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            '        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\\n"\n'
            f'        "{{{reg_block}}},"\n'
            '        "%128,%129,p,1,1,0,0;\\n}\\n"\n'
            f"        : {out_constraints}\n"
            '        : "l"(desc_a), "l"(desc_b));\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_m64n256k16_fn({acc}, {desc_a}, {desc_b})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n256k16)
def wgmma_m64n256k16(acc, desc_a, desc_b):
    """WGMMA m64n256k16 bf16 instruction with 128-float accumulator."""
    raise RuntimeError("should not be called in compilation")


def _require_compile_time_tnsp(value, name):
    """WGMMA trans immediates must be PTX literals: require Python ints 0/1."""
    try:
        v = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"wgmma_m64n256k16_tnsp: {name} must be a compile-time Python int "
                         f"0 or 1 (the tnsp field is a PTX immediate), got {value!r}")
    if v not in (0, 1):
        raise ValueError(f"wgmma_m64n256k16_tnsp: {name} must be 0 or 1, got {v}")
    return v


def codegen_wgmma_m64n256k16_tnsp(acc, desc_a, desc_b, tnspA, tnspB):
    """WGMMA m64n256k16 with explicit tnspA/tnspB descriptor-layout immediates."""
    ta = _require_compile_time_tnsp(tnspA, "tnspA")
    tb = _require_compile_time_tnsp(tnspB, "tnspB")
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(128)])
    reg_lines = []
    for row_start in range(0, 128, 16):
        regs = ",".join([f"%{i}" for i in range(row_start, row_start + 16)])
        reg_lines.append(regs)
    reg_block = ",\\n".join(reg_lines)

    body = ("\n__device__ __forceinline__ void wgmma_m64n256k16_tnsp_fn("
            "float* c, uint64_t desc_a, uint64_t desc_b) {\n"
            "    asm volatile(\n"
            '        "{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            '        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "%128,%129,p,1,1,{ta},{tb};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            '        : "l"(desc_a), "l"(desc_b));\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_m64n256k16_tnsp_fn({acc}, {desc_a}, {desc_b})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n256k16_tnsp)
def wgmma_m64n256k16_tnsp(acc, desc_a, desc_b, tnspA, tnspB):
    """WGMMA m64n256k16 bf16 with explicit transpose immediates.

    Identical instruction to wgmma_m64n256k16 except the tnspA/tnspB PTX
    immediates (byte 6/7 of the descriptor-layout operand) are selected by
    the frontend instead of being hardcoded 0,0:
      - tnsp=0: operand smem descriptor is in K-major ("SS", CUTLASS default)
      - tnsp=1: operand smem descriptor is in MN-major (feeds NN row-major
        operands, e.g. a [K,N] B matrix, with the matching MN-major LBO/SBO
        descriptor encoding)

    Legality contract (caller-owned, not compiler-checked): tnspA/tnspB must
    be Python ints 0 or 1 (PTX immediates, rejected with ValueError
    otherwise), and the smem descriptors must be encoded for exactly the
    selected majorness / swizzle mode. Accumulator layout and count are
    unchanged, so results equal wgmma_m64n256k16 bit-for-bit when both
    immediates are 0."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# High-level WGMMA Accumulator Ops
# Accumulator is declared by user via ll.empty(..., scope="local"); these ops
# take (acc, num_elems, dtype) or (acc, ...) so generated code is parameterized.
# ==============================================================================

# Map C++ scalar type (from _lltype_to_cpp) to zero literal for init/zero
# loops. Integer accumulators (e.g. an s32 accumulator for integer MMA) must
# zero-init with the integer literal 0, never "0.0f"; the fallback stays
# "0.0f" because every historical WGMMA accumulator was f32 and any
# unrecognised token here is a float-family typename.
_ZERO_LITERAL = {
    "float": "0.0f",
    "float32": "0.0f",
    "double": "0.0",
    "float64": "0.0",
    "int8_t": "0",
    "uint8_t": "0",
    "int16_t": "0",
    "uint16_t": "0",
    "int32_t": "0",
    "uint32_t": "0",
    "int64_t": "0",
    "uint64_t": "0",
    "int": "0",
    "unsigned": "0",
}

# C++ scalar types (from _lltype_to_cpp) that map onto PTX integer general
# registers; everything else (float family) uses the f-register constraint.
_ACC_INT_CPP_TYPES = {
    "int8_t",
    "uint8_t",
    "int16_t",
    "uint16_t",
    "int32_t",
    "uint32_t",
    "int64_t",
    "uint64_t",
    "int",
    "unsigned",
}


def _acc_asm_constraint(dtype_cpp):
    """PTX inline-asm register constraint for an accumulator element type.

    Integer accumulators (int8..int64, signed or unsigned) must be fenced
    through the integer register constraint "+r"; float-family accumulators
    keep the historical float constraint "+f". Keyed off the declared C++
    element type, so any integer accumulator dtype is supported, not one
    task-specific spelling.
    """
    return "+r" if str(dtype_cpp) in _ACC_INT_CPP_TYPES else "+f"


def _acc_cpp_type_token(dtype_cpp):
    """Legal, injective C++ identifier token for an accumulator element type.

    Helper function names embed the accumulator dtype so an int32 and a
    float variant of the same epilogue never collide in the emitter's
    per-name builtin cache (call sites with different dtypes produce
    different helper bodies, and every distinct body needs its own name).
    """
    token = str(dtype_cpp)
    for sep in ("::", " ", "<", ">", "*", "&"):
        token = token.replace(sep, "_")
    return token.strip("_") or "unk"


def codegen_wgmma_init_accum(acc, num_elems, dtype_cpp):
    """Emit zero loop only; declaration is from user's ll.empty -> alloc_local_memory (or merged alloc_init)."""
    zero = _ZERO_LITERAL.get(str(dtype_cpp), "0.0f")
    return Builtin(body="", includes=[], return_val=f"for(int _i=0; _i<{num_elems}; _i++) {acc}[_i]={zero}")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_init_accum)
def wgmma_init_accum(acc, num_elems, dtype):
    """Zero the WGMMA accumulator (user declares acc via ll.empty)."""
    raise RuntimeError("should not be called")


def codegen_wgmma_alloc_init_accum(acc, num_elems, dtype_cpp):
    """Emit declaration + zero loop in one (same as legacy single-statement output)."""
    zero = _ZERO_LITERAL.get(str(dtype_cpp), "0.0f")
    return Builtin(body="", includes=[],
                   return_val=f"{dtype_cpp} {acc}[{num_elems}];\nfor(int _i=0; _i<{num_elems}; _i++) {acc}[_i]={zero}")


def _wgmma_alloc_init_accum_eval_arg_type(ctx, acc, num_elems, dtype):
    from little_kernel.core.type_system import Tensor
    if isinstance(acc, str):
        ctx[acc] = Tensor[dtype]


@builtin(
    eval_return_type=void,
    eval_arg_type=_wgmma_alloc_init_accum_eval_arg_type,
    codegen_func=codegen_wgmma_alloc_init_accum,
)
def wgmma_alloc_init_accum(acc, num_elems, dtype):
    """Declare and zero accumulator in one (used when pass merges empty+init_accum)."""
    raise RuntimeError("should not be called")


def codegen_wgmma_zero_accum(acc, num_elems, dtype_cpp):
    zero = _ZERO_LITERAL.get(str(dtype_cpp), "0.0f")
    return Builtin(body="", includes=[],
                   return_val=f"do {{}} while(0);\nfor(int _i=0; _i<{num_elems}; _i++) {acc}[_i]={zero}")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_zero_accum)
def wgmma_zero_accum(acc, num_elems, dtype):
    """Zero the WGMMA accumulator."""
    raise RuntimeError("should not be called")


def codegen_store_accum_swizzle(acc, sD, warp_idx, lane_idx, m_offset, num_elems, num_rows, row_size, rows_per_warp,
                                swizzle_base, dtype):
    # Derive all constants from parameters instead of hardcoding
    # Parameters:
    # - num_elems: total accumulator elements (e.g., 128 for m64n256k16)
    # - num_rows: number of rows in shared memory (e.g., 128 for BM=128)
    # - row_size: BN dimension (number of columns in shared memory, in elements)
    # - rows_per_warp: number of rows per warp (M dimension per warp)
    # - swizzle_base: swizzle pattern base (e.g., 8 for 128B swizzle)
    # - dtype: C++ type name string (e.g., "__nv_bfloat16") - already converted by codegen visitor
    dtype_cpp = dtype  # dtype is already a C++ type string from the visitor

    # STSM x2 always stores 4 floats = 16 bytes
    floats_per_iter = 4
    stsm_size_bytes = 16  # STSM x2 stores 16 bytes (4 floats * sizeof(float))
    warps_per_warpgroup = 4  # WGMMA uses 4 warps per warp group

    body = (
        "\n__device__ __forceinline__ void stsm_x2_fn_("
        "__nv_bfloat162 v0, __nv_bfloat162 v1, void* p) {\n"
        "    uint32_t s0=*reinterpret_cast<uint32_t*>(&v0);\n"
        "    uint32_t s1=*reinterpret_cast<uint32_t*>(&v1);\n"
        '    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16'
        ' [%0], {%1, %2};\\n"\n'
        '        :: "r"((uint32_t)__cvta_generic_to_shared(p)),'
        ' "r"(s0), "r"(s1));\n'
        "}\n"
        "\n"
        "// Store accumulator to SMEM with 128B swizzle layout (DeepGEMM style)\n"
        "// For BM=128: each math warpgroup handles 64 rows, with m_offset indicating row group\n"
        "// SMEM layout: column-major with 128B swizzle for TMA store\n"
        "__device__ __forceinline__ void store_accum_swizzle_fn(\n"
        f"    {dtype_cpp}* sD, float* acc, int warp_idx, int lane_idx, int m_offset) {{\n"
        f"    constexpr int NUM_BANK_GROUP_BYTES = {stsm_size_bytes};  // 16 bytes = 8 bf16 per bank group\n"
        f"    constexpr int WGMMA_M_PER_WARP = {rows_per_warp};  // Rows per warp (M dimension per warp)\n"
        f"    constexpr int ELEMENT_SIZE_BYTES = sizeof({dtype_cpp});  // Element size from dtype\n"
        f"    constexpr int WARPS_PER_WARPGROUP = {warps_per_warpgroup};  // WGMMA uses 4 warps\n"
        f"    constexpr int NUM_ITER = {num_elems} / {floats_per_iter};  // {num_elems} accumulators / {floats_per_iter} iterations\n"
        f"    constexpr int ACC_REGISTERS_PER_WARP = NUM_ITER / WARPS_PER_WARPGROUP;  // Accumulator registers per warp\n"
        f"    constexpr int SWIZZLE_BASE = {swizzle_base};  // Swizzle pattern base (128B swizzle: 8)\n"
        f"    constexpr int SWIZZLE_D_MODE = ({row_size} / 4) * ELEMENT_SIZE_BYTES;  // 128 bytes for BN=256, bfloat16\n"
        f"    constexpr int BM = {num_rows};  // Block M dimension (e.g., 128 elements)\n"
        "    \n"
        "    #pragma unroll\n"
        "    for (int i = 0; i < NUM_ITER; i++) {\n"
        "        // Which accumulator register group does this belong to? (atom_offset)\n"
        "        int atom_offset = i / ACC_REGISTERS_PER_WARP;  // Which atom (0-3 for 128 accumulators)\n"
        "        int in_atom_offset = i % ACC_REGISTERS_PER_WARP;  // Offset within atom (0-7)\n"
        "        \n"
        "        // Reshape and swizzle: kHasShortcut = true since SWIZZLE_D_MODE/NUM_BANK_GROUP_BYTES == 8\n"
        "        int row = (in_atom_offset / SWIZZLE_BASE) + lane_idx;\n"
        "        int col = in_atom_offset;\n"
        "        col ^= row % (SWIZZLE_D_MODE / NUM_BANK_GROUP_BYTES);  // XOR swizzle: row % 8\n"
        "        \n"
        "        // Calculate SMEM address with m_offset for row group (0 or 64)\n"
        "        uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(sD) +\n"
        "            warp_idx * (WGMMA_M_PER_WARP * SWIZZLE_D_MODE) +  // Warp offset in this group\n"
        "            m_offset * SWIZZLE_D_MODE +                        // Row group offset (0 or 64*128)\n"
        f"            atom_offset * {num_rows} * SWIZZLE_D_MODE +      // Atom offset: atom * num_rows * SWIZZLE_D_MODE\n"
        "            row * (NUM_BANK_GROUP_BYTES * 8) +                 // Row offset\n"
        "            col * NUM_BANK_GROUP_BYTES;                        // Column offset\n"
        "        \n"
        f"        __nv_bfloat162 v0 = __floats2bfloat162_rn(acc[i * {floats_per_iter} + 0], acc[i * {floats_per_iter} + 1]);\n"
        f"        __nv_bfloat162 v1 = __floats2bfloat162_rn(acc[i * {floats_per_iter} + 2], acc[i * {floats_per_iter} + 3]);\n"
        "        \n"
        "        stsm_x2_fn_(v0, v1, smem_ptr);\n"
        "    }\n"
        "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"store_accum_swizzle_fn({sD}, {acc}, "
                   f"{warp_idx}, {lane_idx}, {m_offset})")


def _store_accum_swizzle_eval_arg_type(ctx, acc, sD, warp_idx, lane_idx, m_offset, num_elems, num_rows, row_size,
                                       rows_per_warp, swizzle_base, dtype):
    # dtype is passed as LLType object during type inference
    # In codegen function, it will be converted to C++ type string by the visitor
    # This function is called during type inference to validate types
    pass


@builtin(eval_return_type=void, eval_arg_type=_store_accum_swizzle_eval_arg_type,
         codegen_func=codegen_store_accum_swizzle)
def store_accum_swizzle(acc, sD, warp_idx, lane_idx, m_offset, num_elems, num_rows, row_size, rows_per_warp,
                        swizzle_base, dtype):
    """Store WGMMA accumulator to SMEM with swizzle.
    
    Args:
        acc: Accumulator array (float32)
        sD: Shared memory destination tensor
        warp_idx: Warp index within warp group
        lane_idx: Lane index within warp
        m_offset: M dimension offset
        num_elems: Number of accumulator elements (e.g., 128 for m64n256k16)
        num_rows: number of rows in shared memory (e.g., 128 for BM=128)
        row_size: BN dimension (number of columns in shared memory, in elements, e.g., 256)
        rows_per_warp: Number of rows per warp (M dimension per warp, e.g., 16)
        swizzle_base: Swizzle pattern base (e.g., 8 for 128B swizzle)
        dtype: Destination element type (e.g., ll.bfloat16)
    """
    raise RuntimeError("should not be called")


# ==============================================================================
# WGMMA m64n64k16 instruction (32 output registers)
# ==============================================================================


def codegen_wgmma_m64n64k16(acc, desc_a, desc_b):
    """Generate the WGMMA m64n64k16 inline asm (32 output registers)."""
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(32)])
    regs = ",".join([f"%{i}" for i in range(32)])

    body = ("\n__device__ __forceinline__ void wgmma_m64n64k16_fn("
            "float* c, uint64_t desc_a, uint64_t desc_b) {\n"
            "    asm volatile(\n"
            '        "{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            '        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\\n"\n'
            f'        "{{{regs}}},"\n'
            '        "%32,%33,p,1,1,0,0;\\n}\\n"\n'
            f"        : {out_constraints}\n"
            '        : "l"(desc_a), "l"(desc_b));\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_m64n64k16_fn({acc}, {desc_a}, {desc_b})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n64k16)
def wgmma_m64n64k16(acc, desc_a, desc_b):
    """WGMMA m64n64k16 bf16 instruction with 32-float accumulator."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Conflict-11 union: HEAD's m64n64k16 rs/ss parameterized family and
# dff458fb's m64n64/128/256k16 bf16/f16 ss family define disjoint symbols.
# WGMMA m64n64k16 parameterized family (dtype/scale_d/trans flags)
# Legacy `wgmma_m64n64k16` above is byte-identical to the previous release;
# the intrinsics below are pure additions for f16/flag-parameterized issue.
# ==============================================================================


def codegen_wgmma_m64n64k16_ss(acc, desc_a, desc_b, dtype="bf16", scale_d=1, trans_a=0, trans_b=0):
    """Generate the parameterized WGMMA m64n64k16 SS inline asm (32 outputs).

    dtype selects the PTX operand element type ("bf16" or "f16"); scale_d,
    trans_a and trans_b are compile-time 0/1 immediates baked into the asm.
    scale_d may also be a traced runtime int value, in which case it is passed
    as a register operand feeding the scale-d predicate (same pattern as the
    n128 SS form).
    """
    ptx_t = _wgmma_ptx_dtype(dtype)
    ta = _wgmma_imm_flag(trans_a, "trans_a")
    tb = _wgmma_imm_flag(trans_b, "trans_b")
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(32)])
    regs = ",".join([f"%{i}" for i in range(32)])

    if _wgmma_is_compile_time_int(scale_d):
        # Compile-time scale_d must be a 0/1 immediate (raises otherwise);
        # traced runtime values take the register-predicate path below.
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        runtime_sd = False
        in_operands = '"l"(desc_a), "l"(desc_b)'
        call_args = f"{acc}, {desc_a}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sd_ref = "%32,%33,p"
    else:
        runtime_sd = True
        in_operands = '"l"(desc_a), "l"(desc_b), "r"(scale_d)'
        call_args = f"{acc}, {desc_a}, {desc_b}, {scale_d}"
        pred_line = '"{\\n"\n        ".reg .pred p;\\n"\n        "setp.ne.b32 p, %34, 0;\\n"\n'
        sd_ref = "%32,%33,p"

    body = (f"\n__device__ __forceinline__ void wgmma_m64n64k16_ss_{ptx_t}_fn("
            f"float* c, uint64_t desc_a, uint64_t desc_b{', int scale_d' if runtime_sd else ''}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            f'        "wgmma.mma_async.sync.aligned.m64n64k16.f32.{ptx_t}.{ptx_t}\\n"\n'
            f'        "{{{regs}}},"\n'
            f'        "{sd_ref},1,1,{ta},{tb};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_m64n64k16_ss_{ptx_t}_fn({call_args})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n64k16_ss)
def wgmma_m64n64k16_ss(acc, desc_a, desc_b, dtype="bf16", scale_d=1, trans_a=0, trans_b=0):
    """WGMMA m64n64k16 SS (shared-shared) instruction, dtype/flag parameterized.

    Both A and B operands come from shared memory via 64-bit descriptors.

    Args:
        acc: 32-element float accumulator array
        desc_a: Descriptor for A matrix in shared memory
        desc_b: Descriptor for B matrix in shared memory
        dtype: Operand element type, "bf16" or "f16" (default "bf16")
        scale_d: 1 to accumulate, 0 to overwrite (default 1); a traced runtime
            int value is also accepted and passed as a register predicate
        trans_a: 1 to read A column-wise from shared memory (default 0)
        trans_b: 1 to read B column-wise from shared memory (default 0)
    """
    raise RuntimeError("should not be called in compilation")


def codegen_wgmma_m64n64k16_rs(acc, a0, a1, a2, a3, desc_b, dtype="bf16", scale_d=1, trans_b=0):
    """Generate the parameterized WGMMA m64n64k16 RS inline asm (32 outputs).

    A operand is in registers (4 x uint32_t = 8 x 16-bit values per thread).
    B operand is in shared memory via descriptor. Register-A operands have no
    transpose immediate (their layout is fixed); trans_b applies to B only.
    """
    ptx_t = _wgmma_ptx_dtype(dtype)
    tb = _wgmma_imm_flag(trans_b, "trans_b")
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(32)])
    regs = ",".join([f"%{i}" for i in range(32)])

    if _wgmma_is_compile_time_int(scale_d):
        # Compile-time scale_d must be a 0/1 immediate (raises otherwise);
        # traced runtime values take the register-predicate path below.
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        runtime_sd = False
        in_operands = ('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b)')
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        op_ref = "{%32,%33,%34,%35},%36,p"
    else:
        runtime_sd = True
        in_operands = ('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), '
                       '"r"(scale_d)')
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}, {scale_d}"
        pred_line = '"{\\n"\n        ".reg .pred p;\\n"\n        "setp.ne.b32 p, %37, 0;\\n"\n'
        op_ref = "{%32,%33,%34,%35},%36,p"

    body = (f"\n__device__ __forceinline__ void wgmma_m64n64k16_rs_{ptx_t}_fn("
            "float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, "
            f"uint64_t desc_b{', int scale_d' if runtime_sd else ''}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            f'        "wgmma.mma_async.sync.aligned.m64n64k16.f32.{ptx_t}.{ptx_t}\\n"\n'
            f'        "{{{regs}}},"\n'
            f'        "{op_ref},1,1,{tb};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_m64n64k16_rs_{ptx_t}_fn({call_args})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n64k16_rs)
def wgmma_m64n64k16_rs(acc, a0, a1, a2, a3, desc_b, dtype="bf16", scale_d=1, trans_b=0):
    """WGMMA m64n64k16 RS (register-shared) instruction, dtype/flag parameterized.

    A operand from registers, B operand from shared memory. Used when the A
    tile is produced in accumulator registers by a preceding wgmma (e.g. an
    attention-style fused transform) and packed in place, avoiding a shared
    memory round-trip.

    Args:
        acc: 32-element float accumulator array
        a0, a1, a2, a3: A matrix in registers (4 x uint32_t = 8 x 16-bit values)
        desc_b: Descriptor for B matrix in shared memory
        dtype: Operand element type, "bf16" or "f16" (default "bf16")
        scale_d: 1 to accumulate, 0 to overwrite (default 1); a traced runtime
            int value is also accepted and passed as a register predicate
        trans_b: 1 to read B column-wise from shared memory (default 0)
    """
    raise RuntimeError("should not be called in compilation")


# Generic WGMMA m64nNk16 SS instructions: operand dtype + trans/scale immediates
#
# The instruction variant is selected from explicit, recorded facts at the
# call site (dtype in the builtin name, tile shape N in the name) rather than
# inferred from names of surrounding ops (TileLang GEMM lowering skill steps
# 1-2). transpose/scale_d operands are PTX immediates, so they must be
# integer literals 0 or 1 at compile time: a non-literal or out-of-range
# value is a hard compile error, never a silently demoted instruction.
# ==============================================================================

_WGMMA_SS_DTYPES = ("f16", "bf16")
_WGMMA_SS_NCOLS = (64, 128, 256)


def _wgmma_const_flag(value, label, builtin_name):
    """Validate a WGMMA immediate flag that must be a literal 0 or 1."""
    try:
        parsed = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{builtin_name}: {label} is a PTX immediate and must be an "
                         f"integer literal 0 or 1 at compile time (got {value!r}); zero "
                         "the accumulator before the loop instead of splitting scale_d")
    if parsed not in (0, 1):
        raise ValueError(f"{builtin_name}: {label} must be 0 or 1 (got {parsed})")
    return parsed


def _make_codegen_wgmma_ss(dtype_ptx, ncols):
    """Build the codegen for one WGMMA m64nNk16 SS dtype variant.

    The generated helper name carries the immediate values, so call sites
    with different trans/scale combinations never share a definition, and
    identical combinations emit identical helper text.
    """
    n_out = ncols // 2
    builtin_name = f"wgmma_m64n{ncols}k16_{dtype_ptx}"

    def codegen_wgmma_ss(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
        ta = _wgmma_const_flag(trans_a, "trans_a", builtin_name)
        tb = _wgmma_const_flag(trans_b, "trans_b", builtin_name)
        sd = _wgmma_const_flag(scale_d, "scale_d", builtin_name)

        # Fragment-aware path (ir.wgmma.acc_fragment): when the accumulator
        # operand names a registered typed register fragment, bind each of
        # its per-thread scalars as an individual "+f" asm operand through a
        # float&-parameter helper instead of taking a float* array address.
        # This is what keeps the fragment in registers when the frontend
        # subscripts elements between MMA calls (online-softmax epilogues).
        frag_args = _try_resolve_fragment_arg(acc, n_out, builtin_name)
        if frag_args is not None:
            helper = (f"wgmma_m64n{ncols}k16_{dtype_ptx}"
                      f"_a{ta}b{tb}d{sd}_frag_fn")
            out_constraints = ",".join([f'"+f"(c{i})' for i in range(n_out)])
            reg_block = _wgmma_reg_block(n_out)
            params = ", ".join([f"float& c{i}" for i in range(n_out)])
            body = (f"\n__device__ __forceinline__ void {helper}("
                    f"{params}, uint64_t desc_a, uint64_t desc_b) {{\n"
                    "    asm volatile(\n"
                    f'        "{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd}, 0;\\n"\n'
                    '        "wgmma.mma_async.sync.aligned.'
                    f'm64n{ncols}k16.f32.{dtype_ptx}.{dtype_ptx}\\n"\n'
                    f'        "{{{reg_block}}},"\n'
                    f'        "%{n_out},%{n_out + 1},p,1,1,{ta},{tb};\\n}}\\n"\n'
                    f"        : {out_constraints}\n"
                    '        : "l"(desc_a), "l"(desc_b));\n'
                    "}\n")
            return Builtin(body=body, includes=[], return_val=f"{helper}({frag_args}, {desc_a}, {desc_b})")

        helper = f"wgmma_m64n{ncols}k16_{dtype_ptx}_a{ta}b{tb}d{sd}_fn"
        out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(n_out)])
        reg_lines = []
        for row_start in range(0, n_out, 8):
            regs = ",".join([f"%{i}" for i in range(row_start, row_start + 8)])
            reg_lines.append(regs)
        reg_block = ",\\n".join(reg_lines)

        body = (f"\n__device__ __forceinline__ void {helper}("
                "float* c, uint64_t desc_a, uint64_t desc_b) {\n"
                "    asm volatile(\n"
                f'        "{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd}, 0;\\n"\n'
                '        "wgmma.mma_async.sync.aligned.'
                f'm64n{ncols}k16.f32.{dtype_ptx}.{dtype_ptx}\\n"\n'
                f'        "{{{reg_block}}},"\n'
                f'        "%{n_out},%{n_out + 1},p,1,1,{ta},{tb};\\n}}\\n"\n'
                f"        : {out_constraints}\n"
                '        : "l"(desc_a), "l"(desc_b));\n'
                "}\n")
        return Builtin(body=body, includes=[], return_val=f"{helper}({acc}, {desc_a}, {desc_b})")

    return codegen_wgmma_ss


_WGMMA_SS_DOC = """WGMMA m64n{n}k16 SS {dt} instruction with {regs}-float accumulator.

Operand dtype and tile N select the instruction
`wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{ptx}.{ptx}`; trans_a/trans_b
become the two transpose immediates (0 = K-major descriptor layout,
1 = MN-major descriptor layout) and scale_d gates accumulation
(1 = accumulate into the accumulator, 0 = overwrite it, computed zero) as
compile-time-baked PTX immediates. All three flags must be integer literals
0 or 1; any other value is a compile error rather than a silent demotion.

Args:
    acc: float accumulator array ({regs} elements across the warpgroup)
    desc_a: shared-memory matrix descriptor for A
    desc_b: shared-memory matrix descriptor for B
    trans_a: 0 or 1 literal, A layout selector immediate
    trans_b: 0 or 1 literal, B layout selector immediate
    scale_d: 0 or 1 literal, accumulate-or-overwrite immediate
"""


def _wgmma_ss_doc(dtype_ptx, ncols):
    return _WGMMA_SS_DOC.format(dt=dtype_ptx.upper(), n=ncols, ptx=dtype_ptx, regs=ncols // 2)


# Each exported builtin below keeps an inspectable signature and docstring
# for frontend authors; the shared parameterized generator above produces
# the inline-asm helper for a given (dtype, N) pair.

codegen_wgmma_m64n64k16_f16 = _make_codegen_wgmma_ss("f16", 64)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n64k16_f16)
def wgmma_m64n64k16_f16(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
    raise RuntimeError("should not be called in compilation")


wgmma_m64n64k16_f16.__doc__ = _wgmma_ss_doc("f16", 64)

codegen_wgmma_m64n64k16_bf16 = _make_codegen_wgmma_ss("bf16", 64)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n64k16_bf16)
def wgmma_m64n64k16_bf16(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
    raise RuntimeError("should not be called in compilation")


wgmma_m64n64k16_bf16.__doc__ = _wgmma_ss_doc("bf16", 64)

codegen_wgmma_m64n128k16_f16 = _make_codegen_wgmma_ss("f16", 128)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n128k16_f16)
def wgmma_m64n128k16_f16(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
    raise RuntimeError("should not be called in compilation")


wgmma_m64n128k16_f16.__doc__ = _wgmma_ss_doc("f16", 128)

codegen_wgmma_m64n128k16_bf16 = _make_codegen_wgmma_ss("bf16", 128)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n128k16_bf16)
def wgmma_m64n128k16_bf16(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
    raise RuntimeError("should not be called in compilation")


wgmma_m64n128k16_bf16.__doc__ = _wgmma_ss_doc("bf16", 128)

codegen_wgmma_m64n256k16_f16 = _make_codegen_wgmma_ss("f16", 256)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n256k16_f16)
def wgmma_m64n256k16_f16(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
    raise RuntimeError("should not be called in compilation")


wgmma_m64n256k16_f16.__doc__ = _wgmma_ss_doc("f16", 256)

codegen_wgmma_m64n256k16_bf16 = _make_codegen_wgmma_ss("bf16", 256)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n256k16_bf16)
def wgmma_m64n256k16_bf16(acc, desc_a, desc_b, trans_a, trans_b, scale_d):
    raise RuntimeError("should not be called in compilation")


wgmma_m64n256k16_bf16.__doc__ = _wgmma_ss_doc("bf16", 256)

# ==============================================================================
# Generic-N WGMMA dispatchers (full PTX tiling whitelist + scale immediates)
#
# PTX `wgmma.mma_async.sync.aligned.m64nNk16.f32.{f16,bf16}.{f16,bf16}` is
# defined for every N in {8, 16, 24, ..., 256} (multiples of 8), with a float
# accumulator of N/2 registers per thread and ±1 scale_a/scale_b immediates.
# The fixed-shape intrinsics above only expose N in {64, 128, 256} and pin
# scale_a/scale_b to 1; the dispatchers below take the tile N and the ±1
# scale immediates as whitelist-checked compile-time arguments so any legal
# tensor-core tiling (and negative-operand accumulation) has a typed path.
#
# Legality model (cw.codegen.tilelang-ptx-inline-assembly): n, dtype, flags
# and scales are validated against an explicit whitelist BEFORE any asm text
# is composed; illegal configurations raise ValueError, never a malformed
# template. The accumulator epilogue helpers (store_acc_to_global,
# store_acc_to_smem_bf16) already take num_regs/wgmma_n parameters and their
# chunked coordinate mapping stays exact for every N in the whitelist, so no
# epilogue change is needed for the newly reachable shapes.
# ==============================================================================

# Every PTX-legal N for m64nNk16 with f16/bf16 operands and f32 accumulator.
_WGMMA_LEGAL_NCOLS = tuple(range(8, 257, 8))


def _wgmma_check_ncols(n, builtin_name, family="m64nNk16 (f16/bf16 operands, "
                       "f32 accumulator)"):
    """Validate a compile-time WGMMA tile N against the PTX shape whitelist."""
    try:
        parsed = int(n)
    except (ValueError, TypeError):
        raise ValueError(f"{builtin_name}: n is a PTX shape immediate and must be a "
                         f"compile-time integer in [8, 256] step 8 (got {n!r})")
    if isinstance(n, float) and n != parsed:
        raise ValueError(f"{builtin_name}: n is a PTX shape immediate and must be an "
                         f"integer, not {n!r}")
    if parsed not in _WGMMA_LEGAL_NCOLS:
        raise ValueError(f"{builtin_name}: illegal WGMMA tile N {parsed}; PTX defines "
                         f"{family} for N a multiple of 8 in [8, 256]")
    return parsed


def _wgmma_signed_scale(value, name, builtin_name):
    """Validate a PTX ±1 scale immediate (scale_a / scale_b)."""
    try:
        parsed = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{builtin_name}: {name} is a PTX immediate and must be a "
                         f"compile-time integer, +1 or -1 (got {value!r})")
    if parsed not in (1, -1):
        raise ValueError(f"{builtin_name}: {name} must be +1 or -1 (got {parsed}); PTX "
                         "provides no other operand-scale values")
    return parsed


def _wgmma_scale_token(value):
    """Legal, injective C++ name token for a validated ±1 scale immediate.

    scale_a/scale_b are PTX immediates restricted to {+1, -1}; interpolating
    the signed integer directly into the helper name yields ``...sa-1...``,
    which is not a legal C++ identifier (``-`` is an operator). Encode +1 as
    ``1`` and -1 as ``n1`` instead; the emitted PTX operand keeps its real
    signed value, only the helper identifier is affected.
    """
    if value == 1:
        return "1"
    if value == -1:
        return "n1"
    raise ValueError(f"_wgmma_scale_token: only ±1 scale immediates are encodable "
                     f"(got {value!r}); run _wgmma_signed_scale validation first")


def _wgmma_reg_block(n_out):
    """Register block text for an n_out-float accumulator (8 per line)."""
    reg_lines = []
    for row_start in range(0, n_out, 8):
        regs = ",".join([f"%{i}" for i in range(row_start, min(row_start + 8, n_out))])
        reg_lines.append(regs)
    return ",\\n".join(reg_lines)


# ==============================================================================
# Integer WGMMA (m64nNk32.s32.{s8,u8}.{s8,u8}) SS/RS dispatchers
#
# PTX defines integer wgmma for N a multiple of 8 in [8, 256] with a k32
# depth, an exact s32 accumulator of N/2 registers per thread, and 8-bit
# integer operands from shared-memory descriptors (SS) or registers (RS).
# Unlike the f16/bf16 family, the integer opcodes take NO scale_a/scale_b
# operand-scale immediates and NO tnspA/tnspB transpose immediates: the only
# immediate operand is scale-d (accumulate vs overwrite) as a predicate. This
# is exactly the opcode issued by the integer GEMM teacher
# (wgmma.mma_async.sync.aligned.m64n128k32.s32.s8.s8): scale_a/scale_b must
# be +1 and trans_a/trans_b must be 0 or the call is a compile error, never a
# silently demoted instruction.
# ==============================================================================


def _wgmma_int_check_flags(scale_a, scale_b, trans_a, trans_b, builtin_name):
    """Integer wgmma has no scale/trans immediates: pin them to identity."""
    sa = _wgmma_signed_scale(scale_a, "scale_a", builtin_name)
    sb = _wgmma_signed_scale(scale_b, "scale_b", builtin_name)
    if sa != 1 or sb != 1:
        raise ValueError(f"{builtin_name}: integer wgmma (s8/u8 -> s32) has no operand-"
                         f"scale immediates; scale_a/scale_b must both be +1 (got "
                         f"scale_a={sa}, scale_b={sb})")
    ta = _wgmma_const_flag(trans_a, "trans_a", builtin_name)
    tb = _wgmma_const_flag(trans_b, "trans_b", builtin_name)
    if ta != 0 or tb != 0:
        raise ValueError(f"{builtin_name}: integer wgmma (s8/u8 -> s32) has no tnsp "
                         f"transpose immediates; trans_a/trans_b must both be 0 (got "
                         f"trans_a={ta}, trans_b={tb}); use a K-major SWIZZLE descriptor "
                         "layout")


def _wgmma_int_reg_block(n_out):
    """Register block text for an n_out-int32 accumulator (8 per line)."""
    reg_lines = []
    for row_start in range(0, n_out, 8):
        regs = ",".join([f"%{i}" for i in range(row_start, min(row_start + 8, n_out))])
        reg_lines.append(regs)
    return ",\\n".join(reg_lines)


def _wgmma_is_int_dtype(dtype):
    """True iff a frontend dtype token names an integer wgmma operand type."""
    token = dtype
    if not isinstance(token, str):
        token = getattr(token, "name", None) or str(token)
    token = str(token).strip().strip("'\"").lower()
    return token in _WGMMA_INT_DTYPE_ALIASES


def codegen_wgmma_mma_async_ss_int(acc, desc_a, desc_b, n, dtype="s8", scale_a=1, scale_b=1, scale_d=1, trans_a=0,
                                   trans_b=0):
    """Generate an integer WGMMA m64nNk32 SS instruction (n//2 s32 outputs).

    Emits `wgmma.mma_async.sync.aligned.m64n{n}k32.s32.{t}.{t}` with a read-
    write ("+r") int32 accumulator: scale-d selects accumulate (1) vs
    overwrite (0) as either a compile-time immediate baked into the asm or a
    traced runtime int feeding the predicate register. scale_a/scale_b must
    be +1 and trans_a/trans_b must be 0 (integer wgmma has neither
    immediate); anything else is a ValueError raised before any asm text is
    composed.
    """
    builtin_name = "wgmma_mma_async_ss"
    ptx_t = _wgmma_int_ptx_dtype(dtype)
    ncols = _wgmma_check_ncols(n, builtin_name, "m64nNk32 (s8/u8 operands, s32 accumulator)")
    _wgmma_int_check_flags(scale_a, scale_b, trans_a, trans_b, builtin_name)
    n_out = ncols // 2
    # Signed 32-bit accumulator registers: "+r" read-write constraints (the
    # instruction reads the accumulator when scale-d=1, so write-only "=r"
    # would let the compiler drop prior MMA results).
    out_constraints = ",".join([f'"+r"(c[{i}])' for i in range(n_out)])
    reg_block = _wgmma_int_reg_block(n_out)

    if _wgmma_is_compile_time_int(scale_d):
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        helper = f"wgmma_mma_ss_n{ncols}k32_s32_{ptx_t}_{ptx_t}_d{sd_imm}_fn"
        call_args = f"{acc}, {desc_a}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sig_extra, in_operands = "", '"l"(desc_a), "l"(desc_b)'
    else:
        helper = f"wgmma_mma_ss_n{ncols}k32_s32_{ptx_t}_{ptx_t}_dr_fn"
        call_args = f"{acc}, {desc_a}, {desc_b}, {scale_d}"
        pred_line = ('"{\\n"\n        ".reg .pred p;\\n"\n'
                     f'        "setp.ne.b32 p, %{n_out + 2}, 0;\\n"\n')
        sig_extra, in_operands = (", int scale_d", '"l"(desc_a), "l"(desc_b), "r"(scale_d)')

    body = (f"\n__device__ __forceinline__ void {helper}("
            f"int32_t* c, uint64_t desc_a, uint64_t desc_b{sig_extra}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            '        "wgmma.mma_async.sync.aligned.'
            f'm64n{ncols}k32.s32.{ptx_t}.{ptx_t}\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "%{n_out},%{n_out + 1},p;\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"{helper}({call_args})")


# ==============================================================================
# FP8 WGMMA (m64nNk32.f32.{e4m3,e5m2}.{e4m3,e5m2}) SS/RS dispatchers
#
# PTX defines FP8 wgmma for N a multiple of 8 in [8, 256] with a k32 depth,
# an f32 accumulator of N/2 registers per thread, and 8-bit floating-point
# operands from shared-memory descriptors (SS) or registers (RS). The two
# operand element types may differ: e4m3/e5m2 mix freely across the A/B dot
# slots (wgmma.mma_async.sync.aligned.m64nNk32.f32.e4m3.e5m2 etc.). Like the
# integer family, FP8 wgmma has NO scale_a/scale_b operand-scale immediates
# and NO tnspA/tnspB transpose immediates: the only immediate operand is
# scale-d (accumulate vs overwrite) as a predicate. scale_a/scale_b must be
# +1 and trans_a/trans_b must be 0 or the call is a compile error, never a
# silently demoted instruction.
# ==============================================================================


def _wgmma_fp8_check_flags(scale_a, scale_b, trans_a, trans_b, builtin_name):
    """FP8 wgmma has no scale/trans immediates: pin them to identity."""
    sa = _wgmma_signed_scale(scale_a, "scale_a", builtin_name)
    sb = _wgmma_signed_scale(scale_b, "scale_b", builtin_name)
    if sa != 1 or sb != 1:
        raise ValueError(f"{builtin_name}: FP8 wgmma (e4m3/e5m2 -> f32) has no operand-"
                         f"scale immediates; scale_a/scale_b must both be +1 (got "
                         f"scale_a={sa}, scale_b={sb})")
    ta = _wgmma_const_flag(trans_a, "trans_a", builtin_name)
    tb = _wgmma_const_flag(trans_b, "trans_b", builtin_name)
    if ta != 0 or tb != 0:
        raise ValueError(f"{builtin_name}: FP8 wgmma (e4m3/e5m2 -> f32) has no tnsp "
                         f"transpose immediates; trans_a/trans_b must both be 0 (got "
                         f"trans_a={ta}, trans_b={tb}); use a K-major SWIZZLE descriptor "
                         "layout")


def codegen_wgmma_mma_async_ss_fp8(acc, desc_a, desc_b, n, dtype="e4m3", dtype_b=None, scale_a=1, scale_b=1, scale_d=1,
                                   trans_a=0, trans_b=0):
    """Generate an FP8 WGMMA m64nNk32 SS instruction (n//2 f32 outputs).

    Emits `wgmma.mma_async.sync.aligned.m64n{n}k32.f32.{ta}.{tb}` where ta/tb
    are the A/B operand element types drawn independently from {e4m3, e5m2}
    (mixed FP8 forms are PTX-legal). scale-d selects accumulate (1) vs
    overwrite (0) as either a compile-time immediate baked into the asm or a
    traced runtime int feeding the predicate register. scale_a/scale_b must
    be +1 and trans_a/trans_b must be 0 (FP8 wgmma has neither immediate);
    anything else is a ValueError raised before any asm text is composed.
    """
    builtin_name = "wgmma_mma_async_ss"
    ptx_a = _wgmma_fp8_ptx_dtype(dtype)
    ptx_b = (_wgmma_fp8_ptx_dtype(dtype_b) if dtype_b is not None else ptx_a)
    ncols = _wgmma_check_ncols(n, builtin_name, "m64nNk32 (e4m3/e5m2 operands, f32 accumulator)")
    _wgmma_fp8_check_flags(scale_a, scale_b, trans_a, trans_b, builtin_name)
    n_out = ncols // 2
    # f32 accumulator registers: "+f" read-write constraints (the instruction
    # reads the accumulator when scale-d=1, so write-only "=f" would let the
    # compiler drop prior MMA results).
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(n_out)])
    reg_block = _wgmma_reg_block(n_out)

    if _wgmma_is_compile_time_int(scale_d):
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        helper = (f"wgmma_mma_ss_n{ncols}k32_f32_{ptx_a}_{ptx_b}"
                  f"_d{sd_imm}_fn")
        call_args = f"{acc}, {desc_a}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sig_extra, in_operands = "", '"l"(desc_a), "l"(desc_b)'
    else:
        helper = (f"wgmma_mma_ss_n{ncols}k32_f32_{ptx_a}_{ptx_b}"
                  f"_dr_fn")
        call_args = f"{acc}, {desc_a}, {desc_b}, {scale_d}"
        pred_line = ('"{\\n"\n        ".reg .pred p;\\n"\n'
                     f'        "setp.ne.b32 p, %{n_out + 2}, 0;\\n"\n')
        sig_extra, in_operands = (", int scale_d", '"l"(desc_a), "l"(desc_b), "r"(scale_d)')

    body = (f"\n__device__ __forceinline__ void {helper}("
            f"float* c, uint64_t desc_a, uint64_t desc_b{sig_extra}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            '        "wgmma.mma_async.sync.aligned.'
            f'm64n{ncols}k32.f32.{ptx_a}.{ptx_b}\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "%{n_out},%{n_out + 1},p;\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"{helper}({call_args})")


def codegen_wgmma_mma_async_ss(acc, desc_a, desc_b, n, dtype="bf16", scale_a=1, scale_b=1, scale_d=1, trans_a=0,
                               trans_b=0, dtype_b=None):
    """Generate a genericN WGMMA m64nNk16 SS instruction (n//2 outputs).

    n selects the tile shape from the full PTX whitelist {8..256, step 8};
    dtype selects f16/bf16 operands; scale_a/scale_b are ±1 operand-scale
    immediates; trans_a/trans_b are 0/1 layout immediates. scale_d may be a
    compile-time 0/1 immediate or a traced runtime int (register predicate).

    Integer operand dtypes (s8/u8) route to the integer k32 s32-accumulator
    opcode family instead (see codegen_wgmma_mma_async_ss_int). FP8 operand
    dtypes (e4m3/e5m2) route to the FP8 k32 f32-accumulator family (see
    codegen_wgmma_mma_async_ss_fp8); the optional dtype_b selects a different
    B-operand element type for the PTX-legal mixed FP8 forms.
    """
    _wgmma_check_dtype_b_family(dtype, dtype_b, "wgmma_mma_async_ss")
    if _wgmma_is_int_dtype(dtype):
        return codegen_wgmma_mma_async_ss_int(acc, desc_a, desc_b, n, dtype=dtype, scale_a=scale_a, scale_b=scale_b,
                                              scale_d=scale_d, trans_a=trans_a, trans_b=trans_b)
    if _wgmma_is_fp8_dtype(dtype):
        return codegen_wgmma_mma_async_ss_fp8(acc, desc_a, desc_b, n, dtype=dtype, dtype_b=dtype_b, scale_a=scale_a,
                                              scale_b=scale_b, scale_d=scale_d, trans_a=trans_a, trans_b=trans_b)
    builtin_name = "wgmma_mma_async_ss"
    ptx_t = _wgmma_ptx_dtype(dtype)
    ncols = _wgmma_check_ncols(n, builtin_name)
    sa = _wgmma_signed_scale(scale_a, "scale_a", builtin_name)
    sb = _wgmma_signed_scale(scale_b, "scale_b", builtin_name)
    ta = _wgmma_const_flag(trans_a, "trans_a", builtin_name)
    tb = _wgmma_const_flag(trans_b, "trans_b", builtin_name)
    n_out = ncols // 2
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(n_out)])
    reg_block = _wgmma_reg_block(n_out)

    if _wgmma_is_compile_time_int(scale_d):
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        helper = (f"wgmma_mma_ss_n{ncols}_{ptx_t}"
                  f"_sa{_wgmma_scale_token(sa)}sb{_wgmma_scale_token(sb)}"
                  f"d{sd_imm}ta{ta}tb{tb}_fn")
        call_args = f"{acc}, {desc_a}, {desc_b}"
        sd_ref = f"%{n_out},%{n_out + 1},p"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sig_extra, in_operands = "", '"l"(desc_a), "l"(desc_b)'
    else:
        helper = (f"wgmma_mma_ss_n{ncols}_{ptx_t}"
                  f"_sa{_wgmma_scale_token(sa)}sb{_wgmma_scale_token(sb)}"
                  f"drta{ta}tb{tb}_fn")
        call_args = f"{acc}, {desc_a}, {desc_b}, {scale_d}"
        sd_ref = f"%{n_out},%{n_out + 1},p"
        pred_line = ('"{\\n"\n        ".reg .pred p;\\n"\n'
                     f'        "setp.ne.b32 p, %{n_out + 2}, 0;\\n"\n')
        sig_extra, in_operands = (", int scale_d", '"l"(desc_a), "l"(desc_b), "r"(scale_d)')

    body = (f"\n__device__ __forceinline__ void {helper}("
            f"float* c, uint64_t desc_a, uint64_t desc_b{sig_extra}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            '        "wgmma.mma_async.sync.aligned.'
            f'm64n{ncols}k16.f32.{ptx_t}.{ptx_t}\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "{sd_ref},{sa},{sb},{ta},{tb};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"{helper}({call_args})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_mma_async_ss)
def wgmma_mma_async_ss(acc, desc_a, desc_b, n, dtype="bf16", scale_a=1, scale_b=1, scale_d=1, trans_a=0, trans_b=0,
                       dtype_b=None):
    """GenericN WGMMA m64nNk16 SS (shared-shared) instruction.

    Dispatches to `wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{t}.{t}` for
    any PTX-legal tile N, with explicit operand-scale immediates. Integer
    operand dtypes ("s8"/"int8", "u8"/"uint8") route to the integer
    k32 s32-accumulator family `...m64n{n}k32.s32.{t}.{t}` instead, which has
    no operand-scale or transpose immediates (scale_a/scale_b must be +1 and
    trans_a/trans_b must be 0, else ValueError). FP8 operand dtypes
    ("e4m3"/"fp8_e4m3", "e5m2"/"fp8_e5m2") route to the FP8 k32
    f32-accumulator family `...m64n{n}k32.f32.{ta}.{tb}`; FP8 wgmma also has
    no scale/trans immediates, but the two operand element types may differ
    (dtype_b defaulting to dtype) so the PTX-legal mixed forms
    `.e4m3.e5m2` / `.e5m2.e4m3` are expressible.

    Args:
        acc: float accumulator array (n//2 elements per thread), or int32 for
            integer operand dtypes
        desc_a: shared-memory matrix descriptor for A
        desc_b: shared-memory matrix descriptor for B
        n: tile N, a compile-time multiple of 8 in [8, 256]
        dtype: operand element type, "bf16" or "f16" (default "bf16"); "s8"
            or "u8" select the integer k32 -> s32 family, "e4m3"/"e5m2"
            select the FP8 k32 -> f32 family
        scale_a: +1 or -1 operand-scale immediate for A (default +1; must be
            +1 for integer and FP8 dtypes)
        scale_b: +1 or -1 operand-scale immediate for B (default +1; must be
            +1 for integer and FP8 dtypes)
        scale_d: 1 to accumulate, 0 to overwrite (default 1); a traced runtime
            int value is also accepted and passed as a register predicate
        trans_a: 1 to read A column-wise from shared memory (default 0; must
            be 0 for integer and FP8 dtypes)
        trans_b: 1 to read B column-wise from shared memory (default 0; must
            be 0 for integer and FP8 dtypes)
        dtype_b: B-operand element type override (default None = same as
            dtype); only meaningful for FP8 operand families, where the two
            operand types may differ
    """
    raise RuntimeError("should not be called in compilation")


def codegen_wgmma_mma_async_rs_int(acc, a0, a1, a2, a3, desc_b, n, dtype="s8", scale_a=1, scale_b=1, scale_d=1,
                                   trans_b=0):
    """Generate an integer WGMMA m64nNk32 RS instruction (n//2 s32 outputs).

    A operand is in registers (4 x uint32_t = 16 x 8-bit values per thread,
    layout fixed by the opcode, so there is no trans immediate); B comes from
    shared memory via descriptor. Emits
    `wgmma.mma_async.sync.aligned.m64n{n}k32.s32.{t}.{t}` with a read-write
    ("+r") int32 accumulator. scale_a/scale_b must be +1 and trans_b must be
    0 (integer wgmma has neither immediate); anything else is a ValueError
    raised before any asm text is composed, mirroring the SS dispatcher.
    """
    builtin_name = "wgmma_mma_async_rs"
    ptx_t = _wgmma_int_ptx_dtype(dtype)
    ncols = _wgmma_check_ncols(n, builtin_name, "m64nNk32 (s8/u8 operands, s32 accumulator)")
    _wgmma_int_check_flags(scale_a, scale_b, 0, trans_b, builtin_name)
    n_out = ncols // 2
    out_constraints = ",".join([f'"+r"(c[{i}])' for i in range(n_out)])
    reg_block = _wgmma_int_reg_block(n_out)
    a_ref = (f"{{%{(n_out)},%{n_out + 1},%{n_out + 2},%{n_out + 3}}}"
             f",%{n_out + 4},p")

    if _wgmma_is_compile_time_int(scale_d):
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        helper = f"wgmma_mma_rs_n{ncols}k32_s32_{ptx_t}_{ptx_t}_d{sd_imm}_fn"
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sig_extra = ""
        in_operands = '"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b)'
    else:
        helper = f"wgmma_mma_rs_n{ncols}k32_s32_{ptx_t}_{ptx_t}_dr_fn"
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}, {scale_d}"
        pred_line = ('"{\\n"\n        ".reg .pred p;\\n"\n'
                     f'        "setp.ne.b32 p, %{n_out + 5}, 0;\\n"\n')
        sig_extra = ", int scale_d"
        in_operands = ('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), '
                       '"r"(scale_d)')

    body = (f"\n__device__ __forceinline__ void {helper}("
            f"int32_t* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, "
            f"uint64_t desc_b{sig_extra}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            '        "wgmma.mma_async.sync.aligned.'
            f'm64n{ncols}k32.s32.{ptx_t}.{ptx_t}\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "{a_ref};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=["<cstdint>"], return_val=f"{helper}({call_args})")


def codegen_wgmma_mma_async_rs_fp8(acc, a0, a1, a2, a3, desc_b, n, dtype="e4m3", dtype_b=None, scale_a=1, scale_b=1,
                                   scale_d=1, trans_b=0):
    """Generate an FP8 WGMMA m64nNk32 RS instruction (n//2 f32 outputs).

    A operand is in registers (4 x uint32_t = 32 x 8-bit FP8 values per
    thread, layout fixed by the opcode, so there is no trans immediate); B
    comes from shared memory via descriptor. Emits
    `wgmma.mma_async.sync.aligned.m64n{n}k32.f32.{ta}.{tb}` with a read-write
    ("+f") float accumulator; ta/tb are drawn independently from
    {e4m3, e5m2}. scale_a/scale_b must be +1 and trans_b must be 0 (FP8
    wgmma has neither immediate); anything else is a ValueError raised
    before any asm text is composed, mirroring the SS dispatcher.
    """
    builtin_name = "wgmma_mma_async_rs"
    ptx_a = _wgmma_fp8_ptx_dtype(dtype)
    ptx_b = (_wgmma_fp8_ptx_dtype(dtype_b) if dtype_b is not None else ptx_a)
    ncols = _wgmma_check_ncols(n, builtin_name, "m64nNk32 (e4m3/e5m2 operands, f32 accumulator)")
    _wgmma_fp8_check_flags(scale_a, scale_b, 0, trans_b, builtin_name)
    n_out = ncols // 2
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(n_out)])
    reg_block = _wgmma_reg_block(n_out)
    a_ref = (f"{{%{(n_out)},%{n_out + 1},%{n_out + 2},%{n_out + 3}}}"
             f",%{n_out + 4},p")

    if _wgmma_is_compile_time_int(scale_d):
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        helper = (f"wgmma_mma_rs_n{ncols}k32_f32_{ptx_a}_{ptx_b}"
                  f"_d{sd_imm}_fn")
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sig_extra = ""
        in_operands = '"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b)'
    else:
        helper = (f"wgmma_mma_rs_n{ncols}k32_f32_{ptx_a}_{ptx_b}"
                  f"_dr_fn")
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}, {scale_d}"
        pred_line = ('"{\\n"\n        ".reg .pred p;\\n"\n'
                     f'        "setp.ne.b32 p, %{n_out + 5}, 0;\\n"\n')
        sig_extra = ", int scale_d"
        in_operands = ('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), '
                       '"r"(scale_d)')

    body = (f"\n__device__ __forceinline__ void {helper}("
            f"float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, "
            f"uint64_t desc_b{sig_extra}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            '        "wgmma.mma_async.sync.aligned.'
            f'm64n{ncols}k32.f32.{ptx_a}.{ptx_b}\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "{a_ref};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"{helper}({call_args})")


def codegen_wgmma_mma_async_rs(acc, a0, a1, a2, a3, desc_b, n, dtype="bf16", scale_a=1, scale_b=1, scale_d=1, trans_b=0,
                               dtype_b=None):
    """Generate a genericN WGMMA m64nNk16 RS instruction (n//2 outputs).

    A operand is in registers (4 x uint32_t = 8 x 16-bit values per thread,
    layout fixed by the opcode, so there is no trans_a immediate); B comes
    from shared memory via descriptor. Whitelist/flag handling matches the
    SS dispatcher. Integer dtypes route to the k32 s32-accumulator family
    and FP8 dtypes (e4m3/e5m2, optionally mixed via dtype_b) route to the
    k32 f32-accumulator family (32 x 8-bit A values per thread).
    """
    _wgmma_check_dtype_b_family(dtype, dtype_b, "wgmma_mma_async_rs")
    if _wgmma_is_int_dtype(dtype):
        return codegen_wgmma_mma_async_rs_int(acc, a0, a1, a2, a3, desc_b, n, dtype=dtype, scale_a=scale_a,
                                              scale_b=scale_b, scale_d=scale_d, trans_b=trans_b)
    if _wgmma_is_fp8_dtype(dtype):
        return codegen_wgmma_mma_async_rs_fp8(acc, a0, a1, a2, a3, desc_b, n, dtype=dtype, dtype_b=dtype_b,
                                              scale_a=scale_a, scale_b=scale_b, scale_d=scale_d, trans_b=trans_b)
    builtin_name = "wgmma_mma_async_rs"
    ptx_t = _wgmma_ptx_dtype(dtype)
    ncols = _wgmma_check_ncols(n, builtin_name)
    sa = _wgmma_signed_scale(scale_a, "scale_a", builtin_name)
    sb = _wgmma_signed_scale(scale_b, "scale_b", builtin_name)
    tb = _wgmma_const_flag(trans_b, "trans_b", builtin_name)
    n_out = ncols // 2
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(n_out)])
    reg_block = _wgmma_reg_block(n_out)
    a_ref = (f"{{%{(n_out)},%{n_out + 1},%{n_out + 2},%{n_out + 3}}}"
             f",%{n_out + 4},p")

    if _wgmma_is_compile_time_int(scale_d):
        sd_imm = _wgmma_imm_flag(scale_d, "scale_d")
        helper = (f"wgmma_mma_rs_n{ncols}_{ptx_t}"
                  f"_sa{_wgmma_scale_token(sa)}sb{_wgmma_scale_token(sb)}"
                  f"d{sd_imm}tb{tb}_fn")
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}"
        pred_line = f'"{{\\n.reg .pred p;\\nsetp.ne.b32 p, {sd_imm}, 0;\\n"\n'
        sig_extra = ""
        in_operands = '"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b)'
    else:
        helper = (f"wgmma_mma_rs_n{ncols}_{ptx_t}"
                  f"_sa{_wgmma_scale_token(sa)}sb{_wgmma_scale_token(sb)}"
                  f"drtb{tb}_fn")
        call_args = f"{acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}, {scale_d}"
        pred_line = ('"{\\n"\n        ".reg .pred p;\\n"\n'
                     f'        "setp.ne.b32 p, %{n_out + 5}, 0;\\n"\n')
        sig_extra = ", int scale_d"
        in_operands = ('"r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), '
                       '"r"(scale_d)')

    body = (f"\n__device__ __forceinline__ void {helper}("
            f"float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, "
            f"uint64_t desc_b{sig_extra}) {{\n"
            "    asm volatile(\n"
            f"        {pred_line}"
            '        "wgmma.mma_async.sync.aligned.'
            f'm64n{ncols}k16.f32.{ptx_t}.{ptx_t}\\n"\n'
            f'        "{{{reg_block}}},"\n'
            f'        "{a_ref},{sa},{sb},{tb};\\n}}\\n"\n'
            f"        : {out_constraints}\n"
            f"        : {in_operands});\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"{helper}({call_args})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_mma_async_rs)
def wgmma_mma_async_rs(acc, a0, a1, a2, a3, desc_b, n, dtype="bf16", scale_a=1, scale_b=1, scale_d=1, trans_b=0,
                       dtype_b=None):
    """GenericN WGMMA m64nNk16 RS (register-shared) instruction.

    A operand from registers (e.g. a preceding wgmma accumulator repacked in
    place for a fused-attention P@V stage), B from shared memory, for any
    PTX-legal tile N.

    Args:
        acc: float accumulator array (n//2 elements per thread), or int32 for
            integer operand dtypes
        a0, a1, a2, a3: A matrix in registers (4 x uint32_t)
        desc_b: descriptor for B matrix in shared memory
        n: tile N, a compile-time multiple of 8 in [8, 256]
        dtype: operand element type, "bf16" or "f16" (default "bf16"); "s8"
            or "u8" select the integer k32 -> s32 family (A tile is 4 x
            uint32_t = 16 x 8-bit values), "e4m3"/"e5m2" select the FP8
            k32 -> f32 family (A tile is 4 x uint32_t = 32 x 8-bit values)
        scale_a: +1 or -1 operand-scale immediate for A (default +1; must be
            +1 for integer and FP8 dtypes)
        scale_b: +1 or -1 operand-scale immediate for B (default +1; must be
            +1 for integer and FP8 dtypes)
        scale_d: 1 to accumulate, 0 to overwrite (default 1); a traced runtime
            int value is also accepted and passed as a register predicate
        trans_b: 1 to read B column-wise from shared memory (default 0; must
            be 0 for integer and FP8 dtypes)
        dtype_b: B-operand element type override (default None = same as
            dtype); only meaningful for FP8 operand families, where the two
            operand types may differ
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# RS A-fragment pack from FP32 accumulators (cvt.rn.{bf16,f16}x2.f32)
# ==============================================================================
#
# The register-A path of wgmma_mma_async_rs expects four uint32 words per
# thread per k16 step, each word holding two packed 16-bit operands. In a
# fused-attention epilogue the P tile lives in the 32 FP32 accumulator
# registers of the preceding Q@K^T wgmma, so the typed IR needs a first-class
# way to convert those accumulator values in registers (no shared-memory
# round trip) into the A-fragment words wgmma_mma_async_rs consumes:
#   fragment word j of k-step ks <- {hi = cvt(acc[8*ks + 2*j + 1]),
#                                    lo = cvt(acc[8*ks + 2*j])}
# with cvt.rn.bf16x2.f32 (or cvt.rn.f16x2.f32) rounding to nearest even,
# bit-identical to the hand-written CUDA idiom that feeds the RS descriptor
# op in flash-attention-style kernels. Two spellings of the same semantics:
# a scalar two-value pair pack (cvt_bf16x2_f32_u32 / cvt_f16x2_f32_u32) for
# frontends that unroll per-lane value flow themselves, and a fragment-level
# builtin (wgmma_pack_rs_frag) that consumes a float accumulator local and a
# [4]uint32 fragment local with the documented lane/register pairing.


def codegen_cvt_bf16x2_f32_u32(lo, hi):
    # cvt.rn.bf16x2.f32 d, a, b writes cvt(a) into the HIGH bf16 lane and
    # cvt(b) into the LOW lane, so the (lo, hi) spelling maps to operands
    # (%2, %1): d.lo = cvt(lo), d.hi = cvt(hi). One PTX instruction, one
    # rounding boundary (round-to-nearest-even on both lanes).
    body = ("\n__device__ __forceinline__ uint32_t lk_cvt_bf16x2_f32_u32_fn("
            "float lo, float hi) {\n"
            "    uint32_t d;\n"
            '    asm("cvt.rn.bf16x2.f32 %0, %2, %1;"\n'
            '        : "=r"(d) : "f"(lo), "f"(hi));\n'
            "    return d;\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"lk_cvt_bf16x2_f32_u32_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvt_bf16x2_f32_u32)
def cvt_bf16x2_f32_u32(lo, hi):
    """Pack two float32 values into one bf16x2 uint32 word via PTX
    ``cvt.rn.bf16x2.f32`` (round-to-nearest-even, both lanes).

    Emits the literal single instruction (not the ``__float22bfloat162_rn``
    intrinsic of bf16x2_pack_u32) so the emitted PTX is the exact CUDA idiom
    that sm90 RS WGMMA A-fragment recipes name; operand order is
    d.lo = cvt(lo), d.hi = cvt(hi).
    """
    raise RuntimeError("should not call cvt_bf16x2_f32_u32 in compilation")


def codegen_cvt_f16x2_f32_u32(lo, hi):
    # f16 twin of cvt_bf16x2_f32_u32: cvt.rn.f16x2.f32 d, a, b with
    # d.lo = cvt(lo), d.hi = cvt(hi).
    body = ("\n__device__ __forceinline__ uint32_t lk_cvt_f16x2_f32_u32_fn("
            "float lo, float hi) {\n"
            "    uint32_t d;\n"
            '    asm("cvt.rn.f16x2.f32 %0, %2, %1;"\n'
            '        : "=r"(d) : "f"(lo), "f"(hi));\n'
            "    return d;\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"lk_cvt_f16x2_f32_u32_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvt_f16x2_f32_u32)
def cvt_f16x2_f32_u32(lo, hi):
    """Pack two float32 values into one f16x2 uint32 word via PTX
    ``cvt.rn.f16x2.f32`` (round-to-nearest-even, both lanes).

    Mirror of cvt_bf16x2_f32_u32 for the f16 RS WGMMA operand family;
    d.lo = cvt(lo), d.hi = cvt(hi).
    """
    raise RuntimeError("should not call cvt_f16x2_f32_u32 in compilation")


def codegen_wgmma_pack_rs_frag(frag, acc, ks, dtype="bf16"):
    """Emit the 4-instruction ``cvt.rn.{bf16,f16}x2.f32`` A-fragment pack.

    frag is a [4]uint32 local array (the A fragment operand of one k16 step
    of wgmma_m64n64k16_rs / wgmma_mma_async_rs), acc the 32-element float
    accumulator local of the producing wgmma, ks the k-step index (compile
    time or runtime). For j in 0..3 emits

        cvt.rn.<t>x2.f32  frag[j], acc[8*ks+2*j+1], acc[8*ks+2*j]

    matching the documented RS lane/register pairing
    frag[j] = {hi = acc[8*ks+2*j+1], lo = acc[8*ks+2*j]}. The pack is a pure
    register-to-register conversion: no shared-memory traffic and no
    accumulator copy, so the same registers feeding a softmax update feed the
    PV wgmma, bit-identical to the hand-written idiom.
    """
    ptx_t = _wgmma_ptx_dtype(dtype)
    cvt_pairs = []
    for j in range(4):
        lo = f"acc[8 * ks + {2 * j}]"
        hi = f"acc[8 * ks + {2 * j + 1}]"
        cvt_pairs.append(f'    asm("cvt.rn.{ptx_t}x2.f32 %0, %2, %1;"\n'
                         f'        : "=r"(frag[{j}]) : "f"({lo}), "f"({hi}));')
    cvt_block = "\n".join(cvt_pairs)
    body = (f"\n__device__ __forceinline__ void wgmma_pack_rs_frag_{ptx_t}_fn("
            "uint32_t* frag, const float* acc, int ks) {\n"
            f"{cvt_block}\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_pack_rs_frag_{ptx_t}_fn({frag}, {acc}, {ks})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_pack_rs_frag)
def wgmma_pack_rs_frag(frag, acc, ks, dtype="bf16"):
    """Build the 4-uint32 register A-fragment set of one sm90 RS WGMMA k16
    step directly from FP32 accumulator values in registers.

    Applies ``cvt.rn.bf16x2.f32`` (or ``cvt.rn.f16x2.f32``) pairwise packing
    with the documented lane/register pairing: fragment word j of k-step ks
    receives {hi = acc[8*ks + 2*j + 1], lo = acc[8*ks + 2*j]}. This is the
    typed-IR equivalent of the flash-attention epilogue idiom that reuses the
    Q@K^T softmax accumulator registers as the P operand of the PV wgmma,
    avoiding the shared-memory round trip (extra traffic plus barriers) of
    the SS route.

    Args:
        frag: [4]uint32 local array; on return holds the four A-fragment
            words a0..a3 for wgmma_m64n64k16_rs / wgmma_mma_async_rs.
        acc: float32 accumulator local array with at least 8*(ks+1) elements
            (32 elements per thread cover four k16 steps of a 16x64 tile).
        ks: k-step index selecting the 8-accumulator window (compile-time or
            runtime integer).
        dtype: operand element type, "bf16" or "f16" (default "bf16"); must
            match the dtype of the consuming RS wgmma.

    Bit-identical to the CUDA idiom ``asm cvt.rn.bf16x2.f32`` feeding the RS
    descriptor op: one PTX conversion instruction per fragment word, one
    rounding boundary, no moves of the accumulator values.
    """
    raise RuntimeError("should not call wgmma_pack_rs_frag in compilation")


# ==============================================================================
# WGMMA fence operand (compiler optimization barrier for accumulators)
# ==============================================================================


def _acc_helper_suffix(acc_cpp):
    """Helper-name suffix for an accumulator C++ type.

    The historical accumulator helpers are named ``*_fn`` with float
    signatures; keep those names for float so existing float kernels emit
    byte-identical text, and suffix non-float dtypes so an int32 variant
    never collides with (or redefines) the float helper in the emitter's
    per-name builtin cache.
    """
    t = str(acc_cpp)
    if t in ("float", "float32"):
        return ""
    return "_" + _acc_cpp_type_token(t)


def _acc_dtype_from_arg_dtypes(_arg_dtypes, index):
    """Resolve one accumulator element LLType from the emitter's opt-in
    ``_arg_dtypes`` channel, or None when the channel is unavailable (older
    emitter paths, direct codegen-func calls in tests: fall back to the
    historical float spelling)."""
    if not _arg_dtypes or len(_arg_dtypes) <= index:
        return None
    return _arg_dtypes[index]


def _acc_lltype_to_cpp(ll_type):
    """Map a scalar accumulator LLType to its C++ type string.

    Mirrors the scalar entries of CudaEmitter._lltype_to_cpp (repeated here
    because language.intrin cannot import the emitter without a circular
    import). Tensor/local-tensor wrappers are unwrapped to their element
    type first (``_arg_dtypes`` hands the declared expression type, which
    may be tensor[s32] for a local accumulator). Returns None for anything
    unrecognised so callers fall back to the historical float spelling.
    """
    if ll_type is None:
        return None
    seen = 0
    while getattr(ll_type, "kind", None) not in ("int", "float", "void"):
        for attr in ("element_type", "inner_type"):
            inner = getattr(ll_type, attr, None)
            if inner is not None and inner is not ll_type:
                ll_type = inner
                break
        else:
            return None
        seen += 1
        if seen > 8 or ll_type is None:
            return None
    kind = getattr(ll_type, "kind", None)
    if kind == "int":
        if getattr(ll_type, "special", None) == "bool":
            return "bool"
        sign_prefix = "u" if not ll_type.signed else ""
        return f"{sign_prefix}int{ll_type.bits}_t"
    if kind == "float":
        return {
            "fp8_e5m2": "__nv_fp8_e5m2",
            "fp8_e4m3": "__nv_fp8_e4m3",
            "bfloat16": "__nv_bfloat16",
            "float16": "__half",
            "float32": "float",
            "float64": "double",
        }.get(getattr(ll_type, "fmt", None))
    return None


def codegen_wgmma_fence_operand(reg, _arg_dtypes=None):
    acc_cpp = _acc_lltype_to_cpp(_acc_dtype_from_arg_dtypes(_arg_dtypes, 0)) or "float"
    constraint = _acc_asm_constraint(acc_cpp)
    suffix = _acc_helper_suffix(acc_cpp)
    body = f"""
__device__ __forceinline__ void wgmma_fence_operand{suffix}_fn({acc_cpp}& r) {{
    asm volatile("" : "{constraint}"(r) :: "memory");
}}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_operand{suffix}_fn({reg})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_operand)
def wgmma_fence_operand(reg):
    """Prevent compiler from optimizing away accumulator dependency."""
    raise RuntimeError("should not be called")


setattr(wgmma_fence_operand, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_wgmma_fence_operand_array(arr, count, _arg_dtypes=None):
    acc_cpp = _acc_lltype_to_cpp(_acc_dtype_from_arg_dtypes(_arg_dtypes, 0)) or "float"
    constraint = _acc_asm_constraint(acc_cpp)
    suffix = _acc_helper_suffix(acc_cpp)
    body = f"""
__device__ __forceinline__ void wgmma_fence_operand_array{suffix}_fn({acc_cpp}* a, int n) {{
    #pragma unroll
    for (int i = 0; i < n; i++) {{
        asm volatile("" : "{constraint}"(a[i]) :: "memory");
    }}
}}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_operand_array{suffix}_fn({arr}, {count})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_operand_array)
def wgmma_fence_operand_array(arr, count):
    """Fence all registers in an accumulator array (dtype-generic)."""
    raise RuntimeError("should not be called")


setattr(wgmma_fence_operand_array, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_wgmma_fence_acc(acc, num_elems, _arg_dtypes=None):
    acc_cpp = _acc_lltype_to_cpp(_acc_dtype_from_arg_dtypes(_arg_dtypes, 0)) or "float"
    constraint = _acc_asm_constraint(acc_cpp)
    suffix = _acc_helper_suffix(acc_cpp)
    body = f"""
__device__ __forceinline__ void wgmma_fence_acc{suffix}_fn({acc_cpp}* a, int n) {{
    #pragma unroll
    for (int i = 0; i < n; i++) {{
        asm volatile("" : "{constraint}"(a[i]) :: "memory");
    }}
}}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_acc{suffix}_fn({acc}, {num_elems})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_acc)
def wgmma_fence_acc(acc, num_elems):
    """Fence an accumulator array (dtype-generic: +r for int, +f for float)."""
    raise RuntimeError("should not be called")


setattr(wgmma_fence_acc, CODEGEN_NEEDS_TYPES_ATTR, True)


def codegen_wgmma_fence_u32_array(arr, count):
    body = """
__device__ __forceinline__ void wgmma_fence_u32_array_fn(uint32_t* a, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+r"(a[i]) :: "memory");
    }
}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_u32_array_fn({arr}, {count})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_u32_array)
def wgmma_fence_u32_array(arr, count):
    """Fence all registers in a uint32 array (for P matrix in RS WGMMA)."""
    raise RuntimeError("should not be called")


# ==============================================================================
# Store accumulator to global memory - Generic version
# ==============================================================================


def codegen_store_acc_to_global(C, acc, bm, bn, M, N, tid, num_regs, wgmma_n, _arg_dtypes=None):
    """Generate code to store WGMMA accumulator to global memory.

    This is a generic version that works for any WGMMA m64nNk16 instruction
    and any accumulator element dtype. The coordinate mapping follows the
    WGMMA register layout:
    - For m64n64k16: 32 registers, each 64-column chunk uses 32 regs
    - For m64n128k16: 64 registers, 2 chunks of 32 regs each
    - For m64n256k16: 128 registers, 4 chunks of 32 regs each

    The C and accumulator C++ element types come from the declared tensor
    dtypes (via the emitter's ``_arg_dtypes`` channel) so an s32 accumulator
    (integer MMA) stores bitwise through ``int32_t*`` with no float
    bitcast; when the channel is unavailable both fall back to float, which
    reproduces the historical float-only template byte-for-byte.

    Args:
        num_regs: Number of accumulator registers (32, 64, or 128)
        wgmma_n: N dimension of WGMMA output (64, 128, or 256)
    """
    acc_cpp = _acc_lltype_to_cpp(_acc_dtype_from_arg_dtypes(_arg_dtypes, 1)) or "float"
    c_cpp = _acc_lltype_to_cpp(_acc_dtype_from_arg_dtypes(_arg_dtypes, 0)) or acc_cpp
    suffix = _acc_helper_suffix(acc_cpp)
    body = f"""
// Generic WGMMA accumulator coordinate calculation
// Works for m64n64k16 (32 regs), m64n128k16 (64 regs), m64n256k16 (128 regs)
__device__ __forceinline__ void get_wgmma_coord_{wgmma_n}{suffix}_(int tid, int reg, int& row, int& col) {{
    // Each 64-column chunk uses 32 registers
    int chunk = reg / 32;
    int local_reg = reg % 32;

    // Thread decomposition within warpgroup (128 threads)
    int t0 = tid % 4;           // Quad position (0-3)
    int t1 = (tid / 4) % 8;     // Row within warp (0-7)
    int t2 = tid / 32;          // Warp index (0-3)

    // Register decomposition within 32-reg chunk
    int r0 = local_reg % 2;           // Column pair (0-1)
    int r1 = (local_reg / 2) % 2;     // Row pair (0-1)
    int r2 = local_reg / 4;           // 8-element group (0-7)

    // Linear index calculation
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64;
    col = chunk * 64 + (lin / 64);
}}

__device__ __forceinline__ void store_acc_global_{wgmma_n}{suffix}_fn(
    {c_cpp}* C, {acc_cpp}* acc, int bm, int bn, int M, int N, int tid) {{
    #pragma unroll
    for (int r = 0; r < {num_regs}; r++) {{
        int lm, ln;
        get_wgmma_coord_{wgmma_n}{suffix}_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = acc[r];
    }}
}}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"store_acc_global_{wgmma_n}{suffix}_fn({C}, {acc}, {bm}, {bn}, {M}, {N}, {tid})")


@builtin(eval_return_type=void, codegen_func=codegen_store_acc_to_global)
def store_acc_to_global(C, acc, bm, bn, M, N, tid, num_regs, wgmma_n):
    """Store WGMMA accumulator to global memory (dtype-generic version).

    Args:
        C: Global memory destination (element type follows the tensor dtype)
        acc: Accumulator array (element type follows the tensor dtype; an
            integer accumulator stores through an integer pointer, no float
            bitcast)
        bm, bn: Block offsets in M and N dimensions
        M, N: Matrix dimensions
        tid: Thread ID within warpgroup (0-127)
        num_regs: Number of accumulator registers (32, 64, or 128)
        wgmma_n: N dimension of WGMMA output (64, 128, or 256)
    """
    raise RuntimeError("should not be called")


setattr(store_acc_to_global, CODEGEN_NEEDS_TYPES_ATTR, True)

# ==============================================================================
# Multi-accumulator support for 2x2 tiling (128x128 with 4x m64n64k16)
# ==============================================================================


def codegen_wgmma_init_4acc(acc_00, acc_01, acc_10, acc_11, num_elems, dtype_cpp):
    zero = _ZERO_LITERAL.get(str(dtype_cpp), "0.0f")
    loop = (f"for(int _i=0;_i<{num_elems};_i++){{{acc_00}[_i]={zero};{acc_01}[_i]={zero};"
            f"{acc_10}[_i]={zero};{acc_11}[_i]={zero};}}")
    return Builtin(body="", includes=[], return_val=loop)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_init_4acc)
def wgmma_init_4acc(acc_00, acc_01, acc_10, acc_11, num_elems, dtype):
    """Zero 4 m64n64 accumulators (user declares via ll.empty)."""
    raise RuntimeError("should not be called")


def codegen_wgmma_compute_00(acc_00, da, db):
    return codegen_wgmma_m64n64k16(acc_00, da, db)


def codegen_wgmma_compute_01(acc_01, da, db):
    return codegen_wgmma_m64n64k16(acc_01, da, db)


def codegen_wgmma_compute_10(acc_10, da, db):
    return codegen_wgmma_m64n64k16(acc_10, da, db)


def codegen_wgmma_compute_11(acc_11, da, db):
    return codegen_wgmma_m64n64k16(acc_11, da, db)


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_compute_00)
def wgmma_compute_00(acc_00, desc_a, desc_b):
    """WGMMA m64n64k16 on acc_00."""
    raise RuntimeError("should not be called")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_compute_01)
def wgmma_compute_01(acc_01, desc_a, desc_b):
    """WGMMA m64n64k16 on acc_01."""
    raise RuntimeError("should not be called")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_compute_10)
def wgmma_compute_10(acc_10, desc_a, desc_b):
    """WGMMA m64n64k16 on acc_10."""
    raise RuntimeError("should not be called")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_compute_11)
def wgmma_compute_11(acc_11, desc_a, desc_b):
    """WGMMA m64n64k16 on acc_11."""
    raise RuntimeError("should not be called")


def codegen_wgmma_fence_4acc(acc_00, acc_01, acc_10, acc_11, num_elems):
    body = """
__device__ __forceinline__ void wgmma_fence_4acc_fn(float* a0, float* a1, float* a2, float* a3, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a0[i]), "+f"(a1[i]), "+f"(a2[i]), "+f"(a3[i]) :: "memory");
    }
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"wgmma_fence_4acc_fn({acc_00}, {acc_01}, {acc_10}, {acc_11}, {num_elems})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_4acc)
def wgmma_fence_4acc(acc_00, acc_01, acc_10, acc_11, num_elems):
    """Fence all 4 m64n64 accumulators."""
    raise RuntimeError("should not be called")


def codegen_store_4acc_to_global_f32(C, acc_00, acc_01, acc_10, acc_11, bm, bn, M, N, tid, tile_m=64, tile_n=64):
    """Store 4 WGMMA accumulators in 2x2 layout to global memory.
    
    Args:
        tile_m: M dimension of each tile (default 64)
        tile_n: N dimension of each tile (default 64)
    """
    num_regs = (tile_n // 64) * 32  # 32 regs per 64 columns

    body = f"""
// Store 4 WGMMA accumulators in 2x2 layout
// Each accumulator covers tile_m x tile_n elements
__device__ __forceinline__ void get_coord_4acc_(int tid, int reg, int& row, int& col) {{
    int chunk = reg / 32;
    int local_reg = reg % 32;
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = local_reg % 2, r1 = (local_reg / 2) % 2, r2 = local_reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64;
    col = chunk * 64 + (lin / 64);
}}

__device__ __forceinline__ void store_4acc_f32_fn(
    float* C, float* a00, float* a01, float* a10, float* a11,
    int bm, int bn, int M, int N, int tid) {{
    #pragma unroll
    for (int r = 0; r < {num_regs}; r++) {{
        int lm, ln;
        get_coord_4acc_(tid, r, lm, ln);
        if (bm + lm < M && bn + ln < N)
            C[(int64_t)(bm + lm) * N + bn + ln] = a00[r];
        if (bm + lm < M && bn + {tile_n} + ln < N)
            C[(int64_t)(bm + lm) * N + bn + {tile_n} + ln] = a01[r];
        if (bm + {tile_m} + lm < M && bn + ln < N)
            C[(int64_t)(bm + {tile_m} + lm) * N + bn + ln] = a10[r];
        if (bm + {tile_m} + lm < M && bn + {tile_n} + ln < N)
            C[(int64_t)(bm + {tile_m} + lm) * N + bn + {tile_n} + ln] = a11[r];
    }}
}}
"""
    return Builtin(
        body=body, includes=[],
        return_val=f"store_4acc_f32_fn({C}, {acc_00}, {acc_01}, {acc_10}, {acc_11}, {bm}, {bn}, {M}, {N}, {tid})")


@builtin(eval_return_type=void, codegen_func=codegen_store_4acc_to_global_f32)
def store_4acc_to_global_f32(C, acc_00, acc_01, acc_10, acc_11, bm, bn, M, N, tid, tile_m=64, tile_n=64):
    """Store 4 WGMMA accumulators (2x2 layout) to global float32.
    
    Args:
        C: Global memory destination (float*)
        acc_00, acc_01, acc_10, acc_11: Four accumulator arrays
        bm, bn: Block offsets in M and N dimensions
        M, N: Matrix dimensions
        tid: Thread ID within warpgroup (0-127)
        tile_m: M dimension of each tile (default 64)
        tile_n: N dimension of each tile (default 64)
    """
    raise RuntimeError("should not be called")


# ==============================================================================
# Store accumulator to SMEM as BF16 - Generic version
# ==============================================================================


def codegen_store_acc_to_smem_bf16(sC, acc, ltid, row_offset, num_regs, row_stride):
    """Generate code to store WGMMA accumulator to SMEM as BF16.
    
    This is a generic version that works for any WGMMA m64nNk16 instruction.
    
    Args:
        num_regs: Number of accumulator registers (32, 64, or 128)
        row_stride: Number of columns in SMEM row (e.g., 64, 128, or 256)
    """
    # Builtin codegen receives rendered C++ operands, including literal strings.
    # These two operands specialize the helper and must be compile-time integers.
    try:
        num_regs, row_stride = int(num_regs), int(row_stride)
    except (ValueError, TypeError) as error:
        raise ValueError("store_acc_to_smem_bf16 requires literal num_regs and row_stride") from error
    if num_regs not in (32, 64, 128) or row_stride < 2 * num_regs:
        raise ValueError("store_acc_to_smem_bf16 requires 32/64/128 registers and a row stride covering the tile")
    helper = f"store_acc_smem_bf16_{num_regs}_{row_stride}_fn"
    # Each iteration stores 4 floats = 2 bf16 pairs
    # For 128 regs: 32 iterations, for 64 regs: 16 iterations, for 32 regs: 8 iterations
    num_iters = num_regs // 4

    body = f"""
// Generic WGMMA accumulator to SMEM BF16 store
// Works for m64n64k16 (32 regs), m64n128k16 (64 regs), m64n256k16 (128 regs)
__device__ __forceinline__ void {helper}(
    __nv_bfloat16* sC, float* acc, int ltid, int row_offset) {{
    int warp = ltid >> 5;
    int lane_id = ltid & 31;
    int row0 = row_offset + warp * 16 + (lane_id >> 2);
    int row1 = row0 + 8;
    int col_base = (lane_id & 3) * 2;
    #pragma unroll
    for (int i = 0; i < {num_iters}; i++) {{
        int col = col_base + i * 8;
        sC[row0 * {row_stride} + col + 0] = __float2bfloat16(acc[i * 4 + 0]);
        sC[row0 * {row_stride} + col + 1] = __float2bfloat16(acc[i * 4 + 1]);
        sC[row1 * {row_stride} + col + 0] = __float2bfloat16(acc[i * 4 + 2]);
        sC[row1 * {row_stride} + col + 1] = __float2bfloat16(acc[i * 4 + 3]);
    }}
}}
"""
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"{helper}({sC}, {acc}, {ltid}, {row_offset})")


@builtin(eval_return_type=void, codegen_func=codegen_store_acc_to_smem_bf16)
def store_acc_to_smem_bf16(sC, acc, ltid, row_offset, num_regs, row_stride):
    """Store WGMMA accumulator to SMEM as BF16 row-major (generic version).
    
    Args:
        sC: Shared memory destination (__nv_bfloat16*)
        acc: Accumulator array (float*)
        ltid: Thread ID within warpgroup (0-127)
        row_offset: Row offset in SMEM
        num_regs: Number of accumulator registers (32, 64, or 128)
        row_stride: Number of columns in SMEM row (e.g., 64, 128, or 256)
    """
    raise RuntimeError("should not be called")


# ==============================================================================
# WGMMA m64n128k16 instruction (64 output registers) - for Flash Attention
# ==============================================================================


def codegen_wgmma_m64n128k16_ss(acc, desc_a, desc_b, scale_d):
    """Generate WGMMA m64n128k16 SS (shared-shared) inline asm (64 output registers)."""
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(64)])
    reg_lines = []
    for row_start in range(0, 64, 8):
        regs = ",".join([f"%{i}" for i in range(row_start, row_start + 8)])
        reg_lines.append(regs)
    reg_block = ",\\n".join(reg_lines)

    body = ("\n__device__ __forceinline__ void wgmma_m64n128k16_ss_fn("
            "float* c, uint64_t desc_a, uint64_t desc_b, int scale_d) {\n"
            "    asm volatile(\n"
            '        "{\\n"\n'
            '        ".reg .pred p;\\n"\n'
            '        "setp.ne.b32 p, %66, 0;\\n"\n'
            '        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\\n"\n'
            f'        "{{{reg_block}}},"\n'
            '        "%64,%65,p,1,1,0,0;\\n}\\n"\n'
            f"        : {out_constraints}\n"
            '        : "l"(desc_a), "l"(desc_b), "r"(scale_d));\n'
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"wgmma_m64n128k16_ss_fn({acc}, {desc_a}, {desc_b}, {scale_d})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n128k16_ss)
def wgmma_m64n128k16_ss(acc, desc_a, desc_b, scale_d):
    """WGMMA m64n128k16 SS bf16 instruction with 64-float accumulator.
    
    Args:
        acc: 64-element float accumulator array
        desc_a: Descriptor for A matrix in shared memory
        desc_b: Descriptor for B matrix in shared memory
        scale_d: 1 to accumulate, 0 to overwrite
    """
    raise RuntimeError("should not be called in compilation")


def codegen_wgmma_m64n128k16_rs(acc, a0, a1, a2, a3, desc_b, scale_d):
    """Generate WGMMA m64n128k16 RS (register-shared) inline asm (64 output registers).
    
    A operand is in registers (4 x uint32_t = 8 bf16 values per thread).
    B operand is in shared memory via descriptor.
    """
    out_constraints = ",".join([f'"+f"(c[{i}])' for i in range(64)])
    reg_lines = []
    for row_start in range(0, 64, 8):
        regs = ",".join([f"%{i}" for i in range(row_start, row_start + 8)])
        reg_lines.append(regs)
    reg_block = ",\\n".join(reg_lines)

    body = (
        "\n__device__ __forceinline__ void wgmma_m64n128k16_rs_fn("
        "float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, "
        "uint64_t desc_b, int scale_d) {\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred p;\\n"\n'
        '        "setp.ne.b32 p, %69, 0;\\n"\n'
        '        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\\n"\n'
        f'        "{{{reg_block}}},"\n'
        '        "{%64,%65,%66,%67},%68,p,1,1,1;\\n}\\n"\n'  # layoutB=1 for col-major
        f"        : {out_constraints}\n"
        '        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc_b), "r"(scale_d));\n'
        "}\n")
    return Builtin(body=body, includes=[],
                   return_val=f"wgmma_m64n128k16_rs_fn({acc}, {a0}, {a1}, {a2}, {a3}, {desc_b}, {scale_d})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_m64n128k16_rs)
def wgmma_m64n128k16_rs(acc, a0, a1, a2, a3, desc_b, scale_d):
    """WGMMA m64n128k16 RS bf16 instruction with 64-float accumulator.
    
    A operand from registers, B operand from shared memory.
    Used for P@V matmul in Flash Attention where P is already in registers.
    
    Args:
        acc: 64-element float accumulator array
        a0, a1, a2, a3: A matrix in registers (4 x uint32_t = 8 bf16 values)
        desc_b: Descriptor for B matrix in shared memory
        scale_d: 1 to accumulate, 0 to overwrite
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# WGMMA Descriptor Creation
# ==============================================================================


def codegen_make_desc(lo, hi):
    """Generate code to create a WGMMA descriptor from lo and hi parts using mov.b64."""
    body = """
__device__ __forceinline__ uint64_t make_desc_fn(uint32_t lo, uint32_t hi) {
    uint64_t desc;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(desc) : "r"(lo), "r"(hi));
    return desc;
}
"""
    return Builtin(body=body, includes=[], return_val=f"make_desc_fn({lo}, {hi})")


# SM90 WGMMA descriptor layout-type (swizzle) whitelist, bits [63:62] of the
# 64-bit shared-memory matrix descriptor. PTX defines exactly four modes:
#   0 = no swizzle (interleave none)
#   1 = 128-byte swizzle
#   2 = 64-byte swizzle
#   3 = 32-byte swizzle
# The field is a PTX descriptor immediate: selecting anything else is a hard
# compile error, never a silently demoted descriptor.
_WGMMA_DESC_LAYOUTS = {"none": 0, "swizzle_128b": 1, "swizzle_64b": 2, "swizzle_32b": 3}


def _wgmma_desc_layout_code(layout):
    """Normalize a descriptor layout selector to its PTX 2-bit code.

    Accepts a Python int 0..3, a string key of _WGMMA_DESC_LAYOUTS, or the
    decimal text of a code (e.g. ``"1"``) -- the form codegen hands over
    when the layout reaches it as a positional integer literal in the C++
    source text. Raises ValueError for anything else: the layout field is a
    fixed 2-bit PTX descriptor field, so legality is enforced by this
    whitelist, not strings.
    """
    if isinstance(layout, str):
        token = layout.strip().strip("'\"").lower()
        if token in _WGMMA_DESC_LAYOUTS:
            return _WGMMA_DESC_LAYOUTS[token]
        # Decimal text of a code ("0".."3"), as delivered by codegen for
        # positional int literals.
        if token in ("0", "1", "2", "3"):
            return int(token)
        raise ValueError(f"encode_wgmma_smem_desc: layout must be one of "
                         f"{sorted(_WGMMA_DESC_LAYOUTS)} or a code \"0\"..\"3\" "
                         f"(got {layout!r})")
    if isinstance(layout, bool) or not isinstance(layout, int):
        raise ValueError(f"encode_wgmma_smem_desc: layout must be a Python int 0..3 or "
                         f"one of {sorted(_WGMMA_DESC_LAYOUTS)} (got {layout!r}); the "
                         "layout field is a compile-time PTX descriptor immediate")
    if layout not in _WGMMA_DESC_LAYOUTS.values():
        raise ValueError(f"encode_wgmma_smem_desc: illegal descriptor layout code "
                         f"{layout}; PTX defines 0 (none), 1 (128B), 2 (64B), 3 (32B) "
                         "swizzle modes for SM90 WGMMA smem descriptors")
    return layout


def codegen_encode_wgmma_smem_desc(addr, lbo_bytes, sbo_bytes, layout=1):
    """Encode an SM90 WGMMA 64-bit shared-memory matrix descriptor.

    Bit layout (matches the PTX ISA matrix-descriptor format and the layout
    code validated above, baked in as a compile-time immediate so call sites
    with different swizzle modes emit distinct helpers):
      bits [13:0]  = (addr >> 4) & 0x3FFF    (smem addr, 16-byte units)
      bits [29:16] = (lbo_bytes >> 4) & 0x3FFF  (leading byte offset / 16)
      bits [45:32] = (sbo_bytes >> 4) & 0x3FFF  (stride byte offset / 16)
      bits [63:62] = layout (0 none, 1 128B, 2 64B, 3 32B swizzle)

    addr/lbo_bytes/sbo_bytes are uint32 byte quantities (runtime or compile
    time); the hardware consumes them in 16-byte units. layout is a
    compile-time whitelist-checked selector: int 0..3 or "none" /
    "swizzle_128b" / "swizzle_64b" / "swizzle_32b" (default 128B).
    """
    code = _wgmma_desc_layout_code(layout)
    helper = f"encode_wgmma_smem_desc_l{code}_fn"
    body = (f"""
__device__ __forceinline__ uint64_t {helper}(
    uint32_t addr, uint32_t lbo_bytes, uint32_t sbo_bytes) {{
    uint64_t d = 0;
    d |= (uint64_t)((addr >> 4) & 0x3FFF);
    d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
    d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
    d |= (uint64_t){code} << 62;   // layout/swizzle mode (compile-time)
    return d;
}}
""")
    return Builtin(body=body, includes=[], return_val=f"{helper}({addr}, {lbo_bytes}, {sbo_bytes})")


@builtin(eval_return_type=uint64, codegen_func=codegen_encode_wgmma_smem_desc)
def encode_wgmma_smem_desc(addr, lbo_bytes, sbo_bytes, layout=1):
    """Encode an SM90 WGMMA shared-memory matrix descriptor (typed).

    Args:
        addr: shared-memory byte address (uint32, e.g. from cvta or an
            smem-offset helper); hardware consumes it in 16-byte units
        lbo_bytes: leading byte offset in bytes (uint32); only live for
            MN-major operands, else 0
        sbo_bytes: stride byte offset in bytes (uint32); e.g. 1024 for an
            8x128B swizzle atom
        layout: swizzle selector, int 0..3 or "none" / "swizzle_128b" /
            "swizzle_64b" / "swizzle_32b" (compile-time; default 128B)

    Returns:
        64-bit WGMMA shared-memory descriptor for sm_90a wgmma.mma_async,
        composable with ll.make_desc-free typed draft code.
    """
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=uint64, codegen_func=codegen_make_desc)
def make_desc(lo, hi):
    """Create a WGMMA descriptor from low and high 32-bit parts.

    Uses mov.b64 PTX instruction to combine two uint32 into one uint64.
    This is the FA3 style optimization for descriptor creation.

    Args:
        lo: Low 32 bits of descriptor (address and leading_byte_offset)
        hi: High 32 bits of descriptor (stride and swizzle mode)

    Returns:
        64-bit WGMMA descriptor
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Typed warpgroup register fragments (ir.wgmma.acc_fragment)
#
# Motivation: the existing accumulator model declares ``float acc[N];`` via
# ll.empty and passes the *array address* into a ``float* c`` WGMMA helper
# whose asm constraints bind ``c[i]`` as ``"+f"`` lvalues. The moment the
# frontend also subscripts the array for an elementwise epilogue (online
# softmax statistics, causal masking), C semantics force the array to an
# addressable object — nvcc/ptxas demote it to LOCAL MEMORY and the whole
# MMA/epilogue path spills, orders of magnitude off the hand-written CUDA.
#
# The hand-written form instead keeps N independent float scalars per
# thread (``float c0, c1, ..., c63;``) and binds each scalar as a separate
# ``"+f"`` asm operand; element access between MMAs is then a pure register
# SSA value with no address taken anywhere.
#
# This section implements exactly that model as typed IR:
#   * ``ll.wgmma_acc_fragment_f32(name, num_elems)`` declares ``num_elems``
#     standalone float scalars ``<name>_0 .. <name>_{N-1}`` (zero-initialized)
#     and registers the fragment so codegen resolves ``<name>[i]`` to the
#     scalar ``<name>_<i>`` at compile-time-unrolled indices (see
#     codegen/visitors/expression_codegen.py::_wgmma_fragment_element).
#   * the existing typed SS family (``wgmma_m64n64k16_{f16,bf16}``,
#     ``wgmma_m64n128k16_{f16,bf16}``, ``wgmma_m64n256k16_{f16,bf16}``) is
#     FRAGMENT-AWARE: when its acc operand names a declared fragment, codegen
#     emits a helper taking ``float& c0, float& c1, ...`` with one ``"+f"``
#     constraint per element — the same operand shape the teacher's hand CUDA
#     uses — so the SAME fragment object flows through several MMA calls with
#     per-call scale_d immediates and interleaved elementwise reads/writes.
#     Legacy ``float acc[N]`` operands keep the original array path.
#   * ``wgmma_frag_pack_bf16x2(frag, i)`` packs element pair (2i, 2i+1) into
#     one bf16x2 uint32 word via cvt.rn.bf16x2.f32, reading the individual
#     fragment scalars in registers.
#   * ``wgmma_m64n64k16_accumulator_coordinate`` /
#     ``wgmma_m64n128k16_accumulator_coordinate`` expose the generic fragment
#     layout (row = 16*(t>>5) + 8*((i>>1)&1) + ((t>>2)&7) with
#     t = warpgroup tid, i = fragment index; col = (i>>2)*8 + 2*(t&3) + (i&1))
#     for n64/n128, extending the existing m64n256 helper.
#
# Legality: fragment element indices must be compile-time constants at
# codegen (a literal, a provably-static int local, or an ll.unroll(range(N))
# counter binding inside the unrolled copy). Anything else is a compile
# error, never a silent local-memory demotion.
# ==============================================================================

# Accepted fragment sizes: every PTX-legal m64nNk16 accumulator (N/2 floats
# per thread for N in {64, 128, 256}; the family the dependent flash-attention
# backward epilogue composes). The set is deliberately closed so a typo is a
# diagnosing compile error rather than a malformed asm operand list.
_WGMMA_ACC_FRAGMENT_ELEMS = (32, 64, 128)


def _wgmma_fragment_registry(emitter):
    """Codegen-side fragment registry {name: {"num_elems": int, "dtype": t}}.

    Lives in the emitter's compile ctx under __WGMMA_ACC_FRAGMENTS_CTX_KEY__;
    created on first declaration. Returns None when no emitter is active
    (defensive: a codegen function always runs inside emission_scope).
    """
    if emitter is None:
        return None
    from little_kernel.core.internal import __WGMMA_ACC_FRAGMENTS_CTX_KEY__
    ctx = getattr(emitter, "ctx", None)
    if ctx is None:
        return None
    return ctx.setdefault(__WGMMA_ACC_FRAGMENTS_CTX_KEY__, {})


def _wgmma_acc_fragment_eval_arg_type(ctx, name, num_elems):
    """Bind the fragment NAME in the typing scope as a float32 tensor.

    The binding lets the type inferencer accept element subscripts
    (``name[i]`` loads/stores yield float32) using exactly the same scope
    registration the legacy ``wgmma_alloc_init_accum`` declaration builtin
    uses for its array. Codegen NEVER treats the name as an array: the
    fragment map (written by the declaration's codegen, which always runs
    before any use) intercepts every subscript and MMA operand reference and
    lowers them to the per-element register scalars.
    """
    if isinstance(name, str):
        from little_kernel.core.type_system import Tensor, float32 as _f32
        existing = ctx.get(name)
        if existing is not None and existing != Tensor[_f32]:
            # Rebinding an existing typed name (e.g. another tensor shape or
            # dtype) as a fragment is almost certainly a name collision bug;
            # diagnose it rather than silently shadowing.
            raise TypeError(f"wgmma_acc_fragment_f32: name {name!r} already bound to "
                            f"{existing}; choose a fresh fragment identifier")
        ctx[name] = Tensor[_f32]


def codegen_wgmma_acc_fragment_f32(name, num_elems):
    """Declare a typed WGMMA accumulator register fragment.

    Emits ``float <name>_0 = 0.0f; ... float <name>_{N-1} = 0.0f;`` — N
    INDEPENDENT per-thread register scalars, never an array — and registers
    the fragment so subsequent ``<name>[i]`` subscripts (constant index) and
    the fragment-aware WGMMA builtins resolve against the individual scalars.
    num_elems arrives as a static int (or its source text) and must be one of
    _WGMMA_ACC_FRAGMENT_ELEMS so the fragment shapes match the PTX m64nNk16
    accumulator whitelist.
    """
    from little_kernel.codegen.policy import current_emitter
    try:
        count = int(num_elems)
    except (TypeError, ValueError):
        raise ValueError(f"wgmma_acc_fragment_f32: num_elems must be a compile-time "
                         f"integer, got {num_elems!r}")
    if count not in _WGMMA_ACC_FRAGMENT_ELEMS:
        raise ValueError(f"wgmma_acc_fragment_f32: num_elems must be one of "
                         f"{_WGMMA_ACC_FRAGMENT_ELEMS} (PTX m64nNk16 accumulator sizes), "
                         f"got {count}")
    # Codegen hands string literals through as their quoted source text.
    if isinstance(name, str):
        name = name.strip().strip("'\"")
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"wgmma_acc_fragment_f32: fragment name must be a C identifier "
                         f"string, got {name!r}")

    registry = _wgmma_fragment_registry(current_emitter())
    if registry is not None:
        prev = registry.get(name)
        if prev is not None and prev["num_elems"] != count:
            raise ValueError(f"wgmma_acc_fragment_f32: fragment {name!r} redeclared with "
                             f"a different size ({prev['num_elems']} vs {count})")
        from little_kernel.core.type_system import float32 as _f32
        registry[name] = {"num_elems": count, "dtype": _f32}

    decls = "\n".join(f"float {name}_{i} = 0.0f;" for i in range(count))
    # return_val is inserted as a statement; the declaration text IS the
    # statement (semicolon-terminated lines), so the builtin contributes no
    # trailing expression.
    return Builtin(body="", includes=[], return_val=decls)


@builtin(
    eval_return_type=void,
    eval_arg_type=_wgmma_acc_fragment_eval_arg_type,
    codegen_func=codegen_wgmma_acc_fragment_f32,
)
def wgmma_acc_fragment_f32(name, num_elems):
    """Declare a typed float32 WGMMA accumulator register fragment.

    Args:
        name: C identifier (string literal) for the fragment; element access
            in kernel source is ``<name>[i]`` with a compile-time-unrolled
            index (literal, provably-static int, or ll.unroll counter),
            lowered to the per-element register scalar ``<name>_<i>``.
        num_elems: accumulator elements per thread: 32 (m64n64k16),
            64 (m64n128k16) or 128 (m64n256k16).

    The fragment owns ``num_elems`` independent float scalars per thread;
    pass the SAME name as the acc operand of the existing fragment-aware
    ``wgmma_m64nNk16_{f16,bf16}`` builtins to accumulate in registers across
    calls with per-call scale_d gating, and read/write elements between MMAs
    for online-softmax statistics without any local-memory traffic. Dynamic
    (non-constant) element indices are a compile error.
    """
    raise RuntimeError("should not be called in compilation")


def _resolve_fragment_arg(frag, n_out, builtin_name):
    """Resolve a WGMMA builtin's accumulator operand to fragment scalars.

    ``frag`` is the fragment name (or its quoted source text) as visited by
    codegen. Returns the comma-separated per-element scalar argument list
    ``name_0, name_1, ...`` for the per-scalar-helper call. Raises ValueError
    when the name is not a registered fragment or its size does not match the
    instruction's accumulator.
    """
    from little_kernel.codegen.policy import current_emitter
    token = frag.strip().strip("'\"") if isinstance(frag, str) else str(frag)
    registry = _wgmma_fragment_registry(current_emitter())
    rec = registry.get(token) if registry else None
    if rec is None:
        raise ValueError(f"{builtin_name}: accumulator operand {frag!r} is not a typed "
                         f"WGMMA register fragment; declare it first with "
                         f"ll.wgmma_acc_fragment_f32({token!r}, {n_out}) before the "
                         f"MMA call")
    if rec["num_elems"] != n_out:
        raise ValueError(f"{builtin_name}: fragment {token!r} has {rec['num_elems']} "
                         f"elements but m64n{n_out * 2}k16 needs exactly {n_out}; declare "
                         f"the fragment with the matching size")
    return ", ".join(f"{token}_{i}" for i in range(n_out))


def _try_resolve_fragment_arg(acc, n_out, builtin_name):
    """Probe whether an accumulator operand names a register fragment.

    Returns the per-element scalar argument list (like _resolve_fragment_arg)
    when ``acc`` names a registered fragment; returns None when the operand is
    plainly not a fragment name (legacy array path stays available). A size
    mismatch between a fragment and the instruction is still a hard
    ValueError — never a silent operand-shape demotion.
    """
    from little_kernel.codegen.policy import current_emitter
    token = acc.strip().strip("'\"") if isinstance(acc, str) else str(acc)
    if not token.isidentifier():
        return None
    emitter = current_emitter()
    if emitter is None:
        return None
    ctx = getattr(emitter, "ctx", None) or {}
    from little_kernel.core.internal import __WGMMA_ACC_FRAGMENTS_CTX_KEY__
    registry = ctx.get(__WGMMA_ACC_FRAGMENTS_CTX_KEY__)
    if not registry or token not in registry:
        return None
    return _resolve_fragment_arg(token, n_out, builtin_name)


def codegen_wgmma_frag_pack_bf16x2(lo, hi):
    """Pack two fragment scalar values into a bf16x2 u32 word.

    This is the pair pack of the fragment epilogue written in the frontend as
    ``wgmma_frag_pack_bf16x2(frag[2*i], frag[2*i+1])``: the subscripts lower
    to individual register scalars before this codegen runs, so the two
    operand texts are pure SSA values and the emitted instruction is the
    single-instruction cvt.rn.bf16x2.f32 idiom (d.lo = cvt(lo), d.hi =
    cvt(hi)) — element-for-element identical to cvt_bf16x2_f32_u32, with a
    name that documents the register-fragment register-to-register repack.
    """
    body = ("\n__device__ __forceinline__ uint32_t lk_frag_pack_bf16x2_fn("
            "float lo, float hi) {\n"
            "    uint32_t d;\n"
            '    asm("cvt.rn.bf16x2.f32 %0, %2, %1;"\n'
            '        : "=r"(d) : "f"(lo), "f"(hi));\n'
            "    return d;\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"lk_frag_pack_bf16x2_fn({lo}, {hi})")


@builtin(eval_return_type=uint32, codegen_func=codegen_wgmma_frag_pack_bf16x2)
def wgmma_frag_pack_bf16x2(lo, hi):
    """Register-to-register bf16x2 pack of two WGMMA fragment elements.

    Typed spelling of ``cvt_bf16x2_f32_u32`` for the register-fragment
    repack step of the flash-attention backward epilogue: the frontend
    passes two fragment element scalars (e.g.
    ``wgmma_frag_pack_bf16x2(p[2*j], p[2*j+1])``) and receives them packed
    into one uint32 via PTX ``cvt.rn.bf16x2.f32`` (round-to-nearest-even,
    both lanes), ready for stmatrix restaging or an RS WGMMA A operand
    without a local/shared round trip.

    Args:
        lo: float value for the LOW bf16 lane (e.g. fragment element 2*i)
        hi: float value for the HIGH bf16 lane (e.g. fragment element 2*i+1)
    """
    raise RuntimeError("should not be called in compilation")


def wgmma_m64n64k16_accumulator_coordinate(warpgroup_tid, register):
    """Map one FP32 accumulator fragment element of m64n64k16 to (row, col).

    Same generic fragment layout as wgmma_m64n256k16_accumulator_coordinate
    specialized to the 32-element fragment: row = 16*(t>>5) + 8*((i>>1)&1) +
    ((t>>2)&7), col = (i>>2)*8 + 2*(t&3) + (i&1) with t = warpgroup tid in
    [0, 128) and i = fragment index in [0, 32).
    """
    if (not isinstance(warpgroup_tid, int) or isinstance(warpgroup_tid, bool) or not 0 <= warpgroup_tid < 128):
        raise ValueError(f"warpgroup_tid must be in [0, 128), got {warpgroup_tid!r}")
    if (not isinstance(register, int) or isinstance(register, bool) or not 0 <= register < 32):
        raise ValueError(f"m64n64k16 register must be in [0, 32), got {register!r}")
    warp = warpgroup_tid // 32
    lane = warpgroup_tid % 32
    row = 16 * warp + 8 * ((register // 2) % 2) + (lane // 4) % 8
    column = (register // 4) * 8 + 2 * (lane % 4) + register % 2
    return row, column


def wgmma_m64n128k16_accumulator_coordinate(warpgroup_tid, register):
    """Map one FP32 accumulator fragment element of m64n128k16 to (row, col).

    Generic fragment layout specialized to the 64-element fragment:
    row = 16*(t>>5) + 8*((i>>1)&1) + ((t>>2)&7), col = (i>>2)*8 + 2*(t&3) +
    (i&1) with t = warpgroup tid in [0, 128) and i = fragment index in
    [0, 64).
    """
    if (not isinstance(warpgroup_tid, int) or isinstance(warpgroup_tid, bool) or not 0 <= warpgroup_tid < 128):
        raise ValueError(f"warpgroup_tid must be in [0, 128), got {warpgroup_tid!r}")
    if (not isinstance(register, int) or isinstance(register, bool) or not 0 <= register < 64):
        raise ValueError(f"m64n128k16 register must be in [0, 64), got {register!r}")
    warp = warpgroup_tid // 32
    lane = warpgroup_tid % 32
    row = 16 * warp + 8 * ((register // 2) % 2) + (lane // 4) % 8
    column = (register // 4) * 8 + 2 * (lane % 4) + register % 2
    return row, column


# ==============================================================================
# Fragment-native stmatrix staging into a 128B-swizzled bf16 smem tile
# ==============================================================================
#
# The existing store_accum_swizzle epilogue takes
# the accumulator as a whole-array ``float*`` operand — correct for
# ll.empty-local accumulators, but unusable for a typed register fragment
# (passing the fragment would require an addressable array, the exact
# local-memory demotion this family eliminates). ``smem_st_frag_swizzle_128b``
# is the fragment twin: it takes the fragment NAME plus a raw byte-offset
# smem tile pointer and emits the same proven 16B-chunk-permuted
# (``chunk ^ (row & 7)``) STMATRIX.x2 mapping, reading element scalars
# ``<name>_<i>`` as individual register SSA values in the fully-unrolled
# pack loop. The destination layout follows the same chunk-XOR permutation
# as store_accum_swizzle, so a tile staged this way is directly consumable
# by a subsequent WGMMA SS smem descriptor (dS@K, dS^T@Q, P^T@dO in the
# flash-attention backward epilogue) or by a swizzle_bytes=128 TMA store.
#
# Emitted mapping (the store_accum_swizzle mapping, re-specialized):
#   rows_per_warp = 16, warps_per_warpgroup = 4, num_rows = 64,
#   row_size (bf16 elems) = num_elems (i.e. 64 elems -> 64-col tile with
#   128B rows, or 128 elems -> 256-col tile with 512B rows; both are whole
#   128B-swizzle atoms per 8-row group).
#   per iteration i (4 floats = 16B stmatrix.x2 target):
#       atom_offset    = i / ACC_REGISTERS_PER_WARP
#       in_atom_offset = i % ACC_REGISTERS_PER_WARP
#       row = (in_atom_offset / 8) + lane_idx            (within warp group)
#       col = in_atom_offset
#       col ^= row % (SWIZZLE_D_MODE / 16)               (TMA 128B swizzle)
#       addr = smem + byte_offset
#              + warp_idx*(16*SWIZZLE_D_MODE)
#              + atom_offset*num_rows*SWIZZLE_D_MODE
#              + row*128 + col*16
# ==============================================================================


def codegen_smem_st_frag_swizzle_128b(frag, smem_byte_ptr, byte_offset, warp_idx, lane_idx):
    """stmatrix staging of a typed register fragment into a swizzled tile.

    Resolves the fragment against the codegen registry; its element count
    pins the tile geometry (64 coeffs -> 64x128-elem bf16 tile at 16 rows
    per warp; 128 coeffs -> 64x256). The pack loop is unrolled AT CODEGEN so
    each ``floats2bfloat162`` reads the individual fragment scalars
    ``<name>_<4i+t>`` as register SSA values — the same operand shape the
    teacher's hand-CUDA two-box restage uses.
    """
    token = frag.strip().strip("'\"") if isinstance(frag, str) else str(frag)
    from little_kernel.codegen.policy import current_emitter
    registry = _wgmma_fragment_registry(current_emitter())
    rec = registry.get(token) if registry else None
    if rec is None:
        raise ValueError(f"smem_st_frag_swizzle_128b: {frag!r} is not a declared WGMMA "
                         f"register fragment (declare with ll.wgmma_acc_fragment_f32)")

    num_elems = rec["num_elems"]
    floats_per_iter = 4
    stsm_size_bytes = 16
    warps_per_warpgroup = 4
    num_rows = 64
    rows_per_warp = 16
    swizzle_base = 8

    num_iter = num_elems // floats_per_iter
    acc_regs_per_warp = num_iter // warps_per_warpgroup
    # SWIZZLE_D_MODE is the byte size of ONE 128B-swizzle core-matrix box row
    # (64 bf16 columns * 2B == 128B), matching the reference
    # smem_st_acc_swizzle_like store_accum_swizzle computation
    # (row_size/4)*sizeof(elt) == 128
    # for its BN=256 bf16 tiles. The stmatrix/stsm x2 mapping only ever
    # addresses WITHIN one box: advancing to the next box is handled by the
    # atom_offset * (num_rows * 128) byte term below, so boxes stack row-major
    # (the "two-box layout" of a 64x128 bf16 tile = 2 side-by-side boxes per
    # 8-row group).
    swizzle_d_mode = 128

    # Fully unrolled staging: one stmatrix.x2 per 4-float group, with the
    # fragment scalars interpolated directly as conversion operands. Every
    # address term except lane_idx is compile-time specialization of the
    # documented mapping (atom_offset and in_atom_offset are Python ints
    # baked into the text; warp_idx contributes only a base byte offset).
    iters = []
    for i in range(num_iter):
        atom_offset = i // acc_regs_per_warp
        in_atom_offset = i % acc_regs_per_warp
        atom_byte_term = atom_offset * num_rows * swizzle_d_mode
        row_div = in_atom_offset // swizzle_base
        iters.append("    {\n"
                     f"        int row_ = {row_div} + lane_idx_;\n"
                     f"        int col_ = {in_atom_offset};\n"
                     f"        col_ ^= row_ % {swizzle_d_mode // stsm_size_bytes};\n"
                     "        uint8_t* p_ = base_"
                     f" + {atom_byte_term}"
                     f" + row_ * {stsm_size_bytes * 8}"
                     f" + col_ * {stsm_size_bytes};\n"
                     f"        __nv_bfloat162 v0_ = __floats2bfloat162_rn("
                     f"c{i * floats_per_iter + 0}, "
                     f"c{i * floats_per_iter + 1});\n"
                     f"        __nv_bfloat162 v1_ = __floats2bfloat162_rn("
                     f"c{i * floats_per_iter + 2}, "
                     f"c{i * floats_per_iter + 3});\n"
                     f"        stsm_x2_frag_n{num_elems}_(v0_, v1_, p_);\n"
                     "    }")
    body_iters = "\n".join(iters)

    # The inner stmatrix helper carries the same size suffix as the outer
    # one: writeln_builtin extracts every function name from a builtin body
    # and skips the whole body only when ALL names are already emitted, so a
    # shared fixed name here would be re-emitted alongside a differently
    # sized outer helper and redefine the function in the same TU.
    body = (
        f"\n__device__ __forceinline__ void stsm_x2_frag_n{num_elems}_("
        "__nv_bfloat162 v0, __nv_bfloat162 v1, void* p) {\n"
        "    uint32_t s0=*reinterpret_cast<uint32_t*>(&v0);\n"
        "    uint32_t s1=*reinterpret_cast<uint32_t*>(&v1);\n"
        '    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16'
        ' [%0], {%1, %2};\\n"\n'
        '        :: "r"((uint32_t)__cvta_generic_to_shared(p)),'
        ' "r"(s0), "r"(s1));\n'
        "}\n"
        "\n"
        "// Fragment-native stmatrix restage: fp32 register-scalar fragment\n"
        "// -> bf16 128B-swizzled smem tile, same (chunk ^ (row & 7))\n"
        "// permutation as store_accum_swizzle's array path.\n"
        # The helper name embeds the fragment size: writeln_builtin dedups
        # builtin bodies by function NAME, and the signature arity is 1:1
        # with num_elems, so two differently-sized fragments staged from one
        # kernel must emit two differently-named definitions (same scheme as
        # the m64n{N}k16 WGMMA _frag_fn helpers).
        f"__device__ __forceinline__ void smem_st_frag_swizzle_128b_n{num_elems}_fn(\n"
        f"    void* smem_, int byte_offset_, int warp_idx_, int lane_idx_,\n"
        f"    {', '.join(f'float c{i}' for i in range(num_elems))}) {{\n"
        f"    constexpr int SWIZZLE_D_MODE_ = {swizzle_d_mode};\n"
        f"    static_assert(SWIZZLE_D_MODE_ % 128 == 0,\n"
        '        "smem_st_frag_swizzle_128b: row bytes must be 128B-swizzle multiples");\n'
        f"    uint8_t* base_ = reinterpret_cast<uint8_t*>(smem_) + byte_offset_\n"
        f"        + warp_idx_ * ({rows_per_warp} * SWIZZLE_D_MODE_);\n"
        f"{body_iters}\n"
        "}\n")
    # Fragment scalars are passed BY VALUE as conversion operands only: the
    # helper never mutates them, so the caller's fragment registers are not
    # constrained (no read-write asm operand needed for a pure consumer).
    frag_args = ", ".join(f"{token}_{i}" for i in range(num_elems))
    return Builtin(
        body=body, includes=["<cuda_bf16.h>"], return_val=f"smem_st_frag_swizzle_128b_n{num_elems}_fn("
        f"{smem_byte_ptr}, "
        f"{byte_offset}, {warp_idx}, {lane_idx}, {frag_args})")


@builtin(eval_return_type=void, codegen_func=codegen_smem_st_frag_swizzle_128b)
def smem_st_frag_swizzle_128b(frag, smem_byte_ptr, byte_offset, warp_idx, lane_idx):
    """stmatrix-restage a typed WGMMA register fragment into a 128B-swizzled
    bf16 smem tile.

    The fragment-native twin of the ``store_accum_swizzle`` array-path
    staging: the same proven 16B-chunk-permuted (``chunk ^ (row & 7)``)
    shared-store layout, but the source is a typed register fragment
    (per-element register SSA values), so restaging a softmax-rescaled score
    fragment for consumption as the A operand of a subsequent WGMMA (dS@K,
    dS^T@Q, P^T@dO) does not route the fragment through local memory. The
    emitted tile layout follows the same chunk-XOR permutation as the array
    form and composes with WGMMA smem descriptors and swizzle_bytes=128 TMA
    stores.

    Args:
        frag: fragment name (identifier declared via wgmma_acc_fragment_f32)
        smem_byte_ptr: dynamic-shared base pointer / byte arena the paired
            descriptor's shared window is anchored at (1024B aligned)
        byte_offset: byte offset of this tile inside the arena; multiple of
            1024 (whole 8-row x 128B swizzle atoms)
        warp_idx: warp index within the warpgroup (runtime)
        lane_idx: lane index within the warp (runtime)
    """
    raise RuntimeError("should not be called in compilation")
