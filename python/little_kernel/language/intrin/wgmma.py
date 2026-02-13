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

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import void

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


# ==============================================================================
# WGMMA m64n256k16 instruction
# ==============================================================================


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


# ==============================================================================
# High-level WGMMA Accumulator Ops
# Accumulator is declared by user via ll.empty(..., scope="local"); these ops
# take (acc, num_elems, dtype) or (acc, ...) so generated code is parameterized.
# ==============================================================================

# Map C++ scalar type (from _lltype_to_cpp) to zero literal for init/zero loops
_ZERO_LITERAL = {"float": "0.0f", "float32": "0.0f", "double": "0.0", "float64": "0.0"}


def _build_wgmma_fn_body():
    oc = ",".join([f'"+f"(c[{i}])' for i in range(128)])
    parts = []
    for s in range(0, 128, 16):
        parts.append(",".join([f"%{i}" for i in range(s, s + 16)]))
    rb = ",".join(parts)
    return ("\n__device__ __forceinline__ void wgmma_m64n256k16_fn("
            "float* c, uint64_t da, uint64_t db) {\n"
            "    asm volatile(\n"
            '        "{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            '        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\\n"\n'
            f'        "{{{rb}}},"\n'
            '        "%128,%129,p,1,1,0,0;\\n}\\n"\n'
            f"        : {oc}\n"
            '        : "l"(da), "l"(db));\n'
            "}\n")


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


def codegen_wgmma_compute(acc, desc_a, desc_b):
    return Builtin(body=_build_wgmma_fn_body(), includes=[],
                   return_val=f"wgmma_m64n256k16_fn({acc}, {desc_a}, {desc_b})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_compute)
def wgmma_compute(acc, desc_a, desc_b):
    """Execute WGMMA m64n256k16 on the given accumulator."""
    raise RuntimeError("should not be called")


def codegen_store_accum_swizzle(acc, sD, warp_idx, lane_idx, m_offset):
    body = ("\n__device__ __forceinline__ void stsm_x2_fn_("
            "__nv_bfloat162 v0, __nv_bfloat162 v1, void* p) {\n"
            "    uint32_t s0=*reinterpret_cast<uint32_t*>(&v0);\n"
            "    uint32_t s1=*reinterpret_cast<uint32_t*>(&v1);\n"
            '    asm volatile("stmatrix.sync.aligned.x2.m8n8.shared.b16'
            ' [%0], {%1, %2};\\n"\n'
            '        :: "r"((uint32_t)__cvta_generic_to_shared(p)),'
            ' "r"(s0), "r"(s1));\n'
            "}\n"
            "\n__device__ __forceinline__ void store_accum_swizzle_fn(\n"
            "    __nv_bfloat16* sD, float* ac, int wi, int li, int mo) {\n"
            "    #pragma unroll\n"
            "    for (int i=0;i<32;i++) {\n"
            "        int ao=i/8, iao=i%8;\n"
            "        int row=iao/8+li, col=iao;\n"
            "        col^=row%8;\n"
            "        uint8_t* sp=reinterpret_cast<uint8_t*>(sD)+\n"
            "            wi*(16*128)+mo*128+ao*128*128+row*128+col*16;\n"
            "        __nv_bfloat162 v0=__floats2bfloat162_rn(ac[i*4],ac[i*4+1]);\n"
            "        __nv_bfloat162 v1=__floats2bfloat162_rn(ac[i*4+2],ac[i*4+3]);\n"
            "        stsm_x2_fn_(v0,v1,sp);\n"
            "    }\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"], return_val=f"store_accum_swizzle_fn({sD}, {acc}, "
                   f"{warp_idx}, {lane_idx}, {m_offset})")


@builtin(eval_return_type=void, codegen_func=codegen_store_accum_swizzle)
def store_accum_swizzle(acc, sD, warp_idx, lane_idx, m_offset):
    """Store WGMMA accumulator to SMEM with 128B swizzle."""
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
# High-level m64n64 Accumulator Ops (parameterized acc, num_elems, dtype)
# ==============================================================================


def _build_wgmma_64x64_fn_body():
    oc = ",".join([f'"+f"(c[{i}])' for i in range(32)])
    regs = ",".join([f"%{i}" for i in range(32)])
    return ("\n__device__ __forceinline__ void wgmma_m64n64k16_fn_("
            "float* c, uint64_t da, uint64_t db) {\n"
            "    asm volatile(\n"
            '        "{\\n.reg .pred p;\\nsetp.ne.b32 p, 1, 0;\\n"\n'
            '        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\\n"\n'
            f'        "{{{regs}}},"\n'
            '        "%32,%33,p,1,1,0,0;\\n}\\n"\n'
            f"        : {oc}\n"
            '        : "l"(da), "l"(db));\n'
            "}\n")


def codegen_wgmma_init_accum_64x64(acc, num_elems, dtype_cpp):
    zero = _ZERO_LITERAL.get(str(dtype_cpp), "0.0f")
    return Builtin(body="", includes=[], return_val=f"for(int _i=0; _i<{num_elems}; _i++) {acc}[_i]={zero}")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_init_accum_64x64)
def wgmma_init_accum_64x64(acc, num_elems, dtype):
    """Zero the WGMMA m64n64 accumulator (user declares via ll.empty)."""
    raise RuntimeError("should not be called")


def codegen_wgmma_zero_accum_64x64(acc, num_elems, dtype_cpp):
    zero = _ZERO_LITERAL.get(str(dtype_cpp), "0.0f")
    return Builtin(body="", includes=[],
                   return_val=f"do {{}} while(0);\nfor(int _i=0; _i<{num_elems}; _i++) {acc}[_i]={zero}")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_zero_accum_64x64)
def wgmma_zero_accum_64x64(acc, num_elems, dtype):
    """Zero the WGMMA m64n64 accumulator."""
    raise RuntimeError("should not be called")


def codegen_wgmma_compute_64x64(acc, desc_a, desc_b):
    return Builtin(body=_build_wgmma_64x64_fn_body(), includes=[],
                   return_val=f"wgmma_m64n64k16_fn_({acc}, {desc_a}, {desc_b})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_compute_64x64)
def wgmma_compute_64x64(acc, desc_a, desc_b):
    """Execute WGMMA m64n64k16 on the given accumulator."""
    raise RuntimeError("should not be called")


# ==============================================================================
# WGMMA fence operand (compiler optimization barrier for accumulators)
# ==============================================================================


def codegen_wgmma_fence_operand(reg):
    body = """
__device__ __forceinline__ void wgmma_fence_operand_fn(float& r) {
    asm volatile("" : "+f"(r) :: "memory");
}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_operand_fn({reg})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_operand)
def wgmma_fence_operand(reg):
    """Prevent compiler from optimizing away accumulator dependency."""
    raise RuntimeError("should not be called")


def codegen_wgmma_fence_operand_array(arr, count):
    body = """
__device__ __forceinline__ void wgmma_fence_operand_array_fn(float* a, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_operand_array_fn({arr}, {count})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_operand_array)
def wgmma_fence_operand_array(arr, count):
    """Fence all registers in an accumulator array."""
    raise RuntimeError("should not be called")


def codegen_wgmma_fence_acc64(acc, num_elems):
    body = """
__device__ __forceinline__ void wgmma_fence_acc64_fn(float* a, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_acc64_fn({acc}, {num_elems})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_acc64)
def wgmma_fence_acc64(acc, num_elems):
    """Fence the m64n64 accumulator array."""
    raise RuntimeError("should not be called")


def codegen_wgmma_fence_acc(acc, num_elems):
    body = """
__device__ __forceinline__ void wgmma_fence_acc_fn(float* a, int n) {
    #pragma unroll
    for (int i = 0; i < n; i++) {
        asm volatile("" : "+f"(a[i]) :: "memory");
    }
}
"""
    return Builtin(body=body, includes=[], return_val=f"wgmma_fence_acc_fn({acc}, {num_elems})")


@builtin(eval_return_type=void, codegen_func=codegen_wgmma_fence_acc)
def wgmma_fence_acc(acc, num_elems):
    """Fence the m64n256 accumulator array."""
    raise RuntimeError("should not be called")


# ==============================================================================
# Store accumulator to global memory (m64n64 -> float32)
# ==============================================================================


def codegen_store_acc64_to_global_f32(C, acc, bm, bn, M, N, tid):
    body = """
__device__ __forceinline__ void get_coord_64x64_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}
__device__ __forceinline__ void store_acc64_global_f32_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_64x64_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"store_acc64_global_f32_fn({C}, {acc}, {bm}, {bn}, {M}, {N}, {tid})")


@builtin(eval_return_type=void, codegen_func=codegen_store_acc64_to_global_f32)
def store_acc64_to_global_f32(C, acc, bm, bn, M, N, tid):
    """Store m64n64 accumulator to global float32 memory."""
    raise RuntimeError("should not be called")


def codegen_store_acc_f32_to_global(C, acc, bm, bn, M, N, tid):
    body = """
__device__ __forceinline__ void get_coord_64x64_g_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}
__device__ __forceinline__ void store_acc_global_f32_fn(
    float* C, float* ac, int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_64x64_g_(tid, r, lm, ln);
        int gm = bm + lm, gn = bn + ln;
        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];
    }
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"store_acc_global_f32_fn({C}, {acc}, {bm}, {bn}, {M}, {N}, {tid})")


@builtin(eval_return_type=void, codegen_func=codegen_store_acc_f32_to_global)
def store_acc_f32_to_global(C, acc, bm, bn, M, N, tid):
    """Store any 32-float accumulator to global float32 memory (generic version)."""
    raise RuntimeError("should not be called")


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
    return Builtin(body=_build_wgmma_64x64_fn_body(), includes=[],
                   return_val=f"wgmma_m64n64k16_fn_({acc_00}, {da}, {db})")


def codegen_wgmma_compute_01(acc_01, da, db):
    return Builtin(body=_build_wgmma_64x64_fn_body(), includes=[],
                   return_val=f"wgmma_m64n64k16_fn_({acc_01}, {da}, {db})")


def codegen_wgmma_compute_10(acc_10, da, db):
    return Builtin(body=_build_wgmma_64x64_fn_body(), includes=[],
                   return_val=f"wgmma_m64n64k16_fn_({acc_10}, {da}, {db})")


def codegen_wgmma_compute_11(acc_11, da, db):
    return Builtin(body=_build_wgmma_64x64_fn_body(), includes=[],
                   return_val=f"wgmma_m64n64k16_fn_({acc_11}, {da}, {db})")


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


def codegen_store_4acc_to_global_f32(C, acc_00, acc_01, acc_10, acc_11, bm, bn, M, N, tid):
    body = """
__device__ __forceinline__ void get_coord_4ac_(int tid, int reg, int& row, int& col) {
    int t0 = tid % 4, t1 = (tid / 4) % 8, t2 = tid / 32;
    int r0 = reg % 2, r1 = (reg / 2) % 2, r2 = reg / 4;
    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;
    row = lin % 64; col = lin / 64;
}
__device__ __forceinline__ void store_4acc_f32_fn(
    float* C, float* a00, float* a01, float* a10, float* a11,
    int bm, int bn, int M, int N, int tid) {
    #pragma unroll
    for (int r = 0; r < 32; r++) {
        int lm, ln;
        get_coord_4ac_(tid, r, lm, ln);
        if (bm + lm < M && bn + ln < N)
            C[(int64_t)(bm + lm) * N + bn + ln] = a00[r];
        if (bm + lm < M && bn + 64 + ln < N)
            C[(int64_t)(bm + lm) * N + bn + 64 + ln] = a01[r];
        if (bm + 64 + lm < M && bn + ln < N)
            C[(int64_t)(bm + 64 + lm) * N + bn + ln] = a10[r];
        if (bm + 64 + lm < M && bn + 64 + ln < N)
            C[(int64_t)(bm + 64 + lm) * N + bn + 64 + ln] = a11[r];
    }
}
"""
    return Builtin(
        body=body, includes=[],
        return_val=f"store_4acc_f32_fn({C}, {acc_00}, {acc_01}, {acc_10}, {acc_11}, {bm}, {bn}, {M}, {N}, {tid})")


@builtin(eval_return_type=void, codegen_func=codegen_store_4acc_to_global_f32)
def store_4acc_to_global_f32(C, acc_00, acc_01, acc_10, acc_11, bm, bn, M, N, tid):
    """Store 4 m64n64 accumulators (2x2 layout) to global float32."""
    raise RuntimeError("should not be called")


# ==============================================================================
# Store m64n256 accumulator (128 regs) to global float32
# ==============================================================================


def codegen_store_acc_to_global_n256(C, acc, bm, bn, M, N, tid):
    body = ("\n__device__ __forceinline__ void get_coord_n256_fn("
            "int ltid, int r, int& row, int& col) {\n"
            "    int chunk = r / 32;\n"
            "    int local_reg = r % 32;\n"
            "    int t0 = ltid % 4, t1 = (ltid / 4) % 8, t2 = ltid / 32;\n"
            "    int r0 = local_reg % 2, r1 = (local_reg / 2) % 2, r2 = local_reg / 4;\n"
            "    int lin = t0 * 128 + t1 * 1 + t2 * 16 + r0 * 64 + r1 * 8 + r2 * 512;\n"
            "    row = lin % 64;\n"
            "    col = chunk * 64 + (lin / 64);\n"
            "}\n"
            "__device__ __forceinline__ void store_acc_global_n256_fn(\n"
            "    float* C, float* ac, int bm, int bn, int M, int N, int tid) {\n"
            "    #pragma unroll\n"
            "    for (int r = 0; r < 128; r++) {\n"
            "        int lm, ln;\n"
            "        get_coord_n256_fn(tid, r, lm, ln);\n"
            "        int gm = bm + lm, gn = bn + ln;\n"
            "        if (gm < M && gn < N) C[(int64_t)gm * N + gn] = ac[r];\n"
            "    }\n"
            "}\n")
    return Builtin(body=body, includes=[],
                   return_val=f"store_acc_global_n256_fn({C}, {acc}, {bm}, {bn}, {M}, {N}, {tid})")


@builtin(eval_return_type=void, codegen_func=codegen_store_acc_to_global_n256)
def store_acc_to_global_n256(C, acc, bm, bn, M, N, tid):
    """Store m64n256 accumulator to global float32."""
    raise RuntimeError("should not be called")


# ==============================================================================
# Store m64n256 accumulator to SMEM as BF16 (for TMA store epilogue)
# ==============================================================================


def codegen_store_acc_to_smem_bf16_n256(sC, acc, ltid, row_offset):
    body = ("\n__device__ __forceinline__ void store_acc_smem_bf16_n256_fn(\n"
            "    __nv_bfloat16* sC, float* ac, int ltid, int row_offset) {\n"
            "    int warp = ltid >> 5;\n"
            "    int lane_id = ltid & 31;\n"
            "    int row0 = row_offset + warp * 16 + (lane_id >> 2);\n"
            "    int row1 = row0 + 8;\n"
            "    int col_base = (lane_id & 3) * 2;\n"
            "    #pragma unroll\n"
            "    for (int i = 0; i < 32; i++) {\n"
            "        int col = col_base + i * 8;\n"
            "        sC[row0 * 256 + col + 0] = __float2bfloat16(ac[i * 4 + 0]);\n"
            "        sC[row0 * 256 + col + 1] = __float2bfloat16(ac[i * 4 + 1]);\n"
            "        sC[row1 * 256 + col + 0] = __float2bfloat16(ac[i * 4 + 2]);\n"
            "        sC[row1 * 256 + col + 1] = __float2bfloat16(ac[i * 4 + 3]);\n"
            "    }\n"
            "}\n")
    return Builtin(body=body, includes=["<cuda_bf16.h>"],
                   return_val=f"store_acc_smem_bf16_n256_fn({sC}, {acc}, {ltid}, {row_offset})")


@builtin(eval_return_type=void, codegen_func=codegen_store_acc_to_smem_bf16_n256)
def store_acc_to_smem_bf16_n256(sC, acc, ltid, row_offset):
    """Store m64n256 accumulator to SMEM as BF16 row-major."""
    raise RuntimeError("should not be called")
