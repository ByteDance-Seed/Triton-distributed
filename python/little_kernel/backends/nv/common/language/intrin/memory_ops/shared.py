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
Shared-memory pointers and shared-memory vector access.

RF10 canonical owner module; see little_kernel.backends.nv.common.language.intrin.contracts
for the shared operand-unit/lane/alignment/cache-policy/effect inventory.
"""

from little_kernel.language.builtin_base import builtin, Builtin, CODEGEN_NEEDS_TYPES_ATTR
from little_kernel.core.type_system import LLType, void, Tensor, int8, uint8, uint16, uint32, uint64, int32, bfloat16, float16, float32, u32x2, u32x4
import little_kernel.backends.nv.common.language.intrin.contracts as contracts
from little_kernel.backends.nv.common.language.intrin.memory_ops.conversion import _gen_gmem_ld_16bit, _gen_gmem_st_16bit

_make_gmem_ld_out_lane_type_check = contracts.make_gmem_ld_out_lane_type_check
_make_gmem_st_in_lane_type_check = contracts.make_gmem_st_in_lane_type_check

# ==============================================================================
# Pointer casting / byte-offset utilities
# ==============================================================================


def codegen_smem_bf16_ptr(base_name, offset):
    base = base_name.strip('"') if isinstance(base_name, str) else base_name
    return Builtin(body="", includes=["<cuda_bf16.h>"],
                   return_val=f"reinterpret_cast<__nv_bfloat16*>({base} + ({offset}))")


@builtin(eval_return_type=Tensor[bfloat16], codegen_func=codegen_smem_bf16_ptr)
def smem_bf16_ptr(base_name: str, offset):
    """Cast shared memory at base+offset to __nv_bfloat16 pointer."""
    raise RuntimeError("should not be called in compilation")


def codegen_smem_bf16_ptr_from_addr(addr):
    return Builtin(
        body="",
        includes=["<cuda_bf16.h>"],
        return_val=f"reinterpret_cast<__nv_bfloat16*>(__cvta_shared_to_generic({addr}))",
    )


@builtin(eval_return_type=Tensor[bfloat16], codegen_func=codegen_smem_bf16_ptr_from_addr)
def smem_bf16_ptr_from_addr(addr):
    """Convert a raw 32-bit shared-memory address to a BF16 generic pointer."""
    raise RuntimeError("should not be called in compilation")


def codegen_smem_u64_ptr(base_name, offset):
    base = base_name.strip('"') if isinstance(base_name, str) else base_name
    return Builtin(body="", includes=[], return_val=f"reinterpret_cast<uint64_t*>({base} + ({offset}))")


@builtin(eval_return_type=Tensor[uint64], codegen_func=codegen_smem_u64_ptr)
def smem_u64_ptr(base_name: str, offset):
    """Cast shared memory at base+offset to uint64_t pointer."""
    raise RuntimeError("should not be called in compilation")


def codegen_ptr_byte_offset(base_ptr, byte_offset):
    # Const-preserving byte-offset helper: the non-const overload yields
    # uint8_t*, the const overload yields const uint8_t*. Call sites thus
    # keep the source pointer's constness (entry-kernel params are emitted
    # `const` when never stored through); an implicit conversion to
    # (const) void* downstream works either way, while a store through a
    # const source fails loudly at nvcc time.
    body = ("\ntemplate <typename T>\n"
            "__device__ __forceinline__ uint8_t* lk_ptr_byte_offset(T* p, long long off) {\n"
            "    return reinterpret_cast<uint8_t*>(p) + off;\n"
            "}\n"
            "template <typename T>\n"
            "__device__ __forceinline__ const uint8_t* lk_ptr_byte_offset(const T* p, long long off) {\n"
            "    return reinterpret_cast<const uint8_t*>(p) + off;\n"
            "}\n")
    return Builtin(body=body, includes=[], return_val=f"lk_ptr_byte_offset({base_ptr}, ({byte_offset}))")


@builtin(eval_return_type=Tensor[bfloat16], codegen_func=codegen_ptr_byte_offset)
def ptr_byte_offset(base_ptr, byte_offset):
    """Byte-offset pointer arithmetic: (uint8_t*)base + offset, returned as void*."""
    raise RuntimeError("should not be called in compilation")


def codegen_cvta_generic_to_shared(ptr):
    return Builtin(body="", includes=[], return_val=f"(uint32_t)__cvta_generic_to_shared({ptr})")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvta_generic_to_shared)
def cvta_generic_to_shared(ptr):
    """Convert a generic pointer to its uint32 shared-window byte address.

    The result is a ``.shared`` state-space address usable directly in PTX
    address operands. Canonical typed idiom for an mbarrier slot::

        ll.alloc_shared_memory("s_bar", ll.uint64, 1, 8)
        bar: ll.uint32 = ll.cvta_generic_to_shared(s_bar)
        ll.init_smem_barrier(bar, 1)
        ll.mbarrier_arrive_and_expect_tx(bar, tx_bytes)
        ll.mbarrier_wait(bar, phase)

    The CTA-scoped mbarrier ops accept either this uint32 address (used
    directly in ``[%0]``) or the shared tensor itself (they then run this
    conversion internally); do not apply cvta twice and do not pass a scalar
    load of the slot contents.
    """
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Shared-memory vector access: 16-byte (v4) and 8-byte (v2) u32 lanes
# ==============================================================================
#
# Generic contract (no task or shape assumptions): the shared-memory mirror
# of the global v4/v2 families above.
#   * ``smem``         -- any LittleKernel Tensor with shared scope; offset by
#                         ``elem_offset`` elements of the tensor's own dtype.
#   * Legality preconditions of PTX ``ld/st.shared.v4.u32`` /
#     ``ld/st.shared.v2.u32``:
#       - byte address of (smem + elem_offset * sizeof(dtype)) MUST be
#         16-byte (v4) or 8-byte (v2) aligned; violating it is a fault;
#       - the address must lie inside the CTA's shared window
#         (``__cvta_generic_to_shared`` truncates to the 32-bit shared
#         address; a non-shared generic pointer is a fault).
#   * ``out``/``vals`` -- a local-scope uint32 array of exactly 4 (v4) or
#                         2 (v2) elements; lanes map to bytes in order.
#   * Shared memory has no cache policy; normal ld/st.shared ordering rules
#     apply (visibility to other threads requires the usual barrier). These
#     are bit-exact plain shared accesses; there is no performance-hint knob.
#
# Usage (frontend):
#     frag = ll.empty([4], dtype=ll.uint32, scope="local")
#     ll.st_shared_v4_u32(tile, smem_off, frag)
#     __syncthreads(); ll.ld_shared_v4_u32(tile, smem_off, frag)


def _smem_v_body(inst, width, is_load):
    """Emit one inline-asm helper for a vector shared-memory access."""
    name = f"smem_{inst.replace('.', '_')}_fn"
    if is_load:
        outs = ", ".join(f'"=r"(out[{i}])' for i in range(width))
        regs = ", ".join(f"%{i}" for i in range(width))
        body = ("\n__device__ __forceinline__ void " + name + f"(const void* ptr, uint32_t (&out)[{width}]) {{\n"
                f'    asm volatile("{inst} {{{regs}}}, [%{width}];"\n'
                f'                 : {outs}\n'
                '                 : "r"((uint32_t)__cvta_generic_to_shared(ptr)) : "memory");\n'
                "}\n")
    else:
        ins = ", ".join(f'"r"(vals[{i}])' for i in range(width))
        regs = ", ".join(f"%{i + 1}" for i in range(width))
        body = ("\n__device__ __forceinline__ void " + name + f"(void* ptr, const uint32_t (&vals)[{width}]) {{\n"
                f'    asm volatile("{inst} [%0], {{{regs}}};"\n'
                f'                 :: "r"((uint32_t)__cvta_generic_to_shared(ptr)), {ins}\n'
                '                 : "memory");\n'
                "}\n")
    return name, body


def _make_smem_v(inst, width, is_load):

    def codegen(smem, elem_offset, words):
        name, body = _smem_v_body(inst, width, is_load)
        return Builtin(body=body, includes=["<cstdint>"], return_val=f"{name}({smem} + {elem_offset}, {words})")

    return codegen


@builtin(eval_return_type=void, codegen_func=_make_smem_v("ld.shared.v4.u32", 4, True))
def ld_shared_v4_u32(smem, elem_offset, out):
    """16-byte aligned shared load into out[0..4) (uint32)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_smem_v("st.shared.v4.u32", 4, False))
def st_shared_v4_u32(smem, elem_offset, vals):
    """16-byte aligned shared store of vals[0..4) (uint32)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_smem_v("ld.shared.v2.u32", 2, True))
def ld_shared_v2_u32(smem, elem_offset, out):
    """8-byte aligned shared load into out[0..2) (uint32)."""
    raise RuntimeError("should not be called in compilation")


@builtin(eval_return_type=void, codegen_func=_make_smem_v("st.shared.v2.u32", 2, False))
def st_shared_v2_u32(smem, elem_offset, vals):
    """8-byte aligned shared store of vals[0..2) (uint32)."""
    raise RuntimeError("should not be called in compilation")


# ==============================================================================
# Shared-memory twins of the typed 16-bit vector intrinsics
# ==============================================================================
#
# Same generated code and same legality contract as the gmem family
# (caller guarantees natural lanes*2-byte alignment and in-bounds lanes;
# head/tail stay scalar): generic addressing reaches __shared__ arrays
# through reinterpret_cast identically.

codegen_smem_ld_bf16x8 = _gen_gmem_ld_16bit("smem_ld_bf16x8", "__nv_bfloat16", "uint4", 8, "__bfloat162float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("smem_ld_bf16x8", float32), eval_return_type=void,
         codegen_func=codegen_smem_ld_bf16x8)
def smem_ld_bf16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Load 8 aligned (16 B) contiguous bf16 values from SMEM into FP32 registers."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_ld_bf16x4 = _gen_gmem_ld_16bit("smem_ld_bf16x4", "__nv_bfloat16", "uint2", 4, "__bfloat162float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("smem_ld_bf16x4", float32), eval_return_type=void,
         codegen_func=codegen_smem_ld_bf16x4)
def smem_ld_bf16x4(mem, index, f0, f1, f2, f3):
    """Load 4 aligned (8 B) contiguous bf16 values from SMEM into FP32 registers."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_ld_f16x8 = _gen_gmem_ld_16bit("smem_ld_f16x8", "__half", "uint4", 8, "__half2float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("smem_ld_f16x8", float32), eval_return_type=void,
         codegen_func=codegen_smem_ld_f16x8)
def smem_ld_f16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Load 8 aligned (16 B) contiguous fp16 values from SMEM into FP32 registers."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_ld_f16x4 = _gen_gmem_ld_16bit("smem_ld_f16x4", "__half", "uint2", 4, "__half2float")


@builtin(eval_arg_type=_make_gmem_ld_out_lane_type_check("smem_ld_f16x4", float32), eval_return_type=void,
         codegen_func=codegen_smem_ld_f16x4)
def smem_ld_f16x4(mem, index, f0, f1, f2, f3):
    """Load 4 aligned (8 B) contiguous fp16 values from SMEM into FP32 registers."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_st_f32_to_bf16x8 = _gen_gmem_st_16bit("smem_st_f32_to_bf16x8", "__nv_bfloat16", "uint4", 8,
                                                   "__floats2bfloat162_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("smem_st_f32_to_bf16x8", float32), eval_return_type=void,
         codegen_func=codegen_smem_st_f32_to_bf16x8)
def smem_st_f32_to_bf16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Round 8 FP32 lanes to bf16 (single RNE) and store to SMEM in one 16 B transaction."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_st_f32_to_bf16x4 = _gen_gmem_st_16bit("smem_st_f32_to_bf16x4", "__nv_bfloat16", "uint2", 4,
                                                   "__floats2bfloat162_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("smem_st_f32_to_bf16x4", float32), eval_return_type=void,
         codegen_func=codegen_smem_st_f32_to_bf16x4)
def smem_st_f32_to_bf16x4(mem, index, f0, f1, f2, f3):
    """Round 4 FP32 lanes to bf16 (single RNE) and store to SMEM in one 8 B transaction."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_st_f32_to_f16x8 = _gen_gmem_st_16bit("smem_st_f32_to_f16x8", "__half", "uint4", 8, "__floats2half2_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("smem_st_f32_to_f16x8", float32), eval_return_type=void,
         codegen_func=codegen_smem_st_f32_to_f16x8)
def smem_st_f32_to_f16x8(mem, index, f0, f1, f2, f3, f4, f5, f6, f7):
    """Round 8 FP32 lanes to fp16 (single RNE) and store to SMEM in one 16 B transaction."""
    raise RuntimeError("should not be called in compilation")


codegen_smem_st_f32_to_f16x4 = _gen_gmem_st_16bit("smem_st_f32_to_f16x4", "__half", "uint2", 4, "__floats2half2_rn")


@builtin(eval_arg_type=_make_gmem_st_in_lane_type_check("smem_st_f32_to_f16x4", float32), eval_return_type=void,
         codegen_func=codegen_smem_st_f32_to_f16x4)
def smem_st_f32_to_f16x4(mem, index, f0, f1, f2, f3):
    """Round 4 FP32 lanes to fp16 (single RNE) and store to SMEM in one 8 B transaction."""
    raise RuntimeError("should not be called in compilation")


def _k_major_bf16_index(rows, row, col):
    return (f"(((({col}) >> 6) * ({rows}) + ({row})) * 64 + "
            f"((((({col}) & 63) >> 3) ^ (({row}) & 7)) * 8) + (({col}) & 7))")


def codegen_smem_st_bf16_k_major(mem, rows, row, col, value):
    return Builtin(body="", includes=[], return_val=f"{mem}[{_k_major_bf16_index(rows, row, col)}] = {value}")


@builtin(eval_return_type=void, codegen_func=codegen_smem_st_bf16_k_major)
def smem_st_bf16_k_major(mem, rows, row, col, value):
    """Store BF16 by logical coordinate in canonical K-major SMEM."""
    raise RuntimeError("should not be called in compilation")


def codegen_smem_ld_bf16_k_major(mem, rows, row, col):
    return Builtin(body="", includes=[], return_val=f"{mem}[{_k_major_bf16_index(rows, row, col)}]")


@builtin(eval_return_type=bfloat16, codegen_func=codegen_smem_ld_bf16_k_major)
def smem_ld_bf16_k_major(mem, rows, row, col):
    """Load BF16 by logical coordinate in canonical K-major SMEM."""
    raise RuntimeError("should not be called in compilation")
