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

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import (LLType, void, Tensor, uint8, uint32, uint64, bfloat16)


def codegen_alloc_dynamic_shared_memory(size_bytes, align_bytes, name):
    # Strip C++ string literal quotes if present (name is a variable identifier)
    var_name = name.strip('"')
    stmt = f"extern __shared__ __align__({align_bytes}) uint8_t {var_name}[]"

    return Builtin(body="", includes=[], return_val=stmt)


def alloc_dynamic_shared_memory_eval_arg_type(ctx, size_bytes, align_bytes, name):
    ctx[name] = uint8


@builtin(eval_return_type=void, eval_arg_type=alloc_dynamic_shared_memory_eval_arg_type,
         codegen_func=codegen_alloc_dynamic_shared_memory)
def alloc_dynamic_shared_memory(size_bytes: int, align_bytes: int, name: str):
    """Allocate dynamic shared memory with the given size in bytes."""
    raise RuntimeError("alloc_dynamic_shared_memory should never be called in compilation")


def codegen_alloc_local_memory(var_name, dtype, elems):
    stmt = f"{dtype} {var_name}[{elems}]"

    return Builtin(body="", includes=[], return_val=stmt)


def alloc_local_memory_eval_arg_type(ctx, var_name, dtype, elems):
    # Use Tensor[dtype] so that runtime subscript (e.g. arr[stage]) correctly
    # infers element type.  Consistent with alloc_shared_memory_eval_arg_type.
    ctx[var_name] = Tensor[dtype]


@builtin(eval_return_type=lambda var_name, dtype, elems: Tensor[dtype], eval_arg_type=alloc_local_memory_eval_arg_type,
         codegen_func=codegen_alloc_local_memory)
def alloc_local_memory(var_name: str, dtype: LLType, elems: int):
    """Allocate local memory with the given number of elements."""
    raise RuntimeError("alloc_local_memory should never be called in compilation")


def codegen_alloc_shared_memory(var_name, dtype, elems, align_bytes="0"):
    """Generate static shared memory declaration: __shared__ dtype var_name[elems]
    
    If align_bytes > 0, generates: __shared__ __align__(align_bytes) dtype var_name[elems]
    """
    # align_bytes comes in as a string from codegen
    try:
        align_val = int(align_bytes)
    except (ValueError, TypeError):
        align_val = 0

    if align_val > 0:
        stmt = f"__shared__ __align__({align_val}) {dtype} {var_name}[{elems}]"
    else:
        stmt = f"__shared__ {dtype} {var_name}[{elems}]"
    return Builtin(body="", includes=[], return_val=stmt)


def alloc_shared_memory_eval_arg_type(ctx, var_name, dtype, elems, align_bytes=0):
    """Set variable type in context for shared memory allocation."""
    ctx[var_name] = Tensor[dtype]


@builtin(eval_return_type=lambda var_name, dtype, elems, align_bytes=0: Tensor[dtype],
         eval_arg_type=alloc_shared_memory_eval_arg_type, codegen_func=codegen_alloc_shared_memory)
def alloc_shared_memory(var_name: str, dtype: LLType, elems: int, align_bytes: int = 0):
    """Allocate static shared memory with the given number of elements.
    
    Args:
        var_name: Variable name for the shared memory
        dtype: Data type of elements
        elems: Number of elements
        align_bytes: Alignment in bytes (0 means no alignment specified)
    """
    raise RuntimeError("alloc_shared_memory should never be called in compilation")


def codegen_slice_dynamic_shared_memory(var_name, dtype, start, size, shmem_buf_name):
    # TODO(zhengsize): find a better way to handle such hacky case
    if '[' in var_name and ']' in var_name:
        stmt = f"{var_name} = reinterpret_cast<{dtype}>({shmem_buf_name} + {start}); /*size = {size} bytes*/"
    else:
        stmt = f"{dtype} {var_name} = reinterpret_cast<{dtype}>({shmem_buf_name} + {start}); /*size = {size} bytes*/"

    return Builtin(body="", includes=[], return_val=stmt)


def slice_dynamic_shared_memory_eval_arg_type(ctx, var_name, dtype, start, size, shmem_buf_name):
    ctx[var_name] = dtype


@builtin(eval_return_type=lambda var_name, dtype, start, size, shmem_buf_name: dtype,
         eval_arg_type=slice_dynamic_shared_memory_eval_arg_type, codegen_func=codegen_slice_dynamic_shared_memory)
def slice_dynamic_shared_memory(var_name: str, dtype: LLType, start: int, size: int, shmem_buf_name: str):
    """Slice dynamic shared memory with the given start and size."""
    raise RuntimeError("slice_dynamic_shared_memory should never be called in compilation")


def codegen_align_memory(align_bytes, scope):
    return Builtin(body="", includes=[], return_val=f"/* align {scope} to {align_bytes} */")


@builtin(eval_return_type=void, codegen_func=codegen_align_memory)
def align_memory(align_bytes: int, scope: str):
    """Align memory to the given byte alignment."""
    raise RuntimeError("align_memory should never be called in compilation")


def codegen_empty(shape, dtype, scope):
    raise RuntimeError("empty should never be called in code generation")


@builtin(eval_return_type=lambda shape, dtype, scope: Tensor[dtype], codegen_func=codegen_empty)
def empty(shape: list, dtype: LLType, scope: str):
    """Create an empty tensor with the given shape, dtype, and scope."""
    raise RuntimeError("empty should never be called in compilation")


def codegen_ldcg(ptr, offset):
    return Builtin(body="", includes=["<device_functions.h>"], return_val=f"__ldcg({ptr} + {offset})")


def eval_return_type_ldcg(dtype, offset_dtype):
    """Evaluate return type for __ldcg function."""
    from little_kernel.core.type_system import TensorType
    if dtype is None:
        raise TypeError("__ldcg: cannot infer type of first argument (ptr)")
    if isinstance(dtype, TensorType):
        return dtype.element_type
    else:
        raise TypeError(f"__ldcg expects Tensor as first argument, got {type(dtype).__name__}: {dtype}")


@builtin(eval_return_type=eval_return_type_ldcg, codegen_func=codegen_ldcg)
def __ldcg(ptr: Tensor, offset: int):
    """Load a value from the given pointer with the given offset."""
    raise RuntimeError("ldcg should never be called in compilation")


def codegen_ldca(ptr, offset):
    return Builtin(body="", includes=["<device_functions.h>"], return_val=f"__ldca({ptr} + {offset})")


def eval_return_type_ldca(dtype, offset_dtype):
    """Evaluate return type for __ldca function."""
    from little_kernel.core.type_system import TensorType
    if dtype is None:
        raise TypeError("__ldca: cannot infer type of first argument (ptr)")
    if isinstance(dtype, TensorType):
        return dtype.element_type
    else:
        raise TypeError(f"__ldca expects Tensor as first argument, got {type(dtype).__name__}: {dtype}")


@builtin(eval_return_type=eval_return_type_ldca, codegen_func=codegen_ldca)
def __ldca(ptr: Tensor, offset: int):
    """Load a value from the given pointer with the given offset."""
    raise RuntimeError("ldca should never be called in compilation")


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


def codegen_smem_u64_ptr(base_name, offset):
    base = base_name.strip('"') if isinstance(base_name, str) else base_name
    return Builtin(body="", includes=[], return_val=f"reinterpret_cast<uint64_t*>({base} + ({offset}))")


@builtin(eval_return_type=Tensor[uint64], codegen_func=codegen_smem_u64_ptr)
def smem_u64_ptr(base_name: str, offset):
    """Cast shared memory at base+offset to uint64_t pointer."""
    raise RuntimeError("should not be called in compilation")


def codegen_ptr_byte_offset(base_ptr, byte_offset):
    return Builtin(body="", includes=[],
                   return_val=f"(void*)(reinterpret_cast<uint8_t*>({base_ptr}) + ({byte_offset}))")


@builtin(eval_return_type=Tensor[bfloat16], codegen_func=codegen_ptr_byte_offset)
def ptr_byte_offset(base_ptr, byte_offset):
    """Byte-offset pointer arithmetic: (uint8_t*)base + offset, returned as void*."""
    raise RuntimeError("should not be called in compilation")


def codegen_cvta_generic_to_shared(ptr):
    return Builtin(body="", includes=[], return_val=f"(uint32_t)__cvta_generic_to_shared({ptr})")


@builtin(eval_return_type=uint32, codegen_func=codegen_cvta_generic_to_shared)
def cvta_generic_to_shared(ptr):
    """Convert generic pointer to shared memory address (uint32)."""
    raise RuntimeError("should not be called in compilation")


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
