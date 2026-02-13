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
from ..builtin_base import const_func, const_type_func, builtin, Builtin
from little_kernel.core.type_system import (
    LLType,
    int32,
    uint32,
    int64,
    uint64,
    bool_,
    int8,
    uint8,
    float32,
    float16,
    float64,
    bfloat16,
)


def codegen_sizeof(dtype):
    return Builtin(body="", includes=[], return_val=f"sizeof({dtype})")


@const_func
@builtin(eval_return_type=int32, codegen_func=codegen_sizeof)
def sizeof(dtype):
    assert isinstance(dtype, LLType)
    assert dtype.is_scalar()
    assert dtype.bits >= 8
    return dtype.bits // 8


def codegen_typeof(val):
    return Builtin(body="", includes=[], return_val=f"decltype({val})")


@const_type_func
@builtin(eval_return_type=lambda val_type: val_type, codegen_func=codegen_typeof)
def typeof(val):
    """Get the type of value. This is interpreted by const_fold pass"""
    assert hasattr(val, "dtype"), "value should has dtype"
    return val.dtype


def codegen_val_cast(val, dtype):
    # dtype can be a string (C++ type name) or an LLType object
    if isinstance(dtype, str):
        cpp_type = dtype
    else:
        # Convert LLType to C++ type name
        from little_kernel.codegen.codegen_cuda import _lltype_to_cpp
        cpp_type = _lltype_to_cpp(dtype)
    return Builtin(body="", includes=[], return_val=f"static_cast<{cpp_type}>({val})")


@builtin(eval_return_type=lambda val_type, dtype: dtype
         if isinstance(dtype, LLType) else val_type, codegen_func=codegen_val_cast)
def val_cast(val, dtype):
    """Cast a value to a specific type.
    
    Usage:
        val_cast(value, ll.uint32)  # Cast value to uint32
        ll.uint32(value)  # Same as above (via __call__)
    """
    raise RuntimeError("val_cast should not be called in compilation")


def codegen_to(val, dtype):
    # This is called as val.to(ll.uint32), so val is the first argument
    # and dtype is the second argument
    if isinstance(dtype, str):
        cpp_type = dtype
    else:
        from little_kernel.codegen.codegen_cuda import _lltype_to_cpp
        cpp_type = _lltype_to_cpp(dtype)
    return Builtin(body="", includes=[], return_val=f"static_cast<{cpp_type}>({val})")


@builtin(eval_return_type=lambda val_type, dtype: dtype
         if isinstance(dtype, LLType) else val_type, codegen_func=codegen_to)
def to(val, dtype):
    """Convert a value to a specific type.
    
    Usage:
        value.to(ll.uint32)  # Convert value to uint32
    """
    raise RuntimeError("to should not be called in compilation")


def codegen_ptr_cast(val, dtype):
    return Builtin(body="", includes=[], stmt=f"reinterpret_cast<{dtype}>({val})")


@builtin(eval_return_type=lambda val_type, dtype: dtype, codegen_func=codegen_ptr_cast)
def ptr_cast(val, dtype):
    raise RuntimeError("ptr_cast should not be called in compilation")


def ll_type_to_torch_type(dtype):
    import torch
    if dtype == int32:
        return torch.int32
    elif dtype == uint32:
        return torch.uint32
    elif dtype == int64:
        return torch.int64
    elif dtype == uint64:
        # PyTorch does not fully support torch.uint64; use int64 as fallback
        return torch.int64
    elif dtype == bool_:
        return torch.bool
    elif dtype == int8:
        return torch.int8
    elif dtype == uint8:
        return torch.uint8
    elif dtype == float32:
        return torch.float32
    elif dtype == bfloat16:
        return torch.bfloat16
    elif dtype == float16:
        return torch.float16
    elif dtype == float64:
        return torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
