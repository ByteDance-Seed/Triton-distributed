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
"""Logical SIMT indices shared by CUDA and HIP C++ backends."""
from functools import partial
from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.core.type_system import int32, uint32, void


def codegen_thread_idx(thread_idx):
    return Builtin(body="", includes=[], return_val=f"threadIdx.{thread_idx}")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "x"))
def threadIdx_x():
    """Thread index in the x dimension."""
    raise RuntimeError("threadIdx_x should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "y"))
def threadIdx_y():
    """Thread index in the y dimension."""
    raise RuntimeError("threadIdx_y should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "z"))
def threadIdx_z():
    """Thread index in the z dimension."""
    raise RuntimeError("threadIdx_z should never be called in compilation")


def codegen_block_idx(block_idx):
    return Builtin(body="", includes=[], return_val=f"blockIdx.{block_idx}")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_idx, "x"))
def blockIdx_x():
    """Block index in the x dimension."""
    raise RuntimeError("blockIdx_x should never be called in compilation")


def codegen_block_dim(dim):
    return Builtin(body="", includes=[], return_val=f"blockDim.{dim}")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_dim, "x"))
def blockDim_x():
    """Block dimension in the x dimension."""
    raise RuntimeError("blockDim_x should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_dim, "y"))
def blockDim_y():
    """Block dimension in the y dimension."""
    raise RuntimeError("blockDim_y should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_dim, "z"))
def blockDim_z():
    """Block dimension in the z dimension."""
    raise RuntimeError("blockDim_z should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "y"))
def thread_y():
    """Thread index in the y dimension."""
    raise RuntimeError("thread_y should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_thread_idx, "z"))
def thread_z():
    """Thread index in the z dimension."""
    raise RuntimeError("thread_z should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_idx, "y"))
def blockIdx_y():
    """Block index in the y dimension."""
    raise RuntimeError("blockIdx_y should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_block_idx, "z"))
def blockIdx_z():
    """Block index in the z dimension."""
    raise RuntimeError("blockIdx_z should never be called in compilation")


def codegen_grid_dim(dim):
    return Builtin(body="", includes=[], return_val=f"gridDim.{dim}")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_grid_dim, "x"))
def gridDim_x():
    """Grid dimension in the x dimension."""
    raise RuntimeError("gridDim_x should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_grid_dim, "y"))
def gridDim_y():
    """Grid dimension in the y dimension."""
    raise RuntimeError("gridDim_y should never be called in compilation")


@builtin(eval_return_type=int32, codegen_func=partial(codegen_grid_dim, "z"))
def gridDim_z():
    """Grid dimension in the z dimension."""
    raise RuntimeError("gridDim_z should never be called in compilation")
