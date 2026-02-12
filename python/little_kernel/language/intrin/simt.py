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
SIMT intrinsics: thread/block/warp operations.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).
"""

from ..builtin_base import builtin, Builtin
from little_kernel.core.type_system import int32, uint32, void
from functools import partial

# ==============================================================================
# Thread/Block indices and dimensions
# ==============================================================================


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


# ==============================================================================
# SM identification
# ==============================================================================


def codegen_smid():
    body = """
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t smid;
    asm ("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}
"""
    return Builtin(body=body, includes=[], return_val="get_smid()")


@builtin(eval_return_type=uint32, codegen_func=codegen_smid)
def smid():
    """Get the SM ID of the current thread."""
    raise RuntimeError("smid should never be called in compilation")


# ==============================================================================
# Warp-level operations
# ==============================================================================


def codegen_get_lane_idx():
    return Builtin(
        body="\n__device__ __forceinline__ uint32_t get_lane_idx() {\n"
        "    uint32_t lane_id;\n"
        '    asm ("mov.u32 %0, %%laneid;" : "=r"(lane_id));\n'
        "    return lane_id;\n"
        "}\n",
        includes=[],
        return_val="get_lane_idx()",
    )


@builtin(eval_return_type=int32, codegen_func=codegen_get_lane_idx)
def get_lane_idx():
    """Get the lane index of the current thread."""
    raise RuntimeError("get_lane_idx should never be called in compilation")


def codegen_elect_one_sync():
    body = """
__device__ __forceinline__ bool elect_one_sync_fn() {
    uint32_t pred;
    asm volatile(
        "{\\n"
        ".reg .pred p;\\n"
        "elect.sync _|p, 0xFFFFFFFF;\\n"
        "selp.b32 %0, 1, 0, p;\\n"
        "}\\n"
        : "=r"(pred));
    return pred != 0;
}
"""
    return Builtin(body=body, includes=[], return_val="elect_one_sync_fn()")


@builtin(eval_return_type=int32, codegen_func=codegen_elect_one_sync)
def elect_one_sync():
    """Elect one thread in the warp."""
    raise RuntimeError("elect_one_sync should not be called in compilation")


# ==============================================================================
# Warp voting and bit operations
# ==============================================================================


def codegen_ballot_sync(mask, predicate):
    return Builtin(body="", includes=[], return_val=f"__ballot_sync({mask}, {predicate})")


@builtin(eval_return_type=uint32, codegen_func=codegen_ballot_sync)
def ballot_sync(mask, predicate):
    """Warp ballot -- returns bitmask of threads where predicate is true."""
    raise RuntimeError("should not call ballot_sync in compilation")


def codegen_match_any_sync(mask, value):
    return Builtin(body="", includes=[], return_val=f"__match_any_sync({mask}, {value})")


@builtin(eval_return_type=uint32, codegen_func=codegen_match_any_sync)
def match_any_sync(mask, value):
    """Warp match -- returns bitmask of threads holding the same value."""
    raise RuntimeError("should not call match_any_sync in compilation")


def codegen_popc(x):
    return Builtin(body="", includes=[], return_val=f"__popc({x})")


@builtin(eval_return_type=int32, codegen_func=codegen_popc)
def popc(x):
    """Population count (number of set bits)."""
    raise RuntimeError("should not call popc in compilation")


def codegen_ffs(x):
    return Builtin(body="", includes=[], return_val=f"__ffs({x})")


@builtin(eval_return_type=int32, codegen_func=codegen_ffs)
def ffs(x):
    """Find first set bit (1-indexed, 0 if no bits set)."""
    raise RuntimeError("should not call ffs in compilation")


# ==============================================================================
# Grid dimensions
# ==============================================================================


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


# ==============================================================================
# Warpgroup register management
# ==============================================================================


def codegen_warpgroup_reg_alloc(num_registers):
    return Builtin(
        body=(f"\n__device__ __forceinline__ void warpgroup_reg_alloc_{num_registers}_fn() {{\n"
              f'    asm volatile("setmaxnreg.inc.sync.aligned.u32 {num_registers};\\n");\n'
              f"}}\n"), includes=[], return_val=f"warpgroup_reg_alloc_{num_registers}_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_warpgroup_reg_alloc)
def warpgroup_reg_alloc(num_registers):
    """Allocate registers for warpgroup (setmaxnreg.inc)."""
    raise RuntimeError("should not call warpgroup_reg_alloc in compilation")


def codegen_warpgroup_reg_dealloc(num_registers):
    return Builtin(
        body=(f"\n__device__ __forceinline__ void warpgroup_reg_dealloc_{num_registers}_fn() {{\n"
              f'    asm volatile("setmaxnreg.dec.sync.aligned.u32 {num_registers};\\n");\n'
              f"}}\n"), includes=[], return_val=f"warpgroup_reg_dealloc_{num_registers}_fn()")


@builtin(eval_return_type=void, codegen_func=codegen_warpgroup_reg_dealloc)
def warpgroup_reg_dealloc(num_registers):
    """Deallocate registers for warpgroup (setmaxnreg.dec)."""
    raise RuntimeError("should not call warpgroup_reg_dealloc in compilation")
