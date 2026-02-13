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
from little_kernel.core.type_system import uint64, void


def codegen_get_clock():
    return Builtin(body="", includes=[], return_val="clock64()")


@builtin(eval_return_type=uint64, codegen_func=codegen_get_clock)
def clock64():
    """Get the clock value."""
    raise RuntimeError("clock64 should never be called in compilation")


# ==============================================================================
# globaltimer -- nanosecond wall-clock, consistent across SMs
# ==============================================================================


def codegen_globaltimer():
    body = """
__device__ __forceinline__ uint64_t globaltimer_fn() {
    uint64_t val;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(val));
    return val;
}
"""
    return Builtin(body=body, includes=[], return_val="globaltimer_fn()")


@builtin(eval_return_type=uint64, codegen_func=codegen_globaltimer)
def globaltimer():
    """Get the global nanosecond timer (consistent across SMs)."""
    raise RuntimeError("globaltimer should never be called in compilation")


# ==============================================================================
# nanosleep -- PTX nanosleep for controlled delays
# ==============================================================================


def codegen_nanosleep(ns):
    body = """
__device__ __forceinline__ void nanosleep_fn(uint32_t ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}
"""
    return Builtin(body=body, includes=[], return_val=f"nanosleep_fn({ns})")


@builtin(eval_return_type=void, codegen_func=codegen_nanosleep)
def nanosleep(ns):
    """Sleep for approximately ns nanoseconds."""
    raise RuntimeError("nanosleep should never be called in compilation")
