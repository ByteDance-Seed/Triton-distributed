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
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp
from little_kernel.core.type_system import uint64, void


@builtin
class clock64(BuiltinOp):
    """Get the clock value (clock64())."""

    @staticmethod
    def return_type():
        return uint64

    @staticmethod
    def codegen():
        return Builtin(body="", includes=[], return_val="clock64()")


# ==============================================================================
# globaltimer -- nanosecond wall-clock, consistent across SMs
# ==============================================================================


@builtin
class globaltimer(BuiltinOp):
    """Get the global nanosecond timer (consistent across SMs)."""

    @staticmethod
    def return_type():
        return uint64

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ uint64_t globaltimer_fn() {
    uint64_t val;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(val));
    return val;
}
"""
        return Builtin(body=body, includes=[], return_val="globaltimer_fn()")


# ==============================================================================
# nanosleep -- PTX nanosleep for controlled delays
# ==============================================================================


@builtin
class nanosleep(BuiltinOp):
    """Sleep for approximately ns nanoseconds."""

    @staticmethod
    def return_type(ns):
        return void

    @staticmethod
    def codegen(ns):
        body = """
__device__ __forceinline__ void nanosleep_fn(uint32_t ns) {
    asm volatile("nanosleep.u32 %0;" :: "r"(ns));
}
"""
        return Builtin(body=body, includes=[], return_val=f"nanosleep_fn({ns})")
