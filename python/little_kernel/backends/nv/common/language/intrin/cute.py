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

from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.core.type_system import int32, void


def codegen_elect_one_sync():
    return Builtin(body="", includes=[], return_val="cute::elect_one_sync()")


@builtin(eval_return_type=int32, codegen_func=codegen_elect_one_sync)
def elect_one_sync():
    """Elect one thread in the warp."""
    raise RuntimeError("elect_one_sync should not be called in compilation")


def codegen_prefetch_tma_descriptor(tensor_map):
    return Builtin(body="", includes=[], return_val=f"cute::prefetch_tma_descriptor(&{tensor_map})")


@builtin(eval_return_type=void, codegen_func=codegen_prefetch_tma_descriptor)
def prefetch_tma_descriptor(tensor_map):
    """Prefetch TMA descriptor."""
    raise RuntimeError("prefetch_tma_descriptor should not be called in compilation")
