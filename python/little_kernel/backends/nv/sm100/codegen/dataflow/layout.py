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
"""CUDA descriptor fields for a previously planned shared-memory layout."""
from little_kernel.backends.nv.sm100.language.dataflow import SMemLayoutPlan
from little_kernel.backends.nv.sm100.passes.dataflow.channels import dtype_size as _dtype_size


def _layout_fields(plan: SMemLayoutPlan) -> tuple[int, int, int, int]:
    elem_bytes = _dtype_size(plan.dtype)
    other = plan.logical_shape[0] if plan.major == "K" else plan.logical_shape[1]
    if plan.swizzle_bytes == 0:
        # INTERLEAVE (no swizzle): simple row-major layout
        atom = 1
        lbo = 16 if plan.major == "K" else other * elem_bytes
        sbo = plan.logical_shape[1] * elem_bytes if plan.major == "K" else 16
        layout_type = 0
        return atom, lbo, sbo, layout_type
    atom = plan.swizzle_bytes // elem_bytes
    lbo = 16 if plan.major == "K" else other * atom * elem_bytes
    sbo = 8 * atom * elem_bytes
    layout_type = {32: 6, 64: 4, 128: 2}[plan.swizzle_bytes]
    return atom, lbo, sbo, layout_type
