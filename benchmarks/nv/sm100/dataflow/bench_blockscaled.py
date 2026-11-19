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
"""Shared Torch baseline helpers for block-scaled GEMM benchmarks.nv.common."""

import torch
import torch.nn.functional as F


def pack_torch_scales(logical: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Pack logical (MN, K/vector_size) scales for SWIZZLE_32_4_4."""
    mn, groups = logical.shape
    k_atoms = (groups + 3) // 4
    packed = torch.zeros(
        ((mn + 127) // 128) * k_atoms * 32,
        16,
        dtype=torch.uint8,
        device=logical.device,
    )
    mg, kg = torch.meshgrid(
        torch.arange(mn, device=logical.device),
        torch.arange(groups, device=logical.device),
        indexing="ij",
    )
    row = (mg // 128) * k_atoms * 32 + (kg // 4) * 32 + mg % 32
    col = (mg // 32) % 4 * 4 + kg % 4
    packed[row, col] = logical
    return packed.view(dtype).reshape(-1)


def torch_blockscaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    vector_size: int,
) -> torch.Tensor:
    recipe = {
        16: F.ScalingType.BlockWise1x16,
        32: F.ScalingType.BlockWise1x32,
    }[vector_size]
    swizzle = F.SwizzleType.SWIZZLE_32_4_4
    return F.scaled_mm(
        a,
        b,
        scale_a,
        recipe,
        scale_b,
        recipe,
        swizzle,
        swizzle,
        output_dtype=torch.bfloat16,
    )
