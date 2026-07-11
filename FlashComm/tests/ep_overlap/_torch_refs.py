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
from __future__ import annotations

import torch


def grouped_matmul_bf16(A: torch.Tensor, B_raw: torch.Tensor, expert_counts: torch.Tensor,
                        padded_offsets: torch.Tensor) -> torch.Tensor:
    """Per-expert bf16 ``A_e @ B_raw[e].t()`` (byte-equal to CuTeDSL group_gemm)."""
    n = int(B_raw.shape[1])
    out = torch.zeros(A.shape[0], n, dtype=A.dtype, device=A.device)
    for e, count in enumerate(expert_counts):
        count = int(count)
        if count <= 0:
            continue
        start = int(padded_offsets[e])
        a_e = A[start:start + count]
        w_e = B_raw[e]
        out[start:start + count] = torch.matmul(a_e, w_e.t())
    return out


def grouped_matmul_fp32_accum(A: torch.Tensor, B_raw: torch.Tensor, expert_counts: torch.Tensor,
                              padded_offsets: torch.Tensor) -> torch.Tensor:
    """Per-expert fp32 GEMM after bf16 -> fp32 upcast; TF32 forced off.

    K-direction reduction order differs from the CuTeDSL MMA, so the
    resulting fp32 accumulator drifts by a few fp32 ULPs (~1-3 BF16
    ULPs after the epilogue cast); enforced by ``assert_close_bf16_ulp``.
    """
    n = int(B_raw.shape[1])
    out = torch.zeros(A.shape[0], n, dtype=torch.float32, device=A.device)
    saved_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for e, count in enumerate(expert_counts):
            count = int(count)
            if count <= 0:
                continue
            start = int(padded_offsets[e])
            a_e = A[start:start + count].to(torch.float32)
            w_e = B_raw[e].to(torch.float32)
            out[start:start + count] = torch.matmul(a_e, w_e.t())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved_tf32
    return out


__all__ = [
    "grouped_matmul_bf16",
    "grouped_matmul_fp32_accum",
]
