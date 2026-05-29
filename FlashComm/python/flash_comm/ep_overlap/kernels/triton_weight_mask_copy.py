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

import torch
import triton
import triton.language as tl


@triton.jit
def _masked_weight_copy_kernel(
    src_ptr,
    idx_ptr,
    out_ptr,
    n_elems,
    drop_sentinel,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)
    mask = offs < n_elems
    idx_vals = tl.load(idx_ptr + offs, mask=mask, other=drop_sentinel)
    src_vals = tl.load(src_ptr + offs, mask=mask)
    valid = idx_vals != drop_sentinel
    out_vals = tl.where(valid, src_vals, tl.zeros_like(src_vals))
    tl.store(out_ptr + offs, out_vals, mask=mask)


def masked_weight_copy(
    src: torch.Tensor,
    topk_indices: torch.Tensor,
    drop_sentinel: int,
    out: "torch.Tensor | None" = None,
    *,
    block_elems: int = 4096,
) -> torch.Tensor:
    """``out[i, k] = src[i, k] if topk_indices[i, k] != drop_sentinel else 0``.

    ``src`` / ``topk_indices`` must be contiguous and same shape;
    ``topk_indices`` is int32; ``out`` defaults to a fresh
    ``torch.empty_like(src)``.
    """
    if src.shape != topk_indices.shape:
        raise ValueError(f"masked_weight_copy: shape mismatch src={tuple(src.shape)} "
                         f"!= topk_indices={tuple(topk_indices.shape)}")
    if not src.is_contiguous():
        raise ValueError("masked_weight_copy: src must be contiguous")
    if not topk_indices.is_contiguous():
        raise ValueError("masked_weight_copy: topk_indices must be contiguous")
    if topk_indices.dtype != torch.int32:
        raise TypeError("masked_weight_copy: topk_indices must be int32; got "
                        f"{topk_indices.dtype}")
    if out is None:
        out = torch.empty_like(src)
    else:
        if out.shape != src.shape:
            raise ValueError(f"masked_weight_copy: out shape {tuple(out.shape)} "
                             f"!= src shape {tuple(src.shape)}")
        if out.dtype != src.dtype:
            raise TypeError(f"masked_weight_copy: out dtype {out.dtype} != src "
                            f"dtype {src.dtype}")
        if not out.is_contiguous():
            raise ValueError("masked_weight_copy: out must be contiguous")
    n_elems = src.numel()
    if n_elems == 0:
        return out
    grid = ((n_elems + block_elems - 1) // block_elems, )
    _masked_weight_copy_kernel[grid](
        src,
        topk_indices,
        out,
        n_elems,
        int(drop_sentinel),
        BLOCK=block_elems,
    )
    return out
