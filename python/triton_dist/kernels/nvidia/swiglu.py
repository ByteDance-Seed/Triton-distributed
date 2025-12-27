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

import math
import torch
import triton
import triton.language as tl
from dataclasses import dataclass


def calculate_settings(n):
    # ref: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43
    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}.")
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@dataclass
class SwiGLUContext:
    BLOCK_SIZE: int
    num_warps: int


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


@triton.jit()
def _swiglu_forward_kernel(
    A_ptr,
    A_row_stride,
    B_ptr,
    B_row_stride,
    C_ptr,
    C_row_stride,
    scale_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    A_ptr += row_idx * A_row_stride
    B_ptr += row_idx * B_row_stride
    C_ptr += row_idx * C_row_stride

    A_row = tl.load(A_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    C_row = silu(A_row) * B_row

    if scale_ptr is not None:
        C_row = C_row * tl.load(scale_ptr + row_idx)

    tl.store(C_ptr + col_offsets, C_row, mask=mask)


@triton.jit()
def _swiglu_forward_kernel_persistent(
    A_ptr,
    A_row_stride,
    B_ptr,
    B_row_stride,
    C_ptr,
    C_row_stride,
    scale_ptr,
    n_cols,
    n_rows,
    available_sms,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    A_ptr += row_start * A_row_stride
    B_ptr += row_start * B_row_stride
    C_ptr += row_start * C_row_stride

    for row_idx in range(row_start, row_end):
        A_row = tl.load(A_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

        C_row = silu(A_row) * B_row

        if scale_ptr is not None:
            C_row = C_row * tl.load(scale_ptr + row_idx)

        tl.store(C_ptr + col_offsets, C_row, mask=mask)

        A_ptr += A_row_stride
        B_ptr += B_row_stride
        C_ptr += C_row_stride


@triton.jit()
def _swiglu_backward_kernel(
    dC_ptr,
    dC_row_stride,
    A_ptr,
    A_row_stride,
    B_ptr,
    B_row_stride,
    scale_ptr,
    dA_ptr,
    dA_row_stride,
    dB_ptr,
    dB_row_stride,
    dscale_ptr,
    n_cols,
    n_rows,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dC_ptr += row_idx * dC_row_stride
    A_ptr += row_idx * A_row_stride
    B_ptr += row_idx * B_row_stride

    dC_row = tl.load(dC_ptr + col_offsets, mask=mask, other=0)
    A_row = tl.load(A_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    # Recompute for backward pass
    sigmoid_A = tl.sigmoid(A_row)
    silu_A = A_row * sigmoid_A

    scale_v = 1.0
    if scale_ptr is not None:
        scale_v = tl.load(scale_ptr + row_idx)

    dA_row = dC_row * (silu_A * (1 - sigmoid_A) + sigmoid_A) * B_row * scale_v
    dB_row = dC_row * silu_A * scale_v

    tl.store(dA_ptr + col_offsets, dA_row, mask=mask)
    tl.store(dB_ptr + col_offsets, dB_row, mask=mask)

    if dscale_ptr is not None:
        tl.store(dscale_ptr + row_idx, tl.sum(silu_A * B_row * dC_row))


@triton.jit()
def _swiglu_backward_kernel_persistent(
    dC_ptr,
    dC_row_stride,
    A_ptr,
    A_row_stride,
    B_ptr,
    B_row_stride,
    scale_ptr,
    dA_ptr,
    dA_row_stride,
    dB_ptr,
    dB_row_stride,
    dscale_ptr,
    n_cols,
    n_rows,
    available_sms,
    rows_per_program,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0).to(tl.int64)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dC_ptr += row_start * dC_row_stride
    A_ptr += row_start * A_row_stride
    B_ptr += row_start * B_row_stride

    for row_idx in range(row_start, row_end):
        dC_row = tl.load(dC_ptr + col_offsets, mask=mask, other=0)
        A_row = tl.load(A_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
        B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

        # Recompute for backward pass
        sigmoid_A = tl.sigmoid(A_row)
        silu_A = A_row * sigmoid_A

        scale_v = 1.0
        if scale_ptr is not None:
            scale_v = tl.load(scale_ptr + row_idx)

        dA_row = dC_row * (silu_A * (1 - sigmoid_A) + sigmoid_A) * B_row * scale_v
        dB_row = dC_row * silu_A * scale_v

        tl.store(dA_ptr + col_offsets, dA_row, mask=mask)
        tl.store(dB_ptr + col_offsets, dB_row, mask=mask)

        if dscale_ptr is not None:
            tl.store(dscale_ptr + row_idx, tl.sum(silu_A * B_row * dC_row))

        dC_ptr += dC_row_stride
        A_ptr += A_row_stride
        B_ptr += B_row_stride


def swiglu_forward(AB, scale=None, sm_margin=0, use_aot=False):
    shape = AB.shape
    dim = shape[-1]
    AB = AB.view(-1, dim)
    dim = dim // 2
    A = AB[:, :dim]
    B = AB[:, dim:]
    n_rows, n_cols = A.shape

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    C = torch.empty_like(A)

    sm_count = torch.cuda.get_device_properties(A.device).multi_processor_count
    available_sms = sm_count - sm_margin
    available_sms = max(1, available_sms)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This swiglu doesn't support feature dim >= 64KB.")

    if sm_margin > 0:
        rows_per_program = math.ceil(n_rows / available_sms)
        grid = (available_sms, )

        _swiglu_forward_kernel_persistent[grid](
            A,
            A.stride(0),
            B,
            B.stride(0),
            C,
            C.stride(0),
            scale,
            n_cols,
            n_rows,
            available_sms,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _swiglu_forward_kernel[(n_rows, )](
            A,
            A.stride(0),
            B,
            B.stride(0),
            C,
            C.stride(0),
            scale,
            n_cols,
            n_rows,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    ret_ctx = SwiGLUContext(BLOCK_SIZE, num_warps)
    return C.view(*AB.shape[:-1], dim), ret_ctx


def swiglu_backward(dC, AB, scale=None, ctx: SwiGLUContext = None, sm_margin=0, use_aot=False):
    shape = dC.shape
    dim = shape[-1]
    dC = dC.view(-1, dim)

    dAB = torch.empty_like(AB)

    AB = AB.view(-1, dim * 2)
    dAB = dAB.view(-1, dim * 2)
    A = AB[:, :dim]
    B = AB[:, dim:]
    dA = dAB[:, :dim]
    dB = dAB[:, dim:]

    if scale is not None:
        dscale = torch.empty_like(scale)
    else:
        dscale = None

    n_rows, n_cols = dC.shape

    if ctx is None:
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    else:
        BLOCK_SIZE, num_warps = ctx.BLOCK_SIZE, ctx.num_warps

    sm_count = torch.cuda.get_device_properties(dC.device).multi_processor_count
    available_sms = sm_count - sm_margin
    available_sms = max(1, available_sms)

    if n_cols > BLOCK_SIZE:
        raise RuntimeError("This swiglu doesn't support feature dim >= 64KB.")

    if sm_margin > 0:
        rows_per_program = math.ceil(n_rows / available_sms)
        grid = (available_sms, )

        # Use the same memory for input and output gradients to save memory
        _swiglu_backward_kernel_persistent[grid](
            dC,
            dC.stride(0),
            A,
            A.stride(0),
            B,
            B.stride(0),
            scale,
            dA,
            dA.stride(0),
            dB,
            dB.stride(0),
            dscale,
            n_cols,
            n_rows,
            available_sms,
            rows_per_program,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    else:
        _swiglu_backward_kernel[(n_rows, )](
            dC,
            dC.stride(0),
            A,
            A.stride(0),
            B,
            B.stride(0),
            scale,
            dA,
            dA.stride(0),
            dB,
            dB.stride(0),
            dscale,
            n_cols,
            n_rows,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    return dAB, dscale
