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
AMD EP A2A kernels and helpers.
Provides bincount and re-exports intra-node kernels from ep_a2a_intra_node.
"""

import torch
import triton.language as tl
import triton_dist
from triton_dist.language.extra.hip.language_extra import tid, ld, atomic_add
from triton_dist.language.extra.language_extra import threads_per_warp


@triton_dist.jit(do_not_specialize=["n", "length", "num_sm"])
def kernel_bincount(n, input, output, length, num_sm, num_warps: tl.constexpr):
    """
    GPU bincount: count occurrences of each index in [0, length). AMD version using tid(0)
    and fixed threads_per_block (no simt_exec_region). Same semantics as nvidia/ep_a2a.py.
    """
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    threads_per_block = num_warps * threads_per_warp()
    for i in range(pid * threads_per_block + thread_idx, n, num_pid * threads_per_block):
        val = ld(input + i)
        if val < length:
            atomic_add(output + val, 1, scope="agent", semantic="relaxed")


def bincount(input_tensor, length, output=None, output_dtype=torch.int32, num_sm=16, num_warps=8):
    """GPU bincount for AMD (no AOT). input_tensor: 1D int32 on device; output: length elements."""
    if output is None:
        output = torch.zeros(length, dtype=output_dtype, device=input_tensor.device)
    assert input_tensor.dim() == 1 and input_tensor.is_contiguous()
    assert output.size(0) >= length and output.dtype == output_dtype
    n = input_tensor.size(0)
    grid = (num_sm, )
    kernel_bincount[grid](n, input_tensor, output, length, num_sm, num_warps=num_warps)
    return output


# Re-export intra-node kernels and helpers so layer can import from this module only.
from triton_dist.kernels.amd.ep_a2a_intra_node import (
    kernel_combine_token_intra_node,
    kernel_dispatch_token_intra_node,
    get_ag_splits_and_recv_offset_for_dispatch_intra_node,
    kernel_skipped_token_local_dispatch_intra_node,
    kernel_skipped_token_inplace_local_combine_intra_node,
)

__all__ = [
    "kernel_bincount",
    "bincount",
    "kernel_combine_token_intra_node",
    "kernel_dispatch_token_intra_node",
    "get_ag_splits_and_recv_offset_for_dispatch_intra_node",
    "kernel_skipped_token_local_dispatch_intra_node",
    "kernel_skipped_token_inplace_local_combine_intra_node",
]
