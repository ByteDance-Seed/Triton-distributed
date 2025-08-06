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
import sys
import os
from typing import List

import torch
import torch.distributed

import triton
import triton.language as tl
from triton.language.extra.cuda.language_extra import (
    tid,
    multimem_ld_reduce_v4,
    st_v4_b32,
)
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import (
    finalize_distributed,
    initialize_distributed,
    nvshmem_barrier_all_on_stream,
    nvshmem_free_tensor_sync,
    nvshmem_create_tensor,
    is_nvshmem_multimem_supported,
)
from triton.language.extra.cuda.utils import num_warps


@triton.jit
def _multimem_ld_reduce_kernel(symm_in_ptr, out_ptr, elems, ACC_DTYPE: tl.constexpr):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)
    pid = tl.program_id(0)
    thread_idx = tid(axis=0)
    num_pid = tl.num_programs(axis=0)

    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_in_ptr)
    VEC_SIZE = 128 // tl.constexpr(symm_in_ptr.dtype.element_ty.primitive_bitwidth)

    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE, ACC_DTYPE)
        st_v4_b32(out_ptr + idx * VEC_SIZE, val0, val1, val2, val3)


def run_multimem_ld_reduce(symm_tensor: torch.Tensor, acc_dtype: torch.dtype, num_grids=4, num_warps=32):
    out = torch.empty_like(symm_tensor)
    current_stream = torch.cuda.current_stream()
    nvshmem_barrier_all_on_stream(current_stream)
    _multimem_ld_reduce_kernel[(num_grids, )](symm_tensor, out, symm_tensor.numel(), {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }[acc_dtype], num_warps=num_warps)
    nvshmem_barrier_all_on_stream(current_stream)
    return out


def run_reduce(t: torch.Tensor, reduce_order: List[int], acc_dtype: torch.dtype = None):
    gathered = torch.empty(
        (WORLD_SIZE, ) + t.shape,
        dtype=t.dtype,
        device=t.device,
    )
    torch.distributed.all_gather_into_tensor(gathered, t, group=TP_GROUP)
    acc_dtype = acc_dtype or t.dtype
    gathered = gathered.to(acc_dtype)
    out = torch.zeros_like(t).to(acc_dtype)
    for n in reduce_order:
        out += gathered[n]
    return out.to(x.dtype)


def run_reduce_in_order(t: torch.Tensor, acc_dtype: torch.dtype = None):
    return run_reduce(t, list(range(WORLD_SIZE)), acc_dtype)


def run_reduce_ring(t: torch.Tensor, acc_dtype: torch.dtype = None):
    reduce_order = list(range(RANK, WORLD_SIZE)) + list(range(0, RANK))
    return run_reduce(t, reduce_order, acc_dtype)


def is_bitwise_match(x: torch.Tensor, y: torch.Tensor):
    assert x.dtype == y.dtype and x.numel() == y.numel()
    return bool((x.view(torch.int8) == y.view(torch.int8)).sum() == x.nbytes)


def _is_all_ranks_bitwise_match(t: torch.Tensor):
    gathered = torch.empty(
        (WORLD_SIZE, ) + t.shape,
        dtype=t.dtype,
        device=t.device,
    )
    torch.distributed.all_gather_into_tensor(gathered, t, group=TP_GROUP)
    return all([is_bitwise_match(gathered[0], gathered[n]) for n in range(1, WORLD_SIZE)])


def compare_ld_reduce_precision(symm_tensor: torch.Tensor, acc_dtype: torch.dtype):
    if symm_tensor.dtype == torch.bfloat16:
        assert acc_dtype in [torch.bfloat16, torch.float32]
    out_in_order = run_reduce_in_order(symm_tensor, acc_dtype)
    out_ring = run_reduce_ring(symm_tensor, acc_dtype)
    out_multimem = run_multimem_ld_reduce(symm_tensor, acc_dtype)

    print("reduce in order 0+1+2+3+...")
    print(out_in_order)
    print("is bitwise match cross ranks: ", _is_all_ranks_bitwise_match(out_in_order))

    print("reduce in ring order")
    print(out_ring)
    print("is bitwise match cross ranks: ", _is_all_ranks_bitwise_match(out_ring))

    print("reduce with multimem.ld_reduce")
    print(out_multimem)
    print("is bitwise match cross ranks: ", _is_all_ranks_bitwise_match(out_multimem))

    print(is_bitwise_match(out_in_order, out_ring))
    print(is_bitwise_match(out_in_order, out_multimem))

    # is bitwise with the same data in many iters
    for n in range(10000):
        out = run_multimem_ld_reduce(symm_tensor, acc_dtype)
        if not is_bitwise_match(out_multimem, out):
            print(f"❌ multimem.ld_reduce not bitwise match for iter {n}")
            break
    else:
        print("✅ multimem.ld_reduce bitwise match for all iter")

    # is bitwise with the same data but not the same grids in many iters
    for n in range(1024):
        num_grids = n + 1
        out = run_multimem_ld_reduce(symm_tensor, acc_dtype, num_grids=num_grids)
        if not is_bitwise_match(out_multimem, out):
            print(f"❌ multimem.ld_reduce not bitwise match for iter {num_grids}")
            break
    else:
        print("✅ multimem.ld_reduce bitwise match for num_grids=1~1024")

    # is bitwise with the same data but not the same grids in many iters
    for nwarps in [1, 2, 4, 8, 16, 32]:
        out = run_multimem_ld_reduce(symm_tensor, acc_dtype, num_warps=nwarps)
        if not is_bitwise_match(out_multimem, out):
            print(f"❌ multimem.ld_reduce not bitwise match for num_warps {nwarps}")
            break
    else:
        print("✅ multimem.ld_reduce bitwise match num_warps=1~32")

    # torch.save(out_multimem, f"out_multimem_{RANK}_{ld_reduce_acc_dtype}.pt")


if __name__ == "__main__":
    if not is_nvshmem_multimem_supported():
        print("nvshmem multimem is not supported, skip...")
        sys.exit(0)

    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))

    TP_GROUP = initialize_distributed()

    val = 1024 if RANK == 0 else 1
    N = 8 * 1024 * 1024  # with num_warps=32, with 1024 CTAs
    x = nvshmem_create_tensor((N, ), torch.bfloat16)
    t_val = torch.ones((N, ), device="cuda", dtype=torch.bfloat16) * val
    x.copy_(t_val)
    current_stream = torch.cuda.current_stream()
    nvshmem_barrier_all_on_stream(current_stream)

    print("acc with float16 precision...")
    compare_ld_reduce_precision(x, torch.bfloat16)

    print("acc with float32 precision...")
    compare_ld_reduce_precision(x, torch.float32)

    nvshmem_free_tensor_sync(x)
    finalize_distributed()
