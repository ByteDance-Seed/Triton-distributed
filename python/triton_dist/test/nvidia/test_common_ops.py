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
import os
import random
import pytest

import torch
import triton
import triton.language as tl

from triton_dist.kernels.nvidia.common_ops import (barrier_all_intra_node_non_atomic,
                                                   barrier_all_intra_node_non_atomic_block, barrier_on_this_grid,
                                                   barrier_all_intra_node_atomic_cas_block,
                                                   cooperative_barrier_on_this_grid, bisect_left_kernel,
                                                   bisect_left_kernel_aligned, bisect_right_kernel,
                                                   bisect_right_kernel_aligned)
from triton_dist.utils import supports_p2p_native_atomic, finalize_distributed, initialize_distributed, launch_cooperative_grid_options, nvshmem_barrier_all_on_stream, nvshmem_free_tensor_sync, nvshmem_create_tensor, sleep_async, support_launch_cooperative_grid

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))


def _random_sleep():
    if random.random() > 0.9:
        sleep_async(int(random.random() * 100))
    elif random.random() > 0.5:
        sleep_async(int(random.random() * 30))


def test_barrier_on_this_grid():
    print(">> barrier_on_this_grid start...")
    flag = torch.zeros((1, ), dtype=torch.int32, device="cuda")
    for _ in range(100):
        barrier_on_this_grid[(random.randint(1, 1024), )](flag, **launch_cooperative_grid_options())
    print("✅ barrier_on_this_grid passed")

    for _ in range(100):
        cooperative_barrier_on_this_grid[(random.randint(1, 1024), )](**launch_cooperative_grid_options())
    print("✅ cooperative_barrier_on_this_grid passed")

    # If launch_cooperative_grid is False, then it should raise an error.
    if support_launch_cooperative_grid():
        with pytest.raises(RuntimeError, match="an illegal memory access was encountered"):
            cooperative_barrier_on_this_grid[(random.randint(1, 1024), )](launch_cooperative_grid=False)
            torch.cuda.synchronize()
        print("✅ cooperative_barrier_on_this_grid with launch_cooperative_grid=False passed")


def test_barrier_all_intra_node_non_atomic():
    print(">> barrier_all_intra_node_non_atomic start...")
    symm_flag = nvshmem_create_tensor((LOCAL_WORLD_SIZE * 3, ), torch.int32)
    symm_flag.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    for n in range(1000):
        _random_sleep()
        # print(f"iter {n}", flush=True)
        barrier_all_intra_node_non_atomic_block[(1, )](LOCAL_RANK, LOCAL_WORLD_SIZE, symm_flag, n + 1)

    print("✅ barrier_all_intra_node_non_atomic_block passed")

    for n in range(1000):
        _random_sleep()
        # print(f"iter {n}", flush=True)
        barrier_all_intra_node_non_atomic[(random.randint(1, 1024), )](LOCAL_RANK, RANK, LOCAL_WORLD_SIZE, symm_flag,
                                                                       n + 1, use_cooperative=True)

    print("✅ barrier_all_intra_node_non_atomic passed")
    nvshmem_free_tensor_sync(symm_flag)


def test_barrier_all_intra_node():
    if not supports_p2p_native_atomic():
        print("P2P native atomic access is not supported. skip this test...")
        return

    print(">> barrier_all_intra_node_atomic_cas_block start...")
    symm_flag = nvshmem_create_tensor((LOCAL_WORLD_SIZE, ), torch.int32)
    symm_flag.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    for n in range(1000):
        _random_sleep()
        barrier_all_intra_node_atomic_cas_block[(1, )](LOCAL_RANK, RANK, LOCAL_WORLD_SIZE, symm_flag)

    print("✅ barrier_all_intra_node_atomic_cas_block passed")
    nvshmem_free_tensor_sync(symm_flag)


def bisect_triton(sorted_tensor, values_tensor, side="left", aligned=False):

    @triton.jit
    def _bisect_kernel(
        output_ptr,
        sorted_values_ptr,
        target_values_ptr,
        M,
        N: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        func: tl.constexpr,
    ):  # Get thread ID and total threads
        pid = tl.program_id(0)
        offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offset < M  # Assuming N is number of values to search

        # Load value for this thread
        target_values = tl.load(target_values_ptr + offset, mask=mask)

        tl.store(
            output_ptr + offset,
            func(sorted_values_ptr, target_values, N),
            mask=mask,
        )

    assert sorted_tensor.dim() == 1, "Sorted array must be 1D"
    assert values_tensor.dim() == 1, "Values array must be 1D"

    n = values_tensor.size(0)
    output = torch.empty_like(values_tensor, dtype=torch.int64)

    # Calculate launch parameters
    BLOCK_SIZE = 32
    grid = (triton.cdiv(n, BLOCK_SIZE), )
    func = {
        ("left", False): bisect_left_kernel,
        ("right", False): bisect_right_kernel,
        ("left", True): bisect_left_kernel_aligned,
        ("right", True): bisect_right_kernel_aligned,
    }[(side, aligned)]

    N = sorted_tensor.size(0)

    _bisect_kernel[grid](
        output,
        sorted_tensor,
        values_tensor,
        M=values_tensor.size(0),
        N=N,
        func=func,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=min(32, BLOCK_SIZE // 32),
    )
    return output


def test_bisect_cases():

    def _test_bisect(side="left", aligned=False):
        # Generate test data
        sorted_tensor = torch.tensor(range(1, 33, 1), device="cuda", dtype=torch.int32)

        values = torch.randint(0, 36, (32, ), device="cuda", dtype=torch.int32)
        values = torch.arange(0, 64, dtype=torch.int32, device="cuda")

        # Triton implementation
        triton_result = bisect_triton(sorted_tensor, values, side=side, aligned=aligned)

        # Reference implementation
        torch_result = torch.searchsorted(sorted_tensor, values, side=side)

        # Verify results
        assert torch.allclose(triton_result, torch_result), f"Triton: {triton_result}\nTorch: {torch_result}"

        print(f"bisect_{side} {'aligned' if aligned else ''} passed!")

    _test_bisect(side="right", aligned=False)
    _test_bisect(side="right", aligned=True)
    _test_bisect(side="left", aligned=False)
    _test_bisect(side="left", aligned=True)


if __name__ == "__main__":
    torch.cuda.set_device(LOCAL_RANK)
    initialize_distributed()

    test_barrier_all_intra_node_non_atomic()
    test_barrier_all_intra_node()

    # this test corrupt the CUDA context. leave it in last
    test_barrier_on_this_grid()
    finalize_distributed()

    test_bisect_cases()
