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
P2P Bandwidth Test: Shape [8192, 4096] (64.00 MB)

CP Engine P2P Bandwidth Matrix (GB/s):
Src\Dst  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
--------------------------------------------------------------------------
 0      |   -   | 45.42 | 45.70 | 45.76 | 44.78 | 44.76 | 44.06 | 44.46 |
 1      | 45.51 |   -   | 45.25 | 45.80 | 45.08 | 44.94 | 44.26 | 44.49 |
 2      | 45.61 | 45.24 |   -   | 45.91 | 45.42 | 45.37 | 44.76 | 44.62 |
 3      | 45.75 | 45.56 | 45.88 |   -   | 45.14 | 45.49 | 44.76 | 44.92 |
 4      | 44.66 | 45.25 | 45.61 | 45.14 |   -   | 45.82 | 45.88 | 45.73 |
 5      | 44.42 | 44.83 | 45.50 | 45.53 | 45.98 |   -   | 45.43 | 45.65 |
 6      | 44.22 | 44.47 | 44.66 | 45.04 | 45.57 | 45.39 |   -   | 45.65 |
 7      | 44.40 | 44.48 | 44.68 | 44.90 | 45.58 | 45.67 | 45.72 |   -   |
--------------------------------------------------------------------------

Copy Kernel P2P Bandwidth Matrix (GB/s):
Src\Dst  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
--------------------------------------------------------------------------
 0      |   -   | 45.04 | 45.20 | 45.42 | 44.48 | 44.38 | 43.67 | 44.15 |
 1      | 45.14 |   -   | 44.84 | 45.44 | 44.65 | 44.57 | 43.90 | 44.12 |
 2      | 45.20 | 44.83 |   -   | 45.53 | 45.05 | 44.95 | 44.35 | 44.28 |
 3      | 45.32 | 45.10 | 45.49 |   -   | 44.75 | 45.15 | 44.37 | 44.57 |
 4      | 44.29 | 44.91 | 45.18 | 44.73 |   -   | 45.40 | 45.49 | 45.36 |
 5      | 44.06 | 44.43 | 45.15 | 45.20 | 45.57 |   -   | 45.00 | 45.28 |
 6      | 43.86 | 44.13 | 44.32 | 44.71 | 45.14 | 44.97 |   -   | 45.27 |
 7      | 44.01 | 44.18 | 44.37 | 44.55 | 45.14 | 45.31 | 45.36 |   -   |
--------------------------------------------------------------------------

AG Bandwidth Test
========================================================================================================================
Size (MB)  Shape           CP Engine (GB/s) (ms)     Copy Kernel (GB/s) (ms)     Torch AG (GB/s) (ms)    {'Correctness'}
========================================================================================================================
0.02       2x4096          0.43 (0.251)              20.23 (0.005)               2.25 (0.048)            CP:✓, Copy:✓
0.03       4x4096          0.87 (0.245)              37.34 (0.006)               4.86 (0.044)            CP:✓, Copy:✓
0.06       8x4096          1.58 (0.271)              82.87 (0.005)               7.12 (0.060)            CP:✓, Copy:✓
0.12       16x4096         3.41 (0.251)              117.78 (0.007)              13.78 (0.062)           CP:✓, Copy:✓
0.25       32x4096         6.08 (0.281)              151.98 (0.011)              26.43 (0.065)           CP:✓, Copy:✓
0.50       64x4096         13.94 (0.245)             182.39 (0.019)              49.40 (0.069)           CP:✓, Copy:✓
1.00       128x4096        25.67 (0.266)             192.16 (0.036)              80.95 (0.084)           CP:✓, Copy:✓
2.00       256x4096        49.17 (0.278)             176.79 (0.077)              113.05 (0.121)          CP:✓, Copy:✓
4.00       512x4096        99.25 (0.275)             212.46 (0.129)              163.31 (0.167)          CP:✓, Copy:✓
8.00       1024x4096       215.42 (0.254)            225.34 (0.243)              210.99 (0.259)          CP:✓, Copy:✓
16.00      2048x4096       247.84 (0.441)            242.75 (0.451)              243.72 (0.449)          CP:✓, Copy:✓
32.00      4096x4096       277.25 (0.789)            248.05 (0.882)              264.30 (0.828)          CP:✓, Copy:✓
64.00      8192x4096       296.97 (1.473)            252.47 (1.733)              275.87 (1.586)          CP:✓, Copy:✓
========================================================================================================================
"""

import torch.distributed
from python.triton_dist.utils import sleep_async
import torch
import torch.profiler
import argparse
import os
import datetime
from typing import List
import triton
import triton.language as tl
from hip import hip
from triton_dist.utils import HIP_CHECK
import pyrocshmem
from triton_dist.utils import group_profile, perf_func


@triton.jit
def copy_kernel_2d(
    src_tensor,
    dst_tensors_ptrs,
    barrier_ptrs,
    chunk_counters_ptr,
    rank,
    num_ranks,
    M,
    N,
    M_PER_CHUNK,
    stride_src_m,
    stride_src_n,
    stride_dst_m,
    stride_dst_n,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_rank = tl.program_id(axis=0)  # Which target rank this block handles
    pid_block = tl.program_id(axis=1)  # Which block within that rank's data

    if pid_rank >= num_ranks:
        return

    target_rank = pid_rank
    if target_rank >= rank:
        target_rank += 1

    dst_tensor = tl.load(dst_tensors_ptrs + target_rank).to(tl.pointer_type(dtype))
    dst_tensor = tl.multiple_of(dst_tensor, 16)

    blocks_per_rank_m = tl.cdiv(M, BLOCK_SIZE_M)
    blocks_per_rank_n = tl.cdiv(N, BLOCK_SIZE_N)
    blocks_per_rank = blocks_per_rank_m * blocks_per_rank_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    num_blocks_y = tl.num_programs(axis=1)
    for block_id in range(pid_block, blocks_per_rank, num_blocks_y):
        block_m = block_id // blocks_per_rank_n
        block_n = block_id % blocks_per_rank_n

        src_m_start = block_m * BLOCK_SIZE_M
        src_n_start = block_n * BLOCK_SIZE_N

        dst_m_start = rank * M + src_m_start
        dst_n_start = src_n_start

        src_ptrs = src_tensor + (src_m_start + offs_m[:, None]) * stride_src_m + \
                   (src_n_start + offs_n[None, :]) * stride_src_n
        dst_ptrs = dst_tensor + (dst_m_start + offs_m[:, None]) * stride_dst_m + \
                   (dst_n_start + offs_n[None, :]) * stride_dst_n

        mask = (src_m_start + offs_m[:, None] < M) & (src_n_start + offs_n[None, :] < N)

        data = tl.load(src_ptrs, mask=mask)
        tl.store(dst_ptrs, data, mask=mask)


def test_cp_engine_push_allgather_bandwidth(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensors: List[torch.Tensor],
    streams: List[torch.cuda.Stream],
    comm_buf_ptr: torch.Tensor,
    warmup_iters: int = 5,
    test_iters: int = 10,
):
    """Test CP Engine bandwidth using multi-stream push-mode AllGather."""
    local_tensor_size = local_tensor.numel() * local_tensor.element_size()
    my_data_offset_bytes = rank * local_tensor_size
    num_streams = len(streams)

    def run_allgather():
        current_stream = torch.cuda.current_stream()
        for s in streams:
            current_stream.wait_stream(s)
        for target_rank in range(num_ranks):
            if target_rank == rank:
                continue
            stream = streams[target_rank % num_streams]
            dest_ptr = remote_tensors[target_rank].data_ptr() + my_data_offset_bytes

            cp_res = hip.hipMemcpyAsync(
                dest_ptr,
                local_tensor.data_ptr(),
                local_tensor_size,
                hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
                stream.cuda_stream,
            )
            HIP_CHECK(cp_res)
        for s in streams:
            s.wait_stream(current_stream)

    # get data
    remote_tensors[rank].fill_(0)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    run_allgather()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    output = remote_tensors[rank].clone()

    sleep_async(200)
    _, latency = perf_func(run_allgather, iters=test_iters, warmup_iters=warmup_iters)
    tensor_size = local_tensor.numel() * local_tensor.element_size()
    total_bytes_received = tensor_size * (num_ranks - 1)
    bandwidth_gbps = (total_bytes_received / latency * 1000) / (1024**3) if latency > 0 else 0

    return bandwidth_gbps, latency, output


def test_copy_kernel_push_allgather_bandwidth(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensors: List[torch.Tensor],
    streams: List[torch.cuda.Stream],
    M_PER_CHUNK: int = 1024,
    warmup_iters: int = 5,
    test_iters: int = 10,
    num_sms: int = 56,
):
    """Test copy kernel bandwidth using 2D grid single kernel push-mode AllGather."""
    M, K = local_tensor.shape

    # Create array of tensor pointers
    dst_tensor_ptrs = torch.zeros(num_ranks, dtype=torch.int64, device='cuda')
    for i in range(num_ranks):
        dst_tensor_ptrs[i] = remote_tensors[i].data_ptr()

    # Use 2D grid: (num_ranks, num_sms // (num_ranks - 1))
    assert num_sms % (num_ranks - 1) == 0, "num_sms must be divisible by (num_ranks - 1)"
    grid = (num_ranks - 1, num_sms // (num_ranks - 1))
    M_PER_CHUNK = M  # Set M_PER_CHUNK to M

    def run_allgather():
        copy_kernel_2d[grid](
            local_tensor,
            dst_tensor_ptrs,
            None,
            None,
            rank,
            num_ranks,
            M,
            K,
            M_PER_CHUNK,
            local_tensor.stride(0),
            local_tensor.stride(1),
            remote_tensors[0].stride(0),
            remote_tensors[0].stride(1),
            dtype=tl.float16 if local_tensor.dtype == torch.float16 else tl.bfloat16,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
        )

    # get data
    remote_tensors[rank].fill_(0)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    run_allgather()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    output = remote_tensors[rank].clone()

    sleep_async(200)
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    _, latency = perf_func(run_allgather, iters=test_iters, warmup_iters=warmup_iters)
    tensor_size = local_tensor.numel() * local_tensor.element_size()
    total_bytes_received = tensor_size * (num_ranks - 1)
    bandwidth_gbps = (total_bytes_received / latency * 1000) / (1024**3) if latency > 0 else 0

    return bandwidth_gbps, latency, output


def test_torch_allgather_bandwidth(
    rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    tp_group,
    warmup_iters: int = 5,
    test_iters: int = 10,
):
    """Test PyTorch AllGather bandwidth as a reference."""
    global_tensor = torch.empty((local_tensor.shape[0] * num_ranks, *local_tensor.shape[1:]), dtype=local_tensor.dtype,
                                device=local_tensor.device)

    def run_allgather():
        torch.distributed.all_gather_into_tensor(global_tensor, local_tensor, group=tp_group)

    torch.cuda.synchronize()
    sleep_async(200)
    _, latency = perf_func(run_allgather, iters=test_iters, warmup_iters=warmup_iters)
    tensor_size = local_tensor.numel() * local_tensor.element_size()
    total_bytes_received = tensor_size * (num_ranks - 1)
    bandwidth_gbps = (total_bytes_received / latency * 1000) / (1024**3) if latency > 0 else 0

    return bandwidth_gbps, latency, global_tensor


def test_cp_engine_push_p2p_bandwidth(
    src_rank: int,
    dst_rank: int,
    local_tensor: torch.Tensor,
    remote_tensors: List[torch.Tensor],
    warmup_iters: int = 5,
    test_iters: int = 10,
):
    local_tensor_size = local_tensor.numel() * local_tensor.element_size()
    my_data_offset_bytes = src_rank * local_tensor_size

    def run_p2p():
        current_stream = torch.cuda.current_stream()
        dest_ptr = remote_tensors[dst_rank].data_ptr() + my_data_offset_bytes

        cp_res = hip.hipMemcpyAsync(
            dest_ptr,
            local_tensor.data_ptr(),
            local_tensor_size,
            hip.hipMemcpyKind.hipMemcpyDeviceToDeviceNoCU,
            current_stream.cuda_stream,
        )
        HIP_CHECK(cp_res)

    # get data
    M_per_rank = local_tensor.shape[0]
    remote_tensors[dst_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank].fill_(0)
    torch.cuda.synchronize()
    run_p2p()
    torch.cuda.synchronize()
    output = remote_tensors[dst_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank].clone()

    sleep_async(200)
    _, latency = perf_func(run_p2p, iters=test_iters, warmup_iters=warmup_iters)
    tensor_size = local_tensor.numel() * local_tensor.element_size()
    bandwidth_gbps = (tensor_size / latency * 1000) / (1024**3) if latency > 0 else 0
    assert torch.allclose(output, local_tensor)

    return bandwidth_gbps, latency, output


@triton.jit
def copy_kernel_p2p(
    src_tensor,
    dst_tensors_ptrs,
    rank,
    target_rank,
    M,
    N,
    M_PER_CHUNK,
    stride_src_m,
    stride_src_n,
    stride_dst_m,
    stride_dst_n,
    dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_block = tl.program_id(axis=0)
    dst_tensor = tl.load(dst_tensors_ptrs + target_rank).to(tl.pointer_type(dtype))
    dst_tensor = tl.multiple_of(dst_tensor, 16)

    blocks_per_rank_m = tl.cdiv(M, BLOCK_SIZE_M)
    blocks_per_rank_n = tl.cdiv(N, BLOCK_SIZE_N)
    blocks_per_rank = blocks_per_rank_m * blocks_per_rank_n

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    num_blocks_y = tl.num_programs(axis=0)
    for block_id in range(pid_block, blocks_per_rank, num_blocks_y):
        block_m = block_id // blocks_per_rank_n
        block_n = block_id % blocks_per_rank_n

        src_m_start = block_m * BLOCK_SIZE_M
        src_n_start = block_n * BLOCK_SIZE_N

        dst_m_start = rank * M + src_m_start
        dst_n_start = src_n_start

        src_ptrs = src_tensor + (src_m_start + offs_m[:, None]) * stride_src_m + \
                   (src_n_start + offs_n[None, :]) * stride_src_n
        dst_ptrs = dst_tensor + (dst_m_start + offs_m[:, None]) * stride_dst_m + \
                   (dst_n_start + offs_n[None, :]) * stride_dst_n

        mask = (src_m_start + offs_m[:, None] < M) & (src_n_start + offs_n[None, :] < N)

        data = tl.load(src_ptrs, mask=mask)
        tl.store(dst_ptrs, data, mask=mask)


def test_copy_kernel_p2p_bandwidth(
    src_rank: int,
    dst_rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    remote_tensors: List[torch.Tensor],
    M_PER_CHUNK: int = 1024,
    warmup_iters: int = 5,
    test_iters: int = 10,
    num_sms: int = 56,
):
    """Test copy kernel bandwidth using 2D grid single kernel push-mode AllGather."""
    M, K = local_tensor.shape

    # Create array of tensor pointers
    dst_tensor_ptrs = torch.zeros(num_ranks, dtype=torch.int64, device='cuda')
    for i in range(num_ranks):
        dst_tensor_ptrs[i] = remote_tensors[i].data_ptr()

    # Use 2D grid: (num_ranks, num_sms // (num_ranks - 1))
    assert num_sms % (num_ranks - 1) == 0, "num_sms must be divisible by (num_ranks - 1)"
    grid = (num_sms // (num_ranks - 1), )
    M_PER_CHUNK = M  # TODO: fix it

    def run_p2p():
        copy_kernel_p2p[grid](
            local_tensor,
            dst_tensor_ptrs,
            src_rank,
            dst_rank,
            M,
            K,
            M_PER_CHUNK,
            local_tensor.stride(0),
            local_tensor.stride(1),
            remote_tensors[0].stride(0),
            remote_tensors[0].stride(1),
            dtype=tl.float16 if local_tensor.dtype == torch.float16 else tl.bfloat16,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_N=256,
        )

    # get data
    M_per_rank = local_tensor.shape[0]
    remote_tensors[dst_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank].fill_(0)
    torch.cuda.synchronize()
    run_p2p()
    torch.cuda.synchronize()
    output = remote_tensors[dst_rank][src_rank * M_per_rank:(src_rank + 1) * M_per_rank].clone()

    sleep_async(200)
    _, latency = perf_func(run_p2p, iters=test_iters, warmup_iters=warmup_iters)
    tensor_size = local_tensor.numel() * local_tensor.element_size()
    bandwidth_gbps = (tensor_size / latency * 1000) / (1024**3) if latency > 0 else 0
    assert torch.allclose(output, local_tensor)

    return bandwidth_gbps, latency, output


def parse_args():
    parser = argparse.ArgumentParser(description="Bandwidth/Profiler test for Push-Mode AllGather")
    parser.add_argument("--M", type=int, default=8192, help="Matrix M dimension")
    parser.add_argument("--K", type=int, default=4096, help="Matrix K dimension (fixed to 4096 in size sweep)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for benchmark")
    parser.add_argument("--iters", type=int, default=10, help="Test iterations for benchmark")
    parser.add_argument("--num_sms", type=int, default=56, help="Number of SMs for copy kernel")
    parser.add_argument("--num_streams", type=int, default=8, help="Number of streams for multi-stream tests")
    parser.add_argument("--size_sweep", action="store_true", help="Run size sweep with a fixed list of M dimensions.")
    parser.add_argument("--check", action="store_true", help="Enable correctness check")
    parser.add_argument("--profile", action="store_true",
                        help="Enable PyTorch Profiler for a fixed size (M=1024, K=8192)")
    return parser.parse_args()


def generate_size_configs(dtype):
    """Generate tensor configs for a fixed list of M values with K=4096."""
    element_size = torch.tensor([], dtype=dtype).element_size()
    configs = []

    m_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    K = 4096

    for M in m_values:
        size_byte = M * K * element_size
        configs.append((M, K, size_byte))

    return configs


def run_ag_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, comm_buf_ptr, args):
    """Run bandwidth benchmark for a single tensor size"""
    torch.manual_seed(42 + RANK)
    local_tensor = torch.randn(M, K, dtype=dtype, device=torch.cuda.current_device())
    workspace_tensors = pyrocshmem.rocshmem_create_tensor_list_intra_node([M * WORLD_SIZE, K], dtype)
    num_streams = min(WORLD_SIZE, args.num_streams)
    multi_streams = [torch.cuda.Stream(priority=-1) for _ in range(num_streams)]

    tensor_size_mb = local_tensor.numel() * local_tensor.element_size() / (1024**2)

    prof = group_profile("bandwidth_test_ag", args.profile, group=TP_GROUP)
    with prof:
        torch_bandwidth, torch_time, torch_result = test_torch_allgather_bandwidth(RANK, WORLD_SIZE, local_tensor,
                                                                                   TP_GROUP, args.warmup, args.iters)
        cp_bandwidth, cp_time, cp_result = test_cp_engine_push_allgather_bandwidth(RANK, WORLD_SIZE, local_tensor,
                                                                                   workspace_tensors, multi_streams,
                                                                                   comm_buf_ptr, args.warmup,
                                                                                   args.iters)
        copy_bandwidth, copy_time, copy_result = test_copy_kernel_push_allgather_bandwidth(
            RANK, WORLD_SIZE, local_tensor, workspace_tensors, multi_streams, comm_buf_ptr, args.warmup, args.iters,
            args.num_sms)
    cp_correct = None
    copy_correct = None
    if args.check:
        # ignore local rank result
        torch_result.view(WORLD_SIZE, M, K)[RANK].fill_(0)
        cp_correct = torch.allclose(cp_result, torch_result, atol=1e-2, rtol=1e-2)
        copy_correct = torch.allclose(copy_result, torch_result, atol=1e-2, rtol=1e-2)
        if RANK == 0:
            print(f"  Correctness check - CP Engine: {'✓' if cp_correct else '✗'}, "
                  f"Copy Kernel: {'✓' if copy_correct else '✗'}")

    return {
        'size_mb': tensor_size_mb, 'shape': (M, K), 'cp_bandwidth': cp_bandwidth, 'copy_bandwidth': copy_bandwidth,
        'torch_bandwidth': torch_bandwidth, 'cp_time': cp_time, 'copy_time': copy_time, 'torch_time': torch_time,
        'cp_correct': cp_correct, 'copy_correct': copy_correct
    }


def run_p2p_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args):
    """
    Run a CORRECTED P2P bandwidth benchmark for a single tensor size for all GPU pairs.
    This version serializes the tests to avoid contention.
    """
    # Setup local data and shared memory workspace for remote writes
    torch.manual_seed(42 + RANK)
    local_tensor = torch.randn(M, K, dtype=dtype, device=torch.cuda.current_device())
    workspace_tensors = pyrocshmem.rocshmem_create_tensor_list_intra_node([M * WORLD_SIZE, K], dtype)

    tensor_size_mb = local_tensor.numel() * local_tensor.element_size() / (1024**2)
    if RANK == 0:
        print(f"\n--- P2P Bandwidth Test: Shape [{M}, {K}] ({tensor_size_mb:.2f} MB) ---")

    cp_bandwidths = torch.zeros(WORLD_SIZE, device='cuda', dtype=torch.float32)
    copy_bandwidths = torch.zeros(WORLD_SIZE, device='cuda', dtype=torch.float32)

    prof = group_profile("bandwidth_test_p2p", args.profile, group=TP_GROUP)
    with prof:
        # Iterate over all possible source and destination pairs to serialize the tests
        for src_rank in range(WORLD_SIZE):
            for dst_rank in range(WORLD_SIZE):
                if src_rank == dst_rank:
                    continue
                if RANK == src_rank:
                    # Test P2P using the CP Engine (hipMemcpyAsync)
                    cp_bw, _, _ = test_cp_engine_push_p2p_bandwidth(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        local_tensor=local_tensor,
                        remote_tensors=workspace_tensors,
                        warmup_iters=args.warmup,
                        test_iters=args.iters,
                    )
                    cp_bandwidths[dst_rank] = cp_bw

                    # Test P2P using the custom Triton copy kernel
                    copy_bw, _, _ = test_copy_kernel_p2p_bandwidth(
                        src_rank=src_rank,
                        dst_rank=dst_rank,
                        num_ranks=WORLD_SIZE,
                        local_tensor=local_tensor,
                        remote_tensors=workspace_tensors,
                        warmup_iters=args.warmup,
                        test_iters=args.iters,
                        num_sms=args.num_sms,
                    )
                    copy_bandwidths[dst_rank] = copy_bw

                torch.distributed.barrier(TP_GROUP)

    all_cp_bandwidths = torch.zeros(WORLD_SIZE, WORLD_SIZE, device='cuda', dtype=torch.float32)
    all_copy_bandwidths = torch.zeros(WORLD_SIZE, WORLD_SIZE, device='cuda', dtype=torch.float32)

    torch.distributed.all_gather_into_tensor(all_cp_bandwidths, cp_bandwidths.view(1, WORLD_SIZE))
    torch.distributed.all_gather_into_tensor(all_copy_bandwidths, copy_bandwidths.view(1, WORLD_SIZE))

    torch.distributed.barrier(TP_GROUP)

    if RANK == 0:

        def print_bw_matrix(title, matrix):
            print(f"\n{title} (GB/s):")
            label = "Src\Dst"
            header = f"{label:<9}|"
            for i in range(WORLD_SIZE):
                header += f" {i:^5} |"
            print(header)
            print("-" * len(header))
            for i in range(WORLD_SIZE):
                row_str = f" {i:<7}|"
                for j in range(WORLD_SIZE):
                    if i == j:
                        val_str = "  -  "
                    else:
                        val_str = f"{matrix[i, j]:<5.2f}"
                    row_str += f" {val_str} |"
                print(row_str)
            print("-" * len(header))

        print_bw_matrix("CP Engine P2P Bandwidth Matrix", all_cp_bandwidths.cpu())
        print_bw_matrix("Copy Kernel P2P Bandwidth Matrix", all_copy_bandwidths.cpu())


def main():
    args = parse_args()

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(backend="nccl", world_size=WORLD_SIZE, rank=RANK,
                                         timeout=datetime.timedelta(seconds=1800))

    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)))
    torch.distributed.barrier(TP_GROUP)

    pyrocshmem.init_rocshmem_by_uniqueid(TP_GROUP)

    comm_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([WORLD_SIZE], torch.int32)
    comm_bufs[RANK].fill_(0)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_bufs], device=torch.cuda.current_device(),
                                requires_grad=False)
    torch.cuda.synchronize()
    torch.distributed.barrier(TP_GROUP)

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    if args.profile:
        M, K = args.M, args.K
        run_ag_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, comm_buf_ptr, args)
        run_p2p_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

    elif args.size_sweep:
        size_configs = generate_size_configs(dtype)

        for i, (M, K, size_bytes) in enumerate(size_configs):
            run_p2p_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

        results = []
        if RANK == 0:
            print(f"Running Push-Mode AllGather bandwidth sweep (fixed K=4096, dtype={args.dtype})")
        for i, (M, K, size_bytes) in enumerate(size_configs):
            if RANK == 0:
                print(f"\nTesting config {i+1}/{len(size_configs)}: [{M}, {K}] ({size_bytes/(1024**2):.2f} MB)")
            result = run_ag_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, comm_buf_ptr, args)
            results.append(result)

        if RANK == 0:
            header = f"{'Size (MB)':<10} {'Shape':<15} {'CP Engine (GB/s) (ms)':<25} {'Copy Kernel (GB/s) (ms)':<27} {'Torch AG (GB/s) (ms)':<23}"
            if args.check: header += " {'Correctness'}"
            print(f"\n{'='*len(header)}")
            print(header)
            print(f"{'='*len(header)}")
            for result in results:
                shape_str = f"{result['shape'][0]}x{result['shape'][1]}"
                cp_str = f"{result['cp_bandwidth']:.2f} ({result['cp_time']:.3f})"
                copy_str = f"{result['copy_bandwidth']:.2f} ({result['copy_time']:.3f})"
                torch_str = f"{result['torch_bandwidth']:.2f} ({result['torch_time']:.3f})"
                row = f"{result['size_mb']:<10.2f} {shape_str:<15} {cp_str:<25} {copy_str:<27} {torch_str:<23}"
                if args.check and result['cp_correct'] is not None:
                    cp_status = "✓" if result['cp_correct'] else "✗"
                    copy_status = "✓" if result['copy_correct'] else "✗"
                    row += f" CP:{cp_status}, Copy:{copy_status}"
                print(row)
            print(f"{'='*len(header)}")
    else:
        if RANK == 0:
            print(f"Running Push-Mode AllGather bandwidth test with shape [{args.M}, {args.K}], dtype={args.dtype}")
        result = run_ag_single_test(args.M, args.K, dtype, RANK, WORLD_SIZE, TP_GROUP, comm_buf_ptr, args)
        run_p2p_single_test(args.M, args.K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

        if RANK == 0:
            header = f"{'Method':<20} {'Bandwidth (GB/s) (Latency ms)':<35}"
            if args.check: header += " {'Correct'}"
            print(f"\n{'='*len(header)}")
            print(header)
            print(f"{'='*len(header)}")
            cp_str = f"{result['cp_bandwidth']:.2f} ({result['cp_time']:.3f})"
            copy_str = f"{result['copy_bandwidth']:.2f} ({result['copy_time']:.3f})"
            torch_str = f"{result['torch_bandwidth']:.2f} ({result['torch_time']:.3f})"
            row = f"{'CP Engine Push':<20} {cp_str:<35}"
            if args.check and result['cp_correct'] is not None: row += f" {'✓' if result['cp_correct'] else '✗'}"
            print(row)
            row = f"{'Copy Kernel Push':<20} {copy_str:<35}"
            if args.check and result['copy_correct'] is not None: row += f" {'✓' if result['copy_correct'] else '✗'}"
            print(row)
            row = f"{'Torch AllGather':<20} {torch_str:<35}"
            if args.check: row += " reference"
            print(row)
            print(f"{'='*len(header)}")

    pyrocshmem.rocshmem_finalize()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
