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
MoRI SHMEM Bandwidth Test For P2P And IBGDA
==============================
Example usage:
  # P2P TEST
  torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_endpoint=127.0.0.1:23457 ./python/triton_dist/test/amd/test_mori_shmem_bw.py
  
  # IBGDA TEST
  torchrun --node_rank=0 --nproc_per_node=8 --nnodes=1 --rdzv_endpoint=127.0.0.1:23457 ./python/triton_dist/test/amd/test_mori_shmem_bw.py --test_ibgda
  
Baseline:
    --- MoRI SHMEM Bandwidth Test: Shape [8192, 4096] (64.00 MB) ---
    # P2P TEST on MI308
    MoRI SHMEM P2P Bandwidth Matrix (GB/s):
    Src\Dst  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
    --------------------------------------------------------------------------
      0      |   -   | 47.71 | 47.49 | 47.47 | 46.12 | 46.34 | 46.81 | 47.14 |
      1      | 47.74 |   -   | 47.69 | 47.67 | 46.62 | 46.58 | 46.88 | 46.63 |
      2      | 47.63 | 47.67 |   -   | 47.66 | 46.52 | 47.43 | 47.44 | 47.50 |
      3      | 47.41 | 47.78 | 47.64 |   -   | 46.64 | 46.98 | 47.49 | 47.38 |
      4      | 46.35 | 46.64 | 46.70 | 46.56 |   -   | 47.44 | 47.22 | 47.72 |
      5      | 46.06 | 46.25 | 47.30 | 46.84 | 47.46 |   -   | 47.78 | 47.37 |
      6      | 46.96 | 46.79 | 47.20 | 47.44 | 47.22 | 47.78 |   -   | 47.76 |
      7      | 46.87 | 46.87 | 47.96 | 47.43 | 47.71 | 47.38 | 47.67 |   -   |
    --------------------------------------------------------------------------

    # IBGDA TEST on MI308 + BRCM dual-port 400Gb Thor2 NIC
    MoRI SHMEM IBGDA Bandwidth Matrix (GB/s):
    Src\Dst  |   0   |   1   |   2   |   3   |   4   |   5   |   6   |   7   |
    --------------------------------------------------------------------------
      0      |   -   | 43.84 | 43.83 | 43.83 | 43.90 | 43.83 | 43.89 | 43.75 |
      1      | 43.78 |   -   | 43.88 | 43.88 | 43.83 | 43.90 | 43.81 | 43.89 |
      2      | 43.85 | 43.89 |   -   | 43.90 | 43.87 | 43.93 | 43.87 | 43.94 |
      3      | 43.97 | 43.88 | 43.96 |   -   | 43.88 | 43.90 | 43.91 | 43.86 |
      4      | 43.95 | 43.82 | 43.92 | 43.73 |   -   | 43.85 | 43.89 | 43.88 |
      5      | 43.90 | 43.84 | 43.91 | 43.84 | 43.89 |   -   | 43.93 | 43.77 |
      6      | 43.85 | 43.83 | 43.91 | 43.84 | 43.89 | 43.86 |   -   | 43.78 |
      7      | 43.87 | 43.92 | 43.85 | 43.86 | 43.82 | 43.99 | 43.89 |   -   |
    --------------------------------------------------------------------------
"""

import os
import sys
import argparse

# Set default environment variables
os.environ.setdefault('TRITON_DIST_SHMEM_BACKEND', 'mori_shmem')
os.environ["MORI_SHMEM_HEAP_SIZE"] = "4G"

_test_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_test_dir, "../../../.."))
_triton_dist_python_path = os.path.join(_workspace_root, "python")
if _triton_dist_python_path not in sys.path:
    sys.path.insert(0, _triton_dist_python_path)
current_pythonpath = os.environ.get('PYTHONPATH', '')
if _triton_dist_python_path not in current_pythonpath:
    os.environ[
        'PYTHONPATH'] = f"{_triton_dist_python_path}:{current_pythonpath}" if current_pythonpath else _triton_dist_python_path

# Add upstream Triton python path
_triton_python_path = os.path.join(_workspace_root, "3rdparty/triton/python")
if os.path.exists(_triton_python_path):
    sys.path.insert(0, _triton_python_path)

import torch
import torch.distributed

from triton_dist.utils import initialize_distributed
from triton_dist.profiler_utils import perf_func, group_profile
import mori.shmem as mori_shmem
from mori.shmem import (
    mori_shmem_create_tensor,
)
import triton_dist
import triton.language as tl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import threads_per_warp
from triton_dist.language.extra.hip.language_extra import tid


@triton_dist.jit
def mori_putmem_p2p_kernel(
    src_tensor,
    dst_tensor,
    rank,
    target_rank,
    M,
    N,
    stride_src_m,
    stride_src_n,
    stride_dst_m,
    stride_dst_n,
    num_qps: tl.constexpr,
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_warps: tl.constexpr,
    test_ibgda: tl.constexpr,
):
    """
    MoRI SHMEM P2P kernel using putmem_nbi (thread-level)
    Each thread independently transfers a chunk of data
    
    Memory layout:
    - src_tensor: local data [M, N]
    - dst_tensor: full symmetric memory [WORLD_SIZE * M, N]
    - Each PE writes to dst_tensor[rank * M : (rank+1) * M, :] on target PE
    """
    WARP_SIZE = threads_per_warp()
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    thread_idx = tid(0)
    warp_id = thread_idx // WARP_SIZE
    lane_id = thread_idx % WARP_SIZE

    # Total threads and warps across all blocks
    threads_per_block = num_warps * WARP_SIZE
    total_threads = num_pid * threads_per_block
    global_thread_id = pid * threads_per_block + thread_idx
    global_warp_id = global_thread_id // WARP_SIZE

    # Total elements to transfer
    total_elements = M * N

    # Element size in bytes
    elem_size = tl.constexpr(2 if dtype == tl.float16 else 4)

    # Each thread handles multiple chunks with stride
    for elem_offset in range(global_thread_id * BLOCK_SIZE, total_elements, total_threads * BLOCK_SIZE):
        # Calculate how many elements this thread will process
        remaining = total_elements - elem_offset
        chunk_size = tl.minimum(remaining, BLOCK_SIZE)

        if chunk_size > 0:
            # start_row = elem_offset // N
            # start_col = elem_offset % N
            # src_ptr = src_tensor + start_row * stride_src_m + start_col * stride_src_n
            # dst_row = rank * M + start_row
            # dst_ptr = dst_tensor + dst_row * stride_dst_m + start_col * stride_dst_n

            src_ptr = src_tensor + elem_offset
            dst_offset = rank * M * N + elem_offset
            dst_ptr = dst_tensor + dst_offset
            # Calculate byte size for this chunk
            nbytes = chunk_size * elem_size

            # QP selection based on thread ID
            qp_id = (global_thread_id // WARP_SIZE) % num_qps

            # Each thread independently calls putmem_nbi
            libshmem_device.putmem_nbi(dst_ptr, src_ptr, nbytes, target_rank, qp_id)
    if test_ibgda:
        if global_warp_id == 0:
            libshmem_device.quiet_pe(target_rank)


def print_bw_matrix(title: str, matrix: torch.Tensor, world_size: int):
    print(f"\n{title} (GB/s):")
    label = "Src\\Dst"
    header = f"{label:<9}|"
    for i in range(world_size):
        header += f" {i:^5} |"
    print(header)
    print("-" * len(header))

    for i in range(world_size):
        row_str = f"  {i:<7}|"
        for j in range(world_size):
            if i == j:
                val_str = "  -  "
            else:
                bw = matrix[i][j].item()
                val_str = f"{bw:5.2f}"
            row_str += f" {val_str} |"
        print(row_str)
    print("-" * len(header))


def test_mori_shmem_bandwidth(
    src_rank: int,
    dst_rank: int,
    num_ranks: int,
    local_tensor: torch.Tensor,
    symm_tensor: torch.Tensor,
    num_qps: int = 4,
    num_sms: int = 56,
    num_warps: int = 8,
    BLOCK_SIZE: int = 8,
    warmup_iters: int = 5,
    test_iters: int = 10,
    test_ibgda: bool = False,
):
    """
    Test MoRI SHMEM P2P bandwidth using putmem_nbi
    
    Args:
        src_rank: Source PE rank
        dst_rank: Destination PE rank
        num_ranks: Total number of PEs
        local_tensor: Local data slice to send (view of symm_tensor)
        symm_tensor: Full symmetric tensor [M * WORLD_SIZE, K]
        num_qps: Number of QPs for MoRI SHMEM
        num_sms: Number of SMs for kernel launch
        num_warps: Number of warps per block (default: 8)
        BLOCK_SIZE: Elements per thread per iteration (default: 8)
        warmup_iters: Warmup iterations
        test_iters: Test iterations
    """
    M, K = local_tensor.shape

    # Grid configuration: use multiple blocks
    grid = (num_sms, )

    # grid = (16, )

    def run_p2p():
        mori_putmem_p2p_kernel[grid](
            local_tensor,
            symm_tensor,  # Full symmetric tensor as base
            src_rank,
            dst_rank,
            M,
            K,
            local_tensor.stride(0),
            local_tensor.stride(1),
            symm_tensor.stride(0),
            symm_tensor.stride(1),
            num_qps=num_qps,
            dtype=tl.float16 if local_tensor.dtype == torch.float16 else tl.bfloat16,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            test_ibgda=test_ibgda,
        )

    # Clear target region first (data will be written to dst_rank's region in dst PE's symmetric memory)
    # Note: We're clearing src_rank's slice on the dst PE, which will receive the data
    M_per_rank = M
    symm_tensor[src_rank * M_per_rank:(src_rank + 1) * M_per_rank].fill_(0)
    torch.cuda.synchronize()

    # Run once to get output
    run_p2p()
    torch.cuda.synchronize()

    # After putmem, data from src_rank should appear in src_rank's slice of dst PE's symm_tensor
    output = symm_tensor[src_rank * M_per_rank:(src_rank + 1) * M_per_rank].clone()

    # Benchmark
    torch.cuda.synchronize()
    _, latency = perf_func(run_p2p, iters=test_iters, warmup_iters=warmup_iters)
    tensor_size = local_tensor.numel() * local_tensor.element_size()
    bandwidth_gbps = (tensor_size / latency * 1000) / (1024**3) if latency > 0 else 0

    # Verify correctness
    assert torch.allclose(output, local_tensor, atol=1e-2, rtol=1e-2), \
        f"Data mismatch between src_rank={src_rank} and dst_rank={dst_rank}"

    return bandwidth_gbps, latency, output


def run_p2p_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args):
    # Create a single symmetric memory tensor for all ranks [M * WORLD_SIZE, K]
    # All operations are performed directly on this symmetric tensor
    symm_tensor = mori_shmem_create_tensor([M * WORLD_SIZE, K], dtype)

    # Initialize local source data in symmetric memory
    # Each rank writes to its own slice: [RANK * M : (RANK + 1) * M, :]
    torch.manual_seed(42 + RANK)
    local_tensor = symm_tensor[RANK * M:(RANK + 1) * M, :]
    local_tensor.copy_(torch.randn(M, K, dtype=dtype, device='cuda'))

    tensor_size_mb = local_tensor.numel() * local_tensor.element_size() / (1024**2)
    assert args.num_sms % (WORLD_SIZE - 1) == 0, "num_sms must be divisible by (WORLD_SIZE - 1)"
    num_sms = args.num_sms // (WORLD_SIZE - 1) * 2
    num_warps_per_sm = 8
    if RANK == 0:
        print(f"\n--- MoRI SHMEM P2P Bandwidth Test: Shape [{M}, {K}] ({tensor_size_mb:.2f} MB) ---")
        print(f"    num_sms={num_sms},   num_warps_per_sm={num_warps_per_sm}")

    mori_bandwidths = torch.zeros(WORLD_SIZE, device='cuda', dtype=torch.float32)

    prof = group_profile("mori_shmem_p2p_bw", args.profile, group=TP_GROUP)
    with prof:
        for src_rank in range(WORLD_SIZE):
            for dst_rank in range(WORLD_SIZE):
                if src_rank == dst_rank:
                    continue

                if RANK == src_rank:
                    bandwidth, latency, _ = test_mori_shmem_bandwidth(
                        src_rank,
                        dst_rank,
                        WORLD_SIZE,
                        local_tensor,
                        symm_tensor,
                        num_qps=args.num_qps,
                        num_sms=num_sms,
                        num_warps=num_warps_per_sm,
                        BLOCK_SIZE=8,
                        warmup_iters=args.warmup,
                        test_iters=args.iters,
                        test_ibgda=False,
                    )
                    mori_bandwidths[dst_rank] = bandwidth
                    print(f"[Rank {RANK}] PE{src_rank} -> PE{dst_rank}: "
                          f"{bandwidth:.2f} GB/s ({latency:.3f} ms)")

                torch.distributed.barrier(TP_GROUP)

    # Gather all bandwidth results
    all_mori_bandwidths = torch.zeros(WORLD_SIZE, WORLD_SIZE, device='cuda', dtype=torch.float32)
    torch.distributed.all_gather_into_tensor(all_mori_bandwidths, mori_bandwidths.view(1, WORLD_SIZE))
    torch.distributed.barrier(TP_GROUP)

    if RANK == 0:
        print_bw_matrix("MoRI SHMEM P2P Bandwidth Matrix", all_mori_bandwidths, WORLD_SIZE)


def run_ibgda_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args):
    """
    IBGDA (In-Place Broadcast Gather with Direct Addressing) bandwidth test
    Similar to P2P test but can be configured with different parameters
    """
    # Create a single symmetric memory tensor for all ranks [M * WORLD_SIZE, K]
    # All operations are performed directly on this symmetric tensor
    symm_tensor = mori_shmem_create_tensor([M * WORLD_SIZE, K], dtype)

    # Initialize local source data in symmetric memory
    # Each rank writes to its own slice: [RANK * M : (RANK + 1) * M, :]
    torch.manual_seed(42 + RANK)
    local_tensor = symm_tensor[RANK * M:(RANK + 1) * M, :]
    local_tensor.copy_(torch.randn(M, K, dtype=dtype, device='cuda'))

    tensor_size_mb = local_tensor.numel() * local_tensor.element_size() / (1024**2)

    assert args.num_sms % (WORLD_SIZE - 1) == 0, "num_sms must be divisible by (WORLD_SIZE - 1)"
    num_sms = args.num_sms // (WORLD_SIZE - 1) // 4
    num_warps_per_sm = 4
    if RANK == 0:
        print(f"\n--- MoRI SHMEM IBGDA Bandwidth Test: Shape [{M}, {K}] ({tensor_size_mb:.2f} MB) ---")
        print(f"    num_sms={num_sms},   num_warps_per_sm={num_warps_per_sm},   num_qps={args.num_qps}")

    mori_bandwidths = torch.zeros(WORLD_SIZE, device='cuda', dtype=torch.float32)

    prof = group_profile("mori_shmem_ibgda_bw", args.profile, group=TP_GROUP)
    with prof:
        for src_rank in range(WORLD_SIZE):
            for dst_rank in range(WORLD_SIZE):
                if src_rank == dst_rank:
                    continue

                if RANK == src_rank:
                    bandwidth, latency, _ = test_mori_shmem_bandwidth(
                        src_rank,
                        dst_rank,
                        WORLD_SIZE,
                        local_tensor,
                        symm_tensor,
                        num_qps=args.num_qps,
                        num_sms=num_warps_per_sm,
                        num_warps=4,
                        BLOCK_SIZE=16 * 4096,
                        warmup_iters=args.warmup,
                        test_iters=args.iters,
                        test_ibgda=True,
                    )
                    mori_bandwidths[dst_rank] = bandwidth
                    print(f"[Rank {RANK}] IBGDA PE{src_rank} -> PE{dst_rank}: "
                          f"{bandwidth:.2f} GB/s ({latency:.3f} ms)")

                torch.distributed.barrier(TP_GROUP)

    # Gather all bandwidth results
    all_mori_bandwidths = torch.zeros(WORLD_SIZE, WORLD_SIZE, device='cuda', dtype=torch.float32)
    torch.distributed.all_gather_into_tensor(all_mori_bandwidths, mori_bandwidths.view(1, WORLD_SIZE))
    torch.distributed.barrier(TP_GROUP)

    if RANK == 0:
        print_bw_matrix("MoRI SHMEM IBGDA Bandwidth Matrix", all_mori_bandwidths, WORLD_SIZE)


def parse_args():
    parser = argparse.ArgumentParser(description="MoRI SHMEM P2P Bandwidth Test")
    parser.add_argument("--M", type=int, default=8192, help="Matrix M dimension")
    parser.add_argument("--K", type=int, default=4096, help="Matrix K dimension (fixed to 4096 in size sweep)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations for benchmark")
    parser.add_argument("--iters", type=int, default=10, help="Test iterations for benchmark")
    parser.add_argument("--num-sms", type=int, default=56, help="Number of SMs for kernel launch")
    parser.add_argument("--num-qps", type=int, default=4, help="Number of QPs for MoRI SHMEM")
    parser.add_argument("--size-sweep", action="store_true", help="Run size sweep with a fixed list of M dimensions.")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch Profiler for a fixed size")
    parser.add_argument("--test_ibgda", action="store_true", help="Enable ibgda test")
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


def main():
    args = parse_args()
    if args.test_ibgda:
        os.environ["MORI_DISABLE_P2P"] = "ON"
        os.environ["MORI_NUM_QP_PER_PE"] = str(args.num_qps)
    # Get rank info from environment variables
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed training
    TP_GROUP = initialize_distributed()
    torch.distributed.barrier(TP_GROUP)

    if not args.test_ibgda:
        if RANK == 0:
            print("=" * 80)
            print("MoRI SHMEM P2P Bandwidth Test")
            print("=" * 80)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map[args.dtype]

        if args.profile:
            M, K = args.M, args.K
            run_p2p_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

        elif args.size_sweep:
            size_configs = generate_size_configs(dtype)

            if RANK == 0:
                print(f"\nRunning MoRI SHMEM P2P bandwidth sweep (fixed K=4096, dtype={args.dtype})")

            for i, (M, K, size_bytes) in enumerate(size_configs):
                if RANK == 0:
                    print(f"\n{'='*80}")
                    print(f"Testing config {i+1}/{len(size_configs)}: [{M}, {K}] ({size_bytes/(1024**2):.2f} MB)")
                    print(f"{'='*80}")
                run_p2p_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

        else:
            if RANK == 0:
                print(f"\nRunning MoRI SHMEM P2P bandwidth test with shape [{args.M}, {args.K}], dtype={args.dtype}")
            run_p2p_single_test(args.M, args.K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)
        torch.distributed.barrier(TP_GROUP)
    else:
        if RANK == 0:
            print("=" * 80)
            print("MoRI SHMEM IBGDA Bandwidth Test")
            print("=" * 80)

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map[args.dtype]

        if args.profile:
            M, K = args.M, args.K
            run_ibgda_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

        elif args.size_sweep:
            size_configs = generate_size_configs(dtype)

            if RANK == 0:
                print(f"\nRunning MoRI SHMEM IBGDA bandwidth sweep (fixed K=4096, dtype={args.dtype})")

            for i, (M, K, size_bytes) in enumerate(size_configs):
                if RANK == 0:
                    print(f"\n{'='*80}")
                    print(f"Testing config {i+1}/{len(size_configs)}: [{M}, {K}] ({size_bytes/(1024**2):.2f} MB)")
                    print(f"{'='*80}")
                run_ibgda_single_test(M, K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)

        else:
            if RANK == 0:
                print(f"\nRunning MoRI SHMEM IBGDA bandwidth test with shape [{args.M}, {args.K}], dtype={args.dtype}")
            run_ibgda_single_test(args.M, args.K, dtype, RANK, WORLD_SIZE, TP_GROUP, args)
        torch.distributed.barrier(TP_GROUP)

    # cleanup
    mori_shmem.shmem_finalize()
    torch.distributed.destroy_process_group()

    if RANK == 0:
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)


if __name__ == "__main__":
    main()
