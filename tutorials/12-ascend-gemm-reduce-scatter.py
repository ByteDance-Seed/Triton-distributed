# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

import os
import torch
import torch_npu
import shmem as ash
import torch.distributed as dist
import triton
import triton.language as tl
import triton_dist.language as dl
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.ascend.algorithm import (
    dist_swizzle2d_Nz,
    gemm_swizzle2d_Nz,
)
from triton.language.extra.cann.extension import sub_vec_id
import numpy as np

g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8666"

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


@triton.jit
def kernel_gemm_reduce_scatter(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    peer_mem_ptr,
    # Distributed parameters
    rank,
    rank_size,
    buffer_num,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,
    # Meta-parameters
    pvalue: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    COMM_BLOCK_SIZE_M: tl.constexpr,
    COMM_BLOCK_SIZE_N: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    loop_num_per_comm = ncore * pvalue
    m_per_rank = M // rank_size
    num_loops_m = tl.cdiv(m_per_rank, BLOCK_SIZE_M) * rank_size
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_loops = num_loops_m * num_loops_n
    num_loops_comm = tl.cdiv(total_loops, loop_num_per_comm)
    buffer_row_size = BLOCK_SIZE_M * loop_num_per_comm
    for global_id in range(0, num_loops_comm):
        buffer_id = global_id % buffer_num
        actual_loop_num_per_comm = loop_num_per_comm
        output_block_offset_in_rank = global_id * loop_num_per_comm // rank_size
        # process tail block
        if global_id == num_loops_comm - 1:
            actual_loop_num_per_comm = total_loops - global_id * loop_num_per_comm
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        num_blocks_per_rank = actual_loop_num_per_comm // rank_size
        for block_id in range(pid, actual_loop_num_per_comm, ncore):
            block_id_in_rank = (
                output_block_offset_in_rank + block_id % num_blocks_per_rank
            )
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                block_id_in_rank,
                m_per_rank,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            rank_idx = block_id // num_blocks_per_rank
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = (
                m_per_rank * rank_idx
                + block_id_m * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            matmul_msk_am = matmul_offs_am[:, None] < (m_per_rank * (rank_idx + 1))
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(0, num_k_blocks):
                offs_k = tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K
                a_ptrs = a_ptr + (
                    matmul_offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
                )
                b_ptrs = b_ptr + (
                    offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
                )
                a = tl.load(
                    a_ptrs, mask=(offs_k[None, :] < K) & matmul_msk_am, other=0.0
                )
                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & msk_n, other=0.0)
                # We accumulate along the K dimension.
                accumulator += tl.dot(a, b)
            c = accumulator
            # -----------------------------------------------------------
            # Write the block of the output matrix C with masks.
            offs_peer_mem_m = (
                buffer_id * buffer_row_size
                + block_id * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            offs_peer_mem_n = tl.arange(0, BLOCK_SIZE_N)
            peer_mem_ptrs = (
                peer_mem_ptr
                + BLOCK_SIZE_N * offs_peer_mem_m[:, None]
                + offs_peer_mem_n[None, :]
            )
            tl.store(peer_mem_ptrs, c)
        libshmem_device.barrier_all()
        comm_problem_size_m_per_rank = tl.cdiv(
            actual_loop_num_per_comm * BLOCK_SIZE_M, rank_size
        )
        comm_block_num_per_rank = tl.cdiv(
            comm_problem_size_m_per_rank, COMM_BLOCK_SIZE_M
        ) * tl.cdiv(BLOCK_SIZE_N, COMM_BLOCK_SIZE_N)
        if subblock_idx == 0:
            for idx in range(pid, comm_block_num_per_rank * rank_size, ncore):
                block_id_m, block_id_n, target_rank, comm_row_shape, comm_col_shape = (
                    dist_swizzle2d_Nz(
                        idx,
                        rank_size,
                        comm_problem_size_m_per_rank,
                        BLOCK_SIZE_N,
                        COMM_BLOCK_SIZE_M,
                        COMM_BLOCK_SIZE_N,
                    )
                )
                remote_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                comm_offs_m = (
                    tl.arange(0, COMM_BLOCK_SIZE_M)
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + rank * comm_problem_size_m_per_rank
                    + buffer_id * loop_num_per_comm * BLOCK_SIZE_M
                )
                comm_offs_n = (
                    tl.arange(0, COMM_BLOCK_SIZE_N) + block_id_n * COMM_BLOCK_SIZE_N
                )
                remote_ptrs = remote_ptr + (
                    comm_offs_m[:, None] * BLOCK_SIZE_N + comm_offs_n[None, :]
                )
                block_id = block_id_m * COMM_BLOCK_SIZE_M // BLOCK_SIZE_M
                block_id_in_rank = output_block_offset_in_rank + block_id
                block_id_gemm_m, block_id_gemm_n = gemm_swizzle2d_Nz(
                    block_id_in_rank, m_per_rank, N, BLOCK_SIZE_M, BLOCK_SIZE_N
                )
                m_offset_in_block = (block_id_m * COMM_BLOCK_SIZE_M) % BLOCK_SIZE_M
                n_offset_in_block = (block_id_n * COMM_BLOCK_SIZE_N) % BLOCK_SIZE_N
                offs_cm = (
                    block_id_gemm_m * BLOCK_SIZE_M
                    + m_offset_in_block
                    + tl.arange(0, COMM_BLOCK_SIZE_M)
                )
                offs_cn = (
                    block_id_gemm_n * BLOCK_SIZE_N
                    + n_offset_in_block
                    + tl.arange(0, COMM_BLOCK_SIZE_N)
                )
                c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
                c_mask = (offs_cm[:, None] < m_per_rank) & (offs_cn[None, :] < N)
                c_temp = tl.load(remote_ptrs)
                tl.atomic_add(c_ptr + c_offs, c_temp, mask=c_mask)


def gemm_reduce_scatter(
    A,
    B,
    C,
    peer_mem,
    rank,
    rank_size,
    buffer_num,
    pvalue,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    COMM_BLOCK_SIZE_M,
    COMM_BLOCK_SIZE_N,
):
    """consume gemm"""
    M, K = A.shape
    _, N = B.shape
    ncore = 20
    assert ncore <= 20, "block num should less than or equal physical aicore num"
    kernel_gemm_reduce_scatter[ncore, 1, 1](
        A,
        B,
        C,
        peer_mem,
        rank,
        rank_size,
        buffer_num,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        pvalue,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        COMM_BLOCK_SIZE_M,
        COMM_BLOCK_SIZE_N,
    )


def torch_gemm_reduce_scatter(A_list, B_list, rank, world_size):
    """
    torch reduce-scatter gemm using torch.distributed.all_gather
    - A_list: list of A tensors from all ranks (gathered)
    - B_list: list of B tensors from all ranks (gathered)
    - rank: current rank
    - world_size: number of ranks
    Returns: C_golden for this rank (1/world_size of total)
    """
    # Compute local C_i = A_i @ B_i for each rank
    C_list = [
        torch.matmul(A_list[i].to(torch.float32), B_list[i].to(torch.float32))
        for i in range(world_size)
    ]

    # Sum all C tensors across all ranks
    C_sum = sum(C_list)

    # Each rank gets 1/world_size of the result (M // world_size rows)
    m_per_rank = C_sum.shape[0] // world_size
    start_idx = rank * m_per_rank
    end_idx = start_idx + m_per_rank

    return C_sum[start_idx:end_idx].to(torch.float16)


def run_test_distributed():
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256
    COMM_BLOCK_SIZE_M = 8
    COMM_BLOCK_SIZE_N = 256
    buffer_num = 2
    pvalue = 4
    ncore = 20
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = pe
    attributes.n_ranks = world_size
    attributes.local_mem_size = g_ash_size
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")

    # Calculate peer_mem size based on world_size
    peer_mem_size = BLOCK_SIZE_M * pvalue * ncore * buffer_num * BLOCK_SIZE_N
    peer_mem = ash.aclshmem_create_tensor(
        [peer_mem_size],
        dtype=torch.float32,
        device_id=pe,
    )

    # Each rank creates its own local A and B tensors
    A_local = torch.randn([M, K], dtype=dtype).npu()
    B_local = torch.randn([K, N], dtype=dtype).npu()
    C = torch.zeros([M // world_size, N], dtype=torch.float32).npu()

    # Gather all A and B tensors from all ranks
    A_list = [torch.empty_like(A_local) for _ in range(world_size)]
    B_list = [torch.empty_like(B_local) for _ in range(world_size)]
    dist.all_gather(A_list, A_local)
    dist.all_gather(B_list, B_local)

    # Compute golden reference using reduce-scatter
    C_golden = torch_gemm_reduce_scatter(A_list, B_list, pe, world_size)

    # Run the distributed kernel with local A and B
    gemm_reduce_scatter(
        A_local,
        B_local,
        C,
        peer_mem,
        pe,
        world_size,
        buffer_num,
        pvalue,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        COMM_BLOCK_SIZE_M,
        COMM_BLOCK_SIZE_N,
    )

    passed = torch.tensor([1], dtype=torch.int32).npu()
    error_msg = ""
    try:
        torch.testing.assert_close(C_golden, C.to(torch.float16), rtol=1e-3, atol=1e-3)
    except AssertionError as e:
        passed[0] = 0
        error_msg = str(e)
        raise

    # Gather all ranks' pass/fail status
    all_passed = [torch.zeros(1, dtype=torch.int32).npu() for _ in range(world_size)]
    dist.all_gather(all_passed, passed)

    # Print sequentially, one rank at a time
    dist.barrier()
    for rank_id in range(world_size):
        if pe == rank_id:
            if all_passed[rank_id].item() == 1:
                print(
                    f"{GREEN}[PASS]{RESET} Rank {pe}: C_golden and C match within tolerances (rtol=1e-3, atol=1e-3).",
                    flush=True,
                )
            else:
                print(
                    f"{RED}[FAIL]{RESET} Rank {pe}: C_golden and C do NOT match. Details:\n{error_msg}",
                    flush=True,
                )
        dist.barrier()
    # Raise if any rank failed
    if any(r.item() == 0 for r in all_passed):
        if passed.item() == 0:
            raise AssertionError(error_msg)

    ash.aclshmem_free_tensor(peer_mem)
    _ = ash.aclshmem_finialize()


if __name__ == "__main__":
    local_pe = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_pe)
    dist.init_process_group(backend="hccl", rank=local_pe)
    dtype, M, N, K = [torch.float16, 4096, 4096, 4096]
    world_size = dist.get_world_size()

    print(f"[INFO] Rank {local_pe} of {world_size} initialized")
    dist.barrier()
    run_test_distributed()
    if local_pe == 0:
        print(f"[INFO] Test passed successfully for {world_size} ranks!")
