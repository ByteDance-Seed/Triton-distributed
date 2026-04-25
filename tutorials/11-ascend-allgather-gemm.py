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
def kernel_allgather_gemm(
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
    COMM_BLOCK_SIZE_K: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    dtype = tl.float16
    subblock_idx = sub_vec_id()
    ncore = tl.num_programs(axis=0)
    pid = tl.program_id(axis=0)
    num_loops_m = tl.cdiv(M, BLOCK_SIZE_M * pvalue)
    num_loops_n = tl.cdiv(N, BLOCK_SIZE_N)
    buffer_row_size = BLOCK_SIZE_M * pvalue * rank_size
    for global_id_m in range(0, num_loops_m):
        buffer_id = global_id_m % buffer_num
        actual_block_size_m = BLOCK_SIZE_M * pvalue
        # process tail block
        if global_id_m == num_loops_m - 1:
            actual_block_size_m = M - global_id_m * BLOCK_SIZE_M * pvalue
        num_k_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        comm_num_m_blocks = tl.cdiv(actual_block_size_m, COMM_BLOCK_SIZE_M)
        comm_num_k_blocks = tl.cdiv(K, COMM_BLOCK_SIZE_K)
        if subblock_idx == 0:
            for k in range(
                pid, comm_num_m_blocks * comm_num_k_blocks * rank_size, ncore
            ):
                block_id_m, block_id_k, target_rank, comm_row_shape, comm_col_shape = (
                    dist_swizzle2d_Nz(
                        k,
                        rank_size,
                        actual_block_size_m,
                        K,
                        COMM_BLOCK_SIZE_M,
                        COMM_BLOCK_SIZE_K,
                    )
                )
                remote_ptr = dl.symm_at(peer_mem_ptr, target_rank)
                comm_offs_m = (
                    tl.arange(0, COMM_BLOCK_SIZE_M)
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + global_id_m * BLOCK_SIZE_M * pvalue
                )
                comm_offs_k = (
                    tl.arange(0, COMM_BLOCK_SIZE_K) + block_id_k * COMM_BLOCK_SIZE_K
                )
                a_ptrs = a_ptr + (
                    comm_offs_m[:, None] * stride_am + comm_offs_k[None, :] * stride_ak
                )
                peermem_comm_offs_m = (
                    buffer_id * buffer_row_size
                    + rank * BLOCK_SIZE_M * pvalue
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + tl.arange(0, COMM_BLOCK_SIZE_M)
                )
                remote_ptrs = remote_ptr + (
                    peermem_comm_offs_m[:, None] * stride_am
                    + comm_offs_k[None, :] * stride_ak
                )
                comm_msk_m = comm_offs_m[:, None] < M
                peermem_comm_msk_m = (
                    peermem_comm_offs_m[:, None]
                    < buffer_id * buffer_row_size
                    + BLOCK_SIZE_M * rank * pvalue
                    + block_id_m * COMM_BLOCK_SIZE_M
                    + comm_row_shape
                )
                a = tl.load(
                    a_ptrs, mask=(comm_offs_k[None, :] < K) & comm_msk_m, other=0.0
                )
                tl.store(
                    remote_ptrs, a, mask=(comm_offs_k[None, :] < K) & peermem_comm_msk_m
                )
        libshmem_device.barrier_all()
        num_tiles_m = tl.cdiv(actual_block_size_m, BLOCK_SIZE_M)
        for block_id in range(pid, num_tiles_m * num_loops_n * rank_size, ncore):
            block_id_m, block_id_n = gemm_swizzle2d_Nz(
                block_id,
                rank_size * BLOCK_SIZE_M * num_tiles_m,
                N,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
            )
            rank_idx = block_id_m // num_tiles_m
            block_id_m = rank_idx * pvalue + (block_id_m % num_tiles_m)
            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            matmul_offs_am = (
                buffer_id * buffer_row_size
                + block_id_m * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            matmul_msk_am = matmul_offs_am[:, None] < (
                buffer_id * buffer_row_size
                + BLOCK_SIZE_M * rank_idx * pvalue
                + actual_block_size_m
            )
            offs_bn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            msk_n = offs_bn[None, :] < N
            for block_id_k in range(0, num_k_blocks):
                offs_k = tl.arange(0, BLOCK_SIZE_K) + block_id_k * BLOCK_SIZE_K
                a_ptrs = peer_mem_ptr + (
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
            c = accumulator.to(dtype)
            # -----------------------------------------------------------
            # Write back the block of the output matrix C with masks.
            offs_cm = (
                block_id_m // pvalue * M
                + global_id_m * BLOCK_SIZE_M * pvalue
                + (block_id_m % pvalue) * BLOCK_SIZE_M
                + tl.arange(0, BLOCK_SIZE_M)
            )
            offs_cn = block_id_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
            c_mask = (offs_cm[:, None] < M * (block_id_m // pvalue + 1)) & (
                offs_cn[None, :] < N
            )
            tl.store(c_ptrs, c, mask=c_mask)


def allgather_gemm(
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
    COMM_BLOCK_SIZE_K,
):
    """consume gemm"""
    M, K = A.shape
    _, N = B.shape
    ncore = 20
    assert ncore <= 20, "block num should less than or equal physical aicore num"
    kernel_allgather_gemm[ncore, 1, 1](
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
        COMM_BLOCK_SIZE_K,
    )


def torch_allgather_gemm(A_local, B, world_size):
    """
    torch allgather gemm using torch.distributed.all_gather
    - A_local: local A tensor on this rank
    - B: B tensor (same across all ranks)
    - world_size: number of ranks
    Returns: C_golden computed from all-gathered A tensors
    """
    # Create a list to hold gathered A tensors from all ranks
    A_list = [torch.empty_like(A_local) for _ in range(world_size)]

    # Gather all A tensors from all ranks
    dist.all_gather(A_list, A_local)

    # Concatenate all A tensors along the first dimension
    A_golden = torch.cat(A_list, dim=0)

    # Compute golden reference: C = A_golden @ B

    C_golden = torch.matmul(A_golden, B)
    return C_golden


def run_test_distributed():
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256
    COMM_BLOCK_SIZE_M = 20
    COMM_BLOCK_SIZE_K = 256
    buffer_num = 2
    pvalue = 4
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
    peer_mem_size = (
        BLOCK_SIZE_M * pvalue * world_size * buffer_num * max(K, BLOCK_SIZE_K)
    )
    peer_mem = ash.aclshmem_create_tensor(
        [peer_mem_size],
        dtype=dtype,
        device_id=pe,
    )

    # Each rank creates its own local A and B tensors
    A_local = torch.randn([M, K], dtype=dtype).npu()
    B = torch.randn([K, N], dtype=dtype).npu()
    C = torch.zeros([M * world_size, N], dtype=dtype).npu()

    # Compute golden reference using torch.distributed.all_gather
    C_golden = torch_allgather_gemm(A_local, B, world_size)

    # Run the distributed kernel with local A
    allgather_gemm(
        A_local,
        B,
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
        COMM_BLOCK_SIZE_K,
    )

    passed = torch.tensor([1], dtype=torch.int32).npu()
    error_msg = ""
    try:
        torch.testing.assert_close(C_golden, C, rtol=1e-3, atol=1e-3)
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
