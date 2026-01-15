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
Overlapping AllGather GEMM
==========================

In this tutorial, you will write a simple Allgather GEMM fusion kernel using Triton-distributed.

In doing so, you will learn about:

* Writing a GEMM kernel that consume the results of AllGather.

* Optimizing the internode communication with 2D Allgather.

    # To run this tutorial
    source ./scripts/setenv.sh
    bash ./scripts/launch.sh ./tutorials/07-overlapping-allgather-gemm.py

"""

import os
from typing import Optional

import nvshmem.core
import torch
from triton_dist.utils import cudart

import triton
import triton.language as tl
import triton_dist
import triton_dist.language as dl
from triton_dist.kernels.common_ops import set_signal, wait_eq
from triton_dist.kernels.nvidia.allgather_gemm import create_ag_gemm_context
from triton_dist.language.extra import libshmem_device
from triton_dist.utils import initialize_distributed, nvshmem_barrier_all_on_stream

# %%
# Now, let's write a GEMM kernel to consume the transfered tensors!
# We use tma to optimize the GEMM performance on the SM90 platform


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K = args["M"], args["N"], args["K"]
    ret["name"] = f"{kernel.name} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2.0 * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


@triton.jit(launch_metadata=_matmul_launch_metadata)
def kernel_consumer_gemm_persistent(
        a_ptr,
        b_ptr,
        c_ptr,  #
        M,
        N,
        K,  #
        rank: tl.constexpr,
        num_ranks: tl.constexpr,
        ready_ptr,
        comm_buf_ptr,
        BLOCK_SIZE_M: tl.constexpr,  #
        BLOCK_SIZE_N: tl.constexpr,  #
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        EPILOGUE_SUBTILE: tl.constexpr,  #
        NUM_SMS: tl.constexpr,
        ready_value: tl.constexpr = 1,
        local_world_size: tl.constexpr = 8):  #
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    k_tiles = tl.cdiv(K, BLOCK_SIZE_K)
    num_tiles = num_pid_m * num_pid_n
    node_id = rank // local_world_size
    nnodes = num_ranks // local_world_size

    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )

    tiles_per_SM = num_tiles // NUM_SMS
    if start_pid < num_tiles % NUM_SMS:
        tiles_per_SM += 1

    tile_id = start_pid - NUM_SMS
    ki = -1

    pid_m = 0
    pid_n = 0
    offs_am = 0
    offs_bn = 0

    M_per_rank = M // num_ranks
    pid_ms_per_rank = tl.cdiv(M_per_rank, BLOCK_SIZE_M)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for _ in range(0, k_tiles * tiles_per_SM):
        ki = tl.where(ki == k_tiles - 1, 0, ki + 1)
        if ki == 0:
            tile_id += NUM_SMS
            group_id = tile_id // num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % num_pid_in_group) // group_size_m

            # swizzle m
            if nnodes == 1:
                alpha = 0
                beta = 0
                pid_m = (pid_m + ((((rank ^ alpha) + beta) % num_ranks) *
                                  pid_ms_per_rank)) % num_pid_m
            else:
                m_rank = pid_m // pid_ms_per_rank
                pid_m_intra_rank = pid_m - m_rank * pid_ms_per_rank
                m_node_id = m_rank // local_world_size
                m_local_rank = m_rank % local_world_size
                swizzle_m_node_id = (m_node_id + node_id) % nnodes
                swizzle_m_local_rank = (m_local_rank + rank) % local_world_size
                swizzle_m_rank = swizzle_m_node_id * local_world_size + swizzle_m_local_rank

                pid_m = swizzle_m_rank * pid_ms_per_rank + pid_m_intra_rank

            offs_am = pid_m * BLOCK_SIZE_M
            offs_bn = pid_n * BLOCK_SIZE_N

            rank_beg = offs_am // M_per_rank
            rank_end = (min(offs_am + BLOCK_SIZE_M, M) - 1) // M_per_rank
            # Each tile wait for the corresponding data to ready
            token = dl.wait(ready_ptr + rank_beg,
                            rank_end - rank_beg + 1,
                            "gpu",
                            "acquire",
                            waitValue=ready_value)
            a_desc = dl.consume_token(a_desc, token)

        offs_k = ki * BLOCK_SIZE_K
        # Iteration along k-dimension, and performing multiply.
        a = a_desc.load([offs_am, offs_k])
        b = b_desc.load([offs_bn, offs_k])
        accumulator = tl.dot(a, b.T, accumulator)

        if ki == k_tiles - 1:
            if EPILOGUE_SUBTILE:
                acc = tl.reshape(accumulator,
                                 (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
                acc = tl.permute(acc, (0, 2, 1))
                acc0, acc1 = tl.split(acc)
                c0 = acc0.to(dtype)
                c_desc.store([offs_am, offs_bn], c0)
                c1 = acc1.to(dtype)
                c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
            else:
                c = accumulator.to(dtype)
                c_desc.store([offs_am, offs_bn], c)

            accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                                   dtype=tl.float32)


# %%
# To fully utilize the bandwidth, internode AllGather Kernel is composed of two parts considering the bandwidth gap between intra-node links and inter-node links.
# --------------


def inter_node_allgather(local_tensor: torch.Tensor,
                         ag_buffer: list[torch.Tensor],
                         signal_buffer: list[torch.Tensor], signal_target,
                         rank, local_world_size, world_size,
                         intranode_ag_stream, internode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    M_per_rank, N = local_tensor.shape

    # Each rank sends the local_tensor to ranks of other nodes with the same local_rank
    # Assuming there are 2 nodes, each with 4 workers
    # 0-th local tensor ([0] -> [4]), 4-th local tensor ([4] -> [0])
    # 1-th local tensor ([1] -> [5]), 5-th local tensor ([5] -> [1])
    # 2-th local tensor ([2] -> [6]), 6-th local tensor ([6] -> [2])
    # 3-th local tensor ([3] -> [7]), 7-th local tensor ([7] -> [3])
    with torch.cuda.stream(internode_ag_stream):
        grid = lambda META: (int(n_nodes - 1), )
        nvshmem_device_producer_p2p_put_block_kernel[grid](
            ag_buffer[local_rank],
            signal_buffer[local_rank],
            M_per_rank * N,
            local_tensor.element_size(),
            signal_target,
            rank,
            local_world_size,
            world_size,
            num_warps=32,  # each sm launches 1024 threads
        )

    # Each rank sends the local_tensor and the received internode tensors to intranode ranks.
    # 0-th and 4-th local tensors ([0]->[1,2,3])
    # 1-th and 5-th local tensors ([1]->[0,2,3])
    # 2-th and 6-th local tensors ([2]->[0,1,3])
    # 3-th and 7-th local tensors ([3]->[0,1,2])
    # 0-th and 4-th local tensors ([4]->[5,6,7])
    # 1-th and 5-th local tensors ([5]->[4,6,7])
    # 2-th and 6-th local tensors ([6]->[4,5,7])
    # 3-th and 7-th local tensors ([7]->[4,5,6])
    with torch.cuda.stream(intranode_ag_stream):
        cp_engine_producer_all_gather_put(local_tensor, ag_buffer,
                                          signal_buffer, M_per_rank, N,
                                          signal_target, rank,
                                          local_world_size, world_size,
                                          intranode_ag_stream)

    intranode_ag_stream.wait_stream(internode_ag_stream)


# %%
# Let's declare a function to perform internode communication.


@triton_dist.jit
def nvshmem_device_producer_p2p_put_block_kernel(
    ag_buffer_ptr,  # *Pointer* to allgather output vector. The rank-th index has been loaded with local tensor
    signal_buffer_ptr,  # *Pointer* to signal barrier.
    elem_per_rank,
    size_per_elem,
    signal_target,
    rank,
    local_world_size,
    world_size,
):
    pid = tl.program_id(axis=0)
    num_pid = tl.num_programs(axis=0)

    n_nodes = world_size // local_world_size
    local_rank = rank % local_world_size
    node_rank = rank // local_world_size

    for i in range(pid, n_nodes - 1, num_pid):
        # Each SM is assigned to one peer.
        # Peer id is caculated based on pid and local_rank.
        peer = local_rank + (node_rank + i + 1) % n_nodes * local_world_size
        # We use putmem_signal_block to send data and notify the peer.
        # Since this is the allgather operation, the offsets of both src and dst tensor are both *rank*.
        libshmem_device.putmem_signal_block(
            ag_buffer_ptr + rank * elem_per_rank,
            ag_buffer_ptr + rank * elem_per_rank,
            elem_per_rank * size_per_elem,
            signal_buffer_ptr + rank,
            signal_target,
            libshmem_device.NVSHMEM_SIGNAL_SET,
            peer,
        )


# %%
# Let's also declare a function to perform intranode communication.


def cp_engine_producer_all_gather_put(local_tensor, ag_buffer, signal_buffer,
                                      M_per_rank, N, signal_target, rank,
                                      local_world_size, world_size,
                                      intranode_ag_stream):
    local_rank = rank % local_world_size
    n_nodes = world_size // local_world_size
    node_rank = rank // local_world_size

    for i in range(1, local_world_size):
        segment = rank * M_per_rank * N
        local_dst_rank = (local_rank + local_world_size - i) % local_world_size
        src_ptr = ag_buffer[local_rank].data_ptr(
        ) + segment * local_tensor.element_size()
        dst_ptr = ag_buffer[local_dst_rank].data_ptr(
        ) + segment * local_tensor.element_size()
        # Using copy engine to perform intranode transmission
        # Sending rank-th local tensor to other ranks inside the node.
        (err, ) = cudart.cudaMemcpyAsync(
            dst_ptr,
            src_ptr,
            M_per_rank * N * local_tensor.element_size(),
            cudart.cudaMemcpyKind.cudaMemcpyDefault,
            intranode_ag_stream.cuda_stream,
        )
        # Notify the peer that the transmission is done.
        set_signal(signal_buffer[local_dst_rank][rank].data_ptr(),
                   signal_target, intranode_ag_stream, True)

    for i in range(1, n_nodes):
        recv_rank = local_rank + (node_rank + n_nodes -
                                  i) % n_nodes * local_world_size
        recv_segment = recv_rank * M_per_rank * N
        # Waiting for the internode data ready
        wait_eq(signal_buffer[local_rank][recv_rank].data_ptr(), signal_target,
                intranode_ag_stream, True)
        src_ptr = ag_buffer[local_rank].data_ptr(
        ) + recv_segment * local_tensor.element_size()
        for j in range(1, local_world_size):
            local_dst_rank = (local_rank + local_world_size -
                              j) % local_world_size
            dst_ptr = ag_buffer[local_dst_rank].data_ptr(
            ) + recv_segment * local_tensor.element_size()
            # Sending (local_rank + j*local_world_size) % world_size -th local tensor to other ranks inside the node.
            (err, ) = cudart.cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                M_per_rank * N * local_tensor.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyDefault,
                intranode_ag_stream.cuda_stream,
            )
            # Notify the peer that the transmission is done.
            set_signal(signal_buffer[local_dst_rank][recv_rank].data_ptr(),
                       signal_target, intranode_ag_stream, True)


# %%
# Now we combine all the kernels here.


def ag_gemm_persistent_op(a,
                          b,
                          c,
                          rank,
                          num_ranks,
                          workspace_tensors,
                          barrier_tensors,
                          comm_buf,
                          ag_stream=None,
                          internode_ag_stream=None,
                          BLOCK_M=128,
                          BLOCK_N=256,
                          BLOCK_K=64,
                          stages=3,
                          local_world_size=8,
                          signal_target=1):
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M_per_rank, K = a.shape
    M = M_per_rank * num_ranks
    N_per_rank, K = b.shape

    local_rank = rank % local_world_size
    n_nodes = num_ranks // local_world_size
    num_ag_sms = n_nodes - 1  # only use n_node-1 SMs for internode communication
    num_gemm_sms = torch.cuda.get_device_properties(
        "cuda").multi_processor_count - num_ag_sms

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)

    inter_node_allgather(a, workspace_tensors, barrier_tensors, signal_target,
                         rank, local_world_size, num_ranks, ag_stream,
                         internode_ag_stream)

    compiled = None

    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda META: (min(
        num_gemm_sms,
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(
            N_per_rank, META["BLOCK_SIZE_N"]),
    ), )
    compiled = kernel_consumer_gemm_persistent[grid](
        workspace_tensors[local_rank][:M],
        b,
        c,  #
        M,
        N_per_rank,
        K,  #
        rank,
        num_ranks,
        barrier_tensors[local_rank],
        comm_buf,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        8,
        False,
        NUM_SMS=num_gemm_sms,
        ready_value=signal_target,
        num_stages=stages,
        num_warps=8,
    )

    current_stream.wait_stream(internode_ag_stream)
    current_stream.wait_stream(ag_stream)
    return compiled


# Non-overlap baseline implemented with torch
def torch_ag_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    ag_out: torch.Tensor,
):
    torch.distributed.all_gather_into_tensor(ag_out, local_input, pg)
    ag_gemm_output = torch.matmul(ag_out, local_weight)
    return ag_gemm_output


if __name__ == "__main__":
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "-1"))
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "-1"))

    if WORLD_SIZE == LOCAL_WORLD_SIZE:
        print(
            "Skip the test because this should be performed with 2 nodes or higher"
        )
        import sys
        sys.exit()

    if torch.cuda.get_device_capability()[0] < 9:
        print("Skip the test because the device is not sm90 or higher")
        import sys
        sys.exit()

    TP_GROUP = initialize_distributed()
    rank = TP_GROUP.rank()

    M = 8192
    N = 49152
    K = 12288
    config = {"BM": 128, "BN": 256, "BK": 64, "stage": 3}
    dtype = torch.float16

    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    N_per_rank = N // WORLD_SIZE

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([N_per_rank, K], dtype=dtype, device="cuda").T

    ag_buffer = torch.empty([M, K], dtype=dtype, device="cuda")
    golden = torch_ag_gemm(TP_GROUP, A, B, ag_buffer)

    # We can use a context to wrap all the tensors used at runtime.
    # We rely on NVSHMEM to allocate the symmetric memory for communication
    # In practice, the following parts are encapsulated in ag_gemm_inter_node() of triton_dist.kernels.nvidia.allgather_gemm.py

    C = torch.empty([M, N_per_rank], dtype=dtype, device="cuda")
    ctx = create_ag_gemm_context(A,
                                 B,
                                 rank,
                                 WORLD_SIZE,
                                 max_M=M,
                                 BLOCK_M=config["BM"],
                                 BLOCK_N=config["BN"],
                                 BLOCK_K=config["BK"],
                                 stages=config["stage"])
    ctx.symm_barrier.fill_(0)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    # copy local data to the ctx
    ctx.symm_workspace[rank * M_per_rank:(rank + 1) * M_per_rank, :].copy_(A)
    set_signal(ctx.symm_barrier[rank].data_ptr(), 1,
               torch.cuda.current_stream(), True)

    # launch the ag_gemm kernel
    ag_gemm_persistent_op(A,
                          B,
                          C,
                          ctx.rank,
                          ctx.num_ranks,
                          ctx.symm_workspaces,
                          ctx.symm_barriers,
                          ctx.symm_comm_buf,
                          ag_stream=ctx.ag_intranode_stream,
                          internode_ag_stream=ctx.ag_internode_stream,
                          local_world_size=LOCAL_WORLD_SIZE,
                          signal_target=1)

    assert torch.allclose(golden, C, atol=1e-3, rtol=1e-3)
    print("Pass!")

    ctx.finalize()
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
