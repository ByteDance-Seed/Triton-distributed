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
import triton_dist.language as dl
from typing import Optional, List
from dataclasses import dataclass
from triton.language.extra.cuda.language_extra import tid, st

from triton_dist.utils import (nvshmem_create_tensors, nvshmem_free_tensor_sync, NVSHMEM_SIGNAL_DTYPE)
from triton_dist.kernels.nvidia.common_ops import _set_signal_cuda, barrier_all_intra_node_non_atomic, nvshmem_barrier_all_on_stream


def all_to_all_copy_engine(input: torch.Tensor, input_scale: Optional[torch.Tensor],
                           comm_data_buffers: List[torch.Tensor], comm_scale_buffers: Optional[List[torch.Tensor]],
                           barrier_buffers: List[torch.Tensor], rank: int, world_size: int, stream: torch.cuda.Stream):
    """
    Args:
        input: Input tensor of shape (M, K)
        input_scale: Optional input scales (M,) 
        comm_data_buffers: List of communication buffers
        comm_scale_buffers: Optional list of scale buffers
        barrier_buffers: List of barrier buffers (one per rank)
        rank: Current rank
        world_size: Total number of ranks
        stream: CUDA stream to use
    """
    M, K = input.shape

    if M % world_size != 0:
        raise ValueError(
            f"The first dimension of the input tensor ({M}) must be divisible by world_size ({world_size}).")
    m_per_rank = M // world_size

    for i in range(world_size):
        target_rank = (rank + i) % world_size

        comm_data_buffers[target_rank][rank * m_per_rank:(rank + 1) * m_per_rank, :].copy_(
            input[target_rank * m_per_rank:(target_rank + 1) * m_per_rank, :])

        if input_scale is not None and comm_scale_buffers is not None:
            comm_scale_buffers[target_rank][rank * m_per_rank:(rank + 1) * m_per_rank].copy_(
                input_scale[target_rank * m_per_rank:(target_rank + 1) * m_per_rank])

        _set_signal_cuda(barrier_buffers[target_rank][rank], 1, stream)


@triton.jit
def kernel_all_to_all_gemm_consumer(
    comm_data_ptr,
    weight_ptr,
    output_ptr,
    comm_scale_ptr,
    weight_scale_ptr,
    barrier_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # Data type flags
    USE_INT8: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Group size for swizzling
    GROUP_SIZE_M: tl.constexpr = 8,
    INPUT_IS_E5M2: tl.constexpr = True,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    m_per_rank = M // world_size
    original_pid_m = first_pid_m + (pid % group_size_m)

    m_blocks_per_rank = m_per_rank // BLOCK_M
    data_rank = original_pid_m // m_blocks_per_rank
    m_block_in_chunk = original_pid_m % m_blocks_per_rank
    target_rank = (rank - data_rank + world_size) % world_size
    pid_m = target_rank * m_blocks_per_rank + m_block_in_chunk
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m >= num_pid_m or pid_n >= num_pid_n:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    rank_beg = (pid_m * BLOCK_M) // m_per_rank
    rank_end = min((pid_m + 1) * BLOCK_M - 1, M - 1) // m_per_rank

    rank_beg = tl.minimum(rank_beg, world_size - 1)
    rank_end = tl.minimum(rank_end, world_size - 1)

    num_ranks_to_wait = rank_end - rank_beg + 1
    token = dl.wait(barrier_ptr + rank_beg, num_ranks_to_wait, "gpu", "acquire", waitValue=1)

    if INPUT_IS_E5M2:
        FP8_TYPE = tl.float8e5
    else:
        FP8_TYPE = tl.float8e4nv

    comm_data_ptr = dl.consume_token(comm_data_ptr, token)
    if USE_INT8:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if USE_INT8 or USE_FP8:
        a_scale = tl.load(comm_scale_ptr + rm, mask=rm < M, other=1.0).to(tl.float32)
        b_scale = tl.load(weight_scale_ptr + rn, mask=rn < N, other=1.0).to(tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        a_ptrs = comm_data_ptr + rm[:, None] * K + rk[None, :]
        a_mask = (rm[:, None] < M) & (rk[None, :] < K)

        if USE_INT8:
            a = tl.load(a_ptrs, mask=a_mask, other=0).to(tl.int8)
        elif USE_FP8:
            a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(FP8_TYPE)
        else:
            a = tl.load(a_ptrs, mask=a_mask, other=0)

        b_ptrs = weight_ptr + rn[:, None] * K + rk[None, :]
        b_mask = (rn[:, None] < N) & (rk[None, :] < K)

        if USE_INT8:
            b = tl.load(b_ptrs, mask=b_mask, other=0).to(tl.int8)
        elif USE_FP8:
            b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(FP8_TYPE)
        else:
            b = tl.load(b_ptrs, mask=b_mask, other=0)

        acc += tl.dot(a, tl.trans(b))

    if USE_INT8:
        acc_fp = acc.to(tl.float32)
        c = (acc_fp * a_scale[:, None] * b_scale[None, :]).to(tl.bfloat16)
    elif USE_FP8:
        acc = acc * a_scale[:, None] * b_scale[None, :]
        c = acc.to(tl.bfloat16)
    else:
        c = acc.to(tl.bfloat16)

    c_ptrs = output_ptr + rm[:, None] * N + rn[None, :]
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@dataclass
class AllToAllSingleGemmContext:
    """Context for All-to-All Single GEMM operation with multi-stream support"""

    # Communication buffers
    comm_data_buffers: List[torch.Tensor]
    comm_input_scale_buffers: Optional[List[torch.Tensor]] = None
    barrier_buffers: Optional[List[torch.Tensor]] = None
    symm_comm_bufs: Optional[List[torch.Tensor]] = None
    comm_stream: Optional[torch.cuda.Stream] = None

    # Configuration
    max_m: int = 0
    n: int = 0
    k: int = 0
    rank: int = 0
    local_world_size: int = 0
    dtype: torch.dtype = torch.bfloat16
    scale_dtype: torch.dtype = torch.float32
    has_scales: bool = False

    # phase
    phase: int = 1

    def __post_init__(self):
        self.has_scales = self.dtype in [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2]

        if self.comm_stream is None:
            self.comm_stream = torch.cuda.Stream()

    def finalize(self):
        for buf in self.comm_data_buffers:
            if buf is not None:
                nvshmem_free_tensor_sync(buf)

        if self.comm_input_scale_buffers:
            for buf in self.comm_input_scale_buffers:
                if buf is not None:
                    nvshmem_free_tensor_sync(buf)

        if self.barrier_buffers is not None:
            for buf in self.barrier_buffers:
                if buf is not None:
                    nvshmem_free_tensor_sync(buf)


def create_all_to_all_single_gemm_context(max_m: int, n: int, k: int, rank: int, local_world_size: int,
                                          dtype: torch.dtype = torch.bfloat16,
                                          scale_dtype: torch.dtype = torch.float32) -> AllToAllSingleGemmContext:
    """
    Create context for All-to-All Single GEMM operation
    
    Args:
        max_m: Maximum number of rows
        n: Number of columns for output
        k: Number of columns for input
        rank: Current rank
        local_world_size: Number of ranks in local node
        dtype: Data type for computation (int8, float8_e4m3fn, float8_e5m2, bfloat16)
        scale_dtype: Data type for scales
    """
    comm_data_buffers = nvshmem_create_tensors((max_m, k), dtype, rank, local_world_size)

    comm_input_scale_buffers = None
    has_scales = dtype in [torch.int8, torch.float8_e4m3fn, torch.float8_e5m2]

    if has_scales:
        comm_input_scale_buffers = nvshmem_create_tensors((max_m, ), scale_dtype, rank, local_world_size)

    barrier_buffers = nvshmem_create_tensors((local_world_size, ), NVSHMEM_SIGNAL_DTYPE, rank, local_world_size)
    symm_comm_bufs = nvshmem_create_tensors((3 * local_world_size, ), torch.int32, rank, local_world_size)
    symm_comm_bufs[rank].fill_(0)

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()

    return AllToAllSingleGemmContext(
        comm_data_buffers=comm_data_buffers,
        comm_input_scale_buffers=comm_input_scale_buffers,
        barrier_buffers=barrier_buffers,
        symm_comm_bufs=symm_comm_bufs,
        max_m=max_m,
        n=n,
        k=k,
        rank=rank,
        local_world_size=local_world_size,
        dtype=dtype,
        scale_dtype=scale_dtype,
    )


@triton.jit(do_not_specialize=["local_rank", "rank", "num_ranks", "flag_value"])
def barrier_all_intra_node_kernel(
    local_rank,
    rank,
    num_ranks,
    symm_barrier_ptr,
    symm_sync_ptr,
    flag_value,
    use_cooperative: tl.constexpr,
):
    barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_sync_ptr, flag_value, use_cooperative)
    thread_idx = tid(0)
    if thread_idx < num_ranks:
        st(symm_barrier_ptr + thread_idx, 0)
    barrier_all_intra_node_non_atomic(local_rank, rank, num_ranks, symm_sync_ptr, flag_value + 1, use_cooperative)


def all_to_all_single_gemm(
    input: torch.Tensor,
    weight: torch.Tensor,
    context: AllToAllSingleGemmContext,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    num_comm_sms: int = -1,
    sm_margin: int = 0,
) -> torch.Tensor:
    """
    All-to-All Single GEMM operation with separated communication and computation
    
    Uses copy engine (cudaMemcpyAsync) for communication and separate kernels 
    on different streams to achieve true overlap:
    - Communication runs on comm_stream using copy engine
    - GEMM kernel runs on current_stream
    
    Args:
        input: Input tensor of shape (M, K)
        weight: Weight tensor of shape (K, N)
        context: AllToAllSingleGemmContext instance
        input_scale: Optional input scales for quantization (M,)
        weight_scale: Optional weight scales for quantization (N,)
        output: Optional output tensor of shape (M, N)
        num_comm_sms: Number of SMs for communication (not used with copy engine)
        sm_margin: SM margin for computation
    
    Returns:
        Output tensor of shape (M, N)
    """
    M, K = input.shape
    N, K_w = weight.shape  # weight is (N, K) now
    assert K == K_w, f"Dimension mismatch: input K={K}, weight K={K_w}"

    # Validate quantization parameters
    dtype = input.dtype
    use_int8 = dtype == torch.int8
    use_fp8 = dtype in [torch.float8_e4m3fn, torch.float8_e5m2]
    has_scale = use_int8 or use_fp8

    if has_scale:
        assert input_scale is not None, "Input scale required for quantized types"
        assert weight_scale is not None, "Weight scale required for quantized types"
        if input_scale.dim() == 2:
            input_scale = input_scale.squeeze(1)
        if weight_scale.dim() == 2:
            weight_scale = weight_scale.squeeze(0)

    if output is None:
        output_dtype = torch.bfloat16 if (use_int8 or use_fp8) else input.dtype
        output = torch.empty((M, N), dtype=output_dtype, device=input.device)

    current_stream = torch.cuda.current_stream()
    barrier_all_intra_node_kernel[(1, )](context.rank, context.rank, context.local_world_size,
                                         context.barrier_buffers[context.rank],
                                         symm_sync_ptr=context.symm_comm_bufs[context.rank], flag_value=context.phase,
                                         use_cooperative=False)
    context.phase += 2
    context.comm_stream.wait_stream(current_stream)

    with torch.cuda.stream(context.comm_stream):
        all_to_all_copy_engine(input, input_scale if has_scale else None, context.comm_data_buffers,
                               context.comm_input_scale_buffers if has_scale else None, context.barrier_buffers,
                               context.rank, context.local_world_size, context.comm_stream)
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    gemm_grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    local_comm_data = context.comm_data_buffers[context.rank]
    local_comm_scale = None
    if context.comm_input_scale_buffers:
        local_comm_scale = context.comm_input_scale_buffers[context.rank]

    if weight.dtype == torch.float8_e5m2:
        input_is_e5m2_flag = True
    else:
        input_is_e5m2_flag = False

    kernel_all_to_all_gemm_consumer[gemm_grid](
        local_comm_data,
        weight,
        output,
        local_comm_scale if has_scale else local_comm_data,
        weight_scale if has_scale else weight,
        context.barrier_buffers[context.rank],
        M,
        N,
        K,
        context.rank,
        context.local_world_size,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        use_int8,
        use_fp8,
        GROUP_SIZE_M,
        input_is_e5m2_flag,
    )
    current_stream.wait_stream(context.comm_stream)
    return output


def gemm_only(
    input: torch.Tensor,
    weight: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GEMM-only operation without All-to-All communication
    Used for benchmarking GEMM performance separately
    """
    M, K = input.shape
    N, K_w = weight.shape
    assert K == K_w, f"Dimension mismatch: input K={K}, weight K={K_w}"

    dtype = input.dtype
    use_int8 = dtype == torch.int8
    use_fp8 = dtype in [torch.float8_e4m3fn, torch.float8_e5m2]

    if output is None:
        output_dtype = torch.bfloat16 if (use_int8 or use_fp8) else input.dtype
        output = torch.empty((M, N), dtype=output_dtype, device=input.device)

    if use_int8 or use_fp8:
        assert input_scale is not None, "Input scale required for quantized types"
        assert weight_scale is not None, "Weight scale required for quantized types"
        if input_scale.dim() == 2:
            input_scale = input_scale.squeeze(1)
        if weight_scale.dim() == 2:
            weight_scale = weight_scale.squeeze(0)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64
    GROUP_SIZE_M = 8

    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )

    dummy_barrier = torch.ones(1, dtype=NVSHMEM_SIGNAL_DTYPE, device=input.device)

    if weight.dtype == torch.float8_e5m2:
        input_is_e5m2_flag = True
    else:
        input_is_e5m2_flag = False

    kernel_all_to_all_gemm_consumer[grid](
        input,
        weight,
        output,
        input_scale if input_scale is not None else input,
        weight_scale if weight_scale is not None else weight,
        dummy_barrier,
        M,
        N,
        K,
        0,  # rank
        1,  # world_size
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        use_int8,
        use_fp8,
        GROUP_SIZE_M,
        input_is_e5m2_flag,
    )

    return output
