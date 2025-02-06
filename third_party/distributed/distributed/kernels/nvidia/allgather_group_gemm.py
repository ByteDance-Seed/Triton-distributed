from dataclasses import dataclass
import torch
import torch.distributed
import triton
import triton.language as tl
import triton.distributed.language as dl
from triton.distributed.utils import CUDA_CHECK, TP_GROUP
from typing import Optional, List
from cuda import cuda
from triton._C.libtriton_distributed.distributed import moe_ag_scatter_align_block_size

import pynvshmem


def torch_dtype_to_triton_dtype(dtype):
    if dtype == torch.float32:
        return tl.float32
    elif dtype == torch.float16:
        return tl.float16
    elif dtype == torch.int32:
        return tl.int32
    elif dtype == torch.int8:
        return tl.int8
    else:
        raise RuntimeError(f"unsupported dtype: {dtype}")


def sort_topk_ids_align_block_size(
    topk_ids: torch.Tensor,
    num_experts: int,
    num_ranks: int,
    num_tokens_per_rank: int,
    block_size: int,
    moe_stream: torch.cuda.Stream = None,
):
    """
    Sort and align the token distribution across experts to be compatible with block size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the top-k expert indices for each token.
    - num_experts: The total number of experts.
    - num_ranks: The total number of ranks.
    - num_tokens_per_rank: The total number of tokens (not topked) per rank.
    - block_size: The block size used in block matrix multiplication.
    -  moe_stream (torch.cuda.streams.Stream, optional): The stream used for cuda kernel, if not provided, use current stream. Defaults to None.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - block_barrier_ids: A tensor indicating the assigned barrier index for each block.
    - rank_block_num: A tensor indicating the number of blocks for each rank of tokens.
    - num_tokens_post_padded: The total number of tokens after padding, ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]], block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts, where each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible by block_size for proper block matrix operations.
    """
    num_topk = topk_ids.shape[1]
    sorted_ids = torch.empty(
        ((num_tokens_per_rank * num_topk + num_experts * (block_size - 1)) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        ((num_tokens_per_rank * num_topk + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    block_barrier_ids = torch.empty(
        ((num_tokens_per_rank * num_topk + num_experts) * num_ranks, ),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    rank_block_num = torch.empty(
        num_ranks,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    moe_stream = torch.cuda.current_stream() if moe_stream is None else moe_stream
    moe_ag_scatter_align_block_size(
        topk_ids,
        num_experts,
        num_ranks,
        num_tokens_per_rank * topk_ids.shape[1],
        block_size,
        sorted_ids,
        expert_ids,
        block_barrier_ids,
        rank_block_num,
        num_tokens_post_pad,
        moe_stream.cuda_stream,
    )
    torch.cuda.current_stream().wait_stream(moe_stream)
    return (
        sorted_ids,
        expert_ids,
        block_barrier_ids,
        rank_block_num,
        num_tokens_post_pad,
    )


def cp_engine_producer_all_gather_full_mesh_push(
    rank,
    num_ranks,
    local_tensor: torch.Tensor,
    remote_tensor_buffers: List[torch.Tensor],
    barrier_buffers: List[torch.Tensor],
    exec_stream: torch.cuda.Stream,
):
    M_per_rank, N_per_rank = local_tensor.shape
    push_order = [(rank + i) % num_ranks for i in range(num_ranks)]
    src = local_tensor
    with torch.cuda.stream(exec_stream):
        for dst_rank in push_order:
            dst = remote_tensor_buffers[dst_rank][rank * M_per_rank:(rank + 1) * M_per_rank, :]
            dst.copy_(src)

            (err, ) = cuda.cuStreamWriteValue32(
                exec_stream.cuda_stream,
                barrier_buffers[dst_rank][rank].data_ptr(),
                1,
                cuda.CUstreamWriteValue_flags.CU_STREAM_WRITE_VALUE_DEFAULT,
            )
            CUDA_CHECK(err)


@triton.jit
def kernel_consumer_m_parallel_scatter_group_gemm(
    in_features_ptr,
    expert_weights_ptr,
    out_features_ptr,
    block_barrier_ptr,
    sorted_token_ids_ptr,
    token_expert_ids_ptr,
    num_tokens_post_padded,
    block_barrier_id_ptr,
    num_valid_tokens,
    M,
    N,
    K,
    stride_in_m,
    stride_in_k,
    stride_weight_e,
    stride_weight_k,
    stride_weight_n,
    stride_out_m,
    stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    TOP_K: tl.constexpr,
    compute_dtype: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    swizzle_offset: tl.constexpr = 3,
):
    pid = tl.program_id(axis=0)
    num_block_m = tl.cdiv(M, BLOCK_M)
    num_block_n = tl.cdiv(N, BLOCK_N)

    num_blocks_per_group = GROUP_M * num_block_n
    group_id = pid // num_blocks_per_group
    group_size = min(num_block_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + pid % group_size
    pid_n = pid % num_blocks_per_group // group_size

    # swizzle along m-dimension
    num_rank_m = world_size
    num_rank_n = 1
    m_per_rank = num_block_m // num_rank_m
    rank_m = rank // num_rank_n
    m_offset = m_per_rank * ((rank_m + swizzle_offset) % num_rank_m)
    pid_m = (pid_m + m_offset) % num_block_m

    num_tokens_post_padded_value = tl.load(num_tokens_post_padded)

    if pid_m * BLOCK_M >= num_tokens_post_padded_value:
        return

    offs_token_id = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = (in_features_ptr + offs_token[:, None] // TOP_K * stride_in_m + offs_k[None, :] * stride_in_k)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_be = tl.load(token_expert_ids_ptr + pid_m)

    b_ptrs = (expert_weights_ptr + offs_be * stride_weight_e + offs_k[:, None] * stride_weight_k +
              offs_bn[None, :] * stride_weight_n)

    offs_barrier = tl.load(block_barrier_id_ptr + pid_m)
    token = dl.wait(block_barrier_ptr + offs_barrier, 1, "gpu", "acquire")
    a_ptrs = dl.consume_token(a_ptrs, token)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_K))
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k * BLOCK_K))

        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_in_k
        b_ptrs += BLOCK_K * stride_weight_k

    accumulator = accumulator.to(compute_dtype)

    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (out_features_ptr + offs_token[:, None] * stride_out_m + offs_cn[None, :] * stride_out_n)
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def rowise_intranode_ag_scatter_group_gemm(
    a,  # local tensor
    a_buffer,
    a_tensors: List[torch.Tensor],
    b,  # local weight
    c,  # output
    block_barriers,
    topk,
    full_sorted_token_ids,
    full_token_expert_ids,
    block_barrier_ids,
    full_num_tokens_post_padded_list,
    full_numel,
    rank,
    world_size,
    ag_stream=None,
    group_gemm_stream=None,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    warps=4,
):
    M_per_rank, K_per_rank = a.shape
    _, K = (
        M_per_rank * world_size,
        K_per_rank,
    )
    _, _, N = b.shape
    dtype = a.dtype
    assert (dtype == torch.float16 or dtype == torch.float8_e4m3fn), "Currently only used for float16 or float8_e4m3fn"

    ag_stream = torch.cuda.Stream() if ag_stream is None else ag_stream
    group_gemm_stream = (torch.cuda.current_stream() if group_gemm_stream is None else group_gemm_stream)
    current_stream = torch.cuda.current_stream()
    ag_stream.wait_stream(current_stream)
    group_gemm_stream.wait_stream(current_stream)

    cp_engine_producer_all_gather_full_mesh_push(
        rank,
        world_size,
        a,
        a_tensors,
        block_barriers,
        ag_stream,
    )

    with torch.cuda.stream(group_gemm_stream):
        EM = full_sorted_token_ids.shape[0]

        grid = lambda META: (triton.cdiv(EM, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]), )
        compiled = kernel_consumer_m_parallel_scatter_group_gemm[grid](
            a_buffer,
            b,
            c,  #
            block_barriers[rank],
            full_sorted_token_ids,
            full_token_expert_ids,
            full_num_tokens_post_padded_list,
            block_barrier_ids,
            full_numel,
            EM,
            N,
            K,  #
            a_buffer.stride(0),
            a_buffer.stride(1),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            GROUP_SIZE_M,
            topk,
            torch_dtype_to_triton_dtype(dtype),
            rank,
            world_size,
            num_stages=stages,
            num_warps=warps,
        )

    current_stream.wait_stream(ag_stream)
    current_stream.wait_stream(group_gemm_stream)
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    block_barriers[rank].zero_()
    return compiled


@dataclass
class MoE_AllGatherGroupGEMMTensorParallelContext:
    rank: int
    num_ranks: int
    num_experts: int
    topk: int
    full_num_tokens: int
    sorted_topk_ids: torch.Tensor
    aligned_expert_ids: torch.Tensor
    aligned_barrier_ids: torch.Tensor
    aligned_num_tokens: torch.Tensor
    workspace_tensors: List[torch.Tensor]
    barrier_tensors: List[torch.Tensor]
    ag_stream: Optional[torch.cuda.streams.Stream] = None
    group_gemm_stream: Optional[torch.cuda.streams.Stream] = None
    BLOCK_M: int = 128
    BLOCK_N: int = 256
    BLOCK_K: int = 64
    GROUP_SIZE_M: int = 8
    stages: int = 3
    warps: int = 8

    def update(
        self,
        full_topk_ids: torch.Tensor,
        rank,
        num_ranks,
        num_experts,
        topk,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=64,
        GROUP_SIZE_M=8,
        stages=3,
        warps=8,
        ag_stream=None,
        group_gemm_stream=None,
    ):
        self.rank = rank
        self.num_ranks = num_ranks
        self.num_experts = num_experts
        self.topk = topk
        self.full_num_tokens = full_topk_ids.numel()
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.GROUP_SIZE_M = GROUP_SIZE_M
        self.stages = stages
        self.warps = warps
        self.ag_stream = ag_stream
        self.group_gemm_stream = group_gemm_stream
        M_per_rank = full_topk_ids.shape[0] // num_ranks
        (
            self.sorted_topk_ids,
            self.aligned_expert_ids,
            self.aligned_barrier_ids,
            _,
            self.aligned_num_tokens,
        ) = sort_topk_ids_align_block_size(full_topk_ids, num_experts, num_ranks, M_per_rank, BLOCK_M)


def create_ag_group_gemm_intra_node_context(
    tp_pg,
    tensor_A,
    tensor_B,
    full_topk_ids,
    num_experts,
    max_M=2**14,
    BLOCK_M=128,
    BLOCK_N=256,
    BLOCK_K=64,
    GROUP_SIZE_M=8,
    stages=3,
    warps=8,
    ag_stream=None,
    group_gemm_stream=None,
):
    """create context for allgather gemm intra-node

    Args:
        tensor_A (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        tensor_B (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        topk_id (torch.Tensor<int32_t>): local topk ids. shape: [M_per_rank, topk]
        max_M: max value of M
        BLOCK_M (int, optional): Group GEMM tiling factor for M dim. Defaults to 128.
        BLOCK_N (int, optional): Group GEMM tiling factor for N dim. Defaults to 256.
        BLOCK_K (int, optional): Group GEMM tiling factor for K dim. Defaults to 64.
        GROUP_SIZE_M (int, optional): Group size of block for M dim (not size of group GEMM). Defaults to 8.
        stages (int, optional): GEMM async-copy stages. Defaults to 3.
        warps (int, optional): No.of used warps. Defaults to 8.
        ag_stream (torch.cuda.streams.Stream, optional): The stream used for allgather, if not provided, create a new one. Defaults to None.
        group_gemm_stream (torch.cuda.streams.Stream, optional): The stream used for group gemm, if not provided, use current stream. Defaults to None.

    Returns:
        MoE_AllGatherGroupGEMMTensorParallelContext
    """
    M_per_rank, K = tensor_A.shape
    _, topk = full_topk_ids.shape
    assert (
        tensor_B.shape[1] == K), f"tensor_B should has shape (col_major) [N_per_rank, {K}], but get [{tensor_B.shape}]"
    assert tensor_A.dtype == tensor_B.dtype
    dtype = tensor_A.dtype
    num_ranks = tp_pg.size()
    rank = tp_pg.rank()

    workspaces = pynvshmem.nvshmem_create_tensor_list_intra_node([max_M, K], dtype)
    barriers = pynvshmem.nvshmem_create_tensor_list_intra_node([num_ranks], torch.int32)

    # sort the full topk ids.
    (
        full_sorted_token_ids,
        full_token_expert_ids,
        block_wait_barriers,
        _,
        full_num_tokens_post_padded_list,
    ) = sort_topk_ids_align_block_size(full_topk_ids, num_experts, num_ranks, M_per_rank, BLOCK_M)

    barriers[rank].fill_(0)
    current_stream = torch.cuda.current_stream()
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    ret = MoE_AllGatherGroupGEMMTensorParallelContext(
        rank=rank,
        num_ranks=num_ranks,
        num_experts=num_experts,
        topk=topk,
        full_num_tokens=full_topk_ids.numel(),
        sorted_topk_ids=full_sorted_token_ids,
        aligned_expert_ids=full_token_expert_ids,
        aligned_barrier_ids=block_wait_barriers,
        aligned_num_tokens=full_num_tokens_post_padded_list,
        workspace_tensors=workspaces,
        barrier_tensors=barriers,
        ag_stream=ag_stream,
        group_gemm_stream=group_gemm_stream,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        stages=stages,
        warps=warps,
    )

    return ret


def ag_group_gemm_intra_node(a, b, ctx=None, full_topk_ids=None, num_experts=None):
    """intra-node allgather group gemm

    Allgather global matrix A and do matmul with local matrix B, produces local matrix C

    Args:
        a (torch.Tensor<float>): local matmul A matrix. shape: [M_per_rank, K]
        b (torch.Tensor<float>): local matmul B matrix. shape: [E, K, N_per_rank]
        ctx: (Optional[MoE_AllGatherGroupGEMMTensorParallelContext]): if not provided, created immediately

    Returns:
        c (torch.Tensor<float>): local matmul C matrix. shape: [M * topk, N_per_rank]
    """
    if ctx is None:
        assert full_topk_ids is not None and num_experts is not None
        tp_pg = TP_GROUP()
        ctx = create_ag_group_gemm_intra_node_context(tp_pg, a, b, full_topk_ids, num_experts)
    M_per_rank, _ = a.shape
    _, _, N_per_rank = b.shape
    C = torch.empty(
        [ctx.topk * M_per_rank * ctx.num_ranks, N_per_rank],
        dtype=a.dtype,
        device=a.device,
    )

    rowise_intranode_ag_scatter_group_gemm(
        a,
        ctx.workspace_tensors[ctx.rank],
        ctx.workspace_tensors,
        b,
        C,
        ctx.barrier_tensors,
        ctx.topk,
        ctx.sorted_topk_ids,
        ctx.aligned_expert_ids,
        ctx.aligned_barrier_ids,
        ctx.aligned_num_tokens,
        ctx.full_num_tokens,
        ctx.rank,
        ctx.num_ranks,
        ag_stream=ctx.ag_stream,
        group_gemm_stream=ctx.group_gemm_stream,
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N=ctx.BLOCK_N,
        BLOCK_K=ctx.BLOCK_K,
        GROUP_SIZE_M=ctx.GROUP_SIZE_M,
        stages=ctx.stages,
        warps=ctx.warps,
    )

    return C
