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
import torch
import triton
import triton.language as tl
from triton.language import core as tlc
import triton_dist.language as dl

from typing import Optional
import itertools

import triton_dist
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, supports_p2p_native_atomic, torch_stream_max_priority
from triton_dist.kernels.nvidia.common_ops import barrier_all_intra_node_atomic_cas_block, _wait_eq_cuda
from triton_dist.language.extra.language_extra import tid, __syncthreads, st


@tlc.extern
def load_v4_b32(ptr, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="ld.global.v4.b32 {$0,$1,$2,$3}, [$4];",
        constraints=("=r,=r,=r,=r,l"),
        args=[ptr],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@tlc.extern
def store_v4_b32(ptr, val0, val1, val2, val3, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="""
        st.global.v4.b32 [$1], {$2,$3,$4,$5};
        mov.u32 $0, 0;
        """,
        constraints=("=r,l,r,r,r,r"),  # no use output
        args=[ptr, val0, val1, val2, val3],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@tlc.extern
def load_v4_b32_cond(ptr, mask, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $5, 1;
            @%p0 ld.global.v4.b32 {$0,$1,$2,$3}, [$4];
        }
        """,
        constraints=("=r,=r,=r,=r,l,r"),
        args=[ptr, mask.to(tl.int32, _semantic=_semantic)],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@tlc.extern
def store_v4_b32_cond(ptr, val0, val1, val2, val3, mask, _semantic=None):
    return tl.inline_asm_elementwise(
        asm="""
        {
            .reg .pred %p0;
            setp.eq.s32 %p0, $6, 1;
            @%p0 st.global.v4.b32 [$1], {$2,$3,$4,$5};
            mov.u32 $0, 0;
        }
        """,
        constraints=("=r,l,r,r,r,r,r"),  # no use output
        args=[ptr, val0, val1, val2, val3, mask.to(tl.int32, _semantic=_semantic)],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (tile_id % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def _matmul_launch_metadata(grid, kernel, args):
    ret = {}
    M, N, K, WS = args["M"], args["N"], args["K"], args.get("WARP_SPECIALIZE", False)
    ws_str = "_ws" if WS else ""
    ret["name"] = f"{kernel.name}{ws_str} [M={M}, N={N}, K={K}]"
    if "c_ptr" in args:
        bytes_per_elem = args["c_ptr"].element_size()
    else:
        bytes_per_elem = 1 if args["FP8_OUTPUT"] else 2
    ret[f"flops{bytes_per_elem * 8}"] = 2. * M * N * K
    ret["bytes"] = bytes_per_elem * (M * K + N * K + M * N)
    return ret


def _kernel_consumer_gemm_persistent_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]

    return f"cutlass_triton3x_sm{cap_major}{cap_minor}_a2a_consumer_gemm_persistent_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_ntn"


@triton.jit(do_not_specialize=["sp_rank"], launch_metadata=_matmul_launch_metadata,
            repr=_kernel_consumer_gemm_persistent_repr)
def matmul_kernel_descriptor_persistent(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,  #
    gemm_barrier_ptr,
    sp_rank,
    sp_size: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,  #
    A2A_TILE_M: tl.constexpr,
    A2A_TILE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    EPILOGUE_SUBTILE: tl.constexpr,  #
    NUM_GEMM_SMS: tl.constexpr,  #
    WARP_SPECIALIZE: tl.constexpr,  #
    HAS_BIAS: tl.constexpr,
):
    # Matmul using TMA and device-side descriptor creation
    dtype = c_ptr.dtype.element_ty
    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n

    tl.static_assert(K % sp_size == 0, f"K {K} must be divisible by sp_size {sp_size}")
    K_per_sp_rank: tl.constexpr = K // sp_size
    tl.static_assert(K_per_sp_rank % BLOCK_SIZE_K == 0,
                     f"K_per_sp_rank {K_per_sp_rank} must be divisible by BLOCK_SIZE_K {BLOCK_SIZE_K}")
    k_tiles: tl.constexpr = K // BLOCK_SIZE_K

    tl.static_assert(A2A_TILE_N % BLOCK_SIZE_K == 0,
                     f"A2A_TILE_N {A2A_TILE_N} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_K}")
    NUM_K_PER_TILE: tl.constexpr = A2A_TILE_N // BLOCK_SIZE_K
    # This is used for k-swizzle
    # k_tiles_per_rank: tl.constexpr = K_per_sp_rank // BLOCK_SIZE_K
    # k_vec_tiles_per_rank: tl.constexpr = k_tiles_per_rank // NUM_K_PER_TILE

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
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2],
    )

    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue
    tile_id_c = start_pid - NUM_GEMM_SMS
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS, flatten=False, warp_specialize=WARP_SPECIALIZE):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        offs_am = pid_m * BLOCK_SIZE_M
        offs_bn = pid_n * BLOCK_SIZE_N

        chunk_beg = pid_m * BLOCK_SIZE_M // A2A_TILE_M
        chunk_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // A2A_TILE_M

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for ki in range(k_tiles):
            # k-swizzle: as the all-to-all comes in non-serial order, a swizzle may help in performance
            # vec = NUM_K_PER_TILE
            # vec = 4
            # ki_vec = ki // vec
            # ki_elem = ki % vec
            # swizzle_ki_vec = (ki_vec % sp_size + sp_rank) % sp_size
            # ki = (swizzle_ki_vec * k_vec_tiles_per_rank + ki_vec // sp_size) * vec + ki_elem

            if ki % NUM_K_PER_TILE == 0:
                for chunk_id in range(chunk_beg, chunk_end + 1):
                    token = dl.wait(gemm_barrier_ptr + chunk_id * (k_tiles // NUM_K_PER_TILE) + ki // NUM_K_PER_TILE, 1,
                                    scope="sys", semantic="acquire", waitValue=1)
                    a_desc = dl.consume_token(a_desc, token)
            offs_k = ki * BLOCK_SIZE_K
            a = a_desc.load([offs_am, offs_k])
            b = b_desc.load([offs_bn, offs_k])
            accumulator = tl.dot(a, b.T, accumulator)

        tile_id_c += NUM_GEMM_SMS
        pid_m, pid_n = _compute_pid(tile_id_c, num_pid_in_group, num_pid_m, GROUP_SIZE_M, NUM_GEMM_SMS)
        offs_cm = pid_m * BLOCK_SIZE_M
        offs_cn = pid_n * BLOCK_SIZE_N

        if HAS_BIAS:
            offs_bias_n = tl.arange(0, BLOCK_SIZE_N)
            bias_data = tl.load(bias_ptr + offs_cn + offs_bias_n, mask=(offs_cn + offs_bias_n < N)).to(tl.float32)
            accumulator = accumulator + bias_data[None, :]

        if EPILOGUE_SUBTILE:
            acc = tl.reshape(accumulator, (BLOCK_SIZE_M, 2, BLOCK_SIZE_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c_desc.store([offs_cm, offs_cn], c0)
            c1 = acc1.to(dtype)
            c_desc.store([offs_cm, offs_cn + BLOCK_SIZE_N // 2], c1)
        else:
            c = accumulator.to(dtype)
            c_desc.store([offs_cm, offs_cn], c)


def matmul_descriptor_persistent(sp_rank, sp_size, a, b, bias, c, gemm_barrier, gemm_config: triton.Config,
                                 warp_specialize: bool = False):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"  # b is transposed
    assert a.dtype == b.dtype, "Incompatible dtypes"

    M, K = a.shape
    N, K = b.shape

    # TMA descriptors require a global memory allocation
    def alloc_fn(size: int, alignment: int, stream: Optional[int]):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (min(META["NUM_GEMM_SMS"],
                             triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"])), )
    matmul_kernel_descriptor_persistent[grid](
        a,
        b,
        bias,
        c,
        gemm_barrier,  #
        sp_rank,
        sp_size,
        M,
        N,
        K,  #
        EPILOGUE_SUBTILE=False,  #
        WARP_SPECIALIZE=warp_specialize,  #
        **gemm_config.all_kwargs(),  #
        HAS_BIAS=1 if bias is not None else 0,
    )
    return c


def _kernel_consumer_gemm_repr(proxy):
    constexprs = proxy.constants
    cap_major, cap_minor = torch.cuda.get_device_capability()
    a_dtype = proxy.signature["a_ptr"].lstrip("*")
    b_dtype = proxy.signature["b_ptr"].lstrip("*")
    c_dtype = proxy.signature["c_ptr"].lstrip("*")
    BM, BN, BK = constexprs["BLOCK_SIZE_M"], constexprs["BLOCK_SIZE_N"], constexprs["BLOCK_SIZE_K"]

    return f"cutlass_triton3x_sm{cap_major}{cap_minor}_a2a_consumer_gemm_tensorop_{a_dtype}_{b_dtype}_{c_dtype}_{BM}x{BN}x{BK}_ntn"


@triton.jit(do_not_specialize=["sp_rank"], launch_metadata=_matmul_launch_metadata, repr=_kernel_consumer_gemm_repr)
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,  #
    gemm_barrier_ptr,
    sp_rank,
    sp_size: tl.constexpr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,  #
    stride_am,
    stride_ak,  #
    stride_bn,
    stride_bk,  #
    stride_cm,
    stride_cn,  #
    A2A_TILE_M: tl.constexpr,
    A2A_TILE_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,  #
    BLOCK_SIZE_N: tl.constexpr,  #
    BLOCK_SIZE_K: tl.constexpr,  #
    GROUP_SIZE_M: tl.constexpr,  #
    NUM_GEMM_SMS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    tl.static_assert(K % sp_size == 0, f"K {K} must be divisible by sp_size {sp_size}")
    K_per_sp_rank: tl.constexpr = K // sp_size
    tl.static_assert(K_per_sp_rank % BLOCK_SIZE_K == 0,
                     f"K_per_sp_rank {K_per_sp_rank} must be divisible by BLOCK_SIZE_K {BLOCK_SIZE_K}")
    k_tiles: tl.constexpr = K // BLOCK_SIZE_K

    tl.static_assert(A2A_TILE_N % BLOCK_SIZE_K == 0,
                     f"A2A_TILE_N {A2A_TILE_N} must be divisible by BLOCK_SIZE_N {BLOCK_SIZE_K}")
    NUM_K_PER_TILE: tl.constexpr = A2A_TILE_N // BLOCK_SIZE_K

    start_m = pid_m * BLOCK_SIZE_M
    start_n = pid_n * BLOCK_SIZE_N

    offs_am = start_m + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = start_n + tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(offs_am < M, offs_am, 0)
    offs_bn = tl.where(offs_bn < N, offs_bn, 0)

    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(offs_bn, BLOCK_SIZE_N), BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    chunk_beg = pid_m * BLOCK_SIZE_M // A2A_TILE_M
    chunk_end = (min((pid_m + 1) * BLOCK_SIZE_M, M) - 1) // A2A_TILE_M

    for k in range(0, k_tiles):
        if k % NUM_K_PER_TILE == 0:
            for chunk_id in range(chunk_beg, chunk_end + 1):
                token = dl.wait(gemm_barrier_ptr + chunk_id * (k_tiles // NUM_K_PER_TILE) + k // NUM_K_PER_TILE, 1,
                                scope="sys", semantic="acquire", waitValue=1)
                a_ptrs = dl.consume_token(a_ptrs, token)
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b.T, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(c_ptr.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(sp_rank, sp_size, a, b, c, gemm_barrier, gemm_config: triton.Config):
    # Check constraints.
    assert a.shape[1] == b.shape[1], "Incompatible dimensions"
    assert a.dtype == b.dtype, "Incompatible dtypes"
    M, K = a.shape
    N, _ = b.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]), )
    matmul_kernel[grid](
        a, b, c,  #
        gemm_barrier, sp_rank, sp_size, M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        **gemm_config.all_kwargs(),  #
    )
    return c


@triton_dist.jit(do_not_specialize=["rank", "sp_rank"])
def kernel_all2all_push_intra_node_nvl(
    attn_out_ptr,
    a2a_out_ptr,
    cum_seqlen_cpu_tuple,
    cum_seqlen_gpu_ptr,
    barrier_ptr,
    intra_node_sync_buf_ptr,
    local_head: tl.constexpr,
    global_head,
    head_dim: tl.constexpr,
    sp_size: tl.constexpr,
    rank,
    sp_rank,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_COMM_SM: tl.constexpr,
    FUSE_SYNC: tl.constexpr,
    SUPPORT_ATOMIC: tl.constexpr,
    VEC: tl.constexpr,
    SKIP_BARRIER: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if SKIP_BARRIER:
        num_pids = tl.num_programs(0)
        empty_pids = num_pids - NUM_COMM_SM
        if pid < empty_pids:
            return
        pid = pid - empty_pids

    if FUSE_SYNC:
        tl.static_assert(SUPPORT_ATOMIC, "FUSE_SYNC requires SUPPORT_ATOMIC to be True")
        barrier_all_intra_node_atomic_cas_block(sp_rank, rank, sp_size, intra_node_sync_buf_ptr + pid * sp_size)

    for i in tl.static_range(sp_size + 1):
        tl.store(cum_seqlen_gpu_ptr + i, cum_seqlen_cpu_tuple[i])
    __syncthreads()

    rank_offset = rank - sp_rank

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N // VEC)

    if NUM_COMM_SM >= sp_size:
        tl.static_assert(NUM_COMM_SM % sp_size == 0,
                         f"NUM_COMM_SM {NUM_COMM_SM} must be divisible by sp_size {sp_size}")
        NUM_SM_PER_SP: tl.constexpr = NUM_COMM_SM // sp_size
        NUM_SP_PER_SM: tl.constexpr = 1
    else:
        tl.static_assert(sp_size % NUM_COMM_SM == 0,
                         f"sp_size {sp_size} must be divisible by NUM_COMM_SM {NUM_COMM_SM}")
        NUM_SM_PER_SP: tl.constexpr = 1
        NUM_SP_PER_SM: tl.constexpr = sp_size // NUM_COMM_SM

    for tile in range(NUM_SP_PER_SM):
        remote_sp_rank = pid * NUM_SP_PER_SM // NUM_SM_PER_SP + tile
        remote_rank = remote_sp_rank + rank_offset
        remote_a2a_out_ptr = dl.symm_at(a2a_out_ptr, remote_rank)
        remote_barrier_ptr = dl.symm_at(barrier_ptr, remote_rank)
        pid_in_sp = pid % NUM_SM_PER_SP
        seq_beg = tl.load(cum_seqlen_gpu_ptr + remote_sp_rank)
        seq_end = tl.load(cum_seqlen_gpu_ptr + remote_sp_rank + 1)
        remote_seq_len = seq_end - seq_beg
        num_tile_m = tl.cdiv(remote_seq_len, BLOCK_M)
        tl.static_assert(local_head * head_dim % BLOCK_N == 0,
                         f"local_head * head_dim {local_head * head_dim} must be divisible by BLOCK_N {BLOCK_N}")
        num_tile_n = local_head * head_dim // BLOCK_N

        for tile_id_m_outer_n_tail in range(0, tl.cdiv(num_tile_m, GROUP_SIZE_M) * num_tile_n):
            tile_id_m_outer_tail = tile_id_m_outer_n_tail // num_tile_n
            tile_id_n_tail = tile_id_m_outer_n_tail % num_tile_n
            for tile_id_m_inner_tail in range(pid_in_sp, GROUP_SIZE_M, NUM_SM_PER_SP):
                tile_id_m_tail = tile_id_m_outer_tail * GROUP_SIZE_M + tile_id_m_inner_tail
                if tile_id_m_tail < num_tile_m:
                    attn_offs_m = seq_beg + tile_id_m_tail * BLOCK_M + offs_m
                    attn_mask_m = attn_offs_m < seq_end
                    attn_offs_n = tile_id_n_tail * BLOCK_N + offs_n * VEC
                    data0, data1, data2, data3 = load_v4_b32_cond(
                        attn_out_ptr + attn_offs_m[:, None] * local_head * head_dim + attn_offs_n[None, :],
                        mask=attn_mask_m[:, None])

                    out_offs_m = tile_id_m_tail * BLOCK_M + offs_m
                    out_mask_m = out_offs_m < remote_seq_len
                    out_offs_n = sp_rank * local_head * head_dim + tile_id_n_tail * BLOCK_N + offs_n * VEC
                    store_v4_b32_cond(
                        remote_a2a_out_ptr + out_offs_m[:, None] * global_head * head_dim + out_offs_n[None, :], data0,
                        data1, data2, data3, mask=out_mask_m[:, None])

                    if not SKIP_BARRIER:
                        __syncthreads()
                        notify_barrier_ptr = remote_barrier_ptr + tile_id_m_tail * num_tile_n * sp_size + sp_rank * num_tile_n + tile_id_n_tail
                        thread_idx = tid(0)
                        if thread_idx == 0:
                            st(notify_barrier_ptr, 1, scope="sys", semantic="release")


class SpUlysessOAll2AllGemmKernel:

    def __init__(
        self,
        world_group: torch.distributed.ProcessGroup,
        nnodes: int,
        sp_size: int,
        max_batch: int,
        num_head: int,
        max_seqlen: int,
        head_dim: int,
        max_num_comm_buf: int,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        a2a_only: bool = True,
        fuse_sync: bool = True,
        use_persistent: bool = True,
    ):
        self.world_group = world_group
        self.world_size = world_group.size()
        self.rank = world_group.rank()
        self.nnodes = nnodes
        assert self.world_size % nnodes == 0, f"world_size {self.world_size} must be divisible by nnodes {nnodes}"
        self.local_world_size = self.world_size // nnodes
        self.local_rank = self.rank % self.local_world_size
        self.sp_size = sp_size
        assert self.local_world_size % self.sp_size == 0, f"local_world_size {self.local_world_size} must be divisible by sp_size {sp_size}"
        self.sp_rank = self.local_rank % self.sp_size
        self.max_batch = max_batch
        self.num_head = num_head
        self.max_seqlen = max_seqlen
        self.head_dim = head_dim
        self.max_num_comm_buf = max_num_comm_buf
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.a2a_only = a2a_only
        assert self.a2a_only, "Only support a2a_only mode"
        self.fuse_sync = fuse_sync
        self.use_persistent = use_persistent

        self.compute_stream = torch.cuda.Stream(priority=torch_stream_max_priority())
        self.cp_event = torch.cuda.Event(enable_timing=False)
        self.ready_event = torch.cuda.Event(enable_timing=False)
        self.compute_event = torch.cuda.Event(enable_timing=False)

        self.p2p_atomic_supported = supports_p2p_native_atomic()
        self.max_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

        # GEMM config
        if self.use_persistent:
            self.BLOCK_SIZE_M = 128
            self.BLOCK_SIZE_N = 256
            self.BLOCK_SIZE_K = 64
            self.GROUP_SIZE_M = 4
            self.A2A_TILE_M = 128
            self.A2A_TILE_N = 256
            self.max_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            self.num_warps = 8
            self.num_stages = 3
            self.warp_specialize = False

            # For H20
            if self.max_gemm_sms < 100:
                self.BLOCK_SIZE_N = 128
                self.BLOCK_SIZE_K = 32
        else:
            self.BLOCK_SIZE_M = 128
            self.BLOCK_SIZE_N = 128
            self.BLOCK_SIZE_K = 32
            self.GROUP_SIZE_M = 8
            self.A2A_TILE_M = 128
            self.A2A_TILE_N = 256
            self.max_gemm_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            self.num_warps = 4
            self.num_stages = 4

        self.init_symm_buffer()
        self.init_local_buffer()

    def __del__(self):
        self.finalize()

    def finalize(self):
        self.deinit_symm_buffer()

    def init_symm_buffer(self):
        max_local_seq = self.max_seqlen // self.sp_size
        self._comm_output_buffer = nvshmem_create_tensor(
            [self.max_num_comm_buf, self.max_batch, max_local_seq, self.num_head * self.head_dim], self.input_dtype)
        self._barrier_buffer = nvshmem_create_tensor(
            [triton.cdiv(self.max_batch * self.max_seqlen, self.BLOCK_SIZE_M) * self.num_head], torch.int32)
        self._barrier_buffer.zero_()
        self._intra_node_sync_buffer = nvshmem_create_tensor([self.sp_size * self.max_sms], torch.int32)
        self._intra_node_sync_buffer.zero_()
        self._sp_group_sync_buffer = nvshmem_create_tensor([self.world_size], torch.int32)
        self._sp_group_sync_buffer.zero_()

    def deinit_symm_buffer(self):
        if hasattr(self, "_comm_output_buffer"):
            nvshmem_free_tensor_sync(self._comm_output_buffer)
            del self._comm_output_buffer
        if hasattr(self, "_barrier_buffer"):
            nvshmem_free_tensor_sync(self._barrier_buffer)
            del self._barrier_buffer
        if hasattr(self, "_intra_node_sync_buffer"):
            nvshmem_free_tensor_sync(self._intra_node_sync_buffer)
            del self._intra_node_sync_buffer
        if hasattr(self, "_sp_group_sync_buffer"):
            nvshmem_free_tensor_sync(self._sp_group_sync_buffer)
            del self._sp_group_sync_buffer

    def init_local_buffer(self):
        self._cum_seqlen_gpu = torch.empty([self.sp_size + 1], dtype=torch.int32, device="cuda")

    def sp_group_barrier_all_intra_node(self, stream=None):
        stream = torch.cuda.current_stream() if stream is None else stream
        sp_local_rank = self.local_rank % self.sp_size
        with torch.cuda.stream(stream):
            barrier_all_intra_node_atomic_cas_block[(1, )](sp_local_rank, self.rank, self.sp_size,
                                                           self._sp_group_sync_buffer)

    def reset_cusum_seq_lens(self, local_seqlen, seq_lens_cpu=None):
        if seq_lens_cpu is None:
            seq_lens_cpu = [local_seqlen] * self.sp_size
        else:
            seq_lens_cpu = seq_lens_cpu.tolist()
        assert local_seqlen == seq_lens_cpu[
            self.local_rank % self.
            sp_size], f"local_seqlen {local_seqlen} != seq_lens_cpu[{self.local_rank % self.sp_size}]={seq_lens_cpu[self.local_rank % self.sp_size]}"
        cum_seqlen_cpu = [0] + list(itertools.accumulate(seq_lens_cpu))
        self._cum_seq_len_cpu_tuple = tuple(cum_seqlen_cpu)

    def forward(self, inputs: torch.Tensor, weight: torch.Tensor, seq_lens_cpu: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None, output: Optional[torch.Tensor] = None,
                a2a_output: Optional[torch.Tensor] = None, transpose_weight: bool = False, num_comm_sms: int = -1,
                sm_margin: int = 0):
        if num_comm_sms == -1:
            num_comm_sms = self.world_size
        assert num_comm_sms >= 0, "num_comm_sms must be non-negative"
        assert len(weight.shape) == 2, f"weight must be 2D tensor, got {len(weight.shape)}D"
        assert len(inputs.shape) == 4, f"inputs must be 4D tensor, got {len(inputs.shape)}D"
        bs, total_seq_len, local_head, head_dim = inputs.shape
        assert head_dim == self.head_dim, f"head_dim {head_dim} must be equal to self.head_dim {self.head_dim}"
        assert weight.is_contiguous(), "weight must be contiguous"
        assert inputs.is_contiguous(), "inputs must be contiguous"
        assert not transpose_weight, "transpose_weight is not supported in this kernel"

        if not transpose_weight:
            N = weight.shape[0]
            K = weight.shape[1]
        else:
            N = weight.shape[1]
            K = weight.shape[0]

        if seq_lens_cpu is not None:
            assert seq_lens_cpu.is_cpu, "seq_lens_cpu must be a CPU tensor"
            assert seq_lens_cpu.dtype == torch.int32, "seq_lens_cpu must be int32"
            assert seq_lens_cpu.is_contiguous(), "seq_lens_cpu must be contiguous"

            seq_lens_cpu_tuple = tuple(seq_lens_cpu.tolist())
            local_seq_len = seq_lens_cpu_tuple[self.sp_rank]
            M = local_seq_len * bs
        else:
            assert total_seq_len % self.sp_size == 0, f"total_seq_len {total_seq_len} must be divisible by sp_size {self.sp_size}"
            local_seq_len = total_seq_len // self.sp_size
            M = local_seq_len * bs

        self.reset_cusum_seq_lens(local_seqlen=local_seq_len, seq_lens_cpu=seq_lens_cpu)

        gemm_input_a = self._comm_output_buffer.view(-1)[:M * K].view([M, K])

        cur_stream = torch.cuda.current_stream()

        if output is None:
            output = torch.empty([bs, local_seq_len, N], device=inputs.device, dtype=self.output_dtype)

        self._barrier_buffer.zero_()
        if not self.fuse_sync:
            self.sp_group_barrier_all_intra_node(cur_stream)

        self.ready_event.record(cur_stream)
        self.compute_stream.wait_event(self.ready_event)

        grid = (num_comm_sms, )
        kernel_all2all_push_intra_node_nvl[grid](
            inputs,
            gemm_input_a,
            self._cum_seq_len_cpu_tuple,
            self._cum_seqlen_gpu,
            self._barrier_buffer,
            self._intra_node_sync_buffer,  # no need to initialize
            local_head,
            local_head * self.sp_size,
            self.head_dim,
            self.sp_size,
            self.rank,
            self.sp_rank,
            self.A2A_TILE_M,
            self.A2A_TILE_N,
            self.GROUP_SIZE_M,
            num_comm_sms,
            self.fuse_sync,
            self.p2p_atomic_supported,
            VEC=(16 // inputs.dtype.itemsize),
            num_warps=32,
        )

        assert len(output.shape) == 3, f"output must be 4D tensor, got {len(output.shape)}D"
        assert output.shape[0] == bs, f"output batch size {output.shape[0]} must be equal to input batch size {bs}"
        assert output.shape[
            1] == local_seq_len, f"output seq_len {output.shape[1]} must be equal to local_seq_len {local_seq_len}"
        assert output.shape[2] == N, f"output head {output.shape[2]} must be equal to output size {N}"
        assert output.is_contiguous(), "output must be contiguous"

        assert self.max_gemm_sms - num_comm_sms - sm_margin > 0, f"max_gemm_sms {self.max_gemm_sms} - num_comm_sms {num_comm_sms} - sm_margin {sm_margin} must be greater than 0"
        gemm_config = triton.Config(
            {
                'BLOCK_SIZE_M': self.BLOCK_SIZE_M, 'BLOCK_SIZE_N': self.BLOCK_SIZE_N, 'BLOCK_SIZE_K': self.BLOCK_SIZE_K,
                'GROUP_SIZE_M': self.GROUP_SIZE_M, 'A2A_TILE_M': self.A2A_TILE_M, 'A2A_TILE_N': self.A2A_TILE_N,
                'NUM_GEMM_SMS': self.max_gemm_sms - num_comm_sms - sm_margin
            }, num_stages=self.num_stages, num_warps=self.num_warps)

        if (os.getenv("CUDA_DEVICE_MAX_CONNECTIONS", -1) != 1) and (not self.use_persistent):
            # to aovid dead-lock
            _wait_eq_cuda(self._barrier_buffer, 1, self.compute_stream)

        with torch.cuda.stream(self.compute_stream):
            if self.use_persistent:
                matmul_descriptor_persistent(self.sp_rank, self.sp_size, gemm_input_a, weight, bias, output,
                                             self._barrier_buffer, gemm_config, self.warp_specialize)
            else:
                assert bias is None
                matmul(self.sp_rank, self.sp_size, gemm_input_a, weight.view([-1, K]), output.view([M, -1]),
                       self._barrier_buffer, gemm_config)

        if a2a_output is not None:
            assert a2a_output.shape == (
                bs, local_seq_len, local_head * self.sp_size, head_dim
            ), f"a2a_output shape {a2a_output.shape} must be equal to (bs, local_seq_len, local_head * self.sp_size, head_dim) ({bs}, {local_seq_len}, {local_head * self.sp_size}, {head_dim})"
            assert a2a_output.is_contiguous(), f"a2a_output must be contiguous, got {a2a_output.shape}"
            a2a_output.copy_(gemm_input_a.view(bs, local_seq_len, local_head * self.sp_size * head_dim))
            ret = (output, a2a_output)
        else:
            ret = (output, )

        self.compute_event.record(self.compute_stream)
        cur_stream.wait_event(self.compute_event)

        return ret

    def post_attn_a2a(
        self,
        inputs: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        return_comm_buf: bool = False,
        comm_buf_idx: int = 0,
        num_comm_sms: int = -1,
    ):
        if num_comm_sms == -1:
            num_comm_sms = self.world_size
        assert num_comm_sms >= 0, "num_comm_sms must be non-negative"
        assert len(inputs.shape) == 4, f"inputs must be 4D tensor, got {len(inputs)}D"
        bs, total_seq_len, local_head, head_dim = inputs.shape
        assert head_dim == self.head_dim, f"head_dim {head_dim} must be equal to self.head_dim {self.head_dim}"
        assert inputs.is_contiguous(), f"inputs must be contiguous, got {inputs.shape}"

        if seq_lens_cpu is not None:
            assert seq_lens_cpu.is_cpu, "seq_lens_cpu must be a CPU tensor"
            assert seq_lens_cpu.dtype == torch.int32, "seq_lens_cpu must be int32"
            assert seq_lens_cpu.is_contiguous(), "seq_lens_cpu must be contiguous"

            seq_lens_cpu_tuple = tuple(seq_lens_cpu.tolist())
            local_seq_len = seq_lens_cpu_tuple[self.sp_rank]
            M = local_seq_len * bs
        else:
            assert total_seq_len % self.sp_size == 0, f"total_seq_len {total_seq_len} must be divisible by sp_size {self.sp_size}"
            local_seq_len = total_seq_len // self.sp_size
            M = local_seq_len * bs

        K = local_head * self.sp_size * head_dim

        self.reset_cusum_seq_lens(local_seqlen=local_seq_len, seq_lens_cpu=seq_lens_cpu)

        assert comm_buf_idx < self.max_num_comm_buf, f"comm_buf_idx {comm_buf_idx} must be less than num_comm_buf {self.max_num_comm_buf}"
        gemm_input_a = self._comm_output_buffer[comm_buf_idx].view(-1)[:M * K].view([M, K])

        cur_stream = torch.cuda.current_stream()

        if not self.fuse_sync:
            self.sp_group_barrier_all_intra_node(cur_stream)

        grid = (self.max_gemm_sms, )
        kernel_all2all_push_intra_node_nvl[grid](
            inputs,
            gemm_input_a,
            self._cum_seq_len_cpu_tuple,
            self._cum_seqlen_gpu,
            self._barrier_buffer,
            self._intra_node_sync_buffer,  # no need to initialize
            local_head,
            local_head * self.sp_size,
            self.head_dim,
            self.sp_size,
            self.rank,
            self.sp_rank,
            256,
            256,
            16,
            num_comm_sms,
            self.fuse_sync,
            self.p2p_atomic_supported,
            VEC=(16 // inputs.dtype.itemsize),
            SKIP_BARRIER=True,
            num_warps=32,
        )

        if return_comm_buf:
            return gemm_input_a
        else:
            self.sp_group_barrier_all_intra_node(cur_stream)
            return gemm_input_a.clone()

    def post_attn_a2a_no_cpy(
        self,
        inputs: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
        comm_buf_idx: int = 0,
        num_comm_sms: int = -1,
    ):
        return self.post_attn_a2a(
            inputs,
            seq_lens_cpu,
            return_comm_buf=True,
            comm_buf_idx=comm_buf_idx,
            num_comm_sms=num_comm_sms,
        )
