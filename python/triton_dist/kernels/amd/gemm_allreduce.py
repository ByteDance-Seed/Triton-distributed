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
import dataclasses
from typing import List
import triton
import triton.language as tl
import triton_dist.language as dl
import pyrocshmem
from triton_dist.language.extra import libshmem_device
from triton.language.extra.hip.libdevice import store_release_system
import triton_dist.tune
from triton.runtime.driver import driver


@dataclasses.dataclass
class GemmARContext:
    rank: int
    num_ranks: int
    comm_bufs: List[torch.Tensor]
    comm_buf_ptr: torch.Tensor
    symm_gemm_out_buf: torch.Tensor
    tile_completed_buf: torch.Tensor
    ar_stream: torch.cuda.Stream

    def get_gemm_out_buf(self, A: torch.Tensor, weight):
        M, N = A.shape[0], weight.shape[0]
        assert self.symm_gemm_out_buf.numel() >= M * N
        return self.symm_gemm_out_buf.reshape(-1)[:M * N].reshape(M, N)


def create_gemm_ar_context(ar_stream: torch.cuda.Stream, rank, world_size, max_M, N, dtype, MIN_BLOCK_SIZE_M=64,
                           MIN_BLOCK_SIZE_N=64):
    comm_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([world_size], torch.int32)
    comm_buf_ptr = torch.tensor([t.data_ptr() for t in comm_bufs], device=torch.cuda.current_device(),
                                requires_grad=False)
    comm_bufs[rank].zero_()
    gemm_out_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([max_M, N], dtype)
    num_tiles = triton.cdiv(max_M, MIN_BLOCK_SIZE_M) * triton.cdiv(N, MIN_BLOCK_SIZE_N)
    tile_signal_bufs = pyrocshmem.rocshmem_create_tensor_list_intra_node([num_tiles * world_size], torch.int32)
    tile_signal_bufs[rank].zero_()
    gemm_out_buf = gemm_out_bufs[rank]
    tile_completed_buf = tile_signal_bufs[rank]
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return GemmARContext(rank=rank, num_ranks=world_size, comm_bufs=comm_bufs, comm_buf_ptr=comm_buf_ptr,
                         symm_gemm_out_buf=gemm_out_buf, tile_completed_buf=tile_completed_buf, ar_stream=ar_stream)


@triton.jit(do_not_specialize=["rank"])
def reset_signal_and_barrier_all_kernel(
    rank,
    num_ranks,
    comm_buf_base_ptrs,
    tile_signal_ptr,
    num_tiles,
):
    sm_id = tl.program_id(axis=0)
    num_sms = tl.num_programs(axis=0)

    for i in range(sm_id, num_tiles, num_sms):
        tl.store(tile_signal_ptr + i, 0)

    if sm_id == 0:
        for i in range(num_ranks):
            remote_base_ptr = tl.load(comm_buf_base_ptrs + i).to(tl.pointer_type(tl.int32))
            while tl.atomic_cas(remote_base_ptr + rank, 0, 1, scope="sys", sem="release") != 0:
                pass

        for i in range(num_ranks):
            local_base_ptr = tl.load(comm_buf_base_ptrs + rank).to(tl.pointer_type(tl.int32))
            while tl.atomic_cas(local_base_ptr + i, 1, 0, scope="sys", sem="acquire") != 1:
                pass

        tl.debug_barrier()


@triton.jit
def _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M):
    group_id = tile_id // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((tile_id % num_pid_in_group) % group_size_m)
    pid_n = (tile_id % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def kernel_persistent_gemm_notify_ar(
    a_ptr,
    b_ptr,
    c_ptr,  # Input/Output pointers
    tile_signal_ptr,  # Tile completion signals
    ctx,  # rocshmem context
    M,
    N,
    K,  # Matrix dimensions
    stride_am,
    stride_ak,  # A strides
    stride_bn,
    stride_bk,  # B strides
    stride_cm,
    stride_cn,  # C strides
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_GEMM_SMS: tl.constexpr,
):
    libshmem_device.set_rocshmem_ctx(ctx)
    rank = dl.rank()
    world_size = dl.num_ranks()

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)

    start_pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_tiles = num_pid_m * num_pid_n
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    for tile_id in tl.range(start_pid, num_tiles, NUM_GEMM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)
        signal_offset = tile_id * world_size + rank
        for remote in range(world_size):
            remote_signal_ptr = dl.symm_at(tile_signal_ptr, remote)
            store_release_system(remote_signal_ptr + signal_offset, 1)


@triton.jit
def consumer_all_reduce_kernel(symm_buf_ptr, tile_signal_ptr, ctx,  # rocshmem context
                               M, N, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                               GROUP_SIZE_M: tl.constexpr, NUM_COMM_SMS: tl.constexpr):
    libshmem_device.set_rocshmem_ctx(ctx)
    rank = dl.rank()
    world_size = dl.num_ranks()
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * tl.cdiv(N, BLOCK_SIZE_N)
    for tile_id in range(pid, num_tiles, NUM_COMM_SMS):
        pid_m, pid_n = _compute_pid(tile_id, num_pid_in_group, num_pid_m, GROUP_SIZE_M)
        owner_rank = tile_id % world_size
        if rank == owner_rank:
            signal_base = tile_id * world_size
            offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

            final_acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
            for i in range(world_size):
                target_rank = (i + rank) % world_size
                remote_c_ptr = dl.symm_at(symm_buf_ptr, target_rank)
                token = dl.wait(tile_signal_ptr + signal_base + target_rank, 1, "sys", "acquire", waitValue=1)
                remote_c_ptr = dl.consume_token(remote_c_ptr, token)
                remote_c_ptrs = remote_c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
                remote_data = tl.load(remote_c_ptrs, mask=c_mask, other=0.0)
                final_acc += remote_data

            c = final_acc.to(symm_buf_ptr.dtype.element_ty)
            for remote_rank in range(world_size):
                remote_buf_ptr = dl.symm_at(symm_buf_ptr, remote_rank)
                remote_buf_ptrs = remote_buf_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
                tl.store(remote_buf_ptrs, c, mask=c_mask)


DEFAULT_GEMM_CONFIG = triton.Config(
    kwargs={"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4, "waves_per_eu": 2},
    num_warps=8, num_stages=2)


def get_hip_autotune_config():
    return [
        DEFAULT_GEMM_CONFIG,
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'waves_per_eu': 0,
                'matrix_instr_nonkdim': 16, 'kpack': 2
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=8, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,
                'matrix_instr_nonkdim': 16, 'kpack': 1
            }, num_warps=4, num_stages=2),
        ##
    ]


def key_fn(ctx: GemmARContext, A: torch.Tensor, B: torch.Tensor, *args, **kwargs):
    return triton_dist.tune.to_hashable(A), triton_dist.tune.to_hashable(B), ctx.num_ranks


def prune_fn_by_shared_memory(config, ctx: GemmARContext, A: torch.Tensor, *args, **kwargs):
    itemsize = A.itemsize
    gemm_config: triton.Config = config["gemm_config"]
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    BLOCK_SIZE_K = gemm_config.kwargs["BLOCK_SIZE_K"]
    num_stages = max(0, gemm_config.num_stages - 1)
    shared_mem_size = num_stages * (BLOCK_SIZE_M * BLOCK_SIZE_K * itemsize + BLOCK_SIZE_N * BLOCK_SIZE_K * itemsize)
    device = torch.cuda.current_device()
    if shared_mem_size > driver.active.utils.get_device_properties(device)["max_shared_mem"]:
        return False
    return True


@triton_dist.tune.autotune(
    config_space=[{"gemm_config": c} for c in get_hip_autotune_config()],
    key_fn=key_fn,
    prune_fn=prune_fn_by_shared_memory,
)
def gemm_allreduce_op(ctx: GemmARContext, A: torch.Tensor, B: torch.Tensor, gemm_config: triton.Config):
    NUM_COMM_SMS = 32
    assert A.shape[1] == B.shape[1], "Incompatible dimensions"
    assert A.dtype == B.dtype, "Incompatible dtypes"

    symm_c = ctx.get_gemm_out_buf(A, B)
    tile_signal = ctx.tile_completed_buf

    current_stream = torch.cuda.current_stream()
    ar_stream = ctx.ar_stream
    ar_stream.wait_stream(current_stream)
    rocshmem_ctx = pyrocshmem.rocshmem_get_device_ctx()
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    M, K = A.shape
    N, K = B.shape
    NUM_GEMM_SMS = num_sms - NUM_COMM_SMS
    assert NUM_GEMM_SMS > 0, "Not enough SMs to run GEMM kernel"
    BLOCK_SIZE_M = gemm_config.kwargs["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = gemm_config.kwargs["BLOCK_SIZE_N"]
    GROUP_SIZE_M = gemm_config.kwargs["GROUP_SIZE_M"]
    kernel_persistent_gemm_notify_ar[(NUM_GEMM_SMS, )](A, B, symm_c, tile_signal, rocshmem_ctx, M, N, K, A.stride(0),
                                                       A.stride(1), B.stride(0), B.stride(1), symm_c.stride(0),
                                                       symm_c.stride(1), NUM_GEMM_SMS=NUM_GEMM_SMS,
                                                       **gemm_config.all_kwargs())
    with torch.cuda.stream(ar_stream):
        consumer_all_reduce(symm_c, tile_signal, rocshmem_ctx, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N,
                            GROUP_SIZE_M=GROUP_SIZE_M, NUM_COMM_SMS=NUM_COMM_SMS)
    current_stream.wait_stream(ar_stream)
    reset_signal_and_barrier_all_kernel[(num_sms, )](ctx.rank, ctx.num_ranks, ctx.comm_buf_ptr, tile_signal,
                                                     tile_signal.shape[0], num_warps=16)
    return symm_c


def consumer_all_reduce(symm_buf, tile_signal, rocshmem_ctx, BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, GROUP_SIZE_M=1,
                        NUM_COMM_SMS=16):
    M, N = symm_buf.shape
    consumer_all_reduce_kernel[(NUM_COMM_SMS, )](symm_buf, tile_signal, rocshmem_ctx, M, N, symm_buf.stride(0),
                                                 symm_buf.stride(1), BLOCK_SIZE_M=BLOCK_SIZE_M,
                                                 BLOCK_SIZE_N=BLOCK_SIZE_N, GROUP_SIZE_M=GROUP_SIZE_M,
                                                 NUM_COMM_SMS=NUM_COMM_SMS, num_warps=16)
