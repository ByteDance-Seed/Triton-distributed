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
import triton_dist.language as dl
import triton.language as tl
import nvshmem.core
from triton_dist.profiler_utils import perf_func
from triton_dist.utils import dist_print, initialize_distributed, nvshmem_free_tensor_sync, nvshmem_barrier_all_on_stream, nvshmem_create_tensor, requires_p2p_native_atomic, supports_p2p_native_atomic
from triton.language.extra.cuda.language_extra import atomic_cas, tid


@triton.jit
def kernel_ag_intra_node_nvlink_small_msg_split_msg(
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    symm_data_ptr,
    local_out_ptr,
    num_bytes,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)
    # mype = nvshmem_my_pe_wrapper()
    # npes = nvshmem_n_pes_wrapper()
    num_bytes_per_pid = tl.cdiv(num_bytes, num_pids)
    for peer in range(0, num_ranks):
        ptr_in = dl.symm_at(symm_data_ptr, peer)
        ptr_out = local_out_ptr + num_bytes * peer
        offs = tl.arange(0, BLOCK_SIZE)

        for i in range(0, tl.cdiv(num_bytes_per_pid, BLOCK_SIZE)):
            mask = offs < num_bytes - i * BLOCK_SIZE - pid * num_bytes_per_pid
            data = tl.load(ptr_in + pid * num_bytes_per_pid + i * BLOCK_SIZE + offs, mask=mask)
            tl.store(
                ptr_out + pid * num_bytes_per_pid + i * BLOCK_SIZE + offs,
                data,
                mask=mask,
            )


@triton.jit
def kernel_ag_intra_node_nvlink_small_msg_split_rank(
    rank,
    num_ranks: tl.constexpr,
    symm_data_ptr,
    local_out_ptr,
    comm_buf_ptr,
    num_bytes,
    BLOCK_SIZE: tl.constexpr,
    NEED_SYNC: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)

    if NEED_SYNC:
        thread_idx = tid(0)

        if thread_idx < num_ranks:
            remote_ptr = dl.symm_at(comm_buf_ptr + pid * num_ranks + rank, thread_idx)
            while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
                pass
            while (atomic_cas(comm_buf_ptr + pid * num_ranks + thread_idx, 1, 0, "sys", "acquire") != 1):
                pass

    # mype = nvshmem_my_pe_wrapper()
    # npes = nvshmem_n_pes_wrapper()
    num_ranks_per_pid = num_ranks // num_pids
    for local_peer in range(0, num_ranks_per_pid):
        peer = local_peer + pid * num_ranks_per_pid
        ptr_in = dl.symm_at(symm_data_ptr, peer)
        ptr_out = local_out_ptr + num_bytes * peer
        offs = tl.arange(0, BLOCK_SIZE)

        for i in range(0, tl.cdiv(num_bytes, BLOCK_SIZE)):
            mask = offs < num_bytes - i * BLOCK_SIZE
            data = tl.load(ptr_in + i * BLOCK_SIZE + offs, mask=mask)
            tl.store(ptr_out + i * BLOCK_SIZE + offs, data, mask=mask)


def nearest_power_two_u32(v):
    if v == 0:
        return 1
    v -= 1
    v = (v >> 1) | v
    v = (v >> 2) | v
    v = (v >> 4) | v
    v = (v >> 8) | v
    v = (v >> 16) | v
    v += 1
    return v


@requires_p2p_native_atomic
def ag_intra_node_nvlink_small_msg(symm_data, comm_buf, rank, num_ranks, need_block_level_sync=False):
    symm_data = symm_data.to(torch.int8)
    msg_size_bytes = symm_data.shape[0]
    local_out = torch.empty([num_ranks, msg_size_bytes], dtype=torch.int8, device="cuda")
    grid = (8, )
    kernel_ag_intra_node_nvlink_small_msg_split_rank[grid](
        rank,
        num_ranks,
        symm_data,
        local_out,
        comm_buf,
        msg_size_bytes,
        nearest_power_two_u32(msg_size_bytes),
        need_block_level_sync,
    )
    return local_out


if __name__ == "__main__":
    if not supports_p2p_native_atomic():
        print("p2p native atomic not supported, skip test")
        exit(0)

    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()

    current_stream = torch.cuda.current_stream()
    use_block_sync = True

    # test ag
    msg_size_bytes = 1024 * 8
    symm_data = nvshmem_create_tensor((msg_size_bytes, ), torch.int8)
    # at most world_size blocks, each block world_size barriers
    comm_buf = nvshmem_create_tensor((WORLD_SIZE * WORLD_SIZE, ), torch.int32)
    comm_buf.fill_(0)
    symm_data.fill_(RANK)
    nvshmem_barrier_all_on_stream(current_stream)
    torch.cuda.synchronize()

    def func():
        if not use_block_sync:
            nvshmem_barrier_all_on_stream(current_stream)
        results = ag_intra_node_nvlink_small_msg(symm_data, comm_buf, RANK, WORLD_SIZE,
                                                 need_block_level_sync=use_block_sync)
        return results

    results = func()
    print(results)

    _, perf = perf_func(func, iters=100, warmup_iters=10)
    dist_print(f"rank{RANK}", perf, need_sync=True, allowed_ranks=list(range(WORLD_SIZE)))

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            profile_memory=True,
    ) as profiler:
        for i in range(20):
            func()
    print(results)

    prof_dir = "prof/trace_ag_intra_node_nvlink_small_msg"
    os.makedirs(prof_dir, exist_ok=True)
    profiler.export_chrome_trace(f"{prof_dir}/rank{RANK}.json")

    nvshmem_free_tensor_sync(symm_data)
    nvshmem_free_tensor_sync(comm_buf)
    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
