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
import triton
import triton.language as tl
import torch
import pynvshmem
import os
import datetime
from triton.language.extra import libshmem_device


@triton.jit
def ring_put(ptr):
    # ptr_out = nvshmem_ptr(ptr, 1)
    mype = libshmem_device.my_pe()
    npes = libshmem_device.n_pes()
    peer = (mype + 1) % npes
    libshmem_device.int_p(ptr, mype, peer)


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
        ptr_in = libshmem_device.remote_ptr(symm_data_ptr, peer.to(tl.int32)).to(tl.pointer_type(tl.int8))
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
def ld_signal(barrier_ptr, semantic: tl.constexpr = "acquire", scope: tl.constexpr = "gpu"):
    return tl.inline_asm_elementwise(
        asm=f"""{{
        ld.global.{semantic}.{scope}.b32 $0, [$1];
        }}
        """,
        constraints=("=r,l"),
        args=[barrier_ptr],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@tl.core.extern
def atomic_cas(
    ptr,
    value,
    target_value,
    scope: tl.constexpr,
    semantic: tl.constexpr,
    _builder=None,
):
    return tl.inline_asm_elementwise(
        asm=f"atom.{semantic.value}.{scope.value}.global.cas.b32 $0, [$1], $2, $3;",
        constraints=("=r,l,r,r"),
        args=[
            ptr,
            value,
            target_value,
        ],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
        _builder=_builder,
    )


@tl.core.extern
def thread_id(axis: tl.constexpr, _builder=None):
    return tl.inline_asm_elementwise(
        asm=f"mov.u32 $0, %tid.{axis.value};",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
        _builder=_builder,
    )


@triton.jit
def kernel_ag_intra_node_nvlink_small_msg_split_rank(
    rank: tl.constexpr,
    num_ranks: tl.constexpr,
    symm_data_ptr,
    local_out_ptr,
    comm_buf_ptr,
    num_bytes,
    BLOCK_SIZE: tl.constexpr,
    need_sync: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pids = tl.num_programs(axis=0)

    if need_sync:
        tid = thread_id(axis="x")

        if tid < num_ranks:
            remote_ptr = libshmem_device.remote_ptr(comm_buf_ptr + pid * num_ranks + rank, tid.to(tl.int32))
            while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
                pass
            while (atomic_cas(comm_buf_ptr + pid * num_ranks + tid, 1, 0, "sys", "acquire") != 1):
                pass

    # mype = nvshmem_my_pe_wrapper()
    # npes = nvshmem_n_pes_wrapper()
    num_ranks_per_pid = num_ranks // num_pids
    for local_peer in range(0, num_ranks_per_pid):
        peer = local_peer + pid * num_ranks_per_pid
        ptr_in = libshmem_device.remote_ptr(symm_data_ptr, peer.to(tl.int32)).to(tl.pointer_type(tl.int8))
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


def perf_func(func, iters, warmup_iters):
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    for n in range(iters + warmup_iters):
        if n == warmup_iters:
            start_event.record()
        output = func()
    stop_event.record()
    start_event.wait()
    stop_event.wait()
    torch.cuda.current_stream().synchronize()
    duration_ms = start_event.elapsed_time(stop_event)
    return output, duration_ms / iters


def dist_print(*args, **kwargs):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    prefix = False
    if "allowed_ranks" in kwargs:
        allowed_ranks = kwargs["allowed_ranks"]
        if isinstance(allowed_ranks, str) and allowed_ranks == "all":
            allowed_ranks = list(range(world_size))

        del kwargs["allowed_ranks"]
    else:
        allowed_ranks = [0]
    if "prefix" in kwargs:
        prefix = kwargs["prefix"]

        del kwargs["prefix"]

    need_sync = False
    if "need_sync" in kwargs:
        need_sync = kwargs["need_sync"]

        del kwargs["need_sync"]

    for allowed in allowed_ranks:
        if need_sync:
            torch.distributed.barrier()
        if rank == allowed:
            if prefix:
                print(f"[rank:{rank}]", end="")
            print(*args, **kwargs)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier()

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()

    t = pynvshmem.nvshmem_create_tensor((32, ), torch.int32)
    # pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    ring_put[(1, )](t)
    print("Ring put test good, nvshmem works.")
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    use_block_sync = True

    # test ag
    msg_size_bytes = 1024 * 8
    symm_data = pynvshmem.nvshmem_create_tensor([msg_size_bytes], torch.int8)
    # at most world_size blocks, each block world_size barriers
    comm_buf = pynvshmem.nvshmem_create_tensor([WORLD_SIZE * WORLD_SIZE], torch.int32)
    comm_buf.fill_(0)
    symm_data.fill_(RANK)
    pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
    torch.cuda.synchronize()

    def func():
        if not use_block_sync:
            pynvshmem.nvshmem_barrier_all_on_stream(current_stream.cuda_stream)
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

    torch.distributed.destroy_process_group()
