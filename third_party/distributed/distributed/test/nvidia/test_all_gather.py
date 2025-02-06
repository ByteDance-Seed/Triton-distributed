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
import triton.distributed.language as dl
import pynvshmem

import os
import datetime


# all gather(pull mode)
@triton.jit
def all_gather_kernel(
    ag_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    rank = dl.rank()
    world_size = dl.num_ranks()
    n_elements_per_rank = n_elements // world_size
    num_tiles_per_rank = tl.cdiv(n_elements_per_rank, BLOCK_SIZE)
    num_blocks = tl.num_programs(axis=0)
    start_id = tl.program_id(axis=0)
    # pull mode
    for i in range(1, world_size):
        src_rank = (rank + i) % world_size
        remote_ptr = dl.symm_at(ag_ptr, src_rank)
        rank_offset = src_rank * n_elements_per_rank
        for pid in range(start_id, num_tiles_per_rank, num_blocks):
            block_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = block_offsets < n_elements_per_rank
            val = tl.load(remote_ptr + block_offsets + rank_offset, mask=mask)
            tl.store(ag_ptr + block_offsets + rank_offset, val, mask=mask)


def triton_all_gather(ag_buffer):
    n_elements = ag_buffer.numel()
    BLOCK_SIZE = 4096
    all_gather_kernel[(32, )](ag_buffer, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_warps=16)
    return ag_buffer


if __name__ == "__main__":
    nelems_per_rank = 64 * 8192
    dtype = torch.int32
    warmup_iters = 30
    iters = 500

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

    torch.cuda.synchronize()
    pynvshmem.init_nvshmem_by_uniqueid(TP_GROUP)

    n_elements = nelems_per_rank * WORLD_SIZE
    ref_tensor = torch.arange(n_elements, dtype=dtype).cuda()

    ag_buffer = pynvshmem.nvshmem_create_tensor((n_elements, ), dtype)
    # local copy
    ag_buffer[nelems_per_rank * RANK:nelems_per_rank * (RANK + 1)].copy_(ref_tensor[nelems_per_rank *
                                                                                    RANK:nelems_per_rank * (RANK + 1)])

    pynvshmem.nvshmem_barrier_all()
    triton_all_gather(ag_buffer)
    pynvshmem.nvshmem_barrier_all()

    try:
        torch.testing.assert_close(ag_buffer, ref_tensor, atol=0, rtol=0)
    except Exception as e:
        print(f"❌ RANK[{RANK}] check failed")
        raise e
    else:
        print(f"✅ RANK[{RANK}] check passed")

    total_iters = warmup_iters + iters
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(total_iters)]

    pynvshmem.nvshmem_barrier_all()
    for i in range(total_iters):
        start_events[i].record()
        triton_all_gather(ag_buffer)
        end_events[i].record()
    torch.cuda.current_stream().synchronize()

    ag_times = []
    for i in range(total_iters):
        end_events[i].synchronize()
        if i >= warmup_iters:
            ag_times.append(start_events[i].elapsed_time(end_events[i]) / 1000)
    ag_time_ms = sum(ag_times) / iters * 1000

    gbps = lambda ms: ag_buffer.numel() * ag_buffer.element_size() * 1e-9 / (ms * 1e-3) * (WORLD_SIZE - 1) / WORLD_SIZE
    print(f"RANK = {RANK}, Bandwith = {gbps(ag_time_ms)} GB/S")
    torch.distributed.destroy_process_group()
