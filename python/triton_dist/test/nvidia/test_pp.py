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
import argparse
import torch
import random
import datetime
import torch.distributed
from triton_dist.utils import init_nvshmem_by_torch_process_group, init_seed, finalize_distributed, nvshmem_barrier_all_on_stream

from triton_dist.layers.nvidia.p2p import CommOp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", default=8192, type=int, help="max number of tokens")
    parser.add_argument("--hidden_size", default=6144, type=int, help="hidden dimension size")
    parser.add_argument("--num_pp_groups", default=2, type=int, help="number of pp groups")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def split_torch_process_group(pg: torch.distributed.ProcessGroup, num_groups: int) -> torch.distributed.ProcessGroup:
    size = pg.size()
    rank = pg.rank()
    group_size = size // num_groups
    group_id = rank // group_size
    if size % group_size != 0:
        raise ValueError(f"Process group size {size} is not divisible by group size {group_size}.")
    # Create list of ranks per group
    for n in range(num_groups):
        subgroup_ranks = [i + n * group_size for i in range(group_size)]
        # Create new NCCL group
        print("subgroup_ranks", subgroup_ranks)
        subgroup_ = torch.distributed.new_group(ranks=subgroup_ranks, backend="nccl")
        if n == group_id:
            subgroup = subgroup_
    return subgroup


comm_op = None
pp_stream = None

buffer_size = 8
down_offset = 4
recv_offset = 2
up_send_buffer_id = 0
down_send_buffer_id = 0
up_recv_buffer_id = 0
down_recv_buffer_id = 0


def send(ctx, ts, rank, dst_rank):
    global comm_op
    global up_send_buffer_id
    global down_send_buffer_id
    assert rank != dst_rank
    if rank < dst_rank:
        buffer_id = up_send_buffer_id
    else:
        buffer_id = down_send_buffer_id + down_offset
    comm_op.wait_signal(rank, buffer_id, 0)
    buffer = comm_op.get_buffer(buffer_id)
    buffer.copy_(ts)
    comm_op.set_signal(rank, buffer_id, 1)
    if rank < dst_rank:
        up_send_buffer_id = (up_send_buffer_id + 1) % 2
    else:
        down_send_buffer_id = (down_send_buffer_id + 1) % 2


def recv(ctx, rank, src_rank):
    global comm_op
    global pp_stream
    global up_recv_buffer_id
    global down_recv_buffer_id
    event = torch.cuda.Event()
    assert rank != src_rank
    with torch.cuda.stream(pp_stream):
        pp_stream.wait_stream(torch.cuda.default_stream())
        if rank > src_rank:
            buffer_id = up_recv_buffer_id + recv_offset
            src_buffer_id = up_recv_buffer_id
        else:
            buffer_id = down_recv_buffer_id + down_offset + recv_offset
            src_buffer_id = down_recv_buffer_id + down_offset
        comm_op.wait_signal(src_rank, src_buffer_id, 1)
        buffer = comm_op.get_buffer(buffer_id)
        comm_op.read(src_rank, src_buffer_id, buffer, sm=4)
        comm_op.set_signal(src_rank, src_buffer_id, 0)
        if rank > src_rank:
            up_recv_buffer_id = (up_recv_buffer_id + 1) % 2
        else:
            down_recv_buffer_id = (down_recv_buffer_id + 1) % 2
        event.record()
    return buffer, event


if __name__ == "__main__":
    args = parse_args()
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
    # use all ranks as tp group
    WORLD_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(WORLD_GROUP)

    seed = args.seed
    init_seed(seed=seed if seed is not None else RANK)

    init_nvshmem_by_torch_process_group(WORLD_GROUP)

    ep_size = args.num_pp_groups
    pp_size = WORLD_SIZE // ep_size

    pp_group = split_torch_process_group(WORLD_GROUP, ep_size)
    dtype = torch.bfloat16

    comm_op = CommOp(
        args.max_tokens,
        args.hidden_size,
        pp_group.rank(),
        pp_group.size(),
        pp_group,
        dtype=dtype,
        num_buffers=buffer_size,
    )

    pp_stream = torch.cuda.Stream()

    tensor = torch.randn([args.max_tokens, args.hidden_size], dtype=dtype, device="cuda")
    answers = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(answers, tensor, group=WORLD_GROUP)
    ctx = None

    num_tests = 2
    assert pp_size > 1
    assert num_tests <= 2
    send_recv_table = [[random.randint(0, pp_size - 1), random.randint(1, pp_size - 1)] for _ in range(num_tests)]
    # make src->dst items
    goldens = []
    for item in send_recv_table:
        item[1] = (item[0] + item[1]) % pp_size
        src_global_rank = torch.distributed.get_global_rank(pp_group, item[0])
        goldens.append(answers[src_global_rank])

    outputs = []
    events = []

    for src, dst in send_recv_table:
        if pp_group.rank() == src:
            send(ctx, tensor, src, dst)
        if pp_group.rank() == dst:
            out, event = recv(ctx, dst, src)
            outputs.append(out)
            events.append(event)

    # check results
    for output, event, golden in zip(outputs, events, goldens):
        event.synchronize()

        print("rank:", pp_group.rank())
        print("out:")
        print(out)
        print("golden:")
        print(golden)
        assert torch.allclose(out, golden)

    print("rank:", RANK, "✅ pass!")

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    del comm_op
    finalize_distributed()
