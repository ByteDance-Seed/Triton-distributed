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
from triton_dist.utils import init_nvshmem_by_torch_process_group, init_seed, finalize_distributed, nvshmem_barrier_all_on_stream, perf_func, group_profile, assert_allclose

from triton_dist.layers.nvidia.p2p import CommOp


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_tokens", default=8192, type=int, help="max number of tokens")
    parser.add_argument("--hidden_size", default=6144, type=int, help="hidden dimension size")
    parser.add_argument("--num_pp_groups", default=2, type=int, help="number of pp groups")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iters", type=int, default=20, help="number of iterations for performance test")
    parser.add_argument("--warmup_iters", type=int, default=5, help="number of warmup iterations")
    parser.add_argument("--num_sms", default=8, type=int, help="number of SMs")
    parser.add_argument("--profile", default=False, action="store_true", help="dump torch.profiler.profile")

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


def put(ctx, ts, rank, dst_rank, num_sm):
    global comm_op
    global up_send_buffer_id
    global down_send_buffer_id
    assert rank != dst_rank
    if rank < dst_rank:
        buffer_id = up_send_buffer_id
        dst_buffer_id = buffer_id + recv_offset
    else:
        buffer_id = down_send_buffer_id + down_offset
        dst_buffer_id = buffer_id + recv_offset
    comm_op.wait_signal(dst_rank, dst_buffer_id, 0)
    comm_op.write(dst_rank, buffer_id, dst_buffer_id, ts, 1, sm=num_sm)
    if rank < dst_rank:
        up_send_buffer_id = (up_send_buffer_id + 1) % 2
    else:
        down_send_buffer_id = (down_send_buffer_id + 1) % 2


def recv(ctx, rank, src_rank, num_sm):
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
        comm_op.read(src_rank, src_buffer_id, buffer, sm=num_sm, fused=False)
        comm_op.set_signal(src_rank, src_buffer_id, 0)
        if rank > src_rank:
            up_recv_buffer_id = (up_recv_buffer_id + 1) % 2
        else:
            down_recv_buffer_id = (down_recv_buffer_id + 1) % 2
        event.record()
    return buffer, event


def get(ctx, rank, src_rank, num_sm):
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
        else:
            buffer_id = down_recv_buffer_id + down_offset + recv_offset
        comm_op.wait_signal(rank, buffer_id, 1, num_barriers=num_sm)
        buffer = comm_op.get_buffer(buffer_id)
        comm_op.set_signal(rank, buffer_id, 0, num_barriers=num_sm)
        if rank > src_rank:
            up_recv_buffer_id = (up_recv_buffer_id + 1) % 2
        else:
            down_recv_buffer_id = (down_recv_buffer_id + 1) % 2
        event.record()
        return buffer, event


# PyTorch native P2P implementation
class PyTorchP2P:

    def __init__(self, max_tokens, hidden_size, pp_rank, pp_size, pp_group, dtype=torch.bfloat16, num_buffers=8):
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.dtype = dtype
        self.num_buffers = num_buffers

        self.send_buffers = [
            torch.zeros([max_tokens, hidden_size], dtype=dtype, device="cuda") for _ in range(num_buffers)
        ]
        self.recv_buffers = [
            torch.zeros([max_tokens, hidden_size], dtype=dtype, device="cuda") for _ in range(num_buffers)
        ]
        self.events = [torch.cuda.Event() for _ in range(num_buffers)]
        self.up_send_idx = 0
        self.down_send_idx = 0
        self.up_recv_idx = 0
        self.down_recv_idx = 0

    def send(self, ctx, ts, rank, dst_rank):
        if rank < dst_rank:
            buffer_idx = self.up_send_idx
            self.up_send_idx = (self.up_send_idx + 1) % (self.num_buffers // 4)
        else:
            buffer_idx = self.down_send_idx + (self.num_buffers // 2)
            self.down_send_idx = (self.down_send_idx + 1) % (self.num_buffers // 4)

        self.send_buffers[buffer_idx].copy_(ts)
        dst_global_rank = torch.distributed.get_global_rank(self.pp_group, dst_rank)

        torch.distributed.send(self.send_buffers[buffer_idx], dst_global_rank)

    def recv(self, ctx, rank, src_rank):
        if rank > src_rank:
            buffer_idx = self.up_recv_idx + (self.num_buffers // 4)
            self.up_recv_idx = (self.up_recv_idx + 1) % (self.num_buffers // 4)
        else:
            buffer_idx = self.down_recv_idx + (3 * self.num_buffers // 4)
            self.down_recv_idx = (self.down_recv_idx + 1) % (self.num_buffers // 4)

        src_global_rank = torch.distributed.get_global_rank(self.pp_group, src_rank)
        event = self.events[buffer_idx]
        torch.distributed.recv(self.recv_buffers[buffer_idx], src_global_rank)
        event.record()

        return self.recv_buffers[buffer_idx], event


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

    num_sms = args.num_sms

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

    pytorch_p2p = PyTorchP2P(
        args.max_tokens,
        args.hidden_size,
        pp_group.rank(),
        pp_group.size(),
        pp_group,
        dtype=dtype,
        num_buffers=buffer_size,
    )

    tensor = torch.randn([args.max_tokens, args.hidden_size], dtype=dtype, device="cuda")
    answers = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    torch.distributed.all_gather(answers, tensor, group=WORLD_GROUP)
    ctx = None

    assert pp_size > 1
    send_recv_table = [[i, random.randint(1, pp_size - 1)] for i in range(pp_size)]
    send_recv_table_tensor = torch.tensor(send_recv_table, device="cuda", dtype=torch.int32)
    torch.distributed.broadcast(send_recv_table_tensor, src=0)
    send_recv_table = send_recv_table_tensor.cpu().tolist()
    # make src->dst items
    goldens = []
    for item in send_recv_table:
        item[1] = (item[0] + item[1]) % pp_size
        src_global_rank = torch.distributed.get_global_rank(pp_group, item[0])
        if item[1] == pp_group.rank():
            goldens.append(answers[src_global_rank])

    if RANK == 0:
        print("send_recv_table:")
        print(send_recv_table)

    # Test Triton implementation
    outputs = []
    events = []

    for src, dst in send_recv_table:
        if pp_group.rank() == src:
            send(ctx, tensor, src, dst)
        if pp_group.rank() == dst:
            out, event = recv(ctx, dst, src, num_sm=num_sms)
            outputs.append(out)
            events.append(event)

    for output, event, golden in zip(outputs, events, goldens):
        event.synchronize()
        assert torch.allclose(output, golden)

    print(f"rank: {RANK}, Triton implementation ✅ pass!")

    # # Test Triton put implementation
    outputs = []
    events = []

    for src, dst in send_recv_table:
        if pp_group.rank() == src:
            put(ctx, tensor, src, dst, num_sm=num_sms)
        if pp_group.rank() == dst:
            out, event = get(ctx, dst, src, num_sm=num_sms)
            outputs.append(out)
            events.append(event)

    for i in range(WORLD_SIZE):
        if i == RANK:
            for output, event, golden in zip(outputs, events, goldens):
                event.synchronize()
                assert_allclose(output, golden, atol=1e-5, rtol=1e-5)

            print(f"rank: {RANK}, Triton implementation ✅ pass!")
        torch.distributed.barrier(WORLD_GROUP)

    # Test PyTorch implementation
    outputs = []
    events = []

    for src, dst in send_recv_table:
        if pp_group.rank() == src:
            pytorch_p2p.send(ctx, tensor, src, dst)
        if pp_group.rank() == dst:
            out, event = pytorch_p2p.recv(ctx, dst, src)
            outputs.append(out)
            events.append(event)

    for output, event, golden in zip(outputs, events, goldens):
        event.synchronize()
        assert torch.allclose(output, golden)

    print(f"rank: {RANK}, PyTorch native implementation ✅ pass!")

    # Only test one transaction
    def test_triton_send_recv():
        events = []
        for src, dst in send_recv_table[:1]:
            if pp_group.rank() == src:
                send(ctx, tensor, src, dst)
            if pp_group.rank() == dst:
                out, event = recv(ctx, dst, src, num_sm=num_sms)
                events.append(event)
        for event in events:
            event.synchronize()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def test_triton_put_get():
        events = []
        for src, dst in send_recv_table[:1]:
            if pp_group.rank() == src:
                put(ctx, tensor, src, dst, num_sm=num_sms)
            if pp_group.rank() == dst:
                out, event = get(ctx, dst, src, num_sm=num_sms)
                events.append(event)
        for event in events:
            event.synchronize()
        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def test_pytorch_send_recv():
        events = []
        for src, dst in send_recv_table[:1]:
            if pp_group.rank() == src:
                pytorch_p2p.send(ctx, tensor, src, dst)
            if pp_group.rank() == dst:
                out, event = pytorch_p2p.recv(ctx, dst, src)
        for event in events:
            event.synchronize()
        torch.distributed.barrier(WORLD_GROUP)

    torch.distributed.barrier(WORLD_GROUP)

    # Test Triton implementation
    triton_duration_ms = None
    pytorch_duration_ms = None

    with group_profile("pp_triton", args.profile, group=WORLD_GROUP):
        torch.cuda.synchronize()
        _, triton_duration_ms = perf_func(test_triton_send_recv, args.iters, args.warmup_iters)
        torch.cuda.synchronize()

    with group_profile("pp_put_get_triton", args.profile, group=WORLD_GROUP):
        torch.cuda.synchronize()
        _, triton_duration_put_get_ms = perf_func(test_triton_put_get, args.iters, args.warmup_iters)
        torch.cuda.synchronize()

    # Test PyTorch implementation
    with group_profile("pp_pytorch", args.profile, group=WORLD_GROUP):
        torch.cuda.synchronize()
        _, pytorch_duration_ms = perf_func(test_pytorch_send_recv, args.iters, args.warmup_iters)
        torch.cuda.synchronize()

    tensor_size_bytes = float(tensor.element_size() * tensor.nelement())
    triton_bandwidth_gbps = (tensor_size_bytes / 1e9) / (triton_duration_ms / 1000)
    triton_put_get_bandwidth_gbps = (tensor_size_bytes / 1e9) / (triton_duration_put_get_ms / 1000)
    pytorch_bandwidth_gbps = (tensor_size_bytes / 1e9) / (pytorch_duration_ms / 1000)
    speedup = pytorch_duration_ms / triton_duration_ms if triton_duration_ms > 0 else float('inf')
    speedup_put_get = pytorch_duration_ms / triton_duration_put_get_ms if triton_duration_put_get_ms > 0 else float(
        'inf')

    for i in range(WORLD_SIZE):
        if i == RANK:
            print(
                f"Triton: {triton_duration_ms:.3f} ms, Torch: {pytorch_duration_ms:.3f} ms, Speedup: {speedup:.2f}x, Effective Bandwidth: Triton: {triton_bandwidth_gbps:.2f} GB/s, PyTorch native: {pytorch_bandwidth_gbps:.2f} GB/s"
            )
            print(
                f"Triton put/get: {triton_duration_put_get_ms:.3f} ms, Speedup: {speedup_put_get:.2f}x, Effective Bandwidth: Triton put/get: {triton_put_get_bandwidth_gbps:.2f} GB/s"
            )
        torch.distributed.barrier(WORLD_GROUP)

    del comm_op
    finalize_distributed()
