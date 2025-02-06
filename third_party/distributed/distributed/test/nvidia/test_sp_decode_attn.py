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
from typing import List, Optional

import argparse
import os
import sys
from cuda import cuda, cudart
import datetime
import numpy as np
import pynvshmem

from triton.distributed.layers.nvidia import SpGQAFlashDecodeAttention
from triton.distributed.utils import perf_func, group_profile, dist_print


def broadcast_cpu(tensor: torch.Tensor, src: int, group: torch.distributed.ProcessGroup):
    if not tensor.is_cuda:
        tensor_gpu = tensor.cuda()
        torch.distributed.broadcast(tensor_gpu, src=src, group=group)
        tensor.copy_(tensor_gpu)
    else:
        torch.distributed.broadcast(tensor, src=src, group=group)
    torch.cuda.synchronize()


def init_nvshmem_by_uniqueid(group: torch.distributed.ProcessGroup):
    rank, nranks = group.rank(), group.size()
    if rank == 0:
        unique_id: bytes = pynvshmem.nvshmemx_get_uniqueid()
        unique_id = torch.frombuffer(unique_id, dtype=torch.uint8).clone()
    else:
        unique_id = torch.empty(128, dtype=torch.uint8)

    broadcast_cpu(tensor=unique_id, group=group, src=0)

    unique_id = unique_id.numpy().tobytes()
    pynvshmem.nvshmemx_init_attr_with_uniqueid(rank, nranks, unique_id)


ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--shape_id", type=str, default="")
    parser.add_argument("--profile", action="store_true")

    args = parser.parse_args()
    return args


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


def CUDA_CHECK(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cuda Error: {err}: {cuda.cuGetErrorName(err)}")
    elif isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cuda Error: {err}: {cudart.cudaGetErrorString(err)}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens_per_rank: List[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: List[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens_per_rank[i]
        q = query[start_idx:start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = torch.triu(empty_mask,
                                             diagonal=kv_len - (query_len + sliding_window) + 1).bool().logical_not()
            mask |= sliding_window_mask
        if soft_cap > 0.0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


@register_test("correctness")
def test_triton_decode_with_paged_kv(args) -> None:
    kv_lens_per_rank = [128 * 12]
    num_heads = 96
    head_size = 128
    block_size = 1
    dtype = torch.float16
    soft_cap = 0

    torch.set_default_device("cuda")

    num_seqs = len(kv_lens_per_rank)
    num_query_heads = num_heads
    num_kv_heads = num_query_heads // 8
    assert num_query_heads % num_kv_heads == 0
    scale = head_size**-0.5

    NUM_BLOCKS_PER_RANK = 128 * 12 + 1
    NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks  # Large enough to test overflow in index calculation.

    ths_op = SpGQAFlashDecodeAttention(args.rank, args.rank // args.local_num_ranks, args.num_ranks,
                                       args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads, head_size,
                                       head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
                                       max_allowed_batch=1, thrink_buffer_threshold=500, stages=20)
    for _ in range(200):
        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype) / 10
        args.default_group.broadcast(query, root=0)

        key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype) / 10
        args.default_group.broadcast(key_value_cache, root=0)
        key_cache = key_value_cache[:, 0, :, :, :].contiguous()
        value_cache = key_value_cache[:, 1, :, :, :].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                        NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                            NUM_BLOCKS_PER_RANK].contiguous()

        max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
        block_tables_list = [
            torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, max_num_blocks_per_seq_per_rank), dtype=torch.int32)
            for i in range(args.num_ranks)
        ]
        block_tables_list_shift = [
            torch.zeros((num_seqs, max_num_blocks_per_seq_per_rank)).to(torch.int32) + i * NUM_BLOCKS_PER_RANK
            for i in range(args.num_ranks)
        ]
        block_tables_shift = torch.cat(block_tables_list_shift, dim=-1)
        block_tables_this_rank = block_tables_list[args.rank]
        torch.distributed.all_gather(block_tables_list, block_tables_this_rank, group=args.default_group)
        block_tables = torch.cat(block_tables_list, dim=-1) + block_tables_shift

        global_kv_lens = [i * args.num_ranks for i in kv_lens_per_rank]
        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        query = torch.randn_like(query)
        args.default_group.broadcast(query, root=0)
        output = ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor,
                        block_tables_this_rank)
        new_query = torch.empty_like(query).copy_(query)

        ref_output = ref_paged_attn(query=new_query, key_cache=key_cache, value_cache=value_cache,
                                    query_lens=[1] * num_seqs, kv_lens_per_rank=global_kv_lens,
                                    block_tables=block_tables, scale=scale, soft_cap=soft_cap)

        torch.testing.assert_close(output, ref_output, atol=0.05, rtol=1e-2), \
            f"{torch.max(torch.abs(output - ref_output))}"
    dist_print("Pass!", allowed_ranks=[0])


@register_test("perf")
def perf_decode(args):
    for kv_len_per_rank in [2**i for i in range(10, 18)]:
        kv_lens_per_rank = [kv_len_per_rank]
        num_heads = 96
        head_size = 128
        block_size = 1
        dtype = torch.float16
        soft_cap = 0

        torch.set_default_device("cuda")

        num_seqs = len(kv_lens_per_rank)
        num_query_heads = num_heads
        num_kv_heads = num_query_heads // 8
        assert num_query_heads % num_kv_heads == 0
        scale = head_size**-0.5

        NUM_BLOCKS_PER_RANK = kv_lens_per_rank[0] + 1
        NUM_BLOCKS = NUM_BLOCKS_PER_RANK * args.num_ranks  # Large enough to test overflow in index calculation.

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)
        args.default_group.broadcast(query, root=0)

        key_value_cache = torch.randn(NUM_BLOCKS, 2, block_size, num_kv_heads, head_size, dtype=dtype)
        args.default_group.broadcast(key_value_cache, root=0)
        key_cache = key_value_cache[:, 0, :, :, :].contiguous()
        value_cache = key_value_cache[:, 1, :, :, :].contiguous()
        key_cache_this_rank = key_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                        NUM_BLOCKS_PER_RANK].contiguous()
        value_cache_this_rank = value_cache[args.rank * NUM_BLOCKS_PER_RANK:(args.rank + 1) *
                                            NUM_BLOCKS_PER_RANK].contiguous()

        max_num_blocks_per_seq_per_rank = NUM_BLOCKS_PER_RANK
        block_tables_list = [
            torch.randint(0, NUM_BLOCKS_PER_RANK, (num_seqs, max_num_blocks_per_seq_per_rank), dtype=torch.int32)
            for i in range(args.num_ranks)
        ]
        block_tables_this_rank = block_tables_list[args.rank]

        kv_lens_tensor = torch.tensor(kv_lens_per_rank, dtype=torch.int32, device=query.device)
        global_kv_lens_tensor = torch.cat([kv_lens_tensor.view(1, -1) for _ in range(args.num_ranks)], dim=0)

        ths_op = SpGQAFlashDecodeAttention(args.rank, args.rank // args.local_num_ranks, args.num_ranks,
                                           args.num_ranks // args.local_num_ranks, num_query_heads, num_kv_heads,
                                           head_size, head_size, page_size=block_size, scale=scale, soft_cap=soft_cap,
                                           max_allowed_batch=1, thrink_buffer_threshold=500)
        torch.cuda.synchronize()
        pynvshmem.nvshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)

        def func():
            return ths_op(query, key_cache_this_rank, value_cache_this_rank, global_kv_lens_tensor,
                          block_tables_this_rank)

        perf_func(func, iters=100, warmup_iters=20)

        pynvshmem.nvshmem_barrier_all_on_stream(torch.cuda.current_stream().cuda_stream)

        with group_profile(f"sp_flash_decode_kv{kv_len_per_rank}", do_prof=args.profile, group=TP_GROUP):
            torch.cuda._sleep(1000000000)  # in case CPU bound
            _, time_ms = perf_func(
                func,
                warmup_iters=20,
                iters=100,
            )
        torch.distributed.barrier(args.default_group)
        dist_print(f"rank: {args.rank} KV len={kv_lens_per_rank[0]} Performance is {time_ms} ms", allowed_ranks="all",
                   need_sync=True)


if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", 0))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    torch.cuda.set_device(LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl",
        world_size=WORLD_SIZE,
        rank=RANK,
        timeout=datetime.timedelta(seconds=1800),
    )
    assert torch.distributed.is_initialized()
    TP_GROUP = torch.distributed.new_group(ranks=list(range(WORLD_SIZE)), backend="nccl")
    torch.distributed.barrier(TP_GROUP)

    torch.use_deterministic_algorithms(False, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + RANK)
    torch.cuda.manual_seed_all(3 + RANK)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + RANK)

    current_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    init_nvshmem_by_uniqueid(TP_GROUP)
    pynvshmem.nvshmem_barrier_all()
    torch.cuda.synchronize()

    args = get_args()
    args.default_group = TP_GROUP
    args.rank = RANK
    args.num_ranks = WORLD_SIZE
    args.local_rank = LOCAL_RANK
    args.local_num_ranks = LOCAL_WORLD_SIZE
    if args.list:
        help()
        sys.exit()
    func = ALL_TESTS[args.case]
    func(args)

    torch.distributed.destroy_process_group()
