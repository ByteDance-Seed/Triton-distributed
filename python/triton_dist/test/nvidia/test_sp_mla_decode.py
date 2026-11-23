################################################################################
#
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
"""Tests for sequence-parallel MLA decode (``SpMLAFlashDecodeAttention``).

Every rank holds the same absorbed query and a contiguous slice of each
sequence's compressed KV. The result is compared against a dense fp32 MLA
decode over the whole sequence on one GPU.

run:
    bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_sp_mla_decode.py --case correctness
    bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_sp_mla_decode.py --case cuda_graph
    bash ./scripts/launch.sh python/triton_dist/test/nvidia/test_sp_mla_decode.py --case perf
"""
import argparse
import itertools
import os
import sys

import torch

from triton_dist.layers.nvidia import SpMLAFlashDecodeAttention
from triton_dist.profiler_utils import perf_func
from triton_dist.utils import dist_print, finalize_distributed, initialize_distributed, nvshmem_barrier_all_on_stream

ALL_TESTS = {}

DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
PARTITIONS = ["even", "skewed", "one_empty", "ragged"]


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func
        return func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()))
    parser.add_argument("--kv_lora_rank", type=int, default=512)
    parser.add_argument("--qk_rope_head_dim", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="", choices=[""] + list(DTYPES.keys()),
                        help="restrict correctness/cuda_graph to one dtype (default: both)")
    parser.add_argument("--iters", type=int, default=20, help="replays for cuda_graph")
    parser.add_argument("--kv_split", type=int, default=32)
    parser.add_argument("--block_n", type=int, default=32)
    parser.add_argument("--num_stages", type=int, default=2)
    return parser.parse_args()


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


def partition(seq_len, rank, world, mode, row):
    """This rank's ``[lo, hi)`` of a sequence of ``seq_len`` tokens in batch row ``row``."""
    if mode == "even":
        return (seq_len * rank) // world, (seq_len * (rank + 1)) // world
    if mode == "one_empty":
        # rank 0 holds everything, every other rank is empty
        return (0, seq_len) if rank == 0 else (seq_len, seq_len)
    if mode == "skewed":
        # rank 0 holds almost everything, every other rank holds one token
        assert seq_len >= world
        if rank == 0:
            return 0, seq_len - (world - 1)
        base = seq_len - (world - 1) + (rank - 1)
        return base, base + 1
    if mode == "ragged":
        # row ``row`` is spread over the first ``row % world + 1`` ranks, the rest are empty for it
        used = row % world + 1
        if rank >= used:
            return seq_len, seq_len
        return (seq_len * rank) // used, (seq_len * (rank + 1)) // used
    raise ValueError(f"unknown partition mode {mode!r}")


def dense_mla_decode_reference(q, kv, seq_lens, kv_lora_rank, scale):
    """fp32 MLA decode over full sequences. q: [B, H, K], kv: [B, S_max, K]. Returns [B, H, kv_lora_rank]."""
    out = []
    for b, seq_len in enumerate(seq_lens):
        qb = q[b].float()
        kb = kv[b, :seq_len].float()
        p = torch.softmax((qb @ kb.T) * scale, dim=-1)
        out.append(p @ kb[:, :kv_lora_rank])
    return torch.stack(out)


def build_shard(kv, seq_lens, args, mode, page_size, max_pages=None):
    """Page this rank's slice of every sequence into a shuffled paged cache.

    ``max_pages`` fixes the per-row page budget, so layouts from different partitions have the same shapes.

    Returns (kv_cache [num_pages, page_size, 1, K], block_table [B, max_pages], global_kv_lens [W, B]).
    """
    batch, _, head_dim = kv.shape
    ranges = [[partition(seq_len, r, args.num_ranks, mode, b)
               for b, seq_len in enumerate(seq_lens)]
              for r in range(args.num_ranks)]
    global_kv_lens = torch.tensor([[hi - lo for lo, hi in row] for row in ranges], dtype=torch.int32,
                                  device="cuda").contiguous()

    local = ranges[args.rank]
    needed = max(1, max((hi - lo + page_size - 1) // page_size for lo, hi in local))
    max_pages = needed if max_pages is None else max_pages
    assert needed <= max_pages
    num_pages = batch * max_pages
    gen = torch.Generator(device="cpu").manual_seed(1234 + args.rank)
    perm = torch.randperm(num_pages, generator=gen).to(device="cuda", dtype=torch.int32)
    block_table = perm.view(batch, max_pages).contiguous()

    kv_cache = torch.zeros(num_pages, page_size, 1, head_dim, dtype=kv.dtype, device="cuda")
    flat = kv_cache.view(num_pages * page_size, head_dim)
    for b, (lo, hi) in enumerate(local):
        n = hi - lo
        if n == 0:
            continue
        pos = torch.arange(n, device="cuda")
        slots = block_table[b, pos // page_size].long() * page_size + pos % page_size
        flat[slots] = kv[b, lo:hi]
    return kv_cache, block_table, global_kv_lens


def make_inputs(args, batch, num_heads, seq_lens, dtype):
    head_dim = args.kv_lora_rank + args.qk_rope_head_dim
    q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device="cuda")
    kv = torch.randn(batch, max(seq_lens), head_dim, dtype=dtype, device="cuda")
    torch.distributed.broadcast(q, src=0, group=args.default_group)
    torch.distributed.broadcast(kv, src=0, group=args.default_group)
    return q, kv


def tolerance(dtype):
    return dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=1e-2, rtol=1e-2)


def make_op(args, num_heads, page_size, scale, max_batch):
    return SpMLAFlashDecodeAttention(args.rank, args.num_ranks, num_heads, kv_lora_rank=args.kv_lora_rank,
                                     qk_rope_head_dim=args.qk_rope_head_dim, page_size=page_size, scale=scale,
                                     max_allowed_batch=max_batch, kv_split=args.kv_split, block_n=args.block_n,
                                     num_stages=args.num_stages)


def selected_dtypes(args):
    return [DTYPES[args.dtype]] if args.dtype else list(DTYPES.values())


@register_test("correctness")
def test_correctness(args):
    scale = (128 + args.qk_rope_head_dim)**-0.5
    configs = itertools.product(
        selected_dtypes(args),
        [1, 4],  # batch
        [8, 16, 128],  # heads
        [1, 64],  # page size
        PARTITIONS,
    )
    max_batch = 4
    ops = {}
    num_cases = 0
    for dtype, batch, num_heads, page_size, mode in configs:
        key = (num_heads, page_size)
        if key not in ops:
            ops[key] = make_op(args, num_heads, page_size, scale, max_batch)
        op = ops[key]
        # different lengths per row, including ones that are not multiples of the page or tile size
        seq_lens = [4096 + 517 * b for b in range(batch)]
        q, kv = make_inputs(args, batch, num_heads, seq_lens, dtype)
        kv_cache, block_table, global_kv_lens = build_shard(kv, seq_lens, args, mode, page_size)

        out = op(q, kv_cache, global_kv_lens, block_table)
        ref = dense_mla_decode_reference(q, kv, seq_lens, args.kv_lora_rank, scale)
        assert out.dtype == dtype and out.shape == (batch, num_heads, args.kv_lora_rank)
        try:
            torch.testing.assert_close(out.float(), ref, **tolerance(dtype))
        except AssertionError:
            dist_print(
                f"FAIL dtype={dtype} batch={batch} heads={num_heads} page_size={page_size} partition={mode} "
                f"max_abs_err={(out.float() - ref).abs().max().item():.3e}", allowed_ranks="all")
            raise
        num_cases += 1
    torch.cuda.synchronize()
    for op in ops.values():
        op.finalize()
    dist_print(f"correctness: {num_cases} cases passed", allowed_ranks=[0])


def capture(func, warmup=3):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            func()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = func()
    return graph, out


@register_test("cuda_graph")
def test_cuda_graph(args):
    """Capture once, then replay with new q / KV / lengths written in place between replays."""
    scale = (128 + args.qk_rope_head_dim)**-0.5
    batch, num_heads, page_size = 4, 16, 64
    seq_lens = [8192, 5000, 777, 3001]
    for dtype in selected_dtypes(args):
        op = make_op(args, num_heads, page_size, scale, batch)
        q, kv = make_inputs(args, batch, num_heads, seq_lens, dtype)
        # Every layout gets the page budget of a whole sequence, so all replays fit the captured tensors.
        max_pages = (max(seq_lens) + page_size - 1) // page_size
        kv_cache, block_table, global_kv_lens = build_shard(kv, seq_lens, args, "even", page_size, max_pages)
        static_q, static_kv_cache = q.clone(), kv_cache.clone()
        static_block_table, static_lens = block_table.clone(), global_kv_lens.clone()

        graph, static_out = capture(lambda: op(static_q, static_kv_cache, static_lens, static_block_table))
        for it in range(args.iters):
            mode = PARTITIONS[it % len(PARTITIONS)]
            q, kv = make_inputs(args, batch, num_heads, seq_lens, dtype)
            kv_cache, block_table, global_kv_lens = build_shard(kv, seq_lens, args, mode, page_size, max_pages)
            static_q.copy_(q)
            static_kv_cache.copy_(kv_cache)
            static_block_table.copy_(block_table)
            static_lens.copy_(global_kv_lens)

            graph.replay()
            ref = dense_mla_decode_reference(q, kv, seq_lens, args.kv_lora_rank, scale)
            try:
                torch.testing.assert_close(static_out.float(), ref, **tolerance(dtype))
            except AssertionError:
                dist_print(
                    f"FAIL replay={it} dtype={dtype} partition={mode} "
                    f"max_abs_err={(static_out.float() - ref).abs().max().item():.3e}", allowed_ranks="all")
                raise
        torch.cuda.synchronize()
        del graph
        op.finalize()
        dist_print(f"cuda_graph {dtype}: {args.iters} replays passed", allowed_ranks=[0])


@register_test("perf")
def perf_decode(args):
    """Eager and CUDA-graph latency of one forward, KV split evenly across ranks."""
    scale = (128 + args.qk_rope_head_dim)**-0.5
    head_dim = args.kv_lora_rank + args.qk_rope_head_dim
    dtype = DTYPES[args.dtype] if args.dtype else torch.bfloat16
    page_size = 64
    for num_heads, batch, total_len in itertools.product([16, 128], [1, 8], [2**i for i in range(12, 18)]):
        local_len = total_len // args.num_ranks
        max_pages = (local_len + page_size - 1) // page_size
        q = torch.randn(batch, num_heads, head_dim, dtype=dtype, device="cuda")
        kv_cache = torch.randn(batch * max_pages, page_size, 1, head_dim, dtype=dtype, device="cuda")
        block_table = torch.randperm(batch * max_pages, device="cuda").to(torch.int32).view(batch, max_pages)
        global_kv_lens = torch.full((args.num_ranks, batch), local_len, dtype=torch.int32, device="cuda")
        op = make_op(args, num_heads, page_size, scale, batch)

        def func():
            return op(q, kv_cache, global_kv_lens, block_table)

        nvshmem_barrier_all_on_stream()
        _, eager_ms = perf_func(func, iters=100, warmup_iters=20)
        graph, _ = capture(func)
        nvshmem_barrier_all_on_stream()
        _, graph_ms = perf_func(graph.replay, iters=100, warmup_iters=20)
        torch.cuda.synchronize()
        dist_print(
            f"heads={num_heads} batch={batch} total_kv={total_len} kv_per_rank={local_len} {dtype}: "
            f"eager {eager_ms * 1000:.1f} us, graph {graph_ms * 1000:.1f} us", allowed_ranks=[0], need_sync=True)
        del graph
        op.finalize()


if __name__ == "__main__":
    args = get_args()
    if args.list:
        help()
        sys.exit()
    args.default_group = initialize_distributed()
    args.rank = int(os.environ.get("RANK", 0))
    args.num_ranks = int(os.environ.get("WORLD_SIZE", 1))
    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    args.local_num_ranks = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    assert args.num_ranks == args.local_num_ranks, "SpMLAFlashDecodeAttention is single-node"
    torch.manual_seed(args.rank)
    ALL_TESTS[args.case](args)
    finalize_distributed()
