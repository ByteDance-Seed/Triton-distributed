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
"""Sequence-parallel MLA decode (issue #60).

Each rank owns a contiguous slice ``[lo, hi)`` of the global KV sequence and
computes the absorbed-query partial statistics

    m = max_t score[t]        l = sum_t exp(score[t] - m)
    u = sum_t exp(score[t] - m) * c_kv[t]

    score[t] = scale * (q_latent[h] . c_kv[t] + q_rope[h] . k_rope[t])

which are then merged across ranks. Sharing the query and the KV tensor means
the shards are slices of *one* problem; only the statistics travel between
ranks, so the reduction payload does not scale with the sequence length.

The merge resolves the global max first and rescales each rank's numerator,
so an empty shard contributes ``m=-inf, l=0`` and is weighted out while still
taking part in the control protocol.

What is deliberately out of scope here, and covered by the companion kernel
work instead: the final D_v up-projection, deeper compute/communication
overlap, and graph execution. See the issue for the measured trade-off between
merging the latent numerator and projecting before the reduction.

Run::

    python test_sp_mla_decode.py --case correctness
    python test_sp_mla_decode.py --case perf
    python test_sp_mla_decode.py --list
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import triton
import triton.language as tl

import nvshmem.core

try:
    # The suite's own bootstrap and symmetric-tensor helpers, so this test
    # behaves like the other distributed tests and inherits their process group
    # (cpu:gloo + cuda:nccl) and NVSHMEM uid handshake.
    from triton_dist.utils import (dist_print, finalize_distributed, initialize_distributed, nvshmem_create_tensor,
                                   nvshmem_free_tensor_sync)
except ImportError as exc:  # pragma: no cover - depends on the installed triton
    # triton_dist.utils imports triton_dist.language, which needs a newer Triton
    # than some environments pair with NVSHMEM. The semantics below are the ones
    # that module implements, inlined so the test still runs there; the guard is
    # narrow on purpose and does not swallow other failure modes.
    _REASON = exc

    def dist_print(*args, **_kwargs):
        print(*args, flush=True)

    def initialize_distributed(seed=None, initialize_shmem: bool = True):
        from cuda.core import Device

        rank = int(os.environ.get("RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", 1))
        # torchrun sets these; defaulting them keeps a plain two-process launch
        # working on a single node.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        torch.cuda.set_device(local_rank())
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
        pg = torch.distributed.group.WORLD
        if initialize_shmem:
            uid = nvshmem.core.get_unique_id(empty=(rank != 0))
            box = [uid]
            torch.distributed.broadcast_object_list(box, src=0)
            nvshmem.core.init(device=Device(torch.cuda.current_device()),
                              uid=box[0],
                              rank=rank,
                              nranks=world,
                              initializer_method="uid")
        return pg

    def finalize_distributed() -> None:
        nvshmem.core.finalize()
        torch.distributed.destroy_process_group()

    def nvshmem_create_tensor(shape, dtype) -> torch.Tensor:
        torch.cuda.synchronize()
        tensor = nvshmem.core.tensor(shape, dtype=dtype)
        torch.cuda.synchronize()
        return tensor

    def nvshmem_free_tensor_sync(tensor) -> None:
        torch.cuda.synchronize()
        nvshmem.core.free_tensor(tensor)
        torch.cuda.synchronize()

ALL_TESTS = {}


def register_test(name):

    def wrapper(func):
        assert name not in ALL_TESTS
        ALL_TESTS[name] = func

    return wrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", type=str, choices=list(ALL_TESTS.keys()), default="correctness")
    parser.add_argument("--S", type=int, default=4096, help="global KV length")
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--R", type=int, default=512, help="latent (absorption) dim")
    parser.add_argument("--D-rope", type=int, default=64)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--partition", type=str, default="even",
                        choices=["even", "skewed", "one_empty"])
    return parser.parse_args()


def help():
    print(f"""
Available choices: {list(ALL_TESTS.keys())}.
run: python {os.path.abspath(__file__)} --case XXX
""")


class _StreamAdapter:
    """Adapt a torch stream to the ``__cuda_stream__`` protocol.

    nvshmem4py takes the stream as an object exposing ``__cuda_stream__()``
    returning ``(device, handle)``. Newer torch streams implement that directly;
    the torch this was first brought up against exposes only the raw
    ``cuda_stream`` handle, so it is wrapped. Passing the object through when it
    already implements the protocol keeps the newer path unbranched.
    """

    __slots__ = ("_stream",)

    def __init__(self, stream):
        self._stream = stream

    def __cuda_stream__(self):
        return (self._stream.device.index, self._stream.cuda_stream)


def current_stream():
    stream = torch.cuda.current_stream()
    return stream if hasattr(stream, "__cuda_stream__") else _StreamAdapter(stream)


def local_rank() -> int:
    """Rank within the node, defaulting to the global rank on a single node."""
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))


def device() -> torch.device:
    return torch.device("cuda", local_rank())


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------
@triton.jit
def mla_partial_kernel(
    q_latent_ptr, q_rope_ptr, c_kv_ptr, k_rope_ptr,
    m_ptr, l_ptr, u_ptr,
    scale,
    seq_len,
    stride_cl, stride_cr,
    stride_kr,
    stride_ul, stride_ur,
    R: tl.constexpr,
    D_ROPE: tl.constexpr,
    H: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One (head, tile) pair: online-softmax partial over this rank's slice."""
    h = tl.program_id(0)
    t = tl.program_id(1)

    offs_s = t * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_r = tl.arange(0, BLOCK_R)
    offs_d = tl.arange(0, BLOCK_D)
    s_mask = offs_s < seq_len
    r_mask = offs_r < R
    d_mask = offs_d < D_ROPE

    q_l = tl.load(q_latent_ptr + h * R + offs_r, mask=r_mask, other=0.0)
    q_r = tl.load(q_rope_ptr + h * D_ROPE + offs_d, mask=d_mask, other=0.0)

    c = tl.load(c_kv_ptr + offs_s[:, None] * stride_cl + offs_r[None, :] * stride_cr,
                mask=s_mask[:, None] & r_mask[None, :], other=0.0)
    k = tl.load(k_rope_ptr + offs_s[:, None] * stride_kr + offs_d[None, :],
                mask=s_mask[:, None] & d_mask[None, :], other=0.0)

    score = scale * (tl.sum(c * q_l[None, :], axis=1) + tl.sum(k * q_r[None, :], axis=1))
    score = tl.where(s_mask, score, float("-inf"))

    tile_max = tl.max(score, axis=0)
    tile_max = tl.where(tile_max == float("-inf"), -1e30, tile_max)
    e = tl.where(s_mask, tl.exp(score - tile_max), 0.0)
    tile_l = tl.sum(e, axis=0)
    tile_u = tl.sum(e[:, None] * c, axis=0)

    # an empty tile reports l == 0 so the merge can weight it out
    n_active = tl.sum(tl.where(s_mask, 1, 0), axis=0)
    empty = n_active == 0
    tl.store(m_ptr + t * H + h, tl.where(empty, float("-inf"), tile_max))
    tl.store(l_ptr + t * H + h, tl.where(empty, 0.0, tile_l))
    tl.store(u_ptr + (t * H + h) * stride_ul + offs_r * stride_ur,
             tl.where(empty, 0.0, tile_u), mask=r_mask)


@triton.jit
def mla_merge_partials_kernel(
    m_ptr, l_ptr, u_ptr,
    m_out_ptr, l_out_ptr, u_out_ptr,
    n_tiles,
    stride_ul, stride_ur,
    R: tl.constexpr,
    H: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Combine per-tile partials into one shard partial.

    Two passes on purpose: resolving the running max first makes every later
    exponential finite, which a single pass cannot do without either a
    loop-carried "seen a valid tile" flag or the ``exp(-inf - -inf)`` NaN that
    accumulating from ``-inf`` produces when the first tile is empty.
    """
    h = tl.program_id(0)
    offs_r = tl.arange(0, BLOCK_R)
    r_mask = offs_r < R

    m = tl.full((), float("-inf"), tl.float32)
    l_any = 0.0
    for t in range(n_tiles):
        m = tl.maximum(m, tl.load(m_ptr + t * H + h))
        l_any = tl.maximum(l_any, tl.load(l_ptr + t * H + h))
    has_valid = l_any > 0.0

    l = 0.0
    u = tl.zeros((BLOCK_R, ), dtype=tl.float32)
    for t in range(n_tiles):
        lt = tl.load(l_ptr + t * H + h)
        mt = tl.load(m_ptr + t * H + h)
        ut = tl.load(u_ptr + (t * H + h) * stride_ul + offs_r * stride_ur,
                     mask=r_mask, other=0.0)
        mt_safe = tl.where(lt > 0.0, mt, -1e30)
        w = tl.exp(mt_safe - m)
        l = l + w * lt
        u = u + w * ut

    l = tl.where(has_valid, l, 0.0)
    u = tl.where(has_valid, u, 0.0)
    tl.store(m_out_ptr + h, m)
    tl.store(l_out_ptr + h, l)
    tl.store(u_out_ptr + h * stride_ul + offs_r * stride_ur, u, mask=r_mask)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def choose_block_s(r: int, target_elems: int = 16384, lo: int = 32, hi: int = 128) -> int:
    """Pick BLOCK_S so the tile's score vector stays near ``target_elems``.

    A fixed 128 spills badly once R is large (255 registers / 1938 spills at
    R=512 on SM89), which is a blocking cost rather than a parallelism loss.
    """
    bs = max(1, target_elems // max(1, r))
    bs = 1 << (bs.bit_length() - 1)
    return max(lo, min(hi, bs))


def partition(seq_len: int, rank: int, world: int, mode: str) -> tuple[int, int]:
    """This rank's ``[lo, hi)`` over the global KV sequence."""
    if mode == "even":
        return (seq_len * rank) // world, (seq_len * (rank + 1)) // world
    if mode == "one_empty":
        return (0, seq_len) if rank == 0 else (seq_len, seq_len)
    if mode == "skewed":
        if rank == 0:
            return 0, max(0, seq_len - (world - 1))
        base = seq_len - (world - 1) + (rank - 1)
        return base, base + 1
    raise ValueError(f"unknown partition mode {mode!r}")


def run_local_partial(q_latent, q_rope, c_kv, k_rope, scale, block_s=None):
    """Tiled local partial. Shapes: [H,R], [H,D], [n,R], [n,D]."""
    h, r = q_latent.shape
    if block_s is None:
        block_s = choose_block_s(r)
    d_rope = q_rope.shape[1]
    seq_len = c_kv.shape[0]
    dev = q_latent.device
    block_r = triton.next_power_of_2(r)

    if seq_len == 0:
        return (torch.full((h, ), float("-inf"), device=dev),
                torch.zeros(h, device=dev),
                torch.zeros(h, r, device=dev))

    n_tiles = triton.cdiv(seq_len, block_s)
    m_t = torch.empty(n_tiles, h, device=dev)
    l_t = torch.empty(n_tiles, h, device=dev)
    u_t = torch.empty(n_tiles, h, r, device=dev)

    mla_partial_kernel[(h, n_tiles)](
        q_latent, q_rope, c_kv, k_rope, m_t, l_t, u_t, scale, seq_len,
        c_kv.stride(0), c_kv.stride(1), k_rope.stride(0),
        u_t.stride(1), u_t.stride(2),
        R=r, D_ROPE=d_rope, H=h, BLOCK_S=block_s, BLOCK_R=block_r,
        BLOCK_D=triton.next_power_of_2(d_rope))

    m_o = torch.empty(h, device=dev)
    l_o = torch.empty(h, device=dev)
    u_o = torch.empty(h, r, device=dev)
    mla_merge_partials_kernel[(h, )](
        m_t, l_t, u_t, m_o, l_o, u_o, n_tiles,
        u_o.stride(0), u_o.stride(1), R=r, H=h, BLOCK_R=block_r)
    return m_o, l_o, u_o


def dense_reference(q_latent, q_rope, c_kv, k_rope, scale):
    """Unsharded softmax over the whole sequence.

    With no tokens the softmax is undefined; the contract shared with the merge
    is that the all-empty case yields zeros rather than NaN or a crash.
    """
    if c_kv.shape[0] == 0:
        return torch.zeros(q_latent.shape[0], c_kv.shape[1],
                           dtype=q_latent.dtype, device=q_latent.device)
    latent = q_latent @ c_kv.T
    rope = q_rope @ k_rope.T
    s = scale * (latent + rope)
    e = torch.exp(s - s.max(dim=1, keepdim=True).values)
    return (e / e.sum(dim=1, keepdim=True)) @ c_kv


def merge_partials(ms, ls, us):
    """Resolve the global max, rescale, then reduce. Same algebra as the M1 oracle."""
    m = torch.stack(ms)
    l = torch.stack(ls)
    u = torch.stack(us)

    m_max = m.max(dim=0).values
    w = torch.exp(torch.where(l > 0, m - m_max, torch.full_like(m, -1e30)))
    l_tot = (w * l).sum(dim=0)
    u_tot = (w[:, :, None] * u).sum(dim=0)

    has = l_tot > 0
    safe = torch.where(has, l_tot, torch.ones_like(l_tot))
    return torch.where(has[:, None], u_tot / safe[:, None], torch.zeros_like(u_tot))


def make_problem(args, rank, dev):
    """One problem for all ranks: every rank slices the same KV tensor."""
    torch.manual_seed(37)
    h, r, d = args.H, args.R, args.D_rope
    q_latent = torch.randn(h, r, device=dev)
    q_rope = torch.randn(h, d, device=dev)
    c_kv = torch.randn(args.S, r, device=dev)
    k_rope = torch.randn(args.S, d, device=dev)
    return q_latent, q_rope, c_kv, k_rope


class Exchange:
    """Reusable NVSHMEM symmetric buffers for the three partials."""

    def __init__(self, h: int, r: int):
        self.m = nvshmem_create_tensor((h, ), torch.float32)
        self.l = nvshmem_create_tensor((h, ), torch.float32)
        self.u = nvshmem_create_tensor((h, r), torch.float32)

    def publish(self, m, l, u):
        self.m[:] = m
        self.l[:] = l
        self.u[:] = u
        torch.cuda.synchronize()
        nvshmem.core.quiet(stream=current_stream())
        nvshmem.core.barrier_all(stream=current_stream())
        torch.cuda.synchronize()

    def gather(self, rank: int, world: int):
        ms, ls, us = [], [], []
        for peer in range(world):
            if peer == rank:
                ms.append(self.m)
                ls.append(self.l)
                us.append(self.u)
            else:
                ms.append(nvshmem.core.get_peer_tensor(self.m, peer).clone())
                ls.append(nvshmem.core.get_peer_tensor(self.l, peer).clone())
                us.append(nvshmem.core.get_peer_tensor(self.u, peer).clone())
        nvshmem.core.quiet(stream=current_stream())
        nvshmem.core.barrier_all(stream=current_stream())
        torch.cuda.synchronize()
        return ms, ls, us

    def free(self):
        for buf in (self.m, self.l, self.u):
            nvshmem_free_tensor_sync(buf)


def report_and_check(args, rank, world, got, want, n_local, exch):
    """One line per rank; every rank including the empty ones must match."""
    err = float((got - want).abs().max().item())
    finite = bool(torch.isfinite(got).all())
    dist_print(f"rank {rank}: n_local={n_local} max_abs_err={err:.3e} finite={finite} "
               f"{'OK' if err < 1e-4 and finite else 'FAIL'}")
    return err < 1e-4 and finite


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@register_test("correctness")
def test_correctness(args, rank, world, tp_group, dev):
    h, r = args.H, args.R
    q_latent, q_rope, c_kv, k_rope = make_problem(args, rank, dev)
    scale = 1.0 / (r ** 0.5)
    lo, hi = partition(args.S, rank, world, args.partition)
    n_local = hi - lo

    exch = Exchange(h, r)
    ok = True
    for it in range(3):
        local_c = c_kv[lo:hi].contiguous()
        local_k = k_rope[lo:hi].contiguous()
        m, l, u = run_local_partial(q_latent, q_rope, local_c, local_k, scale)
        exch.publish(m, l, u)
        ms, ls, us = exch.gather(rank, world)
        got = merge_partials(ms, ls, us)
        want = dense_reference(q_latent, q_rope, c_kv, k_rope, scale)
        ok &= report_and_check(args, rank, world, got, want, n_local, exch)
        if it + 1 < 3:
            torch.distributed.barrier(tp_group)
    exch.free()

    # every rank must agree, so the result is reduced rather than per-rank
    agree = torch.tensor([1.0 if ok else 0.0], device=dev)
    torch.distributed.all_reduce(agree, op=torch.distributed.ReduceOp.MIN, group=tp_group)
    dist_print(f"RESULT: {'PASS' if agree.item() == 1.0 else 'FAIL'} "
               f"(partition={args.partition} S={args.S} H={h} R={r} world={world})")


@register_test("perf")
def test_perf(args, rank, world, tp_group, dev):
    h, r = args.H, args.R
    q_latent, q_rope, c_kv, k_rope = make_problem(args, rank, dev)
    scale = 1.0 / (r ** 0.5)
    lo, hi = partition(args.S, rank, world, args.partition)
    local_c = c_kv[lo:hi].contiguous()
    local_k = k_rope[lo:hi].contiguous()

    exch = Exchange(h, r)

    # Warm up before timing: the first call JIT-compiles the kernels, and the
    # compile cost otherwise lands on whichever rank reaches it first, which is
    # exactly the asymmetry this measurement is supposed to report.
    for _ in range(3):
        run_local_partial(q_latent, q_rope, local_c, local_k, scale)
        torch.cuda.synchronize()
    torch.distributed.barrier(tp_group)

    local_ms, exchange_ms = [], []
    for _ in range(args.iters):
        # Wall-clock around an explicit device sync, not CUDA events: the barrier
        # is host-blocking, so events placed before/after it time the host
        # round-trip rather than any GPU work. What is reported below therefore
        # includes host overhead and is the cost a caller actually observes.
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m, l, u = run_local_partial(q_latent, q_rope, local_c, local_k, scale)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        exch.publish(m, l, u)
        exch.gather(rank, world)
        t2 = time.perf_counter()
        local_ms.append((t1 - t0) * 1e3)
        exchange_ms.append((t2 - t1) * 1e3)

    exch.free()
    dist_print(f"rank {rank}: local_p50={statistics.median(local_ms):.4f}ms "
               f"exchange_p50={statistics.median(exchange_ms):.4f}ms "
               f"layer_p50={statistics.median(local_ms) + statistics.median(exchange_ms):.4f}ms "
               f"(S={args.S} n_local={hi - lo} world={world})")


if __name__ == "__main__":
    args = get_args()
    if args.list:
        help()
        sys.exit()

    torch.cuda.set_device(device())
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    TP_GROUP = initialize_distributed()
    ALL_TESTS[args.case](args, RANK, WORLD_SIZE, TP_GROUP, device())
    finalize_distributed()
