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
"""Build, validate, and benchmark SM100 Ulysses post-attention A2A plus o-proj GEMM."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT, role_example_path, role_example_source

import argparse
import gc
import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from benchmarks.nv.sm100.dataflow.bench_report import ReportRow, SpeedupRow, print_benchmark_report, print_speedups
from tools.merge_rank_traces import merge_rank_traces
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
from examples.nv.sm100.a2a_gemm_fused_sm100 import A2AGemmFuse
from examples.nv.sm100.a2a_tma_push_sm100 import TmaPushA2A
from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

GEMM_SOURCE = role_example_source("a2a_gemm")
GEMM_ONLY_EPOCH = (1 << 30) - 1
CHECK_RTOL = 6e-2
CHECK_ATOL = 6e-2
REFRESH_SECONDS = 0.4


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(GEMM_SOURCE, "<role-v2-a2a-gemm>", "exec"), ns)
    return ns


def build_gemm_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    c = _cfg()
    return CompiledKernel(
        compile_dataflow_cuda(GEMM_SOURCE),
        "a2a_gemm_sm100_kernel",
        grid=(1, 1, 1),
        block=(c["NUM_THREADS"], 1, 1),
        shared_mem_bytes=c["SHARED_MEM_BYTES"],
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c["a2a_gemm_sm100_kernel"].cluster_shape,
        cta_group_shape=c["a2a_gemm_sm100_kernel"].cta_group_shape,
        allow_non_portable_cluster=True,
    )


def _gemm_descriptors(workspace: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    c = _cfg()
    m, k = workspace.shape
    n = b.shape[0]
    d_a = create_tma_2d_descriptor(
        workspace,
        gmem_inner_dim=k,
        gmem_outer_dim=m,
        smem_inner_dim=c["K_TILE"],
        smem_outer_dim=c["CTA_TILE_M"],
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    d_b = create_tma_2d_descriptor(
        b,
        gmem_inner_dim=k,
        gmem_outer_dim=n,
        smem_inner_dim=c["K_TILE"],
        smem_outer_dim=c["CTA_TILE_N_SHARD"],
        gmem_outer_stride=k,
        oob_fill=True,
        l2_promotion=3,
        swizzle_mode=128,
    )
    d_d = create_tma_2d_descriptor(
        out,
        gmem_inner_dim=n,
        gmem_outer_dim=m,
        smem_inner_dim=c["ST_TILE_N"],
        smem_outer_dim=c["ST_TILE_M"],
        gmem_outer_stride=n,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )
    return d_a, d_b, d_d


@dataclass
class A2AGemm:
    gemm_kernel: CompiledKernel
    attn_out: torch.Tensor
    b: torch.Tensor
    workspace: torch.Tensor
    out: torch.Tensor
    ready: torch.Tensor
    d_a: object
    d_b: object
    d_d: object
    a2a: TmaPushA2A
    standalone_a2a: TmaPushA2A
    torch_send: torch.Tensor
    torch_recv: torch.Tensor
    torch_gathered: torch.Tensor
    torch_out: torch.Tensor
    rank: int
    world_size: int
    seq_len: int
    local_m: int
    n: int
    k: int
    band_features: int
    q_heads: int
    head_dim: int
    compute_ctas: int
    full_compute_ctas: int
    reserved_comm_ctas: int
    workspace_handle: object
    ready_handle: object
    comm_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    epoch: int = 0

    def next_epoch(self) -> int:
        self.epoch = self.epoch % ((1 << 30) - 2) + 1
        return self.epoch

    def launch_gemm(self, epoch: int, *, standalone: bool = False, stream: torch.cuda.Stream | None = None) -> None:
        num_ctas = self.full_compute_ctas if standalone else self.compute_ctas
        self.gemm_kernel(
            self.d_a,
            self.d_b,
            self.d_d,
            self.ready,
            self.local_m,
            self.n,
            self.k,
            self.band_features,
            self.world_size,
            epoch,
            num_ctas,
            grid=(num_ctas, 1, 1),
            stream=self.compute_stream if stream is None else stream,
        )

    def launch_a2a(self, epoch: int, *, standalone: bool = False, stream: torch.cuda.Stream | None = None) -> None:
        target = self.standalone_a2a if standalone else self.a2a
        target.launch(epoch, stream=self.comm_stream if stream is None else stream)


@dataclass
class TritonDistA2AGemm:
    op: object
    inputs: torch.Tensor
    weight: torch.Tensor
    output: torch.Tensor
    num_comm_sms: int
    sm_margin: int
    out: torch.Tensor | None = None

    def launch(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            result = self.op.forward(
                self.inputs,
                self.weight,
                output=self.output,
                num_comm_sms=self.num_comm_sms,
                sm_margin=self.sm_margin,
            )
            self.out = result[0]

    def release(self) -> None:
        self.op.finalize()
        self.op = self.inputs = self.weight = None
        self.output = None
        self.out = None


def prepare(
    gemm_kernel: CompiledKernel,
    a2a_kernel: CompiledKernel,
    *,
    seq_len: int,
    hidden: int,
    q_heads: int,
    head_dim: int,
    comm_ctas: int,
    comm_reserve_ctas: int | None,
) -> A2AGemm:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    c = _cfg()

    if seq_len % world_size:
        raise ValueError("seq_len must be divisible by world_size")
    if q_heads % world_size:
        raise ValueError("q_heads must be divisible by world_size")
    if TmaPushA2A.PUSH_BM != c["CTA_TILE_M"]:
        raise ValueError("push publication granularity must match the GEMM CTA M tile")

    local_m = seq_len // world_size
    local_q_heads = q_heads // world_size
    band_features = local_q_heads * head_dim
    k = q_heads * head_dim
    n = hidden
    if local_m % c["PAIR_TILE_M"] or n % c["PAIR_TILE_N"] or k % c["K_TILE"]:
        raise ValueError(f"local seq, hidden, and QKV features must be divisible by "
                         f"{c['PAIR_TILE_M']}, {c['PAIR_TILE_N']}, and {c['K_TILE']}")
    if band_features % c["K_TILE"]:
        raise ValueError(f"per-rank K band must be divisible by K_TILE={c['K_TILE']}")

    pair_tiles = (local_m // c["PAIR_TILE_M"]) * (n // c["PAIR_TILE_N"])
    if pair_tiles < 2:
        raise ValueError("A2A-GEMM requires at least two pair tiles")

    a_gen = torch.Generator(device=device)
    a_gen.manual_seed(2026 + rank)
    b_gen = torch.Generator(device=device)
    b_gen.manual_seed(1729)
    attn_out = torch.randn((seq_len, band_features), dtype=torch.bfloat16, device=device, generator=a_gen).mul_(0.1)
    b = torch.randn((n, k), dtype=torch.bfloat16, device=device, generator=b_gen).mul_(0.1)

    workspace = symm_mem.empty((local_m, k), dtype=torch.bfloat16, device=device)
    workspace_handle = symm_mem.rendezvous(workspace, group=dist.group.WORLD)
    workspace.zero_()

    num_m_blocks = local_m // c["CTA_TILE_M"]
    ready = symm_mem.empty((world_size * num_m_blocks * TmaPushA2A.KSPLIT, ), dtype=torch.int32, device=device)
    ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
    ready.zero_()

    out = torch.empty((local_m, n), dtype=torch.bfloat16, device=device)
    d_a, d_b, d_d = _gemm_descriptors(workspace, b, out)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    a2a = TmaPushA2A.create(
        a2a_kernel,
        attn_out=attn_out,
        workspace_handle=workspace_handle,
        workspace=workspace,
        ready_handle=ready_handle,
        ready=ready,
        rank=rank,
        world_size=world_size,
        comm_ctas=comm_ctas,
    )
    standalone_a2a = TmaPushA2A.create(
        a2a_kernel,
        attn_out=attn_out,
        workspace_handle=workspace_handle,
        workspace=workspace,
        ready_handle=ready_handle,
        ready=ready,
        rank=rank,
        world_size=world_size,
        comm_ctas=sms,
    )

    requested_reserve = a2a.num_ctas if comm_reserve_ctas is None else comm_reserve_ctas
    if requested_reserve <= 0:
        raise ValueError("comm_reserve_ctas must be positive")
    reserved = max(1, min(requested_reserve, sms - c["CLUSTER_SIZE"]))
    compute_sms = max(c["CLUSTER_SIZE"], sms - reserved)
    compute_ctas = (min(compute_sms, pair_tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    full_compute_ctas = (min(sms, pair_tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    if compute_ctas == 0:
        raise ValueError("problem is too small for a two-CTA GEMM cluster")

    torch_send = attn_out.view(world_size, local_m, band_features)
    torch_recv = torch.empty((world_size, local_m, band_features), dtype=torch.bfloat16, device=device)
    torch_gathered = torch.empty((local_m, k), dtype=torch.bfloat16, device=device)
    torch_out = torch.empty((local_m, n), dtype=torch.bfloat16, device=device)
    return A2AGemm(
        gemm_kernel=gemm_kernel,
        attn_out=attn_out,
        b=b,
        workspace=workspace,
        out=out,
        ready=ready,
        d_a=d_a,
        d_b=d_b,
        d_d=d_d,
        a2a=a2a,
        standalone_a2a=standalone_a2a,
        torch_send=torch_send,
        torch_recv=torch_recv,
        torch_gathered=torch_gathered,
        torch_out=torch_out,
        rank=rank,
        world_size=world_size,
        seq_len=seq_len,
        local_m=local_m,
        n=n,
        k=k,
        band_features=band_features,
        q_heads=q_heads,
        head_dim=head_dim,
        compute_ctas=compute_ctas,
        full_compute_ctas=full_compute_ctas,
        reserved_comm_ctas=reserved,
        workspace_handle=workspace_handle,
        ready_handle=ready_handle,
        # High-priority comm stream matches GEMM-RS overlap: A2A CTAs should
        # win scheduling against compute when both kernels are resident.
        comm_stream=torch.cuda.Stream(priority=-1),
        compute_stream=torch.cuda.Stream(),
    )


def prepare_fused(ctx: A2AGemm, kernel: CompiledKernel) -> A2AGemmFuse:
    return A2AGemmFuse.create(
        kernel,
        attn_out=ctx.attn_out,
        workspace_handle=ctx.workspace_handle,
        workspace=ctx.workspace,
        dB=ctx.d_b,
        out_shape=ctx.out.shape,
        m=ctx.local_m,
        n=ctx.n,
        k=ctx.k,
        rank=ctx.rank,
        world_size=ctx.world_size,
    )


def prepare_triton_dist(
    ctx: A2AGemm,
    world_group,
    *,
    num_comm_sms: int,
    sm_margin: int,
    use_persistent: bool,
) -> TritonDistA2AGemm:
    try:
        from triton_dist.kernels.nvidia.sp_ulysess_o_all2all_gemm import SpUlysessOAll2AllGemmKernel
        from triton_dist.utils import nvshmem_barrier_all_on_stream
    except ImportError as exc:
        raise RuntimeError("Triton-distributed is required by default; install it or pass --no-triton-dist") from exc

    sms = torch.cuda.get_device_properties(ctx.attn_out.device).multi_processor_count
    if num_comm_sms <= 0 or num_comm_sms + sm_margin >= sms:
        raise ValueError("triton-dist comm SMs must be positive and leave at least one GEMM SM")
    if num_comm_sms >= ctx.world_size:
        if num_comm_sms % ctx.world_size:
            raise ValueError("triton-dist comm SMs must be divisible by world_size when comm SMs >= world_size")
    elif ctx.world_size % num_comm_sms:
        raise ValueError("world_size must be divisible by triton-dist comm SMs when comm SMs < world_size")

    op = SpUlysessOAll2AllGemmKernel(
        world_group,
        nnodes=1,
        sp_size=ctx.world_size,
        max_batch=1,
        num_head=ctx.q_heads,
        max_seqlen=ctx.seq_len,
        head_dim=ctx.head_dim,
        max_num_comm_buf=1,
        input_dtype=ctx.attn_out.dtype,
        output_dtype=ctx.attn_out.dtype,
        a2a_only=True,
        fuse_sync=True,
        use_persistent=use_persistent,
    )
    nvshmem_barrier_all_on_stream()
    output = torch.empty((1, ctx.local_m, ctx.n), dtype=ctx.attn_out.dtype, device=ctx.attn_out.device)
    return TritonDistA2AGemm(
        op=op,
        inputs=ctx.attn_out.view(1, ctx.seq_len, ctx.q_heads // ctx.world_size, ctx.head_dim),
        weight=ctx.b,
        output=output,
        num_comm_sms=num_comm_sms,
        sm_margin=sm_margin,
    )


def _torch_reference(ctx: A2AGemm) -> torch.Tensor:
    dist.all_to_all_single(ctx.torch_recv, ctx.torch_send)
    ctx.torch_gathered.view(ctx.local_m, ctx.world_size, ctx.band_features).copy_(ctx.torch_recv.transpose(0, 1))
    torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ctx.torch_out)
    return ctx.torch_out


def _torch_max_perf(ctx: A2AGemm) -> torch.Tensor:
    dist.all_to_all_single(ctx.torch_recv, ctx.torch_send)
    torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ctx.torch_out)
    return ctx.torch_out


def _barrier_all(device_index: int) -> None:
    try:
        from triton_dist.utils import nvshmem_barrier_all_on_stream
        nvshmem_barrier_all_on_stream()
    except Exception:
        torch.cuda.synchronize()
        dist.barrier(device_ids=[device_index])


def _assert_close_distributed(ctx: A2AGemm, actual: torch.Tensor, expected: torch.Tensor, *, label: str) -> None:
    if actual.ndim == expected.ndim + 1 and actual.shape[0] == 1:
        actual = actual[0]
    if actual.shape != expected.shape:
        raise AssertionError(f"{label} shape mismatch: {actual.shape} vs {expected.shape}")
    close = torch.allclose(actual.float(), expected.float(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
    finite = torch.isfinite(actual).all().item()
    failed = torch.tensor(0 if close and finite else 1, dtype=torch.int32, device=actual.device)
    max_error = (actual.float() - expected.float()).abs().max()
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    if failed.item():
        raise AssertionError(f"{label} mismatch across ranks; max_abs_error={max_error.item():.6g}")
    if ctx.rank == 0:
        print(f"{label} correctness: pass")


def _join(ctx: A2AGemm) -> None:
    current = torch.cuda.current_stream()
    current.wait_stream(ctx.comm_stream)
    current.wait_stream(ctx.compute_stream)
    current.synchronize()


def _launch_split(ctx: A2AGemm, *, overlap: bool) -> None:
    current = torch.cuda.current_stream()
    epoch = ctx.next_epoch()
    if overlap:
        ctx.comm_stream.wait_stream(current)
        ctx.compute_stream.wait_stream(current)
        ctx.launch_a2a(epoch)
        ctx.launch_gemm(epoch)
        _join(ctx)
    else:
        ctx.launch_a2a(epoch, standalone=True, stream=current)
        ctx.launch_gemm(epoch, standalone=True, stream=current)
        current.synchronize()


def check_correctness(ctx: A2AGemm, fused: A2AGemmFuse, triton_dist_target: TritonDistA2AGemm | None) -> None:
    for iteration in range(2):
        if iteration:
            ctx.attn_out.add_(0.01)
        reference = _torch_reference(ctx).clone()

        ctx.workspace.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier(device_ids=[ctx.attn_out.device.index])
        ctx.standalone_a2a.launch(ctx.next_epoch(), stream=torch.cuda.current_stream())
        # Local synchronize only waits for this rank. Workspace still has
        # remote TMA stores in flight until every rank finishes the push.
        _barrier_all(ctx.attn_out.device.index)
        _assert_close_distributed(ctx, ctx.workspace, ctx.torch_gathered, label="littlekernel_tma_a2a")

        ctx.workspace.fill_(float("nan"))
        ctx.out.fill_(float("nan"))
        # The NaN fill must be globally visible before any rank's push lands in
        # this rank's workspace; otherwise a fast rank's push is overwritten.
        torch.cuda.synchronize()
        dist.barrier(device_ids=[ctx.attn_out.device.index])
        _launch_split(ctx, overlap=False)
        _assert_close_distributed(ctx, ctx.out, reference, label="littlekernel_sequential")

        ctx.workspace.fill_(float("nan"))
        ctx.out.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier(device_ids=[ctx.attn_out.device.index])
        _launch_split(ctx, overlap=True)
        _assert_close_distributed(ctx, ctx.out, reference, label="littlekernel_stream_overlap")

        ctx.workspace.fill_(float("nan"))
        fused.out.fill_(float("nan"))
        torch.cuda.synchronize()
        dist.barrier(device_ids=[ctx.attn_out.device.index])
        fused.launch(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        _assert_close_distributed(ctx, fused.out, reference, label="littlekernel_role_fused")

        if triton_dist_target is not None:
            triton_dist_target.output.fill_(float("nan"))
            triton_dist_target.launch(torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert triton_dist_target.out is not None
            _assert_close_distributed(ctx, triton_dist_target.out, reference, label="triton_dist")


def _time_once(ctx: A2AGemm, mode: str) -> float:
    current = torch.cuda.current_stream()
    if mode == "gemm":
        # A2A-only timing overwrites the normal epoch tokens. Restore a
        # persistent ready value before starting the GEMM timer.
        ctx.ready.fill_(GEMM_ONLY_EPOCH)
        current.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(current)

    if mode == "torch_a2a":
        dist.all_to_all_single(ctx.torch_recv, ctx.torch_send)
    elif mode == "torch_gemm":
        torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ctx.torch_out)
    elif mode == "torch_sequential":
        _torch_max_perf(ctx)
    elif mode == "a2a":
        ctx.launch_a2a(ctx.next_epoch(), standalone=True, stream=current)
    elif mode == "gemm":
        ctx.launch_gemm(GEMM_ONLY_EPOCH, standalone=True, stream=current)
    else:
        epoch = ctx.next_epoch()
        if mode == "stream_overlap":
            ctx.comm_stream.wait_event(start)
            ctx.compute_stream.wait_event(start)
            ctx.launch_a2a(epoch)
            ctx.launch_gemm(epoch)
            comm_done = torch.cuda.Event()
            gemm_done = torch.cuda.Event()
            comm_done.record(ctx.comm_stream)
            gemm_done.record(ctx.compute_stream)
            current.wait_event(comm_done)
            current.wait_event(gemm_done)
        elif mode == "sequential":
            ctx.launch_a2a(epoch, standalone=True, stream=current)
            ctx.launch_gemm(epoch, standalone=True, stream=current)
        else:
            raise ValueError(f"unknown timing mode: {mode}")

    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _time_launch_once(launch: Callable[..., None]) -> float:
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(current)
    launch(stream=current)
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _bench(ctx: A2AGemm, once: Callable[[], float], *, warmup: int, iters: int) -> float:
    device_id = ctx.attn_out.device.index
    torch.cuda.synchronize(device_id)
    dist.barrier(device_ids=[device_id])
    time.sleep(REFRESH_SECONDS)
    dist.barrier(device_ids=[device_id])
    once()
    for _ in range(warmup):
        dist.barrier(device_ids=[device_id])
        once()
    samples = []
    for _ in range(iters):
        dist.barrier(device_ids=[device_id])
        samples.append(once())
    ms = torch.tensor(samples, dtype=torch.float64, device=ctx.attn_out.device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return statistics.median(ms.tolist())


def bench_compare(
    ctx: A2AGemm,
    *,
    warmup: int,
    iters: int,
    fused: A2AGemmFuse,
    triton_dist_target: TritonDistA2AGemm | None,
    modes: list[str] | None = None,
) -> None:
    _torch_reference(ctx)
    # Deliberate upper bound: the gather layout conversion happens once outside
    # timing. The timed A2A and GEMM have stream ordering but no data
    # dependency through torch_gathered.
    ctx.workspace.copy_(ctx.torch_gathered)
    ctx.ready.fill_(GEMM_ONLY_EPOCH)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[ctx.attn_out.device.index])

    funcs: dict[str, Callable[[], float]] = {
        "torch_a2a": partial(_time_once, ctx, "torch_a2a"),
        "torch_gemm": partial(_time_once, ctx, "torch_gemm"),
        "torch_sequential": partial(_time_once, ctx, "torch_sequential"),
        "a2a": partial(_time_once, ctx, "a2a"),
        "gemm": partial(_time_once, ctx, "gemm"),
        "sequential": partial(_time_once, ctx, "sequential"),
        "stream_overlap": partial(_time_once, ctx, "stream_overlap"),
        "role_fused": partial(_time_launch_once, fused.launch),
    }
    if triton_dist_target is not None:
        funcs["triton_dist"] = partial(_time_launch_once, triton_dist_target.launch)
    if modes is not None:
        funcs = {name: fn for name, fn in funcs.items() if name in modes}

    times = {name: _bench(ctx, fn, warmup=warmup, iters=iters) for name, fn in funcs.items()}
    if ctx.rank != 0:
        return

    flops = 2.0 * ctx.local_m * ctx.n * ctx.k
    remote_bytes = ctx.local_m * ctx.band_features * ctx.attn_out.element_size() * (ctx.world_size - 1)
    rows = [
        ReportRow("torch_a2a(no_pack):", "torch_a2a", "busbw", "bus"),
        ReportRow("ours_a2a(tma):", "a2a", "busbw", "bus"),
        ReportRow("torch_gemm:", "torch_gemm", "tflops", "compute"),
        ReportRow("ours_gemm:", "gemm", "tflops", "compute"),
        ReportRow("torch_sequential:", "torch_sequential", "tflops", "sum", "(sum bound)"),
        ReportRow("ours_sequential:", "sequential", "tflops", "sum", "(sum bound)"),
        ReportRow("stream_overlap:", "stream_overlap", "tflops", "max", "(max bound)"),
        ReportRow("role_fused:", "role_fused", "tflops", "max", "(max bound)"),
        ReportRow("triton_dist:", "triton_dist", "tflops", "max", "(max bound)"),
    ]
    print_benchmark_report(
        title=(f"A2A-GEMM(o-proj) seq={ctx.seq_len} local_seq={ctx.local_m} hidden={ctx.n} "
               f"q_heads={ctx.q_heads} head_dim={ctx.head_dim} ranks={ctx.world_size}"),
        times=times,
        rows=rows,
        flops=flops,
        comm_bytes=remote_bytes,
        device=ctx.attn_out.device,
        comm_label="remote payload",
    )

    print_speedups(
        times,
        [
            SpeedupRow("stream_speedup:", "torch_sequential", "stream_overlap", "(torch_sequential / stream_overlap)"),
            SpeedupRow("fused_speedup:", "torch_sequential", "role_fused", "(torch_sequential / role_fused)"),
            SpeedupRow("td_speedup:", "torch_sequential", "triton_dist", "(torch_sequential / triton_dist)"),
        ],
    )


def export_trace(
    ctx: A2AGemm,
    fused: A2AGemmFuse,
    triton_dist_target: TritonDistA2AGemm | None,
    trace_dir: str | Path,
    *,
    merge: bool,
    trace_iters: int,
    with_stack: bool,
) -> None:
    from torch.profiler import ProfilerActivity, profile, record_function

    _torch_reference(ctx)
    funcs: dict[str, Callable[[], None]] = {
        "torch_sequential": lambda: _time_once(ctx, "torch_sequential"),
        "ours_sequential": lambda: _time_once(ctx, "sequential"),
        "ours_stream_overlap": lambda: _time_once(ctx, "stream_overlap"),
        "ours_role_fused": lambda: _time_launch_once(fused.launch),
    }
    if triton_dist_target is not None:
        funcs["triton_dist"] = lambda: _time_launch_once(triton_dist_target.launch)

    for fn in funcs.values():
        dist.barrier(device_ids=[ctx.attn_out.device.index])
        fn()

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
    )
    dist.barrier(device_ids=[ctx.attn_out.device.index])
    prof.start()
    dist.barrier(device_ids=[ctx.attn_out.device.index])
    with record_function("trace_warmup"):
        funcs["torch_sequential"]()
    for label, fn in funcs.items():
        for _ in range(trace_iters):
            dist.barrier(device_ids=[ctx.attn_out.device.index])
            with record_function(label):
                fn()
    prof.stop()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"a2a_gemm_s{ctx.seq_len}_h{ctx.n}_q{ctx.q_heads}_d{ctx.head_dim}_r{ctx.world_size}"
    rank_trace = trace_dir / f"{stem}_rank{ctx.rank}.json.gz"
    prof.export_chrome_trace(str(rank_trace))
    dist.barrier(device_ids=[ctx.attn_out.device.index])
    if merge and ctx.rank == 0:
        rank_traces = [trace_dir / f"{stem}_rank{rank}.json.gz" for rank in range(ctx.world_size)]
        print(f"merged trace: {merge_rank_traces(rank_traces, trace_dir / f'{stem}_merged.json.gz')}")
    elif ctx.rank == 0:
        print(f"rank traces: {trace_dir / f'{stem}_rank*.json.gz'}")
    dist.barrier(device_ids=[ctx.attn_out.device.index])


def _release_symmetric_memory(ctx: A2AGemm) -> None:
    workspace_handle, ready_handle = ctx.workspace_handle, ctx.ready_handle
    workspace, ready = ctx.workspace, ctx.ready
    ctx.workspace_handle = ctx.ready_handle = None
    ctx.workspace = ctx.ready = None
    del workspace_handle, ready_handle, workspace, ready
    gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark SM100 Ulysses post-attention A2A plus o-proj GEMM")
    p.add_argument("--seq-len", type=int, default=65536, help="global sequence length")
    p.add_argument("--hidden", type=int, default=6144, help="o-proj output features")
    p.add_argument("--q-heads", type=int, default=128)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--comm-ctas", type=int, default=32)
    p.add_argument(
        "--comm-reserve-ctas",
        type=int,
        default=None,
        help="GEMM CTAs reserved for communication; defaults to --comm-ctas",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--modes", nargs="*", default=None, help="restrict timed modes (debug tuning)")
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_a2a_gemm")
    p.add_argument("--no-check", action="store_true")
    p.add_argument(
        "--triton-dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark sp_ulysess_o_all2all_gemm.py (default: enabled)",
    )
    p.add_argument("--triton-dist-comm-sms", type=int, default=32)
    p.add_argument("--triton-dist-sm-margin", type=int, default=0)
    p.add_argument("--triton-dist-persistent", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-dir", default=f"{os.path.dirname(__file__)}/lk_trace_role_v2_a2a_gemm")
    p.add_argument("--trace-iters", type=int, default=5)
    p.add_argument("--trace-with-stack", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--merge-trace", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.warmup < 0 or args.iters <= 0:
        p.error("--warmup must be non-negative and --iters must be positive")
    if args.trace_iters <= 0:
        p.error("--trace-iters must be positive")

    finalize_triton_dist = None
    world_group = None
    if args.triton_dist:
        try:
            from triton_dist.utils import finalize_distributed, initialize_distributed
        except ImportError as exc:
            raise RuntimeError("Triton-dist is required by default; install it or pass --no-triton-dist") from exc
        world_group = initialize_distributed()
        finalize_triton_dist = finalize_distributed
    else:
        dist.init_process_group("nccl")
    if world_group is None:
        world_group = dist.group.WORLD

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    if world_size != int(os.environ.get("LOCAL_WORLD_SIZE", world_size)):
        raise ValueError("this benchmark currently supports single-node execution only")

    gemm_kernel = build_gemm_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    a2a_kernel = TmaPushA2A.build(cache_dir=args.cache_dir, verbose=args.verbose)
    fused_kernel = A2AGemmFuse.build(cache_dir=args.cache_dir, verbose=args.verbose)
    kernels = (gemm_kernel, a2a_kernel, fused_kernel)

    def touch_launchers() -> None:
        for kernel in kernels:
            _ = kernel.launcher

    if rank == 0:
        touch_launchers()
    dist.barrier(device_ids=[local_rank])
    if rank != 0:
        touch_launchers()
    dist.barrier(device_ids=[local_rank])

    ctx = prepare(
        gemm_kernel,
        a2a_kernel,
        seq_len=args.seq_len,
        hidden=args.hidden,
        q_heads=args.q_heads,
        head_dim=args.head_dim,
        comm_ctas=args.comm_ctas,
        comm_reserve_ctas=args.comm_reserve_ctas,
    )
    fused = prepare_fused(ctx, fused_kernel)
    td_target = (prepare_triton_dist(
        ctx,
        world_group,
        num_comm_sms=args.triton_dist_comm_sms,
        sm_margin=args.triton_dist_sm_margin,
        use_persistent=args.triton_dist_persistent,
    ) if args.triton_dist else None)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print(f"cache_dir: {args.cache_dir}")
        print(f"a2a_ctas: standalone={ctx.standalone_a2a.num_ctas}, overlap={ctx.a2a.num_ctas}, "
              f"reserved={ctx.reserved_comm_ctas}; "
              f"compute_ctas: standalone={ctx.full_compute_ctas}, overlap={ctx.compute_ctas}; "
              f"fused_ctas={fused.num_ctas} (comm={fused.comm_ctas})")

    if not args.no_check:
        check_correctness(ctx, fused, td_target)
    bench_compare(
        ctx,
        warmup=args.warmup,
        iters=args.iters,
        fused=fused,
        triton_dist_target=td_target,
        modes=args.modes,
    )
    if args.trace:
        export_trace(
            ctx,
            fused,
            td_target,
            args.trace_dir,
            merge=args.merge_trace,
            trace_iters=args.trace_iters,
            with_stack=args.trace_with_stack,
        )

    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    if td_target is not None:
        td_target.release()
    fused.release()
    ctx.a2a.release()
    ctx.standalone_a2a.release()
    del td_target, fused
    gc.collect()
    _release_symmetric_memory(ctx)
    del ctx
    gc.collect()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    if finalize_triton_dist is not None:
        finalize_triton_dist()
    else:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
