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
"""Build, check, and benchmark split and fused SM100 GEMM reduce-scatter."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT, role_example_path, role_example_source

import argparse
import gc
import importlib.util
import os
import statistics
import sysconfig
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
from examples.nv.sm100.gemm_rs_fused_sm100 import GemmRSFuseV2
from examples.nv.sm100.rs_tma_pull_sm100 import TmaPullReduceScatter
from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

GEMM_RS_SOURCE = role_example_source("gemm_rs")
DEFAULT_THUNDERKITTENS_ROOT = Path(os.environ.get("THUNDERKITTENS_ROOT", str(REPO_ROOT.parent / "ThunderKittens")))
RS_ONLY_EPOCH = (1 << 30) - 1
CHECK_RTOL = 6e-2
CHECK_ATOL = 6e-2
REFRESH_SECONDS = 0.4


def _load_thunderkittens_extension(root: Path) -> object:
    op_dir = root.expanduser().resolve() / "kernels" / "parallel" / "gemm_rs"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("could not determine the Python extension suffix")
    extension_path = op_dir / f"_C{extension_suffix}"
    build_hint = f"ARCH=SM100 make -C {op_dir}"
    if not extension_path.is_file():
        raise RuntimeError(f"ThunderKittens GEMM-RS extension not found at {extension_path}. "
                           f"Build it with the active Python/PyTorch environment: {build_hint}")

    spec = importlib.util.spec_from_file_location("little_kernel_thunderkittens_gemm_rs._C", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create an import spec for {extension_path}")
    try:
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"failed to load {extension_path}; rebuild it against the active "
                           f"Python/PyTorch environment with: {build_hint}") from exc
    for symbol in ("TKParallelTensor", "matmul_reduce_scatter"):
        if not hasattr(extension, symbol):
            raise RuntimeError(f"{extension_path} does not export {symbol}")
    return extension


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(GEMM_RS_SOURCE, "<role-v2-gemm-rs>", "exec"), ns)
    return ns


def build_gemm_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    c = _cfg()
    return CompiledKernel(
        compile_dataflow_cuda(GEMM_RS_SOURCE),
        "gemm_rs_sm100_kernel",
        grid=(1, 1, 1),
        block=(c["NUM_THREADS"], 1, 1),
        shared_mem_bytes=c["SHARED_MEM_BYTES"],
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c["gemm_rs_sm100_kernel"].cluster_shape,
        cta_group_shape=c["gemm_rs_sm100_kernel"].cta_group_shape,
        allow_non_portable_cluster=True,
    )


def _gemm_descriptors(a: torch.Tensor, b: torch.Tensor, partial_c: torch.Tensor):
    c = _cfg()
    m, k = a.shape
    n = b.shape[0]
    d_a_standalone = create_tma_2d_descriptor(
        a,
        gmem_inner_dim=k,
        gmem_outer_dim=m,
        smem_inner_dim=c["K_TILE"],
        smem_outer_dim=c["CTA_TILE_M"],
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    d_a_fused_cta256 = create_tma_2d_descriptor(
        a,
        gmem_inner_dim=k,
        gmem_outer_dim=m,
        smem_inner_dim=GemmRSFuseV2.K_TILE,
        smem_outer_dim=GemmRSFuseV2.CTA_TILE_M,
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    d_b_standalone = create_tma_2d_descriptor(
        b,
        gmem_inner_dim=k,
        gmem_outer_dim=n,
        smem_inner_dim=c["K_TILE"],
        smem_outer_dim=c["CTA_TILE_N_SHARD"],
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    d_b_fused_n128 = create_tma_2d_descriptor(
        b,
        gmem_inner_dim=k,
        gmem_outer_dim=n,
        smem_inner_dim=GemmRSFuseV2.K_TILE,
        smem_outer_dim=GemmRSFuseV2.CTA_TILE_N_SHARD,
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    d_partial_standalone = create_tma_2d_descriptor(
        partial_c,
        gmem_inner_dim=n,
        gmem_outer_dim=m,
        smem_inner_dim=c["ST_TILE_N"],
        smem_outer_dim=c["ST_TILE_M"],
        gmem_outer_stride=n,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )
    return (
        d_a_standalone,
        d_b_standalone,
        d_partial_standalone,
        d_a_fused_cta256,
        d_b_fused_n128,
    )


@dataclass
class GemmRS:
    gemm_kernel: CompiledKernel
    a: torch.Tensor
    b: torch.Tensor
    partial_c: torch.Tensor
    torch_partial: torch.Tensor
    torch_out: torch.Tensor
    ready: torch.Tensor
    standalone_counters: torch.Tensor
    d_a_standalone: object
    d_b_standalone: object
    d_partial_standalone: object
    d_a_fused_cta256: object
    d_b_fused_n128: object
    rs: TmaPullReduceScatter
    standalone_rs: TmaPullReduceScatter
    rank: int
    world_size: int
    m: int
    n: int
    k: int
    k_local: int
    m_per_rank: int
    compute_ctas: int
    full_compute_ctas: int
    workspace_handle: object
    ready_handle: object
    comm_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    epoch: int = 0

    def next_epoch(self) -> int:
        self.epoch = self.epoch % ((1 << 30) - 2) + 1
        return self.epoch

    def launch_gemm(self, epoch: int, *, standalone: bool = False) -> None:
        num_ctas = self.full_compute_ctas if standalone else self.compute_ctas
        self.gemm_kernel(
            self.d_a_standalone,
            self.d_b_standalone,
            self.d_partial_standalone,
            self.ready,
            self.standalone_counters,
            self.m,
            self.n,
            self.k_local,
            self.world_size,
            epoch,
            num_ctas,
            grid=(num_ctas, 1, 1),
            stream=self.compute_stream,
        )

    def launch_rs(self, epoch: int, *, standalone: bool = False) -> None:
        target = self.standalone_rs if standalone else self.rs
        target.launch(epoch, stream=self.comm_stream)


@dataclass
class TritonDistGemmRS:
    op: object
    context: object
    a: torch.Tensor
    b: torch.Tensor
    autotune: bool
    reduce_st: bool = False
    gemm_config: object | None = None
    out: torch.Tensor | None = None

    def launch(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            if self.reduce_st:
                self.out = self.op.fn(
                    self.a,
                    self.b,
                    self.context,
                    gemm_config=self.gemm_config,
                    persistent=True,
                    fuse_scatter=True,
                    reduce_st=True,
                )
            elif self.autotune:
                self.out = self.op(self.a, self.b, self.context, autotune=True)
            else:
                self.out = self.op.fn(
                    self.a,
                    self.b,
                    self.context,
                    gemm_config=self.gemm_config,
                    persistent=True,
                    fuse_scatter=True,
                    reduce_st=False,
                )

    def release(self) -> None:
        rs_ctx = getattr(self.context, "rs_ctx", None)
        barrier = getattr(rs_ctx, "barrier", None)
        self.context.finalize()
        # The TritonDist revision paired with this benchmark does not release
        # ReduceScatter2DContext's nested BarrierAllContext.
        if barrier is not None and getattr(barrier, "symm_barrier", None) is not None:
            barrier.finalize()
            barrier.symm_barrier = None
        self.context = self.a = self.b = self.out = self.gemm_config = None


@dataclass
class ThunderKittensGemmRS:
    extension: object
    op: object
    a: torch.Tensor
    b: torch.Tensor
    parallel_out: object
    out: torch.Tensor
    barrier: object

    def prepare_output(self) -> None:
        self.out.zero_()
        torch.cuda.current_stream().synchronize()
        dist.barrier(device_ids=[self.out.device.index])

    def launch(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            self.op(self.a, self.b, self.parallel_out, self.barrier)

    def release(self) -> None:
        self.out = None
        self.parallel_out = None
        gc.collect()
        self.barrier = None
        gc.collect()
        self.extension = self.op = self.a = self.b = None


def prepare(
    gemm_kernel: CompiledKernel,
    rs_kernel: CompiledKernel,
    *,
    m: int,
    n: int,
    k: int,
    comm_ctas: int,
) -> GemmRS:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    c = _cfg()

    if m % world_size or k % world_size:
        raise ValueError("M and global K must be divisible by world_size")
    m_per_rank, k_local = m // world_size, k // world_size
    if m_per_rank % c["PAIR_TILE_M"]:
        raise ValueError(f"M/world_size must be divisible by PAIR_TILE_M={c['PAIR_TILE_M']}")
    if n % c["PAIR_TILE_N"]:
        raise ValueError(f"N must be divisible by PAIR_TILE_N={c['PAIR_TILE_N']}")
    if k_local % c["K_TILE"]:
        raise ValueError(f"K/world_size must be divisible by K_TILE={c['K_TILE']}")
    gemm_tiles = (m // c["PAIR_TILE_M"]) * (n // c["PAIR_TILE_N"])
    if gemm_tiles < 2:
        raise ValueError("GEMM-RS requires at least two output tiles")

    gen = torch.Generator(device=device)
    gen.manual_seed(2026 + rank + m + n + k)
    a = torch.randn((m, k_local), dtype=torch.bfloat16, device=device, generator=gen)
    b = torch.randn((n, k_local), dtype=torch.bfloat16, device=device, generator=gen)
    a.mul_(0.01 * (rank + 1))

    partial_c = symm_mem.empty((m, n), dtype=torch.bfloat16, device=device)
    workspace_handle = symm_mem.rendezvous(partial_c, group=dist.group.WORLD)
    partial_c.zero_()

    ready = symm_mem.empty((gemm_tiles, ), dtype=torch.int32, device=device)
    ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
    ready.zero_()
    standalone_counters = torch.zeros_like(ready)

    (d_a_standalone, d_b_standalone, d_partial_standalone, d_a_fused_cta256,
     d_b_fused_n128) = _gemm_descriptors(a, b, partial_c)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rs = TmaPullReduceScatter.create(
        rs_kernel,
        workspace_handle=workspace_handle,
        partial=partial_c,
        ready_handle=ready_handle,
        ready=ready,
        rank=rank,
        world_size=world_size,
        comm_ctas=comm_ctas,
    )
    standalone_rs = TmaPullReduceScatter.create(
        rs_kernel,
        workspace_handle=workspace_handle,
        partial=partial_c,
        ready_handle=ready_handle,
        ready=ready,
        rank=rank,
        world_size=world_size,
        comm_ctas=sms,
    )

    reserved = max(1, min(rs.num_ctas, sms - c["CLUSTER_SIZE"]))
    compute_sms = max(c["CLUSTER_SIZE"], sms - reserved)
    compute_ctas = (min(compute_sms, gemm_tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    full_compute_ctas = (min(sms, gemm_tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    if compute_ctas == 0:
        raise ValueError("problem is too small for a two-CTA GEMM cluster")

    return GemmRS(
        gemm_kernel=gemm_kernel,
        a=a,
        b=b,
        partial_c=partial_c,
        torch_partial=torch.empty_like(partial_c),
        torch_out=torch.empty((m_per_rank, n), dtype=torch.bfloat16, device=device),
        ready=ready,
        standalone_counters=standalone_counters,
        d_a_standalone=d_a_standalone,
        d_b_standalone=d_b_standalone,
        d_partial_standalone=d_partial_standalone,
        d_a_fused_cta256=d_a_fused_cta256,
        d_b_fused_n128=d_b_fused_n128,
        rs=rs,
        standalone_rs=standalone_rs,
        rank=rank,
        world_size=world_size,
        m=m,
        n=n,
        k=k,
        k_local=k_local,
        m_per_rank=m_per_rank,
        compute_ctas=compute_ctas,
        full_compute_ctas=full_compute_ctas,
        workspace_handle=workspace_handle,
        ready_handle=ready_handle,
        comm_stream=torch.cuda.Stream(priority=-1),
        compute_stream=torch.cuda.Stream(),
    )


def prepare_fused(ctx: GemmRS, kernel: CompiledKernel, *, comm_ctas: int | None) -> GemmRSFuseV2:
    return GemmRSFuseV2.create(
        kernel,
        workspace_handle=ctx.workspace_handle,
        partial=ctx.partial_c,
        ready_handle=ctx.ready_handle,
        ready_local=ctx.ready,
        dA=ctx.d_a_fused_cta256,
        dB=ctx.d_b_fused_n128,
        m=ctx.m,
        n=ctx.n,
        k=ctx.k_local,
        rank=ctx.rank,
        world_size=ctx.world_size,
        comm_ctas=comm_ctas,
    )


def prepare_triton_dist(ctx: GemmRS, *, autotune: bool, reduce_st: bool = False) -> TritonDistGemmRS:
    try:
        from triton_dist.kernels.nvidia import create_gemm_rs_context, gemm_rs
        from triton_dist.kernels.nvidia.gemm import get_config_space
    except ImportError as exc:
        raise RuntimeError("Triton-distributed is required by default; install it or pass --no-triton-dist") from exc

    rs_stream = torch.cuda.Stream(priority=-1)
    td_context = create_gemm_rs_context(
        ctx.m,
        ctx.n,
        ctx.rank,
        ctx.world_size,
        ctx.world_size,
        ctx.a.dtype,
        rs_stream,
        reduce_st=reduce_st,
    )
    gemm_config = None if autotune and not reduce_st else get_config_space(True)[0]
    return TritonDistGemmRS(
        op=gemm_rs,
        context=td_context,
        a=ctx.a,
        b=ctx.b.t(),
        autotune=autotune and not reduce_st,
        reduce_st=reduce_st,
        gemm_config=gemm_config,
    )


def prepare_thunderkittens(ctx: GemmRS, *, root: Path) -> ThunderKittensGemmRS:
    if ctx.m % 512 or ctx.m_per_rank % 128:
        raise ValueError("ThunderKittens GEMM-RS requires M divisible by 512 and M/world_size divisible by 128")
    if ctx.n % 256:
        raise ValueError("ThunderKittens GEMM-RS requires N divisible by 256")
    if ctx.k_local % 64:
        raise ValueError("ThunderKittens GEMM-RS requires K/world_size divisible by 64")

    extension = _load_thunderkittens_extension(root)
    parallel_out = extension.TKParallelTensor((ctx.m_per_rank, ctx.n), ctx.a.dtype, ctx.rank, ctx.world_size, False)
    barrier = extension.TKParallelTensor((1, 1), torch.int32, ctx.rank, ctx.world_size, True)
    out = parallel_out.data_
    out.zero_()
    barrier.data_.zero_()
    return ThunderKittensGemmRS(
        extension=extension,
        op=extension.matmul_reduce_scatter,
        a=ctx.a,
        b=ctx.b,
        parallel_out=parallel_out,
        out=out,
        barrier=barrier,
    )


def _join(ctx: GemmRS) -> None:
    current = torch.cuda.current_stream()
    current.wait_stream(ctx.comm_stream)
    current.wait_stream(ctx.compute_stream)
    current.synchronize()


def _torch_reference(ctx: GemmRS) -> torch.Tensor:
    torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    dist.reduce_scatter_tensor(ctx.torch_out, ctx.torch_partial)
    return ctx.torch_out


def _assert_close_distributed(ctx: GemmRS, actual: torch.Tensor, expected: torch.Tensor, *, label: str) -> None:
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


def _launch_split(ctx: GemmRS, *, overlap: bool) -> None:
    current = torch.cuda.current_stream()
    ctx.comm_stream.wait_stream(current)
    ctx.compute_stream.wait_stream(current)
    epoch = ctx.next_epoch()
    if overlap:
        ctx.launch_rs(epoch)
        ctx.launch_gemm(epoch)
    else:
        ctx.launch_gemm(epoch, standalone=True)
        gemm_done = torch.cuda.Event()
        gemm_done.record(ctx.compute_stream)
        ctx.comm_stream.wait_event(gemm_done)
        ctx.launch_rs(epoch, standalone=True)
    _join(ctx)


def check_correctness(
    ctx: GemmRS,
    fused: GemmRSFuseV2,
    triton_dist_target: TritonDistGemmRS | None,
    triton_dist_reduce_st_target: TritonDistGemmRS | None,
    thunderkittens_target: ThunderKittensGemmRS | None,
) -> None:
    for iteration in range(2):
        if iteration:
            ctx.a.add_(0.01)
        reference = _torch_reference(ctx).clone()

        # Isolate the standalone communication kernel from the LK GEMM.
        ctx.partial_c.copy_(ctx.torch_partial)
        rs_epoch = ctx.next_epoch()
        ctx.ready.fill_(rs_epoch)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[ctx.a.device.index])
        ctx.standalone_rs.out.fill_(float("nan"))
        ctx.standalone_rs.launch(rs_epoch, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        _assert_close_distributed(ctx, ctx.standalone_rs.out, reference, label="tma_rs")

        ctx.partial_c.fill_(float("nan"))
        ctx.standalone_rs.out.fill_(float("nan"))
        _launch_split(ctx, overlap=False)
        _assert_close_distributed(ctx, ctx.standalone_rs.out, reference, label="littlekernel_sequential")

        ctx.partial_c.fill_(float("nan"))
        ctx.rs.out.fill_(float("nan"))
        _launch_split(ctx, overlap=True)
        _assert_close_distributed(ctx, ctx.rs.out, reference, label="stream_overlap")

        ctx.partial_c.fill_(float("nan"))
        fused.out.fill_(float("nan"))
        fused.launch(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        _assert_close_distributed(ctx, fused.out, reference, label="fuse_overlap")

        if thunderkittens_target is not None:
            thunderkittens_target.prepare_output()
            thunderkittens_target.launch(torch.cuda.current_stream())
            torch.cuda.synchronize()
            _assert_close_distributed(ctx, thunderkittens_target.out, reference, label="thunderkittens")

        triton_dist_targets = (
            ("triton_dist", triton_dist_target),
            ("triton_dist_reduce_st", triton_dist_reduce_st_target),
        )
        for label, target in triton_dist_targets:
            if target is None:
                continue
            target.launch(torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert target.out is not None
            _assert_close_distributed(ctx, target.out, reference, label=label)


def _time_once(ctx: GemmRS, mode: str) -> float:
    current = torch.cuda.current_stream()
    if mode == "rs":
        # GEMM-only timing overwrites the normal epoch tokens. Restore a
        # persistent ready value before starting the communication timer.
        ctx.ready.fill_(RS_ONLY_EPOCH)
        current.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(current)

    if mode == "torch_gemm":
        torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    elif mode == "torch_rs":
        dist.reduce_scatter_tensor(ctx.torch_out, ctx.torch_partial)
    elif mode == "torch":
        torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
        dist.reduce_scatter_tensor(ctx.torch_out, ctx.torch_partial)
    elif mode == "gemm":
        epoch = ctx.next_epoch()
        ctx.compute_stream.wait_event(start)
        ctx.launch_gemm(epoch, standalone=True)
        done = torch.cuda.Event()
        done.record(ctx.compute_stream)
        current.wait_event(done)
    elif mode == "rs":
        ctx.comm_stream.wait_event(start)
        ctx.launch_rs(RS_ONLY_EPOCH, standalone=True)
        done = torch.cuda.Event()
        done.record(ctx.comm_stream)
        current.wait_event(done)
    else:
        epoch = ctx.next_epoch()
        ctx.comm_stream.wait_event(start)
        ctx.compute_stream.wait_event(start)
        if mode == "stream_overlap":
            ctx.launch_rs(epoch)
            ctx.launch_gemm(epoch)
        elif mode == "sequential":
            ctx.launch_gemm(epoch, standalone=True)
            gemm_done = torch.cuda.Event()
            gemm_done.record(ctx.compute_stream)
            ctx.comm_stream.wait_event(gemm_done)
            ctx.launch_rs(epoch, standalone=True)
        else:
            raise ValueError(f"unknown timing mode: {mode}")
        rs_done = torch.cuda.Event()
        gemm_done = torch.cuda.Event()
        rs_done.record(ctx.comm_stream)
        gemm_done.record(ctx.compute_stream)
        current.wait_event(rs_done)
        current.wait_event(gemm_done)

    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _time_launch_once(launch: Callable[..., None], prepare: Callable[[], None] | None = None) -> float:
    if prepare is not None:
        prepare()
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(current)
    launch(stream=current)
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def _bench(ctx: GemmRS, once: Callable[[], float], *, warmup: int, iters: int) -> float:
    device_id = ctx.a.device.index
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
    # Per-iter MAX across ranks, then median — one AR for the whole run.
    ms = torch.tensor(samples, dtype=torch.float64, device=ctx.a.device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return statistics.median(ms.tolist())


def bench_compare(
    ctx: GemmRS,
    *,
    warmup: int,
    iters: int,
    fused: GemmRSFuseV2,
    triton_dist_target: TritonDistGemmRS | None,
    triton_dist_reduce_st_target: TritonDistGemmRS | None,
    thunderkittens_target: ThunderKittensGemmRS | None,
) -> None:
    torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    ctx.partial_c.copy_(ctx.torch_partial)
    ctx.ready.fill_(RS_ONLY_EPOCH)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[ctx.a.device.index])

    funcs: dict[str, Callable[[], float]] = {
        "torch_gemm": partial(_time_once, ctx, "torch_gemm"),
        "torch_rs": partial(_time_once, ctx, "torch_rs"),
        "torch": partial(_time_once, ctx, "torch"),
        "gemm": partial(_time_once, ctx, "gemm"),
        "rs": partial(_time_once, ctx, "rs"),
        "sequential": partial(_time_once, ctx, "sequential"),
        "stream_overlap": partial(_time_once, ctx, "stream_overlap"),
        "fuse_overlap": partial(_time_launch_once, fused.launch),
    }
    if triton_dist_target is not None:
        funcs["triton_dist"] = partial(_time_launch_once, triton_dist_target.launch)
    if triton_dist_reduce_st_target is not None:
        funcs["triton_dist_reduce_st"] = partial(_time_launch_once, triton_dist_reduce_st_target.launch)
    if thunderkittens_target is not None:
        funcs["thunderkittens"] = partial(
            _time_launch_once,
            thunderkittens_target.launch,
            thunderkittens_target.prepare_output,
        )

    times = {name: _bench(ctx, fn, warmup=warmup, iters=iters) for name, fn in funcs.items()}
    if ctx.rank != 0:
        return

    flops = 2.0 * ctx.m * ctx.n * ctx.k_local
    rs_bytes = ctx.m * ctx.n * ctx.a.element_size() * (ctx.world_size - 1) / ctx.world_size
    rows = [
        ReportRow("torch_gemm:", "torch_gemm", "tflops", "compute"),
        ReportRow("ours_gemm:", "gemm", "tflops", "compute"),
        ReportRow("torch_rs:", "torch_rs", "busbw", "bus"),
        ReportRow("ours_rs(tma):", "rs", "busbw", "bus"),
        ReportRow("torch_sequential:", "torch", "tflops", "sum", "(sum bound)"),
        ReportRow("ours_sequential:", "sequential", "tflops", "sum", "(sum bound)"),
        ReportRow("stream_overlap:", "stream_overlap", "tflops", "max", "(max bound)"),
        ReportRow("fuse_overlap:", "fuse_overlap", "tflops", "max", "(max bound)"),
        ReportRow("triton_dist:", "triton_dist", "tflops", "max", "(max bound)"),
        ReportRow("triton_dist_reduce_st:", "triton_dist_reduce_st", "tflops", "max", "(max bound)"),
        ReportRow("ThunderKittens:", "thunderkittens", "tflops", "max", "(max bound)"),
    ]
    print_benchmark_report(
        title=f"GEMM-RS M={ctx.m} N={ctx.n} K={ctx.k} K/rank={ctx.k_local} ranks={ctx.world_size}",
        times=times,
        rows=rows,
        flops=flops,
        comm_bytes=rs_bytes,
        device=ctx.a.device,
    )

    print_speedups(
        times,
        [
            SpeedupRow("stream_speedup:", "torch", "stream_overlap", "(torch_sequential / stream_overlap)"),
            SpeedupRow("fuse_speedup:", "torch", "fuse_overlap", "(torch_sequential / fuse_overlap)"),
            SpeedupRow("td_speedup:", "torch", "triton_dist", "(torch_sequential / triton_dist)"),
            SpeedupRow("thunderkittens_speedup:", "torch", "thunderkittens", "(torch_sequential / ThunderKittens)"),
        ],
    )


def export_trace(
    ctx: GemmRS,
    fused: GemmRSFuseV2,
    triton_dist_target: TritonDistGemmRS | None,
    triton_dist_reduce_st_target: TritonDistGemmRS | None,
    thunderkittens_target: ThunderKittensGemmRS | None,
    trace_dir: str | Path,
    *,
    merge: bool,
    trace_iters: int,
    with_stack: bool,
) -> None:
    from torch.profiler import ProfilerActivity, profile, record_function

    funcs: dict[str, Callable[[], None]] = {
        "torch_sequential": lambda: _time_once(ctx, "torch"),
        "ours_sequential": lambda: _time_once(ctx, "sequential"),
        "ours_stream_overlap": lambda: _time_once(ctx, "stream_overlap"),
        "ours_fuse_overlap": lambda: _time_launch_once(fused.launch),
    }
    if triton_dist_target is not None:
        funcs["triton_dist"] = lambda: _time_launch_once(triton_dist_target.launch)
    if triton_dist_reduce_st_target is not None:
        funcs["triton_dist_reduce_st"] = lambda: _time_launch_once(triton_dist_reduce_st_target.launch)
    if thunderkittens_target is not None:
        funcs["thunderkittens"] = lambda: _time_launch_once(
            thunderkittens_target.launch,
            thunderkittens_target.prepare_output,
        )

    for fn in funcs.values():
        dist.barrier(device_ids=[ctx.a.device.index])
        fn()

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=with_stack,
    )
    dist.barrier(device_ids=[ctx.a.device.index])
    prof.start()
    dist.barrier(device_ids=[ctx.a.device.index])
    # Prime Kineto/CUPTI activity buffers so the first labeled iteration does
    # not absorb one-time profiler-side launch overhead.
    with record_function("trace_warmup"):
        funcs["torch_sequential"]()
    for label, fn in funcs.items():
        for _ in range(trace_iters):
            dist.barrier(device_ids=[ctx.a.device.index])
            with record_function(label):
                fn()
    prof.stop()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"gemm_rs_m{ctx.m}_n{ctx.n}_k{ctx.k}_r{ctx.world_size}"
    rank_trace = trace_dir / f"{stem}_rank{ctx.rank}.json.gz"
    prof.export_chrome_trace(str(rank_trace))
    dist.barrier(device_ids=[ctx.a.device.index])
    if merge and ctx.rank == 0:
        rank_traces = [trace_dir / f"{stem}_rank{rank}.json.gz" for rank in range(ctx.world_size)]
        print(f"merged trace: {merge_rank_traces(rank_traces, trace_dir / f'{stem}_merged.json.gz')}")
    elif ctx.rank == 0:
        print(f"rank traces: {trace_dir / f'{stem}_rank*.json.gz'}")
    dist.barrier(device_ids=[ctx.a.device.index])


def _release_symmetric_memory(ctx: GemmRS) -> None:
    workspace_handle, ready_handle = ctx.workspace_handle, ctx.ready_handle
    partial_c, ready = ctx.partial_c, ctx.ready
    ctx.workspace_handle = ctx.ready_handle = None
    ctx.partial_c = ctx.ready = None
    del workspace_handle, ready_handle, partial_c, ready
    gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark SM100 GEMM reduce-scatter")
    p.add_argument("--M", type=int, default=16384)
    p.add_argument("--N", type=int, default=16384)
    p.add_argument("--K", type=int, default=8192, help="global K; each rank owns K/world_size")
    p.add_argument("--comm-ctas", type=int, default=28)
    p.add_argument("--fuse-comm-ctas", type=int, default=None)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_gemm_rs")
    p.add_argument("--no-check", action="store_true")
    p.add_argument(
        "--triton-dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark Triton-distributed GEMM-RS (default: enabled)",
    )
    p.add_argument("--triton-dist-autotune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--triton-dist-reduce-st",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark TritonDist's persistent reduce_st GEMM-RS path",
    )
    p.add_argument(
        "--thunderkittens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark the ThunderKittens GEMM-RS baseline (default: enabled)",
    )
    p.add_argument("--thunderkittens-root", type=Path, default=DEFAULT_THUNDERKITTENS_ROOT)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-dir", default=f"{os.path.dirname(__file__)}/lk_trace_role_v2_gemm_rs")
    p.add_argument("--trace-iters", type=int, default=5)
    p.add_argument("--trace-with-stack", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--merge-trace", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.trace_iters <= 0:
        p.error("--trace-iters must be positive")

    finalize_triton_dist = None
    if args.triton_dist:
        try:
            from triton_dist.utils import finalize_distributed, initialize_distributed
        except ImportError as exc:
            raise RuntimeError(
                "Triton-distributed is required by default; install it or pass --no-triton-dist") from exc
        initialize_distributed()
        finalize_triton_dist = finalize_distributed
    else:
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    if dist.get_world_size() != int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size())):
        raise ValueError("this benchmark currently supports single-node execution only")

    gemm_kernel = build_gemm_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    rs_kernel = TmaPullReduceScatter.build(cache_dir=args.cache_dir, verbose=args.verbose)
    fused_kernel = GemmRSFuseV2.build(cache_dir=args.cache_dir, verbose=args.verbose)
    kernels = (gemm_kernel, rs_kernel, fused_kernel)

    def touch_launchers() -> None:
        for kernel in kernels:
            _ = kernel.launcher

    if rank == 0:
        touch_launchers()
    dist.barrier(device_ids=[local_rank])
    if rank != 0:
        touch_launchers()
    dist.barrier(device_ids=[local_rank])

    ctx = prepare(gemm_kernel, rs_kernel, m=args.M, n=args.N, k=args.K, comm_ctas=args.comm_ctas)
    fused = prepare_fused(ctx, fused_kernel, comm_ctas=args.fuse_comm_ctas)
    td_target = prepare_triton_dist(ctx, autotune=args.triton_dist_autotune) if args.triton_dist else None
    td_reduce_st_target = (prepare_triton_dist(
        ctx,
        autotune=False,
        reduce_st=True,
    ) if args.triton_dist and args.triton_dist_reduce_st else None)
    thunderkittens_target = (prepare_thunderkittens(ctx, root=args.thunderkittens_root)
                             if args.thunderkittens else None)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])

    if rank == 0:
        print(f"cache_dir: {args.cache_dir}")
        print(f"rs_ctas: standalone={ctx.standalone_rs.num_ctas}, "
              f"overlap={ctx.rs.num_ctas}; compute_ctas: "
              f"standalone={ctx.full_compute_ctas}, overlap={ctx.compute_ctas}; "
              f"fuse_comm_ctas: {fused.comm_ctas}")
        if thunderkittens_target is not None:
            print(f"thunderkittens_root: {args.thunderkittens_root}")

    if not args.no_check:
        check_correctness(ctx, fused, td_target, td_reduce_st_target, thunderkittens_target)
    bench_compare(
        ctx,
        warmup=args.warmup,
        iters=args.iters,
        fused=fused,
        triton_dist_target=td_target,
        triton_dist_reduce_st_target=td_reduce_st_target,
        thunderkittens_target=thunderkittens_target,
    )
    if args.trace:
        export_trace(
            ctx,
            fused,
            td_target,
            td_reduce_st_target,
            thunderkittens_target,
            args.trace_dir,
            merge=args.merge_trace,
            trace_iters=args.trace_iters,
            with_stack=args.trace_with_stack,
        )

    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    if td_target is not None:
        td_target.release()
    if td_reduce_st_target is not None:
        td_reduce_st_target.release()
    if thunderkittens_target is not None:
        thunderkittens_target.release()
    fused.release()
    ctx.rs.release()
    ctx.standalone_rs.release()
    del td_target, td_reduce_st_target, thunderkittens_target, fused
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
