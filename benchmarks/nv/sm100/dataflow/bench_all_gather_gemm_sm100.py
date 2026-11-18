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
"""Build, check, and benchmark split-path SM100 TMA AG-GEMM."""

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

from examples.nv.sm100.ag_tma_push_sm100 import CHUNK_ROWS, NUM_THREADS as AG_NUM_THREADS, SHARED_MEM_BYTES as AG_SHARED_MEM_BYTES, ag_tma_push_sm100_kernel
from examples.nv.sm100.ag_gemm_fused_sm100 import AGGemmFuse
from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
from little_kernel.core.passes import PASSES
from benchmarks.nv.sm100.dataflow.bench_report import ReportRow, SpeedupRow, print_benchmark_report, print_speedups
from tools.merge_rank_traces import merge_rank_traces
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

AG_GEMM_SOURCE = role_example_source("ag_gemm")
DEFAULT_THUNDERKITTENS_ROOT = Path(os.environ.get("THUNDERKITTENS_ROOT", str(REPO_ROOT.parent / "ThunderKittens")))
TRACE_ITERS = 5
REFRESH_SECONDS = 0.4


def _load_thunderkittens_extension(root: Path) -> object:
    op_dir = root.expanduser().resolve() / "kernels" / "parallel" / "ag_gemm"
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("could not determine the Python extension suffix")
    extension_path = op_dir / f"_C{extension_suffix}"
    build_hint = f"ARCH=SM100 make -C {op_dir}"
    if not extension_path.is_file():
        raise RuntimeError(f"ThunderKittens AG-GEMM extension not found at {extension_path}. "
                           f"Build it with the active Python/PyTorch environment: {build_hint}")

    spec = importlib.util.spec_from_file_location("little_kernel_thunderkittens_ag_gemm._C", extension_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create an import spec for {extension_path}")
    try:
        extension = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extension)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"failed to load {extension_path}; rebuild it against the active "
                           f"Python/PyTorch environment with: {build_hint}") from exc
    for symbol in ("TKParallelTensor", "all_gather_matmul"):
        if not hasattr(extension, symbol):
            raise RuntimeError(f"{extension_path} does not export {symbol}")
    return extension


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(AG_GEMM_SOURCE, "<role-v2-ag-gemm>", "exec"), ns)
    return ns


def _gemm_shared_mem_bytes() -> int:
    c = _cfg()
    return (c["NUM_STORE_STAGES"] * c["ST_TILE_M"] * c["SWIZZLE_CD_BYTES"] + c["NUM_A_STAGES"] * c["SMEM_A"] +
            c["NUM_B_STAGES"] * c["SMEM_B"] + 1024)


def build_ag_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    return ag_tma_push_sm100_kernel.build(
        passes=PASSES["cuda"],
        codegen_func=codegen_cuda,
        grid=(1, 1, 1),
        block=(AG_NUM_THREADS, 1, 1),
        shared_mem_bytes=AG_SHARED_MEM_BYTES,
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
    )


def build_gemm_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    c = _cfg()
    return CompiledKernel(
        compile_dataflow_cuda(AG_GEMM_SOURCE),
        "ag_gemm_sm100_kernel",
        grid=(1, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=_gemm_shared_mem_bytes(),
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c["ag_gemm_sm100_kernel"].cluster_shape,
        cta_group_shape=c["ag_gemm_sm100_kernel"].cta_group_shape,
        allow_non_portable_cluster=True,
    )


def _descriptors(workspace: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    c = _cfg()
    m, k = workspace.shape
    n = b.shape[0]
    dA = create_tma_2d_descriptor(
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
    dB = create_tma_2d_descriptor(
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
    dD = create_tma_2d_descriptor(
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
    return dA, dB, dD


@dataclass
class AGGemm:
    ag_kernel: CompiledKernel
    gemm_kernel: CompiledKernel
    a: torch.Tensor
    b: torch.Tensor
    out: torch.Tensor
    workspace: torch.Tensor
    torch_gathered: torch.Tensor
    ready: torch.Tensor
    workspace_ptrs: torch.Tensor
    ready_ptrs: torch.Tensor
    dA: object
    dB: object
    dD: object
    rank: int
    world_size: int
    m: int
    n: int
    k: int
    m_per_rank: int
    chunks_per_rank: int
    comm_ctas: int
    compute_ctas: int
    workspace_handle: object
    ready_handle: object
    comm_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    epoch: int = 0

    def next_epoch(self) -> int:
        self.epoch += 1
        return self.epoch

    def launch_ag(self, epoch: int) -> None:
        self.ag_kernel(
            self.a,
            self.workspace_ptrs,
            self.ready_ptrs,
            self.rank,
            self.world_size,
            self.m_per_rank,
            self.k,
            self.chunks_per_rank,
            epoch,
            grid=(self.comm_ctas, 1, 1),
            stream=self.comm_stream,
        )

    def launch_gemm(self, epoch: int) -> None:
        self.gemm_kernel(
            self.dA,
            self.dB,
            self.dD,
            self.ready,
            self.m,
            self.n,
            self.k,
            self.m_per_rank,
            self.rank,
            epoch,
            self.compute_ctas,
            grid=(self.compute_ctas, 1, 1),
            stream=self.compute_stream,
        )


@dataclass
class TritonDistAGGemm:
    op: object
    context: object
    a: torch.Tensor
    local_a: torch.Tensor
    b: torch.Tensor
    autotune: bool
    local_copy: bool
    use_cooperative: bool
    gemm_config: object | None = None
    graph: torch.cuda.CUDAGraph | None = None
    capture_stream: torch.cuda.Stream | None = None
    out: torch.Tensor | None = None

    def stage_input(self) -> None:
        if not self.local_copy:
            self.local_a.copy_(self.a)

    def launch_eager(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            if self.gemm_config is None and self.autotune:
                self.out = self.op(
                    self.a,
                    self.b,
                    ctx=self.context,
                    autotune=True,
                    local_copy=self.local_copy,
                    use_cooperative=self.use_cooperative,
                )
                key = self.op.key_fn(self.a, self.b, self.context)
                self.gemm_config = self.op.best_configs[key]["gemm_config"]
                return
            if self.gemm_config is None:
                self.gemm_config = self.op.config_space[0]["gemm_config"]
            self.out = self.op.fn(
                self.a,
                self.b,
                ctx=self.context,
                gemm_config=self.gemm_config,
                local_copy=self.local_copy,
                use_cooperative=self.use_cooperative,
            )

    def launch(self, stream: torch.cuda.Stream) -> None:
        if self.graph is None:
            self.launch_eager(stream)
            return
        with torch.cuda.stream(stream):
            self.graph.replay()

    def capture(self, stream: torch.cuda.Stream) -> None:
        if self.gemm_config is None:
            self.launch_eager(stream)
            torch.cuda.synchronize()
        self.capture_stream = torch.cuda.Stream()
        self.capture_stream.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            self.launch_eager(self.capture_stream)
        stream.wait_stream(self.capture_stream)

    def release(self) -> None:
        self.graph = None
        self.capture_stream = None
        self.context.finalize()
        self.context = self.a = self.local_a = self.b = None
        self.gemm_config = self.out = None


@dataclass
class ThunderKittensAGGemm:
    extension: object
    op: object
    parallel_a: object
    b: torch.Tensor
    out: torch.Tensor
    barrier: object
    local_a: torch.Tensor
    rank: int
    m_per_rank: int
    num_comm_sms: int

    def stage_input(self) -> None:
        begin = self.rank * self.m_per_rank
        self.parallel_a.data_[begin:begin + self.m_per_rank].copy_(self.local_a)

    def launch(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            self.op(self.parallel_a, self.b, self.out, self.barrier, self.num_comm_sms)

    def release(self) -> None:
        self.parallel_a = None
        gc.collect()
        self.barrier = None
        gc.collect()
        self.extension = self.op = self.b = self.out = self.local_a = None


def prepare(
    ag_kernel: CompiledKernel,
    gemm_kernel: CompiledKernel,
    *,
    m: int,
    n: int,
    k: int,
    comm_ctas: int,
) -> AGGemm:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    c = _cfg()

    assert m % world_size == 0 and n % world_size == 0
    m_per_rank, n_local = m // world_size, n // world_size
    assert m_per_rank % c["PAIR_TILE_M"] == 0
    assert m_per_rank % CHUNK_ROWS == 0
    assert k % c["K_TILE"] == 0
    assert n_local % c["ST_TILE_N"] == 0

    gen = torch.Generator(device=device)
    gen.manual_seed(2026 + rank + m + n + k)

    workspace = symm_mem.empty((m, k), dtype=torch.bfloat16, device=device)
    workspace_handle = symm_mem.rendezvous(workspace, group=dist.group.WORLD)
    workspace.zero_()
    a = workspace[rank * m_per_rank:(rank + 1) * m_per_rank]
    a.normal_(generator=gen)
    workspace_ptrs = torch.tensor(workspace_handle.buffer_ptrs, dtype=torch.int64, device=device)

    b = torch.randn((n_local, k), dtype=torch.bfloat16, device=device, generator=gen)
    out = torch.empty((m, n_local), dtype=torch.bfloat16, device=device)
    torch_gathered = torch.empty((m, k), dtype=torch.bfloat16, device=device)

    chunks_per_rank = m_per_rank // CHUNK_ROWS
    ready = symm_mem.empty((world_size * chunks_per_rank, ), dtype=torch.int32, device=device)
    ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
    ready_ptrs = torch.tensor(ready_handle.buffer_ptrs, dtype=torch.int64, device=device)
    ready.zero_()

    tiles = ((m + c["PAIR_TILE_M"] - 1) // c["PAIR_TILE_M"]) * ((n_local + c["PAIR_TILE_N"] - 1) // c["PAIR_TILE_N"])
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    comm_ctas = max(1, min(comm_ctas, chunks_per_rank, sms - c["CLUSTER_SIZE"]))
    compute_sms = max(c["CLUSTER_SIZE"], sms - comm_ctas)
    compute_ctas = (min(compute_sms, tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    assert compute_ctas > 0

    dA, dB, dD = _descriptors(workspace, b, out)
    return AGGemm(
        ag_kernel=ag_kernel,
        gemm_kernel=gemm_kernel,
        a=a,
        b=b,
        out=out,
        workspace=workspace,
        torch_gathered=torch_gathered,
        ready=ready,
        workspace_ptrs=workspace_ptrs,
        ready_ptrs=ready_ptrs,
        dA=dA,
        dB=dB,
        dD=dD,
        rank=rank,
        world_size=world_size,
        m=m,
        n=n_local,
        k=k,
        m_per_rank=m_per_rank,
        chunks_per_rank=chunks_per_rank,
        comm_ctas=comm_ctas,
        compute_ctas=compute_ctas,
        workspace_handle=workspace_handle,
        ready_handle=ready_handle,
        comm_stream=torch.cuda.Stream(priority=-1),
        compute_stream=torch.cuda.Stream(),
    )


def prepare_fused(ctx: AGGemm, kernel: CompiledKernel, *, comm_ctas: int | None) -> AGGemmFuse:
    return AGGemmFuse.create(
        kernel,
        workspace_handle=ctx.workspace_handle,
        workspace=ctx.workspace,
        dB=ctx.dB,
        out_shape=ctx.out.shape,
        m=ctx.m,
        n=ctx.n,
        k=ctx.k,
        m_per_rank=ctx.m_per_rank,
        rank=ctx.rank,
        world_size=ctx.world_size,
        comm_ctas=comm_ctas,
    )


def prepare_triton_dist(ctx: AGGemm, *, autotune: bool, local_copy: bool, use_cooperative: bool) -> TritonDistAGGemm:
    try:
        from triton_dist.kernels.nvidia import ag_gemm, create_ag_gemm_context
    except ImportError as exc:
        raise RuntimeError("Triton-distributed is required by default; install it or pass --no-triton-dist") from exc

    td_context = create_ag_gemm_context(
        ctx.m,
        ctx.n * ctx.world_size,
        ctx.k,
        ctx.a.dtype,
        ctx.rank,
        ctx.world_size,
        num_local_ranks=ctx.world_size,
    )
    return TritonDistAGGemm(
        op=ag_gemm,
        context=td_context,
        a=ctx.a,
        local_a=td_context.symm_workspace[ctx.rank * ctx.m_per_rank:(ctx.rank + 1) * ctx.m_per_rank],
        b=ctx.b.t(),
        autotune=autotune,
        local_copy=local_copy,
        use_cooperative=use_cooperative,
    )


def prepare_thunderkittens(ctx: AGGemm, *, root: Path, num_comm_sms: int) -> ThunderKittensAGGemm:
    if num_comm_sms <= 0 or num_comm_sms >= 148 or num_comm_sms % 2:
        raise ValueError("ThunderKittens num_comm_sms must be an even integer in [2, 146]")
    if ctx.m % (256 * ctx.world_size):
        raise ValueError("ThunderKittens AG-GEMM requires M divisible by 256 * world_size")
    if ctx.n % 256:
        raise ValueError("ThunderKittens AG-GEMM requires N/world_size divisible by 256")
    if ctx.k % 128:
        raise ValueError("ThunderKittens AG-GEMM requires K divisible by 128")

    extension = _load_thunderkittens_extension(root)
    parallel_a = extension.TKParallelTensor((ctx.m, ctx.k), ctx.a.dtype, ctx.rank, ctx.world_size, True)
    barrier = extension.TKParallelTensor((2, 1024, 1024), torch.int32, ctx.rank, ctx.world_size, True)
    barrier.data_.zero_()
    target = ThunderKittensAGGemm(
        extension=extension,
        op=extension.all_gather_matmul,
        parallel_a=parallel_a,
        b=ctx.b,
        out=torch.empty_like(ctx.out),
        barrier=barrier,
        local_a=ctx.a,
        rank=ctx.rank,
        m_per_rank=ctx.m_per_rank,
        num_comm_sms=num_comm_sms,
    )
    target.stage_input()
    return target


def _join(ctx: AGGemm) -> None:
    current = torch.cuda.current_stream()
    current.wait_stream(ctx.comm_stream)
    current.wait_stream(ctx.compute_stream)
    current.synchronize()


def _release_symmetric_memory(ctx: AGGemm) -> None:
    workspace_handle, ready_handle = ctx.workspace_handle, ctx.ready_handle
    workspace, ready, a = ctx.workspace, ctx.ready, ctx.a
    ctx.workspace_handle = ctx.ready_handle = None
    ctx.workspace = ctx.ready = ctx.a = None
    del workspace_handle, ready_handle, workspace, ready, a
    gc.collect()


def _check_correctness(
    ctx: AGGemm,
    launch: Callable[[], None],
    get_out: Callable[[], torch.Tensor],
    *,
    message: str,
    iters: int = 1,
    pre: Callable[[], None] | None = None,
    check_gather: bool = False,
) -> None:
    ref: torch.Tensor | None = None
    for iteration in range(iters):
        if iteration:
            ctx.a.add_(0.125)
        if pre is not None:
            pre()
        launch()
        torch.cuda.synchronize()
        out = get_out()
        if ref is None:
            ref = torch.empty_like(out)
        dist.all_gather_into_tensor(ctx.torch_gathered, ctx.a)
        torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ref)
        torch.cuda.synchronize()
        if check_gather:
            torch.testing.assert_close(ctx.workspace, ctx.torch_gathered, rtol=0, atol=0)
        torch.testing.assert_close(out.float(), ref.float(), rtol=5e-4, atol=5e-4)
    if ctx.rank == 0:
        print(message)


def check_correctness(ctx: AGGemm) -> None:

    def launch() -> None:
        current = torch.cuda.current_stream()
        ctx.comm_stream.wait_stream(current)
        ctx.compute_stream.wait_stream(current)
        epoch = ctx.next_epoch()
        ctx.launch_ag(epoch)
        ctx.launch_gemm(epoch)
        _join(ctx)

    _check_correctness(
        ctx,
        launch,
        lambda: ctx.out,
        message=f"correctness: M={ctx.m} N={ctx.n * ctx.world_size} K={ctx.k} pass",
        iters=2,
        check_gather=True,
    )


def check_fused_correctness(ctx: AGGemm, fused: AGGemmFuse) -> None:
    _check_correctness(
        ctx,
        lambda: fused.launch(stream=torch.cuda.current_stream()),
        lambda: fused.out,
        message="fuse_overlap correctness: pass",
        iters=2,
        pre=lambda: dist.barrier(device_ids=[ctx.a.device.index]),
        check_gather=True,
    )


def check_triton_dist_correctness(ctx: AGGemm, target: TritonDistAGGemm) -> None:

    def pre() -> None:
        dist.barrier(device_ids=[ctx.a.device.index])
        target.stage_input()

    _check_correctness(
        ctx,
        lambda: target.launch(torch.cuda.current_stream()),
        lambda: target.out,
        message="triton_dist correctness: pass",
        pre=pre,
    )


def check_thunderkittens_correctness(ctx: AGGemm, target: ThunderKittensAGGemm) -> None:
    _check_correctness(
        ctx,
        lambda: target.launch(torch.cuda.current_stream()),
        lambda: target.out,
        message="thunderkittens correctness: pass",
        iters=2,
        pre=target.stage_input,
    )


def _time_once(ctx: AGGemm, mode: str) -> float:
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(current)

    if mode == "torch_ag":
        dist.all_gather_into_tensor(ctx.torch_gathered, ctx.a)
    elif mode == "torch_gemm":
        torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ctx.out)
    elif mode == "torch":
        dist.all_gather_into_tensor(ctx.torch_gathered, ctx.a)
        torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ctx.out)
    elif mode == "gemm":
        ctx.compute_stream.wait_event(start)
        ctx.launch_gemm(ctx.epoch)
        done = torch.cuda.Event()
        done.record(ctx.compute_stream)
        current.wait_event(done)
    else:
        epoch = ctx.next_epoch()
        ctx.comm_stream.wait_event(start)
        ctx.launch_ag(epoch)
        ag_done = torch.cuda.Event()
        ag_done.record(ctx.comm_stream)

        if mode == "ag":
            current.wait_event(ag_done)
        else:
            ctx.compute_stream.wait_event(start if mode == "overlap" else ag_done)
            ctx.launch_gemm(epoch)
            gemm_done = torch.cuda.Event()
            gemm_done.record(ctx.compute_stream)
            current.wait_event(ag_done)
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


def _bench(ctx: AGGemm, once: Callable[[], float], *, warmup: int, iters: int) -> float:
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
        dist.barrier(device_ids=[ctx.a.device.index])
        samples.append(once())
    value = torch.tensor(statistics.median(samples), dtype=torch.float64, device=ctx.a.device)
    dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value.item()


def bench_compare(
    ctx: AGGemm,
    *,
    warmup: int,
    iters: int,
    fused: AGGemmFuse | None = None,
    triton_dist_target: TritonDistAGGemm | None = None,
    thunderkittens_target: ThunderKittensAGGemm | None = None,
) -> None:
    func_table = {
        "ag": partial(_time_once, ctx, "ag"),
        "gemm": partial(_time_once, ctx, "gemm"),
        "sequential": partial(_time_once, ctx, "sequential"),
        "stream_overlap": partial(_time_once, ctx, "overlap"),
        "torch_ag": partial(_time_once, ctx, "torch_ag"),
        "torch_gemm": partial(_time_once, ctx, "torch_gemm"),
        "torch": partial(_time_once, ctx, "torch"),
    }
    if fused is not None:
        func_table["fused"] = partial(_time_launch_once, fused.launch)
    if triton_dist_target is not None:
        func_table["td_without_cudagraph"] = partial(
            _time_launch_once,
            triton_dist_target.launch_eager,
            triton_dist_target.stage_input,
        )
        if triton_dist_target.graph is not None:
            func_table["td_with_cudagraph"] = partial(
                _time_launch_once,
                triton_dist_target.launch,
                triton_dist_target.stage_input,
            )
    if thunderkittens_target is not None:
        func_table["thunderkittens"] = partial(
            _time_launch_once,
            thunderkittens_target.launch,
            thunderkittens_target.stage_input,
        )

    times = {name: _bench(ctx, run, warmup=warmup, iters=iters) for name, run in func_table.items()}
    if ctx.rank == 0:
        flops = 2.0 * ctx.m * ctx.n * ctx.k
        # NCCL all-gather busbw convention: total bytes * (P - 1) / P / time.
        recv_bytes = (ctx.m - ctx.m_per_rank) * ctx.k * ctx.a.element_size()
        fused_note = f"(comm_ctas={fused.comm_ctas})" if fused is not None else ""
        td_note = ""
        if triton_dist_target is not None:
            td_note = f"(local_copy={triton_dist_target.local_copy}, cooperative={triton_dist_target.use_cooperative})"
        tk_note = ""
        if thunderkittens_target is not None:
            tk_note = f"(num_comm_sms={thunderkittens_target.num_comm_sms})"
        rows = [
            ReportRow("ours_ag(tma):", "ag", "busbw", "bus"),
            ReportRow("ours_gemm:", "gemm", "tflops", "compute"),
            ReportRow("torch_ag:", "torch_ag", "busbw", "bus"),
            ReportRow("torch_gemm:", "torch_gemm", "tflops", "compute"),
            ReportRow("torch_sequential:", "torch", "tflops", "sum", "(sum bound)"),
            ReportRow("ours_sequential:", "sequential", "tflops", "sum", "(sum bound)"),
            ReportRow("stream_overlap:", "stream_overlap", "tflops", "max", "(max bound)"),
            ReportRow("fuse_overlap:", "fused", "tflops", "max", f"(max bound) {fused_note}".rstrip()),
            ReportRow("td_wo_cudagraph:", "td_without_cudagraph", "tflops", "max", f"(max bound) {td_note}".rstrip()),
            ReportRow("td_wi_cudagraph:", "td_with_cudagraph", "tflops", "max", f"(max bound) {td_note}".rstrip()),
            ReportRow("ThunderKittens:", "thunderkittens", "tflops", "max", f"(max bound) {tk_note}".rstrip()),
        ]
        print_benchmark_report(
            title=f"AG-GEMM M={ctx.m} N={ctx.n * ctx.world_size} K={ctx.k} ranks={ctx.world_size}",
            times=times,
            rows=rows,
            flops=flops,
            comm_bytes=recv_bytes,
            device=ctx.a.device,
        )

        print_speedups(
            times,
            [
                SpeedupRow("stream_speedup:", "torch", "stream_overlap", "(torch_sequential / stream_overlap)"),
                SpeedupRow("fused_speedup:", "torch", "fused", "(torch_sequential / fuse_overlap)"),
                SpeedupRow("td_wo_cudagraph_speedup:", "torch", "td_without_cudagraph",
                           "(torch_sequential / td_wo_cudagraph)"),
                SpeedupRow("td_wi_cudagraph_speedup:", "torch", "td_with_cudagraph",
                           "(torch_sequential / td_wi_cudagraph)"),
                SpeedupRow("thunderkittens_speedup:", "torch", "thunderkittens", "(torch_sequential / ThunderKittens)"),
            ],
        )


def export_trace(
    ctx: AGGemm,
    fused: AGGemmFuse,
    triton_dist_target: TritonDistAGGemm | None,
    thunderkittens_target: ThunderKittensAGGemm | None,
    trace_dir: str | Path,
    *,
    merge: bool = True,
) -> None:
    from torch.profiler import ProfilerActivity, profile, record_function

    def run_stream_overlap() -> None:
        current = torch.cuda.current_stream()
        ctx.comm_stream.wait_stream(current)
        ctx.compute_stream.wait_stream(current)
        epoch = ctx.next_epoch()
        ctx.launch_ag(epoch)
        ctx.launch_gemm(epoch)
        _join(ctx)

    def run_fuse_overlap() -> None:
        current = torch.cuda.current_stream()
        fused.launch(stream=current)
        current.synchronize()

    def run_triton_dist(*, use_cudagraph: bool) -> None:
        assert triton_dist_target is not None
        if use_cudagraph:
            assert triton_dist_target.graph is not None
        current = torch.cuda.current_stream()
        triton_dist_target.stage_input()
        if use_cudagraph:
            triton_dist_target.launch(stream=current)
        else:
            triton_dist_target.launch_eager(stream=current)
        current.synchronize()

    def run_torch_non_overlap() -> None:
        current = torch.cuda.current_stream()
        dist.all_gather_into_tensor(ctx.torch_gathered, ctx.a)
        torch.matmul(ctx.torch_gathered, ctx.b.t(), out=ctx.out)
        current.synchronize()

    def run_thunderkittens() -> None:
        assert thunderkittens_target is not None
        current = torch.cuda.current_stream()
        thunderkittens_target.stage_input()
        thunderkittens_target.launch(current)
        current.synchronize()

    device_id = ctx.a.device.index
    if triton_dist_target is not None and triton_dist_target.graph is None:
        dist.barrier(device_ids=[device_id])
        triton_dist_target.stage_input()
        triton_dist_target.capture(torch.cuda.current_stream())
        torch.cuda.synchronize()

    func_table = {}
    if triton_dist_target is not None:
        func_table.update({
            "td_without_cudagraph": partial(run_triton_dist, use_cudagraph=False),
            "td_with_cudagraph": partial(run_triton_dist, use_cudagraph=True),
        })
    if thunderkittens_target is not None:
        func_table["thunderkittens"] = run_thunderkittens
    func_table.update({
        "ours_stream_overlap": run_stream_overlap,
        "ours_fused_overlap": run_fuse_overlap,
        "torch_sequential": run_torch_non_overlap,
    })

    for run in func_table.values():
        dist.barrier(device_ids=[device_id])
        run()
    dist.barrier(device_ids=[device_id])

    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    )
    prof.start()
    for label, run in func_table.items():
        for _ in range(TRACE_ITERS):
            dist.barrier(device_ids=[device_id])
            with record_function(label):
                run()

    prof.stop()

    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ag_gemm_m{ctx.m}_n{ctx.n * ctx.world_size}_k{ctx.k}_r{ctx.world_size}"
    rank_trace = trace_dir / f"{stem}_rank{ctx.rank}.json.gz"
    prof.export_chrome_trace(str(rank_trace))
    dist.barrier(device_ids=[device_id])

    if merge and ctx.rank == 0:
        rank_traces = [trace_dir / f"{stem}_rank{rank}.json.gz" for rank in range(ctx.world_size)]
        merged_trace = merge_rank_traces(rank_traces, trace_dir / f"{stem}_merged.json.gz")
        print(f"merged trace: {merged_trace}")
    elif not merge and ctx.rank == 0:
        print(f"rank traces: {trace_dir / f'{stem}_rank*.json.gz'}")
    dist.barrier(device_ids=[device_id])


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark split-path role-proposal SM100 AG-GEMM")
    p.add_argument("--M", type=int, default=16384)
    p.add_argument("--N", type=int, default=16384)
    p.add_argument("--K", type=int, default=8192)
    p.add_argument("--comm-ctas", type=int, default=36)
    p.add_argument("--fuse-comm-ctas", type=int, default=None, help="comm CTAs inside the fused kernel (default: 32)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_ag_gemm")
    p.add_argument("--no-check", action="store_true")
    p.add_argument(
        "--triton-dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark Triton-distributed AG-GEMM (default: enabled)",
    )
    p.add_argument(
        "--triton-dist-autotune",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Autotune Triton-distributed before timing (default: enabled)",
    )
    p.add_argument(
        "--triton-dist-local-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include TD's local input copy in timing (default: enabled)",
    )
    p.add_argument(
        "--triton-dist-cooperative",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use TD's cooperative local-copy barrier (default: enabled)",
    )
    p.add_argument(
        "--triton-dist-cudagraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replay TD AG-GEMM through a CUDA graph (default: enabled)",
    )
    p.add_argument(
        "--thunderkittens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark the ThunderKittens AG-GEMM baseline (default: enabled)",
    )
    p.add_argument("--thunderkittens-root", type=Path, default=DEFAULT_THUNDERKITTENS_ROOT)
    p.add_argument(
        "--thunderkittens-comm-sms",
        type=int,
        default=4,
        help="Communication SMs used by ThunderKittens AG-GEMM",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true", help="Export a compressed torch.profiler trace")
    p.add_argument("--trace-dir", default=f"{os.path.dirname(__file__)}/lk_trace_role_v2_ag_gemm")
    p.add_argument(
        "--merge-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Merge all rank traces (default: enabled)",
    )
    args = p.parse_args()

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
    assert dist.get_world_size() == int(os.environ.get("LOCAL_WORLD_SIZE", dist.get_world_size()))

    ag_kernel = build_ag_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    gemm_kernel = build_gemm_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    fused_kernel = AGGemmFuse.build(cache_dir=args.cache_dir, verbose=args.verbose)
    kernels = (ag_kernel, gemm_kernel, fused_kernel)

    def _touch_launchers() -> None:
        for kernel in kernels:
            _ = kernel.launcher

    # Rank 0 links first to populate the on-disk cache, then the rest reuse it.
    if rank == 0:
        _touch_launchers()
    dist.barrier(device_ids=[local_rank])
    if rank != 0:
        _touch_launchers()
    dist.barrier(device_ids=[local_rank])

    ctx = prepare(ag_kernel, gemm_kernel, m=args.M, n=args.N, k=args.K, comm_ctas=args.comm_ctas)
    fused = prepare_fused(ctx, fused_kernel, comm_ctas=args.fuse_comm_ctas)
    triton_dist_target = (prepare_triton_dist(
        ctx,
        autotune=args.triton_dist_autotune,
        local_copy=args.triton_dist_local_copy,
        use_cooperative=args.triton_dist_cooperative,
    ) if args.triton_dist else None)
    thunderkittens_target = (prepare_thunderkittens(
        ctx,
        root=args.thunderkittens_root,
        num_comm_sms=args.thunderkittens_comm_sms,
    ) if args.thunderkittens else None)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    if triton_dist_target is not None and args.triton_dist_cudagraph:
        current = torch.cuda.current_stream()
        triton_dist_target.stage_input()
        triton_dist_target.launch(current)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_rank])
        triton_dist_target.capture(current)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print(f"cache_dir: {args.cache_dir}")
        print(f"comm_ctas: {ctx.comm_ctas}, compute_ctas: {ctx.compute_ctas}, "
              f"fuse_overlap_comm_ctas: {fused.comm_ctas}")
        if thunderkittens_target is not None:
            print(f"thunderkittens_root: {args.thunderkittens_root}; "
                  f"num_comm_sms: {thunderkittens_target.num_comm_sms}")

    if not args.no_check:
        check_fused_correctness(ctx, fused)
        if triton_dist_target is not None:
            check_triton_dist_correctness(ctx, triton_dist_target)
        if thunderkittens_target is not None:
            check_thunderkittens_correctness(ctx, thunderkittens_target)
        check_correctness(ctx)
    bench_compare(
        ctx,
        warmup=args.warmup,
        iters=args.iters,
        fused=fused,
        triton_dist_target=triton_dist_target,
        thunderkittens_target=thunderkittens_target,
    )
    if args.trace:
        export_trace(ctx, fused, triton_dist_target, thunderkittens_target, args.trace_dir, merge=args.merge_trace)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    if triton_dist_target is not None:
        triton_dist_target.release()
    if thunderkittens_target is not None:
        thunderkittens_target.release()
    del triton_dist_target, thunderkittens_target
    fused.release()
    del fused
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
