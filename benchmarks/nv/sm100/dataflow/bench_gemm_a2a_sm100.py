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
"""Build, validate, and benchmark SM100 QKV projection plus Ulysses all-to-all."""

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
from examples.nv.sm100.a2a_tma_pull_sm100 import TmaPullQKVA2A
from examples.nv.sm100.gemm_a2a_fused_sm100 import GemmA2AFusePush
from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

GEMM_A2A_SOURCE = role_example_source("gemm_a2a")
A2A_ONLY_EPOCH = (1 << 30) - 1
CHECK_RTOL = 6e-2
CHECK_ATOL = 6e-2
REFRESH_SECONDS = 0.4


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(GEMM_A2A_SOURCE, "<role-v2-gemm-a2a>", "exec"), ns)
    return ns


def build_gemm_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    c = _cfg()
    return CompiledKernel(
        compile_dataflow_cuda(GEMM_A2A_SOURCE),
        "gemm_a2a_sm100_kernel",
        grid=(1, 1, 1),
        block=(c["NUM_THREADS"], 1, 1),
        shared_mem_bytes=c["SHARED_MEM_BYTES"],
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c["gemm_a2a_sm100_kernel"].cluster_shape,
        cta_group_shape=c["gemm_a2a_sm100_kernel"].cta_group_shape,
        allow_non_portable_cluster=True,
    )


def _gemm_descriptors(a: torch.Tensor, b: torch.Tensor, partial: torch.Tensor):
    c = _cfg()
    m, k = a.shape
    n = b.shape[0]
    # Standalone GEMM-A2A uses one pair-level 256x256 output tile.
    d_a = create_tma_2d_descriptor(
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
    d_b = create_tma_2d_descriptor(
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
    d_partial = create_tma_2d_descriptor(
        partial,
        gmem_inner_dim=n,
        gmem_outer_dim=m,
        smem_inner_dim=c["ST_TILE_N"],
        smem_outer_dim=c["ST_TILE_M"],
        gmem_outer_stride=n,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )

    # The fused push kernel intentionally keeps its faster 512x256 pair
    # geometry and therefore owns distinct A/B descriptors.
    d_a_fused = create_tma_2d_descriptor(
        a,
        gmem_inner_dim=k,
        gmem_outer_dim=m,
        smem_inner_dim=GemmA2AFusePush.K_TILE,
        smem_outer_dim=GemmA2AFusePush.CTA_TILE_M,
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    d_b_fused = create_tma_2d_descriptor(
        b,
        gmem_inner_dim=k,
        gmem_outer_dim=n,
        smem_inner_dim=GemmA2AFusePush.K_TILE,
        smem_outer_dim=GemmA2AFusePush.CTA_TILE_N_SHARD,
        gmem_outer_stride=k,
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    return d_a, d_b, d_partial, d_a_fused, d_b_fused


@dataclass
class GemmA2A:
    gemm_kernel: CompiledKernel
    a: torch.Tensor
    b: torch.Tensor
    partial: torch.Tensor
    ready: torch.Tensor
    counters: torch.Tensor
    d_a: object
    d_b: object
    d_partial: object
    d_a_fused: object
    d_b_fused: object
    a2a: TmaPullQKVA2A
    standalone_a2a: TmaPullQKVA2A
    torch_partial: torch.Tensor
    torch_sends: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    torch_recvs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    torch_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    rank: int
    world_size: int
    seq_len: int
    local_m: int
    n: int
    k: int
    q_heads: int
    kv_heads: int
    head_dim: int
    compute_ctas: int
    full_compute_ctas: int
    standalone_group_m: int
    overlap_group_m: int
    reserved_comm_ctas: int
    workspace_handle: object
    ready_handle: object
    comm_stream: torch.cuda.Stream
    compute_stream: torch.cuda.Stream
    epoch: int = 0

    @property
    def local_q_heads(self) -> int:
        return self.q_heads // self.world_size

    @property
    def local_kv_heads(self) -> int:
        return self.kv_heads // self.world_size

    @property
    def outputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.a2a.outputs

    def next_epoch(self) -> int:
        self.epoch = self.epoch % ((1 << 30) - 2) + 1
        return self.epoch

    def launch_gemm(self, epoch: int, *, standalone: bool = False, stream: torch.cuda.Stream | None = None) -> None:
        num_ctas = self.full_compute_ctas if standalone else self.compute_ctas
        self.gemm_kernel(
            self.d_a,
            self.d_b,
            self.d_partial,
            self.ready,
            self.counters,
            self.local_m,
            self.n,
            self.k,
            self.local_q_heads * self.head_dim,
            self.local_kv_heads * self.head_dim,
            self.world_size,
            self.standalone_group_m if standalone else self.overlap_group_m,
            epoch,
            num_ctas,
            grid=(num_ctas, 1, 1),
            stream=self.compute_stream if stream is None else stream,
        )

    def launch_a2a(self, epoch: int, *, standalone: bool = False, stream: torch.cuda.Stream | None = None) -> None:
        target = self.standalone_a2a if standalone else self.a2a
        target.launch(epoch, stream=self.comm_stream if stream is None else stream)


@dataclass
class TritonDistGemmA2A:
    op: object
    a: torch.Tensor
    b: torch.Tensor
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    num_comm_sms: int
    sm_margin: int
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def launch(self, stream: torch.cuda.Stream) -> None:
        with torch.cuda.stream(stream):
            result = self.op.forward(
                self.a,
                self.b,
                outputs=self.outputs,
                num_comm_sms=self.num_comm_sms,
                sm_margin=self.sm_margin,
            )
            self.out = tuple(result)

    def release(self) -> None:
        self.op.finalize()
        self.op = self.a = self.b = None
        self.outputs = ()
        self.out = None


def _torch_a2a_buffers(
    *,
    local_m: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    world_size: int,
    dtype: torch.dtype,
    device: torch.device,
):
    local_q_heads, local_kv_heads = q_heads // world_size, kv_heads // world_size
    sends = (
        torch.empty((world_size, local_m, local_q_heads, head_dim), dtype=dtype, device=device),
        torch.empty((world_size, local_m, local_kv_heads, head_dim), dtype=dtype, device=device),
        torch.empty((world_size, local_m, local_kv_heads, head_dim), dtype=dtype, device=device),
    )
    recvs = tuple(torch.empty_like(send) for send in sends)
    outputs = tuple(recv.view(local_m * world_size, recv.shape[2], head_dim) for recv in recvs)
    return sends, recvs, outputs


def prepare(
    gemm_kernel: CompiledKernel,
    a2a_kernel: CompiledKernel,
    *,
    seq_len: int,
    hidden: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    comm_ctas: int,
    comm_reserve_ctas: int | None,
    standalone_group_m: int,
    overlap_group_m: int,
) -> GemmA2A:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    c = _cfg()

    if seq_len % world_size:
        raise ValueError("seq_len must be divisible by world_size")
    if q_heads % world_size or kv_heads % world_size:
        raise ValueError("q_heads and kv_heads must be divisible by world_size")
    if q_heads % kv_heads:
        raise ValueError("q_heads must be an integer multiple of kv_heads for the TritonDist GQA baseline")
    if standalone_group_m <= 0 or overlap_group_m <= 0:
        raise ValueError("GEMM grouped-M scheduler sizes must be positive")

    local_m = seq_len // world_size
    n = (q_heads + 2 * kv_heads) * head_dim
    local_q_features = q_heads // world_size * head_dim
    local_kv_features = kv_heads // world_size * head_dim
    if local_m % c["PAIR_TILE_M"] or n % c["PAIR_TILE_N"] or hidden % c["K_TILE"]:
        raise ValueError(
            f"local seq, QKV features, and hidden must be divisible by {c['PAIR_TILE_M']}, {c['PAIR_TILE_N']}, and {c['K_TILE']}"
        )
    if local_m % TmaPullQKVA2A.TILE_M:
        raise ValueError(f"local seq must be divisible by A2A TILE_M={TmaPullQKVA2A.TILE_M}")
    if local_q_features % TmaPullQKVA2A.Q_TILE_N:
        raise ValueError(f"local Q features must be divisible by A2A Q_TILE_N={TmaPullQKVA2A.Q_TILE_N}")
    if local_kv_features % TmaPullQKVA2A.TILE_N:
        raise ValueError(f"local KV features must be divisible by A2A TILE_N={TmaPullQKVA2A.TILE_N}")

    gemm_tiles = (local_m // c["PAIR_TILE_M"]) * (n // c["PAIR_TILE_N"])
    if gemm_tiles < 2 * c["CLUSTER_SIZE"]:
        raise ValueError("GEMM-A2A requires at least four output tiles")

    a_gen = torch.Generator(device=device)
    a_gen.manual_seed(2026 + rank)
    b_gen = torch.Generator(device=device)
    b_gen.manual_seed(1729)
    a = torch.randn((local_m, hidden), dtype=torch.bfloat16, device=device, generator=a_gen).mul_(0.1)
    b = torch.randn((n, hidden), dtype=torch.bfloat16, device=device, generator=b_gen).mul_(0.1)

    partial_out = symm_mem.empty((local_m, n), dtype=torch.bfloat16, device=device)
    workspace_handle = symm_mem.rendezvous(partial_out, group=dist.group.WORLD)
    partial_out.zero_()

    num_ready = (local_m // TmaPullQKVA2A.READY_BM) * (n // TmaPullQKVA2A.READY_BN)
    ready = symm_mem.empty((num_ready, ), dtype=torch.int32, device=device)
    ready_handle = symm_mem.rendezvous(ready, group=dist.group.WORLD)
    ready.zero_()
    counters = torch.zeros_like(ready)

    d_a, d_b, d_partial, d_a_fused, d_b_fused = _gemm_descriptors(a, b, partial_out)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    a2a = TmaPullQKVA2A.create(
        a2a_kernel,
        workspace_handle=workspace_handle,
        partial=partial_out,
        ready_handle=ready_handle,
        ready=ready,
        rank=rank,
        world_size=world_size,
        head_dim=head_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        comm_ctas=comm_ctas,
    )
    standalone_a2a = TmaPullQKVA2A.create(
        a2a_kernel,
        workspace_handle=workspace_handle,
        partial=partial_out,
        ready_handle=ready_handle,
        ready=ready,
        rank=rank,
        world_size=world_size,
        head_dim=head_dim,
        q_heads=q_heads,
        kv_heads=kv_heads,
        comm_ctas=sms,
    )

    requested_reserve = a2a.num_ctas if comm_reserve_ctas is None else comm_reserve_ctas
    if requested_reserve <= 0:
        raise ValueError("comm_reserve_ctas must be positive")
    reserved = max(1, min(requested_reserve, sms - c["CLUSTER_SIZE"]))
    compute_sms = max(c["CLUSTER_SIZE"], sms - reserved)
    compute_ctas = (min(compute_sms, gemm_tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    full_compute_ctas = (min(sms, gemm_tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]
    if compute_ctas == 0:
        raise ValueError("problem is too small for a two-CTA GEMM cluster")

    torch_sends, torch_recvs, torch_outputs = _torch_a2a_buffers(
        local_m=local_m,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        world_size=world_size,
        dtype=torch.bfloat16,
        device=device,
    )
    return GemmA2A(
        gemm_kernel=gemm_kernel,
        a=a,
        b=b,
        partial=partial_out,
        ready=ready,
        counters=counters,
        d_a=d_a,
        d_b=d_b,
        d_partial=d_partial,
        d_a_fused=d_a_fused,
        d_b_fused=d_b_fused,
        a2a=a2a,
        standalone_a2a=standalone_a2a,
        torch_partial=torch.empty_like(partial_out),
        torch_sends=torch_sends,
        torch_recvs=torch_recvs,
        torch_outputs=torch_outputs,
        rank=rank,
        world_size=world_size,
        seq_len=seq_len,
        local_m=local_m,
        n=n,
        k=hidden,
        q_heads=q_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        compute_ctas=compute_ctas,
        full_compute_ctas=full_compute_ctas,
        standalone_group_m=standalone_group_m,
        overlap_group_m=overlap_group_m,
        reserved_comm_ctas=reserved,
        workspace_handle=workspace_handle,
        ready_handle=ready_handle,
        # High-priority comm stream matches GEMM-RS overlap: A2A CTAs should
        # win scheduling against compute when both kernels are resident.
        comm_stream=torch.cuda.Stream(priority=-1),
        compute_stream=torch.cuda.Stream(),
    )


def prepare_fused(ctx: GemmA2A, kernel: CompiledKernel) -> GemmA2AFusePush:
    return GemmA2AFusePush.create(
        kernel,
        d_a=ctx.d_a_fused,
        d_b=ctx.d_b_fused,
        m=ctx.local_m,
        n=ctx.n,
        k=ctx.k,
        rank=ctx.rank,
        world_size=ctx.world_size,
        head_dim=ctx.head_dim,
        q_heads=ctx.q_heads,
        kv_heads=ctx.kv_heads,
    )


def prepare_triton_dist(
    ctx: GemmA2A,
    world_group,
    *,
    num_comm_sms: int,
    sm_margin: int,
    use_persistent: bool,
) -> TritonDistGemmA2A:
    try:
        from triton_dist.kernels.nvidia.sp_ulysess_qkv_gemm_all2all import SpUlysessQKVGemmAll2AllKernel
        from triton_dist.utils import nvshmem_barrier_all_on_stream
    except ImportError as exc:
        raise RuntimeError("Triton-distributed is required by default; install it or pass --no-triton-dist") from exc

    sms = torch.cuda.get_device_properties(ctx.a.device).multi_processor_count
    if num_comm_sms <= 0 or num_comm_sms + sm_margin >= sms:
        raise ValueError("triton-dist comm SMs must be positive and leave at least one GEMM SM")
    if num_comm_sms >= ctx.world_size:
        if num_comm_sms % ctx.world_size:
            raise ValueError("triton-dist comm SMs must be divisible by world_size when comm SMs >= world_size")
    elif ctx.world_size % num_comm_sms:
        raise ValueError("world_size must be divisible by triton-dist comm SMs when comm SMs < world_size")

    op = SpUlysessQKVGemmAll2AllKernel(
        world_group,
        nnodes=1,
        sp_size=ctx.world_size,
        max_batch_size=1,
        max_seq_len=ctx.seq_len,
        hidden_size=ctx.k,
        head_dim=ctx.head_dim,
        qkv_out_features=ctx.n,
        input_dtype=ctx.a.dtype,
        output_dtype=ctx.a.dtype,
        gqa=ctx.q_heads // ctx.kv_heads,
        max_num_comm_buf=1,
        use_persistent=use_persistent,
    )
    nvshmem_barrier_all_on_stream()
    outputs = (
        torch.empty((1, ctx.seq_len, ctx.local_q_heads, ctx.head_dim), dtype=ctx.a.dtype, device=ctx.a.device),
        torch.empty((1, ctx.seq_len, ctx.local_kv_heads, ctx.head_dim), dtype=ctx.a.dtype, device=ctx.a.device),
        torch.empty((1, ctx.seq_len, ctx.local_kv_heads, ctx.head_dim), dtype=ctx.a.dtype, device=ctx.a.device),
    )
    return TritonDistGemmA2A(
        op=op,
        a=ctx.a.view(1, ctx.local_m, ctx.k),
        b=ctx.b,
        outputs=outputs,
        num_comm_sms=num_comm_sms,
        sm_margin=sm_margin,
    )


def _pack_torch_qkv_for_a2a(ctx: GemmA2A) -> None:
    q_end = ctx.q_heads * ctx.head_dim
    k_end = q_end + ctx.kv_heads * ctx.head_dim
    sources = (
        ctx.torch_partial[:, :q_end].view(ctx.local_m, ctx.q_heads, ctx.head_dim),
        ctx.torch_partial[:, q_end:k_end].view(ctx.local_m, ctx.kv_heads, ctx.head_dim),
        ctx.torch_partial[:, k_end:].view(ctx.local_m, ctx.kv_heads, ctx.head_dim),
    )
    local_heads = (ctx.local_q_heads, ctx.local_kv_heads, ctx.local_kv_heads)
    for source, send, heads in zip(sources, ctx.torch_sends, local_heads):
        send.copy_(source.view(ctx.local_m, ctx.world_size, heads, ctx.head_dim).permute(1, 0, 2, 3))


def _torch_qkva2a(
    ctx: GemmA2A,
    q_buffer: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run three A2As on prepacked destination-major buffers without copies."""
    for send, recv in zip((q_buffer, k_buffer, v_buffer), ctx.torch_recvs):
        dist.all_to_all_single(recv, send)
    return ctx.torch_outputs


def _torch_correctness_ref(ctx: GemmA2A) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    _pack_torch_qkv_for_a2a(ctx)
    return _torch_qkva2a(ctx, *ctx.torch_sends)


def _torch_max_perf(ctx: GemmA2A) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    return _torch_qkva2a(ctx, *ctx.torch_sends)


def _assert_outputs(
    ctx: GemmA2A,
    actual: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    label: str,
) -> None:
    failed = torch.zeros((), dtype=torch.int32, device=ctx.a.device)
    max_error = torch.zeros((), dtype=torch.float32, device=ctx.a.device)
    for actual_tensor, expected_tensor in zip(actual, expected):
        if actual_tensor.ndim == expected_tensor.ndim + 1 and actual_tensor.shape[0] == 1:
            actual_tensor = actual_tensor[0]
        if actual_tensor.shape != expected_tensor.shape:
            failed.fill_(1)
            continue
        finite = torch.isfinite(actual_tensor).all()
        error = (actual_tensor.float() - expected_tensor.float()).abs().max()
        close = torch.allclose(actual_tensor.float(), expected_tensor.float(), rtol=CHECK_RTOL, atol=CHECK_ATOL)
        if not close or not finite.item():
            failed.fill_(1)
        max_error = torch.maximum(max_error, error)

    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    dist.all_reduce(max_error, op=dist.ReduceOp.MAX)
    if failed.item():
        raise AssertionError(f"{label} mismatch across ranks; max_abs_error={max_error.item():.6g}")
    if ctx.rank == 0:
        print(f"{label} correctness: pass")


def _join(ctx: GemmA2A) -> None:
    current = torch.cuda.current_stream()
    current.wait_stream(ctx.comm_stream)
    current.wait_stream(ctx.compute_stream)
    current.synchronize()


def _launch_split(ctx: GemmA2A, *, overlap: bool) -> None:
    current = torch.cuda.current_stream()
    epoch = ctx.next_epoch()
    if overlap:
        ctx.comm_stream.wait_stream(current)
        ctx.compute_stream.wait_stream(current)
        ctx.launch_gemm(epoch)
        ctx.launch_a2a(epoch)
        _join(ctx)
    else:
        ctx.launch_gemm(epoch, standalone=True, stream=current)
        ctx.launch_a2a(epoch, standalone=True, stream=current)
        current.synchronize()


def check_correctness(ctx: GemmA2A, fused: GemmA2AFusePush, triton_dist_target: TritonDistGemmA2A | None) -> None:
    for iteration in range(2):
        if iteration:
            ctx.a.add_(0.01)
        reference = _torch_correctness_ref(ctx)

        ctx.partial.copy_(ctx.torch_partial)
        a2a_epoch = ctx.next_epoch()
        ctx.ready.fill_(a2a_epoch)
        torch.cuda.synchronize()
        dist.barrier(device_ids=[ctx.a.device.index])
        for output in ctx.standalone_a2a.outputs:
            output.fill_(float("nan"))
        ctx.standalone_a2a.launch(a2a_epoch, stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        _assert_outputs(ctx, ctx.standalone_a2a.outputs, reference, label="littlekernel_tma_a2a")

        ctx.partial.fill_(float("nan"))
        for output in ctx.standalone_a2a.outputs:
            output.fill_(float("nan"))
        _launch_split(ctx, overlap=False)
        _assert_outputs(ctx, ctx.standalone_a2a.outputs, reference, label="littlekernel_sequential")

        ctx.partial.fill_(float("nan"))
        for output in ctx.a2a.outputs:
            output.fill_(float("nan"))
        _launch_split(ctx, overlap=True)
        _assert_outputs(ctx, ctx.a2a.outputs, reference, label="littlekernel_stream_overlap")

        for output in fused.outputs:
            output.fill_(float("nan"))
        fused.launch(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        _assert_outputs(ctx, fused.outputs, reference, label="littlekernel_role_fused")

        if triton_dist_target is not None:
            triton_dist_target.launch(torch.cuda.current_stream())
            torch.cuda.synchronize()
            assert triton_dist_target.out is not None
            _assert_outputs(ctx, triton_dist_target.out, reference, label="triton_dist")


def _time_once(ctx: GemmA2A, mode: str) -> float:
    current = torch.cuda.current_stream()
    if mode == "a2a":
        ctx.ready.fill_(A2A_ONLY_EPOCH)
        current.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(current)

    if mode == "torch_gemm":
        torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    elif mode == "torch_a2a":
        _torch_qkva2a(ctx, *ctx.torch_sends)
    elif mode == "torch_sequential":
        _torch_max_perf(ctx)
    elif mode == "gemm":
        epoch = ctx.next_epoch()
        ctx.launch_gemm(epoch, standalone=True, stream=current)
    elif mode == "a2a":
        ctx.launch_a2a(A2A_ONLY_EPOCH, standalone=True, stream=current)
    else:
        epoch = ctx.next_epoch()
        if mode == "stream_overlap":
            ctx.comm_stream.wait_event(start)
            ctx.compute_stream.wait_event(start)
            ctx.launch_gemm(epoch)
            ctx.launch_a2a(epoch)
            comm_done = torch.cuda.Event()
            gemm_done = torch.cuda.Event()
            comm_done.record(ctx.comm_stream)
            gemm_done.record(ctx.compute_stream)
            current.wait_event(comm_done)
            current.wait_event(gemm_done)
        elif mode == "sequential":
            ctx.launch_gemm(epoch, standalone=True, stream=current)
            ctx.launch_a2a(epoch, standalone=True, stream=current)
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


def _bench(ctx: GemmA2A, once: Callable[[], float], *, warmup: int, iters: int) -> float:
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
    ms = torch.tensor(samples, dtype=torch.float64, device=ctx.a.device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return statistics.median(ms.tolist())


def bench_compare(
    ctx: GemmA2A,
    *,
    warmup: int,
    iters: int,
    fused: GemmA2AFusePush,
    triton_dist_target: TritonDistGemmA2A | None,
) -> None:
    torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    # Deliberate upper bound: pack once outside timing. The timed GEMM and A2A
    # have stream ordering but no data dependency through torch_partial.
    _pack_torch_qkv_for_a2a(ctx)
    ctx.partial.copy_(ctx.torch_partial)
    ctx.ready.fill_(A2A_ONLY_EPOCH)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[ctx.a.device.index])

    funcs: dict[str, Callable[[], float]] = {
        "torch_gemm": partial(_time_once, ctx, "torch_gemm"),
        "torch_a2a": partial(_time_once, ctx, "torch_a2a"),
        "torch_sequential": partial(_time_once, ctx, "torch_sequential"),
        "gemm": partial(_time_once, ctx, "gemm"),
        "a2a": partial(_time_once, ctx, "a2a"),
        "sequential": partial(_time_once, ctx, "sequential"),
        "stream_overlap": partial(_time_once, ctx, "stream_overlap"),
        "role_fused": partial(_time_launch_once, fused.launch),
    }
    if triton_dist_target is not None:
        funcs["triton_dist"] = partial(_time_launch_once, triton_dist_target.launch)

    times = {name: _bench(ctx, fn, warmup=warmup, iters=iters) for name, fn in funcs.items()}
    if ctx.rank != 0:
        return

    flops = 2.0 * ctx.local_m * ctx.n * ctx.k
    remote_bytes = ctx.local_m * ctx.n * ctx.a.element_size() * (ctx.world_size - 1) / ctx.world_size
    gqa = ctx.q_heads // ctx.kv_heads
    rows = [
        ReportRow("torch_gemm:", "torch_gemm", "tflops", "compute"),
        ReportRow("ours_gemm:", "gemm", "tflops", "compute"),
        ReportRow("torch_a2a(no_pack):", "torch_a2a", "busbw", "bus"),
        ReportRow("ours_a2a(tma):", "a2a", "busbw", "bus"),
        ReportRow("torch_sequential:", "torch_sequential", "tflops", "sum", "(sum bound)"),
        ReportRow("ours_sequential:", "sequential", "tflops", "sum", "(sum bound)"),
        ReportRow("stream_overlap:", "stream_overlap", "tflops", "max", "(max bound)"),
        ReportRow("role_fused:", "role_fused", "tflops", "max", "(max bound)"),
        ReportRow("triton_dist:", "triton_dist", "tflops", "max", "(max bound)"),
    ]
    print_benchmark_report(
        title=(f"QKV GEMM-A2A seq={ctx.seq_len} local_seq={ctx.local_m} hidden={ctx.k} "
               f"q_heads={ctx.q_heads} kv_heads={ctx.kv_heads} head_dim={ctx.head_dim} "
               f"gqa={gqa} ranks={ctx.world_size}"),
        times=times,
        rows=rows,
        flops=flops,
        comm_bytes=remote_bytes,
        device=ctx.a.device,
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
    ctx: GemmA2A,
    fused: GemmA2AFusePush,
    triton_dist_target: TritonDistGemmA2A | None,
    trace_dir: str | Path,
    *,
    merge: bool,
    trace_iters: int,
    with_stack: bool,
) -> None:
    from torch.profiler import ProfilerActivity, profile, record_function

    torch.matmul(ctx.a, ctx.b.t(), out=ctx.torch_partial)
    _pack_torch_qkv_for_a2a(ctx)
    funcs: dict[str, Callable[[], None]] = {
        "torch_sequential": lambda: _time_once(ctx, "torch_sequential"),
        "ours_sequential": lambda: _time_once(ctx, "sequential"),
        "ours_stream_overlap": lambda: _time_once(ctx, "stream_overlap"),
        "ours_role_fused": lambda: _time_launch_once(fused.launch),
    }
    if triton_dist_target is not None:
        funcs["triton_dist"] = lambda: _time_launch_once(triton_dist_target.launch)

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
    stem = f"gemm_a2a_s{ctx.seq_len}_h{ctx.k}_q{ctx.q_heads}_kv{ctx.kv_heads}_d{ctx.head_dim}_r{ctx.world_size}"
    rank_trace = trace_dir / f"{stem}_rank{ctx.rank}.json.gz"
    prof.export_chrome_trace(str(rank_trace))
    dist.barrier(device_ids=[ctx.a.device.index])
    if merge and ctx.rank == 0:
        rank_traces = [trace_dir / f"{stem}_rank{rank}.json.gz" for rank in range(ctx.world_size)]
        print(f"merged trace: {merge_rank_traces(rank_traces, trace_dir / f'{stem}_merged.json.gz')}")
    elif ctx.rank == 0:
        print(f"rank traces: {trace_dir / f'{stem}_rank*.json.gz'}")
    dist.barrier(device_ids=[ctx.a.device.index])


def _release_symmetric_memory(ctx: GemmA2A) -> None:
    workspace_handle, ready_handle = ctx.workspace_handle, ctx.ready_handle
    partial_out, ready = ctx.partial, ctx.ready
    ctx.workspace_handle = ctx.ready_handle = None
    ctx.partial = ctx.ready = None
    del workspace_handle, ready_handle, partial_out, ready
    gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark SM100 QKV projection plus Ulysses all-to-all")
    p.add_argument("--seq-len", type=int, default=65536, help="global sequence length")
    p.add_argument("--hidden", type=int, default=6144, help="QKV projection input features")
    p.add_argument("--q-heads", type=int, default=128)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--comm-ctas", type=int, default=36)
    p.add_argument(
        "--comm-reserve-ctas",
        type=int,
        default=36,
        help="GEMM CTAs reserved for communication",
    )
    p.add_argument("--standalone-group-m", type=int, default=8)
    p.add_argument("--overlap-group-m", type=int, default=32)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_gemm_a2a")
    p.add_argument("--no-check", action="store_true")
    p.add_argument(
        "--triton-dist",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark sp_ulysess_qkv_gemm_all2all.py (default: enabled)",
    )
    p.add_argument("--triton-dist-comm-sms", type=int, default=32)
    p.add_argument("--triton-dist-sm-margin", type=int, default=0)
    p.add_argument("--triton-dist-persistent", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--trace-dir", default=f"{os.path.dirname(__file__)}/lk_trace_role_v2_gemm_a2a")
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
    a2a_kernel = TmaPullQKVA2A.build(cache_dir=args.cache_dir, verbose=args.verbose)
    fused_kernel = GemmA2AFusePush.build(cache_dir=args.cache_dir, verbose=args.verbose)
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
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        comm_ctas=args.comm_ctas,
        comm_reserve_ctas=args.comm_reserve_ctas,
        standalone_group_m=args.standalone_group_m,
        overlap_group_m=args.overlap_group_m,
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
              f"fused_ctas={fused.num_ctas}")

    if not args.no_check:
        check_correctness(ctx, fused, td_target)
    bench_compare(ctx, warmup=args.warmup, iters=args.iters, fused=fused, triton_dist_target=td_target)
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
