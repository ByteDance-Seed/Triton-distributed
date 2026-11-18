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
"""Build, check, and benchmark the role-proposal SM100 GEMM."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT

import argparse
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
import triton.testing

from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
from benchmarks.nv.sm100.dataflow._paths import role_example_source

GEMM_KERNEL = role_example_source("gemm")

TEST_SHAPES = [
    (1024, 1024, 1024),
    (8192, 8192, 8192),
]
BENCH_SHAPES = TEST_SHAPES
REFRESH_SECONDS = 0.4


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(GEMM_KERNEL, "<role-v2-gemm>", "exec"), ns)
    return ns


def _shared_mem_bytes() -> int:
    c = _cfg()
    return (c["NUM_TMA_STORE_STAGES"] * c["ST_TILE_M"] * c["SWIZZLE_CD_BYTES"] + c["NUM_STAGES"] *
            (c["SMEM_A"] + c["SMEM_B"]) + 1024)


def build_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    code = compile_dataflow_cuda(GEMM_KERNEL)
    c = _cfg()
    return CompiledKernel(
        code,
        "gemm_sm100_kernel",
        grid=(1, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=_shared_mem_bytes(),
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c["gemm_sm100_kernel"].cluster_shape,
        cta_group_shape=c["gemm_sm100_kernel"].cta_group_shape,
        allow_non_portable_cluster=True,
    )


def _descriptors(a: torch.Tensor, b: torch.Tensor, d: torch.Tensor, m: int, n: int, k: int):
    c = _cfg()
    dA = create_tma_2d_descriptor(
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
        d,
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


def _num_ctas(m: int, n: int) -> int:
    c = _cfg()
    tiles = ((m + c["PAIR_TILE_M"] - 1) // c["PAIR_TILE_M"]) * ((n + c["PAIR_TILE_N"] - 1) // c["PAIR_TILE_N"])
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return (min(sms, tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]


def make_launch(kernel: CompiledKernel, a: torch.Tensor, b: torch.Tensor, m: int, n: int, k: int):
    d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    dA, dB, dD = _descriptors(a, b, d, m, n, k)
    num_ctas = _num_ctas(m, n)

    def launch():
        kernel(dA, dB, dD, m, n, k, num_ctas, grid=(num_ctas, 1, 1))

    return launch, d


def _tflops(m: int, n: int, k: int, ms: float) -> float:
    return 2.0 * m * n * k / (ms * 1e9)


def _refresh(run) -> None:
    torch.cuda.synchronize()
    time.sleep(REFRESH_SECONDS)
    run()
    torch.cuda.synchronize()


def check_correctness(kernel: CompiledKernel, mnk: tuple[int, int, int]) -> None:
    m, n, k = mnk
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2026 + m + n + k)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    launch, out = make_launch(kernel, a, b, m, n, k)
    launch()
    torch.cuda.synchronize()
    ref = torch.empty_like(out)
    torch.matmul(a, b.t(), out=ref)
    torch.testing.assert_close(out.float(), ref.float(), rtol=5e-4, atol=5e-4)
    print(f"correctness: {m}x{n}x{k} pass")


def bench_compare(kernel: CompiledKernel, mnk: tuple[int, int, int], *, warmup: int, iters: int) -> None:
    m, n, k = mnk
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2026 + m + n + k)
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda", generator=gen)
    b = torch.randn(n, k, dtype=torch.bfloat16, device="cuda", generator=gen)

    launch, _ = make_launch(kernel, a, b, m, n, k)
    _refresh(launch)
    role_ms = triton.testing.do_bench(launch, warmup=warmup, rep=iters, return_mode="median")

    out = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    torch_launch = lambda: torch.matmul(a, b.t(), out=out)
    _refresh(torch_launch)
    torch_ms = triton.testing.do_bench(torch_launch, warmup=warmup, rep=iters, return_mode="median")

    role_tflops = _tflops(m, n, k, role_ms)
    torch_tflops = _tflops(m, n, k, torch_ms)
    speedup = torch_ms / role_ms

    print(f"\n=== GEMM {m}x{n}x{k} ===")
    print(f"role_v2:      {role_ms:.3f} ms, {role_tflops:.1f} TFLOPS")
    print(f"torch.matmul: {torch_ms:.3f} ms, {torch_tflops:.1f} TFLOPS")
    print(f"speedup:      {speedup:.2f}x (torch / role_v2)")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark role-proposal SM100 GEMM")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_gemm")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    kernel = build_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    print(f"cache_dir: {args.cache_dir}")

    if not args.no_check:
        for shape in TEST_SHAPES:
            check_correctness(kernel, shape)

    for shape in BENCH_SHAPES:
        bench_compare(kernel, shape, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
