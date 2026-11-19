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
"""Build, check, and benchmark the role-proposal SM100 MXFP8 blockscaled GEMM."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT, role_example_path, role_example_source

import argparse
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
import triton.testing

from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor
from benchmarks.nv.sm100.dataflow.bench_blockscaled import pack_torch_scales, torch_blockscaled_mm
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda

KERNEL_SRC = role_example_source("mxfp8")

TEST_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]
BENCH_SHAPES = [(8192, 8192, 8192)]
REFRESH_SECONDS = 0.4

# MXFP8 constants
VS = 32  # vector size: one SF per 32 elements
E4M3_MAX = 448.0  # max finite value of E4M3
SF_SMEM_M = 32  # SF SMEM rows (CUTLASS interleaved layout)
SF_SMEM_K = 32  # SF SMEM bytes per row (padded from 16 for TMA)


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(KERNEL_SRC, "<role-v2-mxfp8-gemm>", "exec"), ns)
    return ns


def _shared_mem_bytes() -> int:
    c = _cfg()
    return (c["NUM_TMA_STORE_STAGES"] * c["ST_TILE_M"] * c["ST_TILE_N"] * 2 + c["NUM_STAGES"] *
            (c["SMEM_A"] + c["SMEM_B"] + c["SMEM_SFA"] + c["SMEM_SFB"]) + 1024)


def build_kernel(*, cache_dir: str | None = None, verbose: bool = False) -> CompiledKernel:
    code = compile_dataflow_cuda(KERNEL_SRC)
    c = _cfg()
    return CompiledKernel(
        code,
        "mxfp8_gemm_sm100_kernel",
        grid=(1, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=_shared_mem_bytes(),
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c["mxfp8_gemm_sm100_kernel"].cluster_shape,
        cta_group_shape=c["mxfp8_gemm_sm100_kernel"].cta_group_shape,
        extra_nvcc_options=["-lineinfo"],
        allow_non_portable_cluster=True,
    )


# ---------------------------------------------------------------------------
# MXFP8 quantization helpers
# ---------------------------------------------------------------------------
def _make_interleaved_sf(sf_u8: torch.Tensor, M: int, K: int, vs: int = VS) -> torch.Tensor:
    """Build the duplicated 2CTA SF tensor from logical (M, K//vs) scales.

    CUTLASS blockscaled layout: 128 M-rows packed into 32 rows × 16 bytes
    (4 SF bytes per 16-byte row, one from each block of 32 M-rows).
    Inner dim padded from 16 to 32 for TMA alignment.
    """
    num_k_tiles = K // 128
    outer_dim = M * K // 256
    dev = sf_u8.device
    m_idx = torch.arange(M, device=dev)
    k0_idx = torch.arange(K // vs, device=dev)
    mg, kg = torch.meshgrid(m_idx, k0_idx, indexing="ij")
    row = (mg // 256) * num_k_tiles * 128 + (kg // 4) * 128 + (mg // 128) % 2 * 32 + mg % 32
    col = (mg // 32) % 4 * 4 + kg % 4
    out = torch.zeros(outer_dim, 32, dtype=torch.uint8, device=dev)
    out[row, col] = sf_u8
    out[row + 64, col] = sf_u8
    return out


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize (M, K) tensor to MXFP8 E4M3 + E8M0 scale factors.

    Returns:
        x_q: (M, K) uint8 view of float8_e4m3fn
        sf:  (M*K//256, 32) uint8 E8M0 scales in the duplicated 2CTA layout
        torch_sf: E8M0 scales packed for torch.nn.functional.scaled_mm
    """
    M, K = x.shape
    assert K % VS == 0, f"K={K} must be divisible by VS={VS}"
    x_f32 = x.float().reshape(M, K // VS, VS)
    x_abs_max = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    # E8M0: value = 2^(e8 - 127).  Choose sf = 2^round(log2(amax / E4M3_MAX)).
    sf_log2 = torch.log2(x_abs_max / E4M3_MAX).round()
    sf_e8m0 = (sf_log2 + 127).clamp(0, 255).to(torch.uint8)
    sf_float = (2.0**sf_log2).to(torch.float32)
    x_scaled = (x_f32 / sf_float).clamp(-E4M3_MAX, E4M3_MAX)
    x_q = x_scaled.to(torch.float8_e4m3fn).reshape(M, K)
    sf_u8 = sf_e8m0.squeeze(-1)  # (M, K//VS)
    sf_interleaved = _make_interleaved_sf(sf_u8, M, K)
    torch_sf = pack_torch_scales(sf_u8, torch.float8_e8m0fnu)
    return x_q, sf_interleaved, torch_sf


def dequantize_mxfp8(x_q: torch.Tensor, sf_interleaved: torch.Tensor, M: int, K: int, vs: int = VS) -> torch.Tensor:
    """Dequantize MXFP8 tensor back to float32 from interleaved SF layout."""
    x_f32 = x_q.float().reshape(M, K // VS, VS)
    # Recover SF from interleaved layout
    num_k_tiles = K // 128
    dev = x_q.device
    m_idx = torch.arange(M, device=dev)
    k0_idx = torch.arange(K // vs, device=dev)
    mg, kg = torch.meshgrid(m_idx, k0_idx, indexing="ij")
    row = (mg // 256) * num_k_tiles * 128 + (kg // 4) * 128 + (mg // 128) % 2 * 32 + mg % 32
    col = (mg // 32) % 4 * 4 + kg % 4
    sf_u8 = sf_interleaved[row, col]  # (M, K//vs)
    sf_float = (2.0**(sf_u8.float() - 127)).unsqueeze(-1)
    return (x_f32 * sf_float).reshape(M, K)


# ---------------------------------------------------------------------------
# TMA descriptors
# ---------------------------------------------------------------------------


def _descriptors(a_q, b_q, sfa, sfb, d, m, n, k):
    c = _cfg()
    # A: (M, K) uint8, one K=128 stage per SMEM row.
    dA = create_tma_2d_descriptor(
        a_q,
        gmem_inner_dim=k,
        gmem_outer_dim=m,
        smem_inner_dim=c["K_TILE"],
        smem_outer_dim=c["CTA_TILE_M"],
        gmem_outer_stride=k,
        swizzle_mode=c["K_TILE"],
        oob_fill=False,
        l2_promotion=3,
    )
    # B: (N, K) uint8, 32B swizzle
    dB = create_tma_2d_descriptor(
        b_q,
        gmem_inner_dim=k,
        gmem_outer_dim=n,
        smem_inner_dim=c["K_TILE"],
        smem_outer_dim=c["CTA_TILE_N_SHARD"],
        gmem_outer_stride=k,
        swizzle_mode=c["K_TILE"],
        oob_fill=False,
        l2_promotion=3,
    )
    sf_outer_a = m * k // 256
    sf_outer_b = n * k // 256
    dSFA = create_tma_2d_descriptor(
        sfa,
        gmem_inner_dim=SF_SMEM_K,
        gmem_outer_dim=sf_outer_a,
        smem_inner_dim=SF_SMEM_K,
        smem_outer_dim=SF_SMEM_M,
        gmem_outer_stride=SF_SMEM_K,
        swizzle_mode=32,
        oob_fill=False,
        l2_promotion=3,
    )
    # SFB: (N*K//128, 32) interleaved SF, 32B swizzle
    dSFB = create_tma_2d_descriptor(
        sfb,
        gmem_inner_dim=SF_SMEM_K,
        gmem_outer_dim=sf_outer_b,
        smem_inner_dim=SF_SMEM_K,
        smem_outer_dim=SF_SMEM_M,
        gmem_outer_stride=SF_SMEM_K,
        swizzle_mode=32,
        oob_fill=False,
        l2_promotion=3,
    )
    # D: (M, N) bf16, 128B swizzle
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
    return dA, dB, dSFA, dSFB, dD


def _num_ctas(m: int, n: int) -> int:
    c = _cfg()
    tiles = ((m + c["PAIR_TILE_M"] - 1) // c["PAIR_TILE_M"]) * ((n + c["PAIR_TILE_N"] - 1) // c["PAIR_TILE_N"])
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return (min(sms, tiles * c["CLUSTER_SIZE"]) // c["CLUSTER_SIZE"]) * c["CLUSTER_SIZE"]


def make_launch(kernel: CompiledKernel, a_q, b_q, sfa, sfb, m, n, k):
    d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    dA, dB, dSFA, dSFB, dD = _descriptors(a_q, b_q, sfa, sfb, d, m, n, k)
    num_ctas = _num_ctas(m, n)

    def launch():
        kernel(dA, dB, dSFA, dSFB, dD, m, n, k, num_ctas, grid=(num_ctas, 1, 1))

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
    a = torch.randn(m, k, dtype=torch.float32, device="cuda", generator=gen)
    b = torch.randn(n, k, dtype=torch.float32, device="cuda", generator=gen)

    a_q, sfa, torch_sfa = quantize_mxfp8(a)
    b_q, sfb, torch_sfb = quantize_mxfp8(b)

    launch, out = make_launch(kernel, a_q, b_q, sfa, sfb, m, n, k)
    launch()
    torch.cuda.synchronize()

    ref = torch.matmul(
        dequantize_mxfp8(a_q, sfa, m, k),
        dequantize_mxfp8(b_q, sfb, n, k).t(),
    ).to(torch.bfloat16)
    torch_ref = torch_blockscaled_mm(
        a_q,
        b_q.t(),
        torch_sfa,
        torch_sfb,
        32,
    )
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(torch_ref.float(), ref.float(), rtol=1e-2, atol=1e-2)
    print(f"correctness: {m}x{n}x{k} pass")


def bench_compare(kernel: CompiledKernel, mnk: tuple[int, int, int], *, warmup: int, iters: int) -> None:
    m, n, k = mnk
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2026 + m + n + k)
    a = torch.randn(m, k, dtype=torch.float32, device="cuda", generator=gen)
    b = torch.randn(n, k, dtype=torch.float32, device="cuda", generator=gen)

    a_q, sfa, torch_sfa = quantize_mxfp8(a)
    b_q, sfb, torch_sfb = quantize_mxfp8(b)

    launch, _ = make_launch(kernel, a_q, b_q, sfa, sfb, m, n, k)
    _refresh(launch)
    role_ms = triton.testing.do_bench(launch, warmup=warmup, rep=iters, return_mode="median")

    torch_launch = lambda: torch_blockscaled_mm(
        a_q,
        b_q.t(),
        torch_sfa,
        torch_sfb,
        32,
    )
    _refresh(torch_launch)
    torch_ms = triton.testing.do_bench(torch_launch, warmup=warmup, rep=iters, return_mode="median")

    role_tflops = _tflops(m, n, k, role_ms)
    torch_tflops = _tflops(m, n, k, torch_ms)
    speedup = torch_ms / role_ms

    print(f"\n=== MXFP8 GEMM {m}x{n}x{k} ===")
    print(f"role_v2:      {role_ms:.3f} ms, {role_tflops:.1f} TFLOPS")
    print(f"torch F.scaled_mm: {torch_ms:.3f} ms, {torch_tflops:.1f} TFLOPS")
    print(f"speedup:      {speedup:.2f}x (torch / role_v2)")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark role-proposal SM100 MXFP8 blockscaled GEMM")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_mxfp8_gemm")
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
