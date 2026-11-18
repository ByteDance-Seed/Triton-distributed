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
"""Build, check, and benchmark the NVFP4 blockscaled GEMM on SM100."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT, role_example_path, role_example_source

import argparse
import time
from pathlib import Path

import torch
import triton.testing

from benchmarks.nv.sm100.dataflow.bench_blockscaled import pack_torch_scales, torch_blockscaled_mm
from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor
from examples.nv.sm100.nvfp4_gemm import A_SMEM_K, CLUSTER_SIZE, CTA_GROUP_SHAPE, CTA_TILE_M, CTA_TILE_N_SHARD, K_TILE, NUM_STAGES, NUM_TMA_STORE_STAGES, PAIR_TILE_M, PAIR_TILE_N, SF_SMEM_M, SFA_SMEM_K, SFB_SMEM_K, SMEM_A, SMEM_B, SMEM_SFA, SMEM_SFB, ST_TILE_M, ST_TILE_N

TEST_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]
BENCH_SHAPES = [(8192, 8192, 8192)]
REFRESH_SECONDS = 0.4
VS = 16  # vector size: one SF per 16 K elements

# E2M1 positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
_E2M1_POS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _float_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Quantize float tensor to E2M1 4-bit values (uint8, 0-15)."""
    sign = (x < 0).to(torch.uint8) * 8
    ax = x.abs()
    enc = torch.zeros_like(ax, dtype=torch.uint8)
    enc = torch.where(ax >= 0.25, torch.tensor(1, dtype=torch.uint8), enc)
    enc = torch.where(ax >= 0.75, torch.tensor(2, dtype=torch.uint8), enc)
    enc = torch.where(ax >= 1.25, torch.tensor(3, dtype=torch.uint8), enc)
    enc = torch.where(ax >= 1.75, torch.tensor(4, dtype=torch.uint8), enc)
    enc = torch.where(ax >= 2.5, torch.tensor(5, dtype=torch.uint8), enc)
    enc = torch.where(ax >= 3.5, torch.tensor(6, dtype=torch.uint8), enc)
    enc = torch.where(ax >= 5.0, torch.tensor(7, dtype=torch.uint8), enc)
    return enc | sign


def _e2m1_to_float(e2m1: torch.Tensor) -> torch.Tensor:
    """Convert E2M1 4-bit values to float32."""
    sign = (e2m1 >> 3) & 1
    mag = (e2m1 & 7).long()
    vals = _E2M1_POS.to(e2m1.device)[mag]
    return torch.where(sign == 1, -vals, vals)


def quantize_nvfp4(x: torch.Tensor, *, duplicate_for_2sm: bool, vs: int = VS):
    """Quantize (M, K) float tensor to NVFP4.

    Returns:
        data: (M, K//2) uint8 — E2M1 packed 2-per-byte (low=even, high=odd)
        sf: E4M3 scales in the CUTLASS SM100 shared-memory order
        torch_sf: E4M3 scales packed for torch.nn.functional.scaled_mm
    """
    M, K = x.shape
    assert K % vs == 0
    x_f32 = x.float()
    x_grouped = x_f32.reshape(M, K // vs, vs)
    amax = x_grouped.abs().amax(dim=-1).clamp(min=1e-12)  # (M, K//vs)
    scale = amax / 6.0  # E2M1 max = 6
    sf_e4m3 = scale.to(torch.float8_e4m3fn)  # (M, K//vs)
    sf_float = sf_e4m3.float().unsqueeze(-1)  # (M, K//vs, 1)
    x_scaled = x_grouped / sf_float
    e2m1 = _float_to_e2m1(x_scaled).reshape(M, K)  # (M, K) uint8 0-15
    # Pack: even K → low nibble, odd K → high nibble
    packed = (e2m1[:, 0::2] | (e2m1[:, 1::2] << 4)).to(torch.uint8).contiguous()
    # Interleaved SF layout
    sf_u8 = sf_e4m3.view(torch.uint8).reshape(M, K // vs)
    sf_interleaved = _make_interleaved_sf(sf_u8, M, K, vs, duplicate_for_2sm)
    torch_sf = pack_torch_scales(sf_u8, torch.float8_e4m3fn)
    return packed, sf_interleaved, torch_sf


def _make_interleaved_sf(sf_u8: torch.Tensor, M: int, K: int, vs: int, duplicate_for_2sm: bool) -> torch.Tensor:
    """Arrange logical scales as CUTLASS's compact 32-row SM100 SF tiles."""
    num_k_tiles = K // K_TILE
    dev = sf_u8.device
    m_idx = torch.arange(M, device=dev)
    k0_idx = torch.arange(K // vs, device=dev)
    mg, kg = torch.meshgrid(m_idx, k0_idx, indexing="ij")
    tile = (mg // PAIR_TILE_M) * num_k_tiles + kg // (K_TILE // vs)
    if duplicate_for_2sm:
        row = tile * SF_SMEM_M * CLUSTER_SIZE + mg % SF_SMEM_M
        col = (kg % 16 // 4) * 32 + (mg % PAIR_TILE_M // 32) * 4 + kg % 4
        out = torch.empty(M * K // 1024, SFB_SMEM_K, dtype=torch.uint8, device=dev)
        out[row, col] = sf_u8
        out[row + SF_SMEM_M, col] = sf_u8
        return out

    row = (tile * CLUSTER_SIZE + mg % PAIR_TILE_M // CTA_TILE_M) * SF_SMEM_M + mg % SF_SMEM_M
    col = (kg % 16 // 4) * 16 + (mg % CTA_TILE_M // 32) * 4 + kg % 4
    out = torch.empty(M * K // 1024, SFA_SMEM_K, dtype=torch.uint8, device=dev)
    out[row, col] = sf_u8
    return out


def dequantize_nvfp4(data: torch.Tensor, sf_interleaved: torch.Tensor, M: int, K: int, *, duplicate_for_2sm: bool,
                     vs: int = VS) -> torch.Tensor:
    """Dequantize NVFP4 to float32 for reference computation."""
    e2m1_even = data & 0xF
    e2m1_odd = (data >> 4) & 0xF
    e2m1 = torch.stack([e2m1_even, e2m1_odd], dim=-1).reshape(M, K)
    x_val = _e2m1_to_float(e2m1)  # (M, K)
    # Recover SF from interleaved layout
    num_k_tiles = K // K_TILE
    dev = data.device
    m_idx = torch.arange(M, device=dev)
    k0_idx = torch.arange(K // vs, device=dev)
    mg, kg = torch.meshgrid(m_idx, k0_idx, indexing="ij")
    tile = (mg // PAIR_TILE_M) * num_k_tiles + kg // (K_TILE // vs)
    if duplicate_for_2sm:
        row = tile * SF_SMEM_M * CLUSTER_SIZE + mg % SF_SMEM_M
        col = (kg % 16 // 4) * 32 + (mg % PAIR_TILE_M // 32) * 4 + kg % 4
    else:
        row = (tile * CLUSTER_SIZE + mg % PAIR_TILE_M // CTA_TILE_M) * SF_SMEM_M + mg % SF_SMEM_M
        col = (kg % 16 // 4) * 16 + (mg % CTA_TILE_M // 32) * 4 + kg % 4
    sf_u8 = sf_interleaved[row, col]  # (M, K//vs)
    sf_f32 = sf_u8.view(torch.float8_e4m3fn).float().unsqueeze(-1)  # (M, K//vs, 1)
    return x_val.reshape(M, K // vs, vs) * sf_f32


def _shared_mem_bytes() -> int:
    c_smem = NUM_TMA_STORE_STAGES * ST_TILE_M * ST_TILE_N * 2
    return c_smem + NUM_STAGES * (SMEM_A + SMEM_B + SMEM_SFA + SMEM_SFB) + 1024


def build_kernel(*, cache_dir: str | None = None, verbose: bool = False) -> CompiledKernel:
    from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda

    source = role_example_source("nvfp4")
    code = compile_dataflow_cuda(source)
    return CompiledKernel(
        code,
        "nvfp4_gemm_sm100_kernel",
        grid=(1, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=_shared_mem_bytes(),
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=(CLUSTER_SIZE, 1, 1),
        cta_group_shape=CTA_GROUP_SHAPE,
        extra_nvcc_options=["-lineinfo"],
        allow_non_portable_cluster=True,
    )


def _descriptors(a_packed, b_packed, sfa, sfb, d, m, n, k):
    k2 = k // 2
    dA = create_tma_2d_descriptor(
        a_packed,
        gmem_inner_dim=k2,
        gmem_outer_dim=m,
        smem_inner_dim=A_SMEM_K,
        smem_outer_dim=CTA_TILE_M,
        gmem_outer_stride=k2,
        swizzle_mode=A_SMEM_K,
        oob_fill=False,
        l2_promotion=3,
    )
    dB = create_tma_2d_descriptor(
        b_packed,
        gmem_inner_dim=k2,
        gmem_outer_dim=n,
        smem_inner_dim=A_SMEM_K,
        smem_outer_dim=CTA_TILE_N_SHARD,
        gmem_outer_stride=k2,
        swizzle_mode=A_SMEM_K,
        oob_fill=False,
        l2_promotion=3,
    )
    sf_outer_a = m * k // 1024
    sf_outer_b = n * k // 1024
    dSFA = create_tma_2d_descriptor(
        sfa,
        gmem_inner_dim=SFA_SMEM_K,
        gmem_outer_dim=sf_outer_a,
        smem_inner_dim=SFA_SMEM_K,
        smem_outer_dim=SF_SMEM_M,
        gmem_outer_stride=SFA_SMEM_K,
        swizzle_mode=SFA_SMEM_K,
        oob_fill=False,
        l2_promotion=3,
    )
    dSFB = create_tma_2d_descriptor(
        sfb,
        gmem_inner_dim=SFB_SMEM_K,
        gmem_outer_dim=sf_outer_b,
        smem_inner_dim=SFB_SMEM_K,
        smem_outer_dim=SF_SMEM_M,
        gmem_outer_stride=SFB_SMEM_K,
        swizzle_mode=SFB_SMEM_K,
        oob_fill=False,
        l2_promotion=3,
    )
    dD = create_tma_2d_descriptor(
        d,
        gmem_inner_dim=n,
        gmem_outer_dim=m,
        smem_inner_dim=ST_TILE_N,
        smem_outer_dim=ST_TILE_M,
        gmem_outer_stride=n,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )
    return dA, dB, dSFA, dSFB, dD


def _num_ctas(m: int, n: int) -> int:
    tiles = ((m + PAIR_TILE_M - 1) // PAIR_TILE_M) * ((n + PAIR_TILE_N - 1) // PAIR_TILE_N)
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    return (min(sms, tiles * CLUSTER_SIZE) // CLUSTER_SIZE) * CLUSTER_SIZE


def make_launch(kernel: CompiledKernel, a_packed, b_packed, sfa, sfb, m, n, k):
    d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    dA, dB, dSFA, dSFB, dD = _descriptors(a_packed, b_packed, sfa, sfb, d, m, n, k)
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
    a_packed, sfa, torch_sfa = quantize_nvfp4(a, duplicate_for_2sm=False)
    b_packed, sfb, torch_sfb = quantize_nvfp4(b, duplicate_for_2sm=True)
    launch, out = make_launch(kernel, a_packed, b_packed, sfa, sfb, m, n, k)
    launch()
    torch.cuda.synchronize()
    # Reference: dequantize then matmul
    a_dq = dequantize_nvfp4(a_packed, sfa, m, k, duplicate_for_2sm=False).reshape(m, k)
    b_dq = dequantize_nvfp4(b_packed, sfb, n, k, duplicate_for_2sm=True).reshape(n, k)
    ref = torch.matmul(a_dq, b_dq.t()).to(torch.bfloat16)
    torch_ref = torch_blockscaled_mm(
        a_packed.view(torch.float4_e2m1fn_x2),
        b_packed.view(torch.float4_e2m1fn_x2).t(),
        torch_sfa,
        torch_sfb,
        16,
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
    a_packed, sfa, torch_sfa = quantize_nvfp4(a, duplicate_for_2sm=False)
    b_packed, sfb, torch_sfb = quantize_nvfp4(b, duplicate_for_2sm=True)

    launch, _ = make_launch(kernel, a_packed, b_packed, sfa, sfb, m, n, k)
    _refresh(launch)
    role_ms = triton.testing.do_bench(launch, warmup=warmup, rep=iters, return_mode="median")

    torch_launch = lambda: torch_blockscaled_mm(
        a_packed.view(torch.float4_e2m1fn_x2),
        b_packed.view(torch.float4_e2m1fn_x2).t(),
        torch_sfa,
        torch_sfb,
        16,
    )
    _refresh(torch_launch)
    torch_ms = triton.testing.do_bench(torch_launch, warmup=warmup, rep=iters, return_mode="median")

    role_tflops = _tflops(m, n, k, role_ms)
    torch_tflops = _tflops(m, n, k, torch_ms)
    speedup = torch_ms / role_ms

    print(f"\n=== NVFP4 GEMM {m}x{n}x{k} ===")
    print(f"role_v2:      {role_ms:.3f} ms, {role_tflops:.1f} TFLOPS")
    print(f"torch F.scaled_mm: {torch_ms:.3f} ms, {torch_tflops:.1f} TFLOPS")
    print(f"speedup:      {speedup:.2f}x (torch / role_v2)")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark NVFP4 blockscaled GEMM")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()

    kernel = build_kernel(verbose=args.verbose)
    if args.build_only:
        print("build: pass")
        return

    if not args.no_check:
        for shape in TEST_SHAPES:
            check_correctness(kernel, shape)
    for shape in BENCH_SHAPES:
        bench_compare(kernel, shape, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
