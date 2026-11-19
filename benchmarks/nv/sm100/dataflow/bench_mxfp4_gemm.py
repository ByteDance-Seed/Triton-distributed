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
"""Build, check, and benchmark the MXFP4 blockscaled GEMM on SM100."""

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

KERNEL_SRC = role_example_source("mxfp4")
KERNEL_NAME = "mxfp4_gemm_sm100_kernel"

TEST_SHAPES = [
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
]
BENCH_SHAPES = [(8192, 8192, 8192)]
REFRESH_SECONDS = 0.4

# ── MXFP4 quantization helpers ──────────────────────────────────

# E2M1 (MXFP4) lookup table: bit pattern → float
# Bit pattern: (sign << 3) | (exp << 1) | mant
# exp bias = 1; exp=0 is subnormal (value = mant * 0.5)
_E2M1_LUT = torch.tensor(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # sign=0
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # sign=1
    ],
    dtype=torch.float32,
)


def _e2m1_lut(device: torch.device) -> torch.Tensor:
    return _E2M1_LUT.to(device)


def quantize_mxfp4(x: torch.Tensor, vs: int = 32):
    """Quantize a float tensor to MXFP4 (E2M1 + E8M0 SF, VS=32).

    Returns:
        packed: uint8 tensor, shape (..., K//2), 2 E2M1 nibbles per byte
        sf:     uint8 tensor, shape (M*K//256, 32), E8M0 scales in the kernel's duplicated 2CTA layout
                interleaved layout (M = numel(x) // K)
        torch_sf: E8M0 scales packed for torch.nn.functional.scaled_mm
    """
    assert x.shape[-1] % vs == 0, f"K={x.shape[-1]} not divisible by VS={vs}"
    device = x.device
    K = x.shape[-1]
    num_groups = K // vs

    # Reshape into groups: (..., num_groups, vs)
    x_g = x.reshape(*x.shape[:-1], num_groups, vs)

    # Per-group max abs → E8M0 scale
    max_abs = x_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    # scale = 2^ceil(log2(max_abs / 6))
    log2_s = torch.ceil(torch.log2(max_abs / 6.0))
    e8m0 = (log2_s + 127).clamp(0, 255).to(torch.uint8)  # (..., num_groups, 1)
    scale = torch.pow(2.0, log2_s)  # (..., num_groups, 1)

    # Quantize to E2M1
    x_scaled = x_g / scale
    x_scaled = torch.clamp(x_scaled, -6.0, 6.0)

    # Find nearest E2M1 value via lookup table
    lut = _e2m1_lut(device)  # (16,)
    dist = (x_scaled.unsqueeze(-1) - lut).abs()
    bits = dist.argmin(dim=-1).to(torch.uint8)  # (..., num_groups, vs), 0-15

    # Pack 2 nibbles per byte: even K → low nibble, odd K → high nibble
    # (matches NVFP4 convention and tcgen05.mma E2M1 layout)
    bits_flat = bits.reshape(*x.shape[:-1], K)
    packed = (bits_flat[..., 0::2] | (bits_flat[..., 1::2] << 4)).to(torch.uint8).contiguous()

    # SF: squeeze the last dim → (..., num_groups), then build the CUTLASS
    # interleaved layout expected by the kernel's SF TMA.
    sf = e8m0.squeeze(-1).to(torch.uint8).reshape(-1, num_groups)
    M = x.numel() // K
    sf_interleaved = _make_interleaved_sf(sf, M, K, vs)
    torch_sf = pack_torch_scales(sf, torch.float8_e8m0fnu)

    return packed, sf_interleaved, torch_sf


def _make_interleaved_sf(sf_u8: torch.Tensor, M: int, K: int, vs: int = 32) -> torch.Tensor:
    """Build the 2CTA interleaved SF tensor from logical (M, K//vs) scales.

    CUTLASS blockscaled layout: 128 M-rows packed into 32 rows × 16 bytes
    (4 SF bytes per 16-byte row, one from each block of 32 M-rows).
    Inner dim padded from 16 to 32 for TMA alignment.
    Two K-block scale factors are adjacent for each native K=64 MMA.
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


def dequantize_mxfp4(packed: torch.Tensor, sf_interleaved: torch.Tensor, vs: int = 32) -> torch.Tensor:
    """Dequantize MXFP4 packed data + interleaved E8M0 SF back to float32."""
    device = packed.device
    K2 = packed.shape[-1]
    K = K2 * 2
    num_groups = K // vs
    M = packed.numel() // K2

    # Unpack nibbles: even K in low nibble, odd K in high nibble
    low = packed & 0x0F  # even K elements
    high = (packed >> 4) & 0x0F  # odd K elements
    bits = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], K)

    # E2M1 → float
    lut = _e2m1_lut(device)
    vals = lut[bits.long()]  # (..., K)
    # Recover SF from CUTLASS interleaved layout
    num_k_tiles = K // 128
    m_idx = torch.arange(M, device=device)
    k0_idx = torch.arange(num_groups, device=device)
    mg, kg = torch.meshgrid(m_idx, k0_idx, indexing="ij")
    row = (mg // 256) * num_k_tiles * 128 + (kg // 4) * 128 + (mg // 128) % 2 * 32 + mg % 32
    col = (mg // 32) % 4 * 4 + kg % 4
    sf_u8 = sf_interleaved[row, col].reshape(*packed.shape[:-1], num_groups)
    # E8M0 → scale
    scale = torch.pow(2.0, sf_u8.float() - 127.0)  # (..., num_groups)
    scale = scale.unsqueeze(-1)  # (..., num_groups, 1)

    # Apply scale
    vals_g = vals.reshape(*vals.shape[:-1], num_groups, vs)
    deq = vals_g * scale
    return deq.reshape(*vals.shape)


# ── Kernel build / launch ────────────────────────────────────────


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(KERNEL_SRC, "<v2_mxfp4_gemm>", "exec"), ns)
    return ns


def _shared_mem_bytes() -> int:
    c = _cfg()
    return (c["NUM_TMA_STORE_STAGES"] * c["ST_TILE_M"] * c["ST_TILE_N"] * 2 + c["NUM_STAGES"] *
            (c["SMEM_A"] + c["SMEM_B"] + c["SMEM_SFA"] + c["SMEM_SFB"]) + 1024)


def build_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    code = compile_dataflow_cuda(KERNEL_SRC)
    c = _cfg()
    return CompiledKernel(
        code,
        KERNEL_NAME,
        grid=(1, 1, 1),
        block=(256, 1, 1),
        shared_mem_bytes=_shared_mem_bytes(),
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=c[KERNEL_NAME].cluster_shape,
        cta_group_shape=c[KERNEL_NAME].cta_group_shape,
        extra_nvcc_options=["-lineinfo"],
        allow_non_portable_cluster=True,
    )


def _descriptors(a_pack, b_pack, sfa, sfb, d, m, n, k):
    c = _cfg()
    k_packed = k // 2
    sf_outer_a = m * k // 256
    sf_outer_b = n * k // 256

    # A: (M, K//2) uint8 packed, 32B swizzle
    dA = create_tma_2d_descriptor(
        a_pack,
        gmem_inner_dim=k_packed,
        gmem_outer_dim=m,
        smem_inner_dim=c["A_SMEM_K"],
        smem_outer_dim=c["CTA_TILE_M"],
        gmem_outer_stride=k_packed,
        swizzle_mode=c["A_SMEM_K"],
        oob_fill=False,
        l2_promotion=3,
    )
    # B: (N, K//2) uint8 packed, 32B swizzle
    dB = create_tma_2d_descriptor(
        b_pack,
        gmem_inner_dim=k_packed,
        gmem_outer_dim=n,
        smem_inner_dim=c["A_SMEM_K"],
        smem_outer_dim=c["CTA_TILE_N_SHARD"],
        gmem_outer_stride=k_packed,
        swizzle_mode=c["A_SMEM_K"],
        oob_fill=False,
        l2_promotion=3,
    )
    # SFA: (M*K//128, 32) interleaved E8M0, 32B swizzle
    dSFA = create_tma_2d_descriptor(
        sfa,
        gmem_inner_dim=c["SF_SMEM_K"],
        gmem_outer_dim=sf_outer_a,
        smem_inner_dim=c["SF_SMEM_K"],
        smem_outer_dim=c["SF_SMEM_M"],
        gmem_outer_stride=c["SF_SMEM_K"],
        swizzle_mode=32,
        oob_fill=False,
        l2_promotion=3,
    )
    # SFB: (N*K//128, 32) interleaved E8M0, 32B swizzle
    dSFB = create_tma_2d_descriptor(
        sfb,
        gmem_inner_dim=c["SF_SMEM_K"],
        gmem_outer_dim=sf_outer_b,
        smem_inner_dim=c["SF_SMEM_K"],
        smem_outer_dim=c["SF_SMEM_M"],
        gmem_outer_stride=c["SF_SMEM_K"],
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


def make_launch(kernel, a_pack, b_pack, sfa, sfb, m, n, k):
    d = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    # Packed data is (M, K//2) uint8 — each 32-byte row is one full K-tile
    # (64 E2M1 elements), matching the 32B swizzle atom size.
    dA, dB, dSFA, dSFB, dD = _descriptors(a_pack, b_pack, sfa, sfb, d, m, n, k)
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


def check_correctness(kernel, mnk: tuple[int, int, int]) -> None:
    m, n, k = mnk
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2026 + m + n + k)
    a = torch.randn(m, k, dtype=torch.float32, device="cuda", generator=gen)
    b = torch.randn(n, k, dtype=torch.float32, device="cuda", generator=gen)

    a_pack, sfa, torch_sfa = quantize_mxfp4(a)
    b_pack, sfb, torch_sfb = quantize_mxfp4(b)

    launch, out = make_launch(kernel, a_pack, b_pack, sfa, sfb, m, n, k)
    launch()
    torch.cuda.synchronize()

    # Reference: dequantize then matmul
    a_deq = dequantize_mxfp4(a_pack, sfa)
    b_deq = dequantize_mxfp4(b_pack, sfb)
    ref = torch.matmul(a_deq, b_deq.t()).to(torch.bfloat16)
    torch_ref = torch_blockscaled_mm(
        a_pack.view(torch.float4_e2m1fn_x2),
        b_pack.view(torch.float4_e2m1fn_x2).t(),
        torch_sfa,
        torch_sfb,
        32,
    )

    # MXFP4 has ~4-bit precision; use generous tolerance
    torch.testing.assert_close(out.float(), ref.float(), rtol=0.15, atol=0.15)
    torch.testing.assert_close(torch_ref.float(), ref.float(), rtol=0.15, atol=0.15)
    print(f"correctness: {m}x{n}x{k} pass")


def bench_compare(kernel, mnk: tuple[int, int, int], *, warmup: int, iters: int) -> None:
    m, n, k = mnk
    gen = torch.Generator(device="cuda")
    gen.manual_seed(2026 + m + n + k)
    a = torch.randn(m, k, dtype=torch.float32, device="cuda", generator=gen)
    b = torch.randn(n, k, dtype=torch.float32, device="cuda", generator=gen)

    a_pack, sfa, torch_sfa = quantize_mxfp4(a)
    b_pack, sfb, torch_sfb = quantize_mxfp4(b)

    launch, _ = make_launch(kernel, a_pack, b_pack, sfa, sfb, m, n, k)
    _refresh(launch)
    role_ms = triton.testing.do_bench(launch, warmup=warmup, rep=iters, return_mode="median")

    torch_launch = lambda: torch_blockscaled_mm(
        a_pack.view(torch.float4_e2m1fn_x2),
        b_pack.view(torch.float4_e2m1fn_x2).t(),
        torch_sfa,
        torch_sfb,
        32,
    )
    _refresh(torch_launch)
    torch_ms = triton.testing.do_bench(torch_launch, warmup=warmup, rep=iters, return_mode="median")

    role_tflops = _tflops(m, n, k, role_ms)
    torch_tflops = _tflops(m, n, k, torch_ms)
    speedup = torch_ms / role_ms

    print(f"\n=== MXFP4 GEMM {m}x{n}x{k} ===")
    print(f"role_v2:      {role_ms:.3f} ms, {role_tflops:.1f} TFLOPS")
    print(f"torch F.scaled_mm: {torch_ms:.3f} ms, {torch_tflops:.1f} TFLOPS")
    print(f"speedup:      {speedup:.2f}x (torch / role_v2)")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark MXFP4 blockscaled GEMM on SM100")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_v2_mxfp4_gemm")
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
