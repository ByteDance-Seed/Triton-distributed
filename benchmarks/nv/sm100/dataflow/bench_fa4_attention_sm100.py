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
"""Build, check, and benchmark the role-proposal FA4 v2 kernel."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT, role_example_path, role_example_source

import argparse
import math
import os
import sys
import time
import types
import warnings
from functools import lru_cache
from pathlib import Path

import torch
import triton.testing

from little_kernel.runtime import CompiledKernel
from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor, create_tma_3d_descriptor
from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
from benchmarks.nv.sm100.dataflow.bench_optional import add_reference_argument, optional_import

FA4_KERNEL = role_example_source("fa4")
TEST_SHAPES = [(1, 8192), (4, 16384), (8, 527), (17, 1023)]
BENCH_SHAPES = TEST_SHAPES
REFRESH_SECONDS = 0.4
_FA4 = None


@lru_cache(maxsize=1)
def _cfg() -> dict:
    ns: dict = {}
    exec(compile(FA4_KERNEL, "<role-v2-fa4>", "exec"), ns)
    return ns


def _shared_mem_bytes() -> int:
    return int(_cfg()["SMEM_TOTAL"])


def build_kernel(*, cache_dir: str | None, verbose: bool = False) -> CompiledKernel:
    c = _cfg()
    code = compile_dataflow_cuda(FA4_KERNEL, num_threads=c["NUM_THREADS"], disable_var_hoisting=True)
    return CompiledKernel(
        code,
        "fa4_sm100_kernel",
        grid=(1, 1, 1),
        block=(c["NUM_THREADS"], 1, 1),
        shared_mem_bytes=_shared_mem_bytes(),
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        # A 512-thread block must fit the SM100 64K register file. Bound
        # ptxas allocation to 128 registers per thread; native correctness
        # tests validate the compiled binary and its attention results.
        extra_nvcc_options=["-DNDEBUG", "--use_fast_math", "--maxrregcount=128"],
    )


def _validate_fixed(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    c = _cfg()
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16 and v.dtype == torch.bfloat16
    assert q.shape[-2:] == (c["H_Q"], c["HEAD_DIM"])
    assert k.shape[-2:] == (c["H_KV"], c["HEAD_DIM"])
    assert v.shape[-2:] == (c["H_KV"], c["HEAD_DIM"])
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()


def _flatten_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if q.dim() != 4:
        raise ValueError("q/k/v must be [B, T, H, D]")
    batch, seqlen, hq, d = q.shape
    _, _, hkv, _ = k.shape
    return (
        q.reshape(batch * seqlen, hq, d),
        k.reshape(batch * seqlen, hkv, d),
        v.reshape(batch * seqlen, hkv, d),
        batch,
        seqlen,
    )


def prepare_fa4(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    _validate_fixed(q, k, v)
    c = _cfg()
    _, kf, _, batch, seqlen = _flatten_qkv(q, k, v)

    seqlen_tma = ((seqlen + c["BN"] - 1) // c["BN"]) * c["BN"]
    total_tma = batch * seqlen_tma
    m_pack_tma = seqlen_tma * c["KV_GROUP"]

    if seqlen_tma == seqlen:
        q_src = q
        k_tma = kf
        v_tma = v.reshape(total_tma, c["H_KV"], c["HEAD_DIM"])
    else:
        # TMA loads entire tiles. Masked zero probabilities still multiply V
        # padding, so uninitialized NaNs there would contaminate valid rows.
        q_pad = torch.zeros((batch, seqlen_tma, c["H_Q"], c["HEAD_DIM"]), device=q.device, dtype=q.dtype)
        k_pad = torch.zeros((batch, seqlen_tma, c["H_KV"], c["HEAD_DIM"]), device=k.device, dtype=k.dtype)
        v_pad = torch.zeros((batch, seqlen_tma, c["H_KV"], c["HEAD_DIM"]), device=v.device, dtype=v.dtype)
        q_pad[:, :seqlen] = q
        k_pad[:, :seqlen] = k
        v_pad[:, :seqlen] = v
        q_src = q_pad
        k_tma = k_pad.reshape(total_tma, c["H_KV"], c["HEAD_DIM"])
        v_tma = v_pad.reshape(total_tma, c["H_KV"], c["HEAD_DIM"])

    q_pack = torch.empty((batch, c["H_KV"], m_pack_tma, c["HEAD_DIM"]), device=q.device, dtype=q.dtype)
    for kvh in range(c["H_KV"]):
        h0 = kvh * c["KV_GROUP"]
        h1 = h0 + c["KV_GROUP"]
        q_pack[:, kvh] = q_src[:, :, h0:h1, :].reshape(batch, m_pack_tma, c["HEAD_DIM"])
    out_pack = torch.empty_like(q_pack)

    tma_q = create_tma_2d_descriptor(
        q_pack.reshape(batch * c["H_KV"] * m_pack_tma, c["HEAD_DIM"]),
        gmem_inner_dim=c["HEAD_DIM"],
        gmem_outer_dim=batch * c["H_KV"] * m_pack_tma,
        smem_inner_dim=c["TMA_COLS"],
        smem_outer_dim=c["BM"],
        gmem_outer_stride=c["HEAD_DIM"],
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    tma_k = create_tma_3d_descriptor(
        k_tma,
        dim0=c["HEAD_DIM"],
        dim1=total_tma,
        dim2=c["H_KV"],
        stride1_bytes=c["H_KV"] * c["HEAD_DIM"] * 2,
        stride2_bytes=c["HEAD_DIM"] * 2,
        tile_dim0=c["TMA_COLS"],
        tile_dim1=c["BN"],
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    tma_v = create_tma_3d_descriptor(
        v_tma,
        dim0=c["HEAD_DIM"],
        dim1=total_tma,
        dim2=c["H_KV"],
        stride1_bytes=c["H_KV"] * c["HEAD_DIM"] * 2,
        stride2_bytes=c["HEAD_DIM"] * 2,
        tile_dim0=c["TMA_COLS"],
        tile_dim1=c["BN"],
        swizzle_mode=128,
        oob_fill=True,
        l2_promotion=3,
    )
    tma_o = create_tma_2d_descriptor(
        out_pack.reshape(batch * c["H_KV"] * m_pack_tma, c["HEAD_DIM"]),
        gmem_inner_dim=c["HEAD_DIM"],
        gmem_outer_dim=batch * c["H_KV"] * m_pack_tma,
        smem_inner_dim=c["STORE_BN"],
        smem_outer_dim=c["BM"],
        gmem_outer_stride=c["HEAD_DIM"],
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )

    scale_log2 = (1.0 / math.sqrt(c["HEAD_DIM"])) * math.log2(math.e)
    num_m_pack_blocks = (seqlen * c["KV_GROUP"] + c["BM"] - 1) // c["BM"]
    num_q_stage_blocks = (num_m_pack_blocks + c["Q_STAGE"] - 1) // c["Q_STAGE"]
    total_work_tiles = num_q_stage_blocks * c["H_KV"] * batch
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid = (min(total_work_tiles, num_sms), 1, 1)
    args = (tma_q, tma_k, tma_v, tma_o, batch, seqlen, seqlen_tma, m_pack_tma, scale_log2)
    keepalive = (q_pack, k_tma, v_tma)
    return args, grid, out_pack, batch, seqlen, seqlen_tma, keepalive


def run_prepared_fa4(kernel: CompiledKernel, prepared):
    c = _cfg()
    args, grid, out_pack, batch, seqlen, seqlen_tma, *_keepalive = prepared
    kernel(*args, grid=grid)
    out_view = out_pack.view(batch, c["H_KV"], seqlen_tma, c["KV_GROUP"], c["HEAD_DIM"])[:, :, :seqlen]
    out = torch.empty((batch, seqlen, c["H_Q"], c["HEAD_DIM"]), device=out_pack.device, dtype=out_pack.dtype)
    for kvh in range(c["H_KV"]):
        h0 = kvh * c["KV_GROUP"]
        h1 = h0 + c["KV_GROUP"]
        out[:, :, h0:h1, :] = out_view[:, kvh]
    return out.contiguous()


def flashattention_cute_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor | None = None):
    warn_orig = warnings.warn

    def warn_filtered(message, *args, **kwargs):
        if str(message).startswith("Use explicit `struct.scalar.ptr` for pointer instead"):
            return
        return warn_orig(message, *args, **kwargs)

    try:
        warnings.warn = warn_filtered
        out_ref, _, _, _ = _FA4._flash_attn_fwd(q, k, v, causal=True, pack_gqa=True, return_lse=False, out=out)
    finally:
        warnings.warn = warn_orig
    return out_ref.contiguous()


def reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    if _FA4 is not None:
        return flashattention_cute_reference(q, k, v)
    import torch.nn.functional as F

    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True,
                                         enable_gqa=True)
    return out.transpose(1, 2).contiguous()


def _inputs(batch: int, seqlen: int):
    c = _cfg()
    torch.manual_seed(2026 + batch + seqlen)
    q = torch.randn((batch, seqlen, c["H_Q"], c["HEAD_DIM"]), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((batch, seqlen, c["H_KV"], c["HEAD_DIM"]), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((batch, seqlen, c["H_KV"], c["HEAD_DIM"]), device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _refresh(run) -> None:
    torch.cuda.synchronize()
    time.sleep(REFRESH_SECONDS)
    run()
    torch.cuda.synchronize()


def check_correctness(kernel: CompiledKernel, shape: tuple[int, int]) -> None:
    batch, seqlen = shape
    q, k, v = _inputs(batch, seqlen)
    ref = reference(q, k, v)
    out = run_prepared_fa4(kernel, prepare_fa4(q, k, v))
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    print(f"correctness: B={batch} T={seqlen} pass")


def bench_compare(kernel: CompiledKernel, shape: tuple[int, int], *, warmup: int, iters: int) -> None:
    c = _cfg()
    batch, seqlen = shape
    q, k, v = _inputs(batch, seqlen)
    prepared = prepare_fa4(q, k, v)
    args, grid = prepared[0], prepared[1]

    role_launch = lambda: kernel(*args, grid=grid)
    _refresh(role_launch)
    role_ms = triton.testing.do_bench(role_launch, warmup=warmup, rep=iters, return_mode="median")

    fa4_out = torch.empty_like(q)
    fa4_launch = lambda: flashattention_cute_reference(q, k, v, out=fa4_out)
    if _FA4 is None:
        flops = 2.0 * batch * c["H_Q"] * seqlen * seqlen * c["HEAD_DIM"]
        role_tflops = flops / (role_ms / 1e3) / 1e12
        print(f"\n=== FA4 B={batch} T={seqlen} ===")
        print(f"role_v2: {role_ms:.3f} ms, {role_tflops:.1f} TFLOP/s")
        print("fa4_ref disabled/unavailable; correctness checked with PyTorch SDPA")
        return
    _refresh(fa4_launch)
    fa4_ms = triton.testing.do_bench(fa4_launch, warmup=warmup, rep=iters, return_mode="median")

    flops = 2.0 * batch * c["H_Q"] * seqlen * seqlen * c["HEAD_DIM"]
    role_tflops = flops / (role_ms / 1e3) / 1e12
    fa4_tflops = flops / (fa4_ms / 1e3) / 1e12
    print(f"\n=== FA4 B={batch} T={seqlen} ===")
    print(f"role_v2: {role_ms:.3f} ms, {role_tflops:.1f} TFLOP/s")
    print(f"fa4_ref: {fa4_ms:.3f} ms, {fa4_tflops:.1f} TFLOP/s")
    print(f"speedup: {fa4_ms / role_ms:.3f}x (fa4_ref / role_v2)")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark role-proposal FA4 v2")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_role_v2_fa4")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--verbose", action="store_true")
    add_reference_argument(p)
    args = p.parse_args()

    global _FA4
    if "flash_attn.flash_attn_interface" not in sys.modules:
        stub = types.ModuleType("flash_attn.flash_attn_interface")
        for name in (
                "flash_attn_func",
                "flash_attn_varlen_func",
                "flash_attn_qkvpacked_func",
                "flash_attn_varlen_qkvpacked_func",
                "flash_attn_kvpacked_func",
                "flash_attn_varlen_kvpacked_func",
                "flash_attn_with_kvcache",
        ):
            setattr(stub, name, None)
        sys.modules["flash_attn.flash_attn_interface"] = stub

    _FA4 = optional_import("flash_attn.cute.interface", "FLASH_ATTN_PYTHON_PATH", mode=args.references)

    kernel = build_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    print(f"cache_dir: {args.cache_dir}")

    if not args.no_check:
        for shape in TEST_SHAPES:
            check_correctness(kernel, shape)

    for shape in BENCH_SHAPES:
        bench_compare(kernel, shape, warmup=args.warmup, iters=args.iters)


if __name__ == "__main__":
    main()
