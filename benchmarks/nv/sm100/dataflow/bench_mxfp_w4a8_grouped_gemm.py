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
"""Build, check, and benchmark SM100 MXFP8-by-MXFP4 grouped GEMM against DeepGEMM."""

from __future__ import annotations
from benchmarks.nv.sm100.dataflow._paths import PYTHON_INCLUDE_ROOT, REPO_ROOT, role_example_path, role_example_source

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

import torch
import triton.testing

from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda
from little_kernel.runtime import CompiledKernel
from benchmarks.nv.sm100.dataflow.bench_optional import add_reference_argument, optional_import
from little_kernel.backends.nv.common.runtime.tma_descriptor import CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B, create_tma_2d_descriptor, create_tma_3d_descriptor

KERNEL_PATH = role_example_path("w4a8_groupgemm")
KERNEL_NAME = "mxfp_w4a8_groupgemm_sm100_kernel"
REFRESH_SECONDS = 0.4
E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _kernel_config() -> tuple[str, dict]:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    namespace: dict = {}
    exec(compile(source, str(KERNEL_PATH), "exec"), namespace)
    return source, namespace


def build_kernel(*, cache_dir: str | None, verbose: bool) -> CompiledKernel:
    source, config = _kernel_config()
    kernel = config[KERNEL_NAME]
    return CompiledKernel(
        compile_dataflow_cuda(source, num_threads=config["NUM_THREADS"]),
        KERNEL_NAME,
        grid=(1, 1, 1),
        block=(config["NUM_THREADS"], 1, 1),
        shared_mem_bytes=config["SHARED_MEM_BYTES"],
        arch="sm_100a",
        cache_dir=cache_dir,
        verbose=verbose,
        include_paths=[PYTHON_INCLUDE_ROOT],
        cluster_shape=kernel.cluster_shape,
        cta_group_shape=kernel.cta_group_shape,
        allow_non_portable_cluster=True,
    )


def _load_deep_gemm(root: Path | None, mode="auto"):
    candidates = sorted((root / "build").glob("lib.*"), reverse=True) + [root] if root else ()
    return optional_import("deep_gemm", "DEEPGEMM_PYTHON_PATH", candidates, mode=mode)


def _pad_expert_rows(a: torch.Tensor, alignment: int) -> tuple[torch.Tensor, int]:
    experts, rows, k = a.shape
    padded_rows = (rows + alignment - 1) // alignment * alignment
    out = torch.zeros((experts * padded_rows, k), dtype=a.dtype, device=a.device)
    for expert in range(experts):
        out[expert * padded_rows:expert * padded_rows + rows].copy_(a[expert])
    return out, padded_rows


def _little_launch(
    kernel: CompiledKernel,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    use_worklist: bool,
    experts: int,
    n: int,
    k: int,
) -> tuple[Callable[[], None], torch.Tensor, int]:
    _, config = _kernel_config()
    padded_m = a.shape[0]
    rows_per_expert = padded_m // experts
    if rows_per_expert % config["PAIR_TILE_M"]:
        raise ValueError(f"LittleKernel rows per expert must be divisible by {config['PAIR_TILE_M']}")

    sfa = torch.full((padded_m * k // 256, 32), 127, dtype=torch.uint8, device=a.device)
    sfb = torch.full((experts, n * k // 256, 32), 127, dtype=torch.uint8, device=a.device)
    out = torch.empty((padded_m, n), dtype=torch.bfloat16, device=a.device)
    blocks_per_expert = rows_per_expert // config["PAIR_TILE_M"]
    num_n_blocks = n // config["PAIR_TILE_N"]
    num_work = experts * blocks_per_expert * num_n_blocks
    if use_worklist:
        work = [(expert, expert * blocks_per_expert + m_block, n_block)
                for expert in range(experts)
                for m_block in range(blocks_per_expert)
                for n_block in range(num_n_blocks)]
        work_experts = torch.tensor([item[0] for item in work], dtype=torch.int32, device=a.device)
        work_m_blocks = torch.tensor([item[1] for item in work], dtype=torch.int32, device=a.device)
        work_n_blocks = torch.tensor([item[2] for item in work], dtype=torch.int32, device=a.device)
    else:
        work_experts = work_m_blocks = work_n_blocks = torch.empty(1, dtype=torch.int32, device=a.device)
    num_clusters = min(torch.cuda.get_device_properties(a.device).multi_processor_count // 2, num_work)

    descriptor_args = dict(oob_fill=False, l2_promotion=3)
    d_a = create_tma_2d_descriptor(a, k, padded_m, config["K_TILE"], config["CTA_TILE_M"], k, swizzle_mode=128,
                                   **descriptor_args)
    d_b = create_tma_3d_descriptor(
        b,
        k,
        n,
        experts,
        k // 2,
        n * k // 2,
        config["K_TILE"],
        config["CTA_TILE_N_SHARD"],
        swizzle_mode=128,
        tensor_map_dtype=CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
        **descriptor_args,
    )
    d_sfa = create_tma_2d_descriptor(sfa, 32, padded_m * k // 256, 32, 32, 32, swizzle_mode=32, **descriptor_args)
    d_sfb = create_tma_3d_descriptor(
        sfb,
        32,
        n * k // 256,
        experts,
        32,
        n * k // 256 * 32,
        32,
        32,
        swizzle_mode=32,
        **descriptor_args,
    )
    d_out = create_tma_2d_descriptor(out, n, padded_m, config["ST_TILE_N"], config["ST_TILE_M"], n, swizzle_mode=128,
                                     **descriptor_args)
    keepalive = (a, b, sfa, sfb, out, work_experts, work_m_blocks, work_n_blocks)

    def launch() -> None:
        _ = keepalive
        kernel(
            d_a,
            d_b,
            d_sfa,
            d_sfb,
            d_out,
            work_experts,
            work_m_blocks,
            work_n_blocks,
            k,
            num_work,
            num_clusters,
            blocks_per_expert,
            num_n_blocks,
            int(use_worklist),
            grid=(num_clusters * 2, 1, 1),
        )

    return launch, out, rows_per_expert


def _deep_gemm_launch(
    deep_gemm,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    experts: int,
    valid_rows: int,
    n: int,
    k: int,
    alignment: int,
) -> tuple[Callable[[], None], torch.Tensor, int]:
    deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)
    padded_m = a.shape[0]
    padded_rows = padded_m // experts
    a_stride = deep_gemm.get_tma_aligned_size(padded_m, 4)
    b_stride = deep_gemm.get_tma_aligned_size(n, 4)
    sfa = torch.empty_strided((padded_m, k // 128), (1, a_stride), dtype=torch.int32, device=a.device)
    sfb = torch.empty_strided((experts, n, k // 128), (b_stride * (k // 128), 1, b_stride), dtype=torch.int32,
                              device=a.device)
    sfa.fill_(0x7F7F7F7F)
    sfb.fill_(0x7F7F7F7F)
    out = torch.empty((padded_m, n), dtype=torch.bfloat16, device=a.device)
    grouped_layout = torch.full((padded_m, ), -1, dtype=torch.int32, device=a.device)
    for expert in range(experts):
        start = expert * padded_rows
        grouped_layout[start:start + valid_rows] = expert

    def launch() -> None:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
            (a, sfa),
            (b, sfb),
            out,
            grouped_layout,
            recipe_a=(1, 32),
            recipe_b=(1, 32),
            ensure_zero_padding=False,
        )

    return launch, out, padded_rows


def _valid_rows(out: torch.Tensor, *, experts: int, rows: int, padded_rows: int) -> torch.Tensor:
    return torch.cat([out[expert * padded_rows:expert * padded_rows + rows] for expert in range(experts)])


def check_correctness(kernel: CompiledKernel, deep_gemm, *, alignment: int) -> None:
    experts, rows, n, k = 2, 256, 256, 128
    generator = torch.Generator(device="cuda")
    generator.manual_seed(2026)
    a = torch.randn((experts, rows, k), dtype=torch.float32, device="cuda", generator=generator).to(torch.float8_e4m3fn)
    b_bits = torch.randint(0, 16, (experts, n, k), dtype=torch.uint8, device="cuda", generator=generator)
    b = (b_bits[..., 0::2] | b_bits[..., 1::2] << 4).contiguous().view(torch.int8)
    b_float = E2M1_LUT.to(device=b.device)[b_bits.long()]
    reference = torch.cat([a[expert].float() @ b_float[expert].t() for expert in range(experts)]).to(torch.bfloat16)

    little_a, _ = _pad_expert_rows(a, 256)
    worklist_launch, worklist_out, little_rows = _little_launch(kernel, little_a, b, use_worklist=True, experts=experts,
                                                                n=n, k=k)
    online_launch, online_out, _ = _little_launch(kernel, little_a, b, use_worklist=False, experts=experts, n=n, k=k)
    worklist_launch()
    online_launch()
    torch.cuda.synchronize()
    worklist_actual = _valid_rows(worklist_out, experts=experts, rows=rows, padded_rows=little_rows)
    online_actual = _valid_rows(online_out, experts=experts, rows=rows, padded_rows=little_rows)
    torch.testing.assert_close(worklist_actual.float(), reference.float(), rtol=2e-2, atol=0.5)
    torch.testing.assert_close(online_actual.float(), reference.float(), rtol=2e-2, atol=0.5)
    torch.testing.assert_close(online_actual, worklist_actual, rtol=0, atol=0)

    if deep_gemm is not None:
        deep_a, _ = _pad_expert_rows(a, alignment)
        deep_launch, deep_out, deep_rows = _deep_gemm_launch(deep_gemm, deep_a, b, experts=experts, valid_rows=rows,
                                                             n=n, k=k, alignment=alignment)
        deep_launch()
        torch.cuda.synchronize()
        deep_actual = _valid_rows(deep_out, experts=experts, rows=rows, padded_rows=deep_rows)
        torch.testing.assert_close(deep_actual.float(), reference.float(), rtol=2e-2, atol=0.5)

    max_abs = float((worklist_actual.float() - reference.float()).abs().max().item())
    print(f"correctness: worklist/online pass and byte-equal; E={experts} M={experts * rows} N={n} K={k} "
          f"max_abs={max_abs:.6g}")


def _bench(run, *, warmup: int, iters: int) -> float:
    torch.cuda.synchronize()
    time.sleep(REFRESH_SECONDS)
    run()
    torch.cuda.synchronize()
    return triton.testing.do_bench(run, warmup=warmup, rep=iters, return_mode="median")


def _bench_ablation(worklist_launch, online_launch, *, warmup: int, iters: int, rounds: int) -> tuple[float, float]:
    worklist_times = []
    online_times = []
    for round_idx in range(rounds):
        if round_idx % 2:
            online_times.append(_bench(online_launch, warmup=warmup, iters=iters))
            worklist_times.append(_bench(worklist_launch, warmup=warmup, iters=iters))
        else:
            worklist_times.append(_bench(worklist_launch, warmup=warmup, iters=iters))
            online_times.append(_bench(online_launch, warmup=warmup, iters=iters))
    return median(worklist_times), median(online_times)


def bench_compare(kernel: CompiledKernel, deep_gemm, args) -> None:
    experts, rows, n, k = args.num_experts, args.rows_per_expert, args.n, args.k
    valid_m = experts * rows
    little_rows = (rows + 255) // 256 * 256
    little_a = torch.empty((experts * little_rows, k), dtype=torch.float8_e4m3fn, device="cuda")
    b = torch.empty((experts, n, k // 2), dtype=torch.int8, device="cuda")
    worklist_launch, _, little_rows = _little_launch(kernel, little_a, b, use_worklist=True, experts=experts, n=n, k=k)
    online_launch, _, _ = _little_launch(kernel, little_a, b, use_worklist=False, experts=experts, n=n, k=k)
    worklist_ms, online_ms = _bench_ablation(worklist_launch, online_launch, warmup=args.warmup, iters=args.iters,
                                             rounds=args.ablation_rounds)

    deep_ms = None
    deep_rows = None
    if deep_gemm is not None:
        deep_rows = (rows + args.deepgemm_alignment - 1) // args.deepgemm_alignment * args.deepgemm_alignment
        deep_a = torch.empty((experts * deep_rows, k), dtype=torch.float8_e4m3fn, device="cuda")
        deep_launch, _, deep_rows = _deep_gemm_launch(
            deep_gemm,
            deep_a,
            b,
            experts=experts,
            valid_rows=rows,
            n=n,
            k=k,
            alignment=args.deepgemm_alignment,
        )
        deep_ms = _bench(deep_launch, warmup=args.warmup, iters=args.iters)

    flops = 2.0 * valid_m * n * k
    num_work = experts * little_rows // 256 * (n // 256)
    print(f"\n=== MXFP W4A8 GroupGEMM E={experts} M={valid_m} N={n} K={k} ===")
    print(f"LK worklist: {worklist_ms:.4f} ms, {flops / (worklist_ms * 1e9):.1f} effective TFLOP/s, "
          f"padded_M={experts * little_rows}")
    print(f"LK online:   {online_ms:.4f} ms, {flops / (online_ms * 1e9):.1f} effective TFLOP/s, "
          f"avoids {num_work * 3 * 4 / 1024:.1f} KiB worklist")
    print(f"online ratio: {worklist_ms / online_ms:.3f}x "
          f"(worklist / online; median of {args.ablation_rounds} paired rounds)")
    if deep_ms is not None:
        print(f"DeepGEMM:    {deep_ms:.4f} ms, {flops / (deep_ms * 1e9):.1f} effective TFLOP/s, "
              f"padded_M={experts * deep_rows}, block_M={args.deepgemm_alignment}")
        print(f"worklist ratio: {deep_ms / worklist_ms:.3f}x (DeepGEMM / LK worklist)")
        print(f"online ratio:   {deep_ms / online_ms:.3f}x (DeepGEMM / LK online)")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-experts", type=int, default=40)
    parser.add_argument("--rows-per-expert", type=int, default=736)
    parser.add_argument("--n", type=int, default=5120)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--deepgemm-root", type=Path,
                        default=Path(os.environ["DEEPGEMM_ROOT"]) if os.environ.get("DEEPGEMM_ROOT") else None)
    parser.add_argument("--deepgemm-alignment", type=int, default=192)
    parser.add_argument("--no-deepgemm", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--ablation-rounds", type=int, default=5)
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--cache-dir", default=f"{os.path.dirname(__file__)}/lk_cache_v2_mxfp_w4a8_groupgemm")
    parser.add_argument("--verbose", action="store_true")
    add_reference_argument(parser)
    args = parser.parse_args()
    if args.num_experts <= 0 or args.rows_per_expert <= 0:
        parser.error("--num-experts and --rows-per-expert must be positive")
    if args.n <= 0 or args.n % 256 or args.k <= 0 or args.k % 128:
        parser.error("--n must be divisible by 256 and --k must be divisible by 128")
    if args.deepgemm_alignment <= 0 or args.deepgemm_alignment > 224 or args.deepgemm_alignment % 32:
        parser.error("--deepgemm-alignment must be a multiple of 32 no greater than 224")
    if args.warmup < 0 or args.iters <= 0 or args.ablation_rounds <= 0:
        parser.error("--warmup must be non-negative; --iters and --ablation-rounds must be positive")
    return args


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("MXFP W4A8 GroupGEMM requires an SM100 GPU")
    deep_gemm = None if args.no_deepgemm else _load_deep_gemm(args.deepgemm_root, args.references)
    kernel = build_kernel(cache_dir=args.cache_dir, verbose=args.verbose)
    _ = kernel.launcher
    print(f"args: {args}")
    if not args.no_check:
        check_correctness(kernel, deep_gemm, alignment=args.deepgemm_alignment)
    bench_compare(kernel, deep_gemm, args)


if __name__ == "__main__":
    main()
