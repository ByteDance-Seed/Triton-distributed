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

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

try:  # Optional dependency; this whole gate needs Torch + an SM100 GPU.
    import torch
except ImportError:  # pragma: no cover - exercised only without Torch
    torch = None

from tests.nv.sm100.dataflow._source_paths import PYTHON_ROOT

M, N, K = 2048, 2047, 1025
MAJORS = (("K", "K"), ("MN", "K"), ("MN", "MN"), ("K", "MN"))
# TS A lives in TMEM and is always K-major; tcgen05.cp does not transpose it.
TS_MAJORS = (("K", "K"), ("K", "MN"))


@dataclass(frozen=True)
class Case:
    mode: str
    cta_group: int
    cluster_shape: tuple[int, int, int]
    bm: int
    bn: int
    bk: int
    a_major: str
    b_major: str

    @property
    def name(self) -> str:
        cluster = "x".join(map(str, self.cluster_shape))
        major = f"{self.a_major.lower()}_{self.b_major.lower()}"
        return (f"{self.mode}-cg{self.cta_group}-cl{cluster}-"
                f"m{self.bm}n{self.bn}k{self.bk}-{major}")


def _cluster_shape(size: int, cta_group: int) -> tuple[int, int, int]:
    if cta_group == 2:
        candidates = [x for x in range(2, size + 1, 2) if size % x == 0]
        x = min(candidates, key=lambda value: abs(value - size // value))
    else:
        x = next(value for value in range(math.isqrt(size), 0, -1) if size % value == 0)
        x = size // x
    return x, size // x, 1


def _case(mode: str, cta_group: int, cluster_size: int, bn: int, bk: int, majors: tuple[str, str]) -> Case:
    return Case(mode, cta_group, _cluster_shape(cluster_size, cta_group), 128 * cta_group, bn, bk, *majors)


def _unique(cases) -> tuple[Case, ...]:
    return tuple(dict.fromkeys(cases))


# BM=64/cg1 and BM=128/cg2 need the unsupported 16-dp TMEM epilogue layout.
# TODO(sm100-dp16): add both after ri.tmem_ld_16dp_32bxn supports that layout.
_TILES = ((1, 64, 32), (1, 256, 64), (2, 128, 32), (2, 256, 64))
SEMANTIC_CASES = tuple(
    _case(mode, cg, cg, bn, bk, majors)
    for mode in ("ss", "ts")
    for cg, bn, bk in _TILES
    for majors in (MAJORS if mode == "ss" else TS_MAJORS))
TOPOLOGY_CASES = tuple(
    _case(
        "ts" if size % 3 == 0 else "ss",
        2 if size % 2 == 0 else 1,
        size,
        (64, 128, 256)[(size - 1) % 3],
        (32, 64)[size % 2],
        (("K", "MN") if size % 2 else ("K", "K")) if size % 3 == 0 else MAJORS[(size - 1) % len(MAJORS)],
    ) for size in range(1, 17))
CASES = _unique((*SEMANTIC_CASES, *TOPOLOGY_CASES))
EXHAUSTIVE_CASES = tuple(
    _case(mode, cg, size, bn, bk, majors)
    for size in range(1, 17)
    for cg in ((1, 2) if size % 2 == 0 else (1, ))
    for mode in ("ss", "ts")
    for bn in (64, 128, 256)
    for bk in (32, 64)
    for majors in (MAJORS if mode == "ss" else TS_MAJORS))
CASE_BY_NAME = {case.name: case for case in EXHAUSTIVE_CASES}


def _kernel_source(case: Case) -> str:
    cg = case.cta_group
    cta_m, cta_n = case.bm // cg, case.bn // cg
    a_inter_dim = 1 if case.a_major == "K" else 0
    b_inter_dim = 1 if case.b_major == "K" else 0
    a_offsets = ("(offset_k, ll.int32(group_m * PAIR_TILE_M))"
                 if case.a_major == "K" else "(ll.int32(group_m * PAIR_TILE_M), offset_k)")
    b_offsets = ("(offset_k, ll.int32(group_n * PAIR_TILE_N))"
                 if case.b_major == "K" else "(ll.int32(group_n * PAIR_TILE_N), offset_k)")
    a_tmem = ("        A_tmem = g.tmem(\"A_tmem\", (PAIR_TILE_M, K_TILE), "
              "dtype=ll.bfloat16)\n" if case.mode == "ts" else "")
    if case.mode == "ss":
        mma_body = f"""                ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, 0))
                bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                ri.umma_f16_ss_{cg}sm_kblock(
                    C_tmem, ad, bd, K_TILE,
                    out_shape=(PAIR_TILE_M, PAIR_TILE_N), zero_init=kb == 0)"""
    else:
        mma_body = f"""                for ki in ll.unroll(range(0, K_TILE, 16)):
                    ad: ll.uint64 = ri.make_umma_smem_desc(A_smem, (0, ki))
                    ri.tcgen05_cp_128x256b_{cg}sm(
                        A_tmem, ll.uint32(A_tmem + ki // 2), ad)
                bd: ll.uint64 = ri.make_umma_smem_desc(B_smem, (0, 0))
                ri.umma_f16_ts_{cg}sm_kblock(
                    C_tmem, A_tmem, bd, K_TILE,
                    out_shape=(PAIR_TILE_M, PAIR_TILE_N), zero_init=kb == 0)"""

    return f'''import little_kernel as lk
import little_kernel.language as ll
from little_kernel.backends.nv.sm100.language.dataflow import intrinsics as ri
from little_kernel.backends.nv.sm100.language.dataflow import HW, dataflow

PAIR_TILE_M, PAIR_TILE_N, K_TILE = {case.bm}, {case.bn}, {case.bk}
CTA_GROUP, CTA_TILE_M, CTA_TILE_N = {cg}, {cta_m}, {cta_n}
CLUSTER_SHAPE = {case.cluster_shape!r}
NUM_STAGES = 2
NUM_THREADS = 256
ST_TILE_M, ST_TILE_N = 128, 64
NUM_N_STORES = PAIR_TILE_N // ST_TILE_N
ELEMS_PER_BG = 8

@lk.ll_kernel(
    backend="cuda",
    is_entry=True,
    cluster_shape=CLUSTER_SHAPE,
    cta_group_shape=({cg}, 1, 1),
)
def umma_gemm_config_kernel(
    dA: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dB: ll.const[ll.grid_constant[ll.TmaDescriptor]],
    dD: ll.const[ll.grid_constant[ll.TmaDescriptor]],
) -> ll.void:
    ll.align_memory(1024, scope="dynamic_shared")
    with dataflow() as g:
        load = g.role("load", hw=HW.TMA, warps=[0])
        mma = g.role("mma", hw=HW.TensorCore, warps=[1])
        epi = g.role("epi", hw=HW.CUDACore, warps=[4, 5, 6, 7])
        A_smem = g.smem("A_smem", (CTA_TILE_M, K_TILE), dtype=ll.bfloat16, stages=NUM_STAGES, major={case.a_major!r})
        B_smem = g.smem("B_smem", (CTA_TILE_N, K_TILE), dtype=ll.bfloat16, stages=NUM_STAGES, major={case.b_major!r})
{a_tmem}        C_tmem = g.tmem("C_tmem", (PAIR_TILE_M, PAIR_TILE_N), dtype=ll.float32)
        C_smem = g.smem("C_smem", (ST_TILE_M, ST_TILE_N), dtype=ll.bfloat16, stages=2, major="K")
        stage_ab = g.depends(src=load, dst=mma, via=[A_smem, B_smem])
        stage_acc = g.depends(src=mma, dst=epi, via=[C_tmem])
        stage_store = g.depends(src=epi, dst=epi, via=[C_smem])
        A_plan = g.tma_transfer(
            mcast_group_dim="y",
            inter_cta_coord_dim={a_inter_dim if cg == 2 else None!r},
            inter_cta_stride={cta_m if cg == 2 else 0},
        )
        B_plan = g.tma_transfer(
            mcast_group_dim="x",
            inter_cta_coord_dim={b_inter_dim if cg == 2 else None!r},
            inter_cta_stride={cta_n if cg == 2 else 0},
        )

    tid: ll.int32 = ll.threadIdx_x()
    rank_in_group: ll.uint32 = ri.rank_in_group()
    group_m: ll.uint32 = ll.blockIdx_x() // CTA_GROUP
    group_n: ll.uint32 = ll.blockIdx_y()

    with load:
        load_issuer: ll.uint32 = ll.elect_one()
        kb: ll.uint32 = ll.uint32(0)
        while kb < {math.ceil(K / case.bk)}:
            stage_ab.bind(kb)
            offset_k: ll.int32 = ll.int32(kb * K_TILE)
            if load_issuer == 1:
                ri.tma_ld_2d(A_smem, dA, {a_offsets}, A_plan, shape=(CTA_TILE_M, K_TILE))
                ri.tma_ld_2d(B_smem, dB, {b_offsets}, B_plan, shape=(CTA_TILE_N, K_TILE))
            kb = kb + 1

    with mma:
        mma_issuer: ll.uint32 = ll.elect_one()
        stage_acc.bind(0)
        kb: ll.uint32 = ll.uint32(0)
        while kb < {math.ceil(K / case.bk)}:
            stage_ab.bind(kb)
            if mma_issuer == 1:
{mma_body}
            kb = kb + 1

    with epi:
        local_tid: ll.int32 = tid - 128
        epi_warp: ll.int32 = local_tid // 32
        stage_acc.bind(0)
        for store in ll.unroll(range(NUM_N_STORES)):
            stage_store.bind(store)
            cd_base: ll.uint32 = ll.cvta_generic_to_shared(C_smem)
            for i in ll.unroll(range(ST_TILE_N // ELEMS_PER_BG)):
                tmem_col: ll.uint32 = ll.uint32(C_tmem + store * ST_TILE_N + i * ELEMS_PER_BG)
                smem_addr: ll.uint32 = ll.smem_swizzle_128b_addr(cd_base, ll.uint32(local_tid), ll.uint32(i))
                ri.tmem_ld_32dp_f32_to_smem_bf16_x8(C_tmem, C_smem, tmem_col, smem_addr)
            if epi_warp == 0:
                store_issuer: ll.uint32 = ll.elect_one()
                if store_issuer == 1:
                    offset_n = ll.int32(group_n * PAIR_TILE_N + store * ST_TILE_N)
                    offset_m = ll.int32(group_m * PAIR_TILE_M + rank_in_group * CTA_TILE_M)
                    ri.tma_st_2d(C_smem, dD, (offset_n, offset_m), shape=(ST_TILE_M, ST_TILE_N))
'''


def _ceil_to(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _launch_shape(case: Case) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    cluster_groups = (case.cluster_shape[0] // case.cta_group, case.cluster_shape[1], case.cluster_shape[2])
    groups_m = _ceil_to(math.ceil(M / case.bm), cluster_groups[0])
    groups_n = _ceil_to(math.ceil(N / case.bn), cluster_groups[1])
    grid = (groups_m * case.cta_group, groups_n, 1)
    padded = (groups_m * case.bm, groups_n * case.bn, _ceil_to(K, case.bk))
    return grid, padded


def _swizzle(elements: int) -> int:
    return next(size for size in (128, 64, 32) if elements * 2 % size == 0)


def _generate(case: Case) -> tuple[str, str]:
    from little_kernel.backends.nv.sm100.passes.dataflow import compile_dataflow_cuda

    source = _kernel_source(case)
    return source, compile_dataflow_cuda(source, num_threads=256)


def _descriptor(tensor, major: str, rows: int, k: int, tile_rows: int, tile_k: int):
    from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

    inner, outer = (k, rows) if major == "K" else (rows, k)
    tile_inner, tile_outer = ((tile_k, tile_rows) if major == "K" else (tile_rows, tile_k))
    return create_tma_2d_descriptor(
        tensor,
        gmem_inner_dim=inner,
        gmem_outer_dim=outer,
        smem_inner_dim=tile_inner,
        smem_outer_dim=tile_outer,
        gmem_outer_stride=inner,
        swizzle_mode=_swizzle(tile_inner),
        oob_fill=False,
        l2_promotion=3,
    )


def _run_case(case: Case, generated: tuple[str, str] | None = None) -> None:
    from little_kernel.runtime import CompiledKernel
    from little_kernel.backends.nv.common.runtime.tma_descriptor import create_tma_2d_descriptor

    source, code = generated or _generate(case)
    del source
    grid, (padded_m, padded_n, padded_k) = _launch_shape(case)
    cta_m, cta_n = case.bm // case.cta_group, case.bn // case.cta_group
    shared_mem_bytes = 2 * (cta_m * case.bk * 2 + cta_n * case.bk * 2 + 128 * 64 * 2) + 4096
    kernel = CompiledKernel(
        code,
        "umma_gemm_config_kernel",
        grid=grid,
        block=(256, 1, 1),
        shared_mem_bytes=shared_mem_bytes,
        arch="sm_100a",
        cache_dir=os.environ.get("LK_UMMA_CONFIG_TEST_CACHE"),
        include_paths=[PYTHON_ROOT],
        cluster_shape=case.cluster_shape,
        cta_group_shape=(case.cta_group, 1, 1),
        allow_non_portable_cluster=True,
    )

    torch.manual_seed(17)
    a = torch.zeros((padded_m, padded_k), device="cuda", dtype=torch.bfloat16)
    b = torch.zeros((padded_n, padded_k), device="cuda", dtype=torch.bfloat16)
    a[:M, :K].normal_().mul_(0.1)
    b[:N, :K].normal_().mul_(0.1)
    a_physical = a if case.a_major == "K" else a.T.contiguous()
    b_physical = b if case.b_major == "K" else b.T.contiguous()
    out = torch.full((padded_m, padded_n), float("nan"), device="cuda", dtype=torch.bfloat16)
    d_a = _descriptor(a_physical, case.a_major, padded_m, padded_k, cta_m, case.bk)
    d_b = _descriptor(b_physical, case.b_major, padded_n, padded_k, cta_n, case.bk)
    d_out = create_tma_2d_descriptor(
        out,
        gmem_inner_dim=padded_n,
        gmem_outer_dim=padded_m,
        smem_inner_dim=64,
        smem_outer_dim=128,
        gmem_outer_stride=padded_n,
        swizzle_mode=128,
        oob_fill=False,
        l2_promotion=3,
    )

    kernel(d_a, d_b, d_out)
    kernel.synchronize()
    result = out[:M, :N]
    assert torch.isfinite(result).all().item()
    reference = (a[:M, :K].float() @ b[:N, :K].float().T).to(torch.bfloat16)
    torch.testing.assert_close(result, reference, rtol=2e-2, atol=2e-1)


def _run_dispatch_case() -> None:
    from examples.nv.sm100.dispatch_group_gemm_swiglu_fused_sm100 import DispatchGroupGemmSwigluFuse

    count_values = (50, 300, 600, 1)
    hidden = DispatchGroupGemmSwigluFuse.COMM_TILE_K
    n_full = DispatchGroupGemmSwigluFuse.PAIR_TILE_N * DispatchGroupGemmSwigluFuse.CLUSTER_N
    torch.manual_seed(23)
    inputs = torch.randn((sum(count_values), hidden), device="cuda", dtype=torch.bfloat16).mul_(0.1)
    weights = torch.randn((len(count_values), n_full, hidden), device="cuda", dtype=torch.bfloat16).mul_(0.02)
    counts = torch.tensor(count_values, device="cuda", dtype=torch.int32)
    offsets = [0]
    for count in count_values:
        align = DispatchGroupGemmSwigluFuse.EXPERT_ALIGNMENT
        offsets.append(offsets[-1] + (count + align - 1) // align * align)
    metadata = torch.zeros((offsets[-1], ), device="cuda", dtype=torch.int64)
    compact_offset = 0
    for expert, count in enumerate(count_values):
        metadata[offsets[expert]:offsets[expert] + count] = torch.arange(compact_offset, compact_offset + count,
                                                                         device="cuda", dtype=torch.int64)
        compact_offset += count
    workspace = torch.empty((offsets[-1], hidden), device="cuda", dtype=torch.bfloat16)
    compiled = DispatchGroupGemmSwigluFuse.build(cache_dir=os.environ.get("LK_UMMA_CONFIG_TEST_CACHE"))
    runner = DispatchGroupGemmSwigluFuse.create(
        compiled,
        input_peers=[inputs],
        workspace=workspace,
        weights_interleaved=weights,
        token_src_rank_topk_and_indices=metadata,
        expert_counts=counts,
        num_ctas=2 * DispatchGroupGemmSwigluFuse.CLUSTER_SIZE,
    )

    reference_parts = []
    compact_offset = 0
    for expert, count in enumerate(count_values):
        accum = torch.matmul(inputs[compact_offset:compact_offset + count], weights[expert].t()).float()
        pairs = accum.reshape(count, -1, 2 * DispatchGroupGemmSwigluFuse.SWIGLU_EPI_TILE_N)
        gate, up = pairs.chunk(2, dim=2)
        reference_parts.append((torch.nn.functional.silu(gate) * up).reshape(count, -1).to(torch.bfloat16))
        compact_offset += count
    reference = torch.cat(reference_parts)
    for _ in range(2):
        workspace.fill_(float("nan"))
        runner.out.fill_(float("nan"))
        runner.launch()
        compiled.synchronize()
        workspace_valid = torch.cat([workspace[offsets[i]:offsets[i] + count] for i, count in enumerate(count_values)])
        output_valid = torch.cat([runner.out[offsets[i]:offsets[i] + count] for i, count in enumerate(count_values)])
        torch.testing.assert_close(workspace_valid, inputs, rtol=0, atol=0)
        torch.testing.assert_close(output_valid, reference, rtol=8e-2, atol=8e-2)


def _sm100_available() -> bool:
    return torch is not None and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_umma_gemm_config(case: Case) -> None:
    if not _sm100_available():
        pytest.skip("requires an SM100-class CUDA GPU")
    env = os.environ.copy()
    python_path = str(PYTHON_ROOT)
    env["PYTHONPATH"] = (python_path if not env.get("PYTHONPATH") else python_path + os.pathsep + env["PYTHONPATH"])
    subprocess.run([sys.executable, __file__, "--run-case", case.name], check=True, timeout=180, env=env)


def test_dispatch_topology() -> None:
    if not _sm100_available():
        pytest.skip("requires an SM100-class CUDA GPU")
    env = os.environ.copy()
    python_path = str(PYTHON_ROOT)
    env["PYTHONPATH"] = (python_path if not env.get("PYTHONPATH") else python_path + os.pathsep + env["PYTHONPATH"])
    subprocess.run([sys.executable, __file__, "--run-dispatch"], check=True, timeout=180, env=env)


def _main() -> None:
    parser = argparse.ArgumentParser(description=("Generate or correctness-check SM100 SS/TS UMMA configurations; "
                                                  f"logical MNK is fixed at [{M}, {N}, {K}]."))
    parser.add_argument("--list", action="store_true", help="list the default coverage set")
    parser.add_argument("--config", help="configuration name from --list or --list --exhaustive")
    parser.add_argument("--emit", choices=("lk", "cuda", "both"), help="source to print; --config defaults to both")
    parser.add_argument("--check", action="store_true", help="also launch the selected config")
    parser.add_argument("--all", action="store_true", help="check the default coverage set")
    parser.add_argument("--exhaustive", action="store_true",
                        help="use the full valid Cartesian product (never the default)")
    parser.add_argument("--run-case", help=argparse.SUPPRESS)
    parser.add_argument("--run-dispatch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_case:
        _run_case(CASE_BY_NAME[args.run_case])
        return
    if args.run_dispatch:
        _run_dispatch_case()
        return
    selected = EXHAUSTIVE_CASES if args.exhaustive else CASES
    if args.list:
        print("\n".join(case.name for case in selected))
        return
    if args.config:
        case = CASE_BY_NAME.get(args.config)
        if case is None:
            parser.error(f"unknown config {args.config!r}; use --list --exhaustive")
        generated = _generate(case)
        emit = args.emit or "both"
        if emit in {"lk", "both"}:
            print("===== LittleKernel source =====")
            print(generated[0])
        if emit in {"cuda", "both"}:
            print("===== Generated CUDA =====")
            print(generated[1])
        if args.check:
            _run_case(case, generated)
        return
    if args.all or args.exhaustive:
        for case in selected:
            print(f"checking {case.name}", flush=True)
            _run_case(case)
        return
    if args.check or args.emit:
        parser.error("--check/--emit requires --config")
    parser.print_help()


if __name__ == "__main__":
    _main()
