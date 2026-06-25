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
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, ...
#
################################################################################
"""
Correctness test for the AMD fused EP-A2A + grouped-GEMM "mega kernel" forward.

Exercises the full fused MoE forward:
    dispatch + gemm1 (mega)  ->  relu  ->  gemm2 + combine (mega)
and compares against a single-process PyTorch reference (every rank holds an
identical full expert-weight table; the fused path only uses this rank's slice).

Checks run BY DEFAULT (pass --no-check to only init/finalize). Cases (each builds
its own context; all ranks run the same sequence so collectives stay matched):
  - baseline    : random distinct top-k routing
  - topk1       : top-1 routing
  - skew_rank0  : every token -> rank-0's experts; other ranks receive 0 tokens
                  (exercises the M_local == 0 path that must NOT early-return)
  - ktail       : M / H / I not multiples of the tile sizes, incl. K not a
                  multiple of BLOCK_SIZE_K (reduction-dim tail masking)
  - more_experts: experts_per_rank = 4 and topk = 4 (wider local-expert indexing)
  - fp16        : float16 dtype
  - ungated     : combine without routing weights (topk_weights=None)
  - partial     : num_tokens < max_tokens (+ per-iter varying counts -> ctx reuse)
  - capacity    : capacity too small -> every rank must raise a clear RuntimeError

Usage (mori_shmem; RCCL on this firmware requires HSA_NO_SCRATCH_RECLAIM=1):
  1. For 4-GPU run
    HIP_VISIBLE_DEVICES=4,5,6,7 ARNOLD_WORKER_GPU=4 \\
    TRITON_DIST_SHMEM_BACKEND=mori_shmem HSA_NO_SCRATCH_RECLAIM=1 \\
    bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ep_a2a_fused_kernel.py
  2. For 8-GPU run
    TRITON_DIST_SHMEM_BACKEND=mori_shmem HSA_NO_SCRATCH_RECLAIM=1 \\
    bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ep_a2a_fused_kernel.py
"""

import os
import sys
import argparse
import random

os.environ.setdefault("TRITON_DIST_SHMEM_BACKEND", "mori_shmem")
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "4G")

_test_dir = os.path.dirname(os.path.abspath(__file__))
_workspace_root = os.path.abspath(os.path.join(_test_dir, "../../../.."))
_triton_dist_python_path = os.path.join(_workspace_root, "python")
if _triton_dist_python_path not in sys.path:
    sys.path.insert(0, _triton_dist_python_path)
_current_pythonpath = os.environ.get("PYTHONPATH", "")
if _triton_dist_python_path not in _current_pythonpath:
    os.environ["PYTHONPATH"] = (f"{_triton_dist_python_path}:{_current_pythonpath}"
                                if _current_pythonpath else _triton_dist_python_path)
_triton_python_path = os.path.join(_workspace_root, "3rdparty/triton/python")
if os.path.exists(_triton_python_path):
    sys.path.insert(0, _triton_python_path)

import torch
import torch.distributed

from triton_dist.utils import initialize_distributed, finalize_distributed
from triton_dist.kernels.amd.ep_all2all_fused import (
    create_ep_a2a_fused_context,
    fused_dispatch_token_moe_grouped_gemm,
    fused_group_gemm_combine_token,
    _build_dispatch_metadata,
    _build_dispatch_metadata_device,
)

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args():
    p = argparse.ArgumentParser(description="AMD fused EP-A2A + grouped GEMM test")
    p.add_argument("--iters", type=int, default=2, help="check rounds per case")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()), help="default dtype")
    p.add_argument("--num_sms", type=int, default=64)
    p.add_argument("--num_dispatch_tasks", type=int, default=16)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--check", action="store_true", help="(deprecated; checks run by default)")
    p.add_argument("--no-check", dest="no_check", action="store_true", help="skip checks (init/finalize only)")
    return p.parse_args()


# ---------------------------------------------------------------------------
#  Routing generators (all produce [n, topk] int32, distinct experts per token)
# ---------------------------------------------------------------------------
def gen_indices(routing, n, G, topk, epr):
    if routing == "random":
        exp_list = list(range(G))
        rows = [random.sample(exp_list, topk) for _ in range(n)]
    elif routing == "skew_rank0":
        assert topk <= epr  # every token -> rank-0's experts [0, epr)
        rows = [list(range(topk)) for _ in range(n)]
    elif routing == "balanced_rr":
        rows = [[(t * topk + j) % G for j in range(topk)] for t in range(n)]
    else:
        raise ValueError(routing)
    return torch.tensor(rows, dtype=torch.int32)


def ref_moe(x, idx, weights, W1, W2):
    """out[t] = sum_j w[t,j] * relu(x[t] @ W1[e]) @ W2[e]  (model-dtype matmul, fp32 reduce)."""
    T, k = idx.shape
    H = x.shape[1]
    flat_e = idx.reshape(-1).long()
    xe = x.repeat_interleave(k, dim=0)
    h = torch.relu(torch.bmm(xe.unsqueeze(1), W1[flat_e]).squeeze(1))
    y = torch.bmm(h.unsqueeze(1), W2[flat_e]).squeeze(1)
    y = y.float() * weights.reshape(-1, 1).float()
    return y.reshape(T, k, H).sum(dim=1).to(x.dtype)


def run_metadata_parity(args, EP_GROUP, rank, world_size, device, *, name, M, H, I, G, topk, capacity, routing,
                        case_seed, block_m=128):
    """Assert the device-side metadata kernels produce a layout byte-identical to the host torch path."""
    epr = G // world_size
    ctx = create_ep_a2a_fused_context(EP_GROUP, max_tokens=M, hidden=H, topk=topk, num_tot_experts=G, rank=rank,
                                      world_size=world_size, dtype=torch.bfloat16, weight_dtype=torch.float32,
                                      capacity=capacity)
    try:
        torch.manual_seed(args.seed + 131 * case_seed)
        random.seed(args.seed + 131 * case_seed)
        idx = gen_indices(routing, M, G, topk, epr).to(device)
        # device first, then host (both are collectives; all ranks run the same order)
        dev = _build_dispatch_metadata_device(ctx, idx, block_m, num_sms=args.num_sms)
        host = _build_dispatch_metadata(ctx, idx, block_m)

        assert dev["M_local"] == host["M_local"], f"[{name}] M_local {dev['M_local']} vs {host['M_local']}"
        for key in ("recv_buf_offset", "num_input_tokens_per_rank", "num_recv_tokens_per_rank", "split_size",
                    "local_splits"):
            assert torch.equal(dev[key].reshape(-1).cpu(), host[key].reshape(-1).cpu()), f"[{name}] mismatch: {key}"
        t = int(host["total_tiles"].item())
        assert int(dev["total_tiles"].item()) == t, f"[{name}] total_tiles mismatch"
        for key in ("expert_ids", "split_size_cum", "tile_num", "tile_num_cum"):
            assert torch.equal(dev[key][:t].cpu(), host[key][:t].cpu()), f"[{name}] tiling mismatch: {key}"
    finally:
        ctx.finalize()
    torch.distributed.barrier(EP_GROUP)
    if rank == 0:
        print(f"  [parity:{name}] device==host metadata", flush=True)


def run_local_splits_alias_regression(args, EP_GROUP, rank, world_size, device):
    """Regression: a device-metadata desc must own a stable `local_splits` copy.

    Building a second desc on the same ctx must NOT mutate the first desc's local_splits
    (was a real bug when `meta['local_splits']` aliased the shared `ctx.local_splits_buf`).
    Collective-safe: every rank runs the same sequence.
    """
    G, topk, M, H = 2 * world_size, 1, 64, 256
    ctx = create_ep_a2a_fused_context(EP_GROUP, max_tokens=M, hidden=H, topk=topk, num_tot_experts=G, rank=rank,
                                      world_size=world_size, dtype=torch.bfloat16, weight_dtype=torch.float32,
                                      capacity=float(world_size))
    try:
        idxA = torch.zeros((M, topk), dtype=torch.int32, device=device)  # all -> expert 0
        idxB = torch.ones((M, topk), dtype=torch.int32, device=device)  # all -> expert 1
        metaA = _build_dispatch_metadata_device(ctx, idxA, 128, num_sms=args.num_sms)
        snapA = metaA["local_splits"].clone()
        _ = _build_dispatch_metadata_device(ctx, idxB, 128, num_sms=args.num_sms)
        assert metaA["local_splits"].data_ptr() != ctx.local_splits_buf.data_ptr(), \
            "local_splits must not alias ctx.local_splits_buf"
        assert torch.equal(metaA["local_splits"], snapA), \
            "metaA['local_splits'] was mutated by a later preprocess (aliasing regression)"
    finally:
        ctx.finalize()
    torch.distributed.barrier(EP_GROUP)
    if rank == 0:
        print("  [regression:local_splits_alias] desc owns a stable copy", flush=True)


def run_invalid_id_regression(args, EP_GROUP, rank, world_size, device):
    """Regression: building metadata with an out-of-range expert id must fail early & clearly
    (both backends), not silently produce a filtered descriptor. Local assertion -> all ranks
    raise symmetrically (collective-safe)."""
    G, topk, M, H = 2 * world_size, 1, 64, 256
    ctx = create_ep_a2a_fused_context(EP_GROUP, max_tokens=M, hidden=H, topk=topk, num_tot_experts=G, rank=rank,
                                      world_size=world_size, dtype=torch.bfloat16, weight_dtype=torch.float32,
                                      capacity=float(world_size))
    try:
        bad = torch.zeros((M, topk), dtype=torch.int32, device=device)
        bad[0, 0] = -1  # invalid / drop sentinel
        for builder, tag in ((_build_dispatch_metadata_device, "device"), (_build_dispatch_metadata, "host")):
            raised = False
            try:
                builder(ctx, bad, 128)
            except (AssertionError, RuntimeError):
                raised = True
            assert raised, f"[invalid_id:{tag}] expected metadata build to reject out-of-range id"
    finally:
        ctx.finalize()
    torch.distributed.barrier(EP_GROUP)
    if rank == 0:
        print("  [regression:invalid_id] preprocess rejects out-of-range ids (device+host)", flush=True)


def run_combine_autofollow_regression(args, EP_GROUP, rank, world_size, device):
    """Regression: combine with use_device_metadata=None must follow meta['metadata_backend']
    (locks the auto-follow branch) and stay numerically correct for both backends."""
    G, topk, M, H, I = 2 * world_size, 2, 128, 256, 256
    epr = G // world_size
    wgen = torch.Generator(device="cuda").manual_seed(args.seed + 4242)
    W1 = (torch.randn(G, H, I, generator=wgen, device=device, dtype=torch.float32) * (1.0 / H)**0.5).to(torch.bfloat16)
    W2 = (torch.randn(G, I, H, generator=wgen, device=device, dtype=torch.float32) * (1.0 / I)**0.5).to(torch.bfloat16)
    W1_local = W1[rank * epr:(rank + 1) * epr].contiguous()
    W2_local = W2[rank * epr:(rank + 1) * epr].contiguous()
    ctx = create_ep_a2a_fused_context(EP_GROUP, max_tokens=M, hidden=H, topk=topk, num_tot_experts=G, rank=rank,
                                      world_size=world_size, dtype=torch.bfloat16, weight_dtype=torch.float32,
                                      capacity=float(world_size))
    try:
        for backend_is_device in (False, True):
            torch.manual_seed(args.seed + 99)
            random.seed(args.seed + 99)
            x = (torch.randn(M, H, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
            idx = gen_indices("random", M, G, topk, epr).to(device)
            weights = torch.rand(M, topk, device=device, dtype=torch.float32) + 0.5
            ref = ref_moe(x, idx, weights, W1, W2)
            gemm1_out, meta = fused_dispatch_token_moe_grouped_gemm(ctx, x, idx, W1_local,
                                                                    use_device_metadata=backend_is_device,
                                                                    num_sms=args.num_sms)
            assert meta["metadata_backend"] == ("device" if backend_is_device else "host")
            act = torch.relu(gemm1_out).contiguous()
            # use_device_metadata omitted -> None -> must follow meta["metadata_backend"]
            out = fused_group_gemm_combine_token(ctx, act, W2_local, meta, idx, topk_weights=weights,
                                                 num_sms=args.num_sms)
            torch.cuda.synchronize()
            torch.testing.assert_close(out.float(), ref.float(), atol=args.atol, rtol=args.rtol)
    finally:
        ctx.finalize()
    torch.distributed.barrier(EP_GROUP)
    if rank == 0:
        print("  [regression:combine_autofollow] combine follows meta backend (host+device)", flush=True)


def run_precomputed_meta_bad_idx_regression(args, EP_GROUP, rank, world_size, device):
    """Regression: reusing a precomputed (validated) meta with a DIFFERENT, out-of-range
    topk_indices must still be rejected. dispatch validates the *current* indices it scatters on,
    never trusting the meta's backend tag as proof the indices are valid (closes a bypass where a
    tagged meta would skip the range check). Local assert -> all ranks raise symmetrically before
    any collective."""
    G, topk, M, H, I = 2 * world_size, 1, 64, 256, 256
    epr = G // world_size
    wgen = torch.Generator(device="cuda").manual_seed(args.seed + 555)
    W1 = (torch.randn(G, H, I, generator=wgen, device=device, dtype=torch.float32) * (1.0 / H)**0.5).to(torch.bfloat16)
    W1_local = W1[rank * epr:(rank + 1) * epr].contiguous()
    ctx = create_ep_a2a_fused_context(EP_GROUP, max_tokens=M, hidden=H, topk=topk, num_tot_experts=G, rank=rank,
                                      world_size=world_size, dtype=torch.bfloat16, weight_dtype=torch.float32,
                                      capacity=float(world_size))
    try:
        x = (torch.randn(M, H, device=device, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        good = torch.zeros((M, topk), dtype=torch.int32, device=device)
        _, meta = fused_dispatch_token_moe_grouped_gemm(ctx, x, good, W1_local, num_sms=args.num_sms)
        bad = torch.full((M, topk), G + 1, dtype=torch.int32, device=device)  # out of range
        raised = False
        try:
            fused_dispatch_token_moe_grouped_gemm(ctx, x, bad, W1_local, meta=meta, num_sms=args.num_sms)
        except (AssertionError, RuntimeError):
            raised = True
        assert raised, "dispatch must reject out-of-range topk_indices even when a precomputed meta is passed"
    finally:
        ctx.finalize()
    torch.distributed.barrier(EP_GROUP)
    if rank == 0:
        print("  [regression:precomputed_meta_bad_idx] dispatch revalidates current indices", flush=True)


def run_case(args, EP_GROUP, rank, world_size, device, *, name, M, H, I, G, topk, capacity, routing, case_seed,
             dtype="bfloat16", n_tokens=None, gated=True, expect_capacity_error=False, use_device_metadata=True):
    dt = DTYPE_MAP[dtype]
    epr = G // world_size
    assert G % world_size == 0 and topk <= G

    wgen = torch.Generator(device="cuda").manual_seed(args.seed + 7919 * case_seed)
    W1 = (torch.randn(G, H, I, generator=wgen, device=device, dtype=torch.float32) * (1.0 / H)**0.5).to(dt)
    W2 = (torch.randn(G, I, H, generator=wgen, device=device, dtype=torch.float32) * (1.0 / I)**0.5).to(dt)
    W1_local = W1[rank * epr:(rank + 1) * epr].contiguous()
    W2_local = W2[rank * epr:(rank + 1) * epr].contiguous()

    ctx = create_ep_a2a_fused_context(EP_GROUP, max_tokens=M, hidden=H, topk=topk, num_tot_experts=G, rank=rank,
                                      world_size=world_size, dtype=dt, weight_dtype=torch.float32, capacity=capacity)
    max_abs = max_rel = 0.0
    try:
        for it in range(args.iters):
            torch.manual_seed(args.seed + 1000 * it + rank + 131 * case_seed)
            random.seed(args.seed + 1000 * it + rank + 131 * case_seed)
            # vary token count per iter (<= max_tokens) to exercise context reuse
            n = M if n_tokens is None else min(n_tokens + 16 * it, M)
            x = (torch.randn(n, H, device=device, dtype=torch.float32) * 0.5).to(dt)
            idx = gen_indices(routing, n, G, topk, epr).to(device)
            weights = torch.rand(n, topk, device=device, dtype=torch.float32) + 0.5

            if expect_capacity_error:
                raised = False
                try:
                    fused_dispatch_token_moe_grouped_gemm(ctx, x, idx, W1_local,
                                                          use_device_metadata=use_device_metadata,
                                                          num_sms=args.num_sms,
                                                          num_dispatch_tasks=args.num_dispatch_tasks)
                except RuntimeError as e:
                    raised = "cap_tokens" in str(e)
                assert raised, f"[{name}] expected a capacity RuntimeError but none was raised"
                continue

            ref_w = weights if gated else torch.ones_like(weights)
            ref = ref_moe(x, idx, ref_w, W1, W2)
            gemm1_out, meta = fused_dispatch_token_moe_grouped_gemm(ctx, x, idx, W1_local,
                                                                    use_device_metadata=use_device_metadata,
                                                                    num_sms=args.num_sms,
                                                                    num_dispatch_tasks=args.num_dispatch_tasks)
            act = torch.relu(gemm1_out).contiguous()
            out = fused_group_gemm_combine_token(ctx, act, W2_local, meta, idx,
                                                 topk_weights=(weights if gated else None),
                                                 use_device_metadata=use_device_metadata, num_sms=args.num_sms)
            torch.cuda.synchronize()
            assert out.shape == ref.shape, f"[{name}] shape {out.shape} vs {ref.shape}"
            torch.testing.assert_close(out.float(), ref.float(), atol=args.atol, rtol=args.rtol)
            d = (out.float() - ref.float()).abs()
            max_abs = max(max_abs, float(d.max()))
            max_rel = max(max_rel, float((d / (ref.float().abs() + 1e-6)).max()))
    finally:
        ctx.finalize()
    torch.distributed.barrier(EP_GROUP)
    if rank == 0:
        extra = "" if expect_capacity_error else f" max_abs={max_abs:.3e} max_rel={max_rel:.3e}"
        print(f"  [{name}] passed ({args.iters} iters){extra}", flush=True)


def main():
    args = parse_args()
    EP_GROUP = initialize_distributed()
    rank = EP_GROUP.rank()
    world_size = EP_GROUP.size()
    device = torch.cuda.current_device()
    ws = world_size

    cases = [
        dict(name="baseline", M=256, H=512, I=1024, G=2 * ws, topk=2, capacity=float(ws), routing="random"),
        dict(name="topk1", M=256, H=512, I=1024, G=2 * ws, topk=1, capacity=float(ws), routing="random"),
        dict(name="skew_rank0", M=128, H=512, I=512, G=2 * ws, topk=2, capacity=float(ws), routing="skew_rank0"),
        # K-tail: H and I are NOT multiples of BLOCK_SIZE_K(=64); M not a multiple of BLOCK_M
        dict(name="ktail", M=200, H=328, I=520, G=2 * ws, topk=2, capacity=float(ws), routing="random"),
        # wider local-expert indexing: experts_per_rank = 4, topk = 4
        dict(name="more_experts", M=128, H=512, I=512, G=4 * ws, topk=4, capacity=float(ws), routing="random"),
        # high top-k: 8 experts per token (subset of G = 4*ws)
        dict(name="topk8", M=128, H=512, I=512, G=4 * ws, topk=8, capacity=float(ws), routing="random"),
        dict(name="fp16", M=256, H=512, I=512, G=2 * ws, topk=2, capacity=float(ws), routing="random",
             dtype="float16"),
        dict(name="ungated", M=192, H=512, I=512, G=2 * ws, topk=2, capacity=float(ws), routing="random", gated=False),
        # num_tokens < max_tokens, and per-iter varying token counts (context reuse)
        dict(name="partial", M=256, H=512, I=512, G=2 * ws, topk=2, capacity=float(ws), routing="random",
             n_tokens=100),
        dict(name="capacity_too_small", M=256, H=256, I=256, G=2 * ws, topk=2, capacity=0.5, routing="balanced_rr",
             expect_capacity_error=True),
    ]

    # device-vs-host metadata parity (covers default routing, 0-recv, K-tail, wider experts)
    parity_cases = [
        dict(name="baseline", M=256, H=512, I=1024, G=2 * ws, topk=2, capacity=float(ws), routing="random"),
        dict(name="skew_rank0", M=128, H=512, I=512, G=2 * ws, topk=2, capacity=float(ws), routing="skew_rank0"),
        dict(name="ktail", M=200, H=328, I=520, G=2 * ws, topk=2, capacity=float(ws), routing="random"),
        dict(name="more_experts", M=128, H=512, I=512, G=4 * ws, topk=4, capacity=float(ws), routing="random"),
    ]

    if args.no_check:
        if rank == 0:
            print("--no-check: skipping correctness checks", flush=True)
    else:
        if rank == 0:
            print("== metadata parity (device vs host) ==", flush=True)
        for i, c in enumerate(parity_cases):
            run_metadata_parity(args, EP_GROUP, rank, world_size, device, case_seed=1000 + i, **c)
        run_local_splits_alias_regression(args, EP_GROUP, rank, world_size, device)
        run_invalid_id_regression(args, EP_GROUP, rank, world_size, device)
        run_combine_autofollow_regression(args, EP_GROUP, rank, world_size, device)
        run_precomputed_meta_bad_idx_regression(args, EP_GROUP, rank, world_size, device)
        # run the full suite once per metadata path: device (default) then host fallback
        for mode in ("device", "host"):
            use_dev = (mode == "device")
            if rank == 0:
                print(f"== forward suite (metadata={mode}) ==", flush=True)
            for i, c in enumerate(cases):
                run_case(args, EP_GROUP, rank, world_size, device, case_seed=i, use_device_metadata=use_dev, **c)
        print(f"RANK[{rank}]: all checks passed.", flush=True)
        torch.distributed.barrier(EP_GROUP)
        if rank == 0:
            print("✅ all close!", flush=True)

    finalize_distributed()


if __name__ == "__main__":
    main()
