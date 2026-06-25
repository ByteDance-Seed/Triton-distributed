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
"""
Layer / function -level test for the AMD fused EP MoE.

Drives the layer ``layers/amd.EpAll2AllFusedOp`` and the functional / autograd
entries ``function/amd.{fused_ep_moe, fused_ep_moe_autograd,
TritonDistFusedEpMoeFunction}``, checking correctness against a single-process
PyTorch reference.

The default config family (only ``ntokens`` varies) is used to:
  1. run the fused MoE forward (``fused_ep_moe``) and check it against the
     reference (every rank holds the full expert table; the fused path only uses
     this rank's local expert slice);
  2. benchmark forward latency / peak memory (``benchmark_latency_memory``);
  3. (once) assert the autograd Function raises ``NotImplementedError`` on backward.

It then runs extra correctness cases (each builds/tears down its own op) covering
layer plumbing the default loop does not: ungated (``routing_weights=None``),
fp16, non-multiple tail dims, the decomposed 3-stage API
(``preprocess`` -> ``mega_dispatch_group_gemm`` -> ``mega_group_gemm_combine``),
0-recv skew, and capacity overflow (expects ``RuntimeError``).

The process exits non-zero if any correctness check fails.

Routing uses distinct valid top-k expert ids; weights are this rank's local expert
slices (``w1=[epr,hidden,inter]``, ``w2=[epr,inter,hidden]``).

Usage (mori_shmem; RCCL on this firmware requires HSA_NO_SCRATCH_RECLAIM=1):
  1. For 4-GPU run
    HIP_VISIBLE_DEVICES=4,5,6,7 ARNOLD_WORKER_GPU=4 \\
    TRITON_DIST_SHMEM_BACKEND=mori_shmem HSA_NO_SCRATCH_RECLAIM=1 \\
    bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ep_moe_fused.py
  2. For 8-GPU run
    TRITON_DIST_SHMEM_BACKEND=mori_shmem HSA_NO_SCRATCH_RECLAIM=1 \\
    bash ./scripts/launch_amd.sh ./python/triton_dist/test/amd/test_ep_moe_fused.py
"""

import argparse
import math
import os
import sys

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
from triton_dist.profiler_utils import benchmark_latency_memory, print_benchmark_comparison
from triton_dist.layers.amd.ep_a2a_fused_layer import EpAll2AllFusedOp
from triton_dist.function.amd.ep_moe_fused import (
    fused_ep_moe,
    fused_ep_moe_autograd,
    TritonDistFusedEpMoeFunction,
)

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def parse_args():
    p = argparse.ArgumentParser(description="AMD fused EP MoE layer/function test")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()), help="Data type")
    p.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    p.add_argument("--iters", type=int, default=10, help="Benchmark iterations")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--ntokens", default=4096, type=int, help="Max total number of tokens (across ranks)")
    p.add_argument("--hidden_dim", default=1024, type=int, help="Hidden dimension")
    p.add_argument("--ffn_dim", default=1024, type=int, help="FFN intermediate dimension")
    p.add_argument("--topk", default=4, type=int, help="Top-K experts per token")
    p.add_argument("--num_experts", default=16, type=int, help="Total number of experts")
    p.add_argument("--num_ranks", type=int, default=None, help="Number of EP ranks (default: WORLD_SIZE)")
    p.add_argument("--capacity", type=float, default=None,
                   help="Expert-group capacity (default: world_size, the worst-case safe value)")
    p.add_argument("--atol", type=float, default=2e-2, help="Correctness atol")
    p.add_argument("--rtol", type=float, default=2e-2, help="Correctness rtol")
    p.add_argument("--no-check", dest="no_check", action="store_true", help="Skip correctness checks")
    p.add_argument("--no-bench", dest="no_bench", action="store_true", help="Skip benchmarking")
    p.add_argument("--no-extra-cases", dest="no_extra_cases", action="store_true",
                   help="Skip the extra correctness cases (ungated/fp16/tail/decomposed/skew/capacity)")
    return p.parse_args()


def uniform_split_tokens(ntokens, nsplits):
    """Uniformly split tokens across ranks."""
    ret = [ntokens // nsplits for _ in range(nsplits)]
    ret[-1] = ntokens - sum(ret[:-1])
    assert all(x >= 0 for x in ret)
    return ret


def make_weights(num_experts, hidden, inter, world_size, rank, dtype, device, seed):
    """Full expert tables (identical on every rank via a fixed seed) + this rank's local slice.

    AMD layout: ``W1=[E, hidden, inter]`` (up), ``W2=[E, inter, hidden]`` (down).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    W1 = (torch.randn(num_experts, hidden, inter, generator=g, device=device, dtype=torch.float32) *
          (1.0 / hidden)**0.5).to(dtype)
    W2 = (torch.randn(num_experts, inter, hidden, generator=g, device=device, dtype=torch.float32) *
          (1.0 / inter)**0.5).to(dtype)
    epr = num_experts // world_size
    w1_local = W1[rank * epr:(rank + 1) * epr].contiguous()
    w2_local = W2[rank * epr:(rank + 1) * epr].contiguous()
    return W1, W2, w1_local, w2_local


def gen_routing(routing, n, num_experts, topk, epr, device, g):
    """Top-k expert ids per token. ``random`` = distinct ids via top-k over random logits;
    ``skew_rank0`` = every token -> rank-0's first ``topk`` experts (other ranks get 0 recv)."""
    if routing == "skew_rank0":
        assert topk <= epr, "skew_rank0 needs topk <= experts_per_rank"
        return torch.arange(topk, device=device, dtype=torch.int32).unsqueeze(0).expand(n, topk).contiguous()
    logits = torch.rand(n, num_experts, generator=g, device=device)
    return torch.topk(logits, k=topk, dim=-1).indices.to(torch.int32)


def prepare_inputs(n, hidden, num_experts, topk, dtype, device, seed):
    """Per-rank activations: hidden states, distinct top-k expert ids, routing weights."""
    g = torch.Generator(device=device).manual_seed(seed)
    hidden_states = (torch.randn(n, hidden, generator=g, device=device, dtype=torch.float32) * 0.5).to(dtype)
    expert_index = gen_routing("random", n, num_experts, topk, num_experts, device, g)
    gate_weights = torch.rand(n, topk, generator=g, device=device, dtype=torch.float32) + 0.5
    return hidden_states, expert_index, gate_weights


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


def run_layer_case(args, EP_GROUP, rank, world_size, device, *, case_idx, name, hidden, ffn, topk, experts, dtype, n,
                   gated=True, decomposed=False, capacity=None, routing="random", expect_capacity_error=False):
    """Build a dedicated op for one correctness case, run it, compare to the torch reference,
    and return the cross-rank-agreed pass flag. (Each case has its own op / shape / dtype.)"""
    dt = DTYPE_MAP[dtype]
    epr = experts // world_size
    cap = float(world_size) if capacity is None else capacity
    W1, W2, w1l, w2l = make_weights(experts, hidden, ffn, world_size, rank, dt, device, seed=args.seed + 17 * case_idx)
    layer = EpAll2AllFusedOp(EP_GROUP, max_tokens=n, hidden=hidden, topk=topk, num_tot_experts=experts, dtype=dt,
                             capacity=cap)
    g = torch.Generator(device=device).manual_seed(args.seed + 1000 * case_idx + rank + 1)
    hs = (torch.randn(n, hidden, generator=g, device=device, dtype=torch.float32) * 0.5).to(dt)
    idx = gen_routing(routing, n, experts, topk, epr, device, g)
    weights = torch.rand(n, topk, generator=g, device=device, dtype=torch.float32) + 0.5
    ok = True
    try:
        if expect_capacity_error:
            raised = False
            try:
                fused_ep_moe(layer, hs, idx, w1l, w2l, routing_weights=weights)
            except RuntimeError as e:
                raised = "cap_tokens" in str(e) or "capacity" in str(e)
            ok = raised
        else:
            ref = ref_moe(hs, idx, weights if gated else torch.ones_like(weights), W1, W2)
            gate = weights if gated else None
            if decomposed:
                desc = layer.preprocess(idx)
                g1, desc = layer.mega_dispatch_group_gemm(hs, idx, desc, w1l)
                out = layer.mega_group_gemm_combine(torch.relu(g1).contiguous(), w2l, desc, gate_input=gate)
            else:
                out = fused_ep_moe(layer, hs, idx, w1l, w2l, routing_weights=gate)
            torch.cuda.synchronize()
            ok = out.shape == ref.shape
            if ok:
                try:
                    torch.testing.assert_close(out.float(), ref.float(), atol=args.atol, rtol=args.rtol)
                except AssertionError:
                    ok = False
    finally:
        layer.finalize()
    flag = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=EP_GROUP)
    ok = bool(flag.item())
    if rank == 0:
        print(f"  [case {name}] {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    EP_GROUP = initialize_distributed()
    rank = EP_GROUP.rank()
    world_size = EP_GROUP.size()
    num_ranks = args.num_ranks if args.num_ranks is not None else world_size
    assert num_ranks == world_size, "num_ranks must equal WORLD_SIZE"
    assert args.num_experts % world_size == 0, "num_experts must be divisible by world_size"
    device = torch.cuda.current_device()
    dtype = DTYPE_MAP[args.dtype]
    capacity = float(world_size) if args.capacity is None else args.capacity

    H, I, topk, G = args.hidden_dim, args.ffn_dim, args.topk, args.num_experts

    # token configs (total tokens), filtered to <= --ntokens
    base = [512, 1024, 2048, 4096, 8192]
    ntokens_list = [t for t in base if t <= args.ntokens] or [args.ntokens]
    max_per_rank = max(int(math.ceil(t / num_ranks)) for t in ntokens_list)

    if rank == 0:
        print(f"AMD MoE Test: hidden={H}, ffn={I}, topk={topk}, num_experts={G}, ranks={num_ranks}, "
              f"dtype={dtype}, capacity={capacity}, ntokens_list={ntokens_list}\n", flush=True)

    # full expert tables (identical across ranks) + this rank's local slice
    W1, W2, w1_local, w2_local = make_weights(G, H, I, world_size, rank, dtype, device, seed=args.seed)

    # one fused op, reused across configs (only token count varies)
    layer = EpAll2AllFusedOp(EP_GROUP, max_tokens=max_per_rank, hidden=H, topk=topk, num_tot_experts=G, dtype=dtype,
                             capacity=capacity)

    all_implementations = {}
    global_ok = True
    try:
        for ntokens in ntokens_list:
            token_splits = uniform_split_tokens(ntokens, num_ranks)
            n = token_splits[rank]
            hidden_states, expert_index, gate_weights = prepare_inputs(
                n, H, G, topk, dtype, device, seed=args.seed + 1000 + rank + 7 * ntokens)
            torch.distributed.barrier(EP_GROUP)

            precision = "N/A"
            if not args.no_check:
                ref = ref_moe(hidden_states, expert_index, gate_weights, W1, W2)
                out = fused_ep_moe(layer, hidden_states, expert_index, w1_local, w2_local,
                                   routing_weights=gate_weights)
                torch.cuda.synchronize()
                ok = out.shape == ref.shape
                if ok:
                    try:
                        torch.testing.assert_close(out.float(), ref.float(), atol=args.atol, rtol=args.rtol)
                    except AssertionError:
                        ok = False
                # agree across ranks (min over the 0/1 flags)
                flag = torch.tensor([1 if ok else 0], device=device, dtype=torch.int32)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN, group=EP_GROUP)
                ok = bool(flag.item())
                precision = ok
                global_ok = global_ok and ok

            latency, memory = 0.0, 0.0
            if not args.no_bench:
                def fwd():
                    return fused_ep_moe(layer, hidden_states, expert_index, w1_local, w2_local,
                                        routing_weights=gate_weights)

                latency, memory = benchmark_latency_memory(fwd, args.iters, args.warmup)

            all_implementations[(ntokens, H, I)] = {
                "amd_fused_ep_moe_fwd": {"latency": latency, "memory": memory, "precision": precision},
            }
            if rank == 0:
                print(f"  config ntokens={ntokens} (n/rank={n}): precision={precision} "
                      f"latency={latency:.3f}ms mem={memory:.2f}MB", flush=True)

        # backward must raise NotImplementedError (check once)
        if not args.no_check:
            hs = hidden_states.clone().detach().requires_grad_(True)
            out_ag = fused_ep_moe_autograd(layer, hs, expert_index, w1_local, w2_local, routing_weights=gate_weights)
            raised = False
            try:
                out_ag.float().sum().backward()
            except NotImplementedError:
                raised = True
            assert raised, "TritonDistFusedEpMoeFunction.backward must raise NotImplementedError"
            assert TritonDistFusedEpMoeFunction is not None
            torch.distributed.barrier(EP_GROUP)
            if rank == 0:
                print("  backward NotImplementedError: raised as expected", flush=True)
    finally:
        layer.finalize()

    # extra correctness cases (each builds/tears down its own op)
    if not args.no_check and not args.no_extra_cases:
        extra_cases = [
            dict(name="ungated", hidden=512, ffn=512, topk=2, experts=2 * num_ranks, dtype=args.dtype, n=128,
                 gated=False),
            dict(name="fp16", hidden=512, ffn=512, topk=2, experts=2 * num_ranks, dtype="float16", n=128),
            dict(name="tail_dims", hidden=1000, ffn=600, topk=2, experts=2 * num_ranks, dtype=args.dtype, n=96),
            dict(name="decomposed_api", hidden=512, ffn=512, topk=2, experts=2 * num_ranks, dtype=args.dtype, n=128,
                 decomposed=True),
            dict(name="skew_rank0", hidden=512, ffn=512, topk=2, experts=2 * num_ranks, dtype=args.dtype, n=96,
                 routing="skew_rank0"),
            dict(name="capacity_overflow", hidden=256, ffn=256, topk=2, experts=2 * num_ranks, dtype=args.dtype,
                 n=256, capacity=0.25, expect_capacity_error=True),
        ]
        for i, c in enumerate(extra_cases):
            global_ok = run_layer_case(args, EP_GROUP, rank, num_ranks, device, case_idx=i + 1, **c) and global_ok

    if rank == 0:
        print_benchmark_comparison(all_implementations, "AMD Fused EP MoE (forward)",
                                   param_names=["Ntokens", "Hidden", "FFN"],
                                   title_params={"topk": topk, "num_experts": G})
        if not args.no_check:
            print("\n✅ all close!" if global_ok else "\n❌ correctness FAILED", flush=True)

    finalize_distributed()
    if not args.no_check and not global_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
