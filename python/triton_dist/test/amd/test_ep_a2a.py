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
AMD EP All-to-All test — dispatch + combine verification and performance.

Usage:
  # Verify correctness (default 3 rounds x 5 iterations):
  TRITON_DIST_SHMEM_BACKEND=mori_shmem bash ./scripts/launch_amd.sh \\
      ./python/triton_dist/test/amd/test_ep_a2a.py --check

  # With scatter indices:
  ... --check --with-scatter-indices

  # With local combine:
  ... --check --enable-local-combine

  # Performance benchmark:
  TRITON_DIST_SHMEM_BACKEND=mori_shmem bash ./scripts/launch_amd.sh \\
      ./python/triton_dist/test/amd/test_ep_a2a.py --rounds 5 --bench_iters 10

Performance (MI325X, 8 GPU, N=7168, G=256, topk=8, bench_iters=10,
             dispatch_grid=512, combine_grid=304):
  PT = PyTorch (all_to_all baseline), TD = Triton-dist, speedup = PT / TD

  Default mode:
    M     | PT disp | PT comb | TD disp | TD comb | disp  | comb  | total
    ------|---------|---------|---------|---------|-------|-------|------
    4096  | 2.19ms  | 3.31ms  | 1.80ms  | 1.92ms  | 1.22x | 1.72x | 1.48x
    3780  | 1.99ms  | 3.05ms  | 1.62ms  | 1.52ms  | 1.23x | 2.01x | 1.60x
    3206  | 2.06ms  | 3.12ms  | 1.66ms  | 1.59ms  | 1.25x | 1.96x | 1.59x
    2638  | 2.06ms  | 3.13ms  | 1.65ms  | 1.70ms  | 1.24x | 1.84x | 1.55x
    2395  | 1.93ms  | 3.04ms  | 1.55ms  | 1.52ms  | 1.25x | 2.00x | 1.62x
    2264  | 1.80ms  | 2.83ms  | 1.47ms  | 1.37ms  | 1.22x | 2.06x | 1.63x

  enable-local-combine mode:
    M     | PT disp | PT comb | TD disp | TD comb | disp  | comb  | total
    ------|---------|---------|---------|---------|-------|-------|------
    4096  | 2.20ms  | 3.29ms  | 1.80ms  | 2.15ms  | 1.22x | 1.53x | 1.36x
    3780  | 1.98ms  | 3.03ms  | 1.62ms  | 1.71ms  | 1.22x | 1.77x | 1.51x
    3206  | 2.07ms  | 3.16ms  | 1.67ms  | 1.84ms  | 1.24x | 1.72x | 1.49x
    2638  | 2.05ms  | 3.13ms  | 1.66ms  | 1.83ms  | 1.24x | 1.71x | 1.47x
    2395  | 1.93ms  | 2.95ms  | 1.56ms  | 1.72ms  | 1.24x | 1.71x | 1.48x
    2264  | 1.80ms  | 2.81ms  | 1.47ms  | 1.58ms  | 1.22x | 1.78x | 1.53x

  with-scatter-indices mode:
    M     | PT disp | PT comb | TD disp | TD comb | disp  | comb  | total
    ------|---------|---------|---------|---------|-------|-------|------
    4096  | 2.20ms  | 3.30ms  | 1.80ms  | 1.93ms  | 1.22x | 1.71x | 1.47x
    3780  | 1.97ms  | 3.08ms  | 1.62ms  | 1.52ms  | 1.22x | 2.03x | 1.60x
    3206  | 2.08ms  | 3.17ms  | 1.65ms  | 1.59ms  | 1.26x | 2.00x | 1.62x
    2638  | 2.07ms  | 3.33ms  | 1.65ms  | 1.72ms  | 1.25x | 1.94x | 1.60x
    2395  | 1.94ms  | 2.98ms  | 1.55ms  | 1.52ms  | 1.25x | 1.96x | 1.59x
    2264  | 1.81ms  | 2.78ms  | 1.47ms  | 1.36ms  | 1.23x | 2.04x | 1.62x

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
from functools import partial
from triton_dist.utils import initialize_distributed, finalize_distributed
from triton_dist.layers.amd.ep_a2a_layer import EPAll2AllLayer

# ---------------------------------------------------------------------------
#  Globals (set in main)
# ---------------------------------------------------------------------------
EP_GROUP = None
RANK = int(os.environ.get("RANK", 0))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


# ---------------------------------------------------------------------------
#  Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AMD EP A2A test")
    parser.add_argument("-M", type=int, default=4096, help="max tokens")
    parser.add_argument("-N", type=int, default=7168, help="hidden size")
    parser.add_argument("-G", type=int, default=256, help="total num experts")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--iters", default=3, type=int, help="check rounds")
    parser.add_argument("--verify-iters", default=5, type=int, help="iterations per check round")
    parser.add_argument("--bench_iters", default=5, type=int, help="perf iterations")
    parser.add_argument("--rounds", default=5, type=int, help="perf rounds")
    parser.add_argument("--sm_margin", default=64, type=int, help="num SMs for kernels")
    parser.add_argument("--dispatch-grid", default=512, type=int, help="dispatch kernel grid size")
    parser.add_argument("--combine-grid", default=304, type=int, help="combine kernel grid size")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--check", action="store_true", help="run correctness check")
    parser.add_argument("--with-scatter-indices", action="store_true")
    parser.add_argument("--enable-local-combine", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def generate_random_exp_indices(token_num, total_num_experts, topk):
    exp_indices = []
    exp_list = list(range(total_num_experts))
    for _ in range(token_num):
        exp_indices.append(random.sample(exp_list, topk))
    return torch.tensor(exp_indices, dtype=torch.int32)


def sort_by_vectors(x):
    """Sort 2D tensor rows lexicographically (order-agnostic comparison)."""
    assert len(x.shape) <= 2
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    M, K = x.shape
    current_order = torch.arange(M, device=x.device)
    for k in reversed(range(K)):
        current_col = x[current_order, k]
        _, sorted_indices = torch.sort(current_col, stable=True)
        current_order = current_order[sorted_indices]
    return x[current_order]


def calc_scatter_index_stable(chosen_experts: torch.Tensor):
    return (chosen_experts.flatten().argsort(stable=True).argsort().int().view(chosen_experts.shape))


def calc_full_scatter_indices(exp_indices, max_tokens, world_size):
    n_token = exp_indices.size(0)
    topk = exp_indices.size(1)
    input_len = torch.tensor([n_token], dtype=torch.int32, device=exp_indices.device)
    ag_input_len = torch.zeros(world_size, dtype=torch.int32, device=exp_indices.device)
    torch.distributed.all_gather_into_tensor(ag_input_len, input_len)
    ag_input_len_list = ag_input_len.cpu().tolist()

    padded_indices = torch.empty([max_tokens, topk], dtype=torch.int32, device=exp_indices.device)
    padded_indices[:n_token] = exp_indices
    ag_padded_indices = [torch.empty_like(padded_indices) for _ in range(world_size)]
    torch.distributed.all_gather(ag_padded_indices, padded_indices)

    ag_indices = torch.cat([t[:ag_input_len_list[i], :] for i, t in enumerate(ag_padded_indices)])
    return calc_scatter_index_stable(ag_indices)


def perf_func(fn, iters=100, warmup_iters=20):
    """Run function with warmup, return last result and avg time in ms."""
    result = None
    for _ in range(warmup_iters):
        result = fn()
    torch.cuda.synchronize()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        result = fn()
    ed.record()
    torch.cuda.synchronize()
    avg_time = st.elapsed_time(ed) / iters
    return result, avg_time


# ---------------------------------------------------------------------------
#  PyTorch reference implementations
# ---------------------------------------------------------------------------
def torch_forward_single(input, exp_indices, num_experts):
    """Reference dispatch: scatter → all_to_all → permute by (expert, rank)."""
    topk = exp_indices.size(1)
    hidden = input.size(1)
    ep_size = WORLD_SIZE
    experts_per_rank = num_experts // ep_size

    splits_gpu = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)[:num_experts]
    splits_cpu = splits_gpu.cpu()

    gather_idx = exp_indices.flatten().argsort(stable=True).to(torch.int32)
    token_indices = gather_idx // topk
    scattered_input = input[token_indices]

    a2a_splits = torch.empty_like(splits_gpu)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu)
    a2a_splits_cpu = a2a_splits.cpu()

    input_split_sizes = splits_cpu.reshape(ep_size, -1).sum(-1).tolist()
    output_split_sizes = a2a_splits_cpu.reshape(ep_size, -1).sum(-1).tolist()
    a2a_output = torch.empty([int(a2a_splits_cpu.sum()), hidden], dtype=input.dtype, device=input.device)
    torch.distributed.all_to_all_single(
        output=a2a_output,
        input=scattered_input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

    a2a_expert_list = torch.split(a2a_output, a2a_splits_cpu.tolist())
    permuted_list = []
    for eir in range(experts_per_rank):
        for src in range(ep_size):
            permuted_list.append(a2a_expert_list[src * experts_per_rank + eir])
    return torch.cat(permuted_list, dim=0)


def torch_backward_single(input, exp_indices, num_experts):
    """Reference combine: reverse permute → all_to_all back → gather → sum topk."""
    topk = exp_indices.size(1)
    ep_size = WORLD_SIZE
    experts_per_rank = num_experts // ep_size

    splits_gpu = torch.bincount(exp_indices.view(-1), minlength=num_experts).to(torch.int32)[:num_experts]
    splits_cpu = splits_gpu.cpu()

    _, index_sorted = exp_indices.flatten().sort(stable=True)
    gather_index = index_sorted.to(torch.int32) // topk
    topk_index = torch.arange(0, topk, dtype=torch.int32, device="cuda").repeat(exp_indices.size(0))[index_sorted]
    new_index = topk * gather_index + topk_index

    a2a_splits = torch.empty_like(splits_gpu)
    torch.distributed.all_to_all_single(a2a_splits, splits_gpu)
    a2a_splits_cpu = a2a_splits.cpu()

    permute_a2a_splits_cpu = (a2a_splits_cpu.reshape(-1, experts_per_rank).permute(-1, -2).flatten())
    permute_list = torch.split(input, permute_a2a_splits_cpu.tolist())
    a2a_list = []
    for src in range(ep_size):
        for eir in range(experts_per_rank):
            a2a_list.append(permute_list[eir * ep_size + src])
    a2a_expert_output = torch.cat(a2a_list, dim=0)

    count_before_drop = exp_indices.numel()
    count_after_drop = splits_cpu.sum().item()

    all2all_out = torch.empty([count_after_drop, input.shape[-1]], device=input.device, dtype=input.dtype)
    torch.distributed.all_to_all_single(
        output=all2all_out,
        input=a2a_expert_output,
        output_split_sizes=splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
        input_split_sizes=a2a_splits_cpu.reshape(ep_size, -1).sum(dim=-1).tolist(),
    )

    all2all_out_padded = torch.zeros(
        (count_before_drop, all2all_out.size(1)),
        device=all2all_out.device,
        dtype=all2all_out.dtype,
    )
    all2all_out_padded[:count_after_drop] = all2all_out
    gather_output = torch.zeros_like(all2all_out_padded)
    gather_output[new_index] = all2all_out_padded
    topk_reduce = gather_output.view((gather_output.size(0) // topk, topk, gather_output.size(-1))).sum(1)
    return topk_reduce


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    EP_GROUP = initialize_distributed()
    RANK = EP_GROUP.rank()
    WORLD_SIZE = EP_GROUP.size()
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", WORLD_SIZE))

    assert args.G % WORLD_SIZE == 0, (f"num_experts {args.G} must be divisible by world_size {WORLD_SIZE}")

    input_dtype = DTYPE_MAP[args.dtype]

    triton_a2a_op = EPAll2AllLayer(
        EP_GROUP,
        max_tokens=args.M,
        hidden=args.N,
        topk=args.topk,
        rank=RANK,
        num_tot_experts=args.G,
        local_world_size=LOCAL_WORLD_SIZE,
        world_size=WORLD_SIZE,
        dtype=input_dtype,
        weight_dtype=torch.float32,
        num_sm=args.sm_margin,
        enable_local_combine=args.enable_local_combine,
        dispatch_grid_size=args.dispatch_grid,
        combine_grid_size=args.combine_grid,
    )

    def _make_data(token_num):
        exp_indices = generate_random_exp_indices(token_num, args.G, args.topk).to("cuda")
        input = torch.randn(token_num, args.N, dtype=input_dtype, device="cuda")
        if args.with_scatter_indices:
            full_scatter_indices = calc_full_scatter_indices(exp_indices, args.M, WORLD_SIZE)
        else:
            full_scatter_indices = None
        return input, exp_indices, full_scatter_indices

    # -----------------------------------------------------------------------
    #  Correctness check
    # -----------------------------------------------------------------------
    if args.check:
        for n in range(args.iters):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            input_list = [_make_data(random.randint(1, args.M)) for _ in range(args.verify_iters)]
            torch_dispatch_list = []
            torch_combine_list = []
            triton_dispatch_list = []
            triton_combine_list = []

            # --- PyTorch reference ---
            for input, exp_indices, full_scatter_indices in input_list:
                ref_dispatch = torch_forward_single(input, exp_indices, args.G)
                ref_combine = torch_backward_single(ref_dispatch, exp_indices, args.G)
                torch_dispatch_list.append(ref_dispatch)
                torch_combine_list.append(ref_combine)

            # --- Triton distributed ---
            for input, exp_indices, full_scatter_indices in input_list:
                dispatch_out, _, layout_desc = triton_a2a_op.dispatch(
                    input,
                    exp_indices,
                    weight=None,
                    full_scatter_indices=full_scatter_indices,
                )
                combine_out = triton_a2a_op.combine(dispatch_out, layout_desc)
                triton_dispatch_list.append(dispatch_out)
                triton_combine_list.append(combine_out)

            # --- Verify dispatch ---
            for idx, (ref, tri) in enumerate(zip(torch_dispatch_list, triton_dispatch_list)):
                assert ref.shape == tri.shape, (f"dispatch shape mismatch: ref {ref.shape} vs triton {tri.shape}")
                if args.with_scatter_indices:
                    torch.testing.assert_close(tri, ref, atol=0, rtol=0)
                else:
                    torch.testing.assert_close(sort_by_vectors(tri), sort_by_vectors(ref), atol=0, rtol=0)

            # --- Verify combine ---
            for idx, (ref, tri) in enumerate(zip(torch_combine_list, triton_combine_list)):
                assert ref.shape == tri.shape, (f"combine shape mismatch: ref {ref.shape} vs triton {tri.shape}")
                torch.testing.assert_close(tri, ref, atol=1e-2, rtol=1e-2)

            if RANK == 0:
                print(f"  Round {n+1}/{args.iters}: "
                      f"{args.verify_iters} iterations, dispatch + combine verified.")

        print(f"RANK[{RANK}]: all checks passed.")
        triton_a2a_op.finalize()
        finalize_distributed()
        exit(0)

    # -----------------------------------------------------------------------
    #  Performance benchmark
    # -----------------------------------------------------------------------
    for rid in range(-1, args.rounds):
        token_num = args.M if rid == -1 else random.randint(args.M // 2, args.M)
        if RANK == 0:
            tag = "fixed" if rid == -1 else f"{rid+1}/{args.rounds}"
            print(f"\n--- Round {tag}, tokens={token_num} ---")

        input, exp_indices, full_scatter_indices = _make_data(token_num)

        # PyTorch reference
        ref_dispatch, ref_dispatch_time = perf_func(
            partial(torch_forward_single, input, exp_indices, args.G),
            iters=args.bench_iters,
            warmup_iters=20,
        )
        ref_combine, ref_combine_time = perf_func(
            partial(torch_backward_single, ref_dispatch, exp_indices, args.G),
            iters=args.bench_iters,
            warmup_iters=20,
        )

        # Triton distributed
        (triton_dispatch, _, layout_desc), triton_dispatch_time = perf_func(
            partial(
                triton_a2a_op.dispatch,
                input,
                exp_indices,
                weight=None,
                full_scatter_indices=full_scatter_indices,
            ),
            iters=args.bench_iters,
            warmup_iters=20,
        )
        triton_combine, triton_combine_time = perf_func(
            partial(triton_a2a_op.combine, triton_dispatch, layout_desc),
            iters=args.bench_iters,
            warmup_iters=20,
        )

        torch.cuda.synchronize()
        torch.distributed.barrier()

        # Verify correctness
        if args.with_scatter_indices:
            torch.testing.assert_close(triton_dispatch, ref_dispatch, atol=0, rtol=0)
        else:
            torch.testing.assert_close(
                sort_by_vectors(triton_dispatch),
                sort_by_vectors(ref_dispatch),
                atol=0,
                rtol=0,
            )
        torch.testing.assert_close(triton_combine, ref_combine, atol=1e-2, rtol=1e-2)

        if RANK == 0:
            recv_tokens = triton_dispatch.shape[0]
            elem_bytes = args.N * input_dtype.itemsize
            algo_bytes = recv_tokens * elem_bytes
            experts_per_rank = args.G // WORLD_SIZE
            rank_ids = exp_indices // experts_per_rank
            unique_remote_puts = sum(len(set(r.item() for r in row if r.item() != RANK)) for row in rank_ids)
            bus_bytes = unique_remote_puts * elem_bytes

            def _bw(nbytes, ms):
                return nbytes / (ms * 1e-3) / 1e9

            print(f"  PyTorch:  dispatch={ref_dispatch_time:.3f}ms, "
                  f"combine={ref_combine_time:.3f}ms")
            print(f"  Triton:   dispatch={triton_dispatch_time:.3f}ms, "
                  f"combine={triton_combine_time:.3f}ms")
            print(f"  Speedup:  dispatch={ref_dispatch_time/triton_dispatch_time:.2f}x, "
                  f"combine={ref_combine_time/triton_combine_time:.2f}x")
            print(f"  AlgoBW:   dispatch={_bw(algo_bytes, triton_dispatch_time):.1f} GB/s, "
                  f"combine={_bw(algo_bytes, triton_combine_time):.1f} GB/s "
                  f"(recv={recv_tokens} tokens, {algo_bytes/1e6:.1f} MB)")
            print(f"  BusBW:    dispatch={_bw(bus_bytes, triton_dispatch_time):.1f} GB/s, "
                  f"combine={_bw(bus_bytes, triton_combine_time):.1f} GB/s "
                  f"(unique_remote_puts={unique_remote_puts}, {bus_bytes/1e6:.1f} MB)")
            print("  Correctness: dispatch OK, combine OK")

    triton_a2a_op.finalize()
    finalize_distributed()
    if RANK == 0:
        print("\nAll done.")
