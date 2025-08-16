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
import argparse
import os

import torch
import torch.distributed
import nvshmem.core

from triton_dist.kernels.nvidia import (ag_group_gemm, create_ag_group_gemm_context)
from triton_dist.kernels.nvidia.allgather_group_gemm import run_moe_ag_triton_non_overlap, sort_topk_ids_align_block_size
from triton_dist.kernels.nvidia.comm_perf_model import (estimate_all_gather_time_ms, get_nic_gbps_per_gpu)
from triton_dist.kernels.nvidia.gemm_perf_model import (get_dram_gbps, get_tensorcore_tflops)
from triton_dist.utils import (assert_allclose, dist_print, get_device_max_shared_memory_size, get_intranode_max_speed,
                               group_profile, initialize_distributed, perf_func, sleep_async)


def torch_moe_scatter_group_gemm(in_features, expert_weights, topk_ids):
    M, K = in_features.shape
    in_features = (in_features.view(M, -1, K).repeat(1, topk_ids.shape[1], 1).reshape(-1, K))
    out = torch.zeros(
        M * topk_ids.shape[1],
        expert_weights.shape[2],
        dtype=in_features.dtype,
        device=in_features.device,
    )

    topk_ids = topk_ids.view(-1)

    for i in range(expert_weights.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = in_features[mask] @ expert_weights[i]
    return out


def torch_ag_group_gemm(
    pg: torch.distributed.ProcessGroup,
    local_input: torch.Tensor,
    local_weight: torch.Tensor,
    full_topk_ids: torch.Tensor,
):
    M_per_rank, K = local_input.shape
    M = M_per_rank * pg.size()
    a_tensor_golden = torch.zeros(M, K, dtype=local_input.dtype).cuda()
    torch.distributed.all_gather_into_tensor(a_tensor_golden, local_input, group=pg)
    tensor_golden = torch_moe_scatter_group_gemm(a_tensor_golden, local_weight, full_topk_ids)
    return a_tensor_golden, tensor_golden


def estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype: torch.dtype, b_dtype: torch.dtype, stages: int):
    return (BM * BK * a_dtype.itemsize + BN * BK * b_dtype.itemsize) * (stages - 1)


def estimate_gemm_max_stages(BM, BN, BK, a_dtype, b_dtype, shared_memory_limit: int):
    return shared_memory_limit // estimate_gemm_shared_memory_size(BM, BN, BK, a_dtype, b_dtype, 2) + 1


def perf_test(name, input_len, dtype: torch.dtype, config, pg: torch.distributed.ProcessGroup, debug=False):
    M = input_len
    N = config["N"]
    K = config["K"]
    E = config["E"]
    topk = config["TOPK"]
    RANK = pg.rank()
    WORLD_SIZE = pg.size()
    LOCAL_WORLD_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", WORLD_SIZE))

    assert M % WORLD_SIZE == 0
    assert N % WORLD_SIZE == 0
    M_per_rank = M // WORLD_SIZE
    N_per_rank = N // WORLD_SIZE

    if RANK == 0:
        print(f"shape: M={M}, N={N}, K={K}; num experts={E}, topk={topk}")
        flops_per_rank = 2 * M * N * K * topk // WORLD_SIZE
        tflops = get_tensorcore_tflops(dtype)
        tensorcore_ms = flops_per_rank / tflops / 1e9
        dram_gbps = get_dram_gbps()
        memory_read_per_rank = M * K * dtype.itemsize + K * N_per_rank * E * dtype.itemsize
        memory_write_per_rank = M * N_per_rank * dtype.itemsize
        memory_read_ms = memory_read_per_rank / 1e6 / dram_gbps
        memory_write_ms = memory_write_per_rank / 1e6 / dram_gbps
        moe_sol_ms = max(tensorcore_ms, memory_read_ms + memory_write_ms)
        print("  MOE perf estimate")
        print(f"   TensorCore: {flops_per_rank/1e12:0.2f} TFLOPs {tensorcore_ms:0.2f} ms expected")
        print(f"   Memory read: {memory_read_per_rank/1e9:0.2f} GB, {memory_read_ms:0.2f} ms expected")
        print(f"   Memory write: {memory_write_per_rank/1e9:0.2f} GB, {memory_write_ms:0.2f} ms expected")
        print(f"   SOL time: {moe_sol_ms:0.2f} ms")
        print("  AllGather perf estimate")
        intranode_bw = get_intranode_max_speed()
        internode_bw = get_nic_gbps_per_gpu()
        allgather_sol_ms = estimate_all_gather_time_ms(M * K * dtype.itemsize, WORLD_SIZE, LOCAL_WORLD_SIZE,
                                                       intranode_bw, internode_bw)
        print(f"   SOL time: {allgather_sol_ms:0.2f} ms")
        print(f" AG+MOE SOL time: {max(allgather_sol_ms, moe_sol_ms):0.2f} ms")

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([E, K, N_per_rank], dtype=dtype, device="cuda")

    score = ((-2 * torch.randn((M_per_rank, E), device="cuda", dtype=dtype) + 1) / 100 * (RANK + 1))
    score = torch.softmax(score, dim=-1)
    _, local_topk_ids = torch.topk(score, topk)
    if debug:
        A.fill_(1 + RANK // K)
        for n in range(E):
            B[n, :, :].fill_(n + 1)

        local_topk_ids = torch.arange(topk, dtype=local_topk_ids.dtype).cuda().repeat((M_per_rank, 1))

    full_topk_ids = torch.zeros(M_per_rank * WORLD_SIZE, topk, dtype=local_topk_ids.dtype).cuda()
    torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids, group=pg)

    BM, BN, BK, stage = config["BM"], config["BN"], config["BK"], config["num_stages"]
    shared_memory_limit = get_device_max_shared_memory_size(torch.cuda.current_device())
    max_stages = estimate_gemm_max_stages(BM, BN, BK, A.dtype, B.dtype, shared_memory_limit)
    if stage > max_stages:
        print(f"stage {stage} exceeds max stages {max_stages}, force set to {max_stages}...")
        config["num_stages"] = max_stages

    ctx = create_ag_group_gemm_context(
        M,
        N_per_rank,
        K,
        E,
        topk,
        dtype,
        RANK,
        WORLD_SIZE,
        LOCAL_WORLD_SIZE,
        BLOCK_SIZE_M=config["BM"],
        BLOCK_SIZE_N=config["BN"],
        BLOCK_SIZE_K=config["BK"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        stages=config["num_stages"],
        num_warps=config["num_warps"],
    )

    C_triton = ag_group_gemm(A, B, ctx, full_topk_ids)
    C_triton_non_overlap = run_moe_ag_triton_non_overlap(A, B, full_topk_ids)
    _, C_torch = torch_ag_group_gemm(pg, A, B, full_topk_ids)
    try:
        assert_allclose(C_torch, C_triton, atol=1e-3, rtol=1e-3, verbose=False)
        assert_allclose(C_torch, C_triton_non_overlap, atol=1e-3, rtol=1e-3, verbose=False)
    except Exception as e:
        torch.save(C_torch, f"{name}_C_torch_{RANK}.pt")
        torch.save(C_triton, f"{name}_C_triton_{RANK}.pt")
        raise e
    else:
        print(f"✅ RANK {RANK} {name} pass")

    triton_func = lambda: ag_group_gemm(A, B, ctx, full_topk_ids)
    torch_func = lambda: torch_ag_group_gemm(pg, A, B, full_topk_ids)
    triton_non_overlap_func = lambda: run_moe_ag_triton_non_overlap(A, B, full_topk_ids)

    name = name.lower().replace(" ", "_").replace("-", "_")
    with group_profile(f"ag_moe_{name}_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile, group=pg):
        sleep_async(100)
        _, duration_triton_ms = perf_func(triton_func, iters=args.iters, warmup_iters=args.warmup_iters)

    with group_profile(f"ag_moe_torch_{name}_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile, group=pg):
        sleep_async(100)
        _, duration_torch_ms = perf_func(torch_func, iters=args.iters, warmup_iters=args.warmup_iters)

    with group_profile(f"ag_moe_triton_non_overlap_{name}_{os.environ['TORCHELASTIC_RUN_ID']}", do_prof=args.profile,
                       group=pg):
        sleep_async(100)
        _, duration_triton_non_overlap_ms = perf_func(triton_non_overlap_func, iters=args.iters,
                                                      warmup_iters=args.warmup_iters)

    sort_func = lambda: sort_topk_ids_align_block_size(full_topk_ids, E, RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, BM)
    sleep_async(100)
    _, duration_context_ms = perf_func(sort_func, iters=args.iters, warmup_iters=args.warmup_iters)

    dist_print(
        f"RANK {RANK} perf: calc sorted_gather_index {duration_context_ms:0.3f} ms, dist-triton={duration_triton_ms:0.3f} ms, torch={duration_torch_ms:0.3f} ms, triton-non-overlap={duration_triton_non_overlap_ms:0.3f} ms; speedup={duration_torch_ms/duration_triton_ms:0.2f}",
        need_sync=True,
        allowed_ranks=list(range(WORLD_SIZE)),
    )
    # TODO(houqi.1993) do not release nvshmem tensor: due to a BUG from nvshmem4py
    # ctx.finalize()


layer_configs = {
    "Dummy-Model": {
        "N": 8192, "K": 8192, "E": 32, "TOPK": 3, "BM": 128, "BN": 128, "BK": 32, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "Qwen1.5-MoE-A2.7B": {
        "N": 1408, "K": 2048, "E": 60, "TOPK": 4, "BM": 128, "BN": 128, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "Mixtral-8x7B": {
        "N": 4096, "K": 14336, "E": 8, "TOPK": 2, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "Mixtral-8x22B": {
        "N": 6144, "K": 16384, "E": 8, "TOPK": 2, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
    "DeepSeek-MoE": {
        "N": 2048, "K": 1408, "E": 64, "TOPK": 6, "BM": 128, "BN": 256, "BK": 64, "GROUP_SIZE_M": 8, "num_stages": 4,
        "num_warps": 8
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    parser.add_argument("--profile", default=False, action="store_true")
    parser.add_argument("--autotune", default=False, action="store_true")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    if args.autotune:
        import triton
        from triton_dist.autotuner import contextual_autotune
        from triton_dist.kernels.nvidia import allgather_group_gemm

        configs = [
            triton.Config({"BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK}, num_stages=s, num_warps=w)
            for BN in [128, 256]
            for BK in [32, 64]
            for s in [3, 4]
            for w in [4, 8]
        ]
        allgather_group_gemm.kernel_consumer_m_parallel_scatter_group_gemm = triton.autotune(
            configs=configs, key=["M", "N", "K"])(allgather_group_gemm.kernel_consumer_m_parallel_scatter_group_gemm)
        ag_group_gemm = contextual_autotune(is_dist=True)(ag_group_gemm)

    TP_GROUP = initialize_distributed()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]

    for name, config in layer_configs.items():
        perf_test(name, args.M, dtype, config, debug=args.debug, pg=TP_GROUP)

    nvshmem.core.finalize()
    torch.distributed.destroy_process_group()
