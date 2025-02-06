import torch
import torch.distributed

import argparse
import os

from triton.distributed.kernels.nvidia import (
    create_ag_group_gemm_intra_node_context,
    ag_group_gemm_intra_node,
)
from triton.distributed.kernels.nvidia.allgather_group_gemm import sort_topk_ids_align_block_size
from triton.distributed.utils import (
    initialize_distributed,
    TP_GROUP,
    perf_func,
    dist_print,
)

dtype = torch.float16


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
    a_tensor_golden_part_k = torch.zeros(M, K, dtype=local_input.dtype).cuda()
    torch.distributed.all_gather_into_tensor(a_tensor_golden_part_k, local_input, group=pg)
    a_tensor_golden = (a_tensor_golden_part_k.reshape(pg.size(), 1, M_per_rank, K).transpose(1, 2).reshape(M, K))
    tensor_golden = torch_moe_scatter_group_gemm(a_tensor_golden, local_weight, full_topk_ids)
    return a_tensor_golden, tensor_golden


def perf_test(input_len, config):
    M = input_len
    N = config["N"]
    K = config["K"]
    E = config["E"]
    topk = config["TOPK"]
    tp_group = TP_GROUP()

    assert M % tp_group.size() == 0
    assert N % tp_group.size() == 0
    M_per_rank = M // tp_group.size()
    N_per_rank = N // tp_group.size()

    if tp_group.rank() == 0:
        print(f"shape: M={M}, N={N}, K={K}; num experts={E}, topk={topk}")

    A = torch.randn([M_per_rank, K], dtype=dtype, device="cuda")
    B = torch.randn([E, K, N_per_rank], dtype=dtype, device="cuda")
    score = ((-2 * torch.randn((M_per_rank, E), device="cuda", dtype=dtype) + 1) / 100 * (tp_group.rank() + 1))
    score = torch.softmax(score, dim=-1)
    _, local_topk_ids = torch.topk(score, topk)

    full_topk_ids = torch.zeros(M_per_rank * tp_group.size(), topk, dtype=local_topk_ids.dtype).cuda()
    torch.distributed.all_gather_into_tensor(full_topk_ids, local_topk_ids, group=tp_group)

    ctx = create_ag_group_gemm_intra_node_context(
        tp_group,
        A,
        B,
        full_topk_ids,
        E,
        max_M=M,
        BLOCK_M=config["BM"],
        BLOCK_N=config["BN"],
        BLOCK_K=config["BK"],
        GROUP_SIZE_M=config["GROUP_M"],
        stages=config["stage"],
        warps=config["warp"],
        ag_stream=torch.cuda.Stream(),
        group_gemm_stream=torch.cuda.Stream(),
    )

    result = ag_group_gemm_intra_node(A, B, ctx)

    _, C_golden = torch_ag_group_gemm(tp_group, A, B, full_topk_ids)

    assert torch.allclose(C_golden, result, atol=1e-3, rtol=1e-3)

    def sort_func():
        return sort_topk_ids_align_block_size(full_topk_ids, E, tp_group.size(), M_per_rank, config["BM"])[0]

    def triton_func():
        return ag_group_gemm_intra_node(A, B, ctx)

    def torch_func():
        return torch_ag_group_gemm(tp_group, A, B, full_topk_ids)

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.CPU,
            ],
            record_shapes=True,
            profile_memory=True,
    ) as profiler:
        _, permute_perf = perf_func(sort_func, iters=10, warmup_iters=10)
        _, triton_perf = perf_func(triton_func, iters=10, warmup_iters=10)
        _, torch_perf = perf_func(torch_func, iters=10, warmup_iters=10)

        dist_print(
            f"RANK {tp_group.rank()} perf: compute permute {permute_perf} ms, dist-triton={triton_perf} ms, torch={torch_perf} ms; speedup={torch_perf/triton_perf}",
            need_sync=True,
            allowed_ranks=list(range(tp_group.size())),
        )

    prof_dir = "prof/trace_ag_scatter_group_gemm_intra_node"
    os.makedirs(prof_dir, exist_ok=True)
    profiler.export_chrome_trace(f"{prof_dir}/m_{M}_n_{N}_k_{K}_rank_{tp_group.rank()}.json")


layer_configs = {
    "Dummy Model": {
        "K": 8192,
        "N": 8192,
        "E": 32,
        "TOPK": 3,
        "BM": 128,
        "BN": 128,
        "BK": 32,
        "GROUP_M": 8,
        "stage": 4,
        "warp": 8,
    },
    "Qwen1.5-MoE-A2.7B": {
        "K": 1408,
        "N": 2048,
        "E": 60,
        "TOPK": 4,
        "BM": 128,
        "BN": 128,
        "BK": 64,
        "GROUP_M": 8,
        "stage": 4,
        "warp": 8,
    },
    "Mixtral-8x7B": {
        "K": 14336,
        "N": 4096,
        "E": 8,
        "TOPK": 2,
        "BM": 128,
        "BN": 256,
        "BK": 64,
        "GROUP_M": 8,
        "stage": 4,
        "warp": 8,
    },
    "Mixtral-8x22B": {
        "K": 16384,
        "N": 6144,
        "E": 8,
        "TOPK": 2,
        "BM": 128,
        "BN": 256,
        "BK": 64,
        "GROUP_M": 8,
        "stage": 4,
        "warp": 8,
    },
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=8192)
    args = parser.parse_args()

    initialize_distributed()

    for _, config in layer_configs.items():
        perf_test(args.M, config)

    torch.distributed.destroy_process_group()
