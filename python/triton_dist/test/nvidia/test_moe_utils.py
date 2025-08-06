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

import triton
import torch
from triton_dist.kernels.nvidia.moe_utils import (
    histogram_by_expert_triton,
    histogram_by_expert_torch,
    calc_gather_scatter_index_triton,
    reduce_topk_non_tma,
    reduce_topk_tma,
)
from triton_dist.utils import perf_func, assert_allclose, sleep_async


def _generate_random_choosed_experts(ntokens, topk, nexperts, generator: torch.Generator = None):
    chosen_experts = torch.multinomial(
        torch.ones(ntokens, nexperts, device="cuda", dtype=torch.float32),
        topk,
        replacement=False,
        generator=generator,
    ).to(torch.int32)
    return chosen_experts


def test_histogram(ntokens, topk, nexperts):
    chosen_experts = _generate_random_choosed_experts(ntokens, topk, nexperts)
    ntokens_by_expert_triton = histogram_by_expert_triton(chosen_experts, nexperts)
    ntokens_by_expert_torch = histogram_by_expert_torch(chosen_experts, nexperts)
    assert torch.allclose(ntokens_by_expert_triton, ntokens_by_expert_torch)
    print("✅ test_histogram passes")


def test_calc_gather_scatter_index(ntokens, topk, nexperts):
    chosen_experts = _generate_random_choosed_experts(ntokens, topk, nexperts)
    ntokens_by_expert, scatter_index, gather_index, expert_index, M_pad = calc_gather_scatter_index_triton(
        chosen_experts, nexperts)
    torch.testing.assert_close(
        scatter_index.flatten().sort()[0],
        torch.arange(ntokens * topk, device="cuda", dtype=torch.int32),
    )

    for n in range(nexperts):
        M_start = int(ntokens_by_expert[:n].sum())
        M_end = M_start + int(ntokens_by_expert[n])
        torch.testing.assert_close(
            scatter_index[chosen_experts == n].flatten().sort()[0],
            torch.arange(M_start, M_end, device="cuda", dtype=torch.int32),
        )

        token_index: torch.Tensor = gather_index[M_start:M_end] // topk  # has a expert_id of n
        assert torch.all(torch.any(chosen_experts[token_index, :] == n, 1))

    print("✅ test_calc_gather_scatter_index passes")
    sleep_async(100)
    _, duration_ms = perf_func(
        lambda: calc_gather_scatter_index_triton(chosen_experts, nexperts),
        iters=10,
        warmup_iters=5,
    )
    print(f"calc gather_scatter_index: {duration_ms:0.3f} ms/iter")


def test_reduce_topk(ntokens, topk, N, dtype: torch.dtype, n_split, nexperts):
    t_in = torch.randn(ntokens * topk, N, dtype=dtype, device="cuda")
    chosen_experts = _generate_random_choosed_experts(ntokens, topk, nexperts)
    out_torch = torch.sum(t_in.view(ntokens, topk, N), dim=1)
    out_triton = torch.empty_like(out_torch)

    reduce_topk_fns = [reduce_topk_non_tma]
    if torch.cuda.get_device_capability() >= (9, 0):
        reduce_topk_fns.append(reduce_topk_tma)

    assert N % n_split == 0
    N_per_split = N // n_split
    assert N_per_split == triton.next_power_of_2(N_per_split)
    weight = torch.ones_like(chosen_experts)

    for reduce_topk_fn in reduce_topk_fns:
        for n in range(n_split):
            n_start = N_per_split * n
            n_end = n_start + N_per_split
            reduce_topk_fn(t_in[:, n_start:n_end], weight, None, out_triton[:, n_start:n_end])
        assert_allclose(out_triton, y=out_torch, rtol=1e-2, atol=1e-2, verbose=False)
        print(f"✅ test_reduce_topk with {reduce_topk_fn.__name__} passes")

        sleep_async(100)
        _, duration_ms = perf_func(
            lambda: reduce_topk_fn(t_in, weight, None, out_triton),
            iters=10,
            warmup_iters=5,
        )
        memory_read = ntokens * topk * N * dtype.itemsize
        read_gbps = memory_read / duration_ms / 1e6
        print(
            f"calc reduce_topk: {duration_ms:0.3f} ms/iter, memory_read: {memory_read / 1e9:.3f} GB, read_gbps: {read_gbps:.3f} GB/s"
        )


if __name__ == "__main__":
    test_histogram(1024, 5, 32)
    test_calc_gather_scatter_index(8192, 5, 64)
    test_reduce_topk(1024, 2, 4096, torch.float16, 1, 64)
