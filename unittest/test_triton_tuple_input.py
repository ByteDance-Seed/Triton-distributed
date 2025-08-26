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
import triton.language as tl
import torch
import random
import itertools

from triton_dist.utils import perf_func, assert_allclose
from triton.language.extra.cuda.language_extra import __syncthreads


@triton.jit
def kernel_tuple_list_input(input_ptr, output_ptr, hidden_size: tl.constexpr,
                            rank, cum_seqlen_cpu_tuple, cum_seqlen_gpu_ptr,
                            BLOCK_SIZE: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    for i in tl.static_range(len(cum_seqlen_cpu_tuple)):
        tl.store(cum_seqlen_gpu_ptr + i, cum_seqlen_cpu_tuple[i])
    __syncthreads()
    seq_start = tl.load(cum_seqlen_gpu_ptr + rank)
    seq_end = tl.load(cum_seqlen_gpu_ptr + rank + 1)
    seq_len = seq_end - seq_start

    num_tiles = tl.cdiv(seq_len, BLOCK_SIZE)

    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, hidden_size)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        src_ptrs = input_ptr + (seq_start + tile_id * BLOCK_SIZE +
                                offs_m[:, None]) * hidden_size + offs_n[
                                    None, :]
        dst_ptrs = output_ptr + (tile_id * BLOCK_SIZE + offs_m[:, None]
                                 ) * hidden_size + offs_n[None, :]
        mask = (seq_start + tile_id * BLOCK_SIZE + offs_m
                < seq_end)[:, None] & (offs_n < hidden_size)[None, :]
        tl.store(dst_ptrs, tl.load(src_ptrs, mask=mask), mask=mask)


@triton.jit
def kernel_pin_list_input(input_ptr, output_ptr, hidden_size: tl.constexpr,
                          rank, cum_seqlen_cpu_pin_ptr,
                          BLOCK_SIZE: tl.constexpr, NUM_SMS: tl.constexpr):
    pid = tl.program_id(axis=0)
    seq_start = tl.load(cum_seqlen_cpu_pin_ptr + rank)
    seq_end = tl.load(cum_seqlen_cpu_pin_ptr + rank + 1)
    seq_len = seq_end - seq_start

    num_tiles = tl.cdiv(seq_len, BLOCK_SIZE)

    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_n = tl.arange(0, hidden_size)

    for tile_id in range(pid, num_tiles, NUM_SMS):
        src_ptrs = input_ptr + (seq_start + tile_id * BLOCK_SIZE +
                                offs_m[:, None]) * hidden_size + offs_n[
                                    None, :]
        dst_ptrs = output_ptr + (tile_id * BLOCK_SIZE + offs_m[:, None]
                                 ) * hidden_size + offs_n[None, :]
        mask = (seq_start + tile_id * BLOCK_SIZE + offs_m
                < seq_end)[:, None] & (offs_n < hidden_size)[None, :]
        tl.store(dst_ptrs, tl.load(src_ptrs, mask=mask), mask=mask)


if __name__ == "__main__":
    num_seqs = 8
    rank = 3
    seqlen_cpu = [random.randint(32, 1024) for _ in range(num_seqs)]
    cum_seqlen_gpu = torch.empty([num_seqs + 1],
                                 dtype=torch.int32,
                                 device="cuda")
    cum_seqlen_cpu_pin = torch.empty([num_seqs + 1],
                                     dtype=torch.int32,
                                     device="cpu",
                                     pin_memory=True)
    seqlen = sum(seqlen_cpu)
    hidden_size = 2048
    inputs = torch.randn([seqlen, hidden_size],
                         dtype=torch.bfloat16,
                         device="cuda")
    outputs = torch.empty([seqlen_cpu[rank], hidden_size],
                          dtype=torch.bfloat16,
                          device="cuda")

    def test_tuple_kernel():
        num_sms = 16
        grid = (num_sms, )

        def func():
            cum_seqlen_cpu = [0] + list(itertools.accumulate(seqlen_cpu))
            kernel_tuple_list_input[grid](inputs, outputs, hidden_size, rank,
                                          tuple(cum_seqlen_cpu),
                                          cum_seqlen_gpu, 128, num_sms)
            return cum_seqlen_cpu

        cum_seqlen_cpu = func()
        assert_allclose(outputs,
                        inputs[cum_seqlen_cpu[rank]:cum_seqlen_cpu[rank + 1]],
                        atol=1e-4,
                        rtol=1e-4)

        _, tuple_perf = perf_func(func, iters=100, warmup_iters=10)
        print("tuple kernel perf is", tuple_perf, "ms")

    def test_pin_kernel():
        num_sms = 16
        grid = (num_sms, )

        def func():
            cum_seqlen_cpu = [0] + list(itertools.accumulate(seqlen_cpu))
            for i in range(len(cum_seqlen_cpu)):
                cum_seqlen_cpu_pin[i] = cum_seqlen_cpu[i]
            kernel_pin_list_input[grid](inputs, outputs, hidden_size, rank,
                                        cum_seqlen_cpu_pin, 128, num_sms)
            return cum_seqlen_cpu

        cum_seqlen_cpu = func()
        assert_allclose(outputs,
                        inputs[cum_seqlen_cpu[rank]:cum_seqlen_cpu[rank + 1]],
                        atol=1e-4,
                        rtol=1e-4)

        _, tuple_perf = perf_func(func, iters=100, warmup_iters=10)
        print("pin kernel perf is", tuple_perf, "ms")

    test_tuple_kernel()
    test_pin_kernel()
