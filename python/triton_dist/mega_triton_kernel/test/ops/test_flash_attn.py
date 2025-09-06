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
import torch
import os
import argparse

import triton
from triton_dist.utils import get_torch_prof_ctx, perf_func
from triton_dist.mega_triton_kernel import ModelBuilder
from functools import partial
from triton_dist.mega_triton_kernel.test.triton_impl_utils import triton_attn


def attention_ref(input_tensors, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM, IS_CAUSAL, qkv_pack):
    # gqa is unsupported.
    if torch.__version__ < '2.6':
        return None

    if qkv_pack:
        qkv = input_tensors[0]
        q, k, v = qkv.split([Q_HEAD, KV_HEAD, KV_HEAD], dim=-2)
    else:
        q, k, v = input_tensors
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=IS_CAUSAL, enable_gqa=True,
                                                               scale=HEAD_DIM**-0.5)
    out_ref = out_ref.permute(0, 2, 1, 3).contiguous()
    return out_ref


def mega_kernel_runner(builder, input_tensors, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM, IS_CAUSAL, qkv_pack):
    out = torch.empty((BATCH, N_CTX, Q_HEAD, HEAD_DIM), dtype=input_tensors[0].dtype, device=input_tensors[0].device)
    if qkv_pack:
        qkv = input_tensors[0]
        builder.make_qkv_pack_flash_attn(qkv, out, is_causal=IS_CAUSAL)
    else:
        q, k, v = input_tensors
        builder.make_flash_attn(q, k, v, out, is_causal=IS_CAUSAL)
    builder.compile()

    def _run():
        return builder.run()

    return _run, out


def bench_flash_attn(builder, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM, IS_CAUSAL, dtype=torch.bfloat16, qkv_pack=False):
    device = torch.cuda.current_device()
    qkv = torch.randn((BATCH, N_CTX, Q_HEAD + KV_HEAD * 2, HEAD_DIM), dtype=dtype, device=device, requires_grad=False)
    if not qkv_pack:
        q, k, v = qkv.split([Q_HEAD, KV_HEAD, KV_HEAD], dim=-2)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        input_tensors = [q, k, v]
    else:
        input_tensors = [qkv]
    num_iters = 30
    warm_iters = 10
    out_ref, torch_time = perf_func(
        partial(attention_ref, input_tensors, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM, IS_CAUSAL, qkv_pack), num_iters,
        warm_iters)

    triton_out, triton_time = perf_func(
        partial(triton_attn, input_tensors, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM, IS_CAUSAL, qkv_pack), num_iters,
        warm_iters)
    mega_runner, mega_out = mega_kernel_runner(builder, input_tensors, BATCH, N_CTX, Q_HEAD, KV_HEAD, HEAD_DIM,
                                               IS_CAUSAL, qkv_pack)
    _, mega_time, = perf_func(mega_runner, num_iters, warm_iters)
    torch.cuda.synchronize()

    flop = 2 * BATCH * Q_HEAD * (N_CTX * HEAD_DIM * N_CTX + N_CTX * N_CTX * HEAD_DIM) / (2 if IS_CAUSAL else 1)
    print(f"FLOP = {flop * 1.0/ 1e12} TFLOP")

    skip_torch = (out_ref is None)
    if not skip_torch:
        print(f"torch: time = {torch_time}ms, {flop * 1.0 / 1e9 / torch_time} TFLOPS")
    print(f"triton: time = {triton_time}ms, {flop * 1.0 / 1e9 / triton_time} TFLOPS")
    print(f"mega kernel: time = {mega_time}ms, {flop * 1.0 / 1e9 / mega_time} TFLOPS")

    if not skip_torch:
        torch.testing.assert_close(out_ref, triton_out, atol=1e-2, rtol=0)
        torch.testing.assert_close(out_ref, mega_out, atol=1e-2, rtol=0)
    torch.testing.assert_close(triton_out, mega_out, atol=0, rtol=0)


def default_alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qkv_pack", default=False, action="store_true", help="pack qkv")
    parser.add_argument("--profile", default=False, action="store_true", help="enable kernel level profiling")
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    profile = args.profile

    builder = ModelBuilder(num_warps=8, enable_profiling=args.intra_kernel_profile)

    triton.set_allocator(default_alloc_fn)
    ctx = get_torch_prof_ctx(profile)
    with ctx:
        bench_flash_attn(builder, 1, 4096, 64, 8, 128, IS_CAUSAL=True, qkv_pack=args.qkv_pack)

    if args.intra_kernel_profile:
        builder.dump_trace()
        print(f"sm act = {builder.get_sm_activity()}")
    builder.finalize()

    if profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/flash_attn.json.gz")
