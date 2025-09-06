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
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import (
    prepare_cos_sin_cache,
    rmsnorm_ref,
    apply_rotary_pos_emb,
)
from triton_dist.mega_triton_kernel.test.triton_impl_utils import triton_qkv_pack_qk_norm_rope_split_v


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable kernel level profiling")
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    profile = args.profile
    MAX_SEQ_LEN = 65536
    l2_cache = torch.randn((256, 1024, 1024)).cuda()
    builder = ModelBuilder(num_warps=8, enable_profiling=args.intra_kernel_profile)
    batch = 3
    seq_len = 1234
    dtype = torch.bfloat16
    tp_size = 8
    num_q_heads = 64 // tp_size
    num_kv_heads = 8 // tp_size
    q_head_dim, v_head_dim = 128, 128
    rope_theta = 1000000
    rms_eps = 1e-6

    assert seq_len <= MAX_SEQ_LEN
    # input
    kv_lens = torch.tensor([123, 3, 3276], dtype=torch.int32, device=torch.cuda.current_device())
    assert kv_lens.shape[0] == batch
    qkv = torch.randn((batch, seq_len, num_q_heads + 2 * num_kv_heads, q_head_dim), dtype=dtype,
                      device=torch.cuda.current_device())

    # rms norm
    q_rms_weight = torch.randn(q_head_dim, dtype=dtype, device=torch.cuda.current_device())
    k_rms_weight = torch.randn(q_head_dim, dtype=dtype, device=torch.cuda.current_device())

    q_norm_rope = torch.empty(batch, seq_len, num_q_heads, q_head_dim, dtype=dtype, device=torch.cuda.current_device())
    k_norm_rope = torch.empty(batch, seq_len, num_kv_heads, v_head_dim, dtype=dtype, device=torch.cuda.current_device())
    v_split = torch.empty(batch, seq_len, num_kv_heads, v_head_dim, dtype=dtype, device=torch.cuda.current_device())

    assert q_head_dim == v_head_dim
    cos_cache, sin_cache = prepare_cos_sin_cache(q_head_dim, max_position_embeddings=MAX_SEQ_LEN, rope_theta=rope_theta)
    sin_cache = sin_cache.to(torch.float32)
    cos_cache = cos_cache.to(torch.float32)
    cos_sin_cache = torch.cat((cos_cache[:, :q_head_dim // 2], sin_cache[:, :q_head_dim // 2]), dim=-1)
    sin_cache = sin_cache.to(torch.float32).unsqueeze(0)
    cos_cache = cos_cache.to(torch.float32).unsqueeze(0)

    # mega kernel
    builder.make_qkv_pack_qk_norm_rope_split_v(qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache,
                                               q_norm_rope, k_norm_rope, v_split, q_rms_eps=rms_eps, k_rms_eps=rms_eps)

    builder.compile()
    ctx = get_torch_prof_ctx(profile)
    with ctx:
        for i in range(30):
            # mega impl
            l2_cache.zero_()
            tmp_input = torch.randn(qkv.shape, dtype=qkv.dtype).cuda()
            qkv.copy_(tmp_input)
            builder.run()

            # triton impl
            q_norm_rope_triton, k_norm_rope_triton, v_triton = triton_qkv_pack_qk_norm_rope_split_v(
                qkv, kv_lens, q_rms_weight, k_rms_weight, rms_eps, rms_eps, sin_cache, cos_cache, num_q_heads,
                num_kv_heads)

            # torch impl
            l2_cache.zero_()
            q, k, v_ref = torch.split(qkv.reshape(batch, seq_len, (num_q_heads + 2 * num_kv_heads), q_head_dim),
                                      [num_q_heads, num_kv_heads, num_kv_heads], dim=-2)
            q = q.contiguous()
            k = k.contiguous()
            v_ref = v_ref.contiguous()
            k_norm_ref = rmsnorm_ref(k, k_rms_weight, rms_eps)
            q_norm_ref = rmsnorm_ref(q, q_rms_weight, rms_eps)
            position_ids_list = []
            for b in range(batch):
                position_ids_list.append(
                    torch.arange(kv_lens[b], kv_lens[b] + seq_len, dtype=torch.int64, device="cuda").unsqueeze(0))
            position_ids = torch.cat(position_ids_list, dim=0)
            q_norm_rope_ref, k_norm_rope_ref = apply_rotary_pos_emb(q_norm_ref, k_norm_ref, position_ids, cos_sin_cache)

            torch.testing.assert_close(v_ref, v_triton, atol=0, rtol=0)
            torch.testing.assert_close(q_norm_rope_ref, q_norm_rope_triton, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(k_norm_rope_ref, k_norm_rope_triton, atol=1e-2, rtol=1e-2)

            torch.testing.assert_close(v_triton, v_split, atol=0, rtol=0)
            torch.testing.assert_close(q_norm_rope_triton, q_norm_rope, atol=0, rtol=0)
            torch.testing.assert_close(k_norm_rope_triton, k_norm_rope, atol=0, rtol=0)

    if args.intra_kernel_profile:
        builder.dump_trace()
        print(f"sm act = {builder.get_sm_activity()}")
    builder.finalize()

    if profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/qk_norm_rope.json.gz")
