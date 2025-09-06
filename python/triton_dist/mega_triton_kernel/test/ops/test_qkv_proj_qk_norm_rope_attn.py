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
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import (
    prepare_cos_sin_cache,
    rmsnorm_ref,
    apply_rotary_pos_emb,
)
from triton_dist.mega_triton_kernel.test.triton_impl_utils import (
    triton_attn, )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable kernel level profiling")
    parser.add_argument("--intra_kernel_profile", default=False, action="store_true",
                        help="enable intra kernel profiling")

    return parser.parse_args()


def attention_ref(q, k, v, IS_CAUSAL):
    HEAD_DIM = q.shape[-1]
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


def default_alloc_fn(size: int, align: int, _):
    return torch.empty(size, dtype=torch.int8, device="cuda")


if __name__ == "__main__":
    args = parse_args()

    torch.cuda.set_device(0)
    # torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    profile = args.profile
    MAX_SEQ_LEN = 65536
    l2_cache = torch.randn((256, 1024, 1024)).cuda()
    builder = ModelBuilder(num_warps=8, enable_profiling=args.intra_kernel_profile)
    batch = 1
    seq_len = 3278
    dtype = torch.bfloat16
    tp_size = 8
    hidden_size = 5120
    num_q_heads = 64 // tp_size
    num_kv_heads = 8 // tp_size
    q_head_dim, v_head_dim = 128, 128
    rope_theta = 1000000
    qkv_proj_dim = (num_q_heads + num_kv_heads * 2) * q_head_dim
    rms_eps = 1e-6
    IS_CAUSAL = True

    triton.set_allocator(default_alloc_fn)

    assert seq_len <= MAX_SEQ_LEN
    # input
    attn_layer_input = torch.randn((batch * seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())

    # qkv
    qkv_proj_weight = torch.randn((qkv_proj_dim, hidden_size), dtype=dtype, device=torch.cuda.current_device())

    kv_lens = torch.tensor([123], dtype=torch.int32, device=torch.cuda.current_device())
    assert kv_lens.shape[0] == batch

    qkv = torch.empty((batch, seq_len, num_q_heads + 2 * num_kv_heads, q_head_dim), dtype=dtype,
                      device=torch.cuda.current_device())
    qkv_proj_out = qkv.reshape(batch * seq_len, qkv_proj_dim)

    # rms norm
    q_rms_weight = torch.randn(q_head_dim, dtype=dtype, device=torch.cuda.current_device())
    k_rms_weight = torch.randn(q_head_dim, dtype=dtype, device=torch.cuda.current_device())

    q_norm_rope = torch.empty(batch, seq_len, num_q_heads, q_head_dim, dtype=dtype, device=torch.cuda.current_device())
    k_norm_rope = torch.empty(batch, seq_len, num_kv_heads, v_head_dim, dtype=dtype, device=torch.cuda.current_device())
    v_split = torch.empty(batch, seq_len, num_kv_heads, v_head_dim, dtype=dtype, device=torch.cuda.current_device())

    assert q_head_dim == v_head_dim
    # rope
    cos_cache, sin_cache = prepare_cos_sin_cache(q_head_dim, max_position_embeddings=MAX_SEQ_LEN, rope_theta=rope_theta)
    sin_cache = sin_cache.to(torch.float32)
    cos_cache = cos_cache.to(torch.float32)
    cos_sin_cache = torch.cat((cos_cache[:, :q_head_dim // 2], sin_cache[:, :q_head_dim // 2]), dim=-1)
    sin_cache = sin_cache.to(torch.float32).unsqueeze(0)
    cos_cache = cos_cache.to(torch.float32).unsqueeze(0)

    # attn
    attn_out = torch.empty((batch, seq_len, num_q_heads, q_head_dim), dtype=dtype, device=torch.cuda.current_device())

    # mega kernel
    builder.make_qkv_proj(attn_layer_input, qkv_proj_weight, qkv_proj_out)
    assert qkv.data_ptr() == qkv_proj_out.data_ptr()
    builder.make_qkv_pack_qk_norm_rope_split_v(qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache,
                                               q_norm_rope, k_norm_rope, v_split, q_rms_eps=rms_eps, k_rms_eps=rms_eps)
    builder.make_flash_attn(q_norm_rope, k_norm_rope, v_split, attn_out, is_causal=IS_CAUSAL)
    ctx = get_torch_prof_ctx(profile)

    with ctx:
        builder.compile()
        for i in range(100):
            # mega impl
            l2_cache.zero_()
            tmp_input = torch.rand(attn_layer_input.shape, dtype=attn_layer_input.dtype).cuda()
            attn_layer_input.copy_(tmp_input)
            builder.run()

            # torch impl
            l2_cache.zero_()
            qkv_ref = torch.nn.functional.linear(attn_layer_input, qkv_proj_weight)
            qkv_ref = qkv_ref.reshape(batch, seq_len, (num_q_heads + 2 * num_kv_heads), q_head_dim)
            q, k, v_ref = torch.split(qkv_ref, [num_q_heads, num_kv_heads, num_kv_heads], dim=-2)
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
            attn_out_ref = triton_attn([q_norm_rope_ref, k_norm_rope_ref, v_ref], batch, seq_len, num_q_heads,
                                       num_kv_heads, q_head_dim, IS_CAUSAL=IS_CAUSAL, qkv_pack=False)

            torch.testing.assert_close(v_ref, v_split, atol=0, rtol=0)
            torch.testing.assert_close(qkv_ref, qkv, atol=0, rtol=0)
            torch.testing.assert_close(q_norm_rope_ref, q_norm_rope, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(k_norm_rope_ref, k_norm_rope, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(attn_out_ref, attn_out, atol=3e-2, rtol=2e-2)

    if args.intra_kernel_profile:
        builder.dump_trace()
        print(f"sm act = {builder.get_sm_activity()}")
    builder.finalize()

    if profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/qkv_proj_qk_norm_rope_attn.json.gz")
