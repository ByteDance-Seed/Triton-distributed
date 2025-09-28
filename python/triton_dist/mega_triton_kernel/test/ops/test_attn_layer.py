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
import argparse
import os
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import (
    prepare_cos_sin_cache,
    rmsnorm_ref,
    apply_rotary_pos_emb,
    ref_paged_attn,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    parser.add_argument("--skip_rmsnorm", default=False, action="store_true", help="whether to skip rms norm")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(0)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    profile = args.profile
    l2_cache = torch.randn((256, 1024, 1024)).cuda()
    builder = ModelBuilder()
    batch = 1
    seq_len = 1
    PAGE_SIZE = 1
    MAX_SEQ_LEN = 32 * 1024  # 32k
    MAX_NUM_KV_BLOCKS = 128 * 1024
    MAX_NUM_BLOCKS_PER_SEQ = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    dtype = torch.bfloat16
    tp_size = 1
    num_q_heads = 64 // tp_size
    num_kv_heads = 8 // tp_size
    q_head_dim, v_head_dim = 128, 128
    hidden_size = 5120
    intermediate_size = 25600 // tp_size
    rope_theta = 1000000
    rms_eps = 1e-6
    skip_rmsnorm = args.skip_rmsnorm
    # qkv
    attn_layer_input = torch.randn((batch * seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())
    attn_layer_input_norm_out = torch.randn((batch * seq_len, hidden_size), dtype=dtype,
                                            device=torch.cuda.current_device())
    attn_layer_redisual_out = torch.randn((batch * seq_len, hidden_size), dtype=dtype,
                                          device=torch.cuda.current_device())
    qkv_proj_weight = torch.randn(((num_q_heads + num_kv_heads * 2) * q_head_dim, hidden_size), dtype=dtype).cuda() / 10
    qkv_proj_out = torch.randn((batch * seq_len, (num_q_heads + 2 * num_kv_heads) * q_head_dim), dtype=dtype,
                               device=torch.cuda.current_device())
    # rms norm
    rms_weight = torch.randn(q_head_dim, dtype=dtype, device=torch.cuda.current_device())
    pre_norm_weight = torch.randn(hidden_size, dtype=dtype, device=torch.cuda.current_device())
    post_norm_weight = torch.randn(hidden_size, dtype=dtype, device=torch.cuda.current_device())
    q_norm_rope = torch.empty(batch, seq_len, num_q_heads, q_head_dim, dtype=dtype, device=torch.cuda.current_device())
    # page attn
    key_cache = torch.randn(MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, q_head_dim, dtype=dtype).cuda()
    value_cache = torch.randn(MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, v_head_dim, dtype=dtype).cuda()
    block_tables = torch.randint(0, MAX_NUM_KV_BLOCKS, (batch, MAX_NUM_BLOCKS_PER_SEQ), dtype=torch.int32).cuda()
    block_tables_np = block_tables.cpu().numpy()
    kv_lens = torch.tensor([3], dtype=torch.int32, device=block_tables.device)
    sm_scale = q_head_dim**-0.5
    soft_cap = 0.0
    attn_out = torch.randn((batch, seq_len, num_q_heads, v_head_dim), dtype=dtype, device=attn_layer_input.device)
    assert q_head_dim == v_head_dim
    cos_cache, sin_cache = prepare_cos_sin_cache(q_head_dim, max_position_embeddings=MAX_SEQ_LEN, rope_theta=rope_theta)
    sin_cache = sin_cache.to(torch.float32)
    cos_cache = cos_cache.to(torch.float32)
    cos_sin_cache = torch.cat((cos_cache[:, :q_head_dim // 2], sin_cache[:, :q_head_dim // 2]), dim=-1)
    sin_cache = sin_cache.to(torch.float32).unsqueeze(0)
    cos_cache = cos_cache.to(torch.float32).unsqueeze(0)
    # out proj
    o_proj_weight = torch.randn((hidden_size, num_q_heads * q_head_dim), dtype=dtype).cuda() / 10
    o_proj_out = torch.randn((batch * seq_len, hidden_size), dtype=dtype, device=torch.cuda.current_device())
    builder.make_rms_norm(attn_layer_input, pre_norm_weight, attn_layer_input_norm_out, rms_eps)
    builder.make_qkv_proj(attn_layer_input_norm_out, qkv_proj_weight, qkv_proj_out)
    qkv_proj_out_bsnh = qkv_proj_out.reshape(batch, seq_len, (num_q_heads + 2 * num_kv_heads), q_head_dim)
    builder.make_qk_norm_rope_update_kvcache(qkv_proj_out_bsnh, key_cache, value_cache, block_tables, kv_lens,
                                             rms_weight, rms_weight, cos_cache, sin_cache, q_norm_rope, rms_eps,
                                             rms_eps, rope_theta, skip_q_norm=skip_rmsnorm, skip_k_norm=skip_rmsnorm)
    builder.make_flash_decode(q_norm_rope, key_cache, value_cache, block_tables, kv_lens, attn_out, sm_scale, soft_cap)
    attn_out_2d = attn_out.reshape(batch * seq_len, o_proj_weight.shape[1])
    builder.make_o_proj(attn_out_2d, o_proj_weight, o_proj_out)
    builder.make_add(attn_layer_input, o_proj_out, attn_layer_redisual_out)
    builder.compile()
    ctx = get_torch_prof_ctx(profile)

    with ctx:
        for i in range(30):
            l2_cache.zero_()
            tmp_input = torch.randn(attn_layer_input.shape, dtype=dtype).cuda()
            # inplace update kv lens
            kv_lens += seq_len
            attn_layer_input.copy_(tmp_input)
            builder.run()
            # torch impl
            l2_cache.zero_()
            attn_layer_input_norm_out_ref = rmsnorm_ref(attn_layer_input, pre_norm_weight, rms_eps)
            qkv_proj_out_ref = torch.nn.functional.linear(attn_layer_input_norm_out_ref, qkv_proj_weight)
            q, k, v = torch.split(
                qkv_proj_out_ref.reshape(batch, seq_len, (num_q_heads + 2 * num_kv_heads), q_head_dim),
                [num_q_heads, num_kv_heads, num_kv_heads], dim=-2)
            if not skip_rmsnorm:
                k_norm_ref = rmsnorm_ref(k, rms_weight, rms_eps)
                q_norm_ref = rmsnorm_ref(q, rms_weight, rms_eps)
            else:
                q_norm_ref = q
                k_norm_ref = k
            position_ids_list = []
            for b in range(batch):
                position_ids_list.append(
                    torch.arange(kv_lens[b] - seq_len, kv_lens[b], dtype=torch.int64, device="cuda").unsqueeze(0))
            position_ids = torch.cat(position_ids_list, dim=0)
            q_norm_rope_ref, k_norm_rope_ref = apply_rotary_pos_emb(q_norm_ref, k_norm_ref, position_ids, cos_sin_cache)
            attn_out_ref = ref_paged_attn(query=q_norm_rope_ref.reshape(batch, num_q_heads, q_head_dim),
                                          key_cache=key_cache, value_cache=value_cache, query_lens=[1] * batch,
                                          kv_lens=kv_lens, block_tables=block_tables, scale=sm_scale, soft_cap=soft_cap)
            o_proj_out_ref = torch.nn.functional.linear(attn_out_ref.reshape(-1, o_proj_weight.shape[1]), o_proj_weight)
            attn_layer_redisual_out_ref = o_proj_out_ref + attn_layer_input
            # get kv cache
            last_k_list = []
            last_v_list = []
            for b in range(batch):
                num_kv_blocks = (kv_lens[b] + PAGE_SIZE - 1) // PAGE_SIZE
                block_indices = block_tables[b, :num_kv_blocks]
                fetch_full_k = key_cache[block_indices].view(-1, num_kv_heads, q_head_dim)
                last_k = fetch_full_k[kv_lens[b] - seq_len:kv_lens[b]]
                fetch_full_v = value_cache[block_indices].view(-1, num_kv_heads, q_head_dim)
                last_v = fetch_full_v[kv_lens[b] - seq_len:kv_lens[b]]
                last_k_list.append(last_k)
                last_v_list.append(last_v)
            last_k = torch.cat(last_k_list, dim=0)
            last_v = torch.cat(last_v_list, dim=0)

            torch.testing.assert_close(q_norm_rope_ref, q_norm_rope, atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(last_k.reshape(k_norm_rope_ref.shape), k_norm_rope_ref, atol=2e-2, rtol=1e-2)
            torch.testing.assert_close(last_v, v.reshape(last_v.shape), atol=2e-2, rtol=1e-2)

    if profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/attn_layer.json.gz")
