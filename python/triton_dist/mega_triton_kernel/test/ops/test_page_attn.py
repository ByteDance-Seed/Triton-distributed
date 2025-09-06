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
import os
import argparse
import torch
from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.utils import get_torch_prof_ctx
from triton_dist.mega_triton_kernel.test.torch_impl_utils import ref_paged_attn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=False, action="store_true", help="enable profiling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    l2_cache = torch.randn((256, 1024, 1024)).cuda()

    builder = ModelBuilder()
    batch = 1
    seq_len = 1
    PAGE_SIZE = 1
    MAX_SEQ_LEN = 32 * 1024  # 32k
    MAX_NUM_BLOCKS_PER_SEQ = (MAX_SEQ_LEN + PAGE_SIZE - 1) // PAGE_SIZE
    MAX_NUM_KV_BLOCKS = MAX_NUM_BLOCKS_PER_SEQ * batch
    dtype = torch.bfloat16
    tp_size = 8
    num_q_heads = 64 // tp_size
    num_kv_heads = 8 // tp_size
    q_head_dim, v_head_dim = 128, 128

    key_cache = torch.randn(MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, q_head_dim, dtype=dtype).cuda()
    value_cache = torch.randn(MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, v_head_dim, dtype=dtype).cuda()
    block_tables = torch.randint(0, MAX_NUM_KV_BLOCKS, (batch, MAX_NUM_BLOCKS_PER_SEQ), dtype=torch.int32).cuda()
    kv_lens = torch.tensor([103], dtype=torch.int32, device=block_tables.device)
    query = torch.randn((batch, seq_len, num_q_heads, q_head_dim), dtype=dtype, device=torch.cuda.current_device())

    attn_out = torch.randn((batch, seq_len, num_q_heads, v_head_dim), dtype=dtype, device=query.device)
    sm_scale = q_head_dim**-0.5
    soft_cap = 0.0

    num_layers = 1
    builder.make_flash_decode(query, key_cache, value_cache, block_tables, kv_lens, attn_out, sm_scale, soft_cap)
    builder.compile()

    ctx = get_torch_prof_ctx(args.profile)
    with ctx:
        for i in range(30):
            l2_cache.zero_()
            tmp_input = torch.randn((batch, seq_len, num_q_heads, q_head_dim), dtype=dtype).cuda()
            query.copy_(tmp_input)
            builder.run()

            # torch impl
            l2_cache.zero_()
            attn_out_ref = ref_paged_attn(query=query.reshape(batch, num_q_heads, q_head_dim), key_cache=key_cache,
                                          value_cache=value_cache, query_lens=[1] * batch, kv_lens=kv_lens,
                                          block_tables=block_tables, scale=sm_scale, soft_cap=soft_cap)
            torch.testing.assert_close(attn_out_ref.reshape(attn_out.shape), attn_out, atol=1e-2, rtol=1e-2)

    if args.profile:
        import os
        prof_dir = "prof/"
        os.makedirs(prof_dir, exist_ok=True)
        ctx.export_chrome_trace(f"{prof_dir}/paged_attn.json.gz")
