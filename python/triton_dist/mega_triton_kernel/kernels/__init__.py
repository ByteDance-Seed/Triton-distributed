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
from .mlp_fc1 import fc1_task_compute
from .task_context import TaskBaseInfo
from .activation import silu_mul_up_task_compute
from .flash_decode import attn_gqa_fwd_batch_decode_combine_task_compute, attn_gqa_fwd_batch_decode_split_kv_task_compute
from .flash_attn import qkv_pack_flash_attn_task_compute, flash_attn_task_compute
from .norm import rmsnorm_rope_update_kv_cache_task_compute, rmsnorm_task_compute, qkv_pack_qk_norm_rope_split_v_task_compute
from .elementwise import add_task_compute
from .allreduce import allreduce_task_compute
from .barrier import barrier_all_intra_node_task_compute
from .linear import linear_task_compute
from .prefetch import prefetch_task_compute

__all__ = [
    "fc1_task_compute",
    "TaskBaseInfo",
    "silu_mul_up_task_compute",
    "attn_gqa_fwd_batch_decode_combine_task_compute",
    "attn_gqa_fwd_batch_decode_split_kv_task_compute",
    "rmsnorm_rope_update_kv_cache_task_compute",
    "rmsnorm_task_compute",
    "add_task_compute",
    "allreduce_task_compute",
    "barrier_all_intra_node_task_compute",
    "linear_task_compute",
    "prefetch_task_compute",
    "qkv_pack_flash_attn_task_compute",
    "flash_attn_task_compute",
    "qkv_pack_qk_norm_rope_split_v_task_compute",
]
