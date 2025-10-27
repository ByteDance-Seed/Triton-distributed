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
from triton_dist.kernels.nvidia.ulysses_sp_dispatch import create_ulysses_sp_pre_attn_comm_context, pre_attn_qkv_pack_a2a_op


class UlyssesSPAllToAllLayer(torch.nn.Module):

    def __init__(self, sp_group, bs, max_seq, q_nheads, kv_nheads, k_head_dim, v_head_dim, group_size, dtype,
                 local_world_size=8):

        self.pre_attn_a2a_ctx = create_ulysses_sp_pre_attn_comm_context(bs=bs, max_seq=max_seq, q_nheads=q_nheads,
                                                                        k_head_dim=k_head_dim, v_head_dim=v_head_dim,
                                                                        group_size=group_size, dtype=dtype,
                                                                        local_world_size=local_world_size, pg=sp_group)

    def pre_attn_qkv_pack_a2a(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, skip_q_a2a=False,
                              return_comm_buf=False):
        assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
        out_q, out_k, out_v = pre_attn_qkv_pack_a2a_op(self.pre_attn_a2a_ctx, q, k, v, skip_q_a2a=skip_q_a2a,
                                                       return_comm_buf=return_comm_buf)
        return out_q, out_k, out_v

    def finalize(self):
        self.pre_attn_a2a_ctx.finalize()

    def get_comm_nbytes(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, skip_q_a2a=False):
        elem_size = q.element_size()
        local_world_size = self.pre_attn_a2a_ctx.local_world_size
        bs, local_seq, q_nheads, k_head_dim = q.shape
        kv_nheads, v_head_dim = v.shape[-2], v.shape[-1]
        local_q_nheads = self.pre_attn_a2a_ctx.local_q_nheads(q_nheads)
        local_kv_nheads = self.pre_attn_a2a_ctx.local_kv_nheads(q_nheads)
        nnodes = self.pre_attn_a2a_ctx.world_size // local_world_size
        # wo opt
        if not skip_q_a2a:
            inter_node_nbytes_wo_opt = (nnodes - 1) * local_world_size * bs * local_seq * (
                local_q_nheads * k_head_dim + 2 * local_kv_nheads * v_head_dim) * elem_size
            intra_node_nbytes_wo_opt = nnodes * local_world_size * bs * local_seq * (
                local_q_nheads * k_head_dim + 2 * local_kv_nheads * v_head_dim) * elem_size
        else:
            inter_node_nbytes_wo_opt = (nnodes - 1) * local_world_size * bs * local_seq * (2 * local_kv_nheads *
                                                                                           v_head_dim) * elem_size
            intra_node_nbytes_wo_opt = nnodes * local_world_size * bs * local_seq * (2 * local_kv_nheads *
                                                                                     v_head_dim) * elem_size
        if kv_nheads >= self.pre_attn_a2a_ctx.world_size:
            return inter_node_nbytes_wo_opt, intra_node_nbytes_wo_opt, inter_node_nbytes_wo_opt, intra_node_nbytes_wo_opt

        # with opt
        group_size = self.pre_attn_a2a_ctx.group_size
        q_nheads_per_node = local_q_nheads * local_world_size
        inter_node_nbytes_with_opt = 0
        for node_id in range(1, nnodes):
            q_nheads_start = node_id * q_nheads_per_node
            q_nheads_end = q_nheads_start + q_nheads_per_node
            kv_nheads_start = q_nheads_start // group_size
            kv_nheads_end = (q_nheads_end + group_size - 1) // group_size
            if not skip_q_a2a:
                num_comm_nheads = (kv_nheads_end - kv_nheads_start) * 2 * v_head_dim + q_nheads_per_node * k_head_dim
            else:
                num_comm_nheads = (kv_nheads_end - kv_nheads_start) * 2 * v_head_dim
            inter_node_nbytes_with_opt += bs * local_seq * num_comm_nheads * elem_size
        if not skip_q_a2a:
            intra_node_nbytes_with_opt = nnodes * local_world_size * bs * local_seq * (
                local_q_nheads * k_head_dim + 2 * local_kv_nheads * v_head_dim) * elem_size
        else:
            intra_node_nbytes_with_opt = nnodes * local_world_size * bs * local_seq * (2 * local_kv_nheads *
                                                                                       v_head_dim) * elem_size
        return inter_node_nbytes_wo_opt, intra_node_nbytes_wo_opt, inter_node_nbytes_with_opt, intra_node_nbytes_with_opt
