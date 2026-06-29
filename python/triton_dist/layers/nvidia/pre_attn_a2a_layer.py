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
"""Pre-attention AllToAll operator using warp-specialized TMA pipeline.

Data layout
-----------
  Input:  (local_seq, total_nh, hd)  where total_nh = q_nh + k_nh + v_nh
  Output: Q (seq_len, local_q_nh, hd),
          K (seq_len, local_k_nh, hd),
          V (seq_len, local_v_nh, hd)

Uses a 2-warp kernel (producer TMA G2S from remote + consumer TMA S2G to local
Q/K/V) with multi-stage pipeline.
Aligned with Flux PreAttnQKVPackA2AOp TMA-store path.
"""
from typing import Tuple
import torch
import torch.distributed
import triton

from triton_dist.utils import (
    nvshmem_create_tensor,
    nvshmem_free_tensor_sync,
    nvshmem_barrier_all_on_stream,
)
from triton_dist.tma_utils import create_tmap_scratch
from triton_dist.kernels.nvidia.pre_attn_a2a import kernel_pre_attn_a2a

_SMEM_CAPACITY = 228 * 1024


def _compute_stages(bm, bn, elem_bytes=2):
    """Max pipeline stages that fit in shared memory (minus barrier overhead)."""
    mbar_overhead = 1024
    bytes_per_stage = bm * bn * elem_bytes
    return (_SMEM_CAPACITY - mbar_overhead) // bytes_per_stage


def _select_tile_bn(*col_widths, max_bn=128, min_bn=64):
    """Pick the largest BN <= max_bn that divides all non-zero col_widths."""
    bn = max_bn
    while bn >= min_bn:
        if all(w % bn == 0 for w in col_widths if w > 0):
            return bn
        bn //= 2
    raise ValueError(f"Cannot find BN in [{min_bn}, {max_bn}] dividing all of {col_widths}")


class PreAttnQKVPackA2AOp:
    """Pre-attention QKV Pack AllToAll via warp-specialized TMA pipeline.

    API mirrors Flux PreAttnQKVPackA2AOp.
    For single tensor mode, set k_nheads = v_nheads = 0.
    """

    def __init__(
        self,
        sp_group,
        max_seq_len,
        q_nheads,
        k_nheads,
        v_nheads,
        head_dim,
        dtype=torch.bfloat16,
    ):
        self.sp_group = sp_group
        self.max_seq_len = max_seq_len
        self.q_nheads = q_nheads
        self.k_nheads = k_nheads
        self.v_nheads = v_nheads
        self.head_dim = head_dim
        self.dtype = dtype
        self.world_size = sp_group.size()
        self.rank = sp_group.rank()

        assert q_nheads % self.world_size == 0, (
            f"q_nheads ({q_nheads}) must be divisible by world_size ({self.world_size})")
        assert k_nheads % self.world_size == 0 or k_nheads == 0
        assert v_nheads % self.world_size == 0 or v_nheads == 0
        assert max_seq_len % self.world_size == 0, (
            f"max_seq_len ({max_seq_len}) must be divisible by world_size ({self.world_size})")

        self.local_q_nh = q_nheads // self.world_size
        self.local_k_nh = k_nheads // self.world_size if k_nheads > 0 else 0
        self.local_v_nh = v_nheads // self.world_size if v_nheads > 0 else 0
        self.max_local_seq = max_seq_len // self.world_size
        self.total_nh = q_nheads + k_nheads + v_nheads

        self._comm_input_buf = nvshmem_create_tensor((self.max_local_seq, self.total_nh, self.head_dim),
                                                     dtype=self.dtype)

        torch.distributed.barrier(group=self.sp_group)
        torch.cuda.synchronize()

    def _canonicalize(self, inp):
        if inp.ndim == 4:
            assert inp.size(0) == 1, "batch must be 1 for 4D"
            inp = inp.squeeze(0)
        assert inp.ndim == 3
        assert inp.dtype == self.dtype and inp.is_cuda
        assert inp.size(1) == self.total_nh and inp.size(2) == self.head_dim
        local_seq = inp.size(0)
        assert local_seq <= self.max_local_seq
        return inp.contiguous()

    def finalize(self):
        nvshmem_free_tensor_sync(self._comm_input_buf)

    @torch.no_grad()
    def forward(
        self,
        input_tensor,
        *,
        q_output=None,
        k_output=None,
        v_output=None,
        num_sm=16,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inp3d = self._canonicalize(input_tensor)
        local_seq = inp3d.size(0)
        seq_len = local_seq * self.world_size

        self._comm_input_buf.narrow(0, 0, local_seq).copy_(inp3d)

        nvshmem_barrier_all_on_stream()

        BM = 128
        q_n = self.local_q_nh * self.head_dim
        k_n = self.local_k_nh * self.head_dim
        v_n = self.local_v_nh * self.head_dim
        BN = _select_tile_bn(q_n, k_n, v_n)
        NUM_STAGES = _compute_stages(BM, BN)
        assert NUM_STAGES >= 2, (f"BM={BM} BN={BN} yields only {NUM_STAGES} stages; need >=2 for pipeline")
        assert num_sm >= self.world_size, (f"num_sm ({num_sm}) must be >= world_size ({self.world_size})")
        PIPE_CNT = min(2, NUM_STAGES - 1)

        if q_output is None:
            q_output = torch.empty((seq_len, self.local_q_nh, self.head_dim), dtype=self.dtype, device="cuda")
        if k_output is None:
            k_output = torch.empty((seq_len, max(self.local_k_nh, 1), self.head_dim), dtype=self.dtype, device="cuda")
        if v_output is None:
            v_output = torch.empty((seq_len, max(self.local_v_nh, 1), self.head_dim), dtype=self.dtype, device="cuda")

        tmap_scratch = create_tmap_scratch(num_sm, num_descs_per_cta=4)

        def alloc_fn(size, alignment, stream):
            return torch.empty(size, device="cuda", dtype=torch.int8)

        triton.set_allocator(alloc_fn)

        kernel_pre_attn_a2a[(num_sm, )](
            self._comm_input_buf,
            q_output,
            k_output,
            v_output,
            tmap_scratch,
            local_seq,
            self.q_nheads,
            self.k_nheads,
            self.v_nheads,
            self.head_dim,
            self.world_size,
            self.rank,
            self.rank,
            BM,
            BN,
            NUM_STAGES,
            num_sm,
            PIPE_CNT,
            num_warps=2,
        )

        nvshmem_barrier_all_on_stream()
        return q_output, k_output, v_output


class PreAttnA2AOp(PreAttnQKVPackA2AOp):
    """Pre-attention AllToAll for single tensor (k_nheads=v_nheads=0).

    Input:  (local_seq, nh, hd)
    Output: (seq_len, local_nh, hd)
    """

    def __init__(self, sp_group, max_seq_len, nheads, head_dim, dtype=torch.bfloat16):
        super().__init__(sp_group=sp_group, max_seq_len=max_seq_len, q_nheads=nheads, k_nheads=0, v_nheads=0,
                         head_dim=head_dim, dtype=dtype)

    def forward(self, input_tensor, *, output=None, num_sm=16):
        q_out, _, _ = super().forward(input_tensor, q_output=output, num_sm=num_sm)
        return q_out
