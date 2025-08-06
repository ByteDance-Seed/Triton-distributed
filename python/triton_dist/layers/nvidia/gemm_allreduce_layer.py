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
import triton
from typing import Optional
from triton_dist.utils import nvshmem_barrier_all_on_stream
from triton_dist.kernels.nvidia import create_gemm_ar_context, low_latency_gemm_allreduce_op, create_ll_gemm_ar_context, gemm_allreduce_op


class GemmARLayer(torch.nn.Module):

    def __init__(
        self,
        tp_group: torch.distributed.ProcessGroup,
        max_M: int,
        N: int,
        K: int,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        local_world_size: int = -1,
        persistent: bool = True,
        use_ll_kernel: bool = False,
        copy_to_local: bool = True,
        NUM_COMM_SMS: int = 16,
        NUM_SM_MARGIN: int = 0,
        user_gemm_config: triton.Config = None,
    ):
        super().__init__()
        self.tp_group = tp_group
        self.rank: int = tp_group.rank()
        self.world_size = tp_group.size()
        self.local_world_size = local_world_size if local_world_size != -1 else self.world_size
        self.local_rank = self.rank % self.local_world_size
        self.use_ll_kernel = use_ll_kernel
        self.max_M: int = max_M
        self.N = N
        self.K = K
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        NUM_GEMM_SMS = NUM_SMS - NUM_COMM_SMS - NUM_SM_MARGIN
        self.NUM_GEMM_SMS = NUM_GEMM_SMS
        self.NUM_COMM_SMS = NUM_COMM_SMS
        self.USE_MULTIMEM_ST = True
        self.copy_to_local = copy_to_local
        self.user_gemm_config = user_gemm_config

        self._init_ctx()
        nvshmem_barrier_all_on_stream()
        self.persistent = persistent

    def _init_ctx(self):
        if self.use_ll_kernel:
            BM, BN, BK = 32, 64, 256
            num_stages = 5
            num_warps = 4
            self.ctx = create_ll_gemm_ar_context(self.rank, self.world_size, self.local_world_size, self.max_M, self.N,
                                                 self.output_dtype, NUM_COMM_SMS=self.NUM_COMM_SMS)
        else:
            BM, BN, BK = 128, 256, 64
            num_stages = 4
            num_warps = 8
            self.ar_stream = torch.cuda.Stream()
            self.ctx = create_gemm_ar_context(self.ar_stream, self.rank, self.world_size, self.local_world_size,
                                              self.max_M, self.N, self.output_dtype, NUM_COMM_SMS=self.NUM_COMM_SMS)
        self.gemm_config = triton.Config(
            {
                'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN, "BLOCK_SIZE_K": BK, "GROUP_SIZE_M": 1, "NUM_GEMM_SMS":
                self.NUM_GEMM_SMS
            }, num_stages=num_stages, num_warps=num_warps)
        if self.user_gemm_config is not None:
            self.gemm_config = self.user_gemm_config
        nvshmem_barrier_all_on_stream()

    def finalize(self):
        if self.ctx is not None:
            self.ctx.finalize()

    def forward(
        self,
        input: torch.Tensor,  # [M, local_K]
        weight: torch.Tensor,  # [N, local_K]
        bias: Optional[torch.Tensor] = None,
    ):
        assert input.shape[0] <= self.max_M and weight.shape[0] == self.N
        if self.use_ll_kernel:
            ar_out = low_latency_gemm_allreduce_op(self.ctx, input, weight, self.gemm_config,
                                                   copy_to_local=self.copy_to_local,
                                                   USE_MULTIMEM_ST=self.USE_MULTIMEM_ST)
        else:
            ar_out = gemm_allreduce_op(self.ctx, input, weight, self.gemm_config, copy_to_local=self.copy_to_local,
                                       USE_MULTIMEM_ST=self.USE_MULTIMEM_ST)
        return ar_out
