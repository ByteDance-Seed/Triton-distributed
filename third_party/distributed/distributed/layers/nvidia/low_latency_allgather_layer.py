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
import pynvshmem
import torch
import torch.distributed
from triton.distributed.kernels.nvidia import _forward_push_2d_ll_kernel, _forward_push_2d_kernel, _forward_pull_kernel, _forward_push_2d_ll_multimem_kernel


class AllGatherLayer:

    def __init__(self, nnodes, world_size, rank, max_buffer_size: int = 2 * 32 * 128 * 128, stages=2):
        self.rank = rank
        self.size = world_size
        self.signal = pynvshmem.nvshmem_create_tensor((
            stages,
            self.size,
        ), torch.uint64)
        self.signal_bar = pynvshmem.nvshmem_create_tensor((
            stages,
            self.size,
        ), torch.uint64)
        self.max_buffer_size = max_buffer_size
        self.ll_buffers = pynvshmem.nvshmem_create_tensor((
            stages,
            self.max_buffer_size,
        ), torch.int8)
        self.signal_target = 15  # avoid 1 to constexpr
        for i in range(stages):
            self.signal[i].zero_()
            self.signal_bar[i].fill_(self.signal_target)
        self.nnodes = nnodes
        self.stages = stages

    def forward_pull(self, symm_buffer: torch.Tensor):
        _forward_pull_kernel[(self.size, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            self.signal,
            self.size,
            self.rank,
            self.signal_target,
            num_warps=32,
        )
        self.signal_target += 1
        return symm_buffer

    def forward_push_2d(self, symm_buffer: torch.Tensor):
        _forward_push_2d_kernel[(self.size // self.nnodes, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            self.signal,
            self.nnodes,
            self.size,
            self.rank,
            self.signal_target,
            num_warps=32,
        )
        self.signal_target += 1
        return symm_buffer

    def forward_push_2d_ll(self, symm_buffer: torch.Tensor):
        assert symm_buffer.nbytes * 2 < self.max_buffer_size
        signal = self.signal[self.signal_target % self.stages]
        ll_buffer = self.ll_buffers[self.signal_target % self.stages]
        _forward_push_2d_ll_kernel[(self.size, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            signal,
            ll_buffer,
            self.nnodes,
            self.size,
            self.rank,
            self.signal_target,
            num_warps=32,
        )
        self.signal_target += 1
        return symm_buffer

    def forward_push_2d_ll_multimem(self, symm_buffer: torch.Tensor):
        assert symm_buffer.nbytes * 2 < self.max_buffer_size
        ll_buffer = self.ll_buffers[self.signal_target % self.stages]
        _forward_push_2d_ll_multimem_kernel[(self.size, )](
            symm_buffer,
            symm_buffer.nbytes // self.size,
            ll_buffer,
            self.nnodes,
            self.size,
            self.rank,
            self.signal_target,
            num_warps=32,
        )
        self.signal_target += 1

        return symm_buffer
