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
# Part of the code from nvshmem4py.
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# See COPYRIGHT.txt for license information
import torch
import nvshmem
import nvshmem.core
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync, nvshmem_barrier_all_on_stream, TorchStreamWrapper, nvshmem_signal_wait
from triton_dist.kernels.nvidia.p2p import p2p_copy_kernel, p2p_copy_remote_to_local_kernel
from triton_dist.language.extra import libshmem_device


class CommOp(torch.nn.Module):

    def __init__(
        self,
        max_tokens,
        token_dim,
        pp_rank,
        pp_size,
        pp_group,
        dtype,
        num_buffers,
        workspace_size=128,
    ):
        super().__init__()
        self.max_tokens = max_tokens
        self.token_dim = token_dim
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.dtype = dtype

        self._comm_buffers = [nvshmem_create_tensor([max_tokens, token_dim], dtype=dtype) for _ in range(num_buffers)]

        self._signals = [nvshmem_create_tensor([1], dtype=torch.int64) for _ in range(num_buffers)]

        self._workspaces = [nvshmem_create_tensor([workspace_size], dtype=torch.uint8) for _ in range(num_buffers)]

        for i in range(num_buffers):
            self._signals[i].zero_()

        compiled_kernel = p2p_copy_kernel.warmup(self._comm_buffers[0], 0, 0, grid=(1, ))
        compiled_kernel._init_handles()

        nvshmem_barrier_all_on_stream(torch.cuda.current_stream())

    def __del__(self):
        for t in self._comm_buffers:
            nvshmem_free_tensor_sync(t)
        for t in self._signals:
            nvshmem_free_tensor_sync(t)
        for t in self._workspaces:
            nvshmem_free_tensor_sync(t)

    def get_buffer(self, buffer_id):
        assert buffer_id < len(self._comm_buffers)
        return self._comm_buffers[buffer_id]

    def read(self, pp_rank, buffer_id, local_tensor, sm=4, stream=None, fused=False):
        assert buffer_id < len(self._comm_buffers)
        remote_pe = torch.distributed.get_global_rank(self.pp_group, pp_rank)
        stream = stream if stream is not None else torch.cuda.current_stream()
        if fused:
            p2p_copy_remote_to_local_kernel[(sm, )](self._comm_buffers[buffer_id], remote_pe, local_tensor,
                                                    local_tensor.nbytes, sm, BLOCK_SIZE=1024)
        else:
            p2p_copy_kernel[(sm, )](
                self._comm_buffers[buffer_id],
                remote_pe,
                local_tensor.nbytes,
            )
            local_tensor.copy_(self._comm_buffers[buffer_id].to(local_tensor.dtype)[:local_tensor.numel()])

    def set_signal(self, pp_rank, buffer_id, value, stream=None):
        assert buffer_id < len(self._comm_buffers)
        remote_pe = torch.distributed.get_global_rank(self.pp_group, pp_rank)
        stream = stream if stream is not None else torch.cuda.current_stream()
        nvshmem.core.put_signal(
            self._workspaces[buffer_id],
            self._workspaces[buffer_id],
            nvshmem.core.tensor_get_buffer(self._signals[buffer_id])[0],
            value,  # sig value
            nvshmem.core.SignalOp.SIGNAL_SET,
            remote_pe=remote_pe,
            stream=TorchStreamWrapper(stream),
        )

    def wait_signal(self, pp_rank, buffer_id, value, stream=None):
        assert buffer_id < len(self._comm_buffers)
        remote_pe = torch.distributed.get_global_rank(self.pp_group, pp_rank)
        stream = stream if stream is not None else torch.cuda.current_stream()
        nvshmem_signal_wait(
            self._signals[buffer_id],
            remote_pe,
            value,
            libshmem_device.NVSHMEM_CMP_EQ,
            stream=stream,
        )

    def wait_local_signal(self, buffer_id, value, stream=None):
        self.wait_signal(self.pp_rank, buffer_id, value, stream)
