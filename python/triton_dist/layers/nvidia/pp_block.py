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
from .p2p import CommOp

# Constants for PP communication
BUFFER_SIZE = 8
DOWN_OFFSET = 4
RECV_OFFSET = 2


# PyTorch native P2P implementation, adapted from test_pp.py
class PyTorchP2P:
    """
    A wrapper for PyTorch's native P2P communication (send/recv).
    This class manages buffers and communication logic for a pipeline stage.
    """

    def __init__(self, max_tokens, hidden_size, pp_rank, pp_size, pp_group, dtype=torch.bfloat16, num_buffers=8):
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.dtype = dtype
        self.num_buffers = num_buffers

        # Get the actual global ranks in our pipeline group for communication
        import torch.distributed as dist
        self.pp_group_ranks = dist.get_process_group_ranks(pp_group)

        # Allocate separate buffers for send and receive to avoid data corruption
        self.send_buffers = [
            torch.zeros([max_tokens, hidden_size], dtype=dtype, device="cuda") for _ in range(num_buffers)
        ]
        self.recv_buffers = [
            torch.zeros([max_tokens, hidden_size], dtype=dtype, device="cuda") for _ in range(num_buffers)
        ]

        # Buffer indices for different communication directions
        # This setup assumes 2 buffers for up-send, 2 for down-send, 2 for up-recv, 2 for down-recv
        self.up_send_idx = 0
        self.down_send_idx = 0
        self.up_recv_idx = 0
        self.down_recv_idx = 0

    def send(self, ts: torch.Tensor, rank: int, dst_rank: int):
        # Determine buffer based on communication direction (forward vs backward)
        if rank < dst_rank:  # Forward communication
            buffer_idx = self.up_send_idx
            self.up_send_idx = (self.up_send_idx + 1) % (self.num_buffers // 4)
        else:  # Backward communication
            buffer_idx = self.down_send_idx + (self.num_buffers // 2)
            self.down_send_idx = (self.down_send_idx + 1) % (self.num_buffers // 4)

        # Copy tensor to buffer and send
        self.send_buffers[buffer_idx].copy_(ts)
        # Convert PP rank to global rank for communication
        dst_global_rank = self.pp_group_ranks[dst_rank]
        # Use global rank without group parameter for correct communication
        torch.distributed.send(self.send_buffers[buffer_idx], dst=dst_global_rank)

    def recv(self, rank: int, src_rank: int):
        # Determine buffer based on communication direction
        if rank > src_rank:  # Receiving from previous stage (forward)
            buffer_idx = self.up_recv_idx + (self.num_buffers // 4)
            self.up_recv_idx = (self.up_recv_idx + 1) % (self.num_buffers // 4)
        else:  # Receiving from next stage (backward)
            buffer_idx = self.down_recv_idx + (3 * self.num_buffers // 4)
            self.down_recv_idx = (self.down_recv_idx + 1) % (self.num_buffers // 4)

        # Receive into buffer
        # Convert PP rank to global rank for communication
        src_global_rank = self.pp_group_ranks[src_rank]
        # Use global rank without group parameter for correct communication
        torch.distributed.recv(self.recv_buffers[buffer_idx], src=src_global_rank)

        return self.recv_buffers[buffer_idx]


class PPCommLayer:
    """Pipeline parallelism communication layer that can switch between
    a custom Triton-Dist backend (CommOp) and PyTorch's native backend.
    """

    def __init__(self, pp_rank: int, pp_size: int, pp_group: torch.distributed.ProcessGroup, max_tokens: int,
                 hidden_size: int, dtype: torch.dtype = torch.bfloat16, num_buffers: int = 8,
                 init_triton_backend: bool = False):

        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.pp_group = pp_group
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.num_buffers = num_buffers

        # Initialize backends
        self.pp_comm_op = None
        self.pytorch_p2p = None
        self._triton_initialized = False
        self._torch_initialized = False

        # Buffer tracking variables for Triton backend
        self.up_send_buffer_id = 0
        self.down_send_buffer_id = 0
        self.up_recv_buffer_id = 0
        self.down_recv_buffer_id = 0

        # All ranks must create NVSHMEM tensors together
        if init_triton_backend and pp_size > 1:
            self._init_triton_backend()

    def _init_triton_backend(self):
        """Lazy initialization of Triton-dist backend"""
        if not self._triton_initialized and self.pp_size > 1:
            self.pp_comm_op = CommOp(max_tokens=self.max_tokens, token_dim=self.hidden_size, pp_rank=self.pp_rank,
                                     pp_size=self.pp_size, pp_group=self.pp_group, dtype=self.dtype,
                                     num_buffers=self.num_buffers)
            self._triton_initialized = True

    def init_triton_backend_collective(self):
        """Initialize Triton-dist backend collectively across all ranks.
        This must be called by all ranks in the PP group to avoid hang."""
        if self.pp_size > 1:
            self._init_triton_backend()
            # Ensure all ranks have created their NVSHMEM tensors
            import torch.distributed as dist
            dist.barrier(group=self.pp_group)

    def _init_torch_backend(self):
        """Lazy initialization of PyTorch native backend"""
        if not self._torch_initialized and self.pp_size > 1:
            self.pytorch_p2p = PyTorchP2P(max_tokens=self.max_tokens, hidden_size=self.hidden_size,
                                          pp_rank=self.pp_rank, pp_size=self.pp_size, pp_group=self.pp_group,
                                          dtype=self.dtype, num_buffers=self.num_buffers)
            self._torch_initialized = True

    def _pp_put(self, ts, rank, dst_rank, num_sm):
        """Triton Dist PP put operation based on test_pp.py"""
        assert rank != dst_rank
        if rank < dst_rank:
            buffer_id = self.up_send_buffer_id
            dst_buffer_id = buffer_id + RECV_OFFSET
        else:
            buffer_id = self.down_send_buffer_id + DOWN_OFFSET
            dst_buffer_id = buffer_id + RECV_OFFSET
        self.pp_comm_op.wait_signal(dst_rank, dst_buffer_id, 0)
        self.pp_comm_op.write(dst_rank, buffer_id, dst_buffer_id, ts, 1, sm=num_sm)

    def _pp_get(self, rank, src_rank, num_sm):
        """Triton Dist PP get operation based on test_pp.py"""
        assert rank != src_rank
        if rank > src_rank:
            buffer_id = self.up_recv_buffer_id + RECV_OFFSET
        else:
            buffer_id = self.down_recv_buffer_id + DOWN_OFFSET + RECV_OFFSET
        self.pp_comm_op.wait_signal(rank, buffer_id, 1, num_barriers=num_sm)
        buffer = self.pp_comm_op.get_buffer(buffer_id)
        self.pp_comm_op.set_signal(rank, buffer_id, 0, num_barriers=num_sm)
        return buffer

    def triton_dist_send(self, hidden_states: torch.Tensor, dst_rank: int, num_sms: int = 8):
        """Triton Dist Send hidden states to the next PP stage"""
        self._init_triton_backend()
        self._pp_put(hidden_states, self.pp_rank, dst_rank, num_sms)

    def triton_dist_recv(self, src_rank: int, num_sms: int = 8, expected_shape: tuple = None):
        """Triton Dist Receive hidden states from the previous PP stage"""
        self._init_triton_backend()
        buffer = self._pp_get(self.pp_rank, src_rank, num_sms)
        num_elements = torch.prod(torch.tensor(expected_shape)).item()
        return buffer.flatten()[:num_elements].view(expected_shape)

    def torch_send(self, hidden_states: torch.Tensor, dst_rank: int):
        """(PyTorch) Send hidden states to the next PP stage"""
        self._init_torch_backend()
        self.pytorch_p2p.send(hidden_states, self.pp_rank, dst_rank)

    def torch_recv(self, src_rank: int, expected_shape: tuple = None):
        """(PyTorch) Receive hidden states from the previous PP stage"""
        self._init_torch_backend()
        buffer = self.pytorch_p2p.recv(self.pp_rank, src_rank)

        num_elements = torch.prod(torch.tensor(expected_shape)).item()
        return buffer.flatten()[:num_elements].view(expected_shape)

    def is_first_stage(self) -> bool:
        """Check if this is the first PP stage"""
        return self.pp_rank == 0

    def is_last_stage(self) -> bool:
        """Check if this is the last PP stage"""
        return self.pp_rank == self.pp_size - 1

    def get_prev_rank(self) -> int:
        """Get the rank of the previous PP stage"""
        if self.is_first_stage():
            return None
        return self.pp_rank - 1

    def get_next_rank(self) -> int:
        """Get the rank of the next PP stage"""
        if self.is_last_stage():
            return None
        return self.pp_rank + 1

    def send(self, x, dst_rank: int, backend: str = 'triton_dist', num_sms: int = 8):
        """Send hidden states to the next PP stage using the specified backend"""
        if backend == 'triton_dist':
            self.triton_dist_send(x, dst_rank, num_sms)
        elif backend == 'torch':
            self.torch_send(x, dst_rank)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def recv(self, src_rank: int, expected_shape: tuple = None, backend: str = 'triton_dist', num_sms: int = 8):
        """Receive hidden states from the previous PP stage using the specified backend"""
        if backend == 'triton_dist':
            return self.triton_dist_recv(src_rank, num_sms, expected_shape)
        elif backend == 'torch':
            return self.torch_recv(src_rank, expected_shape)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
