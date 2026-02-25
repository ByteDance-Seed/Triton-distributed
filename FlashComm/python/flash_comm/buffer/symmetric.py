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
import torch.distributed as dist
import flash_comm._C.buffer as _buffer

# Expose Enums
BlockBackend = _buffer.BlockBackend


class SymmetricBuffer:

    def __init__(self, size: int, group: dist.ProcessGroup, backend: str = "auto"):
        """
        Allocate a symmetric buffer across the process group.
        
        Args:
            size (int): Size in bytes.
            group (dist.ProcessGroup): The process group to use.
            backend (str, optional): 
                - "auto" (default): Try VMM (Fabric), fallback to CUDA_IPC.
                - "vmm": Force VMM backend (requires Hopper or newer GPU with Fabric support).
                - "cuda_ipc": Force CUDA_IPC (cudaMalloc) backend.
        """
        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.global_rank = dist.get_rank()
        self.size = size

        # Check device support
        dev_id = torch.cuda.current_device()
        vmm_supported = _buffer.is_vmm_supported(dev_id)
        fabric_supported = _buffer.is_fabric_supported(dev_id)

        # Determine Backend configuration
        backend_enum = BlockBackend.CUDA_IPC

        # Validate inputs
        if backend not in ["auto", "vmm", "cuda_ipc"]:
            raise ValueError(f"Invalid backend: {backend}")

        # Logic Resolution
        if backend == "cuda_ipc":
            backend_enum = BlockBackend.CUDA_IPC

        elif backend == "vmm":
            if not vmm_supported:
                raise RuntimeError("VMM backend requested but not supported on this device.")
            if not fabric_supported:
                raise RuntimeError("VMM backend requires Fabric support.")
            backend_enum = BlockBackend.VMM

        else:  # backend == "auto"
            if vmm_supported and fabric_supported:
                backend_enum = BlockBackend.VMM
            else:
                backend_enum = BlockBackend.CUDA_IPC

        # 1. Allocate local memory
        self._mem = _buffer.SymmetricMemory(size, backend_enum)
        self.backend, self.handle_bytes = self._mem.get_handle()

        # 2. Exchange Handles
        all_handles = [None] * self.world_size
        dist.all_gather_object(all_handles, self.handle_bytes, group=self.group)

        # 3. Register Peers
        for i, h_bytes in enumerate(all_handles):
            if i == self.rank:
                continue
            self._mem.register_peer(i, h_bytes)

    def get_local_ptr(self):
        return self._mem.get_local_ptr()

    def get_peer_ptr(self, peer_rank):
        return self._mem.get_peer_ptr(peer_rank)

    def get_peer_ptrs_tensor(self):
        """
        Returns a CUDA tensor containing pointers to symmetric buffers of all ranks.
        Tensor shape: [world_size], dtype: torch.int64
        """
        ptrs = []
        for i in range(self.world_size):
            if i == self.rank:
                ptrs.append(self.get_local_ptr())
            else:
                p = self.get_peer_ptr(i)
                if p == 0:
                    raise RuntimeError(f"Peer pointer for rank {i} is null")
                ptrs.append(p)
        return torch.tensor(ptrs, dtype=torch.int64, device='cuda')


class SymmetricTensor:

    def __init__(self, shape: tuple, dtype: torch.dtype, group: dist.ProcessGroup, backend: str = "auto"):
        """
        A tensor-like wrapper around SymmetricBuffer.
        
        Args:
            shape (tuple): Shape of the tensor.
            dtype (torch.dtype): Data type of the tensor.
            group (dist.ProcessGroup): The process group to use.
            backend (str, optional): Backend strategy ("auto", "vmm", "cuda_ipc").
        """
        self.shape = shape
        self.dtype = dtype

        element_size = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel = numel
        self.size_bytes = numel * element_size

        self.buffer = SymmetricBuffer(self.size_bytes, group, backend)

    def get_local_tensor(self):
        return self.get_peer_tensor(self.buffer.rank)

    def get_peer_tensor(self, rank):
        ptr = 0
        if rank == self.buffer.rank:
            ptr = self.buffer.get_local_ptr()
        else:
            ptr = self.buffer.get_peer_ptr(rank)

        if ptr == 0:
            raise RuntimeError(f"Pointer for rank {rank} is null")

        current_device = torch.cuda.current_device()
        return _buffer.create_tensor_from_ptr(ptr, list(self.shape), self.dtype, current_device)

    @property
    def ptrs(self):
        return self.buffer.get_peer_ptrs_tensor()
