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
import os

# Expose Enums
BlockBackend = _buffer.BlockBackend

VALID_BACKENDS = ("auto", "vmm", "cuda_ipc", "nccl", "torch_ipc")


def _torch_ipc_available() -> bool:
    """torch_ipc shares a torch-allocated buffer through torch's own CUDA IPC
    reduction (``torch.multiprocessing.reductions``). Availability is
    deterministic across ranks running the same torch build, so ``auto``
    resolution can never desync the collective handshake.
    """
    try:
        from torch.multiprocessing.reductions import (  # noqa: F401
            reduce_tensor, rebuild_cuda_tensor,
        )
        return True
    except Exception:
        return False


class _CppSymmMem:
    """cuda_ipc / vmm backend: physical memory owned by the C++ ShareableBlock.

    Memory comes from a raw cudaMalloc / cuMemCreate, so it is invisible to the
    torch caching allocator. Handles are fixed-size bytes exchanged via collective.
    """

    needs_exchange = True

    def __init__(self, size: int, backend_enum):
        self._mem = _buffer.SymmetricMemory(size, backend_enum)

    def local_ptr(self) -> int:
        return self._mem.get_local_ptr()

    def export_handle(self):
        # get_handle() -> (BlockBackend, bytes); only the bytes are exchanged.
        return self._mem.get_handle()[1]

    def import_peer(self, peer_local_rank: int, handle) -> None:
        self._mem.register_peer(peer_local_rank, handle)

    def peer_ptr(self, peer_local_rank: int) -> int:
        return self._mem.get_peer_ptr(peer_local_rank)


class _NcclSymmMem:
    """nccl backend: ncclMemAlloc + NCCL window; peer access via the window, so
    no explicit handle handshake is required here."""

    needs_exchange = False

    def __init__(self, size: int):
        self._mem = _buffer.NcclSymmetricMemory(size)

    def local_ptr(self) -> int:
        return self._mem.get_local_ptr()

    def peer_ptr(self, peer_local_rank: int) -> int:
        return self._mem.get_peer_ptr(peer_local_rank)

    def window_handle(self) -> int:
        return int(self._mem.get_window_handle())


class _TorchIpcSymmMem:
    """Intranode backend whose memory is owned by torch's caching allocator.

    Correct N-way sharing (the leak fix)
    ------------------------------------
    torch keeps one IPC reference counter per shared *storage*. Each
    ``reduce_tensor`` call (one ``_share_cuda_``) charges that counter by +1;
    each consumer that rebuilds the storage and later drops it releases -1. The
    producer storage is reclaimed only when the counter returns to zero, so the
    invariant is **#exports == #consumer releases**.

    Cross-device retarget
    ---------------------
    ``reduce_tensor`` records the *producer's* device index, and
    ``rebuild_cuda_tensor`` does ``cudaSetDevice(storage_device)`` before
    ``cudaIpcOpenMemHandle`` -- so the mapping and its lazily-enabled peer access
    are created for whatever device is current at open time. In EP each rank
    drives a *different* physical GPU, so the producer's index is meaningless to
    the consumer; left as-is the peer pointer is not accessible from the
    consumer's GPU (illegal access inside the kernel). We rewrite that device
    field to the consumer's current device so the mapping targets the GPU that
    will actually dereference it.

    Limitation: CUDA IPC is single-node only -- use the ``vmm`` / ``nccl``
    backend to reach peers across an NVLink Fabric (MNNVL).
    """

    needs_exchange = True

    def __init__(self, size: int, local_world_size: int, local_rank: int):
        from torch.multiprocessing.reductions import reduce_tensor
        self._local_rank = local_rank
        self._buf = torch.empty(size, dtype=torch.uint8, device="cuda")
        # One dedicated export per consumer peer (see class docstring): the IPC
        # counter is charged once per export, and each peer releases exactly once,
        # so the counter returns to zero on teardown instead of underflowing.
        self._exports = {peer: reduce_tensor(self._buf) for peer in range(local_world_size) if peer != local_rank}
        # peer_local_rank -> rebuilt torch tensor (the consumer-side reference).
        self._peers = {}

    def local_ptr(self) -> int:
        return self._buf.data_ptr()

    def export_handle(self):
        # {consumer_local_rank: (rebuild_fn, args)} -- each consumer takes the
        # export minted for it, so every export is consumed exactly once.
        return self._exports

    def import_peer(self, peer_local_rank: int, peer_bundle) -> None:
        rebuild_fn, args = peer_bundle[self._local_rank]
        args = self._retarget_to_consumer_device(rebuild_fn, args)
        self._peers[peer_local_rank] = rebuild_fn(*args)

    def peer_ptr(self, peer_local_rank: int) -> int:
        t = self._peers.get(peer_local_rank)
        return t.data_ptr() if t is not None else 0

    @staticmethod
    def _retarget_to_consumer_device(rebuild_fn, args):
        """Point the IPC open at the consumer's GPU (see class docstring). Only
        the known ``rebuild_cuda_tensor`` layout is touched; any other rebuild
        schema is passed through untouched so we never corrupt unknown args."""
        from torch.multiprocessing.reductions import rebuild_cuda_tensor
        if rebuild_fn is not rebuild_cuda_tensor:
            return args
        args = list(args)
        # rebuild_cuda_tensor(tensor_cls, size, stride, offset, storage_cls,
        #                     dtype, storage_device, storage_handle, ...):
        # storage_device is positional index 6 and has been stable across torch
        # versions (verified through 2.9).
        args[6] = torch.cuda.current_device()
        return tuple(args)

    def release_peers(self) -> None:
        """Consumer side: drop every rebuilt peer tensor. Each drop releases one
        ref on the producer's IPC counter; with one export minted per consumer
        the counter returns to zero so the producer storage becomes collectible."""
        self._peers.clear()

    def free_local(self) -> None:
        """Producer side: drop the storage and its per-peer exports. Once every
        consumer has released (counter == 0) the storage moves to torch's IPC
        limbo, which ``ipc_collect`` (driven by free_symmetric_tensors) reclaims."""
        self._buf = None
        self._exports = None


class SymmetricBuffer:

    def __init__(self, size: int, group: dist.ProcessGroup, backend: str = "auto", local_world_size: int = 0):
        """
        Allocate a symmetric buffer across the process group.

        Args:
            size (int): Size in bytes.
            group (dist.ProcessGroup): The process group to use.
            backend (str, optional):
                - "auto" (default): VMM (Fabric) if available (MNNVL-safe),
                  else CUDA_IPC. Never resolves to torch_ipc (see below).
                - "torch_ipc": Allocate via torch's caching allocator and share
                  through torch's own CUDA IPC reduction (reduce_tensor), one
                  dedicated export per local peer so the IPC ref-counter balances
                  and the buffer is reclaimed leak-free on teardown. Single-node
                  only; allocator-aware (works with the default allocator and
                  expandable_segments). Opt-in only and never chosen by "auto".
                - "vmm": Force VMM backend (Hopper+ with Fabric support).
                - "cuda_ipc": Force CUDA_IPC (raw cudaMalloc) backend.
                - "nccl": Allocate with ncclMemAlloc and register a NCCL window.
        """
        self.group = group
        self.group_rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        self.local_world_size = local_world_size or self.world_size
        if self.local_world_size <= 0 or self.world_size % self.local_world_size != 0:
            raise ValueError("local_world_size must divide group world_size")
        self.local_rank = self.group_rank % self.local_world_size
        self.node_rank_begin = self.group_rank - self.local_rank
        self.size = size

        # Check device support
        dev_id = torch.cuda.current_device()
        vmm_supported = _buffer.is_vmm_supported(dev_id)
        fabric_supported = _buffer.is_fabric_supported(dev_id)

        # The env override only applies to "auto" call sites. Explicit backends
        # must be honored as-is, so internode EP can mix env-driven intranode
        # buffers with the "nccl" buffers it requests directly (full_splits /
        # rdma_rail) without the env var clobbering them.
        if backend == "auto" and os.environ.get("FLASH_COMM_BUFFER_BACKEND", None) is not None:
            backend = os.environ.get("FLASH_COMM_BUFFER_BACKEND")

        if backend not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend: {backend}")

        if backend == "auto":
            if vmm_supported and fabric_supported:
                backend = "vmm"
            else:
                backend = "cuda_ipc"

        self.backend = backend

        # Construct the backend strategy.
        if backend == "nccl":
            self._backend = _NcclSymmMem(size)
        elif backend == "torch_ipc":
            if not _torch_ipc_available():
                raise RuntimeError("torch_ipc backend requires torch.multiprocessing CUDA IPC "
                                   "reduction helpers, which are unavailable in this torch build.")
            self._backend = _TorchIpcSymmMem(size, self.local_world_size, self.local_rank)
        elif backend == "cuda_ipc":
            self._backend = _CppSymmMem(size, BlockBackend.CUDA_IPC)
        elif backend == "vmm":
            if not vmm_supported:
                raise RuntimeError("VMM backend requested but not supported on this device.")
            if not fabric_supported:
                raise RuntimeError("VMM backend requires Fabric support.")
            self._backend = _CppSymmMem(size, BlockBackend.VMM)

        # Exchange handles and register peers in the local LSA/NVL team only.
        # Cross-node peers are reached through NCCL windows/GIN, not VA pointers.
        if getattr(self._backend, "needs_exchange", False):
            all_handles = [None] * self.world_size
            dist.all_gather_object(all_handles, self._backend.export_handle(), group=self.group)
            for local_rank in range(self.local_world_size):
                peer_group_rank = self.node_rank_begin + local_rank
                if peer_group_rank == self.group_rank:
                    continue
                self._backend.import_peer(local_rank, all_handles[peer_group_rank])

    def get_local_ptr(self):
        return self._backend.local_ptr()

    def get_peer_ptr(self, peer_rank):
        if peer_rank < 0 or peer_rank >= self.local_world_size:
            raise ValueError(f"peer_rank must be in local team, got {peer_rank}")
        return self._backend.peer_ptr(peer_rank)

    def get_window_handle(self):
        if self.backend != "nccl":
            raise RuntimeError("NCCL window is only available for backend='nccl'")
        return self._backend.window_handle()

    def get_peer_ptrs_tensor(self):
        """
        Returns CUDA peer pointers for the local LSA/NVL team.
        Tensor shape: [local_world_size], dtype: torch.int64
        """
        if self.backend == "nccl":
            ptrs = [self.get_peer_ptr(i) for i in range(self.local_world_size)]
            return torch.tensor(ptrs, dtype=torch.int64, device='cuda')
        ptrs = []
        for i in range(self.local_world_size):
            if i == self.local_rank:
                ptrs.append(self.get_local_ptr())
            else:
                p = self.get_peer_ptr(i)
                if p == 0:
                    raise RuntimeError(f"Peer pointer for rank {i} is null")
                ptrs.append(p)
        return torch.tensor(ptrs, dtype=torch.int64, device='cuda')


class SymmetricTensor:

    def __init__(self, shape: tuple, dtype: torch.dtype, group: dist.ProcessGroup, backend: str = "auto",
                 local_world_size: int = 0):
        """
        A tensor-like wrapper around SymmetricBuffer.
        
        Args:
            shape (tuple): Shape of the tensor.
            dtype (torch.dtype): Data type of the tensor.
            group (dist.ProcessGroup): The process group to use.
            backend (str, optional): Backend strategy ("auto", "vmm", "cuda_ipc", "nccl").
        """
        self.shape = shape
        self.dtype = dtype

        element_size = torch.tensor([], dtype=dtype).element_size()
        numel = 1
        for dim in shape:
            numel *= dim
        self.numel = numel
        self.size_bytes = numel * element_size

        self.buffer = SymmetricBuffer(self.size_bytes, group, backend, local_world_size)

    def get_local_tensor(self):
        return self.get_peer_tensor(self.buffer.local_rank)

    def get_peer_tensor(self, rank):
        ptr = 0
        if rank == self.buffer.local_rank:
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

    def get_window_handle(self) -> int:
        return self.buffer.get_window_handle()

    def free(self, group=None) -> None:
        """Collective, leak-free release of this tensor's symmetric buffer.

        See free_symmetric_tensors for the rationale. Every rank in the buffer's
        group must call this together. ``group`` defaults to the creation group.
        """
        free_symmetric_tensors([self], group or self.buffer.group)


def free_symmetric_tensors(symm_tensors, group) -> None:
    """Collectively release SymmetricTensors / SymmetricBuffers without leaking.

    The torch_ipc backend shares one producer buffer to every peer in the local
    team via torch's CUDA IPC reduction. Each peer holds a rebuilt tensor that
    maps into the producer's storage, and torch's IPC ref-counter only lets the
    producer storage return to the caching pool once every consumer has released.
    The release is therefore staged so it is correct regardless of rank ordering:
      1. every rank drops its consumer (peer) tensors -> counter -1 per peer;
      2. a barrier, so no peer maps into a buffer that is about to be freed;
      3. every rank drops its producer storage (now counter == 0) to torch's IPC
         limbo, and ipc_collect reclaims it back into the caching pool.

    Because each producer mints one export per consumer (see _TorchIpcSymmMem),
    the counter is charged to N and the N releases bring it cleanly to zero, so
    step 3 reclaims fully -- no per-consumer-count underflow leak when N>1 peers
    import the same buffer.

    Backends other than torch_ipc (cuda_ipc / vmm / nccl) own raw driver memory
    that frees deterministically in their C++ destructor, so for them this is just
    a local immediate free and the barrier / ipc_collect are skipped entirely.

    Collective contract: all ranks must call this together over ``group`` with a
    set whose torch_ipc / non-torch_ipc split matches across ranks (true for
    symmetrically-created EP buffers). Aliased tensors (e.g. combine reusing the
    dispatch buffer) are de-duplicated by underlying buffer identity. Idempotent:
    already-freed buffers are skipped.
    """
    seen = set()
    buffers = []
    for t in symm_tensors:
        if t is None:
            continue
        buf = getattr(t, "buffer", t)  # accept SymmetricTensor or SymmetricBuffer
        if buf is None or id(buf) in seen:
            continue
        seen.add(id(buf))
        buffers.append(buf)

    ipc_backends = [b._backend for b in buffers if isinstance(getattr(b, "_backend", None), _TorchIpcSymmMem)]

    if ipc_backends:
        if group is None:
            raise ValueError("free_symmetric_tensors needs a process group to release torch_ipc "
                             "buffers: a collective barrier is required so every rank drops its "
                             "peer mappings before any producer storage is reclaimed.")
        # Make sure in-flight kernels that touch these buffers have retired
        # before we drop the peer mappings out from under them.
        torch.cuda.synchronize()

    for be in ipc_backends:
        be.release_peers()
    if ipc_backends:
        # Let the consumer-release CUDA events retire so the counter decrements
        # land, then fence cross-process before any producer storage is freed.
        torch.cuda.synchronize()
        dist.barrier(group=group)
    for be in ipc_backends:
        be.free_local()
    if ipc_backends:
        # Counter is back to zero; drain torch's IPC limbo to actually return the
        # producer storage to the caching pool (otherwise it lingers as a leak).
        torch.cuda.ipc_collect()

    # Detach backends: non-torch_ipc memory frees here via the C++ dtor, and every
    # freed buffer becomes inert so a stale reference can't be reused.
    for b in buffers:
        b._backend = None
