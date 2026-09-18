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
from contextlib import contextmanager
from typing import Dict, Optional

import torch
import torch.distributed as dist

import flash_comm._C.buffer as _buffer
import flash_comm._C.ep_internode as _ep_inter
from flash_comm.buffer import SymmetricTensor, free_symmetric_tensors
from flash_comm.ep import EPConfig
from flash_comm.ep.ep_context import RDMARailSendLayout

from ._ops._base import GEMM_CLUSTER_TILE_M

# Sentinel the layout C kernel writes into ``recv_token_count_cpu``
# before publishing the real count; kernel re-arms it on every call.
_PENDING_RECV_COUNT_SENTINEL = -1


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


@contextmanager
def _scoped_nccl_gin_env():
    env_map = {
        "NCCL_TOPO_FILE": os.environ.get("FLASH_COMM_NCCL_TOPO_FILE"),
        "NCCL_NETDEVS_POLICY": os.environ.get("FLASH_COMM_NCCL_NETDEVS_POLICY"),
        "NCCL_GIN_NCONNECTIONS": os.environ.get("FLASH_COMM_NCCL_GIN_NCONNECTIONS"),
    }
    old_env = {key: os.environ.get(key) for key in env_map}
    try:
        for key, value in env_map.items():
            if value is not None:
                os.environ[key] = value
        yield
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class EPOverlapContext:
    """CuTeDSL EP-overlap buffer container; construct via :meth:`create`."""

    def __init__(self, config: EPConfig, group: dist.ProcessGroup, *, num_worst_tokens: int = -1,
                 capacity_coeff: float = 1.2, check_num_worst_tokens: bool = False, alloc_alignment: int = 1024,
                 expert_alignment: int = 1):
        if expert_alignment < 1:
            raise ValueError(f"expert_alignment must be >= 1, got {expert_alignment}")
        # GEMM invariant: padded expert M must be a multiple of the
        # M-contig group-GEMM per-cluster tile, else the scheduler would
        # issue a partial tile spanning the next expert.
        if expert_alignment > 1 and expert_alignment % GEMM_CLUSTER_TILE_M != 0:
            raise ValueError(f"expert_alignment={expert_alignment} must be a multiple of "
                             f"GEMM_CLUSTER_TILE_M={GEMM_CLUSTER_TILE_M} (the M-contig "
                             f"group-GEMM per-cluster tile)")
        self.config = config
        self.group = group
        self.num_worst_tokens = int(num_worst_tokens)
        self.capacity_coeff = float(capacity_coeff)
        self.check_num_worst_tokens = bool(check_num_worst_tokens)
        self.alloc_alignment = int(alloc_alignment)
        self.expert_alignment = int(expert_alignment)

        # Symmetric eager buffers (peers / layout kernel reach into them).
        self.nvl_barrier_buf: torch.Tensor = None
        self.nvl_barrier_buf_ptrs: torch.Tensor = None
        self.full_splits_buf: torch.Tensor = None
        self.full_splits_buf_ptrs: torch.Tensor = None
        self.full_splits_win_handle: int = 0
        self.token_src_rank_topk_and_indices_buf: torch.Tensor = None
        self.token_src_rank_topk_and_indices_buf_ptrs: torch.Tensor = None
        # Current capacity of dynamically sized receive-data buffers.
        # Receiver metadata has its own fixed worst-case allocation.
        self.max_recv_tokens: int = 0

        # Local eager buffers. In inter-node mode expert_signal_state is
        # backed by a local-world peer-visible SymmetricTensor so fused
        # dispatch+GEMM kernels can signal across local ranks.
        self.recv_token_count_cpu: torch.Tensor = None
        self.recv_token_count: torch.Tensor = None
        # ``expert_signal_state`` is (2, experts_per_rank) int32:
        # row 0 = producer-ready flags, row 1 = CTA-arrival counters.
        # Single backing tensor so :meth:`reset_expert_signals` issues
        # one zero kernel; the row views alias and stay contiguous.
        self.expert_signal_state_symm_tensor: Optional[SymmetricTensor] = None
        self.expert_signal_state: torch.Tensor = None
        self.expert_signal_state_ptrs: Optional[torch.Tensor] = None
        self.expert_signals: torch.Tensor = None
        self.expert_signal_counters: torch.Tensor = None

        # Internode NCCL GIN resources (None / 0 for intranode).  CuTeDSL
        # kernels dereference ncclDevComm from device code, so the host
        # struct is copied into a CUDA tensor and kept alive here.
        self._owns_nccl_gin: bool = False
        self.nccl_gin_dev_comm_buf: Optional[torch.Tensor] = None
        self._dispatch_signal_range: tuple[int, int] | None = None
        self._combine_signal_range: tuple[int, int] | None = None
        self.rdma_rail_send_buf: Optional[torch.Tensor] = None
        self.rdma_rail_send_win_handle: int = 0
        self.rdma_rail_send_layout: Optional[RDMARailSendLayout] = None

        # Inter-node standalone dispatch writes into local-world visible
        # output/scatter buffers, matching the CUDA GIN+TMA protocol.
        self.dispatch_output_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_output_buf: Optional[torch.Tensor] = None
        self.dispatch_output_ptrs: Optional[torch.Tensor] = None
        self.dispatch_topk_scatter_indices_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_topk_scatter_indices_buf: Optional[torch.Tensor] = None
        self.dispatch_topk_scatter_indices_ptrs: Optional[torch.Tensor] = None
        self.dispatch_group_gemm_output_weight_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_group_gemm_output_weight_buf: Optional[torch.Tensor] = None
        self.dispatch_group_gemm_output_weight_ptrs: Optional[torch.Tensor] = None

        # Lazy CuTeDSL staging buffers.
        self.dispatch_input_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_input_buf: Optional[torch.Tensor] = None
        self.dispatch_input_ptrs: Optional[torch.Tensor] = None

        # Optional symmetric weight staging used by the dispatch
        # ``has_weight`` side channel. Lazy-allocated on first use:
        # peers pull (max_m, topk) FP32 weights and the kernel
        # side-writes per-row scalars into the caller-provided local
        # output tensor.
        self.dispatch_input_weight_symm_tensor: Optional[SymmetricTensor] = None
        self.dispatch_input_weight_buf: Optional[torch.Tensor] = None
        self.dispatch_input_weight_ptrs: Optional[torch.Tensor] = None

        self.combine_output_symm_tensor: Optional[SymmetricTensor] = None
        self.combine_output_buf: Optional[torch.Tensor] = None
        self.combine_output_ptrs: Optional[torch.Tensor] = None

        self.group_gemm_combine_output_symm_tensor: Optional[SymmetricTensor] = None
        self.group_gemm_combine_output_buf: Optional[torch.Tensor] = None
        self.group_gemm_combine_output_ptrs: Optional[torch.Tensor] = None
        self.group_gemm_combine_output_n_out: Optional[int] = None

        # Optional symmetric output-weight staging used by the
        # group_gemm_combine ``has_weight`` side channel. Lazy-allocated
        # on first use: peers push a 4 B FP32 weight per dispatched row
        # back to the source rank's ``(max_m, topk)`` symmetric slot.
        self.group_gemm_combine_output_weight_symm_tensor: Optional[SymmetricTensor] = None
        self.group_gemm_combine_output_weight_buf: Optional[torch.Tensor] = None
        self.group_gemm_combine_output_weight_ptrs: Optional[torch.Tensor] = None

        # Strong refs so SymmetricTensor GPU allocations survive GC.
        self._symm_tensors: Dict[str, SymmetricTensor] = {}
        self._closed = False

    @classmethod
    def create(cls, *, max_m: int, hidden: int, topk: int, num_experts: int, group: dist.ProcessGroup,
               local_world_size: int, capacity_coeff: float = 1.2, num_worst_tokens: int = -1,
               check_num_worst_tokens: bool = False, expert_alignment: int = 1) -> "EPOverlapContext":
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        config = EPConfig(max_m=max_m, hidden=hidden, topk=topk, num_experts=num_experts, rank=rank,
                          world_size=world_size, local_world_size=local_world_size)
        if config.num_experts % config.world_size != 0:
            raise ValueError(f"num_experts ({config.num_experts}) must be divisible by "
                             f"world_size ({config.world_size})")

        ctx = cls(config=config, group=group, num_worst_tokens=num_worst_tokens, capacity_coeff=capacity_coeff,
                  check_num_worst_tokens=check_num_worst_tokens, expert_alignment=expert_alignment)
        ctx._init_eager_buffers()
        return ctx

    def _init_eager_buffers(self) -> None:
        if self.config.nnodes > 1:
            self._init_internode_nccl_gin()
            self._init_internode_eager_buffers()
            return

        cfg = self.config
        group = self.group

        # nvl_barrier must start zeroed group-wide; stale non-zero
        # would deadlock the next barrier kernel.
        nvl_barrier_symm = SymmetricTensor(
            shape=(cfg.local_world_size, ),
            dtype=torch.int32,
            group=group,
            local_world_size=cfg.local_world_size,
        )
        nvl_barrier_symm.get_local_tensor().fill_(0)
        self.nvl_barrier_buf = nvl_barrier_symm.get_local_tensor()
        self.nvl_barrier_buf_ptrs = nvl_barrier_symm.ptrs
        self._symm_tensors["nvl_barrier"] = nvl_barrier_symm

        full_splits_symm = SymmetricTensor(
            shape=(cfg.world_size, cfg.num_experts + 1),
            dtype=cfg.offset_dtype,
            group=group,
            local_world_size=cfg.local_world_size,
        )
        self.full_splits_buf = full_splits_symm.get_local_tensor()
        self.full_splits_buf_ptrs = full_splits_symm.ptrs
        self._symm_tensors["full_splits"] = full_splits_symm

        # Pinned page; layout kernel writes -1 -> real count via two
        # async copies, so the poll loop sees the latest value without
        # a CPU-side reset.
        self.recv_token_count_cpu = torch.empty(
            (cfg.world_size, ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.recv_token_count_cpu.fill_(_PENDING_RECV_COUNT_SENTINEL)
        self.recv_token_count = torch.empty(
            (cfg.world_size, ),
            dtype=torch.int32,
            device="cuda",
        )

        if self.num_worst_tokens > 0:
            self.max_recv_tokens = self.num_worst_tokens
        else:
            self.max_recv_tokens = _round_up(
                int(cfg.max_m * cfg.topk * self.capacity_coeff),
                self.alloc_alignment,
            )
        self._alloc_token_src_meta(self._max_token_src_meta_tokens())

        # Per-expert producer/consumer signals for fused dispatch+gemm.
        # Single (2, experts_per_rank) int32 backing tensor; the two
        # row views alias storage and are zeroed in one kernel by
        # :meth:`reset_expert_signals` before each fused call.
        experts_per_rank = cfg.num_experts // cfg.world_size
        self.expert_signal_state = torch.zeros(
            (2, experts_per_rank),
            dtype=torch.int32,
            device="cuda",
        )
        self.expert_signal_state_ptrs = None
        self.expert_signals = self.expert_signal_state[0]
        self.expert_signal_counters = self.expert_signal_state[1]

    def _alloc_token_src_meta(self, num_alloc_tokens: int) -> None:
        """Allocate the symmetric receiver-side dispatch metadata.

        Symmetric because peers write into this rank's slots from the
        layout C kernel. Sized once to the worst-case receive count.
        """
        token_src_symm = SymmetricTensor(
            shape=(num_alloc_tokens, ),
            dtype=torch.int64,
            group=self.group,
            local_world_size=self.config.local_world_size,
        )
        # -1 sentinel: padded receive-slots (expert_alignment > 1) read
        # as "no source" so CuTeDSL skips them via the src_rank check.
        token_src_symm.get_local_tensor().fill_(-1)
        self.token_src_rank_topk_and_indices_buf = (token_src_symm.get_local_tensor())
        self.token_src_rank_topk_and_indices_buf_ptrs = token_src_symm.ptrs
        self._symm_tensors["token_src_rank_topk_and_indices"] = token_src_symm

    def _max_token_src_meta_tokens(self) -> int:
        """Return a routing-independent upper bound for receiver metadata."""
        cfg = self.config
        max_tokens = cfg.world_size * cfg.max_m * cfg.topk
        if self.expert_alignment > 1:
            experts_per_rank = cfg.num_experts // cfg.world_size
            max_tokens += experts_per_rank * (self.expert_alignment - 1)
        if self.num_worst_tokens > 0:
            max_tokens = max(max_tokens, self.num_worst_tokens)
        return int(max_tokens)

    def _alloc_dispatch_group_gemm_output_weight(self) -> None:
        if self.dispatch_output_buf is None:
            raise RuntimeError("dispatch_output_buf must be allocated before "
                               "dispatch_group_gemm output weights")
        cfg = self.config
        symm = SymmetricTensor(
            shape=(int(self.dispatch_output_buf.shape[0]), ),
            dtype=cfg.weight_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        self.dispatch_group_gemm_output_weight_symm_tensor = symm
        self.dispatch_group_gemm_output_weight_buf = symm.get_local_tensor()
        self.dispatch_group_gemm_output_weight_ptrs = symm.ptrs
        self._symm_tensors["dispatch_group_gemm_output_weight"] = symm

    def ensure_dispatch_input(self) -> None:
        """Lazy alloc of the symmetric dispatch staging buffer.

        First call host-barriers so peers see the new ptrs; subsequent
        calls are no-ops.
        """
        if self.dispatch_input_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m, cfg.hidden),
            dtype=cfg.token_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        self.dispatch_input_symm_tensor = symm
        self.dispatch_input_buf = symm.get_local_tensor()
        self.dispatch_input_ptrs = symm.ptrs
        self._symm_tensors["dispatch_input"] = symm
        dist.barrier(group=self.group)

    def ensure_dispatch_input_weight(self) -> None:
        """Lazy alloc of the symmetric topk-weight staging buffer.

        Same alloc-time barrier discipline as
        :meth:`ensure_dispatch_input`. Buffer holds the full
        ``(max_m, topk)`` FP32 weight matrix so peers can pull a
        4 B scalar per ``(token, topk_idx)`` row.
        """
        if self.dispatch_input_weight_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m, cfg.topk),
            dtype=cfg.weight_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        self.dispatch_input_weight_symm_tensor = symm
        self.dispatch_input_weight_buf = symm.get_local_tensor()
        self.dispatch_input_weight_ptrs = symm.ptrs
        self._symm_tensors["dispatch_input_weight"] = symm
        dist.barrier(group=self.group)

    def ensure_dispatch_group_gemm_output_weight(self) -> None:
        """Lazy alloc of the inter-node FC1 dispatch weight output buffer.

        The inter-node fused dispatch+GEMM kernel writes complete
        ``A_padded`` rows into ``dispatch_output_buf`` from local peers.
        Its optional FC1 dispatch-weight side channel therefore needs the
        same peer-visible ownership: one FP32 scalar per M-contiguous
        dispatched row, addressed through a local-world pointer table.
        """
        if self.dispatch_group_gemm_output_weight_symm_tensor is not None:
            return
        self._alloc_dispatch_group_gemm_output_weight()
        dist.barrier(group=self.group)

    def ensure_combine_output(self) -> None:
        """Lazy alloc of the symmetric combine staging buffer.

        Same alloc-time barrier discipline as :meth:`ensure_dispatch_input`.
        """
        if self.combine_output_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m * cfg.topk, cfg.hidden),
            dtype=cfg.token_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        self.combine_output_symm_tensor = symm
        self.combine_output_buf = symm.get_local_tensor()
        self.combine_output_ptrs = symm.ptrs
        self._symm_tensors["combine_output"] = symm
        dist.barrier(group=self.group)

    def ensure_group_gemm_combine_output_weight(self) -> None:
        """Lazy alloc of the symmetric output-weight buffer.

        Same alloc-time barrier discipline as
        :meth:`ensure_dispatch_input`. Holds the per-rank
        ``(max_m, topk)`` FP32 weight matrix; the fused
        group_gemm_combine kernel push-writes one 4 B FP32 scalar per
        dispatched row to its source rank's ``(src_token_idx,
        src_topk_idx)`` slot (one write per token, gated to the
        leftmost N-tile so the multi-N-tile epilogue does not re-emit
        the same scalar).
        """
        if self.group_gemm_combine_output_weight_symm_tensor is not None:
            return
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m, cfg.topk),
            dtype=cfg.weight_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        self.group_gemm_combine_output_weight_symm_tensor = symm
        self.group_gemm_combine_output_weight_buf = symm.get_local_tensor()
        self.group_gemm_combine_output_weight_ptrs = symm.ptrs
        self._symm_tensors["group_gemm_combine_output_weight"] = symm
        dist.barrier(group=self.group)

    def ensure_group_gemm_combine_output(self, n_out: int) -> None:
        """Lazy alloc of the fused group-GEMM + combine staging buffer.

        Re-allocates on ``n_out`` change (each alloc triggers a
        cross-rank sync via SymmetricTensor).
        """
        if (self.group_gemm_combine_output_symm_tensor is not None and self.group_gemm_combine_output_n_out == n_out):
            return
        old_symm = self._symm_tensors.pop("group_gemm_combine_output", None)
        if old_symm is not None:
            # Changing N changes the peer-visible allocation. Retire all kernels,
            # drop every non-owning view, and release the old allocation before
            # constructing the replacement so N sweeps do not accumulate memory
            # or temporarily hold both large buffers.
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            self.group_gemm_combine_output_symm_tensor = None
            self.group_gemm_combine_output_buf = None
            self.group_gemm_combine_output_ptrs = None
            self.group_gemm_combine_output_n_out = None
            free_symmetric_tensors([old_symm], self.group)
        cfg = self.config
        symm = SymmetricTensor(
            shape=(cfg.max_m * cfg.topk, n_out),
            dtype=cfg.token_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        self.group_gemm_combine_output_symm_tensor = symm
        self.group_gemm_combine_output_buf = symm.get_local_tensor()
        self.group_gemm_combine_output_ptrs = symm.ptrs
        self.group_gemm_combine_output_n_out = n_out
        self._symm_tensors["group_gemm_combine_output"] = symm
        dist.barrier(group=self.group)

    def _init_internode_nccl_gin(self) -> None:
        cfg = self.config
        if _buffer.nccl_gin_is_initialized():
            gin_rank = _buffer.nccl_gin_rank()
            gin_nranks = _buffer.nccl_gin_nranks()
            gin_local_world_size = _buffer.nccl_gin_local_world_size()
            if (gin_rank != cfg.rank or gin_nranks != cfg.world_size or gin_local_world_size != cfg.local_world_size):
                raise ValueError("Existing NCCL GIN communicator does not match this EP group: "
                                 f"rank/nranks/local_world_size="
                                 f"{gin_rank}/{gin_nranks}/{gin_local_world_size}, expected "
                                 f"{cfg.rank}/{cfg.world_size}/{cfg.local_world_size}")
            lsa_size = _buffer.nccl_gin_lsa_size()
            lsa_rank = _buffer.nccl_gin_lsa_rank()
            if lsa_size < cfg.local_world_size or (lsa_rank % cfg.local_world_size) != cfg.local_rank:
                raise ValueError("Existing NCCL GIN LSA team does not cover local_world_size/local_rank")
            self._owns_nccl_gin = False
            self._set_ep_signal_ranges()
            self._refresh_nccl_gin_dev_comm_buf()
            return

        ep_num_qps = int(os.environ.get("FLASH_COMM_EP_NUM_QPS", "1") or "1")
        if ep_num_qps <= 0:
            raise ValueError(f"FLASH_COMM_EP_NUM_QPS must be positive, got {ep_num_qps}")
        gin_contexts = ep_num_qps

        root_global_rank = dist.get_global_rank(self.group, 0)
        uid = [_buffer.nccl_gin_get_unique_id() if cfg.rank == 0 else None]
        dist.broadcast_object_list(uid, src=root_global_rank, group=self.group)
        root_ep_num_qps = [ep_num_qps if cfg.rank == 0 else None]
        dist.broadcast_object_list(root_ep_num_qps, src=root_global_rank, group=self.group)
        if ep_num_qps != root_ep_num_qps[0]:
            raise ValueError("FLASH_COMM_EP_NUM_QPS must be identical across the EP group: "
                             f"rank {cfg.rank} has {ep_num_qps}, root has {root_ep_num_qps[0]}")

        gin_signals = max(
            16,
            _round_up(int(_ep_inter.ep_required_gin_signal_count(cfg.nnodes)), 16),
        )
        gin_rail_barriers = max(
            16,
            torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        )
        gin_queue_depth = int(os.environ.get("FLASH_COMM_NCCL_GIN_QUEUE_DEPTH", "4096"))
        if gin_queue_depth < 0:
            raise ValueError("FLASH_COMM_NCCL_GIN_QUEUE_DEPTH must be >= 0")
        with _scoped_nccl_gin_env():
            _buffer.nccl_gin_init(
                uid[0],
                cfg.rank,
                cfg.world_size,
                cfg.local_world_size,
                gin_contexts,
                gin_signals,
                gin_rail_barriers,
                gin_queue_depth,
                int(_buffer.NCCL_GIN_CONNECTION_FULL),
                ep_num_qps=ep_num_qps,
            )
        self._owns_nccl_gin = True
        self._set_ep_signal_ranges()
        self._refresh_nccl_gin_dev_comm_buf()

    def _set_ep_signal_ranges(self) -> None:
        nnodes = self.config.nnodes
        self._dispatch_signal_range = (0, int(_ep_inter.ep_dispatch_signal_count(nnodes)))
        self._combine_signal_range = (int(_ep_inter.ep_combine_signal_base(nnodes)),
                                      int(_ep_inter.ep_required_gin_signal_count(nnodes)))

    def _refresh_nccl_gin_dev_comm_buf(self) -> None:
        """Keep a CUDA-resident ncclDevComm copy for CuTeDSL device API.

        Passing a pageable host ncclDevComm pointer to CuTeDSL GIN kernels
        can fail on H800.  The working path is to copy the struct bytes to
        CUDA memory and pass that device pointer to ``nccl_cute.DevComm``.
        """
        host_dev_comm = _buffer.nccl_gin_dev_comm_bytes()
        self.nccl_gin_dev_comm_buf = host_dev_comm.to(
            device=torch.device("cuda", torch.cuda.current_device()),
            non_blocking=False,
        ).contiguous()

    def nccl_gin_dev_comm_ptr(self) -> int:
        if self.nccl_gin_dev_comm_buf is None:
            self._refresh_nccl_gin_dev_comm_buf()
        return int(self.nccl_gin_dev_comm_buf.data_ptr())

    def _init_internode_eager_buffers(self) -> None:
        cfg = self.config
        group = self.group

        nvl_barrier_symm = SymmetricTensor(
            shape=(cfg.local_world_size, ),
            dtype=torch.int32,
            group=group,
            local_world_size=cfg.local_world_size,
        )
        nvl_barrier_symm.get_local_tensor().zero_()
        self.nvl_barrier_buf = nvl_barrier_symm.get_local_tensor()
        self.nvl_barrier_buf_ptrs = nvl_barrier_symm.ptrs
        self._symm_tensors["nvl_barrier"] = nvl_barrier_symm

        full_splits_symm = SymmetricTensor(
            shape=(cfg.world_size, cfg.num_experts + 2),
            dtype=cfg.offset_dtype,
            group=group,
            backend="nccl",
            local_world_size=cfg.local_world_size,
        )
        self.full_splits_buf = full_splits_symm.get_local_tensor()
        self.full_splits_buf_ptrs = full_splits_symm.ptrs
        self.full_splits_win_handle = full_splits_symm.get_window_handle()
        self._symm_tensors["internode_full_splits"] = full_splits_symm

        self.rdma_rail_send_layout = RDMARailSendLayout(
            max_tokens=cfg.max_m,
            hidden=cfg.hidden,
            topk=cfg.topk,
            nnodes=cfg.nnodes,
            token_dtype=cfg.token_dtype,
            offset_dtype=cfg.offset_dtype,
            weight_dtype=cfg.weight_dtype,
        )
        rdma_rail_send_bytes = self.rdma_rail_send_layout.buffer_bytes_for_slots(self.rdma_rail_send_layout.num_slots)
        rdma_rail_send_symm = SymmetricTensor(
            shape=(rdma_rail_send_bytes, ),
            dtype=torch.uint8,
            group=group,
            backend="nccl",
            local_world_size=cfg.local_world_size,
        )
        self.rdma_rail_send_buf = rdma_rail_send_symm.get_local_tensor()
        self.rdma_rail_send_win_handle = rdma_rail_send_symm.get_window_handle()
        self._symm_tensors["internode_rdma_rail_send"] = rdma_rail_send_symm

        self.recv_token_count_cpu = torch.empty(
            (cfg.world_size, ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.recv_token_count_cpu.fill_(_PENDING_RECV_COUNT_SENTINEL)
        self.recv_token_count = torch.empty(
            (cfg.world_size, ),
            dtype=torch.int32,
            device="cuda",
        )

        if self.num_worst_tokens > 0:
            dispatch_recv_tokens = self.num_worst_tokens
        else:
            dispatch_recv_tokens = _round_up(
                int(cfg.max_m * cfg.topk * self.capacity_coeff),
                self.alloc_alignment,
            )
        self._alloc_token_src_meta(self._max_token_src_meta_tokens())
        self._alloc_dispatch_output_buffers(dispatch_recv_tokens)

        experts_per_rank = cfg.num_experts // cfg.world_size
        expert_signal_symm = SymmetricTensor(
            shape=(2, experts_per_rank),
            dtype=torch.int32,
            group=group,
            local_world_size=cfg.local_world_size,
        )
        expert_signal_symm.get_local_tensor().zero_()
        self.expert_signal_state_symm_tensor = expert_signal_symm
        self.expert_signal_state = expert_signal_symm.get_local_tensor()
        self.expert_signal_state_ptrs = expert_signal_symm.ptrs
        self._symm_tensors["expert_signal_state"] = expert_signal_symm
        self.expert_signals = self.expert_signal_state[0]
        self.expert_signal_counters = self.expert_signal_state[1]

    def _alloc_dispatch_output_buffers(self, dispatch_recv_tokens: int) -> None:
        cfg = self.config
        dispatch_output_symm = SymmetricTensor(
            shape=(dispatch_recv_tokens, cfg.hidden),
            dtype=cfg.token_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        dispatch_scatter_symm = SymmetricTensor(
            shape=(dispatch_recv_tokens, cfg.topk),
            dtype=cfg.offset_dtype,
            group=self.group,
            local_world_size=cfg.local_world_size,
        )
        dispatch_scatter_symm.get_local_tensor().fill_(-1)
        self.dispatch_output_symm_tensor = dispatch_output_symm
        self.dispatch_output_buf = dispatch_output_symm.get_local_tensor()
        self.dispatch_output_ptrs = dispatch_output_symm.ptrs
        self.dispatch_topk_scatter_indices_symm_tensor = dispatch_scatter_symm
        self.dispatch_topk_scatter_indices_buf = dispatch_scatter_symm.get_local_tensor()
        self.dispatch_topk_scatter_indices_ptrs = dispatch_scatter_symm.ptrs
        self._symm_tensors["dispatch_output"] = dispatch_output_symm
        self._symm_tensors["dispatch_topk_scatter_indices"] = dispatch_scatter_symm
        self.max_recv_tokens = int(dispatch_recv_tokens)

    def reallocate_dispatch_buffers(self, num_alloc_tokens: int) -> bool:
        """Collectively grow inter-node receive-data symmetric buffers.

        The caller must first retire device work and enter this method on every
        EP rank. Old allocations are released before replacements are created to
        avoid a transient peak. Receiver metadata is allocated once at its
        routing-independent worst case and is intentionally not replaced here.
        """
        if self._closed:
            raise RuntimeError("cannot reallocate a finalized EPOverlapContext")
        if self.config.nnodes <= 1:
            return False
        num_alloc_tokens = int(num_alloc_tokens)
        if num_alloc_tokens <= self.max_recv_tokens:
            return False

        had_dispatch_weight = self.dispatch_group_gemm_output_weight_symm_tensor is not None
        keys = ["dispatch_output", "dispatch_topk_scatter_indices"]
        if had_dispatch_weight:
            keys.append("dispatch_group_gemm_output_weight")
        old_tensors = [self._symm_tensors.pop(key, None) for key in keys]

        # Drop non-owning tensor views before releasing their backing storage.
        self.dispatch_output_symm_tensor = None
        self.dispatch_output_buf = None
        self.dispatch_output_ptrs = None
        self.dispatch_topk_scatter_indices_symm_tensor = None
        self.dispatch_topk_scatter_indices_buf = None
        self.dispatch_topk_scatter_indices_ptrs = None
        self.dispatch_group_gemm_output_weight_symm_tensor = None
        self.dispatch_group_gemm_output_weight_buf = None
        self.dispatch_group_gemm_output_weight_ptrs = None

        free_symmetric_tensors(old_tensors, self.group)

        self._alloc_dispatch_output_buffers(num_alloc_tokens)
        if had_dispatch_weight:
            self._alloc_dispatch_group_gemm_output_weight()
        return True

    def rdma_rail_send_slot_views(self, num_token: int, hidden: int, topk: int, max_slot_num_token: int = 0):
        """Views into this rank's NCCL symmetric RDMA rail-send slot."""
        if self.rdma_rail_send_buf is None:
            raise RuntimeError("rdma_rail_send_buf is only allocated for internode EP")
        if self.rdma_rail_send_layout is None:
            raise RuntimeError("rdma_rail_send_layout is only available for internode EP")
        if max_slot_num_token <= 0:
            max_slot_num_token = num_token
        layout = self.rdma_rail_send_layout
        if layout.max_tokens != max_slot_num_token or layout.hidden != hidden or layout.topk != topk:
            raise ValueError("rdma_rail_send_layout must match the allocation-time layout")
        return layout.views(self.rdma_rail_send_buf, self.config.node_id, num_token)

    def release_internode_nccl_resources(self) -> None:
        """Deregister NCCL windows before destroying the EP NCCL communicator."""
        if self.config.nnodes <= 1:
            return
        # Free NCCL-backed windows first so deregister cannot race communicator teardown.
        nccl_tensors = []
        for name in ("internode_rdma_rail_send", "internode_full_splits"):
            tensor = self._symm_tensors.pop(name, None)
            if tensor is not None:
                nccl_tensors.append(tensor)
        if nccl_tensors:
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            free_symmetric_tensors(nccl_tensors, self.group)
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
        self.rdma_rail_send_buf = None
        self.rdma_rail_send_win_handle = 0
        self.rdma_rail_send_layout = None
        self.full_splits_buf = None
        self.full_splits_buf_ptrs = None
        self.full_splits_win_handle = 0

    def destroy_internode_nccl_gin(self) -> None:
        """Destroy the context-owned GIN communicator after windows are gone."""
        if self.config.nnodes <= 1:
            return
        if self._owns_nccl_gin and _buffer.nccl_gin_is_initialized():
            _buffer.nccl_gin_destroy()
        self._owns_nccl_gin = False
        self.nccl_gin_dev_comm_buf = None
        self._dispatch_signal_range = None
        self._combine_signal_range = None

    def free_buffers(self) -> None:
        """Collectively release all remaining symmetric and local workspace.

        For internode contexts, :meth:`release_internode_nccl_resources` and
        :meth:`destroy_internode_nccl_gin` must run first. Idempotent when all
        ranks call teardown in the same order.
        """
        if self._closed:
            return
        nccl_keys = {"internode_rdma_rail_send", "internode_full_splits"}
        live_nccl_keys = nccl_keys.intersection(self._symm_tensors)
        if live_nccl_keys:
            raise RuntimeError("release_internode_nccl_resources must run before free_buffers; "
                               f"live NCCL allocations: {sorted(live_nccl_keys)}")

        tensors = list(self._symm_tensors.values())
        self._symm_tensors.clear()
        free_symmetric_tensors(tensors, self.group)

        self.nvl_barrier_buf = None
        self.nvl_barrier_buf_ptrs = None
        self.full_splits_buf = None
        self.full_splits_buf_ptrs = None
        self.full_splits_win_handle = 0
        self.token_src_rank_topk_and_indices_buf = None
        self.token_src_rank_topk_and_indices_buf_ptrs = None
        self.max_recv_tokens = 0
        self.dispatch_output_symm_tensor = None
        self.dispatch_output_buf = None
        self.dispatch_output_ptrs = None
        self.dispatch_topk_scatter_indices_symm_tensor = None
        self.dispatch_topk_scatter_indices_buf = None
        self.dispatch_topk_scatter_indices_ptrs = None
        self.dispatch_group_gemm_output_weight_symm_tensor = None
        self.dispatch_group_gemm_output_weight_buf = None
        self.dispatch_group_gemm_output_weight_ptrs = None
        self.dispatch_input_symm_tensor = None
        self.dispatch_input_buf = None
        self.dispatch_input_ptrs = None
        self.dispatch_input_weight_symm_tensor = None
        self.dispatch_input_weight_buf = None
        self.dispatch_input_weight_ptrs = None
        self.combine_output_symm_tensor = None
        self.combine_output_buf = None
        self.combine_output_ptrs = None
        self.group_gemm_combine_output_symm_tensor = None
        self.group_gemm_combine_output_buf = None
        self.group_gemm_combine_output_ptrs = None
        self.group_gemm_combine_output_n_out = None
        self.group_gemm_combine_output_weight_symm_tensor = None
        self.group_gemm_combine_output_weight_buf = None
        self.group_gemm_combine_output_weight_ptrs = None
        self.expert_signal_state_symm_tensor = None
        self.expert_signal_state = None
        self.expert_signal_state_ptrs = None
        self.expert_signals = None
        self.expert_signal_counters = None
        self.recv_token_count_cpu = None
        self.recv_token_count = None
        self.rdma_rail_send_buf = None
        self.rdma_rail_send_win_handle = 0
        self.rdma_rail_send_layout = None
        self.nccl_gin_dev_comm_buf = None
        self._closed = True

    def reset_expert_signals(self) -> None:
        """Single-kernel zero of both signal rows (ready flags + CTA counters)."""
        self.expert_signal_state.zero_()
