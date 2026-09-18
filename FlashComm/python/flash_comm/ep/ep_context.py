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

import dataclasses

import torch
import torch.distributed as dist

import flash_comm._C.ep_internode as _ep_inter
from flash_comm.buffer import SymmetricTensor, free_symmetric_tensors


@dataclasses.dataclass(frozen=True)
class RDMARailSendLayout:
    """Packed RDMA rail-send buffer layout for one rail rank.

    C++ owns the physical slot layout.  Python only consumes the exported
    offsets/stride descriptor to build typed tensor views.

    Field strides use max_tokens to keep offsets stable across dispatch rounds:
      x                         : [max_tokens, hidden] token_dtype
      topk_indices              : [max_tokens, topk] offset_dtype
      topk_send_mask            : [max_tokens, topk] int32
      token_dst_scatter_indices : [max_tokens, topk] offset_dtype
      topk_weights              : [max_tokens, topk] weight_dtype
    """

    max_tokens: int
    hidden: int
    topk: int
    nnodes: int
    token_dtype: torch.dtype = torch.bfloat16
    offset_dtype: torch.dtype = torch.int32
    weight_dtype: torch.dtype = torch.float32
    token_region_bytes: int = dataclasses.field(init=False)
    meta_region_bytes: int = dataclasses.field(init=False)
    topk_indices_offset: int = dataclasses.field(init=False)
    topk_send_mask_offset: int = dataclasses.field(init=False)
    token_dst_scatter_offset: int = dataclasses.field(init=False)
    topk_weights_offset: int = dataclasses.field(init=False)
    slot_payload_bytes: int = dataclasses.field(init=False)
    slot_stride_bytes: int = dataclasses.field(init=False)
    source_slot_base: int = dataclasses.field(init=False)
    outgoing_slot_base: int = dataclasses.field(init=False)
    incoming_slot_base: int = dataclasses.field(init=False)
    num_slots: int = dataclasses.field(init=False)
    buffer_bytes: int = dataclasses.field(init=False)

    def __post_init__(self):
        if self.token_dtype != torch.bfloat16:
            raise ValueError("RDMA rail-send layout currently requires bfloat16 token dtype")
        if self.offset_dtype != torch.int32:
            raise ValueError("RDMA rail-send layout currently requires int32 offset dtype")
        if self.weight_dtype != torch.float32:
            raise ValueError("RDMA rail-send layout currently requires float32 weight dtype")
        desc = _ep_inter.rdma_rail_send_layout_desc(self.max_tokens, self.hidden, self.topk, self.nnodes)
        # Explicit field contract with the C++ exporter: a silently dropped or
        # renamed field on either side must fail loudly here, not produce a
        # half-initialized layout.
        echo_keys = {
            "max_slot_num_token": self.max_tokens, "hidden_size": self.hidden, "topk": self.topk, "nnodes": self.nnodes
        }
        layout_keys = {f.name for f in dataclasses.fields(self) if not f.init}
        exported_keys = set(desc.keys()) - set(echo_keys)
        if exported_keys != layout_keys:
            raise ValueError("rdma_rail_send_layout_desc field mismatch between C++ and Python: "
                             f"missing={sorted(layout_keys - exported_keys)}, "
                             f"unexpected={sorted(exported_keys - layout_keys)}")
        for key, expected in echo_keys.items():
            if int(desc[key]) != int(expected):
                raise ValueError(f"rdma_rail_send_layout_desc echoed {key}={desc[key]}, expected {expected}")
        for key in layout_keys:
            object.__setattr__(self, key, int(desc[key]))

    @staticmethod
    def _element_size(dtype: torch.dtype) -> int:
        return torch.empty((), dtype=dtype).element_size()

    def buffer_bytes_for_slots(self, num_slots: int) -> int:
        if num_slots == self.num_slots:
            return self.buffer_bytes
        return self.slot_stride_bytes * num_slots

    def node_slot(self, rdma_rail_send_buf: torch.Tensor, node_id: int) -> torch.Tensor:
        off = node_id * self.slot_stride_bytes
        return rdma_rail_send_buf[off:off + self.slot_stride_bytes]

    def views(self, rdma_rail_send_buf: torch.Tensor, node_id: int, num_tokens: int):
        if num_tokens > self.max_tokens:
            raise ValueError(f"num_tokens must be <= max_tokens, got {num_tokens} > {self.max_tokens}")
        slot = self.node_slot(rdma_rail_send_buf, node_id)
        meta_bytes = num_tokens * self.topk * self._element_size(self.offset_dtype)
        token_bytes = num_tokens * self.hidden * self._element_size(self.token_dtype)

        x = slot[:token_bytes].view(self.token_dtype).reshape(num_tokens, self.hidden)
        meta_offset = self.topk_indices_offset
        topk_indices = slot[meta_offset:meta_offset + meta_bytes].view(self.offset_dtype).reshape(num_tokens, self.topk)
        meta_offset += meta_bytes
        topk_send_mask = slot[meta_offset:meta_offset + meta_bytes].view(torch.int32).reshape(num_tokens, self.topk)
        meta_offset += meta_bytes
        token_dst_scatter = slot[meta_offset:meta_offset + meta_bytes].view(self.offset_dtype).reshape(
            num_tokens, self.topk)
        weight_bytes = num_tokens * self.topk * self._element_size(self.weight_dtype)
        topk_weights = slot[self.topk_weights_offset:self.topk_weights_offset + weight_bytes].view(
            self.weight_dtype).reshape(num_tokens, self.topk)
        return {
            "x": x,
            "topk_indices": topk_indices,
            "topk_send_mask": topk_send_mask,
            "token_dst_scatter": token_dst_scatter,
            "topk_weights": topk_weights,
        }


@dataclasses.dataclass
class EPConfig:
    max_m: int
    hidden: int
    topk: int
    num_experts: int

    rank: int
    world_size: int  # ep group size
    local_world_size: int  # shared domain size
    local_rank: int = None
    node_id: int = None
    nnodes: int = None

    token_dtype: torch.dtype = torch.bfloat16  # bfloat16
    weight_dtype: torch.dtype = torch.float32  # float32
    offset_dtype: torch.dtype = torch.int32  # int32

    def __post_init__(self):
        self.local_rank = self.rank % self.local_world_size
        self.node_id = self.rank // self.local_world_size
        self.nnodes = self.world_size // self.local_world_size


@dataclasses.dataclass
class EPContext:
    config: EPConfig
    group: dist.ProcessGroup  # ep group
    # barrier in nvlink domain
    nvl_barrier_buf: torch.Tensor  # init with 0
    nvl_barrier_buf_ptrs: torch.Tensor

    # *_ptrs only contains pointers within same shared memory domain.
    full_splits_buf: torch.Tensor  # (world_size, num_tot_experts + 1)
    full_splits_buf_ptrs: torch.Tensor  # NVL peer ptrs for intranode full_splits/local gather

    dispatch_output_buf: torch.Tensor  # (dispatch_recv_tokens, hidden)
    dispatch_output_buf_ptrs: torch.Tensor  # (local_world_size)
    dispatch_topk_weights_buf: torch.Tensor  # (dispatch_recv_tokens, topk), original topk weights for each token
    dispatch_topk_weights_buf_ptrs: torch.Tensor  # (local_world_size)
    dispatch_topk_scatter_indices_buf: torch.Tensor  # (dispatch_recv_tokens, topk), original topk indices for each token, init with -1
    dispatch_topk_scatter_indices_buf_ptrs: torch.Tensor  # (local_world_size)

    # dispatch output buf can be reused as combine input buf
    combine_input_buf: torch.Tensor  # (dispatch_recv_tokens, hidden)
    combine_input_buf_ptrs: torch.Tensor  # (local_world_size)
    combine_topk_weights_buf: torch.Tensor  # (dispatch_recv_tokens, topk), original topk weights for each token
    combine_topk_weights_buf_ptrs: torch.Tensor  # (local_world_size)

    # internode NCCL GIN resources (None / 0 for single-node)
    full_splits_win_handle: int = 0
    rdma_rail_send_buf: torch.Tensor | None = None
    rdma_rail_send_win_handle: int = 0
    rdma_rail_send_layout: RDMARailSendLayout | None = None

    def reallocate_buffers(self, num_alloc_tokens: int, combine_input_reuse: bool = True):
        if num_alloc_tokens <= self.dispatch_output_buf.shape[0]:
            return

        # Coordinated, leak-free release of the old buffers before reallocating.
        # Required for torch_ipc (whose producer storage would otherwise linger in
        # torch's CUDA IPC limbo); a deterministic local free for other backends.
        # Collective over self.group -- reallocate_buffers is entered by all ranks
        # together (its trigger is derived from an all-ranks token count).
        old_tensors = [
            self._symm_tensors.pop(k, None)
            for k in ('dispatch_output', 'dispatch_topk_weights', 'dispatch_topk_scatter_indices', 'combine_input',
                      'combine_weight')
        ]
        # Drop the non-owning local views so nothing aliases the storage we free.
        self.dispatch_output_buf = None
        self.dispatch_topk_weights_buf = None
        self.dispatch_topk_scatter_indices_buf = None
        self.combine_input_buf = None
        self.combine_topk_weights_buf = None
        free_symmetric_tensors(old_tensors, self.group)

        dispatch_output_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.hidden),
                                                      dtype=self.config.token_dtype, group=self.group,
                                                      local_world_size=self.config.local_world_size)
        dispatch_topk_weights_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.topk),
                                                            dtype=self.config.weight_dtype, group=self.group,
                                                            local_world_size=self.config.local_world_size)
        dispatch_topk_scatter_indices_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.topk),
                                                                    dtype=self.config.offset_dtype, group=self.group,
                                                                    local_world_size=self.config.local_world_size)

        self.dispatch_output_buf = dispatch_output_symm_tensor.get_local_tensor()
        self.dispatch_output_buf_ptrs = dispatch_output_symm_tensor.ptrs
        self.dispatch_topk_weights_buf = dispatch_topk_weights_symm_tensor.get_local_tensor()
        self.dispatch_topk_weights_buf_ptrs = dispatch_topk_weights_symm_tensor.ptrs
        self.dispatch_topk_scatter_indices_buf = dispatch_topk_scatter_indices_symm_tensor.get_local_tensor()
        self.dispatch_topk_scatter_indices_buf_ptrs = dispatch_topk_scatter_indices_symm_tensor.ptrs
        self.dispatch_topk_scatter_indices_buf.fill_(-1)  # init with -1, indicating the invalid index

        if combine_input_reuse:
            combine_input_symm_tensor = dispatch_output_symm_tensor
            combine_weight_symm_tensor = dispatch_topk_weights_symm_tensor
        else:
            combine_input_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.hidden),
                                                        dtype=self.config.token_dtype, group=self.group,
                                                        local_world_size=self.config.local_world_size)
            combine_weight_symm_tensor = SymmetricTensor(shape=(num_alloc_tokens, self.config.topk),
                                                         dtype=self.config.weight_dtype, group=self.group,
                                                         local_world_size=self.config.local_world_size)

        self.combine_input_buf = combine_input_symm_tensor.get_local_tensor()
        self.combine_input_buf_ptrs = combine_input_symm_tensor.ptrs
        self.combine_topk_weights_buf = combine_weight_symm_tensor.get_local_tensor()
        self.combine_topk_weights_buf_ptrs = combine_weight_symm_tensor.ptrs

        # update references to SymmetricTensor objects
        self._symm_tensors.update({
            'dispatch_output': dispatch_output_symm_tensor,
            'dispatch_topk_weights': dispatch_topk_weights_symm_tensor,
            'dispatch_topk_scatter_indices': dispatch_topk_scatter_indices_symm_tensor,
            'combine_input': combine_input_symm_tensor,
            'combine_weight': combine_weight_symm_tensor,
        })

    def free_buffers(self):
        """Collectively release ALL symmetric buffers without leaking.

        MUST be called by every rank (e.g. through EPKernels.finalize) before the
        EPContext is dropped when the torch_ipc backend is in use; otherwise the
        producer buffers stay parked in torch's CUDA IPC limbo and accumulate
        across EPKernels re-creation. Other backends free deterministically, so
        this is just a cheap local free for them. Idempotent.
        """
        if not getattr(self, "_symm_tensors", None):
            return
        tensors = list(self._symm_tensors.values())
        self._symm_tensors.clear()
        free_symmetric_tensors(tensors, self.group)
        # Drop non-owning views into the freed storage.
        self.nvl_barrier_buf = None
        self.full_splits_buf = None
        self.dispatch_output_buf = None
        self.dispatch_topk_weights_buf = None
        self.dispatch_topk_scatter_indices_buf = None
        self.combine_input_buf = None
        self.combine_topk_weights_buf = None
        self.rdma_rail_send_buf = None

    def release_internode_nccl_resources(self):
        """Deregister NCCL windows before destroying the EP NCCL communicator."""
        if not hasattr(self, "_symm_tensors"):
            return
        tensors = []
        for name in ("internode_rdma_rail_send", "internode_full_splits"):
            tensor = self._symm_tensors.pop(name, None)
            if tensor is not None:
                tensors.append(tensor)
        if tensors:
            # GIN kernels and NCCL window deregistration both touch driver state.
            # Fence all ranks before and after deregistering so communicator
            # teardown cannot race a peer that is still using the windows.
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
            free_symmetric_tensors(tensors, self.group)
            torch.cuda.synchronize()
            dist.barrier(group=self.group)
        self.rdma_rail_send_buf = None
        self.rdma_rail_send_win_handle = 0
        self.full_splits_buf = None
        self.full_splits_win_handle = 0

    # need to perform ep group barrier after initialization
    @staticmethod
    def create(max_m: int, hidden: int, topk: int, num_experts: int, group: dist.ProcessGroup, local_world_size: int,
               capacity_coeff: float = 1.2, num_worst_tokens: int = -1,
               combine_input_reuse: bool = True) -> "EPContext":
        """Create EP buffers. ``group`` is the EP group; symmetric ptrs expose the local LSA team."""
        rank = dist.get_rank(group=group)
        world_size = dist.get_world_size(group=group)
        config = EPConfig(max_m=max_m, hidden=hidden, topk=topk, num_experts=num_experts, rank=rank,
                          world_size=world_size, local_world_size=local_world_size)

        if config.nnodes > 1:
            return EPContext._create_internode(
                config=config,
                group=group,
                max_m=max_m,
                hidden=hidden,
                topk=topk,
                capacity_coeff=capacity_coeff,
                num_worst_tokens=num_worst_tokens,
                combine_input_reuse=combine_input_reuse,
            )

        nvl_barrier_symm_tensor = SymmetricTensor(shape=(config.local_world_size, ), dtype=torch.int32, group=group,
                                                  local_world_size=config.local_world_size)
        nvl_barrier_symm_tensor.get_local_tensor().fill_(0)
        full_splits_symm_tensor = SymmetricTensor(shape=(config.world_size, config.num_experts + 1),
                                                  dtype=config.offset_dtype, group=group,
                                                  local_world_size=config.local_world_size)
        dispatch_recv_tokens = int((max_m * topk * capacity_coeff + 1023) / 1024 * 1024)
        if num_worst_tokens > 0:
            dispatch_recv_tokens = num_worst_tokens
        dispatch_output_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, hidden), dtype=config.token_dtype,
                                                      group=group, local_world_size=config.local_world_size)
        dispatch_topk_weights_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk),
                                                            dtype=config.weight_dtype, group=group,
                                                            local_world_size=config.local_world_size)
        dispatch_topk_scatter_indices_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk),
                                                                    dtype=config.offset_dtype, group=group,
                                                                    local_world_size=config.local_world_size)
        dispatch_topk_scatter_indices_symm_tensor.get_local_tensor().fill_(
            -1)  # init with -1, indicating the invalid index

        # reuse dispatch output buf as combine input buf
        if combine_input_reuse:
            combine_input_symm_tensor = dispatch_output_symm_tensor
            combine_weight_symm_tensor = dispatch_topk_weights_symm_tensor
        else:
            combine_input_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, hidden), dtype=config.token_dtype,
                                                        group=group, local_world_size=config.local_world_size)
            combine_weight_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk), dtype=config.weight_dtype,
                                                         group=group, local_world_size=config.local_world_size)
        combine_topk_weights_buf = combine_weight_symm_tensor.get_local_tensor()
        combine_topk_weights_buf_ptrs = combine_weight_symm_tensor.ptrs

        ctx = EPContext(config=config, group=group, nvl_barrier_buf=nvl_barrier_symm_tensor.get_local_tensor(),
                        nvl_barrier_buf_ptrs=nvl_barrier_symm_tensor.ptrs,
                        full_splits_buf=full_splits_symm_tensor.get_local_tensor(),
                        full_splits_buf_ptrs=full_splits_symm_tensor.ptrs,
                        dispatch_output_buf=dispatch_output_symm_tensor.get_local_tensor(),
                        dispatch_output_buf_ptrs=dispatch_output_symm_tensor.ptrs,
                        dispatch_topk_weights_buf=dispatch_topk_weights_symm_tensor.get_local_tensor(),
                        dispatch_topk_weights_buf_ptrs=dispatch_topk_weights_symm_tensor.ptrs,
                        dispatch_topk_scatter_indices_buf=dispatch_topk_scatter_indices_symm_tensor.get_local_tensor(),
                        dispatch_topk_scatter_indices_buf_ptrs=dispatch_topk_scatter_indices_symm_tensor.ptrs,
                        combine_input_buf=combine_input_symm_tensor.get_local_tensor(),
                        combine_input_buf_ptrs=combine_input_symm_tensor.ptrs,
                        combine_topk_weights_buf=combine_topk_weights_buf,
                        combine_topk_weights_buf_ptrs=combine_topk_weights_buf_ptrs)

        # keep references to SymmetricTensor objects to prevent garbage collection
        # the underlying memory would be freed if these objects are collected
        ctx._symm_tensors = {
            'nvl_barrier': nvl_barrier_symm_tensor,
            'full_splits': full_splits_symm_tensor,
            'dispatch_output': dispatch_output_symm_tensor,
            'dispatch_topk_weights': dispatch_topk_weights_symm_tensor,
            'dispatch_topk_scatter_indices': dispatch_topk_scatter_indices_symm_tensor,
            'combine_input': combine_input_symm_tensor,
            'combine_weight': combine_weight_symm_tensor,
        }
        return ctx

    @staticmethod
    def _create_internode(
        *,
        config: EPConfig,
        group: dist.ProcessGroup,
        max_m: int,
        hidden: int,
        topk: int,
        capacity_coeff: float,
        num_worst_tokens: int,
        combine_input_reuse: bool,
    ) -> "EPContext":
        nvl_barrier_symm_tensor = SymmetricTensor(shape=(config.local_world_size, ), dtype=torch.int32, group=group,
                                                  local_world_size=config.local_world_size)
        nvl_barrier_symm_tensor.get_local_tensor().zero_()

        # Internode-only layout exchange. This is a SymmetricTensor with NCCL backend
        # so GIN can put each rank's row into the same logical full_splits table.
        full_splits_row_ints = config.num_experts + 2
        full_splits_symm_tensor = SymmetricTensor(
            shape=(config.world_size, full_splits_row_ints),
            dtype=config.offset_dtype,
            group=group,
            backend="nccl",
            local_world_size=config.local_world_size,
        )
        full_splits_buf = full_splits_symm_tensor.get_local_tensor()
        full_splits_win_handle = full_splits_symm_tensor.get_window_handle()
        dispatch_recv_tokens = int((max_m * topk * capacity_coeff + 1023) / 1024 * 1024)
        if num_worst_tokens > 0:
            dispatch_recv_tokens = num_worst_tokens

        rdma_rail_send_layout = RDMARailSendLayout(
            max_tokens=max_m,
            hidden=hidden,
            topk=topk,
            nnodes=config.nnodes,
            token_dtype=config.token_dtype,
            offset_dtype=config.offset_dtype,
            weight_dtype=config.weight_dtype,
        )
        # One rank belongs to exactly one rail (same local_rank across nodes).
        # The first nnodes slots are dispatch/source slots. Combine uses two
        # explicit regions after that:
        #   [nnodes, 2 * nnodes): outgoing scratch, indexed by owner node.
        #   [2 * nnodes, 3 * nnodes): incoming partials, indexed by contributor node.
        # This keeps send/compute overlap while making each state owner clear.
        rdma_rail_send_bytes = rdma_rail_send_layout.buffer_bytes_for_slots(rdma_rail_send_layout.num_slots)
        rdma_rail_send_symm_tensor = SymmetricTensor(shape=(rdma_rail_send_bytes, ), dtype=torch.uint8, group=group,
                                                     backend="nccl", local_world_size=config.local_world_size)
        rdma_rail_send_buf = rdma_rail_send_symm_tensor.get_local_tensor()
        rdma_rail_send_win_handle = rdma_rail_send_symm_tensor.get_window_handle()

        # Reuse the original intranode communication buffers inside the local NVLink domain.
        # Internode support only adds NCCL full_splits and RDMA rail-send resources above.
        dispatch_output_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, hidden), dtype=config.token_dtype,
                                                      group=group, local_world_size=config.local_world_size)
        dispatch_topk_weights_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk),
                                                            dtype=config.weight_dtype, group=group,
                                                            local_world_size=config.local_world_size)
        dispatch_topk_scatter_indices_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk),
                                                                    dtype=config.offset_dtype, group=group,
                                                                    local_world_size=config.local_world_size)
        dispatch_topk_scatter_indices_symm_tensor.get_local_tensor().fill_(-1)

        if combine_input_reuse:
            combine_input_symm_tensor = dispatch_output_symm_tensor
            combine_weight_symm_tensor = dispatch_topk_weights_symm_tensor
        else:
            combine_input_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, hidden), dtype=config.token_dtype,
                                                        group=group, local_world_size=config.local_world_size)
            combine_weight_symm_tensor = SymmetricTensor(shape=(dispatch_recv_tokens, topk), dtype=config.weight_dtype,
                                                         group=group, local_world_size=config.local_world_size)

        combine_topk_weights_buf = combine_weight_symm_tensor.get_local_tensor()
        combine_topk_weights_buf_ptrs = combine_weight_symm_tensor.ptrs

        ctx = EPContext(
            config=config,
            group=group,
            nvl_barrier_buf=nvl_barrier_symm_tensor.get_local_tensor(),
            nvl_barrier_buf_ptrs=nvl_barrier_symm_tensor.ptrs,
            full_splits_buf=full_splits_buf,
            full_splits_buf_ptrs=full_splits_symm_tensor.ptrs,
            full_splits_win_handle=full_splits_win_handle,
            rdma_rail_send_buf=rdma_rail_send_buf,
            rdma_rail_send_win_handle=rdma_rail_send_win_handle,
            rdma_rail_send_layout=rdma_rail_send_layout,
            dispatch_output_buf=dispatch_output_symm_tensor.get_local_tensor(),
            dispatch_output_buf_ptrs=dispatch_output_symm_tensor.ptrs,
            dispatch_topk_weights_buf=dispatch_topk_weights_symm_tensor.get_local_tensor(),
            dispatch_topk_weights_buf_ptrs=dispatch_topk_weights_symm_tensor.ptrs,
            dispatch_topk_scatter_indices_buf=dispatch_topk_scatter_indices_symm_tensor.get_local_tensor(),
            dispatch_topk_scatter_indices_buf_ptrs=dispatch_topk_scatter_indices_symm_tensor.ptrs,
            combine_input_buf=combine_input_symm_tensor.get_local_tensor(),
            combine_input_buf_ptrs=combine_input_symm_tensor.ptrs,
            combine_topk_weights_buf=combine_topk_weights_buf,
            combine_topk_weights_buf_ptrs=combine_topk_weights_buf_ptrs,
        )
        ctx._symm_tensors = {
            "nvl_barrier": nvl_barrier_symm_tensor,
            "internode_full_splits": full_splits_symm_tensor,
            "internode_rdma_rail_send": rdma_rail_send_symm_tensor,
            "dispatch_output": dispatch_output_symm_tensor,
            "dispatch_topk_weights": dispatch_topk_weights_symm_tensor,
            "dispatch_topk_scatter_indices": dispatch_topk_scatter_indices_symm_tensor,
            "combine_input": combine_input_symm_tensor,
            "combine_weight": combine_weight_symm_tensor,
        }
        return ctx

    def rdma_rail_send_slot_views(self, num_token: int, hidden: int, topk: int, max_slot_num_token: int = 0):
        """Views into this rank's NCCL symmetric RDMA rail-send slot."""
        if self.rdma_rail_send_buf is None:
            raise RuntimeError("rdma_rail_send_buf is only allocated for internode EP")
        if self.rdma_rail_send_layout is None:
            raise RuntimeError("rdma_rail_send_layout is only available for internode EP")
        if max_slot_num_token <= 0:
            max_slot_num_token = num_token
        layout = self.rdma_rail_send_layout
        if (layout.max_tokens != max_slot_num_token or layout.hidden != hidden or layout.topk != topk):
            raise ValueError("rdma_rail_send_layout must match the allocation-time layout")
        return layout.views(self.rdma_rail_send_buf, self.config.node_id, num_token)
