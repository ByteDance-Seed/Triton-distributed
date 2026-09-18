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

from typing import Optional

import torch

from ._base import (
    CuTeDSLEPOverlapOpBase,
    GEMM_CLUSTER_TILE_M,
    cute_compile_options,
    make_moe_jit_dummies,
    mark_dynamic,
    resolve_nvcc_opt_level,
)


class CuTeDSLDispatchGroupGemmInterOp(CuTeDSLEPOverlapOpBase):
    """Fused inter-node dispatch + intra-node dispatch + FC1 GEMM op."""

    def __init__(self, *, rank: int, world_size: int, expert_alignment: int):
        super().__init__(rank=rank, world_size=world_size)
        self.expert_alignment = int(expert_alignment)

    def _compile(
        self,
        *,
        dtype: torch.dtype,
        experts_per_rank: int,
        n_out: int,
        hidden: int,
        dispatch_num_stages: int,
        num_sm: int,
        topk: int,
        local_world_size: int,
        max_slot_num_token: int,
        max_recv_tokens: int,
        weight_dtype: torch.dtype,
        has_weight: bool,
    ):
        arch_major, arch_minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        variant_args = (
            dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden),
            int(dispatch_num_stages),
            int(num_sm),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.expert_alignment),
            int(local_world_size),
            int(max_slot_num_token),
            int(max_recv_tokens),
            weight_dtype,
            bool(has_weight),
            int(arch_major),
            int(arch_minor),
        )

        def factory():
            return self._build(
                dtype=dtype,
                experts_per_rank=experts_per_rank,
                n_out=n_out,
                hidden=hidden,
                dispatch_num_stages=dispatch_num_stages,
                num_sm=num_sm,
                topk=topk,
                local_world_size=local_world_size,
                max_slot_num_token=max_slot_num_token,
                max_recv_tokens=max_recv_tokens,
                weight_dtype=weight_dtype,
                has_weight=has_weight,
                arch_major=arch_major,
                arch_minor=arch_minor,
            )

        opt_level = resolve_nvcc_opt_level()
        return self._get_cached_kernel(
            variant_args=variant_args,
            builder=factory,
            compile_options=("--enable-tvm-ffi", f"--opt-level {opt_level}"),
        )

    def _build(
        self,
        *,
        dtype: torch.dtype,
        experts_per_rank: int,
        n_out: int,
        hidden: int,
        dispatch_num_stages: int,
        num_sm: int,
        topk: int,
        local_world_size: int,
        max_slot_num_token: int,
        max_recv_tokens: int,
        weight_dtype: torch.dtype,
        has_weight: bool,
        arch_major: int,
        arch_minor: int,
    ):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        self._assert_supported_dtype(
            dtype,
            "CuTeDSL inter-node dispatch+groupgemm",
        )
        if arch_major >= 10:
            from ..kernels.cutedsl_dispatch_group_gemm_inter import (
                CuTeDSLDispatchGroupGemmInterKernelSM100 as Kernel, )
            mma_tiler = (256, 256)
            cluster = (2, 2)
            use_2cta = True
        elif arch_major == 9:
            from ..kernels.cutedsl_dispatch_group_gemm_sm90_inter import (
                CuTeDSLDispatchGroupGemmInterKernelSM90 as Kernel, )
            mma_tiler = (128, 256)
            cluster = (2, 1)
            use_2cta = False
        else:
            raise RuntimeError("CuTeDSL inter-node dispatch+groupgemm requires Hopper "
                               f"(sm90) or newer, got sm{arch_major}{arch_minor}")
        if getattr(Kernel, "is_scaffold", False):
            raise NotImplementedError("The selected inter-node dispatch_group_gemm kernel is "
                                      "scaffolded; implement it before "
                                      "running inter-node dispatch_group_gemm")

        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        hardware_info = utils.HardwareInfo()
        max_active_clusters = min(
            hardware_info.get_max_active_clusters(cluster[0] * cluster[1]),
            max(1,
                int(num_sm) // (cluster[0] * cluster[1])),
        )
        expert_alignment = max(
            int(self.expert_alignment),
            GEMM_CLUSTER_TILE_M,
        )

        nnodes = self.world_size // int(local_world_size)
        dummy_num_tokens_per_rank = torch.empty(
            (self.world_size, ),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_recv_x_ptrs = torch.empty(
            (local_world_size, ),
            dtype=torch.int64,
            device="cuda",
        )
        dummy_recv_weight_ptrs = torch.empty_like(dummy_recv_x_ptrs)
        dummy_recv_scatter_ptrs = torch.empty_like(dummy_recv_x_ptrs)
        dummy_node_topk_indices = torch.empty(
            (nnodes, max_slot_num_token, topk),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_node_mask = torch.empty_like(dummy_node_topk_indices)
        dummy_node_scatter = torch.empty_like(dummy_node_topk_indices)
        dummy_full_splits = torch.empty(
            (self.world_size, self.world_size * experts_per_rank + 2),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_A, dummy_B, dummy_C, dummy_psm = make_moe_jit_dummies(
            num_experts=experts_per_rank,
            n=n_out,
            hidden_in=hidden,
            ab_dtype=cutlass_torch.dtype(ab_dtype),
        )
        dummy_signals = torch.empty(
            (experts_per_rank, ),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_output_weight = torch.empty(
            (1, ),
            dtype=weight_dtype,
            device="cuda",
        )
        dummy_signal_state_ptrs = torch.empty(
            (local_world_size, ),
            dtype=torch.int64,
            device="cuda",
        )

        opt_level = resolve_nvcc_opt_level()
        kernel = Kernel(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster,
            dispatch_num_stages=dispatch_num_stages,
        )
        return cute.compile(
            kernel,
            cutlass.Int64(0),
            cutlass.Int64(0),
            mark_dynamic(dummy_num_tokens_per_rank, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_x_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_weight_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_recv_scatter_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_node_topk_indices, enable_tvm_ffi=True),
            mark_dynamic(dummy_node_mask, enable_tvm_ffi=True),
            mark_dynamic(dummy_node_scatter, enable_tvm_ffi=True),
            mark_dynamic(dummy_full_splits, enable_tvm_ffi=True),
            mark_dynamic(dummy_A, enable_tvm_ffi=True),
            mark_dynamic(dummy_B, enable_tvm_ffi=True),
            mark_dynamic(dummy_C, enable_tvm_ffi=True),
            int(experts_per_rank),
            int(n_out),
            int(hidden),
            mark_dynamic(dummy_psm, assumed_align=16, enable_tvm_ffi=True),
            int(max_active_clusters),
            int(max_active_clusters),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            mark_dynamic(dummy_signals, enable_tvm_ffi=True),
            mark_dynamic(dummy_signal_state_ptrs, enable_tvm_ffi=True),
            int(self.rank),
            int(self.world_size),
            int(local_world_size),
            int(max_slot_num_token),
            int(max_recv_tokens),
            int(expert_alignment),
            mark_dynamic(dummy_output_weight, enable_tvm_ffi=True),
            int(topk),
            int(bool(has_weight)),
            options=cute_compile_options(
                enable_tvm_ffi=True,
                opt_level=opt_level,
            ),
        )

    def run(
        self,
        *,
        dev_comm_ptr: int,
        A_padded: torch.Tensor,
        B: torch.Tensor,
        recv_expert_counts: torch.Tensor,
        output: torch.Tensor,
        output_weight: Optional[torch.Tensor],
        num_tokens_per_rank: torch.Tensor,
        recv_x_ptrs: torch.Tensor,
        recv_weight_ptrs: Optional[torch.Tensor],
        recv_topk_scatter_indices_ptrs: torch.Tensor,
        node_topk_indices: torch.Tensor,
        node_topk_send_mask: torch.Tensor,
        node_token_dst_scatter_indices: torch.Tensor,
        full_splits: torch.Tensor,
        expert_signals: torch.Tensor,
        expert_signal_counters: torch.Tensor,
        expert_signal_state_ptrs: torch.Tensor,
        rdma_rail_send_buf: torch.Tensor,
        rdma_rail_send_win_handle: int,
        max_slot_num_token: int,
        max_recv_tokens: int,
        experts_per_rank: int,
        local_world_size: int,
        dispatch_num_stages: int,
        num_sm: int,
        topk: int,
        weight_dtype: torch.dtype,
        has_weight: bool,
    ):
        _ = rdma_rail_send_buf
        _ = expert_signal_counters
        experts_per_rank = int(experts_per_rank)
        hidden = int(A_padded.shape[1])
        n_out = int(B.shape[0])
        has_weight = bool(has_weight)

        if int(recv_expert_counts.shape[0]) != experts_per_rank:
            raise ValueError("recv_expert_counts.shape[0] "
                             f"({recv_expert_counts.shape[0]}) != "
                             f"experts_per_rank ({experts_per_rank})")
        if B.shape != (n_out, hidden, experts_per_rank):
            raise ValueError(f"B must be (N, K={hidden}, L={experts_per_rank}); "
                             f"got {tuple(B.shape)}")
        if output.shape != (A_padded.shape[0], n_out):
            raise ValueError(f"output must be (M={A_padded.shape[0]}, N={n_out}); "
                             f"got {tuple(output.shape)}")
        if recv_x_ptrs.dtype != torch.int64 or recv_x_ptrs.shape != (int(local_world_size), ):
            raise ValueError("recv_x_ptrs must be int64 with shape "
                             f"({local_world_size},); got {recv_x_ptrs.dtype} "
                             f"{tuple(recv_x_ptrs.shape)}")
        if recv_topk_scatter_indices_ptrs.dtype != torch.int64:
            raise TypeError("recv_topk_scatter_indices_ptrs must be int64")
        if full_splits.dtype != torch.int32 or full_splits.shape != (
                self.world_size,
                self.world_size * experts_per_rank + 2,
        ):
            raise ValueError("full_splits must be int32 with shape "
                             f"({self.world_size}, "
                             f"{self.world_size * experts_per_rank + 2}); "
                             f"got {full_splits.dtype} {tuple(full_splits.shape)}")
        if expert_signal_state_ptrs.dtype != torch.int64:
            raise TypeError("expert_signal_state_ptrs must be int64")
        if has_weight:
            if recv_weight_ptrs is None or output_weight is None:
                raise ValueError("recv_weight_ptrs and output_weight are "
                                 "required when has_weight=True")
            if weight_dtype != torch.float32:
                raise TypeError("dispatch_group_gemm_inter with has_weight=True currently requires "
                                f"float32 weights; got {weight_dtype}")
            if output_weight.dtype != weight_dtype:
                raise TypeError(f"output_weight dtype must be {weight_dtype}; "
                                f"got {output_weight.dtype}")
            if (output_weight.ndim != 1 or output_weight.shape[0] != A_padded.shape[0]):
                raise ValueError("output_weight must be 1D with length "
                                 f"{A_padded.shape[0]}; got "
                                 f"{tuple(output_weight.shape)}")
            launch_recv_weight_ptrs = recv_weight_ptrs
            launch_output_weight = output_weight
        else:
            launch_recv_weight_ptrs = torch.empty(
                (int(local_world_size), ),
                dtype=torch.int64,
                device=A_padded.device,
            )
            launch_output_weight = torch.empty(
                (A_padded.shape[0], ),
                dtype=weight_dtype,
                device=A_padded.device,
            )

        compiled = self._compile(
            dtype=A_padded.dtype,
            experts_per_rank=experts_per_rank,
            n_out=n_out,
            hidden=hidden,
            dispatch_num_stages=int(dispatch_num_stages),
            num_sm=int(num_sm),
            topk=int(topk),
            local_world_size=int(local_world_size),
            max_slot_num_token=int(max_slot_num_token),
            max_recv_tokens=int(max_recv_tokens),
            weight_dtype=weight_dtype,
            has_weight=has_weight,
        )
        compiled(int(dev_comm_ptr), int(rdma_rail_send_win_handle), num_tokens_per_rank, recv_x_ptrs,
                 launch_recv_weight_ptrs, recv_topk_scatter_indices_ptrs, node_topk_indices, node_topk_send_mask,
                 node_token_dst_scatter_indices, full_splits, A_padded, B, output, recv_expert_counts, expert_signals,
                 expert_signal_state_ptrs, launch_output_weight)
        return output
