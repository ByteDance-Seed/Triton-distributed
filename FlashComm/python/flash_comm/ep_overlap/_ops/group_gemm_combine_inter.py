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
    cute_compile_options,
    make_moe_jit_dummies,
    mark_compact_dynamic,
    mark_dynamic,
    resolve_nvcc_opt_level,
)


class CuTeDSLGroupGemmCombineInterOp(CuTeDSLEPOverlapOpBase):
    """Fused inter-node FC2 GEMM + intra/inter-node combine."""

    def __init__(self, *, rank: int, world_size: int, local_world_size: int):
        super().__init__(rank=rank, world_size=world_size)
        self.local_world_size = int(local_world_size)

    def _compile(
        self,
        *,
        dtype: torch.dtype,
        weight_dtype: torch.dtype,
        experts_per_rank: int,
        n_out: int,
        hidden_in: int,
        topk: int,
        max_slot_num_token: int,
        num_sm: int,
        has_weight: bool,
        run_reduce: bool,
    ):
        arch_major, arch_minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        variant_args = (
            dtype,
            weight_dtype,
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            int(num_sm),
            bool(has_weight),
            bool(run_reduce),
            int(arch_major),
            int(arch_minor),
        )

        def factory():
            return self._build(
                dtype=dtype,
                weight_dtype=weight_dtype,
                experts_per_rank=experts_per_rank,
                n_out=n_out,
                hidden_in=hidden_in,
                topk=topk,
                max_slot_num_token=max_slot_num_token,
                num_sm=num_sm,
                has_weight=has_weight,
                run_reduce=run_reduce,
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
        weight_dtype: torch.dtype,
        experts_per_rank: int,
        n_out: int,
        hidden_in: int,
        topk: int,
        max_slot_num_token: int,
        num_sm: int,
        has_weight: bool,
        run_reduce: bool,
        arch_major: int,
        arch_minor: int,
    ):
        import cutlass
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        self._assert_supported_dtype(dtype, "CuTeDSL inter-node group_gemm_combine")
        if arch_major >= 10:
            from ..kernels.cutedsl_group_gemm_combine_inter import (
                MegaMoEGroupGEMMCombineInter as Kernel, )
            mma_tiler = (256, 256)
            cluster = (2, 2)
            use_2cta = True
        elif arch_major == 9:
            from ..kernels.cutedsl_group_gemm_combine_sm90_inter import (
                MegaMoEGroupGEMMCombineInterSM90 as Kernel, )
            mma_tiler = (128, 256)
            cluster = (2, 1)
            use_2cta = False
        else:
            raise RuntimeError("CuTeDSL inter-node group_gemm_combine requires Hopper "
                               f"(sm90) or newer; got sm{arch_major}{arch_minor}")
        if getattr(Kernel, "is_scaffold", False):
            self._ensure_kernel_implemented()

        ab_dtype = cutlass.BFloat16 if dtype == torch.bfloat16 else cutlass.Float16
        hardware_info = utils.HardwareInfo()
        max_active_clusters = min(
            hardware_info.get_max_active_clusters(cluster[0] * cluster[1]),
            max(1,
                int(num_sm) // (cluster[0] * cluster[1])),
        )

        dummy_A, dummy_B, dummy_C, dummy_psm = make_moe_jit_dummies(
            num_experts=experts_per_rank,
            n=n_out,
            hidden_in=hidden_in,
            ab_dtype=cutlass_torch.dtype(ab_dtype),
        )
        dummy_output = torch.empty((1, n_out), dtype=dtype, device="cuda")
        dummy_output_weight = torch.empty((1, topk), dtype=weight_dtype, device="cuda")
        dummy_ptrs = torch.empty((self.local_world_size, ), dtype=torch.int64, device="cuda")
        dummy_num_tokens_per_rank = torch.empty((self.world_size, ), dtype=torch.int32, device="cuda")
        dummy_meta = torch.empty(
            (self.world_size // self.local_world_size, max_slot_num_token, topk),
            dtype=torch.int32,
            device="cuda",
        )
        dummy_barrier_workspace = torch.empty(
            (max(8, 2 * int(experts_per_rank)), ),
            dtype=torch.int32,
            device="cuda",
        )
        opt_level = resolve_nvcc_opt_level()
        kernel = Kernel(
            cutlass.Float32,
            use_2cta,
            mma_tiler,
            cluster,
        )
        return cute.compile(
            kernel,
            mark_dynamic(dummy_A, enable_tvm_ffi=True),
            mark_dynamic(dummy_B, enable_tvm_ffi=True),
            mark_dynamic(dummy_C, assumed_align=16, enable_tvm_ffi=True),
            mark_compact_dynamic(dummy_output, assumed_align=16, enable_tvm_ffi=True),
            mark_dynamic(dummy_output_weight, enable_tvm_ffi=True),
            mark_dynamic(dummy_ptrs, enable_tvm_ffi=True),
            mark_dynamic(dummy_barrier_workspace, enable_tvm_ffi=True),
            mark_dynamic(dummy_ptrs, enable_tvm_ffi=True),
            cutlass.Int64(0),
            cutlass.Int64(0),
            mark_dynamic(dummy_num_tokens_per_rank, enable_tvm_ffi=True),
            mark_dynamic(dummy_meta, enable_tvm_ffi=True),
            mark_dynamic(dummy_meta, enable_tvm_ffi=True),
            mark_dynamic(dummy_meta, enable_tvm_ffi=True),
            cutlass.Int32(0),
            int(experts_per_rank),
            int(n_out),
            int(hidden_in),
            int(topk),
            int(self.rank),
            int(self.world_size),
            int(self.local_world_size),
            int(max_slot_num_token),
            mark_dynamic(dummy_psm, assumed_align=16, enable_tvm_ffi=True),
            int(max_active_clusters),
            int(max_active_clusters),
            int(bool(has_weight)),
            int(bool(run_reduce)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=cute_compile_options(enable_tvm_ffi=True, opt_level=opt_level),
        )

    @staticmethod
    def _ensure_kernel_implemented() -> None:
        from ..kernels.cutedsl_group_gemm_combine_sm90_inter import (
            MegaMoEGroupGEMMCombineInterSM90, )

        if getattr(MegaMoEGroupGEMMCombineInterSM90, "is_scaffold", False):
            raise NotImplementedError("cutedsl_group_gemm_combine_sm90_inter.py is scaffolded; "
                                      "implement MegaMoEGroupGEMMCombineInterSM90 before running "
                                      "inter-node group_gemm_combine")

    def run(
        self,
        *,
        dev_comm_ptr: int,
        A_padded: torch.Tensor,
        B: torch.Tensor,
        recv_expert_counts: torch.Tensor,
        num_tokens_per_rank: torch.Tensor,
        local_topk_indices: torch.Tensor,
        local_topk_send_mask: torch.Tensor,
        local_token_dst_scatter_indices: torch.Tensor,
        recv_topk_scatter_indices: Optional[torch.Tensor],
        fc2_output: torch.Tensor,
        combine_x_ptrs: torch.Tensor,
        barrier_workspace: torch.Tensor,
        barrier_workspace_ptrs: torch.Tensor,
        output: torch.Tensor,
        output_weight: Optional[torch.Tensor],
        rdma_rail_send_buf: torch.Tensor,
        rdma_rail_send_win_handle: int,
        max_slot_num_token: int,
        experts_per_rank: int,
        num_sm: int,
        topk: int,
        weight_dtype: torch.dtype,
        dispatched_weights: Optional[torch.Tensor],
        run_reduce: bool,
    ):
        has_weight = dispatched_weights is not None or output_weight is not None
        if has_weight and (dispatched_weights is None or output_weight is None):
            raise ValueError("dispatched_weights and output_weight must be both set or both None")
        if A_padded.dtype != output.dtype:
            raise TypeError(f"A_padded dtype {A_padded.dtype} must match output dtype {output.dtype}")
        if A_padded.ndim != 2 or output.ndim != 2:
            raise ValueError("A_padded and output must be rank-2 tensors")
        if B.ndim != 3:
            raise ValueError(f"B must be rank-3 (N, K, experts_per_rank); got {tuple(B.shape)}")
        if B.shape[1] != A_padded.shape[1]:
            raise ValueError(f"B K dim {B.shape[1]} must match A_padded hidden {A_padded.shape[1]}")
        if B.shape[2] != int(experts_per_rank):
            raise ValueError(f"B expert dim {B.shape[2]} must match experts_per_rank {experts_per_rank}")
        if B.shape[0] != output.shape[1]:
            raise ValueError(f"B N dim {B.shape[0]} must match output hidden {output.shape[1]}")
        for tensor, name in (
            (A_padded, "A_padded"),
            (B, "B"),
            (fc2_output, "fc2_output"),
            (output, "output"),
            (rdma_rail_send_buf, "rdma_rail_send_buf"),
        ):
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
        if not A_padded.is_contiguous():
            raise ValueError("A_padded must be contiguous")
        if fc2_output.dtype != A_padded.dtype:
            raise TypeError(f"fc2_output dtype {fc2_output.dtype} must match A_padded dtype {A_padded.dtype}")
        if fc2_output.ndim != 2 or fc2_output.shape != (A_padded.shape[0], B.shape[0]):
            raise ValueError("fc2_output must have shape "
                             f"({A_padded.shape[0]}, {B.shape[0]}); got "
                             f"{tuple(fc2_output.shape)}")
        if not fc2_output.is_contiguous():
            raise ValueError("fc2_output must be contiguous")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")
        if combine_x_ptrs.dtype != torch.int64 or combine_x_ptrs.shape != (self.local_world_size, ):
            raise ValueError("combine_x_ptrs must be int64 with shape "
                             f"({self.local_world_size},); got "
                             f"{combine_x_ptrs.dtype} {tuple(combine_x_ptrs.shape)}")
        if barrier_workspace.dtype != torch.int32 or barrier_workspace.ndim != 1:
            raise ValueError("barrier_workspace must be a 1D int32 tensor")
        if barrier_workspace.numel() < 6:
            raise ValueError("barrier_workspace must contain at least 6 int32 slots")
        if not barrier_workspace.is_cuda or not barrier_workspace.is_contiguous():
            raise ValueError("barrier_workspace must be a contiguous CUDA tensor")
        if barrier_workspace_ptrs.dtype != torch.int64 or barrier_workspace_ptrs.shape != (self.local_world_size, ):
            raise ValueError("barrier_workspace_ptrs must be int64 with shape "
                             f"({self.local_world_size},); got "
                             f"{barrier_workspace_ptrs.dtype} {tuple(barrier_workspace_ptrs.shape)}")
        if recv_expert_counts.dtype != torch.int32:
            raise TypeError(f"recv_expert_counts must be int32; got {recv_expert_counts.dtype}")
        if recv_expert_counts.shape != (int(experts_per_rank), ):
            raise ValueError("recv_expert_counts must have shape "
                             f"({experts_per_rank},); got {tuple(recv_expert_counts.shape)}")
        if num_tokens_per_rank.dtype != torch.int32 or num_tokens_per_rank.shape != (self.world_size, ):
            raise ValueError("num_tokens_per_rank must be int32 with shape "
                             f"({self.world_size},); got "
                             f"{num_tokens_per_rank.dtype} {tuple(num_tokens_per_rank.shape)}")
        expected_meta_shape = (self.world_size // self.local_world_size, int(max_slot_num_token), int(topk))
        for tensor, name in (
            (local_topk_indices, "local_topk_indices"),
            (local_topk_send_mask, "local_topk_send_mask"),
            (local_token_dst_scatter_indices, "local_token_dst_scatter_indices"),
        ):
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must be int32; got {tensor.dtype}")
            if not tensor.is_cuda or not tensor.is_contiguous():
                raise ValueError(f"{name} must be a contiguous CUDA tensor")
            if tuple(tensor.shape) != expected_meta_shape:
                raise ValueError(f"{name} must have shape {expected_meta_shape}; "
                                 f"got {tuple(tensor.shape)}")
        if recv_topk_scatter_indices is not None:
            if recv_topk_scatter_indices.dtype != torch.int32:
                raise TypeError("recv_topk_scatter_indices must be int32; "
                                f"got {recv_topk_scatter_indices.dtype}")
            if not recv_topk_scatter_indices.is_cuda or not recv_topk_scatter_indices.is_contiguous():
                raise ValueError("recv_topk_scatter_indices must be a contiguous CUDA tensor")
            if recv_topk_scatter_indices.ndim != 2 or recv_topk_scatter_indices.shape[1] != int(topk):
                raise ValueError("recv_topk_scatter_indices must have shape "
                                 f"(A_padded.shape[0], {topk}); got "
                                 f"{tuple(recv_topk_scatter_indices.shape)}")
            if recv_topk_scatter_indices.shape[0] != A_padded.shape[0]:
                raise ValueError("recv_topk_scatter_indices rows must match A_padded rows: "
                                 f"{recv_topk_scatter_indices.shape[0]} vs {A_padded.shape[0]}")
        if has_weight:
            if dispatched_weights.dtype != weight_dtype:
                raise TypeError(f"dispatched_weights dtype must be {weight_dtype}; got {dispatched_weights.dtype}")
            if dispatched_weights.ndim != 1 or dispatched_weights.shape[0] != A_padded.shape[0]:
                raise ValueError("dispatched_weights must be 1D with length A_padded.shape[0]; "
                                 f"got {tuple(dispatched_weights.shape)}")
            if not dispatched_weights.is_cuda or not dispatched_weights.is_contiguous():
                raise ValueError("dispatched_weights must be a contiguous CUDA tensor")
            if output_weight.dtype != weight_dtype:
                raise TypeError(f"output_weight dtype must be {weight_dtype}; got {output_weight.dtype}")
            if output_weight.ndim != 2 or output_weight.shape[1] != int(topk):
                raise ValueError("output_weight must have shape (num_tokens, topk); "
                                 f"got {tuple(output_weight.shape)}")
            if output_weight.shape[0] != output.shape[0]:
                raise ValueError("output_weight rows must match output rows: "
                                 f"{output_weight.shape[0]} vs {output.shape[0]}")
            if not output_weight.is_cuda or not output_weight.is_contiguous():
                raise ValueError("output_weight must be a contiguous CUDA tensor")

        self._ensure_kernel_implemented()

        compiled = self._compile(
            dtype=A_padded.dtype,
            weight_dtype=weight_dtype,
            experts_per_rank=int(experts_per_rank),
            n_out=int(B.shape[0]),
            hidden_in=int(A_padded.shape[1]),
            topk=int(topk),
            max_slot_num_token=int(max_slot_num_token),
            num_sm=int(num_sm),
            has_weight=has_weight,
            run_reduce=run_reduce,
        )
        launch_output_weight = output_weight
        if not has_weight:
            launch_output_weight = torch.empty((1, int(topk)), dtype=weight_dtype, device=A_padded.device)
        compiled(A_padded, B, fc2_output, output, launch_output_weight,
                 combine_x_ptrs, barrier_workspace, barrier_workspace_ptrs, int(dev_comm_ptr),
                 int(rdma_rail_send_win_handle), num_tokens_per_rank, local_topk_indices, local_topk_send_mask,
                 local_token_dst_scatter_indices, int(output.shape[0]), recv_expert_counts)
        _ = rdma_rail_send_buf
        _ = recv_topk_scatter_indices
        _ = dispatched_weights
        return output, output_weight
