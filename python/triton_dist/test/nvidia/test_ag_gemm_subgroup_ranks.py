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
import unittest
from unittest.mock import MagicMock, patch
import torch

from triton_dist.kernels.nvidia.allgather import get_auto_all_gather_method, AllGatherMethod
from triton_dist.kernels.allreduce import get_auto_all_reduce_method, OverlappingAllReduceMethod
from triton_dist.kernels.nvidia.allgather_gemm import create_ag_gemm_context
from triton_dist.kernels.nvidia.allgather_group_gemm import (
    MoEAllGatherGroupGEMMTensorParallelContext,
)


class TestAGSubgroupRanks(unittest.TestCase):

    def test_auto_allgather_subgroup_local_ranks(self):
        """When num_ranks < num_local_ranks (e.g., TP=4 on an 8-GPU node),
        num_local_ranks should clamp to num_ranks and select intra-node method.
        """
        get_auto_all_gather_method.cache_clear()
        with patch("triton_dist.kernels.nvidia.allgather.has_fullmesh_nvlink", return_value=True):
            method = get_auto_all_gather_method(4, 8)
            self.assertEqual(method, AllGatherMethod.All2All_IntraNode)

        get_auto_all_gather_method.cache_clear()
        with patch("triton_dist.kernels.nvidia.allgather.has_fullmesh_nvlink", return_value=False), \
             patch("triton_dist.kernels.nvidia.allgather.get_triton_dist_world", return_value=MagicMock()), \
             patch("triton_dist.kernels.nvidia.allgather.get_group_numa_world_size", return_value=4):
            method = get_auto_all_gather_method(4, 8)
            self.assertEqual(method, AllGatherMethod.Ring1D_IntraNode)

    def test_auto_allreduce_subgroup_local_ranks(self):
        """get_auto_all_reduce_method should clamp num_local_ranks to num_ranks."""
        get_auto_all_reduce_method.cache_clear()
        with patch("triton_dist.kernels.allreduce.is_nvshmem_multimem_supported", return_value=False):
            method = get_auto_all_reduce_method(4, 8)
            self.assertEqual(method, OverlappingAllReduceMethod.Consumer_Load)

    def test_ag_gemm_context_subgroup_clamping(self):
        """AllGatherGEMMTensorParallelContext should clamp num_local_ranks
        to num_ranks when TP subgroups are smaller than default 8 GPUs per node.
        """
        with patch("triton_dist.kernels.nvidia.allgather_gemm.nvshmem_create_tensors", return_value=[torch.zeros(1)] * 4), \
             patch("triton_dist.kernels.nvidia.allgather_gemm.nvshmem_create_tensor", return_value=torch.zeros(1)), \
             patch("triton_dist.kernels.nvidia.allgather_gemm.nvshmem_barrier_all_on_stream"), \
             patch("torch.cuda.synchronize"), \
             patch("torch.cuda.get_device_properties", return_value=MagicMock(multi_processor_count=108)):

            ctx = create_ag_gemm_context(
                max_M=128,
                N=256,
                K=512,
                dtype=torch.float16,
                rank=2,
                num_ranks=4,
                num_local_ranks=8,
            )
            self.assertEqual(ctx.num_local_ranks, 4)
            self.assertFalse(ctx.is_multinode)
            self.assertEqual(ctx.n_nodes, 1)
            self.assertEqual(ctx.node_rank, 0)
            self.assertEqual(ctx.local_rank, 2)

    def test_moe_ag_group_gemm_context_subgroup_clamping(self):
        """MoEAllGatherGroupGEMMTensorParallelContext should clamp num_local_ranks
        to num_ranks when num_ranks < num_local_ranks.
        """
        with patch("triton_dist.kernels.nvidia.allgather_group_gemm.nvshmem_create_tensors", return_value=[torch.zeros(1)] * 4), \
             patch("triton_dist.kernels.nvidia.allgather_group_gemm.nvshmem_barrier_all_on_stream"), \
             patch("torch.cuda.synchronize"), \
             patch("torch.cuda.get_device_properties", return_value=MagicMock(multi_processor_count=108)):

            ctx = MoEAllGatherGroupGEMMTensorParallelContext(
                rank=1,
                num_ranks=4,
                num_local_ranks=8,
                max_ntokens=128,
                num_experts=8,
                topk=2,
                N_per_rank=64,
                K=512,
                dtype=torch.float16,
            )
            self.assertEqual(ctx.num_local_ranks, 4)
            self.assertFalse(ctx.is_multinode)
            self.assertEqual(ctx.n_nodes, 1)
            self.assertEqual(ctx.node_rank, 0)
            self.assertEqual(ctx.local_rank, 1)

    def test_multinode_preservation(self):
        """When num_ranks > num_local_ranks, multinode configuration must be preserved."""
        with patch("triton_dist.kernels.nvidia.allgather_gemm.nvshmem_create_tensors", return_value=[torch.zeros(1)] * 8), \
             patch("triton_dist.kernels.nvidia.allgather_gemm.nvshmem_create_tensor", return_value=torch.zeros(1)), \
             patch("triton_dist.kernels.nvidia.allgather_gemm.nvshmem_barrier_all_on_stream"), \
             patch("torch.cuda.synchronize"), \
             patch("torch.cuda.get_device_properties", return_value=MagicMock(multi_processor_count=108)):

            ctx = create_ag_gemm_context(
                max_M=128,
                N=256,
                K=512,
                dtype=torch.float16,
                rank=10,
                num_ranks=16,
                num_local_ranks=8,
            )
            self.assertEqual(ctx.num_local_ranks, 8)
            self.assertTrue(ctx.is_multinode)
            self.assertEqual(ctx.n_nodes, 2)
            self.assertEqual(ctx.node_rank, 1)
            self.assertEqual(ctx.local_rank, 2)


if __name__ == "__main__":
    unittest.main()
