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
from .allgather_gemm import ag_gemm_intra_node, create_ag_gemm_intra_node_context, ag_gemm_inter_node, \
create_ag_gemm_inter_node_context, gemm, inter_node_allgather, local_copy_and_barrier_all
from .utils import get_numa_node, has_fullmesh_mxlink, has_fullmesh_mxlink_ngpus, get_numa_world_size

__all__ = [
    "ag_gemm_intra_node",
    "create_ag_gemm_intra_node_context",
    "ag_gemm_inter_node",
    "create_ag_gemm_inter_node_context",
    "gemm",
    "get_numa_node",
    "has_fullmesh_mxlink",
    "has_fullmesh_mxlink_ngpus",
    "get_numa_world_size",
    "inter_node_allgather",
    "local_copy_and_barrier_all",
]
