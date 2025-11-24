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
import triton.language as tl
import triton_dist
from triton_dist.language.extra.language_extra import tid, __syncthreads
from .task_context import TaskBaseInfo, Scoreboard
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.cuda.language_extra import (st_v4_b32, multimem_ld_reduce_v4)
from triton.language.extra.cuda.utils import num_warps


@triton_dist.jit
def allreduce_one_shot_multimem_intra_node_kernel(pid, num_pid, symm_in_ptr, out_ptr, elems):
    symm_in_ptr = tl.cast(symm_in_ptr, out_ptr.dtype)

    data_mc_ptr = libshmem_device.remote_mc_ptr(libshmem_device.NVSHMEMX_TEAM_NODE, symm_in_ptr)
    VEC_SIZE = 128 // tl.constexpr(symm_in_ptr.dtype.element_ty.primitive_bitwidth)

    thread_idx = tid(axis=0)
    block_dim = num_warps() * 32
    for idx in range(thread_idx + block_dim * pid, elems // VEC_SIZE, num_pid * block_dim):
        val0, val1, val2, val3 = multimem_ld_reduce_v4(data_mc_ptr + idx * VEC_SIZE, acc_dtype=tl.float32)
        st_v4_b32(out_ptr + idx * VEC_SIZE, val0, val1, val2, val3)
    __syncthreads()


@triton_dist.jit
def allreduce_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
    BLOCK_SIZE: tl.constexpr,
):
    input_tensor = task_base_info.get_tensor(0)
    output_tensor = task_base_info.get_tensor(1)

    input_ptr = input_tensor.data_ptr(tl.bfloat16)
    output_ptr = output_tensor.data_ptr(tl.bfloat16)

    n_elements = output_tensor.size(0)
    tile_id = task_base_info.tile_id_or_start
    num_pid = tl.cdiv(n_elements, BLOCK_SIZE)
    allreduce_one_shot_multimem_intra_node_kernel(tile_id, num_pid, input_ptr, output_ptr, n_elements)
    scoreboard.release_tile(task_base_info, task_base_info.tile_id_or_start)
