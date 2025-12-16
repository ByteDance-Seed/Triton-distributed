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
import triton
import triton.language as tl
from triton_dist.language.extra.language_extra import tid, __syncthreads, atomic_cas
import triton_dist.language as dl
from .task_context import TaskBaseInfo, Scoreboard


@triton.jit
def barrier_all_intra_node_atomic_cas_block(local_rank, local_world_size, symm_flag_ptr):
    """ NOTE: this function should only be called with atomic support. memory over PCI-e does not support atomic r/w. DON'T use this function on such platforms.
    """

    thread_idx = tid(0)
    if thread_idx < local_world_size:  # thread_idx => local_rank
        remote_ptr = dl.symm_at(symm_flag_ptr + local_rank, thread_idx)
        while atomic_cas(remote_ptr, 0, 1, "sys", "release") != 0:
            pass

    if thread_idx < local_world_size:  # thread_idx => local_rank
        while (atomic_cas(symm_flag_ptr + thread_idx, 1, 0, "sys", "acquire") != 1):
            pass
    __syncthreads()


@triton.jit
def barrier_all_intra_node_task_compute(
    task_base_info: TaskBaseInfo,
    scoreboard: Scoreboard,
):
    symm_flag_tensor = task_base_info.get_tensor(0)
    symm_flag_ptr = symm_flag_tensor.data_ptr(tl.int32)
    extra_params_ptr = task_base_info.get_extra_params_ptr(1)

    local_rank = tl.load(extra_params_ptr + 0).to(tl.int32)
    local_world_size = tl.load(extra_params_ptr + 1).to(tl.int32)
    barrier_all_intra_node_atomic_cas_block(local_rank, local_world_size, symm_flag_ptr)
    scoreboard.release_tile(task_base_info, 0)
