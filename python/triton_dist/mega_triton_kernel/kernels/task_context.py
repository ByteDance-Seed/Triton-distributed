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
from triton.language.extra.cuda.language_extra import tid, st, ld_acquire, __syncthreads, ld
from triton.language.extra.cuda.utils import num_warps


@tl.core._aggregate
class TensorDesc:
    base_ptr: tl.tensor  # pointer to (data_ptr, shape[0], shape[1], ...)

    def __init__(self, base_ptr):
        self.base_ptr = base_ptr

    @triton.jit
    def data_ptr(self, dtype):
        buf_ptr = self.base_ptr.to(tl.pointer_type(tl.uint64))
        data_ptr = tl.load(buf_ptr).to(tl.pointer_type(dtype))
        data_ptr = tl.multiple_of(data_ptr, 16)
        return data_ptr

    @triton.jit
    def size(self, i, multiple=tl.constexpr(1)):
        int_per_data_ptr = 2
        dim = tl.load(self.base_ptr + i + int_per_data_ptr)
        dim = tl.multiple_of(dim, multiple)
        dim = dim.to(tl.int32)
        return dim


@tl.core._aggregate
class TaskBaseInfo:
    io_tensors_ptr: tl.tensor
    layer_id: tl.tensor
    task_id: tl.tensor
    tile_id_or_start: tl.tensor
    depend_entry_start: tl.tensor
    depend_entry_end: tl.tensor
    MAX_NUM_TENSOR_DIMS: tl.constexpr
    INT_PER_TENSOR: tl.constexpr
    is_tile_wise: tl.constexpr

    def __init__(self, io_tensors_ptr, layer_id, task_id, tile_id_or_start, depend_entry_start, depend_entry_end,
                 MAX_NUM_TENSOR_DIMS, is_tile_wise=tl.constexpr(True)):
        self.io_tensors_ptr = io_tensors_ptr
        self.layer_id = layer_id
        self.task_id = task_id
        self.tile_id_or_start = tile_id_or_start
        self.depend_entry_start = depend_entry_start
        self.depend_entry_end = depend_entry_end
        self.MAX_NUM_TENSOR_DIMS = MAX_NUM_TENSOR_DIMS
        self.is_tile_wise = is_tile_wise
        self.INT_PER_TENSOR = self.MAX_NUM_TENSOR_DIMS + 2  # (data_ptr, shape[0], shape[1], ..., shape[MAX_NUM_TENSOR_DIMS - 1])

    @triton.jit
    def get_tensor(self, idx):
        return TensorDesc(self.io_tensors_ptr + idx * self.INT_PER_TENSOR)

    @triton.jit
    def get_extra_params_ptr(self, num_io_tensors):
        return self.io_tensors_ptr + num_io_tensors * self.INT_PER_TENSOR


@tl.core._aggregate
class Scoreboard:
    scoreboard_table: tl.tensor
    task_deps_ptr: tl.tensor
    INT_PER_DEPS: tl.constexpr
    MAX_TASK_ID: tl.constexpr
    MAX_NUM_TASK_PER_OP: tl.constexpr
    TILE_READY_SIGNAL: tl.constexpr
    NUM_THREADS: tl.constexpr
    WARP_SIZE: tl.constexpr

    def __init__(self, task_deps_ptr, INT_PER_DEPS, scoreboard_table, MAX_TASK_ID, MAX_NUM_TASK_PER_OP,
                 TILE_READY_SIGNAL, NUM_THREADS):
        self.task_deps_ptr = task_deps_ptr
        self.INT_PER_DEPS = INT_PER_DEPS
        self.scoreboard_table = scoreboard_table
        self.MAX_TASK_ID = MAX_TASK_ID
        self.MAX_NUM_TASK_PER_OP = MAX_NUM_TASK_PER_OP
        self.TILE_READY_SIGNAL = TILE_READY_SIGNAL
        self.NUM_THREADS = NUM_THREADS
        self.WARP_SIZE = tl.constexpr(32)

    @triton.jit
    def wait_deps(self, task_base_info: TaskBaseInfo):
        # Don't exit early, otherwise it will cause function inlining to fail.

        thread_idx = tid(0)
        entry_start = task_base_info.depend_entry_start
        entry_end = task_base_info.depend_entry_end

        # for t in range(entry_start, entry_end):
        #     l = ld(self.task_deps_ptr + t * self.INT_PER_DEPS + 0)
        #     r = ld(self.task_deps_ptr + t * self.INT_PER_DEPS + 1)
        #     num_signals = r - l
        #     sb_wait_base_ptr = self.scoreboard_table + l
        #     for i in range(thread_idx, num_signals, self.NUM_THREADS):
        #         while ld_acquire(sb_wait_base_ptr + i, "gpu") != self.TILE_READY_SIGNAL:
        #             pass
        lane_id = thread_idx % self.WARP_SIZE
        warp_id = thread_idx // self.WARP_SIZE
        for t in range(entry_start + warp_id, entry_end, num_warps()):
            l = ld(self.task_deps_ptr + t * self.INT_PER_DEPS + 0)
            r = ld(self.task_deps_ptr + t * self.INT_PER_DEPS + 1)
            num_signals = r - l
            sb_wait_base_ptr = self.scoreboard_table + l
            for i in range(lane_id, num_signals, self.WARP_SIZE):
                while ld_acquire(sb_wait_base_ptr + i, "gpu") != self.TILE_READY_SIGNAL:
                    pass

        __syncthreads()

    @triton.jit
    def release_tile(self, task_base_info: TaskBaseInfo, tile_id):
        sb_set_base_ptr = self.task_scoredboard_start(task_base_info)
        __syncthreads()  # ensure that `store` on all threads is finish
        thread_idx = tid(0)
        if thread_idx == 0:
            st(sb_set_base_ptr + tile_id, self.TILE_READY_SIGNAL, "gpu", "release")
        __syncthreads()  # avoid divergence

    @triton.jit
    def task_scoredboard_start(self, task_base_info: TaskBaseInfo):
        sb_layer_offset = task_base_info.layer_id * self.MAX_TASK_ID * self.MAX_NUM_TASK_PER_OP
        sb_set_base_ptr = self.scoreboard_table + sb_layer_offset + task_base_info.task_id * self.MAX_NUM_TASK_PER_OP
        return sb_set_base_ptr
