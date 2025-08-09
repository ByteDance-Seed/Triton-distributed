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
from triton.language.extra.cuda.language_extra import (
    st,
    pack_b32_v2,
)

from triton.language.core import to_tensor
from triton.language import core

NUM_BITS_ID = 20  # global_id = block_id * num_blocks + group_id
NUM_BITS_TASK_TYPE = 11
NUM_BITS_EVENT = 1  # only start/end
NUM_BITS_PAYLOAD = 32


@core.extern
def globaltimer_lo(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %globaltimer_lo;",
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def smid(_semantic=None):
    return tl.inline_asm_elementwise(
        asm="mov.u32 $0, %smid;",
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@core.extern
def membar(scope: core.constexpr = core.constexpr("cta"), _semantic=None):
    return tl.inline_asm_elementwise(
        asm=f"""
        membar.{scope.value};
        mov.u32 $0, 0;
        """,
        constraints=("=r"),
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
        _semantic=_semantic,
    )


@tl.core._aggregate
class Profiler:
    buffer: tl.tensor  # pointer to profiler buffer
    stride: tl.tensor
    is_leader: tl.tensor
    group_id: tl.tensor
    num_groups: tl.constexpr

    ENABLE_PROFILING: tl.constexpr
    NUM_BITS_ID: tl.constexpr
    NUM_BITS_TASK_TYPE: tl.constexpr
    NUM_BITS_EVENT: tl.constexpr
    NUM_BITS_PAYLOAD: tl.constexpr

    @triton.jit
    def create(profiler_buffer, group_id, num_groups=1, is_leader=True, num_blocks=None, ENABLE_PROFILING=True):
        pid_size_x = tl.num_programs(axis=0)
        pid_size_y = tl.num_programs(axis=1)
        pid_size_z = tl.num_programs(axis=2)
        if num_blocks is None:
            num_blocks = pid_size_x * pid_size_y * pid_size_z
        pid = tl.program_id(0) + tl.program_id(1) * pid_size_x + tl.program_id(2) * pid_size_x * pid_size_y

        NUM_SLOTS_GLOBAL_META: tl.constexpr = 1  # (num_blocks, num_groups) # noqa: F841
        NUM_SLOTS_BLOCK_META: tl.constexpr = 1  # (block_id, sm_id) # noqa: F841
        NUM_SLOTS_GROUP_META: tl.constexpr = 0  # None # noqa: F841

        # init global/block meta
        profiler_buffer = profiler_buffer.to(tl.pointer_type(tl.uint64))
        if ENABLE_PROFILING:
            global_meta = pack_b32_v2(num_blocks, num_groups)
            st(profiler_buffer, global_meta)

            block_meta = pack_b32_v2(pid, smid())
            st(profiler_buffer + 1 + pid, block_meta)

            profiler_buffer = profiler_buffer + 1 + num_blocks
        stride = num_blocks * num_groups
        return Profiler(profiler_buffer + pid * num_groups + group_id, stride, to_tensor(group_id), num_groups,
                        is_leader, num_blocks, ENABLE_PROFILING)

    def __init__(self, profiler_buffer, stride, group_id, num_groups, is_leader, num_blocks, ENABLE_PROFILING):
        """
           is_leader: select a thread as leader to track start and end
        """
        self.group_id = group_id
        self.stride = stride
        self.buffer = profiler_buffer
        self.is_leader = is_leader
        self.num_groups = num_groups
        self.ENABLE_PROFILING = ENABLE_PROFILING
        self.NUM_BITS_ID = tl.constexpr(NUM_BITS_ID)
        self.NUM_BITS_PAYLOAD = tl.constexpr(NUM_BITS_PAYLOAD)
        self.NUM_BITS_TASK_TYPE = tl.constexpr(NUM_BITS_TASK_TYPE)
        self.NUM_BITS_EVENT = tl.constexpr(NUM_BITS_EVENT)

    @triton.jit
    def encode_tag(self, is_start, task_type):
        """
            TAG = GLOBAL ID | TASK TYPE | EVENT(IS_START)
        """
        tl.static_assert(self.NUM_BITS_TASK_TYPE + self.NUM_BITS_EVENT + self.NUM_BITS_ID <= 32)
        pid_size_x = tl.num_programs(axis=0)
        pid_size_y = tl.num_programs(axis=1)
        pid = tl.program_id(0) + tl.program_id(1) * pid_size_x + tl.program_id(2) * pid_size_x * pid_size_y
        global_id = pid * self.num_groups + self.group_id
        GLOBAL_ID_OFFSET: tl.constexpr = self.NUM_BITS_TASK_TYPE + self.NUM_BITS_EVENT
        TASK_TYPE_OFFSET: tl.constexpr = self.NUM_BITS_EVENT
        return (global_id << GLOBAL_ID_OFFSET) | (task_type << TASK_TYPE_OFFSET) | (is_start)

    @triton.jit
    def get_timestamp(self):
        return globaltimer_lo()

    @triton.jit
    def get_profile_entry(self, is_start, task_type):
        timestamp = self.get_timestamp()
        tag = self.encode_tag(is_start, task_type)
        return pack_b32_v2(tag, timestamp)

    @triton.jit
    def record(self, is_start, task_type):
        if self.ENABLE_PROFILING:
            membar("cta")
            if self.is_leader:
                entry = self.get_profile_entry(is_start, task_type)
                st(self.buffer, entry)
            self.buffer += self.stride
            membar("cta")
        return self
