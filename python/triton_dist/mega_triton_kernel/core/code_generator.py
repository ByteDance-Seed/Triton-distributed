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
from typing import List, Dict, Tuple
import textwrap
from .registry import registry
from .task_base import TaskBase, CodeGenKey


def make_mega_kernel_src(tasks_dispatch_code: str, enalbe_profiling: bool, task_types_and_str: Dict[int, str]) -> str:
    """
        max_task_type: profiling use only
    """
    max_task_type = 0
    for k, v in task_types_and_str.items():
        max_task_type = max(max_task_type, k)
    scoreboard_wait_deps_task_type = max_task_type + 1
    task_decoding_task_type = scoreboard_wait_deps_task_type + 1
    task_types_and_str[scoreboard_wait_deps_task_type] = "scoreboard_wait_deps"
    task_types_and_str[task_decoding_task_type] = "task_decoding"

    src = f"""
import triton
import triton.language as tl
from triton_dist.mega_triton_kernel.kernels import *

from triton_dist.mega_triton_kernel.kernels.task_context import Scoreboard
from triton_dist.tools.profiler import Profiler
from triton.language.extra.cuda.language_extra import tid
@triton.jit
def MEGA_TRITON_KERNEL(
    {"profiler_buf, # ensor<uint64>" if enalbe_profiling else ""}
    work_queues, # [MAX_INS, NUM_SMS, INS], int32
    num_tasks_per_wq, #[num_sms,]
    scoreboard_ptr,
    task_deps_ptr,  # [num_deps_entry_of_all_tasks, INT_PER_DEPS]

    INT_PER_DEPS: tl.constexpr,
    INT_PER_TASK: tl.constexpr,
    MAX_TASK_ID: tl.constexpr,
    MAX_NUM_TILES_PER_OP: tl.constexpr,
    MAX_NUM_TENSOR_DIMS: tl.constexpr,
    NUM_SMS: tl.constexpr,
    num_warps: tl.constexpr
):
    {f"profiler = Profiler.create(profiler_buf, 0, is_leader=(tid(0) == 0), ENABLE_PROFILING={enalbe_profiling})" if enalbe_profiling else ""}

    WARP_SIZE: tl.constexpr = 32
    NUM_THREADS: tl.constexpr = num_warps * WARP_SIZE
    scoreboard = Scoreboard(task_deps_ptr, INT_PER_DEPS, scoreboard_ptr, MAX_TASK_ID, MAX_NUM_TILES_PER_OP, tl.constexpr(1), NUM_THREADS)
    sm_id = tl.program_id(axis=0)
    num_tasks = tl.load(num_tasks_per_wq + sm_id)
    offset = INT_PER_TASK * NUM_SMS

    TASK_TYPE_OFFSET = 0
    LAYER_ID_OFFSET = 1
    TASK_ID_OFFSET = 2
    TILE_ID_OR_START_OFFSET = 3
    DEPEND_ENTRY_START_OFFSET = 4
    DEPEND_ENTRY_END_OFFSET = 5
    IO_TENSORS_OFFSET = 6

    for i in range(num_tasks):
        task_type = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_TYPE_OFFSET).to(tl.int32)
        layer_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + LAYER_ID_OFFSET).to(tl.int32)
        task_id = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TASK_ID_OFFSET).to(tl.int32)
        tile_id_or_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + TILE_ID_OR_START_OFFSET).to(tl.int32)
        depend_entry_start = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_START_OFFSET).to(tl.int32)
        depend_entry_end = tl.load(work_queues + i * offset + sm_id * INT_PER_TASK + DEPEND_ENTRY_END_OFFSET).to(tl.int32)
        
        io_tensors_ptr = work_queues + i * offset + sm_id * INT_PER_TASK + IO_TENSORS_OFFSET
        task_base_info = TaskBaseInfo(io_tensors_ptr, layer_id, task_id, tile_id_or_start, depend_entry_start, depend_entry_end, MAX_NUM_TENSOR_DIMS)

        # task kernel need to set signal for each tile
        {f"profiler = profiler.record(is_start=True, task_type={scoreboard_wait_deps_task_type})" if enalbe_profiling else ""}
        scoreboard.wait_deps(task_base_info)
        {f"profiler = profiler.record(is_start=False, task_type={scoreboard_wait_deps_task_type})" if enalbe_profiling else ""}

        #### run task ####
        {"profiler = profiler.record(is_start=True, task_type=task_type)" if enalbe_profiling else ""}
{textwrap.indent(tasks_dispatch_code.strip(), '        ')}
        {"profiler = profiler.record(is_start=False, task_type=task_type)" if enalbe_profiling else ""}
"""
    return src, task_types_and_str


class CodeGenerator:

    def __init__(self):
        self._condition_and_codes: Dict[int, List[Tuple[CodeGenKey, str]]] = {}
        self._variable_names: Dict[str, str] = {
            "layer_id": "task_base_info.layer_id",
            "task_id": "task_base_info.task_id",
            "task_type": "task_type",
        }
        self._task_types_and_str: Dict[int, str] = {}

    def generate_task_dispatch_code(self, condition: CodeGenKey, code: str, is_first_branch=True) -> str:
        if condition.only_use_task_type():
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {condition.task_type}: # {self._task_types_and_str[condition.task_type]}
{textwrap.indent(code.strip(), '    ')}
"""
        else:
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {condition.task_type}: # {self._task_types_and_str[condition.task_type]}
    if {self._variable_names["layer_id"]} == {condition.layer_id} and {self._variable_names["task_id"]} == {condition.task_id}:
{textwrap.indent(code.strip(), '        ')}
"""

    def generate_for_each_task(self, condition: CodeGenKey, code: str, is_first_branch=True):
        return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["layer_id"]} == {condition.layer_id} and {self._variable_names["task_id"]} == {condition.task_id}:
{textwrap.indent(code.strip(), '    ')}
"""

    def generate_for_each_task_type(self, key_and_tasks_list, is_first_branch=True) -> str:
        assert len(key_and_tasks_list) > 0
        same_code = True
        task_type = key_and_tasks_list[0][0].task_type
        for key, code in key_and_tasks_list:
            if code != key_and_tasks_list[0][1]:
                same_code = False
        if same_code:
            code = key_and_tasks_list[0][1]
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {task_type}: # {self._task_types_and_str[task_type]}
{textwrap.indent(code.strip(), '    ')}
"""
        else:
            # each op may split into multi task, these tasks have same (task_type, layer_id, task_id)
            # only need to generate code once for these tasks
            already_generated = set()
            all_codes = ""
            is_first_task = True
            for key, code in key_and_tasks_list:
                if key in already_generated:
                    continue
                already_generated.add(key)
                cur_code = self.generate_for_each_task(key, code, is_first_task)
                all_codes += cur_code
                is_first_task = False
            return f"""
{'if' if is_first_branch else 'elif'} {self._variable_names["task_type"]} == {task_type}: # {self._task_types_and_str[task_type]}
{textwrap.indent(all_codes.strip(), '    ')}
"""

    def generate_code(self, tasks: List['TaskBase'], enable_profiling=False) -> str:
        self._condition_and_codes.clear()
        self._task_types_and_str.clear()

        for task in tasks:
            key = task.get_codegen_key(task.layer_id, task.task_id)
            assert isinstance(key, CodeGenKey)
            task_type = type(task)
            code = registry.get_codegen(task_type)(task)
            if key.task_type not in self._condition_and_codes:
                self._condition_and_codes[key.task_type] = []
            self._condition_and_codes[key.task_type].append((key, code))
            self._task_types_and_str[key.task_type] = task_type.__name__

        # TODO(zhengxuegui.0): branch optimization
        is_first_branch = True
        tasks_dispatch_code = ""

        for task_type, key_and_tasks_list in self._condition_and_codes.items():
            tasks_dispatch_code += self.generate_for_each_task_type(key_and_tasks_list, is_first_branch)
            is_first_branch = False

        mege_kernel_src, self._task_types_and_str = make_mega_kernel_src(tasks_dispatch_code, enable_profiling,
                                                                         self._task_types_and_str)
        return mege_kernel_src, self._task_types_and_str
