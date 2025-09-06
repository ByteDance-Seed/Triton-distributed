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
import torch

from enum import Enum
import copy


class SchedulingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    ZIG_ZAG = "zig_zag"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)


def work_queue_list_to_device_tensor(sm_wq_list):
    num_tasks_per_sm = [len(q) for q in sm_wq_list]
    max_num_tasks = max(num_tasks_per_sm)
    padding = -1
    max_tuple_len = 0
    all_task_deps = []

    max_num_tiles_per_op = 1
    max_layer_id = 1
    max_task_id = 1

    # task encoding
    for i in range(len(sm_wq_list)):
        for j in range(len(sm_wq_list[i])):
            deps_l = len(all_task_deps)
            if isinstance(sm_wq_list[i][j].dependency, (tuple, list)):
                deps_r = deps_l + len(sm_wq_list[i][j].dependency)
                all_task_deps += sm_wq_list[i][j].dependency
            else:
                deps_r = deps_l + 1
                all_task_deps += [sm_wq_list[i][j].dependency]

            task = sm_wq_list[i][j]
            max_num_tiles_per_op = max(max_num_tiles_per_op, task.num_tiles)
            max_layer_id = max(task.layer_id, max_layer_id)
            max_task_id = max(task.task_id, max_task_id)

            sm_wq_list[i][j] = sm_wq_list[i][j].encoding_with_deps(deps_l, deps_r)

    # task_id/layer_id start from zero
    scoreboard = torch.zeros((max_layer_id + 1, max_task_id + 1, max_num_tiles_per_op), dtype=torch.int32,
                             device=torch.cuda.current_device())
    task_deps_entrys = []
    for dep in all_task_deps:
        sb_offset_base = dep.layer_id * scoreboard.stride(0) + dep.task_id * scoreboard.stride(1)
        sb_start = sb_offset_base + dep.start_tiles
        sb_end = sb_offset_base + dep.end_tiles
        task_deps_entrys.append((sb_start, sb_end))

    for que in sm_wq_list:
        for task in que:
            max_tuple_len = max(max_tuple_len, len(task))

    for i in range(len(sm_wq_list)):
        queue = sm_wq_list[i]
        for j in range(len(queue)):
            if len(queue[j]) < max_tuple_len:
                queue[j] = queue[j] + (padding, ) * (max_tuple_len - len(queue[j]))

        if len(queue) < max_num_tasks:
            sm_wq_list[i] = queue + (max_num_tasks - len(queue)) * [(padding, ) * max_tuple_len]
    # print(queue_per_sm)
    # use uint32 to avoid data_ptr overflow
    wq_tensor = torch.tensor(sm_wq_list, dtype=torch.uint32, device=torch.cuda.current_device())
    num_tasks_tensor = torch.tensor(num_tasks_per_sm, dtype=torch.int32, device=torch.cuda.current_device())
    wq_tensor = wq_tensor.permute(1, 0, 2).contiguous()
    task_deps_tensor = torch.tensor(task_deps_entrys, dtype=torch.int32, device=torch.cuda.current_device())
    if len(task_deps_entrys) == 0:
        task_deps_tensor = task_deps_tensor.reshape(0, 2)
    return wq_tensor, num_tasks_tensor, scoreboard, task_deps_tensor


def round_robin_scheduler(num_sms, megakernel_tasks):
    sm_wq_list = [[] for i in range(num_sms)]
    for idx, task in enumerate(megakernel_tasks):
        sm_wq_list[idx % num_sms].append(task)
    return sm_wq_list


def zig_zag_scheduler(num_sms, megakernel_tasks):
    sm_wq_list = [[] for i in range(num_sms)]
    iter = 1
    for idx, task in enumerate(megakernel_tasks):
        if idx % num_sms == 0:
            iter = iter ^ 1

        if iter == 0:
            sm_wq_list[idx % num_sms].append(task)
        else:
            sm_wq_list[num_sms - 1 - idx % num_sms].append(task)

    for idx, wq in enumerate(sm_wq_list):
        print(f"len(wq) = {len(wq)}")
    return sm_wq_list


def task_dependency_opt(sm_wq_list):
    for sm_id in range(len(sm_wq_list)):
        wq = sm_wq_list[sm_id]
        num_tasks = len(wq)
        deps_range = {}

        if num_tasks > 0:
            for idx in range(0, num_tasks):
                new_dependency = []
                assert isinstance(wq[idx].dependency, list)
                for j in range(len(wq[idx].dependency)):
                    cur_dep = copy.deepcopy(wq[idx].dependency[j])
                    key = cur_dep.key()
                    if key not in deps_range:
                        deps_range[key] = []
                    range_list = deps_range.get(key, [])
                    convered_by_pre_task = False
                    for l, r in range_list:
                        if l <= cur_dep.start_tiles and r >= cur_dep.end_tiles:
                            convered_by_pre_task = True
                            break
                    if not convered_by_pre_task:
                        new_dependency.append(cur_dep)

                    deps_range[key] = range_list + [(cur_dep.start_tiles, cur_dep.end_tiles)]
                wq[idx].dependency = new_dependency
        sm_wq_list[sm_id] = wq
    return sm_wq_list


def enque_tasks(num_sms, megakernel_tasks, strategy: SchedulingStrategy = SchedulingStrategy.ROUND_ROBIN,
                enable_dependency_opt=True):

    if strategy == SchedulingStrategy.ROUND_ROBIN:
        sm_wq_list = round_robin_scheduler(num_sms, megakernel_tasks)
    elif strategy == SchedulingStrategy.ZIG_ZAG:
        sm_wq_list = zig_zag_scheduler(num_sms, megakernel_tasks)
    else:
        raise NotImplementedError(f"Unsupport strategy {strategy}")
    if enable_dependency_opt:
        sm_wq_list = task_dependency_opt(sm_wq_list)
    return work_queue_list_to_device_tensor(sm_wq_list)
