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
    # task encoding
    for i in range(len(sm_wq_list)):
        for j in range(len(sm_wq_list[i])):
            sm_wq_list[i][j] = sm_wq_list[i][j].encoding()

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
    return wq_tensor, num_tasks_tensor


def round_robin_scheduler(num_sms, megakernel_tasks):
    sm_wq_list = [[] for i in range(num_sms)]
    for idx, task in enumerate(megakernel_tasks):
        sm_wq_list[idx % num_sms].append(task)
    return sm_wq_list


def zig_zag_scheduler(num_sms, megakernel_tasks):
    sm_wq_list = [[] for i in range(num_sms)]
    iter = 0
    for idx, task in enumerate(megakernel_tasks):
        if iter == 0:
            sm_wq_list[idx % num_sms].append(task)
        else:
            sm_wq_list[num_sms - 1 - idx % num_sms].append(task)
        iter = iter ^ 1
    return sm_wq_list


def task_dependency_opt(sm_wq_list):
    for sm_id in range(len(sm_wq_list)):
        wq = sm_wq_list[sm_id]
        num_tasks = len(wq)
        if num_tasks > 0:
            pre_dependency = wq[0].dependency
            for idx in range(1, num_tasks):
                if pre_dependency.cover(wq[idx].dependency):
                    wq[idx].dependency = copy.deepcopy(wq[idx].dependency)
                    wq[idx].dependency.to_empty()
                else:
                    pre_dependency = wq[idx].dependency
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
    wq_tensor, num_tasks_tensor = work_queue_list_to_device_tensor(sm_wq_list)
    return wq_tensor, num_tasks_tensor
