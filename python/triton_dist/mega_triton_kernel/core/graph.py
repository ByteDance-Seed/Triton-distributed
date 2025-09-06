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
from dataclasses import dataclass
import torch
from triton_dist.models.utils import logger
from .task_base import TaskDependency


def _list_to_intervals(nums):
    if not nums:
        return []

    sorted_nums = sorted(nums)
    intervals = []

    start = sorted_nums[0]

    for i in range(1, len(sorted_nums)):
        if sorted_nums[i] != sorted_nums[i - 1] + 1:
            intervals.append((start, sorted_nums[i - 1] + 1))
            start = sorted_nums[i]
    intervals.append((start, sorted_nums[-1] + 1))

    return intervals


def _deps_list_to_dependency(deps_list, layer_id, task_id):
    intervals = _list_to_intervals(deps_list)
    ret = []
    for l, r in intervals:
        ret.append(TaskDependency(layer_id=layer_id, task_id=task_id, start_tiles=l, end_tiles=r))
    return ret


class Node:

    def __init__(self, tasks, op_type, io_tensors, extra_params, input_producers: Dict[int, 'Node'] = None):
        self.tasks = tasks
        self.op_type = op_type
        self.io_tensors = io_tensors
        self.extra_params = extra_params
        self.input_producers: Dict[int, Tuple[Node, int]] = {} if input_producers is None else input_producers

    def add_input_producer(self, input_index, src_node, src_out_idx):
        assert isinstance(src_node, Node)
        self.input_producers[input_index] = (src_node, src_out_idx)

    def __repr__(self):
        input_producers_str = "{"
        for input_idx, (src_node, src_idx) in self.input_producers.items():
            cur_kv = f"{input_idx}: ({src_idx}, Node(op_type = {src_node.op_type}, layer_id = {src_node.tasks[0].layer_id}, task_id = {src_node.tasks[0].task_id})), "
            input_producers_str += cur_kv
        input_producers_str += "}"
        return f"Node(op_type = {self.op_type}, layer_id = {self.tasks[0].layer_id}, task_id = {self.tasks[0].task_id},  input_producers = {input_producers_str})"


@dataclass
class Buffer:
    ptr: int
    nbytes: int

    def __hash__(self):
        return hash((self.ptr, self.nbytes))


@dataclass
class OutputTensor:
    node: Node
    idx: int
    tensor: torch.Tensor
    buf: Buffer

    def __repr__(self):
        return f"OutputTensor(node={self.node}, idx={self.idx}, buf={self.buf})"


class Graph:

    def __init__(self):
        self._logger = logger
        self._nodes: List[Node] = []
        self.tensor_mapping: Dict[Buffer, OutputTensor] = {}

    def find_producer(self, cur_tensor: torch.Tensor) -> OutputTensor:
        tensor_buffer = Buffer(ptr=cur_tensor.data_ptr(), nbytes=cur_tensor.nbytes)
        ret = self.tensor_mapping.get(tensor_buffer, None)
        return ret

    def set_producer(self, out_tensor: torch.Tensor, out_idx: int, node: Node):
        buf = Buffer(ptr=out_tensor.data_ptr(), nbytes=out_tensor.nbytes)
        o = OutputTensor(node=node, idx=out_idx, tensor=out_tensor, buf=buf)
        self.tensor_mapping[buf] = o

    def new_node(self, tasks, op_type, io_tensors, extra_params) -> Node:
        cur_node = Node(tasks=tasks, op_type=op_type, io_tensors=io_tensors, extra_params=extra_params)
        input_tensors, output_tensors = io_tensors

        for idx, ten in enumerate(input_tensors):
            producer = self.find_producer(ten)
            self._logger.log(f"op type = {op_type}, idx = {idx}, producer = {producer}", level="debug")
            if producer is not None:
                cur_node.add_input_producer(idx, producer.node, producer.idx)

        for idx, ten in enumerate(output_tensors):
            self.set_producer(ten, idx, cur_node)

        self._nodes.append(cur_node)
        return cur_node

    def to_tasks(self):
        all_tasks_after_build_deps = []
        for node in self._nodes:
            assert len(node.tasks) > 0
            for cur_task in node.tasks:
                task_deps_cur_task = []
                for input_idx, (src_node, src_out_idx) in node.input_producers.items():
                    producer_tasks_list = src_node.tasks
                    if len(producer_tasks_list) > 0:

                        deps_tile_ids_from_cur_producer = []
                        for producer_task in producer_tasks_list:
                            if cur_task.has_data_dependency(producer_task, src_out_idx, input_idx):
                                deps_tile_ids_from_cur_producer.append(producer_task.tile_id_or_start)
                        task_deps_cur_task += _deps_list_to_dependency(deps_tile_ids_from_cur_producer,
                                                                       layer_id=producer_tasks_list[0].layer_id,
                                                                       task_id=producer_tasks_list[0].task_id)
                if len(cur_task.dependency) != len(
                        task_deps_cur_task) or task_deps_cur_task[0].start_tiles != cur_task.dependency[
                            0].start_tiles or task_deps_cur_task[0].end_tiles != cur_task.dependency[0].end_tiles:
                    self._logger.log(f"cur_task = {cur_task}, task_deps_cur_task = {task_deps_cur_task}", level="debug")
                cur_task.dependency = task_deps_cur_task
                all_tasks_after_build_deps.append(cur_task)
        return all_tasks_after_build_deps
