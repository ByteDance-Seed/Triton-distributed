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
from typing import Tuple, List
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, MAX_NUM_TENSOR_DIMS
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class BarrierAllIntraNodeConfig(ConfigBase):
    pass


@dataclass
class BarrierAllIntraNodeTask(TaskBase):
    config: BarrierAllIntraNodeConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        return (self.extra_params["local_rank"], self.extra_params["local_world_size"])

    def io_to_tuple(self):
        io_tuple = tuple()
        assert len(self.io_tensors) == 2
        assert len(self.io_tensors[0]) - len(self.io_tensors[1]) == 1
        # inputs/outputs except barrier are only used to build dependency in graph level and are useless for the kernel.
        barrier_tensor = self.io_tensors[0][0]

        data_ptr = barrier_tensor.data_ptr()
        ptr_high = (data_ptr >> 32) & 0xFFFFFFFF
        ptr_low = data_ptr & 0xFFFFFFFF

        shape = list(barrier_tensor.shape)
        assert MAX_NUM_TENSOR_DIMS >= len(shape)
        padded_shape = shape + [1] * (MAX_NUM_TENSOR_DIMS - len(shape))

        tensor_tuple = (ptr_low, ptr_high) + tuple(padded_shape)
        assert len(tensor_tuple) % 2 == 0, "tensor data_ptr alignemnt"
        io_tuple += tensor_tuple
        return io_tuple


def barrier_all_intra_node_config_factory(**kwargs) -> BarrierAllIntraNodeConfig:
    return dataclasses.replace(BarrierAllIntraNodeConfig(), **kwargs)


def codegen_barrier_all_intra_node(task: BarrierAllIntraNodeConfig) -> str:
    code = """
barrier_all_intra_node_task_compute(task_base_info, scoreboard)
"""
    return code


@registry.register_task(op_type="barrier_all_intra_node", task_cls=BarrierAllIntraNodeTask,
                        config_factory=barrier_all_intra_node_config_factory,
                        codegen_func=codegen_barrier_all_intra_node)
class BarrierAllIntraNodeTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        kernel_config = cls.create_config()
        num_tiles = 1
        task_id = cls.get_task_id(layer_id)
        cls.log(
            f"BarrierAllIntraNode Task: num_tiles = {num_tiles}, task_id = {task_id}, dependency = {dependency}, extra_params = {extra_params}"
        )
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks
