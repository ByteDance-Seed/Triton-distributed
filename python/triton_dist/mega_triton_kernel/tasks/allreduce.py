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
from typing import List
from .utils import cdiv
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class AllReduceConfig(ConfigBase):
    BLOCK_SIZE: int = 1024


@dataclass
class AllReduceTask(TaskBase):
    config: AllReduceConfig


def allreduce_config_factory(**kwargs) -> AllReduceConfig:
    return dataclasses.replace(AllReduceConfig(), **kwargs)


def codegen_allreduce(task: AllReduceConfig) -> str:
    config: AllReduceConfig = task.config

    code = f"""
allreduce_task_compute(task_base_info, scoreboard, BLOCK_SIZE={config.BLOCK_SIZE})
"""
    return code


@registry.register_task(op_type="allreduce", task_cls=AllReduceTask, config_factory=allreduce_config_factory,
                        codegen_func=codegen_allreduce)
class AllReduceTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        input, output = io_tensors[0][0], io_tensors[1][0]
        num_elements = output.numel()
        assert input.shape == output.shape and len(input.shape) == 1
        assert num_elements * output.element_size() % 128 == 0
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        num_tiles = cdiv(num_elements, kernel_config.BLOCK_SIZE)

        cls.log(
            f"AllReduce Task: num_tiles = {num_tiles}, num_elements = {num_elements}, BLOCK_SIZE = {kernel_config.BLOCK_SIZE}, dependency = {dependency}"
        )
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks
