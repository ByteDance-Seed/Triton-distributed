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
from ..core.task_base import TaskBase, TaskDependency
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class PrefetchConfig(ConfigBase):
    pass


@dataclass
class PrefetchTask(TaskBase):
    config: PrefetchConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        return (self.extra_params["num_pid"], -1)


def prefetch_config_factory(**kwargs) -> PrefetchConfig:
    return dataclasses.replace(PrefetchConfig(), **kwargs)


def codegen_prefetch(task: PrefetchConfig) -> str:
    code = """
prefetch_task_compute(task_base_info, scoreboard)
"""
    return code


@registry.register_task(op_type="prefetch", task_cls=PrefetchTask, config_factory=prefetch_config_factory,
                        codegen_func=codegen_prefetch)
class PrefetchTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        weight = io_tensors[0][0]
        assert len(weight.shape) == 2
        M, N = weight.shape
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()

        num_tiles = 1
        cls.log(f"Prefetch Task: num_tiles = {num_tiles}, dependency = {dependency}")
        # persistent kernel
        num_tasks = num_tiles
        tasks = []
        extra_params["num_pid"] = num_tasks
        for i in range(num_tasks):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks
