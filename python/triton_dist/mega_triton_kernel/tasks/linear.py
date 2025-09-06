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
from typing import Any, Dict, List
from .utils import cdiv
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc, DeviceProp
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class LinearConfig(ConfigBase):
    BLOCK_SIZE_M: int = 16
    BLOCK_SIZE_N: int = 128
    BLOCK_SIZE_K: int = 128
    NUM_STAGES: int = 4


@dataclass
class LinearTask(TaskBase):
    config: LinearConfig


@dataclass
class MLPFC1Config(LinearConfig):
    pass


@dataclass
class MLPFC1Task(LinearTask):
    config: MLPFC1Config


@dataclass
class MLPFC2Config(LinearConfig):
    pass


@dataclass
class MLPFC2Task(LinearTask):
    config: MLPFC2Config


@dataclass
class QKVProjTask(LinearTask):
    config: LinearConfig


@dataclass
class OProjTask(LinearTask):
    config: LinearConfig


def linear_config_factory(**kwargs) -> LinearConfig:
    return dataclasses.replace(LinearConfig(), **kwargs)


def mlp_fc1_config_factory(**kwargs) -> MLPFC1Config:
    default = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 128,
        'NUM_STAGES': 6,
    }
    default.update(kwargs)
    return MLPFC1Config(**default)


def mlp_fc2_config_factory(**kwargs) -> MLPFC2Config:
    default = {
        'BLOCK_SIZE_M': 16,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 256,
        'NUM_STAGES': 6,
    }
    default.update(kwargs)
    return MLPFC2Config(**default)


def codegen_linear(task: LinearTask) -> str:
    config: MLPFC1Config = task.config
    a, b = task.io_tensors[0]
    M, K = a.shape
    ALIGNMENT_K = 1
    if K % 16 == 0:
        ALIGNMENT_K = 16
    code = f"""
linear_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES}, ALIGNMENT_K={ALIGNMENT_K})
"""
    return code


def codegen_mlp_fc1(task: MLPFC1Task) -> str:
    config: MLPFC1Config = task.config
    code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    return code


def codegen_mlp_fc2(task: MLPFC2Task) -> str:
    config: MLPFC2Config = task.config
    code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    return code


def codegen_qkv_proj(task: QKVProjTask) -> str:
    config: LinearConfig = task.config
    code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    return code


def codegen_o_proj(task: OProjTask) -> str:
    config: LinearConfig = task.config
    code = f"""
fc1_task_compute(task_base_info, scoreboard, BLOCK_SIZE_M={config.BLOCK_SIZE_M}, BLOCK_SIZE_N={config.BLOCK_SIZE_N},
                BLOCK_SIZE_K={config.BLOCK_SIZE_K}, NUM_STAGES={config.NUM_STAGES})
"""
    return code


@registry.register_task(op_type="linear", task_cls=LinearTask, config_factory=linear_config_factory,
                        codegen_func=codegen_linear)
class LinearTaskBuilder(TaskBuilderBase):

    @classmethod
    def get_problem_size(cls, io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]):
        a, b = io_tensors[0]
        M, K = a.shape
        N, K = b.shape
        return (M, N, K)

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True, config_args={}) -> List[TaskBase]:
        assert tile_wise == True  # noqa: E712
        kernel_config = cls.create_config(**config_args)
        task_id = cls.get_task_id(layer_id)
        BLOCK_SIZE_M = kernel_config.BLOCK_SIZE_M
        BLOCK_SIZE_N = kernel_config.BLOCK_SIZE_N
        M, N, K = cls.get_problem_size(io_tensors, extra_params)
        num_tiles_m = cdiv(M, BLOCK_SIZE_M)
        num_tiles_n = cdiv(N, BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_n
        x, w = io_tensors[0]
        y = io_tensors[1][0]

        num_sm = device_prop.NUM_SMS
        tasks = []
        cls.log(
            f"Linear Task: M = {M}, N = {N}, K = {K}, num_tiles = {num_tiles}, num_sm = {num_sm}, tile_wise = {tile_wise}, dependency = {dependency}, BLOCK_SIZE_M ={BLOCK_SIZE_M}, BLOCK_SIZE_N = {BLOCK_SIZE_N}"
        )
        for tm in range(num_tiles_m):
            for tn in range(num_tiles_n):
                tile_id = tm * num_tiles_n + tn
                bm = min(BLOCK_SIZE_M, M - tm * BLOCK_SIZE_M)
                bn = min(BLOCK_SIZE_N, N - tn * BLOCK_SIZE_N)
                x_desc = InputDependencyDesc(x, require_full=False, start_indices=(tm * BLOCK_SIZE_M, 0),
                                             data_sizes=(bm, K))
                w_desc = InputDependencyDesc(w, require_full=False, start_indices=(tn * BLOCK_SIZE_N, 0),
                                             data_sizes=(bn, K))
                y_desc = OutputTilingDesc(tile_sizes=(BLOCK_SIZE_M, BLOCK_SIZE_N),
                                          start_indices=(tm * BLOCK_SIZE_M, tn * BLOCK_SIZE_N))
                inputs_dep = {x: x_desc, w: w_desc}
                outs_tile_mapping = {y: y_desc}
                tasks.append(
                    cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                     extra_params, inputs_dep, outs_tile_mapping))
        return tasks

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params)


@registry.register_task(op_type="mlp_fc1", task_cls=MLPFC1Task, config_factory=mlp_fc1_config_factory,
                        codegen_func=codegen_mlp_fc1)
class MLPFC1TaskBuilder(LinearTaskBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True)


# reduce branch in mega kernel, just use task type as condition
@registry.register_task(op_type="mlp_fc2", task_cls=MLPFC2Task, config_factory=mlp_fc2_config_factory,
                        codegen_func=codegen_mlp_fc2)
class MLPFC2TaskBuilder(LinearTaskBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True)


@registry.register_task(op_type="qkv_proj", task_cls=QKVProjTask, config_factory=linear_config_factory,
                        codegen_func=codegen_qkv_proj)
class QKVProjTaskBuilder(LinearTaskBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        config_args = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 256,
            "NUM_STAGES": 5,
        }
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True,
                                     config_args=config_args)


@registry.register_task(op_type="o_proj", task_cls=OProjTask, config_factory=linear_config_factory,
                        codegen_func=codegen_o_proj)
class OProjTaskBuilder(LinearTaskBuilder):

    @classmethod
    def build_tasks(cls, device_prop: 'DeviceProp', layer_id: int, dependency: TaskDependency,
                    io_tensors: List[List['torch.Tensor']], extra_params: Dict[str, Any]) -> List[TaskBase]:
        config_args = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "NUM_STAGES": 7,
        }
        return cls._build_tasks_impl(device_prop, layer_id, dependency, io_tensors, extra_params, tile_wise=True,
                                     config_args=config_args)
