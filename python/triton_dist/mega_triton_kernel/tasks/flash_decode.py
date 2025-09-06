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
import math
from typing import Tuple, List
from .utils import cdiv
import triton
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class AttnConfig(ConfigBase):
    BLOCK_N: int = 64
    BLOCK_HEAD_DIM: int = 128
    BLOCK_DPE: int = 0
    BLOCK_DV: int = 128
    BLOCK_H: int = 16
    NUM_KV_SPLITS: int = 32


@dataclass
class AttnSplitTask(TaskBase):
    config: AttnConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # sm_scale/soft_cap as constexpr in codegen
        return ()


@dataclass
class AttnCombineTask(TaskBase):
    config: AttnConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # sm_scale/soft_cap as constexpr in codegen
        return ()


def attn_config_factory(**kwargs) -> AttnConfig:
    return dataclasses.replace(AttnConfig(), **kwargs)


def codegen_attn_split(task: AttnSplitTask) -> str:
    config: AttnConfig = task.config
    query, key_cache, v_cache, block_tables, kv_lens = task.io_tensors[0]
    partial_out, lse = task.io_tensors[1]
    NUM_Q_HEADS, Q_HEAD_DIM = query.shape[-2], query.shape[-1]
    PAGE_SIZE, NUM_KV_HEADS, V_HEAD_DIM = v_cache.shape[-3], v_cache.shape[-2], v_cache.shape[-1]
    MAX_NUM_BLOCKS_PER_SEQ = block_tables.shape[-1]

    code = f"""
attn_gqa_fwd_batch_decode_split_kv_task_compute(
    task_base_info, scoreboard, SM_SCALE={task.extra_params["sm_scale"]}, SOFT_CAP={task.extra_params["soft_cap"]},
    NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, Q_HEAD_DIM={Q_HEAD_DIM}, V_HEAD_DIM={V_HEAD_DIM}, PAGE_SIZE={PAGE_SIZE},
    MAX_NUM_BLOCKS_PER_SEQ={MAX_NUM_BLOCKS_PER_SEQ}, BLOCK_N={config.BLOCK_N}, BLOCK_HEAD_DIM={config.BLOCK_HEAD_DIM},
    BLOCK_DPE={config.BLOCK_DPE}, BLOCK_DV={config.BLOCK_DV}, BLOCK_H={config.BLOCK_H}, NUM_KV_SPLITS={config.NUM_KV_SPLITS}
)
"""
    return code


def codegen_attn_combine(task: AttnSplitTask) -> str:
    config: AttnConfig = task.config
    kv_lens, partial_out, lse = task.io_tensors[0]
    output = task.io_tensors[1][0]
    NUM_Q_HEADS, V_HEAD_DIM = output.shape[-2], output.shape[-1]
    code = f"""
attn_gqa_fwd_batch_decode_combine_task_compute(
    task_base_info, scoreboard, NUM_Q_HEADS={NUM_Q_HEADS}, V_HEAD_DIM={V_HEAD_DIM}, BLOCK_DV={config.BLOCK_DV}, NUM_KV_SPLITS={config.NUM_KV_SPLITS}
)
"""
    return code


@registry.register_task(op_type="attn_split", task_cls=AttnSplitTask, config_factory=attn_config_factory,
                        codegen_func=codegen_attn_split)
class AttnSplitTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        query, key_cache, v_cache, block_tables, kv_lens = io_tensors[0]
        partial_out, lse = io_tensors[1]
        assert len(query.shape) == 4, f"query shape mismatch, expect (bs, seq, nheads, head_dim), but got {query.shape}"
        q_head_dim = query.shape[-1]
        v_head_dim = v_cache.shape[-1]
        num_q_heads = query.shape[-2]
        num_v_heads = v_cache.shape[-2]
        assert num_q_heads % num_v_heads == 0
        num_q_heads_per_group = num_q_heads // num_v_heads
        batch = query.shape[0]
        BLOCK_HEAD_DIM = 2**int(math.log2(q_head_dim))
        BLOCK_DPE = q_head_dim - BLOCK_HEAD_DIM
        BLOCK_DV = triton.next_power_of_2(v_head_dim)
        NUM_KV_SPLITS = extra_params["NUM_KV_SPLITS"]
        kernel_config = cls.create_config(**{
            "BLOCK_HEAD_DIM": BLOCK_HEAD_DIM, "BLOCK_DPE": BLOCK_DPE, "BLOCK_DV": BLOCK_DV, "NUM_KV_SPLITS":
            NUM_KV_SPLITS
        })
        task_id = cls.get_task_id(layer_id)

        BLOCK_H = kernel_config.BLOCK_H
        NUM_KV_SPLITS = kernel_config.NUM_KV_SPLITS
        num_split_tiles = batch * cdiv(num_q_heads, min(num_q_heads_per_group, BLOCK_H)) * NUM_KV_SPLITS
        tasks = []
        cls.log(f"Attn Split Task: num_tiles = {num_split_tiles}, kernel_config = {kernel_config}")
        for i in range(num_split_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_split_tiles, kernel_config, dependency, io_tensors,
                                 extra_params))
        return tasks


@registry.register_task(op_type="attn_combine", task_cls=AttnCombineTask, config_factory=attn_config_factory,
                        codegen_func=codegen_attn_combine)
class AttnCombineTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        kv_lens, partial_out, lse = io_tensors[0]
        output = io_tensors[1][0]
        v_head_dim = output.shape[-1]
        num_q_heads = output.shape[-2]
        BLOCK_DV = triton.next_power_of_2(v_head_dim)
        batch = output.shape[0]
        num_combine_tile = batch * num_q_heads

        # guarantee that NUM_KV_SPLITS same as AttnSplit
        NUM_KV_SPLITS = extra_params["NUM_KV_SPLITS"]
        kernel_config = cls.create_config(**{"BLOCK_DV": BLOCK_DV, "NUM_KV_SPLITS": NUM_KV_SPLITS})
        task_id = cls.get_task_id(layer_id)

        tasks = []
        cls.log(f"Attn Combine Task: num_tiles = {num_combine_tile}, kernel_config = {kernel_config}")
        for i in range(num_combine_tile):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_combine_tile, kernel_config, dependency, io_tensors,
                                 extra_params))
        return tasks
