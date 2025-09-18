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
from triton import next_power_of_2
from typing import Tuple, List
import dataclasses
from dataclasses import dataclass
from .utils import build_tile_desc, torch_dtype_to_triton_dtype_str, cdiv
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class QKVPackQKNormRopeSplitVConfig(ConfigBase):
    BLOCK_SEQ: int = 128
    BLOCK_HD: int = 256


@dataclass
class QKNormRopeUpdateKVCacheConfig(ConfigBase):
    pass

@dataclass
class RMSNormConfig(ConfigBase):
    BLOCK_SIZE_N: int = 2048


@dataclass
class QKNormRopeUpdateKVCacheTask(TaskBase):
    config: QKNormRopeUpdateKVCacheConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps/rope_theta as constexpr in codegen
        return ()


@dataclass
class QKVPackQKNormRopeSplitVTask(TaskBase):
    config: QKVPackQKNormRopeSplitVConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps as constexpr in codegen
        return ()


@dataclass
class RMSNormTask(TaskBase):
    config: RMSNormConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # rms_eps as constexpr in codegen
        return ()


def qk_norm_rope_update_kvcache_config_factory(**kwargs) -> QKNormRopeUpdateKVCacheConfig:
    return dataclasses.replace(QKNormRopeUpdateKVCacheConfig(), **kwargs)


def qkv_pack_qk_norm_rope_split_v_config_factory(**kwargs) -> QKVPackQKNormRopeSplitVConfig:
    return dataclasses.replace(QKVPackQKNormRopeSplitVConfig(), **kwargs)


def rms_norm_config_factory(**kwargs) -> RMSNormConfig:
    return dataclasses.replace(RMSNormConfig(), **kwargs)


def codegen_qk_norm_rope_update_kvcache(task: QKNormRopeUpdateKVCacheTask) -> str:
    qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = task.io_tensors[0]
    key_cache, value_cache, q_norm_rope = task.io_tensors[1]
    Q_HEAD_DIM = qkv.shape[-1]
    V_HEAD_DIM = value_cache.shape[-1]
    NUM_KV_HEADS = key_cache.shape[-2]
    NUM_Q_HEADS = qkv.shape[-2] - 2 * NUM_KV_HEADS
    PAGE_SIZE, NUM_KV_HEADS, V_HEAD_DIM = value_cache.shape[-3], value_cache.shape[-2], value_cache.shape[-1]
    MAX_NUM_BLOCKS_PER_SEQ = block_tables.shape[-1]
    code = f"""
rmsnorm_rope_update_kv_cache_task_compute(
    task_base_info, scoreboard, NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, Q_HEAD_DIM={Q_HEAD_DIM},
    V_HEAD_DIM={V_HEAD_DIM}, PAGE_SIZE={PAGE_SIZE}, MAX_NUM_BLOCKS_PER_SEQ={MAX_NUM_BLOCKS_PER_SEQ},
    Q_RMS_EPS={task.extra_params["q_rms_eps"]}, K_RMS_EPS={task.extra_params["k_rms_eps"]}
)
"""
    return code


def codegen_rms_norm(task: RMSNormTask) -> str:
    config: RMSNormConfig = task.config
    code = f"""
rmsnorm_task_compute(task_base_info, scoreboard, RMS_EPS={task.extra_params["rms_eps"]}, BLOCK_SIZE_N = {config.BLOCK_SIZE_N})
"""
    return code


def codegen_qkv_pack_qk_norm_rope_split_v(task: QKVPackQKNormRopeSplitVTask) -> str:
    qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = task.io_tensors[0]
    q_norm_rope, k_norm_rope, v = task.io_tensors[1]
    HEAD_DIM = qkv.shape[-1]
    triton_dtype = torch_dtype_to_triton_dtype_str(qkv.dtype)
    NUM_Q_HEADS = q_norm_rope.shape[-2]
    NUM_KV_HEADS = k_norm_rope.shape[-2]
    code = f"""
qkv_pack_qk_norm_rope_split_v_task_compute(
    task_base_info, scoreboard, DTYPE={triton_dtype}, NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM},
    Q_RMS_EPS={task.extra_params["q_rms_eps"]}, K_RMS_EPS={task.extra_params["k_rms_eps"]},
    BLOCK_SEQ={task.config.BLOCK_SEQ}, BLOCK_HD={task.config.BLOCK_HD},
)
"""
    return code


@registry.register_task(op_type="qk_norm_rope_update_kvcache", task_cls=QKNormRopeUpdateKVCacheTask,
                        config_factory=qk_norm_rope_update_kvcache_config_factory,
                        codegen_func=codegen_qk_norm_rope_update_kvcache)
class QKNormRopeUpdateKVCacheTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = io_tensors[0]
        key_cache, value_cache, q_norm_rope = io_tensors[1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        assert len(qkv.shape) == 4
        batch, seq_len, num_qkv_heads, head_dim = qkv.shape
        num_kv_heads = key_cache.shape[-2]
        num_qk_heads = num_qkv_heads - num_kv_heads
        num_tiles = batch * seq_len * num_qk_heads
        cls.log(f"KNormRopeUpdateKVCache Task: num_tiles = {num_tiles}")
        tasks = []
        for i in range(num_tiles):
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params))
        return tasks


@registry.register_task(op_type="rms_norm", task_cls=RMSNormTask, config_factory=rms_norm_config_factory,
                        codegen_func=codegen_rms_norm)
class RMSNormTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        input, weight = io_tensors[0]
        output = io_tensors[1][0]
        num_tiles = output.numel() // output.shape[-1]
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config()
        cls.log(f"RMS Norm Task: num_tiles = {num_tiles}")
        tasks = []
        tile_size = output.shape[-1]
        for i in range(num_tiles):
            in_start_indices, in_data_sizes = build_tile_desc(input.shape, [1, tile_size], i, return_valid_size=True)
            out_start_indices, out_data_sizes = build_tile_desc(output.shape, [1, tile_size], i)
            input_desc = InputDependencyDesc(input, require_full=False, start_indices=in_start_indices,
                                             data_sizes=in_data_sizes)
            weight_desc = InputDependencyDesc(weight, require_full=True)
            out_desc = OutputTilingDesc(start_indices=out_start_indices, tile_sizes=out_data_sizes)
            inputs_dep = {input: input_desc, weight: weight_desc}
            outs_tile_mapping = {output: out_desc}
            tasks.append(
                cls._create_task(layer_id, task_id, i, num_tiles, kernel_config, dependency, io_tensors, extra_params,
                                 inputs_dep, outs_tile_mapping))
        return tasks


@registry.register_task(op_type="qkv_pack_qk_norm_rope_split_v", task_cls=QKVPackQKNormRopeSplitVTask,
                        config_factory=qkv_pack_qk_norm_rope_split_v_config_factory,
                        codegen_func=codegen_qkv_pack_qk_norm_rope_split_v)
class QKVPackQKNormRopeSplitVTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        qkv, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache = io_tensors[0]
        q_norm_rope, k_norm_rope, v = io_tensors[1]
        assert len(qkv.shape) == 4
        bs, seq_len, num_total_heads, head_dim = qkv.shape
        num_q_heads = q_norm_rope.shape[-2]
        num_kv_heads = k_norm_rope.shape[-2]
        group_size = num_q_heads // num_kv_heads
        BLOCK_HD = next_power_of_2(head_dim)
        task_id = cls.get_task_id(layer_id)
        kernel_config = cls.create_config(BLOCK_HD=BLOCK_HD)
        num_tiles_seq = cdiv(seq_len, kernel_config.BLOCK_SEQ)
        num_tiles = bs * num_tiles_seq * num_total_heads
        cls.log(f"QKVPackQKNormRopeSplitVTask Task: num_tiles = {num_tiles}")
        tasks = []
        BLOCK_SEQ = kernel_config.BLOCK_SEQ
        for tile_id in range(num_tiles):
            head_type = tile_id % (group_size + 2)
            tile_id_bs_seq = tile_id // num_total_heads
            bs_idx = tile_id_bs_seq // num_tiles_seq
            tild_id_seq = tile_id_bs_seq % num_tiles_seq
            head_group_id = (tile_id % num_total_heads) // (group_size + 2)
            if head_type == group_size + 1:  # v
                head_id_out = head_group_id
                head_id_input = head_id_out + num_q_heads + num_kv_heads
            elif head_type < group_size:  # q
                head_id_out = head_group_id * group_size + head_type
                head_id_input = head_id_out
            else:  # k
                head_id_out = head_group_id
                head_id_input = head_id_out + num_q_heads

            seq_tile_size = min(seq_len - tild_id_seq * BLOCK_SEQ, BLOCK_SEQ)
            input_desc = InputDependencyDesc(qkv, require_full=False,
                                             start_indices=(bs_idx, tild_id_seq * BLOCK_SEQ, head_id_input, 0),
                                             data_sizes=(1, seq_tile_size, 1, head_dim))
            out_desc = OutputTilingDesc(start_indices=(bs_idx, tild_id_seq * BLOCK_SEQ, head_id_out, 0),
                                        tile_sizes=(1, BLOCK_SEQ, 1, head_dim))
            empty_out_desc = OutputTilingDesc(start_indices=(0, 0, 0, 0), tile_sizes=(0, 0, 0, 0))
            if head_type < group_size:  # q
                outs_tile_mapping = {q_norm_rope: out_desc, k_norm_rope: empty_out_desc, v: empty_out_desc}
            elif head_type == group_size:
                outs_tile_mapping = {q_norm_rope: empty_out_desc, k_norm_rope: out_desc, v: empty_out_desc}
            else:
                outs_tile_mapping = {q_norm_rope: empty_out_desc, k_norm_rope: empty_out_desc, v: out_desc}
            inputs_dep = {qkv: input_desc}
            tasks.append(
                cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                 extra_params, inputs_dep, outs_tile_mapping))
        return tasks
