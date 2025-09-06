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
from .utils import cdiv, torch_dtype_to_triton_dtype_str
import dataclasses
from dataclasses import dataclass
from ..core.task_base import TaskBase, TaskDependency, InputDependencyDesc, OutputTilingDesc
from ..core.builder import TaskBuilderBase
from ..core.registry import registry
from ..core.config import ConfigBase


@dataclass
class FlashAttnConfig(ConfigBase):
    BLOCK_M: int = 128
    BLOCK_N: int = 128
    NUM_STAGES: int = 3


@dataclass
class QKVPackFlashAttnConfig(FlashAttnConfig):
    pass


@dataclass
class FlashAttnTask(TaskBase):
    config: QKVPackFlashAttnConfig

    def extra_params_to_tuple(self) -> Tuple[int]:
        # sm_scale/soft_cap as constexpr in codegen
        return ()


@dataclass
class QKVPackFlashAttnTask(FlashAttnTask):
    config: QKVPackFlashAttnConfig


def flash_attn_config_factory(**kwargs) -> FlashAttnConfig:
    return dataclasses.replace(FlashAttnConfig(), **kwargs)


def qkv_pack_flash_attn_config_factory(**kwargs) -> QKVPackFlashAttnConfig:
    return dataclasses.replace(QKVPackFlashAttnConfig(), **kwargs)


def _codegen_flash_attn_impl(task: FlashAttnTask, NUM_KV_HEADS, qkv_pack=False) -> str:
    config: QKVPackFlashAttnConfig = task.config
    out = task.io_tensors[1][0]

    HEAD_DIM = out.shape[-1]
    NUM_Q_HEADS = out.shape[-2]

    INPUT_DTYPE = torch_dtype_to_triton_dtype_str(task.io_tensors[0][0].dtype)
    OUTPUT_DTYPE = torch_dtype_to_triton_dtype_str(out.dtype)

    fn_name = "qkv_pack_flash_attn_task_compute" if qkv_pack else "flash_attn_task_compute"
    code = f"""
{fn_name}(
    task_base_info, scoreboard, SM_SCALE={task.extra_params["sm_scale"]}, SOFT_CAP={task.extra_params["soft_cap"]},
    INPUT_DTYPE={INPUT_DTYPE}, OUTPUT_DTYPE={OUTPUT_DTYPE},
    NUM_Q_HEADS={NUM_Q_HEADS}, NUM_KV_HEADS={NUM_KV_HEADS}, HEAD_DIM={HEAD_DIM},
    BLOCK_M={config.BLOCK_M}, BLOCK_N={config.BLOCK_N}, NUM_STAGES={config.NUM_STAGES},
    IS_CAUSAL={task.extra_params["is_causal"]}
)
"""
    return code


def codegen_flash_attn(task: FlashAttnTask):
    q, k, v = task.io_tensors[0]
    NUM_KV_HEADS = k.shape[-2]
    return _codegen_flash_attn_impl(task, NUM_KV_HEADS, qkv_pack=False)


def codegen_qkv_flash_attn(task: FlashAttnTask):
    out = task.io_tensors[1][0]
    qkv = task.io_tensors[0][0]
    NUM_Q_HEADS = out.shape[-2]
    NUM_TOTAL_HEADS = qkv.shape[-2]
    NUM_KV_HEADS = (NUM_TOTAL_HEADS - NUM_Q_HEADS) // 2

    return _codegen_flash_attn_impl(task, NUM_KV_HEADS, qkv_pack=True)


@registry.register_task(op_type="qkv_pack_flash_attn", task_cls=QKVPackFlashAttnTask,
                        config_factory=qkv_pack_flash_attn_config_factory, codegen_func=codegen_qkv_flash_attn)
class QKVPackFlashAttnTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        qkv = io_tensors[0][0]
        out = io_tensors[1][0]

        assert len(qkv.shape) == 4, f"qkv shape mismatch, expect (bs, seq, nheads, head_dim), but got {qkv.shape}"
        assert "is_causal" in extra_params.keys()
        assert "sm_scale" in extra_params.keys()
        assert "soft_cap" in extra_params.keys()

        batch, seq, num_q_heads, head_dim = out.shape
        num_kv_heads = (qkv.shape[-2] - num_q_heads) // 2
        assert num_q_heads % num_kv_heads == 0
        group_size = num_q_heads // num_kv_heads
        kernel_config = cls.create_config()
        task_id = cls.get_task_id(layer_id)

        BLOCK_M = kernel_config.BLOCK_M
        num_tiles = batch * num_q_heads * cdiv(seq, BLOCK_M)
        tasks = []
        cls.log(f"QKV Pack FlashAttn Task: num_tiles = {num_tiles}, kernel_config = {kernel_config}")
        for tile_id in range(num_tiles):
            num_tile_m = cdiv(seq, BLOCK_M)
            tile_id_m = tile_id % num_tile_m

            off_hz = tile_id // num_tile_m
            off_z = off_hz // num_q_heads
            off_hq = off_hz % num_q_heads
            off_hkv = off_hq // group_size

            qkv_desc = InputDependencyDesc(
                qkv, require_full=False, start_indices=(off_z, tile_id_m * BLOCK_M, off_hq, 0),
                data_sizes=(1, min(BLOCK_M,
                                   seq - tile_id_m * BLOCK_M), num_q_heads + num_kv_heads + off_hkv - off_hq, head_dim))
            out_desc = OutputTilingDesc(tile_sizes=(BLOCK_M, head_dim),
                                        start_indices=(off_z, tile_id_m * BLOCK_M, off_hq, 0))
            inputs_dep = {qkv: qkv_desc}
            outs_tile_mapping = {out: out_desc}

            tasks.append(
                cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                 extra_params, inputs_dep, outs_tile_mapping))
        return tasks


@registry.register_task(op_type="flash_attn", task_cls=FlashAttnTask, config_factory=flash_attn_config_factory,
                        codegen_func=codegen_flash_attn)
class FlashAttnTaskBuilder(TaskBuilderBase):

    @classmethod
    def _build_tasks_impl(cls, device_prop, layer_id: int, dependency: TaskDependency, io_tensors, extra_params,
                          tile_wise=True) -> List[TaskBase]:
        q, k, v = io_tensors[0]
        out = io_tensors[1][0]

        assert len(q.shape) == 4, f"qkv shape mismatch, expect (bs, seq, nheads, head_dim), but got {q.shape}"
        assert "is_causal" in extra_params.keys()
        assert "sm_scale" in extra_params.keys()
        assert "soft_cap" in extra_params.keys()

        batch, seq, num_q_heads, head_dim = out.shape
        num_kv_heads = k.shape[-2]
        assert num_q_heads % num_kv_heads == 0
        group_size = num_q_heads // num_kv_heads
        kernel_config = cls.create_config()
        task_id = cls.get_task_id(layer_id)

        BLOCK_M = kernel_config.BLOCK_M
        num_tiles = batch * num_q_heads * cdiv(seq, BLOCK_M)
        tasks = []
        cls.log(f"FlashAttn Task: num_tiles = {num_tiles}, kernel_config = {kernel_config}")
        for tile_id in range(num_tiles):
            num_tile_m = cdiv(seq, BLOCK_M)
            tile_id_m = tile_id % num_tile_m

            off_hz = tile_id // num_tile_m
            off_z = off_hz // num_q_heads
            off_hq = off_hz % num_q_heads
            off_hkv = off_hq // group_size

            q_desc = InputDependencyDesc(q, require_full=False, start_indices=(off_z, tile_id_m * BLOCK_M, off_hq, 0),
                                         data_sizes=(1, min(BLOCK_M, seq - tile_id_m * BLOCK_M), 1, head_dim))
            k_desc = InputDependencyDesc(k, require_full=False, start_indices=(off_z, 0, off_hkv, 0),
                                         data_sizes=(1, min(seq, (tile_id_m + 1) * BLOCK_M), 1, head_dim))
            v_desc = InputDependencyDesc(v, require_full=False, start_indices=(off_z, 0, off_hkv, 0),
                                         data_sizes=(1, min(seq, (tile_id_m + 1) * BLOCK_M), 1, head_dim))

            out_desc = OutputTilingDesc(tile_sizes=(BLOCK_M, head_dim),
                                        start_indices=(off_z, tile_id_m * BLOCK_M, off_hq, 0))
            inputs_dep = {q: q_desc, k: k_desc, v: v_desc}
            outs_tile_mapping = {out: out_desc}

            tasks.append(
                cls._create_task(layer_id, task_id, tile_id, num_tiles, kernel_config, dependency, io_tensors,
                                 extra_params, inputs_dep, outs_tile_mapping))
        return tasks
