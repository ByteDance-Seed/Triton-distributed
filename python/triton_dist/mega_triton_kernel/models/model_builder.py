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
import importlib
import tempfile
import os
import nvshmem
import nvshmem.core

from ..core.code_generator import CodeGenerator
from ..core.registry import registry
from ..core.task_base import TaskBase, DeviceProp, TaskDependency, TaskIDManager, MAX_NUM_TENSOR_DIMS
from ..core.builder import TaskBuilderBase
from typing import List, Dict, Any
from triton_dist.utils import nvshmem_create_tensor, nvshmem_free_tensor_sync
from ..core.scheduler import enque_tasks
from triton_dist.models.utils import logger


def is_multicast_ptr(tensor):
    """
    On unsupported platforms, `mc_ptr` return nullptr
    """
    return nvshmem.bindings.mc_ptr(nvshmem.core.Teams.TEAM_NODE, tensor.data_ptr()) != 0


def check_tensor_shape(tensor, shape):
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(shape, (list, tuple))
    assert list(tensor.shape()) == shape, f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"
    tensor_shape = list(tensor.shape())
    assert len(tensor_shape) == len(shape), f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"
    for (x, y) in zip(tensor_shape, shape):
        assert x == y, f"tensor shape mismatch, tensor shape = {tensor.shape}, shape = {shape}"


def check_tensor_dim(tensor, ndim):
    assert isinstance(tensor, torch.Tensor)
    assert len(tensor.shape) == ndim, f"tensor dim mismatch, tensor dim = {len(tensor.shape)}, ndim = {ndim}"


def check_tensor_dtype(tensor, dtype):
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == dtype, f"tensor dtype mismatch, expect {dtype}, but got {tensor.dtype}"


def check_contiguous(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    for t in tensors:
        assert t.is_contiguous()


def check_alignment(tensors):
    if not isinstance(tensors, (tuple, list)):
        tensors = [tensors]
    for t in tensors:
        assert t.data_ptr() % 16 == 0, f"data_ptr = {t.data_ptr()}"


class ModelBuilder:

    def __init__(self, rank=0, world_size=1, local_world_size=1):
        self.reset()
        self._registry = registry
        self._code_generator = CodeGenerator()
        self._max_tensor_dim = MAX_NUM_TENSOR_DIMS
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        self.device_prop = DeviceProp(NUM_SMS=NUM_SMS)
        self.megakernel_tasks = []
        self.scoreboard = None
        self.wq_tensor = None  # work queue
        self.num_task_tensor = None  # num task in each work queue
        self.MAX_NUM_TILES_PER_OP = 1
        self.last_dependency = TaskDependency()
        self.max_layer_id = 0
        self.max_task_id = 0
        self.num_warps = 4
        self._metrics = {"memory": 0}
        self.world_size = world_size
        self.local_world_size = local_world_size
        self.rank = rank
        self.local_rank = self.rank % self.local_world_size
        assert self.world_size % self.local_world_size == 0
        assert self.world_size > 0 and self.local_world_size > 0
        self.all_symm_tensors = []
        if self.world_size > 1:
            self.barrier_all_intra_node_buf = self.create_symm_tensor([
                world_size,
            ], torch.int32)
            self.barrier_all_intra_node_buf.zero_()
            torch.distributed.barrier()
        else:
            self.barrier_all_intra_node_buf = None
        self.logger = logger

    def create_symm_tensor(self, shape, dtype) -> torch.Tensor:
        tensor = nvshmem_create_tensor(shape, dtype)
        self.all_symm_tensors.append(tensor)
        return tensor

    def _update_metrics(self, op_type: str, io_tensors: List[List[torch.Tensor]], extra_params: Dict[str, Any] = {}):
        # avoid kv cache tensor being counted
        if "attn" in op_type or "kvcache" in op_type:
            return

        nbytes = 0
        all_tensors = io_tensors[0] + io_tensors[1]
        for ten in all_tensors:
            nbytes += ten.numel() * ten.element_size()

        if "memory" not in self._metrics:
            self._metrics["memory"] = 0
        self._metrics["memory"] += nbytes

    def get_memory_size(self):
        return self._metrics["memory"]

    def _update_tasks(self, tasks: List[TaskBase], do_not_update_dependency=False):
        assert len(tasks) > 0
        last_task = tasks[-1]
        if not do_not_update_dependency:
            self.last_dependency = TaskDependency(layer_id=last_task.layer_id, task_id=last_task.task_id, start_tiles=0,
                                                  end_tiles=last_task.num_tiles)
        self.megakernel_tasks += tasks
        for task in tasks:
            self.MAX_NUM_TILES_PER_OP = max(self.MAX_NUM_TILES_PER_OP, task.num_tiles)
            self.max_layer_id = max(task.layer_id, self.max_layer_id)
            self.max_task_id = max(task.task_id, self.max_task_id)

    def get_task_builder(self, op_type: str) -> 'TaskBuilderBase':
        task_type = self._registry.get_op_mapping(op_type)
        if not task_type:
            raise ValueError(f"Unsupported op type: {op_type}")
        builder_cls = registry.get_builder(task_type)
        return builder_cls

    def _convert_op(self, op_type: str, layer_id: int, io_tensors: List[List[torch.Tensor]],
                    extra_params: Dict[str, Any] = {}) -> List[TaskBase]:
        assert len(io_tensors) == 2
        check_contiguous(io_tensors[0])
        check_contiguous(io_tensors[1])
        check_alignment(io_tensors[0])
        check_alignment(io_tensors[1])

        builder_cls = self.get_task_builder(op_type)
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=self.last_dependency, io_tensors=io_tensors,
                                        extra_params=extra_params)
        self._update_tasks(tasks)
        self._update_metrics(op_type, io_tensors, extra_params)
        return tasks

    def _make_fc(self, op_type: str, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor,
                 layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        assert K % 32 == 0

        self._convert_op(op_type, layer_id, [[input, weight], [output]])

    def make_fc1(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc1", input, weight, output, layer_id)

    def make_fc2(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("mlp_fc2", input, weight, output, layer_id)

    def make_qkv_proj(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("qkv_proj", input, weight, output, layer_id)

    def make_o_proj(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        self._make_fc("o_proj", input, weight, output, layer_id)

    def make_linear(self, input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor, layer_id: int = 0):
        check_tensor_dim(input, 2)
        check_tensor_dim(weight, 2)
        M, K = input.shape
        N, wK = weight.shape
        oM, oN = output.shape
        assert K == wK
        assert oM == M and oN == N
        self._convert_op("linear", layer_id, [[input, weight], [output]])

    def make_attn(self, query, key_cache: torch.Tensor, value_cache: torch.Tensor, block_tables: torch.Tensor,
                  kv_lens: torch.Tensor, output: torch.Tensor, sm_scale=None, soft_cap=0.0, layer_id: int = 0):
        """
            query: (batch, seq_len, num_q_heads, q_head_dim)
            key_cache: (MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, q_head_dim)
            value_cache: (MAX_NUM_KV_BLOCKS, PAGE_SIZE, num_kv_heads, v_head_dim)
            block_tables: (batch, MAX_NUM_BLOCKS_PER_SEQ)
            kv_lens: (batch)
            output: (batch, seq_len, num_q_heads, v_head_dim)
        """
        check_tensor_dim(query, 4)
        check_tensor_dim(key_cache, 4)
        check_tensor_dim(value_cache, 4)
        check_tensor_dim(block_tables, 2)
        check_tensor_dim(kv_lens, 1)
        check_tensor_dim(output, 4)
        check_tensor_dtype(query, torch.bfloat16)
        check_tensor_dtype(key_cache, torch.bfloat16)
        check_tensor_dtype(value_cache, torch.bfloat16)
        check_tensor_dtype(block_tables, torch.int32)
        check_tensor_dtype(kv_lens, torch.int32)
        check_tensor_dtype(output, torch.bfloat16)
        assert query.shape[0] == kv_lens.shape[0]

        batch, q_seq_len, num_q_heads, q_head_dim = query.shape
        v_head_dim = value_cache.shape[-1]
        assert q_seq_len == 1, "currently flash decoding only support q_seq_len == 1"
        if sm_scale is None:
            sm_scale = q_head_dim**-0.5
        soft_cap = 0.0
        NUM_KV_SPLITS = 32
        extra_params = {"sm_scale": sm_scale, "soft_cap": soft_cap, "NUM_KV_SPLITS": NUM_KV_SPLITS}
        # Assuming q_seq_len is 1, ignore the seq_len dimension to reduce the number of tensor dimensions (MAX_TENSOR_DIM = 4)
        partial_out = torch.empty([batch, num_q_heads, NUM_KV_SPLITS, v_head_dim], dtype=torch.float32,
                                  device=query.device)
        lse = torch.empty([batch, num_q_heads, NUM_KV_SPLITS], dtype=torch.float32, device=query.device)
        self._convert_op("attn_split", layer_id,
                         [[query, key_cache, value_cache, block_tables, kv_lens], [partial_out, lse]], extra_params)
        self._convert_op("attn_combine", layer_id, [[kv_lens, partial_out, lse], [output]], extra_params)

    def make_silu_mul_up(self, fc1_out, act_out, layer_id=0):
        check_tensor_dim(fc1_out, 2)
        check_tensor_dim(act_out, 2)
        M, N = fc1_out.shape
        assert act_out.shape[0] == M
        assert act_out.shape[1] * 2 == N
        self._convert_op("silu_mul_up", layer_id, [[fc1_out], [act_out]])

    def make_qk_norm_rope_update_kvcache(self, qkv: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor,
                                         block_tables: torch.Tensor, kv_lens: torch.Tensor, q_rms_weight: torch.Tensor,
                                         k_rms_weight: torch.Tensor, cos_cache: torch.Tensor, sin_cache: torch.Tensor,
                                         q_norm_rope: torch.Tensor, q_rms_eps: float = 1e-6, k_rms_eps: float = 1e-6,
                                         rope_theta: int = 1000000, layer_id=0):
        """
            this op assume that kv_lens has been update (kv_lens = history_kv_len + seq_len(qkv.shape[1]))
            inplace update new kv to key_cache/value_cache
            cos_cache/sin_cache: [batch, seq_len, head_dim]
            rms_weight: [head_dim]
        """
        check_tensor_dim(qkv, 4)
        check_tensor_dim(key_cache, 4)
        check_tensor_dim(value_cache, 4)
        check_tensor_dim(block_tables, 2)
        check_tensor_dim(kv_lens, 1)
        check_tensor_dim(q_norm_rope, 4)
        check_tensor_dim(cos_cache, 3)
        check_tensor_dim(sin_cache, 3)
        check_tensor_dim(q_rms_weight, 1)
        check_tensor_dim(k_rms_weight, 1)
        check_tensor_dtype(qkv, torch.bfloat16)
        check_tensor_dtype(key_cache, torch.bfloat16)
        check_tensor_dtype(value_cache, torch.bfloat16)
        check_tensor_dtype(block_tables, torch.int32)
        check_tensor_dtype(kv_lens, torch.int32)
        check_tensor_dtype(q_norm_rope, torch.bfloat16)
        check_tensor_dtype(cos_cache, torch.float32)
        check_tensor_dtype(sin_cache, torch.float32)
        check_tensor_dtype(q_rms_weight, torch.bfloat16)
        check_tensor_dtype(k_rms_weight, torch.bfloat16)

        assert qkv.shape[0] == kv_lens.shape[0]
        assert cos_cache.shape[0] == sin_cache.shape[0]
        assert cos_cache.shape[0] == qkv.shape[0] or cos_cache.shape[0] == 1
        extra_params = {"q_rms_eps": q_rms_eps, "k_rms_eps": k_rms_eps, "rope_theta": rope_theta}
        self._convert_op("qk_norm_rope_update_kvcache", layer_id,
                         [[qkv, block_tables, kv_lens, q_rms_weight, k_rms_weight, cos_cache, sin_cache],
                          [key_cache, value_cache, q_norm_rope]], extra_params)

    def make_rms_norm(self, input: torch.Tensor, rms_weight: torch.Tensor, output: torch.Tensor, rms_eps: float = 1e-6,
                      layer_id=0):
        check_tensor_dtype(input, torch.bfloat16)
        check_tensor_dtype(rms_weight, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)
        check_tensor_dim(rms_weight, 1)
        # reshape to 2d tensor
        input = input.reshape(-1, input.shape[-1])
        output = output.reshape(-1, input.shape[-1])

        assert input.shape == output.shape
        assert input.shape[-1] == rms_weight.shape[0]
        extra_params = {"rms_eps": rms_eps}
        self._convert_op("rms_norm", layer_id, [[input, rms_weight], [output]], extra_params)

    def make_add(self, lhs: torch.Tensor, rhs: torch.Tensor, output: torch.Tensor, layer_id=0):
        check_tensor_dtype(lhs, torch.bfloat16)
        check_tensor_dtype(rhs, torch.bfloat16)
        check_tensor_dtype(output, torch.bfloat16)

        assert lhs.shape == rhs.shape and lhs.shape == output.shape
        lhs = lhs.reshape(-1)
        rhs = rhs.reshape(-1)
        output = output.reshape(-1)

        self._convert_op("add", layer_id, [[lhs, rhs], [output]])

    def make_barrier_all_intra_node(self, layer_id=0):
        assert self.world_size > 1
        extra_params = {"local_rank": self.local_rank, "local_world_size": self.local_world_size}
        self._convert_op("barrier_all_intra_node", layer_id, [[self.barrier_all_intra_node_buf], []], extra_params)

    def make_allreduce(self, input: torch.Tensor, output: torch.Tensor, double_input_buffer=False, layer_id=0):
        """
            if double_input_buffer is True, user needs to ensure that the input of two consecutive allreduce are completely different buffers,
            otherwise, the output may be wrong.
        """
        assert self.world_size > 1
        if not is_multicast_ptr(input):
            raise ValueError(
                "The input tensor needs to be a symmetric buffer and AR is only supported in Hopper and later GPU architectures"
            )
        assert os.getenv("NVSHMEM_DISABLE_CUDA_VMM", "1") == "0"  # for multicast
        input = input.reshape(-1)
        output = output.reshape(-1)
        nbytes = input.numel() * input.element_size()
        assert nbytes % 128 == 0
        assert input.shape == output.shape and input.dtype == output.dtype
        self.make_barrier_all_intra_node(layer_id)
        self._convert_op("allreduce", layer_id, [[input], [output]])
        if not double_input_buffer:
            self.make_barrier_all_intra_node(layer_id)

    def make_prefetch(self, weight: torch.tensor, layer_id=0):
        io_tensors = [[weight], []]
        check_contiguous(io_tensors[0])
        check_contiguous(io_tensors[1])
        check_alignment(io_tensors[0])
        check_alignment(io_tensors[1])
        assert weight.dtype == torch.bfloat16
        check_tensor_dim(weight, 2)
        check_tensor_dtype(weight, torch.bfloat16)
        assert weight.shape[1] % 32 == 0

        builder_cls = self.get_task_builder("prefetch")
        tasks = builder_cls.build_tasks(device_prop=self.device_prop, layer_id=layer_id,
                                        dependency=TaskDependency(),  # no dependency
                                        io_tensors=io_tensors, extra_params={})
        self._update_tasks(tasks, do_not_update_dependency=True)

    def reset(self):
        TaskIDManager.reset_all_ids()

    def compile(self):
        self.logger.log(f"num_total_tasks = {len(self.megakernel_tasks)}", level="debug")
        self.wq_tensor, self.num_tasks_tensor = enque_tasks(self.device_prop.NUM_SMS, self.megakernel_tasks,
                                                            "round_robin")
        self.scoreboard = torch.zeros((self.max_layer_id + 1, self.max_task_id + 1, self.MAX_NUM_TILES_PER_OP),
                                      dtype=torch.int32, device=torch.cuda.current_device())

        src = self._code_generator.generate_code(self.megakernel_tasks)
        self.logger.log(src, level="debug")
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
            tmp.write(src.encode('utf-8'))
            tmp_path = tmp.name

        module_name = os.path.basename(tmp_path)[:-3]
        spec = importlib.util.spec_from_file_location(module_name, tmp_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._gen_kernel = module.MEGA_TRITON_KERNEL

    def run(self):
        grid = lambda META: (self.device_prop.NUM_SMS, )
        self._gen_kernel[grid](
            self.wq_tensor,
            self.num_tasks_tensor,
            self.scoreboard,
            INT_PER_TASK=self.wq_tensor.shape[2],
            MAX_TASK_ID=self.scoreboard.shape[1],
            MAX_NUM_TILES_PER_OP=self.MAX_NUM_TILES_PER_OP,
            MAX_NUM_TENSOR_DIMS=self._max_tensor_dim,
            NUM_SMS=self.device_prop.NUM_SMS,
            num_warps=self.num_warps,
        )
        self.scoreboard.zero_()

    def finalize(self):
        for ten in self.all_symm_tensors:
            nvshmem_free_tensor_sync(ten)
