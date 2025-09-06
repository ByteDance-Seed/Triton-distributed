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
from typing import Dict, Type, List, Any, Tuple, Union
from dataclasses import dataclass, field
from .config import ConfigBase
import torch
from .utils import has_slice_intersection

# To satisfy the alignment requirement of tensor data_ptr, MAX_NUM_TENSOR_DIMS must be an even number.
MAX_NUM_TENSOR_DIMS = 4

UNUSED_KEY = -1


class CodeGenKey:

    def __init__(self, task_type: int, layer_id: int = UNUSED_KEY, task_id: int = UNUSED_KEY):
        self.task_type = task_type
        self.layer_id = layer_id
        self.task_id = task_id
        self._validate()

    def _validate(self):
        is_layer_invalid = (self.layer_id == UNUSED_KEY)
        is_task_invalid = (self.task_id == UNUSED_KEY)
        if is_layer_invalid != is_task_invalid and self.task_type != UNUSED_KEY:
            raise ValueError(f"illegal codegen key {str(self)}")

    def only_use_task_type(self):
        return self.layer_id == UNUSED_KEY and self.task_id == UNUSED_KEY

    def __eq__(self, other):
        if not isinstance(other, CodeGenKey):
            return False

        if self.task_type == other.task_type:
            if UNUSED_KEY in {self.layer_id, other.layer_id} and self.layer_id != other.layer_id:
                raise ValueError(f"compare with illegal codegen key x = {str(self)}, y = {str(other)}")

        return ((self.task_type == other.task_type) and (self.layer_id == other.layer_id)
                and (self.task_id == other.task_id))

    def __hash__(self):
        return hash((
            self.task_type,
            self.layer_id if self.layer_id != UNUSED_KEY else None,
            self.task_id if self.task_id != UNUSED_KEY else None,
        ))

    def __repr__(self):
        return f"CodeGenKey({self.task_type}, {self.layer_id}, {self.task_id})"


class TaskIDManager:
    _type_id_counter: int = 0
    _type_id_map: Dict[Type, int] = {}
    _task_id_map: Dict[int, int] = {}

    @classmethod
    def get_task_type_id(cls, task_cls: Type) -> int:
        if task_cls not in cls._type_id_map:
            if cls._type_id_counter >= 2**31 - 1:
                raise OverflowError("task_type_id exceeded int32 range")

            cls._type_id_map[task_cls] = cls._type_id_counter
            cls._type_id_counter += 1
        return cls._type_id_map[task_cls]

    @classmethod
    def get_task_id(cls, layer_id: int) -> int:
        # Get a unique task_id for a specific layer_id
        current = cls._task_id_map.get(layer_id, 0)
        if current >= 2**31 - 1:
            raise OverflowError(f"task_id exceeded int32 range for layer {layer_id}")

        cls._task_id_map[layer_id] = current + 1
        return current

    @classmethod
    def reset_task_ids(cls):
        cls._task_id_map.clear()

    @classmethod
    def reset_all_ids(cls):
        cls._type_id_counter = 0
        cls._type_id_map.clear()
        cls._task_id_map.clear()


@dataclass
class TaskDependency:
    layer_id: int
    task_id: int
    start_tiles: int  # include
    end_tiles: int  # exclude

    def __init__(self, layer_id=-1, task_id=-1, start_tiles=0, end_tiles=0):
        self.layer_id = layer_id
        self.task_id = task_id
        self.start_tiles = start_tiles
        self.end_tiles = end_tiles

    def cover(self, other):
        return other.layer_id == self.layer_id and other.task_id == self.task_id and other.start_tiles >= self.start_tiles and other.end_tiles <= self.end_tiles

    def to_empty(self):
        self.start_tiles = 0
        self.end_tiles = 0

    def key(self):
        return (self.layer_id, self.task_id)


@dataclass
class OutputTilingDesc:
    start_indices: Tuple[int]
    tile_sizes: Union[Tuple[int], None]


@dataclass
class InputDependencyDesc:
    input: 'torch.Tensor'
    start_indices: Tuple[int]
    data_sizes: Tuple[int]
    # only require_full == false, start_indices/data_sizes are valid
    require_full: bool = True

    def __init__(self, input, require_full=True, start_indices: Tuple[int] = (), data_sizes: Tuple[int] = ()):
        self.input = input
        self.require_full = require_full
        if require_full:
            self.start_indices = (0, ) * len(input.shape)
            self.data_sizes = input.shape
        else:
            self.start_indices = start_indices
            self.data_sizes = data_sizes


@dataclass
class TaskBase:
    layer_id: int
    task_id: int
    tile_id_or_start: int
    num_tiles: int
    config: ConfigBase  # kernel config (e.g. BLOCK_SIZE/NUM_STAGE)
    dependency: List[TaskDependency]
    io_tensors: List[List['torch.Tensor']]  # inputs and outputs
    extra_params: Dict[str, Any]
    inputs_dep: Dict['torch.Tensor', InputDependencyDesc] = field(default_factory=dict)
    outs_tile_mapping: Dict['torch.Tensor', OutputTilingDesc] = field(default_factory=dict)

    def __str__(self):
        return f"{self.__class__.__name__}(layer_id={self.layer_id}, task_id={self.task_id}, tile_id_or_start={self.tile_id_or_start}, dependency = {self.dependency}, num_tiles = {self.num_tiles}, config = {self.config})"

    def __repr__(self):
        return self.__str__()

    @classmethod
    def get_task_type_id(cls) -> int:
        return TaskIDManager.get_task_type_id(cls)

    @classmethod
    def get_codegen_key(cls, layer_id: int, task_id: int) -> CodeGenKey:
        return CodeGenKey(task_type=cls.get_task_type_id(), layer_id=layer_id, task_id=task_id)

    def get_out_tiling_desc(self, out_idx):
        out_tensor_list = self.io_tensors[1]
        assert len(out_tensor_list) > out_idx
        out_tensor = out_tensor_list[out_idx]
        return self.outs_tile_mapping.get(out_tensor, None)

    def get_input_dep_desc(self, input_idx):
        input_tensor_list = self.io_tensors[0]
        assert len(input_tensor_list) > input_idx
        input_tensor = input_tensor_list[input_idx]
        return self.inputs_dep.get(input_tensor, None)

    def io_to_tuple(self):
        io_tuple = tuple()
        assert len(self.io_tensors) == 2
        all_tensors = self.io_tensors[0] + self.io_tensors[1]
        for tensor in all_tensors:
            data_ptr = tensor.data_ptr()
            ptr_high = (data_ptr >> 32) & 0xFFFFFFFF
            ptr_low = data_ptr & 0xFFFFFFFF

            shape = list(tensor.shape)
            assert MAX_NUM_TENSOR_DIMS >= len(shape)
            padded_shape = shape + [1] * (MAX_NUM_TENSOR_DIMS - len(shape))

            tensor_tuple = (ptr_low, ptr_high) + tuple(padded_shape)
            assert len(tensor_tuple) % 2 == 0, "tensor data_ptr alignemnt"
            io_tuple += tensor_tuple
        return io_tuple

    def extra_params_to_tuple(self) -> Tuple[int]:
        assert len(self.extra_params) == 0
        return ()

    def encoding_with_deps(self, l, r) -> Tuple[int]:
        """
        task_type | layer_id | task_id | tile_id_or_start | dependency(l, r) | io_tensors | extra_params
        """
        entrys = []
        entrys.append(self.get_task_type_id())
        entrys.append(self.layer_id)
        entrys.append(self.task_id)
        entrys.append(self.tile_id_or_start)
        entrys += (l, r)
        assert len(entrys) % 2 == 0, "tensor data_ptr alignemnt"
        entrys += self.io_to_tuple()
        entrys += self.extra_params_to_tuple()
        for x in entrys:
            if not isinstance(x, (int, )):
                raise ValueError(f"got unexpected value {x}")

        return tuple(entrys)

    def has_data_dependency(self, producer_task, producer_out_idx, consumer_input_idx):
        input_dep_desc = self.get_input_dep_desc(consumer_input_idx)
        out_tiling_desc = producer_task.get_out_tiling_desc(producer_out_idx)
        if input_dep_desc is None or out_tiling_desc is None:
            return True
        if input_dep_desc.require_full:
            return True
        return has_slice_intersection(
            start_indices1=input_dep_desc.start_indices,
            data_sizes1=input_dep_desc.data_sizes,
            shape1=input_dep_desc.input.shape,
            start_indices2=out_tiling_desc.start_indices,
            data_sizes2=out_tiling_desc.tile_sizes,
            shape2=producer_task.io_tensors[1][producer_out_idx].shape,
        )


@dataclass
class DeviceProp:
    NUM_SMS: int
