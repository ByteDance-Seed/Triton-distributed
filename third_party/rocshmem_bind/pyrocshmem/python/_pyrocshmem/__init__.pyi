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
from typing import List, Sequence

import numpy as np
import torch


def rocshmemx_cumodule_init(module: np.intp) -> None:
    ...


def rocshmemx_cumodule_finalize(module: np.intp) -> None:
    ...


def rocshmem_malloc(size: np.uint) -> np.intp:
    ...


def rocshmemx_get_uniqueid() -> bytes:
    ...


def rocshmemx_init_attr_with_uniqueid(rank: np.int32, nranks: np.int32, unique_id: bytes) -> None:
    ...


def rocshmem_int_p(ptr: np.intp, src: np.int32, dst: np.int32) -> None:
    ...


def rocshmem_barrier_all():
    ...


def rocshmem_barrier_all_on_stream():
    ...


# torch related
def rocshmem_create_tensor(shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    ...


def rocshmem_create_tensor_list_intra_node(shape: Sequence[int], dtype: torch.dtype) -> List[torch.Tensor]:
    ...
