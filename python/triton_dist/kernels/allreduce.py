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
from enum import IntEnum


class AllReduceMethod(IntEnum):
    Unknown = 0
    OneShot = 1
    TwoShot = 2  # TODO(houqi.1993) not implemented
    DoubleTree = 3
    OneShot_TMA = 4
    OneShot_Multimem = 5
    TwoShot_Multimem = 6
    TwoShot_Multimem_ST = 7
    OneShot_LL = 8  # TODO(houqi.1993) not implemented
    OneShot_Multimem_LL = 9  # TODO(houqi.1993) not implemented
    AllReduceEnumMax = 10


_ALLREDUCE_METHODS = {
    "double_tree": AllReduceMethod.DoubleTree,
    "one_shot": AllReduceMethod.OneShot,
    "two_shot": AllReduceMethod.TwoShot,
    "one_shot_tma": AllReduceMethod.OneShot_TMA,
    "one_shot_multimem": AllReduceMethod.OneShot_Multimem,  # requires nbytes symmetric buffer
    "two_shot_multimem": AllReduceMethod.TwoShot_Multimem,  # requires
    # deprecated: TwoShot_Multimem_ST use multimem but not fully use multimem instructions.
    "two_shot_multimem_st": AllReduceMethod.TwoShot_Multimem_ST,
}


def to_allreduce_method(method: str) -> AllReduceMethod:
    if method not in _ALLREDUCE_METHODS:
        raise ValueError(f"Invalid method name {method}. Supported methods: {list(_ALLREDUCE_METHODS.keys())}")
    return _ALLREDUCE_METHODS[method]


def get_allreduce_methods():
    return list(_ALLREDUCE_METHODS.keys())
