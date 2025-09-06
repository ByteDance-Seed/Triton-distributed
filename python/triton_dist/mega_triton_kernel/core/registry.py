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
from typing import Dict, Type, Callable
from .task_base import TaskBase
from .builder import TaskBuilderBase


class Registry:

    def __init__(self):
        self._builder_to_task_cls: Dict[Type['TaskBuilderBase'], Type['TaskBase']] = {}
        self._builders: Dict[Type['TaskBase'], Type['TaskBuilderBase']] = {}  # build task
        self._config_factories: Dict[Type['TaskBase'], Callable] = {}  # kernel config
        self._codegens: Dict[Type['TaskBase'], Callable] = {}  # code generator corresponding to each task
        self._op_mapping: Dict[str, Type['TaskBase']] = {}  # map op to task

    def register_task(self, op_type: str, task_cls: Type['TaskBase'], config_factory: Callable, codegen_func: Callable):

        def decorator(builder_cls: Type['TaskBuilderBase']):

            self._builders[task_cls] = builder_cls
            self._config_factories[task_cls] = config_factory
            self._codegens[task_cls] = codegen_func
            self._op_mapping[op_type] = task_cls

            builder_cls._create_config = config_factory
            builder_cls.TASK_CLS = task_cls

            return builder_cls

        return decorator

    def get_op_mapping(self, op_type: str) -> Type['TaskBase']:
        if op_type not in self._op_mapping:
            raise ValueError(f"Unsupport Op {op_type}")
        return self._op_mapping[op_type]

    def get_builder(self, task_cls: Type['TaskBase']) -> Type['TaskBuilderBase']:
        if task_cls not in self._builders:
            raise ValueError(f"No builder registered for task class {task_cls.__name__}")
        return self._builders[task_cls]

    def get_task_cls(self, builder_cls: Type['TaskBuilderBase']) -> Type['TaskBase']:
        if builder_cls not in self._builder_to_task_cls:
            raise ValueError(f"No task_cls registered for builder class {builder_cls.__name__}")
        return self._builder_to_task_cls[builder_cls]

    def get_config_factory(self, task_cls: Type['TaskBase']) -> Callable:
        if task_cls not in self._config_factories:
            raise ValueError(f"No config factor registered for task class {task_cls.__name__}")
        return self._config_factories[task_cls]

    def get_codegen(self, task_cls: Type['TaskBase']) -> Callable:
        if task_cls not in self._codegens:
            raise ValueError(f"No codegen registered for task class {task_cls.__name__}")
        return self._codegens[task_cls]


registry = Registry()
