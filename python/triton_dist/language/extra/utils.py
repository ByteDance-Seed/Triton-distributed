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
from typing import Dict, Callable, Any
from functools import wraps
from triton.language import core


class ModuleProxy:

    def __init__(self, module_list: Dict[Callable, Any]):
        active_modules = [module for is_active, module in module_list if is_active()]
        assert len(active_modules) == 1, "only one module can be active"
        self._module = active_modules[0]

    def __getattr__(self, name) -> Any:
        return getattr(self._module, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_module":
            super().__setattr__(name, value)
        else:
            setattr(self._module, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._module, name)

    def dispatch(self, func: Callable) -> Any:

        @core.builtin
        @wraps(func)
        def wrapper(*args, **kwargs):
            method = getattr(self._module, func.__name__)
            return method(*args, **kwargs)

        return wrapper
