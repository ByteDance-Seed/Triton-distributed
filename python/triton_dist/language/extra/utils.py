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
import inspect
from triton.language import core


def _pointer_type_hash(self):
    return hash((self.name, self.element_ty, "tt_ptr"))


def patch_hash_method_for_pointer_type():
    # Triton pointer_type defines __eq__ but not __hash__, so dict keys fail.
    elem_dtype_list = (core.dtype.SINT_TYPES + core.dtype.UINT_TYPES + core.dtype.FP_TYPES + core.dtype.OTHER_TYPES)
    for elem_dtype in elem_dtype_list:
        ptr_ty = type(core.pointer_type(core.dtype(elem_dtype)))
        ptr_ty.__hash__ = _pointer_type_hash


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
        # Triton's code generator injects `_semantic` / `_generator` into a builtin
        # only when those names appear in the builtin's *signature* (it reads
        # `fn.signature`, which `core.builtin` populates from `inspect.signature`).
        # `functools.wraps(func)` would copy the dispatch stub's signature (no
        # `_semantic`) onto the wrapper, so the injection would be skipped and the
        # real backend op (an `@core.extern`) would be called without `_semantic`,
        # raising "Did you forget to add @triton.jit ?". We therefore expose
        # `_semantic`/`_generator` explicitly on the wrapper (and avoid `wraps`, which
        # would re-hide them via `__wrapped__`), then forward each to the resolved
        # backend method only when it actually accepts it.
        name = func.__name__

        @core.builtin
        def wrapper(*args, _semantic=None, _generator=None, **kwargs):
            method = getattr(self._module, name)
            params = inspect.signature(method).parameters
            if "_semantic" in params:
                kwargs["_semantic"] = _semantic
            if "_generator" in params:
                kwargs["_generator"] = _generator
            return method(*args, **kwargs)

        wrapper.__name__ = name
        wrapper.__qualname__ = getattr(func, "__qualname__", name)
        wrapper.__doc__ = func.__doc__
        return wrapper
