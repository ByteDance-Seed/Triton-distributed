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
"""Per-emission target policy for portable builtin callbacks."""
from contextlib import contextmanager
from contextvars import ContextVar

_emitter = ContextVar("little_kernel_emitter", default=None)


@contextmanager
def emission_scope(emitter):
    token = _emitter.set(emitter)
    try:
        yield
    finally:
        _emitter.reset(token)


def cpp_type(dtype):
    emitter = _emitter.get()
    if emitter is not None:
        return emitter._lltype_to_cpp(dtype)
    from .registries.type_converter import TypeConverter
    return TypeConverter({}).lltype_to_cpp(dtype)


def current_emitter():
    return _emitter.get()
