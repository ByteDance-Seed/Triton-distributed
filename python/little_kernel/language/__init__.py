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
"""Shared language types and lazily resolved backend extensions."""
from importlib import import_module
from little_kernel.core.type_system import *
from little_kernel.core.type_system import __all__ as _type_exports
from little_kernel.language.builtin_base import builtin, Builtin
from little_kernel.language.simple_builtin import simple_builtin, asm
from little_kernel.backends.registry import language_symbols, resolve_language_symbol, load_symbol

_COMMON_SYMBOLS = {
    'bitcast': 'little_kernel.language.intrin.dtype:bitcast', 'll_type_to_torch_type':
    'little_kernel.language.intrin.dtype:ll_type_to_torch_type', 'neg_infinity':
    'little_kernel.language.intrin.dtype:neg_infinity', 'pos_infinity':
    'little_kernel.language.intrin.dtype:pos_infinity', 'ptr_cast': 'little_kernel.language.intrin.dtype:ptr_cast',
    'sizeof': 'little_kernel.language.intrin.dtype:sizeof', 'tensor_view':
    'little_kernel.language.intrin.dtype:tensor_view', 'to': 'little_kernel.language.intrin.dtype:to', 'typeof':
    'little_kernel.language.intrin.dtype:typeof', 'val_cast': 'little_kernel.language.intrin.dtype:val_cast', 'unroll':
    'little_kernel.language.intrin.loop:unroll', 'no_unroll': 'little_kernel.language.intrin.loop:no_unroll',
    'unroll_dyn': 'little_kernel.language.intrin.loop:unroll_dyn', 'cdiv':
    'little_kernel.language.intrin.portable:cdiv', 'align_power_of_2':
    'little_kernel.language.intrin.portable:align_power_of_2', 'min_val':
    'little_kernel.language.intrin.portable:min_val', 'max_val': 'little_kernel.language.intrin.portable:max_val',
    'select': 'little_kernel.language.intrin.portable:select', 'extract_field4':
    'little_kernel.language.intrin.nibble:extract_field4', 'insert_field4':
    'little_kernel.language.intrin.nibble:insert_field4', 'pack_field4_pair':
    'little_kernel.language.intrin.nibble:pack_field4_pair'
}
COMMON_EXPORTS = frozenset(_type_exports) | frozenset(_COMMON_SYMBOLS) | {"builtin", "Builtin", "simple_builtin", "asm"}
__all__ = sorted(COMMON_EXPORTS | set(language_symbols()))


def __getattr__(name):
    if name in _COMMON_SYMBOLS:
        value = load_symbol(_COMMON_SYMBOLS[name])
    else:
        value = resolve_language_symbol(name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
