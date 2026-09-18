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
# yapf: disable
# forward import torch to load libtorch_cpu.so and libtorch_cuda.so
import torch  # noqa: F401
# Register the out-of-tree distributed plugin on TRITON_PLUGIN_PATHS *before*
# Triton is imported, so libtriton dlopens it during its (one-time) init.
from ._plugin import register_plugin as _register_plugin  # noqa: E402

_register_plugin()
import triton  # noqa: F401,E402
from packaging.version import Version  # noqa: E402
# yapf: enable

from . import language  # noqa: F401, E402
from .jit import jit  # noqa: F401, E402
from .tools.monkey_inductor import apply_triton340_inductor_patch  # noqa: F401, E402
from .tools.monkey_inductor import TORCH_VERSION  # noqa: E402

triton_version = Version(triton.__version__)
require_patched_torch = (Version("2.7.0a0") <= TORCH_VERSION < Version("2.8.1"))

if require_patched_torch and triton_version == Version("3.4.0"):
    apply_triton340_inductor_patch()

__all__ = ["language", "jit"]
