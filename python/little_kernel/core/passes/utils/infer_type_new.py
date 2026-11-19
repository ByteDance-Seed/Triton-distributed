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
"""Compatibility shim for the abandoned ``infer_type_new`` module (RF17).

The historical ``infer_type_new.py`` was a second, unfinished rewrite of the
type inferencer. It imported sibling modules (``type_inference_core``,
``type_inference_visitors``, ``resolve_attribute``) that only exist inside the
:mod:`little_kernel.core.passes.utils.type_inference` package, so it raised
``ModuleNotFoundError`` at import time and had no supported caller.

The single maintained inferencer lives in
:mod:`little_kernel.core.passes.utils.type_inference` and is used by the
compiler passes. This module only re-exports it so that the old import path
(``from ...utils.infer_type_new import TypeInferencer, infer_type``) keeps
working. It must not grow its own behavior: fixes belong in the canonical
implementation.
"""

from little_kernel.core.passes.utils.type_inference import TypeInferencer, infer_type

__all__ = ["TypeInferencer", "infer_type"]
