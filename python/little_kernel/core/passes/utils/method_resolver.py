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
"""Compatibility shim for the legacy flat ``method_resolver`` module (RF17).

This file used to contain a second, divergent copy of ``MethodResolver`` which
(a) imported the removed ``little_kernel.codegen.special_struct_registry``
location instead of ``little_kernel.codegen.registries.special_struct_registry``
and (b) lacked the struct-stub registry fallback and unannotated tuple-return
inference maintained in the canonical resolver. It had no supported caller;
the active inferencer components (``type_inference_call`` and
``type_inference_ast_visitors``) import the canonical resolver from
:mod:`little_kernel.core.passes.utils.type_inference.method_resolver`.

This shim re-exports that canonical class unchanged (same constructor and
``resolve_method_return_type(var_name, method_name, var_type=None)`` signature)
so legacy imports keep resolving to the one authoritative implementation. It
must not grow its own behavior.
"""

from little_kernel.core.passes.utils.type_inference.method_resolver import MethodResolver

__all__ = ["MethodResolver"]
