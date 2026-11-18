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
"""Logical dataflow intrinsic lookup, independent of native short-name registration."""
from __future__ import annotations
from typing import Optional
from little_kernel.core.memory_effects import Completion, Effect
from little_kernel.language.builtin_base import MEMORY_EFFECTS_ATTR, CTA_GROUP_ATTR, get_intrin_ctx


def get_dataflow_intrinsic(name: Optional[str]):
    if not name:
        return None
    from little_kernel.backends.nv.sm100.language.dataflow import intrinsics
    registry = get_intrin_ctx()
    return registry.get(f"dataflow.{name}") or registry.get(name)


def get_intrin_mem_effects(name: Optional[str]) -> tuple[Effect, ...]:
    fn = get_dataflow_intrinsic(name)
    return tuple(getattr(fn, MEMORY_EFFECTS_ATTR, ())) if fn else ()


def intrin_has_completion(name: Optional[str], *completions: Completion) -> bool:
    return any(effect.kind.completion in completions for effect in get_intrin_mem_effects(name))


def get_intrin_cta_group(name: Optional[str]) -> int:
    fn = get_dataflow_intrinsic(name)
    return int(getattr(fn, CTA_GROUP_ATTR, 1)) if fn else 1
