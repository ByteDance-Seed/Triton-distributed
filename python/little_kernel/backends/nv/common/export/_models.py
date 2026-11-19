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
"""Compatibility module for the typed export models.

The real typed owners live in :mod:`._records` (authored/resolved immutable
state) and :mod:`._validation` (construction and idempotent validation). This
module re-exports them so earlier internal imports keep resolving to the same
canonical objects.
"""

from __future__ import annotations

from little_kernel.backends.nv.common.export._records import ConstructorArg, ConstexprPlan, ForeachPlan, LaunchPlan, OutputDecl, PipelineStage, Record, ResolvedFacts, ResolvedSpec, ScalarDecl, TensorInput, TensorListDecl, TmaDescriptor, ValidatedSpec, WorkspaceDecl
from little_kernel.backends.nv.common.export._schema import RawSpec
from little_kernel.backends.nv.common.export._validation import from_resolved_view, validate_spec

__all__ = [
    "RawSpec",
    "ValidatedSpec",
    "ResolvedSpec",
    "ResolvedFacts",
    "Record",
    "TensorInput",
    "ConstructorArg",
    "ScalarDecl",
    "OutputDecl",
    "WorkspaceDecl",
    "TensorListDecl",
    "TmaDescriptor",
    "LaunchPlan",
    "ConstexprPlan",
    "ForeachPlan",
    "PipelineStage",
    "from_resolved_view",
    "validate_spec",
]
