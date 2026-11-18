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
"""Typed lowered AST/IR operations (RF06).

This package defines the explicit, typed nodes that lowering passes use when
they need to speak backend (CUDA C++) directly:

* :class:`TargetExpression` -- an explicitly typed backend expression escape
  hatch (replaces raw strings hidden on ``ast.Name``);
* :class:`TargetStatement` -- a backend statement emitted into the kernel body;
* :class:`TargetInclude` -- a backend ``#include`` directive;
* :class:`TargetHelper` -- a backend helper/device-function definition;
* :class:`ResourceReference` -- a reference to a lowered memory/register
  resource with declared address space and pointer/value type;
* :class:`StageReference` -- a pipeline-stage-indexed view over a resource.

Every node carries a declared :class:`~little_kernel.core.ir.spans.SourceSpan`
and declared effects, and every expression node carries a declared
:class:`~little_kernel.core.type_system.LLType` result. Nothing here defaults an
unvalidated string to ``int32`` or emits it as an ordinary identifier.

Transitional read compatibility for the legacy ``_lk_cpp_*`` attributes lives
in :mod:`little_kernel.core.ir.legacy` and is used only until the dataflow
worker migrates all producers. The nodes defined here never require those
attributes.
"""

from little_kernel.core.ir.spans import SourceSpan, span_from_node
from little_kernel.core.ir.effects import IROpEffect, PURE_EFFECT, READ_EFFECT, WRITE_EFFECT, READWRITE_EFFECT, BARRIER_EFFECT, ESCAPE_EFFECT, EFFECT_CATEGORIES
from little_kernel.core.ir.nodes import LoweredIRNode, TargetExpression, TargetStatement, TargetInclude, TargetHelper, ResourceReference, StageReference, TARGET_SECTIONS, RESOURCE_SPACES, target_expression, target_statement, target_include, target_helper, resource_reference, stage_reference, is_lowered_node, iter_lowered_nodes, is_valid_ident_path, clone_lowered_node, map_lowered_node_children, grouped_target_spelling
from little_kernel.core.ir.verifier import LoweredIRVerificationError, verify_lowered_node, verify_lowered_ir
from little_kernel.core.ir.legacy import LEGACY_CPP_EXPR_NAME, is_legacy_cpp_node, legacy_cpp_spelling, legacy_cpp_kind, legacy_cpp_expr_type, upgrade_legacy_node, LegacyCppKind

__all__ = [
    "SourceSpan",
    "span_from_node",
    "IROpEffect",
    "PURE_EFFECT",
    "READ_EFFECT",
    "WRITE_EFFECT",
    "READWRITE_EFFECT",
    "BARRIER_EFFECT",
    "ESCAPE_EFFECT",
    "EFFECT_CATEGORIES",
    "LoweredIRNode",
    "TargetExpression",
    "TargetStatement",
    "TargetInclude",
    "TargetHelper",
    "ResourceReference",
    "StageReference",
    "TARGET_SECTIONS",
    "RESOURCE_SPACES",
    "target_expression",
    "target_statement",
    "target_include",
    "target_helper",
    "resource_reference",
    "stage_reference",
    "is_lowered_node",
    "iter_lowered_nodes",
    "is_valid_ident_path",
    "clone_lowered_node",
    "map_lowered_node_children",
    "grouped_target_spelling",
    "LoweredIRVerificationError",
    "verify_lowered_node",
    "verify_lowered_ir",
    "LEGACY_CPP_EXPR_NAME",
    "is_legacy_cpp_node",
    "legacy_cpp_spelling",
    "legacy_cpp_kind",
    "legacy_cpp_expr_type",
    "upgrade_legacy_node",
    "LegacyCppKind",
]
