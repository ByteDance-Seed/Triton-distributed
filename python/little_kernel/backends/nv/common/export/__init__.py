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
"""Generic LittleKernel export API.

This package is the single owner of typed export specifications, schema
validation and allowlists, host-side shape expressions, launch/alias
contracts, CUDA translation-unit emission and host bindings. Evaluator
adapters consume only this public surface; the legacy
``integrations.vibe_cuda_eval`` module is a thin compatibility facade.

Public functions:

* :func:`load_spec` / :func:`validate` -- parse and validate a raw manifest
  (path, mapping or :class:`RawSpec`) into an immutable
  :class:`ResolvedSpec`. An already-resolved spec is accepted idempotently.
* :func:`render_bundle` -- render an authored entry plus a typed spec into an
  in-memory :class:`ExportBundle` (CUDA + binding + model strings), with no
  evaluator directory layout or compile policy.
* :func:`write_bundle` -- write an :class:`ExportBundle` into a generic output
  directory.
* :func:`export` -- compatibility workspace writer preserving the legacy
  evaluator ``task/``/``agent/`` layout, stale-file cleanup and compile/rollback
  behavior.
* :func:`check` -- verify an on-disk compatibility submission still matches.

There is no task-name dispatch and no reference/oracle fallback: a missing or
malformed authored program fails honestly.
"""

from __future__ import annotations

from little_kernel.backends.nv.common.export._models import ConstructorArg, ConstexprPlan, ForeachPlan, LaunchPlan, OutputDecl, PipelineStage, Record, ResolvedFacts, ResolvedSpec, ScalarDecl, TensorInput, TensorListDecl, TmaDescriptor, ValidatedSpec
from little_kernel.backends.nv.common.export._render import ExportBundle, render_bundle, write_bundle
from little_kernel.backends.nv.common.export._runner import check, export
from little_kernel.backends.nv.common.export._schema import RawSpec, SchemaError, check_identifier
from little_kernel.backends.nv.common.export._validation import validate_spec

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
    "TensorListDecl",
    "TmaDescriptor",
    "LaunchPlan",
    "ConstexprPlan",
    "ForeachPlan",
    "PipelineStage",
    "ExportBundle",
    "SchemaError",
    "check_identifier",
    "load_spec",
    "validate",
    "render_bundle",
    "write_bundle",
    "export",
    "check",
]

# A ValidatedSpec is the authored layer of a ResolvedSpec; load_spec returns
# the fully resolved object, which is also a valid typed validated input.
load_spec = validate_spec
validate = validate_spec
