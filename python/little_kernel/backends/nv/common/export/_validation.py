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
"""Raw -> typed validated/resolved specification construction.

The byte-stable manifest resolver in :mod:`._spec` performs the single owned
parse/validation/derivation and returns a working dict. This module converts
that result into the immutable typed owners exactly once; it never re-parses
and never maintains a second allowlist. The authored raw object is deep-copied
before resolution, so validation cannot mutate the caller's manifest.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from little_kernel.backends.nv.common.export._records import _VALIDATED_BUILD_TOKEN, ConstructorArg, ConstexprPlan, ForeachPlan, LaunchPlan, OutputDecl, PipelineStage, Record, ResolvedFacts, ResolvedSpec, ScalarDecl, TensorInput, TensorListDecl, TmaDescriptor, ValidatedSpec, freeze
from little_kernel.backends.nv.common.export._schema import RawSpec, SchemaError
from little_kernel.backends.nv.common.export._spec import _resolve

# Top-level derived keys legacy_mapping() reconstructs explicitly. Anything
# else still present in the resolved view is an authored top-level section
# carried verbatim and must be a public manifest key.
_REBUILT_KEYS = frozenset({
    "kernel",
    "function",
    "inputs",
    "constructor",
    "scalars",
    "workspaces",
    "transforms",
    "pre_transforms",
    "post_transforms",
    "tensor_lists",
    "tma_descriptors",
    "launch",
    "output",
    "outputs",
    "_outputs",
    "_ranks",
    "_tensor_types",
    "_scalar_types",
    "_constructor_types",
    "_host_lists",
    "_optional_inputs",
    "_present_inputs",
    "_presence_args",
    "_kernel_args",
    "_runtime_kernel_args",
    "_launch_scalars",
    "_constexpr_constructor",
    "_constexpr_derived",
    "_constexpr_names",
    "_descriptor_names",
    "_descriptor_bundles",
    "_list_names",
    "_list_param_types",
    "_input_from",
    "_forward_containers",
    "_forward_pool",
    "_forward_unpack",
    "_forward_args",
    "_passthrough_forward_args",
    "_ignored_forward_args",
    "_optional_trailing",
    "_equalities_cpp",
    "_pipeline_stages",
    "_foreach_inputs",
    "_foreach_single_launch",
    "_foreach_dispatch",
    "_foreach_max_group",
    "_foreach_group_scalars",
    "_no_alias",
    "_wrap_single_as_tuple",
    "_kernel",
    "_function",
})


def _records(items, cls):
    return tuple(cls(item) for item in items)


def from_resolved_view(view: dict) -> ResolvedSpec:
    """Convert the single resolver's working dict into typed immutable owners."""
    authored = ValidatedSpec(
        _validation_key=_VALIDATED_BUILD_TOKEN,
        kernel=view["_kernel"],
        function=view["_function"],
        inputs=_records(view["inputs"], TensorInput),
        outputs=_records(view["_outputs"], OutputDecl),
        constructor=_records(view["constructor"], ConstructorArg),
        scalars=_records(view["scalars"], ScalarDecl),
        workspaces=_records(view["workspaces"], Record),
        transforms=_records(view["transforms"], Record),
        pre_transforms=_records(view["pre_transforms"], Record),
        post_transforms=_records(view["post_transforms"], Record),
        tensor_lists=_records(view["tensor_lists"], TensorListDecl),
        descriptors=_records(view["tma_descriptors"], TmaDescriptor),
        single_output="output" in view,
        no_alias=bool(view.get("_no_alias", False)),
        wrap_single_as_tuple=bool(view.get("_wrap_single_as_tuple", False)),
    )

    constexpr = ConstexprPlan(
        values=view["_constexpr_constructor"],
        derived=view["_constexpr_derived"],
        names=view["_constexpr_names"],
    )
    foreach = ForeachPlan(
        inputs=view["_foreach_inputs"],
        single_launch=bool(view["_foreach_single_launch"]),
        dispatch=view["_foreach_dispatch"],
        max_group=view.get("_foreach_max_group"),
        group_scalars=view["_foreach_group_scalars"],
    )
    launch = LaunchPlan(view["launch"])
    stages = tuple(
        PipelineStage(kernel=stage["kernel"], kernel_args=stage["kernel_args"], launch=LaunchPlan(stage["launch"]))
        for stage in view["_pipeline_stages"])
    bundles = {
        name: TmaDescriptor(item.to_legacy_item() if isinstance(item, Record) else item)
        for name, item in view["_descriptor_bundles"].items()
    }

    facts = ResolvedFacts(
        ranks=view["_ranks"],
        tensor_types=view["_tensor_types"],
        scalar_types=view["_scalar_types"],
        constructor_types=view["_constructor_types"],
        host_lists=view["_host_lists"],
        optional_inputs=view["_optional_inputs"],
        present_inputs=view["_present_inputs"],
        presence_args=view["_presence_args"],
        kernel_args=view["_kernel_args"],
        runtime_kernel_args=view["_runtime_kernel_args"],
        launch=launch,
        launch_scalars=view["_launch_scalars"],
        constexpr=constexpr,
        foreach=foreach,
        descriptor_names=view["_descriptor_names"],
        descriptor_bundles=bundles,
        list_names=view["_list_names"],
        list_param_types=view["_list_param_types"],
        input_from=view["_input_from"],
        forward_containers=view.get("_forward_containers", ()),
        forward_pool=view.get("_forward_pool", []),
        forward_unpack=view.get("_forward_unpack", {}),
        forward_args=view.get("_forward_args"),
        passthrough=view.get("_passthrough_forward_args", ()),
        ignored=view.get("_ignored_forward_args", ()),
        optional_trailing=bool(view.get("_optional_trailing", False)),
        equalities_cpp=view.get("_equalities_cpp", ()),
        pipeline_stages=stages,
    )

    # Derived markers produced by the single resolution pipeline that are
    # carried verbatim for the emitters (they are not independently typed).
    _DERIVED_EXTRA = frozenset({"_kernel_param_check"})
    authored_top = {}
    extra = {}
    for key, value in view.items():
        if key in _REBUILT_KEYS:
            continue
        if key.startswith("_") and key not in _DERIVED_EXTRA:
            raise RuntimeError(  # pragma: no cover - internal completeness guard
                f"resolver produced unowned derived field: {key!r}")
        if key in _DERIVED_EXTRA:
            extra[key] = freeze(value)
        else:
            authored_top[key] = freeze(value)
    return ResolvedSpec(authored, facts, authored_top, extra, _validation_key=_VALIDATED_BUILD_TOKEN)


def _coerce_raw(source: Any) -> dict:
    if isinstance(source, RawSpec):
        return copy.deepcopy(dict(source.data))
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source))
    raise SchemaError("spec must be a JSON object")


def validate_spec(source: Any) -> ResolvedSpec:
    """Validate a manifest (path, mapping, :class:`RawSpec` or resolved model).

    An already-resolved :class:`ResolvedSpec` is returned idempotently. Raw
    input is deep-copied before resolution, so the caller's object is never
    mutated.
    """
    if isinstance(source, ResolvedSpec):
        return source
    if isinstance(source, (str, Path)):
        raw = json.loads(Path(source).read_text())
    else:
        raw = _coerce_raw(source)
    # Resolve on an independent working copy; authored state is isolated.
    view = _resolve(raw)
    return from_resolved_view(view)
