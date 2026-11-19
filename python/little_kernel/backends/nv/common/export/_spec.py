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
"""Raw -> validated -> resolved export specifications.

The manifest parser lives here once: constructor/constexpr contracts, inputs
(tensors, host int-lists, container projections, tensor lists, optional
inputs), outputs and their alias/zero/optional contracts, workspaces,
transforms, shape equalities, scalars (runtime vs derived), forward-ABI
permutations and unpack tuples, TMA descriptor records, the kernel-arg plan,
multi-stage pipelines and foreach groups.

Internally the rest of the export pipeline consumes the immutable
``ResolvedSpec`` typed model; ``ResolvedSpec.legacy_mapping`` reconstructs
the documented compatibility view the byte-stable emitters consume.
"""

from __future__ import annotations

import ast
import json
import keyword
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from little_kernel.backends.nv.common.export._launch import _validate_launch
from little_kernel.backends.nv.common.export._schema import SchemaError, RawSpec, CAST_KEYS, CONSTRUCTOR_KEYS, DESCRIPTOR_KEYS, FOREACH_KEYS, INPUT_KEYS, LAUNCH_KEYS, OUTPUT_KEYS, PIPELINE_KEYS, PIPELINE_STAGE_KEYS, SCALAR_KEYS, TENSOR_LIST_KEYS, TOP_LEVEL_KEYS, TRANSFORM_KEYS, WORKSPACE_KEYS, _CONSTRUCTOR_CPP, _DTYPES, _IDENTIFIER, _SCALAR_COMPONENT, _SCALAR_DTYPES, _TMA_DTYPES, _TMA_DESCRIPTOR_FIELDS, _TMA_SWIZZLES, _identifier, _positive_int, reject_unknown_fields, schema_keys
from little_kernel.backends.nv.common.export._shapes import _derived_constexpr_expr, _flattened_leading_tma_sources, _shape_expr
from little_kernel.backends.nv.common.export._foreach import _FOREACH_FUSED_MAX, _validate_foreach_single

# Byte span swizzled by each supported swizzle mode; keyed by the manifest
# ``swizzle_bytes`` value, mirroring the swizzle-span rule enforced by the
# runtime ``TiledTensorMap2DPlan`` (see
# little_kernel/backends/nv/common/runtime/tma_descriptor.py).
_TMA_SWIZZLE_SPAN_BYTES = {32: 32, 64: 64, 128: 128}


def _check_tma_box_legality(name: str, box, swizzle_bytes: int, element_size: int) -> None:
    """Apply the driver legality rules shared by every ``agent_lk_make_tma_*``.

    The host emitters encode every manifest descriptor with
    ``CU_TENSOR_MAP_INTERLEAVE_NONE``, so the same two rules the Driver checks
    for that interleave mode apply at compile time to every rank:

    * the innermost box extent must be a whole number of 16-byte TMA atoms,
      which for a tensor of a given element size puts a dtype-dependent floor
      on ``box[0]`` (e.g. 4 elements for float32, 8 for bf16 — a float32
      small-box no-swizzle form with a 4-column inner box is exactly one
      atom and is legal);
    * with swizzling enabled the innermost box byte extent must fit the
      selected swizzle span.

    Validation lives here, once, so every emission path manifest descriptors
    flow through reports a precise compile-time error instead of an opaque
    ``cuTensorMapEncodeTiled`` failure at launch.
    """
    box_bytes = box[0] * element_size
    if box_bytes % 16 != 0:
        raise ValueError(f"TMA descriptor {name}: inner box of {box[0]} elements is "
                         f"{box_bytes} bytes, not a multiple of 16 bytes (the TMA atom "
                         f"size with no interleave)")
    swizzle_span = _TMA_SWIZZLE_SPAN_BYTES.get(swizzle_bytes)
    if swizzle_span is not None and box_bytes > swizzle_span:
        raise ValueError(f"TMA descriptor {name}: inner box of {box_bytes} bytes exceeds "
                         f"the selected swizzle span of {swizzle_span} bytes")


def _resolve(data: dict) -> dict:
    """Validate and resolve an in-memory raw manifest into the legacy view."""
    if not isinstance(data, dict):
        raise ValueError("spec must be a JSON object")
    # Single-schema ownership: a typo'd top-level key is a hard error rather
    # than a silently ignored option (the same contract the TMA descriptor
    # record already honors). Derived ``_*`` view fields are never authored.
    unknown_top = set(data) - TOP_LEVEL_KEYS
    if unknown_top:
        raise SchemaError(f"unknown export spec fields: {sorted(unknown_top)}")
    kernel = _identifier(data.get("kernel"), "kernel")
    function = _identifier(data.get("function", "lk_forward"), "function")
    inputs = data.get("inputs")
    output = data.get("output")
    outputs = data.get("outputs")
    constructor = data.get("constructor", [])
    scalars = data.get("scalars", [])
    workspaces = data.get("workspaces", [])
    transforms = data.get("transforms", [])
    pre_transforms = data.get("pre_transforms", [])
    post_transforms = data.get("post_transforms", [])
    descriptors = data.get("tma_descriptors", [])
    launch = data.get("launch")
    if not isinstance(constructor, list):
        raise ValueError("constructor must be a list")
    data["constructor"] = constructor
    constructor_types = {}
    constexpr_values = {}
    # Derived constexpr selectors in declaration order: each maps a bool
    # constructor name to the expression over preceding runtime constructor
    # args that fixes its value when the Model is constructed.
    derived_constexpr = []
    # Ordinary (runtime, non-ignored) constructor args seen so far; the only
    # names a later derived constexpr expression may reference.
    preceding_constructor: dict[str, str] = {}
    shape_constructor_names = set()
    saw_default = False
    for item in constructor:
        if not isinstance(item, dict):
            raise ValueError("each constructor argument must be an object")
        reject_unknown_fields(item, CONSTRUCTOR_KEYS, "constructor argument")
        name = _identifier(item.get("name"), "constructor argument")
        dtype = item.get("dtype")
        if (name in constructor_types or name in ("min", "max", "ceil_div", "present")
                or dtype not in ("int32", "int64", "float32", "bool", "ignored")):
            raise ValueError("constructor names must be unique and have a supported dtype")
        if "constexpr" in item:
            if dtype == "ignored":
                raise ValueError("ignored constructor arguments cannot be constexpr")
            value = item["constexpr"]
            if isinstance(value, str):
                # A string constexpr is a derived selector: its value is bound
                # at Model.__init__ from the actual preceding constructor
                # values, and every value of its (bool) domain gets its own
                # compile-time-pruned kernel variant at export. The source
                # args it reads stay ordinary runtime kernel arguments.
                if dtype != "bool":
                    raise ValueError("derived constexpr expression requires bool dtype "
                                     "(its False/True domain is compiled as kernel variants)")
                if "default" in item:
                    raise ValueError("constexpr constructor arguments must not declare a default")
                cpp = _derived_constexpr_expr(value, frozenset(preceding_constructor))
                derived_constexpr.append({"name": name, "expr": value, "cpp": cpp})
                item["_derived_constexpr"] = True
            else:
                valid_constexpr = ((dtype in ("int32", "int64") and type(value) is int)
                                   or (dtype == "float32" and type(value) in (int, float) and math.isfinite(value))
                                   or (dtype == "bool" and type(value) is bool))
                if not valid_constexpr:
                    raise ValueError("constructor constexpr value does not match its dtype")
                if "default" in item:
                    raise ValueError("constexpr constructor arguments must not declare a default")
                constexpr_values[name] = value
        if "default" in item:
            default = item["default"]
            if not (default is None or type(default) in (bool, int, float, str)):
                raise ValueError("constructor defaults must be JSON scalars")
            if type(default) is float and not math.isfinite(default):
                raise ValueError("constructor defaults must be finite")
            valid_default = (dtype == "ignored" or (dtype in ("int32", "int64") and type(default) is int)
                             or (dtype == "float32" and type(default) in (int, float))
                             or (dtype == "bool" and type(default) is bool))
            if not valid_default:
                raise ValueError("constructor default does not match its dtype")
            saw_default = True
        elif saw_default:
            raise ValueError("required constructor arguments must precede defaults")
        constructor_types[name] = dtype
        if dtype in ("int32", "int64"):
            shape_constructor_names.add(name)
        if "constexpr" not in item and dtype != "ignored":
            # Ordinary runtime constructor arg: eligible as a source for a
            # later derived constexpr selector expression.
            preceding_constructor[name] = dtype
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("spec needs at least one input")
    if outputs is None:
        if not isinstance(output, dict):
            raise ValueError("spec needs an output or outputs list")
        outputs = [output]
    elif output is not None or not isinstance(outputs, list) or not outputs:
        raise ValueError("use either output or a nonempty outputs list")
    data["_outputs"] = outputs
    if not isinstance(scalars, list) or not isinstance(workspaces, list) or not isinstance(launch, dict):
        raise ValueError("scalars/workspaces must be lists and launch must be an object")
    # Normalize defaulted sections back onto the manifest so downstream
    # renderers observe a complete record even when the field was omitted.
    data["scalars"] = scalars
    if (not isinstance(transforms, list) or not isinstance(pre_transforms, list)
            or not isinstance(post_transforms, list) or not isinstance(descriptors, list)):
        raise ValueError("transforms, pre_transforms, post_transforms and tma_descriptors must be lists")
    data["transforms"] = transforms
    data["pre_transforms"] = pre_transforms
    data["post_transforms"] = post_transforms
    data["tma_descriptors"] = descriptors
    data["workspaces"] = workspaces
    ranks = {}
    tensor_types = {}
    optional_inputs = set()
    # host int-list inputs (name -> declared-optional flag), declared with
    # {"kind": "int_list"}. The name never enters ranks/tensor_types, so
    # tensor subscript/stride expressions cannot reference it; the list is
    # observed only through the sum/len/has() shape-expression forms, the
    # host_cumsum1d/host_upload1d staging transforms and optional_if_none
    # outputs. Every render path that reads spec["_host_lists"] (binding
    # decls, host staging transforms, helper emission) must observe a
    # coherent, initialized record; tensor-only manifests keep it empty and
    # the host-only forms stay rejected by validation downstream.
    host_lists: dict[str, bool] = {}
    # Container-projection inputs: name -> (container, key) for inputs
    # declared {"name": ..., "from": container, "key": ...}.
    input_from = {}
    for item in inputs:
        if not isinstance(item, dict):
            raise ValueError("each input must be an object")
        reject_unknown_fields(item, INPUT_KEYS, "input")
        name = _identifier(item.get("name"), "input name")
        if (name in ranks or name in host_lists or name in constructor_types
                or name in {"output", "stream", "sum", "len", "has"}):
            raise ValueError(f"duplicate or reserved input name: {name}")
        kind = item.get("kind", "tensor")
        if kind == "int_list":
            # A host int-list input carries no tensor dimensions and is not a
            # projection source. Its declared device staging dtype is int32
            # (the binding allocates at::kInt), so its dtype must be int32;
            # rank/contiguous and container projection do not apply; an
            # "optional" list becomes std::optional<std::vector<int64_t>> and
            # None at the Python ABI, tracked only here -- never in
            # optional_inputs, which drives tensor-specific presence channels
            # (rebind, present() scalars, nullptr launchers).
            if item.get("dtype") != "int32":
                raise ValueError(f"host int-list dtype for {name} must be int32 "
                                 "(staged device tensors are int32)")
            if "rank" in item or "contiguous" in item:
                raise ValueError(f"host int-list input {name} does not take rank/contiguous")
            if "from" in item or "key" in item:
                raise ValueError(f"host int-list input {name} cannot project from a container")
            if type(item.get("optional", False)) is not bool:
                raise ValueError(f"optional flag for {name} must be boolean")
            host_lists[name] = bool(item.get("optional", False))
            item["_kind"] = "int_list"
            continue
        if kind != "tensor":
            raise ValueError(f"unsupported kind for input {name}")
        # An input may optionally project out of a container forward argument:
        # {"name": "ck", ..., "from": "cache"} declares tensor ck whose
        # call-site spelling is cache["ck"] (override the mapping key with
        # "key"), while the generated Model.forward keeps one parameter per
        # container in first-use order. Opt-in; plain inputs are unchanged.
        container = item.get("from")
        if container is not None:
            container = _identifier(container, "input container")
            key = item.get("key", name)
            if not isinstance(key, str) or not key:
                raise ValueError(f"container key for input {name} must be a nonempty string")
            input_from[name] = (container, key)
        elif "key" in item:
            raise ValueError(f"input {name} key requires a container")
        # Rank 0 declares a device scalar: the binding emits a dim()==0 check,
        # and dimension/stride expressions cannot reference it (the rank check
        # inside _shape_expr rejects any subscript on a rank-0 name). Kept
        # nonnegative (not _positive_int) because existing manifests declare
        # rank-0 scalar tensors.
        rank = item.get("rank")
        if type(rank) is not int or rank < 0:
            raise ValueError(f"rank for {name} must be a nonnegative integer (rank 0 is a 0-dim scalar tensor)")
        if item.get("dtype") not in _DTYPES:
            raise ValueError(f"unsupported dtype for {name}")
        if type(item.get("contiguous", True)) is not bool:
            raise ValueError(f"contiguous flag for {name} must be boolean")
        if type(item.get("optional", False)) is not bool:
            raise ValueError(f"optional flag for {name} must be boolean")
        if item.get("optional", False):
            # The forward ABI slot accepts None (pybind converts it to an
            # empty c10::optional); the binding rebinds the name to a possibly
            # undefined tensor alias, guards every check behind .defined() and
            # passes nullptr to the launcher when absent. Kernels test presence
            # through a scalar declared with the present() expression.
            optional_inputs.add(name)
        ranks[name] = rank
        tensor_types[name] = item["dtype"]
    data["_optional_inputs"] = [x["name"] for x in inputs if x.get("optional", False)]
    data["_host_lists"] = host_lists
    # tensor_lists declare Python list-of-tensor forward arguments bound as
    # std::vector<torch::Tensor>. Each list flattens to one raw device
    # pointer per declared kernel param (element k's data_ptr(), or nullptr
    # when the runtime list is shorter than k+1), letting one kernel launch
    # cover the ragged, batched group. List names and param names never enter
    # `ranks`, so shape/stride/launch expressions cannot reference them.
    tensor_lists = data.get("tensor_lists", [])
    if not isinstance(tensor_lists, list):
        raise ValueError("tensor_lists must be a list")
    data["tensor_lists"] = tensor_lists
    list_names: list[str] = []
    list_param_types: dict[str, str] = {}
    for item in tensor_lists:
        if not isinstance(item, dict):
            raise ValueError("each tensor_list must be an object")
        reject_unknown_fields(item, TENSOR_LIST_KEYS, "tensor_list")
        name = _identifier(item.get("name"), "tensor_list name")
        if name in ranks or name in constructor_types or name in list_names or name in list_param_types:
            raise ValueError(f"duplicate tensor_list name: {name}")
        if item.get("dtype") not in _DTYPES:
            raise ValueError(f"unsupported dtype for tensor_list {name}")
        rank = item.get("rank")
        if type(rank) is not int or rank < 0:
            raise ValueError(f"rank for tensor_list {name} must be a nonnegative integer")
        max_len = item.get("max_len")
        if type(max_len) is not int or max_len <= 0:
            raise ValueError(f"max_len for tensor_list {name} must be a positive integer")
        params = item.get("kernel_params")
        if (not isinstance(params, list) or len(params) != max_len
                or any(not isinstance(p, str) or not _IDENTIFIER.fullmatch(p) or keyword.iskeyword(p) for p in params)
                or len(set(params)) != len(params)):
            raise ValueError(f"tensor_list {name} kernel_params must be {max_len} distinct identifiers")
        for param in params:
            if param in ranks or param in constructor_types or param in list_param_types or param in list_names:
                raise ValueError(f"tensor_list kernel param collides: {param}")
            list_param_types[param] = item["dtype"]
        item["_name"] = name
        list_names.append(name)
    data["_list_names"] = list_names
    data["_list_param_types"] = list_param_types
    optional_input_names = {x["name"] for x in inputs if x.get("optional", False)}
    if inputs and inputs[0]["name"] in optional_input_names:
        raise ValueError("the first input cannot be optional: the first input must not "
                         "be optional since output tensors are allocated from its options()")
    # Pre-scan runtime scalar declarations so output/workspace shape
    # expressions and shape equalities may reference runtime forward scalars
    # by name (they are C++ function parameters, hence in scope where outputs
    # are allocated). Full validation of the scalar list still happens below;
    # only integer scalars may appear in integer shape expressions.
    early_runtime_scalars = set()
    for item in scalars:
        if not isinstance(item, dict):
            raise ValueError("each scalar must be an object")
        if item.get("runtime", False) is True and item.get("dtype", "int32") in ("int32", "int64"):
            early_runtime_scalars.add(_identifier(item.get("name"), "scalar name"))
    # Pre-scan validated derived integer scalars so output/workspace shapes
    # may reference a derived extent by name. The generated binding defines
    # each derived scalar (const int64_t name = expr;) only *after* the
    # output/workspace at::empty sites, so a shape cannot reference the
    # variable; instead the allocation substitutes the computed expression
    # inline. A scalar joins the substitution map only when its expression
    # resolves in the allocation-site environment available there: input
    # tensor dims and strides (outputs/workspaces are not yet allocated),
    # constructor integers, runtime integer scalars, host int-list helpers
    # and earlier substitution-eligible derived scalars. A derived scalar
    # whose expression needs a later extent (an output or workspace dim)
    # simply stays out of the map: its own definition site (after the
    # allocations, and launch.grid after that) still references it by name
    # as before, and a shape that names it keeps the historical rejection.
    # The pre-scan never mutates manifest fields and never turns a legal
    # manifest into an error: full scalar validation below owns rejections.
    allocation_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
    derived_substitutions: dict[str, str] = {}
    reserved_scalar_names = {"min", "max", "ceil_div", "present"}
    for item in scalars:
        if not isinstance(item, dict):
            raise ValueError("each scalar must be an object")
        if item.get("runtime", False) is True:
            continue
        name = item.get("name")
        if (item.get("dtype", "int32") not in ("int32", "int64") or "expr" not in item or not isinstance(name, str)
                or not _IDENTIFIER.fullmatch(name) or name in ranks or name in constructor_types or name in host_lists
                or name in list_names or name in list_param_types or name in early_runtime_scalars
                or name in derived_substitutions or name in reserved_scalar_names):
            continue
        try:
            cpp = _shape_expr(item.get("expr"), allocation_ranks, shape_constructor_names | early_runtime_scalars,
                              optional_inputs=optional_inputs, lists=host_lists, derived=derived_substitutions)
        except ValueError:
            continue
        derived_substitutions[name] = cpp
    output_names = []
    input_names_for_alias = {x["name"] for x in inputs}
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError("each output must be an object")
        reject_unknown_fields(item, OUTPUT_KEYS, "output")
        out_name = _identifier(item.get("name"), "output name")
        shape = item.get("shape")
        alias_of = item.get("alias_of")
        # A self-aliased output names the input it updates in place: the
        # kernel writes through the caller's buffer and the wrapper returns
        # that same buffer object, so no fresh tensor is created anywhere.
        # Compute this BEFORE the uniqueness check so a self-alias (name ==
        # one of the input names == alias_of) is recognized as the in-place
        # canvas pattern rather than as a name collision.
        self_alias = alias_of == out_name and out_name in input_names_for_alias
        if (out_name in constructor_types or out_name in host_lists or out_name in list_names
                or out_name in list_param_types or item.get("dtype") not in _DTYPES):
            raise ValueError("output name must be unique and dtype supported")
        if out_name in ranks and not self_alias:
            raise ValueError("output name must be unique and dtype supported")
        optional_if_none = item.get("optional_if_none")
        if optional_if_none is not None:
            # A conditionally present output: the generated binding returns the
            # freshly allocated tensor when the named optional host int-list
            # input carries a value, and Python None when it does not.
            optional_if_none = _identifier(optional_if_none, "optional output condition")
            if alias_of is not None:
                raise ValueError("optional outputs cannot alias an input")
            if not host_lists.get(optional_if_none, False):
                raise ValueError("optional_if_none must name an optional host int-list input")
            item["_optional_if_none"] = optional_if_none
        if alias_of is not None:
            # An aliased output is the named model input itself (updated in
            # place by the kernel) rather than a freshly allocated tensor.
            alias_of = _identifier(alias_of, "output alias source")
            if alias_of not in input_names_for_alias:
                raise ValueError("output alias_of must name a model input")
            if alias_of in optional_input_names:
                raise ValueError("an optional input cannot alias an output: "
                                 "output alias_of must not alias an optional input")
            if tensor_types[alias_of] != item["dtype"]:
                raise ValueError("output alias_of dtype must match the aliased input")
            item["alias_of"] = alias_of
            if shape is None:
                shape = [f"{alias_of}[{dim}]" for dim in range(ranks[alias_of])]
                item["shape"] = shape
            elif isinstance(shape, list) and len(shape) != ranks[alias_of]:
                # Ambiguous alias: the binding IS the input tensor, so a
                # declared rank that differs from the aliased input's rank
                # would misreport the shape metadata every launch/scalar
                # expression resolves against. Reject rather than guess.
                raise ValueError("output alias_of with an explicit shape must "
                                 "match the aliased input's rank")
        if not isinstance(shape, list):
            raise ValueError("output shape must be a list (may be empty for a 0-dim scalar output)")
        if any(type(value) is not int and not isinstance(value, str) for value in shape):
            raise ValueError("output shape dimensions must be ints or shape expressions")
        # An empty shape list declares a rank-0 (scalar) output, matching
        # PyTorch references that return 0-dim tensors (e.g. out.sum()): the
        # binding renders at::empty({}, ...) and the kernel stores element 0.
        # The C++ initializer {} allocates a 1-element 0-dim tensor, whose
        # numel()==1 keeps the existing empty-output early-return semantics
        # intact. Rank-0 is legal only for freshly allocated outputs; the
        # binding launch plumbing is rank-agnostic (data_ptr), and shape
        # expressions referencing a rank-0 tensor's dims are rejected by
        # _shape_expr. Integer runtime forward scalars may be referenced by
        # name in shape expressions (they are binding parameters in scope at
        # the allocation site).
        # zero_init validation: a freshly allocated output may request a
        # stream-ordered memset before the kernel launch; an aliased output
        # has no fresh allocation to zero, so the combination is rejected
        # rather than silently skipping the memset.
        if type(item.get("zero_init", False)) is not bool:
            raise ValueError("output zero_init must be boolean")
        if item.get("zero_init") and item.get("alias_of") is not None:
            raise ValueError("output zero_init cannot be combined with alias_of")
        # An optional input may be absent at runtime, so its sizes must never
        # feed output shape expressions; removing it from the ranks map makes
        # any such reference a validation error.
        expr_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
        item["_shape_cpp"] = [
            _shape_expr(value, expr_ranks, shape_constructor_names | early_runtime_scalars,
                        optional_inputs=optional_inputs, lists=host_lists, derived=derived_substitutions)
            for value in shape
        ]
        ranks[out_name] = len(shape)
        tensor_types[out_name] = item["dtype"]
        output_names.append(out_name)
    # Opt-in non-aliasing assertion: when "no_alias" is true the generated
    # entry kernel marks its pointer parameters __restrict__. The author
    # thereby asserts that pointer arguments never overlap, so the flag is a
    # contradiction (compile-time error) whenever an output aliases an input.
    no_alias = data.get("no_alias", False)
    if type(no_alias) is not bool:
        raise ValueError("no_alias must be a boolean")
    if no_alias and any(item.get("alias_of") is not None for item in outputs):
        raise ValueError("no_alias contradicts an aliased (alias_of) output")
    data["_no_alias"] = no_alias
    # Opt-in singleton-tuple return contract: some PyTorch reference models
    # return a one-element tuple `(y,)` rather than the bare tensor. When
    # "wrap_single_as_tuple" is true the generated Model.forward wraps the
    # single output as `return (value,)` so the drop-in structure matches the
    # reference exactly. Multi-output exports already return a tuple and
    # foreach exports return their own group structure, so the flag is only
    # meaningful for a plain single-output export.
    wrap_single_as_tuple = data.get("wrap_single_as_tuple", False)
    if type(wrap_single_as_tuple) is not bool:
        raise ValueError("wrap_single_as_tuple must be a boolean")
    if wrap_single_as_tuple:
        if len(outputs) != 1:
            raise ValueError("wrap_single_as_tuple requires exactly one output")
        if data.get("foreach"):
            raise ValueError("wrap_single_as_tuple cannot be combined with foreach")
    data["_wrap_single_as_tuple"] = wrap_single_as_tuple
    for item in workspaces:
        if not isinstance(item, dict):
            raise ValueError("each workspace must be an object")
        reject_unknown_fields(item, WORKSPACE_KEYS, "workspace")
        name = _identifier(item.get("name"), "workspace name")
        shape = item.get("shape")
        if (name in ranks or name in constructor_types or name in list_names or name in list_param_types
                or item.get("dtype") not in _DTYPES or not isinstance(shape, list) or not shape):
            raise ValueError("workspace needs a new name, supported dtype and nonempty shape")
        workspace_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
        item["_shape_cpp"] = [
            _shape_expr(value, workspace_ranks, shape_constructor_names | early_runtime_scalars,
                        optional_inputs=optional_inputs, lists=host_lists, derived=derived_substitutions)
            for value in shape
        ]
        if type(item.get("zero", False)) is not bool:
            raise ValueError("workspace zero must be boolean")
        ranks[name] = len(shape)
        tensor_types[name] = item["dtype"]
    for item in transforms:
        if not isinstance(item, dict) or item.get("op") not in ("transpose2d", "prefix_sum1d", "alias", "host_cumsum1d",
                                                                "host_upload1d"):
            raise ValueError("unsupported transform op")
        reject_unknown_fields(item, TRANSFORM_KEYS, "transform")
        name = _identifier(item.get("name"), "transform name")
        source = _identifier(item.get("source"), "transform source")
        if item["op"] in ("host_cumsum1d", "host_upload1d"):
            # Host staging reads a declared host int-list input — including an
            # optional one, whose absent value stages a 0-element tensor and
            # skips the H2D copy — and produces a fresh 1-D int32 device
            # tensor (the binding allocates at::kInt). The generic source
            # check below requires a tensor-typed model input, which a host
            # list never is; that check applies to the tensor transforms.
            if (name in ranks or name in constructor_types or name in list_names or name in list_param_types
                    or source not in host_lists or source in output_names):
                raise ValueError("host staging must name a new tensor from a declared host int-list input")
            ranks[name] = 1
            tensor_types[name] = "int32"
            continue
        if (name in ranks or name in constructor_types or name in list_names or name in list_param_types
                or source not in tensor_types or source in output_names):
            raise ValueError("transform must name a new tensor from a model input")
        if source in optional_inputs:
            raise ValueError("an optional input cannot be a transform source")
        if item["op"] == "transpose2d":
            if ranks[source] != 2 or tensor_types[source] not in ("float16", "bfloat16"):
                raise ValueError("transpose2d requires a 2-D float16/bfloat16 input")
            ranks[name] = 2
        elif item["op"] == "prefix_sum1d":
            if ranks[source] != 1 or tensor_types[source] != "int32":
                raise ValueError("prefix_sum1d requires a 1-D int32 input")
            ranks[name] = 1
        else:
            ranks[name] = ranks[source]
        tensor_types[name] = tensor_types[source]
    for phase, items in (("pre", pre_transforms), ("post", post_transforms)):
        seen_targets = set()
        for item in items:
            if not isinstance(item, dict) or item.get("op") != ("pad_cast2d" if phase == "pre" else "crop_cast2d"):
                raise ValueError(f"unsupported {phase} transform")
            reject_unknown_fields(item, CAST_KEYS, f"{phase} transform")
            source = _identifier(item.get("source"), "cast source")
            target = _identifier(item.get("target"), "cast target")
            if (source not in tensor_types or target not in tensor_types or source == target or ranks[source] != 2
                    or ranks[target] != 2 or target in seen_targets or source in optional_inputs):
                raise ValueError("cast transform requires distinct rank-2 tensors and a unique target")
            if phase == "pre" and target not in [x["name"] for x in workspaces]:
                raise ValueError("pad_cast2d target must be a workspace")
            if phase == "post" and target not in output_names:
                raise ValueError("crop_cast2d target must be an output")
            if (tensor_types[source], tensor_types[target]) != (("float32", "bfloat16") if phase == "pre" else
                                                                ("bfloat16", "float32")):
                raise ValueError("cast transform expects float32->bfloat16 before or bfloat16->float32 after")
            seen_targets.add(target)
    equalities = data.get("equalities", [])
    if not isinstance(equalities, list):
        raise ValueError("equalities must be a list")
    data["_equalities_cpp"] = []
    equality_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
    for pair in equalities:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("each equality must contain two shape expressions")
        data["_equalities_cpp"].append(
            (_shape_expr(pair[0], equality_ranks, shape_constructor_names | early_runtime_scalars,
                         optional_inputs=optional_inputs, lists=host_lists),
             _shape_expr(pair[1], equality_ranks, shape_constructor_names | early_runtime_scalars,
                         optional_inputs=optional_inputs, lists=host_lists)))
    used = set(ranks) | set(constructor_types) | set(list_names) | set(list_param_types)
    scalar_names = set()
    for item in scalars:
        if not isinstance(item, dict):
            raise ValueError("each scalar must be an object")
        reject_unknown_fields(item, SCALAR_KEYS, "scalar")
        name = _identifier(item.get("name"), "scalar name")
        if name in used or name in {"min", "max", "ceil_div", "present"}:
            raise ValueError(f"duplicate scalar name: {name}")
        used.add(name)
        dtype = item.get("dtype", "int32")
        if dtype not in _SCALAR_DTYPES:
            raise ValueError(f"scalar dtype must be one of {_SCALAR_DTYPES}")
        item["_dtype"] = dtype
        item["_runtime"] = item.get("runtime", False)
        if type(item["_runtime"]) is not bool:
            raise ValueError("scalar runtime flag must be boolean")
        if item["_runtime"]:
            if "expr" in item:
                raise ValueError("a runtime forward scalar must not declare an expr")
        else:
            if dtype in _SCALAR_COMPONENT:
                raise ValueError("complex scalars must be runtime forward arguments")
            constructor_expr_names = (
                {name
                 for name, kind in constructor_types.items()
                 if kind in ("int32", "int64", "float32")} if dtype in ("float32",
                                                                        "float64") else shape_constructor_names)
            # Optional inputs may be absent at runtime, so their sizes must
            # never feed constructor scalar expressions; removing them from
            # the ranks map makes any such reference a validation error.
            constructor_expr_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
            item["_expr_cpp"] = _shape_expr(item.get("expr"), constructor_expr_ranks,
                                            scalar_names | constructor_expr_names, allow_float=dtype
                                            in ("float32", "float64"), optional_inputs=optional_inputs,
                                            lists=host_lists)
        scalar_names.add(name)
    runtime_scalar_names = [item["name"] for item in scalars if item["_runtime"]]
    # Optional-input binding mode is selected from the manifest alone.
    # pybind11 requires defaulted parameters to be trailing: whenever the
    # binding signature carries any argument after the tensor inputs (runtime
    # forward scalars, non-ignored/non-constexpr constructor values, or tensor
    # lists), an optional parameter cannot stay in input position, so those
    # manifests use trailing c10::optional parameters guarded by .has_value(),
    # pybind11::arg(name) = pybind11::none(), model-side None defaults and
    # keyword calls. Manifests without such tail arguments keep the positional
    # lk_opt_ rebind spelling (an unused optional input may appear mid-list).
    signature_tail_params = bool(list_names or runtime_scalar_names
                                 or any(item["dtype"] != "ignored" and "constexpr" not in item
                                        for item in data["constructor"]))
    data["_optional_trailing"] = bool(optional_inputs) and signature_tail_params
    if data["_optional_trailing"]:
        first_optional = next(pos for pos, item in enumerate(inputs) if item["name"] in optional_inputs)
        if any(item["name"] not in optional_inputs for item in inputs[first_optional:]):
            raise ValueError("optional inputs must be the trailing model inputs")
    # Record which optional inputs are observed through a present() scalar.
    # Those use a std::optional value parameter plus zero-sized
    # materialization and a has_value() scalar; plain optional inputs use a
    # c10::optional const ref, possibly-undefined alias, guarded checks and a
    # nullptr launch argument. _shape_expr already validated each present()
    # target as a declared optional input, and its constant-expression form
    # cannot contain a present() call.
    present_inputs = set()
    for item in scalars:
        if item["_runtime"]:
            continue
        expr_source = item["expr"] if isinstance(item["expr"], str) else str(item["expr"])
        for node in ast.walk(ast.parse(expr_source)):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "present"):
                present_inputs.add(node.args[0].id)
    data["_present_inputs"] = present_inputs
    # forward_unpack maps a new forward tuple-argument name to a nonempty
    # list of distinct runtime scalar names; the generated model forward
    # takes the tuple and unpacks it in Python before calling the binding
    # (whose signature still receives the plain runtime scalars). Members
    # remain ordinary runtime scalars: kernel args, launch grids and shape
    # expressions may use them, or they may be left unused entirely.
    forward_unpack = data.get("forward_unpack")
    unpack_map = {}
    if forward_unpack is not None:
        if not isinstance(forward_unpack, dict) or not forward_unpack:
            raise ValueError("forward_unpack must be a nonempty object")
        runtime_set = set(runtime_scalar_names)
        covered = set()
        for member_key, members in forward_unpack.items():
            tuple_name = _identifier(member_key, "forward_unpack tuple name")
            if tuple_name in used:
                raise ValueError(f"duplicate forward_unpack tuple name: {tuple_name}")
            if (not isinstance(members, list) or not members
                    or any(not isinstance(member, str) or member not in runtime_set for member in members)
                    or len(set(members)) != len(members) or any(member in covered for member in members)):
                raise ValueError("forward_unpack members must be distinct runtime scalar names")
            covered.update(members)
            unpack_map[tuple_name] = members
        used.update(unpack_map)
    data["_forward_unpack"] = unpack_map
    # The model forward's effective parameter pool: inputs, then tuple args,
    # then the runtime scalars not covered by any tuple.
    covered_scalars = {member for members in unpack_map.values() for member in members}
    forward_pool = ([x["name"] for x in inputs] + list(unpack_map) +
                    [name for name in runtime_scalar_names if name not in covered_scalars])
    data["_forward_pool"] = forward_pool
    passthrough = data.get("passthrough_forward_args", [])
    if not isinstance(passthrough, list):
        raise ValueError("passthrough_forward_args must be a list")
    passthrough_names = []
    for item in passthrough:
        # Passthrough slots are part of the fixed forward ABI (e.g. string
        # selectors) but carry no value usable by the kernel: the generated
        # model accepts them positionally and never forwards them.
        name = _identifier(item, "passthrough forward argument")
        if name in used or name in passthrough_names:
            raise ValueError(f"passthrough forward argument must be a new unique name: {name}")
        used.add(name)
        passthrough_names.append(name)
    data["_passthrough_forward_args"] = passthrough_names
    forward_args = data.get("forward_args")
    if forward_args is not None:
        # forward_args selects the positional order of the generated model's
        # forward signature; the extension is still called in binding order.
        # The permutable pool is inputs, then tensor-list arguments, then
        # forward_unpack tuple names, then the runtime scalars not covered by
        # any tuple, then passthrough ABI slots. With no tuples this equals
        # inputs+lists+runtime scalars+passthrough; with no lists it equals
        # forward_pool+passthrough.
        expected_forward = ([x["name"] for x in inputs] + list_names + list(unpack_map) +
                            [name for name in runtime_scalar_names if name not in covered_scalars] + passthrough_names)
        if (not isinstance(forward_args, list) or len(forward_args) != len(expected_forward)
                or any(not isinstance(name, str) or name not in expected_forward for name in forward_args)
                or len(set(forward_args)) != len(forward_args)):
            raise ValueError("forward_args must be a permutation of the forward parameter pool "
                             "(inputs, tensor lists, forward_unpack tuples, runtime scalars, "
                             "passthrough args)")
    elif passthrough_names:
        # Positions of ABI slots that carry no kernel value must come from the
        # manifest record, never from implicit name matching.
        raise ValueError("passthrough_forward_args requires forward_args to fix positions")
    data["_forward_args"] = forward_args
    # Python and pybind11 both require defaulted parameters to be trailing,
    # so in the effective forward order nothing may follow an optional input
    # except another optional input.
    forward_order = (forward_args if forward_args is not None else [x["name"] for x in inputs] + runtime_scalar_names)
    # The trailing-arguments constraint applies only under the trailing
    # optional ABI (model forward gives plain optionals None defaults, which
    # must stay trailing); the positional lk_opt_ ABI imposes no ordering.
    if data["_optional_trailing"]:
        for position, forward_name in enumerate(forward_order):
            if forward_name not in optional_inputs and any(prior in optional_inputs
                                                           for prior in forward_order[:position]):
                raise ValueError("optional inputs must be the trailing forward arguments")
    descriptor_names = set()
    for item in descriptors:
        if not isinstance(item, dict):
            raise ValueError("each TMA descriptor must be an object")
        name = _identifier(item.get("name"), "TMA descriptor name")
        tensor = _identifier(item.get("tensor"), "TMA tensor")
        # A rank-5 tensor is legal only as the source of a rank-4 per-batch
        # bundle (one leading batch axis over the four mapped dims), matching
        # the hardware tensor-map input-rank cap of 5.
        if (name in used or tensor not in tensor_types or ranks[tensor] not in (1, 2, 3, 4, 5)
                or tensor in optional_inputs):
            raise ValueError("TMA descriptor needs a new name and a rank-1/2/3/4/5 tensor")
        if tensor_types[tensor] not in _TMA_DTYPES:
            raise ValueError("unsupported TMA tensor dtype")
        unknown_fields = set(item) - _TMA_DESCRIPTOR_FIELDS
        if unknown_fields:
            raise ValueError(f"unknown TMA descriptor fields: {sorted(unknown_fields)}")
        used.add(name)
        descriptor_names.add(name)
        rank = item.get("rank", 2)
        if type(rank) is not int or rank not in (2, 3, 4):
            raise ValueError("TMA descriptor rank must be 2, 3 or 4")
        if ranks[tensor] == 1 and rank != 2:
            raise ValueError("rank-1 tensors require a rank-2 TMA descriptor with outer=1")
        flatten_leading = item.get("flatten_leading", False)
        if type(flatten_leading) is not bool:
            raise ValueError("TMA flatten_leading must be boolean")
        if flatten_leading and (rank != 2 or ranks[tensor] < 2):
            raise ValueError("flatten_leading requires a rank-2 TMA descriptor over a rank-2+ tensor")
        item["_rank"] = rank
        # Per-batch descriptor bundle flag (the warp-specialized BMM
        # MapBundle): ``batch_count`` switches the entry from one
        # grid_constant-by-value CUtensorMap to a device-resident array of
        # ``count`` identical maps, one per leading batch slice, which the
        # entry kernel indexes at runtime through a const CUtensorMap*
        # parameter (dB[b]). Checked before extent parsing so a tensor
        # without a leading batch axis gets the semantic error rather than
        # a dimension-out-of-range on the slice expressions.
        item["_bundle"] = item.get("batch_count") is not None
        if item["_bundle"] and ranks[tensor] < rank + 1:
            raise ValueError("per-batch TMA descriptor bundle requires a tensor with a "
                             "leading batch axis (tensor rank must exceed descriptor rank)")
        if ranks[tensor] == 5 and not (item["_bundle"] and rank == 4):
            raise ValueError("rank-5 TMA tensors require a rank-4 per-batch descriptor "
                             "bundle (one leading batch axis over four mapped dims)")
        if not item["_bundle"] and "batch_stride" in item:
            raise ValueError("TMA descriptor batch_stride requires batch_count")
        fields = {
            2: ("inner", "outer", "outer_stride"), 3: ("dim0", "dim1", "dim2", "stride1", "stride2"), 4:
            ("dim0", "dim1", "dim2", "dim3", "stride1", "stride2", "stride3")
        }[rank]
        # Descriptor extents/strides are host expressions too: an optional
        # tensor may be absent at runtime, so its sizes are filtered out of
        # the ranks map here exactly as for output/workspace shapes and
        # launch expressions.
        tma_ranks = {key: value for key, value in ranks.items() if key not in optional_inputs}
        if flatten_leading:
            canonical = _flattened_leading_tma_sources(tensor, ranks[tensor])
            for field, canonical_source in canonical.items():
                if field in item and item[field] is not None:
                    actual_cpp = _shape_expr(item[field], tma_ranks, scalar_names | shape_constructor_names,
                                             optional_inputs=optional_inputs, lists=host_lists)
                    canonical_cpp = _shape_expr(canonical_source, tma_ranks, scalar_names | shape_constructor_names,
                                                optional_inputs=optional_inputs, lists=host_lists)
                    if actual_cpp != canonical_cpp:
                        raise ValueError(f"flatten_leading TMA descriptor {field} must be {canonical_source}")
                else:
                    item[field] = canonical_source
        for field in fields:
            item[f"_{field}_cpp"] = _shape_expr(item.get(field), tma_ranks, scalar_names | shape_constructor_names,
                                                optional_inputs=optional_inputs, lists=host_lists)
        if ranks[tensor] == 1 and item["_outer_cpp"] != "1":
            raise ValueError("rank-1 TMA descriptor must use outer=1")
        box = item.get("box")
        if not isinstance(box, list) or len(box) != rank:
            raise ValueError("TMA box must have one dimension per descriptor rank")
        item["_box"] = [_positive_int(value, "TMA box dimension") for value in box]
        swizzle = item.get("swizzle_bytes", 0)
        if type(swizzle) is not int or swizzle not in _TMA_SWIZZLES:
            raise ValueError("unsupported TMA swizzle")
        item["_swizzle"] = _TMA_SWIZZLES[swizzle]
        _check_tma_box_legality(name, item["_box"], swizzle, _TMA_DTYPES[tensor_types[tensor]][1])
        if type(item.get("oob_fill", False)) is not bool:
            raise ValueError("oob_fill must be boolean")
        item["_dtype"] = tensor_types[tensor]
        if item["_bundle"]:
            # A single by-value descriptor cannot carry a runtime batch
            # coordinate, and re-encoding on device is unavailable, so the
            # host ABI builds the whole bundle once per launch.
            batch_count = item["batch_count"]
            item["_batch_count_cpp"] = _shape_expr(batch_count, tma_ranks, scalar_names | shape_constructor_names,
                                                   optional_inputs=optional_inputs, lists=host_lists)
            batch_stride = item.get("batch_stride")
            if batch_stride is None:
                # Contiguous leading-axis layout: one slice spans all outer
                # rows (rank 2), the full dim0*dim1 plane (rank 3) or the
                # full dim0*dim1*dim2 volume (rank 4).
                if rank == 2:
                    item["_batch_stride_cpp"] = (f"(({item['_outer_cpp']}) * ({item['_outer_stride_cpp']}))")
                elif rank == 3:
                    item["_batch_stride_cpp"] = (f"(({item['_dim2_cpp']}) * ({item['_stride2_cpp']}))")
                else:
                    item["_batch_stride_cpp"] = (f"(({item['_dim3_cpp']}) * ({item['_stride3_cpp']}))")
            else:
                item["_batch_stride_cpp"] = _shape_expr(batch_stride, tma_ranks, scalar_names | shape_constructor_names,
                                                        optional_inputs=optional_inputs, lists=host_lists)
    launch_scalars = scalar_names | shape_constructor_names
    # Validated through the same helper every pipeline stage uses, so the
    # single-launch schema and each stage's launch record share exactly one
    # rule and one host-side shape-expression language.
    _validate_launch(launch, ranks, launch_scalars, optional_inputs, host_lists)
    # A kernel consumes the list's flattened per-element pointers via the
    # declared kernel_params, never the list object itself.
    available = set(ranks) | scalar_names | descriptor_names | set(list_param_types) | {
        name
        for name, dtype in constructor_types.items()
        if dtype != "ignored"
    }
    default_args = list(
        dict.fromkeys([x["name"] for x in inputs if x.get("_kind", "tensor") == "tensor"] + output_names +
                      [x["name"] for x in scalars]))
    kernel_args = data.get("kernel_args", default_args)
    if (not isinstance(kernel_args, list) or not kernel_args
            or any(not isinstance(x, str) or x not in available for x in kernel_args)
            or len(kernel_args) != len(set(kernel_args))):
        raise ValueError("kernel_args must list distinct declared tensors, scalars or descriptors")
    # Every plain optional tensor in kernel_args is followed by an injected
    # int32 presence scalar `<name>_present` (1 when the tensor is defined,
    # else 0); the frontend declares it as a trailing `ll.int32` kernel
    # parameter and gates the tensor's loads/stores on it. present()-driven
    # inputs are observed through their declared present() scalar instead —
    # injecting a second presence argument would give one optional tensor two
    # presence channels and desynchronize the frontend parameter list, so
    # they are skipped here (tests/nv/sm90/unit/test_vibe_abi_records.py).
    presence_args = {}
    if optional_inputs:
        expanded = []
        for arg_name in kernel_args:
            expanded.append(arg_name)
            if arg_name in optional_inputs and arg_name not in present_inputs:
                presence_name = arg_name + "_present"
                if (presence_name in available or presence_name in presence_args
                        or presence_name in data.get("reserved", ())):
                    raise ValueError(f"presence argument name collides with a declared name: {presence_name}")
                presence_args[presence_name] = arg_name
                expanded.append(presence_name)
        kernel_args = expanded
    data["_presence_args"] = presence_args
    data["_kernel_args"] = kernel_args
    data["_ranks"] = ranks
    data["_launch_scalars"] = launch_scalars
    # Constexpr constructor scalars stay in kernel_args (the frontend signature still
    # declares them) but leave the runtime kernel ABI: the CUDA launch and binding
    # argument lists below use only these runtime names.
    data["_constexpr_constructor"] = constexpr_values
    # Derived selectors keep the same runtime-ABI behavior as literal
    # constexpr args (their name never reaches the launcher), but their value
    # is bound per Model instance and fans the export out into one baked
    # kernel variant per domain value.
    data["_constexpr_derived"] = derived_constexpr
    derived_names = {item["name"] for item in derived_constexpr}
    data["_constexpr_names"] = set(constexpr_values) | derived_names
    data["_runtime_kernel_args"] = [
        name for name in kernel_args if name not in constexpr_values and name not in derived_names
    ]
    data["_tensor_types"] = tensor_types
    data["_constructor_types"] = constructor_types
    data["_descriptor_names"] = descriptor_names
    # Bundle descriptor names map to a device CUtensorMap array transport
    # (CUtensorMap* kernel arg) instead of a by-value grid_constant map.
    data["_descriptor_bundles"] = {item["name"]: item for item in descriptors if item.get("_bundle")}
    data["_scalar_types"] = {item["name"]: item["_dtype"] for item in scalars}
    # Optional multi-kernel pipeline: a root "pipeline" record declares the
    # ordered launch plan for forward. Each stage names an entry kernel in the
    # same draft.py, its own kernel_args drawn from the same declared pool as
    # the top-level kernel_args, and its own launch record validated with the
    # same rules as the single-launch record (grid/block shape expressions
    # resolve against the same tensor ranks and scalar/constructor names).
    # The binding launches the stages sequentially on the current stream, so
    # stream order is the happens-before edge between a stage's writes and a
    # later stage's reads (e.g. a split-reduction workspace).
    pipeline = data.get("pipeline")
    pipeline_stages = []
    if pipeline is not None:
        if not isinstance(pipeline, dict) or not isinstance(pipeline.get("stages"), list) \
                or not pipeline["stages"]:
            raise ValueError("pipeline must be an object with a nonempty stages list")
        reject_unknown_fields(pipeline, PIPELINE_KEYS, "pipeline")
        seen_stage_kernels = set()
        for stage in pipeline["stages"]:
            if not isinstance(stage, dict):
                raise ValueError("pipeline stages must be objects")
            reject_unknown_fields(stage, PIPELINE_STAGE_KEYS, "pipeline stage")
            stage_kernel = _identifier(stage.get("kernel"), "pipeline stage kernel")
            if stage_kernel in seen_stage_kernels:
                raise ValueError("pipeline stage kernels must be unique")
            seen_stage_kernels.add(stage_kernel)
            stage_args = stage.get("kernel_args")
            if (not isinstance(stage_args, list) or not stage_args
                    or any(not isinstance(x, str) or x not in available for x in stage_args)
                    or len(stage_args) != len(set(stage_args))):
                raise ValueError("pipeline stage kernel_args must list distinct declared "
                                 "tensors, scalars or descriptors")
            stage_launch = stage.get("launch")
            if not isinstance(stage_launch, dict):
                raise ValueError("pipeline stage launch must be an object")
            _validate_launch(stage_launch, ranks, launch_scalars, optional_inputs, host_lists)
            pipeline_stages.append({"kernel": stage_kernel, "kernel_args": stage_args, "launch": stage_launch})
        if kernel not in seen_stage_kernels:
            raise ValueError("pipeline stages must include the top-level kernel exactly once")
        if optional_inputs:
            # Presence-channel plumbing (injected _present args / nullptr
            # transport) is defined for the single-entry ABI only; reject here
            # rather than silently mis-binding stage launches.
            raise ValueError("pipeline cannot be combined with optional inputs")
        if data.get("foreach"):
            raise ValueError("pipeline cannot be combined with foreach")
    data["_pipeline_stages"] = pipeline_stages
    foreach = data.get("foreach")
    if foreach is not None and tensor_lists:
        raise ValueError("foreach cannot be combined with tensor_lists")
    if foreach is not None:
        if optional_inputs:
            # The grouped foreach binding threads per-group tensors through
            # list plumbing that never rebinds optional names behind
            # .defined(), so the combination is rejected rather than emitting
            # bindings that silently mis-handle None.
            raise ValueError("foreach cannot be combined with optional inputs")
        if not isinstance(foreach, dict) or not isinstance(foreach.get("inputs"), list):
            raise ValueError("foreach requires a list of tensor input names")
        reject_unknown_fields(foreach, FOREACH_KEYS, "foreach")
        names = foreach["inputs"]
        if not names or len(names) != len(set(names)) or any(
                name not in [x["name"] for x in inputs if x.get("_kind", "tensor") == "tensor"] for name in names):
            raise ValueError("foreach.inputs must name distinct tensor inputs")
        mode = foreach.get("mode", "loop")
        if mode not in ("loop", "single_launch"):
            raise ValueError("foreach.mode must be 'loop' or 'single_launch'")
        if len(outputs) > 1 and mode == "loop":
            # Multiple loop-mode foreach outputs are legal only when each
            # output aliases a distinct foreach input: the grouped binding
            # then returns one list per output (tuple-of-lists), preserving
            # the input structure exactly (aliased in-place outputs, no extra
            # kernel work). single_launch allocates/collects every output
            # per element in the fused table, so mixed aliased/fresh outputs
            # are legal there.
            used = []
            for item in outputs:
                alias = item.get("alias_of")
                if alias not in names or alias in used:
                    raise ValueError("foreach with multiple outputs requires every output to "
                                     "alias_of a distinct foreach input")
                used.append(alias)
        if optional_input_names:
            raise ValueError("foreach cannot be combined with optional inputs")
        data["_foreach_inputs"] = names
        # There are two single-launch foreach variants, selected by disjoint
        # manifest keys:
        # * mode "single_launch": per-element scalars/pointers ride a
        #   fixed-capacity by-value table struct, element index on grid.z;
        #   per-element grid limits clip smaller elements early.
        # * dispatch "single": per-group block-prefix, pointer and
        #   group-scalar tables packed in kernel parameter space; one launch,
        #   blocks binary-search the prefix table.
        # Both reuse the grouped/loop path and entry kernel body verbatim;
        # a manifest selects at most one.
        data["_foreach_single_launch"] = mode == "single_launch"
        max_group = foreach.get("max_group", _FOREACH_FUSED_MAX)
        if (type(max_group) is not int or not 1 <= max_group <= _FOREACH_FUSED_MAX):
            raise ValueError(f"foreach.max_group must be an int in [1, {_FOREACH_FUSED_MAX}]")
        data["_foreach_max_group"] = max_group
        if data["_foreach_single_launch"]:
            if launch["_cluster"] is not None:
                raise ValueError("single_launch foreach does not support cluster_shape")
            for key in ("transforms", "pre_transforms", "post_transforms", "workspaces", "tma_descriptors"):
                if data[key]:
                    raise ValueError(f"single_launch foreach does not support {key}")
        dispatch = foreach.get("dispatch", "grouped")
        if dispatch not in ("grouped", "single"):
            raise ValueError('foreach.dispatch must be "grouped" or "single"')
        if dispatch == "single" and data["_foreach_single_launch"]:
            raise ValueError('foreach.dispatch "single" cannot be combined with '
                             'mode "single_launch"')
        data["_foreach_dispatch"] = dispatch
        # Group-dependent scalars are those whose expression transitively
        # references a foreach tensor (directly, e.g. `params[0]`, or through
        # another scalar name): their value differs per group and therefore
        # travels in the packed per-group tables rather than as a launch arg.
        group_scalars = set()
        changed = True
        foreach_set = set(names)
        while changed:
            changed = False
            for item in scalars:
                if item["_runtime"] or item["name"] in group_scalars:
                    continue
                refs = set(re.findall(r"[A-Za-z_]\w*", item.get("_expr_cpp", "")))
                if refs & foreach_set or refs & group_scalars:
                    group_scalars.add(item["name"])
                    changed = True
        data["_foreach_group_scalars"] = group_scalars
        if dispatch == "single":
            _validate_foreach_single(data)
    else:
        data["_foreach_inputs"] = []
        data["_foreach_single_launch"] = False
        data["_foreach_dispatch"] = "grouped"
        data["_foreach_group_scalars"] = set()
    if derived_constexpr and data["_foreach_inputs"]:
        # The variant fanout wraps one entry kernel per selector value; the
        # foreach repackaging paths assume a single entry kernel body.
        raise ValueError("derived constexpr selectors cannot be combined with a foreach group")
    # Container projections and ignored forward args: the container name is a
    # new caller-facing ABI slot and must not collide with any declared name;
    # ignored args are accept-and-drop slots.
    reserved = set(ranks) | set(constructor_types) | scalar_names | descriptor_names | {"output", "stream"}
    forward_containers = []
    for name, (container, key) in input_from.items():
        if container in reserved:
            raise ValueError(f"input {name} container name collides with a declared name")
        if container not in forward_containers:
            forward_containers.append(container)
    ignored = data.get("ignored_forward_args", [])
    if not isinstance(ignored, list):
        raise ValueError("ignored_forward_args must be a list")
    ignored = [_identifier(name, "ignored forward argument") for name in ignored]
    if (any(name in reserved or name in forward_containers for name in ignored) or len(set(ignored)) != len(ignored)):
        raise ValueError("ignored forward arguments must be unique and differ from "
                         "declared tensors, scalars, descriptors, constructors, and containers")
    if forward_containers and forward_args is not None:
        raise ValueError("forward_args cannot be combined with container-projection inputs")
    if forward_containers and data["_foreach_inputs"]:
        raise ValueError("foreach cannot be combined with container-projection inputs")
    data["_forward_containers"] = forward_containers
    data["_input_from"] = input_from
    data["_ignored_forward_args"] = ignored
    data["_kernel"] = kernel
    data["_function"] = function
    return data


def _load_spec(path: Path) -> dict:
    """Compatibility entry point: load, validate and resolve a manifest file.

    New code should use :func:`little_kernel.backends.nv.common.export.load_spec`, which returns
    the immutable typed :class:`~little_kernel.backends.nv.common.export.ResolvedSpec`; this
    returns a fresh mutable mapping view for transition consumers.
    """
    return _resolve(json.loads(Path(path).read_text()))
