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
"""Typed immutable export records and resolved facts.

Validation produces two distinct, immutable layers:

* :class:`ValidatedSpec` holds the *authored* contract: declaration-ordered
  frozen records for tensor/host-list inputs, constructor arguments, scalars,
  outputs, workspaces, transforms, tensor lists and TMA descriptors, plus the
  authored launch/foreach/pipeline records.
* :class:`ResolvedSpec` adds the *derived* facts computed from that contract:
  the rank/type environment, resolved launch plan, constexpr plan, kernel-arg
  plan, optional/presence channels, forward-ABI permutation, foreach plan and
  pipeline stages.

Nothing here stores a mutable working dictionary. Container fields are tuples,
frozensets or deep-frozen mappings, so nested mutation
(``spec.launch.grid[0] = ...``) is impossible. The byte-stable CUDA/host
emitters predate these models and receive a *freshly reconstructed* mutable
mapping via :meth:`ResolvedSpec.legacy_mapping`, derived solely from these
immutable owners.
"""

from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from little_kernel.backends.nv.common.export._schema import check_identifier

_VALIDATED_BUILD_TOKEN = object()

# ---------------------------------------------------------------------------
# Deep immutability helpers
# ---------------------------------------------------------------------------


def freeze(value: Any) -> Any:
    """Return a deeply immutable copy of a JSON-like value.

    dicts become read-only mappings of frozen values, lists/tuples become
    tuples and sets become frozensets. Leaf scalars are returned unchanged.
    The result shares nothing mutable with the input.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


def thaw(value: Any) -> Any:
    """Inverse of :func:`freeze`: a fresh mutable copy for the emitter boundary."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, frozenset):
        return {thaw(item) for item in value}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return copy.deepcopy(value) if _is_mutable_leaf(value) else value


def _is_mutable_leaf(value: Any) -> bool:
    # Strings/ints/bools/None are immutable; anything unexpected is copied to
    # guarantee the caller cannot reach back into the validated model.
    return not isinstance(value, (str, bytes, int, float, bool, type(None)))


# ---------------------------------------------------------------------------
# Authored, declaration-ordered records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """A validated authored record.

    ``item`` is the deep-frozen, resolved record (authored fields plus the
    per-record derived render fields, e.g. ``shape_cpp``). Typed convenience
    properties expose the contract fields; :meth:`to_legacy_item` reconstructs
    the mutable record the byte-stable emitters expect.
    """

    item: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.item, Mapping):
            raise TypeError("record content must be a frozen mapping")
        # Defensive deep freeze so a caller cannot seed a record with mutable
        # nested state. object.__setattr__ is required because the dataclass is
        # frozen.
        object.__setattr__(self, "item", freeze(dict(self.item)))

    @property
    def name(self) -> str:
        return self.item["name"]

    def get(self, key: str, default: Any = None) -> Any:
        return self.item.get(key, default)

    def to_legacy_item(self) -> dict:
        return thaw(self.item)


@dataclass(frozen=True)
class TensorInput(Record):
    """A tensor (or host int-list / container-projection) input record."""

    @property
    def kind(self) -> str:
        return self.item.get("_kind", "tensor")

    @property
    def dtype(self) -> str:
        return self.item["dtype"]

    @property
    def rank(self) -> int:
        return self.item["rank"]

    @property
    def optional(self) -> bool:
        return bool(self.item.get("optional", False))

    @property
    def contiguous(self) -> bool:
        return bool(self.item.get("contiguous", True))

    @property
    def container(self) -> str | None:
        return self.item.get("from")

    @property
    def key(self) -> Any:
        return self.item.get("key")


@dataclass(frozen=True)
class ConstructorArg(Record):
    """A constructor argument: runtime, literal-constexpr or derived selector."""

    @property
    def dtype(self) -> str:
        return self.item["dtype"]

    @property
    def is_constexpr(self) -> bool:
        return "constexpr" in self.item

    @property
    def is_derived_constexpr(self) -> bool:
        return bool(self.item.get("_derived_constexpr", False))

    @property
    def has_default(self) -> bool:
        return "default" in self.item


@dataclass(frozen=True)
class ScalarDecl(Record):
    """A forward scalar: runtime argument or host-derived expression."""

    @property
    def dtype(self) -> str:
        return self.item["_dtype"]

    @property
    def runtime(self) -> bool:
        return bool(self.item["_runtime"])

    @property
    def expr_cpp(self) -> str | None:
        return self.item.get("_expr_cpp")


@dataclass(frozen=True)
class OutputDecl(Record):
    """An output tensor: fresh allocation, alias or conditionally present."""

    @property
    def dtype(self) -> str:
        return self.item["dtype"]

    @property
    def shape(self) -> tuple:
        return tuple(self.item["shape"])

    @property
    def shape_cpp(self) -> tuple:
        return tuple(self.item["_shape_cpp"])

    @property
    def alias_of(self) -> str | None:
        return self.item.get("alias_of")

    @property
    def is_self_alias(self) -> bool:
        return self.alias_of == self.name

    @property
    def zero_init(self) -> bool:
        return bool(self.item.get("zero_init", False))


@dataclass(frozen=True)
class WorkspaceDecl(Record):
    """A temporary workspace allocation."""

    @property
    def dtype(self) -> str:
        return self.item["dtype"]

    @property
    def shape_cpp(self) -> tuple:
        return tuple(self.item["_shape_cpp"])


@dataclass(frozen=True)
class TensorListDecl(Record):
    """A ragged list-of-tensors forward argument."""

    @property
    def dtype(self) -> str:
        return self.item["dtype"]

    @property
    def max_len(self) -> int:
        return self.item["max_len"]

    @property
    def kernel_params(self) -> tuple:
        return tuple(self.item["kernel_params"])


@dataclass(frozen=True)
class TmaDescriptor(Record):
    """A validated TMA descriptor (single map or per-batch bundle)."""

    @property
    def tensor(self) -> str:
        return self.item["tensor"]

    @property
    def descriptor_rank(self) -> int:
        return self.item["_rank"]

    @property
    def is_bundle(self) -> bool:
        return bool(self.item["_bundle"])

    @property
    def box(self) -> tuple:
        return tuple(self.item["_box"])


@dataclass(frozen=True)
class LaunchPlan:
    """The resolved launch contract for one entry kernel."""

    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", freeze(dict(self.record)))

    # -- authored ------------------------------------------------------------

    @property
    def grid(self) -> tuple:
        return tuple(self.record["grid"])

    @property
    def block(self) -> tuple:
        return tuple(self.record["block"])

    # -- resolved ------------------------------------------------------------

    @property
    def grid_cpp(self) -> tuple:
        return tuple(self.record["_grid_cpp"])

    @property
    def block_cpp(self) -> tuple:
        return tuple(self.record["_block_cpp"])

    @property
    def block_cap(self) -> tuple:
        return tuple(self.record["_block"])

    @property
    def threads(self) -> int:
        return self.record["_threads"]

    @property
    def dynamic_block(self) -> bool:
        return bool(self.record["_block_dynamic"])

    @property
    def cluster(self) -> tuple | None:
        value = self.record["_cluster"]
        return None if value is None else tuple(value)

    def to_legacy(self) -> dict:
        return thaw(self.record)


@dataclass(frozen=True)
class ConstexprPlan:
    """Literal constexpr values and derived selector fanout."""

    values: Mapping[str, Any]
    derived: tuple
    names: frozenset

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", freeze(self.values))
        object.__setattr__(self, "derived", freeze(self.derived))
        object.__setattr__(self, "names", frozenset(self.names))


@dataclass(frozen=True)
class ForeachPlan:
    """The resolved ragged-group foreach contract (or an empty plan)."""

    inputs: tuple = ()
    single_launch: bool = False
    dispatch: str = "grouped"
    max_group: int | None = None
    group_scalars: frozenset = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "group_scalars", frozenset(self.group_scalars))

    @property
    def active(self) -> bool:
        return bool(self.inputs)


@dataclass(frozen=True)
class PipelineStage:
    """One stage of a multi-kernel forward pipeline."""

    kernel: str
    kernel_args: tuple
    launch: LaunchPlan

    def __post_init__(self) -> None:
        object.__setattr__(self, "kernel_args", tuple(self.kernel_args))
        if not isinstance(self.launch, LaunchPlan):
            raise TypeError("pipeline stage launch must be a LaunchPlan")


# ---------------------------------------------------------------------------
# Validated authored state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedSpec:
    """The validated authored contract, independent of derived facts.

    The owned validation pipeline checks the manifest's semantic contracts.
    Construction additionally enforces identifiers and immutable record owners.
    """

    kernel: str
    function: str
    inputs: tuple
    outputs: tuple
    constructor: tuple
    scalars: tuple
    workspaces: tuple = ()
    transforms: tuple = ()
    pre_transforms: tuple = ()
    post_transforms: tuple = ()
    tensor_lists: tuple = ()
    descriptors: tuple = ()
    # Authored single-output spelling and boolean options.
    single_output: bool = False
    no_alias: bool = False
    wrap_single_as_tuple: bool = False
    _validation_key: InitVar[object] = None

    def __post_init__(self, _validation_key) -> None:
        check_identifier(self.kernel, "kernel")
        check_identifier(self.function, "function")
        if _validation_key is not _VALIDATED_BUILD_TOKEN:
            raise TypeError("ValidatedSpec is produced by load_spec()/validate(); validate the authored manifest")
        for label, members, expected in (("inputs", self.inputs, TensorInput), ("outputs", self.outputs, OutputDecl),
                                         ("constructor", self.constructor, ConstructorArg), ("scalars", self.scalars,
                                                                                             ScalarDecl)):
            members = tuple(members)
            object.__setattr__(self, label, members)
            if not all(isinstance(member, expected) for member in members):
                raise TypeError(f"{label} must contain typed {expected.__name__} records")
        for label, members in (("workspaces", self.workspaces), ("transforms", self.transforms),
                               ("pre_transforms", self.pre_transforms), ("post_transforms", self.post_transforms),
                               ("tensor_lists", self.tensor_lists), ("descriptors", self.descriptors)):
            members = tuple(members)
            object.__setattr__(self, label, members)
            if not all(isinstance(member, Record) for member in members):
                raise TypeError(f"{label} must contain typed records")

    # Typed convenience accessors used by adapters.
    @property
    def input_names(self) -> tuple:
        return tuple(record.name for record in self.inputs)

    @property
    def output_names(self) -> tuple:
        return tuple(record.name for record in self.outputs)


@dataclass(frozen=True)
class ResolvedFacts:
    """Derived, resolved facts computed from the validated authored state."""

    ranks: Mapping[str, int]
    tensor_types: Mapping[str, str]
    scalar_types: Mapping[str, str]
    constructor_types: Mapping[str, str]
    host_lists: Mapping[str, bool]
    optional_inputs: tuple
    present_inputs: frozenset
    presence_args: Mapping[str, str]
    kernel_args: tuple
    runtime_kernel_args: tuple
    launch: LaunchPlan
    launch_scalars: frozenset
    constexpr: ConstexprPlan
    foreach: ForeachPlan
    descriptor_names: frozenset
    descriptor_bundles: Mapping[str, TmaDescriptor]
    list_names: tuple
    list_param_types: Mapping[str, str]
    input_from: Mapping[str, tuple]
    forward_containers: tuple
    forward_pool: tuple
    forward_unpack: Mapping[str, tuple]
    forward_args: tuple | None
    passthrough: tuple
    ignored: tuple
    optional_trailing: bool
    equalities_cpp: tuple
    pipeline_stages: tuple

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranks", MappingProxyType(dict(self.ranks)))
        object.__setattr__(self, "tensor_types", MappingProxyType(dict(self.tensor_types)))
        object.__setattr__(self, "scalar_types", MappingProxyType(dict(self.scalar_types)))
        object.__setattr__(self, "constructor_types", MappingProxyType(dict(self.constructor_types)))
        object.__setattr__(self, "host_lists", MappingProxyType(dict(self.host_lists)))
        for label in ("optional_inputs", "kernel_args", "runtime_kernel_args", "launch_scalars", "descriptor_names",
                      "list_names", "forward_containers", "forward_pool", "passthrough", "ignored", "equalities_cpp",
                      "pipeline_stages"):
            object.__setattr__(self, label, tuple(getattr(self, label)))
        object.__setattr__(self, "present_inputs", frozenset(self.present_inputs))
        object.__setattr__(self, "list_param_types", MappingProxyType(dict(self.list_param_types)))
        object.__setattr__(self, "input_from", freeze(dict(self.input_from)))
        object.__setattr__(self, "presence_args", MappingProxyType(dict(self.presence_args)))
        object.__setattr__(self, "descriptor_bundles", MappingProxyType(dict(self.descriptor_bundles)))
        object.__setattr__(self, "forward_unpack", freeze(dict(self.forward_unpack)))
        if self.forward_args is not None:
            object.__setattr__(self, "forward_args", tuple(self.forward_args))


@dataclass(frozen=True)
class ResolvedSpec:
    """A validated spec together with its fully resolved, immutable facts.

    This is the object :func:`little_kernel.backends.nv.common.export.load_spec` returns. It owns
    typed immutable state only; the emitter compatibility view is rebuilt on
    demand by :meth:`legacy_mapping`.
    """

    authored: ValidatedSpec
    facts: ResolvedFacts
    # Immutable snapshots of the authored top-level sections that do not have
    # dedicated record types (equalities pairs and option/forward records).
    authored_top: Mapping[str, Any]
    extra: Mapping[str, Any] = field(default_factory=dict)
    _validation_key: InitVar[object] = None

    def __post_init__(self, _validation_key) -> None:
        if _validation_key is not _VALIDATED_BUILD_TOKEN:
            raise TypeError(
                "ResolvedSpec is produced by load_spec()/validate(); resolved facts cannot bypass validation")
        if not isinstance(self.authored, ValidatedSpec):
            raise TypeError("authored state must be a ValidatedSpec")
        if not isinstance(self.facts, ResolvedFacts):
            raise TypeError("resolved facts must be a ResolvedFacts")
        object.__setattr__(self, "authored_top", freeze(dict(self.authored_top)))
        object.__setattr__(self, "extra", freeze(dict(self.extra)))

    # -- public read-only accessors ----------------------------------------

    @property
    def kernel(self) -> str:
        return self.authored.kernel

    @property
    def function(self) -> str:
        return self.authored.function

    @property
    def ranks(self) -> Mapping[str, int]:
        return self.facts.ranks

    @property
    def tensor_dtypes(self) -> Mapping[str, str]:
        return self.facts.tensor_types

    @property
    def scalar_dtypes(self) -> Mapping[str, str]:
        return self.facts.scalar_types

    @property
    def optional_inputs(self) -> tuple:
        return self.facts.optional_inputs

    @property
    def kernel_args(self) -> tuple:
        return self.facts.kernel_args

    @property
    def runtime_kernel_args(self) -> tuple:
        return self.facts.runtime_kernel_args

    @property
    def foreach_inputs(self) -> tuple:
        return self.facts.foreach.inputs

    @property
    def launch(self) -> LaunchPlan:
        return self.facts.launch

    # -- emitter compatibility boundary -------------------------------------

    def legacy_mapping(self) -> dict:
        """Reconstruct a fresh mutable legacy mapping from typed owners.

        The returned dict is an independent working copy; renderers may attach
        their own derived fields (declared dtypes, shared-memory floor, packed
        foreach layout) without touching this validated model.
        """
        a = self.authored
        f = self.facts
        top = self.authored_top
        view: dict[str, Any] = {}
        view["kernel"] = a.kernel
        view["function"] = a.function
        view["inputs"] = [record.to_legacy_item() for record in a.inputs]
        view["constructor"] = [record.to_legacy_item() for record in a.constructor]
        output_items = [record.to_legacy_item() for record in a.outputs]
        if a.single_output:
            view["output"] = output_items[0]
        else:
            view["outputs"] = output_items
        view["_outputs"] = output_items
        view["scalars"] = [record.to_legacy_item() for record in a.scalars]
        view["workspaces"] = [record.to_legacy_item() for record in a.workspaces]
        view["transforms"] = [record.to_legacy_item() for record in a.transforms]
        view["pre_transforms"] = [record.to_legacy_item() for record in a.pre_transforms]
        view["post_transforms"] = [record.to_legacy_item() for record in a.post_transforms]
        view["tensor_lists"] = [record.to_legacy_item() for record in a.tensor_lists]
        view["tma_descriptors"] = [record.to_legacy_item() for record in a.descriptors]
        view["launch"] = f.launch.to_legacy()
        # Authored top-level sections retained verbatim (frozen snapshots).
        for key, value in top.items():
            view[key] = thaw(value)
        # Resolved derived facts.
        view["_ranks"] = dict(f.ranks)
        view["_tensor_types"] = dict(f.tensor_types)
        view["_scalar_types"] = dict(f.scalar_types)
        view["_constructor_types"] = dict(f.constructor_types)
        view["_host_lists"] = dict(f.host_lists)
        view["_optional_inputs"] = list(f.optional_inputs)
        view["_present_inputs"] = set(f.present_inputs)
        view["_presence_args"] = dict(f.presence_args)
        view["_kernel_args"] = list(f.kernel_args)
        view["_runtime_kernel_args"] = list(f.runtime_kernel_args)
        view["_launch_scalars"] = set(f.launch_scalars)
        view["_constexpr_constructor"] = dict(f.constexpr.values)
        view["_constexpr_derived"] = thaw(f.constexpr.derived)
        view["_constexpr_names"] = set(f.constexpr.names)
        view["_descriptor_names"] = set(f.descriptor_names)
        view["_descriptor_bundles"] = {name: bundle.to_legacy_item() for name, bundle in f.descriptor_bundles.items()}
        view["_list_names"] = list(f.list_names)
        view["_list_param_types"] = dict(f.list_param_types)
        view["_input_from"] = {name: tuple(pair) for name, pair in f.input_from.items()}
        view["_forward_containers"] = list(f.forward_containers)
        view["_forward_pool"] = list(f.forward_pool)
        view["_forward_unpack"] = {name: list(members) for name, members in f.forward_unpack.items()}
        view["_forward_args"] = None if f.forward_args is None else list(f.forward_args)
        view["_passthrough_forward_args"] = list(f.passthrough)
        view["_ignored_forward_args"] = list(f.ignored)
        view["_optional_trailing"] = f.optional_trailing
        view["_equalities_cpp"] = [tuple(pair) for pair in f.equalities_cpp]
        view["_pipeline_stages"] = [{
            "kernel": stage.kernel, "kernel_args": list(stage.kernel_args), "launch": stage.launch.to_legacy()
        } for stage in f.pipeline_stages]
        view["_foreach_inputs"] = list(f.foreach.inputs)
        view["_foreach_single_launch"] = f.foreach.single_launch
        view["_foreach_dispatch"] = f.foreach.dispatch
        if f.foreach.max_group is not None:
            view["_foreach_max_group"] = f.foreach.max_group
        view["_foreach_group_scalars"] = set(f.foreach.group_scalars)
        view["_no_alias"] = a.no_alias
        view["_wrap_single_as_tuple"] = a.wrap_single_as_tuple
        view["_kernel"] = a.kernel
        view["_function"] = a.function
        for key, value in self.extra.items():
            view[key] = thaw(value)
        return view
