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
"""Single ownership of the export manifest schema.

This module owns every manifest spelling the generic export API accepts:
identifier rules, the dtype/transport tables, the top-level key allowlist and
the nested record allowlists. Validation, launch planning, CUDA emission and
host bindings all import their spellings from here, so parser validation can
never drift from an independently maintained allowlist.
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*\Z")

_DTYPES = {
    "float16": ("__half", "at::kHalf", "at::kHalf"),
    "bfloat16": ("__nv_bfloat16", "at::kBFloat16", "at::kBFloat16"),
    "float32": ("float", "at::kFloat", "at::kFloat"),
    "float64": ("double", "at::kDouble", "at::kDouble"),
    "int32": ("int32_t", "at::kInt", "at::kInt"),
    "int64": ("int64_t", "at::kLong", "at::kLong"),
    # int8/uint8/int16/uint16 tensor element types have distinct device and
    # torch scalar-type spellings; every mapping is listed exactly once here.
    # Complex tensors are exposed to CUDA kernels as pointers to interleaved
    # real/imaginary components; shapes and ranks count complex elements.
    "complex64": ("float", "at::kComplexFloat", "at::kComplexFloat"),
    "complex128": ("double", "at::kComplexDouble", "at::kComplexDouble"),
    # Byte-addressed 8-bit storage dtypes: kernels see uint8_t*; the wrapper
    # allocates views with the fp8 dtype for torch-level comparisons.
    "float8_e4m3fn": ("uint8_t", "at::kFloat8_e4m3fn", "at::kFloat8_e4m3fn"),
    "float8_e5m2": ("uint8_t", "at::kFloat8_e5m2", "at::kFloat8_e5m2"),
    # One-byte logical payloads: torch.bool flags and byte views/int8
    # codebooks for quantized-value I/O (at::Char/at::Byte). The wrapper
    # allocates the matching torch dtype, and torch kBool storage is one
    # byte per element holding 0/1.
    "bool": ("bool", "at::kBool", "at::kBool"),
    # Keep bool distinct from int8/uint8 at the ABI level: the void*
    # transport cast in _cuda_kernel_arg is keyed by the frontend's declared
    # tensor element type (spec["_declared_dtypes"], not this storage table),
    # so a draft declaring ll.Tensor[ll.bool_] receives bool* while the
    # legacy byte-view pattern (spec dtype "bool", draft declaring
    # ll.Tensor[ll.uint8]) still receives uint8_t*. A naive shared mapping
    # here cannot serve both, because C++ has no implicit conversion between
    # uint8_t* and bool*: the cast must equal the generated kernel
    # signature's parameter type or the translation unit fails to compile.
    "int8": ("int8_t", "at::kChar", "at::kChar"),
    "uint8": ("uint8_t", "at::kByte", "at::kByte"),
    # Two-byte integer payloads: genuine torch.int16/torch.uint16 storage
    # semantics (at::kShort / at::kUInt16 are distinct scalar types in the
    # pinned c10), not byte-pair reinterpretations. The device side already
    # maps ll int16/uint16 to int16_t/uint16_t (codegen_cuda IntType
    # bit_map), so the entry signature and this table agree by construction.
    "int16": ("int16_t", "at::kShort", "at::kShort"),
    "uint16": ("uint16_t", "at::kUInt16", "at::kUInt16"),
}

_CONSTRUCTOR_CPP = {
    "int32": "int32_t",
    "int64": "int64_t",
    "float32": "float",
    "bool": "bool",
}

_SCALAR_DTYPES = ("int32", "int64", "float32", "float64", "complex64", "complex128")

_SCALAR_COMPONENT = {"complex64": "float", "complex128": "double"}

_SCALAR_RUNTIME_CPP = {
    "int32": "int64_t",
    "int64": "int64_t",
    "float32": "float",
    "float64": "double",
    "complex64": "std::complex<float>",
    "complex128": "std::complex<double>",
}

_TMA_DTYPES = {
    "float16": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT16", 2),
    "bfloat16": ("CU_TENSOR_MAP_DATA_TYPE_BFLOAT16", 2),
    "float32": ("CU_TENSOR_MAP_DATA_TYPE_FLOAT32", 4),
    "int32": ("CU_TENSOR_MAP_DATA_TYPE_INT32", 4),
    # 1-byte payloads move through TMA as raw bytes: the fp8 storage dtypes
    # are only interpreted as e5m2/e4m3 by the consumer (e.g. WGMMA), so the
    # descriptor carries the UINT8 tensor-map dtype and the kernel keeps its
    # byte view (the storage table above already exposes fp8 pointers as
    # uint8_t*). element_size stays 1 so the emitted `stride * element_size`
    # global-stride multiplication in agent_lk_make_tma_{2d,3d} remains a
    # byte count; a 128-column fp8 tile then boxes 128 bytes, matching
    # CU_TENSOR_MAP_SWIZZLE_128B.
    "uint8": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 1),
    "float8_e4m3fn": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 1),
    "float8_e5m2": ("CU_TENSOR_MAP_DATA_TYPE_UINT8", 1),
}

_TMA_SWIZZLES = {
    0: "CU_TENSOR_MAP_SWIZZLE_NONE",
    32: "CU_TENSOR_MAP_SWIZZLE_32B",
    64: "CU_TENSOR_MAP_SWIZZLE_64B",
    128: "CU_TENSOR_MAP_SWIZZLE_128B",
}

_TMA_DESCRIPTOR_FIELDS = frozenset({
    "name",
    "tensor",
    "rank",
    "inner",
    "outer",
    "outer_stride",
    "dim0",
    "dim1",
    "dim2",
    "stride1",
    "stride2",
    # Rank-4 descriptors (e.g. a BSHD attention operand viewed innermost-out
    # as (D, S, H, B)) extend the rank-3 field pattern with one more
    # dimension/stride pair; the hardware tensor-map cap is rank 5.
    "dim3",
    "stride3",
    "box",
    "swizzle_bytes",
    "oob_fill",
    # flatten_leading collapses all leading tensor dimensions into one outer
    # axis (canonical row-major [leading product, last dim] view), so a
    # rank-2 TMA descriptor can serve a rank-3/rank-4 tensor.
    "flatten_leading",
    # Per-batch descriptor bundle (MapBundle): ``batch_count`` host shape
    # expression selects the number of identical per-slice maps encoded into
    # one device-resident array; ``batch_stride`` (elements between slices)
    # defaults to the contiguous leading-axis slice size.
    "batch_count",
    "batch_stride",
})


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value) or keyword.iskeyword(value):
        raise ValueError(f"{label} must be a Python/C++ identifier")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


class SchemaError(ValueError):
    """A manifest violates the single owned export schema.

    Schema violations are ordinary ``ValueError`` subclasses so legacy
    ``except ValueError`` contracts keep their exact failure behavior; the
    distinct type lets typed API callers distinguish schema problems from
    later lowering failures.
    """


# Every top-level export.json key the generic export API understands. The
# underscore-prefixed ``_*`` entries are derived view fields produced by
# validation and are listed here so the legacy mapping view round-trips; raw
# input manifests may only carry the public names.
TOP_LEVEL_KEYS: frozenset[str] = frozenset({
    "kernel",
    "function",
    "inputs",
    "output",
    "outputs",
    "constructor",
    "scalars",
    "workspaces",
    "transforms",
    "pre_transforms",
    "post_transforms",
    "tma_descriptors",
    "launch",
    "equalities",
    "tensor_lists",
    "kernel_args",
    "pipeline",
    "foreach",
    "no_alias",
    "wrap_single_as_tuple",
    "forward_unpack",
    "passthrough_forward_args",
    "forward_args",
    "ignored_forward_args",
    "reserved",
})

# Record-level field allowlists. One owner, shared by validation and every
# emitter; an unknown field is rejected rather than silently ignored.
INPUT_KEYS: frozenset[str] = frozenset({
    "name",
    "kind",
    "dtype",
    "rank",
    "contiguous",
    "optional",
    "from",
    "key",
})
CONSTRUCTOR_KEYS: frozenset[str] = frozenset({
    "name",
    "dtype",
    "constexpr",
    "default",
})
SCALAR_KEYS: frozenset[str] = frozenset({"name", "dtype", "runtime", "expr"})
OUTPUT_KEYS: frozenset[str] = frozenset({
    "name",
    "dtype",
    "shape",
    "alias_of",
    "optional_if_none",
    "zero_init",
})
WORKSPACE_KEYS: frozenset[str] = frozenset({"name", "dtype", "shape", "zero"})
TENSOR_LIST_KEYS: frozenset[str] = frozenset({"name", "dtype", "rank", "max_len", "kernel_params"})
TRANSFORM_KEYS: frozenset[str] = frozenset({"op", "name", "source"})
CAST_KEYS: frozenset[str] = frozenset({"op", "source", "target"})
DESCRIPTOR_KEYS: frozenset[str] = _TMA_DESCRIPTOR_FIELDS
LAUNCH_KEYS: frozenset[str] = frozenset({
    "grid",
    "block",
    "block_cap",
    "block_max_threads",
    "shared_memory_bytes",
    "disable_var_hoisting",
    "max_blocks_per_sm",
    "cluster_shape",
    "min_blocks_per_sm",
})
FOREACH_KEYS: frozenset[str] = frozenset({"inputs", "mode", "max_group", "dispatch"})
PIPELINE_KEYS: frozenset[str] = frozenset({"stages"})
PIPELINE_STAGE_KEYS: frozenset[str] = frozenset({"kernel", "kernel_args", "launch"})

# The record tables the typed validated model exposes directly. Each tuple
# mirrors the declaration order of the manifest (an ABI-relevant property).
_RECORD_KEYS = {
    "input": INPUT_KEYS,
    "constructor": CONSTRUCTOR_KEYS,
    "scalar": SCALAR_KEYS,
    "output": OUTPUT_KEYS,
    "workspace": WORKSPACE_KEYS,
    "tensor_list": TENSOR_LIST_KEYS,
    "transform": TRANSFORM_KEYS,
    "cast": CAST_KEYS,
    "descriptor": DESCRIPTOR_KEYS,
    "launch": LAUNCH_KEYS,
    "foreach": FOREACH_KEYS,
    "pipeline_stage": PIPELINE_STAGE_KEYS,
}


def reject_unknown_fields(record: dict, allowed: frozenset[str], label: str) -> None:
    """Raise SchemaError on fields outside the single owned allowlist."""
    unknown = set(record) - allowed
    if unknown:
        raise SchemaError(f"unknown {label} fields: {sorted(unknown)}")


def schema_keys(kind: str) -> frozenset[str]:
    """Return the owned field allowlist for one record kind."""
    return _RECORD_KEYS[kind]


def check_identifier(value: object, label: str = "identifier") -> str:
    """Public identifier rule shared by every export-adjacent manifest."""
    return _identifier(value, label)


@dataclass(frozen=True)
class RawSpec:
    """The raw, unvalidated export specification.

    ``data`` is the parsed JSON object exactly as authored; nothing is derived
    or mutated. Call ``validate_spec`` to obtain the validated/resolved model.
    """

    data: dict[str, Any]

    @classmethod
    def from_object(cls, obj: Any) -> "RawSpec":
        if not isinstance(obj, dict):
            raise SchemaError("spec must be a JSON object")
        return cls(obj)

    @classmethod
    def from_path(cls, path) -> "RawSpec":
        return cls.from_object(json.loads(Path(path).read_text()))
