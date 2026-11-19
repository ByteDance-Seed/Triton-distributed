from little_kernel.language.intrin.identifiers import strip_codegen_identifier
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
"""Shared contracts for the canonical memory intrinsic families (RF10).

This module is the single owner of the rules that used to be copy/pasted
across the 6,922-line ``memory.py`` (and its divergent ``codegen/memory.py``
twin):

* by-reference output-lane / by-value input-lane type validation
  (:func:`make_gmem_ld_out_lane_type_check` /
  :func:`make_gmem_st_in_lane_type_check`);
* cache-policy alias normalization (:func:`normalize_cache_policy`,
  :data:`POLICY_ALIASES`);
* codegen-identifier quote stripping (:func:`strip_codegen_identifier`);
* an explicit, inspectable inventory of the canonical operation families:
  address space, operand/offset **units**, **packed bit-lanes vs numerical
  lanes**, alignment, cache policies, effects and target requirements.

Nothing here emits CUDA text; these contracts are imported by the owner
modules under :mod:`little_kernel.backends.nv.common.language.intrin.memory_ops` and by the
backward-compatible facades.  The lazy type-inference import inside the lane
validators keeps ``language.intrin`` free of any import-time dependency on
the inference package (import-cycle safety).
"""
from dataclasses import dataclass
from typing import Tuple

from little_kernel.core.type_system import LLType, int8, uint8, uint32, uint64, int32, float16, bfloat16, float32

# ---------------------------------------------------------------------------
# Cache policy
# ---------------------------------------------------------------------------
#
# PTX global-memory cache qualifiers and the user-facing aliases accepted by
# the policy-parameterized families (gmem_ld_v4/gmem_st_v4/gmem_ld_x2/
# gmem_ld_scalar/gmem_st_scalar).  Policies only change cache *allocation*;
# every policy is numerically bit-identical, except "nc" which additionally
# carries the CUDA __ldg read-only legality obligation.
POLICY_ALIASES = {
    "default": "wb",
    "normal": "wb",
    "cache_all": "ca",
    "l2_only": "cg",
    "evict_first": "cs",
    "last_use": "lu",
    "read_only": "nc",
    "no_cache": "cv",
    "write_through": "wt",
}

# Policy sets, per direction.
LOAD_POLICIES: Tuple[str, ...] = ("wb", "ca", "cg", "cs", "lu", "nc", "cv")
STORE_POLICIES: Tuple[str, ...] = ("wb", "cg", "cs", "wt")


def normalize_cache_policy(policy, allowed, default, caller):
    """Normalize a cache-policy argument arriving from codegen.

    ``policy`` reaches the codegen function as a C++ string literal (with
    quotes) or None when not supplied.  Returns the bare PTX policy name.
    A bad literal is rejected at codegen with the accepted set in the
    message; it is never silently reinterpreted (no silent fallback).
    """
    if policy is None:
        return default
    name = str(policy).strip().strip('"').strip("'")
    name = POLICY_ALIASES.get(name, name)
    if name not in allowed:
        raise ValueError(f"{caller}: cache policy '{name}' not in "
                         f"{sorted(allowed)} (aliases: {sorted(POLICY_ALIASES)})")
    return name


# Backward-compatible private spellings (pre-split module names).
_POLICY_ALIASES = POLICY_ALIASES
_normalize_cache_policy = normalize_cache_policy

strip_quotes = strip_codegen_identifier
_strip_quotes = strip_codegen_identifier


# ---------------------------------------------------------------------------
# By-reference lane type validation
# ---------------------------------------------------------------------------
def make_gmem_ld_out_lane_type_check(intrin_name: str, lane_type: LLType, leading_operands: int = 2):
    """eval_arg_type validator for a by-reference vector-load intrinsic.

    The codegen of every ``gmem_ld_*``/``gmem_vld_*`` load emits ``&lane``
    into a typed C++ pointer parameter (e.g. ``float*``), so each
    output-lane local must be declared with exactly the lane element type.
    A mismatched lane local otherwise compiles through the whole frontend
    and surfaces only as an nvcc invalid-pointer-conversion error; validating
    here turns it into a frontend compile-time diagnostic at the responsible
    call.

    ``leading_operands`` is the number of non-lane positional arguments
    before the lane locals (the address operands).  It is 2 for the normal
    ``(mem, index, *lanes)`` families, 1 for byte-addressed
    ``(gmem_ptr, *lanes)`` intrinsics (e.g. gmem_ld_v4_f32) and 3 for
    ``(mem, index, head, *lanes)`` shifted-window loads; only the trailing
    lane locals are type-checked.

    Legality rules: only ``ast.Name`` arguments already present in the
    current scope-type map are checked (anything else cannot be typed
    statically at this point and keeps legacy behavior).
    """

    def _check(ctx, *args, **_kwargs):
        # Imported lazily so language.intrin never depends on the inference
        # package at module import time (import-cycle safety).
        from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
        lanes = args[leading_operands:]
        for lane in lanes:
            if not isinstance(lane, str) or lane not in ctx:
                continue
            actual = ctx[lane]
            if actual != lane_type:
                raise TypeInferenceError(f"{intrin_name}: output lane '{lane}' has type {actual} "
                                         f"but this intrinsic writes through &{lane} with lane "
                                         f"type {lane_type}; declare '{lane}' as {lane_type} (or "
                                         f"use a differently named local of that type)")

    return _check


def make_gmem_st_in_lane_type_check(intrin_name: str, lane_type: LLType, leading_operands: int = 2):
    """eval_arg_type validator for a by-value vector-store intrinsic.

    Store-side mirror of :func:`make_gmem_ld_out_lane_type_check`: the
    codegen of a typed ``gmem_st_*`` store passes each value lane into a
    typed C++ value parameter (e.g. ``float r0``), so a lane local of the
    wrong type would otherwise only surface as an nvcc implicit-conversion
    result.  ``leading_operands`` has the same meaning as on the load-side
    factory (2 for ``(mem, index, *lanes)``, 1 for byte-addressed
    ``(gmem_ptr, *lanes)``).

    Same legality rules: only ``ast.Name`` arguments already present in the
    current scope-type map are checked.
    """

    def _check(ctx, *args, **_kwargs):
        from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
        lanes = args[leading_operands:]
        for lane in lanes:
            if not isinstance(lane, str) or lane not in ctx:
                continue
            actual = ctx[lane]
            if actual != lane_type:
                raise TypeInferenceError(f"{intrin_name}: input lane '{lane}' has type {actual} "
                                         f"but this intrinsic reads {lane} with lane "
                                         f"type {lane_type}; declare '{lane}' as {lane_type} (or "
                                         f"use a differently named local of that type)")

    return _check


# Backward-compatible private spellings (pre-split module names).
_make_gmem_ld_out_lane_type_check = make_gmem_ld_out_lane_type_check
_make_gmem_st_in_lane_type_check = make_gmem_st_in_lane_type_check

# ---------------------------------------------------------------------------
# Operation-family inventory
# ---------------------------------------------------------------------------
# Effect vocabulary (conservative, machine-readable).
EFFECT_LOAD = "load"  # reads global/shared memory
EFFECT_STORE = "store"  # writes global/shared memory
EFFECT_LOAD_STORE = "load_store"
EFFECT_NONE = "none"  # pure register/address arithmetic


@dataclass(frozen=True)
class MemoryOperationContract:
    """Declared contract of one canonical memory operation family.

    Attributes
    ----------
    representative:
        Canonical intrinsic name whose codegen/validator owns the family.
    aliases:
        Equivalent spellings that share the SAME codegen/validator owner
        and the identical numeric/bit behavior (cache-hint twins and
        suffix-spelling aliases).
    address_space:
        ``"global"`` or ``"shared"``.
    offset_unit:
        Units of the offset operand: ``"element"`` (C++ pointer arithmetic
        in the tensor's own dtype), ``"byte"`` (raw ``uint8_t*`` arithmetic)
        or ``"pair"`` (offset counts packed pairs, i.e. 2 elements).
    transaction_bytes:
        Size of the single memory transaction.
    lane_representation:
        ``"raw"`` lanes carry packed *bit patterns* with no numerical
        conversion (uint containers); ``"numerical"`` lanes carry real
        element values (and may widen/narrow with rounding).
    lane_count:
        Number of exchanged lanes/registers.
    lane_dtypes:
        Canonical LLType(s) a lane local must have.
    alignment_bytes:
        Hard alignment legality precondition of the addressed bytes.
    cache_policies:
        Accepted policy spellings for the family.
    effect:
        Declared memory effect.
    read_only_contract:
        Whether the read-only (``nc``/``__ldg``) path requires the loaded
        bytes to be unwritten for the whole launch.
    target:
        Backend target requirement.
    """
    representative: str
    aliases: Tuple[str, ...] = ()
    address_space: str = "global"
    offset_unit: str = "element"
    transaction_bytes: int = 16
    lane_representation: str = "raw"
    lane_count: int = 4
    lane_dtypes: Tuple[LLType, ...] = (uint32, )
    alignment_bytes: int = 16
    cache_policies: Tuple[str, ...] = ("wb", )
    effect: str = EFFECT_LOAD
    read_only_contract: bool = False
    target: str = "cuda"

    @property
    def members(self) -> Tuple[str, ...]:
        return (self.representative, ) + tuple(self.aliases)


def _c(representative, aliases=(), **kw):
    return MemoryOperationContract(representative=representative, aliases=tuple(aliases), **kw)


# Canonical family inventory.  Packed ("raw") families move bit containers
# and MUST NOT be merged with the numerically converting ("numerical")
# families even when their names look alike:
#   * gmem_ld_f32x4 / gmem_ld_u32x4 / gmem_ld_{f16,bf16}x8_u32x4  -> raw
#   * gmem_vld_f32x4 / gmem_ld_v4f32 / gmem_ld_{f16,bf16}x8_f32x8  -> numerical
_MEMORY_CONTRACTS = [
    # ---- raw 16-byte, four uint32 lanes, element offset ------------------
    _c("gmem_ld_f32x4", aliases=("gmem_ld_cs_f32x4", "gmem_ld_f32x4_cs", "gmem_ld_ro_f32x4"), lane_dtypes=(uint32, ),
       effect=EFFECT_LOAD, cache_policies=("wb", "cs", "nc"), read_only_contract=True),
    _c("gmem_st_f32x4", aliases=("gmem_st_cs_f32x4", "gmem_st_f32x4_cs"), lane_dtypes=(uint32, ), effect=EFFECT_STORE,
       cache_policies=("wb", "cs")),
    _c("gmem_ld_f32x2", aliases=("gmem_ld_cs_f32x2", "gmem_ld_f32x2_cs", "gmem_ld_ro_f32x2"), transaction_bytes=8,
       lane_count=2, lane_dtypes=(uint32, ), alignment_bytes=8, effect=EFFECT_LOAD, cache_policies=("wb", "cs", "nc"),
       read_only_contract=True),
    _c("gmem_st_f32x2", aliases=("gmem_st_cs_f32x2", "gmem_st_f32x2_cs"), transaction_bytes=8, lane_count=2,
       lane_dtypes=(uint32, ), alignment_bytes=8, effect=EFFECT_STORE, cache_policies=("wb", "cs")),
    # ---- numerical 16-byte float32 lanes ---------------------------------
    _c("gmem_vld_f32x4", aliases=("gmem_ld_v4f32", "gmem_ld_v4_f32", "gmem_ld_f32x4_val", "gmem_ld_cs_v4_f32",
                                  "gmem_ld_nc_v4_f32"), lane_representation="numerical", lane_dtypes=(float32, ),
       effect=EFFECT_LOAD, cache_policies=("wb", "cs", "nc"), read_only_contract=True),
    _c("gmem_vst_f32x4", aliases=("gmem_st_v4f32", "gmem_st_v4_f32", "gmem_st_f32x4_val", "gmem_st_v4f32_cg",
                                  "gmem_st_cs_v4_f32"), lane_representation="numerical", lane_dtypes=(float32, ),
       effect=EFFECT_STORE, cache_policies=("wb", "cg", "cs")),
    # ---- numerical int32 lanes -------------------------------------------
    _c("gmem_ld_v4s32", aliases=("gmem_ld_v4s32_nc", ), lane_representation="numerical", lane_dtypes=(int32, ),
       effect=EFFECT_LOAD, cache_policies=("wb", "nc"), read_only_contract=True),
    _c("gmem_st_v4s32", aliases=("gmem_st_v4s32_cg", ), lane_representation="numerical", lane_dtypes=(int32, ),
       effect=EFFECT_STORE, cache_policies=("wb", "cg")),
    # ---- raw, byte offset -------------------------------------------------
    _c("gmem_ld_u32x4", aliases=("gmem_ld_u32x4_cg", "gmem_ld_u32x4_cs"), offset_unit="byte", lane_dtypes=(uint32, ),
       effect=EFFECT_LOAD, cache_policies=("wb", "cg", "cs")),
    _c("gmem_st_u32x4", aliases=("gmem_st_u32x4_wt", ), offset_unit="byte", lane_dtypes=(uint32, ), effect=EFFECT_STORE,
       cache_policies=("wb", "wt")),
    _c("gmem_ld_u32x2", aliases=("gmem_ld_u32x2_cg", "gmem_ld_u32x2_cs", "gmem_ld_u32x2_nc"), offset_unit="element",
       transaction_bytes=8, lane_count=2, alignment_bytes=8, lane_dtypes=(uint32, ), effect=EFFECT_LOAD,
       cache_policies=("wb", "cg", "cs", "nc"), read_only_contract=True),
    _c("gmem_st_u32x2", aliases=("gmem_st_u32x2_cs", "gmem_st_u32x2_wt"), offset_unit="element", transaction_bytes=8,
       lane_count=2, alignment_bytes=8, lane_dtypes=(uint32, ), effect=EFFECT_STORE, cache_policies=("wb", "cs", "wt")),
    # ---- raw sixteen int8 lanes -------------------------------------------
    _c("gmem_ld_v16s8", aliases=("gmem_ld_v16s8_nc", ), lane_representation="numerical", lane_count=16,
       lane_dtypes=(int8, ), effect=EFFECT_LOAD, cache_policies=("wb", "nc"), read_only_contract=True),
    _c("gmem_st_v16s8", aliases=("gmem_st_v16s8_cs", ), lane_representation="numerical", lane_count=16,
       lane_dtypes=(int8, ), effect=EFFECT_STORE, cache_policies=("wb", "cs")),
    # ---- widened 16-bit -> 8 x float32 ------------------------------------
    _c("gmem_ld_v8f16", aliases=("gmem_ld_v8f16_nc", ), lane_representation="numerical", lane_count=8,
       lane_dtypes=(float32, ), effect=EFFECT_LOAD, cache_policies=("wb", "nc"), read_only_contract=True),
    _c("gmem_st_v8f16", aliases=("gmem_st_v8f16_cg", ), lane_representation="numerical", lane_count=8,
       lane_dtypes=(float32, ), effect=EFFECT_STORE, cache_policies=("wb", "cg")),
    _c("gmem_ld_v8bf16", aliases=("gmem_ld_v8bf16_nc", ), lane_representation="numerical", lane_count=8,
       lane_dtypes=(float32, ), effect=EFFECT_LOAD, cache_policies=("wb", "nc"), read_only_contract=True),
    _c("gmem_st_v8bf16", aliases=("gmem_st_v8bf16_cg", ), lane_representation="numerical", lane_count=8,
       lane_dtypes=(float32, ), effect=EFFECT_STORE, cache_policies=("wb", "cg")),
    # ---- raw 16-bit pairs as four uint32 ----------------------------------
    _c("gmem_ld_f16x8_u32x4", aliases=("gmem_ld_ro_f16x8", ), lane_dtypes=(uint32, ), effect=EFFECT_LOAD,
       cache_policies=("wb", "nc"), read_only_contract=True),
    _c("gmem_ld_bf16x8_u32x4", aliases=("gmem_ld_ro_bf16x8", "gmem_ld_cs_bf16x8"), lane_dtypes=(uint32, ),
       effect=EFFECT_LOAD, cache_policies=("wb", "cs", "nc"), read_only_contract=True),
    _c("gmem_st_f16x8_u32x4", lane_dtypes=(uint32, ), effect=EFFECT_STORE),
    _c("gmem_st_bf16x8_u32x4", lane_dtypes=(uint32, ), effect=EFFECT_STORE),
    # ---- raw packed pairs as two uint64 -----------------------------------
    _c("gmem_ld_bf16x8", aliases=("gmem_ld_bf16x8_cg", "gmem_ld_bf16x8_streaming"), lane_count=2,
       lane_dtypes=(uint64, ), effect=EFFECT_LOAD, cache_policies=("wb", "cg", "cs")),
    _c("gmem_ld_f16x8", aliases=("gmem_ld_f16x8_cg", "gmem_ld_f16x8_streaming"), lane_count=2, lane_dtypes=(uint64, ),
       effect=EFFECT_LOAD, cache_policies=("wb", "cg", "cs")),
    _c("gmem_st_bf16x8", aliases=("gmem_st_bf16x8_cg", "gmem_st_bf16x8_streaming"), lane_representation="numerical",
       lane_count=8, lane_dtypes=(float32, ), effect=EFFECT_STORE, cache_policies=("wb", "cg", "cs")),
    _c("gmem_st_f16x8", aliases=("gmem_st_f16x8_cg", "gmem_st_f16x8_streaming"), lane_representation="numerical",
       lane_count=8, lane_dtypes=(float32, ), effect=EFFECT_STORE, cache_policies=("wb", "cg", "cs")),
    _c("gmem_ld_u8x16", aliases=("gmem_ld_u8x16_cg", "gmem_ld_u8x16_streaming", "gmem_ld_u8x16_nc"), lane_count=2,
       lane_dtypes=(uint64, ), effect=EFFECT_LOAD, cache_policies=("wb", "cg", "cs", "nc"), read_only_contract=True),
    _c("gmem_st_u8x16", aliases=("gmem_st_u8x16_cg", "gmem_st_u8x16_streaming"), lane_count=2, lane_dtypes=(uint64, ),
       effect=EFFECT_STORE, cache_policies=("wb", "cg", "cs")),
    # ---- packed bf16x2 pair, offset counts pairs --------------------------
    _c("gmem_ld_bf16x2_u32", offset_unit="pair", transaction_bytes=4, lane_count=1, alignment_bytes=4,
       lane_dtypes=(uint32, ), effect=EFFECT_LOAD),
    _c("gmem_st_bf16x2_u32", offset_unit="pair", transaction_bytes=4, lane_count=1, alignment_bytes=4,
       lane_dtypes=(uint32, ), effect=EFFECT_STORE),
    # ---- scalar element access --------------------------------------------
    _c("gmem_ld_scalar", transaction_bytes=0, lane_count=1, lane_dtypes=(), alignment_bytes=0, effect=EFFECT_LOAD,
       cache_policies=LOAD_POLICIES, read_only_contract=True),
    _c("gmem_st_scalar", transaction_bytes=0, lane_count=1, lane_dtypes=(), alignment_bytes=0, effect=EFFECT_STORE,
       cache_policies=STORE_POLICIES),
    # ---- cp.async global -> shared ----------------------------------------
    _c("cp_async_16", address_space="global", transaction_bytes=16, lane_count=0, lane_dtypes=(),
       effect=EFFECT_LOAD_STORE, cache_policies=("ca", "cg"), alignment_bytes=16),
    _c("cp_async_wait_group", address_space="shared", transaction_bytes=0, lane_count=0, lane_dtypes=(),
       effect=EFFECT_NONE, cache_policies=(), alignment_bytes=0),
]

# Public lookup tables.
MEMORY_CONTRACTS = {m: c for c in _MEMORY_CONTRACTS for m in c.members}
CANONICAL_FAMILIES = tuple(c.representative for c in _MEMORY_CONTRACTS)

# Equivalent-family groups (canonical -> equivalent aliases), as tuples so the
# relationship is directly testable.  ``gmem_ld_v4f32`` vs
# ``gmem_vld_f32x4`` are the RF10 example: both deliver four numerical float32
# lanes (float4 dereference through ``float*``), so they share the float32
# out-lane validator; they remain distinct public builtins with distinct
# helper names/PTX-free C++ carriers.
EQUIVALENT_FAMILIES = {c.representative: c.aliases for c in _MEMORY_CONTRACTS if c.aliases}


def contract_for(name: str):
    """Return the declared :class:`MemoryOperationContract` for ``name``.

    Raises ``KeyError`` for an uninventoried name rather than guessing a
    contract from name substrings (no task-name dispatch).
    """
    return MEMORY_CONTRACTS[name]


__all__ = [
    "POLICY_ALIASES",
    "LOAD_POLICIES",
    "STORE_POLICIES",
    "normalize_cache_policy",
    "strip_codegen_identifier",
    "make_gmem_ld_out_lane_type_check",
    "make_gmem_st_in_lane_type_check",
    "MemoryOperationContract",
    "MEMORY_CONTRACTS",
    "CANONICAL_FAMILIES",
    "EQUIVALENT_FAMILIES",
    "contract_for",
    "EFFECT_LOAD",
    "EFFECT_STORE",
    "EFFECT_LOAD_STORE",
    "EFFECT_NONE",
]
