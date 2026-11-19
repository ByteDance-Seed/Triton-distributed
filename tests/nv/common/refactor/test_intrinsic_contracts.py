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

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""RF10 canonical memory-intrinsic contract tests.

Covers:

* facade identity (``intrin.memory`` and ``codegen.memory`` re-export the one
  canonical implementation, with no divergent copy);
* canonical builtin registration and metadata preservation;
* equivalent-family validator unification, including the RF10 example
  ``gmem_ld_v4f32`` vs ``gmem_vld_f32x4`` (both numerical float32 lanes) and
  their distinction from the genuinely different raw-bit family
  ``gmem_ld_f32x4`` (uint32 bit-pattern lanes);
* operand units (byte vs element vs pair offsets), packed vs numerical lane
  representations, alignment/cache-policy/effects from the shared contracts;
* cache-policy normalization (no silent fallback on a bad policy);
* representative emitted PTX/C++ helper identity (numeric behavior, not just
  names).
"""
import pytest

from little_kernel.language.builtin_base import BUILTIN_ATTR, CODEGEN_FUNC_ATTR, EVAL_ARG_TYPE_ATTR, EVAL_RETURN_TYPE_ATTR, get_intrin_ctx
import little_kernel.backends.nv.common.language.intrin.contracts as contracts
import little_kernel.backends.nv.common.language.intrin.memory as memory
import little_kernel.backends.nv.common.language.intrin.memory_ops as ops
from little_kernel.core.type_system import float32, int32, uint32, bfloat16, float16

TypeInferenceError = pytest.importorskip(
    "little_kernel.core.passes.utils.type_inference.type_inference_core").TypeInferenceError


# --------------------------------------------------------------------------- #
# Facade identity
# --------------------------------------------------------------------------- #
def test_memory_facade_reexports_every_canonical_name_by_identity():
    names = ops.__all__
    assert len(names) >= 500  # 545 at split time
    for name in names:
        canonical = getattr(ops, name)
        facade = getattr(memory, name)
        assert facade is canonical, f"facade duplicated {name!r}"


def test_facade_has_no_private_implementation_duplicates():
    # The canonical codegen owner and the facade attribute must be the very
    # same function object (one maintained implementation per operation).
    pairs = [
        ("codegen_gmem_ld_v4f32", ),
        ("codegen_gmem_vld_f32x4", ),
        ("codegen_alloc_shared_memory", ),
        ("codegen_cp_async_16", ),
        ("_normalize_cache_policy", ),
    ]
    for (name, ) in pairs:
        owner = getattr(ops.global_vector if name.startswith("codegen_gmem") else ops, name, None) or getattr(ops, name)
        assert getattr(memory, name) is owner


def test_codegen_memory_is_a_canonical_reexport_not_a_copy():
    import little_kernel.backends.nv.common.codegen.memory_compat as cm
    # The divergent historical copies were retired: every historical
    # codegen.memory name now resolves to the canonical implementation.
    for name in [
            "alloc_dynamic_shared_memory", "alloc_local_memory", "alloc_shared_memory", "empty", "zeros", "full",
            "ptr_byte_offset", "cvta_generic_to_shared", "stmatrix_x4", "load_128bit", "store_128bit",
            "codegen_alloc_shared_memory", "codegen_empty", "_ZERO_LITERAL", "_INIT_LIST_THRESHOLD"
    ]:
        assert getattr(cm, name) is getattr(memory, name), name
    # The canonical module is documented and exposed.
    assert cm.canonical_memory is memory


def test_builtin_registry_contains_facade_objects_once():
    ctx = get_intrin_ctx()
    for name in ["gmem_ld_v4f32", "gmem_vld_f32x4", "gmem_ld_f32x4", "alloc_shared_memory", "cp_async_16"]:
        assert ctx[name] is getattr(memory, name)


# --------------------------------------------------------------------------- #
# Equivalent-family validator unification
# --------------------------------------------------------------------------- #
def _arg_validator(fn):
    return getattr(fn, EVAL_ARG_TYPE_ATTR)


def test_numerical_float_family_validators_accept_float_and_reject_int():
    # gmem_ld_v4f32 and gmem_vld_f32x4 are the RF10 equivalent family: both
    # deliver four NUMERICAL float32 lanes via float* out parameters, so both
    # share the float32 out-lane validator.
    for fn in [memory.gmem_ld_v4f32, memory.gmem_vld_f32x4, memory.gmem_ld_cs_v4_f32, memory.gmem_ld_nc_v4_f32]:
        check = _arg_validator(fn)
        assert check is not None
        # correct float lanes pass
        check({"a": float32, "b": float32}, "x", "i", "a", "b", "a", "b")
        # an int32 lane fed to a float* lane fails loudly at the call
        with pytest.raises(TypeInferenceError) as exc:
            check({"d": int32}, "x", "i", "d", "d", "d", "d")
        msg = str(exc.value)
        assert fn.__name__ in msg and "int32" in msg and "float32" in msg


def test_byte_addressed_float_family_uses_one_leading_operand():
    # gmem_ld_v4_f32 is byte/pointer addressed: (gmem_ptr, f0..f3), so its
    # validator must skip exactly ONE leading operand and validate the four
    # trailing float lanes (regression for the (mem,index,head,...) shift trap).
    check = _arg_validator(memory.gmem_ld_v4_f32)
    check({"a": float32}, "ptr", "a", "a", "a", "a")
    with pytest.raises(TypeInferenceError):
        check({"d": int32}, "ptr", "d", "d", "d", "d")


def test_shifted_window_three_leading_operands():
    # gmem_ld_bf16x8_shifted is (mem, index, shift, f0..f7): the third int32
    # operand is a shift amount, NOT a lane.
    check = _arg_validator(memory.gmem_ld_bf16x8_shifted)
    lanes = {f"s{k}": float32 for k in range(8)}
    lanes["shift"] = int32  # the non-lane shift operand
    check(lanes, "vec", "base", "shift", *[f"s{k}" for k in range(8)])
    bad = dict(lanes, s0=int32)
    with pytest.raises(TypeInferenceError):
        check(bad, "vec", "base", "shift", *[f"s{k}" for k in range(8)])


def test_raw_bit_family_is_distinct_from_numerical_family():
    # gmem_ld_f32x4 moves RAW uint32 bit patterns through uint32_t* lanes, so
    # it must require uint32 lanes -- the opposite legality from the
    # numerical float32 families above.
    raw = _arg_validator(memory.gmem_ld_f32x4)
    raw({"q": uint32}, "x", "i", "q", "q", "q", "q")  # raw uint32 lanes OK
    with pytest.raises(TypeInferenceError) as exc:
        raw({"r": float32}, "x", "i", "r", "r", "r", "r")
    assert "uint32" in str(exc.value) and "float32" in str(exc.value)
    # and it is a different validator object from the numerical family
    num = _arg_validator(memory.gmem_ld_v4f32)
    assert raw is not num


def test_signed_int_lane_family_preserved():
    check = _arg_validator(memory.gmem_ld_v4s32)
    check({"q": int32}, "x", "i", "q", "q", "q", "q")
    with pytest.raises(TypeInferenceError):
        check({"a": float32}, "x", "i", "a", "a", "a", "a")
    # the by-value store twin gained the symmetric validator in RF10
    st_check = _arg_validator(memory.gmem_st_v4s32)
    assert st_check is not None
    with pytest.raises(TypeInferenceError):
        st_check({"a": float32}, "x", "i", "a", "a", "a", "a")


def test_untyped_or_unknown_lane_keeps_legacy_behavior():
    # A non-Name argument or a name not yet in the scope map cannot be typed
    # statically and must not raise (legacy behavior preserved).
    check = _arg_validator(memory.gmem_ld_v4f32)
    assert check({}, "x", "i", "r0", "r1", "r2", "r3") is None
    assert check({"x": float32}, "x", 3, 1, 2, 3, 4) is None


# --------------------------------------------------------------------------- #
# Operand units / packed-vs-numerical inventory
# --------------------------------------------------------------------------- #
def test_contract_inventory_distinguishes_units_and_lanes():
    raw = contracts.contract_for("gmem_ld_u32x4")
    assert raw.offset_unit == "byte"
    assert raw.lane_representation == "raw"
    assert raw.lane_dtypes == (uint32, )
    assert raw.transaction_bytes == 16 and raw.alignment_bytes == 16

    num = contracts.contract_for("gmem_vld_f32x4")
    assert num.offset_unit == "element"
    assert num.lane_representation == "numerical"
    assert num.lane_dtypes == (float32, )

    pair = contracts.contract_for("gmem_ld_bf16x2_u32")
    assert pair.offset_unit == "pair" and pair.transaction_bytes == 4

    scalar = contracts.contract_for("gmem_ld_scalar")
    assert "nc" in scalar.cache_policies
    assert scalar.read_only_contract is True
    assert "wt" not in scalar.cache_policies


def test_equivalent_families_are_inventoried():
    equiv = contracts.EQUIVALENT_FAMILIES
    assert "gmem_ld_v4f32" in equiv or "gmem_vld_f32x4" in equiv
    # the raw family is never listed as an alias of the numerical family
    for aliases in equiv.values():
        assert "gmem_ld_f32x4" not in aliases or "gmem_vld_f32x4" not in aliases
    # contract_for never guesses from a name substring
    with pytest.raises(KeyError):
        contracts.contract_for("gmem_ld_this_is_not_an_inventoried_name")


def test_effects_and_cache_policy_sets_are_explicit():
    assert contracts.EFFECT_LOAD == "load"
    c = contracts.contract_for("cp_async_16")
    assert c.effect == "load_store" and c.transaction_bytes == 16
    w = contracts.contract_for("cp_async_wait_group")
    assert w.effect == "none"


def test_cache_policy_normalization_and_rejection():
    n = contracts.normalize_cache_policy
    assert n(None, contracts.LOAD_POLICIES, "wb", "t") == "wb"
    assert n('"cs"', contracts.LOAD_POLICIES, "wb", "t") == "cs"
    assert n("evict_first", contracts.LOAD_POLICIES, "wb", "t") == "cs"
    assert n("read_only", contracts.LOAD_POLICIES, "wb", "t") == "nc"
    with pytest.raises(ValueError):
        n("bogus", contracts.LOAD_POLICIES, "wb", "gmem_ld_scalar")
    # write-through is not a legal load policy
    with pytest.raises(ValueError):
        n("wt", contracts.LOAD_POLICIES, "wb", "gmem_ld_scalar")


# --------------------------------------------------------------------------- #
# Codegen: operand units and numeric-vs-bit behavior (not just names)
# --------------------------------------------------------------------------- #
def _cg(fn, *args):
    return getattr(fn, CODEGEN_FUNC_ATTR)(*args)


def test_byte_offset_vs_element_offset_codegen():
    # BYTE index: explicit uint8 pointer arithmetic on the byte operand.
    raw = _cg(memory.gmem_ld_u32x4, "mem", "byte_index", "a", "b", "c", "d")
    assert "reinterpret_cast<const uint8_t*>(mem)" in raw.return_val
    assert "+ (byte_index)" in raw.return_val

    # ELEMENT index: typed pointer arithmetic mem + index.
    elem = _cg(memory.gmem_ld_f32x4, "mem", "index", "a", "b", "c", "d")
    assert "gmem_ld_f32x4_fn(mem + index" in elem.return_val
    assert "uint8_t" not in elem.return_val


def test_raw_and_numerical_helpers_have_distinct_lane_carriers():
    raw = _cg(memory.gmem_ld_f32x4, "m", "i", "a", "b", "c", "d")
    num = _cg(memory.gmem_ld_v4f32, "m", "i", "a", "b", "c", "d")
    assert "uint32_t* r0" in raw.body and "const uint4 value" in raw.body
    assert "float* r0" in num.body and "const float4 value" in num.body
    # distinct device helper names (can coexist in one translation unit)
    assert "gmem_ld_f32x4_fn" in raw.return_val
    assert "gmem_ld_v4f32_fn" in num.return_val


def test_rne_pack_store_is_numeric_not_raw():
    # gmem_st_bf16x8 RNE-packs eight fp32 with a single rounding; the raw
    # twin gmem_st_bf16x8_u32x4 moves four uint32 words with no conversion.
    pack = _cg(memory.gmem_st_bf16x8, "m", "i", *[f"f{k}" for k in range(8)])
    assert "__floats2bfloat162_rn" in pack.body
    raw = _cg(memory.gmem_st_bf16x8_u32x4, "m", "i", "a", "b", "c", "d")
    assert "__floats2bfloat162_rn" not in raw.body
    assert "uint4{r0, r1, r2, r3}" in raw.body


def test_equivalent_aliases_share_codegen_owner():
    # suffix-spelling aliases lower through the SAME codegen callable
    assert (getattr(memory.gmem_ld_cs_f32x4, CODEGEN_FUNC_ATTR) is getattr(memory.gmem_ld_f32x4_cs, CODEGEN_FUNC_ATTR))
    assert (getattr(memory.gmem_st_cs_f32x2, CODEGEN_FUNC_ATTR) is getattr(memory.gmem_st_f32x2_cs, CODEGEN_FUNC_ATTR))


def test_representative_ptx_opcodes_preserved():
    ld = _cg(memory.ld_global_v4_u32, "mem", "off", "out")
    assert "ld.global.v4.u32" in ld.body
    st = _cg(memory.gmem_st_v4_u32, "mem", "off", "a", "b", "c", "d")
    assert "st.global.v4.b32" in st.body
    cp = _cg(memory.cp_async_ca_16, "smem", "off", "gmem", "goff")
    assert "cp.async.ca.shared.global" in cp.body
    mm = _cg(memory.mma_m16n8k16, "acc", "a", "b", False, "f16")
    assert "mma.sync.aligned.m16n8k16" in mm.body


# --------------------------------------------------------------------------- #
# No import cycle
# --------------------------------------------------------------------------- #
def test_owner_modules_import_without_cycle():
    import importlib
    for mod in ["allocation", "matrix", "async_copy", "conversion", "scalar_global", "shared", "global_vector"]:
        m = importlib.import_module(
            ("little_kernel.backends.common.gpu.language.intrin.allocation"
             if mod == "allocation" else f"little_kernel.backends.nv.common.language.intrin.memory_ops.{mod}"))
        assert hasattr(m, "builtin")
    assert contracts.MEMORY_CONTRACTS
    assert len(contracts.CANONICAL_FAMILIES) >= 20


def test_builtin_metadata_preserved_through_facade():
    fn = memory.gmem_ld_v4f32
    assert getattr(fn, BUILTIN_ATTR) is True
    assert getattr(fn, EVAL_RETURN_TYPE_ATTR)() is not None
    assert callable(getattr(fn, CODEGEN_FUNC_ATTR))
